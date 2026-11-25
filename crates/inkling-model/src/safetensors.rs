//! Validated safetensors headers and positional reads.
//!
//! Stricter than the format requires: every tensor's byte length must equal its dtype size times its
//! shape, and the tensors must tile the data region exactly (no gaps, no overlaps).

use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    BF16,
    F16,
    F8E4M3,
    U8,
    I32,
    I64,
}

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "F32" => Dtype::F32,
            "BF16" => Dtype::BF16,
            "F16" => Dtype::F16,
            "F8_E4M3" => Dtype::F8E4M3,
            "U8" => Dtype::U8,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F8E4M3 | Dtype::U8 => 1,
            Dtype::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file.
    pub offset: u64,
    pub bytes: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafetensorsFile {
    pub path: PathBuf,
    file: File,
    pub file_len: u64,
    pub data_offset: u64,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

const MAX_HEADER_BYTES: u64 = 100_000_000;

fn parse_tensors(header: &Value, data_offset: u64, payload: u64) -> Result<Vec<TensorInfo>, String> {
    let object = header.as_object().ok_or("header is not a JSON object")?;
    let mut out = Vec::with_capacity(object.len());
    for (name, t) in object {
        if name == "__metadata__" {
            continue;
        }
        let dtype_s = t["dtype"].as_str().ok_or_else(|| format!("{name}: missing dtype"))?;
        let dtype = Dtype::parse(dtype_s).ok_or_else(|| format!("{name}: unsupported dtype {dtype_s}"))?;
        let shape = t["shape"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| format!("{name}: invalid shape")))
            .collect::<Result<Vec<_>, _>>()?;
        let offs = t["data_offsets"].as_array().ok_or_else(|| format!("{name}: missing data_offsets"))?;
        let [start, end] = offs.as_slice() else {
            return Err(format!("{name}: expected two data offsets"));
        };
        let (start, end) = (
            start.as_u64().ok_or_else(|| format!("{name}: invalid start"))?,
            end.as_u64().ok_or_else(|| format!("{name}: invalid end"))?,
        );
        if end < start || end > payload {
            return Err(format!("{name}: [{start}, {end}) lies outside the {payload}-byte data region"));
        }
        let numel = shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
            .ok_or_else(|| format!("{name}: element count overflows"))?;
        let want = numel.checked_mul(dtype.size() as u64).ok_or_else(|| format!("{name}: size overflows"))?;
        if want != end - start {
            return Err(format!("{name}: {} bytes stored, {dtype_s} {shape:?} needs {want}", end - start));
        }
        out.push(TensorInfo { name: name.clone(), dtype, shape, offset: data_offset + start, bytes: end - start });
    }
    out.sort_by_key(|t| (t.offset, t.bytes));
    let mut next = data_offset;
    for t in &out {
        if t.offset != next {
            return Err(format!("{}: overlapping tensors or an unindexed gap at byte {next}", t.name));
        }
        next = t.offset + t.bytes;
    }
    if next != data_offset + payload {
        return Err("tensors do not cover the data region".into());
    }
    Ok(out)
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<SafetensorsFile, String> {
        let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
        let file = File::open(path).map_err(|e| err(&e))?;
        let file_len = file.metadata().map_err(|e| err(&e))?.len();
        let mut prefix = [0u8; 8];
        read_exact_at(&file, &mut prefix, 0).map_err(|e| err(&e))?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
            return Err(err(&format!("invalid header length {header_len}")));
        }
        let mut header = vec![0u8; header_len as usize];
        read_exact_at(&file, &mut header, 8).map_err(|e| err(&e))?;
        if header.first() != Some(&b'{') {
            return Err(err(&"header does not start with a JSON object"));
        }
        let header: Value = serde_json::from_slice(&header).map_err(|e| err(&e))?;
        let data_offset = 8 + header_len;
        let tensors = parse_tensors(&header, data_offset, file_len - data_offset).map_err(|e| err(&e))?;
        let index = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(SafetensorsFile { path: path.to_owned(), file, file_len, data_offset, tensors, index })
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// The tensor `name`, checked against an expected dtype and shape.
    pub fn expect(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&TensorInfo, String> {
        let t = self.get(name).ok_or_else(|| format!("{}: missing tensor {name}", self.path.display()))?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!("{name}: {:?} {:?}, expected {dtype:?} {shape:?}", t.dtype, t.shape));
        }
        Ok(t)
    }

    /// Reads `out.len()` bytes at absolute `offset` (positional; safe to call from several threads).
    pub fn read_at(&self, offset: u64, out: &mut [u8]) -> std::io::Result<()> {
        read_exact_at(&self.file, out, offset)
    }

    /// Reads a whole (small) tensor into host memory.
    pub fn read_tensor(&self, t: &TensorInfo) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; t.bytes as usize];
        self.read_at(t.offset, &mut v).map_err(|e| format!("{}: {e}", t.name))?;
        Ok(v)
    }

    pub fn read_f32(&self, t: &TensorInfo) -> Result<Vec<f32>, String> {
        if t.dtype != Dtype::F32 {
            return Err(format!("{}: {:?}, expected F32", t.name, t.dtype));
        }
        Ok(self.read_tensor(t)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
}

fn read_exact_at(file: &File, mut out: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    while !out.is_empty() {
        let n = file.seek_read(out, offset)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "unexpected end of file"));
        }
        out = &mut out[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn orders_and_validates_regions() {
        let h = json!({
            "scale": {"dtype":"U8", "shape":[1], "data_offsets":[2,3]},
            "zero": {"dtype":"U8", "shape":[0], "data_offsets":[2,2]},
            "weight": {"dtype":"U8", "shape":[2], "data_offsets":[0,2]},
            "__metadata__": {"format":"pt"}
        });
        let t = parse_tensors(&h, 100, 3).unwrap();
        assert_eq!(t.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["weight", "zero", "scale"]);
        assert_eq!(t[0].offset, 100);
        assert!(parse_tensors(&h, 100, 2).is_err());
        assert!(parse_tensors(&h, 100, 4).is_err());
    }

    #[test]
    fn rejects_overlaps_and_size_mismatches() {
        let overlap = json!({
            "a": {"dtype":"U8", "shape":[2], "data_offsets":[0,2]},
            "b": {"dtype":"U8", "shape":[2], "data_offsets":[1,3]}
        });
        assert!(parse_tensors(&overlap, 0, 3).is_err());
        let wrong = json!({"a": {"dtype":"BF16", "shape":[3], "data_offsets":[0,4]}});
        assert!(parse_tensors(&wrong, 0, 4).unwrap_err().contains("needs 6"));
        let unknown = json!({"a": {"dtype":"F8_E5M2", "shape":[1], "data_offsets":[0,1]}});
        assert!(parse_tensors(&unknown, 0, 1).unwrap_err().contains("unsupported dtype"));
    }
}
