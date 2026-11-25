//! `json.dumps` as transformers' chat-template `tojson` filter calls it: `ensure_ascii`, `indent`,
//! `separators` and `sort_keys` keyword arguments, Python's escaping and Python's float spelling.

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Error, ErrorKind, Value};
use std::fmt::Write;

pub struct Options {
    pub ensure_ascii: bool,
    pub indent: Option<usize>,
    pub item_sep: String,
    pub key_sep: String,
    pub sort_keys: bool,
}

impl Options {
    pub fn from_kwargs(kwargs: &Kwargs) -> Result<Options, Error> {
        let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
        let indent: Option<usize> = kwargs.get("indent")?;
        let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
        let separators: Option<Value> = kwargs.get("separators")?;
        kwargs.assert_all_used()?;
        // Python: separators default to (', ', ': '), or (',', ': ') when indent is set.
        let (mut item_sep, mut key_sep) =
            (if indent.is_some() { "," } else { ", " }.to_owned(), ": ".to_owned());
        if let Some(seps) = separators.filter(|v| !v.is_none()) {
            let parts: Vec<Value> = seps.try_iter()?.collect();
            let [a, b] = parts.as_slice() else {
                return Err(Error::new(ErrorKind::InvalidOperation, "separators must be a pair"));
            };
            item_sep = a.as_str().ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "separator"))?.to_owned();
            key_sep = b.as_str().ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "separator"))?.to_owned();
        }
        Ok(Options {
            ensure_ascii: ensure_ascii.unwrap_or(false),
            indent,
            item_sep,
            key_sep,
            sort_keys: sort_keys.unwrap_or(false),
        })
    }
}

fn write_str(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c if ensure_ascii && !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    write!(out, "\\u{:04x}", unit).unwrap();
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: shortest round-trip digits, always with a decimal point or exponent.
fn write_float(out: &mut String, f: f64) -> Result<(), Error> {
    if f.is_nan() {
        out.push_str("NaN");
    } else if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        let a = f.abs();
        if a != 0.0 && !(1e-4..1e16).contains(&a) {
            // Python switches to exponent form outside [1e-4, 1e16): e.g. 1e+16, 1.5e-05.
            let s = format!("{f:e}");
            let (mant, exp) = s.split_once('e').unwrap();
            let exp: i32 = exp.parse().unwrap();
            write!(out, "{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs()).unwrap();
        } else {
            let s = format!("{f}");
            out.push_str(&s);
            if !s.contains('.') {
                out.push_str(".0");
            }
        }
    }
    Ok(())
}

fn newline(out: &mut String, opts: &Options, depth: usize) {
    if let Some(n) = opts.indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', n * depth));
    }
}

pub fn dumps(v: &Value, opts: &Options, depth: usize, out: &mut String) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        // Integers print as integers; floats keep Python's spelling (21.0, not 21).
        ValueKind::Number if v.is_integer() => write!(out, "{v}").unwrap(),
        ValueKind::Number => write_float(out, f64::try_from(v.clone())?)?,
        ValueKind::String => write_str(out, v.as_str().unwrap_or_default(), opts.ensure_ascii),
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                newline(out, opts, depth + 1);
                dumps(item, opts, depth + 1, out)?;
            }
            if !items.is_empty() {
                newline(out, opts, depth);
            }
            out.push(']');
        }
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if opts.sort_keys {
                keys.sort_by_key(|k| k.to_string());
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                newline(out, opts, depth + 1);
                write_str(out, &k.to_string(), opts.ensure_ascii);
                out.push_str(&opts.key_sep);
                dumps(&v.get_item(k)?, opts, depth + 1, out)?;
            }
            if !keys.is_empty() {
                newline(out, opts, depth);
            }
            out.push('}');
        }
        other => {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("tojson: cannot serialise {other:?}")));
        }
    }
    Ok(())
}

pub fn tojson(v: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let opts = Options::from_kwargs(&kwargs)?;
    let mut out = String::new();
    dumps(v, &opts, 0, &mut out)?;
    Ok(Value::from(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dumps_default(v: Value) -> String {
        let opts = Options { ensure_ascii: false, indent: None, item_sep: ", ".into(), key_sep: ": ".into(), sort_keys: false };
        let mut s = String::new();
        dumps(&v, &opts, 0, &mut s).unwrap();
        s
    }

    #[test]
    fn matches_python_json_dumps_spelling() {
        // Python: json.dumps([21, 21.0, 0.1, 1e16, 1.5e-05, "é\n\"", None, True])
        let v = Value::from_serialize(serde_json::json!([21, 21.0, 0.1, 1e16, 1.5e-05, "é\n\"", null, true]));
        assert_eq!(dumps_default(v), r#"[21, 21.0, 0.1, 1e+16, 1.5e-05, "é\n\"", null, true]"#);
    }
}
