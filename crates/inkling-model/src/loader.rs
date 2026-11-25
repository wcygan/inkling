//! Streams byte ranges of a large file into device memory.
//!
//! Reader threads read aligned chunks into a ring of pinned staging buffers (optionally bypassing the
//! Windows file cache); the calling thread, which owns the CUDA context, enqueues asynchronous copies of
//! the parts each chunk holds and recycles a staging buffer once its copies have completed.

use inkling_runtime::{DevSlice, Event, PinnedBuffer, Stream};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::os::windows::fs::{FileExt, OpenOptionsExt};
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

/// `bytes` bytes at absolute file offset `offset` go to `dst` (whose length must equal `bytes`).
#[derive(Clone, Copy, Debug)]
pub struct FileCopy {
    pub offset: u64,
    pub dst: DevSlice,
}

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub chunk_bytes: usize,
    pub staging_buffers: usize,
    pub reader_threads: usize,
    /// Open with `FILE_FLAG_NO_BUFFERING`: reads skip the file cache (chunk size and offsets are kept
    /// sector-aligned; pinned buffers are page-aligned).
    pub unbuffered: bool,
}

impl Default for LoadOptions {
    fn default() -> LoadOptions {
        LoadOptions { chunk_bytes: 64 << 20, staging_buffers: 6, reader_threads: 3, unbuffered: true }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadStats {
    pub bytes_read: u64,
    pub bytes_copied: u64,
    pub chunks: usize,
    pub seconds: f64,
}

const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;

/// Copies every `FileCopy` from `path` to the device, overlapping reads with transfers.
pub fn stream_to_device(path: &Path, copies: &[FileCopy], stream: &Stream, opts: LoadOptions) -> Result<LoadStats, String> {
    assert!(opts.chunk_bytes.is_multiple_of(4096) && opts.staging_buffers >= 2 && opts.reader_threads >= 1);
    let start = Instant::now();
    let mut copies: Vec<FileCopy> = copies.iter().copied().filter(|c| c.dst.bytes > 0).collect();
    copies.sort_by_key(|c| c.offset);
    let chunk = opts.chunk_bytes as u64;

    let mut chunks: Vec<u64> = Vec::new();
    for c in &copies {
        let (first, last) = (c.offset / chunk, (c.offset + c.dst.bytes as u64 - 1) / chunk);
        for k in first..=last {
            if chunks.last() != Some(&k) {
                chunks.push(k);
            }
        }
    }
    chunks.dedup();

    let mut open = OpenOptions::new();
    open.read(true).custom_flags(if opts.unbuffered { FILE_FLAG_NO_BUFFERING } else { FILE_FLAG_SEQUENTIAL_SCAN });
    let file = Arc::new(open.open(path).map_err(|e| format!("{}: {e}", path.display()))?);
    let file_len = file.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();

    let (free_tx, free_rx) = mpsc::channel::<PinnedBuffer>();
    let free_rx = Arc::new(Mutex::new(free_rx));
    for _ in 0..opts.staging_buffers {
        free_tx.send(PinnedBuffer::alloc(opts.chunk_bytes).map_err(|e| e.to_string())?).unwrap();
    }
    let queue = Arc::new(Mutex::new(chunks.iter().copied().collect::<VecDeque<u64>>()));
    let (done_tx, done_rx) = mpsc::channel::<Result<(u64, usize, PinnedBuffer), String>>();

    let readers: Vec<_> = (0..opts.reader_threads)
        .map(|_| {
            let (file, queue, free_rx, done_tx) = (file.clone(), queue.clone(), free_rx.clone(), done_tx.clone());
            std::thread::spawn(move || {
                loop {
                    let Some(k) = queue.lock().unwrap().pop_front() else { return };
                    let Ok(mut buf) = free_rx.lock().unwrap().recv() else { return };
                    // Bytes of this chunk that exist in the file. Unbuffered reads must stay sector
                    // aligned, so the request is rounded up to 4 KiB and the read simply stops at EOF.
                    let want = (file_len.saturating_sub(k * chunk)).min(chunk) as usize;
                    let request = want.next_multiple_of(4096).min(buf.len());
                    let mut got = 0usize;
                    let res = loop {
                        if got >= want {
                            break Ok(());
                        }
                        let out = &mut buf.as_mut_slice()[got..request];
                        match file.seek_read(out, k * chunk + got as u64) {
                            Ok(0) => break Err(format!("chunk {k}: unexpected end of file at {got} of {want} bytes")),
                            Ok(n) => got += n,
                            Err(e) => break Err(format!("read of chunk {k} at +{got}: {e}")),
                        }
                    };
                    let got = got.min(want);
                    if done_tx.send(res.map(|()| (k, got, buf))).is_err() {
                        return;
                    }
                }
            })
        })
        .collect();
    drop(done_tx);

    let mut stats = LoadStats { chunks: chunks.len(), ..Default::default() };
    let mut in_flight: VecDeque<(Event, PinnedBuffer)> = VecDeque::new();
    let result = (|| -> Result<(), String> {
        for _ in 0..chunks.len() {
            let (k, got, buf) = done_rx.recv().map_err(|_| "reader threads stopped early".to_string())??;
            stats.bytes_read += got as u64;
            let (lo, hi) = (k * chunk, k * chunk + got as u64);
            // Copies overlapping [lo, hi): copies are sorted by offset and do not overlap each other.
            let first = copies.partition_point(|c| c.offset + c.dst.bytes as u64 <= lo);
            for c in copies[first..].iter().take_while(|c| c.offset < hi) {
                let s = c.offset.max(lo);
                let e = (c.offset + c.dst.bytes as u64).min(hi);
                let dst = c.dst.sub((s - c.offset) as usize, (e - s) as usize);
                buf.copy_range_to_device_async((s - lo) as usize, &dst, stream).map_err(|e| e.to_string())?;
                stats.bytes_copied += e - s;
            }
            let ev = Event::new_untimed().map_err(|e| e.to_string())?;
            ev.record(stream).map_err(|e| e.to_string())?;
            in_flight.push_back((ev, buf));
            while in_flight.len() > opts.staging_buffers / 2 {
                let (ev, buf) = in_flight.pop_front().unwrap();
                ev.synchronize().map_err(|e| e.to_string())?;
                let _ = free_tx.send(buf);
            }
        }
        for (ev, _) in &in_flight {
            ev.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    queue.lock().unwrap().clear();
    drop(free_tx);
    for r in readers {
        let _ = r.join();
    }
    result?;
    let expected: u64 = copies.iter().map(|c| c.dst.bytes as u64).sum();
    if stats.bytes_copied != expected {
        return Err(format!("copied {} of {expected} bytes (file shorter than expected?)", stats.bytes_copied));
    }
    stats.seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}
