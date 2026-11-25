use crate::error::{Result, check, ck};
use crate::stream::Stream;
use crate::sys;
use std::ffi::c_void;

/// A raw device address with a byte length. Kernels receive the address as a `u64` parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevSlice {
    pub ptr: sys::CUdeviceptr,
    pub bytes: usize,
}

impl DevSlice {
    /// Sub-range `[offset, offset + bytes)`; panics when it leaves the slice.
    pub fn sub(&self, offset: usize, bytes: usize) -> DevSlice {
        assert!(offset.checked_add(bytes).is_some_and(|end| end <= self.bytes), "sub-slice out of range");
        DevSlice { ptr: self.ptr + offset as u64, bytes }
    }

    /// Copies `data` to the device and returns once the device holds it.
    ///
    /// `cuMemcpyHtoD` from pageable memory may return after staging, before the DMA lands, and it does
    /// not order against non-blocking streams. The context sync makes the copy complete before any work
    /// enqueued afterwards on any stream can read the destination.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        let n = size_of_val(data);
        assert!(n <= self.bytes, "upload of {n} bytes into {} bytes", self.bytes);
        ck(unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), n) }, "cuMemcpyHtoD")?;
        ck(unsafe { sys::cuCtxSynchronize() }, "cuCtxSynchronize after cuMemcpyHtoD")
    }

    pub fn download<T: Copy + Default>(&self, n: usize) -> Result<Vec<T>> {
        let bytes = n * size_of::<T>();
        assert!(bytes <= self.bytes, "download of {bytes} bytes from {} bytes", self.bytes);
        let mut v = vec![T::default(); n];
        ck(unsafe { sys::cuMemcpyDtoH_v2(v.as_mut_ptr().cast(), self.ptr, bytes) }, "cuMemcpyDtoH")?;
        Ok(v)
    }

    pub fn download_bytes_at(&self, offset: usize, out: &mut [u8]) -> Result<()> {
        let s = self.sub(offset, out.len());
        ck(unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), s.ptr, out.len()) }, "cuMemcpyDtoH")
    }

    /// Enqueues a byte fill on `stream`.
    pub fn fill_u8(&self, v: u8, stream: &Stream) -> Result<()> {
        ck(unsafe { sys::cuMemsetD8Async(self.ptr, v, self.bytes, stream.raw()) }, "cuMemsetD8Async")
    }

    /// Enqueues a device-to-device copy of `self` into `dst` on `stream`.
    pub fn copy_to(&self, dst: &DevSlice, stream: &Stream) -> Result<()> {
        assert!(self.bytes <= dst.bytes, "copy of {} bytes into {} bytes", self.bytes, dst.bytes);
        ck(unsafe { sys::cuMemcpyDtoDAsync_v2(dst.ptr, self.ptr, self.bytes, stream.raw()) }, "cuMemcpyDtoDAsync")
    }
}

/// An owned device allocation.
pub struct DeviceBuffer {
    slice: DevSlice,
}

impl DeviceBuffer {
    pub fn alloc(bytes: usize) -> Result<DeviceBuffer> {
        let mut ptr = 0;
        check(unsafe { sys::cuMemAlloc_v2(&mut ptr, bytes.max(1)) }, || format!("cuMemAlloc({bytes} bytes)"))?;
        Ok(DeviceBuffer { slice: DevSlice { ptr, bytes } })
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<DeviceBuffer> {
        let b = DeviceBuffer::alloc(size_of_val(data))?;
        b.slice.upload(data)?;
        Ok(b)
    }

    pub fn slice(&self) -> DevSlice {
        self.slice
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { sys::cuMemFree_v2(self.slice.ptr) };
    }
}

/// Page-locked host memory: the source or target of asynchronous copies.
pub struct PinnedBuffer {
    ptr: *mut u8,
    bytes: usize,
}

// The buffer is plain memory; synchronisation with the GPU is the caller's job (events).
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    pub fn alloc(bytes: usize) -> Result<PinnedBuffer> {
        let mut p: *mut c_void = std::ptr::null_mut();
        check(unsafe { sys::cuMemHostAlloc(&mut p, bytes.max(1), sys::CU_MEMHOSTALLOC_PORTABLE) }, || {
            format!("cuMemHostAlloc({bytes} bytes)")
        })?;
        Ok(PinnedBuffer { ptr: p.cast(), bytes })
    }

    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    /// Enqueues `self[..bytes]` → `dst` on `stream`. The buffer must stay untouched until the copy completes.
    pub fn copy_to_device_async(&self, bytes: usize, dst: &DevSlice, stream: &Stream) -> Result<()> {
        assert!(bytes <= self.bytes && bytes <= dst.bytes);
        ck(
            unsafe { sys::cuMemcpyHtoDAsync_v2(dst.ptr, self.ptr.cast(), bytes, stream.raw()) },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Enqueues `self[offset..offset + dst.bytes]` → `dst` on `stream`.
    pub fn copy_range_to_device_async(&self, offset: usize, dst: &DevSlice, stream: &Stream) -> Result<()> {
        assert!(offset.checked_add(dst.bytes).is_some_and(|end| end <= self.bytes), "pinned range out of bounds");
        ck(
            unsafe { sys::cuMemcpyHtoDAsync_v2(dst.ptr, self.ptr.add(offset).cast(), dst.bytes, stream.raw()) },
            "cuMemcpyHtoDAsync",
        )
    }

    /// Enqueues `src` → `self[..src.bytes]` on `stream`.
    pub fn copy_from_device_async(&mut self, src: &DevSlice, stream: &Stream) -> Result<()> {
        assert!(src.bytes <= self.bytes);
        ck(
            unsafe { sys::cuMemcpyDtoHAsync_v2(self.ptr.cast(), src.ptr, src.bytes, stream.raw()) },
            "cuMemcpyDtoHAsync",
        )
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { sys::cuMemFreeHost(self.ptr.cast()) };
    }
}
