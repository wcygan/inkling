use crate::error::{Result, ck};
use crate::sys;
use std::ffi::{CStr, c_char};

/// Device 0 with its primary context current on the calling thread.
pub struct Gpu {
    pub device: sys::CUdevice,
    pub context: sys::CUcontext,
    pub name: String,
    pub driver_version: i32,
    pub sm_count: i32,
    pub compute_capability: (i32, i32),
}

impl Gpu {
    pub fn open() -> Result<Gpu> {
        unsafe {
            ck(sys::cuInit(0), "cuInit")?;
            let mut device = 0;
            ck(sys::cuDeviceGet(&mut device, 0), "cuDeviceGet")?;
            let mut buf = [0 as c_char; 128];
            ck(sys::cuDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, device), "cuDeviceGetName")?;
            let attr = |a| -> Result<i32> {
                let mut v = 0;
                ck(sys::cuDeviceGetAttribute(&mut v, a, device), "cuDeviceGetAttribute")?;
                Ok(v)
            };
            let sm_count = attr(sys::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
            let cc = (
                attr(sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attr(sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            );
            let mut context = std::ptr::null_mut();
            ck(sys::cuDevicePrimaryCtxRetain(&mut context, device), "cuDevicePrimaryCtxRetain")?;
            ck(sys::cuCtxSetCurrent(context), "cuCtxSetCurrent")?;
            let mut driver_version = 0;
            ck(sys::cuDriverGetVersion(&mut driver_version), "cuDriverGetVersion")?;
            Ok(Gpu {
                device,
                context,
                name: CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned(),
                driver_version,
                sm_count,
                compute_capability: cc,
            })
        }
    }

    /// Makes this context current on the calling thread (needed on every thread that touches the GPU).
    pub fn bind(&self) -> Result<()> {
        ck(unsafe { sys::cuCtxSetCurrent(self.context) }, "cuCtxSetCurrent")
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0, 0);
        ck(unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) }, "cuMemGetInfo")?;
        Ok((free, total))
    }

    pub fn synchronize(&self) -> Result<()> {
        ck(unsafe { sys::cuCtxSynchronize() }, "cuCtxSynchronize")
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.device) };
    }
}
