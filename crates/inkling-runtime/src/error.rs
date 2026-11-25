use crate::sys;
use std::ffi::{CStr, c_char};
use std::fmt;

/// A failed driver call: the operation, the driver's error name and its description.
#[derive(Debug, Clone)]
pub struct CudaError {
    pub code: i32,
    pub op: String,
    pub name: String,
    pub message: String,
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} ({}): {}", self.op, self.name, self.code, self.message)
    }
}

impl std::error::Error for CudaError {}

/// Lets callers that report errors as `String` use `?` on driver results.
impl From<CudaError> for String {
    fn from(e: CudaError) -> String {
        e.to_string()
    }
}

pub type Result<T> = std::result::Result<T, CudaError>;

fn driver_string(code: i32, getter: unsafe extern "system" fn(i32, *mut *const c_char) -> i32) -> String {
    let mut s: *const c_char = std::ptr::null();
    let ok = unsafe { getter(code, &mut s) } == sys::CUDA_SUCCESS;
    if ok && !s.is_null() {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    } else {
        "unknown".to_owned()
    }
}

/// Turns a driver status into `Result`, naming the operation that failed.
pub fn check(code: sys::CUresult, op: impl FnOnce() -> String) -> Result<()> {
    if code == sys::CUDA_SUCCESS {
        return Ok(());
    }
    Err(CudaError {
        code,
        op: op(),
        name: driver_string(code, sys::cuGetErrorName),
        message: driver_string(code, sys::cuGetErrorString),
    })
}

/// `check` for a fixed operation name.
pub fn ck(code: sys::CUresult, op: &'static str) -> Result<()> {
    check(code, || op.to_owned())
}
