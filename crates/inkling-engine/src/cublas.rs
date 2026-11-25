//! The cuBLAS call of the wide prefill path: y[rows][out] (f32) = x[rows][inp] (BF16) · w[out][inp]ᵀ (BF16),
//! FP32 accumulation, on a handle bound to one stream with a fixed workspace (so it can be captured).

use std::ffi::c_void;

const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F: i32 = 0;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_GEMM_DEFAULT: i32 = -1;

#[link(name = "cublas")]
unsafe extern "system" {
    fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    fn cublasSetStream_v2(handle: *mut c_void, stream: *mut c_void) -> i32;
    fn cublasSetWorkspace_v2(handle: *mut c_void, workspace: *mut c_void, bytes: usize) -> i32;
    fn cublasGemmEx(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        compute_type: i32,
        algo: i32,
    ) -> i32;
}

pub struct Blas {
    handle: *mut c_void,
}

fn check(status: i32, what: &str) -> Result<(), String> {
    if status == 0 { Ok(()) } else { Err(format!("{what} failed with cuBLAS status {status}")) }
}

impl Blas {
    pub fn new(stream: *mut c_void, workspace: u64, workspace_bytes: usize) -> Result<Blas, String> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            check(cublasCreate_v2(&mut handle), "cublasCreate")?;
            check(cublasSetStream_v2(handle, stream), "cublasSetStream")?;
            check(cublasSetWorkspace_v2(handle, workspace as *mut c_void, workspace_bytes), "cublasSetWorkspace")?;
        }
        Ok(Blas { handle })
    }

    /// y[rows][out] (f32, row stride `ldy`) = x[rows][inp] (BF16) · w[out][inp]ᵀ (BF16).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(&self, x: u64, w: u64, y: u64, rows: usize, inp: usize, out: usize, ldy: usize) -> Result<(), String> {
        let (alpha, beta) = (1.0f32, 0.0f32);
        unsafe {
            check(
                cublasGemmEx(
                    self.handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    out as i32,
                    rows as i32,
                    inp as i32,
                    &alpha as *const f32 as *const c_void,
                    w as *const c_void,
                    CUDA_R_16BF,
                    inp as i32,
                    x as *const c_void,
                    CUDA_R_16BF,
                    inp as i32,
                    &beta as *const f32 as *const c_void,
                    y as *mut c_void,
                    CUDA_R_32F,
                    ldy as i32,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx",
            )
        }
    }
}
