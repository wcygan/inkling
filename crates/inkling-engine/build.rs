// The wide prefill path calls cuBLAS (BF16 GEMMs with FP32 accumulation) from the CUDA Toolkit.
fn main() {
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5".into());
    println!("cargo:rustc-link-search=native={cuda}/lib/x64");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
