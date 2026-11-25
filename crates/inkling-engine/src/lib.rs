//! inkling-rs decode engine.
//!
//! `state` holds the per-sequence device buffers; `forward` records the 835 launches of one exact-mode
//! token step with squid5090's parameter order.

pub mod cublas;
pub mod dspark;
pub mod forward;
pub mod fp8;
pub mod mse;
pub mod nvfp4;
pub mod prefill;
pub mod state;

pub use dspark::{DsBuf, Dspark, DsparkWeights, NUM_DRAFTS};
pub use forward::{Forward, Kernels, Mode, Nvfp4Parts};
pub use fp8::Fp8Dense;
pub use nvfp4::Nvfp4Weights;
pub use prefill::WidePrefill;
pub use state::{B_MAX, State};
