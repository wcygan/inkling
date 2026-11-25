//! CUDA driver runtime for inkling-rs on native Windows.
//!
//! Talks only to `nvcuda.dll` (linked with `raw-dylib`, so no CUDA Toolkit is needed to run). Covers what
//! the engine needs: device and context, device/pinned memory, a planned arena, cubin modules, kernel
//! launches through `cuLaunchKernelEx` with optional programmatic dependent launch (PDL), streams, events,
//! and stream capture into graphs.

pub mod arena;
pub mod device;
pub mod error;
pub mod graph;
pub mod launch;
pub mod memory;
pub mod module;
pub mod stream;
pub mod sys;

pub use arena::{Arena, ArenaPlan, Slot};
pub use device::Gpu;
pub use error::{CudaError, Result};
pub use graph::{GraphExec, capture, capture_switch};
pub use launch::{Args, KernelArg, LaunchConfig, PtrPack};
pub use memory::{DevSlice, DeviceBuffer, PinnedBuffer};
pub use module::{Kernel, KernelInfo, Module};
pub use stream::{Event, Stream};
