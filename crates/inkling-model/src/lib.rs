//! The Inkling-2-Mini model contract for inkling-rs: dimensions checked against `config.json`, the
//! 48-visit schedule, validated safetensors access and the exact-mode weight loader.

pub mod config;
pub mod loader;
pub mod safetensors;
pub mod schedule;
pub mod weights;

pub use config::{Config, dims, is_global_layer};
pub use schedule::{SCHEDULE, Visit};
