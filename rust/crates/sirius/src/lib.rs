//! Safe Rust bindings to the public Sirius C ABI.
//!
//! Build immutable configurations with [`ContextConfigBuilder`].
//! These bindings require no C++ compilation.
//! Existing execution and object-store bindings live in the `sirius-ffi` crate.

mod config;
mod diagnostic;
pub use config::{ConfigError, ContextConfig, ContextConfigBuilder};
