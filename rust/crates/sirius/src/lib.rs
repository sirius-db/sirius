#![warn(missing_docs)]
#![doc(test(attr(forbid(unsafe_code))))]
//! Safe Rust bindings to the public Sirius C ABI.
//!
//! Build immutable configurations with [`ContextConfigBuilder`].
//! These bindings require no C++ compilation.
//! Existing execution and object-store bindings live in the `sirius-ffi` crate.

mod config;
pub use config::{ConfigError, ContextConfig, ContextConfigBuilder};
