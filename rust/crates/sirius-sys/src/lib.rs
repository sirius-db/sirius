//! Raw bindings to the public Sirius C ABI.
//!
//! This crate links `libsirius` without compiling C++ code. Safe wrappers live in `sirius`.
//! Existing C++ FFI bindings live in the `sirius-ffi` crate.

/// Raw configuration bindings to the public C ABI.
pub mod c_api;
