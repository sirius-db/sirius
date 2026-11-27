//! Rust access to the public Sirius C ABI.
//!
//! [`c_api`] exposes the raw declarations. [`handles`] owns native references and
//! provides safe operations over them. Keeping unsafe calls and ownership here lets
//! the end-user `sirius` crate forbid unsafe code.
//! Existing C++ FFI bindings live in the `sirius-ffi` crate.

pub mod handles;

/// Raw configuration bindings to the public C ABI.
pub mod c_api;

#[doc(hidden)]
#[cfg(all(miri, feature = "miri-test-backend"))]
#[path = "../tests/support/backend.rs"]
pub mod miri_test_backend;
