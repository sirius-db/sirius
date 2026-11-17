//! Raw declarations matching the public headers in include/sirius-c.
//!
//! Handles are allocated and released by Sirius. Never free them with Rust's allocator.
//! Output slots must not contain unreleased handles. Diagnostics are optional;
//! always inspect the status even when the diagnostic pointer is null.
//!
//! # Safety
//! Follow each function's C contract: keep borrowed handles and buffers alive throughout
//! the call, provide exclusive writable output slots, and release each owning reference
//! exactly once. Retaining a handle does not permit releasing it concurrently with use.

use std::marker::{PhantomData, PhantomPinned};

// Bindgen's incomplete structs lack these markers. Keep only opaque handle
// definitions handwritten; all function signatures and constants are generated.
/// Opaque immutable configuration handle.
#[repr(C)]
pub struct SiriusContextConfig {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
/// Opaque immutable builder handle.
#[repr(C)]
pub struct SiriusContextConfigBuilder {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
/// Opaque owned diagnostic.
#[repr(C)]
pub struct SiriusError {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

pub use SiriusContextConfig as sirius_context_config;
pub use SiriusContextConfigBuilder as sirius_context_config_builder;
pub use SiriusError as sirius_error;

#[allow(non_camel_case_types)]
mod generated {
    include!("generated.rs");
}
pub use generated::SIRIUS_BINDGEN_ABI_VERSION as SIRIUS_ABI_VERSION;
pub use generated::{
    SIRIUS_ALLOCATION_FAILURE, SIRIUS_CONFIGURATION_IO, SIRIUS_CONTEXT_INITIALIZATION,
    SIRIUS_INTERNAL_ERROR, SIRIUS_INVALID_ARGUMENT, SIRIUS_INVALID_CONFIGURATION,
    SIRIUS_MALFORMED_YAML, SIRIUS_SUCCESS, sirius_context_config_builder_build,
    sirius_context_config_builder_create, sirius_context_config_builder_from_yaml,
    sirius_context_config_builder_release, sirius_context_config_builder_retain,
    sirius_context_config_release, sirius_context_config_retain, sirius_error_destroy,
    sirius_error_message, sirius_error_message_size, sirius_status,
};

/// Return the linked library's C ABI revision.
///
/// Zero denotes an unversioned development API, not a compatibility guarantee.
pub extern "C" fn sirius_abi_version() -> u32 {
    // SAFETY: the C function takes no arguments and has no caller preconditions.
    unsafe { generated::sirius_abi_version() }
}
