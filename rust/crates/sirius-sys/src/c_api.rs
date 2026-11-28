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

/// Opaque immutable schema handle.
#[repr(C)]
pub struct SiriusSchema {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
pub use SiriusSchema as sirius_schema;

/// Opaque immutable expression handle.
#[repr(C)]
pub struct SiriusExpression {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
pub use SiriusExpression as sirius_expression;

/// Opaque immutable plan node handle.
#[repr(C)]
pub struct SiriusPlanNode {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
pub use SiriusPlanNode as sirius_plan_node;

/// Opaque immutable plan handle.
#[repr(C)]
pub struct SiriusPlan {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}
pub use SiriusPlan as sirius_plan;

#[allow(non_camel_case_types)]
mod generated {
    include!("generated.rs");
}
pub use generated::SIRIUS_BINDGEN_ABI_VERSION as SIRIUS_ABI_VERSION;
pub use generated::{
    SIRIUS_ALLOCATION_FAILURE, SIRIUS_COMPARISON_EQUAL, SIRIUS_COMPARISON_GREATER,
    SIRIUS_COMPARISON_GREATER_EQUAL, SIRIUS_COMPARISON_LESS, SIRIUS_COMPARISON_LESS_EQUAL,
    SIRIUS_COMPARISON_NOT_EQUAL, SIRIUS_CONFIGURATION_IO, SIRIUS_CONTEXT_INITIALIZATION,
    SIRIUS_INTERNAL_ERROR, SIRIUS_INVALID_ARGUMENT, SIRIUS_INVALID_CONFIGURATION,
    SIRIUS_MALFORMED_YAML, SIRIUS_NODE_FILTER, SIRIUS_NODE_PARQUET_SCAN, SIRIUS_SUCCESS,
    SIRIUS_TYPE_BOOLEAN, SIRIUS_TYPE_INT64, sirius_comparison, sirius_context_config_builder_build,
    sirius_context_config_builder_create, sirius_context_config_builder_from_yaml,
    sirius_context_config_builder_release, sirius_context_config_builder_retain,
    sirius_context_config_release, sirius_context_config_retain, sirius_data_type,
    sirius_error_destroy, sirius_error_message, sirius_error_message_size,
    sirius_expression_column, sirius_expression_compare, sirius_expression_literal_boolean,
    sirius_expression_literal_int64, sirius_expression_nullable, sirius_expression_release,
    sirius_expression_retain, sirius_expression_type, sirius_field, sirius_plan_describe,
    sirius_plan_filter, sirius_plan_from_root, sirius_plan_node_get_kind, sirius_plan_node_input,
    sirius_plan_node_kind, sirius_plan_node_release, sirius_plan_node_retain,
    sirius_plan_node_schema, sirius_plan_release, sirius_plan_retain, sirius_plan_root,
    sirius_plan_scan_parquet, sirius_schema_create, sirius_schema_field, sirius_schema_field_count,
    sirius_schema_release, sirius_schema_retain, sirius_status,
};

/// Return the linked library's C ABI revision.
///
/// Zero denotes an unversioned development API, not a compatibility guarantee.
pub extern "C" fn sirius_abi_version() -> u32 {
    // SAFETY: the C function takes no arguments and has no caller preconditions.
    unsafe { generated::sirius_abi_version() }
}
