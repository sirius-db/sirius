//! Generated Rust bindings for the StarRocks Thrift IDL.
//!
//! `build.rs` runs the Thrift compiler over the `starrocks/gensrc/thrift`
//! submodule and emits one module per `.thrift` file; this crate re-exports them
//! via the generated module index so other crates can share the same types.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![allow(clippy::all, clippy::deprecated_cfg_attr)]

include!(concat!(env!("OUT_DIR"), "/thrift_mods.rs"));
