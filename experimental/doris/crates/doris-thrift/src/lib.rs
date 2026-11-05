//! Generated Rust bindings for the Apache Doris Thrift IDL.
//!
//! `build.rs` runs the Thrift compiler over the `doris/gensrc/thrift` submodule
//! (pinned to the same Doris tag as the FE binary) and emits one module per
//! `.thrift` file; this crate re-exports them via the generated module index so
//! other crates can share the same types.
//!
//! Modules the backend uses most:
//! - `heartbeat_service`: `HeartbeatService` (`TMasterInfo` → `THeartbeatResult`)
//! - `backend_service`: `BackendService` (periodic FE probes, all stubbed)
//! - `palo_internal_service`: `TPipelineFragmentParamsList` carried by `exec_plan_fragment`
//! - `plan_nodes` / `exprs` / `descriptors` / `data_sinks`: the plan fragment shape
//! - `types` / `status`: `TUniqueId`, `TNetworkAddress`, `TStatus`, `TStatusCode`
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![allow(clippy::all, clippy::deprecated_cfg_attr)]

include!(concat!(env!("OUT_DIR"), "/thrift_mods.rs"));
