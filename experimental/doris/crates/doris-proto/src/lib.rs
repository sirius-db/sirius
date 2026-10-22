//! Generated protobuf/gRPC bindings for the Apache Doris `PBackendService`.
//!
//! `build.rs` runs prost + tonic codegen over the `doris/gensrc/proto` submodule
//! (pinned to the same Doris tag as the FE binary).
//!
//! Types the backend uses most:
//! - `PExecPlanFragmentRequest` / `PExecPlanFragmentStartRequest`: fragment dispatch; the
//!   `request` bytes are a TCompact-encoded `TPipelineFragmentParamsList` (see `doris-thrift`)
//! - `PFetchDataRequest` / `PFetchDataResult`: result delivery (`row_batch` is a TBinary
//!   `TResultBatch`)
//! - `PFetchTableSchemaRequest` / `PGlobRequest`: analysis-time RPCs behind the `local()` TVF
//! - `p_backend_service_server::PBackendService`: the tonic server trait (the client in
//!   `p_backend_service_client` is for tests and, later, BE↔BE exchange)
#![allow(clippy::all, clippy::pedantic)]

pub mod doris {
    tonic::include_proto!("doris");

    pub mod segment_v2 {
        tonic::include_proto!("doris.segment_v2");
    }
}

pub use doris::*;
