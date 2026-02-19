//! Generated protobuf/gRPC bindings for Apache Doris PBackendService.
//!
//! This crate auto-generates Rust types from the Doris proto files
//! (vendored via git submodule at `thirdparty/apache-doris/`).
//!
//! Key types:
//! - `PExecPlanFragmentRequest`: Fragment execution request (contains Thrift-serialized params)
//! - `PCancelPlanFragmentRequest`: Fragment cancellation
//! - `PTransmitDataParams`: Inter-BE data exchange
//! - `p_backend_service_server`: tonic gRPC server trait for PBackendService

#[allow(clippy::all)]
pub mod doris {
    tonic::include_proto!("doris");

    pub mod segment_v2 {
        tonic::include_proto!("doris.segment_v2");
    }
}

pub use doris::*;

#[allow(clippy::all)]
pub mod brpc {
    pub mod policy {
        include!(concat!(env!("OUT_DIR"), "/brpc.policy.rs"));
    }
}

#[allow(clippy::all)]
pub mod nixl {
    include!(concat!(env!("OUT_DIR"), "/doris.nixl.rs"));
}
