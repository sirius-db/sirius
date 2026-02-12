//! Generated Thrift bindings for Apache Doris FE/BE protocol.
//!
//! This crate auto-generates Rust types from the Doris Thrift IDL files
//! (vendored via git submodule at `thirdparty/apache-doris/`).
//!
//! Key modules:
//! - `heartbeat_service`: HeartbeatService for BE registration with FE
//! - `backend_service`: BackendService for agent tasks and metadata ops
//! - `palo_internal_service`: TPipelineFragmentParams for fragment execution
//! - `plan_nodes`: TPlanNode, TPlanNodeType for query plan trees
//! - `exprs`: TExpr, TExprNode for expression trees
//! - `descriptors`: TDescriptorTable, TSlotDescriptor for column resolution
//! - `types`: TPrimitiveType, TTypeDesc, TUniqueId, TNetworkAddress
//! - `status`: TStatus, TStatusCode
//! - `data_sinks`: TDataSink, TDataSinkType

#[allow(
    dead_code,
    unused_imports,
    unused_extern_crates,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::vec_box,
    clippy::wrong_self_convention
)]
mod gen {
    include!(concat!(env!("OUT_DIR"), "/thrift_mods.rs"));
}

pub use gen::*;
