//! Thrift and gRPC server implementations for the Doris BE protocol.
//!
//! - `heartbeat_service`: HeartbeatService (FE registration, health checks)
//! - `backend_service`: BackendService (agent tasks, metadata ops — mostly stubbed)
//! - `fragment_manager`: Fragment lifecycle management

pub mod arrow_to_pblock;
pub mod backend_service;
pub mod brpc_server;
pub mod cuda_driver;
pub mod exchange_buffer;
pub mod exchange_sender;
pub mod fragment_manager;
pub mod grpc_service;
pub mod heartbeat_service;
pub mod nixl_exchange;
pub mod nixl_integration;
pub mod nixl_service;
#[cfg(test)]
mod nixl_exchange_mock;
pub mod pblock_decoder;
