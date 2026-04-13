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
#[cfg(test)]
mod exchange_integration_tests;
pub mod exchange_sender;
pub mod fragment_manager;
pub mod gpu_buffer_reconstruct;
pub mod gpu_staging_buffer;
pub mod grpc_service;
pub mod hash_partitioner;
pub mod heartbeat_service;
pub mod ipc_reorder;
pub mod nixl_exchange;
#[cfg(test)]
mod nixl_exchange_mock;
pub mod nixl_integration;
pub mod nixl_service;
pub mod pblock_decoder;
pub mod transfer_engine;
