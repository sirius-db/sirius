//! Thrift and gRPC server implementations for the Doris BE protocol.
//!
//! - `heartbeat_service`: HeartbeatService (FE registration, health checks)
//! - `backend_service`: BackendService (agent tasks, metadata ops — mostly stubbed)
//! - `fragment_manager`: Fragment lifecycle management

pub mod backend_service;
pub mod brpc_server;
pub mod exchange_buffer;
pub mod fragment_manager;
pub mod grpc_service;
pub mod heartbeat_service;
pub mod pblock_decoder;
