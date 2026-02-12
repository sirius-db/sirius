//! Configuration for the Sirius Doris BE.

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BeConfig {
    /// Heartbeat service port (FE connects here to register this BE).
    pub heartbeat_port: u16,
    /// Main Thrift port for BackendService.
    pub be_port: u16,
    /// gRPC/brpc port for PBackendService (fragment execution).
    pub brpc_port: u16,
    /// HTTP port (for status/metrics).
    pub http_port: u16,
    /// Arrow Flight SQL port for result delivery.
    pub arrow_flight_port: u16,
    /// GPU device IDs to use.
    pub gpu_ids: Vec<i32>,
    /// BE version string reported to FE.
    pub version: String,
    /// Timestamp when this BE process started.
    pub start_time: Instant,
}

impl Default for BeConfig {
    fn default() -> Self {
        Self {
            heartbeat_port: 9050,
            be_port: 9060,
            brpc_port: 8060,
            http_port: 8040,
            arrow_flight_port: 8070,
            gpu_ids: vec![0],
            version: format!("sirius-doris-be {}", env!("CARGO_PKG_VERSION")),
            start_time: Instant::now(),
        }
    }
}
