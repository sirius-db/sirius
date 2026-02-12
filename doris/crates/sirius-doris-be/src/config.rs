//! Configuration for the Sirius Doris BE.

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "sirius-doris-be", about = "Sirius GPU Backend for Apache Doris")]
pub struct BeConfig {
    /// Heartbeat service port (FE connects here to register this BE).
    #[arg(long, default_value_t = 9050)]
    pub heartbeat_port: u16,

    /// BackendService Thrift port.
    #[arg(long, default_value_t = 9060)]
    pub be_port: u16,

    /// PBackendService gRPC port.
    #[arg(long, default_value_t = 8060)]
    pub brpc_port: u16,

    /// HTTP status port.
    #[arg(long, default_value_t = 8040)]
    pub http_port: u16,

    /// Arrow Flight result port.
    #[arg(long, default_value_t = 8071)]
    pub arrow_flight_port: u16,

    /// GPU device IDs (comma-separated).
    #[arg(long, value_delimiter = ',', default_values_t = vec![0])]
    pub gpu_ids: Vec<i32>,

    /// FE MySQL address for self-registration (e.g. 127.0.0.1:9030).
    #[arg(long)]
    pub fe: Option<String>,
}
