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

    /// Advertised hostname/IP for this BE (used in FE registration).
    /// Defaults to the system hostname.
    #[arg(long)]
    pub advertise_host: Option<String>,

    /// Disable CPU fallback: GPU execution errors are returned as-is instead
    /// of silently falling back to DuckDB CPU execution. Useful for evaluating
    /// Sirius GPU coverage (e.g. TPC-H queries).
    #[arg(long, default_value_t = false)]
    pub no_cpu_fallback: bool,

    /// Force CPU execution: all queries use DuckDB's from_substrait/execute_sql
    /// instead of the Sirius GPU pipeline. Useful when the GPU runtime is not
    /// functional (e.g. CUDA errors in Docker) but you still want correct results.
    #[arg(long, default_value_t = false)]
    pub force_cpu: bool,

    /// GPU cache buffer size (e.g. "2GB", "1024MB").
    #[arg(long, default_value = "2GB")]
    pub gpu_cache_size: String,

    /// GPU processing buffer size (e.g. "2GB", "1024MB").
    #[arg(long, default_value = "2GB")]
    pub gpu_processing_size: String,

    /// Disable bRPC exchange fallback: require nixl GPU-direct for all inter-BE
    /// exchanges. Queries fail instead of silently falling back to bRPC.
    /// Useful for debugging/iterating on the nixl exchange path.
    #[arg(long, default_value_t = false)]
    pub nixl_only: bool,

    /// GPU staging buffer size for nixl transfers (e.g. "512MB", "1GB").
    /// When set, a fixed GPU buffer is allocated via cuMemAlloc and registered
    /// with nixl once at startup, avoiding per-transfer alloc+register cycles.
    /// Set to "0" to disable (falls back to per-transfer cuMemAlloc).
    #[arg(long, default_value = "512MB")]
    pub gpu_staging_size: String,
}
