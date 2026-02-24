//! Sirius GPU Backend for Apache Doris.
//!
//! This binary acts as a Doris Backend (BE) that receives query plan fragments
//! from the Doris Frontend (FE) via Thrift RPC, translates them to Substrait plans,
//! executes them on GPUs via the Sirius engine, and returns results via Arrow Flight.

mod config;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::SystemTime;

use clap::Parser;
use tracing::{error, info, instrument, warn};

use doris_rpc::heartbeat_service::BeState;
use result_formatter::result_store::ResultStore;
use sirius_ffi::SiriusEngine;

#[instrument(skip_all, fields(%fe_addr, heartbeat_port, %advertise_host))]
async fn register_with_fe(fe_addr: &str, heartbeat_port: u16, advertise_host: &str) -> anyhow::Result<()> {
    use base64::Engine;
    // Use Doris HTTP SQL API — avoids MySQL protocol incompatibilities.
    let host = fe_addr.split(':').next().unwrap_or("127.0.0.1");
    let mysql_port: u16 = fe_addr.split(':').nth(1).and_then(|p| p.parse().ok()).unwrap_or(9030);
    // HTTP API is on port 8030 (mysql_port - 1000 by convention)
    let http_port = mysql_port - 1000;
    let stmt = format!("ALTER SYSTEM ADD BACKEND '{}:{}'", advertise_host, heartbeat_port);
    let url = format!("http://{}:{}/api/query/default_cluster/information_schema", host, http_port);
    let body = format!(r#"{{"stmt":"{}"}}"#, stmt);
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Basic {}", base64::engine::general_purpose::STANDARD.encode(b"root:")))
        .body(body)
        .send()
        .await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("HTTP {}: {}", status, text);
    }
    Ok(())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_span_events(
            tracing_subscriber::fmt::format::FmtSpan::NEW
                | tracing_subscriber::fmt::format::FmtSpan::CLOSE,
        )
        .init();

    let config = config::BeConfig::parse();
    let version = format!("sirius-doris-be {}", env!("CARGO_PKG_VERSION"));

    let start_time_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let state = Arc::new(BeState {
        be_port: config.be_port as i32,
        http_port: config.http_port as i32,
        brpc_port: config.brpc_port as i32,
        arrow_flight_port: config.arrow_flight_port as i32,
        version: version.clone(),
        start_time_ms,
    });

    let engine = match SiriusEngine::new() {
        Ok(e) => {
            info!("DuckDB engine initialized");
            // Try to initialize GPU buffers for Sirius GPU execution.
            match e.init_gpu_buffers("2GB", "2GB") {
                Ok(()) => info!("GPU buffers initialized (2GB cache, 2GB processing)"),
                Err(err) => warn!(error = %err, "GPU buffer init failed, gpu_execution will fall back to DuckDB CPU"),
            }
            Some(Arc::new(Mutex::new(e)))
        }
        Err(e) => {
            warn!(error = %e, "engine init failed, queries will error");
            None
        }
    };

    let result_store = ResultStore::new();

    let heartbeat_addr = format!("0.0.0.0:{}", config.heartbeat_port);
    let heartbeat_state = state.clone();
    let _heartbeat_thread = thread::Builder::new()
        .name("heartbeat-svc".to_string())
        .spawn(move || {
            if let Err(e) =
                doris_rpc::heartbeat_service::start_heartbeat_server(&heartbeat_addr, heartbeat_state)
            {
                error!(error = %e, "HeartbeatService exited with error");
            }
        })
        .expect("failed to spawn heartbeat thread");

    let backend_addr = format!("0.0.0.0:{}", config.be_port);
    let backend_state = state.clone();
    let _backend_thread = thread::Builder::new()
        .name("backend-svc".to_string())
        .spawn(move || {
            if let Err(e) =
                doris_rpc::backend_service::start_backend_server(&backend_addr, backend_state)
            {
                error!(error = %e, "BackendService exited with error");
            }
        })
        .expect("failed to spawn backend thread");

    let grpc_addr = format!("0.0.0.0:{}", config.brpc_port);
    let flight_addr = format!("0.0.0.0:{}", config.arrow_flight_port);
    let grpc_state = state.clone();
    let grpc_store = result_store.clone();
    let flight_store = result_store.clone();
    let exchange_buffer = doris_rpc::exchange_buffer::ExchangeBuffer::new();

    // Initialize nixl agent for GPU-direct exchange (optional, graceful fallback).
    #[cfg(feature = "nixl")]
    let nixl_agent = {
        let agent_name = format!(
            "sirius-be-{}:{}",
            config.advertise_host.as_deref().unwrap_or("localhost"),
            config.brpc_port
        );
        doris_rpc::nixl_exchange::NixlExchange::try_new(&agent_name)
            .map(|a| std::sync::Arc::new(a))
    };
    #[cfg(feature = "nixl")]
    if nixl_agent.is_some() {
        info!("nixl GPU-direct exchange enabled");
    } else {
        info!("nixl not available, using bRPC exchange fallback");
    }

    #[cfg(not(feature = "nixl"))]
    let nixl_agent = None;

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(run(config, version, grpc_addr, flight_addr, grpc_state, grpc_store, flight_store, engine, exchange_buffer, nixl_agent));
}

#[instrument(name = "sirius_doris_be", skip_all, fields(
    %version,
    heartbeat_port = %config.heartbeat_port,
    be_port = %config.be_port,
    brpc_port = %config.brpc_port,
    arrow_flight_port = %config.arrow_flight_port,
    gpu_ids = ?config.gpu_ids,
))]
async fn run(
    config: config::BeConfig,
    version: String,
    grpc_addr: String,
    flight_addr: String,
    grpc_state: Arc<BeState>,
    grpc_store: ResultStore,
    flight_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
    exchange_buffer: doris_rpc::exchange_buffer::ExchangeBuffer,
    #[cfg(feature = "nixl")]
    nixl_agent: Option<std::sync::Arc<doris_rpc::nixl_exchange::NixlExchange>>,
    #[cfg(not(feature = "nixl"))]
    _nixl_agent: Option<()>,
) {
    if let Some(fe_addr) = &config.fe {
        // Default advertise host: resolve system hostname to an IP.
        let advertise_host = match &config.advertise_host {
            Some(h) => h.clone(),
            None => {
                let hostname = std::process::Command::new("hostname")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "127.0.0.1".to_string());
                // Try to resolve hostname to an IPv4 address
                tokio::net::lookup_host(format!("{}:0", hostname))
                    .await
                    .ok()
                    .and_then(|mut addrs| addrs.find(|a| a.is_ipv4()))
                    .map(|a| a.ip().to_string())
                    .unwrap_or(hostname)
            }
        };
        if let Err(e) = register_with_fe(fe_addr, config.heartbeat_port, &advertise_host).await {
            warn!(error = %e, "FE registration failed (BE may already be registered)");
        }
    }

    tokio::spawn(async move {
        if let Err(e) =
            result_formatter::arrow_flight::start_flight_server(&flight_addr, flight_store).await
        {
            error!(error = %e, "Arrow Flight server exited with error");
        }
    });

    #[cfg(feature = "nixl")]
    let nixl_for_grpc = nixl_agent;
    #[cfg(not(feature = "nixl"))]
    let nixl_for_grpc = None;

    if let Err(e) =
        doris_rpc::grpc_service::start_grpc_server(&grpc_addr, grpc_state, grpc_store, engine, exchange_buffer, nixl_for_grpc).await
    {
        error!(error = %e, "PBackendService gRPC server exited with error");
    }
}
