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

#[instrument(skip_all, fields(%fe_addr, heartbeat_port))]
async fn register_with_fe(fe_addr: &str, heartbeat_port: u16) -> anyhow::Result<()> {
    use mysql_async::prelude::*;
    let url = format!("mysql://root@{}", fe_addr);
    let pool = mysql_async::Pool::new(url.as_str());
    let mut conn = pool.get_conn().await?;
    let stmt = format!(
        "ALTER SYSTEM ADD BACKEND '127.0.0.1:{}'",
        heartbeat_port
    );
    conn.query_drop(&stmt).await?;
    pool.disconnect().await?;
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

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(run(config, version, grpc_addr, flight_addr, grpc_state, grpc_store, flight_store, engine));
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
) {
    if let Some(fe_addr) = &config.fe {
        if let Err(e) = register_with_fe(fe_addr, config.heartbeat_port).await {
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

    if let Err(e) =
        doris_rpc::grpc_service::start_grpc_server(&grpc_addr, grpc_state, grpc_store, engine).await
    {
        error!(error = %e, "PBackendService gRPC server exited with error");
    }
}
