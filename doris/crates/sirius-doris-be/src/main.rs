//! Sirius GPU Backend for Apache Doris.
//!
//! This binary acts as a Doris Backend (BE) that receives query plan fragments
//! from the Doris Frontend (FE) via Thrift RPC, translates them to Substrait plans,
//! executes them on GPUs via the Sirius engine, and returns results via Arrow Flight.

mod config;

use std::sync::Arc;
use std::thread;
use std::time::SystemTime;

use tracing::{error, info};

use doris_rpc::heartbeat_service::BeState;
use result_formatter::result_store::ResultStore;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = config::BeConfig::default();

    info!(
        version = %config.version,
        heartbeat_port = config.heartbeat_port,
        be_port = config.be_port,
        brpc_port = config.brpc_port,
        arrow_flight_port = config.arrow_flight_port,
        gpu_ids = ?config.gpu_ids,
        "starting Sirius Doris BE"
    );

    let start_time_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    let state = Arc::new(BeState {
        be_port: config.be_port as i32,
        http_port: config.http_port as i32,
        brpc_port: config.brpc_port as i32,
        arrow_flight_port: config.arrow_flight_port as i32,
        version: config.version.clone(),
        start_time_ms,
    });

    // Shared result store between gRPC handler and Arrow Flight server.
    let result_store = ResultStore::new();

    // Start HeartbeatService in a dedicated thread (blocking Thrift server)
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

    // Start BackendService in a dedicated thread (blocking Thrift server)
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

    // Start async services (gRPC + Arrow Flight) on the tokio runtime.
    let grpc_addr = format!("0.0.0.0:{}", config.brpc_port);
    let flight_addr = format!("0.0.0.0:{}", config.arrow_flight_port);
    let grpc_state = state.clone();
    let grpc_store = result_store.clone();
    let flight_store = result_store.clone();

    info!("all services starting");

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        // Start Arrow Flight server as a background task.
        tokio::spawn(async move {
            if let Err(e) =
                result_formatter::arrow_flight::start_flight_server(&flight_addr, flight_store)
                    .await
            {
                error!(error = %e, "Arrow Flight server exited with error");
            }
        });

        // Run PBackendService gRPC server on the main async task.
        if let Err(e) =
            doris_rpc::grpc_service::start_grpc_server(&grpc_addr, grpc_state, grpc_store).await
        {
            error!(error = %e, "PBackendService gRPC server exited with error");
        }
    });
}
