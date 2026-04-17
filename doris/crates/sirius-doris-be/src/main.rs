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

/// Parse a size string like "512MB", "1GB", "0" into bytes. Returns None for "0".
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if s == "0" {
        return None;
    }
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("GB") {
        (n.trim(), 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("MB") {
        (n.trim(), 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("KB") {
        (n.trim(), 1024)
    } else {
        (s, 1)
    };
    num_str.parse::<usize>().ok().map(|n| n * multiplier)
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
        // Placeholder: resolved to actual IP/hostname in run() before gRPC starts.
        // Heartbeat/backend services don't use this field.
        advertise_host: config.advertise_host.clone().unwrap_or_else(|| "127.0.0.1".to_string()),
    });

    let engine = SiriusEngine::new()
        .expect("FATAL: engine init failed (substrait + sirius extensions must be loadable)");
    info!("DuckDB engine initialized");
    info!("Super Sirius runtime initialized");
    if config.no_cpu_fallback {
        match engine.set_no_cpu_fallback() {
            Ok(()) => info!("CPU fallback disabled (enable_fallback_check = true)"),
            Err(err) => warn!(error = %err, "failed to set enable_fallback_check"),
        }
    }
    let engine = Some(Arc::new(Mutex::new(engine)));

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
    let staging_size = parse_size(&config.gpu_staging_size);
    let nixl_agent = {
        let agent_name = format!(
            "sirius-be-{}:{}",
            config.advertise_host.as_deref().unwrap_or("localhost"),
            config.brpc_port
        );
        // Create nixl agent without staging first, then allocate staging via C++.
        doris_rpc::nixl_exchange::NixlExchange::try_new_with_staging(
            &agent_name,
            None, // No Rust-side staging (RTLD_LOCAL makes cuMemAlloc inaccessible from C++)
        )
            .map(|a| std::sync::Arc::new(a))
    };
    // Get pre-allocated exchange staging buffers from cuCascade (allocated during SiriusContext init).
    // 1. Send staging: used by C++ cudf::chunked_pack to pack outgoing GPU data
    // 2. Receive staging: registered with nixl, used for incoming GPU transfers
    // They must be separate because the receiver's exchange table data (in receive
    // staging) is read by the GPU scan while the sender packs new data (in send staging).
    if let (Some(ref agent), Some(ref eng)) = (&nixl_agent, &engine) {
        let eng_guard = eng.lock().unwrap();

        // Send staging: C++ packs outgoing data here via cudf::chunked_pack.
        // Registered with nixl so it can transfer to remote BEs.
        let send_ok = match eng_guard.get_exchange_staging("send") {
            Ok((addr, size)) => {
                info!(addr = format_args!("0x{addr:x}"), size_mb = size / (1024 * 1024),
                      "got SEND staging from cuCascade");
                if let Err(e) = agent.register_send_staging(addr, size) {
                    warn!(error = %e, "failed to register send staging with nixl");
                }
                true
            }
            Err(e) => { warn!(error = %e, "send staging get failed"); false }
        };

        // Receive staging: nixl writes incoming transfers here.
        // Exchange tables point into this buffer (via registerExternalTablePacked).
        let recv_ok = match eng_guard.get_exchange_staging("recv") {
            Ok((addr, size)) => {
                info!(addr = format_args!("0x{addr:x}"), size_mb = size / (1024 * 1024),
                      "got RECV staging from cuCascade");
                match agent.register_staging_from_addr(addr, size) {
                    Ok(()) => { info!("recv staging registered with nixl"); true }
                    Err(e) => { warn!(error = %e, "recv staging nixl registration failed"); false }
                }
            }
            Err(e) => { warn!(error = %e, "recv staging get failed"); false }
        };

        if send_ok && recv_ok {
            info!("dual staging buffers ready (send + recv)");
        }

        drop(eng_guard);
    }
    if let Some(ref agent) = nixl_agent {
        let has_staging = agent.staging().is_some();
        info!(
            has_staging,
            staging_size_mb = staging_size.map(|s| s / (1024 * 1024)),
            "nixl GPU-direct exchange enabled"
        );
    } else {
        info!("nixl not available, using bRPC exchange fallback");
    }

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
    nixl_agent: Option<std::sync::Arc<doris_rpc::nixl_exchange::NixlExchange>>,
) {
    // Resolve advertise host: explicit flag, or system hostname → IPv4.
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

    // Update BeState with the resolved advertise_host for Arrow Flight responses.
    let grpc_state = Arc::new(BeState {
        advertise_host: advertise_host.clone(),
        ..(*grpc_state).clone()
    });

    if let Some(fe_addr) = &config.fe {
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

    // Build the local bRPC address for self-transfer detection in exchange sender.
    let local_brpc_addr = format!(
        "{}:{}",
        advertise_host,
        config.brpc_port,
    );

    if let Err(e) =
        doris_rpc::grpc_service::start_grpc_server(&grpc_addr, grpc_state, grpc_store, engine, exchange_buffer, config.no_cpu_fallback, config.force_cpu, config.nixl_only && !config.allow_brpc_fallback, nixl_agent, local_brpc_addr).await
    {
        error!(error = %e, "PBackendService gRPC server exited with error");
    }
}
