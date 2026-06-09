use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::Parser;
use sirius_starrocks_cn::{
    BackendServer, BackendServerShutdown, ComputeNodeConfig, FeConfig, HeartbeatServer,
    HeartbeatServerShutdown, SharedHeartbeatState, register_node, report_to_frontend_once,
    start_backend_server, start_heartbeat_server,
};
use tracing::{debug, error, info, warn};

const REGISTRATION_RETRY_INTERVAL: Duration = Duration::from_secs(1);
// The CN periodically refreshes registration when heartbeats stop arriving.
const REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
// If no heartbeat arrives within this window, registration is considered stale.
const HEARTBEAT_STALE_AFTER: Duration = Duration::from_secs(30);
// StarRocks BEs/CNs regularly report inventory; this skeleton reports empty state.
const FRONTEND_REPORT_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug, Parser)]
/// Command-line arguments for the StarRocks compute-node shim.
struct Args {
    #[command(flatten, next_help_heading = "FE")]
    fe: FeConfig,

    #[command(flatten, next_help_heading = "CN")]
    compute_node: ComputeNodeConfig,

    #[command(flatten, next_help_heading = "Registration")]
    registration: RegistrationConfig,
}

#[derive(Clone, Debug, clap::Args)]
/// Registration retry settings for initial FE registration.
struct RegistrationConfig {
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u32).range(1..))]
    registration_max_attempts: u32,
}

#[tokio::main]
/// Starts heartbeat/backend thrift services, registers with FE, and waits for shutdown.
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sirius_starrocks_cn=info,info".into()),
        )
        .init();

    let args = Args::parse();
    let state = SharedHeartbeatState::new();

    // HeartbeatService tells FE this process is alive and captures FE identity. The configured
    // FE host pins the report target so a hostile heartbeat cannot redirect outbound reports.
    let heartbeat_server = start_heartbeat_server(
        args.compute_node.clone(),
        state.clone(),
        Some(args.fe.host.clone()),
    )?;
    // BackendService exposes the shallow CN RPC skeleton on the normal thrift port.
    let backend_server = start_backend_server(&args.compute_node)?;
    // Initial SQL registration is needed before FE starts heartbeating this CN.
    register_node_with_retries(&args.fe, &args.compute_node, &args.registration).await?;

    // Registration stays active if FE heartbeats disappear after startup.
    let registration_task = tokio::spawn(maintain_registration(
        args.fe.clone(),
        args.compute_node.clone(),
        state.clone(),
    ));
    // Reports start after heartbeat records FE thrift address and backend id.
    let report_task = tokio::spawn(maintain_frontend_report(
        args.compute_node.clone(),
        state.clone(),
    ));

    info!("compute node registered; waiting for FE heartbeats");
    wait_until_shutdown(
        heartbeat_server,
        backend_server,
        registration_task,
        report_task,
    )
    .await
}

/// Registers the CN with FE SQL, retrying during FE startup or transient failures.
async fn register_node_with_retries(
    fe: &FeConfig,
    compute_node: &ComputeNodeConfig,
    registration: &RegistrationConfig,
) -> Result<()> {
    if registration.registration_max_attempts == 0 {
        return Err(anyhow!("registration-max-attempts must be at least 1"));
    }

    for attempt in 1..=registration.registration_max_attempts {
        match register_node(fe, compute_node).await {
            Ok(()) => return Ok(()),
            Err(err) => {
                // The last attempt returns the original registration error with context.
                if attempt == registration.registration_max_attempts {
                    return Err(anyhow!(
                        "failed to register compute node with FE after {} attempts: {err}",
                        registration.registration_max_attempts
                    ));
                }

                warn!(
                    error = %err,
                    attempt,
                    max_attempts = registration.registration_max_attempts,
                    retry_after_secs = REGISTRATION_RETRY_INTERVAL.as_secs(),
                    "failed to register compute node with FE; retrying"
                );
                tokio::time::sleep(REGISTRATION_RETRY_INTERVAL).await;
            }
        }
    }

    unreachable!("registration attempts loop always returns")
}

/// Periodically re-registers when heartbeat state indicates FE is no longer contacting us.
async fn maintain_registration(
    fe: FeConfig,
    compute_node: ComputeNodeConfig,
    state: SharedHeartbeatState,
) {
    loop {
        tokio::time::sleep(REGISTRATION_REFRESH_INTERVAL).await;

        // Recent heartbeat means FE still knows this CN, so no SQL refresh is needed.
        if let Some(elapsed) = state.last_heartbeat_elapsed()
            && elapsed < HEARTBEAT_STALE_AFTER
        {
            continue;
        }

        debug!(
            stale_after_secs = HEARTBEAT_STALE_AFTER.as_secs(),
            "heartbeat is stale or missing; ensuring compute node registration"
        );
        if let Err(err) = register_node(&fe, &compute_node).await {
            warn!(
                error = %err,
                retry_after_secs = REGISTRATION_REFRESH_INTERVAL.as_secs(),
                "failed to refresh compute node registration with FE"
            );
        }
    }
}

/// Periodically sends the FE a truthful empty inventory report after heartbeat identity exists.
async fn maintain_frontend_report(compute_node: ComputeNodeConfig, state: SharedHeartbeatState) {
    loop {
        tokio::time::sleep(FRONTEND_REPORT_INTERVAL).await;

        // Thrift clients are blocking, so run the single report call off the async runtime.
        let compute_node = compute_node.clone();
        let state = state.clone();
        match tokio::task::spawn_blocking(move || report_to_frontend_once(&compute_node, &state))
            .await
        {
            // FE accepted the empty inventory report.
            Ok(Ok(Some(_result))) => {
                debug!("reported compute node inventory to FE");
            }
            // Heartbeat has not yet supplied both FE address and backend id.
            Ok(Ok(None)) => {
                debug!("skipping FE report until heartbeat provides FE address and backend id");
            }
            // Network or FE thrift failures are retried by the next loop iteration.
            Ok(Err(err)) => {
                warn!(
                    error = %err,
                    retry_after_secs = FRONTEND_REPORT_INTERVAL.as_secs(),
                    "failed to report compute node inventory to FE"
                );
            }
            // A panic or cancellation in the blocking worker should not stop the CN process.
            Err(err) => {
                warn!(
                    error = %err,
                    retry_after_secs = FRONTEND_REPORT_INTERVAL.as_secs(),
                    "FE report worker failed"
                );
            }
        }
    }
}

/// Aborts the background maintenance tasks and asks both thrift servers to stop. Safe to call
/// even when a component has already exited: `abort()` on a finished task and `shutdown()` on an
/// already-stopped server are idempotent no-ops, so every shutdown path can stop everything.
fn request_stop(
    registration_task: &tokio::task::JoinHandle<()>,
    report_task: &tokio::task::JoinHandle<()>,
    heartbeat_shutdown: &HeartbeatServerShutdown,
    backend_shutdown: &BackendServerShutdown,
) {
    registration_task.abort();
    report_task.abort();
    heartbeat_shutdown.shutdown();
    backend_shutdown.shutdown();
}

/// Awaits a server's blocking-join task, flattening the join-task error and the server result.
async fn join_server(join: tokio::task::JoinHandle<Result<()>>, label: &str) -> Result<()> {
    join.await
        .map_err(|err| anyhow!("{label} server join task failed: {err}"))?
}

/// Coordinates shutdown across thrift server threads and background maintenance loops.
async fn wait_until_shutdown(
    heartbeat_server: HeartbeatServer,
    backend_server: BackendServer,
    mut registration_task: tokio::task::JoinHandle<()>,
    mut report_task: tokio::task::JoinHandle<()>,
) -> Result<()> {
    let heartbeat_shutdown = heartbeat_server.shutdown_handle();
    let backend_shutdown = backend_server.shutdown_handle();
    let mut heartbeat_join = tokio::task::spawn_blocking(move || heartbeat_server.join());
    let mut backend_join = tokio::task::spawn_blocking(move || backend_server.join());
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|err| anyhow!("failed to install SIGTERM handler: {err}"))?;

    tokio::select! {
        // Interactive shutdown path.
        signal = tokio::signal::ctrl_c() => {
            signal.map_err(|err| anyhow!("failed to wait for ctrl-c: {err}"))?;
            info!(signal = "ctrl-c", "shutdown signal received");
            request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown);
            join_server(heartbeat_join, "heartbeat").await?;
            join_server(backend_join, "backend").await?;
            info!("shutdown complete");
            Ok(())
        }
        // Service-manager shutdown path.
        _ = terminate.recv() => {
            info!(signal = "sigterm", "shutdown signal received");
            request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown);
            join_server(heartbeat_join, "heartbeat").await?;
            join_server(backend_join, "backend").await?;
            info!("shutdown complete");
            Ok(())
        }
        // If the heartbeat server exits first, stop everything else and surface its result.
        result = &mut heartbeat_join => {
            request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown);
            let result = result
                .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))?;
            if let Err(err) = &result {
                error!(error = %err, "heartbeat server exited");
            }
            join_server(backend_join, "backend").await?;
            result
        }
        // If the backend server exits first, stop everything else and surface its result.
        result = &mut backend_join => {
            request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown);
            let result = result
                .map_err(|err| anyhow!("backend server join task failed: {err}"))?;
            if let Err(err) = &result {
                error!(error = %err, "backend server exited");
            }
            join_server(heartbeat_join, "heartbeat").await?;
            result
        }
        // The registration loop is expected to run forever; an exit is treated as a failure.
        result = &mut registration_task => {
            request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown);
            join_server(heartbeat_join, "heartbeat").await?;
            join_server(backend_join, "backend").await?;
            result.map_err(|err| anyhow!("registration monitor task failed: {err}"))?;
            Err(anyhow!("registration monitor exited unexpectedly"))
        }
        // The report loop is expected to run forever; an exit is treated as a failure.
        result = &mut report_task => {
            request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown);
            join_server(heartbeat_join, "heartbeat").await?;
            join_server(backend_join, "backend").await?;
            result.map_err(|err| anyhow!("FE report task failed: {err}"))?;
            Err(anyhow!("FE report task exited unexpectedly"))
        }
    }
}
