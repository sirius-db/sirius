use std::{num::NonZeroU32, time::Duration};

use anyhow::{Result, anyhow};
use backon::{ExponentialBuilder, Retryable};
use clap::Parser;
use sirius_starrocks_cn::{
    BackendServer, BackendServerShutdown, BrpcServer, ComputeNodeConfig, FeConfig, HeartbeatServer,
    HeartbeatServerShutdown, SharedHeartbeatState, register_node, report_to_frontend_once,
    start_backend_server, start_heartbeat_server,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, instrument, warn};
use tracing_subscriber::fmt::format::FmtSpan;

// Initial delay for the startup registration backoff; doubles up to the cap on each retry.
const REGISTRATION_RETRY_INTERVAL: Duration = Duration::from_secs(1);
// Upper bound on the exponential backoff delay so a large attempt count stays bounded.
const REGISTRATION_MAX_RETRY_INTERVAL: Duration = Duration::from_secs(30);
const REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
const HEARTBEAT_STALE_AFTER: Duration = Duration::from_secs(30);
// StarRocks BEs/CNs regularly report inventory; this skeleton reports empty state.
const FRONTEND_REPORT_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug, Parser)]
struct Args {
    /// StarRocks FE connection settings.
    #[command(flatten, next_help_heading = "FE")]
    fe: FeConfig,

    /// Rust CN listener and advertised metadata settings.
    #[command(flatten, next_help_heading = "CN")]
    compute_node: ComputeNodeConfig,

    /// FE registration retry settings.
    #[command(flatten, next_help_heading = "Registration")]
    registration: RegistrationConfig,
}

#[derive(Clone, Debug, clap::Args)]
struct RegistrationConfig {
    /// Maximum FE registration attempts before startup fails.
    #[arg(long, default_value_t = NonZeroU32::new(120).expect("nonzero literal"))]
    registration_max_attempts: NonZeroU32,
}

impl Args {
    /// Starts the CN listeners, registers with FE, and waits for shutdown.
    #[instrument(name = "compute_node", skip_all)]
    async fn run(self) -> Result<()> {
        let state = SharedHeartbeatState::new();

        // HeartbeatService tells FE this process is alive and captures FE identity. The configured
        // FE host pins the report target so a hostile heartbeat cannot redirect outbound reports.
        let heartbeat_server = start_heartbeat_server(
            self.compute_node.clone(),
            state.clone(),
            Some(self.fe.host.clone()),
        )?;
        // BackendService exposes the shallow CN RPC skeleton on the normal thrift port.
        let backend_server = start_backend_server(&self.compute_node)?;
        // BRPC PInternalService dispatches plan fragments on the brpc port.
        let brpc_runtime = BrpcRuntime::start(&self.compute_node)?;
        self.registration
            .register_node_with_retries(&self.fe, &self.compute_node)
            .await?;

        // Reports start after heartbeat records FE thrift address and backend id.
        let report_task = tokio::spawn(maintain_frontend_report(
            self.compute_node.clone(),
            state.clone(),
        ));
        let registration_task =
            tokio::spawn(RegistrationMonitor::new(self.fe, self.compute_node, state).run());

        info!("compute node registered; waiting for FE heartbeats");
        RunningComputeNode {
            heartbeat_server,
            backend_server,
            brpc_runtime,
            registration_task,
            report_task,
        }
        .wait_until_shutdown()
        .await
    }
}

impl RegistrationConfig {
    /// Registers the compute node with FE, retrying with exponential backoff during FE startup
    /// or transient failures up to the configured maximum number of attempts.
    #[instrument(skip_all, fields(max_attempts = self.registration_max_attempts.get()))]
    async fn register_node_with_retries(
        &self,
        fe: &FeConfig,
        compute_node: &ComputeNodeConfig,
    ) -> Result<()> {
        let max_attempts = self.registration_max_attempts.get();
        // `with_max_times` counts retries after the first attempt, so total tries == max_attempts.
        let backoff = ExponentialBuilder::default()
            .with_min_delay(REGISTRATION_RETRY_INTERVAL)
            .with_max_delay(REGISTRATION_MAX_RETRY_INTERVAL)
            .with_max_times(max_attempts as usize - 1);

        (|| register_node(fe, compute_node))
            .retry(backoff)
            .notify(|err, delay| {
                warn!(
                    error = %err,
                    retry_after_secs = delay.as_secs(),
                    "failed to register compute node with FE; retrying"
                );
            })
            .await
            .map_err(|err| {
                anyhow!(
                    "failed to register compute node with FE after {max_attempts} attempts: {err}"
                )
            })
    }
}

/// Periodic FE registration refresher used when heartbeats become stale.
struct RegistrationMonitor {
    /// StarRocks FE connection settings.
    fe: FeConfig,
    /// Rust CN metadata expected to be present in FE.
    compute_node: ComputeNodeConfig,
    /// Shared heartbeat state used to detect missing or stale heartbeats.
    state: SharedHeartbeatState,
}

impl RegistrationMonitor {
    /// Builds the background registration monitor.
    fn new(fe: FeConfig, compute_node: ComputeNodeConfig, state: SharedHeartbeatState) -> Self {
        Self {
            fe,
            compute_node,
            state,
        }
    }

    /// Runs the monitor until the task is aborted during process shutdown.
    async fn run(self) {
        loop {
            tokio::time::sleep(REGISTRATION_REFRESH_INTERVAL).await;

            if let Some(elapsed) = self.state.last_heartbeat_elapsed()
                && elapsed < HEARTBEAT_STALE_AFTER
            {
                continue;
            }

            debug!(
                stale_after_secs = HEARTBEAT_STALE_AFTER.as_secs(),
                "heartbeat is stale or missing; ensuring compute node registration"
            );
            if let Err(err) = register_node(&self.fe, &self.compute_node).await {
                warn!(
                    error = %err,
                    retry_after_secs = REGISTRATION_REFRESH_INTERVAL.as_secs(),
                    "failed to refresh compute node registration with FE"
                );
            }
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

/// Dedicated current-thread Tokio runtime used for the BRPC service future.
struct BrpcRuntime {
    /// Cancellation token passed into `BrpcServer::serve_with_listener_shutdown`.
    shutdown: CancellationToken,
    /// Blocking task that owns the current-thread runtime and BRPC listener.
    join: tokio::task::JoinHandle<Result<()>>,
}

impl BrpcRuntime {
    /// Binds the BRPC listener and starts serving it on a dedicated runtime.
    fn start(compute_node: &ComputeNodeConfig) -> Result<Self> {
        let listener = BrpcServer::bind(compute_node.bind_host.as_str(), compute_node.brpc_port)?;
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = tokio::task::spawn_blocking(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .map_err(|err| anyhow!("failed to create BRPC service runtime: {err}"))?;
            runtime.block_on(
                BrpcServer::new()
                    .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });

        Ok(Self { shutdown, join })
    }
}

/// Active CN listener handles plus the background maintenance tasks.
struct RunningComputeNode {
    /// Blocking thrift heartbeat server handle.
    heartbeat_server: HeartbeatServer,
    /// Blocking thrift backend service server handle.
    backend_server: BackendServer,
    /// BRPC runtime task and shutdown token.
    brpc_runtime: BrpcRuntime,
    /// Background task that refreshes FE registration when heartbeats are stale.
    registration_task: tokio::task::JoinHandle<()>,
    /// Background task that reports empty CN inventory to FE.
    report_task: tokio::task::JoinHandle<()>,
}

impl RunningComputeNode {
    /// Waits until a signal, server exit, or task exit requires process shutdown.
    async fn wait_until_shutdown(self) -> Result<()> {
        let heartbeat_shutdown = self.heartbeat_server.shutdown_handle();
        let backend_shutdown = self.backend_server.shutdown_handle();
        let brpc_shutdown = self.brpc_runtime.shutdown.clone();
        let mut heartbeat_join = tokio::task::spawn_blocking(move || self.heartbeat_server.join());
        let mut backend_join = tokio::task::spawn_blocking(move || self.backend_server.join());
        let mut brpc_join = self.brpc_runtime.join;
        let mut registration_task = self.registration_task;
        let mut report_task = self.report_task;
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .map_err(|err| anyhow!("failed to install SIGTERM handler: {err}"))?;

        tokio::select! {
            // Interactive shutdown path.
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(|err| anyhow!("failed to wait for ctrl-c: {err}"))?;
                info!(signal = "ctrl-c", "shutdown signal received");
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                join_server(heartbeat_join, "heartbeat").await?;
                join_server(backend_join, "backend").await?;
                join_server(brpc_join, "BRPC").await?;
                info!("shutdown complete");
                Ok(())
            }
            // Service-manager shutdown path.
            _ = terminate.recv() => {
                info!(signal = "sigterm", "shutdown signal received");
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                join_server(heartbeat_join, "heartbeat").await?;
                join_server(backend_join, "backend").await?;
                join_server(brpc_join, "BRPC").await?;
                info!("shutdown complete");
                Ok(())
            }
            // If the heartbeat server exits first, stop everything else and surface its result.
            result = &mut heartbeat_join => {
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                let result = result
                    .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))?;
                if let Err(err) = &result {
                    error!(error = %err, "heartbeat server exited");
                }
                join_server(backend_join, "backend").await?;
                join_server(brpc_join, "BRPC").await?;
                result
            }
            // If the backend server exits first, stop everything else and surface its result.
            result = &mut backend_join => {
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                let result = result
                    .map_err(|err| anyhow!("backend server join task failed: {err}"))?;
                if let Err(err) = &result {
                    error!(error = %err, "backend server exited");
                }
                join_server(heartbeat_join, "heartbeat").await?;
                join_server(brpc_join, "BRPC").await?;
                result
            }
            // If the BRPC server exits first, stop everything else and surface its result.
            result = &mut brpc_join => {
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                let result = result
                    .map_err(|err| anyhow!("BRPC server join task failed: {err}"))?;
                if let Err(err) = &result {
                    error!(error = %err, "BRPC server exited");
                }
                join_server(heartbeat_join, "heartbeat").await?;
                join_server(backend_join, "backend").await?;
                result
            }
            // The registration loop is expected to run forever; an exit is treated as a failure.
            result = &mut registration_task => {
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                join_server(heartbeat_join, "heartbeat").await?;
                join_server(backend_join, "backend").await?;
                join_server(brpc_join, "BRPC").await?;
                result.map_err(|err| anyhow!("registration monitor task failed: {err}"))?;
                Err(anyhow!("registration monitor exited unexpectedly"))
            }
            // The report loop is expected to run forever; an exit is treated as a failure.
            result = &mut report_task => {
                request_stop(&registration_task, &report_task, &heartbeat_shutdown, &backend_shutdown, &brpc_shutdown);
                join_server(heartbeat_join, "heartbeat").await?;
                join_server(backend_join, "backend").await?;
                join_server(brpc_join, "BRPC").await?;
                result.map_err(|err| anyhow!("FE report task failed: {err}"))?;
                Err(anyhow!("FE report task exited unexpectedly"))
            }
        }
    }
}

/// Aborts the background tasks and asks every server to stop. Safe to call even when a component
/// has already exited: `abort()` on a finished task, `shutdown()` on a stopped thrift server, and
/// `cancel()` on a fired token are all idempotent no-ops, so every shutdown path can stop everything.
fn request_stop(
    registration_task: &tokio::task::JoinHandle<()>,
    report_task: &tokio::task::JoinHandle<()>,
    heartbeat_shutdown: &HeartbeatServerShutdown,
    backend_shutdown: &BackendServerShutdown,
    brpc_shutdown: &CancellationToken,
) {
    registration_task.abort();
    report_task.abort();
    heartbeat_shutdown.shutdown();
    backend_shutdown.shutdown();
    brpc_shutdown.cancel();
}

/// Awaits a server's blocking-join task, flattening the join-task error and the server result.
async fn join_server(join: tokio::task::JoinHandle<Result<()>>, label: &str) -> Result<()> {
    join.await
        .map_err(|err| anyhow!("{label} server join task failed: {err}"))?
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sirius_starrocks_cn=info,info".into()),
        )
        // Emit span close events so instrumented spans report their busy/idle timings.
        .with_span_events(FmtSpan::CLOSE)
        .init();

    Args::parse().run().await
}
