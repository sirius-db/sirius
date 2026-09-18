use std::{num::NonZeroU32, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use backon::{ExponentialBuilder, Retryable};
use clap::Parser;
#[cfg(feature = "sirius-engine")]
use sirius_doris_be::SiriusEngine;
#[cfg(not(feature = "sirius-engine"))]
use sirius_doris_be::StubExecutor;
use sirius_doris_be::{
    BackendConfig, BackendServer, FeConfig, FragmentExecutor, HeartbeatServer,
    SharedHeartbeatState, SiriusBackendService, register_node, serve_backend_service,
    start_backend_server, start_heartbeat_server,
};
use tokio::task::{JoinError, JoinSet};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, instrument, warn};
use tracing_subscriber::fmt::format::FmtSpan;

// Initial delay for the startup registration backoff; doubles up to the cap on each retry.
const REGISTRATION_RETRY_INTERVAL: Duration = Duration::from_secs(1);
// Upper bound on the exponential backoff delay so a large attempt count stays bounded.
const REGISTRATION_MAX_RETRY_INTERVAL: Duration = Duration::from_secs(30);
const REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
// The FE heartbeats every 5 s (`heartbeat_interval_second`); three misses is stale.
const HEARTBEAT_STALE_AFTER: Duration = Duration::from_secs(30);

#[derive(Debug, Parser)]
struct Args {
    /// Doris FE connection settings.
    #[command(flatten, next_help_heading = "FE")]
    fe: FeConfig,

    /// Backend listener and advertised metadata settings.
    #[command(flatten, next_help_heading = "BE")]
    backend: BackendConfig,

    /// FE registration retry settings.
    #[command(flatten, next_help_heading = "Registration")]
    registration: RegistrationConfig,

    /// Sirius engine bring-up settings.
    #[command(flatten, next_help_heading = "Engine")]
    engine: EngineConfig,
}

#[derive(Clone, Debug, clap::Args)]
struct RegistrationConfig {
    /// Maximum FE registration attempts before startup fails.
    #[arg(long, default_value_t = NonZeroU32::new(120).expect("nonzero literal"))]
    registration_max_attempts: NonZeroU32,
}

#[derive(Clone, Debug, clap::Args)]
/// Sirius engine bring-up settings.
struct EngineConfig {
    /// Path to a Sirius YAML config file. When unset, built-in engine defaults are used.
    #[arg(long)]
    sirius_config: Option<PathBuf>,
}

impl Args {
    /// Starts the backend listeners, registers with FE, and waits for shutdown.
    #[instrument(name = "backend", skip_all)]
    async fn run(self) -> Result<()> {
        // Build the fragment executor before serving any RPC. Compiled with the engine, this brings
        // up the GPU engine on its dedicated thread (fail-fast: a bad config or GPU failure exits
        // before FE can route work here); otherwise it is a stub. The handle is held for the
        // process lifetime and torn down after the servers stop, below.
        #[cfg(feature = "sirius-engine")]
        let executor: Arc<dyn FragmentExecutor> = Arc::new(
            SiriusEngine::start(self.engine.sirius_config.clone()).map_err(|err| anyhow!(err))?,
        );
        #[cfg(not(feature = "sirius-engine"))]
        let executor: Arc<dyn FragmentExecutor> = {
            warn_engine_disabled(&self.engine);
            Arc::new(StubExecutor)
        };

        let state = SharedHeartbeatState::new();

        // HeartbeatService keeps this node Alive in SHOW BACKENDS and records FE identity.
        let heartbeat_server = start_heartbeat_server(self.backend.clone(), state.clone())?;
        // BackendService answers the FE's periodic probes on be_port.
        let backend_server = start_backend_server(&self.backend)?;
        // PBackendService (gRPC) takes TVF schema requests, fragments, and result polls on
        // brpc_port.
        let grpc_server = GrpcServer::start(&self.backend, executor.clone()).await?;
        self.registration
            .register_node_with_retries(&self.fe, &self.backend)
            .await?;

        let registration_task =
            tokio::spawn(RegistrationMonitor::new(self.fe, self.backend, state).run());

        info!("backend registered; waiting for FE heartbeats");
        let result = RunningBackend {
            heartbeat_server,
            backend_server,
            grpc_server,
            registration_task,
        }
        .wait_until_shutdown()
        .await;

        // The servers have stopped by the time `wait_until_shutdown` returns, so no in-flight RPC
        // can touch the engine. Drop the executor last for an ordered teardown — the engine closes
        // its thread and tears down the context (joined) here.
        #[cfg(feature = "sirius-engine")]
        info!("tearing down Sirius engine");
        drop(executor);
        result
    }
}

/// Warns when an engine config is supplied but the engine was compiled out, so the flag is
/// silently ignored rather than honored.
#[cfg(not(feature = "sirius-engine"))]
fn warn_engine_disabled(engine: &EngineConfig) {
    if engine.sirius_config.is_some() {
        warn!("--sirius-config ignored: built without the `sirius-engine` feature");
    }
}

impl RegistrationConfig {
    /// Registers the backend with FE, retrying with exponential backoff during FE startup or
    /// transient failures up to the configured maximum number of attempts.
    #[instrument(skip_all, fields(max_attempts = self.registration_max_attempts.get()))]
    async fn register_node_with_retries(
        &self,
        fe: &FeConfig,
        backend: &BackendConfig,
    ) -> Result<()> {
        let max_attempts = self.registration_max_attempts.get();
        // `with_max_times` counts retries after the first attempt, so total tries == max_attempts.
        let backoff = ExponentialBuilder::default()
            .with_min_delay(REGISTRATION_RETRY_INTERVAL)
            .with_max_delay(REGISTRATION_MAX_RETRY_INTERVAL)
            .with_max_times(max_attempts as usize - 1);

        (|| register_node(fe, backend))
            .retry(backoff)
            .notify(|err, delay| {
                warn!(
                    error = format!("{err:#}"),
                    retry_after_secs = delay.as_secs(),
                    "failed to register backend with FE; retrying"
                );
            })
            .await
            .map_err(|err| {
                anyhow!("failed to register backend with FE after {max_attempts} attempts: {err}")
            })
    }
}

/// Periodic FE registration refresher used when heartbeats become stale (e.g. the FE was
/// wiped and restarted, forgetting this backend).
struct RegistrationMonitor {
    fe: FeConfig,
    backend: BackendConfig,
    state: SharedHeartbeatState,
}

impl RegistrationMonitor {
    fn new(fe: FeConfig, backend: BackendConfig, state: SharedHeartbeatState) -> Self {
        Self { fe, backend, state }
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
                "heartbeat is stale or missing; ensuring backend registration"
            );
            if let Err(err) = register_node(&self.fe, &self.backend).await {
                warn!(
                    error = %err,
                    retry_after_secs = REGISTRATION_REFRESH_INTERVAL.as_secs(),
                    "failed to refresh backend registration with FE"
                );
            }
        }
    }
}

/// The tonic `PBackendService` server task and its shutdown token.
struct GrpcServer {
    shutdown: CancellationToken,
    join: tokio::task::JoinHandle<Result<()>>,
}

impl GrpcServer {
    /// Binds `brpc_port` (fail-fast on a clash) and serves `PBackendService` dispatching
    /// fragments to `executor`.
    async fn start(backend: &BackendConfig, executor: Arc<dyn FragmentExecutor>) -> Result<Self> {
        let listen_addr = format!("{}:{}", backend.bind_host, backend.brpc_port);
        let listener = tokio::net::TcpListener::bind(&listen_addr)
            .await
            .map_err(|err| {
                anyhow!("failed to bind PBackendService gRPC server at {listen_addr}: {err}")
            })?;
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let service = SiriusBackendService::with_executor(executor);
        let join = tokio::spawn(async move {
            serve_backend_service(listener, service, server_shutdown.cancelled_owned())
                .await
                .map_err(|err| anyhow!("PBackendService gRPC server failed: {err}"))
        });
        Ok(Self { shutdown, join })
    }
}

/// Active listener handles plus the background maintenance task.
struct RunningBackend {
    heartbeat_server: HeartbeatServer,
    backend_server: BackendServer,
    grpc_server: GrpcServer,
    registration_task: tokio::task::JoinHandle<()>,
}

impl RunningBackend {
    /// Waits until a signal, a server exit, or a background-task exit requires shutdown, then
    /// stops every component and drains the servers.
    async fn wait_until_shutdown(self) -> Result<()> {
        let heartbeat_shutdown = self.heartbeat_server.shutdown_handle();
        let backend_shutdown = self.backend_server.shutdown_handle();
        let grpc_shutdown = self.grpc_server.shutdown.clone();

        // Drive every server's join as a labelled task so the first exit can be observed in the
        // select and the rest drained with one loop.
        let mut servers: JoinSet<(&'static str, Result<()>)> = JoinSet::new();
        let heartbeat_server = self.heartbeat_server;
        servers.spawn_blocking(move || ("heartbeat", heartbeat_server.join()));
        let backend_server = self.backend_server;
        servers.spawn_blocking(move || ("backend", backend_server.join()));
        let grpc_join = self.grpc_server.join;
        servers.spawn(async move {
            let result = grpc_join
                .await
                .unwrap_or_else(|err| Err(anyhow!("gRPC server task failed: {err}")));
            ("grpc", result)
        });

        let mut registration_task = self.registration_task;
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .map_err(|err| anyhow!("failed to install SIGTERM handler: {err}"))?;

        // Identify the first shutdown trigger and the result to report.
        let outcome = tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(|err| anyhow!("failed to wait for ctrl-c: {err}"))?;
                info!(signal = "ctrl-c", "shutdown signal received");
                Ok(())
            }
            _ = terminate.recv() => {
                info!(signal = "sigterm", "shutdown signal received");
                Ok(())
            }
            Some(joined) = servers.join_next() => server_result(joined),
            result = &mut registration_task => {
                result.map_err(|err| anyhow!("registration monitor task failed: {err}"))?;
                Err(anyhow!("registration monitor exited unexpectedly"))
            }
        };

        // Stop the background task and every server (all idempotent), then drain the servers
        // that have not exited yet, keeping the first failure as the reported error.
        registration_task.abort();
        heartbeat_shutdown.shutdown();
        backend_shutdown.shutdown();
        grpc_shutdown.cancel();

        let mut result = outcome;
        while let Some(joined) = servers.join_next().await {
            if let Err(err) = server_result(joined)
                && result.is_ok()
            {
                result = Err(err);
            }
        }
        if result.is_ok() {
            info!("shutdown complete");
        }
        result
    }
}

/// Flattens a server join task's outcome into a single result, logging a server-side error.
fn server_result(joined: Result<(&'static str, Result<()>), JoinError>) -> Result<()> {
    match joined {
        Ok((server, result)) => {
            if let Err(err) = &result {
                error!(error = %err, server, "server exited");
            }
            result
        }
        Err(err) => Err(anyhow!("server join task failed: {err}")),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sirius_doris_be=info,info".into()),
        )
        // Emit span close events so instrumented spans report their busy/idle timings.
        .with_span_events(FmtSpan::CLOSE)
        .init();

    Args::parse().run().await
}
