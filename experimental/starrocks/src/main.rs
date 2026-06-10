use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::Parser;
use sirius_starrocks_cn::{
    BrpcServer, ComputeNodeConfig, FeConfig, HeartbeatServer, SharedHeartbeatState, register_node,
    start_heartbeat_server,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

const REGISTRATION_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
const HEARTBEAT_STALE_AFTER: Duration = Duration::from_secs(30);

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
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u32).range(1..))]
    registration_max_attempts: u32,
}

impl Args {
    /// Starts both CN listeners, registers with FE, and waits for shutdown.
    async fn run(self) -> Result<()> {
        let state = SharedHeartbeatState::new();

        let heartbeat_server = start_heartbeat_server(self.compute_node.clone(), state.clone())?;
        let brpc_runtime = BrpcRuntime::start(&self.compute_node)?;
        self.registration
            .register_node_with_retries(&self.fe, &self.compute_node)
            .await?;

        let registration_task =
            tokio::spawn(RegistrationMonitor::new(self.fe, self.compute_node, state).run());

        info!("compute node registered; waiting for FE heartbeats");
        RunningComputeNode {
            heartbeat_server,
            brpc_runtime,
            registration_task,
        }
        .wait_until_shutdown()
        .await
    }
}

impl RegistrationConfig {
    /// Registers the compute node with retry handling used during startup.
    async fn register_node_with_retries(
        &self,
        fe: &FeConfig,
        compute_node: &ComputeNodeConfig,
    ) -> Result<()> {
        if self.registration_max_attempts == 0 {
            return Err(anyhow!("registration-max-attempts must be at least 1"));
        }

        for attempt in 1..=self.registration_max_attempts {
            match register_node(fe, compute_node).await {
                Ok(()) => return Ok(()),
                Err(err) => {
                    if attempt == self.registration_max_attempts {
                        return Err(anyhow!(
                            "failed to register compute node with FE after {} attempts: {err}",
                            self.registration_max_attempts
                        ));
                    }

                    warn!(
                        error = %err,
                        attempt,
                        max_attempts = self.registration_max_attempts,
                        retry_after_secs = REGISTRATION_RETRY_INTERVAL.as_secs(),
                        "failed to register compute node with FE; retrying"
                    );
                    tokio::time::sleep(REGISTRATION_RETRY_INTERVAL).await;
                }
            }
        }

        unreachable!("registration attempts loop always returns")
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

/// Active CN listener handles plus the background registration task.
struct RunningComputeNode {
    /// Blocking thrift heartbeat server handle.
    heartbeat_server: HeartbeatServer,
    /// BRPC runtime task and shutdown token.
    brpc_runtime: BrpcRuntime,
    /// Background task that refreshes FE registration when heartbeats are stale.
    registration_task: tokio::task::JoinHandle<()>,
}

impl RunningComputeNode {
    /// Waits until a signal, server exit, or monitor exit requires process shutdown.
    async fn wait_until_shutdown(self) -> Result<()> {
        let heartbeat_shutdown = self.heartbeat_server.shutdown_handle();
        let brpc_shutdown = self.brpc_runtime.shutdown.clone();
        let mut heartbeat_join = tokio::task::spawn_blocking(move || self.heartbeat_server.join());
        let mut brpc_join = self.brpc_runtime.join;
        let mut registration_task = self.registration_task;
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .map_err(|err| anyhow!("failed to install SIGTERM handler: {err}"))?;

        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(|err| anyhow!("failed to wait for ctrl-c: {err}"))?;
                info!(signal = "ctrl-c", "shutdown signal received");
                registration_task.abort();
                heartbeat_shutdown.shutdown();
                brpc_shutdown.cancel();
                heartbeat_join
                    .await
                    .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))??;
                brpc_join
                    .await
                    .map_err(|err| anyhow!("BRPC server join task failed: {err}"))??;
                info!("shutdown complete");
                Ok(())
            }
            _ = terminate.recv() => {
                info!(signal = "sigterm", "shutdown signal received");
                registration_task.abort();
                heartbeat_shutdown.shutdown();
                brpc_shutdown.cancel();
                heartbeat_join
                    .await
                    .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))??;
                brpc_join
                    .await
                    .map_err(|err| anyhow!("BRPC server join task failed: {err}"))??;
                info!("shutdown complete");
                Ok(())
            }
            result = &mut heartbeat_join => {
                registration_task.abort();
                brpc_shutdown.cancel();
                let result = result
                    .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))?;
                brpc_join
                    .await
                    .map_err(|err| anyhow!("BRPC server join task failed: {err}"))??;
                if let Err(err) = &result {
                    error!(error = %err, "heartbeat server exited");
                }
                result
            }
            result = &mut brpc_join => {
                registration_task.abort();
                heartbeat_shutdown.shutdown();
                let result = result
                    .map_err(|err| anyhow!("BRPC server join task failed: {err}"))?;
                heartbeat_join
                    .await
                    .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))??;
                if let Err(err) = &result {
                    error!(error = %err, "BRPC server exited");
                }
                result
            }
            result = &mut registration_task => {
                heartbeat_shutdown.shutdown();
                brpc_shutdown.cancel();
                heartbeat_join
                    .await
                    .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))??;
                brpc_join
                    .await
                    .map_err(|err| anyhow!("BRPC server join task failed: {err}"))??;
                result.map_err(|err| anyhow!("registration monitor task failed: {err}"))?;
                Err(anyhow!("registration monitor exited unexpectedly"))
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sirius_starrocks_cn=info,info".into()),
        )
        .init();

    Args::parse().run().await
}
