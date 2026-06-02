use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::Parser;
use sirius_starrocks_cn::{
    ComputeNodeConfig, FeConfig, HeartbeatServer, SharedHeartbeatState, register_node,
    start_heartbeat_server,
};
use tracing::{debug, error, info, warn};

const REGISTRATION_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const REGISTRATION_REFRESH_INTERVAL: Duration = Duration::from_secs(10);
const HEARTBEAT_STALE_AFTER: Duration = Duration::from_secs(30);

#[derive(Debug, Parser)]
struct Args {
    #[command(flatten, next_help_heading = "FE")]
    fe: FeConfig,

    #[command(flatten, next_help_heading = "CN")]
    compute_node: ComputeNodeConfig,

    #[command(flatten, next_help_heading = "Registration")]
    registration: RegistrationConfig,
}

#[derive(Clone, Debug, clap::Args)]
struct RegistrationConfig {
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u32).range(1..))]
    registration_max_attempts: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sirius_starrocks_cn=info,info".into()),
        )
        .init();

    let args = Args::parse();
    let state = SharedHeartbeatState::new();

    let server = start_heartbeat_server(args.compute_node.clone(), state.clone())?;
    register_node_with_retries(&args.fe, &args.compute_node, &args.registration).await?;

    let registration_task = tokio::spawn(maintain_registration(
        args.fe.clone(),
        args.compute_node.clone(),
        state.clone(),
    ));

    info!("compute node registered; waiting for FE heartbeats");
    wait_until_shutdown(server, registration_task).await
}

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

async fn maintain_registration(
    fe: FeConfig,
    compute_node: ComputeNodeConfig,
    state: SharedHeartbeatState,
) {
    loop {
        tokio::time::sleep(REGISTRATION_REFRESH_INTERVAL).await;

        if let Some(elapsed) = state.last_heartbeat_elapsed() {
            if elapsed < HEARTBEAT_STALE_AFTER {
                continue;
            }
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

async fn wait_until_shutdown(
    server: HeartbeatServer,
    mut registration_task: tokio::task::JoinHandle<()>,
) -> Result<()> {
    let shutdown = server.shutdown_handle();
    let mut server_join = tokio::task::spawn_blocking(move || server.join());
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|err| anyhow!("failed to install SIGTERM handler: {err}"))?;

    tokio::select! {
        signal = tokio::signal::ctrl_c() => {
            signal.map_err(|err| anyhow!("failed to wait for ctrl-c: {err}"))?;
            info!(signal = "ctrl-c", "shutdown signal received");
            registration_task.abort();
            shutdown.shutdown();
            server_join
                .await
                .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))??;
            info!("shutdown complete");
            Ok(())
        }
        _ = terminate.recv() => {
            let signal = "sigterm";
            info!(signal, "shutdown signal received");
            registration_task.abort();
            shutdown.shutdown();
            server_join
                .await
                .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))??;
            info!("shutdown complete");
            Ok(())
        }
        result = &mut server_join => {
            registration_task.abort();
            let result = result
                .map_err(|err| anyhow!("heartbeat server join task failed: {err}"))?;
            if let Err(err) = &result {
                error!(error = %err, "heartbeat server exited");
            }
            result
        }
        result = &mut registration_task => {
            shutdown.shutdown();
            result.map_err(|err| anyhow!("registration monitor task failed: {err}"))?;
            Err(anyhow!("registration monitor exited unexpectedly"))
        }
    }
}
