use anyhow::Result;
use clap::Parser;
use sirius_starrocks_cn::{ComputeNode, ComputeNodeConfig, FeConfig, RegistrationConfig};

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
    let compute_node = ComputeNode::start(args.fe, args.compute_node, args.registration).await?;
    compute_node.wait_for_shutdown().await
}
