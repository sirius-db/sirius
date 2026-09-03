use std::{num::NonZeroU32, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Result, anyhow};
use backon::{ExponentialBuilder, Retryable};
use clap::Parser;
#[cfg(not(feature = "sirius-engine"))]
use sirius_starrocks_cn::StubExecutor;
use sirius_starrocks_cn::{
    BackendServer, BrpcServer, ComputeNodeConfig, EngineReadiness, ExchangeIdentity, FeConfig,
    FragmentExecutor, HeartbeatServer, HttpServer, NixlTransport, SharedHeartbeatState, Tunables,
    register_node, report_to_frontend_once, start_backend_server, start_heartbeat_server,
    start_http_server,
};
#[cfg(feature = "sirius-engine")]
use sirius_starrocks_cn::{
    EngineSettings, SiriusEngine, cpu_affinity_for_gpu, derive_sirius_config_yaml,
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
    /// Conflicts with the memory carve-out flags: a full config already decides memory.
    #[arg(
        long,
        conflicts_with_all = ["gpu_memory_limit", "gpu_memory_fraction", "host_memory_limit"]
    )]
    sirius_config: Option<PathBuf>,

    /// GPU memory carve-out for this CN as an absolute size (e.g. `8GiB`, `0.5TiB`,
    /// `8589934592`). Passed verbatim to the engine's byte parser (K=1000, Ki=1024).
    #[arg(long, conflicts_with = "gpu_memory_fraction", value_parser = parse_byte_size)]
    gpu_memory_limit: Option<String>,

    /// GPU memory carve-out as a fraction of TOTAL device memory (not free); 0 < f <= 1.0.
    #[arg(long, value_parser = parse_memory_fraction)]
    gpu_memory_fraction: Option<f64>,

    /// CUDA device ordinal to run on; exported as `CUDA_VISIBLE_DEVICES` before engine
    /// bring-up. An already-exported `CUDA_VISIBLE_DEVICES` must name this same device, or
    /// bring-up is refused. Unset by default: the engine sees whatever the environment exposes.
    #[arg(long)]
    gpu_device: Option<u32>,

    /// Host (CPU) memory capacity for the engine as an absolute size (e.g. `12GiB`). Passed
    /// verbatim to the engine's byte parser.
    #[arg(long, value_parser = parse_byte_size)]
    host_memory_limit: Option<String>,

    /// Directory for engine artifacts (derived config, logs, telemetry). Defaults to
    /// `sirius-cn-<brpc_port>` under the current working directory.
    #[arg(long)]
    engine_dir: Option<PathBuf>,
}

/// Validates the shape of a byte-size flag: `^[0-9]+(\.[0-9]+)?\s*(B|[KMGT]i?B?)?$`. Only the
/// shape — the authoritative parser is the engine's C++ `parse_bytes`, so the accepted string
/// is passed through verbatim.
fn parse_byte_size(value: &str) -> Result<String, String> {
    let invalid = || {
        format!(
            "invalid byte size '{value}': expected <number>[ ]<unit> where unit is \
             B or K/M/G/T with optional 'i' (binary) and 'B' — e.g. 8GiB, 12GB, 8589934592"
        )
    };
    let digits = value.len() - value.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 {
        return Err(invalid());
    }
    let mut rest = &value[digits..];
    if let Some(fraction) = rest.strip_prefix('.') {
        let fraction_digits = fraction.len()
            - fraction
                .trim_start_matches(|c: char| c.is_ascii_digit())
                .len();
        if fraction_digits == 0 {
            return Err(invalid());
        }
        rest = &fraction[fraction_digits..];
    }
    let suffix = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let suffix_valid = match suffix {
        "" | "B" => true,
        _ => {
            let mut chars = suffix.chars();
            matches!(chars.next(), Some('K' | 'M' | 'G' | 'T'))
                && matches!(chars.as_str(), "" | "i" | "B" | "iB")
        }
    };
    if suffix_valid {
        Ok(value.to_string())
    } else {
        Err(invalid())
    }
}

/// Validates a GPU memory fraction: 0 < f <= 1.0 of TOTAL device memory.
fn parse_memory_fraction(value: &str) -> Result<f64, String> {
    let fraction: f64 = value
        .parse()
        .map_err(|err| format!("invalid GPU memory fraction '{value}': {err}"))?;
    if fraction > 0.0 && fraction <= 1.0 {
        Ok(fraction)
    } else {
        Err(format!(
            "GPU memory fraction must satisfy 0 < f <= 1.0 (fraction of TOTAL device memory), got '{value}'"
        ))
    }
}

impl Args {
    /// Starts the CN listeners, brings up the engine, registers with FE, and waits for shutdown.
    #[instrument(name = "compute_node", skip_all)]
    async fn run(self) -> Result<()> {
        // FIRST, before a port is bound or a GPU pool is reserved: read and validate every
        // transport tunable, and log what this CN actually got. A rejected value fails startup
        // here rather than surfacing as an unexplained timeout mid-sweep, and the log line is
        // the ground truth for the knobs.
        Tunables::resolve().map_err(|err| anyhow!("invalid CN transport tunable: {err}"))?;

        let state = SharedHeartbeatState::new();
        // Closed until the engine is up. Binding the listeners first (below) is what closes the
        // ~7 s window in which this process existed but answered nothing; this gate is what stops
        // that from turning into "answers, therefore gets scheduled, therefore fails a fragment".
        let readiness = EngineReadiness::warming();

        // LISTENERS FIRST, ENGINE SECOND. Engine start-up reserves the entire RMM pool and takes
        // ~7 s on a GB200, while the FE is up in ~4 s and immediately heartbeats every compute node
        // it remembers from its persisted metadata. Building the engine first left every port
        // refusing connections for that whole window, and the FE auto-blacklisted the nodes it
        // could not reach. Ordering it this way means the probe finds a listener; the readiness
        // gate above is what keeps the answer honest until the engine can actually run work.
        //
        // HeartbeatService tells FE this process is alive and captures FE identity. The configured
        // FE host pins the report target so a hostile heartbeat cannot redirect outbound reports.
        let heartbeat_server = start_heartbeat_server(
            self.compute_node.clone(),
            state.clone(),
            Some(self.fe.host.clone()),
            readiness.clone(),
        )?;
        // BackendService exposes the shallow CN RPC skeleton on the normal thrift port.
        let backend_server = start_backend_server(&self.compute_node)?;
        // The HTTP port is advertised in every heartbeat, and the FE refuses to lift a node's
        // blacklist entry until it can open a TCP connection to it.
        let http_server = start_http_server(&self.compute_node)?;

        // Now the expensive part. Compiled with the engine, this brings up the GPU engine on its
        // dedicated thread (fail-fast: a bad config or GPU failure exits before FE can route work
        // here); otherwise it is a stub. The handle is held for the process lifetime and torn down
        // after the servers stop, below. Failing here exits the process, which releases the ports
        // bound above — the FE sees the node go away rather than being told it is healthy.
        #[cfg(feature = "sirius-engine")]
        let executor: Arc<dyn FragmentExecutor> = {
            let settings = self.engine.resolve(&self.compute_node)?;
            self.engine.ensure_gpu_unclaimed()?;
            Arc::new(SiriusEngine::start(settings).map_err(|err| anyhow!(err))?)
        };
        #[cfg(not(feature = "sirius-engine"))]
        let executor: Arc<dyn FragmentExecutor> = {
            warn_engine_disabled(&self.engine);
            Arc::new(StubExecutor)
        };
        // The cross-node exchange tier, when this build carries nixl and the staging arena is
        // configured; a remote exchange destination stays a loud error otherwise.
        let transport = build_nixl_transport(&self.compute_node, executor.clone())?;
        // BRPC PInternalService dispatches plan fragments on the brpc port.
        let brpc_runtime = BrpcRuntime::start(&self.compute_node, executor.clone(), transport)?;

        // Everything that can execute a fragment is now up, so start answering heartbeats OK and
        // let the FE schedule onto this node. Opening the gate before brpc bound would advertise a
        // node whose fragment endpoint is still refusing connections.
        readiness.mark_ready();
        info!("compute node is READY: engine, exchange and BRPC are up");

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
        let result = RunningComputeNode {
            heartbeat_server,
            backend_server,
            http_server,
            brpc_runtime,
            registration_task,
            report_task,
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

#[cfg(feature = "sirius-engine")]
impl EngineConfig {
    /// Resolves the CLI flags into engine settings, writing the derived carve-out config under
    /// the engine directory when a memory flag is set (clap forbids combining those flags with
    /// `--sirius-config`, so the two config sources never race).
    fn resolve(&self, compute_node: &ComputeNodeConfig) -> Result<EngineSettings> {
        let engine_dir = self
            .engine_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("sirius-cn-{}", compute_node.brpc_port)));
        // Confine the engine's otherwise-unpinned thread pools to the socket that owns both this
        // CN's GPU and its `numa_alloc_onnode` host arena. `None` (undiscoverable, or switched
        // off with SIRIUS_CN_CPU_AFFINITY) leaves them free-floating, as before.
        let cpu_affinity = cpu_affinity_for_gpu(self.gpu_device);
        let config = match derive_sirius_config_yaml(
            self.gpu_memory_limit.as_deref(),
            self.gpu_memory_fraction,
            self.host_memory_limit.as_deref(),
            &engine_dir,
            cpu_affinity.as_deref(),
        ) {
            Some(yaml) => {
                std::fs::create_dir_all(&engine_dir).map_err(|err| {
                    anyhow!(
                        "failed to create engine directory {}: {err}",
                        engine_dir.display()
                    )
                })?;
                let path = engine_dir.join("derived-sirius-config.yaml");
                std::fs::write(&path, yaml).map_err(|err| {
                    anyhow!(
                        "failed to write derived Sirius config {}: {err}",
                        path.display()
                    )
                })?;
                info!(config = %path.display(), "wrote derived Sirius config");
                Some(path)
            }
            None => self.sirius_config.clone(),
        };
        Ok(EngineSettings {
            config,
            engine_dir,
            gpu_device: self.gpu_device,
        })
    }

    /// Refuses a default-config bring-up while another process holds the GPU: the built-in
    /// config primes ~0.95x of device memory, so bring-up would abort deep inside rmm instead
    /// of failing with an actionable message. A missing or failing `nvidia-smi` only warns —
    /// the rmm abort remains the backstop.
    fn ensure_gpu_unclaimed(&self) -> Result<()> {
        // A supplied config or GPU carve-out means the operator already sized this CN.
        if self.sirius_config.is_some()
            || self.gpu_memory_limit.is_some()
            || self.gpu_memory_fraction.is_some()
        {
            return Ok(());
        }
        let mut command = std::process::Command::new("nvidia-smi");
        command.args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
        ]);
        if let Some(device) = self.gpu_device {
            command.args(["-i", &device.to_string()]);
        }
        let output = match command.output() {
            Ok(output) => output,
            Err(err) => {
                warn!(
                    error = %err,
                    "nvidia-smi unavailable; skipping the shared-GPU preflight (an \
                     over-committed device will still abort in rmm)"
                );
                return Ok(());
            }
        };
        if !output.status.success() {
            warn!(
                status = %output.status,
                stderr = %String::from_utf8_lossy(&output.stderr).trim(),
                "nvidia-smi failed; skipping the shared-GPU preflight (an over-committed \
                 device will still abort in rmm)"
            );
            return Ok(());
        }
        // Each row is "pid, used_memory MiB" for one compute process on the device.
        let holders: Vec<String> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| match line.split_once(',') {
                Some((pid, used)) => format!("pid {} holds {}", pid.trim(), used.trim()),
                None => line.to_string(),
            })
            .collect();
        if holders.is_empty() {
            return Ok(());
        }
        Err(anyhow!(
            "refusing to start with the default Sirius memory config: it primes ~0.95x of \
             device memory at bring-up, but another process already holds the GPU ({}). \
             Bring-up would abort with the rmm OOM 'std::bad_alloc: out_of_memory: CUDA error \
             (failed to allocate ...) cuda_async_view_memory_resource.hpp:87'. Pass \
             --gpu-memory-limit (e.g. 8GiB) to carve out a slice of the device, or \
             --sirius-config with an explicit memory config",
            holders.join("; ")
        ))
    }
}

/// Warns when engine flags are supplied but the engine was compiled out, so the flags are
/// loudly ignored rather than silently dropped.
#[cfg(not(feature = "sirius-engine"))]
fn warn_engine_disabled(engine: &EngineConfig) {
    let EngineConfig {
        sirius_config,
        gpu_memory_limit,
        gpu_memory_fraction,
        gpu_device,
        host_memory_limit,
        engine_dir,
    } = engine;
    if sirius_config.is_some()
        || gpu_memory_limit.is_some()
        || gpu_memory_fraction.is_some()
        || gpu_device.is_some()
        || host_memory_limit.is_some()
        || engine_dir.is_some()
    {
        warn!(
            "engine flags (--sirius-config / --gpu-memory-limit / --gpu-memory-fraction / \
             --gpu-device / --host-memory-limit / --engine-dir) ignored: built without the \
             `sirius-engine` feature"
        );
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

/// Brings up the nixl exchange transport when the build carries it AND the operator configured
/// the staging arena. Both conditions surface in logs: cross-node placements need this tier,
/// and its absence must be discoverable, not deduced from a later query failure.
#[cfg(feature = "nixl-transport")]
fn build_nixl_transport(
    compute_node: &ComputeNodeConfig,
    executor: Arc<dyn FragmentExecutor>,
) -> Result<Option<NixlTransport>> {
    if std::env::var_os("SIRIUS_EXCHANGE_STAGING_BYTES").is_none() {
        info!(
            "SIRIUS_EXCHANGE_STAGING_BYTES is not set, so there is no exchange staging arena: \
             the nixl cross-node exchange tier stays disabled and any remote exchange \
             destination will fail loudly"
        );
        return Ok(None);
    }
    // The agent is named by this CN's exchange identity, so two CNs on one host get distinct
    // agents and the FE-routed destination address doubles as the peer's agent name.
    let agent_name = format!("{}:{}", compute_node.advertise_host, compute_node.brpc_port);
    NixlTransport::start(executor, agent_name)
        .map(Some)
        .map_err(|err| anyhow!("failed to bring up the nixl exchange transport: {err}"))
}

/// Without the `nixl-transport` feature there is nothing to construct; remote exchange
/// destinations fail loudly with the build-time remedy in the message.
#[cfg(not(feature = "nixl-transport"))]
fn build_nixl_transport(
    _compute_node: &ComputeNodeConfig,
    _executor: Arc<dyn FragmentExecutor>,
) -> Result<Option<NixlTransport>> {
    if std::env::var_os("SIRIUS_EXCHANGE_STAGING_BYTES").is_some() {
        warn!(
            "SIRIUS_EXCHANGE_STAGING_BYTES is set but this CN was built without the \
             `nixl-transport` feature; the cross-node exchange tier stays disabled"
        );
    }
    Ok(None)
}

/// Dedicated current-thread Tokio runtime used for the BRPC service future.
struct BrpcRuntime {
    /// Cancellation token passed into `BrpcServer::serve_with_listener_shutdown`.
    shutdown: CancellationToken,
    /// Blocking task that owns the current-thread runtime and BRPC listener.
    join: tokio::task::JoinHandle<Result<()>>,
}

impl BrpcRuntime {
    /// Binds the BRPC listener and starts serving it on a dedicated runtime, dispatching
    /// fragments to `executor` and remote exchanges to `transport`.
    fn start(
        compute_node: &ComputeNodeConfig,
        executor: Arc<dyn FragmentExecutor>,
        transport: Option<NixlTransport>,
    ) -> Result<Self> {
        let listener = BrpcServer::bind(compute_node.bind_host.as_str(), compute_node.brpc_port)?;
        // The identity the FE routes exchanges by: the advertised host plus this brpc port.
        let identity =
            ExchangeIdentity::new(compute_node.advertise_host.as_str(), compute_node.brpc_port);
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = tokio::task::spawn_blocking(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .map_err(|err| anyhow!("failed to create BRPC service runtime: {err}"))?;
            runtime.block_on(
                BrpcServer::with_executor(executor, identity, transport)
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
    /// Blocking HTTP health listener on the advertised `--http-port`. Serves no CN traffic; it
    /// exists so the FE's blacklist-eviction reachability probe can succeed.
    http_server: HttpServer,
    /// BRPC runtime task and shutdown token.
    brpc_runtime: BrpcRuntime,
    /// Background task that refreshes FE registration when heartbeats are stale.
    registration_task: tokio::task::JoinHandle<()>,
    /// Background task that reports empty CN inventory to FE.
    report_task: tokio::task::JoinHandle<()>,
}

impl RunningComputeNode {
    /// Waits until a signal, a server exit, or a background-task exit requires shutdown, then
    /// stops every component and drains the servers.
    async fn wait_until_shutdown(self) -> Result<()> {
        let heartbeat_shutdown = self.heartbeat_server.shutdown_handle();
        let backend_shutdown = self.backend_server.shutdown_handle();
        let http_shutdown = self.http_server.shutdown_handle();
        let brpc_shutdown = self.brpc_runtime.shutdown.clone();

        // Drive every server's join as a labelled task so the first exit can be observed in the
        // select and the rest drained with one loop, instead of repeating the join logic per arm.
        let mut servers: JoinSet<(&'static str, Result<()>)> = JoinSet::new();
        let heartbeat_server = self.heartbeat_server;
        servers.spawn_blocking(move || ("heartbeat", heartbeat_server.join()));
        let backend_server = self.backend_server;
        servers.spawn_blocking(move || ("backend", backend_server.join()));
        let http_server = self.http_server;
        servers.spawn_blocking(move || ("http", http_server.join()));
        let brpc_join = self.brpc_runtime.join;
        servers.spawn(async move {
            let result = brpc_join
                .await
                .unwrap_or_else(|err| Err(anyhow!("BRPC service task failed: {err}")));
            ("BRPC", result)
        });

        let mut registration_task = self.registration_task;
        let mut report_task = self.report_task;
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
            result = &mut report_task => {
                result.map_err(|err| anyhow!("FE report task failed: {err}"))?;
                Err(anyhow!("FE report task exited unexpectedly"))
            }
        };

        // Stop the background tasks and every server (all idempotent), then drain the servers
        // that have not exited yet, keeping the first failure as the reported error.
        registration_task.abort();
        report_task.abort();
        heartbeat_shutdown.shutdown();
        backend_shutdown.shutdown();
        http_shutdown.shutdown();
        brpc_shutdown.cancel();

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
                .unwrap_or_else(|_| "sirius_starrocks_cn=info,info".into()),
        )
        // Emit span close events so instrumented spans report their busy/idle timings.
        .with_span_events(FmtSpan::CLOSE)
        .init();

    Args::parse().run().await
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use clap::error::ErrorKind;
    use clap::{CommandFactory, Parser};

    use super::*;

    /// Parses CLI arguments the way `main` would, with the binary name prepended.
    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(std::iter::once("sirius-starrocks-cn").chain(args.iter().copied()))
    }

    /// Clap's own consistency check over the whole derived command (conflict targets included).
    #[test]
    fn cli_definition_is_consistent() {
        Args::command().debug_assert();
    }

    /// An absolute limit and a fraction size the same carve-out; only one may be given.
    #[test]
    fn gpu_memory_limit_conflicts_with_fraction() {
        let err = parse(&["--gpu-memory-limit", "8GiB", "--gpu-memory-fraction", "0.5"])
            .expect_err("conflicting flags must not parse");
        assert_eq!(err.kind(), ErrorKind::ArgumentConflict);
    }

    /// A full config file already decides memory, so every memory flag conflicts with it.
    #[test]
    fn sirius_config_conflicts_with_each_memory_flag() {
        for (flag, value) in [
            ("--gpu-memory-limit", "8GiB"),
            ("--gpu-memory-fraction", "0.5"),
            ("--host-memory-limit", "12GiB"),
        ] {
            let err = parse(&["--sirius-config", "sirius.yaml", flag, value])
                .expect_err("memory flags must conflict with --sirius-config");
            assert_eq!(err.kind(), ErrorKind::ArgumentConflict, "{flag}");
        }
    }

    /// `--gpu-device` is env-level (not a YAML key), so it composes with a config file.
    #[test]
    fn sirius_config_composes_with_gpu_device() {
        let args = parse(&["--sirius-config", "sirius.yaml", "--gpu-device", "1"])
            .expect("--gpu-device is orthogonal to --sirius-config");
        assert_eq!(args.engine.gpu_device, Some(1));
        assert_eq!(
            args.engine.sirius_config.as_deref(),
            Some(Path::new("sirius.yaml"))
        );
    }

    /// The validator only checks shape; accepted values pass through verbatim for the C++
    /// `parse_bytes`.
    #[test]
    fn byte_size_validator_accepts_supported_shapes() {
        for value in ["8GiB", "8 GiB", "8589934592", "0.5TiB"] {
            assert_eq!(
                parse_byte_size(value).as_deref(),
                Ok(value),
                "{value:?} must be accepted verbatim"
            );
        }
    }

    /// Malformed sizes fail at the CLI, not deep inside engine bring-up.
    #[test]
    fn byte_size_validator_rejects_malformed_values() {
        for value in ["8GBB", "-1GiB", ""] {
            assert!(
                parse_byte_size(value).is_err(),
                "{value:?} must be rejected"
            );
        }
    }

    /// The fraction is of TOTAL device memory and must satisfy 0 < f <= 1.0.
    #[test]
    fn memory_fraction_validator_enforces_bounds() {
        assert_eq!(parse_memory_fraction("0.5"), Ok(0.5));
        assert_eq!(parse_memory_fraction("1.0"), Ok(1.0));
        for value in ["0", "0.0", "-0.5", "1.5", "NaN", "gpu"] {
            assert!(
                parse_memory_fraction(value).is_err(),
                "{value:?} must be rejected"
            );
        }
    }
}
