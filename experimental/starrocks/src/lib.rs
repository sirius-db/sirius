use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use mysql_async::{OptsBuilder, Pool, Row, prelude::Queryable};
use starrocks_thrift::{
    heartbeat_service::{
        HeartbeatServiceSyncHandler, HeartbeatServiceSyncProcessor, TBackendInfo, THeartbeatResult,
        TMasterInfo,
    },
    status::TStatus,
    status_code::TStatusCode,
    types,
};
use thrift::{
    TransportErrorKind,
    protocol::{
        TBinaryInputProtocolFactory, TBinaryOutputProtocolFactory, TInputProtocolFactory,
        TOutputProtocolFactory,
    },
    server::TProcessor,
    transport::{
        TBufferedReadTransportFactory, TBufferedWriteTransportFactory, TIoChannel,
        TReadTransportFactory, TTcpChannel, TWriteTransportFactory,
    },
};
use tracing::{debug, info, warn};

mod brpc;
mod internal_service;
mod proto;
mod prpc;

pub use brpc::BrpcServer;

type HeartbeatProcessor = HeartbeatServiceSyncProcessor<ComputeNodeHeartbeatHandler>;

const COMPUTE_NODE_PROC_PATH: &str = "/compute_nodes";

/// Non-empty host string accepted by StarRocks FE and CN configuration.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Host {
    /// Hostname or IP address text used in StarRocks-facing network metadata.
    value: String,
}

impl Host {
    /// Creates a host value after rejecting empty or whitespace-only input.
    pub fn new(value: impl Into<String>) -> std::result::Result<Self, HostParseError> {
        let value = value.into();
        if value.trim().is_empty() {
            Err(HostParseError)
        } else {
            Ok(Self { value })
        }
    }

    /// Returns the loopback host used by local StarRocks defaults.
    pub fn local() -> Self {
        Self {
            value: "127.0.0.1".to_string(),
        }
    }

    /// Returns the wildcard bind host used by local listeners.
    pub fn unspecified() -> Self {
        Self {
            value: "0.0.0.0".to_string(),
        }
    }

    /// Borrows the underlying host string.
    pub fn as_str(&self) -> &str {
        &self.value
    }
}

impl fmt::Debug for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Host").field(&self.value).finish()
    }
}

impl fmt::Display for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.value)
    }
}

impl FromStr for Host {
    type Err = HostParseError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Parse error returned for an empty host string.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("host must not be empty")]
pub struct HostParseError;

/// Secret CLI/config string that redacts its debug representation.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct SecretString {
    /// Sensitive text kept out of debug output.
    value: String,
}

impl SecretString {
    /// Wraps a secret string without validation.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
        }
    }

    /// Exposes the raw secret for APIs that need plaintext credentials.
    pub fn expose_secret(&self) -> &str {
        &self.value
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretString(<redacted>)")
    }
}

impl FromStr for SecretString {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self::new(value))
    }
}

/// Rust compute-node listener ports and StarRocks-facing advertised metadata.
#[derive(Clone, Debug, clap::Args)]
pub struct ComputeNodeConfig {
    /// Local interface address for the Rust CN listeners.
    #[arg(long, default_value = "0.0.0.0")]
    pub bind_host: Host,
    /// Hostname or IP address advertised to StarRocks FE.
    #[arg(long, default_value = "127.0.0.1")]
    pub advertise_host: Host,
    /// Thrift heartbeat port expected by StarRocks FE.
    #[arg(long, default_value_t = 9050)]
    pub heartbeat_port: u16,
    /// Legacy backend thrift port reported to FE metadata.
    #[arg(long = "thrift-port", default_value_t = 9060)]
    pub thrift_port: u16,
    /// HTTP port reported to FE metadata.
    #[arg(long, default_value_t = 8040)]
    pub http_port: u16,
    /// BRPC port used for PInternalService plan-fragment dispatch.
    #[arg(long, default_value_t = 8060)]
    pub brpc_port: u16,
    /// Optional Arrow Flight SQL port reported to FE metadata.
    #[arg(long)]
    pub arrow_flight_port: Option<u16>,
    /// Version string advertised through StarRocks heartbeat responses.
    #[arg(skip = ComputeNodeConfig::default_version())]
    pub version: String,
}

impl ComputeNodeConfig {
    /// Returns the default version string advertised to StarRocks FE.
    fn default_version() -> String {
        format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
    }

    /// Builds the StarRocks registration statement for this compute node.
    fn registration_sql(&self) -> String {
        format!(
            "ALTER SYSTEM ADD COMPUTE NODE \"{}:{}\"",
            self.advertise_host, self.heartbeat_port
        )
    }

    /// Checks whether this compute node is present in FE's compute-node registry.
    async fn is_registered_in(&self, conn: &mut mysql_async::Conn) -> Result<bool> {
        let sql = format!("SHOW PROC '{COMPUTE_NODE_PROC_PATH}'");
        let rows: Vec<Row> = conn
            .query(sql)
            .await
            .context("failed to query FE compute node list")?;
        let heartbeat_port = self.heartbeat_port.to_string();

        for row in rows {
            let host = row
                .get::<String, _>("IP")
                .or_else(|| row.get::<String, _>(1));
            let port = row
                .get::<String, _>("HeartbeatPort")
                .or_else(|| row.get::<String, _>(2));

            if host.as_deref() == Some(self.advertise_host.as_str())
                && port.as_deref() == Some(heartbeat_port.as_str())
            {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

impl Default for ComputeNodeConfig {
    fn default() -> Self {
        Self {
            bind_host: Host::unspecified(),
            advertise_host: Host::local(),
            heartbeat_port: 9050,
            thrift_port: 9060,
            http_port: 8040,
            brpc_port: 8060,
            arrow_flight_port: None,
            version: Self::default_version(),
        }
    }
}

/// StarRocks FE connection settings used for compute-node registration.
#[derive(Clone, Debug, clap::Args)]
pub struct FeConfig {
    /// StarRocks FE MySQL protocol host.
    #[arg(long = "fe-host", default_value = "127.0.0.1")]
    pub host: Host,
    /// StarRocks FE MySQL protocol query port.
    #[arg(long = "fe-query-port", default_value_t = 9030)]
    pub query_port: u16,
    /// StarRocks FE user used for compute-node registration.
    #[arg(long = "fe-user", default_value = "root")]
    pub user: String,
    /// StarRocks FE password used for compute-node registration.
    #[arg(long = "fe-password", default_value = "")]
    pub password: SecretString,
}

impl Default for FeConfig {
    fn default() -> Self {
        Self {
            host: Host::local(),
            query_port: 9030,
            user: "root".to_string(),
            password: SecretString::default(),
        }
    }
}

impl FeConfig {
    /// Registers the compute node with this FE through the StarRocks MySQL protocol.
    pub async fn register_compute_node(&self, node: &ComputeNodeConfig) -> Result<()> {
        let opts = OptsBuilder::default()
            .ip_or_hostname(self.host.to_string())
            .tcp_port(self.query_port)
            .prefer_socket(false)
            .user(Some(self.user.clone()))
            .pass(Some(self.password.expose_secret().to_string()));
        let pool = Pool::new(opts);
        let mut conn = pool.get_conn().await.with_context(|| {
            format!(
                "failed to connect to FE at {}:{}",
                self.host, self.query_port
            )
        })?;

        if node.is_registered_in(&mut conn).await? {
            info!(
                host = %node.advertise_host,
                heartbeat_port = node.heartbeat_port,
                "compute node is already registered with FE"
            );
            drop(conn);
            pool.disconnect()
                .await
                .context("failed to disconnect FE MySQL pool")?;
            return Ok(());
        }

        let sql = node.registration_sql();
        info!(sql = %sql, "registering compute node with FE");
        if let Err(err) = conn.query_drop(sql).await {
            warn!(error = %err, "ALTER SYSTEM ADD COMPUTE NODE failed; checking whether compute node already exists");
            if !node.is_registered_in(&mut conn).await? {
                return Err(err).context("failed to register compute node with FE");
            }
        }

        if !node.is_registered_in(&mut conn).await? {
            bail!(
                "FE accepted registration but compute node {}:{} was not found in SHOW PROC '{}'",
                node.advertise_host,
                node.heartbeat_port,
                COMPUTE_NODE_PROC_PATH
            );
        }

        info!(
            host = %node.advertise_host,
            heartbeat_port = node.heartbeat_port,
            "compute node registration confirmed"
        );
        drop(conn);
        pool.disconnect()
            .await
            .context("failed to disconnect FE MySQL pool")?;
        Ok(())
    }
}

/// Point-in-time copy of heartbeat state accepted from StarRocks FE.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeartbeatStateSnapshot {
    /// Cluster id accepted from the first FE heartbeat that supplies one.
    pub cluster_id: Option<types::TClusterId>,
    /// FE token accepted from the first FE heartbeat that supplies one.
    pub token: Option<SecretString>,
    /// Highest FE epoch accepted so far.
    pub epoch: Option<types::TEpoch>,
    /// Compute-node id assigned by FE.
    pub compute_node_id: Option<i64>,
    /// Wall-clock millisecond timestamp of the most recent accepted heartbeat.
    pub last_heartbeat_ms: Option<u128>,
}

/// Mutable heartbeat state protected by `SharedHeartbeatState`.
#[derive(Debug, Default)]
struct HeartbeatState {
    /// Cluster id accepted from the first FE heartbeat that supplies one.
    cluster_id: Option<types::TClusterId>,
    /// FE token accepted from the first FE heartbeat that supplies one.
    token: Option<SecretString>,
    /// Highest FE epoch accepted so far.
    epoch: Option<types::TEpoch>,
    /// Compute-node id assigned by FE.
    compute_node_id: Option<i64>,
    /// Wall-clock millisecond timestamp of the most recent accepted heartbeat.
    last_heartbeat_ms: Option<u128>,
}

/// Shared heartbeat state updated by the thrift heartbeat handler.
#[derive(Clone, Debug)]
pub struct SharedHeartbeatState {
    /// Shared mutable heartbeat state guarded for the thrift listener thread.
    inner: Arc<Mutex<HeartbeatState>>,
}

impl SharedHeartbeatState {
    /// Creates empty heartbeat state for a newly started compute node.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HeartbeatState::default())),
        }
    }

    /// Captures the currently accepted FE heartbeat metadata.
    pub fn snapshot(&self) -> HeartbeatStateSnapshot {
        let state = self.inner.lock().expect("heartbeat state mutex poisoned");
        HeartbeatStateSnapshot {
            cluster_id: state.cluster_id,
            token: state.token.clone(),
            epoch: state.epoch,
            compute_node_id: state.compute_node_id,
            last_heartbeat_ms: state.last_heartbeat_ms,
        }
    }

    /// Returns how long ago the most recent accepted heartbeat arrived.
    pub fn last_heartbeat_elapsed(&self) -> Option<Duration> {
        let state = self.inner.lock().expect("heartbeat state mutex poisoned");
        let last_heartbeat_ms = state.last_heartbeat_ms?;
        let elapsed_ms = WallClock::unix_time_millis().saturating_sub(last_heartbeat_ms);
        Some(Duration::from_millis(
            elapsed_ms.min(u64::MAX as u128) as u64
        ))
    }
}

impl Default for SharedHeartbeatState {
    fn default() -> Self {
        Self::new()
    }
}

/// Thrift heartbeat handler that reports this process as a StarRocks compute node.
#[derive(Clone, Debug)]
pub struct ComputeNodeHeartbeatHandler {
    /// Static compute-node ports and advertised metadata.
    config: ComputeNodeConfig,
    /// Shared heartbeat state updated after accepted FE heartbeats.
    state: SharedHeartbeatState,
    /// Process start time reported as backend reboot time.
    reboot_time_secs: i64,
    /// Hardware thread count reported to StarRocks FE.
    hardware_cores: i32,
}

impl ComputeNodeHeartbeatHandler {
    /// Creates a heartbeat handler bound to static CN configuration and shared state.
    pub fn new(config: ComputeNodeConfig, state: SharedHeartbeatState) -> Self {
        Self {
            config,
            state,
            reboot_time_secs: WallClock::unix_time_secs(),
            hardware_cores: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get().min(i32::MAX as usize) as i32)
                .unwrap_or(0),
        }
    }

    fn handle_master_info(
        &self,
        master_info: &TMasterInfo,
    ) -> std::result::Result<(), HeartbeatError> {
        let received_node_type = master_info
            .node_type
            .ok_or(HeartbeatError::MissingNodeType)?;
        if received_node_type != types::TNodeType::COMPUTE {
            return Err(HeartbeatError::UnexpectedNodeType {
                received: received_node_type,
            });
        }

        let mut state = self
            .state
            .inner
            .lock()
            .map_err(|_| HeartbeatError::StatePoisoned)?;

        if let Some(epoch) = state.epoch
            && master_info.epoch < epoch
        {
            return Err(HeartbeatError::StaleEpoch {
                received: master_info.epoch,
                current: epoch,
            });
        }

        if let Some(cluster_id) = master_info.cluster_id {
            if let Some(current_cluster_id) = state.cluster_id {
                if cluster_id != current_cluster_id {
                    return Err(HeartbeatError::ClusterChanged {
                        received: cluster_id,
                        current: current_cluster_id,
                    });
                }
            } else {
                state.cluster_id = Some(cluster_id);
            }
        }

        if let Some(token) = &master_info.token {
            if let Some(current_token) = &state.token {
                if token != current_token.expose_secret() {
                    return Err(HeartbeatError::TokenChanged);
                }
            } else {
                state.token = Some(SecretString::new(token.clone()));
            }
        }

        state.epoch = Some(master_info.epoch);
        if let Some(compute_node_id) = master_info.backend_id {
            state.compute_node_id = Some(compute_node_id);
        }
        state.last_heartbeat_ms = Some(WallClock::unix_time_millis());

        Ok(())
    }

    fn compute_node_info(&self) -> TBackendInfo {
        TBackendInfo::new(
            i32::from(self.config.thrift_port),
            i32::from(self.config.http_port),
            Some(-1),
            Some(i32::from(self.config.brpc_port)),
            Some(self.config.version.clone()),
            Some(self.hardware_cores),
            None,
            Some(self.reboot_time_secs),
            Some(false),
            Some(0),
            Some(self.config.arrow_flight_port.map(i32::from).unwrap_or(-1)),
        )
    }

    /// Builds a successful StarRocks heartbeat status.
    fn ok_status() -> TStatus {
        TStatus::new(TStatusCode::OK, None)
    }

    /// Builds a failed StarRocks heartbeat status with the rejection reason.
    fn error_status(message: String) -> TStatus {
        TStatus::new(TStatusCode::INTERNAL_ERROR, Some(vec![message]))
    }
}

/// Reasons a FE heartbeat can be rejected.
#[derive(Debug, thiserror::Error)]
enum HeartbeatError {
    #[error(
        "FE heartbeat did not include a node type; Sirius only supports StarRocks compute nodes"
    )]
    MissingNodeType,
    #[error(
        "FE heartbeat targeted this node as {received:?}; Sirius only supports StarRocks compute nodes"
    )]
    UnexpectedNodeType {
        /// Node type supplied by FE.
        received: types::TNodeType,
    },
    #[error("heartbeat state mutex poisoned")]
    StatePoisoned,
    #[error("stale FE epoch {received}, current epoch is {current}")]
    StaleEpoch {
        /// FE epoch from the rejected heartbeat.
        received: types::TEpoch,
        /// Highest accepted FE epoch.
        current: types::TEpoch,
    },
    #[error("cluster id changed from {current} to {received}")]
    ClusterChanged {
        /// Cluster id supplied by the rejected heartbeat.
        received: types::TClusterId,
        /// Cluster id accepted from previous heartbeats.
        current: types::TClusterId,
    },
    #[error("FE token changed")]
    TokenChanged,
}

impl HeartbeatServiceSyncHandler for ComputeNodeHeartbeatHandler {
    fn handle_heartbeat(&self, master_info: TMasterInfo) -> thrift::Result<THeartbeatResult> {
        debug!(
            fe_host = %master_info.network_address.hostname,
            fe_port = master_info.network_address.port,
            epoch = master_info.epoch,
            compute_node_id = ?master_info.backend_id,
            "received FE heartbeat"
        );

        let status = match self.handle_master_info(&master_info) {
            Ok(()) => Self::ok_status(),
            Err(err) => {
                warn!(error = %err, "rejecting FE heartbeat");
                Self::error_status(err.to_string())
            }
        };

        Ok(THeartbeatResult::new(status, self.compute_node_info()))
    }
}

/// Handle for the blocking thrift heartbeat listener thread.
pub struct HeartbeatServer {
    /// Dedicated thread running the blocking thrift heartbeat server.
    join_handle: Option<JoinHandle<Result<()>>>,
    /// Shutdown handle used to wake the listener and close an active socket.
    shutdown: HeartbeatServerShutdown,
    /// Address selected by the operating system after binding.
    local_addr: SocketAddr,
}

impl HeartbeatServer {
    /// Starts the blocking thrift heartbeat server on a dedicated thread.
    pub fn start(config: ComputeNodeConfig, state: SharedHeartbeatState) -> Result<Self> {
        let listen_addr = format!("{}:{}", config.bind_host, config.heartbeat_port);
        let listener = TcpListener::bind(&listen_addr)
            .with_context(|| format!("failed to bind heartbeat Thrift server at {listen_addr}"))?;
        let local_addr = listener
            .local_addr()
            .context("failed to read heartbeat Thrift server address")?;
        let shutdown = HeartbeatServerShutdown::new(Self::listener_wake_addr(local_addr));
        let server_shutdown = shutdown.clone();

        info!(address = %local_addr, "starting heartbeat Thrift server");
        let join_handle =
            thread::spawn(move || Self::run(listener, config, state, server_shutdown));

        Ok(Self {
            join_handle: Some(join_handle),
            shutdown,
            local_addr,
        })
    }

    /// Returns a cloneable shutdown handle for coordinating process shutdown.
    pub fn shutdown_handle(&self) -> HeartbeatServerShutdown {
        self.shutdown.clone()
    }

    /// Requests listener and active-connection shutdown.
    pub fn shutdown(&self) {
        self.shutdown.shutdown();
    }

    /// Returns the address the heartbeat listener bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Waits for the heartbeat listener thread to exit.
    pub fn join(mut self) -> Result<()> {
        let join_handle = self
            .join_handle
            .take()
            .context("heartbeat server join handle was already consumed")?;
        Self::join_thread(join_handle)
    }

    /// Runs the blocking thrift accept loop until shutdown is requested.
    fn run(
        listener: TcpListener,
        config: ComputeNodeConfig,
        state: SharedHeartbeatState,
        shutdown: HeartbeatServerShutdown,
    ) -> Result<()> {
        let processor = Arc::new(HeartbeatServiceSyncProcessor::new(
            ComputeNodeHeartbeatHandler::new(config, state),
        ));

        for stream in listener.incoming() {
            if shutdown.is_requested() {
                break;
            }

            match stream {
                Ok(stream) => {
                    if shutdown.is_requested() {
                        break;
                    }
                    let active_connection = shutdown.track_connection(&stream)?;
                    Self::handle_connection(processor.clone(), stream, active_connection)?;
                    if shutdown.is_requested() {
                        break;
                    }
                }
                Err(_) if shutdown.is_requested() => break,
                Err(err) => warn!(error = %err, "failed to accept heartbeat connection"),
            }
        }

        shutdown.close_active_connection();
        info!("heartbeat Thrift server stopped");
        Ok(())
    }

    /// Processes thrift heartbeat messages on one accepted TCP connection.
    fn handle_connection(
        processor: Arc<HeartbeatProcessor>,
        stream: TcpStream,
        active_connection: ActiveConnectionGuard,
    ) -> Result<()> {
        let _active_connection = active_connection;
        let channel = TTcpChannel::with_stream(stream);
        let (read_channel, write_channel) = channel
            .split()
            .map_err(|err| anyhow!("failed to split heartbeat connection: {err}"))?;
        let read_transport = TBufferedReadTransportFactory::new().create(Box::new(read_channel));
        let write_transport = TBufferedWriteTransportFactory::new().create(Box::new(write_channel));
        let mut input_protocol = TBinaryInputProtocolFactory::new().create(read_transport);
        let mut output_protocol = TBinaryOutputProtocolFactory::new().create(write_transport);

        loop {
            match processor.process(&mut *input_protocol, &mut *output_protocol) {
                Ok(()) => {}
                Err(thrift::Error::Transport(err)) if err.kind == TransportErrorKind::EndOfFile => {
                    return Ok(());
                }
                Err(err) => {
                    warn!(error = %err, "heartbeat processor completed with error");
                    return Ok(());
                }
            }
        }
    }

    /// Joins the heartbeat server thread and converts a panic into an error.
    fn join_thread(join_handle: JoinHandle<Result<()>>) -> Result<()> {
        join_handle
            .join()
            .map_err(|panic| anyhow!("heartbeat server thread panicked: {panic:?}"))?
    }

    /// Returns a loopback address that can wake the listener from another thread.
    fn listener_wake_addr(local_addr: SocketAddr) -> SocketAddr {
        match local_addr.ip() {
            IpAddr::V4(ip) if ip.is_unspecified() => {
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), local_addr.port())
            }
            IpAddr::V6(ip) if ip.is_unspecified() => {
                SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), local_addr.port())
            }
            _ => local_addr,
        }
    }
}

impl Drop for HeartbeatServer {
    fn drop(&mut self) {
        if self.join_handle.is_some() {
            self.shutdown();
        }
    }
}

/// Cloneable signal used to stop the heartbeat listener and active connection.
#[derive(Clone)]
pub struct HeartbeatServerShutdown {
    /// Shared shutdown state for the blocking heartbeat listener.
    inner: Arc<HeartbeatServerShutdownState>,
}

impl HeartbeatServerShutdown {
    fn new(wake_addr: SocketAddr) -> Self {
        Self {
            inner: Arc::new(HeartbeatServerShutdownState {
                requested: AtomicBool::new(false),
                active_connection: Arc::new(Mutex::new(None)),
                wake_addr,
            }),
        }
    }

    /// Requests shutdown for the listener and any active connection.
    pub fn shutdown(&self) {
        if self.inner.requested.swap(true, Ordering::SeqCst) {
            return;
        }

        self.close_active_connection();
        let _ = TcpStream::connect(self.inner.wake_addr);
    }

    fn is_requested(&self) -> bool {
        self.inner.requested.load(Ordering::SeqCst)
    }

    fn track_connection(&self, stream: &TcpStream) -> Result<ActiveConnectionGuard> {
        let shutdown_stream = stream
            .try_clone()
            .context("failed to clone heartbeat client connection")?;

        let mut active_connection = self
            .inner
            .active_connection
            .lock()
            .map_err(|_| anyhow!("active heartbeat connection mutex poisoned"))?;
        *active_connection = Some(shutdown_stream);

        Ok(ActiveConnectionGuard {
            active_connection: self.inner.active_connection.clone(),
        })
    }

    fn close_active_connection(&self) {
        if let Ok(active_connection) = self.inner.active_connection.lock()
            && let Some(connection) = active_connection.as_ref()
        {
            let _ = connection.shutdown(Shutdown::Both);
        }
    }
}

/// Shared shutdown state observed by the blocking heartbeat listener.
struct HeartbeatServerShutdownState {
    /// True after shutdown has been requested.
    requested: AtomicBool,
    /// Clone of the currently accepted socket, used to interrupt blocking reads.
    active_connection: Arc<Mutex<Option<TcpStream>>>,
    /// Loopback address used to wake `TcpListener::incoming`.
    wake_addr: SocketAddr,
}

/// Guard that clears the tracked heartbeat connection when processing ends.
struct ActiveConnectionGuard {
    /// Shared slot cleared when connection handling returns.
    active_connection: Arc<Mutex<Option<TcpStream>>>,
}

impl Drop for ActiveConnectionGuard {
    fn drop(&mut self) {
        if let Ok(mut active_connection) = self.active_connection.lock() {
            *active_connection = None;
        }
    }
}

/// Starts the StarRocks thrift heartbeat server.
pub fn start_heartbeat_server(
    config: ComputeNodeConfig,
    state: SharedHeartbeatState,
) -> Result<HeartbeatServer> {
    HeartbeatServer::start(config, state)
}

/// Registers the compute node with StarRocks FE through the MySQL protocol.
pub async fn register_node(fe: &FeConfig, node: &ComputeNodeConfig) -> Result<()> {
    fe.register_compute_node(node).await
}

/// Thin wall-clock wrapper for associated time helpers.
struct WallClock;

impl WallClock {
    /// Returns the current Unix timestamp in whole seconds.
    fn unix_time_secs() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .min(i64::MAX as u64) as i64
    }

    /// Returns the current Unix timestamp in milliseconds.
    fn unix_time_millis() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a stable compute-node config for heartbeat handler tests.
    fn test_config() -> ComputeNodeConfig {
        ComputeNodeConfig {
            version: "test-version".to_string(),
            ..ComputeNodeConfig::default()
        }
    }

    /// Builds a heartbeat handler with its shared state so tests can inspect side effects.
    fn handler() -> (ComputeNodeHeartbeatHandler, SharedHeartbeatState) {
        let state = SharedHeartbeatState::new();
        (
            ComputeNodeHeartbeatHandler::new(test_config(), state.clone()),
            state,
        )
    }

    /// Builds a valid FE master heartbeat for the requested epoch.
    fn master(epoch: types::TEpoch) -> TMasterInfo {
        TMasterInfo::new(
            types::TNetworkAddress::new("127.0.0.1".to_string(), 9020),
            Some(42),
            epoch,
            Some("token".to_string()),
            Some("127.0.0.1".to_string()),
            Some(8030),
            Some(0),
            Some(10001),
            Some(0),
            Some(types::TRunMode::SHARED_DATA),
            None,
            None,
            Some(false),
            Some(true),
            Some(types::TNodeType::COMPUTE),
        )
    }

    /// Verifies the first valid heartbeat records cluster identity and node metadata.
    #[test]
    fn first_heartbeat_succeeds_and_records_state() {
        let (handler, state) = handler();

        let result = handler.handle_heartbeat(master(7)).unwrap();

        assert_eq!(result.status.status_code, TStatusCode::OK);
        assert_eq!(result.backend_info.be_port, 9060);
        assert_eq!(result.backend_info.http_port, 8040);
        assert_eq!(result.backend_info.brpc_port, Some(8060));
        assert_eq!(result.backend_info.arrow_flight_port, Some(-1));
        assert_eq!(result.backend_info.is_set_storage_path, Some(false));
        let snapshot = state.snapshot();
        assert_eq!(snapshot.cluster_id, Some(42));
        assert_eq!(
            snapshot.token.as_ref().map(SecretString::expose_secret),
            Some("token")
        );
        assert_eq!(snapshot.epoch, Some(7));
        assert_eq!(snapshot.compute_node_id, Some(10001));
        assert!(snapshot.last_heartbeat_ms.is_some());
    }

    /// Verifies equal or increasing heartbeat epochs keep the handler state current.
    #[test]
    fn repeated_same_or_higher_epoch_succeeds() {
        let (handler, state) = handler();

        assert_eq!(
            handler
                .handle_heartbeat(master(7))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );
        assert_eq!(
            handler
                .handle_heartbeat(master(7))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );
        assert_eq!(
            handler
                .handle_heartbeat(master(8))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );
        assert_eq!(state.snapshot().epoch, Some(8));
    }

    /// Verifies a lower epoch is rejected and does not roll back recorded state.
    #[test]
    fn stale_epoch_fails() {
        let (handler, state) = handler();

        assert_eq!(
            handler
                .handle_heartbeat(master(7))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );
        assert_eq!(
            handler
                .handle_heartbeat(master(6))
                .unwrap()
                .status
                .status_code,
            TStatusCode::INTERNAL_ERROR
        );
        assert_eq!(state.snapshot().epoch, Some(7));
    }

    /// Verifies the FE token is sticky once learned from the first valid heartbeat.
    #[test]
    fn token_mismatch_fails() {
        let (handler, state) = handler();
        let mut changed = master(8);
        changed.token = Some("different-token".to_string());

        assert_eq!(
            handler
                .handle_heartbeat(master(7))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );
        assert_eq!(
            handler
                .handle_heartbeat(changed)
                .unwrap()
                .status
                .status_code,
            TStatusCode::INTERNAL_ERROR
        );
        assert_eq!(
            state
                .snapshot()
                .token
                .as_ref()
                .map(SecretString::expose_secret),
            Some("token")
        );
    }

    /// Verifies the cluster id is sticky once learned from the first valid heartbeat.
    #[test]
    fn cluster_mismatch_fails() {
        let (handler, state) = handler();
        let mut changed = master(8);
        changed.cluster_id = Some(43);

        assert_eq!(
            handler
                .handle_heartbeat(master(7))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );
        assert_eq!(
            handler
                .handle_heartbeat(changed)
                .unwrap()
                .status
                .status_code,
            TStatusCode::INTERNAL_ERROR
        );
        assert_eq!(state.snapshot().cluster_id, Some(42));
    }

    /// Verifies backend heartbeats are rejected by the compute-node heartbeat handler.
    #[test]
    fn non_compute_node_heartbeat_fails() {
        let (handler, state) = handler();
        let mut heartbeat = master(7);
        heartbeat.node_type = Some(types::TNodeType::BACKEND);

        assert_eq!(
            handler
                .handle_heartbeat(heartbeat)
                .unwrap()
                .status
                .status_code,
            TStatusCode::INTERNAL_ERROR
        );
        assert_eq!(state.snapshot().epoch, None);
    }

    /// Verifies missing node-type metadata is rejected instead of being treated as compute.
    #[test]
    fn missing_node_type_fails() {
        let (handler, state) = handler();
        let mut heartbeat = master(7);
        heartbeat.node_type = None;

        assert_eq!(
            handler
                .handle_heartbeat(heartbeat)
                .unwrap()
                .status
                .status_code,
            TStatusCode::INTERNAL_ERROR
        );
        assert_eq!(state.snapshot().epoch, None);
    }

    /// Verifies FE registration uses StarRocks compute-node syntax and proc path.
    #[test]
    fn registration_uses_compute_node_surface() {
        let config = ComputeNodeConfig {
            advertise_host: Host::local(),
            heartbeat_port: 9050,
            ..ComputeNodeConfig::default()
        };

        assert_eq!(
            config.registration_sql(),
            "ALTER SYSTEM ADD COMPUTE NODE \"127.0.0.1:9050\""
        );
        assert_eq!(COMPUTE_NODE_PROC_PATH, "/compute_nodes");
    }

    /// Verifies secrets stay redacted in debug output.
    #[test]
    fn secret_string_debug_redacts_value() {
        assert_eq!(
            format!("{:?}", SecretString::new("do-not-log")),
            "SecretString(<redacted>)"
        );
    }

    /// Verifies the heartbeat server shutdown handle wakes and stops the accept loop.
    #[test]
    fn heartbeat_server_shutdown_stops_accept_loop() {
        let mut config = test_config();
        config.bind_host = Host::local();
        config.heartbeat_port = 0;
        let server = match start_heartbeat_server(config, SharedHeartbeatState::new()) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let stream = TcpStream::connect(server.local_addr()).unwrap();

        server.shutdown();
        server.join().unwrap();
        drop(stream);
    }

    /// Detects sandboxed environments where binding a local listener is denied.
    fn is_permission_denied(err: &anyhow::Error) -> bool {
        err.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|err| err.kind() == std::io::ErrorKind::PermissionDenied)
        })
    }
}
