//! Node identity: FE registration, the thrift `HeartbeatService` that keeps this backend
//! `Alive` in `SHOW BACKENDS`, and the thrift `BackendService` skeleton the FE probes
//! periodically.
//!
//! Doris keeps a backend alive purely on heartbeat success (`Backend.handleHbResponse`):
//! every ~5 s the FE leader calls `HeartbeatService.heartbeat(TMasterInfo)` on
//! `heartbeat_port` and expects `THeartbeatResult{status: OK, backend_info}` back. One
//! failure marks the node dead. The `BackendService` calls (`publish_topic_info`,
//! `get_dictionary_status`, `get_tablet_stat`, ...) are periodic and only warn when they
//! fail, so a truthful empty/NOT_IMPLEMENTED answer is enough. Both services are plain
//! buffered `TBinaryProtocol` over TCP (the BE serves them with `THREADED`/`THREAD_POOL`
//! thrift servers, never framed).

use std::{
    collections::BTreeMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use doris_thrift::{
    agent_service, backend_service,
    backend_service::{BackendServiceSyncHandler, BackendServiceSyncProcessor},
    doris_external_service,
    heartbeat_service::{
        HeartbeatServiceSyncHandler, HeartbeatServiceSyncProcessor, TBackendInfo, THeartbeatResult,
        TMasterInfo,
    },
    status::TStatus,
    types,
};
use mysql_async::{OptsBuilder, Pool, Row, prelude::Queryable};
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
use tracing::{debug, info, instrument, warn};

use crate::{
    BackendConfig, FeConfig, Host, SecretString, error_status, not_implemented_status, ok_status,
};

/// `be_node_role` advertised in heartbeats. Must be `mix`: the FE excludes `computation`
/// nodes from TVF scheduling unless `prefer_compute_node_for_external_table` is set.
const BE_NODE_ROLE_MIX: &str = "mix";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Snapshot of FE-provided heartbeat identity.
pub struct HeartbeatStateSnapshot {
    /// Sticky Doris cluster id learned from the first accepted FE heartbeat.
    pub cluster_id: Option<types::TClusterId>,
    /// Sticky FE token learned from the first heartbeat that carried one.
    pub token: Option<SecretString>,
    /// Latest accepted FE epoch (bumped on FE leader change); older epochs are rejected.
    pub epoch: Option<types::TEpoch>,
    /// FE-assigned backend id.
    pub backend_id: Option<i64>,
    /// Host the FE registered this backend under (`TMasterInfo.backend_ip`).
    pub registered_host: Option<String>,
    /// FE leader's thrift address (`TMasterInfo.network_address`).
    pub frontend_address: Option<types::TNetworkAddress>,
    /// Wall-clock timestamp of the latest accepted heartbeat.
    pub last_heartbeat_ms: Option<u128>,
}

#[derive(Debug, Default)]
struct HeartbeatState {
    cluster_id: Option<types::TClusterId>,
    token: Option<SecretString>,
    epoch: Option<types::TEpoch>,
    backend_id: Option<i64>,
    registered_host: Option<String>,
    frontend_address: Option<types::TNetworkAddress>,
    last_heartbeat_ms: Option<u128>,
}

#[derive(Clone, Debug)]
pub struct SharedHeartbeatState(Arc<Mutex<HeartbeatState>>);

impl SharedHeartbeatState {
    /// Creates empty heartbeat state.
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HeartbeatState::default())))
    }

    /// Returns a copy of heartbeat state for tests and monitoring decisions.
    pub fn snapshot(&self) -> HeartbeatStateSnapshot {
        let state = self.0.lock().expect("heartbeat state mutex poisoned");
        HeartbeatStateSnapshot {
            cluster_id: state.cluster_id,
            token: state.token.clone(),
            epoch: state.epoch,
            backend_id: state.backend_id,
            registered_host: state.registered_host.clone(),
            frontend_address: state.frontend_address.clone(),
            last_heartbeat_ms: state.last_heartbeat_ms,
        }
    }

    /// Returns how long ago the last accepted heartbeat arrived, if any.
    pub fn last_heartbeat_elapsed(&self) -> Option<Duration> {
        let state = self.0.lock().expect("heartbeat state mutex poisoned");
        let last_heartbeat_ms = state.last_heartbeat_ms?;
        let elapsed_ms = unix_time_millis().saturating_sub(last_heartbeat_ms);
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

#[derive(Clone, Debug)]
pub struct BackendHeartbeatHandler {
    config: BackendConfig,
    state: SharedHeartbeatState,
    /// Process start time in milliseconds; the FE uses it as the BE epoch (`be_start_time`)
    /// to tell a restarted process from a paused one.
    be_start_time_ms: i64,
}

impl BackendHeartbeatHandler {
    pub fn new(config: BackendConfig, state: SharedHeartbeatState) -> Self {
        Self {
            config,
            state,
            be_start_time_ms: unix_time_millis().min(i64::MAX as u128) as i64,
        }
    }

    fn handle_master_info(
        &self,
        master_info: &TMasterInfo,
    ) -> std::result::Result<(), HeartbeatError> {
        let mut state = self
            .state
            .0
            .lock()
            .map_err(|_| HeartbeatError::StatePoisoned)?;

        // FE epochs are monotonic (bumped on leader election); accepting an older epoch would
        // let a demoted leader roll our view of the cluster back.
        if let Some(epoch) = state.epoch
            && master_info.epoch < epoch
        {
            return Err(HeartbeatError::StaleEpoch {
                received: master_info.epoch,
                current: epoch,
            });
        }

        // Cluster id is learned once and then treated as immutable for this process (a real
        // BE persists it under its storage root and refuses to join another cluster).
        if let Some(current_cluster_id) = state.cluster_id {
            if master_info.cluster_id != current_cluster_id {
                return Err(HeartbeatError::ClusterChanged {
                    received: master_info.cluster_id,
                    current: current_cluster_id,
                });
            }
        } else {
            state.cluster_id = Some(master_info.cluster_id);
        }

        // The FE token is sticky once learned; a change means a different FE is talking to us.
        if let Some(token) = &master_info.token {
            if let Some(current_token) = &state.token {
                if token != current_token.expose_secret() {
                    return Err(HeartbeatError::TokenChanged);
                }
            } else {
                state.token = Some(SecretString::new(token.clone()));
            }
        }

        // `backend_ip` is the host the FE registered us under. A real BE rejects a mismatch
        // against its local interfaces; we only require it to match what we advertise, since
        // the FE will dial that host for every later RPC.
        if let Some(registered_host) = &master_info.backend_ip {
            if registered_host != self.config.advertise_host.as_str() {
                return Err(HeartbeatError::HostMismatch {
                    registered: registered_host.clone(),
                    advertised: self.config.advertise_host.to_string(),
                });
            }
            state.registered_host = Some(registered_host.clone());
        }

        state.epoch = Some(master_info.epoch);
        if let Some(backend_id) = master_info.backend_id {
            state.backend_id = Some(backend_id);
        }
        state.frontend_address = Some(master_info.network_address.clone());
        state.last_heartbeat_ms = Some(unix_time_millis());

        Ok(())
    }

    fn backend_info(&self) -> TBackendInfo {
        // Named here because TBackendInfo::new takes many same-typed positional
        // Option<TPort>/Option<i64> args (see HeartbeatService.thrift `struct TBackendInfo`).
        const BE_RPC_PORT_DISABLED: i32 = -1; // legacy thrift RPC port; unused
        const ARROW_FLIGHT_SQL_PORT_DISABLED: i32 = -1; // no Arrow Flight SQL endpoint
        TBackendInfo::new(
            i32::from(self.config.be_port),         // be_port
            i32::from(self.config.http_port),       // http_port
            Some(BE_RPC_PORT_DISABLED),             // be_rpc_port
            Some(i32::from(self.config.brpc_port)), // brpc_port
            Some(self.config.version.clone()),      // version
            Some(self.be_start_time_ms),            // be_start_time
            Some(BE_NODE_ROLE_MIX.to_string()),     // be_node_role
            Some(false),                            // is_shutdown
            Some(ARROW_FLIGHT_SQL_PORT_DISABLED),   // arrow_flight_sql_port
            None,                                   // be_mem (unreported)
            Some(0),                                // fragment_executing_count
            Some(0),                                // fragment_last_active_time
        )
    }
}

#[derive(Debug, thiserror::Error)]
enum HeartbeatError {
    #[error("heartbeat state mutex poisoned")]
    StatePoisoned,
    #[error("stale FE epoch {received}, current epoch is {current}")]
    StaleEpoch {
        received: types::TEpoch,
        current: types::TEpoch,
    },
    #[error("cluster id changed from {current} to {received}")]
    ClusterChanged {
        received: types::TClusterId,
        current: types::TClusterId,
    },
    #[error("FE token changed")]
    TokenChanged,
    #[error(
        "FE registered this backend as host {registered} but it advertises {advertised}; \
         re-register it with --advertise-host {registered} or drop the stale backend"
    )]
    HostMismatch {
        registered: String,
        advertised: String,
    },
}

impl HeartbeatServiceSyncHandler for BackendHeartbeatHandler {
    fn handle_heartbeat(&self, master_info: TMasterInfo) -> thrift::Result<THeartbeatResult> {
        debug!(
            fe_host = %master_info.network_address.hostname,
            fe_port = master_info.network_address.port,
            epoch = master_info.epoch,
            backend_id = ?master_info.backend_id,
            "received FE heartbeat"
        );

        let status = match self.handle_master_info(&master_info) {
            Ok(()) => ok_status(),
            Err(err) => {
                warn!(error = %err, "rejecting FE heartbeat");
                error_status(err.to_string())
            }
        };

        Ok(THeartbeatResult::new(status, self.backend_info()))
    }
}

#[derive(Clone, Debug, Default)]
/// Doris `BackendService` skeleton for the FE's periodic probes on `be_port`.
///
/// Nothing here carries state: this backend has no tablets, no loads, no caches. Probes that
/// expect an inventory get a truthful empty one; RPCs that would mutate storage or start
/// work get `NOT_IMPLEMENTED_ERROR` naming the RPC.
pub struct BackendServiceHandler;

impl BackendServiceHandler {
    pub fn new() -> Self {
        Self
    }
}

impl BackendServiceSyncHandler for BackendServiceHandler {
    // Agent tasks mutate backend state/storage, so they are rejected explicitly.
    fn handle_submit_tasks(
        &self,
        _tasks: Vec<agent_service::TAgentTaskRequest>,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.submit_tasks"))
    }

    fn handle_make_snapshot(
        &self,
        _snapshot_request: agent_service::TSnapshotRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.make_snapshot"))
    }

    fn handle_release_snapshot(
        &self,
        _snapshot_path: String,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.release_snapshot"))
    }

    fn handle_publish_cluster_state(
        &self,
        _request: agent_service::TAgentPublishRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.publish_cluster_state"))
    }

    // Empty tablet stats are truthful for a storage-less backend (polled every 60 s).
    fn handle_get_tablet_stat(&self) -> thrift::Result<backend_service::TTabletStatResult> {
        Ok(backend_service::TTabletStatResult::new(
            BTreeMap::new(),
            Some(Vec::new()),
        ))
    }

    fn handle_get_trash_used_capacity(&self) -> thrift::Result<i64> {
        Ok(0)
    }

    fn handle_get_disk_trash_used_capacity(
        &self,
    ) -> thrift::Result<Vec<backend_service::TDiskTrashInfo>> {
        Ok(Vec::new())
    }

    fn handle_submit_routine_load_task(
        &self,
        _tasks: Vec<backend_service::TRoutineLoadTask>,
    ) -> thrift::Result<TStatus> {
        Ok(not_implemented_status(
            "BackendService.submit_routine_load_task",
        ))
    }

    // The external scanner (Doris-as-a-source for Spark/Flink) is an execution path we do not
    // serve.
    fn handle_open_scanner(
        &self,
        _params: doris_external_service::TScanOpenParams,
    ) -> thrift::Result<doris_external_service::TScanOpenResult> {
        Ok(doris_external_service::TScanOpenResult::new(
            not_implemented_status("BackendService.open_scanner"),
            None,
            None,
        ))
    }

    fn handle_get_next(
        &self,
        _params: doris_external_service::TScanNextBatchParams,
    ) -> thrift::Result<doris_external_service::TScanBatchResult> {
        Ok(doris_external_service::TScanBatchResult::new(
            not_implemented_status("BackendService.get_next"),
            Some(true),
            None,
        ))
    }

    fn handle_close_scanner(
        &self,
        _params: doris_external_service::TScanCloseParams,
    ) -> thrift::Result<doris_external_service::TScanCloseResult> {
        Ok(doris_external_service::TScanCloseResult::new(
            not_implemented_status("BackendService.close_scanner"),
        ))
    }

    // Polled every 120 s; no stream loads ever happen here.
    fn handle_get_stream_load_record(
        &self,
        _last_stream_record_time: i64,
    ) -> thrift::Result<backend_service::TStreamLoadRecordResult> {
        Ok(backend_service::TStreamLoadRecordResult::new(
            BTreeMap::new(),
        ))
    }

    fn handle_check_storage_format(
        &self,
    ) -> thrift::Result<backend_service::TCheckStorageFormatResult> {
        Ok(backend_service::TCheckStorageFormatResult::new(
            Some(Vec::new()),
            Some(Vec::new()),
        ))
    }

    fn handle_warm_up_cache_async(
        &self,
        _request: backend_service::TWarmUpCacheAsyncRequest,
    ) -> thrift::Result<backend_service::TWarmUpCacheAsyncResponse> {
        Ok(backend_service::TWarmUpCacheAsyncResponse::new(
            not_implemented_status("BackendService.warm_up_cache_async"),
        ))
    }

    fn handle_check_warm_up_cache_async(
        &self,
        _request: backend_service::TCheckWarmUpCacheAsyncRequest,
    ) -> thrift::Result<backend_service::TCheckWarmUpCacheAsyncResponse> {
        Ok(backend_service::TCheckWarmUpCacheAsyncResponse::new(
            not_implemented_status("BackendService.check_warm_up_cache_async"),
            None,
        ))
    }

    fn handle_sync_load_for_tablets(
        &self,
        _request: backend_service::TSyncLoadForTabletsRequest,
    ) -> thrift::Result<backend_service::TSyncLoadForTabletsResponse> {
        Ok(backend_service::TSyncLoadForTabletsResponse::new())
    }

    fn handle_get_top_n_hot_partitions(
        &self,
        _request: backend_service::TGetTopNHotPartitionsRequest,
    ) -> thrift::Result<backend_service::TGetTopNHotPartitionsResponse> {
        Ok(backend_service::TGetTopNHotPartitionsResponse::new(
            0,
            Some(Vec::new()),
        ))
    }

    fn handle_warm_up_tablets(
        &self,
        _request: backend_service::TWarmUpTabletsRequest,
    ) -> thrift::Result<backend_service::TWarmUpTabletsResponse> {
        Ok(backend_service::TWarmUpTabletsResponse::new(
            not_implemented_status("BackendService.warm_up_tablets"),
            None,
            None,
            None,
            None,
        ))
    }

    fn handle_ingest_binlog(
        &self,
        _request: backend_service::TIngestBinlogRequest,
    ) -> thrift::Result<backend_service::TIngestBinlogResult> {
        Ok(backend_service::TIngestBinlogResult::new(
            Some(not_implemented_status("BackendService.ingest_binlog")),
            None,
        ))
    }

    fn handle_query_ingest_binlog(
        &self,
        _request: backend_service::TQueryIngestBinlogRequest,
    ) -> thrift::Result<backend_service::TQueryIngestBinlogResult> {
        Ok(backend_service::TQueryIngestBinlogResult::new(
            Some(backend_service::TIngestBinlogStatus::UNKNOWN),
            Some("BackendService.query_ingest_binlog is not implemented".to_string()),
        ))
    }

    // The FE pushes workload-group topics to every backend (every 30 s); acknowledging keeps
    // the publisher quiet. Nothing here consumes them.
    fn handle_publish_topic_info(
        &self,
        _topic_request: backend_service::TPublishTopicRequest,
    ) -> thrift::Result<backend_service::TPublishTopicResult> {
        Ok(backend_service::TPublishTopicResult::new(ok_status()))
    }

    // Real-time query status for SHOW PROCESSLIST / profile; no running fragment to report.
    fn handle_get_realtime_exec_status(
        &self,
        _request: backend_service::TGetRealtimeExecStatusRequest,
    ) -> thrift::Result<backend_service::TGetRealtimeExecStatusResponse> {
        Ok(backend_service::TGetRealtimeExecStatusResponse::new(
            Some(not_implemented_status(
                "BackendService.get_realtime_exec_status",
            )),
            None,
            None,
        ))
    }

    // Polled every 5 s for the dictionary feature; none exist here.
    fn handle_get_dictionary_status(
        &self,
        _dictionary_ids: Vec<i64>,
    ) -> thrift::Result<backend_service::TDictionaryStatusList> {
        Ok(backend_service::TDictionaryStatusList::new(
            Some(Vec::new()),
        ))
    }

    fn handle_test_storage_connectivity(
        &self,
        _request: backend_service::TTestStorageConnectivityRequest,
    ) -> thrift::Result<backend_service::TTestStorageConnectivityResponse> {
        Ok(backend_service::TTestStorageConnectivityResponse::new(
            Some(not_implemented_status(
                "BackendService.test_storage_connectivity",
            )),
        ))
    }

    fn handle_get_python_envs(&self) -> thrift::Result<Vec<backend_service::TPythonEnvInfo>> {
        Ok(Vec::new())
    }

    fn handle_get_python_packages(
        &self,
        _python_version: String,
    ) -> thrift::Result<Vec<backend_service::TPythonPackageInfo>> {
        Ok(Vec::new())
    }
}

/// Joinable handle for a blocking thrift server thread.
pub struct ThriftServer {
    // Owning the join handle lets shutdown wait for the accept loop to exit.
    join_handle: Option<JoinHandle<Result<()>>>,
    // Shared shutdown state used by signal handlers and Drop.
    shutdown: ThriftServerShutdown,
    // Bound address, useful for tests that bind port zero.
    local_addr: SocketAddr,
    // Static service label used in logs and join errors.
    name: &'static str,
}

/// Heartbeat service server handle.
pub type HeartbeatServer = ThriftServer;
/// Backend service server handle.
pub type BackendServer = ThriftServer;

impl ThriftServer {
    /// Clones the shutdown handle so async orchestration can stop the blocking thread.
    pub fn shutdown_handle(&self) -> ThriftServerShutdown {
        self.shutdown.clone()
    }

    /// Requests shutdown and wakes the blocking accept loop.
    pub fn shutdown(&self) {
        self.shutdown.shutdown();
    }

    /// Returns the actual listen address, including the OS-selected port for port zero.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Waits for the server thread and reports any thread/server error.
    pub fn join(mut self) -> Result<()> {
        let join_handle = self
            .join_handle
            .take()
            .with_context(|| format!("{} server join handle was already consumed", self.name))?;
        join_thrift_thread(self.name, join_handle)
    }
}

impl Drop for ThriftServer {
    fn drop(&mut self) {
        if self.join_handle.is_some() {
            // Dropping an unjoined server should not leave the blocking accept loop alive.
            self.shutdown();
        }
    }
}

#[derive(Clone)]
/// Shared shutdown state for a blocking thrift server.
pub struct ThriftServerShutdown(Arc<ThriftServerShutdownState>);

impl ThriftServerShutdown {
    /// Creates shutdown state and records the address used to wake `TcpListener::incoming`.
    fn new(wake_addr: SocketAddr) -> Self {
        Self(Arc::new(ThriftServerShutdownState {
            requested: AtomicBool::new(false),
            active_connection: Arc::new(Mutex::new(None)),
            wake_addr,
        }))
    }

    /// Marks shutdown requested, closes any in-flight client, and wakes the listener.
    pub fn shutdown(&self) {
        if self.0.requested.swap(true, Ordering::SeqCst) {
            return;
        }

        self.close_active_connection();
        let _ = TcpStream::connect(self.0.wake_addr);
    }

    /// Returns whether shutdown has been requested.
    fn is_requested(&self) -> bool {
        self.0.requested.load(Ordering::SeqCst)
    }

    /// Tracks the current connection so shutdown can interrupt a blocking thrift read.
    fn track_connection(&self, stream: &TcpStream) -> Result<ActiveConnectionGuard> {
        let shutdown_stream = stream
            .try_clone()
            .context("failed to clone thrift client connection")?;

        let mut active_connection = self
            .0
            .active_connection
            .lock()
            .map_err(|_| anyhow!("active thrift connection mutex poisoned"))?;
        *active_connection = Some(shutdown_stream);

        // Close the race with shutdown(): if shutdown was requested between the accept loop's
        // is_requested() check and this store, close_active_connection() may already have run
        // on an empty slot and will not run again. Re-checking under the same lock guarantees
        // this connection is interrupted rather than blocking the processor forever.
        if self.0.requested.load(Ordering::SeqCst)
            && let Some(connection) = active_connection.as_ref()
        {
            let _ = connection.shutdown(Shutdown::Both);
        }

        Ok(ActiveConnectionGuard {
            active_connection: self.0.active_connection.clone(),
        })
    }

    /// Closes the active client connection if the processor is blocked waiting for input.
    fn close_active_connection(&self) {
        if let Ok(active_connection) = self.0.active_connection.lock()
            && let Some(connection) = active_connection.as_ref()
        {
            let _ = connection.shutdown(Shutdown::Both);
        }
    }
}

struct ThriftServerShutdownState {
    // Atomic flag lets the listener thread observe shutdown without locking.
    requested: AtomicBool,
    // The current thrift connection is closed to unblock processor reads on shutdown.
    active_connection: Arc<Mutex<Option<TcpStream>>>,
    // A loopback connection to this address wakes `TcpListener::incoming`.
    wake_addr: SocketAddr,
}

struct ActiveConnectionGuard {
    // Dropping this guard clears the shutdown state's current active connection.
    active_connection: Arc<Mutex<Option<TcpStream>>>,
}

impl Drop for ActiveConnectionGuard {
    fn drop(&mut self) {
        if let Ok(mut active_connection) = self.active_connection.lock() {
            // The processor finished or disconnected, so future shutdowns should not close it.
            *active_connection = None;
        }
    }
}

/// Starts the FE heartbeat thrift service on the configured heartbeat port.
pub fn start_heartbeat_server(
    config: BackendConfig,
    state: SharedHeartbeatState,
) -> Result<HeartbeatServer> {
    let listen_addr = format!("{}:{}", config.bind_host, config.heartbeat_port);
    let listener = TcpListener::bind(&listen_addr)
        .with_context(|| format!("failed to bind heartbeat Thrift server at {listen_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("failed to read heartbeat Thrift server address")?;
    let shutdown = ThriftServerShutdown::new(listener_wake_addr(local_addr));
    let server_shutdown = shutdown.clone();

    info!(address = %local_addr, "starting heartbeat Thrift server");
    // The generated processor owns the handler and is shared across sequential connections.
    let processor = Arc::new(HeartbeatServiceSyncProcessor::new(
        BackendHeartbeatHandler::new(config, state),
    ));
    let join_handle =
        thread::spawn(move || run_thrift_server("heartbeat", listener, processor, server_shutdown));

    Ok(ThriftServer {
        join_handle: Some(join_handle),
        shutdown,
        local_addr,
        name: "heartbeat",
    })
}

/// Starts the Doris `BackendService` thrift skeleton on `--be-port`.
pub fn start_backend_server(config: &BackendConfig) -> Result<BackendServer> {
    let listen_addr = format!("{}:{}", config.bind_host, config.be_port);
    let listener = TcpListener::bind(&listen_addr)
        .with_context(|| format!("failed to bind backend Thrift server at {listen_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("failed to read backend Thrift server address")?;
    let shutdown = ThriftServerShutdown::new(listener_wake_addr(local_addr));
    let server_shutdown = shutdown.clone();

    info!(address = %local_addr, "starting backend Thrift server");
    let processor = Arc::new(BackendServiceSyncProcessor::new(
        BackendServiceHandler::new(),
    ));
    let join_handle =
        thread::spawn(move || run_thrift_server("backend", listener, processor, server_shutdown));

    Ok(ThriftServer {
        join_handle: Some(join_handle),
        shutdown,
        local_addr,
        name: "backend",
    })
}

/// Runs one generated thrift processor behind a blocking TCP listener.
///
/// Connections are served one at a time. The FE keeps one pooled connection per service and
/// issues calls sequentially on it, so this is enough for heartbeats and probes; a second
/// connection waits until the first disconnects.
fn run_thrift_server<P>(
    name: &'static str,
    listener: TcpListener,
    processor: Arc<P>,
    shutdown: ThriftServerShutdown,
) -> Result<()>
where
    P: TProcessor + Send + Sync + 'static,
{
    for stream in listener.incoming() {
        if shutdown.is_requested() {
            break;
        }

        match stream {
            Ok(stream) => {
                if shutdown.is_requested() {
                    break;
                }
                // A transient per-connection error (e.g. fd exhaustion on try_clone/split, or a
                // poisoned mutex) must not tear down the whole server — log and keep accepting.
                let active_connection = match shutdown.track_connection(&stream) {
                    Ok(active_connection) => active_connection,
                    Err(err) => {
                        warn!(service = name, error = %err, "failed to track thrift connection; continuing");
                        continue;
                    }
                };
                if let Err(err) =
                    handle_thrift_connection(name, processor.clone(), stream, active_connection)
                {
                    warn!(service = name, error = %err, "failed to handle thrift connection; continuing");
                }
                if shutdown.is_requested() {
                    break;
                }
            }
            Err(_) if shutdown.is_requested() => break,
            Err(err) => warn!(service = name, error = %err, "failed to accept thrift connection"),
        }
    }

    shutdown.close_active_connection();
    info!(service = name, "Thrift server stopped");
    Ok(())
}

/// Processes one client connection until EOF, thrift error, or server shutdown.
fn handle_thrift_connection<P>(
    name: &'static str,
    processor: Arc<P>,
    stream: TcpStream,
    active_connection: ActiveConnectionGuard,
) -> Result<()>
where
    P: TProcessor + Send + Sync + 'static,
{
    // Keep the guard alive for the full processor loop so shutdown can close this stream.
    let _active_connection = active_connection;
    let channel = TTcpChannel::with_stream(stream);
    let (read_channel, write_channel) = channel
        .split()
        .map_err(|err| anyhow!("failed to split {name} thrift connection: {err}"))?;
    let read_transport = TBufferedReadTransportFactory::new().create(Box::new(read_channel));
    let write_transport = TBufferedWriteTransportFactory::new().create(Box::new(write_channel));
    let mut input_protocol = TBinaryInputProtocolFactory::new().create(read_transport);
    let mut output_protocol = TBinaryOutputProtocolFactory::new().create(write_transport);

    loop {
        match processor.process(&mut *input_protocol, &mut *output_protocol) {
            Ok(()) => {}
            // EOF is a normal client disconnect; the server goes back to accepting.
            Err(thrift::Error::Transport(err)) if err.kind == TransportErrorKind::EndOfFile => {
                return Ok(());
            }
            Err(err) => {
                // The FE may probe unsupported paths; keep the server alive after one error.
                warn!(service = name, error = %err, "thrift processor completed with error");
                return Ok(());
            }
        }
    }
}

/// Converts a blocking thread join into the crate's error type.
fn join_thrift_thread(name: &'static str, join_handle: JoinHandle<Result<()>>) -> Result<()> {
    join_handle
        .join()
        .map_err(|panic| anyhow!("{name} server thread panicked: {panic:?}"))?
}

/// Maps an unspecified bind address to loopback so shutdown can wake the listener locally.
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

fn backend_registration_sql(host: &Host, heartbeat_port: u16) -> String {
    format!("ALTER SYSTEM ADD BACKEND \"{host}:{heartbeat_port}\"")
}

/// Registers this backend with the FE over the MySQL protocol (`ALTER SYSTEM ADD BACKEND`),
/// idempotently: an already-listed `host:heartbeat_port` is left alone.
#[instrument(
    skip_all,
    fields(
        fe_host = %fe.host,
        fe_query_port = fe.query_port,
        advertise_host = %node.advertise_host,
        heartbeat_port = node.heartbeat_port,
    )
)]
pub async fn register_node(fe: &FeConfig, node: &BackendConfig) -> Result<()> {
    let opts = OptsBuilder::default()
        .ip_or_hostname(fe.host.to_string())
        .tcp_port(fe.query_port)
        .prefer_socket(false)
        .user(Some(fe.user.clone()))
        .pass(Some(fe.password.expose_secret().to_string()))
        // Pre-set what the driver would otherwise probe with `SELECT @@max_allowed_packet` /
        // `SELECT @@wait_timeout` on connect: Doris plans those as real queries and fails them
        // with "No backend available as scan node" while no backend is alive yet — which is
        // exactly when this registration runs.
        .max_allowed_packet(Some(16 * 1024 * 1024))
        .wait_timeout(Some(28_800));
    let pool = Pool::new(opts);
    let mut conn = pool
        .get_conn()
        .await
        .with_context(|| format!("failed to connect to FE at {}:{}", fe.host, fe.query_port))?;

    if node_is_registered(&mut conn, node).await? {
        info!(
            host = %node.advertise_host,
            heartbeat_port = node.heartbeat_port,
            "backend is already registered with FE"
        );
        drop(conn);
        pool.disconnect()
            .await
            .context("failed to disconnect FE MySQL pool")?;
        return Ok(());
    }

    let sql = backend_registration_sql(&node.advertise_host, node.heartbeat_port);
    info!(sql = %sql, "registering backend with FE");
    if let Err(err) = conn.query_drop(sql).await {
        warn!(error = %err, "ALTER SYSTEM ADD BACKEND failed; checking whether the backend already exists");
        if !node_is_registered(&mut conn, node).await? {
            return Err(err).context("failed to register backend with FE");
        }
    }

    if !node_is_registered(&mut conn, node).await? {
        bail!(
            "FE accepted registration but backend {}:{} was not found in SHOW BACKENDS",
            node.advertise_host,
            node.heartbeat_port,
        );
    }

    info!(
        host = %node.advertise_host,
        heartbeat_port = node.heartbeat_port,
        "backend registration confirmed"
    );
    drop(conn);
    pool.disconnect()
        .await
        .context("failed to disconnect FE MySQL pool")?;
    Ok(())
}

/// Whether `SHOW BACKENDS` lists a row for this backend's advertised host and heartbeat port.
async fn node_is_registered(conn: &mut mysql_async::Conn, node: &BackendConfig) -> Result<bool> {
    let rows: Vec<Row> = conn
        .query("SHOW BACKENDS")
        .await
        .context("failed to query FE backend list")?;
    let heartbeat_port = node.heartbeat_port.to_string();

    for row in rows {
        // Column names as of Doris 4.1; fall back to the stable positions (id, host, port).
        let host = row
            .get::<String, _>("Host")
            .or_else(|| row.get::<String, _>(1));
        let port = row
            .get::<String, _>("HeartbeatPort")
            .or_else(|| row.get::<String, _>(2));

        if host.as_deref() == Some(node.advertise_host.as_str())
            && port.as_deref() == Some(heartbeat_port.as_str())
        {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Agent-service unsupported result helper for methods returning `TAgentResult`.
fn agent_result(rpc: &str) -> agent_service::TAgentResult {
    agent_service::TAgentResult::new(not_implemented_status(rpc), None, None, None)
}

/// Current Unix time in milliseconds for heartbeat staleness tracking and the BE epoch.
fn unix_time_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use doris_thrift::status::TStatusCode;
    use thrift::{
        protocol::{TBinaryInputProtocol, TBinaryOutputProtocol},
        transport::{TBufferedReadTransport, TBufferedWriteTransport},
    };

    use super::*;

    /// Builds a stable backend config for heartbeat handler tests.
    fn test_config() -> BackendConfig {
        BackendConfig {
            version: "test-version".to_string(),
            ..BackendConfig::default()
        }
    }

    /// Builds a heartbeat handler with its shared state so tests can inspect side effects.
    fn handler() -> (BackendHeartbeatHandler, SharedHeartbeatState) {
        let state = SharedHeartbeatState::new();
        (
            BackendHeartbeatHandler::new(test_config(), state.clone()),
            state,
        )
    }

    /// Builds a valid FE heartbeat for the requested epoch, shaped like
    /// `HeartbeatMgr.BackendHeartbeatHandler.pingOnce` fills it.
    fn master(epoch: types::TEpoch) -> TMasterInfo {
        TMasterInfo {
            network_address: types::TNetworkAddress::new("127.0.0.1".to_string(), 9020),
            cluster_id: 42,
            epoch,
            token: Some("token".to_string()),
            backend_ip: Some("127.0.0.1".to_string()),
            http_port: Some(8030),
            heartbeat_flags: Some(0),
            backend_id: Some(10001),
            ..Default::default()
        }
    }

    #[test]
    fn first_heartbeat_succeeds_and_records_state() {
        let (handler, state) = handler();

        let result = handler.handle_heartbeat(master(7)).unwrap();

        assert_eq!(result.status.status_code, TStatusCode::OK);
        let info = &result.backend_info;
        assert_eq!(info.be_port, 9060);
        assert_eq!(info.http_port, 8040);
        assert_eq!(info.brpc_port, Some(8060));
        assert_eq!(info.be_node_role.as_deref(), Some("mix"));
        assert_eq!(info.is_shutdown, Some(false));
        assert_eq!(info.arrow_flight_sql_port, Some(-1));
        assert!(info.be_start_time.is_some_and(|start| start > 0));
        let snapshot = state.snapshot();
        assert_eq!(snapshot.cluster_id, Some(42));
        assert_eq!(
            snapshot.token.as_ref().map(SecretString::expose_secret),
            Some("token")
        );
        assert_eq!(snapshot.epoch, Some(7));
        assert_eq!(snapshot.backend_id, Some(10001));
        assert_eq!(snapshot.registered_host.as_deref(), Some("127.0.0.1"));
        assert_eq!(
            snapshot.frontend_address,
            Some(types::TNetworkAddress::new("127.0.0.1".to_string(), 9020))
        );
        assert!(snapshot.last_heartbeat_ms.is_some());
    }

    #[test]
    fn be_start_time_is_stable_across_heartbeats() {
        let (handler, _) = handler();
        let first = handler.handle_heartbeat(master(7)).unwrap();
        let second = handler.handle_heartbeat(master(7)).unwrap();
        assert_eq!(
            first.backend_info.be_start_time,
            second.backend_info.be_start_time
        );
    }

    #[test]
    fn repeated_same_or_higher_epoch_succeeds() {
        let (handler, state) = handler();
        for epoch in [7, 7, 8] {
            assert_eq!(
                handler
                    .handle_heartbeat(master(epoch))
                    .unwrap()
                    .status
                    .status_code,
                TStatusCode::OK
            );
        }
        assert_eq!(state.snapshot().epoch, Some(8));
    }

    #[test]
    fn stale_epoch_fails() {
        let (handler, state) = handler();
        handler.handle_heartbeat(master(7)).unwrap();
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

    #[test]
    fn token_mismatch_fails() {
        let (handler, state) = handler();
        let mut changed = master(8);
        changed.token = Some("different-token".to_string());
        handler.handle_heartbeat(master(7)).unwrap();
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

    #[test]
    fn cluster_mismatch_fails() {
        let (handler, state) = handler();
        let mut changed = master(8);
        changed.cluster_id = 43;
        handler.handle_heartbeat(master(7)).unwrap();
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

    #[test]
    fn registered_host_must_match_advertised_host() {
        let (handler, state) = handler();
        let mut mismatched = master(7);
        mismatched.backend_ip = Some("10.0.0.9".to_string());
        let result = handler.handle_heartbeat(mismatched).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR);
        assert!(
            result.status.error_msgs.unwrap()[0].contains("--advertise-host 10.0.0.9"),
            "error names the fix"
        );
        assert_eq!(state.snapshot().epoch, None);
    }

    #[test]
    fn registration_uses_backend_surface() {
        assert_eq!(
            backend_registration_sql(&Host::local(), 9050),
            "ALTER SYSTEM ADD BACKEND \"127.0.0.1:9050\""
        );
    }

    #[test]
    fn backend_tablet_stat_returns_empty_inventory() {
        let result = BackendServiceHandler::new()
            .handle_get_tablet_stat()
            .unwrap();
        assert!(result.tablets_stats.is_empty());
    }

    #[test]
    fn backend_topic_publish_is_acknowledged() {
        let result = BackendServiceHandler::new()
            .handle_publish_topic_info(backend_service::TPublishTopicRequest::default())
            .unwrap();
        assert_eq!(result.status.status_code, TStatusCode::OK);
    }

    #[test]
    fn backend_unsupported_calls_return_not_implemented_status() {
        let handler = BackendServiceHandler::new();

        let routine_load_status = handler.handle_submit_routine_load_task(Vec::new()).unwrap();
        assert_not_implemented(
            &routine_load_status,
            "BackendService.submit_routine_load_task",
        );

        let agent_result = handler.handle_submit_tasks(Vec::new()).unwrap();
        assert_not_implemented(&agent_result.status, "BackendService.submit_tasks");
    }

    #[test]
    fn secret_string_debug_redacts_value() {
        assert_eq!(
            format!("{:?}", SecretString::new("do-not-log")),
            "SecretString(<redacted>)"
        );
    }

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

    /// A client that connects but never sends leaves the server blocked in `process()`;
    /// shutdown must still close that tracked connection and let `join()` return.
    #[test]
    fn shutdown_interrupts_blocked_connection() {
        let mut config = test_config();
        config.bind_host = Host::local();
        config.heartbeat_port = 0;
        let server = match start_heartbeat_server(config, SharedHeartbeatState::new()) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let addr = server.local_addr();

        let stream = TcpStream::connect(addr).unwrap();
        thread::sleep(Duration::from_millis(200));

        let shutdown = server.shutdown_handle();
        let (tx, rx) = std::sync::mpsc::channel();
        let joiner = thread::spawn(move || {
            let _ = tx.send(server.join());
        });
        shutdown.shutdown();

        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(join_result) => join_result.unwrap(),
            Err(_) => panic!("server join did not complete after shutdown (blocked connection)"),
        }
        joiner.join().unwrap();
        drop(stream);
    }

    /// Full request→process→reply round trip over the real thrift transport stack, the way
    /// the FE's pooled `HeartbeatService.Client` (buffered binary) talks to us.
    #[test]
    fn heartbeat_round_trip_over_thrift_transport() {
        use doris_thrift::heartbeat_service::{
            HeartbeatServiceSyncClient, THeartbeatServiceSyncClient,
        };

        let mut config = test_config();
        config.bind_host = Host::local();
        config.heartbeat_port = 0;
        let state = SharedHeartbeatState::new();
        let server = match start_heartbeat_server(config, state.clone()) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let addr = server.local_addr();

        let stream = TcpStream::connect(addr).unwrap();
        let channel = TTcpChannel::with_stream(stream);
        let (read_channel, write_channel) = channel.split().unwrap();
        let input_protocol =
            TBinaryInputProtocol::new(TBufferedReadTransport::new(read_channel), true);
        let output_protocol =
            TBinaryOutputProtocol::new(TBufferedWriteTransport::new(write_channel), true);
        let mut client = HeartbeatServiceSyncClient::new(input_protocol, output_protocol);

        let result = client.heartbeat(master(7)).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::OK);
        assert_eq!(result.backend_info.be_port, 9060);
        assert_eq!(state.snapshot().epoch, Some(7));
        assert_eq!(state.snapshot().backend_id, Some(10001));

        drop(client);
        server.shutdown();
        server.join().unwrap();
    }

    /// Same over the backend-service port: the FE's `get_tablet_stat` probe.
    #[test]
    fn backend_service_round_trip_over_thrift_transport() {
        use doris_thrift::backend_service::{BackendServiceSyncClient, TBackendServiceSyncClient};

        let mut config = test_config();
        config.bind_host = Host::local();
        config.be_port = 0;
        let server = match start_backend_server(&config) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let addr = server.local_addr();

        let stream = TcpStream::connect(addr).unwrap();
        let channel = TTcpChannel::with_stream(stream);
        let (read_channel, write_channel) = channel.split().unwrap();
        let input_protocol =
            TBinaryInputProtocol::new(TBufferedReadTransport::new(read_channel), true);
        let output_protocol =
            TBinaryOutputProtocol::new(TBufferedWriteTransport::new(write_channel), true);
        let mut client = BackendServiceSyncClient::new(input_protocol, output_protocol);

        let result = client.get_tablet_stat().unwrap();
        assert!(result.tablets_stats.is_empty());
        let result = client.get_dictionary_status(vec![1, 2]).unwrap();
        assert!(
            result
                .dictionary_status_list
                .is_some_and(|list| list.is_empty())
        );

        drop(client);
        server.shutdown();
        server.join().unwrap();
    }

    /// Detects sandboxed environments where binding a local listener is denied.
    fn is_permission_denied(err: &anyhow::Error) -> bool {
        err.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|err| err.kind() == std::io::ErrorKind::PermissionDenied)
        })
    }

    /// Verifies a status is NOT_IMPLEMENTED and includes the RPC name for diagnostics.
    fn assert_not_implemented(status: &TStatus, rpc: &str) {
        assert_eq!(status.status_code, TStatusCode::NOT_IMPLEMENTED_ERROR);
        assert!(
            status
                .error_msgs
                .as_ref()
                .is_some_and(|messages| messages.iter().any(|message| message.contains(rpc)))
        );
    }
}
