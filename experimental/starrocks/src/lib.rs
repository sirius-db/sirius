use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::{Read, Write},
    net::{
        IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs,
    },
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
    agent_service, backend_service,
    backend_service::{BackendServiceSyncHandler, BackendServiceSyncProcessor},
    data,
    frontend_service::{FrontendServiceSyncClient, TFrontendServiceSyncClient},
    heartbeat_service::{
        HeartbeatServiceSyncHandler, HeartbeatServiceSyncProcessor, TBackendInfo, THeartbeatResult,
        TMasterInfo,
    },
    internal_service, master_service, starrocks_external_service,
    status::TStatus,
    status_code::TStatusCode,
    types,
};
use thrift::{
    TransportErrorKind,
    protocol::{
        TBinaryInputProtocol, TBinaryInputProtocolFactory, TBinaryOutputProtocol,
        TBinaryOutputProtocolFactory, TInputProtocolFactory, TOutputProtocolFactory,
    },
    server::TProcessor,
    transport::{
        TBufferedReadTransport, TBufferedReadTransportFactory, TBufferedWriteTransport,
        TBufferedWriteTransportFactory, TIoChannel, TReadTransportFactory, TTcpChannel,
        TWriteTransportFactory,
    },
};
use tracing::{debug, info, instrument, warn};

mod brpc;
mod compute_node_service;
#[cfg(feature = "sirius-engine")]
mod engine;
mod engine_settings;
mod file_schema;
mod fragment_executor;
mod gpu_affinity;
mod local_exchange;
mod nixl_transport;
mod proto;
mod prpc;
mod prpc_client;
mod result_encoder;
mod result_store;
mod tunable;
#[cfg(all(test, feature = "sirius-engine"))]
mod wire_type_parity;

pub use brpc::BrpcServer;
pub use compute_node_service::ExchangeIdentity;
#[cfg(feature = "sirius-engine")]
pub use engine::SiriusEngine;
pub use engine_settings::{EngineSettings, derive_sirius_config_yaml};
pub use fragment_executor::{FragmentExecutor, FragmentResult, StubExecutor};
pub use gpu_affinity::{GpuSocket, cpu_affinity_for_gpu, gpu_socket};
pub use nixl_transport::NixlTransport;
pub use tunable::Tunables;

/// Serializes tests that bring up a GPU engine context: the engine keeps process-global GPU
/// state, so at most one context may be live at a time within the test binary.
#[cfg(all(test, feature = "sirius-engine"))]
pub(crate) static GPU_ENGINE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const COMPUTE_NODE_PROC_PATH: &str = "/compute_nodes";

// Bound the blocking FE report call so an unresponsive FE cannot wedge the report loop or
// leak its blocking-pool thread for the process lifetime. Kept well under the report interval.
const FE_REPORT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const FE_REPORT_RW_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Host(String);

impl Host {
    pub fn new(value: impl Into<String>) -> std::result::Result<Self, HostParseError> {
        let value = value.into();
        if value.trim().is_empty() {
            Err(HostParseError)
        } else {
            Ok(Self(value))
        }
    }

    pub fn local() -> Self {
        Self("127.0.0.1".to_string())
    }

    pub fn unspecified() -> Self {
        Self("0.0.0.0".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("Host").field(&self.0).finish()
    }
}

impl fmt::Display for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for Host {
    type Err = HostParseError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("host must not be empty")]
pub struct HostParseError;

#[derive(Clone, Default, Eq, PartialEq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose_secret(&self) -> &str {
        &self.0
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

#[derive(Clone, Debug, clap::Args)]
pub struct ComputeNodeConfig {
    #[arg(long, default_value = "0.0.0.0")]
    pub bind_host: Host,
    #[arg(long, default_value = "127.0.0.1")]
    pub advertise_host: Host,
    #[arg(long, default_value_t = 9050)]
    pub heartbeat_port: u16,
    #[arg(long = "thrift-port", default_value_t = 9060)]
    pub thrift_port: u16,
    #[arg(long, default_value_t = 8040)]
    pub http_port: u16,
    #[arg(long, default_value_t = 8060)]
    pub brpc_port: u16,
    /// TODO: run a real starlet/worker agent on this advertised port.
    #[arg(long, default_value_t = 9070)]
    pub starlet_port: u16,
    #[arg(long)]
    pub arrow_flight_port: Option<u16>,
    #[arg(skip = default_compute_node_version())]
    pub version: String,
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
            starlet_port: 9070,
            arrow_flight_port: None,
            version: default_compute_node_version(),
        }
    }
}

fn default_compute_node_version() -> String {
    format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

fn compute_node_registration_sql(host: &Host, heartbeat_port: u16) -> String {
    format!("ALTER SYSTEM ADD COMPUTE NODE \"{host}:{heartbeat_port}\"")
}

#[derive(Clone, Debug, clap::Args)]
pub struct FeConfig {
    #[arg(long = "fe-host", default_value = "127.0.0.1")]
    pub host: Host,
    #[arg(long = "fe-query-port", default_value_t = 9030)]
    pub query_port: u16,
    #[arg(long = "fe-user", default_value = "root")]
    pub user: String,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Snapshot of FE-provided heartbeat identity and report sequencing state.
pub struct HeartbeatStateSnapshot {
    /// Sticky StarRocks cluster id learned from the first valid FE heartbeat.
    pub cluster_id: Option<types::TClusterId>,
    /// Sticky FE auth token learned from the first valid FE heartbeat.
    pub token: Option<SecretString>,
    /// Latest accepted FE epoch; older epochs are rejected as stale.
    pub epoch: Option<types::TEpoch>,
    /// FE-assigned compute-node/backend id needed for follow-up reports.
    pub compute_node_id: Option<i64>,
    /// FE thrift address used for `FrontendService.report` after heartbeat.
    pub frontend_address: Option<types::TNetworkAddress>,
    /// Local monotonic report version for empty CN inventory reports.
    pub report_version: i64,
    /// Wall-clock timestamp of the latest accepted heartbeat.
    pub last_heartbeat_ms: Option<u128>,
}

#[derive(Debug, Default)]
struct HeartbeatState {
    // Sticky cluster identity prevents a running CN from silently changing FE clusters.
    cluster_id: Option<types::TClusterId>,
    // The FE token is stored redacted and must stay stable after first contact.
    token: Option<SecretString>,
    // Epoch monotonicity lets the CN reject stale FE heartbeat state.
    epoch: Option<types::TEpoch>,
    // FE-assigned id is required before the CN can send inventory reports.
    compute_node_id: Option<i64>,
    // FE thrift endpoint is learned from heartbeat and used for reports.
    frontend_address: Option<types::TNetworkAddress>,
    // Report versions start at zero and increment only when a report is built.
    report_version: i64,
    // Used by the registration refresher to detect missing/stale heartbeats.
    last_heartbeat_ms: Option<u128>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A prepared FE report plus the heartbeat identity it was built from.
pub struct ComputeNodeReport {
    /// FE-assigned backend id currently associated with this compute node.
    pub backend_id: i64,
    /// FE thrift address that should receive the report.
    pub frontend_address: types::TNetworkAddress,
    /// Empty-but-truthful report request for this storage-less CN.
    pub request: master_service::TReportRequest,
}

#[derive(Clone, Debug)]
pub struct SharedHeartbeatState(Arc<Mutex<HeartbeatState>>);

impl SharedHeartbeatState {
    /// Creates empty heartbeat state; reports are unavailable until heartbeat fills identity.
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
            compute_node_id: state.compute_node_id,
            frontend_address: state.frontend_address.clone(),
            report_version: state.report_version,
            last_heartbeat_ms: state.last_heartbeat_ms,
        }
    }

    /// Builds the next FE report only after heartbeat supplies FE address and CN id.
    pub fn next_report(&self, node: &ComputeNodeConfig) -> Option<ComputeNodeReport> {
        let mut state = self.0.lock().expect("heartbeat state mutex poisoned");
        let backend_id = state.compute_node_id?;
        let frontend_address = state.frontend_address.clone()?;
        // Saturating arithmetic avoids wrapping if a long-running process reaches i64::MAX.
        state.report_version = state.report_version.saturating_add(1);

        Some(ComputeNodeReport {
            backend_id,
            frontend_address,
            request: build_report_request(node, state.report_version),
        })
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

/// Whether the GPU engine has finished initializing and this CN can actually execute a fragment.
///
/// The CN binds its listeners BEFORE building the engine, because engine start-up reserves the
/// whole RMM pool and takes ~7 s on a GB200 — and for that entire window the process existed but
/// answered nothing. An FE that boots faster (~4 s) and remembers this node from its persisted
/// metadata probes into that gap, the probe fails, and the node is auto-blacklisted.
///
/// Binding early closes the gap, but it introduces the opposite hazard: a node that answers is a
/// node the FE will schedule work onto, and an engine that is still warming cannot run it. This
/// flag is the interlock. Until it flips, the heartbeat is answered with an ERROR status, which
/// leaves the FE holding the node as not-alive — so it is never scheduled, and (unlike a failed
/// fragment RPC) a failed heartbeat does NOT add a blacklist entry.
#[derive(Clone, Debug)]
pub struct EngineReadiness(Arc<AtomicBool>);

impl EngineReadiness {
    /// Starts closed: the engine is not up yet. Use this in real bring-up.
    pub fn warming() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Starts open. For tests and for builds with no engine to wait for.
    pub fn ready() -> Self {
        Self(Arc::new(AtomicBool::new(true)))
    }

    /// Opens the gate. Idempotent; call once the engine can execute fragments.
    pub fn mark_ready(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether the engine can execute fragments.
    pub fn is_ready(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

impl Default for EngineReadiness {
    fn default() -> Self {
        Self::ready()
    }
}

#[derive(Clone, Debug)]
pub struct ComputeNodeHeartbeatHandler {
    config: ComputeNodeConfig,
    state: SharedHeartbeatState,
    /// Gate that keeps the FE from scheduling onto a still-warming engine. See [`EngineReadiness`].
    readiness: EngineReadiness,
    reboot_time_secs: i64,
    hardware_cores: i32,
    // Operator-configured FE host used to validate the FE-advertised report target. The CN
    // later opens an outbound connection to `frontend_address`; without this guard any peer
    // that reaches the heartbeat port and wins trust-on-first-use could redirect that
    // outbound connection to an arbitrary host. `None` keeps the legacy permissive behavior.
    expected_frontend_host: Option<Host>,
}

impl ComputeNodeHeartbeatHandler {
    pub fn new(
        config: ComputeNodeConfig,
        state: SharedHeartbeatState,
        expected_frontend_host: Option<Host>,
        readiness: EngineReadiness,
    ) -> Self {
        Self {
            config,
            state,
            readiness,
            reboot_time_secs: unix_time_secs(),
            hardware_cores: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get().min(i32::MAX as usize) as i32)
                .unwrap_or(0),
            expected_frontend_host,
        }
    }

    fn handle_master_info(
        &self,
        master_info: &TMasterInfo,
    ) -> std::result::Result<(), HeartbeatError> {
        // Sirius is advertising itself as a StarRocks compute node, so reject BE heartbeats.
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
            .0
            .lock()
            .map_err(|_| HeartbeatError::StatePoisoned)?;

        // StarRocks FE epochs are monotonic; accepting an older epoch would roll state back.
        if let Some(epoch) = state.epoch
            && master_info.epoch < epoch
        {
            return Err(HeartbeatError::StaleEpoch {
                received: master_info.epoch,
                current: epoch,
            });
        }

        // Cluster id is learned once and then treated as immutable for this process.
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

        // Token changes are treated as suspicious because later RPCs depend on FE identity.
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
        // FE sends its thrift address in heartbeat; reports wait until this is known. Only
        // accept the advertised report target when it matches the operator-configured FE host
        // (when one is configured), so a hostile heartbeat cannot redirect the CN's outbound
        // report connection. The heartbeat itself is still accepted for liveness. This assumes
        // a single configured FE; multi-FE leader failover to a different host would need an
        // allowlist here.
        let advertised = &master_info.network_address;
        let report_target_trusted = match &self.expected_frontend_host {
            Some(expected) => frontend_host_trusted(expected.as_str(), &advertised.hostname),
            None => true,
        };
        if report_target_trusted {
            state.frontend_address = Some(advertised.clone());
        } else {
            warn!(
                advertised_host = %advertised.hostname,
                expected_host = %self.expected_frontend_host.as_ref().expect("checked above"),
                "ignoring FE-advertised report address that does not match the configured FE host"
            );
        }
        state.last_heartbeat_ms = Some(unix_time_millis());

        Ok(())
    }

    fn compute_node_info(&self) -> TBackendInfo {
        // Sentinels for capabilities a storage-less compute node does not expose. Named here
        // because TBackendInfo::new takes many same-typed positional Option<TPort>/Option<i64>
        // args (see HeartbeatService.thrift `struct TBackendInfo` for field order).
        const BE_RPC_PORT_DISABLED: i32 = -1; // legacy thrift RPC port; unused by this CN
        const PORT_DISABLED: i32 = -1; // arrow-flight port sentinel when not configured
        const MEM_LIMIT_UNSET: i64 = 0; // no advertised memory limit for this skeleton
        let arrow_flight_port = self
            .config
            .arrow_flight_port
            .map(i32::from)
            .unwrap_or(PORT_DISABLED);
        TBackendInfo::new(
            i32::from(self.config.thrift_port),        // be_port
            i32::from(self.config.http_port),          // http_port
            Some(BE_RPC_PORT_DISABLED),                // be_rpc_port
            Some(i32::from(self.config.brpc_port)),    // brpc_port
            Some(self.config.version.clone()),         // version
            Some(self.hardware_cores),                 // num_hardware_cores
            Some(i32::from(self.config.starlet_port)), // starlet_port
            Some(self.reboot_time_secs),               // reboot_time
            Some(false),                               // is_set_storage_path (CN has no storage)
            Some(MEM_LIMIT_UNSET),                     // mem_limit_bytes
            Some(arrow_flight_port),                   // arrow_flight_port
        )
    }
}

#[derive(Debug, thiserror::Error)]
enum HeartbeatError {
    #[error(
        "FE heartbeat did not include a node type; Sirius only supports StarRocks compute nodes"
    )]
    MissingNodeType,
    #[error(
        "FE heartbeat targeted this node as {received:?}; Sirius only supports StarRocks compute nodes"
    )]
    UnexpectedNodeType { received: types::TNodeType },
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

        // Record FE identity first even while warming: the epoch/token/backend-id this carries is
        // what later reports are addressed with, and dropping it would just move the problem.
        let status = match self.handle_master_info(&master_info) {
            Ok(()) if !self.readiness.is_ready() => {
                // Answer honestly rather than claiming health. An OK here would make the FE mark
                // this node alive and schedule a fragment onto an engine that cannot run it — and
                // THAT failure is a query-path failure, which is what auto-blacklists a node.
                debug!("heartbeat answered NOT READY: GPU engine is still initializing");
                error_status("compute node is still initializing its GPU engine".to_string())
            }
            Ok(()) => ok_status(),
            Err(err) => {
                warn!(error = %err, "rejecting FE heartbeat");
                error_status(err.to_string())
            }
        };

        Ok(THeartbeatResult::new(status, self.compute_node_info()))
    }
}

#[derive(Clone, Debug)]
/// StarRocks `BackendService` skeleton for FE/coordinator requests sent to the CN.
pub struct ComputeNodeBackendHandler;

impl ComputeNodeBackendHandler {
    /// Creates a stateless backend handler because no RPC has executable state yet.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ComputeNodeBackendHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendServiceSyncHandler for ComputeNodeBackendHandler {
    // Plan execution is deferred until a real execution engine/brpc path is wired in.
    fn handle_exec_plan_fragment(
        &self,
        _params: internal_service::TExecPlanFragmentParams,
    ) -> thrift::Result<internal_service::TExecPlanFragmentResult> {
        Ok(internal_service::TExecPlanFragmentResult::new(
            Some(not_implemented_status("BackendService.execPlanFragment")),
            None,
        ))
    }

    // Best-effort cancellation stub: acknowledge with OK so the FE's cancel path completes
    // cleanly instead of surfacing "not implemented" into unrelated queries. Real fragment
    // teardown is a separate work item; fragments are dispatched over brpc, so the brpc
    // `cancel_plan_fragment` is the surface that marks the results registry.
    fn handle_cancel_plan_fragment(
        &self,
        params: internal_service::TCancelPlanFragmentParams,
    ) -> thrift::Result<internal_service::TCancelPlanFragmentResult> {
        info!(
            fragment_instance_id = ?params.fragment_instance_id,
            "acknowledging BackendService.cancelPlanFragment (best-effort: no engine-side abort yet)"
        );
        Ok(internal_service::TCancelPlanFragmentResult::new(Some(
            ok_status(),
        )))
    }

    // Data exchange is unsupported until fragment execution and exchange tracking exist.
    fn handle_transmit_data(
        &self,
        _params: internal_service::TTransmitDataParams,
    ) -> thrift::Result<internal_service::TTransmitDataResult> {
        Ok(internal_service::TTransmitDataResult::new(
            Some(not_implemented_status("BackendService.transmitData")),
            None,
            None,
            None,
        ))
    }

    // Fetch returns an empty required batch plus NOT_IMPLEMENTED instead of panicking.
    fn handle_fetch_data(
        &self,
        _params: internal_service::TFetchDataParams,
    ) -> thrift::Result<internal_service::TFetchDataResult> {
        Ok(internal_service::TFetchDataResult::new(
            data::TResultBatch::new(Vec::new(), false, 0, None),
            true,
            0,
            Some(not_implemented_status("BackendService.fetchData")),
        ))
    }

    // Agent tasks mutate backend state/storage, so they are rejected explicitly.
    fn handle_submit_tasks(
        &self,
        _tasks: Vec<agent_service::TAgentTaskRequest>,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.submitTasks"))
    }

    // Snapshot creation requires tablet storage, which this skeleton CN does not expose.
    fn handle_make_snapshot(
        &self,
        _snapshot_request: agent_service::TSnapshotRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.makeSnapshot"))
    }

    // Snapshot release is paired with snapshot creation and remains unsupported here.
    fn handle_release_snapshot(
        &self,
        _snapshot_path: String,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.releaseSnapshot"))
    }

    // Cluster-state publishing is agent-task style coordination, not safe to fake.
    fn handle_publish_cluster_state(
        &self,
        _request: agent_service::TAgentPublishRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.publishClusterState"))
    }

    // Mini-load ETL work needs local execution/storage and is intentionally absent.
    fn handle_submit_etl_task(
        &self,
        _request: agent_service::TMiniLoadEtlTaskRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.submitEtlTask"))
    }

    // ETL status needs a required enum, so UNKNOWN accompanies NOT_IMPLEMENTED.
    fn handle_get_etl_status(
        &self,
        _request: agent_service::TMiniLoadEtlStatusRequest,
    ) -> thrift::Result<agent_service::TMiniLoadEtlStatusResult> {
        Ok(agent_service::TMiniLoadEtlStatusResult::new(
            not_implemented_status("BackendService.getEtlStatus"),
            types::TEtlState::UNKNOWN,
            None,
            None,
            None,
        ))
    }

    // Deleting ETL files would affect storage paths this CN does not own.
    fn handle_delete_etl_files(
        &self,
        _request: agent_service::TDeleteEtlFilesRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        Ok(agent_result("BackendService.deleteEtlFiles"))
    }

    // Export execution is not present in this first thrift surface.
    fn handle_submit_export_task(
        &self,
        _request: backend_service::TExportTaskRequest,
    ) -> thrift::Result<TStatus> {
        Ok(not_implemented_status("BackendService.submitExportTask"))
    }

    // Export status needs a required state, so UNKNOWN accompanies NOT_IMPLEMENTED.
    fn handle_get_export_status(
        &self,
        _task_id: types::TUniqueId,
    ) -> thrift::Result<internal_service::TExportStatusResult> {
        Ok(internal_service::TExportStatusResult::new(
            not_implemented_status("BackendService.getExportStatus"),
            types::TExportState::UNKNOWN,
            None,
        ))
    }

    // Export task cleanup remains unsupported because no export task can be created.
    fn handle_erase_export_task(&self, _task_id: types::TUniqueId) -> thrift::Result<TStatus> {
        Ok(not_implemented_status("BackendService.eraseExportTask"))
    }

    // Empty tablet stats are truthful for a storage-less compute node.
    fn handle_get_tablet_stat(&self) -> thrift::Result<backend_service::TTabletStatResult> {
        Ok(backend_service::TTabletStatResult::new(BTreeMap::new()))
    }

    // Routine-load execution is deferred until load/execution support exists.
    fn handle_submit_routine_load_task(
        &self,
        _tasks: Vec<backend_service::TRoutineLoadTask>,
    ) -> thrift::Result<TStatus> {
        Ok(not_implemented_status(
            "BackendService.submitRoutineLoadTask",
        ))
    }

    // Stream-load channels imply load state, so this skeleton reports unsupported.
    fn handle_finish_stream_load_channel(
        &self,
        _stream_load_channel: backend_service::TStreamLoadChannel,
    ) -> thrift::Result<TStatus> {
        Ok(not_implemented_status(
            "BackendService.finishStreamLoadChannel",
        ))
    }

    // External scanner open is an execution path and remains unimplemented.
    fn handle_open_scanner(
        &self,
        _params: starrocks_external_service::TScanOpenParams,
    ) -> thrift::Result<starrocks_external_service::TScanOpenResult> {
        Ok(starrocks_external_service::TScanOpenResult::new(
            not_implemented_status("BackendService.openScanner"),
            None,
            None,
        ))
    }

    // Scanner batches return EOS with NOT_IMPLEMENTED to satisfy required result fields.
    fn handle_get_next(
        &self,
        _params: starrocks_external_service::TScanNextBatchParams,
    ) -> thrift::Result<starrocks_external_service::TScanBatchResult> {
        Ok(starrocks_external_service::TScanBatchResult::new(
            not_implemented_status("BackendService.getNext"),
            Some(true),
            Some(Vec::new()),
        ))
    }

    // Scanner close is unsupported because scanner contexts are never created.
    fn handle_close_scanner(
        &self,
        _params: starrocks_external_service::TScanCloseParams,
    ) -> thrift::Result<starrocks_external_service::TScanCloseResult> {
        Ok(starrocks_external_service::TScanCloseResult::new(
            not_implemented_status("BackendService.closeScanner"),
        ))
    }

    // Tablet metadata is safe to report as empty because this CN has no tablet storage.
    fn handle_get_tablets_info(
        &self,
        _request: backend_service::TGetTabletsInfoRequest,
    ) -> thrift::Result<backend_service::TGetTabletsInfoResult> {
        Ok(backend_service::TGetTabletsInfoResult::new(
            ok_status(),
            Some(0),
            Some(BTreeMap::new()),
        ))
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
/// HTTP health listener handle. Not a thrift service, but it has the same lifecycle
/// (bind, blocking accept loop on its own thread, wake-and-join shutdown).
pub type HttpServer = ThriftServer;

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

/// Heartbeat service shutdown handle.
pub type HeartbeatServerShutdown = ThriftServerShutdown;
/// Backend service shutdown handle.
pub type BackendServerShutdown = ThriftServerShutdown;

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

        // Close the race with shutdown(): if shutdown was requested in the window between the
        // accept loop's is_requested() check and this store, close_active_connection() may have
        // already run (and found an empty slot), and shutdown()'s one-shot body will not run
        // again. Re-checking the flag under the same lock that close_active_connection() takes
        // guarantees this connection is interrupted rather than blocking the processor — and the
        // accept loop — forever. SeqCst + the mutex give the needed ordering: either this load
        // observes the request, or close_active_connection() observes this stored stream.
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
    config: ComputeNodeConfig,
    state: SharedHeartbeatState,
    expected_frontend_host: Option<Host>,
    readiness: EngineReadiness,
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
        ComputeNodeHeartbeatHandler::new(config, state, expected_frontend_host, readiness),
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

/// Starts the StarRocks `BackendService` thrift skeleton on `--thrift-port`.
pub fn start_backend_server(config: &ComputeNodeConfig) -> Result<BackendServer> {
    let listen_addr = format!("{}:{}", config.bind_host, config.thrift_port);
    let listener = TcpListener::bind(&listen_addr)
        .with_context(|| format!("failed to bind backend Thrift server at {listen_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("failed to read backend Thrift server address")?;
    let shutdown = ThriftServerShutdown::new(listener_wake_addr(local_addr));
    let server_shutdown = shutdown.clone();

    info!(
        address = %local_addr,
        brpc_port = config.brpc_port,
        "starting backend Thrift server; brpc execution is not implemented yet"
    );
    // BackendService is stateless for now because every non-inventory RPC is unsupported.
    let processor = Arc::new(BackendServiceSyncProcessor::new(
        ComputeNodeBackendHandler::new(),
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

/// Starts the HTTP listener on `--http-port`.
///
/// This exists because the CN ADVERTISES `http_port` in its heartbeat `TBackendInfo` but, until
/// this listener, never bound it — and that gap is not cosmetic. The FE's blacklist eviction path
/// (`HostBlacklist.remove` -> `NetUtils.checkAccessibleForAllPorts(host, [bePort, brpcPort,
/// httpPort])`) opens a raw TCP connection to ALL THREE advertised ports and un-blacklists a node
/// only when every one of them accepts. With `http_port` unbound that check could never pass, so a
/// CN auto-blacklisted by a single transient RPC failure — including the FE-heartbeat-before-CN-ready
/// race during cluster start-up — stayed blacklisted for the FE's entire process lifetime and
/// silently did zero work, while still reporting `Alive = true`.
///
/// Measured on the 4x GB200 box before this fix: 2 of 4 CNs were auto-blacklisted ~2 s after every
/// cluster start, and q14/SF100 ran 48.9% / 0% / 0% / 51.1% across the four CNs. With the blacklist
/// manually cleared, the same query ran 25.6% / 24.6% / 24.7% / 25.1%.
///
/// The FE only needs the port to ACCEPT, so this is deliberately not a real HTTP stack: it answers
/// anything with a fixed 200 and closes. Read/write timeouts stop one slow client from stalling the
/// single-threaded accept loop.
pub fn start_http_server(config: &ComputeNodeConfig) -> Result<HttpServer> {
    let listen_addr = format!("{}:{}", config.bind_host, config.http_port);
    let listener = TcpListener::bind(&listen_addr)
        .with_context(|| format!("failed to bind HTTP server at {listen_addr}"))?;
    let local_addr = listener
        .local_addr()
        .context("failed to read HTTP server address")?;
    let shutdown = ThriftServerShutdown::new(listener_wake_addr(local_addr));
    let server_shutdown = shutdown.clone();

    info!(address = %local_addr, "starting HTTP health listener");
    let join_handle = thread::spawn(move || run_http_server(listener, server_shutdown));

    Ok(ThriftServer {
        join_handle: Some(join_handle),
        shutdown,
        local_addr,
        name: "http",
    })
}

/// Accept loop for the health listener: at most one fixed response per connection, then close.
fn run_http_server(listener: TcpListener, shutdown: ThriftServerShutdown) -> Result<()> {
    const RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 3\r\nConnection: close\r\n\r\nOK\n";
    const IO_TIMEOUT: Duration = Duration::from_secs(2);

    for stream in listener.incoming() {
        if shutdown.is_requested() {
            break;
        }

        match stream {
            Ok(mut stream) => {
                if shutdown.is_requested() {
                    break;
                }
                // The FE's reachability probe connects and closes without sending a byte, so every
                // step here is best-effort: a client that vanishes mid-exchange is the normal case
                // and must never take the listener down.
                let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
                let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
                let mut scratch = [0_u8; 1024];
                let _ = stream.read(&mut scratch);
                if let Err(err) = stream.write_all(RESPONSE) {
                    debug!(error = %err, "HTTP health client went away before the response");
                }
                let _ = stream.flush();
                let _ = stream.shutdown(Shutdown::Both);
            }
            Err(_) if shutdown.is_requested() => break,
            Err(err) => warn!(error = %err, "failed to accept HTTP connection"),
        }
    }

    Ok(())
}

/// Runs one generated thrift processor behind a blocking TCP listener.
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
                // StarRocks thrift clients use one connection for a sequence of calls.
                // A transient per-connection error (e.g. fd exhaustion on try_clone/split, or a
                // poisoned mutex) must not tear down the whole server — log and keep accepting,
                // mirroring the accept-error and processor-error paths.
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
                // StarRocks may probe unsupported paths; keep the server alive after one error.
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

/// Sends one empty inventory report to FE if heartbeat has provided FE address and CN id.
#[instrument(skip_all)]
pub fn report_to_frontend_once(
    node: &ComputeNodeConfig,
    state: &SharedHeartbeatState,
) -> Result<Option<master_service::TMasterResult>> {
    // Returning None lets the caller poll without error before the first heartbeat arrives.
    let report = match state.next_report(node) {
        Some(report) => report,
        None => return Ok(None),
    };

    // FE thrift address comes from TMasterInfo.network_address in heartbeat.
    let frontend = report.frontend_address.clone();
    // Resolve and connect with bounded timeouts so an unreachable or accept-but-never-reply FE
    // cannot wedge this blocking call — and therefore the whole report loop — indefinitely.
    let socket_addrs: Vec<SocketAddr> = (frontend.hostname.as_str(), frontend.port as u16)
        .to_socket_addrs()
        .with_context(|| {
            format!(
                "failed to resolve FE thrift address {}:{}",
                frontend.hostname, frontend.port
            )
        })?
        .collect();
    // Try every resolved address (e.g. an IPv4/IPv6 dual-stack or round-robin host) with a
    // bounded connect, matching TcpStream::connect's multi-address behavior while still
    // enforcing a timeout so an unresponsive FE cannot wedge the report loop.
    let mut stream = None;
    let mut last_err: Option<std::io::Error> = None;
    for socket_addr in &socket_addrs {
        match TcpStream::connect_timeout(socket_addr, FE_REPORT_CONNECT_TIMEOUT) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(err) => last_err = Some(err),
        }
    }
    let stream = stream.ok_or_else(|| {
        let detail = last_err
            .map(|err| err.to_string())
            .unwrap_or_else(|| "host resolved to no socket addresses".to_string());
        anyhow!(
            "failed to connect to FE thrift service at {}:{}: {detail}",
            frontend.hostname,
            frontend.port
        )
    })?;
    stream
        .set_read_timeout(Some(FE_REPORT_RW_TIMEOUT))
        .context("failed to set FE thrift read timeout")?;
    stream
        .set_write_timeout(Some(FE_REPORT_RW_TIMEOUT))
        .context("failed to set FE thrift write timeout")?;
    let channel = TTcpChannel::with_stream(stream);

    // Client and server both use strict binary thrift protocols in this crate.
    let (read_channel, write_channel) = channel
        .split()
        .map_err(|err| anyhow!("failed to split FE thrift connection: {err}"))?;
    let read_transport = TBufferedReadTransport::new(read_channel);
    let write_transport = TBufferedWriteTransport::new(write_channel);
    let input_protocol = TBinaryInputProtocol::new(read_transport, true);
    let output_protocol = TBinaryOutputProtocol::new(write_transport, true);
    let mut client = FrontendServiceSyncClient::new(input_protocol, output_protocol);

    let result = client
        .report(report.request)
        .map_err(|err| anyhow!("failed to report compute node inventory to FE: {err}"))?;
    // The Thrift round-trip succeeding does not mean the FE accepted the report; a non-OK
    // status (e.g. unrecognized node or malformed payload) must surface as an error instead of
    // being silently logged as success by the caller.
    if result.status.status_code != TStatusCode::OK {
        bail!(
            "FE rejected compute node report (status {:?}): {}",
            result.status.status_code,
            result
                .status
                .error_msgs
                .as_ref()
                .map(|messages| messages.join("; "))
                .unwrap_or_default()
        );
    }
    Ok(Some(result))
}

/// Builds the truthful "empty CN" report payload expected by `FrontendService.report`.
fn build_report_request(
    node: &ComputeNodeConfig,
    report_version: i64,
) -> master_service::TReportRequest {
    // A storage-less compute node sends an empty task report carrying its identity and
    // report_version. The FE's ReportHandler picks the report type from WHICH optional fields
    // are present and rejects a request that sets several types at once, and an empty-but-present
    // tablet/disk map would falsely assert this node hosts zero tablets/disks (a Backend-style
    // report). So tablets/disks/tablet_list/compaction-score/workgroups are LEFT UNSET; `tasks`
    // is sent as an empty map because the FE only recognizes a compute node that reports at least
    // one of tasks/resource_usage/datacache_metrics.
    // NOTE: the exact shape the FE accepts is version-dependent; report_to_frontend_once now
    // checks the returned status, so a rejection surfaces instead of being silently logged as
    // success.
    let tasks: BTreeMap<types::TTaskType, BTreeSet<i64>> = BTreeMap::new();

    master_service::TReportRequest::new(
        // StarRocks expects the backend identity to carry host plus thrift/http ports.
        types::TBackend::new(
            node.advertise_host.to_string(),
            i32::from(node.thrift_port),
            i32::from(node.http_port),
        ),
        Some(report_version), // report_version (required)
        Some(tasks),          // tasks (empty: for compute-node recognition only)
        None,                 // tablets (no tablet storage)
        None,                 // disks (no local disks)
        None,                 // force_recovery
        None,                 // tablet_list
        None,                 // tablet_max_compaction_score (only with a tablet report)
        None,                 // active_workgroups
        None,                 // resource_usage
        None,                 // datacache_metrics
    )
}

/// Registers the compute node with FE over the MySQL protocol.
#[instrument(
    skip_all,
    fields(
        fe_host = %fe.host,
        fe_query_port = fe.query_port,
        advertise_host = %node.advertise_host,
        heartbeat_port = node.heartbeat_port,
    )
)]
pub async fn register_node(fe: &FeConfig, node: &ComputeNodeConfig) -> Result<()> {
    let opts = OptsBuilder::default()
        .ip_or_hostname(fe.host.to_string())
        .tcp_port(fe.query_port)
        .prefer_socket(false)
        .user(Some(fe.user.clone()))
        .pass(Some(fe.password.expose_secret().to_string()));
    let pool = Pool::new(opts);
    let mut conn = pool
        .get_conn()
        .await
        .with_context(|| format!("failed to connect to FE at {}:{}", fe.host, fe.query_port))?;

    if node_is_registered(&mut conn, node).await? {
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

    let sql = compute_node_registration_sql(&node.advertise_host, node.heartbeat_port);
    info!(sql = %sql, "registering compute node with FE");
    if let Err(err) = conn.query_drop(sql).await {
        warn!(error = %err, "ALTER SYSTEM ADD COMPUTE NODE failed; checking whether compute node already exists");
        if !node_is_registered(&mut conn, node).await? {
            return Err(err).context("failed to register compute node with FE");
        }
    }

    if !node_is_registered(&mut conn, node).await? {
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

async fn node_is_registered(
    conn: &mut mysql_async::Conn,
    node: &ComputeNodeConfig,
) -> Result<bool> {
    let sql = format!("SHOW PROC '{COMPUTE_NODE_PROC_PATH}'");
    let rows: Vec<Row> = conn
        .query(sql)
        .await
        .context("failed to query FE compute node list")?;
    let heartbeat_port = node.heartbeat_port.to_string();

    for row in rows {
        let host = row
            .get::<String, _>("IP")
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

/// Every compute node the FE currently reports as alive, as `(advertised host, brpc port)` —
/// which is exactly the nixl agent name (`{advertise_host}:{brpc_port}`) of each peer, so the
/// caller can tell itself apart from its peers by string comparison.
///
/// The FE only learns a node's `BrpcPort` from its heartbeat reply (`TBackendInfo`), and that is
/// the same event that flips `Alive` to true — so filtering on `Alive` is what makes the port
/// trustworthy. Column fallbacks mirror [`node_is_registered`]: `ComputeNodeProcDir.java` orders
/// the row `ComputeNodeId, IP, HeartbeatPort, BePort, HttpPort, BrpcPort, LastStartTime,
/// LastHeartbeat, Alive, ...`.
pub async fn list_alive_compute_nodes(fe: &FeConfig) -> Result<Vec<(String, u16)>> {
    let opts = OptsBuilder::default()
        .ip_or_hostname(fe.host.to_string())
        .tcp_port(fe.query_port)
        .prefer_socket(false)
        .user(Some(fe.user.clone()))
        .pass(Some(fe.password.expose_secret().to_string()));
    let pool = Pool::new(opts);
    let mut conn = pool
        .get_conn()
        .await
        .with_context(|| format!("failed to connect to FE at {}:{}", fe.host, fe.query_port))?;
    let rows: Vec<Row> = conn
        .query(format!("SHOW PROC '{COMPUTE_NODE_PROC_PATH}'"))
        .await
        .context("failed to query FE compute node list")?;
    drop(conn);
    pool.disconnect()
        .await
        .context("failed to disconnect FE MySQL pool")?;

    let mut nodes = Vec::with_capacity(rows.len());
    for row in rows {
        let alive = row
            .get::<String, _>("Alive")
            .or_else(|| row.get::<String, _>(8));
        if alive.as_deref() != Some("true") {
            continue;
        }
        let (Some(host), Some(port)) = (
            row.get::<String, _>("IP")
                .or_else(|| row.get::<String, _>(1)),
            row.get::<String, _>("BrpcPort")
                .or_else(|| row.get::<String, _>(5)),
        ) else {
            continue;
        };
        // A node whose brpc port is still unknown (or nonsense) is simply not warmable yet.
        if let Ok(port) = port.parse::<u16>()
            && port != 0
        {
            nodes.push((host, port));
        }
    }
    Ok(nodes)
}

/// StarRocks success status helper.
fn ok_status() -> TStatus {
    TStatus::new(TStatusCode::OK, None)
}

/// StarRocks unsupported status helper that names the RPC for operator diagnostics.
fn not_implemented_status(rpc: &str) -> TStatus {
    TStatus::new(
        TStatusCode::NOT_IMPLEMENTED_ERROR,
        Some(vec![format!("{rpc} is not implemented")]),
    )
}

/// Heartbeat rejection status helper.
fn error_status(message: String) -> TStatus {
    TStatus::new(TStatusCode::INTERNAL_ERROR, Some(vec![message]))
}

/// Agent-service unsupported result helper for methods returning `TAgentResult`.
fn agent_result(rpc: &str) -> agent_service::TAgentResult {
    agent_service::TAgentResult::new(not_implemented_status(rpc), None, None, None)
}

/// Current Unix time in seconds for heartbeat reboot metadata.
fn unix_time_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

/// Current Unix time in milliseconds for heartbeat staleness tracking.
fn unix_time_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Resolves a host (hostname or IP literal) to its IP addresses, ignoring the port. Returns an
/// empty vec if resolution fails so callers can treat "unknown" distinctly from "no match".
fn resolve_host_ips(host: &str) -> Vec<IpAddr> {
    (host, 0u16)
        .to_socket_addrs()
        .map(|addrs| addrs.map(|addr| addr.ip()).collect())
        .unwrap_or_default()
}

/// Decides whether an FE-advertised report host may be trusted against the configured FE host.
/// An exact string match short-circuits without DNS. Otherwise both sides are resolved and a
/// shared IP is required. Resolution failure on either side is treated as inconclusive and
/// trusted, so a transient DNS hiccup never drops a legitimate FE address; only a confident
/// mismatch (both resolve, no shared IP) is rejected.
fn frontend_host_trusted(expected_host: &str, advertised_host: &str) -> bool {
    if expected_host == advertised_host {
        return true;
    }
    let expected_ips = resolve_host_ips(expected_host);
    let advertised_ips = resolve_host_ips(advertised_host);
    expected_ips.is_empty()
        || advertised_ips.is_empty()
        || expected_ips.iter().any(|ip| advertised_ips.contains(ip))
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
            ComputeNodeHeartbeatHandler::new(
                test_config(),
                state.clone(),
                None,
                EngineReadiness::ready(),
            ),
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
        assert_eq!(
            snapshot.frontend_address,
            Some(types::TNetworkAddress::new("127.0.0.1".to_string(), 9020))
        );
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
        assert_eq!(
            compute_node_registration_sql(&Host::local(), 9050),
            "ALTER SYSTEM ADD COMPUTE NODE \"127.0.0.1:9050\""
        );
        assert_eq!(COMPUTE_NODE_PROC_PATH, "/compute_nodes");
    }

    /// Verifies tablet stat inventory is truthfully empty for a storage-less CN.
    #[test]
    fn backend_tablet_stat_returns_empty_inventory() {
        let result = ComputeNodeBackendHandler::new()
            .handle_get_tablet_stat()
            .unwrap();

        assert!(result.tablets_stats.is_empty());
    }

    /// Verifies tablet metadata reporting succeeds with version zero and no tablets.
    #[test]
    fn backend_tablets_info_returns_empty_ok_report() {
        let result = ComputeNodeBackendHandler::new()
            .handle_get_tablets_info(backend_service::TGetTabletsInfoRequest::new())
            .unwrap();

        assert_eq!(result.status.status_code, TStatusCode::OK);
        assert_eq!(result.report_version, Some(0));
        assert!(result.tablets.as_ref().is_some_and(BTreeMap::is_empty));
    }

    /// Verifies unsupported backend RPCs return explicit NOT_IMPLEMENTED statuses.
    #[test]
    fn backend_unsupported_calls_return_not_implemented_status() {
        let handler = ComputeNodeBackendHandler::new();

        let routine_load_status = handler.handle_submit_routine_load_task(Vec::new()).unwrap();
        assert_not_implemented(&routine_load_status, "BackendService.submitRoutineLoadTask");

        let agent_result = handler.handle_submit_tasks(Vec::new()).unwrap();
        assert_not_implemented(&agent_result.status, "BackendService.submitTasks");

        let export_status = handler
            .handle_erase_export_task(types::TUniqueId::new(0, 0))
            .unwrap();
        assert_not_implemented(&export_status, "BackendService.eraseExportTask");
    }

    /// Verifies FE reporting waits until heartbeat supplies FE address and backend id.
    #[test]
    fn report_waits_for_heartbeat_identity() {
        let state = SharedHeartbeatState::new();

        assert!(state.next_report(&test_config()).is_none());
        assert_eq!(state.snapshot().report_version, 0);
    }

    /// Verifies FE report payload identity, empty maps, and monotonic report version.
    #[test]
    fn report_payload_includes_identity_empty_state_and_monotonic_version() {
        let (handler, state) = handler();
        let mut config = test_config();
        config.advertise_host = Host::new("cn.example.test").unwrap();
        config.thrift_port = 19060;
        config.http_port = 18040;

        assert_eq!(
            handler
                .handle_heartbeat(master(7))
                .unwrap()
                .status
                .status_code,
            TStatusCode::OK
        );

        let first = state.next_report(&config).unwrap();
        let second = state.next_report(&config).unwrap();

        assert_eq!(first.backend_id, 10001);
        assert_eq!(
            first.frontend_address,
            types::TNetworkAddress::new("127.0.0.1".to_string(), 9020)
        );
        assert_eq!(first.request.backend.host, "cn.example.test");
        assert_eq!(first.request.backend.be_port, 19060);
        assert_eq!(first.request.backend.http_port, 18040);
        assert_eq!(first.request.report_version, Some(1));
        assert_eq!(second.request.report_version, Some(2));
        // tasks is an empty map (compute-node recognition); the Backend-only inventory fields
        // are left unset so the FE does not interpret this as a tablet/disk report.
        assert!(first.request.tasks.as_ref().is_some_and(BTreeMap::is_empty));
        assert!(first.request.tablets.is_none());
        assert!(first.request.disks.is_none());
        assert!(first.request.tablet_list.is_none());
        assert!(first.request.tablet_max_compaction_score.is_none());
        assert!(first.request.active_workgroups.is_none());
        assert_eq!(state.snapshot().report_version, 2);
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
        let server = match start_heartbeat_server(config, SharedHeartbeatState::new(), None, EngineReadiness::ready()) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let stream = TcpStream::connect(server.local_addr()).unwrap();

        server.shutdown();
        server.join().unwrap();
        drop(stream);
    }

    /// The advertised `http_port` must actually accept a connection, because the FE only lifts a
    /// node's blacklist entry once `NetUtils.checkAccessibleForAllPorts` can reach EVERY advertised
    /// port. While this listener did not exist, a CN auto-blacklisted by one transient RPC failure
    /// stayed blacklisted for the FE's whole lifetime and silently did zero work.
    #[test]
    fn http_server_accepts_and_answers_the_reachability_probe() {
        let mut config = test_config();
        config.bind_host = Host::local();
        config.http_port = 0;
        let server = match start_http_server(&config) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };

        // The FE's probe connects and closes without writing a byte; answer it anyway so the port
        // is useful to a human with curl. Reading to EOF also proves the server closes the socket.
        let mut stream = TcpStream::connect(server.local_addr()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream.write_all(b"GET / HTTP/1.1\r\n\r\n").unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "expected a 200 status line, got {response:?}"
        );

        server.shutdown();
        server.join().unwrap();
    }

    /// The readiness gate is the interlock that makes binding-before-engine-init safe: while the
    /// engine is warming the port must ANSWER (so the FE does not blacklist an unreachable node)
    /// but must NOT claim health (so the FE does not schedule a fragment onto an engine that
    /// cannot run it). Flipping the gate must switch the answer without restarting anything.
    #[test]
    fn warming_heartbeat_is_answered_but_not_ok_until_the_engine_is_ready() {
        let state = SharedHeartbeatState::new();
        let readiness = EngineReadiness::warming();
        let handler = ComputeNodeHeartbeatHandler::new(
            test_config(),
            state.clone(),
            None,
            readiness.clone(),
        );

        // Warming: answered (not a connection failure) but explicitly not OK.
        let warming = handler.handle_heartbeat(master(7)).unwrap();
        assert_ne!(
            warming.status.status_code,
            TStatusCode::OK,
            "a warming CN must not report OK -- the FE would schedule work onto it"
        );
        // FE identity is still captured while warming, so the first post-ready report is addressed
        // correctly instead of waiting for another heartbeat round trip.
        assert_eq!(state.snapshot().epoch, Some(7));
        assert_eq!(state.snapshot().compute_node_id, Some(10001));

        readiness.mark_ready();

        let ready = handler.handle_heartbeat(master(7)).unwrap();
        assert_eq!(
            ready.status.status_code,
            TStatusCode::OK,
            "once the engine is up the same handler must report OK"
        );
        // Same advertised ports either way -- readiness changes the STATUS, not the identity.
        assert_eq!(ready.backend_info.be_port, warming.backend_info.be_port);
    }

    /// `mark_ready` is idempotent and `ready()` starts open, so non-engine builds and tests are
    /// unaffected by the gate.
    #[test]
    fn readiness_defaults_are_open_and_marking_is_idempotent() {
        let r = EngineReadiness::ready();
        assert!(r.is_ready());
        r.mark_ready();
        assert!(r.is_ready());

        let w = EngineReadiness::warming();
        assert!(!w.is_ready());
        let clone = w.clone();
        w.mark_ready();
        assert!(clone.is_ready(), "clones must share one gate, not copy it");
    }

    /// A bare connect-then-close -- exactly what the FE reachability probe does -- must not take
    /// the accept loop down, so a later probe still succeeds.
    #[test]
    fn http_server_survives_a_client_that_never_reads() {
        let mut config = test_config();
        config.bind_host = Host::local();
        config.http_port = 0;
        let server = match start_http_server(&config) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let addr = server.local_addr();

        drop(TcpStream::connect(addr).unwrap());
        // The second probe is the assertion: it can only connect if the first one left the
        // listener alive.
        let mut stream = TcpStream::connect(addr).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response:?}");

        server.shutdown();
        server.join().unwrap();
    }

    /// Regression for the shutdown TOCTOU race: a client that connects but never sends a
    /// request leaves the server blocked in `processor.process()`; shutdown must still close
    /// that tracked connection and let `join()` return promptly rather than hang forever.
    #[test]
    fn shutdown_interrupts_blocked_connection() {
        let mut config = test_config();
        config.bind_host = Host::local();
        config.heartbeat_port = 0;
        let server = match start_heartbeat_server(config, SharedHeartbeatState::new(), None, EngineReadiness::ready()) {
            Ok(server) => server,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let addr = server.local_addr();

        // Connect without sending anything, then give the accept loop time to track it.
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

    /// Verifies a full request->process->reply round trip over the real Thrift transport stack
    /// (not just a direct handler call), covering the machinery shared by the backend server.
    #[test]
    fn heartbeat_round_trip_over_thrift_transport() {
        use starrocks_thrift::heartbeat_service::{
            HeartbeatServiceSyncClient, THeartbeatServiceSyncClient,
        };

        let mut config = test_config();
        config.bind_host = Host::local();
        config.heartbeat_port = 0;
        let state = SharedHeartbeatState::new();
        let server = match start_heartbeat_server(config, state.clone(), None, EngineReadiness::ready()) {
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
        // The state was mutated through the full transport path, not a direct handler call.
        assert_eq!(state.snapshot().epoch, Some(7));
        assert_eq!(state.snapshot().compute_node_id, Some(10001));

        drop(client);
        server.shutdown();
        server.join().unwrap();
    }

    /// Verifies a heartbeat advertising a report address that does not match the configured FE
    /// host is still accepted for liveness, but does not become the outbound report target.
    #[test]
    fn untrusted_frontend_address_is_not_used_as_report_target() {
        let state = SharedHeartbeatState::new();
        let handler = ComputeNodeHeartbeatHandler::new(
            test_config(),
            state.clone(),
            Some(Host::new("127.0.0.1").unwrap()),
            EngineReadiness::ready(),
        );
        // master(7) advertises a network_address of 127.0.0.1:9020 (matches) — flip the host.
        let mut hostile = master(7);
        hostile.network_address = types::TNetworkAddress::new("10.0.0.9".to_string(), 9020);

        let result = handler.handle_heartbeat(hostile).unwrap();

        assert_eq!(result.status.status_code, TStatusCode::OK);
        let snapshot = state.snapshot();
        assert_eq!(snapshot.epoch, Some(7));
        // The mismatched address was rejected, so no report target was learned.
        assert!(snapshot.frontend_address.is_none());
    }

    /// Verifies a matching FE host is accepted as the report target.
    #[test]
    fn trusted_frontend_address_is_used_as_report_target() {
        let state = SharedHeartbeatState::new();
        let handler = ComputeNodeHeartbeatHandler::new(
            test_config(),
            state.clone(),
            Some(Host::new("127.0.0.1").unwrap()),
            EngineReadiness::ready(),
        );

        let result = handler.handle_heartbeat(master(7)).unwrap();

        assert_eq!(result.status.status_code, TStatusCode::OK);
        assert_eq!(
            state.snapshot().frontend_address,
            Some(types::TNetworkAddress::new("127.0.0.1".to_string(), 9020))
        );
    }

    /// Verifies the FE-host trust check matches exactly, intersects resolved IPs (so an alias
    /// like `localhost` is accepted against `127.0.0.1`), and rejects only a confident mismatch.
    #[test]
    fn frontend_host_trust_normalizes_ip_aliases() {
        // Exact string match short-circuits without resolution.
        assert!(frontend_host_trusted("127.0.0.1", "127.0.0.1"));
        // Two distinct IP literals always resolve to themselves and never intersect.
        assert!(!frontend_host_trusted("127.0.0.1", "10.0.0.9"));
        // `localhost` resolves to loopback and intersects the literal (or, if the environment
        // cannot resolve it, is treated as inconclusive and trusted).
        assert!(frontend_host_trusted("127.0.0.1", "localhost"));
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
