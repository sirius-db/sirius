use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::ComputeNodeConfig;
use crate::deadlines::{self, Deadline, Deadlines};
use crate::fe_report::ExecReports;
use crate::recent_queries::{self, ENDED_NORMALLY, QueryEnds, RecentQueries, is_normal_end};
/// Remote outputs ship while their fragment runs unless `SIRIUS_CN_STREAM_OUTPUT` is `0`, which
/// ships them from parked output after the run, as before.
fn stream_output_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SIRIUS_CN_STREAM_OUTPUT").as_deref() != Ok("0"))
}

#[cfg(test)]
use crate::fragment_executor::StubExecutor;
use crate::fragment_executor::{
    DrainHandoff, FilterKeys, FilterRun, FragmentExecutor, FragmentRun, KeySource, SenderSlot,
};
use crate::local_exchange::{
    ExchangeKey, LocalExchange, ReadyExchangeInput, ReadyFragment, RemoteBatch, SenderSource,
};
use crate::nixl_chunk::{self, NixlEndpoint, NixlEnvelope, RingDelivery, StreamHop};
use crate::proto::starrocks::{
    PCancelPlanFragmentRequest, PCancelPlanFragmentResult, PExecBatchPlanFragmentsRequest,
    PExecBatchPlanFragmentsResult, PExecPlanFragmentRequest, PExecPlanFragmentResult,
    PFetchDataRequest, PFetchDataResult, PGetFileSchemaRequest, PGetFileSchemaResult,
    PSlotDescriptor, PTransmitChunkParams, PTransmitChunkResult, StatusPb,
    p_internal_service_brpc::PInternalService,
};
use crate::result_encoder::{self, ThriftBinary};
use crate::result_store::{FragmentInstanceId, ResultStore};
use crate::runtime_filters::{
    self, BuildSite, DeferredScan, FILTER_STREAM_BASE, FilterTopology, FragmentOutput,
    FragmentShape, HeldShare, PendingReceiver, RuntimeFilters,
};
use starrocks_plan_translator::runtime_filter::{
    self, BuildDistribution, FilterInput, ProbeKeys, ProbedFilter, SHARE_KEYS, SkipReason,
};
use starrocks_plan_translator::{
    ExchangeInput, PlanTranslator, StreamInputColumn, StreamInputSchema, TranslatedPlan,
};
use starrocks_thrift::{
    data_sinks::{TDataSinkType, TPlanFragmentDestination, TResultSinkType},
    descriptors::TDescriptorTable,
    internal_service::{
        TExecBatchPlanFragmentsParams, TExecPlanFragmentParams, TGetFileSchemaRequest,
    },
    partitions::TPartitionType,
    plan_nodes::{TFileFormatType, TPlanNodeType},
    status_code::TStatusCode,
    types::TNetworkAddress,
};
use thrift::{
    protocol::{TBinaryInputProtocol, TSerializable},
    transport::TBufferChannel,
};
use tracing::{info, instrument, warn};

/// How long a purge's frees are watched for, to log when they finished.
const PURGE_WATCH: Duration = Duration::from_secs(600);

/// How often the deadline watcher checks for queries whose time is up.
const DEADLINE_TICK: Duration = Duration::from_millis(100);

/// How much longer than its query's deadline a `fetch_data` waits, so the deadline's purge, not
/// the wait, is what fails it, with the timeout as its cause.
const DEADLINE_GRACE: Duration = Duration::from_secs(5);

/// Bounds a `fetch_data` wait on a result fragment whose exchange senders never finish, when the
/// query has no timeout. Short under test, so a readiness regression fails a test instead of
/// hanging it for ten minutes.
const RESULT_WAIT: Duration = Duration::from_secs(if cfg!(test) { 5 } else { 600 });

/// Sirius compute-node implementation of StarRocks PInternalService.
///
/// Plan-fragment translation is the first implemented RPC path; future
/// compute-node tasks should land here behind the generated service facade.
#[derive(Clone, Debug)]
pub(crate) struct SiriusComputeNodeService {
    /// Reusable StarRocks thrift-to-Substrait fragment translator.
    translator: PlanTranslator,
    /// This CN's NIXL side, when it has one: serves peers' `transmit_chunk` requests and ships
    /// output to remote destinations. Declared before `executor` because fields drop in order:
    /// whoever drops the last clone, the transport deregisters the engine's slab before the
    /// engine frees it.
    nixl: Option<Arc<dyn NixlEndpoint>>,
    /// Runs translated fragments, returning a result fragment's Arrow batches or parking a sender
    /// fragment's output. Production injects the GPU-backed `SiriusEngine` (via
    /// [`with_executor`](Self::with_executor)); tests use a stub.
    executor: Arc<dyn FragmentExecutor>,
    /// Buffers executed-fragment results for FE `fetch_data` collection. Shared across BRPC
    /// connections so a `fetch_data` poll sees what an `exec_plan_fragment` buffered.
    results: Arc<ResultStore>,
    /// Descriptor tables retained for StarRocks's per-query cache protocol.
    /// Descriptor tables retained for StarRocks's per-query cache protocol, by query. Dropped
    /// when the query ends here; otherwise kept for the window that ended queries are.
    descriptor_tables: Arc<Mutex<RecentQueries<TDescriptorTable>>>,
    /// Exchange rendezvous: receivers wait here until every sender has parked its output.
    exchanges: Arc<LocalExchange>,
    /// This CN's advertised brpc endpoint, with the host resolved to an IP because the FE
    /// addresses sink destinations by IP (`ComputeNode.getBrpcIpAddress`). A destination is local
    /// only when IP and port both match, so two CNs on one host see each other as remote.
    brpc_address: TNetworkAddress,
    /// Runtime filters this CN builds from broadcast joins or holds a share of, and the scans
    /// waiting for them.
    filters: Arc<RuntimeFilters>,
    /// How long a scan waits for its runtime filters before it runs unfiltered.
    filter_wait: Duration,
    /// The final `reportExecStatus` each dispatched instance still owes the FE.
    reports: Arc<ExecReports>,
    /// Survey mode (`SIRIUS_CN_TRANSLATE_ONLY`): accept and translate every fragment, run none.
    translate_only: bool,
    /// How each recent query ended on this CN, shared with the exchange and the reports.
    ends: Arc<QueryEnds>,
    /// Each query's deadline on this CN, from its query timeout.
    deadlines: Arc<Deadlines>,
    /// Runs a sender fragment inside its `exec_plan_fragment` call, as before senders ran on
    /// their own thread (`SIRIUS_CN_INLINE_SENDERS=1`).
    inline_senders: bool,
    /// Sender fragments accepted and not yet finished running off their dispatch call.
    senders_running: Arc<AtomicUsize>,
}

/// At most this many queries' descriptor tables are cached; past it the oldest is dropped.
const CACHED_DESCRIPTOR_TABLES: usize = 4096;

/// A cancel from the FE, as the CN acts on it.
#[derive(Debug)]
struct Cancel {
    /// A normal end (QUERY_FINISHED, LIMIT_REACH): the query succeeded.
    finished: bool,
    /// The FE's reason, for logs.
    reason: &'static str,
    /// What the query's refusals and reports say: the FE's error message when it sent one.
    error: String,
}

impl Cancel {
    fn of(request: &PCancelPlanFragmentRequest) -> Self {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        let reason = request
            .cancel_reason
            .and_then(|reason| Reason::try_from(reason).ok());
        let finished = matches!(reason, Some(Reason::QueryFinished | Reason::LimitReach));
        if finished {
            let reason = reason.map_or("UNSET", |reason| reason.as_str_name());
            return Self {
                finished,
                reason,
                error: format!("{ENDED_NORMALLY} ({reason})"),
            };
        }
        let described = match reason {
            Some(Reason::QueryFinished | Reason::LimitReach) => unreachable!("handled above"),
            Some(Reason::UserCancel) => "the user cancelled the query",
            Some(Reason::Timeout) => "the query timed out",
            Some(Reason::InternalError) | None => "the FE cancelled the query",
        };
        let message = request
            .error_message
            .as_deref()
            .filter(|message| !message.is_empty())
            .unwrap_or(described);
        Self {
            finished,
            reason: reason.map_or("UNSET", |reason| reason.as_str_name()),
            error: format!("query cancelled: {message}"),
        }
    }
}

/// The inputs of a receiver about to run, by exchange: runtime filter keys read in place before
/// it consumes them.
type HeldKeys = HashMap<ExchangeKey, Vec<KeySource>>;

/// Where a deferred scan's filter keys stand.
enum FilterKeysState {
    /// Some are still to arrive.
    Waiting,
    /// All here: where they sit, and whether they are the keys themselves or only bounds.
    Ready {
        sources: Vec<KeySource>,
        exact: bool,
    },
    /// They never will be usable; the scan runs without this filter.
    Unusable {
        reason: &'static str,
        detail: String,
    },
}

/// The runtime filters a fragment run applies, and the plan to run instead if their keys cannot
/// be copied.
#[derive(Default)]
struct FilterPlan<'a> {
    filters: Vec<FilterRun>,
    fallback: Option<&'a TranslatedPlan>,
}

/// A sender fragment's outputs, as its sink and destinations describe them.
struct SenderPlan {
    broadcast: bool,
    hash_keys: Vec<usize>,
    outputs: Vec<SenderSlot>,
    /// Destinations on this CN, whose outputs park for their receivers.
    local: Vec<SenderSlot>,
    remote: Vec<(SenderSlot, SocketAddr)>,
    /// Which of `outputs` stream to `remote`, in its order.
    streams: Vec<usize>,
    sender_id: i32,
}

/// Why a fragment failed, and whether its remote hops already ended. A hop the NIXL transport
/// took ends with EOS or a failure frame; one it never took still needs a failure frame, or its
/// receiver waits for an EOS that never comes.
#[derive(Debug)]
struct FragmentFailure {
    error: String,
    hops_ended: bool,
}

impl FragmentFailure {
    /// A failure after every remote hop was handed to the transport.
    fn after_hops(error: String) -> Self {
        Self {
            error,
            hops_ended: true,
        }
    }
}

impl From<String> for FragmentFailure {
    fn from(error: String) -> Self {
        Self {
            error,
            hops_ended: false,
        }
    }
}

impl SiriusComputeNodeService {
    /// Test-only constructor with the placeholder [`StubExecutor`]. Production injects a real
    /// executor via [`with_executor`](Self::with_executor).
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_executor(Arc::new(StubExecutor), &ComputeNodeConfig::default(), None)
    }

    /// Builds the service for the CN `compute_node` advertises, with a caller-provided fragment
    /// executor (e.g. the GPU-backed `SiriusEngine`) shared across BRPC connections via the `Arc`.
    pub(crate) fn with_executor(
        executor: Arc<dyn FragmentExecutor>,
        compute_node: &ComputeNodeConfig,
        nixl: Option<Arc<dyn NixlEndpoint>>,
    ) -> Self {
        let ends = Arc::new(QueryEnds::default());
        let service = Self {
            translator: PlanTranslator::new(),
            executor,
            results: Arc::new(ResultStore::default()),
            descriptor_tables: Arc::new(Mutex::new(RecentQueries::new(
                recent_queries::REMEMBER_FOR,
                CACHED_DESCRIPTOR_TABLES,
            ))),
            exchanges: Arc::new(LocalExchange::new(Arc::clone(&ends))),
            brpc_address: TNetworkAddress::new(
                resolve_ip(compute_node.advertise_host.as_str()),
                i32::from(compute_node.brpc_port),
            ),
            nixl,
            filters: Arc::new(RuntimeFilters::default()),
            filter_wait: runtime_filters::wait_limit(),
            reports: Arc::new(ExecReports::new(None, Arc::clone(&ends))),
            translate_only: std::env::var_os("SIRIUS_CN_TRANSLATE_ONLY").is_some(),
            ends,
            deadlines: Arc::default(),
            inline_senders: std::env::var("SIRIUS_CN_INLINE_SENDERS").is_ok_and(|v| v == "1"),
            senders_running: Arc::default(),
        };
        if let Some(nixl) = &service.nixl {
            nixl.set_delivery(Arc::new(service.clone()));
        }
        service
    }

    /// Reports instances' ends only to coordinators on `frontend_host`, the configured FE.
    pub(crate) fn reporting_to(mut self, frontend_host: Option<crate::Host>) -> Self {
        self.reports = Arc::new(ExecReports::new(frontend_host, Arc::clone(&self.ends)));
        self
    }
}

impl RingDelivery for SiriusComputeNodeService {
    /// Runs the receivers the batch completes on their own thread. The ring transport sends the
    /// lane's last credit for the batch only after this returns, so the sender's EOS comes after it.
    fn deliver(
        &self,
        request: &PTransmitChunkParams,
        token: u64,
        rows: u64,
        names: &[String],
    ) -> std::result::Result<(), String> {
        match self.push_packed(request, token, rows, names.to_vec())? {
            Some(Some(ready)) => self.drain_ready_async(ready),
            Some(None) => {}
            // A runtime filter share for a scan that no longer waits for it.
            None => {
                if let Some(nixl) = self.nixl.as_ref().filter(|_| token != 0) {
                    nixl.release(token);
                }
            }
        }
        Ok(())
    }
}

/// Resolves `host` the way the FE does (Java prefers IPv4), or returns it unchanged if it doesn't
/// resolve; an IP literal comes back as is.
fn resolve_ip(host: &str) -> String {
    use std::net::ToSocketAddrs;
    let Ok(addrs) = (host, 0).to_socket_addrs() else {
        return host.to_string();
    };
    let addrs: Vec<_> = addrs.map(|addr| addr.ip()).collect();
    addrs
        .iter()
        .find(|ip| ip.is_ipv4())
        .or(addrs.first())
        .map_or_else(|| host.to_string(), ToString::to_string)
}

impl PInternalService for SiriusComputeNodeService {
    /// Handles a single FE-dispatched plan fragment thrift attachment. A fragment fed by an
    /// exchange returns OK once registered and runs when its senders have parked. A
    /// DATA_STREAM_SINK returns OK once accepted and runs on its own thread, parking its output
    /// for its receivers and reporting how it ended. A RESULT_SINK runs now and buffers its rows
    /// for `fetch_data`.
    #[instrument(skip_all)]
    async fn exec_plan_fragment(
        &self,
        request: PExecPlanFragmentRequest,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PExecPlanFragmentResult>, crate::prpc::Error> {
        // Translate + execute on a blocking worker, not the BRPC current-thread runtime: a real GPU
        // executor blocks for the whole query, so running it inline would stall fetch_data,
        // connection cleanup, and shutdown cancellation until it returns.
        let protocol = request.attachment_protocol;
        let service = self.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            service.exec_single_attachment(protocol.as_deref(), &attachment)
        })
        .await;
        let status = match outcome {
            Ok(Ok(())) => Self::ok_status(),
            Ok(Err(status)) => status,
            Err(join_err) => {
                Self::internal_error(format!("fragment execution task panicked: {join_err}"))
            }
        };
        Ok(Self::exec_plan_result(status).into())
    }

    /// Handles FE batch fragment dispatch: processes every per-instance fragment as
    /// `exec_plan_fragment` does.
    #[instrument(skip_all)]
    async fn exec_batch_plan_fragments(
        &self,
        request: PExecBatchPlanFragmentsRequest,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PExecBatchPlanFragmentsResult>, crate::prpc::Error> {
        // Like `exec_plan_fragment`, an instance can run a fragment on the GPU, so offload to a
        // blocking worker rather than blocking the BRPC current-thread runtime.
        let protocol = request.attachment_protocol;
        let service = self.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            service.translate_batch_attachment(protocol.as_deref(), &attachment)
        })
        .await;
        let status = match outcome {
            Ok(Ok(())) => Self::ok_status(),
            Ok(Err(status)) => status,
            Err(join_err) => Self::internal_error(format!(
                "batch fragment execution task panicked: {join_err}"
            )),
        };
        Ok(PExecBatchPlanFragmentsResult {
            status: Some(status),
        }
        .into())
    }

    /// Returns buffered fragment results to the FE, which polls this until end-of-stream. The
    /// serialized `TResultBatch` rows ride in the BRPC response attachment.
    #[instrument(skip_all)]
    async fn fetch_data(
        &self,
        request: PFetchDataRequest,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PFetchDataResult>, crate::prpc::Error> {
        let id = FragmentInstanceId::from(&request.finst_id);
        // An unknown id is an error, not EOS: it means this CN never buffered a result for the
        // fragment the FE is polling (wrong id, or a dispatch/result-sink path that did not run),
        // and StarRocks treats a missing result buffer as a failure rather than an empty result.
        // A result fragment still waiting on its exchange inputs blocks, so wait off the runtime.
        let results = self.results.clone();
        let wait = self
            .deadlines
            .remaining(id)
            .map_or(RESULT_WAIT, |left| left + DEADLINE_GRACE);
        let outcome = tokio::task::spawn_blocking(move || results.take_next(id, wait))
            .await
            .unwrap_or_else(|join_err| Err(format!("fetch_data wait task panicked: {join_err}")));
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                return Ok(Self::fetch_data_result(Self::internal_error(err), 0, true).into());
            }
        };
        if outcome.eos {
            self.ends.delivered(id);
            self.deadlines.end(id);
            self.forget_descriptor_table(id);
        }
        match outcome.batch {
            Some(batch) => match batch.to_binary() {
                Ok(bytes) => Ok(crate::prpc::Reply::with_attachment(
                    Self::fetch_data_result(Self::ok_status(), outcome.packet_seq, outcome.eos),
                    bytes,
                )),
                Err(err) => Ok(Self::fetch_data_result(
                    Self::internal_error(err),
                    outcome.packet_seq,
                    true,
                )
                .into()),
            },
            None => Ok(
                Self::fetch_data_result(Self::ok_status(), outcome.packet_seq, outcome.eos).into(),
            ),
        }
    }

    /// Infers the schema of the FILES() target so the FE can resolve the table function.
    #[instrument(skip_all)]
    async fn get_file_schema(
        &self,
        _request: PGetFileSchemaRequest,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PGetFileSchemaResult>, crate::prpc::Error> {
        let result = match Self::file_schema_from_attachment(&attachment).await {
            Ok(schema) => PGetFileSchemaResult {
                status: Self::ok_status(),
                schema,
            },
            Err(err) => PGetFileSchemaResult {
                status: Self::internal_error(err),
                schema: Vec::new(),
            },
        };
        Ok(result.into())
    }

    /// Handles the FE's cancel of a failed or finished query. The FE sends one per instance it
    /// still thinks is running, each naming the query, so the first purges the whole query and
    /// the rest only count. A normal end (QUERY_FINISHED, LIMIT_REACH) releases what the query
    /// holds quietly; any other reason fails it. Answers at once: dropping parked output waits on
    /// the engine thread, which may be running another fragment.
    #[instrument(skip_all)]
    async fn cancel_plan_fragment(
        &self,
        request: PCancelPlanFragmentRequest,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PCancelPlanFragmentResult>, crate::prpc::Error> {
        let query = request.query_id.as_ref().map_or_else(
            || FragmentInstanceId::from(&request.finst_id),
            FragmentInstanceId::from,
        );
        let cancel = Cancel::of(&request);
        // Recorded before replying, so a fragment or frame of the query that arrives while the
        // purge below is still queued is already refused.
        let cancels = self.ends.cancel(query, &cancel.error);
        if cancels > 1 {
            info!(%query, cancels, reason = cancel.reason, "the FE cancelled a query again");
            return Ok(PCancelPlanFragmentResult {
                status: Self::ok_status(),
            }
            .into());
        }
        if cancel.finished {
            info!(%query, reason = cancel.reason, "the FE ended a finished query");
        } else {
            warn!(%query, reason = cancel.reason, error = cancel.error, "the FE cancelled a failed query");
        }
        self.reports.cancel_query(query, &cancel.error);
        // It polls no more: its delivered result slots can go now.
        self.results.forget_delivered(query);
        let service = self.clone();
        tokio::task::spawn_blocking(move || {
            if cancel.finished {
                service.release_query(query, &cancel.error);
            } else {
                service.fail_and_purge(query, &cancel.error);
            }
            service.log_leak_counters("cancel");
        });
        Ok(PCancelPlanFragmentResult {
            status: Self::ok_status(),
        }
        .into())
    }

    #[instrument(skip_all)]
    async fn transmit_chunk(
        &self,
        request: PTransmitChunkParams,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PTransmitChunkResult>, crate::prpc::Error> {
        // A lane's open and a delivered batch take the exchanges' locks, which the engine thread
        // can hold while it synchronizes a stream; on the brpc runtime that would stall every RPC.
        // A refusal's leak counters read the same exchanges, so they are logged there too.
        let service = self.clone();
        let (status, reply) = tokio::task::spawn_blocking(move || {
            match service.handle_nixl_chunk(&request, &attachment) {
                Ok(reply) => (Self::ok_status(), reply),
                // A frame of a query that already failed here is refused with that failure's
                // cause, as CANCELLED, so the sender fails with the cause rather than with this
                // refusal.
                Err(err) => match request
                    .finst_id
                    .as_ref()
                    .and_then(|id| service.exchanges.failure(FragmentInstanceId::from(id)))
                {
                    Some(cause) => {
                        // The refused frame's buffers were just freed, after its query's purge
                        // last logged what this CN holds.
                        service.log_leak_counters("refused frame");
                        (Self::cancelled(cause), Vec::new())
                    }
                    None => (Self::internal_error(err), Vec::new()),
                },
            }
        })
        .await
        .unwrap_or_else(|join_err| {
            let err = format!("transmit_chunk task panicked: {join_err}");
            (Self::internal_error(err), Vec::new())
        });
        let result = PTransmitChunkResult {
            status: Some(status),
            receive_timestamp: None,
            receiver_post_process_time: None,
        };
        Ok(crate::prpc::Reply::with_attachment(result, reply))
    }
}

impl SiriusComputeNodeService {
    /// Deserializes one binary-thrift TExecPlanFragmentParams attachment and processes it.
    fn exec_single_attachment(
        &self,
        protocol: Option<&str>,
        attachment: &[u8],
    ) -> std::result::Result<(), StatusPb> {
        Self::ensure_binary_protocol(protocol).map_err(Self::internal_error)?;
        let params =
            Self::deserialize_binary::<TExecPlanFragmentParams>(attachment).map_err(|err| {
                Self::internal_error(format!(
                    "failed to deserialize TExecPlanFragmentParams: {err}"
                ))
            })?;
        self.process_fragment(&params)
            .map_err(|err| self.dispatch_error(&params, err))
    }

    /// The status a failed dispatch of `params` replies with. Once the FE ended the query (it
    /// cancelled it, or this CN delivered its last row), the refusal is CANCELLED, which the FE
    /// ignores after a successful query, rather than an error that would fail or retry a query
    /// that already returned its rows.
    fn dispatch_error(&self, params: &TExecPlanFragmentParams, err: String) -> StatusPb {
        match Self::query_id(params) {
            Some(query) if self.ends.ended_by_fe(query) => Self::cancelled(err),
            _ => Self::internal_error(err),
        }
    }

    /// Runs one fragment, or registers it as a receiver that runs once all its exchange senders
    /// have parked their output. A RESULT_SINK buffers its rows for later `fetch_data`. Shared by
    /// single and batch dispatch so both paths produce fetchable results for a RESULT_SINK
    /// instance.
    fn process_fragment(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<(), String> {
        // A fragment of a query that already ended here, failed or cancelled, never runs: a cancel
        // can overtake its dispatch. Its reply carries the refusal, so it owes no report.
        if let Some(instance) = Self::fragment_instance_id(params)
            && let Some(cause) = self.exchanges.failure(instance)
        {
            info!(%instance, cause, "refusing a fragment of a query that already ended");
            return Err(format!(
                "{cause} (fragment instance {instance} came after its query ended)"
            ));
        }
        self.start_deadline(params);
        self.reports.expect(params);
        let run = || {
            self.run_or_register(params)
                .map_err(|failure| {
                    // The dispatch reply tells the FE.
                    self.reports.settled_by_reply(params);
                    self.end_hops(params, failure)
                })
                .and_then(|ready| self.drain_ready(ready))
        };
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|panic| {
                let error = format!("a fragment panicked: {}", panic_message(panic.as_ref()));
                warn!(error, "failing the panicked fragment's query");
                // The dispatch reply carries the panic; the query's other instances here report
                // it, since whatever ran them unwound.
                self.reports.settled_by_reply(params);
                if let Some(query) = Self::query_id(params) {
                    self.reports.fail_query(query, &error);
                }
                Err(error)
            });
        if let Err(err) = &result
            && let Some(query) = Self::query_id(params)
        {
            self.fail_and_purge(query, err);
        }
        self.log_leak_counters("fragment");
        result
    }

    /// Runs a leaf fragment (a sender on its own thread), or registers a receiver to run once its
    /// senders finish. Returns the receivers that are ready to run.
    fn run_or_register(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<Vec<ReadyFragment>, FragmentFailure> {
        let params = self.resolve_descriptor_table(params)?;
        let dump_seq = Self::dump_fragment(&params);
        // Survey mode: accept every fragment so the FE dispatches (and we dump) the whole
        // plan even when translation fails. Queries still fail at fetch_data.
        if self.translate_only {
            if let Err(err) = self.translate_fragment_logged(&params, &[], dump_seq) {
                tracing::warn!(error = %err, "translate-only mode: accepting untranslatable fragment");
            }
            // Never run, so done as far as the FE is concerned.
            self.reports.finish(&params, Ok(()));
            return Ok(Vec::new());
        }
        if runtime_filters::enabled() {
            Self::log_planned_skips(&params);
            if let (Some(query), Some(topology)) = (
                Self::query_id(&params),
                params
                    .params
                    .as_ref()
                    .and_then(|exec| exec.runtime_filter_params.as_ref()),
            ) {
                self.filters.record_topology(query, topology);
            }
            // After the topology: a merge node's own holders find their probers there.
            self.release_unbuilt_probers(&params);
        }
        let expected_senders = Self::receiver_exchanges(&params)?;
        if !expected_senders.is_empty() {
            let exec = params
                .params
                .as_ref()
                .ok_or_else(|| "exchange receiver is missing execution params".to_string())?;
            let id = FragmentInstanceId::from(&exec.fragment_instance_id);
            let query = FragmentInstanceId::from(&exec.query_id);
            if Self::is_mysql_result_sink(&params)? {
                self.results.reserve(id, query);
            }
            if runtime_filters::enabled() {
                match runtime_filter::built_filters(&params) {
                    Ok(built) if !built.is_empty() => {
                        info!(receiver = %id, filters = ?built, "receiver builds runtime filters");
                        let (broadcast, shares): (Vec<_>, Vec<_>) =
                            built.into_iter().partition(|filter| {
                                filter.distribution == BuildDistribution::Broadcast
                            });
                        // Its probers refuse the filter too (`share_site`), so none waits.
                        let (shares, unshareable): (Vec<_>, Vec<_>) =
                            shares.into_iter().partition(|filter| {
                                runtime_filters::shareable_key(&filter.column_type)
                            });
                        for filter in unshareable {
                            runtime_filters::log_skipped(
                                Some(query),
                                Some(id),
                                filter.filter_id,
                                Some(filter.join_node_id),
                                "key_type",
                                &filter.column_type,
                            );
                            if let Some(merge_node) = filter.merge_node {
                                self.release_probers(query, filter.filter_id, merge_node);
                            }
                        }
                        self.filters.record_builds(id, broadcast);
                        self.filters
                            .record_shares(query, id, exec.sender_id.unwrap_or(0), shares);
                    }
                    Ok(_) => {}
                    Err(err) => warn!(error = %err, "ignoring a receiver's runtime filters"),
                }
                // A receiver never waits for a partitioned join's shares.
                self.close_filter_exchanges(id, &params, &[]);
                self.filters.record_shape(
                    query,
                    FragmentShape {
                        exchanges: expected_senders
                            .iter()
                            .map(|(node_id, _)| *node_id)
                            .collect(),
                        sink: Self::fragment_output(&params),
                    },
                );
                let filters = self.receiver_filter_sites(&params, query, id);
                if !filters.is_empty() {
                    info!(
                        instance = %id,
                        filters = ?filters.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                        "deferring a scan until its runtime filters arrive"
                    );
                    self.filters.defer_receiver(id, PendingReceiver { filters });
                }
            }
            let ready = self
                .exchanges
                .register_receiver(id, expected_senders, params)?;
            if runtime_filters::enabled() {
                // Its build exchange may be complete already: its share can go.
                self.dispatch_filtered_scans();
            }
            return Ok(ready.into_iter().collect());
        }
        if self.defer_for_filters(&params) {
            // A cancel that purged the query after the check at dispatch would leave the scan
            // waiting, then run it unfiltered; take it back instead.
            if let Some(instance) = Self::fragment_instance_id(&params)
                && let Some(cause) = self.exchanges.failure(instance)
                && self.filters.take(instance).is_some()
            {
                return Err(format!(
                    "{cause} (fragment instance {instance} came after its query ended)"
                )
                .into());
            }
            // The scan runs once its filters' keys arrived; they may be here already.
            self.dispatch_filtered_scans();
            return Ok(Vec::new());
        }
        let translated = self.translate_fragment_logged(&params, &[], dump_seq)?;
        if !self.inline_senders
            && matches!(Self::fragment_output(&params), FragmentOutput::Exchange(_))
        {
            // What can fail before the run fails the dispatch, in its reply.
            self.sender_plan(&params, &translated)?;
            self.run_sender_async(params, translated);
            return Ok(Vec::new());
        }
        let ready = self.execute_fragment(
            &params,
            &translated,
            Vec::new(),
            Vec::new(),
            FilterPlan::default(),
        )?;
        self.reports.finish(&params, Ok(()));
        Ok(ready)
    }

    /// Logs the runtime filters `params`' fragment builds or probes that the plan already rules
    /// out: a join filter this CN can't build, and a probe target no CN filters.
    fn log_planned_skips(params: &TExecPlanFragmentParams) {
        let (query, instance) = (Self::query_id(params), Self::fragment_instance_id(params));
        let builds = runtime_filter::skipped_builds(params).unwrap_or_else(|err| {
            warn!(error = %err, "cannot read a fragment's runtime filters");
            Vec::new()
        });
        for skip in builds
            .into_iter()
            .chain(runtime_filter::skipped_probes(params))
        {
            runtime_filters::log_skipped(
                query,
                instance,
                skip.filter_id,
                Some(skip.node_id),
                skip.reason.as_str(),
                "",
            );
        }
    }

    /// Tells the probers of every partitioned join filter `params`' fragment builds but no CN can
    /// that this holder's share won't come. They would otherwise wait for it until the limit.
    fn release_unbuilt_probers(&self, params: &TExecPlanFragmentParams) {
        let Some(query) = Self::query_id(params) else {
            return;
        };
        let unbuilt = runtime_filter::unbuilt_shares(params).unwrap_or_else(|err| {
            warn!(error = %err, "cannot read a fragment's runtime filters");
            Vec::new()
        });
        for filter in unbuilt {
            self.release_probers(query, filter.filter_id, filter.merge_node);
        }
    }

    /// Tells every prober of `query`'s partitioned join filter `filter_id`, found through its
    /// merge node, that this CN's share won't come. Runs off the caller's thread: the merge node
    /// may be another CN.
    fn release_probers(
        &self,
        query: FragmentInstanceId,
        filter_id: i32,
        merge_node: TNetworkAddress,
    ) {
        let service = self.clone();
        let spawned = std::thread::Builder::new()
            .name("filter-release".to_string())
            .spawn(move || {
                match service.filter_topology(query, &merge_node, filter_id) {
                    Ok(topology) => {
                        info!(
                            %query,
                            filter_id,
                            probers = topology.probers.len(),
                            "releasing a runtime filter's probers: no share comes from this CN"
                        );
                        service.abandon_share(filter_id, &topology.probers);
                    }
                    // Without the probers there is no one to tell: they wait out their limit.
                    Err(err) => {
                        warn!(%query, filter_id, error = %err, "cannot release a runtime filter's probers");
                    }
                }
            });
        if let Err(err) = spawned {
            warn!(%query, filter_id, error = %err, "cannot release a runtime filter's probers");
        }
    }

    /// Defers a leaf fragment whose scans probe runtime filters it can wait for: a broadcast
    /// join's that a receiver on this CN builds, and a partitioned join's, whose shares every
    /// build instance sends here. A timer runs it unfiltered if its filters take too long.
    fn defer_for_filters(&self, params: &TExecPlanFragmentParams) -> bool {
        if !runtime_filters::enabled() {
            return false;
        }
        let (Some(query), Some(instance)) =
            (Self::query_id(params), Self::fragment_instance_id(params))
        else {
            return false;
        };
        let probes = runtime_filter::probed_filters(params);
        let broadcast: Vec<i32> = probes
            .iter()
            .filter(|probe| probe.distribution == BuildDistribution::Broadcast)
            .map(|probe| probe.filter_id)
            .collect();
        let mut sites = self.filters.sites(query, &broadcast);
        for probe in &probes {
            if sites.iter().any(|(id, _)| *id == probe.filter_id) {
                continue;
            }
            let skip = |reason: &str, detail: &str| {
                runtime_filters::log_skipped(
                    Some(query),
                    Some(instance),
                    probe.filter_id,
                    Some(probe.scan_node_id),
                    reason,
                    detail,
                );
            };
            match probe.distribution {
                // Built by a receiver on another CN only.
                BuildDistribution::Broadcast => skip("not_built_on_this_cn", ""),
                BuildDistribution::Partitioned => match self.share_site(query, instance, probe) {
                    Ok(site) => sites.push((probe.filter_id, site)),
                    Err((reason, detail)) => skip(reason, &detail),
                },
                BuildDistribution::Other => skip("not_broadcast", ""),
            }
        }
        let waiting: Vec<i32> = sites.iter().map(|(id, _)| *id).collect();
        self.close_filter_exchanges(instance, params, &waiting);
        if sites.is_empty() {
            return false;
        }
        info!(
            %instance,
            filters = ?waiting,
            "deferring a scan until its runtime filters arrive"
        );
        let deferred = self.filters.defer(
            instance,
            DeferredScan {
                params: params.clone(),
                filters: sites,
                deferred_at: std::time::Instant::now(),
                ready: None,
            },
        );
        if deferred.is_err() {
            // Its query was purged meanwhile: it never runs.
            return true;
        }
        self.start_filter_timer(instance);
        true
    }

    /// Runs deferred scan `instance` unfiltered if it still waits once the wait limit is over.
    fn start_filter_timer(&self, instance: FragmentInstanceId) {
        let service = self.clone();
        let wait = self.filter_wait;
        let spawned = std::thread::Builder::new()
            .name("runtime-filter-timer".to_string())
            .spawn(move || {
                std::thread::sleep(wait);
                if let Some(scan) = service.filters.take(instance) {
                    warn!(%instance, ?wait, "runtime filters did not arrive in time; scanning unfiltered");
                    service.run_deferred(scan, false, None);
                }
            });
        if let Err(err) = spawned {
            self.timer_not_started(instance, &err.to_string());
        }
    }

    /// Runs deferred scan `instance` unfiltered now: without its timer nothing would run it if
    /// its filters never came, and a waiting receiver would hold its inputs for good.
    fn timer_not_started(&self, instance: FragmentInstanceId, error: &str) {
        if let Some(scan) = self.filters.take(instance) {
            warn!(%instance, error, "cannot time a runtime filter wait; scanning unfiltered");
            self.run_deferred(scan, false, None);
        }
    }

    /// The broadcast join filters the scans of receiver `instance` can wait for: built by
    /// another receiver on this CN, from a build exchange that doesn't depend on this fragment's
    /// output. Waiting on a build that needs this fragment's output would hold both until the
    /// wait limit. Logs why each other filter isn't waited for.
    fn receiver_filter_sites(
        &self,
        params: &TExecPlanFragmentParams,
        query: FragmentInstanceId,
        instance: FragmentInstanceId,
    ) -> Vec<(i32, BuildSite)> {
        let sink = Self::fragment_output(params);
        let mut sites: Vec<(i32, BuildSite)> = Vec::new();
        for probe in runtime_filter::probed_filters(params) {
            if sites.iter().any(|(id, _)| *id == probe.filter_id) {
                continue;
            }
            let skip = |reason: &str, detail: &str| {
                runtime_filters::log_skipped(
                    Some(query),
                    Some(instance),
                    probe.filter_id,
                    Some(probe.scan_node_id),
                    reason,
                    detail,
                );
            };
            match probe.distribution {
                BuildDistribution::Broadcast => {
                    let Some((_, site)) = self.filters.sites(query, &[probe.filter_id]).pop()
                    else {
                        skip("not_built_on_this_cn", "");
                        continue;
                    };
                    match self.filters.independent(query, sink, site.key.node_id) {
                        Ok(()) => sites.push((probe.filter_id, site)),
                        Err(detail) => skip("deadlock_unproven", &detail),
                    }
                }
                BuildDistribution::Partitioned => skip(
                    SkipReason::NonLeafFragment.as_str(),
                    "a partitioned join's filter",
                ),
                BuildDistribution::Other => skip("not_broadcast", ""),
            }
        }
        sites
    }

    /// Where `params`' fragment sends its output.
    fn fragment_output(params: &TExecPlanFragmentParams) -> FragmentOutput {
        let Some(sink) = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
        else {
            return FragmentOutput::Unknown;
        };
        match (sink.type_, sink.stream_sink.as_ref()) {
            (TDataSinkType::DATA_STREAM_SINK, Some(stream)) => {
                FragmentOutput::Exchange(stream.dest_node_id)
            }
            (TDataSinkType::RESULT_SINK, _) => FragmentOutput::Result,
            _ => FragmentOutput::Unknown,
        }
    }

    /// Holds back ready receiver `ready` if it waits for runtime filters: it runs once their
    /// keys are here, or unfiltered at the wait limit. Returns it when it doesn't wait.
    fn park_for_filters(&self, ready: ReadyFragment) -> Option<ReadyFragment> {
        let instance = Self::fragment_instance_id(&ready.params)?;
        let Some(pending) = self.filters.take_pending(instance) else {
            return Some(ready);
        };
        let scan = DeferredScan {
            params: ready.params.clone(),
            filters: pending.filters,
            deferred_at: std::time::Instant::now(),
            ready: Some(ready),
        };
        match self.filters.defer(instance, scan) {
            Ok(()) => {
                self.start_filter_timer(instance);
                // Its filters' keys may be here already.
                self.dispatch_filtered_scans();
            }
            // Its query was purged meanwhile: free its inputs.
            Err(scan) => self.release_inputs(scan.ready),
        }
        None
    }

    /// Frees a ready receiver's inputs it will never run on.
    fn release_inputs(&self, ready: Option<ReadyFragment>) {
        if let Some(ready) = ready {
            drop(self.take_ready_inputs(ready.inputs));
        }
    }

    /// Where scan `instance` receives the shares of the partitioned join filter `probe`, and how
    /// many make it: from the plan's layout, or else from the filter's merge node.
    fn share_site(
        &self,
        query: FragmentInstanceId,
        instance: FragmentInstanceId,
        probe: &ProbedFilter,
    ) -> std::result::Result<BuildSite, (&'static str, String)> {
        let column_type = probe
            .key_type
            .clone()
            .ok_or(("key_type", "the probe key has no engine type".to_string()))?;
        if !runtime_filters::shareable_key(&column_type) {
            // No holder sends a share of a key it can't read.
            return Err(("key_type", column_type));
        }
        // Holders find their probers through the merge node: without one, no share ever comes.
        let merge_node = probe
            .merge_node
            .as_ref()
            .ok_or(("no_merge_node", String::new()))?;
        let shares = match probe.shares {
            Some(shares) => shares,
            None => {
                self.filter_topology(query, merge_node, probe.filter_id)
                    .map_err(|err| ("shares_unknown", err))?
                    .shares
            }
        };
        Ok(BuildSite {
            key: runtime_filters::filter_exchange(instance, probe.filter_id),
            column: 0,
            column_type,
            shares: Some(shares),
        })
    }

    /// Who probes `query`'s partitioned join filter `filter_id`, asked of its merge node.
    fn filter_topology(
        &self,
        query: FragmentInstanceId,
        merge_node: &TNetworkAddress,
        filter_id: i32,
    ) -> std::result::Result<FilterTopology, String> {
        let topology = match self.peer_of(merge_node)? {
            None => self.filters.topology(query, filter_id),
            Some(peer) => {
                let reply = self
                    .nixl
                    .as_ref()
                    .ok_or("this CN has no NIXL transport to reach the merge node")?
                    .control(peer, &NixlEnvelope::Topology { query, filter_id })?;
                nixl_chunk::decode_topology(&reply)?
            }
        };
        topology.ok_or_else(|| {
            format!(
                "merge node {}:{} has no topology for filter {filter_id}",
                merge_node.hostname, merge_node.port
            )
        })
    }

    /// Marks the filter exchanges of `instance` for the partitioned join filters its fragment
    /// probes as read by no scan, except `waiting`'s, and frees any share already there.
    fn close_filter_exchanges(
        &self,
        instance: FragmentInstanceId,
        params: &TExecPlanFragmentParams,
        waiting: &[i32],
    ) {
        let probed = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.plan.as_ref())
            .into_iter()
            .flat_map(|plan| &plan.nodes)
            .flat_map(|node| node.probe_runtime_filters.iter().flatten())
            .filter(|filter| BuildDistribution::of(filter) == BuildDistribution::Partitioned)
            .filter_map(|filter| filter.filter_id)
            .filter(|filter_id| !waiting.contains(filter_id));
        for filter_id in probed {
            self.drop_shares(runtime_filters::filter_exchange(instance, filter_id));
        }
    }

    /// Frees the shares at filter exchange `key`, and drops any that arrive later.
    fn drop_shares(&self, key: ExchangeKey) {
        let dropped = self.exchanges.close_shares(key);
        if let Some(nixl) = &self.nixl {
            for &token in &dropped.tokens {
                nixl.release(token);
            }
        }
        for &slot in &dropped.slots {
            if let Err(err) = self.executor.drop_parked(slot) {
                warn!(?slot, error = %err, "failed to drop a runtime filter share");
            }
        }
    }

    /// Where a deferred scan's filter keys stand: still arriving, all here (and whether they are
    /// the keys themselves or only bounds), or never usable. `held` are a ready receiver's
    /// inputs, read before it consumes them.
    fn filter_keys(&self, site: &BuildSite, held: Option<&HeldKeys>) -> FilterKeysState {
        let Some(expected) = site.shares else {
            let sources = held
                .and_then(|held| held.get(&site.key).cloned())
                .or_else(|| self.exchanges.complete_sources(site.key));
            return match sources {
                Some(sources) => FilterKeysState::Ready {
                    sources,
                    exact: true,
                },
                None => FilterKeysState::Waiting,
            };
        };
        if self.filters.is_abandoned(site.key) {
            return FilterKeysState::Unusable {
                reason: "shares_abandoned",
                detail: "a holder could not send its share".to_string(),
            };
        }
        let shares = self.exchanges.shares(site.key);
        let mut exact = true;
        for (_, names, _) in &shares {
            let Some(header) = runtime_filters::ShareHeader::parse(names) else {
                return FilterKeysState::Unusable {
                    reason: "share_unreadable",
                    detail: format!("{names:?}"),
                };
            };
            // Every share must agree with this scan on how many make the filter: a scan that ran
            // on fewer than all of them would drop rows that match.
            if header.shares != expected || shares.len() > expected {
                return FilterKeysState::Unusable {
                    reason: "share_count_mismatch",
                    detail: format!(
                        "this scan expects {expected} shares; a share counts {}, and {} arrived",
                        header.shares,
                        shares.len()
                    ),
                };
            }
            if header.key_type != site.column_type {
                return FilterKeysState::Unusable {
                    reason: "key_type_mismatch",
                    detail: format!(
                        "{} keys for a {} probe key",
                        header.key_type, site.column_type
                    ),
                };
            }
            exact &= header.exact;
        }
        if shares.len() < expected || !shares.iter().all(|(_, _, complete)| *complete) {
            return FilterKeysState::Waiting;
        }
        FilterKeysState::Ready {
            sources: shares.into_iter().map(|(source, _, _)| source).collect(),
            exact,
        }
    }

    /// Before a ready receiver consumes its inputs: sends the shares it holds and runs the
    /// deferred scans waiting on its exchanges, reading the keys from those inputs. When a build
    /// exchange is the receiver's last input to complete, the exchange hands the receiver over
    /// at once, and its keys are no longer anywhere else.
    fn use_keys_before_run(&self, ready: &ReadyFragment) {
        let Some(receiver) = Self::fragment_instance_id(&ready.params) else {
            return;
        };
        let held: HeldKeys = ready
            .inputs
            .iter()
            .map(|input| {
                let key = ExchangeKey {
                    fragment_instance_id: receiver,
                    node_id: input.node_id,
                };
                (
                    key,
                    input.sources.iter().map(SenderSource::key_source).collect(),
                )
            })
            .collect();
        for share in self.filters.take_shares_of(receiver) {
            self.send_share(share, Some(&held));
        }
        let scans = self.filters.take_ready(|site| {
            !matches!(
                self.filter_keys(site, Some(&held)),
                FilterKeysState::Waiting
            )
        });
        for scan in scans {
            self.run_deferred(scan, true, Some(&held));
        }
    }

    /// Runs, each on its own thread, the deferred scans whose filters' keys are all here, and
    /// sends the filter shares whose build exchange is complete. Called after every sender
    /// completes an exchange, before that exchange's receiver can run, so the engine copies the
    /// keys before the receiver consumes them.
    fn dispatch_filtered_scans(&self) {
        let ready = self
            .filters
            .take_ready(|site| !matches!(self.filter_keys(site, None), FilterKeysState::Waiting));
        for scan in ready {
            let service = self.clone();
            let _ = std::thread::Builder::new()
                .name("filtered-scan".to_string())
                .spawn(move || service.run_deferred(scan, true, None));
        }
        let shares = self
            .filters
            .take_ready_shares(|site| self.exchanges.complete_sources(site.key).is_some());
        for share in shares {
            let service = self.clone();
            let _ = std::thread::Builder::new()
                .name("filter-share".to_string())
                .spawn(move || service.send_share(share, None));
        }
    }

    /// Sends this CN's share of a partitioned join's filter to every scan probing it. A holder
    /// that can't tells its probers to stop waiting, and they run unfiltered.
    fn send_share(&self, share: HeldShare, held: Option<&HeldKeys>) {
        let receiver = share.site.key.fragment_instance_id;
        let skip = |reason: &str, detail: &str| {
            runtime_filters::log_skipped(
                Some(share.query),
                Some(receiver),
                share.filter_id,
                Some(share.join_node_id),
                reason,
                detail,
            );
        };
        // Without the probers there is no one to tell: they wait out their limit.
        let topology = match self.filter_topology(share.query, &share.merge_node, share.filter_id) {
            Ok(topology) => topology,
            Err(err) => {
                skip("probers_unknown", &err);
                self.log_leak_counters("filter share");
                return;
            }
        };
        match self.ship_share(&share, &topology, held) {
            Ok(ready) => {
                if let Err(err) = self.drain_ready(ready) {
                    warn!(error = %err, "a scan a runtime filter share completed failed");
                }
            }
            Err((reason, detail)) => {
                skip(reason, &detail);
                self.abandon_share(share.filter_id, &topology.probers);
            }
        }
        self.log_leak_counters("filter share");
    }

    /// Reads a share's keys where its build exchange left them, or in `held`, and ships them to
    /// every prober: the distinct keys, or only their bounds when there are more than
    /// [`runtime_filters::max_share_keys`].
    fn ship_share(
        &self,
        share: &HeldShare,
        topology: &FilterTopology,
        held: Option<&HeldKeys>,
    ) -> std::result::Result<Vec<ReadyFragment>, (&'static str, String)> {
        let sources = held
            .and_then(|held| held.get(&share.site.key).cloned())
            .or_else(|| self.exchanges.complete_sources(share.site.key))
            .ok_or(("keys_taken", String::new()))?;
        let keys = FilterKeys {
            column: share.site.column,
            sources,
        };
        let stats = self
            .executor
            .key_stats(&keys)
            .map_err(|err| ("keys_unreadable", err))?;
        let header = runtime_filters::ShareHeader {
            exact: stats.distinct <= runtime_filters::max_share_keys(),
            shares: topology.shares,
            key_type: share.site.column_type.clone(),
        };
        let node_id = FILTER_STREAM_BASE + share.filter_id;
        let stream = StreamInputSchema {
            node_id,
            stream_view: format!("sirius_stream_{node_id}"),
            columns: vec![StreamInputColumn {
                name: SHARE_KEYS.to_string(),
                ty: share.site.column_type.clone(),
            }],
        };
        let plan = self
            .translator
            .translate_filter_share(&stream, (!header.exact).then_some((stats.min, stats.max)))
            .map_err(|err| ("untranslatable", err.to_string()))?;
        let mut outputs = Vec::with_capacity(topology.probers.len());
        let (mut local, mut remote, mut streams) = (Vec::new(), Vec::new(), Vec::new());
        for (instance, address) in &topology.probers {
            let slot = SenderSlot {
                fragment_instance_id: *instance,
                node_id,
                sender_id: share.sender_id,
            };
            match self.peer_of(address).map_err(|err| ("share_failed", err))? {
                Some(peer) => {
                    streams.push(outputs.len());
                    remote.push((slot, peer));
                }
                None => local.push(slot),
            }
            outputs.push(slot);
        }
        if outputs.is_empty() {
            return Err((
                "probers_unknown",
                "the merge node lists no prober".to_string(),
            ));
        }
        info!(
            receiver = %share.site.key.fragment_instance_id,
            filter_id = share.filter_id,
            exact = header.exact,
            rows = stats.rows,
            distinct = stats.distinct,
            probers = outputs.len(),
            "sending a runtime filter share"
        );
        let run = FragmentRun {
            plan: &plan,
            inputs: Vec::new(),
            remote_inputs: Vec::new(),
            outputs,
            broadcast: topology.probers.len() > 1,
            hash_keys: Vec::new(),
            drains: None,
            filters: vec![FilterRun {
                stream_id: node_id as u64,
                keys,
                rows: stats.rows,
            }],
            fallback: None,
            // A purge of the query stops its share's run too.
            query: Some(share.site.key.fragment_instance_id),
        };
        // The column names carry the header; the probers read the keys by position.
        self.run_and_ship(
            run,
            &local,
            &remote,
            streams,
            &header.names(),
            share.sender_id,
        )
        .map_err(|failure| ("share_failed", failure.error))
    }

    /// Tells every prober of filter `filter_id` that one of its shares won't come.
    fn abandon_share(&self, filter_id: i32, probers: &[(FragmentInstanceId, TNetworkAddress)]) {
        for (instance, address) in probers {
            let abandoned = match self.peer_of(address) {
                Ok(None) => {
                    self.filters
                        .abandon(runtime_filters::filter_exchange(*instance, filter_id));
                    Ok(())
                }
                Ok(Some(peer)) => self
                    .nixl
                    .as_ref()
                    .ok_or_else(|| "this CN has no NIXL transport".to_string())
                    .and_then(|nixl| {
                        let envelope = NixlEnvelope::ShareAbandoned {
                            instance: *instance,
                            filter_id,
                        };
                        nixl.control(peer, &envelope).map(drop)
                    }),
                Err(err) => Err(err),
            };
            if let Err(err) = abandoned {
                warn!(%instance, filter_id, error = %err, "could not release a runtime filter's prober");
            }
        }
        self.dispatch_filtered_scans();
    }

    /// Runs a deferred scan with the filters whose keys are worth applying, or unfiltered.
    fn run_deferred(&self, scan: DeferredScan, filtered: bool, held: Option<&HeldKeys>) {
        let query = Self::query_id(&scan.params);
        self.contain_panic(query, || self.run_deferred_scan(scan, filtered, held));
    }

    fn run_deferred_scan(&self, scan: DeferredScan, filtered: bool, held: Option<&HeldKeys>) {
        let query = Self::query_id(&scan.params);
        let DeferredScan {
            params,
            filters,
            deferred_at,
            ready,
        } = scan;
        let ran = self
            .run_with_filters(
                &params,
                &filters,
                deferred_at,
                ready.map(|ready| ready.inputs),
                filtered,
                held,
            )
            .map_err(|failure| self.end_hops(&params, failure));
        self.reports
            .finish(&params, ran.as_ref().map(drop).map_err(String::as_str));
        // The keys were copied, or never will be: the shares go now.
        for (_, site) in &filters {
            if site.shares.is_some() {
                self.drop_shares(site.key);
            }
        }
        let result = ran.and_then(|ready| self.drain_ready(ready));
        if let Err(err) = result {
            warn!(error = %err, "a scan deferred for runtime filters failed");
            if let Some(query) = query {
                self.fail_and_purge(query, &err);
            }
        }
        self.log_leak_counters("filtered scan");
    }

    /// Runs a deferred scan's fragment, on `ready_inputs` for one that reads exchanges, with the
    /// filters whose keys are worth applying, or unfiltered.
    fn run_with_filters(
        &self,
        params: &TExecPlanFragmentParams,
        filters: &[(i32, BuildSite)],
        deferred_at: std::time::Instant,
        ready_inputs: Option<Vec<ReadyExchangeInput>>,
        filtered: bool,
        held: Option<&HeldKeys>,
    ) -> std::result::Result<Vec<ReadyFragment>, FragmentFailure> {
        let mut receiver = ready_inputs
            .map(|inputs| self.take_ready_inputs(inputs))
            .transpose()?;
        let exchanges = receiver
            .as_ref()
            .map(|receiver| receiver.exchanges.clone())
            .unwrap_or_default();
        let (inputs_in, remote_in) = receiver
            .as_mut()
            .map(|receiver| {
                (
                    std::mem::take(&mut receiver.inputs),
                    std::mem::take(&mut receiver.remote),
                )
            })
            .unwrap_or_default();
        let dump_seq = Self::dump_fragment(params);
        let (query, instance) = (Self::query_id(params), Self::fragment_instance_id(params));
        let skip = |filter_id: i32, reason: &str, detail: &str| {
            runtime_filters::log_skipped(query, instance, filter_id, None, reason, detail);
        };
        if !filtered {
            for (filter_id, _) in filters {
                skip(*filter_id, "timeout", "");
            }
        }
        let unfiltered = self
            .translate_fragment_logged(params, &exchanges, dump_seq)
            .inspect_err(|err| {
                for (filter_id, _) in filters.iter().filter(|_| filtered) {
                    skip(*filter_id, "untranslatable", err);
                }
            })?;
        let waited_ms = deferred_at.elapsed().as_millis() as u64;
        let mut inputs = Vec::new();
        let mut runs = Vec::new();
        let mut applying = Vec::new();
        for (filter_id, site) in filters.iter().filter(|_| filtered) {
            let (sources, exact) = match self.filter_keys(site, held) {
                FilterKeysState::Ready { sources, exact } => (sources, exact),
                FilterKeysState::Unusable { reason, detail } => {
                    skip(*filter_id, reason, &detail);
                    continue;
                }
                FilterKeysState::Waiting => {
                    skip(*filter_id, "keys_taken", "");
                    continue;
                }
            };
            let keys = FilterKeys {
                column: site.column,
                sources,
            };
            let stats = match self.executor.key_stats(&keys) {
                Ok(stats) => stats,
                Err(err) => {
                    skip(*filter_id, "keys_unreadable", &err);
                    continue;
                }
            };
            if !exact {
                // A share too large to send exactly sent its bounds; so do they all.
                inputs.push(FilterInput {
                    filter_id: *filter_id,
                    keys: ProbeKeys::Range {
                        min: stats.min,
                        max: stats.max,
                    },
                });
                applying.push((*filter_id, None, stats));
                continue;
            }
            let density = runtime_filters::max_density();
            if !runtime_filters::selective(&stats, density) {
                skip(
                    *filter_id,
                    "dense",
                    &format!("{stats:?} max_density={density}"),
                );
                continue;
            }
            let node_id = FILTER_STREAM_BASE + filter_id;
            inputs.push(FilterInput {
                filter_id: *filter_id,
                keys: ProbeKeys::Stream {
                    node_id,
                    stream_view: format!("sirius_stream_{node_id}"),
                    column: StreamInputColumn {
                        name: SHARE_KEYS.to_string(),
                        ty: site.column_type.clone(),
                    },
                },
            });
            runs.push(FilterRun {
                stream_id: node_id as u64,
                keys,
                rows: stats.rows,
            });
            applying.push((*filter_id, Some(node_id as u64), stats));
        }
        let filtered_plan = if inputs.is_empty() {
            None
        } else {
            match self
                .translator
                .translate_fragment_with_inputs(params, &exchanges, &inputs)
            {
                Ok(plan) => Some(plan),
                Err(err) => {
                    for (filter_id, _, _) in &applying {
                        skip(*filter_id, "untranslatable", &err.to_string());
                    }
                    None
                }
            }
        };
        let ran = match &filtered_plan {
            Some(plan) => {
                // Only the key streams the plan reads; a filter of another scan type stays
                // unbound. Bounds are a predicate on the scan the filter was deferred for.
                let bound = |stream_id: u64| {
                    plan.stream_inputs
                        .iter()
                        .any(|input| input.node_id as u64 == stream_id)
                };
                runs.retain(|run| bound(run.stream_id));
                for (filter_id, stream_id, stats) in &applying {
                    if stream_id.is_none_or(bound) {
                        runtime_filters::log_applied(
                            query,
                            instance,
                            *filter_id,
                            stream_id.is_some(),
                            stats,
                            waited_ms,
                        );
                    } else {
                        skip(*filter_id, "unbound", "no scan of the plan reads its keys");
                    }
                }
                self.execute_fragment(
                    params,
                    plan,
                    inputs_in,
                    remote_in,
                    FilterPlan {
                        filters: runs,
                        fallback: Some(&unfiltered),
                    },
                )
            }
            None => self.execute_fragment(
                params,
                &unfiltered,
                inputs_in,
                remote_in,
                FilterPlan::default(),
            ),
        };
        if ran.is_ok()
            && let Some(receiver) = receiver.as_mut()
        {
            // The engine relayed every parked input and released it.
            receiver.guard.slots.clear();
        }
        ran
    }

    /// Answers one SRNX request on a blocking thread, never waiting on the engine or the transport
    /// thread.
    fn handle_nixl_chunk(
        &self,
        request: &PTransmitChunkParams,
        attachment: &[u8],
    ) -> std::result::Result<Vec<u8>, String> {
        nixl_chunk::reject_native_chunk(request)?;
        let envelope = NixlEnvelope::decode(attachment)?;
        match envelope {
            NixlEnvelope::Topology { query, filter_id } => {
                let topology = self.filters.topology(query, filter_id);
                return Ok(nixl_chunk::encode_topology(topology.as_ref()));
            }
            NixlEnvelope::ShareAbandoned {
                instance,
                filter_id,
            } => {
                self.filters
                    .abandon(runtime_filters::filter_exchange(instance, filter_id));
                self.dispatch_filtered_scans();
                return Ok(Vec::new());
            }
            _ => {}
        }
        let nixl = self.nixl.as_ref().ok_or("this CN has no NIXL transport")?;
        match envelope {
            NixlEnvelope::Topology { .. } | NixlEnvelope::ShareAbandoned { .. } => {
                unreachable!("answered above")
            }
            NixlEnvelope::Md(_) => Ok(nixl.local_md()),
            NixlEnvelope::Packed { token, rows, names } => {
                // A refused frame is never pushed, so its buffers are freed here. A duplicate is
                // accepted as a no-op: it carries the token its first copy already delivered.
                self.deliver(request, token, rows, &names)
                    .inspect_err(|_| nixl.release(token))?;
                if request.eos == Some(true) {
                    self.dispatch_filtered_scans();
                }
                Ok(Vec::new())
            }
            NixlEnvelope::RingOpen(body) => {
                // No lane for a query that already failed here: its sender stops at once, with
                // the cause.
                if let Some(cause) = request
                    .finst_id
                    .as_ref()
                    .and_then(|id| self.exchanges.failure(FragmentInstanceId::from(id)))
                {
                    return Err(cause);
                }
                nixl.ring_open(request, &body)
            }
            NixlEnvelope::RingClose(lane) => {
                nixl.ring_close(lane)?;
                Ok(Vec::new())
            }
            NixlEnvelope::Failed { error } => {
                self.remote_sender_failed(request, &error)?;
                Ok(Vec::new())
            }
        }
    }

    /// A remote sender failed in place of its EOS: fails its receiver's query on this CN with the
    /// sender's error, which is the query's root cause. The query is refused and its result slots
    /// fail before this returns, so a receiver registering later is refused too; what it holds is
    /// freed on another thread, since dropping parked output waits on the engine thread. A frame
    /// for a query that already failed here is ignored, so a cascade of them ends.
    fn remote_sender_failed(
        &self,
        request: &PTransmitChunkParams,
        error: &str,
    ) -> std::result::Result<(), String> {
        let receiver = request
            .finst_id
            .as_ref()
            .map(FragmentInstanceId::from)
            .ok_or("Failed transmit_chunk is missing finst_id")?;
        let (sender_id, node_id) = (request.sender_id, request.node_id);
        if is_normal_end(error) {
            // The query ended normally on the sender's CN: end it here quietly too, reporting
            // CANCELLED, never an error that would fail a query that succeeded.
            if self.ends.end_normally(receiver, error) {
                info!(%receiver, ?sender_id, ?node_id, error, "a remote sender's query ended");
                let service = self.clone();
                let error = error.to_string();
                let _ = std::thread::Builder::new()
                    .name("exchange-ended".to_string())
                    .spawn(move || {
                        service.reports.cancel_query(receiver, &error);
                        service.release_query(receiver, &error);
                        service.log_leak_counters("remote sender ended");
                    });
            }
            return Ok(());
        }
        // Instance ids share their query's hi half, which is all the purge matches on.
        if !self.exchanges.mark_failed(receiver, error) {
            info!(
                %receiver,
                ?sender_id,
                ?node_id,
                error,
                "ignoring a sender failure of a query that already failed"
            );
            return Ok(());
        }
        warn!(
            %receiver,
            ?sender_id,
            ?node_id,
            error,
            "a remote sender failed; failing its query"
        );
        self.results.fail_query(receiver, error);
        let service = self.clone();
        let error = error.to_string();
        let _ = std::thread::Builder::new()
            .name("exchange-failure".to_string())
            .spawn(move || {
                service.fail_and_purge(receiver, &error);
                service.log_leak_counters("remote sender failure");
            });
        Ok(())
    }

    /// Hands a remote sender's batch (none under token 0) and eos to the exchange rendezvous,
    /// returning the receiver it completed. `None` when it isn't kept: a runtime filter share
    /// for a scan that no longer reads its filter exchange, whose batch the caller frees.
    fn push_packed(
        &self,
        request: &PTransmitChunkParams,
        token: u64,
        rows: u64,
        names: Vec<String>,
    ) -> std::result::Result<Option<Option<ReadyFragment>>, String> {
        let missing = |field: &str| format!("Packed transmit_chunk is missing {field}");
        let finst_id = request
            .finst_id
            .as_ref()
            .ok_or_else(|| missing("finst_id"))?;
        let key = ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from(finst_id),
            node_id: request.node_id.ok_or_else(|| missing("node_id"))?,
        };
        let sender_id = request.sender_id.ok_or_else(|| missing("sender_id"))?;
        let seq = request.sequence.ok_or_else(|| missing("sequence"))?;
        let eos = request.eos.ok_or_else(|| missing("eos"))?;
        let batch = (token != 0).then_some(RemoteBatch { token, rows });
        if key.node_id >= FILTER_STREAM_BASE {
            let kept = self
                .exchanges
                .push_share_frame(key, sender_id, seq, eos, names, batch)?;
            return Ok(kept.then_some(None));
        }
        let ready = self
            .exchanges
            .push_remote_frame(key, sender_id, seq, eos, names, batch)?;
        self.release_oversized_shares(key);
        Ok(Some(ready.into_ready()))
    }

    /// Releases the probers of the shares held at build exchange `key` once it has received more
    /// rows than a share sends as exact keys ([`runtime_filters::max_share_keys`]). Such a share
    /// could only go out as bounds, which rarely drop anything, and its probers would wait for the
    /// whole build first: q21's l2 and l3 scans waited about 1.2 s for bounds over every order
    /// key. Rows aren't distinct keys, so a build with many duplicate keys loses a filter it
    /// could have sent exactly.
    fn release_oversized_shares(&self, key: ExchangeKey) {
        if !runtime_filters::enabled() || !self.filters.holds_share_at(key) {
            return;
        }
        let (rows, limit) = (
            self.exchanges.received_rows(key),
            runtime_filters::max_share_keys(),
        );
        if rows <= limit {
            return;
        }
        for share in self.filters.take_ready_shares(|site| site.key == key) {
            runtime_filters::log_skipped(
                Some(share.query),
                Some(key.fragment_instance_id),
                share.filter_id,
                Some(share.join_node_id),
                "too_many_keys",
                &format!("{rows} build rows, more than {limit}"),
            );
            self.release_probers(share.query, share.filter_id, share.merge_node);
        }
    }

    /// Runs sender fragment `params` on its own thread, so its `exec_plan_fragment` call answers
    /// once the fragment is accepted: a sender can run longer than the FE waits for that answer
    /// (`query_delivery_timeout`). How the run ended reaches the FE as the instance's report.
    fn run_sender_async(&self, params: TExecPlanFragmentParams, translated: TranslatedPlan) {
        // Shared with the thread, so a failed spawn leaves the sender here to run inline.
        let pending = Arc::new(Mutex::new(Some((params, translated))));
        let (service, thread_pending) = (self.clone(), Arc::clone(&pending));
        self.senders_running.fetch_add(1, Ordering::SeqCst);
        let spawned = std::thread::Builder::new()
            .name("exchange-sender".to_string())
            .spawn(move || {
                if let Some((params, translated)) = take_pending(&thread_pending) {
                    service.run_sender(&params, &translated);
                }
            });
        if let Err(err) = spawned {
            warn!(error = %err, "cannot spawn a sender thread; running the sender inline");
            if let Some((params, translated)) = take_pending(&pending) {
                self.run_sender(&params, &translated);
            }
        }
    }

    /// Runs sender fragment `params` off the dispatch path, then the receivers its output
    /// completes. A failure is reported once and fails the query here, as a receiver's does.
    fn run_sender(&self, params: &TExecPlanFragmentParams, translated: &TranslatedPlan) {
        let query = Self::query_id(params);
        self.contain_panic(query, || {
            let ran = self
                .execute_fragment(
                    params,
                    translated,
                    Vec::new(),
                    Vec::new(),
                    FilterPlan::default(),
                )
                .map_err(|failure| self.end_hops(params, failure));
            self.reports
                .finish(params, ran.as_ref().map(drop).map_err(String::as_str));
            if let Err(err) = ran.and_then(|ready| self.drain_ready(ready)) {
                warn!(error = %err, "a sender run off the dispatch path failed");
                if let Some(query) = query {
                    self.fail_and_purge(query, &err);
                }
            }
        });
        self.senders_running.fetch_sub(1, Ordering::SeqCst);
        self.log_leak_counters("sender");
    }

    /// Runs a receiver a remote frame completed on its own thread, so `transmit_chunk` answers
    /// the sender without waiting on GPU work.
    fn drain_ready_async(&self, ready: ReadyFragment) {
        // Shared with the thread, so a failed spawn leaves the receiver here to run inline:
        // dropping it would leave its query waiting and its received batches held.
        let pending = Arc::new(Mutex::new(Some(ready)));
        let (service, thread_pending) = (self.clone(), Arc::clone(&pending));
        let spawned = std::thread::Builder::new()
            .name("exchange-receiver".to_string())
            .spawn(move || {
                if let Some(ready) = take_pending(&thread_pending) {
                    service.drain_ready_logged(ready);
                }
            });
        if let Err(err) = spawned {
            tracing::warn!(error = %err, "cannot spawn a receiver thread; running the receiver inline");
            if let Some(ready) = take_pending(&pending) {
                self.drain_ready_logged(ready);
            }
        }
    }

    /// Runs `ready` and every receiver it completes, logging a failure: the query is failed and
    /// purged on this CN either way. A panic is contained as [`contain_panic`] does.
    ///
    /// [`contain_panic`]: Self::contain_panic
    fn drain_ready_logged(&self, ready: ReadyFragment) {
        let query = Self::query_id(&ready.params);
        self.contain_panic(query, || {
            if let Err(err) = self.drain_ready(vec![ready]) {
                tracing::warn!(error = %err, "a receiver fed by a remote sender failed");
            }
        });
    }

    /// Runs `work` off the dispatch path for `query`. A panic in it fails the query on this CN as
    /// an error would: every instance of it still owed reports the panic, and what it holds is
    /// freed. Otherwise the instances it unwound past would never send their final report.
    fn contain_panic(&self, query: Option<FragmentInstanceId>, work: impl FnOnce()) {
        let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) else {
            return;
        };
        let error = format!("a fragment panicked: {}", panic_message(panic.as_ref()));
        warn!(error, "failing the panicked fragment's query");
        if let Some(query) = query {
            self.reports.fail_query(query, &error);
            self.fail_and_purge(query, &error);
        }
        self.log_leak_counters("panic");
    }

    /// Restores descriptor tables omitted by StarRocks's per-query cache protocol.
    fn resolve_descriptor_table(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<TExecPlanFragmentParams, String> {
        let mut resolved = params.clone();
        let Some(query_id) = params
            .params
            .as_ref()
            .map(|exec| FragmentInstanceId::from(&exec.query_id))
        else {
            return Ok(resolved);
        };
        let Some(desc) = params.desc_tbl.as_ref() else {
            return Ok(resolved);
        };
        let mut cache = self
            .descriptor_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if Self::is_cached_reference(desc) {
            resolved.desc_tbl = Some(
                cache
                    .get(query_id.query_hi())
                    .cloned()
                    .ok_or_else(|| format!("descriptor table cache miss for query {query_id}"))?,
            );
        } else {
            *cache.get_or_insert_with(query_id.query_hi(), || desc.clone()) = desc.clone();
        }
        Ok(resolved)
    }

    /// Drops `query`'s (any instance id of it) cached descriptor table: nothing of it runs here
    /// any more.
    fn forget_descriptor_table(&self, query: FragmentInstanceId) {
        self.descriptor_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(query.query_hi());
    }

    /// True for the empty table, marked `is_cached`, that StarRocks sends in place of a
    /// descriptor table it already sent for the query.
    fn is_cached_reference(desc: &TDescriptorTable) -> bool {
        desc.is_cached == Some(true)
            && desc.slot_descriptors.as_ref().is_none_or(Vec::is_empty)
            && desc.tuple_descriptors.is_empty()
            && desc.table_descriptors.as_ref().is_none_or(Vec::is_empty)
    }

    /// Writes the received fragment params to `$SIRIUS_CN_DUMP_FRAGMENTS/fragment-<pid>-<seq>.txt`
    /// (debug format) for offline plan analysis. No-op when the variable is unset.
    fn dump_fragment(params: &TExecPlanFragmentParams) -> Option<u64> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let Ok(dir) = std::env::var("SIRIUS_CN_DUMP_FRAGMENTS") else {
            return None;
        };
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = Self::dump_path(&dir, "fragment", seq, "txt");
        if let Err(err) = std::fs::write(&path, format!("{params:#?}")) {
            tracing::warn!(error = %err, path = %path.display(), "failed to dump fragment params");
        }
        Some(seq)
    }

    /// Runs one fragment on `inputs`, its senders' parked output keyed by exchange node id. A
    /// RESULT_SINK buffers its rows for `fetch_data`; a DATA_STREAM_SINK parks its output for its
    /// receivers and returns the receivers its output completes. An unsupported sink or a missing
    /// fragment instance id fails loudly so integration gaps surface as an error rather than as a
    /// silent empty result at `fetch_data`.
    fn execute_fragment(
        &self,
        params: &TExecPlanFragmentParams,
        translated: &TranslatedPlan,
        inputs: Vec<(i32, Vec<SenderSlot>)>,
        remote_inputs: Vec<(i32, i32, Vec<RemoteBatch>)>,
        filters: FilterPlan<'_>,
    ) -> std::result::Result<Vec<ReadyFragment>, FragmentFailure> {
        if Self::claim_injection("") {
            return Err("injected fragment failure (SIRIUS_CN_FAIL_ONCE_FILE)"
                .to_string()
                .into());
        }
        if Self::is_mysql_result_sink(params)? {
            let id = Self::fragment_instance_id(params).ok_or_else(|| {
                "RESULT_SINK fragment is missing a fragment_instance_id".to_string()
            })?;
            let result = self
                .executor
                .run_fragment(FragmentRun {
                    plan: translated,
                    inputs,
                    remote_inputs,
                    outputs: Vec::new(),
                    broadcast: false,
                    hash_keys: Vec::new(),
                    drains: None,
                    filters: filters.filters,
                    fallback: filters.fallback,
                    query: Self::query_id(params),
                })?
                .ok_or_else(|| "result fragment returned no rows".to_string())?;
            let batch = result_encoder::MysqlResultEncoder::encode(&result.batches, 0)?;
            self.results.insert(id, batch);
            return Ok(Vec::new());
        }

        let SenderPlan {
            broadcast,
            hash_keys,
            outputs,
            local,
            remote,
            streams,
            sender_id,
        } = self.sender_plan(params, translated)?;
        let on_exchange_input = !inputs.is_empty() || !remote_inputs.is_empty();
        if on_exchange_input && !remote.is_empty() && Self::claim_injection("remote-sender") {
            return Err(
                "injected fragment failure (SIRIUS_CN_FAIL_ONCE_FILE, remote-sender)"
                    .to_string()
                    .into(),
            );
        }
        if !remote.is_empty() && Self::claim_injection("hang-sender") {
            let instance = Self::fragment_instance_id(params).map(|id| id.to_string());
            warn!(
                instance = instance.unwrap_or_default(),
                "injected hang (SIRIUS_CN_FAIL_ONCE_FILE, hang-sender): sending nothing"
            );
            return Ok(Vec::new());
        }
        let run = FragmentRun {
            plan: translated,
            inputs,
            remote_inputs,
            outputs,
            broadcast,
            hash_keys,
            drains: None,
            filters: filters.filters,
            fallback: filters.fallback,
            query: Self::query_id(params),
        };
        self.run_and_ship(
            run,
            &local,
            &remote,
            streams,
            &translated.output_names,
            sender_id,
        )
    }

    /// Where sender fragment `params` sends its output, checked before it runs: a sink this CN
    /// cannot serve, or a destination it cannot resolve, fails the fragment with nothing parked.
    fn sender_plan(
        &self,
        params: &TExecPlanFragmentParams,
        translated: &TranslatedPlan,
    ) -> std::result::Result<SenderPlan, FragmentFailure> {
        // Fail closed: without a sink nothing parks for its receiver, which would wait forever.
        let sink = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
            .ok_or_else(|| "fragment carries no output sink".to_string())?;
        if sink.type_ != TDataSinkType::DATA_STREAM_SINK {
            return Err(format!("output sink {:?} is not supported", sink.type_).into());
        }
        let stream_sink = sink.stream_sink.as_ref().ok_or_else(|| {
            "DATA_STREAM_SINK fragment carries no stream_sink payload".to_string()
        })?;
        if stream_sink.limit.is_some_and(|limit| limit >= 0) {
            return Err("data stream sink limits are not supported"
                .to_string()
                .into());
        }
        let exec = params
            .params
            .as_ref()
            .ok_or_else(|| "DATA_STREAM_SINK fragment is missing execution params".to_string())?;
        let destinations = exec
            .destinations
            .as_deref()
            .filter(|destinations| !destinations.is_empty())
            .ok_or_else(|| "DATA_STREAM_SINK fragment has no destinations".to_string())?;
        let (broadcast, hash_keys) = Self::sink_mode(
            stream_sink.output_partition.type_,
            destinations.len(),
            translated.output_partition_columns.as_ref(),
        )?;

        // Every sender of an exchange must be told apart; defaulting would collide at push time,
        // after the fragment already ran and parked.
        let sender_id = exec
            .sender_id
            .ok_or_else(|| "DATA_STREAM_SINK fragment is missing sender_id".to_string())?;
        let mut outputs = Vec::with_capacity(destinations.len());
        let (mut local, mut remote, mut streams) = (Vec::new(), Vec::new(), Vec::new());
        for destination in destinations {
            let slot = SenderSlot {
                fragment_instance_id: FragmentInstanceId::from(&destination.fragment_instance_id),
                node_id: stream_sink.dest_node_id,
                sender_id,
            };
            // Resolved before running so an unreachable destination leaves no output parked.
            match self.remote_peer(destination)? {
                Some(peer) => {
                    streams.push(outputs.len());
                    remote.push((slot, peer));
                }
                None => local.push(slot),
            }
            outputs.push(slot);
        }
        Ok(SenderPlan {
            broadcast,
            hash_keys,
            outputs,
            local,
            remote,
            streams,
            sender_id,
        })
    }

    /// Runs a sender fragment and ships its `remote` outputs, then hands its local outputs to
    /// their receivers and returns the receivers they complete.
    fn run_and_ship(
        &self,
        run: FragmentRun<'_>,
        local: &[SenderSlot],
        remote: &[(SenderSlot, SocketAddr)],
        streams: Vec<usize>,
        names: &[String],
        sender_id: i32,
    ) -> std::result::Result<Vec<ReadyFragment>, FragmentFailure> {
        let shipped = if remote.is_empty() || !stream_output_enabled() {
            self.executor.run_fragment(run)?;
            self.ship_parked(remote, names)
        } else {
            self.run_streaming(run, streams, remote, names)?
        };

        // If a hop failed, the local claims are released too: their receivers can no longer
        // complete, and nothing may stay parked.
        if let Err(err) = shipped {
            for &slot in local {
                let _ = self.executor.drop_parked(slot);
            }
            return Err(FragmentFailure::after_hops(err));
        }
        let mut ready = Vec::new();
        for (index, &slot) in local.iter().enumerate() {
            let key = ExchangeKey {
                fragment_instance_id: slot.fragment_instance_id,
                node_id: slot.node_id,
            };
            let source = SenderSource::LocalParked {
                names: names.to_vec(),
                slot,
            };
            let pushed = if key.node_id >= FILTER_STREAM_BASE {
                self.exchanges
                    .push_share_sender(key, sender_id, source)
                    .map(|dropped| {
                        if dropped.is_some() {
                            // A runtime filter share for a scan that no longer waits for it.
                            let _ = self.executor.drop_parked(slot);
                        }
                        None
                    })
            } else {
                self.exchanges.push_sender(key, sender_id, source)
            };
            // A refused source is not kept (its query was purged, say), so this slot and the
            // ones not yet handed over are dropped here or they stay parked.
            match pushed {
                Ok(completed) => ready.extend(completed),
                Err(err) => {
                    for &unpushed in &local[index..] {
                        let _ = self.executor.drop_parked(unpushed);
                    }
                    // Receivers this sender already completed were handed over, so a purge no
                    // longer reaches them: free their inputs here, since they will not run.
                    for completed in &ready {
                        drop(self.ready_inputs_guard(&completed.inputs));
                    }
                    return Err(FragmentFailure::after_hops(err));
                }
            }
        }
        if !local.is_empty() {
            self.dispatch_filtered_scans();
        }
        Ok(ready)
    }

    /// Runs a sender fragment while its remote outputs ship batch by batch, so its output never
    /// has to be fully resident before the first byte leaves. Local outputs stay parked. An
    /// executor that hands out no drains is shipped from parked output after its run instead.
    ///
    /// The outer error is the run's, and says whether the transport took the hops; the inner one
    /// is the hops'. Either way every remote claim is released.
    fn run_streaming(
        &self,
        mut run: FragmentRun<'_>,
        streams: Vec<usize>,
        remote: &[(SenderSlot, SocketAddr)],
        names: &[String],
    ) -> std::result::Result<std::result::Result<(), String>, FragmentFailure> {
        let nixl = self
            .nixl
            .as_ref()
            .ok_or("this CN has no NIXL transport".to_string())?;
        let deadline = run
            .query
            .and_then(|query| self.deadlines.remaining(query))
            .map(|left| std::time::Instant::now() + left);
        let (respond, handed) = channel();
        run.drains = Some(DrainHandoff { streams, respond });
        let (ran, streamed) = std::thread::scope(|scope| {
            let shipper = scope.spawn(move || {
                // No drains: the executor does not stream, or the fragment failed before it ran.
                let drains = handed.recv().ok()?;
                let hops = remote
                    .iter()
                    .zip(drains)
                    .map(|(&(slot, peer), drain)| StreamHop {
                        peer,
                        slot,
                        names: names.to_vec(),
                        drain,
                        deadline,
                    })
                    .collect();
                Some(nixl.stream(hops))
            });
            let ran = self.executor.run_fragment(run);
            let streamed = shipper
                .join()
                .unwrap_or_else(|_| Some(Err("the output shipper panicked".to_string())));
            (ran, streamed)
        });
        match (ran, streamed) {
            (ran, Some(streamed)) => {
                for &(slot, _) in remote {
                    let _ = self.executor.drop_parked(slot);
                }
                ran.map(|_| streamed).map_err(FragmentFailure::after_hops)
            }
            (Ok(_), None) => Ok(self.ship_parked(remote, names)),
            // No hop was handed to the transport.
            (Err(err), None) => Err(err.into()),
        }
    }

    /// Ships every remote output from its parked batches, one destination at a time. Hops run
    /// until one fails, and every hop's claim is released either way; the first error is returned.
    fn ship_parked(
        &self,
        remote: &[(SenderSlot, SocketAddr)],
        names: &[String],
    ) -> std::result::Result<(), String> {
        let mut shipped: std::result::Result<(), String> = Ok(());
        for &(slot, peer) in remote {
            if shipped.is_ok() {
                shipped = self.ship_remote(slot, peer, names);
            } else {
                // The query is failing: shipping the rest would only write into peers' pools. The
                // receiver of a hop that never ran still gets a failure frame in place of its EOS.
                if let (Some(nixl), Err(err)) = (&self.nixl, &shipped) {
                    nixl.fail(peer, slot, err);
                }
                let _ = self.executor.drop_parked(slot);
            }
        }
        shipped
    }

    /// How a sink to `destinations` fans out: whether it broadcasts, and its hash key columns
    /// (empty unless hash-partitioned). One destination is a gather whatever the partition type.
    /// The translator has already refused hash keys that are not bare slot refs.
    fn sink_mode(
        partition: TPartitionType,
        destinations: usize,
        partition_columns: Option<&Vec<usize>>,
    ) -> std::result::Result<(bool, Vec<usize>), String> {
        let hash_keys = match partition {
            _ if destinations == 1 => Vec::new(),
            TPartitionType::UNPARTITIONED => Vec::new(),
            TPartitionType::HASH_PARTITIONED => partition_columns.cloned().ok_or_else(|| {
                "a hash-partitioned data stream sink translated without partition key columns"
                    .to_string()
            })?,
            other => {
                return Err(format!(
                    "a data stream sink with {destinations} destinations carries partition type \
                     {other:?}, which this CN does not support"
                ));
            }
        };
        Ok((destinations > 1 && hash_keys.is_empty(), hash_keys))
    }

    /// `destination`'s brpc address, or `None` when it is this CN. A remote destination needs
    /// this CN's NIXL transport.
    fn remote_peer(
        &self,
        destination: &TPlanFragmentDestination,
    ) -> std::result::Result<Option<SocketAddr>, String> {
        let id = FragmentInstanceId::from(&destination.fragment_instance_id);
        let address = destination.brpc_server.as_ref().ok_or_else(|| {
            format!(
                "DATA_STREAM_SINK destination for fragment instance {id} has no brpc_server address"
            )
        })?;
        if *address != self.brpc_address && self.nixl.is_none() {
            return Err(format!(
                "DATA_STREAM_SINK destination {}:{} for fragment instance {id} is remote, and \
                 this CN has no NIXL transport",
                address.hostname, address.port
            ));
        }
        self.peer_of(address)
    }

    /// The socket address of the CN advertising brpc `address`, or `None` when it is this CN.
    fn peer_of(
        &self,
        address: &TNetworkAddress,
    ) -> std::result::Result<Option<SocketAddr>, String> {
        if *address == self.brpc_address {
            return Ok(None);
        }
        let host = address.hostname.as_str();
        let port = u16::try_from(address.port)
            .map_err(|_| format!("destination brpc port {} is not a TCP port", address.port))?;
        (host, port)
            .to_socket_addrs()
            .map_err(|err| format!("failed to resolve exchange peer {host}:{port}: {err}"))?
            .next()
            .map(Some)
            .ok_or_else(|| format!("exchange peer {host}:{port} resolved to no address"))
    }

    fn ship_remote(
        &self,
        slot: SenderSlot,
        peer: SocketAddr,
        names: &[String],
    ) -> std::result::Result<(), String> {
        let nixl = self.nixl.as_ref().ok_or("this CN has no NIXL transport")?;
        let sent = nixl.send(peer, slot, names.to_vec(), Arc::clone(&self.executor));
        sent.and(self.executor.drop_parked(slot))
    }

    /// Runs receivers whose sender sets completed, then the receivers their outputs complete. A
    /// failure also fails the query's waiting result slot, so `fetch_data` reports it at once.
    /// One failed receiver does not stop the rest, whose parked inputs would otherwise never be
    /// released; the first error is returned.
    fn drain_ready(&self, mut queue: Vec<ReadyFragment>) -> std::result::Result<(), String> {
        let mut first_error = None;
        while let Some(ready) = queue.pop() {
            let ready = if runtime_filters::enabled() {
                self.use_keys_before_run(&ready);
                match self.park_for_filters(ready) {
                    Some(ready) => ready,
                    None => continue,
                }
            } else {
                ready
            };
            let ReadyFragment { params, inputs } = ready;
            match self.execute_ready_fragment(&params, inputs) {
                Ok(next) => {
                    self.reports.finish(&params, Ok(()));
                    queue.extend(next);
                }
                Err(failure) => {
                    let err = self.end_hops(&params, failure);
                    self.reports.finish(&params, Err(&err));
                    if let Some(query) = Self::query_id(&params) {
                        self.fail_and_purge(query, &err);
                    }
                    first_error.get_or_insert(err);
                }
            }
        }
        self.log_leak_counters("receivers");
        first_error.map_or(Ok(()), Err)
    }

    /// Fails `query` on this CN: its waiting result slots report `error`, and everything the
    /// exchange still holds for it is freed. Without this a failed query's receivers wait forever
    /// on senders that never finish, pinning their parked output and received buffers in GPU
    /// memory for the life of the process. Idempotent; a late frame of the query is refused.
    ///
    /// Receivers and deferred scans dropped here never run, so their remote receivers get a
    /// failure frame in place of the EOS they would have sent.
    fn fail_and_purge(&self, query: FragmentInstanceId, error: &str) {
        self.purge(query, error, true);
    }

    /// Frees what `query` still holds after the FE ended it normally: what is left (a LIMIT that
    /// stopped the query early, say) is dropped quietly, without failure frames or a failure log.
    /// The query is then refused like a failed one, with `reason`.
    fn release_query(&self, query: FragmentInstanceId, reason: &str) {
        self.purge(query, reason, false);
    }

    fn purge(&self, query: FragmentInstanceId, error: &str, failed: bool) {
        let started = std::time::Instant::now();
        self.deadlines.end(query);
        self.results.fail_query(query, error);
        self.forget_descriptor_table(query);
        // Fragments waiting for runtime filters never run: their inputs are freed and their hops
        // end like a waiting receiver's.
        let mut deferred = Vec::new();
        for scan in self.filters.purge_query(query) {
            let instance = Self::fragment_instance_id(&scan.params);
            for (filter_id, _) in &scan.filters {
                runtime_filters::log_skipped(Some(query), instance, *filter_id, None, "purged", "");
            }
            self.release_inputs(scan.ready);
            deferred.push(scan.params);
        }
        let purged = self.exchanges.purge_query(query, error);
        // A failed query's run in progress holds GPU memory and works for nothing: stop it, now
        // that its cause is recorded, so the interrupted fragment reports that cause first. A
        // finished query has nothing left running.
        if failed {
            self.executor.interrupt(query);
        }
        for params in deferred.iter().chain(&purged.receivers) {
            if failed {
                self.fail_remote_hops(params, error);
            }
            self.reports.finish(params, Err(error));
        }
        if let Some(nixl) = &self.nixl {
            for &token in &purged.tokens {
                nixl.release(token);
            }
        }
        // Not waited for: a drop runs on the engine thread, which may be busy with another
        // fragment. Until it ran, a receive allocation that finds the pool full waits for it.
        for &slot in &purged.slots {
            self.executor.drop_parked_later(slot);
        }
        self.watch_frees(query, started);
        let (remote_batches, dropped_parked) = (purged.tokens.len(), purged.slots.len());
        let (dropped_receivers, deferred_scans) = (purged.receivers.len(), deferred.len());
        if failed {
            info!(
                %query,
                remote_batches,
                dropped_parked,
                dropped_receivers,
                deferred_scans,
                error,
                "purged a failed query's exchange state"
            );
        } else {
            info!(
                %query,
                remote_batches,
                dropped_parked,
                dropped_receivers,
                deferred_scans,
                reason = error,
                "released a finished query's exchange state"
            );
        }
    }

    /// Starts the deadline of `params`' query from its first fragment here, when the FE sent a
    /// query timeout, and widens how long ended queries are remembered to match it.
    fn start_deadline(&self, params: &TExecPlanFragmentParams) {
        let (Some(query), Some(timeout)) = (
            Self::query_id(params),
            deadlines::query_timeout(
                params
                    .query_options
                    .as_ref()
                    .and_then(|options| options.query_timeout),
            ),
        ) else {
            return;
        };
        if let Some(window) = self.ends.observe_timeout(timeout) {
            self.results.set_window(window);
            self.descriptor_tables
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .set_window(window);
        }
        if self.deadlines.start(query, timeout) {
            let service = self.clone();
            let spawned = std::thread::Builder::new()
                .name("query-deadlines".to_string())
                .spawn(move || service.watch_deadlines());
            if let Err(err) = spawned {
                warn!(error = %err, "cannot watch query deadlines");
                self.deadlines.watcher_stopped();
            }
        }
    }

    /// Fails each query whose deadline passed, until no deadline is left to watch. A panic
    /// failing one query is contained, so the others' deadlines still fire; should the watcher
    /// die anyway, the next query's first fragment starts another.
    fn watch_deadlines(&self) {
        struct Stopped<'a>(&'a Deadlines);
        impl Drop for Stopped<'_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.watcher_stopped();
                }
            }
        }
        let _stopped = Stopped(&self.deadlines);
        loop {
            std::thread::sleep(DEADLINE_TICK);
            let (passed, left) = self.deadlines.take_passed(std::time::Instant::now());
            for deadline in passed {
                let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.deadline_passed(deadline)
                }));
                if let Err(panic) = failed {
                    warn!(
                        query = %deadline.query,
                        panic = panic_message(panic.as_ref()),
                        "failing a query at its deadline panicked"
                    );
                }
            }
            if !left {
                return;
            }
        }
    }

    /// `deadline`'s query ran out of time here. One that already ended (the FE's own timeout
    /// came first, say) or holds nothing here any more is left alone; any other is failed and
    /// purged with the timeout as its cause, so its receivers stop waiting, its result slots and
    /// its runs fail, and its reports and failure frames go out.
    fn deadline_passed(&self, deadline: Deadline) {
        let query = deadline.query;
        if self.ends.cause(query).is_some() || self.ends.ended_by_fe(query) {
            return;
        }
        let holds = self.exchanges.holds(query)
            || self.filters.holds(query)
            || self.reports.owes(query)
            || self.results.waits_for(query);
        if !holds {
            return;
        }
        let cause = format!(
            "query timed out after {} s on this CN",
            deadline.timeout.as_secs()
        );
        // Claimed atomically: an FE cancel or a normal end since the checks above wins, and the
        // query is then not failed after the fact.
        if !self.exchanges.mark_failed_unless_ended(query, &cause) {
            return;
        }
        warn!(%query, cause, "a query's deadline passed; failing it");
        self.fail_and_purge(query, &cause);
        self.log_leak_counters("deadline");
    }

    /// Logs, on another thread, when what a purge of `query` asked the engine to free is free, and
    /// what the CN holds then: the drops and the interrupted run finish after the purge returns.
    fn watch_frees(&self, query: FragmentInstanceId, started: std::time::Instant) {
        let Some(frees) = self
            .executor
            .pending_frees()
            .filter(|frees| frees.pending() > 0)
        else {
            return;
        };
        let service = self.clone();
        let _ = std::thread::Builder::new()
            .name("purge-watch".to_string())
            .spawn(move || {
                let freed = frees.wait_for_none(PURGE_WATCH);
                info!(
                    %query,
                    freed,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "a purge's frees finished"
                );
                service.log_leak_counters("purge drained");
            });
    }

    /// Ends `params`' remote hops with a failure frame unless the transport already ended them,
    /// and returns the error. A fragment of a query that failed here while it ran (interrupted,
    /// say, or refused its turn on the engine) fails because of that: its error leads with the
    /// query's cause, so whichever reply or report reaches the FE first names the cause.
    fn end_hops(&self, params: &TExecPlanFragmentParams, failure: FragmentFailure) -> String {
        let error = match Self::query_id(params).and_then(|query| self.exchanges.failure(query)) {
            Some(cause) if !failure.error.starts_with(cause.as_str()) => {
                format!("{cause} ({})", failure.error)
            }
            _ => failure.error,
        };
        if !failure.hops_ended {
            self.fail_remote_hops(params, &error);
        }
        error
    }

    /// Sends a failure frame to every remote destination of `params`' DATA_STREAM_SINK, in place
    /// of the EOS this fragment will never send. Without it a receiver on another CN waits until
    /// the FE times the query out. Best effort, and without waiting for the peers: a destination
    /// that cannot be resolved is skipped, and one that cannot be reached is logged.
    fn fail_remote_hops(&self, params: &TExecPlanFragmentParams, error: &str) {
        let Some(nixl) = &self.nixl else {
            return;
        };
        // Nothing failed: the receivers end with their own cancel from the FE. A failure frame
        // would make a CN that has not had it yet report a query that succeeded as failed.
        if is_normal_end(error) {
            return;
        }
        let (Some(sink), Some(exec)) = (
            params
                .fragment
                .as_ref()
                .and_then(|fragment| fragment.output_sink.as_ref())
                .and_then(|sink| sink.stream_sink.as_ref()),
            params.params.as_ref(),
        ) else {
            return;
        };
        for destination in exec.destinations.iter().flatten() {
            let Ok(Some(peer)) = self.remote_peer(destination) else {
                continue;
            };
            let slot = SenderSlot {
                fragment_instance_id: FragmentInstanceId::from(&destination.fragment_instance_id),
                node_id: sink.dest_node_id,
                sender_id: exec.sender_id.unwrap_or(0),
            };
            info!(
                instance = %FragmentInstanceId::from(&exec.fragment_instance_id),
                peer = %peer,
                ?slot,
                "ending an exchange hop with a failure frame"
            );
            nixl.fail(peer, slot, error);
        }
    }

    /// Logs what this CN still holds across queries. Every count is zero on an idle CN; one that
    /// stays non-zero between queries is leaked GPU memory.
    fn log_leak_counters(&self, event: &str) {
        let exchange = self.exchanges.counts();
        info!(
            event,
            receivers = exchange.receivers,
            parked_senders = exchange.parked_senders,
            remote_batches = exchange.remote_batches,
            parked_fragments = self.executor.parked_fragments(),
            direct_buffers = self.nixl.as_ref().map_or(0, |nixl| nixl.outstanding()),
            deferred_scans = self.filters.deferred(),
            owed_reports = self.reports.owed(),
            held_shares = self.filters.held_shares(),
            senders_running = self.senders_running.load(Ordering::SeqCst),
            "leak counters"
        );
    }

    /// Translates a ready receiver against its senders' output names and runs it on their
    /// parked output.
    fn execute_ready_fragment(
        &self,
        params: &TExecPlanFragmentParams,
        ready_inputs: Vec<ReadyExchangeInput>,
    ) -> std::result::Result<Vec<ReadyFragment>, FragmentFailure> {
        let ReceiverInputs {
            mut guard,
            exchanges,
            inputs,
            remote,
        } = self.take_ready_inputs(ready_inputs)?;
        let dump_seq = Self::dump_fragment(params);
        let translated = self.translate_fragment_logged(params, &exchanges, dump_seq)?;
        let next =
            self.execute_fragment(params, &translated, inputs, remote, FilterPlan::default())?;
        // The engine relayed every parked input and released it.
        guard.slots.clear();
        Ok(next)
    }

    /// A ready receiver's inputs as the engine takes them, bound to their exchanges, under a
    /// guard that frees them unless the run relays them.
    fn take_ready_inputs(
        &self,
        ready: Vec<ReadyExchangeInput>,
    ) -> std::result::Result<ReceiverInputs, String> {
        let guard = self.ready_inputs_guard(&ready);
        let exchanges = Self::exchange_inputs(&ready)?;
        let mut inputs = Vec::with_capacity(ready.len());
        let mut remote = Vec::new();
        for input in ready {
            let mut slots = Vec::new();
            for source in input.sources {
                match source {
                    SenderSource::LocalParked { slot, .. } => slots.push(slot),
                    SenderSource::Remote {
                        sender_id, batches, ..
                    } => remote.push((input.node_id, sender_id, batches)),
                }
            }
            inputs.push((input.node_id, slots));
        }
        Ok(ReceiverInputs {
            guard,
            exchanges,
            inputs,
            remote,
        })
    }

    /// Binds each exchange to its senders' output names, which must agree across senders.
    fn exchange_inputs(
        inputs: &[ReadyExchangeInput],
    ) -> std::result::Result<Vec<ExchangeInput>, String> {
        inputs
            .iter()
            .map(|input| {
                let names = input
                    .sources
                    .first()
                    .map(|source| source.names().to_vec())
                    .ok_or_else(|| {
                        format!("exchange node {} has no sender source", input.node_id)
                    })?;
                if input
                    .sources
                    .iter()
                    .any(|source| source.names() != names.as_slice())
                {
                    return Err(format!(
                        "exchange node {} senders produced different output names",
                        input.node_id
                    ));
                }
                Ok(ExchangeInput {
                    node_id: input.node_id,
                    stream_view: format!("sirius_stream_{}", input.node_id),
                    names,
                })
            })
            .collect()
    }

    /// The sender count of every `EXCHANGE_NODE` in the fragment; empty for a leaf fragment.
    fn receiver_exchanges(
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<Vec<(i32, usize)>, String> {
        let exchange_nodes = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.plan.as_ref())
            .map(|plan| {
                plan.nodes
                    .iter()
                    .filter(|node| node.node_type == TPlanNodeType::EXCHANGE_NODE)
                    .map(|node| node.node_id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        exchange_nodes
            .into_iter()
            .map(|node_id| {
                let expected = params
                    .params
                    .as_ref()
                    .and_then(|exec| exec.per_exch_num_senders.get(&node_id))
                    .copied()
                    .ok_or_else(|| {
                        format!("EXCHANGE_NODE {node_id} is missing per_exch_num_senders")
                    })?;
                let expected = usize::try_from(expected).map_err(|_| {
                    format!("EXCHANGE_NODE {node_id} has negative sender count {expected}")
                })?;
                Ok((node_id, expected))
            })
            .collect()
    }

    /// Deserializes a FE batch attachment and merges common params into each instance.
    fn translate_batch_attachment(
        &self,
        protocol: Option<&str>,
        attachment: &[u8],
    ) -> std::result::Result<(), StatusPb> {
        Self::ensure_binary_protocol(protocol).map_err(Self::internal_error)?;
        let batch = Self::deserialize_binary::<TExecBatchPlanFragmentsParams>(attachment).map_err(
            |err| {
                Self::internal_error(format!(
                    "failed to deserialize TExecBatchPlanFragmentsParams: {err}"
                ))
            },
        )?;
        let common = batch.common_param.as_ref().ok_or_else(|| {
            Self::internal_error("TExecBatchPlanFragmentsParams.common_param is missing")
        })?;
        let instances = batch.unique_param_per_instance.as_ref().ok_or_else(|| {
            Self::internal_error(
                "TExecBatchPlanFragmentsParams.unique_param_per_instance is missing",
            )
        })?;

        if instances.is_empty() {
            return Err(Self::internal_error(
                "TExecBatchPlanFragmentsParams.unique_param_per_instance is empty",
            ));
        }

        // The FE puts the query's descriptor table in common_param and gives each instance an
        // empty one marked cached. Like the BE, which prepares common_param before any instance,
        // cache the shared table for the query first.
        self.resolve_descriptor_table(common)
            .map_err(Self::internal_error)?;

        for (idx, instance) in instances.iter().enumerate() {
            let mut params = instance.clone();
            if params
                .desc_tbl
                .as_ref()
                .is_none_or(Self::is_cached_reference)
            {
                params.desc_tbl = common.desc_tbl.clone();
            }
            if params.query_globals.is_none() {
                params.query_globals = common.query_globals.clone();
            }
            if params.query_options.is_none() {
                params.query_options = common.query_options.clone();
            }
            if params.resource_info.is_none() {
                params.resource_info = common.resource_info.clone();
            }
            if params.coord.is_none() {
                params.coord = common.coord.clone();
            }

            self.process_fragment(&params)
                .map_err(|err| self.dispatch_error(&params, format!("fragment {idx}: {err}")))?;
        }

        Ok(())
    }

    /// Converts a StarRocks thrift plan fragment to Substrait, with each exchange in `inputs` read
    /// as a stream, logs its protobuf debug output, and returns the translated plan for execution.
    #[instrument(skip_all)]
    fn translate_fragment_logged(
        &self,
        params: &TExecPlanFragmentParams,
        inputs: &[ExchangeInput],
        dump_seq: Option<u64>,
    ) -> std::result::Result<TranslatedPlan, String> {
        let translated = self
            .translator
            .translate_fragment_with_exchange_inputs(params, inputs)
            .map_err(|err| err.to_string())?;
        info!(
            output_names = ?translated.output_names,
            plan = ?translated.plan,
            "translated StarRocks plan fragment"
        );
        Self::dump_substrait(&translated, dump_seq);
        Ok(translated)
    }

    /// `<dir>/<prefix>-<pid>-<seq>.<extension>`. The CNs of a cluster may share one dump directory
    /// and each numbers its own fragments from 0, so the process id keeps their files apart.
    fn dump_path(dir: &str, prefix: &str, seq: u64, extension: &str) -> std::path::PathBuf {
        std::path::Path::new(dir).join(format!(
            "{prefix}-{}-{seq:04}.{extension}",
            std::process::id()
        ))
    }

    /// Writes the translated Substrait plan bytes to `$SIRIUS_CN_DUMP_FRAGMENTS/plan-<pid>-<seq>.substrait`
    /// so a failing plan can be replayed against the engine in isolation. No-op when unset.
    fn dump_substrait(translated: &TranslatedPlan, dump_seq: Option<u64>) {
        let Ok(dir) = std::env::var("SIRIUS_CN_DUMP_FRAGMENTS") else {
            return;
        };
        let Some(seq) = dump_seq else {
            return;
        };
        let path = Self::dump_path(&dir, "plan", seq, "substrait");
        if let Err(err) = std::fs::write(&path, translated.to_substrait_bytes()) {
            tracing::warn!(error = %err, path = %path.display(), "failed to dump substrait plan");
        }
    }

    /// Classifies the fragment output sink: `Ok(true)` for a MySQL text-protocol RESULT_SINK this
    /// CN can encode, `Ok(false)` for a non-result sink, and `Err` for a RESULT_SINK whose format
    /// is not supported yet (binary rows, HTTP/FILE/Arrow Flight, etc.).
    /// The encoder only emits MySQL text rows, so other result-sink formats must be rejected
    /// rather than returned in the wrong wire format.
    fn is_mysql_result_sink(params: &TExecPlanFragmentParams) -> std::result::Result<bool, String> {
        let Some(sink) = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
        else {
            return Ok(false);
        };
        if sink.type_ != TDataSinkType::RESULT_SINK {
            return Ok(false);
        }
        // A RESULT_SINK with no nested detail defaults to MySQL text rows.
        let Some(result_sink) = sink.result_sink.as_ref() else {
            return Ok(true);
        };
        if matches!(result_sink.is_binary_row, Some(true)) {
            return Err("binary-row result sinks are not supported yet".to_string());
        }
        match result_sink.type_ {
            None | Some(TResultSinkType::MYSQL_PROTOCAL) => Ok(true),
            Some(other) => Err(format!("result sink type {other:?} is not supported yet")),
        }
    }

    /// Test hook: once the file `SIRIUS_CN_FAIL_ONCE_FILE` names appears, the next fragment any
    /// CN sharing that path runs fails, before the engine sees it. Removing the file claims the
    /// failure, so exactly one fragment fails per `touch`: an e2e script can fail one query
    /// partway through and check that nothing it held outlives it.
    ///
    /// What the file holds picks the fragment and what happens to it:
    /// - empty: the next fragment fails;
    /// - `remote-sender`: the next fragment that runs on exchange input and sends to another CN
    ///   fails. When a remote frame started it, no `exec_plan_fragment` reply carries its error,
    ///   and only its failure frames tell the query;
    /// - `hang-sender`: the next fragment that sends to another CN stops without running, sending
    ///   neither EOS nor a failure frame, as a dead peer would. Only a deadline ends its query.
    ///
    /// Returns whether this fragment claimed the file holding `kind`.
    fn claim_injection(kind: &str) -> bool {
        let Some(path) = std::env::var_os("SIRIUS_CN_FAIL_ONCE_FILE") else {
            return false;
        };
        let Ok(wanted) = std::fs::read_to_string(&path) else {
            return false;
        };
        wanted.trim() == kind && std::fs::remove_file(&path).is_ok()
    }

    /// The query a fragment belongs to.
    fn query_id(params: &TExecPlanFragmentParams) -> Option<FragmentInstanceId> {
        params
            .params
            .as_ref()
            .map(|exec| FragmentInstanceId::from(&exec.query_id))
    }

    /// Extracts the fragment instance id the FE later passes to `fetch_data`.
    fn fragment_instance_id(params: &TExecPlanFragmentParams) -> Option<FragmentInstanceId> {
        params
            .params
            .as_ref()
            .map(|exec| FragmentInstanceId::from(&exec.fragment_instance_id))
    }

    /// Extracts every parquet path from the binary-thrift attachment and infers one schema they all share.
    async fn file_schema_from_attachment(
        attachment: &[u8],
    ) -> std::result::Result<Vec<PSlotDescriptor>, String> {
        let request = Self::deserialize_binary::<TGetFileSchemaRequest>(attachment)
            .map_err(|err| format!("failed to deserialize TGetFileSchemaRequest: {err}"))?;
        let broker = request.scan_range.broker_scan_range.ok_or_else(|| {
            "TGetFileSchemaRequest scan_range carries no broker_scan_range".to_string()
        })?;
        if broker.ranges.is_empty() {
            return Err("broker_scan_range carries no file ranges".to_string());
        }
        let paths = broker
            .ranges
            .into_iter()
            .map(|range| {
                if range.format_type != TFileFormatType::FORMAT_PARQUET {
                    return Err(format!(
                        "unsupported file format {:?} for '{}'; only parquet schema inference is implemented",
                        range.format_type, range.path
                    ));
                }
                Ok(range.path)
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        crate::file_schema::parquet_files_schema(&paths).await
    }

    /// Deserializes a thrift struct using the StarRocks binary attachment protocol.
    fn deserialize_binary<T>(bytes: &[u8]) -> thrift::Result<T>
    where
        T: TSerializable,
    {
        let mut channel = TBufferChannel::with_capacity(bytes.len(), 0);
        let bytes_copied = channel.set_readable_bytes(bytes);
        if bytes_copied != bytes.len() {
            return Err(thrift::Error::Application(thrift::ApplicationError::new(
                thrift::ApplicationErrorKind::Unknown,
                "failed to stage complete thrift payload".to_string(),
            )));
        }
        let mut protocol = TBinaryInputProtocol::new(channel, true);
        T::read_from_in_protocol(&mut protocol)
    }

    /// Rejects thrift attachment protocols that are not implemented by the Rust CN yet.
    fn ensure_binary_protocol(protocol: Option<&str>) -> std::result::Result<(), String> {
        match protocol.unwrap_or("binary").to_ascii_lowercase().as_str() {
            "binary" => Ok(()),
            other => Err(format!(
                "attachment protocol '{other}' is not supported yet; expected binary"
            )),
        }
    }

    /// Builds the required single-fragment response wrapper around a StarRocks status.
    fn exec_plan_result(status: StatusPb) -> PExecPlanFragmentResult {
        PExecPlanFragmentResult {
            status,
            closed_scan_nodes: Vec::new(),
        }
    }

    /// Builds a `fetch_data` response carrying the FE's packet-sequence and end-of-stream markers.
    fn fetch_data_result(status: StatusPb, packet_seq: i64, eos: bool) -> PFetchDataResult {
        PFetchDataResult {
            status,
            packet_seq: Some(packet_seq),
            eos: Some(eos),
            query_statistics: None,
        }
    }

    /// StarRocks OK status.
    fn ok_status() -> StatusPb {
        StatusPb {
            status_code: TStatusCode::OK.0,
            error_msgs: Vec::new(),
        }
    }

    /// StarRocks CANCELLED status: the query already failed, for the reason in `message`.
    fn cancelled(message: impl Into<String>) -> StatusPb {
        StatusPb {
            status_code: TStatusCode::CANCELLED.0,
            error_msgs: vec![message.into()],
        }
    }

    /// StarRocks INTERNAL_ERROR status carrying a user-visible error message.
    fn internal_error(message: impl Into<String>) -> StatusPb {
        StatusPb {
            status_code: TStatusCode::INTERNAL_ERROR.0,
            error_msgs: vec![message.into()],
        }
    }
}

impl SiriusComputeNodeService {
    /// A guard over a ready receiver's inputs, which it frees unless disarmed.
    fn ready_inputs_guard(&self, inputs: &[ReadyExchangeInput]) -> ReadyInputsGuard {
        ReadyInputsGuard {
            nixl: self.nixl.clone(),
            executor: Arc::clone(&self.executor),
            tokens: inputs
                .iter()
                .flat_map(|input| &input.sources)
                .filter_map(|source| match source {
                    SenderSource::Remote { batches, .. } => Some(batches),
                    SenderSource::LocalParked { .. } => None,
                })
                .flatten()
                .map(|batch| batch.token)
                .collect(),
            slots: inputs
                .iter()
                .flat_map(|input| &input.sources)
                .filter_map(|source| match source {
                    SenderSource::LocalParked { slot, .. } => Some(*slot),
                    SenderSource::Remote { .. } => None,
                })
                .collect(),
        }
    }
}

/// What a panic said, when it said it with a string.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}

/// A ready receiver's inputs, taken for a run: see `take_ready_inputs`.
struct ReceiverInputs {
    guard: ReadyInputsGuard,
    exchanges: Vec<ExchangeInput>,
    inputs: Vec<(i32, Vec<SenderSlot>)>,
    remote: Vec<(i32, i32, Vec<RemoteBatch>)>,
}

/// Frees a ready receiver's inputs when dropped, whichever step failed: translation, a pre-run
/// check, or the engine run itself. Received batches are always released; one the engine pushed
/// is already consumed, and releasing it is a no-op. Parked inputs are released unless the run
/// succeeded (and so relayed them); one the engine already released just errors.
struct ReadyInputsGuard {
    nixl: Option<Arc<dyn NixlEndpoint>>,
    executor: Arc<dyn FragmentExecutor>,
    tokens: Vec<u64>,
    slots: Vec<SenderSlot>,
}

impl Drop for ReadyInputsGuard {
    fn drop(&mut self) {
        if let Some(nixl) = &self.nixl {
            for &token in &self.tokens {
                nixl.release(token);
            }
        }
        for &slot in &self.slots {
            let _ = self.executor.drop_parked(slot);
        }
    }
}

/// Takes the fragment out of `pending`, once.
fn take_pending<T>(pending: &Mutex<Option<T>>) -> Option<T> {
    pending.lock().ok().and_then(|mut ready| ready.take())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use prost::Message;
    use starrocks_thrift::{
        data::TResultBatch,
        data_sinks::{TDataSink, TDataStreamSink, TResultSink},
        descriptors::{TDescriptorTable, TSlotDescriptor, TTableDescriptor, TTupleDescriptor},
        internal_service::{InternalServiceVersion, TPlanFragmentExecParams},
        partitions::{TDataPartition, TPartitionType},
        plan_nodes::{TExchangeNode, TFileScanNode, TPlan, TPlanNode, TPlanNodeType},
        planner::TPlanFragment,
        types::{
            TPrimitiveType, TScalarType, TTableType, TTypeDesc, TTypeNode, TTypeNodeType, TUniqueId,
        },
    };
    use thrift::{protocol::TBinaryOutputProtocol, transport::TIoChannel};
    use tower::{Service, ServiceExt};

    use super::*;
    use crate::{
        fragment_executor::PendingFrees,
        fragment_executor::{FragmentResult, OutputDrain, RingJob, RingNext},
        local_exchange::ExchangeCounts,
        proto::starrocks::{
            PFetchDataRequest, PUniqueId,
            p_internal_service_brpc::{PInternalServiceRouter, SERVICE_NAME, methods},
        },
        prpc,
    };

    /// Runs leaf fragments like the stub but fails every fragment fed by an exchange.
    #[derive(Debug)]
    struct FailingReceivers;

    impl FragmentExecutor for FailingReceivers {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if run.inputs.is_empty() {
                return StubExecutor.run_fragment(run);
            }
            Err("receiver exploded".to_string())
        }
    }

    #[derive(Debug, Default)]
    struct FakeNixl {
        released: Mutex<Vec<u64>>,
        sent: Mutex<Vec<(SocketAddr, SenderSlot)>>,
        /// Every hop fails, as when the peer refuses a frame.
        fail_sends: bool,
        /// Hops streamed while their fragment ran, with the rows each drain delivered.
        streamed: Mutex<Vec<(SocketAddr, SenderSlot, u64)>>,
        /// Hops ended with a failure frame, and its error.
        failed: Mutex<Vec<(SocketAddr, SenderSlot, String)>>,
        /// The CN every failure frame is delivered to, as `transmit_chunk` would.
        peer: Mutex<Option<SiriusComputeNodeService>>,
        /// What a remote merge node answers about each filter.
        topologies: Mutex<HashMap<i32, FilterTopology>>,
        /// Other control messages sent to peers.
        controls: Mutex<Vec<(SocketAddr, NixlEnvelope)>>,
    }

    impl FakeNixl {
        fn failed_slots(&self) -> Vec<SenderSlot> {
            let failed = self.failed.lock().unwrap();
            failed.iter().map(|(_, slot, _)| *slot).collect()
        }
    }

    impl NixlEndpoint for FakeNixl {
        fn local_md(&self) -> Vec<u8> {
            b"local-md".to_vec()
        }

        fn release(&self, token: u64) {
            self.released.lock().unwrap().push(token);
        }

        fn outstanding(&self) -> usize {
            0
        }

        fn send(
            &self,
            peer: SocketAddr,
            slot: SenderSlot,
            _names: Vec<String>,
            _executor: Arc<dyn FragmentExecutor>,
        ) -> Result<(), String> {
            self.sent.lock().unwrap().push((peer, slot));
            if self.fail_sends {
                // As the transport does: the failed hop ends with a failure frame.
                let err = format!("{peer} refused transmit_chunk");
                self.fail(peer, slot, &err);
                return Err(err);
            }
            Ok(())
        }

        fn stream(&self, hops: Vec<StreamHop>) -> Result<(), String> {
            let ends: Vec<_> = hops.iter().map(|hop| (hop.peer, hop.slot)).collect();
            for mut hop in hops {
                let mut rows = 0;
                loop {
                    match hop.drain.next_ring(Duration::from_millis(1)) {
                        Ok(RingNext::Job(job)) => rows += job.rows,
                        Ok(RingNext::Waiting) => {}
                        Ok(RingNext::End) => break,
                        Err(err) => {
                            // As the transport does: every hop ends with a failure frame.
                            for &(peer, slot) in &ends {
                                self.fail(peer, slot, &err);
                            }
                            return Err(err);
                        }
                    }
                }
                self.streamed
                    .lock()
                    .unwrap()
                    .push((hop.peer, hop.slot, rows));
            }
            Ok(())
        }

        fn ring_open(
            &self,
            _request: &PTransmitChunkParams,
            _body: &[u8],
        ) -> Result<Vec<u8>, String> {
            Ok(b"lane-grant".to_vec())
        }

        fn fail(&self, peer: SocketAddr, slot: SenderSlot, error: &str) {
            self.failed
                .lock()
                .unwrap()
                .push((peer, slot, error.to_string()));
            let receiver = self.peer.lock().unwrap().clone();
            if let Some(receiver) = receiver {
                let failed = NixlEnvelope::Failed {
                    error: error.to_string(),
                };
                receiver
                    .handle_nixl_chunk(&nixl_chunk::failed_params(slot), &failed.encode())
                    .unwrap();
            }
        }

        fn control(&self, peer: SocketAddr, envelope: &NixlEnvelope) -> Result<Vec<u8>, String> {
            match envelope {
                NixlEnvelope::Topology { filter_id, .. } => Ok(nixl_chunk::encode_topology(
                    self.topologies.lock().unwrap().get(filter_id),
                )),
                other => {
                    self.controls.lock().unwrap().push((peer, other.clone()));
                    Ok(Vec::new())
                }
            }
        }
    }

    fn nixl_service(
        executor: Arc<dyn FragmentExecutor>,
    ) -> (SiriusComputeNodeService, Arc<FakeNixl>) {
        let nixl = Arc::new(FakeNixl::default());
        let service = SiriusComputeNodeService::with_executor(
            executor,
            &ComputeNodeConfig::default(),
            Some(nixl.clone()),
        );
        (service, nixl)
    }

    fn transmit(
        service: &SiriusComputeNodeService,
        params: PTransmitChunkParams,
        envelope: NixlEnvelope,
    ) -> (StatusPb, Vec<u8>) {
        let response = route(
            service,
            methods::TRANSMIT_CHUNK,
            params.encode_to_vec(),
            envelope.encode(),
        );
        let result = PTransmitChunkResult::decode(response.body.as_slice()).unwrap();
        (result.status.unwrap(), response.attachment)
    }

    /// Frame `seq` from sender 0 to exchange node 2 of fragment instance 10 of query `query`.
    fn packed(query: i64, seq: i64, token: u64) -> (PTransmitChunkParams, NixlEnvelope) {
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(query, 10),
            node_id: 2,
            sender_id: 0,
        };
        let envelope = NixlEnvelope::Packed {
            token,
            rows: 1,
            names: vec!["id".to_string(), "name".to_string()],
        };
        (
            nixl_chunk::packed_params(Some(slot), seq, token == 0),
            envelope,
        )
    }

    #[test]
    fn transmit_chunk_serves_md() {
        let (service, _nixl) = nixl_service(Arc::new(StubExecutor));
        let control = nixl_chunk::control_params;
        let (status, md) = transmit(&service, control(), NixlEnvelope::Md(b"peer".to_vec()));
        assert_eq!(
            (status.status_code, md),
            (TStatusCode::OK.0, b"local-md".to_vec())
        );
    }

    #[test]
    fn a_lane_for_a_query_that_failed_here_is_refused_with_its_cause() {
        let (service, _) = nixl_service(Arc::new(StubExecutor));
        let open = || {
            let slot = SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(46, 10),
                node_id: 2,
                sender_id: 0,
            };
            transmit(
                &service,
                nixl_chunk::packed_params(Some(slot), 0, false),
                NixlEnvelope::RingOpen(b"lane-open".to_vec()),
            )
            .0
        };
        assert_eq!(open().status_code, TStatusCode::OK.0);
        assert_eq!(cancel(&service, 46).status_code, TStatusCode::OK.0);
        let status = open();
        assert_eq!(
            (status.status_code, status.error_msgs),
            (
                TStatusCode::CANCELLED.0,
                vec!["query cancelled: injected".to_string()]
            )
        );
    }

    #[test]
    fn refused_packed_frame_releases_its_token_and_a_duplicate_keeps_it() {
        let (service, nixl) = nixl_service(Arc::new(StubExecutor));
        let (params, envelope) = packed(12, 0, 5);
        for _ in 0..2 {
            let (status, _) = transmit(&service, params.clone(), envelope.clone());
            assert_eq!(status.status_code, TStatusCode::OK.0);
        }
        let (params, envelope) = packed(12, 2, 6);
        let (status, _) = transmit(&service, params, envelope);
        assert!(status.error_msgs[0].contains("lost"), "{status:?}");
        assert_eq!(*nixl.released.lock().unwrap(), [6]);
    }

    #[test]
    fn failed_receiver_releases_its_received_batches() {
        // The frames arrive first, so registering the receiver runs it at once, and it fails.
        let (service, nixl) = nixl_service(Arc::new(FailingReceivers));
        for (seq, token) in [(0, 5), (1, 6), (2, 0)] {
            let (params, envelope) = packed(13, seq, token);
            assert_eq!(
                transmit(&service, params, envelope).0.status_code,
                TStatusCode::OK.0
            );
        }
        let mut receiver = query_fragment(13, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut receiver, 2, 1);
        assert_eq!(
            exec(&service, &receiver).status_code,
            TStatusCode::INTERNAL_ERROR.0
        );
        assert_eq!(*nixl.released.lock().unwrap(), [5, 6]);
    }

    #[test]
    fn remote_stream_sink_destination_ships_over_nixl() {
        let (service, nixl) = nixl_service(Arc::new(StubExecutor));
        let mut sender = query_fragment(14, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        exec_ok(&service, &sender);
        let sent = nixl.sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        assert_eq!((sent[0].0.port(), sent[0].1.node_id), (18060, 2));
    }

    /// The first of two remote hops fails: the second output isn't shipped into a peer's pool for
    /// a query that is failing, both claims are still released, and both receivers get a failure
    /// frame.
    #[test]
    fn a_failed_hop_stops_the_remaining_hops_and_releases_every_claim() {
        let executor = Arc::new(RecordingExecutor::default());
        let nixl = Arc::new(FakeNixl {
            fail_sends: true,
            ..FakeNixl::default()
        });
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            Some(nixl.clone()),
        );
        let mut sender = query_fragment(23, 12, scan_node(0, 0), stream_sink(2));
        let params = sender.params.as_mut().unwrap();
        params.sender_id = Some(0);
        params.destinations = Some(
            [10, 11]
                .iter()
                .map(|&instance| {
                    TPlanFragmentDestination::new(
                        TUniqueId::new(params.query_id.hi, instance),
                        None,
                        Some(TNetworkAddress::new("127.0.0.1".to_string(), 18060)),
                        None,
                    )
                })
                .collect(),
        );
        assert_eq!(
            sender_end(&service, &mut sender).0,
            TStatusCode::INTERNAL_ERROR
        );
        assert_eq!(
            nixl.sent.lock().unwrap().len(),
            1,
            "the second hop never ran"
        );
        let dropped = executor.dropped.lock().unwrap();
        for instance in [10, 11] {
            assert!(
                dropped.iter().any(|slot| slot.fragment_instance_id
                    == FragmentInstanceId::from_halves(23, instance)),
                "the claim on instance {instance} was released: {dropped:?}"
            );
        }
        // The failed hop and the one that never ran both end with a failure frame.
        let failed = nixl.failed_slots();
        for instance in [10, 11] {
            assert!(
                failed.iter().any(|slot| slot.fragment_instance_id
                    == FragmentInstanceId::from_halves(23, instance)),
                "instance {instance}'s receiver got a failure frame: {failed:?}"
            );
        }
    }

    #[derive(Debug, Default)]
    struct StreamingExecutor {
        rows: u64,
        fail_with: Option<String>,
        dropped: Mutex<Vec<SenderSlot>>,
    }

    #[derive(Debug)]
    struct ScriptedDrain(Vec<Result<RingNext, String>>);

    impl OutputDrain for ScriptedDrain {
        fn next_ring(&mut self, _timeout: Duration) -> Result<RingNext, String> {
            self.0.pop().unwrap_or(Ok(RingNext::End))
        }
    }

    impl FragmentExecutor for StreamingExecutor {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if let Some(handoff) = run.drains {
                let drains = handoff
                    .streams
                    .iter()
                    .map(|_| {
                        let last = match &self.fail_with {
                            Some(err) => Err(err.clone()),
                            None => Ok(RingNext::End),
                        };
                        let job = RingJob {
                            token: 1,
                            rows: self.rows,
                            stream_bytes: 0,
                            header: Vec::new(),
                        };
                        // Popped from the back: waiting, a batch, then the last outcome.
                        let script = vec![last, Ok(RingNext::Job(job)), Ok(RingNext::Waiting)];
                        Box::new(ScriptedDrain(script)) as Box<dyn OutputDrain>
                    })
                    .collect();
                handoff.respond.send(drains).unwrap();
            }
            match &self.fail_with {
                Some(err) => Err(err.clone()),
                None => Ok(None),
            }
        }

        fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
            self.dropped.lock().unwrap().push(slot);
            Ok(())
        }
    }

    #[test]
    fn remote_outputs_stream_while_the_fragment_runs() {
        let executor = Arc::new(StreamingExecutor {
            rows: 42,
            ..Default::default()
        });
        let (service, nixl) = nixl_service(executor.clone());
        let mut sender = query_fragment(14, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        exec_ok(&service, &sender);
        assert!(
            nixl.sent.lock().unwrap().is_empty(),
            "nothing ships from parked output"
        );
        let streamed = nixl.streamed.lock().unwrap();
        assert_eq!(streamed.len(), 1);
        assert_eq!(
            (streamed[0].0.port(), streamed[0].1.node_id, streamed[0].2),
            (18060, 2, 42)
        );
        // The streamed slot's claim is dropped once the hop finished.
        assert_eq!(*executor.dropped.lock().unwrap(), vec![streamed[0].1]);
    }

    #[test]
    fn a_failed_streaming_run_reports_the_run_error() {
        let executor = Arc::new(StreamingExecutor {
            rows: 1,
            fail_with: Some("scan exploded".to_string()),
            ..Default::default()
        });
        let (service, nixl) = nixl_service(executor);
        let mut sender = query_fragment(14, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        let (code, error) = sender_end(&service, &mut sender);
        assert_ne!(code, TStatusCode::OK);
        assert!(error.contains("scan exploded"), "{error}");
        assert!(nixl.sent.lock().unwrap().is_empty());
    }

    /// A `BIGINT` slot reference to slot `slot_id` of tuple `tuple_id`.
    fn slot_ref_expr(slot_id: i32, tuple_id: i32) -> starrocks_thrift::exprs::TExpr {
        use starrocks_thrift::exprs::{TExpr, TExprNode, TExprNodeType, TSlotRef};
        TExpr::new(vec![TExprNode {
            node_type: TExprNodeType::SLOT_REF,
            type_: scalar_type(TPrimitiveType::BIGINT),
            opcode: None,
            num_children: 0,
            agg_expr: None,
            bool_literal: None,
            case_expr: None,
            date_literal: None,
            float_literal: None,
            int_literal: None,
            in_predicate: None,
            is_null_pred: None,
            like_pred: None,
            literal_pred: None,
            slot_ref: Some(TSlotRef::new(slot_id, tuple_id)),
            string_literal: None,
            tuple_is_null_pred: None,
            info_func: None,
            decimal_literal: None,
            output_scale: -1,
            fn_call_expr: None,
            large_int_literal: None,
            output_column: None,
            output_type: None,
            vector_opcode: None,
            fn_: None,
            vararg_start_idx: None,
            child_type: None,
            vslot_ref: None,
            used_subfield_names: None,
            binary_literal: None,
            copy_flag: None,
            check_is_out_of_bounds: None,
            use_vectorized: None,
            has_nullable_child: None,
            is_nullable: None,
            child_type_desc: None,
            is_monotonic: None,
            dict_query_expr: None,
            dictionary_get_expr: None,
            is_index_only_filter: None,
            is_nondeterministic: None,
            cast_struct_by_name: None,
        }])
    }

    /// Broadcast join filter 0 on `id`, probed by scan node 0.
    fn id_filter() -> starrocks_thrift::runtime_filter::TRuntimeFilterDescription {
        starrocks_thrift::runtime_filter::TRuntimeFilterDescription {
            filter_id: Some(0),
            build_expr: Some(slot_ref_expr(1, 0)),
            plan_node_id_to_target_expr: Some(std::collections::BTreeMap::from([(
                0,
                slot_ref_expr(1, 0),
            )])),
            build_join_mode: Some(
                starrocks_thrift::runtime_filter::TRuntimeFilterBuildJoinMode::BROADCAST,
            ),
            build_plan_node_id: Some(5),
            ..Default::default()
        }
    }

    /// Instance 1 of `query`: a join of probe exchange 1 and build exchange 2 that builds
    /// `id_filter`. It never runs: exchange 1 waits for a second sender.
    fn filter_builder(query: i64) -> TExecPlanFragmentParams {
        let mut join = scan_node(5, 0);
        join.node_type = TPlanNodeType::HASH_JOIN_NODE;
        join.num_children = 2;
        join.file_scan_node = None;
        join.hash_join_node = Some(starrocks_thrift::plan_nodes::THashJoinNode::new(
            starrocks_thrift::plan_nodes::TJoinOp::INNER_JOIN,
            Vec::new(),
            None,
            None,
            None,
            None,
            None,
            None,
            Some(vec![id_filter()]),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let plan = TPlan::new(vec![
            join,
            exchange_plan_node(1, 0),
            exchange_plan_node(2, 0),
        ]);
        let mut params = fragment_params(Some(plan), Some(desc_table()));
        params.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        params.params = Some(exec_params(
            TUniqueId::new(query, 0),
            TUniqueId::new(query, 1),
        ));
        expect_senders(&mut params, 1, 2);
        expect_senders(&mut params, 2, 1);
        params
    }

    /// Instance 3 of `query`: a scan probing `id_filter`, sending to exchange 1.
    fn probing_scan(query: i64) -> TExecPlanFragmentParams {
        let mut scan = scan_node(0, 0);
        scan.probe_runtime_filters = Some(vec![id_filter()]);
        let mut params = query_fragment(query, 3, scan, stream_sink(1));
        send_to(&mut params, 1, 8060);
        params
    }

    /// Instance 2 of `query`: the build side's only sender, completing exchange 2.
    fn build_sender(query: i64) -> TExecPlanFragmentParams {
        let mut params = query_fragment(query, 2, scan_node(10, 0), stream_sink(2));
        send_to(&mut params, 1, 8060);
        params
    }

    /// One recorded run: its stream input ids, its filters, and whether it had a fallback.
    type RecordedRun = (Vec<i32>, Vec<FilterRun>, bool);

    /// Records each sender run's filters and key stream schemas; reads `stats` as every filter's
    /// keys.
    #[derive(Debug)]
    struct FilterRecorder {
        stats: crate::fragment_executor::KeyStats,
        runs: Mutex<Vec<RecordedRun>>,
        /// Each run's output names and output slots, in run order.
        outputs: Mutex<Vec<(Vec<String>, Vec<SenderSlot>)>>,
        /// Whether reading keys fails.
        unreadable_keys: bool,
        /// Parked outputs dropped.
        dropped: Mutex<Vec<SenderSlot>>,
    }

    impl FilterRecorder {
        fn new(stats: crate::fragment_executor::KeyStats) -> Self {
            Self {
                stats,
                runs: Mutex::new(Vec::new()),
                outputs: Mutex::new(Vec::new()),
                unreadable_keys: false,
                dropped: Mutex::new(Vec::new()),
            }
        }

        /// The runs recorded so far, once at least `count` happened.
        fn wait_for(&self, count: usize) -> Vec<RecordedRun> {
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                let runs = self.runs.lock().unwrap().clone();
                if runs.len() >= count || std::time::Instant::now() > deadline {
                    return runs;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    impl FragmentExecutor for FilterRecorder {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            self.outputs
                .lock()
                .unwrap()
                .push((run.plan.output_names.clone(), run.outputs.clone()));
            self.runs.lock().unwrap().push((
                run.plan
                    .stream_inputs
                    .iter()
                    .map(|input| input.node_id)
                    .collect(),
                run.filters.clone(),
                run.fallback.is_some(),
            ));
            Ok(Some(FragmentResult::new(Vec::new())))
        }

        fn key_stats(
            &self,
            _keys: &crate::fragment_executor::FilterKeys,
        ) -> Result<crate::fragment_executor::KeyStats, String> {
            if self.unreadable_keys {
                return Err("injected: keys unreadable".to_string());
            }
            Ok(self.stats)
        }

        fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
            self.dropped.lock().unwrap().push(slot);
            Ok(())
        }
    }

    fn sparse_keys() -> crate::fragment_executor::KeyStats {
        crate::fragment_executor::KeyStats {
            rows: 2_000_000,
            distinct: 2_000_000,
            min: 1,
            max: 100_000_000,
        }
    }

    #[test]
    fn a_scan_waits_for_its_runtime_filter_and_reads_the_keys_in_place() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        exec_ok(&service, &filter_builder(31));
        exec_ok(&service, &probing_scan(31));
        assert!(executor.runs.lock().unwrap().is_empty(), "the scan waits");
        assert_eq!(service.filters.deferred(), 1);

        // The build side completes: the scan runs with the filter, reading the parked keys.
        exec_ok(&service, &build_sender(31));
        let runs = executor.wait_for(2);
        assert_eq!(runs.len(), 2, "{runs:?}");
        let (streams, filters, fallback) = &runs[1];
        assert_eq!(streams, &vec![FILTER_STREAM_BASE]);
        assert!(fallback, "the unfiltered plan stays as a fallback");
        assert_eq!(
            filters,
            &vec![FilterRun {
                stream_id: FILTER_STREAM_BASE as u64,
                keys: FilterKeys {
                    column: 0,
                    sources: vec![crate::fragment_executor::KeySource::Parked(SenderSlot {
                        fragment_instance_id: FragmentInstanceId::from_halves(31, 1),
                        node_id: 2,
                        sender_id: 0,
                    })],
                },
                rows: 2_000_000,
            }]
        );
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn keys_that_fill_their_range_are_not_applied() {
        let executor = Arc::new(FilterRecorder::new(crate::fragment_executor::KeyStats {
            rows: 2_000_000,
            distinct: 2_000_000,
            min: 1,
            max: 2_000_000,
        }));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        exec_ok(&service, &filter_builder(32));
        exec_ok(&service, &probing_scan(32));
        exec_ok(&service, &build_sender(32));
        let runs = executor.wait_for(2);
        assert_eq!(runs.len(), 2, "{runs:?}");
        assert_eq!(runs[1], (Vec::new(), Vec::new(), false));
    }

    #[test]
    fn a_scan_whose_filter_is_built_elsewhere_runs_at_once() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        exec_ok(&service, &probing_scan(33));
        assert_eq!(
            *executor.runs.lock().unwrap(),
            vec![(Vec::new(), Vec::new(), false)]
        );
        assert_eq!(service.filters.deferred(), 0);
    }

    /// Runs `run` with the log lines of this thread captured, and returns them.
    fn captured_logs(run: impl FnOnce()) -> String {
        #[derive(Clone)]
        struct Buffer(Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Buffer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let buffer = Buffer(Arc::default());
        let writer = buffer.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, run);
        String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn a_scan_logs_why_its_filters_are_not_applied() {
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FilterRecorder::new(sparse_keys())),
            &ComputeNodeConfig::default(),
            None,
        );
        // A broadcast filter with no receiver building it here.
        // `run_or_register` directly: a routed request runs on another thread.
        let logs = captured_logs(|| {
            service.run_or_register(&probing_scan(35)).unwrap();
        });
        assert!(
            logs.contains(r#"filter_id=0 node=0 outcome="skipped" reason="not_built_on_this_cn""#),
            "{logs}"
        );
        // A colocate join's filter: neither broadcast nor sent in shares.
        let mut colocate = probing_scan(36);
        let plan = colocate.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0].probe_runtime_filters.as_mut().unwrap()[0].build_join_mode =
            Some(starrocks_thrift::runtime_filter::TRuntimeFilterBuildJoinMode::COLOCATE);
        let logs = captured_logs(|| {
            service.run_or_register(&colocate).unwrap();
        });
        assert!(
            logs.contains(r#"outcome="skipped" reason="not_broadcast""#),
            "{logs}"
        );
    }

    #[test]
    fn a_receiver_logs_the_filters_it_cannot_build() {
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FilterRecorder::new(sparse_keys())),
            &ComputeNodeConfig::default(),
            None,
        );
        let mut builder = filter_builder(37);
        let plan = builder.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        let join = plan.nodes[0].hash_join_node.as_mut().unwrap();
        join.build_runtime_filters.as_mut().unwrap()[0].build_join_mode =
            Some(starrocks_thrift::runtime_filter::TRuntimeFilterBuildJoinMode::COLOCATE);
        let logs = captured_logs(|| {
            service.run_or_register(&builder).unwrap();
        });
        assert!(
            logs.contains(r#"filter_id=0 node=5 outcome="skipped" reason="not_broadcast""#),
            "{logs}"
        );
    }

    #[test]
    fn purging_a_query_drops_its_waiting_scans() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        exec_ok(&service, &filter_builder(34));
        exec_ok(&service, &probing_scan(34));
        assert_eq!(service.filters.deferred(), 1);
        service.fail_and_purge(FragmentInstanceId::from_halves(34, 0), "injected");
        assert_eq!(service.filters.deferred(), 0);
    }

    /// The brpc address of a CN built from `ComputeNodeConfig::default()`.
    fn local_brpc() -> TNetworkAddress {
        TNetworkAddress::new("127.0.0.1".to_string(), 8060)
    }

    /// `id_filter`, built by a partitioned join whose `shares` instances each hold a share; the
    /// test CN is its merge node.
    fn share_filter(shares: i32) -> starrocks_thrift::runtime_filter::TRuntimeFilterDescription {
        let mut filter = id_filter();
        filter.build_join_mode =
            Some(starrocks_thrift::runtime_filter::TRuntimeFilterBuildJoinMode::PARTITIONED);
        filter.runtime_filter_merge_nodes = Some(vec![local_brpc()]);
        filter.layout = Some(starrocks_thrift::runtime_filter::TRuntimeFilterLayout {
            filter_id: Some(0),
            num_instances: Some(shares),
            ..Default::default()
        });
        filter
    }

    /// Instance `10 + sender` of `query`: sender `sender` of a partitioned join, holding a share
    /// of `share_filter` in build exchange 2. It never runs: probe exchange 1 waits for a second
    /// sender.
    fn share_holder(query: i64, sender: i32, shares: i32) -> TExecPlanFragmentParams {
        let mut params = filter_builder(query);
        let plan = params.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0]
            .hash_join_node
            .as_mut()
            .unwrap()
            .build_runtime_filters = Some(vec![share_filter(shares)]);
        let exec = params.params.as_mut().unwrap();
        exec.fragment_instance_id = TUniqueId::new(query, 10 + i64::from(sender));
        exec.sender_id = Some(sender);
        params
    }

    /// What the FE sends the root fragment on the merge node: `share_filter` has `shares` shares
    /// and is probed by `probers`.
    fn filter_topology(
        shares: i32,
        probers: &[(i64, i64, TNetworkAddress)],
    ) -> starrocks_thrift::runtime_filter::TRuntimeFilterParams {
        starrocks_thrift::runtime_filter::TRuntimeFilterParams {
            id_to_prober_params: Some(BTreeMap::from([(
                0,
                probers
                    .iter()
                    .map(|(hi, lo, address)| {
                        starrocks_thrift::runtime_filter::TRuntimeFilterProberParams::new(
                            TUniqueId::new(*hi, *lo),
                            address.clone(),
                        )
                    })
                    .collect(),
            )])),
            runtime_filter_builder_number: Some(BTreeMap::from([(0, shares)])),
            runtime_filter_max_size: None,
            skew_join_runtime_filters: None,
        }
    }

    /// Instance 3 of `query`: a scan probing `share_filter`, sending to exchange 1 of instance 10.
    fn share_probing_scan(query: i64, shares: i32) -> TExecPlanFragmentParams {
        let mut scan = scan_node(0, 0);
        scan.probe_runtime_filters = Some(vec![share_filter(shares)]);
        let mut params = query_fragment(query, 3, scan, stream_sink(1));
        send_to(&mut params, 10, 8060);
        params
    }

    /// Instance `20 + sender` of `query`: the build side's only sender to holder `10 + sender`.
    fn share_build_sender(query: i64, sender: i64) -> TExecPlanFragmentParams {
        let mut params = query_fragment(query, 20 + sender, scan_node(10, 0), stream_sink(2));
        send_to(&mut params, 10 + sender, 8060);
        params
    }

    /// A partitioned filter's merge node, two share holders and the probing scan, on one CN.
    /// Returns the scan's filter exchange.
    fn share_query(service: &SiriusComputeNodeService, query: i64) -> ExchangeKey {
        share_query_with(service, query, 2, share_probing_scan(query, 2))
    }

    /// `share_query`, with the merge node counting `topology_shares` shares, and `scan` probing.
    fn share_query_with(
        service: &SiriusComputeNodeService,
        query: i64,
        topology_shares: i32,
        scan: TExecPlanFragmentParams,
    ) -> ExchangeKey {
        let mut root = share_holder(query, 0, 2);
        root.params.as_mut().unwrap().runtime_filter_params = Some(filter_topology(
            topology_shares,
            &[(query, 3, local_brpc())],
        ));
        exec_ok(service, &root);
        exec_ok(service, &share_holder(query, 1, 2));
        exec_ok(service, &scan);
        runtime_filters::filter_exchange(FragmentInstanceId::from_halves(query, 3), 0)
    }

    /// Gives the probe expression of `params`' scan the type `primitive`.
    fn set_probe_type(params: &mut TExecPlanFragmentParams, primitive: TPrimitiveType) {
        let plan = params.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        let filter = &mut plan.nodes[0].probe_runtime_filters.as_mut().unwrap()[0];
        let probe = filter
            .plan_node_id_to_target_expr
            .as_mut()
            .unwrap()
            .get_mut(&0)
            .unwrap();
        probe.nodes[0].type_ = scalar_type(primitive);
    }

    /// `builder` with a join the translator accepts, so the receiver can run: build exchange 2
    /// reads tuple 1 (`o_orderkey`), joined on `l_orderkey = o_orderkey`, which keys its filters.
    fn translatable(mut builder: TExecPlanFragmentParams) -> TExecPlanFragmentParams {
        builder.desc_tbl = Some(tpch_desc_table());
        let plan = builder.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[2] = exchange_plan_node(2, 1);
        let join = plan.nodes[0].hash_join_node.as_mut().unwrap();
        join.eq_join_conjuncts = vec![starrocks_thrift::plan_nodes::TEqJoinCondition {
            left: slot_ref_expr(1, 0),
            right: slot_ref_expr(3, 1),
            opcode: Some(starrocks_thrift::opcodes::TExprOpcode::EQ),
        }];
        for filter in join.build_runtime_filters.as_mut().unwrap() {
            filter.build_expr = Some(slot_ref_expr(3, 1));
        }
        builder
    }

    /// A service on `executor` whose scans wait at most 5 s: a test that sees a scan run sooner
    /// shows it didn't wait out the limit.
    fn short_wait_service(executor: Arc<FilterRecorder>) -> SiriusComputeNodeService {
        let mut service =
            SiriusComputeNodeService::with_executor(executor, &ComputeNodeConfig::default(), None);
        service.filter_wait = Duration::from_secs(5);
        service
    }

    /// Whether `service`'s deferred scan stopped waiting within 3 s, before a 5 s wait runs out.
    fn scan_ran_within_3s(service: &SiriusComputeNodeService) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while service.filters.deferred() > 0 {
            if std::time::Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    }

    /// The runs of `executor` once `count` happened, or after 3 s: before a 5 s wait runs out.
    fn runs_within_3s(executor: &FilterRecorder, count: usize) -> Vec<RecordedRun> {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            let runs = executor.runs.lock().unwrap().clone();
            if runs.len() >= count || std::time::Instant::now() > deadline {
                return runs;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The share slot holder `10 + sender` sends the scan of `query`.
    fn share_slot(query: i64, sender: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(query, 3),
            node_id: FILTER_STREAM_BASE,
            sender_id: sender,
        }
    }

    /// The deferred scan's run among `runs`: the only one with an unfiltered fallback.
    fn filtered_scan_run(runs: &[RecordedRun]) -> Option<&RecordedRun> {
        runs.iter().find(|(_, _, fallback)| *fallback)
    }

    #[test]
    fn a_partitioned_filter_is_applied_once_every_share_arrived() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        let exchange = share_query(&service, 41);
        assert!(executor.runs.lock().unwrap().is_empty(), "the scan waits");
        assert_eq!(service.filters.deferred(), 1);

        // Holder 10's build side completes: it sends its share, the distinct keys of exchange 2.
        exec_ok(&service, &share_build_sender(41, 0));
        let runs = executor.wait_for(2);
        assert_eq!(runs.len(), 2, "{runs:?}");
        let (streams, filters, fallback) = &runs[1];
        assert_eq!(streams, &vec![FILTER_STREAM_BASE]);
        assert!(!fallback);
        assert_eq!(
            filters[0].keys.sources,
            vec![KeySource::Parked(SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(41, 10),
                node_id: 2,
                sender_id: 0,
            })]
        );
        assert_eq!(
            executor.outputs.lock().unwrap()[1],
            (vec![SHARE_KEYS.to_string()], vec![share_slot(41, 0)])
        );
        // One share of two: the scan still waits.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(service.filters.deferred(), 1);
        assert!(filtered_scan_run(&executor.runs.lock().unwrap()).is_none());

        // The second share completes the filter: the scan semi-joins the union of both.
        exec_ok(&service, &share_build_sender(41, 1));
        let runs = executor.wait_for(5);
        let (streams, filters, _) = filtered_scan_run(&runs).expect("the scan ran filtered");
        assert_eq!(streams, &vec![FILTER_STREAM_BASE]);
        assert_eq!(
            filters,
            &vec![FilterRun {
                stream_id: FILTER_STREAM_BASE as u64,
                keys: FilterKeys {
                    column: 0,
                    sources: vec![
                        KeySource::Parked(share_slot(41, 0)),
                        KeySource::Parked(share_slot(41, 1)),
                    ],
                },
                rows: 2_000_000,
            }]
        );
        assert_eq!(service.filters.deferred(), 0);
        // The shares are freed with the scan, and a late copy is dropped.
        assert!(service.exchanges.shares(exchange).is_empty());
        assert!(
            !service
                .exchanges
                .push_share_frame(exchange, 7, 0, true, vec!["late".to_string()], None)
                .unwrap()
        );
    }

    #[test]
    fn a_partitioned_filter_missing_a_share_scans_unfiltered_after_the_wait() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let mut service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        service.filter_wait = Duration::from_millis(100);
        let exchange = share_query(&service, 42);
        exec_ok(&service, &share_build_sender(42, 0));
        // Its build sender, its share, then the scan, unfiltered.
        let runs = executor.wait_for(3);
        assert_eq!(runs.len(), 3, "{runs:?}");
        assert_eq!(runs[2], (Vec::new(), Vec::new(), false));
        assert_eq!(service.filters.deferred(), 0);
        assert!(service.exchanges.shares(exchange).is_empty());

        // The late share is dropped where it lands, not kept for a scan that already ran.
        exec_ok(&service, &share_build_sender(42, 1));
        executor.wait_for(5);
        std::thread::sleep(Duration::from_millis(50));
        assert!(service.exchanges.shares(exchange).is_empty());
    }

    #[test]
    fn purging_a_query_drops_its_shares_and_the_scan_waiting_for_them() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        share_query(&service, 43);
        exec_ok(&service, &share_build_sender(43, 0));
        executor.wait_for(2);
        std::thread::sleep(Duration::from_millis(50));
        assert!(service.exchanges.counts().parked_senders > 0);
        assert_eq!(
            service.filters.held_shares(),
            1,
            "holder 11's build never completes"
        );

        let logs = captured_logs(|| {
            service.fail_and_purge(FragmentInstanceId::from_halves(43, 0), "injected")
        });
        assert!(
            logs.contains(r#"filter_id=0 node=-1 outcome="skipped" reason="purged""#),
            "{logs}"
        );
        assert_eq!(service.filters.deferred(), 0);
        assert_eq!(service.exchanges.counts(), ExchangeCounts::default());
        assert_eq!(service.filters.held_shares(), 0);
        assert!(
            service
                .filters
                .topology(FragmentInstanceId::from_halves(43, 0), 0)
                .is_none()
        );
    }

    #[test]
    fn shares_too_large_to_send_exactly_bound_the_scan_by_their_range() {
        let executor = Arc::new(FilterRecorder::new(crate::fragment_executor::KeyStats {
            rows: 40_000_000,
            distinct: 30_000_000,
            min: 7,
            max: 6_000_000_000,
        }));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        share_query(&service, 44);
        exec_ok(&service, &share_build_sender(44, 0));
        exec_ok(&service, &share_build_sender(44, 1));
        let runs = executor.wait_for(5);
        // Each holder sent only its bounds.
        let outputs = executor.outputs.lock().unwrap().clone();
        let shares: Vec<_> = outputs
            .iter()
            .filter(|(_, slots)| slots.iter().all(|slot| slot.node_id == FILTER_STREAM_BASE))
            .collect();
        assert_eq!(shares.len(), 2, "{outputs:?}");
        assert!(
            shares
                .iter()
                .all(|(names, _)| names == &[runtime_filter::SHARE_BOUNDS])
        );
        // The scan keeps the keys between them: a predicate, with no key stream to read.
        let scan = filtered_scan_run(&runs).expect("the scan ran filtered");
        assert_eq!(scan, &(Vec::new(), Vec::new(), true));
    }

    /// A batch of `rows` rows from remote sender 0 into holder 10's build exchange 2; with no
    /// rows, its end of stream.
    fn build_rows(service: &SiriusComputeNodeService, query: i64, seq: i64, rows: u64) {
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(query, 10),
            node_id: 2,
            sender_id: 0,
        };
        let envelope = NixlEnvelope::Packed {
            token: if rows == 0 { 0 } else { 7 + seq as u64 },
            rows,
            names: vec!["id".to_string(), "name".to_string()],
        };
        let (status, _) = transmit(
            service,
            nixl_chunk::packed_params(Some(slot), seq, rows == 0),
            envelope,
        );
        assert_eq!(status.status_code, TStatusCode::OK.0, "{status:?}");
    }

    #[test]
    fn a_holder_whose_build_outgrows_exact_keys_releases_its_probers_at_once() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let (mut service, _) = nixl_service(executor.clone());
        service.filter_wait = Duration::from_secs(5);
        let exchange = share_query(&service, 58);
        let limit = runtime_filters::max_share_keys();

        // As many rows as a share sends exactly: the scan still waits for the build.
        build_rows(&service, 58, 0, limit);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(service.filters.deferred(), 1);

        // One more row: the share can only be bounds, so the scan runs unfiltered at once,
        // before the build exchange completes.
        build_rows(&service, 58, 1, 1);
        assert_eq!(
            runs_within_3s(&executor, 1),
            vec![(Vec::new(), Vec::new(), false)]
        );
        assert_eq!(service.filters.deferred(), 0);
        assert_eq!(
            service.filters.held_shares(),
            1,
            "holder 11's share is still held"
        );

        // Both builds complete: holder 10's share never ships, and holder 11's, which the scan
        // no longer waits for, is dropped where it lands.
        build_rows(&service, 58, 2, 0);
        exec_ok(&service, &share_build_sender(58, 1));
        let runs = executor.wait_for(3);
        let share_runs: Vec<_> = executor
            .outputs
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, slots)| slots.iter().all(|slot| slot.node_id == FILTER_STREAM_BASE))
            .map(|(_, slots)| slots[0].sender_id)
            .collect();
        assert_eq!(share_runs, vec![1], "{runs:?}");
        std::thread::sleep(Duration::from_millis(50));
        assert!(service.exchanges.shares(exchange).is_empty());
        assert_eq!(service.filters.held_shares(), 0);
    }

    #[test]
    fn a_share_ships_to_a_remote_prober_named_by_a_remote_merge_node() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let (service, nixl) = nixl_service(executor.clone());
        let peer = TNetworkAddress::new("127.0.0.1".to_string(), 8061);
        nixl.topologies.lock().unwrap().insert(
            0,
            FilterTopology {
                shares: 2,
                probers: vec![(FragmentInstanceId::from_halves(45, 3), peer.clone())],
            },
        );
        let mut holder = share_holder(45, 1, 2);
        let plan = holder.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0]
            .hash_join_node
            .as_mut()
            .unwrap()
            .build_runtime_filters
            .as_mut()
            .unwrap()[0]
            .runtime_filter_merge_nodes = Some(vec![peer]);
        exec_ok(&service, &holder);
        exec_ok(&service, &share_build_sender(45, 1));
        executor.wait_for(2);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while nixl.sent.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            *nixl.sent.lock().unwrap(),
            vec![("127.0.0.1:8061".parse().unwrap(), share_slot(45, 1))]
        );
    }

    #[test]
    fn a_scan_does_not_wait_for_a_partitioned_filter_on_a_key_no_holder_can_send() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        let mut scan = share_probing_scan(51, 2);
        set_probe_type(&mut scan, TPrimitiveType::VARCHAR);
        let logs = captured_logs(|| service.process_fragment(&scan).unwrap());
        senders_done(&service);
        assert!(
            logs.contains(r#"filter_id=0 node=0 outcome="skipped" reason="key_type""#),
            "{logs}"
        );
        assert_eq!(service.filters.deferred(), 0);
        assert_eq!(
            executor.runs.lock().unwrap().len(),
            1,
            "the scan ran at once"
        );
    }

    #[test]
    fn a_holder_that_cannot_send_its_share_releases_its_probers_at_once() {
        let mut recorder = FilterRecorder::new(sparse_keys());
        recorder.unreadable_keys = true;
        let executor = Arc::new(recorder);
        let service = short_wait_service(executor.clone());
        share_query(&service, 52);
        exec_ok(&service, &share_build_sender(52, 0));
        // Its build sender, then the scan, unfiltered, without a share run or the wait.
        let runs = runs_within_3s(&executor, 2);
        assert_eq!(runs.len(), 2, "{runs:?}");
        assert_eq!(runs[1], (Vec::new(), Vec::new(), false));
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn a_holder_that_cannot_send_tells_a_remote_prober_which_stops_waiting() {
        let mut recorder = FilterRecorder::new(sparse_keys());
        recorder.unreadable_keys = true;
        let (holder_cn, nixl) = nixl_service(Arc::new(recorder));
        let peer = TNetworkAddress::new("127.0.0.1".to_string(), 8061);
        nixl.topologies.lock().unwrap().insert(
            0,
            FilterTopology {
                shares: 2,
                probers: vec![(FragmentInstanceId::from_halves(53, 3), peer.clone())],
            },
        );
        let mut holder = share_holder(53, 1, 2);
        let plan = holder.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0]
            .hash_join_node
            .as_mut()
            .unwrap()
            .build_runtime_filters
            .as_mut()
            .unwrap()[0]
            .runtime_filter_merge_nodes = Some(vec![peer]);
        exec_ok(&holder_cn, &holder);
        exec_ok(&holder_cn, &share_build_sender(53, 1));
        let abandoned = NixlEnvelope::ShareAbandoned {
            instance: FragmentInstanceId::from_halves(53, 3),
            filter_id: 0,
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while nixl.controls.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            *nixl.controls.lock().unwrap(),
            vec![("127.0.0.1:8061".parse().unwrap(), abandoned.clone())]
        );

        // The prober's CN hears it and runs the scan unfiltered.
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let (prober_cn, _) = nixl_service(executor.clone());
        exec_ok(&prober_cn, &share_probing_scan(53, 2));
        assert_eq!(prober_cn.filters.deferred(), 1);
        let (status, _) = transmit(&prober_cn, nixl_chunk::control_params(), abandoned);
        assert_eq!(status.status_code, TStatusCode::OK.0);
        let runs = runs_within_3s(&executor, 1);
        assert_eq!(runs, vec![(Vec::new(), Vec::new(), false)]);
    }

    /// `share_holder`, building `share_filter` from another join's output instead of an
    /// exchange: no CN can build it.
    fn unbuilt_share_holder(query: i64, sender: i32, shares: i32) -> TExecPlanFragmentParams {
        let mut params = share_holder(query, sender, shares);
        let plan = params.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        let mut build = plan.nodes[0].clone();
        build.node_id = 6;
        build.num_children = 0;
        build.hash_join_node.as_mut().unwrap().build_runtime_filters = None;
        plan.nodes[2] = build;
        params
    }

    #[test]
    fn a_holder_that_cannot_build_its_share_releases_its_probers_at_once() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        let mut root = unbuilt_share_holder(54, 0, 2);
        root.params.as_mut().unwrap().runtime_filter_params =
            Some(filter_topology(2, &[(54, 3, local_brpc())]));
        exec_ok(&service, &root);
        exec_ok(&service, &share_probing_scan(54, 2));
        exec_ok(&service, &unbuilt_share_holder(54, 1, 2));
        // The scan, unfiltered, without a share run or the wait.
        let runs = runs_within_3s(&executor, 1);
        assert_eq!(runs, vec![(Vec::new(), Vec::new(), false)]);
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn a_holder_that_cannot_build_its_share_tells_a_remote_prober() {
        let (holder_cn, nixl) = nixl_service(Arc::new(FilterRecorder::new(sparse_keys())));
        let peer = TNetworkAddress::new("127.0.0.1".to_string(), 8061);
        nixl.topologies.lock().unwrap().insert(
            0,
            FilterTopology {
                shares: 2,
                probers: vec![(FragmentInstanceId::from_halves(56, 3), peer.clone())],
            },
        );
        let mut holder = unbuilt_share_holder(56, 1, 2);
        let plan = holder.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0]
            .hash_join_node
            .as_mut()
            .unwrap()
            .build_runtime_filters
            .as_mut()
            .unwrap()[0]
            .runtime_filter_merge_nodes = Some(vec![peer]);
        exec_ok(&holder_cn, &holder);
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while nixl.controls.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        // The prober the merge node lists hears that no share comes from this CN.
        assert_eq!(
            *nixl.controls.lock().unwrap(),
            vec![(
                "127.0.0.1:8061".parse().unwrap(),
                NixlEnvelope::ShareAbandoned {
                    instance: FragmentInstanceId::from_halves(56, 3),
                    filter_id: 0,
                }
            )]
        );
    }

    #[test]
    fn a_scan_does_not_wait_for_a_partitioned_filter_without_a_merge_node() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        let mut scan = share_probing_scan(55, 2);
        let plan = scan.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0].probe_runtime_filters.as_mut().unwrap()[0].runtime_filter_merge_nodes =
            Some(Vec::new());
        let logs = captured_logs(|| service.process_fragment(&scan).unwrap());
        senders_done(&service);
        assert!(
            logs.contains(r#"filter_id=0 node=0 outcome="skipped" reason="no_merge_node""#),
            "{logs}"
        );
        assert_eq!(service.filters.deferred(), 0);
        assert_eq!(
            executor.runs.lock().unwrap().len(),
            1,
            "the scan ran at once"
        );
    }

    /// Instance 10 of `query`, q17's shape on one CN: join 5 builds partitioned filter 1 on the
    /// output of join 6, which builds partitioned filter 0 on the same key from build exchange 2.
    /// One share makes each filter. It never runs: its probe exchanges wait for second senders.
    fn served_filter_holder(query: i64) -> TExecPlanFragmentParams {
        let mut params = share_holder(query, 0, 1);
        let plan = params.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        let below = plan.nodes[0].clone();
        let mut top = below.clone();
        let mut served = share_filter(1);
        served.filter_id = Some(1);
        top.hash_join_node.as_mut().unwrap().build_runtime_filters = Some(vec![served]);
        let mut below = below;
        below.node_id = 6;
        plan.nodes = vec![
            top,
            exchange_plan_node(1, 0),
            below,
            exchange_plan_node(3, 0),
            exchange_plan_node(2, 0),
        ];
        expect_senders(&mut params, 3, 2);
        // The merge node's topology: filter 1 has one share, probed by instance 3.
        params.params.as_mut().unwrap().runtime_filter_params =
            Some(starrocks_thrift::runtime_filter::TRuntimeFilterParams {
                id_to_prober_params: Some(BTreeMap::from([(
                    1,
                    vec![
                        starrocks_thrift::runtime_filter::TRuntimeFilterProberParams::new(
                            TUniqueId::new(query, 3),
                            local_brpc(),
                        ),
                    ],
                )])),
                runtime_filter_builder_number: Some(BTreeMap::from([(1, 1)])),
                runtime_filter_max_size: None,
                skew_join_runtime_filters: None,
            });
        params
    }

    #[test]
    fn a_filter_built_on_a_joins_output_is_applied_with_the_keys_of_the_exchange_serving_it() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        exec_ok(&service, &served_filter_holder(57));
        let mut scan = share_probing_scan(57, 1);
        let plan = scan.fragment.as_mut().unwrap().plan.as_mut().unwrap();
        plan.nodes[0].probe_runtime_filters.as_mut().unwrap()[0].filter_id = Some(1);
        exec_ok(&service, &scan);
        assert_eq!(service.filters.deferred(), 1, "the scan waits for filter 1");

        // Exchange 2 completes: filter 1's share is filter 0's keys, read from exchange 2.
        exec_ok(&service, &share_build_sender(57, 0));
        let runs = executor.wait_for(3);
        let stream = FILTER_STREAM_BASE + 1;
        let share = runs
            .iter()
            .find(|(_, filters, _)| filters.first().map(|f| f.stream_id) == Some(stream as u64))
            .expect("filter 1's share ran");
        assert_eq!(
            share.1[0].keys.sources,
            vec![KeySource::Parked(SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(57, 10),
                node_id: 2,
                sender_id: 0,
            })]
        );
        let (streams, filters, _) = filtered_scan_run(&runs).expect("the scan ran filtered");
        assert_eq!(streams, &vec![stream]);
        assert_eq!(
            filters[0].keys.sources,
            vec![KeySource::Parked(SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(57, 3),
                node_id: stream,
                sender_id: 0,
            })]
        );
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn a_share_is_sent_when_its_build_exchange_completes_the_receiver() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        // One holder, whose probe exchange is fed by a plain scan, not the probing one.
        let mut root = translatable(share_holder(54, 0, 1));
        expect_senders(&mut root, 1, 1);
        root.params.as_mut().unwrap().runtime_filter_params =
            Some(filter_topology(1, &[(54, 3, local_brpc())]));
        exec_ok(&service, &root);
        let mut scan = share_probing_scan(54, 1);
        send_to(&mut scan, 11, 8060);
        exec_ok(&service, &scan);
        let mut probe_side = query_fragment(54, 30, scan_node(12, 0), stream_sink(1));
        send_to(&mut probe_side, 10, 8060);
        exec_ok(&service, &probe_side);
        // The build side completes the receiver: its share goes before it runs.
        exec_ok(&service, &share_build_sender(54, 0));
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let filtered = loop {
            let runs = executor.runs.lock().unwrap().clone();
            if let Some(run) = filtered_scan_run(&runs) {
                break Some(run.clone());
            }
            if std::time::Instant::now() > deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let (_, filters, _) = filtered.expect("the scan ran filtered, without waiting");
        assert_eq!(
            filters[0].keys.sources,
            vec![KeySource::Parked(share_slot(54, 0))]
        );
    }

    #[test]
    fn a_broadcast_filter_is_read_when_its_build_exchange_completes_the_receiver() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        let mut builder = translatable(filter_builder(55));
        expect_senders(&mut builder, 1, 1);
        exec_ok(&service, &builder);
        let mut scan = probing_scan(55);
        send_to(&mut scan, 11, 8060);
        exec_ok(&service, &scan);
        assert_eq!(service.filters.deferred(), 1);
        let mut probe_side = query_fragment(55, 30, scan_node(12, 0), stream_sink(1));
        send_to(&mut probe_side, 1, 8060);
        exec_ok(&service, &probe_side);
        exec_ok(&service, &build_sender(55));
        let runs = runs_within_3s(&executor, 4);
        let (_, filters, _) = filtered_scan_run(&runs).expect("the scan ran filtered");
        assert_eq!(
            filters[0].keys.sources,
            vec![KeySource::Parked(SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(55, 1),
                node_id: 2,
                sender_id: 0,
            })]
        );
    }

    #[test]
    fn shares_that_disagree_on_their_count_leave_the_scan_unfiltered() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        // The scan's plan says 2 shares; the merge node told the holders 3.
        share_query_with(&service, 56, 3, share_probing_scan(56, 2));
        exec_ok(&service, &share_build_sender(56, 0));
        exec_ok(&service, &share_build_sender(56, 1));
        assert!(scan_ran_within_3s(&service), "the scan waited");
        let runs = runs_within_3s(&executor, 5);
        assert!(filtered_scan_run(&runs).is_none(), "{runs:?}");
    }

    #[test]
    fn shares_of_another_key_type_leave_the_scan_unfiltered() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        // The holders build BIGINT keys; the scan probes an INTEGER key.
        let mut scan = share_probing_scan(57, 2);
        set_probe_type(&mut scan, TPrimitiveType::INT);
        share_query_with(&service, 57, 2, scan);
        exec_ok(&service, &share_build_sender(57, 0));
        assert!(scan_ran_within_3s(&service), "the scan waited");
        let runs = runs_within_3s(&executor, 3);
        assert!(filtered_scan_run(&runs).is_none(), "{runs:?}");
    }

    /// Instance 40 of `query`: a join of a scan probing `id_filter` with exchange 30 (one
    /// sender), sending to exchange `sink_node` of instance `sink_instance`. `filter_builder`'s
    /// instance 1 builds the filter.
    fn nonleaf_prober(query: i64, sink_node: i32, sink_instance: i64) -> TExecPlanFragmentParams {
        let builder = filter_builder(query);
        let mut join = builder.fragment.unwrap().plan.unwrap().nodes[0].clone();
        join.node_id = 41;
        let hash_join = join.hash_join_node.as_mut().unwrap();
        hash_join.build_runtime_filters = None;
        hash_join.eq_join_conjuncts = vec![starrocks_thrift::plan_nodes::TEqJoinCondition {
            left: slot_ref_expr(1, 0),
            right: slot_ref_expr(3, 1),
            opcode: Some(starrocks_thrift::opcodes::TExprOpcode::EQ),
        }];
        let mut scan = scan_node(0, 0);
        scan.probe_runtime_filters = Some(vec![id_filter()]);
        let plan = TPlan::new(vec![join, scan, exchange_plan_node(30, 1)]);
        let mut params = fragment_params(Some(plan), Some(tpch_desc_table()));
        params.fragment.as_mut().unwrap().output_sink = Some(stream_sink(sink_node));
        params.params = Some(exec_params(
            TUniqueId::new(query, 0),
            TUniqueId::new(query, 40),
        ));
        expect_senders(&mut params, 30, 1);
        send_to(&mut params, sink_instance, 8060);
        params
    }

    /// Instance 50 of `query`: the only sender to `nonleaf_prober`'s exchange 30.
    fn nonleaf_input(query: i64) -> TExecPlanFragmentParams {
        let mut params = query_fragment(query, 50, scan_node(12, 1), stream_sink(30));
        params.desc_tbl = Some(tpch_desc_table());
        send_to(&mut params, 40, 8060);
        params
    }

    /// The run of `nonleaf_prober` among `runs`: the one reading exchange 30.
    fn nonleaf_run(runs: &[RecordedRun]) -> Option<&RecordedRun> {
        runs.iter().find(|(streams, _, _)| streams.contains(&30))
    }

    #[test]
    fn a_fragment_reading_exchanges_waits_for_a_filter_built_on_its_cn() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        exec_ok(&service, &filter_builder(61));
        // Its output goes to exchange 1, not to the filter's build exchange 2.
        exec_ok(&service, &nonleaf_prober(61, 1, 1));
        exec_ok(&service, &nonleaf_input(61));
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            nonleaf_run(&executor.runs.lock().unwrap()).is_none(),
            "its inputs are here, its filter isn't"
        );
        assert_eq!(service.filters.deferred(), 1);

        exec_ok(&service, &build_sender(61));
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let run = loop {
            if let Some(run) = nonleaf_run(&executor.runs.lock().unwrap()) {
                break Some(run.clone());
            }
            if std::time::Instant::now() > deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let (streams, filters, fallback) = run.expect("it ran once its filter was built");
        assert!(streams.contains(&FILTER_STREAM_BASE), "{streams:?}");
        assert!(fallback);
        assert_eq!(
            filters[0].keys.sources,
            vec![KeySource::Parked(SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(61, 1),
                node_id: 2,
                sender_id: 0,
            })]
        );
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn a_fragment_whose_filter_build_may_need_its_output_does_not_wait() {
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FilterRecorder::new(sparse_keys())),
            &ComputeNodeConfig::default(),
            None,
        );
        service.run_or_register(&filter_builder(62)).unwrap();
        senders_done(&service);
        // Its output feeds the filter's own build exchange: waiting would hold both.
        let logs = captured_logs(|| {
            service.run_or_register(&nonleaf_prober(62, 2, 1)).unwrap();
        });
        assert!(
            logs.contains(r#"filter_id=0 node=0 outcome="skipped" reason="deadlock_unproven""#),
            "{logs}"
        );
        assert!(logs.contains("reads this fragment's output"), "{logs}");
        // Its output goes to a fragment this CN hasn't seen: nothing proves it safe.
        let mut unknown = nonleaf_prober(62, 99, 7);
        unknown.params.as_mut().unwrap().fragment_instance_id = TUniqueId::new(62, 41);
        let logs = captured_logs(|| {
            service.run_or_register(&unknown).unwrap();
        });
        assert!(
            logs.contains(r#"outcome="skipped" reason="deadlock_unproven""#)
                && logs.contains("isn't on this CN"),
            "{logs}"
        );
        assert!(
            service
                .filters
                .take_pending(FragmentInstanceId::from_halves(62, 40))
                .is_none()
        );
        assert!(
            service
                .filters
                .take_pending(FragmentInstanceId::from_halves(62, 41))
                .is_none()
        );
    }

    #[test]
    fn a_fragment_reading_exchanges_runs_unfiltered_when_its_filter_is_late() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let mut service = short_wait_service(executor.clone());
        service.filter_wait = Duration::from_millis(300);
        exec_ok(&service, &filter_builder(63));
        exec_ok(&service, &nonleaf_prober(63, 1, 1));
        exec_ok(&service, &nonleaf_input(63));
        assert!(
            nonleaf_run(&executor.runs.lock().unwrap()).is_none(),
            "it waits for its filter first"
        );
        assert_eq!(service.filters.deferred(), 1);
        let runs = runs_within_3s(&executor, 2);
        let (streams, filters, fallback) = nonleaf_run(&runs).expect("it ran unfiltered");
        assert_eq!(streams, &vec![30]);
        assert!(filters.is_empty());
        assert!(!fallback);
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn a_fragment_with_a_sink_the_check_cant_follow_does_not_wait() {
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FilterRecorder::new(sparse_keys())),
            &ComputeNodeConfig::default(),
            None,
        );
        service.run_or_register(&filter_builder(66)).unwrap();
        let mut multicast = nonleaf_prober(66, 1, 1);
        let sink = multicast
            .fragment
            .as_mut()
            .unwrap()
            .output_sink
            .as_mut()
            .unwrap();
        sink.type_ = TDataSinkType::MULTI_CAST_DATA_STREAM_SINK;
        sink.stream_sink = None;
        // It fails later as unsupported; what matters is it isn't proven safe to wait.
        let logs = captured_logs(|| {
            let _ = service.run_or_register(&multicast);
        });
        assert!(
            logs.contains(
                r#"outcome="skipped" reason="deadlock_unproven" detail="unsupported sink""#
            ),
            "{logs}"
        );
    }

    #[test]
    fn a_waiting_fragment_whose_timer_cannot_start_runs_unfiltered_at_once() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        exec_ok(&service, &filter_builder(67));
        exec_ok(&service, &nonleaf_prober(67, 1, 1));
        exec_ok(&service, &nonleaf_input(67));
        assert_eq!(service.filters.deferred(), 1);
        service.timer_not_started(FragmentInstanceId::from_halves(67, 40), "injected");
        let runs = executor.runs.lock().unwrap().clone();
        let (streams, filters, _) = nonleaf_run(&runs).expect("it ran at once");
        assert_eq!(streams, &vec![30]);
        assert!(filters.is_empty());
        assert_eq!(service.filters.deferred(), 0);
    }

    #[test]
    fn purging_a_query_frees_a_fragment_waiting_on_its_inputs_for_a_filter() {
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = short_wait_service(executor.clone());
        exec_ok(&service, &filter_builder(64));
        exec_ok(&service, &nonleaf_prober(64, 1, 1));
        exec_ok(&service, &nonleaf_input(64));
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(service.filters.deferred(), 1);
        let logs = captured_logs(|| {
            service.fail_and_purge(FragmentInstanceId::from_halves(64, 0), "injected")
        });
        assert!(logs.contains(r#"reason="purged""#), "{logs}");
        assert_eq!(service.filters.deferred(), 0);
        // The input it held is freed, not left parked.
        assert!(executor.dropped.lock().unwrap().contains(&SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(64, 40),
            node_id: 30,
            sender_id: 0,
        }));
        assert_eq!(service.exchanges.counts(), ExchangeCounts::default());

        // One still waiting for its inputs is forgotten too.
        exec_ok(&service, &filter_builder(65));
        exec_ok(&service, &nonleaf_prober(65, 1, 1));
        service.fail_and_purge(FragmentInstanceId::from_halves(65, 0), "injected");
        assert!(
            service
                .filters
                .take_pending(FragmentInstanceId::from_halves(65, 40))
                .is_none()
        );
    }

    #[test]
    fn the_merge_node_answers_who_probes_a_filter() {
        let (service, _) = nixl_service(Arc::new(StubExecutor));
        let ask = |filter_id| {
            let (status, reply) = transmit(
                &service,
                nixl_chunk::control_params(),
                NixlEnvelope::Topology {
                    query: FragmentInstanceId::from_halves(46, 0),
                    filter_id,
                },
            );
            assert_eq!(status.status_code, TStatusCode::OK.0);
            nixl_chunk::decode_topology(&reply).unwrap()
        };
        assert_eq!(ask(0), None, "not known before the root fragment arrives");
        let mut root = share_holder(46, 0, 2);
        root.params.as_mut().unwrap().runtime_filter_params =
            Some(filter_topology(2, &[(46, 3, local_brpc())]));
        exec_ok(&service, &root);
        assert_eq!(
            ask(0),
            Some(FilterTopology {
                shares: 2,
                probers: vec![(FragmentInstanceId::from_halves(46, 3), local_brpc())],
            })
        );
        assert_eq!(ask(1), None);
    }

    #[test]
    fn exec_plan_fragment_translates_supported_scan() {
        // A supported one-node file scan should translate successfully and return OK.
        let result = call_exec_plan_fragment(
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            },
            serialize_binary(&supported_fragment()),
        );

        assert_eq!(result.status.status_code, TStatusCode::OK.0);
    }

    #[test]
    fn exec_plan_fragment_returns_internal_error_for_bad_attachment() {
        // Malformed thrift attachments are method-level StarRocks failures, not PRPC failures.
        let result = call_exec_plan_fragment(
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            },
            b"not thrift".to_vec(),
        );

        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("failed to deserialize"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn exec_plan_fragment_rejects_unsupported_attachment_protocol() {
        // The generated service accepts the protobuf request, but the CN currently only
        // implements binary thrift attachments.
        let result = call_exec_plan_fragment(
            PExecPlanFragmentRequest {
                attachment_protocol: Some("compact".to_string()),
            },
            serialize_binary(&supported_fragment()),
        );

        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("not supported yet"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn exec_batch_plan_fragments_translates_tpch_single_node_scans() {
        // This mirrors FE batch dispatch (`Deployer.execRemoteBatchFragmentsAsync`): the query's
        // descriptor table is in common_param, and each instance carries its scan plan and an
        // empty descriptor table marked cached.
        let query_id = TUniqueId::new(9, 1);
        let mut desc = tpch_desc_table();
        desc.is_cached = Some(false);
        let mut common = fragment_params(None, Some(desc));
        common.params = Some(exec_params(query_id.clone(), TUniqueId::new(9, 2)));
        let instance = |plan, instance_id| {
            let cached = TDescriptorTable::new(None, Vec::new(), None, Some(true));
            let mut params = fragment_params(Some(plan), Some(cached));
            // #2038: every fragment needs a place to send its results.
            params.fragment.as_mut().unwrap().output_sink = Some(result_sink());
            params.params = Some(exec_params(query_id.clone(), instance_id));
            params
        };
        let batch = TExecBatchPlanFragmentsParams::new(
            Some(common),
            Some(vec![
                instance(scan_plan(0, 0), TUniqueId::new(9, 2)),
                instance(scan_plan(1, 1), TUniqueId::new(9, 3)),
            ]),
        );
        let result = call_exec_batch_plan_fragments(
            PExecBatchPlanFragmentsRequest {
                attachment_protocol: Some("binary".to_string()),
            },
            serialize_binary(&batch),
        );

        let status = result.status.unwrap();
        assert_eq!(
            status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            status.error_msgs
        );
    }

    #[test]
    fn router_rejects_unknown_service_at_prpc_layer() {
        // Unknown services are rejected by the generated BRPC router before protobuf
        // request decoding reaches the concrete PInternalService implementation.
        let request = prpc::Request::new(
            "OtherService",
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&supported_fragment()),
        );

        let err = call_router(request).unwrap_err();

        assert!(err.to_string().contains("service name"));
    }

    #[test]
    fn exec_plan_fragment_executes_result_sink_and_fetch_data_drains_it() {
        // A root RESULT_SINK fragment is executed (stub) and buffered; fetch_data returns the
        // rows once, then reports end-of-stream. exec and fetch share one service so they share
        // the result store.
        let service = SiriusComputeNodeService::new();

        let mut params = supported_fragment();
        params.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        params.params = Some(exec_params(TUniqueId::new(0, 1), TUniqueId::new(0, 7)));

        let exec = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&params),
        );
        let exec = PExecPlanFragmentResult::decode(exec.body.as_slice()).unwrap();
        assert_eq!(exec.status.status_code, TStatusCode::OK.0);

        // First fetch returns the buffered rows in the attachment, eos = false.
        let first = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(0, 7),
            Vec::new(),
        );
        let first_result = PFetchDataResult::decode(first.body.as_slice()).unwrap();
        assert_eq!(first_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(first_result.eos, Some(false));
        let batch = SiriusComputeNodeService::deserialize_binary::<TResultBatch>(&first.attachment)
            .unwrap();
        // The stub emits one row of "stub" per output column ("id", "name"); each is a MySQL
        // length-encoded string (len 4, then the bytes).
        assert_eq!(batch.rows.len(), 1);
        assert_eq!(
            batch.rows[0],
            vec![0x04, b's', b't', b'u', b'b', 0x04, b's', b't', b'u', b'b']
        );

        // Second fetch reports end-of-stream with no attachment.
        let second = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(0, 7),
            Vec::new(),
        );
        let second_result = PFetchDataResult::decode(second.body.as_slice()).unwrap();
        assert_eq!(second_result.eos, Some(true));
        assert!(second.attachment.is_empty());
    }

    #[test]
    fn fetch_data_for_unknown_fragment_is_an_error() {
        // A poll for an id this CN never buffered must fail loudly, not look like an empty result.
        let service = SiriusComputeNodeService::new();
        let response = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(0, 123),
            Vec::new(),
        );
        let result = PFetchDataResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("no buffered result"),
            "{:?}",
            result.status.error_msgs
        );
        assert!(response.attachment.is_empty());
    }

    #[test]
    fn exec_batch_plan_fragments_buffers_result_sink_instance() {
        // The FE may dispatch the root via batch dispatch; it must also execute + buffer the
        // RESULT_SINK instance so fetch_data returns rows instead of a silent empty result.
        let service = SiriusComputeNodeService::new();
        let mut root = fragment_params(Some(scan_plan(0, 0)), None);
        root.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        root.params = Some(exec_params(TUniqueId::new(0, 1), TUniqueId::new(0, 55)));
        let batch = TExecBatchPlanFragmentsParams::new(
            Some(fragment_params(None, Some(desc_table()))),
            Some(vec![root]),
        );

        let exec = route(
            &service,
            methods::EXEC_BATCH_PLAN_FRAGMENTS,
            PExecBatchPlanFragmentsRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&batch),
        );
        let exec = PExecBatchPlanFragmentsResult::decode(exec.body.as_slice()).unwrap();
        assert_eq!(exec.status.unwrap().status_code, TStatusCode::OK.0);

        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(0, 55),
            Vec::new(),
        );
        let fetched_result = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(fetched_result.eos, Some(false));
        let result_batch =
            SiriusComputeNodeService::deserialize_binary::<TResultBatch>(&fetched.attachment)
                .unwrap();
        assert_eq!(result_batch.rows.len(), 1);
    }

    #[test]
    fn cached_descriptor_reference_reuses_query_descriptor_table() {
        let service = SiriusComputeNodeService::new();
        let query_id = TUniqueId::new(4, 2);

        let mut initial = fragment_params(None, Some(desc_table()));
        initial.params = Some(exec_params(query_id.clone(), TUniqueId::new(4, 3)));
        service
            .resolve_descriptor_table(&initial)
            .expect("cache initial descriptor table");

        let cached = TDescriptorTable::new(None, Vec::new(), None, Some(true));
        let mut reference = fragment_params(None, Some(cached));
        reference.params = Some(exec_params(query_id, TUniqueId::new(4, 4)));
        let resolved = service
            .resolve_descriptor_table(&reference)
            .expect("resolve cached descriptor table");

        let desc = resolved.desc_tbl.expect("resolved descriptor table");
        assert_eq!(desc.slot_descriptors.unwrap().len(), 2);
        assert_eq!(desc.tuple_descriptors.len(), 1);
        assert_eq!(desc.table_descriptors.unwrap().len(), 1);
    }

    #[test]
    fn cached_descriptor_reference_requires_prior_query_table() {
        let service = SiriusComputeNodeService::new();
        let cached = TDescriptorTable::new(None, Vec::new(), None, Some(true));
        let mut reference = fragment_params(None, Some(cached));
        reference.params = Some(exec_params(TUniqueId::new(7, 1), TUniqueId::new(7, 2)));

        let err = service.resolve_descriptor_table(&reference).unwrap_err();
        assert!(err.contains("descriptor table cache miss"), "{err}");
    }

    #[test]
    fn exec_plan_fragment_rejects_unsupported_result_sink_format() {
        // The encoder only emits MySQL text rows; a non-MySQL result sink must be rejected rather
        // than returned in the wrong wire format.
        let service = SiriusComputeNodeService::new();
        let mut params = supported_fragment();
        params.fragment.as_mut().unwrap().output_sink =
            Some(result_sink_typed(TResultSinkType::STATISTIC));
        params.params = Some(exec_params(TUniqueId::new(0, 1), TUniqueId::new(0, 9)));

        let response = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&params),
        );
        let result = PExecPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("not supported"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn receiver_runs_once_its_local_sender_parks() {
        // Receiver-first, as the FE dispatches: the RESULT_SINK receiver waits, then a leaf
        // DATA_STREAM_SINK completes its sender set and the result becomes fetchable.
        let service = SiriusComputeNodeService::new();
        let mut receiver = query_fragment(8, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut receiver, 2, 1);
        exec_ok(&service, &receiver);
        let mut sender = query_fragment(8, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        exec_ok(&service, &sender);

        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(8, 10),
            Vec::new(),
        );
        let fetched_result = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(fetched_result.eos, Some(false));
        let batch =
            SiriusComputeNodeService::deserialize_binary::<TResultBatch>(&fetched.attachment)
                .unwrap();
        assert_eq!(batch.rows.len(), 1);
    }

    #[test]
    fn remote_stream_sink_destination_is_an_error() {
        let service = SiriusComputeNodeService::new();
        let mut sender = query_fragment(9, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        let status = exec(&service, &sender);
        assert_eq!(status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            status.error_msgs[0].contains("no NIXL transport"),
            "{:?}",
            status.error_msgs
        );
    }

    #[test]
    fn failed_intermediate_fragment_fails_the_waiting_result() {
        // leaf -> merge -> result on one CN. The merge fails once its sender completes it, and the
        // result's fetch_data reports that error instead of waiting for rows.
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FailingReceivers),
            &ComputeNodeConfig::default(),
            None,
        );
        let mut result = query_fragment(10, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        exec_ok(&service, &result);
        let mut merge = query_fragment(10, 11, exchange_plan_node(3, 0), stream_sink(2));
        expect_senders(&mut merge, 3, 1);
        send_to(&mut merge, 10, 8060);
        exec_ok(&service, &merge);
        let mut leaf = query_fragment(10, 12, scan_node(0, 0), stream_sink(3));
        send_to(&mut leaf, 11, 8060);
        // The leaf itself runs fine; the merge it completes fails on its own.
        assert_eq!(sender_end(&service, &mut leaf).0, TStatusCode::OK);

        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(10, 10),
            Vec::new(),
        );
        let fetched = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            fetched.status.error_msgs[0].contains("receiver exploded"),
            "{:?}",
            fetched.status.error_msgs
        );
    }

    #[test]
    fn failed_receiver_registration_fails_the_waiting_result() {
        // A repeated dispatch of a waiting result fragment is refused, and fetch_data reports that
        // at once instead of waiting out RESULT_WAIT.
        let service = SiriusComputeNodeService::new();
        let mut result = query_fragment(11, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        exec_ok(&service, &result);
        assert_eq!(
            exec(&service, &result).status_code,
            TStatusCode::INTERNAL_ERROR.0
        );

        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(11, 10),
            Vec::new(),
        );
        let fetched = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert!(
            fetched.status.error_msgs[0].contains("duplicate receiver registration"),
            "{:?}",
            fetched.status.error_msgs
        );
    }

    /// Runs fragments like the stub and records the parked output the service drops.
    #[derive(Debug, Default)]
    struct RecordingExecutor {
        dropped: Mutex<Vec<SenderSlot>>,
    }

    impl FragmentExecutor for RecordingExecutor {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
            self.dropped.lock().unwrap().push(slot);
            Ok(())
        }
    }

    fn recording_service() -> (SiriusComputeNodeService, Arc<RecordingExecutor>) {
        let executor = Arc::new(RecordingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        (service, executor)
    }

    fn cancel(service: &SiriusComputeNodeService, query: i64) -> StatusPb {
        cancel_with(service, query, None, Some("injected"))
    }

    /// As the FE sends it: the query id, an instance id, the reason, and for INTERNAL_ERROR the
    /// error that failed the query.
    fn cancel_with(
        service: &SiriusComputeNodeService,
        query: i64,
        reason: Option<crate::proto::starrocks::PPlanFragmentCancelReason>,
        message: Option<&str>,
    ) -> StatusPb {
        let request = PCancelPlanFragmentRequest {
            finst_id: PUniqueId { hi: query, lo: 1 },
            cancel_reason: reason.map(|reason| reason as i32),
            is_pipeline: Some(true),
            query_id: Some(PUniqueId { hi: query, lo: 0 }),
            error_message: message.map(str::to_string),
        };
        let response = route(
            service,
            methods::CANCEL_PLAN_FRAGMENT,
            request.encode_to_vec(),
            Vec::new(),
        );
        PCancelPlanFragmentResult::decode(response.body.as_slice())
            .unwrap()
            .status
    }

    fn fetch_error(service: &SiriusComputeNodeService, query: i64, instance: i64) -> String {
        let fetched = route(
            service,
            methods::FETCH_DATA,
            fetch_request(query, instance),
            Vec::new(),
        );
        let fetched = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        fetched.status.error_msgs.join("; ")
    }

    #[test]
    fn cancel_purges_the_query_and_frees_what_it_held() {
        // A receiver waiting on two senders has two batches from one of them when the FE cancels.
        let (service, nixl) = nixl_service(Arc::new(StubExecutor));
        let mut result = query_fragment(15, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 2);
        exec_ok(&service, &result);
        for (seq, token) in [(0, 5), (1, 6)] {
            let (params, envelope) = packed(15, seq, token);
            assert_eq!(
                transmit(&service, params, envelope).0.status_code,
                TStatusCode::OK.0
            );
        }
        assert_eq!(service.exchanges.counts().remote_batches, 2);

        assert_eq!(cancel(&service, 15).status_code, TStatusCode::OK.0);
        assert_eq!(*nixl.released.lock().unwrap(), [5, 6]);
        assert_eq!(service.exchanges.counts(), ExchangeCounts::default());
        assert!(fetch_error(&service, 15, 10).contains("injected"));

        // A frame that arrives after the cancel is refused with the query's cause, as CANCELLED so
        // its sender passes that cause on, and its buffers are freed.
        let (params, envelope) = packed(15, 2, 7);
        let (status, _) = transmit(&service, params, envelope);
        assert_eq!(
            (status.status_code, status.error_msgs),
            (
                TStatusCode::CANCELLED.0,
                vec!["query cancelled: injected".to_string()]
            )
        );
        assert_eq!(*nixl.released.lock().unwrap(), [5, 6, 7]);
        // A repeated cancel, one per CN instance, is harmless.
        assert_eq!(cancel(&service, 15).status_code, TStatusCode::OK.0);
    }

    #[test]
    fn a_failed_fragment_purges_its_query_on_this_cn() {
        // One sender has parked its output for the waiting result when another fragment of the
        // same query fails: the parked output is dropped, not left waiting forever.
        let (service, executor) = recording_service();
        let mut result = query_fragment(16, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 2);
        exec_ok(&service, &result);
        let mut parked = query_fragment(16, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut parked, 10, 8060);
        exec_ok(&service, &parked);
        assert_eq!(service.exchanges.counts().parked_senders, 1);

        let mut failing = query_fragment(16, 12, scan_node(0, 0), stream_sink(2));
        send_to(&mut failing, 10, 18060);
        assert_eq!(
            exec(&service, &failing).status_code,
            TStatusCode::INTERNAL_ERROR.0
        );
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(16, 10),
            node_id: 2,
            sender_id: 0,
        };
        assert_eq!(*executor.dropped.lock().unwrap(), [slot]);
        assert_eq!(service.exchanges.counts(), ExchangeCounts::default());
        assert!(fetch_error(&service, 16, 10).contains("no NIXL transport"));
    }

    #[test]
    fn a_receiver_that_fails_before_running_frees_its_parked_inputs() {
        // A remote sender and a local one disagree on column names, so the ready receiver fails
        // binding its exchange, before the engine ever takes its inputs.
        let executor = Arc::new(RecordingExecutor::default());
        let (service, nixl) = nixl_service(executor.clone());
        let mut result = query_fragment(17, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 2);
        exec_ok(&service, &result);
        let remote = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(17, 10),
            node_id: 2,
            sender_id: 1,
        };
        let envelope = NixlEnvelope::Packed {
            token: 5,
            rows: 1,
            names: vec!["not_the_scan_output".to_string()],
        };
        let (status, _) = transmit(
            &service,
            nixl_chunk::packed_params(Some(remote), 0, true),
            envelope,
        );
        assert_eq!(status.status_code, TStatusCode::OK.0);

        let mut local = query_fragment(17, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut local, 10, 8060);
        // The sender itself runs fine; the receiver it completes fails on its own.
        assert_eq!(sender_end(&service, &mut local).0, TStatusCode::OK);
        let parked = SenderSlot {
            sender_id: 0,
            ..remote
        };
        assert_eq!(*executor.dropped.lock().unwrap(), [parked]);
        assert_eq!(*nixl.released.lock().unwrap(), [5]);
        assert_eq!(service.exchanges.counts(), ExchangeCounts::default());
    }

    /// Two CNs on one host, each with its own fake NIXL side: `a` serves the default brpc port
    /// 8060 and `b` port 18060, so a destination on one is remote to the other. Each one's failure
    /// frames reach the other, as `transmit_chunk` would deliver them.
    fn two_cns(
        a: Arc<dyn FragmentExecutor>,
        b: Arc<dyn FragmentExecutor>,
    ) -> [(SiriusComputeNodeService, Arc<FakeNixl>); 2] {
        let (a, a_nixl) = nixl_service(a);
        let b_nixl = Arc::new(FakeNixl::default());
        let b = SiriusComputeNodeService::with_executor(
            b,
            &ComputeNodeConfig {
                brpc_port: 18060,
                ..ComputeNodeConfig::default()
            },
            Some(b_nixl.clone()),
        );
        *a_nixl.peer.lock().unwrap() = Some(b.clone());
        *b_nixl.peer.lock().unwrap() = Some(a.clone());
        [(a, a_nixl), (b, b_nixl)]
    }

    /// Whether `service` holds nothing of any query, and owes the FE no report.
    fn idle(service: &SiriusComputeNodeService) -> bool {
        service.exchanges.counts() == ExchangeCounts::default()
            && service.filters.deferred() == 0
            && service.reports.owed() == 0
    }

    /// Waits for the senders `service` runs off their dispatch calls to finish.
    fn senders_done(service: &SiriusComputeNodeService) {
        eventually("the dispatched senders to finish", || {
            service.senders_running.load(Ordering::SeqCst) == 0
        });
    }

    /// Waits for `done`, which another thread makes true.
    fn eventually(what: &str, done: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn exchange_slot(query: i64, instance: i64, node_id: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(query, instance),
            node_id,
            sender_id: 0,
        }
    }

    /// Fails every run before the fragment produces anything.
    #[derive(Debug)]
    struct FailingRuns;

    impl FragmentExecutor for FailingRuns {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, _run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            Err("scan exploded".to_string())
        }
    }

    #[test]
    fn receivers_fail_at_once_when_their_remote_sender_fails() {
        // A sender on CN a broadcasts to two result receivers on CN b and fails: mid-stream, so
        // the transport ends its hops, or before any hop shipped, so the CN does. Either way both
        // receivers get a failure frame, and b's fetch_data reports the sender's error at once
        // instead of waiting out RESULT_WAIT.
        let mid_stream: Arc<dyn FragmentExecutor> = Arc::new(StreamingExecutor {
            rows: 1,
            fail_with: Some("scan exploded".to_string()),
            ..Default::default()
        });
        for (query, sender) in [(40, mid_stream), (41, Arc::new(FailingRuns) as _)] {
            let [(a, a_nixl), (b, b_nixl)] = two_cns(sender, Arc::new(StubExecutor));
            for instance in [10, 12] {
                let mut result =
                    query_fragment(query, instance, exchange_plan_node(2, 0), result_sink());
                expect_senders(&mut result, 2, 1);
                exec_ok(&b, &result);
            }
            let mut sender = query_fragment(query, 11, scan_node(0, 0), stream_sink(2));
            send_to(&mut sender, 10, 18060);
            let mut second = sender
                .params
                .as_ref()
                .unwrap()
                .destinations
                .clone()
                .unwrap();
            second[0].fragment_instance_id = TUniqueId::new(query, 12);
            sender
                .params
                .as_mut()
                .unwrap()
                .destinations
                .as_mut()
                .unwrap()
                .extend(second);
            assert_eq!(
                sender_end(&a, &mut sender).1,
                "scan exploded",
                "query {query}"
            );

            let mut failed = a_nixl.failed_slots();
            failed.sort_unstable_by_key(|slot| slot.fragment_instance_id.to_proto().lo);
            assert_eq!(
                failed,
                [exchange_slot(query, 10, 2), exchange_slot(query, 12, 2)]
            );
            let started = std::time::Instant::now();
            for instance in [10, 12] {
                assert_eq!(fetch_error(&b, query, instance), "scan exploded");
            }
            assert!(started.elapsed() < Duration::from_secs(5));
            eventually("b to free the failed query", || idle(&b));
            assert!(idle(&a));
            assert!(b_nixl.failed.lock().unwrap().is_empty(), "no cascade back");
        }
    }

    #[test]
    fn a_purge_ends_the_hops_of_receivers_and_scans_that_never_run() {
        // CN a holds, for query 42, a receiver whose sender never comes and a scan deferred for a
        // runtime filter, each sending to CN b. Cancelling the query on a ends both hops with a
        // failure frame, so b's receivers fail at once instead of waiting for the FE.
        let [(a, a_nixl), (b, _)] = two_cns(
            Arc::new(FilterRecorder::new(sparse_keys())),
            Arc::new(StubExecutor),
        );
        for (instance, node_id) in [(5, 1), (10, 2)] {
            let mut result =
                query_fragment(42, instance, exchange_plan_node(node_id, 0), result_sink());
            expect_senders(&mut result, node_id, 1);
            exec_ok(&b, &result);
        }
        exec_ok(&a, &filter_builder(42));
        let mut scan = probing_scan(42);
        send_to(&mut scan, 5, 18060);
        exec_ok(&a, &scan);
        let mut receiver = query_fragment(42, 11, exchange_plan_node(3, 0), stream_sink(2));
        expect_senders(&mut receiver, 3, 1);
        send_to(&mut receiver, 10, 18060);
        exec_ok(&a, &receiver);
        assert_eq!(a.filters.deferred(), 1);

        assert_eq!(cancel(&a, 42).status_code, TStatusCode::OK.0);
        eventually("a to purge the query", || idle(&a));
        let mut failed = a_nixl.failed.lock().unwrap().clone();
        failed.sort_unstable_by_key(|(_, slot, _)| slot.node_id);
        let expected = |instance, node_id| {
            (
                "127.0.0.1:18060".parse().unwrap(),
                exchange_slot(42, instance, node_id),
                "query cancelled: injected".to_string(),
            )
        };
        assert_eq!(failed, [expected(5, 1), expected(10, 2)]);
        for instance in [5, 10] {
            assert_eq!(fetch_error(&b, 42, instance), "query cancelled: injected");
        }
        eventually("b to free the failed query", || idle(&b));
    }

    #[test]
    fn a_failure_frame_for_a_query_that_already_failed_is_ignored() {
        // Each CN holds a receiver of query 43 that sends to the other. A cancel on a fails b's
        // receiver, whose own failure frame back to a must end there, not bounce forever.
        let [(a, a_nixl), (b, b_nixl)] = two_cns(Arc::new(StubExecutor), Arc::new(StubExecutor));
        let mut on_a = query_fragment(43, 11, exchange_plan_node(3, 0), stream_sink(2));
        expect_senders(&mut on_a, 3, 1);
        send_to(&mut on_a, 10, 18060);
        exec_ok(&a, &on_a);
        let mut on_b = query_fragment(43, 10, exchange_plan_node(2, 0), stream_sink(4));
        expect_senders(&mut on_b, 2, 1);
        send_to(&mut on_b, 12, 8060);
        exec_ok(&b, &on_b);

        assert_eq!(cancel(&a, 43).status_code, TStatusCode::OK.0);
        eventually("both CNs to free the query", || idle(&a) && idle(&b));
        eventually("b's failure frame", || {
            b_nixl.failed.lock().unwrap().len() == 1
        });
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(a_nixl.failed_slots(), [exchange_slot(43, 10, 2)]);
        assert_eq!(b_nixl.failed_slots(), [exchange_slot(43, 12, 4)]);
        assert_eq!(
            a.exchanges.failure(FragmentInstanceId::from_halves(43, 0)),
            Some("query cancelled: injected".to_string()),
            "a keeps its own cause"
        );
    }

    #[test]
    fn a_failure_frame_before_its_receiver_registers_fails_the_registration() {
        let (b, _) = nixl_service(Arc::new(StubExecutor));
        let failed = NixlEnvelope::Failed {
            error: "scan exploded".to_string(),
        };
        let params = nixl_chunk::failed_params(exchange_slot(44, 10, 2));
        assert_eq!(
            transmit(&b, params, failed).0.status_code,
            TStatusCode::OK.0
        );

        let mut result = query_fragment(44, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        let status = exec(&b, &result);
        assert_eq!(status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(status.error_msgs[0].contains("scan exploded"), "{status:?}");
        assert!(idle(&b));
    }

    #[test]
    fn a_failure_frame_never_completes_its_sender() {
        // Sender 0 of two delivers a batch and its EOS; sender 1 fails. The receiver must not run
        // on sender 0's rows alone: it fails, and the batch it held is released.
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let (b, b_nixl) = nixl_service(executor.clone());
        let mut result = query_fragment(45, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 2);
        exec_ok(&b, &result);
        for (seq, token) in [(0, 5), (1, 0)] {
            let (params, envelope) = packed(45, seq, token);
            assert_eq!(
                transmit(&b, params, envelope).0.status_code,
                TStatusCode::OK.0
            );
        }
        let failed = NixlEnvelope::Failed {
            error: "scan exploded".to_string(),
        };
        let params = nixl_chunk::failed_params(SenderSlot {
            sender_id: 1,
            ..exchange_slot(45, 10, 2)
        });
        assert_eq!(
            transmit(&b, params, failed).0.status_code,
            TStatusCode::OK.0
        );
        assert_eq!(fetch_error(&b, 45, 10), "scan exploded");
        eventually("b to free the failed query", || idle(&b));
        assert!(
            executor.runs.lock().unwrap().is_empty(),
            "the receiver never ran"
        );
        assert_eq!(*b_nixl.released.lock().unwrap(), [5]);
    }

    /// Makes `params` report to `frontend` as backend `backend_num`, as the FE dispatches it.
    fn reports_to(
        params: &mut TExecPlanFragmentParams,
        frontend: &crate::fe_report::fake_frontend::FakeFrontend,
        backend_num: i32,
    ) {
        params.coord = Some(frontend.address.clone());
        params.backend_num = Some(backend_num);
    }

    /// Each report's backend number, status and message, by backend number. Every one is final.
    fn report_ends(
        reports: &[starrocks_thrift::frontend_service::TReportExecStatusParams],
    ) -> Vec<(i32, TStatusCode, String)> {
        let mut ends: Vec<_> = reports
            .iter()
            .map(|report| {
                assert_eq!(report.done, Some(true), "{report:?}");
                let status = report.status.clone().unwrap();
                (
                    report.backend_num.unwrap(),
                    status.status_code,
                    status.error_msgs.unwrap_or_default().join("; "),
                )
            })
            .collect();
        ends.sort_unstable_by_key(|(backend_num, ..)| *backend_num);
        ends
    }

    /// Dispatches sender `params`, which replies OK once accepted, and returns how its run ended
    /// as its report to the FE says: the status and the error.
    fn sender_end(
        service: &SiriusComputeNodeService,
        params: &mut TExecPlanFragmentParams,
    ) -> (TStatusCode, String) {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        reports_to(params, &frontend, 99);
        exec_ok(service, params);
        let ends = report_ends(&frontend.wait_for(1));
        assert_eq!(ends.len(), 1, "{ends:?}");
        let (_, code, error) = ends.into_iter().next().unwrap();
        (code, error)
    }

    #[test]
    fn every_instance_reports_ok_once_it_finished() {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::new();
        let mut result = query_fragment(50, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&service, &result);
        let mut sender = query_fragment(50, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        reports_to(&mut sender, &frontend, 2);
        exec_ok(&service, &sender);
        for _ in 0..2 {
            route(
                &service,
                methods::FETCH_DATA,
                fetch_request(50, 10),
                Vec::new(),
            );
        }
        // The FE's cancel after a successful query finds nothing left to report.
        assert_eq!(cancel(&service, 50).status_code, TStatusCode::OK.0);
        assert_eq!(
            report_ends(&frontend.wait_for(2)),
            [
                (1, TStatusCode::OK, String::new()),
                (2, TStatusCode::OK, String::new())
            ]
        );
        assert_eq!(service.reports.owed(), 0);
    }

    #[test]
    fn a_failed_instance_reports_its_error_once_unless_its_dispatch_reply_did() {
        // A sender run inline on CN a fails mid-stream. Its own dispatch reply carries the error,
        // so it sends no report; b's result receiver, which replied OK long ago, fails on the
        // failure frame and reports the sender's error. The FE's cancel afterwards adds nothing.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let failing = Arc::new(StreamingExecutor {
            rows: 1,
            fail_with: Some("scan exploded".to_string()),
            ..Default::default()
        });
        let [(mut a, _), (b, _)] = two_cns(failing, Arc::new(StubExecutor));
        a.inline_senders = true;
        let mut result = query_fragment(51, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&b, &result);
        let mut sender = query_fragment(51, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        reports_to(&mut sender, &frontend, 2);
        assert_eq!(exec(&a, &sender).error_msgs, ["scan exploded"]);
        eventually("b to free the failed query", || idle(&b));
        for service in [&a, &b] {
            assert_eq!(cancel(service, 51).status_code, TStatusCode::OK.0);
        }
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(1, TStatusCode::INTERNAL_ERROR, "scan exploded".to_string())]
        );
    }

    #[test]
    fn a_failed_sender_reports_its_error_once() {
        // The sender runs after its dispatch replied OK, so its report carries its error, as b's
        // result receiver's does once the failure frame reaches it.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let failing = Arc::new(StreamingExecutor {
            rows: 1,
            fail_with: Some("scan exploded".to_string()),
            ..Default::default()
        });
        let [(a, _), (b, _)] = two_cns(failing, Arc::new(StubExecutor));
        let mut result = query_fragment(54, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&b, &result);
        let mut sender = query_fragment(54, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        reports_to(&mut sender, &frontend, 2);
        exec_ok(&a, &sender);
        eventually("both CNs to free the failed query", || idle(&a) && idle(&b));
        for service in [&a, &b] {
            assert_eq!(cancel(service, 54).status_code, TStatusCode::OK.0);
        }
        let error = "scan exploded".to_string();
        assert_eq!(
            report_ends(&frontend.wait_for(2)),
            [
                (1, TStatusCode::INTERNAL_ERROR, error.clone()),
                (2, TStatusCode::INTERNAL_ERROR, error)
            ]
        );
    }

    /// Runs fragments like the stub, a sender only once the test lets it.
    #[derive(Debug)]
    struct GatedSenders(Mutex<std::sync::mpsc::Receiver<()>>);

    impl FragmentExecutor for GatedSenders {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if !run.outputs.is_empty() {
                self.0
                    .lock()
                    .unwrap()
                    .recv()
                    .map_err(|err| err.to_string())?;
            }
            StubExecutor.run_fragment(run)
        }
    }

    #[test]
    fn a_sender_that_runs_longer_than_its_dispatch_call_still_reports_ok() {
        // The FE waits only so long for a dispatch reply: the sender's comes once it is accepted,
        // and its report once it ran, as does that of the receiver its output completes.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let (release, gate) = channel();
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(GatedSenders(Mutex::new(gate))),
            &ComputeNodeConfig::default(),
            None,
        );
        let mut result = query_fragment(55, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&service, &result);
        let mut sender = query_fragment(55, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        reports_to(&mut sender, &frontend, 2);
        assert_eq!(dispatch(&service, &sender).status_code, TStatusCode::OK.0);
        assert_eq!(service.reports.owed(), 2, "the sender still runs");
        release.send(()).unwrap();
        assert_eq!(
            report_ends(&frontend.wait_for(2)),
            [
                (1, TStatusCode::OK, String::new()),
                (2, TStatusCode::OK, String::new())
            ]
        );
        eventually("the sender to finish", || {
            service.senders_running.load(Ordering::SeqCst) == 0
        });
    }

    #[test]
    fn a_receiver_that_fails_off_the_dispatch_path_reports_its_error() {
        // The receiver runs on the exchange-receiver thread once a remote sender's EOS arrives;
        // no RPC reply carries its failure, so only its report tells the FE.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let (service, _) = nixl_service(Arc::new(FailingReceivers));
        let mut result = query_fragment(52, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 3);
        exec_ok(&service, &result);
        for (seq, token) in [(0, 5), (1, 0)] {
            let (params, envelope) = packed(52, seq, token);
            assert_eq!(
                transmit(&service, params, envelope).0.status_code,
                TStatusCode::OK.0
            );
        }
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(
                3,
                TStatusCode::INTERNAL_ERROR,
                "receiver exploded".to_string()
            )]
        );
        assert_eq!(service.reports.owed(), 0);
    }

    /// Runs leaf fragments like the stub, and panics in every fragment fed by an exchange.
    #[derive(Debug)]
    struct PanickingReceivers;

    impl FragmentExecutor for PanickingReceivers {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if run.inputs.is_empty() && run.remote_inputs.is_empty() {
                return StubExecutor.run_fragment(run);
            }
            panic!("receiver blew up");
        }
    }

    #[test]
    fn a_panicking_fragment_still_ends_every_instance_of_its_query() {
        // Off the dispatch path: a receiver run by a remote sender's EOS panics on the
        // exchange-receiver thread. Its instance reports the panic, and the query is purged.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let (service, _) = nixl_service(Arc::new(PanickingReceivers));
        let mut result = query_fragment(54, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&service, &result);
        for (seq, token) in [(0, 5), (1, 0)] {
            let (params, envelope) = packed(54, seq, token);
            transmit(&service, params, envelope);
        }
        let ends = report_ends(&frontend.wait_for(1));
        assert_eq!(ends.len(), 1);
        assert_eq!((ends[0].0, ends[0].1), (1, TStatusCode::INTERNAL_ERROR));
        assert!(ends[0].2.contains("receiver blew up"), "{ends:?}");
        eventually("the panicked query's purge", || idle(&service));

        // On the dispatch path: the local sender, run inline, runs the receiver, which panics.
        // The reply carries the panic, so the sender sends no report; the receiver reports it.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let mut service = SiriusComputeNodeService::with_executor(
            Arc::new(PanickingReceivers),
            &ComputeNodeConfig::default(),
            None,
        );
        service.inline_senders = true;
        let mut result = query_fragment(55, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&service, &result);
        let mut sender = query_fragment(55, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        reports_to(&mut sender, &frontend, 2);
        let status = exec(&service, &sender);
        assert!(
            status.error_msgs[0].contains("receiver blew up"),
            "{status:?}"
        );
        // The sender finished before the receiver panicked.
        let ends = report_ends(&frontend.wait_for(2));
        assert_eq!(
            ends.iter()
                .map(|(backend, code, _)| (*backend, *code))
                .collect::<Vec<_>>(),
            [(1, TStatusCode::INTERNAL_ERROR), (2, TStatusCode::OK)]
        );
        assert!(idle(&service));
    }

    #[test]
    fn translate_only_mode_ends_each_instance_it_accepts() {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let mut service = SiriusComputeNodeService::new();
        service.translate_only = true;
        let mut scan = query_fragment(56, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut scan, 10, 8060);
        reports_to(&mut scan, &frontend, 4);
        exec_ok(&service, &scan);
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(4, TStatusCode::OK, String::new())]
        );
        assert_eq!(service.reports.owed(), 0);
    }

    #[test]
    fn a_cancel_reports_every_instance_still_owed_as_cancelled() {
        // Query 53 has a receiver waiting for its senders and a scan deferred for a runtime filter
        // when the FE cancels it: each reports CANCELLED at once, and nothing follows.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FilterRecorder::new(sparse_keys())),
            &ComputeNodeConfig::default(),
            None,
        );
        let mut builder = filter_builder(53);
        reports_to(&mut builder, &frontend, 1);
        exec_ok(&service, &builder);
        let mut scan = probing_scan(53);
        reports_to(&mut scan, &frontend, 2);
        exec_ok(&service, &scan);
        assert_eq!(service.filters.deferred(), 1);

        assert_eq!(cancel(&service, 53).status_code, TStatusCode::OK.0);
        eventually("the cancel's purge", || idle(&service));
        assert_eq!(cancel(&service, 53).status_code, TStatusCode::OK.0);
        assert_eq!(
            report_ends(&frontend.wait_for(2)),
            [
                (
                    1,
                    TStatusCode::CANCELLED,
                    "query cancelled: injected".to_string()
                ),
                (
                    2,
                    TStatusCode::CANCELLED,
                    "query cancelled: injected".to_string()
                )
            ]
        );
    }

    #[test]
    fn a_cancel_before_dispatch_keeps_late_fragments_off_the_gpu() {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let executor = Arc::new(FilterRecorder::new(sparse_keys()));
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        assert_eq!(
            cancel_with(&service, 60, Some(Reason::UserCancel), None).status_code,
            TStatusCode::OK.0
        );
        let mut leaf = query_fragment(60, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut leaf, 10, 8060);
        reports_to(&mut leaf, &frontend, 2);
        let mut receiver = query_fragment(60, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut receiver, 2, 1);
        reports_to(&mut receiver, &frontend, 1);
        for late in [&leaf, &receiver] {
            let status = exec(&service, late);
            assert_eq!(status.status_code, TStatusCode::CANCELLED.0);
            assert!(
                status.error_msgs[0].starts_with("query cancelled: the user cancelled the query"),
                "{status:?}"
            );
        }
        assert!(executor.runs.lock().unwrap().is_empty(), "nothing ran");
        assert!(
            frontend.wait_for(0).is_empty(),
            "the replies carried the refusals"
        );
        assert!(idle(&service));
    }

    #[test]
    fn a_cancel_after_the_query_finished_changes_nothing() {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::new();
        let mut result = query_fragment(61, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, &frontend, 1);
        exec_ok(&service, &result);
        let mut sender = query_fragment(61, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        reports_to(&mut sender, &frontend, 2);
        exec_ok(&service, &sender);
        for _ in 0..2 {
            route(
                &service,
                methods::FETCH_DATA,
                fetch_request(61, 10),
                Vec::new(),
            );
        }
        // One cancel per instance the FE still thinks runs; the second is only counted.
        for _ in 0..2 {
            let status = cancel_with(&service, 61, Some(Reason::QueryFinished), None);
            assert_eq!(status.status_code, TStatusCode::OK.0);
        }
        assert_eq!(
            report_ends(&frontend.wait_for(2)),
            [
                (1, TStatusCode::OK, String::new()),
                (2, TStatusCode::OK, String::new())
            ]
        );
        assert_eq!(
            service.ends.cancels(FragmentInstanceId::from_halves(61, 0)),
            2,
            "both cancels counted"
        );
        eventually("the release", || {
            service
                .exchanges
                .failure(FragmentInstanceId::from_halves(61, 0))
                == Some("the query ended normally (QUERY_FINISHED)".to_string())
        });
        assert!(idle(&service));
    }

    /// A failure frame a fake NIXL side sent: where, for which receiver, and the error.
    type FailureFrame = (SocketAddr, SenderSlot, String);

    /// CN a holds a receiver of `query` that waits for a sender and sends to CN b, as when a LIMIT
    /// ends a query early; the FE cancels it with `reason` and `message`. Returns the reports and
    /// the failure frames a sent.
    fn cancel_a_waiting_receiver(
        query: i64,
        reason: crate::proto::starrocks::PPlanFragmentCancelReason,
        message: Option<&str>,
    ) -> (Vec<(i32, TStatusCode, String)>, Vec<FailureFrame>) {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let (a, a_nixl) = nixl_service(Arc::new(StubExecutor));
        let mut receiver = query_fragment(query, 11, exchange_plan_node(3, 0), stream_sink(2));
        expect_senders(&mut receiver, 3, 1);
        send_to(&mut receiver, 10, 18060);
        reports_to(&mut receiver, &frontend, 1);
        exec_ok(&a, &receiver);
        assert_eq!(
            cancel_with(&a, query, Some(reason), message).status_code,
            TStatusCode::OK.0
        );
        eventually("a to release the query", || idle(&a));
        let reports = report_ends(&frontend.wait_for(1));
        let failed = a_nixl.failed.lock().unwrap().clone();
        (reports, failed)
    }

    #[test]
    fn a_normal_end_releases_what_is_left_without_failing_the_query() {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        for (query, reason, said) in [
            (
                62,
                Reason::QueryFinished,
                "the query ended normally (QUERY_FINISHED)",
            ),
            (
                63,
                Reason::LimitReach,
                "the query ended normally (LIMIT_REACH)",
            ),
        ] {
            let (reports, failed) = cancel_a_waiting_receiver(query, reason, None);
            assert_eq!(reports, [(1, TStatusCode::CANCELLED, said.to_string())]);
            assert!(
                failed.is_empty(),
                "no failure frame for a query that succeeded"
            );
        }
    }

    #[test]
    fn a_user_kill_or_an_internal_error_fails_the_query() {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        for (query, reason, message, said) in [
            (
                64,
                Reason::UserCancel,
                None,
                "query cancelled: the user cancelled the query",
            ),
            (
                65,
                Reason::Timeout,
                None,
                "query cancelled: the query timed out",
            ),
            (
                66,
                Reason::InternalError,
                Some("scan exploded on cn2"),
                "query cancelled: scan exploded on cn2",
            ),
        ] {
            let (reports, failed) = cancel_a_waiting_receiver(query, reason, message);
            assert_eq!(reports, [(1, TStatusCode::CANCELLED, said.to_string())]);
            assert_eq!(
                failed
                    .iter()
                    .map(|(_, slot, error)| (*slot, error.as_str()))
                    .collect::<Vec<_>>(),
                [(exchange_slot(query, 10, 2), said)],
                "the waiting receiver's destination is told"
            );
        }
    }

    /// Dispatches `params` alone in a batch, as the FE's batch dispatch does.
    fn exec_batch(
        service: &SiriusComputeNodeService,
        params: &TExecPlanFragmentParams,
    ) -> StatusPb {
        let batch = TExecBatchPlanFragmentsParams::new(
            Some(fragment_params(None, Some(desc_table()))),
            Some(vec![params.clone()]),
        );
        let response = route(
            service,
            methods::EXEC_BATCH_PLAN_FRAGMENTS,
            PExecBatchPlanFragmentsRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&batch),
        );
        PExecBatchPlanFragmentsResult::decode(response.body.as_slice())
            .unwrap()
            .status
            .unwrap()
    }

    #[test]
    fn a_fragment_refused_after_the_fe_ended_its_query_replies_cancelled() {
        // After a normal end the FE ignores CANCELLED but would retry a query that already
        // returned its rows on any other error, so a late dispatch replies CANCELLED, single or
        // batched. A query that failed here, with no word from the FE yet, still replies with the
        // error.
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        let service = SiriusComputeNodeService::new();
        for (query, reason) in [(67, Reason::QueryFinished), (68, Reason::LimitReach)] {
            assert_eq!(
                cancel_with(&service, query, Some(reason), None).status_code,
                TStatusCode::OK.0
            );
            let mut late = query_fragment(query, 11, scan_node(0, 0), stream_sink(2));
            send_to(&mut late, 10, 8060);
            let status = exec(&service, &late);
            assert_eq!(status.status_code, TStatusCode::CANCELLED.0, "{status:?}");
            assert!(
                status.error_msgs[0].starts_with(ENDED_NORMALLY),
                "{status:?}"
            );
            let status = exec_batch(&service, &late);
            assert_eq!(status.status_code, TStatusCode::CANCELLED.0, "{status:?}");
        }
        assert!(
            service
                .exchanges
                .mark_failed(FragmentInstanceId::from_halves(69, 0), "scan exploded")
        );
        let mut late = query_fragment(69, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut late, 10, 8060);
        assert_eq!(
            exec(&service, &late).status_code,
            TStatusCode::INTERNAL_ERROR.0
        );
    }

    #[test]
    fn a_cancel_refuses_the_query_before_its_purge_runs() {
        // The purge waits for a blocking thread; the only one is busy. The cancel's reply must
        // already refuse a fragment of the query arriving in that gap.
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let blocker = runtime.spawn_blocking(move || {
            let _ = gate.recv();
        });
        let service = SiriusComputeNodeService::new();
        let request = PCancelPlanFragmentRequest {
            finst_id: PUniqueId { hi: 70, lo: 1 },
            cancel_reason: Some(Reason::UserCancel as i32),
            is_pipeline: Some(true),
            query_id: Some(PUniqueId { hi: 70, lo: 0 }),
            error_message: None,
        };
        runtime
            .block_on(service.cancel_plan_fragment(request, Vec::new()))
            .unwrap();
        let mut late = query_fragment(70, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut late, 10, 8060);
        let refused = service.process_fragment(&late).unwrap_err();
        assert!(
            refused.starts_with("query cancelled: the user cancelled the query"),
            "{refused}"
        );
        release.send(()).unwrap();
        runtime.block_on(blocker).unwrap();
    }

    /// Fails every run with the given error, before the fragment produces anything.
    #[derive(Debug)]
    struct FailingWith(String);

    impl FragmentExecutor for FailingWith {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, _run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            Err(self.0.clone())
        }
    }

    #[test]
    fn a_normal_end_never_reaches_another_cn_as_a_failure() {
        // On the sender's CN: a fragment that stops because its query ended normally sends no
        // failure frame.
        let ended = format!("{ENDED_NORMALLY} (QUERY_FINISHED)");
        let (a, a_nixl) = nixl_service(Arc::new(FailingWith(ended.clone())));
        let mut sender = query_fragment(71, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 18060);
        assert_ne!(sender_end(&a, &mut sender).0, TStatusCode::OK);
        assert!(a_nixl.failed.lock().unwrap().is_empty());

        // On a receiving CN, should such a frame come anyway: the waiting receiver ends quietly,
        // reporting CANCELLED, and passes nothing on.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let (b, b_nixl) = nixl_service(Arc::new(StubExecutor));
        let mut receiver = query_fragment(72, 10, exchange_plan_node(2, 0), stream_sink(3));
        expect_senders(&mut receiver, 2, 1);
        send_to(&mut receiver, 12, 18060);
        reports_to(&mut receiver, &frontend, 1);
        exec_ok(&b, &receiver);
        let failed = NixlEnvelope::Failed {
            error: ended.clone(),
        };
        let params = nixl_chunk::failed_params(exchange_slot(72, 10, 2));
        assert_eq!(
            transmit(&b, params, failed).0.status_code,
            TStatusCode::OK.0
        );
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(1, TStatusCode::CANCELLED, ended)]
        );
        eventually("b to release the query", || idle(&b));
        assert!(b_nixl.failed.lock().unwrap().is_empty());
    }

    #[test]
    fn a_query_that_ended_frees_its_descriptor_table() {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        let service = SiriusComputeNodeService::new();
        let cached = |query| {
            service
                .descriptor_tables
                .lock()
                .unwrap()
                .get(FragmentInstanceId::from_halves(query, 0).query_hi())
                .is_some()
        };
        // Once its result sink delivered the last row.
        let mut result = query_fragment(73, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        exec_ok(&service, &result);
        let mut sender = query_fragment(73, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        exec_ok(&service, &sender);
        assert!(cached(73));
        for _ in 0..2 {
            route(
                &service,
                methods::FETCH_DATA,
                fetch_request(73, 10),
                Vec::new(),
            );
        }
        assert!(!cached(73));

        // Once the FE cancels it.
        let mut waiting = query_fragment(74, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut waiting, 2, 1);
        exec_ok(&service, &waiting);
        assert!(cached(74));
        cancel_with(&service, 74, Some(Reason::UserCancel), None);
        eventually("the purge", || !cached(74));
    }

    /// Stands in for the engine thread: parks like the stub, records what it is asked to
    /// interrupt, and runs queued drops only once `busy` is cleared, as a busy engine thread would.
    #[derive(Debug, Default)]
    struct BusyEngine {
        interrupted: Mutex<Vec<FragmentInstanceId>>,
        /// The service, to read what it recorded about a query when asked to interrupt it.
        service: Mutex<Option<SiriusComputeNodeService>>,
        /// The query's recorded cause at each interrupt.
        cause_when_interrupted: Mutex<Vec<Option<String>>>,
        busy: std::sync::Arc<(Mutex<bool>, std::sync::Condvar)>,
        frees: Arc<PendingFrees>,
        dropped: Arc<Mutex<Vec<SenderSlot>>>,
    }

    impl BusyEngine {
        fn set_busy(&self, busy: bool) {
            *self.busy.0.lock().unwrap() = busy;
            self.busy.1.notify_all();
        }
    }

    impl FragmentExecutor for BusyEngine {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn drop_parked(&self, _slot: SenderSlot) -> Result<(), String> {
            panic!("a purge must not wait for the engine thread");
        }

        fn drop_parked_later(&self, slot: SenderSlot) {
            let (pending, busy, dropped) = (
                self.frees.begin(),
                Arc::clone(&self.busy),
                Arc::clone(&self.dropped),
            );
            std::thread::spawn(move || {
                let mut engine_busy = busy.0.lock().unwrap();
                while *engine_busy {
                    engine_busy = busy.1.wait(engine_busy).unwrap();
                }
                dropped.lock().unwrap().push(slot);
                drop(pending);
            });
        }

        fn interrupt(&self, query: FragmentInstanceId) {
            self.interrupted.lock().unwrap().push(query);
            let service = self.service.lock().unwrap().clone();
            if let Some(service) = service {
                self.cause_when_interrupted
                    .lock()
                    .unwrap()
                    .push(service.exchanges.failure(query));
            }
        }

        fn pending_frees(&self) -> Option<Arc<PendingFrees>> {
            Some(Arc::clone(&self.frees))
        }
    }

    /// Query `query` with a sender's output parked for a receiver that waits for a second sender.
    fn parked_for_a_waiting_receiver(service: &SiriusComputeNodeService, query: i64) {
        let mut result = query_fragment(query, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 2);
        exec_ok(service, &result);
        let mut sender = query_fragment(query, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        exec_ok(service, &sender);
        assert_eq!(service.exchanges.counts().parked_senders, 1);
    }

    #[test]
    fn a_failure_purge_interrupts_its_query_and_a_normal_end_does_not() {
        let engine = Arc::new(BusyEngine::default());
        let service = SiriusComputeNodeService::with_executor(
            engine.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        *engine.service.lock().unwrap() = Some(service.clone());
        service.fail_and_purge(FragmentInstanceId::from_halves(80, 3), "scan exploded");
        service.release_query(
            FragmentInstanceId::from_halves(81, 0),
            "the query ended normally (QUERY_FINISHED)",
        );
        assert_eq!(
            *engine.interrupted.lock().unwrap(),
            [FragmentInstanceId::from_halves(80, 3)]
        );
        // Recorded first, so the interrupted fragment reports the cause rather than its
        // interruption.
        assert_eq!(
            *engine.cause_when_interrupted.lock().unwrap(),
            [Some("scan exploded".to_string())]
        );
    }

    #[test]
    fn a_purge_returns_while_the_engine_thread_is_busy() {
        let engine = Arc::new(BusyEngine::default());
        let service = SiriusComputeNodeService::with_executor(
            engine.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        parked_for_a_waiting_receiver(&service, 82);
        engine.set_busy(true);
        let started = std::time::Instant::now();
        service.fail_and_purge(FragmentInstanceId::from_halves(82, 0), "scan exploded");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(
            engine.frees.pending(),
            1,
            "the drop waits for the engine thread"
        );
        assert!(engine.dropped.lock().unwrap().is_empty());
        engine.set_busy(false);
        assert!(engine.frees.wait_for_none(Duration::from_secs(10)));
        assert_eq!(engine.dropped.lock().unwrap().len(), 1);
        assert!(idle(&service));
    }

    /// Fails every run after its query failed on this CN meanwhile, as an interrupted run does.
    #[derive(Debug, Default)]
    struct InterruptedRuns {
        service: Mutex<Option<SiriusComputeNodeService>>,
    }

    impl FragmentExecutor for InterruptedRuns {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            let service = self.service.lock().unwrap().clone().unwrap();
            service
                .exchanges
                .mark_failed(run.query.unwrap(), "scan exploded on cn2");
            Err("Interrupted!".to_string())
        }
    }

    #[test]
    fn a_run_stopped_by_its_querys_failure_reports_the_cause_first() {
        let executor = Arc::new(InterruptedRuns::default());
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        *executor.service.lock().unwrap() = Some(service.clone());
        let mut leaf = query_fragment(84, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut leaf, 10, 8060);
        assert_eq!(
            sender_end(&service, &mut leaf).1,
            "scan exploded on cn2 (Interrupted!)"
        );
    }

    /// Gives `params`' query a timeout of `seconds`, as the FE's query options carry it.
    fn with_timeout(mut params: TExecPlanFragmentParams, seconds: i32) -> TExecPlanFragmentParams {
        params.query_options = Some(starrocks_thrift::internal_service::TQueryOptions {
            query_timeout: Some(seconds),
            ..Default::default()
        });
        params
    }

    /// A result receiver of `query`, reporting as backend 1, waiting for a sender that never
    /// comes, with a 1 s query timeout.
    fn waiting_for_a_lost_sender(
        service: &SiriusComputeNodeService,
        frontend: &crate::fe_report::fake_frontend::FakeFrontend,
        query: i64,
    ) {
        let mut result = query_fragment(query, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        reports_to(&mut result, frontend, 1);
        exec_ok(service, &with_timeout(result, 1));
    }

    #[test]
    fn a_receiver_whose_sender_never_comes_fails_at_its_query_timeout() {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::new();
        waiting_for_a_lost_sender(&service, &frontend, 90);
        // fetch_data waits for the query's deadline, not RESULT_WAIT, and reads its cause.
        let started = std::time::Instant::now();
        assert_eq!(
            fetch_error(&service, 90, 10),
            "query timed out after 1 s on this CN"
        );
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(900) && waited < Duration::from_secs(4),
            "{waited:?}"
        );
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(
                1,
                TStatusCode::INTERNAL_ERROR,
                "query timed out after 1 s on this CN".to_string()
            )]
        );
        assert!(idle(&service));
    }

    #[test]
    fn the_fes_timeout_and_the_local_deadline_end_a_query_once() {
        use crate::proto::starrocks::PPlanFragmentCancelReason as Reason;
        // The FE first: the deadline then finds the query ended and does nothing.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::new();
        waiting_for_a_lost_sender(&service, &frontend, 92);
        cancel_with(&service, 92, Some(Reason::Timeout), None);
        std::thread::sleep(Duration::from_millis(1500));
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(
                1,
                TStatusCode::CANCELLED,
                "query cancelled: the query timed out".to_string()
            )]
        );
        assert_eq!(
            service
                .exchanges
                .failure(FragmentInstanceId::from_halves(92, 0)),
            Some("query cancelled: the query timed out".to_string())
        );

        // The deadline first: the FE's cancel then adds nothing.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        waiting_for_a_lost_sender(&service, &frontend, 93);
        eventually("the deadline", || {
            service
                .exchanges
                .failure(FragmentInstanceId::from_halves(93, 0))
                .is_some()
        });
        cancel_with(&service, 93, Some(Reason::Timeout), None);
        assert_eq!(
            report_ends(&frontend.wait_for(1)),
            [(
                1,
                TStatusCode::INTERNAL_ERROR,
                "query timed out after 1 s on this CN".to_string()
            )]
        );
        assert!(idle(&service));
    }

    #[test]
    fn a_deadline_leaves_a_query_that_finished_alone() {
        let service = SiriusComputeNodeService::new();
        let mut result = query_fragment(94, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut result, 2, 1);
        exec_ok(&service, &with_timeout(result, 1));
        let mut sender = query_fragment(94, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        exec_ok(&service, &with_timeout(sender, 1));
        for _ in 0..2 {
            route(
                &service,
                methods::FETCH_DATA,
                fetch_request(94, 10),
                Vec::new(),
            );
        }
        assert_eq!(
            service
                .deadlines
                .remaining(FragmentInstanceId::from_halves(94, 0)),
            None,
            "its last row delivered, the query's deadline is gone"
        );
        std::thread::sleep(Duration::from_millis(1500));
        assert_eq!(
            service
                .exchanges
                .failure(FragmentInstanceId::from_halves(94, 0)),
            None,
            "not failed after the fact"
        );

        // A purged query's deadline goes with it.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        waiting_for_a_lost_sender(&service, &frontend, 95);
        assert!(
            service
                .deadlines
                .remaining(FragmentInstanceId::from_halves(95, 0))
                .is_some()
        );
        service.fail_and_purge(FragmentInstanceId::from_halves(95, 0), "scan exploded");
        assert_eq!(
            service
                .deadlines
                .remaining(FragmentInstanceId::from_halves(95, 0)),
            None
        );
    }

    /// Panics when asked to interrupt query 96, as a bug failing one query at its deadline would.
    #[derive(Debug)]
    struct PanicsInterrupting96;

    impl FragmentExecutor for PanicsInterrupting96 {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn interrupt(&self, query: FragmentInstanceId) {
            if query.query_hi() == FragmentInstanceId::from_halves(96, 0).query_hi() {
                panic!("interrupting query 96 failed");
            }
        }
    }

    #[test]
    fn a_panic_failing_one_query_at_its_deadline_spares_the_others() {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(PanicsInterrupting96),
            &ComputeNodeConfig::default(),
            None,
        );
        waiting_for_a_lost_sender(&service, &frontend, 96);
        let mut later = query_fragment(97, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut later, 2, 1);
        exec_ok(&service, &with_timeout(later, 2));
        assert_eq!(
            fetch_error(&service, 97, 10),
            "query timed out after 2 s on this CN"
        );
    }

    #[test]
    fn sink_mode_follows_destinations_and_partition_type() {
        let keys = vec![1];
        let mode = |partition, destinations| {
            SiriusComputeNodeService::sink_mode(partition, destinations, Some(&keys))
        };
        assert_eq!(
            mode(TPartitionType::HASH_PARTITIONED, 1),
            Ok((false, Vec::new())),
            "one destination is a gather"
        );
        assert_eq!(
            mode(TPartitionType::UNPARTITIONED, 2),
            Ok((true, Vec::new()))
        );
        assert_eq!(
            mode(TPartitionType::HASH_PARTITIONED, 2),
            Ok((false, vec![1]))
        );
        let err = mode(TPartitionType::RANDOM, 2).unwrap_err();
        assert!(err.contains("does not support"), "{err}");
        let err = SiriusComputeNodeService::sink_mode(TPartitionType::HASH_PARTITIONED, 2, None)
            .unwrap_err();
        assert!(err.contains("without partition key columns"), "{err}");
    }

    /// Fails receivers fed by instance `failing`'s exchange, and records every receiver it runs.
    #[derive(Debug)]
    struct FailingInstance {
        failing: FragmentInstanceId,
        ran: Mutex<Vec<FragmentInstanceId>>,
    }

    impl FragmentExecutor for FailingInstance {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            let Some(receiver) = run
                .inputs
                .iter()
                .flat_map(|(_, slots)| slots)
                .map(|slot| slot.fragment_instance_id)
                .next()
            else {
                return StubExecutor.run_fragment(run);
            };
            self.ran.lock().unwrap().push(receiver);
            if receiver == self.failing {
                return Err("receiver exploded".to_string());
            }
            StubExecutor.run_fragment(run)
        }
    }

    /// Makes `params` sender 0 of the given fragment instances of its query, all on this CN.
    fn send_to_all(params: &mut TExecPlanFragmentParams, instances: &[i64]) {
        let exec = params.params.as_mut().unwrap();
        exec.sender_id = Some(0);
        exec.destinations = Some(
            instances
                .iter()
                .map(|&instance| {
                    TPlanFragmentDestination::new(
                        TUniqueId::new(exec.query_id.hi, instance),
                        None,
                        Some(TNetworkAddress::new("127.0.0.1".to_string(), 8060)),
                        None,
                    )
                })
                .collect(),
        );
    }

    /// One sender completes two receivers and the first to run fails: the other still runs.
    #[test]
    fn drain_ready_runs_every_receiver_past_a_failed_one() {
        let executor = Arc::new(FailingInstance {
            failing: FragmentInstanceId::from_halves(19, 11),
            ran: Mutex::default(),
        });
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        for instance in [10, 11] {
            let mut receiver =
                query_fragment(19, instance, exchange_plan_node(2, 0), result_sink());
            expect_senders(&mut receiver, 2, 1);
            exec_ok(&service, &receiver);
        }
        let mut sender = query_fragment(19, 12, scan_node(0, 0), stream_sink(2));
        send_to_all(&mut sender, &[10, 11]);
        // The sender itself runs fine; one receiver it completes fails on its own.
        assert_eq!(sender_end(&service, &mut sender).0, TStatusCode::OK);
        let ran = executor.ran.lock().unwrap();
        assert_eq!(
            *ran,
            [
                FragmentInstanceId::from_halves(19, 11),
                FragmentInstanceId::from_halves(19, 10)
            ]
        );
    }

    /// A sender that completes one receiver and is then refused by another frees the completed
    /// receiver's inputs: it was handed over, so the query's purge cannot reach it.
    #[test]
    fn a_refused_push_frees_receivers_it_already_completed() {
        let (service, executor) = recording_service();
        let mut completed = query_fragment(18, 10, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut completed, 2, 1);
        exec_ok(&service, &completed);
        let mut waiting = query_fragment(18, 11, exchange_plan_node(2, 0), result_sink());
        expect_senders(&mut waiting, 2, 2);
        exec_ok(&service, &waiting);
        let mut first = query_fragment(18, 12, scan_node(0, 0), stream_sink(2));
        send_to(&mut first, 11, 8060);
        exec_ok(&service, &first);

        // Sender 0 again: it completes instance 10, then collides at instance 11.
        let mut second = query_fragment(18, 13, scan_node(0, 0), stream_sink(2));
        send_to_all(&mut second, &[10, 11]);
        let (_, error) = sender_end(&service, &mut second);
        assert!(error.contains("duplicate sender 0"), "{error}");
        let completed = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(18, 10),
            node_id: 2,
            sender_id: 0,
        };
        assert!(
            executor.dropped.lock().unwrap().contains(&completed),
            "{:?}",
            executor.dropped.lock().unwrap()
        );
        assert_eq!(service.exchanges.counts(), ExchangeCounts::default());
    }

    /// A data stream sink without `sender_id` is refused before it runs, and a fragment without
    /// any output sink is refused rather than accepted with nothing reading its output.
    #[test]
    fn a_sender_without_sender_id_or_sink_is_refused() {
        let service = SiriusComputeNodeService::new();
        let mut sender = query_fragment(20, 11, scan_node(0, 0), stream_sink(2));
        send_to(&mut sender, 10, 8060);
        sender.params.as_mut().unwrap().sender_id = None;
        let status = exec(&service, &sender);
        assert!(
            status.error_msgs[0].contains("missing sender_id"),
            "{status:?}"
        );

        // Another query: the refusal above failed query 20, which now refuses its later fragments.
        let mut sinkless = query_fragment(21, 12, scan_node(0, 0), stream_sink(2));
        sinkless.fragment.as_mut().unwrap().output_sink = None;
        let status = exec(&service, &sinkless);
        assert!(
            status.error_msgs[0].contains("no output sink"),
            "{status:?}"
        );
    }

    /// Fragment instance `instance` of query `query`: one plan node feeding `sink`.
    fn query_fragment(
        query: i64,
        instance: i64,
        node: TPlanNode,
        sink: TDataSink,
    ) -> TExecPlanFragmentParams {
        let mut params = fragment_params(Some(TPlan::new(vec![node])), Some(desc_table()));
        params.fragment.as_mut().unwrap().output_sink = Some(sink);
        params.params = Some(exec_params(
            TUniqueId::new(query, 0),
            TUniqueId::new(query, instance),
        ));
        params
    }

    fn expect_senders(params: &mut TExecPlanFragmentParams, node_id: i32, senders: i32) {
        params
            .params
            .as_mut()
            .unwrap()
            .per_exch_num_senders
            .insert(node_id, senders);
    }

    /// The FE sends sink destinations by IP, so a CN advertised by hostname must still see a
    /// destination at its own resolved IP as local.
    #[test]
    fn hostname_advertised_cn_matches_its_ip_destination() {
        assert_eq!(resolve_ip("127.0.0.1"), "127.0.0.1");
        let config = ComputeNodeConfig {
            advertise_host: crate::Host::new("localhost").unwrap(),
            ..ComputeNodeConfig::default()
        };
        let service =
            SiriusComputeNodeService::with_executor(Arc::new(StubExecutor), &config, None);
        let destination = TPlanFragmentDestination::new(
            TUniqueId::new(1, 2),
            None,
            Some(TNetworkAddress::new(
                "127.0.0.1".to_string(),
                i32::from(config.brpc_port),
            )),
            None,
        );
        assert_eq!(service.remote_peer(&destination), Ok(None));
    }

    /// Makes `params` sender 0 of fragment instance `instance` of its query, served on `port`.
    fn send_to(params: &mut TExecPlanFragmentParams, instance: i64, port: i32) {
        let exec = params.params.as_mut().unwrap();
        exec.sender_id = Some(0);
        exec.destinations = Some(vec![TPlanFragmentDestination::new(
            TUniqueId::new(exec.query_id.hi, instance),
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), port)),
            None,
        )]);
    }

    /// Dispatches `params`, then waits for a sender it started to finish running, so the caller
    /// sees what the run did.
    fn exec(service: &SiriusComputeNodeService, params: &TExecPlanFragmentParams) -> StatusPb {
        let status = dispatch(service, params);
        senders_done(service);
        status
    }

    /// The `exec_plan_fragment` reply to `params`, without waiting for a sender it started.
    fn dispatch(service: &SiriusComputeNodeService, params: &TExecPlanFragmentParams) -> StatusPb {
        let response = route(
            service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(params),
        );
        PExecPlanFragmentResult::decode(response.body.as_slice())
            .unwrap()
            .status
    }

    fn exec_ok(service: &SiriusComputeNodeService, params: &TExecPlanFragmentParams) {
        let status = exec(service, params);
        assert_eq!(
            status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            status.error_msgs
        );
    }

    fn stream_sink(dest_node_id: i32) -> TDataSink {
        TDataSink::new(
            TDataSinkType::DATA_STREAM_SINK,
            Some(TDataStreamSink::new(
                dest_node_id,
                TDataPartition::new(TPartitionType::UNPARTITIONED, None, None, None),
                None,
                None,
                None,
                None,
                None,
            )),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn exchange_plan_node(node_id: i32, tuple_id: i32) -> TPlanNode {
        let mut node = scan_node(node_id, tuple_id);
        node.node_type = TPlanNodeType::EXCHANGE_NODE;
        node.file_scan_node = None;
        node.exchange_node = Some(TExchangeNode::new(
            vec![tuple_id],
            None,
            None,
            Some(TPartitionType::UNPARTITIONED),
            Some(true),
            None,
        ));
        node
    }

    fn fetch_request(hi: i64, lo: i64) -> Vec<u8> {
        PFetchDataRequest {
            finst_id: PUniqueId { hi, lo },
        }
        .encode_to_vec()
    }

    fn result_sink_typed(kind: TResultSinkType) -> TDataSink {
        TDataSink::new(
            TDataSinkType::RESULT_SINK,
            None,
            Some(TResultSink::new(Some(kind), None, None, None, None)),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn route(
        service: &SiriusComputeNodeService,
        method: &str,
        body: Vec<u8>,
        attachment: Vec<u8>,
    ) -> prpc::Response {
        // Route through a router built from a clone of `service`; the result store is shared via
        // `Arc`, so buffered results survive across the per-call router clones.
        let mut router = PInternalServiceRouter::new(service.clone());
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                router
                    .ready()
                    .await
                    .unwrap()
                    .call(prpc::Request::new(SERVICE_NAME, method, body, attachment))
                    .await
            })
            .unwrap()
    }

    fn result_sink() -> TDataSink {
        // Only the sink type is read today (is_result_sink); the per-sink payloads stay None.
        TDataSink::new(
            TDataSinkType::RESULT_SINK,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn exec_params(
        query_id: TUniqueId,
        fragment_instance_id: TUniqueId,
    ) -> TPlanFragmentExecParams {
        // Only the ids are needed to key the result store; scan ranges/senders stay empty.
        TPlanFragmentExecParams::new(
            query_id,
            fragment_instance_id,
            BTreeMap::new(),
            BTreeMap::new(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn call_exec_plan_fragment(
        request: PExecPlanFragmentRequest,
        attachment: Vec<u8>,
    ) -> PExecPlanFragmentResult {
        // Route through the generated Tower service so tests cover protobuf decoding,
        // method lookup, service dispatch, and response encoding.
        let response = call_router(prpc::Request::new(
            SERVICE_NAME,
            methods::EXEC_PLAN_FRAGMENT,
            request.encode_to_vec(),
            attachment,
        ))
        .unwrap();
        PExecPlanFragmentResult::decode(response.body.as_slice()).unwrap()
    }

    fn call_exec_batch_plan_fragments(
        request: PExecBatchPlanFragmentsRequest,
        attachment: Vec<u8>,
    ) -> PExecBatchPlanFragmentsResult {
        // Route batch requests through the same generated service path used by BRPC.
        let response = call_router(prpc::Request::new(
            SERVICE_NAME,
            methods::EXEC_BATCH_PLAN_FRAGMENTS,
            request.encode_to_vec(),
            attachment,
        ))
        .unwrap();
        PExecBatchPlanFragmentsResult::decode(response.body.as_slice()).unwrap()
    }

    fn call_router(request: prpc::Request) -> std::result::Result<prpc::Response, prpc::Error> {
        // The generated router is a Tower service; a tiny current-thread runtime is
        // enough because these tests do not spawn the service futures.
        let mut router = PInternalServiceRouter::new(SiriusComputeNodeService::new());
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async { router.ready().await.unwrap().call(request).await })
    }

    fn supported_fragment() -> TExecPlanFragmentParams {
        // Minimal single-node result fragment with a descriptor table for direct translation.
        let mut params = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        params.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        params.params = Some(exec_params(TUniqueId::new(0, 1), TUniqueId::new(0, 2)));
        params
    }

    fn fragment_params(
        plan: Option<TPlan>,
        desc_tbl: Option<TDescriptorTable>,
    ) -> TExecPlanFragmentParams {
        // Only the fields required by the translator are populated in these fixtures.
        TExecPlanFragmentParams {
            protocol_version: InternalServiceVersion::V1,
            fragment: Some(TPlanFragment {
                plan,
                output_exprs: None,
                output_sink: None,
                partition: TDataPartition::new(TPartitionType::UNPARTITIONED, None, None, None),
                min_reservation_bytes: None,
                initial_reservation_total_claims: None,
                query_global_dicts: None,
                load_global_dicts: None,
                cache_param: None,
                query_global_dict_exprs: None,
                group_execution_param: None,
            }),
            desc_tbl,
            params: None,
            coord: None,
            backend_num: None,
            query_globals: None,
            query_options: None,
            enable_profile: None,
            resource_info: None,
            import_label: None,
            db_name: None,
            load_job_id: None,
            load_error_hub_info: None,
            is_pipeline: None,
            pipeline_dop: None,
            per_scan_node_dop: None,
            workgroup: None,
            enable_resource_group: None,
            func_version: None,
            enable_shared_scan: None,
            is_stream_pipeline: None,
            adaptive_dop_param: None,
            group_execution_scan_dop: None,
            pred_tree_params: None,
            exec_stats_node_ids: None,
            arrow_flight_sql_version: None,
        }
    }

    fn scan_plan(node_id: i32, tuple_id: i32) -> TPlan {
        // Build a single-node scan plan so coverage is focused on one-node fragments first.
        TPlan::new(vec![scan_node(node_id, tuple_id)])
    }

    fn serialize_binary<T>(value: &T) -> Vec<u8>
    where
        T: TSerializable,
    {
        // Serialize fixtures exactly like FE BRPC attachments: thrift binary protocol bytes.
        let channel = TBufferChannel::with_capacity(0, 64 * 1024);
        let (_, write) = channel.clone().split().unwrap();
        let mut protocol = TBinaryOutputProtocol::new(write, true);
        value.write_to_out_protocol(&mut protocol).unwrap();
        channel.write_bytes()
    }

    fn scalar_type(primitive: TPrimitiveType) -> TTypeDesc {
        // Descriptor-table slots use StarRocks thrift scalar type descriptors.
        TTypeDesc::new(Some(vec![TTypeNode::new(
            TTypeNodeType::SCALAR,
            Some(TScalarType::new(primitive, None, None, None)),
            None,
            None,
        )]))
    }

    fn slot(id: i32, tuple_id: i32, column_pos: i32, name: &str, ty: TTypeDesc) -> TSlotDescriptor {
        // Materialized slots define the output schema visible to the translator.
        TSlotDescriptor::new(
            Some(id),
            Some(tuple_id),
            Some(ty),
            Some(column_pos),
            None,
            None,
            None,
            Some(name.to_string()),
            None,
            Some(true),
            Some(true),
            Some(true),
            None,
            None,
        )
    }

    fn table_descriptor(id: i64, db: &str, name: &str, num_cols: i32) -> TTableDescriptor {
        // HDFS table descriptors are enough for the translator to recover table names.
        TTableDescriptor::new(
            id,
            TTableType::HDFS_TABLE,
            num_cols,
            0,
            name.to_string(),
            db.to_string(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn desc_table() -> TDescriptorTable {
        // Generic descriptor table for the single-fragment smoke test.
        TDescriptorTable::new(
            Some(vec![
                slot(1, 0, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
                slot(2, 0, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            ]),
            vec![TTupleDescriptor::new(Some(0), None, None, Some(100), None)],
            Some(vec![table_descriptor(100, "tpch", "users", 2)]),
            None,
        )
    }

    fn tpch_desc_table() -> TDescriptorTable {
        // Two TPCH tables let the batch test exercise multiple one-node scan fragments.
        TDescriptorTable::new(
            Some(vec![
                slot(1, 0, 0, "l_orderkey", scalar_type(TPrimitiveType::BIGINT)),
                slot(2, 0, 1, "l_quantity", scalar_type(TPrimitiveType::DOUBLE)),
                slot(3, 1, 0, "o_orderkey", scalar_type(TPrimitiveType::BIGINT)),
                slot(4, 1, 1, "o_orderdate", scalar_type(TPrimitiveType::DATE)),
            ]),
            vec![
                TTupleDescriptor::new(Some(0), None, None, Some(100), None),
                TTupleDescriptor::new(Some(1), None, None, Some(101), None),
            ],
            Some(vec![
                table_descriptor(100, "tpch", "lineitem", 2),
                table_descriptor(101, "tpch", "orders", 2),
            ]),
            None,
        )
    }

    fn scan_node(node_id: i32, tuple_id: i32) -> TPlanNode {
        // File scan nodes are currently a supported translator surface.
        TPlanNode::new(
            node_id,
            TPlanNodeType::FILE_SCAN_NODE,
            0,
            -1,
            vec![tuple_id],
            Vec::new(),
            Some(Vec::new()),
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(TFileScanNode::new(tuple_id, None, None, None)),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }
}
