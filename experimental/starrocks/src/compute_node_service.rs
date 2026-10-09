use std::net::{SocketAddr, ToSocketAddrs};
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
    DrainHandoff, FilterKeys, FilterRun, FragmentExecutor, FragmentRun, SenderSlot,
};
use crate::local_exchange::{
    ExchangeKey, LocalExchange, ReadyExchangeInput, ReadyFragment, RemoteBatch, SenderSource,
};
use crate::nixl_chunk::{self, AllocError, NixlEndpoint, NixlEnvelope, StreamHop};
use crate::proto::starrocks::{
    ExecuteCommandRequestPb, ExecuteCommandResultPb, PCancelPlanFragmentRequest,
    PCancelPlanFragmentResult, PExecBatchPlanFragmentsRequest, PExecBatchPlanFragmentsResult,
    PExecPlanFragmentRequest, PExecPlanFragmentResult, PFetchDataRequest, PFetchDataResult,
    PGetFileSchemaRequest, PGetFileSchemaResult, PSlotDescriptor, PTransmitChunkParams,
    PTransmitChunkResult, StatusPb, p_internal_service_brpc::PInternalService,
};
use crate::result_encoder::{self, ThriftBinary};
use crate::result_store::{FragmentInstanceId, ResultStore};
use crate::runtime_filters::{self, DeferredScan, FILTER_STREAM_BASE, RuntimeFilters};
use starrocks_plan_translator::runtime_filter::{self, FilterInput};
use starrocks_plan_translator::{ExchangeInput, PlanTranslator, StreamInputColumn, TranslatedPlan};
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

/// How long a receive allocation that finds the pool full waits for a purge's frees
/// (`SIRIUS_CN_ALLOC_WAIT_MS`, default 30 s), before it fails.
fn alloc_wait_from_env() -> Duration {
    std::env::var("SIRIUS_CN_ALLOC_WAIT_MS")
        .ok()
        .and_then(|ms| ms.parse().ok())
        .map_or(Duration::from_secs(30), Duration::from_millis)
}

/// How often the deadline watcher checks for queries whose time is up.
const DEADLINE_TICK: Duration = Duration::from_millis(100);

/// How much longer than its query's deadline a `fetch_data` waits, so the deadline's purge, not
/// the wait, is what fails it, with the timeout as its cause.
const DEADLINE_GRACE: Duration = Duration::from_secs(5);

/// Bounds a `fetch_data` wait on a result fragment whose exchange senders never finish, when the
/// query has no timeout.
const RESULT_WAIT: Duration = Duration::from_secs(600);

/// Sirius compute-node implementation of StarRocks PInternalService.
///
/// Plan-fragment translation is the first implemented RPC path; future
/// compute-node tasks should land here behind the generated service facade.
#[derive(Clone, Debug)]
pub(crate) struct SiriusComputeNodeService {
    /// Reusable StarRocks thrift-to-Substrait fragment translator.
    translator: PlanTranslator,
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
    /// This CN's advertised brpc endpoint. A sink destination is local only when host and port
    /// both match, so two CNs on one host see each other as remote.
    brpc_address: TNetworkAddress,
    /// This CN's NIXL side, when it has one: serves peers' `transmit_chunk` requests and ships
    /// output to remote destinations.
    nixl: Option<Arc<dyn NixlEndpoint>>,
    /// Runtime filters this CN builds from broadcast joins, and the scans waiting for them.
    filters: Arc<RuntimeFilters>,
    /// The final `reportExecStatus` each dispatched instance still owes the FE.
    reports: Arc<ExecReports>,
    /// Survey mode (`SIRIUS_CN_TRANSLATE_ONLY`): accept and translate every fragment, run none.
    translate_only: bool,
    /// How each recent query ended on this CN, shared with the exchange and the reports.
    ends: Arc<QueryEnds>,
    /// How long a receive allocation that finds the pool full waits for a purge's frees, at
    /// most: the query's own deadline may cut it shorter.
    alloc_wait: Duration,
    /// Each query's deadline on this CN, from its query timeout.
    deadlines: Arc<Deadlines>,
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

/// The runtime filters a fragment run applies, and the plan to run instead if their keys cannot
/// be copied.
#[derive(Default)]
struct FilterPlan<'a> {
    filters: Vec<FilterRun>,
    fallback: Option<&'a TranslatedPlan>,
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
        Self {
            translator: PlanTranslator::new(),
            executor,
            results: Arc::new(ResultStore::default()),
            descriptor_tables: Arc::new(Mutex::new(RecentQueries::new(
                recent_queries::REMEMBER_FOR,
                CACHED_DESCRIPTOR_TABLES,
            ))),
            exchanges: Arc::new(LocalExchange::new(Arc::clone(&ends))),
            brpc_address: TNetworkAddress::new(
                compute_node.advertise_host.to_string(),
                i32::from(compute_node.brpc_port),
            ),
            nixl,
            filters: Arc::new(RuntimeFilters::default()),
            reports: Arc::new(ExecReports::new(None, Arc::clone(&ends))),
            translate_only: std::env::var_os("SIRIUS_CN_TRANSLATE_ONLY").is_some(),
            ends,
            alloc_wait: alloc_wait_from_env(),
            deadlines: Arc::default(),
        }
    }

    /// Reports instances' ends only to coordinators on `frontend_host`, the configured FE.
    pub(crate) fn reporting_to(mut self, frontend_host: Option<crate::Host>) -> Self {
        self.reports = Arc::new(ExecReports::new(frontend_host, Arc::clone(&self.ends)));
        self
    }
}

impl PInternalService for SiriusComputeNodeService {
    /// Handles a single FE-dispatched plan fragment thrift attachment. A fragment fed by an
    /// exchange returns OK once registered and runs when its senders have parked. Any other
    /// fragment runs now: a RESULT_SINK buffers its rows for `fetch_data`, a DATA_STREAM_SINK
    /// parks its output for its receivers.
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

    /// `ADMIN EXECUTE ON <node_id> '<script>'` — the FE's execute_script RPC, repurposed as this
    /// CN's admin channel (`pin_table`/`unpin_table`, see [`crate::admin_command`]). Runs on a
    /// blocking worker because a pin occupies the engine thread for the whole materialization.
    #[instrument(skip_all)]
    async fn execute_command(
        &self,
        request: ExecuteCommandRequestPb,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<ExecuteCommandResultPb>, crate::prpc::Error> {
        let service = self.clone();
        let outcome =
            tokio::task::spawn_blocking(move || service.handle_execute_command(&request)).await;
        let (status, result) = match outcome {
            Ok(Ok(text)) => (Self::ok_status(), text),
            Ok(Err(err)) => (Self::internal_error(err), String::new()),
            Err(join_err) => (
                Self::internal_error(format!("execute_command task panicked: {join_err}")),
                String::new(),
            ),
        };
        // Both fields always set: the FE dereferences status.statusCode and splits result on
        // '\n' without null checks (ExecuteScriptExecutor.java).
        Ok(ExecuteCommandResultPb {
            status: Some(status),
            result: Some(result),
        }
        .into())
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

    /// Serves a peer CN's NIXL exchange control and batch announces.
    #[instrument(skip_all)]
    async fn transmit_chunk(
        &self,
        request: PTransmitChunkParams,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PTransmitChunkResult>, crate::prpc::Error> {
        let handled = match NixlEnvelope::decode(&attachment) {
            Ok(NixlEnvelope::Alloc(layout)) => self
                .allocate_waiting(&request, layout)
                .await
                .map(|reply| (reply, None)),
            _ => self.handle_nixl_chunk(&request, &attachment),
        };
        let (status, reply) = match handled {
            Ok((reply, ready)) => {
                if let Some(ready) = ready {
                    self.drain_ready_async(ready);
                }
                (Self::ok_status(), reply)
            }
            // A frame of a query that already failed here is refused with that failure's cause,
            // as CANCELLED, so the sender fails with the cause rather than with this refusal.
            Err(err) => match request
                .finst_id
                .as_ref()
                .and_then(|id| self.exchanges.failure(FragmentInstanceId::from(id)))
            {
                Some(cause) => {
                    // The refused frame's buffers were just freed, after its query's purge last
                    // logged what this CN holds.
                    self.log_leak_counters("refused frame");
                    (Self::cancelled(cause), Vec::new())
                }
                None => (Self::internal_error(err), Vec::new()),
            },
        };
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

    /// Executes an `ADMIN EXECUTE` script: parse, then run each command on the fragment
    /// executor in order, stopping at the first failure. Returns one summary line per command
    /// ('\n'-joined — the FE renders each line as a result row).
    fn handle_execute_command(
        &self,
        request: &ExecuteCommandRequestPb,
    ) -> std::result::Result<String, String> {
        // The FE hardcodes this command name for ADMIN EXECUTE (ExecuteScriptExecutor.java);
        // reject anything else by name rather than guessing at its payload.
        match request.command.as_deref() {
            Some("execute_script") => {}
            other => {
                return Err(format!(
                    "unsupported execute_command command {other:?}; this CN only accepts \
                     'execute_script' (ADMIN EXECUTE ON <node_id> '<script>')"
                ));
            }
        }
        let script = request
            .params
            .as_deref()
            .filter(|params| !params.trim().is_empty())
            .ok_or_else(|| {
                "empty script; supported commands: pin_table path=<file-or-glob> \
                 tier=gpu|host name=<name> [cols=c1,c2,...] [format=parquet|duckdb] \
                 [schema=<schema>] | unpin_table <name>"
                    .to_string()
            })?;
        // Bound parse/log cost; the grammar never needs scripts anywhere near this size.
        const MAX_SCRIPT_BYTES: usize = 64 * 1024;
        if script.len() > MAX_SCRIPT_BYTES {
            return Err(format!(
                "script is {} bytes; the CN caps admin scripts at {MAX_SCRIPT_BYTES}",
                script.len()
            ));
        }
        tracing::info!(script, "execute_command admin script accepted");
        let commands = crate::admin_command::parse_script(script)?;
        let total = commands.len();
        let mut lines = Vec::with_capacity(total);
        for (index, command) in commands.into_iter().enumerate() {
            let line = match &command {
                crate::admin_command::AdminCommand::PinTable(spec) => self.executor.pin_table(spec),
                crate::admin_command::AdminCommand::UnpinTable { name } => {
                    self.executor.unpin_table(name)
                }
            }
            .map_err(|err| format!("command {} of {total}: {err}", index + 1))?;
            lines.push(line);
        }
        let outcome = lines.join("\n");
        tracing::info!(outcome, "execute_command admin script finished");
        Ok(outcome)
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

    /// Runs a leaf fragment now, or registers a receiver to run once its senders finish. Returns
    /// the receivers that are ready to run.
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
                        self.filters.record_builds(id, built);
                    }
                    Ok(_) => {}
                    Err(err) => warn!(error = %err, "ignoring a receiver's runtime filters"),
                }
            }
            let ready = self
                .exchanges
                .register_receiver(id, expected_senders, params)?;
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

    /// Defers a leaf fragment whose scans probe runtime filters that a receiver on this CN builds
    /// from a broadcast join. A timer runs it unfiltered if its filters take too long.
    fn defer_for_filters(&self, params: &TExecPlanFragmentParams) -> bool {
        if !runtime_filters::enabled() {
            return false;
        }
        let probed: Vec<i32> = runtime_filter::probed_filters(params)
            .into_iter()
            .map(|probe| probe.filter_id)
            .collect();
        let (Some(query), Some(instance)) =
            (Self::query_id(params), Self::fragment_instance_id(params))
        else {
            return false;
        };
        if probed.is_empty() {
            return false;
        }
        let sites = self.filters.sites(query, &probed);
        if sites.is_empty() {
            info!(%instance, probed = ?probed, "no runtime filter of this scan is built on this CN");
            return false;
        }
        info!(
            %instance,
            filters = ?sites.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "deferring a scan until its runtime filters arrive"
        );
        self.filters.defer(
            instance,
            DeferredScan {
                params: params.clone(),
                filters: sites,
                deferred_at: std::time::Instant::now(),
            },
        );
        let service = self.clone();
        let wait = runtime_filters::wait_limit();
        let _ = std::thread::Builder::new()
            .name("runtime-filter-timer".to_string())
            .spawn(move || {
                std::thread::sleep(wait);
                if let Some(scan) = service.filters.take(instance) {
                    warn!(%instance, ?wait, "runtime filters did not arrive in time; scanning unfiltered");
                    service.run_deferred(scan, false);
                }
            });
        true
    }

    /// Runs, each on its own thread, the deferred scans whose filters' keys are all here. Called
    /// after every sender completes an exchange, before that exchange's receiver can run, so the
    /// engine copies the keys before the receiver consumes them.
    fn dispatch_filtered_scans(&self) {
        let ready = self
            .filters
            .take_ready(|site| self.exchanges.complete_sources(site.key).is_some());
        for scan in ready {
            let service = self.clone();
            let _ = std::thread::Builder::new()
                .name("filtered-scan".to_string())
                .spawn(move || service.run_deferred(scan, true));
        }
    }

    /// Runs a deferred scan with the filters whose keys are worth applying, or unfiltered.
    fn run_deferred(&self, scan: DeferredScan, filtered: bool) {
        let query = Self::query_id(&scan.params);
        self.contain_panic(query, || self.run_deferred_scan(scan, filtered));
    }

    fn run_deferred_scan(&self, scan: DeferredScan, filtered: bool) {
        let query = Self::query_id(&scan.params);
        let ran = self
            .run_with_filters(&scan, filtered)
            .map_err(|failure| self.end_hops(&scan.params, failure));
        self.reports
            .finish(&scan.params, ran.as_ref().map(drop).map_err(String::as_str));
        let result = ran.and_then(|ready| self.drain_ready(ready));
        if let Err(err) = result {
            warn!(error = %err, "a scan deferred for runtime filters failed");
            if let Some(query) = query {
                self.fail_and_purge(query, &err);
            }
        }
        self.log_leak_counters("filtered scan");
    }

    fn run_with_filters(
        &self,
        scan: &DeferredScan,
        filtered: bool,
    ) -> std::result::Result<Vec<ReadyFragment>, FragmentFailure> {
        let params = &scan.params;
        let dump_seq = Self::dump_fragment(params);
        let unfiltered = self.translate_fragment_logged(params, &[], dump_seq)?;
        let mut inputs = Vec::new();
        let mut runs = Vec::new();
        for (filter_id, site) in scan.filters.iter().filter(|_| filtered) {
            let Some(sources) = self.exchanges.complete_sources(site.key) else {
                info!(
                    filter_id,
                    "runtime filter keys were taken before the scan ran"
                );
                continue;
            };
            let keys = FilterKeys {
                column: site.column,
                sources,
            };
            let stats = match self.executor.key_stats(&keys) {
                Ok(stats) => stats,
                Err(err) => {
                    warn!(filter_id, error = %err, "cannot read runtime filter keys; skipping it");
                    continue;
                }
            };
            let density = runtime_filters::max_density();
            if !runtime_filters::selective(&stats, density) {
                info!(
                    filter_id,
                    ?stats,
                    density,
                    "runtime filter keys fill their range; skipping it"
                );
                continue;
            }
            let node_id = FILTER_STREAM_BASE + filter_id;
            inputs.push(FilterInput {
                filter_id: *filter_id,
                node_id,
                stream_view: format!("sirius_stream_{node_id}"),
                column: StreamInputColumn {
                    name: "rf_key".to_string(),
                    ty: site.column_type.clone(),
                },
            });
            runs.push(FilterRun {
                stream_id: node_id as u64,
                keys,
                rows: stats.rows,
            });
            info!(
                filter_id,
                ?stats,
                waited_ms = scan.deferred_at.elapsed().as_millis() as u64,
                "applying runtime filter"
            );
        }
        let filtered_plan = if inputs.is_empty() {
            None
        } else {
            match self
                .translator
                .translate_fragment_with_inputs(params, &[], &inputs)
            {
                Ok(plan) => Some(plan),
                Err(err) => {
                    warn!(error = %err, "cannot translate the scan with its runtime filters");
                    None
                }
            }
        };
        match &filtered_plan {
            Some(plan) => {
                // Only the filters the plan reads; a filter of another scan type stays unbound.
                runs.retain(|run| {
                    plan.stream_inputs
                        .iter()
                        .any(|input| input.node_id as u64 == run.stream_id)
                });
                self.execute_fragment(
                    params,
                    plan,
                    Vec::new(),
                    Vec::new(),
                    FilterPlan {
                        filters: runs,
                        fallback: Some(&unfiltered),
                    },
                )
            }
            None => self.execute_fragment(
                params,
                &unfiltered,
                Vec::new(),
                Vec::new(),
                FilterPlan::default(),
            ),
        }
    }

    /// Answers one SRNX request on the brpc thread, never waiting on the engine or the transport
    /// thread. A Packed frame returns the receiver it completed.
    fn handle_nixl_chunk(
        &self,
        request: &PTransmitChunkParams,
        attachment: &[u8],
    ) -> std::result::Result<(Vec<u8>, Option<ReadyFragment>), String> {
        nixl_chunk::reject_native_chunk(request)?;
        let envelope = NixlEnvelope::decode(attachment)?;
        let nixl = self.nixl.as_ref().ok_or("this CN has no NIXL transport")?;
        match envelope {
            NixlEnvelope::Md(_) => Ok((nixl.local_md(), None)),
            NixlEnvelope::Alloc(layout) => Ok((self.allocate(request, &layout)?, None)),
            NixlEnvelope::Release(token) => {
                nixl.release(token);
                // Only a failed hop releases buffers this way, after its query's purge last
                // logged what this CN holds.
                self.log_leak_counters("released buffer");
                Ok((Vec::new(), None))
            }
            NixlEnvelope::Packed { token, rows, names } => {
                // Sealed before the rendezvous sees it, so a receiver this frame completes always
                // finds it sealed. The batch can then spill to host while it waits.
                if token != 0 {
                    nixl.seal(token).inspect_err(|_| nixl.release(token))?;
                }
                // A refused frame is never pushed, so its buffers are freed here. A duplicate is
                // accepted as a no-op: it carries the token its first copy already delivered.
                let ready = self
                    .push_packed(request, token, rows, names)
                    .inspect_err(|_| nixl.release(token))?;
                if request.eos == Some(true) {
                    self.dispatch_filtered_scans();
                }
                Ok((Vec::new(), ready))
            }
            NixlEnvelope::Failed { error } => {
                self.remote_sender_failed(request, &error)?;
                Ok((Vec::new(), None))
            }
        }
    }

    /// Receive buffers for a batch with `layout`, on behalf of `request`'s sender. None for a
    /// query that already failed here: its sender stops at once, with the cause.
    fn allocate(
        &self,
        request: &PTransmitChunkParams,
        layout: &[u8],
    ) -> std::result::Result<Vec<u8>, AllocError> {
        nixl_chunk::reject_native_chunk(request).map_err(AllocError::Failed)?;
        let nixl = self
            .nixl
            .as_ref()
            .ok_or_else(|| AllocError::Failed("this CN has no NIXL transport".to_string()))?;
        if let Some(cause) = request
            .finst_id
            .as_ref()
            .and_then(|id| self.exchanges.failure(FragmentInstanceId::from(id)))
        {
            return Err(AllocError::Failed(cause));
        }
        nixl.allocate(layout).map(|reply| reply.encode())
    }

    /// [`allocate`](Self::allocate), and when the pool is full while a purge's frees are pending
    /// (or one finished meanwhile), waits for them off the brpc runtime, bounded by
    /// `alloc_wait`, and tries once more. A late free then delays the next query instead of
    /// failing it. Any other failure, a refused query's included, answers at once.
    async fn allocate_waiting(
        &self,
        request: &PTransmitChunkParams,
        layout: Vec<u8>,
    ) -> std::result::Result<Vec<u8>, String> {
        let frees = self.executor.pending_frees();
        let done_before = frees.as_ref().map(|frees| frees.done());
        let err = match self.allocate(request, &layout) {
            Err(AllocError::PoolFull(err)) => err,
            allocated => return allocated.map_err(String::from),
        };
        let Some(frees) =
            frees.filter(|frees| frees.pending() > 0 || Some(frees.done()) != done_before)
        else {
            return Err(err);
        };
        let (service, request) = (self.clone(), request.clone());
        tokio::task::spawn_blocking(move || {
            let started = std::time::Instant::now();
            let wait = request
                .finst_id
                .as_ref()
                .and_then(|id| service.deadlines.remaining(FragmentInstanceId::from(id)))
                // Just past the deadline, so its purge has refused the query by the retry.
                .map_or(service.alloc_wait, |left| {
                    (left + 2 * DEADLINE_TICK).min(service.alloc_wait)
                });
            let freed = frees.wait_for_none(wait);
            info!(
                freed,
                waited_ms = started.elapsed().as_millis() as u64,
                error = err,
                "a receive allocation waited for a purge's frees"
            );
            service.allocate(&request, &layout).map_err(String::from)
        })
        .await
        .unwrap_or_else(|join_err| Err(format!("the allocation task panicked: {join_err}")))
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

    /// Hands a remote sender's batch (none under token 0) and eos to the exchange rendezvous.
    fn push_packed(
        &self,
        request: &PTransmitChunkParams,
        token: u64,
        rows: u64,
        names: Vec<String>,
    ) -> std::result::Result<Option<ReadyFragment>, String> {
        let missing = |field: &str| format!("Packed transmit_chunk is missing {field}");
        let finst_id = request
            .finst_id
            .as_ref()
            .ok_or_else(|| missing("finst_id"))?;
        let key = ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from(finst_id),
            node_id: request.node_id.ok_or_else(|| missing("node_id"))?,
        };
        self.exchanges.push_remote_frame(
            key,
            request.sender_id.ok_or_else(|| missing("sender_id"))?,
            request.sequence.ok_or_else(|| missing("sequence"))?,
            request.eos.ok_or_else(|| missing("eos"))?,
            names,
            (token != 0).then_some(RemoteBatch { token, rows }),
        )
    }

    /// Runs a receiver a remote frame completed on its own thread, so `transmit_chunk` answers
    /// the sender without waiting on GPU work.
    fn drain_ready_async(&self, ready: ReadyFragment) {
        let service = self.clone();
        let _ = std::thread::Builder::new()
            .name("exchange-receiver".to_string())
            .spawn(move || {
                let query = Self::query_id(&ready.params);
                service.contain_panic(query, || {
                    if let Err(err) = service.drain_ready(vec![ready]) {
                        tracing::warn!(error = %err, "a receiver fed by a remote sender failed");
                    }
                });
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
        let is_cached_reference = desc.is_cached == Some(true)
            && desc.slot_descriptors.as_ref().is_none_or(Vec::is_empty)
            && desc.tuple_descriptors.is_empty()
            && desc.table_descriptors.as_ref().is_none_or(Vec::is_empty);
        let mut cache = self
            .descriptor_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if is_cached_reference {
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

    /// Writes the received fragment params to `$SIRIUS_CN_DUMP_FRAGMENTS/fragment-<seq>.txt`
    /// (debug format) for offline plan analysis. No-op when the variable is unset.
    fn dump_fragment(params: &TExecPlanFragmentParams) -> Option<u64> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let Ok(dir) = std::env::var("SIRIUS_CN_DUMP_FRAGMENTS") else {
            return None;
        };
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::path::Path::new(&dir).join(format!("fragment-{seq:04}.txt"));
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

        let Some(sink) = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
        else {
            tracing::warn!("fragment carries no output sink; nothing consumes its output");
            return Ok(Vec::new());
        };
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

        let sender_id = exec.sender_id.unwrap_or(0);
        let mut outputs = Vec::with_capacity(destinations.len());
        let mut remote = Vec::new();
        for destination in destinations {
            let slot = SenderSlot {
                fragment_instance_id: FragmentInstanceId::from(&destination.fragment_instance_id),
                node_id: stream_sink.dest_node_id,
                sender_id,
            };
            outputs.push(slot);
            // Resolved before running so an unreachable destination leaves no output parked.
            if let Some(peer) = self.remote_peer(destination)? {
                remote.push((slot, peer));
            }
        }
        let on_exchange_input = !inputs.is_empty() || !remote_inputs.is_empty();
        if on_exchange_input && !remote.is_empty() && Self::claim_injection("remote-sender") {
            return Err(
                "injected fragment failure (SIRIUS_CN_FAIL_ONCE_FILE, remote-sender)"
                    .to_string()
                    .into(),
            );
        }
        if !remote.is_empty() && Self::claim_injection("hang-sender") {
            warn!(
                instance = %FragmentInstanceId::from(&exec.fragment_instance_id),
                "injected hang (SIRIUS_CN_FAIL_ONCE_FILE, hang-sender): sending nothing"
            );
            return Ok(Vec::new());
        }
        let run = FragmentRun {
            plan: translated,
            inputs,
            remote_inputs,
            outputs: outputs.clone(),
            broadcast,
            hash_keys,
            drains: None,
            filters: filters.filters,
            fallback: filters.fallback,
            query: Self::query_id(params),
        };
        let shipped = if remote.is_empty() || !stream_output_enabled() {
            self.executor.run_fragment(run)?;
            self.ship_parked(&remote, &translated.output_names)
        } else {
            self.run_streaming(run, &outputs, &remote, &translated.output_names)?
        };

        // If a hop failed, the local claims are released too: their receivers can no longer
        // complete, and nothing may stay parked.
        let local = outputs
            .into_iter()
            .filter(|slot| remote.iter().all(|(remote_slot, _)| remote_slot != slot));
        if let Err(err) = shipped {
            for slot in local {
                let _ = self.executor.drop_parked(slot);
            }
            return Err(FragmentFailure::after_hops(err));
        }
        let local: Vec<SenderSlot> = local.collect();
        let mut ready = Vec::new();
        for (index, &slot) in local.iter().enumerate() {
            let key = ExchangeKey {
                fragment_instance_id: slot.fragment_instance_id,
                node_id: slot.node_id,
            };
            let source = SenderSource::LocalParked {
                names: translated.output_names.clone(),
                slot,
            };
            // A refused source is not kept (its query was purged, say), so this slot and the
            // ones not yet handed over are dropped here or they stay parked.
            match self.exchanges.push_sender(key, sender_id, source) {
                Ok(completed) => ready.extend(completed),
                Err(err) => {
                    for &unpushed in &local[index..] {
                        let _ = self.executor.drop_parked(unpushed);
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
        outputs: &[SenderSlot],
        remote: &[(SenderSlot, SocketAddr)],
        names: &[String],
    ) -> std::result::Result<std::result::Result<(), String>, FragmentFailure> {
        let nixl = self
            .nixl
            .as_ref()
            .ok_or("this CN has no NIXL transport".to_string())?;
        let streams = remote
            .iter()
            .map(|(slot, _)| outputs.iter().position(|output| output == slot))
            .collect::<Option<Vec<_>>>()
            .ok_or("a remote destination is not one of the fragment's outputs".to_string())?;
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

    /// Ships every remote output from its parked batches, one destination at a time. Every hop
    /// runs and releases its own claim; the first error is returned.
    fn ship_parked(
        &self,
        remote: &[(SenderSlot, SocketAddr)],
        names: &[String],
    ) -> std::result::Result<(), String> {
        let mut shipped = Ok(());
        for &(slot, peer) in remote {
            let hop = self.ship_remote(slot, peer, names);
            shipped = shipped.and(hop);
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
        if *address == self.brpc_address {
            return Ok(None);
        }
        let host = address.hostname.as_str();
        if self.nixl.is_none() {
            return Err(format!(
                "DATA_STREAM_SINK destination {host}:{} for fragment instance {id} is remote, and \
                 this CN has no NIXL transport",
                address.port
            ));
        }
        let port = u16::try_from(address.port)
            .map_err(|_| format!("destination brpc port {} is not a TCP port", address.port))?;
        (host, port)
            .to_socket_addrs()
            .map_err(|err| format!("failed to resolve exchange peer {host}:{port}: {err}"))?
            .next()
            .map(Some)
            .ok_or_else(|| format!("exchange peer {host}:{port} resolved to no address"))
    }

    /// Ships the output parked under `slot` to `peer`, then releases that claim whether or not
    /// the hop succeeded.
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
        while let Some(ReadyFragment { params, inputs }) = queue.pop() {
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
        let deferred = self.filters.purge_query(query);
        let purged = self.exchanges.purge_query(query, error);
        // A failed query's run in progress holds GPU memory and works for nothing: stop it, now
        // that its cause is recorded, so the interrupted fragment reports that cause first. A
        // finished query has nothing left running.
        if failed {
            self.executor.interrupt(query);
        }
        for params in deferred
            .iter()
            .map(|scan| &scan.params)
            .chain(&purged.receivers)
        {
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
        let (released_buffers, dropped_parked) = (purged.tokens.len(), purged.slots.len());
        let (dropped_receivers, deferred_scans) = (purged.receivers.len(), deferred.len());
        if failed {
            info!(
                %query,
                released_buffers,
                dropped_parked,
                dropped_receivers,
                deferred_scans,
                error,
                "purged a failed query's exchange state"
            );
        } else {
            info!(
                %query,
                released_buffers,
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
        let mut guard = ReadyInputsGuard {
            nixl: self.nixl.clone(),
            executor: Arc::clone(&self.executor),
            tokens: Vec::new(),
            slots: Vec::new(),
        };
        for source in ready_inputs.iter().flat_map(|input| &input.sources) {
            match source {
                SenderSource::Remote { batches, .. } => {
                    guard.tokens.extend(batches.iter().map(|batch| batch.token))
                }
                SenderSource::LocalParked { slot, .. } => guard.slots.push(*slot),
            }
        }
        let exchange_inputs = Self::exchange_inputs(&ready_inputs)?;
        let mut inputs = Vec::with_capacity(ready_inputs.len());
        let mut remote_inputs = Vec::new();
        for input in ready_inputs {
            let mut slots = Vec::new();
            for source in input.sources {
                match source {
                    SenderSource::LocalParked { slot, .. } => slots.push(slot),
                    SenderSource::Remote {
                        sender_id, batches, ..
                    } => remote_inputs.push((input.node_id, sender_id, batches)),
                }
            }
            inputs.push((input.node_id, slots));
        }
        let dump_seq = Self::dump_fragment(params);
        let translated = self.translate_fragment_logged(params, &exchange_inputs, dump_seq)?;
        let next = self.execute_fragment(
            params,
            &translated,
            inputs,
            remote_inputs,
            FilterPlan::default(),
        )?;
        // The engine relayed every parked input and released it.
        guard.slots.clear();
        Ok(next)
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
                    return Err("exchange senders produced different output names".to_string());
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

        for (idx, instance) in instances.iter().enumerate() {
            let mut params = instance.clone();
            if params.desc_tbl.is_none() {
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

    /// Writes the translated Substrait plan bytes to `$SIRIUS_CN_DUMP_FRAGMENTS/plan-<seq>.substrait`
    /// so a failing plan can be replayed against the engine in isolation. No-op when unset.
    fn dump_substrait(translated: &TranslatedPlan, dump_seq: Option<u64>) {
        let Ok(dir) = std::env::var("SIRIUS_CN_DUMP_FRAGMENTS") else {
            return;
        };
        let Some(seq) = dump_seq else {
            return;
        };
        let path = std::path::Path::new(&dir).join(format!("plan-{seq:04}.substrait"));
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

    /// Extracts the parquet path from the binary-thrift attachment and infers its schema.
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

/// What a panic said, when it said it with a string.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
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
        fragment_executor::{DrainNext, ExportedBatch, FragmentResult, OutputDrain},
        local_exchange::ExchangeCounts,
        nixl_chunk::{AllocError, AllocReply},
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

    /// Records what the service asks of its NIXL side.
    #[derive(Debug, Default)]
    struct FakeNixl {
        released: Mutex<Vec<u64>>,
        sealed: Mutex<Vec<u64>>,
        sent: Mutex<Vec<(SocketAddr, SenderSlot)>>,
        /// Hops streamed while their fragment ran, with the rows each drain delivered.
        streamed: Mutex<Vec<(SocketAddr, SenderSlot, u64)>>,
        /// Hops ended with a failure frame, and its error.
        failed: Mutex<Vec<(SocketAddr, SenderSlot, String)>>,
        /// The CN every failure frame is delivered to, as `transmit_chunk` would.
        peer: Mutex<Option<SiriusComputeNodeService>>,
        /// The receive pool has no room: every allocation fails.
        full: std::sync::atomic::AtomicBool,
        /// A free that finishes, making room, just as the pool is found full.
        free_when_full: Mutex<Option<crate::running_query::PendingFree>>,
        /// Allocation fails for a reason waiting would not change.
        broken: std::sync::atomic::AtomicBool,
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

        fn allocate(&self, layout: &[u8]) -> Result<AllocReply, AllocError> {
            use std::sync::atomic::Ordering::SeqCst;
            if self.broken.load(SeqCst) {
                return Err(AllocError::Failed("bad layout".to_string()));
            }
            if self.full.load(SeqCst) {
                if let Some(free) = self.free_when_full.lock().unwrap().take() {
                    self.full.store(false, SeqCst);
                    drop(free);
                }
                return Err(AllocError::PoolFull(
                    "failed to allocate receive buffers: 0 available".to_string(),
                ));
            }
            Ok(AllocReply {
                token: 7,
                device: 1,
                buffers: vec![(0xB000, layout.len() as u64)],
            })
        }

        fn release(&self, token: u64) {
            self.released.lock().unwrap().push(token);
        }

        fn outstanding(&self) -> usize {
            0
        }

        fn seal(&self, token: u64) -> Result<(), String> {
            self.sealed.lock().unwrap().push(token);
            Ok(())
        }

        fn send(
            &self,
            peer: SocketAddr,
            slot: SenderSlot,
            _names: Vec<String>,
            _executor: Arc<dyn FragmentExecutor>,
        ) -> Result<(), String> {
            self.sent.lock().unwrap().push((peer, slot));
            Ok(())
        }

        fn stream(&self, hops: Vec<StreamHop>) -> Result<(), String> {
            let ends: Vec<_> = hops.iter().map(|hop| (hop.peer, hop.slot)).collect();
            for mut hop in hops {
                let mut rows = 0;
                loop {
                    match hop.drain.next(Duration::from_millis(1)) {
                        Ok(DrainNext::Batch(batch)) => rows += batch.rows,
                        Ok(DrainNext::Waiting) => {}
                        Ok(DrainNext::End) => break,
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

    /// Captures pin/unpin calls so execute_command tests can assert what reached the executor.
    #[derive(Debug, Default)]
    struct RecordingPinExecutor {
        pins: Mutex<Vec<crate::fragment_executor::PinTableSpec>>,
        unpins: Mutex<Vec<String>>,
        fail_with: Mutex<Option<String>>,
    }

    impl FragmentExecutor for RecordingPinExecutor {
        fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
            StubExecutor.execute(translated)
        }

        fn pin_table(
            &self,
            spec: &crate::fragment_executor::PinTableSpec,
        ) -> Result<String, String> {
            if let Some(err) = self.fail_with.lock().unwrap().clone() {
                return Err(err);
            }
            self.pins.lock().unwrap().push(spec.clone());
            Ok(format!("pinned '{}'", spec.name))
        }

        fn unpin_table(&self, name: &str) -> Result<String, String> {
            self.unpins.lock().unwrap().push(name.to_string());
            Ok(format!("unpinned '{name}'"))
        }
    }

    fn execute_command_response(
        service: &SiriusComputeNodeService,
        command: Option<&str>,
        params: Option<&str>,
    ) -> ExecuteCommandResultPb {
        let response = route(
            service,
            methods::EXECUTE_COMMAND,
            ExecuteCommandRequestPb {
                command: command.map(str::to_string),
                params: params.map(str::to_string),
            }
            .encode_to_vec(),
            Vec::new(),
        );
        ExecuteCommandResultPb::decode(response.body.as_slice()).unwrap()
    }

    #[test]
    fn execute_command_pins_and_unpins_via_executor() {
        let executor = Arc::new(RecordingPinExecutor::default());
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
        let result = execute_command_response(
            &service,
            Some("execute_script"),
            Some(
                "# warm cache\n\
                 pin_table path=/data/li/*.parquet tier=gpu name=lineitem cols=a,b\n\
                 unpin_table old_pin",
            ),
        );
        let status = result.status.expect("status is always set");
        assert_eq!(status.status_code, TStatusCode::OK.0, "{status:?}");
        let text = result.result.expect("result is always set");
        assert_eq!(text, "pinned 'lineitem'\nunpinned 'old_pin'");
        let pins = executor.pins.lock().unwrap();
        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].name, "lineitem");
        assert_eq!(pins[0].tier, crate::fragment_executor::PinTier::Gpu);
        assert_eq!(pins[0].cols, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(executor.unpins.lock().unwrap().as_slice(), ["old_pin"]);
    }

    #[test]
    fn execute_command_rejects_wrong_command_and_bad_grammar() {
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(RecordingPinExecutor::default()),
            &ComputeNodeConfig::default(),
            None,
        );
        for (command, params) in [
            (Some("run_groovy"), Some("pin_table tier=gpu name=x path=p")),
            (Some("execute_script"), Some("bogus_verb x")),
            (Some("execute_script"), None),
        ] {
            let result = execute_command_response(&service, command, params);
            let status = result.status.expect("status is always set");
            assert_eq!(status.status_code, TStatusCode::INTERNAL_ERROR.0);
            assert!(!status.error_msgs.is_empty());
            assert_eq!(result.result.as_deref(), Some(""));
        }
    }

    #[test]
    fn execute_command_propagates_executor_error_with_command_index() {
        let executor = Arc::new(RecordingPinExecutor::default());
        *executor.fail_with.lock().unwrap() = Some("no parquet files matched".to_string());
        let service =
            SiriusComputeNodeService::with_executor(executor, &ComputeNodeConfig::default(), None);
        let result = execute_command_response(
            &service,
            Some("execute_script"),
            Some("pin_table path=missing.parquet tier=gpu name=x"),
        );
        let status = result.status.expect("status is always set");
        assert_eq!(status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            status.error_msgs[0].contains("command 1 of 1")
                && status.error_msgs[0].contains("no parquet files matched"),
            "{:?}",
            status.error_msgs
        );
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
    fn transmit_chunk_serves_md_alloc_and_release() {
        let (service, nixl) = nixl_service(Arc::new(StubExecutor));
        let control = nixl_chunk::control_params;
        let (status, md) = transmit(&service, control(), NixlEnvelope::Md(b"peer".to_vec()));
        assert_eq!(
            (status.status_code, md),
            (TStatusCode::OK.0, b"local-md".to_vec())
        );
        let (status, reply) = transmit(&service, control(), NixlEnvelope::Alloc(vec![0; 24]));
        assert_eq!(status.status_code, TStatusCode::OK.0);
        assert_eq!(
            AllocReply::decode(&reply).unwrap(),
            AllocReply {
                token: 7,
                device: 1,
                buffers: vec![(0xB000, 24)],
            }
        );
        transmit(&service, control(), NixlEnvelope::Release(7));
        assert_eq!(*nixl.released.lock().unwrap(), [7]);
    }

    #[test]
    fn an_alloc_for_a_query_that_failed_here_is_refused_with_its_cause() {
        let (service, _) = nixl_service(Arc::new(StubExecutor));
        let alloc = || {
            let slot = SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(46, 10),
                node_id: 2,
                sender_id: 0,
            };
            transmit(
                &service,
                nixl_chunk::alloc_params(slot),
                NixlEnvelope::Alloc(vec![0; 24]),
            )
            .0
        };
        assert_eq!(alloc().status_code, TStatusCode::OK.0);
        assert_eq!(cancel(&service, 46).status_code, TStatusCode::OK.0);
        let status = alloc();
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

    /// Hands out drains for the outputs it is asked to stream, each delivering `rows` and then
    /// either the end or `fail_with`, and records the parked output the service drops.
    #[derive(Debug, Default)]
    struct StreamingExecutor {
        rows: u64,
        fail_with: Option<String>,
        dropped: Mutex<Vec<SenderSlot>>,
    }

    #[derive(Debug)]
    struct ScriptedDrain(Vec<Result<DrainNext, String>>);

    impl OutputDrain for ScriptedDrain {
        fn next(&mut self, _timeout: Duration) -> Result<DrainNext, String> {
            self.0.pop().unwrap_or(Ok(DrainNext::End))
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
                            None => Ok(DrainNext::End),
                        };
                        let batch = ExportedBatch {
                            token: 1,
                            rows: self.rows,
                            layout: Vec::new(),
                            src: Vec::new(),
                        };
                        // Popped from the back: waiting, a batch, then the last outcome.
                        let script =
                            vec![last, Ok(DrainNext::Batch(batch)), Ok(DrainNext::Waiting)];
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
        let status = exec(&service, &sender);
        assert_ne!(status.status_code, 0);
        assert!(
            status
                .error_msgs
                .iter()
                .any(|msg| msg.contains("scan exploded")),
            "{:?}",
            status.error_msgs
        );
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
    }

    impl FilterRecorder {
        fn new(stats: crate::fragment_executor::KeyStats) -> Self {
            Self {
                stats,
                runs: Mutex::new(Vec::new()),
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
            self.runs.lock().unwrap().push((
                run.plan
                    .stream_inputs
                    .iter()
                    .map(|input| input.node_id)
                    .collect(),
                run.filters.clone(),
                run.fallback.is_some(),
            ));
            Ok(None)
        }

        fn key_stats(
            &self,
            _keys: &crate::fragment_executor::FilterKeys,
        ) -> Result<crate::fragment_executor::KeyStats, String> {
            Ok(self.stats)
        }
    }

    fn sparse_keys() -> crate::fragment_executor::KeyStats {
        crate::fragment_executor::KeyStats {
            rows: 2_000_000,
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
        // This mirrors FE batch dispatch: shared descriptor metadata in common_param,
        // with per-instance fragments carrying only their scan plan.
        let batch = TExecBatchPlanFragmentsParams::new(
            Some(fragment_params(None, Some(tpch_desc_table()))),
            Some(vec![
                fragment_params(Some(scan_plan(0, 0)), None),
                fragment_params(Some(scan_plan(1, 1)), None),
            ]),
        );
        let result = call_exec_batch_plan_fragments(
            PExecBatchPlanFragmentsRequest {
                attachment_protocol: Some("binary".to_string()),
            },
            serialize_binary(&batch),
        );

        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);
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
        assert_eq!(
            exec(&service, &leaf).status_code,
            TStatusCode::INTERNAL_ERROR.0
        );

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
        let executor = Arc::new(RecordingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            None,
        );
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
        let nixl = Arc::new(FakeNixl::default());
        let service = SiriusComputeNodeService::with_executor(
            executor.clone(),
            &ComputeNodeConfig::default(),
            Some(nixl.clone()),
        );
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
        assert_eq!(
            exec(&service, &local).status_code,
            TStatusCode::INTERNAL_ERROR.0
        );
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
            let status = exec(&a, &sender);
            assert_eq!(status.error_msgs, ["scan exploded"], "query {query}");

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
        // A sender on CN a fails mid-stream. Its own dispatch reply carries the error, so it sends
        // no report; b's result receiver, which replied OK long ago, fails on the failure frame
        // and reports the sender's error. The FE's cancel afterwards adds nothing.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let failing = Arc::new(StreamingExecutor {
            rows: 1,
            fail_with: Some("scan exploded".to_string()),
            ..Default::default()
        });
        let [(a, _), (b, _)] = two_cns(failing, Arc::new(StubExecutor));
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

        // On the dispatch path: the local sender's dispatch runs the receiver, which panics. The
        // reply carries the panic, so the sender sends no report; the receiver reports it.
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(PanickingReceivers),
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
        assert_ne!(exec(&a, &sender).status_code, TStatusCode::OK.0);
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

    #[test]
    fn an_allocation_waits_for_a_purges_frees_up_to_a_bound() {
        let engine = Arc::new(BusyEngine::default());
        let nixl = Arc::new(FakeNixl::default());
        let mut service = SiriusComputeNodeService::with_executor(
            engine.clone(),
            &ComputeNodeConfig::default(),
            Some(nixl.clone()),
        );
        service.alloc_wait = Duration::from_millis(300);
        let alloc = |service: &SiriusComputeNodeService| {
            transmit(
                service,
                nixl_chunk::alloc_params(exchange_slot(83, 10, 2)),
                NixlEnvelope::Alloc(vec![0; 24]),
            )
            .0
        };
        nixl.full.store(true, std::sync::atomic::Ordering::SeqCst);

        // A purge's free is pending: the allocation waits for it, then succeeds.
        let pending = engine.frees.begin();
        let freeing = {
            let nixl = Arc::clone(&nixl);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                nixl.full.store(false, std::sync::atomic::Ordering::SeqCst);
                drop(pending);
            })
        };
        assert_eq!(alloc(&service).status_code, TStatusCode::OK.0);
        freeing.join().unwrap();

        // A free that never comes: it gives up after its bound.
        nixl.full.store(true, std::sync::atomic::Ordering::SeqCst);
        let _stuck = engine.frees.begin();
        let started = std::time::Instant::now();
        let status = alloc(&service);
        assert_eq!(status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(status.error_msgs[0].contains("0 available"), "{status:?}");
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(300) && waited < Duration::from_secs(5),
            "{waited:?}"
        );

        // Waiting would change nothing for a query that already failed here, or for an error
        // other than a full pool: both answer at once, even with a free pending.
        assert!(
            service
                .exchanges
                .mark_failed(FragmentInstanceId::from_halves(83, 0), "scan exploded")
        );
        let started = std::time::Instant::now();
        let status = alloc(&service);
        assert_eq!(
            (status.status_code, status.error_msgs),
            (TStatusCode::CANCELLED.0, vec!["scan exploded".to_string()])
        );
        assert!(started.elapsed() < Duration::from_millis(200));
        nixl.broken.store(true, std::sync::atomic::Ordering::SeqCst);
        let other = |service: &SiriusComputeNodeService| {
            transmit(
                service,
                nixl_chunk::alloc_params(exchange_slot(85, 10, 2)),
                NixlEnvelope::Alloc(vec![0; 24]),
            )
            .0
        };
        let started = std::time::Instant::now();
        let status = other(&service);
        assert_eq!(status.error_msgs, ["bad layout"]);
        assert!(started.elapsed() < Duration::from_millis(200));
        nixl.broken
            .store(false, std::sync::atomic::Ordering::SeqCst);

        // No purge pending: a full pool fails at once.
        drop(_stuck);
        let started = std::time::Instant::now();
        assert_eq!(other(&service).status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(started.elapsed() < Duration::from_millis(200));

        // A free that finishes just after the pool was found full still earns a retry.
        *nixl.free_when_full.lock().unwrap() = Some(engine.frees.begin());
        assert_eq!(other(&service).status_code, TStatusCode::OK.0);
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
        let status = exec(&service, &leaf);
        assert_eq!(
            status.error_msgs,
            ["scan exploded on cn2 (Interrupted!)"],
            "{status:?}"
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
    fn a_receive_allocation_waits_no_longer_than_its_querys_deadline() {
        let frontend = crate::fe_report::fake_frontend::FakeFrontend::start(TStatusCode::OK);
        let engine = Arc::new(BusyEngine::default());
        let nixl = Arc::new(FakeNixl::default());
        let mut service = SiriusComputeNodeService::with_executor(
            engine.clone(),
            &ComputeNodeConfig::default(),
            Some(nixl.clone()),
        );
        service.alloc_wait = Duration::from_secs(30);
        waiting_for_a_lost_sender(&service, &frontend, 91);
        nixl.full.store(true, std::sync::atomic::Ordering::SeqCst);
        let _stuck = engine.frees.begin();
        let started = std::time::Instant::now();
        let status = transmit(
            &service,
            nixl_chunk::alloc_params(exchange_slot(91, 10, 2)),
            NixlEnvelope::Alloc(vec![0; 24]),
        )
        .0;
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        // By then the deadline failed the query, which refuses the allocation with its cause.
        assert_eq!(
            (status.status_code, status.error_msgs),
            (
                TStatusCode::CANCELLED.0,
                vec!["query timed out after 1 s on this CN".to_string()]
            )
        );
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

    fn exec(service: &SiriusComputeNodeService, params: &TExecPlanFragmentParams) -> StatusPb {
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
        // Minimal single-node fragment with a descriptor table for direct translation.
        fragment_params(Some(scan_plan(0, 0)), Some(desc_table()))
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
