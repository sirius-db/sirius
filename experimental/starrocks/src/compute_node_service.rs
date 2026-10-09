use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::ComputeNodeConfig;
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
use crate::nixl_chunk::{self, NixlEndpoint, NixlEnvelope, StreamHop};
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

/// Bounds a `fetch_data` wait on a result fragment whose exchange senders never finish.
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
    descriptor_tables: Arc<Mutex<HashMap<FragmentInstanceId, TDescriptorTable>>>,
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
}

/// The runtime filters a fragment run applies, and the plan to run instead if their keys cannot
/// be copied.
#[derive(Default)]
struct FilterPlan<'a> {
    filters: Vec<FilterRun>,
    fallback: Option<&'a TranslatedPlan>,
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
        Self {
            translator: PlanTranslator::new(),
            executor,
            results: Arc::new(ResultStore::default()),
            descriptor_tables: Arc::new(Mutex::new(HashMap::new())),
            exchanges: Arc::new(LocalExchange::default()),
            brpc_address: TNetworkAddress::new(
                compute_node.advertise_host.to_string(),
                i32::from(compute_node.brpc_port),
            ),
            nixl,
            filters: Arc::new(RuntimeFilters::default()),
        }
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
            Ok(Err(err)) => Self::internal_error(err),
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
            Ok(Err(err)) => Self::internal_error(err),
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
        let outcome = tokio::task::spawn_blocking(move || results.take_next(id, RESULT_WAIT))
            .await
            .unwrap_or_else(|join_err| Err(format!("fetch_data wait task panicked: {join_err}")));
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                return Ok(Self::fetch_data_result(Self::internal_error(err), 0, true).into());
            }
        };
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

    /// Handles the FE's cancel of a failed or finished query. The FE sends one per CN with the
    /// query id and a dummy instance id, so the whole query is purged. Answers at once: dropping
    /// parked output waits on the engine thread, which may be running another fragment.
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
        let reason = request
            .error_message
            .unwrap_or_else(|| "the FE cancelled the query".to_string());
        let service = self.clone();
        tokio::task::spawn_blocking(move || {
            service.fail_and_purge(query, &format!("query cancelled: {reason}"));
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
        let (status, reply) = match self.handle_nixl_chunk(&request, &attachment) {
            Ok((reply, ready)) => {
                if let Some(ready) = ready {
                    self.drain_ready_async(ready);
                }
                (Self::ok_status(), reply)
            }
            Err(err) => (Self::internal_error(err), Vec::new()),
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
    ) -> std::result::Result<(), String> {
        Self::ensure_binary_protocol(protocol)?;
        let params = Self::deserialize_binary::<TExecPlanFragmentParams>(attachment)
            .map_err(|err| format!("failed to deserialize TExecPlanFragmentParams: {err}"))?;
        self.process_fragment(&params)
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
        let result = self.run_or_register(params);
        if let Err(err) = &result
            && let Some(query) = Self::query_id(params)
        {
            self.fail_and_purge(query, err);
        }
        self.log_leak_counters("fragment");
        result
    }

    /// Runs a leaf fragment now, or registers a receiver to run once its senders finish.
    fn run_or_register(&self, params: &TExecPlanFragmentParams) -> std::result::Result<(), String> {
        let params = self.resolve_descriptor_table(params)?;
        let dump_seq = Self::dump_fragment(&params);
        // Survey mode: accept every fragment so the FE dispatches (and we dump) the whole
        // plan even when translation fails. Queries still fail at fetch_data.
        if std::env::var_os("SIRIUS_CN_TRANSLATE_ONLY").is_some() {
            if let Err(err) = self.translate_fragment_logged(&params, &[], dump_seq) {
                tracing::warn!(error = %err, "translate-only mode: accepting untranslatable fragment");
            }
            return Ok(());
        }
        if runtime_filters::enabled() {
            Self::log_planned_skips(&params);
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
            return self.drain_ready(ready.into_iter().collect());
        }
        if self.defer_for_filters(&params) {
            // The scan runs once its filters' keys arrived; they may be here already.
            self.dispatch_filtered_scans();
            return Ok(());
        }
        let translated = self.translate_fragment_logged(&params, &[], dump_seq)?;
        let ready = self.execute_fragment(
            &params,
            &translated,
            Vec::new(),
            Vec::new(),
            FilterPlan::default(),
        )?;
        self.drain_ready(ready)
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

    /// Defers a leaf fragment whose scans probe runtime filters that a receiver on this CN builds
    /// from a broadcast join. A timer runs it unfiltered if its filters take too long.
    fn defer_for_filters(&self, params: &TExecPlanFragmentParams) -> bool {
        if !runtime_filters::enabled() {
            return false;
        }
        let probes = runtime_filter::probed_filters(params);
        let probed: Vec<i32> = probes.iter().map(|probe| probe.filter_id).collect();
        let (Some(query), Some(instance)) =
            (Self::query_id(params), Self::fragment_instance_id(params))
        else {
            return false;
        };
        if probed.is_empty() {
            return false;
        }
        let sites = self.filters.sites(query, &probed);
        for probe in &probes {
            if sites.iter().any(|(id, _)| *id == probe.filter_id) {
                continue;
            }
            // Only a broadcast join's filter can be built here; another holds part of the keys
            // on each CN.
            let reason = if probe.broadcast {
                "not_built_on_this_cn"
            } else {
                "not_broadcast"
            };
            runtime_filters::log_skipped(
                Some(query),
                Some(instance),
                probe.filter_id,
                Some(probe.scan_node_id),
                reason,
                "",
            );
        }
        if sites.is_empty() {
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
        let result = self
            .run_with_filters(&scan, filtered)
            .and_then(|ready| self.drain_ready(ready));
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
    ) -> std::result::Result<Vec<ReadyFragment>, String> {
        let params = &scan.params;
        let dump_seq = Self::dump_fragment(params);
        let unfiltered = self.translate_fragment_logged(params, &[], dump_seq)?;
        let (query, instance) = (Self::query_id(params), Self::fragment_instance_id(params));
        let skip = |filter_id: i32, reason: &str, detail: &str| {
            runtime_filters::log_skipped(query, instance, filter_id, None, reason, detail);
        };
        if !filtered {
            for (filter_id, _) in &scan.filters {
                skip(*filter_id, "timeout", "");
            }
        }
        let waited_ms = scan.deferred_at.elapsed().as_millis() as u64;
        let mut inputs = Vec::new();
        let mut runs = Vec::new();
        let mut applying = Vec::new();
        for (filter_id, site) in scan.filters.iter().filter(|_| filtered) {
            let Some(sources) = self.exchanges.complete_sources(site.key) else {
                skip(*filter_id, "keys_taken", "");
                continue;
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
            applying.push((*filter_id, node_id as u64, stats));
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
                    for (filter_id, _, _) in &applying {
                        skip(*filter_id, "untranslatable", &err.to_string());
                    }
                    None
                }
            }
        };
        match &filtered_plan {
            Some(plan) => {
                // Only the filters the plan reads; a filter of another scan type stays unbound.
                let bound = |stream_id: u64| {
                    plan.stream_inputs
                        .iter()
                        .any(|input| input.node_id as u64 == stream_id)
                };
                runs.retain(|run| bound(run.stream_id));
                for (filter_id, stream_id, stats) in &applying {
                    if bound(*stream_id) {
                        runtime_filters::log_applied(query, instance, *filter_id, stats, waited_ms);
                    } else {
                        skip(*filter_id, "unbound", "no scan of the plan reads its keys");
                    }
                }
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
            NixlEnvelope::Alloc(layout) => Ok((nixl.allocate(&layout)?.encode(), None)),
            NixlEnvelope::Release(token) => {
                nixl.release(token);
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
        }
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
                if let Err(err) = service.drain_ready(vec![ready]) {
                    tracing::warn!(error = %err, "a receiver fed by a remote sender failed");
                }
            });
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
                    .get(&query_id)
                    .cloned()
                    .ok_or_else(|| format!("descriptor table cache miss for query {query_id}"))?,
            );
        } else {
            cache.insert(query_id, desc.clone());
        }
        Ok(resolved)
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
    ) -> std::result::Result<Vec<ReadyFragment>, String> {
        Self::injected_failure()?;
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
            return Err(format!("output sink {:?} is not supported", sink.type_));
        }
        let stream_sink = sink.stream_sink.as_ref().ok_or_else(|| {
            "DATA_STREAM_SINK fragment carries no stream_sink payload".to_string()
        })?;
        if stream_sink.limit.is_some_and(|limit| limit >= 0) {
            return Err("data stream sink limits are not supported".to_string());
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
            return Err(err);
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
                    return Err(err);
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
    /// The outer error is the run's; the inner one is the hops'. Either way every remote claim
    /// is released.
    fn run_streaming(
        &self,
        mut run: FragmentRun<'_>,
        outputs: &[SenderSlot],
        remote: &[(SenderSlot, SocketAddr)],
        names: &[String],
    ) -> std::result::Result<std::result::Result<(), String>, String> {
        let nixl = self.nixl.as_ref().ok_or("this CN has no NIXL transport")?;
        let streams = remote
            .iter()
            .map(|(slot, _)| outputs.iter().position(|output| output == slot))
            .collect::<Option<Vec<_>>>()
            .ok_or("a remote destination is not one of the fragment's outputs")?;
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
        let shipped = match (&ran, streamed) {
            (_, Some(streamed)) => {
                for &(slot, _) in remote {
                    let _ = self.executor.drop_parked(slot);
                }
                streamed
            }
            (Ok(_), None) => return Ok(self.ship_parked(remote, names)),
            (Err(_), None) => Ok(()),
        };
        ran.map(|_| shipped)
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
        while let Some(ready) = queue.pop() {
            let query = ready
                .params
                .params
                .as_ref()
                .map(|exec| FragmentInstanceId::from(&exec.query_id));
            match self.execute_ready_fragment(ready) {
                Ok(next) => queue.extend(next),
                Err(err) => {
                    if let Some(query) = query {
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
    fn fail_and_purge(&self, query: FragmentInstanceId, error: &str) {
        self.results.fail_query(query, error);
        let deferred = self.filters.purge_query(query);
        let purged = self.exchanges.purge_query(query, error);
        if let Some(nixl) = &self.nixl {
            for &token in &purged.tokens {
                nixl.release(token);
            }
        }
        for &slot in &purged.slots {
            if let Err(err) = self.executor.drop_parked(slot) {
                warn!(?slot, error = %err, "failed to drop a purged query's parked output");
            }
        }
        info!(
            %query,
            released_buffers = purged.tokens.len(),
            dropped_parked = purged.slots.len(),
            deferred_scans = deferred,
            error,
            "purged a failed query's exchange state"
        );
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
            "leak counters"
        );
    }

    /// Translates a ready receiver against its senders' output names and runs it on their
    /// parked output.
    fn execute_ready_fragment(
        &self,
        ready: ReadyFragment,
    ) -> std::result::Result<Vec<ReadyFragment>, String> {
        let mut guard = ReadyInputsGuard {
            nixl: self.nixl.clone(),
            executor: Arc::clone(&self.executor),
            tokens: Vec::new(),
            slots: Vec::new(),
        };
        for source in ready.inputs.iter().flat_map(|input| &input.sources) {
            match source {
                SenderSource::Remote { batches, .. } => {
                    guard.tokens.extend(batches.iter().map(|batch| batch.token))
                }
                SenderSource::LocalParked { slot, .. } => guard.slots.push(*slot),
            }
        }
        let exchange_inputs = Self::exchange_inputs(&ready.inputs)?;
        let mut inputs = Vec::with_capacity(ready.inputs.len());
        let mut remote_inputs = Vec::new();
        for input in ready.inputs {
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
        let dump_seq = Self::dump_fragment(&ready.params);
        let translated =
            self.translate_fragment_logged(&ready.params, &exchange_inputs, dump_seq)?;
        let next = self.execute_fragment(
            &ready.params,
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
    ) -> std::result::Result<(), String> {
        Self::ensure_binary_protocol(protocol)?;
        let batch = Self::deserialize_binary::<TExecBatchPlanFragmentsParams>(attachment)
            .map_err(|err| format!("failed to deserialize TExecBatchPlanFragmentsParams: {err}"))?;
        let common = batch
            .common_param
            .as_ref()
            .ok_or_else(|| "TExecBatchPlanFragmentsParams.common_param is missing".to_string())?;
        let instances = batch.unique_param_per_instance.as_ref().ok_or_else(|| {
            "TExecBatchPlanFragmentsParams.unique_param_per_instance is missing".to_string()
        })?;

        if instances.is_empty() {
            return Err(
                "TExecBatchPlanFragmentsParams.unique_param_per_instance is empty".to_string(),
            );
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

            self.process_fragment(&params)
                .map_err(|err| format!("fragment {idx}: {err}"))?;
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
    fn injected_failure() -> std::result::Result<(), String> {
        match std::env::var_os("SIRIUS_CN_FAIL_ONCE_FILE") {
            Some(path) if std::fs::remove_file(&path).is_ok() => {
                Err("injected fragment failure (SIRIUS_CN_FAIL_ONCE_FILE)".to_string())
            }
            _ => Ok(()),
        }
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

    /// StarRocks INTERNAL_ERROR status carrying a user-visible error message.
    fn internal_error(message: impl Into<String>) -> StatusPb {
        StatusPb {
            status_code: TStatusCode::INTERNAL_ERROR.0,
            error_msgs: vec![message.into()],
        }
    }
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
        fragment_executor::{DrainNext, ExportedBatch, FragmentResult, OutputDrain},
        local_exchange::ExchangeCounts,
        nixl_chunk::AllocReply,
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
    }

    impl NixlEndpoint for FakeNixl {
        fn local_md(&self) -> Vec<u8> {
            b"local-md".to_vec()
        }

        fn allocate(&self, layout: &[u8]) -> Result<AllocReply, String> {
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
            for mut hop in hops {
                let mut rows = 0;
                loop {
                    match hop.drain.next(Duration::from_millis(1))? {
                        DrainNext::Batch(batch) => rows += batch.rows,
                        DrainNext::Waiting => {}
                        DrainNext::End => break,
                    }
                }
                self.streamed
                    .lock()
                    .unwrap()
                    .push((hop.peer, hop.slot, rows));
            }
            Ok(())
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
        let logs = captured_logs(|| service.run_or_register(&probing_scan(35)).unwrap());
        assert!(
            logs.contains(r#"filter_id=0 node=0 outcome="skipped" reason="not_built_on_this_cn""#),
            "{logs}"
        );
        // A partitioned join's filter: each CN holds part of the keys.
        let mut partitioned = probing_scan(36);
        let plan = partitioned
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap();
        plan.nodes[0].probe_runtime_filters.as_mut().unwrap()[0].build_join_mode =
            Some(starrocks_thrift::runtime_filter::TRuntimeFilterBuildJoinMode::PARTITIONED);
        let logs = captured_logs(|| service.run_or_register(&partitioned).unwrap());
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
            Some(starrocks_thrift::runtime_filter::TRuntimeFilterBuildJoinMode::PARTITIONED);
        let logs = captured_logs(|| service.run_or_register(&builder).unwrap());
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
        // As the FE sends it: the query id, and a dummy instance id.
        let request = PCancelPlanFragmentRequest {
            finst_id: PUniqueId { hi: 0, lo: 0 },
            cancel_reason: None,
            is_pipeline: Some(true),
            query_id: Some(PUniqueId { hi: query, lo: 0 }),
            error_message: Some("injected".to_string()),
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

        // A frame that arrives after the cancel is refused, and its buffers are freed.
        let (params, envelope) = packed(15, 2, 7);
        let (status, _) = transmit(&service, params, envelope);
        assert!(
            status.error_msgs[0].contains("already failed: query cancelled: injected"),
            "{status:?}"
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
