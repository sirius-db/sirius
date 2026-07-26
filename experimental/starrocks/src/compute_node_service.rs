use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(test)]
use crate::fragment_executor::StubExecutor;
use crate::fragment_executor::{FragmentExecutor, FragmentRun, SenderSlot};
use crate::local_exchange::{ExchangeKey, LocalExchange, ReadyFragment, SenderOutput};
use crate::proto::starrocks::{
    PExecBatchPlanFragmentsRequest, PExecBatchPlanFragmentsResult, PExecPlanFragmentRequest,
    PExecPlanFragmentResult, PFetchDataRequest, PFetchDataResult, PGetFileSchemaRequest,
    PGetFileSchemaResult, PSlotDescriptor, StatusPb, p_internal_service_brpc::PInternalService,
};
use crate::result_encoder::{self, ThriftBinary};
use crate::result_store::{FragmentInstanceId, ResultStore};
use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslatedPlan};
use starrocks_thrift::{
    data_sinks::{TDataSinkType, TResultSinkType},
    descriptors::TDescriptorTable,
    internal_service::{
        TExecBatchPlanFragmentsParams, TExecPlanFragmentParams, TGetFileSchemaRequest,
    },
    plan_nodes::TFileFormatType,
    status_code::TStatusCode,
};
use thrift::{
    protocol::{TBinaryInputProtocol, TSerializable},
    transport::TBufferChannel,
};
use tracing::{info, instrument};

/// Name of the engine view an exchange's input stream is read through.
///
/// With the engine linked this is the engine's own definition, so the name the plan reads and the
/// name the engine creates cannot drift. The no-engine build — a translation-only test path where
/// no view is ever created — mirrors the format, and `stream_view_name_matches_the_engine` pins
/// the two together whenever the engine is present.
fn sirius_stream_view_name(stream_id: u64) -> String {
    #[cfg(feature = "sirius-engine")]
    {
        sirius::stream_view_name(stream_id)
    }
    #[cfg(not(feature = "sirius-engine"))]
    {
        format!("sirius_stream_{stream_id}")
    }
}

/// Sirius compute-node implementation of StarRocks PInternalService.
///
/// Plan-fragment translation is the first implemented RPC path; future
/// compute-node tasks should land here behind the generated service facade.
#[derive(Clone, Debug)]
pub(crate) struct SiriusComputeNodeService {
    /// Reusable StarRocks thrift-to-Substrait fragment translator.
    translator: PlanTranslator,
    /// Executes a translated fragment into Arrow result batches. Production injects the GPU-backed
    /// `SiriusEngine` (via [`with_executor`](Self::with_executor)); tests use a stub.
    executor: Arc<dyn FragmentExecutor>,
    /// Buffers executed-fragment results for FE `fetch_data` collection. Shared across BRPC
    /// connections so a `fetch_data` poll sees what an `exec_plan_fragment` buffered.
    results: Arc<ResultStore>,
    /// Sequential same-node fragment exchange state, shared across BRPC connections.
    exchanges: Arc<LocalExchange>,
    /// StarRocks may send a descriptor table once per query and mark later fragments as cached.
    descriptor_tables: Arc<Mutex<HashMap<FragmentInstanceId, TDescriptorTable>>>,
}

impl SiriusComputeNodeService {
    /// Test-only constructor with the placeholder [`StubExecutor`]. Production injects a real
    /// executor via [`with_executor`](Self::with_executor).
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_executor(Arc::new(StubExecutor))
    }

    /// Builds the service with a caller-provided fragment executor (e.g. the GPU-backed
    /// `SiriusEngine`), shared across BRPC connections via the `Arc`.
    pub(crate) fn with_executor(executor: Arc<dyn FragmentExecutor>) -> Self {
        Self {
            translator: PlanTranslator::new(),
            executor,
            results: Arc::new(ResultStore::default()),
            exchanges: Arc::new(LocalExchange::default()),
            descriptor_tables: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl PInternalService for SiriusComputeNodeService {
    /// Handles a single FE-dispatched plan fragment thrift attachment. A root fragment without an
    /// exchange executes immediately; an exchange receiver waits for its local sender fragments.
    /// The final rows are buffered for `fetch_data`.
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

    /// Handles FE batch fragment dispatch: translate every per-instance fragment and execute the
    /// RESULT_SINK roots among them.
    #[instrument(skip_all)]
    async fn exec_batch_plan_fragments(
        &self,
        request: PExecBatchPlanFragmentsRequest,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PExecBatchPlanFragmentsResult>, crate::prpc::Error> {
        // Like `exec_plan_fragment`, an instance can run a RESULT_SINK fragment on the GPU, so
        // offload to a blocking worker rather than blocking the BRPC current-thread runtime.
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
        let Some(outcome) = self.results.take_next(id) else {
            return Ok(Self::fetch_data_result(
                Self::internal_error(format!("no buffered result for fragment instance {id}")),
                0,
                true,
            )
            .into());
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

    /// Processes one fragment and buffers supported RESULT_SINK rows for later `fetch_data`.
    /// Shared by single and batch dispatch; exchange receivers are registered until their local
    /// sender output is materialized.
    fn process_fragment(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<(), String> {
        let params = self.resolve_descriptor_table(params)?;
        Self::dump_fragment(&params);
        // Survey mode: accept every fragment so the FE dispatches (and we dump) the whole
        // plan even when translation fails. Queries still fail at fetch_data.
        if std::env::var_os("SIRIUS_CN_TRANSLATE_ONLY").is_some() {
            if let Err(err) = self.translate_fragment_logged(&params) {
                tracing::warn!(error = %err, "translate-only mode: accepting untranslatable fragment");
            }
            return Ok(());
        }
        let expected_senders = Self::receiver_exchanges(&params)?;
        if !expected_senders.is_empty() {
            let fragment_instance_id = Self::fragment_instance_id(&params)
                .ok_or_else(|| "exchange receiver is missing a fragment_instance_id".to_string())?;
            if Self::is_mysql_result_sink(&params)? {
                self.results.reserve(fragment_instance_id);
            }
            let ready =
                self.exchanges
                    .register_receiver(fragment_instance_id, expected_senders, params)?;
            if let Some(ready) = ready {
                self.execute_ready_fragment(ready)?;
            }
            return Ok(());
        }

        let translated = self.translate_fragment_logged(&params)?;
        self.execute_fragment(&params, translated)
    }

    /// Restores descriptor tables omitted by StarRocks's per-query descriptor cache protocol.
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

    /// Writes the received fragment params to `$SIRIUS_CN_DUMP_FRAGMENTS/fragment-<seq>.txt`
    /// (debug format) for offline plan analysis. No-op when the variable is unset.
    fn dump_fragment(params: &TExecPlanFragmentParams) {
        use std::sync::atomic::{AtomicU64, Ordering};
        let Ok(dir) = std::env::var("SIRIUS_CN_DUMP_FRAGMENTS") else {
            return;
        };
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::path::Path::new(&dir).join(format!("fragment-{seq:04}.txt"));
        if let Err(err) = std::fs::write(&path, format!("{params:#?}")) {
            tracing::warn!(error = %err, path = %path.display(), "failed to dump fragment params");
        }
    }

    /// Executes a fragment that reads no exchange input.
    fn execute_fragment(
        &self,
        params: &TExecPlanFragmentParams,
        translated: TranslatedPlan,
    ) -> std::result::Result<(), String> {
        self.execute_fragment_with_inputs(params, translated, Vec::new())
    }

    /// Executes a result fragment, or runs a data-stream sender and parks its output on the GPU
    /// for its local receiver. `inputs` names the parked sender outputs this fragment consumes.
    fn execute_fragment_with_inputs(
        &self,
        params: &TExecPlanFragmentParams,
        translated: TranslatedPlan,
        inputs: Vec<(i32, Vec<SenderSlot>)>,
    ) -> std::result::Result<(), String> {
        if Self::is_mysql_result_sink(params)? {
            let id = Self::fragment_instance_id(params).ok_or_else(|| {
                "RESULT_SINK fragment is missing a fragment_instance_id".to_string()
            })?;
            let result = self
                .executor
                .run(FragmentRun {
                    plan: &translated,
                    inputs: inputs.clone(),
                    output: None,
                })?
                .ok_or_else(|| "result fragment returned no rows".to_string())?;
            let batch = result_encoder::MysqlResultEncoder::encode(&result.batches, 0)?;
            self.results.insert(id, batch);
            return Ok(());
        }

        let Some(stream_sink) = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
            .filter(|sink| sink.type_ == TDataSinkType::DATA_STREAM_SINK)
            .and_then(|sink| sink.stream_sink.as_ref())
        else {
            return Ok(());
        };
        if stream_sink.limit.is_some_and(|limit| limit >= 0) {
            return Err("data stream sink limits are not supported".to_string());
        }
        if let Some(columns) = stream_sink
            .output_columns
            .as_ref()
            .filter(|columns| !columns.is_empty())
            && columns
                .iter()
                .copied()
                .ne(0..translated.output_names.len() as i32)
        {
            return Err(
                "non-identity data stream sink output_columns are not supported".to_string(),
            );
        }
        let exec = params
            .params
            .as_ref()
            .ok_or_else(|| "DATA_STREAM_SINK fragment is missing execution params".to_string())?;
        let destinations = exec
            .destinations
            .as_ref()
            .filter(|destinations| !destinations.is_empty())
            .ok_or_else(|| "DATA_STREAM_SINK fragment has no destinations".to_string())?;
        let sender_id = exec.sender_id.unwrap_or(0);
        // One destination only: relaying a stream to N receivers means each batch is claimed by
        // one of them, so a broadcast needs the partitioned sink (#838), not a loop here. Refusing
        // is the honest answer — the alternative silently gives every receiver but one no rows.
        if destinations.len() > 1 {
            return Err(format!(
                "a data stream sink with {} destinations needs partitioned streaming",
                destinations.len()
            ));
        }
        let destination = &destinations[0];
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from(&destination.fragment_instance_id),
            node_id: stream_sink.dest_node_id,
            sender_id,
        };

        // The sender's rows stay on the GPU, parked under `slot` as native batches; only the
        // rendezvous bookkeeping — which sender produced, and its output names — comes back here.
        self.executor.run(FragmentRun {
            plan: &translated,
            inputs,
            output: Some(slot),
        })?;

        let ready = self.exchanges.push_sender(
            ExchangeKey {
                fragment_instance_id: slot.fragment_instance_id,
                node_id: slot.node_id,
            },
            sender_id,
            SenderOutput {
                names: translated.output_names,
                slot,
            },
        )?;
        if let Some(ready) = ready {
            self.execute_ready_fragment(ready)?;
        }
        Ok(())
    }

    /// Translates a receiver whose sender set is complete, binding each exchange to the input
    /// stream its senders parked into, and runs it.
    fn execute_ready_fragment(&self, ready: ReadyFragment) -> std::result::Result<(), String> {
        let exchange_inputs = ready
            .inputs
            .iter()
            .map(|input| {
                let names = input
                    .outputs
                    .first()
                    .map(|output| output.names.clone())
                    .ok_or_else(|| {
                        format!("exchange node {} has no sender output", input.node_id)
                    })?;
                if input.outputs.iter().any(|output| output.names != names) {
                    return Err("exchange senders produced different output names".to_string());
                }
                let stream_id = u64::try_from(input.node_id)
                    .map_err(|_| format!("negative exchange node id {}", input.node_id))?;
                Ok(ExchangeInput {
                    node_id: input.node_id,
                    stream_view: sirius_stream_view_name(stream_id),
                    names,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let inputs = ready
            .inputs
            .iter()
            .map(|input| {
                (
                    input.node_id,
                    input.outputs.iter().map(|output| output.slot).collect(),
                )
            })
            .collect();
        let translated =
            self.translate_fragment_logged_with_inputs(&ready.params, &exchange_inputs)?;
        self.execute_fragment_with_inputs(&ready.params, translated, inputs)
    }

    /// Finds every exchange receiver in a fragment and each expected sender count.
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
                    .filter(|node| {
                        node.node_type == starrocks_thrift::plan_nodes::TPlanNodeType::EXCHANGE_NODE
                    })
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

    /// Converts a StarRocks thrift plan fragment to Substrait, logs substrait-explain output, and
    /// returns the translated plan for execution.
    #[instrument(skip_all)]
    fn translate_fragment_logged(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<TranslatedPlan, String> {
        self.translate_fragment_logged_with_inputs(params, &[])
    }

    fn translate_fragment_logged_with_inputs(
        &self,
        params: &TExecPlanFragmentParams,
        exchange_inputs: &[ExchangeInput],
    ) -> std::result::Result<TranslatedPlan, String> {
        let translated = self
            .translator
            .translate_fragment_with_exchange_inputs(params, exchange_inputs)
            .map_err(|err| err.to_string())?;
        info!(
            output_names = ?translated.output_names,
            plan = %translated.explain(),
            "translated StarRocks plan fragment"
        );
        Self::dump_substrait(&translated);
        Ok(translated)
    }

    /// Writes the translated Substrait plan bytes to `$SIRIUS_CN_DUMP_FRAGMENTS/plan-<seq>.substrait`
    /// so a failing plan can be replayed against the engine in isolation. No-op when unset.
    fn dump_substrait(translated: &TranslatedPlan) {
        use std::sync::atomic::{AtomicU64, Ordering};
        let Ok(dir) = std::env::var("SIRIUS_CN_DUMP_FRAGMENTS") else {
            return;
        };
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::path::Path::new(&dir).join(format!("plan-{seq:04}.substrait"));
        if let Err(err) = std::fs::write(&path, translated.to_substrait_bytes()) {
            tracing::warn!(error = %err, path = %path.display(), "failed to dump substrait plan");
        }
    }

    /// Classifies the fragment output sink: `Ok(true)` for a MySQL text-protocol RESULT_SINK this
    /// CN can encode, `Ok(false)` for a non-result sink, and `Err` for a
    /// RESULT_SINK whose format is not supported yet (binary rows, HTTP/FILE/Arrow Flight, etc.).
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
        // Cross-file sampling and type promotion (what the native scanner does for multi-file
        // FILES()) is a follow-up; reject multiple ranges rather than resolve a partial schema.
        let ranges = broker.ranges;
        if ranges.len() > 1 {
            return Err(format!(
                "multi-file FILES() schema inference is not supported yet ({} files); use a single file path",
                ranges.len()
            ));
        }
        let range = ranges
            .into_iter()
            .next()
            .ok_or_else(|| "broker_scan_range carries no file ranges".to_string())?;
        if range.format_type != TFileFormatType::FORMAT_PARQUET {
            return Err(format!(
                "unsupported file format {:?}; only parquet schema inference is implemented",
                range.format_type
            ));
        }
        crate::file_schema::parquet_file_schema(&range.path).await
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

    /// StarRocks OK status. For these RPCs OK means "fragment accepted and translated", not
    /// "fragment executed" — execution and result delivery are not implemented yet.
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use prost::Message;
    use starrocks_thrift::{
        data::TResultBatch,
        data_sinks::{TDataSink, TDataStreamSink, TPlanFragmentDestination, TResultSink},
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
        fragment_executor::{FragmentResult, StubExecutor},
        proto::starrocks::{
            PFetchDataRequest, PUniqueId,
            p_internal_service_brpc::{PInternalServiceRouter, SERVICE_NAME, methods},
        },
        prpc,
    };

    #[derive(Debug, Default)]
    struct CountingExecutor {
        calls: AtomicUsize,
    }

    impl FragmentExecutor for CountingExecutor {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            StubExecutor.run(run)
        }
    }

    /// The plan names the view; the engine creates it. If the two ever disagree the receiver's
    /// read binds to nothing and the query fails at plan time, so pin them together here rather
    /// than discovering it in a cluster run.
    #[cfg(feature = "sirius-engine")]
    #[test]
    fn stream_view_name_matches_the_engine() {
        for stream_id in [0, 1, 7, 4096] {
            assert_eq!(
                sirius_stream_view_name(stream_id),
                sirius::stream_view_name(stream_id)
            );
        }
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
    fn self_exchange_executes_sender_then_receiver_when_receiver_arrives_first() {
        let executor = Arc::new(CountingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone());
        let query_id = TUniqueId::new(10, 1);
        let receiver_id = TUniqueId::new(10, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id.clone(), receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 1);
        receiver.params = Some(receiver_exec);

        let receiver_response = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&receiver),
        );
        let receiver_response =
            PExecPlanFragmentResult::decode(receiver_response.body.as_slice()).unwrap();
        assert_eq!(receiver_response.status.status_code, TStatusCode::OK.0);
        assert_eq!(executor.calls.load(Ordering::Relaxed), 0);

        let waiting = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(receiver_id.hi, receiver_id.lo),
            Vec::new(),
        );
        let waiting = PFetchDataResult::decode(waiting.body.as_slice()).unwrap();
        assert_eq!(waiting.status.status_code, TStatusCode::OK.0);
        assert_eq!(waiting.eos, Some(false));

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(query_id, TUniqueId::new(10, 3));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            receiver_id.clone(),
            None,
            None,
            None,
        )]);
        sender.params = Some(sender_exec);

        let sender_response = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&sender),
        );
        let sender_response =
            PExecPlanFragmentResult::decode(sender_response.body.as_slice()).unwrap();
        assert_eq!(sender_response.status.status_code, TStatusCode::OK.0);
        assert_eq!(executor.calls.load(Ordering::Relaxed), 2);

        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(receiver_id.hi, receiver_id.lo),
            Vec::new(),
        );
        let fetched_result = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(fetched_result.eos, Some(false));
        assert!(!fetched.attachment.is_empty());
    }

    #[test]
    fn self_exchange_executes_an_intermediate_receiver_and_reuses_cached_descriptors() {
        let executor = Arc::new(CountingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone());
        let query_id = TUniqueId::new(20, 1);
        let root_id = TUniqueId::new(20, 2);
        let middle_id = TUniqueId::new(20, 3);

        let mut root = fragment_params(Some(exchange_plan(9, 0)), Some(desc_table()));
        root.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut root_exec = exec_params(query_id.clone(), root_id.clone());
        root_exec.per_exch_num_senders.insert(9, 1);
        root.params = Some(root_exec);
        assert_exec_ok(&service, &root);

        let cached_desc = TDescriptorTable::new(None, Vec::new(), None, Some(true));
        let mut middle = fragment_params(Some(exchange_plan(7, 0)), Some(cached_desc.clone()));
        middle.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(9));
        let mut middle_exec = exec_params(query_id.clone(), middle_id.clone());
        middle_exec.per_exch_num_senders.insert(7, 1);
        middle_exec.sender_id = Some(0);
        middle_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            root_id.clone(),
            None,
            None,
            None,
        )]);
        middle.params = Some(middle_exec);
        assert_exec_ok(&service, &middle);

        let mut leaf = fragment_params(Some(scan_plan(0, 0)), Some(cached_desc));
        leaf.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut leaf_exec = exec_params(query_id, TUniqueId::new(20, 4));
        leaf_exec.sender_id = Some(0);
        leaf_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            middle_id, None, None, None,
        )]);
        leaf.params = Some(leaf_exec);
        assert_exec_ok(&service, &leaf);

        assert_eq!(executor.calls.load(Ordering::Relaxed), 3);
        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(root_id.hi, root_id.lo),
            Vec::new(),
        );
        let fetched_result = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(fetched_result.eos, Some(false));
        assert!(!fetched.attachment.is_empty());
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

    fn assert_exec_ok(service: &SiriusComputeNodeService, params: &TExecPlanFragmentParams) {
        let response = route(
            service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(params),
        );
        let result = PExecPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(
            result.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            result.status.error_msgs
        );
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

    fn data_stream_sink(dest_node_id: i32) -> TDataSink {
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

    fn exchange_plan(node_id: i32, tuple_id: i32) -> TPlan {
        let mut exchange = scan_node(node_id, tuple_id);
        exchange.node_type = TPlanNodeType::EXCHANGE_NODE;
        exchange.file_scan_node = None;
        exchange.exchange_node = Some(TExchangeNode::new(
            vec![tuple_id],
            None,
            None,
            Some(TPartitionType::UNPARTITIONED),
            Some(true),
            None,
        ));
        TPlan::new(vec![exchange])
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
