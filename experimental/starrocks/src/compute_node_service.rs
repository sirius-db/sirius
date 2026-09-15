use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(test)]
use crate::fragment_executor::StubExecutor;
use crate::fragment_executor::{FragmentExecutor, FragmentRun, SenderSlot, StagedBatch};
use crate::local_exchange::{
    ExchangeKey, LocalExchange, ReadyExchangeInput, ReadyFragment, SenderSource,
};
use crate::nixl_chunk::{NixlMdHandler, StagingLeaseHandler};
use crate::nixl_transport::{NixlTransport, RemoteSendSpec};
use crate::proto::starrocks::{
    PExecBatchPlanFragmentsRequest, PExecBatchPlanFragmentsResult, PExecPlanFragmentRequest,
    PExecPlanFragmentResult, PFetchDataRequest, PFetchDataResult, PGetFileSchemaRequest,
    PGetFileSchemaResult, PSlotDescriptor, PTransmitChunkParams, PTransmitChunkResult, StatusPb,
    p_internal_service_brpc::PInternalService,
};
use crate::result_encoder::{self, ThriftBinary};
use crate::result_store::{FetchOutcome, FragmentInstanceId, ResultStore};
use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslatedPlan};
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

/// Hostname AND port equality against this CN's advertised brpc endpoint — two CNs on one host
/// with different brpc ports see each other as remote.
#[derive(Clone, Debug)]
pub struct ExchangeIdentity {
    pub host: String,
    pub brpc_port: u16,
    pub http_port: u16,
}

impl Default for ExchangeIdentity {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            brpc_port: 8060,
            http_port: 8040,
        }
    }
}

impl ExchangeIdentity {
    fn matches(&self, addr: &TNetworkAddress) -> bool {
        addr.hostname == self.host && addr.port == i32::from(self.brpc_port)
    }
}

/// Where a data-stream sink destination lives relative to this CN.
#[derive(Debug)]
enum DestinationRoute {
    Local,
    Remote { host: String, brpc_port: u16 },
}

/// Sirius compute-node implementation of StarRocks PInternalService.
///
/// Plan-fragment translation is the first implemented RPC path; future
/// compute-node tasks should land here behind the generated service facade.
#[derive(Clone, Debug)]
pub struct SiriusComputeNodeService {
    /// Reusable StarRocks thrift-to-Substrait fragment translator.
    translator: PlanTranslator,
    /// Executes a translated fragment into Arrow result batches. Production injects the GPU-backed
    /// `SiriusEngine` (via [`with_executor`](Self::with_executor)); tests use a stub.
    executor: Arc<dyn FragmentExecutor>,
    /// Buffers executed-fragment results for FE `fetch_data` collection. Shared across BRPC
    /// connections so a `fetch_data` poll sees what an `exec_plan_fragment` buffered.
    results: Arc<ResultStore>,
    /// Descriptor tables retained for StarRocks's per-query cache protocol.
    descriptor_tables: Arc<Mutex<HashMap<FragmentInstanceId, TDescriptorTable>>>,
    /// Sequential exchange rendezvous: receivers wait, senders park or hop packed GPU bytes.
    exchanges: Arc<LocalExchange>,
    /// Advertised brpc identity used to classify destinations as local vs remote.
    identity: ExchangeIdentity,
    /// NIXL transport for a remote hop. Absent when this CN was built or started without it;
    /// a remote destination then fails loudly instead of silently skipping the hop.
    transport: Option<Arc<NixlTransport>>,
    /// Optional Md handler for `transmit_chunk` kind Md. Production falls back to `transport`.
    md: Option<Arc<dyn NixlMdHandler>>,
    /// Optional lease handler for `transmit_chunk` kinds Lease and CanaryRelease. Production
    /// falls back to the fragment executor's staging arena.
    leases: Option<Arc<dyn StagingLeaseHandler>>,
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
    pub fn with_executor(executor: Arc<dyn FragmentExecutor>) -> Self {
        Self::with_executor_and_exchange(
            executor,
            Arc::new(LocalExchange::default()),
            ExchangeIdentity::default(),
            None,
        )
    }

    /// Builds the service with a shared exchange rendezvous, advertised identity, and optional
    /// NIXL transport.
    pub fn with_executor_and_exchange(
        executor: Arc<dyn FragmentExecutor>,
        exchanges: Arc<LocalExchange>,
        identity: ExchangeIdentity,
        transport: Option<Arc<NixlTransport>>,
    ) -> Self {
        Self {
            translator: PlanTranslator::new(),
            executor,
            results: Arc::new(ResultStore::default()),
            descriptor_tables: Arc::new(Mutex::new(HashMap::new())),
            exchanges,
            identity,
            transport,
            md: None,
            leases: None,
        }
    }

    /// Injects Md / Lease handlers used by `transmit_chunk` (tests with a stub arena; production
    /// shares the transport and executor already held by this service).
    pub fn with_nixl_control(
        mut self,
        md: Option<Arc<dyn NixlMdHandler>>,
        leases: Option<Arc<dyn StagingLeaseHandler>>,
    ) -> Self {
        self.md = md;
        self.leases = leases;
        self
    }

    fn md_handler(&self) -> Result<Arc<dyn NixlMdHandler>, String> {
        if let Some(md) = &self.md {
            return Ok(Arc::clone(md));
        }
        #[cfg(feature = "nixl-transport")]
        if let Some(transport) = &self.transport {
            return Ok(Arc::clone(transport) as Arc<dyn NixlMdHandler>);
        }
        Err("this CN has no nixl metadata handler".to_string())
    }

    fn lease_handler(&self) -> Arc<dyn StagingLeaseHandler> {
        match &self.leases {
            Some(leases) => Arc::clone(leases),
            None => Arc::new(Arc::clone(&self.executor)),
        }
    }

    /// Decodes an SRNX `transmit_chunk` attachment. Md/Lease/CanaryRelease never call
    /// [`LocalExchange::push_remote_frame`]; Packed/EOS does.
    fn handle_nixl_chunk(
        &self,
        request: &PTransmitChunkParams,
        attachment: &[u8],
    ) -> std::result::Result<(Vec<u8>, Option<ReadyFragment>), String> {
        crate::nixl_chunk::reject_native_chunk(request)?;
        match crate::nixl_chunk::NixlEnvelope::decode(attachment)? {
            crate::nixl_chunk::NixlEnvelope::Md(peer_md) => {
                Ok((self.md_handler()?.on_peer_md(&peer_md)?, None))
            }
            crate::nixl_chunk::NixlEnvelope::Lease { length } => {
                let lease = self.lease_handler().lease(length)?;
                Ok((
                    crate::nixl_chunk::encode_lease_reply(lease.remote_addr, lease.offset),
                    None,
                ))
            }
            crate::nixl_chunk::NixlEnvelope::CanaryRelease { offset } => {
                self.lease_handler().release(offset)?;
                Ok((Vec::new(), None))
            }
            crate::nixl_chunk::NixlEnvelope::Packed {
                offset,
                length,
                rows,
                names,
                metadata,
            } => {
                let ready = self.ingest_packed(packed_frame_from_request(
                    request, offset, length, rows, names, metadata,
                )?)?;
                Ok((Vec::new(), ready))
            }
        }
    }

    fn ingest_packed(
        &self,
        frame: crate::nixl_chunk::PackedExchangeFrame,
    ) -> std::result::Result<Option<ReadyFragment>, String> {
        self.exchanges.push_remote_frame(
            ExchangeKey {
                fragment_instance_id: frame.fragment_instance_id,
                node_id: frame.dest_stream,
            },
            frame.sender_id,
            frame.seq,
            frame.eos,
            frame.names.clone(),
            frame.staged_batch(),
        )
    }

    /// Runs a receiver whose sender set just completed, on a helper thread so the
    /// `transmit_chunk` handler can return status without waiting for GPU work.
    pub(crate) fn dispatch_ready_async(&self, ready: ReadyFragment) {
        let service = self.clone();
        let _ = std::thread::Builder::new()
            .name("packed-exchange-ready".to_string())
            .spawn(move || {
                if let Err(err) = service.drain_ready(vec![ready]) {
                    warn!(error = %err, "ready receiver from packed hop failed");
                }
            });
    }
}

impl PInternalService for SiriusComputeNodeService {
    /// Handles a single FE-dispatched plan fragment thrift attachment: translate it, and for a
    /// root RESULT_SINK fragment execute it and buffer the rows for `fetch_data`. An OK status
    /// means the fragment was accepted (and, for a result fragment, executed and buffered).
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
        // Exchange receivers reserve a Waiting slot at dispatch; long-poll until rows or failure
        // rather than replying not-ready (that would desync the FE packet counter).
        let results = self.results.clone();
        let outcome =
            tokio::task::spawn_blocking(move || results.wait_ready(id, Duration::from_secs(600)))
                .await;
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(join_err) => {
                return Ok(Self::fetch_data_result(
                    Self::internal_error(format!("fetch_data wait task panicked: {join_err}")),
                    0,
                    true,
                )
                .into());
            }
        };
        let Some(outcome) = outcome else {
            return Ok(Self::fetch_data_result(
                Self::internal_error(format!("no buffered result for fragment instance {id}")),
                0,
                true,
            )
            .into());
        };
        match outcome {
            FetchOutcome::Failed(cause) => {
                Ok(Self::fetch_data_result(Self::internal_error(cause), 0, true).into())
            }
            FetchOutcome::Rows {
                batch: Some(batch),
                packet_seq,
                eos,
            } => match batch.to_binary() {
                Ok(bytes) => Ok(crate::prpc::Reply::with_attachment(
                    Self::fetch_data_result(Self::ok_status(), packet_seq, eos),
                    bytes,
                )),
                Err(err) => {
                    Ok(Self::fetch_data_result(Self::internal_error(err), packet_seq, true).into())
                }
            },
            FetchOutcome::Rows {
                batch: None,
                packet_seq,
                eos,
            } => Ok(Self::fetch_data_result(Self::ok_status(), packet_seq, eos).into()),
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

    /// Serves NIXL control and packed-frame announce on unpatched `transmit_chunk`.
    #[instrument(skip_all)]
    async fn transmit_chunk(
        &self,
        request: PTransmitChunkParams,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PTransmitChunkResult>, crate::prpc::Error> {
        match self.handle_nixl_chunk(&request, &attachment) {
            Ok((reply_attachment, ready)) => {
                if let Some(ready) = ready {
                    self.dispatch_ready_async(ready);
                }
                Ok(crate::prpc::Reply::with_attachment(
                    Self::transmit_chunk_result(Self::ok_status()),
                    reply_attachment,
                ))
            }
            Err(err) => Ok(Self::transmit_chunk_result(Self::internal_error(err)).into()),
        }
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

    /// Translates one fragment and, when it is a supported RESULT_SINK root, executes it and
    /// buffers the rows for later `fetch_data`. Shared by single and batch dispatch so both paths
    /// produce fetchable results for a RESULT_SINK instance.
    fn process_fragment(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<(), String> {
        let params = self.resolve_descriptor_table(params)?;
        let dump_seq = Self::dump_fragment(&params);
        // Survey mode: accept every fragment so the FE dispatches (and we dump) the whole
        // plan even when translation fails. Queries still fail at fetch_data.
        if std::env::var_os("SIRIUS_CN_TRANSLATE_ONLY").is_some() {
            if let Err(err) = self.translate_fragment_logged(&params, dump_seq) {
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
            return self.drain_ready(ready.into_iter().collect());
        }
        let translated = self.translate_fragment_logged(&params, dump_seq)?;
        self.execute_fragment(&params, translated, Vec::new(), Vec::new())
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

    /// Executes a RESULT_SINK fragment and buffers its rows, or runs a DATA_STREAM_SINK sender
    /// and parks / hops its output. `inputs` are same-CN parked slots; `remote_inputs` are packed
    /// batches already on this CN.
    fn execute_fragment(
        &self,
        params: &TExecPlanFragmentParams,
        translated: TranslatedPlan,
        inputs: Vec<(i32, Vec<SenderSlot>)>,
        remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
    ) -> std::result::Result<(), String> {
        if Self::is_mysql_result_sink(params)? {
            let id = Self::fragment_instance_id(params).ok_or_else(|| {
                "RESULT_SINK fragment is missing a fragment_instance_id".to_string()
            })?;
            let result = self
                .executor
                .run_fragment(FragmentRun {
                    plan: &translated,
                    inputs,
                    remote_inputs,
                    outputs: Vec::new(),
                    broadcast: false,
                    hash_keys: Vec::new(),
                })?
                .ok_or_else(|| "result fragment returned no rows".to_string())?;
            let batch = result_encoder::MysqlResultEncoder::encode(&result.batches, 0)?;
            self.results.insert(id, batch);
            return Ok(());
        }

        let Some(sink) = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
        else {
            tracing::warn!("fragment carries no output sink; nothing consumes its output");
            return Ok(());
        };
        let exec = params
            .params
            .as_ref()
            .ok_or_else(|| "DATA_STREAM_SINK fragment is missing execution params".to_string())?;
        let sender_id = exec.sender_id.unwrap_or(0);
        if sink.type_ != TDataSinkType::DATA_STREAM_SINK {
            return Err(format!("output sink {:?} is not supported", sink.type_));
        }
        let stream_sink = sink.stream_sink.as_ref().ok_or_else(|| {
            "DATA_STREAM_SINK fragment carries no stream_sink payload".to_string()
        })?;
        if stream_sink.limit.is_some_and(|limit| limit >= 0) {
            return Err("data stream sink limits are not supported".to_string());
        }
        let destinations = exec
            .destinations
            .as_ref()
            .filter(|destinations| !destinations.is_empty())
            .ok_or_else(|| "DATA_STREAM_SINK fragment has no destinations".to_string())?;

        if destinations.len() > 1 {
            match stream_sink.output_partition.type_ {
                TPartitionType::UNPARTITIONED => {}
                TPartitionType::HASH_PARTITIONED => {
                    if translated.output_partition_columns.is_none() {
                        return Err(
                            "a hash-partitioned data stream sink translated without partition \
                             key columns"
                                .to_string(),
                        );
                    }
                }
                other => {
                    return Err(format!(
                        "a data stream sink with {} destinations carries partition type {:?}, \
                         which this CN does not support",
                        destinations.len(),
                        other
                    ));
                }
            }
        }
        let hash_keys = if destinations.len() > 1 {
            translated
                .output_partition_columns
                .clone()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let broadcast = destinations.len() > 1 && hash_keys.is_empty();

        let mut slots: Vec<SenderSlot> = Vec::with_capacity(destinations.len());
        let mut routes = Vec::with_capacity(destinations.len());
        for destination in destinations {
            let slot = SenderSlot {
                fragment_instance_id: FragmentInstanceId::from(&destination.fragment_instance_id),
                node_id: stream_sink.dest_node_id,
                sender_id,
            };
            if slots.contains(&slot) {
                return Err(format!(
                    "duplicate destination {slot:?} in one data stream sink; two claims would \
                     race over one output stream"
                ));
            }
            let route = self.route_destination(destination)?;
            if matches!(route, DestinationRoute::Remote { .. }) && self.transport.is_none() {
                return Err(format!(
                    "DATA_STREAM_SINK destination for fragment instance {} is remote, but this \
                     CN has no nixl transport",
                    slot.fragment_instance_id
                ));
            }
            slots.push(slot);
            routes.push(route);
        }

        self.executor.run_fragment(FragmentRun {
            plan: &translated,
            inputs,
            remote_inputs,
            outputs: slots.clone(),
            broadcast,
            hash_keys,
        })?;

        let mut ready_receivers = Vec::new();
        for (slot, route) in slots.iter().zip(&routes) {
            if matches!(route, DestinationRoute::Local) {
                let ready = self.exchanges.push_sender(
                    ExchangeKey {
                        fragment_instance_id: slot.fragment_instance_id,
                        node_id: slot.node_id,
                    },
                    sender_id,
                    SenderSource::LocalParked {
                        names: translated.output_names.clone(),
                        slot: *slot,
                    },
                )?;
                ready_receivers.extend(ready);
            }
        }
        for (slot, route) in slots.iter().zip(&routes) {
            if let DestinationRoute::Remote { host, brpc_port } = route {
                self.ship_remote(*slot, &translated.output_names, host, *brpc_port)?;
            }
        }
        self.drain_ready(ready_receivers)
    }

    fn route_destination(
        &self,
        destination: &TPlanFragmentDestination,
    ) -> std::result::Result<DestinationRoute, String> {
        let brpc_server = destination.brpc_server.as_ref().ok_or_else(|| {
            format!(
                "DATA_STREAM_SINK destination for fragment instance {} has no brpc_server address",
                FragmentInstanceId::from(&destination.fragment_instance_id)
            )
        })?;
        if self.identity.matches(brpc_server) {
            Ok(DestinationRoute::Local)
        } else {
            Ok(DestinationRoute::Remote {
                host: brpc_server.hostname.clone(),
                brpc_port: u16::try_from(brpc_server.port).map_err(|_| {
                    format!(
                        "destination brpc port {} is not a valid TCP port",
                        brpc_server.port
                    )
                })?,
            })
        }
    }

    fn ship_remote(
        &self,
        slot: SenderSlot,
        names: &[String],
        host: &str,
        brpc_port: u16,
    ) -> std::result::Result<(), String> {
        let transport = self.transport.as_ref().ok_or_else(|| {
            format!(
                "DATA_STREAM_SINK destination {host}:{brpc_port} is remote, but this CN has no \
                 nixl transport"
            )
        })?;
        let peer = lookup_peer(host, brpc_port)?;
        transport.send_fragment(
            RemoteSendSpec {
                peer,
                peer_agent_name: format!("{host}:{brpc_port}"),
                dest_stream: slot.node_id,
                sender_id: slot.sender_id,
                names: names.to_vec(),
                slot,
            },
            Arc::clone(&self.executor),
        )
    }

    fn drain_ready(&self, ready: Vec<ReadyFragment>) -> std::result::Result<(), String> {
        let mut queue = ready;
        while let Some(fragment) = queue.pop() {
            let result_id = Self::fragment_instance_id(&fragment.params)
                .filter(|_| matches!(Self::is_mysql_result_sink(&fragment.params), Ok(true)));
            match self.execute_ready_fragment(fragment) {
                Ok(next) => queue.extend(next),
                Err(err) => {
                    if let Some(id) = result_id {
                        self.results.fail(id, err.clone());
                    }
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    fn execute_ready_fragment(
        &self,
        ready: ReadyFragment,
    ) -> std::result::Result<Vec<ReadyFragment>, String> {
        let exchange_inputs = Self::exchange_inputs(&ready.inputs)?;
        let mut inputs: Vec<(i32, Vec<SenderSlot>)> = Vec::new();
        let mut remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)> = Vec::new();
        for input in ready.inputs {
            let mut slots = Vec::new();
            for source in input.sources {
                match source {
                    SenderSource::LocalParked { slot, .. } => slots.push(slot),
                    SenderSource::Remote {
                        sender_id,
                        batches,
                        closed,
                        ..
                    } => {
                        if !closed {
                            return Err(format!(
                                "exchange node {} became ready with remote sender {sender_id} \
                                 still open",
                                input.node_id
                            ));
                        }
                        remote_inputs.push((input.node_id, sender_id, batches));
                    }
                }
            }
            if !slots.is_empty() {
                inputs.push((input.node_id, slots));
            }
        }
        let dump_seq = Self::dump_fragment(&ready.params);
        let translated =
            self.translate_fragment_logged_with_inputs(&ready.params, &exchange_inputs, dump_seq)?;
        self.execute_fragment(&ready.params, translated, inputs, remote_inputs)?;
        Ok(Vec::new())
    }

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

    /// Converts a StarRocks thrift plan fragment to Substrait, logs substrait-explain output, and
    /// returns the translated plan for execution.
    #[instrument(skip_all)]
    fn translate_fragment_logged(
        &self,
        params: &TExecPlanFragmentParams,
        dump_seq: Option<u64>,
    ) -> std::result::Result<TranslatedPlan, String> {
        let translated = self
            .translator
            .translate_fragment(params)
            .map_err(|err| err.to_string())?;
        info!(
            output_names = ?translated.output_names,
            plan = %translated.explain(),
            "translated StarRocks plan fragment"
        );
        Self::dump_substrait(&translated, dump_seq);
        Ok(translated)
    }

    fn translate_fragment_logged_with_inputs(
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
            plan = %translated.explain(),
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
    /// CN can encode, `Ok(false)` for a non-result sink (translate-only), and `Err` for a
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

    /// StarRocks OK status. For these RPCs OK means "fragment accepted and translated", not
    /// "fragment executed" — execution and result delivery are not implemented yet.
    fn ok_status() -> StatusPb {
        StatusPb {
            status_code: TStatusCode::OK.0,
            error_msgs: Vec::new(),
        }
    }

    fn transmit_chunk_result(status: StatusPb) -> PTransmitChunkResult {
        PTransmitChunkResult {
            status: Some(status),
            receive_timestamp: None,
            receiver_post_process_time: None,
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

fn lookup_peer(host: &str, port: u16) -> Result<SocketAddr, String> {
    (host, port)
        .to_socket_addrs()
        .map_err(|err| format!("failed to resolve exchange peer {host}:{port}: {err}"))?
        .next()
        .ok_or_else(|| format!("exchange peer {host}:{port} resolved to no addresses"))
}

fn packed_frame_from_request(
    request: &PTransmitChunkParams,
    offset: u64,
    length: u64,
    rows: u64,
    names: Vec<String>,
    metadata: Vec<u8>,
) -> Result<crate::nixl_chunk::PackedExchangeFrame, String> {
    let finst = request
        .finst_id
        .as_ref()
        .ok_or_else(|| "packed transmit_chunk is missing finst_id".to_string())?;
    Ok(crate::nixl_chunk::PackedExchangeFrame {
        fragment_instance_id: FragmentInstanceId::from(finst),
        dest_stream: request
            .node_id
            .ok_or_else(|| "packed transmit_chunk is missing node_id".to_string())?,
        sender_id: request
            .sender_id
            .ok_or_else(|| "packed transmit_chunk is missing sender_id".to_string())?,
        seq: request
            .sequence
            .ok_or_else(|| "packed transmit_chunk is missing sequence".to_string())?,
        eos: request
            .eos
            .ok_or_else(|| "packed transmit_chunk is missing eos".to_string())?,
        names,
        offset,
        length,
        rows: if length == 0 && metadata.is_empty() {
            None
        } else {
            Some(rows)
        },
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use prost::Message;
    use starrocks_thrift::{
        data::TResultBatch,
        data_sinks::{
            TDataSink, TDataSinkType, TDataStreamSink, TPlanFragmentDestination, TResultSink,
        },
        descriptors::{TDescriptorTable, TSlotDescriptor, TTableDescriptor, TTupleDescriptor},
        internal_service::{InternalServiceVersion, TPlanFragmentExecParams},
        partitions::{TDataPartition, TPartitionType},
        plan_nodes::{TExchangeNode, TFileScanNode, TPlan, TPlanNode, TPlanNodeType},
        planner::TPlanFragment,
        types::{
            TNetworkAddress, TPrimitiveType, TScalarType, TTableType, TTypeDesc, TTypeNode,
            TTypeNodeType, TUniqueId,
        },
    };
    use thrift::{protocol::TBinaryOutputProtocol, transport::TIoChannel};
    use tower::{Service, ServiceExt};

    use super::*;
    use crate::{
        proto::starrocks::{
            PFetchDataRequest, PUniqueId,
            p_internal_service_brpc::{PInternalServiceRouter, SERVICE_NAME, methods},
        },
        prpc,
    };

    #[derive(Debug)]
    struct FakeMd {
        mine: Vec<u8>,
        loaded: std::sync::Mutex<Vec<Vec<u8>>>,
    }

    impl NixlMdHandler for FakeMd {
        fn on_peer_md(&self, peer_metadata: &[u8]) -> Result<Vec<u8>, String> {
            self.loaded.lock().unwrap().push(peer_metadata.to_vec());
            Ok(self.mine.clone())
        }
    }

    #[derive(Debug)]
    struct FakeLeases {
        base: u64,
        next: std::sync::Mutex<u64>,
        released: std::sync::Mutex<Vec<u64>>,
    }

    impl StagingLeaseHandler for FakeLeases {
        fn lease(&self, length: u64) -> Result<crate::nixl_chunk::RemoteLease, String> {
            let mut next = self.next.lock().unwrap();
            let offset = *next;
            *next += length;
            Ok(crate::nixl_chunk::RemoteLease {
                remote_addr: self.base + offset,
                offset,
            })
        }

        fn release(&self, offset: u64) -> Result<(), String> {
            self.released.lock().unwrap().push(offset);
            Ok(())
        }
    }

    fn nixl_control_service() -> (
        SiriusComputeNodeService,
        Arc<FakeMd>,
        Arc<FakeLeases>,
        Arc<LocalExchange>,
    ) {
        let md = Arc::new(FakeMd {
            mine: b"local-md".to_vec(),
            loaded: std::sync::Mutex::new(Vec::new()),
        });
        let leases = Arc::new(FakeLeases {
            base: 0xB000_0000,
            next: std::sync::Mutex::new(0),
            released: std::sync::Mutex::new(Vec::new()),
        });
        let exchange = Arc::new(LocalExchange::default());
        let service = SiriusComputeNodeService::with_executor_and_exchange(
            Arc::new(StubExecutor),
            exchange.clone(),
            ExchangeIdentity::default(),
            None,
        )
        .with_nixl_control(Some(md.clone()), Some(leases.clone()));
        (service, md, leases, exchange)
    }

    fn call_transmit(
        service: &SiriusComputeNodeService,
        params: PTransmitChunkParams,
        attachment: Vec<u8>,
    ) -> (PTransmitChunkResult, Vec<u8>) {
        let response = route(
            service,
            methods::TRANSMIT_CHUNK,
            params.encode_to_vec(),
            attachment,
        );
        (
            PTransmitChunkResult::decode(response.body.as_slice()).unwrap(),
            response.attachment,
        )
    }

    #[test]
    fn transmit_chunk_md_returns_local_blob_without_ingesting() {
        let (service, md, _, _) = nixl_control_service();
        let (result, attachment) = call_transmit(
            &service,
            crate::nixl_chunk::control_params(),
            crate::nixl_chunk::NixlEnvelope::Md(b"peer-md".to_vec()).encode(),
        );
        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);
        assert_eq!(attachment, b"local-md");
        assert_eq!(md.loaded.lock().unwrap().as_slice(), &[b"peer-md".to_vec()]);
    }

    #[test]
    fn transmit_chunk_lease_replies_with_addr_and_offset() {
        let (service, _, _, _) = nixl_control_service();
        let (result, attachment) = call_transmit(
            &service,
            crate::nixl_chunk::control_params(),
            crate::nixl_chunk::NixlEnvelope::Lease { length: 64 }.encode(),
        );
        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);
        assert_eq!(
            crate::nixl_chunk::decode_lease_reply(&attachment).unwrap(),
            (0xB000_0000, 0)
        );
        let (result, attachment) = call_transmit(
            &service,
            crate::nixl_chunk::control_params(),
            crate::nixl_chunk::NixlEnvelope::Lease { length: 16 }.encode(),
        );
        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);
        assert_eq!(
            crate::nixl_chunk::decode_lease_reply(&attachment).unwrap(),
            (0xB000_0000 + 64, 64)
        );
    }

    #[test]
    fn transmit_chunk_canary_release_does_not_ingest() {
        let (service, _, leases, _) = nixl_control_service();
        let (result, attachment) = call_transmit(
            &service,
            crate::nixl_chunk::control_params(),
            crate::nixl_chunk::NixlEnvelope::CanaryRelease { offset: 0x2000 }.encode(),
        );
        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);
        assert!(attachment.is_empty());
        assert_eq!(leases.released.lock().unwrap().as_slice(), &[0x2000]);
    }

    #[test]
    fn transmit_chunk_packed_delivers_staged_batch_and_eos_into_the_rendezvous() {
        let (service, _, _, exchange) = nixl_control_service();
        let instance = FragmentInstanceId::from_halves(11, 22);
        let metadata = b"pack-meta".to_vec();
        let data = crate::nixl_chunk::PackedExchangeFrame {
            fragment_instance_id: instance,
            dest_stream: 7,
            sender_id: 0,
            seq: 0,
            eos: false,
            names: vec!["id".to_string()],
            offset: 4096,
            length: 32,
            rows: Some(5),
            metadata: metadata.clone(),
        };
        let (result, attachment) = call_transmit(&service, data.params(), data.envelope().encode());
        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);
        assert!(attachment.is_empty());

        let eos = crate::nixl_chunk::PackedExchangeFrame {
            fragment_instance_id: instance,
            dest_stream: 7,
            sender_id: 0,
            seq: 1,
            eos: true,
            names: vec!["id".to_string()],
            offset: 0,
            length: 0,
            rows: None,
            metadata: Vec::new(),
        };
        let (result, _) = call_transmit(&service, eos.params(), eos.envelope().encode());
        assert_eq!(result.status.unwrap().status_code, TStatusCode::OK.0);

        let ready = exchange
            .register_receiver(instance, vec![(7, 1)], fragment_params(None, None))
            .unwrap()
            .expect("eos already arrived over transmit_chunk");
        let SenderSource::Remote {
            names,
            sender_id,
            batches,
            closed,
        } = &ready.inputs[0].sources[0]
        else {
            panic!("expected a remote source");
        };
        assert_eq!(names, &["id".to_string()]);
        assert_eq!(*sender_id, 0);
        assert!(*closed);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].metadata, metadata);
        assert_eq!(batches[0].offset, 4096);
        assert_eq!(batches[0].len, 32);
        assert_eq!(batches[0].rows, Some(5));
    }

    #[test]
    fn transmit_chunk_rejects_native_chunkpb() {
        let (service, _, _, _) = nixl_control_service();
        let mut params = crate::nixl_chunk::control_params();
        params
            .chunks
            .push(crate::proto::starrocks::ChunkPb::default());
        let (result, _) = call_transmit(&service, params, Vec::new());
        assert_eq!(
            result.status.as_ref().unwrap().status_code,
            TStatusCode::INTERNAL_ERROR.0
        );
        assert!(
            result.status.as_ref().unwrap().error_msgs[0].contains("ChunkPB"),
            "{result:?}"
        );
    }

    #[test]
    fn transmit_chunk_md_and_lease_reply_over_brpc() {
        use crate::brpc::BrpcServer;
        use tokio_util::sync::CancellationToken;

        let (service, md, _, _) = nixl_control_service();
        let listener = match BrpcServer::bind("127.0.0.1", 0) {
            Ok(listener) => listener,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let peer = listener.local_addr().unwrap();
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap();
            runtime.block_on(
                BrpcServer::with_service(service)
                    .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });

        let md_reply = crate::nixl_chunk::transmit_envelope_blocking(
            peer,
            crate::nixl_chunk::control_params(),
            &crate::nixl_chunk::NixlEnvelope::Md(b"peer-md".to_vec()),
        )
        .unwrap();
        assert_eq!(md_reply, b"local-md");
        assert_eq!(md.loaded.lock().unwrap().as_slice(), &[b"peer-md".to_vec()]);

        let lease_reply = crate::nixl_chunk::transmit_envelope_blocking(
            peer,
            crate::nixl_chunk::control_params(),
            &crate::nixl_chunk::NixlEnvelope::Lease { length: 1_048_576 },
        )
        .unwrap();
        assert_eq!(
            crate::nixl_chunk::decode_lease_reply(&lease_reply).unwrap(),
            (0xB000_0000, 0)
        );

        shutdown.cancel();
        join.join().unwrap().unwrap();
    }

    fn is_permission_denied(err: &anyhow::Error) -> bool {
        err.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|err| err.kind() == std::io::ErrorKind::PermissionDenied)
        })
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
    fn park_then_send_local_exchange_feeds_result_sink() {
        // Receiver-first: the FE parks an EXCHANGE RESULT_SINK, then a leaf DATA_STREAM_SINK
        // completes the sender set and the stub result is fetchable.
        let service = SiriusComputeNodeService::new();
        let query = TUniqueId::new(8, 1);
        let receiver_id = TUniqueId::new(8, 10);
        let sender_id = TUniqueId::new(8, 11);

        let mut receiver = fragment_params(
            Some(TPlan::new(vec![exchange_plan_node(2, 0)])),
            Some(desc_table()),
        );
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query.clone(), receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(2, 1);
        receiver.params = Some(receiver_exec);

        let exec = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&receiver),
        );
        let exec = PExecPlanFragmentResult::decode(exec.body.as_slice()).unwrap();
        assert_eq!(
            exec.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            exec.status.error_msgs
        );

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(stream_sink(2));
        let mut sender_exec = exec_params(query, sender_id);
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            receiver_id.clone(),
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 8060)),
            None,
        )]);
        sender.params = Some(sender_exec);

        let exec = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&sender),
        );
        let exec = PExecPlanFragmentResult::decode(exec.body.as_slice()).unwrap();
        assert_eq!(
            exec.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            exec.status.error_msgs
        );

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
    fn remote_stream_sink_without_transport_is_an_error() {
        let service = SiriusComputeNodeService::new();
        let query = TUniqueId::new(9, 1);
        let receiver_id = TUniqueId::new(9, 10);
        let sender_id = TUniqueId::new(9, 11);

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(stream_sink(2));
        let mut sender_exec = exec_params(query, sender_id);
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            receiver_id,
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 18060)),
            None,
        )]);
        sender.params = Some(sender_exec);

        let exec = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&sender),
        );
        let exec = PExecPlanFragmentResult::decode(exec.body.as_slice()).unwrap();
        assert_eq!(exec.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            exec.status.error_msgs[0].contains("nixl transport"),
            "{:?}",
            exec.status.error_msgs
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
        node.num_children = 0;
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
