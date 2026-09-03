use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};

#[cfg(test)]
use crate::fragment_executor::StubExecutor;
use crate::fragment_executor::{FragmentExecutor, FragmentRun, SenderSlot, StagedBatch};
use crate::local_exchange::{ExchangeKey, LocalExchange, ReadyFragment, SenderSource};
use crate::nixl_transport::{NixlTransport, RemoteSendSpec};
use crate::proto::starrocks::{
    PCancelPlanFragmentRequest, PCancelPlanFragmentResult, PExchangeNixlMd, PExchangeNixlMdResult,
    PExecBatchPlanFragmentsRequest, PExecBatchPlanFragmentsResult, PExecPlanFragmentRequest,
    PExecPlanFragmentResult, PFetchDataRequest, PFetchDataResult, PGetFileSchemaRequest,
    PGetFileSchemaResult, PSlotDescriptor, PStagingLeaseRequest, PStagingLeaseResult,
    PTransmitPackedParams, PTransmitPackedResult, StatusPb,
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
    plan_nodes::TFileFormatType,
    status_code::TStatusCode,
    types::TNetworkAddress,
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

/// The exchange endpoint this CN is known by to the FE: advertised host plus brpc port.
///
/// A data-stream sink destination is classified against it before the sender runs: a match
/// keeps the exchange in-process, anything else is a remote CN.
#[derive(Clone, Debug)]
pub struct ExchangeIdentity {
    host: String,
    brpc_port: u16,
}

impl ExchangeIdentity {
    /// Builds the identity from the CN's advertised host and brpc port (`ComputeNodeConfig`).
    pub fn new(host: impl Into<String>, brpc_port: u16) -> Self {
        Self {
            host: host.into(),
            brpc_port,
        }
    }

    /// Hostname AND port equality — the stock BE's locality rule (`exchange_sink_operator.cpp`
    /// compares both), which is exactly what makes two CNs on one host see each other as remote.
    fn matches(&self, addr: &TNetworkAddress) -> bool {
        addr.hostname == self.host && addr.port == i32::from(self.brpc_port)
    }
}

/// Where a data-stream sink's single destination lives relative to this CN. Decided before the
/// sender fragment runs; the cross-node transport tier implements the `Remote` arm.
#[derive(Debug)]
enum DestinationRoute {
    /// The receiver is this process: park output on the GPU and rendezvous in-memory.
    Local,
    /// The receiver is another CN, reached via its advertised brpc endpoint. The port is
    /// validated here, at routing time, so a malformed destination fails before any GPU work.
    Remote { host: String, brpc_port: u16 },
}

/// What processing one fragment produced: the receivers whose sender sets it completed. The
/// caller decides where they execute — RPC handlers hand them to the dispatch worker, the worker
/// itself runs them inline.
#[must_use]
#[derive(Debug, Default)]
struct FragmentOutcome {
    ready: Vec<ReadyFragment>,
}

impl FragmentOutcome {
    /// A fragment that readied `ready` receivers.
    fn from_ready(ready: Vec<ReadyFragment>) -> Self {
        Self { ready }
    }
}

/// Sirius compute-node implementation of StarRocks PInternalService.
///
/// Plan-fragment translation is the first implemented RPC path; future
/// compute-node tasks should land here behind the generated service facade.
#[derive(Clone, Debug)]
pub(crate) struct SiriusComputeNodeService {
    /// Fragment-processing state shared across BRPC connections and the dispatch worker.
    core: Arc<ServiceCore>,
    /// Hands ready receiver fragments to the dispatch worker instead of executing them on the
    /// RPC thread that completed the sender set.
    ready_fragments: mpsc::Sender<ReadyFragment>,
}

/// Shared state behind every clone of [`SiriusComputeNodeService`].
///
/// One `Arc` so a `fetch_data` poll on one BRPC connection sees what an `exec_plan_fragment` on
/// another buffered — and so the dispatch worker holds the state (executor included) without
/// holding a `ready_fragments` sender, letting the channel close and the worker exit when the
/// last service clone drops. That keeps engine teardown ordered behind the servers.
#[derive(Debug)]
struct ServiceCore {
    /// Reusable StarRocks thrift-to-Substrait fragment translator.
    translator: PlanTranslator,
    /// Executes a translated fragment into Arrow result batches. Production injects the GPU-backed
    /// `SiriusEngine` (via [`SiriusComputeNodeService::with_executor`]); tests use a stub.
    executor: Arc<dyn FragmentExecutor>,
    /// Buffers executed-fragment results for FE `fetch_data` collection.
    results: ResultStore,
    /// Sequential same-node fragment exchange state.
    exchanges: LocalExchange,
    /// StarRocks may send a descriptor table once per query and mark later fragments as cached.
    descriptor_tables: Mutex<HashMap<FragmentInstanceId, TDescriptorTable>>,
    /// The exchange endpoint this CN advertises; destinations matching it are local.
    identity: ExchangeIdentity,
    /// The nixl transport tier for remote destinations. `None` keeps every remote destination a
    /// loud error naming how to enable it.
    transport: Option<NixlTransport>,
    /// Staging arena `(base, capacity)`, cached after the first lookup (the arena never moves).
    /// The executor serves every staging call from a thread-safe arena handle, never from the
    /// engine's request queue, so a lease request costs a mutex — not an engine wait.
    staging_info: Mutex<Option<(u64, u64)>>,
}

impl SiriusComputeNodeService {
    /// Test-only constructor with the placeholder [`StubExecutor`] and the default local
    /// identity. Production injects everything via [`with_transport`](Self::with_transport).
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_executor(
            Arc::new(StubExecutor),
            ExchangeIdentity::new("127.0.0.1", 8060),
        )
    }

    /// [`with_transport`](Self::with_transport) without a nixl transport: every remote
    /// destination stays a loud error.
    #[cfg(test)]
    pub(crate) fn with_executor(
        executor: Arc<dyn FragmentExecutor>,
        identity: ExchangeIdentity,
    ) -> Self {
        Self::with_transport(executor, identity, None)
    }

    /// Builds the service with a caller-provided fragment executor (e.g. the GPU-backed
    /// `SiriusEngine`), this CN's advertised exchange identity, and an optional nixl transport
    /// for remote destinations, shared across BRPC connections via the `Arc`. Also spawns the
    /// fragment dispatch worker.
    pub(crate) fn with_transport(
        executor: Arc<dyn FragmentExecutor>,
        identity: ExchangeIdentity,
        transport: Option<NixlTransport>,
    ) -> Self {
        let core = Arc::new(ServiceCore {
            translator: PlanTranslator::new(),
            executor,
            results: ResultStore::default(),
            exchanges: LocalExchange::default(),
            descriptor_tables: Mutex::new(HashMap::new()),
            identity,
            transport,
            staging_info: Mutex::new(None),
        });
        // A dedicated thread with a std channel (not a tokio task): fragment execution is
        // synchronous and blocking, so it gets the same dedicated-thread shape as the engine
        // itself, and the BRPC current-thread runtime is never involved.
        let (ready_fragments, worker_inbox) = mpsc::channel();
        let worker_core = Arc::clone(&core);
        std::thread::Builder::new()
            .name("fragment-dispatch".to_string())
            .spawn(move || dispatch_worker(worker_core, worker_inbox))
            .expect("failed to spawn fragment dispatch worker");
        Self {
            core,
            ready_fragments,
        }
    }

    /// Hands a ready receiver to the dispatch worker so this RPC thread returns immediately
    /// instead of blocking on the receiver's whole execution.
    fn dispatch(&self, ready: ReadyFragment) -> std::result::Result<(), String> {
        self.ready_fragments.send(ready).map_err(|_| {
            "fragment dispatch worker has exited; cannot execute receiver fragment".to_string()
        })
    }

    /// Hands every receiver this fragment readied to the dispatch worker, so the RPC that
    /// completed their sender sets returns without waiting on their execution.
    fn dispatch_ready(&self, outcome: FragmentOutcome) -> std::result::Result<(), String> {
        for ready in outcome.ready {
            self.dispatch(ready)?;
        }
        Ok(())
    }
}

/// Executes ready receiver fragments sequentially, off the RPC threads.
///
/// Exits when every service clone has dropped its sender; the worker then releases its core
/// handle (and with it the executor), keeping engine teardown ordered.
fn dispatch_worker(core: Arc<ServiceCore>, inbox: mpsc::Receiver<ReadyFragment>) {
    while let Ok(ready) = inbox.recv() {
        // A receiver can itself be a sender whose completion readies further receivers (a
        // middle fragment, possibly fanning out); chase the whole set inline — this thread is
        // already off the RPC path.
        let mut queue = vec![ready];
        while let Some(ready) = queue.pop() {
            queue.extend(core.run_ready_fragment(ready));
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

    /// Best-effort cancellation stub: acknowledges the FE with OK so its shared jprotobuf
    /// channel stays healthy — the default unrouted reply is a PRPC-level error frame, and the
    /// FE reaps the timed-out future in a way that misattributes later replies on the channel.
    /// A still-waiting result entry is failed so a `fetch_data` long-poll returns immediately.
    /// Real teardown (aborting the engine run, freeing GPU buffers, dropping parked exchange
    /// state) is a separate work item.
    #[instrument(skip_all)]
    async fn cancel_plan_fragment(
        &self,
        request: PCancelPlanFragmentRequest,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PCancelPlanFragmentResult>, crate::prpc::Error> {
        let id = FragmentInstanceId::from(&request.finst_id);
        info!(
            fragment_instance_id = %id,
            query_id = ?request.query_id.as_ref().map(FragmentInstanceId::from),
            cancel_reason = request.cancel_reason,
            error_message = ?request.error_message,
            "acknowledging cancel_plan_fragment (best-effort: no engine-side abort yet)"
        );
        let mut reason = format!("fragment instance {id} was cancelled by the FE");
        if let Some(message) = request.error_message.as_ref().filter(|msg| !msg.is_empty()) {
            reason = format!("{reason}: {message}");
        }
        self.core.results.cancel(id, reason);
        Ok(PCancelPlanFragmentResult {
            status: Self::ok_status(),
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
        // Long-poll: receivers execute on the dispatch thread now, so a reserved result may not
        // be ready when the FE's first poll arrives. Block off the runtime until it is — every
        // reply consumes a packet-sequence slot in the FE's ResultReceiver, so a not-ready reply
        // would desync the counter and cancel the query ("expect=1, receive=0").
        let core = self.core.clone();
        let outcome = match tokio::task::spawn_blocking(move || {
            core.results
                .wait_ready(id, std::time::Duration::from_secs(600))
        })
        .await
        {
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
        // An unknown id is an error, not EOS: it means this CN never buffered a result for the
        // fragment the FE is polling (wrong id, or a dispatch/result-sink path that did not run),
        // and StarRocks treats a missing result buffer as a failure rather than an empty result.
        let Some(outcome) = outcome else {
            return Ok(Self::fetch_data_result(
                Self::internal_error(format!("no buffered result for fragment instance {id}")),
                0,
                true,
            )
            .into());
        };
        match outcome {
            FetchOutcome::Failed(cause) => Ok(Self::fetch_data_result(
                Self::internal_error(format!("fragment instance {id} failed: {cause}")),
                0,
                true,
            )
            .into()),
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

    /// First-contact handshake of the nixl tier: load the caller's agent metadata, reply with
    /// this CN's. Idempotent, so a peer restart re-exchanges cleanly.
    #[instrument(skip_all)]
    async fn exchange_nixl_md(
        &self,
        request: PExchangeNixlMd,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PExchangeNixlMdResult>, crate::prpc::Error> {
        // The transport thread call blocks on a respond channel; keep the BRPC current-thread
        // runtime free, like every other blocking handler here.
        let service = self.clone();
        let outcome =
            tokio::task::spawn_blocking(move || service.core.handle_exchange_nixl_md(&request))
                .await;
        let result = match outcome {
            Ok(Ok(reply)) => PExchangeNixlMdResult {
                status: Self::ok_status(),
                agent_name: Some(reply.agent_name),
                agent_metadata: Some(reply.metadata),
            },
            Ok(Err(err)) => PExchangeNixlMdResult {
                status: Self::internal_error(err),
                agent_name: None,
                agent_metadata: None,
            },
            Err(join_err) => PExchangeNixlMdResult {
                status: Self::internal_error(format!("exchange_nixl_md task panicked: {join_err}")),
                agent_name: None,
                agent_metadata: None,
            },
        };
        Ok(result.into())
    }

    /// Leases bytes of this CN's staging arena for a peer's nixl WRITE. Served directly from
    /// the executor's arena handle — never the engine's request queue — so a peer's exchange
    /// survives this CN running a long (or wedged) fragment.
    #[instrument(skip_all)]
    async fn request_staging_lease(
        &self,
        request: PStagingLeaseRequest,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PStagingLeaseResult>, crate::prpc::Error> {
        let service = self.clone();
        let outcome =
            tokio::task::spawn_blocking(move || service.core.handle_staging_lease(request.length))
                .await;
        let result = match outcome {
            Ok(Ok((remote_addr, offset))) => PStagingLeaseResult {
                status: Self::ok_status(),
                remote_addr: Some(remote_addr),
                offset: Some(offset),
            },
            Ok(Err(err)) => PStagingLeaseResult {
                status: Self::internal_error(err),
                remote_addr: None,
                offset: None,
            },
            Err(join_err) => PStagingLeaseResult {
                status: Self::internal_error(format!(
                    "request_staging_lease task panicked: {join_err}"
                )),
                remote_addr: None,
                offset: None,
            },
        };
        Ok(result.into())
    }

    /// Ingests one remote exchange frame: a packed batch already WRITTEN into this CN's arena
    /// (metadata in the attachment), an eos, or a canary lease release.
    #[instrument(skip_all)]
    async fn transmit_packed(
        &self,
        request: PTransmitPackedParams,
        attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PTransmitPackedResult>, crate::prpc::Error> {
        // Blocking again: the canary path round-trips the engine thread, and a completed sender
        // set dispatches (cheaply) to the fragment worker.
        let service = self.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            service.handle_transmit_packed(&request, attachment)
        })
        .await;
        let status = match outcome {
            Ok(Ok(())) => Self::ok_status(),
            Ok(Err(err)) => Self::internal_error(err),
            Err(join_err) => {
                Self::internal_error(format!("transmit_packed task panicked: {join_err}"))
            }
        };
        Ok(PTransmitPackedResult { status }.into())
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
    /// Deserializes one binary-thrift TExecPlanFragmentParams attachment and processes it,
    /// handing any completed receiver to the dispatch worker.
    fn exec_single_attachment(
        &self,
        protocol: Option<&str>,
        attachment: &[u8],
    ) -> std::result::Result<(), String> {
        Self::ensure_binary_protocol(protocol)?;
        let params = Self::deserialize_binary::<TExecPlanFragmentParams>(attachment)
            .map_err(|err| format!("failed to deserialize TExecPlanFragmentParams: {err}"))?;
        self.dispatch_ready(self.core.process_fragment(&params)?)
    }

    /// Records one remote exchange frame in the rendezvous (or releases a canary lease), handing
    /// a completed receiver to the dispatch worker.
    fn handle_transmit_packed(
        &self,
        params: &PTransmitPackedParams,
        attachment: Vec<u8>,
    ) -> std::result::Result<(), String> {
        if params.canary() {
            // The bandwidth canary writes into a lease nothing will consume; release it without
            // touching the rendezvous or the engine's input streams.
            let offset = params.offset.ok_or_else(|| {
                "canary transmit_packed frame carries no lease offset".to_string()
            })?;
            return self.core.executor.staging_release(offset);
        }
        let finst_id = params
            .finst_id
            .as_ref()
            .ok_or_else(|| "transmit_packed frame carries no finst_id".to_string())?;
        let node_id = params
            .node_id
            .ok_or_else(|| "transmit_packed frame carries no node_id".to_string())?;
        let sender_id = params
            .sender_id
            .ok_or_else(|| "transmit_packed frame carries no sender_id".to_string())?;
        let seq = params
            .seq
            .ok_or_else(|| "transmit_packed frame carries no seq".to_string())?;
        let eos = params.eos.unwrap_or(false);
        let key = ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from(finst_id),
            node_id,
        };
        // The pack metadata rides the attachment; its presence is what makes a frame a batch.
        let batch = if attachment.is_empty() {
            None
        } else {
            let length = params.length.ok_or_else(|| {
                "transmit_packed batch frame carries no payload length".to_string()
            })?;
            let offset = params
                .offset
                .ok_or_else(|| "transmit_packed batch frame carries no lease offset".to_string())?;
            Some(StagedBatch {
                metadata: attachment,
                offset,
                len: length,
                // None from a sender that predates the wire field: the receiver then skips the
                // stream's cardinality declaration instead of failing the frame.
                rows: params.rows,
            })
        };
        tracing::debug!(
            exchange = ?key,
            sender_id,
            seq,
            eos,
            batch_bytes = batch.as_ref().map(|batch| batch.len),
            "received remote exchange frame"
        );
        if let Some(ready) = self.core.exchanges.push_remote_frame(
            key,
            sender_id,
            seq,
            eos,
            params.column_names.clone(),
            batch,
        )? {
            self.dispatch(ready)?;
        }
        Ok(())
    }
}

impl ServiceCore {
    /// Runs one dispatched receiver, parking a failure where `fetch_data` can see it. Returns
    /// the next receiver when this fragment's own sink completed another sender set.
    fn run_ready_fragment(&self, ready: ReadyFragment) -> Vec<ReadyFragment> {
        let id = Self::fragment_instance_id(&ready.params);
        let query_id = Self::query_id(&ready.params);
        let is_result_fragment = matches!(Self::is_mysql_result_sink(&ready.params), Ok(true));
        match self.execute_ready_fragment(ready) {
            Ok(outcome) => outcome.ready,
            Err(error) => {
                match (id, query_id) {
                    (Some(id), Some(query_id)) => {
                        if is_result_fragment {
                            // The FE polls fetch_data on this id; its reserved entry becomes
                            // the error instead of waiting forever.
                            tracing::error!(fragment_instance_id = %id, error = %error, "dispatched result fragment failed");
                        } else {
                            // An intermediate receiver has no FE-polled entry of its own, so
                            // its failure must reach the query's result-fragment instances —
                            // otherwise the FE polls until its timeout and the stalled
                            // fetch_data wedges this CN's whole frame loop.
                            tracing::error!(
                                fragment_instance_id = %id,
                                error = %error,
                                "dispatched intermediate receiver fragment failed; failing the query's result fragments"
                            );
                        }
                        // Fails this id, every reserved result instance of the query, and
                        // records the failure so a result fragment arriving later fails on
                        // registration instead of waiting on senders that never deliver.
                        self.results.fail_query(query_id, id, error);
                    }
                    (Some(id), None) => {
                        // Defensive: exec params carry both ids or neither, so this arm should
                        // be unreachable. Park the error under the instance id at least.
                        tracing::error!(fragment_instance_id = %id, error = %error, "dispatched receiver fragment without a query_id failed");
                        self.results.fail(id, error);
                    }
                    (None, _) => {
                        tracing::error!(error = %error, "dispatched receiver fragment without a fragment_instance_id failed");
                    }
                }
                Vec::new()
            }
        }
    }

    /// Processes one fragment and buffers supported RESULT_SINK rows for later `fetch_data`.
    /// Shared by single and batch dispatch; exchange receivers are registered until their local
    /// sender output is materialized. Returns a receiver whose sender set this fragment
    /// completed — the caller decides where it executes (RPC handlers hand it to the dispatch
    /// worker; the worker itself runs it inline).
    fn process_fragment(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<FragmentOutcome, String> {
        let params = self.resolve_descriptor_table(params)?;
        let dump_seq = Self::dump_fragment(&params);
        // Survey mode: accept every fragment so the FE dispatches (and we dump) the whole
        // plan even when translation fails. Queries still fail at fetch_data.
        if std::env::var_os("SIRIUS_CN_TRANSLATE_ONLY").is_some() {
            if let Err(err) = self.translate_fragment_logged(&params, dump_seq) {
                tracing::warn!(error = %err, "translate-only mode: accepting untranslatable fragment");
            }
            return Ok(FragmentOutcome::default());
        }
        let expected_senders = Self::receiver_exchanges(&params)?;
        if !expected_senders.is_empty() {
            let fragment_instance_id = Self::fragment_instance_id(&params)
                .ok_or_else(|| "exchange receiver is missing a fragment_instance_id".to_string())?;
            if Self::is_mysql_result_sink(&params)? {
                let query_id = Self::query_id(&params)
                    .ok_or_else(|| "exchange receiver is missing a query_id".to_string())?;
                self.results.reserve(fragment_instance_id, query_id);
            }
            return Ok(FragmentOutcome::from_ready(
                self.exchanges
                    .register_receiver(fragment_instance_id, expected_senders, params)?
                    .into_iter()
                    .collect(),
            ));
        }

        let translated = self.translate_fragment_logged(&params, dump_seq)?;
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

    /// Executes a fragment that reads no exchange input.
    fn execute_fragment(
        &self,
        params: &TExecPlanFragmentParams,
        translated: TranslatedPlan,
    ) -> std::result::Result<FragmentOutcome, String> {
        self.execute_fragment_with_inputs(params, translated, Vec::new(), Vec::new())
    }

    /// Executes a result fragment, or runs a data-stream sender and parks its output on the GPU
    /// for its local receiver (transmitting it when the receiver is remote). `inputs` names the
    /// parked sender outputs this fragment consumes; `remote_inputs` the staged remote batches.
    /// A sender that completes its receiver's sender set returns that receiver for the caller
    /// to run or dispatch.
    fn execute_fragment_with_inputs(
        &self,
        params: &TExecPlanFragmentParams,
        translated: TranslatedPlan,
        inputs: Vec<(i32, Vec<SenderSlot>)>,
        remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
    ) -> std::result::Result<FragmentOutcome, String> {
        if Self::is_mysql_result_sink(params)? {
            let id = Self::fragment_instance_id(params).ok_or_else(|| {
                "RESULT_SINK fragment is missing a fragment_instance_id".to_string()
            })?;
            let result = self
                .executor
                .run(FragmentRun {
                    plan: &translated,
                    inputs: inputs.clone(),
                    remote_inputs,
                    outputs: Vec::new(),
                    broadcast: false,
                    hash_keys: Vec::new(),
                })?
                .ok_or_else(|| "result fragment returned no rows".to_string())?;
            let batch = result_encoder::MysqlResultEncoder::encode(&result.batches, 0)?;
            self.results.insert(id, batch);
            return Ok(FragmentOutcome::default());
        }

        let Some(sink) = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
        else {
            // The FE attaches a sink to every fragment it dispatches (PlanFragment.toThrift),
            // so a sinkless fragment only reaches here from a translate-only fixture.
            tracing::warn!(
                fragment = %Self::fragment_context(params),
                "fragment carries no output sink; nothing consumes its output"
            );
            return Ok(FragmentOutcome::default());
        };
        // Accepting an unhandled sink discards the fragment's whole output: its consumers wait
        // forever, the FE's fetch_data long-poll times out, and its serial channel wedges
        // cluster-wide. Refuse by name so the FE error says which plan shape is unsupported.
        if sink.type_ != TDataSinkType::DATA_STREAM_SINK {
            return Err(format!(
                "{} carries a {} output sink, which this CN does not support",
                Self::fragment_context(params),
                Self::data_sink_type_name(sink.type_)
            ));
        }
        let stream_sink = sink.stream_sink.as_ref().ok_or_else(|| {
            format!(
                "{} carries a DATA_STREAM_SINK with no stream_sink payload",
                Self::fragment_context(params)
            )
        })?;
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
        // Fan-out shape: one destination is a gather regardless of the partition label;
        // UNPARTITIONED with N destinations broadcasts the full output to every receiver
        // (each destination drains its own copy from its own output stream). Hash-partitioned
        // fan-out is the second half of #838 and still refuses.
        if destinations.len() > 1 {
            use starrocks_thrift::partitions::TPartitionType;
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

        // Route every destination BEFORE running: a remote destination without a transport (or
        // a duplicate) must fail before any GPU work happens. Destination i then drains the
        // sender's output stream i -- the FE's destination order, positionally.
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
            if let DestinationRoute::Remote { host, brpc_port } = &route
                && self.transport.is_none()
            {
                return Err(format!(
                    "cross-node exchange to {host}:{brpc_port} needs the nixl transport \
                     tier, which is not active: build the CN with the `nixl-transport` \
                     feature (default) and set SIRIUS_EXCHANGE_STAGING_BYTES so the \
                     exchange staging arena exists"
                ));
            }
            slots.push(slot);
            routes.push(route);
        }

        // The sender's rows stay on the GPU, parked once with one output stream per
        // destination; only rendezvous bookkeeping and packed exports leave the engine thread.
        self.executor.run(FragmentRun {
            plan: &translated,
            inputs,
            remote_inputs,
            outputs: slots.clone(),
            broadcast,
            hash_keys,
        })?;

        // Local destinations first: their rendezvous is immediate bookkeeping and fails fast.
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
        // Remote destinations second, drained one at a time in the FE's destination order. Each
        // `send_fragment` blocks until the transport thread has exported this destination's
        // parked stream into staging leases, WRITTEN every batch into a lease the peer granted,
        // signaled each batch and the eos over brpc, and released the destination's claim on
        // the parked output (dropped on failure, so a dead query does not pin its output). The
        // seq counter and the eos frame live inside that one thread's drain, which is what the
        // receiver's per-(exchange, sender) gap check relies on. A failure fails the sender's
        // RPC — the FE sees the error, never a receiver that waits forever — and the
        // destinations after it are not drained; their claims go with the engine's next wipe.
        for (slot, route) in slots.iter().zip(&routes) {
            if let DestinationRoute::Remote { host, brpc_port } = route {
                let transport = self.transport.as_ref().expect("routed before running");
                transport.send_fragment(RemoteSendSpec {
                    host: host.clone(),
                    brpc_port: *brpc_port,
                    slot: *slot,
                    names: translated.output_names.clone(),
                })?;
            }
        }
        Ok(FragmentOutcome::from_ready(ready_receivers))
    }

    /// Classifies a data-stream sink destination against this CN's advertised exchange
    /// identity. A missing `brpc_server` is a malformed dispatch, not a local default: the FE
    /// always attaches the receiver's brpc address (ExecutionDAG.java:560).
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

    /// Loads a peer's nixl agent metadata into this CN's agent and returns ours.
    fn handle_exchange_nixl_md(
        &self,
        request: &PExchangeNixlMd,
    ) -> std::result::Result<crate::nixl_transport::MdReply, String> {
        let Some(transport) = self.transport.as_ref() else {
            return Err(
                "the nixl transport tier is not active on this CN: build with the \
                 `nixl-transport` feature (default) and set SIRIUS_EXCHANGE_STAGING_BYTES"
                    .to_string(),
            );
        };
        let peer_agent_name = request
            .agent_name
            .clone()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| "exchange_nixl_md request carries no agent name".to_string())?;
        let peer_metadata = request
            .agent_metadata
            .clone()
            .filter(|metadata| !metadata.is_empty())
            .ok_or_else(|| "exchange_nixl_md request carries no agent metadata".to_string())?;
        transport.exchange_md(peer_agent_name, peer_metadata)
    }

    /// Leases `length` bytes of the staging arena and returns `(absolute address, offset)`.
    fn handle_staging_lease(&self, length: u64) -> std::result::Result<(u64, u64), String> {
        let (base, _capacity) = self.staging_info()?;
        let offset = self.executor.staging_lease(length)?;
        Ok((base + offset, offset))
    }

    /// Staging arena `(base, capacity)`, cached after the first successful engine round-trip.
    fn staging_info(&self) -> std::result::Result<(u64, u64), String> {
        let mut cached = self
            .staging_info
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(info) = *cached {
            return Ok(info);
        }
        let info = self.executor.staging_info()?;
        *cached = Some(info);
        Ok(info)
    }

    /// Translates a receiver whose sender set is complete, binding each exchange to the input
    /// stream its senders parked into (or staged, for remote senders), and runs it. Returns the
    /// next receiver when this one's own sink completed another sender set.
    fn execute_ready_fragment(
        &self,
        ready: ReadyFragment,
    ) -> std::result::Result<FragmentOutcome, String> {
        let exchange_inputs = ready
            .inputs
            .iter()
            .map(|input| {
                let names = input
                    .sources
                    .first()
                    .map(|source| source.names().to_vec())
                    .ok_or_else(|| {
                        format!("exchange node {} has no sender source", input.node_id)
                    })?;
                // Local and remote senders alike must agree on the schema they produced; the
                // first source is the reference, disagreement fails the query.
                if input
                    .sources
                    .iter()
                    .any(|source| source.names() != names.as_slice())
                {
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
        // A receiver translates when its sender set completes, not at arrival, so pair its plan
        // dump with a fresh params dump here (the arrival-time dump carried no plan yet).
        let dump_seq = Self::dump_fragment(&ready.params);
        let translated =
            self.translate_fragment_logged_with_inputs(&ready.params, &exchange_inputs, dump_seq)?;
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
                        // take_ready only releases complete sender sets; an open remote source
                        // here is a rendezvous bug, not a recoverable state.
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
        self.execute_fragment_with_inputs(&ready.params, translated, inputs, remote_inputs)
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

    /// Converts a StarRocks thrift plan fragment to Substrait, logs substrait-explain output, and
    /// returns the translated plan for execution.
    #[instrument(skip_all)]
    fn translate_fragment_logged(
        &self,
        params: &TExecPlanFragmentParams,
        dump_seq: Option<u64>,
    ) -> std::result::Result<TranslatedPlan, String> {
        self.translate_fragment_logged_with_inputs(params, &[], dump_seq)
    }

    fn translate_fragment_logged_with_inputs(
        &self,
        params: &TExecPlanFragmentParams,
        exchange_inputs: &[ExchangeInput],
        dump_seq: Option<u64>,
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
        Self::dump_substrait(&translated, dump_seq);
        Ok(translated)
    }

    /// Writes the translated Substrait plan bytes to `$SIRIUS_CN_DUMP_FRAGMENTS/plan-<seq>.substrait`
    /// so a failing plan can be replayed against the engine in isolation; `<seq>` pairs the plan
    /// with its `fragment-<seq>.txt` params dump. No-op when the variable (or the seq) is unset.
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

    /// Names a thrift sink type for an operator. The generated type is a newtype over the wire
    /// number, so its `Debug` prints `TDataSinkType(7)` — useless in an FE error message.
    fn data_sink_type_name(sink_type: TDataSinkType) -> String {
        let name = match sink_type {
            TDataSinkType::DATA_STREAM_SINK => "DATA_STREAM_SINK",
            TDataSinkType::RESULT_SINK => "RESULT_SINK",
            TDataSinkType::DATA_SPLIT_SINK => "DATA_SPLIT_SINK",
            TDataSinkType::MYSQL_TABLE_SINK => "MYSQL_TABLE_SINK",
            TDataSinkType::EXPORT_SINK => "EXPORT_SINK",
            TDataSinkType::OLAP_TABLE_SINK => "OLAP_TABLE_SINK",
            TDataSinkType::MEMORY_SCRATCH_SINK => "MEMORY_SCRATCH_SINK",
            TDataSinkType::MULTI_CAST_DATA_STREAM_SINK => "MULTI_CAST_DATA_STREAM_SINK",
            TDataSinkType::SCHEMA_TABLE_SINK => "SCHEMA_TABLE_SINK",
            TDataSinkType::ICEBERG_TABLE_SINK => "ICEBERG_TABLE_SINK",
            TDataSinkType::HIVE_TABLE_SINK => "HIVE_TABLE_SINK",
            TDataSinkType::TABLE_FUNCTION_TABLE_SINK => "TABLE_FUNCTION_TABLE_SINK",
            TDataSinkType::BLACKHOLE_TABLE_SINK => "BLACKHOLE_TABLE_SINK",
            TDataSinkType::DICTIONARY_CACHE_SINK => "DICTIONARY_CACHE_SINK",
            TDataSinkType::MULTI_OLAP_TABLE_SINK => "MULTI_OLAP_TABLE_SINK",
            TDataSinkType::SPLIT_DATA_STREAM_SINK => "SPLIT_DATA_STREAM_SINK",
            TDataSinkType::NOOP_SINK => "NOOP_SINK",
            TDataSinkType::ICEBERG_DELETE_SINK => "ICEBERG_DELETE_SINK",
            other => return format!("unknown (wire value {})", other.0),
        };
        name.to_string()
    }

    /// Renders the ids that let an operator tie an error back to one FE-dispatched fragment.
    fn fragment_context(params: &TExecPlanFragmentParams) -> String {
        match (Self::fragment_instance_id(params), Self::query_id(params)) {
            (Some(instance), Some(query)) => {
                format!("fragment instance {instance} of query {query}")
            }
            (Some(instance), _) => format!("fragment instance {instance}"),
            (None, Some(query)) => format!("an unidentified instance of query {query}"),
            (None, None) => "an unidentified fragment instance".to_string(),
        }
    }

    /// Extracts the fragment instance id the FE later passes to `fetch_data`.
    fn fragment_instance_id(params: &TExecPlanFragmentParams) -> Option<FragmentInstanceId> {
        params
            .params
            .as_ref()
            .map(|exec| FragmentInstanceId::from(&exec.fragment_instance_id))
    }

    /// Extracts the query id shared by every fragment instance of one query, the scope a
    /// failure propagates across (`ResultStore::fail_query`).
    fn query_id(params: &TExecPlanFragmentParams) -> Option<FragmentInstanceId> {
        params
            .params
            .as_ref()
            .map(|exec| FragmentInstanceId::from(&exec.query_id))
    }
}

impl SiriusComputeNodeService {
    /// Deserializes a FE batch attachment and merges common params into each instance, handing
    /// any completed receiver to the dispatch worker.
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

            let outcome = self
                .core
                .process_fragment(&params)
                .map_err(|err| format!("fragment {idx}: {err}"))?;
            // Per instance, exactly like the single-attachment path, so the `fragment {idx}`
            // attribution of a dispatch failure names the instance that readied the receiver.
            self.dispatch_ready(outcome)
                .map_err(|err| format!("fragment {idx}: {err}"))?;
        }

        Ok(())
    }

    /// Extracts the parquet paths from the binary-thrift attachment and infers their common
    /// schema. The FE sends one request covering every FILES() file, one range per file.
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
        plan_nodes::{
            TBrokerRangeDesc, TBrokerScanRange, TBrokerScanRangeParams, TExchangeNode,
            TFileScanNode, TPlan, TPlanNode, TPlanNodeType, TScanRange,
        },
        planner::TPlanFragment,
        types::{
            TFileType, TNetworkAddress, TPrimitiveType, TScalarType, TTableType, TTypeDesc,
            TTypeNode, TTypeNodeType, TUniqueId,
        },
    };
    use thrift::{protocol::TBinaryOutputProtocol, transport::TIoChannel};
    use tower::{Service, ServiceExt};

    use super::*;
    use crate::{
        file_schema::test_support::write_parquet,
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

    /// Recognizes the run the dispatch worker performs: a receiver consumes exchange inputs and
    /// returns rows instead of parking output.
    fn is_receiver_run(run: &FragmentRun<'_>) -> bool {
        run.outputs.is_empty() && !run.inputs.is_empty()
    }

    /// Blocks receiver execution until the test releases it, proving the sender's RPC thread
    /// does not run the receiver inline.
    #[derive(Debug)]
    struct GatedExecutor {
        release: Mutex<std::sync::mpsc::Receiver<()>>,
        receiver_ran: std::sync::atomic::AtomicBool,
    }

    impl FragmentExecutor for GatedExecutor {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if is_receiver_run(&run) {
                // Bounded wait so a regression fails the test instead of hanging the suite.
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .map_err(|err| format!("gated receiver was never released: {err}"))?;
                self.receiver_ran.store(true, Ordering::SeqCst);
            }
            StubExecutor.run(run)
        }
    }

    /// Fails every receiver run so tests can watch the dispatch worker's failure path.
    #[derive(Debug)]
    struct FailingReceiverExecutor;

    impl FragmentExecutor for FailingReceiverExecutor {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if is_receiver_run(&run) {
                return Err("receiver exploded on the GPU".to_string());
            }
            StubExecutor.run(run)
        }
    }

    /// Fails only an intermediate fragment's run — the one that both consumes exchange input and
    /// parks sender output — so tests can watch a non-result failure reach the FE-polled result id.
    #[derive(Debug)]
    struct FailingIntermediateExecutor;

    impl FragmentExecutor for FailingIntermediateExecutor {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            if !run.inputs.is_empty() && !run.outputs.is_empty() {
                return Err("intermediate receiver exploded on the GPU".to_string());
            }
            StubExecutor.run(run)
        }
    }

    /// The exchange endpoint [`SiriusComputeNodeService::new`] advertises in tests.
    fn test_identity() -> ExchangeIdentity {
        ExchangeIdentity::new("127.0.0.1", 8060)
    }

    /// A destination on this CN. The FE always attaches the receiver's brpc address
    /// (ExecutionDAG.java:560), so fixtures must too.
    fn local_destination(receiver_id: TUniqueId) -> TPlanFragmentDestination {
        TPlanFragmentDestination::new(
            receiver_id,
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 8060)),
            None,
        )
    }

    /// Polls until `predicate` holds; receiver execution now happens on the dispatch worker.
    fn wait_until(what: &str, predicate: impl Fn() -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !predicate() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Polls `fetch_data` until the dispatched receiver's rows land, panicking on an error
    /// status or on EOS before any rows arrived.
    fn fetch_rows_eventually(
        service: &SiriusComputeNodeService,
        hi: i64,
        lo: i64,
    ) -> prpc::Response {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let response = route(
                service,
                methods::FETCH_DATA,
                fetch_request(hi, lo),
                Vec::new(),
            );
            let result = PFetchDataResult::decode(response.body.as_slice()).unwrap();
            assert_eq!(
                result.status.status_code,
                TStatusCode::OK.0,
                "{:?}",
                result.status.error_msgs
            );
            if !response.attachment.is_empty() {
                return response;
            }
            assert_eq!(
                result.eos,
                Some(false),
                "receiver reported EOS before delivering rows"
            );
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for receiver rows"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Polls `fetch_data` until the dispatched receiver's failure surfaces as a non-OK status.
    fn fetch_error_eventually(
        service: &SiriusComputeNodeService,
        hi: i64,
        lo: i64,
    ) -> PFetchDataResult {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let response = route(
                service,
                methods::FETCH_DATA,
                fetch_request(hi, lo),
                Vec::new(),
            );
            let result = PFetchDataResult::decode(response.body.as_slice()).unwrap();
            if result.status.status_code != TStatusCode::OK.0 {
                return result;
            }
            assert!(response.attachment.is_empty());
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for receiver failure"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
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
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
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

        // No intermediate not-ready probe: fetch_data long-polls now (an empty reply would
        // desync the FE's packet counter), so the reserved entry is only fetched after the
        // sender completes -- which exercises the blocking path end to end.
        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(query_id, TUniqueId::new(10, 3));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![local_destination(receiver_id.clone())]);
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
        // The sender ran on the RPC path; the receiver executes on the dispatch worker.
        wait_until("sender and receiver to execute", || {
            executor.calls.load(Ordering::Relaxed) == 2
        });

        let fetched = fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        let fetched_result = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(fetched_result.eos, Some(false));
        assert!(!fetched.attachment.is_empty());
    }

    #[test]
    fn self_exchange_executes_an_intermediate_receiver_and_reuses_cached_descriptors() {
        let executor = Arc::new(CountingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
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
        middle_exec.destinations = Some(vec![local_destination(root_id.clone())]);
        middle.params = Some(middle_exec);
        assert_exec_ok(&service, &middle);

        let mut leaf = fragment_params(Some(scan_plan(0, 0)), Some(cached_desc));
        leaf.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut leaf_exec = exec_params(query_id, TUniqueId::new(20, 4));
        leaf_exec.sender_id = Some(0);
        leaf_exec.destinations = Some(vec![local_destination(middle_id)]);
        leaf.params = Some(leaf_exec);
        assert_exec_ok(&service, &leaf);

        // The leaf ran on the RPC path; the middle and root receivers chain on the dispatch
        // worker after the leaf's RPC already returned.
        wait_until("all three fragments to execute", || {
            executor.calls.load(Ordering::Relaxed) == 3
        });
        let fetched = fetch_rows_eventually(&service, root_id.hi, root_id.lo);
        let fetched_result = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched_result.status.status_code, TStatusCode::OK.0);
        assert_eq!(fetched_result.eos, Some(false));
        assert!(!fetched.attachment.is_empty());
    }

    #[test]
    fn exchange_identity_requires_host_and_port_equality() {
        let identity = ExchangeIdentity::new("cn-a.example", 8060);
        assert!(identity.matches(&TNetworkAddress::new("cn-a.example".to_string(), 8060)));
        // Two CNs on one host differ only by port and must see each other as remote.
        assert!(!identity.matches(&TNetworkAddress::new("cn-a.example".to_string(), 8061)));
        assert!(!identity.matches(&TNetworkAddress::new("cn-b.example".to_string(), 8060)));
    }

    #[test]
    fn data_stream_sink_to_remote_destination_is_a_loud_error() {
        // Until the cross-node transport lands, a remote destination must fail the sender's
        // dispatch loudly — the silent alternative is an FE query that hangs forever.
        let executor = Arc::new(CountingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(TUniqueId::new(30, 1), TUniqueId::new(30, 2));
        sender_exec.sender_id = Some(0);
        // A second CN on the same host: same hostname, different brpc port.
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            TUniqueId::new(30, 3),
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 8061)),
            None,
        )]);
        sender.params = Some(sender_exec);

        let response = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&sender),
        );
        let result = PExecPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0]
                .contains("cross-node exchange to 127.0.0.1:8061 needs the nixl transport tier"),
            "{:?}",
            result.status.error_msgs
        );
        // Routing is decided before the sender runs, so no GPU work happened.
        assert_eq!(executor.calls.load(Ordering::Relaxed), 0);
    }

    /// With a transport wired in, a remote destination runs the sender (parking its output) and
    /// hands the parked slot to the transport with the peer address and output names.
    #[test]
    fn data_stream_sink_to_remote_destination_hands_the_parked_output_to_the_transport() {
        let (requests_tx, requests_rx) = mpsc::channel();
        let fake_transport = std::thread::spawn(move || {
            match requests_rx
                .recv()
                .expect("the sender flow sends one request")
            {
                crate::nixl_transport::TransportRequest::SendFragment { spec, respond } => {
                    respond.send(Ok(())).unwrap();
                    spec
                }
                crate::nixl_transport::TransportRequest::ExchangeMd { .. } => {
                    panic!("the sender flow never exchanges metadata itself")
                }
            }
        });
        let executor = Arc::new(CountingExecutor::default());
        let service = SiriusComputeNodeService::with_transport(
            executor.clone(),
            test_identity(),
            Some(crate::nixl_transport::NixlTransport::for_test(requests_tx)),
        );

        let receiver_id = TUniqueId::new(32, 3);
        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(TUniqueId::new(32, 1), TUniqueId::new(32, 2));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            receiver_id.clone(),
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 8061)),
            None,
        )]);
        sender.params = Some(sender_exec);
        assert_exec_ok(&service, &sender);

        // The sender fragment ran (parked) before the transport was invoked.
        assert_eq!(executor.calls.load(Ordering::Relaxed), 1);
        let spec = fake_transport.join().unwrap();
        assert_eq!(spec.host, "127.0.0.1");
        assert_eq!(spec.brpc_port, 8061);
        assert_eq!(
            spec.slot.fragment_instance_id,
            FragmentInstanceId::from(&receiver_id)
        );
        assert_eq!(spec.slot.node_id, 7);
        assert_eq!(spec.slot.sender_id, 0);
        assert_eq!(spec.names, vec!["id".to_string(), "name".to_string()]);
    }

    /// A transport failure fails the sender's dispatch — the FE sees the error, never a hang.
    #[test]
    fn remote_transmit_failure_fails_the_sender_dispatch() {
        let (requests_tx, requests_rx) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(request) = requests_rx.recv() {
                if let crate::nixl_transport::TransportRequest::SendFragment { respond, .. } =
                    request
                {
                    let _ = respond.send(Err("nixl WRITE timed out".to_string()));
                }
            }
        });
        let service = SiriusComputeNodeService::with_transport(
            Arc::new(CountingExecutor::default()),
            test_identity(),
            Some(crate::nixl_transport::NixlTransport::for_test(requests_tx)),
        );

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(TUniqueId::new(33, 1), TUniqueId::new(33, 2));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            TUniqueId::new(33, 3),
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 8061)),
            None,
        )]);
        sender.params = Some(sender_exec);

        let response = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&sender),
        );
        let result = PExecPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("nixl WRITE timed out"),
            "{:?}",
            result.status.error_msgs
        );
    }

    /// A destination on another CN: same host, a different brpc port.
    fn remote_destination(receiver_id: TUniqueId, brpc_port: i32) -> TPlanFragmentDestination {
        TPlanFragmentDestination::new(
            receiver_id,
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), brpc_port)),
            None,
        )
    }

    /// Collects the specs a fake transport was handed, answering each with `outcome`.
    ///
    /// `recv_timeout` rather than `recv`: a regression that stops sending a destination must
    /// fail the test, never hang the suite.
    fn sent_specs(
        requests: mpsc::Receiver<crate::nixl_transport::TransportRequest>,
        mut outcome: impl FnMut(usize) -> Result<(), String>,
    ) -> Vec<RemoteSendSpec> {
        let mut specs = Vec::new();
        while let Ok(request) =
            requests.recv_timeout(std::time::Duration::from_millis(if specs.is_empty() {
                10_000
            } else {
                1_000
            }))
        {
            match request {
                crate::nixl_transport::TransportRequest::SendFragment { spec, respond } => {
                    let _ = respond.send(outcome(specs.len()));
                    specs.push(spec);
                }
                _ => panic!("the sender flow only sends fragments"),
            }
        }
        specs
    }

    /// Per-destination frame ordering is the invariant the receiver enforces (a seq gap per
    /// exchange+sender fails the query). It holds because each remote destination is sent
    /// exactly once, by one thread, in the FE's destination order. Pin all three properties.
    #[test]
    fn remote_destinations_are_sent_once_each_in_the_fes_order() {
        let (requests_tx, requests_rx) = mpsc::channel();
        let fake_transport = std::thread::spawn(move || sent_specs(requests_rx, |_| Ok(())));
        let service = SiriusComputeNodeService::with_transport(
            Arc::new(CountingExecutor::default()),
            test_identity(),
            Some(crate::nixl_transport::NixlTransport::for_test(requests_tx)),
        );

        let receivers = [
            TUniqueId::new(62, 3),
            TUniqueId::new(62, 4),
            TUniqueId::new(62, 5),
        ];
        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(TUniqueId::new(62, 1), TUniqueId::new(62, 2));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(
            receivers
                .iter()
                .enumerate()
                .map(|(index, id)| remote_destination(id.clone(), 8061 + index as i32))
                .collect(),
        );
        sender.params = Some(sender_exec);
        assert_exec_ok(&service, &sender);

        let specs = fake_transport.join().unwrap();
        assert_eq!(specs.len(), 3, "one send per destination: {specs:?}");
        for (index, (spec, receiver)) in specs.iter().zip(&receivers).enumerate() {
            assert_eq!(
                spec.slot.fragment_instance_id,
                FragmentInstanceId::from(receiver),
                "destination {index} sent out of order: {specs:?}"
            );
            assert_eq!(spec.brpc_port, 8061 + index as u16);
            assert_eq!(spec.slot.node_id, 7);
            assert_eq!(spec.slot.sender_id, 0);
        }
    }

    /// Records what the executor was asked to do on the remote-ingest path.
    #[derive(Debug, Default)]
    struct RecordingExecutor {
        remote_inputs: Mutex<Vec<(i32, i32, Vec<StagedBatch>)>>,
        released: Mutex<Vec<u64>>,
    }

    impl FragmentExecutor for RecordingExecutor {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            self.remote_inputs
                .lock()
                .unwrap()
                .extend(run.remote_inputs.iter().cloned());
            StubExecutor.run(run)
        }

        fn staging_release(&self, offset: u64) -> Result<(), String> {
            self.released.lock().unwrap().push(offset);
            Ok(())
        }
    }

    /// One `transmit_packed` frame as the wire would carry it.
    #[allow(clippy::too_many_arguments)]
    fn transmit_params(
        receiver: &TUniqueId,
        node_id: i32,
        sender_id: i32,
        seq: i64,
        eos: bool,
        offset: u64,
        length: u64,
        names: &[&str],
    ) -> Vec<u8> {
        PTransmitPackedParams {
            finst_id: Some(crate::proto::starrocks::PUniqueId {
                hi: receiver.hi,
                lo: receiver.lo,
            }),
            node_id: Some(node_id),
            sender_id: Some(sender_id),
            eos: Some(eos),
            seq: Some(seq),
            offset: Some(offset),
            length: Some(length),
            column_names: names.iter().map(|name| name.to_string()).collect(),
            canary: None,
            // A fixed per-batch row count on batch frames, so the receiver-side test can assert
            // the count rode the wire into the staged batch; EOS frames carry none.
            rows: if eos { None } else { Some(3) },
        }
        .encode_to_vec()
    }

    /// The receiver registers first (StarRocks dispatch order); remote frames stage batches, the
    /// eos completes the set, and the dispatch worker runs the receiver with the staged batches
    /// as `remote_inputs` — the full receiver half of the nixl tier minus the device WRITE.
    #[test]
    fn transmit_packed_frames_feed_a_dispatched_receiver() {
        let executor = Arc::new(RecordingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(60, 1);
        let receiver_id = TUniqueId::new(60, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id, receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 1);
        receiver.params = Some(receiver_exec);
        assert_exec_ok(&service, &receiver);

        // A data frame: pack metadata in the attachment, payload location in the body.
        let metadata = vec![0xAB; 16];
        let data = route(
            &service,
            methods::TRANSMIT_PACKED,
            transmit_params(&receiver_id, 7, 0, 0, false, 4096, 256, &["id", "name"]),
            metadata.clone(),
        );
        let data = PTransmitPackedResult::decode(data.body.as_slice()).unwrap();
        assert_eq!(
            data.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            data.status.error_msgs
        );

        // The eos frame (no attachment) completes the sender set and dispatches the receiver.
        let eos = route(
            &service,
            methods::TRANSMIT_PACKED,
            transmit_params(&receiver_id, 7, 0, 1, true, 0, 0, &["id", "name"]),
            Vec::new(),
        );
        let eos = PTransmitPackedResult::decode(eos.body.as_slice()).unwrap();
        assert_eq!(
            eos.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            eos.status.error_msgs
        );

        let fetched = fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(!fetched.attachment.is_empty());
        assert_eq!(
            executor.remote_inputs.lock().unwrap().as_slice(),
            &[(
                7,
                0,
                vec![StagedBatch {
                    metadata,
                    offset: 4096,
                    len: 256,
                    rows: Some(3),
                }],
            )],
            "the dispatched receiver consumed exactly the staged batch, row count included"
        );
    }

    /// A lost frame must fail the exchange loudly — silently dropping rows is this subsystem's
    /// cardinal sin.
    #[test]
    fn transmit_packed_sequence_gap_is_an_internal_error() {
        let service = SiriusComputeNodeService::new();
        let receiver_id = TUniqueId::new(61, 2);
        let first = route(
            &service,
            methods::TRANSMIT_PACKED,
            transmit_params(&receiver_id, 7, 0, 0, false, 0, 8, &["id"]),
            vec![1u8; 8],
        );
        let first = PTransmitPackedResult::decode(first.body.as_slice()).unwrap();
        assert_eq!(first.status.status_code, TStatusCode::OK.0);

        let gapped = route(
            &service,
            methods::TRANSMIT_PACKED,
            transmit_params(&receiver_id, 7, 0, 2, false, 0, 8, &["id"]),
            vec![2u8; 8],
        );
        let gapped = PTransmitPackedResult::decode(gapped.body.as_slice()).unwrap();
        assert_eq!(gapped.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            gapped.status.error_msgs[0].contains("skipped from frame seq 1 to 2"),
            "{:?}",
            gapped.status.error_msgs
        );
    }

    /// The bandwidth canary's lease release skips the rendezvous and the engine's input streams.
    #[test]
    fn transmit_packed_canary_releases_the_lease_without_touching_the_rendezvous() {
        let executor = Arc::new(RecordingExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let params = PTransmitPackedParams {
            canary: Some(true),
            offset: Some(7777),
            length: Some(16 << 20),
            ..Default::default()
        };
        let response = route(
            &service,
            methods::TRANSMIT_PACKED,
            params.encode_to_vec(),
            Vec::new(),
        );
        let result = PTransmitPackedResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(
            result.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            result.status.error_msgs
        );
        assert_eq!(executor.released.lock().unwrap().as_slice(), &[7777]);
        assert!(executor.remote_inputs.lock().unwrap().is_empty());
    }

    /// Without a staging arena (stub executor default), a canary release must fail loudly.
    #[test]
    fn transmit_packed_canary_without_an_arena_is_an_internal_error() {
        let service = SiriusComputeNodeService::new();
        let params = PTransmitPackedParams {
            canary: Some(true),
            offset: Some(0),
            ..Default::default()
        };
        let response = route(
            &service,
            methods::TRANSMIT_PACKED,
            params.encode_to_vec(),
            Vec::new(),
        );
        let result = PTransmitPackedResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("staging arena"),
            "{:?}",
            result.status.error_msgs
        );
    }

    /// Without a transport, `exchange_nixl_md` names the remedy instead of pretending.
    #[test]
    fn exchange_nixl_md_without_transport_is_an_internal_error() {
        let service = SiriusComputeNodeService::new();
        let request = PExchangeNixlMd {
            agent_name: Some("127.0.0.1:8062".to_string()),
            agent_metadata: Some(vec![1, 2, 3]),
        };
        let response = route(
            &service,
            methods::EXCHANGE_NIXL_MD,
            request.encode_to_vec(),
            Vec::new(),
        );
        let result = PExchangeNixlMdResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("SIRIUS_EXCHANGE_STAGING_BYTES"),
            "{:?}",
            result.status.error_msgs
        );
    }

    /// Without an arena, a peer's lease request fails loudly (stub executor default).
    #[test]
    fn request_staging_lease_without_an_arena_is_an_internal_error() {
        let service = SiriusComputeNodeService::new();
        let response = route(
            &service,
            methods::REQUEST_STAGING_LEASE,
            PStagingLeaseRequest { length: 1024 }.encode_to_vec(),
            Vec::new(),
        );
        let result = PStagingLeaseResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("staging arena"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn data_stream_sink_destination_without_brpc_server_is_a_loud_error() {
        // The FE always sets brpc_server on a destination; a missing one is malformed dispatch,
        // not an implicit "local".
        let service = SiriusComputeNodeService::new();

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(TUniqueId::new(31, 1), TUniqueId::new(31, 2));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![TPlanFragmentDestination::new(
            TUniqueId::new(31, 3),
            None,
            None,
            None,
        )]);
        sender.params = Some(sender_exec);

        let response = route(
            &service,
            methods::EXEC_PLAN_FRAGMENT,
            PExecPlanFragmentRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&sender),
        );
        let result = PExecPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("has no brpc_server address"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn sender_rpc_returns_before_the_dispatched_receiver_executes() {
        // The last sender's exec_plan_fragment must hand the ready receiver to the dispatch
        // worker and return, not block on the receiver's whole execution. The gate holds the
        // receiver open; under the old inline design the sender RPC could not return until the
        // gate released.
        let (release, gate) = std::sync::mpsc::channel();
        let executor = Arc::new(GatedExecutor {
            release: Mutex::new(gate),
            receiver_ran: std::sync::atomic::AtomicBool::new(false),
        });
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(40, 1);
        let receiver_id = TUniqueId::new(40, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id.clone(), receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 1);
        receiver.params = Some(receiver_exec);
        assert_exec_ok(&service, &receiver);

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(query_id, TUniqueId::new(40, 3));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![local_destination(receiver_id.clone())]);
        sender.params = Some(sender_exec);
        assert_exec_ok(&service, &sender);

        // The sender RPC returned while the receiver is still gated.
        assert!(!executor.receiver_ran.load(Ordering::SeqCst));

        release.send(()).unwrap();
        let fetched = fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(!fetched.attachment.is_empty());
        assert!(executor.receiver_ran.load(Ordering::SeqCst));
    }

    #[test]
    fn dispatched_receiver_failure_surfaces_through_fetch_data() {
        // The receiver fails on the dispatch worker, after every RPC already returned OK — the
        // only remaining signal path to the FE is the fetch_data poll, which must report the
        // cause instead of waiting forever.
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FailingReceiverExecutor),
            test_identity(),
        );
        let query_id = TUniqueId::new(50, 1);
        let receiver_id = TUniqueId::new(50, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id.clone(), receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 1);
        receiver.params = Some(receiver_exec);
        assert_exec_ok(&service, &receiver);

        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut sender_exec = exec_params(query_id, TUniqueId::new(50, 3));
        sender_exec.sender_id = Some(0);
        sender_exec.destinations = Some(vec![local_destination(receiver_id.clone())]);
        sender.params = Some(sender_exec);
        assert_exec_ok(&service, &sender);

        let result = fetch_error_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("failed: receiver exploded on the GPU"),
            "{:?}",
            result.status.error_msgs
        );
        assert_eq!(result.eos, Some(true));
    }

    /// Builds the three-fragment chain used by the failure-propagation tests: a result-sink root
    /// reading exchange 9, a middle receiver reading exchange 7 and sinking into 9, and a leaf
    /// scan sinking into 7. All three share `query_id`.
    fn propagation_chain(
        query_id: &TUniqueId,
        root_id: &TUniqueId,
        middle_id: &TUniqueId,
        leaf_id: &TUniqueId,
    ) -> (
        TExecPlanFragmentParams,
        TExecPlanFragmentParams,
        TExecPlanFragmentParams,
    ) {
        let mut root = fragment_params(Some(exchange_plan(9, 0)), Some(desc_table()));
        root.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut root_exec = exec_params(query_id.clone(), root_id.clone());
        root_exec.per_exch_num_senders.insert(9, 1);
        root.params = Some(root_exec);

        let mut middle = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        middle.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(9));
        let mut middle_exec = exec_params(query_id.clone(), middle_id.clone());
        middle_exec.per_exch_num_senders.insert(7, 1);
        middle_exec.sender_id = Some(0);
        middle_exec.destinations = Some(vec![local_destination(root_id.clone())]);
        middle.params = Some(middle_exec);

        let mut leaf = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        leaf.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(7));
        let mut leaf_exec = exec_params(query_id.clone(), leaf_id.clone());
        leaf_exec.sender_id = Some(0);
        leaf_exec.destinations = Some(vec![local_destination(middle_id.clone())]);
        leaf.params = Some(leaf_exec);

        (root, middle, leaf)
    }

    #[test]
    fn intermediate_fragment_failure_fails_the_fe_polled_result_id() {
        // The middle fragment fails on the dispatch worker. Its own instance id is not polled by
        // anyone; the failure must land on the query's result-fragment id — the one the FE's
        // fetch_data long-poll is blocked on — carrying the original error.
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FailingIntermediateExecutor),
            test_identity(),
        );
        let query_id = TUniqueId::new(70, 1);
        let root_id = TUniqueId::new(70, 2);
        let middle_id = TUniqueId::new(70, 3);
        let (root, middle, leaf) =
            propagation_chain(&query_id, &root_id, &middle_id, &TUniqueId::new(70, 4));

        assert_exec_ok(&service, &root);
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);

        let result = fetch_error_eventually(&service, root_id.hi, root_id.lo);
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        let message = &result.status.error_msgs[0];
        assert!(
            message.contains(&FragmentInstanceId::from(&middle_id).to_string())
                && message.contains("intermediate receiver exploded on the GPU"),
            "{message}"
        );
        assert_eq!(result.eos, Some(true));
    }

    #[test]
    fn intermediate_failure_before_result_registration_still_fails_the_result_poll() {
        // The ordering race: the middle fragment fails before the FE's result fragment ever
        // registers on this CN. The failure must be recorded at query level so the result
        // fragment fails on arrival instead of waiting on a sender that will never deliver.
        let service = SiriusComputeNodeService::with_executor(
            Arc::new(FailingIntermediateExecutor),
            test_identity(),
        );
        let query_id = TUniqueId::new(71, 1);
        let root_id = TUniqueId::new(71, 2);
        let middle_id = TUniqueId::new(71, 3);
        let (root, middle, leaf) =
            propagation_chain(&query_id, &root_id, &middle_id, &TUniqueId::new(71, 4));

        // Middle and leaf only: the middle fails on the dispatch worker while the result
        // fragment is still undelivered. Its parked error doubles as the "failure landed" gate.
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);
        let parked = fetch_error_eventually(&service, middle_id.hi, middle_id.lo);
        assert!(
            parked.status.error_msgs[0].contains("intermediate receiver exploded on the GPU"),
            "{:?}",
            parked.status.error_msgs
        );

        // The result fragment arrives after the failure; its very first poll must report it.
        assert_exec_ok(&service, &root);
        let result = fetch_error_eventually(&service, root_id.hi, root_id.lo);
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        let message = &result.status.error_msgs[0];
        assert!(
            message.contains(&FragmentInstanceId::from(&middle_id).to_string())
                && message.contains("intermediate receiver exploded on the GPU"),
            "{message}"
        );
    }

    #[test]
    fn cancel_plan_fragment_returns_ok_and_unblocks_a_waiting_result_poll() {
        // The route must exist (an unrouted method returns a PRPC-level error frame that poisons
        // the FE's shared channel) and, best-effort, fail the waiting entry so a fetch_data
        // long-poll returns instead of running out its timeout.
        let service = SiriusComputeNodeService::new();
        let query_id = TUniqueId::new(80, 1);
        let receiver_id = TUniqueId::new(80, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id, receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 1);
        receiver.params = Some(receiver_exec);
        assert_exec_ok(&service, &receiver);

        let response = route(
            &service,
            methods::CANCEL_PLAN_FRAGMENT,
            PCancelPlanFragmentRequest {
                finst_id: PUniqueId {
                    hi: receiver_id.hi,
                    lo: receiver_id.lo,
                },
                cancel_reason: None,
                is_pipeline: None,
                query_id: None,
                error_message: Some("exceed big query cpu limit".to_string()),
            }
            .encode_to_vec(),
            Vec::new(),
        );
        let cancel = PCancelPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(
            cancel.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            cancel.status.error_msgs
        );

        let result = fetch_error_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(
            result.status.error_msgs[0].contains("cancelled by the FE")
                && result.status.error_msgs[0].contains("exceed big query cpu limit"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn cancel_plan_fragment_for_an_unknown_instance_is_ok_and_fabricates_nothing() {
        let service = SiriusComputeNodeService::new();
        let response = route(
            &service,
            methods::CANCEL_PLAN_FRAGMENT,
            PCancelPlanFragmentRequest {
                finst_id: PUniqueId { hi: 81, lo: 1 },
                cancel_reason: None,
                is_pipeline: None,
                query_id: None,
                error_message: None,
            }
            .encode_to_vec(),
            Vec::new(),
        );
        let cancel = PCancelPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(cancel.status.status_code, TStatusCode::OK.0);

        // The id stays unknown: a later poll reports "no buffered result", not a cancel error.
        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(81, 1),
            Vec::new(),
        );
        let fetched = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            fetched.status.error_msgs[0].contains("no buffered result"),
            "{:?}",
            fetched.status.error_msgs
        );
    }

    #[test]
    fn cached_descriptor_reference_reuses_query_descriptor_table() {
        let service = SiriusComputeNodeService::new();
        let query_id = TUniqueId::new(4, 2);

        let mut initial = fragment_params(None, Some(desc_table()));
        initial.params = Some(exec_params(query_id.clone(), TUniqueId::new(4, 3)));
        service
            .core
            .resolve_descriptor_table(&initial)
            .expect("cache initial descriptor table");

        let cached = TDescriptorTable::new(None, Vec::new(), None, Some(true));
        let mut reference = fragment_params(None, Some(cached));
        reference.params = Some(exec_params(query_id, TUniqueId::new(4, 4)));
        let resolved = service
            .core
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

        let err = service
            .core
            .resolve_descriptor_table(&reference)
            .unwrap_err();
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
    fn exec_plan_fragment_rejects_unhandled_output_sink() {
        // Accepting a sink this CN does not implement would discard the fragment's output and
        // hang every consumer; the dispatch must fail and name the sink. MULTI_CAST_DATA_STREAM_
        // SINK is the real case (CTE reuse), so it stands in for the whole class here.
        let service = SiriusComputeNodeService::new();
        let mut params = supported_fragment();
        params.fragment.as_mut().unwrap().output_sink =
            Some(sink_of_type(TDataSinkType::MULTI_CAST_DATA_STREAM_SINK));
        params.params = Some(exec_params(TUniqueId::new(0, 11), TUniqueId::new(0, 12)));

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
            result.status.error_msgs[0].contains("MULTI_CAST_DATA_STREAM_SINK")
                && result.status.error_msgs[0].contains("does not support"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[test]
    fn exec_plan_fragment_rejects_data_stream_sink_without_payload() {
        // A DATA_STREAM_SINK with no stream_sink payload has no destinations to route to; it is
        // a malformed dispatch, not a fragment whose output may be dropped.
        let service = SiriusComputeNodeService::new();
        let mut params = supported_fragment();
        params.fragment.as_mut().unwrap().output_sink =
            Some(sink_of_type(TDataSinkType::DATA_STREAM_SINK));
        params.params = Some(exec_params(TUniqueId::new(0, 13), TUniqueId::new(0, 14)));

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
            result.status.error_msgs[0].contains("no stream_sink payload"),
            "{:?}",
            result.status.error_msgs
        );
    }

    #[tokio::test]
    async fn get_file_schema_attachment_infers_across_multiple_ranges() {
        let message = "message m { optional int64 a; optional binary b (UTF8); }";
        let first = write_parquet("svc_multi_a", message);
        let second = write_parquet("svc_multi_b", message);
        let attachment = serialize_binary(&file_schema_request(vec![
            broker_range(&first, TFileFormatType::FORMAT_PARQUET),
            broker_range(&second, TFileFormatType::FORMAT_PARQUET),
        ]));
        let schema = SiriusComputeNodeService::file_schema_from_attachment(&attachment).await;
        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
        let schema = schema.unwrap();
        assert_eq!(schema.len(), 2);
        assert_eq!(schema[0].col_name, "a");
        assert_eq!(schema[1].col_name, "b");
    }

    #[tokio::test]
    async fn get_file_schema_attachment_rejects_non_parquet_range_by_path() {
        let message = "message m { optional int64 a; }";
        let first = write_parquet("svc_fmt_a", message);
        let second = write_parquet("svc_fmt_b", message);
        // The non-parquet range hides behind a parquet one; the error must still name it.
        let attachment = serialize_binary(&file_schema_request(vec![
            broker_range(&first, TFileFormatType::FORMAT_PARQUET),
            broker_range(&second, TFileFormatType::FORMAT_CSV_PLAIN),
        ]));
        let result = SiriusComputeNodeService::file_schema_from_attachment(&attachment).await;
        std::fs::remove_file(&first).ok();
        std::fs::remove_file(&second).ok();
        assert!(
            result.as_ref().is_err_and(|err| {
                err.contains("unsupported file format") && err.contains(second.to_str().unwrap())
            }),
            "{result:?}"
        );
    }

    fn broker_range(path: &std::path::Path, format: TFileFormatType) -> TBrokerRangeDesc {
        TBrokerRangeDesc::new(
            TFileType::FILE_LOCAL,
            format,
            false,
            path.to_str().unwrap().to_string(),
            0,
            0,
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

    /// Wraps FILES() ranges the way the FE does: one broker scan range inside a TScanRange.
    fn file_schema_request(ranges: Vec<TBrokerRangeDesc>) -> TGetFileSchemaRequest {
        let params = TBrokerScanRangeParams::new(
            b'\t' as i8,
            b'\n' as i8,
            0,
            Vec::new(),
            0,
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
        );
        let broker = TBrokerScanRange::new(ranges, params, Vec::new(), None, None, None, None);
        TGetFileSchemaRequest::new(
            TScanRange::new(None, None, Some(broker), None, None, None),
            None,
        )
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
        sink_of_type(TDataSinkType::RESULT_SINK)
    }

    /// A sink carrying nothing but its type, for the paths that only classify the type.
    fn sink_of_type(sink_type: TDataSinkType) -> TDataSink {
        TDataSink::new(
            sink_type, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None,
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
