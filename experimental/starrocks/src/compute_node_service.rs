use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};

#[cfg(test)]
use crate::fragment_executor::StubExecutor;
use crate::fragment_executor::{
    FragmentExecutor, FragmentLabel, FragmentResult, FragmentRun, RetireTrigger, SenderSlot,
    StagedBatch,
};
use crate::local_exchange::{
    ExchangeKey, FuseOffer, LocalExchange, LocalPlan, ReadyExchangeInput, ReadyFragment,
    SenderSource,
};
use crate::nixl_transport::{NixlTransport, RemoteSendSpec};
use crate::proto::starrocks::{
    PCancelPlanFragmentRequest, PCancelPlanFragmentResult, PExchangeNixlMd, PExchangeNixlMdResult,
    PExecBatchPlanFragmentsRequest, PExecBatchPlanFragmentsResult, PExecPlanFragmentRequest,
    PExecPlanFragmentResult, PFetchDataRequest, PFetchDataResult, PGetFileSchemaRequest,
    PGetFileSchemaResult, PPlanFragmentCancelReason, PSlotDescriptor, PStagingLeaseRequest,
    PStagingLeaseResult, PTransmitPackedParams, PTransmitPackedResult, StatusPb,
    p_internal_service_brpc::PInternalService,
};
use crate::result_encoder::{self, ThriftBinary};
use crate::result_store::{FetchOutcome, FragmentInstanceId, ResultStore};
use crate::tunable::{FusionMode, Tunables};
use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslatedPlan, fusion};
use starrocks_thrift::{
    data_sinks::{TDataSinkType, TPlanFragmentDestination, TResultSinkType},
    descriptors::TDescriptorTable,
    internal_service::{
        TExecBatchPlanFragmentsParams, TExecPlanFragmentParams, TGetFileSchemaRequest,
    },
    partitions::TPartitionType,
    plan_nodes::TFileFormatType,
    status_code::TStatusCode,
    types::TNetworkAddress,
};
use thrift::{
    protocol::{TBinaryInputProtocol, TSerializable},
    transport::TBufferChannel,
};
use tracing::{debug, info, instrument};

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

    /// `host:brpc_port`, the name this CN's run log lines carry so one query's halves on
    /// different CNs can be told apart.
    fn endpoint(&self) -> String {
        format!("{}:{}", self.host, self.brpc_port)
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
    /// Which same-node senders are deferred into their receiver's plan instead of running, as a
    /// [`FusionMode`] code. From `SIRIUS_CN_FRAGMENT_FUSION` at bring-up (or a test's setter);
    /// see [`ServiceCore::try_defer_sender`].
    fragment_fusion: std::sync::atomic::AtomicU8,
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
            fragment_fusion: std::sync::atomic::AtomicU8::new(Tunables::get().fusion_mode.code()),
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

    /// Overrides the bring-up fusion mode for this service instance; tests use it to exercise
    /// every mode without touching the process environment.
    #[cfg(test)]
    pub(crate) fn set_fragment_fusion(&self, mode: FusionMode) {
        self.core
            .fragment_fusion
            .store(mode.code(), std::sync::atomic::Ordering::Relaxed);
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

    /// Tears the cancelled query down on this CN. Always answers OK so the FE's shared
    /// jprotobuf channel stays healthy — the default unrouted reply is a PRPC-level error frame,
    /// and the FE reaps the timed-out future in a way that misattributes later replies on the
    /// channel.
    ///
    /// Per instance: a still-waiting result entry is failed so a `fetch_data` long-poll returns
    /// now, and the instance leaves the rendezvous with its staged leases released. Per query
    /// (when the FE sent the id): a failure reason records a query-level failure so later
    /// fragments of the query are refused on arrival, and every reason retires the query's
    /// parked sender output on the executor. What it still cannot do is abort a fragment already
    /// inside `run()`: that one finishes first, and its output is dropped when it ends.
    #[instrument(skip_all)]
    async fn cancel_plan_fragment(
        &self,
        request: PCancelPlanFragmentRequest,
        _attachment: Vec<u8>,
    ) -> Result<crate::prpc::Reply<PCancelPlanFragmentResult>, crate::prpc::Error> {
        let id = FragmentInstanceId::from(&request.finst_id);
        let query_id = request.query_id.as_ref().map(FragmentInstanceId::from);
        let reason_code = request
            .cancel_reason
            .and_then(|code| PPlanFragmentCancelReason::try_from(code).ok());
        let reason_name = reason_code.map_or("none", |code| code.as_str_name());
        let mut reason = format!("fragment instance {id} was cancelled by the FE");
        if let Some(message) = request.error_message.as_ref().filter(|msg| !msg.is_empty()) {
            reason = format!("{reason}: {message}");
        }
        self.core.results.cancel(id, reason.clone());
        if let Some(query_id) = query_id {
            // QUERY_FINISHED / LIMIT_REACH arrive after eos on every successful multi-instance
            // query: they clean up parked and rendezvous state but must not fail the query's
            // result entries (a repeat fetch_data still reports EOS).
            let finished = matches!(
                reason_code,
                Some(
                    PPlanFragmentCancelReason::QueryFinished
                        | PPlanFragmentCancelReason::LimitReach
                )
            );
            if !finished {
                self.core.results.cancel_query(query_id, reason.clone());
            }
            // Phased schedules cancel with a dummy instance (0, 0): nothing to retire by
            // instance, everything by query.
            let released_leases = if id == FragmentInstanceId::from_halves(0, 0) {
                0
            } else {
                self.core
                    .release_sources(self.core.exchanges.retire_receiver(id))
            };
            if let Err(err) = self.core.executor.retire_query(
                query_id,
                RetireTrigger::Cancel(reason_name.to_string()),
                &reason,
            ) {
                tracing::warn!(
                    %query_id,
                    error = %err,
                    "could not retire the cancelled query's parked output"
                );
            }
            info!(
                %query_id,
                fragment_instance_id = %id,
                reason = reason_name,
                released_leases,
                "cancel_plan_fragment retired the query on this CN"
            );
        } else {
            info!(
                fragment_instance_id = %id,
                reason = reason_name,
                "cancel_plan_fragment without a query id: failed the waiting result entry only"
            );
        }
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
        self.process_inline(&params)
    }

    /// Runs one FE-dispatched fragment on the RPC path: processed here, with the receivers it
    /// readied handed to the dispatch worker. A failure is also recorded at query level
    /// (`fail_fragment`), so a result instance of the query reserved on this CN reports the
    /// real cause on its first poll instead of the fetch_data timeout, and the query's later
    /// fragments are refused. Any RPC error fails the query on the FE, so marking here is never
    /// premature. Survey mode returns `Ok` and is unaffected.
    fn process_inline(&self, params: &TExecPlanFragmentParams) -> std::result::Result<(), String> {
        let outcome = self
            .core
            .process_fragment(params)
            .and_then(|outcome| self.dispatch_ready(outcome));
        if let Err(err) = &outcome
            && let (Some(id), Some(query_id)) = (
                ServiceCore::fragment_instance_id(params),
                ServiceCore::query_id(params),
            )
        {
            self.core.fail_fragment(id, query_id, err.clone());
        }
        outcome
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
        // A frame for a receiver the FE already cancelled here: the peer's drain is still
        // running and must complete quietly (it gets OK, and its own drop_parked frees the
        // sender side), but nothing may re-enter the rendezvous. Release the lease it landed in.
        if self.core.exchanges.is_retired(key.fragment_instance_id) {
            if let Some(batch) = &batch
                && batch.len > 0
            {
                self.core.executor.staging_release(batch.offset)?;
            }
            info!(
                receiver_fragment_instance_id = %key.fragment_instance_id,
                stream_id = key.node_id,
                sender_id,
                seq,
                eos,
                "released a remote frame for a retired receiver"
            );
            return Ok(());
        }
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

/// Staged remote batches this CN holds for one fragment, as `(node id, sender id, batches)`.
///
/// Released on drop unless handed to the engine with [`take`](Self::take): the engine releases
/// each lease after pushing it, or in its own sweep on a failed run. Every pre-run error path --
/// a names mismatch, a translation failure, a sink the CN refuses -- therefore returns the leases
/// to the arena instead of pinning it for the process lifetime (arena exhaustion is a hard
/// failure on N CNs).
struct StagedLeases<'e> {
    batches: Vec<(i32, i32, Vec<StagedBatch>)>,
    executor: &'e dyn FragmentExecutor,
}

impl<'e> StagedLeases<'e> {
    fn new(executor: &'e dyn FragmentExecutor) -> Self {
        Self {
            batches: Vec::new(),
            executor,
        }
    }

    fn push(&mut self, node_id: i32, sender_id: i32, batches: Vec<StagedBatch>) {
        self.batches.push((node_id, sender_id, batches));
    }

    /// Hands the batches to the caller; the guard then releases nothing.
    fn take(&mut self) -> Vec<(i32, i32, Vec<StagedBatch>)> {
        std::mem::take(&mut self.batches)
    }
}

impl Drop for StagedLeases<'_> {
    fn drop(&mut self) {
        for (_, _, batches) in &self.batches {
            release_leases(self.executor, batches);
        }
    }
}

/// Returns every lease among `batches` to the arena, warning on failure; returns the count
/// released. `len == 0` batches never held a lease (metadata-only, `StagedBatch` contract).
fn release_leases<'a>(
    executor: &dyn FragmentExecutor,
    batches: impl IntoIterator<Item = &'a StagedBatch>,
) -> usize {
    let mut released = 0;
    for batch in batches {
        if batch.len == 0 {
            continue;
        }
        match executor.staging_release(batch.offset) {
            Ok(()) => released += 1,
            Err(err) => tracing::warn!(
                offset = batch.offset,
                error = %err,
                "failed to release a staged lease of a retired query"
            ),
        }
    }
    released
}

/// A ready receiver after its deferred sender plans were spliced in
/// ([`ServiceCore::fold_deferred_plans`]).
struct FoldedReceiver {
    /// The plan the engine runs: the receiver's params with every deferred sender inline.
    params: TExecPlanFragmentParams,
    /// The exchanges that still read a stream, in node-id order.
    streamed: Vec<ReadyExchangeInput>,
    /// The absorbed senders' fragment instance ids.
    fused: Vec<FragmentInstanceId>,
}

impl ServiceCore {
    /// Runs one dispatched receiver, parking a failure where `fetch_data` can see it. Returns
    /// the next receiver when this fragment's own sink completed another sender set.
    fn run_ready_fragment(&self, ready: ReadyFragment) -> Vec<ReadyFragment> {
        let id = Self::fragment_instance_id(&ready.params);
        let query_id = Self::query_id(&ready.params);
        // Gate 3: a queued fragment of a query this CN already failed is skipped before
        // translation -- no `fragment run started`, no GPU work, no follow-on receivers. Its
        // staged remote leases go back to the arena; its local senders' parked output is
        // dropped by the engine's retire of the query.
        if let Some(query_id) = query_id
            && let Some(cause) = self.results.failure_of(query_id)
        {
            let released_leases = self.release_staged(ready.inputs);
            info!(
                %query_id,
                fragment_instance_id = ?id,
                %cause,
                released_leases,
                "skipping fragment of a retired query"
            );
            return Vec::new();
        }
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
                        // Fails this id, every reserved result instance of the query, records
                        // the failure so a later fragment of the query is refused, and retires
                        // the query's parked output on the executor.
                        self.fail_fragment(id, query_id, error);
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

    /// Records a fragment failure at query level and retires the query's parked output on the
    /// executor. `fail_query` reaches every result instance the FE polls on this CN (and fails a
    /// result fragment that reserves later, on arrival); `retire_query` drops the query's parked
    /// sender outputs and refuses its later runs. Idempotent against the engine's own retire, so
    /// it doubles as the belt for an engine `Err` and is the whole fix for a failure the engine
    /// never sees (a receiver whose translation fails after its senders parked).
    fn fail_fragment(&self, id: FragmentInstanceId, query_id: FragmentInstanceId, error: String) {
        self.results.fail_query(query_id, id, error.clone());
        if let Err(err) = self
            .executor
            .retire_query(query_id, RetireTrigger::CnErr, &error)
        {
            tracing::warn!(
                %query_id,
                error = %err,
                "could not retire the failed query's parked output"
            );
        }
    }

    /// Releases the staged leases of a ready fragment that will not run; returns the count.
    /// Parked local slots need nothing here: the engine's retire drops them by query.
    fn release_staged(&self, inputs: Vec<ReadyExchangeInput>) -> usize {
        inputs
            .into_iter()
            .map(|input| self.release_sources(input.sources))
            .sum()
    }

    /// [`release_staged`](Self::release_staged) over a flat source list. Non-exhaustive on
    /// purpose: only a `Remote` source holds arena leases, and a source kind that holds no GPU
    /// memory must not need an arm here.
    fn release_sources(&self, sources: Vec<SenderSource>) -> usize {
        sources
            .iter()
            .map(|source| match source {
                SenderSource::Remote { batches, .. } => {
                    release_leases(self.executor.as_ref(), batches)
                }
                _ => 0,
            })
            .sum()
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
        // Gate 4: a fragment of a query this CN already failed is refused on arrival. A result
        // fragment keeps the reserve-then-fail contract -- the FE's fetch_data long-poll on its
        // id reports the cause on the first poll -- without registering a receiver that can
        // never complete; anything else is an RPC error, which the FE ignores while it is
        // already cancelling the query.
        if let Some(query_id) = Self::query_id(&params)
            && let Some(cause) = self.results.failure_of(query_id)
        {
            if Self::is_mysql_result_sink(&params)?
                && let Some(id) = Self::fragment_instance_id(&params)
            {
                self.results.reserve(id, query_id);
                return Ok(FragmentOutcome::default());
            }
            return Err(format!(
                "query {query_id} already failed on this CN: {cause}"
            ));
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
        // A leaf whose only destination is a pending local receiver is spliced into that
        // receiver's plan instead of running; the receiver it readies (if any) is the caller's
        // to dispatch, exactly like one a parked sender completed.
        if let Some(ready) = self.try_defer_sender(&params)? {
            return Ok(FragmentOutcome::from_ready(ready));
        }

        let translated = self.translate_fragment_logged(&params, dump_seq)?;
        self.execute_fragment(&params, translated)
    }

    /// The fusion mode in force for this service.
    fn fragment_fusion(&self) -> FusionMode {
        FusionMode::from_code(
            self.fragment_fusion
                .load(std::sync::atomic::Ordering::Relaxed),
        )
        .expect("fragment_fusion only ever holds a FusionMode code")
    }

    /// Tries to defer a sender into its local receiver's plan instead of running it. Returns the
    /// receivers this completed (for the caller to dispatch), or `None` when the sender must
    /// take today's run-and-park path. Every check runs here, with both fragments' params in
    /// hand, before anything is deferred: a decline is logged and never an error, so no shape
    /// that runs today can fail because of fusion.
    ///
    /// Policy exclusions (mode off, a dead query, not a leaf, partition type, remote
    /// destination, a sink the translator will not splice) log at debug: every broadcast leaf
    /// hits one on every query. Rendezvous and structural declines log at info: those are the
    /// arrival-order and plan-shape regressions the acceptance runs count.
    fn try_defer_sender(
        &self,
        params: &TExecPlanFragmentParams,
    ) -> std::result::Result<Option<Vec<ReadyFragment>>, String> {
        let shown = |id: Option<FragmentInstanceId>| id.map(tracing::field::display);
        let mode = self.fragment_fusion();
        let query_id = Self::query_id(params);
        let sender = Self::fragment_instance_id(params);
        if mode == FusionMode::Off {
            debug!(
                query_id = shown(query_id),
                sender_fragment_instance_id = shown(sender),
                reason = %"off",
                "fragment fusion skipped"
            );
            return Ok(None);
        }
        // A leaf of a query this CN already failed is never deferred. `process_fragment`'s
        // gate 4 refuses such a leaf before it reaches this hook; the check stays with the hook
        // so a second call site (perf/profile-sf1000's queued sender path reaches it before
        // gate 4) cannot defer a dead query's leaf. Declining sends the leaf down today's path,
        // where gate 3 in `run_ready_fragment` skips it without translating.
        if let Some(query_id) = query_id
            && let Some(cause) = self.results.failure_of(query_id)
        {
            debug!(
                %query_id,
                sender_fragment_instance_id = shown(sender),
                %cause,
                reason = %"query already failed on this CN",
                "fragment fusion skipped"
            );
            return Ok(None);
        }
        let shape = match fusion::sender_shape(params) {
            Ok(shape) => shape,
            Err(refusal) => {
                debug!(
                    query_id = shown(query_id),
                    sender_fragment_instance_id = shown(sender),
                    reason = %refusal,
                    "fragment fusion skipped"
                );
                return Ok(None);
            }
        };
        // `sender_shape` read the exec params, so both ids are present from here on.
        let (Some(query_id), Some(sender)) = (query_id, sender) else {
            debug!(
                reason = %"sender carries no exec params",
                "fragment fusion skipped"
            );
            return Ok(None);
        };
        // Policy: leaves only (a middle fragment is the later `all` mode), and in `leaf`
        // mode only the shuffle shape, which is the one that parks a fact table whole at 1 CN.
        if !shape.is_leaf {
            debug!(
                %query_id,
                sender_fragment_instance_id = %sender,
                reason = %"sender has exchange inputs",
                "fragment fusion skipped"
            );
            return Ok(None);
        }
        if mode == FusionMode::Leaf && shape.partition != TPartitionType::HASH_PARTITIONED {
            debug!(
                %query_id,
                sender_fragment_instance_id = %sender,
                reason = %"leaf mode fuses HASH_PARTITIONED sinks only",
                "fragment fusion skipped"
            );
            return Ok(None);
        }
        if !matches!(
            self.route_destination(shape.destination)?,
            DestinationRoute::Local
        ) {
            debug!(
                %query_id,
                sender_fragment_instance_id = %sender,
                reason = %"remote destination",
                "fragment fusion skipped"
            );
            return Ok(None);
        }
        let key = ExchangeKey {
            fragment_instance_id: FragmentInstanceId::from(&shape.destination.fragment_instance_id),
            node_id: shape.dest_node_id,
        };
        let plan = LocalPlan {
            params: params.clone(),
            inputs: Vec::new(),
        };
        let offer = self
            .exchanges
            .offer_local_plan(key, shape.sender_id, plan, |receiver| {
                fusion::fusable_edge(receiver, key.node_id, params)
            })?;
        match offer {
            FuseOffer::Fused(ready) => {
                info!(
                    %query_id,
                    sender_fragment_instance_id = %sender,
                    receiver_fragment_instance_id = %key.fragment_instance_id,
                    exchange = key.node_id,
                    mode = ?mode,
                    "fused sender fragment into its local receiver"
                );
                Ok(Some(ready.into_iter().collect()))
            }
            FuseOffer::Declined { skip, .. } => {
                info!(
                    %query_id,
                    sender_fragment_instance_id = %sender,
                    receiver_fragment_instance_id = %key.fragment_instance_id,
                    exchange = key.node_id,
                    reason = %skip,
                    "fragment fusion skipped"
                );
                Ok(None)
            }
        }
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
        self.execute_fragment_with_inputs(
            params,
            translated,
            Vec::new(),
            StagedLeases::new(self.executor.as_ref()),
        )
    }

    /// Executes a result fragment, or runs a data-stream sender and parks its output on the GPU
    /// for its local receiver (transmitting it when the receiver is remote). `inputs` names the
    /// parked sender outputs this fragment consumes; `leases` holds the staged remote batches and
    /// is only emptied into the engine call, so every validation `Err` before it releases them.
    /// A sender that completes its receiver's sender set returns that receiver for the caller
    /// to run or dispatch.
    fn execute_fragment_with_inputs(
        &self,
        params: &TExecPlanFragmentParams,
        translated: TranslatedPlan,
        inputs: Vec<(i32, Vec<SenderSlot>)>,
        mut leases: StagedLeases<'_>,
    ) -> std::result::Result<FragmentOutcome, String> {
        if Self::is_mysql_result_sink(params)? {
            let id = Self::fragment_instance_id(params).ok_or_else(|| {
                "RESULT_SINK fragment is missing a fragment_instance_id".to_string()
            })?;
            let result = self
                .run_labeled(
                    "result",
                    FragmentRun {
                        plan: &translated,
                        inputs: inputs.clone(),
                        remote_inputs: leases.take(),
                        outputs: Vec::new(),
                        broadcast: false,
                        hash_keys: Vec::new(),
                        label: Self::fragment_label(params),
                    },
                )?
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
        self.run_labeled(
            "sender",
            FragmentRun {
                plan: &translated,
                inputs,
                remote_inputs: leases.take(),
                outputs: slots.clone(),
                broadcast,
                hash_keys,
                label: Self::fragment_label(params),
            },
        )?;

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

    /// Translates a receiver whose sender set is complete -- deferred sender plans spliced in
    /// first, then each remaining exchange bound to the input stream its senders parked into (or
    /// staged, for remote senders) -- and runs it. Returns the next receiver when this one's own
    /// sink completed another sender set.
    fn execute_ready_fragment(
        &self,
        ready: ReadyFragment,
    ) -> std::result::Result<FragmentOutcome, String> {
        let FoldedReceiver {
            params,
            streamed,
            fused,
        } = self.fold_deferred_plans(ready)?;
        if !fused.is_empty() {
            info!(
                query_id = Self::query_id(&params).map(tracing::field::display),
                fragment_instance_id = Self::fragment_instance_id(&params).map(tracing::field::display),
                fused = fused.len(),
                senders = ?fused,
                "fused deferred sender plans into receiver"
            );
        }
        let exchange_inputs = match Self::exchange_inputs(&streamed) {
            Ok(exchange_inputs) => exchange_inputs,
            Err(err) => {
                // Nothing extracted yet: hand the staged leases back before failing.
                self.release_staged(streamed);
                return Err(err);
            }
        };
        // Split the sources BEFORE translating, so the staged batches sit in a guard that
        // releases them on every error path from here to the engine call.
        let mut inputs: Vec<(i32, Vec<SenderSlot>)> = Vec::new();
        let mut leases = StagedLeases::new(self.executor.as_ref());
        for input in streamed {
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
                        // Into the guard first, so the error below still releases them.
                        leases.push(input.node_id, sender_id, batches);
                        // take_ready only releases complete sender sets; an open remote source
                        // here is a rendezvous bug, not a recoverable state.
                        if !closed {
                            return Err(format!(
                                "exchange node {} became ready with remote sender {sender_id} \
                                 still open",
                                input.node_id
                            ));
                        }
                    }
                    // The fold consumed every deferred plan; one left here is a bug.
                    SenderSource::LocalPlan(plan) => {
                        return Err(format!(
                            "exchange node {} still carries deferred sender plan {plan:?} after \
                             the fold",
                            input.node_id
                        ));
                    }
                }
            }
            if !slots.is_empty() {
                inputs.push((input.node_id, slots));
            }
        }
        // A receiver translates when its sender set completes, not at arrival, so pair its plan
        // dump with a fresh params dump here (the arrival-time dump carried no plan yet). With
        // deferred senders this is the FUSED plan, the one the substrait dump next to it runs.
        let dump_seq = Self::dump_fragment(&params);
        let translated =
            self.translate_fragment_logged_with_inputs(&params, &exchange_inputs, dump_seq)?;
        self.execute_fragment_with_inputs(&params, translated, inputs, leases)
    }

    /// Splices every deferred sender plan into the receiver's params.
    ///
    /// A splice refusal here is a bug -- the same checks passed at defer time on the same two
    /// params -- and fails the receiver; the staged leases of every input, the refused one
    /// included, go back to the arena first, as on every pre-run error path.
    fn fold_deferred_plans(
        &self,
        ready: ReadyFragment,
    ) -> std::result::Result<FoldedReceiver, String> {
        let receiver = Self::fragment_context(&ready.params);
        let mut worklist = ready.inputs;
        let mut streamed = Vec::new();
        let mut fused = Vec::new();
        match Self::splice_deferred(
            ready.params,
            &receiver,
            &mut worklist,
            &mut streamed,
            &mut fused,
        ) {
            Ok(params) => {
                // The worklist is drained; keep `take_ready`'s deterministic order.
                streamed.sort_by_key(|input| input.node_id);
                Ok(FoldedReceiver {
                    params,
                    streamed,
                    fused,
                })
            }
            Err(err) => {
                streamed.append(&mut worklist);
                self.release_staged(streamed);
                Err(err)
            }
        }
    }

    /// The fold itself: pops inputs off `worklist`, splicing each lone deferred plan into
    /// `params` (and queueing that plan's own inputs, empty while only leaves are deferred) and
    /// moving every other input to `streamed`.
    fn splice_deferred(
        mut params: TExecPlanFragmentParams,
        receiver: &str,
        worklist: &mut Vec<ReadyExchangeInput>,
        streamed: &mut Vec<ReadyExchangeInput>,
        fused: &mut Vec<FragmentInstanceId>,
    ) -> std::result::Result<TExecPlanFragmentParams, String> {
        while let Some(mut input) = worklist.pop() {
            let deferred = input
                .sources
                .iter()
                .filter(|source| matches!(source, SenderSource::LocalPlan(_)))
                .count();
            if deferred == 0 {
                streamed.push(input);
                continue;
            }
            let node_id = input.node_id;
            let total = input.sources.len();
            let plan = match input.sources.pop() {
                Some(SenderSource::LocalPlan(plan)) if total == 1 => *plan,
                popped => {
                    // Cannot happen (`offer_local_plan` admits a plan only as an exchange's
                    // first and only source). The input goes back whole so the caller's
                    // release still sees any staged batch it carries.
                    input.sources.extend(popped);
                    streamed.push(input);
                    return Err(format!(
                        "exchange node {node_id} has {deferred} deferred sender plan(s) among \
                         {total} source(s); a deferred plan must be an exchange's only source"
                    ));
                }
            };
            let LocalPlan {
                params: sender,
                inputs,
            } = plan;
            params = match fusion::splice(params, node_id, &sender) {
                Ok(spliced) => spliced,
                Err(refusal) => {
                    // The plan's own inputs (empty while only leaves defer) stay releasable.
                    worklist.extend(inputs);
                    return Err(format!(
                        "fragment fusion: splicing deferred {} into {receiver} at exchange \
                         {node_id} failed after passing the defer-time checks: {refusal}",
                        Self::fragment_context(&sender)
                    ));
                }
            };
            fused.extend(Self::fragment_instance_id(&sender));
            worklist.extend(inputs);
        }
        Ok(params)
    }

    /// The exchange inputs a ready receiver's plan binds its stream reads to. Local and remote
    /// senders alike must agree on the schema they produced; the first source is the reference,
    /// disagreement fails the query.
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
                let stream_id = u64::try_from(input.node_id)
                    .map_err(|_| format!("negative exchange node id {}", input.node_id))?;
                Ok(ExchangeInput {
                    node_id: input.node_id,
                    stream_view: sirius_stream_view_name(stream_id),
                    names,
                })
            })
            .collect()
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

    /// Runs one fragment on the executor, bracketed by this CN's start/finish lines. Those lines
    /// carry the StarRocks query id, the fragment instance id and this CN's exchange endpoint,
    /// so one query's halves on different CNs can be stitched from the logs, and a fragment
    /// that never runs (skipped, or fused into its receiver) shows as a missing start line.
    fn run_labeled(
        &self,
        role: &'static str,
        run: FragmentRun<'_>,
    ) -> std::result::Result<Option<FragmentResult>, String> {
        let (query_id, fragment_instance_id) = run.label.log_ids();
        let cn = self.identity.endpoint();
        let inputs = run.inputs.len() + run.remote_inputs.len();
        let outputs = run.outputs.len();
        info!(
            %query_id,
            %fragment_instance_id,
            %cn,
            role,
            inputs,
            outputs,
            "fragment run started"
        );
        let started = std::time::Instant::now();
        let result = self.executor.run(run);
        let elapsed_ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok(_) => info!(
                %query_id,
                %fragment_instance_id,
                %cn,
                role,
                elapsed_ms,
                "fragment run finished"
            ),
            Err(error) => info!(
                %query_id,
                %fragment_instance_id,
                %cn,
                role,
                elapsed_ms,
                %error,
                "fragment run failed"
            ),
        }
        result
    }

    /// The identity of a dispatched fragment: its query id and instance id.
    fn fragment_label(params: &TExecPlanFragmentParams) -> FragmentLabel {
        FragmentLabel {
            query_id: Self::query_id(params),
            fragment_instance_id: Self::fragment_instance_id(params),
        }
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

            // Per instance, exactly like the single-attachment path, so the `fragment {idx}`
            // attribution of a failure names the instance that readied the receiver.
            self.process_inline(&params)
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

/// `pub(crate)` so the engine-linked test in `engine.rs` can borrow
/// [`users_shuffle_pair`](tests::users_shuffle_pair) instead of duplicating the thrift fixtures.
#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use prost::Message;
    use starrocks_thrift::{
        data::TResultBatch,
        data_sinks::{TDataSink, TDataStreamSink, TPlanFragmentDestination, TResultSink},
        descriptors::{TDescriptorTable, TSlotDescriptor, TTableDescriptor, TTupleDescriptor},
        exprs::{TExpr, TExprNode, TExprNodeType, TSlotRef},
        internal_service::{InternalServiceVersion, TPlanFragmentExecParams, TScanRangeParams},
        opcodes::TExprOpcode,
        partitions::{TDataPartition, TPartitionType},
        plan_nodes::{
            TBrokerRangeDesc, TBrokerScanRange, TBrokerScanRangeParams, TEqJoinCondition,
            TExchangeNode, TFileScanNode, TFileScanType, THashJoinNode, TJoinOp, TPlan, TPlanNode,
            TPlanNodeType, TScanRange,
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
        local_exchange::FuseSkip,
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

    /// Wraps any executor, recording every `retire_query` call and delegating the rest.
    #[derive(Debug)]
    struct Retiring<E> {
        inner: E,
        retired: Mutex<Vec<(FragmentInstanceId, RetireTrigger, String)>>,
    }

    impl<E> Retiring<E> {
        fn new(inner: E) -> Self {
            Self {
                inner,
                retired: Mutex::new(Vec::new()),
            }
        }

        fn retired(&self) -> Vec<(FragmentInstanceId, RetireTrigger, String)> {
            self.retired.lock().unwrap().clone()
        }
    }

    impl<E: FragmentExecutor> FragmentExecutor for Retiring<E> {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            self.inner.run(run)
        }

        fn staging_info(&self) -> Result<(u64, u64), String> {
            self.inner.staging_info()
        }

        fn staging_lease(&self, len: u64) -> Result<u64, String> {
            self.inner.staging_lease(len)
        }

        fn staging_release(&self, offset: u64) -> Result<(), String> {
            self.inner.staging_release(offset)
        }

        fn export_packed_next(&self, slot: SenderSlot) -> Result<Option<StagedBatch>, String> {
            self.inner.export_packed_next(slot)
        }

        fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
            self.inner.drop_parked(slot)
        }

        fn retire_query(
            &self,
            query_id: FragmentInstanceId,
            trigger: RetireTrigger,
            cause: &str,
        ) -> Result<(), String> {
            self.retired
                .lock()
                .unwrap()
                .push((query_id, trigger, cause.to_string()));
            Ok(())
        }
    }

    /// Records which fragment instances ran, in order, and fails the intermediate one -- so a
    /// test can assert exactly which queued fragments the worker did and did not run. With
    /// `hold_first_intermediate` set, the first intermediate run blocks until that channel fires,
    /// so a test can queue fragments behind it on the dispatch worker deterministically.
    #[derive(Debug, Default)]
    struct RecordingFailingIntermediate {
        ran: Mutex<Vec<Option<FragmentInstanceId>>>,
        hold_first_intermediate: Mutex<Option<mpsc::Receiver<()>>>,
    }

    impl RecordingFailingIntermediate {
        fn ran(&self) -> Vec<Option<FragmentInstanceId>> {
            self.ran.lock().unwrap().clone()
        }
    }

    impl FragmentExecutor for RecordingFailingIntermediate {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            self.ran
                .lock()
                .unwrap()
                .push(run.label.fragment_instance_id);
            if !run.inputs.is_empty() && !run.outputs.is_empty() {
                let gate = self.hold_first_intermediate.lock().unwrap().take();
                if let Some(gate) = gate {
                    gate.recv_timeout(std::time::Duration::from_secs(10))
                        .map_err(|err| format!("held intermediate was never released: {err}"))?;
                }
            }
            FailingIntermediateExecutor.run(run)
        }
    }

    /// A sender-only fragment (scan into a data stream sink) of `query_id`, addressed at one
    /// local receiver.
    fn sender_only(
        query_id: &TUniqueId,
        instance_id: &TUniqueId,
        dest_node_id: i32,
        receiver_id: TUniqueId,
    ) -> TExecPlanFragmentParams {
        let mut sender = fragment_params(Some(scan_plan(0, 0)), Some(desc_table()));
        sender.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(dest_node_id));
        let mut exec = exec_params(query_id.clone(), instance_id.clone());
        exec.sender_id = Some(0);
        exec.destinations = Some(vec![local_destination(receiver_id)]);
        sender.params = Some(exec);
        sender
    }

    /// The `exec_plan_fragment` status for `params`, for the paths that must refuse.
    fn exec_status(
        service: &SiriusComputeNodeService,
        params: &TExecPlanFragmentParams,
    ) -> StatusPb {
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

    fn instance_ids(ids: &[&TUniqueId]) -> Vec<Option<FragmentInstanceId>> {
        ids.iter()
            .map(|id| Some(FragmentInstanceId::from(*id)))
            .collect()
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
    /// "no buffered result" is not that failure but "not run yet": a non-result fragment has no
    /// entry until the dispatch worker runs (and fails) it, so the poll continues through it.
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
            if result.status.status_code == TStatusCode::OK.0 {
                assert!(response.attachment.is_empty());
            } else if !result
                .status
                .error_msgs
                .first()
                .is_some_and(|message| message.contains("no buffered result"))
            {
                return result;
            }
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
                crate::nixl_transport::TransportRequest::ExchangeMd { .. }
                | crate::nixl_transport::TransportRequest::WarmSession { .. } => {
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

    /// Gate 3: the worker inbox still holds another receiver of the failed query when its
    /// intermediate receiver fails. It must be skipped -- no run, no parked output -- while a
    /// queued receiver of another query still runs. Only receivers reach the worker here (a
    /// sender runs inside its RPC), so the first intermediate is held open and the others are
    /// queued behind it.
    #[test]
    fn queued_fragments_of_a_failed_query_are_skipped() {
        let (release, gate) = mpsc::channel();
        let executor = Arc::new(Retiring::new(RecordingFailingIntermediate {
            ran: Mutex::new(Vec::new()),
            hold_first_intermediate: Mutex::new(Some(gate)),
        }));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(90, 1);
        let root_id = TUniqueId::new(90, 2);
        let middle_id = TUniqueId::new(90, 3);
        let leaf_id = TUniqueId::new(90, 4);
        let (root, middle, leaf) = propagation_chain(&query_id, &root_id, &middle_id, &leaf_id);
        assert_exec_ok(&service, &root);
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);
        // The leaf ran inline and readied the middle, which the worker is now holding open.
        wait_until("the middle to start on the worker", || {
            executor.inner.ran().len() == 2
        });

        // A second receiver of the same query, readied by its own leaf while the middle is held,
        // so it queues behind the middle. Its destination never registers: it must never get far
        // enough to look it up.
        let straggler_id = TUniqueId::new(90, 5);
        let mut straggler = fragment_params(Some(exchange_plan(12, 0)), Some(desc_table()));
        straggler.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(11));
        let mut straggler_exec = exec_params(query_id.clone(), straggler_id.clone());
        straggler_exec.per_exch_num_senders.insert(12, 1);
        straggler_exec.sender_id = Some(0);
        straggler_exec.destinations = Some(vec![local_destination(TUniqueId::new(90, 7))]);
        straggler.params = Some(straggler_exec);
        let straggler_leaf_id = TUniqueId::new(90, 6);
        let straggler_leaf = sender_only(&query_id, &straggler_leaf_id, 12, straggler_id);
        assert_exec_ok(&service, &straggler);
        assert_exec_ok(&service, &straggler_leaf);

        // A receiver of another query, queued last: the worker must still reach it.
        let sentinel_query = TUniqueId::new(91, 1);
        let sentinel_root_id = TUniqueId::new(91, 2);
        let sentinel_id = TUniqueId::new(91, 3);
        let sentinel_leaf_id = TUniqueId::new(91, 4);
        let (sentinel_root, sentinel, sentinel_leaf) = propagation_chain(
            &sentinel_query,
            &sentinel_root_id,
            &sentinel_id,
            &sentinel_leaf_id,
        );
        assert_exec_ok(&service, &sentinel_root);
        assert_exec_ok(&service, &sentinel);
        assert_exec_ok(&service, &sentinel_leaf);

        release.send(()).unwrap();
        // FIFO: the straggler sits between the middle and the sentinel, so once the sentinel ran
        // the straggler was either run or skipped, and `ran` says which.
        wait_until("the sentinel to run", || executor.inner.ran().len() == 5);
        assert_eq!(
            executor.inner.ran(),
            instance_ids(&[
                &leaf_id,
                &middle_id,
                &straggler_leaf_id,
                &sentinel_leaf_id,
                &sentinel_id,
            ]),
            "the straggler of the failed query must never run"
        );
        let result = fetch_error_eventually(&service, root_id.hi, root_id.lo);
        assert!(
            result.status.error_msgs[0].contains("intermediate receiver exploded on the GPU"),
            "{:?}",
            result.status.error_msgs
        );
        assert!(
            service
                .core
                .results
                .failure_of(FragmentInstanceId::from(&query_id))
                .is_some()
        );
        // Both intermediates failed on the worker, so both queries were retired, in that order.
        wait_until("both queries to be retired", || {
            executor.retired().len() == 2
        });
        let retired = executor.retired();
        assert_eq!(retired[0].0, FragmentInstanceId::from(&query_id));
        assert_eq!(retired[0].1, RetireTrigger::CnErr);
        assert!(
            retired[0]
                .2
                .contains("intermediate receiver exploded on the GPU"),
            "{}",
            retired[0].2
        );
        assert_eq!(retired[1].0, FragmentInstanceId::from(&sentinel_query));
    }

    /// Gate 4: a sender of the failed query arriving after the failure is refused on arrival,
    /// before translation, with the RPC status naming the cause.
    #[test]
    fn queued_fragments_of_a_failed_query_are_skipped_inline() {
        let executor = Arc::new(Retiring::new(RecordingFailingIntermediate::default()));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(92, 1);
        let root_id = TUniqueId::new(92, 2);
        let middle_id = TUniqueId::new(92, 3);
        let leaf_id = TUniqueId::new(92, 4);
        let (root, middle, leaf) = propagation_chain(&query_id, &root_id, &middle_id, &leaf_id);
        assert_exec_ok(&service, &root);
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);
        // The middle fails on the worker; the root's poll reporting it is the "failure landed" gate.
        fetch_error_eventually(&service, root_id.hi, root_id.lo);

        let straggler = sender_only(&query_id, &TUniqueId::new(92, 5), 11, TUniqueId::new(92, 6));
        let status = exec_status(&service, &straggler);
        assert_eq!(
            status.status_code,
            TStatusCode::INTERNAL_ERROR.0,
            "{status:?}"
        );
        assert!(
            status.error_msgs[0].contains("already failed on this CN"),
            "{:?}",
            status.error_msgs
        );
        assert_eq!(
            executor.inner.ran(),
            instance_ids(&[&leaf_id, &middle_id]),
            "the refused sender never ran"
        );
    }

    /// A CN-side fragment failure retires the query on the executor exactly once, with the
    /// fragment's own error as the cause.
    #[test]
    fn a_fragment_failure_retires_the_query_on_the_executor() {
        let executor = Arc::new(Retiring::new(FailingIntermediateExecutor));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(95, 1);
        let root_id = TUniqueId::new(95, 2);
        let (root, middle, leaf) = propagation_chain(
            &query_id,
            &root_id,
            &TUniqueId::new(95, 3),
            &TUniqueId::new(95, 4),
        );
        assert_exec_ok(&service, &root);
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);
        fetch_error_eventually(&service, root_id.hi, root_id.lo);

        // The result entry fails before the executor is told; wait for the retire to land.
        wait_until("the query to be retired", || executor.retired().len() == 1);
        let retired = executor.retired();
        assert_eq!(retired[0].0, FragmentInstanceId::from(&query_id));
        assert_eq!(retired[0].1, RetireTrigger::CnErr);
        assert!(
            retired[0]
                .2
                .contains("intermediate receiver exploded on the GPU"),
            "{}",
            retired[0].2
        );
    }

    /// A pre-run failure (here: the senders disagree on their output names, caught before
    /// translation) must retire the query -- so the engine drops the local sender's parked slot
    /// -- and release the staged leases of the remote sender it held in hand.
    #[test]
    fn translation_failure_retires_the_slots_in_hand_and_releases_remote_leases() {
        let executor = Arc::new(Retiring::new(RecordingExecutor::default()));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(93, 1);
        let receiver_id = TUniqueId::new(93, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id.clone(), receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 2);
        receiver.params = Some(receiver_exec);
        assert_exec_ok(&service, &receiver);

        // Sender 0 is local (the stub parks nothing, but the CN records its LocalParked slot).
        let local = sender_only(&query_id, &TUniqueId::new(93, 3), 7, receiver_id.clone());
        assert_exec_ok(&service, &local);

        // Sender 1 is remote, with a staged batch and different column names.
        let data = route(
            &service,
            methods::TRANSMIT_PACKED,
            transmit_params(&receiver_id, 7, 1, 0, false, 4096, 512, &["id", "other"]),
            vec![0xAB; 16],
        );
        let data = PTransmitPackedResult::decode(data.body.as_slice()).unwrap();
        assert_eq!(
            data.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            data.status.error_msgs
        );
        let eos = route(
            &service,
            methods::TRANSMIT_PACKED,
            transmit_params(&receiver_id, 7, 1, 1, true, 0, 0, &["id", "other"]),
            Vec::new(),
        );
        let eos = PTransmitPackedResult::decode(eos.body.as_slice()).unwrap();
        assert_eq!(
            eos.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            eos.status.error_msgs
        );

        let result = fetch_error_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(
            result.status.error_msgs[0].contains("different output names"),
            "{:?}",
            result.status.error_msgs
        );
        assert_eq!(
            executor.inner.released.lock().unwrap().as_slice(),
            &[4096],
            "the staged lease of the never-run receiver went back to the arena"
        );
        assert!(
            executor.inner.remote_inputs.lock().unwrap().is_empty(),
            "nothing ran"
        );
        let retired = executor.retired();
        assert_eq!(retired.len(), 1, "{retired:?}");
        assert_eq!(retired[0].0, FragmentInstanceId::from(&query_id));
        assert_eq!(retired[0].1, RetireTrigger::CnErr);
        assert!(
            retired[0].2.contains("different output names"),
            "{}",
            retired[0].2
        );
    }

    /// Gate 4 keeps the reserve-then-fail contract for a result fragment of a dead query: its RPC
    /// is OK and its first poll reports the cause. A non-result fragment of the same dead query
    /// is refused on arrival instead.
    #[test]
    fn a_result_fragment_arriving_after_the_failure_still_reports_the_cause() {
        let executor = Arc::new(Retiring::new(FailingIntermediateExecutor));
        let service = SiriusComputeNodeService::with_executor(executor, test_identity());
        let query_id = TUniqueId::new(94, 1);
        let root_id = TUniqueId::new(94, 2);
        let middle_id = TUniqueId::new(94, 3);
        let (root, middle, leaf) =
            propagation_chain(&query_id, &root_id, &middle_id, &TUniqueId::new(94, 4));
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);
        fetch_error_eventually(&service, middle_id.hi, middle_id.lo);

        assert_exec_ok(&service, &root);
        let result = fetch_error_eventually(&service, root_id.hi, root_id.lo);
        let message = &result.status.error_msgs[0];
        assert!(
            message.contains(&FragmentInstanceId::from(&middle_id).to_string())
                && message.contains("intermediate receiver exploded on the GPU"),
            "{message}"
        );

        let straggler = sender_only(&query_id, &TUniqueId::new(94, 5), 11, TUniqueId::new(94, 6));
        let status = exec_status(&service, &straggler);
        assert_eq!(
            status.status_code,
            TStatusCode::INTERNAL_ERROR.0,
            "{status:?}"
        );
        assert!(
            status.error_msgs[0].contains("already failed on this CN"),
            "{:?}",
            status.error_msgs
        );
    }

    // Same-node fragment fusion: a hash-partitioned single-destination local leaf is spliced into
    // its receiver's plan instead of running (`try_defer_sender`, `fold_deferred_plans`).

    /// Every `tracing` line this test binary emits, from every thread (the RPC blocking pool and
    /// the dispatch worker included), so a test can assert the fusion log contract. One
    /// process-wide subscriber installed once; tests pick their own lines out by query id.
    static CAPTURED_LOGS: Mutex<String> = Mutex::new(String::new());

    struct CaptureLogs;

    impl std::io::Write for CaptureLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            CAPTURED_LOGS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push_str(&String::from_utf8_lossy(buf));
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureLogs {
        type Writer = CaptureLogs;

        fn make_writer(&'a self) -> Self::Writer {
            CaptureLogs
        }
    }

    fn capture_logs() {
        static INSTALL: std::sync::Once = std::sync::Once::new();
        INSTALL.call_once(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::DEBUG)
                .with_ansi(false)
                .with_writer(CaptureLogs)
                .finish();
            // Nothing else in this binary installs a subscriber; if something ever does, the
            // capture stays empty and the log assertions fail loudly rather than pass vacuously.
            let _ = tracing::subscriber::set_global_default(subscriber);
        });
    }

    /// The captured lines that name `query_id`.
    fn logs_of(query_id: &TUniqueId) -> Vec<String> {
        let id = FragmentInstanceId::from(query_id).to_string();
        CAPTURED_LOGS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lines()
            .filter(|line| line.contains(&id))
            .map(str::to_string)
            .collect()
    }

    /// How many captured lines of `query_id` carry `needle`.
    fn log_count(query_id: &TUniqueId, needle: &str) -> usize {
        logs_of(query_id)
            .iter()
            .filter(|line| line.contains(needle))
            .count()
    }

    /// Whether some captured line of `query_id` carries every needle.
    fn logged(query_id: &TUniqueId, needles: &[&str]) -> bool {
        logs_of(query_id)
            .iter()
            .any(|line| needles.iter().all(|needle| line.contains(needle)))
    }

    /// The shape of every run as `(inputs, remote_inputs, outputs, stream_inputs)` counts: a
    /// fused receiver runs with no exchange input at all, a parked one with one stream.
    #[derive(Debug, Default)]
    struct RecordingRunExecutor {
        runs: Mutex<Vec<(usize, usize, usize, usize)>>,
    }

    impl RecordingRunExecutor {
        fn runs(&self) -> Vec<(usize, usize, usize, usize)> {
            self.runs.lock().unwrap().clone()
        }
    }

    impl FragmentExecutor for RecordingRunExecutor {
        fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            self.runs.lock().unwrap().push((
                run.inputs.len(),
                run.remote_inputs.len(),
                run.outputs.len(),
                run.plan.stream_inputs.len(),
            ));
            StubExecutor.run(run)
        }
    }

    /// Fails every run, so a fused receiver's failure path can be watched.
    #[derive(Debug)]
    struct FailingExecutor;

    impl FragmentExecutor for FailingExecutor {
        fn run(&self, _run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
            Err("fused receiver exploded on the GPU".to_string())
        }
    }

    /// A leaf's run: parks one output, reads nothing. A receiver's over one parked stream:
    /// reads one exchange input through one declared stream.
    const LEAF_RUN: (usize, usize, usize, usize) = (0, 0, 1, 0);
    const PARKED_RECEIVER_RUN: (usize, usize, usize, usize) = (1, 0, 0, 1);
    /// A fused receiver's run (or a fused middle fragment's, which parks one output).
    const FUSED_RESULT_RUN: (usize, usize, usize, usize) = (0, 0, 0, 0);
    const FUSED_SENDER_RUN: (usize, usize, usize, usize) = (0, 0, 1, 0);

    /// A result-sink receiver `[EXCHANGE(exchange, tuple 0)]` expecting `senders` senders.
    fn result_receiver(
        query_id: &TUniqueId,
        instance_id: &TUniqueId,
        exchange: i32,
        senders: i32,
    ) -> TExecPlanFragmentParams {
        let mut receiver = fragment_params(Some(exchange_plan(exchange, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut exec = exec_params(query_id.clone(), instance_id.clone());
        exec.per_exch_num_senders.insert(exchange, senders);
        receiver.params = Some(exec);
        receiver
    }

    /// A leaf `[FILE_SCAN(0, tuple 0)]` whose HASH_PARTITIONED sink's only destination is
    /// exchange `dest_node_id` of `receiver_id` on this CN: the shape `leaf` mode fuses.
    fn hash_leaf(
        query_id: &TUniqueId,
        instance_id: &TUniqueId,
        sender_id: i32,
        dest_node_id: i32,
        receiver_id: TUniqueId,
    ) -> TExecPlanFragmentParams {
        let mut leaf = sender_only(query_id, instance_id, dest_node_id, receiver_id);
        leaf.fragment.as_mut().unwrap().output_sink =
            Some(hash_partitioned_data_stream_sink(dest_node_id, 1, 0));
        leaf.params.as_mut().unwrap().sender_id = Some(sender_id);
        leaf
    }

    /// The measured 1-CN shuffle pair over the users descriptor, for the fused-plan tests here
    /// and the engine-linked one in `engine.rs`: query `(hi, 1)`, a result-sink receiver
    /// `(hi, 2)` = `[EXCHANGE(7)]` expecting one sender, and a leaf `(hi, 3)` = `[FILE_SCAN(0)]`
    /// reading the parquet file at `path` (`file_size` bytes) through a HASH_PARTITIONED sink
    /// whose only destination is that receiver on this CN.
    pub(crate) fn users_shuffle_pair(
        hi: i64,
        path: &str,
        file_size: i64,
    ) -> (TExecPlanFragmentParams, TExecPlanFragmentParams) {
        let query_id = TUniqueId::new(hi, 1);
        let receiver_id = TUniqueId::new(hi, 2);
        let receiver = result_receiver(&query_id, &receiver_id, 7, 1);
        let mut leaf = hash_leaf(&query_id, &TUniqueId::new(hi, 3), 0, 7, receiver_id);
        leaf.params
            .as_mut()
            .unwrap()
            .per_node_scan_ranges
            .insert(0, vec![parquet_scan_range(path, file_size)]);
        (receiver, leaf)
    }

    /// A join receiver over two exchanges on `tpch_desc_table()`:
    /// `[HASH_JOIN(l_orderkey = o_orderkey), EXCHANGE(7, tuple 0), EXCHANGE(8, tuple 1)]`, the
    /// shape whose one exchange can fuse while the other keeps its stream.
    fn join_receiver_plan() -> TPlan {
        let bigint = || scalar_type(TPrimitiveType::BIGINT);
        let mut join = scan_node(2, 0);
        join.node_type = TPlanNodeType::HASH_JOIN_NODE;
        join.num_children = 2;
        join.row_tuples = vec![0, 1];
        join.file_scan_node = None;
        join.hash_join_node = Some(THashJoinNode::new(
            TJoinOp::INNER_JOIN,
            vec![TEqJoinCondition::new(
                slot_ref(1, 0, bigint()),
                slot_ref(3, 1, bigint()),
                Some(TExprOpcode::EQ),
            )],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
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
        let mut plan = TPlan::new(vec![join]);
        plan.nodes.extend(exchange_plan(7, 0).nodes);
        plan.nodes.extend(exchange_plan(8, 1).nodes);
        plan
    }

    /// A join receiver `(hi, 2)` of query `(hi, 1)` over exchanges 7 and 8, one sender each.
    fn join_receiver(hi: i64) -> TExecPlanFragmentParams {
        let mut receiver = fragment_params(Some(join_receiver_plan()), Some(tpch_desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut exec = exec_params(TUniqueId::new(hi, 1), TUniqueId::new(hi, 2));
        exec.per_exch_num_senders.insert(7, 1);
        exec.per_exch_num_senders.insert(8, 1);
        receiver.params = Some(exec);
        receiver
    }

    /// The broadcast leaf `(hi, 4)` of query `(hi, 1)` feeding exchange 8 of the join receiver:
    /// `[FILE_SCAN(1, tuple 1)]` (orders) into an UNPARTITIONED sink, which `leaf` mode parks.
    fn orders_broadcast_leaf(hi: i64) -> TExecPlanFragmentParams {
        let mut leaf = sender_only(
            &TUniqueId::new(hi, 1),
            &TUniqueId::new(hi, 4),
            8,
            TUniqueId::new(hi, 2),
        );
        leaf.fragment.as_mut().unwrap().plan = Some(scan_plan(1, 1));
        leaf.desc_tbl = Some(tpch_desc_table());
        leaf
    }

    /// The orders leaf `(hi, 4)` as `leaf` mode fuses it: [`orders_broadcast_leaf`]'s scan
    /// behind a HASH_PARTITIONED sink on `o_orderkey` into exchange 8.
    fn orders_hash_leaf(hi: i64) -> TExecPlanFragmentParams {
        let mut leaf = orders_broadcast_leaf(hi);
        leaf.fragment.as_mut().unwrap().output_sink =
            Some(hash_partitioned_data_stream_sink(8, 3, 1));
        leaf
    }

    #[test]
    fn hash_partitioned_single_destination_leaf_fuses_into_its_receiver() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(200, 1);
        let receiver_id = TUniqueId::new(200, 2);
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 1));
        assert_exec_ok(
            &service,
            &hash_leaf(
                &query_id,
                &TUniqueId::new(200, 3),
                0,
                7,
                receiver_id.clone(),
            ),
        );

        let fetched = fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(!fetched.attachment.is_empty());
        // One run, the receiver's, reading no exchange input: the leaf never ran on its own.
        assert_eq!(executor.runs(), vec![FUSED_RESULT_RUN]);
        let logs = logs_of(&query_id);
        assert!(
            logged(
                &query_id,
                &[
                    "fused sender fragment into its local receiver",
                    "exchange=7",
                    "mode=Leaf"
                ]
            ),
            "{logs:#?}"
        );
        assert!(
            logged(
                &query_id,
                &["fused deferred sender plans into receiver", "fused=1"]
            ),
            "{logs:#?}"
        );
        assert_eq!(
            log_count(&query_id, "fragment fusion skipped"),
            0,
            "{logs:#?}"
        );
    }

    #[test]
    fn broadcast_leaf_still_parks_in_leaf_mode() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(201, 1);
        let receiver_id = TUniqueId::new(201, 2);
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 1));
        assert_exec_ok(
            &service,
            &sender_only(&query_id, &TUniqueId::new(201, 3), 7, receiver_id.clone()),
        );

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(executor.runs(), vec![LEAF_RUN, PARKED_RECEIVER_RUN]);
        let logs = logs_of(&query_id);
        assert_eq!(
            log_count(&query_id, "fused sender fragment"),
            0,
            "{logs:#?}"
        );
        assert!(
            logged(
                &query_id,
                &[
                    "fragment fusion skipped",
                    "leaf mode fuses HASH_PARTITIONED sinks only"
                ]
            ),
            "{logs:#?}"
        );
    }

    #[test]
    fn broadcast_leaf_fuses_in_leaf_any_mode() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        service.set_fragment_fusion(FusionMode::LeafAny);
        let query_id = TUniqueId::new(202, 1);
        let receiver_id = TUniqueId::new(202, 2);
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 1));
        assert_exec_ok(
            &service,
            &sender_only(&query_id, &TUniqueId::new(202, 3), 7, receiver_id.clone()),
        );

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(executor.runs(), vec![FUSED_RESULT_RUN]);
        assert!(
            logged(
                &query_id,
                &[
                    "fused sender fragment into its local receiver",
                    "mode=LeafAny"
                ]
            ),
            "{:#?}",
            logs_of(&query_id)
        );
    }

    #[test]
    fn fusion_off_restores_two_runs() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        service.set_fragment_fusion(FusionMode::Off);
        let query_id = TUniqueId::new(203, 1);
        let receiver_id = TUniqueId::new(203, 2);
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 1));
        assert_exec_ok(
            &service,
            &hash_leaf(
                &query_id,
                &TUniqueId::new(203, 3),
                0,
                7,
                receiver_id.clone(),
            ),
        );

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(executor.runs(), vec![LEAF_RUN, PARKED_RECEIVER_RUN]);
        assert!(
            logged(&query_id, &["fragment fusion skipped", "reason=off"]),
            "{:#?}",
            logs_of(&query_id)
        );
    }

    /// Fusion needs the receiver registered first (the FE deploys stage by stage from the root);
    /// a leaf that races ahead takes today's path and the query still completes.
    #[test]
    fn leaf_arriving_before_its_receiver_falls_back_to_parking() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(204, 1);
        let receiver_id = TUniqueId::new(204, 2);
        assert_exec_ok(
            &service,
            &hash_leaf(
                &query_id,
                &TUniqueId::new(204, 3),
                0,
                7,
                receiver_id.clone(),
            ),
        );
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 1));

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(executor.runs(), vec![LEAF_RUN, PARKED_RECEIVER_RUN]);
        assert!(
            logged(
                &query_id,
                &["fragment fusion skipped", "reason=NoPendingReceiver"]
            ),
            "{:#?}",
            logs_of(&query_id)
        );
    }

    #[test]
    fn receiver_expecting_two_senders_keeps_the_parked_path() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(205, 1);
        let receiver_id = TUniqueId::new(205, 2);
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 2));
        assert_exec_ok(
            &service,
            &hash_leaf(
                &query_id,
                &TUniqueId::new(205, 3),
                0,
                7,
                receiver_id.clone(),
            ),
        );
        assert_exec_ok(
            &service,
            &hash_leaf(
                &query_id,
                &TUniqueId::new(205, 4),
                1,
                7,
                receiver_id.clone(),
            ),
        );

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(
            executor.runs(),
            vec![LEAF_RUN, LEAF_RUN, PARKED_RECEIVER_RUN]
        );
        assert_eq!(
            log_count(&query_id, "reason=ReceiverExpectsMany(2)"),
            2,
            "{:#?}",
            logs_of(&query_id)
        );
    }

    /// One exchange of a join receiver absorbs its hash-partitioned leaf while the other keeps
    /// streaming its broadcast leaf's parked rows.
    #[test]
    fn partial_fusion_keeps_the_other_exchange_as_a_stream() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(206, 1);
        let receiver_id = TUniqueId::new(206, 2);
        assert_exec_ok(&service, &join_receiver(206));
        let mut lineitem = hash_leaf(
            &query_id,
            &TUniqueId::new(206, 3),
            0,
            7,
            receiver_id.clone(),
        );
        lineitem.desc_tbl = Some(tpch_desc_table());
        assert_exec_ok(&service, &lineitem);
        assert!(
            executor.runs().is_empty(),
            "the deferred leaf did not run and the receiver still waits for exchange 8"
        );
        assert_exec_ok(&service, &orders_broadcast_leaf(206));

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(executor.runs(), vec![LEAF_RUN, PARKED_RECEIVER_RUN]);
        assert_eq!(
            log_count(&query_id, "fused sender fragment into its local receiver"),
            1,
            "{:#?}",
            logs_of(&query_id)
        );
    }

    /// The dominant SF1000 shape (q05's join under two shuffled exchanges): both hash leaves of
    /// a join receiver fuse. The fold pops the ready inputs last-first, so the second splice
    /// checks its edge against the receiver the first one already rewrote; the one run reads no
    /// exchange input at all.
    #[test]
    fn two_hash_leaves_fuse_into_one_join_receiver() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(215, 1);
        let receiver_id = TUniqueId::new(215, 2);
        assert_exec_ok(&service, &join_receiver(215));
        let mut lineitem = hash_leaf(
            &query_id,
            &TUniqueId::new(215, 3),
            0,
            7,
            receiver_id.clone(),
        );
        lineitem.desc_tbl = Some(tpch_desc_table());
        assert_exec_ok(&service, &lineitem);
        assert!(
            executor.runs().is_empty(),
            "the receiver still waits for exchange 8"
        );
        assert_exec_ok(&service, &orders_hash_leaf(215));

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        let logs = logs_of(&query_id);
        assert_eq!(executor.runs(), vec![FUSED_RESULT_RUN], "{logs:#?}");
        assert_eq!(
            log_count(&query_id, "fused sender fragment into its local receiver"),
            2,
            "{logs:#?}"
        );
        for exchange in ["exchange=7", "exchange=8"] {
            assert!(
                logged(
                    &query_id,
                    &["fused sender fragment into its local receiver", exchange]
                ),
                "{logs:#?}"
            );
        }
        assert!(
            logged(
                &query_id,
                &["fused deferred sender plans into receiver", "fused=2"]
            ),
            "{logs:#?}"
        );
        assert_eq!(
            log_count(&query_id, "fragment fusion skipped"),
            0,
            "{logs:#?}"
        );
    }

    /// The root <- middle <- leaf chain with cached descriptor references: the leaf fuses into
    /// the middle (whose registered params carry the resolved descriptor table), the middle is a
    /// sender with an exchange of its own and so parks as today, and the root streams it.
    #[test]
    fn middle_fragment_is_not_deferred_in_leaf_mode() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(207, 1);
        let root_id = TUniqueId::new(207, 2);
        let middle_id = TUniqueId::new(207, 3);
        assert_exec_ok(&service, &result_receiver(&query_id, &root_id, 9, 1));

        let cached_desc = TDescriptorTable::new(None, Vec::new(), None, Some(true));
        let mut middle = fragment_params(Some(exchange_plan(7, 0)), Some(cached_desc.clone()));
        middle.fragment.as_mut().unwrap().output_sink = Some(data_stream_sink(9));
        let mut middle_exec = exec_params(query_id.clone(), middle_id.clone());
        middle_exec.per_exch_num_senders.insert(7, 1);
        middle_exec.sender_id = Some(0);
        middle_exec.destinations = Some(vec![local_destination(root_id.clone())]);
        middle.params = Some(middle_exec);
        assert_exec_ok(&service, &middle);

        let mut leaf = hash_leaf(&query_id, &TUniqueId::new(207, 4), 0, 7, middle_id);
        leaf.desc_tbl = Some(cached_desc);
        assert_exec_ok(&service, &leaf);

        fetch_rows_eventually(&service, root_id.hi, root_id.lo);
        assert_eq!(executor.runs(), vec![FUSED_SENDER_RUN, PARKED_RECEIVER_RUN]);
        assert_eq!(
            log_count(&query_id, "fused sender fragment into its local receiver"),
            1,
            "{:#?}",
            logs_of(&query_id)
        );
    }

    #[test]
    fn fusion_applies_on_the_batch_path() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(209, 1);
        let receiver_id = TUniqueId::new(209, 2);
        let mut receiver = result_receiver(&query_id, &receiver_id, 7, 1);
        receiver.desc_tbl = None;
        let mut leaf = hash_leaf(
            &query_id,
            &TUniqueId::new(209, 3),
            0,
            7,
            receiver_id.clone(),
        );
        leaf.desc_tbl = None;
        let batch = TExecBatchPlanFragmentsParams::new(
            Some(fragment_params(None, Some(desc_table()))),
            Some(vec![receiver, leaf]),
        );
        let response = route(
            &service,
            methods::EXEC_BATCH_PLAN_FRAGMENTS,
            PExecBatchPlanFragmentsRequest {
                attachment_protocol: Some("binary".to_string()),
            }
            .encode_to_vec(),
            serialize_binary(&batch),
        );
        let result = PExecBatchPlanFragmentsResult::decode(response.body.as_slice()).unwrap();
        let status = result.status.expect("status is always set");
        assert_eq!(status.status_code, TStatusCode::OK.0, "{status:?}");

        fetch_rows_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(executor.runs(), vec![FUSED_RESULT_RUN]);
        assert_eq!(
            log_count(&query_id, "fused sender fragment into its local receiver"),
            1,
            "{:#?}",
            logs_of(&query_id)
        );
    }

    /// A hash-partitioned leaf with a remote destination behaves exactly as before fusion:
    /// refused before any run without a transport, run and handed to the transport with one.
    #[test]
    fn remote_single_destination_is_never_fused() {
        capture_logs();
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(210, 1);
        let receiver_id = TUniqueId::new(210, 2);
        let mut leaf = hash_leaf(
            &query_id,
            &TUniqueId::new(210, 3),
            0,
            7,
            receiver_id.clone(),
        );
        leaf.params.as_mut().unwrap().destinations =
            Some(vec![remote_destination(receiver_id.clone(), 8061)]);

        let status = exec_status(&service, &leaf);
        assert_eq!(
            status.status_code,
            TStatusCode::INTERNAL_ERROR.0,
            "{status:?}"
        );
        assert!(
            status.error_msgs[0]
                .contains("cross-node exchange to 127.0.0.1:8061 needs the nixl transport tier"),
            "{:?}",
            status.error_msgs
        );
        assert!(executor.runs().is_empty());
        assert!(
            logged(
                &query_id,
                &["fragment fusion skipped", "reason=remote destination"]
            ),
            "{:#?}",
            logs_of(&query_id)
        );

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
                crate::nixl_transport::TransportRequest::ExchangeMd { .. }
                | crate::nixl_transport::TransportRequest::WarmSession { .. } => {
                    panic!("the sender flow never exchanges metadata itself")
                }
            }
        });
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_transport(
            executor.clone(),
            test_identity(),
            Some(crate::nixl_transport::NixlTransport::for_test(requests_tx)),
        );
        assert_exec_ok(&service, &leaf);
        assert_eq!(executor.runs(), vec![LEAF_RUN]);
        let spec = fake_transport.join().unwrap();
        assert_eq!(spec.brpc_port, 8061);
        assert_eq!(spec.slot.node_id, 7);
        assert_eq!(
            spec.slot.fragment_instance_id,
            FragmentInstanceId::from(&receiver_id)
        );
    }

    /// A fused receiver fails under the receiver's ids like any receiver: the FE's poll on the
    /// result id reports the cause.
    #[test]
    fn fused_receiver_failure_fails_the_fe_polled_result() {
        let service =
            SiriusComputeNodeService::with_executor(Arc::new(FailingExecutor), test_identity());
        let query_id = TUniqueId::new(211, 1);
        let receiver_id = TUniqueId::new(211, 2);
        assert_exec_ok(&service, &result_receiver(&query_id, &receiver_id, 7, 1));
        assert_exec_ok(
            &service,
            &hash_leaf(
                &query_id,
                &TUniqueId::new(211, 3),
                0,
                7,
                receiver_id.clone(),
            ),
        );

        let result = fetch_error_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert_eq!(result.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            result.status.error_msgs[0].contains("fused receiver exploded on the GPU"),
            "{:?}",
            result.status.error_msgs
        );
        assert_eq!(result.eos, Some(true));
        assert!(
            service
                .core
                .results
                .failure_of(FragmentInstanceId::from(&query_id))
                .is_some(),
            "the failure is recorded at query level"
        );
    }

    /// The receiver's ready-time dump is the fused params (the plan the engine runs), not the
    /// registered ones: no exchange 7, the leaf's scan and its scan range inline.
    #[test]
    fn fused_receiver_dump_is_the_fused_plan() {
        capture_logs();
        let dir = tempfile::tempdir().unwrap();
        let (receiver, leaf) = users_shuffle_pair(212, "file:///data/users.parquet", 1024);
        // The CN registers what it deserialized from the RPC attachment, and the thrift round
        // trip normalizes an absent optional list (`destinations: None` arrives as `Some([])`),
        // so the expectation is the splice of the wire forms.
        let wire = |params: &TExecPlanFragmentParams| {
            SiriusComputeNodeService::deserialize_binary::<TExecPlanFragmentParams>(
                &serialize_binary(params),
            )
            .unwrap()
        };
        let expected =
            fusion::splice(wire(&receiver), 7, &wire(&leaf)).expect("the pair is fusable");
        let executor = Arc::new(RecordingRunExecutor::default());
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        // The test binary's one environment lock (`tunable::tests::with_env`) guards the write
        // and restores the variable once the fused receiver has run and dumped.
        crate::tunable::tests::with_env(
            &[(
                "SIRIUS_CN_DUMP_FRAGMENTS",
                Some(dir.path().to_str().unwrap()),
            )],
            || {
                assert_exec_ok(&service, &receiver);
                assert_exec_ok(&service, &leaf);
                fetch_rows_eventually(&service, 212, 2);
            },
        );
        let query_id = TUniqueId::new(212, 1);
        assert_eq!(
            executor.runs(),
            vec![FUSED_RESULT_RUN],
            "{:#?}",
            logs_of(&query_id)
        );

        // Other tests may dump into the same directory while the variable is set; the fused
        // receiver's dump is the one that equals the spliced params exactly.
        let dumps: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("fragment-"))
            .map(|entry| std::fs::read_to_string(entry.path()).unwrap())
            .collect();
        let expected_text = format!("{expected:#?}");
        assert!(
            dumps.contains(&expected_text),
            "none of the {} fragment dump(s) is the fused params; {:#?}",
            dumps.len(),
            logs_of(&query_id)
        );
        // What that dump says in plan terms.
        let nodes = &expected
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes;
        assert!(
            nodes
                .iter()
                .all(|node| node.node_type != TPlanNodeType::EXCHANGE_NODE)
        );
        assert!(
            nodes
                .iter()
                .any(|node| node.node_type == TPlanNodeType::FILE_SCAN_NODE && node.node_id == 0)
        );
        let exec = expected.params.as_ref().unwrap();
        assert!(exec.per_exch_num_senders.is_empty());
        assert!(exec.per_node_scan_ranges.contains_key(&0));
        assert_eq!(exec.fragment_instance_id, TUniqueId::new(212, 2));
    }

    /// Cancellation is `cancel_plan_fragment`'s teardown: the deferred plan leaves the rendezvous with the
    /// cancelled receiver, the receiver is retired, and a late leaf of the query is refused on
    /// arrival; nothing of the query ever ran.
    #[test]
    fn cancel_drops_deferred_plans() {
        capture_logs();
        let executor = Arc::new(Retiring::new(RecordingRunExecutor::default()));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(213, 1);
        let receiver_id = TUniqueId::new(213, 2);
        assert_exec_ok(&service, &join_receiver(213));
        let mut lineitem = hash_leaf(
            &query_id,
            &TUniqueId::new(213, 3),
            0,
            7,
            receiver_id.clone(),
        );
        lineitem.desc_tbl = Some(tpch_desc_table());
        assert_exec_ok(&service, &lineitem);
        assert_eq!(
            log_count(&query_id, "fused sender fragment into its local receiver"),
            1,
            "{:#?}",
            logs_of(&query_id)
        );
        assert!(executor.inner.runs().is_empty());

        cancel_ok(
            &service,
            cancel_request(
                &receiver_id,
                Some(&query_id),
                Some(PPlanFragmentCancelReason::InternalError),
                Some("peer failed"),
            ),
        );
        assert!(
            logged(
                &query_id,
                &["cancel_plan_fragment retired the query on this CN"]
            ),
            "{:#?}",
            logs_of(&query_id)
        );
        let receiver = FragmentInstanceId::from(&receiver_id);
        assert!(service.core.exchanges.is_retired(receiver));
        // The deferred plan left with the receiver: a second leaf 7 finds nothing pending.
        let offer = service
            .core
            .exchanges
            .offer_local_plan(
                ExchangeKey {
                    fragment_instance_id: receiver,
                    node_id: 7,
                },
                0,
                LocalPlan {
                    params: lineitem.clone(),
                    inputs: Vec::new(),
                },
                |_| Ok(()),
            )
            .unwrap();
        let FuseOffer::Declined { skip, .. } = offer else {
            panic!("a retired receiver must not accept a plan");
        };
        assert!(matches!(skip, FuseSkip::NoPendingReceiver), "{skip}");

        // A late leaf 8 is refused by gate 4 on arrival.
        let status = exec_status(&service, &orders_broadcast_leaf(213));
        assert_eq!(
            status.status_code,
            TStatusCode::INTERNAL_ERROR.0,
            "{status:?}"
        );
        assert!(
            status.error_msgs[0].contains("already failed on this CN"),
            "{:?}",
            status.error_msgs
        );
        assert!(executor.inner.runs().is_empty(), "nothing of the query ran");
        let result = fetch_error_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(
            result.status.error_msgs[0].contains("cancelled by the FE"),
            "{:?}",
            result.status.error_msgs
        );
        // The cancel retired the query; the refused late leaf's RPC error is recorded at query
        // level too (an idempotent second retire), as for any inline failure.
        let retired = executor.retired();
        assert_eq!(
            retired[0].1,
            RetireTrigger::Cancel("INTERNAL_ERROR".to_string()),
            "{retired:?}"
        );
        assert!(
            retired
                .iter()
                .all(|(query, ..)| *query == FragmentInstanceId::from(&query_id)),
            "{retired:?}"
        );
    }

    /// A `cancel_plan_fragment` request body as the FE builds it.
    fn cancel_request(
        instance: &TUniqueId,
        query_id: Option<&TUniqueId>,
        reason: Option<PPlanFragmentCancelReason>,
        message: Option<&str>,
    ) -> Vec<u8> {
        PCancelPlanFragmentRequest {
            finst_id: PUniqueId {
                hi: instance.hi,
                lo: instance.lo,
            },
            cancel_reason: reason.map(|reason| reason as i32),
            is_pipeline: None,
            query_id: query_id.map(|id| PUniqueId {
                hi: id.hi,
                lo: id.lo,
            }),
            error_message: message.map(str::to_string),
        }
        .encode_to_vec()
    }

    /// Cancels and asserts the FE-facing OK the shared jprotobuf channel depends on.
    fn cancel_ok(service: &SiriusComputeNodeService, body: Vec<u8>) {
        let response = route(service, methods::CANCEL_PLAN_FRAGMENT, body, Vec::new());
        let cancel = PCancelPlanFragmentResult::decode(response.body.as_slice()).unwrap();
        assert_eq!(
            cancel.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            cancel.status.error_msgs
        );
    }

    /// A sender failing inside its own RPC fails the query on this CN, so a result fragment of the query reserved afterwards reports the real cause on its
    /// first poll instead of running out the fetch_data timeout.
    #[test]
    fn an_inline_sender_failure_is_recorded_at_query_level() {
        let executor = Arc::new(Retiring::new(StubExecutor));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(150, 1);

        let mut sender = supported_fragment();
        sender.fragment.as_mut().unwrap().output_sink =
            Some(sink_of_type(TDataSinkType::OLAP_TABLE_SINK));
        sender.params = Some(exec_params(query_id.clone(), TUniqueId::new(150, 2)));
        let status = exec_status(&service, &sender);
        assert_eq!(
            status.status_code,
            TStatusCode::INTERNAL_ERROR.0,
            "{status:?}"
        );
        assert!(
            status.error_msgs[0].contains("OLAP_TABLE_SINK"),
            "{:?}",
            status.error_msgs
        );

        let root_id = TUniqueId::new(150, 3);
        let mut root = fragment_params(Some(exchange_plan(9, 0)), Some(desc_table()));
        root.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut root_exec = exec_params(query_id.clone(), root_id.clone());
        root_exec.per_exch_num_senders.insert(9, 1);
        root.params = Some(root_exec);
        assert_exec_ok(&service, &root);
        let result = fetch_error_eventually(&service, root_id.hi, root_id.lo);
        assert!(
            result.status.error_msgs[0].contains("OLAP_TABLE_SINK"),
            "{:?}",
            result.status.error_msgs
        );

        let retired = executor.retired();
        assert_eq!(retired.len(), 1, "{retired:?}");
        assert_eq!(retired[0].0, FragmentInstanceId::from(&query_id));
        assert_eq!(retired[0].1, RetireTrigger::CnErr);
    }

    /// Every cancel reason retires the query's parked output and rendezvous state; only the
    /// failure reasons (and a missing reason) record a query-level failure, so a finished
    /// query's result entries are never clobbered. Without a query id nothing is retired.
    #[test]
    fn cancel_reason_matrix_retires_and_records_by_reason() {
        use PPlanFragmentCancelReason as Reason;
        let cases = [
            (Some(Reason::InternalError), "INTERNAL_ERROR", true),
            (Some(Reason::Timeout), "TIMEOUT", true),
            (Some(Reason::UserCancel), "USER_CANCEL", true),
            (None, "none", true),
            (Some(Reason::QueryFinished), "QUERY_FINISHED", false),
            (Some(Reason::LimitReach), "LIMIT_REACH", false),
        ];
        for (index, (reason, name, records_failure)) in cases.into_iter().enumerate() {
            let executor = Arc::new(Retiring::new(StubExecutor));
            let service =
                SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
            let hi = 110 + index as i64;
            let query_id = TUniqueId::new(hi, 1);
            let message = format!("cancelled for {name}");
            cancel_ok(
                &service,
                cancel_request(
                    &TUniqueId::new(hi, 2),
                    Some(&query_id),
                    reason,
                    Some(&message),
                ),
            );

            let retired = executor.retired();
            assert_eq!(retired.len(), 1, "{name}: {retired:?}");
            assert_eq!(retired[0].0, FragmentInstanceId::from(&query_id), "{name}");
            assert_eq!(
                retired[0].1,
                RetireTrigger::Cancel(name.to_string()),
                "{name}"
            );
            assert!(retired[0].2.contains(&message), "{name}: {}", retired[0].2);
            assert_eq!(
                service
                    .core
                    .results
                    .failure_of(FragmentInstanceId::from(&query_id))
                    .is_some(),
                records_failure,
                "{name}"
            );
        }

        let executor = Arc::new(Retiring::new(StubExecutor));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        cancel_ok(
            &service,
            cancel_request(
                &TUniqueId::new(117, 2),
                None,
                Some(Reason::InternalError),
                None,
            ),
        );
        assert!(
            executor.retired().is_empty(),
            "no query id, nothing to retire by query"
        );
    }

    /// The FE's QUERY_FINISHED cancels after eos retire parked state but never touch the result
    /// store: a repeat fetch_data still reports EOS and no query-level failure is recorded.
    #[test]
    fn query_finished_cancel_keeps_delivered_rows() {
        let executor = Arc::new(Retiring::new(CountingExecutor::default()));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let hi = 120;
        let query_id = TUniqueId::new(hi, 1);
        let root_id = TUniqueId::new(hi, 2);
        let middle_id = TUniqueId::new(hi, 3);
        let leaf_id = TUniqueId::new(hi, 4);
        let (root, middle, leaf) = propagation_chain(&query_id, &root_id, &middle_id, &leaf_id);
        assert_exec_ok(&service, &root);
        assert_exec_ok(&service, &middle);
        assert_exec_ok(&service, &leaf);
        let fetched = fetch_rows_eventually(&service, root_id.hi, root_id.lo);
        assert!(!fetched.attachment.is_empty());

        for instance in [&root_id, &middle_id, &leaf_id] {
            cancel_ok(
                &service,
                cancel_request(
                    instance,
                    Some(&query_id),
                    Some(PPlanFragmentCancelReason::QueryFinished),
                    None,
                ),
            );
        }

        let second = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(root_id.hi, root_id.lo),
            Vec::new(),
        );
        let second = PFetchDataResult::decode(second.body.as_slice()).unwrap();
        assert_eq!(
            second.status.status_code,
            TStatusCode::OK.0,
            "{:?}",
            second.status.error_msgs
        );
        assert_eq!(second.eos, Some(true));
        assert!(
            service
                .core
                .results
                .failure_of(FragmentInstanceId::from(&query_id))
                .is_none()
        );
        let retired = executor.retired();
        assert_eq!(
            retired.len(),
            3,
            "one retire per cancelled instance: {retired:?}"
        );
        assert!(
            retired.iter().all(|(id, trigger, _)| {
                *id == FragmentInstanceId::from(&query_id)
                    && *trigger == RetireTrigger::Cancel("QUERY_FINISHED".to_string())
            }),
            "{retired:?}"
        );
    }

    /// An INTERNAL_ERROR cancel for a receiver still waiting on a remote sender purges its staged
    /// frames from the rendezvous (leases back to the arena) and releases the frames the peer's
    /// still-draining sender sends afterwards, dispatching nothing.
    #[test]
    fn cancel_purges_the_receivers_staged_frames_and_refuses_late_ones() {
        let executor = Arc::new(Retiring::new(RecordingExecutor::default()));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(130, 1);
        let receiver_id = TUniqueId::new(130, 2);

        let mut receiver = fragment_params(Some(exchange_plan(7, 0)), Some(desc_table()));
        receiver.fragment.as_mut().unwrap().output_sink = Some(result_sink());
        let mut receiver_exec = exec_params(query_id.clone(), receiver_id.clone());
        receiver_exec.per_exch_num_senders.insert(7, 1);
        receiver.params = Some(receiver_exec);
        assert_exec_ok(&service, &receiver);

        let frame_ok = |seq: i64, eos: bool, offset: u64| {
            let (length, attachment) = if eos {
                (0, Vec::new())
            } else {
                (256, vec![0xAB; 16])
            };
            let frame = route(
                &service,
                methods::TRANSMIT_PACKED,
                transmit_params(
                    &receiver_id,
                    7,
                    0,
                    seq,
                    eos,
                    offset,
                    length,
                    &["id", "name"],
                ),
                attachment,
            );
            let frame = PTransmitPackedResult::decode(frame.body.as_slice()).unwrap();
            assert_eq!(
                frame.status.status_code,
                TStatusCode::OK.0,
                "{:?}",
                frame.status.error_msgs
            );
        };
        frame_ok(0, false, 1024);
        frame_ok(1, false, 2048);

        cancel_ok(
            &service,
            cancel_request(
                &receiver_id,
                Some(&query_id),
                Some(PPlanFragmentCancelReason::InternalError),
                Some("peer failed"),
            ),
        );
        assert_eq!(
            executor.inner.released.lock().unwrap().as_slice(),
            &[1024, 2048],
            "the cancelled receiver's staged leases went back to the arena"
        );

        // The peer is still draining: a late data frame and its eos are released and acked.
        frame_ok(2, false, 3072);
        frame_ok(3, true, 0);
        assert_eq!(
            executor.inner.released.lock().unwrap().as_slice(),
            &[1024, 2048, 3072]
        );
        assert!(
            executor.inner.remote_inputs.lock().unwrap().is_empty(),
            "nothing was dispatched"
        );

        // The receiver's own poll reports the cancel, and the query is failed on this CN.
        let result = fetch_error_eventually(&service, receiver_id.hi, receiver_id.lo);
        assert!(
            result.status.error_msgs[0].contains("cancelled by the FE")
                && result.status.error_msgs[0].contains("peer failed"),
            "{:?}",
            result.status.error_msgs
        );
        let retired = executor.retired();
        assert_eq!(retired.len(), 1, "{retired:?}");
        assert_eq!(
            retired[0].1,
            RetireTrigger::Cancel("INTERNAL_ERROR".to_string())
        );
    }

    /// The FE's phased-schedule cancel names the dummy instance (0, 0) with a real query id: it
    /// must still retire the query while fabricating nothing for the dummy instance.
    #[test]
    fn cancel_for_the_phased_dummy_instance_still_retires_the_query() {
        let executor = Arc::new(Retiring::new(StubExecutor));
        let service = SiriusComputeNodeService::with_executor(executor.clone(), test_identity());
        let query_id = TUniqueId::new(140, 1);
        cancel_ok(
            &service,
            cancel_request(
                &TUniqueId::new(0, 0),
                Some(&query_id),
                Some(PPlanFragmentCancelReason::InternalError),
                None,
            ),
        );
        let retired = executor.retired();
        assert_eq!(retired.len(), 1, "{retired:?}");
        assert_eq!(retired[0].0, FragmentInstanceId::from(&query_id));
        assert_eq!(
            retired[0].1,
            RetireTrigger::Cancel("INTERNAL_ERROR".to_string())
        );
        assert!(
            service
                .core
                .results
                .failure_of(FragmentInstanceId::from(&query_id))
                .is_some()
        );

        // The dummy instance stays unknown to fetch_data and is not a receiver to retire.
        let fetched = route(
            &service,
            methods::FETCH_DATA,
            fetch_request(0, 0),
            Vec::new(),
        );
        let fetched = PFetchDataResult::decode(fetched.body.as_slice()).unwrap();
        assert_eq!(fetched.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        assert!(
            fetched.status.error_msgs[0].contains("no buffered result"),
            "{:?}",
            fetched.status.error_msgs
        );
        assert!(
            !service
                .core
                .exchanges
                .is_retired(FragmentInstanceId::from_halves(0, 0))
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

    /// The 1-CN shuffle shape: a HASH_PARTITIONED stream sink into `dest_node_id` whose one
    /// partition expression is a bare reference to slot `slot_id` of tuple `tuple_id`. When such
    /// a leaf does run (fusion off, or declined) the translator resolves that slot as the
    /// partition key, so the reference must be a real one.
    fn hash_partitioned_data_stream_sink(
        dest_node_id: i32,
        slot_id: i32,
        tuple_id: i32,
    ) -> TDataSink {
        let mut sink = data_stream_sink(dest_node_id);
        sink.stream_sink.as_mut().unwrap().output_partition = TDataPartition::new(
            TPartitionType::HASH_PARTITIONED,
            Some(vec![slot_ref(
                slot_id,
                tuple_id,
                scalar_type(TPrimitiveType::BIGINT),
            )]),
            None,
            None,
        );
        sink
    }

    /// A bare slot reference (one expression node), the shape the FE ships for partition keys
    /// and join equalities.
    fn slot_ref(slot_id: i32, tuple_id: i32, ty: TTypeDesc) -> TExpr {
        TExpr::new(vec![TExprNode {
            node_type: TExprNodeType::SLOT_REF,
            type_: ty,
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

    /// A whole-file `FILES()` parquet range for `path` (`file_size` bytes) in the one shape the
    /// translator accepts: a broker file descriptor read with direct (non-broker) access.
    fn parquet_scan_range(path: &str, file_size: i64) -> TScanRangeParams {
        let range = TBrokerRangeDesc::new(
            TFileType::FILE_BROKER,
            TFileFormatType::FORMAT_PARQUET,
            false,
            path.to_string(),
            0,
            -1,
            None,
            Some(file_size),
            None,
            None,
            None,
            None,
            None,
            None,
        );
        let mut params = TBrokerScanRangeParams::new(
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
        params.file_scan_type = Some(TFileScanType::FILES_QUERY);
        params.use_broker = Some(false);
        let broker = TBrokerScanRange::new(vec![range], params, Vec::new(), None, None, None, None);
        TScanRangeParams::new(
            TScanRange::new(None, None, Some(broker), None, None, None),
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
