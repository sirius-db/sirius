//! gRPC `PBackendService` on `brpc_port`: the analysis-time RPCs behind the `local()` TVF
//! (`fetch_table_schema`, `glob`), fragment dispatch (`exec_plan_fragment*`), cancellation,
//! and result delivery (`fetch_data`). Every other method of the 58 is `unimplemented`.
//!
//! The Doris FE talks to a backend's `brpc_port` with grpc-java (gRPC over h2c); a real BE's
//! `brpc::Server` sniffs h2 and baidu_std on the same port, but nothing on the FE→BE path
//! uses baidu_std, so tonic alone is wire-compatible.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;
use doris_plan_translator::{PlanTranslator, TranslatedPlan};
use doris_proto::p_backend_service_server::{PBackendService, PBackendServiceServer};
use doris_proto::{
    PCancelPlanFragmentRequest, PCancelPlanFragmentResult, PExecPlanFragmentRequest,
    PExecPlanFragmentResult, PExecPlanFragmentStartRequest, PFetchDataRequest, PFetchDataResult,
    PFetchTableSchemaRequest, PFetchTableSchemaResult, PGlobRequest, PGlobResponse,
    PQueryStatistics, PStatus,
};
use doris_thrift::plan_nodes::{TFileFormatType, TFileScanRange};
use doris_thrift::status::TStatusCode;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};
use thrift::transport::TBufferChannel;
use tokio::net::TcpListener;
use tonic::{Request, Response, Status};
use tracing::{info, instrument, warn};

#[cfg(test)]
use crate::fragment_executor::StubExecutor;
use crate::fragment_executor::{FragmentExecutor, FragmentResult};
use crate::params::{self, DispatchedFragment, FragmentBatch};
use crate::result_encoder::{MysqlResultEncoder, ThriftBinary};
use crate::result_store::{FetchOutcome, ResultStore, UniqueId};

/// Environment variable that puts the backend in survey mode: every dispatched batch is
/// accepted (and dumped when [`params::DUMP_FRAGMENTS_ENV`] is set) and translated, but not
/// executed; the query then fails at `fetch_data` with a message naming the translation
/// outcome. On by default (the engine-less build has nothing to execute with); set to `0`
/// to execute.
pub const TRANSLATE_ONLY_ENV: &str = "SIRIUS_BE_TRANSLATE_ONLY";

/// Largest gRPC message accepted/emitted; a plan for a wide TPC-H query or a result batch
/// can exceed tonic's 4 MiB default. The FE side is bounded by `max_msg_size_of_result_receiver`.
const MAX_MESSAGE_SIZE: usize = 2 * 1024 * 1024 * 1024 - 1;

/// Generates `PBackendService` methods that answer gRPC `UNIMPLEMENTED` naming the RPC.
///
/// The generated trait is an `#[async_trait]`; that attribute cannot see into a macro call
/// inside the impl block, so the methods are written in async_trait's desugared form.
macro_rules! unimplemented_rpcs {
    ($($name:ident: ($request:ty, $response:ty)),* $(,)?) => {
        $(
            fn $name<'life0, 'async_trait>(
                &'life0 self,
                _request: Request<$request>,
            ) -> ::core::pin::Pin<
                Box<
                    dyn ::core::future::Future<Output = Result<Response<$response>, Status>>
                        + ::core::marker::Send
                        + 'async_trait,
                >,
            >
            where
                'life0: 'async_trait,
                Self: 'async_trait,
            {
                Box::pin(async move {
                    Err(Status::unimplemented(concat!(
                        "PBackendService.",
                        stringify!($name),
                        " is not implemented"
                    )))
                })
            }
        )*
    };
}

/// Sirius implementation of the Doris `PBackendService`.
#[derive(Clone, Debug)]
pub struct SiriusBackendService {
    /// Reusable Doris thrift-to-Substrait fragment translator.
    translator: PlanTranslator,
    /// Executes a translated plan into Arrow result batches. Production injects the GPU-backed
    /// `SiriusEngine` (via [`with_executor`](Self::with_executor)); tests use a stub.
    executor: Arc<dyn FragmentExecutor>,
    /// Buffers query results for FE `fetch_data` collection, shared across connections.
    results: Arc<ResultStore>,
}

impl SiriusBackendService {
    /// Test-only constructor with the placeholder [`StubExecutor`].
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_executor(Arc::new(StubExecutor))
    }

    /// Builds the service with a caller-provided executor (e.g. the GPU-backed `SiriusEngine`).
    pub fn with_executor(executor: Arc<dyn FragmentExecutor>) -> Self {
        Self {
            translator: PlanTranslator::new(),
            executor,
            results: Arc::new(ResultStore::default()),
        }
    }

    /// Whether survey (translate-only) mode is on: unset or anything but `0`/`false`/`off`.
    fn translate_only() -> bool {
        match std::env::var(TRANSLATE_ONLY_ENV) {
            Ok(value) => !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            ),
            Err(_) => true,
        }
    }

    /// Decodes a dispatch, dumps it, registers its result slot, and runs it to completion on
    /// the calling (blocking) thread. Returns the status the RPC reports.
    fn dispatch(&self, request: &PExecPlanFragmentRequest) -> Result<(), String> {
        let batch = params::decode_fragment_params_list(request)?;
        let query_id = UniqueId::from(&batch.query_id);
        if let Some(dir) = params::dump_batch(request, &batch) {
            info!(%query_id, dir = %dir.display(), "dumped dispatched fragments");
        }
        for fragment in &batch.fragments {
            info!(%query_id, index = fragment.index, shape = %fragment.shape(), "received fragment");
        }
        let result_fragment = batch
            .fragments
            .iter()
            .find(|fragment| fragment.is_result_fragment());
        // Register before doing anything slow so an early `fetch_data` parks instead of
        // failing as unknown. A batch without a RESULT_SINK (a multi-node query whose result
        // fragment runs elsewhere) still gets a slot keyed by query id so cancel/evict work.
        let slot = self.results.register(
            query_id,
            result_fragment
                .into_iter()
                .flat_map(|fragment| fragment.instance_ids().map(UniqueId::from)),
        );

        let translated = self.translate_batch_logged(&batch);
        if Self::translate_only() {
            let outcome = match &translated {
                Ok(plan) => format!(
                    "all {} fragments translated into one plan with output {:?}",
                    batch.fragments.len(),
                    plan.output_names
                ),
                Err(err) => format!("translation failed: {err}"),
            };
            slot.fail(format!(
                "{TRANSLATE_ONLY_ENV} is set: query {query_id} was recorded, not executed ({outcome})"
            ));
            return Ok(());
        }

        // MVP-A0 execution: the whole dispatch runs as one plan on this backend, and its
        // rows are delivered under the RESULT_SINK fragment's instance (the only one the FE
        // fetches). A translation error is reported through the result slot rather than
        // failing dispatch: the FE has already committed the query to us at this point.
        if result_fragment.is_none() {
            slot.fail(format!(
                "query {query_id}: no RESULT_SINK fragment was dispatched to this backend; \
                 multi-node execution is not implemented"
            ));
            return Ok(());
        }
        match translated {
            Ok(plan) => match self.execute_timed(query_id, &plan) {
                Ok(result) => match MysqlResultEncoder::encode(result.batches(), 0) {
                    Ok(batch) => {
                        slot.push(batch);
                        slot.close();
                    }
                    Err(err) => slot.fail(err),
                },
                Err(err) => slot.fail(err),
            },
            Err(err) => slot.fail(err),
        }
        Ok(())
    }

    /// Runs the plan on the executor and logs one line per query with the engine time and row
    /// count (`scripts/run-tpch.sh` reads it back for its timings.csv).
    fn execute_timed(
        &self,
        query_id: UniqueId,
        plan: &TranslatedPlan,
    ) -> Result<FragmentResult, String> {
        let started = Instant::now();
        let outcome = self.executor.execute(plan);
        let engine_ms = (started.elapsed().as_secs_f64() * 1e4).round() / 10.0;
        match &outcome {
            Ok(result) => {
                let rows: usize = result.batches().iter().map(RecordBatch::num_rows).sum();
                info!(%query_id, engine_ms, rows, "query executed on the engine");
            }
            Err(err) => warn!(%query_id, engine_ms, error = %err, "query failed on the engine"),
        }
        outcome
    }

    /// Stitches the batch into one plan (MVP-A0) and translates it, logging the explain
    /// text; on failure, also logs how each fragment fares on its own to narrow down the
    /// offending one (two-phase aggregate halves are expected to be rejected there).
    #[instrument(skip_all, fields(query_id = %UniqueId::from(&batch.query_id)))]
    fn translate_batch_logged(&self, batch: &FragmentBatch) -> Result<TranslatedPlan, String> {
        let fragments: Vec<_> = batch.fragments.iter().map(|f| &f.params).collect();
        match self.translator.translate_batch(&fragments) {
            Ok(translated) => {
                info!(
                    fragments = batch.fragments.len(),
                    output_names = ?translated.output_names,
                    plan = %translated.explain(),
                    "translated Doris dispatch into one plan"
                );
                Ok(translated)
            }
            Err(err) => {
                warn!(error = %err, "failed to translate Doris dispatch");
                for fragment in &batch.fragments {
                    let _ = self.translate_fragment_logged(fragment);
                }
                Err(err.to_string())
            }
        }
    }

    /// Converts one fragment to Substrait on its own and logs the explain output or the error.
    #[instrument(skip_all, fields(index = fragment.index, fragment_id = ?fragment.fragment_id()))]
    fn translate_fragment_logged(
        &self,
        fragment: &DispatchedFragment,
    ) -> Result<TranslatedPlan, String> {
        match self.translator.translate_fragment(&fragment.params) {
            Ok(translated) => {
                info!(
                    output_names = ?translated.output_names,
                    plan = %translated.explain(),
                    "fragment translates on its own"
                );
                Ok(translated)
            }
            Err(err) => {
                let message = format!("fragment {}: {err}", fragment.index);
                warn!(error = %err, "fragment does not translate on its own");
                Err(message)
            }
        }
    }

    /// Runs `dispatch` on a blocking worker (execution can take the whole query) and maps the
    /// outcome to the RPC result.
    async fn dispatch_blocking(
        &self,
        request: PExecPlanFragmentRequest,
    ) -> PExecPlanFragmentResult {
        let service = self.clone();
        let outcome = tokio::task::spawn_blocking(move || service.dispatch(&request)).await;
        let status = match outcome {
            Ok(Ok(())) => ok_status(),
            Ok(Err(err)) => {
                warn!(error = %err, "rejecting fragment dispatch");
                internal_error(err)
            }
            Err(join_err) => internal_error(format!("fragment dispatch task panicked: {join_err}")),
        };
        PExecPlanFragmentResult {
            status,
            received_time: None,
            execution_time: None,
            execution_done_time: None,
        }
    }

    /// Decodes the FE's schema request and infers the parquet schema of its first file.
    async fn table_schema(
        request: &PFetchTableSchemaRequest,
    ) -> Result<PFetchTableSchemaResult, String> {
        let bytes = request
            .file_scan_range
            .as_deref()
            .ok_or_else(|| "PFetchTableSchemaRequest.file_scan_range is missing".to_string())?;
        let scan_range = deserialize_binary::<TFileScanRange>(bytes)
            .map_err(|err| format!("failed to deserialize TFileScanRange: {err}"))?;
        let format = scan_range
            .params
            .as_ref()
            .and_then(|params| params.format_type)
            .or_else(|| {
                scan_range
                    .ranges
                    .as_ref()
                    .and_then(|ranges| ranges.first())
                    .and_then(|range| range.format_type)
            });
        if format != Some(TFileFormatType::FORMAT_PARQUET) {
            return Err(format!(
                "unsupported file format {format:?}; only parquet schema inference is implemented"
            ));
        }
        // The FE always sends exactly the first (non-empty) file it listed.
        let path = scan_range
            .ranges
            .as_ref()
            .and_then(|ranges| ranges.first())
            .and_then(|range| range.path.as_deref())
            .ok_or_else(|| "TFileScanRange carries no file range with a path".to_string())?;
        let columns = crate::file_schema::parquet_file_schema(path).await?;
        Ok(PFetchTableSchemaResult {
            status: Some(ok_status()),
            column_nums: Some(columns.len() as i32),
            column_names: columns.iter().map(|column| column.name.clone()).collect(),
            column_types: columns.into_iter().map(|column| column.type_desc).collect(),
        })
    }
}

#[tonic::async_trait]
impl PBackendService for SiriusBackendService {
    /// One-phase dispatch: decode, dump, translate, and (unless translate-only) execute.
    #[instrument(skip_all)]
    async fn exec_plan_fragment(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        Ok(Response::new(
            self.dispatch_blocking(request.into_inner()).await,
        ))
    }

    /// Two-phase dispatch, phase one. The FE uses prepare/start when a query spans several
    /// fragments on a backend; execution here is store-and-forward and results are only
    /// fetched after `start`, so prepare does all the work and start is an acknowledgement.
    #[instrument(skip_all)]
    async fn exec_plan_fragment_prepare(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        Ok(Response::new(
            self.dispatch_blocking(request.into_inner()).await,
        ))
    }

    #[instrument(skip_all)]
    async fn exec_plan_fragment_start(
        &self,
        request: Request<PExecPlanFragmentStartRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        let query_id = request.get_ref().query_id.as_ref().map(UniqueId::from);
        let status = match query_id.and_then(|id| self.results.get(id)) {
            Some(_) => ok_status(),
            None => internal_error(format!(
                "exec_plan_fragment_start for unknown query {:?}",
                query_id.map(|id| id.to_string())
            )),
        };
        Ok(Response::new(PExecPlanFragmentResult {
            status,
            received_time: None,
            execution_time: None,
            execution_done_time: None,
        }))
    }

    /// The FE cancels every query at the end (`FINISHED` included); drop the result slot.
    #[instrument(skip_all)]
    async fn cancel_plan_fragment(
        &self,
        request: Request<PCancelPlanFragmentRequest>,
    ) -> Result<Response<PCancelPlanFragmentResult>, Status> {
        let request = request.into_inner();
        let id = request
            .query_id
            .as_ref()
            .map(UniqueId::from)
            .unwrap_or_else(|| UniqueId::from(&request.finst_id));
        info!(%id, reason = ?request.cancel_reason, "cancel_plan_fragment");
        self.results.evict(id);
        Ok(Response::new(PCancelPlanFragmentResult {
            status: ok_status(),
        }))
    }

    /// Returns the next result packet, parking until one is ready (as a real BE does).
    #[instrument(skip_all)]
    async fn fetch_data(
        &self,
        request: Request<PFetchDataRequest>,
    ) -> Result<Response<PFetchDataResult>, Status> {
        let id = UniqueId::from(&request.get_ref().finst_id);
        // An unknown id is an error, not EOS: the FE only polls ids it dispatched to us.
        let Some(slot) = self.results.get(id) else {
            return Ok(Response::new(fetch_data_result(
                internal_error(format!("no result registered for {id}")),
                None,
                true,
                None,
            )));
        };
        let result = match slot.fetch().await {
            FetchOutcome::Data { batch, packet_seq } => match batch.to_binary() {
                Ok(bytes) => fetch_data_result(ok_status(), Some(packet_seq), false, Some(bytes)),
                Err(err) => fetch_data_result(internal_error(err), Some(packet_seq), true, None),
            },
            FetchOutcome::Eos {
                packet_seq,
                returned_rows,
            } => {
                let mut result = fetch_data_result(ok_status(), Some(packet_seq), true, None);
                result.query_statistics = Some(PQueryStatistics {
                    returned_rows: Some(returned_rows),
                    ..Default::default()
                });
                result
            }
            FetchOutcome::Failed(message) => {
                fetch_data_result(internal_error(message), None, true, None)
            }
        };
        Ok(Response::new(result))
    }

    /// Infers the schema of a TVF's first file so the FE can resolve its columns.
    #[instrument(skip_all)]
    async fn fetch_table_schema(
        &self,
        request: Request<PFetchTableSchemaRequest>,
    ) -> Result<Response<PFetchTableSchemaResult>, Status> {
        let result = match Self::table_schema(request.get_ref()).await {
            Ok(result) => result,
            Err(err) => {
                warn!(error = %err, "fetch_table_schema failed");
                PFetchTableSchemaResult {
                    status: Some(internal_error(err)),
                    column_nums: Some(0),
                    column_names: Vec::new(),
                    column_types: Vec::new(),
                }
            }
        };
        Ok(Response::new(result))
    }

    /// Expands a `local()` path glob (`shared_storage=true` lets the FE ask one backend).
    #[instrument(skip_all)]
    async fn glob(
        &self,
        request: Request<PGlobRequest>,
    ) -> Result<Response<PGlobResponse>, Status> {
        let pattern = request.into_inner().pattern.unwrap_or_default();
        let response = match crate::file_schema::glob_files(&pattern) {
            Ok(files) => PGlobResponse {
                status: ok_status(),
                files,
            },
            Err(err) => {
                warn!(error = %err, pattern, "glob failed");
                PGlobResponse {
                    status: internal_error(err),
                    files: Vec::new(),
                }
            }
        };
        Ok(Response::new(response))
    }

    unimplemented_rpcs! {
        fetch_arrow_data: (doris_proto::PFetchArrowDataRequest, doris_proto::PFetchArrowDataResult),
        tablet_writer_open: (doris_proto::PTabletWriterOpenRequest, doris_proto::PTabletWriterOpenResult),
        open_load_stream: (doris_proto::POpenLoadStreamRequest, doris_proto::POpenLoadStreamResponse),
        tablet_writer_add_block: (doris_proto::PTabletWriterAddBlockRequest, doris_proto::PTabletWriterAddBlockResult),
        tablet_writer_add_block_by_http: (doris_proto::PEmptyRequest, doris_proto::PTabletWriterAddBlockResult),
        tablet_writer_cancel: (doris_proto::PTabletWriterCancelRequest, doris_proto::PTabletWriterCancelResult),
        get_info: (doris_proto::PProxyRequest, doris_proto::PProxyResult),
        update_cache: (doris_proto::PUpdateCacheRequest, doris_proto::PCacheResponse),
        fetch_cache: (doris_proto::PFetchCacheRequest, doris_proto::PFetchCacheResult),
        clear_cache: (doris_proto::PClearCacheRequest, doris_proto::PCacheResponse),
        send_data: (doris_proto::PSendDataRequest, doris_proto::PSendDataResult),
        commit: (doris_proto::PCommitRequest, doris_proto::PCommitResult),
        rollback: (doris_proto::PRollbackRequest, doris_proto::PRollbackResult),
        merge_filter: (doris_proto::PMergeFilterRequest, doris_proto::PMergeFilterResponse),
        send_filter_size: (doris_proto::PSendFilterSizeRequest, doris_proto::PSendFilterSizeResponse),
        sync_filter_size: (doris_proto::PSyncFilterSizeRequest, doris_proto::PSyncFilterSizeResponse),
        apply_filterv2: (doris_proto::PPublishFilterRequestV2, doris_proto::PPublishFilterResponse),
        fold_constant_expr: (doris_proto::PConstantExprRequest, doris_proto::PConstantExprResult),
        rerun_fragment: (doris_proto::PRerunFragmentParams, doris_proto::PRerunFragmentResult),
        reset_global_rf: (doris_proto::PResetGlobalRfParams, doris_proto::PResetGlobalRfResult),
        transmit_rec_cte_block: (doris_proto::PTransmitRecCteBlockParams, doris_proto::PTransmitRecCteBlockResult),
        transmit_block: (doris_proto::PTransmitDataParams, doris_proto::PTransmitDataResult),
        transmit_block_by_http: (doris_proto::PEmptyRequest, doris_proto::PTransmitDataResult),
        check_rpc_channel: (doris_proto::PCheckRpcChannelRequest, doris_proto::PCheckRpcChannelResponse),
        reset_rpc_channel: (doris_proto::PResetRpcChannelRequest, doris_proto::PResetRpcChannelResponse),
        hand_shake: (doris_proto::PHandShakeRequest, doris_proto::PHandShakeResponse),
        request_slave_tablet_pull_rowset: (doris_proto::PTabletWriteSlaveRequest, doris_proto::PTabletWriteSlaveResult),
        response_slave_tablet_pull_rowset: (doris_proto::PTabletWriteSlaveDoneRequest, doris_proto::PTabletWriteSlaveDoneResult),
        outfile_write_success: (doris_proto::POutfileWriteSuccessRequest, doris_proto::POutfileWriteSuccessResult),
        multiget_data: (doris_proto::PMultiGetRequest, doris_proto::PMultiGetResponse),
        multiget_data_v2: (doris_proto::PMultiGetRequestV2, doris_proto::PMultiGetResponseV2),
        get_file_cache_meta_by_tablet_id: (doris_proto::PGetFileCacheMetaRequest, doris_proto::PGetFileCacheMetaResponse),
        warm_up_rowset: (doris_proto::PWarmUpRowsetRequest, doris_proto::PWarmUpRowsetResponse),
        recycle_cache: (doris_proto::PRecycleCacheRequest, doris_proto::PRecycleCacheResponse),
        tablet_fetch_data: (doris_proto::PTabletKeyLookupRequest, doris_proto::PTabletKeyLookupResponse),
        get_column_ids_by_tablet_ids: (doris_proto::PFetchColIdsRequest, doris_proto::PFetchColIdsResponse),
        get_tablet_rowset_versions: (doris_proto::PGetTabletVersionsRequest, doris_proto::PGetTabletVersionsResponse),
        report_stream_load_status: (doris_proto::PReportStreamLoadStatusRequest, doris_proto::PReportStreamLoadStatusResponse),
        group_commit_insert: (doris_proto::PGroupCommitInsertRequest, doris_proto::PGroupCommitInsertResponse),
        get_wal_queue_size: (doris_proto::PGetWalQueueSizeRequest, doris_proto::PGetWalQueueSizeResponse),
        fetch_arrow_flight_schema: (doris_proto::PFetchArrowFlightSchemaRequest, doris_proto::PFetchArrowFlightSchemaResult),
        fetch_remote_tablet_schema: (doris_proto::PFetchRemoteSchemaRequest, doris_proto::PFetchRemoteSchemaResponse),
        test_jdbc_connection: (doris_proto::PJdbcTestConnectionRequest, doris_proto::PJdbcTestConnectionResult),
        alter_vault_sync: (doris_proto::PAlterVaultSyncRequest, doris_proto::PAlterVaultSyncResponse),
        get_be_resource: (doris_proto::PGetBeResourceRequest, doris_proto::PGetBeResourceResponse),
        delete_dictionary: (doris_proto::PDeleteDictionaryRequest, doris_proto::PDeleteDictionaryResponse),
        commit_refresh_dictionary: (doris_proto::PCommitRefreshDictionaryRequest, doris_proto::PCommitRefreshDictionaryResponse),
        abort_refresh_dictionary: (doris_proto::PAbortRefreshDictionaryRequest, doris_proto::PAbortRefreshDictionaryResponse),
        get_tablet_rowsets: (doris_proto::PGetTabletRowsetsRequest, doris_proto::PGetTabletRowsetsResponse),
        fetch_peer_data: (doris_proto::PFetchPeerDataRequest, doris_proto::PFetchPeerDataResponse),
        request_cdc_client: (doris_proto::PRequestCdcClientRequest, doris_proto::PRequestCdcClientResult),
    }
}

/// Serves the gRPC `PBackendService` on an already-bound listener until `shutdown` resolves.
///
/// Taking a bound listener lets the caller fail fast on a port clash and lets tests bind
/// port zero.
pub async fn serve_backend_service(
    listener: TcpListener,
    service: SiriusBackendService,
    shutdown: impl std::future::Future<Output = ()>,
) -> Result<(), tonic::transport::Error> {
    let local_addr: SocketAddr = listener
        .local_addr()
        .expect("bound listener has a local address");
    info!(address = %local_addr, "starting PBackendService gRPC server");
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let result = tonic::transport::Server::builder()
        .add_service(
            PBackendServiceServer::new(service)
                .max_decoding_message_size(MAX_MESSAGE_SIZE)
                .max_encoding_message_size(MAX_MESSAGE_SIZE),
        )
        .serve_with_incoming_shutdown(incoming, shutdown)
        .await;
    info!("PBackendService gRPC server stopped");
    result
}

/// Deserializes a thrift struct using the binary protocol (what the FE's `TSerializer`
/// default emits for `file_scan_range`).
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

/// Builds a `fetch_data` response.
fn fetch_data_result(
    status: PStatus,
    packet_seq: Option<i64>,
    eos: bool,
    row_batch: Option<Vec<u8>>,
) -> PFetchDataResult {
    PFetchDataResult {
        status,
        packet_seq,
        eos: Some(eos),
        query_statistics: None,
        row_batch,
        empty_batch: None,
    }
}

/// Doris OK status.
fn ok_status() -> PStatus {
    PStatus {
        status_code: TStatusCode::OK.0,
        error_msgs: Vec::new(),
    }
}

/// Doris INTERNAL_ERROR status carrying a user-visible error message.
fn internal_error(message: impl Into<String>) -> PStatus {
    PStatus {
        status_code: TStatusCode::INTERNAL_ERROR.0,
        error_msgs: vec![message.into()],
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use doris_proto::p_backend_service_client::PBackendServiceClient;
    use doris_proto::{PFragmentRequestVersion, PUniqueId};
    use doris_thrift::data::TResultBatch;
    use doris_thrift::data_sinks::{TDataSink, TDataSinkType};
    use doris_thrift::descriptors::TDescriptorTable;
    use doris_thrift::palo_internal_service::{
        PaloInternalServiceVersion, TPipelineFragmentParams, TPipelineFragmentParamsList,
        TPipelineInstanceParams,
    };
    use doris_thrift::partitions::{TDataPartition, TPartitionType};
    use doris_thrift::plan_nodes::{
        TFileRangeDesc, TFileScanRangeParams, TPlan, TPlanNode, TPlanNodeType,
    };
    use doris_thrift::planner::TPlanFragment;
    use doris_thrift::types::TUniqueId;
    use thrift::protocol::{TBinaryOutputProtocol, TCompactOutputProtocol, TOutputProtocol};
    use tokio_util::sync::CancellationToken;

    use super::*;

    /// Serializes a thrift struct with the binary protocol (the FE's `TSerializer` default).
    fn serialize_binary<T: TSerializable>(value: &T) -> Vec<u8> {
        let mut buffer = Vec::new();
        let mut protocol = TBinaryOutputProtocol::new(&mut buffer, true);
        value.write_to_out_protocol(&mut protocol).unwrap();
        protocol.flush().unwrap();
        buffer
    }

    /// Serializes a thrift struct with the compact protocol (`use_compact_thrift_rpc`).
    fn serialize_compact<T: TSerializable>(value: &T) -> Vec<u8> {
        let mut buffer = Vec::new();
        let mut protocol = TCompactOutputProtocol::new(&mut buffer);
        value.write_to_out_protocol(&mut protocol).unwrap();
        protocol.flush().unwrap();
        buffer
    }

    /// A single-fragment RESULT_SINK dispatch over a file scan, query (1,2) instance (1,3).
    fn result_dispatch() -> PExecPlanFragmentRequest {
        let params = TPipelineFragmentParams {
            protocol_version: PaloInternalServiceVersion::V1,
            query_id: TUniqueId::new(1, 2),
            fragment_id: Some(0),
            desc_tbl: Some(TDescriptorTable::default()),
            fragment: Some(TPlanFragment {
                plan: Some(TPlan {
                    nodes: vec![TPlanNode {
                        node_id: 0,
                        node_type: TPlanNodeType::FILE_SCAN_NODE,
                        num_children: 0,
                        limit: -1,
                        row_tuples: vec![0],
                        nullable_tuples: vec![false],
                        compact_data: false,
                        ..Default::default()
                    }],
                }),
                output_sink: Some(TDataSink {
                    type_: TDataSinkType::RESULT_SINK,
                    ..Default::default()
                }),
                partition: TDataPartition {
                    type_: TPartitionType::UNPARTITIONED,
                    ..Default::default()
                },
                ..Default::default()
            }),
            local_params: Some(vec![TPipelineInstanceParams {
                fragment_instance_id: TUniqueId::new(1, 3),
                ..Default::default()
            }]),
            ..Default::default()
        };
        let list = TPipelineFragmentParamsList {
            params_list: Some(vec![params]),
            ..Default::default()
        };
        PExecPlanFragmentRequest {
            request: Some(serialize_compact(&list)),
            compact: Some(true),
            version: Some(PFragmentRequestVersion::Version3 as i32),
        }
    }

    /// Starts the gRPC server on a loopback port-0 listener and returns a connected client
    /// plus the token that stops the server.
    async fn start_server(
        service: SiriusBackendService,
    ) -> (
        PBackendServiceClient<tonic::transport::Channel>,
        CancellationToken,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let token = CancellationToken::new();
        let shutdown = token.clone();
        tokio::spawn(async move {
            serve_backend_service(listener, service, shutdown.cancelled_owned())
                .await
                .unwrap();
        });
        let client = PBackendServiceClient::connect(format!("http://{addr}"))
            .await
            .unwrap();
        (client, token)
    }

    /// Runs a closure with `TRANSLATE_ONLY_ENV` forced to `value` for its duration.
    /// Tests that touch the variable are serialized through this lock.
    fn with_translate_only<T>(value: Option<&str>, body: impl FnOnce() -> T) -> T {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(TRANSLATE_ONLY_ENV);
        // SAFETY: tests touching this variable are serialized by LOCK and restore it before
        // releasing the lock.
        unsafe {
            match value {
                Some(value) => std::env::set_var(TRANSLATE_ONLY_ENV, value),
                None => std::env::remove_var(TRANSLATE_ONLY_ENV),
            }
        }
        let result = body();
        unsafe {
            match previous {
                Some(previous) => std::env::set_var(TRANSLATE_ONLY_ENV, previous),
                None => std::env::remove_var(TRANSLATE_ONLY_ENV),
            }
        }
        result
    }

    #[test]
    fn translate_only_defaults_on_and_honors_off_values() {
        with_translate_only(None, || assert!(SiriusBackendService::translate_only()));
        with_translate_only(
            Some("1"),
            || assert!(SiriusBackendService::translate_only()),
        );
        with_translate_only(Some("0"), || {
            assert!(!SiriusBackendService::translate_only())
        });
        with_translate_only(Some("false"), || {
            assert!(!SiriusBackendService::translate_only())
        });
    }

    #[test]
    fn translate_only_dispatch_accepts_and_fails_at_fetch() {
        let service = SiriusBackendService::new();
        with_translate_only(Some("1"), || service.dispatch(&result_dispatch())).unwrap();

        // Both the query id and the result instance id resolve to the same slot.
        let query = UniqueId::from_halves(1, 2);
        let instance = UniqueId::from_halves(1, 3);
        let slot = service.results.get(instance).unwrap();
        assert!(Arc::ptr_eq(&slot, &service.results.get(query).unwrap()));
        let outcome = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(slot.fetch());
        match outcome {
            FetchOutcome::Failed(message) => {
                assert!(message.contains(TRANSLATE_ONLY_ENV), "{message}");
                assert!(
                    message.contains(&format!("query {query} was recorded, not executed")),
                    "{message}"
                );
                // The fixture's scan carries no ranges, which the translator refuses.
                assert!(message.contains("translation failed"), "{message}");
                assert!(
                    message.contains("unsupported scan range at node 0"),
                    "{message}"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    /// The captured TPC-H Q6 dispatch (two fragments, a two-phase aggregate) translates as one
    /// plan, and survey mode says so instead of executing.
    #[test]
    fn translate_only_reports_a_stitched_batch_as_recorded() {
        let payload = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/tpch/q06/batch-00-request.tcompact");
        let request = PExecPlanFragmentRequest {
            request: Some(std::fs::read(payload).unwrap()),
            compact: Some(true),
            version: Some(PFragmentRequestVersion::Version3 as i32),
        };
        let service = SiriusBackendService::new();
        with_translate_only(Some("1"), || service.dispatch(&request)).unwrap();
        let batch = params::decode_fragment_params_list(&request).unwrap();
        let slot = service
            .results
            .get(UniqueId::from(&batch.query_id))
            .unwrap();
        let outcome = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(slot.fetch());
        match outcome {
            FetchOutcome::Failed(message) => {
                assert!(message.contains("was recorded, not executed"), "{message}");
                assert!(
                    message.contains("all 2 fragments translated into one plan"),
                    "{message}"
                );
                assert!(message.contains("[\"revenue\"]"), "{message}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn execute_mode_reports_translator_errors_through_the_slot() {
        let service = SiriusBackendService::new();
        with_translate_only(Some("0"), || service.dispatch(&result_dispatch())).unwrap();
        let slot = service.results.get(UniqueId::from_halves(1, 2)).unwrap();
        let outcome = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(slot.fetch());
        assert!(
            matches!(&outcome, FetchOutcome::Failed(message) if message.contains("unsupported scan range at node 0")),
            "{outcome:?}"
        );
    }

    #[test]
    fn dispatch_rejects_bad_payload() {
        let service = SiriusBackendService::new();
        let request = PExecPlanFragmentRequest {
            request: Some(b"not thrift".to_vec()),
            compact: Some(true),
            version: Some(PFragmentRequestVersion::Version3 as i32),
        };
        let err = service.dispatch(&request).unwrap_err();
        assert!(err.contains("failed to deserialize"), "{err}");
        assert_eq!(service.results.len(), 0);
    }

    #[tokio::test]
    async fn fetch_data_streams_batches_then_eos_over_grpc() {
        let service = SiriusBackendService::new();
        let slot = service.results.register(UniqueId::from_halves(5, 6), []);
        slot.push(TResultBatch::new(vec![vec![0x01, b'x']], false, 0, None));
        slot.close();
        let (mut client, token) = start_server(service).await;

        let finst = PUniqueId { hi: 5, lo: 6 };
        let first = client
            .fetch_data(PFetchDataRequest {
                finst_id: finst,
                resp_in_attachment: Some(false),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(first.status.status_code, TStatusCode::OK.0);
        assert_eq!(first.packet_seq, Some(0));
        assert_eq!(first.eos, Some(false));
        assert!(
            first
                .row_batch
                .as_ref()
                .is_some_and(|bytes| !bytes.is_empty())
        );

        let second = client
            .fetch_data(PFetchDataRequest {
                finst_id: finst,
                resp_in_attachment: Some(false),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(second.status.status_code, TStatusCode::OK.0);
        assert_eq!(second.packet_seq, Some(1));
        assert_eq!(second.eos, Some(true));
        assert!(second.row_batch.is_none());
        assert_eq!(
            second
                .query_statistics
                .and_then(|stats| stats.returned_rows),
            Some(1)
        );

        let unknown = client
            .fetch_data(PFetchDataRequest {
                finst_id: PUniqueId { hi: 9, lo: 9 },
                resp_in_attachment: Some(false),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(unknown.status.status_code, TStatusCode::INTERNAL_ERROR.0);
        token.cancel();
    }

    #[tokio::test]
    async fn cancel_evicts_and_start_checks_registration_over_grpc() {
        let service = SiriusBackendService::new();
        service.results.register(UniqueId::from_halves(5, 6), []);
        let (mut client, token) = start_server(service).await;

        let started = client
            .exec_plan_fragment_start(PExecPlanFragmentStartRequest {
                query_id: Some(PUniqueId { hi: 5, lo: 6 }),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(started.status.status_code, TStatusCode::OK.0);

        let cancelled = client
            .cancel_plan_fragment(PCancelPlanFragmentRequest {
                finst_id: PUniqueId { hi: 0, lo: 0 },
                cancel_reason: None,
                query_id: Some(PUniqueId { hi: 5, lo: 6 }),
                fragment_id: None,
                cancel_status: None,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(cancelled.status.status_code, TStatusCode::OK.0);

        let started = client
            .exec_plan_fragment_start(PExecPlanFragmentStartRequest {
                query_id: Some(PUniqueId { hi: 5, lo: 6 }),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(started.status.status_code, TStatusCode::INTERNAL_ERROR.0);

        let err = client
            .hand_shake(doris_proto::PHandShakeRequest::default())
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unimplemented);
        assert!(err.message().contains("hand_shake"));
        token.cancel();
    }

    #[tokio::test]
    async fn fetch_table_schema_and_glob_over_grpc() {
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("part-0.parquet");
        {
            let schema = Arc::new(
                parse_message_type("message t { optional int64 id; optional binary s (UTF8); }")
                    .unwrap(),
            );
            let file = std::fs::File::create(&path).unwrap();
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap()
                .close()
                .unwrap();
        }
        let (mut client, token) = start_server(SiriusBackendService::new()).await;

        let listed = client
            .glob(PGlobRequest {
                pattern: Some(format!("{}/*.parquet", dir.path().display())),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(listed.status.status_code, TStatusCode::OK.0);
        assert_eq!(listed.files.len(), 1);
        assert_eq!(
            listed.files[0].file.as_deref(),
            Some(path.to_str().unwrap())
        );

        let scan_range = TFileScanRange {
            ranges: Some(vec![TFileRangeDesc {
                path: Some(path.to_str().unwrap().to_string()),
                ..Default::default()
            }]),
            params: Some(TFileScanRangeParams {
                format_type: Some(TFileFormatType::FORMAT_PARQUET),
                ..Default::default()
            }),
            split_source: None,
        };
        let schema = client
            .fetch_table_schema(PFetchTableSchemaRequest {
                file_scan_range: Some(serialize_binary(&scan_range)),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(schema.status.unwrap().status_code, TStatusCode::OK.0);
        assert_eq!(schema.column_nums, Some(2));
        assert_eq!(schema.column_names, vec!["id", "s"]);
        assert_eq!(schema.column_types.len(), 2);

        let mut csv = scan_range.clone();
        csv.params.as_mut().unwrap().format_type = Some(TFileFormatType::FORMAT_CSV_PLAIN);
        let rejected = client
            .fetch_table_schema(PFetchTableSchemaRequest {
                file_scan_range: Some(serialize_binary(&csv)),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            rejected.status.unwrap().status_code,
            TStatusCode::INTERNAL_ERROR.0
        );
        token.cancel();
    }
}
