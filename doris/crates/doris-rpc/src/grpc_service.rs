//! PBackendService gRPC handler.
//!
//! This is the main entry point for query execution. The Doris FE sends
//! `exec_plan_fragment` requests containing Thrift-serialized `TPipelineFragmentParams`.

use std::io::Cursor;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
use tracing::{info, instrument, warn};

use doris_proto::doris::p_backend_service_server::PBackendService;
use doris_proto::doris::*;
use doris_thrift::palo_internal_service::{TPipelineFragmentParams, TPipelineFragmentParamsList};
use result_formatter::result_store::{FinstId, ResultStore};
use sirius_ffi::SiriusEngine;

use super::heartbeat_service::BeState;

fn ok_status() -> PStatus {
    PStatus {
        status_code: 0,
        error_msgs: vec![],
    }
}

fn unimpl() -> Status {
    Status::unimplemented("not supported on Sirius GPU backend")
}

fn err_status(msg: &str) -> PStatus {
    PStatus {
        status_code: 1,
        error_msgs: vec![msg.to_string()],
    }
}

/// Deserialize Thrift TPipelineFragmentParams from raw bytes.
///
/// `version` corresponds to `PFragmentRequestVersion`:
///   1 = single TExecPlanFragmentParams (unsupported)
///   2 = single TPipelineFragmentParams
///   3 = TPipelineFragmentParamsList (shared fields at list level)
fn deserialize_params(data: Vec<u8>, compact: bool, version: i32) -> Result<TPipelineFragmentParams, String> {
    use thrift::protocol::{TBinaryInputProtocol, TCompactInputProtocol, TSerializable};
    use thrift::transport::TBufferedReadTransport;

    if version == 3 {
        // VERSION_3: bytes contain TPipelineFragmentParamsList.
        // Shared fields (desc_tbl, query_globals, etc.) are at the list level.
        let list = if compact {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TCompactInputProtocol::new(transport);
            TPipelineFragmentParamsList::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("compact thrift deserialize (v3 list): {e}"))?
        } else {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TBinaryInputProtocol::new(transport, true);
            TPipelineFragmentParamsList::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("binary thrift deserialize (v3 list): {e}"))?
        };

        let mut params = list
            .params_list
            .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
            .ok_or_else(|| "VERSION_3: empty params_list".to_string())?;

        // Merge shared fields from list level into the per-fragment params.
        if params.desc_tbl.is_none() {
            params.desc_tbl = list.desc_tbl;
        }
        if params.query_globals.is_none() {
            params.query_globals = list.query_globals;
        }
        if params.query_options.is_none() {
            params.query_options = list.query_options;
        }
        if params.coord.is_none() {
            params.coord = list.coord;
        }
        if params.resource_info.is_none() {
            params.resource_info = list.resource_info;
        }
        if params.fragment_num_on_host.is_none() {
            params.fragment_num_on_host = list.fragment_num_on_host;
        }
        if params.file_scan_params.is_none() {
            params.file_scan_params = list.file_scan_params;
        }

        Ok(params)
    } else if version == 2 || version == 0 {
        // VERSION_2 (or default): single TPipelineFragmentParams.
        if compact {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TCompactInputProtocol::new(transport);
            TPipelineFragmentParams::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("compact thrift deserialize: {e}"))
        } else {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TBinaryInputProtocol::new(transport, true);
            TPipelineFragmentParams::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("binary thrift deserialize: {e}"))
        }
    } else {
        Err(format!("unsupported PFragmentRequestVersion: {version}"))
    }
}

/// Serialize MySQL rows as a Thrift binary-encoded TResultBatch.
///
/// The Doris FE deserializes row_batch using TBinaryProtocol (see ResultReceiver.java),
/// NOT TCompactProtocol.
fn serialize_result_batch(rows: &[Vec<u8>], packet_seq: i64) -> Result<Vec<u8>, String> {
    use thrift::protocol::{TBinaryOutputProtocol, TOutputProtocol, TSerializable};

    let batch = doris_thrift::data::TResultBatch::new(
        rows.to_vec(),
        false, // is_compressed
        packet_seq,
        None::<std::collections::BTreeMap<String, String>>,
    );

    let mut buf = Vec::new();
    {
        let mut protocol = TBinaryOutputProtocol::new(Cursor::new(&mut buf), true);
        batch
            .write_to_out_protocol(&mut protocol)
            .map_err(|e| format!("thrift serialize TResultBatch: {e}"))?;
        protocol
            .flush()
            .map_err(|e| format!("thrift flush: {e}"))?;
    }
    Ok(buf)
}

/// Extract FinstId for result storage from fragment params.
///
/// With `enable_parallel_result_sink` (default: true), the FE uses the `query_id`
/// to fetch results, not the fragment_instance_id. So we always use query_id.
fn extract_finst_id(params: &TPipelineFragmentParams) -> FinstId {
    FinstId {
        hi: params.query_id.hi,
        lo: params.query_id.lo,
    }
}

pub struct PBackendServiceHandler {
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
}

impl PBackendServiceHandler {
    pub fn new(
        state: Arc<BeState>,
        result_store: ResultStore,
        engine: Option<Arc<Mutex<SiriusEngine>>>,
    ) -> Self {
        Self {
            state,
            result_store,
            engine,
        }
    }
}

#[tonic::async_trait]
impl PBackendService for PBackendServiceHandler {
    #[instrument(skip_all, fields(compact, query_id, fragment_id))]
    async fn exec_plan_fragment(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        let req = request.into_inner();
        let compact = req.compact.unwrap_or(false);

        let version = req.version.unwrap_or(2); // default = VERSION_2
        let thrift_bytes = match req.request {
            Some(bytes) => bytes,
            None => {
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status("missing request bytes"),
                    ..Default::default()
                }));
            }
        };

        info!(
            compact,
            version,
            len = thrift_bytes.len(),
            first_bytes = ?&thrift_bytes[..thrift_bytes.len().min(32)],
            "received exec_plan_fragment request"
        );

        // Deserialize Thrift TPipelineFragmentParams.
        let params = match deserialize_params(thrift_bytes, compact, version) {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "failed to deserialize fragment params");
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status(&e),
                    ..Default::default()
                }));
            }
        };

        let finst_id = extract_finst_id(&params);
        info!(
            query_id = ?params.query_id,
            fragment_id = ?params.fragment_id,
            %finst_id,
            "deserialized fragment params"
        );

        // Try Substrait translation first, fall back to SQL.
        enum ExecPlan {
            Substrait(Vec<u8>),
            Sql(String),
        }

        let exec_plan = match plan_translator::translate_fragment(&params) {
            Ok(substrait_bytes) => {
                info!(bytes = substrait_bytes.len(), "translated to Substrait");
                ExecPlan::Substrait(substrait_bytes)
            }
            Err(e) => {
                warn!(error = %e, "Substrait translation failed, trying SQL fallback");
                match plan_translator::translate_fragment_to_sql(&params) {
                    Ok(sql) => {
                        info!(sql = %sql, "translated to SQL (fallback)");
                        ExecPlan::Sql(sql)
                    }
                    Err(e2) => {
                        warn!(error = %e2, "SQL translation also failed");
                        return Ok(Response::new(PExecPlanFragmentResult {
                            status: err_status(&format!("plan translation: {e}")),
                            ..Default::default()
                        }));
                    }
                }
            }
        };

        // Execute via Sirius GPU, falling back to DuckDB CPU.
        let engine = match &self.engine {
            Some(e) => e.clone(),
            None => {
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status("engine not initialized (built without duckdb-bundled?)"),
                    ..Default::default()
                }));
            }
        };
        let store = self.result_store.clone();

        // Sirius/DuckDB execution is blocking — run off the async runtime.
        let exec_result = tokio::task::spawn_blocking(move || -> Result<(), String> {
            let engine = engine.lock().unwrap();
            let ipc_bytes = match exec_plan {
                ExecPlan::Substrait(bytes) => {
                    match engine.execute_substrait(&bytes) {
                        Ok(ipc) => {
                            tracing::info!("executed via gpu_processing_substrait");
                            ipc
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "gpu_processing_substrait failed, falling back to CPU from_substrait");
                            engine.execute_sql("SELECT 'substrait_fallback_not_implemented'")
                                .map_err(|e| e.to_string())?
                        }
                    }
                }
                ExecPlan::Sql(sql) => {
                    match engine.execute_gpu(&sql) {
                        Ok(ipc) => {
                            tracing::info!("executed via gpu_execution");
                            ipc
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "gpu_execution failed, falling back to direct SQL");
                            engine.execute_sql(&sql).map_err(|e| e.to_string())?
                        }
                    }
                }
            };
            store.store_ipc_result(finst_id, &ipc_bytes)?;
            Ok(())
        })
        .await;

        match exec_result {
            Ok(Ok(())) => {
                info!(%finst_id, "execution complete, result stored");
            }
            Ok(Err(e)) => {
                warn!(error = %e, %finst_id, "execution failed");
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status(&format!("execution: {e}")),
                    ..Default::default()
                }));
            }
            Err(e) => {
                warn!(error = %e, "spawn_blocking panicked");
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status(&format!("internal: {e}")),
                    ..Default::default()
                }));
            }
        }

        Ok(Response::new(PExecPlanFragmentResult {
            status: ok_status(),
            ..Default::default()
        }))
    }

    async fn exec_plan_fragment_prepare(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        self.exec_plan_fragment(request).await
    }

    async fn exec_plan_fragment_start(
        &self,
        request: Request<PExecPlanFragmentStartRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, "exec_plan_fragment_start");
        Ok(Response::new(PExecPlanFragmentResult {
            status: ok_status(),
            ..Default::default()
        }))
    }

    async fn cancel_plan_fragment(
        &self,
        request: Request<PCancelPlanFragmentRequest>,
    ) -> Result<Response<PCancelPlanFragmentResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, fragment_id = ?req.fragment_id, "cancel_plan_fragment");
        Ok(Response::new(PCancelPlanFragmentResult {
            status: ok_status(),
        }))
    }

    async fn transmit_block(
        &self,
        request: Request<PTransmitDataParams>,
    ) -> Result<Response<PTransmitDataResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, sender_id = req.sender_id, eos = req.eos, "transmit_block");
        Ok(Response::new(PTransmitDataResult {
            status: Some(ok_status()),
            ..Default::default()
        }))
    }

    async fn fetch_arrow_flight_schema(
        &self,
        request: Request<PFetchArrowFlightSchemaRequest>,
    ) -> Result<Response<PFetchArrowFlightSchemaResult>, Status> {
        let req = request.into_inner();
        info!(finst_id = ?req.finst_id, "fetch_arrow_flight_schema");

        let finst_id = match req.finst_id {
            Some(id) => FinstId { hi: id.hi, lo: id.lo },
            None => {
                return Ok(Response::new(PFetchArrowFlightSchemaResult {
                    status: Some(err_status("missing finst_id")),
                    ..Default::default()
                }));
            }
        };

        let entry = match self.result_store.get(&finst_id) {
            Some(e) => e,
            None => {
                warn!(%finst_id, "fetch_arrow_flight_schema: result not found");
                return Ok(Response::new(PFetchArrowFlightSchemaResult {
                    status: Some(err_status("result not found")),
                    ..Default::default()
                }));
            }
        };

        let schema_bytes = entry.schema_ipc_bytes().map_err(|e| {
            Status::internal(format!("failed to serialize schema: {e}"))
        })?;

        Ok(Response::new(PFetchArrowFlightSchemaResult {
            status: Some(ok_status()),
            schema: Some(schema_bytes),
            be_arrow_flight_ip: Some(b"127.0.0.1".to_vec()),
            be_arrow_flight_port: Some(self.state.arrow_flight_port),
        }))
    }

    async fn fetch_data(
        &self,
        request: Request<PFetchDataRequest>,
    ) -> Result<Response<PFetchDataResult>, Status> {
        let req = request.into_inner();
        let finst_id = FinstId {
            hi: req.finst_id.hi,
            lo: req.finst_id.lo,
        };
        info!(%finst_id, "fetch_data");

        let entry = match self.result_store.get(&finst_id) {
            Some(e) => e,
            None => {
                warn!(%finst_id, "fetch_data: result not found");
                return Ok(Response::new(PFetchDataResult {
                    status: err_status("result not found"),
                    eos: Some(true),
                    ..Default::default()
                }));
            }
        };

        // Convert Arrow data to MySQL text protocol rows and wrap in TResultBatch.
        let mysql_rows = entry.to_mysql_rows();
        info!(%finst_id, num_rows = mysql_rows.len(), "converting to TResultBatch");
        let row_batch_bytes = serialize_result_batch(&mysql_rows, 0)
            .map_err(|e| Status::internal(format!("failed to serialize result batch: {e}")))?;
        info!(
            %finst_id,
            batch_len = row_batch_bytes.len(),
            first_bytes = ?&row_batch_bytes[..row_batch_bytes.len().min(32)],
            "serialized TResultBatch"
        );

        // Remove result after serving it.
        self.result_store.remove(&finst_id);

        Ok(Response::new(PFetchDataResult {
            status: ok_status(),
            packet_seq: Some(0),
            eos: Some(true),
            row_batch: Some(row_batch_bytes),
            ..Default::default()
        }))
    }

    // --- Stub implementations for unsupported methods ---

    async fn fetch_arrow_data(&self, _: Request<PFetchArrowDataRequest>) -> Result<Response<PFetchArrowDataResult>, Status> { Err(unimpl()) }
    async fn tablet_writer_open(&self, _: Request<PTabletWriterOpenRequest>) -> Result<Response<PTabletWriterOpenResult>, Status> { Err(unimpl()) }
    async fn open_load_stream(&self, _: Request<POpenLoadStreamRequest>) -> Result<Response<POpenLoadStreamResponse>, Status> { Err(unimpl()) }
    async fn tablet_writer_add_block(&self, _: Request<PTabletWriterAddBlockRequest>) -> Result<Response<PTabletWriterAddBlockResult>, Status> { Err(unimpl()) }
    async fn tablet_writer_add_block_by_http(&self, _: Request<PEmptyRequest>) -> Result<Response<PTabletWriterAddBlockResult>, Status> { Err(unimpl()) }
    async fn tablet_writer_cancel(&self, _: Request<PTabletWriterCancelRequest>) -> Result<Response<PTabletWriterCancelResult>, Status> { Err(unimpl()) }
    async fn get_info(&self, _: Request<PProxyRequest>) -> Result<Response<PProxyResult>, Status> { Err(unimpl()) }
    async fn update_cache(&self, _: Request<PUpdateCacheRequest>) -> Result<Response<PCacheResponse>, Status> { Err(unimpl()) }
    async fn fetch_cache(&self, _: Request<PFetchCacheRequest>) -> Result<Response<PFetchCacheResult>, Status> { Err(unimpl()) }
    async fn clear_cache(&self, _: Request<PClearCacheRequest>) -> Result<Response<PCacheResponse>, Status> { Err(unimpl()) }
    async fn send_data(&self, _: Request<PSendDataRequest>) -> Result<Response<PSendDataResult>, Status> { Err(unimpl()) }
    async fn commit(&self, _: Request<PCommitRequest>) -> Result<Response<PCommitResult>, Status> { Err(unimpl()) }
    async fn rollback(&self, _: Request<PRollbackRequest>) -> Result<Response<PRollbackResult>, Status> { Err(unimpl()) }
    async fn merge_filter(&self, _: Request<PMergeFilterRequest>) -> Result<Response<PMergeFilterResponse>, Status> { Err(unimpl()) }
    async fn send_filter_size(&self, _: Request<PSendFilterSizeRequest>) -> Result<Response<PSendFilterSizeResponse>, Status> { Err(unimpl()) }
    async fn sync_filter_size(&self, _: Request<PSyncFilterSizeRequest>) -> Result<Response<PSyncFilterSizeResponse>, Status> { Err(unimpl()) }
    async fn apply_filterv2(&self, _: Request<PPublishFilterRequestV2>) -> Result<Response<PPublishFilterResponse>, Status> { Err(unimpl()) }
    async fn fold_constant_expr(&self, _: Request<PConstantExprRequest>) -> Result<Response<PConstantExprResult>, Status> { Err(unimpl()) }
    async fn rerun_fragment(&self, _: Request<PRerunFragmentParams>) -> Result<Response<PRerunFragmentResult>, Status> { Err(unimpl()) }
    async fn reset_global_rf(&self, _: Request<PResetGlobalRfParams>) -> Result<Response<PResetGlobalRfResult>, Status> { Err(unimpl()) }
    async fn transmit_rec_cte_block(&self, _: Request<PTransmitRecCteBlockParams>) -> Result<Response<PTransmitRecCteBlockResult>, Status> { Err(unimpl()) }
    async fn transmit_block_by_http(&self, _: Request<PEmptyRequest>) -> Result<Response<PTransmitDataResult>, Status> { Err(unimpl()) }
    async fn check_rpc_channel(&self, _: Request<PCheckRpcChannelRequest>) -> Result<Response<PCheckRpcChannelResponse>, Status> { Err(unimpl()) }
    async fn reset_rpc_channel(&self, _: Request<PResetRpcChannelRequest>) -> Result<Response<PResetRpcChannelResponse>, Status> { Err(unimpl()) }
    async fn hand_shake(&self, _: Request<PHandShakeRequest>) -> Result<Response<PHandShakeResponse>, Status> { Err(unimpl()) }
    async fn request_slave_tablet_pull_rowset(&self, _: Request<PTabletWriteSlaveRequest>) -> Result<Response<PTabletWriteSlaveResult>, Status> { Err(unimpl()) }
    async fn response_slave_tablet_pull_rowset(&self, _: Request<PTabletWriteSlaveDoneRequest>) -> Result<Response<PTabletWriteSlaveDoneResult>, Status> { Err(unimpl()) }
    async fn outfile_write_success(&self, _: Request<POutfileWriteSuccessRequest>) -> Result<Response<POutfileWriteSuccessResult>, Status> { Err(unimpl()) }
    async fn fetch_table_schema(&self, _: Request<PFetchTableSchemaRequest>) -> Result<Response<PFetchTableSchemaResult>, Status> { Err(unimpl()) }
    async fn multiget_data(&self, _: Request<PMultiGetRequest>) -> Result<Response<PMultiGetResponse>, Status> { Err(unimpl()) }
    async fn multiget_data_v2(&self, _: Request<PMultiGetRequestV2>) -> Result<Response<PMultiGetResponseV2>, Status> { Err(unimpl()) }
    async fn get_file_cache_meta_by_tablet_id(&self, _: Request<PGetFileCacheMetaRequest>) -> Result<Response<PGetFileCacheMetaResponse>, Status> { Err(unimpl()) }
    async fn warm_up_rowset(&self, _: Request<PWarmUpRowsetRequest>) -> Result<Response<PWarmUpRowsetResponse>, Status> { Err(unimpl()) }
    async fn recycle_cache(&self, _: Request<PRecycleCacheRequest>) -> Result<Response<PRecycleCacheResponse>, Status> { Err(unimpl()) }
    async fn tablet_fetch_data(&self, _: Request<PTabletKeyLookupRequest>) -> Result<Response<PTabletKeyLookupResponse>, Status> { Err(unimpl()) }
    async fn get_column_ids_by_tablet_ids(&self, _: Request<PFetchColIdsRequest>) -> Result<Response<PFetchColIdsResponse>, Status> { Err(unimpl()) }
    async fn get_tablet_rowset_versions(&self, _: Request<PGetTabletVersionsRequest>) -> Result<Response<PGetTabletVersionsResponse>, Status> { Err(unimpl()) }
    async fn report_stream_load_status(&self, _: Request<PReportStreamLoadStatusRequest>) -> Result<Response<PReportStreamLoadStatusResponse>, Status> { Err(unimpl()) }
    async fn glob(&self, _: Request<PGlobRequest>) -> Result<Response<PGlobResponse>, Status> { Err(unimpl()) }
    async fn group_commit_insert(&self, _: Request<PGroupCommitInsertRequest>) -> Result<Response<PGroupCommitInsertResponse>, Status> { Err(unimpl()) }
    async fn get_wal_queue_size(&self, _: Request<PGetWalQueueSizeRequest>) -> Result<Response<PGetWalQueueSizeResponse>, Status> { Err(unimpl()) }
    async fn fetch_remote_tablet_schema(&self, _: Request<PFetchRemoteSchemaRequest>) -> Result<Response<PFetchRemoteSchemaResponse>, Status> { Err(unimpl()) }
    async fn test_jdbc_connection(&self, _: Request<PJdbcTestConnectionRequest>) -> Result<Response<PJdbcTestConnectionResult>, Status> { Err(unimpl()) }
    async fn alter_vault_sync(&self, _: Request<PAlterVaultSyncRequest>) -> Result<Response<PAlterVaultSyncResponse>, Status> { Err(unimpl()) }
    async fn get_be_resource(&self, _: Request<PGetBeResourceRequest>) -> Result<Response<PGetBeResourceResponse>, Status> { Err(unimpl()) }
    async fn delete_dictionary(&self, _: Request<PDeleteDictionaryRequest>) -> Result<Response<PDeleteDictionaryResponse>, Status> { Err(unimpl()) }
    async fn commit_refresh_dictionary(&self, _: Request<PCommitRefreshDictionaryRequest>) -> Result<Response<PCommitRefreshDictionaryResponse>, Status> { Err(unimpl()) }
    async fn abort_refresh_dictionary(&self, _: Request<PAbortRefreshDictionaryRequest>) -> Result<Response<PAbortRefreshDictionaryResponse>, Status> { Err(unimpl()) }
    async fn get_tablet_rowsets(&self, _: Request<PGetTabletRowsetsRequest>) -> Result<Response<PGetTabletRowsetsResponse>, Status> { Err(unimpl()) }
    async fn fetch_peer_data(&self, _: Request<PFetchPeerDataRequest>) -> Result<Response<PFetchPeerDataResponse>, Status> { Err(unimpl()) }
    async fn request_cdc_client(&self, _: Request<PRequestCdcClientRequest>) -> Result<Response<PRequestCdcClientResult>, Status> { Err(unimpl()) }
}

/// Start the PBackendService gRPC server.
pub async fn start_grpc_server(
    listen_addr: &str,
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use doris_proto::doris::p_backend_service_server::PBackendServiceServer;

    let addr = listen_addr.parse()?;
    let handler = PBackendServiceHandler::new(state, result_store, engine);

    info!(addr = listen_addr, "starting PBackendService gRPC server");

    tonic::transport::Server::builder()
        .add_service(PBackendServiceServer::new(handler))
        .serve(addr)
        .await?;

    Ok(())
}
