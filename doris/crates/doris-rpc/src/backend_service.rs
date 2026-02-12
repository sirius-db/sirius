//! BackendService Thrift handler (stub).
//!
//! Most BackendService methods are for tablet/storage ops that a GPU-only BE
//! doesn't support. They return NOT_IMPLEMENTED_ERROR.
//! Fragment execution goes through PBackendService (gRPC), not here.

use std::collections::BTreeMap;
use std::sync::Arc;

use tracing::warn;

use doris_thrift::agent_service;
use doris_thrift::backend_service::{self, BackendServiceSyncHandler, BackendServiceSyncProcessor};
use doris_thrift::doris_external_service;
use doris_thrift::status;

use super::heartbeat_service::BeState;

pub struct BackendHandler {
    _state: Arc<BeState>,
}

impl BackendHandler {
    pub fn new(state: Arc<BeState>) -> Self {
        Self { _state: state }
    }

    pub fn into_processor(self) -> BackendServiceSyncProcessor<Self> {
        BackendServiceSyncProcessor::new(self)
    }
}

fn not_implemented(method: &str) -> status::TStatus {
    warn!(method, "BackendService method not implemented for GPU BE");
    status::TStatus::new(
        status::TStatusCode::NOT_IMPLEMENTED_ERROR,
        Some(vec![format!(
            "{method} not supported on Sirius GPU backend"
        )]),
    )
}

fn not_impl_agent(method: &str) -> thrift::Result<agent_service::TAgentResult> {
    Ok(agent_service::TAgentResult::new(
        not_implemented(method),
        None::<String>,
        None::<bool>,
        None::<i32>,
    ))
}

impl BackendServiceSyncHandler for BackendHandler {
    fn handle_submit_tasks(
        &self,
        _tasks: Vec<agent_service::TAgentTaskRequest>,
    ) -> thrift::Result<agent_service::TAgentResult> {
        not_impl_agent("submit_tasks")
    }

    fn handle_make_snapshot(
        &self,
        _snapshot_request: agent_service::TSnapshotRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        not_impl_agent("make_snapshot")
    }

    fn handle_release_snapshot(
        &self,
        _snapshot_path: String,
    ) -> thrift::Result<agent_service::TAgentResult> {
        not_impl_agent("release_snapshot")
    }

    fn handle_publish_cluster_state(
        &self,
        _request: agent_service::TAgentPublishRequest,
    ) -> thrift::Result<agent_service::TAgentResult> {
        not_impl_agent("publish_cluster_state")
    }

    fn handle_get_tablet_stat(&self) -> thrift::Result<backend_service::TTabletStatResult> {
        Ok(backend_service::TTabletStatResult::new(
            BTreeMap::new(),
            None::<Vec<backend_service::TTabletStat>>,
        ))
    }

    fn handle_get_trash_used_capacity(&self) -> thrift::Result<i64> {
        Ok(0)
    }

    fn handle_get_disk_trash_used_capacity(
        &self,
    ) -> thrift::Result<Vec<backend_service::TDiskTrashInfo>> {
        Ok(vec![])
    }

    fn handle_submit_routine_load_task(
        &self,
        _tasks: Vec<backend_service::TRoutineLoadTask>,
    ) -> thrift::Result<status::TStatus> {
        Ok(not_implemented("submit_routine_load_task"))
    }

    fn handle_open_scanner(
        &self,
        _params: doris_external_service::TScanOpenParams,
    ) -> thrift::Result<doris_external_service::TScanOpenResult> {
        Ok(doris_external_service::TScanOpenResult::new(
            not_implemented("open_scanner"),
            None::<String>,
            None::<Vec<doris_external_service::TScanColumnDesc>>,
        ))
    }

    fn handle_get_next(
        &self,
        _params: doris_external_service::TScanNextBatchParams,
    ) -> thrift::Result<doris_external_service::TScanBatchResult> {
        Ok(doris_external_service::TScanBatchResult::new(
            not_implemented("get_next"),
            None::<bool>,
            None::<Vec<u8>>,
        ))
    }

    fn handle_close_scanner(
        &self,
        _params: doris_external_service::TScanCloseParams,
    ) -> thrift::Result<doris_external_service::TScanCloseResult> {
        Ok(doris_external_service::TScanCloseResult::new(
            not_implemented("close_scanner"),
        ))
    }

    fn handle_get_stream_load_record(
        &self,
        _last_stream_record_time: i64,
    ) -> thrift::Result<backend_service::TStreamLoadRecordResult> {
        Ok(backend_service::TStreamLoadRecordResult::new(
            BTreeMap::new(),
        ))
    }

    fn handle_check_storage_format(
        &self,
    ) -> thrift::Result<backend_service::TCheckStorageFormatResult> {
        Ok(backend_service::TCheckStorageFormatResult::new(
            None::<Vec<i64>>,
            None::<Vec<i64>>,
        ))
    }

    fn handle_warm_up_cache_async(
        &self,
        _request: backend_service::TWarmUpCacheAsyncRequest,
    ) -> thrift::Result<backend_service::TWarmUpCacheAsyncResponse> {
        Ok(backend_service::TWarmUpCacheAsyncResponse::new(
            not_implemented("warm_up_cache_async"),
        ))
    }

    fn handle_check_warm_up_cache_async(
        &self,
        _request: backend_service::TCheckWarmUpCacheAsyncRequest,
    ) -> thrift::Result<backend_service::TCheckWarmUpCacheAsyncResponse> {
        Ok(backend_service::TCheckWarmUpCacheAsyncResponse::new(
            not_implemented("check_warm_up_cache_async"),
            None::<BTreeMap<i64, bool>>,
        ))
    }

    fn handle_sync_load_for_tablets(
        &self,
        _request: backend_service::TSyncLoadForTabletsRequest,
    ) -> thrift::Result<backend_service::TSyncLoadForTabletsResponse> {
        Ok(backend_service::TSyncLoadForTabletsResponse::new())
    }

    fn handle_get_top_n_hot_partitions(
        &self,
        _request: backend_service::TGetTopNHotPartitionsRequest,
    ) -> thrift::Result<backend_service::TGetTopNHotPartitionsResponse> {
        Ok(backend_service::TGetTopNHotPartitionsResponse::new(
            0,
            None::<Vec<backend_service::THotTableMessage>>,
        ))
    }

    fn handle_warm_up_tablets(
        &self,
        _request: backend_service::TWarmUpTabletsRequest,
    ) -> thrift::Result<backend_service::TWarmUpTabletsResponse> {
        Ok(backend_service::TWarmUpTabletsResponse::new(
            not_implemented("warm_up_tablets"),
            None::<i64>,
            None::<i64>,
            None::<i64>,
            None::<i64>,
        ))
    }

    fn handle_ingest_binlog(
        &self,
        _request: backend_service::TIngestBinlogRequest,
    ) -> thrift::Result<backend_service::TIngestBinlogResult> {
        Ok(backend_service::TIngestBinlogResult::new(
            None::<status::TStatus>,
            None::<bool>,
        ))
    }

    fn handle_query_ingest_binlog(
        &self,
        _request: backend_service::TQueryIngestBinlogRequest,
    ) -> thrift::Result<backend_service::TQueryIngestBinlogResult> {
        Ok(backend_service::TQueryIngestBinlogResult::new(
            None::<backend_service::TIngestBinlogStatus>,
            None::<String>,
        ))
    }

    fn handle_publish_topic_info(
        &self,
        _request: backend_service::TPublishTopicRequest,
    ) -> thrift::Result<backend_service::TPublishTopicResult> {
        Ok(backend_service::TPublishTopicResult::new(
            not_implemented("publish_topic_info"),
        ))
    }

    fn handle_get_realtime_exec_status(
        &self,
        _request: backend_service::TGetRealtimeExecStatusRequest,
    ) -> thrift::Result<backend_service::TGetRealtimeExecStatusResponse> {
        Ok(backend_service::TGetRealtimeExecStatusResponse::new(
            None::<status::TStatus>,
            None::<doris_thrift::frontend_service::TReportExecStatusParams>,
            None::<doris_thrift::frontend_service::TQueryStatistics>,
        ))
    }

    fn handle_get_dictionary_status(
        &self,
        _dictionary_ids: Vec<i64>,
    ) -> thrift::Result<backend_service::TDictionaryStatusList> {
        Ok(backend_service::TDictionaryStatusList::new(None::<
            Vec<backend_service::TDictionaryStatus>,
        >))
    }

    fn handle_test_storage_connectivity(
        &self,
        _request: backend_service::TTestStorageConnectivityRequest,
    ) -> thrift::Result<backend_service::TTestStorageConnectivityResponse> {
        Ok(backend_service::TTestStorageConnectivityResponse::new(
            None::<status::TStatus>,
        ))
    }
}

/// Start the BackendService Thrift server (blocking).
pub fn start_backend_server(listen_addr: &str, state: Arc<BeState>) -> thrift::Result<()> {
    use thrift::protocol::{TBinaryInputProtocolFactory, TBinaryOutputProtocolFactory};
    use thrift::server::TServer;
    use thrift::transport::{TBufferedReadTransportFactory, TBufferedWriteTransportFactory};
    use tracing::info;

    let handler = BackendHandler::new(state);
    let processor = handler.into_processor();

    let i_tr_fact = TBufferedReadTransportFactory::new();
    let i_pr_fact = TBinaryInputProtocolFactory::new();
    let o_tr_fact = TBufferedWriteTransportFactory::new();
    let o_pr_fact = TBinaryOutputProtocolFactory::new();

    let mut server = TServer::new(i_tr_fact, i_pr_fact, o_tr_fact, o_pr_fact, processor, 4);

    info!(addr = listen_addr, "starting BackendService Thrift server");
    server.listen(listen_addr)
}
