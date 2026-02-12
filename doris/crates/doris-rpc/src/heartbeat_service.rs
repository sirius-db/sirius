//! HeartbeatService Thrift handler.
//!
//! Responds to FE heartbeat probes with TBackendInfo (ports, version, memory).

use std::sync::Arc;

use tracing::info;

use doris_thrift::heartbeat_service::{
    HeartbeatServiceSyncHandler, HeartbeatServiceSyncProcessor, TBackendInfo, THeartbeatResult,
    TMasterInfo,
};
use doris_thrift::status;

/// Shared state for the BE that the heartbeat handler reports.
#[derive(Debug, Clone)]
pub struct BeState {
    pub be_port: i32,
    pub http_port: i32,
    pub brpc_port: i32,
    pub arrow_flight_port: i32,
    pub version: String,
    pub start_time_ms: i64,
}

/// Handler implementation for HeartbeatService.
pub struct HeartbeatHandler {
    state: Arc<BeState>,
}

impl HeartbeatHandler {
    pub fn new(state: Arc<BeState>) -> Self {
        Self { state }
    }

    /// Create a processor wrapping this handler for use with TServer.
    pub fn into_processor(self) -> HeartbeatServiceSyncProcessor<Self> {
        HeartbeatServiceSyncProcessor::new(self)
    }
}

impl HeartbeatServiceSyncHandler for HeartbeatHandler {
    fn handle_heartbeat(&self, master_info: TMasterInfo) -> thrift::Result<THeartbeatResult> {
        info!(
            master_addr = ?master_info.network_address,
            "received heartbeat from FE"
        );

        let backend_info = TBackendInfo::new(
            self.state.be_port,            // be_port (required)
            self.state.http_port,          // http_port (required)
            Some(self.state.be_port),      // be_rpc_port
            Some(self.state.brpc_port),    // brpc_port
            Some(self.state.version.clone()), // version
            Some(self.state.start_time_ms),  // be_start_time
            Some("computation".to_string()), // be_node_role
            Some(false),                   // is_shutdown
            Some(self.state.arrow_flight_port), // arrow_flight_sql_port
            None::<i64>,                   // be_mem (let FE discover)
            None::<i64>,                   // fragment_executing_count
            None::<i64>,                   // fragment_last_active_time
        );

        let result = THeartbeatResult::new(
            status::TStatus::new(status::TStatusCode::OK, None::<Vec<String>>),
            backend_info,
        );

        Ok(result)
    }
}

/// Start the HeartbeatService Thrift server (blocking).
///
/// This should be called from a dedicated thread since TServer::listen blocks.
pub fn start_heartbeat_server(
    listen_addr: &str,
    state: Arc<BeState>,
) -> thrift::Result<()> {
    use thrift::protocol::{TBinaryInputProtocolFactory, TBinaryOutputProtocolFactory};
    use thrift::transport::{TBufferedReadTransportFactory, TBufferedWriteTransportFactory};
    use thrift::server::TServer;

    let handler = HeartbeatHandler::new(state);
    let processor = handler.into_processor();

    let i_tr_fact = TBufferedReadTransportFactory::new();
    let i_pr_fact = TBinaryInputProtocolFactory::new();
    let o_tr_fact = TBufferedWriteTransportFactory::new();
    let o_pr_fact = TBinaryOutputProtocolFactory::new();

    let mut server = TServer::new(i_tr_fact, i_pr_fact, o_tr_fact, o_pr_fact, processor, 4);

    info!(addr = listen_addr, "starting HeartbeatService Thrift server");
    server.listen(listen_addr)
}
