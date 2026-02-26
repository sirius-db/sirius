//! NIXL metadata exchange gRPC service (separate from PBackendService).
//!
//! Provides GPU-direct exchange coordination via gRPC methods:
//! - exchange_metadata: sender offers buffers, receiver allocates and returns addresses
//! - transfer_complete: sender notifies receiver that nixl transfer is done

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
use tracing::{info, instrument, warn};

use crate::nixl_exchange::{NixlExchange, NixlRegisteredBuffer};
use crate::exchange_buffer::ExchangeBuffer;

/// NIXL metadata exchange service handler.
pub struct NixlMetadataServiceHandler {
    nixl_agent: Option<Arc<NixlExchange>>,
    exchange_buffer: ExchangeBuffer,
    /// Pending GPU buffers awaiting transfer_complete. RAII cleanup on removal:
    /// deregisters from nixl + frees GPU memory in correct order.
    pending_buffers: Mutex<HashMap<(i64, i64, i32), Vec<NixlRegisteredBuffer>>>,
}

impl NixlMetadataServiceHandler {
    pub fn new(
        nixl_agent: Option<Arc<NixlExchange>>,
        exchange_buffer: ExchangeBuffer,
    ) -> Self {
        Self {
            nixl_agent,
            exchange_buffer,
            pending_buffers: Mutex::new(HashMap::new()),
        }
    }
}

// Implement the tonic-generated trait
#[tonic::async_trait]
impl doris_proto::nixl::NixlMetadataService for NixlMetadataServiceHandler {
    #[instrument(skip_all, fields(peer, num_buffers))]
    async fn exchange_metadata(
        &self,
        request: Request<doris_proto::nixl::PExchangeNixlMetadataRequest>,
    ) -> Result<Response<doris_proto::nixl::PExchangeNixlMetadataResponse>, Status> {
        use doris_proto::nixl::{PExchangeNixlMetadataResponse, PGpuBufferDesc};

        let peer = request
            .remote_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let req = request.into_inner();

        tracing::Span::current().record("peer", &peer);
        tracing::Span::current().record("num_buffers", req.src_buffers.len());

        info!(
            peer = %peer,
            num_buffers = req.src_buffers.len(),
            num_rows = req.num_rows,
            "received nixl metadata exchange request"
        );

        // Check if nixl agent is available.
        let Some(agent) = &self.nixl_agent else {
            warn!("nixl metadata exchange requested but nixl agent not available");
            return Ok(Response::new(PExchangeNixlMetadataResponse {
                dst_buffers: vec![],
                remote_agent_name: String::new(),
                status_code: 1,
                error_msgs: vec!["nixl not available on this BE".to_string()],
                nixl_metadata: vec![],
            }));
        };

        // Step 1+2: Allocate and register destination GPU buffers in one operation.
        // Uses cuMemAlloc directly (not the RMM pool) because the RMM processing
        // pool may be at capacity after GPU query execution. RAII handles cleanup.
        let sizes: Vec<_> = req.src_buffers.iter().map(|b| (b.len, b.device_id)).collect();
        let registered = match agent.allocate_and_register_gpu_buffers(&sizes) {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "failed to allocate/register dst GPU buffers");
                return Ok(Response::new(PExchangeNixlMetadataResponse {
                    dst_buffers: vec![],
                    remote_agent_name: String::new(),
                    status_code: 1,
                    error_msgs: vec![format!("allocate_and_register: {e}")],
                    nixl_metadata: vec![],
                }));
            }
        };

        let dst_buffers: Vec<PGpuBufferDesc> = registered
            .iter()
            .map(|b| PGpuBufferDesc {
                addr: b.addr() as u64,
                len: b.len() as u64,
                device_id: b.device_id(),
            })
            .collect();

        // Store pending buffers for RAII cleanup in transfer_complete.
        let query_id_hi = i64::from_le_bytes(
            req.query_id_hi
                .get(..8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]),
        );
        let query_id_lo = i64::from_le_bytes(
            req.query_id_lo
                .get(..8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]),
        );
        let pending_key = (query_id_hi, query_id_lo, req.node_id);
        self.pending_buffers
            .lock()
            .unwrap()
            .insert(pending_key, registered);

        // Step 3: Load sender's metadata (force-load since sender registered new buffers).
        let remote_name = match agent.force_load_remote_metadata(&peer, &req.nixl_metadata) {
            Ok(name) => name,
            Err(e) => {
                self.pending_buffers.lock().unwrap().remove(&pending_key);
                warn!(error = %e, "failed to load remote nixl metadata");
                return Ok(Response::new(PExchangeNixlMetadataResponse {
                    dst_buffers: vec![],
                    remote_agent_name: String::new(),
                    status_code: 1,
                    error_msgs: vec![format!("load_remote_metadata: {e}")],
                    nixl_metadata: vec![],
                }));
            }
        };

        // Step 4: Get fresh receiver metadata (includes newly registered dst buffers).
        let receiver_metadata = match agent.get_fresh_metadata() {
            Ok(md) => md,
            Err(e) => {
                self.pending_buffers.lock().unwrap().remove(&pending_key);
                warn!(error = %e, "failed to get fresh receiver metadata");
                return Ok(Response::new(PExchangeNixlMetadataResponse {
                    dst_buffers: vec![],
                    remote_agent_name: String::new(),
                    status_code: 1,
                    error_msgs: vec![format!("get_fresh_metadata: {e}")],
                    nixl_metadata: vec![],
                }));
            }
        };

        info!(
            peer = %peer,
            remote_agent = %remote_name,
            num_dst_buffers = dst_buffers.len(),
            receiver_metadata_len = receiver_metadata.len(),
            "nixl metadata exchange complete (allocated + registered + metadata)"
        );

        Ok(Response::new(PExchangeNixlMetadataResponse {
            dst_buffers,
            remote_agent_name: remote_name,
            status_code: 0,
            error_msgs: vec![],
            nixl_metadata: receiver_metadata,
        }))
    }

    /// Handle transfer_complete: sender has finished nixl GPU-direct transfer.
    ///
    /// The sender includes Arrow IPC bytes alongside the GPU transfer. We use
    /// `arrow_ipc_to_pblock` (same path as bRPC) to construct a proper PBlock
    /// and feed it into the ExchangeBuffer. This avoids type ID mismatches
    /// between DuckDB LogicalTypeId and Doris PGenericType::TypeId.
    ///
    /// The GPU buffers (now in receiver VRAM) are freed after processing.
    /// In a future optimization, we'll register GPU buffers directly as DuckDB
    /// tables, skipping the PBlock round-trip entirely.
    #[instrument(skip_all, fields(num_rows, num_buffers, sender_id))]
    async fn transfer_complete(
        &self,
        request: Request<doris_proto::nixl::PNixlTransferCompleteRequest>,
    ) -> Result<Response<doris_proto::nixl::PNixlTransferCompleteResponse>, Status> {
        use doris_proto::nixl::PNixlTransferCompleteResponse;

        let req = request.into_inner();

        tracing::Span::current().record("num_rows", req.num_rows);
        tracing::Span::current().record("num_buffers", req.dst_buffers.len());
        tracing::Span::current().record("sender_id", req.sender_id);

        info!(
            num_rows = req.num_rows,
            num_buffers = req.dst_buffers.len(),
            num_columns = req.columns.len(),
            sender_id = req.sender_id,
            ipc_len = req.arrow_ipc_data.len(),
            "transfer_complete: nixl transfer done, building PBlock from IPC"
        );

        // Parse query_id from LE bytes.
        let query_id_hi = i64::from_le_bytes(
            req.query_id_hi.get(..8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]),
        );
        let query_id_lo = i64::from_le_bytes(
            req.query_id_lo.get(..8)
                .and_then(|s| s.try_into().ok())
                .unwrap_or([0u8; 8]),
        );

        // Free destination GPU buffers via RAII: deregister from nixl + cuda_free.
        let pending_key = (query_id_hi, query_id_lo, req.node_id);
        let removed = self.pending_buffers.lock().unwrap().remove(&pending_key);
        info!(
            freed = removed.as_ref().map_or(0, |v| v.len()),
            "transfer_complete: released pending GPU buffers (RAII)"
        );
        drop(removed);

        // Convert Arrow IPC to PBlock using the same path as bRPC exchange.
        let (pblock, _num_rows) = match crate::arrow_to_pblock::arrow_ipc_to_pblock(&req.arrow_ipc_data) {
            Ok(result) => result,
            Err(e) => {
                warn!(error = %e, "arrow_ipc_to_pblock failed in transfer_complete");
                return Ok(Response::new(PNixlTransferCompleteResponse {
                    status_code: 1,
                    error_msgs: vec![format!("arrow_ipc_to_pblock: {e}")],
                }));
            }
        };

        // Feed into ExchangeBuffer and signal EOS for this sender.
        let key = crate::exchange_buffer::ExchangeKey {
            query_id: (query_id_hi, query_id_lo),
            node_id: req.node_id,
        };

        self.exchange_buffer.add_block(&key, req.sender_id, Some(pblock), false);
        let all_done = self.exchange_buffer.add_block(&key, req.sender_id, None, true);

        info!(
            query_id = ?(query_id_hi, query_id_lo),
            node_id = req.node_id,
            sender_id = req.sender_id,
            all_done,
            "transfer_complete: fed PBlock into ExchangeBuffer"
        );

        Ok(Response::new(PNixlTransferCompleteResponse {
            status_code: 0,
            error_msgs: vec![],
        }))
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    fn exchange_buf() -> ExchangeBuffer {
        ExchangeBuffer::new()
    }

    #[tokio::test]
    async fn test_service_creation() {
        let service = NixlMetadataServiceHandler::new(None, exchange_buf());

        let _ = service;
    }
}
