//! NIXL metadata exchange gRPC service (separate from PBackendService).
//!
//! Provides GPU-direct exchange coordination via gRPC method:
//! - exchange_metadata: sender offers buffers, receiver allocates and returns addresses

#[cfg(feature = "nixl")]
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{info, instrument, warn};

#[cfg(feature = "nixl")]
use crate::nixl_exchange::NixlExchange;

/// NIXL metadata exchange service handler.
pub struct NixlMetadataService {
    #[cfg(feature = "nixl")]
    nixl_agent: Option<Arc<NixlExchange>>,
}

impl NixlMetadataService {
    pub fn new(
        #[cfg(feature = "nixl")]
        nixl_agent: Option<Arc<NixlExchange>>,
    ) -> Self {
        Self {
            #[cfg(feature = "nixl")]
            nixl_agent,
        }
    }

    /// Exchange NIXL metadata for GPU-direct transfer.
    ///
    /// Sender offers GPU buffer descriptors, receiver allocates destination
    /// buffers and returns their addresses for RDMA transfer.
    #[cfg(feature = "nixl")]
    #[instrument(skip_all, fields(peer, num_buffers))]
    pub async fn exchange_metadata(
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
            }));
        };

        // Load sender's nixl metadata (cached after first call).
        let remote_name = match agent.load_remote_metadata(&peer, &req.nixl_metadata) {
            Ok(name) => name,
            Err(e) => {
                warn!(error = %e, "failed to load remote nixl metadata");
                return Ok(Response::new(PExchangeNixlMetadataResponse {
                    dst_buffers: vec![],
                    remote_agent_name: String::new(),
                    status_code: 1,
                    error_msgs: vec![format!("load_remote_metadata: {e}")],
                }));
            }
        };

        // Allocate destination GPU buffers matching source sizes.
        // For now, we'll use a simplified approach: return src_buffers as dst_buffers
        // (mock allocation). In production, this would call:
        //   engine.allocate_gpu_buffers(src_buffers) -> dst_buffers

        let dst_buffers: Vec<PGpuBufferDesc> = req
            .src_buffers
            .iter()
            .map(|src| PGpuBufferDesc {
                addr: src.addr, // Mock: same address (in reality, allocate new)
                len: src.len,
                device_id: src.device_id,
            })
            .collect();

        info!(
            peer = %peer,
            remote_agent = %remote_name,
            num_dst_buffers = dst_buffers.len(),
            "nixl metadata exchange complete"
        );

        Ok(Response::new(PExchangeNixlMetadataResponse {
            dst_buffers,
            remote_agent_name: remote_name,
            status_code: 0,
            error_msgs: vec![],
        }))
    }

    #[cfg(not(feature = "nixl"))]
    pub async fn exchange_metadata(
        &self,
        _: Request<doris_proto::nixl::PExchangeNixlMetadataRequest>,
    ) -> Result<Response<doris_proto::nixl::PExchangeNixlMetadataResponse>, Status> {
        Err(Status::unimplemented(
            "nixl feature not enabled on this BE",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        #[cfg(feature = "nixl")]
        let service = NixlMetadataService::new(None);
        #[cfg(not(feature = "nixl"))]
        let service = NixlMetadataService::new();

        // Service should be created successfully
        let _ = service;
    }

    #[tokio::test]
    #[cfg(not(feature = "nixl"))]
    async fn test_exchange_metadata_without_nixl() {
        let service = NixlMetadataService::new();
        let request = Request::new(doris_proto::nixl::PExchangeNixlMetadataRequest {
            nixl_metadata: vec![],
            src_buffers: vec![],
            columns: vec![],
            num_rows: 0,
            query_id_hi: vec![],
            query_id_lo: vec![],
            node_id: 0,
        });

        let result = service.exchange_metadata(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }
}
