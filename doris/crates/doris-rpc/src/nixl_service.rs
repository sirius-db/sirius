//! NIXL metadata exchange gRPC service (separate from PBackendService).
//!
//! Provides GPU-direct exchange coordination via gRPC method:
//! - exchange_metadata: sender offers buffers, receiver allocates and returns addresses

use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
#[cfg(feature = "nixl")]
use tracing::{info, instrument, warn};

#[cfg(feature = "nixl")]
use crate::nixl_exchange::NixlExchange;
use sirius_ffi::SiriusEngine;

/// NIXL metadata exchange service handler.
#[allow(dead_code)] // engine only read with nixl feature
pub struct NixlMetadataServiceHandler {
    #[cfg(feature = "nixl")]
    nixl_agent: Option<Arc<NixlExchange>>,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
}

impl NixlMetadataServiceHandler {
    pub fn new(
        #[cfg(feature = "nixl")]
        nixl_agent: Option<Arc<NixlExchange>>,
        engine: Option<Arc<Mutex<SiriusEngine>>>,
    ) -> Self {
        Self {
            #[cfg(feature = "nixl")]
            nixl_agent,
            engine,
        }
    }
}

// Implement the tonic-generated trait
#[tonic::async_trait]
impl doris_proto::nixl::NixlMetadataService for NixlMetadataServiceHandler {
    #[cfg(feature = "nixl")]
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
        let dst_buffers: Vec<PGpuBufferDesc> = if let Some(engine) = &self.engine {
            let sizes: Vec<(usize, u64)> = req
                .src_buffers
                .iter()
                .map(|b| (b.len as usize, b.device_id))
                .collect();
            match engine.lock().unwrap().allocate_gpu_buffers(&sizes) {
                Ok(allocs) => allocs
                    .into_iter()
                    .map(|(addr, len, device_id)| PGpuBufferDesc {
                        addr: addr as u64,
                        len: len as u64,
                        device_id,
                    })
                    .collect(),
                Err(e) => {
                    warn!(error = %e, "GPU buffer allocation failed, mirroring src addresses");
                    req.src_buffers
                        .iter()
                        .map(|src| PGpuBufferDesc {
                            addr: src.addr,
                            len: src.len,
                            device_id: src.device_id,
                        })
                        .collect()
                }
            }
        } else {
            // No engine — mirror src buffers (test/stub mode).
            req.src_buffers
                .iter()
                .map(|src| PGpuBufferDesc {
                    addr: src.addr,
                    len: src.len,
                    device_id: src.device_id,
                })
                .collect()
        };

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
    async fn exchange_metadata(
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
    async fn test_service_creation_no_engine() {
        #[cfg(feature = "nixl")]
        let service = NixlMetadataServiceHandler::new(None, None);
        #[cfg(not(feature = "nixl"))]
        let service = NixlMetadataServiceHandler::new(None);

        let _ = service;
    }

    #[tokio::test]
    async fn test_service_creation_with_engine() {
        let engine = sirius_ffi::SiriusEngine::new().ok().map(|e| Arc::new(Mutex::new(e)));

        #[cfg(feature = "nixl")]
        let service = NixlMetadataServiceHandler::new(None, engine);
        #[cfg(not(feature = "nixl"))]
        let service = NixlMetadataServiceHandler::new(engine);

        let _ = service;
    }

    #[tokio::test]
    #[cfg(not(feature = "nixl"))]
    async fn test_exchange_metadata_without_nixl() {
        use doris_proto::nixl::NixlMetadataService;

        let service = NixlMetadataServiceHandler::new(None);
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

    #[tokio::test]
    #[cfg(not(feature = "nixl"))]
    async fn test_exchange_metadata_with_engine_still_unimplemented() {
        use doris_proto::nixl::NixlMetadataService;

        let engine = sirius_ffi::SiriusEngine::new().ok().map(|e| Arc::new(Mutex::new(e)));
        let service = NixlMetadataServiceHandler::new(engine);
        let request = Request::new(doris_proto::nixl::PExchangeNixlMetadataRequest {
            nixl_metadata: vec![],
            src_buffers: vec![],
            columns: vec![],
            num_rows: 0,
            query_id_hi: vec![],
            query_id_lo: vec![],
            node_id: 0,
        });

        // Even with engine, non-nixl build returns unimplemented.
        let result = service.exchange_metadata(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }
}
