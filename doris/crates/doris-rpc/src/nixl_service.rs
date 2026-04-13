//! NIXL metadata exchange gRPC service (separate from PBackendService).
//!
//! Provides GPU-direct exchange coordination via gRPC methods:
//! - exchange_metadata: sender offers buffers, receiver allocates and returns addresses
//! - transfer_complete: sender notifies receiver that nixl transfer is done

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
use tracing::{info, instrument, warn};

use crate::exchange_buffer::ExchangeBuffer;
use crate::nixl_exchange::{NixlExchange, NixlRegisteredBuffer};

/// NIXL metadata exchange service handler.
pub struct NixlMetadataServiceHandler {
    nixl_agent: Option<Arc<NixlExchange>>,
    exchange_buffer: ExchangeBuffer,
    /// Engine for direct GPU table registration (packed cudf transfer path).
    engine: Option<Arc<std::sync::Mutex<sirius_ffi::SiriusEngine>>>,
    /// Pending GPU buffers awaiting transfer_complete. RAII cleanup on removal:
    /// - Registered: deregisters from nixl + frees GPU memory
    /// - Staged: releases leases (staging buffer reclaims space when all drop)
    /// Pending GPU buffers awaiting transfer_complete. Multiple entries per key
    /// because multiple exchange_metadata calls for the same (query_id, node_id)
    /// each allocate separate RECV staging regions.
    pending_buffers: Mutex<HashMap<(i64, i64, i32), Vec<PendingDstBuffers>>>,
}

/// Holds destination GPU buffers until transfer_complete.
enum PendingDstBuffers {
    /// cuMemAlloc + registered individually — RAII deregisters + frees.
    Registered(Vec<NixlRegisteredBuffer>),
    /// Staging buffer sub-allocations — leases release space on drop.
    Staged(Vec<crate::gpu_staging_buffer::StagingLease>),
}

impl NixlMetadataServiceHandler {
    pub fn new(nixl_agent: Option<Arc<NixlExchange>>, exchange_buffer: ExchangeBuffer) -> Self {
        Self {
            nixl_agent,
            exchange_buffer,
            engine: None,
            pending_buffers: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_engine(
        mut self,
        engine: Option<Arc<std::sync::Mutex<sirius_ffi::SiriusEngine>>>,
    ) -> Self {
        self.engine = engine;
        self
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
                dst_null_masks: vec![],
                dst_offsets: vec![],
            }));
        };

        // Step 1+2: Allocate destination GPU buffers for nixl transfer.
        //
        // Try the staging buffer first (pre-registered, avoids per-transfer
        // cuMemAlloc+register overhead). Fall back to allocate_and_register
        // if the staging buffer overflows or is not available.
        let sizes: Vec<_> = req
            .src_buffers
            .iter()
            .map(|b| (b.len, b.device_id))
            .collect();

        // Collect sub-buffer sizes (null_mask and offsets per column).
        let null_mask_sizes: Vec<_> = req
            .src_null_masks
            .iter()
            .map(|b| (b.len, b.device_id))
            .collect();
        let offsets_sizes: Vec<_> = req
            .src_offsets
            .iter()
            .map(|b| (b.len, b.device_id))
            .collect();

        // Flatten all non-zero sizes for allocation.
        let mut all_sizes: Vec<(u64, u64)> = Vec::new();
        // Track which indices in the flat allocation list correspond to what.
        // For each column: data_idx, null_mask_idx (or -1), offsets_idx (or -1).
        let mut buffer_map: Vec<(usize, Option<usize>, Option<usize>)> = Vec::new();

        for (i, &(len, dev)) in sizes.iter().enumerate() {
            let data_idx = all_sizes.len();
            all_sizes.push((len, dev));

            let nm_idx = if let Some(&(nm_len, nm_dev)) = null_mask_sizes.get(i) {
                if nm_len > 0 {
                    let idx = all_sizes.len();
                    all_sizes.push((nm_len, nm_dev));
                    Some(idx)
                } else {
                    None
                }
            } else {
                None
            };

            let off_idx = if let Some(&(off_len, off_dev)) = offsets_sizes.get(i) {
                if off_len > 0 {
                    let idx = all_sizes.len();
                    all_sizes.push((off_len, off_dev));
                    Some(idx)
                } else {
                    None
                }
            } else {
                None
            };

            buffer_map.push((data_idx, nm_idx, off_idx));
        }

        // Try staging buffer first, then fall back to cuMemAlloc+register.
        enum DstBuffers {
            Staged(Vec<crate::gpu_staging_buffer::StagingLease>),
            Registered(Vec<NixlRegisteredBuffer>),
        }

        let dst = match agent.staging() {
            Some(staging) => match staging.try_allocate(&all_sizes) {
                Ok(leases) => {
                    info!(
                        num = leases.len(),
                        staging_used = staging.stats().used,
                        "receiver: allocated dst buffers from staging buffer"
                    );
                    DstBuffers::Staged(leases)
                }
                Err(e) => {
                    info!(error = %e, "receiver: staging overflow, falling back to cuMemAlloc");
                    match agent.allocate_and_register_gpu_buffers(&all_sizes) {
                        Ok(r) => DstBuffers::Registered(r),
                        Err(e) => {
                            warn!(error = %e, "failed to allocate/register dst GPU buffers");
                            return Ok(Response::new(PExchangeNixlMetadataResponse {
                                dst_buffers: vec![],
                                remote_agent_name: String::new(),
                                status_code: 1,
                                error_msgs: vec![format!("allocate_and_register: {e}")],
                                nixl_metadata: vec![],
                                dst_null_masks: vec![],
                                dst_offsets: vec![],
                            }));
                        }
                    }
                }
            },
            None => match agent.allocate_and_register_gpu_buffers(&all_sizes) {
                Ok(r) => DstBuffers::Registered(r),
                Err(e) => {
                    warn!(error = %e, "failed to allocate/register dst GPU buffers");
                    return Ok(Response::new(PExchangeNixlMetadataResponse {
                        dst_buffers: vec![],
                        remote_agent_name: String::new(),
                        status_code: 1,
                        error_msgs: vec![format!("allocate_and_register: {e}")],
                        nixl_metadata: vec![],
                        dst_null_masks: vec![],
                        dst_offsets: vec![],
                    }));
                }
            },
        };

        // Helper to get (addr, len, device_id) for a buffer at index.
        let get_buf = |idx: usize| -> (usize, usize, u64) {
            match &dst {
                DstBuffers::Staged(leases) => (
                    leases[idx].addr(),
                    leases[idx].len(),
                    leases[idx].device_id(),
                ),
                DstBuffers::Registered(regs) => {
                    (regs[idx].addr(), regs[idx].len(), regs[idx].device_id())
                }
            }
        };
        let total_allocated = match &dst {
            DstBuffers::Staged(l) => l.len(),
            DstBuffers::Registered(r) => r.len(),
        };

        // Build response descriptors from the flat allocation.
        let mut dst_buffers: Vec<PGpuBufferDesc> = Vec::new();
        let mut dst_null_masks: Vec<PGpuBufferDesc> = Vec::new();
        let mut dst_offsets: Vec<PGpuBufferDesc> = Vec::new();
        let zero_desc = PGpuBufferDesc {
            addr: 0,
            len: 0,
            device_id: 0,
        };

        for &(data_idx, nm_idx, off_idx) in &buffer_map {
            let (addr, len, dev) = get_buf(data_idx);
            dst_buffers.push(PGpuBufferDesc {
                addr: addr as u64,
                len: len as u64,
                device_id: dev,
            });
            dst_null_masks.push(match nm_idx {
                Some(idx) => {
                    let (addr, len, dev) = get_buf(idx);
                    PGpuBufferDesc {
                        addr: addr as u64,
                        len: len as u64,
                        device_id: dev,
                    }
                }
                None => zero_desc.clone(),
            });
            dst_offsets.push(match off_idx {
                Some(idx) => {
                    let (addr, len, dev) = get_buf(idx);
                    PGpuBufferDesc {
                        addr: addr as u64,
                        len: len as u64,
                        device_id: dev,
                    }
                }
                None => zero_desc.clone(),
            });
        }

        let used_staging = matches!(&dst, DstBuffers::Staged(_));
        info!(
            num_data = dst_buffers.len(),
            num_null_masks = dst_null_masks.iter().filter(|b| b.addr != 0).count(),
            num_offsets = dst_offsets.iter().filter(|b| b.addr != 0).count(),
            total_allocated,
            used_staging,
            "allocated destination GPU buffers (data + sub-buffers)"
        );

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
        let pending = match dst {
            DstBuffers::Staged(leases) => PendingDstBuffers::Staged(leases),
            DstBuffers::Registered(regs) => PendingDstBuffers::Registered(regs),
        };
        self.pending_buffers
            .lock()
            .unwrap()
            .entry(pending_key)
            .or_insert_with(Vec::new)
            .push(pending);

        // Step 3: Load sender's metadata (force-load since sender registered new buffers).
        let remote_name = match agent.force_load_remote_metadata(&peer, &req.nixl_metadata) {
            Ok(name) => name,
            Err(e) => {
                // Pop the last entry we just pushed (don't remove all).
                if let Some(vec) = self.pending_buffers.lock().unwrap().get_mut(&pending_key) {
                    vec.pop();
                }
                warn!(error = %e, "failed to load remote nixl metadata");
                return Ok(Response::new(PExchangeNixlMetadataResponse {
                    dst_buffers: vec![],
                    remote_agent_name: String::new(),
                    status_code: 1,
                    error_msgs: vec![format!("load_remote_metadata: {e}")],
                    nixl_metadata: vec![],
                    dst_null_masks: vec![],
                    dst_offsets: vec![],
                }));
            }
        };

        // Step 4: Get receiver metadata.
        // When staging buffer is used, memory layout is stable → use cached metadata.
        // Otherwise, get fresh metadata (includes newly registered dst buffers).
        let receiver_metadata = match if used_staging {
            agent.get_metadata()
        } else {
            agent.get_fresh_metadata()
        } {
            Ok(md) => md,
            Err(e) => {
                if let Some(vec) = self.pending_buffers.lock().unwrap().get_mut(&pending_key) {
                    vec.pop();
                }
                warn!(error = %e, "failed to get fresh receiver metadata");
                return Ok(Response::new(PExchangeNixlMetadataResponse {
                    dst_buffers: vec![],
                    remote_agent_name: String::new(),
                    status_code: 1,
                    error_msgs: vec![format!("get_fresh_metadata: {e}")],
                    nixl_metadata: vec![],
                    dst_null_masks: vec![],
                    dst_offsets: vec![],
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
            dst_null_masks,
            dst_offsets,
        }))
    }

    /// Handle transfer_complete: sender has finished nixl GPU-direct transfer.
    ///
    /// Tries to reconstruct Arrow data from the GPU buffers via D2H copy,
    /// avoiding the redundant Arrow IPC bytes sent via gRPC. Falls back to
    /// the IPC bytes if GPU reconstruction fails (e.g., unsupported types).
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
            has_sub_buffers = !req.dst_null_masks.is_empty(),
            ipc_len = req.arrow_ipc_data.len(),
            "transfer_complete: nixl transfer done"
        );

        // Parse query_id from LE bytes.
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

        // Fast path: packed cudf transfer — store packed info in ExchangeBuffer.
        // The exchange async task (which holds the engine lock) will call
        // cudf::unpack + gpu_register_table to register the table on GPU.
        if !req.packed_cudf_metadata.is_empty() {
            let dst_buf = req
                .dst_buffers
                .first()
                .ok_or_else(|| Status::internal("packed transfer: no dst buffer"))?;

            info!(
                gpu_addr = format_args!("0x{:x}", dst_buf.addr),
                gpu_size = dst_buf.len,
                metadata_len = req.packed_cudf_metadata.len(),
                "transfer_complete: packed cudf buffer received, storing for GPU-direct registration"
            );

            // FULL buffer checksum: D2H the entire buffer and compute checksum.
            if let Ok(buf) =
                crate::cuda_driver::gpu_to_host(dst_buf.addr as usize, dst_buf.len as usize)
            {
                let checksum: u64 = buf.iter().map(|&b| b as u64).sum();
                info!(
                    gpu_addr = format_args!("0x{:x}", dst_buf.addr),
                    gpu_size = dst_buf.len,
                    checksum,
                    "transfer_complete: RECEIVER buffer checksum"
                );
            }

            // Move ONE staging lease from pending_buffers into the PackedGpuExchange.
            // Pop from the Vec (not remove!) to keep other entries' leases alive.
            let staging_lease = self
                .pending_buffers
                .lock()
                .unwrap()
                .get_mut(&pending_key)
                .and_then(|vec| vec.pop())
                .and_then(|pb| match pb {
                    PendingDstBuffers::Staged(mut leases) => leases.pop(),
                    _ => None,
                });

            let key = crate::exchange_buffer::ExchangeKey {
                query_id: (query_id_hi, query_id_lo),
                node_id: req.node_id,
            };
            self.exchange_buffer.store_packed_gpu(
                key.clone(),
                crate::exchange_buffer::PackedGpuExchange {
                    gpu_addr: dst_buf.addr as usize,
                    gpu_size: dst_buf.len as usize,
                    cudf_metadata: req.packed_cudf_metadata.clone(),
                    _staging_lease: staging_lease,
                },
            );
            // Signal EOS (no PBlock data — the exchange task will use packed GPU path).
            self.exchange_buffer
                .add_block(&key, req.sender_id, None, true);

            return Ok(Response::new(PNixlTransferCompleteResponse {
                status_code: 0,
                error_msgs: vec![],
            }));
        }

        // All nixl transfers must use packed cudf format. The legacy per-column
        // D2H path (reconstruct_ipc_from_gpu_buffers) has been removed.
        warn!(
            "transfer_complete: received non-packed transfer (packed_cudf_metadata is empty). \
             All nixl transfers must use cudf::pack format."
        );

        // Free pending destination GPU buffers for this entry.
        if let Some(vec) = self.pending_buffers.lock().unwrap().get_mut(&pending_key) {
            vec.pop();
        }

        Ok(Response::new(PNixlTransferCompleteResponse {
            status_code: 1,
            error_msgs: vec![
                "non-packed nixl transfer not supported — all transfers must use cudf::pack format"
                    .into(),
            ],
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
