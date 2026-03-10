//! Integration layer between nixl GPU-direct exchange and query execution.
//!
//! Handles:
//! - Detecting GPU-resident results (from sirius-ffi engine)
//! - Extracting GPU buffer pointers for nixl transfer
//! - Coordinating metadata exchange and transfer
//! - Fallback to bRPC when GPU-direct unavailable

use std::sync::Arc;

use crate::nixl_exchange::{GpuBufferDesc, NixlExchange};
use crate::exchange_sender::ExchangeDest;
use crate::exchange_buffer::{ExchangeBuffer, ExchangeKey};
use sirius_ffi::GpuColumnBuffers;

/// Result of attempting to extract GPU buffer information from execution result.
#[derive(Debug)]
pub enum ExecutionLocation {
    /// Result is in CPU memory (Arrow IPC bytes).
    Cpu(Vec<u8>),
    /// Result is in GPU memory (buffer descriptors + schema).
    /// Also carries IPC bytes as fallback for non-exchange paths.
    Gpu {
        /// GPU buffer descriptors (addr, len, device_id).
        buffers: Vec<GpuBufferDesc>,
        /// Column names and type IDs.
        column_info: Vec<(String, i32)>,
        /// Per-column extended buffer info (null masks, string offsets).
        column_buffers: Vec<GpuColumnBuffers>,
        /// Number of rows.
        num_rows: u32,
        /// Arrow IPC schema bytes (for receiver reconstruction).
        schema_ipc: Vec<u8>,
        /// Arrow IPC bytes (fallback for store/fetch_data path).
        ipc_bytes: Vec<u8>,
        /// cuMemAlloc-allocated buffer addresses to free after nixl transfer.
        /// Empty when `rmm_pool_registered` is true (no copy was needed).
        cuda_alloc_addrs: Vec<usize>,
        /// Whether the RMM pool is registered with nixl, meaning the buffers
        /// can be used directly without copying to cuMemAlloc.
        rmm_pool_registered: bool,
    },
}

impl ExecutionLocation {
    /// Extract IPC bytes, consuming self.
    pub fn into_ipc_bytes(self) -> Vec<u8> {
        match self {
            Self::Cpu(bytes) => bytes,
            Self::Gpu { ipc_bytes, .. } => ipc_bytes,
        }
    }

    /// Try to register the RMM pool with nixl so buffers can be used directly.
    ///
    /// Uses `engine.get_pool_info()` to query the processing pool directly from
    /// the GPUBufferManager, avoiding `cuMemGetAddressRange` which returns wrong
    /// sizes for RMM sub-allocations.
    ///
    /// Returns `true` if the pool was registered and copies can be skipped.
    /// On failure, caller should fall back to `copy_gpu_buffers_to_cuda_alloc`.
    pub fn try_register_rmm_pool(&mut self, nixl_agent: &NixlExchange, engine: &sirius_ffi::SiriusEngine) -> bool {
        if let Self::Gpu { buffers, rmm_pool_registered, .. } = self {
            if buffers.is_empty() {
                return false;
            }
            // Query pool info from the Sirius engine (GPUBufferManager).
            let (base, size, device_id) = match engine.get_pool_info() {
                Ok(Some(info)) => info,
                Ok(None) => {
                    tracing::warn!("sirius_get_pool_info returned no data");
                    return false;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "sirius_get_pool_info failed");
                    return false;
                }
            };
            if nixl_agent.ensure_rmm_pool_registered(base, size, device_id as u64) {
                let buf_tuples: Vec<_> = buffers.iter().map(|b| (b.addr, b.len, b.device_id)).collect();
                if nixl_agent.buffers_in_rmm_pool(&buf_tuples) {
                    tracing::info!(
                        num_buffers = buffers.len(),
                        pool_base = format_args!("0x{base:x}"),
                        pool_size_mb = size / (1024 * 1024),
                        "RMM pool registered, skipping cuMemAlloc copy"
                    );
                    *rmm_pool_registered = true;
                    return true;
                }
                tracing::warn!("some buffers outside registered RMM pool, falling back to copy");
            }
        }
        false
    }

    /// Copy GPU buffers to fresh cuMemAlloc-allocated memory.
    ///
    /// MUST be called on the same thread where the GPU execution happened
    /// (inside spawn_blocking), while the engine's CUDA context is still active.
    /// This ensures the RMM sub-allocation pointers are accessible.
    ///
    /// After this call, the buffer addresses point to cuMemAlloc allocations
    /// which UCX can reliably use for GPU-direct transfers.
    pub fn copy_gpu_buffers_to_cuda_alloc(&mut self) {
        if let Self::Gpu { buffers, cuda_alloc_addrs, .. } = self {
            use crate::cuda_driver::{cuda_alloc, cuda_free, cuda_memcpy_dtod_no_ctx as cuda_memcpy_dtod};
            let mut new_buffers = Vec::with_capacity(buffers.len());
            let mut alloc_addrs = Vec::new();
            for b in buffers.iter() {
                match cuda_alloc(b.len) {
                    Ok(dst) => {
                        if let Err(e) = cuda_memcpy_dtod(dst, b.addr, b.len) {
                            tracing::warn!(
                                error = %e,
                                src = format_args!("0x{:x}", b.addr),
                                len = b.len,
                                "cuMemcpyDtoD failed, keeping original buffer"
                            );
                            let _ = cuda_free(dst);
                            new_buffers.push(b.clone());
                        } else {
                            tracing::info!(
                                src = format_args!("0x{:x}", b.addr),
                                dst = format_args!("0x{:x}", dst),
                                len = b.len,
                                "copied GPU buffer to cuMemAlloc"
                            );
                            alloc_addrs.push(dst);
                            new_buffers.push(GpuBufferDesc {
                                addr: dst,
                                len: b.len,
                                device_id: b.device_id,
                            });
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            len = b.len,
                            "cuda_alloc failed, keeping original buffer"
                        );
                        new_buffers.push(b.clone());
                    }
                }
            }
            *buffers = new_buffers;
            *cuda_alloc_addrs = alloc_addrs;
        }
    }

    /// Free any cuMemAlloc-allocated buffer copies.
    pub fn free_cuda_alloc_buffers(&mut self) {
        if let Self::Gpu { cuda_alloc_addrs, .. } = self {
            for &addr in cuda_alloc_addrs.iter() {
                if let Err(e) = crate::cuda_driver::cuda_free(addr) {
                    tracing::warn!(error = %e, "failed to free cuMemAlloc buffer");
                }
            }
            cuda_alloc_addrs.clear();
        }
    }
}

/// Detect whether execution result is in GPU or CPU memory.
///
/// Returns `ExecutionLocation::Gpu` if the engine executed on GPU and result
/// buffers are still GPU-resident, otherwise `ExecutionLocation::Cpu` with
/// the standard Arrow IPC bytes.
pub fn detect_execution_location(
    ipc_bytes: Vec<u8>,
    _engine: &sirius_ffi::SiriusEngine,
) -> ExecutionLocation {
    {
        // Try to extract GPU buffer pointers from the engine.
        // If the last execution was GPU-accelerated and buffers are still in VRAM,
        // the engine can provide the raw GPU addresses.
        match _engine.get_last_gpu_result_buffers() {
            Ok(Some(gpu_info)) => {
                for (i, &(addr, len, dev)) in gpu_info.buffer_addrs.iter().enumerate() {
                    tracing::info!(
                        buf_idx = i,
                        addr = format_args!("0x{:x}", addr),
                        len,
                        device_id = dev,
                        "detect_execution_location: GPU buffer"
                    );
                }
                tracing::info!(
                    num_buffers = gpu_info.buffer_addrs.len(),
                    num_rows = gpu_info.num_rows,
                    "detect_execution_location: GPU buffers found"
                );
                return ExecutionLocation::Gpu {
                    buffers: gpu_info
                        .buffer_addrs
                        .iter()
                        .map(|&(addr, len, device_id)| GpuBufferDesc {
                            addr,
                            len,
                            device_id,
                        })
                        .collect(),
                    column_info: gpu_info.column_info,
                    column_buffers: gpu_info.column_buffers,
                    num_rows: gpu_info.num_rows,
                    schema_ipc: gpu_info.schema_ipc,
                    ipc_bytes,
                    cuda_alloc_addrs: vec![],
                    rmm_pool_registered: false,
                };
            }
            Ok(None) => {
                tracing::info!("detect_execution_location: no GPU buffers (CPU execution)");
            }
            Err(e) => {
                tracing::warn!(error = %e, "detect_execution_location: get_last_gpu_result_buffers failed");
            }
        }
    }

    ExecutionLocation::Cpu(ipc_bytes)
}

/// Send exchange result using nixl GPU-direct if available, otherwise bRPC.
///
/// When `nixl_only` is true, bRPC fallback is disabled — errors surface instead
/// of being silently handled. Useful for debugging nixl exchange issues.
///
/// Self-transfer (destination is `local_brpc_addr`) uses a local fast-path:
/// IPC → PBlock → ExchangeBuffer, bypassing both nixl and bRPC entirely.
pub async fn send_exchange_with_nixl(
    nixl_agent: Option<&Arc<NixlExchange>>,
    location: ExecutionLocation,
    destinations: &[ExchangeDest],
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
    nixl_only: bool,
    local_brpc_addr: &str,
    exchange_buffer: &ExchangeBuffer,
) -> Result<(), String> {
    // Split destinations into local (self) and remote.
    let (local_dests, remote_dests): (Vec<ExchangeDest>, Vec<ExchangeDest>) = destinations
        .iter()
        .cloned()
        .partition(|d| d.brpc_addr == local_brpc_addr);

    // Extract IPC bytes (needed for both local and remote paths).
    let ipc_bytes = match &location {
        ExecutionLocation::Cpu(bytes) => bytes.clone(),
        ExecutionLocation::Gpu { ipc_bytes, .. } => ipc_bytes.clone(),
    };

    // Handle local (self-transfer) destinations via ExchangeBuffer.
    for dest in &local_dests {
        tracing::info!(
            dest = %dest.brpc_addr,
            sender_id,
            "self-transfer: feeding data directly into local ExchangeBuffer"
        );
        let (pblock, _num_rows) = crate::arrow_to_pblock::arrow_ipc_to_pblock(&ipc_bytes)
            .map_err(|e| format!("self-transfer arrow_ipc_to_pblock: {e}"))?;
        let key = ExchangeKey { query_id, node_id };
        exchange_buffer.add_block(&key, sender_id, Some(pblock), false);
        exchange_buffer.add_block(&key, sender_id, None, true);
        tracing::info!(
            dest = %dest.brpc_addr,
            sender_id,
            "self-transfer: complete"
        );
    }

    // If no remote destinations, we're done.
    if remote_dests.is_empty() {
        return Ok(());
    }

    match location {
        ExecutionLocation::Cpu(_) => {
            if nixl_only {
                tracing::error!("nixl-only mode: result is CPU-resident, cannot use nixl GPU-direct transfer");
                return Err("nixl-only: no GPU buffers for GPU-direct exchange (execution fell back to CPU?)".to_string());
            }
            crate::exchange_sender::send_exchange_result(
                &ipc_bytes, &remote_dests, query_id, node_id, sender_id,
            )
            .await
        }
        ExecutionLocation::Gpu {
            buffers,
            column_info,
            column_buffers,
            num_rows,
            schema_ipc: _,
            ipc_bytes: _,
            cuda_alloc_addrs,
            rmm_pool_registered,
        } => {
            let Some(agent) = nixl_agent else {
                if nixl_only {
                    return Err("nixl-only: GPU result but no nixl agent available".to_string());
                }
                tracing::warn!("GPU result but no nixl agent, falling back to bRPC");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes, &remote_dests, query_id, node_id, sender_id,
                ).await;
            };

            // Fast-path: skip nixl if a previous registration already detected
            // that UCX treats GPU memory as host memory (would SIGSEGV).
            if !agent.gpu_transfer_enabled() {
                if nixl_only {
                    return Err("nixl-only: UCX lacks CUDA support for GPU memory".to_string());
                }
                tracing::info!("UCX lacks CUDA support for GPU memory, using bRPC for exchange");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes, &remote_dests, query_id, node_id, sender_id,
                ).await;
            }

            // Free cuMemAlloc copies once we're done (success or error).
            let free_allocs = |addrs: &[usize]| {
                for &addr in addrs {
                    if let Err(e) = crate::cuda_driver::cuda_free(addr) {
                        tracing::warn!(error = %e, addr = format_args!("0x{addr:x}"), "failed to free cuMemAlloc buffer");
                    }
                }
            };

            // Try nixl GPU-direct for each remote destination.
            for dest in &remote_dests {
                if let Err(e) = send_nixl_to_peer(
                    agent, &buffers, &column_info, &column_buffers, num_rows,
                    &ipc_bytes, dest, query_id, node_id, sender_id,
                    rmm_pool_registered,
                ).await {
                    free_allocs(&cuda_alloc_addrs);
                    if nixl_only {
                        return Err(format!("nixl-only: nixl transfer to {} failed: {}", dest.brpc_addr, e));
                    }
                    tracing::warn!(
                        error = %e,
                        dest = %dest.brpc_addr,
                        "nixl transfer failed, falling back to bRPC"
                    );
                    crate::exchange_sender::send_exchange_result(
                        &ipc_bytes, &remote_dests, query_id, node_id, sender_id,
                    ).await?;
                    return Ok(());
                }
            }

            free_allocs(&cuda_alloc_addrs);
            Ok(())
        }
    }
}

/// Send GPU data to a single peer via nixl.
///
/// Full flow: register buffers → exchange metadata → load peer metadata →
/// transfer → notify receiver of completion.
///
/// Transfers all sub-buffers (data, null_mask, offsets) for each column.
/// The RMM pool registration covers all sub-buffers when `rmm_pool_registered`.
async fn send_nixl_to_peer(
    agent: &Arc<NixlExchange>,
    src_buffers: &[GpuBufferDesc],
    column_info: &[(String, i32)],
    column_buffers: &[GpuColumnBuffers],
    num_rows: u32,
    ipc_bytes: &[u8],
    dest: &ExchangeDest,
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
    rmm_pool_registered: bool,
) -> Result<(), String> {
    use doris_proto::nixl::{
        NixlMetadataServiceClient, PColumnInfo, PExchangeNixlMetadataRequest,
        PGpuBufferDesc, PNixlTransferCompleteRequest,
    };
    use tracing::info;

    let grpc_addr = format!("http://{}", dest.brpc_addr);
    let device_id = src_buffers.first().map(|b| b.device_id).unwrap_or(0);

    // Build sub-buffer proto messages (aligned with src_buffers, one per column).
    let src_null_masks: Vec<PGpuBufferDesc> = column_buffers
        .iter()
        .map(|cb| PGpuBufferDesc {
            addr: cb.null_mask_addr as u64,
            len: cb.null_mask_len as u64,
            device_id,
        })
        .collect();
    let src_offsets: Vec<PGpuBufferDesc> = column_buffers
        .iter()
        .map(|cb| PGpuBufferDesc {
            addr: cb.offsets_addr as u64,
            len: cb.offsets_len as u64,
            device_id,
        })
        .collect();
    let null_counts: Vec<i32> = column_buffers.iter().map(|cb| cb.null_count).collect();

    // Collect ALL non-zero buffers (data + null_mask + offsets) for registration and transfer.
    // Order: for each column: data, then null_mask (if present), then offsets (if present).
    let mut all_src_bufs: Vec<(usize, usize, u64)> = Vec::new();
    for (i, b) in src_buffers.iter().enumerate() {
        all_src_bufs.push((b.addr, b.len, b.device_id));
        if let Some(cb) = column_buffers.get(i) {
            if cb.null_mask_addr != 0 && cb.null_mask_len > 0 {
                all_src_bufs.push((cb.null_mask_addr, cb.null_mask_len, b.device_id));
            }
            if cb.offsets_addr != 0 && cb.offsets_len > 0 {
                all_src_bufs.push((cb.offsets_addr, cb.offsets_len, b.device_id));
            }
        }
    }

    // Step 1: Register sender's GPU result buffers with nixl agent.
    // When the RMM pool is registered, the pool registration already covers
    // all sub-allocations — no per-buffer registration needed.
    let _src_registrations = if rmm_pool_registered {
        info!("RMM pool registered, skipping per-buffer nixl registration");
        vec![]
    } else {
        // Buffers have been copied to cuMemAlloc allocations (by
        // copy_gpu_buffers_to_cuda_alloc in spawn_blocking) so UCX can handle them.
        // NOTE: When not pool-registered, sub-buffers are RMM sub-allocations
        // that may not be separately cuMemAlloc'd. Only register data buffers.
        let buf_tuples: Vec<_> = src_buffers.iter().map(|b| (b.addr, b.len, b.device_id)).collect();
        agent.register_gpu_buffers(&buf_tuples)?
    };

    // Step 2: Get fresh metadata (includes newly registered buffers).
    let fresh_md = agent.get_fresh_metadata()?;

    // Step 3: Call exchange_metadata RPC on receiver.
    let request = PExchangeNixlMetadataRequest {
        nixl_metadata: fresh_md,
        src_buffers: src_buffers
            .iter()
            .map(|b| PGpuBufferDesc {
                addr: b.addr as u64,
                len: b.len as u64,
                device_id: b.device_id,
            })
            .collect(),
        columns: column_info
            .iter()
            .enumerate()
            .map(|(i, (name, type_id))| {
                let scale = column_buffers.get(i).map(|cb| cb.scale).unwrap_or(0);
                // Infer precision from decimal width.
                let precision = match *type_id {
                    25 => 9,   // DECIMAL32
                    26 => 18,  // DECIMAL64
                    27 => 38,  // DECIMAL128
                    _ => 0,
                };
                PColumnInfo {
                    name: name.clone(),
                    type_id: *type_id,
                    precision,
                    scale,
                }
            })
            .collect(),
        num_rows,
        query_id_hi: query_id.0.to_le_bytes().to_vec(),
        query_id_lo: query_id.1.to_le_bytes().to_vec(),
        node_id,
        src_null_masks,
        src_offsets,
        null_counts: null_counts.clone(),
    };

    let channel = tonic::transport::Endpoint::from_shared(grpc_addr.clone())
        .map_err(|e| format!("endpoint for {grpc_addr}: {e}"))?
        .connect()
        .await
        .map_err(|e| format!("connect to {grpc_addr}: {e}"))?;
    let mut client = NixlMetadataServiceClient::new(channel)
        .max_decoding_message_size(256 * 1024 * 1024)
        .max_encoding_message_size(256 * 1024 * 1024);

    let response = client
        .exchange_metadata(request)
        .await
        .map_err(|e| format!("exchange_metadata RPC: {e}"))?
        .into_inner();

    if response.status_code != 0 {
        return Err(format!(
            "nixl metadata exchange failed: {}",
            response.error_msgs.join("; ")
        ));
    }

    if response.dst_buffers.len() != src_buffers.len() {
        return Err(format!(
            "data buffer count mismatch: src={}, dst={}",
            src_buffers.len(),
            response.dst_buffers.len()
        ));
    }

    // Step 4: Load receiver's metadata (includes their registered dst buffers).
    let remote_name = agent.force_load_remote_metadata(
        &dest.brpc_addr,
        &response.nixl_metadata,
    )?;

    // Step 5: Build flattened (src, dst) pairs for all buffers including sub-buffers.
    // Order per column: data, null_mask (if present), offsets (if present).
    let mut all_src_ptrs: Vec<(usize, usize)> = Vec::new();
    let mut all_dst_ptrs: Vec<(usize, usize)> = Vec::new();

    for (i, src_b) in src_buffers.iter().enumerate() {
        // Data buffer.
        let dst_b = &response.dst_buffers[i];
        all_src_ptrs.push((src_b.addr, src_b.len));
        all_dst_ptrs.push((dst_b.addr as usize, dst_b.len as usize));

        // Null mask (only if both sender has data and receiver allocated space).
        if let (Some(cb), Some(dst_nm)) = (
            column_buffers.get(i),
            response.dst_null_masks.get(i),
        ) {
            if cb.null_mask_addr != 0 && cb.null_mask_len > 0 && dst_nm.addr != 0 && dst_nm.len > 0 {
                all_src_ptrs.push((cb.null_mask_addr, cb.null_mask_len));
                all_dst_ptrs.push((dst_nm.addr as usize, dst_nm.len as usize));
            }
        }

        // String offsets (only if both sender has data and receiver allocated space).
        if let (Some(cb), Some(dst_off)) = (
            column_buffers.get(i),
            response.dst_offsets.get(i),
        ) {
            if cb.offsets_addr != 0 && cb.offsets_len > 0 && dst_off.addr != 0 && dst_off.len > 0 {
                all_src_ptrs.push((cb.offsets_addr, cb.offsets_len));
                all_dst_ptrs.push((dst_off.addr as usize, dst_off.len as usize));
            }
        }
    }

    let num_total_transfers = all_src_ptrs.len();
    info!(
        num_data_buffers = src_buffers.len(),
        num_total_transfers,
        "nixl transfer: flattened buffer pairs"
    );

    // Execute transfer on blocking thread (nixl uses polling).
    {
        let agent = agent.clone();
        let remote = remote_name.clone();

        tokio::task::spawn_blocking(move || {
            let src_descs = agent.create_gpu_descs(&all_src_ptrs, device_id)?;
            let dst_descs = agent.create_gpu_descs(&all_dst_ptrs, device_id)?;
            agent.transfer_gpu_to_gpu(&src_descs, &dst_descs, &remote)
        })
        .await
        .map_err(|e| format!("transfer spawn_blocking panicked: {e}"))??;
    }

    info!(
        dest = %dest.brpc_addr,
        num_buffers = num_total_transfers,
        "nixl GPU-direct transfer complete"
    );

    // Step 6: Notify receiver that transfer is complete.
    // Still include Arrow IPC bytes as fallback (receiver can choose to use
    // GPU data reconstruction or IPC-based PBlock construction).
    let complete_req = PNixlTransferCompleteRequest {
        query_id_hi: query_id.0.to_le_bytes().to_vec(),
        query_id_lo: query_id.1.to_le_bytes().to_vec(),
        node_id,
        dst_buffers: response.dst_buffers,
        columns: column_info
            .iter()
            .enumerate()
            .map(|(i, (name, type_id))| {
                let scale = column_buffers.get(i).map(|cb| cb.scale).unwrap_or(0);
                let precision = match *type_id {
                    25 => 9, 26 => 18, 27 => 38, _ => 0,
                };
                PColumnInfo {
                    name: name.clone(),
                    type_id: *type_id,
                    precision,
                    scale,
                }
            })
            .collect(),
        num_rows,
        sender_id,
        arrow_ipc_data: ipc_bytes.to_vec(),
        dst_null_masks: response.dst_null_masks,
        dst_offsets: response.dst_offsets,
        null_counts,
    };

    let channel2 = tonic::transport::Endpoint::from_shared(grpc_addr.clone())
        .map_err(|e| format!("endpoint for transfer_complete: {e}"))?
        .connect()
        .await
        .map_err(|e| format!("connect for transfer_complete: {e}"))?;
    let mut client2 = NixlMetadataServiceClient::new(channel2)
        .max_decoding_message_size(256 * 1024 * 1024)
        .max_encoding_message_size(256 * 1024 * 1024);

    let tc_response = client2
        .transfer_complete(complete_req)
        .await
        .map_err(|e| format!("transfer_complete RPC: {e}"))?
        .into_inner();

    if tc_response.status_code != 0 {
        return Err(format!(
            "transfer_complete failed: {}",
            tc_response.error_msgs.join("; ")
        ));
    }

    info!(dest = %dest.brpc_addr, "nixl transfer_complete acknowledged by receiver");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_location_cpu() {
        // Mock IPC bytes.
        let ipc = vec![1, 2, 3, 4];

        // Skip test if engine not available (duckdb-bundled feature not enabled).
        let Ok(engine) = sirius_ffi::SiriusEngine::new() else {
            return;
        };

        let location = detect_execution_location(ipc.clone(), &engine);
        match location {
            ExecutionLocation::Cpu(bytes) => {
                assert_eq!(bytes, ipc);
            }
            ExecutionLocation::Gpu { .. } => {
                // Could happen if engine has GPU result
            }
        }
    }

    #[test]
    fn test_execution_location_gpu() {
        // This test requires GPU execution, skip in CI.
        // Demonstrates API usage pattern:

        // 1. Execute query on GPU
        // 2. Call detect_execution_location
        // 3. Match on Gpu variant to extract buffer descriptors
    }

    #[test]
    fn test_exchange_dest_structure() {
        let dest = ExchangeDest {
            brpc_addr: "10.0.0.1:8060".to_string(),
            finst_id: (100, 200),
        };

        assert_eq!(dest.brpc_addr, "10.0.0.1:8060");
        assert_eq!(dest.finst_id, (100, 200));
    }

    #[test]
    fn test_execution_location_cpu_roundtrip() {
        let data = vec![0x41, 0x52, 0x52, 0x4f, 0x57]; // "ARROW" bytes
        let location = ExecutionLocation::Cpu(data.clone());
        match location {
            ExecutionLocation::Cpu(bytes) => assert_eq!(bytes, data),
            ExecutionLocation::Gpu { .. } => panic!("expected Cpu variant"),
        }
    }

    #[tokio::test]
    async fn test_send_exchange_cpu_location_no_dests() {
        // Verify that Cpu location with no destinations returns Ok (nothing to send).
        let ipc = vec![0xAA, 0xBB, 0xCC];
        let location = ExecutionLocation::Cpu(ipc);
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();

        let result = send_exchange_with_nixl(
            None,
            location,
            &[], // no destinations
            (1, 2),
            0,
            0,
            false,
            "localhost:8060",
            &exchange_buffer,
        )
        .await;
        // No destinations → no work → Ok
        assert!(result.is_ok());
    }
}
