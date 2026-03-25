//! Integration layer between nixl GPU-direct exchange and query execution.
//!
//! Handles:
//! - Detecting GPU-resident results (from sirius-ffi engine)
//! - Extracting GPU buffer pointers for nixl transfer
//! - Coordinating metadata exchange and transfer
//! - Fallback to bRPC when GPU-direct unavailable

use std::sync::Arc;

use crate::gpu_staging_buffer::StagingLease;
use crate::nixl_exchange::{GpuBufferDesc, NixlExchange};
use crate::exchange_sender::ExchangeDest;
use crate::exchange_buffer::{ExchangeBuffer, ExchangeKey, PackedGpuExchange};
use crate::hash_partitioner::{ExchangeInfo, PartitionStrategy};
use sirius_ffi::GpuColumnBuffers;

/// Result of attempting to extract GPU buffer information from execution result.
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
        /// Staging buffer leases. Kept alive until transfer completes.
        /// When present, buffer addresses point into the staging buffer.
        _staging_leases: Vec<StagingLease>,
        /// cudf::pack() metadata for the packed GPU buffer (if packed).
        /// When present, `buffers` contains a single descriptor pointing to
        /// the contiguous packed buffer in the nixl staging region.
        packed_metadata: Option<Vec<u8>>,
        /// Per-partition packed buffers from GPU hash_partition + chunked_pack.
        /// When present, each entry has (staging_offset, size, cudf_metadata, num_rows).
        packed_partitions: Vec<sirius_ffi::PackedPartition>,
    },
}

impl ExecutionLocation {
    /// Create a GPU location from a packed staging buffer.
    ///
    /// This is the primary constructor for GPU results — the packed buffer
    /// is a single contiguous allocation in the pre-registered nixl staging
    /// region, created by cudf::chunked_pack in the result collector.
    pub fn from_packed(addr: usize, size: usize, metadata: Vec<u8>, ipc_bytes: Vec<u8>) -> Self {
        Self::Gpu {
            buffers: vec![GpuBufferDesc { addr, len: size, device_id: 0 }],
            column_info: vec![],
            column_buffers: vec![],
            num_rows: 0,
            schema_ipc: vec![],
            ipc_bytes,
            _staging_leases: vec![],
            packed_metadata: Some(metadata),
            packed_partitions: vec![],
        }
    }

    /// Extract IPC bytes, consuming self.
    pub fn into_ipc_bytes(self) -> Vec<u8> {
        match self {
            Self::Cpu(bytes) => bytes,
            Self::Gpu { ipc_bytes, .. } => ipc_bytes,
        }
    }

    /// Store packed GPU buffer info from cudf::pack().
    ///
    /// The packed buffer is a single contiguous GPU allocation containing all
    /// column data, null masks, and offsets. It's allocated by the C++ CUDA
    /// runtime (same as RMM) so it stays valid after query cleanup.
    pub fn set_packed_gpu(&mut self, addr: usize, size: usize, metadata: Vec<u8>) {
        if let Self::Gpu { buffers, packed_metadata, column_buffers, .. } = self {
            // Replace per-column buffer descriptors with a single packed buffer.
            buffers.clear();
            buffers.push(GpuBufferDesc {
                addr,
                len: size,
                device_id: 0,
            });
            column_buffers.clear();
            *packed_metadata = Some(metadata.clone());
            tracing::info!(
                addr = format_args!("0x{addr:x}"),
                size,
                metadata_len = metadata.len(),
                "set packed GPU buffer for nixl transfer"
            );
        }
    }

    /// Store per-partition packed GPU buffers from GPU hash_partition.
    pub fn set_packed_partitions(&mut self, parts: Vec<sirius_ffi::PackedPartition>) {
        if let Self::Gpu { packed_partitions, .. } = self {
            *packed_partitions = parts;
        }
    }

    /// Apply staging results from C++ `sirius_stage_gpu_buffers()`.
    ///
    /// Updates GPU buffer addresses to point into the staging buffer at the
    /// offsets reported by the C++ side.
    pub fn apply_staging(&mut self, staging_base: usize, staged: &[sirius_ffi::StagedBuffer]) {
        if let Self::Gpu { buffers, column_buffers, .. } = self {
            for entry in staged {
                let addr = staging_base + entry.staged_offset;
                match entry.buf_type.as_str() {
                    "data" => {
                        if let Some(b) = buffers.get_mut(entry.buffer_idx) {
                            b.addr = addr;
                            b.len = entry.len;
                        }
                    }
                    "null_mask" => {
                        if let Some(cb) = column_buffers.get_mut(entry.buffer_idx) {
                            cb.null_mask_addr = addr;
                            cb.null_mask_len = entry.len;
                        }
                    }
                    "offsets" => {
                        if let Some(cb) = column_buffers.get_mut(entry.buffer_idx) {
                            cb.offsets_addr = addr;
                            cb.offsets_len = entry.len;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Try to stage all GPU buffers (data + sub-buffers) into the staging buffer.
    ///
    /// On success, updates buffer addresses to point into the staging buffer
    /// and returns the leases (must be held until transfer completes).
    /// On failure (overflow, copy error), returns `None` — caller should fall
    /// back to `copy_gpu_buffers_to_cuda_alloc`.
    pub fn try_stage_buffers(
        &mut self,
        staging: &crate::gpu_staging_buffer::GpuStagingBuffer,
    ) -> bool {
        if let Self::Gpu { buffers, column_buffers, _staging_leases, .. } = self {
            // Collect ALL buffers: data + null_mask + offsets (same order as transfer).
            let mut all_bufs: Vec<(usize, usize, u64)> = Vec::new();
            // Track which indices map back to data/null_mask/offsets per column.
            let mut data_indices: Vec<usize> = Vec::new();
            let mut null_mask_indices: Vec<Option<usize>> = Vec::new();
            let mut offsets_indices: Vec<Option<usize>> = Vec::new();

            for (i, b) in buffers.iter().enumerate() {
                data_indices.push(all_bufs.len());
                all_bufs.push((b.addr, b.len, b.device_id));

                if let Some(cb) = column_buffers.get(i) {
                    if cb.null_mask_addr != 0 && cb.null_mask_len > 0 {
                        null_mask_indices.push(Some(all_bufs.len()));
                        all_bufs.push((cb.null_mask_addr, cb.null_mask_len, b.device_id));
                    } else {
                        null_mask_indices.push(None);
                    }
                    if cb.offsets_addr != 0 && cb.offsets_len > 0 {
                        offsets_indices.push(Some(all_bufs.len()));
                        all_bufs.push((cb.offsets_addr, cb.offsets_len, b.device_id));
                    } else {
                        offsets_indices.push(None);
                    }
                } else {
                    null_mask_indices.push(None);
                    offsets_indices.push(None);
                }
            }

            let leases = match staging.try_stage(&all_bufs) {
                Ok(l) => l,
                Err(e) => {
                    tracing::info!(error = %e, "staging buffer: falling back to per-buffer cuMemAlloc");
                    return false;
                }
            };

            // Update addresses to point to staged locations.
            for (i, b) in buffers.iter_mut().enumerate() {
                let lease = &leases[data_indices[i]];
                b.addr = lease.addr();
            }
            for (i, cb) in column_buffers.iter_mut().enumerate() {
                if let Some(idx) = null_mask_indices.get(i).copied().flatten() {
                    cb.null_mask_addr = leases[idx].addr();
                }
                if let Some(idx) = offsets_indices.get(i).copied().flatten() {
                    cb.offsets_addr = leases[idx].addr();
                }
            }

            tracing::info!(
                num_staged = leases.len(),
                staging_used = staging.stats().used,
                staging_capacity = staging.stats().capacity,
                "staged GPU buffers into staging buffer"
            );

            *_staging_leases = leases;
            true
        } else {
            false
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
                    _staging_leases: vec![],
                    packed_metadata: None,
                    packed_partitions: vec![],
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
///
/// For `Hash` partition strategy, rows are hashed by partition columns and each
/// destination receives only its partition's rows. For `Broadcast`, all
/// destinations receive all rows (existing behavior).
pub async fn send_exchange_with_nixl(
    nixl_agent: Option<&Arc<NixlExchange>>,
    location: ExecutionLocation,
    exch_info: &ExchangeInfo,
    query_id: (i64, i64),
    sender_id: i32,
    nixl_only: bool,
    local_brpc_addr: &str,
    exchange_buffer: &ExchangeBuffer,
    desc_tbl_slots: Option<&[(i32, String)]>,
) -> Result<(), String> {
    let destinations = &exch_info.destinations;
    let node_id = exch_info.dest_node_id;

    // Extract IPC bytes (needed for both local and remote paths).
    let ipc_bytes = match &location {
        ExecutionLocation::Cpu(bytes) => bytes.clone(),
        ExecutionLocation::Gpu { ipc_bytes, .. } => ipc_bytes.clone(),
    };

    // For hash-partitioned exchange, split the data per destination first,
    // then route each partition's data to its destination (local or remote).
    if let PartitionStrategy::Hash { ref partition_exprs, num_destinations, use_crc32c } = exch_info.partition {
        return send_hash_partitioned(
            nixl_agent, &location, &ipc_bytes, destinations, query_id, node_id,
            sender_id, nixl_only, local_brpc_addr, exchange_buffer,
            partition_exprs, num_destinations, use_crc32c, desc_tbl_slots,
        ).await;
    }

    // Extract packed info from location (if available).
    let (packed_metadata, buffers) = match &location {
        ExecutionLocation::Gpu { packed_metadata, buffers, .. } => (packed_metadata.as_deref(), buffers.as_slice()),
        _ => (None, &[][..]),
    };

    // Broadcast / Random: send all rows to all destinations (existing behavior).
    // Split destinations into local (self) and remote.
    let (local_dests, remote_dests): (Vec<ExchangeDest>, Vec<ExchangeDest>) = destinations
        .iter()
        .cloned()
        .partition(|d| d.brpc_addr == local_brpc_addr);

    // Handle local (self-transfer) destinations via ExchangeBuffer.
    for dest in &local_dests {
        let key = ExchangeKey { query_id, node_id };

        // If packed GPU data is available, store it directly — no PBlock needed.
        if let Some(ref md) = packed_metadata {
            if let Some(buf) = buffers.first() {
                tracing::info!(
                    dest = %dest.brpc_addr,
                    gpu_addr = format_args!("0x{:x}", buf.addr),
                    gpu_size = buf.len,
                    "self-transfer: packed GPU path (zero copies)"
                );
                exchange_buffer.store_packed_gpu(
                    key.clone(),
                    PackedGpuExchange {
                        gpu_addr: buf.addr,
                        gpu_size: buf.len,
                        cudf_metadata: md.to_vec(),
                    },
                );
                exchange_buffer.add_block(&key, sender_id, None, true);
                tracing::info!(dest = %dest.brpc_addr, sender_id, "self-transfer: complete (GPU-direct)");
                continue;
            }
        }

        tracing::info!(
            dest = %dest.brpc_addr,
            sender_id,
            "self-transfer: feeding data directly into local ExchangeBuffer"
        );
        let (pblock, _num_rows) = crate::arrow_to_pblock::arrow_ipc_to_pblock(&ipc_bytes)
            .map_err(|e| format!("self-transfer arrow_ipc_to_pblock: {e}"))?;
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
            _staging_leases,
            packed_partitions: _,
            packed_metadata,
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

            // Packed buffers are always in the staging region.
            let staged = packed_metadata.is_some() || !_staging_leases.is_empty();

            // Try nixl GPU-direct for each remote destination.
            for dest in &remote_dests {
                if let Err(e) = send_nixl_to_peer(
                    agent, &buffers, &column_info, &column_buffers, num_rows,
                    &ipc_bytes, dest, query_id, node_id, sender_id,
                    staged, packed_metadata.as_deref(),
                ).await {
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

            Ok(())
        }
    }
}

/// Hash-partitioned exchange: split data by partition columns and route each
/// partition to its assigned destination.
#[allow(clippy::too_many_arguments)]
async fn send_hash_partitioned(
    _nixl_agent: Option<&Arc<NixlExchange>>,
    _location: &ExecutionLocation,
    ipc_bytes: &[u8],
    destinations: &[ExchangeDest],
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
    _nixl_only: bool,
    local_brpc_addr: &str,
    exchange_buffer: &ExchangeBuffer,
    partition_exprs: &[doris_thrift::exprs::TExpr],
    num_destinations: usize,
    use_crc32c: bool,
    desc_tbl_slots: Option<&[(i32, String)]>,
) -> Result<(), String> {
    use crate::hash_partitioner::{compute_dest_assignments, resolve_partition_columns, split_by_destination};

    // GPU hash-partitioned path: use per-partition packed buffers from staging.
    if let ExecutionLocation::Gpu { ref packed_partitions, ref buffers, ref packed_metadata, .. } = _location {
        if !packed_partitions.is_empty() {
            tracing::info!(
                num_partitions = packed_partitions.len(),
                num_dests = destinations.len(),
                "sending GPU hash-partitioned exchange data"
            );

            // Get the send staging base address from the first buffer.
            // Partitions are at staging_base + partition.staging_offset.
            let staging_base = buffers.first().map(|b| b.addr).unwrap_or(0);
            // Actually, partitions use the send staging buffer, not the packed buffer.
            // The send staging addr was set by set_staging_buffer. We need to know it.
            // For now, compute from the first partition's offset relative to known staging.
            // The packed buffer (from get_packed_gpu) addr is the send staging base.
            let send_staging_base = if let Some(ref _md) = packed_metadata {
                buffers.first().map(|b| b.addr).unwrap_or(0)
            } else {
                0
            };

            // Each partition maps to one destination (1:1).
            for (dest_idx, dest) in destinations.iter().enumerate() {
                let key = ExchangeKey { query_id, node_id };
                let partition = packed_partitions.get(dest_idx);

                let is_local = dest.brpc_addr == local_brpc_addr;

                match partition {
                    Some(p) if p.packed_size > 0 => {
                        let part_addr = send_staging_base + p.staging_offset;

                        if is_local {
                            // Self-transfer: store packed GPU data directly.
                            tracing::info!(
                                dest_idx,
                                dest = %dest.brpc_addr,
                                rows = p.num_rows,
                                size = p.packed_size,
                                "GPU hash partition self-transfer"
                            );
                            exchange_buffer.store_packed_gpu(
                                key.clone(),
                                PackedGpuExchange {
                                    gpu_addr: part_addr,
                                    gpu_size: p.packed_size,
                                    cudf_metadata: p.metadata.clone(),
                                },
                            );
                            exchange_buffer.add_block(&key, sender_id, None, true);
                        } else {
                            // Remote: nixl transfer this partition.
                            if let Some(agent) = _nixl_agent {
                                let part_buf = GpuBufferDesc { addr: part_addr, len: p.packed_size, device_id: 0 };
                                tracing::info!(
                                    dest_idx,
                                    dest = %dest.brpc_addr,
                                    rows = p.num_rows,
                                    size = p.packed_size,
                                    addr = format_args!("0x{part_addr:x}"),
                                    "GPU hash partition nixl transfer"
                                );
                                if let Err(e) = send_nixl_to_peer(
                                    agent, &[part_buf], &[], &[], p.num_rows,
                                    ipc_bytes, dest, query_id, node_id, sender_id,
                                    true, Some(&p.metadata),
                                ).await {
                                    tracing::warn!(error = %e, dest_idx, "nixl partition transfer failed, falling back to bRPC");
                                    // Fall back to bRPC for this partition.
                                    let part_ipc = &ipc_bytes; // TODO: partition-specific IPC
                                    crate::exchange_sender::send_exchange_result(
                                        part_ipc, &[dest.clone()], query_id, node_id, sender_id,
                                    ).await?;
                                }
                            } else {
                                // No nixl — bRPC fallback for this partition.
                                let part_ipc = &ipc_bytes;
                                crate::exchange_sender::send_exchange_result(
                                    part_ipc, &[dest.clone()], query_id, node_id, sender_id,
                                ).await?;
                            }
                        }
                    }
                    _ => {
                        // Empty partition — still need to signal EOS so the receiver
                        // doesn't wait forever for data that will never arrive.
                        if is_local {
                            exchange_buffer.add_block(&key, sender_id, None, true);
                        } else {
                            // Send empty EOS via bRPC for remote empty partitions.
                            crate::exchange_sender::send_eos(
                                dest, query_id, node_id, sender_id,
                            ).await.map_err(|e| format!("empty partition EOS: {e}"))?;
                        }
                        tracing::info!(dest_idx, dest = %dest.brpc_addr, is_local, "empty partition EOS sent");
                    }
                }
            }
            return Ok(());
        }
    }

    // CPU fallback: decode IPC bytes into Arrow RecordBatch for hashing.
    let batch = ipc_to_record_batch(ipc_bytes)?;

    let slots = desc_tbl_slots
        .ok_or_else(|| "hash partition requires descriptor table slots".to_string())?;

    let (col_indices, doris_types) = resolve_partition_columns(
        partition_exprs, slots, batch.schema().as_ref(),
    )?;

    tracing::info!(
        num_rows = batch.num_rows(),
        num_dests = num_destinations,
        partition_cols = ?col_indices,
        use_crc32c,
        "hash-partitioning exchange data"
    );

    let assignments = compute_dest_assignments(
        &batch, &col_indices, num_destinations, use_crc32c, &doris_types,
    );
    let partitions = split_by_destination(&batch, &assignments, num_destinations)?;

    // Log partition distribution.
    let row_counts: Vec<usize> = partitions.iter()
        .map(|p| p.as_ref().map_or(0, |b| b.num_rows()))
        .collect();
    tracing::info!(?row_counts, "hash partition distribution");

    // Send each partition to its destination.
    for (dest_idx, (dest, partition_batch)) in destinations.iter().zip(partitions.iter()).enumerate() {
        let is_local = dest.brpc_addr == local_brpc_addr;
        let key = ExchangeKey { query_id, node_id };

        if is_local {
            // Self-transfer via ExchangeBuffer.
            if let Some(batch) = partition_batch {
                let part_ipc = record_batch_to_ipc(batch)?;
                let (pblock, _) = crate::arrow_to_pblock::arrow_ipc_to_pblock(&part_ipc)
                    .map_err(|e| format!("hash partition self-transfer pblock: {e}"))?;
                exchange_buffer.add_block(&key, sender_id, Some(pblock), false);
            }
            // Always send EOS.
            exchange_buffer.add_block(&key, sender_id, None, true);
            tracing::info!(dest_idx, dest = %dest.brpc_addr, "hash partition self-transfer complete");
        } else {
            // Remote transfer via bRPC.
            // TODO(phase 4): GPU-direct nixl per-partition path.
            if let Some(batch) = partition_batch {
                let part_ipc = record_batch_to_ipc(batch)?;
                let (pblock, _) = crate::arrow_to_pblock::arrow_ipc_to_pblock(&part_ipc)
                    .map_err(|e| format!("hash partition bRPC pblock: {e}"))?;
                crate::exchange_sender::send_transmit_block(
                    dest, query_id, node_id, sender_id, Some(pblock), false, 0,
                ).await.map_err(|e| format!("hash partition send data to {}: {e}", dest.brpc_addr))?;
            }
            // Always send EOS.
            crate::exchange_sender::send_transmit_block(
                dest, query_id, node_id, sender_id, None, true, 1,
            ).await.map_err(|e| format!("hash partition send EOS to {}: {e}", dest.brpc_addr))?;
            tracing::info!(dest_idx, dest = %dest.brpc_addr, "hash partition bRPC send complete");
        }
    }

    Ok(())
}

/// Decode Arrow IPC bytes into a RecordBatch.
fn ipc_to_record_batch(ipc_bytes: &[u8]) -> Result<arrow::record_batch::RecordBatch, String> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    let cursor = Cursor::new(ipc_bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| format!("IPC stream reader: {e}"))?;

    let batches: Vec<_> = reader
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("IPC read batches: {e}"))?;

    if batches.is_empty() {
        return Err("no batches in IPC stream".to_string());
    }

    // Concatenate all batches if multiple (usually just one).
    if batches.len() == 1 {
        Ok(batches.into_iter().next().unwrap())
    } else {
        arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .map_err(|e| format!("concat IPC batches: {e}"))
    }
}

/// Encode a RecordBatch to Arrow IPC stream bytes.
fn record_batch_to_ipc(batch: &arrow::record_batch::RecordBatch) -> Result<Vec<u8>, String> {
    use arrow::ipc::writer::StreamWriter;

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| format!("IPC writer: {e}"))?;
        writer.write(batch).map_err(|e| format!("IPC write batch: {e}"))?;
        writer.finish().map_err(|e| format!("IPC finish: {e}"))?;
    }
    Ok(buf)
}

/// Send GPU data to a single peer via nixl.
///
/// Full flow: register buffers → exchange metadata → load peer metadata →
/// transfer → notify receiver of completion.
///
/// Transfers all sub-buffers (data, null_mask, offsets) for each column.
///
/// When `staged` is true, buffers are already in the staging buffer (cuMemAlloc-backed,
/// pre-registered) — skip per-buffer registration and use cached metadata.
/// Otherwise, fall back to per-buffer registration.
pub async fn send_nixl_to_peer(
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
    staged: bool,
    packed_metadata: Option<&[u8]>,
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
    // - staged: buffers are in pre-registered staging buffer → no registration needed
    // - otherwise: per-buffer registration (buffers should be cuMemAlloc-backed)
    let _src_registrations = if staged {
        info!("buffers staged in pre-registered staging buffer, skipping registration");
        vec![]
    } else {
        let buf_tuples: Vec<_> = src_buffers.iter().map(|b| (b.addr, b.len, b.device_id)).collect();
        agent.register_gpu_buffers(&buf_tuples)?
    };

    // Step 2: Get metadata — cached when staging buffer is active, fresh otherwise.
    let fresh_md = if staged {
        agent.get_metadata()?
    } else {
        agent.get_fresh_metadata()?
    };

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
        packed_cudf_metadata: packed_metadata.map(|m| m.to_vec()).unwrap_or_default(),
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
        packed_cudf_metadata: packed_metadata.map(|m| m.to_vec()).unwrap_or_default(),
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

    /// Build a broadcast ExchangeInfo for testing.
    fn make_broadcast_exch_info(dests: Vec<ExchangeDest>, node_id: i32) -> ExchangeInfo {
        ExchangeInfo {
            dest_node_id: node_id,
            destinations: dests,
            partition: PartitionStrategy::Broadcast,
        }
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
        let exch_info = make_broadcast_exch_info(vec![], 0);

        let result = send_exchange_with_nixl(
            None,
            location,
            &exch_info,
            (1, 2),
            0,
            false,
            "localhost:8060",
            &exchange_buffer,
            None,
        )
        .await;
        // No destinations → no work → Ok
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_location_into_ipc_bytes() {
        let ipc = vec![0x41, 0x52, 0x52, 0x4f, 0x57];
        let location = ExecutionLocation::Gpu {
            buffers: vec![GpuBufferDesc { addr: 0x1000, len: 256, device_id: 0 }],
            column_info: vec![("col1".to_string(), 5)],
            column_buffers: vec![GpuColumnBuffers {
                null_mask_addr: 0, null_mask_len: 0,
                offsets_addr: 0, offsets_len: 0,
                null_count: 0, scale: 0,
            }],
            num_rows: 10,
            schema_ipc: vec![],
            ipc_bytes: ipc.clone(),
            _staging_leases: vec![],
                    packed_metadata: None,
                    packed_partitions: vec![],
        };
        assert_eq!(location.into_ipc_bytes(), ipc);
    }

    #[test]
    fn test_cpu_location_into_ipc_bytes() {
        let ipc = vec![1, 2, 3];
        let location = ExecutionLocation::Cpu(ipc.clone());
        assert_eq!(location.into_ipc_bytes(), ipc);
    }

    #[test]
    fn test_gpu_location_no_removed_fields() {
        // Verify ExecutionLocation::Gpu no longer has cuda_alloc_addrs or rmm_pool_registered.
        // This is a compile-time check — if these fields existed, this wouldn't compile.
        let location = ExecutionLocation::Gpu {
            buffers: vec![],
            column_info: vec![],
            column_buffers: vec![],
            num_rows: 0,
            schema_ipc: vec![],
            ipc_bytes: vec![],
            _staging_leases: vec![],
                    packed_metadata: None,
                    packed_partitions: vec![],
        };
        match location {
            ExecutionLocation::Gpu { buffers, num_rows, _staging_leases, .. } => {
                assert!(buffers.is_empty());
                assert_eq!(num_rows, 0);
                assert!(_staging_leases.is_empty());
            }
            _ => panic!("expected Gpu"),
        }
    }

    #[tokio::test]
    async fn test_send_exchange_gpu_no_agent_brpc_fallback() {
        // GPU location but no nixl agent → should fall back to bRPC.
        // bRPC will fail (no server), but we verify the fallback logic.
        let ipc = vec![0xAA, 0xBB, 0xCC];
        let location = ExecutionLocation::Gpu {
            buffers: vec![GpuBufferDesc { addr: 0x1000, len: 256, device_id: 0 }],
            column_info: vec![("col1".to_string(), 5)],
            column_buffers: vec![GpuColumnBuffers {
                null_mask_addr: 0, null_mask_len: 0,
                offsets_addr: 0, offsets_len: 0,
                null_count: 0, scale: 0,
            }],
            num_rows: 10,
            schema_ipc: vec![],
            ipc_bytes: ipc,
            _staging_leases: vec![],
                    packed_metadata: None,
                    packed_partitions: vec![],
        };
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "10.0.0.99:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_broadcast_exch_info(dests, 0);

        let result = send_exchange_with_nixl(
            None,     // no nixl agent
            location,
            &exch_info,
            (1, 2),
            0,
            false,    // not nixl-only → bRPC fallback
            "localhost:8060",
            &exchange_buffer,
            None,
        )
        .await;
        // bRPC should fail (no server), but the path is exercised.
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_send_exchange_gpu_nixl_only_no_agent() {
        // GPU location, nixl_only=true, no agent → should error.
        let location = ExecutionLocation::Gpu {
            buffers: vec![GpuBufferDesc { addr: 0x1000, len: 256, device_id: 0 }],
            column_info: vec![],
            column_buffers: vec![],
            num_rows: 0,
            schema_ipc: vec![],
            ipc_bytes: vec![],
            _staging_leases: vec![],
                    packed_metadata: None,
                    packed_partitions: vec![],
        };
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "10.0.0.99:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_broadcast_exch_info(dests, 0);

        let result = send_exchange_with_nixl(
            None,
            location,
            &exch_info,
            (1, 2),
            0,
            true, // nixl-only
            "localhost:8060",
            &exchange_buffer,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nixl-only"));
    }

    #[tokio::test]
    async fn test_send_exchange_self_transfer() {
        // Self-transfer: destination matches local_brpc_addr → ExchangeBuffer path.
        let ipc = build_test_ipc();
        let location = ExecutionLocation::Cpu(ipc);
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "localhost:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_broadcast_exch_info(dests, 42);

        let result = send_exchange_with_nixl(
            None,
            location,
            &exch_info,
            (1, 2),
            0,
            false,
            "localhost:8060", // matches dest → self-transfer path
            &exchange_buffer,
            None,
        )
        .await;
        assert!(result.is_ok());
    }

    /// Build minimal valid Arrow IPC bytes for testing.
    fn build_test_ipc() -> Vec<u8> {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;

        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(
            std::sync::Arc::new(schema.clone()),
            vec![std::sync::Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buf
    }

    #[tokio::test]
    async fn test_send_exchange_cpu_nixl_only() {
        // CPU location with nixl_only → should error (can't use nixl for CPU data).
        let ipc = vec![0xAA];
        let location = ExecutionLocation::Cpu(ipc);
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "10.0.0.1:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_broadcast_exch_info(dests, 0);

        let result = send_exchange_with_nixl(
            None,
            location,
            &exch_info,
            (1, 2),
            0,
            true, // nixl-only
            "localhost:8060",
            &exchange_buffer,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nixl-only"));
    }
}
