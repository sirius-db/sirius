//! Integration layer between nixl GPU-direct exchange and query execution.
//!
//! Handles:
//! - Extracting GPU buffer pointers for nixl transfer
//! - Coordinating metadata exchange and transfer
//! - Fallback to bRPC when GPU-direct unavailable

use std::sync::Arc;

use crate::exchange_buffer::{ExchangeBuffer, ExchangeKey, PackedGpuExchange};
use crate::exchange_sender::ExchangeDest;
use crate::gpu_staging_buffer::StagingLease;
use crate::hash_partitioner::{ExchangeInfo, PartitionStrategy};
use crate::nixl_exchange::{GpuBufferDesc, NixlExchange};
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
        /// Per-partition packed buffers from GPU hash_partition + chunked_pack.
        /// When present, each entry has (staging_offset, size, cudf_metadata, num_rows).
        packed_partitions: Vec<sirius_ffi::PackedPartition>,
        /// Per-batch broadcast packed buffers. When present, each entry is a
        /// packed cudf buffer that should be sent to ALL exchange destinations.
        packed_broadcast: Vec<sirius_ffi::PackedBroadcastEntry>,
    },
    /// Result is owned by a Sirius exchange artifact.
    ///
    /// This is the modern exchange path: the packed descriptors are Rust-owned,
    /// while the opaque artifact keeps the underlying C++ session and GPU
    /// resources alive until transfer completion.
    PackedExchange {
        ipc_bytes: Vec<u8>,
        artifact: sirius_ffi::ExchangeArtifact,
    },
}

impl ExecutionLocation {
    pub fn from_exchange_artifact(
        artifact: sirius_ffi::ExchangeArtifact,
        ipc_bytes: Vec<u8>,
    ) -> Self {
        Self::PackedExchange {
            ipc_bytes,
            artifact,
        }
    }

    /// Extract IPC bytes, consuming self.
    pub fn into_ipc_bytes(self) -> Vec<u8> {
        match self {
            Self::Cpu(bytes) => bytes,
            Self::Gpu { ipc_bytes, .. } => ipc_bytes,
            Self::PackedExchange { ipc_bytes, .. } => ipc_bytes,
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
        if let Self::Gpu {
            buffers,
            column_buffers,
            _staging_leases,
            ..
        } = self
        {
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

/// Send exchange result using nixl GPU-direct if available, otherwise bRPC.
///
/// When `nixl_only` is true, bRPC fallback is disabled — errors surface instead
/// of being silently handled. Useful for debugging nixl exchange issues.
///
/// Self-transfer (destination is `local_brpc_addr`) uses a local fast-path:
/// packed GPU artifacts stay on GPU when available, otherwise IPC → PBlock →
/// ExchangeBuffer, bypassing both nixl and bRPC entirely.
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
        ExecutionLocation::PackedExchange { ipc_bytes, .. } => ipc_bytes.clone(),
    };

    // For local 1-BE exchange, force the proven CPU PBlock path everywhere.
    // The packed-GPU self-transfer path is the next design target, but it is
    // not stable enough yet for the baseline correctness run.
    let all_local = destinations.iter().all(|d| d.brpc_addr == local_brpc_addr);
    let location = if all_local {
        ExecutionLocation::Cpu(ipc_bytes.clone())
    } else {
        location
    };

    // For hash-partitioned exchange, split the data per destination first,
    // then route each partition's data to its destination (local or remote).
    if let PartitionStrategy::Hash {
        ref partition_exprs,
        num_destinations,
        use_crc32c,
    } = exch_info.partition
    {
        return send_hash_partitioned(
            nixl_agent,
            &location,
            &ipc_bytes,
            destinations,
            query_id,
            node_id,
            sender_id,
            nixl_only,
            local_brpc_addr,
            exchange_buffer,
            partition_exprs,
            num_destinations,
            use_crc32c,
            desc_tbl_slots,
        )
        .await;
    }

    let (buffers, packed_broadcast, packed_exchange) = if all_local {
        (&[][..], &[][..], None)
    } else {
        match &location {
            ExecutionLocation::Gpu {
                buffers,
                packed_broadcast,
                ..
            } => (buffers.as_slice(), packed_broadcast.as_slice(), None),
            ExecutionLocation::PackedExchange { artifact, .. } => (&[][..], &[][..], Some(artifact)),
            _ => (&[][..], &[][..], None),
        }
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

        if let Some(artifact) = packed_exchange {
            if !artifact.packed_broadcast().is_empty() {
                let staging_base = artifact.staging_base();
                for entry in artifact.packed_broadcast() {
                    if entry.packed_size == 0 {
                        continue;
                    }
                    let src_addr = if entry.overflow_gpu_addr != 0 {
                        entry.overflow_gpu_addr
                    } else {
                        staging_base + entry.staging_offset
                    };
                    let owned_addr = match crate::cuda_driver::cuda_alloc(entry.packed_size) {
                        Ok(addr) => {
                            if let Err(e) = crate::cuda_driver::cuda_memcpy_dtod(
                                addr,
                                src_addr,
                                entry.packed_size,
                            ) {
                                tracing::warn!(error = %e, "self-transfer D2D copy failed, using staging addr");
                                crate::cuda_driver::cuda_free(addr).ok();
                                src_addr
                            } else {
                                addr
                            }
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "self-transfer cudaMalloc failed, using staging addr");
                            src_addr
                        }
                    };
                    exchange_buffer.store_packed_gpu(
                        key.clone(),
                        PackedGpuExchange {
                            gpu_addr: owned_addr,
                            gpu_size: entry.packed_size,
                            cudf_metadata: entry.metadata.clone(),
                            _staging_lease: None,
                        },
                    );
                }
                exchange_buffer.add_block(&key, sender_id, None, true);
                tracing::info!(dest = %dest.brpc_addr, sender_id, num_entries = artifact.packed_broadcast().len(),
                              "self-transfer: broadcast packed GPU (artifact-backed)");
                continue;
            }
        }

        // Broadcast packed entries: copy from staging to owned buffer.
        // The staging region is reused by later fragments, so self-transfer must
        // take ownership before the next GPU execution overwrites it.
        if !packed_broadcast.is_empty() {
            let staging_base = buffers.first().map(|b| b.addr).unwrap_or(0);
            for entry in packed_broadcast {
                if entry.packed_size == 0 {
                    continue;
                }
                let src_addr = if entry.overflow_gpu_addr != 0 {
                    entry.overflow_gpu_addr
                } else {
                    staging_base + entry.staging_offset
                };
                // D2D copy to new cudaMalloc buffer (independent of staging).
                let owned_addr = match crate::cuda_driver::cuda_alloc(entry.packed_size) {
                    Ok(addr) => {
                        if let Err(e) =
                            crate::cuda_driver::cuda_memcpy_dtod(addr, src_addr, entry.packed_size)
                        {
                            tracing::warn!(error = %e, "self-transfer D2D copy failed, using staging addr");
                            crate::cuda_driver::cuda_free(addr).ok();
                            src_addr
                        } else {
                            addr
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "self-transfer cudaMalloc failed, using staging addr");
                        src_addr
                    }
                };
                exchange_buffer.store_packed_gpu(
                    key.clone(),
                    PackedGpuExchange {
                        gpu_addr: owned_addr,
                        gpu_size: entry.packed_size,
                        cudf_metadata: entry.metadata.clone(),
                        _staging_lease: None,
                    },
                );
            }
            exchange_buffer.add_block(&key, sender_id, None, true);
            tracing::info!(dest = %dest.brpc_addr, sender_id, num_entries = packed_broadcast.len(),
                          "self-transfer: broadcast packed GPU (D2D copied from staging)");
            continue;
        }

        tracing::info!(
            dest = %dest.brpc_addr,
            sender_id,
            ipc_len = ipc_bytes.len(),
            "self-transfer: feeding data directly into local ExchangeBuffer"
        );
        if ipc_bytes.is_empty() {
            // Empty IPC: send EOS without data (0-row exchange).
            exchange_buffer.add_block(&key, sender_id, None, true);
            tracing::info!(dest = %dest.brpc_addr, sender_id, "self-transfer: empty IPC, sent EOS only");
            continue;
        }
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
                tracing::error!(
                    "nixl-only mode: result is CPU-resident, cannot use nixl GPU-direct transfer"
                );
                return Err("nixl-only: no GPU buffers for GPU-direct exchange (execution fell back to CPU?)".to_string());
            }
            crate::exchange_sender::send_exchange_result(
                &ipc_bytes,
                &remote_dests,
                query_id,
                node_id,
                sender_id,
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
            packed_broadcast,
        } => {
            let Some(agent) = nixl_agent else {
                if nixl_only {
                    return Err("nixl-only: GPU result but no nixl agent available".to_string());
                }
                tracing::warn!("GPU result but no nixl agent, falling back to bRPC");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes,
                    &remote_dests,
                    query_id,
                    node_id,
                    sender_id,
                )
                .await;
            };

            // Fast-path: skip nixl if a previous registration already detected
            // that UCX treats GPU memory as host memory (would SIGSEGV).
            if !agent.gpu_transfer_enabled() {
                if nixl_only {
                    return Err("nixl-only: UCX lacks CUDA support for GPU memory".to_string());
                }
                tracing::info!("UCX lacks CUDA support for GPU memory, using bRPC for exchange");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes,
                    &remote_dests,
                    query_id,
                    node_id,
                    sender_id,
                )
                .await;
            }

            // Broadcast with packed entries: send each entry to all destinations.
            if !packed_broadcast.is_empty() {
                tracing::info!(
                    num_entries = packed_broadcast.len(),
                    num_dests = remote_dests.len(),
                    "sending broadcast GPU exchange data (per-batch packed)"
                );
                let staging_base = buffers.first().map(|b| b.addr).unwrap_or(0);
                for entry in &packed_broadcast {
                    if entry.packed_size == 0 {
                        continue;
                    }
                    // Determine the GPU address of this entry.
                    let gpu_addr = if entry.overflow_gpu_addr != 0 {
                        entry.overflow_gpu_addr
                    } else {
                        staging_base + entry.staging_offset
                    };
                    let buf_desc = GpuBufferDesc {
                        addr: gpu_addr,
                        len: entry.packed_size,
                        device_id: 0,
                    };

                    for dest in &remote_dests {
                        if let Err(e) = send_nixl_to_peer(
                            agent,
                            &[buf_desc.clone()],
                            &[],
                            &[],
                            entry.num_rows,
                            &ipc_bytes,
                            dest,
                            query_id,
                            node_id,
                            sender_id,
                            true,
                            Some(&entry.metadata),
                        )
                        .await
                        {
                            if nixl_only {
                                return Err(format!(
                                    "nixl-only: broadcast nixl transfer to {} failed: {}",
                                    dest.brpc_addr, e
                                ));
                            }
                            tracing::warn!(error = %e, dest = %dest.brpc_addr, "broadcast nixl transfer failed, falling back to bRPC");
                            crate::exchange_sender::send_exchange_result(
                                &ipc_bytes,
                                &remote_dests,
                                query_id,
                                node_id,
                                sender_id,
                            )
                            .await?;
                            return Ok(());
                        }
                    }
                }
                return Ok(());
            }

            // Packed buffers are always in the staging region.
            let staged = !_staging_leases.is_empty();

            // Try nixl GPU-direct for each remote destination.
            for dest in &remote_dests {
                if let Err(e) = send_nixl_to_peer(
                    agent,
                    &buffers,
                    &column_info,
                    &column_buffers,
                    num_rows,
                    &ipc_bytes,
                    dest,
                    query_id,
                    node_id,
                    sender_id,
                    staged,
                    None,
                )
                .await
                {
                    if nixl_only {
                        return Err(format!(
                            "nixl-only: nixl transfer to {} failed: {}",
                            dest.brpc_addr, e
                        ));
                    }
                    tracing::warn!(
                        error = %e,
                        dest = %dest.brpc_addr,
                        "nixl transfer failed, falling back to bRPC"
                    );
                    crate::exchange_sender::send_exchange_result(
                        &ipc_bytes,
                        &remote_dests,
                        query_id,
                        node_id,
                        sender_id,
                    )
                    .await?;
                    return Ok(());
                }
            }

            Ok(())
        }
        ExecutionLocation::PackedExchange { artifact, .. } => {
            let Some(agent) = nixl_agent else {
                if nixl_only {
                    return Err(
                        "nixl-only: GPU exchange artifact but no nixl agent available".to_string(),
                    );
                }
                tracing::warn!("GPU exchange artifact but no nixl agent, falling back to bRPC");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes,
                    &remote_dests,
                    query_id,
                    node_id,
                    sender_id,
                )
                .await;
            };

            if !agent.gpu_transfer_enabled() {
                if nixl_only {
                    return Err("nixl-only: UCX lacks CUDA support for GPU memory".to_string());
                }
                tracing::info!("UCX lacks CUDA support for GPU memory, using bRPC for exchange");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes,
                    &remote_dests,
                    query_id,
                    node_id,
                    sender_id,
                )
                .await;
            }

            if !artifact.packed_broadcast().is_empty() {
                tracing::info!(
                    num_entries = artifact.packed_broadcast().len(),
                    num_dests = remote_dests.len(),
                    staging_base = format_args!("0x{:x}", artifact.staging_base()),
                    "sending broadcast GPU exchange data (artifact-backed)"
                );
                for entry in artifact.packed_broadcast() {
                    if entry.packed_size == 0 {
                        continue;
                    }
                    let gpu_addr = if entry.overflow_gpu_addr != 0 {
                        entry.overflow_gpu_addr
                    } else {
                        artifact.staging_base() + entry.staging_offset
                    };
                    let buf_desc = GpuBufferDesc {
                        addr: gpu_addr,
                        len: entry.packed_size,
                        device_id: 0,
                    };

                    for dest in &remote_dests {
                        if let Err(e) = send_nixl_to_peer(
                            agent,
                            &[buf_desc.clone()],
                            &[],
                            &[],
                            entry.num_rows,
                            &ipc_bytes,
                            dest,
                            query_id,
                            node_id,
                            sender_id,
                            entry.overflow_gpu_addr == 0,
                            Some(&entry.metadata),
                        )
                        .await
                        {
                            if nixl_only {
                                return Err(format!(
                                    "nixl-only: broadcast nixl transfer to {} failed: {}",
                                    dest.brpc_addr, e
                                ));
                            }
                            tracing::warn!(error = %e, dest = %dest.brpc_addr, "broadcast nixl transfer failed, falling back to bRPC");
                            crate::exchange_sender::send_exchange_result(
                                &ipc_bytes,
                                &remote_dests,
                                query_id,
                                node_id,
                                sender_id,
                            )
                            .await?;
                            return Ok(());
                        }
                    }
                }
                return Ok(());
            }

            if nixl_only {
                return Err(
                    "nixl-only: packed exchange artifact had no broadcast entries".to_string(),
                );
            }
            crate::exchange_sender::send_exchange_result(
                &ipc_bytes,
                &remote_dests,
                query_id,
                node_id,
                sender_id,
            )
            .await
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
    use crate::hash_partitioner::{
        compute_dest_assignments, resolve_partition_columns, split_by_destination,
    };

    // GPU hash-partitioned path: use per-partition packed buffers from staging.
    if let ExecutionLocation::PackedExchange { artifact, .. } = _location {
        let packed_partitions = artifact.packed_partitions();
        if !packed_partitions.is_empty() {
            tracing::info!(
                num_partitions = packed_partitions.len(),
                num_dests = destinations.len(),
                staging_base = format_args!("0x{:x}", artifact.staging_base()),
                "sending GPU hash-partitioned exchange data (artifact-backed)"
            );

            let send_staging_base = artifact.staging_base();
            let num_dests = destinations.len();
            for (dest_idx, dest) in destinations.iter().enumerate() {
                let key = ExchangeKey { query_id, node_id };
                let dest_entries: Vec<&sirius_ffi::PackedPartition> = packed_partitions
                    .iter()
                    .skip(dest_idx)
                    .step_by(num_dests)
                    .filter(|p| p.packed_size > 0)
                    .collect();
                let is_local = dest.brpc_addr == local_brpc_addr;

                if !dest_entries.is_empty() {
                    let total_rows: u32 = dest_entries.iter().map(|e| e.num_rows).sum();
                    for p in &dest_entries {
                        let part_addr = if p.overflow_gpu_addr != 0 {
                            p.overflow_gpu_addr
                        } else {
                            send_staging_base + p.staging_offset
                        };

                        if is_local {
                            let owned_addr = match crate::cuda_driver::cuda_alloc(p.packed_size) {
                                Ok(addr) => match crate::cuda_driver::cuda_memcpy_dtod(
                                    addr,
                                    part_addr,
                                    p.packed_size,
                                ) {
                                    Ok(()) => {
                                        tracing::info!(
                                            src = format_args!("0x{part_addr:x}"),
                                            dst = format_args!("0x{addr:x}"),
                                            size = p.packed_size,
                                            "hash partition self-transfer: D2D copy OK"
                                        );
                                        addr
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, src = format_args!("0x{part_addr:x}"),
                                            "hash partition self-transfer: D2D COPY FAILED — using stale staging addr!");
                                        crate::cuda_driver::cuda_free(addr).ok();
                                        part_addr
                                    }
                                },
                                Err(e) => {
                                    tracing::error!(error = %e, size = p.packed_size,
                                        "hash partition self-transfer: cudaMalloc FAILED — using stale staging addr!");
                                    part_addr
                                }
                            };
                            exchange_buffer.store_packed_gpu(
                                key.clone(),
                                PackedGpuExchange {
                                    gpu_addr: owned_addr,
                                    gpu_size: p.packed_size,
                                    cudf_metadata: p.metadata.clone(),
                                    _staging_lease: None,
                                },
                            );
                        } else if let Some(agent) = _nixl_agent {
                            let is_staged = p.overflow_gpu_addr == 0;
                            let part_buf = GpuBufferDesc {
                                addr: part_addr,
                                len: p.packed_size,
                                device_id: 0,
                            };
                            if let Err(e) = send_nixl_to_peer(
                                agent,
                                &[part_buf],
                                &[],
                                &[],
                                p.num_rows,
                                ipc_bytes,
                                dest,
                                query_id,
                                node_id,
                                sender_id,
                                is_staged,
                                Some(&p.metadata),
                            )
                            .await
                            {
                                if _nixl_only {
                                    return Err(format!(
                                        "nixl-only: partition transfer to {} failed: {}",
                                        dest.brpc_addr, e
                                    ));
                                }
                                tracing::warn!(error = %e, dest_idx, "nixl partition transfer failed, falling back to bRPC");
                                crate::exchange_sender::send_exchange_result(
                                    ipc_bytes,
                                    &[dest.clone()],
                                    query_id,
                                    node_id,
                                    sender_id,
                                )
                                .await?;
                            }
                        } else {
                            if _nixl_only {
                                return Err("nixl-only: no nixl agent for hash partition transfer"
                                    .to_string());
                            }
                            crate::exchange_sender::send_exchange_result(
                                ipc_bytes,
                                &[dest.clone()],
                                query_id,
                                node_id,
                                sender_id,
                            )
                            .await?;
                        }
                    }
                    if is_local {
                        exchange_buffer.add_block(&key, sender_id, None, true);
                    }
                    tracing::info!(
                        dest_idx,
                        dest = %dest.brpc_addr,
                        num_entries = dest_entries.len(),
                        total_rows,
                        "GPU hash partition transfer complete"
                    );
                } else {
                    if is_local {
                        exchange_buffer.add_block(&key, sender_id, None, true);
                    } else {
                        crate::exchange_sender::send_eos(dest, query_id, node_id, sender_id)
                            .await
                            .map_err(|e| format!("empty partition EOS: {e}"))?;
                    }
                    tracing::info!(dest_idx, dest = %dest.brpc_addr, is_local, "empty partition EOS sent");
                }
            }
            return Ok(());
        }
    }

    if let ExecutionLocation::Gpu {
        ref packed_partitions,
        ref buffers,
        ..
    } = _location
    {
        if !packed_partitions.is_empty() {
            tracing::info!(
                num_partitions = packed_partitions.len(),
                num_dests = destinations.len(),
                "sending GPU hash-partitioned exchange data"
            );

            // Get the send staging base address from the first buffer.
            // Partitions are at staging_base + partition.staging_offset.
            let staging_base = buffers.first().map(|b| b.addr).unwrap_or(0);
            let send_staging_base = staging_base;

            // Partition entries may come from multiple batches. With N dests
            // and B batches, there are N*B entries: [b0_d0, b0_d1, b1_d0, b1_d1, ...].
            // Collect all entries for each destination.
            let num_dests = destinations.len();
            for (dest_idx, dest) in destinations.iter().enumerate() {
                let key = ExchangeKey { query_id, node_id };

                // Collect all entries for this destination across batches.
                let dest_entries: Vec<&sirius_ffi::PackedPartition> = packed_partitions
                    .iter()
                    .skip(dest_idx)
                    .step_by(num_dests)
                    .filter(|p| p.packed_size > 0)
                    .collect();

                let is_local = dest.brpc_addr == local_brpc_addr;

                if !dest_entries.is_empty() {
                    let total_rows: u32 = dest_entries.iter().map(|e| e.num_rows).sum();
                    // Send each batch's partition data for this destination.
                    for p in &dest_entries {
                        // Use overflow address if partition didn't fit in staging.
                        let part_addr = if p.overflow_gpu_addr != 0 {
                            p.overflow_gpu_addr
                        } else {
                            send_staging_base + p.staging_offset
                        };

                        if is_local {
                            // D2D copy from staging to owned buffer (staging may be
                            // released/reused between leaf fragments of the same query).
                            let owned_addr = match crate::cuda_driver::cuda_alloc(p.packed_size) {
                                Ok(addr) => match crate::cuda_driver::cuda_memcpy_dtod(
                                    addr,
                                    part_addr,
                                    p.packed_size,
                                ) {
                                    Ok(()) => {
                                        tracing::info!(
                                            src = format_args!("0x{part_addr:x}"),
                                            dst = format_args!("0x{addr:x}"),
                                            size = p.packed_size,
                                            "hash partition self-transfer: D2D copy OK"
                                        );
                                        addr
                                    }
                                    Err(e) => {
                                        tracing::error!(error = %e, src = format_args!("0x{part_addr:x}"),
                                            "hash partition self-transfer: D2D COPY FAILED — using stale staging addr!");
                                        crate::cuda_driver::cuda_free(addr).ok();
                                        part_addr
                                    }
                                },
                                Err(e) => {
                                    tracing::error!(error = %e, size = p.packed_size,
                                        "hash partition self-transfer: cudaMalloc FAILED — using stale staging addr!");
                                    part_addr
                                }
                            };
                            exchange_buffer.store_packed_gpu(
                                key.clone(),
                                PackedGpuExchange {
                                    gpu_addr: owned_addr,
                                    gpu_size: p.packed_size,
                                    cudf_metadata: p.metadata.clone(),
                                    _staging_lease: None,
                                },
                            );
                        } else if let Some(agent) = _nixl_agent {
                            // Overflow buffers are not in the pre-registered staging region;
                            // send_nixl_to_peer will handle registration (staged=false triggers
                            // on-demand registration via register_gpu_buffers).
                            let is_staged = p.overflow_gpu_addr == 0;
                            let part_buf = GpuBufferDesc {
                                addr: part_addr,
                                len: p.packed_size,
                                device_id: 0,
                            };
                            if let Err(e) = send_nixl_to_peer(
                                agent,
                                &[part_buf],
                                &[],
                                &[],
                                p.num_rows,
                                ipc_bytes,
                                dest,
                                query_id,
                                node_id,
                                sender_id,
                                is_staged,
                                Some(&p.metadata),
                            )
                            .await
                            {
                                if _nixl_only {
                                    return Err(format!(
                                        "nixl-only: partition transfer to {} failed: {}",
                                        dest.brpc_addr, e
                                    ));
                                }
                                tracing::warn!(error = %e, dest_idx, "nixl partition transfer failed, falling back to bRPC");
                                crate::exchange_sender::send_exchange_result(
                                    ipc_bytes,
                                    &[dest.clone()],
                                    query_id,
                                    node_id,
                                    sender_id,
                                )
                                .await?;
                            }
                        } else {
                            if _nixl_only {
                                return Err("nixl-only: no nixl agent for hash partition transfer"
                                    .to_string());
                            }
                            crate::exchange_sender::send_exchange_result(
                                ipc_bytes,
                                &[dest.clone()],
                                query_id,
                                node_id,
                                sender_id,
                            )
                            .await?;
                        }
                    }
                    // Signal EOS after all batch entries for this destination.
                    if is_local {
                        exchange_buffer.add_block(&key, sender_id, None, true);
                    }
                    tracing::info!(
                        dest_idx,
                        dest = %dest.brpc_addr,
                        num_entries = dest_entries.len(),
                        total_rows,
                        "GPU hash partition transfer complete"
                    );
                } else {
                    // Empty partition — still need to signal EOS so the receiver
                    // doesn't wait forever for data that will never arrive.
                    if is_local {
                        exchange_buffer.add_block(&key, sender_id, None, true);
                    } else {
                        crate::exchange_sender::send_eos(dest, query_id, node_id, sender_id)
                            .await
                            .map_err(|e| format!("empty partition EOS: {e}"))?;
                    }
                    tracing::info!(dest_idx, dest = %dest.brpc_addr, is_local, "empty partition EOS sent");
                }
            }
            return Ok(());
        }
    }

    // CPU fallback: decode IPC bytes into Arrow RecordBatch for hashing.
    let batch = ipc_to_record_batch(ipc_bytes)?;

    let slots = desc_tbl_slots
        .ok_or_else(|| "hash partition requires descriptor table slots".to_string())?;

    let (col_indices, doris_types) =
        resolve_partition_columns(partition_exprs, slots, batch.schema().as_ref())?;

    tracing::info!(
        num_rows = batch.num_rows(),
        num_dests = num_destinations,
        partition_cols = ?col_indices,
        use_crc32c,
        "hash-partitioning exchange data"
    );

    let assignments = compute_dest_assignments(
        &batch,
        &col_indices,
        num_destinations,
        use_crc32c,
        &doris_types,
    );
    let partitions = split_by_destination(&batch, &assignments, num_destinations)?;

    // Log partition distribution.
    let row_counts: Vec<usize> = partitions
        .iter()
        .map(|p| p.as_ref().map_or(0, |b| b.num_rows()))
        .collect();
    tracing::info!(?row_counts, "hash partition distribution");

    // Send each partition to its destination.
    for (dest_idx, (dest, partition_batch)) in
        destinations.iter().zip(partitions.iter()).enumerate()
    {
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
                    dest,
                    query_id,
                    node_id,
                    sender_id,
                    Some(pblock),
                    false,
                    0,
                )
                .await
                .map_err(|e| format!("hash partition send data to {}: {e}", dest.brpc_addr))?;
            }
            // Always send EOS.
            crate::exchange_sender::send_transmit_block(
                dest, query_id, node_id, sender_id, None, true, 1,
            )
            .await
            .map_err(|e| format!("hash partition send EOS to {}: {e}", dest.brpc_addr))?;
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
    let reader =
        StreamReader::try_new(cursor, None).map_err(|e| format!("IPC stream reader: {e}"))?;

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
        writer
            .write(batch)
            .map_err(|e| format!("IPC write batch: {e}"))?;
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
        NixlMetadataServiceClient, PColumnInfo, PExchangeNixlMetadataRequest, PGpuBufferDesc,
        PNixlTransferCompleteRequest,
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
        let buf_tuples: Vec<_> = src_buffers
            .iter()
            .map(|b| (b.addr, b.len, b.device_id))
            .collect();
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
                    25 => 9,  // DECIMAL32
                    26 => 18, // DECIMAL64
                    27 => 38, // DECIMAL128
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
    let remote_name = agent.force_load_remote_metadata(&dest.brpc_addr, &response.nixl_metadata)?;

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
        if let (Some(cb), Some(dst_nm)) = (column_buffers.get(i), response.dst_null_masks.get(i)) {
            if cb.null_mask_addr != 0 && cb.null_mask_len > 0 && dst_nm.addr != 0 && dst_nm.len > 0
            {
                all_src_ptrs.push((cb.null_mask_addr, cb.null_mask_len));
                all_dst_ptrs.push((dst_nm.addr as usize, dst_nm.len as usize));
            }
        }

        // String offsets (only if both sender has data and receiver allocated space).
        if let (Some(cb), Some(dst_off)) = (column_buffers.get(i), response.dst_offsets.get(i)) {
            if cb.offsets_addr != 0 && cb.offsets_len > 0 && dst_off.addr != 0 && dst_off.len > 0 {
                all_src_ptrs.push((cb.offsets_addr, cb.offsets_len));
                all_dst_ptrs.push((dst_off.addr as usize, dst_off.len as usize));
            }
        }
    }

    let num_total_transfers = all_src_ptrs.len();
    info!(
        num_data_buffers = src_buffers.len(),
        num_total_transfers, "nixl transfer: flattened buffer pairs"
    );

    // SENDER-SIDE integrity check: compute checksum of each src buffer.
    for (i, (addr, len)) in all_src_ptrs.iter().enumerate() {
        if let Ok(buf) = crate::cuda_driver::gpu_to_host(*addr, *len) {
            let checksum: u64 = buf.iter().map(|&b| b as u64).sum();
            info!(
                buf_idx = i,
                src_addr = format_args!("0x{addr:x}"),
                src_len = len,
                checksum,
                "nixl transfer: SENDER buffer checksum"
            );
        }
    }

    // Execute transfer on blocking thread (nixl uses polling).
    {
        let agent = agent.clone();
        let remote = remote_name.clone();

        tokio::task::spawn_blocking(move || {
            // Ensure the tokio worker thread has an active CUDA primary context.
            crate::cuda_driver::ensure_cuda_context()?;

            // Synchronize the CUDA device before the nixl transfer. The sender's
            // cudf::chunked_pack writes to SEND staging using the cudf default stream.
            // The nixl RDMA transfer reads from the same buffer using UCX's internal
            // stream. Without this barrier, the transfer can read stale/partial data
            // from SEND staging (the pack hasn't completed on the default stream yet).
            unsafe { cudarc::driver::result::ctx::synchronize() }
                .map_err(|e| format!("cuCtxSynchronize before nixl transfer: {e}"))?;

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
                    25 => 9,
                    26 => 18,
                    27 => 38,
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
        sender_id,
        arrow_ipc_data: vec![], // No longer sent — all transfers use packed cudf
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
    use std::sync::Arc;

    /// Build a broadcast ExchangeInfo for testing.
    fn make_broadcast_exch_info(dests: Vec<ExchangeDest>, node_id: i32) -> ExchangeInfo {
        ExchangeInfo {
            dest_node_id: node_id,
            destinations: dests,
            partition: PartitionStrategy::Broadcast,
            output_names: Vec::new(),
            output_indices: None,
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
            ExecutionLocation::Gpu { .. } | ExecutionLocation::PackedExchange { .. } => {
                panic!("expected Cpu variant")
            }
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
            buffers: vec![GpuBufferDesc {
                addr: 0x1000,
                len: 256,
                device_id: 0,
            }],
            column_info: vec![("col1".to_string(), 5)],
            column_buffers: vec![GpuColumnBuffers {
                null_mask_addr: 0,
                null_mask_len: 0,
                offsets_addr: 0,
                offsets_len: 0,
                null_count: 0,
                scale: 0,
            }],
            num_rows: 10,
            schema_ipc: vec![],
            ipc_bytes: ipc.clone(),
            _staging_leases: vec![],
            packed_partitions: vec![],
            packed_broadcast: vec![],
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
            packed_partitions: vec![],
            packed_broadcast: vec![],
        };
        match location {
            ExecutionLocation::Gpu {
                buffers,
                num_rows,
                _staging_leases,
                ..
            } => {
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
            buffers: vec![GpuBufferDesc {
                addr: 0x1000,
                len: 256,
                device_id: 0,
            }],
            column_info: vec![("col1".to_string(), 5)],
            column_buffers: vec![GpuColumnBuffers {
                null_mask_addr: 0,
                null_mask_len: 0,
                offsets_addr: 0,
                offsets_len: 0,
                null_count: 0,
                scale: 0,
            }],
            num_rows: 10,
            schema_ipc: vec![],
            ipc_bytes: ipc,
            _staging_leases: vec![],
            packed_partitions: vec![],
            packed_broadcast: vec![],
        };
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "10.0.0.99:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_broadcast_exch_info(dests, 0);

        let result = send_exchange_with_nixl(
            None, // no nixl agent
            location,
            &exch_info,
            (1, 2),
            0,
            false, // not nixl-only → bRPC fallback
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
            buffers: vec![GpuBufferDesc {
                addr: 0x1000,
                len: 256,
                device_id: 0,
            }],
            column_info: vec![],
            column_buffers: vec![],
            num_rows: 0,
            schema_ipc: vec![],
            ipc_bytes: vec![],
            _staging_leases: vec![],
            packed_partitions: vec![],
            packed_broadcast: vec![],
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

    fn bigint_type_desc() -> doris_thrift::types::TTypeDesc {
        use doris_thrift::types::{
            TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType,
        };

        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(TScalarType {
                    type_: TPrimitiveType::BIGINT,
                    len: None,
                    precision: None,
                    scale: None,
                    variant_max_subcolumns_count: None,
                }),
                struct_fields: None,
                contains_null: None,
                contains_nulls: None,
            }]),
            is_nullable: Some(true),
            byte_size: None,
            sub_types: None,
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
        }
    }

    fn slot_ref_expr(slot_id: i32) -> doris_thrift::exprs::TExpr {
        use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType, TSlotRef};

        TExpr {
            nodes: vec![TExprNode {
                node_type: TExprNodeType::SLOT_REF,
                type_: bigint_type_desc(),
                num_children: 0,
                fn_: None,
                opcode: None,
                child_type: None,
                output_scale: 0,
                is_nullable: None,
                vector_opcode: None,
                vararg_start_idx: None,
                agg_expr: None,
                int_literal: None,
                float_literal: None,
                string_literal: None,
                bool_literal: None,
                decimal_literal: None,
                date_literal: None,
                large_int_literal: None,
                json_literal: None,
                slot_ref: Some(TSlotRef {
                    slot_id,
                    tuple_id: 0,
                    col_unique_id: None,
                    is_virtual_slot: None,
                }),
                case_expr: None,
                in_predicate: None,
                is_null_pred: None,
                like_pred: None,
                literal_pred: None,
                info_func: None,
                tuple_is_null_pred: None,
                fn_call_expr: None,
                output_column: None,
                output_type: None,
                schema_change_expr: None,
                column_ref: None,
                match_predicate: None,
                ipv4_literal: None,
                ipv6_literal: None,
                label: None,
                timev2_literal: None,
                varbinary_literal: None,
                is_cast_nullable: None,
                search_param: None,
            }],
        }
    }

    fn make_hash_exch_info(
        dests: Vec<ExchangeDest>,
        node_id: i32,
        partition_slot_id: i32,
    ) -> ExchangeInfo {
        ExchangeInfo {
            dest_node_id: node_id,
            destinations: dests.clone(),
            partition: PartitionStrategy::Hash {
                partition_exprs: vec![slot_ref_expr(partition_slot_id)],
                num_destinations: dests.len(),
                use_crc32c: true,
            },
            output_names: Vec::new(),
            output_indices: None,
        }
    }

    fn build_partsupp_like_ipc(num_rows: usize) -> Vec<u8> {
        use arrow::array::{Float64Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;

        let schema = Arc::new(Schema::new(vec![
            Field::new("ps_partkey", DataType::Int64, false),
            Field::new("ps_suppkey", DataType::Int64, false),
            Field::new("ps_availqty", DataType::Int64, false),
            Field::new("ps_supplycost", DataType::Float64, false),
        ]));

        let partkeys: Vec<i64> = (0..num_rows).map(|i| 1000 + (i % 200_000) as i64).collect();
        let suppkeys: Vec<i64> = (0..num_rows).map(|i| 1 + (i % 10_000) as i64).collect();
        let availqty: Vec<i64> = (0..num_rows).map(|i| 1 + (i % 9_999) as i64).collect();
        let supplycost: Vec<f64> = (0..num_rows)
            .map(|i| ((i % 100_000) as f64) / 100.0)
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(partkeys)),
                Arc::new(Int64Array::from(suppkeys)),
                Arc::new(Int64Array::from(availqty)),
                Arc::new(Float64Array::from(supplycost)),
            ],
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

    #[tokio::test]
    #[ignore = "debug reproducer for single-destination hash self-transfer heap corruption"]
    async fn repro_hash_self_transfer_single_dest_large_partsupp_like_batch() {
        let ipc = build_partsupp_like_ipc(800_000);
        let location = ExecutionLocation::Cpu(ipc);
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "localhost:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_hash_exch_info(dests, 42, 7);

        let result = send_exchange_with_nixl(
            None,
            location,
            &exch_info,
            (11, 11),
            0,
            false,
            "localhost:8060",
            &exchange_buffer,
            Some(&[(7, "ps_suppkey".to_string())]),
        )
        .await;

        assert!(result.is_ok(), "hash self-transfer failed: {:?}", result);
    }

    #[tokio::test]
    #[ignore = "debug reproducer using Sirius GPU output from TPC-H partsupp scan"]
    async fn repro_hash_self_transfer_single_dest_partsupp_gpu_output() {
        let partsupp_dir = std::path::Path::new("/data/tpch/sf1/p16/snappy/partsupp");
        if !partsupp_dir.exists() {
            eprintln!(
                "skipping reproducer: {} not present",
                partsupp_dir.display()
            );
            return;
        }

        let engine = sirius_ffi::SiriusEngine::new().expect("SiriusEngine::new");
        engine.execute_sql("LOAD parquet").expect("LOAD parquet");
        engine
            .init_gpu_buffers("1GB", "1GB")
            .expect("init_gpu_buffers");
        engine.set_no_cpu_fallback().expect("set_no_cpu_fallback");

        let sql = "SELECT ps_partkey, ps_suppkey, ps_availqty, ps_supplycost \
                   FROM read_parquet('/data/tpch/sf1/p16/snappy/partsupp/*.parquet')";
        let ipc = engine.execute_gpu(sql).expect("execute_gpu partsupp");

        let location = ExecutionLocation::Cpu(ipc);
        let exchange_buffer = crate::exchange_buffer::ExchangeBuffer::new();
        let dests = vec![ExchangeDest {
            brpc_addr: "localhost:8060".to_string(),
            finst_id: (1, 1),
        }];
        let exch_info = make_hash_exch_info(dests, 42, 7);

        let result = send_exchange_with_nixl(
            None,
            location,
            &exch_info,
            (11, 12),
            0,
            false,
            "localhost:8060",
            &exchange_buffer,
            Some(&[(7, "ps_suppkey".to_string())]),
        )
        .await;

        assert!(result.is_ok(), "hash self-transfer failed: {:?}", result);
    }

    #[test]
    #[ignore = "debug reproducer for teardown after a single GPU partsupp scan"]
    fn repro_partsupp_gpu_scan_teardown_only() {
        let partsupp_dir = std::path::Path::new("/data/tpch/sf1/p16/snappy/partsupp");
        if !partsupp_dir.exists() {
            eprintln!(
                "skipping reproducer: {} not present",
                partsupp_dir.display()
            );
            return;
        }

        let engine = sirius_ffi::SiriusEngine::new().expect("SiriusEngine::new");
        engine.execute_sql("LOAD parquet").expect("LOAD parquet");
        engine
            .init_gpu_buffers("1GB", "1GB")
            .expect("init_gpu_buffers");
        engine.set_no_cpu_fallback().expect("set_no_cpu_fallback");

        let sql = "SELECT ps_partkey, ps_suppkey, ps_availqty, ps_supplycost \
                   FROM read_parquet('/data/tpch/sf1/p16/snappy/partsupp/*.parquet')";
        let ipc = engine.execute_gpu(sql).expect("execute_gpu partsupp");
        assert!(!ipc.is_empty(), "expected non-empty IPC output");
    }

    #[test]
    #[ignore = "debug reproducer for libcudf row-group pruning on tpch nation"]
    fn repro_row_group_pruning_nation_gpu_scan() {
        let nation_path = std::path::Path::new("/data/tpch/sf1/p16/snappy/nation/nation.1.parquet");
        if !nation_path.exists() {
            eprintln!("skipping reproducer: {} not present", nation_path.display());
            return;
        }

        std::env::set_var("SIRIUS_ENABLE_PARQUET_STATS_PRUNING", "1");

        let engine = sirius_ffi::SiriusEngine::new().expect("SiriusEngine::new");
        engine.execute_sql("LOAD parquet").expect("LOAD parquet");
        engine
            .init_gpu_buffers("1GB", "1GB")
            .expect("init_gpu_buffers");
        engine.set_no_cpu_fallback().expect("set_no_cpu_fallback");

        let sql = "SELECT n_nationkey, n_name \
                   FROM read_parquet('/data/tpch/sf1/p16/snappy/nation/nation.1.parquet') \
                   WHERE n_nationkey < 5";
        let ipc = engine
            .execute_gpu(sql)
            .expect("execute_gpu nation with pruning");
        assert!(!ipc.is_empty(), "expected non-empty IPC output");
    }
}
