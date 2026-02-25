//! Integration layer between nixl GPU-direct exchange and query execution.
//!
//! Handles:
//! - Detecting GPU-resident results (from sirius-ffi engine)
//! - Extracting GPU buffer pointers for nixl transfer
//! - Coordinating metadata exchange and transfer
//! - Fallback to bRPC when GPU-direct unavailable

#[cfg(feature = "nixl")]
use std::sync::Arc;

#[cfg(feature = "nixl")]
use crate::nixl_exchange::{GpuBufferDesc, NixlExchange};
use crate::exchange_sender::ExchangeDest;

/// Result of attempting to extract GPU buffer information from execution result.
#[derive(Debug)]
pub enum ExecutionLocation {
    /// Result is in CPU memory (Arrow IPC bytes).
    Cpu(Vec<u8>),
    /// Result is in GPU memory (buffer descriptors + schema).
    /// Also carries IPC bytes as fallback for non-exchange paths.
    #[cfg(feature = "nixl")]
    Gpu {
        /// GPU buffer descriptors (addr, len, device_id).
        buffers: Vec<GpuBufferDesc>,
        /// Column names and type IDs.
        column_info: Vec<(String, i32)>,
        /// Number of rows.
        num_rows: u32,
        /// Arrow IPC schema bytes (for receiver reconstruction).
        schema_ipc: Vec<u8>,
        /// Arrow IPC bytes (fallback for store/fetch_data path).
        ipc_bytes: Vec<u8>,
    },
}

impl ExecutionLocation {
    /// Extract IPC bytes, consuming self.
    pub fn into_ipc_bytes(self) -> Vec<u8> {
        match self {
            Self::Cpu(bytes) => bytes,
            #[cfg(feature = "nixl")]
            Self::Gpu { ipc_bytes, .. } => ipc_bytes,
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
    #[cfg(feature = "nixl")]
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
                    num_rows: gpu_info.num_rows,
                    schema_ipc: gpu_info.schema_ipc,
                    ipc_bytes,
                };
            }
            Ok(None) => {
                tracing::debug!("detect_execution_location: no GPU buffers (CPU execution)");
            }
            Err(e) => {
                tracing::debug!(error = %e, "detect_execution_location: get_last_gpu_result_buffers failed");
            }
        }
    }

    ExecutionLocation::Cpu(ipc_bytes)
}

/// Send exchange result using nixl GPU-direct if available, otherwise bRPC.
#[cfg(feature = "nixl")]
pub async fn send_exchange_with_nixl(
    nixl_agent: Option<&Arc<NixlExchange>>,
    location: ExecutionLocation,
    destinations: &[ExchangeDest],
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
) -> Result<(), String> {
    match location {
        ExecutionLocation::Cpu(ipc_bytes) => {
            crate::exchange_sender::send_exchange_result(
                &ipc_bytes, destinations, query_id, node_id, sender_id,
            )
            .await
        }
        ExecutionLocation::Gpu {
            buffers,
            column_info,
            num_rows,
            schema_ipc: _,
            ipc_bytes,
        } => {
            let Some(agent) = nixl_agent else {
                tracing::warn!("GPU result but no nixl agent, falling back to bRPC");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes, destinations, query_id, node_id, sender_id,
                ).await;
            };

            // Fast-path: skip nixl if a previous registration already detected
            // that UCX treats GPU memory as host memory (would SIGSEGV).
            if !agent.gpu_transfer_enabled() {
                tracing::info!("UCX lacks CUDA support for GPU memory, using bRPC for exchange");
                return crate::exchange_sender::send_exchange_result(
                    &ipc_bytes, destinations, query_id, node_id, sender_id,
                ).await;
            }

            // Try nixl GPU-direct for each destination, fall back to bRPC on failure.
            for dest in destinations {
                if let Err(e) = send_nixl_to_peer(
                    agent, &buffers, &column_info, num_rows,
                    &ipc_bytes, dest, query_id, node_id, sender_id,
                ).await {
                    tracing::warn!(
                        error = %e,
                        dest = %dest.brpc_addr,
                        "nixl transfer failed, falling back to bRPC"
                    );
                    crate::exchange_sender::send_exchange_result(
                        &ipc_bytes, destinations, query_id, node_id, sender_id,
                    ).await?;
                    return Ok(());
                }
            }

            Ok(())
        }
    }
}

/// Send GPU data to a single peer via nixl.
///
/// Full flow: register buffers → exchange metadata → load peer metadata →
/// transfer → notify receiver of completion.
#[cfg(feature = "nixl")]
async fn send_nixl_to_peer(
    agent: &Arc<NixlExchange>,
    src_buffers: &[GpuBufferDesc],
    column_info: &[(String, i32)],
    num_rows: u32,
    ipc_bytes: &[u8],
    dest: &ExchangeDest,
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
) -> Result<(), String> {
    use doris_proto::nixl::{
        NixlMetadataServiceClient, PColumnInfo, PExchangeNixlMetadataRequest,
        PGpuBufferDesc, PNixlTransferCompleteRequest,
    };
    use tracing::info;

    let grpc_addr = format!("http://{}", dest.brpc_addr);

    // Step 1: Register sender's GPU result buffers with nixl agent.
    // Held for the duration of the transfer; dropped at function return.
    let buf_tuples: Vec<_> = src_buffers.iter().map(|b| (b.addr, b.len, b.device_id)).collect();
    let _src_registrations = agent.register_gpu_buffers(&buf_tuples)?;

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
            .map(|(name, type_id)| PColumnInfo {
                name: name.clone(),
                type_id: *type_id,
            })
            .collect(),
        num_rows,
        query_id_hi: query_id.0.to_le_bytes().to_vec(),
        query_id_lo: query_id.1.to_le_bytes().to_vec(),
        node_id,
    };

    let mut client = NixlMetadataServiceClient::connect(grpc_addr.clone())
        .await
        .map_err(|e| format!("connect to {grpc_addr}: {e}"))?;

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
            "buffer count mismatch: src={}, dst={}",
            src_buffers.len(),
            response.dst_buffers.len()
        ));
    }

    // Step 4: Load receiver's metadata (includes their registered dst buffers).
    let remote_name = agent.force_load_remote_metadata(
        &dest.brpc_addr,
        &response.nixl_metadata,
    )?;

    // Step 5: Create descriptor lists and execute transfer.
    // The transfer uses a blocking poll loop, so run it on a blocking thread
    // to avoid stalling the tokio runtime (which needs to handle concurrent
    // exchange data, heartbeats, etc.).
    {
        let agent = agent.clone();
        let src_ptrs: Vec<_> = src_buffers.iter().map(|b| (b.addr, b.len)).collect();
        let dst_ptrs: Vec<_> = response
            .dst_buffers
            .iter()
            .map(|b| (b.addr as usize, b.len as usize))
            .collect();
        let device_id = src_buffers.first().map(|b| b.device_id).unwrap_or(0);
        let remote = remote_name.clone();

        tokio::task::spawn_blocking(move || {
            let src_descs = agent.create_gpu_descs(&src_ptrs, device_id)?;
            let dst_descs = agent.create_gpu_descs(&dst_ptrs, device_id)?;
            agent.transfer_gpu_to_gpu(&src_descs, &dst_descs, &remote)
        })
        .await
        .map_err(|e| format!("transfer spawn_blocking panicked: {e}"))??;
    }

    info!(
        dest = %dest.brpc_addr,
        num_buffers = src_buffers.len(),
        "nixl GPU-direct transfer complete"
    );

    // Step 6: Notify receiver that transfer is complete.
    // Include Arrow IPC bytes so receiver can construct a proper PBlock
    // using the same arrow_ipc_to_pblock path as bRPC (avoiding type ID mismatches
    // between DuckDB LogicalTypeId and Doris PGenericType::TypeId).
    let complete_req = PNixlTransferCompleteRequest {
        query_id_hi: query_id.0.to_le_bytes().to_vec(),
        query_id_lo: query_id.1.to_le_bytes().to_vec(),
        node_id,
        dst_buffers: response.dst_buffers,
        columns: column_info
            .iter()
            .map(|(name, type_id)| PColumnInfo {
                name: name.clone(),
                type_id: *type_id,
            })
            .collect(),
        num_rows,
        sender_id,
        arrow_ipc_data: ipc_bytes.to_vec(),
    };

    let mut client2 = NixlMetadataServiceClient::connect(grpc_addr.clone())
        .await
        .map_err(|e| format!("connect for transfer_complete: {e}"))?;

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

/// Non-nixl version: always use bRPC.
#[cfg(not(feature = "nixl"))]
pub async fn send_exchange_with_nixl(
    _nixl_agent: Option<&()>,
    location: ExecutionLocation,
    destinations: &[ExchangeDest],
    query_id: (i64, i64),
    node_id: i32,
    sender_id: i32,
) -> Result<(), String> {
    match location {
        ExecutionLocation::Cpu(ipc_bytes) => {
            crate::exchange_sender::send_exchange_result(
                &ipc_bytes,
                destinations,
                query_id,
                node_id,
                sender_id,
            )
            .await
        }
    }
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
            #[cfg(feature = "nixl")]
            ExecutionLocation::Gpu { .. } => {
                // Could happen if engine has GPU result
            }
        }
    }

    #[cfg(feature = "nixl")]
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
            #[cfg(feature = "nixl")]
            ExecutionLocation::Gpu { .. } => panic!("expected Cpu variant"),
        }
    }

    #[tokio::test]
    async fn test_send_exchange_cpu_location_wraps_brpc() {
        // Verify that Cpu location delegates to bRPC sender.
        // We can't easily test the actual send (needs TCP), but we can verify
        // the wrapping logic: Cpu variant extracts ipc_bytes for send_exchange_result.
        let ipc = vec![0xAA, 0xBB, 0xCC];
        let location = ExecutionLocation::Cpu(ipc);

        // With invalid IPC bytes and no destinations, the bRPC sender will still
        // try to parse. We just verify the function is callable and the type system works.
        let result = send_exchange_with_nixl(
            #[cfg(feature = "nixl")]
            None,
            #[cfg(not(feature = "nixl"))]
            None,
            location,
            &[], // no destinations — but arrow_ipc_to_pblock still runs
            (1, 2),
            0,
            0,
        )
        .await;
        // Will fail because IPC bytes are invalid, which is expected.
        assert!(result.is_err());
    }
}
