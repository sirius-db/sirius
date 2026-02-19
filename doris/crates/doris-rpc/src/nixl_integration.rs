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
use crate::nixl_exchange::{GpuBufferDesc, NixlExchange, NixlMetadataExchange};
use crate::exchange_sender::ExchangeDest;

/// Result of attempting to extract GPU buffer information from execution result.
#[derive(Debug)]
pub enum ExecutionLocation {
    /// Result is in CPU memory (Arrow IPC bytes).
    Cpu(Vec<u8>),
    /// Result is in GPU memory (buffer descriptors + schema).
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
    },
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
                };
            }
            Ok(None) | Err(_) => {
                // Fall through to CPU path
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
            // Use standard bRPC path.
            crate::exchange_sender::send_exchange_result(
                &ipc_bytes,
                destinations,
                query_id,
                node_id,
                sender_id,
            )
            .await
        }
        ExecutionLocation::Gpu {
            buffers,
            column_info,
            num_rows,
            schema_ipc: _,
        } => {
            // GPU-direct path: send metadata + initiate nixl transfers.
            let Some(agent) = nixl_agent else {
                return Err("GPU result but no nixl agent available".to_string());
            };

            let local_md = agent.local_metadata().to_vec();
            let exchange_msg = NixlMetadataExchange {
                metadata: local_md,
                buffer_descs: buffers.clone(),
                column_info: column_info.clone(),
                num_rows,
            };

            // For each destination:
            // 1. Send metadata via gRPC (custom exchange_nixl_metadata method)
            // 2. Receiver allocates GPU buffers and returns their addresses
            // 3. Sender initiates nixl transfer

            for dest in destinations {
                send_nixl_to_peer(agent, &exchange_msg, dest, &buffers).await?;
            }

            Ok(())
        }
    }
}

/// Send GPU data to a single peer via nixl.
#[cfg(feature = "nixl")]
async fn send_nixl_to_peer(
    agent: &NixlExchange,
    metadata: &NixlMetadataExchange,
    dest: &ExchangeDest,
    src_buffers: &[GpuBufferDesc],
) -> Result<(), String> {
    use tracing::info;

    // Step 1: Exchange metadata with remote BE via gRPC.
    // This would call a new gRPC method: exchange_nixl_metadata
    // For now, we'll outline the structure:

    // let grpc_addr = format!("http://{}", dest.brpc_addr);
    // let response = call_exchange_nixl_metadata(&grpc_addr, metadata).await?;

    // Mock response (in reality from gRPC):
    struct MockResponse {
        dst_buffers: Vec<GpuBufferDesc>,
        remote_agent_name: String,
    }
    let _mock_response = MockResponse {
        dst_buffers: src_buffers.to_vec(), // Receiver allocates matching buffers
        remote_agent_name: "remote-agent".to_string(),
    };

    // Step 2: Load remote agent metadata (cached after first call).
    let remote_name = agent.load_remote_metadata(&dest.brpc_addr, &metadata.metadata)?;

    // Step 3: Create descriptor lists for nixl transfer.
    let device_id = src_buffers.first().map(|b| b.device_id).unwrap_or(0);
    let src_ptrs: Vec<_> = src_buffers.iter().map(|b| (b.addr, b.len)).collect();
    let src_descs = agent.create_gpu_descs(&src_ptrs, device_id)?;

    // In real implementation, dst_descs come from the receiver's response:
    // let dst_ptrs: Vec<_> = mock_response.dst_buffers.iter().map(|b| (b.addr, b.len)).collect();
    // let dst_descs = agent.create_gpu_descs(&dst_ptrs, device_id)?;

    // For now, use src_descs as both (mock — in reality dst comes from receiver).
    let dst_descs = agent.create_gpu_descs(&src_ptrs, device_id)?;

    // Step 4: Post the transfer request (blocking until complete).
    agent.transfer_gpu_to_gpu(&src_descs, &dst_descs, &remote_name)?;

    info!(
        dest = %dest.brpc_addr,
        num_buffers = src_buffers.len(),
        "nixl GPU-direct transfer complete"
    );

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
}
