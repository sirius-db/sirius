//! NIXL metadata exchange gRPC service (separate from PBackendService).
//!
//! Provides GPU-direct exchange coordination via gRPC methods:
//! - exchange_metadata: sender offers buffers, receiver allocates and returns addresses
//! - transfer_complete: sender notifies receiver that nixl transfer is done

use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
#[cfg(feature = "nixl")]
use tracing::{info, instrument, warn};

#[cfg(feature = "nixl")]
use crate::nixl_exchange::NixlExchange;
use crate::exchange_buffer::ExchangeBuffer;
use sirius_ffi::SiriusEngine;

/// NIXL metadata exchange service handler.
#[allow(dead_code)] // fields only read with nixl feature
pub struct NixlMetadataServiceHandler {
    #[cfg(feature = "nixl")]
    nixl_agent: Option<Arc<NixlExchange>>,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
    exchange_buffer: ExchangeBuffer,
}

impl NixlMetadataServiceHandler {
    pub fn new(
        #[cfg(feature = "nixl")]
        nixl_agent: Option<Arc<NixlExchange>>,
        engine: Option<Arc<Mutex<SiriusEngine>>>,
        exchange_buffer: ExchangeBuffer,
    ) -> Self {
        Self {
            #[cfg(feature = "nixl")]
            nixl_agent,
            engine,
            exchange_buffer,
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
                nixl_metadata: vec![],
            }));
        };

        // Step 1: Allocate destination GPU buffers using raw CUDA driver API.
        // We use cuMemAlloc directly (not the RMM pool) because the RMM processing
        // pool may be at capacity after GPU query execution.
        let mut dst_buffers: Vec<PGpuBufferDesc> = Vec::with_capacity(req.src_buffers.len());
        for b in &req.src_buffers {
            info!(
                addr = format_args!("0x{:x}", b.addr),
                len = b.len,
                device_id = b.device_id,
                "exchange_metadata: received src_buffer"
            );
            match cuda_alloc(b.len as usize) {
                Ok(dst_addr) => {
                    info!(
                        dst_addr = format_args!("0x{:x}", dst_addr),
                        len = b.len,
                        "exchange_metadata: allocated dst GPU buffer"
                    );
                    dst_buffers.push(PGpuBufferDesc {
                        addr: dst_addr as u64,
                        len: b.len,
                        device_id: b.device_id,
                    });
                }
                Err(e) => {
                    // Free any already-allocated buffers.
                    for alloc in &dst_buffers {
                        let _ = cuda_free(alloc.addr as usize);
                    }
                    warn!(error = %e, "GPU buffer allocation failed");
                    return Ok(Response::new(PExchangeNixlMetadataResponse {
                        dst_buffers: vec![],
                        remote_agent_name: String::new(),
                        status_code: 1,
                        error_msgs: vec![format!("cuda_alloc: {e}")],
                        nixl_metadata: vec![],
                    }));
                }
            }
        }

        // Step 2: Register allocated dst buffers with receiver's nixl agent.
        let buf_tuples: Vec<_> = dst_buffers
            .iter()
            .map(|b| (b.addr as usize, b.len as usize, b.device_id))
            .collect();
        if let Err(e) = agent.register_gpu_buffers(&buf_tuples) {
            warn!(error = %e, "failed to register dst GPU buffers with nixl");
            return Ok(Response::new(PExchangeNixlMetadataResponse {
                dst_buffers: vec![],
                remote_agent_name: String::new(),
                status_code: 1,
                error_msgs: vec![format!("register_gpu_buffers: {e}")],
                nixl_metadata: vec![],
            }));
        }

        // Step 3: Load sender's metadata (force-load since sender registered new buffers).
        let remote_name = match agent.force_load_remote_metadata(&peer, &req.nixl_metadata) {
            Ok(name) => name,
            Err(e) => {
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

    #[cfg(not(feature = "nixl"))]
    async fn exchange_metadata(
        &self,
        _: Request<doris_proto::nixl::PExchangeNixlMetadataRequest>,
    ) -> Result<Response<doris_proto::nixl::PExchangeNixlMetadataResponse>, Status> {
        Err(Status::unimplemented(
            "nixl feature not enabled on this BE",
        ))
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
    #[cfg(feature = "nixl")]
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

        // Free destination GPU buffers (data was transferred but we use IPC for exchange).
        for buf in &req.dst_buffers {
            if let Err(e) = cuda_free(buf.addr as usize) {
                warn!(error = %e, addr = buf.addr, "failed to free dst GPU buffer");
            }
        }

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

    #[cfg(not(feature = "nixl"))]
    async fn transfer_complete(
        &self,
        _: Request<doris_proto::nixl::PNixlTransferCompleteRequest>,
    ) -> Result<Response<doris_proto::nixl::PNixlTransferCompleteResponse>, Status> {
        Err(Status::unimplemented(
            "nixl feature not enabled on this BE",
        ))
    }
}

/// CUDA driver API functions loaded at runtime via dlopen.
/// All functions are cached in a single OnceLock to avoid multiple dlopen calls.
#[cfg(feature = "nixl")]
struct CudaDriverFns {
    memcpy_dtoh: unsafe extern "C" fn(*mut u8, u64, usize) -> i32,
    mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    mem_free: unsafe extern "C" fn(u64) -> i32,
    ctx_set_current: unsafe extern "C" fn(u64) -> i32,
    device_primary_ctx_retain: unsafe extern "C" fn(*mut u64, i32) -> i32,
    init: unsafe extern "C" fn(u32) -> i32,
    /// Cached primary context handle for device 0.
    primary_ctx: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "nixl")]
fn get_cuda_fns() -> Result<&'static CudaDriverFns, String> {
    use std::sync::OnceLock;
    static CUDA_FNS: OnceLock<Result<CudaDriverFns, String>> = OnceLock::new();

    CUDA_FNS.get_or_init(|| {
        let lib = unsafe {
            libc::dlopen(
                b"libcuda.so\0".as_ptr() as _,
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            )
        };
        if lib.is_null() {
            return Err("failed to dlopen libcuda.so".to_string());
        }

        unsafe fn load_sym<T>(lib: *mut libc::c_void, name: &[u8]) -> Result<T, String> {
            let sym = libc::dlsym(lib, name.as_ptr() as _);
            if sym.is_null() {
                let name_str = std::str::from_utf8(&name[..name.len() - 1]).unwrap_or("?");
                return Err(format!("{name_str} not found in libcuda.so"));
            }
            Ok(std::mem::transmute_copy(&sym))
        }

        Ok(CudaDriverFns {
            memcpy_dtoh: unsafe { load_sym(lib, b"cuMemcpyDtoH_v2\0")? },
            mem_alloc: unsafe { load_sym(lib, b"cuMemAlloc_v2\0")? },
            mem_free: unsafe { load_sym(lib, b"cuMemFree_v2\0")? },
            ctx_set_current: unsafe { load_sym(lib, b"cuCtxSetCurrent\0")? },
            device_primary_ctx_retain: unsafe { load_sym(lib, b"cuDevicePrimaryCtxRetain\0")? },
            init: unsafe { load_sym(lib, b"cuInit\0")? },
            primary_ctx: std::sync::atomic::AtomicU64::new(0),
        })
    }).as_ref().map_err(|e| e.clone())
}

/// Ensure this thread has a valid CUDA context (needed for driver API calls
/// from non-DuckDB threads like gRPC handlers).
#[cfg(feature = "nixl")]
fn ensure_cuda_context() -> Result<(), String> {
    use std::sync::atomic::Ordering;
    let fns = get_cuda_fns()?;

    // Fast path: context already cached.
    let cached = fns.primary_ctx.load(Ordering::Relaxed);
    if cached != 0 {
        let rc = unsafe { (fns.ctx_set_current)(cached) };
        if rc == 0 {
            return Ok(());
        }
        // Context might have been destroyed, fall through to re-acquire.
    }

    // Slow path: initialize CUDA and get primary context.
    let rc = unsafe { (fns.init)(0) };
    if rc != 0 {
        return Err(format!("cuInit failed with CUDA error {rc}"));
    }

    let mut ctx: u64 = 0;
    let rc = unsafe { (fns.device_primary_ctx_retain)(&mut ctx, 0) };
    if rc != 0 {
        return Err(format!("cuDevicePrimaryCtxRetain failed with CUDA error {rc}"));
    }

    let rc = unsafe { (fns.ctx_set_current)(ctx) };
    if rc != 0 {
        return Err(format!("cuCtxSetCurrent failed with CUDA error {rc}"));
    }

    fns.primary_ctx.store(ctx, Ordering::Relaxed);
    Ok(())
}

/// Copy GPU memory to host using CUDA driver API.
/// Currently unused — will be needed for direct GPU table registration.
#[cfg(feature = "nixl")]
#[allow(dead_code)]
fn gpu_to_host(device_addr: usize, len: usize) -> Result<Vec<u8>, String> {
    ensure_cuda_context()?;
    let fns = get_cuda_fns()?;
    let mut buf = vec![0u8; len];
    let rc = unsafe { (fns.memcpy_dtoh)(buf.as_mut_ptr(), device_addr as u64, len) };
    if rc != 0 {
        return Err(format!("cuMemcpyDtoH_v2 failed with CUDA error {rc}"));
    }
    Ok(buf)
}

/// Allocate GPU memory using raw CUDA driver API (bypasses RMM pool).
#[cfg(feature = "nixl")]
fn cuda_alloc(len: usize) -> Result<usize, String> {
    ensure_cuda_context()?;
    let fns = get_cuda_fns()?;
    let mut dev_ptr: u64 = 0;
    let rc = unsafe { (fns.mem_alloc)(&mut dev_ptr, len) };
    if rc != 0 {
        return Err(format!("cuMemAlloc_v2({len} bytes) failed with CUDA error {rc}"));
    }
    Ok(dev_ptr as usize)
}

/// Free GPU memory allocated by cuda_alloc.
#[cfg(feature = "nixl")]
fn cuda_free(dev_addr: usize) -> Result<(), String> {
    ensure_cuda_context()?;
    let fns = get_cuda_fns()?;
    let rc = unsafe { (fns.mem_free)(dev_addr as u64) };
    if rc != 0 {
        return Err(format!("cuMemFree_v2(0x{dev_addr:x}) failed with CUDA error {rc}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange_buf() -> ExchangeBuffer {
        ExchangeBuffer::new()
    }

    #[tokio::test]
    async fn test_service_creation_no_engine() {
        #[cfg(feature = "nixl")]
        let service = NixlMetadataServiceHandler::new(None, None, exchange_buf());
        #[cfg(not(feature = "nixl"))]
        let service = NixlMetadataServiceHandler::new(None, exchange_buf());

        let _ = service;
    }

    #[tokio::test]
    async fn test_service_creation_with_engine() {
        let engine = sirius_ffi::SiriusEngine::new().ok().map(|e| Arc::new(Mutex::new(e)));

        #[cfg(feature = "nixl")]
        let service = NixlMetadataServiceHandler::new(None, engine, exchange_buf());
        #[cfg(not(feature = "nixl"))]
        let service = NixlMetadataServiceHandler::new(engine, exchange_buf());

        let _ = service;
    }

    #[tokio::test]
    #[cfg(not(feature = "nixl"))]
    async fn test_exchange_metadata_without_nixl() {
        use doris_proto::nixl::NixlMetadataService;

        let service = NixlMetadataServiceHandler::new(None, exchange_buf());
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
        let service = NixlMetadataServiceHandler::new(engine, exchange_buf());
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

    #[tokio::test]
    #[cfg(not(feature = "nixl"))]
    async fn test_transfer_complete_without_nixl() {
        use doris_proto::nixl::NixlMetadataService;

        let service = NixlMetadataServiceHandler::new(None, exchange_buf());
        let request = Request::new(doris_proto::nixl::PNixlTransferCompleteRequest {
            query_id_hi: vec![],
            query_id_lo: vec![],
            node_id: 0,
            dst_buffers: vec![],
            columns: vec![],
            num_rows: 0,
            sender_id: 0,
            arrow_ipc_data: vec![],
        });

        let result = service.transfer_complete(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }
}
