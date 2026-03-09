//! CUDA driver API access via cudarc.
//!
//! Provides GPU memory allocation and context management using cudarc's
//! dynamic-loading feature (runtime dlopen of libcuda.so, no build-time CUDA).
//!
//! Used by:
//! - `nixl_exchange.rs`: ensure CUDA context before UCX backend creation
//! - `nixl_service.rs`: allocate/free GPU buffers for nixl transfers

use std::sync::OnceLock;

use cudarc::driver::{result, sys, sys::CUcontext};

/// Wrapper for CUcontext that is Send + Sync.
///
/// SAFETY: CUDA primary contexts are process-global and thread-safe.
/// The context handle can safely be shared across threads.
struct SendCUcontext(CUcontext);
unsafe impl Send for SendCUcontext {}
unsafe impl Sync for SendCUcontext {}

/// Get or initialize the CUDA primary context for device 0.
fn get_primary_ctx() -> Result<CUcontext, String> {
    static CTX: OnceLock<Result<SendCUcontext, String>> = OnceLock::new();

    CTX.get_or_init(|| {
        result::init().map_err(|e| format!("cuInit: {e}"))?;
        let ctx = unsafe { result::primary_ctx::retain(0) }
            .map_err(|e| format!("cuDevicePrimaryCtxRetain: {e}"))?;
        Ok(SendCUcontext(ctx))
    })
    .as_ref()
    .map(|s| s.0)
    .map_err(|e| e.clone())
}

/// Ensure this thread has a valid CUDA context.
///
/// Must be called before any CUDA driver API operations and **before**
/// nixl UCX backend creation so that UCX can detect GPU memory types.
/// (Without an active CUDA context, UCX treats GPU pointers as host memory,
/// leading to SIGSEGV during transfers.)
pub fn ensure_cuda_context() -> Result<(), String> {
    let ctx = get_primary_ctx()?;
    unsafe { result::ctx::set_current(ctx) }.map_err(|e| format!("cuCtxSetCurrent: {e}"))
}

/// Copy GPU memory to host using CUDA driver API.
#[allow(dead_code)]
pub fn gpu_to_host(device_addr: usize, len: usize) -> Result<Vec<u8>, String> {
    ensure_cuda_context()?;
    let mut buf = vec![0u8; len];
    unsafe { result::memcpy_dtoh_sync(&mut buf, device_addr as u64) }
        .map_err(|e| format!("cuMemcpyDtoH: {e}"))?;
    Ok(buf)
}

/// Allocate GPU memory using raw CUDA driver API (bypasses RMM pool).
pub fn cuda_alloc(len: usize) -> Result<usize, String> {
    ensure_cuda_context()?;
    let dev_ptr =
        unsafe { result::malloc_sync(len) }.map_err(|e| format!("cuMemAlloc({len} bytes): {e}"))?;
    Ok(dev_ptr as usize)
}

/// Free GPU memory allocated by `cuda_alloc`.
pub fn cuda_free(dev_addr: usize) -> Result<(), String> {
    ensure_cuda_context()?;
    unsafe { result::free_sync(dev_addr as u64) }
        .map_err(|e| format!("cuMemFree(0x{dev_addr:x}): {e}"))
}

/// Copy GPU memory device-to-device using CUDA driver API.
pub fn cuda_memcpy_dtod(dst: usize, src: usize, len: usize) -> Result<(), String> {
    ensure_cuda_context()?;
    unsafe { result::memcpy_dtod_sync(dst as u64, src as u64, len) }
        .map_err(|e| format!("cuMemcpyDtoD(dst=0x{dst:x}, src=0x{src:x}, len={len}): {e}"))
}

/// Query the base address and size of the containing allocation for a device pointer.
///
/// For RMM pool sub-allocations, this returns the base of the `cudaMalloc` allocation
/// that backs the pool — i.e., the entire pool region. This can be registered with
/// nixl once, allowing all sub-allocations within it to be used directly for transfers.
pub fn cuda_mem_get_address_range(dev_addr: usize) -> Result<(usize, usize), String> {
    ensure_cuda_context()?;
    let mut base: u64 = 0;
    let mut size: usize = 0;
    unsafe {
        sys::cuMemGetAddressRange_v2(&mut base, &mut size, dev_addr as u64)
    }
    .result()
    .map_err(|e| format!("cuMemGetAddressRange(0x{dev_addr:x}): {e}"))?;
    Ok((base as usize, size))
}

/// Query the CUDA memory type of a device pointer.
///
/// Returns `CU_MEMORYTYPE_DEVICE` (2) for GPU memory, `CU_MEMORYTYPE_HOST` (1) for host.
/// Useful for verifying whether UCX will recognize a pointer as GPU memory.
pub fn cuda_pointer_get_memory_type(dev_addr: usize) -> Result<u32, String> {
    ensure_cuda_context()?;
    let mut mem_type: u32 = 0;
    unsafe {
        sys::cuPointerGetAttribute(
            &mut mem_type as *mut u32 as *mut std::ffi::c_void,
            sys::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            dev_addr as u64,
        )
    }
    .result()
    .map_err(|e| format!("cuPointerGetAttribute(MEMORY_TYPE, 0x{dev_addr:x}): {e}"))?;
    Ok(mem_type)
}
