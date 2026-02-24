//! CUDA driver API access via dlopen.
//!
//! Provides GPU memory allocation and context management without a compile-time
//! CUDA dependency. Functions are loaded from `libcuda.so` at runtime.
//!
//! Used by:
//! - `nixl_exchange.rs`: ensure CUDA context before UCX backend creation
//! - `nixl_service.rs`: allocate/free GPU buffers for nixl transfers

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// CUDA driver API functions loaded at runtime via dlopen.
struct CudaDriverFns {
    memcpy_dtoh: unsafe extern "C" fn(*mut u8, u64, usize) -> i32,
    mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    mem_free: unsafe extern "C" fn(u64) -> i32,
    ctx_set_current: unsafe extern "C" fn(u64) -> i32,
    device_primary_ctx_retain: unsafe extern "C" fn(*mut u64, i32) -> i32,
    init: unsafe extern "C" fn(u32) -> i32,
    /// Cached primary context handle for device 0.
    primary_ctx: AtomicU64,
}

fn get_cuda_fns() -> Result<&'static CudaDriverFns, String> {
    static CUDA_FNS: OnceLock<Result<CudaDriverFns, String>> = OnceLock::new();

    CUDA_FNS
        .get_or_init(|| {
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
                device_primary_ctx_retain: unsafe {
                    load_sym(lib, b"cuDevicePrimaryCtxRetain\0")?
                },
                init: unsafe { load_sym(lib, b"cuInit\0")? },
                primary_ctx: AtomicU64::new(0),
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Ensure this thread has a valid CUDA context.
///
/// Must be called before any CUDA driver API operations and **before**
/// nixl UCX backend creation so that UCX can detect GPU memory types.
/// (Without an active CUDA context, UCX treats GPU pointers as host memory,
/// leading to SIGSEGV during transfers.)
pub fn ensure_cuda_context() -> Result<(), String> {
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
        return Err(format!(
            "cuDevicePrimaryCtxRetain failed with CUDA error {rc}"
        ));
    }

    let rc = unsafe { (fns.ctx_set_current)(ctx) };
    if rc != 0 {
        return Err(format!("cuCtxSetCurrent failed with CUDA error {rc}"));
    }

    fns.primary_ctx.store(ctx, Ordering::Relaxed);
    Ok(())
}

/// Copy GPU memory to host using CUDA driver API.
#[allow(dead_code)]
pub fn gpu_to_host(device_addr: usize, len: usize) -> Result<Vec<u8>, String> {
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
pub fn cuda_alloc(len: usize) -> Result<usize, String> {
    ensure_cuda_context()?;
    let fns = get_cuda_fns()?;
    let mut dev_ptr: u64 = 0;
    let rc = unsafe { (fns.mem_alloc)(&mut dev_ptr, len) };
    if rc != 0 {
        return Err(format!(
            "cuMemAlloc_v2({len} bytes) failed with CUDA error {rc}"
        ));
    }
    Ok(dev_ptr as usize)
}

/// Free GPU memory allocated by `cuda_alloc`.
pub fn cuda_free(dev_addr: usize) -> Result<(), String> {
    ensure_cuda_context()?;
    let fns = get_cuda_fns()?;
    let rc = unsafe { (fns.mem_free)(dev_addr as u64) };
    if rc != 0 {
        return Err(format!(
            "cuMemFree_v2(0x{dev_addr:x}) failed with CUDA error {rc}"
        ));
    }
    Ok(())
}
