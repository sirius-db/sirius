//! NIXL GPU-direct buffer transfer validation.
//!
//! Tests:
//! 1. DRAM-to-DRAM transfer (baseline, no GPU needed)
//! 2. VRAM-to-VRAM transfer (GPU-direct, the real deal)

use nixl_sys::{
    Agent, MemType, MemoryRegion, NixlDescriptor, NixlRegistration, OptArgs, RegistrationHandle,
    SystemStorage, XferDescList, XferOp,
};
use std::time::{Duration, Instant};

const AGENT_SRC: &str = "sirius-be-src";
const AGENT_DST: &str = "sirius-be-dst";

fn main() {
    println!("=== NIXL Buffer Transfer Validation ===\n");

    // Initialize CUDA with primary context so UCX sees GPU from the start.
    let cuda_ok = unsafe { cuInit(0) } == 0;
    if cuda_ok {
        let mut device: i32 = 0;
        let mut ctx: usize = 0;
        unsafe {
            cuda_check(cuDeviceGet(&mut device, 0), "cuDeviceGet").unwrap();
            cuda_check(
                cuDevicePrimaryCtxRetain(&mut ctx, device),
                "cuDevicePrimaryCtxRetain",
            )
            .unwrap();
            cuda_check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent").unwrap();
        }
        println!("CUDA initialized with primary context (device {device})\n");
    } else {
        println!("CUDA not available, VRAM test will be skipped\n");
    }

    if let Err(e) = test_dram_transfer() {
        eprintln!("DRAM transfer FAILED: {e}");
        std::process::exit(1);
    }

    println!();

    if !cuda_ok {
        println!("VRAM transfer SKIPPED (no CUDA)");
    } else {
        match test_vram_transfer() {
            Ok(()) => {}
            Err(e) => {
                eprintln!("VRAM transfer FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    println!("\n=== All tests passed ===");
}

fn test_dram_transfer() -> Result<(), String> {
    println!("--- Test: DRAM-to-DRAM transfer ---");

    let agent_src = Agent::new(AGENT_SRC).map_err(|e| format!("create src agent: {e}"))?;
    let agent_dst = Agent::new(AGENT_DST).map_err(|e| format!("create dst agent: {e}"))?;
    println!("  created agents: {AGENT_SRC}, {AGENT_DST}");

    let plugins = agent_src
        .get_available_plugins()
        .map_err(|e| format!("get_available_plugins: {e}"))?;
    print!("  available plugins:");
    for p in plugins.iter() {
        if let Ok(name) = p {
            print!(" {name}");
        }
    }
    println!();

    let (_, params_src) = agent_src
        .get_plugin_params("UCX")
        .map_err(|e| format!("get_plugin_params src: {e}"))?;
    let (_, params_dst) = agent_dst
        .get_plugin_params("UCX")
        .map_err(|e| format!("get_plugin_params dst: {e}"))?;

    let backend_src = agent_src
        .create_backend("UCX", &params_src)
        .map_err(|e| format!("create_backend src: {e}"))?;
    let backend_dst = agent_dst
        .create_backend("UCX", &params_dst)
        .map_err(|e| format!("create_backend dst: {e}"))?;
    println!("  created UCX backends");

    let buf_size = 4096;
    let mut storage_src = SystemStorage::new(buf_size).map_err(|e| format!("alloc src: {e}"))?;
    let mut storage_dst = SystemStorage::new(buf_size).map_err(|e| format!("alloc dst: {e}"))?;

    storage_src.memset(0xAA);
    storage_dst.memset(0x00);

    let mut opt_src = OptArgs::new().map_err(|e| format!("opt_args src: {e}"))?;
    opt_src
        .add_backend(&backend_src)
        .map_err(|e| format!("add_backend src: {e}"))?;
    let mut opt_dst = OptArgs::new().map_err(|e| format!("opt_args dst: {e}"))?;
    opt_dst
        .add_backend(&backend_dst)
        .map_err(|e| format!("add_backend dst: {e}"))?;

    storage_src
        .register(&agent_src, Some(&opt_src))
        .map_err(|e| format!("register src: {e}"))?;
    storage_dst
        .register(&agent_dst, Some(&opt_dst))
        .map_err(|e| format!("register dst: {e}"))?;
    println!("  registered {buf_size} bytes on each agent");

    let md_src = agent_src
        .get_local_md()
        .map_err(|e| format!("get_local_md src: {e}"))?;
    let md_dst = agent_dst
        .get_local_md()
        .map_err(|e| format!("get_local_md dst: {e}"))?;

    agent_src
        .load_remote_md(&md_dst)
        .map_err(|e| format!("load_remote_md on src: {e}"))?;
    agent_dst
        .load_remote_md(&md_src)
        .map_err(|e| format!("load_remote_md on dst: {e}"))?;
    println!(
        "  metadata exchanged (src={} bytes, dst={} bytes)",
        md_src.len(),
        md_dst.len()
    );

    assert_eq!(storage_src.as_slice()[0], 0xAA);
    assert_eq!(storage_dst.as_slice()[0], 0x00);

    let xfer_size = 256;
    let mut src_desc = XferDescList::new(MemType::Dram).map_err(|e| format!("src desc: {e}"))?;
    let src_ptr = unsafe { storage_src.as_ptr() as usize };
    src_desc.add_desc(src_ptr, xfer_size, 0);

    let mut dst_desc = XferDescList::new(MemType::Dram).map_err(|e| format!("dst desc: {e}"))?;
    let dst_ptr = unsafe { storage_dst.as_ptr() as usize };
    dst_desc.add_desc(dst_ptr, xfer_size, 0);

    let xfer_req = agent_src
        .create_xfer_req(XferOp::Write, &src_desc, &dst_desc, AGENT_DST, None)
        .map_err(|e| format!("create_xfer_req: {e}"))?;
    agent_src
        .post_xfer_req(&xfer_req, None)
        .map_err(|e| format!("post_xfer_req: {e}"))?;
    println!("  posted transfer: {xfer_size} bytes");

    let (elapsed, _) = poll_transfer(&agent_src, &xfer_req)?;
    println!("  transfer complete in {elapsed:?}");

    let dst_data = storage_dst.as_slice();
    for i in 0..xfer_size {
        if dst_data[i] != 0xAA {
            return Err(format!("byte {i}: expected 0xAA, got 0x{:02X}", dst_data[i]));
        }
    }
    if buf_size > xfer_size && dst_data[xfer_size] != 0x00 {
        return Err(format!(
            "byte {xfer_size}: expected 0x00, got 0x{:02X}",
            dst_data[xfer_size]
        ));
    }

    println!("  PASSED: {xfer_size} bytes transferred correctly");
    Ok(())
}

// --- CUDA driver API FFI (minimal, for GPU memory allocation) ---

#[allow(non_camel_case_types)]
type CUdeviceptr = u64;
#[allow(non_camel_case_types)]
type CUresult = i32;

extern "C" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> CUresult;
    fn cuDevicePrimaryCtxRetain(pctx: *mut usize, dev: i32) -> CUresult;
    fn cuCtxSetCurrent(ctx: usize) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const u8, bytesize: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut u8, src: CUdeviceptr, bytesize: usize) -> CUresult;
}

fn cuda_check(result: CUresult, op: &str) -> Result<(), String> {
    if result != 0 {
        Err(format!("{op} failed with CUDA error {result}"))
    } else {
        Ok(())
    }
}

/// GPU memory wrapper that implements nixl's NixlDescriptor traits for VRAM registration.
#[derive(Debug)]
struct GpuStorage {
    ptr: CUdeviceptr,
    size: usize,
    device_id: u64,
    handle: Option<RegistrationHandle>,
}

impl MemoryRegion for GpuStorage {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
    fn size(&self) -> usize {
        self.size
    }
}

impl NixlDescriptor for GpuStorage {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }
    fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl NixlRegistration for GpuStorage {
    fn register(
        &mut self,
        agent: &Agent,
        opt_args: Option<&OptArgs>,
    ) -> Result<(), nixl_sys::NixlError> {
        let handle = agent.register_memory(self, opt_args)?;
        self.handle = Some(handle);
        Ok(())
    }
}

fn test_vram_transfer() -> Result<(), String> {
    println!("--- Test: VRAM-to-VRAM transfer ---");

    let buf_size: usize = 1024 * 1024; // 1 MB
    let xfer_size: usize = buf_size;

    // Allocate GPU memory.
    let mut gpu_src: CUdeviceptr = 0;
    let mut gpu_dst: CUdeviceptr = 0;
    unsafe {
        cuda_check(cuMemAlloc_v2(&mut gpu_src, buf_size), "cuMemAlloc src")?;
        cuda_check(cuMemAlloc_v2(&mut gpu_dst, buf_size), "cuMemAlloc dst")?;
    }
    println!(
        "  allocated GPU memory: src=0x{gpu_src:x}, dst=0x{gpu_dst:x} ({} bytes each)",
        buf_size
    );

    // Fill source GPU buffer with pattern via host staging.
    let host_pattern: Vec<u8> = vec![0xBE; buf_size];
    unsafe {
        cuda_check(
            cuMemcpyHtoD_v2(gpu_src, host_pattern.as_ptr(), buf_size),
            "cuMemcpyHtoD src",
        )?;
    }
    // Clear destination.
    let host_zeros: Vec<u8> = vec![0x00; buf_size];
    unsafe {
        cuda_check(
            cuMemcpyHtoD_v2(gpu_dst, host_zeros.as_ptr(), buf_size),
            "cuMemcpyHtoD dst",
        )?;
    }
    println!("  GPU buffers initialized (src=0xBE, dst=0x00)");

    // Create nixl agents.
    let agent_src =
        Agent::new("vram-src").map_err(|e| format!("create src agent: {e}"))?;
    let agent_dst =
        Agent::new("vram-dst").map_err(|e| format!("create dst agent: {e}"))?;

    let (_, params_src) = agent_src
        .get_plugin_params("UCX")
        .map_err(|e| format!("get_plugin_params src: {e}"))?;
    let (_, params_dst) = agent_dst
        .get_plugin_params("UCX")
        .map_err(|e| format!("get_plugin_params dst: {e}"))?;

    let backend_src = agent_src
        .create_backend("UCX", &params_src)
        .map_err(|e| format!("create_backend src: {e}"))?;
    let backend_dst = agent_dst
        .create_backend("UCX", &params_dst)
        .map_err(|e| format!("create_backend dst: {e}"))?;
    println!("  created nixl agents with UCX backends");

    // Register GPU memory with nixl via GpuStorage wrapper.
    let mut gpu_storage_src = GpuStorage {
        ptr: gpu_src,
        size: buf_size,
        device_id: 0,
        handle: None,
    };
    let mut gpu_storage_dst = GpuStorage {
        ptr: gpu_dst,
        size: buf_size,
        device_id: 0,
        handle: None,
    };

    let mut opt_src = OptArgs::new().map_err(|e| format!("opt_args src: {e}"))?;
    opt_src
        .add_backend(&backend_src)
        .map_err(|e| format!("add_backend src: {e}"))?;
    let mut opt_dst = OptArgs::new().map_err(|e| format!("opt_args dst: {e}"))?;
    opt_dst
        .add_backend(&backend_dst)
        .map_err(|e| format!("add_backend dst: {e}"))?;

    gpu_storage_src
        .register(&agent_src, Some(&opt_src))
        .map_err(|e| format!("register src VRAM: {e}"))?;
    gpu_storage_dst
        .register(&agent_dst, Some(&opt_dst))
        .map_err(|e| format!("register dst VRAM: {e}"))?;
    println!("  registered GPU memory with nixl");

    // Exchange metadata.
    let md_src = agent_src
        .get_local_md()
        .map_err(|e| format!("get_local_md src: {e}"))?;
    let md_dst = agent_dst
        .get_local_md()
        .map_err(|e| format!("get_local_md dst: {e}"))?;
    agent_src
        .load_remote_md(&md_dst)
        .map_err(|e| format!("load_remote_md on src: {e}"))?;
    agent_dst
        .load_remote_md(&md_src)
        .map_err(|e| format!("load_remote_md on dst: {e}"))?;
    println!("  metadata exchanged");

    // Create transfer descriptors.
    let mut src_desc = XferDescList::new(MemType::Vram).map_err(|e| format!("src desc: {e}"))?;
    src_desc.add_desc(gpu_src as usize, xfer_size, 0);

    let mut dst_desc = XferDescList::new(MemType::Vram).map_err(|e| format!("dst desc: {e}"))?;
    dst_desc.add_desc(gpu_dst as usize, xfer_size, 0);

    // Execute transfer.
    let xfer_req = agent_src
        .create_xfer_req(XferOp::Write, &src_desc, &dst_desc, "vram-dst", None)
        .map_err(|e| format!("create_xfer_req: {e}"))?;
    agent_src
        .post_xfer_req(&xfer_req, None)
        .map_err(|e| format!("post_xfer_req: {e}"))?;
    println!("  posted GPU transfer: {} bytes", xfer_size);

    let (elapsed, _) = poll_transfer(&agent_src, &xfer_req)?;
    let throughput_gbps =
        (xfer_size as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!("  transfer complete in {elapsed:?} ({throughput_gbps:.1} GB/s)");

    // Verify: copy dst GPU buffer back to host and check.
    let mut host_verify = vec![0u8; buf_size];
    unsafe {
        cuda_check(
            cuMemcpyDtoH_v2(host_verify.as_mut_ptr(), gpu_dst, buf_size),
            "cuMemcpyDtoH verify",
        )?;
    }

    for i in 0..xfer_size {
        if host_verify[i] != 0xBE {
            return Err(format!(
                "byte {i}: expected 0xBE, got 0x{:02X}",
                host_verify[i]
            ));
        }
    }

    // Cleanup.
    unsafe {
        cuMemFree_v2(gpu_src);
        cuMemFree_v2(gpu_dst);
    }

    println!(
        "  PASSED: {} bytes GPU-to-GPU transfer verified (0xBE pattern)",
        xfer_size
    );
    Ok(())
}

fn poll_transfer(
    agent: &Agent,
    req: &nixl_sys::XferRequest,
) -> Result<(Duration, ()), String> {
    let start = Instant::now();
    let timeout = Duration::from_secs(10);
    loop {
        let status = agent
            .get_xfer_status(req)
            .map_err(|e| format!("get_xfer_status: {e}"))?;
        if status.is_success() {
            return Ok((start.elapsed(), ()));
        }
        if start.elapsed() > timeout {
            return Err("transfer timed out after 10s".to_string());
        }
        std::thread::yield_now();
    }
}
