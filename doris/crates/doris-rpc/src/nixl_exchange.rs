//! NIXL (NVIDIA Inference Xfer Library) GPU-direct exchange transport.
//!
//! Provides GPU-to-GPU memory transfer between Sirius BE instances,
//! bypassing CPU serialization. Falls back to bRPC when nixl is unavailable.
//!
//! # Architecture
//!
//! Each BE process creates a single `NixlAgent` at startup:
//! ```text
//! startup:
//!   agent = nixl::Agent::new("sirius-be-{host}")
//!   backend = agent.create_backend("UCX", params)
//!   local_md = agent.get_local_md()
//! ```
//!
//! Metadata exchange happens via a gRPC side-channel:
//! - Sender calls `exchange_nixl_metadata` on receiver's gRPC endpoint
//! - Both sides load each other's metadata
//! - Cached per peer address (invalidated on BE restart)
//!
//! Transfer path (GPU → GPU):
//! ```text
//! 1. Fragment result in GPU memory (Arrow buffers)
//! 2. Create XferDescList for source GPU buffers
//! 3. Send descriptor info to receiver via gRPC
//! 4. Receiver allocates dest GPU buffers, creates XferDescList
//! 5. Receiver calls create_xfer_req + post_xfer_req
//! 6. Poll get_xfer_status until complete
//! 7. Data in receiver GPU memory → register directly
//! ```

use std::collections::HashMap;
use std::sync::Mutex;

use nixl_sys::{
    Agent, AgentConfig, Backend, MemType, MemoryRegion, NixlDescriptor, OptArgs,
    RegistrationHandle, XferDescList, XferOp, XferStatus,
};
use tracing::{info, warn};

use crate::cuda_driver::{cuda_alloc, cuda_free, ensure_cuda_context};
use crate::gpu_staging_buffer::GpuStagingBuffer;

/// Wrapper for a GPU memory region that implements nixl registration traits.
#[derive(Debug)]
struct GpuBufferWrapper {
    addr: usize,
    len: usize,
    device_id: u64,
}

// SAFETY: GPU pointer addresses can be sent across threads.
unsafe impl Send for GpuBufferWrapper {}
unsafe impl Sync for GpuBufferWrapper {}

impl MemoryRegion for GpuBufferWrapper {
    fn size(&self) -> usize {
        self.len
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        self.addr as *const u8
    }
}

impl NixlDescriptor for GpuBufferWrapper {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }
    fn device_id(&self) -> u64 {
        self.device_id
    }
}

/// RAII wrapper for a GPU buffer registered with nixl.
///
/// Ensures correct cleanup order on drop:
/// 1. Deregister from nixl (while GPU memory is still valid)
/// 2. Free GPU memory (only if this buffer owns the allocation)
///
/// Two ownership modes:
/// - `owns_memory: true` — receiver path: we allocated this buffer, free on drop
/// - `owns_memory: false` — sender path: GPU engine owns the memory, just deregister
pub struct NixlRegisteredBuffer {
    addr: usize,
    len: usize,
    device_id: u64,
    registration: Option<RegistrationHandle>,
    owns_memory: bool,
}

impl NixlRegisteredBuffer {
    pub fn addr(&self) -> usize {
        self.addr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn device_id(&self) -> u64 {
        self.device_id
    }
}

impl Drop for NixlRegisteredBuffer {
    fn drop(&mut self) {
        // 1. Deregister from nixl FIRST (while GPU memory still valid).
        if let Some(reg) = self.registration.take() {
            drop(reg);
        }
        // 2. Free GPU memory only if we own it.
        if self.owns_memory {
            if let Err(e) = cuda_free(self.addr) {
                warn!(
                    error = %e,
                    addr = format_args!("0x{:x}", self.addr),
                    "NixlRegisteredBuffer: failed to free GPU memory"
                );
            }
        }
    }
}

/// Tracks GPU-direct transfer health with adaptive cooldown.
///
/// Instead of permanently disabling GPU transfers on the first timeout,
/// uses a 3-strike cooldown: after 3 consecutive failures, disables for
/// 60 seconds, then re-enables. Successful transfers reset the counter.
struct TransferHealth {
    consecutive_failures: std::sync::atomic::AtomicU32,
    disabled_until: Mutex<Option<std::time::Instant>>,
}

impl TransferHealth {
    fn new() -> Self {
        Self {
            consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            disabled_until: Mutex::new(None),
        }
    }

    /// Check if GPU transfers are currently enabled.
    fn is_enabled(&self) -> bool {
        let guard = self.disabled_until.lock().unwrap();
        match *guard {
            None => true,
            Some(until) => {
                if std::time::Instant::now() >= until {
                    // Cooldown expired — re-enable.
                    drop(guard);
                    self.consecutive_failures
                        .store(0, std::sync::atomic::Ordering::Relaxed);
                    *self.disabled_until.lock().unwrap() = None;
                    info!("GPU-direct transfers re-enabled after cooldown");
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Record a successful transfer.
    fn record_success(&self) {
        self.consecutive_failures
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a transfer failure. Returns true if transfers are now disabled.
    fn record_failure(&self) -> bool {
        let failures = self
            .consecutive_failures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if failures >= 3 {
            let cooldown = std::time::Duration::from_secs(60);
            *self.disabled_until.lock().unwrap() = Some(std::time::Instant::now() + cooldown);
            warn!(
                consecutive_failures = failures,
                cooldown_secs = 60,
                "GPU-direct transfers disabled for 60s cooldown after {} consecutive failures",
                failures
            );
            true
        } else {
            false
        }
    }

    /// Compute timeout duration based on data size.
    /// Base: 30s. Scales with data: +1s per 100MB.
    fn timeout_for_size(&self, data_bytes: usize) -> std::time::Duration {
        let base_secs = 30u64;
        let size_secs = (data_bytes / (100 * 1024 * 1024)) as u64;
        std::time::Duration::from_secs(base_secs + size_secs)
    }
}

/// Wraps a nixl Agent with cached peer metadata.
///
/// SAFETY: The UnsafeCell<Option<GpuStagingBuffer>> is only mutated at startup
/// (register_staging_from_addr) before any concurrent access. After that, it's
/// only read via staging().
unsafe impl Sync for NixlExchange {}
pub struct NixlExchange {
    agent: Agent,
    backend: Backend,
    /// Cache of loaded remote agent metadata: peer_addr → agent_name
    remote_agents: Mutex<HashMap<String, String>>,
    /// GPU-direct transfer health tracking with adaptive cooldown.
    transfer_health: TransferHealth,
    /// Pre-allocated GPU staging buffer for nixl transfers. When present,
    /// source buffers are packed into this buffer via cudf::chunked_pack.
    /// UnsafeCell allows mutation at startup via register_staging_from_addr.
    staging: std::cell::UnsafeCell<Option<GpuStagingBuffer>>,
    /// Cached nixl metadata from startup (valid when using staging buffer,
    /// since the registered memory region doesn't change).
    cached_metadata: Mutex<Option<Vec<u8>>>,
}

impl NixlExchange {
    /// Create a new NixlExchange agent.
    ///
    /// Initializes the UCX backend for GPU-direct transfers.
    /// If `staging_size` is `Some(n)`, allocates an n-byte GPU staging buffer
    /// and registers it with nixl once at startup.
    /// Returns `None` if nixl initialization fails (fallback to bRPC).
    pub fn try_new(agent_name: &str) -> Option<Self> {
        Self::try_new_with_staging(agent_name, None)
    }

    /// Create with an optional staging buffer size.
    pub fn try_new_with_staging(agent_name: &str, staging_size: Option<usize>) -> Option<Self> {
        let mut cfg = AgentConfig::default();
        // Default pthr_delay_us=0 causes 100% CPU spin. Use 100µs polling interval.
        cfg.pthr_delay_us = 100;
        let agent = match Agent::new_configured(agent_name, &cfg) {
            Ok(a) => a,
            Err(e) => {
                warn!(error = %e, "nixl Agent::new failed, GPU-direct exchange disabled");
                return None;
            }
        };

        // Initialize CUDA context before UCX backend creation so that UCX
        // can detect GPU memory types. Without this, UCX treats GPU pointers
        // as host memory ("memory is detected as host"), causing SIGSEGV.
        match ensure_cuda_context() {
            Ok(()) => info!("CUDA context initialized for UCX GPU support"),
            Err(e) => {
                warn!(error = %e, "CUDA init failed, UCX will not support GPU-direct transfers")
            }
        }

        // Create UCX backend for GPU transfers
        let (_, params) = match agent.get_plugin_params("UCX") {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "nixl get_plugin_params(UCX) failed");
                return None;
            }
        };
        let backend = match agent.create_backend("UCX", &params) {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, "nixl UCX backend creation failed");
                return None;
            }
        };

        info!(
            agent = agent_name,
            "nixl agent initialized with UCX backend"
        );

        // Try to create staging buffer if requested.
        let staging = staging_size.and_then(|size| {
            if size == 0 {
                return None;
            }
            match GpuStagingBuffer::new(size, 0, &agent, &backend) {
                Ok(buf) => {
                    info!(size, size_mb = size / (1024 * 1024), "GPU staging buffer ready");
                    Some(buf)
                }
                Err(e) => {
                    warn!(error = %e, size, "failed to create GPU staging buffer, will use per-transfer alloc");
                    None
                }
            }
        });

        let exchange = Self {
            agent,
            backend,
            remote_agents: Mutex::new(HashMap::new()),
            transfer_health: TransferHealth::new(),
            staging: std::cell::UnsafeCell::new(staging),
            cached_metadata: Mutex::new(None),
        };

        // Cache startup metadata if staging buffer is present (memory layout is stable).
        if unsafe { &*exchange.staging.get() }.is_some() {
            match exchange.agent.get_local_md() {
                Ok(md) => {
                    info!(
                        md_len = md.len(),
                        "cached nixl metadata (staging buffer registered)"
                    );
                    *exchange.cached_metadata.lock().unwrap() = Some(md);
                }
                Err(e) => warn!(error = %e, "failed to cache startup metadata"),
            }
        }

        Some(exchange)
    }

    /// Register a pre-allocated GPU address as the staging buffer with nixl.
    ///
    /// Used when the staging buffer was allocated via C++ cudaMalloc (to ensure
    /// it's in the same CUDA runtime as RMM). The buffer is registered with nixl
    /// once and reused for all transfers.
    ///
    /// Must be called at startup before any concurrent access (before gRPC server starts).
    pub fn register_staging_from_addr(&self, addr: usize, size: usize) -> Result<(), String> {
        let staging = GpuStagingBuffer::from_existing(addr, size, 0, &self.agent, &self.backend)?;
        // Cache metadata after registration.
        match self.agent.get_local_md() {
            Ok(md) => {
                info!(
                    md_len = md.len(),
                    "cached nixl metadata (C++ staging buffer registered)"
                );
                *self.cached_metadata.lock().unwrap() = Some(md);
            }
            Err(e) => warn!(error = %e, "failed to cache metadata after staging registration"),
        }
        // SAFETY: called at startup before any concurrent access.
        unsafe {
            *self.staging.get() = Some(staging);
        }
        Ok(())
    }

    /// Register a send staging buffer with nixl (for outgoing transfers).
    /// This buffer is separate from the recv staging (used for incoming transfers).
    /// The send buffer is where cudf::chunked_pack writes packed GPU data.
    pub fn register_send_staging(&self, addr: usize, size: usize) -> Result<(), String> {
        use crate::gpu_staging_buffer::GpuStagingBuffer;
        // Just register with nixl — no bump allocator needed (C++ manages this buffer).
        let _send_staging =
            GpuStagingBuffer::from_existing(addr, size, 0, &self.agent, &self.backend)?;
        // Re-cache metadata after registering the new memory region.
        match self.agent.get_local_md() {
            Ok(md) => {
                info!(
                    md_len = md.len(),
                    "cached nixl metadata (send + recv staging registered)"
                );
                *self.cached_metadata.lock().unwrap() = Some(md);
            }
            Err(e) => warn!(error = %e, "failed to re-cache metadata after send staging"),
        }
        // Keep the registration alive by leaking it (startup-only, lives for process lifetime).
        std::mem::forget(_send_staging);
        Ok(())
    }

    /// Access the nixl Agent (for staging buffer creation etc.).
    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Access the nixl Backend.
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Access the GPU staging buffer, if allocated.
    pub fn staging(&self) -> Option<&GpuStagingBuffer> {
        // SAFETY: staging is only mutated at startup before concurrent access.
        unsafe { &*self.staging.get() }.as_ref()
    }

    /// Get cached metadata (from startup, when staging buffer is registered).
    /// Returns `None` if no cached metadata is available.
    pub fn get_cached_metadata(&self) -> Option<Vec<u8>> {
        self.cached_metadata.lock().unwrap().clone()
    }

    /// Get metadata: cached if staging buffer is present, otherwise fresh.
    ///
    /// When the staging buffer is registered, the memory layout is stable,
    /// so we can reuse the startup metadata. When per-buffer registration
    /// changes the memory map, we need fresh metadata.
    pub fn get_metadata(&self) -> Result<Vec<u8>, String> {
        if let Some(cached) = self.get_cached_metadata() {
            return Ok(cached);
        }
        self.get_fresh_metadata()
    }

    /// Whether GPU-direct transfers (RDMA/UCX) are currently available.
    ///
    /// Uses adaptive cooldown: after 3 consecutive failures, disabled for 60s,
    /// then automatically re-enabled.
    pub fn gpu_transfer_enabled(&self) -> bool {
        self.transfer_health.is_enabled()
    }

    /// Force-disable GPU-direct transfers (for testing or manual override).
    /// Internally records enough consecutive failures to trigger cooldown.
    pub fn disable_gpu_transfer(&self) {
        for _ in 0..3 {
            self.transfer_health.record_failure();
        }
    }

    /// Ensure the RMM pool containing `any_rmm_ptr` is registered with nixl.
    ///
    /// Uses `cuMemGetAddressRange` to discover the pool's base address and size
    /// from any sub-allocation pointer. The registration is cached — subsequent
    /// calls with pointers in the same pool are no-ops.
    /// Register GPU memory buffers with the nixl agent (no ownership transfer).
    ///
    /// Returns `NixlRegisteredBuffer` handles with `owns_memory: false` — the
    /// caller's GPU engine still owns the memory. Dropping the handles only
    /// deregisters from nixl.
    pub fn register_gpu_buffers(
        &self,
        buffers: &[(usize, usize, u64)], // (addr, len, device_id)
    ) -> Result<Vec<NixlRegisteredBuffer>, String> {
        // Ensure the current thread has a CUDA context. The context created
        // in try_new() is per-thread, so tokio worker threads need their own.
        // Without this, UCX's cuPointerGetAttribute calls fail and GPU memory
        // is misidentified as host memory.
        if let Err(e) = ensure_cuda_context() {
            warn!(error = %e, "CUDA context init failed in register_gpu_buffers");
        }

        let mut opt = OptArgs::new().map_err(|e| format!("OptArgs::new: {e}"))?;
        opt.add_backend(&self.backend)
            .map_err(|e| format!("add_backend: {e}"))?;

        let mut result = Vec::with_capacity(buffers.len());
        for &(addr, len, device_id) in buffers {
            let wrapper = GpuBufferWrapper {
                addr,
                len,
                device_id,
            };
            let handle = self
                .agent
                .register_memory(&wrapper, Some(&opt))
                .map_err(|e| format!("register_memory(addr=0x{addr:x}, len={len}): {e}"))?;
            result.push(NixlRegisteredBuffer {
                addr,
                len,
                device_id,
                registration: Some(handle),
                owns_memory: false,
            });
        }

        info!(
            num_buffers = result.len(),
            "registered GPU buffers with nixl agent"
        );
        Ok(result)
    }

    /// Allocate and register GPU memory buffers in one operation.
    ///
    /// Returns `NixlRegisteredBuffer` handles with `owns_memory: true` — dropping
    /// them deregisters from nixl AND frees the GPU memory. On partial failure,
    /// already-allocated buffers are cleaned up via RAII.
    pub fn allocate_and_register_gpu_buffers(
        &self,
        sizes: &[(u64, u64)], // (len, device_id) pairs
    ) -> Result<Vec<NixlRegisteredBuffer>, String> {
        ensure_cuda_context()?;

        let mut opt = OptArgs::new().map_err(|e| format!("OptArgs::new: {e}"))?;
        opt.add_backend(&self.backend)
            .map_err(|e| format!("add_backend: {e}"))?;

        let mut result = Vec::with_capacity(sizes.len());
        for &(len, device_id) in sizes {
            // Allocate GPU memory. On failure, `result` drops and cleans up previous.
            let addr = cuda_alloc(len as usize)?;

            // Register with nixl. On failure, free this buffer manually
            // (it's not yet in `result`), and `result` cleans up previous.
            let wrapper = GpuBufferWrapper {
                addr,
                len: len as usize,
                device_id,
            };
            let handle = match self.agent.register_memory(&wrapper, Some(&opt)) {
                Ok(h) => h,
                Err(e) => {
                    let _ = cuda_free(addr);
                    return Err(format!("register_memory(0x{addr:x}, {len}): {e}"));
                }
            };

            result.push(NixlRegisteredBuffer {
                addr,
                len: len as usize,
                device_id,
                registration: Some(handle),
                owns_memory: true,
            });
        }

        info!(
            num_buffers = result.len(),
            "allocated and registered GPU buffers"
        );
        Ok(result)
    }

    /// Get fresh metadata that includes any newly registered memory.
    pub fn get_fresh_metadata(&self) -> Result<Vec<u8>, String> {
        self.agent
            .get_local_md()
            .map_err(|e| format!("get_local_md: {e}"))
    }

    /// Load remote metadata, invalidating any cached entry for this peer first.
    ///
    /// Use this instead of `load_remote_metadata` when the peer may have
    /// registered new memory since the last exchange.
    pub fn force_load_remote_metadata(
        &self,
        peer_addr: &str,
        remote_metadata: &[u8],
    ) -> Result<String, String> {
        let mut cache = self.remote_agents.lock().unwrap();

        // Invalidate old entry if present.
        if let Some(old_name) = cache.remove(peer_addr) {
            let _ = self.agent.invalidate_remote_md(&old_name);
        }

        let remote_name = self
            .agent
            .load_remote_md(remote_metadata)
            .map_err(|e| format!("load_remote_md: {e}"))?;

        info!(
            peer = peer_addr,
            remote_agent = %remote_name,
            "loaded remote nixl metadata (forced)"
        );

        cache.insert(peer_addr.to_string(), remote_name.clone());
        Ok(remote_name)
    }

    /// Load a remote peer's metadata. Returns the remote agent name.
    ///
    /// Caches the result so subsequent calls for the same peer are no-ops.
    pub fn load_remote_metadata(
        &self,
        peer_addr: &str,
        remote_metadata: &[u8],
    ) -> Result<String, String> {
        let mut cache = self.remote_agents.lock().unwrap();
        if let Some(name) = cache.get(peer_addr) {
            return Ok(name.clone());
        }

        let remote_name = self
            .agent
            .load_remote_md(remote_metadata)
            .map_err(|e| format!("load_remote_md: {e}"))?;

        info!(
            peer = peer_addr,
            remote_agent = %remote_name,
            "loaded remote nixl metadata"
        );

        cache.insert(peer_addr.to_string(), remote_name.clone());
        Ok(remote_name)
    }

    /// Invalidate cached metadata for a peer (e.g. on BE restart).
    pub fn invalidate_peer(&self, peer_addr: &str) {
        let mut cache = self.remote_agents.lock().unwrap();
        if let Some(name) = cache.remove(peer_addr) {
            let _ = self.agent.invalidate_remote_md(&name);
        }
    }

    /// Post a GPU-to-GPU transfer request.
    ///
    /// `src_descs`: local GPU memory descriptors (source data)
    /// `dst_descs`: remote GPU memory descriptors (destination)
    /// `remote_agent`: name returned by `load_remote_metadata`
    ///
    /// Returns when the transfer is complete. Times out after 10 seconds.
    /// On timeout, disables GPU-direct transfers for all future calls
    /// (indicates UCX cannot handle GPU memory on this system).
    pub fn transfer_gpu_to_gpu(
        &self,
        src_descs: &XferDescList,
        dst_descs: &XferDescList,
        remote_agent: &str,
    ) -> Result<(), String> {
        self.transfer_gpu_to_gpu_with_size(src_descs, dst_descs, remote_agent, 0)
    }

    /// GPU-to-GPU transfer with data-size-based timeout.
    ///
    /// Timeout scales with data size: base 30s + 1s per 100MB.
    /// On failure: records in TransferHealth. After 3 consecutive failures,
    /// disables for 60s cooldown, then auto-re-enables.
    pub fn transfer_gpu_to_gpu_with_size(
        &self,
        src_descs: &XferDescList,
        dst_descs: &XferDescList,
        remote_agent: &str,
        data_bytes: usize,
    ) -> Result<(), String> {
        let req = self
            .agent
            .create_xfer_req(XferOp::Write, src_descs, dst_descs, remote_agent, None)
            .map_err(|e| format!("create_xfer_req: {e}"))?;

        let in_progress = self
            .agent
            .post_xfer_req(&req, None)
            .map_err(|e| format!("post_xfer_req: {e}"))?;

        if !in_progress {
            info!("nixl transfer completed immediately");
            self.transfer_health.record_success();
            return Ok(());
        }

        let timeout = self.transfer_health.timeout_for_size(data_bytes);
        let deadline = std::time::Instant::now() + timeout;
        loop {
            match self
                .agent
                .get_xfer_status(&req)
                .map_err(|e| format!("get_xfer_status: {e}"))?
            {
                XferStatus::Success => {
                    info!("nixl transfer complete");
                    self.transfer_health.record_success();
                    return Ok(());
                }
                XferStatus::InProgress => {
                    if std::time::Instant::now() >= deadline {
                        self.transfer_health.record_failure();
                        return Err(format!(
                            "nixl transfer timed out after {}s (data_size={})",
                            timeout.as_secs(),
                            data_bytes
                        ));
                    }
                    std::thread::yield_now();
                }
            }
        }
    }

    /// Create a descriptor list for GPU memory regions.
    pub fn create_gpu_descs(
        &self,
        gpu_ptrs: &[(usize, usize)], // (addr, len) pairs
        device_id: u64,
    ) -> Result<XferDescList<'_>, String> {
        let mut descs =
            XferDescList::new(MemType::Vram).map_err(|e| format!("XferDescList::new: {e}"))?;
        for &(addr, len) in gpu_ptrs {
            descs.add_desc(addr, len, device_id);
        }
        Ok(descs)
    }
}

/// Descriptor for a GPU memory region to be transferred.
#[derive(Debug, Clone)]
pub struct GpuBufferDesc {
    /// GPU pointer address.
    pub addr: usize,
    /// Buffer size in bytes.
    pub len: usize,
    /// GPU device ID.
    pub device_id: u64,
}

/// Message exchanged via gRPC for nixl metadata handshake.
#[derive(Debug, Clone)]
pub struct NixlMetadataExchange {
    /// The sender's nixl agent metadata bytes.
    pub metadata: Vec<u8>,
    /// GPU buffer descriptors being offered for transfer.
    pub buffer_descs: Vec<GpuBufferDesc>,
    /// Column schema information (name, type_id pairs).
    pub column_info: Vec<(String, i32)>,
    /// Number of rows in the data.
    pub num_rows: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nixl_exchange_initialization() {
        let agent_name = "test-agent";
        let result = NixlExchange::try_new(agent_name);
        match result {
            Some(exchange) => {
                let md = exchange.get_fresh_metadata().unwrap();
                assert!(!md.is_empty());
            }
            None => {
                // Expected when nixl is not available
            }
        }
    }

    #[test]
    fn test_fresh_metadata_retrieval() {
        if let Some(exchange) = NixlExchange::try_new("metadata-test") {
            let md = exchange.get_fresh_metadata().unwrap();
            assert!(!md.is_empty(), "fresh metadata should not be empty");
        }
    }

    #[test]
    fn test_load_remote_metadata() {
        let Some(exchange) = NixlExchange::try_new("peer-test") else {
            // Skip if nixl not available
            return;
        };

        // Simulate remote metadata (in real scenario this comes from another BE).
        let fake_remote_md = b"fake-remote-metadata";
        let result = exchange.load_remote_metadata("remote-be:8060", fake_remote_md);

        // Should either succeed or fail gracefully.
        match result {
            Ok(remote_name) => {
                assert!(!remote_name.is_empty());

                // Loading same peer again should use cache and return same name.
                let result2 = exchange.load_remote_metadata("remote-be:8060", fake_remote_md);
                assert!(result2.is_ok());
                assert_eq!(result2.unwrap(), remote_name);
            }
            Err(e) => {
                // Expected if metadata format is invalid
                assert!(e.contains("load_remote_md") || e.contains("metadata"));
            }
        }
    }

    #[test]
    fn test_invalidate_peer() {
        let Some(exchange) = NixlExchange::try_new("invalidate-test") else {
            return;
        };

        let peer_addr = "peer:8060";
        let fake_md = b"peer-metadata";

        // Load, then invalidate.
        if exchange.load_remote_metadata(peer_addr, fake_md).is_ok() {
            exchange.invalidate_peer(peer_addr);
            // After invalidation, loading again should re-register (no cached entry).
            // We can't easily verify this without internal state inspection, but at least
            // ensure invalidate doesn't panic.
        }
    }

    #[test]
    fn test_create_gpu_descs() {
        let Some(exchange) = NixlExchange::try_new("desc-test") else {
            return;
        };

        let gpu_ptrs = vec![(0x1000, 1024), (0x2000, 2048)];
        let device_id = 0;

        let result = exchange.create_gpu_descs(&gpu_ptrs, device_id);
        match result {
            Ok(_descs) => {
                // Successfully created descriptor list
            }
            Err(e) => {
                // May fail if GPU not available
                assert!(e.contains("XferDescList") || e.contains("desc"));
            }
        }
    }

    #[test]
    fn test_transfer_gpu_to_gpu_mock() {
        // This test requires actual GPU memory, so we can't run it in CI.
        // It serves as documentation of the API usage pattern.

        // Pattern:
        // 1. Both sender and receiver create NixlExchange agents
        // 2. Exchange metadata via gRPC
        // 3. Sender creates src_descs from its GPU buffers
        // 4. Receiver creates dst_descs (allocates GPU buffers)
        // 5. Receiver calls transfer_gpu_to_gpu to pull data

        // For actual testing, this would need:
        // - GPU-enabled environment
        // - Two separate processes or threads simulating sender/receiver
        // - Actual GPU allocations
    }

    #[test]
    fn test_gpu_buffer_desc_clone() {
        let desc = GpuBufferDesc {
            addr: 0xDEADBEEF,
            len: 4096,
            device_id: 0,
        };

        let cloned = desc.clone();
        assert_eq!(cloned.addr, desc.addr);
        assert_eq!(cloned.len, desc.len);
        assert_eq!(cloned.device_id, desc.device_id);
    }

    #[test]
    fn test_nixl_metadata_exchange_structure() {
        let msg = NixlMetadataExchange {
            metadata: vec![1, 2, 3, 4],
            buffer_descs: vec![GpuBufferDesc {
                addr: 0x1000,
                len: 100,
                device_id: 0,
            }],
            column_info: vec![
                ("col1".to_string(), 5),  // INT32
                ("col2".to_string(), 16), // STRING
            ],
            num_rows: 42,
        };

        assert_eq!(msg.metadata.len(), 4);
        assert_eq!(msg.buffer_descs.len(), 1);
        assert_eq!(msg.column_info.len(), 2);
        assert_eq!(msg.num_rows, 42);

        // Test clone
        let cloned = msg.clone();
        assert_eq!(cloned.num_rows, 42);
    }

    #[test]
    fn test_multiple_peer_caching() {
        let Some(exchange) = NixlExchange::try_new("multi-peer-test") else {
            return;
        };

        let peers = vec![
            ("peer1:8060", b"md1".as_slice()),
            ("peer2:8060", b"md2".as_slice()),
            ("peer3:8060", b"md3".as_slice()),
        ];

        let mut names = Vec::new();
        for (addr, md) in &peers {
            if let Ok(name) = exchange.load_remote_metadata(addr, md) {
                names.push(name);
            }
        }

        // At least verify we can load multiple peers without errors.
        // Actual name validation depends on nixl-sys implementation.
    }
}
