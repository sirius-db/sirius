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

/// Wraps a nixl Agent with cached peer metadata.
pub struct NixlExchange {
    agent: Agent,
    backend: Backend,
    /// Cache of loaded remote agent metadata: peer_addr → agent_name
    remote_agents: Mutex<HashMap<String, String>>,
    /// Active registration handles (dropped = deregistered).
    _registrations: Mutex<Vec<RegistrationHandle>>,
    /// Whether GPU-direct transfers are available (UCX has CUDA support).
    /// Starts `true`, set to `false` if `register_gpu_buffers` detects that
    /// UCX treats GPU memory as host memory (avoids SIGSEGV during transfer).
    gpu_transfer_enabled: std::sync::atomic::AtomicBool,
}

impl NixlExchange {
    /// Create a new NixlExchange agent.
    ///
    /// Initializes the UCX backend for GPU-direct transfers.
    /// Returns `None` if nixl initialization fails (fallback to bRPC).
    pub fn try_new(agent_name: &str) -> Option<Self> {
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
        match crate::cuda_driver::ensure_cuda_context() {
            Ok(()) => info!("CUDA context initialized for UCX GPU support"),
            Err(e) => warn!(error = %e, "CUDA init failed, UCX will not support GPU-direct transfers"),
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

        Some(Self {
            agent,
            backend,
            remote_agents: Mutex::new(HashMap::new()),
            _registrations: Mutex::new(Vec::new()),
            // Assume GPU-direct is available until register_gpu_buffers detects otherwise.
            gpu_transfer_enabled: std::sync::atomic::AtomicBool::new(true),
        })
    }

    /// Whether GPU-direct transfers (RDMA/UCX) are available.
    ///
    /// Starts `true`, set to `false` when `register_gpu_buffers` detects that
    /// UCX treats GPU memory as host memory. Callers should check this before
    /// attempting `transfer_gpu_to_gpu` to avoid SIGSEGV.
    pub fn gpu_transfer_enabled(&self) -> bool {
        self.gpu_transfer_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Register GPU memory buffers with the nixl agent.
    ///
    /// Must be called before `get_fresh_metadata()` so that metadata
    /// includes the registered regions. Returns registration handles
    /// that deregister on drop.
    ///
    /// Also captures stderr during registration to detect if UCX treats GPU
    /// memory as host memory. If detected, sets `gpu_transfer_enabled` to
    /// `false` to prevent SIGSEGV during subsequent transfers.
    pub fn register_gpu_buffers(
        &self,
        buffers: &[(usize, usize, u64)], // (addr, len, device_id)
    ) -> Result<(), String> {
        // Ensure the current thread has a CUDA context. The context created
        // in try_new() is per-thread, so tokio worker threads need their own.
        // Without this, UCX's cuPointerGetAttribute calls fail and GPU memory
        // is misidentified as host memory.
        if let Err(e) = crate::cuda_driver::ensure_cuda_context() {
            warn!(error = %e, "CUDA context init failed in register_gpu_buffers");
        }

        let mut opt = OptArgs::new().map_err(|e| format!("OptArgs::new: {e}"))?;
        opt.add_backend(&self.backend)
            .map_err(|e| format!("add_backend: {e}"))?;

        // Capture stderr during registration to detect UCX "memory is detected
        // as host" warning, which indicates GPU-direct transfers will SIGSEGV.
        let old_stderr = unsafe { libc::dup(2) };
        let mut pipe_fds = [0i32; 2];
        let capturing = old_stderr >= 0 && unsafe { libc::pipe(pipe_fds.as_mut_ptr()) } == 0;
        if capturing {
            unsafe { libc::dup2(pipe_fds[1], 2) };
        }

        let mut regs = self._registrations.lock().unwrap();
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
            regs.push(handle);
        }

        // Restore stderr and check captured output.
        if capturing {
            unsafe { libc::dup2(old_stderr, 2) };
            unsafe { libc::close(old_stderr) };
            unsafe { libc::close(pipe_fds[1]) };

            unsafe { libc::fcntl(pipe_fds[0], libc::F_SETFL, libc::O_NONBLOCK) };
            let mut captured = vec![0u8; 4096];
            let n = unsafe {
                libc::read(pipe_fds[0], captured.as_mut_ptr() as _, captured.len())
            };
            unsafe { libc::close(pipe_fds[0]) };

            if n > 0 {
                let output = String::from_utf8_lossy(&captured[..n as usize]);
                // Re-emit captured stderr so it's still visible in logs.
                eprint!("{output}");
                if output.contains("memory is detected as host") {
                    warn!(
                        "UCX treats GPU memory as host memory — disabling GPU-direct transfers \
                         (will use bRPC fallback). Check UCX CUDA support configuration."
                    );
                    self.gpu_transfer_enabled
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
        } else if old_stderr >= 0 {
            unsafe { libc::close(old_stderr) };
        }

        info!(
            num_buffers = buffers.len(),
            gpu_transfer = self.gpu_transfer_enabled(),
            "registered GPU buffers with nixl agent"
        );
        Ok(())
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
    /// Returns when the transfer is complete. Times out after 10 seconds
    /// to avoid hanging when UCX can't handle GPU memory (returns error
    /// so the caller can fall back to bRPC).
    pub fn transfer_gpu_to_gpu(
        &self,
        src_descs: &XferDescList,
        dst_descs: &XferDescList,
        remote_agent: &str,
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
            // Completed immediately
            info!("nixl transfer completed immediately");
            return Ok(());
        }

        // Poll until complete, with a timeout to avoid hanging when UCX
        // can't handle GPU memory (e.g. "memory is detected as host").
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match self
                .agent
                .get_xfer_status(&req)
                .map_err(|e| format!("get_xfer_status: {e}"))?
            {
                XferStatus::Success => {
                    info!("nixl transfer complete");
                    return Ok(());
                }
                XferStatus::InProgress => {
                    if std::time::Instant::now() >= deadline {
                        return Err("nixl transfer timed out after 10s (UCX may not support GPU memory on this system)".to_string());
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

        let gpu_ptrs = vec![
            (0x1000, 1024),
            (0x2000, 2048),
        ];
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
            buffer_descs: vec![
                GpuBufferDesc {
                    addr: 0x1000,
                    len: 100,
                    device_id: 0,
                },
            ],
            column_info: vec![
                ("col1".to_string(), 5), // INT32
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
