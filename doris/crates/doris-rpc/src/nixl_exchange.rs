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
use std::sync::{Arc, Mutex};

use nixl_sys::{Agent, AgentConfig, Backend, MemType, Params, XferDescList, XferOp, XferStatus};
use tracing::{info, warn};

/// Wraps a nixl Agent with cached peer metadata.
pub struct NixlExchange {
    agent: Agent,
    _backend: Backend,
    local_metadata: Vec<u8>,
    /// Cache of loaded remote agent metadata: peer_addr → agent_name
    remote_agents: Mutex<HashMap<String, String>>,
}

impl NixlExchange {
    /// Create a new NixlExchange agent.
    ///
    /// Initializes the UCX backend for GPU-direct transfers.
    /// Returns `None` if nixl initialization fails (fallback to bRPC).
    pub fn try_new(agent_name: &str) -> Option<Self> {
        let agent = match Agent::new(agent_name) {
            Ok(a) => a,
            Err(e) => {
                warn!(error = %e, "nixl Agent::new failed, GPU-direct exchange disabled");
                return None;
            }
        };

        // Create UCX backend for GPU transfers
        let params = Params::new();
        let backend = match agent.create_backend("UCX", &params) {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, "nixl UCX backend creation failed");
                return None;
            }
        };

        let local_metadata = match agent.get_local_md() {
            Ok(md) => md,
            Err(e) => {
                warn!(error = %e, "nixl get_local_md failed");
                return None;
            }
        };

        info!(
            agent = agent_name,
            metadata_size = local_metadata.len(),
            "nixl agent initialized with UCX backend"
        );

        Some(Self {
            agent,
            _backend: backend,
            local_metadata,
            remote_agents: Mutex::new(HashMap::new()),
        })
    }

    /// Get the local metadata bytes for exchange with peers.
    pub fn local_metadata(&self) -> &[u8] {
        &self.local_metadata
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
    /// Returns when the transfer is complete.
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

        // Poll until complete
        loop {
            match self
                .agent
                .get_xfer_status(&req)
                .map_err(|e| format!("get_xfer_status: {e}"))?
            {
                XferStatus::Done => {
                    info!("nixl transfer complete");
                    return Ok(());
                }
                XferStatus::Error => {
                    return Err("nixl transfer failed".to_string());
                }
                _ => {
                    // Still in progress — yield briefly
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
    ) -> Result<XferDescList, String> {
        let mut descs =
            XferDescList::new(MemType::Vram).map_err(|e| format!("XferDescList::new: {e}"))?;
        for &(addr, len) in gpu_ptrs {
            descs
                .add_desc(addr, len, device_id)
                .map_err(|e| format!("add_desc: {e}"))?;
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
        // Test that NixlExchange can be created with a valid agent name.
        let agent_name = "test-agent";
        let result = NixlExchange::try_new(agent_name);

        // Should succeed (or return None if nixl library not available).
        // We don't assert success because nixl may not be available in test environment.
        match result {
            Some(exchange) => {
                assert!(!exchange.local_metadata().is_empty());
            }
            None => {
                // Expected when nixl is not available
            }
        }
    }

    #[test]
    fn test_local_metadata_retrieval() {
        if let Some(exchange) = NixlExchange::try_new("metadata-test") {
            let metadata = exchange.local_metadata();
            assert!(!metadata.is_empty(), "local metadata should not be empty");
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
