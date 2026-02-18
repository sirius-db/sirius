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
