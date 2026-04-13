//! Mock nixl-sys implementation for testing.
//!
//! Provides stub implementations of nixl-sys types and functions for unit tests
//! without requiring GPU hardware or the actual NIXL library.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock Agent for nixl testing.
#[derive(Debug)]
pub struct Agent {
    name: String,
    local_md: Vec<u8>,
    remote_agents: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

/// Mock Backend.
#[derive(Debug)]
pub struct Backend {
    _name: String,
}

/// Mock agent configuration.
#[derive(Debug)]
pub struct AgentConfig;

/// Mock transfer parameters.
#[derive(Debug, Clone)]
pub struct Params;

impl Params {
    pub fn new() -> Self {
        Params
    }
}

/// Memory type for transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    /// GPU memory (VRAM).
    Vram,
    /// System memory.
    System,
}

/// Transfer operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XferOp {
    /// Write from local to remote.
    Write,
    /// Read from remote to local.
    Read,
}

/// Transfer status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XferStatus {
    /// Transfer in progress.
    InProgress,
    /// Transfer complete.
    Done,
    /// Transfer failed.
    Error,
}

/// Transfer descriptor list.
#[derive(Debug)]
pub struct XferDescList {
    mem_type: MemType,
    descriptors: Vec<(usize, usize, u64)>, // (addr, len, device_id)
}

impl XferDescList {
    pub fn new(mem_type: MemType) -> Result<Self, String> {
        Ok(Self {
            mem_type,
            descriptors: Vec::new(),
        })
    }

    pub fn add_desc(&mut self, addr: usize, len: usize, device_id: u64) -> Result<(), String> {
        self.descriptors.push((addr, len, device_id));
        Ok(())
    }

    pub fn descriptors(&self) -> &[(usize, usize, u64)] {
        &self.descriptors
    }
}

/// Transfer request handle.
#[derive(Debug)]
pub struct XferRequest {
    id: u64,
}

impl Agent {
    pub fn new(name: &str) -> Result<Self, String> {
        // Simulate local metadata (8 bytes for simplicity).
        let local_md = format!("md:{}", name).into_bytes();
        Ok(Self {
            name: name.to_string(),
            local_md,
            remote_agents: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn create_backend(&self, backend_type: &str, _params: &Params) -> Result<Backend, String> {
        if backend_type != "UCX" {
            return Err(format!("unsupported backend: {}", backend_type));
        }
        Ok(Backend {
            _name: backend_type.to_string(),
        })
    }

    pub fn get_local_md(&self) -> Result<Vec<u8>, String> {
        Ok(self.local_md.clone())
    }

    pub fn load_remote_md(&self, metadata: &[u8]) -> Result<String, String> {
        // Parse remote agent name from metadata.
        let remote_name = String::from_utf8(metadata.to_vec())
            .map_err(|e| format!("invalid metadata: {}", e))?
            .strip_prefix("md:")
            .ok_or("invalid metadata format")?
            .to_string();

        let mut cache = self.remote_agents.lock().unwrap();
        cache.insert(remote_name.clone(), metadata.to_vec());
        Ok(remote_name)
    }

    pub fn invalidate_remote_md(&self, remote_name: &str) -> Result<(), String> {
        let mut cache = self.remote_agents.lock().unwrap();
        cache.remove(remote_name);
        Ok(())
    }

    pub fn create_xfer_req(
        &self,
        _op: XferOp,
        _src_descs: &XferDescList,
        _dst_descs: &XferDescList,
        _remote_agent: &str,
        _params: Option<&Params>,
    ) -> Result<XferRequest, String> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(XferRequest { id })
    }

    pub fn post_xfer_req(
        &self,
        _req: &XferRequest,
        _params: Option<&Params>,
    ) -> Result<bool, String> {
        // Return true to indicate async (not completed immediately).
        Ok(true)
    }

    pub fn get_xfer_status(&self, _req: &XferRequest) -> Result<XferStatus, String> {
        // Mock: always return Done (immediate completion).
        Ok(XferStatus::Done)
    }
}
