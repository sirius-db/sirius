//! Transfer engine abstraction for inter-BE exchange.
//!
//! Provides two transfer backends:
//! - [`NixlTransferEngine`]: GPU-direct via nixl (staging buffer → RDMA)
//! - [`BrpcTransferEngine`]: CPU serialization via bRPC
//!
//! [`TransferDispatcher`] orchestrates engine selection and fallback.

use std::sync::Arc;

use tracing::{info, warn};

use crate::exchange_sender::ExchangeDest;
use crate::nixl_exchange::{GpuBufferDesc, NixlExchange};
use sirius_ffi::GpuColumnBuffers;

/// GPU execution result ready for inter-BE transfer.
pub struct GpuTransferPayload {
    /// GPU buffer descriptors (addr, len, device_id).
    pub buffers: Vec<GpuBufferDesc>,
    /// Column names and type IDs.
    pub column_info: Vec<(String, i32)>,
    /// Per-column extended buffer info (null masks, string offsets).
    pub column_buffers: Vec<GpuColumnBuffers>,
    /// Number of rows.
    pub num_rows: u32,
    /// Arrow IPC bytes (used by bRPC path and as nixl fallback).
    pub ipc_bytes: Vec<u8>,
}

/// Staged payload ready for network transfer.
///
/// Buffer addresses may differ from the original `GpuTransferPayload` —
/// they point into the staging buffer or cuMemAlloc copies.
///
/// The `_guard` field holds RAII handles (staging leases or cuMemAlloc
/// addresses) that are cleaned up when the payload is dropped.
pub struct StagedPayload {
    /// GPU buffer descriptors with staged addresses.
    pub buffers: Vec<GpuBufferDesc>,
    /// Column names and type IDs.
    pub column_info: Vec<(String, i32)>,
    /// Per-column extended buffer info (with staged addresses).
    pub column_buffers: Vec<GpuColumnBuffers>,
    /// Number of rows.
    pub num_rows: u32,
    /// Arrow IPC bytes.
    pub ipc_bytes: Vec<u8>,
    /// Whether the staging buffer was used (skip per-buffer registration).
    pub staged: bool,
    /// RAII guard: holds staging leases or cuMemAlloc addresses.
    /// Dropped after transfer completes.
    _guard: Box<dyn Send>,
}

impl std::fmt::Debug for StagedPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedPayload")
            .field("num_buffers", &self.buffers.len())
            .field("num_rows", &self.num_rows)
            .field("staged", &self.staged)
            .field("ipc_len", &self.ipc_bytes.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// NixlTransferEngine
// ---------------------------------------------------------------------------

/// GPU-direct transfer via nixl (staging buffer → RDMA).
pub struct NixlTransferEngine {
    agent: Arc<NixlExchange>,
}

impl NixlTransferEngine {
    pub fn new(agent: Arc<NixlExchange>) -> Self {
        Self { agent }
    }

    pub fn name(&self) -> &str {
        "nixl"
    }

    pub fn is_available(&self) -> bool {
        self.agent.gpu_transfer_enabled()
    }

    pub fn stage(&self, payload: GpuTransferPayload) -> Result<StagedPayload, String> {
        use crate::nixl_integration::ExecutionLocation;

        // Wrap payload in ExecutionLocation::Gpu to reuse existing staging logic.
        let mut loc = ExecutionLocation::Gpu {
            buffers: payload.buffers,
            column_info: payload.column_info,
            column_buffers: payload.column_buffers,
            num_rows: payload.num_rows,
            schema_ipc: vec![],
            ipc_bytes: payload.ipc_bytes,
            _staging_leases: vec![],
            packed_metadata: None,
            packed_partitions: vec![],
        };

        // Try staging buffer (D2D copy into pre-registered cuMemAlloc region).
        let _staged = self
            .agent
            .staging()
            .map(|s| loc.try_stage_buffers(s))
            .unwrap_or(false);

        // Destructure to extract the staged data.
        match loc {
            ExecutionLocation::Gpu {
                buffers,
                column_info,
                column_buffers,
                num_rows,
                ipc_bytes,
                _staging_leases,
                ..
            } => {
                let has_staging = !_staging_leases.is_empty();

                Ok(StagedPayload {
                    buffers,
                    column_info,
                    column_buffers,
                    num_rows,
                    ipc_bytes,
                    staged: has_staging,
                    _guard: Box::new(_staging_leases),
                })
            }
            ExecutionLocation::Cpu(_) => unreachable!("constructed as Gpu"),
        }
    }

    pub async fn transfer_to_peer(
        &self,
        staged: &StagedPayload,
        dest: &ExchangeDest,
        query_id: (i64, i64),
        node_id: i32,
        sender_id: i32,
    ) -> Result<(), String> {
        crate::nixl_integration::send_nixl_to_peer(
            &self.agent,
            &staged.buffers,
            &staged.column_info,
            &staged.column_buffers,
            staged.num_rows,
            &staged.ipc_bytes,
            dest,
            query_id,
            node_id,
            sender_id,
            staged.staged,
            None, // packed_metadata — transfer engine uses per-column path
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// BrpcTransferEngine
// ---------------------------------------------------------------------------

/// CPU serialization transfer via bRPC.
pub struct BrpcTransferEngine;

impl BrpcTransferEngine {
    pub fn name(&self) -> &str {
        "brpc"
    }

    pub fn is_available(&self) -> bool {
        true
    }

    pub fn stage(&self, payload: GpuTransferPayload) -> Result<StagedPayload, String> {
        Ok(StagedPayload {
            buffers: payload.buffers,
            column_info: payload.column_info,
            column_buffers: payload.column_buffers,
            num_rows: payload.num_rows,
            ipc_bytes: payload.ipc_bytes,
            staged: false,
            _guard: Box::new(()),
        })
    }

    pub async fn transfer_to_peer(
        &self,
        staged: &StagedPayload,
        dest: &ExchangeDest,
        query_id: (i64, i64),
        node_id: i32,
        sender_id: i32,
    ) -> Result<(), String> {
        crate::exchange_sender::send_exchange_result(
            &staged.ipc_bytes,
            &[dest.clone()],
            query_id,
            node_id,
            sender_id,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// TransferEngine enum dispatch
// ---------------------------------------------------------------------------

/// Transfer engine enum — dispatches to the appropriate backend.
pub enum TransferEngine {
    Nixl(NixlTransferEngine),
    Brpc(BrpcTransferEngine),
}

impl TransferEngine {
    pub fn name(&self) -> &str {
        match self {
            Self::Nixl(e) => e.name(),
            Self::Brpc(e) => e.name(),
        }
    }

    pub fn is_available(&self) -> bool {
        match self {
            Self::Nixl(e) => e.is_available(),
            Self::Brpc(e) => e.is_available(),
        }
    }

    pub fn stage(&self, payload: GpuTransferPayload) -> Result<StagedPayload, String> {
        match self {
            Self::Nixl(e) => e.stage(payload),
            Self::Brpc(e) => e.stage(payload),
        }
    }

    pub async fn transfer_to_peer(
        &self,
        staged: &StagedPayload,
        dest: &ExchangeDest,
        query_id: (i64, i64),
        node_id: i32,
        sender_id: i32,
    ) -> Result<(), String> {
        match self {
            Self::Nixl(e) => {
                e.transfer_to_peer(staged, dest, query_id, node_id, sender_id)
                    .await
            }
            Self::Brpc(e) => {
                e.transfer_to_peer(staged, dest, query_id, node_id, sender_id)
                    .await
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TransferDispatcher
// ---------------------------------------------------------------------------

/// Orchestrates transfer engine selection and fallback.
///
/// Tries engines in priority order (nixl first, bRPC fallback).
/// When `nixl_only` is set, bRPC fallback is disabled.
pub struct TransferDispatcher {
    engines: Vec<TransferEngine>,
    nixl_only: bool,
}

impl TransferDispatcher {
    /// Create a dispatcher with the given engines in priority order.
    pub fn new(engines: Vec<TransferEngine>, nixl_only: bool) -> Self {
        Self { engines, nixl_only }
    }

    /// Create the standard dispatcher: nixl (if available) → bRPC.
    pub fn standard(nixl_agent: Option<Arc<NixlExchange>>, nixl_only: bool) -> Self {
        let mut engines = Vec::new();
        if let Some(agent) = nixl_agent {
            engines.push(TransferEngine::Nixl(NixlTransferEngine::new(agent)));
        }
        if !nixl_only {
            engines.push(TransferEngine::Brpc(BrpcTransferEngine));
        }
        Self { engines, nixl_only }
    }

    /// Whether any engine is available.
    pub fn has_engine(&self) -> bool {
        self.engines.iter().any(|e| e.is_available())
    }

    /// Number of configured engines.
    pub fn engine_count(&self) -> usize {
        self.engines.len()
    }

    /// Stage a payload using the first available engine.
    ///
    /// Returns the staged payload and engine index (for `transfer_to_peer`).
    pub fn stage(&self, payload: GpuTransferPayload) -> Result<(StagedPayload, usize), String> {
        for (i, engine) in self.engines.iter().enumerate() {
            if engine.is_available() {
                match engine.stage(payload) {
                    Ok(staged) => {
                        info!(engine = engine.name(), "staged payload for transfer");
                        return Ok((staged, i));
                    }
                    Err(e) => {
                        warn!(engine = engine.name(), error = %e, "stage failed");
                        return Err(format!("{} stage failed: {e}", engine.name()));
                    }
                }
            }
        }
        Err("no transfer engine available".to_string())
    }

    /// Transfer staged data to a peer using the engine that staged it.
    ///
    /// On failure with the primary engine, falls back to bRPC (using IPC bytes).
    pub async fn transfer_to_peer(
        &self,
        staged: &StagedPayload,
        engine_idx: usize,
        dest: &ExchangeDest,
        query_id: (i64, i64),
        node_id: i32,
        sender_id: i32,
    ) -> Result<(), String> {
        let engine = &self.engines[engine_idx];
        match engine
            .transfer_to_peer(staged, dest, query_id, node_id, sender_id)
            .await
        {
            Ok(()) => Ok(()),
            Err(e) => {
                if self.nixl_only {
                    return Err(format!(
                        "nixl-only: {} transfer to {} failed: {e}",
                        engine.name(),
                        dest.brpc_addr
                    ));
                }
                warn!(
                    engine = engine.name(),
                    error = %e,
                    dest = %dest.brpc_addr,
                    "transfer failed, trying bRPC fallback"
                );
                crate::exchange_sender::send_exchange_result(
                    &staged.ipc_bytes,
                    &[dest.clone()],
                    query_id,
                    node_id,
                    sender_id,
                )
                .await
            }
        }
    }

    /// Send to all remote destinations with fallback.
    pub async fn send_to_all(
        &self,
        staged: &StagedPayload,
        engine_idx: usize,
        destinations: &[ExchangeDest],
        query_id: (i64, i64),
        node_id: i32,
        sender_id: i32,
    ) -> Result<(), String> {
        for dest in destinations {
            self.transfer_to_peer(staged, engine_idx, dest, query_id, node_id, sender_id)
                .await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload() -> GpuTransferPayload {
        GpuTransferPayload {
            buffers: vec![
                GpuBufferDesc {
                    addr: 0x1000,
                    len: 256,
                    device_id: 0,
                },
                GpuBufferDesc {
                    addr: 0x2000,
                    len: 512,
                    device_id: 0,
                },
            ],
            column_info: vec![("col1".to_string(), 5), ("col2".to_string(), 16)],
            column_buffers: vec![
                GpuColumnBuffers {
                    null_mask_addr: 0,
                    null_mask_len: 0,
                    offsets_addr: 0,
                    offsets_len: 0,
                    null_count: 0,
                    scale: 0,
                },
                GpuColumnBuffers {
                    null_mask_addr: 0,
                    null_mask_len: 0,
                    offsets_addr: 0,
                    offsets_len: 0,
                    null_count: 0,
                    scale: 0,
                },
            ],
            num_rows: 100,
            ipc_bytes: vec![0xAA, 0xBB, 0xCC],
        }
    }

    fn sample_dest() -> ExchangeDest {
        ExchangeDest {
            brpc_addr: "10.0.0.1:8060".to_string(),
            finst_id: (100, 200),
        }
    }

    // -- BrpcTransferEngine --

    #[test]
    fn test_brpc_engine_name() {
        assert_eq!(BrpcTransferEngine.name(), "brpc");
    }

    #[test]
    fn test_brpc_engine_always_available() {
        assert!(BrpcTransferEngine.is_available());
    }

    #[test]
    fn test_brpc_stage_noop() {
        let payload = sample_payload();
        let num_rows = payload.num_rows;
        let ipc_len = payload.ipc_bytes.len();
        let num_bufs = payload.buffers.len();

        let staged = BrpcTransferEngine
            .stage(payload)
            .expect("brpc stage never fails");
        assert_eq!(staged.num_rows, num_rows);
        assert_eq!(staged.ipc_bytes.len(), ipc_len);
        assert_eq!(staged.buffers.len(), num_bufs);
        assert!(!staged.staged);
        assert!(!staged.staged);
    }

    #[test]
    fn test_brpc_stage_preserves_data() {
        let staged = BrpcTransferEngine.stage(sample_payload()).unwrap();
        assert_eq!(staged.column_info[0].0, "col1");
        assert_eq!(staged.column_info[1].0, "col2");
        assert_eq!(staged.ipc_bytes, vec![0xAA, 0xBB, 0xCC]);
    }

    // -- NixlTransferEngine --

    #[test]
    fn test_nixl_engine_name() {
        let Some(ex) = NixlExchange::try_new("nixl-engine-name-test") else {
            return;
        };
        assert_eq!(NixlTransferEngine::new(Arc::new(ex)).name(), "nixl");
    }

    #[test]
    fn test_nixl_engine_available_by_default() {
        let Some(ex) = NixlExchange::try_new("nixl-engine-avail-test") else {
            return;
        };
        assert!(NixlTransferEngine::new(Arc::new(ex)).is_available());
    }

    #[test]
    fn test_nixl_engine_unavailable_after_disable() {
        let Some(ex) = NixlExchange::try_new("nixl-engine-disable-test") else {
            return;
        };
        let agent = Arc::new(ex);
        agent.disable_gpu_transfer();
        assert!(!NixlTransferEngine::new(agent).is_available());
    }

    // -- TransferEngine enum --

    #[test]
    fn test_engine_enum_brpc() {
        let e = TransferEngine::Brpc(BrpcTransferEngine);
        assert_eq!(e.name(), "brpc");
        assert!(e.is_available());
    }

    #[test]
    fn test_engine_enum_nixl() {
        let Some(ex) = NixlExchange::try_new("engine-enum-nixl-test") else {
            return;
        };
        let e = TransferEngine::Nixl(NixlTransferEngine::new(Arc::new(ex)));
        assert_eq!(e.name(), "nixl");
        assert!(e.is_available());
    }

    #[test]
    fn test_engine_enum_stage_brpc() {
        let e = TransferEngine::Brpc(BrpcTransferEngine);
        let staged = e.stage(sample_payload()).unwrap();
        assert_eq!(staged.num_rows, 100);
    }

    // -- TransferDispatcher --

    #[test]
    fn test_dispatcher_standard_no_nixl() {
        let d = TransferDispatcher::standard(None, false);
        assert!(d.has_engine());
        assert_eq!(d.engine_count(), 1);
    }

    #[test]
    fn test_dispatcher_standard_nixl_only_no_agent() {
        let d = TransferDispatcher::standard(None, true);
        assert!(!d.has_engine());
        assert_eq!(d.engine_count(), 0);
    }

    #[test]
    fn test_dispatcher_with_nixl() {
        let Some(ex) = NixlExchange::try_new("dispatcher-nixl-test") else {
            return;
        };
        let d = TransferDispatcher::standard(Some(Arc::new(ex)), false);
        assert!(d.has_engine());
        assert_eq!(d.engine_count(), 2);
    }

    #[test]
    fn test_dispatcher_nixl_only_with_agent() {
        let Some(ex) = NixlExchange::try_new("dispatcher-nixl-only-test") else {
            return;
        };
        let d = TransferDispatcher::standard(Some(Arc::new(ex)), true);
        assert!(d.has_engine());
        assert_eq!(d.engine_count(), 1);
    }

    #[test]
    fn test_dispatcher_stage_brpc_fallback() {
        let d = TransferDispatcher::standard(None, false);
        let (staged, idx) = d.stage(sample_payload()).expect("stage should succeed");
        assert_eq!(idx, 0);
        assert_eq!(staged.ipc_bytes, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_dispatcher_no_engines() {
        let d = TransferDispatcher::new(vec![], false);
        let result = d.stage(sample_payload());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no transfer engine available"));
    }

    #[test]
    fn test_dispatcher_stage_selects_first_available() {
        let Some(ex) = NixlExchange::try_new("dispatcher-first-avail-test") else {
            return;
        };
        let d = TransferDispatcher::standard(Some(Arc::new(ex)), false);

        // Nixl should be the first engine.
        assert_eq!(d.engines[0].name(), "nixl");
        assert_eq!(d.engines[1].name(), "brpc");
        // Don't actually stage — fake GPU addresses would SIGSEGV.
        // The engine ordering check verifies selection logic.
    }

    #[test]
    fn test_dispatcher_skips_unavailable_engine() {
        let Some(ex) = NixlExchange::try_new("dispatcher-skip-unavail-test") else {
            return;
        };
        let agent = Arc::new(ex);
        agent.disable_gpu_transfer(); // Make nixl unavailable.

        let d = TransferDispatcher::standard(Some(agent), false);

        // Should skip nixl and use bRPC (index 1).
        let (staged, idx) = d.stage(sample_payload()).expect("stage should succeed with bRPC");
        assert_eq!(idx, 1, "should fall back to bRPC");
        assert_eq!(staged.ipc_bytes, vec![0xAA, 0xBB, 0xCC]);
    }

    #[tokio::test]
    async fn test_dispatcher_brpc_transfer_no_server() {
        let d = TransferDispatcher::standard(None, false);
        let (staged, idx) = d.stage(sample_payload()).unwrap();

        let result = d
            .transfer_to_peer(&staged, idx, &sample_dest(), (1, 2), 0, 0)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("connect")
                || err.contains("Connection refused")
                || err.contains("parse IPC"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_dispatcher_send_to_all_empty_dests() {
        let d = TransferDispatcher::standard(None, false);
        let (staged, idx) = d.stage(sample_payload()).unwrap();
        let result = d.send_to_all(&staged, idx, &[], (1, 2), 0, 0).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dispatcher_send_to_all_multiple_dests() {
        let d = TransferDispatcher::standard(None, false);
        let (staged, idx) = d.stage(sample_payload()).unwrap();

        let dests = vec![
            ExchangeDest {
                brpc_addr: "10.0.0.1:8060".to_string(),
                finst_id: (1, 1),
            },
            ExchangeDest {
                brpc_addr: "10.0.0.2:8060".to_string(),
                finst_id: (2, 2),
            },
        ];

        // Both should fail (no servers) — first failure aborts.
        let result = d.send_to_all(&staged, idx, &dests, (1, 2), 0, 0).await;
        assert!(result.is_err());
    }

    // -- Integration: NixlTransferEngine stage with staging buffer --

    #[test]
    fn test_nixl_engine_stage_with_staging_buffer() {
        // Validates that the staging buffer is accessible and engine reports it.
        // Can't stage real data because sample_payload has fake GPU addresses.
        let Some(ex) = NixlExchange::try_new_with_staging(
            "nixl-stage-staging-test",
            Some(4 * 1024 * 1024),
        ) else {
            return;
        };
        let agent = Arc::new(ex);
        assert!(agent.staging().is_some());

        let engine = NixlTransferEngine::new(agent);
        assert_eq!(engine.name(), "nixl");
        assert!(engine.is_available());
    }

    #[test]
    fn test_nixl_engine_stage_without_staging_buffer() {
        let Some(ex) = NixlExchange::try_new("nixl-stage-no-staging-test") else {
            return;
        };
        let agent = Arc::new(ex);
        assert!(agent.staging().is_none());

        let engine = NixlTransferEngine::new(agent);
        assert_eq!(engine.name(), "nixl");
        assert!(engine.is_available());
    }

    // -- StagedPayload --

    #[test]
    fn test_staged_payload_debug() {
        let staged = BrpcTransferEngine.stage(sample_payload()).unwrap();
        let debug = format!("{:?}", staged);
        assert!(debug.contains("StagedPayload"));
        assert!(debug.contains("num_buffers: 2"));
        assert!(debug.contains("num_rows: 100"));
        assert!(debug.contains("staged: false"));
    }

    #[test]
    fn test_staged_payload_no_rmm_pool_field() {
        // After Phase 3 cleanup, StagedPayload should not have rmm_pool_registered.
        let staged = BrpcTransferEngine.stage(sample_payload()).unwrap();
        let debug = format!("{:?}", staged);
        assert!(!debug.contains("rmm_pool"), "rmm_pool_registered should be removed");
    }

    // -- GpuTransferPayload with sub-buffers --

    #[test]
    fn test_brpc_stage_with_sub_buffers() {
        let payload = GpuTransferPayload {
            buffers: vec![GpuBufferDesc { addr: 0x1000, len: 256, device_id: 0 }],
            column_info: vec![("col1".to_string(), 15)], // VARCHAR
            column_buffers: vec![GpuColumnBuffers {
                null_mask_addr: 0x3000,
                null_mask_len: 32,
                offsets_addr: 0x4000,
                offsets_len: 40,
                null_count: 2,
                scale: 0,
            }],
            num_rows: 10,
            ipc_bytes: vec![0xFF],
        };

        let staged = BrpcTransferEngine.stage(payload).unwrap();
        assert_eq!(staged.column_buffers[0].null_mask_addr, 0x3000);
        assert_eq!(staged.column_buffers[0].offsets_addr, 0x4000);
        assert_eq!(staged.column_buffers[0].null_count, 2);
    }

    // -- Dispatcher engine access --

    #[test]
    fn test_dispatcher_has_engine_when_empty() {
        let d = TransferDispatcher::new(vec![], false);
        assert!(!d.has_engine());
    }

    #[test]
    fn test_dispatcher_has_engine_with_brpc() {
        let d = TransferDispatcher::standard(None, false);
        assert!(d.has_engine());
    }

    #[test]
    fn test_dispatcher_stage_returns_correct_index() {
        // With only bRPC, stage should always return index 0.
        let d = TransferDispatcher::standard(None, false);
        let (_, idx) = d.stage(sample_payload()).unwrap();
        assert_eq!(idx, 0);
    }

    // -- Dispatcher with disabled nixl --

    #[test]
    fn test_dispatcher_disabled_nixl_fallback_to_brpc() {
        let Some(ex) = NixlExchange::try_new("dispatcher-disabled-test") else {
            return;
        };
        let agent = Arc::new(ex);
        agent.disable_gpu_transfer();

        let d = TransferDispatcher::standard(Some(agent), false);
        assert_eq!(d.engine_count(), 2);

        // Nixl is disabled, so should select bRPC at index 1.
        let (staged, idx) = d.stage(sample_payload()).unwrap();
        assert_eq!(idx, 1);
        assert!(!staged.staged);
    }

    #[test]
    fn test_dispatcher_disabled_nixl_only_fails() {
        let Some(ex) = NixlExchange::try_new("dispatcher-disabled-only-test") else {
            return;
        };
        let agent = Arc::new(ex);
        agent.disable_gpu_transfer();

        // nixl_only=true, but nixl is disabled → no engines available.
        let d = TransferDispatcher::standard(Some(agent), true);
        assert_eq!(d.engine_count(), 1);
        assert!(!d.has_engine());

        let result = d.stage(sample_payload());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no transfer engine available"));
    }

    // -- Empty payload --

    #[test]
    fn test_brpc_stage_empty_payload() {
        let payload = GpuTransferPayload {
            buffers: vec![],
            column_info: vec![],
            column_buffers: vec![],
            num_rows: 0,
            ipc_bytes: vec![],
        };
        let staged = BrpcTransferEngine.stage(payload).unwrap();
        assert_eq!(staged.num_rows, 0);
        assert!(staged.buffers.is_empty());
        assert!(staged.ipc_bytes.is_empty());
    }
}
