//! Concurrent buffer for exchange data (PBlocks) arriving via `transmit_block`.
//!
//! Regular Doris BEs scan tablets and send data to the GPU BE via the
//! `transmit_block` gRPC method. This module buffers incoming PBlocks
//! per (query_id, node_id) and signals when all senders have finished (EOS).

use std::collections::HashSet;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::Notify;

use doris_proto::doris::PBlock;
use sirius_ffi::{ExchangeArtifact, PackedBroadcastEntry, PackedPartition};

/// Key identifying an exchange stream: (query_id, dest_node_id).
///
/// Assumption: each (query_id, node_id) pair uniquely identifies an exchange stream.
/// This holds as long as FE sends `parallel_instances = 1` per fragment. If FE ever
/// sends `parallel_instances > 1`, the key would need a fragment-instance dimension.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct ExchangeKey {
    /// query_id as (hi, lo) pair.
    pub query_id: (i64, i64),
    /// Destination EXCHANGE_NODE id.
    pub node_id: i32,
}

struct ExchangeEntry {
    blocks: Vec<PBlock>,
    /// Sender IDs that have sent EOS.
    eos_senders: HashSet<i32>,
    /// Expected number of senders (0 = unknown / auto-detect on first EOS).
    expected_senders: u32,
    /// Signalled when all senders have sent EOS.
    notify: Arc<Notify>,
}

/// GPU-side packed buffer info for direct table registration.
/// Stored by transfer_complete when cudf packed transfer is used.
pub struct PackedGpuExchange {
    /// GPU address of the packed buffer (in receiver's staging region).
    pub gpu_addr: usize,
    /// Size of the packed buffer in bytes.
    pub gpu_size: usize,
    /// cudf::pack() metadata for cudf::unpack() on receiver.
    pub cudf_metadata: Vec<u8>,
    /// Whether the sender already packed this table in receiver/projected order.
    pub projection_already_applied: bool,
    /// Staging lease keeping the RECV buffer region alive until the exchange
    /// fragment finishes processing. Dropped when ExchangeBuffer entry is consumed.
    pub _staging_lease: Option<crate::gpu_staging_buffer::StagingLease>,
}

impl std::fmt::Debug for PackedGpuExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedGpuExchange")
            .field("gpu_addr", &format_args!("0x{:x}", self.gpu_addr))
            .field("gpu_size", &self.gpu_size)
            .field(
                "projection_already_applied",
                &self.projection_already_applied,
            )
            .field("has_lease", &self._staging_lease.is_some())
            .finish()
    }
}

/// Local same-process GPU exchange payload.
///
/// This is the target ownership model for all-local exchange: the artifact owns
/// the underlying C++ session and GPU resources, while the copied descriptors
/// give Rust stable access to the packed layout without another control-plane
/// lookup.
pub struct LocalGpuArtifact {
    _artifact: Option<Arc<ExchangeArtifact>>,
    ipc_bytes: Arc<Vec<u8>>,
    staging_base: usize,
    packed_partitions: Vec<PackedPartition>,
    packed_broadcast: Vec<PackedBroadcastEntry>,
    projection_already_applied: bool,
}

impl LocalGpuArtifact {
    pub fn from_exchange_artifact(artifact: ExchangeArtifact, ipc_bytes: Vec<u8>) -> Self {
        let artifact = Arc::new(artifact);
        let ipc_bytes = Arc::new(ipc_bytes);
        let partition_indices: Vec<usize> = (0..artifact.packed_partitions().len()).collect();
        let broadcast_indices: Vec<usize> = (0..artifact.packed_broadcast().len()).collect();
        Self::select_from_exchange_artifact(
            artifact,
            ipc_bytes,
            &partition_indices,
            &broadcast_indices,
        )
    }

    pub fn select_from_exchange_artifact(
        artifact: Arc<ExchangeArtifact>,
        ipc_bytes: Arc<Vec<u8>>,
        partition_indices: &[usize],
        broadcast_indices: &[usize],
    ) -> Self {
        Self {
            ipc_bytes,
            staging_base: artifact.staging_base(),
            packed_partitions: partition_indices
                .iter()
                .filter_map(|&idx| artifact.packed_partitions().get(idx).cloned())
                .collect(),
            packed_broadcast: broadcast_indices
                .iter()
                .filter_map(|&idx| artifact.packed_broadcast().get(idx).cloned())
                .collect(),
            projection_already_applied: artifact.projection_already_applied(),
            _artifact: Some(artifact),
        }
    }

    pub fn staging_base(&self) -> usize {
        self.staging_base
    }

    pub fn packed_partitions(&self) -> &[PackedPartition] {
        &self.packed_partitions
    }

    pub fn packed_broadcast(&self) -> &[PackedBroadcastEntry] {
        &self.packed_broadcast
    }

    pub fn ipc_bytes(&self) -> &[u8] {
        self.ipc_bytes.as_slice()
    }

    pub fn has_exchange_data(&self) -> bool {
        !self.packed_partitions.is_empty() || !self.packed_broadcast.is_empty()
    }

    pub fn projection_already_applied(&self) -> bool {
        self.projection_already_applied
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        staging_base: usize,
        packed_partition_count: usize,
        packed_broadcast_count: usize,
    ) -> Self {
        Self {
            _artifact: None,
            ipc_bytes: Arc::new(Vec::new()),
            staging_base,
            packed_partitions: (0..packed_partition_count)
                .map(|i| PackedPartition {
                    partition_id: i,
                    staging_offset: i * 64,
                    packed_size: 64,
                    metadata: Vec::new(),
                    num_rows: 1,
                    overflow_gpu_addr: 0,
                })
                .collect(),
            packed_broadcast: (0..packed_broadcast_count)
                .map(|i| PackedBroadcastEntry {
                    entry_id: i,
                    staging_offset: i * 64,
                    packed_size: 64,
                    metadata: Vec::new(),
                    num_rows: 1,
                    overflow_gpu_addr: 0,
                })
                .collect(),
            projection_already_applied: false,
        }
    }
}

impl std::fmt::Debug for LocalGpuArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalGpuArtifact")
            .field("staging_base", &format_args!("0x{:x}", self.staging_base))
            .field("packed_partitions", &self.packed_partitions.len())
            .field("packed_broadcast", &self.packed_broadcast.len())
            .field(
                "projection_already_applied",
                &self.projection_already_applied,
            )
            .finish()
    }
}

/// Concurrent buffer for exchange data arriving from multiple senders.
#[derive(Clone)]
pub struct ExchangeBuffer {
    entries: Arc<DashMap<ExchangeKey, ExchangeEntry>>,
    /// Tracks cancelled query IDs so async tasks can detect cancellation.
    cancelled: Arc<DashMap<(i64, i64), ()>>,
    /// Packed GPU exchange data: when cudf packed transfer is used, the receiver
    /// stores the packed buffer info here. The exchange async task retrieves it
    /// and calls gpu_register_table to make the data available to DuckDB on GPU.
    packed_gpu: Arc<DashMap<ExchangeKey, Vec<PackedGpuExchange>>>,
    /// Local same-process GPU exchange payloads.
    ///
    /// This is not wired into execution yet; the immediate goal is to give the
    /// runtime a typed home for owned local GPU artifacts so the zero-copy
    /// exchange redesign can replace the current CPU/PBlock fallback without
    /// adding another global lifetime path.
    local_gpu: Arc<DashMap<ExchangeKey, Vec<LocalGpuArtifact>>>,
}

impl ExchangeBuffer {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            cancelled: Arc::new(DashMap::new()),
            packed_gpu: Arc::new(DashMap::new()),
            local_gpu: Arc::new(DashMap::new()),
        }
    }

    /// Store packed GPU exchange data for an exchange key (accumulates from multiple senders).
    pub fn store_packed_gpu(&self, key: ExchangeKey, data: PackedGpuExchange) {
        tracing::info!(
            query_id = ?(key.query_id),
            node_id = key.node_id,
            gpu_addr = format_args!("0x{:x}", data.gpu_addr),
            gpu_size = data.gpu_size,
            projection_already_applied = data.projection_already_applied,
            has_lease = data._staging_lease.is_some(),
            "store_packed_gpu"
        );
        self.packed_gpu
            .entry(key)
            .or_insert_with(Vec::new)
            .push(data);
    }

    /// Take all packed GPU exchange data for an exchange key.
    pub fn take_packed_gpu(&self, key: &ExchangeKey) -> Option<Vec<PackedGpuExchange>> {
        self.packed_gpu
            .remove(key)
            .map(|(_, v)| v)
            .filter(|v| !v.is_empty())
    }

    /// Store local same-process GPU exchange payload for an exchange key.
    pub fn store_local_gpu_artifact(&self, key: ExchangeKey, artifact: LocalGpuArtifact) {
        tracing::info!(
            query_id = ?(key.query_id),
            node_id = key.node_id,
            staging_base = format_args!("0x{:x}", artifact.staging_base()),
            packed_partitions = artifact.packed_partitions().len(),
            packed_broadcast = artifact.packed_broadcast().len(),
            "store_local_gpu_artifact"
        );
        self.local_gpu
            .entry(key)
            .or_insert_with(Vec::new)
            .push(artifact);
    }

    /// Take all local same-process GPU payloads for an exchange key.
    pub fn take_local_gpu_artifacts(&self, key: &ExchangeKey) -> Option<Vec<LocalGpuArtifact>> {
        self.local_gpu
            .remove(key)
            .map(|(_, v)| v)
            .filter(|v| !v.is_empty())
    }

    /// Check if a query has been cancelled.
    pub fn is_cancelled(&self, query_hi: i64, query_lo: i64) -> bool {
        self.cancelled.contains_key(&(query_hi, query_lo))
    }

    /// Register an exchange before data arrives.
    ///
    /// Returns an `Arc<Notify>` that will be notified when all senders EOS.
    pub fn register(&self, key: ExchangeKey, expected_senders: u32) -> Arc<Notify> {
        let mut entry = self.entries.entry(key).or_insert_with(|| ExchangeEntry {
            blocks: Vec::new(),
            eos_senders: HashSet::new(),
            expected_senders,
            notify: Arc::new(Notify::new()),
        });
        // If the entry already existed (race: transmit_block arrived first),
        // update expected_senders if it was 0.
        if entry.expected_senders == 0 && expected_senders > 0 {
            entry.expected_senders = expected_senders;
        }
        entry.notify.clone()
    }

    /// Add a block from a sender. If `eos` is true, marks this sender as done.
    ///
    /// Returns `true` when all expected senders have sent EOS.
    /// Auto-creates the entry if `transmit_block` arrives before `register`.
    pub fn add_block(
        &self,
        key: &ExchangeKey,
        sender_id: i32,
        block: Option<PBlock>,
        eos: bool,
    ) -> bool {
        let mut entry = self
            .entries
            .entry(key.clone())
            .or_insert_with(|| ExchangeEntry {
                blocks: Vec::new(),
                eos_senders: HashSet::new(),
                expected_senders: 0,
                notify: Arc::new(Notify::new()),
            });

        if let Some(b) = block {
            entry.blocks.push(b);
        }

        if eos {
            entry.eos_senders.insert(sender_id);
            // With expected_senders == 0, we can't know when all senders are done.
            // The exec_plan_fragment side will call `register` with the correct count.
            // Once expected_senders is set and all have EOS'd, notify.
            if entry.expected_senders > 0
                && entry.eos_senders.len() >= entry.expected_senders as usize
            {
                entry.notify.notify_one();
                return true;
            }
        }

        false
    }

    /// Set or update the expected sender count for a key.
    ///
    /// If all expected senders have already EOS'd, notifies immediately.
    pub fn set_expected_senders(&self, key: &ExchangeKey, expected_senders: u32) {
        if let Some(mut entry) = self.entries.get_mut(key) {
            entry.expected_senders = expected_senders;
            if entry.eos_senders.len() >= expected_senders as usize {
                entry.notify.notify_one();
            }
        }
    }

    /// Cancel all exchange entries for a query.
    ///
    /// Marks the query as cancelled, removes all entries, and notifies any waiters
    /// so async exchange tasks unblock and can check `is_cancelled()`.
    pub fn cancel_query(&self, query_hi: i64, query_lo: i64) {
        self.cancelled.insert((query_hi, query_lo), ());
        let mut keys_to_remove: Vec<ExchangeKey> = self
            .entries
            .iter()
            .filter(|e| e.key().query_id == (query_hi, query_lo))
            .map(|e| e.key().clone())
            .collect();
        keys_to_remove.extend(
            self.packed_gpu
                .iter()
                .filter(|e| e.key().query_id == (query_hi, query_lo))
                .map(|e| e.key().clone()),
        );
        keys_to_remove.extend(
            self.local_gpu
                .iter()
                .filter(|e| e.key().query_id == (query_hi, query_lo))
                .map(|e| e.key().clone()),
        );
        let mut seen = HashSet::new();
        keys_to_remove.retain(|key| seen.insert(key.clone()));
        for key in &keys_to_remove {
            if let Some((_, entry)) = self.entries.remove(key) {
                // Wake any waiting tasks so they see the entry is gone.
                entry.notify.notify_one();
            }
            self.packed_gpu.remove(key);
            self.local_gpu.remove(key);
        }
    }

    /// Clear the cancelled flag for a query (e.g. after cleanup is done).
    pub fn clear_cancelled(&self, query_hi: i64, query_lo: i64) {
        self.cancelled.remove(&(query_hi, query_lo));
    }

    /// Take all buffered blocks for a key, removing the entry.
    pub fn take(&self, key: &ExchangeKey) -> Vec<PBlock> {
        self.entries
            .remove(key)
            .map(|(_, entry)| entry.blocks)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(hi: i64, lo: i64, node: i32) -> ExchangeKey {
        ExchangeKey {
            query_id: (hi, lo),
            node_id: node,
        }
    }

    fn empty_block() -> PBlock {
        PBlock {
            column_metas: vec![],
            column_values: None,
            compressed: None,
            uncompressed_size: None,
            compression_type: None,
            be_exec_version: None,
        }
    }

    #[test]
    fn test_register_and_take() {
        let buf = ExchangeBuffer::new();
        let k = key(1, 2, 3);
        let _notify = buf.register(k.clone(), 1);

        buf.add_block(&k, 0, Some(empty_block()), false);
        buf.add_block(&k, 0, None, true);

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);

        // After take, key is removed.
        let blocks2 = buf.take(&k);
        assert!(blocks2.is_empty());
    }

    #[test]
    fn test_multiple_senders_eos() {
        let buf = ExchangeBuffer::new();
        let k = key(10, 20, 1);
        let _notify = buf.register(k.clone(), 3);

        // Sender 0: 2 blocks + EOS
        buf.add_block(&k, 0, Some(empty_block()), false);
        buf.add_block(&k, 0, Some(empty_block()), false);
        assert!(!buf.add_block(&k, 0, None, true));

        // Sender 1: 1 block + EOS
        buf.add_block(&k, 1, Some(empty_block()), false);
        assert!(!buf.add_block(&k, 1, None, true));

        // Sender 2: EOS only — this completes
        assert!(buf.add_block(&k, 2, None, true));

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 3); // 2 from sender 0 + 1 from sender 1
    }

    #[test]
    fn test_add_block_before_register() {
        // Simulates transmit_block arriving before exec_plan_fragment.
        let buf = ExchangeBuffer::new();
        let k = key(5, 6, 7);

        // Data arrives before registration.
        buf.add_block(&k, 0, Some(empty_block()), false);
        buf.add_block(&k, 0, None, true);

        // Now register with expected_senders=1.
        // Since sender 0 already EOS'd and expected=1, the notify fires.
        let notify = buf.register(k.clone(), 1);
        // The entry already has 1 EOS sender, so set_expected_senders
        // should see it's complete.
        buf.set_expected_senders(&k, 1);

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);
        let _ = notify;
    }

    #[tokio::test]
    async fn test_notify_fires_on_completion() {
        let buf = ExchangeBuffer::new();
        let k = key(100, 200, 5);
        let notify = buf.register(k.clone(), 1);

        let buf2 = buf.clone();
        let k2 = k.clone();
        let handle = tokio::spawn(async move {
            notify.notified().await;
            buf2.take(&k2)
        });

        // Small delay, then send data + EOS.
        tokio::task::yield_now().await;
        buf.add_block(&k, 0, Some(empty_block()), false);
        buf.add_block(&k, 0, None, true);

        let blocks = handle.await.unwrap();
        assert_eq!(blocks.len(), 1);
    }

    #[tokio::test]
    async fn test_notify_one_fires_even_when_data_arrives_before_await() {
        // Regression: notify_waiters() lost notifications when no task was waiting.
        // notify_one() stores a permit so the next notified().await resolves immediately.
        let buf = ExchangeBuffer::new();
        let k = key(42, 43, 1);
        let notify = buf.register(k.clone(), 1);

        // Data + EOS arrive BEFORE anyone calls notified().await.
        buf.add_block(&k, 0, Some(empty_block()), false);
        let done = buf.add_block(&k, 0, None, true);
        assert!(done, "should signal all_done");

        // Now await — should resolve immediately because notify_one stored a permit.
        tokio::time::timeout(std::time::Duration::from_millis(100), notify.notified())
            .await
            .expect("notified() should resolve immediately from stored permit");

        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);
    }

    #[tokio::test]
    async fn test_two_exchange_nodes_both_notify() {
        // Simulate two exchange nodes for the same query (UNION ALL pattern).
        let buf = ExchangeBuffer::new();
        let k1 = key(10, 20, 1);
        let k2 = key(10, 20, 3);
        let notify1 = buf.register(k1.clone(), 1);
        let notify2 = buf.register(k2.clone(), 1);

        // Data arrives for both nodes.
        buf.add_block(&k1, 0, Some(empty_block()), false);
        buf.add_block(&k1, 0, None, true);
        buf.add_block(&k2, 0, Some(empty_block()), false);
        buf.add_block(&k2, 0, None, true);

        // Both should resolve.
        tokio::time::timeout(std::time::Duration::from_millis(100), notify1.notified())
            .await
            .expect("notify1 should resolve");
        tokio::time::timeout(std::time::Duration::from_millis(100), notify2.notified())
            .await
            .expect("notify2 should resolve");

        assert_eq!(buf.take(&k1).len(), 1);
        assert_eq!(buf.take(&k2).len(), 1);
    }

    #[test]
    fn test_set_expected_senders_triggers_notify_when_already_done() {
        let buf = ExchangeBuffer::new();
        let k = key(1, 2, 3);

        // Data arrives before register (expected_senders = 0).
        buf.add_block(&k, 0, Some(empty_block()), false);
        buf.add_block(&k, 0, None, true); // all_done = false (expected_senders = 0)

        // Now register with expected_senders = 1.
        let _notify = buf.register(k.clone(), 1);

        // set_expected_senders should trigger notify since sender already EOS'd.
        buf.set_expected_senders(&k, 1);

        // Data should be available.
        let blocks = buf.take(&k);
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_cancel_query_removes_entries() {
        let buf = ExchangeBuffer::new();
        let k1 = key(10, 20, 1);
        let k2 = key(10, 20, 3);
        let k_other = key(99, 88, 1);

        buf.register(k1.clone(), 1);
        buf.register(k2.clone(), 1);
        buf.register(k_other.clone(), 1);

        buf.add_block(&k1, 0, Some(empty_block()), false);
        buf.add_block(&k2, 0, Some(empty_block()), false);
        buf.add_block(&k_other, 0, Some(empty_block()), false);

        // Cancel query (10, 20) — should remove k1 and k2, keep k_other.
        buf.cancel_query(10, 20);

        assert!(buf.take(&k1).is_empty());
        assert!(buf.take(&k2).is_empty());
        assert_eq!(buf.take(&k_other).len(), 1);
    }

    #[test]
    fn test_store_and_take_local_gpu_artifacts() {
        let buf = ExchangeBuffer::new();
        let k = key(7, 8, 9);

        buf.store_local_gpu_artifact(k.clone(), LocalGpuArtifact::for_test(0x1000, 2, 1));

        let artifacts = buf
            .take_local_gpu_artifacts(&k)
            .expect("local gpu artifact should be stored");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].staging_base(), 0x1000);
        assert_eq!(artifacts[0].packed_partitions().len(), 2);
        assert_eq!(artifacts[0].packed_broadcast().len(), 1);
        assert!(artifacts[0].has_exchange_data());
        assert!(buf.take_local_gpu_artifacts(&k).is_none());
    }

    #[test]
    fn test_cancel_query_removes_packed_and_local_gpu_entries() {
        let buf = ExchangeBuffer::new();
        let k = key(50, 60, 3);

        buf.store_packed_gpu(
            k.clone(),
            PackedGpuExchange {
                gpu_addr: 0x1234,
                gpu_size: 64,
                cudf_metadata: vec![1, 2, 3],
                projection_already_applied: false,
                _staging_lease: None,
            },
        );
        buf.store_local_gpu_artifact(k.clone(), LocalGpuArtifact::for_test(0x2000, 1, 0));

        buf.cancel_query(50, 60);

        assert!(buf.take_packed_gpu(&k).is_none());
        assert!(buf.take_local_gpu_artifacts(&k).is_none());
    }
}
