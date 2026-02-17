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

/// Key identifying an exchange stream: (query_id, dest_node_id).
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

/// Concurrent buffer for exchange data arriving from multiple senders.
#[derive(Clone)]
pub struct ExchangeBuffer {
    entries: Arc<DashMap<ExchangeKey, ExchangeEntry>>,
}

impl ExchangeBuffer {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
        }
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
                entry.notify.notify_waiters();
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
                entry.notify.notify_waiters();
            }
        }
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
}
