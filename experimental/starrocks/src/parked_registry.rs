//! Query-scoped bookkeeping of parked sender output.
//!
//! Generic over the fragment handle so the logic is unit-tested without a GPU; the engine thread
//! instantiates it with `sirius::Fragment<'ctx>`. Without the `sirius-engine` feature nothing but
//! the tests reaches the runtime API, hence the module-wide dead-code allowance for that build.
#![cfg_attr(not(feature = "sirius-engine"), allow(dead_code))]

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::fragment_executor::SenderSlot;
use crate::result_store::FragmentInstanceId;

/// The StarRocks query id every fragment instance of one query shares.
pub(crate) type QueryId = FragmentInstanceId;

/// Why a query was retired; rendered into the retire log line and the poisoned-slot message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RetireTrigger {
    /// A run of the query returned `Err` on the engine thread.
    EngineErr,
    /// A fragment of the query failed on the CN before or around the engine (translation, sink
    /// validation, remote drain).
    CnErr,
    /// `cancel_plan_fragment` with the FE's reason name (`PPlanFragmentCancelReason::as_str_name`,
    /// or "none" when the field was absent).
    Cancel(String),
}

impl fmt::Display for RetireTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EngineErr => f.write_str("engine_err"),
            Self::CnErr => f.write_str("cn_err"),
            Self::Cancel(name) => write!(f, "cancel:{name}"),
        }
    }
}

/// One sender fragment's parked output, shared by its destinations: stream i belongs to
/// destination i.
struct Parked<F> {
    fragment: F,
    /// The StarRocks query this output belongs to; `None` for unlabeled runs (test fixtures).
    query_id: Option<QueryId>,
    /// Destinations that have not yet released their stream (drained + dropped); the fragment
    /// -- and its GPU batches -- drop when it reaches zero.
    outstanding: usize,
}

/// Outcome of releasing one destination's claim.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Release {
    /// Other destinations still hold the fragment.
    Outstanding(usize),
    /// Last claim: the fragment (and its GPU batches) dropped.
    Freed,
    /// The slot was retired with its query before this release; forgotten now, so a second
    /// release is a loud error again.
    AlreadyTornDown,
}

/// What one `retire` removed.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Retired {
    pub fragments: usize,
    pub slots: usize,
}

/// Parked sender outputs keyed by the query they belong to.
///
/// Parked ONCE per fragment; `slots` maps each destination's [`SenderSlot`] to
/// `(park id, its output stream)`. A query's outputs leave together (`retire`), never as
/// collateral of another query's failure.
pub(crate) struct ParkedRegistry<F> {
    parked: HashMap<u64, Parked<F>>,
    slots: HashMap<SenderSlot, (u64, u64)>,
    /// Why a slot's output went away, so the next export/relay/drop of it names the query's real
    /// error instead of the generic message (the TPC-H q08 lesson: the collateral "no parked
    /// sender output" masked an OOM in HASH_JOIN for weeks). Bounded FIFO; an entry is removed
    /// when the slot is parked again or released once.
    torn_down: HashMap<SenderSlot, String>,
    torn_down_order: VecDeque<SenderSlot>,
    next_id: u64,
}

impl<F> Default for ParkedRegistry<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ParkedRegistry<F> {
    pub const TORN_DOWN_CAPACITY: usize = 4096;

    pub fn new() -> Self {
        Self {
            parked: HashMap::new(),
            slots: HashMap::new(),
            torn_down: HashMap::new(),
            torn_down_order: VecDeque::new(),
            next_id: 0,
        }
    }

    /// Parks once; destination i claims `(id, stream i)`. Refuses a duplicate slot -- within
    /// `outputs` or against a slot already parked -- before inserting anything, dropping
    /// `fragment`. Re-parking a slot forgets its torn-down entry.
    pub fn park(
        &mut self,
        query_id: Option<QueryId>,
        outputs: &[SenderSlot],
        fragment: F,
    ) -> Result<(), String> {
        for (index, slot) in outputs.iter().enumerate() {
            if self.slots.contains_key(slot) || outputs[..index].contains(slot) {
                return Err(format!(
                    "duplicate destination slot {slot:?} in one sender fan-out"
                ));
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        for (stream, slot) in outputs.iter().enumerate() {
            self.slots.insert(*slot, (id, stream as u64));
            self.forget_torn_down(slot);
        }
        self.parked.insert(
            id,
            Parked {
                fragment,
                query_id,
                outstanding: outputs.len(),
            },
        );
        Ok(())
    }

    /// The parked fragment and the stream a slot names, for relay and export. `verb` is
    /// "relay" | "export" for the missing-slot message.
    pub fn claim(&mut self, slot: &SenderSlot, verb: &str) -> Result<(&mut F, u64), String> {
        let (id, stream) = self
            .slots
            .get(slot)
            .copied()
            .ok_or_else(|| self.missing(slot, verb))?;
        let entry = self
            .parked
            .get_mut(&id)
            .ok_or_else(|| format!("parked fragment vanished under {slot:?}"))?;
        Ok((&mut entry.fragment, stream))
    }

    /// Read-only lookup for the cardinality declaration.
    pub fn peek(&self, slot: &SenderSlot) -> Option<(&F, u64)> {
        let (id, stream) = self.slots.get(slot).copied()?;
        Some((&self.parked.get(&id)?.fragment, stream))
    }

    /// One destination's release; the fragment (and the GPU memory its remaining batches hold)
    /// drops with the LAST destination. A live slot released twice is a loud error; a slot
    /// retired with its query returns [`Release::AlreadyTornDown`] exactly once.
    pub fn release(&mut self, slot: &SenderSlot) -> Result<Release, String> {
        let Some((id, _)) = self.slots.remove(slot) else {
            if self.forget_torn_down(slot) {
                return Ok(Release::AlreadyTornDown);
            }
            return Err(self.missing(slot, "drop"));
        };
        let entry = self
            .parked
            .get_mut(&id)
            .ok_or_else(|| format!("parked fragment vanished under {slot:?}"))?;
        entry.outstanding -= 1;
        if entry.outstanding == 0 {
            self.parked.remove(&id);
            return Ok(Release::Freed);
        }
        Ok(Release::Outstanding(entry.outstanding))
    }

    /// Drops every parked fragment of `query_id` (`None` matches only unlabeled output), records
    /// `cause` for each of its slots, returns the counts. Idempotent: a second call returns zeros.
    pub fn retire(&mut self, query_id: Option<QueryId>, cause: &str) -> Retired {
        let ids: HashSet<u64> = self
            .parked
            .iter()
            .filter(|(_, parked)| parked.query_id == query_id)
            .map(|(id, _)| *id)
            .collect();
        if ids.is_empty() {
            return Retired::default();
        }
        let mut torn = Vec::new();
        self.slots.retain(|slot, (id, _)| {
            if ids.contains(id) {
                torn.push(*slot);
                false
            } else {
                true
            }
        });
        for slot in &torn {
            self.remember_torn_down(*slot, cause);
        }
        for id in &ids {
            self.parked.remove(id);
        }
        Retired {
            fragments: ids.len(),
            slots: torn.len(),
        }
    }

    pub fn fragments(&self) -> usize {
        self.parked.len()
    }

    #[cfg(test)]
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// The error for a slot whose parked output is gone: the query's real error when it was
    /// retired, the generic message otherwise.
    fn missing(&self, slot: &SenderSlot, verb: &str) -> String {
        match self.torn_down.get(slot) {
            Some(cause) => format!(
                "sender output for {slot:?} was discarded when its query was retired: {cause}"
            ),
            None => format!("no parked sender output to {verb} for {slot:?}"),
        }
    }

    fn remember_torn_down(&mut self, slot: SenderSlot, cause: &str) {
        if self.torn_down.insert(slot, cause.to_string()).is_none() {
            self.torn_down_order.push_back(slot);
        }
        while self.torn_down.len() > Self::TORN_DOWN_CAPACITY {
            match self.torn_down_order.pop_front() {
                Some(oldest) => {
                    self.torn_down.remove(&oldest);
                }
                None => break,
            }
        }
    }

    /// Returns whether the slot had a torn-down entry.
    fn forget_torn_down(&mut self, slot: &SenderSlot) -> bool {
        if self.torn_down.remove(slot).is_none() {
            return false;
        }
        self.torn_down_order.retain(|entry| entry != slot);
        true
    }
}

/// Queries this CN has declared over. A `Run` already queued for one of them is refused instead of
/// re-parking output nobody will consume. Bounded FIFO: StarRocks query ids never recur, so
/// eviction only forgets an explanation, never a live query. First cause wins (matches
/// `ResultStore::fail_query`).
#[derive(Debug, Default)]
pub(crate) struct RetiredQueries {
    order: VecDeque<QueryId>,
    causes: HashMap<QueryId, String>,
}

impl RetiredQueries {
    pub const CAPACITY: usize = 1024;

    /// Returns `false` when the query was already marked (the earlier cause is kept).
    pub fn mark(&mut self, query_id: QueryId, cause: &str) -> bool {
        if self.causes.contains_key(&query_id) {
            return false;
        }
        self.causes.insert(query_id, cause.to_string());
        self.order.push_back(query_id);
        while self.causes.len() > Self::CAPACITY {
            match self.order.pop_front() {
                Some(oldest) => {
                    self.causes.remove(&oldest);
                }
                None => break,
            }
        }
        true
    }

    pub fn cause(&self, query_id: QueryId) -> Option<&str> {
        self.causes.get(&query_id).map(String::as_str)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.causes.len()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;
    use crate::fragment_executor::SenderSlot;
    use crate::result_store::FragmentInstanceId;

    /// A fragment stand-in that counts its own drops.
    #[derive(Debug)]
    struct Counted(Rc<Cell<usize>>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    fn slot(receiver: i64, node_id: i32, sender_id: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(1, receiver),
            node_id,
            sender_id,
        }
    }

    fn query(hi: i64) -> QueryId {
        FragmentInstanceId::from_halves(hi, 0)
    }

    #[test]
    fn park_then_release_last_claim_drops_the_fragment() {
        let drops = Rc::new(Cell::new(0));
        let mut registry = ParkedRegistry::new();
        let (a, b) = (slot(1, 7, 0), slot(2, 7, 0));
        registry
            .park(Some(query(1)), &[a, b], Counted(drops.clone()))
            .unwrap();
        assert_eq!(registry.fragments(), 1);
        assert_eq!(registry.slots(), 2);

        assert_eq!(registry.release(&a).unwrap(), Release::Outstanding(1));
        assert_eq!(drops.get(), 0, "one destination still holds the fragment");
        assert_eq!(registry.release(&b).unwrap(), Release::Freed);
        assert_eq!(drops.get(), 1, "the last claim drops the fragment");
        assert_eq!(registry.fragments(), 0);
        assert_eq!(registry.slots(), 0);
    }

    #[test]
    fn second_release_of_a_live_slot_is_a_loud_error() {
        let mut registry = ParkedRegistry::new();
        let a = slot(1, 7, 0);
        registry
            .park(Some(query(1)), &[a], Counted(Rc::default()))
            .unwrap();
        assert_eq!(registry.release(&a).unwrap(), Release::Freed);
        let err = registry.release(&a).unwrap_err();
        assert!(err.contains("no parked sender output to drop"), "{err}");
    }

    #[test]
    fn duplicate_slot_is_refused_before_anything_is_inserted() {
        let drops = Rc::new(Cell::new(0));
        let mut registry = ParkedRegistry::new();
        let a = slot(1, 7, 0);
        let err = registry
            .park(Some(query(1)), &[a, a], Counted(drops.clone()))
            .unwrap_err();
        assert!(err.contains("duplicate destination slot"), "{err}");
        assert_eq!(registry.slots(), 0);
        assert_eq!(registry.fragments(), 0);
        assert_eq!(
            drops.get(),
            1,
            "the refused fragment is dropped, not leaked"
        );

        // The same rule against a slot another fan-out already holds.
        registry
            .park(Some(query(1)), &[a], Counted(drops.clone()))
            .unwrap();
        let err = registry
            .park(Some(query(2)), &[slot(9, 7, 0), a], Counted(drops.clone()))
            .unwrap_err();
        assert!(err.contains("duplicate destination slot"), "{err}");
        assert_eq!(registry.slots(), 1);
        assert_eq!(registry.fragments(), 1);
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn retire_drops_only_that_query_and_poisons_its_slots() {
        let drops = Rc::new(Cell::new(0));
        let mut registry = ParkedRegistry::new();
        let (a1, a2) = (slot(1, 7, 0), slot(2, 7, 0));
        let (b1, b2, b3) = (slot(3, 9, 0), slot(4, 9, 0), slot(5, 9, 0));
        registry
            .park(Some(query(1)), &[a1, a2], Counted(drops.clone()))
            .unwrap();
        registry
            .park(Some(query(2)), &[b1, b2, b3], Counted(drops.clone()))
            .unwrap();

        let retired = registry.retire(Some(query(1)), "HASH_JOIN ran out of memory");
        assert_eq!(
            retired,
            Retired {
                fragments: 1,
                slots: 2
            }
        );
        assert_eq!(drops.get(), 1, "only A's fragment dropped");
        assert_eq!(registry.fragments(), 1);
        assert_eq!(registry.slots(), 3);

        // B is untouched.
        let (_, stream) = registry.claim(&b2, "relay").unwrap();
        assert_eq!(stream, 1);
        assert!(registry.peek(&b3).is_some());

        // A's slots name the query's real error instead of the generic message.
        let err = registry.claim(&a1, "export").unwrap_err();
        assert!(
            err.contains("was discarded when its query was retired: HASH_JOIN ran out of memory"),
            "{err}"
        );
        assert!(registry.peek(&a1).is_none());

        // Idempotent.
        assert_eq!(registry.retire(Some(query(1)), "again"), Retired::default());
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn retire_of_the_unlabeled_bucket_leaves_labeled_output_alone() {
        let drops = Rc::new(Cell::new(0));
        let mut registry = ParkedRegistry::new();
        let (u, l) = (slot(1, 7, 0), slot(2, 7, 0));
        registry.park(None, &[u], Counted(drops.clone())).unwrap();
        registry
            .park(Some(query(1)), &[l], Counted(drops.clone()))
            .unwrap();

        let retired = registry.retire(None, "fixture failed");
        assert_eq!(
            retired,
            Retired {
                fragments: 1,
                slots: 1
            }
        );
        assert_eq!(drops.get(), 1);
        assert!(registry.claim(&u, "relay").is_err());
        assert!(registry.claim(&l, "relay").is_ok());
    }

    #[test]
    fn release_after_retire_is_ok_exactly_once() {
        let mut registry = ParkedRegistry::new();
        let a = slot(1, 7, 0);
        registry
            .park(Some(query(1)), &[a], Counted(Rc::default()))
            .unwrap();
        registry.retire(Some(query(1)), "boom");

        assert_eq!(registry.release(&a).unwrap(), Release::AlreadyTornDown);
        let err = registry.release(&a).unwrap_err();
        assert!(err.contains("no parked sender output to drop"), "{err}");
    }

    #[test]
    fn repark_of_a_retired_slot_forgets_its_torn_down_entry() {
        let mut registry = ParkedRegistry::new();
        let a = slot(1, 7, 0);
        registry
            .park(Some(query(1)), &[a], Counted(Rc::default()))
            .unwrap();
        registry.retire(Some(query(1)), "boom");

        registry
            .park(Some(query(3)), &[a], Counted(Rc::default()))
            .unwrap();
        assert!(registry.claim(&a, "relay").is_ok());
        assert_eq!(registry.release(&a).unwrap(), Release::Freed);
        // No stale cause survives the re-park: the slot is live-then-freed, so the message is
        // the generic one, not the old query's error.
        let err = registry.claim(&a, "export").unwrap_err();
        assert!(err.contains("no parked sender output to export"), "{err}");
        assert!(!err.contains("boom"), "{err}");
    }

    #[test]
    fn torn_down_is_bounded() {
        let mut registry = ParkedRegistry::new();
        let capacity = ParkedRegistry::<Counted>::TORN_DOWN_CAPACITY;
        let first = slot(0, 7, 0);
        for i in 0..=capacity {
            let s = slot(i as i64, 7, 0);
            registry
                .park(Some(query(i as i64)), &[s], Counted(Rc::default()))
                .unwrap();
            registry.retire(Some(query(i as i64)), "boom");
        }
        // The oldest explanation was evicted; the newest is still there.
        let err = registry.claim(&first, "export").unwrap_err();
        assert!(err.contains("no parked sender output to export"), "{err}");
        let err = registry
            .claim(&slot(capacity as i64, 7, 0), "export")
            .unwrap_err();
        assert!(
            err.contains("was discarded when its query was retired"),
            "{err}"
        );
    }

    #[test]
    fn retired_queries_keep_the_first_cause_and_evict_fifo() {
        let mut retired = RetiredQueries::default();
        assert!(retired.mark(query(1), "a"));
        assert!(!retired.mark(query(1), "b"));
        assert_eq!(retired.cause(query(1)), Some("a"));
        assert_eq!(retired.cause(query(2)), None);
        assert_eq!(retired.len(), 1);

        for i in 2..=(RetiredQueries::CAPACITY as i64 + 1) {
            assert!(retired.mark(query(i), "x"));
        }
        assert_eq!(retired.len(), RetiredQueries::CAPACITY);
        assert_eq!(retired.cause(query(1)), None, "the first mark was evicted");
        assert_eq!(
            retired.cause(query(RetiredQueries::CAPACITY as i64 + 1)),
            Some("x")
        );
    }
}
