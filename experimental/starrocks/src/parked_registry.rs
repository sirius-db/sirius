//! Parked sender output keyed by destination [`SenderSlot`].
//!
//! One sender fragment parks once; output stream i belongs to `outputs[i]`. Each destination
//! claims that stream (relay or packed export) and releases when done. The fragment — and the GPU
//! batches it still holds — drops with the last claim.

use std::collections::HashMap;

use crate::fragment_executor::SenderSlot;

/// Outcome of releasing one destination's claim.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Release {
    /// Other destinations still hold the fragment.
    Outstanding(usize),
    /// Last claim: the fragment dropped.
    Freed,
}

struct Parked<F> {
    fragment: F,
    outstanding: usize,
}

/// Parked sender outputs. Generic so the engine thread can hold `sirius::Fragment<'ctx>`.
pub(crate) struct ParkedRegistry<F> {
    parked: HashMap<u64, Parked<F>>,
    slots: HashMap<SenderSlot, (u64, u64)>,
    next_id: u64,
}

impl<F> Default for ParkedRegistry<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ParkedRegistry<F> {
    pub(crate) fn new() -> Self {
        Self {
            parked: HashMap::new(),
            slots: HashMap::new(),
            next_id: 0,
        }
    }

    /// Parks once; destination i claims `(id, stream i)`. Refuses a duplicate slot before
    /// inserting anything, dropping `fragment`.
    pub(crate) fn park(&mut self, outputs: &[SenderSlot], fragment: F) -> Result<(), String> {
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
        }
        self.parked.insert(
            id,
            Parked {
                fragment,
                outstanding: outputs.len(),
            },
        );
        Ok(())
    }

    /// The parked fragment and the stream a slot names, for relay and packed export.
    pub(crate) fn claim(&mut self, slot: &SenderSlot, verb: &str) -> Result<(&mut F, u64), String> {
        let (id, stream) = self
            .slots
            .get(slot)
            .copied()
            .ok_or_else(|| format!("no parked sender output to {verb} under {slot:?}"))?;
        let entry = self
            .parked
            .get_mut(&id)
            .ok_or_else(|| format!("parked fragment vanished under {slot:?}"))?;
        Ok((&mut entry.fragment, stream))
    }

    /// One destination's release; the fragment drops with the last destination.
    pub(crate) fn release(&mut self, slot: &SenderSlot) -> Result<Release, String> {
        let (id, _) = self
            .slots
            .remove(slot)
            .ok_or_else(|| format!("no parked sender output to drop under {slot:?}"))?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result_store::FragmentInstanceId;

    fn slot(node: i32, sender: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(1, node as i64),
            node_id: node,
            sender_id: sender,
        }
    }

    #[test]
    fn last_release_drops_the_fragment() {
        let mut registry = ParkedRegistry::new();
        let a = slot(2, 0);
        let b = slot(2, 1);
        registry.park(&[a, b], "frag").unwrap();
        assert_eq!(registry.release(&a).unwrap(), Release::Outstanding(1));
        assert_eq!(registry.release(&b).unwrap(), Release::Freed);
        assert!(registry.claim(&a, "relay").is_err());
    }

    #[test]
    fn duplicate_slot_is_refused_before_parking() {
        let mut registry = ParkedRegistry::new();
        let a = slot(4, 0);
        let err = registry.park(&[a, a], "frag").unwrap_err();
        assert!(err.contains("duplicate"), "{err}");
        assert!(registry.claim(&a, "relay").is_err());
    }
}
