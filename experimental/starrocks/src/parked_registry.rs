//! Parked sender output keyed by destination [`SenderSlot`].
//!
//! One sender fragment parks once; output stream i belongs to `outputs[i]`. Each destination
//! claims that stream and releases when done. The fragment — and the GPU batches it still holds —
//! drops with the last claim.

use std::collections::HashMap;

use crate::fragment_executor::SenderSlot;

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

    /// The parked fragment and the stream a slot names.
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
    pub(crate) fn release(&mut self, slot: &SenderSlot) -> Result<(), String> {
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
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

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
        let fragment = Rc::new(());
        let mut registry = ParkedRegistry::new();
        let (a, b) = (slot(2, 0), slot(2, 1));
        registry.park(&[a, b], Rc::clone(&fragment)).unwrap();
        assert_eq!(registry.claim(&b, "relay").unwrap().1, 1);
        registry.release(&a).unwrap();
        assert_eq!(Rc::strong_count(&fragment), 2, "b still holds the fragment");
        assert!(registry.claim(&a, "relay").is_err());
        registry.release(&b).unwrap();
        assert_eq!(Rc::strong_count(&fragment), 1);
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
