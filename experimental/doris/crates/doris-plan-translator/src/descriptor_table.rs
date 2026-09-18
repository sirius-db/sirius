//! The Doris descriptor table (`TDescriptorTable`) as the translator sees it: tuples, their
//! slots in wire order, and the resolution of a `(tuple_id, slot_id)` pair to a column index
//! in a plan node's row layout.
//!
//! # What the FE sends
//!
//! `DescriptorTable.toThrift()` walks every tuple and appends its slots in
//! `TupleDescriptor.getSlots()` order, and that wire order **is** the column order the BE
//! lays a tuple out in (`TupleDescriptor::slots()` on the BE side is filled in the same
//! order; `RowDescriptor::get_column_id` counts through it). `TSlotDescriptor.columnPos`
//! and `slotIdx` are always `-1` and `isMaterialized` is always `true` (all three are
//! deprecated), so the order of `slotDescriptors` is the only layout information there is.
//!
//! Nullability travels in `nullIndicatorBit`: the FE writes `0` for a nullable slot and
//! `-1` for a non-nullable one (`SlotDescriptor.toThrift`); `TTypeDesc.is_nullable` is left
//! unset on slot types.
//!
//! # Row layouts
//!
//! A plan node's output rows are described by `TPlanNode.row_tuples`: the columns are the
//! slots of each listed tuple, tuple by tuple, slot by slot in wire order. A `SLOT_REF`
//! names `(tuple_id, slot_id)` and resolves to its zero-based index in that concatenation
//! ([`DescriptorTable::slot_global_index`]). `TPlanNode.nullable_tuples` is *not* part of
//! this: Nereids computes nullability per slot (it marks the exchange/join tuples of even
//! inner joins as nullable tuples), and the BE reads nullability from the slot as well.
//!
//! # Keys
//!
//! Slots are keyed by `(tuple_id, slot_id)`. Doris' FE draws slot ids from one generator per
//! query, so today an id is unique across tuples, but the `TSlotRef` carries both ids and
//! nothing in the wire format promises uniqueness; the StarRocks translator was bitten twice
//! by keying on the slot id alone.

use std::collections::HashMap;

use doris_thrift::descriptors::TDescriptorTable;
use doris_thrift::types::TPrimitiveType;
use substrait::proto::r#type;
use substrait::proto::{NamedStruct, Type};

use crate::error::{Result, TranslateError};
use crate::type_mapper;

/// One Doris slot: a column of a tuple, with its mapped Substrait type.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotInfo {
    /// `TSlotDescriptor.id`.
    pub slot_id: i32,
    /// `TSlotDescriptor.parent`, the owning tuple.
    pub tuple_id: i32,
    /// `TSlotDescriptor.colName`; empty for slots the planner derived from expressions.
    pub col_name: String,
    /// Slot nullability as the FE encodes it (`nullIndicatorBit != -1`).
    pub nullable: bool,
    /// Doris primitive type of the slot.
    pub primitive: TPrimitiveType,
    /// Substrait type mapped from the slot type and nullability.
    pub substrait_type: Type,
}

impl SlotInfo {
    /// Stable output name for the slot: its column name, or `col_<slot_id>` for a derived slot.
    pub fn output_name(&self) -> String {
        if self.col_name.is_empty() {
            format!("col_{}", self.slot_id)
        } else {
            self.col_name.clone()
        }
    }
}

/// One Doris tuple and its slots in wire (= column) order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TupleInfo {
    /// `TTupleDescriptor.tableId` when the tuple reads a catalog table (never for a TVF).
    pub table_id: Option<i64>,
    /// Slot ids in wire order.
    pub slot_ids: Vec<i32>,
}

/// Key into the slot map: a slot is identified by its owning tuple plus its slot id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SlotKey {
    /// Owning tuple id (`TSlotDescriptor.parent`).
    tuple_id: i32,
    /// `TSlotDescriptor.id`.
    slot_id: i32,
}

/// Lookup structure over a Doris descriptor table.
#[derive(Clone, Debug)]
pub struct DescriptorTable {
    /// Slots keyed by owning tuple + slot id.
    slots: HashMap<SlotKey, SlotInfo>,
    /// Tuples keyed by tuple id.
    tuples: HashMap<i32, TupleInfo>,
}

impl TryFrom<&TDescriptorTable> for DescriptorTable {
    type Error = TranslateError;

    /// Builds the lookups from the FE's descriptor table, mapping every slot type.
    ///
    /// Fails on the first slot whose type is outside the type gate, naming the slot: the
    /// descriptor table is shared by every fragment of the query, so an unsupported column
    /// anywhere in the query fails the query as a whole rather than one fragment at a time.
    fn try_from(desc_tbl: &TDescriptorTable) -> Result<Self> {
        let mut tuples: HashMap<i32, TupleInfo> = HashMap::new();
        for tuple in &desc_tbl.tuple_descriptors {
            let previous = tuples.insert(
                tuple.id,
                TupleInfo {
                    table_id: tuple.table_id,
                    slot_ids: Vec::new(),
                },
            );
            if previous.is_some() {
                return Err(TranslateError::descriptor(format!(
                    "tuple {} appears twice in the descriptor table",
                    tuple.id
                )));
            }
        }

        let slot_descriptors = desc_tbl.slot_descriptors.as_deref().unwrap_or_default();
        let mut slots = HashMap::with_capacity(slot_descriptors.len());
        for slot in slot_descriptors {
            let key = SlotKey {
                tuple_id: slot.parent,
                slot_id: slot.id,
            };
            // Non-nullable slots get -1 for the bit (SlotDescriptor.toThrift); anything else is
            // a nullable slot with a real (always 0) bit index.
            let nullable = slot.null_indicator_bit != -1;
            let map_type = || -> Result<(TPrimitiveType, Type)> {
                Ok((
                    type_mapper::scalar_primitive(&slot.slot_type)?,
                    type_mapper::map_type_desc(&slot.slot_type, nullable)?,
                ))
            };
            let (primitive, substrait_type) =
                map_type().map_err(|source| TranslateError::UnsupportedSlot {
                    tuple_id: slot.parent,
                    slot_id: slot.id,
                    col_name: slot.col_name.clone(),
                    source: Box::new(source),
                })?;
            let tuple = tuples.get_mut(&slot.parent).ok_or_else(|| {
                TranslateError::descriptor(format!(
                    "slot {} references missing tuple {}",
                    slot.id, slot.parent
                ))
            })?;
            if slots
                .insert(
                    key,
                    SlotInfo {
                        slot_id: slot.id,
                        tuple_id: slot.parent,
                        col_name: slot.col_name.clone(),
                        nullable,
                        primitive,
                        substrait_type,
                    },
                )
                .is_some()
            {
                return Err(TranslateError::descriptor(format!(
                    "slot {} appears twice in tuple {}",
                    slot.id, slot.parent
                )));
            }
            tuple.slot_ids.push(slot.id);
        }

        Ok(Self { slots, tuples })
    }
}

impl DescriptorTable {
    /// Looks up a slot by its owning tuple and slot id.
    pub fn slot(&self, tuple_id: i32, slot_id: i32) -> Result<&SlotInfo> {
        self.slots
            .get(&SlotKey { tuple_id, slot_id })
            .ok_or_else(|| {
                TranslateError::descriptor(format!("slot {slot_id} not found in tuple {tuple_id}"))
            })
    }

    /// Looks up a tuple by id.
    pub fn tuple(&self, tuple_id: i32) -> Result<&TupleInfo> {
        self.tuples
            .get(&tuple_id)
            .ok_or_else(|| TranslateError::descriptor(format!("tuple {tuple_id} not found")))
    }

    /// Number of tuples in the table.
    pub fn tuple_count(&self) -> usize {
        self.tuples.len()
    }

    /// Number of slots in the table.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The slots of a tuple in wire (= column) order.
    pub fn tuple_slots(&self, tuple_id: i32) -> Result<Vec<&SlotInfo>> {
        self.tuple(tuple_id)?
            .slot_ids
            .iter()
            .map(|slot_id| self.slot(tuple_id, *slot_id))
            .collect()
    }

    /// Number of columns a tuple contributes to a row layout.
    pub fn tuple_width(&self, tuple_id: i32) -> Result<usize> {
        Ok(self.tuple(tuple_id)?.slot_ids.len())
    }

    /// Number of columns in a row layout made of `row_tuples`.
    pub fn row_width(&self, row_tuples: &[i32]) -> Result<usize> {
        row_tuples
            .iter()
            .map(|tuple_id| self.tuple_width(*tuple_id))
            .sum()
    }

    /// The slots of a row layout, tuple by tuple in wire order.
    pub fn row_slots(&self, row_tuples: &[i32]) -> Result<Vec<&SlotInfo>> {
        let mut slots = Vec::with_capacity(self.row_width(row_tuples)?);
        for tuple_id in row_tuples {
            slots.extend(self.tuple_slots(*tuple_id)?);
        }
        Ok(slots)
    }

    /// Output names for a row layout (see [`SlotInfo::output_name`]).
    pub fn output_names_for_tuples(&self, row_tuples: &[i32]) -> Result<Vec<String>> {
        Ok(self
            .row_slots(row_tuples)?
            .into_iter()
            .map(SlotInfo::output_name)
            .collect())
    }

    /// The Substrait schema of a row layout: names and types of its slots in order.
    pub fn named_struct_for_tuples(&self, row_tuples: &[i32]) -> Result<NamedStruct> {
        let (names, types): (Vec<_>, Vec<_>) = self
            .row_slots(row_tuples)?
            .into_iter()
            .map(|slot| (slot.output_name(), slot.substrait_type.clone()))
            .unzip();
        Ok(NamedStruct {
            names,
            r#struct: Some(r#type::Struct {
                types,
                type_variation_reference: 0,
                nullability: r#type::Nullability::Required as i32,
            }),
        })
    }

    /// The Substrait schema of one tuple.
    pub fn named_struct(&self, tuple_id: i32) -> Result<NamedStruct> {
        self.named_struct_for_tuples(&[tuple_id])
    }

    /// Resolves a `(tuple_id, slot_id)` reference to its zero-based column index in the row
    /// layout `row_tuples` (tuples concatenated in order, each in wire slot order).
    pub fn slot_global_index(
        &self,
        tuple_id: i32,
        slot_id: i32,
        row_tuples: &[i32],
    ) -> Result<usize> {
        let mut offset = 0;
        for &candidate in row_tuples {
            let tuple = self.tuple(candidate)?;
            if candidate == tuple_id
                && let Some(index) = tuple.slot_ids.iter().position(|id| *id == slot_id)
            {
                return Ok(offset + index);
            }
            offset += tuple.slot_ids.len();
        }
        // Distinguish "unknown slot" from "known slot, wrong layout" in the message.
        self.slot(tuple_id, slot_id)?;
        Err(TranslateError::descriptor(format!(
            "slot {slot_id} (tuple {tuple_id}) is not part of row_tuples {row_tuples:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use doris_thrift::descriptors::{TSlotDescriptor, TTupleDescriptor};
    use doris_thrift::types::{TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType};

    use super::*;

    fn scalar_desc(scalar: TScalarType) -> TTypeDesc {
        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(scalar),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn bigint() -> TTypeDesc {
        scalar_desc(TScalarType {
            type_: TPrimitiveType::BIGINT,
            ..Default::default()
        })
    }

    /// A slot the way `SlotDescriptor.toThrift` writes it: `columnPos`/`slotIdx` = -1,
    /// `isMaterialized` = true, nullability in the bit.
    fn slot(
        id: i32,
        tuple_id: i32,
        name: &str,
        nullable: bool,
        slot_type: TTypeDesc,
    ) -> TSlotDescriptor {
        TSlotDescriptor {
            id,
            parent: tuple_id,
            slot_type,
            column_pos: -1,
            byte_offset: 0,
            null_indicator_byte: 0,
            null_indicator_bit: if nullable { 0 } else { -1 },
            col_name: name.to_string(),
            slot_idx: -1,
            is_materialized: true,
            ..Default::default()
        }
    }

    fn tuple(id: i32) -> TTupleDescriptor {
        TTupleDescriptor {
            id,
            ..Default::default()
        }
    }

    /// Two tuples: tuple 0 = {1:a, 2:b}, tuple 1 = {3:c, 4:d}.
    fn two_tuple_table() -> DescriptorTable {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(1, 0, "a", true, bigint()),
                slot(2, 0, "b", false, bigint()),
                slot(3, 1, "c", true, bigint()),
                slot(4, 1, "d", true, bigint()),
            ]),
            tuple_descriptors: vec![tuple(0), tuple(1)],
            table_descriptors: None,
        };
        DescriptorTable::try_from(&desc_tbl).unwrap()
    }

    #[test]
    fn slot_global_index_accumulates_offset_across_tuples() {
        let desc = two_tuple_table();
        assert_eq!(desc.slot_global_index(0, 1, &[0, 1]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(0, 2, &[0, 1]).unwrap(), 1);
        assert_eq!(desc.slot_global_index(1, 3, &[0, 1]).unwrap(), 2);
        assert_eq!(desc.slot_global_index(1, 4, &[0, 1]).unwrap(), 3);
        assert_eq!(desc.row_width(&[0, 1]).unwrap(), 4);
    }

    #[test]
    fn slot_global_index_follows_row_tuple_order() {
        let desc = two_tuple_table();
        assert_eq!(desc.slot_global_index(1, 3, &[1, 0]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(1, 4, &[1, 0]).unwrap(), 1);
        assert_eq!(desc.slot_global_index(0, 1, &[1, 0]).unwrap(), 2);
        assert_eq!(desc.slot_global_index(0, 2, &[1, 0]).unwrap(), 3);
        assert_eq!(
            desc.output_names_for_tuples(&[1, 0]).unwrap(),
            vec!["c", "d", "a", "b"]
        );
    }

    #[test]
    fn slot_global_index_rejects_slot_outside_row_tuples_and_unknown_slots() {
        let desc = two_tuple_table();
        let err = desc.slot_global_index(1, 3, &[0]).unwrap_err();
        assert!(
            matches!(&err, TranslateError::Descriptor(msg) if msg.contains("not part of row_tuples")),
            "{err}"
        );
        let err = desc.slot_global_index(0, 99, &[0, 1]).unwrap_err();
        assert!(
            matches!(&err, TranslateError::Descriptor(msg) if msg.contains("not found")),
            "{err}"
        );
        assert!(matches!(
            desc.slot_global_index(0, 1, &[7]).unwrap_err(),
            TranslateError::Descriptor(_)
        ));
    }

    /// Slot ids are keyed per tuple: the same id in two tuples must resolve to two slots.
    #[test]
    fn overlapping_slot_ids_across_tuples_resolve_per_tuple() {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(1, 0, "a", true, bigint()),
                slot(2, 0, "b", true, bigint()),
                slot(1, 1, "x", true, bigint()),
                slot(2, 1, "y", true, bigint()),
            ]),
            tuple_descriptors: vec![tuple(0), tuple(1)],
            table_descriptors: None,
        };
        let desc = DescriptorTable::try_from(&desc_tbl).unwrap();
        assert_eq!(desc.slot_count(), 4);
        assert_eq!(desc.slot_global_index(0, 1, &[0]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(1, 1, &[0, 1]).unwrap(), 2);
        assert_eq!(desc.slot(0, 1).unwrap().output_name(), "a");
        assert_eq!(desc.slot(1, 1).unwrap().output_name(), "x");
    }

    #[test]
    fn nullability_comes_from_the_null_indicator_bit() {
        let desc = two_tuple_table();
        assert!(desc.slot(0, 1).unwrap().nullable);
        assert!(!desc.slot(0, 2).unwrap().nullable);
        let schema = desc.named_struct(0).unwrap();
        assert_eq!(schema.names, vec!["a", "b"]);
        let types = schema.r#struct.unwrap().types;
        assert_eq!(
            types[0].kind,
            Some(r#type::Kind::I64(r#type::I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            }))
        );
        assert_eq!(
            types[1].kind,
            Some(r#type::Kind::I64(r#type::I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Required as i32,
            }))
        );
    }

    #[test]
    fn derived_slots_without_a_column_name_get_a_stable_fallback_name() {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![slot(17, 3, "", true, bigint())]),
            tuple_descriptors: vec![tuple(3)],
            table_descriptors: None,
        };
        let desc = DescriptorTable::try_from(&desc_tbl).unwrap();
        assert_eq!(desc.slot(3, 17).unwrap().output_name(), "col_17");
        assert_eq!(desc.output_names_for_tuples(&[3]).unwrap(), vec!["col_17"]);
    }

    /// The wire order is the column order: it is neither sorted by slot id nor by name.
    #[test]
    fn slots_keep_wire_order_not_id_order() {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(9, 0, "z", true, bigint()),
                slot(4, 0, "m", true, bigint()),
                slot(6, 0, "a", true, bigint()),
            ]),
            tuple_descriptors: vec![tuple(0)],
            table_descriptors: None,
        };
        let desc = DescriptorTable::try_from(&desc_tbl).unwrap();
        assert_eq!(desc.tuple(0).unwrap().slot_ids, vec![9, 4, 6]);
        assert_eq!(
            desc.output_names_for_tuples(&[0]).unwrap(),
            vec!["z", "m", "a"]
        );
        assert_eq!(desc.slot_global_index(0, 6, &[0]).unwrap(), 2);
    }

    #[test]
    fn slot_referencing_missing_tuple_is_rejected() {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![slot(1, 5, "a", true, bigint())]),
            tuple_descriptors: vec![tuple(0)],
            table_descriptors: None,
        };
        let err = DescriptorTable::try_from(&desc_tbl).unwrap_err();
        assert!(
            matches!(&err, TranslateError::Descriptor(msg) if msg.contains("missing tuple 5")),
            "{err}"
        );
    }

    #[test]
    fn duplicate_tuples_and_slots_are_rejected() {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: None,
            tuple_descriptors: vec![tuple(0), tuple(0)],
            table_descriptors: None,
        };
        assert!(matches!(
            DescriptorTable::try_from(&desc_tbl).unwrap_err(),
            TranslateError::Descriptor(_)
        ));
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(1, 0, "a", true, bigint()),
                slot(1, 0, "b", true, bigint()),
            ]),
            tuple_descriptors: vec![tuple(0)],
            table_descriptors: None,
        };
        assert!(matches!(
            DescriptorTable::try_from(&desc_tbl).unwrap_err(),
            TranslateError::Descriptor(_)
        ));
    }

    /// The type gate fires at table-build time and names the slot (G-01 here).
    #[test]
    fn unsupported_slot_type_fails_the_table_and_names_the_slot() {
        let largeint = scalar_desc(TScalarType {
            type_: TPrimitiveType::LARGEINT,
            ..Default::default()
        });
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(1, 0, "ok", true, bigint()),
                slot(2, 0, "big", true, largeint),
            ]),
            tuple_descriptors: vec![tuple(0)],
            table_descriptors: None,
        };
        match DescriptorTable::try_from(&desc_tbl).unwrap_err() {
            TranslateError::UnsupportedSlot {
                tuple_id,
                slot_id,
                col_name,
                source,
            } => {
                assert_eq!((tuple_id, slot_id, col_name.as_str()), (0, 2, "big"));
                assert!(matches!(
                    *source,
                    TranslateError::UnsupportedType {
                        primitive: Some(TPrimitiveType::LARGEINT),
                        ..
                    }
                ));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn table_id_is_carried_for_catalog_tuples() {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: None,
            tuple_descriptors: vec![
                TTupleDescriptor {
                    id: 0,
                    table_id: Some(42),
                    ..Default::default()
                },
                tuple(1),
            ],
            table_descriptors: None,
        };
        let desc = DescriptorTable::try_from(&desc_tbl).unwrap();
        assert_eq!(desc.tuple(0).unwrap().table_id, Some(42));
        assert_eq!(desc.tuple(1).unwrap().table_id, None);
        assert_eq!(desc.tuple_count(), 2);
        assert_eq!(desc.tuple_width(1).unwrap(), 0);
    }
}
