use std::collections::HashMap;

use starrocks_thrift::descriptors::TDescriptorTable;
use substrait::proto::r#type;
use substrait::proto::{NamedStruct, Type};

use crate::error::{Result, TranslateError};
use crate::type_mapper;

/// Normalized StarRocks slot metadata used while translating row references.
#[derive(Clone, Debug)]
pub struct SlotInfo {
    /// StarRocks slot identifier from `TSlotDescriptor.id`.
    pub slot_id: i32,
    /// Stable output name for the slot, falling back to `col_<slot_id>`.
    pub col_name: String,
    /// Substrait type mapped from the StarRocks slot type and nullability.
    ///
    /// This is only populated for materialized slots, because unused
    /// descriptor slots should not make otherwise supported fragments fail.
    pub substrait_type: Option<Type>,
    /// Whether StarRocks marks the slot as materialized for this fragment.
    pub is_materialized: bool,
}

impl SlotInfo {
    /// Returns the slot's stable output name.
    pub fn output_name(&self) -> String {
        if self.col_name.is_empty() {
            format!("col_{}", self.slot_id)
        } else {
            self.col_name.clone()
        }
    }
}

/// Normalized StarRocks tuple metadata and its ordered materialized slots.
#[derive(Clone, Debug)]
pub struct TupleInfo {
    /// Optional table id referenced by this tuple.
    pub table_id: Option<i64>,
    /// Slot ids sorted in descriptor/output order.
    pub slot_ids: Vec<i32>,
}

/// Minimal table metadata needed to build Substrait named-table reads.
#[derive(Clone, Debug)]
struct TableInfo {
    /// StarRocks database name, empty for unqualified tuple fallbacks.
    db_name: String,
    /// StarRocks table name.
    table_name: String,
}

/// Key into the slot map: a slot is identified by its owning tuple plus its slot id.
///
/// StarRocks slot ids are unique only within a tuple — the same id can appear in several tuples
/// (e.g. a FILES scan's src and dest tuples), so the tuple id is part of the key; keying by slot
/// id alone would let one tuple's slots clobber another's.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SlotKey {
    /// Owning tuple id (`TSlotDescriptor.parent`).
    tuple_id: i32,
    /// StarRocks slot id, unique only within `tuple_id`.
    slot_id: i32,
}

impl SlotKey {
    /// Builds a slot key from its owning tuple and slot id.
    fn new(tuple_id: i32, slot_id: i32) -> Self {
        Self { tuple_id, slot_id }
    }
}

/// Lookup structure for StarRocks descriptor-table ids.
#[derive(Clone, Debug)]
pub struct DescriptorTable {
    /// Slots keyed by [`SlotKey`] (owning tuple + slot id).
    slots: HashMap<SlotKey, SlotInfo>,
    /// Tuples keyed by StarRocks tuple id.
    tuples: HashMap<i32, TupleInfo>,
    /// Tables keyed by StarRocks table id.
    tables: HashMap<i64, TableInfo>,
}

impl TryFrom<&TDescriptorTable> for DescriptorTable {
    type Error = TranslateError;

    /// Builds descriptor lookups from the StarRocks Thrift descriptor table.
    fn try_from(desc_tbl: &TDescriptorTable) -> Result<Self> {
        let tables = desc_tbl
            .table_descriptors
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|table| {
                (
                    table.id,
                    TableInfo {
                        db_name: table.db_name.clone(),
                        table_name: table.table_name.clone(),
                    },
                )
            })
            .collect();

        let mut tuples = desc_tbl
            .tuple_descriptors
            .iter()
            .map(|tuple| {
                let tuple_id = tuple.id.ok_or(TranslateError::MissingField {
                    context: "TTupleDescriptor",
                    field: "id",
                })?;
                Ok((
                    tuple_id,
                    TupleInfo {
                        table_id: tuple.table_id,
                        slot_ids: Vec::new(),
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let slot_descriptors = desc_tbl.slot_descriptors.as_deref().unwrap_or_default();
        let mut slots = HashMap::with_capacity(slot_descriptors.len());

        for slot in slot_descriptors {
            let slot_id = slot.id.ok_or(TranslateError::MissingField {
                context: "TSlotDescriptor",
                field: "id",
            })?;
            let parent_tuple_id = slot.parent.ok_or(TranslateError::MissingField {
                context: "TSlotDescriptor",
                field: "parent",
            })?;
            let is_materialized = slot.is_materialized.unwrap_or(true);
            let substrait_type = if is_materialized {
                let slot_type = slot
                    .slot_type
                    .as_ref()
                    .ok_or(TranslateError::MissingField {
                        context: "TSlotDescriptor",
                        field: "slotType",
                    })?;
                Some(type_mapper::map_type_desc(
                    slot_type,
                    slot.is_nullable.unwrap_or(true),
                )?)
            } else {
                None
            };

            slots.insert(
                SlotKey::new(parent_tuple_id, slot_id),
                SlotInfo {
                    slot_id,
                    col_name: slot
                        .col_name
                        .clone()
                        .unwrap_or_else(|| format!("col_{slot_id}")),
                    substrait_type,
                    is_materialized,
                },
            );

            // Wire order is the column order: the FE appends materialized slots in the same
            // order as the positional expression lists that reference them (grouping keys in
            // GROUP BY order, sort tuples ordering-slots-first). It is not sorted by slot id,
            // and `TSlotDescriptor.column_pos` cannot be used instead -- the FE always sends
            // -1 and the field is deprecated in the IDL.
            tuples
                .get_mut(&parent_tuple_id)
                .ok_or_else(|| {
                    TranslateError::descriptor(format!(
                        "slot {slot_id} references missing tuple {parent_tuple_id}"
                    ))
                })?
                .slot_ids
                .push(slot_id);
        }

        Ok(Self {
            slots,
            tuples,
            tables,
        })
    }
}

impl DescriptorTable {
    /// Looks up a StarRocks slot by its owning tuple and slot id; ids are unique only per tuple.
    pub fn slot(&self, tuple_id: i32, slot_id: i32) -> Result<&SlotInfo> {
        self.slots
            .get(&SlotKey::new(tuple_id, slot_id))
            .ok_or_else(|| {
                TranslateError::descriptor(format!("slot {slot_id} not found in tuple {tuple_id}"))
            })
    }

    /// Looks up a StarRocks tuple by id.
    pub fn tuple(&self, tuple_id: i32) -> Result<&TupleInfo> {
        self.tuples
            .get(&tuple_id)
            .ok_or_else(|| TranslateError::descriptor(format!("tuple {tuple_id} not found")))
    }

    /// Returns materialized slot ids for a tuple in StarRocks descriptor order.
    pub fn materialized_slot_ids(&self, tuple_id: i32) -> Result<Vec<i32>> {
        let tuple = self.tuple(tuple_id)?;
        Ok(tuple
            .slot_ids
            .iter()
            .copied()
            .filter(|slot_id| {
                self.slots
                    .get(&SlotKey::new(tuple_id, *slot_id))
                    .map(|slot| slot.is_materialized)
                    .unwrap_or(false)
            })
            .collect())
    }

    /// Builds output names for a row layout described by tuple ids.
    pub fn output_names_for_tuples(&self, row_tuples: &[i32]) -> Result<Vec<String>> {
        row_tuples
            .iter()
            .copied()
            .map(|tuple_id| {
                self.materialized_slot_ids(tuple_id)?
                    .into_iter()
                    .map(|slot_id| self.slot(tuple_id, slot_id).map(SlotInfo::output_name))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
            .map(|names_by_tuple| names_by_tuple.into_iter().flatten().collect())
    }

    /// Builds the Substrait schema for a StarRocks tuple.
    pub fn named_struct(&self, tuple_id: i32) -> Result<NamedStruct> {
        let (names, types): (Vec<_>, Vec<_>) = self
            .materialized_slot_ids(tuple_id)?
            .into_iter()
            .map(|slot_id| {
                let slot = self.slot(tuple_id, slot_id)?;
                let substrait_type =
                    slot.substrait_type
                        .clone()
                        .ok_or(TranslateError::MissingField {
                            context: "materialized TSlotDescriptor",
                            field: "slotType",
                        })?;
                Ok((slot.output_name(), substrait_type))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
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

    /// Resolves a StarRocks `(tuple id, slot id)` to its zero-based field index in `row_tuples`.
    ///
    /// The tuple id comes from the `TSlotRef` and disambiguates slots: ids are unique only within
    /// a tuple, so the same slot id can name different columns in different tuples.
    pub fn slot_global_index(
        &self,
        tuple_id: i32,
        slot_id: i32,
        row_tuples: &[i32],
    ) -> Result<usize> {
        let mut offset = 0;
        for &candidate_tuple in row_tuples {
            let materialized = self.materialized_slot_ids(candidate_tuple)?;
            if candidate_tuple == tuple_id
                && let Some(index) = materialized
                    .iter()
                    .position(|candidate| *candidate == slot_id)
            {
                return Ok(offset + index);
            }
            offset += materialized.len();
        }

        Err(TranslateError::descriptor(format!(
            "slot {slot_id} (tuple {tuple_id}) is not part of row_tuples {row_tuples:?}"
        )))
    }

    /// Returns the Substrait named-table path for a tuple's backing table.
    pub fn table_names_for_tuple(&self, tuple_id: i32) -> Result<Vec<String>> {
        let tuple = self.tuple(tuple_id)?;
        let Some(table_id) = tuple.table_id else {
            return Ok(vec![format!("tuple_{tuple_id}")]);
        };
        let table = self.tables.get(&table_id).ok_or_else(|| {
            TranslateError::descriptor(format!(
                "tuple {tuple_id} references missing table {table_id}"
            ))
        })?;
        if table.db_name.is_empty() {
            Ok(vec![table.table_name.clone()])
        } else {
            Ok(vec![table.db_name.clone(), table.table_name.clone()])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starrocks_thrift::descriptors::{TSlotDescriptor, TTupleDescriptor};
    use starrocks_thrift::types::{
        TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType,
    };

    /// Builds a scalar BIGINT type descriptor for slot fixtures.
    fn bigint() -> TTypeDesc {
        TTypeDesc::new(Some(vec![TTypeNode::new(
            TTypeNodeType::SCALAR,
            Some(TScalarType::new(TPrimitiveType::BIGINT, None, None, None)),
            None,
            None,
        )]))
    }

    /// Builds a materialized BIGINT slot owned by `tuple_id` at `column_pos`.
    fn slot(id: i32, tuple_id: i32, column_pos: i32, name: &str) -> TSlotDescriptor {
        TSlotDescriptor::new(
            Some(id),
            Some(tuple_id),
            Some(bigint()),
            Some(column_pos),
            None,
            None,
            None,
            Some(name.to_string()),
            None,
            Some(true),
            Some(true),
            Some(true),
            None,
            None,
        )
    }

    /// Builds a two-tuple table: tuple 0 = {1:a, 2:b}, tuple 1 = {3:c, 4:d}.
    fn two_tuple_desc() -> DescriptorTable {
        let desc_tbl = TDescriptorTable::new(
            Some(vec![
                slot(1, 0, 0, "a"),
                slot(2, 0, 1, "b"),
                slot(3, 1, 0, "c"),
                slot(4, 1, 1, "d"),
            ]),
            vec![
                TTupleDescriptor::new(Some(0), None, None, None, None),
                TTupleDescriptor::new(Some(1), None, None, None, None),
            ],
            None,
            None,
        );
        DescriptorTable::try_from(&desc_tbl).unwrap()
    }

    /// Verifies field indices accumulate each preceding tuple's materialized width.
    #[test]
    fn slot_global_index_accumulates_offset_across_tuples() {
        let desc = two_tuple_desc();
        // Tuple 0 contributes two columns, so tuple 1's slots start at index 2.
        assert_eq!(desc.slot_global_index(0, 1, &[0, 1]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(0, 2, &[0, 1]).unwrap(), 1);
        assert_eq!(desc.slot_global_index(1, 3, &[0, 1]).unwrap(), 2);
        assert_eq!(desc.slot_global_index(1, 4, &[0, 1]).unwrap(), 3);
    }

    /// Verifies field indices follow the order tuples appear in `row_tuples`.
    #[test]
    fn slot_global_index_follows_row_tuple_order() {
        let desc = two_tuple_desc();
        // Reversed layout: tuple 1's slots now come before tuple 0's.
        assert_eq!(desc.slot_global_index(1, 3, &[1, 0]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(1, 4, &[1, 0]).unwrap(), 1);
        assert_eq!(desc.slot_global_index(0, 1, &[1, 0]).unwrap(), 2);
        assert_eq!(desc.slot_global_index(0, 2, &[1, 0]).unwrap(), 3);
    }

    /// Verifies a slot whose tuple is absent from the row layout is rejected.
    #[test]
    fn slot_global_index_rejects_slot_outside_row_tuples() {
        let desc = two_tuple_desc();
        // Slot 3 lives in tuple 1, which is absent from this row layout.
        let err = desc.slot_global_index(1, 3, &[0]).unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)));
    }

    /// Verifies an unknown slot id surfaces a descriptor error.
    #[test]
    fn slot_global_index_reports_unknown_slot() {
        let desc = two_tuple_desc();
        let err = desc.slot_global_index(0, 99, &[0, 1]).unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)));
    }

    /// Slot ids are unique only within a tuple; overlapping ids across tuples (as a FILES scan's
    /// src/dest tuples produce) must resolve per tuple rather than collide in the slot map.
    #[test]
    fn overlapping_slot_ids_across_tuples_resolve_per_tuple() {
        // Dest/output tuple 0 = {1:a, 2:b}; src tuple 1 reuses ids {1:x, 2:y}.
        let desc_tbl = TDescriptorTable::new(
            Some(vec![
                slot(1, 0, 0, "a"),
                slot(2, 0, 1, "b"),
                slot(1, 1, 0, "x"),
                slot(2, 1, 1, "y"),
            ]),
            vec![
                TTupleDescriptor::new(Some(0), None, None, None, None),
                TTupleDescriptor::new(Some(1), None, None, None, None),
            ],
            None,
            None,
        );
        let desc = DescriptorTable::try_from(&desc_tbl).unwrap();

        // Resolving slot 1 in tuple 0 must not pick up tuple 1's slot 1.
        assert_eq!(desc.slot_global_index(0, 1, &[0]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(0, 2, &[0]).unwrap(), 1);
        assert_eq!(desc.slot(0, 1).unwrap().output_name(), "a");
        assert_eq!(desc.slot(1, 1).unwrap().output_name(), "x");
        assert_eq!(
            desc.output_names_for_tuples(&[0]).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
    }
}
