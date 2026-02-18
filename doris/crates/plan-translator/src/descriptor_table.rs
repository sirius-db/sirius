//! Descriptor table resolution.
//!
//! Parses TDescriptorTable to build mappings from slot_id → (tuple_id, column_index, column_type).
//! Used by expression and scan translators to resolve column references.

use std::collections::HashMap;

use anyhow::Result;

use doris_thrift::descriptors::TDescriptorTable;
use substrait::proto::r#type;
use substrait::proto::{NamedStruct, Type};

use crate::type_mapper;

/// Information about a single slot (column) in the descriptor table.
pub struct SlotInfo {
    pub slot_id: i32,
    pub parent_tuple_id: i32,
    pub column_pos: i32,
    pub col_name: String,
    pub substrait_type: Type,
    pub is_nullable: bool,
    pub is_materialized: bool,
}

/// Information about a tuple (row) in the descriptor table.
pub struct TupleInfo {
    pub tuple_id: i32,
    pub table_id: Option<i64>,
    /// Slot IDs ordered by column_pos.
    pub slot_ids: Vec<i32>,
}

/// Parsed descriptor table for efficient lookups.
pub struct DescriptorTable {
    slots: HashMap<i32, SlotInfo>,
    tuples: HashMap<i32, TupleInfo>,
    /// Override column lists for tables (table_name → column names in order).
    /// Set by the gRPC handler from DuckDB's actual table columns.
    table_column_overrides: HashMap<String, Vec<String>>,
}

impl DescriptorTable {
    /// Build from Thrift TDescriptorTable.
    pub fn from_thrift(desc_tbl: &TDescriptorTable) -> Result<Self> {
        let mut slots = HashMap::new();
        let mut tuples = HashMap::new();

        // Parse tuple descriptors.
        for td in &desc_tbl.tuple_descriptors {
            tuples.insert(
                td.id,
                TupleInfo {
                    tuple_id: td.id,
                    table_id: td.table_id,
                    slot_ids: Vec::new(),
                },
            );
        }

        // Parse slot descriptors (skip Doris internal system columns).
        if let Some(slot_descs) = &desc_tbl.slot_descriptors {
            for sd in slot_descs {
                if sd.col_name.starts_with("__DORIS_") {
                    continue;
                }
                let substrait_type = type_mapper::map_type_desc(&sd.slot_type)?;
                let is_nullable = sd.slot_type.is_nullable.unwrap_or(true);

                let info = SlotInfo {
                    slot_id: sd.id,
                    parent_tuple_id: sd.parent,
                    column_pos: sd.column_pos,
                    col_name: sd.col_name.clone(),
                    substrait_type,
                    is_nullable,
                    is_materialized: sd.is_materialized,
                };

                if let Some(tuple) = tuples.get_mut(&sd.parent) {
                    tuple.slot_ids.push(sd.id);
                }

                slots.insert(sd.id, info);
            }
        }

        // Sort slot_ids by column_pos within each tuple.
        for tuple in tuples.values_mut() {
            tuple
                .slot_ids
                .sort_by_key(|&sid| slots.get(&sid).map(|s| s.column_pos).unwrap_or(0));
        }

        Ok(Self { slots, tuples, table_column_overrides: HashMap::new() })
    }

    pub fn get_slot(&self, slot_id: i32) -> Result<&SlotInfo> {
        self.slots
            .get(&slot_id)
            .ok_or_else(|| anyhow::anyhow!("slot_id {} not found in descriptor table", slot_id))
    }

    pub fn get_tuple(&self, tuple_id: i32) -> Result<&TupleInfo> {
        self.tuples
            .get(&tuple_id)
            .ok_or_else(|| anyhow::anyhow!("tuple_id {} not found in descriptor table", tuple_id))
    }

    /// Get the column index of a slot within its parent tuple (materialized slots only).
    pub fn slot_column_index(&self, slot_id: i32) -> Result<usize> {
        let slot = self.get_slot(slot_id)?;
        let tuple = self.get_tuple(slot.parent_tuple_id)?;
        let materialized: Vec<i32> = tuple
            .slot_ids
            .iter()
            .copied()
            .filter(|&sid| self.slots.get(&sid).map(|s| s.is_materialized).unwrap_or(false))
            .collect();
        materialized
            .iter()
            .position(|&sid| sid == slot_id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "slot {} not found in materialized slots of tuple {}",
                    slot_id,
                    slot.parent_tuple_id
                )
            })
    }

    /// Get the global column index of a slot within a combined output defined by `row_tuples`.
    ///
    /// Iterates over all tuples in order, counting materialized slots, and returns the
    /// global index of the given slot across all tuples.
    ///
    /// Falls back to name-based matching when the slot's parent tuple isn't in
    /// `row_tuples` (common in aggregation/sort where expressions reference an
    /// intermediate tuple but data comes from a child's scan tuple).
    pub fn slot_global_index(&self, slot_id: i32, row_tuples: &[i32]) -> Result<usize> {
        let slot = self.get_slot(slot_id)?;
        let mut global_idx = 0;
        for &tuple_id in row_tuples {
            let tuple = self.get_tuple(tuple_id)?;
            let materialized: Vec<i32> = tuple
                .slot_ids
                .iter()
                .copied()
                .filter(|&sid| {
                    self.slots
                        .get(&sid)
                        .map(|s| s.is_materialized)
                        .unwrap_or(false)
                })
                .collect();
            if tuple_id == slot.parent_tuple_id {
                if let Some(pos) = materialized.iter().position(|&sid| sid == slot_id) {
                    return Ok(global_idx + pos);
                }
            }
            global_idx += materialized.len();
        }
        // Slot's parent tuple not in row_tuples — fall back.
        // First try resolving against the full table schema (handles late materialization
        // where the scan tuple has a subset of columns but the ReadRel has all columns).
        for &tuple_id in row_tuples {
            if let Ok(tuple) = self.get_tuple(tuple_id) {
                if tuple.table_id.is_some() {
                    // Target tuple references a file table — resolve against full table schema.
                    return self.slot_table_index_by_name(slot_id, tuple_id);
                }
            }
        }
        // Try override columns (handles TVF scans without table_id).
        let slot = self.get_slot(slot_id)?;
        for (_, columns) in &self.table_column_overrides {
            if let Some(pos) = columns.iter().position(|c| c == &slot.col_name) {
                return Ok(pos);
            }
        }
        // No file table or override found — fall back to name-based matching in the given tuples.
        self.slot_index_by_name_in_tuples(slot_id, row_tuples)
    }

    /// Count total materialized columns across the given tuple IDs.
    pub fn count_materialized_columns(&self, row_tuples: &[i32]) -> usize {
        let mut count = 0;
        for &tuple_id in row_tuples {
            if let Ok(tuple) = self.get_tuple(tuple_id) {
                for &slot_id in &tuple.slot_ids {
                    if let Ok(slot) = self.get_slot(slot_id) {
                        if slot.is_materialized {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Resolve a slot's column index by matching its name against a set of target tuples.
    ///
    /// Used when expressions reference slots from a different tuple than the actual data
    /// source (e.g., aggregation expressions reference an output tuple, but the data comes
    /// from a scan tuple with different column ordering).
    pub fn slot_index_by_name_in_tuples(
        &self,
        slot_id: i32,
        target_tuples: &[i32],
    ) -> Result<usize> {
        let slot = self.get_slot(slot_id)?;
        let target_name = &slot.col_name;

        let mut global_idx = 0;
        for &tuple_id in target_tuples {
            let tuple = self.get_tuple(tuple_id)?;
            let materialized: Vec<i32> = tuple
                .slot_ids
                .iter()
                .copied()
                .filter(|&sid| {
                    self.slots
                        .get(&sid)
                        .map(|s| s.is_materialized)
                        .unwrap_or(false)
                })
                .collect();
            for &sid in &materialized {
                if let Some(s) = self.slots.get(&sid) {
                    if s.col_name == *target_name {
                        return Ok(global_idx);
                    }
                }
                global_idx += 1;
            }
        }
        anyhow::bail!(
            "slot {} (name={}) not found in target tuples {:?}",
            slot_id,
            target_name,
            target_tuples
        )
    }

    /// Find a slot by column name across all tuples (first match wins).
    pub fn find_slot_by_name(&self, name: &str) -> Option<&SlotInfo> {
        self.slots.values().find(|s| s.col_name == name && s.is_materialized)
    }

    /// Set an override column list for a specific table name.
    ///
    /// When set, `slot_table_index` and related methods will resolve SLOT_REFs
    /// by matching column names against this list (in order), instead of using
    /// the descriptor table's tuple structure. This handles TVF scans where
    /// the descriptor table doesn't have table_id linkage.
    pub fn set_table_column_override(&mut self, table_name: String, columns: Vec<String>) {
        self.table_column_overrides.insert(table_name, columns);
    }

    /// Resolve a slot's index by name against the override column list for a table.
    pub fn slot_index_in_override(&self, slot_id: i32, table_name: &str) -> Result<usize> {
        let slot = self.get_slot(slot_id)?;
        let columns = self
            .table_column_overrides
            .get(table_name)
            .ok_or_else(|| anyhow::anyhow!("no column override for table {}", table_name))?;
        columns
            .iter()
            .position(|c| c == &slot.col_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "slot {} (name={}) not found in table {} override columns {:?}",
                    slot_id,
                    slot.col_name,
                    table_name,
                    columns
                )
            })
    }

    /// Build a Substrait NamedStruct from a tuple's materialized columns.
    pub fn tuple_to_named_struct(&self, tuple_id: i32) -> Result<NamedStruct> {
        let tuple = self.get_tuple(tuple_id)?;
        let mut names = Vec::new();
        let mut types = Vec::new();

        for &slot_id in &tuple.slot_ids {
            let slot = self.get_slot(slot_id)?;
            if slot.is_materialized {
                names.push(slot.col_name.clone());
                types.push(slot.substrait_type.clone());
            }
        }

        Ok(NamedStruct {
            names,
            r#struct: Some(r#type::Struct {
                types,
                type_variation_reference: 0,
                nullability: r#type::Nullability::Unspecified as i32,
            }),
        })
    }

    /// Build a Substrait NamedStruct from ALL tuples referencing the same table.
    ///
    /// This handles late materialization: the scan tuple may only have sort/filter keys,
    /// but other tuples referencing the same table have the remaining output columns.
    /// The returned schema is the union of all such columns, sorted by column_pos.
    pub fn table_named_struct(&self, tuple_id: i32) -> Result<NamedStruct> {
        let tuple = self.get_tuple(tuple_id)?;
        let table_id = match tuple.table_id {
            Some(id) => id,
            None => return self.tuple_to_named_struct(tuple_id),
        };

        // Find all tuples referencing the same table.
        let related_tuples: Vec<i32> = self
            .tuples
            .values()
            .filter(|t| t.table_id == Some(table_id))
            .map(|t| t.tuple_id)
            .collect();

        // Collect all unique materialized columns across related tuples.
        let mut seen = std::collections::HashSet::new();
        let mut columns: Vec<(i32, String, Type)> = Vec::new();
        for &tid in &related_tuples {
            if let Ok(t) = self.get_tuple(tid) {
                for &slot_id in &t.slot_ids {
                    if let Ok(slot) = self.get_slot(slot_id) {
                        if slot.is_materialized && seen.insert(slot.col_name.clone()) {
                            columns.push((
                                slot.column_pos,
                                slot.col_name.clone(),
                                slot.substrait_type.clone(),
                            ));
                        }
                    }
                }
            }
        }
        columns.sort_by_key(|(pos, _, _)| *pos);

        let names: Vec<String> = columns.iter().map(|(_, n, _)| n.clone()).collect();
        let types: Vec<Type> = columns.iter().map(|(_, _, t)| t.clone()).collect();

        Ok(NamedStruct {
            names,
            r#struct: Some(r#type::Struct {
                types,
                type_variation_reference: 0,
                nullability: r#type::Nullability::Unspecified as i32,
            }),
        })
    }

    /// Resolve a slot's column index within the full table schema (across all tuples with same table_id).
    ///
    /// Used for FILE_SCAN_NODE expressions where the scan tuple may only have a subset
    /// of columns, but field references need to be relative to the full table schema.
    pub fn slot_table_index(&self, slot_id: i32) -> Result<usize> {
        let slot = self.get_slot(slot_id)?;
        let parent = self.get_tuple(slot.parent_tuple_id)?;
        let table_id = match parent.table_id {
            Some(id) => id,
            None => {
                // Try override columns first (handles TVF scans).
                for (_, columns) in &self.table_column_overrides {
                    if let Some(pos) = columns.iter().position(|c| c == &slot.col_name) {
                        return Ok(pos);
                    }
                }
                return self.slot_column_index(slot_id);
            }
        };

        // Build full table column list.
        let mut seen = std::collections::HashSet::new();
        let mut columns: Vec<(i32, String)> = Vec::new();
        for tuple in self.tuples.values() {
            if tuple.table_id == Some(table_id) {
                for &sid in &tuple.slot_ids {
                    if let Ok(s) = self.get_slot(sid) {
                        if s.is_materialized && seen.insert(s.col_name.clone()) {
                            columns.push((s.column_pos, s.col_name.clone()));
                        }
                    }
                }
            }
        }
        columns.sort_by_key(|(pos, _)| *pos);

        columns
            .iter()
            .position(|(_, name)| *name == slot.col_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "slot {} (name={}) not found in table columns",
                    slot_id,
                    slot.col_name
                )
            })
    }

    /// Resolve a slot's column index by name against the full table schema of a reference tuple.
    ///
    /// Used when an expression references a slot from a different tuple (e.g., aggregation
    /// intermediate) but the data comes from a file table identified by `ref_tuple_id`.
    pub fn slot_table_index_by_name(&self, slot_id: i32, ref_tuple_id: i32) -> Result<usize> {
        let slot = self.get_slot(slot_id)?;
        let ref_tuple = self.get_tuple(ref_tuple_id)?;
        let table_id = ref_tuple
            .table_id
            .ok_or_else(|| anyhow::anyhow!("ref tuple {} has no table_id", ref_tuple_id))?;

        // Build full table column list.
        let mut seen = std::collections::HashSet::new();
        let mut columns: Vec<(i32, String)> = Vec::new();
        for tuple in self.tuples.values() {
            if tuple.table_id == Some(table_id) {
                for &sid in &tuple.slot_ids {
                    if let Ok(s) = self.get_slot(sid) {
                        if s.is_materialized && seen.insert(s.col_name.clone()) {
                            columns.push((s.column_pos, s.col_name.clone()));
                        }
                    }
                }
            }
        }
        columns.sort_by_key(|(pos, _)| *pos);

        columns
            .iter()
            .position(|(_, name)| *name == slot.col_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "slot {} (name={}) not found in table {} columns",
                    slot_id,
                    slot.col_name,
                    table_id
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use doris_thrift::types::TPrimitiveType;

    #[test]
    fn test_basic_slot_resolution() {
        let desc_tbl = make_desc_table(
            vec![(0, None)],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "name", TPrimitiveType::VARCHAR),
                (2, 0, 2, "value", TPrimitiveType::DOUBLE),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();

        assert_eq!(desc.slot_column_index(0).unwrap(), 0);
        assert_eq!(desc.slot_column_index(1).unwrap(), 1);
        assert_eq!(desc.slot_column_index(2).unwrap(), 2);
    }

    #[test]
    fn test_slot_not_found() {
        let desc_tbl = make_desc_table(
            vec![(0, None)],
            vec![(0, 0, 0, "id", TPrimitiveType::INT)],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        assert!(desc.get_slot(999).is_err());
    }

    #[test]
    fn test_tuple_not_found() {
        let desc_tbl = make_desc_table(vec![(0, None)], vec![]);
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        assert!(desc.get_tuple(999).is_err());
    }

    #[test]
    fn test_global_index_single_tuple() {
        let desc_tbl = make_desc_table(
            vec![(0, None)],
            vec![
                (10, 0, 0, "a", TPrimitiveType::INT),
                (11, 0, 1, "b", TPrimitiveType::INT),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        assert_eq!(desc.slot_global_index(10, &[0]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(11, &[0]).unwrap(), 1);
    }

    #[test]
    fn test_global_index_multi_tuple() {
        // Simulates a join result with two tuples.
        let desc_tbl = make_desc_table(
            vec![(0, None), (1, None)],
            vec![
                (10, 0, 0, "a", TPrimitiveType::INT),
                (11, 0, 1, "b", TPrimitiveType::INT),
                (20, 1, 0, "c", TPrimitiveType::INT),
                (21, 1, 1, "d", TPrimitiveType::INT),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        assert_eq!(desc.slot_global_index(10, &[0, 1]).unwrap(), 0);
        assert_eq!(desc.slot_global_index(11, &[0, 1]).unwrap(), 1);
        assert_eq!(desc.slot_global_index(20, &[0, 1]).unwrap(), 2);
        assert_eq!(desc.slot_global_index(21, &[0, 1]).unwrap(), 3);
    }

    #[test]
    fn test_table_named_struct_single_tuple() {
        let desc_tbl = make_desc_table(
            vec![(0, Some(100))],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "name", TPrimitiveType::VARCHAR),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        let schema = desc.table_named_struct(0).unwrap();
        assert_eq!(schema.names, vec!["id", "name"]);
    }

    #[test]
    fn test_table_named_struct_multi_tuple_same_table() {
        // Two tuples referencing the same table_id — should merge columns.
        let desc_tbl = make_desc_table(
            vec![(0, Some(100)), (1, Some(100))],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "name", TPrimitiveType::VARCHAR),
                (2, 1, 2, "age", TPrimitiveType::INT),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        let schema = desc.table_named_struct(0).unwrap();
        assert_eq!(schema.names, vec!["id", "name", "age"]);
    }

    #[test]
    fn test_table_column_override() {
        let desc_tbl = make_desc_table(
            vec![(0, None)],
            vec![
                (0, 0, 0, "city", TPrimitiveType::VARCHAR),
                (1, 0, 1, "pop", TPrimitiveType::BIGINT),
            ],
        );
        let mut desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        desc.set_table_column_override(
            "mytable".to_string(),
            vec!["city".to_string(), "state".to_string(), "pop".to_string()],
        );

        // slot_index_in_override should resolve by name against the override.
        assert_eq!(desc.slot_index_in_override(0, "mytable").unwrap(), 0); // city → 0
        assert_eq!(desc.slot_index_in_override(1, "mytable").unwrap(), 2); // pop → 2
    }

    #[test]
    fn test_find_slot_by_name() {
        let desc_tbl = make_desc_table(
            vec![(0, None)],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "name", TPrimitiveType::VARCHAR),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        assert!(desc.find_slot_by_name("id").is_some());
        assert!(desc.find_slot_by_name("name").is_some());
        assert!(desc.find_slot_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_count_materialized_columns() {
        let desc_tbl = make_desc_table(
            vec![(0, None), (1, None)],
            vec![
                (0, 0, 0, "a", TPrimitiveType::INT),
                (1, 0, 1, "b", TPrimitiveType::INT),
                (2, 1, 0, "c", TPrimitiveType::INT),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        assert_eq!(desc.count_materialized_columns(&[0]), 2);
        assert_eq!(desc.count_materialized_columns(&[1]), 1);
        assert_eq!(desc.count_materialized_columns(&[0, 1]), 3);
    }

    #[test]
    fn test_doris_internal_columns_skipped() {
        // __DORIS_ prefixed columns should be skipped.
        let desc_tbl = make_desc_table(
            vec![(0, None)],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "__DORIS_DELETE_SIGN__", TPrimitiveType::INT),
                (2, 0, 2, "name", TPrimitiveType::VARCHAR),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        // __DORIS_ column should not be in the descriptor table.
        assert!(desc.get_slot(1).is_err());
        // Other columns should be present.
        assert_eq!(desc.count_materialized_columns(&[0]), 2);
    }
}
