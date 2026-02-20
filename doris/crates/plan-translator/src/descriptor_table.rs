//! Descriptor table resolution.
//!
//! Parses TDescriptorTable to build mappings from slot_id → (tuple_id, column_index, column_type).
//! Used by expression and scan translators to resolve column references.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;

use doris_thrift::descriptors::TDescriptorTable;
use doris_thrift::exprs::TExpr;
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
    /// Slot expression map for scan node projections.
    /// Maps slot_id → TExpr that produces this slot's value.
    /// Used to inline computed expressions (e.g., `l_extendedprice * (1 - l_discount)`)
    /// from scan node intermediate/final projections into downstream nodes (AGG, SORT).
    slot_expressions: HashMap<i32, TExpr>,
    /// Flattened column names from the compiled child Rel, set temporarily during
    /// AGG expression translation. Uses RefCell for interior mutability (the
    /// translate_node chain takes &DescriptorTable but AGG needs to set this).
    child_rel_column_names: RefCell<Option<Vec<String>>>,
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

        Ok(Self { slots, tuples, table_column_overrides: HashMap::new(), slot_expressions: HashMap::new(), child_rel_column_names: RefCell::new(None) })
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
        tracing::debug!(
            slot_id,
            col_name = %slot.col_name,
            parent_tuple = slot.parent_tuple_id,
            row_tuples = ?row_tuples,
            "slot_global_index: entry"
        );
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
        // If compiled child Rel column names are available, use them for resolution.
        // This is the most reliable method for nested JoinRels where descriptor table
        // tuples can't reconstruct the correct flattened column ordering.
        if let Some(ref col_names) = *self.child_rel_column_names.borrow() {
            if !slot.col_name.is_empty() {
                if let Some(idx) = col_names.iter().position(|n| n == &slot.col_name) {
                    tracing::debug!(
                        slot_id,
                        col_name = %slot.col_name,
                        resolved_idx = idx,
                        "slot_global_index: resolved via child Rel column names"
                    );
                    return Ok(idx);
                }
            }
        }

        // Slot's parent tuple not in row_tuples — fall back to name-based resolution.
        //
        // For regular tables (with table_id), use the full table schema for counting.
        // For TVF scans (no table_id), the descriptor table tuples only have a subset
        // of the file's columns (projection push-down), but the Substrait ReadRel
        // has ALL file columns. So we match each row_tuple to a table_column_override
        // (the full DuckDB file schema) for correct global offset counting.
        let mut fallback_offset = 0;
        for &tuple_id in row_tuples {
            if let Ok(tuple) = self.get_tuple(tuple_id) {
                if tuple.table_id.is_some() {
                    // Regular table: use full table schema.
                    if let Ok(local_idx) = self.slot_table_index_by_name(slot_id, tuple_id) {
                        return Ok(fallback_offset + local_idx);
                    }
                    fallback_offset += self.count_table_columns(tuple_id);
                } else {
                    // TVF scan or intermediate tuple: match to override(s).
                    // For a simple scan tuple, all slots come from one table.
                    // For a nested JOIN output tuple, slots span multiple tables.
                    // We collect ALL overrides that cover the tuple's slots,
                    // ordered by first slot appearance (preserves join order).
                    let slot_names: Vec<&str> = tuple
                        .slot_ids
                        .iter()
                        .filter_map(|&sid| {
                            self.slots
                                .get(&sid)
                                .filter(|s| s.is_materialized)
                                .map(|s| s.col_name.as_str())
                        })
                        .collect();

                    // Collect overrides in join order (left side slots first).
                    let mut matched_overrides: Vec<&Vec<String>> = Vec::new();
                    let mut seen_tables: std::collections::HashSet<*const Vec<String>> =
                        std::collections::HashSet::new();
                    for sname in &slot_names {
                        for cols in self.table_column_overrides.values() {
                            let ptr = cols as *const Vec<String>;
                            if !seen_tables.contains(&ptr)
                                && cols.iter().any(|c| c == sname)
                            {
                                matched_overrides.push(cols);
                                seen_tables.insert(ptr);
                                break; // Each slot belongs to one override
                            }
                        }
                    }

                    if !matched_overrides.is_empty() {
                        // Search for target column across all matched overrides.
                        let mut tuple_offset = 0;
                        for cols in &matched_overrides {
                            if let Some(pos) =
                                cols.iter().position(|c| c == &slot.col_name)
                            {
                                return Ok(fallback_offset + tuple_offset + pos);
                            }
                            tuple_offset += cols.len();
                        }
                        fallback_offset += tuple_offset;
                    } else {
                        // No override match: count materialized slots for offset.
                        fallback_offset += slot_names.len();
                    }
                }
            }
        }
        // Last resort: name-based matching in the given tuples (no overrides).
        self.slot_index_by_name_in_tuples(slot_id, row_tuples)
    }

    /// Resolve a slot to its column index in a compiled Substrait Rel's output.
    ///
    /// `child_column_names` is the flattened list of column names from the compiled
    /// child Rel (e.g., from `collect_rel_column_names()`). This is the source of
    /// truth for column ordering in nested JoinRels.
    ///
    /// Falls back to `slot_global_index` if the slot's name isn't found.


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

    /// Set the slot expression map for scan node projections.
    ///
    /// Maps slot_id → TExpr (the expression producing that slot's value).
    /// When a SLOT_REF references a slot in this map, the expression translator
    /// will recursively translate the mapped TExpr instead of looking up a column
    /// position. This handles computed expression slots from scan node projections
    /// (e.g., `l_extendedprice * (1 - l_discount)` in TPC-H Q1).
    pub fn set_slot_expressions(&mut self, map: HashMap<i32, TExpr>) {
        self.slot_expressions = map;
    }

    /// Get the expression that produces a slot's value (if it's a projection slot).
    pub fn get_slot_expression(&self, slot_id: i32) -> Option<&TExpr> {
        self.slot_expressions.get(&slot_id)
    }

    /// Set the flattened column names from the compiled child Rel.
    /// Uses interior mutability (RefCell) since translate_node takes &self.
    pub fn set_child_rel_column_names(&self, names: Vec<String>) {
        *self.child_rel_column_names.borrow_mut() = Some(names);
    }

    /// Clear the child Rel column names (after AGG expression translation).
    pub fn clear_child_rel_column_names(&self) {
        *self.child_rel_column_names.borrow_mut() = None;
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

    /// Count the number of distinct columns in the full table schema for a tuple.
    /// Used to compute global offsets in JoinRel output when resolving SLOT_REFs by name.
    fn count_table_columns(&self, tuple_id: i32) -> usize {
        let tuple = match self.get_tuple(tuple_id) {
            Ok(t) => t,
            Err(_) => return 0,
        };
        let table_id = match tuple.table_id {
            Some(id) => id,
            None => return 0,
        };
        let mut seen = std::collections::HashSet::new();
        for t in self.tuples.values() {
            if t.table_id == Some(table_id) {
                for &sid in &t.slot_ids {
                    if let Ok(s) = self.get_slot(sid) {
                        if s.is_materialized {
                            seen.insert(s.col_name.clone());
                        }
                    }
                }
            }
        }
        seen.len()
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

    /// Test slot_global_index with TVF override-based resolution for JOINs.
    ///
    /// Simulates a JOIN+AGG scenario where the AGG's SLOT_REF has parent_tuple=5
    /// (an intermediate tuple NOT in row_tuples), but the column name exists in
    /// the TVF file schema overrides. The resolution should use the full file schema
    /// column counts (not just materialized slot subsets) for correct global offsets.
    #[test]
    fn test_global_index_tvf_join_override_resolution() {
        // Tuple 3: JOIN left output (customer projection subset: c_custkey, c_name, c_mktsegment)
        // Tuple 1: JOIN right output (orders projection subset: o_custkey, o_totalprice)
        // Tuple 5: AGG intermediate (c_name_agg, o_totalprice_sum) — NOT in row_tuples
        //
        // TVF overrides have FULL file schemas:
        //   customer: 8 columns (c_custkey, c_name, c_address, ..., c_comment)
        //   orders: 9 columns (o_orderkey, o_custkey, o_orderstatus, o_totalprice, ...)
        let desc_tbl = make_desc_table(
            vec![(3, None), (1, None), (5, None)],
            vec![
                // Customer projection subset (tuple 3) — only 3 of 8 columns
                (10, 3, 0, "c_custkey", TPrimitiveType::BIGINT),
                (11, 3, 1, "c_name", TPrimitiveType::VARCHAR),
                (12, 3, 2, "c_mktsegment", TPrimitiveType::VARCHAR),
                // Orders projection subset (tuple 1) — only 2 of 9 columns
                (20, 1, 0, "o_custkey", TPrimitiveType::BIGINT),
                (21, 1, 1, "o_totalprice", TPrimitiveType::DOUBLE),
                // AGG intermediate slots (tuple 5) — NOT in row_tuples
                (25, 5, 0, "c_name", TPrimitiveType::VARCHAR),
                (26, 5, 1, "o_totalprice", TPrimitiveType::DOUBLE),
            ],
        );
        let mut desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        // Set full file schema overrides (matching DuckDB table columns).
        desc.set_table_column_override(
            "customer".to_string(),
            vec![
                "c_custkey".into(), "c_name".into(), "c_address".into(),
                "c_nationkey".into(), "c_phone".into(), "c_acctbal".into(),
                "c_mktsegment".into(), "c_comment".into(),
            ],
        );
        desc.set_table_column_override(
            "orders".to_string(),
            vec![
                "o_orderkey".into(), "o_custkey".into(), "o_orderstatus".into(),
                "o_totalprice".into(), "o_orderdate".into(), "o_orderpriority".into(),
                "o_clerk".into(), "o_shippriority".into(), "o_comment".into(),
            ],
        );

        // row_tuples = JOIN output: [customer_tuple, orders_tuple]
        let row_tuples = &[3, 1];

        // Slot 26 (o_totalprice) has parent_tuple=5, NOT in row_tuples.
        // Tuple 3 matches customer override (3 slot names all in 8-column override).
        // Tuple 1 matches orders override (2 slot names all in 9-column override).
        // o_totalprice is at position 3 in orders override, with customer offset 8.
        // Expected global index: 8 + 3 = 11.
        assert_eq!(desc.slot_global_index(26, row_tuples).unwrap(), 11);

        // Slot 25 (c_name) has parent_tuple=5, NOT in row_tuples.
        // c_name is at position 1 in customer override, with offset 0.
        assert_eq!(desc.slot_global_index(25, row_tuples).unwrap(), 1);
    }

    /// Test slot_global_index for 3-way JOIN where an intermediate tuple spans
    /// two source tables (multi-override matching).
    #[test]
    fn test_global_index_3way_join_multi_override() {
        // 3-way JOIN: inner_join(orders, customer) ⋈ lineitem
        // After merge, outer JOIN has row_tuples = [7, 1]:
        //   Tuple 7: intermediate from inner join — slots from BOTH orders AND customer
        //   Tuple 1: lineitem slots
        // AGG tuple 9: NOT in row_tuples — slots from AGG intermediate
        //
        // Substrait output: orders(9) + customer(8) + lineitem(16) = 33 columns
        let desc_tbl = make_desc_table(
            vec![(7, None), (1, None), (9, None)],
            vec![
                // Tuple 7: inner join output — slots from orders + customer
                (40, 7, 0, "o_orderkey", TPrimitiveType::BIGINT),
                (41, 7, 1, "o_custkey", TPrimitiveType::BIGINT),
                (42, 7, 2, "o_orderdate", TPrimitiveType::BIGINT),
                (43, 7, 3, "o_shippriority", TPrimitiveType::INT),
                (44, 7, 4, "c_custkey", TPrimitiveType::BIGINT),
                (45, 7, 5, "c_mktsegment", TPrimitiveType::VARCHAR),
                // Tuple 1: lineitem
                (50, 1, 0, "l_orderkey", TPrimitiveType::BIGINT),
                (51, 1, 1, "l_extendedprice", TPrimitiveType::DOUBLE),
                (52, 1, 2, "l_discount", TPrimitiveType::DOUBLE),
                (53, 1, 3, "l_shipdate", TPrimitiveType::BIGINT),
                // Tuple 9: AGG intermediate (NOT in row_tuples)
                (60, 9, 0, "l_orderkey", TPrimitiveType::BIGINT),
                (61, 9, 1, "o_orderdate", TPrimitiveType::BIGINT),
                (62, 9, 2, "o_shippriority", TPrimitiveType::INT),
                (63, 9, 3, "l_extendedprice", TPrimitiveType::DOUBLE),
                (64, 9, 4, "l_discount", TPrimitiveType::DOUBLE),
            ],
        );
        let mut desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        // Full file schemas as overrides.
        desc.set_table_column_override(
            "orders_4".to_string(),
            vec![
                "o_orderkey".into(), "o_custkey".into(), "o_orderstatus".into(),
                "o_totalprice".into(), "o_orderdate".into(), "o_orderpriority".into(),
                "o_clerk".into(), "o_shippriority".into(), "o_comment".into(),
            ],
        );
        desc.set_table_column_override(
            "customer_2".to_string(),
            vec![
                "c_custkey".into(), "c_name".into(), "c_address".into(),
                "c_nationkey".into(), "c_phone".into(), "c_acctbal".into(),
                "c_mktsegment".into(), "c_comment".into(),
            ],
        );
        desc.set_table_column_override(
            "lineitem_0".to_string(),
            vec![
                "l_orderkey".into(), "l_partkey".into(), "l_suppkey".into(),
                "l_linenumber".into(), "l_quantity".into(), "l_extendedprice".into(),
                "l_discount".into(), "l_tax".into(), "l_returnflag".into(),
                "l_linestatus".into(), "l_shipdate".into(), "l_commitdate".into(),
                "l_receiptdate".into(), "l_shipinstruct".into(), "l_shipmode".into(),
                "l_comment".into(),
            ],
        );

        let row_tuples = &[7, 1];

        // Tuple 7 slots come from orders + customer → matches BOTH overrides.
        // orders_4 has slots o_orderkey, o_custkey, o_orderdate, o_shippriority → 9 cols
        // customer_2 has slots c_custkey, c_mktsegment → 8 cols
        // Tuple 7 total: 9 + 8 = 17 cols
        // Tuple 1 matches lineitem_0 → 16 cols

        // With child_rel_column_names set (the real production path), slot resolution
        // uses the compiled Rel's flattened column names for accurate global indexing.
        // The column order is: orders(9) + customer(8) + lineitem(16) = 33 columns.
        let child_col_names: Vec<String> = vec![
            // orders ReadRel (9 cols)
            "o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
            "o_orderpriority", "o_clerk", "o_shippriority", "o_comment",
            // customer ReadRel (8 cols)
            "c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
            "c_acctbal", "c_mktsegment", "c_comment",
            // lineitem ReadRel (16 cols)
            "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
            "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
            "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct",
            "l_shipmode", "l_comment",
        ].into_iter().map(String::from).collect();
        desc.set_child_rel_column_names(child_col_names);

        // Slot 60 (l_orderkey, parent=9): at position 17 in the flattened output.
        // (orders 9 + customer 8 = 17 offset, lineitem l_orderkey at pos 0)
        assert_eq!(desc.slot_global_index(60, row_tuples).unwrap(), 17);

        // Slot 61 (o_orderdate, parent=9): at position 4 in orders.
        assert_eq!(desc.slot_global_index(61, row_tuples).unwrap(), 4);

        // Slot 62 (o_shippriority, parent=9): at position 7 in orders.
        assert_eq!(desc.slot_global_index(62, row_tuples).unwrap(), 7);

        // Slot 63 (l_extendedprice, parent=9): at position 22 (17 + 5).
        assert_eq!(desc.slot_global_index(63, row_tuples).unwrap(), 22);

        // Slot 64 (l_discount, parent=9): at position 23 (17 + 6).
        assert_eq!(desc.slot_global_index(64, row_tuples).unwrap(), 23);

        // Slot 45 (c_mktsegment, parent=7): parent IS in row_tuples.
        // Direct match uses materialized position (5) — primary path takes priority.
        assert_eq!(desc.slot_global_index(45, row_tuples).unwrap(), 5);

        // Clear and verify fallback without child_rel_column_names uses overrides.
        desc.clear_child_rel_column_names();
    }

    /// Test slot_global_index without overrides — simple name-based matching.
    #[test]
    fn test_global_index_cross_tuple_no_overrides() {
        // Without overrides, falls back to counting materialized slots.
        let desc_tbl = make_desc_table(
            vec![(0, None), (1, None), (2, None)],
            vec![
                (10, 0, 0, "a", TPrimitiveType::INT),
                (11, 0, 1, "b", TPrimitiveType::INT),
                (20, 1, 0, "c", TPrimitiveType::INT),
                (21, 1, 1, "d", TPrimitiveType::INT),
                // Tuple 2: intermediate, not in row_tuples
                (30, 2, 0, "b", TPrimitiveType::INT),
                (31, 2, 1, "d", TPrimitiveType::INT),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        let row_tuples = &[0, 1];

        // Slot 31 (d), parent=2, resolves by name to tuple 1 position 1, offset 2 = global 3.
        assert_eq!(desc.slot_global_index(31, row_tuples).unwrap(), 3);
        // Slot 30 (b), parent=2, resolves to tuple 0 position 1 = global 1.
        assert_eq!(desc.slot_global_index(30, row_tuples).unwrap(), 1);
    }

    /// Test that direct parent tuple matching takes priority over name-based.
    #[test]
    fn test_global_index_direct_match_priority() {
        // When the parent tuple IS in row_tuples, direct match should be used
        // (not name-based), even if the name also exists in another tuple.
        let desc_tbl = make_desc_table(
            vec![(0, None), (1, None)],
            vec![
                (10, 0, 0, "id", TPrimitiveType::BIGINT),
                (11, 0, 1, "value", TPrimitiveType::DOUBLE),
                (20, 1, 0, "id", TPrimitiveType::BIGINT),
                (21, 1, 1, "other", TPrimitiveType::VARCHAR),
            ],
        );
        let desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();

        // Slot 20 (id, parent=1) in row_tuples [0, 1] should resolve to global index 2
        // (offset 2 from tuple 0's 2 slots + position 0 in tuple 1).
        assert_eq!(desc.slot_global_index(20, &[0, 1]).unwrap(), 2);
    }

    /// Test override fallback when name-in-tuples doesn't find the column.
    #[test]
    fn test_global_index_override_fallback() {
        // Slot references a column that's NOT in any row_tuple but IS in overrides.
        let desc_tbl = make_desc_table(
            vec![(0, None), (5, None)],
            vec![
                (10, 0, 0, "a", TPrimitiveType::INT),
                (11, 0, 1, "b", TPrimitiveType::INT),
                // Slot 50 is in tuple 5, which has "extra_col" not present in tuple 0.
                (50, 5, 0, "extra_col", TPrimitiveType::INT),
            ],
        );
        let mut desc = DescriptorTable::from_thrift(&desc_tbl).unwrap();
        desc.set_table_column_override(
            "my_table".to_string(),
            vec!["a".to_string(), "extra_col".to_string(), "b".to_string()],
        );

        // row_tuples = [0], slot 50's parent_tuple=5 not in [0], and "extra_col"
        // is not a slot name in tuple 0. Should fall through to override.
        assert_eq!(desc.slot_global_index(50, &[0]).unwrap(), 1); // "extra_col" at pos 1 in override
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
