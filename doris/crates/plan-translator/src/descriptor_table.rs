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

        // Parse slot descriptors.
        if let Some(slot_descs) = &desc_tbl.slot_descriptors {
            for sd in slot_descs {
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

        Ok(Self { slots, tuples })
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
}
