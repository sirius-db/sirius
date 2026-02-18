//! Scan translation: FILE_SCAN_NODE → Substrait ReadRel with NamedTable.
//!
//! The ReadRel uses NamedTable with the table name from the scan node.
//! The C++ bridge registers Parquet files as DuckDB tables before executing.

use std::collections::HashMap;

use anyhow::{Context, Result};

use doris_thrift::plan_nodes::{TFileScanRangeParams, TPlanNode};
use substrait::proto::r#type;
use substrait::proto::read_rel;
use substrait::proto::{rel, NamedStruct, ReadRel, Rel};

use crate::descriptor_table::DescriptorTable;
use crate::type_mapper;

/// Translate a FILE_SCAN_NODE into a Substrait ReadRel.
///
/// When `table_schemas` contains the table name, the DuckDB table's actual column list
/// is used for the ReadRel base_schema (handles TVF late materialization where the
/// descriptor table's scan tuple has only a subset of columns).
pub fn translate_file_scan(
    node: &TPlanNode,
    desc: &DescriptorTable,
    _scan_params: Option<&TFileScanRangeParams>,
    table_schemas: &HashMap<String, Vec<String>>,
) -> Result<Rel> {
    // Get the tuple_id for this scan node's output schema.
    let tuple_id = if let Some(fsn) = &node.file_scan_node {
        fsn.tuple_id.context("TFileScanNode has no tuple_id")?
    } else {
        *node
            .row_tuples
            .first()
            .context("FILE_SCAN_NODE has no row_tuples")?
    };

    // Get table name (append node_id for uniqueness in self-joins).
    let table_name = node
        .file_scan_node
        .as_ref()
        .and_then(|fsn| fsn.table_name.clone())
        .map(|name| format!("{}_{}", name, node.node_id))
        .unwrap_or_else(|| format!("scan_{}", node.node_id));

    // Build the ReadRel base_schema.
    // DuckDB's from_substrait maps ReadRel columns to table columns by POSITION,
    // so the base_schema MUST include ALL table columns in the correct order.
    // Projection to the output subset happens via ProjectRel added by translate_fragment.
    let base_schema = if let Some(columns) = table_schemas.get(&table_name) {
        build_schema_from_columns(columns, desc, tuple_id)?
    } else {
        desc.table_named_struct(tuple_id)?
    };
    tracing::info!(
        tuple_id,
        schema_names = ?base_schema.names,
        "FILE_SCAN_NODE ReadRel base_schema"
    );

    Ok(Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(base_schema),
            read_type: Some(read_rel::ReadType::NamedTable(read_rel::NamedTable {
                names: vec![table_name],
                ..Default::default()
            })),
            ..Default::default()
        }))),
    })
}

/// Build a NamedStruct for an exchange table from column names.
///
/// Used by EXCHANGE_NODE(0 children) → ReadRel translation.
/// Resolves types from the descriptor table, falling back to nullable I64.
pub fn build_exchange_schema(
    columns: &[String],
    desc: &DescriptorTable,
) -> Result<NamedStruct> {
    build_schema_from_columns(columns, desc, 0)
}

/// Build a NamedStruct from DuckDB column names, resolving types from the descriptor table.
///
/// Looks up each column name across all tuples to find its Substrait type.
/// Falls back to nullable I64 for columns not found in the descriptor table.
fn build_schema_from_columns(
    columns: &[String],
    desc: &DescriptorTable,
    _tuple_id: i32,
) -> Result<NamedStruct> {
    let mut names = Vec::new();
    let mut types = Vec::new();

    for col_name in columns {
        names.push(col_name.clone());
        // Find this column's type from any slot in the descriptor table.
        let substrait_type = desc
            .find_slot_by_name(col_name)
            .map(|s| s.substrait_type.clone())
            .unwrap_or_else(|| {
                // Fallback: nullable I64 if column not found in descriptor table.
                substrait::proto::Type {
                    kind: Some(type_mapper::nullable_i64()),
                }
            });
        types.push(substrait_type);
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
