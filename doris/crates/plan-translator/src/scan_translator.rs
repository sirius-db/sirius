//! Scan translation: FILE_SCAN_NODE → Substrait ReadRel with NamedTable.
//!
//! The ReadRel uses NamedTable with the table name from the scan node.
//! The C++ bridge registers Parquet files as DuckDB tables before executing.

use anyhow::{Context, Result};

use doris_thrift::plan_nodes::{TFileScanRangeParams, TPlanNode};
use substrait::proto::read_rel;
use substrait::proto::{rel, ReadRel, Rel};

use crate::descriptor_table::DescriptorTable;

/// Translate a FILE_SCAN_NODE into a Substrait ReadRel.
pub fn translate_file_scan(
    node: &TPlanNode,
    desc: &DescriptorTable,
    _scan_params: Option<&TFileScanRangeParams>,
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

    // Build schema from descriptor table.
    let base_schema = desc.tuple_to_named_struct(tuple_id)?;

    // Get table name.
    let table_name = node
        .file_scan_node
        .as_ref()
        .and_then(|fsn| fsn.table_name.clone())
        .unwrap_or_else(|| format!("scan_{}", node.node_id));

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
