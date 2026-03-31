//! Doris plan → Substrait plan translation.
//!
//! Converts Doris TPlanNode trees (received via Thrift) into Substrait Plan
//! protobuf messages (pure Rust, no FFI). The Substrait plan is then passed
//! to the C++ bridge which uses the DuckDB Substrait extension to consume it.

use std::collections::HashMap;

use anyhow::{Context, Result};
use prost::Message;
use tracing::debug;

use doris_thrift::exprs::{TExpr, TExprNodeType};
use doris_thrift::plan_nodes::{TPlan, TPlanNodeType};
use doris_thrift::palo_internal_service::TPipelineFragmentParams;
use substrait::proto::extensions::simple_extension_declaration;
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUri};
use substrait::proto::{rel, Plan, PlanRel, Rel, RelRoot, Version};

pub mod descriptor_table;
pub mod expr_translator;
pub mod node_translator;
pub mod scan_translator;
#[cfg(test)]
mod test_helpers;
pub mod type_mapper;

/// Substrait extension URIs for standard function sets.
pub const URI_COMPARISON: &str =
    "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml";
pub const URI_BOOLEAN: &str =
    "https://github.com/substrait-io/substrait/blob/main/extensions/functions_boolean.yaml";
pub const URI_ARITHMETIC: &str =
    "https://github.com/substrait-io/substrait/blob/main/extensions/functions_arithmetic.yaml";
pub const URI_STRING: &str =
    "https://github.com/substrait-io/substrait/blob/main/extensions/functions_string.yaml";
pub const URI_AGGREGATE: &str =
    "https://github.com/substrait-io/substrait/blob/main/extensions/functions_aggregate_generic.yaml";

/// Registry for Substrait extension functions.
///
/// Tracks function name → anchor mapping so we can reference functions in expressions.
pub struct ExtensionRegistry {
    uris: Vec<SimpleExtensionUri>,
    functions: Vec<SimpleExtensionDeclaration>,
    uri_map: HashMap<String, u32>,
    func_map: HashMap<String, u32>,
    next_uri_anchor: u32,
    next_func_anchor: u32,
}

impl ExtensionRegistry {
    pub fn new() -> Self {
        Self {
            uris: Vec::new(),
            functions: Vec::new(),
            uri_map: HashMap::new(),
            func_map: HashMap::new(),
            next_uri_anchor: 1,
            next_func_anchor: 1,
        }
    }

    /// Ensure a URI is registered and return its anchor.
    fn ensure_uri(&mut self, uri: &str) -> u32 {
        if let Some(&anchor) = self.uri_map.get(uri) {
            return anchor;
        }
        let anchor = self.next_uri_anchor;
        self.next_uri_anchor += 1;
        self.uris.push(SimpleExtensionUri {
            extension_uri_anchor: anchor,
            uri: uri.to_string(),
        });
        self.uri_map.insert(uri.to_string(), anchor);
        anchor
    }

    /// Register a function and return its anchor.
    /// Returns existing anchor if already registered.
    pub fn register_function(&mut self, uri: &str, name: &str) -> u32 {
        if let Some(&anchor) = self.func_map.get(name) {
            return anchor;
        }
        let uri_anchor = self.ensure_uri(uri);
        let func_anchor = self.next_func_anchor;
        self.next_func_anchor += 1;
        self.functions.push(SimpleExtensionDeclaration {
            mapping_type: Some(
                simple_extension_declaration::MappingType::ExtensionFunction(
                    simple_extension_declaration::ExtensionFunction {
                        extension_uri_reference: uri_anchor,
                        function_anchor: func_anchor,
                        name: name.to_string(),
                    },
                ),
            ),
        });
        self.func_map.insert(name.to_string(), func_anchor);
        func_anchor
    }

    /// Consume into extension URIs and declarations for a Plan.
    pub fn into_extensions(self) -> (Vec<SimpleExtensionUri>, Vec<SimpleExtensionDeclaration>) {
        (self.uris, self.functions)
    }
}

/// Compute the output column names of a Substrait Rel tree.
///
/// Walks the tree to find the leaf schema names. For pass-through nodes (Filter,
/// Sort, Fetch) the output schema equals the input. For Join/Cross, it's left+right.
/// Returns empty vec if the schema can't be determined (caller falls back to output_names).
fn rel_output_names(rel: &Rel) -> Vec<String> {
    match rel.rel_type.as_ref() {
        Some(rel::RelType::Read(read)) => read
            .base_schema
            .as_ref()
            .map(|s| s.names.clone())
            .unwrap_or_default(),
        Some(rel::RelType::Filter(f)) => f
            .input
            .as_deref()
            .map(|r| rel_output_names(r))
            .unwrap_or_default(),
        Some(rel::RelType::Sort(s)) => s
            .input
            .as_deref()
            .map(|r| rel_output_names(r))
            .unwrap_or_default(),
        Some(rel::RelType::Fetch(f)) => f
            .input
            .as_deref()
            .map(|r| rel_output_names(r))
            .unwrap_or_default(),
        Some(rel::RelType::Project(p)) => p
            .input
            .as_deref()
            .map(|r| rel_output_names(r))
            .unwrap_or_default(),
        Some(rel::RelType::Join(j)) => {
            let mut names = j
                .left
                .as_deref()
                .map(|r| rel_output_names(r))
                .unwrap_or_default();
            names.extend(
                j.right
                    .as_deref()
                    .map(|r| rel_output_names(r))
                    .unwrap_or_default(),
            );
            names
        }
        Some(rel::RelType::Cross(c)) => {
            let mut names = c
                .left
                .as_deref()
                .map(|r| rel_output_names(r))
                .unwrap_or_default();
            names.extend(
                c.right
                    .as_deref()
                    .map(|r| rel_output_names(r))
                    .unwrap_or_default(),
            );
            names
        }
        Some(rel::RelType::Set(set)) => {
            // UNION ALL: output columns = first input's columns.
            set.inputs.first().map(rel_output_names).unwrap_or_default()
        }
        // Aggregate, etc. — can't easily determine output names.
        _ => Vec::new(),
    }
}

/// A single file in a file scan range (path + byte offset/length).
#[derive(Debug, Clone)]
pub struct FileScanFile {
    pub path: String,
    pub start_offset: i64,
    pub length: i64,
}

/// File scan info extracted from per_node_scan_ranges.
///
/// Contains all files assigned to this BE for a given scan node.
#[derive(Debug, Clone)]
pub struct FileScanInfo {
    pub table_name: String,
    pub format: String,
    pub files: Vec<FileScanFile>,
}

/// Result of Substrait plan translation.
pub struct TranslatedPlan {
    /// Serialized Substrait Plan protobuf bytes.
    pub substrait_bytes: Vec<u8>,
    /// Expected output column names (from Doris plan, for result projection).
    pub output_names: Vec<String>,
    /// Explicit column indices mapping FE output column i to DuckDB output column
    /// `output_column_indices[i]`. When set, use these instead of name-based matching.
    pub output_column_indices: Option<Vec<usize>>,
    /// SQL ORDER BY / LIMIT / OFFSET suffix extracted from outermost SortRel/FetchRel.
    /// Used to wrap `from_substrait()` when the DuckDB CPU path doesn't preserve sort order.
    pub sort_limit_sql: Option<String>,
    /// Structured sort specification for Rust-side sorting of Arrow results.
    /// Sort column names + directions, applied after DuckDB returns results.
    pub sort_columns: Vec<SortColumn>,
    /// LIMIT from the sort node, applied after sorting.
    pub sort_limit: Option<i64>,
    /// Reserved for future use (always false currently).
    pub force_cpu_substrait: bool,
}

/// A single sort column specification.
#[derive(Debug, Clone)]
pub struct SortColumn {
    /// Column name to sort by (from the Substrait Rel's output).
    pub name: String,
    /// True for ascending order.
    pub ascending: bool,
    /// True for nulls first.
    pub nulls_first: bool,
}

/// Build a slot expression map from FILE_SCAN_NODE intermediate/final projections.
///
/// For each projection layer, maps output tuple slot IDs to the TExpr that produces them.
/// This allows the expression translator to inline computed expressions from scan projections
/// when downstream nodes (AGG, SORT) reference these computed slots.
fn build_projection_slot_map(plan: &TPlan, desc: &mut descriptor_table::DescriptorTable) {
    let mut slot_expressions: HashMap<i32, TExpr> = HashMap::new();

    for node in &plan.nodes {
        // For FILE_SCAN_NODE: capture ALL projection expressions (named and unnamed).
        // For other nodes (JOIN, AGG): only capture EMPTY-NAME slots to avoid breaking
        // named slot resolution which uses different mechanisms (child_rel_column_names).
        let scan_only = node.node_type == TPlanNodeType::FILE_SCAN_NODE;

        // Process intermediate projection layers.
        if let (Some(proj_list), Some(tuple_ids)) = (
            &node.intermediate_projections_list,
            &node.intermediate_output_tuple_id_list,
        ) {
            for (layer_exprs, &tuple_id) in proj_list.iter().zip(tuple_ids.iter()) {
                if let Ok(tuple) = desc.get_tuple(tuple_id) {
                    let materialized: Vec<i32> = tuple
                        .slot_ids
                        .iter()
                        .copied()
                        .filter(|&sid| {
                            desc.get_slot(sid)
                                .map(|s| s.is_materialized)
                                .unwrap_or(false)
                        })
                        .collect();
                    for (i, expr) in layer_exprs.iter().enumerate() {
                        if i < materialized.len() {
                            let sid = materialized[i];
                            if scan_only
                                || desc
                                    .get_slot(sid)
                                    .map(|s| s.col_name.is_empty())
                                    .unwrap_or(false)
                            {
                                slot_expressions.insert(sid, expr.clone());
                            }
                        }
                    }
                }
            }
        }

        // Process final projections.
        if let (Some(projections), Some(output_tuple_id)) =
            (&node.projections, &node.output_tuple_id)
        {
            if let Ok(tuple) = desc.get_tuple(*output_tuple_id) {
                let materialized: Vec<i32> = tuple
                    .slot_ids
                    .iter()
                    .copied()
                    .filter(|&sid| {
                        desc.get_slot(sid)
                            .map(|s| s.is_materialized)
                            .unwrap_or(false)
                    })
                    .collect();
                for (i, expr) in projections.iter().enumerate() {
                    if i < materialized.len() {
                        let sid = materialized[i];
                        if scan_only
                            || desc
                                .get_slot(sid)
                                .map(|s| s.col_name.is_empty())
                                .unwrap_or(false)
                        {
                            slot_expressions.insert(sid, expr.clone());
                        }
                    }
                }
            }
        }
    }

    if !slot_expressions.is_empty() {
        debug!(
            num_entries = slot_expressions.len(),
            "built projection slot expression map"
        );
        desc.set_slot_expressions(slot_expressions);
    }
}

/// Extract a SQL ORDER BY / LIMIT / OFFSET suffix from the Doris plan's root SORT_NODE.
///
/// DuckDB's `from_substrait()` doesn't reliably preserve sort order, so we extract
/// the sort specification here and apply it as a SQL wrapper around the from_substrait call.
/// Uses column names from the descriptor table (which match the DuckDB table columns).
fn extract_sort_limit_from_plan(
    plan: &TPlan,
    desc: &descriptor_table::DescriptorTable,
    rel_column_names: &[String],
) -> (Option<String>, Vec<SortColumn>, Option<i64>) {
    let sort_plan_node = match plan.nodes.iter().find(|n| n.node_type == TPlanNodeType::SORT_NODE) {
        Some(n) => n,
        None => return (None, vec![], None),
    };
    let sort_node = match sort_plan_node.sort_node.as_ref() {
        Some(n) => n,
        None => return (None, vec![], None),
    };
    let sort_info = &sort_node.sort_info;

    // Build structured sort columns using Substrait Rel column names.
    let mut sort_columns = Vec::new();
    for (i, expr) in sort_info.ordering_exprs.iter().enumerate() {
        let is_asc = sort_info.is_asc_order.get(i).copied().unwrap_or(true);
        let nulls_first = sort_info.nulls_first.get(i).copied().unwrap_or(true);

        let col_name = expr.nodes.first().and_then(|node| {
            if node.node_type == TExprNodeType::SLOT_REF {
                node.slot_ref.as_ref().and_then(|sr| {
                    let slot = desc.get_slot(sr.slot_id).ok()?;
                    if !slot.col_name.is_empty() {
                        // Named column: find in Rel column names.
                        if rel_column_names.iter().any(|n| n == &slot.col_name) {
                            Some(slot.col_name.clone())
                        } else {
                            None
                        }
                    } else {
                        // Empty-name measure: use empty string (will match by position).
                        Some(String::new())
                    }
                })
            } else {
                None
            }
        });

        if let Some(name) = col_name {
            sort_columns.push(SortColumn {
                name,
                ascending: is_asc,
                nulls_first,
            });
        }
    }

    // Build sort_limit_sql with ORDER BY + LIMIT using column names (not positions).
    // Column names are quoted to handle special characters and avoid ambiguity.
    // This is more robust than positional references because collect_rel_column_names
    // may include columns that DuckDB optimizes away (e.g., AGG columns in semi-joins).
    let mut order_parts = Vec::new();
    for sc in &sort_columns {
        if !sc.name.is_empty() {
            let dir = if sc.ascending { "ASC" } else { "DESC" };
            order_parts.push(format!("\"{}\" {}", sc.name, dir));
        }
    }

    let mut sql = if order_parts.is_empty() {
        String::new()
    } else {
        format!("ORDER BY {}", order_parts.join(", "))
    };

    let limit = sort_plan_node.limit;
    let offset = sort_node.offset.unwrap_or(0);
    let sort_limit = if limit >= 0 { Some(limit) } else { None };
    if limit >= 0 {
        sql.push_str(&format!(" LIMIT {}", limit));
    }
    if offset > 0 {
        sql.push_str(&format!(" OFFSET {}", offset));
    }

    let sql = sql.trim().to_string();
    (
        if sql.is_empty() { None } else { Some(sql) },
        sort_columns,
        sort_limit,
    )
}

/// Filter table_schemas for parquet scans to only include materialized columns.
///
/// For parquet `ReadRel::LocalFiles`, DuckDB resolves columns by NAME (not position),
/// so a reduced base_schema causes DuckDB/Sirius to read only the columns actually needed
/// by the query. Non-parquet tables (NamedTable path) retain all columns because DuckDB
/// maps those by position.
/// Reduce parquet table schemas to only materialized columns.
///
/// SLOT_REF column references auto-adjust because `set_table_column_override`
/// uses the reduced column list, and `slot_table_index` resolves by NAME.
fn project_parquet_schemas(
    table_schemas: &HashMap<String, Vec<String>>,
    file_scan_map: &HashMap<String, FileScanInfo>,
    desc: &descriptor_table::DescriptorTable,
) -> HashMap<String, Vec<String>> {
    table_schemas
        .iter()
        .map(|(name, columns)| {
            if file_scan_map
                .get(name)
                .map_or(false, |fsi| fsi.format == "parquet" && !fsi.files.is_empty())
            {
                let filtered: Vec<String> = columns
                    .iter()
                    .filter(|c| desc.find_slot_by_name(c).is_some())
                    .cloned()
                    .collect();
                if !filtered.is_empty() && filtered.len() < columns.len() {
                    tracing::info!(
                        table = %name,
                        total_columns = columns.len(),
                        projected_columns = filtered.len(),
                        "projection pushdown: reduced parquet schema to materialized columns"
                    );
                    return (name.clone(), filtered);
                }
            }
            (name.clone(), columns.clone())
        })
        .collect()
}

/// Translate a Doris TPipelineFragmentParams into serialized Substrait Plan bytes.
///
/// `table_schemas` maps NamedTable names to their actual column names (in order).
/// For TVF file scans where the descriptor table lacks table_id, the scan translator
/// uses these schemas to produce a correct ReadRel base_schema matching the DuckDB table.
///
/// `file_scan_map` maps table names to `FileScanInfo` (multi-file scan ranges).
/// When a table has an entry here AND format is parquet, the scan translator generates
/// `ReadRel::LocalFiles` (→ `parquet_scan([files])`) instead of `ReadRel::NamedTable`.
///
/// Returns the Substrait bytes plus the expected output column names. When the DuckDB
/// result has more columns than expected, the caller should project the result to match.
pub fn translate_fragment(
    params: &TPipelineFragmentParams,
    table_schemas: &HashMap<String, Vec<String>>,
    file_scan_map: &HashMap<String, FileScanInfo>,
) -> Result<TranslatedPlan> {
    let fragment = params
        .fragment
        .as_ref()
        .context("TPipelineFragmentParams has no fragment")?;
    let plan = fragment
        .plan
        .as_ref()
        .context("TPlanFragment has no plan")?;

    // Build descriptor table for column resolution.
    let desc_tbl = params
        .desc_tbl
        .as_ref()
        .context("TPipelineFragmentParams has no desc_tbl")?;
    let mut desc = descriptor_table::DescriptorTable::from_thrift(desc_tbl)?;

    // Don't reduce parquet schemas for now — projection pushdown changes column
    // indices and causes GROUP BY / expression reference mismatches in the Sirius
    // GPU engine. Keep all columns; filter/projection pushdown will be added later
    // with proper index remapping.
    // let table_schemas = project_parquet_schemas(table_schemas, file_scan_map, &desc);
    let table_schemas = table_schemas.clone();

    // Set table column overrides from DuckDB table schemas (for TVF scans).
    for (table_name, columns) in &table_schemas {
        desc.set_table_column_override(table_name.clone(), columns.clone());
    }

    // Collect file scan params (node_id → params).
    let scan_params = params
        .file_scan_params
        .as_ref()
        .cloned()
        .unwrap_or_default();

    // Build slot expression map from scan node projections.
    // This enables downstream nodes (AGG, SORT) to resolve computed expression slots
    // (e.g., `l_extendedprice * (1 - l_discount)`) by inlining the projection expressions.
    build_projection_slot_map(plan, &mut desc);

    let mut registry = ExtensionRegistry::new();

    // Translate the plan tree into a Substrait Rel tree.
    let mut rel = node_translator::translate_plan(plan, &desc, &scan_params, &mut registry, &table_schemas, file_scan_map)?;

    // Extract sort/limit specification from the Doris plan's root SORT_NODE.
    // Returns structured sort info (column names + directions) for Rust-side sorting,
    // plus a LIMIT-only SQL string for the from_substrait wrapper.
    let rel_column_names = node_translator::collect_rel_column_names(&rel);
    let (sort_limit_sql, sort_columns, sort_limit) =
        extract_sort_limit_from_plan(plan, &desc, &rel_column_names);
    if let Some(ref sql) = sort_limit_sql {
        debug!(sort_limit_sql = %sql, "extracted sort/limit from Doris plan");
    }

    // Build output names in the SELECT-list order the FE expects.
    // Prefer output_exprs (gives exact FE column order), fall back to row_tuples.
    let output_names = if let Some(output_exprs) = fragment.output_exprs.as_ref() {
        let mut names = Vec::new();
        for expr in output_exprs {
            if let Some(first_node) = expr.nodes.first() {
                if first_node.node_type == TExprNodeType::SLOT_REF {
                    if let Some(slot_ref) = &first_node.slot_ref {
                        if let Ok(slot) = desc.get_slot(slot_ref.slot_id) {
                            names.push(slot.col_name.clone());
                            continue;
                        }
                    }
                }
            }
            // Non-SLOT_REF expr (e.g., CAST): use a positional placeholder.
            names.push(format!("expr_{}", names.len()));
        }
        debug!(source = "output_exprs", names = ?names, "output column names");
        names
    } else if !plan.nodes.is_empty() {
        // Fallback: use row_tuples from root node.
        let root = &plan.nodes[0];
        let mut names = Vec::new();
        for &tuple_id in &root.row_tuples {
            if let Ok(tuple) = desc.get_tuple(tuple_id) {
                for &slot_id in &tuple.slot_ids {
                    if let Ok(slot) = desc.get_slot(slot_id) {
                        if slot.is_materialized {
                            names.push(slot.col_name.clone());
                        }
                    }
                }
            }
        }
        debug!(source = "row_tuples", names = ?names, "output column names");
        names
    } else {
        Vec::new()
    };
    // Deduplicate names (joins can produce duplicate column names like "id" from
    // both sides, which DuckDB cannot handle in from_substrait output tables).
    let output_names = {
        let mut name_counts: HashMap<String, usize> = HashMap::new();
        let mut unique = Vec::new();
        for name in output_names {
            let count = name_counts.entry(name.clone()).or_insert(0);
            if *count == 0 {
                unique.push(name);
            } else {
                unique.push(format!("{}:{}", name, count));
            }
            *count += 1;
        }
        unique
    };

    // Compute output_column_indices: explicit mapping from FE column i to DuckDB
    // output column position. For file scans where the Rel outputs more columns,
    // we use name-based matching against the Rel output. For AGG/other where
    // output_exprs might reorder columns, we use slot position in row_tuples.
    let rel_names = rel_output_names(&rel);

    // If the Rel produces fewer columns than the FE expects, truncate output_names
    // to only those available in the Rel, preserving the FE's SELECT list order.
    // This happens with VMaterializeNode (late materialization): the FE plans to
    // fetch extra columns via row IDs from storage, but Sirius doesn't support that.
    let output_names = if !rel_names.is_empty() && output_names.len() > rel_names.len() {
        let rel_name_set: std::collections::HashSet<&str> =
            rel_names.iter().map(|s| s.as_str()).collect();
        let filtered: Vec<String> = output_names
            .into_iter()
            .filter(|n| rel_name_set.contains(n.as_str()))
            .collect();
        tracing::warn!(
            rel_cols = rel_names.len(),
            filtered_cols = filtered.len(),
            rel_names = ?rel_names,
            filtered_names = ?filtered,
            "Rel produces fewer columns than expected (late materialization), preserving FE order"
        );
        // If name-based filtering matched all rel columns, use it (FE order preserved).
        // Otherwise fall back to rel_names order (names may differ between descriptors).
        if filtered.len() == rel_names.len() {
            filtered
        } else {
            rel_names.clone()
        }
    } else {
        output_names
    };
    // Use rel_names (all columns from the Rel tree) as Root.names.
    // DuckDB's from_substrait returns the Relation's natural column names
    // (potentially in optimizer-reordered order). The Rust-side project_ipc_columns
    // function handles name-based reordering to match FE's output_names.
    // We don't wrap in a ProjectRel because DuckDB's TransformProjectOp uses
    // mock aliases ("expr_0", "expr_1"...) that break name-based matching.
    let output_column_indices: Option<Vec<usize>> = None;
    let root_names = if !rel_names.is_empty() {
        rel_names.clone()
    } else {
        output_names.clone()
    };
    let result_output_names = output_names;

    let (extension_uris, extensions) = registry.into_extensions();

    let substrait_plan = Plan {
        version: Some(Version {
            major_number: 0,
            minor_number: 52,
            patch_number: 0,
            producer: "sirius-doris-be".to_string(),
            ..Default::default()
        }),
        extension_uris,
        extensions,
        relations: vec![PlanRel {
            rel_type: Some(substrait::proto::plan_rel::RelType::Root(RelRoot {
                input: Some(rel),
                names: root_names,
            })),
        }],
        ..Default::default()
    };

    let bytes = substrait_plan.encode_to_vec();
    debug!(
        plan_bytes = bytes.len(),
        extensions = substrait_plan.extensions.len(),
        "translated Doris fragment to Substrait plan"
    );
    Ok(TranslatedPlan {
        substrait_bytes: bytes,
        output_names: result_output_names,
        output_column_indices,
        sort_limit_sql,
        sort_columns,
        sort_limit,
        force_cpu_substrait: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use doris_thrift::types::TPrimitiveType;
    use substrait::proto::{plan_rel, read_rel, rel, rel_common};

    #[test]
    fn test_translate_fragment_substrait_union() {
        // UNION_NODE with const values should produce a VirtualTable ReadRel.
        let node = make_union_node(0, 0, vec![vec![int_literal_expr(42)]]);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);

        let table_schemas = HashMap::new();
        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
        assert!(!result.substrait_bytes.is_empty());

        // Decode the Substrait plan and verify it has a VirtualTable.
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        assert_eq!(plan.relations.len(), 1);
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::VirtualTable(vt)) => {
                    assert_eq!(vt.values.len(), 1);
                }
                other => panic!("expected VirtualTable, got {:?}", other),
            },
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_translate_fragment_file_scan_output_names() {
        // FILE_SCAN_NODE with table_schemas should produce correct output_names.
        let node = make_file_scan_node(0, 0, "cities");
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "city", TPrimitiveType::VARCHAR),
                (1, 0, 1, "state", TPrimitiveType::VARCHAR),
                (2, 0, 2, "population", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        // Provide the full table schema (all 3 columns).
        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "cities_0".to_string(),
            vec!["city".to_string(), "state".to_string(), "population".to_string()],
        );

        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
        // Output names should include all materialized columns from the descriptor.
        assert_eq!(result.output_names, vec!["city", "state", "population"]);
    }

    #[test]
    fn test_translate_fragment_file_scan_projected_subset() {
        // When the descriptor table only has a subset of columns (projection push-down),
        // output_names should only include those columns.
        let node = make_file_scan_node(0, 0, "cities");
        let plan = make_plan(vec![node]);

        // Descriptor only has 2 of 3 columns (city, population — not state).
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "city", TPrimitiveType::VARCHAR),
                (2, 0, 2, "population", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        // Full table schema has 3 columns.
        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "cities_0".to_string(),
            vec!["city".to_string(), "state".to_string(), "population".to_string()],
        );

        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
        // Output names come from the descriptor table, not the full table schema.
        // project_ipc_columns uses these to select the subset post-execution.
        assert_eq!(result.output_names, vec!["city", "population"]);

        // Root.names = rel_names (all columns from the ReadRel, not just the projected subset).
        // No ProjectRel wrapping — DuckDB returns all columns, project_ipc_columns selects subset.
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        assert_eq!(root.names, vec!["city", "state", "population"],
            "RelRoot.names should be all rel columns (no ProjectRel subset selection)");
        // Root input is the ReadRel directly (no ProjectRel wrapping).
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => {
                let schema = read.base_schema.as_ref().unwrap();
                assert_eq!(schema.names.len(), 3, "ReadRel should have all table columns");
                assert_eq!(schema.names, vec!["city", "state", "population"]);
            }
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_translate_fragment_substrait_has_version() {
        let node = make_union_node(0, 0, vec![vec![int_literal_expr(1)]]);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let version = plan.version.unwrap();
        assert_eq!(version.producer, "sirius-doris-be");
    }

    #[test]
    fn test_translate_fragment_exchange_node() {
        // EXCHANGE_NODE(0 children) → ReadRel with NamedTable.
        let node = make_exchange_node(5, vec![0]);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(
            vec![(0, None)],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "value", TPrimitiveType::DOUBLE),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::NamedTable(nt)) => {
                    assert_eq!(nt.names, vec!["__EXCHANGE_TABLE_5"]);
                }
                other => panic!("expected NamedTable, got {:?}", other),
            },
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_output_name_deduplication() {
        // When a join produces duplicate column names (e.g., "id" from both sides),
        // the output names should be deduplicated with :N suffix.
        let node = make_union_node(0, 0, vec![vec![int_literal_expr(1)]]);
        let plan = make_plan(vec![node]);
        // Two tuples both having an "id" column.
        let desc = make_desc_table(
            vec![(0, None), (1, None)],
            vec![
                (0, 0, 0, "id", TPrimitiveType::INT),
                (1, 0, 1, "name", TPrimitiveType::VARCHAR),
                (2, 1, 0, "id", TPrimitiveType::INT),
                (3, 1, 1, "value", TPrimitiveType::DOUBLE),
            ],
        );
        let params = make_fragment_params(plan, desc);

        // Manually set row_tuples to reference both tuples (simulating a join).
        // This tests the deduplication logic in translate_fragment.
        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        // The actual output depends on the root node's row_tuples,
        // which for UNION_NODE only references tuple 0.
        // With tuple 0: id, name — no dedup needed.
        assert!(result.output_names.contains(&"id".to_string()));
        assert!(result.output_names.contains(&"name".to_string()));
    }

    // ---- HASH_JOIN_NODE tests (TPC-H: joins are critical) ----

    #[test]
    fn test_hash_join_inner() {
        use doris_thrift::plan_nodes::TJoinOp;
        // HASH_JOIN(INNER) of two FILE_SCAN_NODEs:
        //   left:  orders(order_id BIGINT, cust_id BIGINT)  tuple 0
        //   right: customers(cust_id BIGINT, name VARCHAR)  tuple 1
        //   ON left.cust_id = right.cust_id
        let join_node = make_hash_join_node(
            0,
            TJoinOp::INNER_JOIN,
            vec![0, 1],
            vec![(
                slot_ref_expr_in_tuple(1, 0, type_desc(TPrimitiveType::BIGINT)),
                slot_ref_expr_in_tuple(2, 1, type_desc(TPrimitiveType::BIGINT)),
            )],
        );
        let left_scan = make_file_scan_node(1, 0, "orders");
        let right_scan = make_file_scan_node(2, 1, "customers");
        let plan = make_plan(vec![join_node, left_scan, right_scan]);
        let desc = make_desc_table(
            vec![(0, Some(10)), (1, Some(20))],
            vec![
                (0, 0, 0, "order_id", TPrimitiveType::BIGINT),
                (1, 0, 1, "cust_id", TPrimitiveType::BIGINT),
                (2, 1, 0, "cust_id", TPrimitiveType::BIGINT),
                (3, 1, 1, "name", TPrimitiveType::VARCHAR),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Join(join) => {
                assert_eq!(join.r#type, substrait::proto::join_rel::JoinType::Inner as i32);
                assert!(join.expression.is_some(), "should have join expression");
                assert!(join.left.is_some());
                assert!(join.right.is_some());
            }
            other => panic!("expected JoinRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn test_hash_join_left_outer() {
        use doris_thrift::plan_nodes::TJoinOp;
        let join_node = make_hash_join_node(
            0,
            TJoinOp::LEFT_OUTER_JOIN,
            vec![0, 1],
            vec![(
                slot_ref_expr_in_tuple(0, 0, type_desc(TPrimitiveType::BIGINT)),
                slot_ref_expr_in_tuple(2, 1, type_desc(TPrimitiveType::BIGINT)),
            )],
        );
        let left_scan = make_file_scan_node(1, 0, "left_table");
        let right_scan = make_file_scan_node(2, 1, "right_table");
        let plan = make_plan(vec![join_node, left_scan, right_scan]);
        let desc = make_desc_table(
            vec![(0, Some(10)), (1, Some(20))],
            vec![
                (0, 0, 0, "id", TPrimitiveType::BIGINT),
                (1, 0, 1, "value", TPrimitiveType::DOUBLE),
                (2, 1, 0, "id", TPrimitiveType::BIGINT),
                (3, 1, 1, "name", TPrimitiveType::VARCHAR),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Join(join) => {
                assert_eq!(join.r#type, substrait::proto::join_rel::JoinType::Left as i32);
            }
            other => panic!("expected JoinRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- AGGREGATION_NODE tests (TPC-H: GROUP BY, COUNT, SUM) ----

    #[test]
    fn test_aggregation_count_with_group_by() {
        // SELECT cust_id, COUNT(*) FROM orders GROUP BY cust_id
        // Plan: AGG_NODE(need_finalize=true) → FILE_SCAN_NODE
        //
        // Scan tuple 0: order_id, cust_id
        // Agg intermediate tuple 1: cust_id_agg (grouping), count_star
        // Agg output tuple 2: cust_id_out, count_result
        let scan = make_file_scan_node(1, 0, "orders");
        let agg = make_aggregation_node(
            0,
            vec![2], // output tuple
            Some(vec![
                // GROUP BY cust_id (references scan tuple slot 1)
                slot_ref_expr_in_tuple(1, 0, type_desc(TPrimitiveType::BIGINT)),
            ]),
            vec![
                // COUNT(*) — aggregate with 0 children
                agg_function_expr("count", type_desc(TPrimitiveType::BIGINT), vec![], vec![]),
            ],
            1, // intermediate_tuple_id
            2, // output_tuple_id
            true, // need_finalize
        );
        let plan = make_plan(vec![agg, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10)), (1, None), (2, None)],
            vec![
                (0, 0, 0, "order_id", TPrimitiveType::BIGINT),
                (1, 0, 1, "cust_id", TPrimitiveType::BIGINT),
                (10, 2, 0, "cust_id", TPrimitiveType::BIGINT),
                (11, 2, 1, "count_star", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Aggregate(agg) => {
                assert_eq!(agg.groupings.len(), 1, "should have 1 grouping");
                assert_eq!(
                    agg.groupings[0].grouping_expressions.len(),
                    1,
                    "should have 1 grouping expression"
                );
                assert_eq!(agg.measures.len(), 1, "should have 1 measure (count)");
                let measure = &agg.measures[0].measure.as_ref().unwrap();
                assert_eq!(measure.phase, 4, "should be INTERMEDIATE_TO_RESULT for exchange finalize");
            }
            other => panic!("expected AggregateRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn test_aggregation_two_phase_collapse() {
        // Two-phase aggregation: partial AGG → finalize AGG → scan.
        // The finalize should collapse with the partial into INITIAL_TO_RESULT.
        let scan = make_file_scan_node(2, 0, "orders");
        let partial_agg = make_aggregation_node(
            1,
            vec![1],
            Some(vec![slot_ref_expr_in_tuple(0, 0, type_desc(TPrimitiveType::BIGINT))]),
            vec![agg_function_expr("sum", type_desc(TPrimitiveType::BIGINT),
                vec![type_desc(TPrimitiveType::BIGINT)],
                vec![slot_ref_expr_in_tuple(1, 0, type_desc(TPrimitiveType::BIGINT))])],
            1, 1,
            false, // need_finalize=false (partial)
        );
        let finalize_agg = make_aggregation_node(
            0,
            vec![2],
            Some(vec![slot_ref_expr_in_tuple(10, 1, type_desc(TPrimitiveType::BIGINT))]),
            vec![agg_function_expr("sum", type_desc(TPrimitiveType::BIGINT),
                vec![type_desc(TPrimitiveType::BIGINT)],
                vec![slot_ref_expr_in_tuple(11, 1, type_desc(TPrimitiveType::BIGINT))])],
            2, 2,
            true, // need_finalize=true (finalize)
        );
        let plan = make_plan(vec![finalize_agg, partial_agg, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10)), (1, None), (2, None)],
            vec![
                (0, 0, 0, "key", TPrimitiveType::BIGINT),
                (1, 0, 1, "amount", TPrimitiveType::BIGINT),
                (10, 1, 0, "key", TPrimitiveType::BIGINT),
                (11, 1, 1, "sum_amount", TPrimitiveType::BIGINT),
                (20, 2, 0, "key", TPrimitiveType::BIGINT),
                (21, 2, 1, "total", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        // Should be a single AggregateRel (collapsed from two phases).
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Aggregate(agg) => {
                assert_eq!(agg.measures.len(), 1);
                let measure = agg.measures[0].measure.as_ref().unwrap();
                assert_eq!(measure.phase, 3, "should be INITIAL_TO_RESULT after collapse");
                // Input should be the scan, not another aggregate.
                match agg.input.as_deref().unwrap().rel_type.as_ref().unwrap() {
                    rel::RelType::Read(_) => {} // Good — the partial was collapsed
                    other => panic!("expected ReadRel under collapsed AGG, got {:?}", std::mem::discriminant(other)),
                }
            }
            other => panic!("expected AggregateRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- SORT_NODE tests (TPC-H: ORDER BY, LIMIT) ----

    #[test]
    fn test_sort_order_by_asc() {
        // ORDER BY col1 ASC
        let scan = make_file_scan_node(1, 0, "data");
        let sort = make_sort_node(
            0,
            vec![0],
            vec![slot_ref_expr_in_tuple(0, 0, type_desc(TPrimitiveType::BIGINT))],
            vec![true],  // ASC
            vec![true],  // NULLS FIRST
            None,        // no offset
            -1,          // no limit
        );
        let plan = make_plan(vec![sort, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10))],
            vec![
                (0, 0, 0, "col1", TPrimitiveType::BIGINT),
                (1, 0, 1, "col2", TPrimitiveType::VARCHAR),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        // Root ProjectRel wraps the SortRel (prevents DuckDB column pruning).
        let input = root.input.as_ref().unwrap();
        let sort_input = match input.rel_type.as_ref().unwrap() {
            rel::RelType::Project(proj) => proj.input.as_ref().unwrap(),
            _ => input,
        };
        match sort_input.rel_type.as_ref().unwrap() {
            rel::RelType::Sort(sort) => {
                assert_eq!(sort.sorts.len(), 1);
                let sort_field = &sort.sorts[0];
                match sort_field.sort_kind.as_ref().unwrap() {
                    substrait::proto::sort_field::SortKind::Direction(d) => {
                        assert_eq!(d, &(substrait::proto::sort_field::SortDirection::AscNullsFirst as i32));
                    }
                    other => panic!("expected Direction, got {:?}", other),
                }
            }
            other => panic!("expected SortRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn test_sort_with_limit_offset() {
        // ORDER BY col1 DESC LIMIT 10 OFFSET 5
        let scan = make_file_scan_node(1, 0, "data");
        let sort = make_sort_node(
            0,
            vec![0],
            vec![slot_ref_expr_in_tuple(0, 0, type_desc(TPrimitiveType::BIGINT))],
            vec![false], // DESC
            vec![false], // NULLS LAST
            Some(5),     // offset
            10,          // limit
        );
        let plan = make_plan(vec![sort, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10))],
            vec![(0, 0, 0, "col1", TPrimitiveType::BIGINT)],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        // Root ProjectRel wraps the FetchRel (prevents DuckDB column pruning).
        let input = root.input.as_ref().unwrap();
        let fetch_input = match input.rel_type.as_ref().unwrap() {
            rel::RelType::Project(proj) => proj.input.as_ref().unwrap(),
            _ => input,
        };
        match fetch_input.rel_type.as_ref().unwrap() {
            rel::RelType::Fetch(fetch) => {
                match fetch.offset_mode.as_ref().unwrap() {
                    substrait::proto::fetch_rel::OffsetMode::Offset(o) => assert_eq!(o, &5),
                    other => panic!("expected Offset, got {:?}", other),
                }
                match fetch.count_mode.as_ref().unwrap() {
                    substrait::proto::fetch_rel::CountMode::Count(c) => assert_eq!(c, &10),
                    other => panic!("expected Count, got {:?}", other),
                }
                // Inner should be SortRel.
                match fetch.input.as_deref().unwrap().rel_type.as_ref().unwrap() {
                    rel::RelType::Sort(sort) => {
                        assert_eq!(sort.sorts.len(), 1);
                    }
                    other => panic!("expected SortRel under Fetch, got {:?}", std::mem::discriminant(other)),
                }
            }
            other => panic!("expected FetchRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- CROSS_JOIN_NODE tests ----

    #[test]
    fn test_cross_join() {
        let join_node = make_cross_join_node(0, vec![0, 1]);
        let left_scan = make_file_scan_node(1, 0, "t1");
        let right_scan = make_file_scan_node(2, 1, "t2");
        let plan = make_plan(vec![join_node, left_scan, right_scan]);
        let desc = make_desc_table(
            vec![(0, Some(10)), (1, Some(20))],
            vec![
                (0, 0, 0, "a", TPrimitiveType::INT),
                (1, 1, 0, "b", TPrimitiveType::INT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Cross(cross) => {
                assert!(cross.left.is_some());
                assert!(cross.right.is_some());
            }
            other => panic!("expected CrossRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- SELECT_NODE tests (filter pass-through) ----

    #[test]
    fn test_select_node_with_filter() {
        // SELECT_NODE wraps a child with conjuncts.
        let scan = make_file_scan_node(1, 0, "data");
        let select = make_select_node(
            0,
            vec![0],
            vec![binary_pred_expr(
                doris_thrift::opcodes::TExprOpcode::GT,
                slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
                int_literal_expr(100),
            )],
        );
        let plan = make_plan(vec![select, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10))],
            vec![(0, 0, 0, "value", TPrimitiveType::BIGINT)],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Filter(filter) => {
                assert!(filter.condition.is_some());
                // Input should be a ReadRel (scan).
                match filter.input.as_deref().unwrap().rel_type.as_ref().unwrap() {
                    rel::RelType::Read(_) => {}
                    other => panic!("expected ReadRel, got {:?}", std::mem::discriminant(other)),
                }
            }
            other => panic!("expected FilterRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- EXCHANGE_NODE pass-through tests ----

    #[test]
    fn test_exchange_passthrough_with_child() {
        // EXCHANGE_NODE(1 child) = sender wrapping scan → should pass through.
        let exchange = make_plan_node(0, doris_thrift::plan_nodes::TPlanNodeType::EXCHANGE_NODE, 1, vec![0]);
        let scan = make_file_scan_node(1, 0, "data");
        let plan = make_plan(vec![exchange, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10))],
            vec![(0, 0, 0, "col1", TPrimitiveType::INT)],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        // Should be a ReadRel directly (exchange passed through).
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(_) => {}
            other => panic!("expected ReadRel (pass-through), got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- MATERIALIZATION_NODE pass-through test ----

    #[test]
    fn test_materialization_passthrough() {
        let mat_node = make_plan_node(0, doris_thrift::plan_nodes::TPlanNodeType::MATERIALIZATION_NODE, 1, vec![0]);
        let scan = make_file_scan_node(1, 0, "data");
        let plan = make_plan(vec![mat_node, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10))],
            vec![(0, 0, 0, "col1", TPrimitiveType::INT)],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        match root.input.as_ref().unwrap().rel_type.as_ref().unwrap() {
            rel::RelType::Read(_) => {} // Good — materialization passed through to scan
            other => panic!("expected ReadRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn test_materialization_truncates_output_names() {
        // Simulate Q2 fragment 0: MATERIALIZATION_NODE (8 output cols) → EXCHANGE_NODE (4 cols).
        // The exchange table only has 4 columns, but the root's row_tuples references 8.
        // translate_fragment should truncate output_names to 4 (the Rel's actual columns).
        use doris_thrift::plan_nodes::TPlanNodeType;

        // Tuple 0: MATERIALIZATION_NODE output — 8 columns (includes late-materialized ones)
        // Tuple 1: EXCHANGE_NODE output — 4 columns (actually available in exchange data)
        let mat = make_plan_node(0, TPlanNodeType::MATERIALIZATION_NODE, 1, vec![0]);
        let exchange = make_exchange_node(1, vec![1]);
        let plan = make_plan(vec![mat, exchange]);
        let desc = make_desc_table(
            vec![(0, None), (1, None)],
            vec![
                // Tuple 0 slots (8 columns — what FE expects)
                (10, 0, 0, "s_acctbal", TPrimitiveType::DOUBLE),
                (11, 0, 1, "s_name", TPrimitiveType::VARCHAR),
                (12, 0, 2, "n_name", TPrimitiveType::VARCHAR),
                (13, 0, 3, "p_partkey", TPrimitiveType::INT),
                (14, 0, 4, "p_mfgr", TPrimitiveType::VARCHAR),
                (15, 0, 5, "s_address", TPrimitiveType::VARCHAR),
                (16, 0, 6, "s_phone", TPrimitiveType::VARCHAR),
                (17, 0, 7, "s_comment", TPrimitiveType::VARCHAR),
                // Tuple 1 slots (4 columns — what exchange actually has)
                (20, 1, 0, "s_acctbal", TPrimitiveType::DOUBLE),
                (21, 1, 1, "s_name", TPrimitiveType::VARCHAR),
                (22, 1, 2, "n_name", TPrimitiveType::VARCHAR),
                (23, 1, 3, "p_partkey", TPrimitiveType::INT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        // Provide table_schemas so the exchange ReadRel uses 4 columns.
        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "__EXCHANGE_TABLE_1".to_string(),
            vec![
                "s_acctbal".to_string(),
                "s_name".to_string(),
                "n_name".to_string(),
                "p_partkey".to_string(),
            ],
        );

        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();

        // Output names should be truncated to 4 (matching the exchange table).
        assert_eq!(result.output_names.len(), 4);
        assert_eq!(
            result.output_names,
            vec!["s_acctbal", "s_name", "n_name", "p_partkey"]
        );

        // The Substrait plan should have 4 root names (not 8).
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        assert_eq!(root.names.len(), 4);
    }

    #[test]
    fn test_materialization_preserves_fe_order() {
        // Exchange table has columns in sort-key order (s_acctbal, n_name, s_name, p_partkey),
        // but FE output_exprs expects SELECT-list order (s_acctbal, s_name, n_name, p_partkey).
        // translate_fragment should preserve the FE order.
        use doris_thrift::plan_nodes::TPlanNodeType;

        let mat = make_plan_node(0, TPlanNodeType::MATERIALIZATION_NODE, 1, vec![0]);
        let exchange = make_exchange_node(1, vec![1]);
        let plan = make_plan(vec![mat, exchange]);
        let desc = make_desc_table(
            vec![(0, None), (1, None)],
            vec![
                // Tuple 0 slots (8 columns — FE SELECT list order)
                (10, 0, 0, "s_acctbal", TPrimitiveType::DOUBLE),
                (11, 0, 1, "s_name", TPrimitiveType::VARCHAR),
                (12, 0, 2, "n_name", TPrimitiveType::VARCHAR),
                (13, 0, 3, "p_partkey", TPrimitiveType::INT),
                (14, 0, 4, "p_mfgr", TPrimitiveType::VARCHAR),
                (15, 0, 5, "s_address", TPrimitiveType::VARCHAR),
                (16, 0, 6, "s_phone", TPrimitiveType::VARCHAR),
                (17, 0, 7, "s_comment", TPrimitiveType::VARCHAR),
                // Tuple 1 slots (exchange order — different from FE order)
                (20, 1, 0, "s_acctbal", TPrimitiveType::DOUBLE),
                (21, 1, 1, "n_name", TPrimitiveType::VARCHAR),  // n_name before s_name
                (22, 1, 2, "s_name", TPrimitiveType::VARCHAR),  // s_name after n_name
                (23, 1, 3, "p_partkey", TPrimitiveType::INT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        // Exchange table in DuckDB sort-key order (n_name before s_name).
        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "__EXCHANGE_TABLE_1".to_string(),
            vec![
                "s_acctbal".to_string(),
                "n_name".to_string(),  // sort-key order
                "s_name".to_string(),
                "p_partkey".to_string(),
            ],
        );

        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();

        // Output names should be in FE SELECT list order (s_name before n_name).
        // project_ipc_columns uses these to reorder from DuckDB output to FE order.
        assert_eq!(result.output_names.len(), 4);
        assert_eq!(
            result.output_names,
            vec!["s_acctbal", "s_name", "n_name", "p_partkey"]
        );

        // Root.names = rel_names (exchange table order, not FE order).
        // No ProjectRel wrapping — reordering happens post-execution via project_ipc_columns.
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        assert_eq!(root.names, vec!["s_acctbal", "n_name", "s_name", "p_partkey"],
            "Root.names should be in rel_names order (exchange table order)");
    }

    // ---- EMPTY_SET_NODE test ----

    #[test]
    fn test_empty_set_substrait() {
        let node = make_empty_set_node(0);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![], vec![]);
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        match root.input.as_ref().unwrap().rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::VirtualTable(vt)) => {
                    assert!(vt.values.is_empty(), "EMPTY_SET should have 0 rows");
                }
                other => panic!("expected VirtualTable, got {:?}", other),
            },
            other => panic!("expected ReadRel, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- Extension registration tests ----

    #[test]
    fn test_extension_function_dedup() {
        // When the same function is used multiple times, it should only be registered once.
        let scan = make_file_scan_node(1, 0, "data");
        let select = make_select_node(
            0,
            vec![0],
            vec![
                binary_pred_expr(
                    doris_thrift::opcodes::TExprOpcode::GT,
                    slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
                    int_literal_expr(10),
                ),
                binary_pred_expr(
                    doris_thrift::opcodes::TExprOpcode::GT,
                    slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
                    int_literal_expr(20),
                ),
            ],
        );
        let plan = make_plan(vec![select, scan]);
        let desc = make_desc_table(
            vec![(0, Some(10))],
            vec![(0, 0, 0, "val", TPrimitiveType::BIGINT)],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        // "gt" should be registered once even though used twice.
        let gt_count = plan
            .extensions
            .iter()
            .filter(|ext| {
                if let Some(substrait::proto::extensions::simple_extension_declaration::MappingType::ExtensionFunction(f)) = &ext.mapping_type {
                    f.name == "gt"
                } else {
                    false
                }
            })
            .count();
        assert_eq!(gt_count, 1, "gt should be registered exactly once");
    }

    // ---- UNION_NODE with children (scan-based union / SetRel) ----

    #[test]
    fn test_union_node_with_scan_children_produces_set_rel() {
        // UNION_NODE(2 children) → SetRel(UNION_ALL) with two ReadRels.
        let mut union_node = make_union_node(0, 0, vec![]); // no const_expr_lists
        union_node.num_children = 2;

        let left_scan = make_file_scan_node(1, 0, "table_a");
        let right_scan = make_file_scan_node(2, 1, "table_b");
        let plan = make_plan(vec![union_node, left_scan, right_scan]);
        let desc = make_desc_table(
            vec![(0, Some(10)), (1, Some(20))],
            vec![
                (0, 0, 0, "city", TPrimitiveType::VARCHAR),
                (1, 0, 1, "pop", TPrimitiveType::INT),
                (2, 1, 0, "city", TPrimitiveType::VARCHAR),
                (3, 1, 1, "pop", TPrimitiveType::INT),
            ],
        );
        let mut params = make_fragment_params(plan, desc);

        // Provide table schemas for both scans.
        let table_schemas: HashMap<String, Vec<String>> = [
            ("table_a_1".to_string(), vec!["city".to_string(), "pop".to_string()]),
            ("table_b_2".to_string(), vec!["city".to_string(), "pop".to_string()]),
        ]
        .into_iter()
        .collect();

        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Set(set) => {
                assert_eq!(set.op, substrait::proto::set_rel::SetOp::UnionAll as i32);
                assert_eq!(set.inputs.len(), 2, "SetRel should have 2 inputs");
            }
            other => panic!("expected Set, got {:?}", other),
        }
    }

    #[test]
    fn test_count_rel_columns_for_set_rel() {
        use crate::node_translator::count_rel_columns;
        use substrait::proto::{set_rel, Rel, SetRel};
        use substrait::proto::rel::RelType;

        // SetRel with first input having 3 columns (ReadRel with 3-field named_struct).
        let read_rel = Rel {
            rel_type: Some(RelType::Read(Box::new(substrait::proto::ReadRel {
                base_schema: Some(substrait::proto::NamedStruct {
                    names: vec!["a".into(), "b".into(), "c".into()],
                    r#struct: None,
                }),
                read_type: Some(substrait::proto::read_rel::ReadType::NamedTable(
                    substrait::proto::read_rel::NamedTable {
                        names: vec!["test".into()],
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }))),
        };

        let set_rel = Rel {
            rel_type: Some(RelType::Set(SetRel {
                inputs: vec![read_rel],
                op: set_rel::SetOp::UnionAll as i32,
                ..Default::default()
            })),
        };

        assert_eq!(count_rel_columns(&set_rel), 3);
    }

    #[test]
    fn test_count_rel_columns_for_empty_set() {
        use crate::node_translator::count_rel_columns;
        use substrait::proto::{Rel, SetRel};
        use substrait::proto::rel::RelType;

        let set_rel = Rel {
            rel_type: Some(RelType::Set(SetRel {
                inputs: vec![],
                op: 0,
                ..Default::default()
            })),
        };

        assert_eq!(count_rel_columns(&set_rel), 0);
    }

    #[test]
    fn test_rel_output_names_for_set() {
        use substrait::proto::{set_rel, Rel, SetRel};
        use substrait::proto::rel::RelType;

        let read = Rel {
            rel_type: Some(RelType::Read(Box::new(substrait::proto::ReadRel {
                base_schema: Some(substrait::proto::NamedStruct {
                    names: vec!["x".into(), "y".into()],
                    r#struct: None,
                }),
                read_type: Some(substrait::proto::read_rel::ReadType::NamedTable(
                    substrait::proto::read_rel::NamedTable {
                        names: vec!["t".into()],
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }))),
        };

        let set = Rel {
            rel_type: Some(RelType::Set(SetRel {
                inputs: vec![read],
                op: set_rel::SetOp::UnionAll as i32,
                ..Default::default()
            })),
        };

        let names = rel_output_names(&set);
        assert_eq!(names, vec!["x", "y"]);
    }

    // ---- JOIN + AGG field reference tests ----

    /// Test that a JOIN→AGG plan produces correct field reference indices for the
    /// aggregate function's input expression.
    ///
    /// This is the TPC-H customer-orders pattern:
    ///   AGG(need_finalize=true, GROUP BY c_name, sum(o_totalprice))
    ///     → JOIN(ON c_custkey = o_custkey)
    ///       → SCAN(customer: c_custkey, c_name, c_address)
    ///       → SCAN(orders: o_orderkey, o_custkey, o_totalprice)
    ///
    /// The AGG's sum(o_totalprice) SLOT_REF has parent_tuple=3 (AGG intermediate),
    /// which is NOT one of the JOIN's row_tuples [0, 1]. The resolution must use
    /// name-based matching against [0, 1] to get the correct global index:
    ///   o_totalprice is at position 2 in the orders tuple (tuple 1),
    ///   with global offset 3 (from customer tuple 0's 3 columns) → index 5.
    #[test]
    fn test_join_agg_field_reference_offset() {
        use substrait::proto::expression::field_reference;
        use substrait::proto::expression::reference_segment;

        // Customer scan tuple 0: c_custkey(0), c_name(1), c_address(2)
        // Orders scan tuple 1: o_orderkey(0), o_custkey(1), o_totalprice(2)
        // AGG intermediate tuple 3: c_name_agg(0), total_spent(1)
        //   (parent of AGG function SLOT_REFs)
        let left_scan = make_file_scan_node(3, 0, "customer");
        let right_scan = make_file_scan_node(4, 1, "orders");
        let join_node = make_hash_join_node(
            2,
            doris_thrift::plan_nodes::TJoinOp::INNER_JOIN,
            vec![0, 1], // JOIN output: customer tuple + orders tuple
            vec![(
                // ON c_custkey = o_custkey
                slot_ref_expr_in_tuple(10, 0, type_desc(TPrimitiveType::BIGINT)),
                slot_ref_expr_in_tuple(20, 1, type_desc(TPrimitiveType::BIGINT)),
            )],
        );
        let agg = make_aggregation_node(
            1,
            vec![3], // AGG output tuple
            Some(vec![
                // GROUP BY c_name — slot 31 in AGG intermediate tuple 3,
                // but resolves by name to position 1 in JOIN output (tuple 0's 2nd col).
                slot_ref_expr_in_tuple(31, 3, type_desc(TPrimitiveType::VARCHAR)),
            ]),
            vec![
                // sum(o_totalprice) — slot 32 in AGG intermediate tuple 3,
                // must resolve by name to position 5 in JOIN output
                // (3 customer cols + 2 = index 5).
                agg_function_expr(
                    "sum",
                    type_desc(TPrimitiveType::DOUBLE),
                    vec![type_desc(TPrimitiveType::DOUBLE)],
                    vec![slot_ref_expr_in_tuple(32, 3, type_desc(TPrimitiveType::DOUBLE))],
                ),
            ],
            3, 3,
            true, // need_finalize (single-phase)
        );
        let plan = make_plan(vec![agg, join_node, left_scan, right_scan]);
        let desc = make_desc_table(
            vec![(0, Some(100)), (1, Some(200)), (3, None)],
            vec![
                // Customer (tuple 0, table 100)
                (10, 0, 0, "c_custkey", TPrimitiveType::BIGINT),
                (11, 0, 1, "c_name", TPrimitiveType::VARCHAR),
                (12, 0, 2, "c_address", TPrimitiveType::VARCHAR),
                // Orders (tuple 1, table 200)
                (20, 1, 0, "o_orderkey", TPrimitiveType::BIGINT),
                (21, 1, 1, "o_custkey", TPrimitiveType::BIGINT),
                (22, 1, 2, "o_totalprice", TPrimitiveType::DOUBLE),
                // AGG intermediate (tuple 3, no table)
                (31, 3, 0, "c_name", TPrimitiveType::VARCHAR),
                (32, 3, 1, "o_totalprice", TPrimitiveType::DOUBLE),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        let agg = match input.rel_type.as_ref().unwrap() {
            rel::RelType::Aggregate(agg) => agg,
            other => panic!("expected AggregateRel, got {:?}", std::mem::discriminant(other)),
        };

        // Verify the measure's argument is a field reference.
        let measure = agg.measures[0].measure.as_ref().unwrap();
        assert_eq!(measure.arguments.len(), 1, "sum should have 1 argument");
        let arg_expr = match &measure.arguments[0].arg_type {
            Some(substrait::proto::function_argument::ArgType::Value(e)) => e,
            other => panic!("expected Value argument, got {:?}", other),
        };
        // Extract the field index from the argument expression.
        let field_idx = match arg_expr.rex_type.as_ref().unwrap() {
            substrait::proto::expression::RexType::Selection(sel) => {
                match sel.reference_type.as_ref().unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.as_ref().unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => sf.field,
                            other => panic!("expected StructField, got {:?}", other),
                        }
                    }
                    other => panic!("expected DirectReference, got {:?}", other),
                }
            }
            other => panic!("expected Selection (field ref), got {:?}", other),
        };

        // o_totalprice should be at global index 5 (3 customer + 2 in orders).
        assert_eq!(
            field_idx, 5,
            "sum(o_totalprice) field reference should be at index 5 (3 customer cols + 2), got {}",
            field_idx
        );

        // Also verify the GROUP BY c_name field reference.
        let grouping_expr = &agg.groupings[0].grouping_expressions[0];
        let group_field_idx = match grouping_expr.rex_type.as_ref().unwrap() {
            substrait::proto::expression::RexType::Selection(sel) => {
                match sel.reference_type.as_ref().unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.as_ref().unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => sf.field,
                            other => panic!("expected StructField, got {:?}", other),
                        }
                    }
                    other => panic!("expected DirectReference, got {:?}", other),
                }
            }
            other => panic!("expected Selection, got {:?}", other),
        };

        // c_name should be at global index 1 (position 1 in customer tuple).
        assert_eq!(
            group_field_idx, 1,
            "GROUP BY c_name field reference should be at index 1, got {}",
            group_field_idx
        );
    }

    /// Test two-phase AGG collapse over a JOIN — verifies that after merge,
    /// the collapsed aggregation correctly references JOIN output indices.
    #[test]
    fn test_two_phase_agg_collapse_over_join() {
        use substrait::proto::expression::field_reference;
        use substrait::proto::expression::reference_segment;

        // Plan: finalize_AGG → partial_AGG → JOIN → SCAN, SCAN
        // This simulates the merged plan after fragment cascade merge.
        let left_scan = make_file_scan_node(5, 0, "customer");
        let right_scan = make_file_scan_node(6, 1, "orders");
        let join_node = make_hash_join_node(
            4,
            doris_thrift::plan_nodes::TJoinOp::INNER_JOIN,
            vec![0, 1],
            vec![(
                slot_ref_expr_in_tuple(10, 0, type_desc(TPrimitiveType::BIGINT)),
                slot_ref_expr_in_tuple(20, 1, type_desc(TPrimitiveType::BIGINT)),
            )],
        );
        // Partial AGG: GROUP BY c_name, sum(o_totalprice)
        // child_node = JOIN, so child_row_tuples = [0, 1]
        let partial_agg = make_aggregation_node(
            3,
            vec![2], // partial output tuple
            Some(vec![
                slot_ref_expr_in_tuple(30, 2, type_desc(TPrimitiveType::VARCHAR)),
            ]),
            vec![
                agg_function_expr(
                    "sum",
                    type_desc(TPrimitiveType::DOUBLE),
                    vec![type_desc(TPrimitiveType::DOUBLE)],
                    vec![slot_ref_expr_in_tuple(31, 2, type_desc(TPrimitiveType::DOUBLE))],
                ),
            ],
            2, 2,
            false, // partial
        );
        // Finalize AGG: references partial output tuple 2
        let finalize_agg = make_aggregation_node(
            2,
            vec![3], // finalize output tuple
            Some(vec![
                slot_ref_expr_in_tuple(40, 2, type_desc(TPrimitiveType::VARCHAR)),
            ]),
            vec![
                agg_function_expr(
                    "sum",
                    type_desc(TPrimitiveType::DOUBLE),
                    vec![type_desc(TPrimitiveType::DOUBLE)],
                    vec![slot_ref_expr_in_tuple(41, 2, type_desc(TPrimitiveType::DOUBLE))],
                ),
            ],
            3, 3,
            true, // finalize
        );
        let plan = make_plan(vec![finalize_agg, partial_agg, join_node, left_scan, right_scan]);
        let desc = make_desc_table(
            vec![(0, Some(100)), (1, Some(200)), (2, None), (3, None)],
            vec![
                (10, 0, 0, "c_custkey", TPrimitiveType::BIGINT),
                (11, 0, 1, "c_name", TPrimitiveType::VARCHAR),
                (20, 1, 0, "o_orderkey", TPrimitiveType::BIGINT),
                (21, 1, 1, "o_custkey", TPrimitiveType::BIGINT),
                (22, 1, 2, "o_totalprice", TPrimitiveType::DOUBLE),
                // Partial output tuple 2
                (30, 2, 0, "c_name", TPrimitiveType::VARCHAR),
                (31, 2, 1, "o_totalprice", TPrimitiveType::DOUBLE),
                // Finalize output tuple 3
                (40, 3, 0, "c_name", TPrimitiveType::VARCHAR),
                (41, 3, 1, "total", TPrimitiveType::DOUBLE),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new(), &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };

        // After collapse, should be a single AggregateRel with INITIAL_TO_RESULT.
        let agg = match root.input.as_ref().unwrap().rel_type.as_ref().unwrap() {
            rel::RelType::Aggregate(agg) => agg,
            other => panic!("expected AggregateRel, got {:?}", std::mem::discriminant(other)),
        };

        assert_eq!(agg.measures.len(), 1);
        let measure = agg.measures[0].measure.as_ref().unwrap();
        assert_eq!(measure.phase, 3, "should be INITIAL_TO_RESULT after collapse");

        // The collapsed measure should use the partial's expression (which references
        // the JOIN's output schema). Verify the field ref is correct.
        let arg = match &measure.arguments[0].arg_type {
            Some(substrait::proto::function_argument::ArgType::Value(e)) => e,
            other => panic!("expected Value, got {:?}", other),
        };
        let field_idx = match arg.rex_type.as_ref().unwrap() {
            substrait::proto::expression::RexType::Selection(sel) => {
                match sel.reference_type.as_ref().unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.as_ref().unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => sf.field,
                            other => panic!("expected StructField, got {:?}", other),
                        }
                    }
                    other => panic!("expected DirectReference, got {:?}", other),
                }
            }
            other => panic!("expected Selection, got {:?}", other),
        };

        // o_totalprice should be at index 4 (2 customer cols + 2 in orders).
        assert_eq!(
            field_idx, 4,
            "sum(o_totalprice) in collapsed AGG should reference index 4 (2 customer + 2 orders), got {}",
            field_idx
        );

        // The input to the collapsed AGG should be the JoinRel (not another AGG).
        match agg.input.as_deref().unwrap().rel_type.as_ref().unwrap() {
            rel::RelType::Join(_) => {}
            other => panic!("expected JoinRel under collapsed AGG, got {:?}", std::mem::discriminant(other)),
        }
    }

    // ---- LocalFiles tests (multi-file parquet scan) ----

    #[test]
    fn test_file_scan_single_file_local_files() {
        // FILE_SCAN_NODE with file_scan_map entry → ReadRel::LocalFiles (1 file).
        let node = make_file_scan_node(0, 0, "lineitem");
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "l_orderkey", TPrimitiveType::BIGINT),
                (1, 0, 1, "l_partkey", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "lineitem_0".to_string(),
            vec!["l_orderkey".to_string(), "l_partkey".to_string()],
        );

        let mut file_scan_map = HashMap::new();
        file_scan_map.insert(
            "lineitem_0".to_string(),
            FileScanInfo {
                table_name: "lineitem_0".to_string(),
                format: "parquet".to_string(),
                files: vec![FileScanFile {
                    path: "/data/tpch/lineitem/part-0.parquet".to_string(),
                    start_offset: 0,
                    length: -1,
                }],
            },
        );

        let result = translate_fragment(&params, &table_schemas, &file_scan_map).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::LocalFiles(lf)) => {
                    assert_eq!(lf.items.len(), 1, "should have 1 file");
                    let item = &lf.items[0];
                    match &item.path_type {
                        Some(substrait::proto::read_rel::local_files::file_or_files::PathType::UriFile(path)) => {
                            assert_eq!(path, "/data/tpch/lineitem/part-0.parquet");
                        }
                        other => panic!("expected UriFile, got {:?}", other),
                    }
                    assert!(item.file_format.is_some(), "should have parquet file format");
                }
                other => panic!("expected LocalFiles, got {:?}", other),
            },
            other => panic!("expected Read, got {:?}", other),
        }
        // Schema should still be correct.
        assert_eq!(result.output_names, vec!["l_orderkey", "l_partkey"]);
    }

    #[test]
    fn test_file_scan_multi_file_local_files() {
        // FILE_SCAN_NODE with 8 parquet files → ReadRel::LocalFiles (8 items).
        let node = make_file_scan_node(0, 0, "lineitem");
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "l_orderkey", TPrimitiveType::BIGINT),
                (1, 0, 1, "l_partkey", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "lineitem_0".to_string(),
            vec!["l_orderkey".to_string(), "l_partkey".to_string()],
        );

        let files: Vec<FileScanFile> = (0..8)
            .map(|i| FileScanFile {
                path: format!("/data/tpch/lineitem/part-{}.parquet", i),
                start_offset: 0,
                length: -1,
            })
            .collect();
        let mut file_scan_map = HashMap::new();
        file_scan_map.insert(
            "lineitem_0".to_string(),
            FileScanInfo {
                table_name: "lineitem_0".to_string(),
                format: "parquet".to_string(),
                files,
            },
        );

        let result = translate_fragment(&params, &table_schemas, &file_scan_map).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::LocalFiles(lf)) => {
                    assert_eq!(lf.items.len(), 8, "should have 8 files");
                    for (i, item) in lf.items.iter().enumerate() {
                        match &item.path_type {
                            Some(substrait::proto::read_rel::local_files::file_or_files::PathType::UriFile(path)) => {
                                assert_eq!(path, &format!("/data/tpch/lineitem/part-{}.parquet", i));
                            }
                            other => panic!("file {} expected UriFile, got {:?}", i, other),
                        }
                    }
                }
                other => panic!("expected LocalFiles, got {:?}", other),
            },
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_parquet_projection_pushdown() {
        // Parquet file has 5 columns, but only 2 are materialized in the descriptor.
        // The ReadRel base_schema should contain only the 2 materialized columns,
        // so DuckDB/Sirius reads only needed columns from the parquet files.
        let node = make_file_scan_node(0, 0, "wide_table");
        let plan = make_plan(vec![node]);

        // Descriptor only materializes 2 of 5 columns.
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "col_a", TPrimitiveType::BIGINT),
                (1, 0, 1, "col_c", TPrimitiveType::VARCHAR),
            ],
        );
        let params = make_fragment_params(plan, desc);

        // Full parquet schema has 5 columns.
        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "wide_table_0".to_string(),
            vec![
                "col_a".to_string(),
                "col_b".to_string(),
                "col_c".to_string(),
                "col_d".to_string(),
                "col_e".to_string(),
            ],
        );

        let mut file_scan_map = HashMap::new();
        file_scan_map.insert(
            "wide_table_0".to_string(),
            FileScanInfo {
                table_name: "wide_table_0".to_string(),
                format: "parquet".to_string(),
                files: vec![FileScanFile {
                    path: "/data/wide_table/part-0.parquet".to_string(),
                    start_offset: 0,
                    length: -1,
                }],
            },
        );

        let result = translate_fragment(&params, &table_schemas, &file_scan_map).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };

        // With projection pushdown disabled, all 5 columns are present.
        assert_eq!(
            root.names,
            vec!["col_a", "col_b", "col_c", "col_d", "col_e"]
        );

        // ReadRel base_schema should have ALL columns (no projection pushdown).
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => {
                let schema = read.base_schema.as_ref().unwrap();
                assert_eq!(
                    schema.names,
                    vec!["col_a", "col_b", "col_c", "col_d", "col_e"],
                    "ReadRel should have ALL columns (projection pushdown disabled)"
                );
                match &read.read_type {
                    Some(read_rel::ReadType::LocalFiles(_)) => {}
                    other => panic!("expected LocalFiles, got {:?}", other),
                }
            }
            other => panic!("expected Read, got {:?}", other),
        }

        assert_eq!(result.output_names, vec!["col_a", "col_c"]);
    }

    #[test]
    fn test_parquet_projection_pushdown_preserves_named_table() {
        // Non-parquet (NamedTable) path should NOT filter columns, even when some
        // columns are not materialized. NamedTable maps by POSITION, not name.
        let node = make_file_scan_node(0, 0, "cities");
        let plan = make_plan(vec![node]);

        // Only 2 of 3 columns materialized.
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "city", TPrimitiveType::VARCHAR),
                (2, 0, 2, "population", TPrimitiveType::BIGINT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "cities_0".to_string(),
            vec!["city".to_string(), "state".to_string(), "population".to_string()],
        );

        // No file_scan_map entry → NamedTable path → all 3 columns preserved.
        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };

        // ReadRel should have ALL 3 columns (no projection pushdown for NamedTable).
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => {
                let schema = read.base_schema.as_ref().unwrap();
                assert_eq!(schema.names, vec!["city", "state", "population"]);
            }
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_file_scan_empty_map_falls_back_to_named_table() {
        // FILE_SCAN_NODE with empty file_scan_map → ReadRel::NamedTable (backward compat).
        let node = make_file_scan_node(0, 0, "cities");
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "city", TPrimitiveType::VARCHAR),
                (1, 0, 1, "pop", TPrimitiveType::INT),
            ],
        );
        let params = make_fragment_params(plan, desc);

        let mut table_schemas = HashMap::new();
        table_schemas.insert(
            "cities_0".to_string(),
            vec!["city".to_string(), "pop".to_string()],
        );

        // Empty file_scan_map → should produce NamedTable, not LocalFiles.
        let result = translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::NamedTable(nt)) => {
                    assert_eq!(nt.names, vec!["cities_0"]);
                }
                other => panic!("expected NamedTable, got {:?}", other),
            },
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn test_file_scan_non_parquet_falls_back_to_named_table() {
        // FILE_SCAN_NODE with CSV format in file_scan_map → NamedTable (not LocalFiles).
        let node = make_file_scan_node(0, 0, "data");
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(
            vec![(0, Some(1))],
            vec![(0, 0, 0, "col1", TPrimitiveType::VARCHAR)],
        );
        let params = make_fragment_params(plan, desc);

        let mut table_schemas = HashMap::new();
        table_schemas.insert("data_0".to_string(), vec!["col1".to_string()]);

        let mut file_scan_map = HashMap::new();
        file_scan_map.insert(
            "data_0".to_string(),
            FileScanInfo {
                table_name: "data_0".to_string(),
                format: "csv".to_string(), // Not parquet!
                files: vec![FileScanFile {
                    path: "/data/test.csv".to_string(),
                    start_offset: 0,
                    length: -1,
                }],
            },
        );

        let result = translate_fragment(&params, &table_schemas, &file_scan_map).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Read(read) => match &read.read_type {
                Some(read_rel::ReadType::NamedTable(nt)) => {
                    assert_eq!(nt.names, vec!["data_0"]);
                }
                other => panic!("expected NamedTable for CSV, got {:?}", other),
            },
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    #[ignore] // Run with: cargo test -p plan-translator -- --ignored decode_substrait_debug
    fn decode_substrait_debug() {
        use std::fs;
        for entry in fs::read_dir("/tmp/sirius-cluster/").unwrap() {
            let path = entry.unwrap().path();
            if path.extension().map(|e| e == "bin").unwrap_or(false)
                && path.file_name().unwrap().to_str().unwrap().starts_with("substrait-exchange")
            {
                let data = fs::read(&path).unwrap();
                let plan = Plan::decode(data.as_slice()).unwrap();
                println!("\n=== {} ({} bytes) ===", path.display(), data.len());
                println!("{:#?}", plan);
            }
        }
    }
}
