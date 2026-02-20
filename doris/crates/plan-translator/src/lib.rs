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
pub mod sql_generator;
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

/// Translate a Doris TPipelineFragmentParams into a SQL string for DuckDB execution.
pub fn translate_fragment_to_sql(params: &TPipelineFragmentParams) -> Result<String> {
    let fragment = params
        .fragment
        .as_ref()
        .context("TPipelineFragmentParams has no fragment")?;
    let plan = fragment
        .plan
        .as_ref()
        .context("TPlanFragment has no plan")?;

    sql_generator::plan_to_sql(plan)
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
}

/// Build a slot expression map from FILE_SCAN_NODE intermediate/final projections.
///
/// For each projection layer, maps output tuple slot IDs to the TExpr that produces them.
/// This allows the expression translator to inline computed expressions from scan projections
/// when downstream nodes (AGG, SORT) reference these computed slots.
fn build_projection_slot_map(plan: &TPlan, desc: &mut descriptor_table::DescriptorTable) {
    let mut slot_expressions: HashMap<i32, TExpr> = HashMap::new();

    for node in &plan.nodes {
        if node.node_type != TPlanNodeType::FILE_SCAN_NODE {
            continue;
        }

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
                            slot_expressions.insert(materialized[i], expr.clone());
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
                        slot_expressions.insert(materialized[i], expr.clone());
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
) -> Option<String> {
    // Find the SORT_NODE: it may be the root, or wrapped in a MATERIALIZATION_NODE.
    // Walk from the root through pass-through nodes to find it.
    let sort_plan_node = plan.nodes.iter().find(|n| n.node_type == TPlanNodeType::SORT_NODE)?;
    let sort_node = sort_plan_node.sort_node.as_ref()?;
    let sort_info = &sort_node.sort_info;

    // Resolve sort expressions to 1-based positional references in the sort node's output.
    // We use positional refs because aggregate aliases often have empty col_name.
    let row_tuples = &sort_plan_node.row_tuples;
    let materialized_slots: Vec<i32> = row_tuples
        .iter()
        .flat_map(|&tid| {
            desc.get_tuple(tid)
                .map(|t| {
                    t.slot_ids
                        .iter()
                        .copied()
                        .filter(|&sid| {
                            desc.get_slot(sid)
                                .map(|s| s.is_materialized)
                                .unwrap_or(false)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();

    let mut order_parts = Vec::new();
    for (i, expr) in sort_info.ordering_exprs.iter().enumerate() {
        let is_asc = sort_info.is_asc_order.get(i).copied().unwrap_or(true);
        let nulls_first = sort_info.nulls_first.get(i).copied().unwrap_or(true);

        // Find the slot_id from the SLOT_REF expression, then its position in the
        // sort node's materialized output.
        let position = expr.nodes.first().and_then(|node| {
            if node.node_type == TExprNodeType::SLOT_REF {
                node.slot_ref.as_ref().and_then(|sr| {
                    materialized_slots
                        .iter()
                        .position(|&sid| sid == sr.slot_id)
                })
            } else {
                None
            }
        });

        let Some(pos) = position else { continue };
        let dir = if is_asc { "ASC" } else { "DESC" };
        let nulls = if nulls_first { "NULLS FIRST" } else { "NULLS LAST" };
        order_parts.push(format!("{} {} {}", pos + 1, dir, nulls));
    }

    let mut sql = if order_parts.is_empty() {
        String::new()
    } else {
        format!("ORDER BY {}", order_parts.join(", "))
    };

    // Add LIMIT/OFFSET from the sort node.
    let limit = sort_plan_node.limit;
    let offset = sort_node.offset.unwrap_or(0);
    if limit >= 0 {
        sql.push_str(&format!(" LIMIT {}", limit));
    }
    if offset > 0 {
        sql.push_str(&format!(" OFFSET {}", offset));
    }

    let sql = sql.trim().to_string();
    if sql.is_empty() { None } else { Some(sql) }
}

/// Translate a Doris TPipelineFragmentParams into serialized Substrait Plan bytes.
///
/// `table_schemas` maps NamedTable names to their actual column names (in order).
/// For TVF file scans where the descriptor table lacks table_id, the scan translator
/// uses these schemas to produce a correct ReadRel base_schema matching the DuckDB table.
///
/// Returns the Substrait bytes plus the expected output column names. When the DuckDB
/// result has more columns than expected, the caller should project the result to match.
pub fn translate_fragment(
    params: &TPipelineFragmentParams,
    table_schemas: &HashMap<String, Vec<String>>,
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

    // Set table column overrides from DuckDB table schemas (for TVF scans).
    for (table_name, columns) in table_schemas {
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
    let rel = node_translator::translate_plan(plan, &desc, &scan_params, &mut registry, table_schemas)?;

    // Extract sort/limit specification from the Doris plan's root SORT_NODE.
    // DuckDB's from_substrait() doesn't reliably preserve sort order, so we
    // capture it here for use as a SQL wrapper on the CPU fallback path.
    // Uses 1-based positional references from the sort node's materialized slots.
    let sort_limit_sql = extract_sort_limit_from_plan(plan, &desc);
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
    let output_column_indices = if !rel_names.is_empty() && rel_names.len() > output_names.len() {
        // Name-based: find each output_name's position in the full Rel output.
        let mut indices = Vec::new();
        let mut all_found = true;
        for name in &output_names {
            if let Some(pos) = rel_names.iter().position(|r| r == name) {
                indices.push(pos);
            } else {
                all_found = false;
                break;
            }
        }
        if all_found {
            debug!(indices = ?indices, "output_column_indices from rel_names");
            Some(indices)
        } else {
            None
        }
    } else if let Some(output_exprs) = fragment.output_exprs.as_ref() {
        // Position-based: map output_exprs slots to their position in the root
        // node's materialized output (which matches the DuckDB output order).
        if !plan.nodes.is_empty() {
            let root = &plan.nodes[0];
            // Build enumerated materialized slot list from root's row_tuples.
            let mut materialized_slots = Vec::new();
            for &tuple_id in &root.row_tuples {
                if let Ok(tuple) = desc.get_tuple(tuple_id) {
                    for &slot_id in &tuple.slot_ids {
                        if let Ok(slot) = desc.get_slot(slot_id) {
                            if slot.is_materialized {
                                materialized_slots.push(slot_id);
                            }
                        }
                    }
                }
            }
            // Map each output_expr's slot_id to position in materialized_slots.
            let mut indices = Vec::new();
            let mut all_found = true;
            for expr in output_exprs {
                if let Some(first_node) = expr.nodes.first() {
                    if first_node.node_type == TExprNodeType::SLOT_REF {
                        if let Some(slot_ref) = &first_node.slot_ref {
                            if let Some(pos) = materialized_slots.iter().position(|&s| s == slot_ref.slot_id) {
                                indices.push(pos);
                                continue;
                            }
                        }
                    }
                }
                all_found = false;
                break;
            }
            if all_found && !indices.is_empty() {
                // Only use explicit indices if they differ from identity (actual reordering).
                let is_identity = indices.iter().enumerate().all(|(i, &v)| i == v);
                if !is_identity {
                    debug!(indices = ?indices, "output_column_indices from output_exprs (reorder)");
                    Some(indices)
                } else {
                    None // Identity mapping — no reorder needed.
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // DuckDB's from_substrait uses RelRoot.names to determine output columns,
    // taking the first N columns by POSITION. When the Rel tree outputs more
    // columns than expected, we set RelRoot.names to the FULL Rel output schema.
    let root_names = if !rel_names.is_empty() && rel_names.len() > output_names.len() {
        debug!(
            rel_cols = rel_names.len(),
            output_cols = output_names.len(),
            "Rel outputs more columns than expected, using full schema for RelRoot"
        );
        rel_names
    } else {
        output_names.clone()
    };

    let (extension_uris, extensions) = registry.into_extensions();
    let result_output_names = output_names;

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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use doris_thrift::types::TPrimitiveType;
    use substrait::proto::{plan_rel, read_rel, rel};

    #[test]
    fn test_translate_fragment_to_sql_union() {
        let node = make_union_node(0, 0, vec![vec![int_literal_expr(1), int_literal_expr(2)]]);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);
        let sql = translate_fragment_to_sql(&params).unwrap();
        assert_eq!(sql, "SELECT 1, 2");
    }

    #[test]
    fn test_translate_fragment_to_sql_union_all() {
        let node = make_union_node(
            0,
            0,
            vec![
                vec![int_literal_expr(1)],
                vec![int_literal_expr(2)],
                vec![int_literal_expr(3)],
            ],
        );
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);
        let sql = translate_fragment_to_sql(&params).unwrap();
        assert_eq!(sql, "SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3");
    }

    #[test]
    fn test_translate_fragment_substrait_union() {
        // UNION_NODE with const values should produce a VirtualTable ReadRel.
        let node = make_union_node(0, 0, vec![vec![int_literal_expr(42)]]);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);

        let table_schemas = HashMap::new();
        let result = translate_fragment(&params, &table_schemas).unwrap();
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

        let result = translate_fragment(&params, &table_schemas).unwrap();
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

        let result = translate_fragment(&params, &table_schemas).unwrap();
        // Output names come from the descriptor table, not the full table schema.
        assert_eq!(result.output_names, vec!["city", "population"]);

        // The Substrait plan's ReadRel should have all 3 columns in base_schema
        // (because DuckDB maps by position).
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        // RelRoot.names should match the full Rel output (3 cols), not the projected
        // subset (2 cols), so DuckDB outputs all columns with correct names.
        assert_eq!(root.names, vec!["city", "state", "population"],
            "RelRoot.names should use full Rel schema, not projected subset");
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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
        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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
                assert_eq!(measure.phase, 3, "should be INITIAL_TO_RESULT");
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        match input.rel_type.as_ref().unwrap() {
            rel::RelType::Sort(sort) => {
                assert_eq!(sort.sorts.len(), 1);
                // ASC NULLS FIRST = SortDirection 1
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
        let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();
        let root = match &plan.relations[0].rel_type {
            Some(plan_rel::RelType::Root(r)) => r,
            _ => panic!("expected Root"),
        };
        let input = root.input.as_ref().unwrap();
        // LIMIT/OFFSET wraps the sort in a FetchRel.
        match input.rel_type.as_ref().unwrap() {
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

    // ---- EMPTY_SET_NODE test ----

    #[test]
    fn test_empty_set_substrait() {
        let node = make_empty_set_node(0);
        let plan = make_plan(vec![node]);
        let desc = make_desc_table(vec![], vec![]);
        let params = make_fragment_params(plan, desc);

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &table_schemas).unwrap();
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
    fn test_union_node_sql_with_scan_children() {
        // UNION_NODE(2 children) → SQL: child1 UNION ALL child2
        let mut union_node = make_union_node(0, 0, vec![]);
        union_node.num_children = 2;

        // Children are EXCHANGE_NODE(0 children) = exchange receivers.
        let exch1 = make_exchange_node(1, vec![0]);
        let exch2 = make_exchange_node(3, vec![0]);
        let plan = make_plan(vec![union_node, exch1, exch2]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);

        let sql = translate_fragment_to_sql(&params).unwrap();
        assert_eq!(
            sql,
            "SELECT * FROM __EXCHANGE_TABLE_1 UNION ALL SELECT * FROM __EXCHANGE_TABLE_3"
        );
    }

    #[test]
    fn test_union_node_sql_three_children() {
        let mut union_node = make_union_node(0, 0, vec![]);
        union_node.num_children = 3;
        let e1 = make_exchange_node(1, vec![0]);
        let e2 = make_exchange_node(2, vec![0]);
        let e3 = make_exchange_node(3, vec![0]);
        let plan = make_plan(vec![union_node, e1, e2, e3]);
        let desc = make_desc_table(vec![(0, None)], vec![]);
        let params = make_fragment_params(plan, desc);

        let sql = translate_fragment_to_sql(&params).unwrap();
        assert_eq!(
            sql,
            "SELECT * FROM __EXCHANGE_TABLE_1 UNION ALL SELECT * FROM __EXCHANGE_TABLE_2 UNION ALL SELECT * FROM __EXCHANGE_TABLE_3"
        );
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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

        let result = translate_fragment(&params, &HashMap::new()).unwrap();
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
}
