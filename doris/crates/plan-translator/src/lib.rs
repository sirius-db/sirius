//! Doris plan → Substrait plan translation.
//!
//! Converts Doris TPlanNode trees (received via Thrift) into Substrait Plan
//! protobuf messages (pure Rust, no FFI). The Substrait plan is then passed
//! to the C++ bridge which uses the DuckDB Substrait extension to consume it.

use std::collections::HashMap;

use anyhow::{Context, Result};
use prost::Message;
use tracing::debug;

use doris_thrift::palo_internal_service::TPipelineFragmentParams;
use substrait::proto::extensions::simple_extension_declaration;
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUri};
use substrait::proto::{Plan, PlanRel, RelRoot, Version};

pub mod descriptor_table;
pub mod expr_translator;
pub mod node_translator;
pub mod scan_translator;
pub mod sql_generator;
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

/// Translate a Doris TPipelineFragmentParams into serialized Substrait Plan bytes.
///
/// `table_schemas` maps NamedTable names to their actual column names (in order).
/// For TVF file scans where the descriptor table lacks table_id, the scan translator
/// uses these schemas to produce a correct ReadRel base_schema matching the DuckDB table.
pub fn translate_fragment(
    params: &TPipelineFragmentParams,
    table_schemas: &HashMap<String, Vec<String>>,
) -> Result<Vec<u8>> {
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

    let mut registry = ExtensionRegistry::new();

    // Translate the plan tree into a Substrait Rel tree.
    let rel = node_translator::translate_plan(plan, &desc, &scan_params, &mut registry, table_schemas)?;

    // Build output names from the root node's output tuples.
    let output_names = if !plan.nodes.is_empty() {
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
        // Deduplicate names (joins can produce duplicate column names like "id" from
        // both sides, which DuckDB cannot handle in from_substrait output tables).
        let mut name_counts: HashMap<String, usize> = HashMap::new();
        let mut unique = Vec::new();
        for name in names {
            let count = name_counts.entry(name.clone()).or_insert(0);
            if *count == 0 {
                unique.push(name);
            } else {
                unique.push(format!("{}:{}", name, count));
            }
            *count += 1;
        }
        unique
    } else {
        Vec::new()
    };

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
                names: output_names,
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
    Ok(bytes)
}

