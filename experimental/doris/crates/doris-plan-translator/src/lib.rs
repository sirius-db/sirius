//! Doris plan-fragment to Substrait translation.
//!
//! This crate converts a Doris `TPipelineFragmentParams` (one plan fragment the
//! Doris frontend ships to a backend inside a `TPipelineFragmentParamsList`) into
//! a `substrait` `Plan` for the Sirius engine. It follows the structure of the
//! StarRocks translator (`experimental/starrocks/crates/starrocks-plan-translator`):
//! one fragment at a time, checked invariants over breadth, and everything
//! outside the supported surface returns a structured [`TranslateError`] that
//! names the offending node/type.
//!
//! # Wire format: flat preorder
//!
//! Doris encodes both plan trees (`TPlan.nodes`) and expression trees
//! (`TExpr.nodes`) as a *flat* list of nodes in preorder: each node carries a
//! `num_children` count, and its children are the nodes that immediately follow
//! it (depth-first). Node and expression translators must rebuild the tree with
//! a cursor that consumes exactly `num_children` nodes per parent and then
//! asserts the whole slice was consumed; a malformed plan is rejected as
//! [`TranslateError::MalformedPlan`] instead of silently producing a truncated
//! tree. **Any node translator added here must preserve this invariant.**
//!
//! # Modules
//!
//! - [`descriptor_table`]: `TDescriptorTable` → tuples, slots in wire order, and
//!   `(tuple_id, slot_id)` → column-index resolution for a node's `row_tuples`.
//! - [`type_mapper`]: `TTypeDesc` → Substrait type, and the type gate (what is
//!   rejected and which `semantics-gaps.md` entry says why).
//! - [`expr_translator`]: `TExpr` → Substrait expression (literals, slot references,
//!   predicates, arithmetic, casts, `IN`, `CASE`, the scalar-function allowlist)
//!   and the decomposition of `AGG_EXPR` roots into aggregate measures.
//! - [`node_translator`]: `TPlan` → Substrait relation tree, one fragment at a time
//!   (scan, exchange as a named stream, joins, aggregate, sort; conjuncts, limit
//!   and the projection chain on every node).
//! - [`scan_ranges`]: the parquet paths a fragment's scans read, validated.
//! - [`stitcher`]: the MVP-A0 single-plan stitcher — the fragments of one dispatch
//!   joined into one fragment (exchanges replaced by their senders, two-phase
//!   aggregates collapsed) for [`PlanTranslator::translate_batch`].
//!
//! Extension functions are registered through [`ExtensionRegistry`], which
//! de-duplicates anchors by `(urn, name)`.
//!
//! # Entry points
//!
//! - [`PlanTranslator::translate_fragment`]: one fragment in isolation; exchanges
//!   become `ReadRel`s over `sirius_stream_<node_id>` (the shape MVP-A's real
//!   fragments need), two-phase aggregates are rejected.
//! - [`PlanTranslator::translate_batch`]: every fragment the FE sent this backend,
//!   stitched into one plan (MVP-A0).

use std::collections::{HashMap, HashSet};
use std::fmt;

use doris_thrift::palo_internal_service::TPipelineFragmentParams;
use prost::Message;
use substrait::proto::extensions::simple_extension_declaration;
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};
use substrait::proto::{Plan, PlanRel, RelRoot, plan_rel};

pub mod descriptor_table;
pub mod error;
pub mod expr_translator;
pub mod node_translator;
pub mod scan_ranges;
pub mod stitcher;
pub mod type_mapper;

pub use descriptor_table::{DescriptorTable, SlotInfo, TupleInfo};
use error::Result;
pub use error::TranslateError;
pub use scan_ranges::ScanRanges;
pub use stitcher::stitch_fragments;

/// Substrait comparison function extension URN.
pub const URN_COMPARISON: &str = "extension:io.substrait:functions_comparison";
/// Substrait boolean function extension URN.
pub const URN_BOOLEAN: &str = "extension:io.substrait:functions_boolean";
/// Substrait arithmetic function extension URN.
pub const URN_ARITHMETIC: &str = "extension:io.substrait:functions_arithmetic";
/// Substrait string function extension URN.
pub const URN_STRING: &str = "extension:io.substrait:functions_string";
/// Substrait datetime function extension URN.
pub const URN_DATETIME: &str = "extension:io.substrait:functions_datetime";
/// Substrait aggregate function extension URN.
pub const URN_AGGREGATE: &str = "extension:io.substrait:functions_aggregate_generic";

/// Result of translating one Doris plan fragment.
#[derive(Clone, PartialEq)]
pub struct TranslatedPlan {
    /// Structured Substrait protobuf plan.
    pub plan: Plan,
    /// Root output names as emitted in the Substrait plan.
    pub output_names: Vec<String>,
}

impl TranslatedPlan {
    /// Returns a human-readable Substrait text formatter for logging and debugging.
    pub fn explain(&self) -> PlanExplain<'_> {
        PlanExplain { plan: &self.plan }
    }

    /// Encodes the Substrait plan to protobuf bytes on demand.
    pub fn to_substrait_bytes(&self) -> Vec<u8> {
        self.plan.encode_to_vec()
    }
}

impl fmt::Debug for TranslatedPlan {
    /// Renders the translated plan using `substrait-explain` instead of protobuf debug output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TranslatedPlan")
            .field("output_names", &self.output_names)
            .field("plan", &self.explain())
            .finish()
    }
}

/// Display/debug adapter for a Substrait plan in explain-text form.
pub struct PlanExplain<'a> {
    /// Plan to render with `substrait-explain`.
    plan: &'a Plan,
}

impl fmt::Display for PlanExplain<'_> {
    /// Formats the plan text and appends formatter warnings when present.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (text, warnings) = substrait_explain::format(self.plan);
        f.write_str(&text)?;
        if !warnings.is_empty() {
            write!(f, "\nformat warnings: {warnings:?}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for PlanExplain<'_> {
    /// Delegates debug output to the same readable text as display output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Reusable Doris plan-fragment translator.
#[derive(Clone, Debug)]
pub struct PlanTranslator {
    /// Producer string embedded in emitted Substrait plans.
    producer: String,
}

impl PlanTranslator {
    /// Creates a translator with the default Sirius producer string.
    pub fn new() -> Self {
        Self {
            producer: "sirius-doris-plan-translator".to_string(),
        }
    }

    /// Creates a translator with a caller-provided Substrait producer string.
    pub fn with_producer(producer: impl Into<String>) -> Self {
        Self {
            producer: producer.into(),
        }
    }

    /// Producer string embedded in emitted Substrait plans.
    pub fn producer(&self) -> &str {
        &self.producer
    }

    /// Translates one fragment (with the list-level shared fields already merged into it by
    /// the backend's `params` module) into a Substrait plan.
    ///
    /// The fragment is translated in isolation: its `EXCHANGE_NODE`s become reads of the
    /// named tables `sirius_stream_<node_id>`, and a two-phase aggregate is rejected (the
    /// single-plan stitcher rewrites those before calling this).
    pub fn translate_fragment(&self, params: &TPipelineFragmentParams) -> Result<TranslatedPlan> {
        let fragment = params
            .fragment
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "TPipelineFragmentParams",
                field: "fragment",
            })?;
        let plan = fragment.plan.as_ref().ok_or(TranslateError::MissingField {
            context: "TPipelineFragmentParams.fragment",
            field: "plan",
        })?;
        let desc_tbl = params
            .desc_tbl
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "TPipelineFragmentParams",
                field: "desc_tbl",
            })?;
        let desc = DescriptorTable::try_from(desc_tbl)?;
        let scan_ranges = ScanRanges::from_params(params)?;
        let mut registry = ExtensionRegistry::new();
        let translated = node_translator::translate_plan(plan, &desc, &scan_ranges, &mut registry)?;
        let (translated, output_names) = match fragment
            .output_exprs
            .as_deref()
            .filter(|exprs| !exprs.is_empty())
        {
            Some(exprs) => {
                node_translator::project_output_exprs(translated, exprs, &desc, &mut registry)?
            }
            None => {
                let names = node_translator::output_names(&translated, &desc)?;
                (translated, names)
            }
        };
        let output_names = unique_names(output_names).collect::<Vec<_>>();
        let (extension_urns, extensions) = registry.into_extensions();
        let plan = Plan {
            // The spec version comes from the `substrait` crate so it tracks the generated
            // proto types instead of drifting on dependency bumps.
            version: Some(substrait::version::version_with_producer(
                self.producer.clone(),
            )),
            extension_urns,
            extensions,
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(translated.rel),
                    names: output_names.clone(),
                })),
            }],
            ..Default::default()
        };
        Ok(TranslatedPlan { plan, output_names })
    }

    /// Stitches the fragments of one dispatch into a single plan (MVP-A0) and translates it.
    pub fn translate_batch(
        &self,
        fragments: &[&TPipelineFragmentParams],
    ) -> Result<TranslatedPlan> {
        let stitched = stitch_fragments(fragments)?;
        self.translate_fragment(&stitched)
    }
}

impl Default for PlanTranslator {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns collision-free root names, keeping the first spelling of each.
fn unique_names(names: impl IntoIterator<Item = String>) -> impl Iterator<Item = String> {
    let mut used = HashSet::<String>::new();
    let mut next_suffix = HashMap::<String, usize>::new();
    names.into_iter().map(move |name| {
        if used.insert(name.clone()) {
            return name;
        }
        let suffix = next_suffix.entry(name.clone()).or_insert(1);
        loop {
            let candidate = format!("{}_{}", name, *suffix);
            *suffix += 1;
            if used.insert(candidate.clone()) {
                return candidate;
            }
        }
    })
}

/// Registry for the Substrait extension URNs and function anchors a plan emits.
#[derive(Debug, Default)]
pub struct ExtensionRegistry {
    /// URN declarations in anchor allocation order.
    urns: Vec<SimpleExtensionUrn>,
    /// Function declarations in anchor allocation order.
    functions: Vec<SimpleExtensionDeclaration>,
    /// Allocated URN anchors keyed by URN.
    urn_map: HashMap<String, u32>,
    /// Allocated function anchors keyed by extension URN and function name.
    function_map: HashMap<(String, String), u32>,
    /// Next URN anchor to allocate.
    next_urn_anchor: u32,
    /// Next function anchor to allocate.
    next_function_anchor: u32,
}

impl ExtensionRegistry {
    /// Creates an empty registry with anchors starting at one.
    pub fn new() -> Self {
        Self {
            next_urn_anchor: 1,
            next_function_anchor: 1,
            ..Default::default()
        }
    }

    /// Registers or reuses the function anchor for `(urn, name)`.
    pub fn register_function(&mut self, urn: &str, name: &str) -> u32 {
        let key = (urn.to_string(), name.to_string());
        if let Some(anchor) = self.function_map.get(&key) {
            return *anchor;
        }
        let urn_anchor = self.ensure_urn(urn);
        let function_anchor = self.next_function_anchor;
        self.next_function_anchor += 1;
        self.functions.push(SimpleExtensionDeclaration {
            mapping_type: Some(
                simple_extension_declaration::MappingType::ExtensionFunction(
                    simple_extension_declaration::ExtensionFunction {
                        extension_urn_reference: urn_anchor,
                        function_anchor,
                        name: name.to_string(),
                    },
                ),
            ),
        });
        self.function_map.insert(key, function_anchor);
        function_anchor
    }

    /// The function name registered under `anchor`, if any.
    pub fn function_name(&self, anchor: u32) -> Option<&str> {
        self.function_map
            .iter()
            .find(|(_, candidate)| **candidate == anchor)
            .map(|((_, name), _)| name.as_str())
    }

    /// Registers or reuses the anchor for an extension URN.
    fn ensure_urn(&mut self, urn: &str) -> u32 {
        if let Some(anchor) = self.urn_map.get(urn) {
            return *anchor;
        }
        let anchor = self.next_urn_anchor;
        self.next_urn_anchor += 1;
        self.urns.push(SimpleExtensionUrn {
            extension_urn_anchor: anchor,
            urn: urn.to_string(),
        });
        self.urn_map.insert(urn.to_string(), anchor);
        anchor
    }

    /// Consumes the registry into the vectors `substrait::proto::Plan` expects.
    pub fn into_extensions(self) -> (Vec<SimpleExtensionUrn>, Vec<SimpleExtensionDeclaration>) {
        (self.urns, self.functions)
    }
}

#[cfg(test)]
mod tests {
    use doris_thrift::descriptors::{TDescriptorTable, TSlotDescriptor, TTupleDescriptor};
    use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType, TSlotRef};
    use doris_thrift::palo_internal_service::{
        PaloInternalServiceVersion, TPipelineInstanceParams, TScanRangeParams,
    };
    use doris_thrift::partitions::{TDataPartition, TPartitionType};
    use doris_thrift::plan_nodes::{
        TExternalScanRange, TFileFormatType, TFileRangeDesc, TFileScanNode, TFileScanRange, TPlan,
        TPlanNode, TPlanNodeType, TScanRange,
    };
    use doris_thrift::planner::TPlanFragment;
    use doris_thrift::types::{
        TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType, TUniqueId,
    };

    use super::*;

    fn int_desc() -> TTypeDesc {
        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(TScalarType {
                    type_: TPrimitiveType::INT,
                    ..Default::default()
                }),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    /// A one-node fragment: `FILE_SCAN_NODE` over tuple 0 = {1: a INT, 2: b INT}, one
    /// parquet range, output expression `b` labelled `total`.
    fn scan_fragment() -> TPipelineFragmentParams {
        let node = TPlanNode {
            node_id: 7,
            node_type: TPlanNodeType::FILE_SCAN_NODE,
            num_children: 0,
            limit: -1,
            row_tuples: vec![0],
            nullable_tuples: vec![false],
            file_scan_node: Some(TFileScanNode {
                tuple_id: Some(0),
                table_name: None,
            }),
            ..Default::default()
        };
        let output_expr = TExpr {
            nodes: vec![TExprNode {
                node_type: TExprNodeType::SLOT_REF,
                type_: int_desc(),
                num_children: 0,
                output_scale: -1,
                is_nullable: Some(true),
                slot_ref: Some(TSlotRef {
                    slot_id: 2,
                    tuple_id: 0,
                    ..Default::default()
                }),
                label: Some("total".to_string()),
                ..Default::default()
            }],
        };
        let fragment = TPlanFragment {
            plan: Some(TPlan { nodes: vec![node] }),
            output_exprs: Some(vec![output_expr]),
            partition: TDataPartition {
                type_: TPartitionType::UNPARTITIONED,
                ..Default::default()
            },
            ..Default::default()
        };
        let slot = |id, name: &str| TSlotDescriptor {
            id,
            parent: 0,
            slot_type: int_desc(),
            column_pos: -1,
            null_indicator_bit: 0,
            col_name: name.to_string(),
            slot_idx: -1,
            is_materialized: true,
            ..Default::default()
        };
        let range = TScanRangeParams {
            scan_range: TScanRange {
                ext_scan_range: Some(TExternalScanRange {
                    file_scan_range: Some(TFileScanRange {
                        ranges: Some(vec![TFileRangeDesc {
                            path: Some("/data/t.parquet".to_string()),
                            start_offset: Some(0),
                            size: Some(10),
                            file_size: Some(10),
                            format_type: Some(TFileFormatType::FORMAT_PARQUET),
                            ..Default::default()
                        }]),
                        ..Default::default()
                    }),
                }),
                ..Default::default()
            },
            ..Default::default()
        };
        TPipelineFragmentParams {
            protocol_version: PaloInternalServiceVersion::V1,
            query_id: TUniqueId::new(1, 2),
            desc_tbl: Some(TDescriptorTable {
                slot_descriptors: Some(vec![slot(1, "a"), slot(2, "b")]),
                tuple_descriptors: vec![TTupleDescriptor {
                    id: 0,
                    ..Default::default()
                }],
                table_descriptors: None,
            }),
            fragment: Some(fragment),
            local_params: Some(vec![TPipelineInstanceParams {
                fragment_instance_id: TUniqueId::new(1, 3),
                per_node_scan_ranges: [(7, vec![range])].into_iter().collect(),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    #[test]
    fn translates_a_scan_fragment_end_to_end() {
        let plan = PlanTranslator::new()
            .translate_fragment(&scan_fragment())
            .unwrap();
        assert_eq!(plan.output_names, vec!["total"]);
        let text = plan.explain().to_string();
        assert!(text.contains("Root[total]"), "{text}");
        assert!(text.contains("Project[$1]"), "{text}");
        assert!(text.contains("a:i32?, b:i32?"), "{text}");
        assert_eq!(
            plan.plan.version.as_ref().unwrap().producer,
            "sirius-doris-plan-translator"
        );
        assert!(!plan.to_substrait_bytes().is_empty());
    }

    #[test]
    fn output_names_fall_back_to_the_layout_and_are_made_unique() {
        let mut params = scan_fragment();
        params.fragment.as_mut().unwrap().output_exprs = None;
        let plan = PlanTranslator::new().translate_fragment(&params).unwrap();
        assert_eq!(plan.output_names, vec!["a", "b"]);
        assert_eq!(
            unique_names(["x", "x", "y", "x"].map(String::from)).collect::<Vec<_>>(),
            vec!["x", "x_1", "y", "x_2"]
        );
    }

    #[test]
    fn missing_fragment_plan_or_descriptor_table_is_a_missing_field_error() {
        let mut params = scan_fragment();
        params.fragment.as_mut().unwrap().plan = None;
        assert!(matches!(
            PlanTranslator::new()
                .translate_fragment(&params)
                .unwrap_err(),
            TranslateError::MissingField { field: "plan", .. }
        ));
        let mut params = scan_fragment();
        params.fragment = None;
        assert!(matches!(
            PlanTranslator::new()
                .translate_fragment(&params)
                .unwrap_err(),
            TranslateError::MissingField {
                field: "fragment",
                ..
            }
        ));
        let mut params = scan_fragment();
        params.desc_tbl = None;
        assert!(matches!(
            PlanTranslator::new()
                .translate_fragment(&params)
                .unwrap_err(),
            TranslateError::MissingField {
                field: "desc_tbl",
                ..
            }
        ));
    }

    #[test]
    fn empty_plan_is_malformed() {
        let mut params = scan_fragment();
        params.fragment.as_mut().unwrap().plan = Some(TPlan { nodes: vec![] });
        assert!(matches!(
            PlanTranslator::new()
                .translate_fragment(&params)
                .unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
    }
}
