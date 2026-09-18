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
//!
//! Extension functions are registered through [`ExtensionRegistry`], which
//! de-duplicates anchors by `(urn, name)`.
//!
//! # Status
//!
//! P1 in progress. Done: descriptor table and type mapping (P1.1), expression
//! translation (P1.2). Pending: node translation and the single-plan stitcher;
//! until then [`PlanTranslator::translate_fragment`] rejects every fragment with
//! a structured error naming its root node, so the backend's translate-only
//! survey mode records exactly which node types the corpus needs.

use std::collections::HashMap;
use std::fmt;

use doris_thrift::palo_internal_service::TPipelineFragmentParams;
use prost::Message;
use substrait::proto::Plan;
use substrait::proto::extensions::simple_extension_declaration;
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};

pub mod descriptor_table;
pub mod error;
pub mod expr_translator;
pub mod type_mapper;

pub use descriptor_table::{DescriptorTable, SlotInfo, TupleInfo};
use error::Result;
pub use error::TranslateError;

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
    /// P0: every fragment is rejected with an error naming its root node, so the backend's
    /// translate-only survey mode records exactly which node types the corpus needs.
    pub fn translate_fragment(&self, params: &TPipelineFragmentParams) -> Result<TranslatedPlan> {
        let plan = params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.plan.as_ref())
            .ok_or(TranslateError::MissingField {
                context: "TPipelineFragmentParams.fragment",
                field: "plan",
            })?;
        let root = plan
            .nodes
            .first()
            .ok_or_else(|| TranslateError::malformed("fragment plan has no nodes"))?;
        Err(TranslateError::UnsupportedPlanNode {
            node_id: root.node_id,
            node_type: root.node_type,
            reason: "plan translation is not implemented yet (P1)",
        })
    }
}

impl Default for PlanTranslator {
    fn default() -> Self {
        Self::new()
    }
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
    use doris_thrift::palo_internal_service::PaloInternalServiceVersion;
    use doris_thrift::partitions::{TDataPartition, TPartitionType};
    use doris_thrift::plan_nodes::{TPlan, TPlanNode, TPlanNodeType};
    use doris_thrift::planner::TPlanFragment;
    use doris_thrift::types::TUniqueId;

    use super::*;

    fn fragment_with_root(node_type: TPlanNodeType) -> TPipelineFragmentParams {
        let node = TPlanNode {
            node_id: 7,
            node_type,
            num_children: 0,
            limit: -1,
            row_tuples: vec![0],
            nullable_tuples: vec![false],
            compact_data: false,
            ..Default::default()
        };
        let fragment = TPlanFragment {
            plan: Some(TPlan { nodes: vec![node] }),
            partition: TDataPartition {
                type_: TPartitionType::UNPARTITIONED,
                ..Default::default()
            },
            ..Default::default()
        };
        TPipelineFragmentParams {
            protocol_version: PaloInternalServiceVersion::V1,
            query_id: TUniqueId::new(1, 2),
            fragment: Some(fragment),
            ..Default::default()
        }
    }

    #[test]
    fn p0_translator_names_the_root_node_it_rejects() {
        let err = PlanTranslator::new()
            .translate_fragment(&fragment_with_root(TPlanNodeType::FILE_SCAN_NODE))
            .unwrap_err();
        assert_eq!(
            err,
            TranslateError::UnsupportedPlanNode {
                node_id: 7,
                node_type: TPlanNodeType::FILE_SCAN_NODE,
                reason: "plan translation is not implemented yet (P1)",
            }
        );
    }

    #[test]
    fn missing_plan_is_a_missing_field_error() {
        let mut params = fragment_with_root(TPlanNodeType::FILE_SCAN_NODE);
        params.fragment = None;
        let err = PlanTranslator::new()
            .translate_fragment(&params)
            .unwrap_err();
        assert!(matches!(
            err,
            TranslateError::MissingField { field: "plan", .. }
        ));
    }

    #[test]
    fn empty_plan_is_malformed() {
        let mut params = fragment_with_root(TPlanNodeType::FILE_SCAN_NODE);
        params.fragment.as_mut().unwrap().plan = Some(TPlan { nodes: vec![] });
        let err = PlanTranslator::new()
            .translate_fragment(&params)
            .unwrap_err();
        assert!(matches!(err, TranslateError::MalformedPlan(_)));
    }
}
