//! StarRocks plan-fragment to Substrait translation.
//!
//! This crate intentionally supports a narrow first slice: scan, filter, and
//! projection. Everything outside that surface returns a structured
//! `TranslateError` so later expansion has clear ownership boundaries.

use std::collections::{HashMap, HashSet};
use std::fmt;

use prost::Message;
use starrocks_thrift::exprs::{TExpr, TExprNodeType};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use substrait::proto::extensions::simple_extension_declaration;
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};
use substrait::proto::{Plan, PlanRel, RelRoot, Version, plan_rel};

pub mod descriptor_table;
pub mod error;
mod expr_translator;
pub mod node_translator;
pub mod type_mapper;

use descriptor_table::{DescriptorTable, SlotInfo};
use error::Result;
pub use error::TranslateError;

/// Substrait comparison function extension URN used for scalar predicates.
pub const URN_COMPARISON: &str = "extension:io.substrait:functions_comparison";
/// Substrait boolean function extension URN used for compound predicates.
pub const URN_BOOLEAN: &str = "extension:io.substrait:functions_boolean";

/// Result of translating one StarRocks plan fragment.
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

/// Reusable StarRocks plan-fragment translator.
#[derive(Clone, Debug)]
pub struct PlanTranslator {
    /// Producer string embedded in emitted Substrait plans.
    producer: String,
}

impl PlanTranslator {
    /// Creates a translator with the default Sirius producer string.
    pub fn new() -> Self {
        Self {
            producer: "sirius-starrocks-plan-translator".to_string(),
        }
    }

    /// Creates a translator with a caller-provided Substrait producer string.
    pub fn with_producer(producer: impl Into<String>) -> Self {
        Self {
            producer: producer.into(),
        }
    }

    /// Translates a StarRocks execution fragment into a Substrait plan.
    pub fn translate_fragment(&self, params: &TExecPlanFragmentParams) -> Result<TranslatedPlan> {
        let fragment = params
            .fragment
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "TExecPlanFragmentParams",
                field: "fragment",
            })?;
        let plan = fragment.plan.as_ref().ok_or(TranslateError::MissingField {
            context: "TPlanFragment",
            field: "plan",
        })?;
        let desc_tbl = params
            .desc_tbl
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "TExecPlanFragmentParams",
                field: "desc_tbl",
            })?;

        let desc = DescriptorTable::try_from(desc_tbl)?;
        let mut registry = ExtensionRegistry::new();
        let mut translated = node_translator::translate_plan(plan, &desc, &mut registry)?;

        let output_names = if let Some(output_exprs) = fragment
            .output_exprs
            .as_ref()
            .filter(|output_exprs| !output_exprs.is_empty())
        {
            let names = output_exprs
                .iter()
                .enumerate()
                .map(|(idx, expr)| {
                    output_name_for_expr(expr, &desc).unwrap_or_else(|| format!("expr_{idx}"))
                })
                .collect::<Vec<_>>();
            translated =
                node_translator::project_exprs(translated, output_exprs, &desc, &mut registry)?;
            names
        } else {
            desc.output_names_for_tuples(&translated.row_tuples)?
        };

        let output_names = unique_names(output_names).collect::<Vec<_>>();
        let (extension_urns, extensions) = registry.into_extensions();
        let substrait_plan = Plan {
            version: Some(Version {
                major_number: 0,
                minor_number: 62,
                patch_number: 0,
                producer: self.producer.clone(),
                ..Default::default()
            }),
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

        Ok(TranslatedPlan {
            plan: substrait_plan,
            output_names,
        })
    }
}

impl Default for PlanTranslator {
    /// Creates a translator with the default Sirius producer string.
    fn default() -> Self {
        Self::new()
    }
}

/// Registry for Substrait extension URN and function anchors emitted by this translator.
#[derive(Default)]
pub struct ExtensionRegistry {
    /// URN declarations in anchor allocation order.
    urns: Vec<SimpleExtensionUrn>,
    /// Function declarations in anchor allocation order.
    functions: Vec<SimpleExtensionDeclaration>,
    /// Allocated URN anchors keyed by URN.
    urn_map: HashMap<String, u32>,
    /// Allocated function anchors keyed by extension URN and function name.
    function_map: HashMap<(String, String), u32>,
    /// Next Substrait URN anchor to allocate.
    next_urn_anchor: u32,
    /// Next Substrait function anchor to allocate.
    next_function_anchor: u32,
}

impl ExtensionRegistry {
    /// Creates an empty registry with Substrait anchors starting at one.
    pub fn new() -> Self {
        Self {
            next_urn_anchor: 1,
            next_function_anchor: 1,
            ..Default::default()
        }
    }

    /// Registers or reuses a function anchor for a Substrait extension function.
    pub fn register_function(&mut self, urn: &str, name: &str) -> u32 {
        let function_key = (urn.to_string(), name.to_string());
        if let Some(anchor) = self.function_map.get(&function_key) {
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
                        ..Default::default()
                    },
                ),
            ),
        });
        self.function_map.insert(function_key, function_anchor);
        function_anchor
    }

    /// Registers or reuses a URN anchor for an extension URN.
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

    /// Consumes the registry into the vectors expected by `substrait::proto::Plan`.
    fn into_extensions(self) -> (Vec<SimpleExtensionUrn>, Vec<SimpleExtensionDeclaration>) {
        (self.urns, self.functions)
    }
}

/// Translates a StarRocks execution fragment using the default reusable translator.
pub fn translate_fragment(params: &TExecPlanFragmentParams) -> Result<TranslatedPlan> {
    PlanTranslator::new().translate_fragment(params)
}

/// Uses a simple slot-reference output expression as its root output name.
fn output_name_for_expr(expr: &TExpr, desc: &DescriptorTable) -> Option<String> {
    match expr.nodes.as_slice() {
        [node] if node.node_type == TExprNodeType::SLOT_REF => {
            let slot_id = node.slot_ref.as_ref()?.slot_id;
            desc.slot(slot_id).ok().map(SlotInfo::output_name)
        }
        _ => None,
    }
}

/// Returns collision-free root names while preserving first occurrence spelling.
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
