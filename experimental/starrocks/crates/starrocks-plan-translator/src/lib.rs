//! StarRocks plan-fragment to Substrait translation.
//!
//! This crate converts a StarRocks `TExecPlanFragmentParams` (the Thrift plan a
//! StarRocks frontend ships to a backend) into a `substrait` `Plan`. It is meant
//! as a foundation more operators are built on, so it favours explicit, checked
//! invariants over breadth: it translates one fragment at a time, and everything
//! outside the supported surface returns a structured [`TranslateError`] that
//! names the offending node/type — so the next contributor knows exactly what to
//! implement next. In particular `EXCHANGE_NODE` is rejected: a fragment is
//! translated in isolation, and multi-fragment plans (every exchange is a
//! fragment boundary) are a later milestone.
//!
//! # Wire format: flat preorder
//!
//! StarRocks encodes both plan trees (`TPlan.nodes`) and expression trees
//! (`TExpr.nodes`) as a *flat* list of nodes in preorder: each node carries a
//! `num_children` count, and its children are the nodes that immediately follow
//! it (depth-first). The translators rebuild the tree with a cursor
//! (`PlanNodeCursor` / `ExprNodeCursor`) that recursively consumes exactly
//! `num_children` nodes per parent, then asserts via `ensure_consumed` that the
//! whole slice was consumed. A malformed plan — a wrong child count, trailing
//! nodes, or an under-run — is rejected as [`TranslateError::MalformedPlan`]
//! instead of silently producing a truncated tree. **Any new node translator
//! must preserve this invariant.**
//!
//! # Supported surface
//!
//! | Plan node            | Substrait relation |
//! |----------------------|--------------------|
//! | `FILE_SCAN_NODE`     | `ReadRel` (local files) |
//! | `HDFS_SCAN_NODE`     | `ReadRel` (named table) |
//! | `SELECT_NODE`        | `FilterRel`        |
//! | `PROJECT_NODE`       | `ProjectRel`       |
//! | `AGGREGATION_NODE`   | `AggregateRel` (finalized one-phase only, `new_planner_agg_stage=1`) |
//! | `SORT_NODE`          | `ProjectRel` (sort tuple) + `SortRel` (global row-number top-N only) |
//! | `HASH_JOIN_NODE`     | `JoinRel` (inner/outer/left-semi; anti joins are rejected) |
//! | `NESTLOOP_JOIN_NODE` | `CrossRel` (+ `FilterRel`), inner/cross only |
//!
//! Node-level `conjuncts` (scan/filter predicates, HAVING, post-join filters) become a
//! `FilterRel` over the node's output on every supported node.
//!
//! Any node's non-negative `limit` (plus a sort offset) becomes a `FetchRel` on top of its
//! relation.
//!
//! | Expression node   | Substrait expression |
//! |-------------------|----------------------|
//! | `SLOT_REF`        | field reference (resolved by `DescriptorTable::slot_global_index`) |
//! | `*_LITERAL`       | width-matched typed literal (incl. `DATE_LITERAL` as epoch days) |
//! | `BINARY_PRED`     | comparison function (`equal`, `not_equal`, `lt`, `lte`, `gt`, `gte`) |
//! | `COMPOUND_PRED`   | boolean function (`and`, `or`, `not`) |
//! | `CAST_EXPR`       | cast (throwing failure behavior) |
//! | `IS_NULL_PRED`    | `is_null` / `is_not_null` |
//! | `ARITHMETIC_EXPR` | `add`/`subtract`/`multiply`/`divide`/`modulus` |
//! | `IN_PRED`         | singular-or-list (wrapped in `not` for `NOT IN`) |
//! | `CASE_EXPR`       | if-then chain (no leading case operand) |
//! | `FUNCTION_CALL`   | allowlisted scalar functions (`like`, `if`, `substring`, `year`, ...) |
//!
//! Aggregate functions (`sum`, `count`, `min`, `max`, `avg`, and the
//! `multi_distinct_*` distinct forms) are decomposed by `expr_translator::aggregate_call` for
//! `AggregateRel` measures; only non-merge (one-phase) aggregates are accepted.
//!
//! Type mapping lives in `type_mapper`. Intentional v1 omissions return
//! [`TranslateError::UnsupportedType`]: `LARGEINT` (128-bit), `DECIMAL256` and
//! decimal precision &gt; 38 (both exceed the i128 decimal encoding), and
//! non-scalar type nodes. `JSON`/`VARIANT` are surfaced as strings until richer
//! support lands.
//!
//! # Adding a node
//!
//! 1. Add a match arm in `PlanNodeRel`/`NodeExpr` routing to a translator.
//! 2. Build the Substrait relation/expression, translating children first.
//! 3. For relations, set `TranslatedRel::row_tuples` (the visible row layout) and
//!    `TranslatedRel::output_width` (emitted column count) so parent projections
//!    compute correct field offsets. Multi-input relations concatenate child
//!    layouts left-to-right; `DescriptorTable::slot_global_index` then resolves a
//!    right-side slot to `left_width + right_index`.
//! 4. Register any extension function through [`ExtensionRegistry`], which
//!    de-duplicates anchors by `(urn, name)`.

use std::collections::{HashMap, HashSet};
use std::fmt;

use prost::Message;
use starrocks_thrift::exprs::{TExpr, TExprNodeType};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use substrait::proto::extensions::simple_extension_declaration;
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};
use substrait::proto::{Plan, PlanRel, RelRoot, plan_rel};

// These modules are crate-internal plumbing. The public surface this foundation
// commits to is intentionally small: `PlanTranslator`, `translate_fragment`,
// `TranslatedPlan`, `TranslateError`, and the extension registry. Widen a module
// to `pub` only when a real consumer needs it.
pub(crate) mod descriptor_table;
pub mod error;
mod expr_translator;
mod node_translator;
mod scan_paths;
pub(crate) mod type_mapper;

use descriptor_table::{DescriptorTable, SlotInfo};
use error::Result;
pub use error::TranslateError;
use scan_paths::ScanFilePaths;

/// Substrait comparison function extension URN used for scalar predicates.
pub const URN_COMPARISON: &str = "extension:io.substrait:functions_comparison";
/// Substrait boolean function extension URN used for compound predicates.
pub const URN_BOOLEAN: &str = "extension:io.substrait:functions_boolean";
/// Substrait arithmetic function extension URN.
pub const URN_ARITHMETIC: &str = "extension:io.substrait:functions_arithmetic";
/// Substrait string function extension URN.
pub const URN_STRING: &str = "extension:io.substrait:functions_string";
/// Substrait datetime function extension URN.
pub const URN_DATETIME: &str = "extension:io.substrait:functions_datetime";
/// Substrait aggregate function extension URN.
pub const URN_AGGREGATE: &str = "extension:io.substrait:functions_aggregate_generic";

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
        let scan_paths = ScanFilePaths::from_fragment(params, &desc)?;
        let mut registry = ExtensionRegistry::new();
        let mut translated =
            node_translator::translate_plan(plan, &desc, &scan_paths, &mut registry)?;

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
            // Source the spec version from the `substrait` crate so it tracks the
            // generated proto types instead of drifting on dependency bumps (a
            // hardcoded literal silently diverges from the crate it describes).
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
            let slot_ref = node.slot_ref.as_ref()?;
            desc.slot(slot_ref.tuple_id, slot_ref.slot_id)
                .ok()
                .map(SlotInfo::output_name)
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
