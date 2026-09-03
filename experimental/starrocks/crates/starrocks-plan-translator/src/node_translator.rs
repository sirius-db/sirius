use std::collections::{BTreeMap, BTreeSet, HashMap};

use starrocks_thrift::exprs::TExpr;
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::plan_nodes::{TJoinOp, TPlan, TPlanNode, TPlanNodeType, TSortInfo};
use starrocks_thrift::types::TSlotId;
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{LocalFiles, NamedTable, ReadType};
use substrait::proto::{
    AggregateFunction, AggregateRel, AggregationPhase, Expression, FetchRel, FilterRel, JoinRel,
    ProjectRel, ReadRel, Rel, RelCommon, SortField, SortRel, aggregate_rel, expression, fetch_rel,
    function_argument, join_rel, rel, rel_common, sort_field,
};

use crate::agg_phase::{self, AggPhase};
use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, CarriedSlot, ExprContext, TranslateExpr};
use crate::partial_state::{self, WireColumn};
use crate::scan_paths::ScanFilePaths;
use crate::type_mapper;
use crate::{
    ExchangeInput, ExtensionRegistry, StreamInputColumn, StreamInputSchema, URN_AGGREGATE,
    URN_ARITHMETIC, URN_BOOLEAN, URN_COMPARISON,
};

/// Partially translated relation plus the StarRocks row layout it emits.
pub(crate) struct TranslatedRel {
    /// Substrait relation built for the StarRocks subtree.
    pub rel: Rel,
    /// Tuple ids visible in the relation output row.
    ///
    /// For multi-input relations (joins, set ops) this is the left-to-right
    /// concatenation of the child layouts, so `DescriptorTable::slot_global_index`
    /// resolves a right-side slot to `left_width + right_index`. New multi-input
    /// translators MUST follow this ordering.
    pub row_tuples: Vec<i32>,
    /// Number of columns this relation emits.
    ///
    /// Carried as an invariant (rather than recomputed by walking `rel`) so a
    /// parent projection can compute its `output_mapping` base offset without
    /// teaching a column-counting helper about every relation type — a pattern
    /// that silently produced a wrong offset the moment an unknown relation
    /// appeared mid-tree. Every relation built here MUST set its true width.
    pub output_width: usize,
    /// Common-expr columns carried past the descriptor row (see [`translate_project_node`]).
    ///
    /// Invariant: columns `[0, output_width - carried_slots.len())` are exactly the
    /// concatenation of `materialized_slot_ids(t)` for `t` in `row_tuples`; the carried
    /// columns occupy the tail `[output_width - carried_slots.len(), output_width)`.
    /// Deliberately no `Default` impl: every construction site must decide whether carried
    /// columns survive it. Width-preserving nodes pass them through, re-materializing nodes
    /// drop them by construction, joins refuse them, and the fragment root narrows them away.
    pub carried_slots: Vec<CarriedSlot>,
}

/// Mutable state shared by plan-node translators.
struct PlanContext<'a> {
    /// Descriptor lookups for row layouts, tables, and scan schemas.
    desc: &'a DescriptorTable,
    /// Parquet file paths for each scan node, collected from the fragment's broker
    /// scan ranges. Scans with paths emit a `local_files` read; path-less scans
    /// fall back to a named-table read.
    scan_paths: &'a ScanFilePaths,
    /// Input streams keyed by receiver exchange node id.
    exchange_inputs: &'a HashMap<i32, &'a ExchangeInput>,
    /// Substrait extension registry shared across the whole plan.
    registry: &'a mut ExtensionRegistry,
    /// Schema of every exchange lowered to a stream read, in translation order. The caller has to
    /// declare these on the engine before the plan can bind.
    stream_inputs: Vec<StreamInputSchema>,
    /// Positional wire-column overrides, keyed by exchange node id, for exchanges that feed
    /// merge aggregations: the FE declares those columns with intermediate slot types that lie
    /// about what the sender ships, and allocates one slot for a state that can be wider (see
    /// `partial_state`). Computed by [`merge_exchange_overrides`] before translation starts.
    exchange_state_overrides: HashMap<i32, Vec<StateColumns>>,
    /// Set when a partial aggregation emitted more columns than its FE output tuple declares.
    /// The fragment's root output names come from here instead of the descriptor table.
    partial_expansion: Option<PartialExpansion>,
    /// Common-expr slots each project must physically carry because an ancestor consumes them,
    /// keyed by project node id with slot ids ascending. Computed by
    /// [`common_slots_consumed_above`] before translation starts.
    consumed_above: HashMap<i32, Vec<i32>>,
}

/// The wire columns one merge measure's partial state occupies on the exchange row.
///
/// `position` indexes the row the *FE* declared (one column per intermediate slot); the extra
/// types of a wider state are inserted behind it, which is where the partial fragment emitted
/// them.
struct StateColumns {
    /// Index of the measure's own column in the FE-declared exchange row.
    position: usize,
    /// Modeled wire types: the first replaces the FE slot type, the rest are inserted after it.
    types: Vec<substrait::proto::Type>,
}

/// A partial aggregation that emits more columns than its FE output tuple declares, because an
/// avg measure expanded into a sum and a count.
///
/// Every consumer of the fragment's output row layout has to be told: the FE tuple no longer
/// describes what the sender ships, so deriving names or column indices from it would mislabel
/// (or misread) every column at or after the avg.
pub(crate) struct PartialExpansion {
    /// The aggregation node that expanded.
    node_id: i32,
    /// Grouping-key count. Keys keep their FE column index; measures may not.
    pub(crate) keys: usize,
    /// Root output names for the emitted row: the FE's names with the extras inserted.
    pub(crate) names: Vec<String>,
}

/// One translated fragment: its relation tree plus everything the caller has to declare or
/// name because the FE's descriptor table no longer describes it alone.
pub(crate) struct TranslatedFragment {
    /// Root relation and its row layout.
    pub root: TranslatedRel,
    /// Schema of every exchange lowered to a stream read, in translation order.
    pub stream_inputs: Vec<StreamInputSchema>,
    /// Set when a partial aggregation widened the output row (see [`PartialExpansion`]).
    pub partial_expansion: Option<PartialExpansion>,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    fn new(
        desc: &'a DescriptorTable,
        scan_paths: &'a ScanFilePaths,
        exchange_inputs: &'a HashMap<i32, &'a ExchangeInput>,
        registry: &'a mut ExtensionRegistry,
    ) -> Self {
        Self {
            desc,
            scan_paths,
            exchange_inputs,
            registry,
            stream_inputs: Vec::new(),
            exchange_state_overrides: HashMap::new(),
            partial_expansion: None,
            consumed_above: HashMap::new(),
        }
    }

    /// Creates an expression context for an expression over `row_tuples`.
    fn expr_context<'b>(&'b mut self, row_tuples: &'b [i32]) -> ExprContext<'b> {
        ExprContext::new(self.desc, self.registry, row_tuples)
    }

    /// Creates an expression context with synthetic slot-to-column mappings.
    fn expr_context_with_slots<'b>(
        &'b mut self,
        row_tuples: &'b [i32],
        slot_overrides: &'b std::collections::HashMap<(i32, i32), usize>,
    ) -> ExprContext<'b> {
        ExprContext::with_slot_overrides(self.desc, self.registry, row_tuples, slot_overrides)
    }
}

/// Trait implemented by StarRocks plan objects that can become Substrait relations.
trait TranslatePlan {
    /// Translates the receiver into a Substrait relation plus row layout.
    fn translate(&self, ctx: &mut PlanContext<'_>) -> Result<TranslatedRel>;
}

impl TranslatePlan for TPlan {
    /// Translates a flat preorder StarRocks plan into a Substrait relation tree.
    fn translate(&self, ctx: &mut PlanContext<'_>) -> Result<TranslatedRel> {
        if self.nodes.is_empty() {
            return Err(TranslateError::malformed("TPlan.nodes is empty"));
        }
        let mut cursor = PlanNodeCursor::new(&self.nodes);
        let translated = cursor.translate_next(ctx)?;
        cursor.ensure_consumed()?;
        Ok(translated)
    }
}

/// Cursor over StarRocks' flat preorder `TPlan.nodes` representation.
struct PlanNodeCursor<'a> {
    /// Node slice being parsed.
    nodes: &'a [TPlanNode],
    /// Next node index to read.
    idx: usize,
}

impl<'a> PlanNodeCursor<'a> {
    /// Creates a cursor at the start of a plan node list.
    fn new(nodes: &'a [TPlanNode]) -> Self {
        Self { nodes, idx: 0 }
    }

    /// Translates the next preorder node and its subtree.
    fn translate_next(&mut self, ctx: &mut PlanContext<'_>) -> Result<TranslatedRel> {
        let node = self
            .nodes
            .get(self.idx)
            .ok_or_else(|| TranslateError::malformed("unexpected end of plan nodes"))?;
        self.idx += 1;

        if node.num_children < 0 {
            return Err(TranslateError::malformed(format!(
                "node {} has negative child count {}",
                node.node_id, node.num_children
            )));
        }

        let children = (0..node.num_children)
            .map(|_| self.translate_next(ctx))
            .collect::<Result<Vec<_>>>()?;

        translate_plan_node(node, children, ctx)
    }

    /// Verifies that the top-level plan consumed all encoded nodes.
    fn ensure_consumed(&self) -> Result<()> {
        if self.idx != self.nodes.len() {
            return Err(TranslateError::malformed(format!(
                "TPlan had {} trailing node(s)",
                self.nodes.len() - self.idx
            )));
        }
        Ok(())
    }
}

/// Routes a StarRocks plan node to its supported v1 translator once its children
/// have been translated.
fn translate_plan_node(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let translated = match node.node_type {
        TPlanNodeType::FILE_SCAN_NODE => translate_file_scan(node, children, ctx),
        TPlanNodeType::HDFS_SCAN_NODE => translate_hdfs_scan(node, children, ctx),
        TPlanNodeType::SELECT_NODE => translate_select(node, children, ctx),
        TPlanNodeType::PROJECT_NODE => translate_project(node, children, ctx),
        TPlanNodeType::AGGREGATION_NODE => translate_aggregation(node, children, ctx),
        TPlanNodeType::SORT_NODE => translate_sort(node, children, ctx),
        TPlanNodeType::HASH_JOIN_NODE => translate_hash_join(node, children, ctx),
        TPlanNodeType::NESTLOOP_JOIN_NODE => translate_nestloop_join(node, children, ctx),
        TPlanNodeType::EXCHANGE_NODE => translate_exchange(node, children, ctx),
        _ => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "plan node is outside the v1 StarRocks slice",
        }),
    }?;
    Ok(apply_fetch(translated, node))
}

/// Wraps a relation in a Substrait fetch when the StarRocks node carries a limit or offset.
///
/// `TPlanNode::limit` applies to any node type; a skip offset only appears on sort and exchange
/// payloads.
// The deprecated plain offset/count oneof variants share wire tags with their expression
// counterparts and are the fields DuckDB's Substrait consumer reads.
#[allow(deprecated)]
fn apply_fetch(input: TranslatedRel, node: &TPlanNode) -> TranslatedRel {
    let offset = node
        .sort_node
        .as_ref()
        .and_then(|sort| sort.offset)
        .or_else(|| {
            node.exchange_node
                .as_ref()
                .and_then(|exchange| exchange.offset)
        })
        .unwrap_or(0);
    if node.limit < 0 && offset == 0 {
        return input;
    }
    let TranslatedRel {
        rel,
        row_tuples,
        output_width,
        carried_slots,
    } = input;
    // For an offset-only fetch, emit an explicit unlimited count: the consumer reads the plain
    // count field without checking the oneof, and an unset count would decode as `LIMIT 0`.
    let count = if node.limit >= 0 { node.limit } else { -1 };
    let count_mode = Some(fetch_rel::CountMode::Count(count));
    let offset_mode = (offset != 0).then_some(fetch_rel::OffsetMode::Offset(offset));
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Fetch(Box::new(FetchRel {
                input: Some(Box::new(rel)),
                offset_mode,
                count_mode,
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        // A fetch keeps the row layout, carried columns included.
        carried_slots,
    }
}

/// Builds a named-table read for `FILE_SCAN_NODE`.
fn translate_file_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    // `file_scan_node.tuple_id` is required, so a present payload always resolves.
    let tuple_id = node
        .file_scan_node
        .as_ref()
        .map(|scan| scan.tuple_id)
        .or_else(|| node.row_tuples.first().copied())
        .ok_or(TranslateError::MissingField {
            context: "FILE_SCAN_NODE",
            field: "tuple_id",
        })?;
    translate_scan(node, children, tuple_id, ctx)
}

/// Builds a named-table read for `HDFS_SCAN_NODE`.
fn translate_hdfs_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    // `hdfs_scan_node.tuple_id` is optional, so fall back to the node row layout.
    let tuple_id = node
        .hdfs_scan_node
        .as_ref()
        .and_then(|scan| scan.tuple_id)
        .or_else(|| node.row_tuples.first().copied())
        .ok_or(TranslateError::MissingField {
            context: "HDFS_SCAN_NODE",
            field: "tuple_id",
        })?;
    translate_scan(node, children, tuple_id, ctx)
}

/// Builds a leaf named-table read for `tuple_id` and applies any filter conjuncts.
///
/// Shared by every scan node; new scan types (OLAP/connector/lake) only need to
/// resolve their `tuple_id` and delegate here.
fn translate_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    tuple_id: i32,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 0)?;
    let file_paths = ctx.scan_paths.for_node(node.node_id);
    let input = TranslatedRel {
        rel: scan_rel(ctx.desc, tuple_id, file_paths)?,
        row_tuples: vec![tuple_id],
        output_width: ctx.desc.materialized_slot_ids(tuple_id)?.len(),
        carried_slots: Vec::new(),
    };
    apply_conjuncts(input, node, ctx)
}

/// Refuses a node whose `common_slot_map` this translator does not materialize.
///
/// Only `PROJECT_NODE` appends its common slots. On every other node carrying the field the
/// shared sub-expressions would be read past: a conjunct or key that references one of them then
/// fails later with an opaque descriptor error (`slot N (tuple T) is not part of row_tuples`),
/// and a map nothing references is silently ignored. Report the unsupported shape up front.
fn reject_common_slots(
    node: &TPlanNode,
    common_slot_map: Option<&BTreeMap<TSlotId, TExpr>>,
) -> Result<()> {
    if common_slot_map.is_some_and(|map| !map.is_empty()) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "common slots are only materialized on PROJECT_NODE",
        });
    }
    Ok(())
}

/// Wraps the child relation of a `SELECT_NODE` with its filter conjuncts.
fn translate_select(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
    reject_common_slots(
        node,
        node.select_node
            .as_ref()
            .and_then(|select| select.common_slot_map.as_ref()),
    )?;
    apply_conjuncts(children.into_iter().next().unwrap(), node, ctx)
}

/// Builds a project relation for a `PROJECT_NODE` with no ambiguous conjuncts.
fn translate_project(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
    if has_conjuncts(node) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "PROJECT_NODE conjunct row layout is ambiguous in v1",
        });
    }
    let child = children.into_iter().next().unwrap();
    translate_project_node(child, node, ctx)
}

/// Translates a flat preorder StarRocks plan into a Substrait relation tree.
///
/// `root_output_exprs` are the fragment's output expressions (empty when absent); they count
/// as consumers sitting above the fragment root for [`common_slots_consumed_above`].
pub(crate) fn translate_plan(
    plan: &TPlan,
    root_output_exprs: &[TExpr],
    desc: &DescriptorTable,
    scan_paths: &ScanFilePaths,
    exchange_inputs: &HashMap<i32, &ExchangeInput>,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedFragment> {
    let mut ctx = PlanContext::new(desc, scan_paths, exchange_inputs, registry);
    ctx.exchange_state_overrides = merge_exchange_overrides(plan)?;
    ctx.consumed_above = common_slots_consumed_above(plan, root_output_exprs, desc)?;
    let root = plan.translate(&mut ctx)?;
    if let Some(expansion) = &ctx.partial_expansion {
        // Only the fragment root's row leaves this CN by name. Any node above the expanded
        // aggregation would resolve the aggregation's slots at their FE positions, which the
        // expansion moved -- a wrong column read, not a type error the engine would catch.
        if plan.nodes.first().map(|node| node.node_id) != Some(expansion.node_id) {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: expansion.node_id,
                node_type: TPlanNodeType::AGGREGATION_NODE,
                reason: "a partial aggregation over avg must be the fragment root; a node above \
                         it reads the aggregate's columns at the positions the avg state moved \
                         (SET new_planner_agg_stage = 1)",
            });
        }
        if root.output_width != expansion.names.len() {
            return Err(TranslateError::malformed(format!(
                "partial aggregation emits {} columns but named {} of them",
                root.output_width,
                expansion.names.len()
            )));
        }
    }
    Ok(TranslatedFragment {
        root,
        stream_inputs: std::mem::take(&mut ctx.stream_inputs),
        partial_expansion: ctx.partial_expansion,
    })
}

/// Computes the positional wire-column overrides for exchanges that feed merge aggregations.
///
/// Runs over the flat preorder node list before translation, because the exchange is
/// translated before its parent aggregation and must already declare the stream with the
/// modeled partial-state columns — their types, and the extra column a wider state (avg)
/// occupies. In preorder a merge aggregation's single child is simply the next node; anything
/// but an exchange there means the plan reads partial states from a shape this translator
/// cannot type, so it is refused.
fn merge_exchange_overrides(plan: &TPlan) -> Result<HashMap<i32, Vec<StateColumns>>> {
    let mut overrides = HashMap::new();
    for (index, node) in plan.nodes.iter().enumerate() {
        if node.node_type != TPlanNodeType::AGGREGATION_NODE {
            continue;
        }
        // A node without agg_node fails in translate_aggregation with its own error.
        let Some(agg) = node.agg_node.as_ref() else {
            continue;
        };
        if agg_phase::classify(node.node_id, node.node_type, agg)? != AggPhase::Merge {
            continue;
        }
        let child = plan
            .nodes
            .get(index + 1)
            .filter(|child| child.node_type == TPlanNodeType::EXCHANGE_NODE);
        let Some(child) = child else {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "a merge aggregation must read its partial states directly from an \
                         exchange (SET new_planner_agg_stage = 1)",
            });
        };
        let keys = agg.grouping_exprs.as_deref().unwrap_or_default().len();
        let mut columns = Vec::with_capacity(agg.aggregate_functions.len());
        for (position, measure) in agg.aggregate_functions.iter().enumerate() {
            // The exchange row is the intermediate tuple's materialized slots: grouping keys
            // first, then one state slot per measure -- the same layout invariant the
            // aggregation's output tuple uses. A state wider than its slot (avg) claims the
            // columns behind it, which is where the partial fragment emitted them.
            columns.push(StateColumns {
                position: keys + position,
                types: partial_state::wire_columns(measure_function(measure)?)?
                    .into_iter()
                    .map(|column| column.ty)
                    .collect(),
            });
        }
        overrides.insert(child.node_id, columns);
    }
    Ok(overrides)
}

/// Finds, for every project node, the common-expr slots an ancestor consumes even though no
/// output tuple materializes them.
///
/// StarRocks' BE physically outputs a project's common slots when a node above references
/// them, and the FE plans against that; this pre-pass reproduces the FE's view so
/// [`translate_project_node`] can widen the project with the consumed columns. One walk over
/// the flat preorder node list, keeping the open ancestors on a stack (the same span math
/// `PlanNodeCursor` uses): when a `PROJECT_NODE` with a non-empty `common_slot_map` is
/// visited, every expression payload of its strict ancestors, plus `root_output_exprs`,
/// which sit above the fragment root, is scanned for references to its non-materialized
/// common slots. Matching is by slot id alone: new-optimizer slot ids are query-global and an
/// ancestor ref can carry a stale tuple id (see `DescriptorTable::slot_global_index`). A
/// false positive only widens the row with a column the root confinement drops; a false
/// negative leaves today's loud zero-candidate descriptor error. Neither is silent.
fn common_slots_consumed_above(
    plan: &TPlan,
    root_output_exprs: &[TExpr],
    desc: &DescriptorTable,
) -> Result<HashMap<i32, Vec<i32>>> {
    let mut consumed = HashMap::new();
    // Open ancestors of the node being visited: (node index, children not yet completed).
    let mut stack: Vec<(usize, i32)> = Vec::new();
    for (index, node) in plan.nodes.iter().enumerate() {
        if node.node_type == TPlanNodeType::PROJECT_NODE
            && let Some(common) = node
                .project_node
                .as_ref()
                .and_then(|project| project.common_slot_map.as_ref())
                .filter(|common| !common.is_empty())
        {
            let mut candidates = BTreeSet::new();
            for &slot_id in common.keys() {
                let mut materialized = false;
                for &tuple_id in &node.row_tuples {
                    if desc.materialized_slot_ids(tuple_id)?.contains(&slot_id) {
                        materialized = true;
                        break;
                    }
                }
                if !materialized {
                    candidates.insert(slot_id);
                }
            }
            if !candidates.is_empty() {
                let mut hits = BTreeSet::new();
                for &(ancestor, _) in &stack {
                    collect_slot_ref_hits(&plan.nodes[ancestor], &candidates, &mut hits);
                }
                for expr in root_output_exprs {
                    collect_expr_slot_ref_hits(expr, &candidates, &mut hits);
                }
                if !hits.is_empty() {
                    consumed.insert(node.node_id, hits.into_iter().collect());
                }
            }
        }
        if node.num_children > 0 {
            stack.push((index, node.num_children));
        } else {
            // A leaf completes itself, and possibly the subtrees of the ancestors above it.
            while let Some(top) = stack.last_mut() {
                top.1 -= 1;
                if top.1 == 0 {
                    stack.pop();
                } else {
                    break;
                }
            }
        }
    }
    Ok(consumed)
}

/// Adds to `hits` every candidate slot id referenced by any expression payload of `node`.
fn collect_slot_ref_hits(node: &TPlanNode, candidates: &BTreeSet<i32>, hits: &mut BTreeSet<i32>) {
    let mut exprs: Vec<&TExpr> = Vec::new();
    exprs.extend(node.conjuncts.as_deref().unwrap_or_default());
    if let Some(agg) = node.agg_node.as_ref() {
        exprs.extend(agg.grouping_exprs.as_deref().unwrap_or_default());
        exprs.extend(&agg.aggregate_functions);
    }
    if let Some(sort) = node.sort_node.as_ref() {
        exprs.extend(&sort.sort_info.ordering_exprs);
        exprs.extend(
            sort.sort_info
                .sort_tuple_slot_exprs
                .as_deref()
                .unwrap_or_default(),
        );
        // Deprecated node-level duplicate some senders populate instead (see translate_sort).
        exprs.extend(sort.sort_tuple_slot_exprs.as_deref().unwrap_or_default());
    }
    if let Some(join) = node.hash_join_node.as_ref() {
        for eq in &join.eq_join_conjuncts {
            exprs.push(&eq.left);
            exprs.push(&eq.right);
        }
        exprs.extend(join.other_join_conjuncts.as_deref().unwrap_or_default());
    }
    if let Some(join) = node.nestloop_join_node.as_ref() {
        exprs.extend(join.join_conjuncts.as_deref().unwrap_or_default());
    }
    if let Some(project) = node.project_node.as_ref() {
        exprs.extend(
            project
                .slot_map
                .as_ref()
                .into_iter()
                .flatten()
                .map(|(_, expr)| expr),
        );
        exprs.extend(
            project
                .common_slot_map
                .as_ref()
                .into_iter()
                .flatten()
                .map(|(_, expr)| expr),
        );
    }
    for expr in exprs {
        collect_expr_slot_ref_hits(expr, candidates, hits);
    }
}

/// Adds to `hits` every candidate slot id one flat preorder expression references.
fn collect_expr_slot_ref_hits(expr: &TExpr, candidates: &BTreeSet<i32>, hits: &mut BTreeSet<i32>) {
    for node in &expr.nodes {
        if node.node_type == starrocks_thrift::exprs::TExprNodeType::SLOT_REF
            && let Some(slot_ref) = node.slot_ref.as_ref()
            && candidates.contains(&slot_ref.slot_id)
        {
            hits.insert(slot_ref.slot_id);
        }
    }
}

/// Translates an `EXCHANGE_NODE` into a read of the engine stream its senders' batches arrive on.
///
/// An exchange is a fragment boundary: the receiver has nothing of its own to read, so the
/// compute node binds one [`ExchangeInput`] per exchange node, naming the engine view
/// (`sirius_stream_<node_id>`) the input stream is read through. The read's schema comes from
/// the FE's `input_row_tuples`; its column names are the sender's, bound positionally, because
/// the receiver's tuple can spell them differently from what the sender ships. The same schema
/// is recorded on the context as the declaration the engine needs before the plan can bind.
///
/// A merging exchange (`sort_info` present) merges its senders' sorted runs; here it becomes a
/// `SortRel` over the stream read, which yields the same rows in the same order.
fn translate_exchange(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 0)?;
    let exchange = node
        .exchange_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "EXCHANGE_NODE",
            field: "exchange_node",
        })?;
    if exchange.input_row_tuples.is_empty() {
        return Err(TranslateError::MissingField {
            context: "TExchangeNode",
            field: "input_row_tuples",
        });
    }
    let input =
        ctx.exchange_inputs
            .get(&node.node_id)
            .ok_or(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "exchange node has no bound input stream; the compute node binds one per \
                         receiver exchange",
            })?;
    // The view name is the only thing tying this read to the stream the engine fills. An empty
    // one would emit a read of a table named "" and fail at bind time with no mention of the
    // exchange.
    if input.stream_view.is_empty() {
        return Err(TranslateError::malformed(format!(
            "exchange node {} is bound to an input stream with an empty view name",
            node.node_id
        )));
    }
    let mut schema = ctx
        .desc
        .named_struct_for_tuples(&exchange.input_row_tuples)?;
    // An exchange feeding a merge aggregation carries partial-state columns; rewrite their
    // FE-declared slot types to the modeled wire types, and make room for any state the FE
    // declares in fewer columns than Sirius ships. One `Type`, two consumers: the ReadRel
    // base_schema below and the engine's stream declaration derive from the same rewritten
    // entry, so the plan's view of the column and the engine's cannot drift apart.
    if let Some(state_overrides) = ctx.exchange_state_overrides.get(&node.node_id) {
        let types = schema
            .r#struct
            .as_mut()
            .map(|structure| &mut structure.types)
            .ok_or_else(|| TranslateError::malformed("exchange schema has no struct"))?;
        let width = types.len();
        // Positions index the FE-declared row, so every insertion shifts the ones after it.
        let mut inserted = 0usize;
        for column in state_overrides {
            let position = column.position + inserted;
            let slot = types.get_mut(position).ok_or_else(|| {
                TranslateError::malformed(format!(
                    "merge aggregation state column {} is outside the exchange row \
                     ({width} columns)",
                    column.position
                ))
            })?;
            *slot = column.types[0].clone();
            for (offset, ty) in column.types.iter().enumerate().skip(1) {
                types.insert(position + offset, ty.clone());
            }
            inserted += column.types.len() - 1;
        }
    }
    let output_width = schema
        .r#struct
        .as_ref()
        .map(|structure| structure.types.len())
        .unwrap_or(0);
    // The stream's columns are the sender's, named by the sender and bound by position: one
    // name per column of the declared row layout, or a column would be read under the wrong
    // name.
    if input.names.len() != output_width {
        return Err(TranslateError::descriptor(format!(
            "row layout {:?} has {} fields but exchange input has {} names",
            exchange.input_row_tuples,
            output_width,
            input.names.len()
        )));
    }
    schema.names = input.names.clone();

    // The declaration the engine needs, derived from the very types the read carries so the
    // plan's view of a column and the engine's cannot drift apart.
    let columns = schema
        .names
        .iter()
        .cloned()
        .zip(
            schema
                .r#struct
                .as_ref()
                .map(|structure| structure.types.as_slice())
                .unwrap_or_default(),
        )
        .map(|(name, ty)| {
            Ok(StreamInputColumn {
                name,
                ty: type_mapper::duckdb_type_name(ty)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ctx.stream_inputs.push(StreamInputSchema {
        node_id: node.node_id,
        stream_view: input.stream_view.clone(),
        columns,
    });

    let mut translated = TranslatedRel {
        rel: stream_read_rel(schema, &input.stream_view),
        row_tuples: exchange.input_row_tuples.clone(),
        output_width,
        // Streams never carry: the sender's root narrowed any carried columns away, and the
        // declared stream schema describes exactly the descriptor row.
        carried_slots: Vec::new(),
    };
    if let Some(sort_info) = &exchange.sort_info {
        let sorts = sort_fields(sort_info, &translated, ctx)?;
        let row_tuples = translated.row_tuples.clone();
        translated = TranslatedRel {
            rel: Rel {
                rel_type: Some(rel::RelType::Sort(Box::new(SortRel {
                    input: Some(Box::new(translated.rel)),
                    sorts,
                    ..Default::default()
                }))),
            },
            row_tuples,
            output_width,
            carried_slots: Vec::new(),
        };
    }
    apply_conjuncts(translated, node, ctx)
}

/// Translates an `AGGREGATION_NODE` into a Substrait aggregate relation.
///
/// The node's phase (one-shot / partial / merge, see [`agg_phase::classify`]) decides how the
/// measures are emitted. The output row layout is the aggregation output tuple, whose
/// materialized slots are the grouping keys followed by the aggregate results (StarRocks
/// allocates them in that order).
///
/// The grouping keys are emitted in the output tuple's materialized-slot order, not GROUP BY
/// order (see [`grouping_materialization_order`]): every consumer of the tuple — slot refs
/// above this node, the next hop's declared stream schema, output names, hash-partition
/// indices — resolves its columns through the descriptor's order, so the sender reorders to
/// it, exactly as [`translate_sort`] does for its sort tuple.
///
/// Two-phase avg is the one aggregate that does not fit that layout one slot at a time: it is
/// emitted as a sum and a count on the partial side, merged as two sums, and divided back into
/// an average by a projection on top on the merge side. Both sides take the extra column from
/// the same [`partial_state::wire_columns`] model that types the exchange, so the two ends of
/// the hop agree by construction.
fn translate_aggregation(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
    let child = children.into_iter().next().unwrap();
    let agg = node.agg_node.as_ref().ok_or(TranslateError::MissingField {
        context: "AGGREGATION_NODE",
        field: "agg_node",
    })?;
    let phase = agg_phase::classify(node.node_id, node.node_type, agg)?;
    // Never observed in new-optimizer plans (the FE sets the two ids equal in every phase);
    // kept as its own loud error so a plan shape that does split them cannot slip through the
    // slot-layout assumptions below.
    if agg.intermediate_tuple_id != agg.output_tuple_id {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "aggregation node has distinct intermediate and output tuples \
                     (SET new_planner_agg_stage = 1)",
        });
    }
    let output_tuple = agg.output_tuple_id;
    let grouping_exprs = agg.grouping_exprs.as_deref().unwrap_or_default();
    let keys = grouping_exprs.len();

    // Aggregate output types come from the output tuple's slots, which carry the grouping keys
    // first and then one slot per aggregate function.
    let output_slots = ctx.desc.materialized_slot_ids(output_tuple)?;
    if output_slots.len() != keys + agg.aggregate_functions.len() {
        return Err(TranslateError::descriptor(format!(
            "AGGREGATION_NODE {} output tuple {} has {} slots for {} keys + {} aggregates",
            node.node_id,
            output_tuple,
            output_slots.len(),
            keys,
            agg.aggregate_functions.len()
        )));
    }

    // The FE lists the grouping expressions in GROUP BY order, but the output tuple's
    // materialized slots -- the order every consumer above this node resolves the row
    // through -- are sorted by slot id (the FE serializes every aggregation-output slot with
    // column_pos -1). Emitting the keys in GROUP BY order would make the engine row and the
    // descriptor view permutations of each other whenever the GROUP BY list is not in
    // ascending ref-id order, so the keys are emitted in the tuple's order instead. One key
    // needs no pairing: a single grouping expression can only fill the single key slot.
    let key_order: Vec<usize> = if keys > 1 {
        grouping_materialization_order(node, grouping_exprs, &output_slots[..keys])?
    } else {
        (0..keys).collect()
    };
    // The slot pairing above cannot see a mistyped key (nor does it run for a single key), so
    // also require each grouping key's type to match the slot it is paired with. Compare only
    // the type kind: the slot's nullability and decimal width are allowed to differ from the
    // key expression's.
    for (slot_index, &grouping_index) in key_order.iter().enumerate() {
        let expr = &grouping_exprs[grouping_index];
        let slot_id = output_slots[slot_index];
        let Some(key_type) = expr
            .nodes
            .first()
            .map(|node| type_mapper::map_type_desc(&node.type_, true))
            .transpose()?
        else {
            continue;
        };
        let slot = ctx.desc.slot(output_tuple, slot_id)?;
        let Some(slot_type) = slot.substrait_type.as_ref() else {
            continue;
        };
        let kind_of = |ty: &substrait::proto::Type| ty.kind.as_ref().map(std::mem::discriminant);
        if kind_of(&key_type) != kind_of(slot_type) {
            return Err(TranslateError::descriptor(format!(
                "AGGREGATION_NODE {} output tuple {} slot {} does not match grouping key {}",
                node.node_id, output_tuple, slot_id, grouping_index
            )));
        }
    }

    // A two-phase measure occupies its modeled partial-state columns rather than the single
    // slot the FE allocated for it; one-shot measures always occupy exactly their own slot,
    // which is why the model is not consulted for them at all.
    let wire_columns = match phase {
        AggPhase::OneShot => Vec::new(),
        _ => agg
            .aggregate_functions
            .iter()
            .map(|expr| partial_state::wire_columns(measure_function(expr)?))
            .collect::<Result<Vec<_>>>()?,
    };
    let expanded = wire_columns.iter().any(|columns| columns.len() > 1);
    // A merge measure reads the exchange row, which is wider than the FE's intermediate tuple
    // wherever a state expanded, so every slot's column is resolved explicitly instead of
    // through the descriptor's one-slot-per-column view.
    let merge_slots = match phase {
        AggPhase::Merge => Some(merge_state_columns(
            node,
            &child.row_tuples,
            keys,
            &wire_columns,
            ctx.desc,
        )?),
        _ => None,
    };
    // Unreachable today (a merge child must be an exchange, see merge_exchange_overrides, and
    // streams never carry); kept as a guard so the two column-remapping mechanisms can never
    // silently combine.
    if merge_slots.is_some() && !child.carried_slots.is_empty() {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "a merge aggregation cannot read partial states from a child that carries \
                     common-expr columns",
        });
    }

    let mut grouping_expressions = Vec::with_capacity(keys);
    for &grouping_index in &key_order {
        let mut expr_ctx = match &merge_slots {
            Some(slots) => ctx.expr_context_with_slots(&child.row_tuples, &slots.columns),
            None => ctx.expr_context(&child.row_tuples),
        }
        .with_carried(&child.carried_slots);
        grouping_expressions.push(grouping_exprs[grouping_index].translate(&mut expr_ctx)?);
    }

    let mut measures = Vec::with_capacity(agg.aggregate_functions.len());
    // Where each StarRocks measure's own state starts among the emitted measures.
    let mut measure_starts = Vec::with_capacity(agg.aggregate_functions.len());
    for (index, expr) in agg.aggregate_functions.iter().enumerate() {
        measure_starts.push(measures.len());
        let call = {
            let mut expr_ctx = match &merge_slots {
                Some(slots) => ctx.expr_context_with_slots(&child.row_tuples, &slots.columns),
                None => ctx.expr_context(&child.row_tuples),
            }
            .with_carried(&child.carried_slots);
            expr_translator::aggregate_call(expr, &mut expr_ctx, phase == AggPhase::Merge)?
        };
        // The GPU ungrouped-aggregate operator rejects every distinct aggregate, so a
        // grouping-free DISTINCT measure would translate fine and then fail at execution.
        if call.distinct && grouping_expressions.is_empty() {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "distinct aggregates without grouping keys are not supported",
            });
        }
        if phase != AggPhase::OneShot && call.distinct {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "DISTINCT aggregates are not supported in two-phase plans \
                         (SET new_planner_agg_stage = 1)",
            });
        }
        if let Some(state) = wire_columns.get(index).filter(|columns| columns.len() > 1) {
            let state_column = merge_slots.as_ref().map(|slots| slots.measures[index]);
            expand_avg(node, phase, &call, state, state_column, &mut measures, ctx)?;
            continue;
        }
        // Merge functions over partial states -- the engine's own internal merge table:
        // sum->sum, min->min, max->max, and count->SUM. Merging counts must sum the partial
        // counts; merging them with count would count rows and be silently wrong. This
        // substitution is the mechanism that makes the plan phase-correct for the engine,
        // which executes exactly what the function names say.
        let function_name = if phase == AggPhase::Merge {
            match call.name.as_str() {
                "sum" | "min" | "max" => call.name.clone(),
                "count" => "sum".to_string(),
                _ => {
                    return Err(TranslateError::UnsupportedPlanNode {
                        node_id: node.node_id,
                        node_type: node.node_type,
                        reason: "merging this aggregate function is not supported \
                                 (SET new_planner_agg_stage = 1)",
                    });
                }
            }
        } else {
            call.name.clone()
        };
        // A partial measure's output is its partial state, whose wire type is modeled: the
        // FE's intermediate slot type lies about what Sirius emits (see partial_state). The
        // engine ignores measure output types either way; carrying the modeled type keeps
        // dumped plans honest about what is actually on the wire.
        let output_type = match phase {
            AggPhase::Partial => wire_columns[index][0].ty.clone(),
            _ => ctx
                .desc
                .slot(output_tuple, output_slots[keys + index])?
                .substrait_type
                .clone()
                .ok_or(TranslateError::MissingField {
                    context: "aggregate output slot",
                    field: "slotType",
                })?,
        };
        measures.push(build_measure(
            &function_name,
            call.arguments,
            output_type,
            call.distinct,
            phase,
            ctx,
        ));
    }

    let groupings = if grouping_expressions.is_empty() {
        Vec::new()
    } else {
        #[allow(deprecated)]
        let grouping = aggregate_rel::Grouping {
            grouping_expressions: Vec::new(),
            expression_references: (0..grouping_expressions.len() as u32).collect(),
        };
        vec![grouping]
    };

    let output_width = keys + measures.len();
    let aggregated = TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(child.rel)),
                groupings,
                measures,
                grouping_expressions,
                ..Default::default()
            }))),
        },
        row_tuples: vec![output_tuple],
        output_width,
        // The child's carried columns were consumed by the key/measure contexts above; the
        // aggregate materializes a fresh tuple, so nothing is carried past it.
        carried_slots: Vec::new(),
    };

    let aggregated = match phase {
        // Every merge node leaves through the finalizing projection, not only the avg-expanded
        // ones: the engine binds a merged integer count/sum as HUGEINT (the plan-level downcast
        // relabels the aggregate node, not a fragment sink above it), so without the projection's
        // throwing casts the fragment's wire row carries a type its FE-declared slot never
        // announced and the next hop's schema guard refuses it. Where a state expanded, the
        // projection also folds the extra columns back to one column per FE measure -- everything
        // above this node sees the row the descriptor describes, types included.
        AggPhase::Merge => {
            let measure_types =
                declared_measure_types(ctx.desc, output_tuple, &output_slots[keys..])?;
            merge_projection(
                aggregated,
                keys,
                &measure_starts,
                &wire_columns,
                &measure_types,
                ctx,
            )
        }
        // The emitted row is wider than the FE's output tuple, so the fragment's output names
        // are built here: the descriptor table has no slot to name the extra column from.
        AggPhase::Partial if expanded => {
            if has_conjuncts(node) {
                return Err(TranslateError::UnsupportedPlanNode {
                    node_id: node.node_id,
                    node_type: node.node_type,
                    reason: "conjuncts over a partial aggregation with an avg state would read \
                             the aggregate's columns at the positions the state moved \
                             (SET new_planner_agg_stage = 1)",
                });
            }
            if ctx.partial_expansion.is_some() {
                return Err(TranslateError::malformed(
                    "two partial aggregations in one fragment expanded an avg state; only the \
                     fragment root's row is named",
                ));
            }
            let names =
                expanded_output_names(ctx.desc, output_tuple, &output_slots, keys, &wire_columns)?;
            ctx.partial_expansion = Some(PartialExpansion {
                node_id: node.node_id,
                keys,
                names,
            });
            aggregated
        }
        _ => aggregated,
    };
    // Node conjuncts evaluate over the aggregation output (HAVING predicates).
    apply_conjuncts(aggregated, node, ctx)
}

/// Emits the two Sirius measures one StarRocks avg becomes in a two-phase plan.
///
/// Partial: `sum(cast(arg AS DOUBLE))` and `count(arg)`. Merge: both halves merge with `sum`,
/// reading the state columns the exchange row already carries side by side, which is what
/// `state_column` (the physical column the state starts at) points at.
fn expand_avg(
    node: &TPlanNode,
    phase: AggPhase,
    call: &expr_translator::AggregateCall,
    state: &[WireColumn],
    state_column: Option<usize>,
    measures: &mut Vec<aggregate_rel::Measure>,
    ctx: &mut PlanContext<'_>,
) -> Result<()> {
    if call.name != "avg" {
        return Err(TranslateError::malformed(format!(
            "aggregate {:?} has a multi-column partial state but no expansion",
            call.name
        )));
    }
    match phase {
        AggPhase::Partial => {
            let argument = sole_argument(node, &call.raw_arguments)?.clone();
            // The modeled sum column is FP64 for every avg input, so the cast is emitted here
            // rather than left to the decimal-only lowering in `aggregate_call`: an integer
            // avg would otherwise ship a BIGINT sum the receiver reads as a DOUBLE.
            measures.push(build_measure(
                "sum",
                vec![expr_translator::cast_to_fp64(argument.clone())],
                state[0].ty.clone(),
                false,
                phase,
                ctx,
            ));
            // Counting the uncast argument keeps the count over the values avg averages: the
            // cast changes no NULLs, and reading it here would only obscure that.
            measures.push(build_measure(
                "count",
                vec![argument],
                state[1].ty.clone(),
                false,
                phase,
                ctx,
            ));
        }
        AggPhase::Merge => {
            let position = state_column.ok_or_else(|| {
                TranslateError::malformed("merge measure without a resolved state column")
            })?;
            // The column behind the sum is this avg's count only if the measure really reads
            // its own state column; anything else means the FE laid the row out differently
            // than the expansion assumes, and the count would be some other measure's state.
            if direct_field(sole_argument(node, &call.arguments)?) != Some(position as i32) {
                return Err(TranslateError::UnsupportedPlanNode {
                    node_id: node.node_id,
                    node_type: node.node_type,
                    reason: "a merged avg must read its partial state straight from the \
                             exchange row (SET new_planner_agg_stage = 1)",
                });
            }
            // Both halves merge with sum: the division that finalizes them divides summed
            // sums by summed counts, where count here would count arriving rows instead.
            measures.push(build_measure(
                "sum",
                vec![field_selection(position as i32)],
                state[0].ty.clone(),
                false,
                phase,
                ctx,
            ));
            measures.push(build_measure(
                "sum",
                vec![field_selection(position as i32 + 1)],
                state[1].ty.clone(),
                false,
                phase,
                ctx,
            ));
        }
        AggPhase::OneShot => {
            return Err(TranslateError::malformed(
                "a one-shot aggregate has no partial state to expand",
            ));
        }
    }
    Ok(())
}

/// Builds one Substrait measure over already-translated arguments.
fn build_measure(
    name: &str,
    arguments: Vec<Expression>,
    output_type: substrait::proto::Type,
    distinct: bool,
    phase: AggPhase,
    ctx: &mut PlanContext<'_>,
) -> aggregate_rel::Measure {
    // `count` lives in the generic aggregate extension; sum/avg/min/max are declared by the
    // arithmetic extension. Keyed on the emitted name, so a merged count registers as the
    // arithmetic `sum` it became.
    let urn = if name == "count" {
        URN_AGGREGATE
    } else {
        URN_ARITHMETIC
    };
    let anchor = ctx.registry.register_function(urn, name);
    aggregate_rel::Measure {
        measure: Some(AggregateFunction {
            function_reference: anchor,
            arguments: arguments
                .into_iter()
                .map(|expr| substrait::proto::FunctionArgument {
                    arg_type: Some(function_argument::ArgType::Value(expr)),
                })
                .collect(),
            output_type: Some(output_type),
            invocation: if distinct {
                substrait::proto::aggregate_function::AggregationInvocation::Distinct as i32
            } else {
                substrait::proto::aggregate_function::AggregationInvocation::All as i32
            },
            // Advisory only: the engine's Substrait consumer ignores phases and executes
            // exactly what the function names say, so the plan must be correct for a
            // phase-ignoring reader first (substitute functions, then label). The label
            // makes dumped plans self-describing.
            phase: match phase {
                AggPhase::OneShot => AggregationPhase::InitialToResult,
                AggPhase::Partial => AggregationPhase::InitialToIntermediate,
                AggPhase::Merge => AggregationPhase::IntermediateToResult,
            } as i32,
            ..Default::default()
        }),
        filter: None,
    }
}

/// Physical columns of the intermediate row a merge aggregation reads.
struct MergeStateColumns {
    /// Physical column of each `(tuple id, slot id)`, for the expression translator.
    columns: HashMap<(i32, i32), usize>,
    /// Physical column where each measure's state starts, in FE measure order.
    measures: Vec<usize>,
}

/// Resolves every intermediate-tuple slot a merge aggregation reads to its physical column.
///
/// The FE numbers those slots as if every state were one column; a wider state (avg) shifts
/// every slot behind it, so the descriptor's index is the wrong column from that point on.
/// Grouping keys come first and never shift.
fn merge_state_columns(
    node: &TPlanNode,
    row_tuples: &[i32],
    keys: usize,
    wire_columns: &[Vec<WireColumn>],
    desc: &DescriptorTable,
) -> Result<MergeStateColumns> {
    let mut slots = Vec::new();
    for &tuple_id in row_tuples {
        for slot_id in desc.materialized_slot_ids(tuple_id)? {
            slots.push((tuple_id, slot_id));
        }
    }
    if slots.len() != keys + wire_columns.len() {
        return Err(TranslateError::descriptor(format!(
            "AGGREGATION_NODE {} merges {} keys + {} aggregates from an intermediate row \
             {row_tuples:?} of {} slots",
            node.node_id,
            keys,
            wire_columns.len(),
            slots.len()
        )));
    }
    let mut columns = HashMap::with_capacity(slots.len());
    let mut measures = Vec::with_capacity(wire_columns.len());
    let mut position = 0usize;
    for (index, slot) in slots.into_iter().enumerate() {
        columns.insert(slot, position);
        position += match index.checked_sub(keys) {
            Some(measure) => {
                measures.push(position);
                wire_columns[measure].len()
            }
            None => 1,
        };
    }
    Ok(MergeStateColumns { columns, measures })
}

/// Adds the projection that turns a merge aggregation's emitted columns back into the FE's
/// output row: the grouping keys, then one column per StarRocks measure.
///
/// Every measure column is cast to `measure_types`, the types the FE's output tuple declares,
/// which is what the next fragment derives its stream schema from. Stating them is not
/// cosmetic: DuckDB binds `sum(BIGINT)` as HUGEINT, and the engine's HUGEINT-to-BIGINT
/// downcast happens at the aggregate's own sink, which this projection now sits in front of.
/// Without the cast a merged count leaves this fragment as a HUGEINT the receiver declared
/// BIGINT, and the hop refuses it.
fn merge_projection(
    input: TranslatedRel,
    keys: usize,
    measure_starts: &[usize],
    wire_columns: &[Vec<WireColumn>],
    measure_types: &[substrait::proto::Type],
    ctx: &mut PlanContext<'_>,
) -> TranslatedRel {
    let mut expressions: Vec<Expression> = (0..keys as i32).map(field_selection).collect();
    for (index, start) in measure_starts.iter().enumerate() {
        let field = (keys + start) as i32;
        let value = if wire_columns[index].len() > 1 {
            avg_from_state(field, field + 1, ctx)
        } else {
            field_selection(field)
        };
        expressions.push(expr_translator::cast_to(
            value,
            measure_types[index].clone(),
        ));
    }
    let row_tuples = input.row_tuples.clone();
    project_rel(input, expressions, row_tuples, Vec::new())
}

/// Divides a merged avg state back into an average, with SQL's empty-input NULL.
///
/// A group whose avg argument was NULL in every row arrives with a zero count; dividing by it
/// is a division by zero where SQL defines NULL, so the zero case is answered explicitly.
fn avg_from_state(sum: i32, count: i32, ctx: &mut PlanContext<'_>) -> Expression {
    let equal_anchor = ctx.registry.register_function(URN_COMPARISON, "equal");
    let no_values = expr_translator::scalar_function(
        equal_anchor,
        vec![field_selection(count), i64_literal(0)],
        type_mapper::bool_type(),
    );
    let divide_anchor = ctx.registry.register_function(URN_ARITHMETIC, "divide");
    // Both operands FP64: the summed sum already is, and dividing it by a BIGINT count would
    // leave the operand widening to whatever rule the consumer happens to apply.
    let average = expr_translator::scalar_function(
        divide_anchor,
        vec![
            field_selection(sum),
            expr_translator::cast_to_fp64(field_selection(count)),
        ],
        type_mapper::fp64_type(true),
    );
    use substrait::proto::expression::{IfThen, RexType, if_then};
    Expression {
        rex_type: Some(RexType::IfThen(Box::new(IfThen {
            ifs: vec![if_then::IfClause {
                r#if: Some(no_values),
                then: Some(null_literal(type_mapper::fp64_type(true))),
            }],
            r#else: Some(Box::new(average)),
        }))),
    }
}

/// The types the FE's output tuple declares for an aggregation's measures.
///
/// This is the receiver's view of the row (a downstream exchange builds its stream schema
/// from these very slots), so it is what the emitting fragment has to produce.
fn declared_measure_types(
    desc: &DescriptorTable,
    output_tuple: i32,
    measure_slots: &[i32],
) -> Result<Vec<substrait::proto::Type>> {
    measure_slots
        .iter()
        .map(|slot_id| {
            desc.slot(output_tuple, *slot_id)?
                .substrait_type
                .clone()
                .ok_or(TranslateError::MissingField {
                    context: "aggregate output slot",
                    field: "slotType",
                })
        })
        .collect()
}

/// Names the row a partial aggregation emits: the FE's output-tuple names, plus one suffixed
/// name per extra state column so the receiver spells the column the same way.
fn expanded_output_names(
    desc: &DescriptorTable,
    output_tuple: i32,
    output_slots: &[i32],
    keys: usize,
    wire_columns: &[Vec<WireColumn>],
) -> Result<Vec<String>> {
    let mut names = Vec::with_capacity(output_slots.len());
    for (index, slot_id) in output_slots.iter().enumerate() {
        let name = desc.slot(output_tuple, *slot_id)?.output_name();
        match index.checked_sub(keys) {
            Some(measure) => names.extend(
                wire_columns[measure]
                    .iter()
                    .map(|column| format!("{name}{}", column.suffix)),
            ),
            None => names.push(name),
        }
    }
    Ok(names)
}

/// The FE allocates one output slot per grouping expression, reusing the expression's own
/// column-ref id (verified against FE fragment dumps: grouping refs [18, 13, 16] over the
/// input tuple pair with output key slots 18/13/16), and serializes every aggregation-output
/// slot with column_pos -1 -- so `key_slots` lists the keys in ascending slot id, not GROUP BY
/// order. Each grouping expression must be a bare slot ref and the pairing must be a
/// bijection; anything else means the FE laid the tuple out differently than this model
/// (including a measure slot sorted in among the keys, which would also mis-type every
/// declared measure), and reordering on a wrong model would ship wrong columns -- silently.
fn grouping_materialization_order(
    node: &TPlanNode,
    grouping_exprs: &[TExpr],
    key_slots: &[i32],
) -> Result<Vec<usize>> {
    let mut by_slot = std::collections::HashMap::with_capacity(grouping_exprs.len());
    for (index, expr) in grouping_exprs.iter().enumerate() {
        let slot_id = grouping_slot_id(expr).ok_or(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "a grouping expression that is not a bare slot ref leaves the output key \
                     column order unrecoverable",
        })?;
        if by_slot.insert(slot_id, index).is_some() {
            return Err(TranslateError::descriptor(format!(
                "AGGREGATION_NODE {} lists grouping slot {slot_id} twice",
                node.node_id
            )));
        }
    }
    key_slots
        .iter()
        .map(|slot_id| {
            by_slot.remove(slot_id).ok_or_else(|| {
                TranslateError::descriptor(format!(
                    "AGGREGATION_NODE {} output key slot {slot_id} pairs with no grouping \
                     expression",
                    node.node_id
                ))
            })
        })
        .collect()
}

/// Returns the slot id a grouping expression names, if it is a bare slot ref.
///
/// The ref's tuple is the aggregation's input row, not the output tuple, so only the slot id
/// is read: the FE gives one column the same ref id in both tuples.
fn grouping_slot_id(expr: &TExpr) -> Option<i32> {
    let [node] = expr.nodes.as_slice() else {
        return None;
    };
    if node.node_type != starrocks_thrift::exprs::TExprNodeType::SLOT_REF {
        return None;
    }
    Some(node.slot_ref.as_ref()?.slot_id)
}

/// Returns the FE's serialized function for one aggregate measure.
fn measure_function(expr: &TExpr) -> Result<&starrocks_thrift::types::TFunction> {
    expr.nodes
        .first()
        .and_then(|root| root.fn_.as_ref())
        .ok_or(TranslateError::MissingField {
            context: "aggregate expression",
            field: "fn",
        })
}

/// Returns the single argument of a measure that expands into several Sirius measures.
fn sole_argument<'a>(node: &TPlanNode, arguments: &'a [Expression]) -> Result<&'a Expression> {
    match arguments {
        [argument] => Ok(argument),
        _ => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "a two-phase avg over anything but exactly one argument has no sum/count \
                     expansion (SET new_planner_agg_stage = 1)",
        }),
    }
}

/// Translates a `SORT_NODE` into a Substrait sort (plus the fetch added by `apply_fetch` for
/// top-N limits).
///
/// StarRocks sorts materialize a dedicated sort tuple first (`sort_tuple_slot_exprs`, one
/// expression per materialized slot); the ordering expressions then reference that tuple.
fn translate_sort(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
    let child = children.into_iter().next().unwrap();
    let sort = node
        .sort_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "SORT_NODE",
            field: "sort_node",
        })?;
    let sort_tuple = node
        .row_tuples
        .first()
        .copied()
        .ok_or(TranslateError::MissingField {
            context: "SORT_NODE",
            field: "row_tuples",
        })?;
    // StarRocks' sorter applies the limit internally and never evaluates predicates -- its
    // backend asserts as much (`be/src/exec/topn_node.cpp`: `DCHECK_EQ(_conjuncts.size(), 0)
    // << "TopNNode should never have predicates to evaluate."`), because the FE puts the
    // predicate in a SELECT_NODE above instead. There is therefore no reference semantics for
    // where a sort's own conjuncts sit relative to its limit; translating them either way
    // invents an answer, so refuse the shape.
    if has_conjuncts(node) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "SORT_NODE with conjuncts is not supported",
        });
    }
    // A second row tuple means the sorter carries a payload the sort tuple does not describe.
    // Only the first is translated, so the rest would be dropped from the output row.
    if node.row_tuples.len() > 1 {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "SORT_NODE with more than one row tuple is not supported",
        });
    }
    // StarRocks can fold a partial aggregation into the sorter. Substrait's sort has nowhere to
    // put it, so translating the node as a plain sort would return unaggregated rows.
    if sort
        .pre_agg_exprs
        .as_ref()
        .is_some_and(|exprs| !exprs.is_empty())
        || sort
            .pre_agg_output_slot_id
            .as_ref()
            .is_some_and(|slots| !slots.is_empty())
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "SORT_NODE with a pre-aggregation payload is not supported",
        });
    }
    // Partitioned top-N (per-partition limits) and rank-based top-N have no Substrait
    // representation here; a global sort would silently return the wrong row set.
    if sort
        .partition_exprs
        .as_ref()
        .is_some_and(|exprs| !exprs.is_empty())
        || sort
            .topn_type
            .is_some_and(|topn| topn != starrocks_thrift::plan_nodes::TTopNType::ROW_NUMBER)
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "partitioned or rank-based top-N sorts are not supported",
        });
    }

    // The resolved materialization expressions live in `TSortInfo`; the node-level field is a
    // deprecated duplicate some senders omit.
    let sort_tuple_slot_exprs = sort
        .sort_info
        .sort_tuple_slot_exprs
        .as_ref()
        .or(sort.sort_tuple_slot_exprs.as_ref());
    let input = if let Some(slot_exprs) = sort_tuple_slot_exprs.filter(|exprs| !exprs.is_empty()) {
        let materialized = ctx.desc.materialized_slot_ids(sort_tuple)?;
        if slot_exprs.len() != materialized.len() {
            return Err(TranslateError::descriptor(format!(
                "SORT_NODE {} materializes {} exprs for sort tuple {} with {} slots",
                node.node_id,
                slot_exprs.len(),
                sort_tuple,
                materialized.len()
            )));
        }
        // The FE lists the materialization expressions with the ordering keys first, but every
        // consumer of the sort tuple resolves its columns through the descriptor's materialized
        // slot order: the ordering slot refs below, this fragment's output names and sink row,
        // and the receiving fragment's exchange schema on the other side of the wire. So the
        // projection is emitted in materialized-slot order -- the sender is reordered to the
        // one order all consumers already use. The alternative (overriding the receiver's
        // declared stream schema to the FE list order) would satisfy the wire-schema guard and
        // then feed every downstream slot reference, which still resolves through the
        // descriptor order, the wrong column -- silently. The same invariant governs the
        // aggregation node's grouping keys (see grouping_materialization_order): every node
        // that materializes a new tuple must ship it in the tuple's materialized-slot order.
        let fe_slot_order = sort_materialization_order(
            node,
            &sort.sort_info.ordering_exprs,
            sort_tuple,
            &materialized,
        )?;
        let mut translated_by_slot = std::collections::HashMap::with_capacity(slot_exprs.len());
        for (slot_id, expr) in fe_slot_order.iter().zip(slot_exprs) {
            let mut expr_ctx = ctx
                .expr_context(&child.row_tuples)
                .with_carried(&child.carried_slots);
            translated_by_slot.insert(*slot_id, expr.translate(&mut expr_ctx)?);
        }
        let expressions = materialized
            .iter()
            .map(|slot_id| {
                translated_by_slot.remove(slot_id).ok_or_else(|| {
                    TranslateError::descriptor(format!(
                        "sort tuple {sort_tuple} slot {slot_id} has no materialization expression"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The sort-tuple projection re-materializes the row; the child's carried columns were
        // consumed by the expressions above and are dropped here by construction.
        project_rel(child, expressions, vec![sort_tuple], Vec::new())
    } else {
        child
    };

    let sorts = sort_fields(&sort.sort_info, &input, ctx)?;
    let row_tuples = input.row_tuples.clone();
    let output_width = input.output_width;
    // A sort keeps the row layout, so any carried columns (the no-sort-tuple path) survive it.
    let carried_slots = input.carried_slots.clone();
    let sorted = TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Sort(Box::new(SortRel {
                input: Some(Box::new(input.rel)),
                sorts,
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        carried_slots,
    };
    apply_conjuncts(sorted, node, ctx)
}

/// Pairs each `sort_tuple_slot_exprs` entry with the sort-tuple slot it materializes,
/// returning the slot ids in the FE's expression-list order.
///
/// The FE builds the sort tuple in two passes (`PlanFragmentBuilder.buildPartialTopNFragment`):
/// one slot per ordering key first, then one slot per remaining output column in ascending
/// slot-id order; `sort_tuple_slot_exprs` is appended in the same two passes. Each ordering
/// expression is a bare slot ref naming the sort-tuple slot its key fills, and the payload
/// slots are exactly the materialized slots left over -- which `materialized` already lists in
/// ascending slot-id order, because the FE serializes every sort-tuple slot with column_pos -1.
fn sort_materialization_order(
    node: &TPlanNode,
    ordering_exprs: &[TExpr],
    sort_tuple: i32,
    materialized: &[i32],
) -> Result<Vec<i32>> {
    let mut order = Vec::with_capacity(materialized.len());
    for expr in ordering_exprs {
        let slot_id =
            sort_tuple_slot_id(expr, sort_tuple).ok_or(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "a sort ordering expression that is not a slot ref into the sort tuple \
                         leaves the materialized column order unrecoverable",
            })?;
        order.push(slot_id);
    }
    let payload_slots = materialized
        .iter()
        .copied()
        .filter(|slot_id| !order.contains(slot_id))
        .collect::<Vec<_>>();
    order.extend(payload_slots);
    // Equal lengths make `order` a permutation of `materialized`: a duplicate ordering key or a
    // key outside the tuple both inflate it. Anything but a permutation means the FE laid the
    // tuple out differently than the two-pass model above, and reordering on a wrong model
    // would ship wrong columns.
    if order.len() != materialized.len() {
        return Err(TranslateError::descriptor(format!(
            "SORT_NODE {} ordering keys pair with slots {order:?}, but sort tuple {sort_tuple} \
             materializes slots {materialized:?}",
            node.node_id
        )));
    }
    Ok(order)
}

/// Returns the sort-tuple slot a sort ordering expression names, if it is a bare slot ref into
/// that tuple.
fn sort_tuple_slot_id(expr: &TExpr, sort_tuple: i32) -> Option<i32> {
    let [node] = expr.nodes.as_slice() else {
        return None;
    };
    if node.node_type != starrocks_thrift::exprs::TExprNodeType::SLOT_REF {
        return None;
    }
    let slot_ref = node.slot_ref.as_ref()?;
    (slot_ref.tuple_id == sort_tuple).then_some(slot_ref.slot_id)
}

/// Builds Substrait sort fields from a StarRocks sort-info payload against `input`'s row layout.
fn sort_fields(
    sort_info: &TSortInfo,
    input: &TranslatedRel,
    ctx: &mut PlanContext<'_>,
) -> Result<Vec<SortField>> {
    let ordering = &sort_info.ordering_exprs;
    if sort_info.is_asc_order.len() != ordering.len()
        || sort_info.nulls_first.len() != ordering.len()
    {
        return Err(TranslateError::malformed(
            "sort info direction lists do not match ordering expressions",
        ));
    }
    ordering
        .iter()
        .zip(sort_info.is_asc_order.iter().zip(&sort_info.nulls_first))
        .map(|(expr, (asc, nulls_first))| {
            let mut expr_ctx = ctx
                .expr_context(&input.row_tuples)
                .with_carried(&input.carried_slots);
            let expr = expr.translate(&mut expr_ctx)?;
            let direction = match (asc, nulls_first) {
                (true, true) => sort_field::SortDirection::AscNullsFirst,
                (true, false) => sort_field::SortDirection::AscNullsLast,
                (false, true) => sort_field::SortDirection::DescNullsFirst,
                (false, false) => sort_field::SortDirection::DescNullsLast,
            };
            Ok(SortField {
                expr: Some(expr),
                sort_kind: Some(sort_field::SortKind::Direction(direction as i32)),
            })
        })
        .collect()
}

/// Translates a `HASH_JOIN_NODE` into a Substrait join relation.
///
/// StarRocks children are `[probe (left), build (right)]`; the Substrait join condition is
/// evaluated over the concatenated left-then-right row, which is exactly how
/// `slot_global_index` resolves slots against the combined layout.
fn translate_hash_join(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 2)?;
    reject_common_slots(
        node,
        node.hash_join_node
            .as_ref()
            .and_then(|join| join.common_slot_map.as_ref()),
    )?;
    let join = node
        .hash_join_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "HASH_JOIN_NODE",
            field: "hash_join_node",
        })?;
    // Validated before the conjuncts so an unsupported op is reported as such, rather than as
    // missing conjuncts, which some join shapes arrive with once the FE has folded predicates away.
    let (join_type, output) = match join.join_op {
        TJoinOp::INNER_JOIN => (join_rel::JoinType::Inner, JoinOutput::Both),
        TJoinOp::LEFT_OUTER_JOIN => (join_rel::JoinType::Left, JoinOutput::Both),
        TJoinOp::RIGHT_OUTER_JOIN => (join_rel::JoinType::Right, JoinOutput::Both),
        TJoinOp::FULL_OUTER_JOIN => (join_rel::JoinType::Outer, JoinOutput::Both),
        TJoinOp::LEFT_SEMI_JOIN => (join_rel::JoinType::LeftSemi, JoinOutput::Left),
        TJoinOp::LEFT_ANTI_JOIN => (join_rel::JoinType::Left, JoinOutput::LeftAnti),
        TJoinOp::RIGHT_ANTI_JOIN => (join_rel::JoinType::Right, JoinOutput::RightAnti),
        TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN => {
            (join_rel::JoinType::LeftMark, JoinOutput::NullAwareLeftAnti)
        }
        _ => {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "hash join type is unsupported",
            });
        }
    };

    // A null-aware anti join is only equivalent to `LeftMark + NOT(marker)` when the marker's
    // NULL-ness is decided per probe row. Neither executor does that: DuckDB sets one global
    // `has_null` if any build row has a NULL in any equality key and then rewrites every FALSE
    // marker to NULL (`duckdb/src/execution/join_hashtable.cpp:431` and `:1211-1217`), and the
    // GPU path does the same with `set_build_has_null(_build_has_null,
    // table_has_any_null(right_keys))` in `src/op/sirius_physical_hash_join.cpp`. The per-group
    // path that would be correct is only reachable from DuckDB's own delim-join planner, never
    // from a Substrait `JoinRel`.
    //
    // That is exact for a single equality key and nothing else, because then "unmatched with a
    // NULL somewhere on the build side" really is UNKNOWN. It is wrong as soon as another
    // predicate can make a row definitely non-matching: a correlated `NOT IN` puts its
    // correlation predicate in `other_join_conjuncts` (FE `QuantifiedApply2JoinRule` builds
    // `eq AND correlatedConjuncts AND predicate`, and `JoinHelper` filters correlated equalities
    // out of the eq conjuncts), and a tuple `NOT IN` arrives as several eq conjuncts. In both
    // cases a row that is definitely FALSE is reported UNKNOWN and silently dropped, so
    // `NOT IN` returns too few rows -- often none. StarRocks itself does not have this problem;
    // its BE only short-circuits when `_other_join_conjunct_ctxs` is empty.
    if matches!(join.join_op, TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN)
        && (join.eq_join_conjuncts.len() != 1
            || join
                .other_join_conjuncts
                .as_ref()
                .is_some_and(|conjuncts| !conjuncts.is_empty()))
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "null-aware left anti join with correlated or multi-column keys",
        });
    }

    let mut children = children.into_iter();
    let left = children.next().unwrap();
    let right = children.next().unwrap();
    refuse_carried_join_child(node, &left)?;
    refuse_carried_join_child(node, &right)?;

    let combined_tuples = [left.row_tuples.as_slice(), right.row_tuples.as_slice()].concat();
    let mut conditions = Vec::new();
    let mut first_equality = None;
    for eq in &join.eq_join_conjuncts {
        if let Some(opcode) = eq.opcode
            && opcode != TExprOpcode::EQ
        {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "only plain equality join conjuncts are supported",
            });
        }
        let mut expr_ctx = ctx.expr_context(&combined_tuples);
        let left_expr = eq.left.translate(&mut expr_ctx)?;
        let mut expr_ctx = ctx.expr_context(&combined_tuples);
        let right_expr = eq.right.translate(&mut expr_ctx)?;
        if first_equality.is_none() {
            first_equality = Some((left_expr.clone(), right_expr.clone()));
        }
        let anchor = ctx.registry.register_function(URN_COMPARISON, "equal");
        conditions.push(expr_translator::scalar_function(
            anchor,
            vec![left_expr, right_expr],
            crate::type_mapper::bool_type(),
        ));
    }
    for expr in join.other_join_conjuncts.as_deref().unwrap_or_default() {
        let mut expr_ctx = ctx.expr_context(&combined_tuples);
        conditions.push(expr.translate(&mut expr_ctx)?);
    }
    let condition = and_conditions(conditions, ctx).ok_or(TranslateError::UnsupportedPlanNode {
        node_id: node.node_id,
        node_type: node.node_type,
        reason: "hash join without join conjuncts",
    })?;

    let (row_tuples, output_width) = match output {
        JoinOutput::Left => (left.row_tuples.clone(), left.output_width),
        JoinOutput::NullAwareLeftAnti => (left.row_tuples.clone(), left.output_width + 1),
        _ => (
            combined_tuples.clone(),
            left.output_width + right.output_width,
        ),
    };

    let joined = TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
                left: Some(Box::new(left.rel)),
                right: Some(Box::new(right.rel)),
                expression: Some(Box::new(condition)),
                r#type: join_type as i32,
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        // Both children were refused above if they carried anything.
        carried_slots: Vec::new(),
    };
    let joined = match output {
        JoinOutput::LeftAnti => {
            let (_, right_key) = first_equality
                .ok_or_else(|| TranslateError::malformed("left anti join has no equality key"))?;
            require_column_key(node, &right_key)?;
            let filtered = filter_is_null(joined, right_key, ctx);
            emit_columns(
                filtered.rel,
                (0..left.output_width as i32).collect(),
                left.row_tuples,
            )
        }
        JoinOutput::RightAnti => {
            let (left_key, _) = first_equality
                .ok_or_else(|| TranslateError::malformed("right anti join has no equality key"))?;
            require_column_key(node, &left_key)?;
            let filtered = filter_is_null(joined, left_key, ctx);
            let start = left.output_width as i32;
            let end = start + right.output_width as i32;
            emit_columns(filtered.rel, (start..end).collect(), right.row_tuples)
        }
        JoinOutput::NullAwareLeftAnti => {
            let marker = field_selection(left.output_width as i32);
            let not_anchor = ctx.registry.register_function(URN_BOOLEAN, "not");
            let condition = expr_translator::scalar_function(
                not_anchor,
                vec![marker],
                crate::type_mapper::bool_type(),
            );
            let filtered = filter_rel(joined, condition);
            emit_columns(
                filtered.rel,
                (0..left.output_width as i32).collect(),
                left.row_tuples,
            )
        }
        _ => joined,
    };
    // Node conjuncts are post-join predicates over the join's output row.
    apply_conjuncts(joined, node, ctx)
}

#[derive(Clone, Copy)]
enum JoinOutput {
    Both,
    Left,
    LeftAnti,
    RightAnti,
    NullAwareLeftAnti,
}

/// Refuses a join child that physically carries common-expr columns.
///
/// Join conjuncts resolve through `slot_global_index` over the concatenated child rows using
/// DESCRIPTOR materialized widths; a carried (wider-than-descriptor) child would physically
/// shift every right-side column while the descriptor math would not. That is a wrong column
/// read, not a type error the engine would catch. Consumers of a common slot across a join are
/// therefore refused, loudly, at translation.
fn refuse_carried_join_child(node: &TPlanNode, child: &TranslatedRel) -> Result<()> {
    if child.carried_slots.is_empty() {
        return Ok(());
    }
    let slots: Vec<i32> = child
        .carried_slots
        .iter()
        .map(|slot| slot.slot_id)
        .collect();
    Err(TranslateError::descriptor(format!(
        "{:?} {} reads a child that carries common-expr columns for slots {slots:?}; carried \
         columns would shift the join's combined descriptor row, so consumers of a common \
         slot across a join are not supported",
        node.node_type, node.node_id
    )))
}

/// Translates an inner/cross `NESTLOOP_JOIN_NODE` into an equality join on synthetic constants.
/// This preserves Cartesian-product semantics without requiring a GPU cross-product operator.
fn translate_nestloop_join(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 2)?;
    let join = node
        .nestloop_join_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "NESTLOOP_JOIN_NODE",
            field: "nestloop_join_node",
        })?;
    reject_common_slots(node, join.common_slot_map.as_ref())?;
    match join.join_op {
        None | Some(TJoinOp::CROSS_JOIN) | Some(TJoinOp::INNER_JOIN) => {}
        Some(_) => {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "only inner/cross nested-loop joins are supported",
            });
        }
    }
    // Lowering a Cartesian product to a constant-key equality join replaced the rejection that
    // used to refuse it ("the GPU physical planner has no cross-product operator"), so this shape
    // now reaches the GPU instead of failing translation. Nothing here bounds its size: the FE
    // reports `cardinality: 1` for every FILES() external scan, so the translator has no estimate
    // to gate on, and TPC-H q08/q09 at SF100 plan a genuine `NESTLOOP JOIN / CROSS JOIN` whose
    // build side exhausts memory. Bounding it belongs to the executor, which knows the real row
    // counts; refusing it here would also refuse the small cross joins the FE emits from
    // scalar-subquery rewrites.
    let mut children = children.into_iter();
    let left = children.next().unwrap();
    let right = children.next().unwrap();
    refuse_carried_join_child(node, &left)?;
    refuse_carried_join_child(node, &right)?;
    let left_width = left.output_width;
    let right_width = right.output_width;
    let row_tuples = [left.row_tuples.as_slice(), right.row_tuples.as_slice()].concat();
    let left = append_project(left, i32_literal(1));
    let right = append_project(right, i32_literal(1));
    let equal_anchor = ctx.registry.register_function(URN_COMPARISON, "equal");
    let condition = expr_translator::scalar_function(
        equal_anchor,
        vec![
            field_selection(left_width as i32),
            field_selection((left.output_width + right_width) as i32),
        ],
        crate::type_mapper::bool_type(),
    );
    // Kept as a bare `Rel`, not a `TranslatedRel`: the join row carries both synthetic keys, so
    // it is two columns wider than `row_tuples` describes, and a `TranslatedRel` claiming that
    // layout would resolve every right-side slot to the wrong index. The projection below drops
    // the keys and restores the invariant.
    let joined = Rel {
        rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(left.rel)),
            right: Some(Box::new(right.rel)),
            expression: Some(Box::new(condition)),
            r#type: join_rel::JoinType::Inner as i32,
            ..Default::default()
        }))),
    };
    let mut mapping = (0..left_width as i32).collect::<Vec<_>>();
    mapping.extend(left.output_width as i32..left.output_width as i32 + right_width as i32);
    let cross = emit_columns(joined, mapping, row_tuples);
    let filtered = if let Some(conjuncts) = join
        .join_conjuncts
        .as_ref()
        .filter(|conjuncts| !conjuncts.is_empty())
    {
        let mut conditions = Vec::with_capacity(conjuncts.len());
        for expr in conjuncts {
            let mut expr_ctx = ctx.expr_context(&cross.row_tuples);
            conditions.push(expr.translate(&mut expr_ctx)?);
        }
        match and_conditions(conditions, ctx) {
            Some(condition) => {
                let TranslatedRel {
                    rel,
                    row_tuples,
                    output_width,
                    carried_slots,
                } = cross;
                TranslatedRel {
                    rel: Rel {
                        rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                            input: Some(Box::new(rel)),
                            condition: Some(Box::new(condition)),
                            ..Default::default()
                        }))),
                    },
                    row_tuples,
                    output_width,
                    carried_slots,
                }
            }
            None => cross,
        }
    } else {
        cross
    };
    // Node conjuncts are post-join predicates over the join's output row.
    apply_conjuncts(filtered, node, ctx)
}

/// Combines boolean conditions with `and`.
///
/// `None` for an empty list: a zero-argument `and()` is not a valid Substrait expression, so what
/// an absent condition means is the caller's decision.
fn and_conditions(
    mut conditions: Vec<Expression>,
    ctx: &mut PlanContext<'_>,
) -> Option<Expression> {
    match conditions.len() {
        0 => None,
        1 => conditions.pop(),
        _ => {
            let anchor = ctx.registry.register_function(URN_BOOLEAN, "and");
            Some(expr_translator::scalar_function(
                anchor,
                conditions,
                crate::type_mapper::bool_type(),
            ))
        }
    }
}

/// Builds a Substrait read for a StarRocks scan tuple.
///
/// With `file_paths` present (FILE_SCAN broker ranges) it emits a `local_files`
/// parquet read so DuckDB's Substrait reader resolves the scan to
/// `parquet_scan(<paths>)`. v1 assumes parquet files whose column order matches
/// the scan tuple's slot order, which holds for `FILES()` `SELECT *`. Without
/// paths (e.g. HDFS scans) it falls back to a named-table read.
fn scan_rel(desc: &DescriptorTable, tuple_id: i32, file_paths: &[String]) -> Result<Rel> {
    let read_type = if file_paths.is_empty() {
        ReadType::NamedTable(NamedTable {
            names: desc.table_names_for_tuple(tuple_id)?,
            ..Default::default()
        })
    } else {
        ReadType::LocalFiles(LocalFiles {
            items: file_paths
                .iter()
                .map(|path| FileOrFiles {
                    path_type: Some(PathType::UriFile(path.clone())),
                    file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        })
    };
    Ok(Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(desc.named_struct(tuple_id)?),
            read_type: Some(read_type),
            ..Default::default()
        }))),
    })
}

/// Builds a read of an engine stream view with an explicit schema.
///
/// The view is a named table as far as Substrait is concerned; the engine defines it as a read of
/// the corresponding input stream, so the plan never names a file.
fn stream_read_rel(schema: substrait::proto::NamedStruct, stream_view: &str) -> Rel {
    Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(schema),
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec![stream_view.to_string()],
                ..Default::default()
            })),
            ..Default::default()
        }))),
    }
}

/// Translates a StarRocks project node while preserving descriptor output order.
fn translate_project_node(
    child: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let project_node = node
        .project_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "PROJECT_NODE",
            field: "project_node",
        })?;
    let slot_map = project_node
        .slot_map
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TProjectNode",
            field: "slot_map",
        })?;

    let output_tuples = if node.row_tuples.is_empty() {
        return Err(TranslateError::MissingField {
            context: "PROJECT_NODE",
            field: "row_tuples",
        });
    } else {
        node.row_tuples.clone()
    };

    let output_tuple = output_tuples[0];
    let mut input = child;
    let mut common_slots = std::collections::HashMap::new();
    for (&slot_id, expr) in project_node.common_slot_map.as_ref().into_iter().flatten() {
        let expression = {
            // A project stacked on a carrying project references the lower one's carried
            // columns; append_project never shifts existing columns, so the child's carried
            // indices stay valid throughout this loop.
            let mut expr_ctx = ctx
                .expr_context_with_slots(&input.row_tuples, &common_slots)
                .with_carried(&input.carried_slots);
            expr.translate(&mut expr_ctx)?
        };
        let field = input.output_width;
        input = append_project(input, expression);
        common_slots.insert((output_tuple, slot_id), field);
    }

    let mut expressions = Vec::new();
    for &tuple_id in &output_tuples {
        for slot_id in ctx.desc.materialized_slot_ids(tuple_id)? {
            let expr = slot_map.get(&slot_id).ok_or_else(|| {
                TranslateError::descriptor(format!(
                    "PROJECT_NODE node {} missing slot_map expression for slot {}",
                    node.node_id, slot_id
                ))
            })?;
            let mut expr_ctx = ctx
                .expr_context_with_slots(&input.row_tuples, &common_slots)
                .with_carried(&input.carried_slots);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
    }

    // Physically materialize the common-expr slots an ancestor consumes even though no output
    // tuple materializes them: StarRocks' BE outputs them and the FE plans against that (it
    // even emits a slot_map entry for them). They land as extra trailing columns past the
    // descriptor row, and the returned carried_slots bindings are how ancestor refs find them.
    let materialized_count = expressions.len();
    let consumed = ctx
        .consumed_above
        .get(&node.node_id)
        .cloned()
        .unwrap_or_default();
    let mut carried = Vec::with_capacity(consumed.len());
    for (position, &slot_id) in consumed.iter().enumerate() {
        let expression = if let Some(expr) = slot_map.get(&slot_id) {
            // The FE's own slot_map entry for the consumed slot (q14: `35 <-> ref 35`); its
            // ref resolves to the appended common column through the override path.
            let mut expr_ctx = ctx
                .expr_context_with_slots(&input.row_tuples, &common_slots)
                .with_carried(&input.carried_slots);
            expr.translate(&mut expr_ctx)?
        } else if let Some(&column) = common_slots.get(&(output_tuple, slot_id)) {
            field_selection(column as i32)
        } else {
            return Err(TranslateError::descriptor(format!(
                "PROJECT_NODE {} common slot {slot_id} is consumed above but the project has \
                 no slot_map entry or common binding for it",
                node.node_id
            )));
        };
        expressions.push(expression);
        carried.push(CarriedSlot {
            tuple_id: output_tuple,
            slot_id,
            column: materialized_count + position,
        });
    }

    Ok(project_rel(input, expressions, output_tuples, carried))
}

/// Adds a root projection over explicit fragment output expressions.
pub(crate) fn project_exprs(
    input: TranslatedRel,
    exprs: &[TExpr],
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    // Root projections evaluate over already-translated inputs, so there are no
    // scan nodes to resolve file paths for and no exchange nodes to bind streams to.
    let scan_paths = ScanFilePaths::default();
    let exchange_inputs = HashMap::new();
    let mut ctx = PlanContext::new(desc, &scan_paths, &exchange_inputs, registry);
    project_exprs_with_context(input, exprs, &mut ctx)
}

/// Adds a projection with expressions evaluated against the input row layout.
fn project_exprs_with_context(
    input: TranslatedRel,
    exprs: &[TExpr],
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let mut expressions = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let mut expr_ctx = ctx
            .expr_context(&input.row_tuples)
            .with_carried(&input.carried_slots);
        expressions.push(expr.translate(&mut expr_ctx)?);
    }
    // A root projection over fragment output expressions keeps the input layout. It emits
    // only the output expressions, so any carried columns are narrowed away here.
    let row_tuples = input.row_tuples.clone();
    Ok(project_rel(input, expressions, row_tuples, Vec::new()))
}

/// Builds a Substrait project that emits exactly `expressions`.
///
/// The emit `output_mapping` selects the projected expressions, which sit after
/// the input columns, so the base offset is the input's carried `output_width`.
/// `row_tuples` is the output row layout (the project's own tuples, which may
/// reorder or differ from the input's). `carried_slots` names the trailing
/// expressions that carry common-expr columns past the descriptor row; every
/// caller but [`translate_project_node`] passes none.
fn project_rel(
    input: TranslatedRel,
    expressions: Vec<Expression>,
    row_tuples: Vec<i32>,
    carried_slots: Vec<CarriedSlot>,
) -> TranslatedRel {
    let base = input.output_width as i32;
    let output_mapping = (base..base + expressions.len() as i32).collect();
    let output_width = expressions.len();
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping,
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input.rel)),
                expressions,
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        carried_slots,
    }
}

/// Appends one expression to a relation while retaining all existing columns.
fn append_project(input: TranslatedRel, expression: Expression) -> TranslatedRel {
    let output_width = input.output_width + 1;
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping: (0..output_width as i32).collect(),
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input.rel)),
                expressions: vec![expression],
                ..Default::default()
            }))),
        },
        row_tuples: input.row_tuples,
        output_width,
        // The appended column lands at the END of the row, past any carried columns, so the
        // carried indices stay valid and pass through unchanged.
        carried_slots: input.carried_slots,
    }
}

/// Emits selected input columns without evaluating new expressions.
fn emit_columns(input: Rel, output_mapping: Vec<i32>, row_tuples: Vec<i32>) -> TranslatedRel {
    let output_width = output_mapping.len();
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping,
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input)),
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        // An explicit column list re-materializes the row; carried columns never survive it.
        carried_slots: Vec::new(),
    }
}

/// Drops a fragment root's carried common-expr columns, restoring the descriptor row.
///
/// Required, not defensive: without it a width-preserving consumer chain above the carrying
/// project (a SELECT conjunct as the fragment root, say) would ship the carried columns while
/// the sender's names and the receiver's declared stream schema describe only the descriptor
/// row, a drift the receiver-side guard would only catch at runtime on the engine.
pub(crate) fn confine_carried(input: TranslatedRel) -> TranslatedRel {
    if input.carried_slots.is_empty() {
        return input;
    }
    let descriptor_width = input.output_width - input.carried_slots.len();
    emit_columns(
        input.rel,
        (0..descriptor_width as i32).collect(),
        input.row_tuples,
    )
}

/// Builds a direct field selection against the current relation output.
fn field_selection(field: i32) -> Expression {
    use substrait::proto::expression::field_reference;
    use substrait::proto::expression::reference_segment;
    use substrait::proto::expression::{FieldReference, ReferenceSegment};

    Expression {
        rex_type: Some(substrait::proto::expression::RexType::Selection(Box::new(
            FieldReference {
                reference_type: Some(field_reference::ReferenceType::DirectReference(
                    ReferenceSegment {
                        reference_type: Some(reference_segment::ReferenceType::StructField(
                            Box::new(reference_segment::StructField { field, child: None }),
                        )),
                    },
                )),
                root_type: Some(field_reference::RootType::RootReference(
                    field_reference::RootReference {},
                )),
            },
        ))),
    }
}

/// Returns the field index of a direct struct-field reference, if the expression is one.
fn direct_field(expr: &Expression) -> Option<i32> {
    use substrait::proto::expression::{RexType, field_reference, reference_segment};

    let RexType::Selection(selection) = expr.rex_type.as_ref()? else {
        return None;
    };
    let field_reference::ReferenceType::DirectReference(segment) =
        selection.reference_type.as_ref()?
    else {
        return None;
    };
    let reference_segment::ReferenceType::StructField(field) = segment.reference_type.as_ref()?
    else {
        return None;
    };
    field.child.is_none().then_some(field.field)
}

/// Builds a typed Substrait null literal.
fn null_literal(ty: substrait::proto::Type) -> Expression {
    Expression {
        rex_type: Some(substrait::proto::expression::RexType::Literal(
            substrait::proto::expression::Literal {
                literal_type: Some(substrait::proto::expression::literal::LiteralType::Null(ty)),
                ..Default::default()
            },
        )),
    }
}

/// Builds an i64 literal used to compare against a merged count.
fn i64_literal(value: i64) -> Expression {
    Expression {
        rex_type: Some(substrait::proto::expression::RexType::Literal(
            substrait::proto::expression::Literal {
                literal_type: Some(substrait::proto::expression::literal::LiteralType::I64(
                    value,
                )),
                ..Default::default()
            },
        )),
    }
}

/// Builds an i32 literal used as a synthetic Cartesian-product key.
fn i32_literal(value: i32) -> Expression {
    Expression {
        rex_type: Some(substrait::proto::expression::RexType::Literal(
            substrait::proto::expression::Literal {
                literal_type: Some(substrait::proto::expression::literal::LiteralType::I32(
                    value,
                )),
                ..Default::default()
            },
        )),
    }
}

/// Wraps a relation in a filter without changing its row layout.
fn filter_rel(input: TranslatedRel, condition: Expression) -> TranslatedRel {
    let output_width = input.output_width;
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(input.rel)),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        },
        row_tuples: input.row_tuples,
        output_width,
        // A filter keeps the row layout, carried columns included.
        carried_slots: input.carried_slots,
    }
}

/// Refuses an anti join whose null-tested key is not a column reference (casts allowed).
///
/// The outer-join + `is_null(key)` lowering identifies an unmatched row by the NULL padding the
/// join puts on the other side. That is only exact when the key expression propagates NULL: a
/// column reference does, and so does a cast of one, but `if`/`case`/`coalesce` over a column can
/// yield a non-NULL value for the padded row, and the filter would then drop an unmatched row as
/// if it had matched. Refuse those shapes rather than return too few rows.
fn require_column_key(node: &TPlanNode, key: &Expression) -> Result<()> {
    fn is_column_reference(expr: &Expression) -> bool {
        match expr.rex_type.as_ref() {
            Some(expression::RexType::Selection(_)) => true,
            Some(expression::RexType::Cast(cast)) => {
                cast.input.as_deref().is_some_and(is_column_reference)
            }
            _ => false,
        }
    }
    if is_column_reference(key) {
        return Ok(());
    }
    Err(TranslateError::UnsupportedPlanNode {
        node_id: node.node_id,
        node_type: node.node_type,
        reason: "anti join key is not a plain column reference",
    })
}

/// Filters to rows where an equality-key expression is null.
fn filter_is_null(
    input: TranslatedRel,
    key: Expression,
    ctx: &mut PlanContext<'_>,
) -> TranslatedRel {
    let anchor = ctx.registry.register_function(URN_COMPARISON, "is_null");
    let condition =
        expr_translator::scalar_function(anchor, vec![key], crate::type_mapper::bool_type());
    filter_rel(input, condition)
}

/// Wraps a relation in a Substrait filter when the StarRocks node has conjuncts.
fn apply_conjuncts(
    input: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let conjuncts = node.conjuncts.as_deref().unwrap_or_default();
    let mut conditions = Vec::with_capacity(conjuncts.len());
    for expr in conjuncts {
        let mut expr_ctx = ctx
            .expr_context(&input.row_tuples)
            .with_carried(&input.carried_slots);
        conditions.push(expr.translate(&mut expr_ctx)?);
    }
    let Some(condition) = and_conditions(conditions, ctx) else {
        return Ok(input);
    };

    // A filter does not change the column layout, so the width passes through.
    let output_width = input.output_width;
    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(input.rel)),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        },
        row_tuples: input.row_tuples,
        output_width,
        // A filter keeps the row layout, carried columns included.
        carried_slots: input.carried_slots,
    })
}

/// Returns whether a StarRocks plan node carries filter conjuncts.
fn has_conjuncts(node: &TPlanNode) -> bool {
    node.conjuncts
        .as_ref()
        .map(|conjuncts| !conjuncts.is_empty())
        .unwrap_or(false)
}

/// Validates the reconstructed child count for a StarRocks plan node.
fn expect_children(node: &TPlanNode, children: &[TranslatedRel], expected: usize) -> Result<()> {
    if children.len() != expected {
        return Err(TranslateError::malformed(format!(
            "node {} {:?} expected {} child(ren), got {}",
            node.node_id,
            node.node_type,
            expected,
            children.len()
        )));
    }
    Ok(())
}
