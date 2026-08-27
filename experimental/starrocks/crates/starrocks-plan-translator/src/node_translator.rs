use std::collections::BTreeMap;

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
    ProjectRel, ReadRel, Rel, RelCommon, SortField, SortRel, aggregate_rel, fetch_rel,
    function_argument, join_rel, rel, rel_common, sort_field,
};

use crate::agg_phase::{self, AggPhase};
use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, ExprContext, TranslateExpr};
use crate::partial_state;
use crate::scan_paths::ScanFilePaths;
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
}

/// Mutable state shared by plan-node translators.
struct PlanContext<'a> {
    /// Descriptor lookups for row layouts, tables, and scan schemas.
    desc: &'a DescriptorTable,
    /// Parquet file paths for each scan node, collected from the fragment's broker
    /// scan ranges. Scans with paths emit a `local_files` read; path-less scans
    /// fall back to a named-table read.
    scan_paths: &'a ScanFilePaths,
    /// Same-node input streams keyed by receiver exchange node id.
    exchange_inputs: &'a std::collections::HashMap<i32, &'a ExchangeInput>,
    /// Substrait extension registry shared across the whole plan.
    registry: &'a mut ExtensionRegistry,
    /// Schema of every exchange lowered to a stream read, in translation order. The caller has to
    /// declare these on the engine before the plan can bind.
    stream_inputs: Vec<StreamInputSchema>,
    /// Positional wire-type overrides, keyed by exchange node id, for exchanges that feed
    /// merge aggregations: the FE declares those columns with intermediate slot types that lie
    /// about what the sender ships (see `partial_state`). Computed by
    /// [`merge_exchange_overrides`] before translation starts.
    exchange_state_overrides: std::collections::HashMap<i32, Vec<StateColumn>>,
}

/// The wire type one merge measure's partial state occupies on the exchange row.
struct StateColumn {
    /// Index of the measure's column in the FE-declared exchange row.
    position: usize,
    /// Modeled wire type replacing the FE slot type at that position.
    ty: substrait::proto::Type,
}

/// One translated fragment: its relation tree plus the input streams the caller has to declare
/// on the engine before the plan can bind.
pub(crate) struct TranslatedFragment {
    /// Root relation and its row layout.
    pub root: TranslatedRel,
    /// Schema of every exchange lowered to a stream read, in translation order.
    pub stream_inputs: Vec<StreamInputSchema>,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    fn new(
        desc: &'a DescriptorTable,
        scan_paths: &'a ScanFilePaths,
        exchange_inputs: &'a std::collections::HashMap<i32, &'a ExchangeInput>,
        registry: &'a mut ExtensionRegistry,
    ) -> Self {
        Self {
            desc,
            scan_paths,
            exchange_inputs,
            registry,
            stream_inputs: Vec::new(),
            exchange_state_overrides: std::collections::HashMap::new(),
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
    };
    apply_conjuncts(input, node, ctx)
}

/// Wraps the child relation of a `SELECT_NODE` with its filter conjuncts.
/// Refuses a node whose `common_slot_map` this translator does not materialize.
///
/// Only `PROJECT_NODE` appends its common slots. Every other node carrying the field would have
/// its shared sub-expressions dropped, and any slot ref resolving to one of them then reads a
/// column that was never emitted -- wrong values under the right names, with no error.
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
pub(crate) fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    scan_paths: &ScanFilePaths,
    exchange_inputs: &std::collections::HashMap<i32, &ExchangeInput>,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedFragment> {
    let mut ctx = PlanContext::new(desc, scan_paths, exchange_inputs, registry);
    ctx.exchange_state_overrides = merge_exchange_overrides(plan)?;
    let root = plan.translate(&mut ctx)?;
    Ok(TranslatedFragment {
        root,
        stream_inputs: ctx.stream_inputs,
    })
}

/// Computes the positional wire-type overrides for exchanges that feed merge aggregations.
///
/// Runs over the flat preorder node list before translation, because the exchange is
/// translated before its parent aggregation and must already declare the stream with the
/// modeled partial-state column types. In preorder a merge aggregation's single child is
/// simply the next node; anything but an exchange there means the plan reads partial states
/// from a shape this translator cannot type, so it is refused.
fn merge_exchange_overrides(
    plan: &TPlan,
) -> Result<std::collections::HashMap<i32, Vec<StateColumn>>> {
    let mut overrides = std::collections::HashMap::new();
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
        if child.is_none() {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "a merge aggregation must read its partial states directly from an \
                         exchange (SET new_planner_agg_stage = 1)",
            });
        }
        let keys = agg.grouping_exprs.as_deref().unwrap_or_default().len();
        // The exchange row is the intermediate tuple's materialized slots: grouping keys
        // first, then one state slot per measure -- the same layout invariant the
        // aggregation's output tuple uses.
        let mut columns = Vec::with_capacity(agg.aggregate_functions.len());
        for (position, measure) in agg.aggregate_functions.iter().enumerate() {
            columns.push(StateColumn {
                position: keys + position,
                ty: partial_state::wire_type(measure_function(measure)?)?,
            });
        }
        overrides.insert(child.unwrap().node_id, columns);
    }
    Ok(overrides)
}

/// The wire types a partial aggregation's state columns leave the fragment with.
pub(crate) struct PartialStateColumns {
    /// Output-row index of the first state column (= the grouping-key count).
    pub first: usize,
    /// Modeled wire type per measure, in measure order.
    pub types: Vec<substrait::proto::Type>,
}

/// Computes the partial-state wire types for a fragment whose root is a partial aggregation.
///
/// The sender-side mirror of [`merge_exchange_overrides`]: a partial fragment's state columns
/// intentionally leave in the modeled wire types (see `partial_state`), not the FE's
/// intermediate slot types, and the receiving exchange overrides its declared stream the same
/// way. Both ends consult the same pure function of the same FE thrift (function name +
/// `ret_type`), so the sink conformance targets and the receiver's stream schema agree by
/// construction.
///
/// Preorder puts the fragment root at `nodes[0]`; the width-preserving wrappers a root node can
/// grow (conjunct filters, limit fetches) change neither which node is the root nor where its
/// columns sit.
pub(crate) fn partial_root_state_columns(plan: &TPlan) -> Result<Option<PartialStateColumns>> {
    let Some(root) = plan.nodes.first() else {
        return Ok(None);
    };
    if root.node_type != TPlanNodeType::AGGREGATION_NODE {
        return Ok(None);
    }
    // A node without agg_node fails in translate_aggregation with its own error.
    let Some(agg) = root.agg_node.as_ref() else {
        return Ok(None);
    };
    if agg_phase::classify(root.node_id, root.node_type, agg)? != AggPhase::Partial {
        return Ok(None);
    }
    let types = agg
        .aggregate_functions
        .iter()
        .map(|measure| partial_state::wire_type(measure_function(measure)?))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(PartialStateColumns {
        first: agg.grouping_exprs.as_deref().unwrap_or_default().len(),
        types,
    }))
}

/// Replaces a receiver exchange boundary with a read of the sender's output **stream**.
///
/// The sender's rows are already on the GPU, parked in the engine as native batches. So the read
/// resolves to the engine's stream view rather than to a file: nothing is materialized, nothing
/// is re-parsed, and the boundary costs a pointer move. The schema the read declares is also
/// recorded on the context, because the engine has no file to infer it from and the fragment must
/// declare the same one.
///
/// A *merging* exchange (`sort_info` present) additionally carries the cross-fragment ORDER BY.
/// The boundary does not preserve the senders' order — the inputs are read back as an unordered
/// concatenation — so the order is restored here by sorting that concatenation, which is what the
/// merge would have produced. Ordering the union of the senders' rows is correct whether or not
/// each sender's own run arrived sorted, so this holds for any future input model too.
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
                reason: "exchange node requires a bound same-node input stream",
            })?;
    // The stream's columns are the sender's, named by the sender; types and field order come
    // from the receiver's descriptor tuples (FE wire order, the order the sender ships).
    let mut schema = ctx
        .desc
        .named_struct_for_tuples(&exchange.input_row_tuples, Some(&input.names))?;
    // An exchange feeding a merge aggregation carries partial-state columns; rewrite their
    // FE-declared slot types to the modeled wire types. One `Type`, two consumers: the ReadRel
    // base_schema below and the engine's stream declaration derive from the same rewritten
    // entry, so the plan's view of the column and the engine's cannot drift apart.
    if let Some(state_overrides) = ctx.exchange_state_overrides.get(&node.node_id) {
        let types = schema
            .r#struct
            .as_mut()
            .map(|structure| &mut structure.types)
            .ok_or_else(|| TranslateError::malformed("exchange schema has no struct"))?;
        let width = types.len();
        for column in state_overrides {
            let slot = types.get_mut(column.position).ok_or_else(|| {
                TranslateError::malformed(format!(
                    "merge aggregation state column {} is outside the exchange row \
                     ({width} columns)",
                    column.position
                ))
            })?;
            *slot = column.ty.clone();
        }
    }
    let output_width = schema
        .r#struct
        .as_ref()
        .map(|structure| structure.types.len())
        .unwrap_or(0);

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
                ty: crate::type_mapper::duckdb_type_name(ty)?,
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
    };
    // The merge itself: the sort keys resolve against the sender's row layout, the same one
    // `sort_fields` resolves a SORT_NODE's keys against. `apply_fetch` puts the exchange's own
    // limit/offset above this, so a merging exchange with a limit becomes a top-N.
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
/// The grouping keys are emitted in the output tuple's materialized-slot order — on this
/// branch, the FE's wire order (see [`grouping_materialization_order`]): every consumer of the
/// tuple — slot refs above this node, the next hop's declared stream schema, output names,
/// hash-partition indices — resolves its columns through the descriptor's order, so the sender
/// reorders to it.
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

    // A two-phase measure's column on the wire is its modeled partial-state type, not the type
    // the FE's intermediate slot declares; one-shot measures always ship their own slot's
    // type, which is why the model is not consulted for them at all. Unmodeled two-phase
    // functions (avg included) are refused here, loudly.
    let wire_types = match phase {
        AggPhase::OneShot => Vec::new(),
        _ => agg
            .aggregate_functions
            .iter()
            .map(|expr| partial_state::wire_type(measure_function(expr)?))
            .collect::<Result<Vec<_>>>()?,
    };

    // The FE lists the grouping expressions in GROUP BY order; the output tuple's materialized
    // slots are the order every consumer above this node resolves the row through. The two
    // coincide in every FE plan observed on this branch, but the pairing (not the coincidence)
    // is what guarantees the emitted row matches the descriptor view. One key needs no
    // pairing: a single grouping expression can only fill the single key slot.
    let key_order = if keys > 1 {
        grouping_materialization_order(node, grouping_exprs, &output_slots[..keys])?
    } else {
        (0..keys).collect()
    };
    let mut grouping_expressions = Vec::with_capacity(keys);
    for &grouping_index in &key_order {
        let mut expr_ctx = ctx.expr_context(&child.row_tuples);
        grouping_expressions.push(grouping_exprs[grouping_index].translate(&mut expr_ctx)?);
    }

    let mut measures = Vec::with_capacity(agg.aggregate_functions.len());
    for (index, expr) in agg.aggregate_functions.iter().enumerate() {
        let call = {
            let mut expr_ctx = ctx.expr_context(&child.row_tuples);
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
            AggPhase::Partial => wire_types[index].clone(),
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

    let output_width = output_slots.len();
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
    };

    // Every merge node leaves through the finalizing projection: the engine binds a merged
    // integer count/sum as HUGEINT (the plan-level downcast relabels the aggregate node, not a
    // fragment sink above it), so without the projection's throwing casts the fragment's wire
    // row carries a type its FE-declared slot never announced and the next hop's schema guard
    // refuses it.
    let aggregated = match phase {
        AggPhase::Merge => {
            let measure_types =
                declared_measure_types(ctx.desc, output_tuple, &output_slots[keys..])?;
            merge_projection(aggregated, keys, &measure_types)
        }
        _ => aggregated,
    };
    // Node conjuncts evaluate over the aggregation output (HAVING predicates).
    apply_conjuncts(aggregated, node, ctx)
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

/// Adds the projection that turns a merge aggregation's emitted columns back into the FE's
/// output row types: the grouping keys pass through, and every measure column is cast to
/// `measure_types` — the types the FE's output tuple declares, which is what the next fragment
/// derives its stream schema from. Stating them is not cosmetic: DuckDB binds `sum(BIGINT)` as
/// HUGEINT, and the engine's HUGEINT-to-BIGINT downcast happens at the aggregate's own sink,
/// which this projection now sits in front of. Without the cast a merged count leaves this
/// fragment as a HUGEINT the receiver declared BIGINT, and the hop refuses it.
fn merge_projection(
    input: TranslatedRel,
    keys: usize,
    measure_types: &[substrait::proto::Type],
) -> TranslatedRel {
    let mut expressions: Vec<Expression> = (0..keys as i32).map(field_selection).collect();
    for (index, measure_type) in measure_types.iter().enumerate() {
        expressions.push(expr_translator::cast_to(
            field_selection((keys + index) as i32),
            measure_type.clone(),
        ));
    }
    let row_tuples = input.row_tuples.clone();
    project_rel(input, expressions, row_tuples)
}

/// Wraps a data-stream-sink fragment's root in the sink conformance projection: one throwing
/// cast per output column, positionally, to `declared` — the FE-declared wire type of that
/// column (the caller overrides a partial aggregation's state columns to their modeled wire
/// types first). Width, column order, and `row_tuples` all pass through, so partition-column
/// indices resolved against the pre-conformance row stay valid.
///
/// The casts are emitted UNCONDITIONALLY. The translator's own type model is not
/// engine-accurate for function results — the emitted plan claims `year(date)` has the FE's
/// SMALLINT type while the consumer's binder resolves `year` from its own catalog and produces
/// BIGINT — so a pass keyed on "translated type != declared type" would skip exactly the
/// columns that need fixing. The consumer's binder folds a cast to a column's own type at
/// bind, so already-conformant columns cost nothing.
pub(crate) fn sink_conformance_projection(
    input: TranslatedRel,
    declared: &[substrait::proto::Type],
) -> TranslatedRel {
    let expressions = declared
        .iter()
        .enumerate()
        .map(|(index, ty)| expr_translator::cast_to(field_selection(index as i32), ty.clone()))
        .collect();
    let row_tuples = input.row_tuples.clone();
    project_rel(input, expressions, row_tuples)
}

/// The types the FE's output tuple declares for an aggregation's measures.
///
/// This is the receiver's view of the row — a downstream exchange builds its stream schema
/// from these very slots — so it is what the emitting fragment has to produce.
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

/// Pairs each grouping expression with the aggregation-output key slot it materializes,
/// returning the grouping-expression indices in the output tuple's materialized-slot order.
///
/// The FE allocates one output slot per grouping expression, reusing the expression's own
/// column-ref id, and on this branch the descriptor keeps the tuple's slots in the FE's wire
/// order — the same order the FE appends the grouping keys in. The pairing must still be a
/// bijection over bare slot refs: anything else means the FE laid the tuple out differently
/// than this model (including a measure slot serialized among the keys, which would also
/// mis-type every declared measure), and reordering on a wrong model would ship wrong
/// columns -- silently.
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
        let expected = ctx.desc.materialized_slot_ids(sort_tuple)?.len();
        if slot_exprs.len() != expected {
            return Err(TranslateError::descriptor(format!(
                "SORT_NODE {} materializes {} exprs for sort tuple {} with {} slots",
                node.node_id,
                slot_exprs.len(),
                sort_tuple,
                expected
            )));
        }
        let mut expressions = Vec::with_capacity(slot_exprs.len());
        for expr in slot_exprs {
            let mut expr_ctx = ctx.expr_context(&child.row_tuples);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
        project_rel(child, expressions, vec![sort_tuple])
    } else {
        child
    };

    let sorts = sort_fields(&sort.sort_info, &input, ctx)?;
    let row_tuples = input.row_tuples.clone();
    let output_width = input.output_width;
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
    };
    apply_conjuncts(sorted, node, ctx)
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
            let mut expr_ctx = ctx.expr_context(&input.row_tuples);
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
    let join = node
        .hash_join_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "HASH_JOIN_NODE",
            field: "hash_join_node",
        })?;
    reject_common_slots(node, join.common_slot_map.as_ref())?;
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
    // GPU path does the same with `table_has_any_null(right_keys)`
    // (`src/op/sirius_physical_hash_join.cpp:1657`). The per-group path that would be correct is
    // only reachable from DuckDB's own delim-join planner, never from a Substrait `JoinRel`.
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
    };
    let joined = match output {
        JoinOutput::LeftAnti => {
            let (_, right_key) = first_equality
                .ok_or_else(|| TranslateError::malformed("left anti join has no equality key"))?;
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
fn scan_rel(
    desc: &DescriptorTable,
    tuple_id: i32,
    files: &[crate::scan_paths::ScanFile],
) -> Result<Rel> {
    let read_type = if files.is_empty() {
        ReadType::NamedTable(NamedTable {
            names: desc.table_names_for_tuple(tuple_id)?,
            ..Default::default()
        })
    } else {
        return Ok(local_files_rel(desc.named_struct(tuple_id)?, files));
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

/// Builds a local parquet read with an explicit schema, one item per file or byte-range
/// split. A split's `start`/`length` ride the Substrait item; `(0, 0)` — the proto default —
/// is the whole-file encoding, which is why a real range is never emitted as `(0, 0)`.
fn local_files_rel(
    schema: substrait::proto::NamedStruct,
    files: &[crate::scan_paths::ScanFile],
) -> Rel {
    Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(schema),
            read_type: Some(ReadType::LocalFiles(LocalFiles {
                items: files
                    .iter()
                    .map(|file| {
                        let (start, length) = file.range.unwrap_or((0, 0));
                        FileOrFiles {
                            path_type: Some(PathType::UriFile(file.path.clone())),
                            file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                            start,
                            length,
                            ..Default::default()
                        }
                    })
                    .collect(),
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
            let mut expr_ctx = ctx.expr_context_with_slots(&input.row_tuples, &common_slots);
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
            let mut expr_ctx = ctx.expr_context_with_slots(&input.row_tuples, &common_slots);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
    }

    Ok(project_rel(input, expressions, output_tuples))
}

/// Adds a root projection over explicit fragment output expressions.
pub(crate) fn project_exprs(
    input: TranslatedRel,
    exprs: &[TExpr],
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    // Root projections evaluate over already-translated inputs, so there are no
    // scan nodes to resolve file paths for.
    let scan_paths = ScanFilePaths::default();
    let exchange_inputs = std::collections::HashMap::new();
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
        let mut expr_ctx = ctx.expr_context(&input.row_tuples);
        expressions.push(expr.translate(&mut expr_ctx)?);
    }
    // A root projection over fragment output expressions keeps the input layout.
    let row_tuples = input.row_tuples.clone();
    Ok(project_rel(input, expressions, row_tuples))
}

/// Builds a Substrait project that emits exactly `expressions`.
///
/// The emit `output_mapping` selects the projected expressions, which sit after
/// the input columns, so the base offset is the input's carried `output_width`.
/// `row_tuples` is the output row layout (the project's own tuples, which may
/// reorder or differ from the input's).
fn project_rel(
    input: TranslatedRel,
    expressions: Vec<Expression>,
    row_tuples: Vec<i32>,
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
    }
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
    }
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
        let mut expr_ctx = ctx.expr_context(&input.row_tuples);
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
