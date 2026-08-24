use starrocks_thrift::exprs::{TExpr, TExprNodeType};
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::plan_nodes::{TJoinOp, TPlan, TPlanNode, TPlanNodeType, TSortInfo};
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{LocalFiles, NamedTable, ReadType};
use substrait::proto::{
    AggregateFunction, AggregateRel, CrossRel, Expression, FetchRel, FilterRel, JoinRel,
    ProjectRel, ReadRel, Rel, RelCommon, SortField, SortRel, aggregate_rel, fetch_rel,
    function_argument, join_rel, rel, rel_common, sort_field,
};

use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, ExprContext, TranslateExpr};
use crate::scan_paths::ScanFilePaths;
use crate::type_mapper;
use crate::{ExtensionRegistry, URN_AGGREGATE, URN_ARITHMETIC, URN_BOOLEAN, URN_COMPARISON};

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
    /// Substrait extension registry shared across the whole plan.
    registry: &'a mut ExtensionRegistry,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    fn new(
        desc: &'a DescriptorTable,
        scan_paths: &'a ScanFilePaths,
        registry: &'a mut ExtensionRegistry,
    ) -> Self {
        Self {
            desc,
            scan_paths,
            registry,
        }
    }

    /// Creates an expression context for an expression over `row_tuples`.
    fn expr_context<'b>(&'b mut self, row_tuples: &'b [i32]) -> ExprContext<'b> {
        ExprContext::new(self.desc, self.registry, row_tuples)
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
fn translate_select(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
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
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    let mut ctx = PlanContext::new(desc, scan_paths, registry);
    plan.translate(&mut ctx)
}

/// Translates a one-phase `AGGREGATION_NODE` into a Substrait aggregate relation.
///
/// Only finalized single-phase aggregation is supported (run StarRocks with
/// `new_planner_agg_stage = 1`); merge/update phases would require modeling partial aggregate
/// states. The output row layout is the aggregation output tuple, whose materialized slots are
/// the grouping keys followed by the aggregate results (StarRocks allocates them in that order).
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
    if !agg.need_finalize || agg.intermediate_tuple_id != agg.output_tuple_id {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "only finalized one-phase aggregation is supported (new_planner_agg_stage=1)",
        });
    }
    let output_tuple = agg.output_tuple_id;

    let grouping_exprs = agg.grouping_exprs.as_deref().unwrap_or_default();
    let mut grouping_expressions = Vec::with_capacity(grouping_exprs.len());
    for expr in grouping_exprs {
        let mut expr_ctx = ctx.expr_context(&child.row_tuples);
        grouping_expressions.push(expr.translate(&mut expr_ctx)?);
    }

    // Aggregate output types come from the output tuple's slots, which carry the grouping keys
    // first and then one slot per aggregate function.
    let output_slots = ctx.desc.materialized_slot_ids(output_tuple)?;
    let output_width = output_slots.len();
    if output_width != grouping_expressions.len() + agg.aggregate_functions.len() {
        return Err(TranslateError::descriptor(format!(
            "AGGREGATION_NODE {} output tuple {} has {} slots for {} keys + {} aggregates",
            node.node_id,
            output_tuple,
            output_width,
            grouping_expressions.len(),
            agg.aggregate_functions.len()
        )));
    }

    // A count check alone cannot see a permuted output tuple, so also require each grouping
    // key's type to match the slot it is paired with. Compare only the type kind: the slot's
    // nullability and decimal width are allowed to differ from the key expression's.
    for (index, (expr, slot_id)) in grouping_exprs.iter().zip(&output_slots).enumerate() {
        let Some(key_type) = expr
            .nodes
            .first()
            .map(|node| type_mapper::map_type_desc(&node.type_, true))
            .transpose()?
        else {
            continue;
        };
        let slot = ctx.desc.slot(output_tuple, *slot_id)?;
        let Some(slot_type) = slot.substrait_type.as_ref() else {
            continue;
        };
        let kind_of = |ty: &substrait::proto::Type| ty.kind.as_ref().map(std::mem::discriminant);
        if kind_of(&key_type) != kind_of(slot_type) {
            return Err(TranslateError::descriptor(format!(
                "AGGREGATION_NODE {} output tuple {} slot {} does not match grouping key {}",
                node.node_id, output_tuple, slot_id, index
            )));
        }
    }

    let mut measures = Vec::with_capacity(agg.aggregate_functions.len());
    for (expr, slot_id) in agg
        .aggregate_functions
        .iter()
        .zip(&output_slots[grouping_expressions.len()..])
    {
        let mut expr_ctx = ctx.expr_context(&child.row_tuples);
        let call = expr_translator::aggregate_call(expr, &mut expr_ctx)?;
        // The GPU ungrouped-aggregate operator rejects every distinct aggregate, so a
        // grouping-free DISTINCT measure would translate fine and then fail at execution.
        if call.distinct && grouping_expressions.is_empty() {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "distinct aggregates without grouping keys are not supported",
            });
        }
        let output_type = ctx
            .desc
            .slot(output_tuple, *slot_id)?
            .substrait_type
            .clone()
            .ok_or(TranslateError::MissingField {
                context: "aggregate output slot",
                field: "slotType",
            })?;
        // `count` lives in the generic aggregate extension; sum/avg/min/max are declared by
        // the arithmetic extension.
        let urn = if call.name == "count" {
            URN_AGGREGATE
        } else {
            URN_ARITHMETIC
        };
        let anchor = ctx.registry.register_function(urn, &call.name);
        measures.push(aggregate_rel::Measure {
            measure: Some(AggregateFunction {
                function_reference: anchor,
                arguments: call
                    .arguments
                    .into_iter()
                    .map(|expr| substrait::proto::FunctionArgument {
                        arg_type: Some(function_argument::ArgType::Value(expr)),
                    })
                    .collect(),
                output_type: Some(output_type),
                invocation: if call.distinct {
                    substrait::proto::aggregate_function::AggregationInvocation::Distinct as i32
                } else {
                    substrait::proto::aggregate_function::AggregationInvocation::All as i32
                },
                ..Default::default()
            }),
            filter: None,
        });
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
    // Node conjuncts evaluate over the aggregation output (HAVING predicates).
    apply_conjuncts(aggregated, node, ctx)
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
    // Validated before the conjuncts so an unsupported op is reported as such, rather than as the
    // missing conjuncts an anti join arrives with once the FE has folded its predicate away.
    //
    // Anti joins (from NOT IN / NOT EXISTS rewrites) are not translated: DuckDB's Substrait
    // consumer has no left-anti conversion, so an emitted plan would fail downstream anyway.
    let (join_type, semi) = match join.join_op {
        TJoinOp::INNER_JOIN => (join_rel::JoinType::Inner, false),
        TJoinOp::LEFT_OUTER_JOIN => (join_rel::JoinType::Left, false),
        TJoinOp::RIGHT_OUTER_JOIN => (join_rel::JoinType::Right, false),
        TJoinOp::FULL_OUTER_JOIN => (join_rel::JoinType::Outer, false),
        TJoinOp::LEFT_SEMI_JOIN => (join_rel::JoinType::LeftSemi, true),
        _ => {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "hash join type is unsupported",
            });
        }
    };

    let mut children = children.into_iter();
    let left = children.next().unwrap();
    let right = children.next().unwrap();

    let combined_tuples = [left.row_tuples.as_slice(), right.row_tuples.as_slice()].concat();
    let mut conditions = Vec::new();
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

    // Semi joins emit only the probe-side row; other joins emit probe then build columns.
    let (row_tuples, output_width) = if semi {
        (left.row_tuples.clone(), left.output_width)
    } else {
        (combined_tuples, left.output_width + right.output_width)
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
    // Node conjuncts are post-join predicates over the join's output row.
    apply_conjuncts(joined, node, ctx)
}

/// Translates a `NESTLOOP_JOIN_NODE` into a Substrait cross product with the join conjuncts as a
/// filter on top. Only inner/cross nested-loop joins are supported.
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
    let mut children = children.into_iter();
    let left = children.next().unwrap();
    let right = children.next().unwrap();
    let row_tuples = [left.row_tuples.as_slice(), right.row_tuples.as_slice()].concat();
    let output_width = left.output_width + right.output_width;

    let conjuncts = join.join_conjuncts.as_deref().unwrap_or_default();
    let mut conditions = Vec::with_capacity(conjuncts.len());
    for expr in conjuncts {
        let mut expr_ctx = ctx.expr_context(&row_tuples);
        conditions.push(expr.translate(&mut expr_ctx)?);
    }
    // A conjunct-free cross product is rejected: the GPU physical planner has no cross-product
    // operator, so the plan would translate and then fail at execution.
    let condition = and_conditions(conditions, ctx).ok_or(TranslateError::UnsupportedPlanNode {
        node_id: node.node_id,
        node_type: node.node_type,
        reason: "cross joins without join conjuncts are not supported",
    })?;
    if !conjuncts
        .iter()
        .any(|conjunct| becomes_join_condition(conjunct, &left.row_tuples, &right.row_tuples))
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "nested-loop joins need a comparison between the two inputs",
        });
    }

    let cross = Rel {
        rel_type: Some(rel::RelType::Cross(Box::new(CrossRel {
            left: Some(Box::new(left.rel)),
            right: Some(Box::new(right.rel)),
            ..Default::default()
        }))),
    };
    let filtered = TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(cross)),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
    };
    // Node conjuncts are post-join predicates over the join's output row.
    apply_conjuncts(filtered, node, ctx)
}

/// Returns whether DuckDB will lift `conjunct` out of the filter above the cross product and into
/// a join condition.
///
/// `FilterPushdown::PushdownCrossProduct` splits that filter by side: a predicate reading only one
/// input is pushed into that input, and only predicates reading both become join conditions —
/// which `ExtractJoinConditions` then keeps as conditions when the predicate is a comparison, and
/// demotes to an any-join otherwise. With no such predicate the plan lowers to a bare cross
/// product or an any-join, and the GPU planner implements neither
/// (`sirius_physical_plan_generator.cpp`), so it would fail at execution instead of here.
///
/// This approximates the rule from the one side that matters: a comparison whose own operands each
/// read both inputs (`a.x + b.y < 10`) also becomes an any-join, and is accepted here.
fn becomes_join_condition(conjunct: &TExpr, left_tuples: &[i32], right_tuples: &[i32]) -> bool {
    conjunct
        .nodes
        .first()
        .is_some_and(|root| root.node_type == TExprNodeType::BINARY_PRED)
        && reads_tuples(conjunct, left_tuples)
        && reads_tuples(conjunct, right_tuples)
}

/// Returns whether any slot reference in `expr` belongs to one of `tuples`.
fn reads_tuples(expr: &TExpr, tuples: &[i32]) -> bool {
    expr.nodes.iter().any(|node| {
        node.slot_ref
            .as_ref()
            .is_some_and(|slot_ref| tuples.contains(&slot_ref.tuple_id))
    })
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

    let mut expressions = Vec::new();
    for &tuple_id in &output_tuples {
        for slot_id in ctx.desc.materialized_slot_ids(tuple_id)? {
            let expr = slot_map.get(&slot_id).ok_or_else(|| {
                TranslateError::descriptor(format!(
                    "PROJECT_NODE node {} missing slot_map expression for slot {}",
                    node.node_id, slot_id
                ))
            })?;
            let mut expr_ctx = ctx.expr_context(&child.row_tuples);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
    }

    Ok(project_rel(child, expressions, output_tuples))
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
    let mut ctx = PlanContext::new(desc, &scan_paths, registry);
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
