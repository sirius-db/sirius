//! Doris `TPlan` (one fragment's plan nodes) → Substrait relation tree.
//!
//! `TPlan.nodes` is a flat preorder list; [`PlanNodeCursor`] rebuilds the tree by consuming
//! exactly `num_children` nodes per parent and rejects a list that is not consumed exactly
//! once (see the crate docs).
//!
//! # How a Doris node is shaped (4.1.4, Nereids)
//!
//! Every node is "an operation, then bookkeeping", in this order:
//!
//! 1. the operation itself — scan, exchange read, join, aggregate, sort — produces a row whose
//!    layout is a list of tuple ids (`row_tuples`, or the join's intermediate tuple);
//! 2. `conjuncts` filter that row (post-join predicates, `HAVING`, scan predicates);
//! 3. `limit` (with the sort/exchange `offset`) truncates it;
//! 4. `intermediate_projections_list[i]` → `intermediate_output_tuple_id_list[i]` project it
//!    stage by stage, then `projections` → `output_tuple_id`; the node's output layout is the
//!    last tuple produced. Projections do not change the row count, so they commute with the
//!    limit.
//!
//! Tuples are positional aliases: a node whose `row_tuples` differs from its child's layout
//! (sort, exchange, join intermediate tuple) re-labels the same columns under a new tuple id,
//! and the expressions that follow reference the new id. [`TranslatedRel`] therefore carries
//! the layout as tuple ids plus, where a layout cannot be expressed as whole tuples (a semi
//! join keeps one side of the intermediate tuple), a slot-to-column override map.
//!
//! # Node table
//!
//! | Doris node | Substrait |
//! |---|---|
//! | `FILE_SCAN_NODE` | `ReadRel` over `local_files` parquet, schema = destination tuple by column name |
//! | `EXCHANGE_NODE` | `ReadRel` over the named table `sirius_stream_<node_id>` (+ `SortRel` for a merging exchange); the single-plan stitcher replaces it with the sender's subtree |
//! | `HASH_JOIN_NODE` | `JoinRel`; anti joins as an outer join + `is_null` filter (the consumer has no anti type), `RIGHT_*` by swapping sides, `NULL_AWARE_LEFT_ANTI` as a mark join + `not(mark)` |
//! | `CROSS_JOIN_NODE` (nested loop, inner/cross) | equality `JoinRel` on synthetic constant keys + `FilterRel` (Sirius has no cross-product operator) |
//! | `AGGREGATION_NODE` (finalized, single phase) | `AggregateRel` + a `ProjectRel` casting measures to the Doris output types; `agg_sort_info_by_group_key` + `limit` → `SortRel` + `FetchRel` |
//! | `SORT_NODE` | `SortRel` (+ `FetchRel` for top-N) |
//!
//! Update-phase (`need_finalize=false`) and merge-phase aggregates are rejected here: a single
//! fragment cannot produce or consume partial states. The single-plan stitcher rewrites a
//! two-phase pair into one finalized node before translation.
//!
//! # Types at tuple boundaries
//!
//! DuckDB re-derives expression types (decimal `avg` → `DOUBLE`, `year` → `BIGINT`,
//! integer `sum` → `HUGEINT`), while every Doris slot declares the type the FE decided on
//! (G-19). Whenever a projection or measure lands in a slot, the expression is cast to the
//! slot's type unless it is a plain slot reference, so the plan's row types are Doris'.

use doris_thrift::exprs::{TExpr, TExprNodeType};
use doris_thrift::opcodes::TExprOpcode;
use doris_thrift::plan_nodes::{TJoinOp, TPlan, TPlanNode, TPlanNodeType, TSortInfo};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{LocalFiles, NamedTable, ReadType};
use substrait::proto::r#type;
use substrait::proto::{
    AggregateRel, Expression, FetchRel, FilterRel, JoinRel, ProjectRel, ReadRel, Rel, RelCommon,
    SortField, SortRel, Type, aggregate_rel, expression, fetch_rel, join_rel, rel, rel_common,
    sort_field,
};

use crate::descriptor_table::{DescriptorTable, SlotInfo};
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, AggregateCall, ExprContext, SlotOverrides};
use crate::scan_ranges::ScanRanges;
use crate::{ExtensionRegistry, URN_BOOLEAN, URN_COMPARISON};

/// Name prefix of the named table an `EXCHANGE_NODE` reads from.
pub const STREAM_TABLE_PREFIX: &str = "sirius_stream_";

/// A translated subtree plus the Doris row layout it emits.
#[derive(Clone, Debug)]
pub struct TranslatedRel {
    /// Substrait relation for the subtree.
    pub rel: Rel,
    /// Tuple ids describing the output row, concatenated in order; a `SLOT_REF` into one of
    /// them resolves through `DescriptorTable::slot_global_index`.
    pub row_tuples: Vec<i32>,
    /// Number of columns the relation emits. Carried as an invariant so parents can compute
    /// emit offsets without walking `rel`; every constructor here sets the true width.
    pub output_width: usize,
    /// Slots resolved to a column directly, ahead of `row_tuples`: the columns of a join's
    /// intermediate tuple when the join kept only one side of it.
    pub overrides: SlotOverrides,
}

impl TranslatedRel {
    /// A relation whose layout is exactly `row_tuples`.
    fn new(rel: Rel, row_tuples: Vec<i32>, output_width: usize) -> Self {
        Self {
            rel,
            row_tuples,
            output_width,
            overrides: SlotOverrides::new(),
        }
    }
}

/// State shared by the plan-node translators.
pub struct PlanContext<'a> {
    /// Descriptor lookups.
    desc: &'a DescriptorTable,
    /// Parquet paths per scan node.
    scan_ranges: &'a ScanRanges,
    /// Extension registry shared across the plan.
    registry: &'a mut ExtensionRegistry,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    pub fn new(
        desc: &'a DescriptorTable,
        scan_ranges: &'a ScanRanges,
        registry: &'a mut ExtensionRegistry,
    ) -> Self {
        Self {
            desc,
            scan_ranges,
            registry,
        }
    }

    /// Translates `expr` over a relation's layout.
    fn expr(&mut self, expr: &TExpr, input: &TranslatedRel) -> Result<Expression> {
        let mut ctx = ExprContext::with_slot_overrides(
            self.desc,
            self.registry,
            &input.row_tuples,
            &input.overrides,
        );
        expr_translator::translate_expr(expr, &mut ctx)
    }

    /// Translates `expr` over an explicit layout.
    fn expr_over(
        &mut self,
        expr: &TExpr,
        row_tuples: &[i32],
        overrides: &SlotOverrides,
    ) -> Result<Expression> {
        let mut ctx =
            ExprContext::with_slot_overrides(self.desc, self.registry, row_tuples, overrides);
        expr_translator::translate_expr(expr, &mut ctx)
    }

    /// Decomposes an aggregate call over a relation's layout.
    fn aggregate(&mut self, expr: &TExpr, input: &TranslatedRel) -> Result<AggregateCall> {
        let mut ctx = ExprContext::with_slot_overrides(
            self.desc,
            self.registry,
            &input.row_tuples,
            &input.overrides,
        );
        expr_translator::aggregate_call(expr, &mut ctx)
    }

    /// Registers a boolean/comparison function and applies it.
    fn function(&mut self, urn: &str, name: &str, args: Vec<Expression>) -> Expression {
        let anchor = self.registry.register_function(urn, name);
        expr_translator::scalar_function(anchor, args, bool_type())
    }
}

/// Translates a fragment's flat preorder plan into a relation tree.
pub fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    scan_ranges: &ScanRanges,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    if plan.nodes.is_empty() {
        return Err(TranslateError::malformed("TPlan.nodes is empty"));
    }
    let mut ctx = PlanContext::new(desc, scan_ranges, registry);
    let mut cursor = PlanNodeCursor::new(&plan.nodes);
    let translated = cursor.translate_next(&mut ctx)?;
    cursor.ensure_consumed()?;
    Ok(translated)
}

/// Cursor over the flat preorder `TPlan.nodes` list.
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

    /// Translates the next node and its subtree.
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

    /// Verifies that the whole node list was consumed.
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

/// What a node-specific translator already took care of, so the generic tail does not
/// apply it twice.
#[derive(Clone, Copy, Default)]
struct Consumed {
    /// The node's `conjuncts` were applied (they had to sit before a node-specific step).
    conjuncts: bool,
    /// The node's `limit` (and offset) was applied.
    limit: bool,
}

/// Routes a node to its translator, then applies the generic tail: conjuncts, limit,
/// projection chain.
fn translate_plan_node(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let (translated, consumed) = match node.node_type {
        TPlanNodeType::FILE_SCAN_NODE => translate_file_scan(node, children, ctx),
        TPlanNodeType::EXCHANGE_NODE => translate_exchange(node, children, ctx),
        TPlanNodeType::HASH_JOIN_NODE => translate_hash_join(node, children, ctx),
        TPlanNodeType::CROSS_JOIN_NODE => translate_nested_loop_join(node, children, ctx),
        TPlanNodeType::AGGREGATION_NODE => translate_aggregation(node, children, ctx),
        TPlanNodeType::SORT_NODE => translate_sort(node, children, ctx),
        TPlanNodeType::MATERIALIZATION_NODE
        | TPlanNodeType::UNION_NODE
        | TPlanNodeType::INTERSECT_NODE
        | TPlanNodeType::EXCEPT_NODE
        | TPlanNodeType::ANALYTIC_EVAL_NODE
        | TPlanNodeType::REPEAT_NODE
        | TPlanNodeType::ASSERT_NUM_ROWS_NODE
        | TPlanNodeType::SELECT_NODE
        | TPlanNodeType::EMPTY_SET_NODE
        | TPlanNodeType::TABLE_FUNCTION_NODE
        | TPlanNodeType::PARTITION_SORT_NODE
        | TPlanNodeType::OLAP_SCAN_NODE
        | TPlanNodeType::JDBC_SCAN_NODE
        | TPlanNodeType::SCHEMA_SCAN_NODE
        | TPlanNodeType::DATA_GEN_SCAN_NODE
        | TPlanNodeType::REC_CTE_NODE
        | TPlanNodeType::REC_CTE_SCAN_NODE => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "plan node type is not translated (see the node table in node_translator.rs)",
        }),
        _ => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "plan node type is outside the supported slice",
        }),
    }?;
    let translated = if consumed.conjuncts {
        translated
    } else {
        apply_conjuncts(translated, node, ctx)?
    };
    let translated = if consumed.limit {
        translated
    } else {
        apply_limit(translated, node.limit, 0)
    };
    apply_projections(translated, node, ctx)
}

/// `FILE_SCAN_NODE` → parquet read of the destination tuple's columns by name.
fn translate_file_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<(TranslatedRel, Consumed)> {
    expect_children(node, &children, 0)?;
    let tuple_id = node
        .file_scan_node
        .as_ref()
        .and_then(|scan| scan.tuple_id)
        .or_else(|| node.row_tuples.first().copied())
        .ok_or(TranslateError::MissingField {
            context: "FILE_SCAN_NODE",
            field: "tuple_id",
        })?;
    if node.row_tuples != [tuple_id] {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "scan row layout is not exactly its destination tuple",
        });
    }
    let paths = ctx.scan_ranges.for_node(node.node_id);
    if paths.is_empty() {
        return Err(TranslateError::UnsupportedScanRange {
            node_id: node.node_id,
            reason: "no parquet ranges were assigned to this scan on this instance",
        });
    }
    // The consumer projects the file's columns by the schema's names, so every slot has to
    // carry the file column's name (a planner-derived `col_<id>` would not resolve).
    let slots = ctx.desc.tuple_slots(tuple_id)?;
    if let Some(unnamed) = slots.iter().find(|slot| slot.col_name.is_empty()) {
        return Err(TranslateError::descriptor(format!(
            "scan node {} destination tuple {} slot {} has no column name",
            node.node_id, tuple_id, unnamed.slot_id
        )));
    }
    let width = slots.len();
    let rel = Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(ctx.desc.named_struct(tuple_id)?),
            read_type: Some(ReadType::LocalFiles(LocalFiles {
                items: paths
                    .iter()
                    .map(|path| FileOrFiles {
                        path_type: Some(PathType::UriFile(path.clone())),
                        file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            })),
            ..Default::default()
        }))),
    };
    Ok((
        TranslatedRel::new(rel, vec![tuple_id], width),
        Consumed::default(),
    ))
}

/// `EXCHANGE_NODE` → named-table read of the stream the sender fragment feeds, sorted for a
/// merging exchange.
fn translate_exchange(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<(TranslatedRel, Consumed)> {
    expect_children(node, &children, 0)?;
    let exchange = node
        .exchange_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "EXCHANGE_NODE",
            field: "exchange_node",
        })?;
    if node.row_tuples.is_empty() {
        return Err(TranslateError::MissingField {
            context: "EXCHANGE_NODE",
            field: "row_tuples",
        });
    }
    if exchange.input_row_tuples != node.row_tuples {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "exchange input row layout differs from its output layout",
        });
    }
    let width = ctx.desc.row_width(&node.row_tuples)?;
    let rel = Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(ctx.desc.named_struct_for_tuples(&node.row_tuples)?),
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec![format!("{STREAM_TABLE_PREFIX}{}", node.node_id)],
                ..Default::default()
            })),
            ..Default::default()
        }))),
    };
    let mut translated = TranslatedRel::new(rel, node.row_tuples.clone(), width);
    if let Some(sort_info) = &exchange.sort_info {
        let sorts = sort_fields(sort_info, &translated, ctx)?;
        translated = sort_rel(translated, sorts);
    }
    let translated = apply_limit(translated, node.limit, exchange.offset.unwrap_or(0));
    Ok((
        translated,
        Consumed {
            conjuncts: false,
            limit: true,
        },
    ))
}

/// `SORT_NODE` → sort over the child's row, re-labelled as the sort tuple, plus top-N fetch.
fn translate_sort(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<(TranslatedRel, Consumed)> {
    expect_children(node, &children, 1)?;
    let child = children.into_iter().next().unwrap();
    let sort = node
        .sort_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "SORT_NODE",
            field: "sort_node",
        })?;
    if sort.sort_info.use_two_phase_read == Some(true) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "two-phase (row-id) top-N reads are not supported",
        });
    }
    if has_conjuncts(node) {
        // Doris' sorter never evaluates predicates (the FE plans them below the sort), so
        // their position relative to the limit would be a guess.
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "SORT_NODE with conjuncts is not supported",
        });
    }
    let input = match sort
        .sort_info
        .sort_tuple_slot_exprs
        .as_ref()
        .filter(|exprs| !exprs.is_empty())
    {
        // Materialize the sort tuple first: one expression per slot, over the child's row.
        Some(slot_exprs) => {
            let expressions = slot_exprs
                .iter()
                .map(|expr| ctx.expr(expr, &child))
                .collect::<Result<Vec<_>>>()?;
            project_into_tuples(child, expressions, node.row_tuples.clone(), ctx, node)?
        }
        // Without them the sort tuple is the child's row under a new id.
        None => relabel(child, &node.row_tuples, ctx, node)?,
    };
    let sorts = sort_fields(&sort.sort_info, &input, ctx)?;
    let sorted = sort_rel(input, sorts);
    let limited = apply_limit(sorted, node.limit, sort.offset.unwrap_or(0));
    Ok((
        limited,
        Consumed {
            conjuncts: false,
            limit: true,
        },
    ))
}

/// `AGGREGATION_NODE` (finalized, single phase) → aggregate + output-type casts (+ `HAVING`,
/// + sort-by-group-key and limit).
fn translate_aggregation(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<(TranslatedRel, Consumed)> {
    expect_children(node, &children, 1)?;
    let child = children.into_iter().next().unwrap();
    let agg = node.agg_node.as_ref().ok_or(TranslateError::MissingField {
        context: "AGGREGATION_NODE",
        field: "agg_node",
    })?;
    if !agg.need_finalize {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "update-phase aggregate (need_finalize=false) emits partial states; only a \
                     stitched single plan can run it",
        });
    }
    if agg.intermediate_tuple_id != agg.output_tuple_id {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "aggregate with distinct intermediate and output tuples is not supported",
        });
    }
    if agg
        .agg_sort_infos
        .iter()
        .flatten()
        .any(|info| !info.ordering_exprs.is_empty())
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "ordered aggregate functions (agg_sort_infos) are not supported",
        });
    }
    let output_tuple = agg.output_tuple_id;
    if node.row_tuples != [output_tuple] {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "aggregate row layout is not exactly its output tuple",
        });
    }

    let grouping_exprs = agg.grouping_exprs.as_deref().unwrap_or_default();
    let grouping_expressions = grouping_exprs
        .iter()
        .map(|expr| ctx.expr(expr, &child))
        .collect::<Result<Vec<_>>>()?;
    let calls = agg
        .aggregate_functions
        .iter()
        .map(|expr| ctx.aggregate(expr, &child))
        .collect::<Result<Vec<_>>>()?;
    if calls.iter().any(|call| call.is_merge) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "merge-phase aggregate consumes partial states; only a stitched single \
                     plan can run it",
        });
    }
    // The output tuple lays out the grouping keys first, then one slot per aggregate.
    let output_slots = ctx.desc.tuple_slots(output_tuple)?;
    let output_width = output_slots.len();
    if output_width != grouping_expressions.len() + calls.len() {
        return Err(TranslateError::descriptor(format!(
            "AGGREGATION_NODE {} output tuple {} has {} slots for {} keys + {} aggregates",
            node.node_id,
            output_tuple,
            output_width,
            grouping_expressions.len(),
            calls.len()
        )));
    }
    let measure_types: Vec<Type> = output_slots[grouping_expressions.len()..]
        .iter()
        .map(|slot| slot.substrait_type.clone())
        .collect();

    let measures = calls
        .iter()
        .map(|call| {
            let mut expr_ctx = ExprContext::with_slot_overrides(
                ctx.desc,
                ctx.registry,
                &child.row_tuples,
                &child.overrides,
            );
            aggregate_rel::Measure {
                measure: Some(expr_translator::aggregate_function(call, &mut expr_ctx)),
                filter: None,
            }
        })
        .collect();
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
    let key_count = grouping_expressions.len();
    let aggregated = Rel {
        rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(child.rel)),
            groupings,
            measures,
            grouping_expressions,
            ..Default::default()
        }))),
    };
    // Keys pass through; every measure is cast to its Doris slot type (G-19: DuckDB may have
    // computed `avg` as DOUBLE or an integer `sum` as HUGEINT).
    let mut expressions: Vec<Expression> = (0..key_count)
        .map(expr_translator::field_reference)
        .collect();
    for (index, measure_type) in measure_types.into_iter().enumerate() {
        expressions.push(expr_translator::cast(
            expr_translator::field_reference(key_count + index),
            measure_type,
        ));
    }
    let translated = project_rel(
        TranslatedRel::new(aggregated, Vec::new(), output_width),
        expressions,
        vec![output_tuple],
    );

    // HAVING, then the group-key order the FE asked for (a limit on an aggregate means
    // "first N groups in that order", not "any N groups"), then the limit.
    let translated = apply_conjuncts(translated, node, ctx)?;
    let translated = match &agg.agg_sort_info_by_group_key {
        Some(sort_info) if !sort_info.ordering_exprs.is_empty() => {
            let sorts = sort_fields(sort_info, &translated, ctx)?;
            sort_rel(translated, sorts)
        }
        _ => translated,
    };
    let translated = apply_limit(translated, node.limit, 0);
    Ok((
        translated,
        Consumed {
            conjuncts: true,
            limit: true,
        },
    ))
}

/// Which Doris child a join side of the Substrait join comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    /// Doris' left (probe) child.
    Left,
    /// Doris' right (build) child.
    Right,
}

/// How a hash join op is lowered.
struct JoinLowering {
    /// Substrait join type.
    join_type: join_rel::JoinType,
    /// Whether Doris' right child becomes the Substrait left input.
    swap: bool,
    /// Which Doris side(s) the join output keeps.
    output: JoinOutput,
}

/// The Doris side(s) a join emits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum JoinOutput {
    /// Both children, left then right (inner/outer).
    Both,
    /// Only one child (semi/anti joins).
    One(Side),
    /// One child plus the lowering's own bookkeeping (anti via outer join, mark join).
    AntiOuter(Side),
    /// Left child via a mark join and `not(mark)`.
    NullAwareAnti,
}

/// `HASH_JOIN_NODE` → join relation whose output is the Doris intermediate tuple (or one side
/// of it).
fn translate_hash_join(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<(TranslatedRel, Consumed)> {
    expect_children(node, &children, 2)?;
    let join = node
        .hash_join_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "HASH_JOIN_NODE",
            field: "hash_join_node",
        })?;
    if join.is_mark == Some(true)
        || join
            .mark_join_conjuncts
            .as_ref()
            .is_some_and(|conjuncts| !conjuncts.is_empty())
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "mark joins (IN/EXISTS inside a disjunction) are not supported",
        });
    }
    if join.match_condition.is_some() {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "joins with a match condition are not supported",
        });
    }
    let lowering = match join.join_op {
        TJoinOp::INNER_JOIN => JoinLowering {
            join_type: join_rel::JoinType::Inner,
            swap: false,
            output: JoinOutput::Both,
        },
        TJoinOp::LEFT_OUTER_JOIN => JoinLowering {
            join_type: join_rel::JoinType::Left,
            swap: false,
            output: JoinOutput::Both,
        },
        TJoinOp::RIGHT_OUTER_JOIN => JoinLowering {
            join_type: join_rel::JoinType::Right,
            swap: false,
            output: JoinOutput::Both,
        },
        TJoinOp::FULL_OUTER_JOIN => JoinLowering {
            join_type: join_rel::JoinType::Outer,
            swap: false,
            output: JoinOutput::Both,
        },
        TJoinOp::LEFT_SEMI_JOIN => JoinLowering {
            join_type: join_rel::JoinType::LeftSemi,
            swap: false,
            output: JoinOutput::One(Side::Left),
        },
        // A right semi join keeps the build side; with the sides swapped it is a left semi.
        TJoinOp::RIGHT_SEMI_JOIN => JoinLowering {
            join_type: join_rel::JoinType::LeftSemi,
            swap: true,
            output: JoinOutput::One(Side::Right),
        },
        // The consumer has no anti join: outer join, keep the rows the other side padded.
        TJoinOp::LEFT_ANTI_JOIN => JoinLowering {
            join_type: join_rel::JoinType::Left,
            swap: false,
            output: JoinOutput::AntiOuter(Side::Left),
        },
        TJoinOp::RIGHT_ANTI_JOIN => JoinLowering {
            join_type: join_rel::JoinType::Left,
            swap: true,
            output: JoinOutput::AntiOuter(Side::Right),
        },
        TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN => JoinLowering {
            join_type: join_rel::JoinType::LeftMark,
            swap: false,
            output: JoinOutput::NullAwareAnti,
        },
        _ => {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "hash join type is unsupported",
            });
        }
    };
    // `LeftMark + NOT(mark)` is the null-aware anti join only when the marker's NULL-ness is
    // decided per probe row against one key: both DuckDB and the GPU path set one global
    // "build has NULL" flag, which is exact for a single equality key and nothing else. With
    // a second key or another predicate a definitely-false row would be reported UNKNOWN and
    // dropped.
    if lowering.output == JoinOutput::NullAwareAnti
        && (join.eq_join_conjuncts.len() != 1
            || join
                .other_join_conjuncts
                .as_ref()
                .is_some_and(|conjuncts| !conjuncts.is_empty()))
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "null-aware left anti join with several keys or extra predicates",
        });
    }

    let mut children = children.into_iter();
    let doris_left = children.next().unwrap();
    let doris_right = children.next().unwrap();
    let intermediate = intermediate_tuple(node, join.vintermediate_tuple_id_list.as_deref(), ctx)?;
    let join_row = JoinRow::new(
        &doris_left,
        &doris_right,
        &intermediate,
        lowering.swap,
        node,
    )?;

    // Equality keys reference the children's own tuples; other conjuncts reference the
    // intermediate tuple. Both are evaluated over the Substrait join row.
    let mut conditions = Vec::new();
    let mut first_key = None;
    for eq in &join.eq_join_conjuncts {
        let name = match eq.opcode {
            None | Some(TExprOpcode::EQ) => "equal",
            Some(TExprOpcode::EQ_FOR_NULL) => "is_not_distinct_from",
            Some(_) => {
                return Err(TranslateError::UnsupportedPlanNode {
                    node_id: node.node_id,
                    node_type: node.node_type,
                    reason: "only equality (and null-safe equality) join keys are supported",
                });
            }
        };
        let left_key = ctx.expr_over(&eq.left, &join_row.row_tuples, &join_row.child_overrides)?;
        let right_key =
            ctx.expr_over(&eq.right, &join_row.row_tuples, &join_row.child_overrides)?;
        if first_key.is_none() {
            first_key = Some((left_key.clone(), right_key.clone()));
        }
        conditions.push(ctx.function(URN_COMPARISON, name, vec![left_key, right_key]));
    }
    for expr in join.other_join_conjuncts.as_deref().unwrap_or_default() {
        conditions.push(ctx.expr_over(
            expr,
            &join_row.row_tuples,
            &join_row.intermediate_overrides,
        )?);
    }
    if conditions.is_empty()
        && let Some(expr) = &join.vother_join_conjunct
    {
        conditions.push(ctx.expr_over(
            expr,
            &join_row.row_tuples,
            &join_row.intermediate_overrides,
        )?);
    }
    let condition = and_conditions(conditions, ctx).ok_or(TranslateError::UnsupportedPlanNode {
        node_id: node.node_id,
        node_type: node.node_type,
        reason: "hash join without join conjuncts",
    })?;

    let (substrait_left, substrait_right) = if lowering.swap {
        (doris_right, doris_left)
    } else {
        (doris_left, doris_right)
    };
    let left_width = substrait_left.output_width;
    let right_width = substrait_right.output_width;
    let joined = Rel {
        rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(substrait_left.rel)),
            right: Some(Box::new(substrait_right.rel)),
            expression: Some(Box::new(condition)),
            r#type: lowering.join_type as i32,
            ..Default::default()
        }))),
    };

    let translated = match lowering.output {
        JoinOutput::Both => {
            let layout = join_row.output_layout(JoinOutput::Both);
            layout.attach(joined, left_width + right_width)
        }
        JoinOutput::One(side) => {
            // A semi join emits the Substrait left input only (the kept Doris side).
            join_row
                .output_layout(JoinOutput::One(side))
                .attach(joined, left_width)
        }
        JoinOutput::AntiOuter(side) => {
            // The padded side's key is NULL exactly for the rows without a match. The key
            // must propagate NULL (a column, or a cast of one) for that to hold.
            let (left_key, right_key) = first_key
                .ok_or_else(|| TranslateError::malformed("anti join has no equality key"))?;
            let other_key = if lowering.swap { left_key } else { right_key };
            require_column_key(node, &other_key)?;
            let condition = ctx.function(URN_COMPARISON, "is_null", vec![other_key]);
            let filtered = filter_rel(joined, condition);
            let kept = emit_columns(filtered, (0..left_width as i32).collect());
            join_row
                .output_layout(JoinOutput::One(side))
                .attach(kept, left_width)
        }
        JoinOutput::NullAwareAnti => {
            let marker = expr_translator::field_reference(left_width);
            let condition = ctx.function(URN_BOOLEAN, "not", vec![marker]);
            let filtered = filter_rel(joined, condition);
            let kept = emit_columns(filtered, (0..left_width as i32).collect());
            join_row
                .output_layout(JoinOutput::One(Side::Left))
                .attach(kept, left_width)
        }
    };
    Ok((translated, Consumed::default()))
}

/// `CROSS_JOIN_NODE` (nested loop) → inner join on synthetic constant keys, then the join
/// predicates as a filter over the intermediate tuple.
fn translate_nested_loop_join(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<(TranslatedRel, Consumed)> {
    expect_children(node, &children, 2)?;
    let join = node
        .nested_loop_join_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "CROSS_JOIN_NODE",
            field: "nested_loop_join_node",
        })?;
    if !matches!(join.join_op, TJoinOp::INNER_JOIN | TJoinOp::CROSS_JOIN) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "only inner/cross nested-loop joins are supported",
        });
    }
    if join.is_mark == Some(true)
        || join
            .mark_join_conjuncts
            .as_ref()
            .is_some_and(|conjuncts| !conjuncts.is_empty())
        || join.is_output_left_side_only == Some(true)
    {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "mark or left-side-only nested-loop joins are not supported",
        });
    }
    let mut children = children.into_iter();
    let left = children.next().unwrap();
    let right = children.next().unwrap();
    let intermediate = intermediate_tuple(node, join.vintermediate_tuple_id_list.as_deref(), ctx)?;
    let join_row = JoinRow::new(&left, &right, &intermediate, false, node)?;
    let left_width = left.output_width;
    let right_width = right.output_width;

    // Nothing here bounds the product's size: the FE has no statistics for a TVF scan, and
    // the executor is the one that knows the real row counts. The corpus' cross joins are
    // scalar-subquery rewrites (one row on one side) and Q7's nation pair.
    let left = append_project(left, i32_literal(1));
    let right = append_project(right, i32_literal(1));
    let condition = ctx.function(
        URN_COMPARISON,
        "equal",
        vec![
            expr_translator::field_reference(left_width),
            expr_translator::field_reference(left.output_width + right_width),
        ],
    );
    let joined = Rel {
        rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(left.rel)),
            right: Some(Box::new(right.rel)),
            expression: Some(Box::new(condition)),
            r#type: join_rel::JoinType::Inner as i32,
            ..Default::default()
        }))),
    };
    // Drop the two synthetic keys; the remaining columns are the intermediate tuple.
    let mut mapping = (0..left_width as i32).collect::<Vec<_>>();
    mapping.extend(left.output_width as i32..(left.output_width + right_width) as i32);
    let product = join_row
        .output_layout(JoinOutput::Both)
        .attach(emit_columns(joined, mapping), left_width + right_width);

    let conjuncts = match join
        .join_conjuncts
        .as_deref()
        .filter(|conjuncts| !conjuncts.is_empty())
    {
        Some(conjuncts) => conjuncts.to_vec(),
        None => join.vjoin_conjunct.iter().cloned().collect(),
    };
    let conditions = conjuncts
        .iter()
        .map(|expr| ctx.expr(expr, &product))
        .collect::<Result<Vec<_>>>()?;
    let translated = match and_conditions(conditions, ctx) {
        Some(condition) => filter_translated(product, condition),
        None => product,
    };
    Ok((translated, Consumed::default()))
}

/// The join's intermediate tuple: the slots of Doris' left child then its right child (or one
/// side only), under new ids.
struct IntermediateTuple {
    /// Tuple id.
    id: i32,
    /// Slot ids in order.
    slot_ids: Vec<i32>,
}

/// Resolves a join's `vintermediate_tuple_id_list` (exactly one tuple).
fn intermediate_tuple(
    node: &TPlanNode,
    list: Option<&[i32]>,
    ctx: &mut PlanContext<'_>,
) -> Result<IntermediateTuple> {
    let id = match list {
        Some([id]) => *id,
        Some(_) | None => {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "join must declare exactly one intermediate tuple",
            });
        }
    };
    Ok(IntermediateTuple {
        id,
        slot_ids: ctx.desc.tuple(id)?.slot_ids.clone(),
    })
}

/// Which sides the intermediate tuple covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum IntermediateKind {
    /// Left slots then right slots.
    Both,
    /// One side only.
    One(Side),
}

/// Column bookkeeping for one join: where each Doris child and each intermediate slot sits in
/// the Substrait join row, and how the kept side is described afterwards.
struct JoinRow {
    /// Layout of the Substrait join row (left input's tuples, then right input's).
    row_tuples: Vec<i32>,
    /// Overrides of the children's own layouts, offset into the join row.
    child_overrides: SlotOverrides,
    /// Overrides for the intermediate tuple's slots, into the join row.
    intermediate_overrides: SlotOverrides,
    /// Intermediate tuple id.
    intermediate_id: i32,
    /// Intermediate slot ids.
    intermediate_slots: Vec<i32>,
    /// What the intermediate tuple covers.
    kind: IntermediateKind,
    /// Doris left child's width and layout.
    left: (usize, Vec<i32>, SlotOverrides),
    /// Doris right child's width and layout.
    right: (usize, Vec<i32>, SlotOverrides),
    /// Whether Doris' right child is the Substrait left input.
    swap: bool,
}

impl JoinRow {
    fn new(
        doris_left: &TranslatedRel,
        doris_right: &TranslatedRel,
        intermediate: &IntermediateTuple,
        swap: bool,
        node: &TPlanNode,
    ) -> Result<Self> {
        let left_width = doris_left.output_width;
        let right_width = doris_right.output_width;
        let kind = if intermediate.slot_ids.len() == left_width + right_width {
            IntermediateKind::Both
        } else if intermediate.slot_ids.len() == right_width
            && matches!(
                node.hash_join_node.as_ref().map(|join| join.join_op),
                Some(TJoinOp::RIGHT_SEMI_JOIN | TJoinOp::RIGHT_ANTI_JOIN)
            )
        {
            IntermediateKind::One(Side::Right)
        } else if intermediate.slot_ids.len() == left_width {
            IntermediateKind::One(Side::Left)
        } else {
            return Err(TranslateError::descriptor(format!(
                "join node {} intermediate tuple {} has {} slots for {} left + {} right columns",
                node.node_id,
                intermediate.id,
                intermediate.slot_ids.len(),
                left_width,
                right_width
            )));
        };
        // Substrait join row: left input columns then right input columns.
        let (first, second) = if swap {
            (doris_right, doris_left)
        } else {
            (doris_left, doris_right)
        };
        let mut row_tuples = first.row_tuples.clone();
        row_tuples.extend(&second.row_tuples);
        let mut child_overrides = first.overrides.clone();
        for (key, index) in &second.overrides {
            child_overrides.insert(*key, first.output_width + index);
        }
        let left_offset = if swap { right_width } else { 0 };
        let right_offset = if swap { 0 } else { left_width };
        let mut intermediate_overrides = SlotOverrides::new();
        for (position, slot_id) in intermediate.slot_ids.iter().enumerate() {
            let column = match kind {
                IntermediateKind::Both if position < left_width => left_offset + position,
                IntermediateKind::Both => right_offset + (position - left_width),
                IntermediateKind::One(Side::Left) => left_offset + position,
                IntermediateKind::One(Side::Right) => right_offset + position,
            };
            intermediate_overrides.insert((intermediate.id, *slot_id), column);
        }
        Ok(Self {
            row_tuples,
            child_overrides,
            intermediate_overrides,
            intermediate_id: intermediate.id,
            intermediate_slots: intermediate.slot_ids.clone(),
            kind,
            left: (
                left_width,
                doris_left.row_tuples.clone(),
                doris_left.overrides.clone(),
            ),
            right: (
                right_width,
                doris_right.row_tuples.clone(),
                doris_right.overrides.clone(),
            ),
            swap,
        })
    }

    /// The layout of the join's output once only `output` remains.
    fn output_layout(&self, output: JoinOutput) -> OutputLayout {
        match output {
            JoinOutput::Both => {
                // The intermediate tuple is the row itself (left then right, in Doris
                // order). With swapped inputs the row is right-then-left, which needs the
                // override form.
                if !self.swap && self.kind == IntermediateKind::Both {
                    OutputLayout {
                        row_tuples: vec![self.intermediate_id],
                        overrides: SlotOverrides::new(),
                    }
                } else {
                    OutputLayout {
                        row_tuples: Vec::new(),
                        overrides: self.intermediate_overrides.clone(),
                    }
                }
            }
            JoinOutput::One(side) | JoinOutput::AntiOuter(side) => {
                // The kept side is the Substrait left input: its columns start at 0. Describe
                // it as the child's own layout plus the intermediate slots that map onto it.
                let (width, row_tuples, child_overrides) = match side {
                    Side::Left => &self.left,
                    Side::Right => &self.right,
                };
                let mut overrides = child_overrides.clone();
                let side_offset = match (self.kind, side) {
                    (IntermediateKind::Both, Side::Right) => self.left.0,
                    _ => 0,
                };
                for (position, slot_id) in self.intermediate_slots.iter().enumerate() {
                    if (self.kind == IntermediateKind::Both
                        || self.kind == IntermediateKind::One(side))
                        && position >= side_offset
                        && position - side_offset < *width
                    {
                        overrides.insert((self.intermediate_id, *slot_id), position - side_offset);
                    }
                }
                OutputLayout {
                    row_tuples: row_tuples.clone(),
                    overrides,
                }
            }
            JoinOutput::NullAwareAnti => self.output_layout(JoinOutput::One(Side::Left)),
        }
    }
}

/// A row layout to attach to a relation.
struct OutputLayout {
    row_tuples: Vec<i32>,
    overrides: SlotOverrides,
}

impl OutputLayout {
    fn attach(self, rel: Rel, output_width: usize) -> TranslatedRel {
        TranslatedRel {
            rel,
            row_tuples: self.row_tuples,
            output_width,
            overrides: self.overrides,
        }
    }
}

/// Projects a fragment's `output_exprs` (the result sink's columns) over the plan's output,
/// returning the relation and the column names (`TExprNode.label`, the slot name, or
/// `expr_<i>`).
pub fn project_output_exprs(
    input: TranslatedRel,
    exprs: &[TExpr],
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<(TranslatedRel, Vec<String>)> {
    let scan_ranges = ScanRanges::default();
    let mut ctx = PlanContext::new(desc, &scan_ranges, registry);
    let mut expressions = Vec::with_capacity(exprs.len());
    let mut names = Vec::with_capacity(exprs.len());
    for (index, expr) in exprs.iter().enumerate() {
        let root = expr
            .nodes
            .first()
            .ok_or_else(|| TranslateError::malformed("output expression is empty"))?;
        let translated = ctx.expr(expr, &input)?;
        let is_slot_ref = root.node_type == TExprNodeType::SLOT_REF;
        expressions.push(if is_slot_ref {
            translated
        } else {
            let declared =
                crate::type_mapper::map_type_desc(&root.type_, root.is_nullable.unwrap_or(true))?;
            expr_translator::cast(translated, declared)
        });
        let slot_name = root.slot_ref.as_ref().and_then(|slot_ref| {
            desc.slot(slot_ref.tuple_id, slot_ref.slot_id)
                .ok()
                .map(SlotInfo::output_name)
        });
        names.push(
            root.label
                .clone()
                .filter(|label| !label.is_empty())
                .or(slot_name)
                .unwrap_or_else(|| format!("expr_{index}")),
        );
    }
    let width = expressions.len();
    let projected = project_rel(input, expressions, Vec::new());
    Ok((
        TranslatedRel {
            output_width: width,
            ..projected
        },
        names,
    ))
}

/// Column names of a relation's layout.
pub fn output_names(translated: &TranslatedRel, desc: &DescriptorTable) -> Result<Vec<String>> {
    if translated.row_tuples.is_empty() {
        return Ok((0..translated.output_width)
            .map(|index| format!("col_{index}"))
            .collect());
    }
    desc.output_names_for_tuples(&translated.row_tuples)
}

/// Re-labels a relation's columns as `row_tuples` without changing them.
fn relabel(
    input: TranslatedRel,
    row_tuples: &[i32],
    ctx: &mut PlanContext<'_>,
    node: &TPlanNode,
) -> Result<TranslatedRel> {
    let width = ctx.desc.row_width(row_tuples)?;
    if width != input.output_width {
        return Err(TranslateError::descriptor(format!(
            "node {} {:?} declares row layout {:?} ({} columns) over a {}-column input",
            node.node_id, node.node_type, row_tuples, width, input.output_width
        )));
    }
    Ok(TranslatedRel::new(input.rel, row_tuples.to_vec(), width))
}

/// Applies the node's `conjuncts` as a filter over the relation's layout.
fn apply_conjuncts(
    input: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let conjuncts = node.conjuncts.as_deref().unwrap_or_default();
    let conditions = conjuncts
        .iter()
        .map(|expr| ctx.expr(expr, &input))
        .collect::<Result<Vec<_>>>()?;
    Ok(match and_conditions(conditions, ctx) {
        Some(condition) => filter_translated(input, condition),
        None => input,
    })
}

/// Applies the node's projection chain: every intermediate stage, then the final projection
/// into `output_tuple_id`.
fn apply_projections(
    mut input: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let intermediate_exprs = node
        .intermediate_projections_list
        .as_deref()
        .unwrap_or_default();
    let intermediate_tuples = node
        .intermediate_output_tuple_id_list
        .as_deref()
        .unwrap_or_default();
    if intermediate_exprs.len() != intermediate_tuples.len() {
        return Err(TranslateError::malformed(format!(
            "node {} has {} intermediate projection lists for {} intermediate tuples",
            node.node_id,
            intermediate_exprs.len(),
            intermediate_tuples.len()
        )));
    }
    for (exprs, tuple_id) in intermediate_exprs.iter().zip(intermediate_tuples) {
        input = project_stage(input, exprs, *tuple_id, node, ctx)?;
    }
    match (node.projections.as_deref(), node.output_tuple_id) {
        (Some(exprs), Some(tuple_id)) => project_stage(input, exprs, tuple_id, node, ctx),
        (Some(exprs), None) if !exprs.is_empty() => Err(TranslateError::MissingField {
            context: "TPlanNode.projections",
            field: "output_tuple_id",
        }),
        _ => Ok(input),
    }
}

/// One projection stage: `exprs` over the input, landing in `tuple_id` with its slot types.
fn project_stage(
    input: TranslatedRel,
    exprs: &[TExpr],
    tuple_id: i32,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let slots = ctx.desc.tuple_slots(tuple_id)?;
    if slots.len() != exprs.len() {
        return Err(TranslateError::descriptor(format!(
            "node {} projects {} expressions into tuple {} with {} slots",
            node.node_id,
            exprs.len(),
            tuple_id,
            slots.len()
        )));
    }
    let slots: Vec<SlotInfo> = slots.into_iter().cloned().collect();
    let expressions = exprs
        .iter()
        .zip(&slots)
        .map(|(expr, slot)| {
            let translated = ctx.expr(expr, &input)?;
            Ok(cast_unless_slot_ref(expr, translated, slot))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(project_rel(input, expressions, vec![tuple_id]))
}

/// Projects `expressions` into the given tuples (a materialized sort tuple).
fn project_into_tuples(
    input: TranslatedRel,
    expressions: Vec<Expression>,
    row_tuples: Vec<i32>,
    ctx: &mut PlanContext<'_>,
    node: &TPlanNode,
) -> Result<TranslatedRel> {
    let width = ctx.desc.row_width(&row_tuples)?;
    if width != expressions.len() {
        return Err(TranslateError::descriptor(format!(
            "node {} materializes {} expressions for layout {:?} with {} slots",
            node.node_id,
            expressions.len(),
            row_tuples,
            width
        )));
    }
    Ok(project_rel(input, expressions, row_tuples))
}

/// Casts a projected expression to its destination slot's type unless it is a plain slot
/// reference (whose type is the slot's by construction).
fn cast_unless_slot_ref(expr: &TExpr, translated: Expression, slot: &SlotInfo) -> Expression {
    let is_slot_ref = expr
        .nodes
        .first()
        .is_some_and(|root| root.node_type == TExprNodeType::SLOT_REF);
    if is_slot_ref {
        translated
    } else {
        expr_translator::cast(translated, slot.substrait_type.clone())
    }
}

/// Wraps a relation in a fetch when the node carries a limit or offset.
// The deprecated plain offset/count oneof variants share wire tags with their expression
// counterparts and are the fields DuckDB's consumer reads.
#[allow(deprecated)]
fn apply_limit(input: TranslatedRel, limit: i64, offset: i64) -> TranslatedRel {
    if limit < 0 && offset == 0 {
        return input;
    }
    let TranslatedRel {
        rel,
        row_tuples,
        output_width,
        overrides,
    } = input;
    // An offset-only fetch needs an explicit unlimited count: the consumer reads the plain
    // count field, and an unset one would decode as `LIMIT 0`.
    let count = if limit >= 0 { limit } else { -1 };
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Fetch(Box::new(FetchRel {
                input: Some(Box::new(rel)),
                offset_mode: (offset != 0).then_some(fetch_rel::OffsetMode::Offset(offset)),
                count_mode: Some(fetch_rel::CountMode::Count(count)),
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        overrides,
    }
}

/// Builds Substrait sort fields from a Doris sort info over `input`'s layout.
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
            "sort info direction lists do not match its ordering expressions",
        ));
    }
    ordering
        .iter()
        .zip(sort_info.is_asc_order.iter().zip(&sort_info.nulls_first))
        .map(|(expr, (asc, nulls_first))| {
            let expr = ctx.expr(expr, input)?;
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

/// Wraps a relation in a sort without changing its layout.
fn sort_rel(input: TranslatedRel, sorts: Vec<SortField>) -> TranslatedRel {
    let TranslatedRel {
        rel,
        row_tuples,
        output_width,
        overrides,
    } = input;
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Sort(Box::new(SortRel {
                input: Some(Box::new(rel)),
                sorts,
                ..Default::default()
            }))),
        },
        row_tuples,
        output_width,
        overrides,
    }
}

/// Combines conditions with `and`; `None` for no conditions.
fn and_conditions(
    mut conditions: Vec<Expression>,
    ctx: &mut PlanContext<'_>,
) -> Option<Expression> {
    match conditions.len() {
        0 => None,
        1 => conditions.pop(),
        _ => Some(ctx.function(URN_BOOLEAN, "and", conditions)),
    }
}

/// Wraps a relation in a filter without changing its layout.
fn filter_translated(input: TranslatedRel, condition: Expression) -> TranslatedRel {
    let TranslatedRel {
        rel,
        row_tuples,
        output_width,
        overrides,
    } = input;
    TranslatedRel {
        rel: filter_rel(rel, condition),
        row_tuples,
        output_width,
        overrides,
    }
}

/// Wraps a bare relation in a filter.
fn filter_rel(input: Rel, condition: Expression) -> Rel {
    Rel {
        rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
            input: Some(Box::new(input)),
            condition: Some(Box::new(condition)),
            ..Default::default()
        }))),
    }
}

/// Builds a project that emits exactly `expressions`, laid out as `row_tuples`.
///
/// The emit mapping selects the projected expressions, which sit after the input columns, so
/// the base offset is the input's width.
fn project_rel(
    input: TranslatedRel,
    expressions: Vec<Expression>,
    row_tuples: Vec<i32>,
) -> TranslatedRel {
    let base = input.output_width as i32;
    let output_mapping = (base..base + expressions.len() as i32).collect();
    let output_width = expressions.len();
    TranslatedRel::new(
        Rel {
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
    )
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
        overrides: input.overrides,
    }
}

/// Emits selected input columns without evaluating new expressions.
fn emit_columns(input: Rel, output_mapping: Vec<i32>) -> Rel {
    Rel {
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
    }
}

/// Refuses an anti join whose null-tested key is not a column reference (casts allowed): the
/// outer-join + `is_null` lowering identifies unmatched rows by the NULL padding, which only
/// holds when the key expression propagates NULL.
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

/// An i32 literal (synthetic cross-product key).
fn i32_literal(value: i32) -> Expression {
    expr_translator::literal(LiteralType::I32(value))
}

/// The nullable boolean type of a predicate.
fn bool_type() -> Type {
    Type {
        kind: Some(r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: r#type::Nullability::Nullable as i32,
        })),
    }
}

/// Whether a node carries filter conjuncts.
fn has_conjuncts(node: &TPlanNode) -> bool {
    node.conjuncts
        .as_ref()
        .is_some_and(|conjuncts| !conjuncts.is_empty())
}

/// Validates the reconstructed child count for a node.
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

#[cfg(test)]
mod tests {
    use doris_thrift::descriptors::{TDescriptorTable, TSlotDescriptor, TTupleDescriptor};
    use doris_thrift::exprs::{TAggregateExpr, TExprNode, TIntLiteral, TSlotRef};
    use doris_thrift::plan_nodes::{
        TAggregationNode, TEqJoinCondition, TExchangeNode, TFileScanNode, THashJoinNode,
        TNestedLoopJoinNode, TSortNode,
    };
    use doris_thrift::types::{
        TFunction, TFunctionName, TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType,
    };

    use super::*;

    fn scalar_desc(type_: TPrimitiveType) -> TTypeDesc {
        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(TScalarType {
                    type_,
                    ..Default::default()
                }),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn slot(id: i32, parent: i32, name: &str, type_: TPrimitiveType) -> TSlotDescriptor {
        TSlotDescriptor {
            id,
            parent,
            slot_type: scalar_desc(type_),
            column_pos: -1,
            null_indicator_bit: 0,
            col_name: name.to_string(),
            slot_idx: -1,
            is_materialized: true,
            ..Default::default()
        }
    }

    /// Tuple 0 = scan {1: a INT, 2: b STRING}; tuple 1 = {3: a INT} (projection / stream);
    /// tuple 2 = {4: x INT, 5: y INT}; tuple 3 = join intermediate {6: a, 7: x, 8: y};
    /// tuple 4 = agg output {9: a INT, 10: cnt BIGINT}; tuple 5 = {11: only INT}.
    fn desc() -> DescriptorTable {
        let desc_tbl = TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(1, 0, "a", TPrimitiveType::INT),
                slot(2, 0, "b", TPrimitiveType::STRING),
                slot(3, 1, "a", TPrimitiveType::INT),
                slot(4, 2, "x", TPrimitiveType::INT),
                slot(5, 2, "y", TPrimitiveType::INT),
                slot(6, 3, "a", TPrimitiveType::INT),
                slot(7, 3, "x", TPrimitiveType::INT),
                slot(8, 3, "y", TPrimitiveType::INT),
                slot(9, 4, "a", TPrimitiveType::INT),
                slot(10, 4, "", TPrimitiveType::BIGINT),
                slot(11, 5, "only", TPrimitiveType::INT),
            ]),
            tuple_descriptors: (0..6)
                .map(|id| TTupleDescriptor {
                    id,
                    ..Default::default()
                })
                .collect(),
            table_descriptors: None,
        };
        DescriptorTable::try_from(&desc_tbl).unwrap()
    }

    fn node(
        node_id: i32,
        node_type: TPlanNodeType,
        num_children: i32,
        row_tuples: Vec<i32>,
    ) -> TPlanNode {
        TPlanNode {
            node_id,
            node_type,
            num_children,
            limit: -1,
            nullable_tuples: vec![false; row_tuples.len()],
            row_tuples,
            ..Default::default()
        }
    }

    fn slot_ref(tuple_id: i32, slot_id: i32, type_: TPrimitiveType) -> TExpr {
        TExpr {
            nodes: vec![TExprNode {
                node_type: TExprNodeType::SLOT_REF,
                type_: scalar_desc(type_),
                num_children: 0,
                output_scale: -1,
                is_nullable: Some(true),
                slot_ref: Some(TSlotRef {
                    slot_id,
                    tuple_id,
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }
    }

    fn scan(node_id: i32) -> TPlanNode {
        TPlanNode {
            file_scan_node: Some(TFileScanNode {
                tuple_id: Some(0),
                table_name: None,
            }),
            ..node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![0])
        }
    }

    fn exchange(node_id: i32, row_tuples: Vec<i32>) -> TPlanNode {
        TPlanNode {
            exchange_node: Some(TExchangeNode {
                input_row_tuples: row_tuples.clone(),
                ..Default::default()
            }),
            ..node(node_id, TPlanNodeType::EXCHANGE_NODE, 0, row_tuples)
        }
    }

    fn scan_ranges(node_id: i32) -> ScanRanges {
        ScanRanges::from_paths([(node_id, vec!["/data/t.parquet".to_string()])])
    }

    /// Translates a plan and renders it; the registry's extensions are attached so function
    /// names show up in the text.
    fn translate(nodes: Vec<TPlanNode>, ranges: &ScanRanges) -> Result<(TranslatedRel, String)> {
        let desc = desc();
        let mut registry = ExtensionRegistry::new();
        let translated = translate_plan(&TPlan { nodes }, &desc, ranges, &mut registry)?;
        let (extension_urns, extensions) = registry.into_extensions();
        let plan = Plan {
            extension_urns,
            extensions,
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(translated.rel.clone()),
                    names: (0..translated.output_width)
                        .map(|i| format!("c{i}"))
                        .collect(),
                })),
            }],
            ..Default::default()
        };
        let text = substrait_explain::format(&plan).0;
        Ok((translated, text))
    }

    use substrait::proto::{Plan, PlanRel, RelRoot, plan_rel};

    #[test]
    fn scan_with_conjunct_projection_and_limit() {
        let mut scan = scan(0);
        scan.conjuncts = Some(vec![slot_ref(0, 1, TPrimitiveType::INT)]);
        scan.projections = Some(vec![slot_ref(0, 1, TPrimitiveType::INT)]);
        scan.output_tuple_id = Some(1);
        scan.limit = 5;
        let (translated, text) = translate(vec![scan], &scan_ranges(0)).unwrap();
        assert_eq!(translated.row_tuples, vec![1]);
        assert_eq!(translated.output_width, 1);
        // Filter, then limit, then the projection into tuple 1.
        let filter = text.find("Filter").unwrap();
        let fetch = text.find("Fetch").unwrap();
        let project = text.find("Project").unwrap();
        assert!(project < fetch && fetch < filter, "{text}");
    }

    #[test]
    fn scan_without_ranges_or_names_is_rejected() {
        let err = translate(vec![scan(0)], &ScanRanges::default()).unwrap_err();
        assert!(
            matches!(err, TranslateError::UnsupportedScanRange { node_id: 0, .. }),
            "{err}"
        );
        // Tuple 4 has a nameless slot.
        let mut unnamed = scan(0);
        unnamed.file_scan_node.as_mut().unwrap().tuple_id = Some(4);
        unnamed.row_tuples = vec![4];
        let err = translate(vec![unnamed], &scan_ranges(0)).unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)), "{err}");
    }

    #[test]
    fn exchange_reads_a_named_stream_and_sorts_a_merging_exchange() {
        let mut merging = exchange(7, vec![2]);
        merging.exchange_node.as_mut().unwrap().sort_info = Some(TSortInfo {
            ordering_exprs: vec![slot_ref(2, 5, TPrimitiveType::INT)],
            is_asc_order: vec![false],
            nulls_first: vec![false],
            ..Default::default()
        });
        merging.exchange_node.as_mut().unwrap().offset = Some(2);
        merging.limit = 3;
        let (_, text) = translate(vec![merging], &ScanRanges::default()).unwrap();
        assert!(text.contains("sirius_stream_7"), "{text}");
        assert!(text.contains("DescNullsLast"), "{text}");
        assert!(
            text.contains("offset=2") && text.contains("limit=3"),
            "{text}"
        );
        // Layout mismatch between input and output tuples is refused.
        let mut skewed = exchange(8, vec![2]);
        skewed.exchange_node.as_mut().unwrap().input_row_tuples = vec![1];
        assert!(matches!(
            translate(vec![skewed], &ScanRanges::default()).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 8, .. }
        ));
    }

    #[test]
    fn sort_relabels_the_child_row_and_rejects_conjuncts_and_width_mismatch() {
        let sort_node = |row_tuples: Vec<i32>, ordering: TExpr| TPlanNode {
            sort_node: Some(TSortNode {
                sort_info: TSortInfo {
                    ordering_exprs: vec![ordering],
                    is_asc_order: vec![true],
                    nulls_first: vec![true],
                    ..Default::default()
                },
                use_top_n: true,
                offset: Some(0),
                ..Default::default()
            }),
            limit: 10,
            ..node(1, TPlanNodeType::SORT_NODE, 1, row_tuples)
        };
        // The child (stream over tuple 0, two columns) is relabelled as tuple 2.
        let (translated, text) = translate(
            vec![
                sort_node(vec![2], slot_ref(2, 5, TPrimitiveType::INT)),
                exchange(0, vec![0]),
            ],
            &ScanRanges::default(),
        )
        .unwrap();
        assert_eq!(translated.row_tuples, vec![2]);
        assert!(text.contains("Sort[($1, &AscNullsFirst)"), "{text}");
        assert!(text.contains("limit=10"), "{text}");
        // Tuple 5 is one column wide; the child has two.
        let err = translate(
            vec![
                sort_node(vec![5], slot_ref(5, 11, TPrimitiveType::INT)),
                exchange(0, vec![0]),
            ],
            &ScanRanges::default(),
        )
        .unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)), "{err}");
        let mut with_conjuncts = sort_node(vec![2], slot_ref(2, 5, TPrimitiveType::INT));
        with_conjuncts.conjuncts = Some(vec![slot_ref(2, 5, TPrimitiveType::INT)]);
        assert!(matches!(
            translate(
                vec![with_conjuncts, exchange(0, vec![0])],
                &ScanRanges::default()
            )
            .unwrap_err(),
            TranslateError::UnsupportedPlanNode { .. }
        ));
    }

    fn agg_expr(name: &str, is_merge: bool, arg: Option<TExpr>) -> TExpr {
        let mut nodes = vec![TExprNode {
            node_type: TExprNodeType::AGG_EXPR,
            type_: scalar_desc(TPrimitiveType::BIGINT),
            num_children: arg.is_some() as i32,
            output_scale: -1,
            is_nullable: Some(false),
            agg_expr: Some(TAggregateExpr {
                is_merge_agg: is_merge,
                param_types: None,
            }),
            fn_: Some(TFunction {
                name: TFunctionName {
                    db_name: None,
                    function_name: name.to_string(),
                },
                ret_type: scalar_desc(TPrimitiveType::BIGINT),
                ..Default::default()
            }),
            ..Default::default()
        }];
        nodes.extend(arg.into_iter().flat_map(|expr| expr.nodes));
        TExpr { nodes }
    }

    fn aggregation(need_finalize: bool, functions: Vec<TExpr>) -> TPlanNode {
        TPlanNode {
            agg_node: Some(TAggregationNode {
                grouping_exprs: Some(vec![slot_ref(0, 1, TPrimitiveType::INT)]),
                aggregate_functions: functions,
                intermediate_tuple_id: 4,
                output_tuple_id: 4,
                need_finalize,
                ..Default::default()
            }),
            ..node(1, TPlanNodeType::AGGREGATION_NODE, 1, vec![4])
        }
    }

    #[test]
    fn finalized_aggregate_casts_measures_and_orders_by_group_key_before_its_limit() {
        let mut agg = aggregation(true, vec![agg_expr("count", false, None)]);
        agg.conjuncts = Some(vec![slot_ref(4, 9, TPrimitiveType::INT)]);
        agg.agg_node.as_mut().unwrap().agg_sort_info_by_group_key = Some(TSortInfo {
            ordering_exprs: vec![slot_ref(4, 9, TPrimitiveType::INT)],
            is_asc_order: vec![true],
            nulls_first: vec![false],
            ..Default::default()
        });
        agg.limit = 100;
        let (translated, text) =
            translate(vec![agg, exchange(0, vec![0])], &ScanRanges::default()).unwrap();
        assert_eq!(translated.row_tuples, vec![4]);
        // Aggregate → cast project → HAVING filter → sort by key → limit.
        let aggregate = text.find("Aggregate").unwrap();
        let cast = text.find("::!i64").unwrap();
        let filter = text.find("Filter").unwrap();
        let sort = text.find("Sort").unwrap();
        let fetch = text.find("Fetch").unwrap();
        assert!(
            fetch < sort && sort < filter && filter < cast && cast < aggregate,
            "{text}"
        );
        assert!(text.contains("count()"), "{text}");
    }

    #[test]
    fn two_phase_aggregates_are_rejected_by_phase() {
        let update = aggregation(false, vec![agg_expr("count", false, None)]);
        match translate(vec![update, exchange(0, vec![0])], &ScanRanges::default()).unwrap_err() {
            TranslateError::UnsupportedPlanNode { reason, .. } => {
                // G-14: a lone phase of a multi-phase aggregate is refused; the stitcher folds
                // both phases before translation (MVP-A0).
                assert!(reason.contains("update-phase aggregate"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
        let merge = aggregation(
            true,
            vec![agg_expr(
                "count",
                true,
                Some(slot_ref(0, 2, TPrimitiveType::STRING)),
            )],
        );
        match translate(vec![merge, exchange(0, vec![0])], &ScanRanges::default()).unwrap_err() {
            TranslateError::UnsupportedPlanNode { reason, .. } => {
                assert!(reason.contains("merge-phase aggregate"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn aggregate_output_tuple_must_match_keys_plus_measures() {
        // Two measures for a tuple with one key slot and one measure slot.
        let agg = aggregation(
            true,
            vec![
                agg_expr("count", false, None),
                agg_expr("count", false, None),
            ],
        );
        let err = translate(vec![agg, exchange(0, vec![0])], &ScanRanges::default()).unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)), "{err}");
        let mut ordered = aggregation(true, vec![agg_expr("count", false, None)]);
        ordered.agg_node.as_mut().unwrap().agg_sort_infos = Some(vec![TSortInfo {
            ordering_exprs: vec![slot_ref(0, 1, TPrimitiveType::INT)],
            is_asc_order: vec![true],
            nulls_first: vec![true],
            ..Default::default()
        }]);
        assert!(matches!(
            translate(vec![ordered, exchange(0, vec![0])], &ScanRanges::default()).unwrap_err(),
            TranslateError::UnsupportedPlanNode { .. }
        ));
    }

    fn hash_join(op: TJoinOp) -> TPlanNode {
        TPlanNode {
            hash_join_node: Some(THashJoinNode {
                join_op: op,
                eq_join_conjuncts: vec![TEqJoinCondition {
                    left: slot_ref(0, 1, TPrimitiveType::INT),
                    right: slot_ref(2, 4, TPrimitiveType::INT),
                    opcode: Some(TExprOpcode::EQ),
                }],
                vintermediate_tuple_id_list: Some(vec![3]),
                ..Default::default()
            }),
            ..node(2, TPlanNodeType::HASH_JOIN_NODE, 2, vec![0, 2])
        }
    }

    /// Left child = stream over tuple 0 (a, b); right child = stream over tuple 2 (x, y).
    /// Intermediate tuple 3 = (a, x, y) is one column narrower than left + right, so the
    /// join's own width check has to catch it.
    #[test]
    fn join_intermediate_tuple_width_is_checked() {
        let err = translate(
            vec![
                hash_join(TJoinOp::INNER_JOIN),
                exchange(0, vec![0]),
                exchange(1, vec![2]),
            ],
            &ScanRanges::default(),
        )
        .unwrap_err();
        assert!(
            matches!(&err, TranslateError::Descriptor(msg) if msg.contains("intermediate tuple")),
            "{err}"
        );
    }

    /// Left child = tuple 5 (one column), right child = tuple 2 (x, y): intermediate tuple 3
    /// (a, x, y) fits left ⊕ right.
    fn join_plan(mut join: TPlanNode) -> Vec<TPlanNode> {
        join.hash_join_node.as_mut().unwrap().eq_join_conjuncts[0].left =
            slot_ref(5, 11, TPrimitiveType::INT);
        join.row_tuples = vec![5, 2];
        join.nullable_tuples = vec![false, false];
        vec![join, exchange(0, vec![5]), exchange(1, vec![2])]
    }

    #[test]
    fn inner_join_output_is_the_intermediate_tuple_with_projections_over_it() {
        let mut join = hash_join(TJoinOp::INNER_JOIN);
        join.projections = Some(vec![slot_ref(3, 8, TPrimitiveType::INT)]);
        join.output_tuple_id = Some(1);
        join.hash_join_node.as_mut().unwrap().other_join_conjuncts =
            Some(vec![slot_ref(3, 7, TPrimitiveType::INT)]);
        let (translated, text) = translate(join_plan(join), &ScanRanges::default()).unwrap();
        assert_eq!(translated.row_tuples, vec![1]);
        // `y` is column 2 of the intermediate row; the other conjunct's `x` is column 1.
        assert!(text.contains("Project[$2]"), "{text}");
        assert!(text.contains("and(equal($0, $1):boolean?, $1)"), "{text}");
    }

    #[test]
    fn semi_and_anti_joins_keep_one_side() {
        for (op, expected) in [
            (TJoinOp::LEFT_SEMI_JOIN, "Join[&LeftSemi, equal($0, $1)"),
            (TJoinOp::RIGHT_SEMI_JOIN, "Join[&LeftSemi, equal($2, $0)"),
            (TJoinOp::LEFT_ANTI_JOIN, "Filter[is_null($1)"),
            (TJoinOp::RIGHT_ANTI_JOIN, "Filter[is_null($2)"),
            (TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN, "Filter[not($1)"),
        ] {
            let mut join = hash_join(op);
            // Semi/anti joins keep their side's own tuple as the row layout.
            join.hash_join_node
                .as_mut()
                .unwrap()
                .vintermediate_tuple_id_list = Some(vec![match op {
                TJoinOp::RIGHT_SEMI_JOIN | TJoinOp::RIGHT_ANTI_JOIN => 2,
                _ => 5,
            }]);
            let (translated, text) = translate(join_plan(join), &ScanRanges::default()).unwrap();
            assert!(text.contains(expected), "{op:?}: {text}");
            let kept = match op {
                TJoinOp::RIGHT_SEMI_JOIN | TJoinOp::RIGHT_ANTI_JOIN => (vec![2], 2),
                _ => (vec![5], 1),
            };
            assert_eq!(
                (translated.row_tuples, translated.output_width),
                kept,
                "{op:?}"
            );
        }
    }

    #[test]
    fn join_gates() {
        let mut two_keys = hash_join(TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN);
        two_keys
            .hash_join_node
            .as_mut()
            .unwrap()
            .eq_join_conjuncts
            .push(TEqJoinCondition {
                left: slot_ref(5, 11, TPrimitiveType::INT),
                right: slot_ref(2, 5, TPrimitiveType::INT),
                opcode: Some(TExprOpcode::EQ),
            });
        assert!(matches!(
            translate(join_plan(two_keys), &ScanRanges::default()).unwrap_err(),
            TranslateError::UnsupportedPlanNode { reason, .. } if reason.contains("null-aware")
        ));
        let mut mark = hash_join(TJoinOp::INNER_JOIN);
        mark.hash_join_node.as_mut().unwrap().is_mark = Some(true);
        assert!(matches!(
            translate(join_plan(mark), &ScanRanges::default()).unwrap_err(),
            TranslateError::UnsupportedPlanNode { reason, .. } if reason.contains("mark")
        ));
        let mut inequality = hash_join(TJoinOp::INNER_JOIN);
        inequality
            .hash_join_node
            .as_mut()
            .unwrap()
            .eq_join_conjuncts[0]
            .opcode = Some(TExprOpcode::LT);
        assert!(matches!(
            translate(join_plan(inequality), &ScanRanges::default()).unwrap_err(),
            TranslateError::UnsupportedPlanNode { reason, .. } if reason.contains("equality")
        ));
        let mut two_intermediates = hash_join(TJoinOp::INNER_JOIN);
        two_intermediates
            .hash_join_node
            .as_mut()
            .unwrap()
            .vintermediate_tuple_id_list = Some(vec![3, 3]);
        assert!(matches!(
            translate(join_plan(two_intermediates), &ScanRanges::default()).unwrap_err(),
            TranslateError::UnsupportedPlanNode { reason, .. } if reason.contains("intermediate")
        ));
        assert!(matches!(
            translate(
                join_plan(hash_join(TJoinOp::ASOF_LEFT_INNER_JOIN)),
                &ScanRanges::default()
            )
            .unwrap_err(),
            TranslateError::UnsupportedPlanNode { .. }
        ));
    }

    #[test]
    fn nested_loop_join_becomes_a_constant_key_join_plus_filter() {
        let nested = TPlanNode {
            nested_loop_join_node: Some(TNestedLoopJoinNode {
                join_op: TJoinOp::INNER_JOIN,
                vintermediate_tuple_id_list: Some(vec![3]),
                join_conjuncts: Some(vec![slot_ref(3, 7, TPrimitiveType::INT)]),
                ..Default::default()
            }),
            ..node(2, TPlanNodeType::CROSS_JOIN_NODE, 2, vec![5, 2])
        };
        let (translated, text) = translate(
            vec![nested, exchange(0, vec![5]), exchange(1, vec![2])],
            &ScanRanges::default(),
        )
        .unwrap();
        assert_eq!(translated.row_tuples, vec![3]);
        assert!(text.contains("Join[&Inner, equal($1, $4)"), "{text}");
        assert!(text.contains("Project[$0, $2, $3]"), "{text}");
        assert!(text.contains("Filter[$1 =>"), "{text}");
        let mut outer = TPlanNode {
            nested_loop_join_node: Some(TNestedLoopJoinNode {
                join_op: TJoinOp::LEFT_OUTER_JOIN,
                vintermediate_tuple_id_list: Some(vec![3]),
                ..Default::default()
            }),
            ..node(2, TPlanNodeType::CROSS_JOIN_NODE, 2, vec![5, 2])
        };
        outer.nullable_tuples = vec![false, true];
        assert!(matches!(
            translate(
                vec![outer, exchange(0, vec![5]), exchange(1, vec![2])],
                &ScanRanges::default()
            )
            .unwrap_err(),
            TranslateError::UnsupportedPlanNode { .. }
        ));
    }

    #[test]
    fn unsupported_and_malformed_plans_are_named() {
        // G-12 (UNION / INTERSECT / EXCEPT) and G-11 (window functions) are refused by node
        // type; `tests/corpus.rs::gap_corpus_verdicts` checks the same on real FE dispatches.
        for node_type in [
            TPlanNodeType::UNION_NODE,
            TPlanNodeType::INTERSECT_NODE,
            TPlanNodeType::EXCEPT_NODE,
            TPlanNodeType::ANALYTIC_EVAL_NODE,
        ] {
            let unsupported = node(0, node_type, 0, vec![0]);
            let err = translate(vec![unsupported], &ScanRanges::default()).unwrap_err();
            assert!(
                matches!(err, TranslateError::UnsupportedPlanNode { node_type: t, .. } if t == node_type),
                "{node_type:?}: {err}"
            );
            assert!(err.to_string().contains(&format!("{node_type:?}")), "{err}");
        }
        // Trailing node.
        assert!(matches!(
            translate(
                vec![exchange(0, vec![0]), exchange(1, vec![0])],
                &ScanRanges::default()
            )
            .unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
        // Under-run: a sort without its child.
        let sort = TPlanNode {
            sort_node: Some(TSortNode::default()),
            ..node(1, TPlanNodeType::SORT_NODE, 1, vec![0])
        };
        assert!(matches!(
            translate(vec![sort], &ScanRanges::default()).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
        assert!(matches!(
            translate(vec![], &ScanRanges::default()).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn projection_chain_stages_and_width_checks() {
        let mut scan = scan(0);
        scan.intermediate_projections_list = Some(vec![vec![
            slot_ref(0, 2, TPrimitiveType::STRING),
            slot_ref(0, 1, TPrimitiveType::INT),
        ]]);
        scan.intermediate_output_tuple_id_list = Some(vec![2]);
        scan.projections = Some(vec![slot_ref(2, 5, TPrimitiveType::INT)]);
        scan.output_tuple_id = Some(1);
        let (translated, text) = translate(vec![scan.clone()], &scan_ranges(0)).unwrap();
        assert_eq!(translated.row_tuples, vec![1]);
        assert!(text.contains("Project[$1]"), "{text}");
        assert!(text.contains("Project[$1, $0]"), "{text}");
        // A stage whose expression count differs from its tuple width.
        scan.projections = Some(vec![]);
        let err = translate(vec![scan.clone()], &scan_ranges(0)).unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)), "{err}");
        // Lists of different lengths.
        scan.projections = Some(vec![slot_ref(2, 5, TPrimitiveType::INT)]);
        scan.intermediate_output_tuple_id_list = Some(vec![]);
        assert!(matches!(
            translate(vec![scan], &scan_ranges(0)).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn non_slot_projections_are_cast_to_the_slot_type() {
        let literal = TExpr {
            nodes: vec![TExprNode {
                node_type: TExprNodeType::INT_LITERAL,
                type_: scalar_desc(TPrimitiveType::TINYINT),
                num_children: 0,
                output_scale: -1,
                is_nullable: Some(false),
                int_literal: Some(TIntLiteral { value: 1 }),
                ..Default::default()
            }],
        };
        let mut scan = scan(0);
        scan.projections = Some(vec![literal]);
        scan.output_tuple_id = Some(1);
        let (_, text) = translate(vec![scan], &scan_ranges(0)).unwrap();
        assert!(text.contains("(1:i8)::!i32?"), "{text}");
    }
}
