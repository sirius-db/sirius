use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use starrocks_thrift::exprs::{TExpr, TExprNodeType};
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::plan_nodes::{TJoinOp, TPlan, TPlanNode, TPlanNodeType, TSortInfo};
use starrocks_thrift::types::TSlotId;
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{LocalFiles, NamedTable, ReadType};
use substrait::proto::{
    AggregateFunction, AggregateRel, Expression, FetchRel, FilterRel, JoinRel, ProjectRel, ReadRel,
    Rel, RelCommon, SortField, SortRel, aggregate_rel, expression, function_argument, join_rel,
    rel, rel_common, sort_field,
};

use crate::agg_phase::{self, AggPhase};
use crate::descriptor_table::{DescriptorTable, SlotKey};
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, ExprContext, TranslateExpr};
use crate::row_layout::RowLayout;
use crate::runtime_filter::FilterInput;
use crate::scan_paths::ScanFilePaths;
use crate::type_mapper;
use crate::{
    ExchangeInput, ExtensionRegistry, StreamInputColumn, StreamInputSchema, URN_AGGREGATE,
    URN_ARITHMETIC, URN_BOOLEAN, URN_COMPARISON,
};

/// Partially translated relation plus the StarRocks row layout it emits.
pub(crate) struct TranslatedRel {
    rel: Rel,
    layout: RowLayout,
}

impl TranslatedRel {
    pub(crate) fn into_rel(self) -> Rel {
        self.rel
    }

    pub(crate) fn output_names(&self, desc: &DescriptorTable) -> Result<Vec<String>> {
        self.layout.output_names(desc)
    }

    pub(crate) fn resolve(&self, key: SlotKey) -> Result<usize> {
        self.layout.resolve(key)
    }
}

/// One translated fragment: its relation tree plus the stream schemas the caller must declare.
pub(crate) struct TranslatedFragment {
    /// Root relation and its row layout.
    pub root: TranslatedRel,
    /// Schema of every exchange lowered to a stream read, in translation order.
    pub stream_inputs: Vec<StreamInputSchema>,
}

/// Mutable state shared by plan-node translators.
struct PlanContext<'a> {
    /// Descriptor lookups for row layouts, tables, and scan schemas.
    desc: &'a DescriptorTable,
    /// Parquet file paths for each scan node, collected from the fragment's broker
    /// scan ranges. Scans with paths emit a `local_files` read; path-less scans
    /// fall back to a named-table read.
    scan_paths: &'a ScanFilePaths,
    /// Input streams bound to exchange nodes, keyed by receiver node id.
    exchange_inputs: &'a HashMap<i32, &'a ExchangeInput>,
    /// Key streams bound to runtime filters, keyed by filter id.
    filter_inputs: &'a HashMap<i32, &'a FilterInput>,
    /// Substrait extension registry shared across the whole plan.
    registry: &'a mut ExtensionRegistry,
    /// Stream schemas recorded as exchanges are lowered, in translation order.
    stream_inputs: Vec<StreamInputSchema>,
    /// Common-expr slots each project must emit because an ancestor consumes them.
    consumed_above: HashMap<i32, Vec<i32>>,
    /// Common-expr columns projects emitted past their descriptor row.
    carried: HashSet<SlotKey>,
    /// Partial-state column types of the exchanges that feed merge aggregations, keyed by
    /// exchange node id. Computed by [`merge_state_columns`] before translation starts.
    state_columns: HashMap<i32, Vec<(SlotKey, substrait::proto::Type)>>,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    fn new(
        desc: &'a DescriptorTable,
        scan_paths: &'a ScanFilePaths,
        exchange_inputs: &'a HashMap<i32, &'a ExchangeInput>,
        filter_inputs: &'a HashMap<i32, &'a FilterInput>,
        registry: &'a mut ExtensionRegistry,
    ) -> Self {
        Self {
            desc,
            scan_paths,
            exchange_inputs,
            filter_inputs,
            registry,
            stream_inputs: Vec::new(),
            consumed_above: HashMap::new(),
            carried: HashSet::new(),
            state_columns: HashMap::new(),
        }
    }

    fn expr_context<'b>(&'b mut self, layout: &'b RowLayout) -> ExprContext<'b> {
        ExprContext::new(self.registry, layout)
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
        ctx.consumed_above = common_slots_consumed_above(self, ctx.desc)?;
        ctx.state_columns = merge_state_columns(self)?;
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
    // Refuse carried common-expr columns into a join until that path is tested with them.
    if matches!(
        node.node_type,
        TPlanNodeType::HASH_JOIN_NODE | TPlanNodeType::NESTLOOP_JOIN_NODE
    ) && children.iter().any(|child| {
        child
            .layout
            .columns()
            .flatten()
            .any(|key| ctx.carried.contains(&key))
    }) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "common-expr columns carried into a join are not supported",
        });
    }
    let translated = match node.node_type {
        TPlanNodeType::FILE_SCAN_NODE => translate_file_scan(node, children, ctx),
        TPlanNodeType::HDFS_SCAN_NODE => translate_hdfs_scan(node, children, ctx),
        TPlanNodeType::SELECT_NODE => translate_select(node, children, ctx),
        TPlanNodeType::PROJECT_NODE => translate_project(node, children, ctx),
        TPlanNodeType::AGGREGATION_NODE => translate_aggregation(node, children, ctx),
        TPlanNodeType::EXCHANGE_NODE => translate_exchange(node, children, ctx),
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
    let TranslatedRel { rel, layout } = input;
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Fetch(Box::new(FetchRel {
                input: Some(Box::new(rel)),
                offset_expr: (offset != 0).then(|| Box::new(i64_literal(offset))),
                count_expr: (node.limit >= 0).then(|| Box::new(i64_literal(node.limit))),
                ..Default::default()
            }))),
        },
        layout,
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
        layout: RowLayout::from_tuples(ctx.desc, &[tuple_id])?,
    };
    let filtered = apply_conjuncts(input, node, ctx)?;
    apply_runtime_filters(filtered, node, ctx)
}

/// Keeps only the scan rows whose probe key appears in a bound runtime filter's key stream: a
/// left semi join per bound filter the scan probes. The join that built the filter is exact
/// anyway, so dropping rows it would reject changes nothing but the work.
fn apply_runtime_filters(
    mut input: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    for filter in crate::runtime_filter::probe_filters_of(node) {
        let Some(bound) = filter
            .filter_id
            .and_then(|id| ctx.filter_inputs.get(&id).copied())
        else {
            continue;
        };
        let probe = filter
            .plan_node_id_to_target_expr
            .as_ref()
            .and_then(|targets| targets.get(&node.node_id))
            .ok_or_else(|| {
                TranslateError::malformed(format!(
                    "runtime filter {} lists scan {} as a target without its probe expression",
                    bound.filter_id, node.node_id
                ))
            })?;
        let key_type = type_mapper::map_type_desc(
            &probe
                .nodes
                .first()
                .ok_or_else(|| TranslateError::malformed("empty runtime filter probe expression"))?
                .type_,
            true,
        )?;
        let key_type_name = type_mapper::duckdb_type_name(&key_type)?;
        if key_type_name != bound.column.ty {
            return Err(TranslateError::malformed(format!(
                "runtime filter {} probes a {key_type_name} key but its key stream carries {}",
                bound.filter_id, bound.column.ty
            )));
        }
        let mut expr_ctx = ctx.expr_context(&input.layout);
        let probe = probe.translate(&mut expr_ctx)?;
        let anchor = ctx.registry.register_function(URN_COMPARISON, "equal");
        let condition = expr_translator::scalar_function(
            anchor,
            vec![probe, field_selection(input.layout.len() as i32)],
            type_mapper::bool_type(),
        );
        let keys = TranslatedRel {
            rel: stream_read_rel(
                substrait::proto::NamedStruct {
                    names: vec![bound.column.name.clone()],
                    r#struct: Some(substrait::proto::r#type::Struct {
                        types: vec![key_type],
                        nullability: substrait::proto::r#type::Nullability::Required as i32,
                        ..Default::default()
                    }),
                },
                &bound.stream_view,
            ),
            layout: RowLayout::new([None]),
        };
        ctx.stream_inputs.push(StreamInputSchema {
            node_id: bound.node_id,
            stream_view: bound.stream_view.clone(),
            columns: vec![bound.column.clone()],
        });
        input = join_rel(input, keys, condition, join_rel::JoinType::LeftSemi)?;
    }
    Ok(input)
}

/// Refuses a node whose `common_slot_map` this translator does not materialize.
///
/// Only `PROJECT_NODE` appends its common slots. On every other node carrying the field the
/// shared sub-expressions would be read past: a conjunct or key that references one of them then
/// fails later with an opaque descriptor error (`slot N (tuple T) is not part of the row layout`),
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
pub(crate) fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    scan_paths: &ScanFilePaths,
    exchange_inputs: &HashMap<i32, &ExchangeInput>,
    filter_inputs: &HashMap<i32, &FilterInput>,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedFragment> {
    let mut ctx = PlanContext::new(desc, scan_paths, exchange_inputs, filter_inputs, registry);
    let root = plan.translate(&mut ctx)?;
    Ok(TranslatedFragment {
        root,
        stream_inputs: ctx.stream_inputs,
    })
}

/// Partial-state column types, per exchange node, for the exchanges that feed merge
/// aggregations.
///
/// The exchange is translated before the merge aggregation above it, but it has to declare
/// its stream with the types the partial fragment emits ([`agg_phase::partial_state`]), not the
/// descriptor's slot types. In preorder a merge aggregation's only child is the next node; any
/// other child would hand it partial states this rule cannot type, so that plan is refused.
fn merge_state_columns(
    plan: &TPlan,
) -> Result<HashMap<i32, Vec<(SlotKey, substrait::proto::Type)>>> {
    let mut columns = HashMap::new();
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
        let unsupported = |reason| TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason,
        };
        let exchange = plan
            .nodes
            .get(index + 1)
            .filter(|child| child.node_type == TPlanNodeType::EXCHANGE_NODE)
            .ok_or_else(|| {
                unsupported("a merge aggregation must read its partial states from an exchange")
            })?;
        let mut states = Vec::with_capacity(agg.aggregate_functions.len());
        for expr in &agg.aggregate_functions {
            let state = agg_phase::partial_state(
                node.node_id,
                node.node_type,
                agg_phase::measure_function(expr)?,
            )?;
            // Each merge measure reads its own partial-state column, as a bare slot reference.
            let slot = match expr.nodes.as_slice() {
                [_, argument] if argument.node_type == TExprNodeType::SLOT_REF => {
                    argument.slot_ref.as_ref()
                }
                _ => None,
            }
            .ok_or_else(|| {
                unsupported("a merge aggregate must read exactly one partial-state column")
            })?;
            states.push((SlotKey::new(slot.tuple_id, slot.slot_id), state.ty));
        }
        columns.insert(exchange.node_id, states);
    }
    Ok(columns)
}

/// Common-expr slots, per `PROJECT_NODE`, that an ancestor reads even though no output tuple
/// materializes them. Q14's aggregate reads such a slot.
fn common_slots_consumed_above(
    plan: &TPlan,
    desc: &DescriptorTable,
) -> Result<HashMap<i32, Vec<i32>>> {
    let mut consumed = HashMap::new();
    // Open ancestors of the current preorder node: (node index, children not yet visited).
    let mut ancestors: Vec<(usize, i32)> = Vec::new();
    for (index, node) in plan.nodes.iter().enumerate() {
        if let Some(common) = node
            .project_node
            .as_ref()
            .and_then(|project| project.common_slot_map.as_ref())
            .filter(|common| !common.is_empty())
        {
            let mut materialized = BTreeSet::new();
            for &tuple_id in &node.row_tuples {
                materialized.extend(desc.materialized_slot_ids(tuple_id)?);
            }
            let candidates: BTreeSet<i32> = common
                .keys()
                .copied()
                .filter(|slot_id| !materialized.contains(slot_id))
                .collect();
            let mut hits = BTreeSet::new();
            for &(ancestor, _) in &ancestors {
                collect_slot_ref_hits(&plan.nodes[ancestor], &candidates, &mut hits);
            }
            if !hits.is_empty() {
                consumed.insert(node.node_id, hits.into_iter().collect());
            }
        }
        if node.num_children > 0 {
            ancestors.push((index, node.num_children));
        } else {
            while let Some(top) = ancestors.last_mut() {
                top.1 -= 1;
                if top.1 > 0 {
                    break;
                }
                ancestors.pop();
            }
        }
    }
    Ok(consumed)
}

/// Adds the `candidates` that `node`'s conjuncts, aggregate or project expressions reference.
fn collect_slot_ref_hits(node: &TPlanNode, candidates: &BTreeSet<i32>, hits: &mut BTreeSet<i32>) {
    let mut exprs: Vec<&TExpr> = node.conjuncts.iter().flatten().collect();
    if let Some(agg) = &node.agg_node {
        exprs.extend(agg.grouping_exprs.iter().flatten());
        exprs.extend(&agg.aggregate_functions);
    }
    if let Some(project) = &node.project_node {
        exprs.extend(
            project
                .slot_map
                .iter()
                .chain(&project.common_slot_map)
                .flatten()
                .map(|(_, expr)| expr),
        );
    }
    for expr_node in exprs.into_iter().flat_map(|expr| &expr.nodes) {
        if expr_node.node_type == TExprNodeType::SLOT_REF
            && let Some(slot_ref) = &expr_node.slot_ref
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
/// the FE's `input_row_tuples`; its column names are the sender's, bound positionally.
///
/// A merging exchange (`sort_info` present) is a stream read wrapped in a sort.
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
    if input.stream_view.is_empty() {
        return Err(TranslateError::malformed(format!(
            "exchange node {} is bound to an input stream with an empty view name",
            node.node_id
        )));
    }
    let mut schema = ctx
        .desc
        .named_struct_for_tuples(&exchange.input_row_tuples)?;
    let output_width = schema
        .r#struct
        .as_ref()
        .map(|structure| structure.types.len())
        .unwrap_or(0);
    if input.names.len() != output_width {
        return Err(TranslateError::descriptor(format!(
            "row layout {:?} has {} fields but exchange input has {} names",
            exchange.input_row_tuples,
            output_width,
            input.names.len()
        )));
    }
    schema.names = input.names.clone();
    let layout = RowLayout::from_tuples(ctx.desc, &exchange.input_row_tuples)?;
    // Partial states feeding a merge aggregation take their type from the partial-state rule
    // the sender used, not from the descriptor (see `merge_state_columns`).
    if let Some(states) = ctx.state_columns.get(&node.node_id)
        && let Some(structure) = schema.r#struct.as_mut()
    {
        for (key, ty) in states {
            structure.types[layout.resolve(*key)?] = ty.clone();
        }
    }

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
        layout,
    };
    if let Some(sort_info) = &exchange.sort_info {
        let sorts = sort_fields(sort_info, &translated, ctx)?;
        translated = sort_rel(translated, sorts);
    }
    apply_conjuncts(translated, node, ctx)
}

/// Builds a read of an engine stream view with an explicit schema.
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

/// Translates an `AGGREGATION_NODE` into a Substrait aggregate relation.
///
/// The node's phase (one-shot / partial / merge, see [`agg_phase::classify`]) decides what
/// each measure becomes. One-phase keeps the existing aggregate surface. Partial and merge
/// accept SUM, COUNT, MIN and MAX: a partial measure emits its partial state and a merge
/// measure combines partial states, both as [`agg_phase::partial_state`] says. The output row
/// layout is the aggregation output tuple.
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
    if agg.intermediate_tuple_id != agg.output_tuple_id {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "aggregation node has distinct intermediate and output tuples",
        });
    }
    let output_tuple = agg.output_tuple_id;

    let grouping_exprs = agg.grouping_exprs.as_deref().unwrap_or_default();
    let mut grouping_expressions = Vec::with_capacity(grouping_exprs.len());
    for expr in grouping_exprs {
        let mut expr_ctx = ctx.expr_context(&child.layout);
        grouping_expressions.push(expr.translate(&mut expr_ctx)?);
    }

    // Aggregate output types come from the output tuple's slots, which carry the grouping keys
    // first and then one slot per aggregate function.
    let output_slots = ctx.desc.materialized_slot_ids(output_tuple)?;
    if output_slots.len() != grouping_expressions.len() + agg.aggregate_functions.len() {
        return Err(TranslateError::descriptor(format!(
            "AGGREGATION_NODE {} output tuple {} has {} slots for {} keys + {} aggregates",
            node.node_id,
            output_tuple,
            output_slots.len(),
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
    // Integer `sum` measures, by output column, with the type the plan declares for them.
    let mut integer_sums = Vec::new();
    for (expr, slot_id) in agg
        .aggregate_functions
        .iter()
        .zip(&output_slots[grouping_expressions.len()..])
    {
        // A two-phase measure's partial state (and so its merge function) comes from one rule
        // both fragments share; see `agg_phase::partial_state`.
        let state = match phase {
            AggPhase::OneShot => None,
            AggPhase::Partial | AggPhase::Merge => Some(agg_phase::partial_state(
                node.node_id,
                node.node_type,
                agg_phase::measure_function(expr)?,
            )?),
        };
        let mut expr_ctx = ctx.expr_context(&child.layout);
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
        let final_type = || {
            ctx.desc
                .slot(output_tuple, *slot_id)?
                .substrait_type
                .clone()
                .ok_or(TranslateError::MissingField {
                    context: "aggregate output slot",
                    field: "slotType",
                })
        };
        let (function_name, output_type) = match (phase, state) {
            (AggPhase::Partial, Some(state)) => {
                // MIN/MAX emit their argument's type. The merge fragment declares the exchange
                // column with the rule's type, so the two have to agree.
                if matches!(call.name.as_str(), "min" | "max") {
                    let emitted = match expr.nodes.get(1) {
                        Some(argument) => Some(type_mapper::duckdb_type_name(
                            &type_mapper::map_type_desc(&argument.type_, true)?,
                        )?),
                        None => None,
                    };
                    if emitted != Some(type_mapper::duckdb_type_name(&state.ty)?) {
                        return Err(TranslateError::UnsupportedPlanNode {
                            node_id: node.node_id,
                            node_type: node.node_type,
                            reason: "two-phase MIN/MAX whose argument type differs from its \
                                     return type is not supported",
                        });
                    }
                }
                (call.name.clone(), state.ty)
            }
            // The merge step applies the rule's merge function (COUNT merges as SUM) and
            // returns the final value the output slot declares.
            (AggPhase::Merge, Some(state)) => (state.merge_function.to_string(), final_type()?),
            _ => (call.name.clone(), final_type()?),
        };
        // `count` lives in the generic aggregate extension; sum/avg/min/max are declared by
        // the arithmetic extension.
        let urn = if function_name == "count" {
            URN_AGGREGATE
        } else {
            URN_ARITHMETIC
        };
        let anchor = ctx.registry.register_function(urn, &function_name);
        if phase != AggPhase::OneShot && function_name == "sum" && is_integer(&output_type) {
            integer_sums.push((
                grouping_expressions.len() + measures.len(),
                output_type.clone(),
            ));
        }
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
        layout: RowLayout::new(
            output_slots
                .into_iter()
                .map(|slot_id| Some(SlotKey::new(output_tuple, slot_id))),
        ),
    };
    // Node conjuncts evaluate over the aggregation output (HAVING predicates).
    apply_conjuncts(cast_integer_sums(aggregated, &integer_sums), node, ctx)
}

/// DuckDB binds `sum` over an integer column to HUGEINT whatever output type the plan declares,
/// and cuDF has no 128-bit integer: an exchange typed BIGINT then refuses the column, and a
/// HUGEINT literal can't reach the GPU. StarRocks sums integers into BIGINT, so a two-phase
/// plan casts each integer `sum` (including COUNT's merge, a `sum` of counts) back to the type
/// the shared partial-state rule declares.
fn cast_integer_sums(
    aggregated: TranslatedRel,
    integer_sums: &[(usize, substrait::proto::Type)],
) -> TranslatedRel {
    if integer_sums.is_empty() {
        return aggregated;
    }
    let outputs = aggregated
        .layout
        .columns()
        .enumerate()
        .map(|(column, binding)| {
            let field = field_selection(column as i32);
            let expression = match integer_sums.iter().find(|(at, _)| *at == column) {
                Some((_, ty)) => Expression {
                    rex_type: Some(expression::RexType::Cast(Box::new(expression::Cast {
                        r#type: Some(ty.clone()),
                        input: Some(Box::new(field)),
                        failure_behavior: expression::cast::FailureBehavior::ThrowException as i32,
                    }))),
                },
                None => field,
            };
            (expression, binding)
        })
        .collect();
    project_rel(aggregated, outputs)
}

fn is_integer(ty: &substrait::proto::Type) -> bool {
    use substrait::proto::r#type::Kind;
    matches!(
        ty.kind,
        Some(Kind::I8(_) | Kind::I16(_) | Kind::I32(_) | Kind::I64(_))
    )
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
            let mut expr_ctx = ctx.expr_context(&child.layout);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
        project_bound_rel(
            child,
            expressions,
            RowLayout::from_tuples(ctx.desc, &[sort_tuple])?,
        )?
    } else {
        child
    };

    let sorts = sort_fields(&sort.sort_info, &input, ctx)?;
    apply_conjuncts(sort_rel(input, sorts), node, ctx)
}

/// Wraps `input` in a Substrait sort; the row layout is unchanged.
fn sort_rel(input: TranslatedRel, sorts: Vec<SortField>) -> TranslatedRel {
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Sort(Box::new(SortRel {
                input: Some(Box::new(input.rel)),
                sorts,
                ..Default::default()
            }))),
        },
        layout: input.layout,
    }
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
            let mut expr_ctx = ctx.expr_context(&input.layout);
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
/// the combined layout resolves slot references.
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
        // The FE's EXISTS / IN with the subquery on the probe side. DuckDB's Substrait
        // consumer maps `RIGHT_SEMI` to its own RIGHT_SEMI join, which keeps only the build
        // side's columns, and the GPU hash join runs it (with other join conjuncts too).
        TJoinOp::RIGHT_SEMI_JOIN => (join_rel::JoinType::RightSemi, JoinOutput::Right),
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

    let combined_layout = left.layout.concat(&right.layout);
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
        let mut expr_ctx = ctx.expr_context(&combined_layout);
        let left_expr = eq.left.translate(&mut expr_ctx)?;
        let mut expr_ctx = ctx.expr_context(&combined_layout);
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
        let mut expr_ctx = ctx.expr_context(&combined_layout);
        conditions.push(expr.translate(&mut expr_ctx)?);
    }
    let condition = and_conditions(conditions, ctx).ok_or(TranslateError::UnsupportedPlanNode {
        node_id: node.node_id,
        node_type: node.node_type,
        reason: "hash join without join conjuncts",
    })?;

    let left_width = left.layout.len();
    let right_width = right.layout.len();
    let joined = join_rel(left, right, condition, join_type)?;
    let joined = match output {
        JoinOutput::LeftAnti => {
            let (_, right_key) = first_equality
                .ok_or_else(|| TranslateError::malformed("left anti join has no equality key"))?;
            require_column_key(node, &right_key)?;
            let filtered = filter_is_null(joined, right_key, ctx);
            emit_columns(filtered, (0..left_width as i32).collect())?
        }
        JoinOutput::RightAnti => {
            let (left_key, _) = first_equality
                .ok_or_else(|| TranslateError::malformed("right anti join has no equality key"))?;
            require_column_key(node, &left_key)?;
            let filtered = filter_is_null(joined, left_key, ctx);
            let start = left_width as i32;
            let end = start + right_width as i32;
            emit_columns(filtered, (start..end).collect())?
        }
        JoinOutput::NullAwareLeftAnti => {
            let marker = field_selection(left_width as i32);
            let not_anchor = ctx.registry.register_function(URN_BOOLEAN, "not");
            let condition = expr_translator::scalar_function(
                not_anchor,
                vec![marker],
                crate::type_mapper::bool_type(),
            );
            let filtered = filter_rel(joined, condition);
            emit_columns(filtered, (0..left_width as i32).collect())?
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
    Right,
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
    let left_width = left.layout.len();
    let right_width = right.layout.len();
    let left = append_project(left, i32_literal(1), None)?;
    let right = append_project(right, i32_literal(1), None)?;
    let equal_anchor = ctx.registry.register_function(URN_COMPARISON, "equal");
    let condition = expr_translator::scalar_function(
        equal_anchor,
        vec![
            field_selection(left_width as i32),
            field_selection((left.layout.len() + right_width) as i32),
        ],
        crate::type_mapper::bool_type(),
    );
    let right_start = left.layout.len() as i32;
    let joined = join_rel(left, right, condition, join_rel::JoinType::Inner)?;
    let mut mapping = (0..left_width as i32).collect::<Vec<_>>();
    mapping.extend(right_start..right_start + right_width as i32);
    let cross = emit_columns(joined, mapping)?;
    let filtered = if let Some(conjuncts) = join
        .join_conjuncts
        .as_ref()
        .filter(|conjuncts| !conjuncts.is_empty())
    {
        let mut conditions = Vec::with_capacity(conjuncts.len());
        for expr in conjuncts {
            let mut expr_ctx = ctx.expr_context(&cross.layout);
            conditions.push(expr.translate(&mut expr_ctx)?);
        }
        match and_conditions(conditions, ctx) {
            Some(condition) => filter_rel(cross, condition),
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
    for (&slot_id, expr) in project_node.common_slot_map.as_ref().into_iter().flatten() {
        let expression = {
            let mut expr_ctx = ctx.expr_context(&input.layout);
            expr.translate(&mut expr_ctx)?
        };
        input = append_project(input, expression, Some(SlotKey::new(output_tuple, slot_id)))?;
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
            let mut expr_ctx = ctx.expr_context(&input.layout);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
    }

    let layout = RowLayout::from_tuples(ctx.desc, &output_tuples)?;
    if expressions.len() != layout.len() {
        return Err(TranslateError::descriptor(format!(
            "projection has {} expressions for {} output slots",
            expressions.len(),
            layout.len()
        )));
    }
    let mut outputs = expressions
        .into_iter()
        .zip(layout.columns())
        .collect::<Vec<_>>();
    for slot_id in ctx.consumed_above.remove(&node.node_id).unwrap_or_default() {
        let key = SlotKey::new(output_tuple, slot_id);
        let expression = match slot_map.get(&slot_id) {
            Some(expr) => {
                let mut expr_ctx = ctx.expr_context(&input.layout);
                expr.translate(&mut expr_ctx)?
            }
            None => field_selection(input.layout.resolve(key)? as i32),
        };
        ctx.carried.insert(key);
        outputs.push((expression, Some(key)));
    }
    Ok(project_rel(input, outputs))
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
    let exchange_inputs = HashMap::new();
    let filter_inputs = HashMap::new();
    let mut ctx = PlanContext::new(
        desc,
        &scan_paths,
        &exchange_inputs,
        &filter_inputs,
        registry,
    );
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
        let mut expr_ctx = ctx.expr_context(&input.layout);
        expressions.push(expr.translate(&mut expr_ctx)?);
    }
    let bindings = exprs.iter().map(|expr| match expr.nodes.as_slice() {
        [node] if node.node_type == TExprNodeType::SLOT_REF => node
            .slot_ref
            .as_ref()
            .map(|slot| SlotKey::new(slot.tuple_id, slot.slot_id)),
        _ => None,
    });
    Ok(project_rel(
        input,
        expressions.into_iter().zip(bindings).collect(),
    ))
}

/// Pairs descriptor-ordered bindings with the expressions producing them.
fn project_bound_rel(
    input: TranslatedRel,
    expressions: Vec<Expression>,
    layout: RowLayout,
) -> Result<TranslatedRel> {
    if expressions.len() != layout.len() {
        return Err(TranslateError::descriptor(format!(
            "projection has {} expressions for {} output slots",
            expressions.len(),
            layout.len()
        )));
    }
    Ok(project_rel(
        input,
        expressions.into_iter().zip(layout.columns()).collect(),
    ))
}

/// Builds the project and its layout from the same ordered output list.
fn project_rel(input: TranslatedRel, outputs: Vec<(Expression, Option<SlotKey>)>) -> TranslatedRel {
    let base = input.layout.len() as i32;
    let output_mapping = (base..base + outputs.len() as i32).collect();
    let (expressions, bindings): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
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
        layout: RowLayout::new(bindings),
    }
}

/// Appends a temporary expression, optionally binding a common projection slot.
fn append_project(
    mut input: TranslatedRel,
    expression: Expression,
    binding: Option<SlotKey>,
) -> Result<TranslatedRel> {
    if let Some(key) = binding
        && input.layout.columns().any(|column| column == Some(key))
    {
        return Err(TranslateError::descriptor(format!(
            "common slot {} (tuple {}) already exists in the row layout",
            key.slot_id, key.tuple_id
        )));
    }
    input.layout.append(binding);
    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping: (0..input.layout.len() as i32).collect(),
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input.rel)),
                expressions: vec![expression],
                ..Default::default()
            }))),
        },
        layout: input.layout,
    })
}

/// Applies the same selection to the relation and its bindings.
fn emit_columns(input: TranslatedRel, output_mapping: Vec<i32>) -> Result<TranslatedRel> {
    let layout = input.layout.select(&output_mapping)?;
    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping,
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input.rel)),
                ..Default::default()
            }))),
        },
        layout,
    })
}

/// Derives output bindings from the join's actual output shape.
fn join_rel(
    left: TranslatedRel,
    right: TranslatedRel,
    condition: Expression,
    kind: join_rel::JoinType,
) -> Result<TranslatedRel> {
    let layout = match kind {
        join_rel::JoinType::LeftSemi => left.layout,
        join_rel::JoinType::RightSemi => right.layout,
        join_rel::JoinType::LeftMark => {
            let mut layout = left.layout;
            layout.append(None);
            layout
        }
        join_rel::JoinType::Inner
        | join_rel::JoinType::Left
        | join_rel::JoinType::Right
        | join_rel::JoinType::Outer => left.layout.concat(&right.layout),
        _ => {
            return Err(TranslateError::malformed(format!(
                "unsupported translated join kind {kind:?}"
            )));
        }
    };
    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
                left: Some(Box::new(left.rel)),
                right: Some(Box::new(right.rel)),
                expression: Some(Box::new(condition)),
                r#type: kind as i32,
                ..Default::default()
            }))),
        },
        layout,
    })
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

/// Builds an i64 literal for a Substrait fetch expression.
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

/// Wraps a relation in a filter without changing its row layout.
fn filter_rel(input: TranslatedRel, condition: Expression) -> TranslatedRel {
    TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(input.rel)),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        },
        layout: input.layout,
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
        let mut expr_ctx = ctx.expr_context(&input.layout);
        conditions.push(expr.translate(&mut expr_ctx)?);
    }
    let Some(condition) = and_conditions(conditions, ctx) else {
        return Ok(input);
    };

    Ok(filter_rel(input, condition))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unhandled_join_kind_cannot_claim_a_concatenated_layout() {
        let input = || TranslatedRel {
            rel: Rel::default(),
            layout: RowLayout::new([None]),
        };
        assert!(
            join_rel(
                input(),
                input(),
                i32_literal(1),
                join_rel::JoinType::RightAnti
            )
            .is_err()
        );
    }

    #[test]
    fn projected_layout_is_used_by_subsequent_joins_and_selections() {
        let first = SlotKey::new(0, 1);
        let second = SlotKey::new(0, 2);
        let right_key = SlotKey::new(1, 1);
        let input = TranslatedRel {
            rel: Rel::default(),
            layout: RowLayout::new([Some(first), Some(second)]),
        };
        let projected = project_rel(
            input,
            vec![
                (field_selection(1), Some(second)),
                (field_selection(1), Some(second)),
                (i32_literal(7), None),
            ],
        );
        assert_eq!(
            projected.layout.columns().collect::<Vec<_>>(),
            [Some(second), Some(second), None]
        );
        assert!(projected.layout.resolve(first).is_err());
        assert!(
            projected
                .layout
                .resolve(second)
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
        let right = TranslatedRel {
            rel: Rel::default(),
            layout: RowLayout::new([Some(right_key)]),
        };
        let joined = join_rel(projected, right, i32_literal(1), join_rel::JoinType::Inner).unwrap();
        assert_eq!(joined.layout.resolve(right_key).unwrap(), 3);
        let selected = emit_columns(joined, vec![3, 0, 2]).unwrap();
        assert_eq!(
            selected.layout.columns().collect::<Vec<_>>(),
            [Some(right_key), Some(second), None]
        );
        assert_eq!(selected.layout.resolve(second).unwrap(), 1);
    }
}
