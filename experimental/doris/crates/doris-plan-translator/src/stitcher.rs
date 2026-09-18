//! The single-plan stitcher (MVP-A0): the fragments the FE dispatched to this backend, joined
//! back into one plan tree.
//!
//! The FE cuts a query into plan fragments connected by exchanges: a sender fragment ends in
//! a `DATA_STREAM_SINK` whose `dest_node_id` names the `EXCHANGE_NODE` of the receiver. On a
//! single backend with one instance per fragment (`parallel_pipeline_task_num=1`) every
//! exchange has exactly one sender, so the query is a tree of fragments rooted at the
//! `RESULT_SINK` fragment, and the tree can be flattened back into one fragment:
//!
//! - an `EXCHANGE_NODE` is replaced by its sender's plan subtree (the sender's root emits
//!   the exchange's row layout: same tuple ids, same order);
//! - a merging exchange (`sort_info`, with the top-N `limit`/`offset`) becomes a `SORT_NODE`
//!   over that subtree, so the merge order and limit are kept;
//! - a two-phase aggregate — `AGGREGATION_NODE` (merge) over `EXCHANGE_NODE` over
//!   `AGGREGATION_NODE` (update) — collapses into one finalized aggregate: the update phase's
//!   grouping and aggregate expressions (over its input) with the merge phase's output tuple,
//!   `HAVING`, projections and limit. The two phases list their functions in different
//!   orders (Q1 merges `sum, sum, sum, sum, avg, avg, avg, count` over an update phase that
//!   computed `sum, avg, sum, count, ...`); each merge function names the partial-state slot
//!   it reads, and that slot's position in the update output tuple picks the update
//!   function. Doris' `partial_*` states never materialize, so `multi_distinct_count` simply
//!   becomes one `count(DISTINCT)`. A `SELECT DISTINCT` is the same shape with no functions
//!   at all (a two-phase group-by), recognised by the finalizing aggregate sitting directly
//!   on an update-phase one, and its grouping keys must read the update output tuple in
//!   order;
//! - the scan ranges of every fragment are merged into the one instance.
//!
//! The result is an ordinary `TPipelineFragmentParams` that [`crate::PlanTranslator`]
//! translates like any other fragment. Everything the stitcher cannot prove it can join —
//! a missing or duplicated sender, an exchange whose sender emits a different layout, a
//! limited non-merging exchange, a merge aggregate over anything but a matching update
//! aggregate — is refused rather than approximated.

use std::collections::{BTreeMap, HashMap};

use doris_thrift::data_sinks::TDataSinkType;
use doris_thrift::descriptors::TDescriptorTable;
use doris_thrift::exprs::{TExpr, TExprNodeType};
use doris_thrift::palo_internal_service::{TPipelineFragmentParams, TScanRangeParams};
use doris_thrift::plan_nodes::{TAggregationNode, TPlan, TPlanNode, TPlanNodeType, TSortNode};

use crate::error::{Result, TranslateError};

/// Slot ids of every tuple in wire order, straight from the descriptor table.
type TupleSlots = HashMap<i32, Vec<i32>>;

fn tuple_slots(desc_tbl: &TDescriptorTable) -> TupleSlots {
    let mut tuples: TupleSlots = HashMap::new();
    for slot in desc_tbl.slot_descriptors.iter().flatten() {
        tuples.entry(slot.parent).or_default().push(slot.id);
    }
    tuples
}

/// A plan node with its children, rebuilt from the flat preorder list.
#[derive(Clone, Debug)]
struct PlanTree {
    node: TPlanNode,
    children: Vec<PlanTree>,
}

impl PlanTree {
    /// Rebuilds the tree of a fragment's plan, rejecting a malformed preorder list.
    fn parse(plan: &TPlan) -> Result<Self> {
        let mut idx = 0;
        let tree = Self::parse_at(&plan.nodes, &mut idx)?;
        if idx != plan.nodes.len() {
            return Err(TranslateError::malformed(format!(
                "TPlan had {} trailing node(s)",
                plan.nodes.len() - idx
            )));
        }
        Ok(tree)
    }

    fn parse_at(nodes: &[TPlanNode], idx: &mut usize) -> Result<Self> {
        let node = nodes
            .get(*idx)
            .ok_or_else(|| TranslateError::malformed("unexpected end of plan nodes"))?
            .clone();
        *idx += 1;
        if node.num_children < 0 {
            return Err(TranslateError::malformed(format!(
                "node {} has negative child count {}",
                node.node_id, node.num_children
            )));
        }
        let children = (0..node.num_children)
            .map(|_| Self::parse_at(nodes, idx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { node, children })
    }

    /// Flattens the tree back into preorder with consistent child counts.
    fn flatten(self, out: &mut Vec<TPlanNode>) {
        let PlanTree { mut node, children } = self;
        node.num_children = children.len() as i32;
        out.push(node);
        for child in children {
            child.flatten(out);
        }
    }

    /// The row layout the subtree's root emits: its output tuple after projections, else
    /// its row tuples.
    fn output_layout(&self) -> Vec<i32> {
        match self.node.output_tuple_id {
            Some(tuple_id) if self.node.projections.is_some() => vec![tuple_id],
            _ => self.node.row_tuples.clone(),
        }
    }
}

/// A sender fragment, keyed by the exchange node it feeds.
struct Sender<'a> {
    params: &'a TPipelineFragmentParams,
    tree: PlanTree,
}

/// Joins the fragments of one dispatch into a single fragment rooted at the `RESULT_SINK`.
pub fn stitch_fragments(fragments: &[&TPipelineFragmentParams]) -> Result<TPipelineFragmentParams> {
    let mut root = None;
    let mut senders: HashMap<i32, Sender<'_>> = HashMap::new();
    for params in fragments {
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
        let instances = params.local_params.as_deref().unwrap_or_default();
        if instances.len() != 1 {
            return Err(TranslateError::malformed(format!(
                "fragment {} has {} instances; the single-plan stitcher needs exactly one",
                params.fragment_id.unwrap_or(-1),
                instances.len()
            )));
        }
        let tree = PlanTree::parse(plan)?;
        let sink = fragment
            .output_sink
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "TPlanFragment",
                field: "output_sink",
            })?;
        match sink.type_ {
            TDataSinkType::RESULT_SINK => {
                if root.replace(*params).is_some() {
                    return Err(TranslateError::malformed(
                        "dispatch has more than one RESULT_SINK fragment",
                    ));
                }
            }
            TDataSinkType::DATA_STREAM_SINK => {
                let stream = sink
                    .stream_sink
                    .as_ref()
                    .ok_or(TranslateError::MissingField {
                        context: "TDataSink(DATA_STREAM_SINK)",
                        field: "stream_sink",
                    })?;
                if stream
                    .output_exprs
                    .as_ref()
                    .is_some_and(|exprs| !exprs.is_empty())
                    || stream
                        .conjuncts
                        .as_ref()
                        .is_some_and(|exprs| !exprs.is_empty())
                {
                    return Err(TranslateError::UnsupportedPlanNode {
                        node_id: stream.dest_node_id,
                        node_type: TPlanNodeType::EXCHANGE_NODE,
                        reason: "stream sinks with their own projections or conjuncts are not supported",
                    });
                }
                if senders
                    .insert(stream.dest_node_id, Sender { params, tree })
                    .is_some()
                {
                    return Err(TranslateError::malformed(format!(
                        "exchange node {} has more than one sender fragment",
                        stream.dest_node_id
                    )));
                }
            }
            other => {
                return Err(TranslateError::malformed(format!(
                    "fragment {} ends in an unsupported sink {other:?}",
                    params.fragment_id.unwrap_or(-1)
                )));
            }
        }
    }
    let root =
        root.ok_or_else(|| TranslateError::malformed("dispatch has no RESULT_SINK fragment"))?;
    let root_tree = PlanTree::parse(
        root.fragment
            .as_ref()
            .and_then(|fragment| fragment.plan.as_ref())
            .expect("checked above"),
    )?;

    let mut used = Vec::new();
    let stitched = splice(root_tree, &senders, &mut used)?;
    if used.len() != senders.len() {
        let unused: Vec<i32> = senders
            .keys()
            .filter(|exchange| !used.contains(exchange))
            .copied()
            .collect();
        return Err(TranslateError::malformed(format!(
            "sender fragments for exchange nodes {unused:?} are not reachable from the result fragment"
        )));
    }
    let desc_tbl = root.desc_tbl.as_ref().ok_or(TranslateError::MissingField {
        context: "TPipelineFragmentParams",
        field: "desc_tbl",
    })?;
    let stitched = collapse_two_phase_aggregates(stitched, &tuple_slots(desc_tbl))?;

    let mut nodes = Vec::new();
    stitched.flatten(&mut nodes);
    let mut params = root.clone();
    params.fragment.as_mut().expect("checked above").plan = Some(TPlan { nodes });

    // One instance carrying every fragment's scan ranges.
    let mut ranges: BTreeMap<i32, Vec<TScanRangeParams>> = BTreeMap::new();
    for fragment in fragments {
        for instance in fragment.local_params.iter().flatten() {
            for (node_id, node_ranges) in &instance.per_node_scan_ranges {
                if ranges.insert(*node_id, node_ranges.clone()).is_some() {
                    return Err(TranslateError::malformed(format!(
                        "scan node {node_id} has ranges in more than one fragment"
                    )));
                }
            }
        }
    }
    let instance = params
        .local_params
        .as_mut()
        .and_then(|instances| instances.first_mut())
        .expect("checked above");
    instance.per_node_scan_ranges = ranges;
    Ok(params)
}

/// Replaces every exchange in `tree` by its sender's (spliced) subtree.
fn splice(
    tree: PlanTree,
    senders: &HashMap<i32, Sender<'_>>,
    used: &mut Vec<i32>,
) -> Result<PlanTree> {
    let PlanTree { node, children } = tree;
    if node.node_type != TPlanNodeType::EXCHANGE_NODE {
        let children = children
            .into_iter()
            .map(|child| splice(child, senders, used))
            .collect::<Result<Vec<_>>>()?;
        return Ok(PlanTree { node, children });
    }
    let sender = senders.get(&node.node_id).ok_or_else(|| {
        TranslateError::malformed(format!(
            "exchange node {} has no sender fragment in this dispatch",
            node.node_id
        ))
    })?;
    if used.contains(&node.node_id) {
        return Err(TranslateError::malformed(format!(
            "exchange node {} appears twice in the plan",
            node.node_id
        )));
    }
    used.push(node.node_id);
    let _ = sender.params;
    let subtree = splice(sender.tree.clone(), senders, used)?;
    if subtree.output_layout() != node.row_tuples {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "sender fragment emits a different row layout than the exchange declares",
        });
    }
    let exchange = node
        .exchange_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "EXCHANGE_NODE",
            field: "exchange_node",
        })?;
    let offset = exchange.offset.unwrap_or(0);
    match &exchange.sort_info {
        // A merging exchange keeps its order and top-N as a sort over the sender's subtree.
        Some(sort_info) if !sort_info.ordering_exprs.is_empty() => {
            let sort = TPlanNode {
                node_type: TPlanNodeType::SORT_NODE,
                num_children: 1,
                exchange_node: None,
                sort_node: Some(TSortNode {
                    sort_info: sort_info.clone(),
                    use_top_n: node.limit >= 0,
                    offset: Some(offset),
                    merge_by_exchange: Some(true),
                    ..Default::default()
                }),
                ..node
            };
            Ok(PlanTree {
                node: sort,
                children: vec![subtree],
            })
        }
        _ if node.limit >= 0 || offset != 0 => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "limited non-merging exchange is not supported by the stitcher",
        }),
        _ if has_node_work(&node) => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "exchange with conjuncts or projections is not supported by the stitcher",
        }),
        _ => Ok(subtree),
    }
}

/// Whether a node carries conjuncts or projections of its own.
fn has_node_work(node: &TPlanNode) -> bool {
    node.conjuncts
        .as_ref()
        .is_some_and(|exprs| !exprs.is_empty())
        || node
            .projections
            .as_ref()
            .is_some_and(|exprs| !exprs.is_empty())
        || node
            .intermediate_projections_list
            .as_ref()
            .is_some_and(|lists| !lists.is_empty())
}

/// Collapses every `merge aggregate over update aggregate` pair into one finalized aggregate.
fn collapse_two_phase_aggregates(tree: PlanTree, tuples: &TupleSlots) -> Result<PlanTree> {
    let PlanTree { node, children } = tree;
    let mut children = children
        .into_iter()
        .map(|child| collapse_two_phase_aggregates(child, tuples))
        .collect::<Result<Vec<_>>>()?;
    if !is_merge_aggregate(&node, &children)? {
        return Ok(PlanTree { node, children });
    }
    if children.len() != 1 {
        return Err(TranslateError::malformed(format!(
            "aggregate node {} has {} children",
            node.node_id,
            children.len()
        )));
    }
    let update = children.pop().unwrap();
    let update_agg = update_aggregate(&update.node);
    let Some(update_agg) = update_agg else {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "merge-phase aggregate whose input is not its update-phase aggregate",
        });
    };
    if has_node_work(&update.node) || update.node.limit >= 0 {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: update.node.node_id,
            node_type: update.node.node_type,
            reason: "update-phase aggregate with conjuncts, projections or a limit",
        });
    }
    let merge_agg = node
        .agg_node
        .as_ref()
        .expect("checked by is_merge_aggregate");
    let merge_keys = merge_agg.grouping_exprs.as_ref().map_or(0, Vec::len);
    let update_keys = update_agg.grouping_exprs.as_ref().map_or(0, Vec::len);
    if merge_keys != update_keys {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "merge- and update-phase aggregates do not have the same grouping keys",
        });
    }
    if merge_agg.aggregate_functions.len() != update_agg.aggregate_functions.len() {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "merge- and update-phase aggregates do not have the same number of functions",
        });
    }
    // The merge phase reads the update output tuple: grouping key `i` is that tuple's slot
    // `i`, and each merge function reads one partial-state slot, whose position (after the
    // grouping keys) is the update function it continues.
    let update_slots = tuples.get(&update_agg.output_tuple_id).ok_or_else(|| {
        TranslateError::descriptor(format!("tuple {} not found", update_agg.output_tuple_id))
    })?;
    for (index, expr) in merge_agg.grouping_exprs.iter().flatten().enumerate() {
        let reads_key = expr
            .nodes
            .first()
            .filter(|root| root.num_children == 0)
            .and_then(|root| root.slot_ref.as_ref())
            .is_some_and(|slot_ref| {
                slot_ref.tuple_id == update_agg.output_tuple_id
                    && update_slots.get(index) == Some(&slot_ref.slot_id)
            });
        if !reads_key {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "merge-phase grouping key does not read the matching update output slot",
            });
        }
    }
    let update_names = aggregate_names(&update_agg.aggregate_functions)?;
    let mut functions = Vec::with_capacity(merge_agg.aggregate_functions.len());
    for expr in &merge_agg.aggregate_functions {
        let root = &expr.nodes[0];
        let slot_ref = expr
            .nodes
            .get(1)
            .filter(|_| root.num_children == 1)
            .and_then(|child| child.slot_ref.as_ref())
            .filter(|slot_ref| slot_ref.tuple_id == update_agg.output_tuple_id);
        let Some(slot_ref) = slot_ref else {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "merge-phase function does not read one partial-state slot of the update output tuple",
            });
        };
        let position = update_slots
            .iter()
            .position(|slot_id| *slot_id == slot_ref.slot_id)
            .and_then(|position| position.checked_sub(update_keys))
            .filter(|index| *index < update_agg.aggregate_functions.len());
        let Some(index) = position else {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "merge-phase function reads a slot that is not an update-phase state",
            });
        };
        let merge_name = root
            .fn_
            .as_ref()
            .map(|function| function.name.function_name.as_str())
            .unwrap_or_default();
        if merge_name != update_names[index] {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: node.node_id,
                node_type: node.node_type,
                reason: "merge- and update-phase functions on the same state differ",
            });
        }
        functions.push(update_agg.aggregate_functions[index].clone());
    }
    // The finalized node: the merge node's identity, output tuple, HAVING, projections and
    // limit, evaluated over the update node's input with its expressions in merge order.
    let mut collapsed = node;
    let agg = collapsed.agg_node.as_mut().expect("checked");
    agg.grouping_exprs = update_agg.grouping_exprs.clone();
    agg.aggregate_functions = functions;
    agg.need_finalize = true;
    agg.is_first_phase = Some(true);
    Ok(PlanTree {
        node: collapsed,
        children: update.children,
    })
}

/// The aggregate description of an update-phase (`need_finalize=false`) aggregation node.
fn update_aggregate(node: &TPlanNode) -> Option<&TAggregationNode> {
    node.agg_node
        .as_ref()
        .filter(|_| node.node_type == TPlanNodeType::AGGREGATION_NODE)
        .filter(|agg| !agg.need_finalize)
}

/// Whether a node is a finalizing aggregate whose measures merge partial states.
///
/// A `SELECT DISTINCT` / `GROUP BY` without measures has no `is_merge_agg` function to tell
/// its merge phase from a single-phase aggregate (both have `need_finalize=true`,
/// `is_first_phase=false`); there the merge phase is the finalizing aggregate sitting
/// directly on its update phase, which is the only thing an update phase ever feeds.
fn is_merge_aggregate(node: &TPlanNode, children: &[PlanTree]) -> Result<bool> {
    let Some(agg) = &node.agg_node else {
        return Ok(false);
    };
    if node.node_type != TPlanNodeType::AGGREGATION_NODE || !agg.need_finalize {
        return Ok(false);
    }
    if agg.aggregate_functions.is_empty() {
        return Ok(children.len() == 1 && update_aggregate(&children[0].node).is_some());
    }
    let mut merges = 0;
    for expr in &agg.aggregate_functions {
        let root = expr
            .nodes
            .first()
            .ok_or_else(|| TranslateError::malformed("aggregate function TExpr is empty"))?;
        if root
            .agg_expr
            .as_ref()
            .is_some_and(|agg_expr| agg_expr.is_merge_agg)
        {
            merges += 1;
        }
    }
    if merges != 0 && merges != agg.aggregate_functions.len() {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "aggregate mixes merge-phase and update-phase functions",
        });
    }
    Ok(merges > 0)
}

/// The function names of a list of `AGG_EXPR` roots.
fn aggregate_names(exprs: &[TExpr]) -> Result<Vec<String>> {
    exprs
        .iter()
        .map(|expr| {
            let root = expr
                .nodes
                .first()
                .ok_or_else(|| TranslateError::malformed("aggregate function TExpr is empty"))?;
            if root.node_type != TExprNodeType::AGG_EXPR {
                return Err(TranslateError::UnsupportedExpression {
                    node_type: root.node_type,
                    reason: "aggregate function root is not an AGG_EXPR",
                });
            }
            Ok(root
                .fn_
                .as_ref()
                .map(|function| function.name.function_name.clone())
                .unwrap_or_default())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use doris_thrift::data_sinks::{TDataSink, TDataStreamSink};
    use doris_thrift::descriptors::{TSlotDescriptor, TTupleDescriptor};
    use doris_thrift::exprs::{TAggregateExpr, TExprNode, TSlotRef};
    use doris_thrift::palo_internal_service::TPipelineInstanceParams;
    use doris_thrift::partitions::{TDataPartition, TPartitionType};
    use doris_thrift::plan_nodes::{TAggregationNode, TExchangeNode, TSortInfo};
    use doris_thrift::planner::TPlanFragment;
    use doris_thrift::types::{TFunction, TFunctionName, TUniqueId};

    use super::*;

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

    fn exchange(node_id: i32, row_tuples: Vec<i32>) -> TPlanNode {
        TPlanNode {
            exchange_node: Some(TExchangeNode {
                input_row_tuples: row_tuples.clone(),
                ..Default::default()
            }),
            ..node(node_id, TPlanNodeType::EXCHANGE_NODE, 0, row_tuples)
        }
    }

    /// An `AGG_EXPR` root; a merge-phase one reads partial slot `(tuple, slot)`.
    fn agg_expr(name: &str, partial: Option<(i32, i32)>) -> TExpr {
        let mut nodes = vec![TExprNode {
            node_type: TExprNodeType::AGG_EXPR,
            num_children: partial.is_some() as i32,
            output_scale: -1,
            agg_expr: Some(TAggregateExpr {
                is_merge_agg: partial.is_some(),
                param_types: None,
            }),
            fn_: Some(TFunction {
                name: TFunctionName {
                    db_name: None,
                    function_name: name.to_string(),
                },
                ..Default::default()
            }),
            ..Default::default()
        }];
        if let Some((tuple_id, slot_id)) = partial {
            nodes.push(TExprNode {
                node_type: TExprNodeType::SLOT_REF,
                num_children: 0,
                output_scale: -1,
                slot_ref: Some(TSlotRef {
                    slot_id,
                    tuple_id,
                    ..Default::default()
                }),
                ..Default::default()
            });
        }
        TExpr { nodes }
    }

    fn aggregation(
        node_id: i32,
        need_finalize: bool,
        functions: Vec<TExpr>,
        tuple: i32,
    ) -> TPlanNode {
        TPlanNode {
            agg_node: Some(TAggregationNode {
                grouping_exprs: Some(vec![]),
                aggregate_functions: functions,
                intermediate_tuple_id: tuple,
                output_tuple_id: tuple,
                need_finalize,
                ..Default::default()
            }),
            ..node(node_id, TPlanNodeType::AGGREGATION_NODE, 1, vec![tuple])
        }
    }

    /// Update output tuple 5 = {50: partial count, 51: partial sum}; merge output tuple 6.
    fn desc_tbl() -> TDescriptorTable {
        let slot = |id, parent| TSlotDescriptor {
            id,
            parent,
            column_pos: -1,
            slot_idx: -1,
            is_materialized: true,
            ..Default::default()
        };
        TDescriptorTable {
            slot_descriptors: Some(vec![slot(50, 5), slot(51, 5), slot(60, 6), slot(61, 6)]),
            tuple_descriptors: [0, 5, 6]
                .into_iter()
                .map(|id| TTupleDescriptor {
                    id,
                    ..Default::default()
                })
                .collect(),
            table_descriptors: None,
        }
    }

    fn fragment(
        fragment_id: i32,
        sink: TDataSinkType,
        dest_node_id: i32,
        nodes: Vec<TPlanNode>,
        scan_node: Option<i32>,
    ) -> TPipelineFragmentParams {
        let stream_sink = (sink == TDataSinkType::DATA_STREAM_SINK).then(|| TDataStreamSink {
            dest_node_id,
            output_partition: TDataPartition {
                type_: TPartitionType::UNPARTITIONED,
                ..Default::default()
            },
            ..Default::default()
        });
        TPipelineFragmentParams {
            query_id: TUniqueId::new(1, 2),
            fragment_id: Some(fragment_id),
            desc_tbl: Some(desc_tbl()),
            fragment: Some(TPlanFragment {
                plan: Some(TPlan { nodes }),
                output_sink: Some(TDataSink {
                    type_: sink,
                    stream_sink,
                    ..Default::default()
                }),
                partition: TDataPartition {
                    type_: TPartitionType::UNPARTITIONED,
                    ..Default::default()
                },
                ..Default::default()
            }),
            local_params: Some(vec![TPipelineInstanceParams {
                fragment_instance_id: TUniqueId::new(1, fragment_id as i64),
                per_node_scan_ranges: scan_node
                    .map(|node_id| (node_id, vec![TScanRangeParams::default()]))
                    .into_iter()
                    .collect(),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn scan(node_id: i32, tuple: i32) -> TPlanNode {
        node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple])
    }

    fn shape(params: &TPipelineFragmentParams) -> Vec<(i32, TPlanNodeType, i32)> {
        params
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes
            .iter()
            .map(|n| (n.node_id, n.node_type, n.num_children))
            .collect()
    }

    /// Q1's shape: SORT > EXCHANGE(4) ← AGG(merge) > EXCHANGE(2) ← AGG(update) > SCAN, with
    /// the merge phase listing its functions in the opposite order of the update phase.
    fn two_phase_dispatch() -> Vec<TPipelineFragmentParams> {
        let mut sort = node(5, TPlanNodeType::SORT_NODE, 1, vec![6]);
        sort.sort_node = Some(TSortNode::default());
        vec![
            fragment(
                2,
                TDataSinkType::RESULT_SINK,
                0,
                vec![sort, exchange(4, vec![6])],
                None,
            ),
            fragment(
                1,
                TDataSinkType::DATA_STREAM_SINK,
                4,
                vec![
                    aggregation(
                        3,
                        true,
                        vec![
                            agg_expr("sum", Some((5, 51))),
                            agg_expr("count", Some((5, 50))),
                        ],
                        6,
                    ),
                    exchange(2, vec![5]),
                ],
                None,
            ),
            fragment(
                0,
                TDataSinkType::DATA_STREAM_SINK,
                2,
                vec![
                    aggregation(
                        1,
                        false,
                        vec![agg_expr("count", None), agg_expr("sum", None)],
                        5,
                    ),
                    scan(0, 0),
                ],
                Some(0),
            ),
        ]
    }

    #[test]
    fn splices_senders_and_collapses_a_two_phase_aggregate() {
        let fragments = two_phase_dispatch();
        let refs: Vec<_> = fragments.iter().collect();
        let stitched = stitch_fragments(&refs).unwrap();
        assert_eq!(
            shape(&stitched),
            vec![
                (5, TPlanNodeType::SORT_NODE, 1),
                (3, TPlanNodeType::AGGREGATION_NODE, 1),
                (0, TPlanNodeType::FILE_SCAN_NODE, 0),
            ]
        );
        let nodes = &stitched
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes;
        let agg = nodes[1].agg_node.as_ref().unwrap();
        assert!(agg.need_finalize);
        assert_eq!(agg.output_tuple_id, 6);
        // Merge order (sum, count), each the update phase's own (non-merge) expression.
        let names: Vec<_> = agg
            .aggregate_functions
            .iter()
            .map(|expr| {
                let root = &expr.nodes[0];
                assert!(!root.agg_expr.as_ref().unwrap().is_merge_agg);
                root.fn_.as_ref().unwrap().name.function_name.clone()
            })
            .collect();
        assert_eq!(names, vec!["sum", "count"]);
        // The scan's ranges moved into the root fragment's instance.
        let instance = &stitched.local_params.as_ref().unwrap()[0];
        assert_eq!(
            instance.per_node_scan_ranges.keys().collect::<Vec<_>>(),
            vec![&0]
        );
        assert_eq!(stitched.fragment_id, Some(2));
    }

    /// A grouping key `SLOT_REF(tuple, slot)` expression.
    fn key_ref(tuple_id: i32, slot_id: i32) -> TExpr {
        TExpr {
            nodes: vec![TExprNode {
                node_type: TExprNodeType::SLOT_REF,
                num_children: 0,
                output_scale: -1,
                slot_ref: Some(TSlotRef {
                    slot_id,
                    tuple_id,
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }
    }

    /// `SELECT DISTINCT k` as the FE plans it: a finalizing group-by without functions over
    /// an update-phase group-by (update output tuple 5 = {50: k}, merge output tuple 6).
    fn distinct_dispatch() -> Vec<TPipelineFragmentParams> {
        let mut merge = aggregation(3, true, vec![], 6);
        merge.agg_node.as_mut().unwrap().grouping_exprs = Some(vec![key_ref(5, 50)]);
        let mut update = aggregation(1, false, vec![], 5);
        update.agg_node.as_mut().unwrap().grouping_exprs = Some(vec![key_ref(0, 7)]);
        vec![
            fragment(
                2,
                TDataSinkType::RESULT_SINK,
                0,
                vec![exchange(4, vec![6])],
                None,
            ),
            fragment(
                1,
                TDataSinkType::DATA_STREAM_SINK,
                4,
                vec![merge, exchange(2, vec![5])],
                None,
            ),
            fragment(
                0,
                TDataSinkType::DATA_STREAM_SINK,
                2,
                vec![update, scan(0, 0)],
                Some(0),
            ),
        ]
    }

    #[test]
    fn collapses_a_two_phase_distinct_without_functions() {
        let fragments = distinct_dispatch();
        let refs: Vec<_> = fragments.iter().collect();
        let stitched = stitch_fragments(&refs).unwrap();
        assert_eq!(
            shape(&stitched),
            vec![
                (3, TPlanNodeType::AGGREGATION_NODE, 1),
                (0, TPlanNodeType::FILE_SCAN_NODE, 0),
            ]
        );
        let agg = stitched
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes[0]
            .agg_node
            .clone()
            .unwrap();
        assert!(agg.need_finalize);
        assert_eq!(agg.output_tuple_id, 6);
        assert!(agg.aggregate_functions.is_empty());
        // The key is now the update phase's, read from the scan tuple.
        assert_eq!(agg.grouping_exprs, Some(vec![key_ref(0, 7)]));
    }

    #[test]
    fn merge_grouping_keys_must_read_the_update_output_in_order() {
        // The merge key reads a slot that is not the update phase's key.
        let mut fragments = distinct_dispatch();
        fragments[1]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .agg_node
            .as_mut()
            .unwrap()
            .grouping_exprs = Some(vec![key_ref(5, 51)]);
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 3, reason, .. } if reason.contains("grouping key")
        ));
        // A finalizing group-by without functions over a non-aggregate child is a plain
        // single-phase aggregate and is left alone.
        let mut fragments = distinct_dispatch();
        let leaf = &mut fragments[2]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap();
        leaf.nodes.remove(0);
        leaf.nodes[0].row_tuples = vec![5];
        let refs: Vec<_> = fragments.iter().collect();
        let stitched = stitch_fragments(&refs).unwrap();
        assert_eq!(
            shape(&stitched),
            vec![
                (3, TPlanNodeType::AGGREGATION_NODE, 1),
                (0, TPlanNodeType::FILE_SCAN_NODE, 0),
            ]
        );
        let agg = stitched
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes[0]
            .agg_node
            .clone()
            .unwrap();
        assert_eq!(agg.grouping_exprs, Some(vec![key_ref(5, 50)]));
    }

    #[test]
    fn merging_exchange_becomes_a_top_n_sort_over_the_sender() {
        let mut fragments = two_phase_dispatch();
        let root_nodes = &mut fragments[0]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes;
        root_nodes.remove(0);
        let exchange = &mut root_nodes[0];
        exchange.limit = 10;
        exchange.exchange_node.as_mut().unwrap().sort_info = Some(TSortInfo {
            ordering_exprs: vec![agg_expr("unused", None)],
            is_asc_order: vec![true],
            nulls_first: vec![true],
            ..Default::default()
        });
        exchange.exchange_node.as_mut().unwrap().offset = Some(2);
        let refs: Vec<_> = fragments.iter().collect();
        let stitched = stitch_fragments(&refs).unwrap();
        assert_eq!(
            shape(&stitched),
            vec![
                (4, TPlanNodeType::SORT_NODE, 1),
                (3, TPlanNodeType::AGGREGATION_NODE, 1),
                (0, TPlanNodeType::FILE_SCAN_NODE, 0),
            ]
        );
        let sort = &stitched
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes[0];
        assert_eq!(sort.limit, 10);
        assert_eq!(sort.sort_node.as_ref().unwrap().offset, Some(2));
        assert!(sort.exchange_node.is_none());
        assert_eq!(sort.row_tuples, vec![6]);
    }

    #[test]
    fn stitching_gates() {
        // A limited exchange without a merge order.
        let mut fragments = two_phase_dispatch();
        fragments[0]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[1]
            .limit = 10;
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 4, .. }
        ));
        // A sender whose layout differs from the exchange.
        let mut fragments = two_phase_dispatch();
        fragments[0]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[1]
            .row_tuples = vec![9];
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 4, reason, .. } if reason.contains("layout")
        ));
        // A missing sender.
        let fragments = two_phase_dispatch();
        let refs: Vec<_> = fragments.iter().take(2).collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::MalformedPlan(msg) if msg.contains("no sender")
        ));
        // A sender nobody reads.
        let mut fragments = two_phase_dispatch();
        fragments.push(fragment(
            9,
            TDataSinkType::DATA_STREAM_SINK,
            99,
            vec![scan(9, 0)],
            None,
        ));
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::MalformedPlan(msg) if msg.contains("not reachable")
        ));
        // No result fragment.
        let fragments = two_phase_dispatch();
        let refs: Vec<_> = fragments.iter().skip(1).collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::MalformedPlan(msg) if msg.contains("RESULT_SINK")
        ));
        // Merge aggregate over something that is not its update aggregate.
        let mut fragments = two_phase_dispatch();
        fragments[2]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .agg_node
            .as_mut()
            .unwrap()
            .need_finalize = true;
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 3, reason, .. } if reason.contains("update-phase")
        ));
        // A merge function whose state slot holds a different function.
        let mut fragments = two_phase_dispatch();
        fragments[2]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .agg_node
            .as_mut()
            .unwrap()
            .aggregate_functions = vec![agg_expr("sum", None), agg_expr("count", None)];
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 3, reason, .. } if reason.contains("differ")
        ));
        // A merge function that does not read a partial-state slot.
        let mut fragments = two_phase_dispatch();
        fragments[1]
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .agg_node
            .as_mut()
            .unwrap()
            .aggregate_functions[0] = agg_expr("sum", Some((6, 60)));
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::UnsupportedPlanNode { node_id: 3, reason, .. } if reason.contains("partial-state slot")
        ));
        // Two instances.
        let mut fragments = two_phase_dispatch();
        let instances = fragments[2].local_params.as_mut().unwrap();
        instances.push(instances[0].clone());
        let refs: Vec<_> = fragments.iter().collect();
        assert!(matches!(
            stitch_fragments(&refs).unwrap_err(),
            TranslateError::MalformedPlan(msg) if msg.contains("instances")
        ));
    }
}
