//! Same-node fragment fusion: splicing a sender fragment's plan over the receiver's EXCHANGE_NODE.
//!
//! The FE cuts one query into fragments and connects them with exchanges. When a sender has one
//! destination on this CN and the receiving exchange expects one sender, the exchange is the
//! identity on rows (the CN treats a single destination as a gather regardless of the partition
//! label), so the sender's node list can replace the exchange node in the receiver's flat preorder
//! plan. Everything the translator reads per fragment (`per_node_scan_ranges`,
//! `node_to_per_driver_seq_scan_ranges`, `per_exch_num_senders`) is unioned by query-global node
//! id. The row layout is guaranteed by the FE: an ExchangeNode's tuple ids are its child's
//! (`ExchangeNode.java` `computeTupleIds`), shipped as `TExchangeNode.input_row_tuples`.
//!
//! Everything here is a pure function over the thrift structs: no I/O, no engine, no policy. The
//! CN decides *which* senders to offer (the `SIRIUS_CN_FRAGMENT_FUSION` mode, destination
//! routing, rendezvous state: `compute_node_service.rs` `try_defer_sender`, which parks the
//! sender's plan as a `SenderSource::LocalPlan` and splices it in `fold_deferred_plans` when the
//! receiver becomes ready); this module decides whether an offered edge is structurally sound and
//! performs the splice.
//!
//! The invariants the rest of this crate relies on survive the splice by construction or by
//! refusal: the exchange is a preorder leaf and the sender list a complete preorder subtree, so
//! every ancestor's `num_children` stays valid (`PlanNodeCursor::ensure_consumed`); a merge
//! aggregation keeps reading from a real exchange ([`FusionRefusal::AggregationParent`]); a partial
//! aggregation stays a fragment root ([`FusionRefusal::PartialAggregationRoot`]); and no carried
//! common-slot column crosses into the receiver ([`FusionRefusal::CommonSlotProjection`]).

use std::collections::{BTreeMap, BTreeSet};

use starrocks_thrift::data_sinks::{TDataSinkType, TPlanFragmentDestination};
use starrocks_thrift::internal_service::{TExecPlanFragmentParams, TPlanFragmentExecParams};
use starrocks_thrift::partitions::TPartitionType;
use starrocks_thrift::plan_nodes::{TPlanNode, TPlanNodeType};

use crate::agg_phase::{self, AggPhase};

/// Why an exchange keeps its stream boundary instead of absorbing its sender. Every variant is a
/// logged decline on the CN, never a query failure: a refused sender runs and parks as it does
/// without fusion.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FusionRefusal {
    // Sender side (known from the sender alone).
    /// `fragment`, `plan`, `params`, `output_sink` or `stream_sink` is absent on the sender.
    #[error("sender is missing {0}")]
    SenderMissingField(&'static str),
    /// The sender's output sink is not a `DATA_STREAM_SINK`.
    #[error("sender output sink is not a DATA_STREAM_SINK")]
    NotDataStreamSink,
    /// The sink fans out (or has no destination at all); only a single destination is the
    /// identity on rows.
    #[error("sender sink has {destinations} destination(s), not exactly one")]
    NotSingleDestination {
        /// Destination count on the sender's exec params.
        destinations: usize,
    },
    /// `stream_sink.limit >= 0`: the sink truncates its output.
    #[error("sender stream sink carries a limit")]
    SinkLimit,
    /// `stream_sink.output_columns` is non-empty: the sink reprojects its output.
    #[error("sender stream sink carries output_columns")]
    SinkOutputColumns,
    /// `fragment.output_exprs` is non-empty: the sender reprojects its root.
    #[error("sender fragment carries output_exprs")]
    SenderOutputExprs,
    /// The sender's node list is empty, under-runs or has trailing nodes.
    #[error("sender plan is malformed: {0}")]
    MalformedSenderPlan(String),
    /// A `PROJECT_NODE` with a non-empty `common_slot_map`: its carried columns are a
    /// fragment-internal mechanism the receiver's ancestors were never planned against.
    #[error("sender project node {node_id} carries a common_slot_map")]
    CommonSlotProjection {
        /// The project node.
        node_id: i32,
    },
    /// The sender root is an aggregation that emits partial states (or one the translator cannot
    /// classify): partial states must cross a real exchange into a merge aggregation.
    #[error("sender root aggregation node {node_id} emits partial states")]
    PartialAggregationRoot {
        /// The aggregation node.
        node_id: i32,
    },
    // Receiver side (needs the receiver's params).
    /// `fragment`, `plan`, `params` or the exchange's `exchange_node` payload is absent on the
    /// receiver.
    #[error("receiver is missing {0}")]
    ReceiverMissingField(&'static str),
    /// No `EXCHANGE_NODE` with that id in the receiver plan.
    #[error("receiver plan has no EXCHANGE_NODE {node_id}")]
    ExchangeMissing {
        /// The exchange node id the sender addresses.
        node_id: i32,
    },
    /// `num_children != 0` on the exchange (malformed).
    #[error("exchange {node_id} has children")]
    ExchangeHasChildren {
        /// The exchange node.
        node_id: i32,
    },
    /// `node.limit >= 0` on the exchange.
    #[error("exchange {node_id} carries a limit")]
    ExchangeLimit {
        /// The exchange node.
        node_id: i32,
    },
    /// `exchange_node.offset > 0`.
    #[error("exchange {node_id} carries an offset")]
    ExchangeOffset {
        /// The exchange node.
        node_id: i32,
    },
    /// The exchange filters its input.
    #[error("exchange {node_id} carries conjuncts")]
    ExchangeConjuncts {
        /// The exchange node.
        node_id: i32,
    },
    /// `exchange_node.sort_info` is set: a merging exchange.
    #[error("exchange {node_id} merges sorted input")]
    SortedExchange {
        /// The exchange node.
        node_id: i32,
    },
    /// The exchange's preorder parent is an `AGGREGATION_NODE` of any phase: a merge aggregation
    /// must keep reading partial states from a real exchange, and no TPC-H plan feeds a non-merge
    /// aggregation from an exchange, so the broader rule costs nothing and is simpler to reason
    /// about.
    #[error("exchange {exchange} feeds aggregation node {parent}")]
    AggregationParent {
        /// The exchange node.
        exchange: i32,
        /// The aggregation node above it.
        parent: i32,
    },
    /// The exchange's `input_row_tuples` differ from the sender root's `row_tuples`.
    #[error("exchange input row tuples {exchange:?} differ from the sender root's {sender_root:?}")]
    RowTuplesDiffer {
        /// The exchange's declared input layout.
        exchange: Vec<i32>,
        /// The sender root's row layout.
        sender_root: Vec<i32>,
    },
    /// The receiver's `desc_tbl` is absent or has no tuple descriptors (a cached reference the CN
    /// has not resolved).
    #[error("receiver descriptor table is unresolved")]
    ReceiverDescriptorUnresolved,
    /// A sender node's row tuple is not in the receiver's descriptor table.
    #[error("receiver descriptor table lacks tuple {tuple_id}")]
    DescriptorMissingTuple {
        /// The undescribed tuple id.
        tuple_id: i32,
    },
    /// A scan node has scan ranges in both the sender's and the receiver's maps.
    #[error("scan node {node_id} has scan ranges on both sender and receiver")]
    ScanRangeCollision {
        /// The scan node.
        node_id: i32,
    },
    /// An exchange id is declared in both `per_exch_num_senders` maps.
    #[error("exchange {node_id} is declared by both sender and receiver")]
    ExchangeIdCollision {
        /// The exchange node.
        node_id: i32,
    },
}

/// What the CN needs from a sender to apply its policy and routing. Borrowed from the params.
#[derive(Clone, Copy, Debug)]
pub struct SenderShape<'a> {
    /// The receiver `EXCHANGE_NODE` the sink addresses.
    pub dest_node_id: i32,
    /// The sink's only destination.
    pub destination: &'a TPlanFragmentDestination,
    /// `params.sender_id.unwrap_or(0)`, the way the CN keys sender slots.
    pub sender_id: i32,
    /// `stream_sink.output_partition.type_`.
    pub partition: TPartitionType,
    /// No `EXCHANGE_NODE` in the sender plan: it reads only files.
    pub is_leaf: bool,
}

type Refused<T> = Result<T, FusionRefusal>;

/// Sender-only checks: sink shape, output_exprs, plan well-formedness, common-slot projections,
/// partial-aggregation root. Does not look at any receiver.
pub fn sender_shape(sender: &TExecPlanFragmentParams) -> Refused<SenderShape<'_>> {
    let fragment = sender
        .fragment
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("fragment"))?;
    let nodes = fragment
        .plan
        .as_ref()
        .map(|plan| plan.nodes.as_slice())
        .ok_or(FusionRefusal::SenderMissingField("plan"))?;
    let exec = sender
        .params
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("params"))?;
    let sink = fragment
        .output_sink
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("output_sink"))?;
    if sink.type_ != TDataSinkType::DATA_STREAM_SINK {
        return Err(FusionRefusal::NotDataStreamSink);
    }
    let stream_sink = sink
        .stream_sink
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("stream_sink"))?;
    let destinations = exec.destinations.as_deref().unwrap_or_default();
    let [destination] = destinations else {
        return Err(FusionRefusal::NotSingleDestination {
            destinations: destinations.len(),
        });
    };
    // The same three sink-side transformations the CN's run path treats specially: refusing them
    // here guarantees the splice never drops one.
    if stream_sink.limit.is_some_and(|limit| limit >= 0) {
        return Err(FusionRefusal::SinkLimit);
    }
    if stream_sink
        .output_columns
        .as_ref()
        .is_some_and(|columns| !columns.is_empty())
    {
        return Err(FusionRefusal::SinkOutputColumns);
    }
    if fragment
        .output_exprs
        .as_ref()
        .is_some_and(|exprs| !exprs.is_empty())
    {
        return Err(FusionRefusal::SenderOutputExprs);
    }
    let Some(root) = nodes.first() else {
        return Err(FusionRefusal::MalformedSenderPlan(
            "TPlan.nodes is empty".to_string(),
        ));
    };
    let span = preorder_span(nodes, 0)?;
    if span != nodes.len() {
        return Err(FusionRefusal::MalformedSenderPlan(format!(
            "TPlan has {} trailing node(s)",
            nodes.len() - span
        )));
    }
    if let Some(project) = nodes.iter().find(|node| {
        node.node_type == TPlanNodeType::PROJECT_NODE
            && node
                .project_node
                .as_ref()
                .and_then(|project| project.common_slot_map.as_ref())
                .is_some_and(|common| !common.is_empty())
    }) {
        return Err(FusionRefusal::CommonSlotProjection {
            node_id: project.node_id,
        });
    }
    // A finalized (one-shot) aggregation emits plain rows and may cross the boundary; a merge
    // root is left to the translator (it needs an exchange below it, which a fused plan keeps).
    // Anything else emits partial states, or is a phase the translator refuses on every path:
    // declining keeps that failure attributed to the sender.
    if root.node_type == TPlanNodeType::AGGREGATION_NODE
        && let Some(agg) = root.agg_node.as_ref()
        && !matches!(
            agg_phase::classify(root.node_id, root.node_type, agg),
            Ok(AggPhase::OneShot | AggPhase::Merge)
        )
    {
        return Err(FusionRefusal::PartialAggregationRoot {
            node_id: root.node_id,
        });
    }
    Ok(SenderShape {
        dest_node_id: stream_sink.dest_node_id,
        destination,
        sender_id: exec.sender_id.unwrap_or(0),
        partition: stream_sink.output_partition.type_,
        is_leaf: !nodes
            .iter()
            .any(|node| node.node_type == TPlanNodeType::EXCHANGE_NODE),
    })
}

/// Receiver-and-sender checks for the edge `sender -> receiver.EXCHANGE_NODE(exchange_node_id)`.
/// Pure; the CN calls it under the rendezvous lock with the registered receiver's
/// (descriptor-resolved) params, after [`sender_shape`] passed for the same sender.
///
/// In order: the exchange exists, is a leaf, is plain (no limit, offset, conjuncts, sort), does
/// not feed an aggregation; its input layout is the sender root's; the receiver's descriptor
/// table is resolved and describes every sender tuple; the two fragments' scan-range and
/// exchange declarations do not collide.
pub fn fusable_edge(
    receiver: &TExecPlanFragmentParams,
    exchange_node_id: i32,
    sender: &TExecPlanFragmentParams,
) -> Refused<()> {
    let nodes = receiver_nodes(receiver)?;
    let position = exchange_position(nodes, exchange_node_id)?;
    let node = &nodes[position];
    if node.num_children != 0 {
        return Err(FusionRefusal::ExchangeHasChildren {
            node_id: exchange_node_id,
        });
    }
    if node.limit >= 0 {
        return Err(FusionRefusal::ExchangeLimit {
            node_id: exchange_node_id,
        });
    }
    let exchange = node
        .exchange_node
        .as_ref()
        .ok_or(FusionRefusal::ReceiverMissingField("exchange_node"))?;
    if exchange.offset.unwrap_or(0) != 0 {
        return Err(FusionRefusal::ExchangeOffset {
            node_id: exchange_node_id,
        });
    }
    if node
        .conjuncts
        .as_ref()
        .is_some_and(|conjuncts| !conjuncts.is_empty())
    {
        return Err(FusionRefusal::ExchangeConjuncts {
            node_id: exchange_node_id,
        });
    }
    if exchange.sort_info.is_some() {
        return Err(FusionRefusal::SortedExchange {
            node_id: exchange_node_id,
        });
    }
    if let Some(parent) = preorder_parent(nodes, position)
        && nodes[parent].node_type == TPlanNodeType::AGGREGATION_NODE
    {
        return Err(FusionRefusal::AggregationParent {
            exchange: exchange_node_id,
            parent: nodes[parent].node_id,
        });
    }

    let sender_nodes = sender_nodes(sender)?;
    let sender_root = &sender_nodes[0];
    if exchange.input_row_tuples != sender_root.row_tuples {
        return Err(FusionRefusal::RowTuplesDiffer {
            exchange: exchange.input_row_tuples.clone(),
            sender_root: sender_root.row_tuples.clone(),
        });
    }
    // The descriptor table is per query, so the receiver's resolved copy describes the sender's
    // tuples too -- unless the CN handed over an unresolved cached reference.
    let desc = receiver
        .desc_tbl
        .as_ref()
        .filter(|desc| !desc.tuple_descriptors.is_empty())
        .ok_or(FusionRefusal::ReceiverDescriptorUnresolved)?;
    let described: BTreeSet<i32> = desc
        .tuple_descriptors
        .iter()
        .filter_map(|tuple| tuple.id)
        .collect();
    if let Some(tuple_id) = sender_nodes
        .iter()
        .flat_map(|node| node.row_tuples.iter().copied())
        .find(|tuple_id| !described.contains(tuple_id))
    {
        return Err(FusionRefusal::DescriptorMissingTuple { tuple_id });
    }

    let receiver_exec = receiver
        .params
        .as_ref()
        .ok_or(FusionRefusal::ReceiverMissingField("params"))?;
    let sender_exec = sender
        .params
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("params"))?;
    let receiver_scans: BTreeSet<i32> = scan_range_nodes(receiver_exec).collect();
    if let Some(node_id) =
        scan_range_nodes(sender_exec).find(|node_id| receiver_scans.contains(node_id))
    {
        return Err(FusionRefusal::ScanRangeCollision { node_id });
    }
    if let Some(node_id) = sender_exec
        .per_exch_num_senders
        .keys()
        .find(|node_id| receiver_exec.per_exch_num_senders.contains_key(node_id))
    {
        return Err(FusionRefusal::ExchangeIdCollision { node_id: *node_id });
    }
    Ok(())
}

/// The splice. Re-runs [`fusable_edge`] first, so it never applies half a fusion: a refusal
/// here after the same three arguments passed at defer time is a bug in the caller.
///
/// Returns the fused params with the RECEIVER's identity (query_id, fragment_instance_id,
/// output_sink, destinations, sender_id, desc_tbl) and:
/// - `plan.nodes`: the `EXCHANGE_NODE` replaced by `sender.fragment.plan.nodes` (a complete
///   preorder subtree, so every ancestor's `num_children` stays valid);
/// - `per_exch_num_senders`: minus `exchange_node_id`, plus the sender's own entries;
/// - `per_node_scan_ranges` / `node_to_per_driver_seq_scan_ranges`: the union by node id.
pub fn splice(
    mut receiver: TExecPlanFragmentParams,
    exchange_node_id: i32,
    sender: &TExecPlanFragmentParams,
) -> Refused<TExecPlanFragmentParams> {
    fusable_edge(&receiver, exchange_node_id, sender)?;
    let sender_nodes = sender_nodes(sender)?.to_vec();
    let sender_exec = sender
        .params
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("params"))?;

    let nodes = &mut receiver
        .fragment
        .as_mut()
        .ok_or(FusionRefusal::ReceiverMissingField("fragment"))?
        .plan
        .as_mut()
        .ok_or(FusionRefusal::ReceiverMissingField("plan"))?
        .nodes;
    let position = exchange_position(nodes, exchange_node_id)?;
    nodes.splice(position..=position, sender_nodes);

    let exec = receiver
        .params
        .as_mut()
        .ok_or(FusionRefusal::ReceiverMissingField("params"))?;
    exec.per_exch_num_senders.remove(&exchange_node_id);
    exec.per_exch_num_senders.extend(
        sender_exec
            .per_exch_num_senders
            .iter()
            .map(|(node_id, senders)| (*node_id, *senders)),
    );
    exec.per_node_scan_ranges.extend(
        sender_exec
            .per_node_scan_ranges
            .iter()
            .map(|(node_id, ranges)| (*node_id, ranges.clone())),
    );
    if let Some(per_driver) = sender_exec
        .node_to_per_driver_seq_scan_ranges
        .as_ref()
        .filter(|per_driver| !per_driver.is_empty())
    {
        exec.node_to_per_driver_seq_scan_ranges
            .get_or_insert_with(BTreeMap::new)
            .extend(
                per_driver
                    .iter()
                    .map(|(node_id, per_seq)| (*node_id, per_seq.clone())),
            );
    }
    Ok(receiver)
}

/// Index of the preorder parent of `nodes[idx]`, via the same ancestor stack
/// `common_slots_consumed_above` keeps (node_translator.rs). `None` for the root, or when `idx`
/// is out of range.
pub(crate) fn preorder_parent(nodes: &[TPlanNode], idx: usize) -> Option<usize> {
    // Open ancestors of the node being visited: (node index, children not yet completed).
    let mut stack: Vec<(usize, i32)> = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if index == idx {
            return stack.last().map(|&(parent, _)| parent);
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
    None
}

/// Number of nodes in the preorder subtree rooted at `nodes[start]`: the count
/// `PlanNodeCursor::translate_next` would consume from there. `Err` on an under-run or a
/// negative child count.
pub(crate) fn preorder_span(nodes: &[TPlanNode], start: usize) -> Refused<usize> {
    let mut idx = start;
    let mut pending = 1usize;
    while pending > 0 {
        let node = nodes.get(idx).ok_or_else(|| {
            FusionRefusal::MalformedSenderPlan(format!(
                "unexpected end of plan nodes after {} node(s) with {pending} child(ren) pending",
                idx - start
            ))
        })?;
        if node.num_children < 0 {
            return Err(FusionRefusal::MalformedSenderPlan(format!(
                "node {} has negative child count {}",
                node.node_id, node.num_children
            )));
        }
        pending += node.num_children as usize;
        pending -= 1;
        idx += 1;
    }
    Ok(idx - start)
}

/// The receiver's flat preorder node list.
fn receiver_nodes(receiver: &TExecPlanFragmentParams) -> Refused<&[TPlanNode]> {
    receiver
        .fragment
        .as_ref()
        .ok_or(FusionRefusal::ReceiverMissingField("fragment"))?
        .plan
        .as_ref()
        .map(|plan| plan.nodes.as_slice())
        .ok_or(FusionRefusal::ReceiverMissingField("plan"))
}

/// The sender's flat preorder node list, non-empty.
fn sender_nodes(sender: &TExecPlanFragmentParams) -> Refused<&[TPlanNode]> {
    let nodes = sender
        .fragment
        .as_ref()
        .ok_or(FusionRefusal::SenderMissingField("fragment"))?
        .plan
        .as_ref()
        .map(|plan| plan.nodes.as_slice())
        .ok_or(FusionRefusal::SenderMissingField("plan"))?;
    if nodes.is_empty() {
        return Err(FusionRefusal::MalformedSenderPlan(
            "TPlan.nodes is empty".to_string(),
        ));
    }
    Ok(nodes)
}

/// Position of the `EXCHANGE_NODE` with id `exchange_node_id` in `nodes`.
fn exchange_position(nodes: &[TPlanNode], exchange_node_id: i32) -> Refused<usize> {
    nodes
        .iter()
        .position(|node| {
            node.node_type == TPlanNodeType::EXCHANGE_NODE && node.node_id == exchange_node_id
        })
        .ok_or(FusionRefusal::ExchangeMissing {
            node_id: exchange_node_id,
        })
}

/// Every scan node id with ranges in either of the exec params' two scan-range maps.
fn scan_range_nodes(exec: &TPlanFragmentExecParams) -> impl Iterator<Item = i32> + '_ {
    exec.per_node_scan_ranges.keys().copied().chain(
        exec.node_to_per_driver_seq_scan_ranges
            .iter()
            .flat_map(|per_driver| per_driver.keys().copied()),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use starrocks_thrift::data_sinks::{
        TDataSink, TDataSinkType, TDataStreamSink, TPlanFragmentDestination,
    };
    use starrocks_thrift::descriptors::{TDescriptorTable, TTupleDescriptor};
    use starrocks_thrift::exprs::TExpr;
    use starrocks_thrift::internal_service::{
        InternalServiceVersion, TExecPlanFragmentParams, TPlanFragmentExecParams, TScanRangeParams,
    };
    use starrocks_thrift::partitions::{TDataPartition, TPartitionType};
    use starrocks_thrift::plan_nodes::{
        TAggregationNode, TExchangeNode, TFileScanNode, TPlan, TPlanNode, TPlanNodeType,
        TProjectNode, TScanRange, TSortInfo,
    };
    use starrocks_thrift::planner::TPlanFragment;
    use starrocks_thrift::types::{TNetworkAddress, TUniqueId};

    use super::*;

    /// Builds a StarRocks plan node with every node-specific payload cleared.
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
            row_tuples,
            nullable_tuples: Vec::new(),
            conjuncts: None,
            compact_data: false,
            common: None,
            hash_join_node: None,
            agg_node: None,
            sort_node: None,
            merge_node: None,
            exchange_node: None,
            mysql_scan_node: None,
            olap_scan_node: None,
            file_scan_node: None,
            schema_scan_node: None,
            meta_scan_node: None,
            analytic_node: None,
            union_node: None,
            resource_profile: None,
            es_scan_node: None,
            repeat_node: None,
            assert_num_rows_node: None,
            intersect_node: None,
            except_node: None,
            merge_join_node: None,
            raw_values_node: None,
            use_vectorized: None,
            hdfs_scan_node: None,
            project_node: None,
            table_function_node: None,
            probe_runtime_filters: None,
            decode_node: None,
            local_rf_waiting_set: None,
            filter_null_value_columns: None,
            need_create_tuple_columns: None,
            jdbc_scan_node: None,
            connector_scan_node: None,
            cross_join_node: None,
            lake_scan_node: None,
            nestloop_join_node: None,
            stream_scan_node: None,
            stream_join_node: None,
            stream_agg_node: None,
            select_node: None,
            fetch_node: None,
            look_up_node: None,
            cache_stats_scan_node: None,
        }
    }

    fn scan(node_id: i32, tuple_id: i32) -> TPlanNode {
        let mut scan = node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple_id]);
        scan.file_scan_node = Some(TFileScanNode::new(tuple_id, None, None, None));
        scan
    }

    /// A plain exchange leaf: no limit, no offset, no conjuncts, no sort.
    fn exchange(node_id: i32, input_row_tuples: Vec<i32>) -> TPlanNode {
        let mut exchange = node(
            node_id,
            TPlanNodeType::EXCHANGE_NODE,
            0,
            input_row_tuples.clone(),
        );
        exchange.exchange_node = Some(TExchangeNode::new(
            input_row_tuples,
            None,
            Some(0),
            Some(TPartitionType::HASH_PARTITIONED),
            None,
            None,
        ));
        exchange
    }

    fn join(node_id: i32, row_tuples: Vec<i32>) -> TPlanNode {
        node(node_id, TPlanNodeType::HASH_JOIN_NODE, 2, row_tuples)
    }

    /// A one-child project; `with_common` gives it a non-empty `common_slot_map`.
    fn project(node_id: i32, tuple_id: i32, with_common: bool) -> TPlanNode {
        let mut project = node(node_id, TPlanNodeType::PROJECT_NODE, 1, vec![tuple_id]);
        let common = with_common.then(|| BTreeMap::from([(9, TExpr::new(Vec::new()))]));
        project.project_node = Some(TProjectNode::new(Some(BTreeMap::new()), common));
        project
    }

    /// A one-child aggregation with no measures; `need_finalize == false` classifies as
    /// `AggPhase::Partial` (update serialize), `true` as `OneShot`.
    fn aggregation(node_id: i32, tuple_id: i32, need_finalize: bool) -> TPlanNode {
        let mut agg = node(node_id, TPlanNodeType::AGGREGATION_NODE, 1, vec![tuple_id]);
        agg.agg_node = Some(TAggregationNode::new(
            Some(Vec::new()),
            Vec::new(),
            tuple_id,
            tuple_id,
            need_finalize,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        agg
    }

    fn scan_range() -> Vec<TScanRangeParams> {
        vec![TScanRangeParams::new(
            TScanRange::new(None, None, None, None, None, None),
            None,
            None,
            None,
        )]
    }

    fn desc_tbl(tuple_ids: &[i32]) -> TDescriptorTable {
        TDescriptorTable::new(
            None,
            tuple_ids
                .iter()
                .map(|&id| TTupleDescriptor::new(Some(id), None, None, None, None))
                .collect(),
            None,
            None,
        )
    }

    fn exec(instance: i64) -> TPlanFragmentExecParams {
        TPlanFragmentExecParams::new(
            TUniqueId::new(1, 0),
            TUniqueId::new(1, instance),
            BTreeMap::new(),
            BTreeMap::new(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn stream_sink(dest_node_id: i32, partition: TPartitionType) -> TDataSink {
        TDataSink::new(
            TDataSinkType::DATA_STREAM_SINK,
            Some(TDataStreamSink::new(
                dest_node_id,
                TDataPartition::new(partition, None, None, None),
                None,
                None,
                None,
                None,
                None,
            )),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    fn destination(instance: i64) -> TPlanFragmentDestination {
        TPlanFragmentDestination::new(
            TUniqueId::new(1, instance),
            None,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 8060)),
            None,
        )
    }

    fn params(
        nodes: Vec<TPlanNode>,
        exec: TPlanFragmentExecParams,
        desc_tbl: Option<TDescriptorTable>,
    ) -> TExecPlanFragmentParams {
        TExecPlanFragmentParams {
            protocol_version: InternalServiceVersion::V1,
            fragment: Some(TPlanFragment {
                plan: Some(TPlan::new(nodes)),
                output_exprs: None,
                output_sink: None,
                partition: TDataPartition::new(TPartitionType::UNPARTITIONED, None, None, None),
                min_reservation_bytes: None,
                initial_reservation_total_claims: None,
                query_global_dicts: None,
                load_global_dicts: None,
                cache_param: None,
                query_global_dict_exprs: None,
                group_execution_param: None,
            }),
            desc_tbl,
            params: Some(exec),
            coord: None,
            backend_num: None,
            query_globals: None,
            query_options: None,
            enable_profile: None,
            resource_info: None,
            import_label: None,
            db_name: None,
            load_job_id: None,
            load_error_hub_info: None,
            is_pipeline: None,
            pipeline_dop: None,
            per_scan_node_dop: None,
            workgroup: None,
            enable_resource_group: None,
            func_version: None,
            enable_shared_scan: None,
            is_stream_pipeline: None,
            adaptive_dop_param: None,
            group_execution_scan_dop: None,
            pred_tree_params: None,
            exec_stats_node_ids: None,
            arrow_flight_sql_version: None,
        }
    }

    /// A registered receiver (instance 2) declaring `per_exch` and a resolved descriptor table
    /// over `tuples`, with its own result-bound sink so identity fields are visible.
    fn receiver_params(
        nodes: Vec<TPlanNode>,
        per_exch: &[(i32, i32)],
        tuples: &[i32],
    ) -> TExecPlanFragmentParams {
        let mut exec = exec(2);
        exec.per_exch_num_senders = per_exch.iter().copied().collect();
        exec.destinations = Some(vec![destination(9)]);
        exec.sender_id = Some(3);
        let mut receiver = params(nodes, exec, Some(desc_tbl(tuples)));
        receiver.fragment.as_mut().unwrap().output_sink =
            Some(stream_sink(99, TPartitionType::UNPARTITIONED));
        receiver
    }

    /// A sender (instance 3) with a single-destination `partition` stream sink to exchange 7 of
    /// instance 2.
    fn sender_params(nodes: Vec<TPlanNode>, partition: TPartitionType) -> TExecPlanFragmentParams {
        let mut exec = exec(3);
        exec.destinations = Some(vec![destination(2)]);
        exec.sender_id = Some(0);
        let mut sender = params(nodes, exec, None);
        sender.fragment.as_mut().unwrap().output_sink = Some(stream_sink(7, partition));
        sender
    }

    fn stream_sink_mut(params: &mut TExecPlanFragmentParams) -> &mut TDataStreamSink {
        params
            .fragment
            .as_mut()
            .unwrap()
            .output_sink
            .as_mut()
            .unwrap()
            .stream_sink
            .as_mut()
            .unwrap()
    }

    fn exec_mut(params: &mut TExecPlanFragmentParams) -> &mut TPlanFragmentExecParams {
        params.params.as_mut().unwrap()
    }

    fn node_ids(params: &TExecPlanFragmentParams) -> Vec<i32> {
        params
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes
            .iter()
            .map(|node| node.node_id)
            .collect()
    }

    /// The measured q05 shape: a hash-partitioned leaf scan under a join's exchange leaf.
    fn join_receiver_and_leaf() -> (TExecPlanFragmentParams, TExecPlanFragmentParams) {
        let mut receiver = receiver_params(
            vec![join(2, vec![0, 1]), exchange(7, vec![0]), scan(1, 1)],
            &[(7, 1)],
            &[0, 1],
        );
        exec_mut(&mut receiver)
            .per_node_scan_ranges
            .insert(1, scan_range());
        let mut leaf = sender_params(
            vec![project(3, 0, false), scan(0, 0)],
            TPartitionType::HASH_PARTITIONED,
        );
        exec_mut(&mut leaf)
            .per_node_scan_ranges
            .insert(0, scan_range());
        (receiver, leaf)
    }

    #[test]
    fn splice_replaces_the_exchange_leaf_with_the_sender_preorder() {
        let (receiver, leaf) = join_receiver_and_leaf();
        let shape = sender_shape(&leaf).unwrap();
        assert!(shape.is_leaf);
        fusable_edge(&receiver, 7, &leaf).unwrap();

        let fused = splice(receiver, 7, &leaf).unwrap();

        assert_eq!(node_ids(&fused), vec![2, 3, 0, 1]);
        let exec = fused.params.as_ref().unwrap();
        assert!(exec.per_exch_num_senders.is_empty());
        assert_eq!(
            exec.per_node_scan_ranges
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        // The receiver keeps its identity: ids, sink, destinations, sender id, descriptors.
        assert_eq!(exec.query_id, TUniqueId::new(1, 0));
        assert_eq!(exec.fragment_instance_id, TUniqueId::new(1, 2));
        assert_eq!(exec.sender_id, Some(3));
        assert_eq!(
            exec.destinations.as_ref().unwrap()[0].fragment_instance_id,
            TUniqueId::new(1, 9)
        );
        let sink = fused
            .fragment
            .as_ref()
            .unwrap()
            .output_sink
            .as_ref()
            .unwrap();
        assert_eq!(sink.stream_sink.as_ref().unwrap().dest_node_id, 99);
        assert_eq!(fused.desc_tbl.as_ref().unwrap().tuple_descriptors.len(), 2);
        // Every ancestor's child count still describes the spliced preorder.
        let nodes = &fused
            .fragment
            .as_ref()
            .unwrap()
            .plan
            .as_ref()
            .unwrap()
            .nodes;
        assert_eq!(preorder_span(nodes, 0).unwrap(), nodes.len());
        assert_eq!(preorder_parent(nodes, 1), Some(0));
        assert_eq!(preorder_parent(nodes, 2), Some(1));
        assert_eq!(preorder_parent(nodes, 3), Some(0));
    }

    #[test]
    fn splice_merges_per_driver_scan_ranges_and_sender_exchanges() {
        let receiver = receiver_params(vec![exchange(7, vec![0])], &[(7, 1)], &[0]);
        assert!(
            receiver
                .params
                .as_ref()
                .unwrap()
                .node_to_per_driver_seq_scan_ranges
                .is_none()
        );
        let mut leaf = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        exec_mut(&mut leaf).node_to_per_driver_seq_scan_ranges =
            Some(BTreeMap::from([(0, BTreeMap::from([(0, scan_range())]))]));
        // A middle (non-leaf) sender brings its own exchange declarations along; a leaf has none.
        exec_mut(&mut leaf).per_exch_num_senders.insert(5, 1);

        let fused = splice(receiver, 7, &leaf).unwrap();

        let exec = fused.params.as_ref().unwrap();
        let per_driver = exec.node_to_per_driver_seq_scan_ranges.as_ref().unwrap();
        assert_eq!(per_driver.keys().copied().collect::<Vec<_>>(), vec![0]);
        assert_eq!(per_driver[&0][&0].len(), 1);
        assert_eq!(exec.per_exch_num_senders, BTreeMap::from([(5, 1)]));
        assert!(exec.per_node_scan_ranges.is_empty());
    }

    #[test]
    fn fusable_edge_refuses_scan_range_collision() {
        // Receiver holds node 0 in per_node_scan_ranges.
        let (mut receiver, leaf) = join_receiver_and_leaf();
        exec_mut(&mut receiver)
            .per_node_scan_ranges
            .insert(0, scan_range());
        assert_eq!(
            fusable_edge(&receiver, 7, &leaf),
            Err(FusionRefusal::ScanRangeCollision { node_id: 0 })
        );

        // Receiver holds node 0 in the per-driver map instead.
        let (mut receiver, leaf) = join_receiver_and_leaf();
        exec_mut(&mut receiver).node_to_per_driver_seq_scan_ranges =
            Some(BTreeMap::from([(0, BTreeMap::from([(0, scan_range())]))]));
        assert_eq!(
            fusable_edge(&receiver, 7, &leaf),
            Err(FusionRefusal::ScanRangeCollision { node_id: 0 })
        );

        // The sender's per-driver map collides with the receiver's per-node map.
        let (receiver, mut leaf) = join_receiver_and_leaf();
        exec_mut(&mut leaf).per_node_scan_ranges.clear();
        exec_mut(&mut leaf).node_to_per_driver_seq_scan_ranges =
            Some(BTreeMap::from([(1, BTreeMap::from([(0, scan_range())]))]));
        assert_eq!(
            fusable_edge(&receiver, 7, &leaf),
            Err(FusionRefusal::ScanRangeCollision { node_id: 1 })
        );
    }

    #[test]
    fn fusable_edge_refuses_exchange_id_collision() {
        let (mut receiver, mut leaf) = join_receiver_and_leaf();
        exec_mut(&mut receiver).per_exch_num_senders.insert(5, 1);
        exec_mut(&mut leaf).per_exch_num_senders.insert(5, 1);
        assert_eq!(
            fusable_edge(&receiver, 7, &leaf),
            Err(FusionRefusal::ExchangeIdCollision { node_id: 5 })
        );
        // Splicing an edge that fails the checks is refused the same way, never applied.
        assert_eq!(
            splice(receiver, 7, &leaf).unwrap_err(),
            FusionRefusal::ExchangeIdCollision { node_id: 5 }
        );
    }

    #[test]
    fn fusable_edge_refuses_aggregation_parent() {
        // q01's F02 -> merge aggregation shape.
        let receiver = receiver_params(
            vec![aggregation(4, 1, true), exchange(3, vec![0])],
            &[(3, 1)],
            &[0, 1],
        );
        let leaf = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        assert_eq!(
            fusable_edge(&receiver, 3, &leaf),
            Err(FusionRefusal::AggregationParent {
                exchange: 3,
                parent: 4
            })
        );
    }

    #[test]
    fn fusable_edge_refuses_missing_or_malformed_exchange() {
        let (receiver, leaf) = join_receiver_and_leaf();
        assert_eq!(
            fusable_edge(&receiver, 8, &leaf),
            Err(FusionRefusal::ExchangeMissing { node_id: 8 })
        );
        // A scan is not an exchange even when the id matches.
        assert_eq!(
            fusable_edge(&receiver, 1, &leaf),
            Err(FusionRefusal::ExchangeMissing { node_id: 1 })
        );

        let mut with_children =
            receiver_params(vec![exchange(7, vec![0]), scan(0, 0)], &[(7, 1)], &[0]);
        with_children
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .num_children = 1;
        assert_eq!(
            fusable_edge(&with_children, 7, &leaf),
            Err(FusionRefusal::ExchangeHasChildren { node_id: 7 })
        );

        let mut without_payload = receiver_params(vec![exchange(7, vec![0])], &[(7, 1)], &[0]);
        without_payload
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .exchange_node = None;
        assert_eq!(
            fusable_edge(&without_payload, 7, &leaf),
            Err(FusionRefusal::ReceiverMissingField("exchange_node"))
        );

        let mut without_plan = receiver_params(vec![exchange(7, vec![0])], &[(7, 1)], &[0]);
        without_plan.fragment.as_mut().unwrap().plan = None;
        assert_eq!(
            fusable_edge(&without_plan, 7, &leaf),
            Err(FusionRefusal::ReceiverMissingField("plan"))
        );

        let mut without_exec = receiver_params(vec![exchange(7, vec![0])], &[(7, 1)], &[0]);
        without_exec.params = None;
        assert_eq!(
            fusable_edge(&without_exec, 7, &leaf),
            Err(FusionRefusal::ReceiverMissingField("params"))
        );
    }

    #[test]
    fn fusable_edge_refuses_sorted_limited_offset_or_filtered_exchange() {
        let leaf = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        let decorated = |decorate: fn(&mut TPlanNode)| {
            let mut exchange = exchange(7, vec![0]);
            decorate(&mut exchange);
            let receiver = receiver_params(vec![exchange], &[(7, 1)], &[0]);
            fusable_edge(&receiver, 7, &leaf)
        };

        assert_eq!(
            decorated(|node| {
                node.exchange_node.as_mut().unwrap().sort_info =
                    Some(TSortInfo::new(Vec::new(), Vec::new(), Vec::new(), None));
            }),
            Err(FusionRefusal::SortedExchange { node_id: 7 })
        );
        assert_eq!(
            decorated(|node| node.limit = 10),
            Err(FusionRefusal::ExchangeLimit { node_id: 7 })
        );
        assert_eq!(
            decorated(|node| node.exchange_node.as_mut().unwrap().offset = Some(5)),
            Err(FusionRefusal::ExchangeOffset { node_id: 7 })
        );
        assert_eq!(
            decorated(|node| node.conjuncts = Some(vec![TExpr::new(Vec::new())])),
            Err(FusionRefusal::ExchangeConjuncts { node_id: 7 })
        );
        // An empty conjunct list and an unset offset are the plain shape the FE ships.
        assert_eq!(
            decorated(|node| {
                node.conjuncts = Some(Vec::new());
                node.exchange_node.as_mut().unwrap().offset = None;
            }),
            Ok(())
        );
    }

    #[test]
    fn fusable_edge_refuses_row_tuples_mismatch() {
        let receiver = receiver_params(vec![exchange(7, vec![6])], &[(7, 1)], &[3, 6]);
        let leaf = sender_params(vec![scan(0, 3)], TPartitionType::HASH_PARTITIONED);
        assert_eq!(
            fusable_edge(&receiver, 7, &leaf),
            Err(FusionRefusal::RowTuplesDiffer {
                exchange: vec![6],
                sender_root: vec![3]
            })
        );
    }

    #[test]
    fn fusable_edge_refuses_missing_descriptor_tuple() {
        let leaf = sender_params(
            vec![project(3, 6, false), scan(0, 4)],
            TPartitionType::HASH_PARTITIONED,
        );
        // The root's tuple is described but a lower node's is not.
        let receiver = receiver_params(vec![exchange(7, vec![6])], &[(7, 1)], &[6]);
        assert_eq!(
            fusable_edge(&receiver, 7, &leaf),
            Err(FusionRefusal::DescriptorMissingTuple { tuple_id: 4 })
        );

        // A cached descriptor reference (as the FE ships for later instances) is unresolved.
        let mut cached = receiver_params(vec![exchange(7, vec![6])], &[(7, 1)], &[]);
        cached.desc_tbl.as_mut().unwrap().is_cached = Some(true);
        assert_eq!(
            fusable_edge(&cached, 7, &leaf),
            Err(FusionRefusal::ReceiverDescriptorUnresolved)
        );
        let mut absent = receiver_params(vec![exchange(7, vec![6])], &[(7, 1)], &[6, 4]);
        absent.desc_tbl = None;
        assert_eq!(
            fusable_edge(&absent, 7, &leaf),
            Err(FusionRefusal::ReceiverDescriptorUnresolved)
        );
    }

    #[test]
    fn sender_shape_refuses_output_exprs_sink_limit_output_columns_and_fan_out() {
        let mut with_output_exprs =
            sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        with_output_exprs.fragment.as_mut().unwrap().output_exprs =
            Some(vec![TExpr::new(Vec::new())]);
        assert_eq!(
            sender_shape(&with_output_exprs).unwrap_err(),
            FusionRefusal::SenderOutputExprs
        );

        let mut limited = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        stream_sink_mut(&mut limited).limit = Some(0);
        assert_eq!(
            sender_shape(&limited).unwrap_err(),
            FusionRefusal::SinkLimit
        );
        // A negative limit is "no limit" on the wire, as the CN reads it.
        stream_sink_mut(&mut limited).limit = Some(-1);
        assert!(sender_shape(&limited).is_ok());

        let mut reordered = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        stream_sink_mut(&mut reordered).output_columns = Some(vec![1, 0]);
        assert_eq!(
            sender_shape(&reordered).unwrap_err(),
            FusionRefusal::SinkOutputColumns
        );

        let mut fan_out = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        exec_mut(&mut fan_out).destinations = Some(vec![destination(2), destination(4)]);
        assert_eq!(
            sender_shape(&fan_out).unwrap_err(),
            FusionRefusal::NotSingleDestination { destinations: 2 }
        );
        exec_mut(&mut fan_out).destinations = None;
        assert_eq!(
            sender_shape(&fan_out).unwrap_err(),
            FusionRefusal::NotSingleDestination { destinations: 0 }
        );
    }

    #[test]
    fn sender_shape_refuses_non_stream_sinks_and_missing_fields() {
        let mut result_sink = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        result_sink
            .fragment
            .as_mut()
            .unwrap()
            .output_sink
            .as_mut()
            .unwrap()
            .type_ = TDataSinkType::RESULT_SINK;
        assert_eq!(
            sender_shape(&result_sink).unwrap_err(),
            FusionRefusal::NotDataStreamSink
        );

        let mut no_payload = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        no_payload
            .fragment
            .as_mut()
            .unwrap()
            .output_sink
            .as_mut()
            .unwrap()
            .stream_sink = None;
        assert_eq!(
            sender_shape(&no_payload).unwrap_err(),
            FusionRefusal::SenderMissingField("stream_sink")
        );

        let mut no_sink = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        no_sink.fragment.as_mut().unwrap().output_sink = None;
        assert_eq!(
            sender_shape(&no_sink).unwrap_err(),
            FusionRefusal::SenderMissingField("output_sink")
        );

        let mut no_exec = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        no_exec.params = None;
        assert_eq!(
            sender_shape(&no_exec).unwrap_err(),
            FusionRefusal::SenderMissingField("params")
        );

        let mut no_plan = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        no_plan.fragment.as_mut().unwrap().plan = None;
        assert_eq!(
            sender_shape(&no_plan).unwrap_err(),
            FusionRefusal::SenderMissingField("plan")
        );

        let mut no_fragment = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        no_fragment.fragment = None;
        assert_eq!(
            sender_shape(&no_fragment).unwrap_err(),
            FusionRefusal::SenderMissingField("fragment")
        );
    }

    #[test]
    fn sender_shape_refuses_common_slot_projection_and_partial_aggregation_root() {
        let carrying = sender_params(
            vec![project(3, 0, true), scan(0, 0)],
            TPartitionType::HASH_PARTITIONED,
        );
        assert_eq!(
            sender_shape(&carrying).unwrap_err(),
            FusionRefusal::CommonSlotProjection { node_id: 3 }
        );

        let partial = sender_params(
            vec![aggregation(5, 1, false), scan(0, 0)],
            TPartitionType::HASH_PARTITIONED,
        );
        assert_eq!(
            sender_shape(&partial).unwrap_err(),
            FusionRefusal::PartialAggregationRoot { node_id: 5 }
        );

        // A finalized one-phase aggregation root crosses the boundary as plain rows.
        let one_shot = sender_params(
            vec![aggregation(5, 1, true), scan(0, 0)],
            TPartitionType::HASH_PARTITIONED,
        );
        assert!(sender_shape(&one_shot).is_ok());
    }

    #[test]
    fn sender_shape_refuses_malformed_preorder() {
        let underrun = sender_params(vec![project(3, 0, false)], TPartitionType::HASH_PARTITIONED);
        assert!(matches!(
            sender_shape(&underrun).unwrap_err(),
            FusionRefusal::MalformedSenderPlan(_)
        ));

        let trailing = sender_params(
            vec![scan(0, 0), scan(1, 0)],
            TPartitionType::HASH_PARTITIONED,
        );
        assert!(matches!(
            sender_shape(&trailing).unwrap_err(),
            FusionRefusal::MalformedSenderPlan(_)
        ));

        let empty = sender_params(Vec::new(), TPartitionType::HASH_PARTITIONED);
        assert!(matches!(
            sender_shape(&empty).unwrap_err(),
            FusionRefusal::MalformedSenderPlan(_)
        ));

        let mut negative = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        negative
            .fragment
            .as_mut()
            .unwrap()
            .plan
            .as_mut()
            .unwrap()
            .nodes[0]
            .num_children = -1;
        assert!(matches!(
            sender_shape(&negative).unwrap_err(),
            FusionRefusal::MalformedSenderPlan(_)
        ));
    }

    #[test]
    fn sender_shape_reports_leaf_and_partition_type() {
        let mut leaf = sender_params(vec![scan(0, 0)], TPartitionType::HASH_PARTITIONED);
        exec_mut(&mut leaf).sender_id = Some(4);
        let shape = sender_shape(&leaf).unwrap();
        assert!(shape.is_leaf);
        assert_eq!(shape.partition, TPartitionType::HASH_PARTITIONED);
        assert_eq!(shape.sender_id, 4);
        assert_eq!(shape.dest_node_id, 7);
        assert_eq!(shape.destination.fragment_instance_id, TUniqueId::new(1, 2));

        // An unset sender_id reads as 0, the way the CN keys its sender slots.
        exec_mut(&mut leaf).sender_id = None;
        assert_eq!(sender_shape(&leaf).unwrap().sender_id, 0);

        let broadcast = sender_params(vec![scan(0, 0)], TPartitionType::UNPARTITIONED);
        assert_eq!(
            sender_shape(&broadcast).unwrap().partition,
            TPartitionType::UNPARTITIONED
        );

        let middle = sender_params(
            vec![join(2, vec![0, 1]), exchange(5, vec![0]), scan(1, 1)],
            TPartitionType::HASH_PARTITIONED,
        );
        let shape = sender_shape(&middle).unwrap();
        assert!(!shape.is_leaf);
    }

    #[test]
    fn preorder_parent_matches_the_ancestor_stack() {
        // A(2) -> [B(2) -> [C, D], E(1) -> [F(1) -> [G]]]
        let nodes = vec![
            node(0, TPlanNodeType::HASH_JOIN_NODE, 2, vec![0]),
            node(1, TPlanNodeType::HASH_JOIN_NODE, 2, vec![0]),
            node(2, TPlanNodeType::FILE_SCAN_NODE, 0, vec![0]),
            node(3, TPlanNodeType::FILE_SCAN_NODE, 0, vec![0]),
            node(4, TPlanNodeType::PROJECT_NODE, 1, vec![0]),
            node(5, TPlanNodeType::SELECT_NODE, 1, vec![0]),
            node(6, TPlanNodeType::FILE_SCAN_NODE, 0, vec![0]),
        ];
        let parents: Vec<Option<usize>> = (0..nodes.len())
            .map(|idx| preorder_parent(&nodes, idx))
            .collect();
        assert_eq!(
            parents,
            vec![None, Some(0), Some(1), Some(1), Some(0), Some(4), Some(5)]
        );
        assert_eq!(preorder_parent(&nodes, nodes.len()), None);

        assert_eq!(preorder_span(&nodes, 0), Ok(7));
        assert_eq!(preorder_span(&nodes, 1), Ok(3));
        assert_eq!(preorder_span(&nodes, 4), Ok(3));
        assert_eq!(preorder_span(&nodes, 6), Ok(1));
        assert!(matches!(
            preorder_span(&nodes[..5], 4),
            Err(FusionRefusal::MalformedSenderPlan(_))
        ));
    }

    #[test]
    fn refusals_display_on_one_line() {
        let refusals = [
            FusionRefusal::SenderMissingField("plan"),
            FusionRefusal::NotDataStreamSink,
            FusionRefusal::NotSingleDestination { destinations: 4 },
            FusionRefusal::SinkLimit,
            FusionRefusal::SinkOutputColumns,
            FusionRefusal::SenderOutputExprs,
            FusionRefusal::MalformedSenderPlan("TPlan.nodes is empty".to_string()),
            FusionRefusal::CommonSlotProjection { node_id: 3 },
            FusionRefusal::PartialAggregationRoot { node_id: 5 },
            FusionRefusal::ReceiverMissingField("params"),
            FusionRefusal::ExchangeMissing { node_id: 8 },
            FusionRefusal::ExchangeHasChildren { node_id: 7 },
            FusionRefusal::ExchangeLimit { node_id: 7 },
            FusionRefusal::ExchangeOffset { node_id: 7 },
            FusionRefusal::ExchangeConjuncts { node_id: 7 },
            FusionRefusal::SortedExchange { node_id: 7 },
            FusionRefusal::AggregationParent {
                exchange: 9,
                parent: 10,
            },
            FusionRefusal::RowTuplesDiffer {
                exchange: vec![6],
                sender_root: vec![3],
            },
            FusionRefusal::ReceiverDescriptorUnresolved,
            FusionRefusal::DescriptorMissingTuple { tuple_id: 6 },
            FusionRefusal::ScanRangeCollision { node_id: 0 },
            FusionRefusal::ExchangeIdCollision { node_id: 5 },
        ];
        for refusal in refusals {
            let text = refusal.to_string();
            assert!(
                !text.is_empty() && !text.contains('\n'),
                "{refusal:?}: {text:?}"
            );
        }
        assert_eq!(
            FusionRefusal::AggregationParent {
                exchange: 9,
                parent: 10
            }
            .to_string(),
            "exchange 9 feeds aggregation node 10"
        );
    }
}
