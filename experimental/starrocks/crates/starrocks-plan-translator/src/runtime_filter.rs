//! The FE's runtime filters, as far as one compute node can apply them.
//!
//! The FE plans a filter at a hash join and lists the scans that probe it, in other fragments.
//! A compute node can build the filter itself when the join broadcasts its build side: every
//! node then receives the whole build input through one exchange. [`built_filters`] finds those
//! filters in a receiver fragment, with the exchange and the key column to read;
//! [`probed_filters`] finds the scans of a fragment that probe one. Binding a key stream to a
//! probed filter ([`FilterInput`]) makes the translator join the scan's output against it.

use starrocks_thrift::exprs::TExprNodeType;
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::plan_nodes::{TPlanNode, TPlanNodeType};
use starrocks_thrift::runtime_filter::{
    TRuntimeFilterBuildJoinMode, TRuntimeFilterBuildType, TRuntimeFilterDescription,
};

use crate::descriptor_table::{DescriptorTable, SlotKey};
use crate::error::{Result, TranslateError};
use crate::row_layout::RowLayout;
use crate::{StreamInputColumn, type_mapper};

/// A runtime filter that a scan of this fragment probes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbedFilter {
    /// Filter id, unique within the query.
    pub filter_id: i32,
    /// The probing scan node.
    pub scan_node_id: i32,
}

/// A runtime filter this fragment builds at a broadcast hash join whose build side is one
/// exchange, so every instance receives the whole key set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltFilter {
    /// Filter id, unique within the query.
    pub filter_id: i32,
    /// The join that builds it.
    pub join_node_id: i32,
    /// The exchange feeding the join's build side.
    pub exchange_node_id: i32,
    /// Position of the key in the exchange's rows.
    pub column: usize,
    /// DuckDB type name of the key.
    pub column_type: String,
}

/// A key stream bound to a probed filter: the scans probing `filter_id` keep only rows whose key
/// appears in it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterInput {
    /// Filter id, unique within the query.
    pub filter_id: i32,
    /// Engine stream id of the key stream. Must not collide with an exchange node id.
    pub node_id: i32,
    /// Engine view the key stream is read through.
    pub stream_view: String,
    /// The key column; its type must match the probe expression's.
    pub column: StreamInputColumn,
}

/// Join filters the scans of `params`' fragment probe, in plan order.
pub fn probed_filters(params: &TExecPlanFragmentParams) -> Vec<ProbedFilter> {
    plan_nodes(params)
        .iter()
        .filter(|node| is_scan(node))
        .flat_map(|node| {
            probe_filters_of(node).filter_map(move |filter| {
                Some(ProbedFilter {
                    filter_id: filter.filter_id?,
                    scan_node_id: node.node_id,
                })
            })
        })
        .collect()
}

/// Join filters `params`' fragment builds at a broadcast hash join whose build child is an
/// exchange and whose key is a bare column of it. Filters of any other shape are left out.
pub fn built_filters(params: &TExecPlanFragmentParams) -> Result<Vec<BuiltFilter>> {
    let nodes = plan_nodes(params);
    if nodes.is_empty() {
        return Ok(Vec::new());
    }
    let desc_tbl = params
        .desc_tbl
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TExecPlanFragmentParams",
            field: "desc_tbl",
        })?;
    let desc = DescriptorTable::try_from(desc_tbl)?;
    let children = child_indices(nodes)?;
    let mut built = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let Some(filters) = node
            .hash_join_node
            .as_ref()
            .and_then(|join| join.build_runtime_filters.as_ref())
        else {
            continue;
        };
        let Some(exchange) = children[index]
            .get(1)
            .map(|&child| &nodes[child])
            .filter(|child| child.node_type == TPlanNodeType::EXCHANGE_NODE)
        else {
            continue;
        };
        let Some(input_row_tuples) = exchange
            .exchange_node
            .as_ref()
            .map(|exchange| exchange.input_row_tuples.as_slice())
        else {
            continue;
        };
        for filter in filters {
            let Some((filter_id, tuple_id, slot_id)) = broadcast_key(filter) else {
                continue;
            };
            let column = RowLayout::from_tuples(&desc, input_row_tuples)?
                .resolve(SlotKey::new(tuple_id, slot_id))?;
            let schema = desc.named_struct_for_tuples(input_row_tuples)?;
            let ty = schema
                .r#struct
                .as_ref()
                .and_then(|structure| structure.types.get(column))
                .ok_or_else(|| {
                    TranslateError::descriptor(format!(
                        "runtime filter {filter_id} key column {column} is outside exchange {}",
                        exchange.node_id
                    ))
                })?;
            built.push(BuiltFilter {
                filter_id,
                join_node_id: node.node_id,
                exchange_node_id: exchange.node_id,
                column,
                column_type: type_mapper::duckdb_type_name(ty)?,
            });
        }
    }
    Ok(built)
}

/// The probe filters a scan node carries.
pub(crate) fn probe_filters_of(
    node: &TPlanNode,
) -> impl Iterator<Item = &TRuntimeFilterDescription> {
    node.probe_runtime_filters
        .iter()
        .flatten()
        .filter(|filter| is_join_filter(filter))
}

fn is_scan(node: &TPlanNode) -> bool {
    matches!(
        node.node_type,
        TPlanNodeType::FILE_SCAN_NODE | TPlanNodeType::HDFS_SCAN_NODE
    )
}

fn is_join_filter(filter: &TRuntimeFilterDescription) -> bool {
    filter
        .filter_type
        .is_none_or(|kind| kind == TRuntimeFilterBuildType::JOIN_FILTER)
}

/// `(filter id, tuple id, slot id)` of a broadcast join filter keyed on a bare slot.
fn broadcast_key(filter: &TRuntimeFilterDescription) -> Option<(i32, i32, i32)> {
    if !is_join_filter(filter)
        || filter.build_join_mode != Some(TRuntimeFilterBuildJoinMode::BROADCAST)
    {
        return None;
    }
    let [node] = filter.build_expr.as_ref()?.nodes.as_slice() else {
        return None;
    };
    if node.node_type != TExprNodeType::SLOT_REF {
        return None;
    }
    let slot = node.slot_ref.as_ref()?;
    Some((filter.filter_id?, slot.tuple_id, slot.slot_id))
}

fn plan_nodes(params: &TExecPlanFragmentParams) -> &[TPlanNode] {
    params
        .fragment
        .as_ref()
        .and_then(|fragment| fragment.plan.as_ref())
        .map(|plan| plan.nodes.as_slice())
        .unwrap_or_default()
}

/// The child indices of every node of a flat preorder plan.
fn child_indices(nodes: &[TPlanNode]) -> Result<Vec<Vec<usize>>> {
    fn walk(nodes: &[TPlanNode], at: usize, children: &mut [Vec<usize>]) -> Result<usize> {
        let node = nodes
            .get(at)
            .ok_or_else(|| TranslateError::malformed("unexpected end of plan nodes"))?;
        let count = usize::try_from(node.num_children).map_err(|_| {
            TranslateError::malformed(format!(
                "node {} has negative child count {}",
                node.node_id, node.num_children
            ))
        })?;
        let mut next = at + 1;
        for _ in 0..count {
            children[at].push(next);
            next = walk(nodes, next, children)?;
        }
        Ok(next)
    }
    let mut children = vec![Vec::new(); nodes.len()];
    walk(nodes, 0, &mut children)?;
    Ok(children)
}
