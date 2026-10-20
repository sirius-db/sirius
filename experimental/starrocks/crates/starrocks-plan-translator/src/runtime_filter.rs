//! The FE's runtime filters, as far as one compute node can apply them.
//!
//! The FE plans a filter at a hash join and lists the scans that probe it, in other fragments.
//! A compute node can build the filter itself when the join broadcasts its build side: every
//! node then receives the whole build input through one exchange. A partitioned join's instances
//! each receive a share of the keys instead, and the probing scans need the union of every
//! share. [`built_filters`] finds both kinds in a receiver fragment, with the exchange and the
//! key column to read; [`probed_filters`] finds the scans of a fragment that probe one. Binding
//! keys to a probed filter ([`FilterInput`]) makes the translator join the scan's output against
//! a key stream, or keep only the keys within bounds.

use starrocks_thrift::exprs::TExprNodeType;
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::plan_nodes::{THashJoinNode, TPlanNode, TPlanNodeType};
use starrocks_thrift::runtime_filter::{
    TRuntimeFilterBuildJoinMode, TRuntimeFilterBuildType, TRuntimeFilterDescription,
};
use starrocks_thrift::types::TNetworkAddress;

use crate::descriptor_table::{DescriptorTable, SlotKey};
use crate::error::{Result, TranslateError};
use crate::row_layout::RowLayout;
use crate::{StreamInputColumn, type_mapper};

/// How the join that builds a filter spreads its build side over its instances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildDistribution {
    /// Every instance receives the whole key set.
    Broadcast,
    /// A partitioned or bucket-shuffle join: each instance receives a share of the keys, and the
    /// shares together are the whole key set.
    Partitioned,
    /// Any other join (colocate, local bucket): not applied.
    Other,
}

impl BuildDistribution {
    /// How the join building `filter` spreads its keys.
    pub fn of(filter: &TRuntimeFilterDescription) -> Self {
        match filter.build_join_mode {
            Some(TRuntimeFilterBuildJoinMode::BROADCAST) => Self::Broadcast,
            Some(
                TRuntimeFilterBuildJoinMode::PARTITIONED
                | TRuntimeFilterBuildJoinMode::SHUFFLE_HASH_BUCKET,
            ) => Self::Partitioned,
            _ => Self::Other,
        }
    }
}

/// A runtime filter that a scan of this fragment probes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbedFilter {
    /// Filter id, unique within the query.
    pub filter_id: i32,
    /// The probing scan node.
    pub scan_node_id: i32,
    /// How the building join spreads its keys. Only a broadcast join's filter can be built on
    /// the scan's own CN; a partitioned join's needs every instance's share.
    pub distribution: BuildDistribution,
    /// The FE's merge node for the filter (`runtime_filter_merge_nodes`): the CN that runs the
    /// query's root fragment and holds the filter's share count and probers.
    pub merge_node: Option<TNetworkAddress>,
    /// How many build instances hold a share (`layout.num_instances`), when the plan says.
    pub shares: Option<usize>,
    /// DuckDB type name of the scan's probe expression, when it has one.
    pub key_type: Option<String>,
}

/// Why a compute node leaves a runtime filter the FE planned unapplied, as far as the plan
/// shows. Run-time reasons (dense keys, a timeout) are the compute node's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// Neither a broadcast nor a partitioned join (colocate, local bucket).
    NotBroadcast,
    /// A partitioned join's filter names no merge node, so its probers can't be found.
    NoMergeNode,
    /// The join's build side isn't an exchange.
    BuildNotExchange,
    /// The build key is an expression, not a bare column.
    KeyNotSlot,
    /// The join compares the key with `<=>`, which the `=` semi-join can't express.
    NullSafe,
    /// The broadcast branch of a skew join, which holds only the skewed keys.
    Skew,
    /// The probing scan is in the join's own fragment.
    SameFragment,
    /// The probing scan is in another fragment that reads exchanges, and the filter can't be
    /// waited for there (a partitioned join's).
    NonLeafFragment,
    /// The target is a join, exchange or aggregation node; only scans apply filters.
    TargetNotScan,
}

impl SkipReason {
    /// The reason as logged.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotBroadcast => "not_broadcast",
            Self::NoMergeNode => "no_merge_node",
            Self::BuildNotExchange => "build_not_exchange",
            Self::KeyNotSlot => "key_not_slot",
            Self::NullSafe => "null_safe",
            Self::Skew => "skew",
            Self::SameFragment => "same_fragment",
            Self::NonLeafFragment => "non_leaf_fragment",
            Self::TargetNotScan => "target_not_scan",
        }
    }
}

/// A runtime filter left unapplied, at `node_id`: the building join, or the probing target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkippedFilter {
    /// Filter id, unique within the query.
    pub filter_id: i32,
    /// The building join, or the probing target.
    pub node_id: i32,
    pub reason: SkipReason,
}

/// A runtime filter this fragment builds at a hash join whose build side is one exchange. A
/// broadcast join's instance receives the whole key set there; a partitioned join's instance
/// receives its share.
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
    /// [`BuildDistribution::Broadcast`] or [`BuildDistribution::Partitioned`].
    pub distribution: BuildDistribution,
    /// For a partitioned join: the FE's merge node, which knows the filter's probers.
    pub merge_node: Option<TNetworkAddress>,
}

/// Column name of a share holding its exact keys.
pub const SHARE_KEYS: &str = "rf_key";
/// Column name of a share holding only its smallest and largest key.
pub const SHARE_BOUNDS: &str = "rf_bound";

/// Keys bound to a probed filter: the scans probing `filter_id` keep only the rows whose key
/// they admit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterInput {
    /// Filter id, unique within the query.
    pub filter_id: i32,
    pub keys: ProbeKeys,
}

/// What a probing scan keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeKeys {
    /// The rows whose key appears in a key stream (a left semi join).
    Stream {
        /// Engine stream id of the key stream. Must not collide with an exchange node id.
        node_id: i32,
        /// Engine view the key stream is read through.
        stream_view: String,
        /// The key column; its type must match the probe expression's.
        column: StreamInputColumn,
    },
    /// The rows whose key is within `[min, max]`: a partitioned join's filter whose shares were
    /// too large to send exactly.
    Range { min: i64, max: i64 },
}

/// Join filters the scans of `params`' fragment probe, in plan order, except those a join of
/// the same fragment builds ([`SkipReason::SameFragment`]).
pub fn probed_filters(params: &TExecPlanFragmentParams) -> Vec<ProbedFilter> {
    let nodes = plan_nodes(params);
    let built_here = filters_built_in(nodes);
    nodes
        .iter()
        .filter(|node| is_scan(node))
        .flat_map(|node| {
            let built_here = &built_here;
            probe_filters_of(node)
                .filter(move |filter| {
                    filter
                        .filter_id
                        .is_some_and(|filter_id| !built_here.contains(&filter_id))
                })
                .filter_map(move |filter| {
                    Some(ProbedFilter {
                        filter_id: filter.filter_id?,
                        scan_node_id: node.node_id,
                        distribution: BuildDistribution::of(filter),
                        merge_node: merge_node(filter),
                        shares: filter
                            .layout
                            .as_ref()
                            .and_then(|layout| layout.num_instances)
                            .and_then(|instances| usize::try_from(instances).ok())
                            .filter(|&instances| instances > 0),
                        key_type: probe_key_type(filter, node.node_id),
                    })
                })
        })
        .collect()
}

/// Every join filter the hash joins of `nodes` build.
fn filters_built_in(nodes: &[TPlanNode]) -> Vec<i32> {
    nodes
        .iter()
        .filter_map(|node| node.hash_join_node.as_ref()?.build_runtime_filters.as_ref())
        .flatten()
        .filter_map(|filter| filter.filter_id)
        .collect()
}

/// Join filters `params`' fragment builds at a broadcast or partitioned hash join whose build
/// child is an exchange and whose key is a bare column of it. Filters of any other shape are left
/// out, and so are filters the semi-join rewrite would apply wrongly: a key the join compares
/// with `<=>`, and the broadcast branch of a skew join.
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
    for build in join_builds(nodes, &children) {
        if build.skip_reason().is_some() {
            continue;
        }
        let (Some(exchange), Some((filter_id, tuple_id, slot_id))) =
            (build.exchange, slot_key(build.filter))
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
        {
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
                join_node_id: build.join_node_id,
                exchange_node_id: exchange.node_id,
                column,
                column_type: type_mapper::duckdb_type_name(ty)?,
                distribution: BuildDistribution::of(build.filter),
                merge_node: merge_node(build.filter),
            });
        }
    }
    Ok(built)
}

/// Join filters `params`' fragment builds but [`built_filters`] leaves out, with the reason.
pub fn skipped_builds(params: &TExecPlanFragmentParams) -> Result<Vec<SkippedFilter>> {
    let nodes = plan_nodes(params);
    if nodes.is_empty() {
        return Ok(Vec::new());
    }
    let children = child_indices(nodes)?;
    Ok(join_builds(nodes, &children)
        .filter_map(|build| {
            Some(SkippedFilter {
                filter_id: build.filter.filter_id?,
                node_id: build.join_node_id,
                reason: build.skip_reason()?,
            })
        })
        .collect())
}

/// Probe targets in `params`' fragment that a compute node never filters: a target that isn't a
/// scan, and a scan probing a filter a join of its own fragment builds. Other scans aren't
/// listed; whether they're filtered is decided when they run ([`probed_filters`]).
pub fn skipped_probes(params: &TExecPlanFragmentParams) -> Vec<SkippedFilter> {
    let nodes = plan_nodes(params);
    let built_here = filters_built_in(nodes);
    let built_here = &built_here;
    nodes
        .iter()
        .flat_map(|node| {
            probe_filters_of(node).filter_map(move |filter| {
                let filter_id = filter.filter_id?;
                let reason = if !is_scan(node) {
                    SkipReason::TargetNotScan
                } else if built_here.contains(&filter_id) {
                    SkipReason::SameFragment
                } else {
                    return None;
                };
                Some(SkippedFilter {
                    filter_id,
                    node_id: node.node_id,
                    reason,
                })
            })
        })
        .collect()
}

/// One join filter a hash join of the fragment builds.
struct JoinBuild<'a> {
    join_node_id: i32,
    join: &'a THashJoinNode,
    /// The join's build child, when it's an exchange.
    exchange: Option<&'a TPlanNode>,
    filter: &'a TRuntimeFilterDescription,
}

impl JoinBuild<'_> {
    /// Why the compute node can't build this filter from its exchange, if it can't.
    fn skip_reason(&self) -> Option<SkipReason> {
        let distribution = BuildDistribution::of(self.filter);
        if distribution == BuildDistribution::Other {
            Some(SkipReason::NotBroadcast)
        } else if distribution == BuildDistribution::Partitioned
            && merge_node(self.filter).is_none()
        {
            Some(SkipReason::NoMergeNode)
        } else if self.filter.is_broad_cast_join_in_skew == Some(true) {
            Some(SkipReason::Skew)
        } else if self.exchange.is_none() {
            Some(SkipReason::BuildNotExchange)
        } else if slot_key(self.filter).is_none() {
            Some(SkipReason::KeyNotSlot)
        } else if null_safe_key(self.join, self.filter) {
            Some(SkipReason::NullSafe)
        } else {
            None
        }
    }
}

/// Every join filter the hash joins of `nodes` build.
fn join_builds<'a>(
    nodes: &'a [TPlanNode],
    children: &'a [Vec<usize>],
) -> impl Iterator<Item = JoinBuild<'a>> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| Some((index, node, node.hash_join_node.as_ref()?)))
        .flat_map(move |(index, node, join)| {
            let exchange = children[index]
                .get(1)
                .map(|&child| &nodes[child])
                .filter(|child| child.node_type == TPlanNodeType::EXCHANGE_NODE);
            join.build_runtime_filters
                .iter()
                .flatten()
                .filter(|filter| is_join_filter(filter))
                .map(move |filter| JoinBuild {
                    join_node_id: node.node_id,
                    join,
                    exchange,
                    filter,
                })
        })
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

/// DuckDB type name of `filter`'s probe expression at `target`.
fn probe_key_type(filter: &TRuntimeFilterDescription, target: i32) -> Option<String> {
    let probe = filter.plan_node_id_to_target_expr.as_ref()?.get(&target)?;
    let ty = type_mapper::map_type_desc(&probe.nodes.first()?.type_, true).ok()?;
    type_mapper::duckdb_type_name(&ty).ok()
}

/// The FE's merge node for `filter`, where the root fragment's instance runs.
fn merge_node(filter: &TRuntimeFilterDescription) -> Option<TNetworkAddress> {
    filter.runtime_filter_merge_nodes.as_ref()?.first().cloned()
}

/// `(filter id, tuple id, slot id)` of a filter keyed on a bare slot.
///
/// A skew join's broadcast branch (`is_broad_cast_join_in_skew`, [`SkipReason::Skew`]) holds only
/// the skewed keys, and its probe scan also feeds the shuffle branch, so applying it there would
/// drop that branch's rows. StarRocks sends those keys to the merge node instead.
fn slot_key(filter: &TRuntimeFilterDescription) -> Option<(i32, i32, i32)> {
    let [node] = filter.build_expr.as_ref()?.nodes.as_slice() else {
        return None;
    };
    if node.node_type != TExprNodeType::SLOT_REF {
        return None;
    }
    let slot = node.slot_ref.as_ref()?;
    Some((filter.filter_id?, slot.tuple_id, slot.slot_id))
}

/// Whether `join` compares `filter`'s key with `<=>`. The FE plans filters on null-safe joins
/// too, but 4.1.3 doesn't mark them in the filter, so this reads the join condition the filter
/// was built from (`expr_order`), or every condition when it's unset. The semi-join that applies
/// a filter compares with `=`, which would drop the NULL keys `<=>` matches.
fn null_safe_key(join: &THashJoinNode, filter: &TRuntimeFilterDescription) -> bool {
    let null_safe = |condition: &starrocks_thrift::plan_nodes::TEqJoinCondition| {
        condition.opcode == Some(TExprOpcode::EQ_FOR_NULL)
    };
    match filter.expr_order.map(usize::try_from) {
        Some(Ok(order)) => join.eq_join_conjuncts.get(order).is_none_or(null_safe),
        Some(Err(_)) => true,
        None => join.eq_join_conjuncts.iter().any(null_safe),
    }
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
