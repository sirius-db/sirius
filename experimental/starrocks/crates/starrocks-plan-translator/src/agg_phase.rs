//! Phase classification for StarRocks aggregation nodes.
//!
//! StarRocks encodes the phase of a (possibly multi-phase) aggregation in two thrift fields:
//! `TAggregationNode.need_finalize` (does this node produce final values?) and, per measure,
//! `TAggregateExpr.is_merge_agg` (does this measure consume partial states?). The combinations
//! map onto the FE's phase names:
//!
//! | `need_finalize` | `is_merge_agg` | FE phase          | classification          |
//! |-----------------|----------------|-------------------|-------------------------|
//! | true            | none           | one-phase         | [`AggPhase::OneShot`]   |
//! | false           | none           | update serialize  | [`AggPhase::Partial`]   |
//! | true            | all            | merge finalize    | [`AggPhase::Merge`]     |
//! | false           | all            | merge serialize   | error (3/4-phase plan)  |
//! | —               | mixed          | —                 | error                   |

use starrocks_thrift::plan_nodes::{TAggregationNode, TPlanNodeType};

use crate::error::{Result, TranslateError};

/// The role an aggregation node plays in the FE's aggregation plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AggPhase {
    /// One-phase finalized aggregation over raw rows (`new_planner_agg_stage = 1` plans).
    OneShot,
    /// First phase of a two-phase plan ("update serialize"): aggregates raw rows and emits
    /// partial-state columns without finalizing.
    Partial,
    /// Final phase of a two-phase plan ("merge finalize"): merges partial states arriving
    /// from an exchange and finalizes.
    Merge,
}

/// Classifies an aggregation node's phase from `need_finalize` and the measures'
/// `is_merge_agg` flags.
///
/// A measure whose root expression carries no `agg_expr` counts as non-merge here; the
/// expression translator rejects such a measure with its own malformed-aggregate error.
pub(crate) fn classify(
    node_id: i32,
    node_type: TPlanNodeType,
    agg: &TAggregationNode,
) -> Result<AggPhase> {
    let merge_flags = agg.aggregate_functions.iter().map(|expr| {
        expr.nodes
            .first()
            .and_then(|root| root.agg_expr.as_ref())
            .is_some_and(|agg_expr| agg_expr.is_merge_agg)
    });
    let (mut merge, mut update) = (0usize, 0usize);
    for is_merge in merge_flags {
        if is_merge {
            merge += 1;
        } else {
            update += 1;
        }
    }
    match (agg.need_finalize, merge, update) {
        (true, 0, _) => Ok(AggPhase::OneShot),
        (false, 0, _) => Ok(AggPhase::Partial),
        (true, _, 0) => Ok(AggPhase::Merge),
        (false, _, 0) => Err(TranslateError::UnsupportedPlanNode {
            node_id,
            node_type,
            reason: "merge-serialize aggregation (a 3/4-phase DISTINCT plan) is not supported",
        }),
        _ => Err(TranslateError::UnsupportedPlanNode {
            node_id,
            node_type,
            reason: "aggregation node mixes merge and update aggregate functions",
        }),
    }
}
