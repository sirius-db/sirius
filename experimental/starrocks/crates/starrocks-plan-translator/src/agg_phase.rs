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
//!
//! A two-phase measure ships one partial-state column from its partial step to its merge step,
//! across an exchange. [`partial_state`] says what that column is and how the merge step
//! combines it:
//!
//! | function    | partial-state column | merge step          |
//! |-------------|----------------------|---------------------|
//! | `sum`       | BIGINT or DOUBLE     | `sum`               |
//! | `count`     | BIGINT               | `sum` of the counts |
//! | `min`/`max` | the input type       | the same function   |
//!
//! Anything else (AVG, the DISTINCT forms) is refused in a two-phase plan.

use starrocks_thrift::exprs::TExpr;
use starrocks_thrift::plan_nodes::{TAggregationNode, TPlanNodeType};
use starrocks_thrift::types::{TFunction, TPrimitiveType};
use substrait::proto::Type;

use crate::error::{Result, TranslateError};
use crate::expr_translator::is_decimal;
use crate::type_mapper;

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

/// The column a two-phase measure ships between its partial and merge steps.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PartialState {
    /// Type of the column the partial step emits and the merge step reads from the exchange.
    pub ty: Type,
    /// Aggregate function the merge step applies to that column.
    pub merge_function: &'static str,
}

/// Returns the partial state of one two-phase measure.
///
/// The rule reads only the measure's `TFunction` (its name and `ret_type`), which the FE
/// serializes the same way on the partial and the merge node, so both fragments derive the
/// same column. Neither trusts the descriptor's slot type: StarRocks warns that a node's slot
/// types do not always say what it emits. The type is what Sirius emits:
/// - `sum` over decimals is lowered to FP64 by `expr_translator::aggregate_call`, and an
///   integer sum comes back as BIGINT;
/// - `count` is BIGINT. Merging counts has to add them up; a `count` on the merge side would
///   count the partial rows instead;
/// - `min`/`max` keep their input type, which StarRocks also uses as their return type.
pub(crate) fn partial_state(
    node_id: i32,
    node_type: TPlanNodeType,
    function: &TFunction,
) -> Result<PartialState> {
    let unsupported = |reason| TranslateError::UnsupportedPlanNode {
        node_id,
        node_type,
        reason,
    };
    match function.name.function_name.as_str() {
        "sum" => {
            let ty = if is_decimal(&function.ret_type)? {
                type_mapper::fp64_type(true)
            } else {
                match type_mapper::scalar_primitive(&function.ret_type)? {
                    TPrimitiveType::BIGINT => type_mapper::i64_type(true),
                    TPrimitiveType::DOUBLE => type_mapper::fp64_type(true),
                    _ => {
                        return Err(unsupported(
                            "two-phase SUM supports BIGINT, DOUBLE and DECIMAL results only",
                        ));
                    }
                }
            };
            Ok(PartialState {
                ty,
                merge_function: "sum",
            })
        }
        "count" => Ok(PartialState {
            ty: type_mapper::i64_type(true),
            merge_function: "sum",
        }),
        "min" => Ok(PartialState {
            ty: type_mapper::map_type_desc(&function.ret_type, true)?,
            merge_function: "min",
        }),
        "max" => Ok(PartialState {
            ty: type_mapper::map_type_desc(&function.ret_type, true)?,
            merge_function: "max",
        }),
        name if name.starts_with("multi_distinct_") => Err(unsupported(
            "DISTINCT aggregates are not supported in two-phase aggregation",
        )),
        _ => Err(unsupported(
            "two-phase aggregation supports SUM, COUNT, MIN and MAX only",
        )),
    }
}

/// Returns the `TFunction` at the root of an aggregate measure.
pub(crate) fn measure_function(expr: &TExpr) -> Result<&TFunction> {
    expr.nodes
        .first()
        .and_then(|root| root.fn_.as_ref())
        .ok_or(TranslateError::MissingField {
            context: "aggregate expression",
            field: "fn",
        })
}
