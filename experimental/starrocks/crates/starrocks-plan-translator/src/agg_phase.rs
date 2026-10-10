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
//! A two-phase measure ships its partial state from its partial step to its merge step, across
//! an exchange. [`partial_state`] says which columns that is and how the merge step combines
//! them:
//!
//! | function    | partial-state columns           | merge step                     |
//! |-------------|---------------------------------|--------------------------------|
//! | `sum`       | BIGINT, DOUBLE or DECIMAL(38,s) | `sum`                          |
//! | `count`     | BIGINT                          | `sum` of the counts            |
//! | `min`/`max` | the input type                  | the same function              |
//! | `avg`       | DOUBLE sum, then a BIGINT count | `sum` of each, then sum / count |
//!
//! The FE gives a measure one slot of its tuple, whatever it ships: a partial `avg` slot is
//! typed VARBINARY. AVG's count is an extra column right after the slot's, so every later
//! column of the exchange row shifts by one.
//!
//! Anything else (the DISTINCT forms) is refused in a two-phase plan.

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

/// What a two-phase measure ships between its partial and merge steps.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PartialState {
    /// One column, which the merge step combines with `merge_function`.
    Column {
        /// Type of the column the partial step emits and the merge step reads from the exchange.
        ty: Type,
        /// Aggregate function the merge step applies to that column.
        merge_function: &'static str,
    },
    /// AVG: the DOUBLE sum of the inputs, then the BIGINT count of the non-null inputs in the
    /// next column. The merge step sums each and divides the sum by the count.
    Average,
}

impl PartialState {
    /// Types of the columns the measure ships, in exchange-row order.
    pub(crate) fn columns(&self) -> Vec<Type> {
        match self {
            Self::Column { ty, .. } => vec![ty.clone()],
            Self::Average => vec![type_mapper::fp64_type(true), type_mapper::i64_type(true)],
        }
    }
}

/// Returns the partial state of one two-phase measure.
///
/// The rule reads only the measure's `TFunction` (its name and `ret_type`), which the FE
/// serializes the same way on the partial and the merge node, so both fragments derive the
/// same columns. Neither trusts the descriptor's slot type: StarRocks warns that a node's slot
/// types do not always say what it emits, and a partial `avg` slot is VARBINARY. The types are
/// what Sirius emits:
/// - `sum` over decimals is the FE's `DECIMAL128(38,s)`, which DuckDB's exact decimal sum also
///   returns, and an integer sum comes back as BIGINT;
/// - `count` is BIGINT. Merging counts has to add them up; a `count` on the merge side would
///   count the partial rows instead;
/// - `min`/`max` keep their input type, which StarRocks also uses as their return type;
/// - `avg` ships an FP64 sum (decimal and integer inputs are summed as FP64, as a one-phase
///   `avg` computes them) and a BIGINT count. Its return type is DOUBLE, or DECIMAL for decimal
///   inputs; `expr_translator::aggregate_call` refuses the others.
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
                type_mapper::map_type_desc(&function.ret_type, true)?
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
            Ok(PartialState::Column {
                ty,
                merge_function: "sum",
            })
        }
        "count" => Ok(PartialState::Column {
            ty: type_mapper::i64_type(true),
            merge_function: "sum",
        }),
        "min" => Ok(PartialState::Column {
            ty: type_mapper::map_type_desc(&function.ret_type, true)?,
            merge_function: "min",
        }),
        "max" => Ok(PartialState::Column {
            ty: type_mapper::map_type_desc(&function.ret_type, true)?,
            merge_function: "max",
        }),
        "avg" => {
            if !is_decimal(&function.ret_type)?
                && type_mapper::scalar_primitive(&function.ret_type)? != TPrimitiveType::DOUBLE
            {
                return Err(unsupported(
                    "two-phase AVG supports DOUBLE and DECIMAL results only",
                ));
            }
            Ok(PartialState::Average)
        }
        name if name.starts_with("multi_distinct_") => Err(unsupported(
            "DISTINCT aggregates are not supported in two-phase aggregation",
        )),
        _ => Err(unsupported(
            "two-phase aggregation supports SUM, COUNT, MIN, MAX and AVG only",
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
