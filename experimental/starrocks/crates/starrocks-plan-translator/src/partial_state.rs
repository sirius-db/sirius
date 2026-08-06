//! The partial-state wire-type model for two-phase aggregation.
//!
//! In a two-phase plan the partial fragment ships one column per aggregate to the merge
//! fragment. The type of that column is decided by what the engine *binds*, not by what the FE
//! declares: the FE's intermediate slot says `DECIMAL128(38,s)` for a decimal sum (and an
//! opaque `VARBINARY` for avg), while the plan the translator emits casts decimal sum
//! arguments to FP64, so the column on the wire is a DOUBLE. Trusting the FE slot type would
//! declare the receiver's stream with the wrong schema and reinterpret the columns.
//!
//! [`wire_columns`] is that binding, modeled as a pure function of thrift fields the FE
//! serializes identically on the partial and the merge node (the function name and its
//! `ret_type`). Both fragments derive their side of the exchange from it — the partial
//! measure's declared output type and the merge fragment's declared stream schema — so the
//! two ends agree by construction. If this model ever drifts from the engine's real binding,
//! the engine rejects the batch at the hop (`Fragment::push_packed` / `relay_from` validate
//! schemas), so a drift is a loud error, not a wrong number.
//!
//! The rules mirror the engine's binding, pinned by the `[streaming_fragment]` FRAG-6/7/8
//! engine tests:
//! - decimal `sum` is lowered to FP64 by the argument cast in `expr_translator`;
//! - integer `sum` binds to DuckDB HUGEINT and the Sirius planner downcasts it to BIGINT;
//! - `count` is BIGINT;
//! - `min`/`max` keep their input type (identity), including decimals.
//!
//! avg is the one state that is not a single column: StarRocks allocates one opaque VARBINARY
//! slot for it, while Sirius keeps it as a DOUBLE sum plus a BIGINT count. The model is a
//! *list* of columns for that reason — the partial fragment emits two measures for the one FE
//! slot, the exchange row gains the extra column right behind the sum, and the merge fragment
//! divides them back into an average. avg's sum is FP64 for every supported input because the
//! partial fragment casts the argument, so the wire type does not depend on the input width.

use starrocks_thrift::types::TFunction;
use substrait::proto::Type;

use crate::error::{Result, TranslateError};
use crate::expr_translator::is_decimal;
use crate::type_mapper;

/// Name suffix of avg's count column, appended to the FE's intermediate slot name.
///
/// Deterministic on purpose: the partial fragment names the extra column, the name travels to
/// the receiver with the rest of the sender's output names, and both ends have to spell it the
/// same way for the hop to bind.
pub(crate) const COUNT_SUFFIX: &str = "__count";

/// One column of the partial state a two-phase aggregate ships between fragments.
#[derive(Debug)]
pub(crate) struct WireColumn {
    /// Suffix appended to the FE's intermediate slot name; empty for the measure's own column.
    pub suffix: &'static str,
    /// Modeled Substrait type of the column.
    pub ty: Type,
}

/// The wire columns of one two-phase aggregate's partial state, in emission order.
///
/// `function` is the measure's `TFunction` as serialized by the FE — identical on the partial
/// and the merge node, which is what guarantees both ends compute the same columns. All but
/// avg occupy exactly one column, so a longer list is the signal that the emitted row is wider
/// than the FE's tuple.
pub(crate) fn wire_columns(function: &TFunction) -> Result<Vec<WireColumn>> {
    use starrocks_thrift::types::TPrimitiveType;

    let name = function.name.function_name.as_str();
    match name {
        "sum" => {
            if is_decimal(&function.ret_type)? {
                return Ok(one(type_mapper::fp64_type(true)));
            }
            match type_mapper::scalar_primitive(&function.ret_type)? {
                TPrimitiveType::BIGINT => Ok(one(type_mapper::i64_type(true))),
                TPrimitiveType::DOUBLE => Ok(one(type_mapper::fp64_type(true))),
                primitive => Err(TranslateError::malformed(format!(
                    "two-phase sum returning {primitive:?} has no modeled partial state \
                     (SET new_planner_agg_stage = 1)"
                ))),
            }
        }
        "count" => Ok(one(type_mapper::i64_type(true))),
        "min" | "max" => Ok(one(type_mapper::map_type_desc(&function.ret_type, true)?)),
        "avg" => Ok(vec![
            WireColumn {
                suffix: "",
                ty: type_mapper::fp64_type(true),
            },
            WireColumn {
                suffix: COUNT_SUFFIX,
                ty: type_mapper::i64_type(true),
            },
        ]),
        other => Err(TranslateError::malformed(format!(
            "two-phase aggregation supports only sum/count/min/max/avg; {other:?} has a partial \
             state this translator does not model (SET new_planner_agg_stage = 1)"
        ))),
    }
}

/// Wraps a single-column partial state, which is every aggregate but avg.
fn one(ty: Type) -> Vec<WireColumn> {
    vec![WireColumn { suffix: "", ty }]
}

#[cfg(test)]
mod tests {
    use starrocks_thrift::types::{
        TFunction, TFunctionBinaryType, TFunctionName, TPrimitiveType, TScalarType, TTypeDesc,
        TTypeNode, TTypeNodeType,
    };
    use substrait::proto::r#type;

    use super::{WireColumn, wire_columns};

    fn scalar_type(
        primitive: TPrimitiveType,
        precision: Option<i32>,
        scale: Option<i32>,
    ) -> TTypeDesc {
        TTypeDesc::new(Some(vec![TTypeNode::new(
            TTypeNodeType::SCALAR,
            Some(TScalarType::new(primitive, None, precision, scale)),
            None,
            None,
        )]))
    }

    fn function(name: &str, ret_type: TTypeDesc) -> TFunction {
        TFunction::new(
            TFunctionName::new(None, name.to_string()),
            TFunctionBinaryType::BUILTIN,
            Vec::new(),
            ret_type,
            false,
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
            None,
            None,
        )
    }

    fn kind(column: &WireColumn) -> &r#type::Kind {
        column.ty.kind.as_ref().unwrap()
    }

    /// Asserts a state occupies exactly one column and returns it.
    fn sole(columns: Vec<WireColumn>) -> WireColumn {
        assert_eq!(columns.len(), 1, "expected a single-column state");
        columns.into_iter().next().unwrap()
    }

    /// The row that would rot silently if the FE slot type were ever trusted: the FE declares
    /// a decimal sum's intermediate as DECIMAL128, but the wire column is a DOUBLE.
    #[test]
    fn decimal_sum_state_is_fp64_not_decimal() {
        let column = sole(
            wire_columns(&function(
                "sum",
                scalar_type(TPrimitiveType::DECIMAL128, Some(38), Some(2)),
            ))
            .unwrap(),
        );
        assert!(matches!(kind(&column), r#type::Kind::Fp64(_)));
    }

    #[test]
    fn integer_sum_state_is_i64() {
        let column = sole(
            wire_columns(&function(
                "sum",
                scalar_type(TPrimitiveType::BIGINT, None, None),
            ))
            .unwrap(),
        );
        assert!(matches!(kind(&column), r#type::Kind::I64(_)));
    }

    #[test]
    fn double_sum_state_is_fp64() {
        let column = sole(
            wire_columns(&function(
                "sum",
                scalar_type(TPrimitiveType::DOUBLE, None, None),
            ))
            .unwrap(),
        );
        assert!(matches!(kind(&column), r#type::Kind::Fp64(_)));
    }

    #[test]
    fn count_state_is_i64() {
        let column = sole(
            wire_columns(&function(
                "count",
                scalar_type(TPrimitiveType::BIGINT, None, None),
            ))
            .unwrap(),
        );
        assert!(matches!(kind(&column), r#type::Kind::I64(_)));
    }

    /// min/max keep their input type, including decimals (pinned on GPU by FRAG-8).
    #[test]
    fn min_max_state_is_the_identity() {
        for name in ["min", "max"] {
            let column = sole(
                wire_columns(&function(
                    name,
                    scalar_type(TPrimitiveType::DECIMAL64, Some(15), Some(2)),
                ))
                .unwrap(),
            );
            let r#type::Kind::Decimal(decimal) = kind(&column) else {
                panic!("expected decimal, got {:?}", column.ty);
            };
            assert_eq!((decimal.precision, decimal.scale), (15, 2));

            let column = sole(
                wire_columns(&function(
                    name,
                    scalar_type(TPrimitiveType::DATE, None, None),
                ))
                .unwrap(),
            );
            assert!(matches!(kind(&column), r#type::Kind::Date(_)));
        }
    }

    /// avg is the state that changes the row's width: one FE slot, two Sirius columns.
    #[test]
    fn avg_state_is_two_columns() {
        // The FE's own declaration for the slot is the opaque VARBINARY the model ignores.
        let columns = wire_columns(&function(
            "avg",
            scalar_type(TPrimitiveType::VARBINARY, None, None),
        ))
        .unwrap();
        assert_eq!(columns.len(), 2);
        assert!(matches!(kind(&columns[0]), r#type::Kind::Fp64(_)));
        assert!(matches!(kind(&columns[1]), r#type::Kind::I64(_)));
        assert_eq!(columns[0].suffix, "");
        assert_eq!(columns[1].suffix, super::COUNT_SUFFIX);
    }

    /// A decimal avg ships the same two columns: the partial fragment casts the argument, so
    /// the sum column is FP64 whatever the FE's declared return type is.
    #[test]
    fn decimal_avg_state_is_the_same_two_columns() {
        let columns = wire_columns(&function(
            "avg",
            scalar_type(TPrimitiveType::DECIMAL128, Some(38), Some(2)),
        ))
        .unwrap();
        assert_eq!(columns.len(), 2);
        assert!(matches!(kind(&columns[0]), r#type::Kind::Fp64(_)));
        assert!(matches!(kind(&columns[1]), r#type::Kind::I64(_)));
    }

    #[test]
    fn unmodeled_functions_are_refused() {
        let err = wire_columns(&function(
            "multi_distinct_count",
            scalar_type(TPrimitiveType::BIGINT, None, None),
        ))
        .unwrap_err();
        assert!(err.to_string().contains("sum/count/min/max/avg"), "{err}");
    }
}
