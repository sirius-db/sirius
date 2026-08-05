//! The partial-state wire-type model for two-phase aggregation.
//!
//! In a two-phase plan the partial fragment ships one column per aggregate to the merge
//! fragment. The type of that column is decided by what the engine *binds*, not by what the FE
//! declares: the FE's intermediate slot says `DECIMAL128(38,s)` for a decimal sum (and an
//! opaque `VARBINARY` for avg), while the plan the translator emits casts decimal sum
//! arguments to FP64, so the column on the wire is a DOUBLE. Trusting the FE slot type would
//! declare the receiver's stream with the wrong schema and reinterpret the columns.
//!
//! [`wire_type`] is that binding, modeled as a pure function of thrift fields the FE
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
//! avg is not modeled: its Sirius partial state is *two* columns (sum + count) where the FE
//! allocates one slot, a cardinality change a type override cannot express. It stays a loud
//! error until the plan-shape rewrite (sum/count expansion + a division projection) lands.

use starrocks_thrift::types::TFunction;
use substrait::proto::Type;

use crate::error::{Result, TranslateError};
use crate::expr_translator::is_decimal;
use crate::type_mapper;

/// The wire type of the partial-state column a two-phase aggregate ships between fragments.
///
/// `function` is the measure's `TFunction` as serialized by the FE — identical on the partial
/// and the merge node, which is what guarantees both ends compute the same type.
pub(crate) fn wire_type(function: &TFunction) -> Result<Type> {
    use starrocks_thrift::types::TPrimitiveType;

    let name = function.name.function_name.as_str();
    match name {
        "sum" => {
            if is_decimal(&function.ret_type)? {
                return Ok(type_mapper::fp64_type(true));
            }
            match type_mapper::scalar_primitive(&function.ret_type)? {
                TPrimitiveType::BIGINT => Ok(type_mapper::i64_type(true)),
                TPrimitiveType::DOUBLE => Ok(type_mapper::fp64_type(true)),
                primitive => Err(TranslateError::malformed(format!(
                    "two-phase sum returning {primitive:?} has no modeled partial state \
                     (SET new_planner_agg_stage = 1)"
                ))),
            }
        }
        "count" => Ok(type_mapper::i64_type(true)),
        "min" | "max" => type_mapper::map_type_desc(&function.ret_type, true),
        "avg" => Err(TranslateError::malformed(
            "two-phase avg is not supported: StarRocks serializes avg's partial state as an \
             opaque VARBINARY while Sirius represents it as sum+count columns \
             (SET new_planner_agg_stage = 1)",
        )),
        other => Err(TranslateError::malformed(format!(
            "two-phase aggregation supports only sum/count/min/max; {other:?} has a partial \
             state this translator does not model (SET new_planner_agg_stage = 1)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use starrocks_thrift::types::{
        TFunction, TFunctionBinaryType, TFunctionName, TPrimitiveType, TScalarType, TTypeDesc,
        TTypeNode, TTypeNodeType,
    };
    use substrait::proto::r#type;

    use super::wire_type;

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

    fn kind(ty: &substrait::proto::Type) -> &r#type::Kind {
        ty.kind.as_ref().unwrap()
    }

    /// The row that would rot silently if the FE slot type were ever trusted: the FE declares
    /// a decimal sum's intermediate as DECIMAL128, but the wire column is a DOUBLE.
    #[test]
    fn decimal_sum_state_is_fp64_not_decimal() {
        let ty = wire_type(&function(
            "sum",
            scalar_type(TPrimitiveType::DECIMAL128, Some(38), Some(2)),
        ))
        .unwrap();
        assert!(matches!(kind(&ty), r#type::Kind::Fp64(_)), "{ty:?}");
    }

    #[test]
    fn integer_sum_state_is_i64() {
        let ty = wire_type(&function(
            "sum",
            scalar_type(TPrimitiveType::BIGINT, None, None),
        ))
        .unwrap();
        assert!(matches!(kind(&ty), r#type::Kind::I64(_)), "{ty:?}");
    }

    #[test]
    fn double_sum_state_is_fp64() {
        let ty = wire_type(&function(
            "sum",
            scalar_type(TPrimitiveType::DOUBLE, None, None),
        ))
        .unwrap();
        assert!(matches!(kind(&ty), r#type::Kind::Fp64(_)), "{ty:?}");
    }

    #[test]
    fn count_state_is_i64() {
        let ty = wire_type(&function(
            "count",
            scalar_type(TPrimitiveType::BIGINT, None, None),
        ))
        .unwrap();
        assert!(matches!(kind(&ty), r#type::Kind::I64(_)), "{ty:?}");
    }

    /// min/max keep their input type, including decimals (pinned on GPU by FRAG-8).
    #[test]
    fn min_max_state_is_the_identity() {
        for name in ["min", "max"] {
            let ty = wire_type(&function(
                name,
                scalar_type(TPrimitiveType::DECIMAL64, Some(15), Some(2)),
            ))
            .unwrap();
            let r#type::Kind::Decimal(decimal) = kind(&ty) else {
                panic!("expected decimal, got {ty:?}");
            };
            assert_eq!((decimal.precision, decimal.scale), (15, 2));

            let ty = wire_type(&function(
                name,
                scalar_type(TPrimitiveType::DATE, None, None),
            ))
            .unwrap();
            assert!(matches!(kind(&ty), r#type::Kind::Date(_)), "{ty:?}");
        }
    }

    #[test]
    fn avg_state_is_refused_with_the_session_var_workaround() {
        let err = wire_type(&function(
            "avg",
            scalar_type(TPrimitiveType::VARBINARY, None, None),
        ))
        .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("VARBINARY"), "{message}");
        assert!(message.contains("new_planner_agg_stage"), "{message}");
    }

    #[test]
    fn unmodeled_functions_are_refused() {
        let err = wire_type(&function(
            "multi_distinct_count",
            scalar_type(TPrimitiveType::BIGINT, None, None),
        ))
        .unwrap_err();
        assert!(err.to_string().contains("sum/count/min/max"), "{err}");
    }
}
