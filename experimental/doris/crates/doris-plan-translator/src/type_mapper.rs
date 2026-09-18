//! Doris `TTypeDesc` → Substrait type mapping and the type gate.
//!
//! Sirius consumes the plan through DuckDB's Substrait consumer, so the emitted
//! type has to be one that consumer accepts *and* one Sirius can carry on the GPU.
//! The allowlist is deliberately small: the TPC-H corpus over `local()` parquet needs
//! `BOOLEAN`, the four signed integers, `DECIMAL32/64/128I`, `DATEV2`, `VARCHAR` and
//! `STRING`; `FLOAT`/`DOUBLE`/`CHAR`/`DATETIMEV2` are mapped because they are
//! unambiguous. Everything else is rejected with a reason that names the entry in
//! `plan-doc/reference/semantics-gaps.md`:
//!
//! | Doris type | Substrait | Gap |
//! |---|---|---|
//! | `BOOLEAN` | `Bool` | |
//! | `TINYINT`/`SMALLINT`/`INT`/`BIGINT` | `I8`/`I16`/`I32`/`I64` | |
//! | `LARGEINT` | **rejected** | G-01: Sirius narrows 128-bit integers to 64 bits silently |
//! | `FLOAT`/`DOUBLE` | `Fp32`/`Fp64` | |
//! | `DECIMAL32`/`DECIMAL64`/`DECIMAL128I` | `Decimal{precision, scale}` | G-06: precision ≤ 4 rejected (DuckDB stores it as INT16, no cuDF carrier) |
//! | `DECIMALV2` | **rejected** | G-17: legacy `DECIMAL(27,9)` semantics |
//! | `DECIMAL256` | **rejected** | G-05: exceeds the 128-bit carrier |
//! | `DATEV2` | `Date` | |
//! | `DATETIMEV2(scale)` | `PrecisionTimestamp{0\|3\|6}` | scale rounded up to the next unit DuckDB has |
//! | `DATE`/`DATETIME` (v1) | **rejected** | G-17 |
//! | `CHAR(n)`/`VARCHAR(n)` | `VarChar{n}` | DuckDB has no fixed-width char; both land as `VARCHAR` |
//! | `STRING` | `String` | |
//! | `HLL`/`BITMAP`/`QUANTILE_STATE`/`AGG_STATE` | **rejected** | G-07 |
//! | `JSONB`/`VARIANT` | **rejected** | G-08 |
//! | `BINARY`/`VARBINARY`/`IPV4`/`IPV6`/`TIMEV2`/`TIMESTAMPTZ` | **rejected** | G-09 |
//! | `NULL_TYPE` | **rejected** | G-18: Sirius rejects plans whose output carries `SQLNULL` |
//! | `ARRAY`/`MAP`/`STRUCT`/`VARIANT` type nodes | **rejected** | G-10: nested columns rejected as a whole |
//!
//! Every rejection here is a *type-level* decision; expression-level gates (function
//! allowlist, decimal literal widths, cast behaviour) live in the expression translator.

use doris_thrift::types::{TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType};
use substrait::proto::Type;
use substrait::proto::r#type;

use crate::error::{Result, TranslateError};

/// Highest decimal precision the 128-bit carrier holds (Doris `DECIMAL128I` tops out here too).
pub const MAX_DECIMAL_PRECISION: i32 = 38;
/// Decimal precisions at or below this are stored as INT16 by DuckDB and have no cuDF carrier
/// (`sirius::get_cudf_type` throws; semantics-gaps G-06).
pub const MIN_DECIMAL_PRECISION_EXCLUSIVE: i32 = 4;
/// Highest fractional-second scale Doris allows on `DATETIMEV2`.
const MAX_DATETIMEV2_SCALE: i32 = 6;

/// Converts a boolean nullability flag into the Substrait enum value.
pub(crate) fn nullability(nullable: bool) -> i32 {
    if nullable {
        r#type::Nullability::Nullable as i32
    } else {
        r#type::Nullability::Required as i32
    }
}

/// Maps a Doris type descriptor plus the slot/expression nullability into a Substrait type.
///
/// `TTypeDesc.is_nullable` is not consulted: the FE leaves it unset on slot types (slot
/// nullability travels in `TSlotDescriptor.nullIndicatorBit`) and expression nodes carry
/// their own `is_nullable`, so the caller passes the flag that applies.
pub fn map_type_desc(type_desc: &TTypeDesc, nullable: bool) -> Result<Type> {
    map_scalar_type(scalar_type(type_desc)?, nullable)
}

/// Returns the primitive type of a scalar Doris type descriptor.
pub fn scalar_primitive(type_desc: &TTypeDesc) -> Result<TPrimitiveType> {
    Ok(scalar_type(type_desc)?.type_)
}

/// Returns the scalar type of a Doris type descriptor, rejecting nested type trees.
///
/// A scalar `TTypeDesc` has exactly one `SCALAR` node; nested types are a preorder list
/// headed by an `ARRAY`/`MAP`/`STRUCT`/`VARIANT` node (G-10).
pub fn scalar_type(type_desc: &TTypeDesc) -> Result<&TScalarType> {
    let types = type_desc
        .types
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TTypeDesc",
            field: "types",
        })?;
    let node: &TTypeNode = types
        .first()
        .ok_or_else(|| TranslateError::malformed("TTypeDesc.types is empty"))?;
    if node.type_ != TTypeNodeType::SCALAR {
        return Err(TranslateError::UnsupportedType {
            primitive: None,
            node_type: Some(node.type_),
            reason: "nested ARRAY/MAP/STRUCT/VARIANT columns are rejected as a whole (G-10)",
        });
    }
    if types.len() != 1 {
        return Err(TranslateError::malformed(format!(
            "scalar TTypeDesc carries {} type nodes, expected exactly one",
            types.len()
        )));
    }
    node.scalar_type
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TTypeNode",
            field: "scalar_type",
        })
}

/// Maps a Doris scalar type into a Substrait type, applying the type gate.
pub fn map_scalar_type(scalar: &TScalarType, nullable: bool) -> Result<Type> {
    let n = nullability(nullable);
    let reject = |reason: &'static str| TranslateError::UnsupportedType {
        primitive: Some(scalar.type_),
        node_type: Some(TTypeNodeType::SCALAR),
        reason,
    };
    let kind = match scalar.type_ {
        TPrimitiveType::BOOLEAN => r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::TINYINT => r#type::Kind::I8(r#type::I8 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::SMALLINT => r#type::Kind::I16(r#type::I16 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::INT => r#type::Kind::I32(r#type::I32 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::BIGINT => r#type::Kind::I64(r#type::I64 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::LARGEINT => {
            return Err(reject(
                "LARGEINT is 128-bit; Sirius narrows HUGEINT to INT64 and silently corrupts \
                 out-of-range values (G-01)",
            ));
        }
        TPrimitiveType::FLOAT => r#type::Kind::Fp32(r#type::Fp32 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::DOUBLE => r#type::Kind::Fp64(r#type::Fp64 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::DECIMAL32 | TPrimitiveType::DECIMAL64 | TPrimitiveType::DECIMAL128I => {
            let precision = scalar.precision.ok_or(TranslateError::MissingField {
                context: "TScalarType(DECIMAL)",
                field: "precision",
            })?;
            let scale = scalar.scale.ok_or(TranslateError::MissingField {
                context: "TScalarType(DECIMAL)",
                field: "scale",
            })?;
            if precision > MAX_DECIMAL_PRECISION {
                return Err(reject(
                    "decimal precision above 38 exceeds the 128-bit decimal carrier",
                ));
            }
            if precision <= MIN_DECIMAL_PRECISION_EXCLUSIVE {
                return Err(reject(
                    "DECIMAL with precision <= 4 is stored as INT16 by DuckDB and has no cuDF \
                     carrier (G-06)",
                ));
            }
            if scale < 0 || scale > precision {
                return Err(TranslateError::malformed(format!(
                    "decimal scale {scale} is outside 0..={precision}"
                )));
            }
            r#type::Kind::Decimal(r#type::Decimal {
                scale,
                precision,
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::DECIMALV2 => {
            return Err(reject(
                "legacy DECIMALV2 (fixed 27,9) is not translated; only DECIMAL32/64/128I (G-17)",
            ));
        }
        TPrimitiveType::DECIMAL256 => {
            return Err(reject(
                "DECIMAL256 exceeds the 128-bit decimal carrier (G-05)",
            ));
        }
        TPrimitiveType::DATEV2 => r#type::Kind::Date(r#type::Date {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::DATETIMEV2 => {
            // Substrait/DuckDB timestamps come in seconds, milli-, micro- and nanoseconds;
            // a Doris scale in between rounds *up* so no fractional digit is lost.
            let scale = scalar.scale.unwrap_or(0);
            let precision = match scale {
                0 => 0,
                1..=3 => 3,
                4..=MAX_DATETIMEV2_SCALE => 6,
                _ => {
                    return Err(TranslateError::malformed(format!(
                        "DATETIMEV2 scale {scale} is outside 0..={MAX_DATETIMEV2_SCALE}"
                    )));
                }
            };
            r#type::Kind::PrecisionTimestamp(r#type::PrecisionTimestamp {
                precision,
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::DATE | TPrimitiveType::DATETIME => {
            return Err(reject(
                "legacy DATE/DATETIME (v1) are not translated; only DATEV2/DATETIMEV2 (G-17)",
            ));
        }
        TPrimitiveType::CHAR | TPrimitiveType::VARCHAR => {
            // DuckDB's Substrait consumer has no fixed-char type and maps VarChar to VARCHAR
            // regardless of length, so CHAR(n) lands as VARCHAR(n) too.
            let length = scalar.len.ok_or(TranslateError::MissingField {
                context: "TScalarType(CHAR/VARCHAR)",
                field: "len",
            })?;
            r#type::Kind::Varchar(r#type::VarChar {
                length,
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::STRING => r#type::Kind::String(r#type::String {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::HLL
        | TPrimitiveType::BITMAP
        | TPrimitiveType::QUANTILE_STATE
        | TPrimitiveType::AGG_STATE => {
            return Err(reject(
                "Doris aggregate-state types have no Sirius implementation (G-07)",
            ));
        }
        TPrimitiveType::JSONB | TPrimitiveType::VARIANT => {
            return Err(reject(
                "JSONB/VARIANT have structured semantics Sirius would flatten to strings (G-08)",
            ));
        }
        TPrimitiveType::BINARY
        | TPrimitiveType::VARBINARY
        | TPrimitiveType::IPV4
        | TPrimitiveType::IPV6
        | TPrimitiveType::TIMEV2
        | TPrimitiveType::TIMESTAMPTZ => {
            return Err(reject("type has no Substrait/Sirius mapping (G-09)"));
        }
        TPrimitiveType::NULL_TYPE => {
            return Err(reject(
                "NULL_TYPE has no Substrait type; Sirius rejects plans whose output carries \
                 SQLNULL (G-18)",
            ));
        }
        _ => {
            return Err(reject("primitive type has no Substrait mapping"));
        }
    };
    Ok(Type { kind: Some(kind) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(type_: TPrimitiveType) -> TScalarType {
        TScalarType {
            type_,
            ..Default::default()
        }
    }

    fn decimal(type_: TPrimitiveType, precision: i32, scale: i32) -> TScalarType {
        TScalarType {
            type_,
            precision: Some(precision),
            scale: Some(scale),
            ..Default::default()
        }
    }

    fn type_desc(nodes: Vec<TTypeNode>) -> TTypeDesc {
        TTypeDesc {
            types: Some(nodes),
            ..Default::default()
        }
    }

    fn scalar_node(scalar: TScalarType) -> TTypeNode {
        TTypeNode {
            type_: TTypeNodeType::SCALAR,
            scalar_type: Some(scalar),
            ..Default::default()
        }
    }

    fn kind(scalar: &TScalarType) -> r#type::Kind {
        map_scalar_type(scalar, true).unwrap().kind.unwrap()
    }

    fn reject_reason(scalar: &TScalarType) -> &'static str {
        match map_scalar_type(scalar, true).unwrap_err() {
            TranslateError::UnsupportedType {
                primitive, reason, ..
            } => {
                assert_eq!(primitive, Some(scalar.type_));
                reason
            }
            other => panic!("expected UnsupportedType, got {other:?}"),
        }
    }

    #[test]
    fn integers_booleans_and_floats_map_by_width() {
        assert!(matches!(
            kind(&scalar(TPrimitiveType::BOOLEAN)),
            r#type::Kind::Bool(_)
        ));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::TINYINT)),
            r#type::Kind::I8(_)
        ));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::SMALLINT)),
            r#type::Kind::I16(_)
        ));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::INT)),
            r#type::Kind::I32(_)
        ));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::BIGINT)),
            r#type::Kind::I64(_)
        ));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::FLOAT)),
            r#type::Kind::Fp32(_)
        ));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::DOUBLE)),
            r#type::Kind::Fp64(_)
        ));
    }

    #[test]
    fn nullability_flag_is_carried_into_the_type() {
        let nullable = map_scalar_type(&scalar(TPrimitiveType::INT), true).unwrap();
        let required = map_scalar_type(&scalar(TPrimitiveType::INT), false).unwrap();
        assert_eq!(
            nullable.kind,
            Some(r#type::Kind::I32(r#type::I32 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            }))
        );
        assert_eq!(
            required.kind,
            Some(r#type::Kind::I32(r#type::I32 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Required as i32,
            }))
        );
    }

    #[test]
    fn decimals_keep_precision_and_scale_across_all_three_widths() {
        for (type_, precision, scale) in [
            (TPrimitiveType::DECIMAL32, 9, 2),
            (TPrimitiveType::DECIMAL64, 15, 2),
            (TPrimitiveType::DECIMAL128I, 38, 4),
        ] {
            assert_eq!(
                kind(&decimal(type_, precision, scale)),
                r#type::Kind::Decimal(r#type::Decimal {
                    scale,
                    precision,
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                }),
                "{type_:?}"
            );
        }
    }

    /// G-06: precision 5 is the narrowest decimal with a cuDF carrier; 4 and below are rejected.
    #[test]
    fn decimal_precision_at_most_4_is_rejected() {
        assert!(matches!(
            kind(&decimal(TPrimitiveType::DECIMAL32, 5, 2)),
            r#type::Kind::Decimal(_)
        ));
        let reason = reject_reason(&decimal(TPrimitiveType::DECIMAL32, 4, 2));
        assert!(reason.contains("G-06"), "{reason}");
    }

    #[test]
    fn decimal_precision_above_38_is_rejected() {
        let reason = reject_reason(&decimal(TPrimitiveType::DECIMAL128I, 39, 2));
        assert!(reason.contains("above 38"), "{reason}");
    }

    #[test]
    fn decimal_without_precision_or_with_bad_scale_is_rejected() {
        let err = map_scalar_type(&scalar(TPrimitiveType::DECIMAL64), true).unwrap_err();
        assert!(matches!(
            err,
            TranslateError::MissingField {
                field: "precision",
                ..
            }
        ));
        let err = map_scalar_type(&decimal(TPrimitiveType::DECIMAL64, 15, 16), true).unwrap_err();
        assert!(matches!(err, TranslateError::MalformedPlan(_)));
    }

    /// G-01: the one rejection that guards against silently wrong results.
    #[test]
    fn largeint_is_rejected() {
        let reason = reject_reason(&scalar(TPrimitiveType::LARGEINT));
        assert!(reason.contains("G-01"), "{reason}");
    }

    /// G-05.
    #[test]
    fn decimal256_is_rejected() {
        let reason = reject_reason(&decimal(TPrimitiveType::DECIMAL256, 76, 10));
        assert!(reason.contains("G-05"), "{reason}");
    }

    /// G-17: legacy DECIMALV2 / DATE / DATETIME.
    #[test]
    fn legacy_v1_types_are_rejected() {
        for type_ in [
            TPrimitiveType::DECIMALV2,
            TPrimitiveType::DATE,
            TPrimitiveType::DATETIME,
        ] {
            let reason = reject_reason(&scalar(type_));
            assert!(reason.contains("G-17"), "{type_:?}: {reason}");
        }
    }

    /// G-07.
    #[test]
    fn aggregate_state_types_are_rejected() {
        for type_ in [
            TPrimitiveType::HLL,
            TPrimitiveType::BITMAP,
            TPrimitiveType::QUANTILE_STATE,
            TPrimitiveType::AGG_STATE,
        ] {
            let reason = reject_reason(&scalar(type_));
            assert!(reason.contains("G-07"), "{type_:?}: {reason}");
        }
    }

    /// G-08.
    #[test]
    fn jsonb_and_variant_are_rejected() {
        for type_ in [TPrimitiveType::JSONB, TPrimitiveType::VARIANT] {
            let reason = reject_reason(&scalar(type_));
            assert!(reason.contains("G-08"), "{type_:?}: {reason}");
        }
    }

    /// G-09.
    #[test]
    fn binary_ip_time_and_tz_types_are_rejected() {
        for type_ in [
            TPrimitiveType::BINARY,
            TPrimitiveType::VARBINARY,
            TPrimitiveType::IPV4,
            TPrimitiveType::IPV6,
            TPrimitiveType::TIMEV2,
            TPrimitiveType::TIMESTAMPTZ,
        ] {
            let reason = reject_reason(&scalar(type_));
            assert!(reason.contains("G-09"), "{type_:?}: {reason}");
        }
    }

    /// G-18.
    #[test]
    fn null_type_is_rejected() {
        let reason = reject_reason(&scalar(TPrimitiveType::NULL_TYPE));
        assert!(reason.contains("G-18"), "{reason}");
    }

    #[test]
    fn be_internal_and_invalid_primitives_are_rejected() {
        for type_ in [
            TPrimitiveType::INVALID_TYPE,
            TPrimitiveType::UNSUPPORTED,
            TPrimitiveType::LAMBDA_FUNCTION,
            TPrimitiveType::UINT32,
            TPrimitiveType::UINT64,
            TPrimitiveType::FIXED_LENGTH_OBJECT,
            TPrimitiveType::ALL,
            TPrimitiveType(999),
        ] {
            let reason = reject_reason(&scalar(type_));
            assert!(
                reason.contains("no Substrait mapping"),
                "{type_:?}: {reason}"
            );
        }
    }

    #[test]
    fn datev2_maps_to_date() {
        assert!(matches!(
            kind(&scalar(TPrimitiveType::DATEV2)),
            r#type::Kind::Date(_)
        ));
    }

    #[test]
    fn datetimev2_scale_rounds_up_to_a_duckdb_precision() {
        let precision = |scale: Option<i32>| {
            let scalar = TScalarType {
                type_: TPrimitiveType::DATETIMEV2,
                scale,
                ..Default::default()
            };
            match kind(&scalar) {
                r#type::Kind::PrecisionTimestamp(ts) => ts.precision,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(precision(None), 0);
        assert_eq!(precision(Some(0)), 0);
        assert_eq!(precision(Some(1)), 3);
        assert_eq!(precision(Some(3)), 3);
        assert_eq!(precision(Some(4)), 6);
        assert_eq!(precision(Some(6)), 6);
        let scalar = TScalarType {
            type_: TPrimitiveType::DATETIMEV2,
            scale: Some(7),
            ..Default::default()
        };
        assert!(matches!(
            map_scalar_type(&scalar, true).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn strings_map_to_varchar_or_string() {
        let varchar = TScalarType {
            type_: TPrimitiveType::VARCHAR,
            len: Some(65533),
            ..Default::default()
        };
        assert_eq!(
            kind(&varchar),
            r#type::Kind::Varchar(r#type::VarChar {
                length: 65533,
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })
        );
        let char_ = TScalarType {
            type_: TPrimitiveType::CHAR,
            len: Some(1),
            ..Default::default()
        };
        assert!(matches!(kind(&char_), r#type::Kind::Varchar(v) if v.length == 1));
        assert!(matches!(
            kind(&scalar(TPrimitiveType::STRING)),
            r#type::Kind::String(_)
        ));
        let err = map_scalar_type(&scalar(TPrimitiveType::VARCHAR), true).unwrap_err();
        assert!(matches!(
            err,
            TranslateError::MissingField { field: "len", .. }
        ));
    }

    /// G-10: a nested type descriptor is headed by a non-scalar node.
    #[test]
    fn nested_type_descriptors_are_rejected() {
        let array = type_desc(vec![
            TTypeNode {
                type_: TTypeNodeType::ARRAY,
                contains_nulls: Some(vec![true]),
                ..Default::default()
            },
            scalar_node(scalar(TPrimitiveType::INT)),
        ]);
        match map_type_desc(&array, true).unwrap_err() {
            TranslateError::UnsupportedType {
                primitive,
                node_type,
                reason,
            } => {
                assert_eq!(primitive, None);
                assert_eq!(node_type, Some(TTypeNodeType::ARRAY));
                assert!(reason.contains("G-10"), "{reason}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn type_descriptor_shape_is_validated() {
        let empty = type_desc(vec![]);
        assert!(matches!(
            map_type_desc(&empty, true).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
        let missing = TTypeDesc::default();
        assert!(matches!(
            map_type_desc(&missing, true).unwrap_err(),
            TranslateError::MissingField { field: "types", .. }
        ));
        let two_scalars = type_desc(vec![
            scalar_node(scalar(TPrimitiveType::INT)),
            scalar_node(scalar(TPrimitiveType::INT)),
        ]);
        assert!(matches!(
            map_type_desc(&two_scalars, true).unwrap_err(),
            TranslateError::MalformedPlan(_)
        ));
        let no_scalar = type_desc(vec![TTypeNode {
            type_: TTypeNodeType::SCALAR,
            ..Default::default()
        }]);
        assert!(matches!(
            map_type_desc(&no_scalar, true).unwrap_err(),
            TranslateError::MissingField {
                field: "scalar_type",
                ..
            }
        ));
        assert_eq!(
            scalar_primitive(&type_desc(vec![scalar_node(scalar(
                TPrimitiveType::DATEV2
            ))]))
            .unwrap(),
            TPrimitiveType::DATEV2
        );
    }
}
