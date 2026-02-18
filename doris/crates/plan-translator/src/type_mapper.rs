//! Type mapping: Doris TPrimitiveType → Substrait Type.

use anyhow::{bail, Result};

use doris_thrift::types::{TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType};
use substrait::proto::r#type;
use substrait::proto::Type;

fn nullability(nullable: bool) -> i32 {
    if nullable {
        r#type::Nullability::Nullable as i32
    } else {
        r#type::Nullability::Required as i32
    }
}

/// Create a nullable I64 type kind (for fallback when type info is unavailable).
pub fn nullable_i64() -> r#type::Kind {
    r#type::Kind::I64(r#type::I64 {
        type_variation_reference: 0,
        nullability: r#type::Nullability::Nullable as i32,
    })
}

/// Map a Doris TTypeDesc to a Substrait Type.
pub fn map_type_desc(type_desc: &TTypeDesc) -> Result<Type> {
    let types = type_desc
        .types
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("TTypeDesc has no types"))?;
    if types.is_empty() {
        bail!("TTypeDesc has empty types list");
    }
    let nullable = type_desc.is_nullable.unwrap_or(true);
    map_type_node(&types[0], nullable)
}

fn map_type_node(node: &TTypeNode, nullable: bool) -> Result<Type> {
    if node.type_ != TTypeNodeType::SCALAR {
        bail!("only scalar types supported, got type node kind {}", node.type_.0);
    }
    let scalar = node
        .scalar_type
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SCALAR TTypeNode missing scalar_type"))?;
    map_scalar_type(scalar, nullable)
}

/// Map a Doris TScalarType to a Substrait Type.
pub fn map_scalar_type(scalar: &TScalarType, nullable: bool) -> Result<Type> {
    let n = nullability(nullable);
    let kind = if scalar.type_ == TPrimitiveType::BOOLEAN {
        r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::TINYINT {
        r#type::Kind::I8(r#type::I8 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::SMALLINT {
        r#type::Kind::I16(r#type::I16 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::INT {
        r#type::Kind::I32(r#type::I32 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::BIGINT {
        r#type::Kind::I64(r#type::I64 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::FLOAT {
        r#type::Kind::Fp32(r#type::Fp32 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::DOUBLE {
        r#type::Kind::Fp64(r#type::Fp64 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::VARCHAR || scalar.type_ == TPrimitiveType::STRING {
        let len = scalar.len.unwrap_or(65535);
        r#type::Kind::Varchar(r#type::VarChar {
            length: len,
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::CHAR {
        let len = scalar.len.unwrap_or(255);
        r#type::Kind::FixedChar(r#type::FixedChar {
            length: len,
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::DATE || scalar.type_ == TPrimitiveType::DATEV2 {
        r#type::Kind::Date(r#type::Date {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::DATETIME || scalar.type_ == TPrimitiveType::DATETIMEV2
    {
        r#type::Kind::Timestamp(r#type::Timestamp {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::DECIMALV2
        || scalar.type_ == TPrimitiveType::DECIMAL32
        || scalar.type_ == TPrimitiveType::DECIMAL64
        || scalar.type_ == TPrimitiveType::DECIMAL128I
        || scalar.type_ == TPrimitiveType::DECIMAL256
    {
        let precision = scalar.precision.unwrap_or(27);
        let scale = scalar.scale.unwrap_or(9);
        r#type::Kind::Decimal(r#type::Decimal {
            precision,
            scale,
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::LARGEINT {
        // Map LARGEINT (128-bit) to I64 with potential overflow — best effort
        r#type::Kind::I64(r#type::I64 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::HLL
        || scalar.type_ == TPrimitiveType::BITMAP
        || scalar.type_ == TPrimitiveType::QUANTILE_STATE
        || scalar.type_ == TPrimitiveType::AGG_STATE
    {
        // Doris-specific aggregate types — map to Binary (opaque blobs).
        r#type::Kind::Binary(r#type::Binary {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::JSONB || scalar.type_ == TPrimitiveType::VARIANT {
        // JSON/variant types — map to String.
        r#type::Kind::String(r#type::String {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::IPV4 {
        r#type::Kind::I32(r#type::I32 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else if scalar.type_ == TPrimitiveType::TIMEV2 {
        // TIME type — map to I64 (microseconds since midnight).
        r#type::Kind::I64(r#type::I64 {
            type_variation_reference: 0,
            nullability: n,
        })
    } else {
        bail!("unsupported Doris primitive type: {}", scalar.type_.0)
    };
    Ok(Type { kind: Some(kind) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(ptype: TPrimitiveType) -> TScalarType {
        TScalarType {
            type_: ptype,
            len: None,
            precision: None,
            scale: None,
            variant_max_subcolumns_count: None,
        }
    }

    fn scalar_with_len(ptype: TPrimitiveType, len: i32) -> TScalarType {
        TScalarType {
            type_: ptype,
            len: Some(len),
            precision: None,
            scale: None,
            variant_max_subcolumns_count: None,
        }
    }

    fn scalar_decimal(ptype: TPrimitiveType, precision: i32, scale: i32) -> TScalarType {
        TScalarType {
            type_: ptype,
            len: None,
            precision: Some(precision),
            scale: Some(scale),
            variant_max_subcolumns_count: None,
        }
    }

    /// Assert that a scalar type maps to a specific Substrait Kind variant.
    fn assert_kind(s: &TScalarType, nullable: bool, expected_kind: &str) {
        let result = map_scalar_type(s, nullable).expect("map_scalar_type failed");
        let kind = result.kind.expect("Type has no kind");
        let kind_name = match &kind {
            r#type::Kind::Bool(_) => "Bool",
            r#type::Kind::I8(_) => "I8",
            r#type::Kind::I16(_) => "I16",
            r#type::Kind::I32(_) => "I32",
            r#type::Kind::I64(_) => "I64",
            r#type::Kind::Fp32(_) => "Fp32",
            r#type::Kind::Fp64(_) => "Fp64",
            r#type::Kind::Varchar(_) => "Varchar",
            r#type::Kind::FixedChar(_) => "FixedChar",
            r#type::Kind::String(_) => "String",
            r#type::Kind::Binary(_) => "Binary",
            r#type::Kind::Date(_) => "Date",
            r#type::Kind::Timestamp(_) => "Timestamp",
            r#type::Kind::Decimal(_) => "Decimal",
            _ => "Other",
        };
        assert_eq!(kind_name, expected_kind, "type {:?}", s.type_);
    }

    #[test]
    fn test_boolean() {
        assert_kind(&scalar(TPrimitiveType::BOOLEAN), true, "Bool");
    }

    #[test]
    fn test_integer_types() {
        assert_kind(&scalar(TPrimitiveType::TINYINT), true, "I8");
        assert_kind(&scalar(TPrimitiveType::SMALLINT), true, "I16");
        assert_kind(&scalar(TPrimitiveType::INT), true, "I32");
        assert_kind(&scalar(TPrimitiveType::BIGINT), true, "I64");
    }

    #[test]
    fn test_float_types() {
        assert_kind(&scalar(TPrimitiveType::FLOAT), true, "Fp32");
        assert_kind(&scalar(TPrimitiveType::DOUBLE), true, "Fp64");
    }

    #[test]
    fn test_string_types() {
        assert_kind(&scalar(TPrimitiveType::VARCHAR), true, "Varchar");
        assert_kind(&scalar(TPrimitiveType::STRING), true, "Varchar");
        assert_kind(&scalar(TPrimitiveType::CHAR), true, "FixedChar");
    }

    #[test]
    fn test_varchar_length() {
        let s = scalar_with_len(TPrimitiveType::VARCHAR, 100);
        let t = map_scalar_type(&s, true).unwrap();
        match t.kind.unwrap() {
            r#type::Kind::Varchar(vc) => assert_eq!(vc.length, 100),
            k => panic!("expected Varchar, got {:?}", k),
        }
    }

    #[test]
    fn test_date_types() {
        assert_kind(&scalar(TPrimitiveType::DATE), true, "Date");
        assert_kind(&scalar(TPrimitiveType::DATEV2), true, "Date");
        assert_kind(&scalar(TPrimitiveType::DATETIME), true, "Timestamp");
        assert_kind(&scalar(TPrimitiveType::DATETIMEV2), true, "Timestamp");
    }

    #[test]
    fn test_decimal_types() {
        assert_kind(&scalar(TPrimitiveType::DECIMALV2), true, "Decimal");
        assert_kind(&scalar(TPrimitiveType::DECIMAL32), true, "Decimal");
        assert_kind(&scalar(TPrimitiveType::DECIMAL64), true, "Decimal");
        assert_kind(&scalar(TPrimitiveType::DECIMAL128I), true, "Decimal");
        assert_kind(&scalar(TPrimitiveType::DECIMAL256), true, "Decimal");
    }

    #[test]
    fn test_decimal_precision_scale() {
        let s = scalar_decimal(TPrimitiveType::DECIMAL128I, 18, 4);
        let t = map_scalar_type(&s, true).unwrap();
        match t.kind.unwrap() {
            r#type::Kind::Decimal(d) => {
                assert_eq!(d.precision, 18);
                assert_eq!(d.scale, 4);
            }
            k => panic!("expected Decimal, got {:?}", k),
        }
    }

    #[test]
    fn test_largeint() {
        assert_kind(&scalar(TPrimitiveType::LARGEINT), true, "I64");
    }

    #[test]
    fn test_aggregate_types() {
        assert_kind(&scalar(TPrimitiveType::HLL), true, "Binary");
        assert_kind(&scalar(TPrimitiveType::BITMAP), true, "Binary");
        assert_kind(&scalar(TPrimitiveType::QUANTILE_STATE), true, "Binary");
        assert_kind(&scalar(TPrimitiveType::AGG_STATE), true, "Binary");
    }

    #[test]
    fn test_json_variant_types() {
        assert_kind(&scalar(TPrimitiveType::JSONB), true, "String");
        assert_kind(&scalar(TPrimitiveType::VARIANT), true, "String");
    }

    #[test]
    fn test_ipv4() {
        assert_kind(&scalar(TPrimitiveType::IPV4), true, "I32");
    }

    #[test]
    fn test_timev2() {
        assert_kind(&scalar(TPrimitiveType::TIMEV2), true, "I64");
    }

    #[test]
    fn test_nullability() {
        // Nullable
        let t = map_scalar_type(&scalar(TPrimitiveType::INT), true).unwrap();
        match t.kind.unwrap() {
            r#type::Kind::I32(i) => {
                assert_eq!(i.nullability, r#type::Nullability::Nullable as i32);
            }
            k => panic!("expected I32, got {:?}", k),
        }
        // Required (non-nullable)
        let t = map_scalar_type(&scalar(TPrimitiveType::INT), false).unwrap();
        match t.kind.unwrap() {
            r#type::Kind::I32(i) => {
                assert_eq!(i.nullability, r#type::Nullability::Required as i32);
            }
            k => panic!("expected I32, got {:?}", k),
        }
    }

    #[test]
    fn test_map_type_desc() {
        let td = TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(scalar(TPrimitiveType::BIGINT)),
                struct_fields: None,
                contains_null: None,
                contains_nulls: None,
            }]),
            is_nullable: Some(false),
            byte_size: None,
            sub_types: None,
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
        };
        let t = map_type_desc(&td).unwrap();
        match t.kind.unwrap() {
            r#type::Kind::I64(i) => {
                assert_eq!(i.nullability, r#type::Nullability::Required as i32);
            }
            k => panic!("expected I64, got {:?}", k),
        }
    }

    #[test]
    fn test_map_type_desc_nullable_default() {
        // When is_nullable is not set, default to nullable.
        let td = TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(scalar(TPrimitiveType::INT)),
                struct_fields: None,
                contains_null: None,
                contains_nulls: None,
            }]),
            is_nullable: None,
            byte_size: None,
            sub_types: None,
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
        };
        let t = map_type_desc(&td).unwrap();
        match t.kind.unwrap() {
            r#type::Kind::I32(i) => {
                assert_eq!(i.nullability, r#type::Nullability::Nullable as i32);
            }
            k => panic!("expected I32, got {:?}", k),
        }
    }
}
