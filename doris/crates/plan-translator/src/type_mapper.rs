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
    } else if scalar.type_ == TPrimitiveType::DECIMALV2 {
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
    } else {
        bail!("unsupported Doris primitive type: {}", scalar.type_.0)
    };
    Ok(Type { kind: Some(kind) })
}
