use starrocks_thrift::types::{TPrimitiveType, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType};
use substrait::proto::Type;
use substrait::proto::r#type;

use crate::error::{Result, TranslateError};

/// Converts a boolean nullability flag into a Substrait enum value.
fn nullability(nullable: bool) -> i32 {
    if nullable {
        r#type::Nullability::Nullable as i32
    } else {
        r#type::Nullability::Required as i32
    }
}

/// Builds the nullable boolean type used by predicate function outputs.
pub(crate) fn bool_type() -> Type {
    Type {
        kind: Some(r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: r#type::Nullability::Nullable as i32,
        })),
    }
}

/// Builds an FP64 type used to lower arithmetic Sirius cannot execute on decimal columns.
pub(crate) fn fp64_type(nullable: bool) -> Type {
    Type {
        kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
            type_variation_reference: 0,
            nullability: nullability(nullable),
        })),
    }
}

/// Builds an I64 (BIGINT) type, the partial-state width of integer sums and counts.
pub(crate) fn i64_type(nullable: bool) -> Type {
    Type {
        kind: Some(r#type::Kind::I64(r#type::I64 {
            type_variation_reference: 0,
            nullability: nullability(nullable),
        })),
    }
}

/// Renders a Substrait type as the DuckDB type name the engine parses when a fragment declares
/// an input stream's schema.
///
/// A stream has no file to probe, so the engine cannot infer the schema; it is declared. Deriving
/// the declaration from the very `Type` the plan's read carries is what keeps the two from
/// drifting — the alternative, a second StarRocks-to-DuckDB mapping, would be a silent
/// wrong-results bug the first time the two disagreed.
pub fn duckdb_type_name(ty: &Type) -> Result<String> {
    let kind = ty
        .kind
        .as_ref()
        .ok_or_else(|| TranslateError::malformed("stream input column has no type"))?;
    let name = match kind {
        r#type::Kind::Bool(_) => "BOOLEAN".to_string(),
        r#type::Kind::I8(_) => "TINYINT".to_string(),
        r#type::Kind::I16(_) => "SMALLINT".to_string(),
        r#type::Kind::I32(_) => "INTEGER".to_string(),
        r#type::Kind::I64(_) => "BIGINT".to_string(),
        r#type::Kind::Fp32(_) => "FLOAT".to_string(),
        r#type::Kind::Fp64(_) => "DOUBLE".to_string(),
        // DuckDB's CHAR is an alias of VARCHAR, so the declared length carries no information.
        r#type::Kind::FixedChar(_) | r#type::Kind::Varchar(_) | r#type::Kind::String(_) => {
            "VARCHAR".to_string()
        }
        r#type::Kind::Binary(_) | r#type::Kind::FixedBinary(_) => "BLOB".to_string(),
        r#type::Kind::Date(_) => "DATE".to_string(),
        r#type::Kind::PrecisionTimestamp(_) => "TIMESTAMP".to_string(),
        r#type::Kind::Decimal(decimal) => {
            format!("DECIMAL({},{})", decimal.precision, decimal.scale)
        }
        _ => {
            return Err(TranslateError::malformed(
                "stream input column type has no DuckDB name",
            ));
        }
    };
    Ok(name)
}

/// Maps a StarRocks type descriptor and slot nullability into a Substrait type.
pub fn map_type_desc(type_desc: &TTypeDesc, nullable: bool) -> Result<Type> {
    let node = scalar_node(type_desc)?;
    map_type_node(node, nullable)
}

/// Returns the primitive scalar type for a StarRocks type descriptor.
pub(crate) fn scalar_primitive(type_desc: &TTypeDesc) -> Result<TPrimitiveType> {
    let node = scalar_node(type_desc)?;
    if node.type_ != TTypeNodeType::SCALAR {
        return Err(TranslateError::UnsupportedType {
            primitive: None,
            node_type: Some(node.type_),
            reason: "only scalar type nodes are supported",
        });
    }
    Ok(node
        .scalar_type
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TTypeNode",
            field: "scalar_type",
        })?
        .type_)
}

/// Returns the first scalar node in a StarRocks type descriptor.
fn scalar_node(type_desc: &TTypeDesc) -> Result<&TTypeNode> {
    let types = type_desc
        .types
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TTypeDesc",
            field: "types",
        })?;
    let node = types
        .first()
        .ok_or_else(|| TranslateError::malformed("TTypeDesc.types is empty"))?;
    Ok(node)
}

/// Maps a StarRocks type node into a Substrait type.
fn map_type_node(node: &TTypeNode, nullable: bool) -> Result<Type> {
    if node.type_ != TTypeNodeType::SCALAR {
        return Err(TranslateError::UnsupportedType {
            primitive: None,
            node_type: Some(node.type_),
            reason: "only scalar type nodes are supported",
        });
    }
    let scalar = node
        .scalar_type
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TTypeNode",
            field: "scalar_type",
        })?;
    map_scalar_type(scalar, nullable)
}

/// Maps a StarRocks scalar type into a Substrait scalar type.
pub fn map_scalar_type(scalar: &TScalarType, nullable: bool) -> Result<Type> {
    let n = nullability(nullable);
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
            return Err(TranslateError::UnsupportedType {
                primitive: Some(scalar.type_),
                node_type: Some(TTypeNodeType::SCALAR),
                reason: "LARGEINT is 128-bit and has no safe v1 Substrait mapping",
            });
        }
        TPrimitiveType::FLOAT => r#type::Kind::Fp32(r#type::Fp32 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::DOUBLE => r#type::Kind::Fp64(r#type::Fp64 {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::CHAR => {
            // StarRocks thrift occasionally omits scalar length; keep a stable
            // v1 fallback rather than inventing an unbounded fixed-char type.
            r#type::Kind::FixedChar(r#type::FixedChar {
                length: scalar.len.unwrap_or(255),
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::VARCHAR => {
            // Same length fallback as StarRocks' broad VARCHAR default.
            r#type::Kind::Varchar(r#type::VarChar {
                length: scalar.len.unwrap_or(65535),
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::BINARY | TPrimitiveType::VARBINARY => {
            r#type::Kind::Binary(r#type::Binary {
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::DATE => r#type::Kind::Date(r#type::Date {
            type_variation_reference: 0,
            nullability: n,
        }),
        TPrimitiveType::DATETIME => {
            // StarRocks DATETIME is represented at microsecond precision in
            // this slice; revisit when fractional precision is carried through.
            r#type::Kind::PrecisionTimestamp(r#type::PrecisionTimestamp {
                precision: 6,
                type_variation_reference: 0,
                nullability: n,
            })
        }
        TPrimitiveType::DECIMAL
        | TPrimitiveType::DECIMALV2
        | TPrimitiveType::DECIMAL32
        | TPrimitiveType::DECIMAL64
        | TPrimitiveType::DECIMAL128 => {
            let precision = scalar.precision.unwrap_or(38);
            if precision > 38 {
                return Err(TranslateError::UnsupportedType {
                    primitive: Some(scalar.type_),
                    node_type: Some(TTypeNodeType::SCALAR),
                    reason: "decimal precision above 38 has no safe v1 Substrait mapping",
                });
            }
            if precision <= 18 {
                r#type::Kind::Decimal(r#type::Decimal {
                    precision,
                    scale: scalar.scale.unwrap_or(0),
                    type_variation_reference: 0,
                    nullability: n,
                })
            } else {
                r#type::Kind::Fp64(r#type::Fp64 {
                    type_variation_reference: 0,
                    nullability: n,
                })
            }
        }
        TPrimitiveType::DECIMAL256 => {
            return Err(TranslateError::UnsupportedType {
                primitive: Some(scalar.type_),
                node_type: Some(TTypeNodeType::SCALAR),
                reason: "DECIMAL256 can exceed the i128 decimal encoding used in v1",
            });
        }
        TPrimitiveType::JSON | TPrimitiveType::VARIANT => {
            // Substrait has no StarRocks-specific JSON/VARIANT scalar here;
            // expose the serialized value as string until richer support lands.
            r#type::Kind::String(r#type::String {
                type_variation_reference: 0,
                nullability: n,
            })
        }
        _ => {
            return Err(TranslateError::UnsupportedType {
                primitive: Some(scalar.type_),
                node_type: Some(TTypeNodeType::SCALAR),
                reason: "primitive type has no v1 Substrait mapping",
            });
        }
    };

    Ok(Type { kind: Some(kind) })
}
