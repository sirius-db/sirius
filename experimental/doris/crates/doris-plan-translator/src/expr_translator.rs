//! Doris `TExpr` → Substrait `Expression`.
//!
//! A `TExpr` is a flat preorder list of `TExprNode`s; [`ExprNodeCursor`] rebuilds the tree by
//! consuming exactly `num_children` nodes per parent and rejects a list that is not consumed
//! exactly once (see the crate docs).
//!
//! # What the FE ships (4.1.4, Nereids)
//!
//! - Comparisons are `BINARY_PRED` with `opcode` (`EQ`…`GE`); arithmetic is `ARITHMETIC_EXPR`
//!   with **no opcode** — the operator is `fn_.name.function_name` (`add`/`subtract`/…).
//! - Operands are already coerced: the FE inserts `CAST_EXPR` wherever the two sides of a
//!   comparison or arithmetic differ, so no implicit casts are added here.
//! - Every node carries its result type in `type_` and `is_nullable`; literals are typed
//!   (`STRING_LITERAL` as `VARCHAR(65533)`, `DATE_LITERAL` as `DATEV2`, `DECIMAL_LITERAL`
//!   with the precision/scale of the literal itself, e.g. `0.2` is `DECIMAL32(1,1)`).
//! - `NOT LIKE` and `NOT IN` arrive as `COMPOUND_NOT(like)` and `IN_PRED{is_not_in}`.
//! - Aggregates are `AGG_EXPR` roots with `TAggregateExpr{is_merge_agg, param_types}`; the
//!   merge phase's child is a `SLOT_REF` to the serialized partial state (`VARCHAR(65533)`).
//!
//! # What the consumer accepts
//!
//! The plan is executed through DuckDB's Substrait consumer (`substrait/src/from_substrait.cpp`),
//! which drives a few encodings here:
//! - literals: it has `I8`/`I32`/`I64` but **no `I16`**, `Date` but **no timestamp literal**,
//!   `String`/`VarChar` but no `FixedChar`, and a typed `null` becomes an *untyped* `SQLNULL`.
//!   So `SMALLINT` literals go out as `I32`, `DATETIMEV2` literals as a string cast, and every
//!   `NULL_LITERAL` is wrapped in a `Cast` to its Doris type (Sirius rejects `SQLNULL` outputs,
//!   G-18).
//! - functions: only the name matters (the extension URN is dropped); `equal`/`not_equal`/`lt`/
//!   `lte`/`gt`/`gte`/`and`/`or`/`not`/`is_null`/`is_not_null`/`is_not_distinct_from` become
//!   DuckDB operators, `like` → `~~`, `substring` → `substr`, `octet_length` → `strlen`,
//!   `char_length` → `length`, anything else binds to the DuckDB function of that name.
//! - `output_type` on a scalar function is ignored — DuckDB re-derives types — so a Doris
//!   result type that DuckDB would infer differently (decimal `avg`, `year` → `SMALLINT`) has
//!   to be re-imposed with an explicit cast at tuple boundaries; the node translator does that.
//!
//! # Gates (see `plan-doc/reference/semantics-gaps.md`)
//!
//! - `LARGE_INT_LITERAL` → rejected (G-01).
//! - `concat` → rejected (G-02: Doris is NULL-strict, DuckDB's `concat` ignores NULLs).
//! - `like` → constant pattern without a backslash (G-03: Doris escapes with `\`, DuckDB does
//!   not without `ESCAPE`).
//! - `substring`/`substr` → `(expr, constant start > 0, constant length > 0)` (G-04: Doris
//!   follows MySQL for `pos <= 0` and negative positions, DuckDB does not).
//! - decimal literals with precision ≤ 4 are emitted with precision 5 — value-preserving, and it
//!   keeps the constant off the INT16 decimal carrier Sirius cannot use (G-06).
//! - everything outside the allowlist (`if`, `like`, `substring`/`substr`, `year`/`month`/`day`,
//!   `length`/`char_length`, `is_null_pred`/`is_not_null_pred`) → `UnsupportedExpression`.

use std::collections::HashMap;

use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use doris_thrift::opcodes::TExprOpcode;
use doris_thrift::types::{TPrimitiveType, TTypeNodeType};
use substrait::proto::expression::field_reference;
use substrait::proto::expression::reference_segment;
use substrait::proto::expression::{self, FieldReference, ReferenceSegment};
use substrait::proto::r#type;
use substrait::proto::{Expression, FunctionArgument, Type, function_argument};

use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::type_mapper;
use crate::{
    ExtensionRegistry, URN_AGGREGATE, URN_ARITHMETIC, URN_BOOLEAN, URN_COMPARISON, URN_DATETIME,
    URN_STRING,
};

/// Column index a `(tuple_id, slot_id)` resolves to when it is not looked up in the row layout.
pub type SlotOverrides = HashMap<(i32, i32), usize>;

/// State needed while translating one Doris expression tree.
pub struct ExprContext<'a> {
    /// Descriptor lookups for slot references.
    desc: &'a DescriptorTable,
    /// Extension registry shared with the enclosing plan translation.
    registry: &'a mut ExtensionRegistry,
    /// Tuple ids that make up the row an expression is evaluated over.
    row_tuples: &'a [i32],
    /// Slots resolved to a column index directly (a stitched fragment boundary, a synthetic
    /// column) instead of through `row_tuples`.
    slot_overrides: Option<&'a SlotOverrides>,
}

impl<'a> ExprContext<'a> {
    /// Creates an expression context over a row layout.
    pub fn new(
        desc: &'a DescriptorTable,
        registry: &'a mut ExtensionRegistry,
        row_tuples: &'a [i32],
    ) -> Self {
        Self {
            desc,
            registry,
            row_tuples,
            slot_overrides: None,
        }
    }

    /// Creates an expression context whose slot references are resolved through `overrides`
    /// first and the row layout second.
    pub fn with_slot_overrides(
        desc: &'a DescriptorTable,
        registry: &'a mut ExtensionRegistry,
        row_tuples: &'a [i32],
        overrides: &'a SlotOverrides,
    ) -> Self {
        Self {
            desc,
            registry,
            row_tuples,
            slot_overrides: Some(overrides),
        }
    }

    /// The descriptor table.
    pub fn desc(&self) -> &'a DescriptorTable {
        self.desc
    }

    /// The row layout expressions are evaluated over.
    pub fn row_tuples(&self) -> &'a [i32] {
        self.row_tuples
    }

    /// Resolves a slot reference to its column index.
    fn resolve_slot(&self, tuple_id: i32, slot_id: i32) -> Result<usize> {
        if let Some(index) = self
            .slot_overrides
            .and_then(|overrides| overrides.get(&(tuple_id, slot_id)))
        {
            return Ok(*index);
        }
        self.desc
            .slot_global_index(tuple_id, slot_id, self.row_tuples)
    }
}

/// Translates a whole `TExpr` (flat preorder) into one Substrait expression.
pub fn translate_expr(expr: &TExpr, ctx: &mut ExprContext<'_>) -> Result<Expression> {
    if expr.nodes.is_empty() {
        return Err(TranslateError::malformed("TExpr.nodes is empty"));
    }
    let mut cursor = ExprNodeCursor::new(&expr.nodes);
    let translated = cursor.translate_next(ctx)?;
    cursor.ensure_consumed()?;
    Ok(translated)
}

/// Cursor over the flat preorder `TExpr.nodes` list.
struct ExprNodeCursor<'a> {
    /// Node slice being parsed.
    nodes: &'a [TExprNode],
    /// Next node index to read.
    idx: usize,
}

impl<'a> ExprNodeCursor<'a> {
    /// Creates a cursor at the start of an expression node list.
    fn new(nodes: &'a [TExprNode]) -> Self {
        Self { nodes, idx: 0 }
    }

    /// Reads the next node and translates its subtree.
    fn translate_next(&mut self, ctx: &mut ExprContext<'_>) -> Result<Expression> {
        let node = self.next_node()?;
        let children = self.translate_children(node, ctx)?;
        translate_expr_node(node, children, ctx)
    }

    /// Reads the next node without translating it.
    fn next_node(&mut self) -> Result<&'a TExprNode> {
        let node = self
            .nodes
            .get(self.idx)
            .ok_or_else(|| TranslateError::malformed("unexpected end of expression nodes"))?;
        self.idx += 1;
        if node.num_children < 0 {
            return Err(TranslateError::malformed(format!(
                "expression node {:?} has negative child count {}",
                node.node_type, node.num_children
            )));
        }
        Ok(node)
    }

    /// Translates the `num_children` subtrees that follow `node`.
    fn translate_children(
        &mut self,
        node: &TExprNode,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Vec<Expression>> {
        (0..node.num_children)
            .map(|_| self.translate_next(ctx))
            .collect()
    }

    /// Verifies that the whole node list was consumed.
    fn ensure_consumed(&self) -> Result<()> {
        if self.idx != self.nodes.len() {
            return Err(TranslateError::malformed(format!(
                "TExpr had {} trailing node(s)",
                self.nodes.len() - self.idx
            )));
        }
        Ok(())
    }
}

/// Routes one node, whose children are already translated, to its translator.
fn translate_expr_node(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    match node.node_type {
        TExprNodeType::SLOT_REF => translate_slot_ref(node, children, ctx),
        TExprNodeType::BOOL_LITERAL => translate_bool_literal(node, children),
        TExprNodeType::INT_LITERAL => translate_int_literal(node, children),
        TExprNodeType::FLOAT_LITERAL => translate_float_literal(node, children),
        TExprNodeType::STRING_LITERAL => translate_string_literal(node, children),
        TExprNodeType::DECIMAL_LITERAL => translate_decimal_literal(node, children),
        TExprNodeType::DATE_LITERAL => translate_date_literal(node, children),
        TExprNodeType::NULL_LITERAL => translate_null_literal(node, children),
        TExprNodeType::LARGE_INT_LITERAL => Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "LARGEINT literals are 128-bit; Sirius narrows them to INT64 (G-01)",
        }),
        TExprNodeType::BINARY_PRED => translate_binary_pred(node, children, ctx),
        TExprNodeType::COMPOUND_PRED => translate_compound_pred(node, children, ctx),
        TExprNodeType::ARITHMETIC_EXPR => translate_arithmetic(node, children, ctx),
        TExprNodeType::CAST_EXPR => translate_cast(node, children),
        TExprNodeType::IN_PRED => translate_in_pred(node, children, ctx),
        TExprNodeType::IS_NULL_PRED => translate_is_null(node, children, ctx),
        TExprNodeType::CASE_EXPR => translate_case(node, children),
        TExprNodeType::FUNCTION_CALL => translate_function_call(node, children, ctx),
        TExprNodeType::AGG_EXPR => Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "aggregate expressions are only valid as AGGREGATION_NODE measures",
        }),
        _ => Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "expression node type is outside the supported slice",
        }),
    }
}

/// `SLOT_REF` → field selection over the row layout.
fn translate_slot_ref(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let slot_ref = node.slot_ref.as_ref().ok_or(TranslateError::MissingField {
        context: "SLOT_REF",
        field: "slot_ref",
    })?;
    if slot_ref.is_virtual_slot == Some(true) {
        return Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "virtual slots (materialized virtual columns) are not supported",
        });
    }
    let index = ctx.resolve_slot(slot_ref.tuple_id, slot_ref.slot_id)?;
    Ok(field_reference(index))
}

/// Builds a root struct-field reference to column `index`.
pub fn field_reference(index: usize) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::Selection(Box::new(FieldReference {
            reference_type: Some(field_reference::ReferenceType::DirectReference(
                ReferenceSegment {
                    reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                        reference_segment::StructField {
                            field: index as i32,
                            child: None,
                        },
                    ))),
                },
            )),
            root_type: Some(field_reference::RootType::RootReference(
                field_reference::RootReference {},
            )),
        }))),
    }
}

/// `BOOL_LITERAL`.
fn translate_bool_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .bool_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "BOOL_LITERAL",
            field: "bool_literal",
        })?;
    Ok(literal(expression::literal::LiteralType::Boolean(
        lit.value,
    )))
}

/// `INT_LITERAL`, width-matched to the node's type.
///
/// `SMALLINT` is emitted as `I32`: the consumer has no `I16` literal, and DuckDB compares an
/// INT32 constant against a SMALLINT column without widening the column.
fn translate_int_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let value = node
        .int_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "INT_LITERAL",
            field: "int_literal",
        })?
        .value;
    let overflow =
        |width: &str| TranslateError::malformed(format!("{width} literal {value} overflows"));
    let literal_type = match type_mapper::scalar_primitive(&node.type_)? {
        TPrimitiveType::TINYINT => expression::literal::LiteralType::I8(i32::from(
            i8::try_from(value).map_err(|_| overflow("TINYINT"))?,
        )),
        TPrimitiveType::SMALLINT => expression::literal::LiteralType::I32(i32::from(
            i16::try_from(value).map_err(|_| overflow("SMALLINT"))?,
        )),
        TPrimitiveType::INT => expression::literal::LiteralType::I32(
            i32::try_from(value).map_err(|_| overflow("INT"))?,
        ),
        TPrimitiveType::BIGINT => expression::literal::LiteralType::I64(value),
        primitive => {
            return Err(TranslateError::UnsupportedType {
                primitive: Some(primitive),
                node_type: Some(TTypeNodeType::SCALAR),
                reason: "INT_LITERAL has a non-integer type",
            });
        }
    };
    Ok(literal(literal_type))
}

/// `FLOAT_LITERAL`, width-matched to the node's type.
fn translate_float_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let value = *node
        .float_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "FLOAT_LITERAL",
            field: "float_literal",
        })?
        .value;
    let literal_type = match type_mapper::scalar_primitive(&node.type_)? {
        TPrimitiveType::FLOAT => expression::literal::LiteralType::Fp32(value as f32),
        TPrimitiveType::DOUBLE => expression::literal::LiteralType::Fp64(value),
        primitive => {
            return Err(TranslateError::UnsupportedType {
                primitive: Some(primitive),
                node_type: Some(TTypeNodeType::SCALAR),
                reason: "FLOAT_LITERAL has a non-floating type",
            });
        }
    };
    Ok(literal(literal_type))
}

/// `STRING_LITERAL` → string literal (the consumer treats `String` and `VarChar` alike).
fn translate_string_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .string_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "STRING_LITERAL",
            field: "string_literal",
        })?;
    Ok(literal(expression::literal::LiteralType::String(
        lit.value.clone(),
    )))
}

/// `DECIMAL_LITERAL` → 128-bit decimal literal with the literal's own scale.
///
/// The FE types a literal with exactly the digits it has (`0.2` is `DECIMAL32(1,1)`), which
/// can be below the narrowest decimal Sirius carries (G-06); the declared precision is
/// widened to 5 in that case, which changes no value.
fn translate_decimal_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .decimal_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "DECIMAL_LITERAL",
            field: "decimal_literal",
        })?;
    let scalar = type_mapper::scalar_type(&node.type_)?;
    if !matches!(
        scalar.type_,
        TPrimitiveType::DECIMAL32 | TPrimitiveType::DECIMAL64 | TPrimitiveType::DECIMAL128I
    ) {
        return Err(TranslateError::UnsupportedType {
            primitive: Some(scalar.type_),
            node_type: Some(TTypeNodeType::SCALAR),
            reason: "DECIMAL_LITERAL must be typed DECIMAL32/64/128I (DECIMALV2/DECIMAL256 are rejected)",
        });
    }
    let precision = scalar.precision.ok_or(TranslateError::MissingField {
        context: "DECIMAL_LITERAL type",
        field: "precision",
    })?;
    let scale = scalar.scale.ok_or(TranslateError::MissingField {
        context: "DECIMAL_LITERAL type",
        field: "scale",
    })?;
    if precision > type_mapper::MAX_DECIMAL_PRECISION || scale < 0 || scale > precision {
        return Err(TranslateError::malformed(format!(
            "DECIMAL_LITERAL {:?} declared as DECIMAL({precision},{scale})",
            lit.value
        )));
    }
    let precision = precision.max(type_mapper::MIN_DECIMAL_PRECISION_EXCLUSIVE + 1);
    let value = encode_decimal(&lit.value, scale)?;
    Ok(literal(expression::literal::LiteralType::Decimal(
        expression::literal::Decimal {
            value: value.to_vec(),
            precision,
            scale,
        },
    )))
}

/// `DATE_LITERAL` → `Date` (epoch days) for `DATEV2`, or a string cast for `DATETIMEV2`
/// (the consumer has no timestamp literal; DuckDB folds the cast at bind time).
fn translate_date_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .date_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "DATE_LITERAL",
            field: "date_literal",
        })?;
    match type_mapper::scalar_primitive(&node.type_)? {
        TPrimitiveType::DATEV2 => Ok(literal(expression::literal::LiteralType::Date(
            epoch_days_from_date_str(&lit.value)?,
        ))),
        TPrimitiveType::DATETIMEV2 => {
            // Validate the date part so a malformed literal fails here, not inside DuckDB.
            epoch_days_from_date_str(&lit.value)?;
            Ok(cast(
                literal(expression::literal::LiteralType::String(lit.value.clone())),
                type_mapper::map_type_desc(&node.type_, false)?,
            ))
        }
        primitive => Err(TranslateError::UnsupportedType {
            primitive: Some(primitive),
            node_type: Some(TTypeNodeType::SCALAR),
            reason: "DATE_LITERAL must be typed DATEV2 or DATETIMEV2 (G-17)",
        }),
    }
}

/// `NULL_LITERAL` → typed null, wrapped in a cast because the consumer drops the literal's type.
fn translate_null_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let null_type = type_mapper::map_type_desc(&node.type_, true)?;
    Ok(cast(
        literal(expression::literal::LiteralType::Null(null_type.clone())),
        null_type,
    ))
}

/// Parses `YYYY-MM-DD` (ignoring any time suffix) into days since the UNIX epoch.
fn epoch_days_from_date_str(value: &str) -> Result<i32> {
    let invalid = || TranslateError::malformed(format!("invalid date literal {value:?}"));
    let date_part = value.split_whitespace().next().ok_or_else(invalid)?;
    let mut parts = date_part.split('-');
    let mut next = || -> Result<i64> {
        parts
            .next()
            .and_then(|part| part.parse::<i64>().ok())
            .ok_or_else(invalid)
    };
    let (year, month, day) = (next()?, next()?, next()?);
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(invalid());
    }
    // Howard Hinnant's civil-days algorithm; whole dates need no calendar library.
    let year_adjusted = if month <= 2 { year - 1 } else { year };
    let era = year_adjusted.div_euclid(400);
    let year_of_era = year_adjusted - era * 400;
    let month_shifted = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_shifted + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146097 + day_of_era - 719468;
    i32::try_from(days).map_err(|_| invalid())
}

/// `BINARY_PRED` → comparison function.
fn translate_binary_pred(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    expect_child_count(node, &children, 2)?;
    let opcode = node.opcode.ok_or(TranslateError::MissingField {
        context: "BINARY_PRED",
        field: "opcode",
    })?;
    let name = match opcode {
        TExprOpcode::EQ => "equal",
        TExprOpcode::NE => "not_equal",
        TExprOpcode::LT => "lt",
        TExprOpcode::LE => "lte",
        TExprOpcode::GT => "gt",
        TExprOpcode::GE => "gte",
        TExprOpcode::EQ_FOR_NULL => "is_not_distinct_from",
        _ => {
            return Err(TranslateError::UnsupportedExpression {
                node_type: node.node_type,
                reason: "binary predicate opcode is unsupported",
            });
        }
    };
    let anchor = ctx.registry.register_function(URN_COMPARISON, name);
    Ok(scalar_function(anchor, children, bool_type(node)))
}

/// `COMPOUND_PRED` → `and` / `or` / `not`.
fn translate_compound_pred(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    let opcode = node.opcode.ok_or(TranslateError::MissingField {
        context: "COMPOUND_PRED",
        field: "opcode",
    })?;
    let name = match opcode {
        TExprOpcode::COMPOUND_AND | TExprOpcode::COMPOUND_OR => {
            if children.len() < 2 {
                return Err(TranslateError::malformed(format!(
                    "{opcode:?} expected at least 2 children, got {}",
                    children.len()
                )));
            }
            if opcode == TExprOpcode::COMPOUND_AND {
                "and"
            } else {
                "or"
            }
        }
        TExprOpcode::COMPOUND_NOT => {
            expect_child_count(node, &children, 1)?;
            "not"
        }
        _ => {
            return Err(TranslateError::UnsupportedExpression {
                node_type: node.node_type,
                reason: "compound predicate opcode is unsupported",
            });
        }
    };
    let anchor = ctx.registry.register_function(URN_BOOLEAN, name);
    Ok(scalar_function(anchor, children, bool_type(node)))
}

/// `ARITHMETIC_EXPR` → arithmetic function; the operator is the function name (no opcode).
fn translate_arithmetic(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    expect_child_count(node, &children, 2)?;
    let name = match (function_name(node), node.opcode) {
        (Some("add"), _) | (None, Some(TExprOpcode::ADD)) => "add",
        (Some("subtract"), _) | (None, Some(TExprOpcode::SUBTRACT)) => "subtract",
        (Some("multiply"), _) | (None, Some(TExprOpcode::MULTIPLY)) => "multiply",
        (Some("divide"), _) | (None, Some(TExprOpcode::DIVIDE)) => "divide",
        (Some("mod"), _) | (None, Some(TExprOpcode::MOD)) => "modulus",
        (None, None) => {
            return Err(TranslateError::MissingField {
                context: "ARITHMETIC_EXPR",
                field: "fn",
            });
        }
        _ => {
            return Err(TranslateError::UnsupportedExpression {
                node_type: node.node_type,
                reason: "arithmetic operator is unsupported (add/subtract/multiply/divide/mod only)",
            });
        }
    };
    let output_type = node_output_type(node)?;
    let anchor = ctx.registry.register_function(URN_ARITHMETIC, name);
    Ok(scalar_function(anchor, children, output_type))
}

/// `CAST_EXPR` → cast to the node's type with throwing failure behaviour.
fn translate_cast(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 1)?;
    if node
        .opcode
        .is_some_and(|opcode| opcode == TExprOpcode::TRY_CAST)
    {
        return Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "TRY_CAST (null on failure) has no Substrait equivalent Sirius honours",
        });
    }
    let target = node_output_type(node)?;
    Ok(cast(children.into_iter().next().unwrap(), target))
}

/// `IN_PRED` → singular-or-list, negated for `NOT IN`.
fn translate_in_pred(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    let in_pred = node
        .in_predicate
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "IN_PRED",
            field: "in_predicate",
        })?;
    if children.len() < 2 {
        return Err(TranslateError::malformed(
            "IN_PRED expected a value and at least one list entry",
        ));
    }
    let mut children = children.into_iter();
    let value = children.next().unwrap();
    let in_list = Expression {
        rex_type: Some(expression::RexType::SingularOrList(Box::new(
            expression::SingularOrList {
                value: Some(Box::new(value)),
                options: children.collect(),
            },
        ))),
    };
    if in_pred.is_not_in {
        let anchor = ctx.registry.register_function(URN_BOOLEAN, "not");
        Ok(scalar_function(anchor, vec![in_list], bool_type(node)))
    } else {
        Ok(in_list)
    }
}

/// `IS_NULL_PRED` → `is_null` / `is_not_null`.
fn translate_is_null(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    expect_child_count(node, &children, 1)?;
    let is_not_null = node
        .is_null_pred
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "IS_NULL_PRED",
            field: "is_null_pred",
        })?
        .is_not_null;
    let name = if is_not_null {
        "is_not_null"
    } else {
        "is_null"
    };
    let anchor = ctx.registry.register_function(URN_COMPARISON, name);
    Ok(scalar_function(anchor, children, bool_type(node)))
}

/// `CASE_EXPR` → if-then chain.
///
/// Children are `[case] (when then)* [else]` per `TCaseExpr`. A leading case operand is not
/// supported (Nereids rewrites `CASE x WHEN v` into `x = v` comparisons before this point).
fn translate_case(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    let case = node
        .case_expr
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "CASE_EXPR",
            field: "case_expr",
        })?;
    if case.has_case_expr {
        return Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "CASE with a leading case operand is not supported",
        });
    }
    let mut children = children.into_iter();
    let r#else = if case.has_else_expr {
        children
            .next_back()
            .ok_or_else(|| TranslateError::malformed("CASE_EXPR missing its else child"))?
    } else {
        // SQL CASE without ELSE is NULL; the consumer needs the else branch and a typed null.
        let null_type = type_mapper::map_type_desc(&node.type_, true)?;
        cast(
            literal(expression::literal::LiteralType::Null(null_type.clone())),
            null_type,
        )
    };
    let mut ifs = Vec::new();
    while let Some(condition) = children.next() {
        let then = children
            .next()
            .ok_or_else(|| TranslateError::malformed("CASE_EXPR when without then"))?;
        ifs.push(expression::if_then::IfClause {
            r#if: Some(condition),
            then: Some(then),
        });
    }
    if ifs.is_empty() {
        return Err(TranslateError::malformed("CASE_EXPR has no when/then arms"));
    }
    Ok(if_then(ifs, r#else))
}

/// `FUNCTION_CALL` → allowlisted scalar function.
fn translate_function_call(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    let name = function_name(node).ok_or(TranslateError::MissingField {
        context: "FUNCTION_CALL",
        field: "fn",
    })?;
    let (urn, mapped) = match name {
        "if" => {
            expect_child_count(node, &children, 3)?;
            let mut children = children.into_iter();
            let condition = children.next().unwrap();
            let then = children.next().unwrap();
            let otherwise = children.next().unwrap();
            return Ok(if_then(
                vec![expression::if_then::IfClause {
                    r#if: Some(condition),
                    then: Some(then),
                }],
                otherwise,
            ));
        }
        "is_null_pred" => {
            expect_child_count(node, &children, 1)?;
            (URN_COMPARISON, "is_null")
        }
        "is_not_null_pred" => {
            expect_child_count(node, &children, 1)?;
            (URN_COMPARISON, "is_not_null")
        }
        "like" => {
            expect_child_count(node, &children, 2)?;
            match string_literal_value(&children[1]) {
                Some(pattern) if !pattern.contains('\\') => {}
                _ => {
                    return Err(TranslateError::UnsupportedExpression {
                        node_type: node.node_type,
                        reason: "LIKE requires a constant pattern without a backslash escape (G-03)",
                    });
                }
            }
            (URN_STRING, "like")
        }
        "substring" | "substr" => {
            expect_child_count(node, &children, 3)?;
            let constant_positive =
                |expr: &Expression| matches!(integer_literal_value(expr), Some(value) if value > 0);
            if !constant_positive(&children[1]) || !constant_positive(&children[2]) {
                return Err(TranslateError::UnsupportedExpression {
                    node_type: node.node_type,
                    reason: "substring requires constant positive start and length (G-04)",
                });
            }
            (URN_STRING, "substring")
        }
        // Doris `length` counts bytes (DuckDB `strlen`); `char_length` counts code points.
        "length" => {
            expect_child_count(node, &children, 1)?;
            (URN_STRING, "octet_length")
        }
        "char_length" | "character_length" => {
            expect_child_count(node, &children, 1)?;
            (URN_STRING, "char_length")
        }
        "year" | "month" | "day" => {
            expect_child_count(node, &children, 1)?;
            (URN_DATETIME, name)
        }
        "concat" => {
            return Err(TranslateError::UnsupportedExpression {
                node_type: node.node_type,
                reason: "concat is NULL-strict in Doris but ignores NULLs in DuckDB (G-02)",
            });
        }
        _ => {
            return Err(TranslateError::UnsupportedFunction {
                name: name.to_string(),
                reason: "scalar function is not in the allowlist (G-15)",
            });
        }
    };
    let output_type = node_output_type(node)?;
    let anchor = ctx.registry.register_function(urn, mapped);
    Ok(scalar_function(anchor, children, output_type))
}

/// A Doris aggregate call decomposed for an `AggregateRel` measure.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateCall {
    /// Substrait/DuckDB aggregate function name.
    pub name: &'static str,
    /// Translated arguments over the aggregation input row.
    pub arguments: Vec<Expression>,
    /// Whether the aggregate applies to distinct inputs.
    pub distinct: bool,
    /// `TAggregateExpr.is_merge_agg`: the arguments are partial states from an earlier phase.
    pub is_merge: bool,
    /// The Doris result type of the (finalized) aggregate, with its nullability.
    pub return_type: Type,
}

/// Decomposes an `AGG_EXPR` root (one `TAggregationNode.aggregate_functions` entry).
///
/// A merge-phase call is returned as-is (`is_merge`, with its partial-state slot reference as
/// the argument); whether that is acceptable is the caller's decision — a single fragment
/// cannot execute it, the single-plan stitcher replaces it with the update-phase call.
pub fn aggregate_call(expr: &TExpr, ctx: &mut ExprContext<'_>) -> Result<AggregateCall> {
    let root = expr
        .nodes
        .first()
        .ok_or_else(|| TranslateError::malformed("aggregate function TExpr is empty"))?;
    if root.node_type != TExprNodeType::AGG_EXPR {
        return Err(TranslateError::UnsupportedExpression {
            node_type: root.node_type,
            reason: "aggregate function root is not an AGG_EXPR",
        });
    }
    let agg_expr = root.agg_expr.as_ref().ok_or(TranslateError::MissingField {
        context: "AGG_EXPR",
        field: "agg_expr",
    })?;
    let name = function_name(root).ok_or(TranslateError::MissingField {
        context: "AGG_EXPR",
        field: "fn",
    })?;
    // `multi_distinct_sum`/`multi_distinct_avg` are absent on purpose: Sirius honours DISTINCT
    // for count only (COLLECT_SET) and would silently overcount a distinct sum.
    let (mapped, distinct) = match name {
        "sum" => ("sum", false),
        "count" => ("count", false),
        "min" => ("min", false),
        "max" => ("max", false),
        "avg" => ("avg", false),
        "multi_distinct_count" => ("count", true),
        _ => {
            return Err(TranslateError::UnsupportedFunction {
                name: name.to_string(),
                reason: "aggregate function is not in the allowlist (G-16)",
            });
        }
    };
    if distinct && root.num_children != 1 {
        return Err(TranslateError::UnsupportedExpression {
            node_type: root.node_type,
            reason: "distinct aggregates over several columns are not supported",
        });
    }
    let return_type = node_output_type(root)?;
    let mut cursor = ExprNodeCursor::new(&expr.nodes);
    cursor.next_node()?;
    let arguments = cursor.translate_children(root, ctx)?;
    cursor.ensure_consumed()?;
    Ok(AggregateCall {
        name: mapped,
        arguments,
        distinct,
        is_merge: agg_expr.is_merge_agg,
        return_type,
    })
}

/// Builds a Substrait aggregate function invocation for a measure.
pub fn aggregate_function(
    call: &AggregateCall,
    ctx: &mut ExprContext<'_>,
) -> substrait::proto::AggregateFunction {
    let anchor = ctx.registry.register_function(URN_AGGREGATE, call.name);
    substrait::proto::AggregateFunction {
        function_reference: anchor,
        arguments: call
            .arguments
            .iter()
            .cloned()
            .map(|expr| FunctionArgument {
                arg_type: Some(function_argument::ArgType::Value(expr)),
            })
            .collect(),
        output_type: Some(call.return_type.clone()),
        invocation: if call.distinct {
            substrait::proto::aggregate_function::AggregationInvocation::Distinct as i32
        } else {
            substrait::proto::aggregate_function::AggregationInvocation::All as i32
        },
        phase: substrait::proto::AggregationPhase::InitialToResult as i32,
        ..Default::default()
    }
}

/// The function name the FE attached to a node, if any.
fn function_name(node: &TExprNode) -> Option<&str> {
    node.fn_
        .as_ref()
        .map(|function| function.name.function_name.as_str())
}

/// The node's declared result type (`type_` + `is_nullable`, defaulting to nullable).
fn node_output_type(node: &TExprNode) -> Result<Type> {
    type_mapper::map_type_desc(&node.type_, node.is_nullable.unwrap_or(true))
}

/// The boolean type for a predicate node, keeping its declared nullability.
fn bool_type(node: &TExprNode) -> Type {
    Type {
        kind: Some(r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: type_mapper::nullability(node.is_nullable.unwrap_or(true)),
        })),
    }
}

/// Returns a Substrait expression's string-literal payload, if it is one.
fn string_literal_value(expr: &Expression) -> Option<&str> {
    match expr.rex_type.as_ref()? {
        expression::RexType::Literal(literal) => match literal.literal_type.as_ref()? {
            expression::literal::LiteralType::String(value) => Some(value),
            expression::literal::LiteralType::VarChar(varchar) => Some(&varchar.value),
            _ => None,
        },
        _ => None,
    }
}

/// Returns a Substrait expression's integer-literal payload, if it is one.
fn integer_literal_value(expr: &Expression) -> Option<i64> {
    match expr.rex_type.as_ref()? {
        expression::RexType::Literal(literal) => match literal.literal_type.as_ref()? {
            expression::literal::LiteralType::I8(value) => Some(i64::from(*value)),
            expression::literal::LiteralType::I16(value) => Some(i64::from(*value)),
            expression::literal::LiteralType::I32(value) => Some(i64::from(*value)),
            expression::literal::LiteralType::I64(value) => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

/// Verifies that a node has exactly the expected number of children.
fn expect_child_count(node: &TExprNode, children: &[Expression], expected: usize) -> Result<()> {
    if children.len() != expected {
        return Err(TranslateError::malformed(format!(
            "{:?} expected {} child(ren), got {}",
            node.node_type,
            expected,
            children.len()
        )));
    }
    Ok(())
}

/// Builds a literal expression.
pub fn literal(literal_type: expression::literal::LiteralType) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(literal_type),
            ..Default::default()
        })),
    }
}

/// Builds a cast with throwing failure behaviour.
pub fn cast(input: Expression, target: Type) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::Cast(Box::new(expression::Cast {
            r#type: Some(target),
            input: Some(Box::new(input)),
            failure_behavior: expression::cast::FailureBehavior::ThrowException as i32,
        }))),
    }
}

/// Builds an if-then expression.
fn if_then(ifs: Vec<expression::if_then::IfClause>, r#else: Expression) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::IfThen(Box::new(expression::IfThen {
            ifs,
            r#else: Some(Box::new(r#else)),
        }))),
    }
}

/// Builds a scalar-function expression from translated children.
pub fn scalar_function(anchor: u32, children: Vec<Expression>, output_type: Type) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::ScalarFunction(
            expression::ScalarFunction {
                function_reference: anchor,
                arguments: children
                    .into_iter()
                    .map(|expr| FunctionArgument {
                        arg_type: Some(function_argument::ArgType::Value(expr)),
                    })
                    .collect(),
                output_type: Some(output_type),
                ..Default::default()
            },
        )),
    }
}

/// Encodes a decimal literal string as Substrait's little-endian 128-bit unscaled integer.
fn encode_decimal(value: &str, scale: i32) -> Result<[u8; 16]> {
    let invalid = || TranslateError::malformed(format!("invalid decimal literal {value:?}"));
    if scale < 0 {
        return Err(invalid());
    }
    let raw = value.trim();
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.strip_prefix('+').unwrap_or(raw)),
    };
    let (int_part, frac_part) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if unsigned.is_empty()
        || !int_part.chars().all(|ch| ch.is_ascii_digit())
        || !frac_part.chars().all(|ch| ch.is_ascii_digit())
    {
        return Err(invalid());
    }
    let scale = scale as usize;
    if frac_part.len() > scale {
        return Err(TranslateError::malformed(format!(
            "decimal literal {value:?} has more fraction digits than its scale {scale}"
        )));
    }
    let mut digits = String::from(int_part);
    digits.push_str(frac_part);
    digits.extend(std::iter::repeat_n('0', scale - frac_part.len()));
    let digits = digits.trim_start_matches('0');
    let mut unscaled = if digits.is_empty() {
        0
    } else {
        digits.parse::<i128>().map_err(|_| invalid())?
    };
    if negative {
        unscaled = -unscaled;
    }
    Ok(unscaled.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use doris_thrift::descriptors::{TSlotDescriptor, TTupleDescriptor};
    use doris_thrift::exprs::{
        TAggregateExpr, TBoolLiteral, TCaseExpr, TDateLiteral, TDecimalLiteral, TFloatLiteral,
        TInPredicate, TIntLiteral, TIsNullPredicate, TSlotRef, TStringLiteral,
    };
    use doris_thrift::types::{
        TFunction, TFunctionName, TScalarType, TTypeDesc, TTypeNode, TTypeNodeType,
    };
    use substrait::proto::expression::RexType;
    use substrait::proto::expression::literal::LiteralType;

    use super::*;

    fn scalar_desc(type_: TPrimitiveType) -> TTypeDesc {
        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(TScalarType {
                    type_,
                    ..Default::default()
                }),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn decimal_desc(type_: TPrimitiveType, precision: i32, scale: i32) -> TTypeDesc {
        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(TScalarType {
                    type_,
                    precision: Some(precision),
                    scale: Some(scale),
                    ..Default::default()
                }),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn varchar_desc(len: i32) -> TTypeDesc {
        TTypeDesc {
            types: Some(vec![TTypeNode {
                type_: TTypeNodeType::SCALAR,
                scalar_type: Some(TScalarType {
                    type_: TPrimitiveType::VARCHAR,
                    len: Some(len),
                    ..Default::default()
                }),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    fn node(node_type: TExprNodeType, type_: TTypeDesc, num_children: i32) -> TExprNode {
        TExprNode {
            node_type,
            type_,
            num_children,
            output_scale: -1,
            is_nullable: Some(true),
            ..Default::default()
        }
    }

    fn slot_ref(tuple_id: i32, slot_id: i32, type_: TTypeDesc) -> TExprNode {
        TExprNode {
            slot_ref: Some(TSlotRef {
                slot_id,
                tuple_id,
                ..Default::default()
            }),
            ..node(TExprNodeType::SLOT_REF, type_, 0)
        }
    }

    fn int_literal(value: i64, type_: TPrimitiveType) -> TExprNode {
        TExprNode {
            int_literal: Some(TIntLiteral { value }),
            is_nullable: Some(false),
            ..node(TExprNodeType::INT_LITERAL, scalar_desc(type_), 0)
        }
    }

    fn string_literal(value: &str) -> TExprNode {
        TExprNode {
            string_literal: Some(TStringLiteral {
                value: value.to_string(),
            }),
            is_nullable: Some(false),
            ..node(TExprNodeType::STRING_LITERAL, varchar_desc(65533), 0)
        }
    }

    fn decimal_literal(
        value: &str,
        type_: TPrimitiveType,
        precision: i32,
        scale: i32,
    ) -> TExprNode {
        TExprNode {
            decimal_literal: Some(TDecimalLiteral {
                value: value.to_string(),
            }),
            is_nullable: Some(false),
            ..node(
                TExprNodeType::DECIMAL_LITERAL,
                decimal_desc(type_, precision, scale),
                0,
            )
        }
    }

    fn function(name: &str, ret: TTypeDesc) -> TFunction {
        TFunction {
            name: TFunctionName {
                db_name: None,
                function_name: name.to_string(),
            },
            ret_type: ret,
            ..Default::default()
        }
    }

    /// `BINARY_PRED` the way the FE writes it: opcode plus a `fn` named after it.
    fn binary_pred(opcode: TExprOpcode, name: &str) -> TExprNode {
        TExprNode {
            opcode: Some(opcode),
            fn_: Some(function(name, scalar_desc(TPrimitiveType::BOOLEAN))),
            ..node(
                TExprNodeType::BINARY_PRED,
                scalar_desc(TPrimitiveType::BOOLEAN),
                2,
            )
        }
    }

    /// `ARITHMETIC_EXPR` the way the FE writes it: no opcode, operator in `fn`.
    fn arithmetic(name: &str, type_: TTypeDesc) -> TExprNode {
        TExprNode {
            fn_: Some(function(name, type_.clone())),
            ..node(TExprNodeType::ARITHMETIC_EXPR, type_, 2)
        }
    }

    fn function_call(name: &str, type_: TTypeDesc, num_children: i32) -> TExprNode {
        TExprNode {
            fn_: Some(function(name, type_.clone())),
            ..node(TExprNodeType::FUNCTION_CALL, type_, num_children)
        }
    }

    fn expr(nodes: Vec<TExprNode>) -> TExpr {
        TExpr { nodes }
    }

    /// Tuple 0 = {1: a BIGINT, 2: b INT, 3: s STRING, 4: d DATEV2, 5: p DECIMAL64(15,2)};
    /// tuple 1 = {6: x BIGINT}.
    fn desc() -> DescriptorTable {
        let slot = |id, parent, name: &str, slot_type| TSlotDescriptor {
            id,
            parent,
            slot_type,
            column_pos: -1,
            null_indicator_bit: 0,
            col_name: name.to_string(),
            slot_idx: -1,
            is_materialized: true,
            ..Default::default()
        };
        let desc_tbl = doris_thrift::descriptors::TDescriptorTable {
            slot_descriptors: Some(vec![
                slot(1, 0, "a", scalar_desc(TPrimitiveType::BIGINT)),
                slot(2, 0, "b", scalar_desc(TPrimitiveType::INT)),
                slot(3, 0, "s", scalar_desc(TPrimitiveType::STRING)),
                slot(4, 0, "d", scalar_desc(TPrimitiveType::DATEV2)),
                slot(5, 0, "p", decimal_desc(TPrimitiveType::DECIMAL64, 15, 2)),
                slot(6, 1, "x", scalar_desc(TPrimitiveType::BIGINT)),
            ]),
            tuple_descriptors: vec![
                TTupleDescriptor {
                    id: 0,
                    ..Default::default()
                },
                TTupleDescriptor {
                    id: 1,
                    ..Default::default()
                },
            ],
            table_descriptors: None,
        };
        DescriptorTable::try_from(&desc_tbl).unwrap()
    }

    fn translate(nodes: Vec<TExprNode>) -> Result<(Expression, ExtensionRegistry)> {
        let desc = desc();
        let mut registry = ExtensionRegistry::new();
        let row_tuples = [0, 1];
        let translated = {
            let mut ctx = ExprContext::new(&desc, &mut registry, &row_tuples);
            translate_expr(&expr(nodes), &mut ctx)?
        };
        Ok((translated, registry))
    }

    fn translate_ok(nodes: Vec<TExprNode>) -> Expression {
        translate(nodes).unwrap().0
    }

    fn translate_err(nodes: Vec<TExprNode>) -> TranslateError {
        translate(nodes).unwrap_err()
    }

    fn field_index(expr: &Expression) -> i32 {
        match expr.rex_type.as_ref().unwrap() {
            RexType::Selection(selection) => match selection.reference_type.as_ref().unwrap() {
                field_reference::ReferenceType::DirectReference(segment) => {
                    match segment.reference_type.as_ref().unwrap() {
                        reference_segment::ReferenceType::StructField(field) => field.field,
                        other => panic!("{other:?}"),
                    }
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    fn literal_type(expr: &Expression) -> &LiteralType {
        match expr.rex_type.as_ref().unwrap() {
            RexType::Literal(literal) => literal.literal_type.as_ref().unwrap(),
            other => panic!("{other:?}"),
        }
    }

    fn scalar_function_parts(expr: &Expression) -> (u32, Vec<&Expression>) {
        match expr.rex_type.as_ref().unwrap() {
            RexType::ScalarFunction(function) => (
                function.function_reference,
                function
                    .arguments
                    .iter()
                    .map(|arg| match arg.arg_type.as_ref().unwrap() {
                        function_argument::ArgType::Value(value) => value,
                        other => panic!("{other:?}"),
                    })
                    .collect(),
            ),
            other => panic!("{other:?}"),
        }
    }

    fn function_name_of(registry: &ExtensionRegistry, anchor: u32) -> &str {
        registry.function_name(anchor).unwrap()
    }

    #[test]
    fn slot_refs_resolve_through_the_row_layout() {
        let translated = translate_ok(vec![slot_ref(1, 6, scalar_desc(TPrimitiveType::BIGINT))]);
        // Tuple 0 has five slots, so tuple 1's first slot is column 5.
        assert_eq!(field_index(&translated), 5);
        let translated = translate_ok(vec![slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING))]);
        assert_eq!(field_index(&translated), 2);
    }

    #[test]
    fn slot_overrides_take_precedence_over_the_row_layout() {
        let desc = desc();
        let mut registry = ExtensionRegistry::new();
        let row_tuples = [0];
        let mut overrides = SlotOverrides::new();
        overrides.insert((9, 99), 7);
        let mut ctx =
            ExprContext::with_slot_overrides(&desc, &mut registry, &row_tuples, &overrides);
        let translated = translate_expr(
            &expr(vec![slot_ref(9, 99, scalar_desc(TPrimitiveType::BIGINT))]),
            &mut ctx,
        )
        .unwrap();
        assert_eq!(field_index(&translated), 7);
        let err = translate_expr(
            &expr(vec![slot_ref(1, 6, scalar_desc(TPrimitiveType::BIGINT))]),
            &mut ctx,
        )
        .unwrap_err();
        assert!(matches!(err, TranslateError::Descriptor(_)));
    }

    #[test]
    fn unknown_slot_is_a_descriptor_error() {
        let err = translate_err(vec![slot_ref(0, 42, scalar_desc(TPrimitiveType::BIGINT))]);
        assert!(matches!(err, TranslateError::Descriptor(_)));
    }

    #[test]
    fn virtual_slots_are_rejected() {
        let mut node = slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT));
        node.slot_ref.as_mut().unwrap().is_virtual_slot = Some(true);
        assert!(matches!(
            translate_err(vec![node]),
            TranslateError::UnsupportedExpression { .. }
        ));
    }

    #[test]
    fn int_literals_are_width_matched_and_smallint_widens_to_i32() {
        assert_eq!(
            literal_type(&translate_ok(vec![int_literal(1, TPrimitiveType::TINYINT)])),
            &LiteralType::I8(1)
        );
        assert_eq!(
            literal_type(&translate_ok(vec![int_literal(
                1995,
                TPrimitiveType::SMALLINT
            )])),
            &LiteralType::I32(1995)
        );
        assert_eq!(
            literal_type(&translate_ok(vec![int_literal(7, TPrimitiveType::INT)])),
            &LiteralType::I32(7)
        );
        assert_eq!(
            literal_type(&translate_ok(vec![int_literal(
                1 << 40,
                TPrimitiveType::BIGINT
            )])),
            &LiteralType::I64(1 << 40)
        );
        assert!(matches!(
            translate_err(vec![int_literal(300, TPrimitiveType::TINYINT)]),
            TranslateError::MalformedPlan(_)
        ));
        assert!(matches!(
            translate_err(vec![int_literal(70000, TPrimitiveType::SMALLINT)]),
            TranslateError::MalformedPlan(_)
        ));
        assert!(matches!(
            translate_err(vec![int_literal(1, TPrimitiveType::STRING)]),
            TranslateError::UnsupportedType { .. }
        ));
    }

    #[test]
    fn largeint_literals_are_rejected() {
        let node = TExprNode {
            large_int_literal: Some(doris_thrift::exprs::TLargeIntLiteral {
                value: "170141183460469231731687303715884105727".to_string(),
            }),
            ..node(
                TExprNodeType::LARGE_INT_LITERAL,
                scalar_desc(TPrimitiveType::LARGEINT),
                0,
            )
        };
        match translate_err(vec![node]) {
            TranslateError::UnsupportedExpression { reason, .. } => {
                assert!(reason.contains("G-01"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn bool_float_and_string_literals() {
        let bool_node = TExprNode {
            bool_literal: Some(TBoolLiteral { value: true }),
            ..node(
                TExprNodeType::BOOL_LITERAL,
                scalar_desc(TPrimitiveType::BOOLEAN),
                0,
            )
        };
        assert_eq!(
            literal_type(&translate_ok(vec![bool_node])),
            &LiteralType::Boolean(true)
        );
        let float_node = TExprNode {
            float_literal: Some(TFloatLiteral {
                value: 2.5f64.into(),
            }),
            ..node(
                TExprNodeType::FLOAT_LITERAL,
                scalar_desc(TPrimitiveType::DOUBLE),
                0,
            )
        };
        assert_eq!(
            literal_type(&translate_ok(vec![float_node])),
            &LiteralType::Fp64(2.5)
        );
        let float32_node = TExprNode {
            float_literal: Some(TFloatLiteral {
                value: 2.5f64.into(),
            }),
            ..node(
                TExprNodeType::FLOAT_LITERAL,
                scalar_desc(TPrimitiveType::FLOAT),
                0,
            )
        };
        assert_eq!(
            literal_type(&translate_ok(vec![float32_node])),
            &LiteralType::Fp32(2.5)
        );
        assert_eq!(
            literal_type(&translate_ok(vec![string_literal("%BRASS")])),
            &LiteralType::String("%BRASS".to_string())
        );
    }

    #[test]
    fn decimal_literals_encode_unscaled_i128_and_widen_narrow_precisions() {
        let translated = translate_ok(vec![decimal_literal(
            "24.00",
            TPrimitiveType::DECIMAL64,
            15,
            2,
        )]);
        assert_eq!(
            literal_type(&translated),
            &LiteralType::Decimal(expression::literal::Decimal {
                value: 2400i128.to_le_bytes().to_vec(),
                precision: 15,
                scale: 2,
            })
        );
        // Q17's `0.2` is DECIMAL32(1,1): value kept, precision lifted off the INT16 carrier.
        let translated = translate_ok(vec![decimal_literal(
            "0.2",
            TPrimitiveType::DECIMAL32,
            1,
            1,
        )]);
        assert_eq!(
            literal_type(&translated),
            &LiteralType::Decimal(expression::literal::Decimal {
                value: 2i128.to_le_bytes().to_vec(),
                precision: 5,
                scale: 1,
            })
        );
        let translated = translate_ok(vec![decimal_literal(
            "-0.000002",
            TPrimitiveType::DECIMAL32,
            6,
            6,
        )]);
        assert_eq!(
            literal_type(&translated),
            &LiteralType::Decimal(expression::literal::Decimal {
                value: (-2i128).to_le_bytes().to_vec(),
                precision: 6,
                scale: 6,
            })
        );
        // Fewer fraction digits than the scale are padded; more are an error.
        let translated = translate_ok(vec![decimal_literal("1", TPrimitiveType::DECIMAL64, 16, 2)]);
        assert!(matches!(
            literal_type(&translated),
            LiteralType::Decimal(decimal) if decimal.value == 100i128.to_le_bytes().to_vec()
        ));
        assert!(matches!(
            translate_err(vec![decimal_literal(
                "1.234",
                TPrimitiveType::DECIMAL64,
                16,
                2
            )]),
            TranslateError::MalformedPlan(_)
        ));
        assert!(matches!(
            translate_err(vec![decimal_literal(
                "abc",
                TPrimitiveType::DECIMAL64,
                16,
                2
            )]),
            TranslateError::MalformedPlan(_)
        ));
        assert!(matches!(
            translate_err(vec![decimal_literal(
                "1.0",
                TPrimitiveType::DECIMALV2,
                27,
                9
            )]),
            TranslateError::UnsupportedType { .. }
        ));
    }

    #[test]
    fn date_literals_become_epoch_days_and_datetime_literals_a_string_cast() {
        let date_node = TExprNode {
            date_literal: Some(TDateLiteral {
                value: "1998-09-02".to_string(),
            }),
            is_nullable: Some(false),
            ..node(
                TExprNodeType::DATE_LITERAL,
                scalar_desc(TPrimitiveType::DATEV2),
                0,
            )
        };
        assert_eq!(
            literal_type(&translate_ok(vec![date_node])),
            &LiteralType::Date(10471)
        );
        let epoch = TExprNode {
            date_literal: Some(TDateLiteral {
                value: "1970-01-01".to_string(),
            }),
            ..node(
                TExprNodeType::DATE_LITERAL,
                scalar_desc(TPrimitiveType::DATEV2),
                0,
            )
        };
        assert_eq!(
            literal_type(&translate_ok(vec![epoch])),
            &LiteralType::Date(0)
        );
        let datetime_node = TExprNode {
            date_literal: Some(TDateLiteral {
                value: "1998-09-02 12:34:56".to_string(),
            }),
            ..node(
                TExprNodeType::DATE_LITERAL,
                scalar_desc(TPrimitiveType::DATETIMEV2),
                0,
            )
        };
        match translate_ok(vec![datetime_node]).rex_type.unwrap() {
            RexType::Cast(cast) => {
                assert!(matches!(
                    cast.r#type.unwrap().kind,
                    Some(r#type::Kind::PrecisionTimestamp(ts)) if ts.precision == 0
                ));
                assert_eq!(
                    literal_type(cast.input.as_ref().unwrap()),
                    &LiteralType::String("1998-09-02 12:34:56".to_string())
                );
            }
            other => panic!("{other:?}"),
        }
        let bad = TExprNode {
            date_literal: Some(TDateLiteral {
                value: "1998-13-02".to_string(),
            }),
            ..node(
                TExprNodeType::DATE_LITERAL,
                scalar_desc(TPrimitiveType::DATEV2),
                0,
            )
        };
        assert!(matches!(
            translate_err(vec![bad]),
            TranslateError::MalformedPlan(_)
        ));
        let v1 = TExprNode {
            date_literal: Some(TDateLiteral {
                value: "1998-09-02".to_string(),
            }),
            ..node(
                TExprNodeType::DATE_LITERAL,
                scalar_desc(TPrimitiveType::DATE),
                0,
            )
        };
        assert!(matches!(
            translate_err(vec![v1]),
            TranslateError::UnsupportedType { .. }
        ));
    }

    #[test]
    fn null_literals_are_cast_to_their_type() {
        let null_node = node(
            TExprNodeType::NULL_LITERAL,
            scalar_desc(TPrimitiveType::INT),
            0,
        );
        match translate_ok(vec![null_node]).rex_type.unwrap() {
            RexType::Cast(cast) => {
                assert!(matches!(
                    cast.r#type.unwrap().kind,
                    Some(r#type::Kind::I32(_))
                ));
                assert!(matches!(
                    literal_type(cast.input.as_ref().unwrap()),
                    LiteralType::Null(_)
                ));
            }
            other => panic!("{other:?}"),
        }
        // An uncast NULL (NULL_TYPE) has no type to give it (G-18).
        let untyped = node(
            TExprNodeType::NULL_LITERAL,
            scalar_desc(TPrimitiveType::NULL_TYPE),
            0,
        );
        assert!(matches!(
            translate_err(vec![untyped]),
            TranslateError::UnsupportedType { .. }
        ));
    }

    #[test]
    fn binary_predicates_map_every_comparison_opcode() {
        for (opcode, doris, substrait) in [
            (TExprOpcode::EQ, "eq", "equal"),
            (TExprOpcode::NE, "ne", "not_equal"),
            (TExprOpcode::LT, "lt", "lt"),
            (TExprOpcode::LE, "le", "lte"),
            (TExprOpcode::GT, "gt", "gt"),
            (TExprOpcode::GE, "ge", "gte"),
            (
                TExprOpcode::EQ_FOR_NULL,
                "eq_for_null",
                "is_not_distinct_from",
            ),
        ] {
            let (translated, registry) = translate(vec![
                binary_pred(opcode, doris),
                slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
                int_literal(5, TPrimitiveType::INT),
            ])
            .unwrap();
            let (anchor, args) = scalar_function_parts(&translated);
            assert_eq!(function_name_of(&registry, anchor), substrait, "{doris}");
            assert_eq!(args.len(), 2);
            assert_eq!(field_index(args[0]), 1);
        }
        let mut bad = binary_pred(TExprOpcode::EQ, "eq");
        bad.opcode = Some(TExprOpcode::MATCH_ANY);
        assert!(matches!(
            translate_err(vec![
                bad,
                slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
                int_literal(5, TPrimitiveType::INT),
            ]),
            TranslateError::UnsupportedExpression { .. }
        ));
    }

    #[test]
    fn compound_predicates_map_and_or_not() {
        let compound = |opcode, num_children| TExprNode {
            opcode: Some(opcode),
            ..node(
                TExprNodeType::COMPOUND_PRED,
                scalar_desc(TPrimitiveType::BOOLEAN),
                num_children,
            )
        };
        let pred = || {
            vec![
                binary_pred(TExprOpcode::EQ, "eq"),
                slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
                int_literal(5, TPrimitiveType::INT),
            ]
        };
        for (opcode, name) in [
            (TExprOpcode::COMPOUND_AND, "and"),
            (TExprOpcode::COMPOUND_OR, "or"),
        ] {
            let mut nodes = vec![compound(opcode, 2)];
            nodes.extend(pred());
            nodes.extend(pred());
            let (translated, registry) = translate(nodes).unwrap();
            let (anchor, args) = scalar_function_parts(&translated);
            assert_eq!(function_name_of(&registry, anchor), name);
            assert_eq!(args.len(), 2);
        }
        let mut nodes = vec![compound(TExprOpcode::COMPOUND_NOT, 1)];
        nodes.extend(pred());
        let (translated, registry) = translate(nodes).unwrap();
        let (anchor, args) = scalar_function_parts(&translated);
        assert_eq!(function_name_of(&registry, anchor), "not");
        assert_eq!(args.len(), 1);
        // A one-child AND is malformed, not silently accepted.
        let mut nodes = vec![compound(TExprOpcode::COMPOUND_AND, 1)];
        nodes.extend(pred());
        assert!(matches!(
            translate_err(nodes),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn arithmetic_uses_the_function_name_and_keeps_the_doris_result_type() {
        let (translated, registry) = translate(vec![
            arithmetic("multiply", decimal_desc(TPrimitiveType::DECIMAL128I, 30, 4)),
            slot_ref(0, 5, decimal_desc(TPrimitiveType::DECIMAL64, 15, 2)),
            decimal_literal("1.00", TPrimitiveType::DECIMAL64, 15, 2),
        ])
        .unwrap();
        match translated.rex_type.as_ref().unwrap() {
            RexType::ScalarFunction(function) => {
                assert_eq!(
                    function_name_of(&registry, function.function_reference),
                    "multiply"
                );
                assert!(matches!(
                    function.output_type.as_ref().unwrap().kind,
                    Some(r#type::Kind::Decimal(r#type::Decimal {
                        precision: 30,
                        scale: 4,
                        ..
                    }))
                ));
            }
            other => panic!("{other:?}"),
        }
        for (name, expected) in [
            ("add", "add"),
            ("subtract", "subtract"),
            ("divide", "divide"),
            ("mod", "modulus"),
        ] {
            let (translated, registry) = translate(vec![
                arithmetic(name, scalar_desc(TPrimitiveType::BIGINT)),
                slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
                int_literal(2, TPrimitiveType::BIGINT),
            ])
            .unwrap();
            let (anchor, _) = scalar_function_parts(&translated);
            assert_eq!(function_name_of(&registry, anchor), expected);
        }
        // Opcode-only encoding (older FEs) still works.
        let mut opcode_only = arithmetic("add", scalar_desc(TPrimitiveType::BIGINT));
        opcode_only.fn_ = None;
        opcode_only.opcode = Some(TExprOpcode::ADD);
        let (translated, registry) = translate(vec![
            opcode_only,
            slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            int_literal(2, TPrimitiveType::BIGINT),
        ])
        .unwrap();
        let (anchor, _) = scalar_function_parts(&translated);
        assert_eq!(function_name_of(&registry, anchor), "add");
        assert!(matches!(
            translate_err(vec![
                arithmetic("bitand", scalar_desc(TPrimitiveType::BIGINT)),
                slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
                int_literal(2, TPrimitiveType::BIGINT),
            ]),
            TranslateError::UnsupportedExpression { .. }
        ));
    }

    #[test]
    fn casts_target_the_node_type_and_try_cast_is_rejected() {
        let cast_node = TExprNode {
            opcode: Some(TExprOpcode::CAST),
            ..node(
                TExprNodeType::CAST_EXPR,
                decimal_desc(TPrimitiveType::DECIMAL64, 16, 2),
                1,
            )
        };
        match translate_ok(vec![
            cast_node.clone(),
            slot_ref(0, 5, decimal_desc(TPrimitiveType::DECIMAL64, 15, 2)),
        ])
        .rex_type
        .unwrap()
        {
            RexType::Cast(cast) => {
                assert!(matches!(
                    cast.r#type.unwrap().kind,
                    Some(r#type::Kind::Decimal(r#type::Decimal {
                        precision: 16,
                        scale: 2,
                        ..
                    }))
                ));
                assert_eq!(
                    cast.failure_behavior,
                    expression::cast::FailureBehavior::ThrowException as i32
                );
                assert_eq!(field_index(cast.input.as_ref().unwrap()), 4);
            }
            other => panic!("{other:?}"),
        }
        let mut try_cast = cast_node;
        try_cast.opcode = Some(TExprOpcode::TRY_CAST);
        assert!(matches!(
            translate_err(vec![
                try_cast,
                slot_ref(0, 5, decimal_desc(TPrimitiveType::DECIMAL64, 15, 2)),
            ]),
            TranslateError::UnsupportedExpression { .. }
        ));
    }

    #[test]
    fn in_predicates_become_singular_or_list_and_not_in_is_negated() {
        let in_pred = |is_not_in| TExprNode {
            opcode: Some(if is_not_in {
                TExprOpcode::FILTER_NOT_IN
            } else {
                TExprOpcode::FILTER_IN
            }),
            in_predicate: Some(TInPredicate { is_not_in }),
            ..node(
                TExprNodeType::IN_PRED,
                scalar_desc(TPrimitiveType::BOOLEAN),
                3,
            )
        };
        let translated = translate_ok(vec![
            in_pred(false),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            string_literal("FRANCE"),
            string_literal("GERMANY"),
        ]);
        match translated.rex_type.unwrap() {
            RexType::SingularOrList(list) => {
                assert_eq!(field_index(list.value.as_ref().unwrap()), 2);
                assert_eq!(list.options.len(), 2);
            }
            other => panic!("{other:?}"),
        }
        let (translated, registry) = translate(vec![
            in_pred(true),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            string_literal("FRANCE"),
            string_literal("GERMANY"),
        ])
        .unwrap();
        let (anchor, args) = scalar_function_parts(&translated);
        assert_eq!(function_name_of(&registry, anchor), "not");
        assert!(matches!(
            args[0].rex_type.as_ref().unwrap(),
            RexType::SingularOrList(_)
        ));
        let mut lonely = in_pred(false);
        lonely.num_children = 1;
        assert!(matches!(
            translate_err(vec![
                lonely,
                slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING))
            ]),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn is_null_predicates() {
        for (is_not_null, name) in [(false, "is_null"), (true, "is_not_null")] {
            let pred = TExprNode {
                is_null_pred: Some(TIsNullPredicate { is_not_null }),
                ..node(
                    TExprNodeType::IS_NULL_PRED,
                    scalar_desc(TPrimitiveType::BOOLEAN),
                    1,
                )
            };
            let (translated, registry) = translate(vec![
                pred,
                slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            ])
            .unwrap();
            let (anchor, args) = scalar_function_parts(&translated);
            assert_eq!(function_name_of(&registry, anchor), name);
            assert_eq!(args.len(), 1);
        }
    }

    #[test]
    fn case_expressions_become_if_then_chains() {
        let case = |has_else| TExprNode {
            case_expr: Some(TCaseExpr {
                has_case_expr: false,
                has_else_expr: has_else,
            }),
            ..node(
                TExprNodeType::CASE_EXPR,
                scalar_desc(TPrimitiveType::INT),
                if has_else { 3 } else { 2 },
            )
        };
        let when = || {
            vec![
                binary_pred(TExprOpcode::EQ, "eq"),
                slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
                int_literal(5, TPrimitiveType::INT),
            ]
        };
        let mut nodes = vec![case(true)];
        nodes.extend(when());
        nodes.push(int_literal(1, TPrimitiveType::INT));
        nodes.push(int_literal(0, TPrimitiveType::INT));
        match translate_ok(nodes).rex_type.unwrap() {
            RexType::IfThen(if_then) => {
                assert_eq!(if_then.ifs.len(), 1);
                assert_eq!(
                    literal_type(if_then.r#else.as_ref().unwrap()),
                    &LiteralType::I32(0)
                );
            }
            other => panic!("{other:?}"),
        }
        // No ELSE: a typed NULL else branch is synthesized.
        let mut nodes = vec![case(false)];
        nodes.extend(when());
        nodes.push(int_literal(1, TPrimitiveType::INT));
        match translate_ok(nodes).rex_type.unwrap() {
            RexType::IfThen(if_then) => {
                assert!(matches!(
                    if_then.r#else.as_ref().unwrap().rex_type.as_ref().unwrap(),
                    RexType::Cast(_)
                ));
            }
            other => panic!("{other:?}"),
        }
        let mut with_case_operand = case(true);
        with_case_operand.case_expr.as_mut().unwrap().has_case_expr = true;
        let mut nodes = vec![with_case_operand];
        nodes.extend(when());
        nodes.push(int_literal(1, TPrimitiveType::INT));
        nodes.push(int_literal(0, TPrimitiveType::INT));
        assert!(matches!(
            translate_err(nodes),
            TranslateError::UnsupportedExpression { .. }
        ));
    }

    #[test]
    fn if_function_becomes_if_then() {
        let nodes = vec![
            function_call("if", scalar_desc(TPrimitiveType::TINYINT), 3),
            binary_pred(TExprOpcode::EQ, "eq"),
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
            int_literal(5, TPrimitiveType::INT),
            int_literal(1, TPrimitiveType::TINYINT),
            int_literal(0, TPrimitiveType::TINYINT),
        ];
        match translate_ok(nodes).rex_type.unwrap() {
            RexType::IfThen(if_then) => {
                assert_eq!(if_then.ifs.len(), 1);
                assert_eq!(
                    literal_type(if_then.ifs[0].then.as_ref().unwrap()),
                    &LiteralType::I8(1)
                );
                assert_eq!(
                    literal_type(if_then.r#else.as_ref().unwrap()),
                    &LiteralType::I8(0)
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn like_requires_a_constant_pattern_without_backslash() {
        let (translated, registry) = translate(vec![
            function_call("like", scalar_desc(TPrimitiveType::BOOLEAN), 2),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            string_literal("%green%"),
        ])
        .unwrap();
        let (anchor, _) = scalar_function_parts(&translated);
        assert_eq!(function_name_of(&registry, anchor), "like");
        // G-03: an escape in the pattern.
        let err = translate_err(vec![
            function_call("like", scalar_desc(TPrimitiveType::BOOLEAN), 2),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            string_literal("100\\%"),
        ]);
        assert!(
            matches!(err, TranslateError::UnsupportedExpression { reason, .. } if reason.contains("G-03"))
        );
        // G-03: a non-constant pattern.
        let err = translate_err(vec![
            function_call("like", scalar_desc(TPrimitiveType::BOOLEAN), 2),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
        ]);
        assert!(matches!(err, TranslateError::UnsupportedExpression { .. }));
    }

    #[test]
    fn substring_requires_constant_positive_bounds() {
        let (translated, registry) = translate(vec![
            function_call("substring", varchar_desc(2), 3),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            int_literal(1, TPrimitiveType::INT),
            int_literal(2, TPrimitiveType::INT),
        ])
        .unwrap();
        let (anchor, args) = scalar_function_parts(&translated);
        assert_eq!(function_name_of(&registry, anchor), "substring");
        assert_eq!(args.len(), 3);
        for bad_start in [0, -1] {
            let err = translate_err(vec![
                function_call("substr", varchar_desc(2), 3),
                slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
                int_literal(bad_start, TPrimitiveType::INT),
                int_literal(2, TPrimitiveType::INT),
            ]);
            assert!(
                matches!(err, TranslateError::UnsupportedExpression { reason, .. } if reason.contains("G-04"))
            );
        }
        let err = translate_err(vec![
            function_call("substring", varchar_desc(2), 3),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
            int_literal(2, TPrimitiveType::INT),
        ]);
        assert!(matches!(err, TranslateError::UnsupportedExpression { .. }));
    }

    #[test]
    fn date_part_and_length_functions_map_by_name() {
        for (doris, substrait) in [
            ("year", "year"),
            ("month", "month"),
            ("day", "day"),
            ("length", "octet_length"),
            ("char_length", "char_length"),
            ("is_null_pred", "is_null"),
            ("is_not_null_pred", "is_not_null"),
        ] {
            let (translated, registry) = translate(vec![
                function_call(doris, scalar_desc(TPrimitiveType::INT), 1),
                slot_ref(0, 4, scalar_desc(TPrimitiveType::DATEV2)),
            ])
            .unwrap();
            let (anchor, _) = scalar_function_parts(&translated);
            assert_eq!(function_name_of(&registry, anchor), substrait, "{doris}");
        }
    }

    #[test]
    fn concat_and_unlisted_functions_are_rejected_by_name() {
        let err = translate_err(vec![
            function_call("concat", scalar_desc(TPrimitiveType::STRING), 2),
            slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            string_literal("x"),
        ]);
        assert!(
            matches!(err, TranslateError::UnsupportedExpression { reason, .. } if reason.contains("G-02"))
        );
        for name in [
            "upper",
            "lower",
            "trim",
            "round",
            "abs",
            "replace",
            "date_format",
        ] {
            let err = translate_err(vec![
                function_call(name, scalar_desc(TPrimitiveType::STRING), 1),
                slot_ref(0, 3, scalar_desc(TPrimitiveType::STRING)),
            ]);
            assert!(
                matches!(&err, TranslateError::UnsupportedFunction { name: rejected, .. } if rejected == name),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn unsupported_node_types_are_named() {
        for node_type in [
            TExprNodeType::TUPLE_IS_NULL_PRED,
            TExprNodeType::NULL_AWARE_IN_PRED,
            TExprNodeType::LAMBDA_FUNCTION_EXPR,
            TExprNodeType::ARRAY_LITERAL,
            TExprNodeType::PREDICATE,
            TExprNodeType::LITERAL,
        ] {
            let err = translate_err(vec![node(
                node_type,
                scalar_desc(TPrimitiveType::BOOLEAN),
                0,
            )]);
            assert!(
                matches!(err, TranslateError::UnsupportedExpression { node_type: rejected, .. } if rejected == node_type),
                "{node_type:?}"
            );
        }
        let err = translate_err(vec![
            TExprNode {
                agg_expr: Some(TAggregateExpr {
                    is_merge_agg: false,
                    param_types: None,
                }),
                fn_: Some(function("sum", scalar_desc(TPrimitiveType::BIGINT))),
                ..node(
                    TExprNodeType::AGG_EXPR,
                    scalar_desc(TPrimitiveType::BIGINT),
                    1,
                )
            },
            slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
        ]);
        assert!(matches!(err, TranslateError::UnsupportedExpression { .. }));
    }

    #[test]
    fn malformed_preorder_lists_are_rejected() {
        // Under-run: a binary predicate with one child.
        let err = translate_err(vec![
            binary_pred(TExprOpcode::EQ, "eq"),
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
        ]);
        assert!(matches!(err, TranslateError::MalformedPlan(_)));
        // Trailing node.
        let err = translate_err(vec![
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
            int_literal(5, TPrimitiveType::INT),
        ]);
        assert!(matches!(err, TranslateError::MalformedPlan(_)));
        // Negative child count.
        let mut negative = slot_ref(0, 2, scalar_desc(TPrimitiveType::INT));
        negative.num_children = -1;
        assert!(matches!(
            translate_err(vec![negative]),
            TranslateError::MalformedPlan(_)
        ));
        assert!(matches!(
            translate_err(vec![]),
            TranslateError::MalformedPlan(_)
        ));
    }

    #[test]
    fn extension_anchors_are_shared_per_function() {
        let (_, registry) = translate(vec![
            {
                let mut and = node(
                    TExprNodeType::COMPOUND_PRED,
                    scalar_desc(TPrimitiveType::BOOLEAN),
                    2,
                );
                and.opcode = Some(TExprOpcode::COMPOUND_AND);
                and
            },
            binary_pred(TExprOpcode::EQ, "eq"),
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
            int_literal(5, TPrimitiveType::INT),
            binary_pred(TExprOpcode::EQ, "eq"),
            slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            int_literal(6, TPrimitiveType::BIGINT),
        ])
        .unwrap();
        let (urns, functions) = registry.into_extensions();
        assert_eq!(urns.len(), 2);
        assert_eq!(functions.len(), 2);
    }

    fn agg_root(name: &str, is_merge: bool, num_children: i32, ret: TTypeDesc) -> TExprNode {
        TExprNode {
            agg_expr: Some(TAggregateExpr {
                is_merge_agg: is_merge,
                param_types: None,
            }),
            fn_: Some(function(name, ret.clone())),
            ..node(TExprNodeType::AGG_EXPR, ret, num_children)
        }
    }

    fn aggregate(nodes: Vec<TExprNode>) -> Result<AggregateCall> {
        let desc = desc();
        let mut registry = ExtensionRegistry::new();
        let row_tuples = [0];
        let mut ctx = ExprContext::new(&desc, &mut registry, &row_tuples);
        aggregate_call(&expr(nodes), &mut ctx)
    }

    #[test]
    fn aggregate_calls_decompose_name_arguments_and_distinctness() {
        let call = aggregate(vec![
            agg_root(
                "sum",
                false,
                1,
                decimal_desc(TPrimitiveType::DECIMAL128I, 38, 2),
            ),
            slot_ref(0, 5, decimal_desc(TPrimitiveType::DECIMAL64, 15, 2)),
        ])
        .unwrap();
        assert_eq!(call.name, "sum");
        assert_eq!(call.arguments.len(), 1);
        assert_eq!(field_index(&call.arguments[0]), 4);
        assert!(!call.distinct);
        assert!(!call.is_merge);
        assert!(matches!(
            call.return_type.kind,
            Some(r#type::Kind::Decimal(r#type::Decimal {
                precision: 38,
                scale: 2,
                ..
            }))
        ));

        let call = aggregate(vec![agg_root(
            "count",
            false,
            0,
            scalar_desc(TPrimitiveType::BIGINT),
        )])
        .unwrap();
        assert_eq!((call.name, call.arguments.len()), ("count", 0));

        let call = aggregate(vec![
            agg_root(
                "multi_distinct_count",
                false,
                1,
                scalar_desc(TPrimitiveType::BIGINT),
            ),
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
        ])
        .unwrap();
        assert_eq!((call.name, call.distinct), ("count", true));

        for name in ["min", "max", "avg"] {
            let call = aggregate(vec![
                agg_root(name, false, 1, scalar_desc(TPrimitiveType::BIGINT)),
                slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            ])
            .unwrap();
            assert_eq!(call.name, name);
        }
    }

    #[test]
    fn merge_phase_aggregates_are_flagged_not_rejected() {
        let call = aggregate(vec![
            agg_root(
                "sum",
                true,
                1,
                decimal_desc(TPrimitiveType::DECIMAL128I, 38, 2),
            ),
            slot_ref(0, 3, varchar_desc(65533)),
        ])
        .unwrap();
        assert!(call.is_merge);
        assert_eq!(call.name, "sum");
    }

    #[test]
    fn aggregate_gates() {
        // G-16: an aggregate outside the allowlist.
        for name in [
            "stddev",
            "median",
            "approx_count_distinct",
            "multi_distinct_sum",
            "group_concat",
        ] {
            let err = aggregate(vec![
                agg_root(name, false, 1, scalar_desc(TPrimitiveType::BIGINT)),
                slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            ])
            .unwrap_err();
            assert!(
                matches!(&err, TranslateError::UnsupportedFunction { name: rejected, .. } if rejected == name),
                "{name}: {err}"
            );
        }
        // Distinct over two columns.
        let err = aggregate(vec![
            agg_root(
                "multi_distinct_count",
                false,
                2,
                scalar_desc(TPrimitiveType::BIGINT),
            ),
            slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
        ])
        .unwrap_err();
        assert!(matches!(err, TranslateError::UnsupportedExpression { .. }));
        // Not an AGG_EXPR root.
        let err = aggregate(vec![slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT))]).unwrap_err();
        assert!(matches!(err, TranslateError::UnsupportedExpression { .. }));
        // Trailing nodes after the arguments.
        let err = aggregate(vec![
            agg_root("sum", false, 1, scalar_desc(TPrimitiveType::BIGINT)),
            slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
            slot_ref(0, 1, scalar_desc(TPrimitiveType::BIGINT)),
        ])
        .unwrap_err();
        assert!(matches!(err, TranslateError::MalformedPlan(_)));
    }

    #[test]
    fn aggregate_function_invocation_carries_distinct_and_anchor() {
        let desc = desc();
        let mut registry = ExtensionRegistry::new();
        let row_tuples = [0];
        let mut ctx = ExprContext::new(&desc, &mut registry, &row_tuples);
        let call = aggregate_call(
            &expr(vec![
                agg_root(
                    "multi_distinct_count",
                    false,
                    1,
                    scalar_desc(TPrimitiveType::BIGINT),
                ),
                slot_ref(0, 2, scalar_desc(TPrimitiveType::INT)),
            ]),
            &mut ctx,
        )
        .unwrap();
        let function = aggregate_function(&call, &mut ctx);
        assert_eq!(
            function.invocation,
            substrait::proto::aggregate_function::AggregationInvocation::Distinct as i32
        );
        assert_eq!(function.arguments.len(), 1);
        assert_eq!(
            registry.function_name(function.function_reference),
            Some("count")
        );
    }

    #[test]
    fn epoch_days_match_known_dates() {
        assert_eq!(epoch_days_from_date_str("1970-01-01").unwrap(), 0);
        assert_eq!(epoch_days_from_date_str("1970-01-02").unwrap(), 1);
        assert_eq!(epoch_days_from_date_str("1969-12-31").unwrap(), -1);
        assert_eq!(epoch_days_from_date_str("2000-02-29").unwrap(), 11016);
        assert_eq!(
            epoch_days_from_date_str("1998-09-02 00:00:00").unwrap(),
            10471
        );
        assert!(epoch_days_from_date_str("1998-09").is_err());
        assert!(epoch_days_from_date_str("1998-09-02-01").is_err());
        assert!(epoch_days_from_date_str("").is_err());
    }
}
