use starrocks_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::types::TPrimitiveType;
use substrait::proto::expression::field_reference;
use substrait::proto::expression::reference_segment;
use substrait::proto::expression::{self, FieldReference, ReferenceSegment};
use substrait::proto::{Expression, FunctionArgument, Type, function_argument};

use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::type_mapper;
use crate::{
    ExtensionRegistry, URN_ARITHMETIC, URN_BOOLEAN, URN_COMPARISON, URN_DATETIME, URN_STRING,
};

/// Mutable state needed while translating one StarRocks expression tree.
pub(crate) struct ExprContext<'a> {
    /// Descriptor lookups for slot references and row layouts.
    desc: &'a DescriptorTable,
    /// Substrait extension registry shared with the enclosing plan translation.
    registry: &'a mut ExtensionRegistry,
    /// Tuple ids that describe the input row visible to this expression.
    row_tuples: &'a [i32],
}

impl<'a> ExprContext<'a> {
    /// Creates an expression context for a specific input row layout.
    pub(crate) fn new(
        desc: &'a DescriptorTable,
        registry: &'a mut ExtensionRegistry,
        row_tuples: &'a [i32],
    ) -> Self {
        Self {
            desc,
            registry,
            row_tuples,
        }
    }
}

/// Trait implemented by StarRocks expression objects that can become Substrait expressions.
pub(crate) trait TranslateExpr {
    /// Translates the receiver into a Substrait expression.
    fn translate(&self, ctx: &mut ExprContext<'_>) -> Result<Expression>;
}

impl TranslateExpr for TExpr {
    /// Translates StarRocks' flat preorder expression encoding.
    fn translate(&self, ctx: &mut ExprContext<'_>) -> Result<Expression> {
        if self.nodes.is_empty() {
            return Err(TranslateError::malformed("TExpr.nodes is empty"));
        }
        let mut cursor = ExprNodeCursor::new(&self.nodes);
        let translated = cursor.translate_next(ctx)?;
        cursor.ensure_consumed()?;
        Ok(translated)
    }
}

/// Cursor over StarRocks' flat preorder `TExpr.nodes` representation.
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

    /// Translates the next preorder node and its subtree.
    fn translate_next(&mut self, ctx: &mut ExprContext<'_>) -> Result<Expression> {
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

        let children = (0..node.num_children)
            .map(|_| self.translate_next(ctx))
            .collect::<Result<Vec<_>>>()?;

        translate_expr_node(node, children, ctx)
    }

    /// Verifies that the top-level expression consumed all encoded nodes.
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

/// Routes a StarRocks expression node to its supported v1 translator once its
/// child expressions have been translated.
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
        TExprNodeType::NULL_LITERAL => translate_null_literal(node, children),
        TExprNodeType::DECIMAL_LITERAL => translate_decimal_literal(node, children),
        TExprNodeType::DATE_LITERAL => translate_date_literal(node, children),
        TExprNodeType::BINARY_PRED => translate_binary_pred(node, children, ctx),
        TExprNodeType::COMPOUND_PRED => translate_compound_pred(node, children, ctx),
        TExprNodeType::CAST_EXPR => translate_cast(node, children),
        TExprNodeType::IS_NULL_PRED => translate_is_null(node, children, ctx),
        TExprNodeType::ARITHMETIC_EXPR => translate_arithmetic(node, children, ctx),
        TExprNodeType::IN_PRED => translate_in_pred(node, children, ctx),
        TExprNodeType::CASE_EXPR => translate_case(node, children),
        TExprNodeType::FUNCTION_CALL => translate_function_call(node, children, ctx),
        _ => Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "expression node is outside the v1 StarRocks slice",
        }),
    }
}

/// Converts a StarRocks `SLOT_REF` into a Substrait field selection.
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
    let field =
        ctx.desc
            .slot_global_index(slot_ref.tuple_id, slot_ref.slot_id, ctx.row_tuples)? as i32;
    Ok(Expression {
        rex_type: Some(expression::RexType::Selection(Box::new(FieldReference {
            reference_type: Some(field_reference::ReferenceType::DirectReference(
                ReferenceSegment {
                    reference_type: Some(reference_segment::ReferenceType::StructField(Box::new(
                        reference_segment::StructField { field, child: None },
                    ))),
                },
            )),
            root_type: Some(field_reference::RootType::RootReference(
                field_reference::RootReference {},
            )),
        }))),
    })
}

/// Converts a StarRocks `BOOL_LITERAL` into a Substrait literal.
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

/// Converts a StarRocks `INT_LITERAL` into a width-matched Substrait literal.
fn translate_int_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .int_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "INT_LITERAL",
            field: "int_literal",
        })?;
    Ok(literal(integer_literal_type(
        lit.value,
        type_mapper::scalar_primitive(&node.type_)?,
    )?))
}

/// Converts a StarRocks `FLOAT_LITERAL` into a width-matched Substrait literal.
fn translate_float_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .float_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "FLOAT_LITERAL",
            field: "float_literal",
        })?;
    let value = *lit.value;
    let literal_type = match type_mapper::scalar_primitive(&node.type_)? {
        TPrimitiveType::FLOAT => expression::literal::LiteralType::Fp32(value as f32),
        TPrimitiveType::DOUBLE => expression::literal::LiteralType::Fp64(value),
        primitive => {
            return Err(TranslateError::UnsupportedType {
                primitive: Some(primitive),
                node_type: Some(starrocks_thrift::types::TTypeNodeType::SCALAR),
                reason: "FLOAT_LITERAL has non-floating scalar type",
            });
        }
    };
    Ok(literal(literal_type))
}

/// Converts a StarRocks `STRING_LITERAL` into a Substrait string literal.
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

/// Converts a StarRocks `NULL_LITERAL` into a typed Substrait null literal.
fn translate_null_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let null_type = type_mapper::map_type_desc(&node.type_, true)?;
    Ok(literal(expression::literal::LiteralType::Null(null_type)))
}

/// Converts a StarRocks `DECIMAL_LITERAL` into Substrait's i128 encoding.
fn translate_decimal_literal(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 0)?;
    let lit = node
        .decimal_literal
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "DECIMAL_LITERAL",
            field: "decimal_literal",
        })?;
    let decimal_type = type_mapper::map_type_desc(&node.type_, true)?;
    let Some(substrait::proto::r#type::Kind::Decimal(decimal)) = decimal_type.kind else {
        return Err(TranslateError::malformed(
            "DECIMAL_LITERAL has non-decimal type",
        ));
    };
    let value = encode_decimal(&lit.value, decimal.scale)?;
    Ok(literal(expression::literal::LiteralType::Decimal(
        expression::literal::Decimal {
            value: value.to_vec(),
            precision: decimal.precision,
            scale: decimal.scale,
        },
    )))
}

/// Converts a StarRocks `DATE_LITERAL` into a Substrait date literal.
///
/// StarRocks carries the literal as a `YYYY-MM-DD[ HH:MM:SS]` string; Substrait dates are days
/// since the UNIX epoch. Only DATE-typed literals are supported — DATETIME literals would need a
/// precision-timestamp literal the consumer side has not been exercised with.
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
        TPrimitiveType::DATE => Ok(literal(expression::literal::LiteralType::Date(
            epoch_days_from_date_str(&lit.value)?,
        ))),
        primitive => Err(TranslateError::UnsupportedType {
            primitive: Some(primitive),
            node_type: Some(starrocks_thrift::types::TTypeNodeType::SCALAR),
            reason: "only DATE-typed date literals are supported",
        }),
    }
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
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(invalid());
    }
    // Howard Hinnant's civil-days algorithm; no chrono dependency needed for whole dates.
    let year_adjusted = if month <= 2 { year - 1 } else { year };
    let era = year_adjusted.div_euclid(400);
    let year_of_era = year_adjusted - era * 400;
    let month_shifted = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_shifted + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146097 + day_of_era - 719468;
    i32::try_from(days).map_err(|_| invalid())
}

/// Converts a StarRocks `ARITHMETIC_EXPR` into a Substrait arithmetic function.
fn translate_arithmetic(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    let opcode = node.opcode.ok_or(TranslateError::MissingField {
        context: "ARITHMETIC_EXPR",
        field: "opcode",
    })?;
    let name = match opcode {
        TExprOpcode::ADD => "add",
        TExprOpcode::SUBTRACT => "subtract",
        TExprOpcode::MULTIPLY => "multiply",
        TExprOpcode::DIVIDE => "divide",
        TExprOpcode::MOD => "modulus",
        _ => {
            return Err(TranslateError::UnsupportedExpression {
                node_type: node.node_type,
                reason: "arithmetic opcode is unsupported",
            });
        }
    };
    expect_child_count(node, &children, 2)?;
    // Decimal-typed arithmetic is rejected for now: the StarRocks-shaped cast/width combination
    // reliably segfaults the engine's GPU projection (see the starrocks tpch crash repro);
    // fail with a structured error instead of taking down the compute node.
    if is_decimal(&node.type_)? {
        return Err(TranslateError::UnsupportedExpression {
            node_type: node.node_type,
            reason: "decimal arithmetic is not supported yet (crashes the engine projection)",
        });
    }
    let output_type = type_mapper::map_type_desc(&node.type_, node.is_nullable.unwrap_or(true))?;
    let anchor = ctx.registry.register_function(URN_ARITHMETIC, name);
    Ok(scalar_function(anchor, children, output_type))
}

/// Returns whether a StarRocks type descriptor is any decimal flavour.
fn is_decimal(type_desc: &starrocks_thrift::types::TTypeDesc) -> Result<bool> {
    Ok(matches!(
        type_mapper::scalar_primitive(type_desc)?,
        TPrimitiveType::DECIMAL
            | TPrimitiveType::DECIMALV2
            | TPrimitiveType::DECIMAL32
            | TPrimitiveType::DECIMAL64
            | TPrimitiveType::DECIMAL128
            | TPrimitiveType::DECIMAL256
    ))
}

/// Converts a StarRocks `IN_PRED` into a Substrait singular-or-list expression.
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
        Ok(scalar_function(
            anchor,
            vec![in_list],
            type_mapper::bool_type(),
        ))
    } else {
        Ok(in_list)
    }
}

/// Converts a StarRocks `CASE_EXPR` into a Substrait if-then expression chain.
///
/// StarRocks children are `[case?] (when then)* [else?]`, flagged by
/// `TCaseExpr::has_case_expr`/`has_else_expr`. A leading case operand is not supported yet — the
/// frontend normally rewrites `CASE x WHEN ...` into comparisons already.
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
    let mut r#else = if case.has_else_expr {
        Some(Box::new(children.next_back().ok_or_else(|| {
            TranslateError::malformed("CASE_EXPR missing else child")
        })?))
    } else {
        None
    };
    let mut ifs = Vec::new();
    loop {
        let Some(condition) = children.next() else {
            break;
        };
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
    if r#else.is_none() {
        // Substrait if-then requires an else branch; SQL CASE defaults to NULL.
        r#else = Some(Box::new(literal(expression::literal::LiteralType::Null(
            type_mapper::map_type_desc(&node.type_, true)?,
        ))));
    }
    Ok(Expression {
        rex_type: Some(expression::RexType::IfThen(Box::new(expression::IfThen {
            ifs,
            r#else,
        }))),
    })
}

/// Converts a StarRocks `FUNCTION_CALL` into a Substrait expression.
///
/// Functions are allowlisted so an unknown StarRocks builtin fails loudly instead of silently
/// binding to a DuckDB function with different semantics.
fn translate_function_call(
    node: &TExprNode,
    children: Vec<Expression>,
    ctx: &mut ExprContext<'_>,
) -> Result<Expression> {
    let function = node.fn_.as_ref().ok_or(TranslateError::MissingField {
        context: "FUNCTION_CALL",
        field: "fn",
    })?;
    let name = function.name.function_name.as_str();
    let output_type = type_mapper::map_type_desc(&node.type_, node.is_nullable.unwrap_or(true))?;

    // Null checks the frontend planned as builtin calls rather than IS_NULL_PRED nodes.
    let (urn, mapped) = match name {
        "is_null_pred" => {
            expect_child_count(node, &children, 1)?;
            (URN_COMPARISON, "is_null")
        }
        "is_not_null_pred" => {
            expect_child_count(node, &children, 1)?;
            (URN_COMPARISON, "is_not_null")
        }
        "if" => {
            expect_child_count(node, &children, 3)?;
            let mut children = children.into_iter();
            let condition = children.next().unwrap();
            let then = children.next().unwrap();
            let otherwise = children.next().unwrap();
            return Ok(Expression {
                rex_type: Some(expression::RexType::IfThen(Box::new(expression::IfThen {
                    ifs: vec![expression::if_then::IfClause {
                        r#if: Some(condition),
                        then: Some(then),
                    }],
                    r#else: Some(Box::new(otherwise)),
                }))),
            });
        }
        "like" => {
            // The GPU evaluator needs a constant pattern and applies no escape character,
            // while StarRocks treats backslash as the default escape.
            expect_child_count(node, &children, 2)?;
            match string_literal_value(&children[1]) {
                Some(pattern) if !pattern.contains('\\') => {}
                _ => {
                    return Err(TranslateError::UnsupportedExpression {
                        node_type: node.node_type,
                        reason: "LIKE requires a constant pattern without escapes",
                    });
                }
            }
            (URN_STRING, "like")
        }
        "substring" | "substr" => {
            // The GPU evaluator supports exactly `substring(col, start, length)` with constant,
            // positive bounds; two-argument or from-the-end forms would misexecute.
            expect_child_count(node, &children, 3)?;
            let constant_positive = |expr: &Expression| {
                matches!(
                    integer_literal_value(expr),
                    Some(value) if value > 0
                )
            };
            if !constant_positive(&children[1]) || !constant_positive(&children[2]) {
                return Err(TranslateError::UnsupportedExpression {
                    node_type: node.node_type,
                    reason: "substring requires constant positive start and length",
                });
            }
            (URN_STRING, "substring")
        }
        // StarRocks `length` counts bytes; `octet_length` remaps to DuckDB's byte-length
        // (`strlen`), while `char_length` keeps codepoint semantics on both sides.
        "length" => (URN_STRING, "octet_length"),
        "char_length" => (URN_STRING, "char_length"),
        // `concat` is intentionally absent: StarRocks concat is NULL-strict while DuckDB's
        // ignores NULL arguments, so a name-level mapping would change results. Other string
        // and math builtins (upper/lower/trim/abs/floor/...) are absent because the GPU
        // expression evaluator has no implementations for them yet.
        "year" | "month" | "day" => (URN_DATETIME, name),
        _ => {
            return Err(TranslateError::malformed(format!(
                "unsupported StarRocks function call {name:?}"
            )));
        }
    };
    let anchor = ctx.registry.register_function(urn, mapped);
    Ok(scalar_function(anchor, children, output_type))
}

/// Returns a Substrait expression's string-literal payload, if it is one.
fn string_literal_value(expr: &Expression) -> Option<&str> {
    match expr.rex_type.as_ref()? {
        expression::RexType::Literal(literal) => match literal.literal_type.as_ref()? {
            expression::literal::LiteralType::String(value) => Some(value),
            expression::literal::LiteralType::FixedChar(value) => Some(value),
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

/// A StarRocks aggregate call decomposed for Substrait `AggregateRel` measures.
pub(crate) struct AggregateCall {
    /// Substrait/DuckDB aggregate function name.
    pub name: String,
    /// Translated argument expressions over the aggregation input row.
    pub arguments: Vec<Expression>,
    /// Whether the aggregate applies to distinct inputs.
    pub distinct: bool,
}

/// Decomposes a StarRocks aggregate-function expression (the root of a
/// `TAggregationNode::aggregate_functions` entry) into name, arguments, and distinct-ness.
pub(crate) fn aggregate_call(expr: &TExpr, ctx: &mut ExprContext<'_>) -> Result<AggregateCall> {
    let root = expr
        .nodes
        .first()
        .ok_or_else(|| TranslateError::malformed("aggregate function TExpr is empty"))?;
    let is_aggregate_root = matches!(
        root.node_type,
        TExprNodeType::AGG_EXPR | TExprNodeType::FUNCTION_CALL
    ) && root.agg_expr.is_some();
    if !is_aggregate_root {
        return Err(TranslateError::UnsupportedExpression {
            node_type: root.node_type,
            reason: "aggregate function root is not an aggregate expression",
        });
    }
    // One-phase aggregation only: a merge aggregate consumes partial states this translator
    // does not model (run with `new_planner_agg_stage = 1`).
    if root
        .agg_expr
        .as_ref()
        .is_some_and(|agg_expr| agg_expr.is_merge_agg)
    {
        return Err(TranslateError::UnsupportedExpression {
            node_type: root.node_type,
            reason: "merge-phase aggregate functions are not supported (one-phase only)",
        });
    }
    let function = root.fn_.as_ref().ok_or(TranslateError::MissingField {
        context: "aggregate expression",
        field: "fn",
    })?;
    // `multi_distinct_sum` is intentionally absent: the GPU grouped-aggregate path only
    // honors DISTINCT for count and would silently overcount a distinct sum.
    let (name, distinct) = match function.name.function_name.as_str() {
        name @ ("sum" | "count" | "min" | "max" | "avg") => (name, false),
        "multi_distinct_count" => ("count", true),
        name => {
            return Err(TranslateError::malformed(format!(
                "unsupported StarRocks aggregate function {name:?}"
            )));
        }
    };
    // Multi-column COUNT(DISTINCT a, b) needs key packing the executor does not do here.
    if distinct && root.num_children != 1 {
        return Err(TranslateError::UnsupportedExpression {
            node_type: root.node_type,
            reason: "distinct aggregates over multiple columns are not supported",
        });
    }
    // Only double-returning `avg` translates faithfully. StarRocks `avg` over decimals returns
    // a decimal (DuckDB computes a double and the consumer ignores the declared output type),
    // and temporal `avg` has StarRocks-specific day-rounding semantics.
    if name == "avg" && type_mapper::scalar_primitive(&function.ret_type)? != TPrimitiveType::DOUBLE
    {
        return Err(TranslateError::UnsupportedExpression {
            node_type: root.node_type,
            reason: "only avg with a DOUBLE result is supported (decimal/temporal avg differ)",
        });
    }

    let mut cursor = ExprNodeCursor::new(&expr.nodes);
    // Consume the root marker; its children are the aggregate arguments.
    cursor.idx = 1;
    let arguments = (0..root.num_children)
        .map(|_| cursor.translate_next(ctx))
        .collect::<Result<Vec<_>>>()?;
    cursor.ensure_consumed()?;

    Ok(AggregateCall {
        name: name.to_string(),
        arguments,
        distinct,
    })
}

/// Converts supported comparison opcodes into Substrait comparison functions.
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
        _ => {
            return Err(TranslateError::UnsupportedExpression {
                node_type: node.node_type,
                reason: "binary predicate opcode is unsupported",
            });
        }
    };
    let anchor = ctx.registry.register_function(URN_COMPARISON, name);
    Ok(scalar_function(anchor, children, type_mapper::bool_type()))
}

/// Converts supported boolean opcodes into Substrait boolean functions.
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
        TExprOpcode::COMPOUND_AND => {
            if children.len() < 2 {
                return Err(TranslateError::malformed(
                    "COMPOUND_AND expected at least 2 children",
                ));
            }
            "and"
        }
        TExprOpcode::COMPOUND_OR => {
            if children.len() < 2 {
                return Err(TranslateError::malformed(
                    "COMPOUND_OR expected at least 2 children",
                ));
            }
            "or"
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
    Ok(scalar_function(anchor, children, type_mapper::bool_type()))
}

/// Converts a StarRocks cast into a Substrait cast with throwing failure behavior.
fn translate_cast(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    expect_child_count(node, &children, 1)?;
    // Honor StarRocks' declared nullability for the cast target, matching the
    // slot path; default to nullable when the analyzer left it unset.
    let nullable = node.is_nullable.unwrap_or(true);
    Ok(Expression {
        rex_type: Some(expression::RexType::Cast(Box::new(expression::Cast {
            r#type: Some(type_mapper::map_type_desc(&node.type_, nullable)?),
            input: Some(Box::new(children.into_iter().next().unwrap())),
            failure_behavior: expression::cast::FailureBehavior::ThrowException as i32,
        }))),
    })
}

/// Converts StarRocks null checks into Substrait `is_null` or `is_not_null`.
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
    let name = match is_not_null {
        true => "is_not_null",
        false => "is_null",
    };
    let anchor = ctx.registry.register_function(URN_COMPARISON, name);
    Ok(scalar_function(anchor, children, type_mapper::bool_type()))
}

/// Verifies that an expression node has exactly the expected number of children.
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

/// Builds a Substrait literal expression.
fn literal(literal_type: expression::literal::LiteralType) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(literal_type),
            ..Default::default()
        })),
    }
}

/// Builds a width-matched Substrait integer literal.
fn integer_literal_type(
    value: i64,
    primitive: TPrimitiveType,
) -> Result<expression::literal::LiteralType> {
    match primitive {
        TPrimitiveType::TINYINT => i8::try_from(value)
            .map(|value| expression::literal::LiteralType::I8(value as i32))
            .map_err(|_| TranslateError::malformed(format!("TINYINT literal {value} overflows"))),
        TPrimitiveType::SMALLINT => i16::try_from(value)
            .map(|value| expression::literal::LiteralType::I16(value as i32))
            .map_err(|_| TranslateError::malformed(format!("SMALLINT literal {value} overflows"))),
        TPrimitiveType::INT => i32::try_from(value)
            .map(expression::literal::LiteralType::I32)
            .map_err(|_| TranslateError::malformed(format!("INT literal {value} overflows"))),
        TPrimitiveType::BIGINT => Ok(expression::literal::LiteralType::I64(value)),
        primitive => Err(TranslateError::UnsupportedType {
            primitive: Some(primitive),
            node_type: Some(starrocks_thrift::types::TTypeNodeType::SCALAR),
            reason: "INT_LITERAL has non-integer scalar type",
        }),
    }
}

/// Builds a Substrait scalar-function expression from already translated children.
pub(crate) fn scalar_function(
    anchor: u32,
    children: Vec<Expression>,
    output_type: Type,
) -> Expression {
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

/// Encodes a decimal literal as Substrait's little-endian 128-bit unscaled integer.
fn encode_decimal(value: &str, scale: i32) -> Result<[u8; 16]> {
    if scale < 0 {
        return Err(TranslateError::malformed(format!(
            "negative decimal scale {scale}"
        )));
    }
    let raw = value.trim();
    let (negative, unsigned) = raw
        .strip_prefix('-')
        .map(|rest| (true, rest))
        .or_else(|| raw.strip_prefix('+').map(|rest| (false, rest)))
        .unwrap_or((false, raw));
    let mut parts = unsigned.split('.');
    let int_part = parts.next().unwrap_or_default();
    let frac_part = parts.next().unwrap_or_default();
    if parts.next().is_some() {
        return Err(TranslateError::malformed(format!(
            "invalid decimal literal {value}"
        )));
    }
    if !int_part.chars().all(|ch| ch.is_ascii_digit())
        || !frac_part.chars().all(|ch| ch.is_ascii_digit())
    {
        return Err(TranslateError::malformed(format!(
            "invalid decimal literal {value}"
        )));
    }

    let scale = scale as usize;
    if frac_part.len() > scale {
        return Err(TranslateError::malformed(format!(
            "decimal literal {value} exceeds scale {scale}"
        )));
    }

    let mut digits = String::from(if int_part.is_empty() { "0" } else { int_part });
    digits.push_str(frac_part);
    for _ in frac_part.len()..scale {
        digits.push('0');
    }
    let digits = digits.trim_start_matches('0');
    let mut unscaled = if digits.is_empty() {
        0
    } else {
        digits.parse::<i128>().map_err(|_| {
            TranslateError::malformed(format!("decimal literal {value} overflows i128"))
        })?
    };
    if negative {
        unscaled = -unscaled;
    }
    Ok(unscaled.to_le_bytes())
}
