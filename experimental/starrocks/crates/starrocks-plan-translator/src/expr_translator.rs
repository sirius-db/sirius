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
use crate::{ExtensionRegistry, URN_BOOLEAN, URN_COMPARISON};

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

        NodeExpr(node).translate_node(children, ctx)
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

/// Trait implemented by individual StarRocks expression-node translators.
trait TranslateExprNode {
    /// Translates a node after its child expressions have already been translated.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Expression>;
}

/// Dispatcher for a StarRocks expression node.
struct NodeExpr<'a>(&'a TExprNode);

impl TranslateExprNode for NodeExpr<'_> {
    /// Routes a StarRocks node to its supported v1 translator.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        match self.0.node_type {
            TExprNodeType::SLOT_REF => SlotRefExpr(self.0).translate_node(children, ctx),
            TExprNodeType::BOOL_LITERAL => BoolLiteralExpr(self.0).translate_node(children, ctx),
            TExprNodeType::INT_LITERAL => IntLiteralExpr(self.0).translate_node(children, ctx),
            TExprNodeType::FLOAT_LITERAL => FloatLiteralExpr(self.0).translate_node(children, ctx),
            TExprNodeType::STRING_LITERAL => {
                StringLiteralExpr(self.0).translate_node(children, ctx)
            }
            TExprNodeType::NULL_LITERAL => NullLiteralExpr(self.0).translate_node(children, ctx),
            TExprNodeType::DECIMAL_LITERAL => {
                DecimalLiteralExpr(self.0).translate_node(children, ctx)
            }
            TExprNodeType::BINARY_PRED => BinaryPredExpr(self.0).translate_node(children, ctx),
            TExprNodeType::COMPOUND_PRED => CompoundPredExpr(self.0).translate_node(children, ctx),
            TExprNodeType::CAST_EXPR => CastExpr(self.0).translate_node(children, ctx),
            TExprNodeType::IS_NULL_PRED => IsNullExpr(self.0).translate_node(children, ctx),
            _ => Err(TranslateError::UnsupportedExpression {
                node_type: self.0.node_type,
                reason: "expression node is outside the v1 StarRocks slice",
            }),
        }
    }
}

/// Translator for `SLOT_REF` nodes.
struct SlotRefExpr<'a>(&'a TExprNode);

impl TranslateExprNode for SlotRefExpr<'_> {
    /// Converts a StarRocks slot id into a Substrait field selection.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let slot_ref = self
            .0
            .slot_ref
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "SLOT_REF",
                field: "slot_ref",
            })?;
        let field = ctx
            .desc
            .slot_global_index(slot_ref.slot_id, ctx.row_tuples)? as i32;
        Ok(Expression {
            rex_type: Some(expression::RexType::Selection(Box::new(FieldReference {
                reference_type: Some(field_reference::ReferenceType::DirectReference(
                    ReferenceSegment {
                        reference_type: Some(reference_segment::ReferenceType::StructField(
                            Box::new(reference_segment::StructField { field, child: None }),
                        )),
                    },
                )),
                root_type: Some(field_reference::RootType::RootReference(
                    field_reference::RootReference {},
                )),
            }))),
        })
    }
}

/// Translator for `BOOL_LITERAL` nodes.
struct BoolLiteralExpr<'a>(&'a TExprNode);

impl TranslateExprNode for BoolLiteralExpr<'_> {
    /// Converts a StarRocks boolean literal into a Substrait literal.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let lit = self
            .0
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
}

/// Translator for `INT_LITERAL` nodes.
struct IntLiteralExpr<'a>(&'a TExprNode);

impl TranslateExprNode for IntLiteralExpr<'_> {
    /// Converts a StarRocks integer literal into a width-matched Substrait literal.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let lit = self
            .0
            .int_literal
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "INT_LITERAL",
                field: "int_literal",
            })?;
        Ok(literal(integer_literal_type(
            lit.value,
            type_mapper::scalar_primitive(&self.0.type_)?,
        )?))
    }
}

/// Translator for `FLOAT_LITERAL` nodes.
struct FloatLiteralExpr<'a>(&'a TExprNode);

impl TranslateExprNode for FloatLiteralExpr<'_> {
    /// Converts a StarRocks floating-point literal into a width-matched Substrait literal.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let lit = self
            .0
            .float_literal
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "FLOAT_LITERAL",
                field: "float_literal",
            })?;
        let value = *lit.value;
        let literal_type = match type_mapper::scalar_primitive(&self.0.type_)? {
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
}

/// Translator for `STRING_LITERAL` nodes.
struct StringLiteralExpr<'a>(&'a TExprNode);

impl TranslateExprNode for StringLiteralExpr<'_> {
    /// Converts a StarRocks string literal into a Substrait string literal.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let lit = self
            .0
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
}

/// Translator for `NULL_LITERAL` nodes.
struct NullLiteralExpr<'a>(&'a TExprNode);

impl TranslateExprNode for NullLiteralExpr<'_> {
    /// Converts a StarRocks null literal into a typed Substrait null literal.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let null_type = type_mapper::map_type_desc(&self.0.type_, true)?;
        Ok(literal(expression::literal::LiteralType::Null(null_type)))
    }
}

/// Translator for `DECIMAL_LITERAL` nodes.
struct DecimalLiteralExpr<'a>(&'a TExprNode);

impl TranslateExprNode for DecimalLiteralExpr<'_> {
    /// Converts a StarRocks decimal string literal into Substrait's i128 encoding.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 0)?;
        let lit = self
            .0
            .decimal_literal
            .as_ref()
            .ok_or(TranslateError::MissingField {
                context: "DECIMAL_LITERAL",
                field: "decimal_literal",
            })?;
        let decimal_type = type_mapper::map_type_desc(&self.0.type_, true)?;
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
}

/// Translator for `BINARY_PRED` nodes.
struct BinaryPredExpr<'a>(&'a TExprNode);

impl TranslateExprNode for BinaryPredExpr<'_> {
    /// Converts supported comparison opcodes into Substrait comparison functions.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 2)?;
        let opcode = self.0.opcode.ok_or(TranslateError::MissingField {
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
                    node_type: self.0.node_type,
                    reason: "binary predicate opcode is unsupported",
                });
            }
        };
        let anchor = ctx.registry.register_function(URN_COMPARISON, name);
        Ok(scalar_function(anchor, children, type_mapper::bool_type()))
    }
}

/// Translator for `COMPOUND_PRED` nodes.
struct CompoundPredExpr<'a>(&'a TExprNode);

impl TranslateExprNode for CompoundPredExpr<'_> {
    /// Converts supported boolean opcodes into Substrait boolean functions.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        let opcode = self.0.opcode.ok_or(TranslateError::MissingField {
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
                expect_child_count(self.0, &children, 1)?;
                "not"
            }
            _ => {
                return Err(TranslateError::UnsupportedExpression {
                    node_type: self.0.node_type,
                    reason: "compound predicate opcode is unsupported",
                });
            }
        };
        let anchor = ctx.registry.register_function(URN_BOOLEAN, name);
        Ok(scalar_function(anchor, children, type_mapper::bool_type()))
    }
}

/// Translator for `CAST_EXPR` nodes.
struct CastExpr<'a>(&'a TExprNode);

impl TranslateExprNode for CastExpr<'_> {
    /// Converts a StarRocks cast into a Substrait cast with throwing failure behavior.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        _ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 1)?;
        Ok(Expression {
            rex_type: Some(expression::RexType::Cast(Box::new(expression::Cast {
                r#type: Some(type_mapper::map_type_desc(&self.0.type_, true)?),
                input: Some(Box::new(children.into_iter().next().unwrap())),
                failure_behavior: expression::cast::FailureBehavior::ThrowException as i32,
            }))),
        })
    }
}

/// Translator for `IS_NULL_PRED` nodes.
struct IsNullExpr<'a>(&'a TExprNode);

impl TranslateExprNode for IsNullExpr<'_> {
    /// Converts StarRocks null checks into Substrait `is_null` or `is_not_null`.
    fn translate_node(
        &self,
        children: Vec<Expression>,
        ctx: &mut ExprContext<'_>,
    ) -> Result<Expression> {
        expect_child_count(self.0, &children, 1)?;
        let is_not_null = self
            .0
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
