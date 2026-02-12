//! Expression translation: Doris TExpr → Substrait Expression.
//!
//! TExpr.nodes is a flat pre-order traversal. Each TExprNode has `num_children`
//! which defines the tree structure. We reconstruct the tree recursively.

use anyhow::{bail, Context, Result};

use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use doris_thrift::opcodes::TExprOpcode;
use substrait::proto::expression::field_reference;
use substrait::proto::expression::reference_segment;
use substrait::proto::expression::{self, FieldReference, ReferenceSegment};
use substrait::proto::r#type;
use substrait::proto::{Expression, FunctionArgument, Type};

use crate::descriptor_table::DescriptorTable;
use crate::type_mapper;
use crate::ExtensionRegistry;
use crate::{URI_ARITHMETIC, URI_BOOLEAN, URI_COMPARISON, URI_STRING};

/// Translate a Doris TExpr (flat pre-order node list) into a Substrait Expression.
pub fn translate_expr(
    expr: &TExpr,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    let mut idx = 0;
    translate_expr_node(&expr.nodes, &mut idx, desc, registry)
}

fn translate_expr_node(
    nodes: &[TExprNode],
    idx: &mut usize,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    if *idx >= nodes.len() {
        bail!("unexpected end of expression nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    let num_children = node.num_children as usize;
    let children: Vec<Expression> = (0..num_children)
        .map(|_| translate_expr_node(nodes, idx, desc, registry))
        .collect::<Result<_>>()?;

    if node.node_type == TExprNodeType::SLOT_REF {
        translate_slot_ref(node, desc)
    } else if node.node_type == TExprNodeType::INT_LITERAL {
        translate_int_literal(node)
    } else if node.node_type == TExprNodeType::FLOAT_LITERAL {
        translate_float_literal(node)
    } else if node.node_type == TExprNodeType::STRING_LITERAL {
        translate_string_literal(node)
    } else if node.node_type == TExprNodeType::BOOL_LITERAL {
        translate_bool_literal(node)
    } else if node.node_type == TExprNodeType::NULL_LITERAL {
        translate_null_literal(node)
    } else if node.node_type == TExprNodeType::BINARY_PRED {
        translate_binary_pred(node, children, registry)
    } else if node.node_type == TExprNodeType::COMPOUND_PRED {
        translate_compound_pred(node, children, registry)
    } else if node.node_type == TExprNodeType::FUNCTION_CALL {
        translate_function_call(node, children, registry)
    } else if node.node_type == TExprNodeType::CAST_EXPR {
        translate_cast(node, children)
    } else if node.node_type == TExprNodeType::IS_NULL_PRED {
        translate_is_null(children, registry)
    } else if node.node_type == TExprNodeType::IN_PRED {
        translate_in_pred(children)
    } else if node.node_type == TExprNodeType::LITERAL_PRED {
        translate_literal_pred(node)
    } else {
        bail!(
            "unsupported expression node type: {}",
            node.node_type.0
        )
    }
}

fn translate_slot_ref(node: &TExprNode, desc: &DescriptorTable) -> Result<Expression> {
    let slot_ref = node
        .slot_ref
        .as_ref()
        .context("SLOT_REF node missing slot_ref data")?;
    let col_idx = desc.slot_column_index(slot_ref.slot_id)?;

    Ok(Expression {
        rex_type: Some(expression::RexType::Selection(Box::new(FieldReference {
            reference_type: Some(field_reference::ReferenceType::DirectReference(
                ReferenceSegment {
                    reference_type: Some(reference_segment::ReferenceType::StructField(
                        Box::new(reference_segment::StructField {
                            field: col_idx as i32,
                            child: None,
                        }),
                    )),
                },
            )),
            root_type: Some(field_reference::RootType::RootReference(
                field_reference::RootReference {},
            )),
        }))),
    })
}

fn translate_int_literal(node: &TExprNode) -> Result<Expression> {
    let int_lit = node
        .int_literal
        .as_ref()
        .context("INT_LITERAL missing int_literal data")?;
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::I64(int_lit.value)),
            ..Default::default()
        })),
    })
}

fn translate_float_literal(node: &TExprNode) -> Result<Expression> {
    let float_lit = node
        .float_literal
        .as_ref()
        .context("FLOAT_LITERAL missing float_literal data")?;
    // float_lit.value is OrderedFloat<f64>, dereference to get f64
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::Fp64(*float_lit.value)),
            ..Default::default()
        })),
    })
}

fn translate_string_literal(node: &TExprNode) -> Result<Expression> {
    let str_lit = node
        .string_literal
        .as_ref()
        .context("STRING_LITERAL missing string_literal data")?;
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::String(
                str_lit.value.clone(),
            )),
            ..Default::default()
        })),
    })
}

fn translate_bool_literal(node: &TExprNode) -> Result<Expression> {
    let bool_lit = node
        .bool_literal
        .as_ref()
        .context("BOOL_LITERAL missing bool_literal data")?;
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::Boolean(bool_lit.value)),
            ..Default::default()
        })),
    })
}

fn translate_null_literal(node: &TExprNode) -> Result<Expression> {
    let substrait_type = type_mapper::map_type_desc(&node.type_)?;
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::Null(substrait_type)),
            ..Default::default()
        })),
    })
}

/// LITERAL_PRED is a constant true/false predicate.
fn translate_literal_pred(node: &TExprNode) -> Result<Expression> {
    let lit_pred = node
        .literal_pred
        .as_ref()
        .context("LITERAL_PRED missing literal_pred data")?;
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::Boolean(lit_pred.value)),
            ..Default::default()
        })),
    })
}

fn bool_type() -> Type {
    Type {
        kind: Some(r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: r#type::Nullability::Nullable as i32,
        })),
    }
}

fn make_scalar_fn(
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
                    .map(|c| FunctionArgument {
                        arg_type: Some(
                            substrait::proto::function_argument::ArgType::Value(c),
                        ),
                    })
                    .collect(),
                output_type: Some(output_type),
                ..Default::default()
            },
        )),
    }
}

fn translate_binary_pred(
    node: &TExprNode,
    children: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    if children.len() != 2 {
        bail!("BINARY_PRED expected 2 children, got {}", children.len());
    }
    let opcode = node
        .opcode
        .as_ref()
        .context("BINARY_PRED missing opcode")?;
    let func_name = if *opcode == TExprOpcode::EQ {
        "equal"
    } else if *opcode == TExprOpcode::NE {
        "not_equal"
    } else if *opcode == TExprOpcode::LT {
        "lt"
    } else if *opcode == TExprOpcode::LE {
        "lte"
    } else if *opcode == TExprOpcode::GT {
        "gt"
    } else if *opcode == TExprOpcode::GE {
        "gte"
    } else {
        bail!("unsupported binary predicate opcode: {}", opcode.0)
    };
    let anchor = registry.register_function(URI_COMPARISON, func_name);
    Ok(make_scalar_fn(anchor, children, bool_type()))
}

fn translate_compound_pred(
    node: &TExprNode,
    children: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    let opcode = node
        .opcode
        .as_ref()
        .context("COMPOUND_PRED missing opcode")?;
    let func_name = if *opcode == TExprOpcode::COMPOUND_AND {
        "and"
    } else if *opcode == TExprOpcode::COMPOUND_OR {
        "or"
    } else if *opcode == TExprOpcode::COMPOUND_NOT {
        "not"
    } else {
        bail!("unsupported compound predicate opcode: {}", opcode.0)
    };
    let anchor = registry.register_function(URI_BOOLEAN, func_name);
    Ok(make_scalar_fn(anchor, children, bool_type()))
}

fn translate_function_call(
    node: &TExprNode,
    children: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    let fn_ = node.fn_.as_ref().context("FUNCTION_CALL missing fn_ data")?;
    let func_name = &fn_.name.function_name;

    // Pick extension URI based on function name.
    let uri = match func_name.as_str() {
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "equal" | "not_equal" => URI_COMPARISON,
        "and" | "or" | "not" => URI_BOOLEAN,
        "add" | "subtract" | "multiply" | "divide" | "mod" | "negate" | "abs" | "ceil"
        | "floor" | "round" => URI_ARITHMETIC,
        _ => URI_STRING,
    };

    let anchor = registry.register_function(uri, func_name);
    let output_type = type_mapper::map_type_desc(&node.type_)?;
    Ok(make_scalar_fn(anchor, children, output_type))
}

fn translate_cast(node: &TExprNode, children: Vec<Expression>) -> Result<Expression> {
    if children.len() != 1 {
        bail!("CAST_EXPR expected 1 child, got {}", children.len());
    }
    let target_type = type_mapper::map_type_desc(&node.type_)?;
    Ok(Expression {
        rex_type: Some(expression::RexType::Cast(Box::new(expression::Cast {
            input: Some(Box::new(children.into_iter().next().unwrap())),
            r#type: Some(target_type),
            failure_behavior: expression::cast::FailureBehavior::ThrowException as i32,
        }))),
    })
}

fn translate_is_null(
    children: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    if children.len() != 1 {
        bail!("IS_NULL_PRED expected 1 child, got {}", children.len());
    }
    let anchor = registry.register_function(URI_COMPARISON, "is_null");
    Ok(make_scalar_fn(anchor, children, bool_type()))
}

fn translate_in_pred(children: Vec<Expression>) -> Result<Expression> {
    if children.is_empty() {
        bail!("IN_PRED expected at least 1 child");
    }
    let mut iter = children.into_iter();
    let value = iter.next().unwrap();
    let options: Vec<Expression> = iter.collect();
    Ok(Expression {
        rex_type: Some(expression::RexType::SingularOrList(Box::new(
            expression::SingularOrList {
                value: Some(Box::new(value)),
                options,
            },
        ))),
    })
}
