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
    translate_expr_node(&expr.nodes, &mut idx, desc, registry, None)
}

/// Translate a Doris TExpr with explicit row_tuples context for global column indexing.
///
/// Used by join translators where SLOT_REFs must resolve to global indices across
/// a combined schema (e.g. [left_cols | right_cols]).
pub fn translate_expr_in_context(
    expr: &TExpr,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
    row_tuples: &[i32],
) -> Result<Expression> {
    let mut idx = 0;
    translate_expr_node(&expr.nodes, &mut idx, desc, registry, Some(row_tuples))
}


/// Parse an aggregate function TExpr and return (func_name, arguments, output_type, is_distinct).
pub fn translate_agg_expr(
    expr: &TExpr,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
    row_tuples: Option<&[i32]>,
) -> Result<(String, Vec<FunctionArgument>, Type, bool)> {
    if expr.nodes.is_empty() {
        bail!("empty aggregate expression");
    }
    let root = &expr.nodes[0];

    // Get function info from fn_ field (present on both AGG_EXPR and FUNCTION_CALL nodes).
    let fn_ = root
        .fn_
        .as_ref()
        .context("aggregate expression missing fn_ data")?;
    let raw_name = fn_.name.function_name.clone();

    // Get output type.
    let output_type = type_mapper::map_type_desc(&root.type_)?;

    // Translate child arguments (skip root node).
    let num_children = root.num_children as usize;
    let mut idx = 1;
    let mut args = Vec::new();
    for _ in 0..num_children {
        let arg_expr = translate_expr_node(&expr.nodes, &mut idx, desc, registry, row_tuples)?;
        args.push(FunctionArgument {
            arg_type: Some(substrait::proto::function_argument::ArgType::Value(arg_expr)),
        });
    }

    // Detect distinct invocation from function name prefix.
    let (func_name, is_distinct) = if let Some(base) = raw_name.strip_prefix("multi_distinct_") {
        (base.to_string(), true)
    } else {
        (raw_name, false)
    };

    Ok((func_name, args, output_type, is_distinct))
}

pub(crate) fn translate_expr_node(
    nodes: &[TExprNode],
    idx: &mut usize,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
    row_tuples: Option<&[i32]>,
) -> Result<Expression> {
    if *idx >= nodes.len() {
        bail!("unexpected end of expression nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    let num_children = node.num_children as usize;
    let children: Vec<Expression> = (0..num_children)
        .map(|_| translate_expr_node(nodes, idx, desc, registry, row_tuples))
        .collect::<Result<_>>()?;

    if node.node_type == TExprNodeType::SLOT_REF {
        translate_slot_ref(node, desc, row_tuples)
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
    } else if node.node_type == TExprNodeType::CASE_EXPR {
        translate_case_expr(node, children, registry)
    } else if node.node_type == TExprNodeType::LIKE_PRED {
        translate_like_pred(children, registry)
    } else if node.node_type == TExprNodeType::DATE_LITERAL {
        translate_date_literal(node)
    } else if node.node_type == TExprNodeType::DECIMAL_LITERAL {
        translate_decimal_literal(node)
    } else {
        bail!(
            "unsupported expression node type: {}",
            node.node_type.0
        )
    }
}

fn translate_slot_ref(
    node: &TExprNode,
    desc: &DescriptorTable,
    row_tuples: Option<&[i32]>,
) -> Result<Expression> {
    let slot_ref = node
        .slot_ref
        .as_ref()
        .context("SLOT_REF node missing slot_ref data")?;
    let col_idx = if let Some(tuples) = row_tuples {
        desc.slot_global_index(slot_ref.slot_id, tuples)?
    } else {
        desc.slot_table_index(slot_ref.slot_id)?
    };
    if let Ok(slot) = desc.get_slot(slot_ref.slot_id) {
        tracing::debug!(
            slot_id = slot_ref.slot_id,
            col_name = %slot.col_name,
            column_pos = slot.column_pos,
            parent_tuple = slot.parent_tuple_id,
            resolved_idx = col_idx,
            "SLOT_REF resolved"
        );
    }

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

pub(crate) fn bool_type() -> Type {
    Type {
        kind: Some(r#type::Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability: r#type::Nullability::Nullable as i32,
        })),
    }
}

pub(crate) fn make_scalar_fn(
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

fn translate_decimal_literal(node: &TExprNode) -> Result<Expression> {
    let dec_lit = node
        .decimal_literal
        .as_ref()
        .context("DECIMAL_LITERAL missing decimal_literal data")?;
    // Represent decimal as string literal — DuckDB will infer the correct type.
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::String(
                dec_lit.value.clone(),
            )),
            ..Default::default()
        })),
    })
}

fn translate_date_literal(node: &TExprNode) -> Result<Expression> {
    let date_lit = node
        .date_literal
        .as_ref()
        .context("DATE_LITERAL missing date_literal data")?;
    // Date values are stored as strings (e.g., "2024-01-15").
    // DuckDB can parse date strings with implicit or explicit CAST.
    Ok(Expression {
        rex_type: Some(expression::RexType::Literal(expression::Literal {
            literal_type: Some(expression::literal::LiteralType::String(
                date_lit.value.clone(),
            )),
            ..Default::default()
        })),
    })
}

/// Translate LIKE_PRED → Substrait "like" scalar function.
fn translate_like_pred(
    children: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    if children.len() != 2 {
        bail!("LIKE_PRED expected 2 children, got {}", children.len());
    }
    let anchor = registry.register_function(URI_STRING, "like");
    Ok(make_scalar_fn(anchor, children, bool_type()))
}

/// Translate CASE_EXPR → Substrait IfThen expression.
///
/// Doris CASE_EXPR structure:
///   - has_case_expr=false: children = [when_cond1, then1, when_cond2, then2, ..., (else)]
///   - has_case_expr=true:  children = [case_value, when_val1, then1, when_val2, then2, ..., (else)]
///
/// When has_case_expr=true, we convert "CASE x WHEN v THEN ..." into
/// "IF x=v THEN ..." by generating equality comparisons.
fn translate_case_expr(
    node: &TExprNode,
    children: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Result<Expression> {
    let case_expr = node
        .case_expr
        .as_ref()
        .context("CASE_EXPR missing case_expr data")?;

    let has_case = case_expr.has_case_expr;
    let has_else = case_expr.has_else_expr;

    if has_case {
        // CASE value WHEN v1 THEN r1 WHEN v2 THEN r2 ... [ELSE else_val] END
        // children: [case_value, when_val1, then1, when_val2, then2, ..., (else)]
        if children.is_empty() {
            bail!("CASE_EXPR with case_value has no children");
        }
        let mut iter = children.into_iter();
        let case_value = iter.next().unwrap();
        let remaining: Vec<Expression> = iter.collect();

        let (pairs_slice, else_expr) = if has_else {
            if remaining.is_empty() {
                bail!("CASE_EXPR has_else but no remaining children");
            }
            let (pairs, else_part) = remaining.split_at(remaining.len() - 1);
            (pairs.to_vec(), Some(Box::new(else_part[0].clone())))
        } else {
            (remaining, None)
        };

        if pairs_slice.len() % 2 != 0 {
            bail!(
                "CASE_EXPR expected even number of when/then pairs, got {}",
                pairs_slice.len()
            );
        }

        let eq_anchor = registry.register_function(URI_COMPARISON, "equal");
        let mut ifs = Vec::new();
        for chunk in pairs_slice.chunks(2) {
            let when_val = chunk[0].clone();
            let then_val = chunk[1].clone();
            // Generate: case_value = when_val
            let condition = make_scalar_fn(
                eq_anchor,
                vec![case_value.clone(), when_val],
                bool_type(),
            );
            ifs.push(expression::if_then::IfClause {
                r#if: Some(condition),
                then: Some(then_val),
            });
        }

        Ok(Expression {
            rex_type: Some(expression::RexType::IfThen(Box::new(expression::IfThen {
                ifs,
                r#else: else_expr,
            }))),
        })
    } else {
        // Searched CASE: CASE WHEN cond1 THEN r1 WHEN cond2 THEN r2 ... [ELSE else_val] END
        // children: [when_cond1, then1, when_cond2, then2, ..., (else)]
        let (pairs_slice, else_expr) = if has_else {
            if children.is_empty() {
                bail!("CASE_EXPR has_else but no children");
            }
            let (pairs, else_part) = children.split_at(children.len() - 1);
            (pairs.to_vec(), Some(Box::new(else_part[0].clone())))
        } else {
            (children, None)
        };

        if pairs_slice.len() % 2 != 0 {
            bail!(
                "CASE_EXPR expected even number of when/then pairs, got {}",
                pairs_slice.len()
            );
        }

        let mut ifs = Vec::new();
        for chunk in pairs_slice.chunks(2) {
            ifs.push(expression::if_then::IfClause {
                r#if: Some(chunk[0].clone()),
                then: Some(chunk[1].clone()),
            });
        }

        Ok(Expression {
            rex_type: Some(expression::RexType::IfThen(Box::new(expression::IfThen {
                ifs,
                r#else: else_expr,
            }))),
        })
    }
}

/// Add a fixed offset to all field references in an expression.
///
/// Used for join right-side expressions where field indices need to be shifted
/// by the left side's column count in the combined [left|right] schema.
pub fn offset_field_refs(expr: &mut Expression, offset: usize) {
    if offset == 0 {
        return;
    }
    match &mut expr.rex_type {
        Some(expression::RexType::Selection(ref mut field_ref)) => {
            if let Some(field_reference::ReferenceType::DirectReference(ref mut seg)) =
                field_ref.reference_type
            {
                if let Some(reference_segment::ReferenceType::StructField(ref mut sf)) =
                    seg.reference_type
                {
                    sf.field += offset as i32;
                }
            }
        }
        Some(expression::RexType::ScalarFunction(ref mut func)) => {
            for arg in &mut func.arguments {
                if let Some(substrait::proto::function_argument::ArgType::Value(ref mut e)) =
                    arg.arg_type
                {
                    offset_field_refs(e, offset);
                }
            }
        }
        Some(expression::RexType::Cast(ref mut cast)) => {
            if let Some(ref mut input) = cast.input {
                offset_field_refs(input, offset);
            }
        }
        Some(expression::RexType::SingularOrList(ref mut list)) => {
            if let Some(ref mut value) = list.value {
                offset_field_refs(value, offset);
            }
            for opt in &mut list.options {
                offset_field_refs(opt, offset);
            }
        }
        Some(expression::RexType::IfThen(ref mut if_then)) => {
            for clause in &mut if_then.ifs {
                if let Some(ref mut cond) = clause.r#if {
                    offset_field_refs(cond, offset);
                }
                if let Some(ref mut then) = clause.then {
                    offset_field_refs(then, offset);
                }
            }
            if let Some(ref mut else_expr) = if_then.r#else {
                offset_field_refs(else_expr, offset);
            }
        }
        _ => {} // Literals and other types don't contain field references
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_table::DescriptorTable;
    use crate::test_helpers::*;
    use doris_thrift::exprs::{TCaseExpr, TDateLiteral};
    use doris_thrift::opcodes::TExprOpcode;
    use doris_thrift::types::TPrimitiveType;

    fn make_test_desc() -> DescriptorTable {
        let desc_tbl = make_desc_table(
            vec![(0, Some(1))],
            vec![
                (0, 0, 0, "col_a", TPrimitiveType::BIGINT),
                (1, 0, 1, "col_b", TPrimitiveType::VARCHAR),
                (2, 0, 2, "col_c", TPrimitiveType::DOUBLE),
            ],
        );
        DescriptorTable::from_thrift(&desc_tbl).unwrap()
    }

    // ---- Literal tests ----

    #[test]
    fn test_int_literal() {
        let expr = int_literal_expr(42);
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::I64(v) => assert_eq!(v, 42),
                other => panic!("expected I64, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_float_literal() {
        let expr = float_literal_expr(3.14);
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::Fp64(v) => assert!((v - 3.14).abs() < 1e-10),
                other => panic!("expected Fp64, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_string_literal() {
        let expr = string_literal_expr("hello");
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::String(s) => assert_eq!(s, "hello"),
                other => panic!("expected String, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_bool_literal() {
        let expr = bool_literal_expr(true);
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::Boolean(v) => assert!(v),
                other => panic!("expected Boolean, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_null_literal() {
        let expr = null_literal_expr(TPrimitiveType::INT);
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::Null(_) => {} // Good
                other => panic!("expected Null, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_decimal_literal() {
        let expr = decimal_literal_expr("99.99", 10, 2);
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::String(s) => assert_eq!(s, "99.99"),
                other => panic!("expected String (decimal), got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_date_literal() {
        let nodes = vec![make_expr_node_pub(
            doris_thrift::exprs::TExprNodeType::DATE_LITERAL,
            type_desc(TPrimitiveType::DATEV2),
            0,
            |n| n.date_literal = Some(TDateLiteral { value: "2024-01-15".to_string() }),
        )];
        let expr = doris_thrift::exprs::TExpr { nodes };
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::String(s) => assert_eq!(s, "2024-01-15"),
                other => panic!("expected String (date), got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    #[test]
    fn test_literal_pred_true() {
        let expr = literal_pred_expr(true);
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::Boolean(v) => assert!(v),
                other => panic!("expected Boolean, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    // ---- SLOT_REF tests ----

    #[test]
    fn test_slot_ref() {
        let expr = slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT));
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Selection(field_ref) => {
                match field_ref.reference_type.unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => {
                                assert_eq!(sf.field, 0);
                            }
                            other => panic!("expected StructField, got {:?}", other),
                        }
                    }
                    other => panic!("expected DirectReference, got {:?}", other),
                }
            }
            other => panic!("expected Selection, got {:?}", other),
        }
    }

    #[test]
    fn test_slot_ref_second_column() {
        let expr = slot_ref_expr(1, type_desc(TPrimitiveType::VARCHAR));
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Selection(field_ref) => {
                match field_ref.reference_type.unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => {
                                assert_eq!(sf.field, 1);
                            }
                            other => panic!("expected StructField, got {:?}", other),
                        }
                    }
                    other => panic!("expected DirectReference, got {:?}", other),
                }
            }
            other => panic!("expected Selection, got {:?}", other),
        }
    }

    // ---- BINARY_PRED tests ----

    #[test]
    fn test_binary_pred_all_opcodes() {
        let desc = make_test_desc();
        let opcodes = [
            (TExprOpcode::EQ, "equal"),
            (TExprOpcode::NE, "not_equal"),
            (TExprOpcode::LT, "lt"),
            (TExprOpcode::LE, "lte"),
            (TExprOpcode::GT, "gt"),
            (TExprOpcode::GE, "gte"),
        ];
        for (opcode, expected_name) in opcodes {
            let expr = binary_pred_expr(
                opcode,
                slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
                int_literal_expr(10),
            );
            let mut reg = ExtensionRegistry::new();
            let result = translate_expr(&expr, &desc, &mut reg).unwrap();
            match result.rex_type.unwrap() {
                expression::RexType::ScalarFunction(f) => {
                    assert_eq!(f.arguments.len(), 2);
                    // Verify function was registered.
                    let (_, extensions) = reg.into_extensions();
                    let func = extensions.iter().find(|e| {
                        if let Some(substrait::proto::extensions::simple_extension_declaration::MappingType::ExtensionFunction(ef)) = &e.mapping_type {
                            ef.name == expected_name
                        } else {
                            false
                        }
                    });
                    assert!(func.is_some(), "expected function '{}' registered", expected_name);
                }
                other => panic!("expected ScalarFunction for {:?}, got {:?}", opcode, other),
            }
        }
    }

    // ---- COMPOUND_PRED tests ----

    #[test]
    fn test_compound_and() {
        let expr = compound_pred_expr(
            TExprOpcode::COMPOUND_AND,
            vec![
                binary_pred_expr(TExprOpcode::GT, slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)), int_literal_expr(5)),
                binary_pred_expr(TExprOpcode::LT, slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)), int_literal_expr(100)),
            ],
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 2, "AND should have 2 args");
            }
            other => panic!("expected ScalarFunction (and), got {:?}", other),
        }
    }

    #[test]
    fn test_compound_or() {
        let expr = compound_pred_expr(
            TExprOpcode::COMPOUND_OR,
            vec![
                binary_pred_expr(TExprOpcode::EQ, slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)), int_literal_expr(1)),
                binary_pred_expr(TExprOpcode::EQ, slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)), int_literal_expr(2)),
            ],
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 2);
            }
            other => panic!("expected ScalarFunction (or), got {:?}", other),
        }
    }

    #[test]
    fn test_compound_not() {
        let expr = compound_pred_expr(
            TExprOpcode::COMPOUND_NOT,
            vec![binary_pred_expr(
                TExprOpcode::EQ,
                slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
                int_literal_expr(0),
            )],
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 1, "NOT should have 1 arg");
            }
            other => panic!("expected ScalarFunction (not), got {:?}", other),
        }
    }

    // ---- CAST_EXPR tests ----

    #[test]
    fn test_cast_expr() {
        let expr = cast_expr(
            type_desc(TPrimitiveType::DOUBLE),
            int_literal_expr(42),
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::Cast(cast) => {
                assert!(cast.input.is_some());
                assert!(cast.r#type.is_some());
            }
            other => panic!("expected Cast, got {:?}", other),
        }
    }

    // ---- IS_NULL_PRED tests ----

    #[test]
    fn test_is_null() {
        let expr = is_null_expr(slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)));
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 1);
            }
            other => panic!("expected ScalarFunction (is_null), got {:?}", other),
        }
    }

    // ---- IN_PRED tests ----

    #[test]
    fn test_in_pred() {
        let expr = in_pred_expr(
            slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
            vec![int_literal_expr(1), int_literal_expr(2), int_literal_expr(3)],
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::SingularOrList(list) => {
                assert!(list.value.is_some());
                assert_eq!(list.options.len(), 3);
            }
            other => panic!("expected SingularOrList, got {:?}", other),
        }
    }

    // ---- FUNCTION_CALL tests ----

    #[test]
    fn test_function_call_add() {
        let expr = function_call_expr(
            "add",
            type_desc(TPrimitiveType::BIGINT),
            vec![type_desc(TPrimitiveType::BIGINT), type_desc(TPrimitiveType::BIGINT)],
            vec![
                slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
                int_literal_expr(1),
            ],
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 2);
            }
            other => panic!("expected ScalarFunction (add), got {:?}", other),
        }
    }

    #[test]
    fn test_function_call_substring() {
        let expr = function_call_expr(
            "substring",
            type_desc(TPrimitiveType::VARCHAR),
            vec![type_desc(TPrimitiveType::VARCHAR), type_desc(TPrimitiveType::INT), type_desc(TPrimitiveType::INT)],
            vec![
                slot_ref_expr(1, type_desc(TPrimitiveType::VARCHAR)),
                int_literal_expr(1),
                int_literal_expr(5),
            ],
        );
        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 3);
            }
            other => panic!("expected ScalarFunction (substring), got {:?}", other),
        }
    }

    // ---- LIKE_PRED tests ----

    #[test]
    fn test_like_pred() {
        // col_b LIKE '%pattern%'
        // LIKE_PRED has 2 children: the value and the pattern.
        let mut nodes = vec![make_expr_node_pub(
            TExprNodeType::LIKE_PRED,
            type_desc(TPrimitiveType::BOOLEAN),
            2,
            |n| {
                n.like_pred = Some(doris_thrift::exprs::TLikePredicate {
                    escape_char: "\\".to_string(),
                })
            },
        )];
        // child 1: slot ref to col_b
        nodes.extend(slot_ref_expr(1, type_desc(TPrimitiveType::VARCHAR)).nodes);
        // child 2: pattern string
        nodes.extend(string_literal_expr("%pattern%").nodes);
        let expr = doris_thrift::exprs::TExpr { nodes };

        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::ScalarFunction(f) => {
                assert_eq!(f.arguments.len(), 2);
            }
            other => panic!("expected ScalarFunction (like), got {:?}", other),
        }
    }

    // ---- CASE_EXPR tests ----

    #[test]
    fn test_searched_case_with_else() {
        // CASE WHEN col_a > 10 THEN 'big' WHEN col_a > 5 THEN 'medium' ELSE 'small' END
        let mut nodes = vec![make_expr_node_pub(
            TExprNodeType::CASE_EXPR,
            type_desc(TPrimitiveType::VARCHAR),
            5, // 2 when/then pairs + 1 else = 5 children
            |n| n.case_expr = Some(TCaseExpr { has_case_expr: false, has_else_expr: true }),
        )];
        // WHEN col_a > 10
        nodes.extend(binary_pred_expr(
            TExprOpcode::GT,
            slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
            int_literal_expr(10),
        ).nodes);
        // THEN 'big'
        nodes.extend(string_literal_expr("big").nodes);
        // WHEN col_a > 5
        nodes.extend(binary_pred_expr(
            TExprOpcode::GT,
            slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
            int_literal_expr(5),
        ).nodes);
        // THEN 'medium'
        nodes.extend(string_literal_expr("medium").nodes);
        // ELSE 'small'
        nodes.extend(string_literal_expr("small").nodes);
        let expr = doris_thrift::exprs::TExpr { nodes };

        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::IfThen(if_then) => {
                assert_eq!(if_then.ifs.len(), 2, "should have 2 WHEN clauses");
                assert!(if_then.r#else.is_some(), "should have ELSE");
            }
            other => panic!("expected IfThen, got {:?}", other),
        }
    }

    #[test]
    fn test_searched_case_without_else() {
        // CASE WHEN col_a = 1 THEN 'one' END
        let mut nodes = vec![make_expr_node_pub(
            TExprNodeType::CASE_EXPR,
            type_desc(TPrimitiveType::VARCHAR),
            2, // 1 when/then pair
            |n| n.case_expr = Some(TCaseExpr { has_case_expr: false, has_else_expr: false }),
        )];
        nodes.extend(binary_pred_expr(
            TExprOpcode::EQ,
            slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)),
            int_literal_expr(1),
        ).nodes);
        nodes.extend(string_literal_expr("one").nodes);
        let expr = doris_thrift::exprs::TExpr { nodes };

        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::IfThen(if_then) => {
                assert_eq!(if_then.ifs.len(), 1);
                assert!(if_then.r#else.is_none(), "should have no ELSE");
            }
            other => panic!("expected IfThen, got {:?}", other),
        }
    }

    #[test]
    fn test_simple_case_with_value() {
        // CASE col_a WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END
        let mut nodes = vec![make_expr_node_pub(
            TExprNodeType::CASE_EXPR,
            type_desc(TPrimitiveType::VARCHAR),
            6, // case_value + 2*(when+then) + else = 6
            |n| n.case_expr = Some(TCaseExpr { has_case_expr: true, has_else_expr: true }),
        )];
        // case value: col_a
        nodes.extend(slot_ref_expr(0, type_desc(TPrimitiveType::BIGINT)).nodes);
        // WHEN 1
        nodes.extend(int_literal_expr(1).nodes);
        // THEN 'one'
        nodes.extend(string_literal_expr("one").nodes);
        // WHEN 2
        nodes.extend(int_literal_expr(2).nodes);
        // THEN 'two'
        nodes.extend(string_literal_expr("two").nodes);
        // ELSE 'other'
        nodes.extend(string_literal_expr("other").nodes);
        let expr = doris_thrift::exprs::TExpr { nodes };

        let desc = make_test_desc();
        let mut reg = ExtensionRegistry::new();
        let result = translate_expr(&expr, &desc, &mut reg).unwrap();
        match result.rex_type.unwrap() {
            expression::RexType::IfThen(if_then) => {
                assert_eq!(if_then.ifs.len(), 2, "should have 2 WHEN clauses");
                assert!(if_then.r#else.is_some(), "should have ELSE");
                // Each WHEN condition should be an equality function (col_a = value).
                for clause in &if_then.ifs {
                    match clause.r#if.as_ref().unwrap().rex_type.as_ref().unwrap() {
                        expression::RexType::ScalarFunction(f) => {
                            assert_eq!(f.arguments.len(), 2, "equal() should have 2 args");
                        }
                        other => panic!("expected ScalarFunction (equal), got {:?}", other),
                    }
                }
            }
            other => panic!("expected IfThen, got {:?}", other),
        }
    }

    // ---- offset_field_refs tests ----

    #[test]
    fn test_offset_field_refs_basic() {
        let mut expr = Expression {
            rex_type: Some(expression::RexType::Selection(Box::new(FieldReference {
                reference_type: Some(field_reference::ReferenceType::DirectReference(
                    ReferenceSegment {
                        reference_type: Some(reference_segment::ReferenceType::StructField(
                            Box::new(reference_segment::StructField {
                                field: 0,
                                child: None,
                            }),
                        )),
                    },
                )),
                root_type: Some(field_reference::RootType::RootReference(
                    field_reference::RootReference {},
                )),
            }))),
        };
        offset_field_refs(&mut expr, 3);
        match expr.rex_type.unwrap() {
            expression::RexType::Selection(field_ref) => {
                match field_ref.reference_type.unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => {
                                assert_eq!(sf.field, 3, "field should be offset by 3");
                            }
                            other => panic!("expected StructField, got {:?}", other),
                        }
                    }
                    other => panic!("expected DirectReference, got {:?}", other),
                }
            }
            other => panic!("expected Selection, got {:?}", other),
        }
    }

    #[test]
    fn test_offset_zero_is_noop() {
        let original = Expression {
            rex_type: Some(expression::RexType::Selection(Box::new(FieldReference {
                reference_type: Some(field_reference::ReferenceType::DirectReference(
                    ReferenceSegment {
                        reference_type: Some(reference_segment::ReferenceType::StructField(
                            Box::new(reference_segment::StructField {
                                field: 5,
                                child: None,
                            }),
                        )),
                    },
                )),
                root_type: Some(field_reference::RootType::RootReference(
                    field_reference::RootReference {},
                )),
            }))),
        };
        let mut expr = original.clone();
        offset_field_refs(&mut expr, 0);
        // Should be unchanged.
        match expr.rex_type.unwrap() {
            expression::RexType::Selection(field_ref) => {
                match field_ref.reference_type.unwrap() {
                    field_reference::ReferenceType::DirectReference(seg) => {
                        match seg.reference_type.unwrap() {
                            reference_segment::ReferenceType::StructField(sf) => {
                                assert_eq!(sf.field, 5);
                            }
                            _ => panic!("unexpected"),
                        }
                    }
                    _ => panic!("unexpected"),
                }
            }
            _ => panic!("unexpected"),
        }
    }

    #[test]
    fn test_offset_literal_unchanged() {
        let mut expr = Expression {
            rex_type: Some(expression::RexType::Literal(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::I64(42)),
                ..Default::default()
            })),
        };
        offset_field_refs(&mut expr, 5);
        // Literal should be unchanged.
        match expr.rex_type.unwrap() {
            expression::RexType::Literal(lit) => match lit.literal_type.unwrap() {
                expression::literal::LiteralType::I64(v) => assert_eq!(v, 42),
                other => panic!("expected I64, got {:?}", other),
            },
            other => panic!("expected Literal, got {:?}", other),
        }
    }
}
