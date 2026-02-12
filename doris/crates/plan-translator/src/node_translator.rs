//! Plan node translation: Doris TPlanNode tree → Substrait Rel tree.
//!
//! TPlan.nodes is a flat pre-order list. Each TPlanNode has `num_children`.
//! We reconstruct the tree recursively and map each node type to a Substrait Rel.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};

use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use doris_thrift::plan_nodes::{TFileScanRangeParams, TPlan, TPlanNode, TPlanNodeType};
use substrait::proto::r#type;
use substrait::proto::{expression, rel, Expression, FilterRel, FunctionArgument, Rel, Type};

use crate::descriptor_table::DescriptorTable;
use crate::{expr_translator, scan_translator, ExtensionRegistry};

/// Translate a Doris TPlan (flat pre-order node list) into a Substrait Rel tree.
pub fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    scan_params: &BTreeMap<i32, TFileScanRangeParams>,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    if plan.nodes.is_empty() {
        bail!("TPlan has no nodes");
    }
    let mut idx = 0;
    translate_node(&plan.nodes, &mut idx, desc, scan_params, registry)
}

fn translate_node(
    nodes: &[TPlanNode],
    idx: &mut usize,
    desc: &DescriptorTable,
    scan_params: &BTreeMap<i32, TFileScanRangeParams>,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    if *idx >= nodes.len() {
        bail!("unexpected end of plan nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    // Recursively translate children first (pre-order: children follow this node).
    let num_children = node.num_children as usize;
    let children: Vec<Rel> = (0..num_children)
        .map(|_| translate_node(nodes, idx, desc, scan_params, registry))
        .collect::<Result<_>>()?;

    // Translate this node based on type.
    let rel = if node.node_type == TPlanNodeType::FILE_SCAN_NODE {
        let params = scan_params.get(&node.node_id);
        scan_translator::translate_file_scan(node, desc, params)?
    } else if node.node_type == TPlanNodeType::SELECT_NODE {
        // SELECT_NODE passes through its single child, with conjuncts applied as a filter.
        if children.len() != 1 {
            bail!("SELECT_NODE expected 1 child, got {}", children.len());
        }
        let child = children.into_iter().next().unwrap();
        return apply_conjuncts(child, node, desc, registry);
    } else if node.node_type == TPlanNodeType::EXCHANGE_NODE {
        // Exchange nodes receive data from other fragments. For single-fragment
        // plans, just pass through the child.
        if children.len() == 1 {
            return Ok(children.into_iter().next().unwrap());
        }
        bail!("EXCHANGE_NODE with {} children not yet supported", children.len())
    } else if node.node_type == TPlanNodeType::EMPTY_SET_NODE {
        Rel {
            rel_type: Some(rel::RelType::Read(Box::new(
                substrait::proto::ReadRel {
                    read_type: Some(substrait::proto::read_rel::ReadType::VirtualTable(
                        #[allow(deprecated)]
                        substrait::proto::read_rel::VirtualTable {
                            values: vec![],
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
            ))),
        }
    } else if node.node_type == TPlanNodeType::UNION_NODE {
        translate_union_node(node)?
    } else if node.node_type == TPlanNodeType::OLAP_SCAN_NODE {
        bail!("OLAP_SCAN_NODE not supported on Sirius GPU backend (Parquet only)")
    } else if node.node_type == TPlanNodeType::HASH_JOIN_NODE {
        bail!("HASH_JOIN_NODE not yet implemented (Phase 2)")
    } else if node.node_type == TPlanNodeType::AGGREGATION_NODE {
        bail!("AGGREGATION_NODE not yet implemented (Phase 2)")
    } else if node.node_type == TPlanNodeType::SORT_NODE {
        bail!("SORT_NODE not yet implemented (Phase 2)")
    } else {
        bail!("unsupported plan node type: {}", node.node_type.0)
    };

    // Apply conjuncts as FilterRel if present (scan nodes can have pushed-down filters).
    apply_conjuncts(rel, node, desc, registry)
}

/// Translate a UNION_NODE with const_expr_lists into a VirtualTable ReadRel.
fn translate_union_node(node: &TPlanNode) -> Result<Rel> {
    let union_node = node
        .union_node
        .as_ref()
        .context("UNION_NODE missing union_node data")?;

    let mut rows = Vec::new();
    for expr_list in &union_node.const_expr_lists {
        let fields: Vec<expression::Literal> = expr_list
            .iter()
            .map(expr_to_literal)
            .collect::<Result<_>>()?;
        rows.push(expression::literal::Struct { fields });
    }

    Ok(Rel {
        rel_type: Some(rel::RelType::Read(Box::new(substrait::proto::ReadRel {
            read_type: Some(substrait::proto::read_rel::ReadType::VirtualTable(
                #[allow(deprecated)]
                substrait::proto::read_rel::VirtualTable {
                    values: rows,
                    ..Default::default()
                },
            )),
            ..Default::default()
        }))),
    })
}

/// Convert a simple TExpr (literal or cast-of-literal) into a Substrait Literal.
fn expr_to_literal(expr: &TExpr) -> Result<expression::Literal> {
    if expr.nodes.is_empty() {
        bail!("empty expression");
    }
    let mut idx = 0;
    expr_node_to_literal(&expr.nodes, &mut idx)
}

fn expr_node_to_literal(nodes: &[TExprNode], idx: &mut usize) -> Result<expression::Literal> {
    if *idx >= nodes.len() {
        bail!("unexpected end of expression nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    // Consume children for node types that have them (e.g. CAST_EXPR).
    let num_children = node.num_children as usize;

    match node.node_type {
        TExprNodeType::INT_LITERAL => {
            let v = node.int_literal.as_ref().context("INT_LITERAL missing data")?;
            Ok(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::I64(v.value)),
                ..Default::default()
            })
        }
        TExprNodeType::FLOAT_LITERAL => {
            let v = node.float_literal.as_ref().context("FLOAT_LITERAL missing data")?;
            Ok(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::Fp64(*v.value)),
                ..Default::default()
            })
        }
        TExprNodeType::DECIMAL_LITERAL => {
            let v = node.decimal_literal.as_ref().context("DECIMAL_LITERAL missing data")?;
            // Represent decimal as string literal for DuckDB compatibility.
            Ok(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::String(v.value.clone())),
                ..Default::default()
            })
        }
        TExprNodeType::STRING_LITERAL => {
            let v = node.string_literal.as_ref().context("STRING_LITERAL missing data")?;
            Ok(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::String(v.value.clone())),
                ..Default::default()
            })
        }
        TExprNodeType::BOOL_LITERAL => {
            let v = node.bool_literal.as_ref().context("BOOL_LITERAL missing data")?;
            Ok(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::Boolean(v.value)),
                ..Default::default()
            })
        }
        TExprNodeType::NULL_LITERAL => {
            let substrait_type = crate::type_mapper::map_type_desc(&node.type_)?;
            Ok(expression::Literal {
                literal_type: Some(expression::literal::LiteralType::Null(substrait_type)),
                ..Default::default()
            })
        }
        TExprNodeType::CAST_EXPR => {
            // Unwrap CAST: just return the child literal.
            if num_children != 1 {
                bail!("CAST_EXPR expected 1 child, got {}", num_children);
            }
            expr_node_to_literal(nodes, idx)
        }
        _ => bail!("unsupported literal expr type: {:?}", node.node_type),
    }
}

/// Wrap a Rel with a FilterRel for the node's conjuncts.
fn apply_conjuncts(
    input: Rel,
    node: &TPlanNode,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    let conjuncts = match &node.conjuncts {
        Some(c) if !c.is_empty() => c,
        _ => return Ok(input),
    };

    // Translate each conjunct expression.
    let conditions: Vec<Expression> = conjuncts
        .iter()
        .map(|expr| expr_translator::translate_expr(expr, desc, registry))
        .collect::<Result<_>>()?;

    // Combine multiple conjuncts with AND.
    let condition = if conditions.len() == 1 {
        conditions.into_iter().next().unwrap()
    } else {
        let and_anchor = registry.register_function(crate::URI_BOOLEAN, "and");
        Expression {
            rex_type: Some(expression::RexType::ScalarFunction(
                expression::ScalarFunction {
                    function_reference: and_anchor,
                    arguments: conditions
                        .into_iter()
                        .map(|c| FunctionArgument {
                            arg_type: Some(
                                substrait::proto::function_argument::ArgType::Value(c),
                            ),
                        })
                        .collect(),
                    output_type: Some(Type {
                        kind: Some(r#type::Kind::Bool(r#type::Boolean {
                            type_variation_reference: 0,
                            nullability: r#type::Nullability::Nullable as i32,
                        })),
                    }),
                    ..Default::default()
                },
            )),
        }
    };

    Ok(Rel {
        rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
            input: Some(Box::new(input)),
            condition: Some(Box::new(condition)),
            ..Default::default()
        }))),
    })
}
