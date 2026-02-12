//! Doris plan → SQL string generation.
//!
//! Converts Doris TPlanNode trees into SQL strings that can be executed
//! by DuckDB. This is a simpler alternative to the Substrait translation path.

use anyhow::{bail, Context, Result};

use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use doris_thrift::plan_nodes::{TPlan, TPlanNode, TPlanNodeType};

/// Translate a Doris TPlan (flat pre-order node list) into a SQL string.
pub fn plan_to_sql(plan: &TPlan) -> Result<String> {
    if plan.nodes.is_empty() {
        bail!("TPlan has no nodes");
    }
    let mut idx = 0;
    node_to_sql(&plan.nodes, &mut idx)
}

fn node_to_sql(nodes: &[TPlanNode], idx: &mut usize) -> Result<String> {
    if *idx >= nodes.len() {
        bail!("unexpected end of plan nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    let num_children = node.num_children as usize;

    // Recursively translate children first (pre-order).
    let children: Vec<String> = (0..num_children)
        .map(|_| node_to_sql(nodes, idx))
        .collect::<Result<_>>()?;

    match node.node_type {
        TPlanNodeType::UNION_NODE => union_to_sql(node),
        TPlanNodeType::EXCHANGE_NODE => {
            if children.len() == 1 {
                Ok(children.into_iter().next().unwrap())
            } else {
                bail!(
                    "EXCHANGE_NODE with {} children not yet supported for SQL",
                    children.len()
                )
            }
        }
        TPlanNodeType::EMPTY_SET_NODE => Ok("SELECT WHERE FALSE".to_string()),
        _ => bail!(
            "SQL generation not supported for node type: {:?}",
            node.node_type
        ),
    }
}

/// Generate SQL from a UNION_NODE's const_expr_lists.
///
/// For `SELECT 1`, this produces: `SELECT 1`
/// For `SELECT 1 UNION ALL SELECT 2`: `SELECT 1 UNION ALL SELECT 2`
fn union_to_sql(node: &TPlanNode) -> Result<String> {
    let union_node = node
        .union_node
        .as_ref()
        .context("UNION_NODE missing union_node data")?;

    if union_node.const_expr_lists.is_empty() {
        return Ok("SELECT WHERE FALSE".to_string());
    }

    let mut parts = Vec::new();
    for expr_list in &union_node.const_expr_lists {
        let values: Vec<String> = expr_list
            .iter()
            .map(expr_to_sql)
            .collect::<Result<_>>()?;
        parts.push(format!("SELECT {}", values.join(", ")));
    }
    Ok(parts.join(" UNION ALL "))
}

/// Convert a Doris TExpr to a SQL literal string.
fn expr_to_sql(expr: &TExpr) -> Result<String> {
    if expr.nodes.is_empty() {
        bail!("empty expression");
    }
    let mut idx = 0;
    expr_node_to_sql(&expr.nodes, &mut idx)
}

fn expr_node_to_sql(nodes: &[TExprNode], idx: &mut usize) -> Result<String> {
    if *idx >= nodes.len() {
        bail!("unexpected end of expression nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    let num_children = node.num_children as usize;
    let children: Vec<String> = (0..num_children)
        .map(|_| expr_node_to_sql(nodes, idx))
        .collect::<Result<_>>()?;

    match node.node_type {
        TExprNodeType::INT_LITERAL => {
            let v = node
                .int_literal
                .as_ref()
                .context("INT_LITERAL missing data")?;
            Ok(v.value.to_string())
        }
        TExprNodeType::FLOAT_LITERAL => {
            let v = node
                .float_literal
                .as_ref()
                .context("FLOAT_LITERAL missing data")?;
            Ok(format!("{}", *v.value))
        }
        TExprNodeType::STRING_LITERAL => {
            let v = node
                .string_literal
                .as_ref()
                .context("STRING_LITERAL missing data")?;
            // Escape single quotes for SQL.
            Ok(format!("'{}'", v.value.replace('\'', "''")))
        }
        TExprNodeType::BOOL_LITERAL => {
            let v = node
                .bool_literal
                .as_ref()
                .context("BOOL_LITERAL missing data")?;
            Ok(if v.value { "TRUE" } else { "FALSE" }.to_string())
        }
        TExprNodeType::DECIMAL_LITERAL => {
            let v = node
                .decimal_literal
                .as_ref()
                .context("DECIMAL_LITERAL missing data")?;
            Ok(v.value.clone())
        }
        TExprNodeType::NULL_LITERAL => Ok("NULL".to_string()),
        TExprNodeType::CAST_EXPR => {
            if children.len() != 1 {
                bail!("CAST_EXPR expected 1 child, got {}", children.len());
            }
            // For now, pass through without CAST (DuckDB will infer types).
            Ok(children.into_iter().next().unwrap())
        }
        _ => bail!(
            "SQL generation not supported for expr type: {:?}",
            node.node_type
        ),
    }
}
