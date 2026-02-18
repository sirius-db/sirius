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
                // Pass-through: exchange sender wraps its child
                Ok(children.into_iter().next().unwrap())
            } else {
                // EXCHANGE_NODE(0 children) = exchange receiver.
                // Read from the exchange table registered by the PBlock decoder.
                Ok(format!("SELECT * FROM __EXCHANGE_TABLE_{}", node.node_id))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn test_select_single_int() {
        let node = make_union_node(0, 0, vec![vec![int_literal_expr(42)]]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 42");
    }

    #[test]
    fn test_select_multiple_columns() {
        let node = make_union_node(
            0,
            0,
            vec![vec![int_literal_expr(1), int_literal_expr(2), int_literal_expr(3)]],
        );
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 1, 2, 3");
    }

    #[test]
    fn test_union_all() {
        let node = make_union_node(
            0,
            0,
            vec![
                vec![int_literal_expr(1), string_literal_expr("hello")],
                vec![int_literal_expr(2), string_literal_expr("world")],
            ],
        );
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 1, 'hello' UNION ALL SELECT 2, 'world'");
    }

    #[test]
    fn test_float_literal() {
        let node = make_union_node(0, 0, vec![vec![float_literal_expr(3.14)]]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 3.14");
    }

    #[test]
    fn test_bool_literal() {
        let node = make_union_node(
            0,
            0,
            vec![vec![bool_literal_expr(true), bool_literal_expr(false)]],
        );
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT TRUE, FALSE");
    }

    #[test]
    fn test_string_escape() {
        let node = make_union_node(0, 0, vec![vec![string_literal_expr("it's")]]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 'it''s'");
    }

    #[test]
    fn test_decimal_literal() {
        let node = make_union_node(0, 0, vec![vec![decimal_literal_expr("99.99", 10, 2)]]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 99.99");
    }

    #[test]
    fn test_cast_expr_passthrough() {
        // CAST wraps a child expr but in SQL generation, it's a pass-through.
        use doris_thrift::types::TPrimitiveType;
        let inner = int_literal_expr(42);
        let expr = cast_expr(type_desc(TPrimitiveType::DOUBLE), inner);
        let node = make_union_node(0, 0, vec![vec![expr]]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 42");
    }

    #[test]
    fn test_empty_union() {
        let node = make_union_node(0, 0, vec![]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT WHERE FALSE");
    }

    #[test]
    fn test_empty_set_node() {
        let node = make_empty_set_node(0);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT WHERE FALSE");
    }

    #[test]
    fn test_exchange_node_no_children() {
        // EXCHANGE_NODE(0 children) → SELECT from exchange table.
        let node = make_exchange_node(5, vec![0]);
        let plan = make_plan(vec![node]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT * FROM __EXCHANGE_TABLE_5");
    }

    #[test]
    fn test_exchange_node_with_child() {
        // EXCHANGE_NODE(1 child) = pass-through sender wrapping a UNION_NODE.
        use doris_thrift::plan_nodes::TPlanNodeType;
        let exchange = make_plan_node(0, TPlanNodeType::EXCHANGE_NODE, 1, vec![0]);
        let child = make_union_node(1, 0, vec![vec![int_literal_expr(99)]]);
        let plan = make_plan(vec![exchange, child]);
        let sql = plan_to_sql(&plan).unwrap();
        assert_eq!(sql, "SELECT 99");
    }
}
