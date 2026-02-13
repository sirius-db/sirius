//! Plan node translation: Doris TPlanNode tree → Substrait Rel tree.
//!
//! TPlan.nodes is a flat pre-order list. Each TPlanNode has `num_children`.
//! We reconstruct the tree recursively and map each node type to a Substrait Rel.

use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Context, Result};

use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use doris_thrift::plan_nodes::{TFileScanRangeParams, TJoinOp, TPlan, TPlanNode, TPlanNodeType};
use substrait::proto::r#type;
use substrait::proto::{
    aggregate_rel, expression, fetch_rel, join_rel, rel, sort_field, AggregateFunction,
    AggregateRel, CrossRel, Expression, FetchRel, FilterRel, FunctionArgument, JoinRel, Rel,
    SortField, SortRel, Type,
};

use crate::descriptor_table::DescriptorTable;
use crate::{expr_translator, scan_translator, ExtensionRegistry};

/// Translate a Doris TPlan (flat pre-order node list) into a Substrait Rel tree.
pub fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    scan_params: &BTreeMap<i32, TFileScanRangeParams>,
    registry: &mut ExtensionRegistry,
    table_schemas: &HashMap<String, Vec<String>>,
) -> Result<Rel> {
    if plan.nodes.is_empty() {
        bail!("TPlan has no nodes");
    }
    let mut idx = 0;
    translate_node(&plan.nodes, &mut idx, desc, scan_params, registry, table_schemas)
}

fn translate_node(
    nodes: &[TPlanNode],
    idx: &mut usize,
    desc: &DescriptorTable,
    scan_params: &BTreeMap<i32, TFileScanRangeParams>,
    registry: &mut ExtensionRegistry,
    table_schemas: &HashMap<String, Vec<String>>,
) -> Result<Rel> {
    if *idx >= nodes.len() {
        bail!("unexpected end of plan nodes at index {}", *idx);
    }
    let node = &nodes[*idx];
    *idx += 1;

    // Save the first child's TPlanNode reference before advancing through children.
    // Needed for:
    //   - Joins: count left-side columns for field offsets
    //   - Aggregation/Sort: resolve SLOT_REFs against the child's output schema
    let num_children = node.num_children as usize;
    let first_child_node = if num_children >= 1 && *idx < nodes.len() {
        Some(nodes[*idx].clone())
    } else {
        None
    };

    // Recursively translate children first (pre-order: children follow this node).
    let children: Vec<Rel> = (0..num_children)
        .map(|_| translate_node(nodes, idx, desc, scan_params, registry, table_schemas))
        .collect::<Result<_>>()?;

    // Translate this node based on type.
    let rel = if node.node_type == TPlanNodeType::FILE_SCAN_NODE {
        let params = scan_params.get(&node.node_id);
        scan_translator::translate_file_scan(node, desc, params, table_schemas)?
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
    } else if node.node_type == TPlanNodeType::MATERIALIZATION_NODE {
        // Materialization is a pass-through that materializes intermediate results.
        if children.len() != 1 {
            bail!(
                "MATERIALIZATION_NODE expected 1 child, got {}",
                children.len()
            );
        }
        return Ok(children.into_iter().next().unwrap());
    } else if node.node_type == TPlanNodeType::OLAP_SCAN_NODE {
        bail!("OLAP_SCAN_NODE not supported on Sirius GPU backend (Parquet only)")
    } else if node.node_type == TPlanNodeType::HASH_JOIN_NODE {
        return translate_hash_join_node(node, children, first_child_node.as_ref(), desc, registry);
    } else if node.node_type == TPlanNodeType::AGGREGATION_NODE {
        return translate_aggregation_node(node, children, first_child_node.as_ref(), desc, registry);
    } else if node.node_type == TPlanNodeType::SORT_NODE {
        return translate_sort_node(node, children, first_child_node.as_ref(), desc, registry);
    } else if node.node_type == TPlanNodeType::CROSS_JOIN_NODE {
        return translate_nested_loop_join_node(
            node,
            children,
            first_child_node.as_ref(),
            desc,
            registry,
        );
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

/// Count the number of output columns in a Substrait Rel tree.
///
/// Used by join translators to determine the left side's column count,
/// so right-side field references can be offset correctly.
fn count_rel_columns(rel: &Rel) -> usize {
    match rel.rel_type.as_ref() {
        Some(rel::RelType::Read(read)) => read
            .base_schema
            .as_ref()
            .map(|s| s.names.len())
            .unwrap_or(0),
        Some(rel::RelType::Filter(filter)) => filter
            .input
            .as_deref()
            .map(count_rel_columns)
            .unwrap_or(0),
        Some(rel::RelType::Sort(sort)) => {
            sort.input.as_deref().map(count_rel_columns).unwrap_or(0)
        }
        Some(rel::RelType::Fetch(fetch)) => fetch
            .input
            .as_deref()
            .map(count_rel_columns)
            .unwrap_or(0),
        Some(rel::RelType::Join(join)) => {
            let left = join.left.as_deref().map(count_rel_columns).unwrap_or(0);
            let right = join.right.as_deref().map(count_rel_columns).unwrap_or(0);
            left + right
        }
        Some(rel::RelType::Cross(cross)) => {
            let left = cross.left.as_deref().map(count_rel_columns).unwrap_or(0);
            let right = cross
                .right
                .as_deref()
                .map(count_rel_columns)
                .unwrap_or(0);
            left + right
        }
        Some(rel::RelType::Aggregate(agg)) => {
            let groups = agg
                .groupings
                .first()
                .map(|g| g.grouping_expressions.len())
                .unwrap_or(0);
            groups + agg.measures.len()
        }
        _ => 0,
    }
}

/// Map Doris TJoinOp to Substrait JoinType i32 value.
fn map_join_type(op: &TJoinOp) -> i32 {
    if *op == TJoinOp::INNER_JOIN || *op == TJoinOp::CROSS_JOIN {
        join_rel::JoinType::Inner as i32
    } else if *op == TJoinOp::LEFT_OUTER_JOIN {
        join_rel::JoinType::Left as i32
    } else if *op == TJoinOp::RIGHT_OUTER_JOIN {
        join_rel::JoinType::Right as i32
    } else if *op == TJoinOp::FULL_OUTER_JOIN {
        join_rel::JoinType::Outer as i32
    } else if *op == TJoinOp::LEFT_SEMI_JOIN {
        join_rel::JoinType::LeftSemi as i32
    } else if *op == TJoinOp::RIGHT_SEMI_JOIN {
        join_rel::JoinType::RightSemi as i32
    } else if *op == TJoinOp::LEFT_ANTI_JOIN {
        join_rel::JoinType::LeftAnti as i32
    } else if *op == TJoinOp::RIGHT_ANTI_JOIN {
        join_rel::JoinType::RightAnti as i32
    } else {
        // Fallback: treat as inner join
        join_rel::JoinType::Inner as i32
    }
}

/// Combine multiple expressions with AND.
fn and_expressions(
    mut conditions: Vec<Expression>,
    registry: &mut ExtensionRegistry,
) -> Option<Expression> {
    if conditions.is_empty() {
        return None;
    }
    if conditions.len() == 1 {
        return Some(conditions.remove(0));
    }
    let and_anchor = registry.register_function(crate::URI_BOOLEAN, "and");
    Some(expr_translator::make_scalar_fn(
        and_anchor,
        conditions,
        expr_translator::bool_type(),
    ))
}

/// Translate HASH_JOIN_NODE → Substrait JoinRel.
fn translate_hash_join_node(
    node: &TPlanNode,
    children: Vec<Rel>,
    _first_child_node: Option<&TPlanNode>,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    if children.len() != 2 {
        bail!("HASH_JOIN_NODE expected 2 children, got {}", children.len());
    }
    let hash_join = node
        .hash_join_node
        .as_ref()
        .context("HASH_JOIN_NODE missing hash_join_node data")?;

    let mut iter = children.into_iter();
    let left = iter.next().unwrap();
    let right = iter.next().unwrap();

    // Count left-side output columns for right-side field reference offsetting.
    // In a Substrait JoinRel, the combined schema is [left_cols | right_cols].
    let left_col_count = count_rel_columns(&left);

    // Build equality conditions from eq_join_conjuncts.
    let eq_anchor = registry.register_function(crate::URI_COMPARISON, "equal");
    let mut conditions: Vec<Expression> = Vec::new();

    for eq_cond in &hash_join.eq_join_conjuncts {
        // Left side: field refs are 0-based within left schema.
        let left_expr = expr_translator::translate_expr(&eq_cond.left, desc, registry)?;
        // Right side: field refs are 0-based within right schema, offset by left_col_count.
        let mut right_expr = expr_translator::translate_expr(&eq_cond.right, desc, registry)?;
        expr_translator::offset_field_refs(&mut right_expr, left_col_count);
        conditions.push(expr_translator::make_scalar_fn(
            eq_anchor,
            vec![left_expr, right_expr],
            expr_translator::bool_type(),
        ));
    }

    // Add vother_join_conjunct if present.
    let post_join_filter = if let Some(other) = &hash_join.vother_join_conjunct {
        // TODO: properly offset right-side field references in mixed expressions.
        Some(expr_translator::translate_expr(other, desc, registry)?)
    } else {
        None
    };

    let expression = and_expressions(conditions, registry);

    let join_rel = Rel {
        rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            expression: expression.map(Box::new),
            post_join_filter: post_join_filter.map(Box::new),
            r#type: map_join_type(&hash_join.join_op),
            ..Default::default()
        }))),
    };

    // Apply conjuncts as filter if present.
    apply_conjuncts(join_rel, node, desc, registry)
}

/// Translate AGGREGATION_NODE → Substrait AggregateRel.
///
/// Doris splits aggregation into two phases: partial (need_finalize=false) and
/// finalize (need_finalize=true). DuckDB's Substrait extension doesn't support
/// multi-phase aggregation, so when we detect a finalize-on-top-of-partial pattern,
/// we collapse them into a single INITIAL_TO_RESULT aggregation using the partial's
/// expressions (which reference the scan schema).
fn translate_aggregation_node(
    node: &TPlanNode,
    children: Vec<Rel>,
    child_node: Option<&TPlanNode>,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    if children.len() != 1 {
        bail!(
            "AGGREGATION_NODE expected 1 child, got {}",
            children.len()
        );
    }
    let agg_node = node
        .agg_node
        .as_ref()
        .context("AGGREGATION_NODE missing agg_node data")?;

    let input = children.into_iter().next().unwrap();

    // Collapse two-phase aggregation: if this is the finalize phase and the child
    // is already a partial AggregateRel, unwrap the child and use a single aggregation.
    if agg_node.need_finalize {
        if let Some(rel::RelType::Aggregate(child_agg)) = input.rel_type.as_ref() {
            // Reuse the partial aggregation's input, groupings, measures, and expressions,
            // but upgrade all measure phases to INITIAL_TO_RESULT.
            let mut merged = *child_agg.clone();
            for measure in &mut merged.measures {
                if let Some(func) = &mut measure.measure {
                    func.phase = 3; // INITIAL_TO_RESULT
                }
            }
            let agg_rel = Rel {
                rel_type: Some(rel::RelType::Aggregate(Box::new(merged))),
            };
            return apply_conjuncts(agg_rel, node, desc, registry);
        }
    }

    // Use the child node's row_tuples to resolve SLOT_REFs against the child's output.
    let child_row_tuples = child_node.map(|n| n.row_tuples.as_slice());

    // Translate grouping expressions.
    let mut grouping_expressions = Vec::new();
    if let Some(group_exprs) = &agg_node.grouping_exprs {
        for expr in group_exprs {
            let translated = if let Some(tuples) = child_row_tuples {
                expr_translator::translate_expr_in_context(expr, desc, registry, tuples)?
            } else {
                expr_translator::translate_expr(expr, desc, registry)?
            };
            grouping_expressions.push(translated);
        }
    }

    // Build Grouping with expressions directly inside the Grouping struct.
    // DuckDB's Substrait consumer uses the old-style Grouping.grouping_expressions,
    // not the new-style AggregateRel.grouping_expressions + expression_references.
    let grouping = if !grouping_expressions.is_empty() {
        vec![aggregate_rel::Grouping {
            #[allow(deprecated)]
            grouping_expressions: grouping_expressions.clone(),
            expression_references: vec![],
        }]
    } else {
        vec![]
    };

    // Translate aggregate function measures.
    let mut measures = Vec::new();
    for agg_fn_expr in &agg_node.aggregate_functions {
        let (func_name, arguments, output_type, is_distinct) =
            expr_translator::translate_agg_expr(agg_fn_expr, desc, registry, child_row_tuples)?;

        let func_anchor = registry.register_function(crate::URI_AGGREGATE, &func_name);

        let phase = if agg_node.need_finalize {
            3 // INITIAL_TO_RESULT
        } else {
            1 // INITIAL_TO_INTERMEDIATE
        };

        let invocation = if is_distinct { 2 } else { 1 }; // DISTINCT=2, ALL=1

        measures.push(aggregate_rel::Measure {
            measure: Some(AggregateFunction {
                function_reference: func_anchor,
                arguments,
                output_type: Some(output_type),
                phase,
                invocation,
                sorts: vec![],
                options: vec![],
                ..Default::default()
            }),
            filter: None,
        });
    }

    let agg_rel = Rel {
        rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            groupings: grouping,
            measures,
            grouping_expressions: vec![], // old-style: expressions are inside Grouping
            ..Default::default()
        }))),
    };

    apply_conjuncts(agg_rel, node, desc, registry)
}

/// Translate SORT_NODE → Substrait SortRel, optionally wrapped in FetchRel for LIMIT/OFFSET.
fn translate_sort_node(
    node: &TPlanNode,
    children: Vec<Rel>,
    child_node: Option<&TPlanNode>,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    if children.len() != 1 {
        bail!("SORT_NODE expected 1 child, got {}", children.len());
    }
    let sort_node = node
        .sort_node
        .as_ref()
        .context("SORT_NODE missing sort_node data")?;
    let sort_info = &sort_node.sort_info;

    let input = children.into_iter().next().unwrap();

    // Build SortField for each ordering expression.
    let mut sorts = Vec::new();
    for (i, expr) in sort_info.ordering_exprs.iter().enumerate() {
        let is_asc = sort_info.is_asc_order.get(i).copied().unwrap_or(true);
        let nulls_first = sort_info.nulls_first.get(i).copied().unwrap_or(true);

        let direction = match (is_asc, nulls_first) {
            (true, true) => sort_field::SortDirection::AscNullsFirst,
            (true, false) => sort_field::SortDirection::AscNullsLast,
            (false, true) => sort_field::SortDirection::DescNullsFirst,
            (false, false) => sort_field::SortDirection::DescNullsLast,
        };

        let sort_expr = if let Some(child) = child_node {
            expr_translator::translate_expr_in_context(expr, desc, registry, &child.row_tuples)?
        } else {
            expr_translator::translate_expr(expr, desc, registry)?
        };
        sorts.push(SortField {
            expr: Some(sort_expr),
            sort_kind: Some(sort_field::SortKind::Direction(direction as i32)),
        });
    }

    let sort_rel = Rel {
        rel_type: Some(rel::RelType::Sort(Box::new(SortRel {
            input: Some(Box::new(input)),
            sorts,
            ..Default::default()
        }))),
    };

    // Wrap in FetchRel if LIMIT or OFFSET is specified.
    let offset = sort_node.offset.unwrap_or(0);
    let limit = node.limit;

    let result = if limit >= 0 || offset > 0 {
        Rel {
            rel_type: Some(rel::RelType::Fetch(Box::new(FetchRel {
                input: Some(Box::new(sort_rel)),
                offset_mode: Some(fetch_rel::OffsetMode::Offset(offset)),
                count_mode: if limit >= 0 {
                    Some(fetch_rel::CountMode::Count(limit))
                } else {
                    None
                },
                ..Default::default()
            }))),
        }
    } else {
        sort_rel
    };

    apply_conjuncts(result, node, desc, registry)
}

/// Translate NESTED_LOOP_JOIN_NODE → Substrait CrossRel (for CROSS_JOIN) or JoinRel.
fn translate_nested_loop_join_node(
    node: &TPlanNode,
    children: Vec<Rel>,
    _first_child_node: Option<&TPlanNode>,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<Rel> {
    if children.len() != 2 {
        bail!(
            "NESTED_LOOP_JOIN_NODE expected 2 children, got {}",
            children.len()
        );
    }
    let nlj = node
        .nested_loop_join_node
        .as_ref()
        .context("NESTED_LOOP_JOIN_NODE missing nested_loop_join_node data")?;

    let mut iter = children.into_iter();
    let left = iter.next().unwrap();
    let right = iter.next().unwrap();

    // Count left-side columns for field reference offsetting.
    let _left_col_count = count_rel_columns(&left);

    if nlj.join_op == TJoinOp::CROSS_JOIN {
        // CrossRel for pure cross joins.
        let cross_rel = Rel {
            rel_type: Some(rel::RelType::Cross(Box::new(CrossRel {
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                ..Default::default()
            }))),
        };

        // If there's a join condition, wrap in FilterRel.
        // TODO: properly offset right-side field references in mixed expressions.
        let result = if let Some(conjunct) = &nlj.vjoin_conjunct {
            let condition = expr_translator::translate_expr(conjunct, desc, registry)?;
            Rel {
                rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                    input: Some(Box::new(cross_rel)),
                    condition: Some(Box::new(condition)),
                    ..Default::default()
                }))),
            }
        } else {
            cross_rel
        };

        apply_conjuncts(result, node, desc, registry)
    } else {
        // Non-cross nested loop join → JoinRel.
        // TODO: properly offset right-side field references in mixed expressions.
        let expression = if let Some(conjunct) = &nlj.vjoin_conjunct {
            Some(Box::new(expr_translator::translate_expr(
                conjunct, desc, registry,
            )?))
        } else {
            None
        };

        let join_rel = Rel {
            rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                expression,
                r#type: map_join_type(&nlj.join_op),
                ..Default::default()
            }))),
        };

        apply_conjuncts(join_rel, node, desc, registry)
    }
}
