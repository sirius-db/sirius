//! Plan node translation: Doris TPlanNode tree → Substrait Rel tree.
//!
//! TPlan.nodes is a flat pre-order list. Each TPlanNode has `num_children`.
//! We reconstruct the tree recursively and map each node type to a Substrait Rel.

use std::collections::{BTreeMap, HashMap};

use anyhow::{bail, Context, Result};

use doris_thrift::exprs::{TExpr, TExprNode, TExprNodeType};
use doris_thrift::plan_nodes::{TFileScanRangeParams, TJoinOp, TPlan, TPlanNode, TPlanNodeType};
use substrait::proto::r#type;
use substrait::proto::expression::field_reference;
use substrait::proto::expression::reference_segment;
use substrait::proto::{
    aggregate_rel, expression, fetch_rel, join_rel, rel, rel_common, sort_field,
    AggregateFunction, AggregateRel, CrossRel, Expression, FetchRel, FilterRel, FunctionArgument,
    JoinRel, ProjectRel, Rel, RelCommon, SortField, SortRel, Type,
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

    // Debug: log each node being translated
    tracing::debug!(
        node_idx = *idx - 1,
        node_id = node.node_id,
        node_type = ?node.node_type,
        num_children = node.num_children,
        has_agg_node = node.agg_node.is_some(),
        need_finalize = node.agg_node.as_ref().map(|a| a.need_finalize),
        "translate_node: processing plan node"
    );

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
        // Exchange nodes receive data from other fragments.
        if children.len() == 1 {
            // Single-fragment plan: pass through the child.
            return Ok(children.into_iter().next().unwrap());
        }
        if children.is_empty() {
            // No children: this is an exchange receiver. The data will be
            // loaded into a table named "__EXCHANGE_TABLE_{node_id}" by the
            // exchange handling code in grpc_service.rs. The full table name
            // (with query_id prefix) is set at execution time; the plan
            // translator uses the table_schemas map to get the actual name.
            let exchange_table_name = format!("__EXCHANGE_TABLE_{}", node.node_id);

            // Build schema from the exchange node's output tuple.
            let tuple_id = *node
                .row_tuples
                .first()
                .context("EXCHANGE_NODE has no row_tuples")?;
            let base_schema = if let Some(columns) = table_schemas.get(&exchange_table_name) {
                scan_translator::build_exchange_schema(columns, desc)?
            } else {
                desc.table_named_struct(tuple_id)?
            };

            return Ok(Rel {
                rel_type: Some(rel::RelType::Read(Box::new(substrait::proto::ReadRel {
                    base_schema: Some(base_schema),
                    read_type: Some(substrait::proto::read_rel::ReadType::NamedTable(
                        substrait::proto::read_rel::NamedTable {
                            names: vec![exchange_table_name],
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }))),
            });
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
        translate_union_node(node, children)?
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

/// Translate a UNION_NODE.
///
/// Two cases:
/// - **Constant union** (`SELECT 1 UNION ALL SELECT 2`): `const_expr_lists` is populated,
///   children are empty → VirtualTable ReadRel with literal rows.
/// - **Scan union** (`SELECT * FROM t1 UNION ALL SELECT * FROM t2`): children are populated
///   (scan nodes) → Substrait SetRel(UNION_ALL) combining child Rels.
fn translate_union_node(node: &TPlanNode, children: Vec<Rel>) -> Result<Rel> {
    let union_node = node
        .union_node
        .as_ref()
        .context("UNION_NODE missing union_node data")?;

    // Scan-based union: children were recursively translated from FILE_SCAN_NODE etc.
    if !children.is_empty() {
        return Ok(Rel {
            rel_type: Some(rel::RelType::Set(substrait::proto::SetRel {
                inputs: children,
                op: substrait::proto::set_rel::SetOp::UnionAll as i32,
                ..Default::default()
            })),
        });
    }

    // Constant union: literal values from const_expr_lists.
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
pub(crate) fn count_rel_columns(rel: &Rel) -> usize {
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
        Some(rel::RelType::Set(set)) => {
            // UNION ALL: output columns = first input's columns.
            set.inputs.first().map(count_rel_columns).unwrap_or(0)
        }
        _ => 0,
    }
}

/// Collect the flattened output column names from a Substrait Rel tree.
///
/// Returns column names in output order. For JoinRel, this is left columns
/// followed by right columns. Used by AGG expression resolution to map
/// slot names to global column indices in the child Rel's output.
pub(crate) fn collect_rel_column_names(rel: &Rel) -> Vec<String> {
    match rel.rel_type.as_ref() {
        Some(rel::RelType::Read(read)) => read
            .base_schema
            .as_ref()
            .map(|s| s.names.clone())
            .unwrap_or_default(),
        Some(rel::RelType::Filter(filter)) => filter
            .input
            .as_deref()
            .map(collect_rel_column_names)
            .unwrap_or_default(),
        Some(rel::RelType::Sort(sort)) => sort
            .input
            .as_deref()
            .map(collect_rel_column_names)
            .unwrap_or_default(),
        Some(rel::RelType::Fetch(fetch)) => fetch
            .input
            .as_deref()
            .map(collect_rel_column_names)
            .unwrap_or_default(),
        Some(rel::RelType::Join(join)) => {
            let mut names = join
                .left
                .as_deref()
                .map(collect_rel_column_names)
                .unwrap_or_default();
            names.extend(
                join.right
                    .as_deref()
                    .map(collect_rel_column_names)
                    .unwrap_or_default(),
            );
            names
        }
        Some(rel::RelType::Cross(cross)) => {
            let mut names = cross
                .left
                .as_deref()
                .map(collect_rel_column_names)
                .unwrap_or_default();
            names.extend(
                cross
                    .right
                    .as_deref()
                    .map(collect_rel_column_names)
                    .unwrap_or_default(),
            );
            names
        }
        Some(rel::RelType::Aggregate(agg)) => {
            // AGG output: grouping column names + empty names for measures.
            // Get grouping column names from the input Rel at the field reference positions.
            let input_names = agg
                .input
                .as_deref()
                .map(collect_rel_column_names)
                .unwrap_or_default();
            let mut names = Vec::new();
            if let Some(grouping) = agg.groupings.first() {
                #[allow(deprecated)]
                for expr in &grouping.grouping_expressions {
                    // Extract field reference index from grouping expression.
                    let name = extract_field_ref_name(expr, &input_names);
                    names.push(name);
                }
            }
            // Measures: use empty names (aggregate results don't have column names).
            for _ in &agg.measures {
                names.push(String::new());
            }
            names
        }
        Some(rel::RelType::Project(proj)) => {
            // Project: the output is the input columns + projection expressions.
            // DuckDB's Substrait consumer appends projection results to the input.
            // For reordering ProjectRels, just collect input names and use them
            // at the referenced positions.
            let input_names = proj
                .input
                .as_deref()
                .map(collect_rel_column_names)
                .unwrap_or_default();
            let mut names = Vec::new();
            for expr in &proj.expressions {
                let name = extract_field_ref_name(expr, &input_names);
                names.push(name);
            }
            if names.is_empty() {
                input_names
            } else {
                names
            }
        }
        _ => Vec::new(),
    }
}

/// Extract a column name from a FieldReference expression.
fn extract_field_ref_name(expr: &Expression, input_names: &[String]) -> String {
    if let Some(expression::RexType::Selection(sel)) = expr.rex_type.as_ref() {
        if let Some(field_reference::ReferenceType::DirectReference(seg)) =
            sel.reference_type.as_ref()
        {
            if let Some(reference_segment::ReferenceType::StructField(sf)) =
                seg.reference_type.as_ref()
            {
                let idx = sf.field as usize;
                if idx < input_names.len() {
                    return input_names[idx].clone();
                }
            }
        }
    }
    String::new()
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

    // For nested JOINs, the eq_join_conjunct SLOT_REFs may reference intermediate
    // tuples. Use child_rel_column_names from each side's compiled Rel for accurate
    // name-based resolution (slot_table_index fails for intermediate tuple slots).
    let left_col_names = collect_rel_column_names(&left);
    let right_col_names = collect_rel_column_names(&right);

    // Build equality conditions from eq_join_conjuncts.
    let eq_anchor = registry.register_function(crate::URI_COMPARISON, "equal");
    let mut conditions: Vec<Expression> = Vec::new();

    for eq_cond in &hash_join.eq_join_conjuncts {
        // Left side: resolve using the left Rel's column names.
        if !left_col_names.is_empty() {
            desc.set_child_rel_column_names(left_col_names.clone());
        }
        let left_expr = expr_translator::translate_expr(&eq_cond.left, desc, registry)?;
        desc.clear_child_rel_column_names();
        // Right side: resolve using the right Rel's column names, then offset.
        if !right_col_names.is_empty() {
            desc.set_child_rel_column_names(right_col_names.clone());
        }
        let mut right_expr = expr_translator::translate_expr(&eq_cond.right, desc, registry)?;
        desc.clear_child_rel_column_names();
        expr_translator::offset_field_refs(&mut right_expr, left_col_count);
        conditions.push(expr_translator::make_scalar_fn(
            eq_anchor,
            vec![left_expr, right_expr],
            expr_translator::bool_type(),
        ));
    }

    // Add vother_join_conjunct if present.
    let post_join_filter = if let Some(other) = &hash_join.vother_join_conjunct {
        // Set combined column names for cross-table expressions.
        let mut combined = left_col_names;
        combined.extend(right_col_names);
        if !combined.is_empty() {
            desc.set_child_rel_column_names(combined);
        }
        let expr = expr_translator::translate_expr(other, desc, registry)?;
        desc.clear_child_rel_column_names();
        Some(expr)
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
    // Also handles the case where a SortRel sits between finalize and partial
    // (e.g. 3-fragment plans: AGG_finalize → SORT → AGG_partial → SCAN).
    if agg_node.need_finalize {
        // Extract the partial AggregateRel, possibly through a SortRel wrapper.
        let child_rel_type_name = match input.rel_type.as_ref() {
            Some(rel::RelType::Aggregate(_)) => "Aggregate",
            Some(rel::RelType::Sort(_)) => "Sort",
            Some(rel::RelType::Filter(_)) => "Filter",
            Some(rel::RelType::Read(_)) => "Read",
            Some(rel::RelType::Join(_)) => "Join",
            Some(rel::RelType::Cross(_)) => "Cross",
            Some(rel::RelType::Fetch(_)) => "Fetch",
            Some(rel::RelType::Project(_)) => "Project",
            Some(rel::RelType::Set(_)) => "Set",
            None => "None",
            _ => "Other",
        };
        tracing::debug!(
            child_rel_type = child_rel_type_name,
            need_finalize = agg_node.need_finalize,
            "AGG_finalize: inspecting child for two-phase collapse"
        );
        let (child_agg_opt, sort_wrapper) = match input.rel_type.as_ref() {
            Some(rel::RelType::Aggregate(agg)) => (Some(agg.clone()), None),
            Some(rel::RelType::Sort(sort)) => {
                match sort.input.as_ref().and_then(|i| i.rel_type.as_ref()) {
                    Some(rel::RelType::Aggregate(agg)) => {
                        // SortRel wraps partial AggregateRel.
                        (Some(agg.clone()), Some(sort.clone()))
                    }
                    _ => (None, None),
                }
            }
            _ => (None, None),
        };

        if let Some(child_agg) = child_agg_opt {
            // Reuse the partial aggregation's input, groupings, measures, and expressions,
            // but upgrade all measure phases to INITIAL_TO_RESULT.
            let mut merged = *child_agg;

            // The partial's measure order may differ from the finalize's expected order.
            // Detect this by matching each finalize aggregate_function's SLOT_REF child
            // to the partial node's output tuple measure slots.
            // When a SortRel is between them, child_node is the SORT node, not the partial.
            // The SORT node passes through the same output tuple as its child (the partial AGG),
            // so its row_tuples contain the partial's output slots and can be used directly.
            // The partial's measure order may differ from the finalize's expected order.
            // The finalize's aggregate_functions have SLOT_REFs pointing to the finalize's
            // INPUT tuple (which was the EXCHANGE node's output in the original plan).
            // After fragment merging, the EXCHANGE is replaced by the partial, but slot IDs
            // don't change. We resolve through the descriptor table: find the finalize's
            // input tuple by looking up the first SLOT_REF, then compute positions.
            let num_grouping = merged
                .groupings
                .first()
                .map(|g| {
                    #[allow(deprecated)]
                    g.grouping_expressions.len()
                })
                .unwrap_or(0);

            // Find the finalize's input tuple from the first aggregate function's SLOT_REF.
            let input_tuple_slots = agg_node
                .aggregate_functions
                .first()
                .and_then(|agg_fn| {
                    let slot_id = agg_fn
                        .nodes
                        .iter()
                        .find(|n| n.node_type == TExprNodeType::SLOT_REF)
                        .and_then(|n| n.slot_ref.as_ref())
                        .map(|sr| sr.slot_id)?;
                    // Look up this slot to find its parent tuple.
                    let slot = desc.get_slot(slot_id).ok()?;
                    let tuple = desc.get_tuple(slot.parent_tuple_id).ok()?;
                    // Get all materialized slots in the tuple (in order).
                    let mut slots = Vec::new();
                    for &sid in &tuple.slot_ids {
                        if let Ok(s) = desc.get_slot(sid) {
                            if s.is_materialized {
                                slots.push(sid);
                            }
                        }
                    }
                    Some(slots)
                });

            if let Some(input_slots) = input_tuple_slots {
                // Skip grouping key slots to get measure slots only.
                if input_slots.len() > num_grouping {
                    let measure_slots = &input_slots[num_grouping..];

                    let mut permutation = Vec::new();
                    let mut all_found = true;
                    for agg_fn in &agg_node.aggregate_functions {
                        let slot_id = agg_fn
                            .nodes
                            .iter()
                            .find(|n| n.node_type == TExprNodeType::SLOT_REF)
                            .and_then(|n| n.slot_ref.as_ref())
                            .map(|sr| sr.slot_id);

                        if let Some(sid) = slot_id {
                            if let Some(pos) = measure_slots.iter().position(|&s| s == sid) {
                                permutation.push(pos);
                            } else {
                                all_found = false;
                                break;
                            }
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if all_found
                        && !permutation.is_empty()
                        && !permutation.iter().enumerate().all(|(i, &v)| i == v)
                    {
                        tracing::info!(
                            permutation = ?permutation,
                            "reordering collapsed AGG measures to match finalize order"
                        );
                        let old_measures = merged.measures.clone();
                        merged.measures.clear();
                        for &old_pos in &permutation {
                            merged.measures.push(old_measures[old_pos].clone());
                        }
                    }
                }
            }

            for (i, measure) in merged.measures.iter_mut().enumerate() {
                if let Some(func) = &mut measure.measure {
                    tracing::debug!(
                        measure_idx = i,
                        func_ref = func.function_reference,
                        num_args = func.arguments.len(),
                        args = ?func.arguments.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>(),
                        "collapsed AGG measure before phase promotion"
                    );
                    func.phase = 3; // INITIAL_TO_RESULT
                }
            }
            let agg_rel = Rel {
                rel_type: Some(rel::RelType::Aggregate(Box::new(merged))),
            };

            // If there was a SortRel between finalize and partial, reapply it
            // on top of the collapsed aggregate (produces ORDER BY on the final result).
            let result = if let Some(mut sort) = sort_wrapper {
                sort.input = Some(Box::new(agg_rel));
                Rel {
                    rel_type: Some(rel::RelType::Sort(Box::new(*sort))),
                }
            } else {
                agg_rel
            };
            return apply_conjuncts(result, node, desc, registry);
        }
    }

    // Use the child node's row_tuples to resolve SLOT_REFs against the child's output.
    let child_row_tuples = child_node.map(|n| n.row_tuples.as_slice());

    // Set the compiled child Rel's column names on the descriptor table for accurate
    // global index resolution. This is critical for nested JoinRels (3+ table JOINs)
    // where the descriptor table's intermediate tuples can't reconstruct the full
    // flattened column ordering.
    let child_col_names = collect_rel_column_names(&input);
    if !child_col_names.is_empty() {
        desc.set_child_rel_column_names(child_col_names);
    }

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

    // Clear child Rel column names now that expressions are translated.
    desc.clear_child_rel_column_names();

    let num_grouping = grouping_expressions.len();
    let num_measures = measures.len();

    let agg_rel = Rel {
        rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
            input: Some(Box::new(input)),
            groupings: grouping,
            measures,
            grouping_expressions: vec![], // old-style: expressions are inside Grouping
            ..Default::default()
        }))),
    };

    // Apply node projections as a ProjectRel if present (like the C++ reference impl).
    // The Substrait AggregateRel outputs [grouping_0..n, measure_0..m], but the FE's
    // output_tuple may interleave them. The node's `projections` field describes the
    // desired output mapping.
    let agg_rel = apply_node_projections(agg_rel, node, desc, num_grouping, num_measures);

    apply_conjuncts(agg_rel, node, desc, registry)
}

/// Apply a node's projections as a ProjectRel if present.
///
/// Doris TPlanNodes can have `projections` that define the output column mapping.
/// For AggregateRel, this reorders from Substrait's grouping-first output to the
/// FE's expected interleaved order. Each projection TExpr is a SLOT_REF that
/// references the AGG output tuple; we map these to Substrait AGG positions.
fn apply_node_projections(
    rel: Rel,
    node: &TPlanNode,
    desc: &DescriptorTable,
    num_grouping: usize,
    num_measures: usize,
) -> Rel {
    let projections = match &node.projections {
        Some(p) if !p.is_empty() => p,
        _ => return rel,
    };

    let total = num_grouping + num_measures;
    if total == 0 {
        return rel;
    }

    // Get grouping expression column names (in Substrait output order).
    let grouping_names: Vec<String> = node
        .agg_node
        .as_ref()
        .and_then(|a| a.grouping_exprs.as_ref())
        .map(|exprs| {
            exprs
                .iter()
                .filter_map(|expr| {
                    expr.nodes.first().and_then(|n| {
                        if n.node_type == TExprNodeType::SLOT_REF {
                            n.slot_ref.as_ref().and_then(|sr| {
                                desc.get_slot(sr.slot_id).ok().map(|s| s.col_name.clone())
                            })
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Map each projection SLOT_REF to its Substrait AGG output position.
    let mut permutation = Vec::new();
    let mut measure_idx = 0;
    for proj in projections {
        let slot_ref = proj
            .nodes
            .first()
            .filter(|n| n.node_type == TExprNodeType::SLOT_REF)
            .and_then(|n| n.slot_ref.as_ref());
        let Some(sr) = slot_ref else {
            return rel; // Non-SLOT_REF projection — can't handle
        };
        let slot = match desc.get_slot(sr.slot_id) {
            Ok(s) => s,
            Err(_) => return rel,
        };

        if !slot.col_name.is_empty() {
            // Grouping slot: find its position in grouping_names.
            if let Some(pos) = grouping_names.iter().position(|g| g == &slot.col_name) {
                permutation.push(pos);
            } else {
                return rel; // Can't map
            }
        } else {
            // Measure slot: in Substrait order after all grouping.
            if measure_idx < num_measures {
                permutation.push(num_grouping + measure_idx);
                measure_idx += 1;
            } else {
                return rel; // Too many measures
            }
        }
    }

    // Check if reordering is needed.
    if permutation.iter().enumerate().all(|(i, &v)| i == v) {
        return rel;
    }

    tracing::info!(
        permutation = ?permutation,
        num_grouping,
        num_measures,
        "adding ProjectRel to reorder AGG output"
    );

    // Build ProjectRel with field references in the permutation order.
    // Substrait ProjectRel APPENDS expressions to the input (input_cols + projected_cols).
    // Use emit to select only the projected columns (skip the pass-through input).
    let num_input = total; // AGG output column count (grouping + measures)
    let expressions: Vec<Expression> = permutation
        .iter()
        .map(|&src_idx| make_field_ref(src_idx))
        .collect();

    // Emit: select the projected columns (indices num_input..num_input+num_proj-1)
    let output_mapping: Vec<i32> = (num_input as i32..num_input as i32 + expressions.len() as i32)
        .collect();

    Rel {
        rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
            common: Some(RelCommon {
                emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                    output_mapping,
                })),
                ..Default::default()
            }),
            input: Some(Box::new(rel)),
            expressions,
            ..Default::default()
        }))),
    }
}

/// Create a simple FieldReference expression for a column index.
fn make_field_ref(idx: usize) -> Expression {
    Expression {
        rex_type: Some(expression::RexType::Selection(Box::new(
            expression::FieldReference {
                reference_type: Some(field_reference::ReferenceType::DirectReference(
                    expression::ReferenceSegment {
                        reference_type: Some(reference_segment::ReferenceType::StructField(
                            Box::new(reference_segment::StructField {
                                field: idx as i32,
                                child: None,
                            }),
                        )),
                    },
                )),
                root_type: Some(field_reference::RootType::RootReference(
                    field_reference::RootReference {},
                )),
            },
        ))),
    }
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

    tracing::info!(
        node_id = node.node_id,
        row_tuples = ?node.row_tuples,
        child_row_tuples = ?child_node.map(|c| &c.row_tuples),
        num_ordering_exprs = sort_info.ordering_exprs.len(),
        "translate_sort_node: sort context"
    );

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

        // Resolve sort expressions against the sort node's own row_tuples first
        // (these match the sort input/output schema). Fall back to child node's
        // tuples, then to unscoped resolution.
        let sort_expr = {
            let try_tuples = if !node.row_tuples.is_empty() {
                Some(&node.row_tuples[..])
            } else if let Some(child) = child_node {
                Some(&child.row_tuples[..])
            } else {
                None
            };
            if let Some(tuples) = try_tuples {
                expr_translator::translate_expr_in_context(expr, desc, registry, tuples)?
            } else {
                expr_translator::translate_expr(expr, desc, registry)?
            }
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
