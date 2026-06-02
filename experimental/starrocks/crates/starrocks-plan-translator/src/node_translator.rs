use starrocks_thrift::exprs::TExpr;
use starrocks_thrift::plan_nodes::{TPlan, TPlanNode, TPlanNodeType};
use substrait::proto::{FilterRel, ProjectRel, ReadRel, Rel, RelCommon, rel, rel_common};

use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, ExprContext, TranslateExpr};
use crate::{ExtensionRegistry, URN_BOOLEAN};

/// Partially translated relation plus the StarRocks row layout it emits.
pub(crate) struct TranslatedRel {
    /// Substrait relation built for the StarRocks subtree.
    pub rel: Rel,
    /// Tuple ids visible in the relation output row.
    pub row_tuples: Vec<i32>,
}

/// Mutable state shared by plan-node translators.
struct PlanContext<'a> {
    /// Descriptor lookups for row layouts, tables, and scan schemas.
    desc: &'a DescriptorTable,
    /// Substrait extension registry shared across the whole plan.
    registry: &'a mut ExtensionRegistry,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    fn new(desc: &'a DescriptorTable, registry: &'a mut ExtensionRegistry) -> Self {
        Self { desc, registry }
    }

    /// Creates an expression context for an expression over `row_tuples`.
    fn expr_context<'b>(&'b mut self, row_tuples: &'b [i32]) -> ExprContext<'b> {
        ExprContext::new(self.desc, self.registry, row_tuples)
    }
}

/// Trait implemented by StarRocks plan objects that can become Substrait relations.
trait TranslatePlan {
    /// Translates the receiver into a Substrait relation plus row layout.
    fn translate(&self, ctx: &mut PlanContext<'_>) -> Result<TranslatedRel>;
}

impl TranslatePlan for TPlan {
    /// Translates a flat preorder StarRocks plan into a Substrait relation tree.
    fn translate(&self, ctx: &mut PlanContext<'_>) -> Result<TranslatedRel> {
        if self.nodes.is_empty() {
            return Err(TranslateError::malformed("TPlan.nodes is empty"));
        }
        let mut cursor = PlanNodeCursor::new(&self.nodes);
        let translated = cursor.translate_next(ctx)?;
        cursor.ensure_consumed()?;
        Ok(translated)
    }
}

/// Cursor over StarRocks' flat preorder `TPlan.nodes` representation.
struct PlanNodeCursor<'a> {
    /// Node slice being parsed.
    nodes: &'a [TPlanNode],
    /// Next node index to read.
    idx: usize,
}

impl<'a> PlanNodeCursor<'a> {
    /// Creates a cursor at the start of a plan node list.
    fn new(nodes: &'a [TPlanNode]) -> Self {
        Self { nodes, idx: 0 }
    }

    /// Translates the next preorder node and its subtree.
    fn translate_next(&mut self, ctx: &mut PlanContext<'_>) -> Result<TranslatedRel> {
        let node = self
            .nodes
            .get(self.idx)
            .ok_or_else(|| TranslateError::malformed("unexpected end of plan nodes"))?;
        self.idx += 1;

        if node.num_children < 0 {
            return Err(TranslateError::malformed(format!(
                "node {} has negative child count {}",
                node.node_id, node.num_children
            )));
        }

        let children = (0..node.num_children)
            .map(|_| self.translate_next(ctx))
            .collect::<Result<Vec<_>>>()?;

        PlanNodeRel(node).translate_node(children, ctx)
    }

    /// Verifies that the top-level plan consumed all encoded nodes.
    fn ensure_consumed(&self) -> Result<()> {
        if self.idx != self.nodes.len() {
            return Err(TranslateError::malformed(format!(
                "TPlan had {} trailing node(s)",
                self.nodes.len() - self.idx
            )));
        }
        Ok(())
    }
}

/// Trait implemented by individual StarRocks plan-node translators.
trait TranslatePlanNode {
    /// Translates one node after its children have already been translated.
    fn translate_node(
        &self,
        children: Vec<TranslatedRel>,
        ctx: &mut PlanContext<'_>,
    ) -> Result<TranslatedRel>;
}

/// Dispatcher for a StarRocks plan node.
struct PlanNodeRel<'a>(&'a TPlanNode);

impl TranslatePlanNode for PlanNodeRel<'_> {
    /// Routes a StarRocks plan node to its supported v1 translator.
    fn translate_node(
        &self,
        children: Vec<TranslatedRel>,
        ctx: &mut PlanContext<'_>,
    ) -> Result<TranslatedRel> {
        match self.0.node_type {
            TPlanNodeType::FILE_SCAN_NODE => FileScanPlanNode(self.0).translate_node(children, ctx),
            TPlanNodeType::HDFS_SCAN_NODE => HdfsScanPlanNode(self.0).translate_node(children, ctx),
            TPlanNodeType::SELECT_NODE => SelectPlanNode(self.0).translate_node(children, ctx),
            TPlanNodeType::PROJECT_NODE => ProjectPlanNode(self.0).translate_node(children, ctx),
            _ => Err(TranslateError::UnsupportedPlanNode {
                node_id: self.0.node_id,
                node_type: self.0.node_type,
                reason: "plan node is outside the v1 StarRocks slice",
            }),
        }
    }
}

/// Translator for `FILE_SCAN_NODE`.
struct FileScanPlanNode<'a>(&'a TPlanNode);

impl TranslatePlanNode for FileScanPlanNode<'_> {
    /// Builds a named-table read for a file scan tuple.
    fn translate_node(
        &self,
        children: Vec<TranslatedRel>,
        ctx: &mut PlanContext<'_>,
    ) -> Result<TranslatedRel> {
        expect_children(self.0, &children, 0)?;
        let tuple_id = self
            .0
            .file_scan_node
            .as_ref()
            .map(|scan| scan.tuple_id)
            .or_else(|| self.0.row_tuples.first().copied())
            .ok_or(TranslateError::MissingField {
                context: "FILE_SCAN_NODE",
                field: "tuple_id",
            })?;
        let input = TranslatedRel {
            rel: scan_rel(ctx.desc, tuple_id)?,
            row_tuples: vec![tuple_id],
        };
        apply_conjuncts(input, self.0, ctx)
    }
}

/// Translator for `HDFS_SCAN_NODE`.
struct HdfsScanPlanNode<'a>(&'a TPlanNode);

impl TranslatePlanNode for HdfsScanPlanNode<'_> {
    /// Builds a named-table read for an HDFS scan tuple.
    fn translate_node(
        &self,
        children: Vec<TranslatedRel>,
        ctx: &mut PlanContext<'_>,
    ) -> Result<TranslatedRel> {
        expect_children(self.0, &children, 0)?;
        let tuple_id = self
            .0
            .hdfs_scan_node
            .as_ref()
            .and_then(|scan| scan.tuple_id)
            .or_else(|| self.0.row_tuples.first().copied())
            .ok_or(TranslateError::MissingField {
                context: "HDFS_SCAN_NODE",
                field: "tuple_id",
            })?;
        let input = TranslatedRel {
            rel: scan_rel(ctx.desc, tuple_id)?,
            row_tuples: vec![tuple_id],
        };
        apply_conjuncts(input, self.0, ctx)
    }
}

/// Translator for `SELECT_NODE`.
struct SelectPlanNode<'a>(&'a TPlanNode);

impl TranslatePlanNode for SelectPlanNode<'_> {
    /// Wraps the child relation with filter conjuncts.
    fn translate_node(
        &self,
        children: Vec<TranslatedRel>,
        ctx: &mut PlanContext<'_>,
    ) -> Result<TranslatedRel> {
        expect_children(self.0, &children, 1)?;
        apply_conjuncts(children.into_iter().next().unwrap(), self.0, ctx)
    }
}

/// Translator for `PROJECT_NODE`.
struct ProjectPlanNode<'a>(&'a TPlanNode);

impl TranslatePlanNode for ProjectPlanNode<'_> {
    /// Builds a project relation when no ambiguous project-level conjuncts exist.
    fn translate_node(
        &self,
        children: Vec<TranslatedRel>,
        ctx: &mut PlanContext<'_>,
    ) -> Result<TranslatedRel> {
        expect_children(self.0, &children, 1)?;
        if has_conjuncts(self.0) {
            return Err(TranslateError::UnsupportedPlanNode {
                node_id: self.0.node_id,
                node_type: self.0.node_type,
                reason: "PROJECT_NODE conjunct row layout is ambiguous in v1",
            });
        }
        let child = children.into_iter().next().unwrap();
        translate_project_node(child, self.0, ctx)
    }
}

/// Translates a flat preorder StarRocks plan into a Substrait relation tree.
pub(crate) fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    let mut ctx = PlanContext::new(desc, registry);
    plan.translate(&mut ctx)
}

/// Builds a Substrait named-table read for a StarRocks scan tuple.
fn scan_rel(desc: &DescriptorTable, tuple_id: i32) -> Result<Rel> {
    Ok(Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(desc.named_struct(tuple_id)?),
            read_type: Some(substrait::proto::read_rel::ReadType::NamedTable(
                substrait::proto::read_rel::NamedTable {
                    names: desc.table_names_for_tuple(tuple_id)?,
                    ..Default::default()
                },
            )),
            ..Default::default()
        }))),
    })
}

/// Translates a StarRocks project node while preserving descriptor output order.
fn translate_project_node(
    child: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let project_node = node
        .project_node
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "PROJECT_NODE",
            field: "project_node",
        })?;
    let slot_map = project_node
        .slot_map
        .as_ref()
        .ok_or(TranslateError::MissingField {
            context: "TProjectNode",
            field: "slot_map",
        })?;

    let output_tuples = if node.row_tuples.is_empty() {
        return Err(TranslateError::MissingField {
            context: "PROJECT_NODE",
            field: "row_tuples",
        });
    } else {
        node.row_tuples.clone()
    };

    let mut expressions = Vec::new();
    for &tuple_id in &output_tuples {
        for slot_id in ctx.desc.materialized_slot_ids(tuple_id)? {
            let expr = slot_map.get(&slot_id).ok_or_else(|| {
                TranslateError::descriptor(format!(
                    "PROJECT_NODE node {} missing slot_map expression for slot {}",
                    node.node_id, slot_id
                ))
            })?;
            let mut expr_ctx = ctx.expr_context(&child.row_tuples);
            expressions.push(expr.translate(&mut expr_ctx)?);
        }
    }

    let input_columns = count_rel_columns(&child.rel);
    let output_mapping =
        (input_columns as i32..input_columns as i32 + expressions.len() as i32).collect();

    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping,
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(child.rel)),
                expressions,
                ..Default::default()
            }))),
        },
        row_tuples: output_tuples,
    })
}

/// Adds a root projection over explicit fragment output expressions.
pub(crate) fn project_exprs(
    input: TranslatedRel,
    exprs: &[TExpr],
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    let mut ctx = PlanContext::new(desc, registry);
    project_exprs_with_context(input, exprs, &mut ctx)
}

/// Adds a projection with expressions evaluated against the input row layout.
fn project_exprs_with_context(
    input: TranslatedRel,
    exprs: &[TExpr],
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let mut expressions = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let mut expr_ctx = ctx.expr_context(&input.row_tuples);
        expressions.push(expr.translate(&mut expr_ctx)?);
    }
    let input_columns = count_rel_columns(&input.rel);
    let output_mapping =
        (input_columns as i32..input_columns as i32 + expressions.len() as i32).collect();

    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(rel_common::EmitKind::Emit(rel_common::Emit {
                        output_mapping,
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input.rel)),
                expressions,
                ..Default::default()
            }))),
        },
        row_tuples: input.row_tuples,
    })
}

/// Wraps a relation in a Substrait filter when the StarRocks node has conjuncts.
fn apply_conjuncts(
    input: TranslatedRel,
    node: &TPlanNode,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    let Some(conjuncts) = node.conjuncts.as_ref().filter(|_| has_conjuncts(node)) else {
        return Ok(input);
    };
    let mut conditions = Vec::with_capacity(conjuncts.len());
    for expr in conjuncts {
        let mut expr_ctx = ctx.expr_context(&input.row_tuples);
        conditions.push(expr.translate(&mut expr_ctx)?);
    }
    let condition = match conditions.len() {
        1 => conditions.pop().unwrap(),
        _ => {
            let and_anchor = ctx.registry.register_function(URN_BOOLEAN, "and");
            expr_translator::scalar_function(
                and_anchor,
                conditions,
                crate::type_mapper::bool_type(),
            )
        }
    };

    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(input.rel)),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        },
        row_tuples: input.row_tuples,
    })
}

/// Returns whether a StarRocks plan node carries filter conjuncts.
fn has_conjuncts(node: &TPlanNode) -> bool {
    node.conjuncts
        .as_ref()
        .map(|conjuncts| !conjuncts.is_empty())
        .unwrap_or(false)
}

/// Validates the reconstructed child count for a StarRocks plan node.
fn expect_children(node: &TPlanNode, children: &[TranslatedRel], expected: usize) -> Result<()> {
    if children.len() != expected {
        return Err(TranslateError::malformed(format!(
            "node {} {:?} expected {} child(ren), got {}",
            node.node_id,
            node.node_type,
            expected,
            children.len()
        )));
    }
    Ok(())
}

/// Counts visible output columns for the subset of Substrait rels emitted here.
pub(crate) fn count_rel_columns(rel: &Rel) -> usize {
    match rel.rel_type.as_ref() {
        Some(rel::RelType::Read(read)) => read
            .base_schema
            .as_ref()
            .and_then(|schema| schema.r#struct.as_ref())
            .map(|schema| schema.types.len())
            .unwrap_or(0),
        Some(rel::RelType::Filter(filter)) => {
            filter.input.as_deref().map(count_rel_columns).unwrap_or(0)
        }
        Some(rel::RelType::Project(project)) => project
            .common
            .as_ref()
            .and_then(|common| common.emit_kind.as_ref())
            .and_then(|emit_kind| match emit_kind {
                rel_common::EmitKind::Emit(emit) => Some(emit.output_mapping.len()),
                rel_common::EmitKind::Direct(_) => None,
            })
            .unwrap_or_else(|| {
                project.input.as_deref().map(count_rel_columns).unwrap_or(0)
                    + project.expressions.len()
            }),
        _ => 0,
    }
}
