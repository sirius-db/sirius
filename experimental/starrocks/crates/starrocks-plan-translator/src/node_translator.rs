use starrocks_thrift::exprs::TExpr;
use starrocks_thrift::plan_nodes::{TPlan, TPlanNode, TPlanNodeType};
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{LocalFiles, NamedTable, ReadType};
use substrait::proto::{
    Expression, FilterRel, ProjectRel, ReadRel, Rel, RelCommon, rel, rel_common,
};

use crate::descriptor_table::DescriptorTable;
use crate::error::{Result, TranslateError};
use crate::expr_translator::{self, ExprContext, TranslateExpr};
use crate::scan_paths::ScanFilePaths;
use crate::{ExtensionRegistry, URN_BOOLEAN};

/// Partially translated relation plus the StarRocks row layout it emits.
pub(crate) struct TranslatedRel {
    /// Substrait relation built for the StarRocks subtree.
    pub rel: Rel,
    /// Tuple ids visible in the relation output row.
    ///
    /// For multi-input relations (joins, set ops) this is the left-to-right
    /// concatenation of the child layouts, so `DescriptorTable::slot_global_index`
    /// resolves a right-side slot to `left_width + right_index`. New multi-input
    /// translators MUST follow this ordering.
    pub row_tuples: Vec<i32>,
    /// Number of columns this relation emits.
    ///
    /// Carried as an invariant (rather than recomputed by walking `rel`) so a
    /// parent projection can compute its `output_mapping` base offset without
    /// teaching a column-counting helper about every relation type — a pattern
    /// that silently produced a wrong offset the moment an unknown relation
    /// appeared mid-tree. Every relation built here MUST set its true width.
    pub output_width: usize,
}

/// Mutable state shared by plan-node translators.
struct PlanContext<'a> {
    /// Descriptor lookups for row layouts, tables, and scan schemas.
    desc: &'a DescriptorTable,
    /// Parquet file paths for each scan node, collected from the fragment's broker
    /// scan ranges. Scans with paths emit a `local_files` read; path-less scans
    /// fall back to a named-table read.
    scan_paths: &'a ScanFilePaths,
    /// Substrait extension registry shared across the whole plan.
    registry: &'a mut ExtensionRegistry,
}

impl<'a> PlanContext<'a> {
    /// Creates a plan translation context.
    fn new(
        desc: &'a DescriptorTable,
        scan_paths: &'a ScanFilePaths,
        registry: &'a mut ExtensionRegistry,
    ) -> Self {
        Self {
            desc,
            scan_paths,
            registry,
        }
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

        translate_plan_node(node, children, ctx)
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

/// Routes a StarRocks plan node to its supported v1 translator once its children
/// have been translated.
fn translate_plan_node(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    match node.node_type {
        TPlanNodeType::FILE_SCAN_NODE => translate_file_scan(node, children, ctx),
        TPlanNodeType::HDFS_SCAN_NODE => translate_hdfs_scan(node, children, ctx),
        TPlanNodeType::SELECT_NODE => translate_select(node, children, ctx),
        TPlanNodeType::PROJECT_NODE => translate_project(node, children, ctx),
        _ => Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "plan node is outside the v1 StarRocks slice",
        }),
    }
}

/// Builds a named-table read for `FILE_SCAN_NODE`.
fn translate_file_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    // `file_scan_node.tuple_id` is required, so a present payload always resolves.
    let tuple_id = node
        .file_scan_node
        .as_ref()
        .map(|scan| scan.tuple_id)
        .or_else(|| node.row_tuples.first().copied())
        .ok_or(TranslateError::MissingField {
            context: "FILE_SCAN_NODE",
            field: "tuple_id",
        })?;
    translate_scan(node, children, tuple_id, ctx)
}

/// Builds a named-table read for `HDFS_SCAN_NODE`.
fn translate_hdfs_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    // `hdfs_scan_node.tuple_id` is optional, so fall back to the node row layout.
    let tuple_id = node
        .hdfs_scan_node
        .as_ref()
        .and_then(|scan| scan.tuple_id)
        .or_else(|| node.row_tuples.first().copied())
        .ok_or(TranslateError::MissingField {
            context: "HDFS_SCAN_NODE",
            field: "tuple_id",
        })?;
    translate_scan(node, children, tuple_id, ctx)
}

/// Builds a leaf named-table read for `tuple_id` and applies any filter conjuncts.
///
/// Shared by every scan node; new scan types (OLAP/connector/lake) only need to
/// resolve their `tuple_id` and delegate here.
fn translate_scan(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    tuple_id: i32,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 0)?;
    let file_paths = ctx.scan_paths.for_node(node.node_id);
    let input = TranslatedRel {
        rel: scan_rel(ctx.desc, tuple_id, file_paths)?,
        row_tuples: vec![tuple_id],
        output_width: ctx.desc.materialized_slot_ids(tuple_id)?.len(),
    };
    apply_conjuncts(input, node, ctx)
}

/// Wraps the child relation of a `SELECT_NODE` with its filter conjuncts.
fn translate_select(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
    apply_conjuncts(children.into_iter().next().unwrap(), node, ctx)
}

/// Builds a project relation for a `PROJECT_NODE` with no ambiguous conjuncts.
fn translate_project(
    node: &TPlanNode,
    children: Vec<TranslatedRel>,
    ctx: &mut PlanContext<'_>,
) -> Result<TranslatedRel> {
    expect_children(node, &children, 1)?;
    if has_conjuncts(node) {
        return Err(TranslateError::UnsupportedPlanNode {
            node_id: node.node_id,
            node_type: node.node_type,
            reason: "PROJECT_NODE conjunct row layout is ambiguous in v1",
        });
    }
    let child = children.into_iter().next().unwrap();
    translate_project_node(child, node, ctx)
}

/// Translates a flat preorder StarRocks plan into a Substrait relation tree.
pub(crate) fn translate_plan(
    plan: &TPlan,
    desc: &DescriptorTable,
    scan_paths: &ScanFilePaths,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    let mut ctx = PlanContext::new(desc, scan_paths, registry);
    plan.translate(&mut ctx)
}

/// Builds a Substrait read for a StarRocks scan tuple.
///
/// With `file_paths` present (FILE_SCAN broker ranges) it emits a `local_files`
/// parquet read so DuckDB's Substrait reader resolves the scan to
/// `parquet_scan(<paths>)`. v1 assumes parquet files whose column order matches
/// the scan tuple's slot order, which holds for `FILES()` `SELECT *`. Without
/// paths (e.g. HDFS scans) it falls back to a named-table read.
fn scan_rel(desc: &DescriptorTable, tuple_id: i32, file_paths: &[String]) -> Result<Rel> {
    let read_type = if file_paths.is_empty() {
        ReadType::NamedTable(NamedTable {
            names: desc.table_names_for_tuple(tuple_id)?,
            ..Default::default()
        })
    } else {
        ReadType::LocalFiles(LocalFiles {
            items: file_paths
                .iter()
                .map(|path| FileOrFiles {
                    path_type: Some(PathType::UriFile(path.clone())),
                    file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        })
    };
    Ok(Rel {
        rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
            base_schema: Some(desc.named_struct(tuple_id)?),
            read_type: Some(read_type),
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

    Ok(project_rel(child, expressions, output_tuples))
}

/// Adds a root projection over explicit fragment output expressions.
pub(crate) fn project_exprs(
    input: TranslatedRel,
    exprs: &[TExpr],
    desc: &DescriptorTable,
    registry: &mut ExtensionRegistry,
) -> Result<TranslatedRel> {
    // Root projections evaluate over already-translated inputs, so there are no
    // scan nodes to resolve file paths for.
    let scan_paths = ScanFilePaths::default();
    let mut ctx = PlanContext::new(desc, &scan_paths, registry);
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
    // A root projection over fragment output expressions keeps the input layout.
    let row_tuples = input.row_tuples.clone();
    Ok(project_rel(input, expressions, row_tuples))
}

/// Builds a Substrait project that emits exactly `expressions`.
///
/// The emit `output_mapping` selects the projected expressions, which sit after
/// the input columns, so the base offset is the input's carried `output_width`.
/// `row_tuples` is the output row layout (the project's own tuples, which may
/// reorder or differ from the input's).
fn project_rel(
    input: TranslatedRel,
    expressions: Vec<Expression>,
    row_tuples: Vec<i32>,
) -> TranslatedRel {
    let base = input.output_width as i32;
    let output_mapping = (base..base + expressions.len() as i32).collect();
    let output_width = expressions.len();
    TranslatedRel {
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
        row_tuples,
        output_width,
    }
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

    // A filter does not change the column layout, so the width passes through.
    let output_width = input.output_width;
    Ok(TranslatedRel {
        rel: Rel {
            rel_type: Some(rel::RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(input.rel)),
                condition: Some(Box::new(condition)),
                ..Default::default()
            }))),
        },
        row_tuples: input.row_tuples,
        output_width,
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
