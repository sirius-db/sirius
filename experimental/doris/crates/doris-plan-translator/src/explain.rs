//! Readable text for a translated plan.
//!
//! `substrait-explain` 0.9 cannot textify three things the translator emits and prints a
//! `!{...}` placeholder (plus an `Unimplemented` warning) for each: `local_files` reads,
//! decimal literals and `SingularOrList` (`IN`). Before formatting, [`render`] rewrites a
//! *copy* of the plan into display equivalents the formatter does handle:
//!
//! | emitted                                   | shown as                                        |
//! |-------------------------------------------|-------------------------------------------------|
//! | `ReadRel.local_files` (parquet paths)     | `Read["local_files:/path/a.parquet" => ...]`    |
//! | `Literal.decimal { value, p, s }`         | `1.00:decimal<16, 2>` (the formatter's own style)|
//! | `SingularOrList { value, options }`       | `in_list(value, option, ...):boolean?`          |
//! | `AggregateFunction.invocation = DISTINCT` | `count($3, distinct⇒[true])` (a function option)|
//!
//! The formatter also drops an aggregate's `invocation`, so a distinct measure is given a
//! display-only `distinct` option; without it `count(DISTINCT x)` and `count(x)` would read
//! the same.
//! `in_list` is registered as a function of the synthetic URN [`EXPLAIN_URN`] so the
//! `=== Extensions` header names it. The rewrites are display-only; the plan the engine
//! receives is untouched, and any warning the formatter still reports is a real gap.

use substrait::proto::aggregate_function::AggregationInvocation;
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::{self, RexType};
use substrait::proto::extensions::simple_extension_declaration::{ExtensionFunction, MappingType};
use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};
use substrait::proto::function_argument::ArgType;
use substrait::proto::read_rel::local_files::file_or_files::PathType;
use substrait::proto::read_rel::{NamedTable, ReadType};
use substrait::proto::rel::RelType;
use substrait::proto::r#type::{self, Kind, Nullability};
use substrait::proto::{Expression, FunctionArgument, FunctionOption, Plan, Rel, Type, plan_rel};
use substrait_explain::FormatError;

/// URN of the display-only `in_list` function [`render`] substitutes for `SingularOrList`.
pub const EXPLAIN_URN: &str = "extension:sirius:explain-only";

/// Marker wrapped around a decimal literal's text while it travels through the formatter
/// as a string literal; [`render`] strips it (and the string quotes) afterwards.
const DECIMAL_MARKER: &str = "@@decimal@@";

/// Formats `plan` with `substrait-explain` after the display rewrites described in the module
/// documentation. Returns the text and whatever the formatter still could not render.
pub fn render(plan: &Plan) -> (String, Vec<FormatError>) {
    let mut display = plan.clone();
    let mut rewriter = Rewriter::new(&display);
    for plan_rel in &mut display.relations {
        match &mut plan_rel.rel_type {
            Some(plan_rel::RelType::Root(root)) => {
                if let Some(input) = &mut root.input {
                    rewriter.rel(input);
                }
            }
            Some(plan_rel::RelType::Rel(rel)) => rewriter.rel(rel),
            None => {}
        }
    }
    if rewriter.in_list_used {
        rewriter.declare_in_list(&mut display);
    }
    let (text, warnings) = substrait_explain::format(&display);
    (unmark_decimals(&text), warnings)
}

/// Walks relations and expressions applying the three rewrites.
struct Rewriter {
    /// Whether any `SingularOrList` was rewritten (and `in_list` must be declared).
    in_list_used: bool,
    /// URN anchor reserved for [`EXPLAIN_URN`]: one past the plan's highest.
    urn_anchor: u32,
    /// Function anchor reserved for `in_list`: one past the plan's highest.
    in_list_anchor: u32,
}

impl Rewriter {
    fn new(plan: &Plan) -> Self {
        let urn_anchor = plan
            .extension_urns
            .iter()
            .map(|urn| urn.extension_urn_anchor)
            .max()
            .unwrap_or(0)
            + 1;
        let in_list_anchor = plan
            .extensions
            .iter()
            .filter_map(|declaration| match &declaration.mapping_type {
                Some(MappingType::ExtensionFunction(function)) => Some(function.function_anchor),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            + 1;
        Self {
            in_list_used: false,
            urn_anchor,
            in_list_anchor,
        }
    }

    /// Declares `in_list` under [`EXPLAIN_URN`] with the reserved anchors.
    fn declare_in_list(&self, plan: &mut Plan) {
        plan.extension_urns.push(SimpleExtensionUrn {
            extension_urn_anchor: self.urn_anchor,
            urn: EXPLAIN_URN.to_string(),
        });
        plan.extensions.push(SimpleExtensionDeclaration {
            mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                extension_urn_reference: self.urn_anchor,
                function_anchor: self.in_list_anchor,
                name: "in_list".to_string(),
            })),
        });
    }

    fn rel(&mut self, rel: &mut Rel) {
        let Some(rel_type) = &mut rel.rel_type else {
            return;
        };
        match rel_type {
            RelType::Read(read) => {
                if let Some(ReadType::LocalFiles(files)) = &read.read_type {
                    let paths: Vec<String> = files
                        .items
                        .iter()
                        .map(|item| match &item.path_type {
                            Some(PathType::UriPath(path))
                            | Some(PathType::UriPathGlob(path))
                            | Some(PathType::UriFile(path))
                            | Some(PathType::UriFolder(path)) => path.clone(),
                            None => "?".to_string(),
                        })
                        .collect();
                    read.read_type = Some(ReadType::NamedTable(NamedTable {
                        names: vec![format!("local_files:{}", paths.join(","))],
                        advanced_extension: None,
                    }));
                }
                self.expr_opt(read.filter.as_deref_mut());
                self.expr_opt(read.best_effort_filter.as_deref_mut());
            }
            RelType::Filter(filter) => {
                self.expr_opt(filter.condition.as_deref_mut());
                self.rel_opt(filter.input.as_deref_mut());
            }
            RelType::Fetch(fetch) => self.rel_opt(fetch.input.as_deref_mut()),
            RelType::Aggregate(aggregate) => {
                for expr in &mut aggregate.grouping_expressions {
                    self.expr(expr);
                }
                // `Grouping.grouping_expressions` is deprecated in favour of the rel-level
                // list above (which is what the translator emits); nothing to visit there.
                for measure in &mut aggregate.measures {
                    if let Some(function) = &mut measure.measure {
                        self.arguments(&mut function.arguments);
                        if function.invocation == AggregationInvocation::Distinct as i32 {
                            function.options.push(FunctionOption {
                                name: "distinct".to_string(),
                                preference: vec!["true".to_string()],
                            });
                        }
                    }
                    self.expr_opt(measure.filter.as_mut());
                }
                self.rel_opt(aggregate.input.as_deref_mut());
            }
            RelType::Sort(sort) => {
                for field in &mut sort.sorts {
                    self.expr_opt(field.expr.as_mut());
                }
                self.rel_opt(sort.input.as_deref_mut());
            }
            RelType::Join(join) => {
                self.expr_opt(join.expression.as_deref_mut());
                self.expr_opt(join.post_join_filter.as_deref_mut());
                self.rel_opt(join.left.as_deref_mut());
                self.rel_opt(join.right.as_deref_mut());
            }
            RelType::Project(project) => {
                for expr in &mut project.expressions {
                    self.expr(expr);
                }
                self.rel_opt(project.input.as_deref_mut());
            }
            RelType::Cross(cross) => {
                self.rel_opt(cross.left.as_deref_mut());
                self.rel_opt(cross.right.as_deref_mut());
            }
            RelType::Set(set) => {
                for input in &mut set.inputs {
                    self.rel(input);
                }
            }
            // The translator emits none of the remaining relation kinds; leave them alone
            // (anything unrenderable inside surfaces as a formatter warning).
            _ => {}
        }
    }

    fn rel_opt(&mut self, rel: Option<&mut Rel>) {
        if let Some(rel) = rel {
            self.rel(rel);
        }
    }

    fn expr_opt(&mut self, expr: Option<&mut Expression>) {
        if let Some(expr) = expr {
            self.expr(expr);
        }
    }

    fn arguments(&mut self, arguments: &mut [FunctionArgument]) {
        for argument in arguments {
            if let Some(ArgType::Value(value)) = &mut argument.arg_type {
                self.expr(value);
            }
        }
    }

    /// Rewrites children first so a `SingularOrList` nested in another is handled too.
    fn expr(&mut self, expr: &mut Expression) {
        let Some(rex_type) = &mut expr.rex_type else {
            return;
        };
        match rex_type {
            RexType::Literal(literal) => {
                if let Some(LiteralType::Decimal(decimal)) = &literal.literal_type {
                    let text = format!(
                        "{DECIMAL_MARKER}{}:decimal{}<{}, {}>{DECIMAL_MARKER}",
                        decimal_text(&decimal.value, decimal.scale),
                        if literal.nullable { "?" } else { "" },
                        decimal.precision,
                        decimal.scale
                    );
                    literal.literal_type = Some(LiteralType::String(text));
                    literal.nullable = false;
                }
            }
            RexType::ScalarFunction(function) => self.arguments(&mut function.arguments),
            RexType::WindowFunction(function) => self.arguments(&mut function.arguments),
            RexType::Cast(cast) => self.expr_opt(cast.input.as_deref_mut()),
            RexType::IfThen(if_then) => {
                for clause in &mut if_then.ifs {
                    self.expr_opt(clause.r#if.as_mut());
                    self.expr_opt(clause.then.as_mut());
                }
                self.expr_opt(if_then.r#else.as_deref_mut());
            }
            RexType::SwitchExpression(switch) => {
                self.expr_opt(switch.r#match.as_deref_mut());
                for clause in &mut switch.ifs {
                    self.expr_opt(clause.then.as_mut());
                }
                self.expr_opt(switch.r#else.as_deref_mut());
            }
            RexType::SingularOrList(list) => {
                self.expr_opt(list.value.as_deref_mut());
                for option in &mut list.options {
                    self.expr(option);
                }
                let mut arguments = Vec::with_capacity(list.options.len() + 1);
                arguments.extend(list.value.take().map(|value| *value));
                arguments.append(&mut list.options);
                self.in_list_used = true;
                expr.rex_type = Some(RexType::ScalarFunction(expression::ScalarFunction {
                    function_reference: self.in_list_anchor,
                    arguments: arguments
                        .into_iter()
                        .map(|value| FunctionArgument {
                            arg_type: Some(ArgType::Value(value)),
                        })
                        .collect(),
                    output_type: Some(Type {
                        kind: Some(Kind::Bool(r#type::Boolean {
                            nullability: Nullability::Nullable as i32,
                            ..Default::default()
                        })),
                    }),
                    ..Default::default()
                }));
            }
            RexType::MultiOrList(list) => {
                for value in &mut list.value {
                    self.expr(value);
                }
                for record in &mut list.options {
                    for field in &mut record.fields {
                        self.expr(field);
                    }
                }
            }
            // Leaves (and the deprecated `Enum` variant): nothing to rewrite.
            _ => {}
        }
    }
}

/// Renders a little-endian two's-complement decimal as `-12.34` with exactly `scale`
/// fraction digits.
fn decimal_text(value: &[u8], scale: i32) -> String {
    let mut bytes = [0u8; 16];
    let len = value.len().min(16);
    bytes[..len].copy_from_slice(&value[..len]);
    if len < 16 && value.last().is_some_and(|byte| byte & 0x80 != 0) {
        bytes[len..].fill(0xff);
    }
    let unscaled = i128::from_le_bytes(bytes);
    let digits = unscaled.unsigned_abs().to_string();
    let scale = scale.max(0) as usize;
    let mut text = String::new();
    if unscaled < 0 {
        text.push('-');
    }
    if scale == 0 {
        text.push_str(&digits);
    } else if digits.len() > scale {
        let (int_part, frac_part) = digits.split_at(digits.len() - scale);
        text.push_str(int_part);
        text.push('.');
        text.push_str(frac_part);
    } else {
        text.push_str("0.");
        text.extend(std::iter::repeat_n('0', scale - digits.len()));
        text.push_str(&digits);
    }
    text
}

/// Turns `'@@decimal@@1.00:decimal<16, 2>@@decimal@@'` back into `1.00:decimal<16, 2>`.
fn unmark_decimals(text: &str) -> String {
    let open = format!("'{DECIMAL_MARKER}");
    let close = format!("{DECIMAL_MARKER}'");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(&open) {
        out.push_str(&rest[..start]);
        let after = &rest[start + open.len()..];
        match after.find(&close) {
            Some(end) => {
                out.push_str(&after[..end]);
                rest = &after[end + close.len()..];
            }
            None => {
                out.push_str(&rest[start..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use substrait::proto::aggregate_rel::Measure;
    use substrait::proto::expression::literal::Decimal;
    use substrait::proto::expression::{Literal, SingularOrList};
    use substrait::proto::read_rel::LocalFiles;
    use substrait::proto::read_rel::local_files::FileOrFiles;
    use substrait::proto::read_rel::local_files::file_or_files::{FileFormat, ParquetReadOptions};
    use substrait::proto::{
        AggregateFunction, AggregateRel, FilterRel, NamedStruct, PlanRel, ProjectRel, ReadRel,
        RelRoot,
    };

    use super::*;

    fn i32_type() -> Type {
        Type {
            kind: Some(Kind::I32(r#type::I32 {
                nullability: Nullability::Nullable as i32,
                ..Default::default()
            })),
        }
    }

    fn field(index: i32) -> Expression {
        use substrait::proto::expression::field_reference::{ReferenceType, RootType};
        use substrait::proto::expression::reference_segment::{ReferenceType as Seg, StructField};
        use substrait::proto::expression::{FieldReference, ReferenceSegment};
        Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                    reference_type: Some(Seg::StructField(Box::new(StructField {
                        field: index,
                        child: None,
                    }))),
                })),
                root_type: Some(RootType::RootReference(Default::default())),
            }))),
        }
    }

    fn decimal(unscaled: i128, precision: i32, scale: i32) -> Expression {
        Expression {
            rex_type: Some(RexType::Literal(Literal {
                literal_type: Some(LiteralType::Decimal(Decimal {
                    value: unscaled.to_le_bytes().to_vec(),
                    precision,
                    scale,
                })),
                ..Default::default()
            })),
        }
    }

    /// `Aggregate[count(distinct $0)] over Project[$0, 7] over Filter[in_list($0, 1.50, -0.05)]
    /// over Read(local_files)`.
    fn plan() -> Plan {
        let read = Rel {
            rel_type: Some(RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: vec!["a".to_string()],
                    r#struct: Some(r#type::Struct {
                        types: vec![i32_type()],
                        nullability: Nullability::Required as i32,
                        ..Default::default()
                    }),
                }),
                read_type: Some(ReadType::LocalFiles(LocalFiles {
                    items: vec![FileOrFiles {
                        path_type: Some(PathType::UriFile("/data/t.parquet".to_string())),
                        file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        let filter = Rel {
            rel_type: Some(RelType::Filter(Box::new(FilterRel {
                input: Some(Box::new(read)),
                condition: Some(Box::new(Expression {
                    rex_type: Some(RexType::SingularOrList(Box::new(SingularOrList {
                        value: Some(Box::new(field(0))),
                        options: vec![decimal(150, 5, 2), decimal(-5, 5, 2)],
                    }))),
                })),
                ..Default::default()
            }))),
        };
        let project = Rel {
            rel_type: Some(RelType::Project(Box::new(ProjectRel {
                input: Some(Box::new(filter)),
                expressions: vec![decimal(7, 5, 0)],
                ..Default::default()
            }))),
        };
        let aggregate = Rel {
            rel_type: Some(RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(project)),
                measures: vec![Measure {
                    measure: Some(AggregateFunction {
                        function_reference: 3,
                        arguments: vec![FunctionArgument {
                            arg_type: Some(ArgType::Value(field(0))),
                        }],
                        invocation: AggregationInvocation::Distinct as i32,
                        output_type: Some(Type {
                            kind: Some(Kind::I64(r#type::I64 {
                                nullability: Nullability::Required as i32,
                                ..Default::default()
                            })),
                        }),
                        ..Default::default()
                    }),
                    filter: None,
                }],
                ..Default::default()
            }))),
        };
        Plan {
            extension_urns: vec![SimpleExtensionUrn {
                extension_urn_anchor: 1,
                urn: "extension:io.substrait:functions_aggregate_generic".to_string(),
            }],
            extensions: vec![SimpleExtensionDeclaration {
                mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
                    extension_urn_reference: 1,
                    function_anchor: 3,
                    name: "count".to_string(),
                })),
            }],
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(aggregate),
                    names: vec!["n".to_string()],
                })),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn renders_everything_the_translator_emits_without_warnings() {
        let original = plan();
        let (text, warnings) = render(&original);
        assert!(warnings.is_empty(), "{text}\n{warnings:?}");
        assert!(
            text.contains("Read[\"local_files:/data/t.parquet\" => a:i32?]"),
            "{text}"
        );
        assert!(
            text.contains("Filter[in_list($0, 1.50:decimal<5, 2>, -0.05:decimal<5, 2>):boolean?"),
            "{text}"
        );
        assert!(text.contains("Project[$0, 7:decimal<5, 0>]"), "{text}");
        assert!(
            text.contains("Aggregate[_ => count($0, distinct⇒[true]):i64]"),
            "{text}"
        );
        assert!(!text.contains(DECIMAL_MARKER), "{text}");
        // Display only: the plan handed to the engine is unchanged.
        assert_eq!(original, plan());
    }

    #[test]
    fn in_list_anchor_does_not_collide_with_existing_functions() {
        // The fixture declares `count` as #3 under URN @1; `in_list` lands past both.
        let (text, warnings) = render(&plan());
        assert!(warnings.is_empty(), "{text}\n{warnings:?}");
        assert!(text.contains("#  4 @  2: in_list"), "{text}");
        assert!(text.contains(EXPLAIN_URN), "{text}");
    }

    #[test]
    fn decimal_text_handles_sign_scale_and_short_buffers() {
        assert_eq!(decimal_text(&100i128.to_le_bytes(), 2), "1.00");
        assert_eq!(decimal_text(&5i128.to_le_bytes(), 2), "0.05");
        assert_eq!(decimal_text(&(-5i128).to_le_bytes(), 2), "-0.05");
        assert_eq!(decimal_text(&24i128.to_le_bytes(), 0), "24");
        assert_eq!(decimal_text(&123456i128.to_le_bytes(), 10), "0.0000123456");
        // Fewer than 16 bytes: sign-extended.
        assert_eq!(decimal_text(&(-1i16).to_le_bytes(), 1), "-0.1");
        assert_eq!(decimal_text(&[], 1), "0.0");
    }

    #[test]
    fn unmark_decimals_strips_markers_and_quotes_only() {
        assert_eq!(
            unmark_decimals("f('@@decimal@@1.5:decimal<2, 1>@@decimal@@', 'x')"),
            "f(1.5:decimal<2, 1>, 'x')"
        );
        assert_eq!(unmark_decimals("plain 'text'"), "plain 'text'");
        assert_eq!(
            unmark_decimals("'@@decimal@@dangling"),
            "'@@decimal@@dangling"
        );
    }
}
