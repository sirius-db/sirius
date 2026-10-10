//! Translates fragments the StarRocks FE actually sent, recorded from a TPC-H SF1000 run.
//!
//! The fixtures under `tests/fixtures/tpch-sf1000-*` are `TExecPlanFragmentParams` in thrift
//! binary protocol, re-encoded from `SIRIUS_CN_DUMP_FRAGMENTS` dumps by
//! `tests/fixtures/debug_to_thrift.py`. Each directory holds the fragments of the queries one
//! change unblocked.

use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslateError, TranslatedPlan};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::plan_nodes::{TAggregationNode, TJoinOp, TPlanNodeType};
use substrait::proto::join_rel::JoinType;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};

/// Two-phase aggregation fragments that used to fail with "two-phase aggregation supports SUM
/// only".
const TWO_PHASE_AGG: &str = "tpch-sf1000-two-phase-agg";
/// Fragments with RIGHT SEMI joins, which used to fail with "hash join type is unsupported".
const SEMI_ANTI_JOIN: &str = "tpch-sf1000-semi-anti-join";
/// Aggregation fragments whose tuples pin the FE's column-order contract.
const AGG_COLUMN_ORDER: &str = "tpch-sf1000-agg-column-order";
/// Two-phase AVG fragments, which used to fail with "two-phase aggregation supports SUM, COUNT,
/// MIN and MAX only".
const TWO_PHASE_AVG: &str = "tpch-sf1000-two-phase-avg";
/// Fragments whose slot refs name a tuple the input row doesn't carry, which used to fail with
/// "slot N (tuple T) is not part of the row layout". The BE finds a slot ref's column by slot id
/// alone (`Chunk::get_column_by_slot_id`), so those plans are valid.
const SLOT_LAYOUT: &str = "tpch-sf1000-slot-layout";

/// Reads a recorded fragment from a fixture directory.
fn load(dir: &str, name: &str) -> TExecPlanFragmentParams {
    let path = format!(
        "{}/tests/fixtures/{dir}/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut protocol = TBinaryInputProtocol::new(bytes.as_slice(), true);
    TExecPlanFragmentParams::read_from_in_protocol(&mut protocol)
        .unwrap_or_else(|err| panic!("{path}: {err}"))
}

/// Translates a recorded fragment with every exchange bound to a stream, the way the CN binds
/// them. The sender's column names only have to match the exchange row's width.
fn translate(dir: &str, name: &str) -> Result<TranslatedPlan, TranslateError> {
    translate_after(dir, name, &[])
}

/// Like [`translate`], but binds each listed exchange to the output names of its translated
/// sender fragment, as the CN does, instead of one name per descriptor slot.
fn translate_after(
    dir: &str,
    name: &str,
    senders: &[(i32, &TranslatedPlan)],
) -> Result<TranslatedPlan, TranslateError> {
    let params = load(dir, name);
    let slots = params
        .desc_tbl
        .as_ref()
        .and_then(|desc| desc.slot_descriptors.as_deref())
        .unwrap_or_default();
    let inputs: Vec<ExchangeInput> = params
        .fragment
        .as_ref()
        .and_then(|fragment| fragment.plan.as_ref())
        .unwrap()
        .nodes
        .iter()
        .filter(|node| node.node_type == TPlanNodeType::EXCHANGE_NODE)
        .map(|node| {
            let tuples = &node.exchange_node.as_ref().unwrap().input_row_tuples;
            let width = slots
                .iter()
                .filter(|slot| slot.is_materialized != Some(false))
                .filter(|slot| slot.parent.is_some_and(|parent| tuples.contains(&parent)))
                .count();
            let names = match senders.iter().find(|(id, _)| *id == node.node_id) {
                Some((_, sender)) => sender.output_names.clone(),
                None => (0..width).map(|index| format!("c{index}")).collect(),
            };
            ExchangeInput {
                node_id: node.node_id,
                stream_view: format!("sirius_stream_{}", node.node_id),
                names,
            }
        })
        .collect();
    PlanTranslator::new().translate_fragment_with_exchange_inputs(&params, &inputs)
}

/// Returns every extension function name the plan declares.
fn function_names(plan: &TranslatedPlan) -> Vec<String> {
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    plan.plan
        .extensions
        .iter()
        .filter_map(|declaration| match declaration.mapping_type.as_ref() {
            Some(MappingType::ExtensionFunction(function)) => Some(function.name.clone()),
            _ => None,
        })
        .collect()
}

/// Returns the DuckDB column types the plan declares for exchange `node_id`'s stream.
fn stream_types(plan: &TranslatedPlan, node_id: i32) -> Vec<String> {
    plan.stream_inputs
        .iter()
        .find(|input| input.node_id == node_id)
        .unwrap_or_else(|| panic!("no stream declared for exchange {node_id}"))
        .columns
        .iter()
        .map(|column| column.ty.clone())
        .collect()
}

/// q13's merge fragment (SORT <- AGG merge count <- EXCHANGE) sums the partial counts.
#[test]
fn q13_merge_count_sums_partial_counts() {
    let plan = translate(TWO_PHASE_AGG, "q13-merge").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"sum".to_string()), "{names:?}");
    assert!(!names.contains(&"count".to_string()), "{names:?}");
    // c_count (key) and the partial count(*), both BIGINT.
    assert_eq!(stream_types(&plan, 10), ["BIGINT", "BIGINT"]);
}

/// q04's and q21's merge fragments also sum partial counts. Their partial fragments, which also
/// need RIGHT SEMI joins, are in [`SEMI_ANTI_JOIN`].
#[test]
fn q04_and_q21_merge_count_sums_partial_counts() {
    for (name, exchange) in [("q04-merge", 9), ("q21-merge", 25)] {
        let plan = translate(TWO_PHASE_AGG, name).unwrap();
        let names = function_names(&plan);
        assert!(names.contains(&"sum".to_string()), "{name}: {names:?}");
        assert!(!names.contains(&"count".to_string()), "{name}: {names:?}");
        assert_eq!(
            stream_types(&plan, exchange),
            ["VARCHAR", "BIGINT"],
            "{name}"
        );
    }
}

/// q13's partial fragment counts orders per customer (one-phase), then partially counts
/// customers per order count.
#[test]
fn q13_partial_count_translates() {
    let plan = translate(TWO_PHASE_AGG, "q13-partial").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"count".to_string()), "{names:?}");
}

/// q02's merge fragment re-applies MIN to the partial minimums, which cross the exchange as
/// DECIMAL(15,2) like the input.
#[test]
fn q02_merge_min_translates() {
    let plan = translate(TWO_PHASE_AGG, "q02-merge").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"min".to_string()), "{names:?}");
    assert_eq!(stream_types(&plan, 15), ["INTEGER", "DECIMAL(15,2)"]);
}

/// q02's partial fragment joins and partially aggregates MIN(ps_supplycost) per part.
#[test]
fn q02_partial_min_translates() {
    let plan = translate(TWO_PHASE_AGG, "q02-partial").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"min".to_string()), "{names:?}");
}

/// q15's global MAX over the per-supplier revenue: a partial MAX above a merge SUM in one
/// fragment. The merge MAX fragment never reached the CN in the recorded run (the FE stopped
/// deploying once this one failed), so it has no fixture.
#[test]
fn q15_partial_max_translates() {
    let plan = translate(TWO_PHASE_AGG, "q15-partial-max").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"max".to_string()), "{names:?}");
    assert!(names.contains(&"sum".to_string()), "{names:?}");
    // supplier key and the partial revenue sum, the FE's DECIMAL128(38,4).
    assert_eq!(stream_types(&plan, 8), ["INTEGER", "DECIMAL(38,4)"]);
}

/// Join ops of the recorded fragment's hash joins, in plan order.
fn join_ops(dir: &str, name: &str) -> Vec<TJoinOp> {
    load(dir, name)
        .fragment
        .and_then(|fragment| fragment.plan)
        .unwrap()
        .nodes
        .iter()
        .filter_map(|node| node.hash_join_node.as_ref().map(|join| join.join_op))
        .collect()
}

/// Every Substrait join in the plan, outermost first, with the functions its condition calls.
fn joins(plan: &TranslatedPlan) -> Vec<(JoinType, Vec<String>)> {
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    use substrait::proto::{Expression, Rel, expression, function_argument, plan_rel, rel};

    fn called(expr: &Expression, out: &mut Vec<u32>) {
        if let Some(expression::RexType::ScalarFunction(function)) = &expr.rex_type {
            out.push(function.function_reference);
            for argument in &function.arguments {
                if let Some(function_argument::ArgType::Value(value)) = &argument.arg_type {
                    called(value, out);
                }
            }
        }
    }
    fn walk(rel: &Rel, out: &mut Vec<(JoinType, Vec<u32>)>) {
        let inputs: Vec<&Rel> = match rel.rel_type.as_ref().unwrap() {
            rel::RelType::Join(join) => {
                let mut anchors = Vec::new();
                called(join.expression.as_ref().unwrap(), &mut anchors);
                out.push((JoinType::try_from(join.r#type).unwrap(), anchors));
                vec![
                    join.left.as_deref().unwrap(),
                    join.right.as_deref().unwrap(),
                ]
            }
            rel::RelType::Project(project) => vec![project.input.as_deref().unwrap()],
            rel::RelType::Filter(filter) => vec![filter.input.as_deref().unwrap()],
            rel::RelType::Aggregate(aggregate) => vec![aggregate.input.as_deref().unwrap()],
            rel::RelType::Sort(sort) => vec![sort.input.as_deref().unwrap()],
            rel::RelType::Fetch(fetch) => vec![fetch.input.as_deref().unwrap()],
            _ => Vec::new(),
        };
        for input in inputs {
            walk(input, out);
        }
    }

    let names: std::collections::HashMap<u32, String> = plan
        .plan
        .extensions
        .iter()
        .filter_map(|declaration| match declaration.mapping_type.as_ref() {
            Some(MappingType::ExtensionFunction(function)) => {
                Some((function.function_anchor, function.name.clone()))
            }
            _ => None,
        })
        .collect();
    let Some(plan_rel::RelType::Root(root)) = plan.plan.relations[0].rel_type.as_ref() else {
        panic!("expected a root relation");
    };
    let mut found = Vec::new();
    walk(root.input.as_ref().unwrap(), &mut found);
    found
        .into_iter()
        .map(|(kind, anchors)| {
            let called = anchors.iter().map(|anchor| names[anchor].clone()).collect();
            (kind, called)
        })
        .collect()
}

/// q04's partial fragment: `EXISTS (lineitem ...)` is a RIGHT SEMI join keeping the orders
/// side, under the partial COUNT.
#[test]
fn q04_right_semi_join_translates() {
    assert_eq!(
        join_ops(SEMI_ANTI_JOIN, "q04-partial"),
        [TJoinOp::RIGHT_SEMI_JOIN]
    );
    let plan = translate(SEMI_ANTI_JOIN, "q04-partial").unwrap();
    let kinds: Vec<_> = joins(&plan).into_iter().map(|(kind, _)| kind).collect();
    assert_eq!(kinds, [JoinType::RightSemi]);
    assert!(function_names(&plan).contains(&"count".to_string()));
}

/// q20's root fragment: `s_suppkey IN (...)` is a RIGHT SEMI join keeping the supplier side.
#[test]
fn q20_right_semi_join_translates() {
    assert_eq!(
        join_ops(SEMI_ANTI_JOIN, "q20-root"),
        [TJoinOp::RIGHT_SEMI_JOIN]
    );
    let plan = translate(SEMI_ANTI_JOIN, "q20-root").unwrap();
    let kinds: Vec<_> = joins(&plan).into_iter().map(|(kind, _)| kind).collect();
    assert_eq!(kinds, [JoinType::RightSemi]);
}

/// q21's partial fragment: `EXISTS (l2 ...)` is a RIGHT SEMI join and `NOT EXISTS (l3 ...)` a
/// RIGHT ANTI join, both with `l_suppkey <> l1.l_suppkey` as an extra join conjunct. The
/// conjunct has to stay in each join's condition: as a filter after the join it would test
/// one matching row instead of whether any row matches.
#[test]
fn q21_semi_and_anti_joins_keep_their_other_conjuncts() {
    assert_eq!(
        join_ops(SEMI_ANTI_JOIN, "q21-partial"),
        [
            TJoinOp::RIGHT_SEMI_JOIN,
            TJoinOp::INNER_JOIN,
            TJoinOp::RIGHT_ANTI_JOIN
        ]
    );
    let plan = translate(SEMI_ANTI_JOIN, "q21-partial").unwrap();
    let joins = joins(&plan);
    let kinds: Vec<_> = joins.iter().map(|(kind, _)| *kind).collect();
    // The RIGHT ANTI join is a RIGHT join filtered on the probe key being NULL.
    assert_eq!(
        kinds,
        [JoinType::RightSemi, JoinType::Inner, JoinType::Right]
    );
    for (kind, called) in [&joins[0], &joins[2]] {
        assert!(
            called.contains(&"equal".to_string()) && called.contains(&"not_equal".to_string()),
            "{kind:?}: {called:?}"
        );
    }
    assert!(function_names(&plan).contains(&"is_null".to_string()));
}

/// The aggregation node `node_id` of a recorded fragment.
fn aggregation(params: &TExecPlanFragmentParams, node_id: i32) -> &TAggregationNode {
    params
        .fragment
        .as_ref()
        .and_then(|fragment| fragment.plan.as_ref())
        .unwrap()
        .nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .and_then(|node| node.agg_node.as_ref())
        .unwrap_or_else(|| panic!("no aggregation node {node_id}"))
}

/// A tuple's materialized slot ids in descriptor (wire) order.
fn descriptor_order(params: &TExecPlanFragmentParams, tuple_id: i32) -> Vec<i32> {
    params
        .desc_tbl
        .as_ref()
        .and_then(|desc| desc.slot_descriptors.as_deref())
        .unwrap_or_default()
        .iter()
        .filter(|slot| slot.parent == Some(tuple_id) && slot.is_materialized != Some(false))
        .map(|slot| slot.id.unwrap())
        .collect()
}

/// Column `slot` lands in when a tuple's columns follow `order`.
fn column_of(order: &[i32], slot: i32) -> usize {
    order.iter().position(|id| *id == slot).unwrap()
}

/// Every relation of the plan, outermost first.
fn rels(plan: &TranslatedPlan) -> Vec<&substrait::proto::Rel> {
    use substrait::proto::{Rel, plan_rel, rel};
    fn walk<'a>(rel: &'a Rel, out: &mut Vec<&'a Rel>) {
        out.push(rel);
        let inputs: Vec<&Rel> = match rel.rel_type.as_ref().unwrap() {
            rel::RelType::Join(join) => vec![
                join.left.as_deref().unwrap(),
                join.right.as_deref().unwrap(),
            ],
            rel::RelType::Project(project) => vec![project.input.as_deref().unwrap()],
            rel::RelType::Filter(filter) => vec![filter.input.as_deref().unwrap()],
            rel::RelType::Aggregate(aggregate) => vec![aggregate.input.as_deref().unwrap()],
            rel::RelType::Sort(sort) => vec![sort.input.as_deref().unwrap()],
            rel::RelType::Fetch(fetch) => vec![fetch.input.as_deref().unwrap()],
            _ => Vec::new(),
        };
        for input in inputs {
            walk(input, out);
        }
    }
    let Some(plan_rel::RelType::Root(root)) = plan.plan.relations[0].rel_type.as_ref() else {
        panic!("expected a root relation");
    };
    let mut found = Vec::new();
    walk(root.input.as_ref().unwrap(), &mut found);
    found
}

/// The plan's outermost aggregate relation.
fn top_aggregate(plan: &TranslatedPlan) -> &substrait::proto::AggregateRel {
    use substrait::proto::rel;
    rels(plan)
        .into_iter()
        .find_map(|rel| match rel.rel_type.as_ref() {
            Some(rel::RelType::Aggregate(aggregate)) => Some(aggregate.as_ref()),
            _ => None,
        })
        .expect("no aggregate relation")
}

/// The input field a field-reference expression reads, looking through a cast (a decimal sum's
/// argument is cast to FP64).
fn field(expr: &substrait::proto::Expression) -> usize {
    use substrait::proto::expression::{self, field_reference, reference_segment};
    let selection = match expr.rex_type.as_ref() {
        Some(expression::RexType::Selection(selection)) => selection,
        Some(expression::RexType::Cast(cast)) => return field(cast.input.as_ref().unwrap()),
        _ => panic!("expected a field reference, got {expr:?}"),
    };
    let Some(field_reference::ReferenceType::DirectReference(segment)) =
        selection.reference_type.as_ref()
    else {
        panic!("expected a direct reference");
    };
    let Some(reference_segment::ReferenceType::StructField(field)) =
        segment.reference_type.as_ref()
    else {
        panic!("expected a struct field reference");
    };
    field.field as usize
}

/// The input field each measure of an aggregate reads.
fn measure_fields(aggregate: &substrait::proto::AggregateRel) -> Vec<usize> {
    use substrait::proto::function_argument::ArgType;
    aggregate
        .measures
        .iter()
        .map(|measure| {
            let Some(ArgType::Value(argument)) =
                &measure.measure.as_ref().unwrap().arguments[0].arg_type
            else {
                panic!("expected a value argument");
            };
            field(argument)
        })
        .collect()
}

/// The type kind of each measure's output, as a short name.
fn measure_kinds(aggregate: &substrait::proto::AggregateRel) -> Vec<&'static str> {
    use substrait::proto::r#type::Kind;
    aggregate
        .measures
        .iter()
        .map(|measure| {
            let output = measure.measure.as_ref().unwrap().output_type.as_ref();
            match output.and_then(|ty| ty.kind.as_ref()) {
                Some(Kind::I64(_)) => "i64",
                Some(Kind::Fp64(_)) => "fp64",
                Some(Kind::Decimal(_)) => "decimal",
                other => panic!("unexpected measure type {other:?}"),
            }
        })
        .collect()
}

/// The FE's column-order contract for an aggregation that doesn't finalize: grouping key i is
/// slot i of its intermediate tuple and aggregate i slot `keys + i`, in descriptor order. q10's
/// partial aggregation lists its seven keys out of slot-id order, and its sink hash-partitions
/// on them by slot, so the columns they resolve to show which order the translator used.
#[test]
fn q10_partial_aggregation_follows_its_intermediate_tuple_order() {
    let params = load(AGG_COLUMN_ORDER, "q10-partial");
    let agg = aggregation(&params, 17);
    assert!(!agg.need_finalize);
    let order = descriptor_order(&params, agg.intermediate_tuple_id);
    assert_eq!(order, [1, 2, 6, 5, 35, 3, 8, 39]);

    let plan = translate(AGG_COLUMN_ORDER, "q10-partial").unwrap();
    // The sink partitions on slots 1, 2, 6, 5, 35, 3, 8, which are columns 0..7 in descriptor
    // order (by slot id they would be 0, 1, 4, 3, 6, 2, 5).
    assert_eq!(plan.output_partition_columns, Some((0..7).collect()));
    // The one aggregate, the revenue sum, is column 7 = slot 39, sent as DECIMAL(38,4).
    assert_eq!(plan.output_names.len(), 8);
    assert_eq!(measure_kinds(top_aggregate(&plan)), ["decimal"]);
}

/// q22's partial aggregation emits `count(*)` and then `sum(c_acctbal)` after its one key,
/// typed by the partial-state rule as BIGINT and DECIMAL(38,2), in aggregate order.
#[test]
fn q22_partial_aggregates_follow_their_intermediate_tuple_order() {
    let params = load(AGG_COLUMN_ORDER, "q22-partial");
    let agg = aggregation(&params, 14);
    assert!(!agg.need_finalize);
    assert_eq!(
        descriptor_order(&params, agg.intermediate_tuple_id),
        [37, 38, 39]
    );

    let plan = translate(AGG_COLUMN_ORDER, "q22-partial").unwrap();
    assert_eq!(plan.output_partition_columns, Some(vec![0]));
    assert_eq!(measure_kinds(top_aggregate(&plan)), ["i64", "decimal"]);
}

/// The contract for a finalizing aggregation, on its output tuple. q10's merge aggregation reads
/// the exchange in the order its partial step sends (the same descriptor order), and the sort
/// above it reads the output tuple's slots 39, 1, 2, 3, 5, 6, 8, 35, which resolve to their
/// positions in descriptor order.
#[test]
fn q10_merge_aggregation_follows_its_output_tuple_order() {
    use substrait::proto::rel;
    let params = load(AGG_COLUMN_ORDER, "q10-merge");
    let agg = aggregation(&params, 19);
    assert!(agg.need_finalize);
    let order = descriptor_order(&params, agg.output_tuple_id);
    assert_eq!(order, [1, 2, 6, 5, 35, 3, 8, 39]);

    let plan = translate(AGG_COLUMN_ORDER, "q10-merge").unwrap();
    let aggregate = top_aggregate(&plan);
    // The keys and the revenue sum read the exchange in descriptor order.
    let keys: Vec<_> = aggregate.grouping_expressions.iter().map(field).collect();
    assert_eq!(keys, (0..7).collect::<Vec<_>>());
    assert_eq!(measure_fields(aggregate), [7]);
    assert_eq!(
        stream_types(&plan, 18),
        [
            "INTEGER",
            "VARCHAR",
            "DECIMAL(15,2)",
            "VARCHAR",
            "VARCHAR",
            "VARCHAR",
            "VARCHAR",
            "DECIMAL(38,4)"
        ]
    );

    let projection = rels(&plan)
        .into_iter()
        .find_map(|rel| match rel.rel_type.as_ref() {
            Some(rel::RelType::Sort(sort)) => match sort.input.as_ref()?.rel_type.as_ref() {
                Some(rel::RelType::Project(project)) => Some(project.as_ref()),
                _ => None,
            },
            _ => None,
        })
        .expect("a sort over its sort-tuple projection");
    let read: Vec<_> = projection.expressions.iter().map(field).collect();
    let expected: Vec<_> = [39, 1, 2, 3, 5, 6, 8, 35]
        .into_iter()
        .map(|slot| column_of(&order, slot))
        .collect();
    assert_eq!(expected, [7, 0, 1, 5, 3, 2, 6, 4]);
    assert_eq!(read, expected);
}

/// q22's merge aggregation reads aggregate i's partial state from exchange column `keys + i`:
/// the count sum from column 1 and the balance sum from column 2.
#[test]
fn q22_merge_aggregates_read_their_partial_states_in_order() {
    let params = load(AGG_COLUMN_ORDER, "q22-merge");
    let agg = aggregation(&params, 16);
    assert!(agg.need_finalize);
    assert_eq!(descriptor_order(&params, agg.output_tuple_id), [37, 38, 39]);

    let plan = translate(AGG_COLUMN_ORDER, "q22-merge").unwrap();
    assert_eq!(
        stream_types(&plan, 15),
        ["VARCHAR", "BIGINT", "DECIMAL(38,2)"]
    );
    assert_eq!(measure_fields(top_aggregate(&plan)), [1, 2]);
}

/// The function name of each measure of an aggregate, in order.
fn measure_names(plan: &TranslatedPlan, aggregate: &substrait::proto::AggregateRel) -> Vec<String> {
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    aggregate
        .measures
        .iter()
        .map(|measure| {
            let anchor = measure.measure.as_ref().unwrap().function_reference;
            plan.plan
                .extensions
                .iter()
                .find_map(|declaration| match declaration.mapping_type.as_ref() {
                    Some(MappingType::ExtensionFunction(function))
                        if function.function_anchor == anchor =>
                    {
                        Some(function.name.clone())
                    }
                    _ => None,
                })
                .unwrap()
        })
        .collect()
}

/// Translates a recorded partial fragment and the merge fragment it sends to through exchange
/// `exchange`, binding that exchange to the partial fragment's output names as the CN does.
fn translate_two_phase(
    partial: &str,
    merge: &str,
    exchange: i32,
) -> (TranslatedPlan, TranslatedPlan) {
    let partial = translate(TWO_PHASE_AVG, partial).unwrap();
    let merge = translate_after(TWO_PHASE_AVG, merge, &[(exchange, &partial)]).unwrap();
    (partial, merge)
}

/// q01 averages three DECIMAL64(15,2) columns (final type DECIMAL128(38,8), FP64 here) next to
/// four sums and a count. The partial step ships each AVG as a sum and a count, so its 8
/// aggregates become 11 columns after the 2 keys, and the merge step reads exactly those
/// columns: its stream is the partial step's output, column for column, not the descriptor's
/// 10 slots with their VARBINARY AVG states.
#[test]
fn q01_two_phase_avg_partial_and_merge_agree_on_the_exchange_row() {
    let (partial, merge) = translate_two_phase("q01-partial", "q01-merge", 3);
    let aggregate = top_aggregate(&partial);
    assert_eq!(
        measure_names(&partial, aggregate),
        [
            "sum", "sum", "sum", "sum", "sum", "count", "sum", "count", "sum", "count", "count"
        ]
    );
    // The four sums ship as the FE's DECIMAL128(38,s); each AVG as an FP64 sum and a count.
    assert_eq!(
        measure_kinds(aggregate),
        [
            "decimal", "decimal", "decimal", "decimal", "fp64", "i64", "fp64", "i64", "fp64",
            "i64", "i64"
        ]
    );
    assert_eq!(partial.output_names.len(), 13);
    assert_eq!(
        stream_types(&merge, 3),
        [
            "VARCHAR",
            "VARCHAR",
            "DECIMAL(38,2)",
            "DECIMAL(38,2)",
            "DECIMAL(38,4)",
            "DECIMAL(38,6)",
            "DOUBLE",
            "BIGINT",
            "DOUBLE",
            "BIGINT",
            "DOUBLE",
            "BIGINT",
            "BIGINT"
        ]
    );
    // The merge sums every shipped column once: the four sums, each AVG's sum and count, and
    // the count, which reads column 12 now that the AVG counts sit before it.
    let merged = top_aggregate(&merge);
    assert_eq!(measure_fields(merged), (2..13).collect::<Vec<_>>());
    assert!(
        measure_names(&merge, merged)
            .iter()
            .all(|name| name == "sum")
    );
    // The merge emits its output tuple again: 2 keys and 8 aggregates, under the sort.
    assert_eq!(merge.output_names.len(), 10);
}

/// q17's `0.2 * avg(l_quantity)` per part: a grouped AVG whose merge feeds a join in the same
/// fragment.
#[test]
fn q17_two_phase_avg_partial_and_merge_agree_on_the_exchange_row() {
    let (partial, merge) = translate_two_phase("q17-partial", "q17-merge", 4);
    assert_eq!(measure_kinds(top_aggregate(&partial)), ["fp64", "i64"]);
    // The sink still hash-partitions on the part key.
    assert_eq!(partial.output_partition_columns, Some(vec![0]));
    assert_eq!(stream_types(&merge, 4), ["INTEGER", "DOUBLE", "BIGINT"]);
    let names: Vec<_> = rels(&merge)
        .into_iter()
        .filter_map(|rel| match rel.rel_type.as_ref() {
            Some(substrait::proto::rel::RelType::Aggregate(aggregate)) => {
                Some(measure_names(&merge, aggregate))
            }
            _ => None,
        })
        .collect();
    // The fragment's partial SUM above the join, and the merged AVG below it.
    assert_eq!(names, [vec!["sum"], vec!["sum", "sum"]]);
}

/// q22's global `avg(c_acctbal)`: an AVG with no grouping keys, whose merge feeds a cross join.
#[test]
fn q22_two_phase_global_avg_partial_and_merge_agree_on_the_exchange_row() {
    let (partial, merge) = translate_two_phase("q22-partial-avg", "q22-merge-avg", 6);
    let aggregate = top_aggregate(&partial);
    assert!(aggregate.groupings.is_empty());
    assert_eq!(measure_kinds(aggregate), ["fp64", "i64"]);
    assert_eq!(stream_types(&merge, 6), ["DOUBLE", "BIGINT"]);
    assert_eq!(measure_fields(top_aggregate(&merge)), [0, 1]);
    assert_eq!(merge.output_names.len(), 1);
}

/// The grouping-key fields and the measures' argument fields of every aggregation, outermost
/// first.
fn aggregations(plan: &TranslatedPlan) -> Vec<(Vec<usize>, Vec<Vec<usize>>)> {
    use substrait::proto::{function_argument, rel};
    rels(plan)
        .into_iter()
        .filter_map(|rel| match rel.rel_type.as_ref() {
            Some(rel::RelType::Aggregate(aggregate)) => Some(aggregate),
            _ => None,
        })
        .map(|aggregate| {
            let keys = aggregate.grouping_expressions.iter().map(field).collect();
            let arguments = aggregate
                .measures
                .iter()
                .map(|measure| {
                    measure
                        .measure
                        .as_ref()
                        .unwrap()
                        .arguments
                        .iter()
                        .map(|argument| match argument.arg_type.as_ref() {
                            Some(function_argument::ArgType::Value(value)) => field(value),
                            other => panic!("unexpected argument {other:?}"),
                        })
                        .collect()
                })
                .collect();
            (keys, arguments)
        })
        .collect()
}

/// q16's dedupe-and-count fragment (SORT <- AGG count <- AGG merge dedupe <- EXCHANGE). The FE
/// writes the merge dedupe's first key and the count's argument as `ps_suppkey` of the partial
/// fragment's projection tuple (8), while the exchange carries the partial aggregation's tuple
/// (9) with the same slot ids.
#[test]
fn q16_merge_dedupe_resolves_slots_of_another_tuple() {
    let params = load(SLOT_LAYOUT, "q16-dedupe-count");
    let exchange = params
        .fragment
        .as_ref()
        .and_then(|fragment| fragment.plan.as_ref())
        .unwrap()
        .nodes
        .iter()
        .find(|node| node.node_type == TPlanNodeType::EXCHANGE_NODE)
        .unwrap();
    assert_eq!(exchange.row_tuples, [9]);
    let first_key = aggregation(&params, 12).grouping_exprs.as_ref().unwrap()[0].nodes[0]
        .slot_ref
        .as_ref()
        .unwrap();
    assert_eq!((first_key.tuple_id, first_key.slot_id), (8, 2));

    let plan = translate(SLOT_LAYOUT, "q16-dedupe-count").unwrap();
    // Outermost first: count(ps_suppkey) grouped by brand, type, size over the dedupe's
    // (suppkey, brand, type, size); the dedupe groups the exchange's four columns.
    assert_eq!(
        aggregations(&plan),
        [(vec![1, 2, 3], vec![vec![0]]), (vec![0, 1, 2, 3], vec![])]
    );
}

/// q18's partial fragment ends in a top-N over the partial SUM. Its sort tuple is materialized
/// from slots the FE writes as tuple 15 (o_totalprice, o_orderdate) and tuple 14 (the rest),
/// while the aggregation below emits only tuple 14: (c_name, c_custkey, o_orderkey,
/// o_orderdate, o_totalprice, sum).
#[test]
fn q18_top_n_materializes_slots_of_another_tuple() {
    use substrait::proto::rel;

    let plan = translate(SLOT_LAYOUT, "q18-partial-topn").unwrap();
    let rels = rels(&plan);
    let sort = rels
        .iter()
        .position(|rel| matches!(rel.rel_type, Some(rel::RelType::Sort(_))))
        .unwrap();
    let Some(rel::RelType::Project(project)) = rels[sort + 1].rel_type.as_ref() else {
        panic!("expected the sort tuple's projection under the sort");
    };
    // Sort tuple 16: o_totalprice, o_orderdate, c_custkey, c_name, o_orderkey, sum.
    let fields: Vec<_> = project.expressions.iter().map(field).collect();
    assert_eq!(fields, [4, 3, 1, 0, 2, 5]);
    // Under it, the aggregation (past the projection that pins its decimal sum's type).
    assert!(
        rels[sort + 2..]
            .iter()
            .find(|rel| !matches!(rel.rel_type, Some(rel::RelType::Project(_))))
            .is_some_and(|rel| matches!(rel.rel_type, Some(rel::RelType::Aggregate(_))))
    );
}

/// Fragments covering each decimal shape of the TPC-H plans (revenue products and sums, the FE's
/// own DOUBLE comparison, division, AVG, decimal join and group keys).
const DECIMAL: &str = "tpch-sf1000-decimal";

/// Every expression of the plan's relations, each followed by its subexpressions.
fn expressions(plan: &TranslatedPlan) -> Vec<&substrait::proto::Expression> {
    use substrait::proto::expression::RexType;
    use substrait::proto::{Expression, function_argument, rel};

    fn walk<'a>(expr: &'a Expression, out: &mut Vec<&'a Expression>) {
        out.push(expr);
        match expr.rex_type.as_ref() {
            Some(RexType::ScalarFunction(function)) => {
                for argument in &function.arguments {
                    if let Some(function_argument::ArgType::Value(value)) = &argument.arg_type {
                        walk(value, out);
                    }
                }
            }
            Some(RexType::Cast(cast)) => walk(cast.input.as_ref().unwrap(), out),
            Some(RexType::IfThen(if_then)) => {
                for clause in &if_then.ifs {
                    walk(clause.r#if.as_ref().unwrap(), out);
                    walk(clause.then.as_ref().unwrap(), out);
                }
                if let Some(otherwise) = if_then.r#else.as_deref() {
                    walk(otherwise, out);
                }
            }
            _ => {}
        }
    }

    let mut out = Vec::new();
    for rel in rels(plan) {
        let roots: Vec<&Expression> = match rel.rel_type.as_ref().unwrap() {
            rel::RelType::Project(project) => project.expressions.iter().collect(),
            rel::RelType::Filter(filter) => filter.condition.as_deref().into_iter().collect(),
            rel::RelType::Join(join) => join
                .expression
                .as_deref()
                .into_iter()
                .chain(join.post_join_filter.as_deref())
                .collect(),
            rel::RelType::Aggregate(aggregate) => aggregate
                .grouping_expressions
                .iter()
                .chain(aggregate.measures.iter().flat_map(|measure| {
                    measure
                        .measure
                        .as_ref()
                        .unwrap()
                        .arguments
                        .iter()
                        .filter_map(|argument| match argument.arg_type.as_ref() {
                            Some(function_argument::ArgType::Value(value)) => Some(value),
                            _ => None,
                        })
                }))
                .collect(),
            rel::RelType::Sort(sort) => sort
                .sorts
                .iter()
                .filter_map(|field| field.expr.as_ref())
                .collect(),
            _ => Vec::new(),
        };
        for root in roots {
            walk(root, &mut out);
        }
    }
    out
}

/// The (precision, scale) of every cast to a decimal in the plan.
fn decimal_casts(plan: &TranslatedPlan) -> Vec<(i32, i32)> {
    use substrait::proto::expression::RexType;
    use substrait::proto::r#type::Kind;
    expressions(plan)
        .into_iter()
        .filter_map(|expr| match expr.rex_type.as_ref() {
            Some(RexType::Cast(cast)) => match cast.r#type.as_ref()?.kind.as_ref()? {
                Kind::Decimal(decimal) => Some((decimal.precision, decimal.scale)),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Whether anything in the plan is typed FP64: a cast, a function's output, a literal, a
/// measure or a read schema.
fn uses_fp64(plan: &TranslatedPlan) -> bool {
    format!("{:?}", plan.plan.relations).contains("Fp64(")
}

/// The scalar functions of the plan named `name`.
fn calls<'a>(
    plan: &'a TranslatedPlan,
    name: &str,
) -> Vec<&'a substrait::proto::expression::ScalarFunction> {
    use substrait::proto::expression::RexType;
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    let anchors: Vec<u32> = plan
        .plan
        .extensions
        .iter()
        .filter_map(|declaration| match declaration.mapping_type.as_ref() {
            Some(MappingType::ExtensionFunction(function)) if function.name == name => {
                Some(function.function_anchor)
            }
            _ => None,
        })
        .collect();
    expressions(plan)
        .into_iter()
        .filter_map(|expr| match expr.rex_type.as_ref() {
            Some(RexType::ScalarFunction(function))
                if anchors.contains(&function.function_reference) =>
            {
                Some(function)
            }
            _ => None,
        })
        .collect()
}

/// The value expression of a function's `index`th argument.
fn argument(
    function: &substrait::proto::expression::ScalarFunction,
    index: usize,
) -> &substrait::proto::Expression {
    match function.arguments[index].arg_type.as_ref() {
        Some(substrait::proto::function_argument::ArgType::Value(value)) => value,
        other => panic!("expected a value argument, got {other:?}"),
    }
}

/// The target type of a cast expression, as a DuckDB type name.
fn cast_target(expr: &substrait::proto::Expression) -> String {
    use substrait::proto::expression::RexType;
    use substrait::proto::r#type::Kind;
    let Some(RexType::Cast(cast)) = expr.rex_type.as_ref() else {
        panic!("expected a cast, got {expr:?}");
    };
    match cast.r#type.as_ref().unwrap().kind.as_ref().unwrap() {
        Kind::Decimal(decimal) => format!("DECIMAL({},{})", decimal.precision, decimal.scale),
        Kind::Fp64(_) => "DOUBLE".to_string(),
        other => panic!("unexpected cast target {other:?}"),
    }
}

/// The input of a cast expression.
fn cast_input(expr: &substrait::proto::Expression) -> &substrait::proto::Expression {
    match expr.rex_type.as_ref() {
        Some(substrait::proto::expression::RexType::Cast(cast)) => cast.input.as_ref().unwrap(),
        _ => panic!("expected a cast, got {expr:?}"),
    }
}

/// The (precision, scale) of every decimal literal in the plan.
fn decimal_literals(plan: &TranslatedPlan) -> Vec<(i32, i32)> {
    use substrait::proto::expression::RexType;
    use substrait::proto::expression::literal::LiteralType;
    expressions(plan)
        .into_iter()
        .filter_map(|expr| match expr.rex_type.as_ref() {
            Some(RexType::Literal(literal)) => match literal.literal_type.as_ref()? {
                LiteralType::Decimal(decimal) => Some((decimal.precision, decimal.scale)),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// The revenue expression `l_extendedprice * (1 - l_discount)` is DECIMAL end to end, as is the
/// SUM over it: no FP64 anywhere in the fragment. The FE types `1 - l_discount` DECIMAL64(16,2)
/// and the product DECIMAL128(31,4); DuckDB would bind both narrower, so each is cast to the FE's
/// type, a widening. Before, the operands were cast to FP64, and the FE's cast of `1 - l_discount`
/// to DECIMAL(16,2) turned 0.9299999999999999 into 0.92 on the GPU, about 0.1% off.
#[test]
fn revenue_is_computed_and_summed_in_decimal() {
    for name in ["q03-revenue", "q05-partial", "q15-partial", "q14-partial"] {
        let plan = translate(DECIMAL, name).unwrap();
        assert!(!uses_fp64(&plan), "{name}");
        let casts = decimal_casts(&plan);
        assert!(casts.contains(&(16, 2)), "{name}: {casts:?}");
        assert!(casts.contains(&(31, 4)), "{name}: {casts:?}");
        // Every product the FE types wider than 18 digits is computed in DECIMAL128: its
        // operands are widened to DECIMAL(38) at their own scale.
        let products = calls(&plan, "multiply");
        assert!(!products.is_empty(), "{name}");
        for product in products {
            for index in 0..2 {
                assert!(
                    cast_target(argument(product, index)).starts_with("DECIMAL(38,"),
                    "{name}: {product:?}"
                );
            }
        }
        assert!(
            measure_kinds(top_aggregate(&plan))
                .iter()
                .all(|kind| *kind == "decimal"),
            "{name}"
        );
    }
}

/// q01's charge is a three-factor product, `revenue * (1 + l_tax)`, which the FE types
/// DECIMAL128(38,6) and sums as DECIMAL128(38,6). The AVGs stay in FP64.
#[test]
fn q01_three_factor_product_is_decimal_and_avg_stays_fp64() {
    let plan = translate(TWO_PHASE_AVG, "q01-partial").unwrap();
    let casts = decimal_casts(&plan);
    assert!(casts.contains(&(31, 4)), "{casts:?}");
    assert!(casts.contains(&(38, 6)), "{casts:?}");
    let aggregate = top_aggregate(&plan);
    assert_eq!(
        measure_kinds(aggregate),
        [
            "decimal", "decimal", "decimal", "decimal", "fp64", "i64", "fp64", "i64", "fp64",
            "i64", "i64"
        ]
    );
}

/// q15 keeps the supplier whose revenue equals the maximum revenue. Both sides are DECIMAL(38,4)
/// sums now, exact and independent of summation order, so the `=` holds on every run; in FP64 the
/// two sums of the same rows could differ by an ulp and match nothing.
#[test]
fn q15_revenue_equality_compares_exact_decimals() {
    let join = translate(DECIMAL, "q15-join").unwrap();
    assert!(!uses_fp64(&join));
    assert_eq!(stream_types(&join, 3), ["INTEGER", "DECIMAL(38,4)"]);
    assert_eq!(stream_types(&join, 14), ["DECIMAL(38,4)"]);
    assert!(!calls(&join, "equal").is_empty());

    let max = translate(DECIMAL, "q15-merge-max").unwrap();
    assert!(!uses_fp64(&max));
    assert_eq!(stream_types(&max, 12), ["DECIMAL(38,4)"]);
    assert_eq!(measure_names(&max, top_aggregate(&max)), ["max"]);
}

/// q14's `100.00 * sum / sum`: the multiplication is decimal (DECIMAL(38,6), with the literal
/// kept at DECIMAL(5,2)), the division runs in FP64 until the engine has an exact decimal
/// division, with a zero divisor giving NULL, and its quotient is cast to the FE's
/// DECIMAL128(38,12), the result column's type.
#[test]
fn q14_division_runs_in_fp64_and_its_result_is_decimal() {
    let plan = translate(DECIMAL, "q14-merge").unwrap();
    assert_eq!(stream_types(&plan, 8), ["DECIMAL(38,4)", "DECIMAL(38,4)"]);
    let divide = calls(&plan, "divide");
    assert_eq!(divide.len(), 1);
    let (numerator, denominator) = (argument(divide[0], 0), argument(divide[0], 1));
    assert_eq!(cast_target(numerator), "DOUBLE");
    assert_eq!(cast_target(cast_input(numerator)), "DECIMAL(38,6)");
    // The divisor is NULL where it is zero, as StarRocks' division gives NULL.
    assert!(matches!(
        denominator.rex_type,
        Some(substrait::proto::expression::RexType::IfThen(_))
    ));
    assert!(decimal_casts(&plan).contains(&(38, 12)));
    assert!(decimal_literals(&plan).contains(&(5, 2)));
    assert_eq!(plan.output_names.len(), 1);
}

/// q11 compares a DECIMAL(38,2) sum with a DECIMAL(38,12) threshold, whose common type would need
/// precision 48, so the FE itself casts both sides to DOUBLE. Those casts are kept; the threshold
/// is still computed in decimal, and both exchanges carry decimals.
#[test]
fn q11_keeps_the_fes_double_comparison() {
    let root = translate(DECIMAL, "q11-root").unwrap();
    assert_eq!(stream_types(&root, 10), ["INTEGER", "DECIMAL(38,2)"]);
    assert_eq!(stream_types(&root, 25), ["DECIMAL(38,12)"]);
    let comparison = calls(&root, "gt");
    assert_eq!(comparison.len(), 1);
    assert_eq!(cast_target(argument(comparison[0], 0)), "DOUBLE");
    assert_eq!(cast_target(argument(comparison[0], 1)), "DOUBLE");

    let threshold = translate(DECIMAL, "q11-threshold").unwrap();
    assert!(!uses_fp64(&threshold));
    assert!(decimal_casts(&threshold).contains(&(38, 12)));
    assert!(decimal_literals(&threshold).contains(&(10, 10)));
}

/// q17's `0.2 * avg(l_quantity)`: the AVG runs in FP64 and is cast to the FE's DECIMAL128(38,8);
/// the literal `0.2`, DECIMAL(1,1) in the FE, is widened to precision 5; the product is cast to
/// the FE's DECIMAL128(38,9), the type the join compares `l_quantity` with.
#[test]
fn q17_avg_feeds_a_decimal_comparison() {
    let plan = translate_after(
        TWO_PHASE_AVG,
        "q17-merge",
        &[(4, &translate(TWO_PHASE_AVG, "q17-partial").unwrap())],
    )
    .unwrap();
    let casts = decimal_casts(&plan);
    assert!(casts.contains(&(38, 8)), "{casts:?}");
    assert!(casts.contains(&(38, 9)), "{casts:?}");
    assert!(decimal_literals(&plan).contains(&(5, 1)));
    assert!(!calls(&plan, "lt").is_empty());
}

/// q22's global AVG crosses an exchange to the fragment that compares it with each customer's
/// balance. The merge casts the FP64 average to the FE's DECIMAL128(38,8), and the receiver
/// declares that column DECIMAL(38,8), so the two agree.
#[test]
fn q22_avg_crosses_the_exchange_as_the_fes_decimal() {
    let merge = translate_after(
        TWO_PHASE_AVG,
        "q22-merge-avg",
        &[(6, &translate(TWO_PHASE_AVG, "q22-partial-avg").unwrap())],
    )
    .unwrap();
    assert!(decimal_casts(&merge).contains(&(38, 8)));
    let receiver = translate(DECIMAL, "q22-cross-join").unwrap();
    assert_eq!(stream_types(&receiver, 8), ["DECIMAL(38,8)"]);
    let comparison = calls(&receiver, "gt");
    assert!(
        comparison
            .iter()
            .any(|function| cast_target(argument(function, 0)) == "DECIMAL(38,8)"),
        "{comparison:?}"
    );
}

/// Decimal join and group keys stay exact decimals: q02 joins on `ps_supplycost = min(...)`,
/// q10 groups by `c_acctbal`, and q18 groups by `o_totalprice` and filters `sum > 300` with a
/// DECIMAL(38,2) literal.
#[test]
fn decimal_join_and_group_keys_stay_decimal() {
    for (dir, name) in [
        (TWO_PHASE_AGG, "q02-merge"),
        (TWO_PHASE_AGG, "q02-partial"),
        (AGG_COLUMN_ORDER, "q10-merge"),
        (SLOT_LAYOUT, "q18-partial-topn"),
    ] {
        let plan = translate(dir, name).unwrap();
        assert!(!uses_fp64(&plan), "{name}");
    }
    let q18 = translate(SLOT_LAYOUT, "q18-partial-topn").unwrap();
    assert!(decimal_literals(&q18).contains(&(38, 2)));
}
