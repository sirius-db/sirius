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
    // supplier key and the partial revenue sum, a DECIMAL128(38,4) lowered to FP64.
    assert_eq!(stream_types(&plan, 8), ["INTEGER", "DOUBLE"]);
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
    // The one aggregate, the revenue sum, is column 7 = slot 39, sent as FP64.
    assert_eq!(plan.output_names.len(), 8);
    assert_eq!(measure_kinds(top_aggregate(&plan)), ["fp64"]);
}

/// q22's partial aggregation emits `count(*)` and then `sum(c_acctbal)` after its one key,
/// typed by the partial-state rule as BIGINT and FP64, in aggregate order.
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
    assert_eq!(measure_kinds(top_aggregate(&plan)), ["i64", "fp64"]);
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
            "DOUBLE"
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
    assert_eq!(stream_types(&plan, 15), ["VARCHAR", "BIGINT", "DOUBLE"]);
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
    assert_eq!(
        measure_kinds(aggregate),
        [
            "fp64", "fp64", "fp64", "fp64", "fp64", "i64", "fp64", "i64", "fp64", "i64", "i64"
        ]
    );
    assert_eq!(partial.output_names.len(), 13);
    assert_eq!(
        stream_types(&merge, 3),
        [
            "VARCHAR", "VARCHAR", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "BIGINT",
            "DOUBLE", "BIGINT", "DOUBLE", "BIGINT", "BIGINT"
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
