//! Translates fragments the StarRocks FE actually sent, recorded from a TPC-H SF1000 run.
//!
//! The fixtures under `tests/fixtures/tpch-sf1000-*` are `TExecPlanFragmentParams` in thrift
//! binary protocol, re-encoded from `SIRIUS_CN_DUMP_FRAGMENTS` dumps by
//! `tests/fixtures/debug_to_thrift.py`. Each directory holds the fragments of the queries one
//! change unblocked.

use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslateError, TranslatedPlan};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::plan_nodes::{TJoinOp, TPlanNodeType};
use substrait::proto::join_rel::JoinType;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};

/// Two-phase aggregation fragments that used to fail with "two-phase aggregation supports SUM
/// only".
const TWO_PHASE_AGG: &str = "tpch-sf1000-two-phase-agg";
/// Fragments with RIGHT SEMI joins, which used to fail with "hash join type is unsupported".
const SEMI_ANTI_JOIN: &str = "tpch-sf1000-semi-anti-join";

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
            ExchangeInput {
                node_id: node.node_id,
                stream_view: format!("sirius_stream_{}", node.node_id),
                names: (0..width).map(|index| format!("c{index}")).collect(),
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
