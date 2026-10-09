//! Translates fragments the StarRocks FE actually sent, recorded from a TPC-H SF1000 run.
//!
//! The fixtures under `tests/fixtures/tpch-sf1000-two-phase-agg` are `TExecPlanFragmentParams`
//! in thrift binary protocol, re-encoded from `SIRIUS_CN_DUMP_FRAGMENTS` dumps by
//! `tests/fixtures/debug_to_thrift.py`. They are the two-phase aggregation fragments of the
//! queries that used to fail with "two-phase aggregation supports SUM only".

use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslateError, TranslatedPlan};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::plan_nodes::TPlanNodeType;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};

/// Reads a recorded fragment from the fixture directory.
fn load(name: &str) -> TExecPlanFragmentParams {
    let path = format!(
        "{}/tests/fixtures/tpch-sf1000-two-phase-agg/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut protocol = TBinaryInputProtocol::new(bytes.as_slice(), true);
    TExecPlanFragmentParams::read_from_in_protocol(&mut protocol)
        .unwrap_or_else(|err| panic!("{path}: {err}"))
}

/// Translates a recorded fragment with every exchange bound to a stream, the way the CN binds
/// them. The sender's column names only have to match the exchange row's width.
fn translate(name: &str) -> Result<TranslatedPlan, TranslateError> {
    let params = load(name);
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
    let plan = translate("q13-merge").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"sum".to_string()), "{names:?}");
    assert!(!names.contains(&"count".to_string()), "{names:?}");
    // c_count (key) and the partial count(*), both BIGINT.
    assert_eq!(stream_types(&plan, 10), ["BIGINT", "BIGINT"]);
}

/// q04's and q21's merge fragments also sum partial counts. Their partial fragments still fail
/// earlier, on a RIGHT SEMI (q04, q21) and RIGHT ANTI (q21) hash join, so they have no fixture.
#[test]
fn q04_and_q21_merge_count_sums_partial_counts() {
    for (name, exchange) in [("q04-merge", 9), ("q21-merge", 25)] {
        let plan = translate(name).unwrap();
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
    let plan = translate("q13-partial").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"count".to_string()), "{names:?}");
}

/// q02's merge fragment re-applies MIN to the partial minimums, which cross the exchange as
/// DECIMAL(15,2) like the input.
#[test]
fn q02_merge_min_translates() {
    let plan = translate("q02-merge").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"min".to_string()), "{names:?}");
    assert_eq!(stream_types(&plan, 15), ["INTEGER", "DECIMAL(15,2)"]);
}

/// q02's partial fragment joins and partially aggregates MIN(ps_supplycost) per part.
#[test]
fn q02_partial_min_translates() {
    let plan = translate("q02-partial").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"min".to_string()), "{names:?}");
}

/// q15's global MAX over the per-supplier revenue: a partial MAX above a merge SUM in one
/// fragment. The merge MAX fragment never reached the CN in the recorded run (the FE stopped
/// deploying once this one failed), so it has no fixture.
#[test]
fn q15_partial_max_translates() {
    let plan = translate("q15-partial-max").unwrap();
    let names = function_names(&plan);
    assert!(names.contains(&"max".to_string()), "{names:?}");
    assert!(names.contains(&"sum".to_string()), "{names:?}");
    // supplier key and the partial revenue sum, a DECIMAL128(38,4) lowered to FP64.
    assert_eq!(stream_types(&plan, 8), ["INTEGER", "DOUBLE"]);
}
