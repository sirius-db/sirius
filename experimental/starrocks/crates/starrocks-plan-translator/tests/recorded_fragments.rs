//! Translates fragments the StarRocks FE actually sent.
//!
//! The fixtures under `tests/fixtures/` are `TExecPlanFragmentParams` in thrift binary protocol,
//! re-encoded from `SIRIUS_CN_DUMP_FRAGMENTS` dumps by `tests/fixtures/debug_to_thrift.py`.
//! Each directory holds the fragments of one recorded query.

use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslateError, TranslatedPlan};
use starrocks_thrift::internal_service::TExecPlanFragmentParams;
use starrocks_thrift::plan_nodes::TPlanNodeType;
use thrift::protocol::{TBinaryInputProtocol, TSerializable};

/// `SELECT region, SUM(amount) FROM FILES(...) GROUP BY region` on 2 CNs: a leaf fragment
/// (file scan, partial SUM, HASH_PARTITIONED sink on `region`), a merge fragment (EXCHANGE 2,
/// merge SUM, UNPARTITIONED sink) and a gather fragment (EXCHANGE 4, result sink).
const TWO_CN_GROUP_BY: &str = "2cn-files-group-by";

/// Reads a recorded fragment from a fixture directory.
fn load(dir: &str, name: &str) -> TExecPlanFragmentParams {
    let path = format!(
        "{}/tests/fixtures/{dir}/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let mut protocol = TBinaryInputProtocol::new(bytes.as_slice(), true);
    let mut params = TExecPlanFragmentParams::read_from_in_protocol(&mut protocol)
        .unwrap_or_else(|err| panic!("{path}: {err}"));
    read_splits_as_whole_files(&mut params);
    params
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

/// Rewrites every byte-range split in a recorded fragment as one whole-file range per file.
///
/// The SF1000 fixtures come from a stock FE, which splits large parquet files into byte ranges
/// and spreads them over the CNs, so a recorded fragment can hold only some of a file's splits.
/// The translator refuses splits that don't tile their file (it can only scan whole files), so
/// without this those fixtures would fail on their scan before reaching the aggregation, join and
/// decimal translation the tests are about. The rewrite is test-only on purpose: reading a
/// partial split as its whole file in production would duplicate rows across CNs.
///
/// Delete this (and its call in [`load`]) once #2043 lands: the translator then emits the splits
/// as byte-range scans and accepts the recorded fragments as they are.
fn read_splits_as_whole_files(params: &mut TExecPlanFragmentParams) {
    let Some(exec) = params.params.as_mut() else {
        return;
    };
    let per_node = exec
        .per_node_scan_ranges
        .iter_mut()
        .map(|(node, ranges)| (*node, ranges));
    let per_driver = exec
        .node_to_per_driver_seq_scan_ranges
        .iter_mut()
        .flatten()
        .flat_map(|(node, per_seq)| {
            let node = *node;
            per_seq.values_mut().map(move |ranges| (node, ranges))
        });
    let mut seen = std::collections::HashSet::new();
    for (node, ranges) in per_node.chain(per_driver) {
        for range in ranges {
            let Some(broker) = range.scan_range.broker_scan_range.as_mut() else {
                continue;
            };
            // The first split of a file becomes the whole file; later splits of it are dropped.
            broker.ranges.retain_mut(|desc| {
                let Some(file_size) = desc.file_size else {
                    return true;
                };
                desc.start_offset = 0;
                desc.size = file_size;
                seen.insert((node, desc.path.clone()))
            });
        }
    }
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

/// The leaf partially sums per region and hash-partitions its output on `region`.
#[test]
fn group_by_leaf_partial_sum_hash_partitions_on_region() {
    let plan = translate(TWO_CN_GROUP_BY, "leaf").unwrap();
    assert!(function_names(&plan).contains(&"sum".to_string()));
    assert!(plan.stream_inputs.is_empty());
    assert_eq!(plan.output_partition_columns, Some(vec![0]));
}

/// The merge reads the leaves' partial sums from exchange 2 and sums them again.
#[test]
fn group_by_merge_sums_partial_sums_from_the_exchange() {
    let plan = translate(TWO_CN_GROUP_BY, "merge").unwrap();
    assert!(function_names(&plan).contains(&"sum".to_string()));
    assert_eq!(stream_types(&plan, 2), ["VARCHAR", "BIGINT"]);
    assert_eq!(plan.output_partition_columns, None);
}

/// The gather only reads exchange 4 into the result sink.
#[test]
fn group_by_gather_reads_the_exchange() {
    let plan = translate(TWO_CN_GROUP_BY, "gather").unwrap();
    assert!(function_names(&plan).is_empty());
    assert_eq!(stream_types(&plan, 4), ["VARCHAR", "BIGINT"]);
    assert_eq!(plan.output_partition_columns, None);
}
