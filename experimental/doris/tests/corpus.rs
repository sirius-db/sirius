//! Replays the captured TPC-H dispatch corpus (`tests/fixtures/tpch/qNN/`) through the
//! backend's decoder: every payload must decode, its shape must match the summary captured
//! alongside it, and the batch must have the structure the dispatcher relies on.
//!
//! This is the harness the translator (P1) builds on: each `batch-NN-request.tcompact` is a
//! real `TPipelineFragmentParamsList` from Doris FE 4.1.4 for one TPC-H query.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use doris_plan_translator::DescriptorTable;
use doris_proto::{PExecPlanFragmentRequest, PFragmentRequestVersion};
use doris_thrift::data_sinks::TDataSink;
use doris_thrift::exprs::TExpr;
use doris_thrift::plan_nodes::TPlanNode;
use doris_thrift::types::TPrimitiveType;
use sirius_doris_be::{FragmentBatch, decode_fragment_params_list};

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch")
}

/// `(query name, payload path, summary path)` for every captured batch, sorted.
fn captured_batches() -> Vec<(String, PathBuf, PathBuf)> {
    let mut batches = Vec::new();
    for entry in std::fs::read_dir(corpus_dir()).expect("corpus directory") {
        let query_dir = entry.unwrap().path();
        if !query_dir.is_dir() {
            continue;
        }
        let query = query_dir
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        for entry in std::fs::read_dir(&query_dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix("-request.tcompact") {
                let summary = query_dir.join(format!("{stem}-summary.txt"));
                batches.push((query.clone(), path.clone(), summary));
            }
        }
    }
    batches.sort();
    batches
}

fn decode(path: &Path) -> FragmentBatch {
    let request = PExecPlanFragmentRequest {
        request: Some(std::fs::read(path).unwrap()),
        compact: Some(true),
        version: Some(PFragmentRequestVersion::Version3 as i32),
    };
    decode_fragment_params_list(&request).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

#[test]
fn corpus_covers_all_22_queries() {
    let queries: std::collections::BTreeSet<String> = captured_batches()
        .into_iter()
        .map(|(query, _, _)| query)
        .collect();
    let expected: std::collections::BTreeSet<String> =
        (1..=22).map(|n| format!("q{n:02}")).collect();
    assert_eq!(queries, expected);
}

#[test]
fn every_captured_batch_decodes_and_matches_its_summary() {
    for (query, payload, summary_path) in captured_batches() {
        let batch = decode(&payload);
        let summary = std::fs::read_to_string(&summary_path)
            .unwrap_or_else(|err| panic!("{}: {err}", summary_path.display()));

        let expected_shapes: Vec<&str> = summary
            .lines()
            .filter_map(|line| line.split_once("] ").map(|(_, shape)| shape))
            .collect();
        let shapes: Vec<String> = batch.fragments.iter().map(|f| f.shape()).collect();
        assert_eq!(shapes, expected_shapes, "{query}: {}", payload.display());

        let query_id = format!("{:x}-{:x}", batch.query_id.hi, batch.query_id.lo);
        assert!(
            summary.contains(&format!("query_id={query_id}")),
            "{query}: summary query id mismatch"
        );
    }
}

#[test]
fn every_batch_is_root_first_with_one_result_fragment_and_restored_shared_fields() {
    for (query, payload, _) in captured_batches() {
        let batch = decode(&payload);
        let result_fragments: Vec<_> = batch
            .fragments
            .iter()
            .filter(|f| f.is_result_fragment())
            .collect();
        assert_eq!(
            result_fragments.len(),
            1,
            "{query}: exactly one RESULT_SINK fragment"
        );
        assert_eq!(
            result_fragments[0].index, 0,
            "{query}: the FE ships the root fragment first"
        );
        for fragment in &batch.fragments {
            assert!(
                fragment.params.desc_tbl.is_some(),
                "{query} fragment {}: descriptor table restored",
                fragment.index
            );
            assert!(
                fragment.params.query_options.is_some(),
                "{query} fragment {}: query options present",
                fragment.index
            );
            assert!(
                !fragment.node_types().is_empty(),
                "{query} fragment {}: has plan nodes",
                fragment.index
            );
            assert_eq!(
                fragment.instance_ids().count(),
                1,
                "{query} fragment {}: parallel_pipeline_task_num=1 gives one instance per fragment",
                fragment.index
            );
        }
    }
}

/// Every expression a plan node carries, whichever node-specific struct holds it.
fn node_exprs(node: &TPlanNode) -> Vec<&TExpr> {
    let mut exprs: Vec<&TExpr> = Vec::new();
    exprs.extend(node.conjuncts.iter().flatten());
    exprs.extend(node.projections.iter().flatten());
    exprs.extend(
        node.intermediate_projections_list
            .iter()
            .flatten()
            .flatten(),
    );
    if let Some(agg) = &node.agg_node {
        exprs.extend(agg.grouping_exprs.iter().flatten());
        exprs.extend(&agg.aggregate_functions);
    }
    if let Some(join) = &node.hash_join_node {
        for cond in &join.eq_join_conjuncts {
            exprs.push(&cond.left);
            exprs.push(&cond.right);
        }
        exprs.extend(join.other_join_conjuncts.iter().flatten());
        exprs.extend(join.mark_join_conjuncts.iter().flatten());
        exprs.extend(join.src_expr_list.iter().flatten());
        exprs.extend(join.vother_join_conjunct.iter());
    }
    if let Some(join) = &node.nested_loop_join_node {
        exprs.extend(join.join_conjuncts.iter().flatten());
        exprs.extend(join.mark_join_conjuncts.iter().flatten());
        exprs.extend(join.src_expr_list.iter().flatten());
        exprs.extend(join.vjoin_conjunct.iter());
    }
    if let Some(sort) = &node.sort_node {
        exprs.extend(&sort.sort_info.ordering_exprs);
        exprs.extend(sort.sort_info.sort_tuple_slot_exprs.iter().flatten());
    }
    if let Some(exchange) = &node.exchange_node
        && let Some(sort_info) = &exchange.sort_info
    {
        exprs.extend(&sort_info.ordering_exprs);
        exprs.extend(sort_info.sort_tuple_slot_exprs.iter().flatten());
    }
    exprs.extend(node.vconjunct.iter());
    exprs
}

/// Every expression a fragment's output sink carries.
fn sink_exprs(sink: &TDataSink) -> Vec<&TExpr> {
    let mut exprs: Vec<&TExpr> = Vec::new();
    if let Some(stream) = &sink.stream_sink {
        exprs.extend(stream.output_partition.partition_exprs.iter().flatten());
        exprs.extend(stream.output_exprs.iter().flatten());
        exprs.extend(stream.conjuncts.iter().flatten());
    }
    exprs
}

/// Every tuple id a plan node names outside expressions.
fn node_tuple_ids(node: &TPlanNode) -> Vec<i32> {
    let mut ids = node.row_tuples.clone();
    ids.extend(node.output_tuple_id);
    ids.extend(node.intermediate_output_tuple_id_list.iter().flatten());
    if let Some(agg) = &node.agg_node {
        ids.extend([agg.intermediate_tuple_id, agg.output_tuple_id]);
    }
    if let Some(join) = &node.hash_join_node {
        ids.extend(join.voutput_tuple_id);
        ids.extend(join.vintermediate_tuple_id_list.iter().flatten());
    }
    if let Some(join) = &node.nested_loop_join_node {
        ids.extend(join.voutput_tuple_id);
        ids.extend(join.vintermediate_tuple_id_list.iter().flatten());
    }
    if let Some(exchange) = &node.exchange_node {
        ids.extend(&exchange.input_row_tuples);
    }
    if let Some(scan) = &node.file_scan_node {
        ids.extend(scan.tuple_id);
    }
    ids
}

/// P1.1: the descriptor table of every query builds (every slot type passes the type gate),
/// and every tuple and `(tuple_id, slot_id)` the plans and sinks name resolves against it.
#[test]
fn every_descriptor_table_builds_and_every_slot_ref_resolves() {
    let mut primitives = BTreeSet::new();
    for (query, payload, _) in captured_batches() {
        let batch = decode(&payload);
        let desc_tbl = batch.fragments[0].params.desc_tbl.as_ref().unwrap();
        let desc =
            DescriptorTable::try_from(desc_tbl).unwrap_or_else(|err| panic!("{query}: {err}"));
        assert_eq!(
            desc.tuple_count(),
            desc_tbl.tuple_descriptors.len(),
            "{query}"
        );
        assert_eq!(
            desc.slot_count(),
            desc_tbl.slot_descriptors.as_ref().map_or(0, Vec::len),
            "{query}"
        );
        for tuple in &desc_tbl.tuple_descriptors {
            for slot in desc.tuple_slots(tuple.id).unwrap() {
                primitives.insert(format!("{:?}", slot.primitive));
            }
        }

        let mut slot_refs = 0;
        for fragment in &batch.fragments {
            let plan_fragment = fragment.params.fragment.as_ref().unwrap();
            let nodes = &plan_fragment.plan.as_ref().unwrap().nodes;
            let mut exprs: Vec<&TExpr> = Vec::new();
            for node in nodes {
                for tuple_id in node_tuple_ids(node) {
                    desc.tuple(tuple_id).unwrap_or_else(|err| {
                        panic!(
                            "{query} fragment {} node {}: {err}",
                            fragment.index, node.node_id
                        )
                    });
                }
                exprs.extend(node_exprs(node));
            }
            if let Some(sink) = &plan_fragment.output_sink {
                exprs.extend(sink_exprs(sink));
            }
            for expr in exprs {
                for expr_node in &expr.nodes {
                    if let Some(slot_ref) = &expr_node.slot_ref {
                        desc.slot(slot_ref.tuple_id, slot_ref.slot_id)
                            .unwrap_or_else(|err| {
                                panic!("{query} fragment {}: {err}", fragment.index)
                            });
                        slot_refs += 1;
                    }
                }
            }
        }
        assert!(slot_refs > 0, "{query}: no SLOT_REF found");
    }
    // What the corpus exercises; a change here means the FE types the TPC-H schema differently.
    let expected: BTreeSet<String> = [
        TPrimitiveType::BIGINT,
        TPrimitiveType::BOOLEAN,
        TPrimitiveType::DATEV2,
        TPrimitiveType::DECIMAL128I,
        TPrimitiveType::DECIMAL64,
        TPrimitiveType::INT,
        TPrimitiveType::SMALLINT,
        TPrimitiveType::STRING,
        TPrimitiveType::TINYINT,
        TPrimitiveType::VARCHAR,
    ]
    .iter()
    .map(|primitive| format!("{primitive:?}"))
    .collect();
    assert_eq!(primitives, expected);
}

/// Spot-check of q01 against the layout the FE shipped: the scan's dest tuple 0 is
/// `l_quantity, l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus, l_shipdate`
/// in wire order, all nullable; the final aggregate output tuple 6 ends with a NOT NULL
/// BIGINT `count(*)` slot the planner derived (no column name).
#[test]
fn q01_descriptor_layout_matches_the_captured_plan() {
    let (_, payload, _) = captured_batches()
        .into_iter()
        .find(|(query, _, _)| query == "q01")
        .unwrap();
    let batch = decode(&payload);
    let desc =
        DescriptorTable::try_from(batch.fragments[0].params.desc_tbl.as_ref().unwrap()).unwrap();
    assert_eq!(desc.tuple_count(), 8);
    assert_eq!(desc.slot_count(), 68);
    assert_eq!(
        desc.output_names_for_tuples(&[0]).unwrap(),
        [
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate"
        ]
    );
    assert_eq!(desc.slot_global_index(0, 8, &[0]).unwrap(), 4);
    let l_quantity = desc.slot(0, 4).unwrap();
    assert_eq!(l_quantity.primitive, TPrimitiveType::DECIMAL64);
    assert!(l_quantity.nullable);
    let schema = desc.named_struct(0).unwrap();
    assert_eq!(schema.names.len(), 7);
    assert!(matches!(
        schema.r#struct.as_ref().unwrap().types[0].kind,
        Some(substrait::proto::r#type::Kind::Decimal(
            substrait::proto::r#type::Decimal {
                precision: 15,
                scale: 2,
                ..
            }
        ))
    ));
    let count_star = desc.slot(6, 66).unwrap();
    assert_eq!(count_star.primitive, TPrimitiveType::BIGINT);
    assert!(!count_star.nullable);
    assert_eq!(count_star.output_name(), "col_66");
    assert_eq!(desc.slot_global_index(6, 66, &[6]).unwrap(), 9);
}
