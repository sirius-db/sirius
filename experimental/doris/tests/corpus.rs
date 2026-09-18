//! Replays the captured TPC-H dispatch corpus (`tests/fixtures/tpch/qNN/`) through the
//! backend's decoder: every payload must decode, its shape must match the summary captured
//! alongside it, and the batch must have the structure the dispatcher relies on.
//!
//! This is the harness the translator (P1) builds on: each `batch-NN-request.tcompact` is a
//! real `TPipelineFragmentParamsList` from Doris FE 4.1.4 for one TPC-H query.

use std::path::{Path, PathBuf};

use doris_proto::{PExecPlanFragmentRequest, PFragmentRequestVersion};
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
