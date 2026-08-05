// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Headless analyzer-ingest check for a Sirius telemetry session directory
//! (WS18 export-verification rig; see tools/hwsim/export-verify/).
//!
//! Asserts, with a nonzero exit code on failure:
//!   1. every raw ndjson line deserializes (imported-event count == raw line
//!      count per entity stream — the importer silently TRUNCATES a stream at
//!      the first bad line, so a count mismatch is the only externally visible
//!      symptom of a malformed/missing-field line);
//!   2. the analyzer builds a model from the events;
//!   3. the resource tree is non-empty;
//!   4. at least one query is present (printed with labels, for eyeballing
//!      simulated `@knob=value` suffixes).
//!
//! Usage:
//!   cargo run -p sirius-telemetry-analyzer --example ingest_check -- \
//!       <telemetry_output_dir>/<session_uuid>

use std::collections::BTreeMap;
use std::io::BufRead;

use instrumentation_model::{Sirius, SiriusEvent};
use quent_analyzer::resource::tree::ResourceTreeNode;
use quent_analyzer::{Entity, Model};
use quent_query_engine_analyzer::QueryEngineModel;
use quent_query_engine_analyzer::ui::UiAnalyzer;
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use uuid::Uuid;

/// Directory (= `EntityEvent::NAME`) for each event variant.
fn entity_name(event: &SiriusEvent) -> &'static str {
    match event {
        SiriusEvent::Engine(_) => "engine",
        SiriusEvent::Worker(_) => "worker",
        SiriusEvent::QueryGroup(_) => "query_group",
        SiriusEvent::Query(_) => "query",
        SiriusEvent::Plan(_) => "plan",
        SiriusEvent::Operator(_) => "operator",
        SiriusEvent::Port(_) => "port",
        SiriusEvent::GpuDevice(_) => "gpu_device",
        SiriusEvent::ThreadGroup(_) => "thread_group",
        SiriusEvent::TaskQueue(_) => "task_queue",
        SiriusEvent::TaskManagerLoopThread(_) => "task_manager_loop_thread",
        SiriusEvent::ExecutorThread(_) => "executor_thread",
        SiriusEvent::Memory(_) => "memory",
        SiriusEvent::Channel(_) => "channel",
        SiriusEvent::Task(_) => "task",
        SiriusEvent::DataBatch(_) => "data_batch",
        SiriusEvent::BatchPlacement(_) => "batch_placement",
        SiriusEvent::MemoryTier(_) => "memory_tier",
        SiriusEvent::IoRequest(_) => "io_request",
    }
}

fn count_tree(node: &ResourceTreeNode) -> (usize, usize) {
    match node {
        ResourceTreeNode::ResourceGroup(_, children) => {
            let mut groups = 1;
            let mut resources = 0;
            for child in children {
                let (g, r) = count_tree(child);
                groups += g;
                resources += r;
            }
            (groups, resources)
        }
        ResourceTreeNode::Resource(_) => (0, 1),
    }
}

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: ingest_check <session_dir>"),
    );
    let mut failed = false;

    // Raw non-empty line counts per entity subdir (first .ndjson, matching the
    // importer's single-file resolution).
    let mut raw: BTreeMap<String, usize> = BTreeMap::new();
    for entry in std::fs::read_dir(&dir).expect("readable session dir") {
        let path = entry.expect("dir entry").path();
        if !path.is_dir() {
            continue;
        }
        let entity = path.file_name().unwrap().to_string_lossy().to_string();
        let Some(file) = std::fs::read_dir(&path)
            .ok()
            .and_then(|mut it| {
                it.find_map(|e| {
                    let p = e.ok()?.path();
                    (p.extension().and_then(|x| x.to_str()) == Some("ndjson")).then_some(p)
                })
            })
        else {
            continue;
        };
        let reader = std::io::BufReader::new(std::fs::File::open(&file).expect("readable file"));
        let n = reader
            .lines()
            .filter(|l| l.as_ref().map(|s| !s.trim().is_empty()).unwrap_or(false))
            .count();
        raw.insert(entity, n);
    }

    // Imported (deserialized) counts per entity.
    let mut imported: BTreeMap<String, usize> = BTreeMap::new();
    let events = Sirius::import_events(&dir).expect("importable session dir");
    let mut buffered = Vec::new();
    for event in events {
        *imported.entry(entity_name(&event.data).to_string()).or_default() += 1;
        buffered.push(event);
    }

    println!("== import: raw ndjson lines vs deserialized events");
    for (entity, &n_raw) in &raw {
        let n_imp = imported.get(entity).copied().unwrap_or(0);
        let status = if n_imp == n_raw {
            "ok"
        } else {
            failed = true;
            "TRUNCATED"
        };
        println!("   {entity:26} raw={n_raw:8}  imported={n_imp:8}  {status}");
    }
    for entity in imported.keys() {
        if !raw.contains_key(entity) {
            println!("   {entity:26} imported but no raw dir?!");
            failed = true;
        }
    }

    // Engine id = id of the first engine event (same as print_resource_tree).
    let engine_id: Uuid = buffered
        .iter()
        .find_map(|e| matches!(e.data, SiriusEvent::Engine(_)).then_some(e.id))
        .expect("session has at least one engine event");

    println!("== analyzer build (engine {engine_id})");
    let analyzer = match SiriusUiAnalyzer::try_new(engine_id, buffered.into_iter()) {
        Ok(a) => a,
        Err(e) => {
            println!("   BUILD FAILED: {e}");
            std::process::exit(1);
        }
    };
    let model = &analyzer.model;

    let queries: Vec<_> = model.queries().collect();
    println!("   queries: {}", queries.len());
    for q in &queries {
        println!("     {} {}", q.id(), q.instance_name());
    }
    println!(
        "   query_groups: {}  operators: {}  ports: {}  plans: {}  workers: {}",
        model.query_groups().count(),
        model.operators().count(),
        model.ports().count(),
        model.plans().count(),
        model.workers().count(),
    );

    let root_id = model.root().expect("model has a root group").id();
    match ResourceTreeNode::try_new(model, root_id) {
        Ok(tree) => {
            let (groups, resources) = count_tree(&tree);
            println!("   resource tree: {groups} group(s), {resources} resource(s)");
            if groups + resources < 2 {
                println!("   RESOURCE TREE EMPTY");
                failed = true;
            }
        }
        Err(e) => {
            println!("   RESOURCE TREE FAILED: {e}");
            failed = true;
        }
    }

    if queries.is_empty() {
        println!("   NO QUERIES");
        failed = true;
    }

    if failed {
        println!("== INGEST CHECK: FAIL");
        std::process::exit(1);
    }
    println!("== INGEST CHECK: PASS");
}
