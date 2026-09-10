// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Print the resource-group tree of a Sirius telemetry session directory —
//! the same tree the Quent viewer renders as collapsible rows.
//!
//! Usage:
//!   cargo run -p sirius-telemetry-analyzer --example print_resource_tree -- \
//!       <telemetry_output_dir>/<session_uuid>

use instrumentation_model::Sirius;
use quent_analyzer::resource::collection::ResourceCollection;
use quent_analyzer::resource::tree::ResourceTreeNode;
use quent_analyzer::{Entity, Model};
use quent_query_engine_analyzer::ui::UiAnalyzer;
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use uuid::Uuid;

fn print_node(
    model: &sirius_telemetry_analyzer::model::SiriusModel,
    node: &ResourceTreeNode,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    match node {
        ResourceTreeNode::ResourceGroup(id, children) => {
            let group = model.resource_group(*id).expect("group in tree");
            println!(
                "{indent}[{}] {} ({id})",
                group.type_name(),
                group.instance_name()
            );
            for child in children {
                print_node(model, child, depth + 1);
            }
        }
        ResourceTreeNode::Resource(id) => {
            let resource = model.resource(*id).expect("resource in tree");
            println!(
                "{indent}<{}> {} ({id})",
                resource.type_name(),
                resource.instance_name()
            );
        }
    }
}

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: print_resource_tree <session_dir>"),
    );

    // The engine id is the `id` of the first engine event in the session.
    let engine_file = std::fs::read_dir(dir.join("engine"))
        .expect("session dir has an engine/ subdirectory")
        .next()
        .expect("engine/ contains an event file")
        .expect("readable dir entry")
        .path();
    let first_line = std::fs::read_to_string(&engine_file)
        .expect("readable engine event file")
        .lines()
        .next()
        .expect("non-empty engine event file")
        .to_owned();
    let id_start = first_line.find("\"id\":\"").expect("engine event has id") + 6;
    let engine_id: Uuid = first_line[id_start..id_start + 36]
        .parse()
        .expect("valid engine uuid");

    let events: Vec<_> = Sirius::import_events(&dir)
        .expect("importable session dir")
        .collect::<Result<Vec<_>, _>>()
        .expect("importable events");
    let analyzer =
        SiriusUiAnalyzer::try_new(engine_id, events.into_iter()).expect("analyzable events");
    let model = &analyzer.model;

    let root_id = model.root().expect("model has a root group").id();
    let tree = ResourceTreeNode::try_new(model, root_id).expect("resource tree");
    print_node(model, &tree, 0);
}
