// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Print the resource hierarchy of a Sirius telemetry session directory.
//!
//! Usage:
//!   cargo run -p sirius-telemetry-analyzer --example print_resource_tree -- \
//!       <telemetry_output_dir>/<session_uuid>

use quent_analyzer::ref_tree::RefTreeCollection;
use quent_analyzer::resource::tree::ResourceTreeNode;
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_store::event::{EntityEventStore, ModelEventStore, filesystem::Store};
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use sirius_telemetry_store::{Engine, Sirius};
use uuid::Uuid;

fn print_node(
    model: &sirius_telemetry_analyzer::model::SiriusModel,
    node: &ResourceTreeNode,
    depth: usize,
) {
    let indent = "  ".repeat(depth);
    let entity = model
        .ref_tree_entity(node.entity_id)
        .expect("entity in tree");
    if node.is_resource {
        println!("{indent}<{}> ({})", entity.type_name(), node.entity_id);
    } else {
        println!("{indent}[{}] ({})", entity.type_name(), node.entity_id);
    }
    for child in &node.children {
        print_node(model, child, depth + 1);
    }
}

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: print_resource_tree <session_dir>"),
    );

    let context_id: Uuid = dir
        .file_name()
        .and_then(|name| name.to_str())
        .expect("session directory name")
        .parse()
        .expect("session directory name is a UUID");
    let root = dir.parent().expect("session directory has a parent");
    let store = Store::<Sirius>::new(root);
    let engine_id = store
        .entity_events::<Engine>(context_id)
        .expect("importable engine events")
        .next()
        .expect("at least one engine event")
        .expect("valid engine event")
        .id;

    let events: Vec<_> = store
        .events(context_id)
        .expect("importable session")
        .collect::<Result<Vec<_>, _>>()
        .expect("importable events");
    let analyzer =
        SiriusUiAnalyzer::try_new(engine_id, events.into_iter()).expect("analyzable events");
    let model = &analyzer.model;

    let tree = ResourceTreeNode::try_new(model).expect("resource tree");
    print_node(model, &tree, 0);
}
