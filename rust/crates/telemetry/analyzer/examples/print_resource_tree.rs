//! Print the resource tree of one YAML-schema telemetry context.

use quent_analyzer::{Entity, resource::tree::ResourceTreeNode};
use quent_query_engine_analyzer::{QueryEngineModel, ui::UiAnalyzer};
use quent_store::{
    context::ContextSet,
    event::{CombinedEventLoader, filesystem::Loader},
};
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use sirius_telemetry_store::{Sirius, SiriusEvent};
use uuid::Uuid;

fn print_node(node: &ResourceTreeNode, depth: usize) {
    println!(
        "{}{}{}",
        "  ".repeat(depth),
        if node.is_resource { "<" } else { "[" },
        node.entity_id
    );
    for child in &node.children {
        print_node(child, depth + 1);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args()
        .nth(1)
        .ok_or("usage: print_resource_tree <context_dir>")?;
    let dir = std::path::Path::new(&dir);
    let context_id: Uuid = dir
        .file_name()
        .ok_or("missing context id")?
        .to_str()
        .ok_or("invalid context id")?
        .parse()?;
    let root = dir.parent().ok_or("missing context root")?;
    let events = Loader::<Sirius>::new(root, ContextSet::one(context_id))
        .combined_events()?
        .collect::<Result<Vec<_>, _>>()?;
    let engine_id = events
        .iter()
        .find_map(|event| matches!(event.data, SiriusEvent::Engine(_)).then_some(event.id))
        .ok_or("missing engine event")?;
    let analyzer = SiriusUiAnalyzer::try_new(engine_id, events.into_iter())?;
    let tree = ResourceTreeNode::try_new(&analyzer.model)?;
    println!("engine {}", analyzer.model.engine()?.id());
    print_node(&tree, 0);
    Ok(())
}
