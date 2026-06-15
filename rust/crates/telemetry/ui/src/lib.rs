//! Types shared with the UI.

use quent_analyzer::EntityId;
use serde::{Deserialize, Serialize};
use ts_rs::TS;
use uuid::Uuid;

/// A reference to an entity
#[derive(TS, Debug, Serialize)]
pub enum EntityRef {
    Engine(Uuid),
    Worker(Uuid),
    QueryGroup(Uuid),
    Query(Uuid),
    Plan(Uuid),
    Operator(Uuid),
    Port(Uuid),
    Resource(Uuid),
    ResourceGroup(Uuid),
    Task(Uuid),
}

impl EntityId for EntityRef {
    fn is_resource(&self) -> bool {
        matches!(self, EntityRef::Resource(_))
    }

    fn is_resource_group(&self) -> bool {
        matches!(
            self,
            EntityRef::Engine(_)
                | EntityRef::Worker(_)
                | EntityRef::QueryGroup(_)
                | EntityRef::Query(_)
                | EntityRef::Plan(_)
                | EntityRef::Operator(_)
                | EntityRef::Port(_)
                | EntityRef::ResourceGroup(_)
        )
    }
}

#[derive(TS, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryFilter {
    pub query_id: Uuid,
}

#[derive(TS, Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct TaskFilter {
    pub pipeline_uuid: Option<Uuid>,
    pub physical_operator_id: Option<u32>,
}

impl<'de> Deserialize<'de> for TaskFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TaskFilterWire {
            #[serde(default)]
            pipeline_uuid: Option<Uuid>,
            #[serde(default)]
            operator_id: Option<Uuid>,
            #[serde(default)]
            physical_operator_id: Option<u32>,
            #[serde(default)]
            current_operator_id: Option<u32>,
        }

        let wire = TaskFilterWire::deserialize(deserializer)?;
        Ok(Self {
            pipeline_uuid: wire.pipeline_uuid.or(wire.operator_id),
            physical_operator_id: wire.physical_operator_id.or(wire.current_operator_id),
        })
    }
}
