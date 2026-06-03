use quent_analyzer::EntityId;
use serde::{Deserialize, Deserializer, Serialize};
use ts_rs::TS;
use uuid::Uuid;

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
    pub current_operator_id: Option<u64>,
}

impl<'de> Deserialize<'de> for TaskFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawTaskFilter {
            #[serde(default)]
            pipeline_uuid: Option<Uuid>,
            #[serde(default)]
            operator_id: Option<Uuid>,
            #[serde(default)]
            current_operator_id: Option<u64>,
        }

        let raw = RawTaskFilter::deserialize(deserializer)?;
        Ok(Self {
            pipeline_uuid: raw.pipeline_uuid.or(raw.operator_id),
            current_operator_id: raw.current_operator_id,
        })
    }
}
