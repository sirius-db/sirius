use instrumentation_model::task::TaskTransition as ModelTaskTransition;
use quent_analyzer::{
    AnalyzerResult, Entity,
    fsm::{
        FsmUsages, Transition,
        events::{FsmEvents, FsmEventsBuilder},
    },
    resource::{CapacityValue, Usage},
};
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec, to_secs_relative};
use quent_ui::{FiniteStateMachine, FsmTransition, FsmUsage};
use sirius_telemetry_ui::TaskFilter;
use smallvec::SmallVec;
use uuid::Uuid;

/// The reconstructed Task FSM.
pub type Task = FsmEvents<ModelTaskTransition>;

/// Builder for Task FSMs.
pub type TaskBuilder = FsmEventsBuilder<ModelTaskTransition>;

static QUEUE_ENTRY_CAPACITY: CapacityValue = CapacityValue {
    name: "capacity_entries",
    value: Some(1),
};

pub struct NormalizedTaskUsage<'a> {
    entity_id: Uuid,
    resource_id: Uuid,
    capacities: SmallVec<[&'a CapacityValue; 3]>,
    span: SpanUnixNanoSec,
}

impl<'a> NormalizedTaskUsage<'a> {
    fn new(
        entity_id: Uuid,
        resource_id: Uuid,
        capacities: SmallVec<[&'a CapacityValue; 3]>,
        span: SpanUnixNanoSec,
    ) -> Self {
        Self {
            entity_id,
            resource_id,
            capacities,
            span,
        }
    }

    pub(crate) fn resource_id(&self) -> Uuid {
        self.resource_id
    }
}

impl<'a> Usage<'a> for NormalizedTaskUsage<'a> {
    fn entity_id(&self) -> Uuid {
        self.entity_id
    }

    fn resource_id(&self) -> Uuid {
        self.resource_id
    }

    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        self.capacities.iter().copied()
    }

    fn span(&self) -> SpanUnixNanoSec {
        self.span
    }
}

pub trait TaskExt {
    fn pipeline_uuid(&self) -> Option<Uuid>;
    fn uses_current_operator_id(&self, current_operator_id: u64) -> bool;
    fn matches_filter(&self, filter: &TaskFilter) -> bool;
    fn active_span(&self) -> Option<SpanUnixNanoSec>;
    fn normalized_usages<'a>(&'a self) -> Vec<NormalizedTaskUsage<'a>>;
    fn normalized_usages_with_state_names<'a>(&'a self) -> Vec<(&'a str, NormalizedTaskUsage<'a>)>;
    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine>;
}

impl TaskExt for Task {
    fn pipeline_uuid(&self) -> Option<Uuid> {
        self.first_data().and_then(|transition| match transition {
            ModelTaskTransition::Created(data) => Some(data.pipeline_uuid),
            _ => None,
        })
    }

    fn uses_current_operator_id(&self, current_operator_id: u64) -> bool {
        self.transitions().iter().any(|transition| {
            matches!(
                &transition.data,
                ModelTaskTransition::Computing(data)
                    if data.current_operator_id == current_operator_id
            )
        })
    }

    fn matches_filter(&self, filter: &TaskFilter) -> bool {
        filter
            .pipeline_uuid
            .is_none_or(|pipeline_uuid| self.pipeline_uuid() == Some(pipeline_uuid))
            && filter
                .current_operator_id
                .is_none_or(|operator_id| self.uses_current_operator_id(operator_id))
    }

    fn active_span(&self) -> Option<SpanUnixNanoSec> {
        let start = self.transitions().get(1)?.timestamp();
        let end = self.transitions().last()?.timestamp();
        SpanUnixNanoSec::try_new(start, end).ok()
    }

    fn normalized_usages<'a>(&'a self) -> Vec<NormalizedTaskUsage<'a>> {
        self.normalized_usages_with_state_names()
            .into_iter()
            .map(|(_, usage)| usage)
            .collect()
    }

    fn normalized_usages_with_state_names<'a>(&'a self) -> Vec<(&'a str, NormalizedTaskUsage<'a>)> {
        self.usages_with_state_names()
            .map(|(state_name, usage)| {
                let capacities: SmallVec<[&CapacityValue; 3]> = if state_name == "queued" {
                    let mut capacities = SmallVec::new();
                    capacities.push(&QUEUE_ENTRY_CAPACITY);
                    capacities
                } else {
                    usage.capacities().collect()
                };
                (
                    state_name,
                    NormalizedTaskUsage::new(
                        usage.entity_id(),
                        usage.resource_id(),
                        capacities,
                        usage.span(),
                    ),
                )
            })
            .collect()
    }

    fn try_to_ui_fsm(&self, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine> {
        let transitions = self
            .transitions()
            .iter()
            .map(|transition| {
                Ok(FsmTransition {
                    name: transition.name().to_string(),
                    usages: transition
                        .usages
                        .iter()
                        .map(|usage| FsmUsage {
                            resource: usage.resource_id,
                            capacities: usage
                                .capacities
                                .iter()
                                .map(|capacity| (capacity.name.to_string(), capacity.value))
                                .collect(),
                        })
                        .collect(),
                    timestamp: to_secs_relative(transition.timestamp(), epoch),
                })
            })
            .collect::<AnalyzerResult<Vec<_>>>()?;

        Ok(FiniteStateMachine {
            id: self.id(),
            type_name: self.type_name().to_string(),
            instance_name: self.instance_name().to_string(),
            transitions,
        })
    }
}
