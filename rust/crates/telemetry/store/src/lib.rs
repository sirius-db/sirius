//! Schema-generated Sirius stored-event types.

include!(concat!(env!("OUT_DIR"), "/sirius.rs"));

use quent_analyzer::{
    fsm::events::{AnalyzableTransition, AnalyzedUsage, DynamicAttribute},
    resource::CapacityValue,
};
use smallvec::{SmallVec, smallvec};

impl AnalyzableTransition for QueryEvent {
    fn entity_type_name() -> &'static str {
        "query"
    }

    fn sequence(&self) -> u16 {
        match self {
            Self::Init { seq, .. }
            | Self::Planning { seq }
            | Self::Executing { seq }
            | Self::Exit { seq } => *seq,
        }
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Exit { .. })
    }

    fn state_name(&self) -> &'static str {
        match self {
            Self::Init { .. } => "init",
            Self::Planning { .. } => "planning",
            Self::Executing { .. } => "executing",
            Self::Exit { .. } => "exit",
        }
    }

    fn instance_name(&self) -> Option<String> {
        match self {
            Self::Init { instance_name, .. } => Some(instance_name.clone()),
            _ => None,
        }
    }
}

impl AnalyzableTransition for TaskEvent {
    fn entity_type_name() -> &'static str {
        "task"
    }

    fn sequence(&self) -> u16 {
        match self {
            Self::Created { seq, .. }
            | Self::Queued { seq, .. }
            | Self::Routing { seq, .. }
            | Self::Reserving { seq, .. }
            | Self::Downgrading { seq, .. }
            | Self::Preparing { seq, .. }
            | Self::Computing { seq, .. }
            | Self::Finalizing { seq, .. } => *seq,
        }
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Finalizing { .. })
    }

    fn state_name(&self) -> &'static str {
        match self {
            Self::Created { .. } => "created",
            Self::Queued { .. } => "queued",
            Self::Routing { .. } => "routing",
            Self::Reserving { .. } => "reserving",
            Self::Downgrading { .. } => "downgrading",
            Self::Preparing { .. } => "preparing",
            Self::Computing { .. } => "computing",
            Self::Finalizing { .. } => "finalizing",
        }
    }

    fn instance_name(&self) -> Option<String> {
        match self {
            Self::Created { instance_name, .. } => Some(instance_name.clone()),
            _ => None,
        }
    }

    fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> {
        let unit = |resource_id| AnalyzedUsage {
            resource_id,
            capacities: smallvec![CapacityValue::new("unit", 1)],
        };
        let quantity = |resource_id, name, value| AnalyzedUsage {
            resource_id,
            capacities: smallvec![CapacityValue::new(name, value)],
        };

        match self {
            Self::Queued { queue, .. } => queue
                .iter()
                .map(|usage| quantity(usage.target, "entries", usage.data.entries))
                .collect(),
            Self::Routing { manager_thread, .. }
            | Self::Reserving { manager_thread, .. }
            | Self::Downgrading { manager_thread, .. } => manager_thread
                .iter()
                .map(|usage| unit(usage.target))
                .collect(),
            Self::Preparing {
                executor_thread,
                reservation,
                ..
            }
            | Self::Computing {
                executor_thread,
                reservation,
                ..
            } => {
                let mut usages = SmallVec::new();
                usages.extend(executor_thread.iter().map(|usage| unit(usage.target)));
                usages.extend(
                    reservation
                        .iter()
                        .map(|usage| quantity(usage.target, "bytes", usage.data.bytes)),
                );
                usages
            }
            Self::Created { .. } | Self::Finalizing { .. } => SmallVec::new(),
        }
    }

    fn dynamic_attributes(&self) -> Vec<DynamicAttribute> {
        match self {
            Self::Created { pipeline_uuid, .. } => vec![DynamicAttribute::string(
                "pipeline_uuid",
                pipeline_uuid.target.to_string(),
            )],
            Self::Routing {
                preferred_device_id,
                ..
            } => vec![DynamicAttribute::i64(
                "preferred_device_id",
                *preferred_device_id,
            )],
            Self::Reserving {
                requested_bytes,
                input_basis,
                peak_estimate,
                bytes_to_materialize,
                ..
            } => vec![
                DynamicAttribute::u64("requested_bytes", *requested_bytes),
                DynamicAttribute::u64("input_basis", *input_basis),
                DynamicAttribute::u64("peak_estimate", *peak_estimate),
                DynamicAttribute::u64("bytes_to_materialize", *bytes_to_materialize),
            ],
            Self::Downgrading {
                shortfall_bytes,
                partial_bytes,
                ..
            } => vec![
                DynamicAttribute::u64("shortfall_bytes", *shortfall_bytes),
                DynamicAttribute::u64("partial_bytes", *partial_bytes),
            ],
            Self::Preparing {
                origin_tier,
                target_tier,
                input_bytes,
                ..
            } => vec![
                DynamicAttribute::string("origin_tier", origin_tier.clone()),
                DynamicAttribute::string("target_tier", target_tier.clone()),
                DynamicAttribute::u64("input_bytes", *input_bytes),
            ],
            Self::Computing {
                current_operator_id,
                input_bytes,
                peak_allocated_bytes,
                ..
            } => vec![
                DynamicAttribute::u32("current_operator_id", *current_operator_id),
                DynamicAttribute::u64("input_bytes", *input_bytes),
                DynamicAttribute::u64("peak_allocated_bytes", *peak_allocated_bytes),
            ],
            Self::Finalizing { success, .. } => {
                vec![DynamicAttribute::u8("success", u8::from(*success))]
            }
            Self::Queued { .. } => Vec::new(),
        }
    }
}

impl AnalyzableTransition for DataBatchEvent {
    fn entity_type_name() -> &'static str {
        "data_batch"
    }

    fn sequence(&self) -> u16 {
        match self {
            Self::Constructed { seq, .. }
            | Self::Stationary { seq, .. }
            | Self::InTransit { seq, .. }
            | Self::Destructed { seq } => *seq,
        }
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Destructed { .. })
    }

    fn state_name(&self) -> &'static str {
        match self {
            Self::Constructed { .. } => "constructed",
            Self::Stationary { .. } => "stationary",
            Self::InTransit { .. } => "in_transit",
            Self::Destructed { .. } => "destructed",
        }
    }

    fn instance_name(&self) -> Option<String> {
        match self {
            Self::Constructed { instance_name, .. } => Some(instance_name.clone()),
            _ => None,
        }
    }

    fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> {
        let bytes = |resource_id, value| AnalyzedUsage {
            resource_id,
            capacities: smallvec![CapacityValue::new("bytes", value)],
        };
        match self {
            Self::Stationary { memory, .. } => memory
                .iter()
                .map(|usage| bytes(usage.target, usage.data.bytes))
                .collect(),
            Self::InTransit {
                source_memory,
                dest_memory,
                channel,
                ..
            } => {
                let mut usages = SmallVec::new();
                usages.extend(
                    source_memory
                        .iter()
                        .map(|usage| bytes(usage.target, usage.data.bytes)),
                );
                usages.extend(
                    dest_memory
                        .iter()
                        .map(|usage| bytes(usage.target, usage.data.bytes)),
                );
                usages.extend(
                    channel
                        .iter()
                        .map(|usage| bytes(usage.target, usage.data.bytes)),
                );
                usages
            }
            Self::Constructed { .. } | Self::Destructed { .. } => SmallVec::new(),
        }
    }

    fn dynamic_attributes(&self) -> Vec<DynamicAttribute> {
        match self {
            Self::Constructed {
                data_batch_id,
                producer_pipeline_uuid,
                ..
            } => vec![
                DynamicAttribute::u64("data_batch_id", *data_batch_id),
                DynamicAttribute::string(
                    "producer_pipeline_uuid",
                    producer_pipeline_uuid.target.to_string(),
                ),
            ],
            _ => Vec::new(),
        }
    }
}

impl AnalyzableTransition for BatchPlacementEvent {
    fn entity_type_name() -> &'static str {
        "batch_placement"
    }

    fn sequence(&self) -> u16 {
        match self {
            Self::BatchRegistered { seq, .. }
            | Self::BatchQueued { seq, .. }
            | Self::BatchPackaged { seq, .. }
            | Self::BatchProcessing { seq, .. }
            | Self::BatchConsumed { seq, .. } => *seq,
        }
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::BatchConsumed { .. })
    }

    fn state_name(&self) -> &'static str {
        match self {
            Self::BatchRegistered { .. } => "batch_registered",
            Self::BatchQueued { .. } => "batch_queued",
            Self::BatchPackaged { .. } => "batch_packaged",
            Self::BatchProcessing { .. } => "batch_processing",
            Self::BatchConsumed { .. } => "batch_consumed",
        }
    }

    fn instance_name(&self) -> Option<String> {
        match self {
            Self::BatchRegistered { instance_name, .. } => Some(instance_name.clone()),
            _ => None,
        }
    }

    fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> {
        let tier = match self {
            Self::BatchRegistered { tier, .. }
            | Self::BatchQueued { tier, .. }
            | Self::BatchPackaged { tier, .. }
            | Self::BatchProcessing { tier, .. } => tier,
            Self::BatchConsumed { .. } => return SmallVec::new(),
        };
        tier.iter()
            .map(|usage| AnalyzedUsage {
                resource_id: usage.target,
                capacities: smallvec![CapacityValue::new("bytes", usage.data.bytes)],
            })
            .collect()
    }

    fn dynamic_attributes(&self) -> Vec<DynamicAttribute> {
        match self {
            Self::BatchRegistered {
                batch_id,
                pipeline_uuid,
                port_uuid,
                origin,
                ..
            } => {
                let mut attributes = vec![
                    DynamicAttribute::u64("batch_id", *batch_id),
                    DynamicAttribute::string("pipeline_uuid", pipeline_uuid.target.to_string()),
                ];
                if let Some(port_uuid) = port_uuid {
                    attributes.push(DynamicAttribute::string(
                        "port_uuid",
                        port_uuid.target.to_string(),
                    ));
                }
                attributes.push(DynamicAttribute::string("origin", origin.clone()));
                attributes
            }
            Self::BatchPackaged { task_uuid, .. } | Self::BatchProcessing { task_uuid, .. } => {
                vec![DynamicAttribute::string("task_uuid", task_uuid.to_string())]
            }
            Self::BatchConsumed { reason, .. } => {
                vec![DynamicAttribute::string("reason", reason.clone())]
            }
            Self::BatchQueued { .. } => Vec::new(),
        }
    }
}
