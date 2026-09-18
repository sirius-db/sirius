// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Default)]
pub(crate) struct PortAccumulator {
    pub(crate) operator_id: Option<Uuid>,
    pub(crate) instance_name: Option<String>,
    pub(crate) statistics: Option<quent_events::DynamicAttributes>,
}

impl EntityEventAccumulator for PortAccumulator {
    type Event = schema::PortEvent;

    fn push(&mut self, event: Self::Event) {
        match event {
            schema::PortEvent::Declaration {
                operator_id,
                instance_name,
            } => {
                self.operator_id = Some(operator_id.target);
                self.instance_name = Some(instance_name);
            }
            schema::PortEvent::Statistics { custom_attributes } => {
                self.statistics = Some(custom_attributes);
            }
        }
    }
}

#[derive(Debug)]
pub struct Port(AnalyzedEntity<PortAccumulator>);

impl Port {
    pub(crate) fn try_from_event(event: Event<schema::PortEvent>) -> AnalyzerResult<Self> {
        Ok(Self(AnalyzedEntity::try_from_event(event)?))
    }

    pub(crate) fn push(&mut self, event: Event<schema::PortEvent>) -> AnalyzerResult<()> {
        self.0.push(event)
    }

    pub(crate) fn data(&self) -> &PortAccumulator {
        self.0.accumulator()
    }
}

impl Entity for Port {
    fn id(&self) -> Uuid {
        self.0.id()
    }

    fn type_name(&self) -> &str {
        self.0.type_name()
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.0.latest_timestamp()
    }
}

impl RefTreeEntity for Port {
    fn parent_id(&self) -> Option<Uuid> {
        self.0.accumulator().operator_id
    }
}

impl PortEntity for Port {
    fn operator_id(&self) -> Option<Uuid> {
        self.data().operator_id
    }

    fn to_ui(&self, _epoch: TimeUnixNanoSec) -> query_engine_ui::Port {
        let data = self.data();
        query_engine_ui::Port {
            id: self.id(),
            operator_id: data.operator_id,
            instance_name: data.instance_name.clone(),
            statistics: data.statistics.as_ref().map(|statistics| {
                query_engine_ui::PortStatistics {
                    custom_statistics: statistics
                        .iter()
                        .map(|attribute| (attribute.key.clone(), attribute.value.clone()))
                        .collect(),
                }
            }),
        }
    }
}
