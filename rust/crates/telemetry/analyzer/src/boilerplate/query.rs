// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use quent_dynamic_attributes::DynamicAttribute;

pub(crate) const QUERY_GROUP_ATTRIBUTE: &str = "query_group";

pub(crate) struct QueryBuilder {
    fsm: AnalyzedFsmBuilder<schema::QueryEvent>,
    metadata_stream: Option<Uuid>,
    metadata: Option<(u64, schema::QueryMetadata)>,
}

impl QueryBuilder {
    pub(crate) fn try_new(id: Uuid) -> AnalyzerResult<Self> {
        Ok(Self {
            fsm: AnalyzedFsmBuilder::try_new(id)?,
            metadata_stream: None,
            metadata: None,
        })
    }

    pub(crate) fn push_transition(&mut self, event: Event<schema::QueryEvent>) {
        self.fsm.push_transition(event);
    }

    pub(crate) fn push_metadata(
        &mut self,
        stream: Uuid,
        revision: u64,
        metadata: schema::QueryMetadata,
    ) -> AnalyzerResult<()> {
        if revision == 0 || self.metadata_stream.is_some_and(|id| id != stream) {
            return Err(quent_analyzer::AnalyzerError::Validation(
                "query metadata requires one stream with positive revisions".into(),
            ));
        }
        self.metadata_stream = Some(stream);
        if self
            .metadata
            .as_ref()
            .is_none_or(|(previous, _)| revision > *previous)
        {
            self.metadata = Some((revision, metadata));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Query {
    fsm: AnalyzedFsm<schema::QueryEvent>,
    group_label: Option<String>,
    metadata: Option<schema::QueryMetadata>,
}

impl Query {
    pub(crate) fn try_from_builder(builder: QueryBuilder) -> AnalyzerResult<Self> {
        Ok(Self {
            fsm: builder.fsm.try_build()?,
            group_label: None,
            metadata: builder.metadata.map(|(_, metadata)| metadata),
        })
    }

    pub(crate) fn set_group_label(&mut self, label: Option<String>) {
        self.group_label = label;
    }

    pub(crate) fn transitions(&self) -> &[AnalyzedTransition<schema::QueryEvent>] {
        self.fsm.transitions()
    }

    pub(crate) fn query_group_id(&self) -> Option<Uuid> {
        match self.fsm.transition(0).map(|transition| &transition.data)? {
            schema::QueryEvent::Init { query_group_id, .. } => Some(query_group_id.target),
            _ => None,
        }
    }
}

impl Entity for Query {
    fn id(&self) -> Uuid {
        self.fsm.id()
    }

    fn type_name(&self) -> &str {
        self.fsm.type_name()
    }

    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.fsm.earliest_timestamp()
    }

    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.fsm.latest_timestamp()
    }
}

impl Fsm for Query {
    type TransitionType = AnalyzedTransition<schema::QueryEvent>;

    fn len(&self) -> usize {
        self.fsm.len()
    }

    fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
        self.fsm.transition(index)
    }
}

impl<'a> FsmUsages<'a> for Query {
    fn usages_with_state_names(&'a self) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
        self.fsm.usages_with_state_names()
    }
}

impl Using for Query {
    fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
        self.fsm.usages()
    }
}

impl RefTreeEntity for Query {
    fn parent_id(&self) -> Option<Uuid> {
        self.query_group_id()
    }
}

impl QueryEntity for Query {
    fn to_ui(&self) -> AnalyzerResult<query_engine_ui::Query> {
        let transitions = self.transitions();
        let epoch = transitions.first().map(Timestamp::timestamp);
        let mut planning_s = None;
        let mut executing_s = None;
        let mut completed_s = None;

        if let Some(epoch) = epoch {
            for transition in transitions {
                match transition.data {
                    schema::QueryEvent::Planning { .. } => {
                        planning_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                    }
                    schema::QueryEvent::Executing { .. } => {
                        executing_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                    }
                    schema::QueryEvent::Finished { .. } => {
                        completed_s = Some(try_to_secs_relative(transition.timestamp(), epoch)?);
                    }
                    _ => {}
                }
            }
        }

        let mut custom_attributes = Vec::new();
        if let Some(label) = &self.group_label {
            custom_attributes.push(DynamicAttribute::string(
                QUERY_GROUP_ATTRIBUTE,
                label.clone(),
            ));
        }
        if let Some(metadata) = &self.metadata {
            for (key, value) in [
                ("connection_id", metadata.connection_id),
                ("query_ordinal", metadata.query_ordinal),
                ("planning_generation", metadata.planning_generation),
                ("execution_query_id", metadata.execution_query_id),
            ] {
                if let Some(value) = value {
                    custom_attributes.push(DynamicAttribute::u64(key, value));
                }
            }
            for (key, value) in [
                ("disposition", metadata.disposition.as_ref()),
                ("outcome", metadata.outcome.as_ref()),
            ] {
                if let Some(value) = value {
                    custom_attributes.push(DynamicAttribute::string(key, value.clone()));
                }
            }
            if let Some(worker) = &metadata.worker_id {
                custom_attributes.push(DynamicAttribute::string(
                    "worker_id",
                    worker.target.to_string(),
                ));
            }
        }
        let initial_name = transitions
            .first()
            .and_then(|transition| match &transition.data {
                schema::QueryEvent::Init { instance_name, .. } => Some(instance_name),
                _ => None,
            });
        let name = self
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.instance_name.as_ref())
            .or(initial_name);
        Ok(query_engine_ui::Query {
            id: self.id(),
            custom_attributes,
            instance_name: name.cloned(),
            start_unix_ns: epoch,
            planning_s,
            executing_s,
            completed_s,
        })
    }
}
