// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Borrowed access to the analyzed FSM families. Typed storage, validation and
//! application-specific derivations stay in their semantic adapters.
use quent_analyzer::{
    AnalyzerResult, Entity, RefTreeEntity,
    fsm::{Fsm, Transition, collection::FsmCollection},
    resource::{AnalyzedUsage, CapacityValue, Usage},
};
use quent_query_engine_analyzer::entities;
use quent_query_engine_ui::{EntityListResponse, OperatorFilter};
use quent_time::{TimeUnixNanoSec, Timestamp, span::SpanUnixNanoSec};
use quent_ui::{
    FiniteStateMachine,
    fsm::{FsmTypeDecl, FsmTypeDeclaration},
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use uuid::Uuid;

use crate::{
    boilerplate::{DataBatch, QueryActivity, Task},
    model::SiriusModel,
};

/// Semantic query ownership, independent of display lanes and resource usage.
pub(crate) struct QueryScope<'a> {
    pub(crate) query_id: Uuid,
    pub(crate) operators: &'a HashSet<Uuid>,
}

pub(crate) trait ScopedFsm {
    fn visible_in(&self, scope: &QueryScope<'_>) -> bool;
    fn in_reference_tree(&self, scope: &QueryScope<'_>) -> bool;
    fn operator_in_scope(&self, scope: &QueryScope<'_>) -> Option<Uuid>;
    fn matches_operator_filter(&self, filter: &OperatorFilter) -> bool;
    fn to_ui(
        &self,
        model: &SiriusModel,
        epoch: TimeUnixNanoSec,
    ) -> AnalyzerResult<FiniteStateMachine>;
}

macro_rules! fsm_families {
    (resources $resources:tt groups $groups:tt entities $entities:tt
     fsms { $($field:ident: $ty:ident => $variant:ident($name:path),)* }) => {
        #[derive(Clone, Copy)]
        pub(crate) enum FsmRef<'a> { $($variant(&'a $ty),)* }

        impl<'a> FsmRef<'a> {
            fn entity(self) -> &'a dyn RefTreeEntity {
                match self { $(Self::$variant(f) => f,)* }
            }
            fn transition(self, index: usize) -> Option<TransitionRef<'a>> {
                match self { $(Self::$variant(f) => f.transition(index).map(|t| TransitionRef {
                    name: t.name(), timestamp: t.timestamp(), usages: t.usages(),
                }),)* }
            }
            fn len(self) -> usize { match self { $(Self::$variant(f) => f.len(),)* } }
            pub(crate) fn belongs_to(self, query_id: Uuid, operators: &HashSet<Uuid>) -> bool {
                let scope = QueryScope { query_id, operators };
                match self { $(Self::$variant(f) => ScopedFsm::visible_in(f, &scope),)* }
            }
            fn in_reference_tree(self, scope: &QueryScope<'_>) -> bool {
                match self { $(Self::$variant(f) => ScopedFsm::in_reference_tree(f, scope),)* }
            }
            fn operator_in(self, scope: &QueryScope<'_>) -> Option<Uuid> {
                match self { $(Self::$variant(f) => ScopedFsm::operator_in_scope(f, scope),)* }
            }
            pub(crate) fn try_to_ui(self, model: &SiriusModel, epoch: TimeUnixNanoSec) -> AnalyzerResult<FiniteStateMachine> {
                match self { $(Self::$variant(f) => ScopedFsm::to_ui(f, model, epoch),)* }
            }
            pub(crate) fn matches_filter(self, filter: &OperatorFilter) -> bool {
                match self { $(Self::$variant(f) => ScopedFsm::matches_operator_filter(f, filter),)* }
            }
        }
        pub(crate) fn get(model: &SiriusModel, id: Uuid) -> Option<FsmRef<'_>> {
            $(if let Some(f) = model.$field.get(&id) { return Some(FsmRef::$variant(f)); })*
            None
        }
        // Retain source traversal order: rebuilding a map can reorder equal-ranked
        // intervals in viewer responses. Lookup still uses only the selected maps.
        pub(crate) struct ScopedFsms<'a> { $($field: HashMap<Uuid, &'a $ty>,)* order: Vec<FsmRef<'a>> }
        impl<'a> ScopedFsms<'a> {
            pub(crate) fn new(model: &'a SiriusModel, scope: &QueryScope<'_>) -> Self {
                let mut selected = Self { $($field: model.$field.iter()
                    .filter(|(_, f)| ScopedFsm::visible_in(*f, scope))
                    .map(|(id, f)| (*id, f)).collect(),)* order: Vec::new() };
                selected.order = std::iter::empty() $(.chain(model.$field.values().filter(|f| selected.$field.contains_key(&f.id())).map(FsmRef::$variant)))*.collect();
                selected
            }
            $(pub(crate) fn $field(&self) -> impl Iterator<Item = &'a $ty> + '_ {
                self.$field.values().copied()
            })*
            pub(crate) fn iter(&self) -> impl Iterator<Item = FsmRef<'a>> + '_ {
                self.order.iter().copied()
            }
            pub(crate) fn get(&self, id: Uuid) -> Option<FsmRef<'a>> {
                $(if let Some(f) = self.$field.get(&id) { return Some(FsmRef::$variant(f)); })*
                None
            }
            pub(crate) fn reference_tree<'s>(&'s self, scope: &'s QueryScope<'_>) -> impl Iterator<Item = &'a dyn RefTreeEntity> + 's {
                self.iter().filter(move |f| f.in_reference_tree(scope)).map(FsmRef::entity)
            }
        }
        pub(crate) fn is_type(name: &str) -> bool { matches!(name, $($name)|*) }
        pub(crate) fn declarations() -> impl Iterator<Item = FsmTypeDecl> {
            [$($ty::fsm_type_declaration(),)*].into_iter()
        }
        pub(crate) fn list_entities(
            model: &SiriusModel, query_id: Uuid, operators: &HashSet<Uuid>,
            filter: &OperatorFilter, query: entities::ListQuery<'_>,
        ) -> AnalyzerResult<EntityListResponse> {
            match query.filter.entity_type_name.as_deref() {
                $(Some($name) => entities::list_entities(
                    &TypedCollection(&model.$field),
                    |f| FsmRef::$variant(f).belongs_to(query_id, operators) && FsmRef::$variant(f).matches_filter(filter),
                    |f| FsmRef::$variant(f).operator_in(&QueryScope { query_id, operators }),
                    |f, epoch| FsmRef::$variant(f).try_to_ui(model, epoch), query,
                ),)*
                _ => entities::list_entities(
                    &TypedCollection(&model.tasks),
                    |f| FsmRef::Task(f).belongs_to(query_id, operators) && FsmRef::Task(f).matches_filter(filter),
                    |f| FsmRef::Task(f).operator_in(&QueryScope { query_id, operators }),
                    |f, epoch| FsmRef::Task(f).try_to_ui(model, epoch), query,
                ),
            }
        }
    };
}
crate::families::with_families!(fsm_families);

impl Entity for FsmRef<'_> {
    fn id(&self) -> Uuid {
        self.entity().id()
    }
    fn type_name(&self) -> &str {
        self.entity().type_name()
    }
    fn earliest_timestamp(&self) -> TimeUnixNanoSec {
        self.entity().earliest_timestamp()
    }
    fn latest_timestamp(&self) -> TimeUnixNanoSec {
        self.entity().latest_timestamp()
    }
}

impl RefTreeEntity for FsmRef<'_> {
    fn parent_id(&self) -> Option<Uuid> {
        self.entity().parent_id()
    }
}

impl<'a> FsmRef<'a> {
    pub(crate) fn usages_with_state_names(self) -> impl Iterator<Item = (&'a str, UsageRef<'a>)> {
        let len = self.len();
        (0..len).flat_map(move |index| {
            let from = self.transition(index).unwrap();
            let to = self.transition(index + 1).unwrap();
            let span = SpanUnixNanoSec::try_new(from.timestamp, to.timestamp).unwrap();
            from.usages.iter().map(move |usage| {
                (
                    from.name,
                    UsageRef {
                        entity_id: self.id(),
                        usage,
                        span,
                    },
                )
            })
        })
    }

    pub(crate) fn usages(self) -> impl Iterator<Item = UsageRef<'a>> {
        self.usages_with_state_names().map(|(_, usage)| usage)
    }
}

struct TransitionRef<'a> {
    name: &'a str,
    timestamp: TimeUnixNanoSec,
    usages: &'a [AnalyzedUsage],
}

pub(crate) struct UsageRef<'a> {
    entity_id: Uuid,
    usage: &'a AnalyzedUsage,
    span: SpanUnixNanoSec,
}
impl<'a> Usage<'a> for UsageRef<'a> {
    fn entity_id(&self) -> Uuid {
        self.entity_id
    }
    fn resource_id(&self) -> Uuid {
        self.usage.resource_id
    }
    fn capacities(&self) -> impl Iterator<Item = &'a CapacityValue> {
        self.usage.capacities.iter()
    }
    fn span(&self) -> SpanUnixNanoSec {
        self.span
    }
}

// The pinned upstream listing interface requires a concrete Fsm::TransitionType
// and returns references to it. Keep that typed dispatch here until upstream can
// consume borrowed heterogeneous transitions; retain its ranking/paging logic.
struct TypedCollection<'a, F>(&'a HashMap<Uuid, F>);
impl<F: Fsm> FsmCollection for TypedCollection<'_, F> {
    type Fsm = F;
    fn fsms(&self) -> impl Iterator<Item = &F> {
        self.0.values()
    }
}
