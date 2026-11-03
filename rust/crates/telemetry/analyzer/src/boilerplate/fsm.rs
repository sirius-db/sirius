// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared forwarding, transition conversion, and generated FSM declarations.

use super::*;

/// Mechanical forwarding for native typed FSM storage. Parent selection and
/// UI derivations remain in the semantic adapters.
macro_rules! native_fsm {
    ($ty:ty, $event:ty, $field:tt) => {
        impl Entity for $ty {
            fn id(&self) -> Uuid {
                self.$field.id()
            }
            fn type_name(&self) -> &str {
                <$event>::TYPE_NAME
            }
            fn earliest_timestamp(&self) -> TimeUnixNanoSec {
                self.$field.earliest_timestamp()
            }
            fn latest_timestamp(&self) -> TimeUnixNanoSec {
                self.$field.latest_timestamp()
            }
        }
        impl Fsm for $ty {
            type TransitionType = AnalyzedTransition<$event>;
            fn len(&self) -> usize {
                self.$field.len()
            }
            fn transition(&self, index: usize) -> Option<&Self::TransitionType> {
                self.$field.transition(index)
            }
        }
        impl<'a> FsmUsages<'a> for $ty {
            fn usages_with_state_names(
                &'a self,
            ) -> impl Iterator<Item = (&'a str, impl Usage<'a>)> {
                self.$field.usages_with_state_names()
            }
        }
        impl Using for $ty {
            fn usages(&self) -> impl Iterator<Item = impl Usage<'_>> {
                self.$field.usages()
            }
        }
    };
}

pub(super) use native_fsm;

pub(super) fn ui_transition<T: quent_analyzer::fsm::native::TransitionEvent>(
    transition: &AnalyzedTransition<T>,
    epoch: TimeUnixNanoSec,
    attributes: Vec<quent_dynamic_attributes::DynamicAttribute>,
    derived_attributes: Vec<quent_dynamic_attributes::DynamicAttribute>,
) -> quent_ui::FsmTransition {
    use quent_analyzer::fsm::Transition;
    quent_ui::FsmTransition {
        name: transition.name().into(),
        timestamp: quent_time::to_secs_relative(transition.timestamp(), epoch),
        usages: transition
            .usages()
            .iter()
            .map(|usage| quent_ui::FsmUsage {
                resource: usage.resource_id,
                capacities: usage
                    .capacities
                    .iter()
                    .map(|c| (c.name.into(), c.value))
                    .collect(),
            })
            .collect(),
        attributes,
        derived_attributes,
    }
}

use quent_ui::fsm::{FsmStateTypeDecl, FsmTransitionDecl, FsmTypeDecl};
include!(concat!(env!("OUT_DIR"), "/fsm_declarations.rs"));
