// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema adapters and explicit application presentation policy.
use crate::{DataBatchEvent, QueryActivityEvent, QueryEvent, TaskEvent};
use quent_analyzer::{
    fsm::native::TransitionEvent,
    resource::{AnalyzedUsage, CapacityValue},
};
use quent_dynamic_attributes::DynamicAttribute;
use smallvec::{SmallVec, smallvec};

include!(concat!(env!("OUT_DIR"), "/analyzed_events.rs"));

fn activity_state_name(event: &QueryActivityEvent) -> &'static str {
    match event {
        QueryActivityEvent::Started { kind, .. } if kind == "wait" => "waiting",
        QueryActivityEvent::Started { .. } => "working",
        QueryActivityEvent::Finished { .. } => "finished",
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executing_accepts_finished() {
        // Completed queries must survive FSM validation to appear in the UI.
        let executing = QueryEvent::Executing { seq: 2 };
        let finished = QueryEvent::Finished { seq: 3 };

        assert!(executing.is_valid_next(&finished));
    }
    #[test]
    fn generated_activity_attributes_preserve_absence_and_display_order() {
        let event = QueryActivityEvent::Finished {
            seq: 1,
            decision: Some("gpu".into()),
            outcome: Some("success".into()),
        };
        assert_eq!(
            event.attributes(),
            vec![
                DynamicAttribute::string("decision", "gpu"),
                DynamicAttribute::string("outcome", "success"),
            ]
        );
        let absent = QueryActivityEvent::Finished {
            seq: 1,
            decision: None,
            outcome: None,
        };
        assert!(absent.attributes().is_empty());
        assert!(event.is_final());
        assert!(!event.is_initial());
        assert!(!event.is_valid_next(&absent));
        assert!(event.usages().is_empty());
        assert_eq!(
            TaskEvent::Finalizing {
                seq: 1,
                success: true
            }
            .attributes(),
            vec![DynamicAttribute::u8("success", 1)]
        );
    }
}
