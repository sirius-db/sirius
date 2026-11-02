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

mod query_activity;
use query_activity::activity_state_name;

#[cfg(test)]
mod tests;
