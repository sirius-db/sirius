// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary analyzer mappings for generated FSM events.

// TODO(johanpel): Generate this module from schema metadata. See
// https://github.com/rapidsai/quent/issues/288.

use quent_analyzer::{
    fsm::native::TransitionEvent,
    resource::{AnalyzedUsage, CapacityValue},
};
use smallvec::{SmallVec, smallvec};

use crate::{BatchPlacementEvent, DataBatchEvent, QueryEvent, TaskEvent};

mod batch_placement;
mod data_batch;
mod query;
mod task;
