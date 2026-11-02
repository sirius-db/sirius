// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Presentation names for query activity states.

use crate::QueryActivityEvent;

pub(super) fn activity_state_name(event: &QueryActivityEvent) -> &'static str {
    match event {
        QueryActivityEvent::Started { kind, .. } if kind == "wait" => "waiting",
        QueryActivityEvent::Started { .. } => "working",
        QueryActivityEvent::Finished { .. } => "finished",
    }
}
