// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::QueryActivityEvent;

impl TransitionEvent for QueryActivityEvent {
    fn sequence(&self) -> u16 {
        match self {
            Self::Started { seq, .. } | Self::Finished { seq, .. } => *seq,
        }
    }
    fn is_initial(&self) -> bool {
        matches!(self, Self::Started { .. })
    }
    fn is_final(&self) -> bool {
        matches!(self, Self::Finished { .. })
    }
    fn is_valid_next(&self, next: &Self) -> bool {
        self.is_initial() && next.is_final()
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Started { kind, .. } if kind == "wait" => "waiting",
            Self::Started { .. } => "working",
            Self::Finished { .. } => "finished",
        }
    }
    fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> {
        match self {
            Self::Started { track, .. } => smallvec![AnalyzedUsage {
                resource_id: track.target,
                capacities: smallvec![CapacityValue::new("unit", 1)]
            }],
            _ => smallvec![],
        }
    }
}
