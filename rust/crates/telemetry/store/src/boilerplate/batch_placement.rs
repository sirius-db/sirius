// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl TransitionEvent for BatchPlacementEvent {
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

    fn is_valid_next(&self, next: &Self) -> bool {
        match self {
            Self::BatchRegistered { .. } => {
                matches!(next, Self::BatchQueued { .. } | Self::BatchPackaged { .. })
            }
            Self::BatchQueued { .. } => matches!(
                next,
                Self::BatchQueued { .. } | Self::BatchPackaged { .. } | Self::BatchConsumed { .. }
            ),
            Self::BatchPackaged { .. } => matches!(
                next,
                Self::BatchPackaged { .. }
                    | Self::BatchProcessing { .. }
                    | Self::BatchConsumed { .. }
            ),
            Self::BatchProcessing { .. } => matches!(
                next,
                Self::BatchPackaged { .. }
                    | Self::BatchProcessing { .. }
                    | Self::BatchConsumed { .. }
            ),
            Self::BatchConsumed { .. } => false,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::BatchRegistered { .. } => "batch_registered",
            Self::BatchQueued { .. } => "batch_queued",
            Self::BatchPackaged { .. } => "batch_packaged",
            Self::BatchProcessing { .. } => "batch_processing",
            Self::BatchConsumed { .. } => "batch_consumed",
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
}
