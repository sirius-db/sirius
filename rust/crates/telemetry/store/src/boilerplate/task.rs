// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl TransitionEvent for TaskEvent {
    fn sequence(&self) -> u16 {
        match self {
            Self::Created { seq, .. }
            | Self::Queued { seq, .. }
            | Self::Routing { seq, .. }
            | Self::Reserving { seq, .. }
            | Self::Downgrading { seq, .. }
            | Self::Preparing { seq, .. }
            | Self::Computing { seq, .. }
            | Self::Finalizing { seq, .. } => *seq,
        }
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Finalizing { .. })
    }

    fn is_valid_next(&self, next: &Self) -> bool {
        match self {
            Self::Created { .. } => {
                matches!(next, Self::Queued { .. } | Self::Finalizing { .. })
            }
            Self::Queued { .. } => matches!(
                next,
                Self::Routing { .. } | Self::Reserving { .. } | Self::Finalizing { .. }
            ),
            Self::Routing { .. } => matches!(
                next,
                Self::Queued { .. } | Self::Reserving { .. } | Self::Finalizing { .. }
            ),
            Self::Reserving { .. } => matches!(
                next,
                Self::Downgrading { .. } | Self::Preparing { .. } | Self::Finalizing { .. }
            ),
            Self::Downgrading { .. } => {
                matches!(next, Self::Preparing { .. } | Self::Finalizing { .. })
            }
            Self::Preparing { .. } => {
                matches!(next, Self::Computing { .. } | Self::Finalizing { .. })
            }
            Self::Computing { .. } => {
                matches!(next, Self::Computing { .. } | Self::Finalizing { .. })
            }
            Self::Finalizing { .. } => false,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Created { .. } => "created",
            Self::Queued { .. } => "queued",
            Self::Routing { .. } => "routing",
            Self::Reserving { .. } => "reserving",
            Self::Downgrading { .. } => "downgrading",
            Self::Preparing { .. } => "preparing",
            Self::Computing { .. } => "computing",
            Self::Finalizing { .. } => "finalizing",
        }
    }

    fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> {
        let unit = |resource_id| AnalyzedUsage {
            resource_id,
            capacities: smallvec![CapacityValue::new("unit", 1)],
        };
        let quantity = |resource_id, name, value| AnalyzedUsage {
            resource_id,
            capacities: smallvec![CapacityValue::new(name, value)],
        };

        match self {
            Self::Queued { queue, .. } => queue
                .iter()
                .map(|usage| quantity(usage.target, "entries", usage.data.entries))
                .collect(),
            Self::Routing { manager_thread, .. }
            | Self::Reserving { manager_thread, .. }
            | Self::Downgrading { manager_thread, .. } => manager_thread
                .iter()
                .map(|usage| unit(usage.target))
                .collect(),
            Self::Preparing {
                executor_thread,
                reservation,
                ..
            }
            | Self::Computing {
                executor_thread,
                reservation,
                ..
            } => {
                let mut usages = SmallVec::new();
                usages.extend(executor_thread.iter().map(|usage| unit(usage.target)));
                usages.extend(
                    reservation
                        .iter()
                        .map(|usage| quantity(usage.target, "bytes", usage.data.bytes)),
                );
                usages
            }
            Self::Created { .. } | Self::Finalizing { .. } => SmallVec::new(),
        }
    }
}
