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
            | Self::Finalizing { seq, .. }
            | Self::Exit { seq } => *seq,
        }
    }

    fn is_initial(&self) -> bool {
        matches!(self, Self::Created { .. })
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Exit { .. })
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
            Self::Finalizing { .. } => matches!(next, Self::Exit { .. }),
            Self::Exit { .. } => false,
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
            Self::Exit { .. } => "exit",
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
            Self::Queued { queue, .. } => {
                smallvec![quantity(queue.target, "entries", queue.data.entries)]
            }
            Self::Routing { manager_thread, .. }
            | Self::Reserving { manager_thread, .. }
            | Self::Downgrading { manager_thread, .. } => smallvec![unit(manager_thread.target)],
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
                usages.push(unit(executor_thread.target));
                usages.push(quantity(
                    reservation.target,
                    "bytes",
                    reservation.data.bytes,
                ));
                usages
            }
            Self::Created { .. } | Self::Finalizing { .. } | Self::Exit { .. } => SmallVec::new(),
        }
    }
}
