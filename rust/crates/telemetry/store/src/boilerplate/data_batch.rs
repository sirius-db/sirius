// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl TransitionEvent for DataBatchEvent {
    fn sequence(&self) -> u16 {
        match self {
            Self::Constructed { seq, .. }
            | Self::Stationary { seq, .. }
            | Self::InTransit { seq, .. }
            | Self::Destructed { seq }
            | Self::Exit { seq } => *seq,
        }
    }

    fn is_initial(&self) -> bool {
        matches!(self, Self::Constructed { .. })
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Exit { .. })
    }

    fn is_valid_next(&self, next: &Self) -> bool {
        match self {
            Self::Constructed { .. } => matches!(next, Self::Stationary { .. }),
            Self::Stationary { .. } => matches!(
                next,
                Self::Stationary { .. } | Self::InTransit { .. } | Self::Destructed { .. }
            ),
            Self::InTransit { .. } => matches!(next, Self::Stationary { .. }),
            Self::Destructed { .. } => matches!(next, Self::Exit { .. }),
            Self::Exit { .. } => false,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Constructed { .. } => "constructed",
            Self::Stationary { .. } => "stationary",
            Self::InTransit { .. } => "in_transit",
            Self::Destructed { .. } => "destructed",
            Self::Exit { .. } => "exit",
        }
    }

    fn usages(&self) -> SmallVec<[AnalyzedUsage; 1]> {
        let bytes = |resource_id, value| AnalyzedUsage {
            resource_id,
            capacities: smallvec![CapacityValue::new("bytes", value)],
        };
        match self {
            Self::Stationary { memory, .. } => smallvec![bytes(memory.target, memory.data.bytes)],
            Self::InTransit {
                source_memory,
                dest_memory,
                channel,
                ..
            } => {
                let mut usages = SmallVec::new();
                usages.push(bytes(source_memory.target, source_memory.data.bytes));
                usages.push(bytes(dest_memory.target, dest_memory.data.bytes));
                usages.push(bytes(channel.target, channel.data.bytes));
                usages
            }
            Self::Constructed { .. } | Self::Destructed { .. } | Self::Exit { .. } => {
                SmallVec::new()
            }
        }
    }
}
