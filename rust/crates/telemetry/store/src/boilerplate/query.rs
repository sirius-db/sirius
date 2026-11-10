// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl TransitionEvent for QueryEvent {
    fn sequence(&self) -> u16 {
        match self {
            Self::Init { seq, .. }
            | Self::Planning { seq }
            | Self::Executing { seq }
            | Self::Finished { seq } => *seq,
        }
    }

    fn is_initial(&self) -> bool {
        matches!(self, Self::Init { .. })
    }

    fn is_final(&self) -> bool {
        matches!(self, Self::Finished { .. })
    }

    fn is_valid_next(&self, next: &Self) -> bool {
        match self {
            Self::Init { .. } => matches!(next, Self::Planning { .. } | Self::Finished { .. }),
            Self::Planning { .. } => matches!(next, Self::Executing { .. } | Self::Finished { .. }),
            Self::Executing { .. } => matches!(next, Self::Finished { .. }),
            Self::Finished { .. } => false,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Init { .. } => "init",
            Self::Planning { .. } => "planning",
            Self::Executing { .. } => "executing",
            Self::Finished { .. } => "finished",
        }
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
}
