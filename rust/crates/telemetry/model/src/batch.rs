//! Batch placement telemetry: one placement per (physical batch x consuming
//! pipeline input port), fan-out producing one placement per consumer. Every
//! non-terminal state carries a `tier` usage with `bytes` = the batch's data
//! size; a tier change is a self-transition with a different tier usage.

use quent_model::{fsm, resource, state};
use uuid::Uuid;

resource! {
    /// A memory tier data batches can reside in ("GPU-<n>", "HOST", "DISK").
    MemoryTier {
        capacity: { bytes: Option<u64> },
    }
}

state! {
    // batch_id: process-unique, shared by all placements of one batch.
    // pipeline_uuid: the consumer pipeline (== quent Operator id).
    // port_uuid: the consumer's receiving port.
    // origin: a C++ `batch_origin` value (batch_telemetry.hpp).
    BatchRegistered {
        attributes: {
            batch_id: u64,
            pipeline_uuid: Uuid,
            port_uuid: Uuid,
            origin: String,
        },
        usages: {
            tier: MemoryTier,
        },
    }
}

state! {
    // Waiting in the consumer port's data repository.
    BatchQueued {
        usages: {
            tier: MemoryTier,
        },
    }
}

state! {
    // Bound to a task that has not started computing yet.
    BatchPackaged {
        attributes: {
            task_uuid: Uuid,
        },
        usages: {
            tier: MemoryTier,
        },
    }
}

state! {
    // The bound task is actively computing on this batch.
    BatchProcessing {
        attributes: {
            task_uuid: Uuid,
        },
        usages: {
            tier: MemoryTier,
        },
    }
}

state! {
    // reason: a C++ `batch_consumed_reason` value (batch_telemetry.hpp).
    BatchConsumed {
        attributes: {
            reason: String,
        },
    }
}

fsm! {
    BatchPlacement {
        states: {
            batch_registered: BatchRegistered,
            batch_queued: BatchQueued,
            batch_packaged: BatchPackaged,
            batch_processing: BatchProcessing,
            batch_consumed: BatchConsumed,
        },
        entry: batch_registered,
        exit_from: { batch_consumed },
        transitions: {
            batch_registered => batch_queued,    // published to a repo
            batch_registered => batch_packaged,  // lazy registration at claim
            batch_queued => batch_queued,        // tier change in repo
            batch_queued => batch_packaged,      // claimed by a task
            batch_packaged => batch_packaged,    // tier change or re-claim
            batch_packaged => batch_processing,  // task started computing
            batch_processing => batch_processing, // tier change mid-compute
            batch_processing => batch_consumed,  // processed
            batch_packaged => batch_consumed,    // task died before compute
            batch_queued => batch_consumed,      // query-end drain
        },
    }
}
