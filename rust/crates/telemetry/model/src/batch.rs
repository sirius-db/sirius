//! Batch placement telemetry.
//!
//! A Batch entity tracks one *placement* of a physical data batch: the
//! lifecycle of a batch published to one consuming pipeline's input port.
//! Fan-out (the same physical batch pushed to multiple consumer ports)
//! produces one placement per consumer; all placements share the engine's
//! process-unique `batch_id` attribute. Re-packaging after an OOM reschedule
//! keeps the same placement entity (a `packaged => packaged` self-transition
//! with a new `task_uuid`).
//!
//! Every non-terminal state carries a `tier` usage on one of the MemoryTier
//! resources ("GPU", "HOST", "DISK") with `bytes` = the batch's data size, so
//! a tier change (downgrade/spill or prepare-time upgrade) is a self-
//! transition with a different tier usage.

use quent_model::{fsm, resource, state};
use uuid::Uuid;

resource! {
    /// A memory tier data batches can reside in. One instance per tier:
    /// "GPU", "HOST", "DISK".
    MemoryTier {
        capacity: { bytes: Option<u64> },
    }
}

state! {
    // batch_id: the engine's process-unique batch id, shared by all
    //   placements/re-packagings of one physical batch.
    // pipeline_uuid: the consumer pipeline (== quent Operator id).
    // port_uuid: the consumer pipeline's source port receiving the batch.
    // origin: "operator_output" | "partition_output" | "cpu_source" |
    //   "reschedule_intermediate" | "pinned_cache".
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
    // Waiting in the consumer port's shared data repository for the scheduler
    // to package it into a task.
    BatchQueued {
        usages: {
            tier: MemoryTier,
        },
    }
}

state! {
    // Bound to a task as input data; the task is queued/routing/reserving/
    // preparing.
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
    // reason: "processed" | "task_failed" | "query_end".
    BatchConsumed {
        attributes: {
            reason: String,
        },
    }
}

fsm! {
    Batch {
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
            // Normal publish: the producing operator pushed the batch into the
            // consumer port's shared data repository.
            batch_registered => batch_queued,
            // Lazy registration: the batch first became visible to telemetry
            // when a task claimed it (OOM-reschedule intermediates, pinned
            // cache inputs).
            batch_registered => batch_packaged,

            // Tier change (downgrade/spill or upgrade) while idle in a repo.
            batch_queued => batch_queued,

            // A task popped the batch from the repo as part of its input data.
            batch_queued => batch_packaged,

            // Tier change while bound to a queued task, or re-packaged into a
            // rescheduled task (new task_uuid) after the original task died.
            batch_packaged => batch_packaged,

            // The task finished preparing and started computing on the batch;
            // the tier usage reflects post-upgrade residency.
            batch_packaged => batch_processing,

            // Defensive: tier change mid-compute.
            batch_processing => batch_processing,

            // The task holding the batch finished with it.
            batch_processing => batch_consumed,
            // The task was destroyed before compute (error paths).
            batch_packaged => batch_consumed,
            // Query-end drain of batches still sitting in repos.
            batch_queued => batch_consumed,
        },
    }
}
