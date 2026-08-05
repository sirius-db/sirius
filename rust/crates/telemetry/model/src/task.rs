use quent_model::{fsm, resource, state};
use uuid::Uuid;

resource! {
    /// A thread coordinating the execution of tasks.
    TaskManagerLoopThread
}

resource! {
    /// A thread executing the tasks.
    ExecutorThread
}

resource! {
    /// A queue that manages tasks.
    TaskQueue {
        capacity: { entries: Option<u64> },
    }
}

state! {
    Created {
        attributes: {
            pipeline_uuid: Uuid,
        },
    }
}

state! {
    Queued {
        usages: {
            queue: TaskQueue,
        },
    }
}

state! {
    Routing {
        attributes: {
            preferred_device_id: i64,
        },
        usages: {
            manager_thread: TaskManagerLoopThread,
        },
    }
}

state! {
    Reserving {
        attributes: {
            requested_bytes: u64,
            input_basis: u64,
            peak_estimate: u64,
            bytes_to_materialize: u64,
        },
        usages: {
            manager_thread: TaskManagerLoopThread,
            // TODO: add memory resource usage
        },
    }
}

state! {
    Downgrading {
        attributes: {
            shortfall_bytes: u64,
            partial_bytes: u64
        },
        usages: {
            manager_thread: TaskManagerLoopThread,
        },
    }
}

state! {
    // reservation: the processing space (memory reservation) this task holds
    // while it materializes inputs and computes, on the per-device tier
    // resource.
    Preparing {
        attributes: {
            origin_tier: String,
            target_tier: String,
            input_bytes: u64,
        },
        usages: {
            executor_thread: ExecutorThread,
            reservation: crate::batch::MemoryTier,
        },
    }
}

state! {
    // peak_allocated_bytes: allocator peak on this task's stream at the moment
    // this operator started (i.e. after the previous operator finished).
    // input_rows: total rows across the operator's input batches; 0 when the
    // input carries no row-counted batches (e.g. a scan split before decode).
    // Since op i's input is op i-1's output, this is also the previous
    // operator's output row count.
    Computing {
        attributes: {
            current_operator_id: u32,
            input_bytes: u64,
            peak_allocated_bytes: u64,
            input_rows: u64,
        },
        usages: {
            executor_thread: ExecutorThread,
            reservation: crate::batch::MemoryTier,
        },
    }
}

state! {
    // output_rows/output_bytes: the task's final output volume (the last
    // operator's output, before publish); 0 on failure paths or when the task
    // produced no batch output.
    Finalizing {
        attributes: {
            success: bool,
            output_rows: u64,
            output_bytes: u64,
        },
    }
}

fsm! {
    Task {
        states: {
            created: Created,
            queued: Queued,
            routing: Routing,
            reserving: Reserving,
            downgrading: Downgrading,
            preparing: Preparing,
            computing: Computing,
            finalizing: Finalizing,
        },
        entry: created,
        exit_from: { finalizing },
        transitions: {
            // Task object exists and has entered its first scheduling queue.
            created => queued,

            // A manager thread popped a queued task and is choosing where it should run.
            queued => routing,

            // GPU pipeline tasks are routed from the scheduler queue into a selected
            // per-GPU executor queue.
            routing => queued,

            // Scan/source tasks are routed and then reserve memory in the same scan
            // manager loop.
            routing => reserving,
            // GPU tasks reserve after being popped from an executor queue.
            queued => reserving,

            // The reservation request could not be fully satisfied, so the task invokes
            // a downgrade request to free up space.
            reserving => downgrading,


            // Reservation succeeded and the task is ready to be prepared on a worker.
            reserving => preparing,
            // A task may fail to acquire a reservation right-away, and requests the engine to free up space
            // via downgrading to retry the reservation, which, if successful, allows the task to start preparing.
            downgrading => preparing,


            // Worker preparation completed and operator execution has started.
            preparing => computing,

            // GPU pipeline tasks emit one compute event per operator;
            // scan/source tasks usually emit one compute event for the source operator.
            computing => computing,

            // Normal completion or execution failure after compute.
            computing => finalizing,

            // A task was created but never reached a queue. This is a valid
            // cleanup path, but should be treated as abnormal by analysis.
            created => finalizing,
            // Queue drain, interrupted scheduling, or cancellation before routing.
            queued => finalizing,
            // After routing, enqueue to executor can fail/drop if the
            // executor queue is interrupted.
            routing => finalizing,
            // Reservation could fail.
            reserving => finalizing,
            // Downgrade request can be canceled/fail before returning.
            downgrading => finalizing,
            // Upgrading data to required tier might fail.
            preparing => finalizing,
        },
    }
}
