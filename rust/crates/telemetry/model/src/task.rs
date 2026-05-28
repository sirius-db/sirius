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
    /// A queue to enqueue stuff.
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
            requested_bytes: i64,
            input_basis: i64,
            peak_estimate: i64,
            bytes_to_materialize: i64,
        },
        usages: {
            manager_thread: TaskManagerLoopThread,
        },
    }
}

state! {
    Downgrading {
        attributes: {
            shortfall_bytes: i64,
            partial_bytes: i64
        },
        usages: {
            manager_thread: TaskManagerLoopThread,
        },
    }
}

state! {
    Preparing {
        attributes: {
            target_tier: String,
        },
        usages: {
            executor_thread: ExecutorThread,
        },
    }
}

state! {
    Computing {
        attributes: {
            current_operator_id: u64,
            output_bytes: u64,
        },
        usages: {
            executor_thread: ExecutorThread,
        },
    }
}

state! {
    Finalizing {
        attributes: {
            success: bool,
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
            created => queued,
            queued => routing,
            routing => reserving,
            routing => queued,      // GPU task routed into selected executor queue
            queued => reserving,    // executor manager pops and starts reservation
            routing => reserving,   // scan/source route and reserve in same manager loop
            reserving => downgrading,
            downgrading => reserving,
            reserving => finalizing,
            reserving => preparing,
            preparing => computing,
            preparing => finalizing,
            computing => computing,
            computing => finalizing,
        },
    }
}
