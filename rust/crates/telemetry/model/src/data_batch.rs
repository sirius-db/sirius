use quent_model::{fsm, state};
use uuid::Uuid;

state! {
    // producer_task_uuid: the task whose execution constructed this batch;
    // nil for batches created outside a task (scan-manager staging, tests).
    // num_rows/num_columns: 0 when unknown at construction (e.g. host-tier
    // staging batches that have not been decoded).
    Constructed {
        attributes: {
            data_batch_id: u64,
            producer_pipeline_uuid: Uuid,
            producer_task_uuid: Uuid,
            num_rows: u64,
            num_columns: u64,
        },
    }
}

state! {
    Stationary {
        usages: {
            memory: quent_stdlib::memory::Memory,
        },
    }
}

state! {
    InTransit {
        usages: {
            source_memory: quent_stdlib::memory::Memory,
            dest_memory: quent_stdlib::memory::Memory,
            channel: quent_stdlib::channel::Channel,
        },
    }
}

state! {
    Destructed {}
}

fsm! {
    DataBatch {
        states: {
            constructed: Constructed,
            stationary: Stationary,
            in_transit: InTransit,
            destructed: Destructed,
        },
        entry: constructed,
        exit_from: { destructed },
        transitions: {
            constructed => stationary,
            stationary => in_transit,
            in_transit => stationary,
            stationary => stationary,
            stationary => destructed,
        },
    }
}
