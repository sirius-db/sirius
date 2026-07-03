use quent_model::{fsm, state};
use uuid::Uuid;

#[derive(Debug)]
pub enum MemoryTier {
    Gpu,
    Host,
    Storage,
}

state! {
    Constructed {
        attributes: {
            data_batch_id: u64,
            producer_pipeline_uuid: Uuid,
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
            stationary => destructed,
        },
    }
}
