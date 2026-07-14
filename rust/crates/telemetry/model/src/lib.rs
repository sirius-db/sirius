use quent_model::{instrumentation, model};

pub mod batch;
pub mod task;

model! {
    name: Sirius,
    root: quent_query_engine_model::engine::Engine,
    entities: {
        quent_query_engine_model::worker::Worker,
        quent_query_engine_model::query_group::QueryGroup,
        quent_query_engine_model::query::Query,
        quent_query_engine_model::plan::Plan,
        quent_query_engine_model::operator::Operator,
        quent_query_engine_model::port::Port,
        task::Task,
        task::TaskQueue,
        task::TaskManagerLoopThread,
        task::ExecutorThread,
        batch::Batch,
        batch::MemoryTier,
    },
    analyzer: "sirius-telemetry-analyzer",
}

instrumentation!(Sirius);

// Re-export query engine model modules for bridge codegen
pub use quent_query_engine_model::{engine, operator, plan, port, query, query_group, worker};
