use quent_model::{instrumentation, model};

model! {
    Sirius {
        root: quent_query_engine_model::engine::Engine,
        quent_query_engine_model::worker::Worker,
        quent_query_engine_model::query_group::QueryGroup,
        quent_query_engine_model::query::Query,
        quent_query_engine_model::plan::Plan,
        quent_query_engine_model::operator::Operator,
        quent_query_engine_model::port::Port,
        quent_stdlib::memory::Memory,
        quent_stdlib::processor::Processor,
        quent_stdlib::channel::Channel,
    }
}

instrumentation!(Sirius);

// Re-export query engine model modules for bridge codegen
pub use quent_query_engine_model::{engine, operator, plan, port, query, query_group, worker};
pub use quent_stdlib::{channel, memory, processor};
