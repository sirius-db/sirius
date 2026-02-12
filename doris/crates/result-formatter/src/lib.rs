//! GPU result formatting for Doris.
//!
//! Converts Sirius GPU execution results (Arrow IPC bytes from the C++ bridge)
//! into formats consumable by Doris clients (Arrow Flight, TRowBatch).

pub mod arrow_flight;
pub mod result_store;
