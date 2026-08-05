//! Execution of a translated fragment into Arrow result batches.
//!
//! The engine→CN result interchange is the Arrow C Data Interface (see the `SiriusExecutor`
//! TODO). Today a [`StubExecutor`] stands in for the GPU engine so the StarRocks dispatch and
//! result-return plumbing can be exercised end to end without a build tree or a GPU.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use starrocks_plan_translator::TranslatedPlan;

use crate::result_store::FragmentInstanceId;

/// Output of executing one plan fragment: Arrow batches matching the fragment output schema.
///
/// Only a *result* fragment produces one. An intermediate fragment's output never becomes Arrow —
/// it stays on the GPU as native batches for the fragment that consumes it.
#[derive(Clone, Debug)]
pub struct FragmentResult {
    /// Result batches in fragment output order. Empty for a fragment with no output columns.
    pub(crate) batches: Vec<RecordBatch>,
}

impl FragmentResult {
    /// Builds a result from its output batches (in fragment output order).
    pub fn new(batches: Vec<RecordBatch>) -> Self {
        Self { batches }
    }

    /// The result batches in fragment output order.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }
}

/// Where one sender fragment's output is parked until its receiver runs.
///
/// Keyed by the *receiver* it feeds, because that is what the rendezvous looks it up by: a sender
/// is addressed by the exchange it produces into, not by its own identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SenderSlot {
    /// Receiver fragment instance the output is destined for.
    pub fragment_instance_id: FragmentInstanceId,
    /// Receiver `EXCHANGE_NODE` id, which is also the engine-side stream id.
    pub node_id: i32,
    /// Sender ordinal within that exchange's sender set.
    pub sender_id: i32,
}

/// One fragment to run: the plan, where its exchange inputs come from, and where its output goes.
#[derive(Debug)]
pub struct FragmentRun<'a> {
    /// Translated plan, including the schema of every exchange lowered to a stream read.
    pub plan: &'a TranslatedPlan,
    /// Parked sender outputs to relay into this fragment, keyed by receiver exchange node id.
    pub inputs: Vec<(i32, Vec<SenderSlot>)>,
    /// Set for a sender fragment: its output parks under this slot instead of returning rows.
    pub output: Option<SenderSlot>,
}

/// Runs a translated fragment, either parking its output for a downstream fragment or returning
/// its rows.
///
/// The seam is synchronous and one fragment at a time: the engine serializes queries, and a
/// sender's output is parked on the GPU — as native batches, not Arrow — until its receiver is
/// dispatched. A sender's rows therefore never leave the device between fragments.
///
/// TODO(starrocks-execute): dispatch still blocks until a fragment completes. Concurrency needs
/// per-query lifecycle isolation in the engine; until then `run` is called from a blocking worker
/// so the BRPC runtime stays responsive.
pub trait FragmentExecutor: std::fmt::Debug + Send + Sync {
    /// Runs `run`. Returns rows only for a fragment with no `output` slot — a result fragment.
    fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String>;
}

/// Placeholder executor that fabricates one row so the result path works without a GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubExecutor;

impl FragmentExecutor for StubExecutor {
    fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        // A stub sender parks nothing; the rendezvous only needs it to succeed.
        if run.output.is_some() {
            return Ok(None);
        }
        self.execute(run.plan).map(Some)
    }
}

impl StubExecutor {
    /// One placeholder row per output column, so the FE→client path works without a GPU.
    fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
        // TODO(starrocks-execute): replace with a SiriusExecutor that hands
        // `translated.to_substrait_bytes()` to the embedded Sirius engine, executes it on the
        // GPU, and imports the result via the Arrow C Data Interface. That executor will hold an
        // `Arc<sirius::SiriusContext>` threaded in from `main` (see `BrpcServer::with_executor`).
        // For now
        // we emit one placeholder string row per output column so the FE→client path is exercised.
        let names = &translated.output_names;
        if names.is_empty() {
            return Ok(FragmentResult {
                batches: Vec::new(),
            });
        }
        let fields: Vec<Field> = names
            .iter()
            .map(|name| Field::new(name, DataType::Utf8, true))
            .collect();
        let columns: Vec<ArrayRef> = names
            .iter()
            .map(|_| Arc::new(StringArray::from(vec![Some("stub")])) as ArrayRef)
            .collect();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
            .map_err(|err| format!("failed to build stub result batch: {err}"))?;
        Ok(FragmentResult {
            batches: vec![batch],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_with_outputs(names: &[&str]) -> TranslatedPlan {
        TranslatedPlan {
            plan: Default::default(),
            output_names: names.iter().map(|name| name.to_string()).collect(),
            stream_inputs: Vec::new(),
        }
    }

    #[test]
    fn stub_executor_emits_one_row_matching_output_names() {
        let result = StubExecutor
            .execute(&plan_with_outputs(&["id", "name"]))
            .unwrap();
        assert_eq!(result.batches.len(), 1);
        let batch = &result.batches[0];
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.schema().field(0).name(), "id");
        assert_eq!(batch.schema().field(1).name(), "name");
    }

    #[test]
    fn stub_executor_handles_empty_output() {
        let result = StubExecutor.execute(&plan_with_outputs(&[])).unwrap();
        assert!(result.batches.is_empty());
    }
}
