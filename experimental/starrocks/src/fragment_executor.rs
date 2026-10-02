//! Execution of a translated fragment.
//!
//! A result fragment returns Arrow batches for `fetch_data`; a sender fragment parks its native
//! GPU output under [`SenderSlot`]s for a same-CN receiver to relay in. A [`StubExecutor`] stands
//! in for the GPU engine so the StarRocks dispatch and result-return plumbing can be exercised end
//! to end without a build tree or a GPU.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use starrocks_plan_translator::TranslatedPlan;

use crate::result_store::FragmentInstanceId;

/// Where one sender fragment's output is parked until its receiver runs.
///
/// Keyed by the *receiver* it feeds: a sender is addressed by the exchange it produces into,
/// not by its own identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SenderSlot {
    /// Receiver fragment instance the output is destined for.
    pub(crate) fragment_instance_id: FragmentInstanceId,
    /// Receiver `EXCHANGE_NODE` id, which is also the engine-side stream id.
    pub(crate) node_id: i32,
    /// Sender ordinal within that exchange's sender set.
    pub(crate) sender_id: i32,
}

/// Output of executing one plan fragment: Arrow batches matching the fragment output schema.
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

/// One fragment to run: the plan, where its exchange inputs come from, and where its output goes.
#[derive(Debug)]
pub struct FragmentRun<'a> {
    /// Translated plan, including the schema of every exchange lowered to a stream read.
    pub plan: &'a TranslatedPlan,
    /// Parked sender outputs to relay into this fragment, keyed by receiver exchange node id.
    pub inputs: Vec<(i32, Vec<SenderSlot>)>,
    /// Non-empty for a sender fragment: park once, output stream i belongs to `outputs[i]`.
    pub outputs: Vec<SenderSlot>,
    /// Every destination receives the full output (a broadcast sink).
    pub broadcast: bool,
    /// Hash-partition key columns for a hash fan-out (empty otherwise).
    pub hash_keys: Vec<usize>,
}

/// Runs a translated fragment, either parking its output for a downstream fragment or returning
/// its rows.
///
/// This is intentionally a synchronous, fully-materializing seam: `exec_plan_fragment` runs a
/// fragment to completion before returning, and `fetch_data` then drains the buffered rows.
///
/// TODO(starrocks-execute): a real GPU executor should not block dispatch on full materialization.
/// Evolve this into a streaming contract — dispatch registers a running fragment and returns after
/// startup, the executor pushes Arrow batches (e.g. via an Arrow C stream) into a bounded channel
/// the `ResultStore` drains, and execution is cancellable from `cancel_plan_fragment`. Large/slow
/// result queries then stream through `fetch_data` instead of risking dispatch-time timeout/OOM.
pub trait FragmentExecutor: std::fmt::Debug + Send + Sync {
    /// Executes `translated` as a result fragment (no exchange inputs, no output streams).
    fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String>;

    /// Runs one fragment. Returns rows only when `outputs` is empty (a result fragment). The
    /// default ignores parked inputs and parks nothing, which is enough for the stub.
    fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        if !run.outputs.is_empty() {
            return Ok(None);
        }
        self.execute(run.plan).map(Some)
    }
}

/// Placeholder executor that fabricates one row so the result path works without a GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubExecutor;

impl FragmentExecutor for StubExecutor {
    fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
        // TODO(starrocks-execute): replace with a SiriusExecutor that hands
        // `translated.to_substrait_bytes()` to the embedded Sirius engine, executes it on the
        // GPU, and imports the result via the Arrow C Data Interface. That executor will hold an
        // `Arc<sirius::SiriusContext>` threaded in from `main` (see `BrpcServer::new`). For now
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
            output_partition_columns: None,
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
