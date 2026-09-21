//! Execution of a translated fragment into Arrow result batches.
//!
//! A result fragment returns Arrow for `fetch_data`. A sender fragment parks native GPU output
//! under [`SenderSlot`]s; a same-CN receiver relays those slots, a remote hop exports them as
//! packed GPU bytes. [`StubExecutor`] fabricates rows so dispatch can be exercised without a GPU.

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

/// One packed batch sitting in an exchange staging arena as cudf packed bytes.
///
/// The wire shape of the NIXL hop: on the sender it names a lease in the *local* arena
/// (filled by `export_packed`); on the receiver a lease in the *receiver's* arena (filled
/// by a NIXL WRITE). `len == 0` means no lease exists for this batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedBatch {
    /// Host-side cudf pack metadata (travels on `transmit_chunk` kind Packed; the device payload does not).
    pub metadata: Vec<u8>,
    /// Byte offset of the packed payload from the arena base. `0` with `len == 0` means no
    /// lease exists for this batch.
    pub offset: u64,
    /// Length of the packed payload in bytes.
    pub len: u64,
    /// Exact row count of the packed table, when the frame carried `X-Rows`.
    pub rows: Option<u64>,
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
    /// Remote sender batches already on this CN, as `(exchange node id, sender id, batches)`.
    /// Pushed with `push_packed` then `close_input` before `run()`.
    pub remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
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
/// The seam is synchronous and one fragment at a time: the engine serializes queries. A sender's
/// output is parked on the GPU until its receiver is dispatched. `exec_plan_fragment` runs this
/// on a `spawn_blocking` worker so the BRPC runtime stays free for `fetch_data`.
pub trait FragmentExecutor: std::fmt::Debug + Send + Sync {
    /// Executes `translated` as a result fragment (no exchange inputs, no output streams).
    fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String>;

    /// Runs one fragment. Returns rows only when `outputs` is empty (a result fragment).
    /// The default path ignores parked/remote inputs and is enough for the stub.
    fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        if !run.outputs.is_empty() {
            return Ok(None);
        }
        self.execute(run.plan).map(Some)
    }

    /// Exchange staging arena `(device base address, capacity in bytes)`. Errors when this
    /// executor has no arena.
    fn staging_info(&self) -> Result<(u64, u64), String> {
        Err("this fragment executor has no exchange staging arena \
             (engine build with SIRIUS_EXCHANGE_STAGING_BYTES required)"
            .to_string())
    }

    /// Leases `len` bytes of the staging arena, returning the lease offset from the base.
    fn staging_lease(&self, len: u64) -> Result<u64, String> {
        let _ = len;
        Err("this fragment executor has no exchange staging arena \
             (engine build with SIRIUS_EXCHANGE_STAGING_BYTES required)"
            .to_string())
    }

    /// Returns the staging lease at `offset`.
    fn staging_release(&self, offset: u64) -> Result<(), String> {
        let _ = offset;
        Err("this fragment executor has no exchange staging arena \
             (engine build with SIRIUS_EXCHANGE_STAGING_BYTES required)"
            .to_string())
    }

    /// Packs the next batch parked under `slot` into a fresh staging lease; `Ok(None)` once
    /// the parked output is drained. The lease stays outstanding until
    /// [`staging_release`](Self::staging_release).
    fn export_packed_next(&self, slot: SenderSlot) -> Result<Option<StagedBatch>, String> {
        let _ = slot;
        Err("this fragment executor cannot export packed batches \
             (engine build with SIRIUS_EXCHANGE_STAGING_BYTES required)"
            .to_string())
    }

    /// Drops this destination's claim on parked output. The fragment is freed with the last
    /// claim. The default is a no-op because a stub parks nothing.
    fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
        let _ = slot;
        Ok(())
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
