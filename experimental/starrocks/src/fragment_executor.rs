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

/// One packed batch sitting in an exchange staging arena as cudf packed bytes.
///
/// The neutral wire shape of the nixl tier: on the sender it names a lease in the *local* arena
/// (filled by `export_packed`), on the receiver a lease in the *receiver's* arena (filled by a
/// nixl WRITE). Whoever holds the arena releases the lease after the bytes leave it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedBatch {
    /// Host-side cudf pack metadata (travels over brpc next to the device payload).
    pub metadata: Vec<u8>,
    /// Byte offset of the packed payload from the arena base. `0` with `len == 0` means no
    /// lease exists for this batch (a metadata-only empty batch) — nothing to release.
    pub offset: u64,
    /// Length of the packed payload in bytes.
    pub len: u64,
    /// Exact row count of the packed table (from `export_packed`, carried on the transmit
    /// frame). The receiver sums the counts per stream into `declare_input_cardinality` so the
    /// optimizer can size the stream. `None` when the frame predates the wire field: the
    /// receiver then declares nothing for the stream and keeps the legacy blind planning.
    pub rows: Option<u64>,
}

/// One fragment to run: the plan, where its exchange inputs come from, and where its output goes.
#[derive(Debug)]
pub struct FragmentRun<'a> {
    /// Translated plan, including the schema of every exchange lowered to a stream read.
    pub plan: &'a TranslatedPlan,
    /// Parked sender outputs to relay into this fragment, keyed by receiver exchange node id.
    pub inputs: Vec<(i32, Vec<SenderSlot>)>,
    /// Remote sender outputs already staged in this CN's arena, as
    /// `(exchange node id, sender id, batches)`: pushed via `push_packed` + `close_input`
    /// before the fragment runs, with each lease released the moment its push returns.
    pub remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
    /// Non-empty for a sender fragment: the fragment parks ONCE and output stream i belongs to
    /// destination `outputs[i]` (the FE's destination order). Each destination drains its own
    /// stream; the parked fragment drops when the last destination releases it.
    pub outputs: Vec<SenderSlot>,
    /// Every destination receives the full output (a broadcast sink). With `outputs.len() > 1`
    /// and `broadcast == false`, `hash_keys` routes rows instead.
    pub broadcast: bool,
    /// Hash-partition key columns (output column indices, in the exchange's shared
    /// partition-expression order). Non-empty exactly for a hash-partitioned fan-out.
    pub hash_keys: Vec<usize>,
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

    /// Exchange staging arena `(device base address, capacity in bytes)`, for transport memory
    /// registration. Errors when the executor has no arena — the default for every executor
    /// that is not the engine with `SIRIUS_EXCHANGE_STAGING_BYTES` set.
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

    /// Packs the next batch parked under `slot` into a fresh staging lease; `Ok(None)` once the
    /// parked output is drained. The lease stays outstanding until
    /// [`staging_release`](Self::staging_release).
    fn export_packed_next(&self, slot: SenderSlot) -> Result<Option<StagedBatch>, String> {
        let _ = slot;
        Err("this fragment executor cannot export packed batches \
             (engine build with SIRIUS_EXCHANGE_STAGING_BYTES required)"
            .to_string())
    }

    /// Drops the parked fragment under `slot`, releasing the GPU memory its batches hold. Called
    /// after the drained output has been transmitted (or on a failed transmit, so a wedged
    /// cross-node query does not pin its output for the process lifetime).
    fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
        let _ = slot;
        Err("this fragment executor parks nothing to drop".to_string())
    }
}

/// Placeholder executor that fabricates one row so the result path works without a GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubExecutor;

impl FragmentExecutor for StubExecutor {
    fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        // A stub sender parks nothing; the rendezvous only needs it to succeed.
        if !run.outputs.is_empty() {
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
        // For now we emit one placeholder string row per output column so the FE→client path is
        // exercised.
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
            output_partition_columns: None,
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
