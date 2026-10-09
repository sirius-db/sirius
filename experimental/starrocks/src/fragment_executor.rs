//! Execution of a translated fragment.
//!
//! A result fragment returns Arrow batches for `fetch_data`; a sender fragment parks its native
//! GPU output under [`SenderSlot`]s for a same-CN receiver to relay in, or for the NIXL transport
//! to export to a remote one. A [`StubExecutor`] stands in for the GPU engine so the StarRocks
//! dispatch and result-return plumbing can be exercised end to end without a build tree or a GPU.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use starrocks_plan_translator::TranslatedPlan;

use crate::local_exchange::RemoteBatch;
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

/// One parked batch exported for a direct exchange: its buffers stay valid until `token` is
/// released on this CN's direct exchange.
#[derive(Debug)]
pub struct ExportedBatch {
    pub token: u64,
    pub rows: u64,
    /// What the receiver allocates matching buffers from.
    pub layout: Vec<u8>,
    /// `(address, length)` of each buffer.
    pub src: Vec<(u64, u64)>,
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

/// Which memory tier a pinned table lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinTier {
    /// GPU memory — fastest scans, bounded by VRAM.
    Gpu,
    /// Pinned host memory — for tables larger than GPU memory.
    Host,
}

impl PinTier {
    /// The engine-side tier string (`pin_table`'s `tier` argument).
    pub fn as_str(self) -> &'static str {
        match self {
            PinTier::Gpu => "gpu",
            PinTier::Host => "host",
        }
    }
}

/// One table to pin into the engine's scan cache, mirroring `CALL pin_table(...)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinTableSpec {
    /// Parquet file or glob; `None` for `format = duckdb`, where `name` is the catalog table.
    pub path: Option<String>,
    /// Memory tier the pinned columns land in.
    pub tier: PinTier,
    /// Pin-registry key; also selects the engine's compression plan file.
    pub name: String,
    /// Columns to pin; `None` pins every column.
    pub cols: Option<Vec<String>>,
    /// `"parquet"` or `"duckdb"`; `None` infers from the path suffix.
    pub format: Option<String>,
    /// Schema containing the table, for `format = duckdb` only.
    pub schema: Option<String>,
}

/// One fragment to run: the plan, where its exchange inputs come from, and where its output goes.
#[derive(Debug)]
pub struct FragmentRun<'a> {
    /// Translated plan, including the schema of every exchange lowered to a stream read.
    pub plan: &'a TranslatedPlan,
    /// Parked sender outputs to relay into this fragment, keyed by receiver exchange node id.
    pub inputs: Vec<(i32, Vec<SenderSlot>)>,
    /// Batches remote senders wrote into this CN, as `(exchange node id, sender id, batches)`.
    pub remote_inputs: Vec<(i32, i32, Vec<RemoteBatch>)>,
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

    /// Exports the next batch parked under `slot` for a direct exchange; `None` once drained.
    fn export_direct_next(&self, slot: SenderSlot) -> Result<Option<ExportedBatch>, String> {
        Err(format!(
            "this executor cannot export the output parked under {slot:?}"
        ))
    }

    /// Drops one destination's claim on parked output, which is freed with the last claim. The
    /// default parks nothing, so there is nothing to drop.
    fn drop_parked(&self, _slot: SenderSlot) -> Result<(), String> {
        Ok(())
    }

    /// Sender fragments still parked, freed or not. The default parks nothing.
    fn parked_fragments(&self) -> usize {
        0
    }

    /// Pins a table into the engine's scan cache (`CALL pin_table` semantics). Blocks for the
    /// whole materialization, serialized with fragment runs on the engine thread. Returns a
    /// one-line summary.
    fn pin_table(&self, spec: &PinTableSpec) -> Result<String, String> {
        let _ = spec;
        Err("this fragment executor does not support pin_table (engine build required)".to_string())
    }

    /// Removes the pinned entry `name`, releasing its memory. Returns a one-line summary.
    fn unpin_table(&self, name: &str) -> Result<String, String> {
        let _ = name;
        Err(
            "this fragment executor does not support unpin_table (engine build required)"
                .to_string(),
        )
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
