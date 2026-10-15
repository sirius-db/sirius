//! Execution of a translated fragment into Arrow result batches.
//!
//! The engine→BE result interchange is the Arrow C Data Interface. [`StubExecutor`] is the
//! engine-less (`--no-default-features`) executor; [`crate::SiriusEngine`] runs GPU plans in
//! engine-linked builds.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use doris_plan_translator::TranslatedPlan;

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

    /// Moves batches out so the result encoder can release each Arrow batch after encoding it.
    pub fn into_batches(self) -> Vec<RecordBatch> {
        self.batches
    }
}

/// Runs a translated fragment and returns its result batches.
///
/// This is intentionally a synchronous, fully-materializing seam for the single-fragment
/// milestone: `exec_plan_fragment` runs it to completion before returning, and `fetch_data` then
/// drains the buffered rows.
///
/// The streaming follow-up is tracked in #137: dispatch should register a running fragment and return after
/// startup, the executor pushes Arrow batches (e.g. via an Arrow C stream) into a bounded channel
/// the `ResultStore` drains. Cancellation currently skips requests queued for the engine; a
/// request already executing cannot be interrupted. Large/slow result queries should stream
/// through `fetch_data` instead of risking dispatch-time timeout/OOM.
pub trait FragmentExecutor: std::fmt::Debug + Send + Sync {
    /// Executes `translated` and returns its Arrow result batches.
    fn execute(
        &self,
        translated: &TranslatedPlan,
        cancelled: Arc<AtomicBool>,
    ) -> Result<FragmentResult, String>;
}

/// Placeholder executor that fabricates one row so the result path works without a GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubExecutor;

impl FragmentExecutor for StubExecutor {
    fn execute(
        &self,
        translated: &TranslatedPlan,
        cancelled: Arc<AtomicBool>,
    ) -> Result<FragmentResult, String> {
        if cancelled.load(Ordering::Acquire) {
            return Err("query was cancelled before execution".to_string());
        }
        // Emit one placeholder string row per output column for the engine-less protocol path.
        let names = translated.output_names();
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
        use substrait::proto::{Plan, PlanRel, RelRoot, plan_rel};
        TranslatedPlan::new(Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: None,
                    names: names.iter().map(|name| name.to_string()).collect(),
                })),
            }],
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn stub_executor_emits_one_row_matching_output_names() {
        let result = StubExecutor
            .execute(
                &plan_with_outputs(&["id", "name"]),
                Arc::new(AtomicBool::new(false)),
            )
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
        let result = StubExecutor
            .execute(&plan_with_outputs(&[]), Arc::new(AtomicBool::new(false)))
            .unwrap();
        assert!(result.batches.is_empty());
    }
}
