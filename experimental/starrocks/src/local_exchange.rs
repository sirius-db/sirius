//! Sequential same-node exchange for the single-shot Sirius API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use starrocks_thrift::internal_service::TExecPlanFragmentParams;

use crate::fragment_executor::FragmentResult;
use crate::result_store::FragmentInstanceId;

/// Receiver identity used by both the stream sink and exchange node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExchangeKey {
    pub(crate) fragment_instance_id: FragmentInstanceId,
    pub(crate) node_id: i32,
}

/// One fully materialized sender result awaiting its receiver fragment.
#[derive(Clone, Debug)]
pub(crate) struct ExchangeOutput {
    pub(crate) names: Vec<String>,
    pub(crate) result: FragmentResult,
}

/// One complete materialized input of a receiver fragment.
#[derive(Debug)]
pub(crate) struct ReadyExchangeInput {
    pub(crate) node_id: i32,
    pub(crate) outputs: Vec<ExchangeOutput>,
}

/// A receiver fragment whose exchange inputs are all ready for sequential execution.
#[derive(Debug)]
pub(crate) struct ReadyFragment {
    pub(crate) params: TExecPlanFragmentParams,
    pub(crate) inputs: Vec<ReadyExchangeInput>,
}

#[derive(Debug)]
struct PendingReceiver {
    params: TExecPlanFragmentParams,
    expected_senders: HashMap<i32, usize>,
}

#[derive(Debug, Default)]
struct ExchangeState {
    receivers: HashMap<FragmentInstanceId, PendingReceiver>,
    outputs: HashMap<ExchangeKey, HashMap<i32, ExchangeOutput>>,
}

/// Matches receiver-first StarRocks dispatch with later same-node sender results.
#[derive(Debug, Default)]
pub(crate) struct LocalExchange {
    inner: Mutex<ExchangeState>,
}

impl LocalExchange {
    /// Registers a receiver fragment, returning it when every exchange input is complete.
    pub(crate) fn register_receiver(
        &self,
        fragment_instance_id: FragmentInstanceId,
        expected_senders: Vec<(i32, usize)>,
        params: TExecPlanFragmentParams,
    ) -> Result<Option<ReadyFragment>, String> {
        if expected_senders.is_empty() {
            return Err("receiver fragment has no exchange inputs".to_string());
        }
        let mut expected_by_node = HashMap::with_capacity(expected_senders.len());
        for (node_id, expected) in expected_senders {
            if expected == 0 {
                return Err(format!("exchange node {node_id} expects no senders"));
            }
            if expected_by_node.insert(node_id, expected).is_some() {
                return Err(format!(
                    "duplicate exchange node {node_id} in receiver fragment"
                ));
            }
        }
        let mut state = self.lock();
        if state.receivers.contains_key(&fragment_instance_id) {
            return Err(format!(
                "duplicate receiver registration for fragment {fragment_instance_id}"
            ));
        }
        state.receivers.insert(
            fragment_instance_id,
            PendingReceiver {
                params,
                expected_senders: expected_by_node,
            },
        );
        Self::take_ready(&mut state, fragment_instance_id)
    }

    /// Buffers one sender output, returning the receiver when this completes its sender set.
    pub(crate) fn push_sender(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        output: ExchangeOutput,
    ) -> Result<Option<ReadyFragment>, String> {
        let mut state = self.lock();
        let senders = state.outputs.entry(key).or_default();
        if senders.contains_key(&sender_id) {
            return Err(format!("duplicate sender {sender_id} for exchange {key:?}"));
        }
        senders.insert(sender_id, output);
        Self::take_ready(&mut state, key.fragment_instance_id)
    }

    fn take_ready(
        state: &mut ExchangeState,
        fragment_instance_id: FragmentInstanceId,
    ) -> Result<Option<ReadyFragment>, String> {
        let Some(receiver) = state.receivers.get(&fragment_instance_id) else {
            return Ok(None);
        };
        for (&node_id, &expected) in &receiver.expected_senders {
            let key = ExchangeKey {
                fragment_instance_id,
                node_id,
            };
            let actual = state.outputs.get(&key).map(HashMap::len).unwrap_or(0);
            if actual > expected {
                return Err(format!(
                    "exchange {key:?} received {actual} senders but expected {expected}"
                ));
            }
            if actual != expected {
                return Ok(None);
            }
        }

        let receiver = state
            .receivers
            .remove(&fragment_instance_id)
            .expect("receiver checked above");
        let mut node_ids = receiver.expected_senders.into_keys().collect::<Vec<_>>();
        node_ids.sort_unstable();
        let inputs = node_ids
            .into_iter()
            .map(|node_id| {
                let key = ExchangeKey {
                    fragment_instance_id,
                    node_id,
                };
                let mut senders = state.outputs.remove(&key).unwrap_or_default();
                let mut sender_ids = senders.keys().copied().collect::<Vec<_>>();
                sender_ids.sort_unstable();
                let outputs = sender_ids
                    .into_iter()
                    .map(|sender_id| senders.remove(&sender_id).expect("sender id came from map"))
                    .collect();
                ReadyExchangeInput { node_id, outputs }
            })
            .collect();
        Ok(Some(ReadyFragment {
            params: receiver.params,
            inputs,
        }))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ExchangeState> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Temporary parquet file used as the receiver's `local_files` exchange input.
#[derive(Debug)]
pub(crate) struct ExchangeFile {
    path: PathBuf,
    pub(crate) names: Vec<String>,
}

impl ExchangeFile {
    /// Writes every sender batch to one parquet file in deterministic sender order.
    pub(crate) fn materialize(outputs: &[ExchangeOutput]) -> Result<Self, String> {
        let first = outputs
            .first()
            .ok_or_else(|| "cannot materialize an exchange without sender outputs".to_string())?;
        let sender_schema = first.result.schema.clone();
        let names = first.names.clone();
        if sender_schema.fields().len() != names.len() {
            return Err(format!(
                "exchange sender produced {} schema fields but {} output names",
                sender_schema.fields().len(),
                names.len()
            ));
        }
        for output in outputs {
            if output.names != names {
                return Err("exchange senders produced different output names".to_string());
            }
            if output.result.schema != sender_schema {
                return Err("exchange senders produced different Arrow schemas".to_string());
            }
        }

        // Root output names are the parquet column names consumed by the receiver's base schema.
        // Re-label batches without copying their arrays in case the Arrow exporter chose different
        // field names from the Substrait root.
        let fields = sender_schema
            .fields()
            .iter()
            .zip(&names)
            .map(|(field, name)| field.as_ref().clone().with_name(name))
            .collect::<Vec<_>>();
        let schema = std::sync::Arc::new(Schema::new_with_metadata(
            fields,
            sender_schema.metadata().clone(),
        ));

        let exchange_file = Self {
            path: Self::next_path()?,
            names,
        };
        let file = std::fs::File::create(&exchange_file.path).map_err(|err| {
            format!(
                "failed to create exchange file {}: {err}",
                exchange_file.path.display()
            )
        })?;
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None)
            .map_err(|err| format!("failed to create exchange parquet writer: {err}"))?;
        for output in outputs {
            for batch in &output.result.batches {
                let batch = RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
                    .map_err(|err| format!("failed to name exchange batch columns: {err}"))?;
                writer
                    .write(&batch)
                    .map_err(|err| format!("failed to write exchange batch: {err}"))?;
            }
        }
        writer
            .close()
            .map_err(|err| format!("failed to finish exchange parquet file: {err}"))?;
        Ok(exchange_file)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    fn next_path() -> Result<PathBuf, String> {
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join("sirius-starrocks-cn");
        std::fs::create_dir_all(&dir).map_err(|err| {
            format!(
                "failed to create exchange directory {}: {err}",
                dir.display()
            )
        })?;
        let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        Ok(dir.join(format!(
            "exchange-{}-{sequence}.parquet",
            std::process::id()
        )))
    }
}

impl Drop for ExchangeFile {
    fn drop(&mut self) {
        if let Err(err) = std::fs::remove_file(&self.path) {
            tracing::warn!(path = %self.path.display(), error = %err, "failed to remove exchange file");
        }
    }
}
