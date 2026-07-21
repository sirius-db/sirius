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

/// A receiver whose complete sender set is ready for sequential execution.
#[derive(Debug)]
pub(crate) struct ReadyExchange {
    pub(crate) params: TExecPlanFragmentParams,
    pub(crate) key: ExchangeKey,
    pub(crate) outputs: Vec<ExchangeOutput>,
}

#[derive(Debug)]
struct PendingReceiver {
    params: TExecPlanFragmentParams,
    expected_senders: usize,
}

#[derive(Debug, Default)]
struct ExchangeState {
    receivers: HashMap<ExchangeKey, PendingReceiver>,
    outputs: HashMap<ExchangeKey, HashMap<i32, ExchangeOutput>>,
}

/// Matches receiver-first StarRocks dispatch with later same-node sender results.
#[derive(Debug, Default)]
pub(crate) struct LocalExchange {
    inner: Mutex<ExchangeState>,
}

impl LocalExchange {
    /// Registers a receiver, returning it immediately when all senders arrived first.
    pub(crate) fn register_receiver(
        &self,
        key: ExchangeKey,
        expected_senders: usize,
        params: TExecPlanFragmentParams,
    ) -> Result<Option<ReadyExchange>, String> {
        if expected_senders == 0 {
            return Err(format!("exchange node {} expects no senders", key.node_id));
        }
        let mut state = self.lock();
        if state.receivers.contains_key(&key) {
            return Err(format!("duplicate receiver registration for {key:?}"));
        }
        state.receivers.insert(
            key,
            PendingReceiver {
                params,
                expected_senders,
            },
        );
        Self::take_ready(&mut state, key)
    }

    /// Buffers one sender output, returning the receiver when this completes its sender set.
    pub(crate) fn push_sender(
        &self,
        key: ExchangeKey,
        sender_id: i32,
        output: ExchangeOutput,
    ) -> Result<Option<ReadyExchange>, String> {
        let mut state = self.lock();
        let senders = state.outputs.entry(key).or_default();
        if senders.contains_key(&sender_id) {
            return Err(format!("duplicate sender {sender_id} for exchange {key:?}"));
        }
        senders.insert(sender_id, output);
        Self::take_ready(&mut state, key)
    }

    fn take_ready(
        state: &mut ExchangeState,
        key: ExchangeKey,
    ) -> Result<Option<ReadyExchange>, String> {
        let Some(receiver) = state.receivers.get(&key) else {
            return Ok(None);
        };
        let actual = state.outputs.get(&key).map(HashMap::len).unwrap_or(0);
        if actual > receiver.expected_senders {
            return Err(format!(
                "exchange {key:?} received {actual} senders but expected {}",
                receiver.expected_senders
            ));
        }
        if actual != receiver.expected_senders {
            return Ok(None);
        }
        let receiver = state
            .receivers
            .remove(&key)
            .expect("receiver checked above");
        let mut senders = state.outputs.remove(&key).unwrap_or_default();
        let mut sender_ids = senders.keys().copied().collect::<Vec<_>>();
        sender_ids.sort_unstable();
        let outputs = sender_ids
            .into_iter()
            .map(|sender_id| senders.remove(&sender_id).expect("sender id came from map"))
            .collect();
        Ok(Some(ReadyExchange {
            params: receiver.params,
            key,
            outputs,
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
