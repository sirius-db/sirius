//! GPU-backed fragment executor: owns the Sirius engine on a dedicated thread.
//!
//! [`sirius::SiriusContext`] is `!Send`/`!Sync` and the engine serializes queries through a single
//! process-global context, so the context is created, used, and dropped on one dedicated thread.
//! [`SiriusEngine`] talks to that thread over channels — which are `Send`/`Sync` and carry only
//! owned data (`Vec<u8>` in, `Vec<RecordBatch>` out) — so it satisfies `dyn FragmentExecutor:
//! Send + Sync` without ever moving the context across threads.
//!
//! The seam is synchronous (see [`FragmentExecutor`]): `execute()` blocks the caller until the
//! engine thread returns the result. `exec_plan_fragment` runs it on a `spawn_blocking` worker, so
//! the BRPC current-thread runtime stays free to serve `fetch_data`, connection cleanup, and
//! shutdown cancellation while a query runs. The single-fragment limitations are elsewhere: the
//! whole result is materialized before dispatch returns, and the single process-global context
//! serializes queries — both lifted by the streaming evolution.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use arrow_array::RecordBatch;
use sirius::SiriusContext;
use starrocks_plan_translator::TranslatedPlan;
use tracing::info;

use crate::fragment_executor::{FragmentExecutor, FragmentResult};

/// One execution request handed to the engine thread.
struct ExecuteRequest {
    /// Serialized Substrait plan bytes.
    plan: Vec<u8>,
    /// Channel the engine thread sends the result (or a flattened error) back on.
    respond: Sender<Result<Vec<RecordBatch>, String>>,
}

/// GPU-backed [`FragmentExecutor`] running plans on an embedded Sirius engine.
///
/// The engine context lives on a dedicated thread; this handle forwards plans to it and waits for
/// the result. Dropping it closes the request channel, which ends the thread and tears the context
/// down (joined for an ordered teardown).
#[derive(Debug)]
pub struct SiriusEngine {
    /// Sender to the engine thread. `Mutex<Option<..>>` makes the `!Sync` sender shareable and
    /// lets `Drop` close the channel before joining; sends are brief (the thread serializes work).
    requests: Mutex<Option<Sender<ExecuteRequest>>>,
    /// Engine thread handle, taken and joined on drop.
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl SiriusEngine {
    /// Brings up the engine on a dedicated thread (fail-fast) and returns a handle.
    ///
    /// Blocks until the context is initialized — or bring-up fails — so a bad config or GPU
    /// failure surfaces here, before any RPC is served. `config` is the optional Sirius YAML path
    /// (built-in defaults when `None`).
    pub fn start(config: Option<PathBuf>) -> Result<Self, String> {
        let (request_tx, request_rx) = channel::<ExecuteRequest>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let thread = std::thread::Builder::new()
            .name("sirius-engine".to_string())
            .spawn(move || engine_thread(config, request_rx, ready_tx))
            .map_err(|err| format!("failed to spawn sirius-engine thread: {err}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                requests: Mutex::new(Some(request_tx)),
                thread: Mutex::new(Some(thread)),
            }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err("sirius-engine thread exited during bring-up".to_string()),
        }
    }
}

/// Engine-thread body: bring up the context, signal readiness, then serve requests until the
/// request channel closes. The context is dropped here, on this thread, when the loop ends.
fn engine_thread(
    config: Option<PathBuf>,
    requests: Receiver<ExecuteRequest>,
    ready: Sender<Result<(), String>>,
) {
    let mut context = match build_context(config) {
        Ok(context) => {
            // A send error means the caller is already gone; nothing to serve.
            if ready.send(Ok(())).is_err() {
                return;
            }
            context
        }
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };
    info!("sirius-engine thread ready");
    // One query at a time until the handle (and its sender) is dropped.
    while let Ok(request) = requests.recv() {
        // `execute_substrait` drains the Arrow stream and drops the context-referencing wrapper
        // here, on the engine thread, returning owned batches whose buffers are released via their
        // own Arrow C release callbacks — independent of the context. So the batches are safe to
        // send to, and drop on, the caller's thread.
        let result = context
            .execute_substrait(&request.plan)
            .map_err(|err| err.to_string());
        // Ignore a send error: the waiting fragment may have been dropped/cancelled.
        let _ = request.respond.send(result);
    }
    info!("sirius-engine thread shutting down");
}

/// Brings up a [`SiriusContext`] from an optional config path (built-in defaults when `None`).
fn build_context(config: Option<PathBuf>) -> Result<SiriusContext, String> {
    let context = match config {
        Some(path) => SiriusContext::from_config_file(&path),
        None => SiriusContext::new(),
    }
    .map_err(|err| format!("failed to bring up Sirius engine: {err}"))?;
    info!("Sirius engine context created");
    Ok(context)
}

impl FragmentExecutor for SiriusEngine {
    fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
        let (respond_tx, respond_rx) = channel();
        let request = ExecuteRequest {
            plan: translated.to_substrait_bytes(),
            respond: respond_tx,
        };
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .ok_or_else(|| "sirius-engine is shutting down".to_string())?
            .send(request)
            .map_err(|_| "sirius-engine thread is not running".to_string())?;
        let batches = respond_rx
            .recv()
            .map_err(|_| "sirius-engine thread dropped the response".to_string())??;
        Ok(FragmentResult::new(batches))
    }
}

impl Drop for SiriusEngine {
    fn drop(&mut self) {
        // Close the request channel so the engine thread's `recv()` returns and it drops the
        // context, then join for an ordered, complete teardown. The sender must drop before the
        // join or `recv()` would block forever.
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(thread) = self
            .thread
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::*;

    /// Builds a single-file `local_files` parquet read plan with `names` as the root output
    /// names — the shape DuckDB's Substrait reader resolves to `parquet_scan(<path>)`.
    fn local_files_plan(path: &str, names: Vec<String>) -> TranslatedPlan {
        use substrait::proto::read_rel::local_files::FileOrFiles;
        use substrait::proto::read_rel::local_files::file_or_files::{
            FileFormat, ParquetReadOptions, PathType,
        };
        use substrait::proto::read_rel::{LocalFiles, ReadType};
        use substrait::proto::{Plan, PlanRel, ReadRel, Rel, RelRoot, plan_rel, rel};

        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                read_type: Some(ReadType::LocalFiles(LocalFiles {
                    items: vec![FileOrFiles {
                        path_type: Some(PathType::UriFile(path.to_string())),
                        file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        let plan = Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names: names.clone(),
                })),
            }],
            ..Default::default()
        };
        TranslatedPlan {
            plan,
            output_names: names,
        }
    }

    /// Like [`local_files_plan`] but declares a `base_schema` (names + types) on the read — the
    /// shape the translator emits for a `FILES()` scan. DuckDB's Substrait reader projects the
    /// parquet columns onto these names, so a pruned/reordered `base_schema` selects columns by
    /// name rather than by file position. `columns` is `(name, is_string)` in output order.
    fn local_files_plan_with_base_schema(path: &str, columns: &[(&str, bool)]) -> TranslatedPlan {
        use substrait::proto::read_rel::local_files::FileOrFiles;
        use substrait::proto::read_rel::local_files::file_or_files::{
            FileFormat, ParquetReadOptions, PathType,
        };
        use substrait::proto::read_rel::{LocalFiles, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, rel, r#type,
        };

        let names: Vec<String> = columns.iter().map(|(name, _)| name.to_string()).collect();
        let types: Vec<Type> = columns
            .iter()
            .map(|(_, is_string)| {
                let kind = if *is_string {
                    r#type::Kind::String(r#type::String {
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Nullable as i32,
                    })
                } else {
                    r#type::Kind::I64(r#type::I64 {
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Nullable as i32,
                    })
                };
                Type { kind: Some(kind) }
            })
            .collect();
        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.clone(),
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::LocalFiles(LocalFiles {
                    items: vec![FileOrFiles {
                        path_type: Some(PathType::UriFile(path.to_string())),
                        file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        let plan = Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names: names.clone(),
                })),
            }],
            ..Default::default()
        };
        TranslatedPlan {
            plan,
            output_names: names,
        }
    }

    /// Replays a Substrait plan dumped via `SIRIUS_CN_DUMP_FRAGMENTS` (path in
    /// `SIRIUS_SUBSTRAIT_PLAN`) against the engine — a debug harness for diagnosing a captured
    /// plan in isolation, outside the FE/CN loop.
    #[test]
    #[ignore = "debug harness: set SIRIUS_SUBSTRAIT_PLAN to a dumped plan and run with a GPU"]
    fn engine_replays_dumped_substrait_plan() {
        let path = std::env::var("SIRIUS_SUBSTRAIT_PLAN").expect("SIRIUS_SUBSTRAIT_PLAN not set");
        let plan = std::fs::read(&path).expect("read dumped substrait plan");
        let engine = SiriusEngine::start(None).expect("bring up sirius engine");
        let (respond_tx, respond_rx) = channel();
        engine
            .requests
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .send(ExecuteRequest {
                plan,
                respond: respond_tx,
            })
            .unwrap();
        let batches = respond_rx
            .recv()
            .expect("engine response")
            .expect("execute");
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        eprintln!("plan {path} returned {rows} row(s)");
        for batch in &batches {
            eprintln!("{batch:?}");
        }
    }

    /// End-to-end: drive a `local_files` parquet plan through the engine actor and read the rows
    /// back. Exercises the dedicated-thread bring-up, the channel round-trip, and GPU execution.
    /// Requires a GPU and `LD_LIBRARY_PATH` to the built engine (like the `sirius` crate's context
    /// test); the parquet extension path is set from `SIRIUS_BUILD_DIR` (default mirrors sirius-sys).
    #[test]
    fn engine_executes_local_files_plan() {
        // Point the embedded DuckDB at the locally-built parquet extension so it can bind
        // `parquet_scan`. This is the only context-constructing test in the crate, so no other
        // thread reads the environment concurrently.
        if std::env::var_os("SIRIUS_DUCKDB_PARQUET_EXTENSION").is_none() {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let build_dir = std::env::var("SIRIUS_BUILD_DIR")
                .unwrap_or_else(|_| format!("{manifest}/../../build/release"));
            let parquet = format!("{build_dir}/extension/parquet/parquet.duckdb_extension");
            // SAFETY: set before the engine thread brings up the context; no other thread reads it.
            unsafe { std::env::set_var("SIRIUS_DUCKDB_PARQUET_EXTENSION", parquet) };
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        let engine = SiriusEngine::start(None).expect("bring up sirius engine");
        let result = engine.execute(&plan).expect("execute fragment on GPU");
        let total_rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 3, "expected 3 rows from the parquet fixture");

        // A `base_schema` that prunes and reorders the file's columns must bind by name, not by
        // file position (exercises the Substrait reader's `local_files` projection). The fixture
        // file is [id, name, extra]; the plan asks for [name, id], so a positional bind would
        // return the wrong columns.
        let cols_path = dir.path().join("cols.parquet");
        let cols_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("extra", DataType::Int64, false),
        ]));
        let cols_batch = RecordBatch::try_new(
            cols_schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![10, 20, 30])) as ArrayRef,
            ],
        )
        .unwrap();
        {
            let file = std::fs::File::create(&cols_path).unwrap();
            let mut writer = ArrowWriter::try_new(file, cols_schema, None).unwrap();
            writer.write(&cols_batch).unwrap();
            writer.close().unwrap();
        }

        let pruned = local_files_plan_with_base_schema(
            cols_path.to_str().unwrap(),
            &[("name", true), ("id", false)],
        );
        let result = engine
            .execute(&pruned)
            .expect("execute pruned fragment on GPU");
        let batch = result
            .batches
            .iter()
            .find(|batch| batch.num_rows() > 0)
            .expect("a non-empty result batch");
        assert_eq!(batch.num_columns(), 2, "base_schema pruned to two columns");
        assert_eq!(batch.schema().field(0).name(), "name");
        assert_eq!(batch.schema().field(1).name(), "id");
        // Bound by name, not position: column 0 carries the strings, column 1 the ids.
        let name_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("first output column is the utf8 name column");
        assert_eq!(name_col.value(0), "a");
        let id_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("second output column is the int64 id column");
        assert_eq!(id_col.value(0), 1);
    }
}
