//! GPU-backed fragment executor: owns the Sirius engine on a dedicated thread.
//!
//! [`sirius::SiriusContext`] is `!Send`/`!Sync` and the engine serializes queries through a single
//! process-global context, so the context is created, used, and dropped on one dedicated thread.
//! [`SiriusEngine`] talks to that thread over channels — which are `Send`/`Sync` and carry only
//! owned data (`Vec<u8>` in, `FragmentResult` out) — so it satisfies `dyn FragmentExecutor:
//! Send + Sync` without ever moving the context across threads.
//!
//! The seam is synchronous (see [`FragmentExecutor`]): `execute()` blocks the caller until the
//! engine thread returns the result. `exec_plan_fragment` runs it on a `spawn_blocking` worker, so
//! the BRPC current-thread runtime stays free to serve `fetch_data`, connection cleanup, and
//! shutdown cancellation while a query runs. Each fragment result is fully materialized, and the
//! single process-global context serializes fragment execution — both lifted by the streaming
//! evolution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

#[cfg(test)]
use arrow_array::RecordBatch;
use sirius::SiriusContext;
use starrocks_plan_translator::StreamInputSchema;
use tracing::{info, warn};

use crate::engine_settings::EngineSettings;
use crate::fragment_executor::{FragmentExecutor, FragmentResult, FragmentRun, SenderSlot};

/// Output stream id an intermediate fragment sinks into. One per fragment: a gather exchange has
/// a single destination, and fan-out needs the partitioned sink (#838), not more ids here.
const SENDER_OUTPUT_STREAM: u64 = 0;

/// One fragment execution handed to the engine thread.
///
/// Owned data only: the `sirius::Fragment`s themselves never leave that thread, so what crosses
/// the channel is the plan, the schema of each declared stream, and the slots naming which parked
/// sender outputs to relay in.
struct ExecuteRequest {
    /// Serialized Substrait plan bytes.
    plan: Vec<u8>,
    /// Schema of every exchange this plan reads as a stream.
    stream_inputs: Vec<StreamInputSchema>,
    /// Parked sender outputs to relay in, keyed by receiver exchange node id.
    inputs: Vec<(i32, Vec<SenderSlot>)>,
    /// Set for a sender fragment: park the output under this slot instead of returning rows.
    output: Option<SenderSlot>,
    /// Channel the engine thread sends the result (or a flattened error) back on.
    respond: Sender<Result<Option<FragmentResult>, String>>,
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
    /// failure surfaces here, before any RPC is served. `settings` carries the resolved Sirius
    /// YAML path (built-in defaults when `None`), the engine artifact directory, and the CUDA
    /// device pin.
    pub fn start(settings: EngineSettings) -> Result<Self, String> {
        Self::configure_duckdb_extensions()?;
        Self::configure_engine_environment(&settings)?;
        let (request_tx, request_rx) = channel::<ExecuteRequest>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        let thread = std::thread::Builder::new()
            .name("sirius-engine".to_string())
            .spawn(move || engine_thread(settings.config, request_rx, ready_tx))
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

    /// Points the embedded DuckDB context at the locally built extensions used by Substrait.
    fn configure_duckdb_extensions() -> Result<(), String> {
        let build_dir = std::env::var_os("SIRIUS_BUILD_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("build/release")
            });
        let extensions = [
            (
                "SIRIUS_DUCKDB_PARQUET_EXTENSION",
                build_dir.join("extension/parquet/parquet.duckdb_extension"),
            ),
            (
                "SIRIUS_DUCKDB_CORE_FUNCTIONS_EXTENSION",
                build_dir.join("extension/core_functions/core_functions.duckdb_extension"),
            ),
        ];
        for (variable, path) in extensions {
            if std::env::var_os(variable).is_some() {
                continue;
            }
            if !path.is_file() {
                return Err(format!(
                    "DuckDB extension for {variable} is missing at {}",
                    path.display()
                ));
            }
            // SAFETY: this runs before the engine thread and RPC servers start, so no concurrent
            // code reads or writes these process environment variables.
            unsafe { std::env::set_var(variable, &path) };
        }
        Ok(())
    }

    /// Points the engine's log directory at `<engine_dir>/log` and pins the CUDA device when
    /// requested. Environment already set by the operator wins (matching the extension-path
    /// precedent above), but an ignored `--gpu-device` is called out rather than dropped silently.
    fn configure_engine_environment(settings: &EngineSettings) -> Result<(), String> {
        if std::env::var_os("SIRIUS_LOG_DIR").is_none() {
            let log_dir = settings.engine_dir.join("log");
            std::fs::create_dir_all(&log_dir).map_err(|err| {
                format!(
                    "failed to create engine log directory {}: {err}",
                    log_dir.display()
                )
            })?;
            // SAFETY: this runs before the engine thread and RPC servers start, so no concurrent
            // code reads or writes these process environment variables.
            unsafe { std::env::set_var("SIRIUS_LOG_DIR", &log_dir) };
        }
        if let Some(device) = settings.gpu_device {
            if let Some(existing) = std::env::var_os("CUDA_VISIBLE_DEVICES") {
                warn!(
                    existing = %existing.to_string_lossy(),
                    requested = device,
                    "CUDA_VISIBLE_DEVICES is already set; ignoring --gpu-device"
                );
            } else {
                // SAFETY: same pre-thread-spawn window as above; nothing else touches the
                // environment yet.
                unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", device.to_string()) };
            }
        }
        Ok(())
    }
}

/// Engine-thread body: bring up the context, signal readiness, then serve requests until the
/// request channel closes. The context is dropped here, on this thread, when the loop ends.
fn engine_thread(
    config: Option<PathBuf>,
    requests: Receiver<ExecuteRequest>,
    ready: Sender<Result<(), String>>,
) {
    let context = match build_context(config) {
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

    // Sender fragments whose output is parked on the GPU, waiting for their receiver to be
    // dispatched. Declared after `context` so it is dropped *first*: a fragment borrows the
    // engine it runs on, and the borrow checker enforces the order the C++ side depends on.
    let mut parked: HashMap<SenderSlot, sirius::Fragment<'_>> = HashMap::new();

    info!("sirius-engine thread ready");
    // One query at a time until the handle (and its sender) is dropped.
    while let Ok(request) = requests.recv() {
        let result = run_fragment(&context, &mut parked, &request);
        if result.is_err() {
            // A failed query leaves its parked senders unreachable — the receiver that would have
            // consumed them is the thing that just failed. Dropping them releases the GPU memory
            // their batches hold rather than leaking it for the process's lifetime.
            parked.clear();
        }
        // Ignore a send error: the waiting fragment may have been dropped/cancelled.
        let _ = request.respond.send(result);
    }
    // Fragments must be gone before the context they borrow.
    drop(parked);
    info!("sirius-engine thread shutting down");
}

/// Runs one fragment on the engine thread: declare its input streams, relay every parked sender
/// into them as native batches, execute, then either park the output or return the rows.
fn run_fragment<'ctx>(
    context: &'ctx SiriusContext,
    parked: &mut HashMap<SenderSlot, sirius::Fragment<'ctx>>,
    request: &ExecuteRequest,
) -> Result<Option<FragmentResult>, String> {
    let mut fragment = context
        .fragment()
        .map_err(|err| format!("failed to create fragment: {err}"))?;

    for schema in &request.stream_inputs {
        let stream_id = stream_id_of(schema.node_id)?;
        for column in &schema.columns {
            fragment
                .declare_input_column(stream_id, &column.name, &column.ty)
                .map_err(|err| {
                    format!(
                        "failed to declare column {} of stream {stream_id}: {err}",
                        column.name
                    )
                })?;
        }
        let senders = request
            .inputs
            .iter()
            .find(|(node_id, _)| *node_id == schema.node_id)
            .map(|(_, senders)| senders.as_slice())
            .unwrap_or_default();
        if senders.is_empty() {
            return Err(format!(
                "exchange node {} is read as a stream but no sender output is parked for it",
                schema.node_id
            ));
        }
        for slot in senders {
            let sender_id = u32::try_from(slot.sender_id)
                .map_err(|_| format!("negative sender id {}", slot.sender_id))?;
            fragment
                .declare_input_sender(stream_id, sender_id)
                .map_err(|err| format!("failed to declare sender on stream {stream_id}: {err}"))?;
        }
    }

    // An intermediate fragment sinks into its own output stream 0; a result fragment declares no
    // output stream at all and produces Arrow instead.
    if request.output.is_some() {
        fragment
            .declare_output(SENDER_OUTPUT_STREAM)
            .map_err(|err| format!("failed to declare the fragment output stream: {err}"))?;
    }

    fragment
        .build(&request.plan)
        .map_err(|err| format!("failed to plan fragment: {err}"))?;

    // The fragment boundary itself: each sender's batches move into this fragment's input stream
    // as native handles. Nothing is converted, written, or copied.
    for schema in &request.stream_inputs {
        let stream_id = stream_id_of(schema.node_id)?;
        let senders = request
            .inputs
            .iter()
            .find(|(node_id, _)| *node_id == schema.node_id)
            .map(|(_, senders)| senders.as_slice())
            .unwrap_or_default();
        for slot in senders {
            let mut sender = parked
                .remove(slot)
                .ok_or_else(|| format!("no parked sender output for {slot:?}"))?;
            let sender_id = u32::try_from(slot.sender_id)
                .map_err(|_| format!("negative sender id {}", slot.sender_id))?;
            let moved = fragment
                .relay_from(&mut sender, SENDER_OUTPUT_STREAM, stream_id, sender_id)
                .map_err(|err| format!("failed to relay sender {sender_id}: {err}"))?;
            info!(
                stream_id,
                sender_id,
                batches = moved,
                "relayed native batches across a fragment boundary"
            );
        }
    }

    fragment
        .run()
        .map_err(|err| format!("failed to execute fragment: {err}"))?;

    match request.output {
        Some(slot) => {
            parked.insert(slot, fragment);
            Ok(None)
        }
        // `into_arrow` drains the stream and drops the context-referencing wrapper here, on the
        // engine thread, returning owned batches whose buffers are released via their own Arrow C
        // release callbacks — independent of the context. So they are safe to send to, and drop
        // on, the caller's thread.
        None => fragment
            .into_arrow()
            .map(|result| Some(FragmentResult::new(result.batches)))
            .map_err(|err| err.to_string()),
    }
}

/// Engine-side stream id for a receiver exchange node. The node id addresses the stream, so the
/// two sides of a boundary agree without a separate allocation table.
fn stream_id_of(node_id: i32) -> Result<u64, String> {
    u64::try_from(node_id).map_err(|_| format!("negative exchange node id {node_id}"))
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
    fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        let (respond_tx, respond_rx) = channel();
        let request = ExecuteRequest {
            plan: run.plan.to_substrait_bytes(),
            stream_inputs: run.plan.stream_inputs.clone(),
            inputs: run.inputs,
            output: run.output,
            respond: respond_tx,
        };
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .ok_or_else(|| "sirius-engine is shutting down".to_string())?
            .send(request)
            .map_err(|_| "sirius-engine thread is not running".to_string())?;
        respond_rx
            .recv()
            .map_err(|_| "sirius-engine thread dropped the response".to_string())?
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

    use starrocks_plan_translator::TranslatedPlan;

    use super::*;
    use crate::result_store::FragmentInstanceId;

    /// Default-config engine settings pointing engine artifacts at a scratch directory.
    fn test_settings() -> EngineSettings {
        EngineSettings {
            config: None,
            engine_dir: std::env::temp_dir().join("sirius-engine-test"),
            gpu_device: None,
        }
    }

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
            stream_inputs: Vec::new(),
        }
    }

    /// A plan that reads one input stream through the engine's stream view, plus the declaration
    /// the engine needs for it. `columns` is `(name, is_string)` in output order.
    fn stream_plan(node_id: i32, columns: &[(&str, bool)]) -> TranslatedPlan {
        use starrocks_plan_translator::{StreamInputColumn, StreamInputSchema};
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, rel, r#type,
        };

        let view = sirius::stream_view_name(node_id as u64);
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
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![view.clone()],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        TranslatedPlan {
            plan: Plan {
                relations: vec![PlanRel {
                    rel_type: Some(plan_rel::RelType::Root(RelRoot {
                        input: Some(read),
                        names: names.clone(),
                    })),
                }],
                ..Default::default()
            },
            output_names: names.clone(),
            stream_inputs: vec![StreamInputSchema {
                node_id,
                stream_view: view,
                columns: columns
                    .iter()
                    .map(|(name, is_string)| StreamInputColumn {
                        name: name.to_string(),
                        ty: if *is_string { "VARCHAR" } else { "BIGINT" }.to_string(),
                    })
                    .collect(),
            }],
        }
    }

    /// Runs a result fragment (no output slot) and returns its rows.
    fn run_result(engine: &SiriusEngine, plan: &TranslatedPlan) -> FragmentResult {
        engine
            .run(FragmentRun {
                plan,
                inputs: Vec::new(),
                output: None,
            })
            .expect("execute fragment on GPU")
            .expect("a result fragment returns rows")
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
            stream_inputs: Vec::new(),
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
        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");
        let (respond_tx, respond_rx) = channel();
        engine
            .requests
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .send(ExecuteRequest {
                plan,
                stream_inputs: Vec::new(),
                inputs: Vec::new(),
                output: None,
                respond: respond_tx,
            })
            .unwrap();
        let result = respond_rx
            .recv()
            .expect("engine response")
            .expect("execute")
            .expect("a result fragment returns rows");
        let rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
        eprintln!("plan {path} returned {rows} row(s)");
        for batch in &result.batches {
            eprintln!("{batch:?}");
        }
    }

    /// End-to-end: drive a `local_files` parquet plan through the engine actor and read the rows
    /// back. Exercises the dedicated-thread bring-up, the channel round-trip, and GPU execution.
    /// Requires a GPU and `LD_LIBRARY_PATH` to the built engine (like the `sirius` crate's context
    /// test); the parquet extension path is set from `SIRIUS_BUILD_DIR` (default mirrors sirius-sys).
    #[test]
    fn engine_executes_local_files_and_sequential_exchange() {
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

        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");
        let result = run_result(&engine, &plan);
        let total_rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 3, "expected 3 rows from the parquet fixture");

        // The fragment boundary, over native batches: the sender's rows park on the GPU under
        // `slot`, and the receiver reads them through its input stream. Nothing is written to
        // disk and nothing becomes Arrow in between -- the only Arrow here is the receiver's own
        // result, at the very end.
        const EXCHANGE_NODE: i32 = 7;
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(1, 2),
            node_id: EXCHANGE_NODE,
            sender_id: 0,
        };
        assert!(
            engine
                .run(FragmentRun {
                    plan: &plan,
                    inputs: Vec::new(),
                    output: Some(slot),
                })
                .expect("run the sender fragment")
                .is_none(),
            "a sender fragment parks its output instead of returning rows"
        );

        let receiver = stream_plan(EXCHANGE_NODE, &[("id", false), ("name", true)]);
        let exchanged_result = engine
            .run(FragmentRun {
                plan: &receiver,
                inputs: vec![(EXCHANGE_NODE, vec![slot])],
                output: None,
            })
            .expect("execute exchange receiver on GPU")
            .expect("a result fragment returns rows");
        let exchanged_rows: usize = exchanged_result
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum();
        assert_eq!(
            exchanged_rows, 3,
            "the fragment boundary preserved every row"
        );

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
        let result = run_result(&engine, &pruned);
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
