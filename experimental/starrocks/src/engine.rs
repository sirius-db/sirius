//! GPU-backed fragment executor: owns the Sirius engine on a dedicated thread.
//!
//! [`sirius::SiriusContext`] is `!Send`/`!Sync` and the engine serializes queries through a single
//! process-global context, so the context is created, used, and dropped on one dedicated thread.
//! [`SiriusEngine`] talks to that thread over channels — which are `Send`/`Sync` and carry only
//! owned data — so it satisfies `dyn FragmentExecutor: Send + Sync` without ever moving the
//! context across threads.
//!
//! The seam is synchronous (see [`FragmentExecutor`]): `run_fragment` blocks the caller until the
//! engine thread returns. `exec_plan_fragment` runs it on a `spawn_blocking` worker, so the BRPC
//! current-thread runtime stays free to serve `fetch_data`. A sender parks native GPU batches;
//! a receiver `relay_from`s them or a remote hop `pull_arrow`s them. No NIXL, no staging arena.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use arrow_array::RecordBatch;
use sirius::SiriusContext;
use starrocks_plan_translator::{StreamInputSchema, TranslatedPlan};
use tracing::info;

use crate::fragment_executor::{FragmentExecutor, FragmentResult, FragmentRun, SenderSlot};
use crate::parked_registry::ParkedRegistry;

/// One fragment execution handed to the engine thread.
struct ExecuteRequest {
    /// Serialized Substrait plan bytes.
    plan: Vec<u8>,
    /// Schema of every exchange this plan reads as a stream.
    stream_inputs: Vec<StreamInputSchema>,
    /// Parked sender outputs to relay in, keyed by receiver exchange node id.
    inputs: Vec<(i32, Vec<SenderSlot>)>,
    /// Remote sender batches as `(node id, sender id, batches)`.
    remote_inputs: Vec<(i32, i32, Vec<RecordBatch>)>,
    /// Non-empty for a sender fragment: park once, output stream i belongs to `outputs[i]`.
    outputs: Vec<SenderSlot>,
    /// Every destination receives the full output (broadcast sink).
    broadcast: bool,
    /// Hash-partition key columns for a hash fan-out (empty otherwise).
    hash_keys: Vec<usize>,
    /// Channel the engine thread sends the result (or a flattened error) back on.
    respond: Sender<Result<Option<FragmentResult>, String>>,
}

/// One message to the engine thread — the only caller of `SiriusContext`, which is `!Send`.
enum EngineRequest {
    /// Run one fragment.
    Run(ExecuteRequest),
    /// Pull the next parked Arrow batches under `slot` (`None` when that stream is drained).
    PullArrow {
        slot: SenderSlot,
        respond: Sender<Result<Option<Vec<RecordBatch>>, String>>,
    },
    /// Drop the parked fragment claim under `slot`.
    DropParked {
        slot: SenderSlot,
        respond: Sender<Result<(), String>>,
    },
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
    requests: Mutex<Option<Sender<EngineRequest>>>,
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
        let (request_tx, request_rx) = channel::<EngineRequest>();
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

    fn engine_call<T>(
        &self,
        build: impl FnOnce(Sender<Result<T, String>>) -> EngineRequest,
    ) -> Result<T, String> {
        let (respond_tx, respond_rx) = channel();
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .ok_or_else(|| "sirius-engine is shutting down".to_string())?
            .send(build(respond_tx))
            .map_err(|_| "sirius-engine thread is not running".to_string())?;
        respond_rx
            .recv()
            .map_err(|_| "sirius-engine thread dropped the response".to_string())?
    }
}

/// Engine-thread body: bring up the context, signal readiness, then serve requests until the
/// request channel closes. The context is dropped here, on this thread, when the loop ends.
fn engine_thread(
    config: Option<PathBuf>,
    requests: Receiver<EngineRequest>,
    ready: Sender<Result<(), String>>,
) {
    let context = match build_context(config) {
        Ok(context) => {
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
    // Sender fragments whose output is parked on the GPU, waiting for their receivers.
    // Declared after `context` so the fragments drop first: a fragment borrows the engine.
    let mut registry: ParkedRegistry<sirius::Fragment<'_>> = ParkedRegistry::new();
    info!("sirius-engine thread ready");
    while let Ok(request) = requests.recv() {
        match request {
            EngineRequest::Run(request) => {
                let result = run_fragment(&context, &mut registry, &request);
                let _ = request.respond.send(result);
            }
            EngineRequest::PullArrow { slot, respond } => {
                let _ = respond.send(pull_arrow(&mut registry, &slot));
            }
            EngineRequest::DropParked { slot, respond } => {
                let _ = respond.send(registry.release(&slot).map(|_| ()));
            }
        }
    }
    drop(registry);
    info!("sirius-engine thread shutting down");
}

fn pull_arrow(
    registry: &mut ParkedRegistry<sirius::Fragment<'_>>,
    slot: &SenderSlot,
) -> Result<Option<Vec<RecordBatch>>, String> {
    let (fragment, stream) = registry.claim(slot, "pull")?;
    match fragment
        .pull_arrow(stream)
        .map_err(|err| format!("pull_arrow on stream {stream}: {err}"))?
    {
        Some(part) => Ok(Some(part.batches)),
        None => {
            if fragment
                .drained(stream)
                .map_err(|err| format!("drained on stream {stream}: {err}"))?
            {
                Ok(None)
            } else {
                Err(format!(
                    "output stream {stream} under {slot:?} has nothing parked and is not drained"
                ))
            }
        }
    }
}

/// Runs one fragment on the engine thread: declare streams, hop inputs, execute, then park or
/// return Arrow rows.
fn run_fragment<'ctx>(
    context: &'ctx SiriusContext,
    registry: &mut ParkedRegistry<sirius::Fragment<'ctx>>,
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
        let remote_senders = request
            .remote_inputs
            .iter()
            .filter(|(node_id, _, _)| *node_id == schema.node_id)
            .map(|(_, sender_id, _)| *sender_id)
            .collect::<Vec<_>>();
        if senders.is_empty() && remote_senders.is_empty() {
            return Err(format!(
                "exchange node {} is read as a stream but no sender output — parked or remote — \
                 exists for it",
                schema.node_id
            ));
        }
        for sender_id in senders
            .iter()
            .map(|slot| slot.sender_id)
            .chain(remote_senders)
        {
            let sender_id =
                u32::try_from(sender_id).map_err(|_| format!("negative sender id {sender_id}"))?;
            fragment
                .declare_input_sender(stream_id, sender_id)
                .map_err(|err| format!("failed to declare sender on stream {stream_id}: {err}"))?;
        }
    }

    for stream in 0..request.outputs.len() as u64 {
        fragment
            .declare_output(stream)
            .map_err(|err| format!("failed to declare fragment output stream {stream}: {err}"))?;
    }
    if request.broadcast && request.outputs.len() > 1 {
        fragment
            .declare_output_broadcast()
            .map_err(|err| format!("failed to declare the broadcast output mode: {err}"))?;
    } else if !request.hash_keys.is_empty() && request.outputs.len() > 1 {
        for &key in &request.hash_keys {
            let key = u32::try_from(key).map_err(|_| format!("hash key column {key} overflows"))?;
            fragment
                .declare_output_hash_key(key)
                .map_err(|err| format!("failed to declare hash key column {key}: {err}"))?;
        }
    }

    fragment
        .build(&request.plan)
        .map_err(|err| format!("failed to plan fragment: {err}"))?;

    for schema in &request.stream_inputs {
        let stream_id = stream_id_of(schema.node_id)?;
        let senders = request
            .inputs
            .iter()
            .find(|(node_id, _)| *node_id == schema.node_id)
            .map(|(_, senders)| senders.as_slice())
            .unwrap_or_default();
        for slot in senders {
            let sender_id = u32::try_from(slot.sender_id)
                .map_err(|_| format!("negative sender id {}", slot.sender_id))?;
            let (sender, sender_stream) = registry.claim(slot, "relay")?;
            let moved = fragment
                .relay_from(sender, sender_stream, stream_id, sender_id)
                .map_err(|err| format!("failed to relay sender {sender_id}: {err}"))?;
            registry.release(slot)?;
            info!(
                stream_id,
                sender_id,
                batches = moved,
                "relayed native batches across a fragment boundary"
            );
        }
    }

    for (node_id, sender_id, batches) in &request.remote_inputs {
        let stream_id = stream_id_of(*node_id)?;
        let sender = u32::try_from(*sender_id)
            .map_err(|_| format!("negative remote sender id {sender_id}"))?;
        for batch in batches {
            fragment
                .push_arrow(stream_id, batch)
                .map_err(|err| format!("failed to push Arrow from sender {sender_id}: {err}"))?;
        }
        fragment.close_input(stream_id, sender).map_err(|err| {
            format!("failed to close remote sender {sender_id} on stream {stream_id}: {err}")
        })?;
        info!(
            stream_id,
            sender_id,
            batches = batches.len(),
            "received remote Arrow batches"
        );
    }

    fragment
        .run()
        .map_err(|err| format!("failed to execute fragment: {err}"))?;

    if !request.outputs.is_empty() {
        registry.park(&request.outputs, fragment)?;
        return Ok(None);
    }
    fragment
        .result_to_arrow()
        .map(|result| Some(FragmentResult::new(result.batches)))
        .map_err(|err| err.to_string())
}

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
    fn execute(&self, translated: &TranslatedPlan) -> Result<FragmentResult, String> {
        self.run_fragment(FragmentRun {
            plan: translated,
            inputs: Vec::new(),
            remote_inputs: Vec::new(),
            outputs: Vec::new(),
            broadcast: false,
            hash_keys: Vec::new(),
        })?
        .ok_or_else(|| "result fragment returned no rows".to_string())
    }

    fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        self.engine_call(|respond| {
            EngineRequest::Run(ExecuteRequest {
                plan: run.plan.to_substrait_bytes(),
                stream_inputs: run.plan.stream_inputs.clone(),
                inputs: run.inputs,
                remote_inputs: run.remote_inputs,
                outputs: run.outputs,
                broadcast: run.broadcast,
                hash_keys: run.hash_keys,
                respond,
            })
        })
    }

    fn pull_arrow(&self, slot: &SenderSlot) -> Result<Option<Vec<RecordBatch>>, String> {
        self.engine_call(|respond| EngineRequest::PullArrow {
            slot: *slot,
            respond,
        })
    }

    fn drop_parked(&self, slot: &SenderSlot) -> Result<(), String> {
        self.engine_call(|respond| EngineRequest::DropParked {
            slot: *slot,
            respond,
        })
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
            output_partition_columns: None,
            stream_inputs: Vec::new(),
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
            output_partition_columns: None,
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
        let engine = SiriusEngine::start(None).expect("bring up sirius engine");
        let result = engine
            .engine_call(|respond| {
                EngineRequest::Run(ExecuteRequest {
                    plan,
                    stream_inputs: Vec::new(),
                    inputs: Vec::new(),
                    remote_inputs: Vec::new(),
                    outputs: Vec::new(),
                    broadcast: false,
                    hash_keys: Vec::new(),
                    respond,
                })
            })
            .expect("engine response")
            .expect("execute");
        let rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
        eprintln!("plan {path} returned {rows} row(s)");
        for batch in &result.batches {
            eprintln!("{batch:?}");
        }
    }

    /// End-to-end: drive a `local_files` parquet plan through the engine actor and read the rows
    /// back. Exercises the dedicated-thread bring-up, the channel round-trip, and GPU execution.
    /// Requires a GPU and `LD_LIBRARY_PATH` to the built engine, like the `sirius` crate's context
    /// test.
    #[test]
    fn engine_executes_local_files_plan() {
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
