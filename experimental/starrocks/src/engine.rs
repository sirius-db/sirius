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
//! current-thread runtime stays free to serve `fetch_data`, connection cleanup, and shutdown
//! cancellation while a query runs. A sender fragment's output stays parked on the GPU, owned by
//! the engine thread, until a same-CN receiver relays it in. Every fragment is materialized before
//! dispatch returns, and the single process-global context serializes queries — both lifted by the
//! streaming evolution.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use sirius::SiriusContext;
use starrocks_plan_translator::{StreamInputSchema, TranslatedPlan};
use tracing::{info, warn};

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
    /// Non-empty for a sender fragment: park once, output stream i belongs to `outputs[i]`.
    outputs: Vec<SenderSlot>,
    /// Every destination receives the full output (a broadcast sink).
    broadcast: bool,
    /// Hash-partition key columns for a hash fan-out (empty otherwise).
    hash_keys: Vec<usize>,
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
    info!("sirius-engine thread ready");
    // Declared after `context` so parked fragments, which borrow it, drop first.
    let mut parked = ParkedRegistry::new();
    // One fragment at a time until the handle (and its sender) is dropped.
    while let Ok(request) = requests.recv() {
        let result = run_fragment(&context, &mut parked, &request);
        // Ignore a send error: the waiting fragment may have been dropped/cancelled.
        let _ = request.respond.send(result);
    }
    info!("sirius-engine thread shutting down");
}

/// Runs one fragment on the engine thread: declare its streams, build, relay parked sender output
/// in, run, then park the output for its receivers or return the rows.
fn run_fragment<'ctx>(
    context: &'ctx SiriusContext,
    parked: &mut ParkedRegistry<sirius::Fragment<'ctx>>,
    request: &ExecuteRequest,
) -> Result<Option<FragmentResult>, String> {
    let start_us = unix_us();
    let started = Instant::now();
    let mut fragment = context
        .fragment()
        .map_err(|err| format!("failed to create fragment: {err}"))?;

    let mut sender_rows = Vec::with_capacity(request.stream_inputs.len());
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
        let senders = senders_of(request, schema.node_id);
        for slot in senders {
            fragment
                .declare_input_sender(stream_id, sender_id_of(slot)?)
                .map_err(|err| format!("failed to declare sender on stream {stream_id}: {err}"))?;
        }
        sender_rows.push(
            senders
                .iter()
                .map(|slot| {
                    let (sender, sender_stream) = parked.claim(slot, "count").ok()?;
                    sender.output_row_count(sender_stream).ok()
                })
                .collect(),
        );
    }
    match exact_cardinalities(sender_rows) {
        Some(rows) => {
            for (schema, rows) in request.stream_inputs.iter().zip(rows) {
                let stream_id = stream_id_of(schema.node_id)?;
                fragment
                    .declare_input_cardinality(stream_id, rows)
                    .map_err(|err| {
                        format!("failed to declare cardinality of stream {stream_id}: {err}")
                    })?;
                info!(stream_id, rows, "declared input stream cardinality");
            }
        }
        None => warn!("an input stream row count is unknown; planning without cardinalities"),
    }

    for stream in 0..request.outputs.len() as u64 {
        fragment
            .declare_output(stream)
            .map_err(|err| format!("failed to declare fragment output stream {stream}: {err}"))?;
    }
    if request.broadcast {
        fragment
            .declare_output_broadcast()
            .map_err(|err| format!("failed to declare the broadcast output mode: {err}"))?;
    }
    for &key in &request.hash_keys {
        let key = u32::try_from(key).map_err(|_| format!("hash key column {key} overflows"))?;
        fragment
            .declare_output_hash_key(key)
            .map_err(|err| format!("failed to declare hash key column {key}: {err}"))?;
    }

    fragment
        .build(&request.plan)
        .map_err(|err| format!("failed to plan fragment: {err}"))?;
    let built = started.elapsed();

    for schema in &request.stream_inputs {
        let stream_id = stream_id_of(schema.node_id)?;
        for slot in senders_of(request, schema.node_id) {
            let sender_id = sender_id_of(slot)?;
            let (sender, sender_stream) = parked.claim(slot, "relay")?;
            let moved = fragment
                .relay_from(sender, sender_stream, stream_id, sender_id)
                .map_err(|err| format!("failed to relay sender {sender_id}: {err}"))?;
            parked.release(slot)?;
            info!(
                stream_id,
                sender_id,
                batches = moved,
                "relayed native batches across a fragment boundary"
            );
        }
    }
    let relayed = started.elapsed();

    fragment
        .run()
        .map_err(|err| format!("failed to execute fragment: {err}"))?;
    let ran = started.elapsed();

    let result = if request.outputs.is_empty() {
        // `result_to_arrow` drains the Arrow stream here, on the engine thread, returning owned
        // batches whose buffers are released via their own Arrow C release callbacks —
        // independent of the context. So the batches are safe to send to, and drop on, the
        // caller's thread.
        let result = fragment.result_to_arrow().map_err(|err| err.to_string())?;
        Some(FragmentResult::new(result.batches))
    } else {
        parked.park(&request.outputs, fragment)?;
        None
    };
    if timing_enabled() {
        info!(
            start_us,
            end_us = unix_us(),
            build_us = built.as_micros() as u64,
            inputs_us = (relayed - built).as_micros() as u64,
            run_us = (ran - relayed).as_micros() as u64,
            outputs = request.outputs.len(),
            "fragment timing"
        );
    }
    Ok(result)
}

/// The parked sender outputs feeding exchange `node_id`.
fn senders_of(request: &ExecuteRequest, node_id: i32) -> &[SenderSlot] {
    request
        .inputs
        .iter()
        .find(|(id, _)| *id == node_id)
        .map(|(_, senders)| senders.as_slice())
        .unwrap_or_default()
}

/// Each stream's exact row count summed over its senders, or `None` when any count is unknown.
/// All or none: an undeclared stream plans as cardinality 1, so declaring only some streams would
/// make an unknown one look smallest and become the hash join build side.
fn exact_cardinalities(sender_rows: Vec<Vec<Option<u64>>>) -> Option<Vec<u64>> {
    sender_rows
        .into_iter()
        .map(|rows| {
            rows.into_iter()
                .try_fold(0u64, |total, rows| total.checked_add(rows?))
        })
        .collect()
}

fn stream_id_of(node_id: i32) -> Result<u64, String> {
    u64::try_from(node_id).map_err(|_| format!("negative exchange node id {node_id}"))
}

fn sender_id_of(slot: &SenderSlot) -> Result<u32, String> {
    u32::try_from(slot.sender_id).map_err(|_| format!("negative sender id {}", slot.sender_id))
}

/// `SIRIUS_CN_TIMING` set to anything but `0` logs one `fragment timing` line per fragment.
fn timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SIRIUS_CN_TIMING").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// Wall-clock microseconds since the Unix epoch, so timing lines from several CNs line up.
fn unix_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
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
            outputs: Vec::new(),
            broadcast: false,
            hash_keys: Vec::new(),
        })?
        .ok_or_else(|| "result fragment returned no rows".to_string())
    }

    fn run_fragment(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
        let (respond_tx, respond_rx) = channel();
        let request = ExecuteRequest {
            plan: run.plan.to_substrait_bytes(),
            stream_inputs: run.plan.stream_inputs.clone(),
            inputs: run.inputs,
            outputs: run.outputs,
            broadcast: run.broadcast,
            hash_keys: run.hash_keys,
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

    use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray};
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

    /// Like [`local_files_plan_with_base_schema`] but reads exchange `node_id`'s input stream, as
    /// the translator lowers an `EXCHANGE_NODE`.
    fn stream_read_plan(node_id: i32, columns: &[(&str, bool)]) -> TranslatedPlan {
        use starrocks_plan_translator::StreamInputColumn;
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{plan_rel, rel};

        let mut translated = local_files_plan_with_base_schema("", columns);
        let stream_view = sirius::stream_view_name(node_id as u64);
        let Some(plan_rel::RelType::Root(root)) = &mut translated.plan.relations[0].rel_type else {
            panic!("plan has a root relation");
        };
        let Some(rel::RelType::Read(read)) = root
            .input
            .as_mut()
            .and_then(|input| input.rel_type.as_mut())
        else {
            panic!("root reads a table");
        };
        read.read_type = Some(ReadType::NamedTable(NamedTable {
            names: vec![stream_view.clone()],
            ..Default::default()
        }));
        translated.stream_inputs = vec![StreamInputSchema {
            node_id,
            stream_view,
            columns: columns
                .iter()
                .map(|(name, is_string)| StreamInputColumn {
                    name: name.to_string(),
                    ty: if *is_string { "VARCHAR" } else { "BIGINT" }.to_string(),
                })
                .collect(),
        }];
        translated
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
                stream_inputs: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
                respond: respond_tx,
            })
            .unwrap();
        let result = respond_rx
            .recv()
            .expect("engine response")
            .expect("execute")
            .expect("result fragment returned rows");
        let rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
        eprintln!("plan {path} returned {rows} row(s)");
        for batch in &result.batches {
            eprintln!("{batch:?}");
        }
    }

    #[test]
    fn cardinalities_are_declared_for_every_stream_or_none() {
        assert_eq!(
            exact_cardinalities(vec![vec![Some(2), Some(3)], vec![Some(4)]]),
            Some(vec![5, 4])
        );
        assert_eq!(
            exact_cardinalities(vec![vec![Some(2)], vec![Some(4), None]]),
            None,
            "one unknown sender leaves every stream undeclared"
        );
        assert_eq!(
            exact_cardinalities(vec![vec![Some(u64::MAX), Some(1)]]),
            None
        );
        assert_eq!(exact_cardinalities(Vec::new()), Some(Vec::new()));
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

        // Park-then-relay: a sender parks the scan for one receiver, which reads it back as a
        // stream with a declared cardinality.
        let slot = SenderSlot {
            fragment_instance_id: crate::result_store::FragmentInstanceId::from_halves(1, 2),
            node_id: 7,
            sender_id: 0,
        };
        let parked = engine
            .run_fragment(FragmentRun {
                plan: &plan,
                inputs: Vec::new(),
                outputs: vec![slot],
                broadcast: false,
                hash_keys: Vec::new(),
            })
            .expect("park the sender output");
        assert!(parked.is_none(), "a sender fragment returns no rows");
        let receiver = stream_read_plan(7, &[("id", false), ("name", true)]);
        let relayed = engine
            .run_fragment(FragmentRun {
                plan: &receiver,
                inputs: vec![(7, vec![slot])],
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
            })
            .expect("relay into the receiver")
            .expect("the receiver returns rows");
        let relayed_rows: usize = relayed.batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(relayed_rows, 3);
    }
}
