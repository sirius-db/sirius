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
//!
//! Staging-arena calls deliberately BYPASS the request channel: bring-up hands back a
//! `Send + Sync` [`sirius::StagingArena`] handle and every `staging_*` call is served from it on
//! the caller's own thread. Funneling leases through the engine thread turns any engine stall
//! into a peer's exchange stall — a fragment wedged inside `run()` starved the peer CN's
//! `request_staging_lease` for the PRPC timeout and failed the whole query (the q02 wedge's
//! second act) — so leases must never wait behind engine work.

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
use crate::fragment_executor::{
    FragmentExecutor, FragmentResult, FragmentRun, SenderSlot, StagedBatch,
};

/// A sender fragment's parked output, shared by its destinations: stream i belongs to
/// destination i. `outstanding` counts destinations that have not yet released their stream
/// (drained + dropped); the fragment -- and its GPU batches -- drop when it reaches zero.
struct ParkedOutput<'ctx> {
    fragment: sirius::Fragment<'ctx>,
    outstanding: usize,
}

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
    /// Remote sender batches staged in this CN's arena, as `(node id, sender id, batches)`:
    /// pushed via `push_packed` + `close_input` before `run()`, each lease released the moment
    /// its push returns (copy-out-on-arrival makes that safe).
    remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
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
/// Every variant carries its own respond channel, so callers block for exactly their answer.
///
/// Staging lease/release/info are deliberately NOT variants here: they are served from the
/// thread-safe arena handle on the caller's thread (see the module doc), because a request
/// queued here waits for whatever fragment the engine thread is running.
enum EngineRequest {
    /// Run one fragment (the original request shape).
    Run(ExecuteRequest),
    /// Test-only: occupy the engine thread for the duration — a stand-in for a long (or
    /// wedged) fragment run, so a test can prove staging leases do not queue behind it.
    #[cfg(test)]
    Sleep(std::time::Duration),
    /// Pack the next batch parked under `slot` into a fresh staging lease (`None` when drained).
    ExportNext {
        slot: SenderSlot,
        respond: Sender<Result<Option<StagedBatch>, String>>,
    },
    /// Drop the parked fragment under `slot` (after its output was transmitted, or on a failed
    /// transmit so the GPU memory is not pinned by a dead query).
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
    /// Thread-safe staging-arena handle (`None` when `SIRIUS_EXCHANGE_STAGING_BYTES` is unset),
    /// serving every `staging_*` call directly on the caller's thread.
    ///
    /// INVARIANT: staging leases must never funnel through `requests` — a fragment wedged
    /// inside `run()` would starve every peer's `request_staging_lease` and stall their
    /// cross-CN exchanges. Off-thread service is safe because lease/release/base/capacity only
    /// take the arena's internal mutex and make no CUDA calls; the engine thread keeps its own
    /// direct arena access (`Context::staging_release` for remote-input leases, and
    /// `export_packed` leasing internally) — one shared C++ allocator, two entry points.
    staging: Option<sirius::StagingArena>,
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
        let (request_tx, request_rx) = channel::<EngineRequest>();
        // Readiness carries the staging-arena handle out of the engine thread: the context
        // itself never leaves that thread, but the handle is `Send + Sync` by design so
        // staging calls can bypass the request channel (see the module doc).
        let (ready_tx, ready_rx) = channel::<Result<Option<sirius::StagingArena>, String>>();
        let thread = std::thread::Builder::new()
            .name("sirius-engine".to_string())
            .spawn(move || engine_thread(settings.config, request_rx, ready_tx))
            .map_err(|err| format!("failed to spawn sirius-engine thread: {err}"))?;
        match ready_rx.recv() {
            Ok(Ok(staging)) => Ok(Self {
                requests: Mutex::new(Some(request_tx)),
                thread: Mutex::new(Some(thread)),
                staging,
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
    requests: Receiver<EngineRequest>,
    ready: Sender<Result<Option<sirius::StagingArena>, String>>,
) {
    let context = match build_context(config) {
        Ok(context) => {
            // A send error means the caller is already gone; nothing to serve. The staging
            // handle crosses to the caller so leases are served off this thread.
            if ready.send(Ok(context.staging_arena())).is_err() {
                return;
            }
            context
        }
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    // Sender fragments whose output is parked on the GPU, waiting for their receivers to be
    // dispatched. Parked ONCE per fragment; `parked_slots` maps each destination's SenderSlot to
    // (park id, its output stream). Declared after `context` so the fragments drop *first*: a
    // fragment borrows the engine it runs on, and the borrow checker enforces the order the C++
    // side depends on.
    let mut parked: HashMap<u64, ParkedOutput<'_>> = HashMap::new();
    let mut parked_slots: HashMap<SenderSlot, (u64, u64)> = HashMap::new();
    let mut next_park_id: u64 = 0;
    // Why a slot's parked output went away, for the slots the blanket wipe below destroys.
    // Without this the wipe is silent and the NEXT export of an unrelated slot reports
    // "no parked sender output", which is collateral -- it masks the error that actually killed
    // the query. MEASURED: TPC-H q08 reported that error for weeks while the real failure was an
    // OOM in HASH_JOIN. Bounded: an entry is dropped as soon as the slot is parked again.
    let mut poisoned: HashMap<SenderSlot, String> = HashMap::new();

    info!("sirius-engine thread ready");
    // One request at a time until the handle (and its sender) is dropped. Every respond-send
    // error is ignored: the waiting caller may have been dropped/cancelled.
    while let Ok(request) = requests.recv() {
        match request {
            EngineRequest::Run(request) => {
                let result = run_fragment(
                    &context,
                    &mut parked,
                    &mut parked_slots,
                    &poisoned,
                    &mut next_park_id,
                    &request,
                );
                if let Err(err) = &result {
                    // A failed query leaves its parked senders unreachable — the receiver that
                    // would have consumed them is the thing that just failed. Dropping them
                    // releases the GPU memory their batches hold rather than leaking it for the
                    // process's lifetime.
                    //
                    // The wipe is process-wide, so it also destroys OTHER in-flight fragments'
                    // parked output. Record why, and say so out loud: a drain that fails right
                    // after this used to report only "no parked sender output", which names the
                    // victim and hides the culprit.
                    if !parked_slots.is_empty() {
                        warn!(
                            slots = parked_slots.len(),
                            error = %err,
                            "discarding every parked sender output on this CN after a fragment failure"
                        );
                    }
                    // Replace, never accumulate: only the most recent wipe can explain a slot
                    // that is missing now, and this bounds the map by the live slot count.
                    poisoned.clear();
                    for slot in parked_slots.keys() {
                        poisoned.insert(slot.clone(), err.clone());
                    }
                    parked.clear();
                    parked_slots.clear();
                }
                let _ = request.respond.send(result);
            }
            #[cfg(test)]
            EngineRequest::Sleep(duration) => std::thread::sleep(duration),
            EngineRequest::ExportNext { slot, respond } => {
                let result = export_next(&mut parked, &parked_slots, &poisoned, slot);
                let _ = respond.send(result);
            }
            EngineRequest::DropParked { slot, respond } => {
                let result =
                    release_slot(&mut parked, &mut parked_slots, &poisoned, &slot);
                let _ = respond.send(result);
            }
        }
    }
    // Fragments must be gone before the context they borrow.
    drop(parked_slots);
    drop(parked);
    info!("sirius-engine thread shutting down");
}

/// The error for a slot whose parked output is gone. When the blanket wipe took it, name the
/// failure that triggered the wipe — that is the query's REAL error, and reporting only the
/// generic message is what made TPC-H q08 look like an exchange bug for weeks when it was an
/// OOM in HASH_JOIN.
fn missing_slot(poisoned: &HashMap<SenderSlot, String>, slot: &SenderSlot, verb: &str) -> String {
    match poisoned.get(slot) {
        Some(cause) => format!(
            "sender output for {slot:?} was discarded when another fragment on this CN failed: \
             {cause}"
        ),
        None => format!("no parked sender output to {verb} for {slot:?}"),
    }
}

/// Packs the next batch parked under `slot` into a fresh staging lease.
fn export_next(
    parked: &mut HashMap<u64, ParkedOutput<'_>>,
    parked_slots: &HashMap<SenderSlot, (u64, u64)>,
    poisoned: &HashMap<SenderSlot, String>,
    slot: SenderSlot,
) -> Result<Option<StagedBatch>, String> {
    let (park_id, stream) = parked_slots
        .get(&slot)
        .copied()
        .ok_or_else(|| missing_slot(poisoned, &slot, "export"))?;
    let entry = parked
        .get_mut(&park_id)
        .ok_or_else(|| format!("parked fragment vanished under {slot:?}"))?;
    let batch = entry
        .fragment
        .export_packed(stream)
        .map_err(|err| format!("failed to export a packed batch for {slot:?}: {err}"))?;
    Ok(batch.map(|batch| StagedBatch {
        metadata: batch.metadata,
        offset: batch.offset,
        len: batch.len,
    }))
}

/// Releases one destination's claim on a parked fragment; the fragment (and the GPU memory its
/// remaining batches hold) drops when the LAST destination releases. Exactly-once per slot: a
/// second release of the same slot is a loud error, never a silent double-drop.
fn release_slot(
    parked: &mut HashMap<u64, ParkedOutput<'_>>,
    parked_slots: &mut HashMap<SenderSlot, (u64, u64)>,
    poisoned: &HashMap<SenderSlot, String>,
    slot: &SenderSlot,
) -> Result<(), String> {
    let (park_id, _) = parked_slots
        .remove(slot)
        .ok_or_else(|| missing_slot(poisoned, slot, "drop"))?;
    let entry = parked
        .get_mut(&park_id)
        .ok_or_else(|| format!("parked fragment vanished under {slot:?}"))?;
    entry.outstanding -= 1;
    if entry.outstanding == 0 {
        parked.remove(&park_id);
    }
    Ok(())
}

/// Runs one fragment on the engine thread, guaranteeing that the staging leases its remote
/// inputs sit in are released exactly once — immediately after each successful push, or in a
/// sweep when the run fails partway (a leaked lease would pin the arena for later queries).
fn run_fragment<'ctx>(
    context: &'ctx SiriusContext,
    parked: &mut HashMap<u64, ParkedOutput<'ctx>>,
    parked_slots: &mut HashMap<SenderSlot, (u64, u64)>,
    poisoned: &HashMap<SenderSlot, String>,
    next_park_id: &mut u64,
    request: &ExecuteRequest,
) -> Result<Option<FragmentResult>, String> {
    let mut released = std::collections::HashSet::new();
    let result = run_fragment_inner(
        context,
        parked,
        parked_slots,
        poisoned,
        next_park_id,
        request,
        &mut released,
    );
    if result.is_err() {
        for (_, _, batches) in &request.remote_inputs {
            for batch in batches {
                // `len == 0` batches never held a lease (metadata-only), and offsets in
                // `released` already went back in the push loop.
                if batch.len == 0 || released.contains(&batch.offset) {
                    continue;
                }
                if let Err(err) = context.staging_release(batch.offset) {
                    warn!(
                        offset = batch.offset,
                        error = %err,
                        "failed to release a remote-input staging lease after a fragment error"
                    );
                }
            }
        }
    }
    result
}

/// Runs one fragment on the engine thread: declare its input streams, relay every parked sender
/// and push every staged remote batch into them, execute, then either park the output or return
/// the rows. Records each released remote-input lease offset in `released`.
fn run_fragment_inner<'ctx>(
    context: &'ctx SiriusContext,
    parked: &mut HashMap<u64, ParkedOutput<'ctx>>,
    parked_slots: &mut HashMap<SenderSlot, (u64, u64)>,
    poisoned: &HashMap<SenderSlot, String>,
    next_park_id: &mut u64,
    request: &ExecuteRequest,
    released: &mut std::collections::HashSet<u64>,
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

    // An intermediate fragment sinks into one output stream per destination (stream i belongs
    // to destination outputs[i]); a result fragment declares none and produces Arrow instead.
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
            let (park_id, sender_stream) = parked_slots
                .get(slot)
                .copied()
                .ok_or_else(|| format!("no parked sender output for {slot:?}"))?;
            let sender = parked
                .get_mut(&park_id)
                .ok_or_else(|| format!("parked fragment vanished under {slot:?}"))?;
            let sender_id = u32::try_from(slot.sender_id)
                .map_err(|_| format!("negative sender id {}", slot.sender_id))?;
            let moved = fragment
                .relay_from(&mut sender.fragment, sender_stream, stream_id, sender_id)
                .map_err(|err| format!("failed to relay sender {sender_id}: {err}"))?;
            // This destination's stream is drained; release its claim (the fragment drops with
            // the last claim, freeing the GPU batches).
            release_slot(parked, parked_slots, poisoned, slot)?;
            info!(
                stream_id,
                sender_id,
                batches = moved,
                "relayed native batches across a fragment boundary"
            );
        }
    }

    // Remote senders: their packed batches already sit in this CN's staging arena. Push each
    // (deep copy into pool memory), release its lease immediately — copy-out-on-arrival makes
    // that safe — then close the sender.
    for (node_id, sender_id, batches) in &request.remote_inputs {
        let stream_id = stream_id_of(*node_id)?;
        let sender = u32::try_from(*sender_id)
            .map_err(|_| format!("negative remote sender id {sender_id}"))?;
        for batch in batches {
            let staged = sirius::PackedBatch {
                metadata: batch.metadata.clone(),
                offset: batch.offset,
                len: batch.len,
            };
            fragment.push_packed(stream_id, &staged).map_err(|err| {
                format!(
                    "failed to push a staged remote batch from sender {sender_id} into stream \
                     {stream_id}: {err}"
                )
            })?;
            if batch.len > 0 {
                context.staging_release(batch.offset).map_err(|err| {
                    format!(
                        "failed to release the staging lease at offset {} after pushing it: {err}",
                        batch.offset
                    )
                })?;
                released.insert(batch.offset);
            }
        }
        fragment.close_input(stream_id, sender).map_err(|err| {
            format!("failed to close remote sender {sender_id} on stream {stream_id}: {err}")
        })?;
        info!(
            stream_id,
            sender_id,
            batches = batches.len(),
            "received remote batches"
        );
    }

    fragment
        .run()
        .map_err(|err| format!("failed to execute fragment: {err}"))?;

    if !request.outputs.is_empty() {
        // Park ONCE; each destination claims (park id, its stream). A duplicate slot would let
        // two claims race over one stream -- refuse before inserting anything.
        let park_id = *next_park_id;
        *next_park_id += 1;
        for (stream, slot) in request.outputs.iter().enumerate() {
            if parked_slots.contains_key(slot) {
                return Err(format!(
                    "duplicate destination slot {slot:?} in one sender fan-out"
                ));
            }
            parked_slots.insert(slot.clone(), (park_id, stream as u64));
        }
        parked.insert(
            park_id,
            ParkedOutput {
                fragment,
                outstanding: request.outputs.len(),
            },
        );
        return Ok(None);
    }
    {
        // `into_arrow` drains the stream and drops the context-referencing wrapper here, on the
        // engine thread, returning owned batches whose buffers are released via their own Arrow C
        // release callbacks — independent of the context. So they are safe to send to, and drop
        // on, the caller's thread.
        fragment
            .into_arrow()
            .map(|result| Some(FragmentResult::new(result.batches)))
            .map_err(|err| err.to_string())
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

impl SiriusEngine {
    /// Sends one request to the engine thread and blocks for its answer.
    fn engine_call<T>(
        &self,
        make_request: impl FnOnce(Sender<Result<T, String>>) -> EngineRequest,
    ) -> Result<T, String> {
        let (respond_tx, respond_rx) = channel();
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .ok_or_else(|| "sirius-engine is shutting down".to_string())?
            .send(make_request(respond_tx))
            .map_err(|_| "sirius-engine thread is not running".to_string())?;
        respond_rx
            .recv()
            .map_err(|_| "sirius-engine thread dropped the response".to_string())?
    }

    /// The staging-arena handle, or the loud not-configured error (the exact message the C++
    /// arena raises, so operators see one spelling either way).
    fn staging_arena(&self) -> Result<&sirius::StagingArena, String> {
        self.staging.as_ref().ok_or_else(|| {
            "exchange staging arena not configured (set SIRIUS_EXCHANGE_STAGING_BYTES)".to_string()
        })
    }
}

impl FragmentExecutor for SiriusEngine {
    fn run(&self, run: FragmentRun<'_>) -> Result<Option<FragmentResult>, String> {
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

    // The three staging calls below run on the CALLER's thread, never the engine thread: a
    // peer's lease request must succeed even while the engine is deep inside a fragment run.

    fn staging_info(&self) -> Result<(u64, u64), String> {
        let arena = self.staging_arena()?;
        Ok((arena.base() as u64, arena.capacity()))
    }

    fn staging_lease(&self, len: u64) -> Result<u64, String> {
        self.staging_arena()?
            .lease(len)
            .map_err(|err| format!("staging lease of {len} bytes failed: {err}"))
    }

    fn staging_release(&self, offset: u64) -> Result<(), String> {
        self.staging_arena()?
            .release(offset)
            .map_err(|err| format!("staging release of offset {offset} failed: {err}"))
    }

    fn export_packed_next(&self, slot: SenderSlot) -> Result<Option<StagedBatch>, String> {
        self.engine_call(|respond| EngineRequest::ExportNext { slot, respond })
    }

    fn drop_parked(&self, slot: SenderSlot) -> Result<(), String> {
        self.engine_call(|respond| EngineRequest::DropParked { slot, respond })
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
            output_partition_columns: None,
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
            output_partition_columns: None,
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
                remote_inputs: Vec::new(),
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
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
            output_partition_columns: None,
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
            .send(EngineRequest::Run(ExecuteRequest {
                plan,
                stream_inputs: Vec::new(),
                inputs: Vec::new(),
                remote_inputs: Vec::new(),
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
                respond: respond_tx,
            }))
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
    /// Points the embedded DuckDB at the locally-built parquet extension so it can bind
    /// `parquet_scan`. Call under [`crate::GPU_ENGINE_TEST_LOCK`] so no other context bring-up
    /// reads the environment concurrently.
    fn ensure_parquet_extension_env() {
        if std::env::var_os("SIRIUS_DUCKDB_PARQUET_EXTENSION").is_none() {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let build_dir = std::env::var("SIRIUS_BUILD_DIR")
                .unwrap_or_else(|_| format!("{manifest}/../../build/release"));
            let parquet = format!("{build_dir}/extension/parquet/parquet.duckdb_extension");
            // SAFETY: the GPU lock is held, so no other thread touches the environment here.
            unsafe { std::env::set_var("SIRIUS_DUCKDB_PARQUET_EXTENSION", parquet) };
        }
    }

    /// Writes the tiny `(id BIGINT, name VARCHAR)` parquet fixture at `path`:
    /// rows (1, "a"), (2, "b"), (3, "c").
    fn write_users_parquet(path: &std::path::Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// The engine-actor mirror of the sirius crate's `packed_hop_matches_relay_hop`: a sender
    /// parks its output, `export_packed_next` drains it into staging leases, and a receiver run
    /// with `remote_inputs` delivers exactly the fixture rows — proving the staging info/lease/
    /// release handle path, the ExportNext/DropParked plumbing, and the push-then-release
    /// contract, GPU-side, without any network. Requires a GPU and
    /// `SIRIUS_EXCHANGE_STAGING_BYTES` support.
    #[test]
    fn engine_pushes_staged_remote_batches() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        ensure_parquet_extension_env();
        // The arena is constructed at context bring-up, only when this is set.
        // SAFETY: the GPU lock is held, so no other thread touches the environment here.
        unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB") };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");
        let (base, capacity) = engine.staging_info().expect("staging arena info");
        assert_ne!(base, 0);
        assert_eq!(capacity, 64 << 20);

        const EXCHANGE_NODE: i32 = 9;
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(3, 4),
            node_id: EXCHANGE_NODE,
            sender_id: 0,
        };
        engine
            .run(FragmentRun {
                plan: &plan,
                inputs: Vec::new(),
                remote_inputs: Vec::new(),
                outputs: vec![slot],
                broadcast: false,
                hash_keys: Vec::new(),
            })
            .expect("run the sender fragment");

        let mut staged = Vec::new();
        while let Some(batch) = engine.export_packed_next(slot).expect("export packed") {
            assert!(!batch.metadata.is_empty());
            staged.push(batch);
        }
        assert!(!staged.is_empty(), "the sender parked batches to export");
        // The packed bytes live in arena leases, independent of the parked fragment.
        engine.drop_parked(slot).expect("drop the drained sender");
        assert!(
            engine.drop_parked(slot).is_err(),
            "double drop must be a loud error"
        );

        let receiver = stream_plan(EXCHANGE_NODE, &[("id", false), ("name", true)]);
        let result = engine
            .run(FragmentRun {
                plan: &receiver,
                inputs: Vec::new(),
                remote_inputs: vec![(EXCHANGE_NODE, 0, staged)],
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
            })
            .expect("execute the remote-fed receiver on GPU")
            .expect("a result fragment returns rows");
        let rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, 3, "the staged remote hop preserved every row");

        // Every lease went back (push-then-release), so the free list coalesced back to one
        // whole-arena block and the next lease lands at the base.
        let probe = engine.staging_lease(1024).expect("arena drained");
        assert_eq!(probe, 0);
        engine.staging_release(probe).unwrap();

        // Keep the arena out of the other tests' context bring-ups.
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    /// The regression guard for the q02 lease starvation: a peer's `request_staging_lease`
    /// lands while this CN's engine thread is busy running a fragment, and the old
    /// engine-channel funnel made that lease wait for the whole run (forever, for a wedged
    /// one). With the arena handle serving leases on the caller's thread, a lease taken while
    /// the engine thread is occupied must return immediately. Requires a GPU.
    #[test]
    fn staging_lease_does_not_queue_behind_engine_work() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        ensure_parquet_extension_env();
        // The arena is constructed at context bring-up, only when this is set.
        // SAFETY: the GPU lock is held, so no other thread touches the environment here.
        unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB") };

        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");
        // Occupy the engine thread the way a long fragment run would (the raw-request door the
        // dumped-plan harness also uses); anything funneled through the channel now waits.
        let busy_for = std::time::Duration::from_secs(3);
        engine
            .requests
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .send(EngineRequest::Sleep(busy_for))
            .unwrap();

        let started = std::time::Instant::now();
        let offset = engine
            .staging_lease(4096)
            .expect("lease while the engine thread is busy");
        engine
            .staging_release(offset)
            .expect("release while the engine thread is busy");
        let (base, capacity) = engine.staging_info().expect("info while busy");
        assert_ne!(base, 0);
        assert_eq!(capacity, 64 << 20);
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "staging calls queued behind engine work: served in {elapsed:?} while the engine \
             thread was held for {busy_for:?}"
        );

        // Keep the arena out of the other tests' context bring-ups. Dropping `engine` below
        // joins the engine thread, which finishes its sleep first — ordered teardown holds.
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    #[test]
    fn engine_executes_local_files_and_sequential_exchange() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        ensure_parquet_extension_env();

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
                    remote_inputs: Vec::new(),
                    outputs: vec![slot],
                    broadcast: false,
                    hash_keys: Vec::new(),
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
                remote_inputs: Vec::new(),
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
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
