//! GPU-backed fragment executor: owns the Sirius engine on a dedicated thread.
//!
//! [`sirius::SiriusContext`] is `!Send`/`!Sync` and the engine serializes queries through a single
//! process-global context, so the context is created, used, and dropped on one dedicated thread.
//! [`SiriusEngine`] talks to that thread over channels — which are `Send`/`Sync` and carry only
//! owned data (`Vec<u8>` in, `FragmentResult` out) — so it satisfies `dyn FragmentExecutor:
//! Send + Sync` without ever moving the context across threads.
//!
//! The seam is synchronous (see [`FragmentExecutor`]): `run()` blocks the caller until the
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
//!
//! Parked output is owned per query; a query is retired (dropped, later runs refused) by its own
//! engine error or by the CN's `retire_query`, never by another query's failure.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

#[cfg(test)]
use arrow_array::RecordBatch;
use sirius::SiriusContext;
use starrocks_plan_translator::StreamInputSchema;
use tracing::{debug, info, warn};

use crate::engine_settings::{EngineSettings, resolve_cuda_visible_devices};
use crate::fragment_executor::{
    FragmentExecutor, FragmentLabel, FragmentResult, FragmentRun, SenderSlot, StagedBatch,
};
use crate::parked_registry::{ParkedRegistry, QueryId, Release, RetireTrigger, RetiredQueries};

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
    /// Remote sender batches on this CN, as `(node id, sender id, batches)`: ticketed ones are
    /// taken from the inbound store with `push_inbound`, legacy ones pushed from their arena
    /// lease via `push_packed` (lease released the moment the push returns); then `close_input`,
    /// all before `run()`.
    remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
    /// Non-empty for a sender fragment: park once, output stream i belongs to `outputs[i]`.
    outputs: Vec<SenderSlot>,
    /// Every destination receives the full output (broadcast sink).
    broadcast: bool,
    /// Hash-partition key columns for a hash fan-out (empty otherwise).
    hash_keys: Vec<usize>,
    /// The query and instance this run belongs to; parked output is owned by `label.query_id`.
    label: FragmentLabel,
    /// Channel the engine thread sends the result (or a flattened error) back on.
    respond: Sender<Result<Option<FragmentResult>, String>>,
    /// Test-only: runs between `fragment.run()` and the park, standing in for a cancel or a
    /// sibling's failure that lands while this fragment is inside the engine (the gate-2 case).
    #[cfg(test)]
    after_run: Option<Box<dyn Fn() + Send>>,
}

/// One message to the engine thread — the only caller of `SiriusContext`, which is `!Send`.
/// Every variant that answers carries its own respond channel, so callers block for exactly
/// their answer.
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
    /// Drop every parked output of `query_id` and refuse its later runs. Sent by the CN when it
    /// learns a query is over by a route the engine cannot see: a pre-run failure or an FE
    /// cancel. No respond channel: the sender already marked the shared [`RetiredQueries`], so
    /// the refusal is visible to the very next dequeue; the drop happens when the engine thread
    /// is free. Fire-and-forget because the cancel RPC and the dispatch worker must never wait
    /// behind a running fragment.
    RetireQuery {
        query_id: QueryId,
        trigger: RetireTrigger,
        cause: String,
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
    /// Thread-safe inbound-store handle (`None` without an arena): `stage_inbound` copies an
    /// arriving frame out of its lease into pool memory on the caller's thread, so the lease goes
    /// back while the receiver is still being assembled. Same off-thread shape as `staging`, for
    /// the same reason: a frame must never wait for the engine thread to be free.
    inbound: Option<sirius::InboundStore>,
    /// Queries retired on this CN, shared with the engine thread. The CN's `retire_query` marks
    /// here BEFORE queueing its `RetireQuery`, so a `Run` already sitting in the FIFO ahead of it
    /// is refused when the thread dequeues it (the same off-thread shape as `staging`).
    retired: Arc<Mutex<RetiredQueries>>,
}

impl SiriusEngine {
    /// Brings up the engine on a dedicated thread (fail-fast) and returns a handle.
    ///
    /// Blocks until the context is initialized — or bring-up fails — so a bad config or GPU
    /// failure surfaces here, before any RPC is served. `settings` carries the resolved Sirius
    /// YAML path (built-in defaults when `None`), the engine artifact directory, and the CUDA
    /// device pin.
    pub fn start(settings: EngineSettings) -> Result<Self, String> {
        Self::configure_engine_environment(&settings)?;
        let (request_tx, request_rx) = channel::<EngineRequest>();
        // Readiness carries the staging-arena handle out of the engine thread: the context
        // itself never leaves that thread, but the handle is `Send + Sync` by design so
        // staging calls can bypass the request channel (see the module doc).
        let (ready_tx, ready_rx) = channel::<
            Result<(Option<sirius::StagingArena>, Option<sirius::InboundStore>), String>,
        >();
        let retired: Arc<Mutex<RetiredQueries>> = Arc::default();
        let thread_retired = Arc::clone(&retired);
        let thread = std::thread::Builder::new()
            .name("sirius-engine".to_string())
            .spawn(move || engine_thread(settings.config, request_rx, ready_tx, thread_retired))
            .map_err(|err| format!("failed to spawn sirius-engine thread: {err}"))?;
        match ready_rx.recv() {
            Ok(Ok((staging, inbound))) => Ok(Self {
                requests: Mutex::new(Some(request_tx)),
                thread: Mutex::new(Some(thread)),
                staging,
                inbound,
                retired,
            }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err("sirius-engine thread exited during bring-up".to_string()),
        }
    }

    /// Points the engine's log directory at `<engine_dir>/log` and pins the CUDA device when
    /// requested. An operator-exported `SIRIUS_LOG_DIR` wins. An exported `CUDA_VISIBLE_DEVICES`
    /// must name the same device as `--gpu-device`, or bring-up is refused (see
    /// [`resolve_cuda_visible_devices`]): letting the export win silently is how N CNs launched
    /// on N GPUs ended up priming the same device.
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
        let exported =
            std::env::var_os("CUDA_VISIBLE_DEVICES").map(|v| v.to_string_lossy().into_owned());
        if let Some(device) =
            resolve_cuda_visible_devices(settings.gpu_device, exported.as_deref())?
        {
            // SAFETY: same pre-thread-spawn window as above; nothing else touches the
            // environment yet.
            unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", device) };
        }
        Ok(())
    }
}

/// Engine-thread body: bring up the context, signal readiness, then serve requests until the
/// request channel closes. The context is dropped here, on this thread, when the loop ends.
fn engine_thread(
    config: Option<PathBuf>,
    requests: Receiver<EngineRequest>,
    ready: Sender<Result<(Option<sirius::StagingArena>, Option<sirius::InboundStore>), String>>,
    retired: Arc<Mutex<RetiredQueries>>,
) {
    let context = match build_context(config) {
        Ok(context) => {
            // A send error means the caller is already gone; nothing to serve. The staging and
            // inbound-store handles cross to the caller so leases and arrivals are served off
            // this thread.
            if ready
                .send(Ok((context.staging_arena(), context.inbound_store())))
                .is_err()
            {
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
    // dispatched, owned per query. Declared after `context` so the fragments drop *first*: a
    // fragment borrows the engine it runs on, and the borrow checker enforces the order the C++
    // side depends on.
    let mut registry: ParkedRegistry<sirius::Fragment<'_>> = ParkedRegistry::new();

    info!("sirius-engine thread ready");
    // One request at a time until the handle (and its sender) is dropped. Every respond-send
    // error is ignored: the waiting caller may have been dropped/cancelled.
    while let Ok(request) = requests.recv() {
        match request {
            EngineRequest::Run(request) => {
                let result = run_fragment(&context, &mut registry, &retired, &request);
                if let Err(err) = &result {
                    // The failed query's parked senders are unreachable now — the receiver that
                    // would have consumed them is the thing that just failed — and a run of it
                    // still queued behind this one must not park more. Retire the query; every
                    // other query's parked output stays where it is.
                    retire(
                        &mut registry,
                        &retired,
                        request.label.query_id,
                        &RetireTrigger::EngineErr,
                        err,
                    );
                }
                let _ = request.respond.send(result);
            }
            #[cfg(test)]
            EngineRequest::Sleep(duration) => std::thread::sleep(duration),
            EngineRequest::ExportNext { slot, respond } => {
                let _ = respond.send(export_next(&mut registry, &slot));
            }
            EngineRequest::DropParked { slot, respond } => {
                let _ = respond.send(drop_parked(&mut registry, &slot));
            }
            EngineRequest::RetireQuery {
                query_id,
                trigger,
                cause,
            } => retire(&mut registry, &retired, Some(query_id), &trigger, &cause),
        }
    }
    // Fragments must be gone before the context they borrow.
    drop(registry);
    info!("sirius-engine thread shutting down");
}

/// The retired set, recovering from a poisoned mutex (a panic elsewhere must not wedge the gates).
fn lock(retired: &Mutex<RetiredQueries>) -> std::sync::MutexGuard<'_, RetiredQueries> {
    retired.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The cause `query_id` was retired for, if it was.
fn retired_cause(retired: &Mutex<RetiredQueries>, query_id: QueryId) -> Option<String> {
    lock(retired).cause(query_id).map(str::to_owned)
}

/// Retires one query on the engine thread: drops its parked fragments, poisons its slots with
/// `cause`, marks it so later runs are refused. `None` retires only unlabeled output (test
/// fixtures) and marks nothing.
fn retire<F>(
    registry: &mut ParkedRegistry<F>,
    retired: &Mutex<RetiredQueries>,
    query_id: Option<QueryId>,
    trigger: &RetireTrigger,
    cause: &str,
) {
    let dropped = registry.retire(query_id, cause);
    if let Some(query_id) = query_id {
        lock(retired).mark(query_id, cause);
    }
    if dropped.fragments > 0 {
        warn!(
            query_id = ?query_id,
            trigger = %trigger,
            fragments = dropped.fragments,
            slots = dropped.slots,
            still_parked = registry.fragments(),
            cause,
            "retired a query's parked sender outputs"
        );
    } else {
        debug!(query_id = ?query_id, trigger = %trigger, "retire found nothing parked");
    }
}

/// Packs the next batch parked under `slot` into a fresh staging lease.
fn export_next(
    registry: &mut ParkedRegistry<sirius::Fragment<'_>>,
    slot: &SenderSlot,
) -> Result<Option<StagedBatch>, String> {
    let (fragment, stream) = registry.claim(slot, "export")?;
    let batch = fragment
        .export_packed(stream)
        .map_err(|err| format!("failed to export a packed batch for {slot:?}: {err}"))?;
    Ok(batch.map(|batch| StagedBatch {
        metadata: batch.metadata,
        offset: batch.offset,
        len: batch.len,
        rows: Some(batch.rows),
        ticket: None,
    }))
}

/// Releases one destination's claim on a parked fragment; the fragment (and the GPU memory its
/// remaining batches hold) drops when the LAST destination releases. Exactly-once per live slot:
/// a second release is a loud error, never a silent double-drop. A slot already retired with its
/// query returns `Ok` once, so the transport's post-drain and failure-path drops of a dead
/// query's output are not errors.
fn drop_parked<F>(registry: &mut ParkedRegistry<F>, slot: &SenderSlot) -> Result<(), String> {
    if registry.release(slot)? == Release::AlreadyTornDown {
        debug!(
            ?slot,
            "drop_parked for a slot already retired with its query"
        );
    }
    Ok(())
}

/// Runs one fragment on the engine thread, guaranteeing that the staging leases its remote
/// inputs sit in are released exactly once — immediately after each successful push, or in a
/// sweep when the run fails partway (a leaked lease would pin the arena for later queries).
fn run_fragment<'ctx>(
    context: &'ctx SiriusContext,
    registry: &mut ParkedRegistry<sirius::Fragment<'ctx>>,
    retired: &Mutex<RetiredQueries>,
    request: &ExecuteRequest,
) -> Result<Option<FragmentResult>, String> {
    let mut released = std::collections::HashSet::new();
    let mut taken = std::collections::HashSet::new();
    let result = run_fragment_inner(
        context,
        registry,
        retired,
        request,
        &mut released,
        &mut taken,
    );
    if result.is_err() {
        let store = context.inbound_store();
        for (_, _, batches) in &request.remote_inputs {
            for batch in batches {
                if let Some(ticket) = batch.ticket {
                    // A ticketed batch sits in the inbound store, not in a lease; one the push
                    // loop already took belongs to the (now dropped) fragment.
                    if taken.contains(&ticket) {
                        continue;
                    }
                    match &store {
                        Some(store) => {
                            if let Err(err) = store.drop(ticket) {
                                warn!(
                                    ticket,
                                    error = %err,
                                    "failed to drop a staged remote input after a fragment error"
                                );
                            }
                        }
                        None => warn!(ticket, "ticketed remote input but no inbound store"),
                    }
                    continue;
                }
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

#[cfg(test)]
fn run_after_run_hook(request: &ExecuteRequest) {
    if let Some(hook) = &request.after_run {
        hook();
    }
}

#[cfg(not(test))]
fn run_after_run_hook(_request: &ExecuteRequest) {}

/// Runs one fragment on the engine thread: declare its input streams, relay every parked sender
/// and push every staged remote batch into them, execute, then either park the output or return
/// the rows. Records each released remote-input lease offset in `released`.
fn run_fragment_inner<'ctx>(
    context: &'ctx SiriusContext,
    registry: &mut ParkedRegistry<sirius::Fragment<'ctx>>,
    retired: &Mutex<RetiredQueries>,
    request: &ExecuteRequest,
    released: &mut std::collections::HashSet<u64>,
    taken: &mut std::collections::HashSet<u64>,
) -> Result<Option<FragmentResult>, String> {
    // Gate 1: a run of a query this CN already retired is refused before it touches the engine.
    // Here, not at the dequeue, so `run_fragment`'s sweep still releases every remote-input
    // lease the refused run carried.
    if let Some(query_id) = request.label.query_id
        && let Some(cause) = retired_cause(retired, query_id)
    {
        return Err(format!(
            "query {query_id} was already retired on this CN: {cause}"
        ));
    }

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

        // The CN already holds every input of this fragment — parked local sender outputs and
        // staged remote batches — so it can declare the stream's EXACT row count and let
        // DuckDB's optimizer size join order / build-side selection (a stream source binds with
        // no rows behind it, so undeclared streams all estimate cardinality 1: the q07 2-CN
        // regression built its hash join on a multi-GB stream while a 2-row stream probed).
        // Best-effort by design: when any contributor's count is unknown (a spilled parked
        // batch, a remote frame that predates the wire's `rows` field), skip the declaration
        // and keep the legacy blind planning for this stream rather than failing the fragment.
        let local_rows: Option<u64> = senders
            .iter()
            .map(|slot| {
                let (sender, sender_stream) = registry.peek(slot)?;
                sender.output_row_count(sender_stream).ok()
            })
            .sum();
        let remote_rows: Option<u64> = request
            .remote_inputs
            .iter()
            .filter(|(node_id, _, _)| *node_id == schema.node_id)
            .flat_map(|(_, _, batches)| batches.iter().map(|batch| batch.rows))
            .sum();
        match (local_rows, remote_rows) {
            (Some(local), Some(remote)) => {
                let rows = local + remote;
                fragment
                    .declare_input_cardinality(stream_id, rows)
                    .map_err(|err| {
                        format!("failed to declare cardinality of stream {stream_id}: {err}")
                    })?;
                info!(stream_id, rows, "declared input stream cardinality");
            }
            _ => warn!(
                stream_id,
                "input stream row count unknown; planning without a declared cardinality"
            ),
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
            let sender_id = u32::try_from(slot.sender_id)
                .map_err(|_| format!("negative sender id {}", slot.sender_id))?;
            let (sender, sender_stream) = registry.claim(slot, "relay")?;
            let moved = fragment
                .relay_from(sender, sender_stream, stream_id, sender_id)
                .map_err(|err| format!("failed to relay sender {sender_id}: {err}"))?;
            // This destination's stream is drained; release its claim (the fragment drops with
            // the last claim, freeing the GPU batches).
            registry.release(slot)?;
            info!(
                stream_id,
                sender_id,
                batches = moved,
                "relayed native batches across a fragment boundary"
            );
        }
    }

    // Remote senders: a ticketed batch was copied into pool memory when its frame arrived and
    // moves into the stream without another copy; a legacy batch still sits in this CN's
    // staging arena, so push it (deep copy into pool memory) and release its lease at once.
    // Then close the sender.
    for (node_id, sender_id, batches) in &request.remote_inputs {
        let stream_id = stream_id_of(*node_id)?;
        let sender = u32::try_from(*sender_id)
            .map_err(|_| format!("negative remote sender id {sender_id}"))?;
        for batch in batches {
            if let Some(ticket) = batch.ticket {
                fragment.push_inbound(stream_id, ticket).map_err(|err| {
                    format!(
                        "failed to take staged remote batch {ticket} from sender {sender_id} \
                         into stream {stream_id}: {err}"
                    )
                })?;
                taken.insert(ticket);
                continue;
            }
            let staged = sirius::PackedBatch {
                metadata: batch.metadata.clone(),
                offset: batch.offset,
                len: batch.len,
                // push_packed never reads the count; the receiver consumed it before build()
                // through declare_input_cardinality.
                rows: batch.rows.unwrap_or(0),
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
    run_after_run_hook(request);

    if !request.outputs.is_empty() {
        // Gate 2: the query may have been retired while this fragment ran (a cancel mid-run, or
        // a sibling failing on another CN). Output parked now would have no owner-triggered
        // release, so drop the fragment instead of parking it.
        if let Some(query_id) = request.label.query_id
            && let Some(cause) = retired_cause(retired, query_id)
        {
            return Err(format!(
                "fragment of query {query_id} finished after the query was retired on this CN; \
                 its output was dropped instead of parked: {cause}"
            ));
        }
        // Park ONCE; each destination claims (park id, its stream). A duplicate slot would let
        // two claims race over one stream — the registry refuses before inserting anything.
        registry.park(request.label.query_id, &request.outputs, fragment)?;
        return Ok(None);
    }
    // `result_to_arrow` drains the stream and drops the context-referencing wrapper here, on the
    // engine thread, returning owned batches whose buffers are released via their own Arrow C
    // release callbacks — independent of the context. So they are safe to send to, and drop on,
    // the caller's thread.
    fragment
        .result_to_arrow()
        .map(|result| Some(FragmentResult::new(result.batches)))
        .map_err(|err| err.to_string())
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
        self.engine_send(make_request(respond_tx))?;
        respond_rx
            .recv()
            .map_err(|_| "sirius-engine thread dropped the response".to_string())?
    }

    /// Sends one request without waiting for an answer.
    fn engine_send(&self, request: EngineRequest) -> Result<(), String> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .ok_or_else(|| "sirius-engine is shutting down".to_string())?
            .send(request)
            .map_err(|_| "sirius-engine thread is not running".to_string())
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
                label: run.label,
                respond,
                #[cfg(test)]
                after_run: None,
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

    fn inbound_store_available(&self) -> bool {
        self.inbound.is_some()
    }

    fn stage_inbound(&self, batch: &StagedBatch) -> Result<u64, String> {
        let store = self
            .inbound
            .as_ref()
            .ok_or_else(|| "no inbound store (SIRIUS_EXCHANGE_STAGING_BYTES unset)".to_string())?;
        let packed = sirius::PackedBatch {
            metadata: batch.metadata.clone(),
            offset: batch.offset,
            len: batch.len,
            rows: batch.rows.unwrap_or(0),
        };
        store
            .stage(&packed)
            .map_err(|err| format!("failed to stage a {} byte inbound frame: {err}", batch.len))
    }

    fn drop_inbound(&self, ticket: u64) -> Result<(), String> {
        let store = self
            .inbound
            .as_ref()
            .ok_or_else(|| "no inbound store (SIRIUS_EXCHANGE_STAGING_BYTES unset)".to_string())?;
        store
            .drop(ticket)
            .map_err(|err| format!("failed to drop staged inbound batch {ticket}: {err}"))
    }

    fn retire_query(
        &self,
        query_id: QueryId,
        trigger: RetireTrigger,
        cause: &str,
    ) -> Result<(), String> {
        // Mark first: a Run already queued ahead of the RetireQuery is refused at its dequeue.
        lock(&self.retired).mark(query_id, cause);
        self.engine_send(EngineRequest::RetireQuery {
            query_id,
            trigger,
            cause: cause.to_string(),
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
                label: FragmentLabel::default(),
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
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
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
                label: FragmentLabel::default(),
                respond: respond_tx,
                after_run: None,
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
                label: FragmentLabel::default(),
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
                label: FragmentLabel::default(),
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

    /// End-to-end: drive a `local_files` parquet plan through the engine actor and read the rows
    /// back, then carry the same rows across a fragment boundary — a sender parks its output on
    /// the GPU and a receiver reads it through its input stream. Exercises the dedicated-thread
    /// bring-up, the channel round-trip, GPU execution, and the park/relay state machine.
    /// Requires a GPU and `LD_LIBRARY_PATH` to the built engine, like the `sirius` crate's
    /// context test.
    #[test]
    fn engine_executes_local_files_and_sequential_exchange() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());

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
                    label: FragmentLabel::default(),
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
                label: FragmentLabel::default(),
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
        // The relay released the receiver's claim, so the parked fragment is gone: a second
        // release of the same slot must be a loud error, never a silent double-drop.
        assert!(
            engine.drop_parked(slot).is_err(),
            "the relayed slot was already released"
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

    /// Fusion on a real engine: the users fixture read once as the sender + receiver pair the CN
    /// runs today (the leaf parks, the receiver relays stream 7) and once as the fused fragment
    /// `fold_deferred_plans` hands the translator instead (the leaf's scan spliced over the
    /// receiver's exchange, translated with no stream input) returns the same rows. Pins "fused
    /// plan == single-fragment plan" on the GPU without touching engine code.
    #[test]
    fn engine_executes_a_fused_leaf_like_the_sender_receiver_pair() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let path_str = path.to_str().unwrap();
        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");

        // The pair: the leaf parks under stream 7 of receiver (300, 2), which reads it back.
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(300, 2),
            node_id: 7,
            sender_id: 0,
        };
        park(
            &engine,
            &local_files_plan(path_str, vec!["id".to_string(), "name".to_string()]),
            slot,
            labelled(300, 3),
        );
        let paired = users_rows(
            receive(&engine, slot, labelled(300, 2))
                .expect("receive the parked rows")
                .expect("a result fragment returns rows"),
        );

        // The fused fragment: what the CN translates once the leaf's plan was deferred.
        let file_size = std::fs::metadata(&path).unwrap().len() as i64;
        let (receiver, leaf) =
            crate::compute_node_service::tests::users_shuffle_pair(300, path_str, file_size);
        let fused = starrocks_plan_translator::fusion::splice(receiver, 7, &leaf)
            .expect("the pair is fusable");
        let translated = starrocks_plan_translator::PlanTranslator::new()
            .translate_fragment(&fused)
            .expect("the fused plan translates");
        assert!(
            translated.stream_inputs.is_empty(),
            "a fused leaf leaves no stream to declare"
        );
        let fused_rows = users_rows(run_result(&engine, &translated));

        assert_eq!(fused_rows, paired);
        assert_eq!(
            fused_rows,
            vec![
                (1, "a".to_string()),
                (2, "b".to_string()),
                (3, "c".to_string())
            ]
        );
    }

    /// The `(id, name)` rows of a users-fixture result, sorted by id.
    fn users_rows(result: FragmentResult) -> Vec<(i64, String)> {
        let mut rows = Vec::new();
        for batch in &result.batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("the id column is int64");
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("the name column is utf8");
            for row in 0..batch.num_rows() {
                rows.push((ids.value(row), names.value(row).to_string()));
            }
        }
        rows.sort();
        rows
    }

    /// A run labelled with query `hi` and instance `(hi, instance)`.
    fn labelled(hi: i64, instance: i64) -> FragmentLabel {
        FragmentLabel {
            query_id: Some(FragmentInstanceId::from_halves(hi, 0)),
            fragment_instance_id: Some(FragmentInstanceId::from_halves(hi, instance)),
        }
    }

    /// A destination slot on receiver `(100, receiver)`, exchange node `node_id`, sender 0.
    fn slot_for(receiver: i64, node_id: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(100, receiver),
            node_id,
            sender_id: 0,
        }
    }

    /// Parks the users fixture (3 rows) under `slot`, labelled `label`.
    fn park(engine: &SiriusEngine, plan: &TranslatedPlan, slot: SenderSlot, label: FragmentLabel) {
        assert!(
            engine
                .run(FragmentRun {
                    plan,
                    inputs: Vec::new(),
                    remote_inputs: Vec::new(),
                    outputs: vec![slot],
                    broadcast: false,
                    hash_keys: Vec::new(),
                    label,
                })
                .expect("run the sender fragment")
                .is_none()
        );
    }

    /// Runs a receiver over `slot`'s stream, labelled `label`.
    fn receive(
        engine: &SiriusEngine,
        slot: SenderSlot,
        label: FragmentLabel,
    ) -> Result<Option<FragmentResult>, String> {
        let receiver = stream_plan(slot.node_id, &[("id", false), ("name", true)]);
        engine.run(FragmentRun {
            plan: &receiver,
            inputs: vec![(slot.node_id, vec![slot])],
            remote_inputs: Vec::new(),
            outputs: Vec::new(),
            broadcast: false,
            hash_keys: Vec::new(),
            label,
        })
    }

    fn row_count(result: Option<FragmentResult>) -> usize {
        result
            .expect("a result fragment returns rows")
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum()
    }

    /// A run that fails before `build()`: a stream read with no sender behind it.
    fn failing_run(engine: &SiriusEngine, label: FragmentLabel) -> String {
        let plan = stream_plan(99, &[("id", false)]);
        engine
            .run(FragmentRun {
                plan: &plan,
                inputs: Vec::new(),
                remote_inputs: Vec::new(),
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
                label,
            })
            .expect_err("a stream with no sender fails before build")
    }

    /// A run of query A fails: A's parked output is dropped and poisoned with the cause, later
    /// runs of A are refused at gate 1 (still sweeping the remote-input leases they carried),
    /// and query B's parked output is untouched.
    #[test]
    fn a_failed_run_retires_only_its_own_query() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
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

        let (a1, b1) = (slot_for(1, 7), slot_for(2, 8));
        park(&engine, &plan, a1, labelled(200, 1));
        park(&engine, &plan, b1, labelled(201, 1));

        let err = failing_run(&engine, labelled(200, 2));
        assert!(err.contains("no sender output"), "{err}");

        // Gate 1: any later run of A is refused and names the original failure.
        let err = failing_run(&engine, labelled(200, 3));
        assert!(
            err.contains("already retired") && err.contains("no sender output"),
            "{err}"
        );

        // A's parked output is gone, and its slot says why.
        let err = receive(&engine, a1, FragmentLabel::default()).unwrap_err();
        assert!(
            err.contains("was discarded when its query was retired"),
            "{err}"
        );
        engine
            .drop_parked(a1)
            .expect("a slot retired with its query releases Ok once");
        assert!(
            engine.drop_parked(a1).is_err(),
            "and is a loud error the second time"
        );

        // B is untouched.
        assert_eq!(
            row_count(receive(&engine, b1, labelled(201, 2)).unwrap()),
            3
        );

        // A refused run still sweeps the remote-input leases it carried.
        let lease = engine.staging_lease(4096).expect("lease");
        let receiver = stream_plan(9, &[("id", false), ("name", true)]);
        let err = engine
            .run(FragmentRun {
                plan: &receiver,
                inputs: Vec::new(),
                remote_inputs: vec![(
                    9,
                    0,
                    vec![StagedBatch {
                        metadata: vec![1],
                        offset: lease,
                        len: 4096,
                        rows: Some(1),
                        ticket: None,
                    }],
                )],
                outputs: Vec::new(),
                broadcast: false,
                hash_keys: Vec::new(),
                label: labelled(200, 4),
            })
            .expect_err("refused at gate 1");
        assert!(err.contains("already retired"), "{err}");
        let probe = engine.staging_lease(1024).expect("arena drained");
        assert_eq!(probe, 0, "the sweep returned the refused run's lease");
        engine.staging_release(probe).unwrap();

        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    /// The CN-side route: `retire_query` drops the query's parked output (fire-and-forget, but
    /// FIFO ahead of the next run) and refuses its later runs with the CN's cause.
    #[test]
    fn retire_query_drops_parked_output_and_refuses_later_runs() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
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

        let c1 = slot_for(3, 7);
        park(&engine, &plan, c1, labelled(202, 1));
        engine
            .retire_query(
                FragmentInstanceId::from_halves(202, 0),
                RetireTrigger::CnErr,
                "translation failed",
            )
            .expect("retire is a non-blocking send");

        let err = failing_run(&engine, labelled(202, 2));
        assert!(
            err.contains("already retired") && err.contains("translation failed"),
            "{err}"
        );
        let err = receive(&engine, c1, FragmentLabel::default()).unwrap_err();
        assert!(err.contains("retired: translation failed"), "{err}");
        let probe = engine.staging_lease(1024).expect("arena untouched");
        assert_eq!(probe, 0);
        engine.staging_release(probe).unwrap();

        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    /// An unlabeled failure (test fixtures, no StarRocks ids) retires only unlabeled output.
    #[test]
    fn an_unlabeled_failure_retires_only_the_unlabeled_bucket() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );
        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");

        let (u1, u2, l) = (slot_for(5, 7), slot_for(6, 7), slot_for(7, 8));
        park(&engine, &plan, u1, FragmentLabel::default());
        park(&engine, &plan, u2, FragmentLabel::default());
        park(&engine, &plan, l, labelled(203, 1));

        failing_run(&engine, FragmentLabel::default());

        for slot in [u1, u2] {
            let err = receive(&engine, slot, FragmentLabel::default()).unwrap_err();
            assert!(
                err.contains("was discarded when its query was retired"),
                "{err}"
            );
        }
        assert_eq!(row_count(receive(&engine, l, labelled(203, 2)).unwrap()), 3);
    }

    /// Gate 2: a query retired while one of its senders is inside `run()` gets that sender's
    /// output dropped, not parked -- there would be nobody left to release it.
    #[test]
    fn output_parked_after_a_retire_is_dropped_not_parked() {
        let _guard = crate::GPU_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );
        let engine = SiriusEngine::start(test_settings()).expect("bring up sirius engine");

        let d = FragmentInstanceId::from_halves(204, 0);
        let d1 = slot_for(8, 7);
        let retired = Arc::clone(&engine.retired);
        let (respond_tx, respond_rx) = channel();
        engine
            .requests
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .send(EngineRequest::Run(ExecuteRequest {
                plan: plan.to_substrait_bytes(),
                stream_inputs: Vec::new(),
                inputs: Vec::new(),
                remote_inputs: Vec::new(),
                outputs: vec![d1],
                broadcast: false,
                hash_keys: Vec::new(),
                label: labelled(204, 1),
                respond: respond_tx,
                // The cancel lands while the fragment runs.
                after_run: Some(Box::new(move || {
                    lock(&retired).mark(d, "cancelled mid-run");
                })),
            }))
            .unwrap();
        let err = respond_rx
            .recv()
            .expect("engine response")
            .expect_err("dropped instead of parked");
        assert!(
            err.contains("dropped instead of parked") && err.contains("cancelled mid-run"),
            "{err}"
        );

        // Never parked: the drop is the generic loud error, not a torn-down release.
        let err = engine.drop_parked(d1).unwrap_err();
        assert!(err.contains("no parked sender output to drop"), "{err}");
        // And D stays retired.
        let err = failing_run(&engine, labelled(204, 2));
        assert!(err.contains("already retired"), "{err}");
    }
}
