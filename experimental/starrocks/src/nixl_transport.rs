//! The NIXL exchange transport.
//!
//! The engine's GPU pool is one `cudaMalloc` slab, registered once with NIXL as VRAM. To ship a
//! parked batch, the sender exports it, asks the receiver (Alloc) for matching buffers from the
//! receiver's own pool, WRITEs every buffer straight into them, and announces the receiver's token
//! with a Packed frame. No staging copy on either side.
//!
//! One thread owns the [`Agent`] (nixl-sys documents a multithreading deadlock caveat), so every
//! WRITE goes through it. It serves all of a fragment's remote outputs at once, round robin, each
//! with its own window of WRITEs, and ships a streamed output as its fragment produces it. What peers ask of this CN (Md, Alloc, Release) never touches that thread:
//! it reads the cached metadata or calls the [`DirectExchange`] on the brpc thread.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nixl_sys::{
    Agent, MemType, MemoryRegion, NixlDescriptor, OptArgs, RegistrationHandle, XferDescList,
    XferOp, XferRequest, XferStatus,
};
use prost::Message;
use sirius::DirectExchange;
use tracing::{info, warn};

use crate::fragment_executor::{
    DrainNext, ExportedBatch, FragmentExecutor, OutputDrain, SenderSlot,
};
use crate::nixl_chunk::{
    AllocReply, NixlEndpoint, NixlEnvelope, StreamHop, alloc_params, control_params, failed_params,
    packed_params,
};
use crate::proto::starrocks::{
    PTransmitChunkParams, PTransmitChunkResult,
    p_internal_service_brpc::{SERVICE_NAME, methods},
};
use crate::prpc;
use starrocks_thrift::status_code::TStatusCode;

/// Appended to bring-up errors, whose usual cause is the environment.
const ENV_HINT: &str = "source experimental/starrocks/scripts/cn-env.sh (NIXL_PREFIX, \
                        NIXL_PLUGIN_DIR, LD_LIBRARY_PATH) and set \
                        UCX_TLS=cuda_copy,cuda_ipc,tcp,self";

const RPC_TIMEOUT: Duration = Duration::from_secs(60);

/// Bounds a failure frame, from connecting to its reply, which a sender sends while its query is
/// already failing: a peer that does not answer by then is likely the cause.
const FAILED_TIMEOUT: Duration = Duration::from_secs(5);

/// At most this many failure frames are in flight at once, each on its own thread.
const FAILED_IN_FLIGHT: usize = 64;

/// How long the transport waits on one streamed output when every output is waiting on its
/// fragment and no WRITE is in flight.
const IDLE_WAIT: Duration = Duration::from_millis(2);

/// Handle to the transport thread.
#[derive(Debug)]
pub struct NixlTransport {
    /// Taken on drop to end the thread's loop.
    requests: Option<Sender<ShipRequest>>,
    thread: Option<JoinHandle<()>>,
    local_md: Vec<u8>,
    exchange: Arc<DirectExchange>,
}

/// Hops shipped together, and where to report how it went.
struct ShipRequest {
    hops: Vec<Hop>,
    respond: Sender<Result<(), String>>,
}

/// One remote output and where its batches come from.
struct Hop {
    peer: SocketAddr,
    slot: SenderSlot,
    names: Vec<String>,
    source: Source,
}

enum Source {
    /// Output parked after its fragment ran, exported through the engine thread.
    Parked(Arc<dyn FragmentExecutor>),
    /// Output exported while its fragment runs.
    Streamed(Box<dyn OutputDrain>),
}

impl Source {
    fn next(&mut self, slot: SenderSlot, timeout: Duration) -> Result<DrainNext, String> {
        match self {
            Source::Parked(executor) => Ok(executor
                .export_direct_next(slot)?
                .map_or(DrainNext::End, DrainNext::Batch)),
            Source::Streamed(drain) => drain.next(timeout),
        }
    }
}

impl NixlTransport {
    /// Brings up an agent named `agent_name` with the UCX backend on its own thread and registers
    /// `exchange`'s slab with it, so a broken NIXL install fails here, before any query.
    pub fn start(agent_name: String, exchange: Arc<DirectExchange>) -> Result<Self, String> {
        let (request_tx, request_rx) = channel();
        let (ready_tx, ready_rx) = channel();
        let thread_exchange = Arc::clone(&exchange);
        let thread = std::thread::Builder::new()
            .name("nixl-transport".to_string())
            .spawn(
                move || match Transport::bring_up(&agent_name, thread_exchange) {
                    Ok(mut transport) => {
                        if ready_tx.send(Ok(transport.local_md.clone())).is_ok() {
                            transport.serve(request_rx);
                        }
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(err));
                    }
                },
            )
            .map_err(|err| format!("failed to spawn the nixl-transport thread: {err}"))?;
        let local_md = ready_rx
            .recv()
            .map_err(|_| "the nixl-transport thread exited during bring-up".to_string())??;
        Ok(Self {
            requests: Some(request_tx),
            thread: Some(thread),
            local_md,
            exchange,
        })
    }
}

impl Drop for NixlTransport {
    fn drop(&mut self) {
        // Joined so the slab is deregistered before the engine that owns it goes.
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl NixlEndpoint for NixlTransport {
    fn local_md(&self) -> Vec<u8> {
        self.local_md.clone()
    }

    fn allocate(&self, layout: &[u8]) -> Result<AllocReply, String> {
        let (token, buffers) = self
            .exchange
            .allocate(layout)
            .map_err(|err| format!("failed to allocate receive buffers: {err}"))?;
        Ok(AllocReply {
            token,
            device: self.exchange.region().0,
            buffers,
        })
    }

    fn release(&self, token: u64) {
        release(&self.exchange, token);
    }

    fn seal(&self, token: u64) -> Result<(), String> {
        self.exchange
            .seal(token)
            .map_err(|err| format!("failed to seal received batch {token}: {err}"))
    }

    fn outstanding(&self) -> usize {
        self.exchange.outstanding().unwrap_or_else(|err| {
            warn!(error = %err, "failed to count direct-exchange buffers");
            0
        })
    }

    fn send(
        &self,
        peer: SocketAddr,
        slot: SenderSlot,
        names: Vec<String>,
        executor: Arc<dyn FragmentExecutor>,
    ) -> Result<(), String> {
        self.ship(vec![Hop {
            peer,
            slot,
            names,
            source: Source::Parked(executor),
        }])
    }

    fn stream(&self, hops: Vec<StreamHop>) -> Result<(), String> {
        self.ship(
            hops.into_iter()
                .map(|hop| Hop {
                    peer: hop.peer,
                    slot: hop.slot,
                    names: hop.names,
                    source: Source::Streamed(hop.drain),
                })
                .collect(),
        )
    }

    fn fail(&self, peer: SocketAddr, slot: SenderSlot, error: &str) {
        send_failed(vec![(peer, slot)], error);
    }
}

impl NixlTransport {
    /// Ships `hops` on the transport thread. If that thread cannot take them, each hop still ends
    /// with a failure frame.
    fn ship(&self, hops: Vec<Hop>) -> Result<(), String> {
        let ends: Vec<_> = hops.iter().map(|hop| (hop.peer, hop.slot)).collect();
        let (respond, response) = channel();
        let shipped = self
            .requests
            .as_ref()
            .ok_or_else(|| "the nixl transport is shutting down".to_string())
            .and_then(|requests| {
                requests
                    .send(ShipRequest { hops, respond })
                    .map_err(|_| "the nixl-transport thread is not running".to_string())
            })
            .and_then(|()| {
                response
                    .recv()
                    .map_err(|_| "the nixl-transport thread dropped the hop".to_string())
            });
        match shipped {
            Ok(result) => result,
            Err(err) => {
                send_failed(ends, &err);
                Err(err)
            }
        }
    }
}

/// The slab as a NIXL memory region.
#[derive(Debug)]
struct SlabRegion {
    base: usize,
    len: usize,
    device: u64,
}

impl MemoryRegion for SlabRegion {
    unsafe fn as_ptr(&self) -> *const u8 {
        self.base as *const u8
    }

    fn size(&self) -> usize {
        self.len
    }
}

impl NixlDescriptor for SlabRegion {
    fn mem_type(&self) -> MemType {
        MemType::Vram
    }

    fn device_id(&self) -> u64 {
        self.device
    }
}

/// Transport-thread state.
struct Transport {
    // Fields drop in declaration order: the slab is deregistered before `exchange`, which may
    // hold the last reference keeping it allocated.
    _registration: RegistrationHandle,
    exchange: Arc<DirectExchange>,
    agent: Agent,
    device: u64,
    local_md: Vec<u8>,
    /// Remote agent name per peer, once its metadata is loaded.
    peers: HashMap<SocketAddr, String>,
    /// How long a posted WRITE may take (`SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS`).
    xfer_timeout: Duration,
    /// WRITEs that failed or timed out, held until the NIC is done with them.
    quarantine: RefCell<Vec<Quarantined>>,
}

/// A WRITE that failed or timed out while the NIC may still be writing. Dropping its request
/// releases it, and freeing either batch lets the NIC write into memory a later query owns, so
/// all three are held until the WRITE reports success. One that never does stays held for the
/// life of the process, as it would without the quarantine.
struct Quarantined {
    request: XferRequest,
    peer: SocketAddr,
    /// The sender's batch on this CN.
    local: u64,
    /// The receiver's buffers on `peer`, never announced.
    remote: u64,
    since: Instant,
}

/// One posted WRITE, completed oldest first.
struct Write {
    /// `None` once it finished.
    request: Option<XferRequest>,
    /// The sender's batch, released once the WRITE succeeds.
    local: u64,
    /// The receiver's buffers, announced once the WRITE succeeds.
    remote: u64,
    rows: u64,
    bytes: u64,
    posted: Instant,
}

/// One hop being shipped: its source, its WRITEs in flight, and its announce queue.
struct Lane {
    peer: SocketAddr,
    slot: SenderSlot,
    remote: String,
    source: Source,
    streamed: bool,
    inflight: VecDeque<Write>,
    announce: Sender<Announce>,
    /// The error the hop's announcer hit first: the receiver refused a frame, or its CN is gone.
    refused: Arc<OnceLock<String>>,
    /// The source has no more batches.
    drained: bool,
    totals: HopTotals,
}

impl Lane {
    fn done(&self) -> bool {
        self.drained && self.inflight.is_empty()
    }
}

/// What a hop's announcer sends next.
enum Announce {
    /// A finished WRITE: the receiver's token and the batch's rows.
    Batch { token: u64, rows: u64 },
    /// The hop's last frame: EOS, or a failure frame carrying the error that ended the hop.
    End(Result<(), String>),
}

#[derive(Default)]
struct HopTotals {
    frames: u64,
    rows: u64,
    bytes: u64,
    write_us: u64,
}

impl Transport {
    fn bring_up(agent_name: &str, exchange: Arc<DirectExchange>) -> Result<Self, String> {
        check_single_visible_device(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
        let agent = Agent::new(agent_name).map_err(|err| {
            format!("failed to create nixl agent '{agent_name}': {err} ({ENV_HINT})")
        })?;
        let (_, params) = agent
            .get_plugin_params("UCX")
            .map_err(|err| format!("nixl UCX plugin unavailable: {err} ({ENV_HINT})"))?;
        let backend = agent
            .create_backend("UCX", &params)
            .map_err(|err| format!("failed to create the nixl UCX backend: {err} ({ENV_HINT})"))?;
        let mut options =
            OptArgs::new().map_err(|err| format!("failed to create nixl options: {err}"))?;
        options
            .add_backend(&backend)
            .map_err(|err| format!("failed to select the nixl UCX backend: {err}"))?;
        let (device, base, len) = exchange.region();
        let slab = SlabRegion {
            base,
            len: len as usize,
            device: device as u64,
        };
        let registration = agent
            .register_memory(&slab, Some(&options))
            .map_err(|err| {
                format!(
                    "failed to register the {len}-byte GPU slab with nixl: {err}; UCX_TLS must \
                 include cuda_copy ({ENV_HINT})"
                )
            })?;
        let local_md = agent
            .get_local_md()
            .map_err(|err| format!("failed to serialize nixl agent metadata: {err}"))?;
        info!(
            agent = agent_name,
            device, base, len, "nixl transport ready; GPU slab registered"
        );
        Ok(Self {
            _registration: registration,
            exchange,
            agent,
            device: device as u64,
            local_md,
            peers: HashMap::new(),
            xfer_timeout: env_u64("SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS")
                .map_or(Duration::from_secs(30), Duration::from_secs),
            quarantine: RefCell::new(Vec::new()),
        })
    }

    fn serve(&mut self, requests: Receiver<ShipRequest>) {
        for request in requests {
            let result = self.ship(request.hops);
            let _ = request.respond.send(result);
        }
        info!("nixl-transport thread shutting down");
    }

    /// Ships every hop at once, keeping a window of WRITEs in flight per hop. Announces leave in
    /// order from one thread per hop, so a WRITE never waits on the previous batch's announce
    /// round trip. After the first error every hop ends with a failure frame instead of its EOS.
    fn ship(&mut self, hops: Vec<Hop>) -> Result<(), String> {
        self.reap_quarantine();
        let started = Instant::now();
        let remotes = hops
            .iter()
            .map(|hop| self.remote_agent(hop.peer))
            .collect::<Result<Vec<_>, _>>()
            .inspect_err(|err| {
                send_failed(hops.iter().map(|hop| (hop.peer, hop.slot)).collect(), err);
            })?;
        let window = env_u64("SIRIUS_CN_NIXL_WINDOW").map_or(4, |n| n as usize);
        std::thread::scope(|scope| {
            let mut lanes = Vec::with_capacity(hops.len());
            let mut announcers = Vec::with_capacity(hops.len());
            for (hop, remote) in hops.into_iter().zip(remotes) {
                let (announce, frames) = channel::<Announce>();
                let (peer, slot, names) = (hop.peer, hop.slot, hop.names);
                let refused = Arc::new(OnceLock::new());
                let announcer_refused = Arc::clone(&refused);
                announcers.push(scope.spawn(move || {
                    announce_all(
                        slot,
                        &names,
                        frames,
                        &announcer_refused,
                        |params, envelope| call(peer, params, envelope).map(drop),
                        |err| send_failed(vec![(peer, slot)], err),
                    )
                }));
                lanes.push(Lane {
                    peer,
                    slot,
                    remote,
                    streamed: matches!(hop.source, Source::Streamed(_)),
                    source: hop.source,
                    inflight: VecDeque::new(),
                    announce,
                    refused,
                    drained: false,
                    totals: HopTotals::default(),
                });
            }
            let pumped = self.pump(&mut lanes, window);
            // Wait out every WRITE still posted: a finished one frees the sender's batch, and the
            // receiver's buffers it filled are never announced, so they are released.
            for lane in &mut lanes {
                for mut write in std::mem::take(&mut lane.inflight) {
                    if self.complete(&lane.remote, &mut write).is_ok() {
                        let _ = call(
                            lane.peer,
                            control_params(),
                            &NixlEnvelope::Release(write.remote),
                        );
                    } else {
                        self.quarantine(lane.peer, write);
                    }
                }
                let _ = lane.announce.send(Announce::End(pumped.clone()));
            }
            // Dropping the lanes closes every announce queue, so the announcers can finish.
            let refusals: Vec<_> = lanes.iter().map(|lane| Arc::clone(&lane.refused)).collect();
            let shipped: Vec<_> = lanes
                .into_iter()
                .map(|lane| (lane.peer, lane.slot, lane.streamed, lane.totals))
                .collect();
            let mut announced = Ok(());
            for announcer in announcers {
                let result = announcer
                    .join()
                    .unwrap_or_else(|_| Err("the announce thread panicked".to_string()));
                announced = announced.and(result);
            }
            let span_us = started.elapsed().as_micros() as u64;
            for (peer, slot, streamed, totals) in shipped {
                info!(
                    peer = %peer,
                    dest_stream = slot.node_id,
                    sender_id = slot.sender_id,
                    frames = totals.frames,
                    bytes = totals.bytes,
                    rows = totals.rows,
                    window,
                    streamed,
                    write_us = totals.write_us,
                    span_us,
                    "shipping packed exchange hop"
                );
            }
            let refused = refusals.iter().find_map(|refused| refused.get().cloned());
            hops_outcome(pumped, announced, refused)
        })
    }

    /// Serves every lane round robin until each source ended and its WRITEs finished.
    fn pump(&self, lanes: &mut [Lane], window: usize) -> Result<(), String> {
        let mut next_wait = 0;
        loop {
            let mut progressed = false;
            for lane in lanes.iter_mut() {
                refusal(lane)?;
                progressed |= self.fill(lane, window, Duration::ZERO)?;
                progressed |= self.retire(lane)?;
            }
            if lanes.iter().all(Lane::done) {
                return Ok(());
            }
            if progressed {
                continue;
            }
            if lanes.iter().any(|lane| !lane.inflight.is_empty()) {
                std::thread::yield_now();
                continue;
            }
            // Every source is waiting on its fragment: block briefly on one, taking turns.
            let waiting: Vec<usize> = (0..lanes.len()).filter(|&i| !lanes[i].drained).collect();
            let lane = &mut lanes[waiting[next_wait % waiting.len()]];
            next_wait += 1;
            self.fill(lane, window, IDLE_WAIT)?;
        }
    }

    /// Posts the lane's next batches until its window is full or its source has none ready.
    /// Waits up to `timeout` for the first. Returns whether anything changed.
    fn fill(&self, lane: &mut Lane, window: usize, mut timeout: Duration) -> Result<bool, String> {
        let mut progressed = false;
        while !lane.drained && lane.inflight.len() < window {
            match lane.source.next(lane.slot, timeout)? {
                DrainNext::Batch(batch) => {
                    let write = self.post(lane.peer, lane.slot, &lane.remote, batch)?;
                    lane.inflight.push_back(write);
                }
                DrainNext::Waiting => break,
                DrainNext::End => lane.drained = true,
            }
            progressed = true;
            timeout = Duration::ZERO;
        }
        Ok(progressed)
    }

    /// Announces the lane's finished WRITEs, oldest first, without waiting on one in flight.
    fn retire(&self, lane: &mut Lane) -> Result<bool, String> {
        let mut progressed = false;
        while let Some(write) = lane.inflight.front_mut() {
            match self.poll(&lane.remote, write) {
                Ok(false) => break,
                Ok(true) => {
                    let write = lane.inflight.pop_front().expect("a front write");
                    lane.totals.frames += 1;
                    lane.totals.rows += write.rows;
                    lane.totals.bytes += write.bytes;
                    lane.totals.write_us += write.posted.elapsed().as_micros() as u64;
                    let _ = lane.announce.send(Announce::Batch {
                        token: write.remote,
                        rows: write.rows,
                    });
                    progressed = true;
                }
                Err(err) => {
                    let write = lane.inflight.pop_front().expect("a front write");
                    self.quarantine(lane.peer, write);
                    return Err(err);
                }
            }
        }
        Ok(progressed)
    }

    /// Allocates the receiver's buffers for `batch` and posts one WRITE of all of them.
    fn post(
        &self,
        peer: SocketAddr,
        slot: SenderSlot,
        remote: &str,
        batch: ExportedBatch,
    ) -> Result<Write, String> {
        let reply = call(peer, alloc_params(slot), &NixlEnvelope::Alloc(batch.layout))
            .and_then(|reply| AllocReply::decode(&reply))
            .inspect_err(|_| self.release(batch.token))?;
        let request = self
            .create_write(remote, &batch.src, &reply)
            .inspect_err(|_| {
                self.release(batch.token);
                let _ = call(peer, control_params(), &NixlEnvelope::Release(reply.token));
            })?;
        let posted = Instant::now();
        let in_progress = match self.agent.post_xfer_req(&request, None) {
            Ok(in_progress) => in_progress,
            Err(err) => {
                // The NIC may have started; hold the request and both batches as on a timeout.
                self.quarantine.borrow_mut().push(Quarantined {
                    request,
                    peer,
                    local: batch.token,
                    remote: reply.token,
                    since: posted,
                });
                return Err(format!(
                    "failed to post a nixl WRITE to agent '{remote}': {err}"
                ));
            }
        };
        Ok(Write {
            request: in_progress.then_some(request),
            local: batch.token,
            remote: reply.token,
            rows: batch.rows,
            bytes: batch.src.iter().map(|(_, len)| len).sum(),
            posted,
        })
    }

    /// One WRITE of every `src` buffer into the matching buffer of `dst`, not yet posted.
    fn create_write(
        &self,
        remote: &str,
        src: &[(u64, u64)],
        dst: &AllocReply,
    ) -> Result<XferRequest, String> {
        if src.len() != dst.buffers.len() || src.iter().zip(&dst.buffers).any(|(s, d)| s.1 != d.1) {
            return Err(format!(
                "nixl agent '{remote}' allocated buffers {:?} for a batch of {src:?}",
                dst.buffers
            ));
        }
        let descriptors = || {
            XferDescList::new(MemType::Vram)
                .map_err(|err| format!("failed to create a nixl descriptor list: {err}"))
        };
        let (mut local, mut target) = (descriptors()?, descriptors()?);
        for (&(from, len), &(to, _)) in src.iter().zip(&dst.buffers) {
            local.add_desc(from as usize, len as usize, self.device);
            target.add_desc(to as usize, len as usize, dst.device as u64);
        }
        self.agent
            .create_xfer_req(XferOp::Write, &local, &target, remote, None)
            .map_err(|err| format!("failed to create a nixl WRITE to agent '{remote}': {err}"))
    }

    /// Whether `write` finished, without waiting; a finished one frees the sender's batch. A
    /// WRITE that fails or times out may still be in flight, so its request stays in `write`
    /// (dropping it would release it) and both batches stay held; the caller quarantines it.
    fn poll(&self, remote: &str, write: &mut Write) -> Result<bool, String> {
        if let Some(request) = &write.request {
            match self.agent.get_xfer_status(request) {
                Ok(XferStatus::Success) => write.request = None,
                Ok(XferStatus::InProgress) if write.posted.elapsed() < self.xfer_timeout => {
                    return Ok(false);
                }
                status => {
                    let (bytes, timeout) = (write.bytes, self.xfer_timeout);
                    return Err(match status {
                        Ok(_) => format!(
                            "a {bytes}-byte nixl WRITE to agent '{remote}' did not finish within \
                             {timeout:?} (SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS)"
                        ),
                        Err(err) => {
                            format!("a {bytes}-byte nixl WRITE to agent '{remote}' failed: {err}")
                        }
                    });
                }
            }
        }
        self.release(write.local);
        Ok(true)
    }

    /// Waits for `write` to finish; see [`poll`](Self::poll).
    fn complete(&self, remote: &str, write: &mut Write) -> Result<(), String> {
        while !self.poll(remote, write)? {
            std::thread::yield_now();
        }
        Ok(())
    }

    /// The peer's agent name, loading its metadata on first contact.
    fn remote_agent(&mut self, peer: SocketAddr) -> Result<String, String> {
        if let Some(name) = self.peers.get(&peer) {
            return Ok(name.clone());
        }
        let peer_md = call(
            peer,
            control_params(),
            &NixlEnvelope::Md(self.local_md.clone()),
        )?;
        let name = self
            .agent
            .load_remote_md(&peer_md)
            .map_err(|err| format!("failed to load the nixl metadata of {peer}: {err}"))?;
        self.peers.insert(peer, name.clone());
        Ok(name)
    }

    fn release(&self, token: u64) {
        release(&self.exchange, token);
    }

    /// Holds a WRITE that did not finish, with both its batches, until [`reap_quarantine`]
    /// sees it succeed.
    fn quarantine(&self, peer: SocketAddr, mut write: Write) {
        let Some(request) = write.request.take() else {
            return;
        };
        self.quarantine.borrow_mut().push(Quarantined {
            request,
            peer,
            local: write.local,
            remote: write.remote,
            since: write.posted,
        });
        warn!(
            peer = %peer,
            bytes = write.bytes,
            quarantined = self.quarantine.borrow().len(),
            "holding a nixl WRITE that did not finish, and both its batches"
        );
    }

    /// Frees every quarantined WRITE that has since succeeded: the request, the sender's batch,
    /// and the receiver's buffers on the peer. Any other status keeps it held.
    fn reap_quarantine(&self) {
        let mut quarantine = self.quarantine.borrow_mut();
        let held = quarantine.len();
        quarantine.retain(|held| {
            if !matches!(
                self.agent.get_xfer_status(&held.request),
                Ok(XferStatus::Success)
            ) {
                return true;
            }
            self.release(held.local);
            let _ = call(
                held.peer,
                control_params(),
                &NixlEnvelope::Release(held.remote),
            );
            false
        });
        if quarantine.len() < held {
            info!(
                reclaimed = held - quarantine.len(),
                still_held = quarantine.len(),
                oldest_s = quarantine.iter().map(|q| q.since.elapsed().as_secs()).max(),
                "reclaimed quarantined nixl WRITEs"
            );
        }
    }
}

/// Announces batches in order with `send`, then EOS (token 0). The first error, a refused or
/// failed announce or EOS, is recorded in `refused` so the transport stops pulling the hop's
/// output; the batches still to come are released instead of announced. A hop that fails, here
/// or on the transport thread, or that the transport drops without ending, ends with `fail` in
/// place of its EOS: a hop never just stops.
fn announce_all(
    slot: SenderSlot,
    names: &[String],
    frames: Receiver<Announce>,
    refused: &OnceLock<String>,
    send: impl Fn(PTransmitChunkParams, &NixlEnvelope) -> Result<(), String>,
    fail: impl FnOnce(&str),
) -> Result<(), String> {
    let packed = |token, rows| NixlEnvelope::Packed {
        token,
        rows,
        names: names.to_vec(),
    };
    let announce = |params, envelope: &NixlEnvelope| {
        send(params, envelope).inspect_err(|err| {
            let _ = refused.set(err.clone());
        })
    };
    let mut result = Ok(());
    let mut seq = 0;
    let mut frames = frames.into_iter();
    let ended = loop {
        match frames.next() {
            None => break Err("the nixl transport dropped the hop before it ended".to_string()),
            Some(Announce::End(ended)) => break result.clone().and(ended),
            Some(Announce::Batch { token, .. }) if result.is_err() => {
                let _ = send(control_params(), &NixlEnvelope::Release(token));
            }
            Some(Announce::Batch { token, rows }) => {
                result = announce(packed_params(Some(slot), seq, false), &packed(token, rows));
                seq += 1;
            }
        }
    };
    let ended = ended.and_then(|()| announce(packed_params(Some(slot), seq, true), &packed(0, 0)));
    ended.inspect_err(|err| fail(err))
}

/// The hop's announcer failed: the transport stops pulling its output with that error.
fn refusal(lane: &Lane) -> Result<(), String> {
    lane.refused.get().map_or(Ok(()), |err| Err(err.clone()))
}

/// How shipping a set of hops went. A hop's refused announce comes first: the transport stopped
/// because of it, and it carries the receiver's cause, where a later error would be secondary.
fn hops_outcome(
    pumped: Result<(), String>,
    announced: Result<(), String>,
    refused: Option<String>,
) -> Result<(), String> {
    match refused {
        Some(err) => Err(err),
        None => pumped.and(announced),
    }
}

/// Sends each hop's receiver in `ends` a failure frame in place of its EOS, each from its own
/// thread, so a peer that does not answer holds up neither the caller nor the other hops. Best
/// effort: a peer that refuses the frame or does not answer within [`FAILED_TIMEOUT`] is logged,
/// and past [`FAILED_IN_FLIGHT`] frames in flight the rest are dropped.
fn send_failed(ends: Vec<(SocketAddr, SenderSlot)>, error: &str) {
    // Nothing failed when the query ended normally: the receivers end with their own cancel from
    // the FE, and a failure frame would make a CN that has not had it yet fail a finished query.
    if crate::recent_queries::is_normal_end(error) {
        info!(
            error,
            "not sending failure frames: the query ended normally"
        );
        return;
    }
    for (peer, slot) in ends {
        let Some(permit) = FailedPermit::take() else {
            warn!(peer = %peer, ?slot, "too many exchange failure frames in flight; dropping one");
            continue;
        };
        let failed = NixlEnvelope::Failed {
            error: error.to_string(),
        };
        let spawned = std::thread::Builder::new()
            .name("exchange-failed".to_string())
            .spawn(move || {
                let _permit = permit;
                let deadline = Instant::now() + FAILED_TIMEOUT;
                match call_by(peer, failed_params(slot), &failed, deadline) {
                    Ok(_) => info!(peer = %peer, ?slot, "sent an exchange failure frame"),
                    Err(err) => warn!(
                        peer = %peer,
                        ?slot,
                        error = %err,
                        "failed to send an exchange failure frame"
                    ),
                }
            });
        if let Err(err) = spawned {
            warn!(peer = %peer, ?slot, error = %err, "cannot spawn an exchange failure frame");
        }
    }
}

/// One of the [`FAILED_IN_FLIGHT`] failure frames that may be in flight; returned when dropped.
struct FailedPermit;

static FAILED_IN_FLIGHT_NOW: AtomicUsize = AtomicUsize::new(0);

impl FailedPermit {
    fn take() -> Option<Self> {
        FAILED_IN_FLIGHT_NOW
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |now| {
                (now < FAILED_IN_FLIGHT).then_some(now + 1)
            })
            .ok()
            .map(|_| Self)
    }
}

impl Drop for FailedPermit {
    fn drop(&mut self) {
        FAILED_IN_FLIGHT_NOW.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One `transmit_chunk` round trip to `peer`, returning the reply attachment.
fn call(
    peer: SocketAddr,
    params: PTransmitChunkParams,
    envelope: &NixlEnvelope,
) -> Result<Vec<u8>, String> {
    let reply = prpc::call_blocking(
        peer,
        SERVICE_NAME,
        methods::TRANSMIT_CHUNK,
        params.encode_to_vec(),
        envelope.encode(),
        RPC_TIMEOUT,
    );
    transmit_reply(peer, reply)
}

/// [`call`] that gives up once `deadline` passes.
fn call_by(
    peer: SocketAddr,
    params: PTransmitChunkParams,
    envelope: &NixlEnvelope,
    deadline: Instant,
) -> Result<Vec<u8>, String> {
    let reply = prpc::call_blocking_by(
        peer,
        SERVICE_NAME,
        methods::TRANSMIT_CHUNK,
        params.encode_to_vec(),
        envelope.encode(),
        deadline,
    );
    transmit_reply(peer, reply)
}

/// The attachment of `peer`'s `transmit_chunk` reply, or why it failed.
fn transmit_reply(
    peer: SocketAddr,
    reply: anyhow::Result<(Vec<u8>, Vec<u8>)>,
) -> Result<Vec<u8>, String> {
    let (body, attachment) = reply.map_err(|err| format!("transmit_chunk to {peer}: {err:#}"))?;
    let status = PTransmitChunkResult::decode(body.as_slice())
        .map_err(|err| format!("transmit_chunk reply from {peer}: {err}"))?
        .status
        .ok_or_else(|| format!("transmit_chunk reply from {peer} carries no status"))?;
    // The peer already failed the query: its message is that failure's cause, which is what the
    // FE should report rather than this refusal.
    if status.status_code == TStatusCode::CANCELLED.0 {
        return Err(status.error_msgs.join("; "));
    }
    if status.status_code != 0 {
        return Err(format!(
            "{peer} refused transmit_chunk: {}",
            status.error_msgs.join("; ")
        ));
    }
    Ok(attachment)
}

fn release(exchange: &DirectExchange, token: u64) {
    if let Err(err) = exchange.release(token) {
        warn!(token, error = %err, "failed to release a direct-exchange token");
    }
}

/// A positive integer from the environment.
fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok().filter(|&n| n > 0)
}

/// One CN per GPU: cross-process `cuda_ipc` is only proven with each CN seeing its own device.
fn check_single_visible_device(visible: Option<&str>) -> Result<(), String> {
    let Some(visible) = visible else {
        return Ok(());
    };
    let devices = visible.split(',').filter(|d| !d.trim().is_empty()).count();
    if devices > 1 {
        return Err(format!(
            "CUDA_VISIBLE_DEVICES={visible:?} names {devices} devices; pin each CN to one GPU"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::result_store::FragmentInstanceId;

    /// What one hop sent: `(eos, envelope)` per frame, the failure it ended with, and the error
    /// it recorded for the transport.
    type Sent = (Vec<(bool, NixlEnvelope)>, Option<String>, Option<String>);

    /// Runs `announce_all` over `frames`, refusing the announce of `refuse` (a token; 0 is EOS)
    /// with "busy".
    fn announce(frames: Vec<Announce>, refuse: Option<u64>) -> (Result<(), String>, Sent) {
        let slot = SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(1, 2),
            node_id: 3,
            sender_id: 0,
        };
        let (announce, received) = channel();
        for frame in frames {
            announce.send(frame).unwrap();
        }
        drop(announce);
        let sent = Mutex::new(Vec::new());
        let mut failed = None;
        let refused = OnceLock::new();
        let result = announce_all(
            slot,
            &["id".to_string()],
            received,
            &refused,
            |params, envelope| {
                sent.lock()
                    .unwrap()
                    .push((params.eos == Some(true), envelope.clone()));
                match envelope {
                    NixlEnvelope::Packed { token, .. } if Some(*token) == refuse => {
                        Err("busy".to_string())
                    }
                    _ => Ok(()),
                }
            },
            |err| failed = Some(err.to_string()),
        );
        (
            result,
            (sent.into_inner().unwrap(), failed, refused.into_inner()),
        )
    }

    fn batch(token: u64) -> Announce {
        Announce::Batch { token, rows: 1 }
    }

    fn packed(token: u64) -> NixlEnvelope {
        NixlEnvelope::Packed {
            token,
            rows: u64::from(token != 0),
            names: vec!["id".to_string()],
        }
    }

    #[test]
    fn a_hop_ends_with_eos_once_every_batch_is_announced() {
        let (result, (sent, failed, refused)) =
            announce(vec![batch(5), batch(6), Announce::End(Ok(()))], None);
        assert_eq!(result, Ok(()));
        assert_eq!(
            sent,
            vec![(false, packed(5)), (false, packed(6)), (true, packed(0))]
        );
        assert_eq!((failed, refused), (None, None));
    }

    #[test]
    fn a_hop_that_fails_ends_with_a_failure_frame_and_no_eos() {
        // The transport fails the hop: its own error, nothing for the transport to stop on.
        let (result, (sent, failed, refused)) = announce(
            vec![batch(5), Announce::End(Err("WRITE failed".to_string()))],
            None,
        );
        assert_eq!(result, Err("WRITE failed".to_string()));
        assert_eq!(sent, vec![(false, packed(5))]);
        assert_eq!(failed.as_deref(), Some("WRITE failed"));
        assert_eq!(refused, None);

        // The peer refuses an announce: the transport is told to stop, and the batches after it
        // are released, not announced.
        let (result, (sent, failed, refused)) =
            announce(vec![batch(5), batch(6), Announce::End(Ok(()))], Some(5));
        assert_eq!(result, Err("busy".to_string()));
        assert_eq!(
            sent,
            vec![(false, packed(5)), (false, NixlEnvelope::Release(6))]
        );
        assert_eq!(failed.as_deref(), Some("busy"));
        assert_eq!(refused.as_deref(), Some("busy"));

        // The peer refuses the EOS: the hop still ends with a failure frame.
        let (result, (sent, failed, refused)) =
            announce(vec![batch(5), Announce::End(Ok(()))], Some(0));
        assert_eq!(result, Err("busy".to_string()));
        assert_eq!(sent, vec![(false, packed(5)), (true, packed(0))]);
        assert_eq!(failed.as_deref(), Some("busy"));
        assert_eq!(refused.as_deref(), Some("busy"));

        // The transport drops the hop without ending it.
        let (result, (sent, failed, _)) = announce(vec![batch(5)], None);
        assert!(result.is_err());
        assert_eq!(sent, vec![(false, packed(5))]);
        assert!(failed.unwrap().contains("dropped the hop"));
    }

    #[test]
    fn a_refused_hop_stops_the_transport_with_the_receivers_cause() {
        let lane = Lane {
            peer: "127.0.0.1:9".parse().unwrap(),
            slot: SenderSlot {
                fragment_instance_id: FragmentInstanceId::from_halves(1, 2),
                node_id: 3,
                sender_id: 0,
            },
            remote: String::new(),
            source: Source::Streamed(Box::new(NoBatches)),
            streamed: true,
            inflight: VecDeque::new(),
            announce: channel().0,
            refused: Arc::new(OnceLock::new()),
            drained: false,
            totals: HopTotals::default(),
        };
        assert_eq!(refusal(&lane), Ok(()));
        lane.refused.set("scan exploded".to_string()).unwrap();
        assert_eq!(refusal(&lane), Err("scan exploded".to_string()));

        // A later, secondary error of the pump doesn't replace the receiver's cause.
        let secondary = || Err("failed to allocate receive buffers".to_string());
        assert_eq!(
            hops_outcome(
                secondary(),
                Err("scan exploded".to_string()),
                Some("scan exploded".into())
            ),
            Err("scan exploded".to_string())
        );
        assert_eq!(hops_outcome(secondary(), Ok(()), None), secondary());
    }

    #[derive(Debug)]
    struct NoBatches;

    impl OutputDrain for NoBatches {
        fn next(&mut self, _timeout: Duration) -> Result<DrainNext, String> {
            Ok(DrainNext::End)
        }
    }

    #[test]
    fn failure_frames_in_flight_are_capped() {
        let permits: Vec<_> = std::iter::from_fn(FailedPermit::take)
            .take(FAILED_IN_FLIGHT + 1)
            .collect();
        assert_eq!(permits.len(), FAILED_IN_FLIGHT);
        assert!(FailedPermit::take().is_none());
        drop(permits);
        assert!(FailedPermit::take().is_some());
    }

    #[test]
    fn one_cn_sees_one_gpu() {
        assert!(check_single_visible_device(None).is_ok());
        assert!(check_single_visible_device(Some(" 3, ")).is_ok());
        assert!(check_single_visible_device(Some("0,1")).is_err());
    }
}
