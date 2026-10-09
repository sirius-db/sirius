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
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
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
    AllocReply, NixlEndpoint, NixlEnvelope, StreamHop, control_params, packed_params,
};
use crate::proto::starrocks::{
    PTransmitChunkParams, PTransmitChunkResult,
    p_internal_service_brpc::{SERVICE_NAME, methods},
};
use crate::prpc;

/// Appended to bring-up errors, whose usual cause is the environment.
const ENV_HINT: &str = "source experimental/starrocks/scripts/cn-env.sh (NIXL_PREFIX, \
                        NIXL_PLUGIN_DIR, LD_LIBRARY_PATH) and set \
                        UCX_TLS=cuda_copy,cuda_ipc,tcp,self";

const RPC_TIMEOUT: Duration = Duration::from_secs(60);

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
}

impl NixlTransport {
    fn ship(&self, hops: Vec<Hop>) -> Result<(), String> {
        let (respond, response) = channel();
        self.requests
            .as_ref()
            .ok_or_else(|| "the nixl transport is shutting down".to_string())?
            .send(ShipRequest { hops, respond })
            .map_err(|_| "the nixl-transport thread is not running".to_string())?;
        response
            .recv()
            .map_err(|_| "the nixl-transport thread dropped the hop".to_string())?
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
    announce: Sender<(u64, u64)>,
    /// The source has no more batches.
    drained: bool,
    totals: HopTotals,
}

impl Lane {
    fn done(&self) -> bool {
        self.drained && self.inflight.is_empty()
    }
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
    /// round trip. After the first error no hop sends its EOS.
    fn ship(&mut self, hops: Vec<Hop>) -> Result<(), String> {
        self.reap_quarantine();
        let started = Instant::now();
        let remotes = hops
            .iter()
            .map(|hop| self.remote_agent(hop.peer))
            .collect::<Result<Vec<_>, _>>()?;
        let window = env_u64("SIRIUS_CN_NIXL_WINDOW").map_or(4, |n| n as usize);
        std::thread::scope(|scope| {
            let mut lanes = Vec::with_capacity(hops.len());
            let mut announcers = Vec::with_capacity(hops.len());
            for (hop, remote) in hops.into_iter().zip(remotes) {
                let (announce, frames) = channel::<(u64, u64)>();
                let (peer, slot, names) = (hop.peer, hop.slot, hop.names);
                announcers.push(scope.spawn(move || announce_all(peer, slot, &names, frames)));
                lanes.push(Lane {
                    peer,
                    slot,
                    remote,
                    streamed: matches!(hop.source, Source::Streamed(_)),
                    source: hop.source,
                    inflight: VecDeque::new(),
                    announce,
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
                if pumped.is_ok() {
                    let _ = lane.announce.send((0, 0));
                }
            }
            // Dropping the lanes closes every announce queue, so the announcers can finish.
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
            pumped.and(announced)
        })
    }

    /// Serves every lane round robin until each source ended and its WRITEs finished.
    fn pump(&self, lanes: &mut [Lane], window: usize) -> Result<(), String> {
        let mut next_wait = 0;
        loop {
            let mut progressed = false;
            for lane in lanes.iter_mut() {
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
                    let write = self.post(lane.peer, &lane.remote, batch)?;
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
                    let _ = lane.announce.send((write.remote, write.rows));
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
    fn post(&self, peer: SocketAddr, remote: &str, batch: ExportedBatch) -> Result<Write, String> {
        let reply = call(peer, control_params(), &NixlEnvelope::Alloc(batch.layout))
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

/// Sends `(token, rows)` frames in order, then EOS (token 0). After the first failure the rest
/// are not announced, so their receive buffers are released instead.
fn announce_all(
    peer: SocketAddr,
    slot: SenderSlot,
    names: &[String],
    frames: Receiver<(u64, u64)>,
) -> Result<(), String> {
    let mut result = Ok(());
    for (seq, (token, rows)) in frames.into_iter().enumerate() {
        if result.is_err() {
            if token != 0 {
                let _ = call(peer, control_params(), &NixlEnvelope::Release(token));
            }
            continue;
        }
        let envelope = NixlEnvelope::Packed {
            token,
            rows,
            names: names.to_vec(),
        };
        result = call(
            peer,
            packed_params(Some(slot), seq as i64, token == 0),
            &envelope,
        )
        .map(drop);
    }
    result
}

/// One `transmit_chunk` round trip to `peer`, returning the reply attachment.
fn call(
    peer: SocketAddr,
    params: PTransmitChunkParams,
    envelope: &NixlEnvelope,
) -> Result<Vec<u8>, String> {
    let (body, attachment) = prpc::call_blocking(
        peer,
        SERVICE_NAME,
        methods::TRANSMIT_CHUNK,
        params.encode_to_vec(),
        envelope.encode(),
        RPC_TIMEOUT,
    )
    .map_err(|err| format!("transmit_chunk to {peer}: {err:#}"))?;
    let status = PTransmitChunkResult::decode(body.as_slice())
        .map_err(|err| format!("transmit_chunk reply from {peer}: {err}"))?
        .status
        .ok_or_else(|| format!("transmit_chunk reply from {peer} carries no status"))?;
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
    use super::check_single_visible_device;

    #[test]
    fn one_cn_sees_one_gpu() {
        assert!(check_single_visible_device(None).is_ok());
        assert!(check_single_visible_device(Some(" 3, ")).is_ok());
        assert!(check_single_visible_device(Some("0,1")).is_err());
    }
}
