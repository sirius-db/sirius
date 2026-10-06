//! The NIXL exchange transport.
//!
//! The engine's GPU pool is one `cudaMalloc` slab, registered once with NIXL as VRAM. To ship a
//! parked batch, the sender exports it, asks the receiver (Alloc) for matching buffers from the
//! receiver's own pool, WRITEs every buffer straight into them, and announces the receiver's token
//! with a Packed frame. No staging copy on either side.
//!
//! One thread owns the [`Agent`] (nixl-sys documents a multithreading deadlock caveat), so every
//! WRITE goes through it. What peers ask of this CN (Md, Alloc, Release) never touches that thread:
//! it reads the cached metadata or calls the [`DirectExchange`] on the brpc thread.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

use crate::fragment_executor::{ExportedBatch, FragmentExecutor, SenderSlot};
use crate::nixl_chunk::{AllocReply, NixlEndpoint, NixlEnvelope, control_params, packed_params};
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

/// WRITEs kept in flight per hop unless `SIRIUS_CN_NIXL_WINDOW` says otherwise.
const DEFAULT_WINDOW: u64 = 4;
/// Longest a WRITE may take unless `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS` says otherwise.
const DEFAULT_XFER_TIMEOUT_SECS: u64 = 30;
/// Alloc refusals for want of pool memory are retried this many times, backing off up to
/// [`ALLOC_MAX_BACKOFF`]: a receiver's pool frees up as it consumes earlier batches.
const ALLOC_RETRIES: u32 = 8;
const ALLOC_MAX_BACKOFF: Duration = Duration::from_secs(1);

/// Handle to the transport thread.
#[derive(Debug)]
pub struct NixlTransport {
    /// Taken on drop to end the thread's loop.
    requests: Option<Sender<SendRequest>>,
    thread: Option<JoinHandle<()>>,
    local_md: Vec<u8>,
    exchange: Arc<DirectExchange>,
}

struct SendRequest {
    peer: SocketAddr,
    slot: SenderSlot,
    names: Vec<String>,
    executor: Arc<dyn FragmentExecutor>,
    respond: Sender<Result<(), String>>,
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
        let (respond, response) = channel();
        self.requests
            .as_ref()
            .ok_or_else(|| "the nixl transport is shutting down".to_string())?
            .send(SendRequest {
                peer,
                slot,
                names,
                executor,
                respond,
            })
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
    /// WRITEs kept in flight per hop (`SIRIUS_CN_NIXL_WINDOW`).
    window: usize,
    /// Longest a WRITE may take (`SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS`).
    xfer_timeout: Duration,
    /// WRITEs that failed or timed out, held until the NIC is done with them.
    quarantine: RefCell<Vec<Quarantined>>,
    /// Agent name per peer whose metadata a failed hop dropped while the quarantine still held a
    /// WRITE in progress to it, which can only report success while that metadata stays loaded.
    /// It is invalidated once no such WRITE remains.
    stale_peers: HashMap<SocketAddr, String>,
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

#[derive(Default)]
struct HopTotals {
    frames: u64,
    rows: u64,
    bytes: u64,
    write_us: u64,
}

impl Transport {
    fn bring_up(agent_name: &str, exchange: Arc<DirectExchange>) -> Result<Self, String> {
        // A stub nixl-sys (built without NIXL_PREFIX) aborts the process at its first call when
        // libnixl_capi.so can't be loaded, so refuse it here with the hint instead.
        if nixl_sys::is_stub() {
            return Err(format!(
                "nixl-sys was built as a stub ({ENV_HINT}); if it was built before sourcing \
                 cn-env.sh, run `cargo clean -p nixl-sys` and rebuild"
            ));
        }
        check_single_visible_device(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
        let window = positive_env("SIRIUS_CN_NIXL_WINDOW")?.unwrap_or(DEFAULT_WINDOW) as usize;
        let xfer_timeout = Duration::from_secs(
            positive_env("SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS")?.unwrap_or(DEFAULT_XFER_TIMEOUT_SECS),
        );
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
            device,
            base,
            len,
            window,
            xfer_timeout_s = xfer_timeout.as_secs(),
            "nixl transport ready; GPU slab registered"
        );
        Ok(Self {
            _registration: registration,
            exchange,
            agent,
            device: device as u64,
            local_md,
            peers: HashMap::new(),
            window,
            xfer_timeout,
            quarantine: RefCell::new(Vec::new()),
            stale_peers: HashMap::new(),
        })
    }

    fn serve(&mut self, requests: Receiver<SendRequest>) {
        for request in requests {
            let result = self.send(&request);
            let _ = request.respond.send(result);
        }
        info!("nixl-transport thread shutting down");
    }

    /// Ships every batch parked under the request's slot, keeping a window of WRITEs in flight.
    /// Announces leave in order from their own thread, so a WRITE never waits on the previous
    /// batch's announce round trip.
    fn send(&mut self, request: &SendRequest) -> Result<(), String> {
        self.reap_quarantine();
        let started = Instant::now();
        let peer = request.peer;
        let remote = self.remote_agent(peer)?;
        let window = self.window;
        let mut totals = HopTotals::default();
        // Set by the announcer once the peer refused or missed a frame: the pump then stops
        // exporting, allocating and writing batches nobody will receive.
        let stop = AtomicBool::new(false);
        let shipped = std::thread::scope(|scope| {
            let (announce, frames) = channel::<(u64, u64)>();
            let (slot, names) = (request.slot, &request.names);
            let stop_ref = &stop;
            let announcer = std::thread::Builder::new()
                .name("nixl-announce".to_string())
                .spawn_scoped(scope, move || {
                    announce_all(peer, slot, names, frames, stop_ref)
                })
                .map_err(|err| format!("failed to spawn the nixl announce thread: {err}"))?;
            let mut inflight = VecDeque::new();
            let pumped = self.pump(
                request,
                &remote,
                &mut inflight,
                &announce,
                &stop,
                &mut totals,
            );
            // Wait out every WRITE still posted: a finished one frees the sender's batch, and the
            // receiver's buffers it filled are never announced, so they are released. Once the
            // peer stops answering, further Releases would each wait out `RPC_TIMEOUT`.
            let mut peer_up = true;
            for mut write in inflight {
                if self.complete(&remote, &mut write).is_err() {
                    self.quarantine(peer, write);
                } else if peer_up {
                    peer_up = release_remote(peer, write.remote);
                }
            }
            if pumped.is_ok() {
                let _ = announce.send((0, 0));
            }
            drop(announce);
            let announced = announcer
                .join()
                .unwrap_or_else(|_| Err("the announce thread panicked".to_string()));
            // The announcer's error is the cause when it stopped the pump.
            announced.and(pumped)
        });
        if let Err(err) = shipped {
            // A restarted peer comes back under the same agent name with a new registration, so
            // stale metadata would fail every later hop: reload it on the next one.
            self.forget_peer(peer, &remote);
            return Err(err);
        }
        info!(
            peer = %peer,
            dest_stream = request.slot.node_id,
            sender_id = request.slot.sender_id,
            frames = totals.frames,
            bytes = totals.bytes,
            rows = totals.rows,
            window,
            write_us = totals.write_us,
            span_us = started.elapsed().as_micros() as u64,
            "shipping packed exchange hop"
        );
        Ok(())
    }

    fn pump(
        &self,
        request: &SendRequest,
        remote: &str,
        inflight: &mut VecDeque<Write>,
        announce: &Sender<(u64, u64)>,
        stop: &AtomicBool,
        totals: &mut HopTotals,
    ) -> Result<(), String> {
        let mut drained = false;
        loop {
            if stop.load(Ordering::Acquire) {
                return Err("the peer stopped accepting this hop's frames".to_string());
            }
            while !drained && inflight.len() < self.window {
                match request.executor.export_direct_next(request.slot)? {
                    Some(batch) => inflight.push_back(self.post(request.peer, remote, batch)?),
                    None => drained = true,
                }
            }
            let Some(mut write) = inflight.pop_front() else {
                return Ok(());
            };
            if let Err(err) = self.complete(remote, &mut write) {
                self.quarantine(request.peer, write);
                return Err(err);
            }
            totals.frames += 1;
            totals.rows += write.rows;
            totals.bytes += write.bytes;
            totals.write_us += write.posted.elapsed().as_micros() as u64;
            let _ = announce.send((write.remote, write.rows));
        }
    }

    /// Allocates the receiver's buffers for `batch` and posts one WRITE of all of them.
    fn post(&self, peer: SocketAddr, remote: &str, batch: ExportedBatch) -> Result<Write, String> {
        let reply = alloc(peer, batch.layout)
            .and_then(|reply| AllocReply::decode(&reply))
            .inspect_err(|_| self.release(batch.token))?;
        let request = self
            .create_write(remote, &batch.src, &reply)
            .inspect_err(|_| {
                self.release(batch.token);
                release_remote(peer, reply.token);
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

    /// Waits for `write` to finish, then frees the sender's batch. A WRITE that fails or times out
    /// may still be in flight, so its request stays in `write` (dropping it would release it) and
    /// both batches stay held; the caller quarantines it.
    fn complete(&self, remote: &str, write: &mut Write) -> Result<(), String> {
        let timeout = self.xfer_timeout;
        // Polls back off from a few microseconds to 100 us, so a long WRITE doesn't keep a core
        // spinning next to the brpc and engine threads. WRITEs take 0.3-4 ms at SF1000, so a
        // longer cap would notice one finishing that much later.
        let mut backoff = Duration::from_micros(4);
        while let Some(request) = &write.request {
            match self.agent.get_xfer_status(request) {
                Ok(XferStatus::Success) => write.request = None,
                Ok(XferStatus::InProgress) if write.posted.elapsed() < timeout => {
                    std::thread::sleep(backoff);
                    backoff = (backoff * 2).min(Duration::from_micros(100));
                }
                status => {
                    let bytes = write.bytes;
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
        // NIXL took the metadata over the copy still loaded, so the peer did not restart: there is
        // nothing stale left to invalidate.
        self.stale_peers.remove(&peer);
        self.peers.insert(peer, name.clone());
        Ok(name)
    }

    /// Drops `peer`'s cached metadata, so the next hop to it loads it again, and invalidates it in
    /// NIXL. While the quarantine holds a WRITE still in progress to `peer`, the invalidation waits
    /// for [`reap_quarantine`](Self::reap_quarantine): NIXL fails the status of a WRITE whose peer
    /// was invalidated, so it could never be reclaimed. Reloading meanwhile is harmless: NIXL takes
    /// the same peer's metadata again and refuses a restarted one's until the old is invalidated.
    fn forget_peer(&mut self, peer: SocketAddr, remote: &str) {
        if self.peers.remove(&peer).is_none() {
            return;
        }
        if self.writing_to(peer) {
            info!(%peer, remote, "invalidating a peer's nixl metadata once its quarantined WRITEs end");
            self.stale_peers.insert(peer, remote.to_string());
        } else {
            self.invalidate_peer(peer, remote);
        }
    }

    /// Whether the quarantine holds a WRITE to `peer` that NIXL still reports in progress.
    fn writing_to(&self, peer: SocketAddr) -> bool {
        self.quarantine.borrow().iter().any(|held| {
            held.peer == peer
                && matches!(
                    self.agent.get_xfer_status(&held.request),
                    Ok(XferStatus::InProgress)
                )
        })
    }

    fn invalidate_peer(&self, peer: SocketAddr, remote: &str) {
        if let Err(err) = self.agent.invalidate_remote_md(remote) {
            warn!(%peer, remote, error = %err, "failed to invalidate a peer's nixl metadata");
        }
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
    /// and the receiver's buffers on the peer. Any other status keeps it held. Then invalidates
    /// each stale peer that no held WRITE is still in progress to.
    fn reap_quarantine(&mut self) {
        let mut quarantine = self.quarantine.borrow_mut();
        let held = quarantine.len();
        // Peers that did not answer a Release this pass: each further one would wait out
        // `RPC_TIMEOUT`.
        let mut down = HashSet::new();
        quarantine.retain(|held| {
            if !matches!(
                self.agent.get_xfer_status(&held.request),
                Ok(XferStatus::Success)
            ) {
                return true;
            }
            self.release(held.local);
            if !down.contains(&held.peer) && !release_remote(held.peer, held.remote) {
                down.insert(held.peer);
            }
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
        drop(quarantine);
        let stale: Vec<_> = self
            .stale_peers
            .iter()
            .filter(|&(&peer, _)| !self.writing_to(peer))
            .map(|(&peer, remote)| (peer, remote.clone()))
            .collect();
        for (peer, remote) in stale {
            self.stale_peers.remove(&peer);
            // A hop may have reloaded it meanwhile; the next one loads it again.
            self.peers.remove(&peer);
            self.invalidate_peer(peer, &remote);
        }
    }
}

/// Sends `(token, rows)` frames in order, then EOS (token 0). After the first failure it sets
/// `stop` and announces nothing more: the failed frame's and later frames' receive buffers are
/// released instead, until the peer stops answering.
fn announce_all(
    peer: SocketAddr,
    slot: SenderSlot,
    names: &[String],
    frames: Receiver<(u64, u64)>,
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut result = Ok(());
    let mut peer_up = true;
    for (seq, (token, rows)) in frames.into_iter().enumerate() {
        if result.is_err() {
            if token != 0 && peer_up {
                peer_up = release_remote(peer, token);
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
        if result.is_err() {
            stop.store(true, Ordering::Release);
            // The peer may never have taken this frame. Releasing it is a no-op if it did (tokens
            // are never reused), and a failed hop sends no EOS, so its receiver never runs.
            if token != 0 {
                peer_up = release_remote(peer, token);
            }
        }
    }
    result
}

/// Asks `peer` to free the receive buffers under `token`, logging a failure. Returns whether the
/// peer answered: callers stop sending it Releases once it doesn't, since each would wait out
/// `RPC_TIMEOUT` for nothing.
fn release_remote(peer: SocketAddr, token: u64) -> bool {
    match call(peer, control_params(), &NixlEnvelope::Release(token)) {
        Ok(_) => true,
        Err(error) => {
            warn!(%peer, token, %error, "failed to release receive buffers on a peer");
            false
        }
    }
}

/// Asks `peer` for receive buffers matching `layout`, retrying while its pool is momentarily too
/// full: it frees up as the receiver consumes earlier batches.
fn alloc(peer: SocketAddr, layout: Vec<u8>) -> Result<Vec<u8>, String> {
    let envelope = NixlEnvelope::Alloc(layout);
    let mut backoff = Duration::from_millis(20);
    for _ in 0..ALLOC_RETRIES {
        match call(peer, control_params(), &envelope) {
            Err(err) if is_pool_exhausted(&err) => {
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(ALLOC_MAX_BACKOFF);
            }
            other => return other,
        }
    }
    call(peer, control_params(), &envelope)
}

/// Whether a refused Alloc ran into a full receive pool, as the direct exchange reports it.
fn is_pool_exhausted(error: &str) -> bool {
    error.contains("refused transmit_chunk") && error.contains("bytes requested")
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

/// A positive integer from the environment: `None` when unset, an error when not one.
fn positive_env(name: &str) -> Result<Option<u64>, String> {
    parse_positive(name, std::env::var(name).ok().as_deref())
}

fn parse_positive(name: &str, value: Option<&str>) -> Result<Option<u64>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    value
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|&n| n > 0)
        .map(Some)
        .ok_or_else(|| format!("{name}={value:?} must be a positive integer"))
}

/// One CN per GPU: cross-process `cuda_ipc` is only proven with each CN seeing its own device, so
/// CUDA_VISIBLE_DEVICES must name exactly one.
fn check_single_visible_device(visible: Option<&str>) -> Result<(), String> {
    let Some(visible) = visible else {
        return Err("CUDA_VISIBLE_DEVICES is unset; pin each CN to one GPU".to_string());
    };
    let devices = visible.split(',').filter(|d| !d.trim().is_empty()).count();
    if devices != 1 {
        return Err(format!(
            "CUDA_VISIBLE_DEVICES={visible:?} names {devices} devices; pin each CN to one GPU"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_single_visible_device, is_pool_exhausted, parse_positive};

    #[test]
    fn one_cn_sees_one_gpu() {
        assert!(check_single_visible_device(Some(" 3, ")).is_ok());
        assert!(check_single_visible_device(None).is_err());
        assert!(check_single_visible_device(Some("")).is_err());
        assert!(check_single_visible_device(Some("0,1")).is_err());
    }

    #[test]
    fn tunables_reject_garbage_and_zero() {
        assert_eq!(parse_positive("X", None), Ok(None));
        assert_eq!(parse_positive("X", Some(" 8 ")), Ok(Some(8)));
        assert!(parse_positive("X", Some("0")).is_err());
        assert!(parse_positive("X", Some("four")).is_err());
    }

    #[test]
    fn only_a_full_pool_is_retried() {
        assert!(is_pool_exhausted(
            "127.0.0.1:8060 refused transmit_chunk: failed to allocate receive buffers: direct \
             exchange: 645722880 bytes requested, 441915129 available"
        ));
        assert!(!is_pool_exhausted(
            "127.0.0.1:8060 refused transmit_chunk: direct exchange: already closed"
        ));
        assert!(!is_pool_exhausted(
            "transmit_chunk to 127.0.0.1:8060: connection refused"
        ));
    }
}
