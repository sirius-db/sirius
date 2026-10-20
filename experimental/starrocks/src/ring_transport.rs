//! The ring exchange transport.
//!
//! Each CN registers two small `cudaMalloc` rings with NIXL, a send ring and a receive ring of
//! fixed-size frames, and nothing else: the engine's GPU pool stays an async pool that can trim
//! itself on OOM. A batch leaves as a stream of frames (see [`crate::ring_wire`] and the engine's
//! `exchange_ring_wire.hpp`):
//!
//! 1. **Lane setup.** The first time a hop ships to a peer, a `RingOpen` RPC opens a lane: the
//!    receiver grants it standing credits, at least two receive frames (I3), and learns the
//!    sender's agent so it can send credits back.
//! 2. **Frames.** The sender copies the next frame of the batch's stream into a send frame
//!    (`stage`), then WRITEs it into a credited receive frame with a `Frame` notification. Frame 0
//!    carries the batch's header. There is no RPC per batch.
//! 3. **Landing.** On frame 0 the receiver picks where the batch lands without waiting: GPU within
//!    its landing budget, else host, else disk (I1). Each frame is copied out as it arrives; once
//!    the copy is done the frame goes back to the sender as a credit (I2). The last copy seals the
//!    batch into the exchange inbox's token table, and the batch is delivered to its receiver
//!    like a Packed frame, before its last credit leaves.
//! 4. **End.** Once its source ended and every credit is back, the sender closes the lane and
//!    sends EOS over `transmit_chunk`, as before. Failures stay failure frames.
//!
//! One thread owns the NIXL agent and runs one non-blocking loop over every lane, sending and
//! receiving: a lane without credits parks while the loop serves the others. A lane waits for
//! credits, a grant or a peer only until its query's deadline (I4). The only failure a receiver
//! reports for lack of memory is "no tier has room", with each tier's numbers (I5).

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    Receiver, Sender, SyncSender, TryRecvError, TrySendError, channel, sync_channel,
};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::fragment_executor::{FragmentExecutor, OutputDrain, RingJob, RingNext, SenderSlot};
use crate::nixl_chunk::{
    NixlEndpoint, NixlEnvelope, RingDelivery, StreamHop, control_params, packed_params,
};
use crate::proto::starrocks::PTransmitChunkParams;
use crate::ring_wire::{LaneGrant, LaneOpen, RingNote};

/// Standing credits a lane asks for; it opens with at least [`MIN_CREDITS`] (I3).
const DEFAULT_CREDITS: usize = 4;
pub(crate) const MIN_CREDITS: usize = 2;
/// Jobs a lane's feeder exports ahead of the frames in flight.
const FEED_AHEAD: usize = 2;
/// How long a feeder waits on a streamed output before checking whether its lane failed.
const FEED_WAIT: Duration = Duration::from_millis(20);
/// Waits for a credit, a grant or a peer are bounded by the query's deadline; a hop that has none
/// waits at most this long (`SIRIUS_CN_RING_MAX_WAIT_SECS`).
const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(1800);
/// A busy receiver is asked again for a lane after this long, doubling up to a second.
const OPEN_RETRY_FIRST: Duration = Duration::from_millis(2);
const OPEN_RETRY_MAX: Duration = Duration::from_secs(1);
/// The loop sleeps this long, doubling up to [`IDLE_MAX`], when a pass made no progress.
const IDLE_FIRST: Duration = Duration::from_micros(4);
const IDLE_MAX: Duration = Duration::from_micros(100);
/// A feeder whose export ran out of GPU memory asks again after this long, doubling up to
/// [`OOM_RETRY_MAX`].
const OOM_RETRY_FIRST: Duration = Duration::from_millis(10);
const OOM_RETRY_MAX: Duration = Duration::from_secs(1);
/// How often the loop logs the engine's ring counters, when they changed.
const STATS_EVERY: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Landing {
    pub id: u64,
    pub rows: u64,
    pub frames: u32,
}

/// The engine's ends of the ring: frames, copies in and out, and the token table batches land in.
pub(crate) trait RingEngine: Send + Sync + std::fmt::Debug {
    fn frame_bytes(&self) -> u64;
    /// `(device, base, length)` of the receive ring.
    fn recv_region(&self) -> (i32, usize, u64);
    fn lease_send(&self) -> Result<Option<u32>, String>;
    fn release_send(&self, frame: u32) -> Result<(), String>;
    fn lease_recv(&self) -> Result<Option<u32>, String>;
    fn release_recv(&self, frame: u32) -> Result<(), String>;
    fn stage(&self, token: u64, k: u32, frame: u32) -> Result<u64, String>;
    fn release_job(&self, token: u64);
    fn begin(&self, header: &[u8]) -> Result<Landing, String>;
    fn copy_out(&self, landing: u64, k: u32, frame: u32, len: u64) -> Result<u64, String>;
    fn seal(&self, landing: u64) -> Result<u64, String>;
    fn abort(&self, landing: u64);
    fn done(&self, ticket: u64) -> Result<bool, String>;
    /// Frees an exchange-inbox token: a landed batch that will not be pushed.
    fn release_token(&self, token: u64);
    fn outstanding(&self) -> usize;
    fn stats_line(&self) -> String;
}

pub(crate) type WriteId = u64;

/// NIXL as the loop uses it. Owned by the transport thread.
pub(crate) trait Fabric {
    fn local_md(&self) -> Vec<u8>;
    fn local_name(&self) -> String;
    /// Loads a peer's metadata; returns its agent name.
    fn load_peer(&mut self, md: &[u8]) -> Result<String, String>;
    /// WRITEs `len` bytes of send frame `frame` to `dst` on `agent`'s device `device`, with
    /// notification `note` once the data is there.
    fn write(
        &mut self,
        agent: &str,
        frame: u32,
        dst: u64,
        device: i32,
        len: u64,
        note: &str,
    ) -> Result<WriteId, String>;
    /// Whether a WRITE finished; an error leaves it posted (the NIC may still be writing).
    fn poll_write(&mut self, id: WriteId) -> Result<bool, String>;
    fn forget_write(&mut self, id: WriteId);
    fn notify(&mut self, agent: &str, note: &str) -> Result<(), String>;
    /// Notifications received since the last call, by sending agent, in arrival order.
    fn take_notes(&mut self) -> Result<Vec<(String, String)>, String>;
}

/// `transmit_chunk` round trips to peers. Blocking; called off the transport thread.
pub(crate) trait Control: Send + Sync {
    fn call(
        &self,
        peer: SocketAddr,
        params: PTransmitChunkParams,
        envelope: &NixlEnvelope,
    ) -> Result<Vec<u8>, String>;
    /// Sends each hop's receiver a failure frame in place of its EOS.
    fn fail(&self, ends: Vec<(SocketAddr, SenderSlot)>, error: &str);
}

#[derive(Debug)]
struct RecvLane {
    /// The sender's agent, which credits go to.
    agent: String,
    /// Routing of the sender's frames: receiver fragment instance, exchange node, sender id.
    params: PTransmitChunkParams,
    names: Vec<String>,
    /// Receive frames this lane holds: its standing credits.
    frames: Vec<u32>,
    /// Sequence number of the next batch delivered.
    next_seq: i64,
    /// The lane failed here; its frames wait for the sender to close it.
    refused: bool,
    /// The sender closed it: its frames go back once no copy-out reads them.
    closing: bool,
}

/// Receive lanes, shared by the RPC threads that open and close them and the transport thread
/// that lands their frames.
#[derive(Debug, Default)]
pub(crate) struct RecvLanes {
    next: u64,
    lanes: HashMap<u64, RecvLane>,
}

impl RecvLanes {
    /// Opens a lane with up to `want` receive frames and at least [`MIN_CREDITS`]. Returns the
    /// lane and its frames, or `None` when too few frames are free: the sender asks again later.
    fn open(
        &mut self,
        engine: &dyn RingEngine,
        agent: String,
        params: PTransmitChunkParams,
        names: Vec<String>,
        want: usize,
    ) -> Result<Option<(u64, Vec<u32>)>, String> {
        let mut frames = Vec::with_capacity(want);
        while frames.len() < want.max(MIN_CREDITS) {
            match engine.lease_recv()? {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }
        if frames.len() < MIN_CREDITS {
            for frame in frames {
                engine.release_recv(frame)?;
            }
            return Ok(None);
        }
        self.next += 1;
        let lane = self.next;
        self.lanes.insert(
            lane,
            RecvLane {
                agent,
                params,
                names,
                frames: frames.clone(),
                next_seq: 0,
                refused: false,
                closing: false,
            },
        );
        Ok(Some((lane, frames)))
    }

    /// Marks `lane` closing; [`finish_close`](Self::finish_close) returns its frames.
    fn close(&mut self, lane: u64) {
        if let Some(open) = self.lanes.get_mut(&lane) {
            open.closing = true;
        }
    }

    /// Returns the frames of every closing lane that `busy` says no copy-out reads.
    fn finish_close(&mut self, engine: &dyn RingEngine, busy: impl Fn(u64) -> bool) {
        let done: Vec<u64> = self
            .lanes
            .iter()
            .filter(|(lane, open)| open.closing && !busy(**lane))
            .map(|(lane, _)| *lane)
            .collect();
        for lane in done {
            if let Some(open) = self.lanes.remove(&lane) {
                for frame in open.frames {
                    if let Err(err) = engine.release_recv(frame) {
                        warn!(lane, frame, error = %err, "failed to return a ring frame");
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RingSettings {
    pub credits: usize,
    /// Longest wait for a credit, grant or peer when a hop has no deadline.
    pub max_wait: Duration,
    pub xfer_timeout: Duration,
}

impl RingSettings {
    pub(crate) fn from_env() -> Result<Self, String> {
        let positive = |name: &str| match std::env::var(name).ok() {
            None => Ok(None),
            Some(value) => match value.trim().parse::<u64>() {
                Ok(n) if n > 0 => Ok(Some(n)),
                _ => Err(format!("{name} must be a positive integer, not '{value}'")),
            },
        };
        Ok(Self {
            credits: positive("SIRIUS_CN_RING_CREDITS")?
                .map_or(DEFAULT_CREDITS, |n| n as usize)
                .max(MIN_CREDITS),
            max_wait: positive("SIRIUS_CN_RING_MAX_WAIT_SECS")?
                .map_or(DEFAULT_MAX_WAIT, Duration::from_secs),
            xfer_timeout: Duration::from_secs(
                positive("SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS")?.unwrap_or(30),
            ),
        })
    }
}

enum Request {
    Ship(Ship),
    /// Load a peer's metadata before answering its `RingOpen`, so credits can reach it.
    LoadPeer {
        md: Vec<u8>,
        respond: Sender<Result<String, String>>,
    },
    Close(u64),
}

struct Ship {
    hops: Vec<OutHop>,
    respond: Sender<Result<(), String>>,
}

struct OutHop {
    peer: SocketAddr,
    slot: SenderSlot,
    names: Vec<String>,
    source: Source,
    deadline: Instant,
}

enum Source {
    Parked(Arc<dyn FragmentExecutor>),
    Streamed(Box<dyn OutputDrain>),
}

impl Source {
    fn next(&mut self, slot: SenderSlot) -> Result<RingNext, String> {
        match self {
            Source::Parked(executor) => Ok(executor
                .export_ring_next(slot)?
                .map_or(RingNext::End, RingNext::Job)),
            Source::Streamed(drain) => drain.next_ring(FEED_WAIT),
        }
    }
}

pub(crate) struct RingTransport {
    requests: Mutex<Option<Sender<Request>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    shared: Arc<Shared>,
}

struct Shared {
    engine: Arc<dyn RingEngine>,
    control: Arc<dyn Control>,
    lanes: Mutex<RecvLanes>,
    delivery: Mutex<Option<Arc<dyn RingDelivery>>>,
    local_md: Vec<u8>,
    local_name: String,
    settings: RingSettings,
}

impl std::fmt::Debug for RingTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingTransport")
            .field("agent", &self.shared.local_name)
            .field("engine", &self.shared.engine)
            .finish()
    }
}

impl RingTransport {
    pub(crate) fn start<F: Fabric + Send + 'static>(
        fabric: F,
        engine: Arc<dyn RingEngine>,
        control: Arc<dyn Control>,
        settings: RingSettings,
    ) -> Result<Self, String> {
        let shared = Arc::new(Shared {
            engine,
            control,
            lanes: Mutex::new(RecvLanes::default()),
            delivery: Mutex::new(None),
            local_md: fabric.local_md(),
            local_name: fabric.local_name(),
            settings,
        });
        let (request_tx, request_rx) = channel();
        let thread_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("ring-transport".to_string())
            .spawn(move || RingLoop::new(fabric, thread_shared).serve(request_rx))
            .map_err(|err| format!("failed to spawn the ring-transport thread: {err}"))?;
        info!(
            agent = %shared.local_name,
            credits = settings.credits,
            max_wait_s = settings.max_wait.as_secs(),
            "ring transport ready"
        );
        Ok(Self {
            requests: Mutex::new(Some(request_tx)),
            thread: Mutex::new(Some(thread)),
            shared,
        })
    }

    fn request(&self, request: Request) -> Result<(), String> {
        self.requests
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| "the ring transport is shutting down".to_string())?
            .send(request)
            .map_err(|_| "the ring-transport thread is not running".to_string())
    }

    fn ship(&self, hops: Vec<OutHop>) -> Result<(), String> {
        let ends: Vec<_> = hops.iter().map(|hop| (hop.peer, hop.slot)).collect();
        let (respond, response) = channel();
        let shipped = self
            .request(Request::Ship(Ship { hops, respond }))
            .and_then(|()| {
                response
                    .recv()
                    .map_err(|_| "the ring-transport thread dropped the hops".to_string())
            });
        match shipped {
            Ok(result) => result,
            Err(err) => {
                self.shared.control.fail(ends, &err);
                Err(err)
            }
        }
    }

    fn deadline(&self, deadline: Option<Instant>) -> Instant {
        deadline.unwrap_or_else(|| Instant::now() + self.shared.settings.max_wait)
    }
}

impl Drop for RingTransport {
    fn drop(&mut self) {
        self.shut_down();
    }
}

impl NixlEndpoint for RingTransport {
    fn local_md(&self) -> Vec<u8> {
        self.shared.local_md.clone()
    }

    fn release(&self, token: u64) {
        self.shared.engine.release_token(token);
    }

    fn outstanding(&self) -> usize {
        self.shared.engine.outstanding()
    }

    fn send(
        &self,
        peer: SocketAddr,
        slot: SenderSlot,
        names: Vec<String>,
        executor: Arc<dyn FragmentExecutor>,
    ) -> Result<(), String> {
        self.ship(vec![OutHop {
            peer,
            slot,
            names,
            source: Source::Parked(executor),
            deadline: self.deadline(None),
        }])
    }

    fn stream(&self, hops: Vec<StreamHop>) -> Result<(), String> {
        let hops = hops
            .into_iter()
            .map(|hop| OutHop {
                peer: hop.peer,
                slot: hop.slot,
                names: hop.names,
                source: Source::Streamed(hop.drain),
                deadline: self.deadline(hop.deadline),
            })
            .collect();
        self.ship(hops)
    }

    fn fail(&self, peer: SocketAddr, slot: SenderSlot, error: &str) {
        self.shared.control.fail(vec![(peer, slot)], error);
    }

    fn ring_open(&self, request: &PTransmitChunkParams, body: &[u8]) -> Result<Vec<u8>, String> {
        let open = LaneOpen::decode(body)?;
        // Before granting: the first credit goes back to this sender as a notification.
        let (respond, loaded) = channel();
        self.request(Request::LoadPeer {
            md: open.md,
            respond,
        })?;
        let agent = loaded
            .recv()
            .map_err(|_| "the ring-transport thread dropped a peer's metadata".to_string())??;
        let engine = self.shared.engine.as_ref();
        let (device, recv_base, _) = engine.recv_region();
        let granted = self.shared.lanes.lock().unwrap().open(
            engine,
            agent,
            request.clone(),
            open.names,
            self.shared.settings.credits,
        )?;
        let (lane, frames) = granted.unwrap_or_default();
        Ok(LaneGrant {
            lane,
            device,
            recv_base: recv_base as u64,
            frame_bytes: engine.frame_bytes(),
            agent: self.shared.local_name.clone(),
            frames,
        }
        .encode())
    }

    fn ring_close(&self, lane: u64) -> Result<(), String> {
        self.request(Request::Close(lane))
    }

    fn set_delivery(&self, delivery: Arc<dyn RingDelivery>) {
        *self.shared.delivery.lock().unwrap() = Some(delivery);
    }

    fn shut_down(&self) {
        self.shared.delivery.lock().unwrap().take();
        self.requests.lock().unwrap().take();
        if let Some(thread) = self.thread.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}

struct OutLane {
    ship: usize,
    peer: SocketAddr,
    slot: SenderSlot,
    names: Vec<String>,
    deadline: Instant,
    state: OutState,
    feed: Option<Receiver<Result<RingJob, String>>>,
    feeder: Option<JoinHandle<()>>,
    stop_feed: Arc<AtomicBool>,
    job: Option<ActiveJob>,
    staged: VecDeque<Staged>,
    writes: VecDeque<Posted>,
    /// Jobs whose frames are all staged, by token: WRITEs still posted.
    finishing: HashMap<u64, usize>,
    credits: VecDeque<u32>,
    granted: usize,
    batches: u64,
    rows: u64,
    bytes: u64,
    /// When the lane started waiting on credits with a frame to send.
    starved_since: Option<Instant>,
    credit_wait: Duration,
    started: Instant,
}

enum OutState {
    Opening(Receiver<Result<(LaneGrant, Vec<u8>), String>>),
    Open(LaneGrant),
    /// Closing the lane and sending EOS, on a thread of its own.
    Closing(Receiver<Result<(), String>>),
    Done,
    Failed(String),
}

struct ActiveJob {
    job: RingJob,
    batch: u64,
    frames: u32,
    next: u32,
}

struct Staged {
    ticket: u64,
    send_frame: u32,
    recv_frame: u32,
    k: u32,
    len: u64,
    token: u64,
    batch: u64,
    header: Vec<u8>,
}

struct Posted {
    id: WriteId,
    send_frame: u32,
    token: u64,
    posted: Instant,
    bytes: u64,
}

struct InLanding {
    landing: Landing,
    copied: u32,
    /// Frames that arrived before the header: `(k, frame, len)`.
    early: Vec<(u32, u32, u64)>,
}

struct CopyOut {
    ticket: u64,
    lane: u64,
    batch: u64,
    frame: u32,
}

struct RingLoop<F: Fabric> {
    fabric: F,
    shared: Arc<Shared>,
    ships: HashMap<usize, Sender<Result<(), String>>>,
    /// Each ship's first error, until all its lanes ended.
    ship_errors: HashMap<usize, String>,
    next_ship: usize,
    lanes: Vec<OutLane>,
    /// Outgoing lanes by the receiver's lane id, for credits and refusals.
    by_grant: HashMap<(String, u64), usize>,
    landings: HashMap<(u64, u64), InLanding>,
    copies: Vec<CopyOut>,
    credits: HashMap<u64, Vec<u32>>,
    /// WRITEs that failed or timed out, with their send frames, until NIXL says they finished.
    quarantine: Vec<(WriteId, u32)>,
}

impl<F: Fabric> RingLoop<F> {
    fn new(fabric: F, shared: Arc<Shared>) -> Self {
        Self {
            fabric,
            shared,
            ships: HashMap::new(),
            ship_errors: HashMap::new(),
            next_ship: 0,
            lanes: Vec::new(),
            by_grant: HashMap::new(),
            landings: HashMap::new(),
            copies: Vec::new(),
            credits: HashMap::new(),
            quarantine: Vec::new(),
        }
    }

    fn engine(&self) -> &dyn RingEngine {
        self.shared.engine.as_ref()
    }

    fn serve(mut self, requests: Receiver<Request>) {
        let mut idle = IDLE_FIRST;
        let mut open = true;
        let mut stats_at = Instant::now();
        let mut last_stats = String::new();
        loop {
            if stats_at.elapsed() >= STATS_EVERY {
                stats_at = Instant::now();
                let stats = self.engine().stats_line();
                if stats != last_stats {
                    info!(stats = %stats, lanes = self.lanes.len(), "ring exchange stats");
                    last_stats = stats;
                }
            }
            let mut progressed = false;
            loop {
                match requests.try_recv() {
                    Ok(request) => {
                        self.handle(request);
                        progressed = true;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        open = false;
                        break;
                    }
                }
            }
            progressed |= self.pass();
            if !open && self.lanes.is_empty() && self.copies.is_empty() {
                break;
            }
            if progressed {
                idle = IDLE_FIRST;
            } else {
                std::thread::sleep(idle);
                idle = (idle * 2).min(IDLE_MAX);
            }
        }
        for (_, landing) in std::mem::take(&mut self.landings) {
            if landing.landing.id != 0 {
                self.engine().abort(landing.landing.id);
            }
        }
        info!(stats = %self.engine().stats_line(), "ring-transport thread shutting down");
    }

    fn handle(&mut self, request: Request) {
        match request {
            Request::Ship(ship) => self.start_ship(ship),
            Request::LoadPeer { md, respond } => {
                let _ = respond.send(self.fabric.load_peer(&md));
            }
            Request::Close(lane) => {
                self.shared.lanes.lock().unwrap().close(lane);
            }
        }
    }

    fn pass(&mut self) -> bool {
        let mut progressed = self.receive();
        progressed |= self.finish_copies();
        progressed |= self.send_credits();
        let busy: HashSet<u64> = self.copies.iter().map(|c| c.lane).collect();
        self.shared
            .lanes
            .lock()
            .unwrap()
            .finish_close(self.shared.engine.as_ref(), |lane| busy.contains(&lane));
        for i in 0..self.lanes.len() {
            progressed |= self.step_lane(i);
        }
        progressed |= self.finish_ships();
        progressed |= self.reap();
        progressed
    }

    fn start_ship(&mut self, ship: Ship) {
        let id = self.next_ship;
        self.next_ship += 1;
        self.ships.insert(id, ship.respond);
        for hop in ship.hops {
            let (peer, slot) = (hop.peer, hop.slot);
            let open = self.open_lane(&hop);
            let stop_feed = Arc::new(AtomicBool::new(false));
            let (feed, feeder) = match spawn_feeder(
                hop.source,
                slot,
                hop.deadline,
                Arc::clone(&stop_feed),
                Arc::clone(&self.shared.engine),
            ) {
                Ok((feed, feeder)) => (Some(feed), Some(feeder)),
                Err(err) => {
                    self.shared.control.fail(vec![(peer, slot)], &err);
                    self.ship_errors.entry(id).or_insert(err);
                    continue;
                }
            };
            self.lanes.push(OutLane {
                ship: id,
                peer,
                slot,
                names: hop.names,
                deadline: hop.deadline,
                state: OutState::Opening(open),
                feed,
                feeder,
                stop_feed,
                job: None,
                staged: VecDeque::new(),
                writes: VecDeque::new(),
                finishing: HashMap::new(),
                credits: VecDeque::new(),
                granted: 0,
                batches: 0,
                rows: 0,
                bytes: 0,
                starved_since: None,
                credit_wait: Duration::ZERO,
                started: Instant::now(),
            });
        }
    }

    /// Asks the receiver for a lane on a thread, retrying a busy one until the deadline; also
    /// fetches its agent metadata.
    fn open_lane(&self, hop: &OutHop) -> Receiver<Result<(LaneGrant, Vec<u8>), String>> {
        let (tx, rx) = channel();
        let control = Arc::clone(&self.shared.control);
        let (peer, slot, deadline) = (hop.peer, hop.slot, hop.deadline);
        let body = LaneOpen {
            md: self.shared.local_md.clone(),
            names: hop.names.clone(),
        }
        .encode();
        let local_md = self.shared.local_md.clone();
        let spawned = std::thread::Builder::new()
            .name("ring-open".to_string())
            .spawn(move || {
                let opened = (|| {
                    let md = control.call(peer, control_params(), &NixlEnvelope::Md(local_md))?;
                    let mut retry = OPEN_RETRY_FIRST;
                    loop {
                        let reply = control.call(
                            peer,
                            packed_params(Some(slot), 0, false),
                            &NixlEnvelope::RingOpen(body.clone()),
                        )?;
                        let grant = LaneGrant::decode(&reply)?;
                        if !grant.frames.is_empty() {
                            return Ok((grant, md));
                        }
                        if Instant::now() + retry > deadline {
                            return Err(format!(
                                "the ring receiver {peer} granted no lane before the query's \
                                 deadline: every receive frame is held by other lanes"
                            ));
                        }
                        std::thread::sleep(retry);
                        retry = (retry * 2).min(OPEN_RETRY_MAX);
                    }
                })();
                let _ = tx.send(opened);
            });
        if let Err(err) = spawned {
            let (tx, failed) = channel();
            let _ = tx.send(Err(format!("failed to spawn a ring-open thread: {err}")));
            return failed;
        }
        rx
    }

    fn step_lane(&mut self, i: usize) -> bool {
        let mut progressed = false;
        match &self.lanes[i].state {
            OutState::Opening(rx) => match rx.try_recv() {
                Ok(Ok((grant, md))) => {
                    match self.fabric.load_peer(&md) {
                        Ok(agent) if agent == grant.agent => {
                            let lane = &mut self.lanes[i];
                            lane.credits = grant.frames.iter().copied().collect();
                            lane.granted = grant.frames.len();
                            self.by_grant.insert((grant.agent.clone(), grant.lane), i);
                            lane.state = OutState::Open(grant);
                        }
                        Ok(agent) => self.fail_lane(
                            i,
                            format!(
                                "ring receiver {} answered as agent '{agent}' but granted lane on \
                                 '{}'",
                                self.lanes[i].peer, grant.agent
                            ),
                        ),
                        Err(err) => self.fail_lane(i, err),
                    }
                    return true;
                }
                Ok(Err(err)) => {
                    self.fail_lane(i, err);
                    return true;
                }
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => {
                    self.fail_lane(i, "the ring-open thread exited".to_string());
                    return true;
                }
            },
            OutState::Closing(rx) => {
                return match rx.try_recv() {
                    Ok(result) => {
                        self.end_lane(i, result);
                        true
                    }
                    Err(TryRecvError::Empty) => false,
                    Err(TryRecvError::Disconnected) => {
                        self.end_lane(i, Err("the ring-close thread exited".to_string()));
                        true
                    }
                };
            }
            OutState::Done => return false,
            OutState::Failed(_) => {
                // Wait out the WRITEs in flight, then end with a failure frame.
                progressed |= self.poll_writes(i);
                if self.lanes[i].writes.is_empty() {
                    let OutState::Failed(err) =
                        std::mem::replace(&mut self.lanes[i].state, OutState::Done)
                    else {
                        unreachable!()
                    };
                    self.end_lane(i, Err(err));
                    progressed = true;
                }
                return progressed;
            }
            OutState::Open(_) => {}
        }

        progressed |= self.pull_job(i);
        match self.stage_frames(i) {
            Ok(staged) => progressed |= staged,
            Err(err) => {
                self.fail_lane(i, err);
                return true;
            }
        }
        match self.post_writes(i) {
            Ok(posted) => progressed |= posted,
            Err(err) => {
                self.fail_lane(i, err);
                return true;
            }
        }
        progressed |= self.poll_writes(i);
        if let OutState::Failed(_) = self.lanes[i].state {
            return true;
        }

        let lane = &mut self.lanes[i];
        // I4: a lane with a frame to send and no credit waits only until its deadline.
        let starved = lane.job.is_some() && lane.credits.is_empty();
        match (starved, lane.starved_since) {
            (true, None) => lane.starved_since = Some(Instant::now()),
            (false, Some(since)) => {
                lane.credit_wait += since.elapsed();
                lane.starved_since = None;
            }
            _ => {}
        }
        if starved && Instant::now() > lane.deadline {
            let err = format!(
                "ring lane to {} got no credit before the query's deadline ({} of {} frames \
                 outstanding)",
                lane.peer,
                lane.granted - lane.credits.len(),
                lane.granted
            );
            self.fail_lane(i, err);
            return true;
        }

        let lane = &self.lanes[i];
        let drained = lane.feed.is_none()
            && lane.job.is_none()
            && lane.staged.is_empty()
            && lane.writes.is_empty()
            && lane.finishing.is_empty();
        if drained && lane.credits.len() == lane.granted {
            self.close_lane(i);
            return true;
        }
        if drained && Instant::now() > lane.deadline {
            let err = format!(
                "ring lane to {} did not get its last {} credits back before the query's \
                 deadline",
                lane.peer,
                lane.granted - lane.credits.len()
            );
            self.fail_lane(i, err);
            return true;
        }
        progressed
    }

    fn pull_job(&mut self, i: usize) -> bool {
        let frame_bytes = self.engine().frame_bytes();
        let lane = &mut self.lanes[i];
        if lane.job.is_some() {
            return false;
        }
        let Some(feed) = &lane.feed else {
            return false;
        };
        match feed.try_recv() {
            Ok(Ok(job)) => {
                let frames = job.stream_bytes.div_ceil(frame_bytes) as u32;
                let batch = lane.batches + lane.finishing.len() as u64;
                lane.rows += job.rows;
                lane.bytes += job.stream_bytes;
                lane.job = Some(ActiveJob {
                    job,
                    batch,
                    frames,
                    next: 0,
                });
                true
            }
            Ok(Err(err)) => {
                self.fail_lane(i, err);
                true
            }
            Err(TryRecvError::Empty) => false,
            Err(TryRecvError::Disconnected) => {
                lane.feed = None;
                if let Some(feeder) = lane.feeder.take() {
                    let _ = feeder.join();
                }
                true
            }
        }
    }

    fn stage_frames(&mut self, i: usize) -> Result<bool, String> {
        let engine = Arc::clone(&self.shared.engine);
        let lane = &mut self.lanes[i];
        let mut progressed = false;
        while let Some(active) = &mut lane.job {
            if lane.credits.is_empty() {
                break;
            }
            let Some(send_frame) = engine.lease_send()? else {
                break;
            };
            let recv_frame = lane.credits.pop_front().expect("a credit");
            let k = active.next;
            let frame_bytes = engine.frame_bytes();
            let len = (active.job.stream_bytes - u64::from(k) * frame_bytes).min(frame_bytes);
            let ticket = match engine.stage(active.job.token, k, send_frame) {
                Ok(ticket) => ticket,
                Err(err) => {
                    let _ = engine.release_send(send_frame);
                    lane.credits.push_front(recv_frame);
                    return Err(err);
                }
            };
            lane.staged.push_back(Staged {
                ticket,
                send_frame,
                recv_frame,
                k,
                len,
                token: active.job.token,
                batch: active.batch,
                header: if k == 0 {
                    active.job.header.clone()
                } else {
                    Vec::new()
                },
            });
            *lane.finishing.entry(active.job.token).or_insert(0) += 1;
            active.next += 1;
            progressed = true;
            if active.next == active.frames {
                lane.job = None;
            }
        }
        Ok(progressed)
    }

    fn post_writes(&mut self, i: usize) -> Result<bool, String> {
        let OutState::Open(grant) = &self.lanes[i].state else {
            return Ok(false);
        };
        let grant = grant.clone();
        let mut progressed = false;
        while let Some(staged) = self.lanes[i].staged.front() {
            if !self.shared.engine.done(staged.ticket)? {
                break;
            }
            let staged = self.lanes[i].staged.pop_front().expect("a staged frame");
            let note = RingNote::Frame {
                lane: grant.lane,
                batch: staged.batch,
                k: staged.k,
                frame: staged.recv_frame,
                len: staged.len,
                header: staged.header,
            }
            .encode();
            let dst = grant.recv_base + u64::from(staged.recv_frame) * grant.frame_bytes;
            let id = match self.fabric.write(
                &grant.agent,
                staged.send_frame,
                dst,
                grant.device,
                staged.len,
                &note,
            ) {
                Ok(id) => id,
                Err(err) => {
                    let _ = self.shared.engine.release_send(staged.send_frame);
                    return Err(err);
                }
            };
            self.lanes[i].writes.push_back(Posted {
                id,
                send_frame: staged.send_frame,
                token: staged.token,
                posted: Instant::now(),
                bytes: staged.len,
            });
            progressed = true;
        }
        Ok(progressed)
    }

    /// Retires finished WRITEs: frees their send frames and, with a job's last one, the job. A
    /// WRITE that failed or timed out may still be reading its send frame, which stays in
    /// quarantine until NIXL says it finished; its job is released, since the frame holds a copy.
    fn poll_writes(&mut self, i: usize) -> bool {
        let timeout = self.shared.settings.xfer_timeout;
        let mut progressed = false;
        let mut failure = None;
        let mut k = 0;
        while k < self.lanes[i].writes.len() {
            let write = &self.lanes[i].writes[k];
            let status = self.fabric.poll_write(write.id);
            let ok = match status {
                Ok(true) => true,
                Ok(false) if write.posted.elapsed() < timeout => {
                    k += 1;
                    continue;
                }
                _ => false,
            };
            let posted = self.lanes[i].writes.remove(k).expect("a write");
            progressed = true;
            if ok {
                self.fabric.forget_write(posted.id);
                if let Err(err) = self.shared.engine.release_send(posted.send_frame) {
                    warn!(frame = posted.send_frame, error = %err, "failed to return a send frame");
                }
            } else {
                self.quarantine.push((posted.id, posted.send_frame));
                failure.get_or_insert_with(|| match status {
                    Err(err) => format!("a {}-byte ring WRITE failed: {err}", posted.bytes),
                    _ => format!(
                        "a {}-byte ring WRITE did not finish within {timeout:?} \
                         (SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS)",
                        posted.bytes
                    ),
                });
            }
            let lane = &mut self.lanes[i];
            let Some(left) = lane.finishing.get_mut(&posted.token) else {
                continue;
            };
            *left -= 1;
            let staging = lane
                .job
                .as_ref()
                .is_some_and(|active| active.job.token == posted.token);
            if *left == 0 && !staging {
                lane.finishing.remove(&posted.token);
                if ok {
                    lane.batches += 1;
                }
                self.shared.engine.release_job(posted.token);
            }
        }
        if let Some(err) = failure {
            self.fail_lane(i, err);
        }
        progressed
    }

    fn close_lane(&mut self, i: usize) {
        let OutState::Open(grant) = &self.lanes[i].state else {
            return;
        };
        let lane_id = grant.lane;
        let lane = &mut self.lanes[i];
        let (tx, rx) = channel();
        let control = Arc::clone(&self.shared.control);
        let (peer, slot, names, batches) = (lane.peer, lane.slot, lane.names.clone(), lane.batches);
        let spawned = std::thread::Builder::new()
            .name("ring-close".to_string())
            .spawn(move || {
                let closed = control
                    .call(peer, control_params(), &NixlEnvelope::RingClose(lane_id))
                    .and_then(|_| {
                        control.call(
                            peer,
                            packed_params(Some(slot), batches as i64, true),
                            &NixlEnvelope::Packed {
                                token: 0,
                                rows: 0,
                                names,
                            },
                        )
                    })
                    .map(drop);
                let _ = tx.send(closed);
            });
        lane.state = match spawned {
            Ok(_) => OutState::Closing(rx),
            Err(err) => OutState::Failed(format!("failed to spawn a ring-close thread: {err}")),
        };
    }

    /// Stops a lane: no more frames; its WRITEs in flight finish before it ends.
    fn fail_lane(&mut self, i: usize, err: String) {
        let engine = Arc::clone(&self.shared.engine);
        let lane = &mut self.lanes[i];
        if matches!(lane.state, OutState::Failed(_) | OutState::Done) {
            return;
        }
        warn!(peer = %lane.peer, slot = ?lane.slot, error = %err, "ring lane failed");
        if let OutState::Open(grant) = &lane.state {
            // Best effort: the receiver takes its frames back.
            let (control, peer, id) = (Arc::clone(&self.shared.control), lane.peer, grant.lane);
            let _ = std::thread::Builder::new()
                .name("ring-close".to_string())
                .spawn(move || {
                    let _ = control.call(peer, control_params(), &NixlEnvelope::RingClose(id));
                });
        }
        lane.stop_feed.store(true, Ordering::Release);
        for staged in std::mem::take(&mut lane.staged) {
            // The copy may still run: wait for it before the frame is reused.
            while !matches!(engine.done(staged.ticket), Ok(true) | Err(_)) {
                std::thread::yield_now();
            }
            let _ = engine.release_send(staged.send_frame);
        }
        if let Some(active) = lane.job.take()
            && !lane.finishing.contains_key(&active.job.token)
        {
            engine.release_job(active.job.token);
        }
        // Jobs with WRITEs in flight are released as they finish; the rest now.
        let in_flight: HashSet<u64> = lane.writes.iter().map(|w| w.token).collect();
        lane.finishing.retain(|token, _| {
            let keep = in_flight.contains(token);
            if !keep {
                engine.release_job(*token);
            }
            keep
        });
        if let Some(feed) = lane.feed.take() {
            while let Ok(Ok(job)) = feed.try_recv() {
                engine.release_job(job.token);
            }
            drop(feed);
        }
        lane.state = OutState::Failed(err);
    }

    /// Ends a lane: on an error, its receiver gets a failure frame. Its ship is reported once
    /// every lane of it ended ([`finish_ships`](Self::finish_ships)).
    fn end_lane(&mut self, i: usize, result: Result<(), String>) {
        let lane = &mut self.lanes[i];
        if let Err(err) = &result {
            self.shared.control.fail(vec![(lane.peer, lane.slot)], err);
        }
        lane.stop_feed.store(true, Ordering::Release);
        lane.feed.take();
        if let Some(feeder) = lane.feeder.take() {
            let _ = feeder.join();
        }
        let credit_wait = lane.credit_wait
            + lane
                .starved_since
                .map_or(Duration::ZERO, |since| since.elapsed());
        info!(
            peer = %lane.peer,
            dest_stream = lane.slot.node_id,
            sender_id = lane.slot.sender_id,
            batches = lane.batches,
            rows = lane.rows,
            bytes = lane.bytes,
            credits = lane.granted,
            credit_wait_us = credit_wait.as_micros() as u64,
            span_us = lane.started.elapsed().as_micros() as u64,
            ok = result.is_ok(),
            "shipped ring exchange lane"
        );
        lane.state = OutState::Done;
        if let Err(err) = result {
            self.ship_errors.entry(lane.ship).or_insert(err);
        }
    }

    fn finish_ships(&mut self) -> bool {
        let ended: Vec<usize> = self
            .ships
            .keys()
            .copied()
            .filter(|&ship| {
                self.lanes
                    .iter()
                    .filter(|lane| lane.ship == ship)
                    .all(|lane| matches!(lane.state, OutState::Done))
            })
            .collect();
        if ended.is_empty() {
            return false;
        }
        for ship in &ended {
            let error = self.ship_errors.remove(ship);
            if let Some(respond) = self.ships.remove(ship) {
                let _ = respond.send(error.map_or(Ok(()), Err));
            }
        }
        self.lanes.retain(|lane| !ended.contains(&lane.ship));
        self.by_grant = self
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(j, lane)| match &lane.state {
                OutState::Open(grant) => Some(((grant.agent.clone(), grant.lane), j)),
                _ => None,
            })
            .collect();
        true
    }

    fn reap(&mut self) -> bool {
        let before = self.quarantine.len();
        let mut kept = Vec::new();
        for (id, frame) in std::mem::take(&mut self.quarantine) {
            if matches!(self.fabric.poll_write(id), Ok(true)) {
                self.fabric.forget_write(id);
                let _ = self.shared.engine.release_send(frame);
            } else {
                kept.push((id, frame));
            }
        }
        self.quarantine = kept;
        self.quarantine.len() < before
    }

    fn receive(&mut self) -> bool {
        let notes = match self.fabric.take_notes() {
            Ok(notes) => notes,
            Err(err) => {
                warn!(error = %err, "failed to read ring notifications");
                return false;
            }
        };
        let progressed = !notes.is_empty();
        for (agent, text) in notes {
            match RingNote::decode(&text) {
                Ok(RingNote::Frame {
                    lane,
                    batch,
                    k,
                    frame,
                    len,
                    header,
                }) => self.land(&agent, lane, batch, k, frame, len, header),
                Ok(RingNote::Credit { lane, frames }) => {
                    if let Some(&i) = self.by_grant.get(&(agent.clone(), lane)) {
                        self.lanes[i].credits.extend(frames);
                    } else {
                        warn!(%agent, lane, "a ring credit for a lane this CN does not send on");
                    }
                }
                Ok(RingNote::Refuse { lane, error }) => {
                    if let Some(&i) = self.by_grant.get(&(agent.clone(), lane)) {
                        self.fail_lane(i, error);
                    }
                }
                Err(err) => warn!(%agent, error = %err, "ignoring a malformed ring notification"),
            }
        }
        progressed
    }

    #[allow(clippy::too_many_arguments)]
    fn land(
        &mut self,
        agent: &str,
        lane: u64,
        batch: u64,
        k: u32,
        frame: u32,
        len: u64,
        header: Vec<u8>,
    ) {
        {
            let lanes = self.shared.lanes.lock().unwrap();
            match lanes.lanes.get(&lane) {
                Some(open) if open.agent == agent && open.frames.contains(&frame) => {
                    if open.refused {
                        // The sender fails the lane; the frame goes back when it closes.
                        return;
                    }
                }
                _ => {
                    warn!(
                        agent,
                        lane, frame, "a ring frame for a lane this CN does not hold"
                    );
                    return;
                }
            }
        }
        let key = (lane, batch);
        if !header.is_empty() {
            match self.shared.engine.begin(&header) {
                Ok(landing) => {
                    let early = self
                        .landings
                        .remove(&key)
                        .map(|pending| pending.early)
                        .unwrap_or_default();
                    self.landings.insert(
                        key,
                        InLanding {
                            landing,
                            copied: 0,
                            early: Vec::new(),
                        },
                    );
                    self.copy_out(key, k, frame, len);
                    for (k, frame, len) in early {
                        self.copy_out(key, k, frame, len);
                    }
                }
                Err(err) => self.refuse(lane, err),
            }
            return;
        }
        match self.landings.get_mut(&key) {
            Some(landing) if landing.landing.id != 0 => self.copy_out(key, k, frame, len),
            Some(landing) => landing.early.push((k, frame, len)),
            None => {
                // The header's frame is still on its way.
                self.landings.insert(
                    key,
                    InLanding {
                        landing: Landing::default(),
                        copied: 0,
                        early: vec![(k, frame, len)],
                    },
                );
            }
        }
    }

    fn copy_out(&mut self, key: (u64, u64), k: u32, frame: u32, len: u64) {
        let landing = self.landings[&key].landing.id;
        match self.shared.engine.copy_out(landing, k, frame, len) {
            Ok(ticket) => self.copies.push(CopyOut {
                ticket,
                lane: key.0,
                batch: key.1,
                frame,
            }),
            Err(err) => {
                self.shared.engine.abort(landing);
                self.landings.remove(&key);
                self.refuse(key.0, err);
            }
        }
    }

    /// Fails a receive lane: its sender is told, and its frames wait for the sender to close it.
    fn refuse(&mut self, lane: u64, error: String) {
        let agent = {
            let mut lanes = self.shared.lanes.lock().unwrap();
            let Some(open) = lanes.lanes.get_mut(&lane) else {
                return;
            };
            if open.refused {
                return;
            }
            open.refused = true;
            open.agent.clone()
        };
        warn!(lane, %agent, error = %error, "refusing a ring lane");
        // Landings of the lane in progress go too.
        let keys: Vec<_> = self
            .landings
            .keys()
            .filter(|k| k.0 == lane)
            .copied()
            .collect();
        for key in keys {
            if let Some(landing) = self.landings.remove(&key)
                && landing.landing.id != 0
            {
                self.shared.engine.abort(landing.landing.id);
            }
        }
        self.copies.retain(|copy| copy.lane != lane);
        if let Err(err) = self
            .fabric
            .notify(&agent, &RingNote::Refuse { lane, error }.encode())
        {
            warn!(lane, %agent, error = %err, "failed to tell a ring sender its lane was refused");
        }
    }

    fn finish_copies(&mut self) -> bool {
        let mut progressed = false;
        let mut k = 0;
        while k < self.copies.len() {
            let copy = &self.copies[k];
            match self.shared.engine.done(copy.ticket) {
                Ok(false) => {
                    k += 1;
                    continue;
                }
                Ok(true) => {}
                Err(err) => {
                    let lane = copy.lane;
                    self.refuse(lane, err);
                    progressed = true;
                    continue;
                }
            }
            let copy = self.copies.remove(k);
            progressed = true;
            // I2: the frame is free once copied out, whatever becomes of its batch.
            self.credits.entry(copy.lane).or_default().push(copy.frame);
            let key = (copy.lane, copy.batch);
            let Some(landing) = self.landings.get_mut(&key) else {
                continue;
            };
            landing.copied += 1;
            if landing.copied == landing.landing.frames {
                let landing = self.landings.remove(&key).expect("a landing");
                self.deliver(copy.lane, landing.landing);
            }
        }
        progressed
    }

    fn deliver(&mut self, lane: u64, landing: Landing) {
        let token = match self.shared.engine.seal(landing.id) {
            Ok(token) => token,
            Err(err) => {
                self.refuse(lane, err);
                return;
            }
        };
        let (params, names) = {
            let mut lanes = self.shared.lanes.lock().unwrap();
            let Some(open) = lanes.lanes.get_mut(&lane) else {
                self.shared.engine.release_token(token);
                return;
            };
            let mut params = open.params.clone();
            params.sequence = Some(open.next_seq);
            params.eos = Some(false);
            open.next_seq += 1;
            (params, open.names.clone())
        };
        let delivery = self.shared.delivery.lock().unwrap().clone();
        let delivered = match delivery {
            Some(delivery) => delivery.deliver(&params, token, landing.rows, &names),
            None => Err("the ring transport has nowhere to deliver batches".to_string()),
        };
        if let Err(err) = delivered {
            self.shared.engine.release_token(token);
            self.refuse(lane, err);
        }
    }

    fn send_credits(&mut self) -> bool {
        if self.credits.is_empty() {
            return false;
        }
        for (lane, frames) in std::mem::take(&mut self.credits) {
            let agent = {
                let lanes = self.shared.lanes.lock().unwrap();
                match lanes.lanes.get(&lane) {
                    Some(open) if !open.refused => open.agent.clone(),
                    _ => continue,
                }
            };
            let note = RingNote::Credit { lane, frames }.encode();
            if let Err(err) = self.fabric.notify(&agent, &note) {
                warn!(lane, %agent, error = %err, "failed to return ring credits");
            }
        }
        true
    }
}

type Feed = Receiver<Result<RingJob, String>>;

/// Exports a lane's batches on a thread of its own, a few ahead of the frames in flight, so a
/// parked output's export (on the engine thread) or a streamed one's wait never holds up the
/// loop.
fn spawn_feeder(
    mut source: Source,
    slot: SenderSlot,
    deadline: Instant,
    stop: Arc<AtomicBool>,
    engine: Arc<dyn RingEngine>,
) -> Result<(Feed, JoinHandle<()>), String> {
    let (tx, rx): (SyncSender<Result<RingJob, String>>, _) = sync_channel(FEED_AHEAD);
    let handle = std::thread::Builder::new()
        .name("ring-feed".to_string())
        .spawn(move || {
            let stopped = || stop.load(Ordering::Acquire);
            let mut backoff = OOM_RETRY_FIRST;
            loop {
                if stopped() {
                    return;
                }
                let mut item = match source.next(slot) {
                    Ok(RingNext::Job(job)) => Ok(job),
                    Ok(RingNext::Waiting) => continue,
                    Ok(RingNext::End) => return,
                    // Bringing a spilled batch back to the GPU found no memory: the engine kept
                    // the batch, so wait for memory and ask again, until the deadline (I4).
                    Err(err)
                        if err.contains("out_of_memory") && Instant::now() + backoff < deadline =>
                    {
                        std::thread::sleep(backoff);
                        backoff = (backoff * 2).min(OOM_RETRY_MAX);
                        continue;
                    }
                    Err(err) => Err(err),
                };
                backoff = OOM_RETRY_FIRST;
                let failed = item.is_err();
                // A full queue waits; a lane that failed meanwhile takes nothing more.
                loop {
                    match tx.try_send(item) {
                        Ok(()) => break,
                        Err(TrySendError::Full(back)) if !stopped() => {
                            item = back;
                            std::thread::sleep(Duration::from_micros(200));
                        }
                        Err(TrySendError::Full(back) | TrySendError::Disconnected(back)) => {
                            if let Ok(job) = back {
                                engine.release_job(job.token);
                            }
                            return;
                        }
                    }
                }
                if failed {
                    return;
                }
            }
        })
        .map_err(|err| format!("failed to spawn a ring feeder thread: {err}"))?;
    Ok((rx, handle))
}

#[cfg(test)]
mod tests {
    //! Two transports in one process, over a fabric that copies between host "rings" and queues
    //! notifications, an engine that lands frames in host buffers, and a control plane that calls
    //! the peer's handlers directly. They check the protocol, not NIXL or the GPU.

    use super::*;
    use crate::fragment_executor::FragmentResult;
    use crate::result_store::FragmentInstanceId;
    use starrocks_plan_translator::TranslatedPlan;
    use std::sync::atomic::AtomicUsize;

    const FRAME: u64 = 64;

    /// Memory of one node's rings and its notification queue, shared with the fabric.
    #[derive(Debug, Default)]
    struct Node {
        send: Mutex<Vec<u8>>,
        recv: Mutex<Vec<u8>>,
        notes: Mutex<VecDeque<(String, String)>>,
    }

    type Net = Arc<Mutex<HashMap<String, Arc<Node>>>>;

    struct MockFabric {
        name: String,
        node: Arc<Node>,
        net: Net,
        sent: Arc<Mutex<Vec<String>>>,
    }

    impl Fabric for MockFabric {
        fn local_md(&self) -> Vec<u8> {
            self.name.as_bytes().to_vec()
        }
        fn local_name(&self) -> String {
            self.name.clone()
        }
        fn load_peer(&mut self, md: &[u8]) -> Result<String, String> {
            let name = String::from_utf8(md.to_vec()).map_err(|e| e.to_string())?;
            if !self.net.lock().unwrap().contains_key(&name) {
                return Err(format!("no agent '{name}'"));
            }
            Ok(name)
        }
        fn write(
            &mut self,
            agent: &str,
            frame: u32,
            dst: u64,
            _device: i32,
            len: u64,
            note: &str,
        ) -> Result<WriteId, String> {
            let peer = self.net.lock().unwrap()[agent].clone();
            let from = (frame as u64 * FRAME) as usize;
            let bytes = self.node.send.lock().unwrap()[from..from + len as usize].to_vec();
            peer.recv.lock().unwrap()[dst as usize..dst as usize + len as usize]
                .copy_from_slice(&bytes);
            peer.notes
                .lock()
                .unwrap()
                .push_back((self.name.clone(), note.to_string()));
            Ok(1)
        }
        fn poll_write(&mut self, _id: WriteId) -> Result<bool, String> {
            Ok(true)
        }
        fn forget_write(&mut self, _id: WriteId) {}
        fn notify(&mut self, agent: &str, note: &str) -> Result<(), String> {
            self.sent.lock().unwrap().push(note.to_string());
            let peer = self.net.lock().unwrap()[agent].clone();
            peer.notes
                .lock()
                .unwrap()
                .push_back((self.name.clone(), note.to_string()));
            Ok(())
        }
        fn take_notes(&mut self) -> Result<Vec<(String, String)>, String> {
            Ok(self.node.notes.lock().unwrap().drain(..).collect())
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct Behavior {
        /// begin() fails with "no tier has room".
        no_room: bool,
        /// Copy-outs never finish.
        stuck: bool,
    }

    #[derive(Debug, Default)]
    struct EngineState {
        send_free: Vec<u32>,
        recv_free: Vec<u32>,
        jobs: HashMap<u64, Vec<u8>>,
        landings: HashMap<u64, (u64, Vec<u8>)>,
        tickets: HashSet<u64>,
        landed: HashMap<u64, Vec<u8>>,
        next: u64,
        released_jobs: usize,
        peak_recv_leased: usize,
    }

    #[derive(Debug)]
    struct MockEngine {
        node: Arc<Node>,
        recv_frames: usize,
        state: Mutex<EngineState>,
        behavior: Behavior,
    }

    impl MockEngine {
        fn new(
            node: Arc<Node>,
            send_frames: usize,
            recv_frames: usize,
            behavior: Behavior,
        ) -> Self {
            *node.send.lock().unwrap() = vec![0; send_frames * FRAME as usize];
            *node.recv.lock().unwrap() = vec![0; recv_frames * FRAME as usize];
            Self {
                node,
                recv_frames,
                state: Mutex::new(EngineState {
                    send_free: (0..send_frames as u32).rev().collect(),
                    recv_free: (0..recv_frames as u32).rev().collect(),
                    next: 1,
                    ..Default::default()
                }),
                behavior,
            }
        }

        /// A job whose stream is `len` bytes counting up from `seed`; its header says so.
        fn add_job(&self, seed: u8, len: u64, rows: u64) -> RingJob {
            let stream: Vec<u8> = (0..len).map(|i| seed.wrapping_add(i as u8)).collect();
            let mut state = self.state.lock().unwrap();
            let token = state.next;
            state.next += 1;
            state.jobs.insert(token, stream);
            let mut header = len.to_le_bytes().to_vec();
            header.extend_from_slice(&rows.to_le_bytes());
            RingJob {
                token,
                rows,
                stream_bytes: len,
                header,
            }
        }
    }

    impl RingEngine for MockEngine {
        fn frame_bytes(&self) -> u64 {
            FRAME
        }
        fn recv_region(&self) -> (i32, usize, u64) {
            (0, 0, self.node.recv.lock().unwrap().len() as u64)
        }
        fn lease_send(&self) -> Result<Option<u32>, String> {
            Ok(self.state.lock().unwrap().send_free.pop())
        }
        fn release_send(&self, frame: u32) -> Result<(), String> {
            self.state.lock().unwrap().send_free.push(frame);
            Ok(())
        }
        fn lease_recv(&self) -> Result<Option<u32>, String> {
            let mut state = self.state.lock().unwrap();
            let frame = state.recv_free.pop();
            let leased = self.recv_frames - state.recv_free.len();
            state.peak_recv_leased = state.peak_recv_leased.max(leased);
            Ok(frame)
        }
        fn release_recv(&self, frame: u32) -> Result<(), String> {
            self.state.lock().unwrap().recv_free.push(frame);
            Ok(())
        }
        fn stage(&self, token: u64, k: u32, frame: u32) -> Result<u64, String> {
            let mut state = self.state.lock().unwrap();
            let stream = state.jobs.get(&token).ok_or("unknown job")?.clone();
            let from = (k as u64 * FRAME) as usize;
            let to = (from + FRAME as usize).min(stream.len());
            let at = (frame as u64 * FRAME) as usize;
            self.node.send.lock().unwrap()[at..at + (to - from)].copy_from_slice(&stream[from..to]);
            let ticket = state.next;
            state.next += 1;
            state.tickets.insert(ticket);
            Ok(ticket)
        }
        fn release_job(&self, token: u64) {
            let mut state = self.state.lock().unwrap();
            if state.jobs.remove(&token).is_some() {
                state.released_jobs += 1;
            }
        }
        fn begin(&self, header: &[u8]) -> Result<Landing, String> {
            if self.behavior.no_room {
                return Err(
                    "ring exchange: no tier has room for a batch: GPU 0, host 0, disk none"
                        .to_string(),
                );
            }
            let len = u64::from_le_bytes(header[..8].try_into().unwrap());
            let rows = u64::from_le_bytes(header[8..16].try_into().unwrap());
            let mut state = self.state.lock().unwrap();
            let id = state.next;
            state.next += 1;
            state.landings.insert(id, (rows, vec![0; len as usize]));
            Ok(Landing {
                id,
                rows,
                frames: len.div_ceil(FRAME) as u32,
            })
        }
        fn copy_out(&self, landing: u64, k: u32, frame: u32, len: u64) -> Result<u64, String> {
            let mut state = self.state.lock().unwrap();
            let at = (frame as u64 * FRAME) as usize;
            let bytes = self.node.recv.lock().unwrap()[at..at + len as usize].to_vec();
            let (_, buffer) = state.landings.get_mut(&landing).ok_or("unknown landing")?;
            let to = (k as u64 * FRAME) as usize;
            buffer[to..to + len as usize].copy_from_slice(&bytes);
            let ticket = state.next;
            state.next += 1;
            if !self.behavior.stuck {
                state.tickets.insert(ticket);
            }
            Ok(ticket)
        }
        fn seal(&self, landing: u64) -> Result<u64, String> {
            let mut state = self.state.lock().unwrap();
            let (_, bytes) = state.landings.remove(&landing).ok_or("unknown landing")?;
            let token = state.next;
            state.next += 1;
            state.landed.insert(token, bytes);
            Ok(token)
        }
        fn abort(&self, landing: u64) {
            self.state.lock().unwrap().landings.remove(&landing);
        }
        fn done(&self, ticket: u64) -> Result<bool, String> {
            Ok(self.state.lock().unwrap().tickets.remove(&ticket))
        }
        fn release_token(&self, token: u64) {
            self.state.lock().unwrap().landed.remove(&token);
        }
        fn outstanding(&self) -> usize {
            self.state.lock().unwrap().landed.len()
        }
        fn stats_line(&self) -> String {
            String::new()
        }
    }

    /// A delivered batch: its sequence number, token, rows and column names.
    type Delivered = (i64, u64, u64, Vec<String>);

    #[derive(Debug, Default)]
    struct Sink {
        batches: Mutex<Vec<Delivered>>,
        eos: Mutex<Vec<i64>>,
        failed: Mutex<Vec<String>>,
    }

    impl RingDelivery for Sink {
        fn deliver(
            &self,
            request: &PTransmitChunkParams,
            token: u64,
            rows: u64,
            names: &[String],
        ) -> Result<(), String> {
            self.batches.lock().unwrap().push((
                request.sequence.unwrap(),
                token,
                rows,
                names.to_vec(),
            ));
            Ok(())
        }
    }

    /// A peer's transport, what it received, and its agent name.
    type Peer = (Arc<RingTransport>, Arc<Sink>, String);

    #[derive(Default)]
    struct MockControl {
        peers: Mutex<HashMap<SocketAddr, Peer>>,
    }

    impl Control for MockControl {
        fn call(
            &self,
            peer: SocketAddr,
            params: PTransmitChunkParams,
            envelope: &NixlEnvelope,
        ) -> Result<Vec<u8>, String> {
            let (transport, sink, md) = self.peers.lock().unwrap()[&peer].clone();
            match envelope {
                NixlEnvelope::Md(_) => Ok(md.into_bytes()),
                NixlEnvelope::RingOpen(body) => transport.ring_open(&params, body),
                NixlEnvelope::RingClose(lane) => transport.ring_close(*lane).map(|()| Vec::new()),
                NixlEnvelope::Packed { token: 0, .. } if params.eos == Some(true) => {
                    sink.eos.lock().unwrap().push(params.sequence.unwrap());
                    Ok(Vec::new())
                }
                other => Err(format!("unexpected control call {other:?}")),
            }
        }
        fn fail(&self, ends: Vec<(SocketAddr, SenderSlot)>, error: &str) {
            for (peer, _) in ends {
                if let Some((_, sink, _)) = self.peers.lock().unwrap().get(&peer) {
                    sink.failed.lock().unwrap().push(error.to_string());
                }
            }
        }
    }

    #[derive(Debug)]
    struct Jobs(Mutex<VecDeque<RingJob>>);

    impl FragmentExecutor for Jobs {
        fn execute(&self, _: &TranslatedPlan) -> Result<FragmentResult, String> {
            unreachable!()
        }
        fn export_ring_next(&self, _slot: SenderSlot) -> Result<Option<RingJob>, String> {
            Ok(self.0.lock().unwrap().pop_front())
        }
    }

    #[derive(Default)]
    struct Cluster {
        net: Net,
        control: Arc<MockControl>,
    }

    struct Cn {
        transport: Arc<RingTransport>,
        engine: Arc<MockEngine>,
        sink: Arc<Sink>,
        addr: SocketAddr,
        sent: Arc<Mutex<Vec<String>>>,
    }

    impl Cluster {
        fn cn(
            &self,
            port: u16,
            send: usize,
            recv: usize,
            behavior: Behavior,
            settings: RingSettings,
        ) -> Cn {
            let name = format!("cn{port}");
            let node = Arc::new(Node::default());
            self.net
                .lock()
                .unwrap()
                .insert(name.clone(), Arc::clone(&node));
            let engine = Arc::new(MockEngine::new(Arc::clone(&node), send, recv, behavior));
            let sent = Arc::default();
            let fabric = MockFabric {
                name: name.clone(),
                node,
                net: Arc::clone(&self.net),
                sent: Arc::clone(&sent),
            };
            let transport = Arc::new(
                RingTransport::start(
                    fabric,
                    Arc::clone(&engine) as Arc<dyn RingEngine>,
                    Arc::clone(&self.control) as Arc<dyn Control>,
                    settings,
                )
                .unwrap(),
            );
            let sink = Arc::new(Sink::default());
            transport.set_delivery(Arc::clone(&sink) as Arc<dyn RingDelivery>);
            let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
            self.control
                .peers
                .lock()
                .unwrap()
                .insert(addr, (Arc::clone(&transport), Arc::clone(&sink), name));
            Cn {
                transport,
                engine,
                sink,
                addr,
                sent,
            }
        }
    }

    fn settings() -> RingSettings {
        RingSettings {
            credits: 4,
            max_wait: Duration::from_secs(10),
            xfer_timeout: Duration::from_secs(10),
        }
    }

    fn settings_with_wait(ms: u64) -> RingSettings {
        RingSettings {
            max_wait: Duration::from_millis(ms),
            ..settings()
        }
    }

    fn slot(node_id: i32, sender_id: i32) -> SenderSlot {
        SenderSlot {
            fragment_instance_id: FragmentInstanceId::from_halves(1, 2),
            node_id,
            sender_id,
        }
    }

    fn names() -> Vec<String> {
        vec!["a".to_string(), "b".to_string()]
    }

    fn ship(a: &Cn, b: &Cn, jobs: impl FragmentExecutor + 'static) -> Result<(), String> {
        a.transport
            .send(b.addr, slot(5, 0), names(), Arc::new(jobs))
    }

    fn landed(cn: &Cn, token: u64) -> Vec<u8> {
        cn.engine.state.lock().unwrap().landed[&token].clone()
    }

    #[test]
    fn batches_land_whole_in_order_then_eos() {
        let cluster = Cluster::default();
        let a = cluster.cn(1, 4, 8, Behavior::default(), settings());
        let b = cluster.cn(2, 4, 8, Behavior::default(), settings());
        // Sizes around the frame: smaller, exact, one byte over, many frames.
        let sizes = [10u64, 64, 65, 1000, 1, 640];
        let jobs: VecDeque<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &len)| a.engine.add_job(i as u8 * 31, len, len * 2))
            .collect();
        ship(&a, &b, Jobs(Mutex::new(jobs))).unwrap();

        let batches = b.sink.batches.lock().unwrap().clone();
        assert_eq!(batches.len(), sizes.len());
        for (i, (seq, token, rows, got_names)) in batches.iter().enumerate() {
            assert_eq!(*seq, i as i64, "sequence numbers count delivered batches");
            assert_eq!(got_names, &names());
            let bytes = landed(&b, *token);
            let len = bytes.len() as u64;
            assert_eq!(*rows, len * 2);
            let k = sizes.iter().position(|&s| s == len).unwrap();
            let expect: Vec<u8> = (0..len)
                .map(|j| (k as u8 * 31).wrapping_add(j as u8))
                .collect();
            assert_eq!(bytes, expect, "batch {i} landed byte for byte");
        }
        // EOS after every batch, numbered after them.
        assert_eq!(*b.sink.eos.lock().unwrap(), vec![sizes.len() as i64]);
        assert!(b.sink.failed.lock().unwrap().is_empty());
        // Every job was released on the sender, and every frame went back.
        let state = a.engine.state.lock().unwrap();
        assert_eq!(state.released_jobs, sizes.len());
        assert_eq!(state.send_free.len(), 4);
        drop(state);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(b.engine.state.lock().unwrap().recv_free.len(), 8);
    }

    #[test]
    fn i2_frames_come_back_per_copy_so_a_batch_larger_than_the_lane_flows() {
        let cluster = Cluster::default();
        let lane_of_two = RingSettings {
            credits: MIN_CREDITS,
            ..settings()
        };
        let a = cluster.cn(1, 2, 2, Behavior::default(), lane_of_two);
        let b = cluster.cn(2, 2, 2, Behavior::default(), lane_of_two);
        // 20 frames through a lane of 2: each frame is free once copied out, long before the
        // batch is whole.
        let job = a.engine.add_job(7, 20 * FRAME - 3, 1);
        ship(&a, &b, Jobs(Mutex::new([job].into()))).unwrap();
        let batches = b.sink.batches.lock().unwrap().clone();
        assert_eq!(batches.len(), 1);
        assert_eq!(landed(&b, batches[0].1).len() as u64, 20 * FRAME - 3);
        assert_eq!(b.engine.state.lock().unwrap().peak_recv_leased, 2);
    }

    #[test]
    fn i3_every_lane_opens_with_two_credits_or_waits_for_them() {
        let engine = MockEngine::new(Arc::new(Node::default()), 1, 7, Behavior::default());
        let mut lanes = RecvLanes::default();
        let open = |lanes: &mut RecvLanes, engine: &MockEngine, want| {
            lanes
                .open(
                    engine,
                    "x".to_string(),
                    packed_params(None, 0, false),
                    names(),
                    want,
                )
                .unwrap()
        };
        let (first, frames) = open(&mut lanes, &engine, 3).unwrap();
        assert_eq!(frames.len(), 3);
        let (_, frames) = open(&mut lanes, &engine, 3).unwrap();
        assert_eq!(frames.len(), 3);
        // One frame left: not enough for a lane; nothing is taken.
        assert!(open(&mut lanes, &engine, 3).is_none());
        assert_eq!(engine.state.lock().unwrap().recv_free.len(), 1);
        // A closed lane's frames serve the next one once no copy-out reads them.
        lanes.close(first);
        lanes.finish_close(&engine, |_| true);
        assert!(open(&mut lanes, &engine, 3).is_none());
        lanes.finish_close(&engine, |_| false);
        let (_, frames) = open(&mut lanes, &engine, 3).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(lanes.lanes.len(), 2);
        // Asking for one credit still gets two.
        let spare = MockEngine::new(Arc::new(Node::default()), 1, 4, Behavior::default());
        let (_, frames) = open(&mut RecvLanes::default(), &spare, 1).unwrap();
        assert_eq!(frames.len(), MIN_CREDITS);
    }

    #[test]
    fn i3_more_lanes_than_frames_take_turns() {
        let cluster = Cluster::default();
        let a = cluster.cn(1, 8, 4, Behavior::default(), settings());
        // 4 receive frames: two lanes of two at a time.
        let b = cluster.cn(
            2,
            8,
            4,
            Behavior::default(),
            RingSettings {
                credits: 2,
                ..settings()
            },
        );
        let hops: Vec<_> = (0..5)
            .map(|sender| {
                let jobs: VecDeque<_> = (0..3)
                    .map(|i| a.engine.add_job(sender as u8, 100 + i, 1))
                    .collect();
                (slot(5, sender), Arc::new(Jobs(Mutex::new(jobs))))
            })
            .collect();
        std::thread::scope(|scope| {
            for (slot, jobs) in hops {
                let (transport, peer) = (Arc::clone(&a.transport), b.addr);
                scope.spawn(move || transport.send(peer, slot, names(), jobs).unwrap());
            }
        });
        assert_eq!(b.sink.batches.lock().unwrap().len(), 15);
        assert_eq!(b.sink.eos.lock().unwrap().len(), 5);
    }

    #[test]
    fn i4_a_lane_without_credits_fails_at_its_deadline() {
        let cluster = Cluster::default();
        let short = settings_with_wait(300);
        let a = cluster.cn(1, 4, 4, Behavior::default(), short);
        let b = cluster.cn(
            2,
            4,
            4,
            Behavior {
                stuck: true,
                ..Behavior::default()
            },
            short,
        );
        let job = a.engine.add_job(1, 10 * FRAME, 1);
        let started = Instant::now();
        let err = ship(&a, &b, Jobs(Mutex::new([job].into()))).unwrap_err();
        assert!(
            err.contains("no credit before the query's deadline"),
            "{err}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
        // The receiver got a failure frame in place of EOS.
        assert!(b.sink.eos.lock().unwrap().is_empty());
        assert_eq!(b.sink.failed.lock().unwrap().len(), 1);
        assert_eq!(a.engine.state.lock().unwrap().released_jobs, 1);
    }

    #[test]
    fn i4_a_busy_receiver_is_waited_for_only_until_the_deadline() {
        let cluster = Cluster::default();
        let short = settings_with_wait(200);
        let a = cluster.cn(1, 4, 4, Behavior::default(), short);
        let b = cluster.cn(2, 4, 4, Behavior::default(), short);
        // Every receive frame is held by someone else.
        for _ in 0..4 {
            b.engine.lease_recv().unwrap().unwrap();
        }
        let job = a.engine.add_job(1, 10, 1);
        let err = ship(&a, &b, Jobs(Mutex::new([job].into()))).unwrap_err();
        assert!(
            err.contains("granted no lane before the query's deadline"),
            "{err}"
        );
    }

    #[test]
    fn i5_no_room_on_the_receiver_fails_the_hop_with_its_numbers() {
        let cluster = Cluster::default();
        let a = cluster.cn(1, 4, 4, Behavior::default(), settings());
        let b = cluster.cn(
            2,
            4,
            4,
            Behavior {
                no_room: true,
                ..Behavior::default()
            },
            settings(),
        );
        let job = a.engine.add_job(1, 3 * FRAME, 1);
        let err = ship(&a, &b, Jobs(Mutex::new([job].into()))).unwrap_err();
        assert!(err.contains("no tier has room"), "{err}");
        assert!(err.contains("GPU 0, host 0, disk none"), "{err}");
        assert!(b.sink.batches.lock().unwrap().is_empty());
        assert_eq!(b.sink.failed.lock().unwrap().len(), 1);
        // The refused lane's frames went back once the sender closed it.
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(b.engine.state.lock().unwrap().recv_free.len(), 4);
        assert!(a.sent.lock().unwrap().is_empty());
        assert!(
            b.sent
                .lock()
                .unwrap()
                .iter()
                .any(|note| matches!(RingNote::decode(note), Ok(RingNote::Refuse { .. })))
        );
    }

    #[test]
    fn two_cns_ship_to_each_other_at_once() {
        let cluster = Cluster::default();
        let a = cluster.cn(1, 4, 6, Behavior::default(), settings());
        let b = cluster.cn(2, 4, 6, Behavior::default(), settings());
        let to_b: VecDeque<_> = (0..20).map(|i| a.engine.add_job(i, 300, 1)).collect();
        let to_a: VecDeque<_> = (0..20).map(|i| b.engine.add_job(i, 300, 1)).collect();
        std::thread::scope(|scope| {
            let (ta, tb) = (Arc::clone(&a.transport), Arc::clone(&b.transport));
            let (pa, pb) = (a.addr, b.addr);
            scope.spawn(move || {
                ta.send(pb, slot(5, 0), names(), Arc::new(Jobs(Mutex::new(to_b))))
                    .unwrap()
            });
            scope.spawn(move || {
                tb.send(pa, slot(6, 1), names(), Arc::new(Jobs(Mutex::new(to_a))))
                    .unwrap()
            });
        });
        assert_eq!(a.sink.batches.lock().unwrap().len(), 20);
        assert_eq!(b.sink.batches.lock().unwrap().len(), 20);
    }

    /// Fails its first exports for want of memory, as an engine bringing a spilled batch back
    /// would, then hands out its jobs.
    #[derive(Debug)]
    struct Tight {
        jobs: Mutex<VecDeque<RingJob>>,
        ooms: AtomicUsize,
    }

    impl FragmentExecutor for Tight {
        fn execute(&self, _: &TranslatedPlan) -> Result<FragmentResult, String> {
            unreachable!()
        }
        fn export_ring_next(&self, _slot: SenderSlot) -> Result<Option<RingJob>, String> {
            if self.ooms.load(Ordering::Relaxed) > 0 {
                self.ooms.fetch_sub(1, Ordering::Relaxed);
                return Err(
                    "std::bad_alloc: out_of_memory: exchange inbox: 59916192 bytes to \
                            bring a spilled batch back"
                        .to_string(),
                );
            }
            Ok(self.jobs.lock().unwrap().pop_front())
        }
    }

    #[test]
    fn an_export_out_of_memory_waits_and_asks_again() {
        let cluster = Cluster::default();
        let a = cluster.cn(1, 4, 4, Behavior::default(), settings());
        let b = cluster.cn(2, 4, 4, Behavior::default(), settings());
        let jobs: VecDeque<_> = (0..3).map(|i| a.engine.add_job(i, 100, 1)).collect();
        let tight = Tight {
            jobs: Mutex::new(jobs),
            ooms: AtomicUsize::new(5),
        };
        ship(&a, &b, tight).unwrap();
        assert_eq!(b.sink.batches.lock().unwrap().len(), 3);
    }

    #[test]
    fn an_export_out_of_memory_past_the_deadline_fails_the_hop() {
        let cluster = Cluster::default();
        let short = settings_with_wait(200);
        let a = cluster.cn(1, 4, 4, Behavior::default(), short);
        let b = cluster.cn(2, 4, 4, Behavior::default(), short);
        let tight = Tight {
            jobs: Mutex::default(),
            ooms: AtomicUsize::new(usize::MAX),
        };
        let err = ship(&a, &b, tight).unwrap_err();
        assert!(err.contains("out_of_memory"), "{err}");
    }

    #[test]
    fn shutting_down_drops_the_delivery() {
        let cluster = Cluster::default();
        let a = cluster.cn(1, 2, 2, Behavior::default(), settings());
        let held = Arc::strong_count(&a.sink);
        a.transport.shut_down();
        assert_eq!(Arc::strong_count(&a.sink), held - 1);
        a.transport.shut_down();
        assert!(
            a.transport
                .send(
                    a.addr,
                    slot(1, 1),
                    names(),
                    Arc::new(Jobs(Mutex::default()))
                )
                .is_err()
        );
    }
}
