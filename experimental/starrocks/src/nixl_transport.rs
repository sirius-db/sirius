//! The nixl exchange transport tier (PLAN-PATH-B B5).
//!
//! One dedicated thread owns the nixl [`Agent`] — the Rust binding documents a multithreading
//! deadlock caveat, so every agent touch funnels through the request channel, mirroring the
//! engine-thread pattern. The [`NixlTransport`] handle and its request types compile in every
//! build so the service can hold and test the seam; only [`NixlTransport::start`] and the thread
//! body need libnixl and are gated on the `nixl-transport` feature.
//!
//! Wire shape (WRITE-based; every lease lifetime stays process-local):
//! sender `export_packed` → `request_staging_lease` at the peer → nixl WRITE lease→lease →
//! `transmit_packed` (pack metadata in the brpc attachment) → release the local lease; a final
//! `transmit_packed{eos}` closes the sender on the peer's rendezvous. EOS and sender-set
//! completion stay on brpc — one source of truth.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::fragment_executor::SenderSlot;
use crate::prpc_client::PrpcClient;

/// Bring-up pre-establishment of the peer sessions; see the module for why it exists.
#[cfg(feature = "nixl-transport")]
mod warmup;

/// A bare `nixl_capi_is_stub()` build would dlopen-fail at agent creation; every startup error
/// message points here so the fix is discoverable.
#[cfg(feature = "nixl-transport")]
const ENV_HINT: &str = "source tools/nvda_nixl/ENV.sh (NIXL_PREFIX/NIXL_PLUGIN_DIR/\
                        LD_LIBRARY_PATH) and set UCX_TLS=cuda_copy,cuda_ipc,tcp,self";

/// Reply to a metadata exchange: this CN's agent identity for the peer to load.
#[derive(Clone, Debug)]
pub(crate) struct MdReply {
    /// This CN's nixl agent name (`{advertise_host}:{brpc_port}`).
    pub(crate) agent_name: String,
    /// This CN's serialized agent metadata (getLocalMD blob).
    pub(crate) metadata: Vec<u8>,
}

/// One parked sender output to transmit to a remote receiver.
#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
pub(crate) struct RemoteSendSpec {
    /// Peer CN advertised host.
    pub(crate) host: String,
    /// Peer CN brpc port.
    pub(crate) brpc_port: u16,
    /// Where the engine parked the sender's batches; also carries the receiver instance id,
    /// exchange node id, and sender ordinal the wire frames address.
    pub(crate) slot: SenderSlot,
    /// Sender fragment output column names, repeated on every frame (pack metadata carries none).
    pub(crate) names: Vec<String>,
}

/// One message to the transport thread.
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
pub(crate) enum TransportRequest {
    /// A peer opened a session: load its metadata, reply with ours. Idempotent.
    ExchangeMd {
        peer_agent_name: String,
        peer_metadata: Vec<u8>,
        respond: Sender<Result<MdReply, String>>,
    },
    /// Drain one parked sender output to a remote receiver. Handled inline on the transport
    /// thread, one drain at a time; `respond` fires once every batch and the eos have been
    /// transmitted, and the requester decides when to wait for it (see [`DrainTicket`]).
    SendFragment {
        spec: RemoteSendSpec,
        respond: Sender<Result<(), String>>,
    },
    /// Install one peer session whose metadata handshake the [`warmup`] thread already ran off
    /// this thread; only the agent-local load and the bandwidth canary happen here. Idempotent.
    WarmSession {
        host: String,
        brpc_port: u16,
        client: PrpcClient,
        peer: MdReply,
        respond: Sender<Result<(), String>>,
    },
}

/// The bring-up session warmup thread and its stop flag; see the [`warmup`] module.
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
#[derive(Debug)]
pub(crate) struct SessionWarmup {
    stop: Arc<AtomicBool>,
    thread: JoinHandle<()>,
}

impl SessionWarmup {
    /// Asks the warmup thread to stop and joins it. An attempt already in flight still has to
    /// finish its brpc call (bounded by `PrpcClient`'s reply timeout); `main`'s shutdown grace
    /// timer is the backstop.
    fn stop_and_join(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }
}

/// One posted remote drain, and the channel its outcome arrives on.
///
/// The transport thread owns the whole drain once the request is queued — every batch frame, the
/// eos frame, and the exactly-once `drop_parked` that releases that destination's claim on the
/// parked fragment. So dropping a ticket cancels nothing; it only throws away the one error the
/// FE would have seen. [`join`](Self::join) is the sanctioned consumer and `Drop` shouts if
/// anything else gets there first.
#[must_use = "an unjoined drain ticket silently discards the drain's failure"]
#[derive(Debug)]
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
pub(crate) struct DrainTicket {
    /// Peer `host:port`, for the message when a drain fails or a ticket leaks.
    destination: String,
    /// Destination this drain serves; identifies it in logs.
    slot: SenderSlot,
    /// `None` once joined, which is how `Drop` tells a leak from a normal consume.
    outcome: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
}

impl DrainTicket {
    /// Blocks until this destination's drain has transmitted every batch and its eos frame.
    pub(crate) fn join(mut self) -> Result<(), String> {
        self.outcome
            .take()
            .expect("a drain ticket is joined exactly once")
            .recv()
            .map_err(|_| {
                format!(
                    "nixl transport thread dropped the response for the drain to {} ({:?})",
                    self.destination, self.slot
                )
            })?
    }

    /// Test seam: a ticket whose outcome the test drives directly.
    #[cfg(test)]
    pub(crate) fn for_test(
        destination: &str,
        slot: SenderSlot,
        outcome: std::sync::mpsc::Receiver<Result<(), String>>,
    ) -> Self {
        Self {
            destination: destination.to_string(),
            slot,
            outcome: Some(outcome),
        }
    }
}

impl Drop for DrainTicket {
    fn drop(&mut self) {
        if self.outcome.is_some() {
            // Not fatal — the transport thread still finishes the drain and still releases the
            // slot — but a failure that never reaches the FE would look like a hung query, so
            // make the leak loud rather than let it hide.
            tracing::error!(
                destination = %self.destination,
                slot = ?self.slot,
                "a remote drain ticket was dropped without being joined; its outcome is lost"
            );
        }
    }
}

/// Handle to the transport thread. Constructible only with the `nixl-transport` feature (via
/// [`start`](Self::start)); without it the type still exists so the service seam — and its
/// pure-Rust tests — compile everywhere.
#[derive(Debug)]
pub struct NixlTransport {
    /// Sender to the transport thread. `Mutex<Option<..>>` makes the `!Sync` sender shareable
    /// and lets `Drop` close the channel before joining; sends are brief.
    requests: Mutex<Option<Sender<TransportRequest>>>,
    /// Transport thread handle, taken and joined on drop.
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Bring-up peer-session warmup; `None` when it is disabled or this build has no nixl.
    warmup: Mutex<Option<SessionWarmup>>,
}

impl NixlTransport {
    /// Sends one request to the transport thread and blocks for its answer.
    fn transport_call<T>(
        &self,
        make_request: impl FnOnce(Sender<Result<T, String>>) -> TransportRequest,
    ) -> Result<T, String> {
        let (respond_tx, respond_rx) = channel();
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .ok_or_else(|| "nixl transport is shutting down".to_string())?
            .send(make_request(respond_tx))
            .map_err(|_| "nixl transport thread is not running".to_string())?;
        respond_rx
            .recv()
            .map_err(|_| "nixl transport thread dropped the response".to_string())?
    }

    /// Loads a peer's agent metadata and returns ours (the `exchange_nixl_md` handler body).
    pub(crate) fn exchange_md(
        &self,
        peer_agent_name: String,
        peer_metadata: Vec<u8>,
    ) -> Result<MdReply, String> {
        self.transport_call(|respond| TransportRequest::ExchangeMd {
            peer_agent_name,
            peer_metadata,
            respond,
        })
    }

    /// Queues one parked sender output for transmission to a remote receiver and returns at
    /// once, handing back the [`DrainTicket`] its outcome arrives on.
    ///
    /// The drain itself is unchanged: ONE transport thread still services `SendFragment`
    /// inline, so the drains it is handed run one at a time, in the order they were posted, and
    /// every frame of a destination is still issued by that single thread. What changes is who
    /// waits — the caller can dispatch a ready local receiver before joining, instead of the
    /// receiver waiting out this CN's whole (N-1)-destination drain.
    ///
    /// ORDERING (the invariant the receiver enforces): `local_exchange.rs:180-197` fails a query
    /// on a `seq` gap per (exchange key, sender ordinal). Both the counter and the eos frame live
    /// inside the transport thread's `send_fragment`, which this does not touch, and the request
    /// channel is FIFO — so the wire sequence is byte-identical to the blocking loop's.
    pub(crate) fn start_fragment(&self, spec: RemoteSendSpec) -> Result<DrainTicket, String> {
        let (respond, outcome) = channel();
        let destination = format!("{}:{}", spec.host, spec.brpc_port);
        let slot = spec.slot;
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .ok_or_else(|| "nixl transport is shutting down".to_string())?
            .send(TransportRequest::SendFragment { spec, respond })
            .map_err(|_| "nixl transport thread is not running".to_string())?;
        Ok(DrainTicket {
            destination,
            slot,
            outcome: Some(outcome),
        })
    }

    /// Test seam: a handle whose requests land on `requests` instead of a real transport thread.
    #[cfg(test)]
    pub(crate) fn for_test(requests: Sender<TransportRequest>) -> Self {
        Self {
            requests: Mutex::new(Some(requests)),
            thread: Mutex::new(None),
            warmup: Mutex::new(None),
        }
    }
}

impl Drop for NixlTransport {
    fn drop(&mut self) {
        // Stop the warmup first: it holds its own clone of the request sender, and the transport
        // thread's `recv()` only returns once every sender is gone.
        if let Some(warmup) = self
            .warmup
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            warmup.stop_and_join();
        }
        // Close the request channel so the thread's `recv()` returns, then join for an ordered
        // teardown (the thread holds an executor handle that must release before the engine).
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

#[cfg(feature = "nixl-transport")]
mod agent_tier {
    //! Everything that touches libnixl: the transport thread body and its helpers.

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::mpsc::Receiver;
    use std::time::{Duration, Instant};

    use nixl_sys::{
        Agent, MemType, MemoryRegion, NixlDescriptor, OptArgs, RegistrationHandle, XferDescList,
        XferOp, XferStatus,
    };
    use prost::Message;
    use starrocks_thrift::status_code::TStatusCode;
    use tracing::{info, warn};

    use super::*;
    use crate::FeConfig;
    use crate::fragment_executor::FragmentExecutor;
    use crate::proto::starrocks::p_internal_service_brpc::methods;
    use crate::proto::starrocks::{
        PExchangeNixlMd, PExchangeNixlMdResult, PStagingLeaseRequest, PStagingLeaseResult,
        PTransmitPackedParams, PTransmitPackedResult, PUniqueId, StatusPb,
    };

    /// Bytes of the mandatory first-contact bandwidth canary (finding F1): pool memory over
    /// cuda_ipc silently degrades ~220x with correct bytes, so a slow link must be refused, not
    /// tolerated.
    pub(super) const CANARY_BYTES: u64 = 16 << 20;
    /// A small first WRITE settles UCX connection wireup so the canary times the steady link,
    /// not the handshake.
    pub(super) const WARMUP_BYTES: u64 = 1 << 20;
    /// Floor under which the link is declared degraded. The healthy same-host cuda_ipc path
    /// measured ~85-90 GB/s; the degraded staged-copy path ~0.4 GB/s.
    pub(super) const CANARY_FLOOR_GBPS: f64 = 2.0;
    /// Bound on waiting for one posted WRITE to reach DONE.
    const XFER_TIMEOUT: Duration = Duration::from_secs(30);

    impl NixlTransport {
        /// Brings up the transport on a dedicated thread (fail-fast): nixl agent named
        /// `agent_name`, UCX backend, and the executor's staging arena registered as VRAM.
        /// Blocks until the agent is ready — or bring-up fails — so a missing libnixl, plugin
        /// dir, or arena surfaces here, before any cross-node query is accepted.
        ///
        /// Then starts the [`warmup`](super::warmup) thread, which discovers this CN's peers
        /// through `fe` and pre-establishes every directed session off the query path. That is
        /// best-effort: a warmup failure is loud but never fails bring-up, because a cold peer
        /// still works (slowly) through [`TransportState::ensure_session`].
        pub fn start(
            executor: Arc<dyn FragmentExecutor>,
            agent_name: String,
            fe: FeConfig,
        ) -> Result<Self, String> {
            let (request_tx, request_rx) = channel::<TransportRequest>();
            // The agent's serialized metadata comes back out of bring-up because the warmup
            // thread sends it to peers itself, without borrowing the transport thread.
            let (ready_tx, ready_rx) = channel::<Result<Vec<u8>, String>>();
            let thread_agent_name = agent_name.clone();
            let thread = std::thread::Builder::new()
                .name("nixl-transport".to_string())
                .spawn(move || transport_thread(executor, thread_agent_name, request_rx, ready_tx))
                .map_err(|err| format!("failed to spawn nixl-transport thread: {err}"))?;
            match ready_rx.recv() {
                Ok(Ok(local_md)) => {
                    let warmup = warmup::spawn(agent_name, local_md, request_tx.clone(), fe);
                    Ok(Self {
                        requests: Mutex::new(Some(request_tx)),
                        thread: Mutex::new(Some(thread)),
                        warmup: Mutex::new(warmup),
                    })
                }
                Ok(Err(err)) => {
                    let _ = thread.join();
                    Err(err)
                }
                Err(_) => Err("nixl-transport thread exited during bring-up".to_string()),
            }
        }
    }

    /// Transport-thread body: bring the agent up, signal readiness, then serve requests until
    /// the channel closes.
    fn transport_thread(
        executor: Arc<dyn FragmentExecutor>,
        agent_name: String,
        requests: Receiver<TransportRequest>,
        ready: Sender<Result<Vec<u8>, String>>,
    ) {
        let mut state = match TransportState::bring_up(executor, agent_name) {
            Ok(state) => {
                // A send error means the caller is already gone; nothing to serve.
                if ready.send(Ok(state.local_md.clone())).is_err() {
                    return;
                }
                state
            }
            Err(err) => {
                let _ = ready.send(Err(err));
                return;
            }
        };

        while let Ok(request) = requests.recv() {
            // Respond-send errors are ignored: the waiting caller may have been dropped.
            match request {
                TransportRequest::ExchangeMd {
                    peer_agent_name,
                    peer_metadata,
                    respond,
                } => {
                    let _ = respond.send(state.exchange_md(&peer_agent_name, &peer_metadata));
                }
                TransportRequest::SendFragment { spec, respond } => {
                    let result = state.send_fragment(&spec);
                    if result.is_err() {
                        // Best-effort GPU cleanup: without it a failed transmit pins the parked
                        // output for the process lifetime (per-query GC is cut from Path B).
                        if let Err(drop_err) = state.executor.drop_parked(spec.slot) {
                            warn!(
                                slot = ?spec.slot,
                                error = %drop_err,
                                "failed to drop the parked output of a failed remote transmit"
                            );
                        }
                    }
                    let _ = respond.send(result);
                }
                TransportRequest::WarmSession {
                    host,
                    brpc_port,
                    client,
                    peer,
                    respond,
                } => {
                    let key = format!("{host}:{brpc_port}");
                    let _ = respond.send(state.install_session(key, client, peer));
                }
            }
        }
        info!("nixl-transport thread shutting down");
    }

    /// The staging arena as a nixl memory descriptor: device-resident (`cudaMalloc` by the
    /// arena's contract — pool memory silently degrades over cuda_ipc), device ordinal 0
    /// (`CUDA_VISIBLE_DEVICES` pins the device per process).
    #[derive(Debug)]
    struct ArenaRegion {
        base: usize,
        len: usize,
    }

    impl MemoryRegion for ArenaRegion {
        unsafe fn as_ptr(&self) -> *const u8 {
            self.base as *const u8
        }

        fn size(&self) -> usize {
            self.len
        }
    }

    impl NixlDescriptor for ArenaRegion {
        fn mem_type(&self) -> MemType {
            MemType::Vram
        }

        fn device_id(&self) -> u64 {
            0
        }
    }

    /// One established peer: a cached brpc control-plane connection plus the loaded nixl agent.
    struct PeerSession {
        client: PrpcClient,
        remote_agent: String,
    }

    /// Thread-local transport state; the agent never leaves this thread.
    pub(super) struct TransportState {
        pub(super) executor: Arc<dyn FragmentExecutor>,
        pub(super) agent: Agent,
        pub(super) agent_name: String,
        pub(super) local_md: Vec<u8>,
        pub(super) staging_base: u64,
        /// Keeps the arena registered with the agent for the thread's lifetime.
        _arena_registration: RegistrationHandle,
        peers: HashMap<String, PeerSession>,
    }

    /// Creates one nixl agent with a UCX backend and the staging arena registered as VRAM.
    /// Returns the agent, the registration (kept alive for the agent's lifetime), and the
    /// serialized local metadata peers load.
    ///
    /// Reachable from the tests so they can register an arena the engine does not own.
    pub(super) fn bring_up_agent(
        agent_name: &str,
        staging_base: u64,
        staging_capacity: u64,
    ) -> Result<(Agent, RegistrationHandle, Vec<u8>), String> {
        let agent = Agent::new(agent_name).map_err(|err| {
            format!("failed to create nixl agent '{agent_name}': {err} — {ENV_HINT}")
        })?;
        let (_mem_types, params) = agent
            .get_plugin_params("UCX")
            .map_err(|err| format!("nixl UCX plugin unavailable: {err} — {ENV_HINT}"))?;
        let backend = agent
            .create_backend("UCX", &params)
            .map_err(|err| format!("failed to create the nixl UCX backend: {err} — {ENV_HINT}"))?;
        let mut opt_args =
            OptArgs::new().map_err(|err| format!("failed to create nixl opt args: {err}"))?;
        opt_args
            .add_backend(&backend)
            .map_err(|err| format!("failed to select the UCX backend: {err}"))?;
        let arena = ArenaRegion {
            base: staging_base as usize,
            len: staging_capacity as usize,
        };
        let arena_registration = agent.register_memory(&arena, Some(&opt_args)).map_err(|err| {
            format!(
                "failed to register the {staging_capacity}-byte staging arena with nixl: {err} — \
                 UCX_TLS must include cuda_copy for VRAM detection ({ENV_HINT})"
            )
        })?;
        let local_md = agent
            .get_local_md()
            .map_err(|err| format!("failed to serialize nixl agent metadata: {err}"))?;
        Ok((agent, arena_registration, local_md))
    }

    impl TransportState {
        pub(super) fn bring_up(
            executor: Arc<dyn FragmentExecutor>,
            agent_name: String,
        ) -> Result<Self, String> {
            let (staging_base, staging_capacity) = executor
                .staging_info()
                .map_err(|err| format!("nixl transport needs the exchange staging arena: {err}"))?;
            let (agent, arena_registration, local_md) =
                bring_up_agent(&agent_name, staging_base, staging_capacity)?;
            info!(
                agent = %agent_name,
                staging_base,
                staging_capacity,
                "nixl transport ready; staging arena registered"
            );
            Ok(Self {
                executor,
                agent,
                agent_name,
                local_md,
                staging_base,
                _arena_registration: arena_registration,
                peers: HashMap::new(),
            })
        }

        /// Receiver side of first contact: load the peer's metadata, reply with ours.
        fn exchange_md(
            &mut self,
            peer_agent_name: &str,
            peer_metadata: &[u8],
        ) -> Result<MdReply, String> {
            let loaded = self.agent.load_remote_md(peer_metadata).map_err(|err| {
                format!("failed to load nixl metadata of peer '{peer_agent_name}': {err}")
            })?;
            if loaded != peer_agent_name {
                return Err(format!(
                    "peer announced agent name '{peer_agent_name}' but its metadata decodes to \
                     '{loaded}'"
                ));
            }
            Ok(MdReply {
                agent_name: self.agent_name.clone(),
                metadata: self.local_md.clone(),
            })
        }

        /// Establishes the peer session on first contact: metadata exchange over brpc, then the
        /// mandatory bandwidth canary. Returns the session key.
        ///
        /// This is the LAZY path, and it is the one that hangs a cold cluster: the outbound
        /// `exchange_nixl_md` below blocks this thread on the peer's transport thread, which may
        /// itself be blocked calling back here (see the [`warmup`](super::warmup) module). It
        /// survives as the fallback for peers the warmup never reached; the warmup is what keeps
        /// queries off it.
        fn ensure_session(&mut self, host: &str, brpc_port: u16) -> Result<String, String> {
            let key = format!("{host}:{brpc_port}");
            if self.peers.contains_key(&key) {
                return Ok(key);
            }
            let mut client = PrpcClient::new(host, brpc_port);
            let peer = rpc_exchange_md(&mut client, &self.agent_name, &self.local_md)?;
            self.install_session(key.clone(), client, peer)?;
            Ok(key)
        }

        /// Second half of session set-up, once the peer's metadata is in hand: load it into this
        /// agent and clear the bandwidth canary. Split out so the warmup can run the metadata
        /// handshake on its own thread and hand the result in here — none of this blocks on the
        /// peer's transport thread (`request_staging_lease`/`transmit_packed` are served from the
        /// peer's blocking pool), so it cannot take part in the first-contact cycle.
        fn install_session(
            &mut self,
            key: String,
            mut client: PrpcClient,
            peer: MdReply,
        ) -> Result<(), String> {
            if self.peers.contains_key(&key) {
                return Ok(());
            }
            let loaded = self
                .agent
                .load_remote_md(&peer.metadata)
                .map_err(|err| format!("failed to load nixl metadata of peer {key}: {err}"))?;
            if loaded != peer.agent_name {
                return Err(format!(
                    "peer {key} announced agent name '{}' but its metadata decodes to '{loaded}'",
                    peer.agent_name
                ));
            }
            self.bandwidth_canary(&mut client, &loaded)?;
            self.peers.insert(
                key,
                PeerSession {
                    client,
                    remote_agent: loaded,
                },
            );
            Ok(())
        }

        /// F1's silent-degradation guard: nothing in nixl/UCX flags the ~220x staged-copy path
        /// (wrongly-allocated memory still transfers correct bytes), so the first contact WRITEs
        /// a 16 MiB lease→lease probe and refuses the tier below the floor.
        fn bandwidth_canary(
            &mut self,
            client: &mut PrpcClient,
            remote_agent: &str,
        ) -> Result<(), String> {
            let local_offset = self
                .executor
                .staging_lease(CANARY_BYTES)
                .map_err(|err| format!("failed to lease canary staging bytes locally: {err}"))?;
            let result = (|| {
                let lease = rpc_request_lease(client, CANARY_BYTES)?;
                let local_addr = self.staging_base + local_offset;
                // Warmup settles connection wireup; the timed WRITE measures the steady link.
                write_and_wait(
                    &self.agent,
                    remote_agent,
                    local_addr,
                    lease.remote_addr,
                    WARMUP_BYTES,
                )?;
                let elapsed = write_and_wait(
                    &self.agent,
                    remote_agent,
                    local_addr,
                    lease.remote_addr,
                    CANARY_BYTES,
                )?;
                // The canary flag makes the peer release its lease without touching its engine.
                rpc_transmit(
                    client,
                    PTransmitPackedParams {
                        canary: Some(true),
                        offset: Some(lease.offset),
                        length: Some(CANARY_BYTES),
                        ..Default::default()
                    },
                    Vec::new(),
                )?;
                let gbps = CANARY_BYTES as f64 / elapsed.as_secs_f64() / 1e9;
                info!(
                    peer = %client.peer(),
                    gbps = format!("{gbps:.1}"),
                    bytes = CANARY_BYTES,
                    "nixl bandwidth canary"
                );
                if gbps < CANARY_FLOOR_GBPS {
                    return Err(format!(
                        "nixl link to {} measured {gbps:.2} GB/s, below the {CANARY_FLOOR_GBPS} \
                         GB/s floor — the silent cuda_ipc degradation trap (F1: non-cudaMalloc \
                         staging memory, or UCX_TLS missing cuda_ipc). Refusing the transport \
                         tier",
                        client.peer()
                    ));
                }
                Ok(())
            })();
            if let Err(err) = self.executor.staging_release(local_offset) {
                warn!(error = %err, "failed to release the local canary lease");
            }
            result
        }

        /// Sender flow: drain the parked output batch by batch through the peer's staging arena.
        pub(super) fn send_fragment(&mut self, spec: &RemoteSendSpec) -> Result<(), String> {
            let key = self.ensure_session(&spec.host, spec.brpc_port)?;
            let session = self.peers.get_mut(&key).expect("session ensured above");
            let (hi, lo) = spec.slot.fragment_instance_id.as_halves();
            let finst_id = PUniqueId { hi, lo };
            let mut seq: i64 = 0;
            let mut batches: u64 = 0;
            let mut bytes: u64 = 0;

            while let Some(mut batch) = self.executor.export_packed_next(spec.slot)? {
                // A metadata-only empty batch carries no payload: no peer lease, no WRITE, and
                // the receiver knows `len == 0` means nothing to release.
                let metadata = std::mem::take(&mut batch.metadata);
                let sent = (|| {
                    let (remote_offset, length) = if batch.len > 0 {
                        let lease = rpc_request_lease(&mut session.client, batch.len)?;
                        write_and_wait(
                            &self.agent,
                            &session.remote_agent,
                            self.staging_base + batch.offset,
                            lease.remote_addr,
                            batch.len,
                        )?;
                        (lease.offset, batch.len)
                    } else {
                        (0, 0)
                    };
                    rpc_transmit(
                        &mut session.client,
                        PTransmitPackedParams {
                            finst_id: Some(finst_id),
                            node_id: Some(spec.slot.node_id),
                            sender_id: Some(spec.slot.sender_id),
                            eos: Some(false),
                            seq: Some(seq),
                            offset: Some(remote_offset),
                            length: Some(length),
                            column_names: spec.names.clone(),
                            canary: None,
                        },
                        metadata,
                    )
                })();
                // The local lease goes back whether the send succeeded or not: a lease left
                // outstanding on an error path pins the arena for every later query.
                if batch.len > 0 {
                    if let Err(release_err) = self.executor.staging_release(batch.offset) {
                        match sent {
                            Ok(_) => return Err(release_err),
                            // The send error is the root cause; the release failure rides along
                            // as a log line rather than masking it.
                            Err(_) => warn!(
                                offset = batch.offset,
                                error = %release_err,
                                "failed to release the local staging lease after a send error"
                            ),
                        }
                    }
                }
                sent?;
                seq += 1;
                batches += 1;
                bytes += batch.len;
            }

            rpc_transmit(
                &mut session.client,
                PTransmitPackedParams {
                    finst_id: Some(finst_id),
                    node_id: Some(spec.slot.node_id),
                    sender_id: Some(spec.slot.sender_id),
                    eos: Some(true),
                    seq: Some(seq),
                    offset: None,
                    length: None,
                    column_names: spec.names.clone(),
                    canary: None,
                },
                Vec::new(),
            )?;
            self.executor.drop_parked(spec.slot)?;
            info!(
                stream_id = spec.slot.node_id,
                sender_id = spec.slot.sender_id,
                dest = %session.client.peer(),
                batches,
                bytes,
                "transmitted batches via nixl"
            );
            Ok(())
        }
    }

    /// Posts one WRITE `[local_addr, +len)` → `[remote_addr, +len)` and polls it to DONE within
    /// [`XFER_TIMEOUT`]. Returns the elapsed post-to-done time.
    pub(super) fn write_and_wait(
        agent: &Agent,
        remote_agent: &str,
        local_addr: u64,
        remote_addr: u64,
        len: u64,
    ) -> Result<Duration, String> {
        let mut local = XferDescList::new(MemType::Vram)
            .map_err(|err| format!("failed to create the local descriptor list: {err}"))?;
        local.add_desc(local_addr as usize, len as usize, 0);
        let mut remote = XferDescList::new(MemType::Vram)
            .map_err(|err| format!("failed to create the remote descriptor list: {err}"))?;
        remote.add_desc(remote_addr as usize, len as usize, 0);
        let request = agent
            .create_xfer_req(XferOp::Write, &local, &remote, remote_agent, None)
            .map_err(|err| {
                format!("failed to create a {len}-byte WRITE to agent '{remote_agent}': {err}")
            })?;
        let start = Instant::now();
        let mut in_progress = agent
            .post_xfer_req(&request, None)
            .map_err(|err| format!("failed to post a {len}-byte WRITE: {err}"))?;
        while in_progress {
            if start.elapsed() > XFER_TIMEOUT {
                return Err(format!(
                    "a {len}-byte nixl WRITE to agent '{remote_agent}' did not complete within \
                     {XFER_TIMEOUT:?}"
                ));
            }
            match agent
                .get_xfer_status(&request)
                .map_err(|err| format!("failed to poll a nixl WRITE: {err}"))?
            {
                XferStatus::Success => in_progress = false,
                XferStatus::InProgress => std::thread::yield_now(),
            }
        }
        Ok(start.elapsed())
    }

    /// Fails on a non-OK StarRocks method status, naming the peer's error messages.
    fn check_status(what: &str, status: &StatusPb) -> Result<(), String> {
        if status.status_code == TStatusCode::OK.0 {
            return Ok(());
        }
        Err(format!(
            "{what} failed with status {}: {}",
            status.status_code,
            status.error_msgs.join("; ")
        ))
    }

    /// `exchange_nixl_md` over brpc: our identity out, the peer's identity back. Reachable from
    /// the [`warmup`](super::warmup) thread on purpose — this is the one call that must NOT run
    /// on the transport thread.
    pub(super) fn rpc_exchange_md(
        client: &mut PrpcClient,
        agent_name: &str,
        local_md: &[u8],
    ) -> Result<MdReply, String> {
        let body = PExchangeNixlMd {
            agent_name: Some(agent_name.to_string()),
            agent_metadata: Some(local_md.to_vec()),
        }
        .encode_to_vec();
        let response = client.call(methods::EXCHANGE_NIXL_MD, body, Vec::new())?;
        let result = PExchangeNixlMdResult::decode(response.body.as_slice())
            .map_err(|err| format!("undecodable exchange_nixl_md reply: {err}"))?;
        check_status("exchange_nixl_md", &result.status)?;
        let agent_name = result
            .agent_name
            .filter(|name| !name.is_empty())
            .ok_or_else(|| "exchange_nixl_md reply carries no agent name".to_string())?;
        let metadata = result
            .agent_metadata
            .filter(|metadata| !metadata.is_empty())
            .ok_or_else(|| "exchange_nixl_md reply carries no agent metadata".to_string())?;
        Ok(MdReply {
            agent_name,
            metadata,
        })
    }

    /// A lease of the peer's staging arena to WRITE into.
    pub(super) struct RemoteLease {
        pub(super) remote_addr: u64,
        pub(super) offset: u64,
    }

    /// `request_staging_lease` over brpc.
    fn rpc_request_lease(client: &mut PrpcClient, length: u64) -> Result<RemoteLease, String> {
        let body = PStagingLeaseRequest { length }.encode_to_vec();
        let response = client.call(methods::REQUEST_STAGING_LEASE, body, Vec::new())?;
        let result = PStagingLeaseResult::decode(response.body.as_slice())
            .map_err(|err| format!("undecodable request_staging_lease reply: {err}"))?;
        check_status("request_staging_lease", &result.status)?;
        let remote_addr = result
            .remote_addr
            .filter(|addr| *addr != 0)
            .ok_or_else(|| "request_staging_lease reply carries no remote address".to_string())?;
        let offset = result
            .offset
            .ok_or_else(|| "request_staging_lease reply carries no lease offset".to_string())?;
        Ok(RemoteLease {
            remote_addr,
            offset,
        })
    }

    /// `transmit_packed` over brpc; the pack metadata rides the attachment.
    fn rpc_transmit(
        client: &mut PrpcClient,
        params: PTransmitPackedParams,
        metadata: Vec<u8>,
    ) -> Result<(), String> {
        let response = client.call(methods::TRANSMIT_PACKED, params.encode_to_vec(), metadata)?;
        let result = PTransmitPackedResult::decode(response.body.as_slice())
            .map_err(|err| format!("undecodable transmit_packed reply: {err}"))?;
        check_status("transmit_packed", &result.status)
    }

    #[cfg(test)]
    mod tests {
        use std::sync::Arc;

        use super::*;
        use crate::engine::SiriusEngine;
        use crate::engine_settings::EngineSettings;

        /// GPU + libnixl smoke for the agent tier in one process: proves that two real agents
        /// come up over the engine's `cudaMalloc` staging arena (registered as VRAM by both,
        /// like the two CNs of the demo), the metadata handshake loads, a cross-agent WRITE
        /// between two leases reaches DONE, and the measured bandwidth clears the F1 canary
        /// floor. It does NOT verify the transferred bytes (Rust has no view into the device
        /// leases) — value verification is the B6 end-to-end query, where real results are
        /// compared. nixl 1.3.2 refuses loading an agent's own metadata, so the "two processes"
        /// are two agents here.
        #[test]
        #[ignore = "GPU + libnixl smoke: source tools/nvda_nixl/ENV.sh, set UCX_TLS, run with --ignored"]
        fn nixl_cross_agent_write_between_arena_leases() {
            let _guard = crate::GPU_ENGINE_TEST_LOCK
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            // The arena is constructed at context bring-up, only when this is set.
            // SAFETY: the GPU lock is held, so no other thread touches the environment here.
            unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "128MiB") };

            let engine_dir = std::env::temp_dir().join("sirius-nixl-smoke");
            let executor: Arc<dyn FragmentExecutor> = Arc::new(
                SiriusEngine::start(EngineSettings {
                    config: None,
                    engine_dir,
                    gpu_device: None,
                })
                .expect("bring up sirius engine"),
            );
            let (base, capacity) = executor.staging_info().expect("staging arena info");

            let (sender_agent, _sender_registration, _sender_md) =
                bring_up_agent("127.0.0.1:18060", base, capacity)
                    .expect("bring up the sender-side nixl agent");
            let (receiver_agent, _receiver_registration, receiver_md) =
                bring_up_agent("127.0.0.1:18061", base, capacity)
                    .expect("bring up the receiver-side nixl agent");

            // The brpc handshake, minus the wire: the sender loads the receiver's metadata.
            let receiver_name = sender_agent
                .load_remote_md(&receiver_md)
                .expect("load the receiver agent's metadata");
            assert_eq!(receiver_name, "127.0.0.1:18061");

            let source = executor.staging_lease(CANARY_BYTES).unwrap();
            let target = executor.staging_lease(CANARY_BYTES).unwrap();
            assert_ne!(source, target, "two live leases must not alias");

            write_and_wait(
                &sender_agent,
                &receiver_name,
                base + source,
                base + target,
                WARMUP_BYTES,
            )
            .expect("warmup cross-agent WRITE");
            let elapsed = write_and_wait(
                &sender_agent,
                &receiver_name,
                base + source,
                base + target,
                CANARY_BYTES,
            )
            .expect("timed cross-agent WRITE");
            let gbps = CANARY_BYTES as f64 / elapsed.as_secs_f64() / 1e9;
            eprintln!(
                "nixl cross-agent WRITE: {CANARY_BYTES} bytes in {elapsed:?} = {gbps:.1} GB/s"
            );
            assert!(
                gbps >= CANARY_FLOOR_GBPS,
                "cross-agent WRITE measured {gbps:.2} GB/s, below the {CANARY_FLOOR_GBPS} GB/s \
                 canary floor — the F1 silent-degradation trap"
            );

            executor.staging_release(target).unwrap();
            executor.staging_release(source).unwrap();
            // Drop the agents (and their registrations) before the engine that owns the arena.
            drop(sender_agent);
            drop(receiver_agent);
            drop(executor);
            // SAFETY: the GPU lock is still held.
            unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
        }
    }
}

/// Two-process NVLink benchmark of this module's primitives, mirroring the `nixl-nvlink-repro`
/// Python harness. Declared here, not at the crate root, because it drives `agent_tier`'s
/// `pub(super)` `TransportState`/`write_and_wait` — the same calls production uses.
#[cfg(all(test, feature = "nixl-transport"))]
mod nixl_bench;

/// Two-host GPU buffer echo over those same primitives: the multi-box counterpart of
/// `nixl_bench`, which only ever crosses GPUs within one machine.
#[cfg(all(test, feature = "nixl-transport"))]
mod nixl_echo;

/// Scaffolding both multi-process tests share (CUDA shim, control socket, verification pattern,
/// NVLink counters).
#[cfg(all(test, feature = "nixl-transport"))]
mod two_node_harness;
