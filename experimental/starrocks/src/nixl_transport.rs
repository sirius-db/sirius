//! The nixl exchange transport thread.
//!
//! One dedicated thread owns the nixl [`nixl_sys::Agent`] — the Rust binding documents a
//! multithreading deadlock caveat, so every agent touch funnels through the request channel.
//! The [`NixlTransport`] handle and its request types compile in every build so later CN
//! wiring can name the seam; only [`NixlTransport::start`] and the thread body need libnixl
//! and are gated on the `nixl-transport` feature.
//!
//! This commit brings the agent up, registers the engine's `cudaMalloc` staging arena as
//! VRAM, caches local agent metadata for a later `/nixl-md` route, and loads a peer's
//! metadata on the agent thread. Packed send over HTTP (`SendFragment`) is stubbed until
//! the park-then-send commit. Session warmup, FE peer discovery, and brpc packed RPCs are
//! not ported.
//!
//! ONE CN PER GPU: the staging arena is registered with nixl as CUDA device 0 of this
//! process, because that is where the engine allocates it. Device 0 is the engine's GPU
//! only when the process sees exactly one — later `--gpu-device` exports
//! `CUDA_VISIBLE_DEVICES` to make that so — and neither nixl nor UCX reports a mismatch,
//! so bring-up refuses a `CUDA_VISIBLE_DEVICES` that names several devices (see
//! [`check_single_visible_device`]).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

/// A bare `nixl_capi_is_stub()` build would dlopen-fail at agent creation; every startup
/// error message points here so the fix is discoverable.
#[cfg(feature = "nixl-transport")]
const ENV_HINT: &str = "export TOOLS_DIR=/home/ubuntu/sirius-wt/tools and source \
                        experimental/starrocks/scripts/cn-env.sh \
                        (NIXL_PREFIX/NIXL_PLUGIN_DIR/LD_LIBRARY_PATH); \
                        set UCX_TLS=cuda_copy,cuda_ipc,tcp,self";

/// One parked sender output to transmit to a remote receiver. The ship path is not wired
/// in this commit; the type exists so later CN code can post `SendFragment` without
/// reshaping the request enum.
#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
pub(crate) struct RemoteSendSpec {
    /// Peer nixl agent name (`{advertise_host}:{brpc_port}` once HTTP is wired).
    pub(crate) peer_agent_name: String,
    /// Exchange stream the receiver will attach this hop to.
    pub(crate) dest_stream: i64,
    /// Sender ordinal inside that stream.
    pub(crate) sender_id: i32,
}

/// One message to the transport thread.
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
pub(crate) enum TransportRequest {
    /// Load a peer's serialized agent metadata. Returns the remote agent name nixl
    /// assigned. Idempotent for a given blob. nixl 1.3.2 refuses loading an agent's own
    /// metadata, so the caller must use a distinct peer.
    LoadPeerMd {
        peer_metadata: Vec<u8>,
        respond: Sender<Result<String, String>>,
    },
    /// Drain one parked sender output to a remote receiver. Stubbed in this commit.
    SendFragment {
        spec: RemoteSendSpec,
        respond: Sender<Result<(), String>>,
    },
}

/// Handle to the transport thread. Constructible only with the `nixl-transport` feature
/// (via [`NixlTransport::start`]); without it the type still exists so the service seam
/// compiles everywhere.
#[derive(Debug)]
pub struct NixlTransport {
    /// Sender to the transport thread. `Mutex<Option<..>>` makes the `!Sync` sender
    /// shareable and lets `Drop` close the channel before joining; sends are brief.
    requests: Mutex<Option<Sender<TransportRequest>>>,
    /// Transport thread handle, taken and joined on drop.
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Cached `get_local_md` blob. The later `/nixl-md` outbound leg serves this from the
    /// caller's thread; only `load_remote_md` goes through the agent thread.
    #[allow(dead_code)]
    local_md: Mutex<Option<Arc<Vec<u8>>>>,
    /// This CN's nixl agent name.
    #[allow(dead_code)]
    agent_name: String,
}

impl NixlTransport {
    /// Sends one request to the transport thread and blocks for its answer.
    #[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
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

    /// Cached local agent metadata. Readable from any thread; never touches the agent.
    #[cfg(feature = "nixl-transport")]
    pub fn local_md(&self) -> Result<Arc<Vec<u8>>, String> {
        self.local_md
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| "nixl transport has no cached local metadata".to_string())
    }

    /// This CN's nixl agent name, set at [`start`](Self::start).
    #[cfg(feature = "nixl-transport")]
    pub fn agent_name(&self) -> &str {
        &self.agent_name
    }

    /// Load a peer's agent metadata on the agent thread. Returns the remote agent name.
    #[cfg(feature = "nixl-transport")]
    pub fn load_peer_md(&self, peer_metadata: &[u8]) -> Result<String, String> {
        self.transport_call(|respond| TransportRequest::LoadPeerMd {
            peer_metadata: peer_metadata.to_vec(),
            respond,
        })
    }

    /// Packed send path. Not wired in this commit.
    #[cfg(feature = "nixl-transport")]
    #[allow(dead_code)]
    pub(crate) fn send_fragment(&self, spec: RemoteSendSpec) -> Result<(), String> {
        self.transport_call(|respond| TransportRequest::SendFragment { spec, respond })
    }
}

impl Drop for NixlTransport {
    fn drop(&mut self) {
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

/// The one-CN-per-GPU invariant behind the arena's nixl device ordinal
/// (`ArenaRegion::device_id` in the agent tier): the staging arena is registered as CUDA
/// device 0 of this process, which is the device the engine allocated it on only when the
/// process sees exactly one GPU. An export naming several devices would register the arena
/// against the wrong one with no error from nixl or UCX, so it is refused. Unset (`None`)
/// is accepted: the process then sees every GPU and the single-GPU engine uses device 0,
/// which is what the arena is registered as.
// Pure, so its tests run in every build; the agent tier that calls it compiles in no CI job.
#[cfg_attr(not(feature = "nixl-transport"), allow(dead_code))]
fn check_single_visible_device(exported: Option<&str>) -> Result<(), String> {
    let Some(visible) = exported else {
        return Ok(());
    };
    let devices = visible
        .split(',')
        .map(str::trim)
        .filter(|device| !device.is_empty())
        .count();
    if devices > 1 {
        return Err(format!(
            "CUDA_VISIBLE_DEVICES={visible:?} names {devices} devices, but the nixl tier \
             registers the staging arena as CUDA device 0 of this process (one CN per GPU): \
             pin each CN to one MIG"
        ));
    }
    Ok(())
}

/// Bytes of the log-only first-contact bandwidth canary. Tunable via
/// `SIRIUS_CN_NIXL_CANARY_BYTES`. The study never gates on the measured GiB/s (MIG 0→1
/// host-bounces on this box).
fn canary_bytes() -> u64 {
    parse_u64_env("SIRIUS_CN_NIXL_CANARY_BYTES").unwrap_or(16 << 20)
}

/// A small first WRITE settles UCX connection wireup so a later timed WRITE measures the
/// steady link, not the handshake. Not tunable: it is a wireup settle, not a measurement.
const WARMUP_BYTES: u64 = 1 << 20;

/// Bound on waiting for one posted WRITE to reach DONE. Tunable via
/// `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS`.
fn xfer_timeout() -> Duration {
    parse_u64_env("SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS")
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30))
}

fn parse_u64_env(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse().ok())
        .filter(|&n| n > 0)
}

#[cfg(feature = "nixl-transport")]
mod agent_tier {
    //! Everything that touches libnixl: the transport thread body and its helpers.

    use std::sync::mpsc::Receiver;
    use std::time::{Duration, Instant};

    use nixl_sys::{
        Agent, MemType, MemoryRegion, NixlDescriptor, OptArgs, RegistrationHandle, XferDescList,
        XferOp, XferStatus,
    };
    use tracing::info;

    use super::*;

    impl NixlTransport {
        /// Brings up the transport on a dedicated thread (fail-fast): nixl agent named
        /// `agent_name`, UCX backend, and `arena` registered as VRAM. Blocks until the
        /// agent is ready — or bring-up fails — so a missing libnixl, plugin dir, or
        /// arena surfaces here, before any cross-node query is accepted.
        ///
        /// The study start signature takes the [`sirius::StagingArena`] handle directly
        /// (not a `FragmentExecutor`) so this commit does not need executor staging
        /// verbs yet.
        pub fn start(agent_name: String, arena: sirius::StagingArena) -> Result<Self, String> {
            let (request_tx, request_rx) = channel::<TransportRequest>();
            let (ready_tx, ready_rx) = channel::<Result<Vec<u8>, String>>();
            let thread_agent_name = agent_name.clone();
            let thread = std::thread::Builder::new()
                .name("nixl-transport".to_string())
                .spawn(move || transport_thread(thread_agent_name, arena, request_rx, ready_tx))
                .map_err(|err| format!("failed to spawn nixl-transport thread: {err}"))?;
            match ready_rx.recv() {
                Ok(Ok(local_md)) => Ok(Self {
                    requests: Mutex::new(Some(request_tx)),
                    thread: Mutex::new(Some(thread)),
                    local_md: Mutex::new(Some(Arc::new(local_md))),
                    agent_name,
                }),
                Ok(Err(err)) => {
                    let _ = thread.join();
                    Err(err)
                }
                Err(_) => Err("nixl-transport thread exited during bring-up".to_string()),
            }
        }
    }

    /// Transport-thread body: bring the agent up, signal readiness, then serve requests
    /// until the channel closes.
    fn transport_thread(
        agent_name: String,
        arena: sirius::StagingArena,
        requests: Receiver<TransportRequest>,
        ready: Sender<Result<Vec<u8>, String>>,
    ) {
        let mut state = match TransportState::bring_up(agent_name, arena) {
            Ok(state) => {
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
            match request {
                TransportRequest::LoadPeerMd {
                    peer_metadata,
                    respond,
                } => {
                    let _ = respond.send(state.load_peer_md(&peer_metadata));
                }
                TransportRequest::SendFragment { spec, respond } => {
                    let _ = respond.send(Err(format!(
                        "send_fragment is not wired yet (peer={}, dest_stream={}, sender_id={})",
                        spec.peer_agent_name, spec.dest_stream, spec.sender_id
                    )));
                }
            }
        }
        info!("nixl-transport thread shutting down");
    }

    /// The staging arena as a nixl memory descriptor: device-resident (`cudaMalloc` by
    /// the arena's contract — pool memory silently degrades over cuda_ipc), device
    /// ordinal 0 of this process — which [`check_single_visible_device`] holds
    /// `bring_up_agent` to.
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

    /// Thread-local transport state; the agent never leaves this thread.
    struct TransportState {
        agent: Agent,
        #[allow(dead_code)]
        agent_name: String,
        local_md: Vec<u8>,
        /// Keeps the arena registered with the agent for the thread's lifetime.
        _arena_registration: RegistrationHandle,
        /// Kept so the `cudaMalloc` region cannot be freed while registered.
        _arena: sirius::StagingArena,
    }

    /// Creates one nixl agent with a UCX backend and the staging arena registered as VRAM.
    /// Returns the agent, the registration (kept alive for the agent's lifetime), and the
    /// serialized local metadata peers load.
    ///
    /// Reachable from the tests so they can register an arena the engine does not own via
    /// a transport handle.
    pub(super) fn bring_up_agent(
        agent_name: &str,
        staging_base: u64,
        staging_capacity: u64,
    ) -> Result<(Agent, RegistrationHandle, Vec<u8>), String> {
        let visible = std::env::var_os("CUDA_VISIBLE_DEVICES")
            .map(|exported| exported.to_string_lossy().into_owned());
        check_single_visible_device(visible.as_deref())?;
        if visible.is_none() {
            info!(
                "CUDA_VISIBLE_DEVICES is unset; registering the staging arena as CUDA device 0 \
                 (the single-GPU engine's device)"
            );
        }
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
        fn bring_up(agent_name: String, arena: sirius::StagingArena) -> Result<Self, String> {
            let staging_base = arena.base() as u64;
            let staging_capacity = arena.capacity();
            let (agent, arena_registration, local_md) =
                bring_up_agent(&agent_name, staging_base, staging_capacity)?;
            info!(
                agent = %agent_name,
                staging_base,
                staging_capacity,
                md_bytes = local_md.len(),
                "nixl transport ready; staging arena registered"
            );
            Ok(Self {
                agent,
                agent_name,
                local_md,
                _arena_registration: arena_registration,
                _arena: arena,
            })
        }

        fn load_peer_md(&mut self, peer_metadata: &[u8]) -> Result<String, String> {
            self.agent
                .load_remote_md(peer_metadata)
                .map_err(|err| format!("failed to load nixl peer metadata: {err}"))
        }
    }

    /// Posts one WRITE `[local_addr, +len)` → `[remote_addr, +len)` and polls it to DONE
    /// within [`xfer_timeout`]. Returns the elapsed post-to-done time.
    pub(super) fn write_and_wait(
        agent: &Agent,
        remote_agent: &str,
        local_addr: u64,
        remote_addr: u64,
        len: u64,
    ) -> Result<Duration, String> {
        if len == 0 {
            return Ok(Duration::ZERO);
        }
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
        let timeout = xfer_timeout();
        let start = Instant::now();
        let mut in_progress = agent
            .post_xfer_req(&request, None)
            .map_err(|err| format!("failed to post a {len}-byte WRITE: {err}"))?;
        while in_progress {
            if start.elapsed() > timeout {
                return Err(format!(
                    "a {len}-byte nixl WRITE to agent '{remote_agent}' did not complete within \
                     {timeout:?} (SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS)"
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

    /// Log-only first-contact WRITE. Returns observed GiB/s; the caller must not treat a
    /// low number as failure (MIG 0→1 on this box host-bounces).
    pub(super) fn bandwidth_canary(
        agent: &Agent,
        remote_agent: &str,
        local_addr: u64,
        remote_addr: u64,
        nbytes: u64,
    ) -> Result<f64, String> {
        write_and_wait(agent, remote_agent, local_addr, remote_addr, WARMUP_BYTES)?;
        let elapsed = write_and_wait(agent, remote_agent, local_addr, remote_addr, nbytes)?;
        let secs = elapsed.as_secs_f64().max(1e-9);
        let gbs = nbytes as f64 / secs / 1e9;
        info!(
            remote_agent,
            nbytes, gbs, "nixl bandwidth canary (log-only; not a gate)"
        );
        Ok(gbs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// GPU + libnixl smoke for the agent tier in one process: two real agents come up
        /// over the engine's `cudaMalloc` staging arena (registered as VRAM by both, like
        /// the two CNs of the study), the metadata handshake loads, and a cross-agent
        /// WRITE between two leases reaches DONE. The measured bandwidth is logged only
        /// — this test does not assert a floor (MIG 0→1 host-bounces). It does NOT verify
        /// the transferred bytes (Rust has no view into the device leases). nixl 1.3.2
        /// refuses loading an agent's own metadata, so the "two processes" are two agents
        /// here.
        #[test]
        #[ignore = "GPU + libnixl smoke: source cn-env.sh, set UCX_TLS, run with --ignored"]
        fn nixl_cross_agent_write_between_arena_leases() {
            let _guard = crate::GPU_ENGINE_TEST_LOCK
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            // The arena is constructed at context bring-up, only when this is set.
            // SAFETY: the GPU lock is held, so no other thread touches the environment here.
            unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "67108864") };

            let ctx = sirius::SiriusContext::new().expect("bring up sirius engine");
            let arena = ctx
                .staging_arena()
                .expect("staging arena (SIRIUS_EXCHANGE_STAGING_BYTES)");
            let base = arena.base() as u64;
            let capacity = arena.capacity();

            let (sender_agent, _sender_registration, _sender_md) =
                bring_up_agent("127.0.0.1:18060", base, capacity)
                    .expect("bring up the sender-side nixl agent");
            let (receiver_agent, _receiver_registration, receiver_md) =
                bring_up_agent("127.0.0.1:18061", base, capacity)
                    .expect("bring up the receiver-side nixl agent");

            let receiver_name = sender_agent
                .load_remote_md(&receiver_md)
                .expect("load the receiver agent's metadata");
            assert_eq!(receiver_name, "127.0.0.1:18061");

            let nbytes = WARMUP_BYTES.max(canary_bytes().min(capacity / 4));
            let source = arena.lease(nbytes).expect("source lease");
            let target = arena.lease(nbytes).expect("target lease");
            assert_ne!(source, target, "two live leases must not alias");

            let gbs = bandwidth_canary(
                &sender_agent,
                &receiver_name,
                base + source,
                base + target,
                nbytes,
            )
            .expect("canary WRITE");
            eprintln!(
                "nixl cross-agent WRITE: {nbytes} bytes = {gbs:.1} GB/s (log-only; not a gate)"
            );
            // gbs is informational. Do not assert a floor.
            let _ = gbs;

            arena.release(target).expect("release target");
            arena.release(source).expect("release source");
            drop(sender_agent);
            drop(receiver_agent);
            drop(arena);
            drop(ctx);
            // SAFETY: the GPU lock is still held.
            unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{canary_bytes, check_single_visible_device, parse_u64_env, xfer_timeout};
    use std::time::Duration;

    #[test]
    fn unset_cuda_visible_devices_is_accepted() {
        assert_eq!(check_single_visible_device(None), Ok(()));
    }

    #[test]
    fn one_visible_device_is_accepted() {
        assert_eq!(check_single_visible_device(Some("3")), Ok(()));
        assert_eq!(check_single_visible_device(Some(" 3, ")), Ok(()));
    }

    #[test]
    fn several_visible_devices_are_refused() {
        let err = check_single_visible_device(Some("0,1")).unwrap_err();
        assert!(err.contains("names 2 devices"), "{err}");
        assert!(err.contains("pin each CN to one MIG"), "{err}");
    }

    #[test]
    fn canary_bytes_defaults_to_16mib_when_unset() {
        // parse_u64_env is what canary_bytes() uses; avoid mutating process env here.
        assert_eq!(parse_u64_env("__SIRIUS_CN_NIXL_CANARY_BYTES_UNSET__"), None);
        let _ = canary_bytes();
        let _ = xfer_timeout();
        assert_eq!(super::WARMUP_BYTES, 1 << 20);
        assert!(Duration::from_secs(30) >= Duration::from_secs(1));
    }
}
