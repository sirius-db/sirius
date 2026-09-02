//! Two-**node** GPU buffer echo over the CN's own nixl primitives.
//!
//! The question this answers is narrow and load-bearing for multi-box Sirius: can one CN put a
//! GPU buffer into another CN's GPU *on a different machine*, and can that machine put it back,
//! with every byte intact? Everything else about distributed execution (FE planning, fragment
//! scheduling, pack/unpack) sits on top of that one guarantee.
//!
//! Two processes, one per host, one GPU each:
//!
//! ```text
//!   origin (host A, GPU n)                     echo (host B, GPU m)
//!     out   ──── nixl WRITE ────────────────►   recv          leg 1
//!     back  ◄─────────────────── nixl WRITE ─   recv          leg 2
//!     verify(back)                              verify(recv)
//! ```
//!
//! Both legs are [`write_and_wait`] — the exact call `send_fragment` and `bandwidth_canary` post
//! their WRITEs with — over a [`TransportState::bring_up`] agent whose staging arena is
//! registered as VRAM. Only the control plane is substituted: brpc needs a running StarRocks
//! cluster, so a TCP socket carries the same three payloads (md blob, peer lease address, "it
//! landed"), exactly as [`nixl_bench`](super::nixl_bench) does.
//!
//! Both sides verify: the echo host checks the bytes it received before echoing them, and the
//! origin checks the bytes that came back. Each side's destination buffer is zeroed before every
//! phase, so a stale buffer cannot fake a pass, and the payload seed varies per size so bytes
//! left by an earlier phase cannot either.
//!
//! On a GB200 NVL72 the two hosts share one NVLink domain (IMEX), so UCX can pick `cuda_ipc`
//! *across machines* and neither leg touches a NIC. Each side samples its own GPU's NVLink
//! counters and reports the delta, which is what distinguishes that from a host-bounced fallback.
//!
//! Run it through `scripts/nixl-echo-2node.sh`; every knob is an environment variable.

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nixl_sys::{Agent, RegistrationHandle};

use super::agent_tier::{TransportState, bring_up_agent, write_and_wait};
use super::two_node_harness::{
    check_pattern, connect_retry, cuda, cuda_vmm, env_or, env_required, expect_text, fill_pattern,
    human_bytes, nvlink_counters, percentile, phase_seed, recv_frame, recv_text, send_frame,
    send_text, tune_socket,
};
use crate::engine::SiriusEngine;
use crate::engine_settings::{EngineSettings, derive_sirius_config_yaml};
use crate::fragment_executor::FragmentExecutor;

/// 1 MiB, 16 MiB, 256 MiB: a fragment-sized payload, a large one, and one big enough that
/// per-transfer overhead stops mattering.
const DEFAULT_SIZES: &str = "1048576,16777216,268435456";
/// The origin waits this long for the echo host's engine to come up and start listening. Engine
/// bring-up allocates an RMM pool, which is not instant on a 185 GiB device.
const PEER_DEADLINE: Duration = Duration::from_secs(900);

// ---------------------------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------------------------

/// Which end of the echo this process is. The roles are asymmetric only in who dials whom and
/// who owns the round-trip timer; both run the same bring-up and both verify bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Sends the payload and checks what comes back. Dials the control socket.
    Origin,
    /// Receives the payload, checks it, and writes it straight back. Listens.
    Echo,
}

struct Config {
    role: Role,
    /// `host:port` of the echo side. The origin dials it; the echo side binds `0.0.0.0:port`
    /// from it, so both processes can be handed the identical value.
    control_addr: String,
    sizes: Vec<u64>,
    iterations: usize,
    warmup: usize,
    agent_name: String,
    engine_dir: PathBuf,
    gpu_memory_limit: String,
    staging_bytes: String,
    /// Physical GPU ordinal for NVLink counter sampling. NVML ignores `CUDA_VISIBLE_DEVICES`,
    /// so this is the box-level ordinal, not the process-visible 0.
    nvlink_gpu: u32,
    /// Exchange through the engine's real staging arena instead of a standalone `cudaMalloc`.
    /// Off by default: see [`Endpoint`].
    use_engine: bool,
    /// How the standalone arena is allocated. `cudamalloc` reproduces the engine's arena;
    /// `fabric` allocates through the VMM API so the pages can be exported over multi-node
    /// NVLink. See [`Endpoint`].
    arena_kind: ArenaKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArenaKind {
    CudaMalloc,
    Fabric,
}

impl Config {
    fn from_env() -> Self {
        let role = match env_required("NIXL_ECHO_ROLE").as_str() {
            "origin" => Role::Origin,
            "echo" => Role::Echo,
            other => panic!("NIXL_ECHO_ROLE must be 'origin' or 'echo', not '{other}'"),
        };
        let role_name = if role == Role::Origin {
            "origin"
        } else {
            "echo"
        };
        let sizes = env_or("NIXL_ECHO_SIZES", DEFAULT_SIZES)
            .split(',')
            .map(|value| value.trim().parse::<u64>().expect("payload size in bytes"))
            .collect::<Vec<_>>();
        assert!(!sizes.is_empty(), "at least one payload size");
        assert!(
            sizes.iter().all(|size| size % 8 == 0),
            "payload sizes must be multiples of 8 (int64 verification lanes)"
        );
        Self {
            role,
            control_addr: env_or("NIXL_ECHO_CONTROL", "127.0.0.1:18090"),
            sizes,
            iterations: env_or("NIXL_ECHO_ITERATIONS", "10").parse().unwrap(),
            warmup: env_or("NIXL_ECHO_WARMUP", "3").parse().unwrap(),
            // Agent names must be unique across the pair and are what the peer addresses its
            // WRITEs to; production uses `{advertise_host}:{brpc_port}`, so mirror that shape.
            agent_name: env_or("NIXL_ECHO_AGENT", &format!("{}:{role_name}", hostname())),
            engine_dir: PathBuf::from(env_or(
                "NIXL_ECHO_ENGINE_DIR",
                &format!("/tmp/sirius-nixl-echo-{role_name}"),
            )),
            gpu_memory_limit: env_or("NIXL_ECHO_GPU_MEMORY_LIMIT", "8GiB"),
            staging_bytes: env_or("SIRIUS_EXCHANGE_STAGING_BYTES", "2GiB"),
            nvlink_gpu: env_or("NIXL_ECHO_NVLINK_GPU", "0").parse().unwrap(),
            use_engine: env_or("NIXL_ECHO_USE_ENGINE", "0") == "1",
            arena_kind: match env_or("NIXL_ECHO_ARENA", "cudamalloc").as_str() {
                "cudamalloc" => ArenaKind::CudaMalloc,
                "fabric" => ArenaKind::Fabric,
                other => panic!("NIXL_ECHO_ARENA must be 'cudamalloc' or 'fabric', not '{other}'"),
            },
        }
    }

    fn role_name(&self) -> &'static str {
        if self.role == Role::Origin {
            "origin"
        } else {
            "echo"
        }
    }

    /// The echo side binds every interface on the control port: the origin reaches it over
    /// whichever of the host's several fabrics is routable, which this process cannot know.
    fn bind_addr(&self) -> String {
        let port = self
            .control_addr
            .rsplit_once(':')
            .expect("NIXL_ECHO_CONTROL is 'host:port'")
            .1;
        format!("0.0.0.0:{port}")
    }
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

// ---------------------------------------------------------------------------------------------
// Bring-up (identical on both roles).
// ---------------------------------------------------------------------------------------------

/// One side's nixl agent plus the registered VRAM it exchanges through.
///
/// The arena is a plain `cudaMalloc` by default. That is not a simplification: the engine's
/// `exchange_staging_arena` *is* one `cudaMalloc` with 256-byte-aligned bump leases (pool memory
/// would silently degrade ~220x over cuda_ipc — finding F1), so registering an equivalent
/// allocation exercises the same transport against the same kind of memory. What it buys is
/// independence: this test then has no opinion about the engine's DuckDB version or its RMM
/// pool, and bring-up is one allocation instead of a full engine start.
///
/// `NIXL_ECHO_ARENA=fabric` allocates that standalone arena through the VMM API instead, which
/// is what a cross-node NVLink (MNNVL) transfer requires — see [`cuda_vmm`]. Comparing the two
/// is the point: same code, same transport, different allocator, and only one of them can leave
/// the box over NVLink.
///
/// `NIXL_ECHO_USE_ENGINE=1` swaps in the engine's real arena, reached through
/// [`TransportState::bring_up`] exactly as production does.
enum Endpoint {
    Standalone {
        agent: Agent,
        agent_name: String,
        local_md: Vec<u8>,
        base: u64,
        capacity: u64,
        kind: ArenaKind,
        /// Bump cursor, the arena's own lease policy.
        cursor: u64,
        /// Keeps the arena registered for the agent's lifetime; dropped before the arena is.
        _registration: RegistrationHandle,
    },
    Engine {
        state: TransportState,
        /// Field order matters: `state` (agent + registration) drops before the engine that
        /// owns the arena it registered.
        executor: Arc<dyn FragmentExecutor>,
        leases: Vec<u64>,
    },
}

impl Endpoint {
    fn agent(&self) -> &Agent {
        match self {
            Endpoint::Standalone { agent, .. } => agent,
            Endpoint::Engine { state, .. } => &state.agent,
        }
    }

    fn agent_name(&self) -> &str {
        match self {
            Endpoint::Standalone { agent_name, .. } => agent_name,
            Endpoint::Engine { state, .. } => &state.agent_name,
        }
    }

    fn local_md(&self) -> &[u8] {
        match self {
            Endpoint::Standalone { local_md, .. } => local_md,
            Endpoint::Engine { state, .. } => &state.local_md,
        }
    }

    /// Carves `bytes` out of the arena, returning an absolute device address. Leases never
    /// overlap, so the origin's outbound and return buffers cannot alias.
    fn lease(&mut self, bytes: u64) -> u64 {
        match self {
            Endpoint::Standalone {
                base,
                capacity,
                cursor,
                ..
            } => {
                let offset = *cursor;
                let end = offset + bytes;
                assert!(
                    end <= *capacity,
                    "staging arena of {capacity} bytes cannot serve a lease to {end}; raise \
                     SIRIUS_EXCHANGE_STAGING_BYTES"
                );
                // 256-byte alignment, matching the engine arena's lease granularity.
                *cursor = end.next_multiple_of(256);
                *base + offset
            }
            Endpoint::Engine {
                state,
                executor,
                leases,
            } => {
                let offset = executor.staging_lease(bytes).expect("lease staging bytes");
                leases.push(offset);
                state.staging_base + offset
            }
        }
    }

    /// Releases the leases and the arena. Explicit rather than a `Drop` impl because the agent
    /// must go first, and only the caller knows it is done posting transfers.
    fn shutdown(self) {
        match self {
            Endpoint::Standalone {
                agent, base, kind, ..
            } => {
                drop(agent);
                match kind {
                    ArenaKind::CudaMalloc => cuda::runtime().free_device(base),
                    // The VMM mapping is left in place: unmapping needs the handle and the
                    // reservation back, and the process is about to exit anyway.
                    ArenaKind::Fabric => {}
                }
            }
            Endpoint::Engine {
                state,
                executor,
                leases,
            } => {
                for offset in leases {
                    executor.staging_release(offset).expect("release lease");
                }
                drop(state);
                drop(executor);
            }
        }
    }
}

/// Brings up the arena and the nixl agent registered over it. Outside every timed loop.
fn bring_up(config: &Config) -> Endpoint {
    let capacity = parse_bytes(&config.staging_bytes);
    if config.use_engine {
        return bring_up_engine(config);
    }

    let cuda = cuda::runtime();
    cuda.set_device_zero();
    let (base, capacity) = match config.arena_kind {
        ArenaKind::CudaMalloc => (cuda.alloc_device(capacity as usize), capacity),
        ArenaKind::Fabric => {
            let (base, size) = cuda_vmm::alloc_fabric(capacity as usize);
            (base, size as u64)
        }
    };
    let (agent, registration, local_md) =
        bring_up_agent(&config.agent_name, base, capacity).expect("bring up the nixl agent");
    eprintln!(
        "[{}] host={} agent={} standalone {} arena: base={base:#x} capacity={} MiB",
        config.role_name(),
        hostname(),
        config.agent_name,
        match config.arena_kind {
            ArenaKind::CudaMalloc => "cudaMalloc",
            ArenaKind::Fabric => "fabric (VMM)",
        },
        capacity >> 20
    );
    Endpoint::Standalone {
        agent,
        agent_name: config.agent_name.clone(),
        local_md,
        base,
        capacity,
        kind: config.arena_kind,
        cursor: 0,
        _registration: registration,
    }
}

/// The `NIXL_ECHO_USE_ENGINE=1` path: the engine's own staging arena, through the production
/// [`TransportState::bring_up`].
fn bring_up_engine(config: &Config) -> Endpoint {
    // The arena is constructed at context bring-up, only when this is set. The engine has not
    // started yet, so nothing else reads the environment.
    // SAFETY: single-threaded, before any engine or transport thread exists.
    unsafe {
        std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", &config.staging_bytes);
    }
    std::fs::create_dir_all(&config.engine_dir).expect("create engine dir");

    // Cap the engine's GPU budget: this test needs the arena, not an RMM pool over the whole
    // device, and a small pool keeps bring-up quick.
    let yaml = derive_sirius_config_yaml(
        Some(&config.gpu_memory_limit),
        None,
        None,
        &config.engine_dir,
        None,
    )
    .expect("a gpu memory limit yields a derived config");
    let config_path = config.engine_dir.join("sirius-echo.yaml");
    std::fs::write(&config_path, yaml).expect("write derived engine config");

    let executor: Arc<dyn FragmentExecutor> = Arc::new(
        SiriusEngine::start(EngineSettings {
            config: Some(config_path),
            engine_dir: config.engine_dir.clone(),
            // CUDA_VISIBLE_DEVICES is set by the driver script, one GPU per process.
            gpu_device: None,
        })
        .expect("bring up the sirius engine"),
    );
    let (base, capacity) = executor.staging_info().expect("staging arena info");
    eprintln!(
        "[{}] host={} agent={} engine arena: base={base:#x} capacity={} MiB",
        config.role_name(),
        hostname(),
        config.agent_name,
        capacity >> 20
    );

    let state = TransportState::bring_up(executor.clone(), config.agent_name.clone())
        .expect("bring up the nixl agent over the staging arena");
    Endpoint::Engine {
        state,
        executor,
        leases: Vec::new(),
    }
}

/// Parses `2GiB` / `512MiB` / a bare byte count.
fn parse_bytes(value: &str) -> u64 {
    let trimmed = value.trim();
    for (suffix, scale) in [("GiB", 1u64 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)] {
        if let Some(number) = trimmed.strip_suffix(suffix) {
            return number.trim().parse::<u64>().expect("byte count") * scale;
        }
    }
    trimmed.parse::<u64>().expect("byte count")
}

/// What the peer told us on first contact.
struct Peer {
    name: String,
    /// The address in the peer's registered arena that *we* WRITE into. Each side announces the
    /// buffer it wants to receive in, which makes the handshake symmetric.
    inbox_addr: u64,
}

/// Exchanges `(agent name, inbox address)` and the nixl md blob, then loads the peer's md.
///
/// This is `rpc_exchange_md`'s payload plus `rpc_request_lease`'s reply, carried on the control
/// socket instead of brpc. The origin speaks first, mirroring the request/reply shape of
/// `ensure_session` -> `exchange_md`.
fn handshake(control: &mut TcpStream, endpoint: &Endpoint, inbox_addr: u64, first: bool) -> Peer {
    let hello = format!("{} {inbox_addr}", endpoint.agent_name());
    let announce = |control: &mut TcpStream| {
        send_text(control, &hello);
        send_frame(control, endpoint.local_md());
    };
    let listen = |control: &mut TcpStream| {
        let text = recv_text(control);
        let (name, addr) = text.split_once(' ').expect("hello is 'name addr'");
        (
            Peer {
                name: name.to_string(),
                inbox_addr: addr.parse::<u64>().expect("inbox address"),
            },
            recv_frame(control),
        )
    };
    let (peer, metadata) = if first {
        announce(control);
        listen(control)
    } else {
        let peer = listen(control);
        announce(control);
        peer
    };

    // Both sides load: leg 1 is origin -> echo and leg 2 is echo -> origin, so each agent must
    // be able to address the other. `exchange_md` does the same on the brpc handler side.
    let loaded = endpoint
        .agent()
        .load_remote_md(&metadata)
        .expect("load the peer agent's nixl metadata");
    assert_eq!(
        loaded, peer.name,
        "peer announced '{}' but its metadata decodes to '{loaded}'",
        peer.name
    );
    peer
}

// ---------------------------------------------------------------------------------------------
// Results.
// ---------------------------------------------------------------------------------------------

struct PhaseResult {
    nbytes: u64,
    /// Full round trip measured at the origin: leg 1 + a control round trip + leg 2.
    round_trip: Vec<f64>,
    /// `write_and_wait`'s own post->DONE duration for leg 1 (origin -> echo).
    leg1: Vec<f64>,
    /// The same, for leg 2 (echo -> origin), as reported by the echo host.
    leg2: Vec<f64>,
    origin_nvlink_tx: u64,
    echo_nvlink_tx: u64,
    nvlink_available: bool,
    /// The echo host's verdict on the bytes it received (leg 1).
    echo_verdict: Result<String, String>,
    /// The origin's verdict on the bytes that came back (leg 2).
    origin_verdict: Result<String, String>,
}

impl PhaseResult {
    fn median(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return f64::NAN;
        }
        let mut values = samples.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        percentile(&values, 0.5)
    }

    /// One-way GB/s at `seconds`, for a single leg of `nbytes`.
    fn gbps(&self, seconds: f64) -> f64 {
        self.nbytes as f64 / seconds / 1e9
    }

    fn verified(&self) -> bool {
        self.echo_verdict.is_ok() && self.origin_verdict.is_ok()
    }
}

fn report(config: &Config, results: &[PhaseResult]) {
    let mut log = String::new();
    log.push_str(&format!(
        "nixl two-node GPU echo — {} <-> peer — UCX_TLS={} UCX_NET_DEVICES={} — {} timed + {} \
         warmup\n",
        hostname(),
        env_or("UCX_TLS", "(unset)"),
        env_or("UCX_NET_DEVICES", "(unset)"),
        config.iterations,
        config.warmup
    ));
    log.push_str(
        "payload      round trip   leg1 (out)   leg2 (back)   leg1 GB/s   leg2 GB/s   nvlink Tx \
         (origin/echo)  verified\n",
    );
    for result in results {
        let leg1 = PhaseResult::median(&result.leg1);
        let leg2 = PhaseResult::median(&result.leg2);
        log.push_str(&format!(
            "{:>10}  {:>9.3} ms  {:>8.3} ms  {:>9.3} ms  {:>9.2}  {:>9.2}  {:>10} / {:<10}  {}\n",
            human_bytes(result.nbytes),
            PhaseResult::median(&result.round_trip) * 1e3,
            leg1 * 1e3,
            leg2 * 1e3,
            result.gbps(leg1),
            result.gbps(leg2),
            human_bytes(result.origin_nvlink_tx),
            human_bytes(result.echo_nvlink_tx),
            match (&result.echo_verdict, &result.origin_verdict) {
                (Ok(_), Ok(_)) => "yes (both legs)".to_string(),
                (Err(detail), _) => format!("NO, leg1: {detail}"),
                (_, Err(detail)) => format!("NO, leg2: {detail}"),
            }
        ));
    }
    print!("{log}");
    eprint!("{log}");
}

// ---------------------------------------------------------------------------------------------
// The test entry point.
// ---------------------------------------------------------------------------------------------

/// Two-node GPU buffer echo: a buffer goes out over nixl and comes back, byte-verified on both
/// hosts.
///
/// Run one process per host (`NIXL_ECHO_ROLE=echo` first, then `origin`), each pinned to one GPU
/// with `CUDA_VISIBLE_DEVICES`; `scripts/nixl-echo-2node.sh` does exactly that.
#[test]
#[ignore = "two-host GPU + libnixl echo: driven by scripts/nixl-echo-2node.sh"]
fn two_node_gpu_echo() {
    let config = Config::from_env();
    match config.role {
        Role::Origin => run_origin(&config),
        Role::Echo => run_echo(&config),
    }
}

fn run_origin(config: &Config) {
    let max_bytes = *config.sizes.iter().max().unwrap();
    let mut endpoint = bring_up(config);
    let cuda = cuda::runtime();
    cuda.set_device_zero();

    // Two leases for the whole run — leasing, like registration, stays outside the timed loop.
    // `out` holds the payload we send; `back` is where the echo host writes it back, so it is
    // the address we announce. They must not alias, or leg 2 would land on its own source.
    let out_addr = endpoint.lease(max_bytes);
    let back_addr = endpoint.lease(max_bytes);
    assert_ne!(out_addr, back_addr, "two live leases must not alias");

    let mut control = connect_retry(&config.control_addr, PEER_DEADLINE);
    tune_socket(&control, 1 << 20);
    let peer = handshake(&mut control, &endpoint, back_addr, true);
    eprintln!(
        "[origin] peer={} out={out_addr:#x} back={back_addr:#x} peer_inbox={:#x}",
        peer.name, peer.inbox_addr
    );

    let mut payload = vec![0u8; max_bytes as usize];
    let mut returned = vec![0u8; max_bytes as usize];
    let mut results = Vec::new();

    for &nbytes in &config.sizes {
        let n = nbytes as usize;
        let seed = phase_seed("echo", nbytes);

        // Announce the phase; the echo host zeroes its inbox and replies. Zeroing on both sides
        // is what stops a stale buffer from faking a pass.
        send_text(&mut control, &format!("phase {nbytes} {seed}"));
        expect_text(&mut control, "ready");

        fill_pattern(&mut payload[..n], seed);
        cuda.copy_to_device(out_addr, &payload[..n]);
        cuda.memset_zero(back_addr, n);

        let before = nvlink_counters(config.nvlink_gpu);
        let mut round_trip = Vec::with_capacity(config.iterations);
        let mut leg1 = Vec::with_capacity(config.iterations);
        let mut leg2 = Vec::with_capacity(config.iterations);

        for step in 0..(config.warmup + config.iterations) {
            send_text(&mut control, &format!("go {nbytes}"));
            let start = Instant::now();
            let out = write_and_wait(
                endpoint.agent(),
                &peer.name,
                out_addr,
                peer.inbox_addr,
                nbytes,
            )
            .expect("leg 1: nixl WRITE to the echo host");
            // The echo host cannot see the WRITE land (it is one-sided), so tell it.
            send_text(&mut control, &format!("sent {nbytes}"));
            let echoed = recv_text(&mut control);
            let elapsed = start.elapsed();
            let back_nanos: u64 = echoed
                .strip_prefix("echoed ")
                .expect("echo host acknowledges with 'echoed <nanos>'")
                .parse()
                .expect("leg 2 duration in nanoseconds");
            if step >= config.warmup {
                round_trip.push(elapsed.as_secs_f64());
                leg1.push(out.as_secs_f64());
                leg2.push(Duration::from_nanos(back_nanos).as_secs_f64());
            }
        }

        let after = nvlink_counters(config.nvlink_gpu);

        // Both sides check their own destination buffer.
        send_text(&mut control, &format!("verify {nbytes}"));
        let echo_verdict = parse_verdict(&recv_text(&mut control));
        let echo_tx: u64 = recv_text(&mut control).parse().unwrap_or(0);

        cuda.copy_to_host(&mut returned[..n], back_addr);
        let origin_verdict = check_pattern(&returned[..n], seed)
            .map(|()| format!("{n}/{n} bytes match seed {seed}"));

        let result = PhaseResult {
            nbytes,
            round_trip,
            leg1,
            leg2,
            origin_nvlink_tx: after.tx.saturating_sub(before.tx),
            echo_nvlink_tx: echo_tx,
            nvlink_available: before.available && after.available,
            echo_verdict,
            origin_verdict,
        };
        eprintln!(
            "[origin] {nbytes}B: round trip median {:.3} ms, leg1 {:.2} GB/s, leg2 {:.2} GB/s, \
             verified {} (nvlink counters {})",
            PhaseResult::median(&result.round_trip) * 1e3,
            result.gbps(PhaseResult::median(&result.leg1)),
            result.gbps(PhaseResult::median(&result.leg2)),
            result.verified(),
            if result.nvlink_available {
                "available"
            } else {
                "unavailable"
            }
        );
        results.push(result);
    }

    send_text(&mut control, "done");
    report(config, &results);
    endpoint.shutdown();

    for result in &results {
        if let Err(detail) = &result.echo_verdict {
            panic!(
                "leg 1 (origin -> echo) corrupted at {} bytes: {detail}",
                result.nbytes
            );
        }
        if let Err(detail) = &result.origin_verdict {
            panic!(
                "leg 2 (echo -> origin) corrupted at {} bytes: {detail}",
                result.nbytes
            );
        }
    }
}

fn run_echo(config: &Config) {
    let max_bytes = *config.sizes.iter().max().unwrap();
    let mut endpoint = bring_up(config);
    let cuda = cuda::runtime();
    cuda.set_device_zero();

    // One lease: the origin writes into it, and leg 2 reads straight back out of it, so the
    // bytes the origin verifies are the bytes this host received.
    let inbox_addr = endpoint.lease(max_bytes);

    let bind_addr = config.bind_addr();
    let listener = TcpListener::bind(&bind_addr).expect("bind the control port");
    eprintln!("[echo] listening on {bind_addr} — inbox={inbox_addr:#x}");
    let (mut control, from) = listener.accept().expect("accept control");
    eprintln!("[echo] origin connected from {from}");
    tune_socket(&control, 1 << 20);

    let peer = handshake(&mut control, &endpoint, inbox_addr, false);
    eprintln!(
        "[echo] peer={} inbox={inbox_addr:#x} peer_inbox={:#x}",
        peer.name, peer.inbox_addr
    );

    let mut received = vec![0u8; max_bytes as usize];

    loop {
        let message = recv_text(&mut control);
        if message == "done" {
            break;
        }
        let tokens: Vec<&str> = message.split_whitespace().collect();
        assert_eq!(
            tokens[0], "phase",
            "expected a phase announcement, got '{message}'"
        );
        let nbytes: u64 = tokens[1].parse().unwrap();
        let seed: u64 = tokens[2].parse().unwrap();
        let n = nbytes as usize;

        cuda.memset_zero(inbox_addr, n);
        send_text(&mut control, "ready");

        let before = nvlink_counters(config.nvlink_gpu);
        for _ in 0..(config.warmup + config.iterations) {
            expect_text(&mut control, &format!("go {nbytes}"));
            // Leg 1 lands in registered device memory without this host's participation; the
            // origin's "sent" is the only signal it is complete, same as `transmit_packed`.
            expect_text(&mut control, &format!("sent {nbytes}"));
            let back = write_and_wait(
                endpoint.agent(),
                &peer.name,
                inbox_addr,
                peer.inbox_addr,
                nbytes,
            )
            .expect("leg 2: nixl WRITE back to the origin host");
            send_text(&mut control, &format!("echoed {}", back.as_nanos()));
        }
        let after = nvlink_counters(config.nvlink_gpu);

        expect_text(&mut control, &format!("verify {nbytes}"));
        cuda.copy_to_host(&mut received[..n], inbox_addr);
        match check_pattern(&received[..n], seed) {
            Ok(()) => send_text(
                &mut control,
                &format!("verified {n}/{n} bytes match seed {seed}"),
            ),
            Err(detail) => send_text(&mut control, &format!("corrupt {detail}")),
        }
        send_text(
            &mut control,
            &after.tx.saturating_sub(before.tx).to_string(),
        );
        eprintln!("[echo] {nbytes}B phase complete");
    }

    endpoint.shutdown();
}

/// Splits the echo host's `verified …` / `corrupt …` reply into a result.
fn parse_verdict(verdict: &str) -> Result<String, String> {
    match verdict.split_once(' ') {
        Some(("verified", detail)) => Ok(detail.to_string()),
        Some(("corrupt", detail)) => Err(detail.to_string()),
        _ => Err(format!("unparseable verdict '{verdict}'")),
    }
}
