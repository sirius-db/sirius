//! Standalone NVLink/nixl benchmark driven entirely by the CN's own transport primitives.
//!
//! Mirrors `nixl-nvlink-repro/benchmark/bench.py` (the Python/PyTorch/nixl-wheel harness that
//! produced `results/report-20260808.md` on 4x H100) but replaces every nixl call with the
//! production code path this crate actually uses:
//!
//!   * agent bring-up + UCX backend + staging-arena VRAM registration — [`TransportState::bring_up`]
//!     (which is `bring_up_agent` verbatim: `Agent::new` -> `get_plugin_params("UCX")` ->
//!     `create_backend` -> `OptArgs::add_backend` -> `register_memory(ArenaRegion)` ->
//!     `get_local_md`), holding the `RegistrationHandle` for the run.
//!   * metadata handshake — `Agent::load_remote_md`, the same call `exchange_md` /
//!     `ensure_session` make; only the brpc envelope is replaced by the control socket (see
//!     "Substitutions" below).
//!   * the transfer itself — [`write_and_wait`] verbatim, the exact function `bandwidth_canary`
//!     and `send_fragment` post their WRITEs with.
//!   * the memory — `FragmentExecutor::staging_{info,lease,release}`, i.e. the engine's
//!     `exchange_staging_arena` (one plain `cudaMalloc`, 256-byte-aligned bump leases).
//!
//! Substitutions, and why (nothing else is substituted):
//!
//!   * **brpc control plane.** `rpc_exchange_md` / `rpc_request_lease` / `rpc_transmit` only
//!     carry three payloads: the md blob, the peer's `(remote_addr, offset)`, and a "it landed"
//!     notification. Driving them needs a running `compute_node_service` on both sides, i.e. a
//!     StarRocks cluster, which this benchmark is forbidden to start. A loopback TCP control
//!     socket carries exactly those three payloads instead — which is also what the reference
//!     harness did, so the timed window stays comparable. The nixl calls underneath are
//!     untouched.
//!   * **`send_fragment`** is not used: it needs a parked fragment output from a real query.
//!     The bench leases raw bytes and WRITEs them, which is the same lease -> WRITE -> ack
//!     sequence with the pack/unpack elided.
//!   * **CUDA runtime** (`cudaMemset`/`cudaMemcpy`/`cudaHostAlloc`) is reached by `dlopen`ing
//!     libcudart, because the Rust side of the crate has no device-memory API and the reference
//!     harness's byte verification requires one. No nixl/UCX call goes through it.
//!
//! The scaffolding around the measurement — that CUDA shim, the control-socket framing, the
//! verification pattern, the NVLink counters and the percentile rule — lives in
//! [`two_node_harness`](super::two_node_harness), shared with the two-node echo test. The
//! definitions moved there unchanged, so these numbers stay comparable across the move.
//!
//! Two processes, one GPU each (`CUDA_VISIBLE_DEVICES` per process), because `ArenaRegion`'s
//! `NixlDescriptor::device_id()` is hardcoded to 0 and `write_and_wait` passes dev_id 0 to
//! `add_desc`: that is only correct when the process sees exactly one device, which is the
//! invariant production relies on (one CN per GPU). The in-process two-agent smoke test in this
//! module's sibling `tests` shares one arena and therefore never crosses NVLink.
//!
//! Run it through `scratchpad/nixl-bench/run.sh`; every knob is an environment variable.

#![allow(clippy::too_many_arguments)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::agent_tier::{TransportState, write_and_wait};
use super::two_node_harness::{
    check_pattern, connect_retry, cuda, env_or, env_required, expect_text, fill_pattern,
    human_bytes, nvlink_counters, percentile, phase_seed, recv_frame, recv_text, send_frame,
    send_text, tune_socket,
};
use crate::engine::SiriusEngine;
use crate::engine_settings::{EngineSettings, derive_sirius_config_yaml};
use crate::fragment_executor::FragmentExecutor;

/// Reference harness threshold (`report.py:NVLINK_CARRIED`): a phase counts as NVLink-carried
/// when the sender's Tx counter delta covers at least this fraction of the bytes moved.
const NVLINK_CARRIED: f64 = 0.5;
/// Payload sizes of the reference sweep: 1 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB.
const DEFAULT_SIZES: &str = "1048576,16777216,67108864,268435456,1073741824";

// ---------------------------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------------------------

struct Config {
    role: String,
    control_addr: String,
    data_addr: String,
    sizes: Vec<u64>,
    iterations: usize,
    warmup: usize,
    methods: Vec<String>,
    sender_gpu: u32,
    receiver_gpu: u32,
    profile: String,
    agent_name: String,
    engine_dir: PathBuf,
    gpu_memory_limit: String,
    staging_bytes: String,
    socket_buffer_bytes: i32,
    output_json: PathBuf,
    output_log: PathBuf,
}

impl Config {
    fn from_env() -> Self {
        let role = env_required("NIXL_BENCH_ROLE");
        let sizes = env_or("NIXL_BENCH_SIZES", DEFAULT_SIZES)
            .split(',')
            .map(|value| value.trim().parse::<u64>().expect("payload size in bytes"))
            .collect::<Vec<_>>();
        assert!(!sizes.is_empty(), "at least one payload size");
        assert!(
            sizes.iter().all(|size| size % 8 == 0),
            "payload sizes must be multiples of 8 (int64 verification lanes)"
        );
        Self {
            control_addr: env_or("NIXL_BENCH_CONTROL", "127.0.0.1:18070"),
            data_addr: env_or("NIXL_BENCH_DATA", "127.0.0.1:18071"),
            sizes,
            iterations: env_or("NIXL_BENCH_ITERATIONS", "10").parse().unwrap(),
            warmup: env_or("NIXL_BENCH_WARMUP", "3").parse().unwrap(),
            methods: env_or("NIXL_BENCH_METHODS", "tcp-naive,nixl")
                .split(',')
                .map(|method| method.trim().to_string())
                .filter(|method| !method.is_empty())
                .collect(),
            sender_gpu: env_or("NIXL_BENCH_SENDER_GPU", "0").parse().unwrap(),
            receiver_gpu: env_or("NIXL_BENCH_RECEIVER_GPU", "1").parse().unwrap(),
            profile: env_or("NIXL_BENCH_PROFILE", "nvlink"),
            agent_name: env_or("NIXL_BENCH_AGENT", &format!("bench-{role}")),
            engine_dir: PathBuf::from(env_or(
                "NIXL_BENCH_ENGINE_DIR",
                &format!("/tmp/sirius-nixl-bench-{role}"),
            )),
            gpu_memory_limit: env_or("NIXL_BENCH_GPU_MEMORY_LIMIT", "8GiB"),
            staging_bytes: env_or("SIRIUS_EXCHANGE_STAGING_BYTES", "2GiB"),
            socket_buffer_bytes: env_or("NIXL_BENCH_SOCKET_BUFFER", "16777216")
                .parse()
                .unwrap(),
            output_json: PathBuf::from(env_or(
                "NIXL_BENCH_OUT_JSON",
                "/home/prestouser/aocsa/benchmark-results/nixl-bench.json",
            )),
            output_log: PathBuf::from(env_or(
                "NIXL_BENCH_OUT_LOG",
                "/home/prestouser/aocsa/benchmark-results/nixl-bench.log",
            )),
            role,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Engine + transport bring-up (identical on both roles).
// ---------------------------------------------------------------------------------------------

/// Brings up the engine (which allocates the `cudaMalloc` staging arena), then the nixl agent
/// with the arena registered as VRAM — `TransportState::bring_up`, the production call.
///
/// Everything here is outside every timed loop, as the reference harness requires.
fn bring_up(config: &Config) -> (Arc<dyn FragmentExecutor>, TransportState) {
    // The arena is constructed at context bring-up, only when this is set. The engine has not
    // started yet, so nothing else reads the environment.
    // SAFETY: single-threaded, before any engine or transport thread exists.
    unsafe {
        std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", &config.staging_bytes);
    }
    std::fs::create_dir_all(&config.engine_dir).expect("create engine dir");

    // Cap the engine's GPU budget: the benchmark needs the arena, not an RMM pool over the whole
    // 185 GiB device, and a small pool keeps bring-up quick and the device quiet.
    // No `cpu_affinity`: this harness measures the GPU-to-GPU transport, so thread placement is
    // left to whatever pins the process (`numactl`, a cpuset) and its numbers stay comparable
    // with runs recorded before the CN started emitting affinity.
    let yaml = derive_sirius_config_yaml(
        Some(&config.gpu_memory_limit),
        None,
        None,
        &config.engine_dir,
        None,
    )
    .expect("a gpu memory limit yields a derived config");
    let config_path = config.engine_dir.join("sirius-bench.yaml");
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
        "[{}] staging arena: base={base:#x} capacity={capacity} ({} MiB)",
        config.role,
        capacity >> 20
    );

    let state = TransportState::bring_up(executor.clone(), config.agent_name.clone())
        .expect("bring up the nixl agent over the staging arena");
    (executor, state)
}

/// One side of the metadata exchange: `(peer agent name, peer staging lease address, peer md)`.
struct Peer {
    name: String,
    lease_addr: u64,
    metadata: Vec<u8>,
}

/// Sends `(agent_name, staging lease address)` and the local md blob; returns the peer's.
///
/// This is `rpc_exchange_md`'s payload plus `rpc_request_lease`'s reply, carried on the control
/// socket instead of brpc. `speak_first` is true on the sender, mirroring the request/reply
/// shape of `ensure_session` -> `exchange_md`.
fn handshake(
    control: &mut TcpStream,
    speak_first: bool,
    agent_name: &str,
    lease_addr: u64,
    local_md: &[u8],
) -> Peer {
    let hello = format!("{agent_name} {lease_addr}");
    let announce = |control: &mut TcpStream| {
        send_text(control, &hello);
        send_frame(control, local_md);
    };
    let listen = |control: &mut TcpStream| {
        let text = recv_text(control);
        let (name, addr) = text.split_once(' ').expect("hello is 'name addr'");
        Peer {
            name: name.to_string(),
            lease_addr: addr.parse::<u64>().expect("lease address"),
            metadata: recv_frame(control),
        }
    };
    if speak_first {
        announce(control);
        listen(control)
    } else {
        let peer = listen(control);
        announce(control);
        peer
    }
}

// ---------------------------------------------------------------------------------------------
// Results.
// ---------------------------------------------------------------------------------------------

struct PhaseResult {
    method: String,
    nbytes: u64,
    /// Reference-comparable window: transfer + one control round trip, per iteration.
    samples: Vec<f64>,
    /// `write_and_wait`'s own post->DONE duration (nixl only); a strictly narrower window.
    post_to_done: Vec<f64>,
    bytes_moved: u64,
    nvlink_sender_tx: u64,
    nvlink_receiver_rx: u64,
    nvlink_available: bool,
    verified: bool,
    verify_detail: String,
}

impl PhaseResult {
    fn sorted(&self) -> Vec<f64> {
        let mut values = self.samples.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values
    }

    fn median_s(&self) -> f64 {
        percentile(&self.sorted(), 0.5)
    }

    fn best_s(&self) -> f64 {
        self.sorted()[0]
    }

    fn p90_s(&self) -> f64 {
        percentile(&self.sorted(), 0.9)
    }

    fn gbps(&self, seconds: f64) -> f64 {
        self.nbytes as f64 / seconds / 1e9
    }

    fn post_to_done_median_s(&self) -> Option<f64> {
        if self.post_to_done.is_empty() {
            return None;
        }
        let mut values = self.post_to_done.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(percentile(&values, 0.5))
    }

    fn on_nvlink(&self) -> bool {
        self.nvlink_available
            && (self.nvlink_sender_tx as f64) >= NVLINK_CARRIED * self.bytes_moved as f64
    }
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_results(config: &Config, results: &[PhaseResult]) {
    if let Some(parent) = config.output_json.parent() {
        std::fs::create_dir_all(parent).expect("create results dir");
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"unix_time\": {stamp},\n"));
    json.push_str(&format!(
        "  \"profile\": \"{}\",\n",
        json_escape(&config.profile)
    ));
    json.push_str(&format!(
        "  \"ucx_tls\": \"{}\",\n",
        json_escape(&env_or("UCX_TLS", ""))
    ));
    json.push_str(&format!("  \"sender_gpu\": {},\n", config.sender_gpu));
    json.push_str(&format!("  \"receiver_gpu\": {},\n", config.receiver_gpu));
    json.push_str(&format!("  \"iterations\": {},\n", config.iterations));
    json.push_str(&format!("  \"warmup\": {},\n", config.warmup));
    json.push_str(&format!(
        "  \"staging_bytes\": \"{}\",\n",
        json_escape(&config.staging_bytes)
    ));
    json.push_str("  \"harness\": \"sirius-starrocks-cn nixl_transport primitives (Rust)\",\n");
    json.push_str("  \"results\": [\n");
    for (index, result) in results.iter().enumerate() {
        let samples = result
            .samples
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(", ");
        let post = result
            .post_to_done
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(", ");
        json.push_str("    {\n");
        json.push_str(&format!(
            "      \"method\": \"{}\",\n",
            json_escape(&result.method)
        ));
        json.push_str(&format!("      \"bytes\": {},\n", result.nbytes));
        json.push_str(&format!("      \"median_s\": {:.9},\n", result.median_s()));
        json.push_str(&format!("      \"best_s\": {:.9},\n", result.best_s()));
        json.push_str(&format!("      \"p90_s\": {:.9},\n", result.p90_s()));
        json.push_str(&format!(
            "      \"median_gbps\": {:.4},\n",
            result.gbps(result.median_s())
        ));
        json.push_str(&format!(
            "      \"best_gbps\": {:.4},\n",
            result.gbps(result.best_s())
        ));
        match result.post_to_done_median_s() {
            Some(seconds) => {
                json.push_str(&format!("      \"post_to_done_median_s\": {seconds:.9},\n"));
                json.push_str(&format!(
                    "      \"post_to_done_median_gbps\": {:.4},\n",
                    result.gbps(seconds)
                ));
            }
            None => json.push_str("      \"post_to_done_median_s\": null,\n"),
        }
        json.push_str(&format!("      \"bytes_moved\": {},\n", result.bytes_moved));
        json.push_str(&format!(
            "      \"nvlink_sender_tx\": {},\n",
            result.nvlink_sender_tx
        ));
        json.push_str(&format!(
            "      \"nvlink_receiver_rx\": {},\n",
            result.nvlink_receiver_rx
        ));
        json.push_str(&format!(
            "      \"nvlink_available\": {},\n",
            result.nvlink_available
        ));
        json.push_str(&format!("      \"on_nvlink\": {},\n", result.on_nvlink()));
        json.push_str(&format!(
            "      \"payload_verified\": {},\n",
            result.verified
        ));
        json.push_str(&format!(
            "      \"verify_detail\": \"{}\",\n",
            json_escape(&result.verify_detail)
        ));
        json.push_str(&format!("      \"samples_s\": [{samples}],\n"));
        json.push_str(&format!("      \"post_to_done_s\": [{post}]\n"));
        json.push_str(if index + 1 == results.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    json.push_str("  ]\n}\n");
    std::fs::write(&config.output_json, json).expect("write results json");

    let mut log = String::new();
    log.push_str(&format!(
        "nixl bench — profile={} UCX_TLS={} GPU{} -> GPU{} — {} timed + {} warmup\n",
        config.profile,
        env_or("UCX_TLS", "(unset)"),
        config.sender_gpu,
        config.receiver_gpu,
        config.iterations,
        config.warmup
    ));
    log.push_str(
        "method        payload      median GB/s   best GB/s   p90 ms   moved      nvlink Tx  \
         nvlink Rx  onNVLink  verified\n",
    );
    for result in results {
        log.push_str(&format!(
            "{:<13} {:>10}  {:>11.2}  {:>10.2}  {:>7.3}  {:>9}  {:>9}  {:>9}  {:<8}  {}\n",
            result.method,
            human_bytes(result.nbytes),
            result.gbps(result.median_s()),
            result.gbps(result.best_s()),
            result.p90_s() * 1e3,
            human_bytes(result.bytes_moved),
            human_bytes(result.nvlink_sender_tx),
            human_bytes(result.nvlink_receiver_rx),
            result.on_nvlink(),
            if result.verified {
                "yes".to_string()
            } else {
                format!("NO ({})", result.verify_detail)
            }
        ));
        if let Some(seconds) = result.post_to_done_median_s() {
            log.push_str(&format!(
                "              post->DONE only (write_and_wait): median {:.3} ms = {:.2} GB/s\n",
                seconds * 1e3,
                result.gbps(seconds)
            ));
        }
    }
    print!("{log}");
    std::fs::write(&config.output_log, &log).expect("write results log");
    eprintln!(
        "[sender] results written to {} and {}",
        config.output_json.display(),
        config.output_log.display()
    );
}

// ---------------------------------------------------------------------------------------------
// The test entry point.
// ---------------------------------------------------------------------------------------------

/// Two-process GPU-to-GPU benchmark of the CN's nixl transport primitives.
///
/// Run one process per role (`NIXL_BENCH_ROLE=receiver` first, then `sender`), each pinned to
/// one GPU with `CUDA_VISIBLE_DEVICES`; `scratchpad/nixl-bench/run.sh` does exactly that.
#[test]
#[ignore = "two-process GPU + libnixl benchmark: driven by scratchpad/nixl-bench/run.sh"]
fn gpu_to_gpu_bench() {
    let config = Config::from_env();
    match config.role.as_str() {
        "sender" => run_sender(&config),
        "receiver" => run_receiver(&config),
        other => panic!("NIXL_BENCH_ROLE must be 'sender' or 'receiver', not '{other}'"),
    }
}

fn run_sender(config: &Config) {
    let max_bytes = *config.sizes.iter().max().unwrap();
    let (executor, state) = bring_up(config);
    let cuda = cuda::runtime();
    cuda.set_device_zero();

    // One lease for the whole run: leasing, like registration, stays outside the timed loop.
    let lease_offset = executor
        .staging_lease(max_bytes)
        .expect("lease the sender staging buffer");
    let local_addr = state.staging_base + lease_offset;

    let mut control = connect_retry(&config.control_addr, Duration::from_secs(900));
    tune_socket(&control, 1 << 20);
    let mut data = connect_retry(&config.data_addr, Duration::from_secs(60));
    tune_socket(&data, config.socket_buffer_bytes);

    // Sender speaks first, mirroring `rpc_exchange_md`'s request/reply shape.
    let peer = handshake(
        &mut control,
        true,
        &state.agent_name,
        local_addr,
        &state.local_md,
    );
    let loaded = state
        .agent
        .load_remote_md(&peer.metadata)
        .expect("load the peer agent's nixl metadata");
    assert_eq!(
        loaded, peer.name,
        "peer announced '{}' but its metadata decodes to '{loaded}'",
        peer.name
    );
    let peer_name = peer.name;
    let peer_addr = peer.lease_addr;
    eprintln!(
        "[sender] agent={} peer={peer_name} local_addr={local_addr:#x} peer_addr={peer_addr:#x}",
        state.agent_name
    );

    let pinned = cuda.alloc_pinned(max_bytes as usize);
    let mut pattern = vec![0u8; max_bytes as usize];
    let mut results = Vec::new();

    for method in &config.methods {
        for &nbytes in &config.sizes {
            results.push(run_sender_phase(
                config,
                &state,
                &peer_name,
                local_addr,
                peer_addr,
                &mut control,
                &mut data,
                pinned,
                &mut pattern,
                method,
                nbytes,
            ));
        }
    }

    send_text(&mut control, "done");
    write_results(config, &results);

    executor
        .staging_release(lease_offset)
        .expect("release the sender staging lease");
    // The agent (and its arena registration) must go before the engine that owns the arena.
    drop(state);
    drop(executor);

    for result in &results {
        assert!(
            result.verified,
            "{} at {} bytes did not verify: {}",
            result.method, result.nbytes, result.verify_detail
        );
    }
}

fn run_sender_phase(
    config: &Config,
    state: &TransportState,
    peer_name: &str,
    local_addr: u64,
    peer_addr: u64,
    control: &mut TcpStream,
    data: &mut TcpStream,
    pinned: &mut [u8],
    pattern: &mut [u8],
    method: &str,
    nbytes: u64,
) -> PhaseResult {
    let seed = phase_seed(method, nbytes);
    let n = nbytes as usize;

    // Announce the phase; the receiver zeroes its lease and replies. Zeroing first is what stops
    // a stale buffer from faking a pass.
    send_text(control, &format!("phase {method} {nbytes} {seed}"));
    expect_text(control, "ready");

    let cuda = cuda::runtime();
    fill_pattern(&mut pattern[..n], seed);
    cuda.copy_to_device(local_addr, &pattern[..n]);

    let before_tx = nvlink_counters(config.sender_gpu);
    let before_rx = nvlink_counters(config.receiver_gpu);

    let mut samples = Vec::with_capacity(config.iterations);
    let mut post_to_done = Vec::with_capacity(config.iterations);
    for step in 0..(config.warmup + config.iterations) {
        send_text(control, &format!("go {nbytes}"));
        let start = Instant::now();
        let inner = match method {
            "nixl" => Some(
                write_and_wait(&state.agent, peer_name, local_addr, peer_addr, nbytes)
                    .expect("nixl WRITE"),
            ),
            "tcp-naive" => {
                cuda.copy_to_host(&mut pinned[..n], local_addr);
                data.write_all(&pinned[..n]).expect("tcp send");
                data.flush().expect("tcp flush");
                None
            }
            other => panic!("unknown method '{other}'"),
        };
        send_text(control, &format!("sent {nbytes}"));
        expect_text(control, &format!("landed {nbytes}"));
        let elapsed = start.elapsed();
        if step >= config.warmup {
            samples.push(elapsed.as_secs_f64());
            if let Some(inner) = inner {
                post_to_done.push(inner.as_secs_f64());
            }
        }
    }

    let after_tx = nvlink_counters(config.sender_gpu);
    let after_rx = nvlink_counters(config.receiver_gpu);

    send_text(control, &format!("verify {nbytes}"));
    let verdict = recv_text(control);
    let (verified, detail) = match verdict.split_once(' ') {
        Some(("verified", detail)) => (true, detail.to_string()),
        Some(("corrupt", detail)) => (false, detail.to_string()),
        _ => (false, format!("unparseable verdict '{verdict}'")),
    };

    let result = PhaseResult {
        method: method.to_string(),
        nbytes,
        samples,
        post_to_done,
        // Includes the warm-up, because the counter deltas bracket the whole phase — exactly
        // what `bench.py` charges to `bytes_moved`.
        bytes_moved: nbytes * (config.warmup + config.iterations) as u64,
        nvlink_sender_tx: after_tx.tx.saturating_sub(before_tx.tx),
        nvlink_receiver_rx: after_rx.rx.saturating_sub(before_rx.rx),
        nvlink_available: before_tx.available && after_tx.available,
        verified,
        verify_detail: detail,
    };
    eprintln!(
        "[sender] {method} {nbytes}B: median {:.3} ms = {:.2} GB/s, moved {}, nvlink Tx {}, \
         verified {}",
        result.median_s() * 1e3,
        result.gbps(result.median_s()),
        human_bytes(result.bytes_moved),
        human_bytes(result.nvlink_sender_tx),
        result.verified
    );
    result
}

fn run_receiver(config: &Config) {
    let max_bytes = *config.sizes.iter().max().unwrap();
    let (executor, state) = bring_up(config);
    let cuda = cuda::runtime();
    cuda.set_device_zero();

    let lease_offset = executor
        .staging_lease(max_bytes)
        .expect("lease the receiver staging buffer");
    let local_addr = state.staging_base + lease_offset;

    let control_listener = TcpListener::bind(&config.control_addr).expect("bind control port");
    let data_listener = TcpListener::bind(&config.data_addr).expect("bind data port");
    eprintln!(
        "[receiver] listening on {} / {} — local_addr={local_addr:#x}",
        config.control_addr, config.data_addr
    );
    let (mut control, _) = control_listener.accept().expect("accept control");
    tune_socket(&control, 1 << 20);
    let (mut data, _) = data_listener.accept().expect("accept data");
    tune_socket(&data, config.socket_buffer_bytes);

    let peer = handshake(
        &mut control,
        false,
        &state.agent_name,
        local_addr,
        &state.local_md,
    );
    // The receiver loads the sender's metadata too, exactly as `TransportState::exchange_md`
    // does on the brpc handler side.
    let loaded = state
        .agent
        .load_remote_md(&peer.metadata)
        .expect("load the peer agent's nixl metadata");
    assert_eq!(loaded, peer.name);
    eprintln!("[receiver] agent={} peer={}", state.agent_name, peer.name);

    let pinned = cuda.alloc_pinned(max_bytes as usize);
    let mut verify = vec![0u8; max_bytes as usize];

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
        let method = tokens[1].to_string();
        let nbytes: u64 = tokens[2].parse().unwrap();
        let seed: u64 = tokens[3].parse().unwrap();
        let n = nbytes as usize;

        // Zero the receiver's lease before the phase: the reference's guard against a stale
        // buffer passing verification.
        cuda.memset_zero(local_addr, n);
        send_text(&mut control, "ready");

        for _ in 0..(config.warmup + config.iterations) {
            expect_text(&mut control, &format!("go {nbytes}"));
            match method.as_str() {
                // The WRITE lands in registered device memory without the receiver's
                // participation — same as `transmit_packed` telling it the bytes arrived.
                "nixl" => {}
                "tcp-naive" => {
                    data.read_exact(&mut pinned[..n]).expect("tcp receive");
                    cuda.copy_to_device(local_addr, &pinned[..n]);
                }
                other => panic!("unknown method '{other}'"),
            }
            expect_text(&mut control, &format!("sent {nbytes}"));
            send_text(&mut control, &format!("landed {nbytes}"));
        }

        expect_text(&mut control, &format!("verify {nbytes}"));
        cuda.copy_to_host(&mut verify[..n], local_addr);
        match check_pattern(&verify[..n], seed) {
            Ok(()) => send_text(
                &mut control,
                &format!("verified {n}/{n} bytes match seed {seed}"),
            ),
            Err(detail) => send_text(&mut control, &format!("corrupt {detail}")),
        }
        eprintln!("[receiver] {method} {nbytes}B phase complete");
    }

    executor
        .staging_release(lease_offset)
        .expect("release the receiver staging lease");
    drop(state);
    drop(executor);
}
