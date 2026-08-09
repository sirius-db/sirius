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
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::agent_tier::{TransportState, write_and_wait};
use crate::engine::SiriusEngine;
use crate::engine_settings::{EngineSettings, derive_sirius_config_yaml};
use crate::fragment_executor::FragmentExecutor;

/// Reference harness constant (`bench.py:PATTERN_MULTIPLIER`): the int64 lane stride of the
/// verification pattern. Kept identical so a lane mismatch here means the same thing there.
const PATTERN_MULTIPLIER: u64 = 2654435761;
/// Reference harness threshold (`report.py:NVLINK_CARRIED`): a phase counts as NVLink-carried
/// when the sender's Tx counter delta covers at least this fraction of the bytes moved.
const NVLINK_CARRIED: f64 = 0.5;
/// Payload sizes of the reference sweep: 1 MiB, 16 MiB, 64 MiB, 256 MiB, 1 GiB.
const DEFAULT_SIZES: &str = "1048576,16777216,67108864,268435456,1073741824";
/// Long enough that a 1 GiB TCP leg or an engine bring-up cannot trip it, short enough that a
/// wedged peer does not hang the box.
const SOCKET_TIMEOUT: Duration = Duration::from_secs(900);

// ---------------------------------------------------------------------------------------------
// CUDA runtime, via dlopen.
// ---------------------------------------------------------------------------------------------

mod cuda {
    use std::ffi::{CString, c_char, c_int, c_uint, c_void};
    use std::sync::OnceLock;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_NOW: c_int = 2;
    const RTLD_GLOBAL: c_int = 0x100;

    pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
    pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;

    type FnMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
    type FnMemset = unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int;
    type FnSync = unsafe extern "C" fn() -> c_int;
    type FnHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int;
    type FnSetDevice = unsafe extern "C" fn(c_int) -> c_int;

    pub struct Runtime {
        memcpy: FnMemcpy,
        memset: FnMemset,
        sync: FnSync,
        host_alloc: FnHostAlloc,
        set_device: FnSetDevice,
    }

    // Every field is a plain function pointer into a process-global library.
    unsafe impl Send for Runtime {}
    unsafe impl Sync for Runtime {}

    static RUNTIME: OnceLock<Runtime> = OnceLock::new();

    /// `dlopen`s libcudart and resolves the five entry points the benchmark needs.
    ///
    /// The crate links the engine (`libsirius.so`), which does not re-export the CUDA runtime,
    /// and adding `-lcudart` would change `RUSTFLAGS` and force a full rebuild of every
    /// dependency; `dlopen` keeps the documented build line intact. Two CUDA runtime instances
    /// in one process share the driver's primary context, so pointers cross freely.
    pub fn runtime() -> &'static Runtime {
        RUNTIME.get_or_init(|| {
            let candidates = [
                "libcudart.so.13",
                "libcudart.so.12",
                "libcudart.so",
                "/usr/local/cuda/lib64/libcudart.so.13",
                "/usr/local/cuda/lib64/libcudart.so",
            ];
            let mut handle = std::ptr::null_mut();
            for name in candidates {
                let c = CString::new(name).unwrap();
                // SAFETY: `c` is a valid NUL-terminated path; dlopen tolerates a miss.
                handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
                if !handle.is_null() {
                    break;
                }
            }
            assert!(
                !handle.is_null(),
                "failed to dlopen the CUDA runtime (tried {candidates:?}); add the CUDA libdir \
                 to LD_LIBRARY_PATH"
            );
            // SAFETY: each symbol is looked up in a live libcudart handle and transmuted to its
            // documented CUDA runtime signature.
            unsafe {
                let sym = |name: &str| -> *mut c_void {
                    let c = CString::new(name).unwrap();
                    let ptr = dlsym(handle, c.as_ptr());
                    assert!(!ptr.is_null(), "libcudart exports no {name}");
                    ptr
                };
                Runtime {
                    memcpy: std::mem::transmute::<*mut c_void, FnMemcpy>(sym("cudaMemcpy")),
                    memset: std::mem::transmute::<*mut c_void, FnMemset>(sym("cudaMemset")),
                    sync: std::mem::transmute::<*mut c_void, FnSync>(sym("cudaDeviceSynchronize")),
                    host_alloc: std::mem::transmute::<*mut c_void, FnHostAlloc>(sym(
                        "cudaHostAlloc",
                    )),
                    set_device: std::mem::transmute::<*mut c_void, FnSetDevice>(sym(
                        "cudaSetDevice",
                    )),
                }
            }
        })
    }

    fn check(what: &str, code: c_int) {
        assert_eq!(code, 0, "{what} failed with CUDA error {code}");
    }

    impl Runtime {
        /// Binds this thread to the process's only visible device (ordinal 0), the same device
        /// `NixlDescriptor::device_id()` hardcodes.
        pub fn set_device_zero(&self) {
            // SAFETY: resolved cudaSetDevice, valid ordinal.
            check("cudaSetDevice(0)", unsafe { (self.set_device)(0) });
        }

        /// Host -> device copy of `src` into `dst_device`, followed by a device sync.
        pub fn copy_to_device(&self, dst_device: u64, src: &[u8]) {
            // SAFETY: `dst_device` is a registered staging-arena address with at least
            // `src.len()` bytes; `src` is a live host slice.
            check("cudaMemcpy H2D", unsafe {
                (self.memcpy)(
                    dst_device as *mut c_void,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    MEMCPY_HOST_TO_DEVICE,
                )
            });
            self.device_synchronize();
        }

        /// Device -> host copy of `len` bytes into `dst`, followed by a device sync.
        pub fn copy_to_host(&self, dst: &mut [u8], src_device: u64) {
            // SAFETY: `src_device` is a registered staging-arena address with at least
            // `dst.len()` bytes; `dst` is a live host slice.
            check("cudaMemcpy D2H", unsafe {
                (self.memcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    src_device as *const c_void,
                    dst.len(),
                    MEMCPY_DEVICE_TO_HOST,
                )
            });
            self.device_synchronize();
        }

        /// Zeroes `len` device bytes at `device_ptr` (the receiver's pre-phase wipe).
        pub fn memset_zero(&self, device_ptr: u64, len: usize) {
            // SAFETY: `device_ptr` is a registered staging-arena address with `len` bytes.
            check("cudaMemset", unsafe {
                (self.memset)(device_ptr as *mut c_void, 0, len)
            });
            self.device_synchronize();
        }

        pub fn device_synchronize(&self) {
            // SAFETY: resolved cudaDeviceSynchronize, no arguments.
            check("cudaDeviceSynchronize", unsafe { (self.sync)() });
        }

        /// Page-locked host allocation for the TCP baseline, matching the reference harness's
        /// `pin_memory=True` staging buffer.
        pub fn alloc_pinned(&self, len: usize) -> &'static mut [u8] {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a valid out-parameter; flags 0 is cudaHostAllocDefault.
            check("cudaHostAlloc", unsafe {
                (self.host_alloc)(&mut ptr as *mut *mut c_void, len, 0)
            });
            assert!(!ptr.is_null(), "cudaHostAlloc returned NULL");
            // SAFETY: cudaHostAlloc returned `len` writable, page-locked bytes; the allocation
            // lives for the process (never freed), so 'static is sound.
            unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Control-socket framing: 4-byte big-endian length + body. Same shape as the reference harness.
// ---------------------------------------------------------------------------------------------

fn send_frame(stream: &mut TcpStream, body: &[u8]) {
    let len = u32::try_from(body.len()).expect("control frame fits in u32");
    stream
        .write_all(&len.to_be_bytes())
        .expect("write frame length");
    stream.write_all(body).expect("write frame body");
    stream.flush().expect("flush control frame");
}

fn recv_frame(stream: &mut TcpStream) -> Vec<u8> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).expect("read frame length");
    let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
    stream.read_exact(&mut body).expect("read frame body");
    body
}

fn send_text(stream: &mut TcpStream, text: &str) {
    send_frame(stream, text.as_bytes());
}

fn recv_text(stream: &mut TcpStream) -> String {
    String::from_utf8(recv_frame(stream)).expect("control frame is UTF-8")
}

fn expect_text(stream: &mut TcpStream, expected: &str) {
    let got = recv_text(stream);
    assert_eq!(got, expected, "unexpected control message");
}

/// Sets `TCP_NODELAY` and 16 MiB socket buffers, matching `bench.py:tune_socket` so the TCP
/// baseline is measured under the same (deliberately generous) conditions.
fn tune_socket(stream: &TcpStream, buffer_bytes: i32) {
    unsafe extern "C" {
        fn setsockopt(
            fd: std::ffi::c_int,
            level: std::ffi::c_int,
            name: std::ffi::c_int,
            value: *const std::ffi::c_void,
            len: u32,
        ) -> std::ffi::c_int;
    }
    const SOL_SOCKET: std::ffi::c_int = 1;
    const SO_SNDBUF: std::ffi::c_int = 7;
    const SO_RCVBUF: std::ffi::c_int = 8;

    stream.set_nodelay(true).expect("TCP_NODELAY");
    stream
        .set_read_timeout(Some(SOCKET_TIMEOUT))
        .expect("read timeout");
    stream
        .set_write_timeout(Some(SOCKET_TIMEOUT))
        .expect("write timeout");
    for name in [SO_SNDBUF, SO_RCVBUF] {
        // SAFETY: a live socket fd, a 4-byte int option value, and its exact length.
        unsafe {
            setsockopt(
                stream.as_raw_fd(),
                SOL_SOCKET,
                name,
                &buffer_bytes as *const i32 as *const std::ffi::c_void,
                std::mem::size_of::<i32>() as u32,
            );
        }
    }
}

/// Dials `addr`, retrying until the peer's engine has come up and it is listening.
fn connect_retry(addr: &str, deadline: Duration) -> TcpStream {
    let start = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => return stream,
            Err(err) => {
                assert!(
                    start.elapsed() < deadline,
                    "could not connect to {addr} within {deadline:?}: {err}"
                );
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Verification pattern (identical to the reference harness).
// ---------------------------------------------------------------------------------------------

/// `seed = nbytes ^ (sum(method bytes) << 8)` — unique per (method, payload), so a stale buffer
/// from an earlier phase cannot pass verification.
fn phase_seed(method: &str, nbytes: u64) -> u64 {
    let method_sum: u64 = method.bytes().map(u64::from).sum();
    nbytes ^ (method_sum << 8)
}

/// Fills `buf` with the int64 lane pattern `lane[i] = i * PATTERN_MULTIPLIER + seed`.
fn fill_pattern(buf: &mut [u8], seed: u64) {
    assert_eq!(buf.len() % 8, 0, "payload sizes are multiples of 8");
    for (index, lane) in buf.chunks_exact_mut(8).enumerate() {
        let value = (index as u64)
            .wrapping_mul(PATTERN_MULTIPLIER)
            .wrapping_add(seed);
        lane.copy_from_slice(&value.to_le_bytes());
    }
}

/// Byte-for-byte check of `buf` against the pattern, returning the first differing lane.
fn check_pattern(buf: &[u8], seed: u64) -> Result<(), String> {
    for (index, lane) in buf.chunks_exact(8).enumerate() {
        let expected = (index as u64)
            .wrapping_mul(PATTERN_MULTIPLIER)
            .wrapping_add(seed);
        let got = u64::from_le_bytes(lane.try_into().unwrap());
        if got != expected {
            return Err(format!(
                "lane {index} (byte offset {}) is {got:#x}, expected {expected:#x}",
                index * 8
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// NVLink hardware counters.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct NvlinkCounters {
    tx: u64,
    rx: u64,
    available: bool,
}

/// Sums `Data Tx` / `Data Rx` over every link of one *physical* GPU.
///
/// NVML ignores `CUDA_VISIBLE_DEVICES`, so both GPUs of the pair can be sampled from whichever
/// process runs this. An unreadable counter is reported as unavailable rather than fatal — a
/// missing proof must not abort a timing run (same policy as `bench.py:read_nvlink_counters`).
fn nvlink_counters(gpu: u32) -> NvlinkCounters {
    let output = std::process::Command::new("nvidia-smi")
        .args(["nvlink", "-gt", "d", "-i", &gpu.to_string()])
        .output();
    let Ok(output) = output else {
        return NvlinkCounters::default();
    };
    if !output.status.success() {
        return NvlinkCounters::default();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut counters = NvlinkCounters {
        tx: 0,
        rx: 0,
        available: false,
    };
    for line in text.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        // "Link 0: Data Tx: 13508379 KiB"
        let Some(position) = tokens
            .iter()
            .position(|token| *token == "Tx:" || *token == "Rx:")
        else {
            continue;
        };
        if tokens.len() < position + 3 {
            continue;
        }
        let Ok(value) = tokens[position + 1].parse::<u64>() else {
            continue;
        };
        let scale = match tokens[position + 2] {
            "B" => 1,
            "KiB" => 1024,
            "MiB" => 1024 * 1024,
            "GiB" => 1024 * 1024 * 1024,
            _ => continue,
        };
        counters.available = true;
        if tokens[position] == "Tx:" {
            counters.tx += value * scale;
        } else {
            counters.rx += value * scale;
        }
    }
    counters
}

// ---------------------------------------------------------------------------------------------
// Statistics, replicating the reference harness's index rule exactly.
// ---------------------------------------------------------------------------------------------

/// `bench.py:percentile`: `index = min(n-1, max(0, round(f*(n-1))))` with Python's
/// banker's rounding. For n = 10 that makes the median `sorted[4]` and p90 `sorted[8]` — not
/// the mean of the middle pair, so the numbers line up with the reference to the last decimal.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let n = sorted.len();
    assert!(n > 0, "no samples");
    let raw = fraction * (n - 1) as f64;
    let floor = raw.floor();
    let index = if (raw - floor - 0.5).abs() < f64::EPSILON {
        // Half-to-even, as Python's round() does.
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        raw.round()
    };
    sorted[(index as usize).min(n - 1)]
}

// ---------------------------------------------------------------------------------------------
// Configuration.
// ---------------------------------------------------------------------------------------------

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_required(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} must be set"))
}

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

fn human_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
        ("B", 1),
    ];
    for (name, scale) in UNITS {
        if bytes >= scale {
            return format!("{:.2} {name}", bytes as f64 / scale as f64);
        }
    }
    "0 B".to_string()
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
