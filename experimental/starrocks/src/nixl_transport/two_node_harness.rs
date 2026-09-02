//! Scaffolding shared by the multi-process nixl tests ([`nixl_bench`](super::nixl_bench) and
//! [`nixl_echo`](super::nixl_echo)).
//!
//! Everything here is *around* the transport, never part of it: a `dlopen`ed CUDA runtime so the
//! Rust side can touch device memory at all, a length-prefixed control socket standing in for the
//! brpc control plane, the int64 verification pattern, and the NVLink hardware counters. The nixl
//! calls those tests measure live in [`agent_tier`](super::agent_tier) and are used verbatim.
//!
//! This module was factored out of `nixl_bench` when the two-node echo test needed the same
//! pieces; the definitions are unchanged, so benchmark numbers stay comparable across the move.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

/// Reference harness constant (`bench.py:PATTERN_MULTIPLIER`): the int64 lane stride of the
/// verification pattern. Kept identical so a lane mismatch here means the same thing there.
const PATTERN_MULTIPLIER: u64 = 2654435761;
/// Long enough that a 1 GiB TCP leg or an engine bring-up cannot trip it, short enough that a
/// wedged peer does not hang the box.
pub(super) const SOCKET_TIMEOUT: Duration = Duration::from_secs(900);

// ---------------------------------------------------------------------------------------------
// CUDA runtime, via dlopen.
// ---------------------------------------------------------------------------------------------

pub(super) mod cuda {
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
    type FnMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
    type FnFree = unsafe extern "C" fn(*mut c_void) -> c_int;

    pub struct Runtime {
        memcpy: FnMemcpy,
        memset: FnMemset,
        sync: FnSync,
        host_alloc: FnHostAlloc,
        set_device: FnSetDevice,
        malloc: FnMalloc,
        free: FnFree,
    }

    // Every field is a plain function pointer into a process-global library.
    unsafe impl Send for Runtime {}
    unsafe impl Sync for Runtime {}

    static RUNTIME: OnceLock<Runtime> = OnceLock::new();

    /// `dlopen`s libcudart and resolves the five entry points these tests need.
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
                    malloc: std::mem::transmute::<*mut c_void, FnMalloc>(sym("cudaMalloc")),
                    free: std::mem::transmute::<*mut c_void, FnFree>(sym("cudaFree")),
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

        /// Plain `cudaMalloc` of `len` device bytes, returning the device address.
        ///
        /// This is what the engine's `exchange_staging_arena` is — a single `cudaMalloc`, not
        /// pool memory, which matters because pool memory silently degrades ~220x over cuda_ipc
        /// (finding F1). A test can therefore register one of these as its arena and exercise
        /// the real transport without standing up the engine.
        pub fn alloc_device(&self, len: usize) -> u64 {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a valid out-parameter for cudaMalloc.
            check("cudaMalloc", unsafe {
                (self.malloc)(&mut ptr as *mut *mut c_void, len)
            });
            assert!(!ptr.is_null(), "cudaMalloc returned NULL");
            ptr as u64
        }

        /// Releases an [`alloc_device`](Self::alloc_device) allocation.
        pub fn free_device(&self, device_ptr: u64) {
            // SAFETY: `device_ptr` came from cudaMalloc and is not freed twice.
            check("cudaFree", unsafe {
                (self.free)(device_ptr as *mut c_void)
            });
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
// Fabric-capable device memory, via the CUDA driver's VMM API.
// ---------------------------------------------------------------------------------------------

/// A device allocation exportable as an NVLink **fabric** handle.
///
/// This is the difference between a cross-node transfer riding NVLink and falling back to
/// RDMA/TCP. `cudaMalloc` memory can only be shared through `cuIpcGetMemHandle`, whose handles
/// are node-local, so UCX's `cuda_ipc` lane silently drops for an off-box peer. Multi-node
/// NVLink instead needs memory created through the VMM API with
/// `CU_MEM_HANDLE_TYPE_FABRIC` requested, which UCX can export to a peer in the same IMEX
/// domain. (NVIDIA documents the same requirement for NIXL under vLLM: without VMM-registered
/// buffers "NIXL can only use RDMA/TCP for cross-node" transfers.)
pub(super) mod cuda_vmm {
    use std::ffi::{CString, c_char, c_int, c_void};
    use std::sync::OnceLock;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    const RTLD_GLOBAL: c_int = 0x100;

    const CU_MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
    const CU_MEM_HANDLE_TYPE_FABRIC: u32 = 8;
    const CU_MEM_LOCATION_TYPE_DEVICE: u32 = 1;
    const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: u32 = 3;
    const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: u32 = 1;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Location {
        kind: u32,
        id: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct AllocFlags {
        compression_type: u8,
        gpu_direct_rdma_capable: u8,
        usage: u16,
        reserved: [u8; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct AllocationProp {
        kind: u32,
        requested_handle_types: u32,
        location: Location,
        win32_handle_meta_data: *mut c_void,
        alloc_flags: AllocFlags,
    }

    // The pointer field is a null Win32-only handle; nothing dereferences it on Linux.
    unsafe impl Send for AllocationProp {}

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct AccessDesc {
        location: Location,
        flags: u32,
    }

    type FnInit = unsafe extern "C" fn(u32) -> c_int;
    type FnDeviceGet = unsafe extern "C" fn(*mut i32, c_int) -> c_int;
    type FnGranularity = unsafe extern "C" fn(*mut usize, *const AllocationProp, u32) -> c_int;
    type FnMemCreate = unsafe extern "C" fn(*mut u64, usize, *const AllocationProp, u64) -> c_int;
    type FnAddressReserve = unsafe extern "C" fn(*mut u64, usize, usize, u64, u64) -> c_int;
    type FnMemMap = unsafe extern "C" fn(u64, usize, usize, u64, u64) -> c_int;
    type FnSetAccess = unsafe extern "C" fn(u64, usize, *const AccessDesc, usize) -> c_int;

    struct Driver {
        granularity: FnGranularity,
        create: FnMemCreate,
        reserve: FnAddressReserve,
        map: FnMemMap,
        set_access: FnSetAccess,
    }

    unsafe impl Send for Driver {}
    unsafe impl Sync for Driver {}

    static DRIVER: OnceLock<Driver> = OnceLock::new();

    fn driver() -> &'static Driver {
        DRIVER.get_or_init(|| {
            let name = CString::new("libcuda.so.1").unwrap();
            // SAFETY: a valid NUL-terminated library name.
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            assert!(!handle.is_null(), "failed to dlopen libcuda.so.1");
            // SAFETY: every symbol is resolved from a live libcuda and transmuted to its
            // documented driver-API signature.
            unsafe {
                let sym = |name: &str| -> *mut c_void {
                    let c = CString::new(name).unwrap();
                    let ptr = dlsym(handle, c.as_ptr());
                    assert!(!ptr.is_null(), "libcuda exports no {name}");
                    ptr
                };
                // The primary context already exists (the CUDA runtime made it), but cuInit is
                // cheap and idempotent, and the driver API is undefined without it.
                let init = std::mem::transmute::<*mut c_void, FnInit>(sym("cuInit"));
                assert_eq!(init(0), 0, "cuInit failed");
                let device_get =
                    std::mem::transmute::<*mut c_void, FnDeviceGet>(sym("cuDeviceGet"));
                let mut device = 0i32;
                assert_eq!(device_get(&mut device, 0), 0, "cuDeviceGet(0) failed");
                Driver {
                    granularity: std::mem::transmute::<*mut c_void, FnGranularity>(sym(
                        "cuMemGetAllocationGranularity",
                    )),
                    create: std::mem::transmute::<*mut c_void, FnMemCreate>(sym("cuMemCreate")),
                    reserve: std::mem::transmute::<*mut c_void, FnAddressReserve>(sym(
                        "cuMemAddressReserve",
                    )),
                    map: std::mem::transmute::<*mut c_void, FnMemMap>(sym("cuMemMap")),
                    set_access: std::mem::transmute::<*mut c_void, FnSetAccess>(sym(
                        "cuMemSetAccess",
                    )),
                }
            }
        })
    }

    /// Reserves, backs and maps `len` bytes of fabric-exportable device memory on device 0,
    /// returning `(device address, mapped length)`. The length is rounded up to the allocation
    /// granularity, which `cuMemCreate` requires.
    ///
    /// Panics with the driver's error code if the platform cannot produce a fabric handle — on a
    /// box without IMEX, or without permission on `/dev/nvidia-caps-imex-channels`, `cuMemCreate`
    /// fails here rather than silently degrading later.
    pub fn alloc_fabric(len: usize) -> (u64, usize) {
        let driver = driver();
        let prop = AllocationProp {
            kind: CU_MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: CU_MEM_HANDLE_TYPE_FABRIC,
            location: Location {
                kind: CU_MEM_LOCATION_TYPE_DEVICE,
                // Process-visible ordinal 0, the device `NixlDescriptor::device_id()` hardcodes.
                id: 0,
            },
            ..Default::default()
        };

        // SAFETY: all five calls take a valid prop/out-pointer pair and are checked; the address
        // range stays reserved and mapped for the process's lifetime.
        unsafe {
            let mut granularity = 0usize;
            let status = (driver.granularity)(
                &mut granularity,
                &prop,
                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            );
            assert_eq!(status, 0, "cuMemGetAllocationGranularity failed: {status}");
            assert!(granularity > 0, "zero allocation granularity");
            let size = len.next_multiple_of(granularity);

            let mut handle = 0u64;
            let status = (driver.create)(&mut handle, size, &prop, 0);
            assert_eq!(
                status, 0,
                "cuMemCreate with CU_MEM_HANDLE_TYPE_FABRIC failed: {status} — this box cannot \
                 export fabric handles (is nvidia-imex running, and are the \
                 /dev/nvidia-caps-imex-channels devices accessible to this user?)"
            );

            let mut ptr = 0u64;
            let status = (driver.reserve)(&mut ptr, size, granularity, 0, 0);
            assert_eq!(status, 0, "cuMemAddressReserve failed: {status}");

            let status = (driver.map)(ptr, size, 0, handle, 0);
            assert_eq!(status, 0, "cuMemMap failed: {status}");

            let access = AccessDesc {
                location: prop.location,
                flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            };
            let status = (driver.set_access)(ptr, size, &access, 1);
            assert_eq!(status, 0, "cuMemSetAccess failed: {status}");

            (ptr, size)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Control-socket framing: 4-byte big-endian length + body. Same shape as the reference harness.
// ---------------------------------------------------------------------------------------------

pub(super) fn send_frame(stream: &mut TcpStream, body: &[u8]) {
    let len = u32::try_from(body.len()).expect("control frame fits in u32");
    stream
        .write_all(&len.to_be_bytes())
        .expect("write frame length");
    stream.write_all(body).expect("write frame body");
    stream.flush().expect("flush control frame");
}

pub(super) fn recv_frame(stream: &mut TcpStream) -> Vec<u8> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).expect("read frame length");
    let mut body = vec![0u8; u32::from_be_bytes(len) as usize];
    stream.read_exact(&mut body).expect("read frame body");
    body
}

pub(super) fn send_text(stream: &mut TcpStream, text: &str) {
    send_frame(stream, text.as_bytes());
}

pub(super) fn recv_text(stream: &mut TcpStream) -> String {
    String::from_utf8(recv_frame(stream)).expect("control frame is UTF-8")
}

pub(super) fn expect_text(stream: &mut TcpStream, expected: &str) {
    let got = recv_text(stream);
    assert_eq!(got, expected, "unexpected control message");
}

/// Sets `TCP_NODELAY` and the requested socket buffers, matching `bench.py:tune_socket` so the
/// TCP baseline is measured under the same (deliberately generous) conditions.
pub(super) fn tune_socket(stream: &TcpStream, buffer_bytes: i32) {
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
pub(super) fn connect_retry(addr: &str, deadline: Duration) -> TcpStream {
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
pub(super) fn phase_seed(method: &str, nbytes: u64) -> u64 {
    let method_sum: u64 = method.bytes().map(u64::from).sum();
    nbytes ^ (method_sum << 8)
}

/// Fills `buf` with the int64 lane pattern `lane[i] = i * PATTERN_MULTIPLIER + seed`.
pub(super) fn fill_pattern(buf: &mut [u8], seed: u64) {
    assert_eq!(buf.len() % 8, 0, "payload sizes are multiples of 8");
    for (index, lane) in buf.chunks_exact_mut(8).enumerate() {
        let value = (index as u64)
            .wrapping_mul(PATTERN_MULTIPLIER)
            .wrapping_add(seed);
        lane.copy_from_slice(&value.to_le_bytes());
    }
}

/// Byte-for-byte check of `buf` against the pattern, returning the first differing lane.
pub(super) fn check_pattern(buf: &[u8], seed: u64) -> Result<(), String> {
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
pub(super) struct NvlinkCounters {
    pub(super) tx: u64,
    pub(super) rx: u64,
    pub(super) available: bool,
}

/// Sums `Data Tx` / `Data Rx` over every link of one *physical* GPU.
///
/// NVML ignores `CUDA_VISIBLE_DEVICES`, so any GPU of the host can be sampled from whichever
/// process runs this. An unreadable counter is reported as unavailable rather than fatal — a
/// missing proof must not abort a timing run (same policy as `bench.py:read_nvlink_counters`).
pub(super) fn nvlink_counters(gpu: u32) -> NvlinkCounters {
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
pub(super) fn percentile(sorted: &[f64], fraction: f64) -> f64 {
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

pub(super) fn human_bytes(bytes: u64) -> String {
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
// Configuration.
// ---------------------------------------------------------------------------------------------

pub(super) fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

pub(super) fn env_required(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} must be set"))
}
