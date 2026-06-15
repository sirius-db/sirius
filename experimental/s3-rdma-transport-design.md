# S3 RDMA Transport for Sirius — Design for Team Review

| | |
|---|---|
| **Status** | Draft for team review |
| **Date** | 2026-06-13 |
| **Author** | Sirius Contributors |
| **Benchmark** | <https://github.com/sirius-db/s3RDMA-benchmarktool> (~91 Gbps GPU-mode GET on CX-6 RoCE) |
| **Background** | Builds on prior internal IO-layer design notes (full working spec; the RDMA-S3-ioctx v8 and IO-layer plans) — not in this repo; this doc is self-contained for review. |

## TL;DR

Add an RDMA transport to the Sirius S3 stack: large parquet column reads are
`RDMA_WRITE`n by a cuObject-speaking gateway into a small pre-registered GPU
**landing arena**, then moved to their RMM destination by a GPU-local
device-to-device copy — eliminating today's pinned-host staging and PCIe H2D
leg entirely. The `s3://` URI surface, SigV4 auth, datasource abstraction, and
parquet scanner are unchanged; the new code is one reactor plus a thin ioctx
under the existing `templated_ioctx` machinery. **Ask: review decisions D1–D9
and answer the open questions in §8.**

## 1. Motivation & evidence

Sirius reads S3 parquet today via libcurl range GETs into pinned host staging
blocks, then `cudaMemcpyAsync` H2D (`s3_reactor`). Every byte crosses host
memory and the PCIe bus twice (NIC→host, host→GPU).

The `s3RDMA-benchmarktool` project (published under sirius-db) proves the
alternative end-to-end on NVIDIA cuObject: an HTTP control plane carries an
RDMA descriptor token; the server `RDMA_WRITE`s object bytes **directly into
client GPU memory** (GPUDirect). Measured: **~91 Gbps sustained GPU-mode GET
(~91% of line rate) on CX-6 100G RoCE, reaching line rate at 8 client
threads**; host-mode and GPU-mode throughput are identical, and
alloc+register is a one-time sub-second cost amortized by a
register-once / read-many pattern.

## 2. Goals / Non-goals

**Goals**

- Large device reads over RDMA directly into GPU memory; zero host bounce.
- Preserve `s3://` URIs, SigV4 authorization, `sirius_datasource` semantics,
  and the existing HTTP backend as a config-selectable alternative.
- Land behind the existing IO seam — no parquet scanner changes.

**Non-goals (v1)**

- No replacement of the HTTP/libcurl backend; no new URI scheme.
- No AWS S3 support — this requires an RDMA-capable, cuObject-speaking
  gateway (on-prem / Sirius-controlled).
- No `AUTO` capability probing (the existing `AUTO` enum value resolves to
  HTTP) and **no runtime HTTP fallback**: RDMA failures surface as I/O
  errors; switching back to HTTP is an operator config action.
- No AWS-CRT dependency: the client base is the benchmark's bare
  `cuObjClient` + HTTP control plane.

## 3. Architecture overview

```text
Parquet/GPU scan
  → sirius_datasource::device_read(...)              (RMM device_buffer dst)
  → templated_ioctx<cuobj_rdma_reactor>              (1 MiB chunking — reused)
  → cuobj_rdma_reactor worker
      ① acquire free arena slot                      (pre-registered GPU memory)
      ② cuObjGet → SigV4-signed HTTP control GET
           x-amz-rdma-token: base64(cuObject RDMA descriptor; encodes slot addr)
           Range: [off, off+1MiB)
      ③ gateway validates auth/range/size → RDMA_WRITEs into the slot
      ④ cuObjGet returns (host-side completion fact)
      ⑤ [flush if platform requires]                 (visibility, §A.3)
      ⑥ cudaMemcpyAsync(r.dst ← slot, D2D, r.stream) (GPU-local, ~TB/s)
      ⑦ callback marks done + wakes → worker recycles slot → chunk_done()
  → cuDF decode kernels on r.stream                  (stream order ⇒ visible)
```

Relative to `s3_reactor` (GET → pinned host staging → H2D → stream callback),
this backend replaces the host staging pool with a small **GPU-resident
arena** and the PCIe H2D copy with a GPU-local D2D (<1% overhead vs the
network). The completion discipline — CUDA callback only marks+wakes, a
worker thread finishes the chunk — is identical to `s3_reactor`'s proven
pattern. Host reads (parquet footers, HEAD, small metadata) stay on the
existing HTTP path.

## 4. Key decisions

**D1 — Keep `s3://`; select the transport by config.** *(Decided)*
`object_store_config::transport { AUTO, HTTP, RDMA }` already exists in
Sirius. The scan manager selects and owns exactly one S3 backend per Sirius
context (in `_io_ctxs`): `HTTP` → today's `s3_ioctx`; `RDMA` → the new backend.
No new URI scheme, no second routing mechanism.

**D2 — Shape: `s3_rdma_ioctx : templated_ioctx<cuobj_rdma_reactor>`.**
*(Decided)* One new reactor satisfying the `io_reactor_c` concept, plus thin
but load-bearing ioctx overrides: an instance `create_io_object` (HEAD via
the SigV4 `s3_request_authorizer` — a reactor static can't carry
credentials), `host_read_ranges_async_io` (strict range validation — errors
via the handler, not the generic version's silent skip of invalid/too-small
ranges; footer/prewarm/metadata paths all hit it), and counters. Chunking,
pooling, async completion, the datasource adapter, and the factory are reused
unchanged. This mirrors how `s3_ioctx` itself is built.

**D3 — One hybrid ioctx, not two.** *(Decided)* In RDMA mode, host reads
(footers/HEAD/metadata) go over HTTP and device reads over RDMA — inside the
same ioctx. Note: `s3_reactor` is **never instantiated** in RDMA mode — the
HTTP host-read path is the reactor's own sync, SigV4-signed GET; the two
backends are mutually exclusive per process, never composed. Callers see
identical `sirius_datasource` semantics across backends.

**D4 — Landing arena + GPU-local D2D, not direct RDMA into RMM buffers.**
*(Decided; the central design choice)* Super Sirius device memory is
`rmm::mr::cuda_async_memory_resource` (cudaMallocAsync mempool, via
cucascade). That memory is (a) not reliably registrable for GPUDirect RDMA
and (b) stream-ordered — an address may lack physical backing when the NIC
writes. Therefore the reactor owns a small `cudaMalloc`'d arena per device
(default `max_inflight × 1 MiB` = 8 MiB at the 8-worker default), registered **once**; RDMA lands
in arena slots and a `cudaMemcpyAsync` D2D on the request stream delivers to
the RMM destination. D2D runs at GPU memory bandwidth (~1–2 TB/s) vs
~12.5 GB/s network ⇒ <1% overhead; bytes never touch host memory. Slot
exhaustion is natural backpressure. (Direct-into-RMM remains a gated future
spike — Appendix A.5.)

**D5 — Completion: callback-only-wake; the worker finishes the chunk.**
*(Decided)* `request_context::chunk_done()` fires the user handler
synchronously on the last chunk, so it must never be called from a CUDA
callback thread. The CUDA callback only records the copy status and wakes the
worker; the worker observes the status → safely recycles the slot → calls
`chunk_done()`/`chunk_failed()`. Contract: the user handler always fires on a
reactor/worker thread, with the bytes already in the final `r.dst`. This is
`s3_reactor`'s exact structure. Exhaustive error paths in Appendix A.2.

**D6 — v1 scope: explicit opt-in, fail loudly.** *(Decided)* RDMA requires
`s3_transport=RDMA`; `AUTO` resolves to HTTP (no probe). No runtime HTTP
fallback — a failed RDMA transfer is an I/O error through the existing
handler path. **All device reads go RDMA in v1** (with the byte-range chunk-prewarm
off per D9, so nothing short-circuits them) — a small-read HTTP threshold
is deferred to P5, pending RTT measurements on the rig. Rationale:
fallbacks and premature thresholds mask misconfiguration in exactly the
environments where RDMA is deliberately deployed.

**D7 — Optional build feature; never vendor NVIDIA SDKs.** *(Decided)*
conda-forge ships `libcuobjclient`/`libcufile`; Sirius's lockfile today
resolves `libcufile` only. Gate the backend behind a build flag (e.g.
`SIRIUS_ENABLE_S3_RDMA`) that adds the `libcuobjclient` dependency or accepts
`CUOBJ_SDK_ROOT`; default-off builds are unchanged. Apache-2.0 repos carry no
NVIDIA binaries (same policy as the benchmark repo).

**D8 — Dev rig is unsigned; production gateways must validate SigV4.**
*(Decided)* The benchmark's own server (local files + `RDMA_WRITE`) is the
dev/test rig — it does not validate SigV4 and is for isolated networks only.
Production gateways must validate SigV4 **before** honoring RDMA headers
(RDMA headers must never bypass auth). The Sirius client signs
unconditionally. Vendor appliances with different token formats (e.g. Dell
ECS/ObjectScale) are out of scope until the token relay is abstracted.

**D9 — Byte-range chunk-prewarm off in RDMA mode.** *(Decided)*
`sirius_ioctx::device_read` consults the prefetch `_cache` **before**
`device_read_io`: a byte-range cache hit copies from host-pinned slices to the
device and never reaches the reactor, which would bypass RDMA and re-introduce
host staging — directly contradicting "all device reads go RDMA".

The control is **not the backend's** — it lives in `scan_manager_config`:
`enable_prefetch_cache` (whole cache) and `enable_chunk_prewarm` (the
byte-range prewarm, ignored when the cache is off). So when
`s3_transport=RDMA`, the `SiriusContext`/scan-config composition must set
**`enable_chunk_prewarm = false`** while keeping **`enable_prefetch_cache =
true`** (metadata-only caching: parquet footer / `describe_parquet` stays on).
Caveat: `enable_chunk_prewarm` is a global scan-manager knob, so turning it off
also disables prewarm for local-file (uring) reads in the same context;
preserving local prewarm alongside RDMA-S3 would require the scanner to
distinguish the backend — i.e. it would break "zero parquet scanner changes",
and is deferred. v1 accepts the global off (RDMA deployments are S3-dominated).
This reaffirms the inherited `rdma-s3-ioctx-plan.md` v8 decision.

## 5. Rollout

| Phase | Scope | Hardware |
|---|---|---|
| P0 | **Migrate** the benchmark to the standard wire protocol (drop the 4 non-standard headers; server derives the slot address from the token, not a header) **and** harden: short-write = error (authoritative check is the `cuObjGet` result), capacity from the `cuObjGet` `size` arg, O_DIRECT edges, document callback-once (Appendix C.3) | none |
| P1 | Wire `object_store_config.s3_transport` into `scan_manager_config`; the scan-manager constructor selects the HTTP/RDMA backend; `AUTO`→HTTP; config tests | none |
| P2 | `cuobj_rdma_reactor` against a **mock** client; full hardware-free test matrix (slot state machine, error paths, concept check) | none |
| P3 | Real cuObject path: single GPU, arena + D2D, flush semantics, visibility stress test on the dev rig | CX-6 rig |
| P4 | Multi-GPU (per-device arenas, NIC↔GPU affinity validation), RDMA-internal retries, metrics, tuning | CX-6 rig |
| P5 | Range coalescing/batching; direct-into-RMM spike; small-read RTT → threshold go/no-go; benchmark vs HTTP path (target ≥ HTTP everywhere, approach ~91 Gbps) | CX-6 rig |

## 6. Risks

1. **CUDA visibility after external RDMA writes** — top correctness risk;
   confined to one flush-before-copy point; dedicated stress test gates P3.
2. Requires a controlled cuObject-speaking gateway; plain AWS S3 does not apply.
3. NIC↔GPU PCIe affinity: cross-root-complex peer DMA is slow or unsupported;
   silent far-NIC selection is a perf bug (validation step in P4).
4. Server slot budget: each in-flight GET holds a gateway DCI channel + staging
   pair; the gateway's `max_concurrency` must cover `max_inflight` per RDMA
   ioctx / client process (the global worker-pool ceiling), multiplied only
   across independent Sirius contexts/processes — not by queries or GPU count.
5. Control-plane RTT + D2D overhead on small reads — measured in P5; an
   ioctx-level whole-read HTTP threshold is the prepared mitigation if
   warranted (none ships in v1).
6. cuObject's blocking GET shapes the concurrency model (worker pool);
   revisit if an async cuObject API appears.

## 7. First milestone

`s3_transport=RDMA` with `s3://` unchanged; host reads still HTTP (SigV4 as
today); large device reads through a mockable RDMA client; failures error out
loudly; zero parquet scanner changes. First code lands on the chain
`object_store_config.s3_transport → scan_manager_config → sirius_scan_manager
selects the HTTP vs RDMA backend (held in `_io_ctxs`) →
sirius_scan_manager::create_datasource() → parquet_gpu_ingestible` (P1+P2),
then the real client swaps in (P3). `SiriusContext` only composes the
scan_manager_config from `object_store_config`; the backend is owned by the
scan manager.

## 8. Open questions for team review

1. **Arena / worker sizing defaults.** Two coupled knobs, both seeded from the
   benchmark's evidence — the question is whether to ship these starting points
   (all config-overridable) given Sirius's concurrent-scan profile:
   - `s3_rdma_max_inflight` (default **8**) is the **worker-thread count**;
     since each worker runs one blocking `cuObjGet`, it is *also* the global
     in-flight ceiling (not a per-device bound — see §A.4). The benchmark
     reaches ~91 Gbps line rate at **8 client threads** (1 ≈ 48 Gbps, 2 ≈ 79,
     8–32 flat at ~91), so 8 is the knee; below it leaves bandwidth on the
     table, above it adds threads/sessions for no throughput.
   - `s3_rdma_arena_slot_size` (default **1 MiB**, = `templated_ioctx`'s chunk)
     gives a per-device arena = `max_inflight × slot_size` ≈ **8 MiB/device**
     (worst case: all workers on one device) of registered GPU memory —
     negligible, so the default errs large/safe.
   - Coupling to the **server**: each in-flight GET holds one gateway DCI
     channel + staging pair, so the gateway's `max_concurrency` must cover
     `max_inflight` total (global, **not** × GPUs). A high client default can
     starve a small-pool gateway — is 8 the right *client* default, or should
     it track a negotiated server budget?
2. **OWNER-ordering flush semantics** — who owns verifying against the CUDA
   docs/header whether `CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER` still
   requires `cuFlushGPUDirectRDMAWrites` for same-device consumers (P3 gate)?
3. **Dev-rig hardware** — can we allocate the CX-6 pair (client GPU node +
   server node) for P3–P5? Single-NIC is sufficient for P3.
4. **cucascade registrable IO sub-MR** — is there appetite for a
   `cudaMalloc`-backed, registrable allocation path for IO destination
   buffers? It would unlock direct-into-RMM (removing the <1% D2D) — low
   urgency, but it affects cucascade's memory architecture.
5. **Metrics integration.** The *mechanism* is settled — follow the existing
   `s3_ioctx` pattern: `std::atomic` counters live in the reactor, the ioctx
   aggregates across reactors, and they surface as pull-based
   `…_total() const noexcept` accessors (mirroring `bytes_read_total` /
   `device_copies_total` / `device_stream_sync_total`). Genuinely open: (a) the
   RDMA counter set (Appendix C.2) and its naming alignment with the existing
   S3 counters; (b) whether IO counters should *also* be pushed into
   `telemetry_context` — today IO metrics are pull-only, and `telemetry_context`
   carries plan telemetry, not IO. (Note: P0 changes to `s3RDMA-benchmarktool`
   are owned by us and coordinated across both repos — not a team decision.)

---

# Appendix A — Detailed data-path design

## A.1 Registration & the landing arena

cuObject can only GET into GPU memory previously registered via
`cuMemObjGetDescriptor` (sub-second per region; the benchmark measured
~650–720 ms alloc+register for 128 MiB, one-time). Per-chunk or
per-`device_buffer` registration would dominate a ~100 µs 1 MiB transfer —
hence register-once arenas:

- **`s3_rdma_ioctx` constructs `templated_ioctx` with a single reactor**
  (`n_reactors = 1`): `cuObjGet` blocks, so concurrency lives entirely in that
  reactor's worker pool — there is no benefit to the multi-reactor round-robin
  the libcurl backend uses. This makes the reactor the natural owner of the
  per-device arenas and the inflight budget, with no separate ioctx-level
  arena manager. (If a future need forces multiple reactors, the arenas +
  budget must move to a shared ioctx-level `s3_rdma_runtime` — explicitly out
  of scope for v1.)
- The reactor maintains an arena **per device**, created lazily on first use:
  a `cudaMalloc`'d region of `s3_rdma_max_inflight` slots of `slot_size` bytes
  each (`arena bytes = max_inflight × slot_size`)
  (`slot_size` ≥ chunk size, default 1 MiB), registered once at creation.
- A chunk GET targets a free slot; slot acquisition blocks when exhausted
  (backpressure, like the benchmark server's slot pool).
- The slot is recycled by the worker only after it observes the
  copy-completion callback status (§A.2).
- Why not direct into RMM buffers: `cuda_async_memory_resource` memory is not
  reliably registrable for GPUDirect (`nvidia_peermem`; dmabuf support for
  mempools is not established) **and** its backing is stream-ordered — an
  address may have no physical backing when the NIC writes. The arena
  (`cudaMalloc`, always backed, registered once) sidesteps both; the D2D on
  the allocation stream is ordered after the RMM buffer's backing by
  construction.

## A.2 Completion semantics & error paths

Per chunk request `r` (worker thread): acquire slot → `cuObjGet`-driven
control GET (`Range: [r.file_off, r.file_off + r.io_size)`; the slot address is
encoded in `x-amz-rdma-token`) → server `RDMA_WRITE`s → HTTP 200/206 with
`x-amz-rdma-reply` (RDMA status) → `cuObjGet` returns `n` → flush if required
(§A.3) → `cudaMemcpyAsync(r.dst, slot, n, DeviceToDevice, r.stream)` →
register completion callback → worker finishes.

**Completion discipline** (mirrors `s3_reactor`): the CUDA callback only
records status, release-stores a done flag, and wakes the worker — it never
calls `chunk_done()` (which fires the user handler synchronously on the last
chunk) and never recycles the slot. The worker: observe status → safely
recycle the slot → `chunk_done()` / `chunk_failed(ep)`. The user handler
always fires on a worker/reactor thread with bytes already in `r.dst`.

**Error paths (exhaustive):**

| Failure | Handling |
|---|---|
| `cuObjGet` rc < 0; `n != r.data_size`; HTTP ≠ 200/206; WC error | `chunk_failed(ep)`; slot recycled immediately (no copy queued) |
| Flush failure | `chunk_failed(ep)`; slot recycled immediately |
| `cudaMemcpyAsync` enqueue failure | `chunk_failed(ep)`; slot recycled immediately |
| Callback registration failure **after** the copy was queued | `cudaStreamSynchronize(r.stream)` first (the copy may still read the slot), then `chunk_failed(ep)` |
| D2D completion error (callback status ≠ `cudaSuccess`) | worker observes, recycles slot, then `chunk_failed(ep)` |

Slot lifetime rule: once the D2D is enqueued, the slot belongs to the stream
until the worker has observed the callback status (or sync-drained on the
failure path). No retry across transports; RDMA-internal retries arrive in P4.

Invariant: `align_to_physical` is identity for this backend (the server
handles O_DIRECT alignment), so `r.data_off == 0` and
`r.io_size == r.data_size` — asserted.

## A.3 Visibility & ordering

Placement is host-side certain: RDMA reliable-connection semantics + the
server's WC check guarantee the bytes are in the slot before the HTTP reply;
`cuObjGet` returning implies placement. GPU-consumer visibility is the only
work: at init read `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING` —

- `ALL_DEVICES` (typical modern x86 + Ampere/Hopper): no flush.
- `OWNER`/`NONE`: `cuFlushGPUDirectRDMAWrites` after `cuObjGet` returns and
  **before enqueuing that chunk's D2D**, so the copy engine observes the
  NIC's writes. If per-chunk flush measures hot, batch several completed
  slots behind one flush. (Exact target/scope enums to be pinned against the
  CUDA header in P3; whether `OWNER` requires the flush for same-device
  consumers is verified, not assumed — open question 2.)

Downstream needs nothing: the D2D and the decode kernels share `r.stream`, so
stream order guarantees the decode sees the copied bytes.

## A.4 Concurrency, multi-GPU, teardown

- **Workers**: `cuObjGet` blocks ⇒ each reactor owns N workers (default 8 —
  the benchmark's line-rate point), each with its own `cuObjClient` session
  and a persistent keep-alive control-plane connection — a **libcurl easy
  handle** (Sirius already links libcurl for `s3_reactor`; the benchmark's
  cpp-httplib and its connection-per-GET pattern are not carried over). The
  callback inside `cuObjGet` is synchronous, so a blocking easy-handle request
  is the right shape; the same connection serves the reactor's host reads.
  Workers are *not* bound to GPUs.
- **`s3_rdma_max_inflight` scope**: it is the **worker-thread count**, and
  because each worker runs one *blocking* `cuObjGet` at a time, it is therefore
  the **global** ceiling on concurrent in-flight GETs — there is no separate
  per-device inflight bound. "Per device" applies only to **arena placement**:
  a worker `cudaSetDevice`s to `r.device_id` and takes a slot from that
  device's arena, so each active device's arena is sized for the worst case
  (all workers targeting it) = `max_inflight × slot_size`. Total registered GPU
  memory = `active_GPUs × max_inflight × slot_size`, but total in-flight stays
  `max_inflight` regardless of GPU count.
- **Multi-GPU**: the S3 ioctx is a process singleton; every request carries
  the caller's device id (stamped at enqueue). Per request:
  `cudaSetDevice(r.device_id)` → use **that device's** arena → flush on that
  device's context → D2D on `r.stream`. Slot and `r.dst` are same-device by
  construction (never a cross-device copy). NIC↔GPU PCIe affinity: list all
  RoCE NICs in `cufile.json`'s `rdma_dev_addr_list`; cufile selects the
  nearest NIC per GPU; P4 validates with per-GPU `gdscheck -p` + bandwidth
  sweeps. With a single NIC all GPUs share ~12.5 GB/s; the gateway slot budget
  must cover `max_inflight` total (the global worker-pool ceiling), **not**
  `max_inflight × GPUs` — in-flight is bounded by the worker pool regardless of
  how the requests spread across devices.
- **Memory budget**: zero host staging for bulk object bytes (vs `s3_reactor`'s 128 × 1 MiB pinned
  staging per reactor); device-side cost is the arena (~8 MiB/device at defaults) +
  descriptors + sessions.
- **Teardown**: stop intake → workers drain the queue and complete/fail all
  outstanding D2D completions (the workers are the ones that observe callbacks,
  so this must happen *before* they exit) → join workers → deregister + free
  arenas (`cuMemObjPutDescriptor`,
  `cudaFree`) → destroy sessions. Mirrors the mandated
  scan_manager → ioctx → pool order.

## A.5 Future: direct-into-RMM (gated spike)

Revisit only if (a) a registrable allocation path for consumer buffers
appears (dmabuf for mempools, or a cucascade registrable IO sub-MR — §8 Q4),
and (b) enclosing-region descriptors prove valid for interior
pointers on the rig. Any large-region registration is bounded by
`CUOBJ_MAX_MEMORY_REG_SIZE` (4 GiB − 64 KiB), so pool-scale registration
needs segmenting regardless. Upside is the <1% D2D — low priority.

# Appendix B — Verified source facts

> Line numbers are a snapshot (verified 2026-06-13 against the local tree) and
> must be re-confirmed before any implementation PR — the IO layer moves.

| Fact | Source |
|---|---|
| Super Sirius device memory is `cuda_async_memory_resource` (cudaMallocAsync) | `cucascade/src/memory/memory_space.cpp:121-126`, `memory/common.cpp:256-258` |
| Only the dead legacy path uses a `cudaMalloc`-backed pool | `sirius/src/legacy/gpu_buffer_manager.cpp:195-196` |
| The S3 backend is owned by the scan manager (one ioctx in `_io_ctxs`), built from the scan_manager_config; `SiriusContext` only composes that config | `sirius/src/sirius_context.cpp:553`, `sirius/src/scan_manager/sirius_scan_manager.cpp:76` |
| `create_datasource(path)` is the resolution entry: iterate `_io_ctxs` → `supports → create_io_object → make_datasource` | `sirius/src/scan_manager/sirius_scan_manager.cpp:493` |
| `device_read()` consults `_cache` before `device_read_io`; a byte-range hit copies host-pinned slices → device, bypassing the reactor. Scan-side byte-range prewarm is gated by `chunk_prewarm_enabled()`; `describe_parquet`'s insert is metadata-only | `sirius/src/io/io_context.cpp:219-226`, `sirius/src/op/scan/parquet_gpu_ingestible.cpp:414` |
| `s3_ioctx` overrides `host_read_ranges_async_io` for strict validation; the generic `templated_ioctx` version silently skips invalid/too-small ranges | `sirius/src/include/io/s3/s3_ioctx.hpp:67`, `sirius/src/include/io/templated_ioctx.hpp:243` |
| Per-GPU uring map exists for `file` scheme (contrast) | `sirius/src/sirius_context.cpp:422-464` |
| Requests carry the caller's device id (stamped at enqueue) | `sirius/src/include/io/templated_ioctx.hpp:328-347` |
| `s3_reactor` sets device before each H2D copy | `sirius/src/io/s3/s3_reactor.cpp:687-688` |
| `chunk_done()` fires the user handler synchronously on the last chunk | `sirius/src/include/io/types.hpp:173` |
| CUDA callback only marks + wakes; reactor thread completes | `s3_reactor.cpp:749` (`cuda_copy_done`), `:728` (`poll_device_copies`) |
| Callback-registration failure ⇒ sync drain before staging release | `s3_reactor.cpp:713` (and `:705` for enqueue failure) |
| `is_device_read_preferred` returns unconditional `true` — device reads go RDMA immediately on backend swap | `sirius/src/io/sirius_datasource.cpp:45-47` |
| `s3_ioctx : templated_ioctx<s3_reactor>` with instance `create_io_object` via the authorizer | `sirius/src/include/io/s3/s3_ioctx.hpp` |
| `object_store_config::transport { AUTO, HTTP, RDMA }` + SigV4 presigned/header modes already exist | `sirius/src/include/io/object_store_config.hpp` |
| Sirius `pixi.lock` resolves `libcufile` but not `libcuobjclient` | `sirius/pixi.lock` |
| `CUOBJ_MAX_MEMORY_REG_SIZE = 4 GiB − 64 KiB` | cuObject SDK `cuobjclient.h:22` |
| Per the cuObject spec, the client conveys the RDMA descriptor — which **encodes the remote buffer/slot address** — in a single request tag `x-amz-rdma-token` (inserted by `cuObjGet`); the gateway replies with `x-amz-rdma-reply` carrying RDMA status. `cuObjGet`'s `size`/`offset` args (not headers) bound the transfer. **Verified 2026-06-15 against the public doc**: both tags confirmed; the reply tag conveys RDMA **status** (no byte count specified) | cuObject spec §1.3.2 (RDMA GET/PUT workflow) + §1.3.3 (data-flow sequence), docs.nvidia.com/gpudirect-storage/cuobject |
| Token encodes the address **even on the pinned 1.0.0 SDK**: client `cuMemObjGetRDMAToken(ptr, size, buffer_offset, …)` modifies the descriptor's address/size fields. But local `handleGetObject` still takes `remote_buf_start` as a separate param (1.0.0; public doc is 1.2.0) — see C.3 migration caveat | `s3-over-rdma/third_party/cuobj-1.0.0/.../cuobjclient.h:183`, `cuobjserver.h:227` |
| Benchmark: ~91 Gbps GPU GET, line rate at 8 threads; register-once pattern | `s3RDMA-benchmarktool/doc/test.md` |
| `cuObjGet` invokes the client callback once per object; chunking is server-side | `s3RDMA-benchmarktool/doc/internals.md` |

# Appendix C — Protocol & config reference

## C.1 Control-plane headers

Per the cuObject spec (§1.3.2, §1.3.3), the standard RDMA control protocol uses
exactly **two** custom header tags, plus a standard `Range`:

| Header | Direction | Meaning |
|---|---|---|
| `x-amz-rdma-token` | client → gateway | base64 of the opaque cuObject RDMA descriptor; **encodes the remote buffer/slot address** (inserted by `cuObjGet`/`cuObjPut`) |
| `x-amz-rdma-reply` | gateway → client | standard RDMA status of the offloaded transfer. The **authoritative** short-write check is `cuObjGet`'s returned `n == expected` (not HTTP 200/206 alone); the reply tag carries the status — don't assume it itself equals a byte count unless the reply payload format is confirmed |

Plus standard `Range: bytes=a-b` (206 for range responses). On production
gateways every control request is SigV4-signed and validated before the RDMA
tag is honored.

The benchmark had accreted **four** extra headers that are **not** part of the
cuObject spec; Sirius drops them to stay on the standard protocol:

| Dropped header | Why it's gone |
|---|---|
| `x-cuobj-remote-addr` | the remote slot address is encoded inside `x-amz-rdma-token`; a separate header is redundant under the spec |
| `x-cuobj-chunk-size` | benchmark tuning value, not a cuObject primitive; derive it from the arena `slot_size` or make it gateway config |
| `x-cuobj-size` | destination capacity is the `cuObjGet` `size` arg, not a wire header; the fixed `slot_size` + bounded `Range` already prevent over-writes |
| `x-amz-rdma-bytes-transferred` | superseded by the spec's `x-amz-rdma-reply` (RDMA status); the authoritative byte-count check is `cuObjGet`'s return value |

## C.2 Config (under the existing `object_store_config`)

```text
s3_transport = HTTP | RDMA           # existing enum; AUTO resolves to HTTP in v1
s3_rdma_endpoint / s3_rdma_port      # control-plane endpoint if distinct
s3_rdma_chunk_size                  # local 1 MiB chunking (templated_ioctx); NOT a wire header — arena slot_size derives from it
s3_rdma_max_inflight                # worker count = global in-flight ceiling (NOT per-device; see A.4)
s3_rdma_arena_slot_size             # ≥ chunk size; arena = max_inflight × slot_size
```

Possible P5 addition (post-measurement): `s3_rdma_min_device_read_size` — an
ioctx-level whole-read threshold routing small device reads over HTTP. Not in
v1.

Proposed counters: `s3_rdma_bytes_total`, `s3_rdma_requests_total`,
`s3_rdma_arena_slot_wait_total`, `s3_rdma_flush_total`,
`s3_rdma_short_write_total`, `s3_rdma_error_total`, `s3_rdma_inflight_peak`.

Integration follows the existing `s3_ioctx` pattern (no new mechanism): each is
a `std::atomic<uint64_t>` in the reactor, the ioctx aggregates across reactors,
and they surface as pull-based `…_total() const noexcept` accessors alongside
the inherited `bytes_read_total` / `device_copies_total` /
`device_stream_sync_total` (`s3_ioctx.cpp:141-169`, `s3_reactor.hpp:250-252`).
Pushing IO counters into `telemetry_context` (today pull-only) is the open part
— see §8 Q5.

## C.3 P0 — benchmark protocol migration + hardening (lands in `s3RDMA-benchmarktool`)

P0 is a **wire-protocol migration**, not just hardening: the benchmark today
sends `x-cuobj-remote-addr` / `x-cuobj-size` / `x-cuobj-chunk-size` and its
server *requires* the remote-addr header (`client/s3_client.cpp`,
`server/s3_server.cpp`). Moving to the standard cuObject protocol means:

- Drop the four non-standard headers; the server obtains the slot address by
  **decoding it from the token** rather than reading `x-cuobj-remote-addr`.
  Implementation caveat: the pinned SDK is cuObject **1.0.0**, whose
  `handleGetObject` still takes `remote_buf_start` as an explicit param — so
  this needs either a server-side token-decode path or an SDK bump toward the
  public **1.2.0** protocol. Do not assume the old benchmark server runs
  unchanged.
- Server enforces the destination capacity (from the `cuObjGet` `size` arg /
  the fixed arena `slot_size`) and rejects writes larger than the destination.
- Short positive RDMA writes are errors — the **authoritative** check is
  `cuObjGet`'s returned `n == expected`; `x-amz-rdma-reply` is the standard
  status tag (do not assume it itself equals the byte count).
- O_DIRECT alignment edge cases.
- Document callback-once-per-object semantics.
- Optional stretch: SigV4 validation mode in the benchmark server.
