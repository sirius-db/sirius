# PLAN-01 — Copy-out on arrival

**Status: PLAN ONLY. Nothing in this document has been implemented.**
Written 2026-08-20 against `/home/ubuntu/sirius` @ branch `demo-multi-cn` (default branch is `dev`).

Every claim below was checked against the tree at that commit and is cited `file:line`. Anything
that could not be checked is marked **UNVERIFIED**. Line numbers drift — re-`grep` the quoted text
before trusting a number.

---

## 0. Orientation for a fresh session

You are working on **Sirius**, a GPU SQL engine that also ships as an embeddable engine behind a
C++/Rust FFI. `experimental/starrocks/` contains a Rust **StarRocks compute node (CN)** that links
that engine and runs plan fragments on one GPU. A cluster is N CNs (one per GPU) plus a StarRocks
FE. Fragments exchange data across CNs over **nixl/UCX RDMA writes into a device staging arena**.

Required reading before touching engine code: `docs/super-sirius/README.md` and, for this task,
`docs/super-sirius/memory-management.md`. **Everything under `src/legacy/` is dead — do not modify it.**

Build and test:

```bash
# C++ engine (repo root)
pixi run make                       # full build
pixi run make clean                 # after a failed build
pixi run make test                  # build + Catch2 unit tests
pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[staging_arena]"   # by tag

# Rust CN (experimental/starrocks)
pixi run cn-build                   # engine-linked release binary  (pixi.toml:131)
pixi run cn-test                    # engine-linked Rust tests      (pixi.toml:136)
pixi run cn-test-no-engine          # pure-Rust tests, no GPU; what CI runs (pixi.toml:148)

# Rust FFI wrapper tests (repo root) — require a GPU
cargo test -p sirius
```

The box this plan targets: 2× **RTX PRO 6000 Blackwell**, `nvidia-smi` reports 97887 MiB/card but
**638 MiB is driver-reserved — 94.97 GiB is allocatable** (verified: `nvidia-smi --query-gpu` on
2026-08-20 reports `97887 MiB`, driver `580.126.09`; the 94.97 GiB figure is carried from
`bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md:6-7`). 2 CNs, one per GPU. Dataset
`/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64` (159 GiB, DECIMAL cast to DOUBLE).

**To see any engine log at all you MUST set `SIRIUS_LOG_BACKEND=spdlog`.** Only `duckdb`, `spdlog`,
`noop` are accepted (`src/sirius_context.cpp:1556-1578`). On the CN's FFI path an unknown value is
**silently discarded**: the `throw` for an unrecognised backend is guarded by `else if (db)`
(`src/sirius_context.cpp:1573-1577`) and the CN calls `install_configured_log_sink(nullptr)`
(`src/sirius_ffi.cpp:177`). Also set `SIRIUS_LOG_DIR` and `SIRIUS_LOG_LEVEL`
(`src/sirius_ffi.cpp:170-180`). `up-sf500-x.sh` already does all three.

---

## 1. Problem statement

### 1.1 What the code claims

Two comments in the CN assert that copy-out-on-arrival is already implemented:

* `experimental/starrocks/src/engine.rs:60-63` — the `ExecuteRequest::remote_inputs` doc:
  *"pushed via `push_packed` + `close_input` before `run()`, each lease released the moment its push
  returns (copy-out-on-arrival makes that safe)."*
* `experimental/starrocks/src/engine.rs:543-545` — *"Push each (deep copy into pool memory), release
  its lease immediately — copy-out-on-arrival makes that safe — then close the sender."*
* `experimental/starrocks/src/fragment_executor.rs:74-77` — the same sentence on `FragmentRun::remote_inputs`.
* `rust/crates/sirius/src/lib.rs:334-341` and `:131-134`, and `src/include/sirius_ffi.hpp:101-103`,
  all repeat *"copy-out-on-arrival"*.

It is also a named architecture decision: `PLAN-PATH-B.md:41-42`, **D-B5 — copy-out-on-arrival**:
*"received batches leave the lease immediately into ordinary pool memory — leases stay short-lived,
no lease-aware spill needed."*

### 1.2 What the code does

**The copy happens at RUN time, not at ARRIVAL time.** Verified end to end:

1. `SiriusComputeNodeService::handle_transmit_packed` (`experimental/starrocks/src/compute_node_service.rs:605-666`)
   builds a `StagedBatch { metadata, offset, len }` from the frame (`:637-651`) and hands it to
   `LocalExchange::push_remote_frame` (`:659-670`). **It performs no device work whatsoever.** The
   only device-touching branch is the canary lease release at `:610-617`.
2. `LocalExchange::push_remote_frame` (`experimental/starrocks/src/local_exchange.rs:159-247`)
   appends the batch to `SenderSource::Remote { batches: Vec<StagedBatch>, .. }` —
   `batches.push(batch)` at `local_exchange.rs:240`; the field is declared at `local_exchange.rs:44`.
   The lease is never touched.
3. `LocalExchange::take_ready` (`experimental/starrocks/src/local_exchange.rs:248-311`) returns
   `Ok(None)` until **every** sender of **every** exchange input of the receiver has closed —
   `if complete != expected { return Ok(None); }` at `local_exchange.rs:277-279`.
4. Only then is the receiver dispatched (`compute_node_service.rs:668` → `dispatch` at `:257-261` →
   the `fragment-dispatch` thread at `:243-248`, body `dispatch_worker` at `:304-314`) and run
   (`execute_ready_fragment` at `:1129`, remote batches collected at `:1186`).
5. The lease is released inside `run_fragment_inner`'s push loop —
   `experimental/starrocks/src/engine.rs:541-575`: `fragment.push_packed(...)` at `:556`, then
   `context.staging_release(batch.offset)` at `:563`, recorded at `:569`.
6. `Fragment::push_packed` is where the deep copy actually happens:
   `auto table = std::make_unique<cudf::table>(unpacked, stream, gpu_space->get_default_allocator());`
   at **`src/sirius_ffi.cpp:849`**, preceded by the "Copy-out-on-arrival (PLAN-PATH-B D-B5)" comment
   at `:845-847`, followed by `stream.synchronize()` at `:850` and
   `sirius::make_data_batch(...)` + `session().push(...)` at `:853-858`.

**Therefore peak arena demand is the receiver's ENTIRE remote input, not batch × window.**

### 1.3 Why that is worse than "just a big number"

The arena is a hard-failing, fixed-size `cudaMalloc` slab that lives **outside every pool** by
contract (`src/include/exec/exchange_staging_arena.hpp:26-38`). `lease()` is an address-ordered
first fit that **throws on first-fit failure and never blocks** —
`src/exec/exchange_staging_arena.cpp:212-258`, the exhaustion throw at `:245-256`. There is no
back-pressure anywhere on the lease path.

So the failure mode is a **deadlock-shaped ratchet**:

* `push_packed`'s copy allocates from the RMM pool (`src/sirius_ffi.cpp:849`).
* If the pool is at its ceiling, that allocation throws
  (`cucascade/src/memory/reservation_aware_resource_adaptor.cpp:460-483`, `LIMIT_EXCEEDED` at `:477-481`).
* The push loop never reaches `staging_release` (`engine.rs:563`), so no lease is returned.
* Peer senders keep asking for leases (`request_staging_lease`, `compute_node_service.rs:512-540`
  → `handle_staging_lease` `:1106-1110`), which are served off the engine thread by design
  (`engine.rs:112-121`, `:690-700`) and therefore keep succeeding until the arena is full.
* Arena occupancy ratchets to capacity and never comes back.

**Measured proof** (`bench/rtxpro6000-2gpu/SF500-CONFIG-AND-ARCHITECTURE.md:79-82`): at a 45 GiB
pool / 48 GiB arena split, four `arena exhausted` events **82 seconds apart** reported
**byte-identical** state:

```
642284544 free of 51539607552 capacity in 10 blocks, 67 leases outstanding holding 50897323008 bytes
```

`51539607552` = 48 GiB; `50897323008` = **47.40 GiB live**. Once wedged, not one byte was ever
released. The message format is `src/exec/exchange_staging_arena.cpp:245-256`.

The same q21, same SF, same CN count measured **47.40 GiB** arena live with a starved pool versus
**≤ 26.78 GiB** with a healthy one (`SF500-CONFIG-AND-ARCHITECTURE.md:83-85`). No `(SF, N)` sizing
formula can produce a 1.8× spread — the arena is a **pressure gauge on the pool**, not an
independent demand.

### 1.4 Why this is what q09 needs

Current best on this box: SF500 **21/22** with `GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB` plus
operator budgets `hash_partition_bytes=1GiB max_build_hash_table_bytes=2GiB scan_task_batch_size=1GiB
concat_batch_bytes=1GiB` (`SF500-CONFIG-AND-ARCHITECTURE.md:13-24`). Only **q09** genuinely fails.

q09's shape (`SF500-CONFIG-AND-ARCHITECTURE.md:101-116`): within **one** window it grows
5.85 GiB → 56.53 GiB and peaks at **64,410,655,403 B = 99.98 %** of the 60 GiB cap
(60 GiB = 64,424,509,440 B). It does **not** respond to batch size — identical failure at 2.39 GiB,
1 GiB and 512 MiB. Every split fails, alternating which resource breaks:

| Pool / Arena | Budgets | Failure |
|---|---|---|
| 60/32 | 2.39GiB / 1GiB / 512MiB | pool OOM (all three) |
| 68/24 | default | pool OOM |
| 76/16 | default | arena exhausted |
| 70/22 | 512MiB | arena exhausted |
| 65/27 | 512MiB | arena exhausted |

The arena "needs" ≥ 24 GiB **only because of the defect in §1.2**. Remove it and arena demand
collapses to roughly `peers_draining_concurrently × p_max` (≈ 1–2.5 GiB here, see §8.1), freeing
~30 GiB/CN for the pool. That is the headroom q09 needs. q09's SQL is a six-way join
(`experimental/starrocks/benchmarks/tpch/queries/q09.sql`).

### 1.5 The corroborating architecture facts

* Sender drains are **strictly serial**: `TransportState::send_fragment`
  (`experimental/starrocks/src/nixl_transport.rs:697-780`) does export → peer lease (`:712`) →
  `write_and_wait` (`:713-719`) → `transmit_packed` (`:724-738`) → release own local lease (`:743`),
  one batch at a time, with **no pipelining and no depth knob**. So a *sender* holds exactly one
  remote lease at a time.
* There is **one transport thread per CN** handling every drain inline
  (`nixl_transport.rs:375-431`, `send_fragment` called at `:406`).
* `transmit_packed` is only sent after the RDMA WRITE reports DONE (`nixl_transport.rs:815-829`,
  then `:724`) — so the bytes in the lease are complete and stable when the receiver sees the frame.
* The arena therefore accumulates **only** because the receiver never drains it until the whole
  sender set has closed. Fix that and per-peer live leases drop to one.

---

## 2. Current control flow, end to end

```
SENDER CN                                        RECEIVER CN
---------                                        -----------
engine thread
  Fragment::export_packed                        (arena is the receiver's)
    src/sirius_ffi.cpp:707-779
    leases total + 8 MiB locally (:757)
      kPackChunkBytes = 8 MiB (sirius_ffi.cpp:704)
      ↑ via EngineRequest::ExportNext (engine.rs:86-91, arm :304-307, export_next :336-357)

transport thread (one per CN)
  nixl_transport.rs:375-431 → send_fragment :697
    rpc_request_lease(len)   :712  ─────────────►  transmit RPC thread pool
                                                   request_staging_lease
                                                     compute_node_service.rs:512-540
                                                     spawn_blocking :519
                                                   handle_staging_lease :1106-1110
                                                   SiriusEngine::staging_lease engine.rs:690-694
                                                     → StagingArena handle, OFF the engine thread
                                                       (INVARIANT engine.rs:112-121)
                                                   exchange_staging_arena::lease
                                                     src/exec/exchange_staging_arena.cpp:212-258
                                                     throws on exhaustion :245-256
    write_and_wait(...)      :713-719 ═══RDMA══►  bytes land in [base+offset, +len)
    rpc_transmit(seq, meta)  :724-738 ─────────►  transmit_packed
                                                     compute_node_service.rs:546-566
                                                     spawn_blocking :554-557
                                                   handle_transmit_packed :605-666
                                                     builds StagedBatch :637-651
                                                     push_remote_frame :659-670
                                                       local_exchange.rs:159-247
                                                       dedup by seq :182-192 (:189 log)
                                                       gap → loud error :193-198
                                                       batches.push(batch) :240
                                                       eos sets closed :242-244
                                                       take_ready :246 → :248-311
                                                         returns None until
                                                         complete == expected :277-279
                                                   ── NO DEVICE WORK, NO COPY, NO RELEASE ──
    release local lease      :743
    ... loop ...
    rpc_transmit(eos)        :762-776 ─────────►  same handler; eos completes the sender set
    drop_parked              :777                 take_ready returns Some(ReadyFragment)
                                                   dispatch(ready) :668 → :257-261
                                                     std::mpsc → "fragment-dispatch" thread
                                                       spawned :243-248, body :304-314
                                                   run_ready_fragment :677
                                                   execute_ready_fragment :1129-1193
                                                     remote_inputs.push(...) :1186
                                                   SiriusEngine::run → EngineRequest::Run
                                                     engine.rs:666-681
                                                   ──► "sirius-engine" thread (engine.rs:139-142,
                                                        loop :263; SiriusContext is !Send,
                                                        module doc engine.rs:1-22)
                                                     run_fragment :385-421
                                                       run_fragment_inner :427-...
                                                         remote push loop :541-575
                                                           Fragment::push_packed :556
                                                             src/sirius_ffi.cpp:784-858
                                                               range guard :797-803
                                                               cudf::unpack :808 (aliases lease)
                                                               schema validation :812-836
                                                               memory space :838-843
                                                          ►►► DEEP COPY :849  ◄◄◄
                                                               stream.synchronize() :850
                                                               make_data_batch + push :853-858
                                                           staging_release :563  ← FIRST RELEASE
                                                         close_input :572
                                                       fragment.run() :578
```

Supporting thread-model facts (all verified):

* `transmit_packed` runs on `spawn_blocking`, not the BRPC I/O thread
  (`compute_node_service.rs:554-557`). The BRPC server is a **current-thread** tokio runtime
  (`main.rs:578-585`) itself inside a `spawn_blocking` of the process runtime (`main.rs:577`).
* One task per accepted connection (`brpc.rs:123-128`); `handle_connection` is a **sequential**
  read → dispatch → write loop that awaits the handler before reading the next frame
  (`brpc.rs:150-192`, await at `:177`).
* `PrpcClient` keeps **one cached connection per peer with one request in flight**
  (`prpc_client.rs:36-47`).
* ⇒ **Frames from a given sender CN are serialized at the receiver. Different sender CNs are
  concurrent.** This is load-bearing for the design in §3.
* Receiver execution is single-threaded: one `fragment-dispatch` thread
  (`compute_node_service.rs:304-314`) feeding the single `sirius-engine` thread
  (`engine.rs:139-142`, `:263`).
* Timeouts: `SIRIUS_CN_RPC_TIMEOUT_SECS` (default 60 s, `tunable.rs:37-42`) is the socket
  connect/read/write timeout applied per RPC in `prpc_client.rs:152-168`.
  `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS` (default 30 s, `tunable.rs:49-54`) bounds one RDMA WRITE
  (`nixl_transport.rs:810-821`).

---

## 3. Design

### 3.1 The move, in one sentence

Perform the `cudf::unpack` + deep copy **inside `handle_transmit_packed`, on the RPC blocking
worker, before the frame is recorded in the rendezvous** — and release the lease there. The
rendezvous then holds a device-resident, pool-owned table instead of a lease reference, and
`push_packed` at run time is replaced by a zero-copy `push_resident`.

### 3.2 What owns the destination buffer, and which memory space it comes from

**It must come from the RMM pool** — specifically the same allocator `push_packed` uses today:
`gpu_space->get_default_allocator()` where `gpu_space =
ctx.context->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0)`
(`src/sirius_ffi.cpp:838-843`). That allocator is the
`cucascade::memory::reservation_aware_resource_adaptor` wrapping an
`rmm::mr::cuda_async_memory_resource` (`cucascade/src/memory/memory_space.cpp:103-135`, allocator
returned at `:360-372`).

Three consequences, all of them the point of the change:

1. The bytes are **checked against the pool ceiling** (`memory.gpu.usage_limit_bytes`,
   `src/sirius_config.cpp:318-322`): `do_allocate_unmanaged` does
   `_total_allocated_bytes.try_add(tracking_bytes, _capacity)` and throws `LIMIT_EXCEEDED` otherwise
   (`cucascade/src/memory/reservation_aware_resource_adaptor.cpp:460-483`).
2. The bytes are **visible to the spill/downgrade trigger**, which reads the same atomic
   (`cucascade/src/memory/memory_space.cpp:292-313` → `get_available_memory()` at `:329-341` →
   adaptor `:276-279`). Arena bytes are invisible to it by construction
   (`src/include/exec/exchange_staging_arena.hpp:26-38`).
3. The batch becomes an ordinary `cucascade::data_batch` at push time and is accounted and
   spillable like any other.

**Ownership.** A new opaque FFI type owns the `std::unique_ptr<cudf::table>` from the moment the
copy returns until the engine thread pushes it (or the query dies). Rust holds it as a
`UniquePtr<ResidentBatch>` inside `SenderSource::Remote`, so its lifetime is exactly the
rendezvous entry's lifetime. **Do not** build the `cucascade::data_batch` at arrival:
`make_data_batch` is a cheap wrapper (`src/include/data/data_batch_utils.hpp:154-167`; it only
moves the table, bumps an atomic id at `:42/:47`, and records a writer event), so building it at
push time on the engine thread costs nothing and keeps the batch out of any repository until it
belongs to a running fragment.

### 3.3 Which thread does the copy — and why not a dedicated one

**Do it inline on the `transmit_packed` blocking worker.** Reasons, in order of weight:

1. **It cannot be the engine thread.** `duckdb::SiriusContext` is `!Send`/`!Sync` and lives on one
   dedicated thread (`engine.rs:1-22`, `:139-142`). Routing the copy through
   `EngineRequest` would queue it behind whatever fragment that thread is running — exactly the
   starvation the staging-lease invariant already forbids (`engine.rs:112-121`, and the q02 wedge
   it documents). A copy that waits for the engine thread reproduces today's bug.
2. **A dedicated copier thread with a queue re-creates the bug in a new place.** If arrivals are
   queued for a background copier, the queue holds leases, and a slow copier lets the queue — and
   the arena — ratchet exactly as today. Copying inline makes the lease's lifetime equal to one
   RPC handler invocation, with no queue anywhere.
3. **Inline copying is the back-pressure** (see §4). Because `handle_connection` awaits the handler
   before reading the next frame (`brpc.rs:150-192`, await at `:177`) and `PrpcClient` keeps one
   in-flight request per peer (`prpc_client.rs:36-47`) and the sender's drain is serial
   (`nixl_transport.rs:697-780`), a slow copy directly throttles that peer's sender. No new
   mechanism is needed.
4. **Bounded concurrency.** From (3), at most **one** arrival copy per peer CN can be in flight.
   Peak concurrent arena leases per receiver becomes `(number of peer CNs) × 1`.

But the copy must be made thread-correct, which today's `push_packed` is not, because today it only
ever runs on the engine thread:

* **Set the device.** `Fragment::push_packed` (`src/sirius_ffi.cpp:784-858`) contains no
  `cudaSetDevice`. Every other GPU-touching pool pins itself —
  `src/pipeline/gpu_pipeline_executor.cpp:79-86` and `src/downgrade/downgrade_executor.cpp:83-96`.
  The new entry point must do `rmm::cuda_set_device_raii{gpu_space->get_device_id()}`. It happens to
  be harmless on this box (each CN exports one GPU via `CUDA_VISIBLE_DEVICES`, set from
  `--gpu-device` in `SiriusEngine::configure_engine_environment`,
  `experimental/starrocks/src/engine.rs:205-218`), but it is wrong on any
  multi-GPU-per-process build.
* **Do NOT use `cudf::get_default_stream()`.** Per-thread default streams are **OFF** in this build
  (no `CUDA_API_PER_THREAD_DEFAULT_STREAM` anywhere in `CMakeLists.txt`/`cmake/`/`pixi.toml`; the
  installed cudf exports empty `CUDF_CXX_DEFINITIONS`, and `cudf::is_ptds_enabled()` in the shipped
  `libcudf.so` returns `false`). So `cudf::get_default_stream()` is the **legacy null stream**, and
  `stream.synchronize()` at `src/sirius_ffi.cpp:850` is a process-wide blocking-stream barrier. On an
  RPC worker that would serialize the whole engine against every arriving batch. Use
  `gpu_space->acquire_stream()` instead — a real non-blocking stream from the space's 16-stream RMM
  pool (`cucascade/include/cucascade/memory/memory_space.hpp:105`,
  `cucascade/src/memory/memory_space.cpp:269-275`, pool built at `:108-111`).
* **Keep the synchronize.** After the copy on that stream, `stream.synchronize()` is what makes the
  lease reusable on return. It is now a per-stream sync, not a global one.
* **Thread-safety of everything else is already established.** `get_memory_space` is a lookup in a
  map built once in the ctor and never mutated
  (`cucascade/src/memory/memory_reservation_manager.cpp:167-177`, built at `:122-142`/`:316-334`).
  `memory_space` documents thread-safe reservation management
  (`cucascade/include/cucascade/memory/memory_space.hpp:61`) and the adaptor documents
  "Thread-safe operations using atomic operations and mutexes"
  (`cucascade/include/cucascade/memory/reservation_aware_resource_adaptor.hpp:43`); the allocation
  counters are atomics (`cucascade/src/memory/reservation_aware_resource_adaptor.cpp:460-483`).
  `make_data_batch`'s only global is an atomic id (`src/include/data/data_batch_utils.hpp:42,47`),
  and the telemetry probe is a no-op when `batch_telemetry_info.context == nullptr`, which is what
  the FFI passes (`src/sirius_ffi.cpp:854`).
* **Allocation-accounting asymmetry: pre-existing, not introduced.** Sirius configures
  `AllocationTrackingScope::PER_THREAD` (`src/sirius_config.cpp:67` forces
  `per_stream_reservation = false`; also `:310`, applied `:350`), and the tracker state is
  `thread_local` (`cucascade/src/memory/reservation_aware_resource_adaptor.cpp:95-150`, TLS map at
  `:113`). A thread with no attached reservation allocates through the untracked path
  (`:406-416` → `:418-433` → `:460-483`); `deallocate` (`:484-513`) subtracts from whatever
  reservation the **deallocating** thread happens to hold, without recording which allocations were
  reservation-backed. **This asymmetry already exists today**: `push_packed` runs on the engine
  thread *before* `fragment.run()`, and the engine thread never attaches a reservation (reservations
  are attached on GPU pipeline task threads — `src/pipeline/gpu_pipeline_task.cpp:514-518`), while
  the batch is freed later on a task thread that may hold one. Moving the allocation from the engine
  thread to an RPC worker does not change the class. It is listed as a risk in §9 with a detection
  procedure, but it is **not a blocker for this plan**.

### 3.4 New FFI surface

Mirroring the existing `StagingArena` split (a `Send + Sync` handle carved out of the `!Sync`
context — `src/include/sirius_ffi.hpp:115-119`, `:142-172`; `src/sirius_ffi.cpp:329-358`;
bridge `rust/crates/sirius-sys/src/lib.rs:76-107`; wrapper `rust/crates/sirius/src/lib.rs:383-431`),
add two types.

**`src/include/sirius_ffi.hpp`** (new declarations near `StagingArena` at `:142`):

```cpp
/// One remote batch already copied out of the staging arena into pool memory. Opaque owner of a
/// cudf::table; the lease it came from is already released when this exists.
class SIRIUS_FFI_EXPORT ResidentBatch {          // move-only, NOT copyable
 public:
  ~ResidentBatch();
  [[nodiscard]] std::uint64_t num_rows() const noexcept;
  [[nodiscard]] std::uint64_t device_bytes() const noexcept;   // for logging/telemetry
 private:
  struct state; std::unique_ptr<state> impl_;    // holds std::unique_ptr<cudf::table>
};

/// Thread-safe handle that turns an arena lease into a `ResidentBatch`. Shares the arena
/// `shared_ptr` with the Context (same allocator, never a mirror) and holds a raw pointer to the
/// GPU memory_space, which is owned by the memory manager and outlives every fragment.
class SIRIUS_FFI_EXPORT ArrivalCopier {
 public:
  /// Unpack `length` bytes at arena `offset` with `metadata_len` bytes of cudf pack metadata at
  /// `metadata_addr`, deep-copy the table into pool memory, and RELEASE THE LEASE.
  /// Callable from any thread: sets the device, uses a stream from the space's pool, and
  /// synchronizes before returning.
  /// @throws on a bad range, a failed allocation (pool ceiling), or any CUDA error. On throw the
  ///         lease is NOT released — the caller decides (see §6).
  std::unique_ptr<ResidentBatch> copy_out(std::uintptr_t metadata_addr, std::size_t metadata_len,
                                          std::uint64_t offset, std::uint64_t length) const;
};
```

`Context::arrival_copier_handle() const -> std::unique_ptr<ArrivalCopier>` (null when no arena),
alongside `staging_arena_handle()` at `src/include/sirius_ffi.hpp:119`.

**`Fragment` gains** `void push_resident(std::uint64_t stream_id, ResidentBatch& batch)`, which:
* runs **the same schema validation that lives at `src/sirius_ffi.cpp:812-836` today**, against the
  owned table's view rather than the unpacked lease view — this guard is the only thing that stops a
  declaration/payload disagreement being reinterpreted downstream, and it must not be lost;
* calls `sirius::make_data_batch(std::move(table), *gpu_space, writer_stream, {})`
  (`src/include/data/data_batch_utils.hpp:154-167`) and `session().push(...)`, i.e. exactly
  `src/sirius_ffi.cpp:853-858` minus the copy.

`ResidentBatch` must carry the writer stream (or an event) recorded by `copy_out` so
`make_data_batch`'s STREAM-LINEAGE contract (`src/include/data/data_batch_utils.hpp:113-131`) is
honoured. Since `copy_out` synchronizes, passing the same stream view is sufficient and correct.

**Implementation sketch of `ArrivalCopier::copy_out`** (all pieces exist today in
`src/sirius_ffi.cpp:784-858`; this is a re-composition, not new algorithm):

```cpp
// 1. range guard — verbatim from sirius_ffi.cpp:797-803
// 2. rmm::cuda_set_device_raii dev{space_->get_device_id()};            // NEW (§3.3)
// 3. auto view = cudf::unpack(metadata, base + offset);                 // = sirius_ffi.cpp:808
// 4. auto stream = space_->acquire_stream();                            // NEW: not the null stream
// 5. auto table = std::make_unique<cudf::table>(view, stream, space_->get_default_allocator());
// 6. stream.synchronize();                                              // = sirius_ffi.cpp:850
// 7. arena_->release(offset);                                           // NEW: release HERE
// 8. return ResidentBatch{std::move(table), stream};
```

Note the schema check is deliberately **not** in `copy_out`: at arrival the receiver fragment has
not been built, so `impl_->resolved_inputs` (`src/sirius_ffi.cpp:411`, populated at `:588`) does not
exist yet. The check moves to `push_resident`, unchanged in substance.

**Rust bridge** (`rust/crates/sirius-sys/src/lib.rs`, alongside `:83-107`): declare
`type ResidentBatch`, `type ArrivalCopier`, `fn arrival_copier_handle(self: &Context) ->
UniquePtr<ArrivalCopier>`, `unsafe fn copy_out(...) -> Result<UniquePtr<ResidentBatch>>`,
`fn push_resident(self: Pin<&mut Fragment>, stream_id: u64, batch: Pin<&mut ResidentBatch>) -> Result<()>`.

**Rust wrapper** (`rust/crates/sirius/src/lib.rs`, alongside `:383-431`):
`pub struct ArrivalCopier` with `unsafe impl Send + Sync` (same justification as
`StagingArena`'s at `:392-399`, extended: `copy_out` sets the device and uses a pool stream, and
the memory space and allocator are thread-safe — cite `memory_space.hpp:61`,
`reservation_aware_resource_adaptor.hpp:43`); `pub struct ResidentBatch` with `unsafe impl Send`
**only** (single owner; never shared).

### 3.5 CN-side changes

**Carry both shapes during migration** (`experimental/starrocks/src/fragment_executor.rs`, next to
`StagedBatch` at `:50-65`):

```rust
/// One remote batch, either still in the arena (legacy) or already copied into pool memory.
#[derive(Debug)]
pub enum ArrivedBatch {
    Staged(StagedBatch),                 // legacy: lease released at run time
    Resident(sirius::ResidentBatch),     // copy-out-on-arrival: no lease outstanding
}
```

* `SenderSource::Remote.batches` becomes `Vec<ArrivedBatch>` (`local_exchange.rs:44`).
* `FragmentRun::remote_inputs` and `ExecuteRequest::remote_inputs` become
  `Vec<(i32, i32, Vec<ArrivedBatch>)>` (`fragment_executor.rs:74-77`, `engine.rs:60-63`).
* `run_fragment_inner`'s push loop (`engine.rs:541-575`) matches: `Staged` → today's
  `push_packed` + `staging_release`; `Resident` → `push_resident`, nothing to release.
* The error sweep in `run_fragment` (`engine.rs:385-421`, releases at `:411`) only walks `Staged`
  entries; `Resident` entries free themselves when dropped.
* **`SenderSource` currently derives `Clone`** (`local_exchange.rs:25`). `ResidentBatch` is a
  `UniquePtr` and cannot be `Clone`. Verified: the only `.clone()` in `local_exchange.rs` is on
  `names` (`:205`), and no other module clones a `SenderSource`
  (`compute_node_service.rs:1004,1170,1171` construct/match only). **Drop `Clone` from the derive**;
  if a test needs it, construct instead.
* `StagedBatch` also derives `Clone, Debug, PartialEq, Eq` (`fragment_executor.rs:56`); leave it.

**Two-phase rendezvous ingest** — the copy must happen **outside** the rendezvous mutex (it is a
multi-hundred-MiB device copy) but **after** the duplicate/gap decision, because a brpc
reconnect-replay must not be copied twice and must not release an already-released lease
(`local_exchange.rs:182-192` explains the replay case; today it is safe precisely because nothing
was released). Split `push_remote_frame`:

```rust
/// Validates names/shape, then atomically decides this frame's fate and advances the
/// per-sender sequence. Takes the mutex briefly; does no device work.
pub(crate) fn claim_remote_frame(&self, key, sender_id, seq, eos, names, has_batch)
    -> Result<FrameClaim, String>;

pub(crate) enum FrameClaim {
    Duplicate,      // brpc replay: drop, release nothing, copy nothing
    Accept,         // caller must now copy out, then commit
}

/// Records the (already-copied) batch and/or the eos, in `seq` order, and runs `take_ready`.
pub(crate) fn commit_remote_frame(&self, key, sender_id, seq, eos, names,
                                  batch: Option<ArrivedBatch>)
    -> Result<Option<ReadyFragment>, String>;
```

`commit_remote_frame` must insert **by `seq`**, not by arrival order, so the documented
"arrival (sequence) order" of `batches` (`local_exchange.rs:41-44`) survives even if two commits
ever interleave. Today they cannot (per-connection serialization, §2), but the invariant should not
depend on that.

`handle_transmit_packed` (`compute_node_service.rs:605-666`) becomes:

```
canary?                       → staging_release, return                 (unchanged, :610-617)
parse frame fields                                                       (unchanged, :618-651)
claim_remote_frame(...)?
  Duplicate                   → log + return Ok(())                      (matches today's :182-192)
  Accept, no batch (eos only) → commit_remote_frame(None)
  Accept, with batch:
    if copy_on_arrival:
        resident = copier.copy_out(&meta, offset, len)?  // releases the lease inside
        commit_remote_frame(Some(ArrivedBatch::Resident(resident)))
    else:
        commit_remote_frame(Some(ArrivedBatch::Staged(StagedBatch{ .. })))
if ready → dispatch(ready)                                               (unchanged, :668)
```

**Where the copier handle comes from.** `SiriusEngine::start` already carries the `StagingArena`
handle out of the engine thread on the readiness channel (`engine.rs:139-152`, sent at `:234`).
Widen that channel's payload to a small struct carrying both handles, store the copier next to
`staging` on `SiriusEngine` (`engine.rs:121`), and expose it on the `FragmentExecutor` trait
(`fragment_executor.rs:100-146`) as a defaulted method so `StubExecutor` and the no-engine CI build
(`pixi run cn-test-no-engine`) keep compiling:

```rust
/// Copies a staged remote batch out of the arena into pool memory and releases its lease.
fn staging_copy_out(&self, batch: &StagedBatch) -> Result<sirius::ResidentBatch, String> {
    Err("this fragment executor has no exchange staging arena ...".to_string())
}
```

(Under `--no-default-features` the CN builds without the engine; the `ArrivedBatch::Resident` arm
must be feature-gated the same way the rest of the `sirius::` usage is.)

---

## 4. Back-pressure

### 4.1 Today

Two hard-fail points, neither of which can wait:

* `handle_staging_lease` (`compute_node_service.rs:1106-1110`) either grants or propagates an error.
  The error becomes a non-OK `StatusPB` in `request_staging_lease` (`:527-531`) — the RPC itself
  *succeeded*, so `PrpcClient`'s one-shot reconnect retry (`prpc_client.rs:73-99`) does not apply.
* `exchange_staging_arena::lease` throws on first-fit failure and **never blocks**
  (`src/exec/exchange_staging_arena.cpp:212-258`, throw at `:245-256`; contract stated at
  `src/include/exec/exchange_staging_arena.hpp:85-88`).

The sender treats that as fatal: `check_status` fails in `rpc_request_lease`
(`nixl_transport.rs:883-890`) → `send_fragment` returns `Err` → drain error → `dispatch_then_join`
returns it (`compute_node_service.rs:275-300`) → the whole query fails.

### 4.2 After the change

The pressure point moves from the arena to the pool, and that is strictly better for three reasons:

1. **The pool can actually relieve pressure; the arena cannot.** Arena bytes are outside every pool
   by contract (`src/include/exec/exchange_staging_arena.hpp:26-38`), so no amount of spilling frees
   a lease — measured: *"356 downgrade requests freed 0 bytes"*
   (`SF500-CONFIG-AND-ARCHITECTURE.md:132-135`). Copy-out bytes go through
   `_total_allocated_bytes`, which is exactly what `should_downgrade_memory()` reads
   (`cucascade/src/memory/memory_space.cpp:292-313`). A full pool triggers downgrade; a full arena
   triggers only a throw.
2. **Flow control becomes end-to-end and needs no new mechanism.** The receiver's handler does not
   return until the copy completes; `handle_connection` will not read that peer's next frame until it
   does (`brpc.rs:150-192`, await at `:177`); the peer's `PrpcClient` has one request in flight
   (`prpc_client.rs:36-47`); the peer's drain is serial and inline on its single transport thread
   (`nixl_transport.rs:697-780`, `:375-431`). So the sender is throttled to the receiver's copy rate,
   automatically. The arena stops being a queue and becomes a one-slot-per-peer handoff buffer.
3. **The failure becomes attributable.** "Pool cannot accept a 900 MiB exchange batch while
   allocated=59.7 GiB of 60 GiB" names the real constraint. "Arena exhausted, 47.40 GiB live, no
   change in 82 s" names a symptom.

**What should happen when the pool cannot accept the copy.**

* **Phase 1 (this plan): fail fast, loudly, and leave the lease intact.** `copy_out` throws without
  releasing; `handle_transmit_packed` releases the lease itself, then returns `Err`. The message must
  name **both** sides: the requested bytes, the pool's `allocated/peak/capacity` (available from
  `reservation_aware_resource_adaptor::get_total_allocated_bytes()/get_peak_total_allocated_bytes()`,
  the same accessors `src/sirius_context.cpp:248-259` logs as `[gpu_pool]`), and the arena's
  `live_bytes()/total_free()/largest_free()`
  (`src/include/exec/exchange_staging_arena.hpp:105-120`). Rationale: today the operator gets a
  diagnostic that points at the wrong knob; the whole value of the move is a message that points at
  the right one.
* **Phase 3 (optional, only if measurements demand it): a bounded wait.** Retry the allocation with
  a bounded budget before failing, so a transient pool peak does not kill a query. The budget must be
  strictly less than the sender's socket read timeout (`SIRIUS_CN_RPC_TIMEOUT_SECS`, default 60 s,
  `tunable.rs:37-42`, applied `prpc_client.rs:158`), otherwise the sender times out first and the
  wait accomplishes nothing. Proposed knob `SIRIUS_CN_EXCHANGE_COPY_WAIT_MS`, default `0` (off),
  max bounded to `rpc_timeout - 5 s` **validated at bring-up in `tunable.rs`** — a knob that can be
  set above the RPC timeout is a knob that silently converts a recoverable stall into a peer
  timeout. Do not add this in Phase 1; measure first.

**What must NOT happen:** blocking inside `exchange_staging_arena::lease`. Turning the arena into a
blocking allocator re-introduces the head-of-line coupling the `staging_*`-off-the-engine-thread
invariant exists to prevent (`engine.rs:112-121`).

---

## 5. Staged migration

Each phase is independently buildable, testable, and revertible. **The current path keeps working
until Phase 3.**

### Phase 0 — Instrumentation only (no behaviour change)

Goal: be able to measure the arena high-water reliably, before and after.

1. Expose the arena stats that already exist in C++ through the FFI. They exist
   (`src/include/exec/exchange_staging_arena.hpp:105-120`: `outstanding`, `peak_live_bytes`,
   `live_bytes`, `largest_free`, `total_free`; implementations
   `src/exec/exchange_staging_arena.cpp:298-338`) but **only `outstanding()` crosses the FFI**
   (`src/include/sirius_ffi.hpp:166-168`, `rust/crates/sirius-sys/src/lib.rs:103-106`,
   `rust/crates/sirius/src/lib.rs:427-429`). Add the other four to `ffi::StagingArena`, the cxx
   bridge, and the safe wrapper.
2. Add a CN sampler: a thread that logs `live/peak/free/largest/outstanding` every N seconds via
   `tracing::info!`, gated by a new tunable `SIRIUS_CN_STAGING_SAMPLE_SECS` (default `0` = off) in
   `experimental/starrocks/src/tunable.rs`, resolved by `Tunables::resolve()` at `main.rs:162`.
3. Log the same line once per receiver fragment completion in `dispatch_worker`
   (`compute_node_service.rs:304-314`).

**Why this is Phase 0 and not an afterthought.** The only arena high-water number in the system today
is the destructor line at `src/exec/exchange_staging_arena.cpp:159-176` (format at `:168`), and it
**only prints on a clean shutdown**. Measured on this box: in
`/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log` there are **96** constructor lines
(`exchange_staging_arena.cpp:79`, `"exchange staging arena: N bytes (cudaMalloc)"`) and only **48**
`"peak live"` teardown lines (`:168`) — **50 % of arena lifetimes produced no number at all.**
(The task brief quotes 61 %; that figure comes from a different log corpus and is **UNVERIFIED**
here — 50 % is what this box's surviving log shows.) The cause is structural: the sweep restarts
CNs with `pkill` (SIGTERM) via `/opt/dlami/nvme/sirius-build/restart-sf500x.sh`, and the CN
force-exits after `SHUTDOWN_GRACE = 15 s` or on a second signal
(`experimental/starrocks/src/main.rs:31-34`, escalation task `:667-682`) — precisely when the engine
thread is wedged, i.e. precisely the runs whose number you want.

Note `grep` on these logs is `ugrep`, which treats them as binary because of the box-drawing
characters in the plan dumps. **Use `grep -a`** or the counts come back empty and silently wrong.

**Deliverable:** a baseline arena high-water for the full SF500 sweep at the current 60/32 config.

### Phase 1 — The copy path, behind a flag, default OFF

1. C++: `ResidentBatch`, `ArrivalCopier`, `Context::arrival_copier_handle()`,
   `Fragment::push_resident` (§3.4). Keep `Fragment::push_packed` untouched.
2. Bridge + wrapper: `rust/crates/sirius-sys/src/lib.rs`, `rust/crates/sirius/src/lib.rs`.
3. CN: `ArrivedBatch`, the `claim`/`commit` rendezvous split, `staging_copy_out` on
   `FragmentExecutor`, the `handle_transmit_packed` branch, the `engine.rs:541-575` match.
4. **The flag.** `experimental/starrocks/src/tunable.rs` is this repo's env-knob validation seam —
   see commit `a27615d5` ("feat(cn): validate transport tunables at bring-up"), which added the
   registry and its three rules at `tunable.rs:1-20`: out-of-range/unparsable values are **rejected,
   never clamped or ignored**; the resolved set is **logged** at bring-up; **unset means the
   documented default**. Add:

   ```rust
   /// Copy each arriving remote exchange batch out of the staging arena into pool memory in the
   /// `transmit_packed` handler, releasing its lease immediately, instead of holding every lease
   /// until the receiver fragment runs (PLAN-PATH-B D-B5, PLAN-01).
   const EXCHANGE_COPY_ON_ARRIVAL: Knob<bool> = Knob {
       name: "SIRIUS_CN_EXCHANGE_COPY_ON_ARRIVAL",
       default: false,   // flipped to true in Phase 3
       min: false, max: true,   // or a dedicated Knob<bool> without a range
   };
   ```

   `tunable.rs` today has `Knob<u64>` (`:112-148`) and `Knob<f64>` (`:150-183`) readers only. Add a
   `Knob<bool>` `read()` that accepts exactly `true`/`false`/`1`/`0` (case-insensitive) and
   **rejects everything else**, consistent with rule 1. Add the field to `Tunables`
   (`:202-215`), `DEFAULTS` (`:219-227`), `from_env` (`:234-260`), and the bring-up
   `tracing::info!` (`:277-286`). Add a row to `experimental/starrocks/docs/TUNABLES.md` under
   "Exchange staging". Add a unit test in `tunable.rs`'s `mod tests` (`:295-296`) that a bad value
   fails resolution.
5. Fix the comments that currently assert the untrue property, in the same commit, so the tree stops
   lying either way: `engine.rs:62`, `engine.rs:543-545`, `fragment_executor.rs:75-77`,
   `rust/crates/sirius/src/lib.rs:131-134` and `:334-341`, `src/include/sirius_ffi.hpp:101-103`
   (which is doubly stale — it still describes a **bump head**, and the arena has been a coalescing
   free list since commit `a27615d5`; see `src/exec/exchange_staging_arena.cpp:260-296`).

**Exit criterion:** with the flag off, `pixi run make test`, `pixi run cn-test`,
`cargo test -p sirius` and a full SF500 sweep are bit-identical to Phase 0.

### Phase 2 — Rendezvous eviction (REQUIRED before flipping the default)

`LocalExchange` has **no eviction path**: `ExchangeState` holds `receivers`, `sources`, `remote_seq`
(`local_exchange.rs:88-95`) and the only removal is inside `take_ready`
(`local_exchange.rs:282-308`), which runs only on success. `cancel_plan_fragment` is a stub that
logs, fails the result entry and returns OK — its own comment says *"Real teardown (aborting the
engine run, freeing GPU buffers, dropping parked exchange state) is a separate work item"*
(`compute_node_service.rs:374-380`).

Today an abandoned query leaks **arena leases**. After Phase 1 it leaks **pool memory** — the scarce
resource this plan is trying to free. That is a regression unless it is closed.

1. Add `LocalExchange::drop_fragment(fragment_instance_id) -> usize` that removes the receiver, its
   `sources`, and its `remote_seq` entries, returning how many batches it dropped. Dropping a
   `Vec<ArrivedBatch>` releases arena leases (`Staged`) and pool memory (`Resident`) automatically —
   give `ArrivedBatch::Staged` a `Drop` that releases, or release explicitly in `drop_fragment`
   using the executor handle. Prefer explicit release with a logged failure: `Drop` cannot report an
   error, and a double release is a loud error by design
   (`src/exec/exchange_staging_arena.cpp:260-268`).
2. Call it from `cancel_plan_fragment` (`compute_node_service.rs:381-403`) and from the
   receiver-failure path in `run_ready_fragment` (`compute_node_service.rs:677-700`).
3. Log the count. A nonzero count on cancel is the evidence that this path is live.

### Phase 3 — Flip the default and re-tune

1. `EXCHANGE_COPY_ON_ARRIVAL.default = true`.
2. Re-run the sweep at the current 60/32 to prove no regression, then walk `STAGING` down and
   `GPU_MEM` up (see §7.3). Target `STAGING=8GiB GPU_MEM=84GiB` (8 + 84 + ~2 GiB CUDA context
   ≈ 94 GiB ≤ 94.97 GiB allocatable).
3. Run the q09 ladder.

### Phase 4 — Remove the legacy arm (separate PR)

Delete `ArrivedBatch::Staged`, the flag, `Fragment::push_packed`, and the arena-release logic in
`engine.rs:541-575` / `:385-421`. Only after Phase 3 has held across at least one full cold sweep.

---

## 6. Failure semantics

| Event | Required behaviour | Where |
|---|---|---|
| **`copy_out` throws (pool OOM, CUDA error, bad range)** | `copy_out` must NOT have released the lease. `handle_transmit_packed` releases it (`executor.staging_release(offset)`), then returns `Err`. The frame's `seq` was already consumed by `claim_remote_frame`, so a retry of the same frame would be dropped as a duplicate — the query **must** fail, and it does. | `compute_node_service.rs:605-666`; sender sees non-OK status → `check_status` (`nixl_transport.rs:903-915`) → drain error → `dispatch_then_join` (`:275-300`) → FE (`:337`) |
| **Copy partially done then throws** | `cudf::table`'s own construction is all-or-nothing; a partially constructed table unwinds its columns. No manual cleanup needed beyond (1). | `src/sirius_ffi.cpp:849` pattern |
| **`push_resident` throws (schema mismatch, stream already ended)** | Same class as today's `push_packed` failure. The `ResidentBatch` is dropped by `run_fragment`'s error path, freeing pool memory. `run_fragment`'s existing lease sweep (`engine.rs:393-419`) must skip `Resident` entries — they hold no lease. | `engine.rs:385-421` |
| **Dead peer mid-drain** | Sender-side: unchanged (`nixl_transport.rs:405-419` drops the parked output, ticket carries the error). Receiver-side: the partially filled `SenderSource::Remote` is orphaned in the rendezvous. **Phase 2's `drop_fragment` is what reclaims it.** Without Phase 2 this is a pool leak, not just an arena leak. | `local_exchange.rs:88-95` (no eviction today) |
| **Cancelled query (`cancel_plan_fragment`)** | Same as dead peer: `drop_fragment` must run. Today the handler only cancels the result entry (`compute_node_service.rs:398`). The FE sends this for **every** fragment instance of every query — 629 times in one sweep (`SF500-CONFIG-AND-ARCHITECTURE.md:128-130`) — so the path is well exercised and must be cheap and idempotent (removing an already-removed fragment id is a no-op, not an error). | `compute_node_service.rs:374-403` |
| **brpc reconnect replay (duplicate `seq`)** | `claim_remote_frame` returns `Duplicate` **before** any copy or release. Nothing is copied, nothing is released, `Ok(())` is returned — matching today's behaviour at `local_exchange.rs:182-192`. Critically: after Phase 1 the original lease is already gone, so a copy on the replay path would read reused memory. This ordering is the correctness argument for the two-phase split. | `local_exchange.rs:182-192` |
| **Sequence gap (lost frame)** | Unchanged loud error; no copy attempted. | `local_exchange.rs:193-198` |
| **eos-only frame (no attachment)** | No lease, no copy. `claim` → `commit(None)`. Matches today's `attachment.is_empty()` branch. | `compute_node_service.rs:637-651`, `nixl_transport.rs:762-776` |
| **Zero-row batch (`len == 0`)** | Metadata-only, holds no lease (`src/sirius_ffi.cpp:749-753`), must not be released. It still carries schema, so it must still become a `ResidentBatch` (an empty table) rather than being dropped — otherwise a stream that receives only empty batches loses its column types. Verify against the existing empty-batch test at `rust/crates/sirius/src/lib.rs:1185-1260`. | `sirius_ffi.cpp:749-753`, `engine.rs:562` |
| **Engine thread wedged** | Unchanged and *improved*: arrivals no longer accumulate leases while it is wedged. Leases are now bounded by peer count regardless of engine state. This is the whole point. | `engine.rs:112-121` |
| **Process killed (SIGKILL / force-exit)** | Device memory and the arena are reclaimed by the driver at process death (`main.rs:667-682` states this). No new hazard. | — |

---

## 7. Tests

### 7.1 C++ unit tests

Add to `test/cpp/exec/test_exchange_staging_arena.cpp` (existing tag `[staging_arena]`; the file is
registered at `CMakeLists.txt:642`). These need no GPU beyond a CUDA context, matching the existing
ARENA-1…ARENA-13 cases:

* **ARENA-14 — concurrent lease/copy-out/release keeps `live_bytes()` bounded.** N threads each
  lease → release in a loop; assert `peak_live_bytes() <= N * lease_size` and
  `total_free() + live_bytes() == capacity` at every step (extends ARENA-8/ARENA-12).

Add a **new** file `test/cpp/exec/test_arrival_copier.cpp` (register it in the `TEST_SOURCES` list at
`CMakeLists.txt:642`), tag `[arrival_copier]`, requires a GPU:

* **COPY-1 — round trip.** Pack a known table into a lease (`cudf::chunked_pack`, mirroring
  `src/sirius_ffi.cpp:740-777`), `copy_out`, assert the returned table equals the source
  (`cudf::test::expect_tables_equal`) and that `arena.outstanding() == 0` afterwards.
* **COPY-2 — the lease is genuinely free.** After `copy_out`, `arena.lease(capacity)` must succeed —
  the same proof the existing Rust test uses at `rust/crates/sirius/src/lib.rs:1239-1244`.
* **COPY-3 — overwriting the lease does not corrupt the copy.** After `copy_out`, `cudaMemset` the
  whole arena, then compare the table again. This is the assertion that actually proves "copy", not
  "view".
* **COPY-4 — a bad range throws and releases nothing.** Offsets past `capacity()`, `length`
  overflowing, zero metadata (mirrors the guards at `src/sirius_ffi.cpp:793-803`).
* **COPY-5 — concurrent copy-out from several threads.** 4 threads × 32 batches; assert all tables
  correct and `outstanding() == 0`. This is the test that would catch a device-not-set or
  shared-stream bug.
* **COPY-6 — pool ceiling.** Build a context with a small `usage_limit_bytes`, copy until it throws;
  assert the exception names `LIMIT_EXCEEDED`-class information and that the arena lease is still
  outstanding (so the caller can release it per §6).

Run: `pixi run make test`, or by tag
`pixi run build/release/extension/sirius/test/cpp/sirius_unittest "[arrival_copier]"`.

### 7.2 Rust tests

**`rust/crates/sirius/src/lib.rs`** (GPU tests; take `GPU_CONTEXT_LOCK` at `:481-483`, and set
`SIRIUS_EXCHANGE_STAGING_BYTES` before constructing the context as at `:1085-1087`):

* `resident_hop_matches_relay_hop` — the mirror of the existing
  `packed_hop_matches_relay_hop` (`:1079-1183`), using `copy_out` + `push_resident`. Same decisive
  equivalence: a fragment hop carried by copy-out must deliver exactly the values `relay_from`
  delivers.
* `copy_out_releases_the_lease` — after `copy_out`, `ctx.staging_lease(capacity)` succeeds
  (pattern from `:1239-1244`).
* `push_resident_rejects_a_mismatched_schema` — the mirror of `push_packed_rejects_a_mismatched_schema`
  (`:2734-2775`). This is the test that proves the schema guard at `src/sirius_ffi.cpp:812-836`
  survived the move to `push_resident`.
* `empty_resident_batch_carries_its_schema` — mirror of the empty-result test at `:1185-1260`.

**`experimental/starrocks/src/local_exchange.rs`** `mod tests` (`:325-326`, no GPU — these run under
`pixi run cn-test-no-engine`):

* `claim_then_commit_records_in_seq_order` — commit out of order, assert `batches` come back in
  `seq` order.
* `duplicate_seq_is_claimed_as_duplicate_before_any_copy` — extends
  `duplicate_remote_seq_is_dropped_idempotently` (`:438`); assert `claim` returns `Duplicate` and
  that no commit is required.
* `seq_gap_is_rejected_at_claim` — extends `remote_seq_gap_is_a_loud_error` (`:474`).
* `drop_fragment_removes_receiver_sources_and_seq` (Phase 2) — assert the count returned and that a
  later frame for that fragment is a clean error, not a panic.

**`experimental/starrocks/src/tunable.rs`** `mod tests` (`:295-296`):

* `exchange_copy_on_arrival_rejects_a_bad_value` — `SIRIUS_CN_EXCHANGE_COPY_ON_ARRIVAL=yes` must
  fail `Tunables::from_env()` with a message naming the variable and the accepted values.

**`experimental/starrocks/src/compute_node_service.rs`** `mod tests` — with the flag on and a stub
executor whose `staging_copy_out` returns a fabricated `ResidentBatch`, assert that
`handle_transmit_packed` calls it exactly once per batch frame and zero times for eos/duplicate
frames. (If a stub `ResidentBatch` is impossible without the engine, gate this test on the
engine feature and run it under `pixi run cn-test`.)

Run: `pixi run cn-test-no-engine` (CI path) and `pixi run cn-test` (engine-linked).

### 7.3 The SF500 proof

**The harness has NO correctness gate** — `bench.sh` says so itself at
`experimental/starrocks/benchmarks/tpch/bench.sh:55-57` (*"this script times and counts rows only
— it does not check answers"*), and the pass condition is only `rc == 0 && -s "$f" && ! head -1 |
grep ERROR` (`bench.sh:175`). **Correctness must be checked separately with `compare.py`.**

Baseline (Phase 0, flag off) and each subsequent phase:

```bash
# Full 22, fresh cluster per query (~68 s restart each, ~25 min overhead).
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB \
HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
OUT=/opt/dlami/nvme/sirius-build/bench/SF500-PLAN01-BASE/timings.csv \
  /opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh

# Correctness — MANDATORY, the sweep does not do this.
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/SF500-PLAN01-BASE \
  /opt/dlami/nvme/sirius-build/oracle-sf500f64
```

`sweep-sf500x-cold.sh` sets `TPCH_DATA=/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64`,
`MIN_BACKENDS=2`, `RESTART_CMD=restart-sf500x.sh`, `FE_QUERY_TIMEOUT=1800`, and `OUT` defaults to
`bench/SF500XCOLD/timings.csv`; per-query output lands next to the CSV as `<q>.r<N>.out`
(`bench.sh:169`), which is exactly what `compare.py` reads. `restart-sf500x.sh` propagates
`GPU_MEM/STAGING/HOST_MEM/HPB/MBHT/STB/CBB/MSPB` into `up-sf500-x.sh` — **it must, or the restart
silently reverts to 60GiB/32GiB and invalidates the experiment** (the script says so).

`up-sf500-x.sh` already exports `SIRIUS_LOG_BACKEND=spdlog`,
`SIRIUS_LOG_DIR=/opt/dlami/nvme/sirius-build/siriuslog`, `SIRIUS_LOG_LEVEL=info` and
`SIRIUS_EXCHANGE_STAGING_BYTES=$STAGING`. Add `SIRIUS_CN_EXCHANGE_COPY_ON_ARRIVAL` and
`SIRIUS_CN_STAGING_SAMPLE_SECS` to its exported environment.

The q09 ladder (Phase 3), each entry a full cold sweep of `q09` alone plus a full-22 regression run:

| Pool / Arena | Expect |
|---|---|
| 60/32 | must still be 21/22 — no regression |
| 70/16 | 21/22, arena high-water ≪ 16 GiB |
| 78/10 | q09 attempt |
| 84/8  | **q09 target** |

Capture the arena high-water from the Phase-0 sampler after every run:

```bash
grep -a "staging arena sample\|peak live" /opt/dlami/nvme/sirius-build/siriuslog/sirius_*.log | tail -40
```

(`grep -a` is required — see the note in §5, Phase 0.)

---

## 8. Success criteria (numeric)

1. **Arena high-water per CN over the full SF500 22-query cold sweep ≤ 4 GiB**, measured by the
   Phase-0 sampler (`live_bytes()` peak) and cross-checked against `peak_live_bytes()` on any clean
   shutdown. Baseline to beat: **26.78 GiB** healthy / **47.40 GiB** wedged
   (`SF500-CONFIG-AND-ARCHITECTURE.md:83-85, 95-97`). The predicted bound is
   `peers_draining_concurrently × p_max`, where `p_max` is the largest single packed batch; with
   2 CNs (1 peer) and `CBB=HPB=STB=1GiB` this should land at **1–2.5 GiB**. Receive-side leases are
   exactly `batch.len` (`nixl_transport.rs:712`), not `len + 8 MiB` — the 8 MiB `kPackChunkBytes`
   slack (`src/sirius_ffi.cpp:704`, `:757`) is a **send-side** lease only.
2. **Arena high-water is independent of scale factor and CN count.** Re-run at SF100 and SF300 and
   confirm the high-water stays within ±20 % of the SF500 number. Today it tracks SF and CN count,
   which is what the retired `STAGING ≈ 96GiB × SF/500 / N` heuristic encoded
   (`SF500-CONFIG-AND-ARCHITECTURE.md:85-86`).
3. **q09 passes at SF500** with `GPU_MEM=84GiB STAGING=8GiB` (or any split that fits 94.97 GiB) and
   `compare.py` reports `q09  MATCH` against `oracle-sf500f64/q09.tsv`.
4. **No regression in the 21.** `compare.py` reports **≥ 21 MATCH** on the full sweep at the new
   config. Note q11's oracle is header-only (17 bytes) because `queries/q11.sql` hardcodes the SF1
   threshold — DuckDB also returns 0 rows, so `MATCH` with 0 rows is the correct verdict, while
   `bench.sh:175`'s `[ -s "$f" ]` files it as a wedge (harness defect, out of scope here;
   `SF500-CONFIG-AND-ARCHITECTURE.md:40-43`).
5. **Zero leaked leases.** `StagingArena::outstanding()` (already exposed,
   `rust/crates/sirius/src/lib.rs:427-429`) is `0` after every query completes, logged by the
   Phase-0 per-fragment line.
6. **No new pool retention.** `[gpu_pool] ... QueryBegin allocated=` (`src/sirius_context.cpp:254`)
   must not ratchet upward across the 22 queries by more than the parked-output floor already
   measured (11.309 GiB per q07 run per CN, `SF500-CONFIG-AND-ARCHITECTURE.md:132-133`). A larger
   ratchet means the copies are not being freed — see risk R1.
7. **`arena exhausted` no longer appears** in any sweep log at `STAGING ≥ 8GiB`
   (`grep -a "arena exhausted"` returns nothing).
8. **Warm-run latency does not regress by more than 10 %** versus
   `bench/rtxpro6000-2gpu/results/sf500xcold.csv` on the queries that pass today.

---

## 9. Risks, and how to detect them early

| # | Risk | Why it is plausible | Early detection |
|---|---|---|---|
| **R1** | **Deallocation accounting asymmetry**: allocation on a reservation-less thread, free on a thread holding a reservation, silently loses the bytes from `_total_allocated_bytes` (`upstream_reclaimed_bytes = 0` branch), eroding the usable pool. | `deallocate` subtracts from whatever reservation the deallocating thread holds, without recording which allocations were reservation-backed — `cucascade/src/memory/reservation_aware_resource_adaptor.cpp:484-513`; tracking is `PER_THREAD` (`src/sirius_config.cpp:67`, tracker TLS at `reservation_aware_resource_adaptor.cpp:113`). **Pre-existing** — `push_packed` already allocates on the reservation-less engine thread today — so this plan does not create it, but it makes the volume larger. | Success criterion 6. Watch `[gpu_pool] QueryBegin allocated=` (`src/sirius_context.cpp:254`) across a 22-query sweep with `--cold-restart` off; a monotone climb that idle time does not release is the signature. |
| **R2** | **Legacy null-stream barrier.** If the copy uses `cudf::get_default_stream()` on an RPC worker, `stream.synchronize()` becomes a process-wide barrier. | PTDS is **off** in this build (no `CUDA_API_PER_THREAD_DEFAULT_STREAM` in `CMakeLists.txt`/`cmake/`/`pixi.toml`; the shipped `libcudf.so`'s `cudf::is_ptds_enabled()` returns false). | Code review gate: `grep -n "get_default_stream" src/sirius_ffi.cpp` must show no occurrence inside `ArrivalCopier`. Symptom: total sweep wall clock inflates while per-query GPU time does not. |
| **R3** | **Device not set on the arrival thread.** | `Fragment::push_packed` has no `cudaSetDevice` (`src/sirius_ffi.cpp:784-858`), unlike `src/pipeline/gpu_pipeline_executor.cpp:79-86`. Masked on this box by one visible GPU per CN. | COPY-5 (multi-threaded copy test). On a multi-GPU-per-process build it manifests as wrong-device pointers or `cudaErrorIllegalAddress`. |
| **R4** | **Pool leak on abandoned rendezvous entries** — a cancelled/failed query's `ArrivedBatch::Resident` batches are never dropped, so pool memory (not just arena) leaks. | `LocalExchange` has no eviction (`local_exchange.rs:88-95`, removal only in `take_ready` `:282-308`); `cancel_plan_fragment` is a stub (`compute_node_service.rs:374-380`). | This is exactly why Phase 2 gates Phase 3. Detect via criterion 6 plus the `drop_fragment` count log; run a deliberately cancelled query and assert the pool returns to baseline. |
| **R5** | **The schema guard is lost in the move.** Today `push_packed` validates the unpacked columns against the declared stream (`src/sirius_ffi.cpp:812-836`); at arrival there is no declared stream, so a careless refactor drops the check. | The check depends on `impl_->resolved_inputs` (`src/sirius_ffi.cpp:411`, filled at `:588`), which only exists after `build()`. | `push_resident_rejects_a_mismatched_schema` (§7.2) — port `rust/crates/sirius/src/lib.rs:2734-2775`. |
| **R6** | **Duplicate-frame double copy / double release.** A brpc reconnect replay after the lease was released would copy from reused memory or throw on a double release. | Replays are real and handled today (`local_exchange.rs:182-192`); the release used to be a no-op there. | `duplicate_seq_is_claimed_as_duplicate_before_any_copy` (§7.2). Symptom in the wild: `"exchange staging arena: release of offset N which is not an outstanding lease (double release?)"` (`src/exec/exchange_staging_arena.cpp:259-268`). |
| **R7** | **Batch ordering.** `commit_remote_frame` inserts by completion order instead of `seq`. | The doc contract says arrival (sequence) order (`local_exchange.rs:41-44`). Today it holds trivially because frames from one peer are serialized (`brpc.rs:150-192`, `prpc_client.rs:36-47`); the invariant should not silently depend on that. | `claim_then_commit_records_in_seq_order` (§7.2). |
| **R8** | **Latency regression:** the copy now sits on the RPC critical path, and the sender's socket read timeout is 60 s (`tunable.rs:37-42`, `prpc_client.rs:158`). A very large batch on a busy GPU could exceed it. | `p_max` today can reach the derived default batch size, 2.39 GiB on this card (`SF500-CONFIG-AND-ARCHITECTURE.md:47-55`). | Criterion 8. Symptom: sender-side `transmit_packed` timeouts appearing where lease exhaustion used to. Mitigation: the operator budgets already in use (`CBB=HPB=1GiB`) bound `p_max`; if needed, raise `SIRIUS_CN_RPC_TIMEOUT_SECS`. |
| **R9** | **`SenderSource: Clone` breaks the build.** `ResidentBatch` is a `UniquePtr`. | `local_exchange.rs:25` derives `Clone`. | Compile error, immediately. Verified the derive is unused outside the file (`.clone()` at `:205` is on `names`; `compute_node_service.rs:1004,1170,1171` construct/match only). |
| **R10** | **`--no-default-features` CI build breaks.** `ArrivedBatch::Resident` names a `sirius::` type that does not exist without the engine. | `pixi run cn-test-no-engine` is what CI runs (`pixi.toml:148`). | Run `pixi run cn-test-no-engine` in Phase 1 before anything else. |
| **R11** | **The measurement itself is unreliable** and a "success" is unobservable. | The only high-water number today prints on clean shutdown, and 48 of 96 arena lifetimes in the surviving log produced nothing (§5, Phase 0). | Phase 0 exists solely to close this. Do not skip it, and do not accept a result whose arena number came only from the destructor line. |

---

## 10. Files this plan touches

| File | Phase | Change |
|---|---|---|
| `src/include/exec/exchange_staging_arena.hpp` | 0 | — (accessors already exist, `:105-120`) |
| `src/include/sirius_ffi.hpp` | 0,1 | expose 4 arena stats on `StagingArena` (`:142-172`); add `ResidentBatch`, `ArrivalCopier`, `Context::arrival_copier_handle`, `Fragment::push_resident`; fix stale doc `:101-103` |
| `src/sirius_ffi.cpp` | 0,1 | implement the above; `push_packed` (`:784-858`) unchanged until Phase 4; fix comment `:845-847` |
| `rust/crates/sirius-sys/src/lib.rs` | 0,1 | bridge the new stats and types (`:59-107`, `:168-190`) |
| `rust/crates/sirius/src/lib.rs` | 0,1 | safe wrappers + `unsafe impl Send/Sync`; fix stale docs `:131-134`, `:334-341`; new tests |
| `experimental/starrocks/src/tunable.rs` | 0,1 | `Knob<bool>`; `SIRIUS_CN_EXCHANGE_COPY_ON_ARRIVAL`; `SIRIUS_CN_STAGING_SAMPLE_SECS` |
| `experimental/starrocks/src/fragment_executor.rs` | 1 | `ArrivedBatch`; `staging_copy_out` on the trait (`:100-146`); fix stale doc `:74-77` |
| `experimental/starrocks/src/local_exchange.rs` | 1,2 | `claim_remote_frame`/`commit_remote_frame` split of `:159-247`; `batches: Vec<ArrivedBatch>` (`:44`); drop `Clone` (`:25`); `drop_fragment` |
| `experimental/starrocks/src/compute_node_service.rs` | 0,1,2 | `handle_transmit_packed` (`:605-666`); sampler log in `dispatch_worker` (`:304-314`); `drop_fragment` from `cancel_plan_fragment` (`:381-403`) and the failure path (`:677-700`) |
| `experimental/starrocks/src/engine.rs` | 1 | carry the copier handle out of bring-up (`:139-152`, `:234`); `ArrivedBatch` match in the push loop (`:541-575`) and the error sweep (`:385-421`); fix stale comments `:62`, `:543-545` |
| `experimental/starrocks/src/main.rs` | 0 | start the sampler thread near `Tunables::resolve()` (`:162`) |
| `experimental/starrocks/docs/TUNABLES.md` | 1 | document both new knobs |
| `test/cpp/exec/test_arrival_copier.cpp` (new) | 1 | COPY-1…COPY-6 |
| `CMakeLists.txt` | 1 | register the new test file in `TEST_SOURCES` (`:642`) |
| `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` | 3 | export the two new knobs |

---

## 11. Explicitly out of scope

* **Query-scoped park ownership and a real `cancel_plan_fragment`** — the separately measured
  11.309 GiB/q07/CN leak (`SF500-CONFIG-AND-ARCHITECTURE.md:118-140`). Phase 2 here adds only the
  *rendezvous* eviction that this change makes mandatory; parked *sender* outputs are a different
  leak with a different fix.
* **`derived_default_batch_size()` keying off the physical card instead of the configured pool**
  (`SF500-CONFIG-AND-ARCHITECTURE.md:142-147`) and a bring-up validation that refuses
  `pool + arena + overhead > allocatable`.
* **The q21 scheduler stall** (`SF500-CONFIG-AND-ARCHITECTURE.md:44-46, 168-173`).
* **Harness fixes**: `bench.sh:175` filing a correct empty result as a wedge, and the missing
  correctness gate. Until they land, always run `compare.py`.
* **Pipelining the sender drain** (more than one batch in flight per destination,
  `nixl_transport.rs:697-780`). Doing that *after* this change would raise arena demand again in a
  controlled, bounded way; doing it before would make things worse.
