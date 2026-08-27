# PLAN-09 — No backpressure on the exchange staging-arena lease path

**Status:** analysis only. No code has been written. This document is self-contained: it assumes
zero prior context, cites `file:line` for every claim, and gives two reproductions that were
actually observed.

**Companion:** [`PLAN-01-copy-out-on-arrival.md`](PLAN-01-copy-out-on-arrival.md) covers the
retention half of this problem. PLAN-01 *shrinks* the queue; PLAN-09 is about *bounding* it.
They are complementary, not alternatives — see [§8.4](#84-option-4--copy-out-on-arrival-plan-01).

---

## 1. One-paragraph statement of the defect

Sirius-as-StarRocks-CN moves exchange data between compute nodes through a fixed-size device
**staging arena** (plain `cudaMalloc`, outside every RMM pool). A sender leases space in the
*receiver's* arena, RDMA-WRITEs a packed batch into it, and the receiver releases that lease only
after deep-copying the batch **into pool memory**. There is **no backpressure anywhere on that
path**: the lease RPC either grants or throws, and nothing slows a sender down. Because the drain
(release) step requires a *pool* allocation, a pool at its ceiling stops the drain while senders
keep leasing. Arena occupancy then ratchets monotonically to capacity, and — because there is no
cleanup path for leases held by an exchange whose sender set never completes — **it never
recovers**. The query dies with `exchange staging arena exhausted`, and the arena's reported
demand becomes a function of *pool health* rather than of the query, which is why no
`(scale factor, node count)` sizing formula works.

---

## 2. Orientation — the box, the repo, the moving parts

| Thing | Value |
|---|---|
| Repo | `/home/ubuntu/sirius`, branch `demo-multi-cn` (default branch is `dev`) |
| Build | `pixi run make` — tests `pixi run make test` |
| Dead code | everything under `src/legacy/` is the retired `gpu_processing` path — ignore it |
| Box | 2× RTX PRO 6000 Blackwell. `nvidia-smi` reports 97887 MiB/card, but 638 MiB is driver-reserved, so **94.97 GiB is allocatable**, not 95.59 |
| Cluster | StarRocks FE + **2 compute nodes**, one CN per GPU. CN source: `experimental/starrocks/` |
| Datasets | SF500 f64 `/opt/dlami/nvme/tpch/tpch_parquet_sf500_f64`; SF300 f64 `/opt/dlami/nvme/tpch/tpch_parquet_sf300_f64`; SF100 f64 `/opt/dlami/nvme/tpch/tpch_parquet_sf100_f64`. (`/home/ubuntu/tpch_parquet_sf100` also exists but is the **DECIMAL** dataset, not the float64 twin.) |
| Bring-up | `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` — every knob is an env var |
| Healthy baseline | `GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB` → SF500 **21/22** (only q09 fails) |
| Context doc | [`../SF500-CONFIG-AND-ARCHITECTURE.md`](../SF500-CONFIG-AND-ARCHITECTURE.md) |

### 2.1 The staging arena

`src/include/exec/exchange_staging_arena.hpp:41-146` and `src/exec/exchange_staging_arena.cpp`.

* One arena per CN process, created once at FFI bring-up from `SIRIUS_EXCHANGE_STAGING_BYTES`
  (`src/sirius_ffi.cpp:228`, `exchange_staging_arena.hpp:49`).
* **Plain `cudaMalloc`, deliberately outside every pool.** The header explains why
  (`exchange_staging_arena.hpp:30-33`): UCX's `cuda_ipc` path cannot export `cudaMallocAsync`
  allocations and silently degrades ~220× to staged host copies. This is the reason the arena and
  the RMM pool compete for the *same* device budget — raising one takes memory from the other.
* Allocator: a mutex-protected, **coalescing, address-ordered first-fit free list**
  (`exchange_staging_arena.cpp:230-240` for the fit scan, `:259-291` for release + two-sided
  coalesce). `kAlignment = 256` (`exchange_staging_arena.hpp:44`).
* Instrumentation already present: `peak_live_bytes()`, `live_bytes()`, `total_free()`,
  `largest_free()`, `outstanding()` (`exchange_staging_arena.hpp:105-120`, definitions at
  `exchange_staging_arena.cpp:294-336`).

### 2.2 Thread map of one CN

Four independent execution contexts. Knowing which is which is required for the deadlock analysis
in [§8](#8-the-backpressure-design-space).

| Thread | Spawned at | Serves |
|---|---|---|
| `sirius-engine` (one, `!Send` context) | `experimental/starrocks/src/engine.rs:139` | `EngineRequest::Run` (whole fragment), `ExportNext`, `DropParked` — strictly serialised through one channel (`engine.rs:80-99`, dispatch at `engine.rs:304-312`) |
| `nixl-transport` (one) | `experimental/starrocks/src/nixl_transport.rs:351` | `TransportRequest::SendFragment` **inline, one drain at a time** (`nixl_transport.rs:395-419`; the doc comment at `:215-217` states this explicitly) |
| `fragment-dispatch` (one) | `experimental/starrocks/src/compute_node_service.rs:243-248` | runs ready receiver fragments **sequentially** (`compute_node_service.rs:300-314`) |
| BRPC tokio + `spawn_blocking` pool | — | `request_staging_lease` (`compute_node_service.rs:512-540`), `transmit_packed` (`:546-565`), `exec_plan_fragment` (`:321-334`) |

Critically, **`staging_lease` / `staging_release` deliberately bypass the engine request channel**
and run on the caller's own thread against a `Send + Sync` arena handle
(`engine.rs:682-700`, with the rationale in the module doc at `engine.rs:16-22`: funnelling leases
through the engine thread once turned an engine stall into a peer's exchange stall and failed the
whole query). That design decision is correct and must be preserved by any fix.

### 2.3 The lease lifecycle, end to end

```
SENDER CN                                          RECEIVER CN
---------                                          -----------
export_packed(stream)                    sirius_ffi.cpp:707
  arena.lease(total + 8 MiB)             sirius_ffi.cpp:757      [local lease]
  chunked_pack -> lease                  sirius_ffi.cpp:761-765
send_fragment loop                       nixl_transport.rs:706
  rpc_request_lease(peer, batch.len)     nixl_transport.rs:712 --> handle_staging_lease
                                                                   compute_node_service.rs:1106-1110
                                                                     executor.staging_lease()
                                                                       engine.rs:690-694
                                                                         arena.lease()
                                                                           arena.cpp:212 [remote lease]
  write_and_wait (nixl RDMA WRITE)       nixl_transport.rs:713-719
  rpc_transmit(metadata, offset, len)    nixl_transport.rs:724-739 --> handle_transmit_packed
                                                                   compute_node_service.rs:605
                                                                     push_remote_frame
                                                                       local_exchange.rs:159, :240
                                                                         batches.push(batch)   <-- JUST A Vec
  arena.release(local offset)            nixl_transport.rs:742-743  [local lease returned]

                                         ... time passes; nothing releases the REMOTE lease ...

                                         take_ready fires only when EVERY sender of EVERY
                                         exchange input has closed      local_exchange.rs:248-313
                                         run_fragment_inner push loop   engine.rs:546-570
                                           push_packed  -> DEEP COPY INTO POOL  sirius_ffi.cpp:849
                                           staging_release(offset)              engine.rs:563
```

---

## 3. The causal chain (each link verified)

**Link 1 — the drain step allocates from the pool.**
`Fragment::push_packed` deep-copies each staged batch into ordinary pool memory:

```cpp
// src/sirius_ffi.cpp:845-850
// Copy-out-on-arrival (PLAN-PATH-B D-B5): the batch the engine keeps lives in ordinary pool
// memory, so the lease is reusable the moment this call returns and the batch is fully
// accounted and spillable like any other.
auto stream = cudf::get_default_stream();
auto table  = std::make_unique<cudf::table>(unpacked, stream, gpu_space->get_default_allocator());
stream.synchronize();
```

Note `get_default_allocator()` (`cucascade/include/cucascade/memory/memory_space.hpp:119`) — a raw
pool reference. Unlike the pipeline executor, this path takes **no reservation** and has **no
downgrade-then-retry loop** (contrast `src/pipeline/gpu_pipeline_executor.cpp:225-270`). If the
pool is at its cap, this throws `std::bad_alloc: out_of_memory`.

**Link 2 — the lease is released only after that copy succeeds.**

```rust
// experimental/starrocks/src/engine.rs:556-570  (inside the push loop at :546-571)
fragment.push_packed(stream_id, &staged).map_err(|err| { ... })?;   // :556  <- can throw
if batch.len > 0 {
    context.staging_release(batch.offset).map_err(|err| { ... })?;  // :563  <- only reached on success
    released.insert(batch.offset);
}
```

**Link 3 — therefore arena drain requires a successful pool allocation.** Pool at ceiling ⇒ the
copy stalls or throws ⇒ the lease is not released on that iteration.

Directly observed, from `/tmp/cluster-sf500.log` at `GPU_MEM=45GiB STAGING=48GiB`:

> `failed to push a staged remote batch from sender 1 into stream 4: std::bad_alloc:
> out_of_memory: std::bad_alloc: out_of_memory: CUDA error (failed to allocate 157255572 bytes)`

That is exactly `sirius_ffi.cpp:849` failing.

**Link 4 — senders cannot be slowed.** There is no credit, no window, no retry and no blocking
wait anywhere on the lease path. Verify by reading the whole path:

* `handle_staging_lease` — grants or propagates the error, nothing else
  (`compute_node_service.rs:1105-1110`).
* The RPC wrapper turns an `Err` straight into `internal_error` (`compute_node_service.rs:527-531`).
* `SiriusEngine::staging_lease` just forwards (`engine.rs:690-694`).
* `exchange_staging_arena::lease` scans the free list once and **throws** on first-fit failure
  (`exchange_staging_arena.cpp:230-256`; the throw is at `:245-256`).
* The sender side has no retry either: `rpc_request_lease` makes one call and `check_status`
  converts a bad status into an error (`nixl_transport.rs:883-899`).

**Link 5 — occupancy ratchets to capacity and never recovers.** Two *distinct* retention
mechanisms are at work, and the second is the one that makes it permanent:

* **R1 — stalled drain.** Leases already handed to `run_fragment_inner` are held for the duration
  of a pool-starved copy, and on failure the whole fragment errors out. (`run_fragment` at
  `engine.rs:385-423` *does* release the un-pushed leases on the error path, `:403-420`, with the
  release itself at `:411`, so R1 alone is recoverable.)
* **R2 — orphaned rendezvous state, unrecoverable.** Batches that have arrived but whose receiver
  has not yet become ready sit in `SenderSource::Remote { batches: Vec<StagedBatch>, .. }`
  (`local_exchange.rs:44`). `state.sources` is **only ever removed inside `take_ready`**
  (`local_exchange.rs:295`) — grep the file: there is no other mutation that drops a source, no
  `impl Drop`, and no timeout GC. `StagedBatch` (`fragment_executor.rs:51-65`) is a plain struct
  holding an integer offset; **dropping it does not release the lease**. And
  `cancel_plan_fragment` (`compute_node_service.rs:381`) is an acknowledged stub — verbatim from
  its doc comment at `compute_node_service.rs:378-379`:

  > *"Real teardown (aborting the engine run, freeing GPU buffers, dropping parked exchange state)
  > is a separate work item."*

  So when a query dies mid-exchange (e.g. its senders hit pool OOM), every lease already parked in
  that rendezvous is pinned **for the process lifetime**.

R2 is what turns a transient pool squeeze into a permanent arena wedge.

---

## 4. The asymmetry: the send side is bounded, the receive side is the entire cost

This matters because it tells you where a fix has to go.

### 4.1 Send side — exactly ONE lease live at a time per CN

* The drain loop is strictly one batch at a time: lease locally → request a **peer** lease →
  **synchronous** `write_and_wait` → `rpc_transmit` → release the local lease
  (`nixl_transport.rs:706-757`; local release at `:742-743`, and its comment at `:740-741`
  explains that the release happens on both success and failure paths).
* Exactly **one** transport thread services `SendFragment` drains **inline**, so drains run one at
  a time in posting order (`nixl_transport.rs:215-217`, loop body at `:405-419`).
* `ExportNext` — the call that produces the next packed batch — is serialised through the single
  engine request channel (`engine.rs:88-91`, `:702-704`, handler at `:304-307`).

Lease size is `align256(packed_bytes + 8 MiB)`: `arena.lease(total + kPackChunkBytes)` at
`src/sirius_ffi.cpp:757`, with `kPackChunkBytes = 8u << 20` at `:704` (the slack exists because
every `chunked_pack::next()` span is a full chunk long).

So per CN the send side's contribution to *its own* arena is bounded by one lease. It contributes
one lease at a time to *each peer's* arena too — but N-1 peers each draining into you is still
O(N) leases in flight, not O(batches).

### 4.2 Receive side — peak live bytes equals the receiver's ENTIRE remote input

* `handle_transmit_packed` → `push_remote_frame` merely appends the arriving `StagedBatch` to a
  `Vec` (`local_exchange.rs:159` and `:240`). No copy, no release, no bound on the `Vec`.
* The lease is released only inside `run_fragment_inner`'s push loop (`engine.rs:546-570`)…
* …which cannot start until `take_ready` sees **every sender of every exchange input close**
  (`local_exchange.rs:248-313`; the completeness check is `complete != expected` at `:277-279`,
  and a remote sender counts only once its `eos` arrived, `:59-63`).

**Therefore peak receive-side live bytes = the pending receiver's entire remote input**, not
`batch_size × window`. That is the demand curve, and it scales with data, not with a tunable.

### 4.3 The code's comments already claim the good behaviour — and are wrong

Two places assert the invariant that does not hold:

```rust
// experimental/starrocks/src/engine.rs:543-545
// Remote senders: their packed batches already sit in this CN's staging arena. Push each
// (deep copy into pool memory), release its lease immediately — copy-out-on-arrival makes
// that safe — then close the sender.
```

```rust
// experimental/starrocks/src/fragment_executor.rs:74-77
/// Remote sender outputs already staged in this CN's arena, as
/// `(exchange node id, sender id, batches)`: pushed via `push_packed` + `close_input`
/// before the fragment runs, with each lease released the moment its push returns.
pub remote_inputs: Vec<(i32, i32, Vec<StagedBatch>)>,
```

Both are true *relative to the push loop* and false *relative to arrival*. The copy-out happens at
**run** time, not **arrival** time. `src/sirius_ffi.cpp:845` carries the same phrase. Anyone
reading only the comments will conclude the arena is bounded by one batch; it is not.

---

## 5. Evidence

### 5.1 Direct proof the arena never recovers (Repro A, `GPU_MEM=45GiB STAGING=48GiB`)

Source: `/tmp/cluster-sf500.log` (cluster launched by
`/opt/dlami/nvme/sirius-build/up-sf500.sh`, which hardcodes `GPU_MEM=45GiB STAGING=48GiB`, see
that script's line `export NUM_CNS=2 GPU_MEM=45GiB HOST_MEM=200GiB STAGING=48GiB`).

Four `arena exhausted` lines. Three are CN-side `ERROR` events; the fourth is the FE echoing the
first. All four report **byte-identical arena state**:

| Timestamp (UTC) | Emitter | Whose arena | Requested | Arena state |
|---|---|---|---|---|
| `19:11:38.630655` | CN, `request_staging_lease` failed | the **peer's**, via the failed lease RPC | 1 248 153 024 B | identical ↓ |
| `19:12:26.047` | FE echo of the above | — | 1 248 153 024 B | identical ↓ |
| `19:12:57.745808` | CN, `export_packed` failed (`sirius_ffi.cpp:757`) | its **own**, locally | 792 272 448 B | identical ↓ |
| `19:13:00.669564` | CN, `export_packed` failed (`sirius_ffi.cpp:757`) | its **own**, locally | 1 256 360 768 B | identical ↓ |

```
642284544 free of 51539607552 capacity in 10 blocks (largest 523966720),
67 leases outstanding holding 50897323008 bytes
```

The mixed remote/local observation is not a coincidence: at `19:12:56.042` the FE logs
`get bad heartbeat response: type: BACKEND, status: BAD, msg: java.net.ConnectException`, i.e.
**one CN had already died**. The three CN-side events are therefore consistent with all three
describing the **same** arena — the surviving CN's — first seen remotely (its peer's lease request
bounced off it) and then locally (its own `export_packed`). The tracing prefix does not tag the
emitting process, so that identification is inferential; the byte-identity of the state across all
three is not.

* 50 897 323 008 B = **47.40 GiB live of 48.00 GiB** (98.75 %).
* 642 284 544 + 50 897 323 008 = 51 539 607 552 — **conservation is exact**, so this is real
  retention, not an accounting bug.
* First to last CN event: `19:11:38.630` → `19:13:00.669` = **82.04 s**. Free bytes, block count,
  largest block, lease count and live bytes are all **identical to the byte**. Not one byte was
  released in 82 seconds.

In the same process, `/tmp/cluster-sf500.log` contains **78 lines mentioning `out_of_memory`**
(86 occurrences), including the `push_packed` failure quoted in §3 Link 1.

> **UNVERIFIED:** the figures "303 `out_of_memory` lines" and "pool pinned at exactly
> 48 318 382 080 B" could not be reproduced. `48318382080` (= 45 GiB exactly) appears in **no**
> surviving log, and the measured count in `/tmp/cluster-sf500.log` is 78 lines / 86 occurrences.
> The reason is a measurement gap, not a contradiction: that cluster was brought up by
> `up-sf500.sh`, which does **not** set `SIRIUS_LOG_BACKEND=spdlog`, so the engine's `[gpu_pool]`
> lines were never captured at all for this run (see §7.1). Treat the pool-at-cap claim for Repro A
> as plausible but unmeasured; re-run Repro A with logging on to establish it.

### 5.2 Arena demand is a function of pool health, not of the query

The same q21, same scale factor (SF500), same CN count (2), measured:

* **47.40 GiB** live with a starved pool (§5.1);
* **26.78 GiB** peak with a healthy pool — `28750023168 of 34359738368 bytes`, logged at
  `2026-08-19 20:36:57.881` in `/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`, at
  `GPU_MEM=60GiB STAGING=32GiB`, on the one CN that shut down cleanly at the end of the warm
  SF500X sweep whose last passing query was q21 (`results/sf500x.csv`).

A **1.8× spread** on identical inputs. No `(SF, N)` sizing formula can produce that.

**The folklore rule `STAGING ≈ 96 GiB × SF/500 / N` is retired.** It was wrong in both
directions: it over-sizes when the pool is healthy (wasting device memory the pool needs) and
under-sizes when the pool is starved (because demand is then unbounded by construction).

### 5.3 Measured arena high-water, for calibration

From `/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`, teardown lines emitted by
`exchange_staging_arena.cpp:159-176`:

| Peak live | Capacity | Timestamp | Attribution |
|---|---|---|---|
| 26.78 GiB | 32 GiB | 20:36:57.881 | end of warm SF500X sweep, last query q21 |
| 18.68 GiB | 32 GiB | 21:13:06.535 | immediately after `q18.r1.out` (21:13:06) → **q18** |
| 16.06 GiB | 32 GiB | 21:11:36.912 | immediately after `q17.r1.out` (21:11:36) → **q17** |
| 14.92 GiB | 32 GiB | 21:11:36.943 | same instant, the peer CN → **q17** |
| 9.59 / 9.50 GiB | 32 GiB | 20:41:27.188 / .145 | immediately after `q03.r1.out` (20:41:26) → **q03** |
| **47.40 GiB** | 48 GiB | — | Repro A, from the exhaustion message (§5.1), not a teardown line |

(Attribution works because `sweep-sf500x-cold.sh` restarts the cluster before each query, so a
teardown timestamp lands right after the previous query's last output file. Output mtimes are in
`/opt/dlami/nvme/sirius-build/bench/SF500XCOLD/`.)

> **UNVERIFIED:** "SF100 full 22/22 sweep 6.51 GiB of 32". No surviving log contains a peak
> anywhere near 6.51 GiB — the only three logs on the box with `peak live` lines are the spdlog
> file above (max 26.78 GiB) and two unit-test logs (max 0.00 GiB). Re-measure before relying on
> it. **SF300 was never measured at all.**

### 5.4 Survivorship bias — most runs measure nothing

The teardown line that reports `peak live` fires **only on clean shutdown**
(`exchange_staging_arena.cpp:159-176` is the destructor). A SIGKILLed CN yields nothing.

Measured in `/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log`:

* **96** arena creations (`exchange_staging_arena.cpp:79`) = 96 CN process lifetimes
  (48 bring-ups × 2 CNs);
* **48** teardowns (`exchange_staging_arena.cpp:168`);
* ⇒ **50 % of process lifetimes measured nothing.**

The cause is now understood and has been fixed in the harness: the CN allows
`SHUTDOWN_GRACE = 15 s` for graceful teardown (`experimental/starrocks/src/main.rs:34`, used at
`:675`), and `restart-sf500x.sh` used to `sleep 8` after `pkill`, killing CNs mid-teardown. The
script now sleeps 20 s and documents exactly this.

> **UNVERIFIED:** the "61 %" figure quoted elsewhere. The number measured here is **50 %** for the
> one surviving corpus. Either way the bias is the same and it points the same direction: the
> corpus is skewed toward healthy runs, because a run that ends in a wedge is exactly the run that
> gets SIGKILLed. **This bias already produced one wrong conclusion** (the retired sizing formula
> in §5.2). Treat any "measured arena high-water" number as a lower bound on true demand until the
> 20 s grace change has been in effect for a full sweep.

### 5.5 Exchange state is unreachable by the spill/downgrade machinery

The downgrade executor enumerates only the **per-query** repository registry
(`src/downgrade/downgrade_executor.cpp:223`: `auto const managers = _data_repo_registry.get_all();`).
Exchange repositories are by construction outside it — `src/exec/streaming_fragment.cpp:64-66`:

> *"Created here, outside `data_repository_manager_`, so `QueryEnd()`'s
> `clear_all_repositories()` cannot destroy them. This is what lets a sender's output outlive its
> own fragment."*

Measured consequence, from the spdlog corpus: **356 downgrade events, every single one freeing
exactly 0 bytes** (`GPU Pipeline Executor: after downgrade (0 bytes freed), reservation still
partial …`, emitted at `src/pipeline/gpu_pipeline_executor.cpp:262-270`). Distinct values of the
"bytes freed" field across all 356 events: `{0}`.

So under exchange pressure the spill machinery is a no-op. This is load-bearing for
[§8.6](#86-option-6--spill-staged-batches-to-host-memory).

---

## 6. Diagnostic rule — how to read an `arena exhausted` message

The exhaustion message deliberately reports both `total_free` and `largest_free`
(`exchange_staging_arena.cpp:242-256`, and the rationale comment at `:242-244`). Read it
**together with** the pool's `[gpu_pool] … peak=` line from the same window
(`SiriusContext::log_pool_stats`, `src/sirius_context.cpp:229`, emitted at `:254-259`, called at
QueryBegin `:371` and QueryEnd `:448`).

| Reading | Meaning | Action |
|---|---|---|
| arena **> 90 %** full **and** pool `peak` == pool cap | pathological ratchet (§3) | **Do NOT raise the arena — fix the pool.** Raising the arena takes device memory *from* the pool (§2.1) and makes it worse. |
| arena **> 90 %** full, pool well under cap | genuine demand | raise the arena |
| arena **< 70 %** with `largest_free < request ≤ total_free` | external fragmentation | a bigger arena may not help; look at lease size distribution |

A live example of the third row, from Repro B (§7.3): request `376283136`, `total_free`
`464351488`, `largest_free` `230149888` — the space existed but not contiguously. (That same
event was *also* 97.3 % full, so both the first and third rows applied at once; check the
occupancy ratio first.)

---

## 7. Reproduction

Both reproductions below were observed. **Repro B is the cheap iteration loop** (~10–15 s to
failure); Repro A is the definitive demonstration but takes minutes.

### 7.0 Read these scripts before running anything

* `/opt/dlami/nvme/sirius-build/up-sf500-x.sh` — bring-up. Knobs, all env vars with defaults:
  `GPU_MEM` (60GiB), `HOST_MEM` (200GiB), `STAGING` (32GiB), `NUM_CNS` (2), `DGT`/`DGS`
  (downgrade trigger/stop fractions, 0.8/0.6), `DISK` (`/opt/dlami/nvme/sirius_spill`), and the
  `operator_params` group `HPB` / `MBHT` / `STB` / `CBB` / `MSPB` (unset ⇒ engine default).
  It writes one `.cn$i-x.yaml` per CN, echoes a `=== SF500 experiment config ===` header, starts
  the FE and `NUM_CNS` CNs on port base 9100 stride 10, and `wait -n`s.
  It exports `SIRIUS_EXCHANGE_STAGING_BYTES=$STAGING`, `SIRIUS_LOG_BACKEND=spdlog`,
  `SIRIUS_LOG_DIR=/opt/dlami/nvme/sirius-build/siriuslog`, `SIRIUS_LOG_LEVEL=info`,
  `SIRIUS_QUERY_WATCHDOG_SECS=280`.
* `/opt/dlami/nvme/sirius-build/sweep-sf500x.sh` — warm sweep (`bench.sh --cold`), one cluster for
  the whole run, restart only on failure. `OUT` defaults to `bench/SF500X/timings.csv`.
* `/opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh` — **fresh cluster per query**
  (`bench.sh --cold-restart`), `FE_QUERY_TIMEOUT=1800`, `OUT` defaults to
  `bench/SF500XCOLD/timings.csv`. Costs ~68 s of restart per query.
* `/opt/dlami/nvme/sirius-build/restart-sf500x.sh` — the `RESTART_CMD` both sweeps use. It
  `pkill`s CN + FE, sleeps 20 s (see §5.4), re-exports `GPU_MEM`/`STAGING`/`HOST_MEM` and the
  operator params, relaunches `up-sf500-x.sh` with output **redirected to
  `/tmp/cluster-sf500x.log` using `>`**, sleeps 60 s, then sets the FE's `query_timeout`.

### 7.1 Gotchas — every one of these cost real time

1. **Engine logs exist ONLY with `SIRIUS_LOG_BACKEND=spdlog`.** Only `duckdb`, `spdlog`, `noop`
   are accepted. On the CN's FFI path an unknown value is **silently discarded**: the throw in
   `install_configured_log_sink` is guarded by `if (db)` and the CN passes `nullptr`
   (`src/sirius_context.cpp:1550-1579`; the `else if (db)` guard is at `:1573`, the CN's call is
   `src/sirius_ffi.cpp:177`). Also set `SIRIUS_LOG_DIR` and `SIRIUS_LOG_LEVEL`
   (`src/sirius_context.cpp:1583-1585`). Without this there are **no** `[gpu_pool]`, downgrade, or
   arena telemetry lines at all — which is exactly why Repro A's pool state is unmeasured (§5.1).
2. **Both CNs append to ONE shared log file.** Distinguish by the `instance=0x…` token, which
   rides the `QueryBegin` window label (`SIRIUS_LOG_INFO("QueryBegin: {}", window_label)`,
   `src/sirius_context.cpp:372`) — e.g.
   `QueryBegin: instance=0x7650c80ba520 connection=1 query=1`. Note the deployed binary that
   produced the archived corpus prints `[sirius_context.cpp:312]` in its line prefix, so the
   prefix line numbers in old logs do **not** match the current tree; use the message text, not
   the prefix.
   Concurrent appends also **tear lines**: a real example from the corpus is
   `peak live 3585662976 o leases outstanding, 1 free blocks, …` at `21:15:59.355` — the middle of
   the line was clobbered. Parsers must tolerate it.
3. **A restart that redirects the cluster log with `>` OVERWRITES it.** `restart-sf500x.sh` does
   exactly this. **Archive `/tmp/cluster-sf500x.log` after every run.** This is not theoretical:
   the cluster logs for experiments E4A…E8 are already gone, which is why several config→failure
   attributions in §7.3 are unverifiable.
4. **`grep` fails silently on these logs.** Plain `grep -c arena` on
   `siriuslog/sirius_2026-08-19.log` returns **nothing with rc=1**, while `grep -a -c arena`
   returns **144**. (The torn lines from gotcha 2 make the file look binary; the wrapper also
   passes `-I`.) **Always use `grep -a`, or parse with `python3`** using
   `open(p,'rb').read().decode('utf-8','replace')`.
5. **The FE's `query_timeout` defaults to 300 s** — `private int queryTimeoutS = 300;` at
   `experimental/starrocks/starrocks/fe/fe-core/src/main/java/com/starrocks/qe/SessionVariable.java:1317`
   — and it aborts healthy slow queries server-side regardless of any client-side timeout. Run
   `SET GLOBAL query_timeout=1800;`.
6. **A restart script that does not re-export `GPU_MEM`/`STAGING` silently falls back to bring-up
   defaults** (60GiB/32GiB), invalidating the experiment without any error.
   **Always assert the config actually used** by reading the echoed
   `=== SF500 experiment config ===` header out of the cluster log.
7. **`bench.sh` has NO correctness gate.** Its only checks are `rc == 0`, non-empty output, and no
   `ERROR` on the first line (`experimental/starrocks/benchmarks/tpch/bench.sh:175`). It never
   compares against an oracle. **Always** diff with
   `python3 /opt/dlami/nvme/sirius-build/compare.py <out_dir> /opt/dlami/nvme/sirius-build/oracle-sf500f64`.
8. Do not raise `STAGING` and `GPU_MEM` together past the card. Device occupancy per GPU is
   `GPU_MEM + STAGING + ~2 GiB CUDA context` against **94.97 GiB allocatable** — the bring-up
   script echoes this formula but does not enforce it.

### 7.2 Repro A — starved pool, large arena (the classic ratchet)

```bash
GPU_MEM=45GiB STAGING=48GiB HOST_MEM=200GiB NUM_CNS=2 \
  nohup /opt/dlami/nvme/sirius-build/up-sf500-x.sh > /tmp/cluster-reproA.log 2>&1 &
sleep 60
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL query_timeout=1800;"

# then run q21 at SF500
GPU_MEM=45GiB STAGING=48GiB OUT=/tmp/reproA.csv \
  /opt/dlami/nvme/sirius-build/sweep-sf500x.sh 1 q21
```

Expected: `exchange staging arena exhausted` with the arena at ~47.4 GiB of 48 (98.7 %), the state
line byte-identical across repeated events tens of seconds apart, and `out_of_memory` from
`push_packed` in the same window. Cost: minutes per iteration.

### 7.3 Repro B — small arena, healthy pool (fast and cheap)

**Verified instance, `GPU_MEM=76GiB STAGING=16GiB`, q08 at SF500** — recorded in
`results/sf500e3.csv` (q08 cold `refused` 11 592 ms, warm `refused` 1 689 ms) with the CN log
preserved as `/tmp/cluster-e3.log`, whose header reads
`GPU_MEM=76GiB STAGING=16GiB HOST_MEM=200GiB CNs=2`.

Four CN-side exhaustion events, arena capacity 17 179 869 184 B = 16 GiB:

| Time | Reporting sender¹ | Requested | Free | Blocks | Largest | Leases | Live |
|---|---|---|---|---|---|---|---|
| `21:33:15.708564` | :9101 (F02) | 1 249 445 376 | 961 310 464 | 4 | 944 076 544 | 20 | 16 218 558 720 (**15.10 GiB, 94.4 %**) |
| `21:33:15.760427` | :9111 (F02) | 1 247 395 840 | 1 011 332 352 | 6 | 957 617 408 | 17 | 16 168 536 832 (15.06 GiB) |
| `21:33:17.367275` | :9111 (F08) | **376 283 136** | 464 351 488 | 7 | 230 149 888 | 22 | 16 715 517 696 (**15.57 GiB, 97.3 %**) |
| `21:33:17.470804` | :9111 (F11) | 312 075 392 | 305 262 848 | 7 | 194 237 696 | 22 | 16 874 606 336 (15.72 GiB) |

¹ All four are `request_staging_lease failed` — so the **arena state described is the
destination's**, while the FE's `backend=127.0.0.1:PORT` (thrift port; CN0 = 9101, CN1 = 9111 at
port base 9100 stride 10) names the *sender* whose drain failed. Do not read the port as the owner
of the arena.

The first pair is q08's **cold** run (cluster came up ~21:33:04; 11.6 s later ⇒ matches the CSV's
11 592 ms exactly). The second pair is the **warm** run 1.66 s later (matches 1 689 ms). Note the
warm run failed on a **376 MB** request against a 16 GiB arena holding ~15.6 GiB — and that
`largest_free (230 149 888) < request (376 283 136) ≤ total_free (464 351 488)`, the fragmentation
signature from §6.

```bash
# Repro B, ~10-15 s to failure
GPU_MEM=76GiB STAGING=16GiB HOST_MEM=200GiB NUM_CNS=2 \
  nohup /opt/dlami/nvme/sirius-build/up-sf500-x.sh > /tmp/cluster-reproB.log 2>&1 &
sleep 60
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL query_timeout=1800;"
GPU_MEM=76GiB STAGING=16GiB OUT=/tmp/reproB.csv \
  /opt/dlami/nvme/sirius-build/sweep-sf500x.sh 1 q08

# read the outcome (grep -a is mandatory, see gotcha 4)
grep -a "arena exhausted" /tmp/cluster-reproB.log
grep -a "\[gpu_pool\]"    /opt/dlami/nvme/sirius-build/siriuslog/sirius_*.log | tail -20
mkdir -p /tmp/archive && cp /tmp/cluster-reproB.log /tmp/archive/reproB-$(date +%s).log   # gotcha 3
```

> **UNVERIFIED (logs overwritten, gotcha 3):** three further Repro-B-shaped datapoints are on
> record but their cluster logs were destroyed by `restart-sf500x.sh`'s `>` redirect:
> `GPU_MEM=68GiB STAGING=24GiB` q08, `GPU_MEM=70GiB STAGING=22GiB` q09,
> `GPU_MEM=65GiB STAGING=27GiB` q09. Their CSVs survive (`results/sf500e{4a,5,6,7,8}.csv`) but the
> failure *mode* cannot be confirmed from them. There is also a **direct contradiction** in the
> record for 68/24 — `../SF500-CONFIG-AND-ARCHITECTURE.md` lists it as *pool OOM* while other notes
> call it *arena exhausted*. Do not cite 68/24 either way until it is re-run with the log archived.
> `results/sf500e3.csv` + `/tmp/cluster-e3.log` (76/16) is the one fully evidenced Repro B.

### 7.4 Regression baseline

Any fix must not regress the healthy baseline:

```bash
GPU_MEM=60GiB STAGING=32GiB HOST_MEM=200GiB HPB=1GiB MBHT=2GiB STB=1GiB CBB=1GiB \
  /opt/dlami/nvme/sirius-build/sweep-sf500x-cold.sh
python3 /opt/dlami/nvme/sirius-build/compare.py \
  /opt/dlami/nvme/sirius-build/bench/SF500XCOLD /opt/dlami/nvme/sirius-build/oracle-sf500f64
```

Current result at that config: SF500 **21/22** (only q09 fails).

---

## 8. The backpressure design space

For each option: mechanism, where it lives, cost, deadlock risk, transport interaction, and how to
measure whether it worked.

### 8.1 Option 1 — Credit-based flow control

**Mechanism.** The receiver grants each sender a byte (or batch) budget for a given exchange. A
sender may only request a staging lease while it holds credit; credit is decremented on grant and
returned when the receiver releases the corresponding lease. A sender out of credit blocks until
credit is replenished. This is the textbook answer and the one every mature shuffle engine lands
on.

**Where it lives.**
* Credit accounting alongside the rendezvous state in `experimental/starrocks/src/local_exchange.rs`
  — a per-`(ExchangeKey, sender_id)` counter next to `remote_seq`
  (`ExchangeState`, `local_exchange.rs:88-95`), so it shares the existing `Mutex<ExchangeState>`.
* Credit **granted** on the `request_staging_lease` reply — extend `PStagingLeaseResult` with a
  `credit_remaining` field and have `handle_staging_lease`
  (`compute_node_service.rs:1105-1110`) consult the counter before calling
  `executor.staging_lease`.
* Credit **returned** where the lease is released: the push loop at `engine.rs:562-570` and the
  error path at `engine.rs:403-420`. Because `staging_release` already runs on the caller's
  thread against a `Send + Sync` handle (`engine.rs:696-700`), returning credit there does not
  touch the engine channel.
* Sender-side wait in the drain loop, immediately before `rpc_request_lease`
  (`nixl_transport.rs:712`).

**Given that `ExportNext` is already serialised through the single engine thread**
(`engine.rs:88-91`, `:702-704`), the sender never has more than one batch in hand. So the natural
credit unit is *bytes*, checked once per drain iteration, and the whole sender-side change is one
gate at `nixl_transport.rs:711`. A sender blocking there parks the **transport thread**, which is
exactly the right thread to park: it stops that CN producing more exchange bytes without touching
the engine thread, the dispatch thread, or the BRPC runtime.

**Cost.** Wire-format change (`PStagingLeaseRequest`/`Result` and/or `PTransmitPackedParams`),
new state in `ExchangeState`, and a wakeup mechanism (condvar or a channel) so a returning credit
unblocks a waiting sender. Moderate: days, not hours. Also needs a credit-refund path for a query
that dies, or the counters leak the same way §3 R2 leaks leases.

**Deadlock risk — the central problem.** Credit is only returned when the receiver's push loop
runs, and the push loop only runs after `take_ready` sees **every** sender close
(`local_exchange.rs:248-313`). If a sender must block for credit that only the receiver's run can
return, and the receiver cannot run until that sender closes, **that is a guaranteed cycle, not a
risk**. Credit-based flow control therefore **does not work on its own**: it requires that the
receiver be able to consume (and thus return credit) *before* its sender set is complete — i.e.
it requires [§8.4](#84-option-4--copy-out-on-arrival-plan-01). With copy-out-on-arrival in place
the cycle is broken, because credit is returned by the arrival handler, which runs on a BRPC
`spawn_blocking` thread and depends on nothing.

**Transport interaction.** Blocking before `rpc_request_lease` (`nixl_transport.rs:712`) keeps the
nixl agent idle rather than mid-transfer, so it does not interact with `write_and_wait`'s poll
loop or its `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS` timeout (`nixl_transport.rs:792-830`). Blocking
*after* posting an `XferReq` would be much worse — do not do that. Note the single transport
thread means a blocked sender blocks **all** of that CN's outbound drains
(`nixl_transport.rs:215-217`); with N=2 that is acceptable, at larger N it is a throughput
hazard and argues for per-destination credit with a non-blocking "skip to the next destination"
policy.

**How to measure.** Arena `peak_live_bytes()` must become **bounded and roughly independent of
scale factor**: run Repro B (q08 at SF500) at `STAGING=16GiB` and at `STAGING=8GiB` and require
that both pass and that the reported peak is within ~2× of `max_batch_bytes + 8 MiB`. Add a
counter for "sender blocked on credit, cumulative ms" and assert it is > 0 under Repro B (proving
the mechanism engaged) and ≈ 0 on the 60/32 baseline (proving it costs nothing when healthy).

### 8.2 Option 2 — Blocking or retrying lease instead of throwing

**Mechanism.** `exchange_staging_arena::lease` currently scans once and throws
(`exchange_staging_arena.cpp:230-256`). Replace with a bounded wait: park on a condvar until a
`release` signals, retry the fit scan, give up with the existing error after a timeout.

**Where it lives.** `src/exec/exchange_staging_arena.cpp:212-257` (`lease`) and `:259-291`
(`release`, which would `notify_all`). A `lease_for(len, timeout)` overload plus a new
`SIRIUS_EXCHANGE_STAGING_LEASE_TIMEOUT_SECS` knob. Optionally also expose it through
`engine.rs:690-694` so the wait is configurable per call site.

**Cost.** By far the cheapest change — hours. Header, one `.cpp`, one env knob, one unit test.

**Deadlock risk — high, and the analysis matters.** The comforting fact is that the blocking
thread and the releasing thread are **different**: a lease requested over RPC is served on a
`spawn_blocking` worker (`compute_node_service.rs:519`), whereas releases happen on the engine
thread (`engine.rs:563`) or the transport thread (`nixl_transport.rs:743`). The engine thread is
single and its context is `!Send`/`!Sync` (`engine.rs:1-22`), so it must never be the thread that
blocks — and with the current routing it is not.

But the *logical* cycle from §8.1 is still present and is the dominant risk:

> Sender S blocks in `lease()` waiting for space. That space can only come from receiver R's push
> loop. R's push loop cannot start until `take_ready` sees S close. S cannot close because it is
> blocked. **Deadlock, resolved only by the timeout.**

So a blocking lease **converts a fast failure into a slow failure** in exactly the pathological
case, and only helps in the transient case where some *other* receiver is about to drain. It is
still worth doing as a cheap partial mitigation — many arena-exhausted events are momentary — but
it must be time-bounded well under the FE's `query_timeout` (§7.1 gotcha 5) and under
`SIRIUS_QUERY_WATCHDOG_SECS` (default 280 s here; parsed at `src/sirius_engine.cpp:99-105`).
Suggested bound: 5–15 s.

Secondary hazard: blocking `spawn_blocking` workers. Tokio's blocking pool is finite; N-1 peers
all waiting on leases plus `transmit_packed` and `exec_plan_fragment` also using `spawn_blocking`
(`compute_node_service.rs:554`, `:331`) could starve the pool and stall arrivals — which would
prevent the very releases being waited for. Cap the number of concurrently-waiting leases.

**Transport interaction.** None directly: the wait happens on the receiver before any `XferReq`
exists. But the sender's `rpc_request_lease` is a synchronous BRPC call
(`nixl_transport.rs:885-887`), so a long receiver-side wait shows up as a long RPC and must stay
under whatever BRPC/PRPC timeout applies. **Check that timeout before choosing the wait bound** —
if it is shorter than the lease wait, the sender will fail anyway and you will have gained
nothing.

**How to measure.** Repro B at 76/16 should change from `refused` in ~11 s to either pass (if the
pressure was transient) or fail at the timeout (if it was the cycle). Instrument
`lease_wait_ms_total` and `lease_wait_timeouts`. If timeouts dominate, the diagnosis is the cycle
and you need §8.4/§8.5 — a useful result either way, obtained in one 15 s iteration.

### 8.3 Option 3 — Sender-side admission (query the destination before packing)

**Mechanism.** Add a cheap `staging_headroom` RPC (or piggyback `total_free`/`largest_free` on the
`transmit_packed` reply, which is already round-tripped every batch). Before calling
`export_packed_next` — i.e. before any packing work happens — the sender checks whether the
destination has room and refuses/waits early.

**Where it lives.** New arm in `PInternalService` next to `request_staging_lease`
(`compute_node_service.rs:512`), reading `arena.total_free()` / `largest_free()`
(`exchange_staging_arena.hpp:117-120`). Sender side: the top of the drain loop,
`nixl_transport.rs:706`, before `export_packed_next`.

**Cost.** Low — hours to a day. No wire-format break if piggybacked on `PTransmitPackedResult`.

**Deadlock risk.** Low if it only *fails earlier* (advisory admission). If it *waits*, it inherits
exactly the §8.2 cycle.

**Value.** Real but limited. Its genuine benefit is that today the failure happens **after** the
sender has already packed the batch into its own arena and, for `export_packed`, after the
`chunked_pack` gather has run against the pool (`sirius_ffi.cpp:746-765`). Failing before that
saves the pack cost and produces a much clearer error (`destination CN1 has 464 MB free, need
1.2 GB` instead of a peer-side exhaustion dump). It also makes the fragmentation case (§6 row 3)
visible to the sender, which could choose a smaller `kPackChunkBytes` slack or split the batch.

**Transport interaction.** One extra small RPC per batch, or zero if piggybacked. Negligible next
to a multi-hundred-MB RDMA write.

**How to measure.** The failure message changes and the time-to-failure drops. Not a fix; a
diagnosability and cost-avoidance improvement. Measure "bytes packed then discarded" before and
after; it should go to zero.

### 8.4 Option 4 — Copy-out on arrival (PLAN-01)

**Covered in depth by [`PLAN-01-copy-out-on-arrival.md`](PLAN-01-copy-out-on-arrival.md); not
duplicated here.** In one line: move the `push_packed` deep-copy from the run-time push loop
(`engine.rs:546-570`) to the arrival handler (`handle_transmit_packed`,
`compute_node_service.rs:605`), which is what the comments at `engine.rs:543-545`,
`fragment_executor.rs:74-77` and `sirius_ffi.cpp:845-847` already **claim** happens.

**How it interacts with backpressure — the key point of this section:**

* It **collapses retention**. Peak live arena bytes drop from *the receiver's entire remote input*
  (§4.2) to roughly one in-flight batch per sender, ≈ `p_max + 8 MiB`. On the measured corpus that
  is the difference between 26.78 GiB and single-digit GiB.
* It **does not bound the queue**. The bytes move from the arena into the RMM pool; they do not
  disappear. A receiver with a huge remote input still accumulates the whole thing, just in pool
  memory instead of arena memory. **Copy-out-on-arrival converts an arena-exhaustion failure into
  a pool-OOM failure.** That is a strictly better failure (the pool can spill; the arena cannot,
  §5.5) but it is not a bound.
* It is **a precondition for §8.1 and makes §8.2 work.** Both credit and blocking-lease are
  deadlocked by the "release only happens after every sender closes" cycle. Copy-out-on-arrival
  breaks that cycle by making lease release depend only on the arrival handler.

**Conclusion: PLAN-01 is necessary but not sufficient. PLAN-09 is the bound that PLAN-01 enables.**

### 8.5 Option 5 — Reserve pool capacity for the drain path

**Mechanism.** Guarantee that the arena→pool copy can always proceed by giving it a dedicated
reservation, so the drain never stalls on pool pressure and the ratchet (§3 Link 3) is broken at
the source.

**Where it lives.** `Fragment::push_packed`, `src/sirius_ffi.cpp:839-850`. Today it allocates from
`gpu_space->get_default_allocator()` (`:849`) — a raw pool reference with no reservation and no
retry.

**Does cucascade already support this? Yes.** `reservation_aware_resource_adaptor`
(`cucascade/include/cucascade/memory/reservation_aware_resource_adaptor.hpp:52`) is exactly the
adaptor the GPU pool already uses — `src/sirius_context.cpp:251-252` casts every GPU memory space
to it in order to emit `[gpu_pool]`. It exposes:

* `reserve(bytes, release_notifier)` → `std::unique_ptr<reserved_arena>` or `nullptr`
  (`reservation_aware_resource_adaptor.hpp:184`, impl
  `cucascade/src/memory/reservation_aware_resource_adaptor.cpp:365-373`, backed by
  `do_reserve` at `:525-533`, which is a bounded `try_add` against the memory limit);
* `reserve_upto(bytes, …)` for a best-effort partial reservation (`.hpp:192`, impl `.cpp:376-382`);
* `grow_reservation_by` / `shrink_reservation_to_fit` (`.cpp:384-400`).

And Sirius already knows how to use it under pressure: the pipeline executor does
`make_reservation_or_null` → `request_downgrade` → retry → proceed-with-partial
(`src/pipeline/gpu_pipeline_executor.cpp:225-280`). **The exchange drain path simply does not
participate in any of that machinery.**

**Two shapes:**
1. *Standing drain reservation.* At bring-up, reserve a fixed slice of the pool
   (e.g. `max_exchange_batch_bytes × in_flight_leases`) and allocate every `push_packed` copy out
   of it. Guarantees forward progress absolutely; costs that slice unconditionally.
2. *Per-copy reservation with downgrade.* Wrap `push_packed`'s allocation in the same
   reserve→downgrade→retry loop the pipeline executor uses. Costs nothing when healthy, but
   §5.5 shows the downgrade sweep currently frees **0 bytes** under exchange pressure, so this
   only helps if the pressure comes from *non-exchange* state.

**Cost.** Shape 1: hours to a day, plus a sizing decision (which is a new tuning knob — a real
downside given §5.2's lesson about sizing formulas). Shape 2: hours, but see the caveat.

**Deadlock risk.** Low. A standing reservation is acquired once at bring-up. The failure mode is
not deadlock but *under-sizing*: too small and the drain stalls anyway; too large and the pool
loses capacity the query needs. Note that a reservation is charged against
`_total_allocated_bytes` immediately (`reservation_aware_resource_adaptor.cpp:527-531`), so
reserving 4 GiB is exactly equivalent to lowering `GPU_MEM` by 4 GiB from the query's point of
view.

**Transport interaction.** None. This is purely receiver-internal.

**How to measure.** Under Repro A, `push_packed` must never emit `out_of_memory`, and the arena's
`live_bytes()` must show a **sawtooth** rather than a monotone ramp. Sample `live_bytes()` every
second and assert `max(live) - min(live)` over any 10 s window is > 0 — that single assertion is
the definition of "the ratchet is broken".

### 8.6 Option 6 — Spill staged batches to host memory under arena pressure

**Mechanism.** When arena occupancy crosses a threshold, evict the oldest staged batches to host
memory (they are already contiguous, packed, device-resident byte ranges — an ideal DMA source),
release their leases, and re-stage on the way into `push_packed`.

**Where it lives.** A hook in `push_remote_frame`, plus a re-stage step before `push_packed` in
`engine.rs:556`.

**Where the hook goes.** `push_remote_frame` (`local_exchange.rs:159-246`) is the arrival point;
a sweeper would walk `ExchangeState::sources` (`local_exchange.rs:90`).

**Blocker.** The existing downgrade machinery **cannot see this state**. The downgrade executor
enumerates only the per-query repository registry
(`src/downgrade/downgrade_executor.cpp:223`), and exchange repositories are deliberately created
outside it (`src/exec/streaming_fragment.cpp:64-66`). Measured: **356 downgrade requests, 0 bytes
freed** (§5.5). So this option is really *two* work items: (a) make exchange state reachable by
the sweep, (b) implement the spill. Item (a) overlaps heavily with the park-ownership work
(PLAN-02) and should not be done twice.

**Cost.** Highest of all options. Days to weeks, and it introduces a host-memory budget as a new
failure surface.

**Deadlock risk.** Moderate — a spill that itself needs a pinned-host allocation under memory
pressure is a classic deadlock. Would need a pre-pinned host staging slab, i.e. yet another
reservation.

**Verdict.** Correct in the limit, wrong for now. Spilling is what you do when you have already
bounded the queue and still need headroom. Bound it first.

### 8.7 Option 7 — Bound in-flight bytes per exchange or per query

**Mechanism.** A pure admission gate, weaker than credit: a per-`ExchangeKey` (or per-query) cap
on total leased bytes, enforced inside `handle_staging_lease` before the arena is touched. Over
cap ⇒ reject with a *distinct, retriable* status.

**Where it lives.** `compute_node_service.rs:1105-1110`, reading a counter kept next to the
rendezvous state (`ExchangeState`, `local_exchange.rs:88-95`). Decrement wherever
`staging_release` is called: `engine.rs:563` (push loop), `engine.rs:411` (fragment error path),
`nixl_transport.rs:743` (sender's own local lease), `compute_node_service.rs:616` (canary).

**Cost.** Lowest of the real fixes — a counter, a config value, an error code. Hours.

**Value.** It converts an *unfair* global failure into a *fair, attributable* local one. Today one
runaway exchange starves every other exchange and every other query on the CN; the error names
whichever innocent request happened to arrive last. With a per-exchange cap, the exchange that
actually exceeded its budget is the one that fails, and the message can say so. That is a large
diagnosability win for a very small change, and it composes with everything else.

**Deadlock risk.** Zero if it only rejects. Nonzero if it waits (same cycle as §8.2).

**Transport interaction.** None beyond a new status code the sender must classify.

**How to measure.** Under Repro B, the failing message should name the offending exchange node id
and its budget, and other exchanges in the same query should no longer report exhaustion. Count
distinct exchange keys appearing in exhaustion messages: should go from many to one.

---

## 9. Recommendation

An ordered combination, to be landed in this sequence. Each step is independently valuable and
each unlocks the next.

**Step 0 (hours) — make the failure legible and stop losing the evidence.**
Land §8.7 (per-exchange in-flight cap, reject-only) plus the §8.3 advisory headroom check, and add
`live_bytes()` / `peak_live_bytes()` / `total_free()` / `largest_free()` to a periodic
`[staging_arena]` log line next to `[gpu_pool]` (`src/sirius_context.cpp:229`). Also fix the
harness gaps from §7.1: archive cluster logs per run, and always run `compare.py`.
*Why first:* every subsequent step is validated by these numbers, and §5.4 shows the current
corpus is 50 % blind.

**Step 1 (days) — PLAN-01, copy-out on arrival.**
This is the single highest-value change: it collapses arena retention from "the receiver's whole
remote input" to "one batch per sender", makes the existing comments true, and — decisively —
**breaks the release-depends-on-completion cycle** that deadlocks every form of blocking
backpressure (§8.4).

**Step 2 (hours) — bounded blocking lease (§8.2), 5–15 s cap.**
Only safe *after* Step 1. Absorbs transient pressure at near-zero cost. Verify the BRPC/PRPC
timeout on `request_staging_lease` first (§8.2), and cap concurrent waiters so the
`spawn_blocking` pool cannot be starved.

**Step 3 (hours–days) — drain reservation (§8.5, shape 1).**
Guarantees the arena→pool copy can always make progress, which is the last remaining way the
ratchet can re-form. Size it as `max_exchange_batch_bytes × expected_concurrent_arrivals`, not as
a fraction of the pool — a fraction is exactly the kind of formula §5.2 killed.

**Step 4 (days) — credit-based flow control (§8.1).**
The real bound. Do this once Steps 1–3 have shown that the release path is reliable; credit on an
unreliable release path is just a slower deadlock.

**Deferred: §8.6 (spilling).** Requires making exchange state reachable by the downgrade sweep
(§5.5), which overlaps PLAN-02. Revisit only after Step 4, and only if a measured workload still
exceeds a bounded queue.

### 9.1 The minimal experiment that validates the recommendation

Repro B is ~15 s per iteration, so this is cheap.

| Arm | Config | Query | Expected today | Expected after Steps 0–2 |
|---|---|---|---|---|
| B1 | `GPU_MEM=76GiB STAGING=16GiB` | q08 | `refused`, arena 94–97 % full | pass, or a message naming the offending exchange |
| B2 | `GPU_MEM=76GiB STAGING=8GiB` | q08 | `refused` | pass (the strong signal: demand decoupled from capacity) |
| A1 | `GPU_MEM=45GiB STAGING=48GiB` | q21 | `refused`, byte-identical state over 82 s | `live_bytes()` sawtooths; pass or a clean pool-OOM |
| BASE | `60/32 + 1GiB` operator budgets, full 22 | all | 21/22, max rel dev 3.3e-10 | **≥ 21/22, no correctness regression** |

### 9.2 Numeric success criteria

1. **Bound.** With Steps 0–2 landed, arena `peak_live_bytes()` on q08 at SF500 must be
   ≤ `4 × (max_batch_bytes + 8 MiB)` and must **not** grow when `STAGING` is doubled from 8 GiB to
   16 GiB (within 10 %). Demand must stop tracking capacity.
2. **No ratchet.** Under Repro A, sampling `live_bytes()` at 1 Hz for 60 s, the value must
   decrease at least once in every 10 s window. Zero decreases over 60 s = the ratchet is still
   there.
3. **No permanent leak.** After every failed query, `outstanding()` must return to its pre-query
   value within 30 s. Today it does not (§3 R2).
4. **Coverage.** ≥ 90 % of CN process lifetimes must emit a teardown `peak live` line
   (currently 50 %, §5.4).
5. **No regression.** SF500 full sweep stays at ≥ 21/22 with `compare.py` max relative deviation
   ≤ 1e-9 against `/opt/dlami/nvme/sirius-build/oracle-sf500f64`.
6. **No cost when healthy.** On the 60/32 baseline, cumulative sender block time and lease wait
   time must both be < 1 % of wall clock, and per-query timings must be within 5 % of the current
   `results/sf500xcold.csv`.

### 9.3 Risks

* **Deadlock is the dominant risk and it is structural, not incidental.** Any waiting mechanism
  introduced before Step 1 will deadlock on the cycle in §8.1. Do not reorder the steps.
* **The engine thread must never block.** It is single and its context is `!Send`/`!Sync`
  (`engine.rs:1-22`), and the module doc records that funnelling leases through it once starved a
  peer's exchange for the PRPC timeout and failed the whole query (`engine.rs:16-22`). Leases must
  keep bypassing the engine channel (`engine.rs:682-700`).
* **Tokio blocking-pool starvation.** `request_staging_lease`, `transmit_packed` and
  `exec_plan_fragment` all use `spawn_blocking` (`compute_node_service.rs:519`, `:553`, `:330`).
  Unbounded waiters there will stall arrivals and thus stall releases.
* **The single transport thread is a head-of-line hazard at larger N.** One blocked destination
  blocks all outbound drains (`nixl_transport.rs:215-217`). Fine at N=2, needs per-destination
  handling before scaling out.
* **A drain reservation is a new tuning knob**, and §5.2 is a cautionary tale about sizing knobs
  derived from formulas. Size it from measured batch sizes, and log when it is exhausted.
* **The measurement corpus is biased toward healthy runs** (§5.4). Do not declare victory from a
  corpus in which the failing runs are the ones that produced no data. Verify the 20 s
  `SHUTDOWN_GRACE` fix is actually raising teardown coverage before trusting new high-water
  numbers.
* **Copy-out-on-arrival moves the pressure rather than removing it** (§8.4): expect
  arena-exhaustion failures to be replaced by pool-OOM failures until Step 4 lands. Budget for
  that in the schedule so it is not mistaken for a regression.

---

## 10. Appendix — claims that could NOT be verified

| Claim | Status | What was found instead |
|---|---|---|
| 303 `out_of_memory` lines on one CN in the Repro A process | **UNVERIFIED** | 78 lines / 86 occurrences in `/tmp/cluster-sf500.log` |
| Pool pinned at exactly 48 318 382 080 B (45 GiB) during Repro A | **UNVERIFIED** | that value appears in no surviving log; `up-sf500.sh` never set `SIRIUS_LOG_BACKEND=spdlog`, so `[gpu_pool]` was never emitted for that run |
| SF100 full 22/22 sweep peaked at 6.51 GiB of 32 | **UNVERIFIED** | no surviving log contains a peak near that value |
| SF300 arena high-water | **NEVER MEASURED** | — |
| `GPU_MEM=68GiB STAGING=24GiB` q08 → arena exhausted | **UNVERIFIED + CONTRADICTED** | `../SF500-CONFIG-AND-ARCHITECTURE.md` records 68/24 as *pool OOM*; the cluster log was overwritten |
| `GPU_MEM=70GiB STAGING=22GiB` q09 → arena exhausted | **UNVERIFIED** | CSV survives (`results/sf500e{6,7}.csv`), cluster log overwritten |
| `GPU_MEM=65GiB STAGING=27GiB` q09 → arena exhausted | **UNVERIFIED** | CSV survives (`results/sf500e8.csv`), cluster log overwritten; `/tmp/cluster-sf500x.log` now begins at 21:57:20, after that experiment |
| 61 % of process lifetimes measured nothing | **REVISED** | measured **50 %** (96 creations vs 48 teardowns) in the one surviving corpus |
| 94.97 GiB allocatable per card | not re-checked here | taken from `../SF500-CONFIG-AND-ARCHITECTURE.md`; no GPU was touched while writing this document |

### 10.1 Commands used to verify the log claims

```bash
# arena exhaustion events (plain grep returns nothing on these files -- see gotcha 4)
grep -a "arena exhausted" /tmp/cluster-sf500.log /tmp/cluster-e3.log

# arena create/teardown accounting and peak distribution
python3 - <<'EOF'
import re
p='/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log'
d=open(p,'rb').read().decode('utf-8','replace')
ls=d.split('\n')
create=[l for l in ls if 'exchange_staging_arena.cpp:79'  in l]
tear  =[l for l in ls if 'exchange_staging_arena.cpp:168' in l]
print(len(create),'creations',len(tear),'teardowns')
for l in tear:
    m=re.search(r'peak live (\d+) of (\d+)',l)
    if m: print('%7.2f GiB of %5.1f GiB %s'%(int(m.group(1))/2**30,int(m.group(2))/2**30,l[1:24]))
EOF

# downgrade effectiveness
python3 -c "
import re,collections
d=open('/opt/dlami/nvme/sirius-build/siriuslog/sirius_2026-08-19.log','rb').read().decode('utf-8','replace')
f=[int(m.group(1)) for m in re.finditer(r'after downgrade \((\d+) bytes freed\)',d)]
print(len(f),'events; freed values:',collections.Counter(f))"

# config actually used (gotcha 6)
grep -a -A4 "SF500 experiment config" /tmp/cluster-e3.log
```
