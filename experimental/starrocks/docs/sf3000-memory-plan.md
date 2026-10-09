# Plan: make q05, q08 and q09 fit at TPC-H SF3000 on 4 GPUs

## Problem

At SF3000 on 4× GB200 (one CN per GPU, a 156 GiB slab pool each), q05, q08 and q09 run out of
GPU memory, even on a fresh cluster. Each one shuffles an unfiltered `lineitem` before any
selective join, and the CN materializes the whole shuffle at once:

- **Own scan output.** Each CN scans its quarter of `lineitem` and parks the whole partitioned
  output on its GPU: about 126 GB of projected columns. It ships nothing until the scan fragment
  has finished.
- **Incoming data.** At the same time the other three CNs push about 94 GB into it.
- **Total.** About 220 GB against a 168 GB pool.

Four independent fixes attack different parts of that peak. This plan covers each one, how much
it helps each query, and the order to do them in.

| # | fix | q05 | q08 | q09 | effort |
|---|---|---|---|---|---|
| 4 | join hints in the benchmark SQL | fits | fits | fits | low |
| 1 | runtime filters in the CN | fits (`lineitem` ÷5) | **no help** | fits (`lineitem` ÷18) | medium–high |
| 3 | spill exchange data to host memory | completes, slower | completes, slower | completes, slower | medium |
| 2 | stream scan output while the scan runs | peak ≈126 GB (tight) | peak ≈126 GB | peak ≈126 GB | medium |

**Recommended order:** #4 first, to get SF3000 numbers now. Then #3, the safety net that makes
every query complete. Then #1, which makes q05 and q09 fast. Then #2, which lowers the peak and
overlaps scanning with shipping.

---

## #1 Apply the FE's runtime filters in the CN

**What the FE already sends.** Every FILE_SCAN fragment of `lineitem` carries
`probe_runtime_filters` (`TRuntimeFilterDescription`, `gensrc/thrift/RuntimeFilter.thrift:100`),
and the join nodes carry the matching `build_runtime_filters` (`PlanNodes.thrift:788`). The plan
translator reads neither, and the CN doesn't implement the `transmit_runtime_filter` RPC.

In the SF3000 plans, the filters into `lineitem` are:

| query | filter | build side |
|---|---|---|
| q05 | `s_suppkey → l_suppkey` | `supplier` joined to `nation` and `region` (ASIA): about 1/5 of suppliers |
| q08 | `s_suppkey → l_suppkey` | `supplier` joined to `nation`, no filter |
| q09 | `→ l_partkey` | `part like '%green%'`: about 5% of parts |

So #1 fixes q05 and q09. It does nothing for q08, whose selective `p_type` filter never reaches
`lineitem`; q08 relies on #4, #3 or #2.

**Why it isn't just a translator change.** The CN runs a scan fragment as soon as the FE
dispatches it, and the FE deploys fragments in waves (`AllAtOnceExecutionSchedule.java:102`).
The scan can't block in its RPC or on the engine thread waiting for a filter: the build side may
be in a later wave, and blocking would deadlock.

**Stages:**

0. **Translator.** Parse `probe_runtime_filters` on scan nodes into
   `(filter_id, probe column, build_plan_node_id)`, and `build_runtime_filters` on join nodes into
   `(filter_id, build_expr)`. No behavior change.
1. **Deferred scan, filters built on the same CN.** In q05 and q09 the filter's build side is a
   broadcast join, so every CN already receives the full build input.
   - In `run_or_register` (`compute_node_service.rs`), a scan with a remote filter is parked in a
     per-query "pending filters" table, like receivers are, and the RPC returns OK.
   - When `LocalExchange` completes the exchange that feeds the filter's build side, summarize
     the key column from the parked or received batches: min, max, and the distinct keys up to a
     cap. Then dispatch the scan with an extra predicate.
   - A timer thread, not the engine thread, runs the scan **unfiltered** after a timeout, or
     right away if the filter shape isn't supported or the build side is too large. Filters are
     only an optimization, so the fallback is always correct.
   - Timeout: the larger of a CN default (about 500 ms) and `runtime_filter_scan_wait_time_ms`.
     The FE's 20 ms is too short for GPU fragments that run one at a time.
   - `fail_and_purge` also drops the query's pending scans.
2. **Predicate shape.** Always add `k BETWEEN min AND max`: it becomes decode ranges and prunes
   row groups (`scan_filter_analysis.hpp`). Add `k IN (...)` when there are up to about 64k
   distinct keys; the translator already emits `SingularOrList`, and it runs as a GPU `in_list`
   filter right above the scan. Above that cap, use the range only.
   - q05's ASIA suppliers at SF3000 are about 6M keys, so q05 needs stage 3. Its range alone is
     useless, because supplier keys are spread across the whole domain.
3. **Key-set FFI.** Add `Fragment::declare_scan_key_filter(read, column, keys)`. It builds the
   engine's existing `in_list` or bloom dynamic filter (`sirius_dynamic_filter.hpp`) and attaches
   it to the scan, so millions of keys never pass through Substrait literals. This makes q05 and
   q09 work at SF3000.
4. **Cross-CN filters (later).** For partitioned build sides, implement `transmit_runtime_filter`
   with a Sirius-specific payload. Merge the partial filters at `runtime_filter_merge_nodes`, then
   forward them to the probe scans.

**Tests:**
- Translator: parse filters from the recorded fixtures.
- Service: a pending scan dispatches when its build input completes, falls back after the
  timeout, and is purged on cancel.
- FFI: a key-set filter drops rows.
- End to end: SF1000 7/7 with no slowdown; SF3000 q05 and q09 pass, and the CN logs the filter
  being applied.

---

## #2 Ship scan output batch by batch

**Today.** The engine already streams inside a fragment. `sirius_physical_streaming_sink`
hash-partitions each batch into a thread-safe per-destination `batch_stream` while the fragment
runs (`src/op/sirius_physical_streaming_sink.cpp:159`). Four gates stop the CN from using that:

- `streaming_fragment::pull` refuses to run before the fragment finishes
  (`src/exec/streaming_fragment.cpp:461`).
- `Fragment::export_direct` treats "nothing yet" as end of stream (`src/sirius_ffi.cpp:623`).
- The Rust engine thread handles one request at a time, so exports wait behind the running
  fragment (`engine.rs:179`).
- `ship_remote` serves one peer at a time.

**A1, sender-side streaming:**
1. Add a C++ output handle that co-owns the `batch_stream` and the direct exchange. It offers
   `export_next(timeout)`, which returns a batch, *waiting*, end of stream, or an error, built on
   `classify`/`wait`/`try_pull`. It's thread-safe, so the NIXL thread drains it without going
   through the engine thread.
2. Allow `pull` while the fragment runs. Send end of stream only on `END_OF_STREAM`, never on
   "nothing yet". A failed run poisons the outputs, so the exporter sees the error.
3. In the CN, hand out the handles before `Run`, and start shipping to every remote destination
   at once. Release each batch after its WRITE completes. The local destination stays parked.

**Effect.** A CN's own output no longer has to be fully resident before the first byte ships.
The peak drops from about 220 GB toward about 126 GB plus working memory: q05 might just fit,
and scanning overlaps shipping.

**A2 (later).** Receivers that consume input as it arrives. The engine's streaming source already
handles open streams, but `run()` refuses open inputs and there's one query window per context.
Lifting that is a large change.

**Risks:**
- There's still no backpressure, so a slow peer grows the sender's queue. #3 absorbs that.
- Cancellation must stop in-flight streaming sends and keep the WRITE quarantine intact.
- The cardinality the CN reports for parked output must be recorded at sink time.

**Tests:**
- A two-thread `test_streaming_fragment` case: drain during `run()`, see *waiting* before end of
  stream, and get the error on an injected failure.
- A Rust engine test with exports interleaved with a run.
- `INJECT_FAILURE_QUERY` with the leak check.
- SF3000 with `SIRIUS_CN_TIMING=1`, showing the ship span overlapping `run_us`.

---

## #3 Spill exchange data to host memory

**Today.** The downgrade executor only sweeps repositories in the
`data_repository_manager_registry` (`src/downgrade/downgrade_executor.cpp:271`), and exchange data
isn't there:

- stream output repositories, which is where parked sender output lives;
- stream input repositories;
- direct-exchange receive buffers, which are raw `rmm::device_buffer`s.

So under pressure nothing moves to host. Receive allocation then fails outright, and the sender
fails the query (`exchange_direct.cpp:307`).

**B1, make exchange data spillable** (no NIXL change):
1. Register a staging repository manager for the CN's lifetime, under a reserved query id that
   is spilled first.
2. After a run, move the parked output repositories into it.
3. On a Packed frame, wrap the received buffers (`take(token)`) as a `data_batch` in a staging
   repository, rather than holding the raw token until the receiver runs.
4. `export_batch` moves a host-tier batch back to the GPU (the `memory_prefetcher` pattern:
   reserve, attach, convert). Stream inputs already accept host batches
   (`lock_and_prepare_batch`).
5. Record row counts at sink time, since a spilled batch can't report them.
6. In `allocate`, when the reservation fails, call `request_free_memory_and_wait(total)` and
   retry once. If it still fails, return a typed *retry* so the sender backs off instead of
   failing the query.

**B2 (later).** Let NIXL write straight into a registered pinned host slab (`MemType::Dram`) when
the GPU is full. That needs a contiguous pinned region and a host-table wrapper, because the
cucascade host tier is 1 MiB blocks.

**Settings it depends on:**
- `memory.host.capacity_bytes`: the run script uses 128 GiB per CN. SF3000 needs more, about
  256–320 GiB.
- The GPU downgrade trigger/stop fractions (0.8 / 0.6).

**Tests:**
- A downgrade unit test: a forced `request_free_memory` moves the staging repository to host.
- An exchange test: a host batch moves back to the GPU on export, and `allocate` against an
  exhausted pool spills and then succeeds.
- SF3000: q05, q08 and q09 complete (slower), downgrade logs show host bytes, and the leak
  counters return to zero.

---

## #4 Join hints in the benchmark SQL (no CN change)

**Why the FE picks this plan.** FILES() tables have no statistics, so StarRocks' cost-based
optimizer can't see that `part like '%green%'`, `p_type = '...'` or the ASIA region are
selective. It hash-shuffles `lineitem` against `orders` first.

**The workaround.** Hinted variants of the three queries that broadcast-join the selective
dimension into the `lineitem` scan before any shuffle:

| query | change | effect |
|---|---|---|
| q09 | `lineitem JOIN [BROADCAST] (part WHERE p_name LIKE '%green%')` first | `lineitem` ÷18 before the shuffle |
| q08 | `lineitem JOIN [BROADCAST] (part WHERE p_type = 'ECONOMY ANODIZED STEEL')` first | `lineitem` ÷150 |
| q05 | `lineitem JOIN [BROADCAST] (supplier ⋈ nation ⋈ region='ASIA')` first, then the shuffle join with `orders` | `lineitem` ÷5 |

Each variant runs with `SET disable_join_reorder = true` so the written order holds.

**In the tools repo:**
- the variants go in `tests/tpch-hinted/`;
- a `TPCH_SQL_DIR` setting selects them, with a matching oracle cache key, since DuckDB runs the
  same SQL minus the hints;
- the results are labelled "hinted".

**Caveat.** These aren't the official query texts, so they're a benchmark workaround, not a fix.
Statistics would let the FE find these plans itself, but FILES() can't be `ANALYZE`d; that would
need an external catalog.

**Tests:**
- SF1 and SF1000: the hinted results match DuckDB.
- SF3000: q05, q08 and q09 pass.
- `EXPLAIN` shows the broadcast placement.

---

## What `GPU_FRACTION` controls

`GPU_FRACTION` (default 0.85) becomes `sirius.memory.gpu.usage_limit_fraction` in each CN's
`sirius.yaml`, next to `allocator: slab`:

- **It sizes the CN's GPU pool.** At startup the engine does one `cudaMalloc` of
  `fraction × VRAM`, rounded down to 2 MiB (`slab_memory_resource.cpp:88`), and builds an RMM
  pool over it whose initial size equals its maximum, so it never grows. 0.85 × 184 GiB gives the
  **156.4 GiB** that appears in every out-of-memory error.
- **It's the CN's whole working memory.** Scan output, hash tables, parked exchange output and
  the receive buffers peers write into all come from this pool.
- **NIXL registers this region once,** which is why it has to be one fixed allocation.
- **The remaining 15% (about 28 GiB) is outside the pool.** It's left for the CUDA context,
  cuDF and NCCL/UCX internals, `cuda_ipc` mappings, and libraries that allocate outside RMM. The
  engine's own default is 0.95; the script uses 0.85 for headroom, since four NIXL/UCX endpoints
  share each box.
- **Reservations sit on top.** `reservation_limit_fraction` (1.0) caps what pipeline tasks may
  reserve within the pool. `downgrade_trigger_fraction` / `downgrade_stop_fraction` (0.8 / 0.6)
  decide when the downgrade executor starts and stops moving data to host. That only helps data
  it can see, which is why #3 matters.

**Raising it to 0.95 doesn't fix SF3000.** It adds about 18 GiB, while the shortfall is about
50 GB, and it shrinks the headroom outside the pool, which risks failures outside RMM.
