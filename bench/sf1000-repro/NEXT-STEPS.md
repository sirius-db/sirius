# Where the remaining 8.180 s goes, and what to try next

Companion to `README.md`. Everything here is measured on the **8.180 s** stack
(GB300, SF1000, 2026-08-01) unless marked as an estimate.

---

## Profile locations

Both were captured on the final build (`514e28b9`, bloom fast-range + cuco bucket 4).

| what | path |
|---|---|
| **quent** (task/operator telemetry, all 22 queries) | `/localhome/local-faramburu/quent_final/019fbe84-6b20-71f3-8ea3-6718f0901b61/` |
| **nsys** (kernel attribution, q9/q18/q13/q1/q21/q10) | `test/tpch_performance/output/tpch_20260801_181228_nsys-profile_gpu_iter2_prof_final/sirius/q*/nsys.sqlite` |
| earlier nsys at 8.501 s (q9/q18/q21/q13) | `.../tpch_20260801_*_prof_c/` |
| earlier nsys at 10.209 s (q1/q9/q18/q21 and q6/q10/q12/q16) | `.../*prof_now2*/`, `.../*prof_b*/` |

Provenance: the quent run measures 8.218 s vs the clean 8.180 s — a 0.5% telemetry
overhead. **Use quent for proportions, nsys and the clean run for absolute numbers.**

nsys needs an OpenSSL preload on this box or it dies at launch:
```
LD_PRELOAD=$PIXI/lib/libcrypto.so.3:$PIXI/lib/libssl.so.3:<custom libcudf>
```

---

## Where the time is

**Task lifecycle** (4,731 tasks): Queued 54.7%, Computing 32.7%, Preparing 12.5%,
everything else 0.1%. Scheduling machinery costs nothing. Queued is high because
8 pipeline threads serve 4,731 tasks and **GPU-busy is 91–97% of wall** — the queue is
full because the GPU is saturated, not because scheduling is failing.

> **CORRECTED AGAIN (later on 2026-08-02).** The NVTX table below is ALSO biased, in a
> different way from the quent one it replaced. Operator *range durations* are inflated by
> thread fan-out: q13's ranges sum to 2499 ms against 774 ms of wall (3.23x) because
> GPU_SCAN runs as 14 concurrent tasks on 8 host threads, each mostly **blocked**, while
> HASH_JOIN runs as **one** range whose 385.7 ms is almost exactly its 383.1 ms of GPU work.
> Normalising within a query divides out wall but NOT the per-operator thread count, so
> **fan-out operators are inflated and single-range operators are deflated**.
>
> Fair-share GPU attribution (each ns of GPU-busy shared among concurrent kernels, summing
> exactly to GPU-busy) gives the corrected picture:
>
> | operator | NVTX range (below) | **fair-share GPU** |
> |---|---|---|
> | GPU_SCAN | 33.5% | **19.07%** |
> | HASH_JOIN | 12.4% | **24.20%** |
> | HASH_GROUP_BY | 17.9% | 16.66% |
> | DYNAMIC_FILTER | 26.0% | 13.10% |
> | `<none>` (decompression on pool threads, outside any operator range) | - | 18.30% |
>
> **HASH_JOIN is the largest operator in the engine, not GPU_SCAN.** Individual instances
> are affected the same way: q13's GPU_SCAN is **3.72% of suite, not 7.61%**.
> Use fair-share GPU attribution for ranking. Use NVTX ranges only to find *which* kernels
> belong to an operator, never to size it.

**By operator** — *superseded, see the correction above*. Share of **suite
wall clock**, from nsys NVTX operator ranges across **all 22 queries**:

| operator | % suite | seconds |
|---|---|---|
| GPU_SCAN | 33.49% | 2.740 |
| DYNAMIC_FILTER | 26.00% | 2.128 |
| HASH_GROUP_BY | 17.87% | 1.462 |
| HASH_JOIN | 12.40% | 1.015 |
| FILTER | 2.67% | 0.218 |
| PROJECTION | 2.06% | 0.169 |
| MERGE_GROUP_BY | 1.72% | 0.141 |
| CONCAT | 1.67% | 0.137 |

**Hottest individual operator instances** (this is the actual target list):

| rank | query | operator | % suite |
|---|---|---|---|
| 1 | q18 | HASH_GROUP_BY P3(4) | 7.84% |
| 2 | q13 | GPU_SCAN P3(3) | 7.61% |
| 3 | q1 | HASH_GROUP_BY P0(3) | 5.44% |
| 4 | q9 | DYNAMIC_FILTER P21(23) | 3.97% |
| 5 | q21 | DYNAMIC_FILTER P12(14) | 3.54% |
| 6 | q10 | GPU_SCAN P3(3) | 3.23% |
| 7 | q1 | GPU_SCAN P0(0) | 2.83% |
| 8 | q12 | GPU_SCAN P0(0) | 2.30% |
| 9 | q7 | GPU_SCAN P18(22) | 2.28% |
| 10 | q8 | DYNAMIC_FILTER P33(37) | 2.19% |

DYNAMIC_FILTER dominates a whole cluster of mid-size queries that had never been
profiled: q17 94.5%, q21 80.2%, q8 75.2%, q5 60.9%, q4 56.0%.

### Method note — the previous table on this line was wrong

It read GPU_SCAN 40.6%, DYNAMIC_FILTER 20.0%, PROJECTION 11.5%, HASH_GROUP_BY 10.3%,
FILTER 7.7%, HASH_JOIN 5.0%, and was derived from **quent**. That was a mistake:
quent's operator stream carries *only declarations*, and its timing unit is the whole
**pipeline**, not the operator. Those percentages were an **even split of each
pipeline's task time across the operator names in its chain**. 86% of active time is
in multi-operator chains, so the approximation was doing nearly all the work, and
neither DYNAMIC_FILTER nor FILTER ever appears as a solo pipeline, so nothing in the
quent data anchored them at all.

The clean case study is q1's main pipeline, `GPU_SCAN -> PROJECTION -> PROJECTION ->
HASH_GROUP_BY`: the even split credits each 25%; NVTX measures HASH_GROUP_BY 59.5%,
GPU_SCAN 31.0%, the two PROJECTIONs 9.2% combined.

Net effect of the correction: **HASH_GROUP_BY and HASH_JOIN were understated ~2x**,
**PROJECTION and FILTER were overstated 3-4x** (they were inflated purely by riding
along in chains), GPU_SCAN was overstated.

**Use nsys NVTX for operator attribution, never quent.** Sirius emits one range per
operator instance named `Pipeline <P>: <OPERATOR> (id=<N>)`, nested inside a per-task
range; they are non-overlapping siblings covering ~78% of task time. Scripts:
`nvtx_ops.py` (per-query shares) and `nvtx_suite.py` (suite-wall-weighted) in the
session scratchpad. quent remains the right tool for task *lifecycle* states,
queueing and memory-tier behaviour — just not for per-operator cost.

Scan-rooted pipelines are ~60% of all active time; every one of the top 18 pipelines
begins with `GPU_SCAN`.

**By kernel across the top six queries (58% of suite):**

| kernel | share |
|---|---|
| `single_pass_shmem_aggs_kernel` (q1) | 28.2% |
| groupby `compute_single_pass_aggs_sparse_output_fn` (q18) | 18.9% |
| `transform_kernel` (AST_JIT expressions) | 10.9% |
| cuco `retrieve` + `insert_if_n` + `count` | 21.4% |

---

## Ranked next experiments

### 1. Eliminate the cuco `count` pass — est. ~2.3% of suite

`count` is 15.3% of q9 and 7.3% of q13. It is a **separate probe pass that only sizes
the output** before `retrieve` fills it: 8.6 ms/call against `retrieve`'s 12.9 ms/call,
so ~66% of it is duplicated probe work.

It cannot be fused as-is — `retrieve` appends via an atomic counter and output is
unbounded for many-to-many. The viable route is a **build-side uniqueness fact**: TPC-H
joins are overwhelmingly PK-FK, and for a unique build side an inner join's output is
bounded by probe rows, so the allocation needs no count. **No cudf API exposes this
today** (`inner_join_size`/`inner_join` and the `*_match_context`/`partitioned_*` family
all count first), so this is an upstream API change, not a local one.

### 2. Close q1's groupby cardinality gap — est. up to 3% of suite

`single_pass_shmem_aggs_kernel` is 58.5% of q1 and the largest single kernel in the
engine. The shmem replication patch microbenchmarked **4.39x at cardinality 4** and
delivered **~5%** end-to-end, because real per-block cardinality is far above 4. The
win collapses with cardinality (4.39x at 4, 1.70x at 64, 1.01x at 127).

**First step is measurement, not code:** instrument the actual per-block distinct-group
count in q1's shmem path. If it is ~64, the patch is already near its ceiling and this
line closes. If it is ~4 and the win is still only 5%, the model is wrong and that is
worth knowing.

### 3. q18's sparse-output groupby — 18.9% of the top six, mechanism unknown

`compute_single_pass_aggs_sparse_output_fn` is 54.8% of q18. Two things are already
ruled out: the aggregation itself is **at the memory floor** (a plain non-atomic scatter
measured 8.945 ms vs cudf's 9.006 ms), and the hash table is not the problem
(q18 takes the global path at α≈0.125, terminating in ~1.07 slots). The remaining cost
is the `insert_and_find` probe plus the row-comparison gather. Needs a fresh NCU pass
to attribute within the kernel.

### 4. Two upstream cuco defects, both found and neither exploited

- **`bucket_type` has no `alignas`.** `cuda::std::array<T,BucketSize>` gives `alignof` 4
  despite `bucket_storage::alignment` guaranteeing 16; the bucket-4 groupby kernel
  emitted **zero `LDG.E.128`**, while the join's instantiations do contain them.
  Untested lever.
- **`fast_int` division is discarded.** `make_iterator` computes
  `hash % (upper_bound / BucketSize)`, and `fast_int::operator/(fast_int, Rhs)` returns a
  *plain* integer, dropping the precomputed magic numbers — so the modulo lowers to a
  full software division, twice per key. **Measured 0.99x in our IN-list probe** (ALU
  hidden behind probe latency), but the *join* probe retires in ~1.07 steps and has
  little latency to hide behind, so it may bite there.

### 5. Bloom `add` is atomic-bound — 34.5 ms/iteration

The 730.8M-key `orders` Bloom costs ~80 ms/iteration (build + probe) to remove ~51% of
rows from half the splits. Fast-range does not help it (34.52 vs 34.53 ms). It is the
weakest of q21's three filters; a cheaper build or a decision not to publish that one
filter is worth investigating.

### 6. `Preparing` is 41% inside DYNAMIC_FILTER vs 12.5% overall

Decode is disproportionately concentrated in the filter path. This is the one place a
late-materialisation scheme could still pay — but see the refutation below before
investing.

### 7. Decisions, not experiments

- **`ast_jit` default.** Worth −4.17%. Two queries regressed in isolation (q19 +26%,
  q22 +17.6%) but both recovered in combination. The JIT cache persists to disk, so
  compile cost is one-time and pre-warmable. Needs a call, not a measurement.
- **Physical clustering at pin time.** Sorting `lineitem` by `l_shipdate` would make
  bitpack zone maps real (they currently prune 0/4000). Trades pin-time cost for
  query-time gain across the date-filtered queries. Pin-time sensitivity unmeasured.

---

## Session 2 (2026-08-02) — what was established

Baseline re-measured this session: **8.183 s**, 22/22 byte-identical against the stored
reference. Within 0.43% of the 8.218 s telemetry run and 0.04% of the recorded 8.180 s.
Every query within +-3.3%. Per-query CSV:
`test/tpch_performance/output/tpch_20260802_074620_grouped_gpu_iter3_baseline_session2/`

**nsys now covers all 22 queries**, not six:
`test/tpch_performance/output/tpch_20260802_075002_nsys-profile_gpu_iter2_prof_all22/sirius/q*/nsys.sqlite`
The previously unprofiled 16 queries were 43.3% of the suite. They contained real hot
spots -- `q12 GPU_SCAN(0)` at 2.30% of suite and `q19 GPU_SCAN(4)` at 2.18% had never
been looked at.

### Confirmed in emitted code (SASS of the shipped libcudf), not yet measured

1. **cudf's aggregation atomics default to SYSTEM scope.** `cuda::atomic_ref<T>` in
   `device_atomics.cuh:43` takes no scope argument. Found independently by two agents in
   two unrelated kernels (q1's shared-memory path, q18's global path). On a *shared*
   address it emits a runtime address-space dispatch that issues
   `ATOM.E.ADD.64.STRONG.SYS` -- which **fails on shared** -- then falls back to
   `QSPC.E.S` + `LDS.64` + `ATOMS.CAST.SPIN.64` retry. ~7 wasted instructions per
   aggregation. Whole-kernel census: `ATOMS.ADD` = 0.
   **The obvious fix is refuted**: `thread_scope_block`/`_device`/`_system` all emit the
   same chain, because sm_103a has no generic 64-bit shared atomic add. What works is
   handing the compiler a *shared-space pointer* (`atomicAdd((unsigned long long*)p,v)`).
   Bonus: int32 COUNT via raw `atomicAdd(int*,1)` lowers to `ATOMS.POPC.INC.32`
   (one warp-aggregated increment) vs 32 separate system-scope atomics -- and **4 of q1's
   9 aggregation columns are COUNT**.

2. **cuco `bucket_type` really does emit no `LDG.E.128`** -- confirmed in the shipped
   join `retrieve` (2x `LDG.E.64` per bucket) and in Sirius's IN-list object (0x128,
   60x64), whose own source comment claims a single 16/32 B fetch.
   **`alignas` is refuted as the fix** -- byte-identical SASS *and* PTX. nvcc lowers
   `cuda::std::array<T,N>` element-wise and merges only up to `alignof(T)`. The fix is an
   explicit homogeneous vector load (`uint4`/`longlong2`), which does produce `LDG.E.128`.

3. **`fast_int` magic reciprocals are discarded** -- confirmed in the join, not just
   groupby. Two 64-bit software modulos per key *inside* the per-key probe loop
   (`I2F.U32.RP`/`MUFU.RCP`/`F2I` + ~17 fixups), ~44 instructions incl. 6 XU-pipe ops, on
   the dependency chain before the address can form. `probing_iterator::operator++` keeps
   the magic; only the initial hash->index modulo is slow.
   Candidate rewrite `hash % (N/stride) * stride` -> `(hash % N) & ~(stride-1)` cuts join
   `count` static instructions 608 -> 504. **Caveat: it is a different permutation, so
   match emission order changes** -- must be checked against the byte-identical harness.

4. **q1's landed shmem patch did hit its target.** `num_cols` is 9 (not 11 --
   `extract_single_pass_aggs` dedupes same-kind aggs), `multiplication_factor` = 32 as
   intended, single pass over columns. The 1.09x in-situ figure is not a patch failure.
   Separately, **the replica layout is 8-way shared-memory-bank-conflicted**:
   `slot = g + 4*lane` puts an 8-byte column at byte offset `8g + 32*lane`, so a 64-bit
   warp access needs 8 wavefronts where 2 would do. The 4.39x microbenchmark had this too,
   so it is unexploited headroom rather than a regression.
   Occupancy is capped at 6 blocks/SM by `mapping_indices_kernel`'s 74 registers while the
   aggregation kernel (40 regs) could host 12 -- but the two kernels **must** share a grid
   (`local_mapping_index` is a rank within the mapping kernel's block), and raising the
   grid shrinks the shared-memory budget below one replica per lane. Decoupling is a
   redesign, not a patch.

5. **q18's kernel is two different workloads.** Split by NVTX range:
   phase A (`HASH_GROUP_BY`, lineitem 6.0e9 rows, sorted keys, decimal64) runs at
   19.5 G rows/s; phase B (`MERGE_GROUP_BY`, orders 1.5e9 rows, random order, **decimal128**)
   at 9.2 G rows/s -- 2.1x more per row. Sirius widens decimal64->decimal128 on the
   *partials* at `src/op/aggregate/gpu_aggregate_impl.cpp:409-416`, which forces every one
   of 1.5e9 merge rows through a 16-byte load + 16-byte CAS retry loop instead of one
   `ATOM.ADD.64`. cudf itself does not widen. Moving the widen after the merge saves only
   ~0.08% of suite *outside* the kernel; the prize is inside it and is unmeasured.

### Refuted this session

| claim | verdict |
|---|---|
| "No cudf API exposes count-free join" (this doc's own item 1) | **False.** `cudf::distinct_hash_join::inner_join` is that design verbatim, and Sirius **already routes to it** via `prove_unique_columns()`. It is dark at SF1000 only because tables are registered as `read_parquet` views, so `LogicalGet::GetTable()` returns nullptr. A Sirius metadata gap, not an upstream gap. |
| Dropping q21's orders Bloom is a win | **Neutral.** It costs 59.5 ms/iter GPU-serial (8.2% of q21) yet suppressing it moved q21 only -0.33% (0.7261 -> 0.7237 s), inside its +-1.7% noise. Downstream absorbs the saving. **Generalise this: removing GPU-serial work does not reliably become wall-clock.** |
| Adding `alignas` to cuco's `bucket_type` | Byte-identical SASS and PTX (see above). |
| Changing cudf's atomic scope | All three scopes emit the same chain (see above). |
| q1 groupby makes multiple passes over columns | 10,368 of 18,856 B -- one pass. |
| q1's DECIMAL128 columns are the villain | They are the *clean* path (`LDS.128`/`ATOMS.CAS.128`, no wasted atomic) because they use the CUDA `atomicCAS` builtin rather than `cuda::atomic_ref`. |

### Host-tier (C2C) payload columns — makes pin-once possible, costs ~13%

Sirius **cannot** tier columns of one table separately: `_pinned_entries` is keyed by the
`name` argument and `pinned_entry` holds a single `tier`; a second `pin_table` with the
same name *replaces*. The workaround is **two entries under two names**
(`orders` gpu / `orders_comment` host) — `try_match_cached_entry` takes the first entry
that is a column superset, so q13 can only be served by the host one. Caveat: a query
whose columns are a subset of *both* is served nondeterministically.

All arms 22/22 byte-identical, **zero CPU fallbacks** (unlike plain pin-once):

| arm | suite | vs 8.183 s |
|---|---|---|
| tier control — host payloads, per-query pin | 8.919 s | **+9.00%** |
| tier split — host payloads, pin-once, 2 orders entries | 9.238 s | +12.89% |
| tier A — pin-once, everything but lineitem on host | 9.456 s | +15.56% |

**The C2C transfer is the +9.00%** (the control holds pinning constant); pin-once plus
round-robin ordering adds only +3.89%. Per-query cost tracks host-resident column count:
q2 +130.7%, q4 +42.7%, q9 +32.1%, q12 +25.3% against q1 +1.0% (lineitem-only, still GPU).

Note `cached_databatch_provider::get_host_databatch` projects *before* transfer
(`host.slice(_column_indices)`), so a host entry only moves the columns actually read.

**The metric excludes pin time** (`performance_test.py:415-455` pins outside the timed
region in every mode), so pin-once can only ever *lose* on it. Measured against wall clock
including setup, ~25 s of per-query re-pinning would dominate a 1.06 s regression. Pin-once
is a latency/throughput trade, not a dead end.

**Re-measured on the it7 fused binary (2026-08-03, `bench/sf1000-repro/run-pinonce.sh`):
7.866 s, 22/22 byte-identical.** The fused wins survive the tier split nearly intact
(−14.9% vs the same layout pre-fused, against −15.4% per-query-pin) because lineitem and
orders stay GPU-compressed, so every selection decoder and membership mask keeps firing.

| arm (same binary, gate on) | suite | note |
|---|---|---|
| per-query pin, grouped (it7 bank) | 6.918 s | — |
| tier-split pin-once, sequential | **7.866 s** | +13.7% |
| tier-split pin-once, pre-fused (2026-08-02) | 9.238 s | fused = −14.9% on this arm |

The +0.948 s premium is the same C2C shape as before — q2 +127%, q9 +29%, q15 +26%,
q20 +26%, q19 +23% track host-resident column count; q1 is +2.8% (lineitem-only, still
GPU). Same caveat as above: the metric excludes pin time, so per-query mode also hides
~25 s of wall-clock re-pinning that pin-once pays once per session.

### Pin-once (union of all 22 queries' columns): OOM, cause identified

`--mode sequential` already implements it. It failed on the 8th pin (`orders`) at
**235.5 GiB peak against a 237.0 GiB limit** -- but resident was only 197.0 GiB, so it is
the *transient compression working set* that fails, not the footprint.
**Cause: only `lineitem` and `orders` have simpatico plans**; the other six tables log
`pin_table_compression is enabled but no plan file was found ...; pinning uncompressed`
and account for 63.3 GiB of that 197. `simpatico_cli` (the plan generator, a real CMake
target) has never been built in this tree.

Note: **nvidia-smi is useless for diagnosing pin memory** -- it reads a flat ~243 GB
throughout, which is Sirius's pre-allocated RMM pool, not live data. Use the run's own
`log_dir/sirius_*.log` `[gpu_pool] ... allocated= / peak=` lines.

---

## Do not retry — measured dead ends

| attempt | result |
|---|---|
| Bitpack zone maps / chunk skipping | **0/4000 chunks prunable** — every chunk spans the full domain because lineitem is orderkey-ordered and dbgen randomises dates per order |
| rle predicate pushdown | structurally ineligible — the only occurrence is an interior node that never produces a column's final value |
| str_split length pruning | cuDF already does it in `string_view::operator==`, which short-circuits on `size_bytes()` |
| Fusing filter into decode | ceiling **1–2%**, not the 20–26% first estimated — fusing does not remove the decode, only the full-width write and a subsequent pass worth ~1% |
| q18 sort-based merge | **+5.2%** — the ~37 GB gather under a random permutation cost more than the hash table it removed |
| q18 range-gap fast path | never fired — hash partitioning leaves partials range-overlapping |
| cuco bucket size in cudf's **groupby** | monotonically worse (bucket 1 optimal) — linear probing walks contiguous slots, unlike our double-hashed set |
| Sizing the groupby set by distinct instead of rows | α=0.125 **beats** α=0.5 (2.134 vs 2.583 ms); the oversizing is worth its footprint |
| `strings::like` warp-parallel path | already exists, gated at 72; **2x slower** at our 48.5-byte mean |
| Dropping `cudaMemcpyFlagPreferOverlapWithCompute` outright | **5x worse** on small buffers (0.19x at 64 KiB) |
| Dict-encoding short group keys | q1 **+19.9%** |
| Decompression prefetching | +1.6% — decode is SM work, nothing to overlap with |
| Lookahead scheduling / CONCAT barrier overlap | +0.6%, q10 +10.9% |
| BUILD_PROBE probe-side split | neutral |
| Twelve config knobs | only `scan_task_batch_size` mattered — see README |
| `enable_dynamic_filter_pushdown: false` | **livelocks**; the filter is load-bearing for memory feasibility |
| cuco `bucket_type` `alignas` (item 4) | **`alignas` emits byte-identical SASS *and* PTX** — nvcc lowers `cuda::std::array<T,N>` element-wise. Correct fix is an explicit vector load, which does yield `LDG.E.128` — but measured **−16.5% only L2-resident**, −0.6% in the join, and **+2.9% (worse)** for int64 once L2 spills. Real SF1000 join table is 2.4 GB vs a 115 MB L2. Dead. |
| cuco `fast_int` fast-modulo (item 4) | Real (two 64-bit software divides per key, ~44 instrs incl. 6 XU ops, confirmed in the shipped `retrieve` SASS). Removing **100%** of it is **~3% SLOWER** in the join, both regimes. The divide hides entirely in load shadow; the ~40 replacement ALU instrs do not. Order-preserving patch shelved with a bit-identity proof (50,100 points) in case a future arch is issue-bound. |
| Dropping q21's orders Bloom (item 5) | **Neutral.** Costs 59.5 ms/iter GPU-serial (8.2% of q21); suppressing it moves q21 −0.33%, inside its noise. Downstream absorbs the saving. |
| q21 Bloom `add`: halving atomics (u64 words) | **0.004 ms.** `add` is DRAM-bound on one random 32 B sector RMW per key, not atomic-bound. Read-only floor is 20.75 ms vs 34.26 ms — only 1.65×, the RMW traffic penalty. Irreducible for a blocked Bloom. |
| q18 deferring the decimal64→128 widen | **~0.13% of suite.** The 16-byte CAS loop is 6.9% of phase B at face value, ~3% after subtracting width-dependent non-kernel work. Not worth a correctness-risky narrowing. |
| q18 key-in-slot (avoid the comparator gather) | **+1.1% phase A, +22.5% worse phase B** — doubles the table's random-read footprint to save a cheap reference. |
| Pin-once (union of all 22 queries' columns) | **Pins but cannot run.** 203.7 GiB resident fits a 237 GiB pool, but q9/q13/q18 OOM to CPU (61.4 s, +650%). `after downgrade (0 bytes freed)` — **pinned memory is not evictable**, so the downgrade executor has nothing to reclaim; q18 needed 12.5 GB, got 65 MB, exhausted 100 retries. Not a residency problem, a working-set one. |
| cudf `device_atomics.cuh` SYSTEM-scope atomics | **+0.0%.** Three agents independently found it in SASS (`cuda::atomic_ref<T>` with no scope arg → `ATOM.E.ADD.64.STRONG.SYS`). An isolation rung differing *only* in the scope suffix (272 instrs each) measures SYSTEM 20.73 ms vs DEVICE 20.73 ms. Real in the disassembly, worth nothing. Would have shipped as an upstream patch on SASS evidence alone. |
| I-cache pressure as the groupby kernel's floor | **Refuted by NCU.** `no_instruction` is 2.90 of ~21 stall cycles; `long_scoreboard` is 14.56. Also not issue-bound: +512 ALU ops/row costs +17.8%, the first ~256 are ~free, and SM throughput is 25.3%. |
| Spin-lock backoff as the q1 layout×atomic mechanism | **Falsified by its own pre-registered test** — pacing was predicted to help under layout A and hurt under B; it hurt under both. The interaction is real and reproducible but currently **unexplained**. |

**The governing constraint:** GPU-busy is 91–97% of wall. Scheduling, overlap, prefetch
and parallelism ideas are dead on arrival here — only removing work or raising achieved
bandwidth moves the clock.

---

## Session 3 (2026-08-02) — filter-pushdown into decompression (payload skip)

**Distinct from the dead "fusing filter into decode" row above.** That was variant 1: fuse
the predicate into the *filter column's own* decode (1–2% ceiling, confirmed again here —
K1 vs K0+F0 is only 1.72×). What was never measured before is the **payload side**: produce a
bitmask (K1) or compact index list (K2, 10-bit chunk-relative) during the filter column's
decode, then have *other columns'* decoders consume it and emit **compacted** output directly,
skipping the full-width write + CUB select + gather round trip.

Microbench: `scratchpad/fusebench/` (fusebench.cu, sm_103a, n=2²⁹, simpatico chunk layout
1024-row chunks / chunk_min / per-chunk bits, all outputs verified bit-identical to baseline).

**Result: alive.** End-to-end (compressed → compacted payloads), best fused variant vs
decode-all→filter→gather baseline:

| sel | P=1 | P=3 | P=6 | winning variant |
|-----|-----|-----|-----|-----------------|
| 1.9% | 2.20× | 2.91× | 3.49× | FUSE_I (index list) |
| 15% | 1.94× | 2.32× | 2.56× | FUSE_I |
| 50% | 2.06× | 1.98× | 1.95× | FUSE_M (mask) |
| 98.5% | 2.23× | 2.19× | 2.17× | FUSE_M |

- Per-kernel: K3 (mask-consuming payload decode) −40% vs K0 at sel 1.9%; K4 (index-consuming)
  −77%, and 5.1× vs K0+gather. Mask/index crossover sits between 15% and 50% sel — the engine
  can pick per batch using the survivor count it must compute anyway (CNT wave).
- q6 shape (3-col conjunction, one mask): BASE3 8.16 ms → FUSE_M3 3.42 ms = **2.39×**.
- Dict codes (K1d predicate on 2–6-bit codes → gather only survivors): **2.1–2.6×** across all
  code-selectivities — uniform but below the pre-registered 3× bar (missed because baseline
  measured ~1.7× faster than the model anchor: K0 runs at 3.3 TB/s output on GB300, not the
  1.69 TB/s plan-file figure; FUSE absolute times landed on-model).
- 24-bit payloads: same story as 13-bit (2.38–3.05× at P=3).
- Projected filter columns ride the index variant for free (`_fc` case: FUSE_I unchanged,
  FUSE_M degrades to 1.6–1.8×).
- Known bench inefficiency: CNT.popc costs a flat 0.27 ms in every fused pipeline (~10× too
  slow); fixing it raises low-sel ratios another ~10%.
- Even at 98.5% selectivity fused wins 2.2× — the baseline's write-then-reread-then-rewrite of
  full-width payloads is pure overhead at any selectivity. This is *removing work*, which is
  the only thing that moves the clock per the governing constraint.

**Engine translation (estimate, not yet measured):** addressable pool is ~13.3% of suite scan
GPU + the write share of the 18.3% decompress pool; at measured 2–3.5× ratios that's roughly
**−3..−6% suite** after GPU-busy absorption, concentrated in q1 (harness literal makes
l_shipdate sel 52.6%, not 98.5%), q19, q10, q3, q12, q6, q14, q15. Validation order: q6 first
(3-col K1m3 + one K3, noise class H), then q1 (K1 + 4×K3 + masked dict gather), q14/q15 ride.

Engine plumbing required (from the code readers): numeric-range directive through
`compression_converters.cpp:94-109` (today strings-equality only); two-wave scheduling in
`decompress_columns_parallel` (simpatico_codegen.cpp:271-293); a ROW_FILTERED output contract
through decompress → converter → `gpu_ingestible.cpp:56-91` (full-column alloc at
decompress.cpp:543 must become count-first); renderer variants at the store-loop seam
(decode/jit/renderer.cpp:395-398) with predicate constants as kernel *params* (else every
literal is a ~300 ms NVRTC compile); mixed-mask combine wave for str_split conjuncts
(q19/q12 l_shipmode).

### Engine implementation (same day, branch `exp/fused-scan-filter`)

Implemented by a 4-agent team (JIT variants / extraction+directives / output contract /
wave orchestration) + orchestrator wiring, all behind `SIRIUS_EXP_FUSED_SCAN_FILTER`.
**Suite gate-on: 7.951 s, 22/22 byte-identical = −3.0% vs same-binary gate-off (8.196 s),
−2.8% vs the 8.180 s baseline** — inside the −3..−6% pre-registered band, iteration 1 only.

Per-query steady-state (all byte-identical):

| query | off | on | Δ | shape |
|---|---|---|---|---|
| q6 | 0.1584 | 0.0764 | **−51.7%** | 3-col mask, 4 tier-A, sel .019 |
| q14 | 0.1546 | 0.1033 | **−33.2%** | 1-col mask, 4 tier-A, sel .013 |
| q15 | 0.1489 | 0.1003 | **−32.6%** | same, sel .038 |
| q20 | 0.2047 | 0.1626 | **−20.6%** | same, sel .152 |
| q5 | 0.2708 | 0.2712 | ±0 (policy veto) | `o_orderkey` is delta→bitpack ⇒ tier-B |
| q1 | 0.7535 | 0.7562 | ±0 (policy veto) | 2 dict-string tier-B cols at sel .526 |

Without the policy, q1 measured **+43.5%** and q5 **+6.2%**: a tier-B column (full decode +
survivor gather) is strictly more work than classic post-filter compaction, and K3≈K0 at
sel .5. Hence the two-rule enable policy (W4): RULE 1 static — fuse only when every projected
column is tier-A (probe-verified bitpack leaf); RULE 2 dynamic — post-CNT bail when
survivors/rows > `SIRIUS_EXP_FUSED_SCAN_MAX_SEL` (default 0.35), reusing the mid-flight-failure
classic rerun (~1 ms/batch insurance). One debugging note for posterity: the M2 silent
fallback was a leftover `kSelectionMaskDecodeAvailable=false` scaffold in the orchestrator;
found by INFO-tracing each chain stage (SIRIUS_LOG_DEBUG never reaches the duckdb sink).

Iteration-3 queue, value order: (1) masked dict gather (K5) for q1's returnflag/linestatus —
microbench says 2.2–2.6× on exactly that shape, converts q1's veto into the top win;
(2) delta→bitpack selection-consuming decode — un-tier-Bs `o_orderkey`/`l_orderkey`,
unlocking q5/q3/q10/q12 payloads; (3) low-sel tier-B re-admission (q19: 3 tier-A + one
re-gathered col at sel .044); (4) K4 index-list variant below the 15% crossover;
(5) `bp_offsets` transient reuse across K1+K3 on the filter column.

### Iteration 3 results (2026-08-03, commit 56b0fe31): **suite 7.600 s, −7.3%, 22/22**

Items (1) and (2) shipped. **q1 −38.3%** (0.756→0.467; survivors 52.63% exactly as
predicted; dict-K5 general route + row_filtered tag) — the microbench's
"dict fusion wins at any selectivity" claim transferred. q5 −3.2% (delta tier flipped
its +6.2%), q10 −2.4%, q12 +0.6% (RULE-1 refusal is free). Suite gate-on **7.600 s,
22/22 byte-identical = −7.3% vs same-binary gate-off, −7.1% vs the 8.180 baseline** —
above the top of the pre-registered −3..−6% band. Campaign total 15.99 → 7.60 = −52.5%.

Debugging cost two blind cycles: a stale mid-edit binary (dict tier refused before the
classifier landed) and DEBUG-invisible logs. Fixed structurally: `SIRIUS_EXP_FUSED_SCAN_DIAG`
env-gated tracing + tag-vs-classifier cross-checks + single-ground-truth probes. Three
seam bugs were caught by the team's own review passes before they could bite (mask word
sizing ceil(n/32)→ChunksFor·32, a TierB gather race, a tier-collapsing adapter).

Known remainder: q3 +4.4% — RULE-2 bail insurance paid per batch (~36×) at sel .535;
fix in flight = bail memoization (selectivity is uniform across batches, zone-map study;
first bail disables fused for the scan's remaining batches). Then: q19-class low-sel
tier-B re-admission, K4 below the 15% crossover, `bp_offsets` reuse, K1_fc for projected
filter columns (q7/q3 economics).

**Iteration 3.1 (commit 00094c0b): bail memoization landed — suite 7.584 s, 22/22.**
q3 +4.4% → +2.4% (residual = the in-flight batches that pay insurance before the
per-operator latch is visible across the 4-stream convert pipeline; a pre-flight
selectivity estimate would close it). q1 −37.1% and q6 −51.3% unchanged.
**Final iteration-3 state: 7.584 s = −7.5% vs same-binary gate-off, −7.3% vs the
8.180 baseline; campaign 15.99 → 7.584 = −52.6%.**

### Iterations 4–6 + plan re-selection (2026-08-03, commits e9c18f6d..799b6889)

**It4** (e9c18f6d): K4 index decode (q6 −51→−63%, q14 −42%, q15 −44%) + low-sel TierB
re-admission + bool8 mask plumbing (bool8-ONLY masks measured +5.8% on q19 → routing
requires a range/pair source; dual-delivery documented for a future ranges+bool8 customer).
**It5** (f5bc9378): K6 str_split masked gather (q12 −35.8%) + compositional emitters +
pair machinery (dark: FilterCombiner folds q12's pairs into constant hulls — pairs have no
suite customer). **Suite 7.436 s, campaign −53.5%.**

**It6** (f4c4b6c2, REVERTED in 799b6889): sync surgery + decode memos. Sanitizer-clean,
formally audited — and **NEUTRAL** (post-revert control run: 7.484 ≈ the 7.485 it6 median;
the +0.46% vs the 7.451 stash run is attributable to the KEPT MAX_INDICES fix, whose four
newly-compressed pins carry projected/join-key columns — the role rule's third confirmation,
hiding inside a correctness fix). Revert stands per wins-only (neutral = debt).
Surgical follow-up available: role-correct those four pins' column plans (o_custkey et al.
→ identity) to reclaim the ~33 ms, or fold into the arm-D deployment decision.
H-already-optimal confirmed: decode kernels measured 1.7–3.5 TB/s (within
~10% of hand-tuned reference; the plan-file rates were 6.7–12× stale), the GPU is
saturated, the syncs were free. Kernel-level decoder work is DEAD by direct measurement.
Kept from the audit (35e65949): dict-encode row-cap fix (4 pins/suite had silently pinned
raw all campaign — narrow pins exceed rows-per-chunk proxies; q4 −2.4%), per-pin coverage
logging, explorer event-bracketed rates + frontier TSVs.

**Plan re-selection verdict**: current lineitem/orders plans validated (every proposed
change lost in-engine: ans/lz4 orderkeys re-confirmed the ee8fe639 lesson; dict l_shipmode
lost BOTH fused-K5 (+100 ms vs K6) and projected decode (194 vs 823 GB/s)). **Selection
rule: PROJECTED columns need fast full decode; filter/fused-served columns want max
ratio** — measured twice (l_shipmode, p_type q8-vs-q14). Arm D (surgical side tables:
dict on p_brand/p_container/c_mktsegment, identity on 12 projected string payloads) is
**suite-neutral with q10 −4.5% / q11 −5.4% / q4 / q8 wins and large pin-memory savings**
— deploy is a footprint call; it also reopens pin-once viability.

**Next frontier (unchanged, quantified by the quent census)**: DYNAMIC_FILTER masks —
19.3 s of task time sits immediately after GPU_SCAN (100% of that operator's occurrences),
structurally identical to what K1 feeds the selection decoders today. Plus: q19-shape
dual-delivery bool8, c_phone-class K6 widening, explorer per-column role input.

### Iteration 7 (2026-08-03, commit 73f46779): the dynamic-filter frontier collected —
**suite 6.918 s, 22/22 ×2 (0.01% spread) = −15.4% this session, campaign −56.7%.**

Membership masks live (Phase A, zero new kernels): q17 −20.9%, q9 −7.1%, q8 −18.6%,
q21 −10.2% — the 19.3 s DYNAMIC_FILTER bucket attacked from both ends (payloads compact
during decode; the operator self-disables). Dual-delivery bool8: q19 −25.0%. Debugging
lessons banked: drain-time snapshots precede join publication (fix: decode-time
re-snapshot, the disk path's late-binding pattern); multi-probe cost is pure wave-1
volume (nsys-attributed) → MAX_MEMBER=1 cap with keep-ordered sources (set forms before
Blooms) turned both regressors into wins. The duckdb log sink ignores levels
(duckdb_sink.cpp:61) — root cause of every DEBUG-invisible mystery this campaign; use
SET sirius_log_backend='spdlog'.

Iteration-8 shelf, pre-sized: K7 single-pass multi-probe with per-row short-circuit
(re-admits q8's full conjunction, ~0.19 predicted, lifts the cap); K1-delta + pair
machinery landed dark and unit-proven; dict/dual-delivery for arm-D side tables
(q17's part scan with p_brand/p_container code masks); explorer per-column role input.

---

## On this machine (pmgb300ws-0163) — concrete paths

Everything below already exists here; nothing needs regenerating.

| what | path |
|---|---|
| **SF1000 parquet** (265 GB, 8 table dirs) | `/localhome/local-faramburu/tpch_parquet_sf1000` |
| SF10 parquet (for the DuckDB-CPU oracle) | `/localhome/local-faramburu/tpch_parquet_sf10` |
| **built worktree at the 8.180 s state** | `/localhome/local-faramburu/repos/sirius/.claude/worktrees/perf-query-prs-integration-b64899` |
| **built patched libcudf** | `/localhome/local-faramburu/cudf-src/cpp/build/libcudf.so` |
| cuDF source (3 patches, branch `perf/sirius-sf1000-repro`) | `/localhome/local-faramburu/cudf-src` |
| compression plans in use | `/localhome/local-faramburu/plans_fastkey` |
| tuned config in use | `~/.sirius/sw_batch8.yaml` |

**Reproduce without rebuilding anything** — the worktree above already contains the exact
binary the profiles correspond to:

```bash
cd /localhome/local-faramburu/repos/sirius/.claude/worktrees/perf-query-prs-integration-b64899
DATA=/localhome/local-faramburu/tpch_parquet_sf1000 \
  pixi run bash bench/sf1000-repro/run.sh
```

`run.sh` defaults `CUDF_SO` to `$HOME/cudf-src/cpp/build/libcudf.so`; on this box the tree is at
`/localhome/local-faramburu/cudf-src`, which is the same path since `$HOME` is
`/localhome/local-faramburu`. Verify with `ls -la $CUDF_SO` before trusting a number.

**Serialise GPU access.** Only one workload at a time on this card, or every measurement in
flight is invalid. Two runs collided today and both OOM'd at extension init with
`failed to allocate 254503655833 bytes`, which looks like a bug and is not. Use the lock:

```bash
bash /tmp/claude-2099/-localhome-local-faramburu-repos-sirius--claude-worktrees-perf-query-prs-integration-b64899/f1f267de-6d11-4546-bb2e-9cd0411f45c2/scratchpad/gpu_lock.sh <your command>   # flock-based, atomic
```

Do **not** gate on `pgrep -f performance_test.py` — the pattern matches the wrapper shell's own
command line and returns a false positive. Gate on
`nvidia-smi --query-compute-apps=pid --format=csv,noheader` being empty, or use the lock.

**Rebuilding costs measurement time.** A full Sirius build is ~200 targets; libcudf is 557 cold
(~25 min) and ~25 s incremental. Build incrementally; never clean unless a build is actually
broken. And never run two `ninja` invocations against the libcudf build dir concurrently — it
corrupts `.ninja_deps` and costs two full rebuilds.
