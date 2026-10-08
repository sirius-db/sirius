# Spill compression: handoff

State of the `compress-spills-v2` branch, how to benchmark it, what has been
measured, and what is open. The design and the measurement history behind it live
in [spill-compression-plan.md](spill-compression-plan.md); this document is the
operational companion — where things stand and how to pick them up.

## What the branch does

Spilled GPU batches are compressed with Simpatico on the way down the memory
hierarchy (GPU → host, GPU → disk, and in place on the device), and decompressed
transparently when a consumer locks them again. Per-column plans are keyed by the
spill edge (the producing repository), seeded from the offline per-table plans via
column lineage, explored on first contact when no seed exists, and re-explored
adaptively. The encoder allocates from a dedicated device arena
(`compression.device_pool_bytes`), never from the query pool whose exhaustion
triggered the spill.

Key configuration (`sirius.compression.*`):

| Key | Notes |
|---|---|
| `enable_spill_compression` | Master switch. |
| `device_pool_bytes` | The encoder's arena. Carved out of the GPU memory space's capacity (the query pool shrinks by the same amount; logged at startup), so it no longer over-commits the device. **Installed only from the config file at startup**: `SET spill_compression = true` at session start flips the flag but leaves the encoder allocating from the query pool — the documented pathology (compression latching on/off, a downgrade-request storm, the query killed). Sizing is a cliff, not a gradient: 1 GiB was too small for concurrent encodes on a 96 GB card and the query failed outright; 4 GiB worked there. 3 GiB is used on the 32 GB card and fills on SF3000 q18. |
| `input_plan_dir` | Offline table plans. Loaded whenever set, because the spill path seeds per-column plans from them through lineage, independent of pin compression. Only `lineitem` and `orders` plans are enabled under `plans/tpch_sf1000`; the rest were renamed `*_disabled.txt` in cbd95163 pending validation. |

## Branch state (2026-10-07)

- Merged `upstream/main` at 5a3ae17f (merge 9f5ea5d4). Frugal IO and the dense
  count-join fixes landed upstream as squashes of work this branch had merged, so
  those files took upstream's version and this branch's later commits were
  re-applied. Dropped in the merge: the scheduler stall watchdog (it read a single
  `_query`; the scheduler is now multi-query), the ctrack aggregate report, and the
  LUMP-only `query_stage_manager` / `stream_ordered_retirer` / `prefetch_census`.
- **cuCascade dependency.** The `cucascade` gitlink points at a350613 on
  `joosthooz/cuCascade:feat/try-release-table-on-dev`, which adds
  `gpu_table_representation::try_release_table(stream)` on top of cuCascade main.
  The spill encoder needs it to free each column as it encodes without
  `release_table()`'s deep copy of view-backed batches. It is not upstream; when
  cuCascade moves, rebase that commit and re-point the gitlink.
- Fixes found while benchmarking, both in IO code this branch carries:
  - c29bf4e4 — a multi-file split freed in-flight read buffers when a later
    source's allocation threw (an OOM retry): use-after-free, SIGSEGV in
    `cuMemcpyBatchAsync` on the REST reactor thread.
  - 456672d0 — fused reads were stranded in the REST reactor's `ready` queue, so a
    split stalled for exactly one upkeep interval (15 s) whenever nothing else was
    in flight; it hit almost every probe-side scan after a join build. SF1000 q2
    22.5 s → 3.9 s, q5 35.5 s → 8.6 s.
- Merge tooling: `tools/merge-dev.sh` merges `upstream/main` and runs
  `tools/merge-guard.sh` (anchor counts for registered-not-called code a wholesale
  take-theirs can delete without failing the build), then the build and the unit
  tests. Add an anchor whenever landing something a future merge could silently
  revert. Unit tests (`pixi run make test`) have not been run on the 5a3ae17f
  merge yet.

## Benchmarking

Harness: `bench/s3-sf1000/` (named for its first use; it does SF3000 and local
datasets too).

```bash
# one-time on a fresh box: instance-store NVMe at /mnt/nvme, re-made on every boot
sudo bash bench/s3-sf1000/setup-nvme-scratch.sh
aws sso login                                # S3 inputs; keys are exported per run

# an A/B, one process per query so a failing query does not abort the arm
SF=3000 NAME=base bash bench/s3-sf1000/run-each.sh
SF=3000 NAME=comp SPILL_COMPRESSION=1 bash bench/s3-sf1000/run-each.sh
python3 bench/s3-sf1000/ab-report.py test/tpch_performance/output/each_sf3000_{base,comp}.tsv
```

- `render-config.sh` fills `sirius-s3.yaml.in` and appends S3 keys exported from
  the SSO session (mode 0600, under `~/.sirius/bench`, never in the repo): Sirius'
  object-store factory takes static keys only. Re-rendered for every query, so a
  long arm survives role-credential expiry as long as the SSO session lives. Every
  value in the template is an env knob; see the header of `render-config.sh`.
- `run.sh` is one harness run. Also takes local datasets with pinning
  (`DATA=/path PIN=gpu|host|mixed PIN_COMPRESSION=1`), the experimental fused scan
  filter (`FUSED_SCAN_FILTER=1`), and the JIT expression evaluator
  (`EXPR_EVAL=ast_jit`). Always sets `enable_duckdb_fallback = false`.
- `run-each.sh` waits for the GPU to drain between queries, enforces
  `QUERY_TIMEOUT_S` (default 20 min), runs `memory-watchdog.sh` (kills the harness, not the box,
  below `WATCHDOG_FLOOR_MIB` of MemAvailable), and classifies failures: `gpu_oom`,
  `staging_exhausted`, `pin_oom`, `watchdog_killed`, `timeout`.
- `ab-report.py` — per-query time, host/disk spill, compression in → out, and an
  answer check between the arms (order-insensitive within ORDER BY ties).
  `spill-summary.py` does spill attribution for one multi-query harness run;
  `nic-summary.py` summarizes the NIC/CPU/MemAvailable samples every run records.

### Box: RTX PRO 4500 (32 GB), 32 vCPU, 120 GB RAM, 100 Gbit/s, 866 GB NVMe

The template defaults. GPU pool 0.86 of the device (both arms the same, so the
compression arena is a partition of the device and the A/B measures compression,
not pool size); host tier 100 GB pinned with 48 GiB of it pinned at startup
(pinning on demand mid-query cost q1 ~4 s on a cold cache); 1 MiB host blocks
(4 and 8 MiB gave no throughput and cost memory); disk tier on the NVMe. 8 REST
reactors × 128 connections with every read dispatched to all reactors
(`SIRIUS_DISPATCH_FANOUT=all`), 16 pipeline and scan threads; scan batch 2 GB, hash
partition and build 4 GB.

Prefetching cache: `cache.mode: sirius` with idle eviction and readahead 48 lifts
SF1000 scan throughput (q6 9.2 GB/s, NIC mean 60–66 Gb/s over the suite, peak
101 Gb/s). The cache allocates from the **same pinned host tier as spills and REST
staging**, and its ceiling is `eviction_threshold_fraction` of that tier — 0.8 by
default, i.e. 80 GB of 100. At SF1000 that pushed q9's spills to disk; at SF3000 it
exhausted REST staging and failed q9/q10. The template caps it at 0.2
(`CACHE_MAX_FRACTION`); `CACHE_MODE=none` turns it off.

**Update (after 1fe7c4c1 / c0139f4e): the cache works at SF3000.** The cache is now
hard-capped at its budget (demand reads over the cap go through REST staging,
uncached) and the readahead is bounded by resident bytes and host pressure. With
the cap, q3/q5/q13/q14/q17/q21 all complete with the cache on (q3 351 s vs 312 s
cache off; q13 29.8 s, q17 31.5 s, q21 48.0 s match or beat cache off). A
`max_readahead_bytes` sweep (4/8/16/32 GB) found 4 GB best: SF3000 q5 28.7 s and
q14 23.2 s (= cache off) vs 33.0 / 31.8 s at 32 GB, with SF1000 q1/q6 scan time
within 0.6 s -- prefetching only needs a few GB. The bench defaults to 4 GB
(`READAHEAD_BYTES`). The rest of this paragraph is the pre-fix diagnosis.

**Use `CACHE_MODE=none` at SF3000.** The cap does not bound the cache's real
footprint: with it at 20 GB the host tier still peaked at 100 of 100 GB, spills
went to disk (q5: 41 GB to disk capped vs 0 with the cache off), and five queries
that complete with the cache off failed (q3, q13 on the GPU retry limit; q17, q21
on exhausted REST staging) or slowed sharply (q14 23 → 44 s). The likely holder is
readahead — prefetched chunks not yet consumed are not evictable, and 48 splits at
SF3000 split sizes is tens of GB — but that is a hypothesis, untested.

### Box: RTX PRO 6000 (96 GB), 8 vCPU, 62 GB RAM, local NVMe at /mnt/datasets

The earlier harness (`bench/compress-spills-v2`, removed) ran TPC-H SF1000 from a
local NVMe copy with the input tables pinned. Its settings, as `run.sh` knobs:

```bash
DATA=/mnt/datasets/tpch/sf1000 PIN=gpu PIN_COMPRESSION=1 FUSED_SCAN_FILTER=1 \
EXPR_EVAL=ast_jit CACHE_MODE=none NEED_S3=0 \
GPU_USAGE_FRACTION=0.908 HOST_CAPACITY_BYTES=40000000000 HOST_BLOCK_SIZE=67108864 \
HOST_INITIAL_POOLS=32 DISK_CAPACITY_BYTES=1000000000000 SPILL_DIR=/mnt/datasets/sirius_spill \
PIPELINE_THREADS=6 SCAN_THREADS=4 SCAN_BATCH=4GB HASH_PARTITION=12GB HASH_BUILD=12GB \
DEVICE_POOL_BYTES=4GiB WATCHDOG_FLOOR_MIB=6144 \
bash bench/s3-sf1000/run-each.sh
```

(That box also used 2 uring reactors, 3 memory-prefetcher threads, a 4 GB concat
batch and 1 downgrade thread; set them in the template if reproducing exactly.)

What that box taught:

- **GPU fraction.** Applied to `cudaMemGetInfo` total (101.98 GB) and reserved
  eagerly. A bare CUDA probe tops out at 0.992; 0.95 peaks at 93 GB resident;
  0.908 leaves room for the 4 GiB arena at equal query memory in both arms.
- **A `bad_alloc` for about half the card at startup** is a redundant `LOAD`: the
  `build/release/duckdb` binary links Sirius statically and initializes it, so an
  explicit `LOAD` builds a second pool of the same size. Drop the `LOAD`; do not
  lower the fraction. (The Python harness loads the extension module, where the
  `LOAD` is correct.)
- **Host tier.** Pinned and not reclaimable: 40 GB of 62 GB, 16.4 GB of it at
  startup. 52 GB (50 pools) left ~10 GB of headroom and drew the OOM killer during
  startup probing.
- **The OOM killer took the box down** when a GPU OOM fell back to a DuckDB CPU
  replay of q18 at SF1000 in the ~22 GB the host tier left. Hence
  `enable_duckdb_fallback = false` everywhere and the memory watchdog (floor 6 GB;
  that box idles at 19–22 GB available during a run).
- **Pinning.** Several SF1000 queries (q9 first) have a pinned column set larger
  than the GPU tier, and pinned data is not evictable, so the pin hard-fails —
  the reason for one process per query. q1's compressed `lineitem` pin is 71.2 GB,
  larger than the box's RAM, so it cannot be host-pinned; the workable mixed split
  keeps `lineitem`/`orders` on the GPU and pushes
  `PART CUSTOMER SUPPLIER NATION REGION PARTSUPP` to host.
- **Wedged shutdown.** An engine whose shutdown wedges holds the whole device pool
  and makes the next query's startup look like an OOM; `run-each.sh` waits for the
  device to drain before each query.

## Results

TPC-H from `s3://sirius-s3-test/datasets/tpch_sf<N>/`, RTX PRO 4500 box, template
defaults, after both fixes above.

**SF1000** — 21 queries (q18 fails in both arms: `MERGE_GROUP_BY` /
`HASH_GROUP_BY` at the OOM retry limit), prefetching cache on (uncapped), best of 2:

| | no compression | spill compression |
|---|---|---|
| total | 185.9 s | **140.4 s (−24%)** |
| q9 (the big spiller) | 66.9 s, 201 GB to disk | 30.3 s, 84 GB to disk |
| q7 | 9.6 s, 11 GB to disk | 7.1 s, 0.7 GB to disk |
| compression | — | 238 GB → 82 GB (2.89×), 46 batches declined |

Compression pays off when spills reach disk. When everything fits in the host
tier (cache off, same suite) it was a wash to +5%: it saves PCIe bytes and host
memory, neither of which was the bottleneck, and costs encode/decode time.

**SF3000** — one process per query, 1 iteration, 20-minute timeout. Reproduce the
tables with `ab-report.py` on `each_sf3000_sf3k_nc_{base,comp}.tsv` and
`each_sf3000_sf3k_cap20_{base,comp}.tsv` under `test/tpch_performance/output/`.

| Cache | Completed (off / on) | Total over queries both arms ran | Notable |
|---|---|---|---|
| off | 19 / 17 | 422.0 s → 412.7 s (−2%), 17 queries | q21 −11%, q2 −20%; most others ±8%. **q3 and q13 fail with compression** (GPU retry limit) where the baseline completes them (312 s, 31 s). |
| capped at 20 GB | 14 / 14 | 417.2 s → 385.5 s (−8%), 15 queries | Spills reach disk here, and compression pays: q14 −27%, q5 −25%, q7 −10%. |

Answers are identical between arms on every query that completed in both (q11
differs only in ORDER BY tie order). Failing in every configuration: q9 (GPU OOM in
`HASH_JOIN`), q10, q18 (thrashes on OOM retries past 20 minutes). Compression
ratios on the heavy spillers: 2.3–4.7× (q3 471 → 121 GB, q9 325 → 114 GB).

The pattern across both scale factors: **compression pays when spills would
otherwise reach disk, is a wash when they fit in the host tier, and under extreme
spill pressure can turn a completion into a GPU retry-limit failure** — through the
retry policy and the compression arena's device over-commit, not slower eviction
(next section).

## Open items

- **The GPU retry policy cannot wait for memory — the most important open problem.**
  SF3000 q3 and q13 complete with the cache off and compression off, but fail at
  the retry limit with compression on (1,838 and 1,518 retries) or with the cache
  on. Not because compressed spills free memory slowly: downgrades in the
  compression runs were fast (~39 GB/s). The retry budget is 100 per original task
  for its whole life, never reset on progress, with a fixed 50 ms sleep
  (`gpu_pipeline_executor.cpp`, sized for ~5 s of SF100 contention) that waits on
  no eviction; a downgrade that frees 0 bytes still lets the task retry. In the
  compression runs everything evictable had already been spilled (1,605
  predicate-only requests in the last 5 s freeing nothing) and the device was
  physically over-committed — the 0.86 pool reserves ~30 GB plus the 3 GiB arena on
  a ~33.7 GB card — giving real `cudaErrorMemoryAllocation` failures (1,313 in q3,
  809 in q13; 0 with compression off) that the defragmenter will not trim for
  (it requires free space >= 10x the request). With the cache on, the host tier is
  wedged by readahead (above), so GPU memory has nowhere to spill. Candidates:
  count a retry only when a full reservation was granted, wait on a
  memory-release event instead of a fixed sleep, reset on query progress with a
  wall-clock backstop.
  **Done (P6):** the arena is now subtracted from the GPU memory space's capacity
  (`sirius_config::carve_compression_arena`), so cuCascade's accounting and the
  physical pool agree, and the defragmenter trims at 1x the request once 8
  physical failures have accumulated since its last trim. Consequence for A/B
  runs: at the same `usage_limit_fraction` the compression arm now has
  `device_pool_bytes` less query pool than the baseline, where before it had the
  same pool plus an over-committed arena. Not yet re-measured at SF3000; the
  retry-policy candidates above remain open.
- **Lazy per-thread stream pools in the compressed decode path.**
  `simpatico::thread_device_stream_pool` (`src/util/stream_pool.cpp`) creates 4
  CUDA streams the first time each thread decodes. Deep in a memory-starved query
  `cudaStreamCreateWithFlags` fails, the pool throws `std::runtime_error` (not an
  OOM, so the executor cannot retry) without the CUDA error code, and the query
  fails: SF3000 q3, capped cache, compression on (`stream_pool: failed to create 4
  streams on device 0`). Create the pools eagerly, or throw an OOM so it retries,
  or fall back to the caller's stream.
  **Done (P7):** a creation failure is now `rmm::out_of_memory` carrying the CUDA
  error (so the executor reschedules instead of failing the query); the GPU
  pipeline workers and the downgrade workers/processing thread create their pool
  at thread start whenever a compression feature is configured
  (`prewarm_compression_streams`); and the converters' separate thread-local
  `column_pool` now shares `thread_device_stream_pool`, so a thread holds one set
  of four streams per device instead of two. The pool stays per thread: every
  column-parallel call ends with a `sync_all()` over its streams, which would wait
  on other threads' work if they were shared. pin_table's DuckDB threads are not
  prewarmed; their lazy failure is an OOM that the pin's existing per-chunk
  fallback turns into an uncompressed pin.
- **Compression strictly optional, decompression retryable — done.** Any
  compress-side failure (arena or query-pool OOM, stream creation, encode
  exceptions of any type, plan/explore failures, output compression, the
  in-place device tier) now ends in an uncompressed spill or publication; the
  exception is a failure after `spill_release_columns_early` (default off)
  released the source. Spills skip compression up front when the arena is ≥ 90%
  allocated or within 1 s of a physical device OOM; skips and fallbacks are
  counted (`[compression_fallback]` in the downgrade monitor's debug log). Decode
  converters rethrow transient device-resource failures as `rmm::out_of_memory`
  so the executor retries them. The compressed-scan decode inside an operator is
  not wrapped (its stream creation is an OOM; other CUDA errors there propagate
  as before). The arena-nearly-full gate has no unit test: installing an arena is
  process-wide and permanent.
- **Prefetching cache footprint at SF3000** — test the readahead hypothesis above
  (`REST_MAX_CONCURRENT_SCANS=16`; at SF1000 16 and 48 performed the same) before
  using the cache at this scale.
- **q18 at SF1000; q9, q10, q18 at SF3000** — fail in every configuration on a
  32 GB card (GPU OOM at the retry limit, or thrashing past the timeout); engine
  limits rather than tuning, but the first candidates if the aggregation/join
  memory estimates are revisited.
- **S3 throughput.** At ~3 MB column-chunk GETs the box measures ~8 GB/s raw from
  S3; 8 MB GETs reach ~10.8 GB/s. The REST worker now coalesces runs of cache
  fills, populate-on-read fills and staged reads into one GET (up to 16 MiB) and
  bridges gaps up to `rest.merge_max_gap`, which targets exactly this; it has not
  been re-measured here yet.
- **Cache accounting.** The per-query cache summary reports `reads=0 hits=0` on the
  REST path, and nothing reports the cache's resident bytes, so the cache/spill
  split of the host tier cannot be observed directly.
- **Upstream the IO work together with its fixes.** The fused object-store reads
  (a5bfdc55) and the multi-file read overlap (2e9f4a0b) exist only on this branch;
  if they go upstream, 456672d0 and c29bf4e4 respectively must go with them.
- **cuCascade** — get `try_release_table` upstream so the gitlink can leave the fork.
