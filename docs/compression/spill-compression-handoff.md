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
| `device_pool_bytes` | The encoder's arena. **Installed only from the config file at startup**: `SET spill_compression = true` at session start flips the flag but leaves the encoder allocating from the query pool — the documented pathology (compression latching on/off, a downgrade-request storm, the query killed). Sizing is a cliff, not a gradient: 1 GiB was too small for concurrent encodes on a 96 GB card and the query failed outright; 4 GiB worked there. 3 GiB is used on the 32 GB card and fills on SF3000 q18. |
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
  `QUERY_TIMEOUT_S`, runs `memory-watchdog.sh` (kills the harness, not the box,
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

**SF3000** — still running at the time of writing; compare the arms with
`ab-report.py` on `each_sf3000_sf3k_nc_{base,comp}.tsv` (cache off) and
`each_sf3000_sf3k_cap20_{base,comp}.tsv` (cache capped at 20 GB). Known so far:
q3 spills 280–730 GB to disk depending on how much host tier the cache leaves
(312 s cache off, 610 s uncapped); q9 fails on GPU memory (`HASH_JOIN`) even with
the cache off; the uncapped cache failed q9/q10 on exhausted REST staging.

## Open items

- **SF3000 A/B** — finish and record (above). Expect the disk-bound queries (q3,
  q9 if it can run) to be where compression matters.
- **q18 at SF1000, q9 at SF3000** — GPU OOM at the retry limit on a 32 GB card,
  with or without compression; engine limits rather than tuning, but the first
  candidates if the aggregation/join memory estimates are revisited.
- **S3 throughput.** At ~3 MB column-chunk GETs the box measures ~8 GB/s raw from
  S3; 8 MB GETs reach ~10.8 GB/s. Fusion only joins exactly adjacent chunks, and
  `rest.merge_max_gap` applies only to the prefetching cache, so the demand path
  cannot make bigger GETs today. With the cache on, fetches are host-block sized
  (1 MiB) and bound by request latency × concurrency.
- **Cache accounting.** The per-query cache summary reports `reads=0 hits=0` on the
  REST path, and nothing reports the cache's resident bytes, so the cache/spill
  split of the host tier cannot be observed directly.
- **Upstream the IO work together with its fixes.** The fused object-store reads
  (a5bfdc55) and the multi-file read overlap (2e9f4a0b) exist only on this branch;
  if they go upstream, 456672d0 and c29bf4e4 respectively must go with them.
- **cuCascade** — get `try_release_table` upstream so the gitlink can leave the fork.
