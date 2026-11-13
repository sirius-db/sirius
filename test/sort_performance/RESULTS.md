# Sort merge benchmark: `windowed` against `one_shot`

A summary of the measurements taken on 2026-11-13 on the workstation `felipe-TRX50-AERO-D`
(192.168.0.248) with `sort_merge_bench.py`. No number here comes from the development machine.
The per-run result files (configs, SQL, CSVs, log tails) are not committed; rerun the harness to
reproduce them ([README.md](README.md)).

## Summary

- **At 5× a whole GPU, `windowed` finished all 10 runs and `one_shot` none of its 10.** Each run
  sorts 470 GiB on one GPU with a 94 GiB limit.
- `one_shot` failed in three ways: 4 runs were killed at the timeout (4× the windowed time of the
  same cell), 4 stopped on a cuDF row-count limit, and 2 ran out of GPU memory after 100 retries.
- `windowed` took 404–999 s per run (0.47–1.16 GiB/s). Spill compression made every dataset
  faster, by 22–56 %.
- **On data that fits the GPU** (32 GiB under an 80 GiB limit), `windowed` is 22–29 % slower than
  `one_shot` where `one_shot` finishes; `one_shot` ran out of memory on the other three datasets.
- **Recommendation:** deprecate `one_shot` now and remove it once the conditions in
  [Recommendation](#recommendation-one_shot-deprecation) hold; the speed gap on data that fits
  the GPU is the one not met yet.
- Limits: one repeat per cell at 470 GiB, order checked at 40 GiB (not 470 GiB), and a GPU limit
  so close to the card's size that every windowed run retried tasks after real device allocation
  failures (1,362–19,989 retries per run).

## Setup (whole-GPU matrix)

| | |
|---|---|
| Host | x86_64, AMD Ryzen Threadripper 7960X (48 threads), 125 GiB RAM, Ubuntu 24.04 |
| GPU | One NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB, driver 590.48.01, pinned with `--device 0` |
| Disk | `/mnt/disk_2` (NVMe), 1,018 GiB free before the run |
| Code | Commit `8290517a` plus the changes to `src/sirius_context.cpp` and `src/op/sirius_physical_discard_sink.hpp` that were committed later in `a9aaacb5`. The [follow-up measurements](#follow-up-measurements-after-the-integration) ran on the integrated code. Release build, `CUDAARCHS=120-real`. |
| GPU limit | `sirius.memory.gpu.usage_limit_bytes = 100931731456` (94 GiB; 95 and 96 GiB are rejected as above the device's capacity) |
| Host cap | `sirius.memory.host.capacity_bytes = 103079215104` (96 GiB) |
| Data | `--scale 5`: 470 GiB per run, 8,000,000 rows per batch, seed 42 |
| Compressed arm | `SET spill_compression = true`, `sirius.compression.spill_min_batch_bytes = 1048576` |
| Run order | One cell at a time: `windowed`, then `one_shot` of the same dataset and compression. `one_shot` timeout 4× the windowed time (at least 1,800 s). 20 runs, 5 h 32 min. |
| Validation | Row count returned by the `COPY` and counted by the discard sink. |

Each run is one statement in a fresh process:
`COPY (SELECT * FROM sirius_generate(...) ORDER BY <keys>) TO '/dev/null' (FORMAT sirius_discard)`.

| Dataset | Rows | Bytes per row |
|---|---|---|
| `uniform-1key` | 31,544,000,000 | 16 |
| `zipf-1key` | 31,544,000,000 | 16 |
| `sequential` | 31,544,000,000 | 16 |
| `few_distinct-strings` | 22,936,000,000 | 22 |
| `normal-multikey` | 18,024,000,000 | 28 |

## Results: 470 GiB per run, 94 GiB limit

| Dataset | Compression | windowed | Wall s | GiB/s | one_shot | Wall s until it ended |
|---|---|---|---|---|---|---|
| uniform-1key | off | ok | 837 | 0.56 | timeout | 3,350 |
| uniform-1key | on | ok | 649 | 0.72 | timeout | 2,598 |
| zipf-1key | off | ok | 797 | 0.59 | error (cuDF row limit) | 454 |
| zipf-1key | on | ok | 498 | 0.94 | error (cuDF row limit) | 374 |
| sequential | off | ok | 999 | 0.47 | oom (retry limit) | 383 |
| sequential | on | ok | 436 | 1.08 | oom (retry limit) | 156 |
| few_distinct-strings | off | ok | 580 | 0.81 | error (cuDF row limit) | 363 |
| few_distinct-strings | on | ok | 404 | 1.16 | error (cuDF row limit) | 342 |
| normal-multikey | off | ok | 836 | 0.56 | timeout | 3,342 |
| normal-multikey | on | ok | 490 | 0.96 | timeout | 1,958 |

**How `one_shot` fails.**

| Outcome | Datasets | What happened |
|---|---|---|
| Timeout | `uniform-1key`, `normal-multikey` | Still running at 4× the windowed time, after moving 3.2–3.8 TB to host memory and 1.6–3.6 TB to disk. |
| cuDF row limit | `zipf-1key`, `few_distinct-strings` | `merge.cu: Total number of merged rows exceeds row limit`: a partition holds more rows than one cuDF merge can produce. More memory would not fix it. |
| Out of memory | `sequential` | `GPU pipeline task exceeded maximum retry limit (100)` while preparing the partition. |

**Compression (windowed).**

| Dataset | Wall s, off → on | Change | Spill to disk GB, off → on |
|---|---|---|---|
| `uniform-1key` | 837 → 649 | −22 % | 671 → 455 |
| `zipf-1key` | 797 → 498 | −37 % | 623 → 207 |
| `sequential` | 999 → 436 | −56 % | 977 → 343 |
| `few_distinct-strings` | 580 → 404 | −30 % | 547 → 0 |
| `normal-multikey` | 836 → 490 | −41 % | 489 → 192 |

Compression was really applied: 25,011–138,867 runs were read from compressed batches.

**Windowed behaviour.** Every partition took the windowed path with no fallback conversions.
Every run needed intermediate fan-in passes, so the merge wrote 1.99–2.69 rows per input row.
Uncompressed runs moved 489–977 GB to disk for 505 GB of input. The smallest round emitted
14–58 rows on three of the five datasets; the mean round is 0.27–1.24 million rows.

**Peak GPU** (`nvidia-smi` `memory.used`) is not a measure of what the sort used: Sirius reserves
its pool at startup, so the column starts at about the limit plus 0.6 GiB. At this limit it sat
at 97,250 MiB, the card's size, in every run.

## Follow-up measurements after the integration

Same workstation and GPU, on the integrated code, spill compression off.

**Data that fits the GPU: 32 GiB under an 80 GiB limit, three repeats** (wall seconds):

| Dataset | one_shot | windowed | auto | windowed ÷ one_shot |
|---|---|---|---|---|
| `uniform-1key` | 2.70 / 2.68 / 2.69 | 3.48 / 3.48 / 3.49 | 3.48 / 3.48 / 3.47 | 1.29 |
| `zipf-1key` | 2.24 / 2.25 / 2.24 | 2.73 / 2.74 / 2.74 | 2.75 / 2.73 / 2.73 | 1.22 |
| `sequential` | oom ×3 | 1.42 / 1.42 / 1.42 | 1.42 / 1.42 / 1.43 | — |
| `few_distinct-strings` | oom ×3 | 14.17 / 14.20 / 14.18 | 14.17 / 14.16 / 14.16 | — |
| `normal-multikey` | oom ×3 | 6.72 / 6.70 / 6.68 | 6.70 / 6.73 / 6.72 | — |

`auto` took the windowed route almost everywhere and matched its times.

**Twice the GPU: 160 GiB under an 80 GiB limit, one repeat.** `windowed` and `auto` finished
`uniform-1key` and `zipf-1key` in about a minute (54–67 s); `one_shot` ran out of memory on both
(449 s and 771 s).

**Order check at a size that reaches the disk:** 40 GiB, 4 GiB GPU limit, 4 GiB host cap,
`--validate`: 0 order violations in all six windowed runs (`uniform-1key`, `zipf-1key`,
`normal-multikey`, compression off and on), with intermediate passes and the disk tier in use.
At this small limit compression made the runs 1.2–5.0× slower, the opposite of the whole-GPU
result; not diagnosed.

## Recommendation: `one_shot` deprecation

**Deprecate `one_shot` now** (the setting stays and logs a warning naming the release that
removes it), **and remove it once the conditions below hold.**

Why now: at 5× a whole GPU `one_shot` completed 0 of 10 runs; four failures are a cuDF row limit
no memory setting changes; failing runs wrote terabytes to disk before failing; and the default
`auto` already uses the windowed path under memory pressure.

Before removal:

1. **`windowed` within 20 % of `one_shot` wherever `one_shot` completes.** Not met: 22–29 %
   slower at 32 GiB. Close the gap, or let `auto` use an in-memory merge for data that fits.
2. **`auto` measured.** Done at two sizes; it follows the windowed route and its times.
3. **Order check at scale.** Met at 40 GiB with the disk tier and intermediate passes, not at
   470 GiB.

## Open problems found

1. **A limit near the device size causes thousands of device OOM retries** on every windowed run
   (1,362–19,989 per run at 94 GiB; 0–5 at 24 GiB). They cost time; how much is not measured.
2. **Device memory exceeds the limit by 4–10 GiB when it can** (seen at a 24 GiB limit). Not
   diagnosed: pool fragmentation or allocations outside the reservation accounting.
3. **`one_shot` hits the cuDF row limit** on large partitions (one-shot path of
   `src/op/sirius_physical_merge_sort.cpp`).
4. **The default host capacity (0.9 of RAM) can fill the machine's RAM**; the benchmark needs a
   host cap in its config.
