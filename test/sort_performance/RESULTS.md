# Sort merge benchmark: `windowed` against `one_shot` at 5× a whole GPU

Measured on 2026-11-13 on the workstation `felipe-TRX50-AERO-D` (192.168.0.248) through
`.ai-helper/workstation.sh`. No number here comes from the development machine.

## Summary

- **`windowed` finished all 10 runs. `one_shot` finished none of its 10.** Each run sorts 470 GiB
  on one GPU with a 94 GiB limit.
- `one_shot` failed in three ways: 4 runs were killed at the timeout (4× the windowed time of the
  same cell), 4 stopped on a cuDF row-count limit, and 2 ran out of GPU memory after 100 retries.
- `windowed` took 404–999 s per run (0.47–1.16 GiB/s). Spill compression made every dataset
  faster, by 22–56 %.
- **Recommendation:** deprecate `one_shot` now and remove it one release later, once the three
  conditions in [Recommendation](#recommendation-one_shot-deprecation) hold. The main one is
  speed on inputs small enough for `one_shot` to finish, which this run does not measure.
- Three things limit how far these numbers go: one repeat per cell, no order check at this scale
  (the row count is checked), and a GPU limit so close to the card's size that every windowed run
  retried tasks after real device allocation failures (1,362–19,989 retries per run).

| Dataset | Compression | windowed | Wall s | GiB/s | one_shot | Wall s until it ended | one_shot timeout s |
|---|---|---|---|---|---|---|---|
| uniform-1key | off | ok | 837 | 0.56 | timeout | 3,350 | 3,348 |
| uniform-1key | on | ok | 649 | 0.72 | timeout | 2,598 | 2,596 |
| zipf-1key | off | ok | 797 | 0.59 | error (cudf row limit) | 454 | 3,184 |
| zipf-1key | on | ok | 498 | 0.94 | error (cudf row limit) | 374 | 1,992 |
| sequential | off | ok | 999 | 0.47 | oom (retry limit) | 383 | 3,996 |
| sequential | on | ok | 436 | 1.08 | oom (retry limit) | 156 | 1,800 |
| few_distinct-strings | off | ok | 580 | 0.81 | error (cudf row limit) | 363 | 2,320 |
| few_distinct-strings | on | ok | 404 | 1.16 | error (cudf row limit) | 342 | 1,800 |
| normal-multikey | off | ok | 836 | 0.56 | timeout | 3,342 | 3,340 |
| normal-multikey | on | ok | 490 | 0.96 | timeout | 1,958 | 1,956 |

## Setup

| | |
|---|---|
| Host | `felipe-TRX50-AERO-D`, Ubuntu 24.04, x86_64, AMD Ryzen Threadripper 7960X (48 threads), 125 GiB RAM |
| GPU | One NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB, driver 590.48.01, pinned with `--device 0` |
| Disk | `/mnt/disk_2` (NVMe), 1,018 GiB free before the run, spill under `/mnt/disk_2/sirius_spill` |
| Code | Commit `8290517a` plus the uncommitted changes to `src/sirius_context.cpp` and `src/op/sirius_physical_discard_sink.hpp` (diff and checksums in `results/full/code_state.txt`). Release build, `CUDAARCHS=120-real`. |
| GPU limit | `sirius.memory.gpu.usage_limit_bytes = 100931731456` (94 GiB) |
| Host cap | `sirius.memory.host.capacity_bytes = 103079215104` (96 GiB), from `--base-config` |
| Data | `--scale 5`: 470.0 GiB per run, 8,000,000 rows per batch, seed 42 |
| Compressed arm | `SET spill_compression = true` and `sirius.compression.spill_min_batch_bytes = 1048576` |
| Repeats | 1 per cell |
| Validation | Row count returned by the `COPY` and counted by the discard sink. `--validate` (order check) was not used at this scale; the smoke run checked order at 12 GiB. |

**GPU limit.** The requested 96 GiB (103079215104) does not start: Sirius rejects it with
`sirius.memory.gpu.usage_limit_bytes exceeds the capacity of GPU 0`. So do 97,887 MiB and 95 GiB.
94 GiB starts and is what every run used. Values between 94 and 95 GiB were not tried
(`results/full/whole-gpu-5x/probe/`).

**Data size per dataset.**

| Dataset | Batches | Rows | Bytes per row | Size |
|---|---|---|---|---|
| `uniform-1key` | 3,943 | 31,544,000,000 | 16 | 470.0 GiB |
| `zipf-1key` | 3,943 | 31,544,000,000 | 16 | 470.0 GiB |
| `sequential` | 3,943 | 31,544,000,000 | 16 | 470.0 GiB |
| `few_distinct-strings` | 2,867 | 22,936,000,000 | 22 | 469.9 GiB |
| `normal-multikey` | 2,253 | 18,024,000,000 | 28 | 470.0 GiB |

**Run order.** One cell at a time, as a pair: the `windowed` run, then the `one_shot` run of the
same dataset and compression setting. Each run is one harness invocation
(`results/full/whole-gpu-5x/driver.sh`). The `one_shot` timeout is 4× the windowed wall time of
the cell, at least 1,800 s. The windowed timeout was 10,800 s and was never reached. Before each
run the driver removed leftover spill directories and logged the free disk space
(`driver_console.txt`). The 20 runs took 5 h 32 min. No dataset was dropped or shrunk.

**Each run** is one statement in a fresh process:
`COPY (SELECT * FROM sirius_generate(...) ORDER BY <keys>) TO '/dev/null' (FORMAT sirius_discard)`.
The exact SQL and config of every run are in `results/full/whole-gpu-5x/per_run/<run>/`.

## Results per dataset

Columns: "Spill to host" and "Spill to disk" are the bytes the downgrade executor moved to each
tier, summed over the run, so a batch moved twice counts twice. "Peak GPU" is the whole device's
`memory.used` (see [Peak GPU](#what-peak-gpu-measures)). Sizes in GB are 10⁹ bytes. A failed
`one_shot` run has no merge statistics line, so the merge tables hold the windowed runs only.
"Rows emitted ÷ input rows" counts every row a merge round wrote, including intermediate passes.

### `uniform-1key` — 31,544,000,000 rows, 470.0 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 837 | 97,250 | 748 | 671 | 79 | 17,479 |
| one_shot | off | timeout | 3,350 | 97,250 | 3,685 | 3,585 | 2,034 | 19,475 |
| windowed | on | ok | 649 | 97,250 | 937 | 455 | 75 | 15,311 |
| one_shot | on | timeout | 2,598 | 97,250 | 3,640 | 2,542 | 38,797 | 20,210 |

Merge statistics of the windowed runs:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 17 | 9,518 | 6,242 | 5,242 | 117,445 | 646,011 / 18 / 1,027,839 | 2.41 | 276 / 768 / 0 | 0 | 20,817,204,762 | 2,829 |
| on | 17 | 5,732 | 3,729 | 1,439 | 61,339 | 1,236,908 / 15 / 2,516,119 | 2.41 | 0 / 0 / 1,032 | 110,677 | 23,547,399,379 | 236 |

### `zipf-1key` — 31,544,000,000 rows, 470.0 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 797 | 97,250 | 669 | 623 | 1,109 | 17,766 |
| one_shot | off | error (cudf row limit) | 454 | 97,250 | 124 | 246 | 2,263 | 20,089 |
| windowed | on | ok | 498 | 97,250 | 582 | 207 | 1,398 | 19,989 |
| one_shot | on | error (cudf row limit) | 374 | 97,250 | 102 | 132 | 2,118 | 21,222 |

Merge statistics of the windowed runs:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 16 | 4,850 | 2,934 | 1,181 | 61,036 | 1,243,059 / 14 / 1,835,789 | 2.41 | 234 / 719 / 0 | 0 | 17,317,814,851 | 118 |
| on | 16 | 6,542 | 3,961 | 1,046 | 77,534 | 978,556 / 41 / 1,835,599 | 2.41 | 0 / 0 / 900 | 101,923 | 15,928,217,410 | 21 |

### `sequential` — 31,544,000,000 rows, 470.0 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 999 | 97,250 | 1,082 | 977 | 117 | 2,252 |
| one_shot | off | oom (retry limit) | 383 | 97,250 | 403 | 539 | 7,053 | 2,378 |
| windowed | on | ok | 436 | 97,250 | 1,167 | 343 | 77 | 1,362 |
| one_shot | on | oom (retry limit) | 156 | 97,250 | 415 | 308 | 2,838 | 1,320 |

Merge statistics of the windowed runs:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 17 | 13,527 | 8,516 | 79 | 175,125 | 485,064 / 8,070 / 9,411,765 | 2.69 | 252 / 966 / 0 | 0 | 407,676,992 | 0 |
| on | 17 | 19,585 | 12,310 | 79 | 313,009 | 271,388 / 2,494 / 9,411,765 | 2.69 | 2 / 2 / 1,196 | 25,011 | 283,960,070 | 0 |

### `few_distinct-strings` — 22,936,000,000 rows, 469.9 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 580 | 97,246 | 652 | 547 | 68 | 5,592 |
| one_shot | off | error (cudf row limit) | 363 | 97,246 | 339 | 421 | 209 | 3,740 |
| windowed | on | ok | 404 | 97,250 | 892 | 0 | 62 | 12,513 |
| one_shot | on | error (cudf row limit) | 342 | 97,250 | 670 | 0 | 496 | 8,755 |

Merge statistics of the windowed runs:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 17 | 6,586 | 4,199 | 3,182 | 79,288 | 662,357 / 476 / 1,157,296 | 2.29 | 211 / 478 / 0 | 0 | 788,623,724 | 1,711 |
| on | 17 | 6,817 | 4,399 | 4,305 | 83,290 | 630,531 / 225 / 1,221,832 | 2.29 | 2 / 0 / 791 | 74,305 | 729,258,243 | 2,584 |

### `normal-multikey` — 18,024,000,000 rows, 470.0 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 836 | 97,250 | 467 | 489 | 1,802 | 10,206 |
| one_shot | off | timeout | 3,342 | 97,250 | 3,228 | 3,166 | 3,147 | 6,915 |
| windowed | on | ok | 490 | 97,250 | 833 | 192 | 49 | 7,329 |
| one_shot | on | timeout | 1,958 | 97,250 | 3,761 | 1,602 | 43,481 | 7,606 |

Merge statistics of the windowed runs:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 17 | 4,363 | 2,275 | 612 | 54,502 | 659,491 / 58 / 995,483 | 1.99 | 159 / 554 / 0 | 0 | 11,621,813,316 | 0 |
| on | 17 | 8,380 | 4,333 | 616 | 103,094 | 348,649 / 14 / 775,545 | 1.99 | 0 / 0 / 728 | 138,867 | 16,525,419,050 | 2 |
## Observations

### Where `one_shot` fails

| Outcome | Runs | What happened |
|---|---|---|
| Timeout | `uniform-1key` and `normal-multikey`, both compression settings | Still running at 4× the windowed time. By then each had moved 3.2–3.8 TB to host memory and 1.6–3.6 TB to disk, up to 7 times the input, and had retried 6,915–20,210 tasks. Whether they would finish with more time is not known. |
| cuDF row limit | `zipf-1key` and `few_distinct-strings`, both compression settings | `CUDF failure at: .../merge/merge.cu:628: Total number of merged rows exceeds row limit`, after 342–454 s. `one_shot` concatenates a partition's runs and merges them in one call, and a partition here holds more rows than one cuDF column can. More memory would not fix this. |
| Out of memory | `sequential`, both compression settings | `GPU pipeline task exceeded maximum retry limit (100) ... OOM while preparing batches for processing ... not enough capacity to allocate memory`, after 156–383 s. |

The error text of each run is in `results.csv` (`error`) and in the run's `row.json`.

The cuDF failures are reported to the user as
`CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: ...`.
The fallback message comes first only because the benchmark's source has no CPU version. A query
on a real table would fall back to the CPU here.

### `windowed`

- All 10 runs returned the expected row count, and the discard sink counted the same number.
  Every partition took the windowed path (`one_shot_partitions` 0, `fallback_conversions` 0,
  `window_growths` 0).
- **Throughput** is 0.47–0.81 GiB/s uncompressed and 0.72–1.16 GiB/s compressed.
- **Uniform data, where `one_shot` is at its best, has no A/B number at this size**: `one_shot`
  did not finish in 4× the windowed time. The only sizes where both finished are in the smoke run
  (12 GiB of data, 4 GiB limit, one repeat, `results/smoke/NOTES.md`). There `one_shot` was
  faster: 3.2 s against 5.5 s on `uniform-1key` and 2.3 s against 5.5 s on `normal-multikey`,
  uncompressed. Those runs last seconds and are an indication only.
- **Fan-in passes.** Every run needed intermediate passes: 2,275–12,310 of the 4,363–19,585
  merge tasks were intermediate, and the merge wrote 1.99–2.69 rows per input row. So each row
  was merged about twice or more before it reached the sink. `sequential` is the worst case (2.69)
  and the slowest uncompressed run (999 s).
- **Disk.** Uncompressed runs moved 489–977 GB to disk for 505 GB of input, because intermediate
  passes spill their output again. The most the disk held at one time was not sampled.
- **Small rounds.** The smallest round emitted 14–58 rows on three of the five datasets (476 and
  225 on `few_distinct-strings`, 8,070 and 2,494 on `sequential`). The mean round is
  0.27–1.24 million rows.
- **OOM retries.** Every windowed run retried tasks: 1,362–19,989 per run. See the next section.

### OOM retries at this GPU limit

The limit of 94 GiB leaves the device about 0.6 GiB: `memory.used` was 97,246–97,250 MiB of
97,887 MiB in every run. The retries are real device allocation failures
(`cudaErrorMemoryAllocation out of memory` from RMM), not the configured limit being reached.
In the first run's log the first one came 2 s after the start, for a 93 MiB allocation, with
Sirius's own count at 94,176 MiB against a limit of 96,256 MiB. That log was read while the run
was in progress; the harness keeps only the last 400 lines.

The retries did not stop any windowed run. They do cost time, and how much is not measured. At
the 24 GiB limit, where the device had room, the same datasets ran with 0–5 retries
([earlier runs](#earlier-runs-24-gib-limit-10-data-windowed-only)).

Windowed inputs shrank their window for a retry 2,829 times on `uniform-1key` uncompressed, and
between 0 and 2,584 times on the other runs (`shrinks`).

### Compression

| Dataset | Wall s, off → on | Change | Spill to disk GB, off → on |
|---|---|---|---|
| `uniform-1key` | 837 → 649 | −22 % | 671 → 455 |
| `zipf-1key` | 797 → 498 | −37 % | 623 → 207 |
| `sequential` | 999 → 436 | −56 % | 977 → 343 |
| `few_distinct-strings` | 580 → 404 | −30 % | 547 → 0 |
| `normal-multikey` | 836 → 490 | −41 % | 489 → 192 |

- Compression was really applied: 25,011–138,867 runs were read from compressed batches, and
  728–1,196 GB of refill came from compressed runs.
- With compression `few_distinct-strings` never reached the disk: its 18 distinct values compress
  well enough for the host tier to hold everything.
- The runs that saved the most disk bytes also saved the most time (`sequential`), except
  `few_distinct-strings`. The benchmark does not separate disk time from compression time.
- Compression did not help `one_shot`: the same cells failed in the same way.

## What "Peak GPU" measures

`peak_gpu_mib` is the highest `memory.used` that `nvidia-smi` reported for the whole device
during the run. It is not Sirius's count of its own allocations.

- **Sirius reserves the limit when it starts.** A probe that sorted 1.9 GiB under the 94 GiB
  limit showed 96,862 MiB used on the device, while Sirius's own peak was 10.0 GB
  (`gpu_pool ... peak=10030801136` in the probe's log). The smoke run saw the same: 4,682 MiB
  with a 4,096 MiB limit and nothing running. So the column starts at the limit plus about
  0.6 GiB (the CUDA context and whatever else the process holds) and says little about how much
  the sort used.
- **Sirius's own peak was exactly the limit** in all six earlier runs
  (`gpu_pool ... QueryEnd ... peak=25769803776`) and in all ten windowed runs here
  (`peak=100931731456`). By its own accounting it did not exceed the limit.
- **The device still went 4–10 GiB above the limit plus 0.6 GiB** in the earlier runs
  (29,454–35,512 MiB under a 24,576 MiB limit). That memory was taken from the device on top of
  the reserved pool. The cause is not diagnosed here. The two candidates are fragmentation
  inside the CUDA memory pool (a free block of the right size is missing, so the pool grows) and
  allocations that the reservation accounting does not count.
- **In the whole-GPU runs the column cannot go higher**: 97,250 MiB is the card. The growth that
  had room at 24 GiB fails here, which is where the OOM retries come from.

A peak that can be compared with the limit has to come from Sirius's accounting, and that
accounting currently reports the limit itself.

## Earlier runs: 24 GiB limit, 10× data, windowed only

These six runs are **not part of the A/B**. They were measured before the owner changed the
sizing: GPU limit 25,769,803,776 bytes (24 GiB), `--scale 10` (240 GiB per run), the same 96 GiB
host cap, same code, same GPU, one repeat, windowed only. The run was stopped during the seventh
(`few_distinct-strings` uncompressed), which has no result. Files: `results/full/limit-24g-10x/`.

#### `uniform-1key` — 16,104,000,000 rows, 240.0 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 423 | 30,972 | 427 | 323 | 111 | 0 |
| windowed | on | ok | 306 | 35,512 | 498 | 155 | 108 | 0 |

Merge statistics:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 36 | 8,249 | 4,464 | 1,152 | 229,677 | 139,216 / 10 / 204,704 | 1.99 | 168 / 323 / 0 | 0 | 23,251,706,760 | 0 |
| on | 36 | 8,344 | 4,464 | 1,152 | 402,460 | 79,448 / 1 / 150,804 | 1.99 | 0 / 0 / 484 | 320,649 | 24,431,513,335 | 0 |

#### `zipf-1key` — 16,104,000,000 rows, 240.0 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 465 | 29,660 | 505 | 351 | 127 | 0 |
| windowed | on | ok | 204 | 29,750 | 521 | 72 | 104 | 0 |

Merge statistics:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 31 | 9,482 | 5,660 | 1,259 | 558,157 | 68,345 / 1 / 104,776 | 2.37 | 222 / 350 / 0 | 0 | 15,220,529,109 | 0 |
| on | 32 | 8,879 | 5,003 | 1,369 | 489,693 | 70,337 / 3 / 138,061 | 2.14 | 0 / 0 / 520 | 219,253 | 14,579,594,622 | 0 |

#### `normal-multikey` — 9,200,000,000 rows, 239.9 GiB

| Mode | Compression | Status | Wall s | Peak GPU MiB | Spill to host GB | Spill to disk GB | Downgrade requests | OOM retries |
|---|---|---|---|---|---|---|---|---|
| windowed | off | ok | 615 | 29,454 | 509 | 337 | 121 | 0 |
| windowed | on | ok | 500 | 29,512 | 517 | 71 | 101 | 5 |

Merge statistics:

| Compression | Partitions | Merge tasks | Intermediate passes | Mergers built | Rounds | Rows per round (mean / min / max) | Rows emitted ÷ input rows | Refill GB (host / disk / compressed) | Runs read compressed | Rows compacted | Shrinks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 37 | 9,126 | 5,329 | 814 | 326,955 | 66,570 / 3 / 94,258 | 2.37 | 203 / 288 / 0 | 0 | 15,548,845,116 | 0 |
| on | 38 | 8,220 | 4,337 | 732 | 274,947 | 68,528 / 5 / 140,938 | 2.05 | 0 / 0 / 442 | 274,718 | 14,491,540,950 | 5 |
Against the whole-GPU runs: OOM retries are 0–5 here against 1,362–19,989, and throughput is
0.39–1.18 GiB/s here against 0.47–1.16 GiB/s, on half the data.

### A run without the host cap

The first attempt used the default host capacity (0.9 of RAM) with the 24 GiB limit. The process
grew to 113 GiB, the machine had 4.6 GiB available and was swapping, and the first run
(`uniform-1key` windowed uncompressed) had not finished after 12 minutes when it was stopped.
With the 96 GiB cap the same run took 423 s. Files: `results/full/attempt1-default-host-capacity/`.

## Recommendation: `one_shot` deprecation

**Deprecate `one_shot` now. Remove it one release later, when the conditions below hold.**

Why now:

- At 5× a whole GPU `one_shot` completed 0 of 10 runs and `windowed` 10 of 10.
- Four of the `one_shot` failures are a cuDF row limit that no memory setting changes.
- A failing `one_shot` run is expensive before it fails: the four timed-out runs each wrote
  1.6–3.6 TB to the disk.
- The default is already `auto`, which uses the windowed path under memory pressure or when an
  input is not on the GPU. `one_shot` is reached on large data only when someone sets it.

Deprecating means: the setting stays, and choosing `one_shot` logs a warning that names the
release in which it goes away.

What must be true before it is removed:

1. **`windowed` is within 20 % of `one_shot` wherever `one_shot` completes.** That is data that
   fits the GPU or is a small multiple of it. Measure it with at least 3 repeats, on runs of
   minutes, with a limit that leaves the device headroom. Today the only evidence is the smoke
   run, where `windowed` took 1.7–4 times as long on runs of a few seconds. If the gap is real,
   close it or keep an in-memory merge inside `auto`; do not keep `one_shot` as a user setting
   for it.
2. **`auto` is measured.** This benchmark forced each mode. `auto` is what users run, and it
   decides per partition. It needs the same matrix, or at least `uniform-1key` and
   `zipf-1key`, at one size that fits the GPU and one that does not.
3. **The windowed path has an order check at scale.** Order was validated at 12 GiB only. Run
   one dataset with `--validate` at a size that reaches the disk tier and intermediate passes.

Not blocking, but worth doing before the removal: understand the OOM retries at a limit close to
the device size (they cost time on every windowed run here), and the rounds of 14–58 rows.

## Problems found

No source was changed. Each item names the command, the output and where to look.

1. **A 96 GiB or 95 GiB limit is rejected on a 97,887 MiB card.**
   Command: `results/full/whole-gpu-5x/probe/probe.sh`. Output:
   `sirius.memory.gpu.usage_limit_bytes exceeds the capacity of GPU 0`. Code: the capacity check
   in `parsed_sirius_config::resolve`, `src/sirius_config.cpp`. The capacity it compares with is
   below what `nvidia-smi` reports as total; the exact value was not read.
2. **A limit near the device size causes thousands of device OOM retries.** Runs: every windowed
   run in `whole-gpu-5x/results.csv`, column `oom_retries`. See
   [OOM retries](#oom-retries-at-this-gpu-limit). Code to look at: how the GPU pool is created
   and whether the limit should leave headroom (`cucascade/src/memory/memory_space.cpp`, the
   `gpu_memory_space` constructor).
3. **Device memory exceeds the limit by 4–10 GiB when it can** (24 GiB runs). See
   [Peak GPU](#what-peak-gpu-measures). Not diagnosed.
4. **`one_shot` hits the cuDF row limit.** Runs: `zipf-1key__one_shot__*` and
   `few_distinct-strings__one_shot__*`. Error: `merge.cu:628: Total number of merged rows exceeds
   row limit`. Code: the one-shot path of `src/op/sirius_physical_merge_sort.cpp`.
5. **The default host capacity can fill the machine's RAM.** See
   [A run without the host cap](#a-run-without-the-host-cap). The default is 0.9 of RAM.
6. **`DISCARD_SINK` logs a warning per output batch**:
   `gpu_pipeline_task: operator 'DISCARD_SINK' (id=5) output batch 0 column count mismatch: got 1, expected 2`
   (`src/pipeline/gpu_pipeline_task.cpp:83`). It fills the log: 390 of the last 400 lines of
   `normal-multikey__windowed__compressed__r1/sirius_log_tail.txt`. The results are correct. The
   sink returns a one-column count where the task expects the input's column count.
7. **Harness (`sort_merge_bench.py`).** A timed-out run has no `discard_rows` and no merge
   statistics, because the process is killed before it logs them. The full log is deleted unless
   `--keep-logs` is given, so the OOM messages of a long run are gone once it ends.

## Files

| Path under `test/sort_performance/results/full/` | Content |
|---|---|
| `whole-gpu-5x/results.csv` | The 20 A/B rows, merged from the 20 harness invocations in run-pair order |
| `whole-gpu-5x/per_run/<run>/` | `config.yaml`, `query.sql`, `job.json`, `result.json`, `row.json`, `run_info.json`, `child_stdout.txt`, `sirius_log_tail.txt` (last 400 log lines) |
| `whole-gpu-5x/driver.sh`, `driver_console.txt`, `base_config.yml` | The script that ran the pairs, its console output with the disk and memory checks before each run, and the host cap |
| `whole-gpu-5x/probe/` | The GPU limit probe: script, one CSV and one `row.json` per value tried |
| `limit-24g-10x/` | The six earlier windowed runs: `results.csv`, `results.md`, `per_run/`, console output |
| `attempt1-default-host-capacity/` | Log, config and console output of the stopped run without the host cap |
| `code_state.txt` | Commit, uncommitted source diff, checksums of the sources on both machines |

The harness's `runs/` directory is `per_run/` here and `child_output.txt` is `child_stdout.txt`,
because the repository's `.gitignore` ignores `runs/` and `*output.txt`. Paths inside the JSON
files are the ones used on the workstation.

**Cleanup.** `/mnt/disk_2/sirius_spill` is empty. The harness deleted each run's spill directory,
and the driver removed the timed-out runs' leftovers before the next run. The small result
directories remain on the workstation under `/mnt/disk_2/sort_merge_bench/`.
