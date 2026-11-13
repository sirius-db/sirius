# Smoke run of the sort-merge benchmark (bench-smoke)

Run on the workstation `felipe-TRX50-AERO-D` (192.168.0.248, x86_64, 2 × RTX PRO 6000 Blackwell,
125 GiB RAM) on 2026-11-13, through `.ai-helper/workstation.sh`, on GPU 0 only. Nothing here was
measured on the Spark. This is a smoke run: one repeat, 3× a 4 GiB limit. The times below show
that the pieces work together; they are not benchmark results.

- **GPU limit:** `sirius.memory.gpu.usage_limit_bytes = 4294967296` (4 GiB).
- **Data:** 3× the limit, about 12 GiB per dataset (464–808 million rows, 8,000,000 rows per batch).
- **Code:** the worktree at commit `8290517a` plus the uncommitted discard-sql-tests changes to
  `src/sirius_context.cpp`, `src/op/sirius_physical_discard_sink.hpp` and
  `test/cpp/integration/test_gpu_execution_copy_discard.cpp`.

## Result

The matrix runs end to end and the harness exits 0. Every check in the task spec passes except
the GPU peak, which is above the limit by more than a small margin (problem 2). One problem in
the workstation script made the first attempt fail on a stale build (problem 1).

| Check | Result |
|---|---|
| C++ tags of the phase on the workstation | Pass: 2958 assertions in 43 test cases (`cpp_tags_console.txt`) |
| Harness never crashes | Pass: 20 of 20 runs recorded, exit code 0 |
| Windowed runs: `ok`, count = expected, `order_violations = 0` | Pass: 10 of 10 |
| Windowed route taken, `rounds > 0` | Pass: `windowed_partitions` 12–29, `one_shot_partitions` 0, `rounds` 16,681–665,741 |
| Compressed arm really compressed | Pass: `runs_read_compressed` 633–4,633, `refill_bytes_compressed` 18.5–34.8 GB in all 5 windowed compressed runs; 0 in every uncompressed run |
| `one_shot` runs complete or fail cleanly | Pass: 4 `ok`, 6 `oom`, each with its error text |
| Peak GPU ≤ limit + a small margin | **Not met**: see problem 2 |
| Disk tier (extra run, not in the spec) | Pass: 4 of 4 windowed runs `ok` with 1.7–15.6 GB spilled to disk |
| Spill data deleted on the workstation | Done: `/mnt/disk_2/sirius_spill` is empty, `/mnt/disk_2/sort_merge_bench` removed |

## What was run

1. Workstation check: `.ai-helper/workstation.sh --no-build nvidia-smi` (both GPUs idle) and
   `df -h /mnt/disk_2` (1019 GiB free).
2. C++ tags:
   `.ai-helper/workstation.sh build/release/test/cpp/sirius_unittest "[copy_discard],[merge_sort_stats_log],[merger_stats],[physical_discard_sink],[discard_copy_plan],[discard_copy_function]"`.
   The first attempt failed on a stale build (problem 1); the second passed.
3. Smoke matrix (`make` for the extension is part of the script's build):
   `.ai-helper/workstation.sh python test/sort_performance/sort_merge_bench.py --gpu-limit-bytes 4294967296 --scale 3 --device 0 --validate --out-dir /mnt/disk_2/sort_merge_bench/smoke`.
   The first attempt failed on the same stale build; the second took 6 minutes for 20 runs.
4. Disk-tier run, added because the matrix above never reaches the disk (problem 3): the same
   command with `--no-build`, `--datasets uniform-1key,normal-multikey --modes windowed
   --timeout-s 1200` and a `--base-config` holding `sirius.memory.host.capacity_bytes: 4294967296`.

## Files

| Path | Content |
|---|---|
| `results.csv`, `results.md`, `run_info.json` | The smoke matrix (second attempt) |
| `per_run/<run>/` | Per run: `config.yaml`, `query.sql`, `job.json`, `result.json`, `row.json`, `child_stdout.txt`, `sirius_log_tail.txt` (last 200 log lines) |
| `harness_console.txt` | The harness's console output |
| `cpp_tags_console.txt` | The passing C++ tag run |
| `disk-tier/` | The extra run with a 4 GiB host tier, same layout, plus `base_config.yml` |
| `attempt1-stale-build/` | The first attempt: failing C++ tag output, harness output, CSV and markdown (no per-run files) |

## Smoke matrix

Peak GPU is the whole device's `memory.used`. A process that has only loaded the extension
already shows 4682 MiB (the 20 runs of the first attempt, which failed in 0.0 s).

| Dataset | Mode | Compression | Status | Wall s | Peak GPU MiB | Rounds | Runs read compressed | OOM retries |
|---|---|---|---|---|---|---|---|---|
| uniform-1key | one_shot | off | ok | 3.2 | 7030 | 0 | 0 | 0 |
| uniform-1key | one_shot | on | ok | 3.1 | 6778 | 0 | 0 | 1 |
| uniform-1key | windowed | off | ok | 5.5 | 6038 | 23,717 | 0 | 0 |
| uniform-1key | windowed | on | ok | 6.8 | 5846 | 28,401 | 4,633 | 0 |
| zipf-1key | one_shot | off | oom | 8.8 | 7638 | – | – | 203 |
| zipf-1key | one_shot | on | oom | 11.6 | 7510 | – | – | 277 |
| zipf-1key | windowed | off | ok | 6.1 | 5714 | 42,674 | 0 | 0 |
| zipf-1key | windowed | on | ok | 72.4 | 5776 | 665,741 | 2,342 | 0 |
| normal-multikey | one_shot | off | ok | 2.3 | 6536 | 0 | 0 | 1 |
| normal-multikey | one_shot | on | ok | 4.5 | 6612 | 0 | 0 | 2 |
| normal-multikey | windowed | off | ok | 5.5 | 5316 | 20,174 | 0 | 0 |
| normal-multikey | windowed | on | ok | 17.9 | 5186 | 84,282 | 1,852 | 0 |
| few_distinct-strings | one_shot | off | oom | 82.3 | 5530 | – | – | 902 |
| few_distinct-strings | one_shot | on | oom | 81.4 | 5766 | – | – | 1,001 |
| few_distinct-strings | windowed | off | ok | 7.4 | 4822 | 16,681 | 0 | 0 |
| few_distinct-strings | windowed | on | ok | 8.1 | 4902 | 21,512 | 1,767 | 0 |
| sequential | one_shot | off | oom | 6.5 | 5010 | – | – | 100 |
| sequential | one_shot | on | oom | 6.0 | 5140 | – | – | 113 |
| sequential | windowed | off | ok | 7.2 | 5072 | 22,330 | 0 | 0 |
| sequential | windowed | on | ok | 9.0 | 4976 | 29,405 | 633 | 0 |

All windowed runs: `rows_counted` = `discard_rows` = expected rows, `order_violations` 0,
`shrinks` 0, `fallback_conversions` 0, nothing spilled to disk.

The six `one_shot` failures all end with
`GPU pipeline task exceeded maximum retry limit (100) … OOM at operator MERGE_SORT` (zipf,
few_distinct) or `… OOM while preparing batches for processing` (sequential). They have no
`[merge_sort][stats]` line, so their merge columns are empty.

## Disk-tier run (host tier 4 GiB)

| Dataset | Compression | Status | Wall s | Peak GPU MiB | To disk GB | Refill from disk GB | Refill compressed GB | Runs read compressed | OOM retries |
|---|---|---|---|---|---|---|---|---|---|
| uniform-1key | off | ok | 6.6 | 5476 | 15.2 | 15.1 | 0 | 0 | 0 |
| uniform-1key | on | ok | 8.6 | 5678 | 7.8 | 0 | 20.3 | 7,461 | 0 |
| normal-multikey | off | ok | 6.7 | 5494 | 15.6 | 13.3 | 0 | 0 | 0 |
| normal-multikey | on | ok | 13.5 | 5182 | 1.7 | 0.08 | 25.8 | 1,931 | 3 |

All four: count = expected, `order_violations` 0. The spill directories were deleted by the
harness when each run ended.

## Problems found

### 1. `.ai-helper/workstation.sh` can leave a stale object file on the workstation (script, not Sirius code)

- **File:** `.ai-helper/workstation.sh`, the `rsync -a` sync followed by `make release && make`.
- **What happened:** the first tag run synced while another task was still editing
  `src/sirius_context.cpp` (final mtime 23:57:43). The workstation compiled the older content at
  23:58:16. The next sync copied the finished file with its original mtime (23:57:43), which is
  older than the object, so ninja did not recompile it. The mirror's source was correct (same
  md5 as the Spark's) and the build was not.
- **Evidence:** `strings build/release/CMakeFiles/sirius_objects.dir/src/sirius_context.cpp.o | grep -c sirius_discard`
  gave 0 on the workstation and 1 on the Spark, for identical sources.
- **Effect:** the discard COPY was not sent to the GPU. `[copy_discard]` failed 4 of 5 cases
  (`attempt1-stale-build/cpp_tags_console.txt`) while passing on the Spark (1920 assertions), and
  all 20 benchmark runs failed in 0.0 s with
  `Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true)`
  (`attempt1-stale-build/harness_console.txt`).
- **What I did:** deleted the workstation object files compiled between 23:56 and 00:00
  (build output only) and rebuilt. Tags and matrix then passed.
- **Suggested fix:** sync without preserving mtimes for changed files (for example
  `rsync -rlpD --checksum`, or `-a --no-times --checksum`), so a file whose content changed gets a
  new mtime on the workstation. Any agent that syncs while another edits can hit this again.

### 2. Device memory goes well above the configured limit

- **Command:** the smoke matrix above; column `peak_gpu_mib` of `results.csv`.
- **Output:** the limit is 4096 MiB and an idle process with the extension loaded shows 4682 MiB.
  Windowed runs peak at 4822–6038 MiB, up to 1356 MiB above the idle level and 1942 MiB above
  the limit (`uniform-1key` windowed uncompressed: 6038; `zipf-1key`: 5714 and 5776).
  `one_shot` runs peak at up to 7638 MiB (`zipf-1key`), 3.5 GiB above the limit.
- **Reading:** part of the device's memory use is outside what the limit governs. The harness
  number is for the whole device, so it cannot say which allocations those are. Not diagnosed
  here. Candidates to look at: the merge's working memory (`src/op/sirius_physical_merge_sort.cpp`
  and the windowed merger), decompression scratch, and whether `usage_limit_bytes` is a hard cap
  on the pool.
- **Consequence for the full run:** "peak GPU ≤ limit" cannot be claimed from `peak_gpu_mib` as
  it stands. The write-up needs either a peak taken from Sirius's own accounting or the measured
  overshoot stated next to the limit.

### 3. The spec's smoke command never reaches the disk tier

- **Command:** the smoke matrix above. **Output:** `spill_bytes_disk = 0` and
  `refill_bytes_disk = 0` in all 20 runs.
- **Cause:** 12 GiB of data against a host tier of about 90 GiB (0.9 × 125 GiB × 0.8).
- **What I did:** the extra `disk-tier/` run with `sirius.memory.host.capacity_bytes: 4294967296`.
  Disk spill, refill from disk runs and compressed runs on disk all work there.
- **Suggestion:** `test/sort_performance/README.md` could give this `--base-config` as the way to
  smoke-test the disk path.

### 4. Windowed merge with compression is 12× slower on `zipf-1key` (performance, not correctness)

- **Command:** the smoke matrix; runs `zipf-1key__windowed__uncompressed__r1` and
  `zipf-1key__windowed__compressed__r1`.
- **Output:** 6.1 s and 42,674 rounds uncompressed; 72.4 s and 665,741 rounds compressed
  (`round_rows_min` 5, `mergers_built` 932 against 475, `intermediate_tasks` 1,421 against 774).
  `normal-multikey` shows the same in smaller form: 5.5 s / 20,174 rounds against
  17.9 s / 84,282 rounds. The other three datasets differ by 10–25 %.
- **Reading:** with compressed runs and heavily tied keys the merger runs many very small rounds.
  The code is the windowed merger's round sizing and refill on compressed runs
  (`src/op/sirius_physical_merge_sort.cpp` and the `windowed_merger` it drives). One repeat only,
  so treat the ratio as an indication.
- **Consequence for the full run:** at 10× a large limit this cell may take hours. Check it
  against `--timeout-s` (default 7200) before the long run.

### 5. Failing `one_shot` runs are slow and spill heavily before they give up

- **Command:** the smoke matrix; runs `few_distinct-strings__one_shot__*`.
- **Output:** 82 s each, 902 and 1,001 OOM retries, and `spill_bytes_host` of 941 GB and 1,077 GB
  (cumulative bytes moved, for 12 GiB of data) before the retry limit of 100 ends the query.
- **Consequence for the full run:** these cells are expected to fail, and they will take much
  longer and move far more data at full scale. Budget time for them, or bound them with
  `--timeout-s`. With a disk tier in use, check free space while they run.

### 6. Small harness observations (`test/sort_performance/sort_merge_bench.py`)

- `sirius_log_tail.txt` holds the last 200 lines. That was enough for the statistics lines here.
  For a failed run it holds only retry messages; use `--keep-logs` when a failure needs study.
- A failed `one_shot` run still reports `discard_rows` (636,194,674 of 808,000,000 for
  `zipf-1key`): rows of partitions that finished before the failure. The value is correct and
  easy to misread as a row mismatch.
- No harness defect found: statuses, the exit code, the per-run configs (the compressed arm alone
  has `spill_min_batch_bytes: 1048576`), the parsers and both output files behaved as the README
  says.

## Cleanup

- `/mnt/disk_2/sirius_spill` is empty; the harness removed each run's spill directory.
- `/mnt/disk_2/sort_merge_bench` (results, copied here first) and my scratch directories under
  the workstation's `/tmp` are deleted.
- The mirror `/mnt/disk_2/sirius-scalable-sort` is left as the script keeps it.

## File names

The repository's `.gitignore` ignores `runs/` and `*output.txt`. So that these results can be
committed, the harness's `runs/` directory is `per_run/` here, `child_output.txt` is
`child_stdout.txt`, and the console captures are `*_console.txt`. The contents are unchanged;
`row.json`, `job.json` and `run_info.json` still hold the paths used on the workstation.
