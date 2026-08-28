# Pinned vs unpinned TPC-H benchmark (Sirius StarRocks CNs)

Measures what `pin_table` buys on a multi-GPU StarRocks+Sirius cluster: the full
22-query TPC-H sweep run twice on identical clusters — **arm A** unpinned, **arm B**
with lineitem + orders pinned (compressed) on every CN — and compared per query.

Reference result (2× RTX PRO 6000, SF1000, host-tier compressed pins): 14 queries
comparable, total 109.2 s → 65.2 s (**1.68×**); the 9 queries actually served from
pins total 85.5 s → 41.4 s (**2.07×**), best case q06 at 5.85×. Raw numbers in
`bench/rtxpro6000-2gpu/` history.

Everything here is parameterized by `NUM_CNS` / GPU size / data path — this
document walks a **fresh 4-GPU GB200 box** end to end.

## What you need before the benchmark exists

1. **The code.** Branch with the pinning stack (`feat/pin-table-cn` until merged):
   - `pin_table`/`unpin_table` over the FFI and the CN's `ADMIN EXECUTE` channel,
   - file-subset pin serving in the scan manager,
   - the FE patch `patches/files-query-whole-file-ranges.patch` (applied by
     `apply-starrocks-patches.sh` like the nixl proto patch — the FE **must** be
     built after the patch is applied).
2. **The stack**, built per
   [`bench/rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md`](../../../../bench/rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md)
   (the `build-sirius-starrocks` skill wraps it): UCX → nixl → libsirius →
   StarRocks FE → the Rust CN.

   **GB200 is aarch64** — three extra toolchain shims are mandatory for the CN
   build (`g++`, `ld`, `libnvidia-ml.so`; §7 of the build doc). Put them on
   persistent storage and export the PATH prefix *inside* `pixi run bash -c` —
   an outer prefix is defeated by pixi. Driver must be r580+ for the CUDA 13
   engine (GB200 ships newer).
3. **The dataset** at SF1000 parquet:

   ```bash
   cd test/tpch_performance
   pixi run bash generate_tpch_data.sh 1000 --format parquet --output /data/tpch/tpch_parquet_sf1000
   ```

   ~265 GB, lineitem split into 60 part files. **Many files per table matters**:
   whole-file scan assignment (below) balances load only when file count is a
   few × (CNs × dop) and each file is well under `totalBytes / instances`.
4. **Compression plans** for the dataset ship in the repo:
   `src/compression/simpatico_codegen/plans/tpch_sf1000/` (lineitem + orders
   active). For another SF, regenerate with the simpatico codegen — plans are
   value-distribution-specific.

## Why the pieces below exist (read once)

- **Pins never serve byte-range scans**, and StarRocks normally cuts byte ranges
  for every distributed `FILES()` scan. The FE config
  `files_query_whole_file_ranges=true` (our FE patch) switches FILES() *queries*
  to whole-file assignment; each CN then serves exactly its assigned file subset
  from its pin (`serves operator ... as a file subset` in the CN log).
- **The compression keys are YAML-only** (`sirius.compression.*`), and the CN's
  `--gpu-memory-limit` flags are mutually exclusive with `--sirius-config` — so
  the benchmark launches CNs with full per-CN YAMLs (`gen-config.sh` + `up.sh`)
  instead of `cluster8.sh`'s carve-out flags.
- **Pins are in-process state**: any CN restart loses them, so the pinned arm's
  `RESTART_CMD` re-pins (`PIN_AFTER_RESTART=1`), and `pin-all.sh` retries because
  the FE's brpc channel can throw `Unable to validate object` for the first few
  seconds after a restart — without the retry, queries silently run unpinned and
  the benchmark lies.
- **`ADMIN EXECUTE` has a hard 600 s FE ceiling** (statement default + brpc stub
  cap, not raisable from SQL). Size pins to finish under it (SF1000
  lineitem+orders compressed took ~60 s/CN on NVMe); on timeout do NOT retry —
  watch the CN log for `pin_table finished`.

## Sizing a 4-GPU GB200

Read the card first: `nvidia-smi --query-gpu=memory.total --format=csv`.

| Knob | Rule | GB200 (≈186 GiB HBM, 4 GPUs) |
|---|---|---|
| `STAGING` | from a measured config for the box class (`tpch-bench` skill for the GB200 kit) | 8–32 GiB |
| `GPU_MEM` | `card − STAGING − 3 GiB` ceiling; leave working room for joins | 96–128 GiB |
| `HOST_MEM` | per-CN pinned host capacity; `NUM_CNS × HOST_MEM` must leave the OS ~15% of RAM | size to box RAM / 4 |
| `PIN_TIER` | `gpu` when the compressed pin (+ query working set) fits `GPU_MEM`, else `host` | **gpu** — the ~115 GB/CN compressed lineitem+orders pin fits 186 GiB cards |
| dataset layout | files ≥ a few × (NUM_CNS × dop), each ≪ totalBytes/instances | 60-file lineitem is fine for 4 CNs |

GPU-tier pins are worth it where they fit: on the 2-GPU box, Q6 was 5.9× from a
host pin but 12.6× from a GPU pin. On GB200 also add `l_shipmode` back into
`LCOLS` in `pin-all.sh` (and `o_comment` if q13 matters) — the RTX box dropped
them purely for memory.

## Running it

All commands from `experimental/starrocks`; `mysql` lives in the `client` pixi env.

```bash
# 0. Preflight — must all be empty/quiet, or you corrupt FE registries:
pgrep -f 'sirius-starrocks-cn|StarRocksFE'
nvidia-smi --query-compute-apps=pid --format=csv,noheader
ss -ltn 'sport = :9030'

# 1. Per-CN configs, cluster up
NUM_CNS=4 GPU_MEM=110GiB HOST_MEM=200GiB benchmarks/pinned/gen-config.sh
NUM_CNS=4 STAGING=16GiB benchmarks/pinned/up.sh   # BLOCKS — own terminal/background
NUM_CNS=4 benchmarks/pinned/restart.sh            # or: waits alive + sets FE lever/timeout

# 2. Arm A — unpinned sweep (2 warm runs per query)
export TPCH_DATA=/data/tpch/tpch_parquet_sf1000
QUERY_TIMEOUT=600 COLD_TIMEOUT=600 COLD=1 MIN_BACKENDS=4 \
  RESTART_CMD="env NUM_CNS=4 benchmarks/pinned/restart.sh" \
  pixi run -e client benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 2

# 3. Pin every CN (arm B state), then the pinned sweep.
#    PIN_AFTER_RESTART=1 makes every mid-sweep restart re-pin.
export PIN_AFTER_RESTART=1 PIN_TIER=gpu
NUM_CNS=4 benchmarks/pinned/restart.sh            # fresh cluster + pins
QUERY_TIMEOUT=600 COLD_TIMEOUT=600 COLD=1 MIN_BACKENDS=4 \
  RESTART_CMD="env NUM_CNS=4 PIN_AFTER_RESTART=1 benchmarks/pinned/restart.sh" \
  pixi run -e client benchmarks/tpch/bench.sh /tmp/bench/B/timings.csv 2

# 4. Compare
python3 benchmarks/pinned/compare.py /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv

# 5. Teardown — compute-apps must be EMPTY (memory.used never reaches 0)
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

## Verify before quoting anything

1. **Pins really landed, every window.** `grep 'PIN CN'` in the sweep log — a
   `0s`/`1s` pin right after `restarting cluster...` means the brpc retry was
   exhausted and that window ran unpinned. Re-measure those queries on a
   verified-pinned cluster (`bench.sh out.csv 2 q06 q07 ...`) and overlay the
   CSV: `compare.py A.csv B.csv B-fix.csv`.
2. **Subset serving actually happened.** Per pinned query, each CN logs
   `pinned entry 'lineitem' serves operator '0' as a file subset: N/60 files`;
   the per-CN N must sum to the full file count across CNs.
3. **Compression engaged.** `compressing with plan for 13 column(s)` per pin —
   a `pinning uncompressed` warning means the plan dir/table name didn't match.
4. **Values, not just exit codes.** `bench.sh` never compares a value. Run the
   DuckDB oracle (`bench/rtxpro6000-2gpu/tools/oracle.py` + `compare.py`, pip
   `duckdb` — never the repo binary, it fights the CNs for the GPU) at least on
   the pin-served queries, and diff arm A vs arm B row outputs.

## Known behavior at SF1000 (RTX reference run)

- q05/q08/q09/q17/q18/q21 refused with GPU OOM in **both** arms on 96 GiB cards
  with the SF500 budgets — a tuning wall independent of pinning (`cn-tuning`).
  GB200's larger HBM should clear some of these; retune, don't copy budgets.
- q11 empty is **correct** (SF1 threshold hardcoded in the query); the harness
  misfiles it as a wedge. q15 is float-flaky.
- q12/q13/q19 show 1.00× when `l_shipmode`/`o_comment` are unpinned (the RTX
  budget choice) — misses by design, not bugs.
