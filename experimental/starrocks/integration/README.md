# StarRocks CN integration branch (temporary)

`tmp/sr-cn-integration-2026-10-20f` combines the open PRs of the Sirius StarRocks compute-node (CN) work on `main`, so they can be built and tested together before the stacks merge. **It is temporary and is not meant to be merged.** The PRs themselves are on sirius-db/sirius.

It replaces `tmp/sr-cn-integration-2026-10-20c`. Since then stack A became two PRs: #2038, then one ring PR (#2258) that folds in #2251, #2039, #2061, #2256 and #2257. Stack C was rebased onto it, #2124/#2125 became #2254, and #2259 is the pin/byte-range rule.

## What's in it

The base is `main` at e034d221 (#2241). Merged in this order (heads as of Oct 20, 21:00 UTC):

| step | PR (head) | what |
|---|---|---|
| 1 | #2121 (cbd01058) | stack A: #2038 multiple fragments per CN (28629cce) → #2258 the ring exchange (412d4f4b); stack C root #2062 (e0520f50); failures #2115 → #2121 |
| 2 | #2137 (bf6f241d) | filters #2102 → #2137 |
| 3 | #2254 (2a29128d) | GPU numeric casts round and range-check like DuckDB |
| 4 | #2217 (325d323d) | a streaming sink converts its hash keys again |
| 5 | #2259 (1642053d) | #2253 pinned tables + never serve a byte-range scan from a pin |
| 6 | #2129 (2e260070) | exact decimal division |
| 7 | #2128 (29fc3c16) | unary minus on the GPU |
| 8 | #2130 (8cfb73f7) | coverage chain #2103, #2109, #2110, #2111, #2127, #2130 |
| 9 | #2104 (27b900c9) | RIGHT SEMI joins |
| 10 | #2112 (ff3b6281) | DATE/TIMESTAMP partition keys |
| 11 | #2107 (0d41dfc0) | Substrait `FetchRel.count_expr`/`offset_expr` (substrait at bdfb4eac, ext#5) |
| 12 | #2100 (c0364803) | batch dispatch uses the shared descriptor table |
| 13 | #2101 (d63046bf) | the CN heartbeat reports its StarRocks release |

Integration-only changes, in no PR:
- **filters × failures** (in the step-2 merge): `RecentQueries: Default`; `run_and_ship` returns `FragmentFailure`; a filter share's run carries its query, so a purge stops it; a purge keeps its deferred scans' params to end their hops; SRNX kinds TOPOLOGY 5, SHARE_ABANDONED 6, FAILED 7.
- **#2217 × #2258:** `partition_key_cast::normalize` in the sink's single `hash_partition` call.
- **#2253 × stack A:** `pin_table`/`unpin_table` take the context through its `RefCell`; #2253's tests go after A's.
- **#2129 × #2128:** ids abs 33 (main), decimal_divide 34, negate 35 (36 in all); #2128's signature table gains `{"sirius_decimal_divide", 3, 3}`.
- **#2100 × stack A/failures:** `resolve_descriptor_table` maps to `StatusPb`; the batch test's instances get a result sink.
- **#2104:** two translator tests dropped whose fixtures no PR carries (20c f6dff01e).

**Not included:** #1965 (host Arrow input), #2131 (`regexp_replace`), #2232 (NUMA placement for pins), the CN pin command (needs a port onto #2258).

## Results on 4× GB200 (189 GB each, one CN per GPU, every result checked against DuckDB)

Settings: `GPU_ALLOCATOR=async_capped`, `SIRIUS_CN_RING_GPU_LANDING_FRACTION=0.25` (env.sh), 85% of each GPU for the pool,
128 GiB host tier per CN (SF1000), 4 pipeline threads, data on `/scratch/sirius/datasets` (GPFS). Tools repo: `box/gb200`
rebased on `main` plus 75c3494 (the harness reads `GPU_ALLOCATOR`/`DISK_DIR`).

- **Unit tests:** C++ tests of the 25 test files the branch touches against `main`: 524 cases pass. CN: CI checks pass; the
  engine-linked CN tests pass (208, 1 ignored).
- **SF1 smoke:** 22/22 (the harness's `nixl-hops` check fails: it looks for the direct path's log line).
- **SF1000, 2 iterations on fresh clusters:** 44/44, sum of per-query medians **41.1 s**. Same box, Oct 19 builds on the old
  direct NIXL exchange: v10 38.5 s (both runs), v9 40.4 s.

| query | 20f median | v10 median (2 runs) | Δ vs v10 |
|---|---:|---:|---:|
| q01 | 4.81 | 4.60 | +5% |
| q02 | 1.23 | 1.15 | +7% |
| q03 | 1.73 | 1.68 | +3% |
| q04 | 1.12 | 1.16 | -3% |
| q05 | 1.52 | 1.56 | -2% |
| q06 | 1.22 | 1.34 | -9% |
| q07 | 1.91 | 1.87 | +2% |
| q08 | 2.21 | 1.97 | +12% |
| q09 | 3.25 | 3.44 | -6% |
| q10 | 2.09 | 1.74 | +20% |
| q11 | 0.71 | 0.66 | +7% |
| q12 | 1.24 | 1.19 | +4% |
| q13 | 1.18 | 1.09 | +9% |
| q14 | 1.31 | 1.33 | -2% |
| q15 | 2.20 | 2.23 | -1% |
| q16 | 0.46 | 0.43 | +7% |
| q17 | 2.34 | 2.07 | +13% |
| q18 | 2.42 | 1.94 | +25% |
| q19 | 1.59 | 1.54 | +3% |
| q20 | 1.69 | 1.60 | +6% |
| q21 | 4.28 | 3.33 | +28% |
| q22 | 0.57 | 0.56 | +2% |
| **sum** | **41.1** | **38.5** | **+6.7%** |

## 20c results on 2× RTX PRO 4500 Blackwell (32 GB each, one CN per GPU, every result checked against DuckDB)

Settings: `GPU_ALLOCATOR=async_capped`, `SIRIUS_CN_RING_GPU_LANDING_FRACTION=0.25`, a 256 GiB disk tier on the instance NVMe (`DISK_DIR`), 36 GiB host tier per CN. `env.sh` sets the first two; see "Setup".

- **SF1 smoke:** 22/22.
- **SF1000, all 22 queries in one run:** 22/22 in **608.9 s** (sum of query times), no leak-check failures. Runtime filters applied 48 times.

| | this branch | ring without stack C | Oct 19 ring build with stack C |
|---|---:|---:|---:|
| q09 | 148.4 s | 231.6 s | 162.2 s |
| q21 | 115.8 s | 145.0 s | 146.3 s |
| q08 | 78.5 s | 71.6 s | 95.4 s |
| q18 | 56.8 s | 72.4 s | 73.0 s |
| q05 | 49.8 s | 49.2 s | 66.1 s |
| q17 | 47.2 s | 87.6 s | 65.6 s |
| **22 queries** | **608.9 s** | 773.6 s | 718.5 s |

- **Unit tests:** the CN's CI checks (`cargo fmt --check`, clippy `-D warnings`, `cargo test --workspace --no-default-features`) pass; the engine-linked CN tests pass (208, 1 ignored). C++: `[sirius_ffi]`, `[exchange_inbox]`, `[exchange_ring]`, `[streaming_fragment]`, `[interrupt]`, `[late_failure]`, `[decimal_cast]` and `[function_id]` pass, apart from one byte-range estimate test (see "Known issues").
- **The harness's `nixl-hops` check fails** on every run: it looks for the direct path's log line, which the ring doesn't write. Query results are unaffected.

## Setup, test, bench

You need `pixi` in `~/.pixi/bin`, the TPC-H parquet data, and a box profile in the tools repo:
- **4× GB200:** `box/gb200` (the default).
- **AWS g7e, and 2× RTX PRO 4500 boxes:** `export TOOLS_BRANCH=box/aws-g7e`. On 32 GB GPUs also set `DISK_DIR` to a directory on local NVMe (for example `/opt/dlami/nvme/sirius-disk`): at SF1000 the ring lands about 25 GiB per CN on disk.

The tools repo needs commit `75c3494` or later on its box branch (the harness reads `GPU_ALLOCATOR` and `DISK_DIR`).

```bash
export WORK=/opt/dlami/nvme/sr-cn   # local disk with about 40 GB free, not the NFS $HOME
git clone -b tmp/sr-cn-integration-2026-10-20c https://github.com/aocsa/sirius.git $WORK/sirius
cd $WORK/sirius/experimental/starrocks/integration

export TOOLS_BRANCH=box/aws-g7e DISK_DIR=/opt/dlami/nvme/sirius-disk   # 2× RTX PRO 4500 / g7e only
./setup.sh            # tools repo next to this checkout, submodules, then UCX/NIXL, engine, FE, CN
./test.sh smoke       # 2 CNs, SF1, all 22 queries: a quick check of the build
GPUS="0 1" ./test.sh sf1000      # one CN per GPU, SF1000, all 22 queries, each checked against DuckDB
./bench.sh            # SF1000, 3 iterations: runtimes.csv + summary.md
```

- **`setup.sh`:**
  - clones https://github.com/aocsa/starrocks-setup-test-tools (`$TOOLS_BRANCH`) to `$WORK/starrocks-setup-test-tools`, and rebases it on `main` for the q11 fix;
  - checks out the submodules, `substrait` first (below);
  - builds UCX/NIXL (into `$WORK/tools`), the engine and the FE in parallel, then the CN.

  It runs the same steps as the tools repo's `setup/all.sh`, apart from the engine step: the tools' `build-engine.sh` still looks for `libsirius.so` in `build/release/extension/sirius`, and since DuckDB v2 the library is in `build/release`. Logs go under `$RUN_ROOT_BASE/setup-logs/`.
- **`test.sh`** wraps `tests/4cn_tpch_joins_sf1000.sh`, and **`bench.sh`** wraps `scripts/bench.sh`. Both run all 22 queries unless told otherwise. Settings pass through the environment; the tools repo's `README.md` lists them all. The ones most used here:
  - `TPCH_QUERIES="q07 q18"`: run only some queries.
  - `GPUS="0 1"`: use only these GPUs, one CN each.
  - `SIRIUS_CN_RUNTIME_FILTERS=0`: runtime filters off.
  - `ENGINE_LOGS=1`: keep each CN's engine log (`SIRIUS_LOG_LEVEL=trace` adds per-operator timings).
  - `INJECT_FAILURE_QUERY=q05 ./test.sh`: first run q05 with one fragment failed on purpose, and check that the CNs hold nothing afterwards.
  - `./test.sh sf3000`: SF3000, with 256 GiB of host spill per CN.
- **`env.sh`** (sourced by the others) sets what the tools repo doesn't know about this branch:
  - `SIRIUS_LIB=$SIRIUS_DIR/build/release`: the tests look for `libsirius.so` there.
  - The ring's settings: `GPU_ALLOCATOR=async_capped` (the harness defaults to the slab the direct path needed) and `SIRIUS_CN_RING_GPU_LANDING_FRACTION=0.25` (the share of the GPU pool that received batches may hold before they land on host, then disk). The other ring settings are `SIRIUS_CN_RING_*` (frame, send and receive ring sizes, host landing bytes, disk writers, credits, wait limit) and `SIRIUS_CN_NIXL_XFER_TIMEOUT_SECS`, documented in `experimental/starrocks/scripts/cn-env.sh`.
  - `NO_COLOR=1`: ANSI codes in the CN logs break the harness's log checks.
  - `DATA_ROOT` on `box/gb200`: it reads `/scratch/sirius/datasets`. When `/scratch` is unreachable (a stale file handle), `env.sh` switches to `/opt/sirius-ci/datasets` and says so. That copy is about 5× slower at SF1000, so don't compare its timings with `/scratch` ones. Set `DATA_ROOT` yourself to choose.
- **Where results go:**
  - CN, FE and query logs: under `$RUN_ROOT_BASE` (printed at the start of a run; the box profile sets it).
  - Bench results: under `.../bench/<timestamp>_<label>/`.
  - The cached DuckDB answers: under `.../oracle`, shared by `test.sh` and `bench.sh`. They are keyed by the data path. The first run at a scale factor computes them, which takes several minutes per query at SF1000.
- **Before starting a run,** check `nvidia-smi`. The scripts refuse to start if the GPUs are busy. On a shared box, other jobs can hold the GPUs.

## The `substrait` submodule

#2107 needs the `substrait` submodule at bdfb4eac, the fix that makes the Substrait consumer read `FetchRel.count_expr`/`offset_expr`. It sits on 922489d, the v2 commit `main` pins.

bdfb4eac is on sirius-db/duckdb-substrait-extension's branch `fetch-count-expr-v2`, the head of PR #5 against `v2`. So a normal `git submodule update --init --recursive` fetches it. If that branch is deleted after the PR is squash-merged, the commit stays on GitHub as `refs/pull/5/head`. `ensure-substrait.sh` (run by `setup.sh`) then fetches it from there. Run it by hand after a fresh clone if `git submodule update` complains about `substrait`.

## Known issues

- **SF1000:** none failing on 2× RTX PRO 4500 with the settings above.
- **`FFI byte ranges: a ranged scan is not estimated above the rows it can own`** fails (60000 rows estimated against a bound of 30000). It comes from #2043 with #2253, not the ring or stack C, and changes no results.
- **Spill-heavy scans:** in q09, q08 and q05 one lineitem fragment scans all 3 billion rows, partitions them, and only then applies the runtime filter as a semi-join. Most of that fragment's time goes to reloading its own spilled scan output (q09: 302 GiB spilled). Filtering inside the scan (#2197) is the fix.
- **q17's large lineitem shuffle is not filtered** (`target_not_scan`): the filter applies to the other lineitem branch only.
- **Fragments run one at a time on each CN.** The engine serializes them; overlapping a fragment with the one that feeds it is not done yet.
- **Not measured on this branch:** SF3000, pinning (`--pin`, `PIN_COMPRESSION=1`) and the 4× GB200 box.
- **First query on a fresh cluster:** about 2 s slower. Each CN sets up its NIXL peers and ring lanes, and the FE is cold.
- **DECIMAL→DOUBLE:** DuckDB v2 rounds this cast correctly, and the GPU's cuDF cast may differ in the last bit. TPC-H results are compared with a tolerance, so it doesn't show there.
