# StarRocks CN integration branch (temporary)

`tmp/sr-cn-integration-2026-10-20c` combines the open PRs of the Sirius StarRocks compute-node (CN) work on `main`, so they can be built and tested together before the stacks merge. **It is temporary and is not meant to be merged.** The PRs themselves are on sirius-db/sirius.

It replaces `tmp/sr-cn-integration-2026-10-15`. Since then, the exchange between CNs moved from the NIXL direct path (#2040, #2041, both closed) to the ring exchange (#2256 → #2257 → #2258), and the runtime-filter and failure stacks were restacked onto the ring.

## What's in it

The base is `main` at 9c2a44f6 (#2135).

| stack | PRs (head) |
|---|---|
| A: exchange and ring | #2038 multiple fragments per CN through a local exchange → #2039 received batches held by token (`exchange_inbox`) → #2061 exchange data spills to host → #2256 the ring's engine side → #2257 the ring's disk tier → #2258 the ring transport between CNs (a5cba00f) |
| C: runtime filters, on #2258 | #2062 broadcast runtime filters (9df44f53) |
| C: filters, on #2062 | #2102 (8c78b0f6) → #2114 (0e701a85) → #2122 (c135b923) → #2134 (3a4bfa26) → #2136 (ecd4cca0) → #2137 (764257b7) |
| C: failures, on #2062 | #2115 (bafe517a) → #2116 (c8e7a7a0) → #2118 (a9fc33be) → #2119 (faa96815) → #2120 (3f58b03c) → #2121 (33a8115d) |

The filters and failures chains are merged together here. The PRs merged on top, at these heads:

| PR | head | what |
|---|---|---|
| #2043 | 85e314f2 | split parquet files across CNs by byte range (merged into `main` after 9c2a44f6) |
| #2253 | 199482df | pinned parquet tables serve file-subset scans, and `pin_table` through the FFI |
| #2125 | 36aafdd5 | GPU casts round like DuckDB (closed; #2254 replaces it and is not in this branch) |
| #2217 | 325d323d | a streaming sink converts its hash keys again |
| #2129 | 2cfe909e | exact decimal division |
| #2130 | 8cfb73f7 | exact AVG and `/` (brings #2103, #2109, #2110, #2111 and #2127) |
| #2104 | 27b900c9 | RIGHT SEMI joins |
| #2112 | ff3b6281 | DATE and TIMESTAMP partition keys |
| #2128 | dc5b8721 | unary minus on the GPU |
| #2107 | 05bdf818 | the Substrait consumer reads `FetchRel.count_expr`/`offset_expr` |
| #2100 | c0364803 | batch dispatch uses the shared descriptor table |
| #2101 | d63046bf | the CN heartbeat reports its StarRocks release |
| #2251 | 17379938 | engine group A: `async_capped` allocator and slice combining |

Integration-only commits, in no PR:
- `spill_skip::recheck` for the ring's subscribed-batch check (#2256 with #2251).
- The filters × failures merge: `RecentQueries: Default`; `run_and_ship` returns `FragmentFailure`; a filter share's run carries its query, so a purge stops it; a purge keeps its deferred scans' params to end their hops.
- #2253 on the ring stack: `pin_table`/`unpin_table` placed in `SiriusContext`, and the FFI interrupt test closed before #2253's pin test.
- #2100 on the failures stack: the batch test uses a result sink, and the shared-table error maps to `StatusPb`.
- Two translator tests dropped that an older #2104 head carried; their fixture files are in no branch.

**Not included:** #2254 (casts; this branch has #2125), #2259 (never serve a byte-range scan from a pin), #2131 (`regexp_replace`).

## Results on 2× RTX PRO 4500 Blackwell (32 GB each, one CN per GPU, every result checked against DuckDB)

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
