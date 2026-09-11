# Prompt: work the top-3 fixes for the 2-CN TPC-H SF1000 OOM edge on the RTX PRO 6000 box

Paste everything below into a new Claude Code session started in `/home/ubuntu/sirius-wt`.

---

You are continuing performance work on Sirius (GPU SQL engine, DuckDB extension, cuDF/RMM/cuCascade) running as a StarRocks compute node (CN).
Goal: make TPC-H SF1000 pass 22/22 with TWO CNs (one per GPU) at integration speed, by removing the per-CN materialization created by the
lineitem shuffle. Three work packages, in this order, each with a measurable gate. Read the referenced documents before touching code.

## 0. Read first (30 minutes)

1. `/home/ubuntu/starrocks-tools/docs/CHECKLIST-2cn-oom-edge-2026-09-07.md` — what was tried, what is on each branch, what is open. Do not repeat §1 or §2.
2. `/home/ubuntu/sirius-wt/all22/ANALYSIS-oom-edge-2cn-staging.md` — root cause with log/code citations; §6 recommendations; §6 "three arms to run first".
3. `/home/ubuntu/sirius-wt/all22/REVIEW-sf1000-q08-memory-staging-analysis.md` — verified review of the external analysis
   (`/home/ubuntu/sirius-wt/all22/external/sf1000-q08-memory-staging-analysis.md`, from sirius-db/sirius `docs/sf1000-q08-memory-staging`).
4. Plans 00-17: `/home/ubuntu/sirius-wt/mcn/experimental/starrocks/docs/performance/NN-*.md` (10 small-batch policy, 12 nonblocking fragment graph,
   05 dispatch/drain overlap, 11 fragment fusion, 01 lease lifecycle, 03 spill/reload, 04 credits) and `TUNABLES.md`.
5. Results: `arms/R-status-20260906.md`, `arms/R-controls-20260906.md`, `arms/SA-vs-MCN.md`, `arms/R2-vs-R-compare.md` (if present), `all22/LOG.md` (append a row per step).
6. Memory notes for this box: `/home/ubuntu/.claude/projects/-home-ubuntu-sirius/memory/` (rtxpro6000-box-layout, oom-edge-2cn-root-cause, mcn-branch-benchmark,
   perf-path04-receive-credits, standalone-vs-cn-bench).

## 1. Box and trees (all built; do NOT rebuild unless you change code)

- Source `/home/ubuntu/sirius-wt/env.sh` first (pixi PATH, TOOLS_DIR nixl/UCX, CONDA_OVERRIDE_CUDA=13, JAVA_HOME, TMPDIR on nvme). Run engine commands with `pixi run`.
- `/home/ubuntu/sirius-wt/demo`  = aocsa/sirius `all22/integration` 95bec853 (baseline engine + CN; own FE build under experimental/starrocks/starrocks/output/fe).
- `/home/ubuntu/sirius-wt/perf`  = `perf/exchange-04-07-06` 222b646a (path 04 credits + staged-frame spill; 20/22 at 2 CNs). FE symlinked from demo.
- `/home/ubuntu/sirius-wt/mcn`   = `perf/multi-cn-ingress-packing-transfer` 2e0cbf51 (Codex: owned host ingress, pack worker, fair transport; opt-in
  `SIRIUS_EXCHANGE_OPTIMIZED=1`, `SIRIUS_CN_NIXL_TRANSFER_WINDOW`; 21/22 at 76/16). FE symlinked from demo. **Do not run its prep script's FE step** (multi-hour build).
- `/home/ubuntu/sirius-wt/pin`   = `feat/pin-table-cn` d24f02c4 (has `benchmarks/pinned/gen-config.sh` + `up.sh`: the only launcher that passes a full `--sirius-config`).
- Engine build: `cd <tree> && pixi run make` (after a header/class-layout change run `pixi run make clean` first: stale TUs caused heap corruption once).
  CN build: `cd <tree>/experimental/starrocks && pixi run bash -c 'source scripts/cn-env.sh && cargo build --release -p sirius-starrocks-cn'`.
  GPU-linked tests: `cargo test -p sirius-starrocks-cn --release -- engine_` and `cd ../../rust && cargo test -p sirius --lib`, under the GPU lock. If a grep of
  test output is silently empty the binary aborted: run it under gdb (`pixi exec --spec gdb -- gdb`).
- Dataset `/home/ubuntu/tpch_parquet_sf1000` -> `/opt/dlami/nvme/tpch/...` (265 GiB, 90 files). If the box was restarted the nvme is empty: `bash ~/regen_tpch.sh 1000`
  (6 min) and `mkdir -p` the dangling cache symlink targets (see rtxpro6000-box-layout memory). Oracle: `/home/ubuntu/starrocks-tools/oracle/tpch_sf1000`.
- GPUs are 95.6 GiB: CN pool + staging arena <= 92 GiB (84/8 or 76/16). The arena is a separate cudaMalloc outside the pool.

## 2. Tools (use these; do not write new harness code unless a knob is missing)

- `all22/gpu-lock.sh <cmd>` — serializes anything touching the GPUs. Everything GPU-side goes through it. `capture-arm.sh` refuses to start if a CN is already running.
- `harness/capture-arm.sh <tree> <ncn> <tag> <timeout> <runs> q01..q22` — one arm: cluster up, readiness, queries (1 cold + N warm), cluster down, logs to
  `arms/<tag>/` (runs/runs.csv, runs/qNN.rK.err, cluster.log, engine-.cnN.log, cnlog.txt). Env: `GPU_MEM STAGING HOST_MEM EXTRA_ENV QD FE_SETUP_SQL
  RESTART_ON_FAIL=1 SIRIUS_QUERY_WATCHDOG_SECS PIN_MODE`. Drivers to copy from: `all22/rerun-all.sh`, `all22/rerun2-2cn.sh`, `all22/controls-20260906.sh`, `all22/arms-MCN.sh`.
- `harness/compare.py <arm>/runs <oracle>` — oracle check (rel 1e-6) on every run; `pin_compare.py <arm>...` — per-query ratios/pass sets across arms;
  `harness/results_table.py tag=path/runs.csv ...` — timing table; `all22/evidence.sh <arm>` — restarts, stage OOM, arena exhausted, EXPORT_CAPACITY_EXCEEDED,
  credit waits, `[exchange_ingress]/[exchange_reload]/[downgrade]` counts, pack_wait_us/queue_wait_us stats, warm-run spread.
- `harness/cnlog_extract.py` (per-query frags + tx_bytes), `harness/explain_summary.py` (FE plans; `arms/GATEA-2cn/explain/` has EXPLAIN COSTS for all 22).
- Standalone comparison: `all22/arms-SA.sh <tag> [qNN..]` (DuckDB CLI + extension, config `all22/sirius-standalone-rtx6000.yaml`; SQL must be the CN kit with FILES() -> read_parquet).
- Fragment dumps at 1 CN: `arms/P04s2-1cn/dump/`. Engine pool snapshots: `[gpu_pool]`/`[host_pool]` lines at QueryBegin/QueryEnd in engine-.cnN.log.

Protocol: 2 CNs one per GPU, SF1000 kit `demo/experimental/starrocks/benchmarks/tpch/queries`, `FE_SETUP_SQL="SET GLOBAL cbo_cte_reuse_rate = 1.15"`,
TO=300, watchdog 240, RUNS=2, `RESTART_ON_FAIL=1`, `ASYNC=1` (default), oracle on every run, record `config.txt` with commit/dirty/sha256 of the CN binary, compare
same-day arms only (the box changed host on 2026-09-06), never drop failed queries from a table. Log every arm in `all22/LOG.md`. Kill your own processes with
`pkill -f '[p]attern'` (a bare pattern matches your shell).

## 3. Work package 1 — Plan 10: bounded frames and batches (start here; config-only first)

Why: mcn's per-frame cap is `arena/4 - 8 MiB` (`mcn/src/sirius_ffi.cpp:1176-1182`, receiver caps `exchange_protocol.rs:162-165,284-289`, TX `nixl_transport/pipeline.rs:602-668`);
the default engine batch is pool/40 (`src/sirius_config.cpp:583-591`) = 2.25 GB at 84 GiB, so optimized mode at 84/8 dies with EXPORT_CAPACITY_EXCEEDED
(q07 q09 q16 q22) and had to run at 76/16.
Step A (shell only): add an optional `CN_SIRIUS_CONFIG` hook to `mcn/experimental/starrocks/benchmarks/cluster8.sh` (line ~86 passes `--gpu-memory-limit`/`--host-memory-limit`;
`--sirius-config` is mutually exclusive with them, see `experimental/starrocks/src/main.rs:65-87`). Generate one YAML per CN with `usage_limit_bytes: 84GiB`,
`reservation_limit_fraction: 1.0`, host `capacity_bytes: 160GiB`, `operator_params: {scan_task_batch_size: 1GiB, hash_partition_bytes: 1GiB, concat_batch_bytes: 1GiB,
max_build_hash_table_bytes: 2GiB}` (model: `pin/experimental/starrocks/benchmarks/pinned/gen-config.sh`, standalone `all22/sirius-standalone-rtx6000.yaml`), and the
CPU-affinity lists the CN writes today (`mcn/experimental/starrocks/.cn0/derived-sirius-config.yaml`).
Arms: `MCN-1g-84-2cn` (84/8, `SIRIUS_EXCHANGE_OPTIMIZED=1 SIRIUS_CN_NIXL_TRANSFER_WINDOW=2`) and `MCN-1g-76-2cn` (76/16, same operator_params) so the pool effect is
isolated with batch bytes pinned. Gate: 0 EXPORT_CAPACITY_EXCEEDED, >= 21/22, oracle MATCH every run, common-16 warm sum within 5% of R-MCN16-2cn (121.9 s on 2026-09-06).
Step B (code, if A passes): make the sender split an oversized packed table into self-contained frames (own metadata, exact rows, sequence, ownership) instead of
throwing; preserve NULL masks/strings/offsets; bound the parent reload (`sirius_ffi.cpp:1120` reloads the whole source batch before checking size). Tests: size
boundaries, variable-width/NULL data, zero rows, asymmetric peers, concurrent TX/RX, cancellation, duplicate publication. Re-run the two arms plus `evidence.sh`.

## 4. Work package 2 — Plan shape: filter and broadcast before shuffling lineitem

Why: FE cardinalities are all 1 (`arms/GATEA-2cn/explain/off/q08.costs.txt`), so lineitem is hash-partitioned for its first join; q08 ships 101 GB for a 2-row result.
One CN and standalone pass because there is no shuffle (fusion / dynamic-filtered probe).
Step A (diagnostic, SQL only): write q05/q08/q09 with explicit joins and `JOIN [BROADCAST]` on the filtered dimensions (part/region/nation/customer branch), lineitem
first (grammar `StarRocks.g4:2536-2567`; `JoinHelper.java:344-348`; hinted joins bypass the row-limit guard `EnforceAndCostTask.java:291-306`). Put them in a
separate `QD` directory; gate each on EXPLAIN showing lineitem not shuffled. Arms `H-I2-2cn` (demo 84/8) and `H-MCN16-2cn` (mcn 76/16). Gate: q08 passes in demo at
84/8 with `tx_bytes` < 10 GB (cnlog.txt) and MATCH; `[gpu_pool]` join-window peak well below 90.2 GB.
Step B (durable, FE patch): apply `PREDICATE_UNKNOWN_FILTER_COEFFICIENT` (0.25, `StatisticsEstimateCoefficient.java:31`, currently unused) to equality/LIKE
predicates on UNKNOWN FILES() columns in `BinaryPredicateStatisticCalculator.java:86-120`, feed real FILES row counts (fix-3 knob, `dp/files-cardinality`), and raise
`broadcast_row_limit` (0.25 x 200M = 50M still > 15M). FE lives in `demo/experimental/starrocks/starrocks` (patched submodule; FE rebuild is long: plan it, keep the demo FE
for the other trees). Validate with `harness/explain_summary.py` on all 22 (no regressions in plan shape) before any arm.
Optional engine side: accept a small stream-fed build as dynamic-filter evidence (`build_filter_evidence.cpp:23-48`, `dynamic_filter_keep_threshold`).

## 5. Work package 3 — Plan 12 + 05: stream the shuffle (structural; start in parallel)

Why: the CN parks the entire sender output and dispatches the receiver only after every sender's EOS (`local_exchange.rs:573-611`, `engine.rs:640-745`,
`compute_node_service.rs:1210-1243`; export runs from `export_provider` after producer completion). Per-CN working set = whole partition + build side.
Design: dispatch the receiving fragment once local senders are parked and the first remote frame is readable; push frames during Run; close on EOS; export during Run.
Keep blocking-operator semantics (join build completion). Bound in-flight bytes with the existing credits (path 04) or mcn's window. Start with a design note
(read plans 12 and 05, and `docs/super-sirius/memory-management.md`), get it reviewed, then implement on a branch off `perf/multi-cn-ingress-packing-transfer`.
Gate: q05/q08/q09 pass at 2 CNs 84/8 with the unhinted plans; join-window `[gpu_pool]` peak <= scan + window; common-16 within 5% of integration; 22 queries MATCH x3.

## 6. Before any of the above: telemetry (half a day, do it first)

Promote the downgrade outcome to INFO with candidates / skipped_subscribed / lock_failed / host_reserve_failed per tier (`mcn/src/downgrade/downgrade_executor.cpp:366-400`),
add `[gpu_pool]`/`[host_pool]` snapshots inside the `:367` "downgrade request not satisfied" warning, log per-query arena peak live and receive-credit state.
Arm `M-dbg-2cn` (mcn 76/16, opt on, window 2, q08 x3): the futility line must become attributable (which owners hold the 81.57 GB, why 0 bytes were convertible).

## 7. Reporting

For each arm: tag, config.txt, pass+MATCH count, failures with error class (stage OOM / engine OOM / EXPORT_CAPACITY_EXCEEDED / retry limit / timeout),
`evidence.sh` output, per-query table vs the same-day reference arm (`pin_compare.py`), and a LOG.md row. For each code change: branch name, commit, tests run
(with counts), and the arm that proves it. Stop and report if a gate fails twice; do not widen scope to other plans without saying so.
