# Box, trees, tools and protocol for the top-3 work on the g7e.4xlarge (1x RTX PRO 6000 in MIG mode, 2x 2g.48gb)

Adapted 2026-09-09 from `original/PROMPT-top3-2cn-oom-edge.md` §1-§2. Mechanism sections §3-§6 of the original are superseded by
`PLAN-top3-2mig.md`; numbers come from `BOX-TRANSLATION.md`. Where this file and BOX-TRANSLATION disagree, BOX-TRANSLATION wins.

## 1. Box and trees (all built except where stated; do NOT rebuild unless you change code)

- `source /home/ubuntu/sirius-wt/env.sh` first (pixi PATH, TOOLS_DIR nixl/UCX, CONDA_OVERRIDE_CUDA=13, JAVA_HOME, TMPDIR on nvme). Engine commands through `pixi run`.
- GPU: one RTX PRO 6000 split into two MIG 2g.48gb instances (47.4 GiB each), CUDA ordinals 0 and 1. Every cluster launch exports `GPU_DEVICES=0,1`
  (cluster8.sh override; present on `bench/sf500-2-mig-gpus` and on every worktree cut from it or patched with
  `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/scripts/sf500-2mig/sirius-patches/0001-*.patch`). Never set `CUDA_VISIBLE_DEVICES` yourself; never use MIG UUIDs.
- Host: 124 GiB, no swap, 8 cores / 16 threads. `HOST_MEM=40GiB` per CN (44 GiB ceiling). Builds are cold and slow (BOX-TRANSLATION §1).
- Trees (git worktrees under `/home/ubuntu/sirius-wt`):
  - `demo`  = aocsa/sirius `all22/integration` 95bec853 (baseline engine + CN; the FE build every tree symlinks: `demo/experimental/starrocks/starrocks/output/fe`).
  - `perf`  = `perf/exchange-04-07-06` 222b646a (path-04 credits + staged-frame spill). FE symlinked from demo.
  - `mcn`   = `perf/multi-cn-ingress-packing-transfer` 2e0cbf51 (opt-in `SIRIUS_EXCHANGE_OPTIMIZED=1`, `SIRIUS_CN_NIXL_TRANSFER_WINDOW`). FE symlinked from demo.
    **Do not run its prep script's FE step.**
  - `pin`   = `feat/pin-table-cn` d24f02c4 (`benchmarks/pinned/gen-config.sh` + `up.sh`: the model for a full `--sirius-config` launch).
  - `s500-mig` = `bench/sf500-2-mig-gpus` 59ab07c3 off all22/integration: the GPU_DEVICES/MIG_DEVICES launcher, this plan folder (`plan-2mig/`); build, CN target and FE are symlinks into demo.
  - Work packages get NEW worktrees off the tree the plan names: `wp0/exchange-telemetry` and `wp1/bounded-frames` off mcn, `wp2/fe-statistics` off mcn (FE work),
    `wp3/streaming-shuffle` off mcn. Recipe: `git -C /home/ubuntu/sirius-wt/mcn worktree add /home/ubuntu/sirius-wt/<name> -b <branch> 2e0cbf51`; apply the GPU_DEVICES
    patch; `ln -sfn /home/ubuntu/sirius-wt/base/.pixi .pixi`; `ln -sfn /home/ubuntu/sirius-wt/base/experimental/starrocks/.pixi experimental/starrocks/.pixi`;
    `mkdir -p experimental/starrocks/starrocks/output && ln -sfn /home/ubuntu/sirius-wt/demo/experimental/starrocks/starrocks/output/fe experimental/starrocks/starrocks/output/fe`;
    a shell-only worktree also symlinks `build` and `experimental/starrocks/target` from mcn; a code worktree needs `git submodule update --init --recursive`,
    `bash experimental/starrocks/scripts/apply-starrocks-patches.sh`, then its own `pixi run make` (hours) and `cargo build --release -p sirius-starrocks-cn`.
- Engine build: `cd <tree> && pixi run make` (after a header/class-layout change `pixi run make clean` first: stale TUs caused heap corruption once).
  CN build: `cd <tree>/experimental/starrocks && pixi run bash -c 'source scripts/cn-env.sh && cargo build --release -p sirius-starrocks-cn'`.
  GPU-linked tests: `cargo test -p sirius-starrocks-cn --release -- engine_` and `cd ../../rust && cargo test -p sirius --lib`, under the GPU lock. A silently empty
  grep of test output means the binary aborted: run it under gdb (`pixi exec --spec gdb -- gdb`).
- Dataset `/home/ubuntu/tpch_parquet_sf500` -> `/opt/dlami/nvme/tpch/tpch_parquet_sf500` (132 GiB, 30 lineitem files). After a stop/start the nvme is empty:
  `for l in ~/.cache/rattler ~/.cache/sccache ~/.cargo ~/.rustup ~/.m2; do mkdir -p "$(readlink $l)"; done; mkdir -p /opt/dlami/nvme/tmp; bash ~/regen_tpch.sh 500`
  (232 s; `RESULT: FAIL` on lineitem is the known verifier-constant bug). Oracle: `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/oracle/tpch_sf500`.
- MIG instances are 47.4 GiB: CN pool + staging arena <= 45 GiB (L-A 40/4, L-B 36/8, L-C 36/4). The arena is a separate cudaMalloc outside the pool.

## 2. Tools (use these; add harness code only for the knobs the plan lists)

- `all22/gpu-lock.sh <cmd>` serializes anything touching the GPU. `capture-arm.sh` refuses to start if a CN is already running.
- `harness/capture-arm.sh <tree> <ncn> <tag> <timeout> <runs> q01..q22` one arm (cluster up, readiness, queries, cluster down, `arms/<tag>/`: runs/runs.csv,
  runs/qNN.rK.{out,err}, cluster.log, engine-.cnN.log, cnlog.txt, quent/). Env: `GPU_MEM STAGING HOST_MEM EXTRA_ENV QD FE_SETUP_SQL RESTART_ON_FAIL=1
  SIRIUS_QUERY_WATCHDOG_SECS PIN_MODE GPU_DEVICES TPCH_SF`. Driver to copy: `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus/scripts/sf500-2mig/arms-S500-mig.sh`
  (SF500, MIG, 40/4/40; oracle reuse), or `all22/rerun2-2cn.sh` for the config.txt sha256 lines.
- `harness/compare.py <arm>/runs <oracle>` oracle check (rel 1e-6) on every run; `all22/pin_compare.py <arm>...` per-query ratios/pass sets across arms;
  `harness/results_table.py tag=path/runs.csv ...` timing table; `all22/evidence.sh <arm>` restarts, stage OOM, arena exhausted, EXPORT_CAPACITY_EXCEEDED,
  credit waits, `[exchange_ingress]/[exchange_reload]/[downgrade]` counts, pack_wait_us/queue_wait_us, warm spread; `all22/compare_rounds.py`.
- `harness/cnlog_extract.py` (per-query frags + tx_bytes; today loses mapping after a restart: "no CN events mapped"), `harness/explain_summary.py` (FE plans),
  `all22/gate-a.sh` (translate-only EXPLAIN capture; hardcodes WT=dp-fix3 and the 2-GPU sizes: parameterize before use).
- Standalone comparison: `all22/arms-SA.sh` needs a MIG-sized YAML first (BOX-TRANSLATION §2).
- Launch detached: Claude Code's background shell is capped at 10 minutes; drivers run as
  `setsid nohup bash all22/gpu-lock.sh bash <driver> > logs/<tag>.log 2>&1 < /dev/null &`.

Protocol: 2 CNs one per MIG instance (`GPU_DEVICES=0,1`), SF500 kit (`TPCH_SF=500`), `FE_SETUP_SQL="SET GLOBAL cbo_cte_reuse_rate = 1.15"`, TO=300, watchdog 240,
RUNS=2, `RESTART_ON_FAIL=1`, `ASYNC=1`, oracle on every run, config.txt with commit/dirty/sha256 of the CN binary and the engine .so, the parent GPU UUID and both MIG
UUIDs, GPU_MEM/STAGING/HOST_MEM/B/frame bound/window/DISK tier in absolute bytes; same-box same-day comparisons only; never drop failed queries from a table; log every arm in
`all22/LOG.md`; kill your own processes with `pkill -f '[p]attern'`; if you kill a driver, also kill its capture-arm.sh / run-queries.sh / wait-ready.sh children and run
`harness/stop-cluster.sh <tree>` (2026-09-09: a half-killed driver ran q06 against the next driver's cluster).
