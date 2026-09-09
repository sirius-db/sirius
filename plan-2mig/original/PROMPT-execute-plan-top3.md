# Prompt: execute PLAN-top3-general.md in a fresh Claude Code session on this box

Start the session in `/home/ubuntu/sirius-wt` and paste everything below the line.

---

You are executing an engineering plan on the 2x RTX PRO 6000 box. Sirius is a GPU SQL engine (DuckDB extension, cuDF/RMM/cuCascade) running as a StarRocks
compute node (CN). The plan removes the per-CN memory blow-up created by shuffling a large table between CNs, in three table-agnostic work packages.
TPC-H SF1000 is only the validation workload; no mechanism you build may be keyed to a specific table, query or predicate.

## Read in this order (45 minutes, before any change)

1. `/home/ubuntu/sirius-wt/all22/PLAN-top3-general.md` — THE plan. Phases, files with line anchors, tests, gates, sequencing. It is authoritative; where this prompt
   and the plan disagree, the plan wins.
2. `/home/ubuntu/starrocks-tools/docs/PROMPT-top3-2cn-oom-edge.md` §1-§2 — the box, the four source trees, build/test commands and their traps, every harness tool,
   the run protocol and the lock rule. Do not re-derive any of it.
3. `/home/ubuntu/starrocks-tools/docs/CHECKLIST-2cn-oom-edge-2026-09-07.md` — what was already tried (do not repeat), what is on each branch, known defects.
4. `/home/ubuntu/sirius-wt/all22/INCIDENT-q08-wrong-answer-mcn.md` — the optimized exchange mode returns silent wrong answers in ~2% of runs; §5 lists the
   instrumentation the plan makes a prerequisite. Every arm you run compares every run against the oracle; a VALUES-DIFFER is a failure, never a pass.
5. Skim `/home/ubuntu/sirius-wt/all22/ANALYSIS-oom-edge-2cn-staging.md` §1 and §6 and `all22/REVIEW-sf1000-q08-memory-staging-analysis.md` §3-§4 for the evidence
   behind the plan, and the plan docs it builds on: `/home/ubuntu/sirius-wt/mcn/experimental/starrocks/docs/performance/{10,12,05,03,04,09,11,13}-*.md`.
6. Memory notes for this box: `/home/ubuntu/.claude/projects/-home-ubuntu-sirius/memory/` (rtxpro6000-box-layout, oom-edge-2cn-root-cause, mcn-branch-benchmark,
   perf-path04-receive-credits, standalone-vs-cn-bench).

## Ground rules

- `source /home/ubuntu/sirius-wt/env.sh` first in every shell. Trees: `demo` (all22/integration 95bec853), `perf` (perf/exchange-04-07-06 222b646a),
  `mcn` (perf/multi-cn-ingress-packing-transfer 2e0cbf51), `pin` (feat/pin-table-cn). Work on NEW branches off the tree the plan names for each work package
  (`wp1/bounded-frames`, `wp2/fe-statistics`, `wp3/streaming-shuffle`, plus `wp0/exchange-telemetry`), in a new worktree per branch
  (`git worktree add /home/ubuntu/sirius-wt/<name> -b <branch> <base>`; then `git submodule update --init --recursive`; symlink `.pixi`, `build` and the FE
  `experimental/starrocks/starrocks/output/fe` from the base tree the way `sirius-wt/prep-perf.sh` does, and only rebuild what you change). Never rebuild the FE
  unless WP2 requires it, and then plan for hours.
- Everything GPU-side runs under `all22/gpu-lock.sh`; check `nvidia-smi` and `pgrep -af 'sirius-starrocks-cn --gpu-device|StarRocksFE'` before starting a cluster;
  another agent may be using the GPUs. If `/home/ubuntu/tpch_parquet_sf1000` dangles, the box was restarted: follow the recovery in
  `/home/ubuntu/starrocks-tools/docs/PROMPT-codex-rerun-sf1000-status.md` step 1. Record GPU UUIDs in every arm's config.txt.
- Benchmark protocol and tools exactly as in PROMPT-top3-2cn-oom-edge.md §2 (capture-arm.sh via a driver copied from `all22/rerun2-2cn.sh`, `compare.py` on
  every run, `evidence.sh`, `pin_compare.py`, `compare_rounds.py`). Reference arms for gates are the same-day R2/R3 arms named in the plan; if the GPU UUIDs
  differ from theirs, run the reference arm again before claiming a speed number.
- Order of work: the plan's cross-cutting prerequisites first (telemetry and correctness instrumentation), then WP1 phase 0 (config-only), then the WPs in
  the plan's sequencing section; WP3 design note before WP3 code. Do not start a phase whose entry condition is not met; do not skip a gate; stop and report if
  a gate fails twice. Do not widen scope to other plan docs (00-17) without saying so.
- Engine changes: `pixi run make` in the worktree (`pixi run make clean` first after any header/class-layout change), then the Catch2 tags the plan names and
  `pixi run make test` before any arm. CN changes: `cargo build --release -p sirius-starrocks-cn` and `cargo test -p sirius-starrocks-cn --release`, GPU-linked
  tests under the lock. FE changes: unit tests in `fe-core`, then `harness/explain_summary.py` over all 22 plans to prove no plan-shape regression before any arm.
- Commit small, one mechanism per commit, message `wpN(scope): ...`; do not push unless asked. Keep `all22/LOG.md` current (one row per step) and write a
  short `all22/WPn-STATUS.md` per work package: phase reached, gates passed with arm names and numbers, open issues.

## Deliverable per work package

Branch + commits; tests run with counts; the arms that prove each gate (tag, config.txt, pass+MATCH count, failure classes, per-query table vs the reference,
`evidence.sh` counters); a paragraph on what changed for any table (not just the TPC-H case); and the next phase's entry state. When all gates of a WP pass,
update `/home/ubuntu/starrocks-tools/docs/CHECKLIST-2cn-oom-edge-2026-09-07.md` §3-§4 accordingly.
