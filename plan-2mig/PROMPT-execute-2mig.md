# Prompt: execute PLAN-top3-2mig.md on the g7e.4xlarge (1x RTX PRO 6000 in MIG mode)

Start the session in `/home/ubuntu/sirius-wt` and paste everything below the line.

---

You are executing an engineering plan on a single-GPU box whose RTX PRO 6000 is split into two MIG 2g.48gb instances. Sirius is a GPU SQL engine (DuckDB
extension, cuDF/RMM/cuCascade) running as a StarRocks compute node (CN). The plan removes the per-CN memory blow-up created by shuffling a large table between
CNs, in three table-agnostic work packages. TPC-H SF500 is only the validation workload; no mechanism you build may be keyed to a specific table, query or predicate.

## Read in this order (45 minutes, before any change)

1. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/BOX-TRANSLATION.md` -- every number that differs from the 2-GPU box, with derivations. Wins over every other file.
2. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/PLAN-top3-2mig.md` -- the plan as adapted; it defers to `plan-2mig/original/PLAN-top3-general.md` for mechanisms,
   file anchors, tests and principles. Read both; where they disagree on a number, the adapted one wins; on a mechanism, the original wins.
3. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/PROMPT-top3-2mig.md` -- box, trees, build/test commands and their traps, harness tools, run protocol, lock rule.
4. `/home/ubuntu/sirius-wt/s500-mig/plan-2mig/CHECKLIST-2mig.md` -- what is established on this box, what is open; then `plan-2mig/original/CHECKLIST-*.md` for
   what was tried on the 2-GPU box (do not repeat its §1-§2).
5. `/home/ubuntu/sirius-wt/all22/INCIDENT-q08-wrong-answer-mcn.md` -- optimized mode returns silent wrong answers in ~2% of runs; §5 lists the instrumentation the
   plan makes a prerequisite. Every arm compares every run against the oracle; VALUES-DIFFER is a failure, never a pass.
6. Skim `all22/ANALYSIS-oom-edge-2cn-staging.md` §1, §6 and `all22/REVIEW-sf1000-q08-memory-staging-analysis.md` §3-§4, and
   `/home/ubuntu/sirius-wt/mcn/experimental/starrocks/docs/performance/{10,12,05,03,04,09,11,13}-*.md`.
7. Memory notes: `/home/ubuntu/.claude/projects/-home-ubuntu-sirius/memory/` (g7e-single-gpu-box-layout first; the rtxpro6000 notes describe the OLD box).

## Ground rules

- `source /home/ubuntu/sirius-wt/env.sh` first in every shell. Export `GPU_DEVICES=0,1 TPCH_SF=500` for every cluster; `HOST_MEM` 40 GiB (44 ceiling); layouts
  L-A 40/4, L-B 36/8, L-C 36/4. Work on NEW branches off mcn 2e0cbf51 (`wp0/exchange-telemetry`, `wp1/bounded-frames`, `wp2/fe-statistics`,
  `wp3/streaming-shuffle`) in a new worktree per branch (recipe in PROMPT-top3-2mig.md §1); apply the GPU_DEVICES launcher patch to each; only rebuild what you
  change; the FE is rebuilt only by WP2, in its own tree, never in demo.
- Everything GPU-side runs under `all22/gpu-lock.sh`, detached with `setsid nohup` (the interactive background shell dies at 10 minutes); check `nvidia-smi` and
  `pgrep -af 'sirius-starrocks-cn|StarRocksFE'` before starting a cluster. If `/home/ubuntu/tpch_parquet_sf500` dangles the box was restarted: recovery in
  PROMPT-top3-2mig.md §1. Record the parent GPU UUID and both MIG UUIDs in every config.txt.
- Order of work = PLAN-top3-2mig.md §6: steps 0-2 need no rebuild (launcher + harness knobs, then the same-box mode-off reference arms R5-*); step 3 is the one
  shared instrumented mcn rebuild; no optimized-mode arm before it. Do not start a phase whose entry condition is not met; do not skip a gate; stop and report if
  a gate fails twice. Do not widen scope to plan docs 00-17 without saying so.
- HOST memory: any optimized-mode failure of class "HOST evacuation unavailable" gets a DISK-tier twin arm before it is attributed to a work package.
- Engine changes: `pixi run make` (clean first after header/class-layout changes), Catch2 tags the plan names, `pixi run make test` before any arm. CN changes:
  `cargo build --release -p sirius-starrocks-cn`, `cargo test -p sirius-starrocks-cn --release`, GPU-linked tests under the lock. FE changes: `fe-core` unit
  tests, then `harness/explain_summary.py` over all 22 plans before any arm.
- Commit small, one mechanism per commit, message `wpN(scope): ...`; do not push unless asked. Keep `all22/LOG.md` current and write `all22/WPn-STATUS.md`
  per work package (phase reached, gates passed with arm names and numbers, open issues). Mirror drivers and evidence into
  `/home/ubuntu/starrocks-tools-wt/sf500-2-mig-gpus` and commit there.

## Deliverable per work package

Branch + commits; tests run with counts; the arms that prove each gate (tag, config.txt, pass+MATCH count, failure classes incl. HOST layout, per-query table vs
the same-box reference, `evidence.sh` counters); a paragraph on what changed for any table; the next phase's entry state. When all gates of a WP pass, update
`plan-2mig/CHECKLIST-2mig.md` §1-§3.
