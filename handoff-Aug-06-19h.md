# Handoff — Sirius distributed TPC-H work (session of 2026-08-05 → 08-07)

Repo: `/home/ubuntu/git/sirius-db/sirius-worktrees/integration`, branch `demo-multi-cn`,
HEAD `5b4cfc7a`. The user's arc: make TPC-H run distributed on the StarRocks-FE + Sirius-GPU-CN
demo cluster, then **benchmark it (engine A) against stock StarRocks (engine B)** — table + plot.

## Read these first (all repo-root, git-excluded via `sirius-worktrees/.bare/info/exclude`)

- `ROADMAP-8CN-TPCH.md` — overall feature roadmap and status (items 0, #1, #2, #4-scalar DONE).
- `PARTITIONED-OUTPUT-PLAN.md`, `BYTE-RANGE-SPLITS-PLAN.md`, `TWO-PHASE-AGG-PLAN.md` — per-stack
  decisions + progress logs with verification evidence for everything already committed.
- `REVIEW-GUIDE.md` — per-commit review guide, Parts 1–3 (Part 4 for the partitioned-output
  stack `6c7217aa..5b4cfc7a` is still TODO).
- `TPCH-SURVEY.md` — the Q1–Q22 plan survey + addenda (what blocks each query).
- `SCALAR-SUBQUERY-PLAN.md` — ASSERT_NUM_ROWS Tier 1/Tier 2 plan; NOTE: 0 assert instances in
  390 captured fragments — re-diagnose from a clean sweep before implementing anything.
- `BENCHMARK-A-VS-B.md` — benchmark methodology, fairness caveats, status log.
- Committed docs: `experimental/starrocks/DEMO.md` (up to date through hash fan-out).

## CRITICAL: uncommitted working-tree state (10 modified files)

Three overlapping pieces, all by a (now stopped) implementation agent; **none committed**:

1. **Two-phase avg** (translator crate: `partial_state.rs`, `node_translator.rs`, `lib.rs`,
   `expr_translator.rs`, `descriptor_table.rs`, `tests/translate.rs`) — COMPLETE and was green
   (13+99 translator, 95+7 CN-no-engine tests). Design: partial `avg(x)` → `sum(cast(x AS
   DOUBLE))` + `count(x)` (deterministic `__count` name suffix); merge side sums states + a
   finalize ProjectRel (`if count==0 then NULL else sum/count`); exchange schema expansion.
2. **Engine empty-partition hang fix** (`src/creator/task_creator.cpp`,
   `src/pipeline/task_scheduler.{hpp,cpp}`, ~55 lines) — COMPLETE and verified: an input
   stream that ends with zero batches (a hash partition owning no keys) used to hang
   `execute()` forever, silently; now completes. C++ suite was green (2189/1 skipped) and the
   Rust GPU harness 18/18 (`rust/crates/sirius/src/lib.rs` gained 9 watchdogged tests).
3. **HUGEINT cast fix — PARTIAL, agent stopped mid-edit.** Live gate exposed: the merge
   fragment's finalize projection passes the merged `count(*)` through as HUGEINT (DuckDB's
   sum(BIGINT) widening bypasses the engine's downcast), receiver declares BIGINT → loud C1
   guard error: `relay into stream 6 column 5 is declared BIGINT but the source sink produces
   HUGEINT`. The fix direction (agreed): in the finalize ProjectRel, cast merged integer-state
   passthroughs to their modeled wire type (I64). The agent had added the cast wrapper and was
   **mid-way updating translator tests** when stopped — `cargo test -p starrocks-plan-translator`
   is the first thing to run; expect failures to finish/align.

First actions for a fresh agent: run the translator + CN test suites, finish the cast fix +
tests, `pixi run make` (engine fix needs it — the .so at build/release may already include it),
re-run the live gate below, then commit as 2–3 reviewable commits (engine fix; avg translator;
follow docs pattern in the plan docs' progress logs). **Commits are made by the main session
only; sub-agents never commit. Full commit messages with verification evidence, per the
existing style (`git log`).**

## The live gate (definitive avg check)

```bash
cd <repo>/experimental/starrocks && pixi run cluster2   # background; wait 2 CNs alive on :9030
# then via `pixi run --manifest-path experimental/starrocks/pixi.toml bash -c 'mysql --host 127.0.0.1 --port 9030 --user root --batch -e ...'`
WITH lineitem AS (SELECT * FROM FILES("path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem/*.parquet","format"="parquet"))
SELECT l_returnflag, l_linestatus, sum(l_quantity), avg(l_quantity), avg(l_extendedprice), count(*)
FROM lineitem WHERE l_shipdate <= date '1998-09-02' GROUP BY 1,2 ORDER BY 1,2;
-- oracle (DuckDB, same file): A|F 37734107 25.522005853257337 38273.129734621674 1478493;
-- N|F 991417 25.516471920522985 38284.4677608483 38854; N|O 74476040 25.50222676958499
-- 38249.11798890827 2920374; R|F 37719753 25.50579361269077 38250.85462609966 1478870
```
Scalar `SELECT avg(l_quantity)` already verified oracle-exact (25.507967136654827).

## The benchmark (the user's active request: A-vs-B table + plot)

Assets copied session-independent to **`/tmp/sirius-tpch-bench/`**: `queries/` (22 FILES()-CTE
TPC-H queries), `bench.sh <label> <runs> [restart_cmd]` (1 warm-up + N timed runs → per-label
`timings.csv`; NOTE it writes under the OLD session scratchpad `$SP` — edit its SP variable to
a durable path first), `analyze.py` (markdown table + log-scale PNG; also edit its SP),
`setup-b.sh`, `run-live2.sh`.

- **Engine B: DONE.** Stock StarRocks 3.5.20 (binaries extracted from Docker
  `starrocks/artifacts-ubuntu:3.5.20` — release tarball URLs 403). Layout at
  `/home/ubuntu/git/starrocks-bench/` (fe/, cn1/, cn2/ **as BEs**: shared-nothing FE +
  `start_be.sh` + `ALTER SYSTEM ADD BACKEND "127.0.0.1:9050"` and `:9052`; CN mode failed
  FILES() with "No alive backends"). `JAVA_HOME=/usr/lib/jvm/java-21-amazon-corretto`.
  Results: **22/22 pass**, 3 timed runs each — `/tmp/sirius-tpch-bench/bench/B/timings.csv`.
- **Engine A: pending** the avg fix. Then: `bench.sh A 3 '<cluster2 restart cmd>'` (wedge
  isolation matters less now — the empty-partition fix made failures loud — but keep it).
  Expected A refusals: Q16 (DISTINCT agg), possibly Q2-class (unknown; see scalar-subquery
  doc). Run with B fully down (shared ports 9030/9050/9052/…, shared CPUs).
- Then `analyze.py` → table + plot → paste into `BENCHMARK-A-VS-B.md` Results section.

## Operational gotchas (hard-won)

- `pkill -f '[S]tarRocksFE'` bracket pattern (self-match kills your shell otherwise); engine B's
  processes are `starrocks_be` + the same FE class name.
- Never chain `... &` inside a `run_in_background` Bash call — the cluster dies with the shell.
  Launch `pixi run cluster2` as its own background task.
- Rust GPU tests: `pixi run bash -c 'export LD_LIBRARY_PATH=$PWD/build/release/extension/sirius:$LD_LIBRARY_PATH; cargo test --manifest-path rust/Cargo.toml -p sirius --lib -- --test-threads=1'`
  (NO /usr/bin/gcc linker pin for this crate; that pin is only for the CN crate via pixi tasks).
- Translator/CN tests: `pixi run --manifest-path experimental/starrocks/pixi.toml
  cn-test-no-engine | cn-test`, and `cargo test -p starrocks-plan-translator` inside that env.
- An AI-artifact gate hook blocks Bash when the tree holds debug scaffolds / stray root files;
  root-level working .md docs must be listed in `sirius-worktrees/.bare/info/exclude`.
- A query failing MID-EXECUTION used to wedge the cluster (restart CNs); post-fix this should
  be rare but the FE meta remembers dropped/added nodes across restarts (`ALTER SYSTEM DROP
  COMPUTE NODE` to clean stale registrations).
- `cargo fmt --check` fails on pre-existing CN files (nixl_transport.rs etc.) — do NOT
  workspace-format; format only touched crates.
- Known engine A limits: no cancel/GC (roadmap #5), DISTINCT agg refused, avg `output_columns`
  identity check risk at `compute_node_service.rs:726` (flagged, undecided).

## Suggested skills

- `pre-commit-cleanup` — run before every commit (the artifact gate enforces it anyway).
- `module-context` — load cudf/rmm/duckdb/cucascade API docs before touching engine operator
  or expression code (per repo CLAUDE.md).
- `handoff` — regenerate this document at the next session boundary.

## Task list state

Task #19 ("Implement two-phase avg + rerun the live sweep") is in_progress and maps to the
CRITICAL section above; the benchmark completion is its tail.
