# Agent handoff: implement and evaluate Track A (issue #1014) end to end

You are an agent on a performance machine. Your job is to implement, measure, and (conditionally)
land the Track A sequence for [sirius-db/sirius#1124](https://github.com/sirius-db/sirius/issues/1124)
(which resolves [#1014](https://github.com/sirius-db/sirius/issues/1014)): instrument the dynamic-filter
pipeline, add a switch for the build-priority pass, run the acceptance measurement, and flip the
default if the gate passes. Deleting the pass (A4) is **not** in this engagement — it happens a
release later.

**Read these two files before writing any code** (they should sit next to this file; if missing,
stop and ask for them):

1. `A-1014-priority-pass.md` — the authoritative implementation plan. It has exact file:line
   targets, C++ signatures, test lists, and a review-resolution appendix. Follow it. This handoff
   only sequences your work and adds machine/process guidance; where they disagree, the plan wins.
2. `../issue-1010-dynamic-filter-sip-design.md` — the design. Read § "Issue #1014" and
   § "Verified execution-model facts" for why the work is shaped this way.

## Definition of done

1. **One Track A PR merged**, containing A1 (instrumentation) and A2 (the
   `dynamic_filter_build_priority={legacy,off}` switch, default `legacy`) as clean, separately
   revertible commits: A1a, A1b, A2. Both stages are zero-behavior-change at defaults; the PR
   body carries each stage's gate evidence.
2. The three-config measurement executed per the runbook, results posted as a comment on #1124
   with the raw data attached.
3. **Stop for human sign-off.** If the human approves and the gate passed: PR A3 (default flip,
   ~10 LOC — separate because it is decision-gated on the measurement). A4 is out of scope
   regardless (release-gated).

## Ground rules

- Branch off `dev`; PRs against `sirius-db/sirius` `dev` via `gh`. A1+A2 ship as one PR with
  stage-per-commit history; A3 is its own small PR after sign-off. Reference #1124 in each PR
  body and check off its task list as you go.
- Baseline for all file:line citations in the plan is commit `506a1d9f`. `dev` may have moved —
  verify each cited site before editing it; if a site has materially changed, stop and report
  rather than guessing.
- **A1 and A2 must not change behavior** (A1: no dispatch/publication change; A2: `legacy`
  default is byte-identical scheduling). The merge gates in the plan define the proof. Treat any
  behavioral diff you introduce as a bug in your change.
- Do not start Track C work (no `dynamic_filter_publication_plan` value-type migration, no
  adapter, no consumer changes). Another machine owns that. The one file both tracks edit is
  `src/op/sirius_physical_hash_join.cpp` — A1 lands first by agreement, so you are not blocked,
  but keep your edits there minimal (the plan already scopes them).
- **Cross-machine coordination item:** A1 introduces `src/include/op/dynamic_filter_ids.hpp` and
  the `[dynf_summary]` field vocabulary. After A1 merges, post a short comment on #1124 listing
  the header path, the enum/ID names, and the exact summary field names — the C-track implementer
  consumes them.
- Run `pixi run pre-commit run -a` before each push. New test files must be registered in
  `TEST_SOURCES` (enforced by the orphan-test hook).

## Setup checklist

```bash
git clone --recursive git@github.com:sirius-db/sirius.git && cd sirius   # or update existing
git checkout dev && git pull && git submodule update --init --recursive
pixi run make            # full build
pixi run make test       # sanity: suite green before you change anything
```

Machine notes (apply if this is the GB10 box `pdx02-zeno-01`; otherwise adapt):

- Single NVIDIA GB10, unified memory (~119 GB), arm64. Single-GPU only.
- TPC-H datasets in `~/Code/sirius_1/test_datasets/` (`tpch_parquet_sf30/`, `tpch_sf30.duckdb`,
  `tpch_sf50.duckdb`). Query files in `scripts/` (`tpch-queries-run.sql`, `run-tpch.sh` — read the
  script before using it).
- `~/.sirius/sirius.yaml` may be stale (unknown-key startup error) — always set
  `SIRIUS_CONFIG_FILE` explicitly. Default `usage_limit_fraction` (0.95) fails allocation on
  GB10; use ~0.3–0.55.
- Do not put trivial marker queries (`select 42`) in benchmark scripts — CPU_SOURCE→DUMMY_SCAN
  pipelines have hung under GPU interception on this box.
- Log analysis: run with `SIRIUS_LOG_LEVEL=trace` + `SIRIUS_LOG_DIR`, then
  `python3 tools/log_analyzer/parse_logs.py <log>`.

## Stage 1 — commits A1a/A1b (instrumentation)

Implement exactly per `A-1014-priority-pass.md` § "A1", as two clean commits: **A1a** (IDs +
outcome/channel/coverage events + analyzer) then **A1b** (high-water sampler + feeder/lifecycle
counters + summary) — each ~450–550 LOC and independently revertible.

The plan is detailed; these are the mistakes it specifically armors against — do not reintroduce
them:

- The feeder running gauge uses the RAII `feeder_running_scope` at worker-lambda entry. The
  lambda has **seven** exit paths; manual decrements leak.
- `close_for_new_filters`: the `exchange` stays **inside** the existing `_mu` scope; log after
  unlock. Same compute-under-lock / emit-after-scope split in `on_finalize_operator`
  (`op_state_mutex`). Never log under a lock.
- `probe_target::target_id` is appended **last** (positional init at
  `sirius_plan_comparison_join.cpp:447` must keep compiling).
- Normal zero-delivery completion reports `NO_MATERIALIZATION(NO_BUILD_DELIVERY)`; `CANCELLED`
  is never emitted in A1.
- No peak-reset calls on memory resources — high-water uses the baseline/delta scheme in the plan.
- Per-batch events are DEBUG/TRACE; per-query summaries are INFO `[dynf_summary]`; every event
  line carries the `[dynf] ` anchor. The analyzer (`tools/log_analyzer`) gets its module +
  `SHAPE_VERSION` bump **in the same PR**.

**Stage gate (prove all of it, put the evidence in the PR body):** full `pixi run make test`
green; one TPC-H A/B run (A1 vs base) with identical results and identical
`Pushed {} dynamic filter(s)` line multisets; `[dynf_summary] feeder running_end=0` on every
query; log_analyzer reports no `FormatWarnings` for the new patterns — this last check requires
**one untimed DEBUG-level run** so the per-batch patterns actually appear in the log.

## Stage 2 — commit A2 (the switch), then open the PR

Implement per `A-1014-priority-pass.md` § "A2" as the final commit, then open the single Track A PR. The one thing that matters most:

- The flag gates **only** the priority `pop_if` dispatch branch. `collect_filter_build_pipelines`
  and the feeder-telemetry install run unconditionally in both modes — the measurement needs
  feeder metrics under `off`. Under `off`, dispatch lines show `feeder=1 prioritized=0`.
- Mode is snapshotted once per query at `prepare_for_query` (`_priority_dispatch_enabled`), so a
  mid-query `SET` cannot tear the dispatch loop.

**Stage gate (also in the PR body):** default `legacy` is behaviorally identical; the CPU unit tests + the GPU
integration test from the plan pass (`filter_build_pipeline_count_for_testing() > 0` in both
modes; `priority_dispatch_enabled_for_testing()` flips).

## Stage 3 — the measurement (the actual point of this engagement)

This is the #1014 acceptance gate. Follow `A-1014-priority-pass.md` § "A2→A3 measurement protocol"
exactly. Summary of the protocol — the plan is authoritative on details:

**Configs** (each in its own process; `LOAD` the extension before any `SET` — a pre-LOAD `SET` is
silently dropped; never interleave connections):

1. `SET enable_dynamic_filter_pushdown=false`
2. `SET enable_dynamic_filter_pushdown=true; SET dynamic_filter_build_priority='legacy'`
3. `SET enable_dynamic_filter_pushdown=true; SET dynamic_filter_build_priority='off'`

**Workload:** TPC-H Q5, Q7, Q8, Q9, Q21 at SF≥10 (SF30 on the GB10 box), plus a synthetic
many-join chain (≥6 chained joins over one fact table — write it, check it into the benchmark
script, no trivial marker queries).

**Passes:**

- **Timing passes at INFO** (the default log level): ≥7 iterations per query per config, first
  discarded. All gate metrics come from these: wall time (QueryBegin/QueryEnd segments),
  per-space resident high-water (note the `exact=` flag), feeder queued/dispatched/prioritized/
  running-hwm, build batches delivered, pinned-build/hash-table hwm, replica-bytes hwm, and
  per-channel pre/post-publication coverage from the INFO `[dynf_summary] channel_coverage`
  aggregates.
- **Forensics passes at DEBUG**, 1 iteration per (query, config), run after the timing passes,
  **excluded from all wall-time statistics**. Only for diagnosing coverage deltas.
- Record the log level of every pass in a run manifest. Keep all raw logs.

**Results comparison:** config-2 vs config-3 outputs bag-compared (exact order only for ORDER BY
queries); config-1 is the no-filter result oracle for both.

**Pass criterion:** config-3 shows no material wall-time or resident-peak regression vs config-2
outside config-2's own run-to-run spread, results equivalent, and the coverage numbers explain
any movement. Coverage is diagnostic, not a veto.

**Report:** post a comment on #1124 with: the verdict (pass/fail per criterion), a per-query
table (median wall time, peak bytes, coverage pre/post-publication) for all three configs, the
run-variance you measured, machine + SF + commit SHAs, and attach the analyzer outputs. Then
**stop and wait for human sign-off**.

## Stage 4 — PR A3 (only after human approval of the Stage 3 report)

One-line default flip to `OFF` + doc/release note per the plan. Rollback stays available via
`SET dynamic_filter_build_priority='legacy'` for one release. Do **not** proceed to A4.

## When to stop and ask a human

- Any plan-cited call site that no longer matches the code in a way that changes the approach.
- Any A1/A2 gate check that fails and isn't clearly your bug.
- Measurement results that are ambiguous against the pass criterion (e.g., regression inside
  variance on some queries but not others) — post the data, propose a reading, wait.
- Anything that tempts you to change publication behavior, dispatch order (beyond the flag), or
  Track C files. That temptation means scope creep; stop.
