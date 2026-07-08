# Implementation Plan: General Dynamic Filters (SIP) + #1014

Companion to [issue-1010-dynamic-filter-sip-design.md](issue-1010-dynamic-filter-sip-design.md);
baseline `dev` commit `506a1d9f`.

This is the program-level overview. Each PR cluster has a self-contained implementation plan under
[issue-1010-plans/](issue-1010-plans/), produced against the finalized design, adversarially
reviewed against the code at the baseline commit, and corrected (each cluster doc ends with a
"Review resolution" appendix mapping every review finding to its fix).

| Cluster doc | PRs | One line |
|---|---|---|
| [A-1014-priority-pass.md](issue-1010-plans/A-1014-priority-pass.md) | A1–A4 | Instrumentation (IDs, waiter-free publication outcomes, resident high-water, lifecycle counters, coverage), then measure → disable → delete the build-priority pass. Packaging: A1+A2 = one PR (stage-per-commit); A3/A4 separate (decision- and release-gated) |
| [B-phase0-duckdb-backport.md](issue-1010-plans/B-phase0-duckdb-backport.md) | B1 | **Deferred** (decision 2026-07-08): no fork/backport; executes as the pin-bump playbook when a released DuckDB contains duckdb#22963 (sirius-db/sirius#1123) |
| [C1ab-adapter-foundation.md](issue-1010-plans/C1ab-adapter-foundation.md) | C1a, C1b | Version-pinned DuckDB candidate adapter, Sirius value snapshots, publisher + claim decoupling; strong target-key types, materialization telemetry, shadow selectivity signal |
| [C1cde-producer-flags.md](issue-1010-plans/C1cde-producer-flags.md) | C1c–C1e | Target-key entry restructure (fail-closed arity retained), selectivity-gate enforcement A/B, blanket `build_side_has_filter` gate removal A/B |
| [C2-probe-consumer.md](issue-1010-plans/C2-probe-consumer.md) | C2 | `probe_batch_handle` refactor, `hash_join_probe_filter_consumer` component, shared gated-mask extraction, mode-aware memory estimator |
| [C3-routes-and-freeze.md](issue-1010-plans/C3-routes-and-freeze.md) | C3, C4 (+ Track D sketch) | Discovery/lineage/topology-freeze registry, SIP channels + layered targets behind `enable_dynamic_filter_sip`, C3 experiment, C4 default-on gate |

## Dependency graph and starting order

```text
A1 ──► A2 ──► A3 ──► A4                      (#1014: instrument → switch → default-off → delete)
B1: deferred — runs at the next DuckDB pin bump; blocks nothing (#1123)
C1a ──► C1b ──► C1d(enforce)                 (adapter → strong types + shadow signal → enforce)
  │        │
  │        └──► C1c (restructure; compat-gated)
  ├──► C3 ◄── C2                             (registry needs C1a's values; consumer needs C2)
  │      │
  │      └──► C4 (default-on; gate also consumes A3's measured state)
  └── A1 IDs feed C1a telemetry fields (soft: land A1 first)

D (ordered activation) — contingent on C3 telemetry.  E (vNext) — after C4 + aggregate filter budget.
```

**Start immediately, in parallel:** A1, C1a-groundwork, C2-groundwork. (B1 is deferred to the next pin bump.)

**One sequencing trap:** A1, C1a, and C2 all modify `src/op/sirius_physical_hash_join.cpp`
(A1: publication outcome instrumentation; C1a: claim-condition rewrite at `:1333`/`:1364` +
publisher decoupling; C2: the probe-path `probe_batch_handle` refactor, atomics, estimator).
Land them in that order — A1 → C1a → C2 — to avoid rebase churn.

## Program-wide conventions

Every cluster doc follows these; deviations are bugs.

- **Identity.** `dynamic_filter_publication_plan_id` / `target_id` / `channel_id` / `filter_id`
  are query-relative monotonic values minted at plan time (design § "Publication, target, channel,
  and filter identity"). Object addresses are never event identity. A1 introduces the IDs; C1a
  extends events with the fields A1 defers (`target_id` on `channel_filter_visible`,
  `replica_bytes`/`devices` on `publication_completed`). **Open coordination item:** the
  `[dynf_summary]` field vocabulary (plan/target IDs, `keys_admitted` split across planner and
  ctor lines) must be reconciled between the A1 and C1b PRs before C1b merges — both cluster docs
  flag it.
- **Logging/telemetry.** New log lines keep the bracketed component prefix already used by their
  file (e.g. `[sirius_physical_hash_join]`). Machine-parsed per-query summaries use
  `[dynf_summary]` at INFO. Per-batch/per-split coverage lines are DEBUG/TRACE. **Every
  measurement runbook states the log level per pass**: wall-time passes run at INFO (coverage
  taken from INFO-level summary aggregates); DEBUG/TRACE passes are separate and excluded from
  timing statistics. Structured (quent) telemetry additions are optional follow-ups — new quent
  event kinds require Rust model + codegen changes, so v1 telemetry is log-based and
  `tools/log_analyzer/` grows the matching patterns.
- **Resident high-water.** Per-query peak requires calling
  `reset_peak_allocated_bytes` at QueryBegin (the currently logged `peak=` is process-lifetime);
  the A-cluster doc owns that change and its interaction with the existing pool-stats lines.
- **Flags** (all config-file + `SET`-registered per the existing `sirius_config` pattern):
  `dynamic_filter_build_priority={legacy,off}` (A2; gates only the dispatch preference — the
  feeder-pipeline set is always collected so telemetry stays live in all configs),
  `dynamic_filter_selectivity_gate_mode={off,shadow,enforce}` (C1b records, C1d enforces),
  the C1e candidate-expansion flag (default off), and `enable_dynamic_filter_sip` (C3, default
  off). Flags are read at plan/prepare time — the A/B unit is the query, and runbooks toggle via
  `SET` between iterations.
- **Oracles.** An unpatched DuckDB CPU run is never a correctness oracle for
  LIMIT/TOP-N-shaped tests (it shares the pinned bug); use explicit expected rows or a
  filters-disabled reference run.
- **Fail-closed.** A runtime `probe_info`-arity mismatch means structural corruption: the
  whole-target skip stays, in every flag mode. Per-key suppression applies only to
  independently-gated components of a structurally valid target.

## Empirical findings that shaped the plans

These came out of executing the review protocol against the baseline (details and transcripts in
the cluster docs):

1. **The pinned LIMIT/TOP-N bug is masked at default settings** by `late_materialization`
   (rewrites LIMIT/TOP-N-over-scan into a rowid semi-join whose scan target is legal),
   `compressed_materialization` (its internal compress projections kill the pushdown — partly a
   small-integer test artifact), and pipeline-timing races. Runtime wrong results were reproduced
   only in a dependency-ordered shape with both optimizers disabled. Consequently B1's
   deterministic red→green gate is **metadata-level** (`probe_info` empty vs non-empty with
   `late_materialization` disabled in the harness), runtime rows are patched-pin regressions, and
   the Phase 1 bug issue is framed as latent-but-real, not actively-wrong-at-defaults. This
   grounded the decision (2026-07-08) to defer the backport entirely and wait for a released fix
   (#1123); the oracle rule below still applies while on the unpatched pin.
2. **DuckDB pushes range comparisons into `join_condition`**, not just equalities — the adapter
   keeps full candidate arity and marks non-equality ordinals per-key not-admitted, mirroring
   today's runtime skip. Treating them as malformed would silently change plan shape for
   eq+range joins.
3. **Today's publisher fan-out is already per-key** except the fail-closed arity guard — so C1c
   is a restructure with a compatibility gate, not a behavior-changing A/B.
4. **`sirius_engine::initialize` resets engine state before `initialize_internal`** — the C3
   registry must be released after topology finalization inside `initialize_internal`, never in
   `reset()`, and a flag-on test must assert the `topology_frozen` event exists (the failure mode
   is a silent no-op that green tests would not catch).
5. **Delim joins route through `plan_comparison_join`** — SIP discovery must explicitly reject
   `DELIM_JOIN`/`ASOF` producers (`PRODUCER_SHAPE` rejection reason).
6. **BUILD_PROBE probe-only tasks carry one input batch and predate memory history** — the C2
   estimator snapshots build rows/bytes into atomics when the build completes rather than reading
   a second input batch that isn't there.

## Measurement runbooks (shared shape)

Both gates use the same three-configuration protocol on nested/star TPC-H (Q2, Q5, Q8, Q9, Q17,
Q18, Q21) plus a synthetic many-join chain, with per-pass log levels stated, results compared as
bags (exact order only where SQL guarantees it), and thresholds from measured run variance:

- **#1014 gate (A2→A3):** filters disabled / legacy priority / priority off. Read: wall time,
  per-memory-space resident high-water, live pinned build batches + hash tables, feeder-task
  counts (collected in *all* configs), channel coverage before/after publication. Pass = no
  material wall-time or peak regression outside variance, with coverage explaining any movement.
- **SIP gate (C3→C4):** SIP off / SIP on (× priority legacy/off where relevant). Read: per-layer
  coverage (scan-caught / C1-caught / C2-caught), hash-probe rows and bytes avoided, mask/gather
  overhead and gate-disable rate, publication outcomes, resident filter bytes, wall time.
  Default-on requires bit-equivalent results and value the telemetry can attribute; route classes
  with systematic misses go to Track D consideration instead of default-on.

## Status tracking

Suggested issue structure: one umbrella issue per track (A, B, C) referencing the cluster docs,
plus the standalone Phase 1 LIMIT/TOP-N bug issue (text in the B cluster doc). Each PR links its
cluster doc section and states its gate outcome in the PR description.
