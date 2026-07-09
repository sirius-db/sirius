# Implementation Plan: General Dynamic Filters (SIP) + #1014

Companion to [issue-1010-dynamic-filter-sip-design.md](issue-1010-dynamic-filter-sip-design.md).
Track C is based on `dev` `fac81e87`, which contains merged PR #1134 at `1eecaf97`.
Original file:line citations were captured at `506a1d9f`; re-grep every anchor before editing.

This is the program-level overview. Each PR cluster has a self-contained implementation plan under
[issue-1010-plans/](issue-1010-plans/), produced against the finalized design, adversarially
reviewed against the code at the baseline commit, and corrected (each cluster doc ends with a
"Review resolution" appendix mapping every review finding to its fix).

## Status reconciliation and Track C re-evaluation (2026-07-09)

**Track A shipped delete-only; Track B is deferred. Track C starts here.**

- **A (#1014):** the build-scan priority pass was measured on a fork (results in
  [`issue-1124-results/`](issue-1124-results/), verdict PASS) and then **deleted
  outright** in PR #1134 (development commit `51da72ac`, merged on `dev` as
  `1eecaf97`). The staged scaffolding described
  below — A1 instrumentation, the A2 `dynamic_filter_build_priority={legacy,off}`
  flag, and the A3 default-flip — **was never merged.** Read every "A1/A2/A3
  will…" reference in this doc and the cluster docs as superseded by the points
  below.
- **B:** deferred — no fork/backport; executes as the pin-bump playbook when a
  released DuckDB contains duckdb#22963 (#1123).

**The following decisions are authoritative for Track C:**

1. **`publishes_dynamic_filters()` is gone.** #1134 removed it from
   `sirius_physical_hash_join.hpp` as dead code (its only caller was the deleted
   pass). Before freeze, planner/C3 bind code reads only C1a-2's immutable
   `dynamic_filter_planning_view`; after freeze, runtime reads `dynamic_filter_plan()`. Neither
   surface reintroduces the deleted boolean or exposes the mutable builder.
2. **C1a-2 owns one canonical identity vocabulary and allocator.**
   `dynamic_filter_identity.hpp` defines `dynamic_filter_publication_plan_id`,
   `dynamic_filter_target_id`, `dynamic_filter_channel_id`, and `dynamic_filter_id`; `0` is
   invalid. One executable-plan allocator mints every kind, one publication ID is memoized per
   producer, and a filter keeps the same ID across every target fan-out. C3 reuses these IDs and
   never starts a second counter domain. The same header defines strong
   `dynamic_filter_execution_generation` separately from the monotonic-clock event epoch.
3. **Candidate extraction happens once.** C1a-2 builds a generator-local immutable candidate
   cache keyed by logical join. C1b's shadow-domain snapshot is attached during that extraction;
   C3 discovery and `plan_comparison_join` read the same cached value non-destructively. Evidence
   is path-invariant: unknown means the source/cardinality is semantically unprovable or
   untraceable, never that explicit, FFI, validation, or transparent-replan planning skipped
   capture.
4. **The runtime publication plan has an explicit freeze boundary.** C1a-2 provides a planning
   view plus a one-shot installation of `shared_ptr<const dynamic_filter_publish_plan>`. C3a
   discovers/resolves and validates an immutable planning descriptor without installing runtime
   endpoints. C3b performs all grouping/allocation/validation after pipeline conversion, then
   installs prepared producer plans and C2 proof tokens with noexcept-only operations. Runtime code
   cannot observe an unfrozen or intermediate commit.
5. **The waiter-free lifecycle lost with A1 moves into C1.** Before policy enforcement or SIP,
   C1a-2 supplies the exactly-once publication-attempt FSM, reasoned terminal outcomes, stable
   filter/channel identity, and the execution-scoped reset/fresh-state contract. It covers empty
   build, unsupported mode, policy skip, unavailable source, closed targets, failure, and
   cancellation. C1b adds structured events with a monotonic sequence/timestamp, per-target
   acceptance and scan coverage, and matching analyzer support.
6. **Planning telemetry is executable-plan-only.** The plan generator carries an explicit
   `validation` versus `execution` purpose, buffers planning events, and flushes them only after
   the top-level executable plan succeeds and verifies. A failed GPU plan or transparent
   validation plan emits no accepted-plan summaries.
7. **Log-analyzer patterns + `SHAPE_VERSION` are Track C's responsibility.** Every
   "coordinate the `[dynf_summary]` vocabulary with A1 / A1 owns the
   `SHAPE_VERSION` bump" item is void. C1a preserves the existing log shape and
   does not bump it. C1b is the first PR that emits new machine-parsed lines, so
   it owns the dynamic-filter metric module, parser wiring/tests, canonical field
   names, shape-version bump, and INFO-only parser mode used by timing runs.
8. **Per-query resident high-water was A1's and never merged.** C1e and C3/C4 must not treat the
   process-lifetime `peak=` value as a per-query measurement. The owning PR adds comprehensive
   sampling, or the memory protocol runs every measured query/configuration in a fresh process and
   marks lower-bound samples explicitly.
9. **The `dynamic_filter_build_priority` flag does not exist.** C3's experiment
   matrix drops the priority×SIP dimension — "priority off" is the permanent,
   only state.
10. **C1c is retired as a standalone flagged PR.** Current publication is already per-key except
    for the fail-closed structural arity guard. C1b absorbs target compaction, per-key reason
    telemetry, and parity tests. There is no `enable_dynamic_filter_per_key_fanout` flag.
11. **Execution-scoped state is mandatory.** Immutable slots and the canonical
    descriptor/fingerprint persist with cached prepared execution data; the query-owned
    `dynamic_filter_execution_plan` is a fresh borrower for one begin/end interval and owns no
    commit bit. A reused physical plan must begin each execution with
    open, empty channels; reset gates/counters/outcomes; and bounded batch-identity tracking. The
    coordinator centrally resets each unique channel, attempt,
    and filter-ID counter exactly once, sets every channel to one strong execution generation, and
    establishes a separate clock epoch. It validates all endpoint generations before
    allocation-free/noexcept C2 begin hooks. Canonical success/abort end quiesces work, emits
    normal/partial summaries, closes channels, calls C2 local end, then cancels residual attempts.
    No component may silently reuse or independently reset closed channels or trained gates.
12. **B1 blocks nothing in Track C.** Until the pin contains duckdb#22963, LIMIT/TOP-N tests use
    explicit expected rows or a filters-disabled reference. B1 is not a merge, experiment, or
    default-on prerequisite for C1e, C3, or C4.
13. **Baseline re-anchor.** Track C builds directly on `dev` `fac81e87`; #1134 is already in
    that history as `1eecaf97`. Treat older line numbers as navigation hints only.

The A cluster docs remain the historical record; the B cluster doc is the deferred pin-bump
playbook. The design architecture is retained, while this implementation plan and the current
cluster plans are authoritative for delivery.

| Cluster doc | PRs | One line |
|---|---|---|
| [A-1014-priority-pass.md](issue-1010-plans/A-1014-priority-pass.md) | A1–A4 | Shipped **delete-only**: measured on fork (#1124, PASS), pass deleted in merged #1134 (`1eecaf97`). A1/A2/A3 scaffolding never merged. |
| [B-phase0-duckdb-backport.md](issue-1010-plans/B-phase0-duckdb-backport.md) | B1 | **Deferred** (decision 2026-07-08): no fork/backport; executes as the pin-bump playbook when a released DuckDB contains duckdb#22963 (sirius-db/sirius#1123) |
| [C1ab-adapter-foundation.md](issue-1010-plans/C1ab-adapter-foundation.md) | C1a-1, C1a-2, C1b | Adapter/preservation; C1a-2 values/cache/identity/freeze plus waiter-free lifecycle/reset; then C1b target compaction, scan coverage, telemetry, and shadow selectivity |
| [C1cde-producer-flags.md](issue-1010-plans/C1cde-producer-flags.md) | C1d, C1e | Membership selectivity enforcement after an ID-based audit; default-off unfiltered-build expansion. Former C1c is absorbed into C1b |
| [C2-probe-consumer.md](issue-1010-plans/C2-probe-consumer.md) | C2a, C2b | Behavior-neutral probe handle plus preallocated proof-token consumer; then a history-aware full mode/state reservation floor |
| [C3-routes-and-freeze.md](issue-1010-plans/C3-routes-and-freeze.md) | C3a, C3b, C4 (+ Track D sketch) | Discovery/resolution and planning-descriptor validation first; grouped prepare/noexcept-commit plus runtime fan-out/consumer second; default-on only after the measured gate |

## Dependency graph and starting order

```text
A (#1014): merged delete-only in #1134; A1/A2/A3 scaffolding never merged
B1: deferred — runs at the next DuckDB pin bump; blocks nothing (#1123)
C1a-1 ──► C1a-2 ──► C1b ──► C2a ──► C2b ─────────────────────┐
                     ├──► C1d (membership enforcement)        │
                     ├──► C1e (default-off expansion)         ├──► C3b ──► C4
                     └──► C3a (descriptor validation) ────────┘    (runtime) (measured default-on)

D (ordered activation) — contingent on C3 telemetry.  E (vNext) — after C4 + aggregate filter budget.
```

Former C1c target compaction/parity is part of C1b. C1d and C1e are independent policy experiments,
not prerequisites for C3; every C3 run records their current settings.

**Start here:** A is done on `dev`. Begin C1a-1. C2a design/test groundwork can proceed in
parallel, but merge order is C1a-2 → C1b → C2a → C2b so C2 preserves C1b's channel entries and
scan-coverage hooks. C3a follows C1b and may proceed in parallel with C2a/C2b; it produces only the
validated planning descriptor. C3b is blocked on both C3a and corrected C2b before it installs
runtime endpoints or fan-out.

**Preflight:** start Track C from `dev` `fac81e87` or a later descendant; that history already
contains #1134 at `1eecaf97`. Verify every submodule is at the gitlink recorded by the chosen
`dev` commit. The current planning branch predates the merge and has documentation/submodule
work in progress, so preserve it with a commit or stash before rebasing.

**One sequencing trap:** C1a-2, C1b, C2a, and C2b all modify
`src/op/sirius_physical_hash_join.cpp`. Land C1a-2 → C1b → C2a → C2b and rebase at each boundary.
C3b then integrates only against the frozen C1 publication-plan and C2 consumer seams. Re-grep
anchors before every walkthrough—the older line numbers are navigation hints, not patch
coordinates.

## Program-wide conventions

Every cluster doc follows these; deviations are bugs.

- **Identity.** `dynamic_filter_publication_plan_id` / `dynamic_filter_target_id` /
  `dynamic_filter_channel_id` / `dynamic_filter_id` and all ordinal spaces are strong value types;
  `0` is an invalid ID. One executable-plan allocator assigns all identities, and C3 reuses it.
  `dynamic_filter_execution_generation` is also strong: central begin sets every channel to
  the exact execution generation, while the event timestamp epoch is a separate clock value.
  Object addresses are never event identity.
- **Adapter/cache boundary.** Only the version-pinned adapter reads DuckDB candidate metadata.
  Each logical join is extracted once into the generator-local immutable cache. Planner, policy,
  and SIP discovery consume values from that cache; runtime never dereferences
  `JoinFilterPushdownInfo`.
- **Freeze/runtime boundary.** Planning values remain mutable only inside their builder. Topology
  preparation performs all allocation/validation, then a statically-noexcept commit installs
  immutable producer plans and validated consumer proof tokens before `create_query`.
  Tasks cannot observe the intermediate moves. Matching cached-plan verification reuses the
  assigned values; direct second assignment or an incompatible descriptor is an internal error.
- **Publication lifecycle.** Every enabled publication plan reaches exactly one reasoned terminal
  outcome. Policy PRs set outcomes through this common state; they do not invent log-only
  approximations. Target acceptance/closure is recorded independently from plan outcome.
- **Execution lifecycle.** Runtime channels, filters, gates, outcomes, and batch counters are
  execution-scoped. Central begin resets unique C1 channel/attempt/identity state, sets one exact
  strong generation and separate event epoch, preflights C2 endpoints, then invokes allocation-free
  local begin hooks. Success/abort end emits summaries and closes channels before C2 local end and
  residual cancellation. Prepared-plan tests change parameters or source data between executions
  and assert that no state leaks or double reset occurs.
- **Logging/telemetry.** New log lines keep the bracketed component prefix already used by their
  file (e.g. `[sirius_physical_hash_join]`). Machine-parsed per-query summaries use
  `[dynf_summary]` at INFO. Per-batch/per-split coverage lines are DEBUG/TRACE. **Every
  measurement runbook states the log level per pass**: wall-time passes run at INFO (coverage
  taken from INFO-level summary aggregates); DEBUG/TRACE passes are separate and excluded from
  timing statistics. Structured (quent) telemetry additions are optional follow-ups — new quent
  event kinds require Rust model + codegen changes, so v1 telemetry is log-based and
  `tools/log_analyzer/` grows the matching patterns. The analyzer accepts INFO-only timing logs;
  DEBUG/TRACE datasets are optional detail inputs.
- **Resident high-water.** The logged `peak=` is process-lifetime. A gate may use only
  comprehensive query-scoped sampling or fresh-process memory passes; a baseline/delta lower bound
  is diagnostic and is labeled `exact=0`.
- **Flags** (all config-file + `SET`-registered per the existing `sirius_config` pattern):
  `dynamic_filter_selectivity_gate={off,shadow,enforce}` (C1b records, C1d enforces),
  `enable_dynamic_filter_unfiltered_build` (C1e, default off), and
  `enable_dynamic_filter_sip` (C3, default off). No C1c flag exists. Flags are snapshotted into
  the plan at prepare time — the A/B unit is the executable plan, so a cached prepared statement
  must be re-prepared after a `SET` change.
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
   is absorbed into C1b's compact-target restructure and parity tests; a second runtime loop and
   chicken-bit would add risk without an independent behavior to measure.
4. **`sirius_engine::initialize` resets engine state before `initialize_internal`** — C3a's
   immutable planning descriptor must survive that reset on prepared data; only the engine's
   working registry reference is dropped after C3b staging. A C3a flag-on test asserts
   `sip_descriptor_frozen` with zero runtime endpoints, while a C3b route-bearing test asserts
   `topology_frozen sip_targets>0`. This distinguishes a valid planning-only run from the silent
   runtime no-op failure.
5. **Delim joins route through `plan_comparison_join`** — SIP discovery must explicitly reject
   `DELIM_JOIN`/`ASOF` producers (`PRODUCER_SHAPE` rejection reason).
6. **BUILD_PROBE probe-only tasks carry one input batch and predate memory history** — the C2
   estimator snapshots build rows/bytes into atomics when the build completes rather than reading
   a second input batch that isn't there. The BUILT transition publishes those snapshots with
   release/acquire ordering.
7. **History can become stale when an opportunistic channel grows.** The pipeline bypasses
   operator no-history estimates after any successful sample, so C2b adds a filter-generation and
   join-state-aware reservation floor that is evaluated even when history exists.
8. **Fresh-scan `input_stats.bytes` is already decoded projected output.** C2b preserves
   `working_set_bytes`, sizes the mask phase from decoded bytes, and does not multiply the
   retained table by the scan operator's separate `8×` execution heuristic.
9. **General INNER output is not linearly bounded by input rows.** Duplicate-heavy joins may
   produce probe×build matches. C2b uses join-path-specific multiplicity and projected row-width
   bounds; SEMI/distinct paths keep their tighter bounds.
10. **A prepared physical plan cannot reuse dynamic-filter runtime state.** The old Phase 1
    behavior is not a precedent to preserve: closed channels, filters, gate decisions, outcome
    state, and repetition counters must be fresh for each execution. One central begin resets
    unique C1 state and preflights generation before allocation-free C2 local hooks; canonical
    success/abort end emits summaries/closes channels, calls C2 local teardown, then cancels any
    residual publication attempt.

## Measurement runbooks (shared shape)

Gates use nested/star TPC-H (Q2, Q5, Q8, Q9, Q17, Q18, Q21) plus a synthetic many-join chain,
with per-pass log levels stated, results compared as bags (exact order only where SQL guarantees
it), and thresholds from measured run variance:

- **(Executed — see #1124 results, PASS.)** **#1014 gate (A2→A3):** filters disabled / legacy priority / priority off. Read: wall time,
  per-memory-space resident high-water, live pinned build batches + hash tables, feeder-task
  counts (collected in *all* configs), channel coverage before/after publication. Pass = no
  material wall-time or peak regression outside variance, with coverage explaining any movement.
- **C1d gate:** shadow / membership-enforce. Join producer decisions to the first consumer
  membership measurement by stable filter/channel ID; every first measurement records KEEP or
  SKIP. Zone maps remain shadow-only until separate row-group/AST effectiveness telemetry exists.
- **C1e gate:** unfiltered-build expansion off / on, crossed with the selected C1d mode. Report
  entry path and known/unknown domain separately, and require path parity: the same traceable
  logical source has the same evidence under transparent, explicit, FFI, validation, and
  transparent-replan planning. Unknown means semantically unprovable/untraceable evidence, not
  capture lost by an entry path. Global enablement is ineligible while unknown-domain candidates
  lack a measured cost policy.
- **SIP gate (C3→C4):** SIP off / SIP on; no priority dimension exists. Read: per-layer
  coverage (scan-caught / C1-caught / C2-caught), hash-probe rows and bytes avoided, mask/gather
  overhead and gate-disable rate, publication outcomes, resident filter bytes, per-space query
  high-water, and wall time. Default-on requires exact bag-equivalent results (order compared only
  where SQL guarantees it), valid memory evidence, and value the telemetry can attribute; route
  classes with systematic misses go to Track D consideration instead of default-on.

Timing runs use INFO-only logs and the analyzer must accept them. DEBUG/TRACE coverage-detail runs
are separate and never contribute timing samples. Memory-gate runs either use comprehensive
query-scoped sampling or launch a fresh process per query/configuration; process-lifetime peaks
from serialized runs are not acceptance evidence. Until B1 executes, LIMIT/TOP-N shapes use the
explicit-or-filters-disabled oracle rule.

## Status tracking

Suggested issue structure: one umbrella issue per track (A, B, C) referencing the cluster docs,
plus the standalone Phase 1 LIMIT/TOP-N bug issue (text in the B cluster doc). Each PR links its
cluster doc section and states its gate outcome in the PR description.

Current state (2026-07-09): **A is closed** on `dev` via merged #1134 (`1eecaf97`; measured on
fork #1124, PASS; A1/A2/A3 never merged). **B is deferred** to the next DuckDB pin bump (#1123)
and blocks nothing. **C is plan-reviewed and awaiting implementation from C1a-1.**
