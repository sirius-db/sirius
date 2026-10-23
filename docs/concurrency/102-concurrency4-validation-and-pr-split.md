# concurrency4: implementation, validation and review guide

> The approved plan below has been implemented as a draft PR stack. See
> [the extraction record](105-pr-stack-extraction.md) for actual branch/commit identifiers,
> validation results, source equivalence and remaining qualification gates.
> The 2026-10-08 reorder moves memory progress (#2001) after the benchmark runner (#2015);
> the extraction record contains the current order. The plan below retains its historical order.

Work started 2026-09-29; original validation resumed 2026-09-30. The rebase and
full-stack split proposal below were updated **2026-10-05**. Review this alongside
[the commit journal](101-concurrency4-implementation-journal.md),
[the simplification journal](104-simplification-implementation.md), and
[the runtime contract](../super-sirius/concurrent-queries.md).

## Branch and scope

The extraction scope is **all 33 commits** originally in
`ca4f700ccd64611d9c4a7d4e00092cf0004f37b3^..f9234950b5cb6d57136e594e178f265292ed4f47`:
the six `concurrency3_0` commits, the `concurrency3_0_1` lease/guard commit, and the
26 subsequent commits through the four larger simplifications. Earlier versions
of this plan concentrated on the last branch and left the inherited work unmapped.

On 2026-10-05, fetched `origin/main` and rebased all three branches bottom-up onto
`c2f7f3a7539bf5f19717090c5c8ef720f8fb3aab`. Local `main` already matched that tip.
The old base was `ce7d947b022a3371857bb9228c3038f4aef3d773`; main added 12 commits.

| Branch | Before rebase | After rebase | Preserved backup branch |
|---|---|---|---|
| `concurrency3_0` | `a4db7d6be` | `987ca5b53` | `concurrency3_0_before_stack_rebase_20261005` |
| `concurrency3_0_1` | `2f7430f29` | `c5261bac7` | `concurrency3_0_1_before_stack_rebase_20261005` |
| `concurrency4` | `f9234950b` | `7f1615c20` | `concurrency4_before_stack_rebase_20261005` |

All 33 commits replayed without conflicts or dropped commits. `git range-diff`
shows 31 unchanged patches and two patches whose displayed differences are only
context around main's NVTX/configuration additions. Before this plan edit, the
net change against the new main is **131 files, 7,530 insertions, 2,210 deletions**.
These are whole-series counts, not estimates for individual PRs.

The original 2026-09-29 tip remains preserved as
`concurrency4_before_rebase_20260929`. No stack was initialized, no extraction
branches were created, and nothing was pushed or submitted during this planning pass.
The scope does not include the separate benchmark-runner work on `concur_test`.

The implementation targets separate DuckDB connections sharing one DatabaseInstance. Admission
is configured with startup `sirius.max_concurrent_queries`, default **1**. It includes query
initialization and retirement. Maintenance remains exclusive. Ordinary failures are query-local;
a fatal CUDA context or unprovable cleanup invalidates the shared runtime.

**This is an implementation and one-GPU validation result, not completed release qualification.**
The available host has one RTX PRO 6000 Blackwell GPU. Two-GPU concurrency and peer-copy fallback
have not been run. Keep that gate visible when deciding whether to advertise multi-GPU support.

## Mapping to the investigation

Numbers below refer to entries in the commit journal, not proposed PR numbers.

| Finding in plan section 3 | Implementation | Journal entries |
|---|---|---|
| A: admission versus maintenance | Bounded FIFO query permits, shared planning, exclusive maintenance, cancellable waits and closing protocol | 6 |
| B: gaps between queues and workers | Submission guards and continuous work leases through creator, scheduler, dispatch, retry and completion; retained physical plan | 1–2 |
| C: cross-query cleanup/errors | Query-specific rollback/retirement; shared managers continue; completion observers and fatal-runtime health | 2, 9, 11–12 |
| D: repository and spill borrowing | Actual victim leases, no global downgrade drain, Tier-2 task ownership through conversion/return | 2 |
| E: memory progress | Nonblocking reservations, parked tasks release worker capacity, bounded retries/timeouts, real HOST result reservations, conservative sort caps | 3, 6 |
| F: FIFO and readiness | Atomic oldest-compatible selection; retry eligibility; readiness consumed only on accepted dispatch; explicit ID exhaustion | 1, 3, 6 |
| G: settings and observability | Connection-local overrides, immutable execution snapshots, query-owned telemetry, diagnostics and synchronized logging | 4–6, 11, 13 |
| H: pin/cache/scan/prefetch | Complete pin generations; execution-local compression plan selection; connection-local Iceberg memo; shared I/O budget; separate producer/coalescer pools | 4–5, 8, 10 |
| I: streams and multiple GPUs | Exclusive runtime stream leases; admitted-device prefetch; thread-local decoder device cache; fatal prefetch health propagation | 7, 12 |

### Deliberate choices

- Keep the existing lifecycle registry and extend its use. `accepts_work()` remains advisory;
  guarded publication and leases provide the actual lifetime guarantee.
- Keep repositories owned by their query manager. Victim leases protect raw repository use;
  no broad cuCascade shared-repository API migration was required.
- Use a 31-bit exhaustion guard rather than widening packed priorities in this series.
- Use short scheduler retry deadlines rather than blocking GPU management threads or moving
  blocking reservations into a finite worker pool. Reservation no-progress fails after 30 seconds.
- Demand I/O can borrow beyond a speculative budget; outstanding demand debt suppresses further
  speculation. This prioritizes progress over treating prefetch limits as hard demand limits.
- Keep per-thread reservation tracking for concurrent SQL. Reject per-stream tracking with N>1
  because stream changes/reset semantics in the underlying tracker need a separate contract.
- Pin/unpin/reset/index mutation drains existing users. Complete generations improve publication
  safety but do not introduce online SQL pin replacement.
- Concurrent FFI fragments, independent DatabaseInstances sharing allocator state, disjoint GPU
  allocation and multi-GPU late materialization remain outside this SQL concurrency contract.
  Nested retained execution windows fail explicitly rather than waiting for their own permit.

## Builds and tests actually run

### Rebase verification (2026-10-05)

The rebase preserved all source commits and the three branch relationships.
cuCascade was synchronized to main's recorded `12f284084` revision, then
`pixi run make` passed. The focused test run passed **143 cases / 1,967 assertions**,
including concurrent SQL watchdog subprocesses, lifecycle/retirement, scheduler,
pin epochs, device health, configuration and main's expression-fallback changes:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[query_lifecycle_gate],[query_retirement],[concurrent_queries],[task_scheduler],[pin_registry_epoch],[device_health],[config]~[backend],[expression_plan_fallback]'
```

The backend exclusion avoids the existing two-GPU configuration gate on this
one-GPU host. Formatting/lint and `git diff --check` passed. Logs are
`/tmp/concurrency-stack-rebase-20261005-final-build.log`,
`/tmp/concurrency-stack-rebase-20261005-tests.log`, and
`/tmp/concurrency-stack-plan-20261005-format.log`.

This check builds the rebased tip, not every rebased intermediate commit or any
proposed PR. Per-layer validation is part of the extraction procedure below.

### Earlier implementation validation

Every implementation commit was built before committing. Compiler failures were followed by
`pixi run make clean` and a clean rebuild. The journal records the checks for each increment.
The full build includes the extension, DuckDB executable and C++ unit-test executable.

These are separate invocations with overlapping coverage; **do not add their case counts**.
They are targeted suites, not a claim that every test in the repository passed.

| Validation | Result |
|---|---|
| Full clean build after concurrency/prefetch integration | Passed, 1,199 build steps |
| Final clean build including the explicit multi-GPU qualification target | Passed, 1,200 build steps |
| Final one-GPU concurrent SQL/logging rerun after the all-device pressure harness change | 2 parent cases / 107 assertions passed, including 20 SQL scenarios |
| Configuration (excluding two-GPU backend gate), scheduler, creator, Iceberg, dynamic filters, telemetry, pinned MVCC inserts | 523 cases / 26,061 assertions passed |
| Concurrent SQL, logging, prefetch, downgrade, lifecycle, completion, device health, shared I/O budget, dispatcher | 50 cases / 483 assertions passed |
| Query-owned batch telemetry and exhausted spill capacity | 8 cases / 57 assertions passed |
| Pin generation, MVCC, uniqueness and statistics regressions | 60 cases / 610 assertions passed |
| Dispatcher rejection and a 10,000-task pending chain | Included in 16 cases / 225 assertions passed |
| Worker-pressure watchdog with the existing TPC-H parquet fixture | 1 case / 6 assertions passed |
| CPU admission smoke under ASan/UBSan and TSan | Passed |
| Shared scan budget under ASan/UBSan | 3 cases / 21 assertions passed |
| Registry register/retire plus diagnostic observer under ASan/UBSan and TSan | 2,000 cycles passed |

The SQL harness forces genuine overlapping execution windows with a barrier, checks admission
occupancy and N+1 queueing, disables CPU fallback, checks results against expected/CPU values,
and verifies no query registrations remain after retirement. Its 20 one-GPU scenarios cover:

- N=2 and N=4; queued and active cancellation; one and two simultaneous failures.
- Maintenance/reset/unpin waiting; more scans than producer threads.
- Mixed join, aggregation and sort; different MVCC snapshots; HOST/GPU pins and actual
  compressed HOST/GPU chunks; GPU prefetch on HOST scenarios.
- Shared local-parquet cache; prepared re-execution and 80 executions through reused connections.
- Exhausted GPU reservation capacity with observable memory waiters, prompt cancellation and
  the terminal no-progress timeout; peers and subsequent queries complete after capacity returns.

Spill component tests exhaust HOST and configured DISK reservation capacity, verify that refused
conversion preserves the GPU source, then release DISK capacity and retry successfully. They do
not fill the filesystem or simulate every possible I/O error. Existing S3 harness tests passed
in the broad suite; a dedicated overlapping remote-I/O stress run was not performed.

### Reproduction commands

Run from the repository root in the Pixi environment. GPU tests need access to the NVIDIA device.

```bash
pixi run make
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[config]~[backend],[task_scheduler],[task_creator],[iceberg],[dynamic_filter],[telemetry_context],[pin_table_mvcc_insert]'
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[concurrent_queries],[concurrent_logging],[memory_prefetcher],[downgrade_disk],[query_lifecycle_gate],[completion_handler],[device_health],[shared_scan_budget],[scoped_dispatcher]'
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[batch_query_ownership],[downgrade_disk]'
pixi run env SIRIUS_TEST_TPCH_DIR=test/cpp/integration/data/parquet build/release/extension/sirius/test/cpp/sirius_unittest 'worker pressure leaves bounded CPU capacity'
```

The backend exclusion avoids an existing test requiring two physical GPUs; it is not a waiver
of backend qualification. On a suitable host run that gate as well as the explicit concurrency
target below. The explicit target requires two devices and fails rather than silently skipping:

```bash
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[concurrent_queries_mgpu]'
pixi run build/release/extension/sirius/test/cpp/sirius_unittest '[backend]'
```

The multi-GPU target reuses watchdog children with `integration-2gpu.yaml`: N=2/N=4, mixed
operators, simultaneous errors, cancellation, ordinary/compressed HOST/GPU pins, HOST prefetch,
cache, all-device reservation pressure and repeated execution. It is a starting qualification
matrix; success does not by itself prove the peer-copy fallback was exercised.

Development logs are in `/tmp/concurrency4-*.log` on this host. Those temporary logs and standalone
sanitizer harnesses are not durable repository artifacts; the regression tests and results above
are the review record.

## Remaining release gates

1. Run the explicit two-GPU matrix and existing multi-GPU operator/routing tests on actual
   hardware. Verify sharing/spanning device placement, concurrent spill and dynamic-filter
   publication, and both peer-copy and host-staging fallback paths. Capture device topology and
   per-device activity; a successful scalar result alone does not establish placement coverage.
2. Exercise fatal CUDA faults in disposable subprocesses on a suitable GPU test host. Current
   classification/health tests establish control-plane behavior, not recovery from a physically
   poisoned CUDA context. GPU memory sanitizers have not been run.
3. Before production release, expand long-duration workload and fault testing: concurrent remote
   I/O/cancellation, actual filesystem failure, initialization/cleanup allocation failures and
   repeated mixed TPC-H workloads with memory-use tracking. The current 80-execution regression
   detects retained query registrations; it is not a proof against every device/host resource leak.
4. Review the chosen scope and policy constants: SQL connections on one DatabaseInstance, default
   N=1, 30-second reservation deadline, maintenance exclusivity and explicit unsupported FFI scope.

## Proposed full-stack extraction — for review

### Recommendation and ordering

Use **11 implementation PRs plus one documentation/audit PR**, in one linear
`gh-stack` stack. This replaces the old six-PR proposal. Separating the lifecycle
contract from its integration, and settings from metadata and pin publication,
gives reviewers smaller subjects to reason about. The final audit PR preserves
all development reports without making them prerequisites for reading the code.

The implementation PRs include their own tests, API documentation and relevant
architecture updates. PR 12 does not postpone correctness tests or the user-facing
concurrency contract. PR 11 enables N>1 only after every safety prerequisite is
present. Until then, preserve main's serialized execution-window behavior; a
configured capacity value or a scan-pool size is not permission for overlapping SQL.

Extract the **final implementation**, folding fixes and simplifications into the
PR that introduces the affected behavior. Do not publish temporary nullable
registry wiring, metadata mutation APIs, or a second retirement sequence only to
remove them in a later PR. Historical commits are provenance, not cherry-pick units.
The new top's tree must match `7f1615c20` plus this reviewed plan update.

Every branch below starts with `stacked/concurrency-`; the table gives the suffix.
Each PR targets the preceding branch; PR 1 targets `main`. The logical dependencies
column identifies why features need each other, independently of that linear order.

| PR / branch suffix | Proposed title | Principal logical dependencies |
|---|---|---|
| 1. `01-query-errors-workers` | `fix(exec): make worker accounting and dispatch failure safe` | Main |
| 2. `02-lifecycle-contract` | `feat(exec): track query publishers and work lifetimes` | Main; consumed together with 1 by PR 3 |
| 3. `03-lifecycle-integration` | `fix(exec): retain query resources through publication and retirement` | 1–2 |
| 4. `04-memory-progress` | `fix(exec): yield workers while queries wait for memory` | 3's task ownership and retirement |
| 5. `05-session-options` | `fix(config): snapshot connection-local execution options` | Existing serialized execution window |
| 6. `06-metadata-ownership` | `fix(metadata): retire shared bookkeeping by query owner` | 3's retirement phases |
| 7. `07-pin-publication` | `fix(pin): publish complete immutable pin generations` | 5's compression policy, existing maintenance serialization |
| 8. `08-cuda-streams` | `fix(cuda): lease runtime streams and prefetch admitted devices` | 3's lifetime binding, 5's GPU subset snapshot |
| 9. `09-scan-capacity` | `fix(scan): coordinate shared I/O budgets and producer capacity` | 1's dispatcher safety, 3's producer retirement |
| 10. `10-runtime-health` | `feat(exec): report query activity and contain fatal runtime errors` | 3–4 and 8–9 for complete failure propagation |
| 11. `11-concurrent-admission` | `feat(exec): admit bounded concurrent SQL queries` | All implementation PRs above |
| 12. `12-review-record` | `docs(concurrency): record implementation and qualification evidence` | Completed implementation and extraction results |

### PR contents and review questions

#### 1. Worker accounting and safe failure reporting

Workers need per-query attribution before another thread can wait for them.
Introduce `bounded_thread_pool` query accounting, exception-safe
`scoped_dispatcher` / `thread_pool` submission and construction, and the surviving
creator start/stop synchronization fixes. Include the bounded crash handler and
core-dump ignore rules as a small, explicitly identified process-failure diagnostic
fix. Main already has per-query creator maps and completion handlers; reuse them.
Only their remaining failure-reporting changes belong in this series.

Read primarily `creator/task_creator.*`, `pipeline/completion_handler.hpp`,
`exec/{bounded_thread_pool,scoped_dispatcher,thread_pool}.hpp`, and their callers.
Retain existing serialized SQL admission and engine/executor retirement until
PR 3. In particular, do not switch to per-query-only draining before continuous
leases cover tasks between queues and workers. The pool primitives can be tested
independently here; publication and cleanup callers migrate together in PR 3.

Gate: completion isolation, creator state, per-query worker waits, rejected
submission accounting and the nonrecursive pending-work chain. Stop/join must
never hold a mutex needed by the worker. Runtime recovery after query failure is
checked with the complete cleanup integration in PR 3.

#### 2. Lifecycle registry contract

Introduce the registry's final core contract: registered/open versus quiescing
queries; refusing unknown IDs; move-only submission guards and work leases;
retained resource ownership; and separate publication and borrower barriers.
Keep the single-lock lookup improvements. Include primitive tests with the API,
including reentrancy, both retirement orders, refusal and retained-resource release.

Read `exec/query_lifecycle_registry.hpp`, its unit tests, and the contract portion
of `super-sirius/query-lifecycle.md`. This is a deliberately narrow foundation
PR; its consumers arrive in PR 3. Diagnostic history/memory-wait fields belong to
PR 10. Any minimal failure latch required by safe retirement belongs with PR 3;
PR 10 extends it rather than introducing an alternative owner.

Gate: registration, unknown-ID refusal, concurrent publisher/borrower retirement,
resource retention and destruction outside registry locks. Re-run the available
CPU sanitizer scenarios if kept as reproducible extraction checks.

#### 3. Mandatory binding, continuous ownership and retirement

Wire every runtime component and standalone fixture to a required registry
reference. Introduce `pipeline::begin_submission()` immediately in its shared
form. Creator requests, tasks, completion callbacks, retries and spill victims
must retain a continuous lease. Fold all retirement paths into
`SiriusContext::retire_query_work()` followed by final state release. Include
retained physical plans, query-specific queue disposal, producer quiescence,
shutdown order, and protection against freeing resources after failed cleanup.

Apply the remaining query-local completion/error and drain changes here, keeping
shared managers alive once the lifetime protocol makes retirement safe.
Read the publication path from creator to scheduler to executor, then spill
borrowing and runtime retirement. Preserve caller ownership on refusal and guard
before task declaration order. Include all constructor/fixture migrations in this
PR; do not add a nullable compatibility path. Keep scheduler barriers needed by
standalone users, success validation, and partial/repeated teardown behavior.

This is the largest coupled review unit: publishers and resource owners need one
consistent protocol. Use separate buildable commits for wiring/publication and
retirement where feasible, with a short ownership diagram in the PR description.
Do not split it into independently mergeable PRs if that introduces an interval
with untracked live work or premature resource release.

Gate: rejection and exception destruction, pop-to-worker races, lookahead,
spill borrowing, OOM handoffs, engine/window teardown, partial initialization,
failed cleanup, streaming fragments and standalone plan fixtures. Ordinary failures
must leave shared managers usable, including a subsequent successful query.

#### 4. Scheduler and memory progress

Move failed reservations and OOM retries back to the spill-visible scheduler,
release worker capacity, preserve accepted-dispatch readiness, choose the oldest
compatible runnable query, and enforce the no-progress deadline. Include actual
HOST result reservations and query-ID exhaustion checks. Place the existing
HOST/DISK capacity regressions with this behavior rather than in a later test PR.

Read `pipeline/{task_scheduler,gpu_pipeline_executor}.*`,
`op/sirius_physical_result_collector.cpp`, queue selection, and `query_id.hpp`.
The admission-dependent automatic sort cap belongs to PR 11. Memory-wait
observability is added in PR 10; the underlying progress guarantee is tested here.

Gate: compatible FIFO, retry eligibility, refusal without lost readiness, bounded
retries, cancellation, and preservation of the source batch when spilling cannot
reserve capacity. Actual concurrent-SQL pressure scenarios accompany PR 11.

#### 5. Session options and synchronized logging

Move mutable SQL execution settings to connection-local state, snapshot them for
planning/execution, and install/restore expression policy on workers. Include
compression-policy selection per invocation, GPU subset selection, RESET/default
behavior, rejected GLOBAL writes, and synchronized shared logging configuration.
Preserve main's new NVTX opt-in behavior while changing configuration boundaries.

Read `sirius_extension.cpp`, `sirius_context.*`, planning consumers and
`expression_evaluator/query_policy.hpp`. Include relevant configuration docs now.

Gate: two-connection option isolation, immutable execution snapshots, expression
policy restoration, disabled-runtime settings, RESET, and logging rollback/races.

#### 6. Query-owned telemetry, metadata and cache bookkeeping

Attribute shared batch placements/ports to query owners and retire only that
owner's records. Scope Iceberg memoization to the connection/statement, including
internal queries and planning declines. Synchronize cache-summary baselines.
Keep the final private/test-safe telemetry interfaces from the simplification.

Read `telemetry/batch_telemetry.*`, Iceberg metadata readers, context cleanup, and
cache reporting. Complete pin publication is the next PR; it is a distinct
publication invariant rather than a telemetry cleanup concern.

Gate: the same batch used by two query owners, retiring one while the other still
publishes/consumes, plus Iceberg statement and connection isolation regressions.

#### 7. Complete pin publication

Publish data, visibility/MVCC facts, uniqueness proofs, compression information,
and handles together. Preserve old generations for existing readers and preserve
the old entry on failed replacement. Column merges must apply proofs only to the
bytes actually accepted. Introduce this in its final form without the obsolete
`attach_*` mutation APIs; migrate all affected fixtures in the same PR.

Read pin insertion/merge paths in `sirius_scan_manager.*`, `pinned_chunk_stats.hpp`,
pin SQL callers, and uniqueness/epoch/MVCC tests. Maintenance remains exclusive;
this does not promise online pin replacement during SQL execution.

Gate: initial lookup metadata, failed replacement, old-reader stability, column
merge proof provenance, epochs, late-materialization handles and SQL MVCC checks.

#### 8. CUDA stream ownership and device-aware prefetch

Add runtime-owned exclusive stream leases and migrate pin materialization,
dynamic-filter publication/replication and memory prefetch. Prefetch the query's
admitted GPU subset using device guards and reservations. Include thread-local
string decoder device caching and partial prefetch-worker construction cleanup.

Read `memory/runtime_stream_pool.hpp`, dynamic-filter CUDA callers,
`scan_manager/memory_prefetcher.*`, and the decoder cache. Device placement
snapshots already exist after PR 5. Fatal prefetch propagation is completed in
PR 10 before concurrent admission is enabled.

Gate: simultaneous exclusive leases, continued stream lifetime after return,
prefetch accounting, dynamic-filter regressions, and device routing. Keep actual
multi-GPU qualification explicitly outstanding on a one-GPU host.

#### 9. Shared I/O capacity and independent scan producers

Introduce the runtime shared scan budget with FIFO speculative allocation and
nonblocking demand debt. Retain budget tokens through actual completion. Separate
blocking coalescers from metadata producers, bound speculative eviction retries,
and fix cache entry lookup across rehash. Query retirement stops both producer
paths before providers are destroyed.

Read `shared_scan_budget.hpp`, `readahead_scan_manager.*`, scan-manager pool
construction, and `io/cache/prefetching_cache.*`. The internal scan capacity field
can be introduced here, with single-query runtime admission still enforced.
Canonical public `sirius.max_concurrent_queries` parsing/alias behavior moves to
PR 11 so users get one final configuration contract.

Gate: shared-budget saturation, demand progress/cancellation, token lifetime,
cache regressions and producer/coalescer separation. The SQL scan-oversubscription
watchdog runs when PR 11 provides actual overlapping admission.

#### 10. Runtime health and query diagnostics

Complete fatal CUDA classification/propagation across execution, pinning and
prefetch, and refuse new registration after shared runtime failure. Keep ordinary
allocation/input failures query-local. Add first-error/timestamp diagnostics,
memory-wait counters, once-only completion observers and bounded retirement
history without retained resource references.

Read `cuda/device_health.hpp`, registry diagnostic extensions, completion
observers and all fatal-error callers. Retain the existing health ownership
structure: larger opportunity 5 in the simplification report was not implemented
and is outside this extraction. Admission-specific occupancy/timestamps follow
in PR 11 because that controller does not exist yet.

Gate: recoverable/fatal classification, late registration refusal, observer
exceptions, history bounds, resource release and diagnostics during retirement.
Actual destructive CUDA fault injection remains a release gate.

#### 11. Bounded concurrent SQL admission and integration qualification

Add the FIFO admission/maintenance controller and transferable execution permits.
Switch from the serialized execution window only here. Include canonical startup
configuration plus its old scan-key alias, conflicting-value rejection, configured
scan capacity, cancellable admission, shutdown, nested-window rejection,
per-thread reservation enforcement and concurrency-aware automatic sort caps.
Maintenance covers pin/unpin/reset/index mutation and the established UPDATE
checks. Admission diagnostics complete PR 10's query observability.

Read `query_admission.hpp`, `sirius_context.*`, configuration parsing and SQL
entry points. Include the concurrent SQL contract and all integration scenarios
that require N>1: separate connections, overlap/N+1 queueing, failures and
cancellation, maintenance, scan oversubscription, pins/compression, cache, memory
pressure, repeated execution and the explicit two-GPU qualification target.
Default stays N=1; submissions may queue beyond that configured limit.

Gate: build, controller tests, one-GPU concurrent SQL matrix, memory-pressure
watchdogs and fragment behavior. Test fatal-health refusal at admission as well
as registration. Multi-GPU tests compile here; execution on suitable hardware is
required before claiming that qualification. Document configuration changes in
inline docs, PR description and the repository's applicable changelog/release notes.

#### 12. Review and validation record

Preserve `docs/concurrency/101` through `104`, including this complete source map,
historical decisions and hardware limits. Update the extraction results to the
actual new branch/commit identifiers. Cross-cutting navigation can live here;
feature-specific docs and correctness tests must already be in PRs 1–11.

The historical journals describe earlier development states, including interfaces
removed by subsequent simplifications. Label them as historical and point readers
to the final contract; do not copy obsolete APIs into the extracted implementation.
Run Markdown checks and verify links. Include this layer because the requested
scope includes these tracked reports; do not silently drop them from the final tree.

### Complete source-commit map

Original hashes are stable references through the backup branches. Rebased hashes
identify the current source. A multi-PR row must be split by behavior; it is not a
request to duplicate a patch. Documentation and test registration follow their
feature, except the historical reports assigned to PR 12.

| Original | Rebased | Subject / surviving work | Destination PRs |
|---|---|---|---|
| `ca4f700cc` | `c994c8967` | Query failures stop killing shared subsystems | 1, 3 |
| `62c5b1eb8` | `84bc83a55` | Cleanup of the initial change | Fold into 1, 3; keep deleted material deleted |
| `9de8f0e77` | `dd1968e12` | Lifecycle gate, creator lock order, core ignores | 1–3 |
| `481fa1da6` | `2635bc76a` | Formatting | Fold into affected feature |
| `d618d452a` | `37b7a1fc0` | Unknown-ID refusal and diagnostics | 2–3 |
| `a4db7d6be` | `987ca5b53` | Lifecycle documentation | 2–3 |
| `2f7430f29` | `c5261bac7` | Submission guards and work leases | 2–3 |
| `b153357e6` | `96ad1028c` | Per-query completion and error paths | 1, 3 |
| `751f41b31` | `42cef4dcf` | Bounded crash handler | 1 |
| `9a0c4b9af` | `f10991d76` | In-flight tracking and lookahead | 1, 3–4 |
| `aa28534e4` | `afce9c3de` | Query-aware pools and per-query waits | 1, 3 |
| `d9607cfeb` | `d08112ab0` | Shutdown order, failed-cleanup retention, loud drops | 3 |
| `f47e992aa` | `45bf8f7ab` | Cache rehash safety and scan concurrency capacity | 9, 11 |
| `534caf2d3` | `3291d6c74` | Continuous handoffs, safe queue disposal, readiness | 3–4 |
| `ffb08ba59` | `863938585` | Spill victims, retained plans and isolated cleanup | 3 |
| `f7b3c76d5` | `3b8f0a1f7` | Memory progress, FIFO, HOST reservation, ID exhaustion | 4 |
| `26880b8b3` | `b6719d3f5` | Owned metadata and invocation-local compression selection | 5–6 |
| `4f42818bb` | `aa333a6e6` | Session snapshots and logging | 5 |
| `7241a1b2e` | `cbc6e71ec` | Admission, maintenance, config and sort caps | 11 |
| `a948c8157` | `5666cee02` | Stream leases and device-aware prefetch | 8 |
| `24907f3ad` | `f91ee2288` | Shared I/O budget and separate scan pools | 9 |
| `af123aa86` | `ef3124720` | Dispatcher failure and thread construction | 1 |
| `136405abd` | `8e9aec5c5` | Complete pin generations | 7 |
| `099835f7f` | `b42539eb7` | Device health, diagnostics and completion observers | 3's minimal failure backstop; 10–11 |
| `bb9e445f1` | `98b7392a7` | Concurrent SQL matrix and fatal prefetch propagation | 10–11 |
| `48f30c829` | `b7963337d` | Spill exhaustion and telemetry ownership tests | 4, 6 |
| `bf0a2cf16` | `36a6f4d03` | Multi-GPU qualification and documentation | 11–12; architecture docs with their feature |
| `d442bce54` | `55250ba94` | Test adaptations to newer main | 3, 5, 7, 11 with affected tests |
| `a356cee83` | `f0f8689ff` | Trivial simplifications | Fold into 1–6, 9, 11 as applicable |
| `c81ec8ae6` | `c70270b74` | Required lifecycle binding, unknown resource refusal, reports | 2–3, 12 |
| `2ea940459` | `939d1384d` | Remove pin mutation APIs and migrate coverage | 7; journal in 12 |
| `1d3cfd9e8` | `9593dfd9b` | Central retirement phases | 3; journal in 12 |
| `f9234950b` | `7f1615c20` | Shared publication adapter | 3; journal in 12 |

The large simplifications are therefore part of the first reviewed version of
lifecycle integration and pin publication. There is no separate late cleanup PR.
For shared files such as `sirius_context.cpp`, `sirius_extension.cpp`, the registry,
scan manager and integration harness, extract hunks by feature, not whole files.
Add each new test source to `cmake/sirius-test-sources.cmake` in its owning PR.

### Extraction and verification procedure after plan approval

1. Preserve an immutable reference to the rebased, reviewed source tree including
   this plan. Keep the three pre-rebase backups. Treat `concurrency4` as the source,
   not a branch to rewrite while assembling the review stack.
2. Build each layer locally from the agreed main SHA, using the map above to
   reconstruct the final behavior. Preserve author attribution. Squash obsolete
   intermediate implementations and fold fixture adaptations into their feature.
   Record actual file/line counts and logical dependencies for every PR. A large
   layer needs either a smaller safe boundary or an explicit reviewability rationale.
3. Build **each prospective PR tip** and run its focused tests before constructing
   the next layer. If a PR contains multiple commits, each commit must compile.
   Use `pixi run make`; after a failed build follow AGENTS' clean-rebuild rule.
   Check N=1 behavior throughout; exercise the new controller only once introduced.
4. At the top, compare the tracked tree against the preserved reviewed source with
   `git diff --exit-code <source-ref> <top-branch>`. Require equality; review and
   document any necessary extraction-only change separately. A range-diff cannot
   establish this after deliberately regrouping/squashing commits. Account for all
   131 source-diff paths plus this plan edit, and preserve main's changes.
5. Run the combined concurrency/lifecycle/pin/scan/settings suites and the ordinary
   unit-test runner as a stack gate. Hardware/data-dependent gates must be listed
   separately with supplied fixtures; a skip is not a pass. Preserve the outstanding
   multi-GPU and destructive-fault gates above. Earlier validation counts do not
   prove that reordered intermediate PRs build or behave correctly.
6. Initialize `gh-stack` only once these branches exist and their contents are
   reviewable. `origin` currently points to `sirius-db/sirius`, matching the
   maintainer stacked-PR convention. Adopt the branches bottom-up with the installed
   extension, explicitly naming `main` as trunk:

```bash
gh stack init --base main \
  stacked/concurrency-01-query-errors-workers \
  stacked/concurrency-02-lifecycle-contract \
  stacked/concurrency-03-lifecycle-integration \
  stacked/concurrency-04-memory-progress \
  stacked/concurrency-05-session-options \
  stacked/concurrency-06-metadata-ownership \
  stacked/concurrency-07-pin-publication \
  stacked/concurrency-08-cuda-streams \
  stacked/concurrency-09-scan-capacity \
  stacked/concurrency-10-runtime-health \
  stacked/concurrency-11-concurrent-admission \
  stacked/concurrency-12-review-record
gh stack view
```

7. Prepare self-contained PR descriptions: motivation, ownership/ordering invariant,
   reviewer reading order, configuration changes, exact tests and remaining limits.
   The installed `gh stack submit --auto` creates drafts by default. Its interactive
   editor defaults new PRs to ready, so explicitly switch every PR to Draft if using
   that route. Do not pass `--open`. Supply the prepared descriptions before marking
   any PR ready; do not rely on generated titles/descriptions as the review guide.
   Creation/submission is a later phase after this plan is reviewed.
8. Use `gh stack rebase` for later conflict management and `gh stack submit` to
   publish those updates. Merge bottom-up through the merge queue as documented in
   CONTRIBUTING; use `gh stack sync --prune` when needed after a lower layer merges.
   Never use `gh stack merge` or the Web UI's "Enqueue stack." Decide explicitly
   whether existing concurrency PRs are superseded before closing or retargeting
   any of them; this planning pass does not alter them.

### Decisions for review

- **Granularity:** recommend the 11 implementation layers plus the documentation
  record above. If fewer PRs are preferred, merge 5–6 (settings/metadata) or 8–9
  (CUDA/scan coordination), accepting larger mixed-subsystem reviews. Keep pin
  publication separate. Do not join admission to an early foundation PR.
- **Lifecycle integration size:** recommend one coupled PR 3 with an explicit
  publication-to-release reading order. If extraction shows a safe buildable
  boundary between task ownership and spill/retirement integration, propose that
  split with evidence before adding another layer.
- **Hardware qualification:** recommend keeping PR 11 draft until its agreed
  qualification gate is met. If only one-GPU support is to be advertised first,
  state that scope explicitly; do not describe the unrun multi-GPU matrix as passed.
- **Historical documents:** recommend retaining the separate record PR so the
  requested complete tree is preserved. Moving development reports outside the
  stack would change the requested scope and needs an explicit decision.

No new runtime redesign is proposed here. The objective is reviewable extraction
of the implemented final state, with safety fixes present at their introduction.
