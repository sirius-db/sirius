# concurrency4: implementation, validation and review guide

Work started 2026-09-29; final validation resumed 2026-09-30. Review this alongside
[the commit journal](101-concurrency4-implementation-journal.md) and
[the runtime contract](../super-sirius/concurrent-queries.md).

## Branch and scope

`concurrency4` now descends from `concurrency3_0_1` (`432f0630b`). The original tip,
`98280218b`, is preserved as `concurrency4_before_rebase_20260929`. Six commits from the
old `concurrency3` stack were replayed before the implementation commits documented below.
Nothing was pushed and no PR was opened.

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

## Suggested stacked PR extraction

The current commits are buildable review checkpoints. They are **not yet the final PR stack**.
Some later tests cross several earlier changes, and public N>1 enablement should move after its
safety prerequisites when extracting PRs.

| Proposed PR | Contents | Current journal entries |
|---|---|---|
| 1. Continuous lifetime and retirement | Guarded handoffs, retained plans, query-local cleanup, victim borrowing, exception-safe dispatcher | 1–2, 9 |
| 2. Scheduler and memory progress | Compatible FIFO, accepted readiness, nonblocking reservation/retry, result capacity | 3 plus readiness pieces of 1 |
| 3. Session and shared metadata isolation | Options snapshots, logging, telemetry/Iceberg ownership, complete pin publication | 4–5, 10, 13 telemetry |
| 4. Scan and CUDA resource coordination | Exclusive stream leases, shared I/O budgets, independent producer capacity, device-aware prefetch | 7–8, prefetch-health portion of 12 |
| 5. Health and diagnostics | Runtime failure boundary, completion observers, bounded query history, memory-wait counters | 11 plus related health tests |
| 6. Bounded admission and qualification | Config migration, permits/maintenance, conservative sort caps, integration matrix and documentation | 6, 12–14 and final documentation |

Keep N=1 as the production default throughout extraction. Do not expose N>1 as supported in an
earlier PR while its prerequisites are still in a later PR. Move tests with the feature they
validate, rebuild each extracted base, then run integration gates on the assembled stack. The
existing six rebased commits also need an explicit home when deciding the final PR boundaries.

Follow CONTRIBUTING's stacked-PR workflow, keep drafts until the reviewability checklist passes,
and merge bottom-up. This document proposes a split; it does not create or publish one.
