# Concurrency stack: simplification review

> Historical development record. The implementation was regrouped into the review stack;
> see [the extraction and validation record](105-pr-stack-extraction.md) for current PRs and
> validation. Earlier interfaces described below may have been removed. The current
> contracts are [query lifetime](../super-sirius/query-lifecycle.md) and
> [concurrent queries](../super-sirius/concurrent-queries.md).

Implementation progress is recorded in
[104-simplification-implementation.md](104-simplification-implementation.md).
The assessment below records the state before those changes.

## Scope and recommendation

Reviewed the combined changes from `main` (`ce7d947b0`) through `concurrency4`
(`d442bce54`), including `concurrency3_0` and `concurrency3_0_1`, on October 2,
2026. The starting diff contained 104 files, 6,465 insertions and 1,757 deletions,
including documentation and tests. Caller searches also covered code outside the
diff, since a helper is only unused if its other callers are gone too.

The highest-value simplifications are removing legacy ungated execution paths,
removing post-publication pin mutation, and consolidating retirement choreography.
The first two reduce the number of supported behaviors; the third makes the
lifetime ordering easier to audit. They deserve separate changes with focused
validation. They are **proposals**, not changes made by this review.

The small changes below are implemented in the working tree. No branch history
has been rewritten and no commits have been created for this review. This is a
simplification assessment, not a claim that every possible concurrency race has
been excluded.

## Small changes implemented

| Area | Change and rationale |
| --- | --- |
| Query IDs | Removed `next_query_id()`: only its own test called it. Production IDs already come from admission tickets. Kept the priority-range check and adjusted its test to exercise the real API. |
| Lifecycle snapshots | Removed the unused `query_activity::resources` field. Actual resource ownership remains in `query_control::resources`. |
| Pool waiting | `bounded_thread_pool::drain_and_wait(query)` now reuses `wait_for_query(query)`. Dropped work is still destroyed outside the pool mutex before waiting. |
| Whole-executor drain | Reused `quiesce_manager()` and `resume_manager()` around the queue drain; the interrupt/join/wait/resume order is unchanged. |
| Queue internals | Removed unused private `drop_level()`, superseded by the drain implementation that destroys tasks outside the queue mutex. |
| Repository access | Removed the uncalled `SiriusContext::get_data_repository_managers()` facade. Production spilling already uses the registry's identity-bearing `candidates()` snapshot. |
| Creator worker check | Removed a one-caller wrapper around the existing thread-local worker marker. The stop assertion uses the marker directly. |
| Session settings | Removed 23 unreachable null checks after `get_operator_params()`. Session state is created on demand; the helper returns its address or throws. |
| Telemetry | Removed default query IDs from `register_consumer_port()` and `on_packaged()`. All callers already supply IDs; future callers must do so too. |
| Guard misuse | Explicitly deleted copy assignment for `shared_scan_budget::ticket` and `scoped_expression_policy`. Both already prohibited copy construction, but implicit assignment could overwrite accounting ownership or restoration state. Sharing a budget token through its `shared_ptr` remains supported. |
| Comments | Updated descriptions of per-query waits, spill-task leases and error drains that still described intermediate versions of the stack. |

These changes do not remove lifetime barriers, change query priority or introduce
additional runtime synchronization. Compile-time checks confirmed the assignment
loopholes before the change and their rejection afterward.

## Larger opportunities

### 1. Make lifecycle binding mandatory

**Recommendation: do this before splitting the stack. Medium scope.**

Relevant code: `SiriusContext::initialize()`, `task_creator`, `task_scheduler`,
`itask_executor`, `downgrade_executor`, `memory_prefetcher`, and
`convertible_gpu_pipeline_task`.

Production initialization binds a lifecycle registry, but several component APIs
default to a null registry and silently run without admission/lease checks.
Standalone tests still use this behavior. It creates two execution modes to
maintain and permits a future production caller to accidentally bypass the
safety mechanism. Some fallback paths also stop a shared manager or retry work
differently from the registered path.

`query_lifecycle_registry::retain_resources()` has a related exception: unknown
IDs are silently ignored specifically to support standalone plan fixtures.

Plan:

1. Give standalone fixtures a small real registry and register their query IDs.
2. Require the registry when constructing runtime components, or enforce binding
   before they start if construction dependencies prevent reference injection.
3. Remove null-registry branches and make retaining resources for an unknown ID
   a programming error. Preserve harmless repeated cleanup of retired IDs.
4. Reassess GPU retry paths without a creator after fixtures use production
   wiring; remove them if no supported caller remains.

Validate unknown/closed IDs, rejected publication, spill returns, OOM retries,
initialization failure and per-query teardown with a neighboring query active.
The gain is fewer branches and a stronger default API; the main cost is fixture
migration. Avoid introducing a public "disable lifetime tracking" test flag.

### 2. Publish pin metadata once and remove the mutation APIs

**Recommendation: do this before splitting the stack. Medium scope.**

Relevant code: `sirius_scan_manager::{attach_mvcc_metadata,
attach_proven_unique_columns,insert_pinned_entry}` and `apply_pin_metadata()`.

SQL pinning now supplies `pinned_entry_metadata` during publication. The two
`attach_*` methods have no production callers; remaining callers are tests in
`test_pin_table_mvcc_foundation.cpp`, `test_pin_uniqueness.cpp` and
`test_pin_registry_epoch_mutations.cpp`. These methods retain a second, mutable
publication protocol with reader-count checks, epoch changes and duplicated
metadata handling. Their guards matter today, but the alternate protocol need
not remain part of the public interface.

Plan:

1. Migrate those fixtures to supply metadata when inserting a pin generation.
2. Test complete publication, publication failure, column merges and old-handle
   stability through that supported path.
3. Delete both mutation methods and their duplicated checks/documentation.
4. Retain `apply_pin_metadata()` as the common implementation for publication.

Preserve incarnation matching and uniqueness proofs for the bytes actually
published. In particular, a column merge must not transfer a proof from incoming
bytes that were discarded. Failed publication must preserve the old generation.
This reduces both code and opportunities for misuse without adding a hot-path
lock. Removing the tests without replacing their meaningful coverage would not
be an acceptable shortcut.

### 3. Centralize query retirement choreography

**Recommendation: valuable follow-up after item 1. Larger, safety-sensitive scope.**

Relevant code: `sirius_engine::~sirius_engine()`, both exception paths in
`sirius_engine::execute()`, `task_scheduler::{drain_after_error,
wait_for_completion}`, and `SiriusContext::{run_mandatory_cleanup,
drop_query_runtime_state_best_effort}`.

These paths repeat overlapping sequences of closing publication, stopping scan
producers, draining queues and waiting for borrowers. Much of the repetition is
deliberate protection for different failure stages, but the ordering is scattered
across owners. A future fix can easily update only one path.

Plan:

1. Describe the required phases in one runtime-owned retirement interface:
   close publication and settle publishers; stop producers; settle queued/running
   work; wait for leases; release query resources.
2. Consolidate repeated call sequences first, keeping the existing barriers.
3. Keep success validation distinct from error draining: a successful query
   leaving queued tasks must still be reported, not silently cleaned up.
4. Preserve early borrower retirement before engine-owned objects disappear,
   followed by final runtime-state removal. Partial initialization and repeated
   best-effort cleanup must remain supported.
5. Only remove redundant waits after tests demonstrate that the surviving
   barrier covers every publisher and borrower.

Validate pauses between pop and slot attachment, task destruction callbacks,
borrowed spill repositories, cancellation during initialization, and cleanup
failure. Ordinary query failure must leave other queries running. This can
reduce repeated lookups and waits, but code reduction is secondary to preserving
ordering; avoid adding a global retirement mutex.

### 4. Reuse a small publication helper after lifecycle binding is uniform

**Recommendation: combine with or follow item 1. Medium scope.**

`task_creator::begin_submission()` already centralizes one component's admission
checks. Scheduler, executor and spill-return paths repeat parts of admission,
unknown-ID reporting and lease attachment. Extend the existing pattern or use a
small component adapter rather than adding a general guard framework.

First document what the helper returns on refusal and who owns the rejected task.
Then migrate one publisher at a time, keeping component-specific error reporting
outside the low-level registry. Test destruction on rejection and exceptions.
Guard declaration/destruction order is significant: rejected tasks can invoke
callbacks and must be destroyed before an admitted publication guard is released.
Likewise, do not replace an existing work lease during a handoff or change
`try_push(task&)` to an interface that loses ownership on refusal.

### 5. Consolidate runtime health reporting

**Recommendation: worthwhile separate change. Medium scope.**

`SiriusContext::runtime_unavailable_` and the lifecycle registry's
`runtime_failed_` both contribute to runtime health. CUDA classification,
recording an exception, marking failure and quiescing queries recur across
worker callbacks, prefetch, engine and SQL boundaries.

Introduce one runtime health owner with explicit failure causes, then a narrow
reporting helper for the repeated fatal-error sequence. Keep ordinary query
errors local. Preserve the distinction between reporting an error from a worker
and performing teardown on the query thread; a worker must not wait for itself.

There is already `util/error_utils.hpp`, but its logging helpers assume an active
catch and are not a drop-in replacement for nonthrowing destructor/callback
reporting. Extend it only with a clearly defined `exception_ptr`/nonthrowing
contract. Logging failure must not replace the original exception.

Validate sticky device failure, cleanup failure, simultaneous query failures and
subsequent CPU fallback. This primarily improves consistency, with little expected
steady-state performance benefit.

### 6. Evaluate overlapping scan budgets without changing scheduling policy

**Recommendation: defer unless profiling or maintenance cost justifies it.**

`readahead_scan_manager` uses both the existing per-query `gatekeeper` and the new
runtime-wide `shared_scan_budget`, plus a demand-ticket map. Both account for
demand borrowing, but their responsibilities differ: local arming/worker control
versus shared backend capacity and older-query priority.

A useful investigation is to separate arming/stopping from capacity accounting,
then determine whether shared budget tokens can become the sole I/O capacity
ledger. Preserve any intentional per-query limit. Do not substitute a single
semaphore: multi-backend acquisition must remain atomic, and demand must progress
even before speculative prefetch is armed.

Require tests for demand debt, cancellation of one subscriber, multi-backend
requests, starvation and complete token return, plus throughput measurements.
The current two layers are not enough evidence that either can simply be deleted.

### 7. Give exclusive streams a more direct owner

**Recommendation: defer or handle with a cuCascade change.**

`runtime_stream_pool` already reuses cuCascade's `exclusive_stream_pool` and
`borrowed_stream`. Its extra machinery is the static mutex/map keyed by memory
space, with install/remove bookkeeping in the memory manager.

Consider moving exclusive-pool ownership into cuCascade's memory space, or
passing a runtime stream provider explicitly to users. Either could eliminate
global lookup and registration. The existing `memory_space::acquire_stream()`
returns a raw stream from a round-robin pool and is not an equivalent replacement.

Before changing ownership, prove streams remain alive for buffer deallocation
after the operation's borrow ends. Preserve per-device ownership, cross-device
transfer behavior and nonblocking growth when callers hold batch locks. This is
a broad dependency/lifetime change; the current reuse is preferable to another
custom stream-lease implementation.

## Unused and test-oriented API follow-ups

| API | Current use | Proposed action |
| --- | --- | --- |
| `data_repository_manager_registry::get_all()` | Only registry tests remain after removing the unused context facade. | Migrate tests to identity-bearing `candidates()` and remove the duplicate snapshot API. Production spilling needs query identity to acquire the victim's lease. |
| `query_lifecycle_registry::wait_for_submissions()` | Tests call the separate wait; production uses `quiesce_and_wait_for_submissions()`. | Decide during retirement consolidation whether an external coordinator actually needs the split API; otherwise remove it and test the combined barrier. Keep `wait_for_work()`, which has production users and a different ordering requirement. |
| Lifecycle/pool/admission/budget count snapshots | Mostly test and diagnostic observation. | Keep read-only snapshots where they expose useful state. They do not justify adding mutation hooks or duplicating execution paths. |
| Pin epoch test hook | Pre-existing public `bump_pin_registry_epoch_for_testing()`. | Consider a private fixture accessor separately; it was not introduced by this stack. Prioritize removal of the two mutation APIs in item 2. |

"Only called by a test" is evidence to inspect an API, not by itself a reason to
delete a useful observable invariant. Mutation and bypass interfaces deserve
more scrutiny than snapshots.

## Guard distinctions to preserve

| Mechanism | What it protects |
| --- | --- |
| Admission permit | Number of live queries and exclusive maintenance versus query admission. |
| `accepts_work()` | Advisory cancellation check; it does not reserve publication rights. |
| Submission guard | Queue publication that can race closure and drain. |
| Work lease | Plan/repository borrowers across queues, running tasks and spill operations. |
| Pool slot and query accounting | Worker capacity and per-query waiting for dispatched work. |
| Batch/pin ownership and locks | Data mutation exclusion and immutable pin-generation lifetime. |

These are not interchangeable counters. In particular, replacing submission
guards with `accepts_work()` reintroduces a check-then-push race, and holding a
repository manager `shared_ptr` does not prevent its repositories being cleared.
Unifying their call sites is a safer first step than merging the mechanisms.

## Making the eventual PRs smaller

1. Fold the small cleanups into the commits that introduce the affected APIs.
2. Implement pin API cleanup and mandatory lifecycle binding as separate,
   compilable changes; reassess whether retirement consolidation fits this stack.
3. Consider a preparatory extraction of large operation bodies before adding
   exception boundaries. Much of the `sirius_extension.cpp` and creator diff is
   reindentation around exception handling. Keep ownership and error ordering
   explicit; use `git diff -w` as a review aid, not as a substitute for review.
4. Keep crash-handler hardening in a separate reviewable change. Retain useful
   implementation/validation records, but keep the final architecture docs about
   current behavior rather than intermediate stages.
5. Share only genuinely repeated fixture setup/barriers in the large concurrency
   integration tests. Keep scenario-specific assertions visible and preserve
   subprocess timeouts and cleanup behavior.

Suggested decisions: approve items 1 and 2 first; choose whether item 3 should be
part of this stack or follow it; defer items 6 and 7 pending a stronger benefit.
Item 4 becomes easier after item 1; item 5 can remain independently reviewable.

## Validation of the implemented changes

- `pixi run make`: passed, including the final compilation after deleting the
  unused context facade.
- Focused existing C++ tests: **86 cases, 660 assertions passed**, including
  bounded pools, indexed queues, lifecycle guards, query IDs, task executors,
  OOM retries, telemetry ownership, session options, shared scan budgets and
  concurrent-query integration scenarios. These ran before the final deletion
  of the uncalled context facade.
- Compile-time type-trait checks: both guards reject copy construction and copy
  assignment. Before the change, both were copy-assignable.
- Formatting/lint hooks for changed files and `git diff --check`: passed.
- No throughput benchmark or multi-GPU runtime qualification was performed;
  runtime testing used the available single GPU.

Logs: `/tmp/concurrency-simplification-build.log`,
`/tmp/concurrency-simplification-tests.log`,
`/tmp/concurrency-simplification-format.log` and
`/tmp/concurrency-simplification-final-format.log`.

Excluding this report, the patch removes a net **93 lines** across 17 source/test
files. The proposed structural changes above are not included in that count.
