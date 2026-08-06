# Final QA review: multi-partition dynamic-filter remediation

Reviewed baseline `bdeaa56b9e63bd5dd67924edfe04454fbabc90bd` plus the complete 22-file uncommitted change set (diff SHA-256 `b39c2e77bb2def3fecdaec38936f78d4ac36934401d5f572256510359ab0883f`), including ignored `DYNAMIC_FILTER_REPORT.md`.

## Findings

### [Medium, non-blocking] No deterministic finalization-versus-contribution regression

**Defect.** Tests overlap an in-flight duplicate and two final contributions (`test/cpp/operator/test_dynamic_filter_publisher.cpp:983-1069`) and test incomplete hash-join finalization sequentially (`test/cpp/operator/test_dynamic_filter_publication_claim.cpp:253-268`), but never overlap `on_finalize_operator()` with an in-flight final contribution. The accumulator mutex and hash-join CAS protocol are sound by inspection (`src/op/sirius_physical_hash_join.cpp:898-920,2218-2250`), but this correctness-critical cross-layer linearization remains uncovered.

**Required property.** Deterministically cover both winners: finalize first yields no fan-out and one failed-stat commit; contribution first yields one fan-out and one finished-stat commit.

**Cheapest remedy.** Add one narrow synchronized hash-join test around the existing post-insert seam (or an equivalent join-level seam), reversing release order for the two cases. No physical multi-GPU hardware is needed.

**Release impact.** Non-blocking for this default-off implementation; add it before enabling the feature by default.

### [Low, non-blocking] Concurrency-test rendezvous can hang instead of fail

**Defect.** The duplicate test waits indefinitely on latches (`test/cpp/operator/test_dynamic_filter_publisher.cpp:991-1013`); the competing-final-contributions test uses an uncancellable barrier followed by blocking `future::get()` calls (`test/cpp/operator/test_dynamic_filter_publisher.cpp:1033-1053`). A worker exception before rendezvous, or a synchronization regression, can hang until the outer CI timeout.

**Required property.** Test rendezvous must fail within a bounded interval and release/join every worker on every exit path.

**Cheapest remedy.** Use a bounded condition-variable/promise rendezvous with scope-guarded release, or an equivalent harness timeout that guarantees worker release.

**Release impact.** Non-blocking; this affects failure diagnostics and CI latency, not production behavior or the validity of a passing run.

## Prior findings: resolved

- **Captured task-stream lifetime:** resolved. CUCO partials are constructed on the owning GPU memory-space stream and initialized before task-stream insertion (`src/op/dynamic_filter/dynamic_filter_publisher.cpp:595-615`). The short-lived-task-stream teardown regression is at `test/cpp/operator/test_dynamic_filter_publisher.cpp:953-980`.
- **Aborted statistics dropped:** resolved. The winning `ACCUMULATING -> FAILED` caller folds the aborted outcome exactly once (`src/op/sirius_physical_hash_join.cpp:910-920`), covered at `test/cpp/operator/test_dynamic_filter_publication_claim.cpp:270-305`.
- **Missing overlap/replica-failure tests:** resolved for the prior review's required cases. Tests now force an in-flight duplicate, competing final contributions, and strict-replica failure before fan-out (`test/cpp/operator/test_dynamic_filter_publisher.cpp:983-1100`). Hooks default empty, and only tests supply non-empty hooks (`src/op/dynamic_filter/dynamic_filter_publisher.cpp:653-667`).

## Additional acceptance checks

- Root reduction uses one durable root stream, synchronizes on success, and best-effort drains before failure escapes (`src/op/dynamic_filter/dynamic_filter_publisher.cpp:468-502`). Scratch and non-root partials are released only after successful drain.
- Accumulator abort/publication is mutex-serialized; a single hash-join CAS winner commits terminal state and statistics. Strict replication completes for all active keys before any fan-out (`src/op/dynamic_filter/dynamic_filter_publisher.cpp:513-535,636-649,679-699`).
- `enable_dynamic_filter_multi_partition` defaults false and is passed by the planner only under the `enable_dynamic_filter` master gate (`src/include/sirius_config.hpp:137-142`; `src/sirius_extension.cpp:2351-2365`; `src/planner/sirius_plan_comparison_join.cpp:404-418,661-676`).
- Updated docs and `DYNAMIC_FILTER_REPORT.md` consistently describe the default-off rollout, whole-build invariant, durable-stream ownership, strict replication, and deferred physical multi-GPU validation.
- No new production leak or data race was found. The code uses scoped RAII ownership/locking, atomics, `std::span`, and C++20 synchronization facilities consistently with the reviewed project references.
- `git diff --check HEAD` passed. Developer-supplied evidence reports a release build passing 15/15 targets and focused tests passing 284 assertions in 18 cases. Physical multi-GPU tests were not run, as requested.

## Verdict

**Accepted.** All three prior remediation gates are resolved and no production correctness blocker remains. The two residual test findings are explicitly non-blocking while the feature remains default-off. Physical multi-GPU validation remains required before claiming production validation or changing the default.
