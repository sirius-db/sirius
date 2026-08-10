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

---

# Follow-up QA review: per-GPU Bloom policy cap

Reviewed baseline `4016c598332210ab4e4a38b44f7ccb129c5abd1c` plus the final 20-file uncommitted cap change set (diff SHA-256 `976d682c5b268a44a11fc0bfec3f85f3a9582a890dd09a34fdfca5ec7b06a9a3`). This section is scoped to the Bloom-cap follow-up; the prior review and its two non-blocking residual findings remain unchanged above.

## Findings

No open cap-specific finding remains.

## Findings resolved during this review

### [Medium] Saturated estimates could pass an unlimited cap

**Defect.** `estimated_bytes()` used `SIZE_MAX` as the saturation sentinel when raw multiplication or CUDA allocation alignment was not representable, but the initial helper admitted `bloom_budget_allows(SIZE_MAX, 1, UINT64_MAX)` by equality. The constructor also admitted the alignment-only boundary with raw bytes `SIZE_MAX - 31`; CuCascade's subsequent 256-byte `rmm::align_up` can wrap that accounting charge to zero.

**Required property.** A saturated or otherwise nonrepresentable tracked footprint must be rejected before CUCO allocation, while equality remains admissible for representable footprints.

**Cheapest remedy.** Reject the existing sentinel in the budget helper and mirror the alignment-overflow guard in the public empty-Bloom constructor; no new sizing abstraction is required.

**Resolution.** The estimator now uses overflow-safe geometry and saturation (`src/cuda/sirius_dynamic_bloom_filter.cu:53-64,248-255`), the helper rejects the sentinel (`src/include/op/dynamic_filter/dynamic_filter_source_policy.hpp:79-98`), and the constructor rejects raw and alignment overflow before constructing CUCO storage (`src/cuda/sirius_dynamic_bloom_filter.cu:266-285`). Tests pin both boundaries (`test/cpp/operator/test_dynamic_filter_source_policy.cpp:194-206`; `test/cpp/operator/test_dynamic_filter_publisher.cpp:378-391`).

### [Low] Configuration-to-statistics behavior lacked one crossing test

**Defect.** Initial tests covered SQL setting storage and direct publisher outcomes separately. Omitting planner transport, hash-join folding, or snapshot exposure would have left them green.

**Required property.** A configured rejection must preserve results, complete fail-open, construct and push no Bloom, and increment the cumulative skip counter through the production planning path.

**Cheapest remedy.** Add one zero-cap section to the existing deterministic forced multi-partition integration case.

**Resolution.** The section asserts result parity, an enabled producer, a positive size-gate count, zero membership construction, successful publication, and zero fan-out (`test/cpp/integration/test_gpu_execution_dynamic_filter_sip.cpp:100-131,365-375`). It crosses planner transport (`src/planner/sirius_plan_comparison_join.cpp:654-677`), outcome folding (`src/op/sirius_physical_hash_join.cpp:89-102`), and snapshot exposure (`src/include/op/dynamic_filter/dynamic_filter_stats.hpp:118-132`).

## Acceptance checks

- Units match CUCO's 32-byte Bloom blocks and CuCascade's 256-byte tracking alignment; aggregate admission uses division and rejects the saturation sentinel.
- One-shot rejection gates the complete Bloom candidate set before construction while exact IN-lists and zone maps remain eligible (`src/op/dynamic_filter/dynamic_filter_publisher.cpp:223-295`).
- Multi-partition rejection uses the global row count and all active keys before partial allocation, emits no Bloom, and completes fail-open after exact batch accounting (`src/op/dynamic_filter/dynamic_filter_publisher.cpp:410-449`).
- The cap is per join on each GPU: it is neither multiplied by replicas nor divided by partitions. Source/partial storage remains allocator-accounted; destinations reserve their aligned allocation (`src/cuda/sirius_dynamic_bloom_filter.cu:350-385`). Scratch and host overhead are documented exclusions.
- The 256 MiB default is consistent across `operator_params`, YAML, SQL UBIGINT registration, planner capture, outcome folding, and the atomic snapshot.
- No new raw ownership, leak, or data race was found. RAII ownership/reservations, immutable policy, mutex-serialized accumulation, and atomic statistics remain consistent with the reviewed C++ guidance.
- User, design, report, and Doxygen text consistently cover representable equality, scope, all-or-none rejection, zero-cap behavior, reservation distinctions, excluded scratch, statistics, and deferred physical multi-GPU validation.

## Validation

- Full post-cap release build: 222/222 targets passed; post-review remediation rebuilt 52/52 incremental targets.
- Final `[bloom_budget]`: 50 assertions in 6 cases passed, including both overflow regressions.
- Final forced multi-partition parent case, section `a multi-partition build obeys the subordinate switch`: 166 assertions in 1 case passed, including zero-cap config-to-stats fail-open behavior and result parity.
- Broader publication/accumulator/reduction selection: 143 assertions in 14 cases passed. YAML/multi-partition SQL selection: 19 assertions in 2 cases passed.
- `git diff --check HEAD` and `clang-format --dry-run --Werror` over all changed C++/CUDA files passed. The temporary test-only GPU-pool reduction was restored exactly; final status contains only the intended 20 modified files and no backup artifacts.
- Physical multi-GPU execution remains deferred as requested; static placement/reservation review is not a substitute for that hardware matrix.

## Verdict

**Accepted.** The cap is correct for one-shot and multi-partition publication, overflow-safe at policy and constructor boundaries, consistently wired and observable, fail-open on rejection, and covered by focused and end-to-end single-GPU tests. No cap-specific correctness, ownership, race, documentation, style, or test blocker remains. Physical multi-GPU validation is still required before claiming multi-GPU production validation or enabling the multi-partition feature by default.
