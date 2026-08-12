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

---

# QA review: PR1451 publication API-contract refactor

Reviewed baseline '13e0eb4078451f4b9a666e0bcf4a1d90dfb805f8' plus the 13-file uncommitted change set (1,117 insertions / 464 deletions; diff SHA-256 '7992ded634c2df475165081d2be911230dc54bfeaed337d2ce910bdefc0cc0b6'). This section supersedes the earlier acceptance verdicts only for this later refactor.

## Findings

### [High, blocking] One-shot failure can release the pinned build input while CUDA still reads it

**Defect.** One-shot publication can enqueue a source-reading copy for an earlier key ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:292-301') and then throw while processing a later key, for example when that key's runtime ordinal is out of range ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:225-230') or a later allocation fails. Its only construction-stream synchronization is on the success path ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:347-352'). The new session catch marks the attempt failed and immediately rethrows without draining the stream ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:949-954'). The caller's 'build_ro' is the pin that keeps the source representation alive ('src/op/sirius_physical_hash_join.cpp:1997-2030'); it unwinds after the call at 'src/op/sirius_physical_hash_join.cpp:2070'. The small-IN constructor concretely enqueues an asynchronous device copy from the input column ('src/cuda/sirius_dynamic_small_in_list_filter.cu:144-165'). This violates the documented backing-storage precondition ('src/include/op/dynamic_filter/sirius_dynamic_filter.hpp:284-291'; 'docs/super-sirius/dynamic-filters.md:540-542') and can turn an ordinary publication exception into a use-after-release by in-flight GPU work.

The current ordinal regression has only one invalid key ('test/cpp/operator/test_dynamic_filter_publisher.cpp:531-556'), so it never creates outstanding work before throwing.

**Required property.** On every exit after the first source read is enqueued, all reads of 'build_view' must have completed (or been safely cancelled) before control returns to a caller that may release its pin. Failure cleanup must not replace the original exception or throw from a no-throw cleanup path.

**Cheapest remedy.** Add a best-effort exceptional stream drain at the public publisher boundary (or, at minimum, in 'publish_one_shot' before terminal commit/rethrow). Add a deterministic two-key regression: key 0 queues construction, key 1 throws, and a blocked construction stream proves the call does not unwind until the queued read is released and completed.

### [Medium] Cross-GPU preparation replaces the frozen logical batch ID

**Defect.** The snapshot freezes repository batch IDs ('src/op/sirius_physical_partition.cpp:354-365'), but PARTITION contributes the prepared accessor's ID ('src/op/sirius_physical_partition.cpp:188-195'). Legitimate cross-GPU preparation clones the batch under a fresh 'get_next_batch_id()' ('src/include/pipeline/batch_lock_utils.hpp:108-124'), and preparation deliberately replaces the task's original batch with that clone ('src/op/sirius_physical_operator.cpp:123-130'). The accumulator then sees the clone ID as unknown and aborts ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:658-672'). This contradicts the report's claim that the contributed pre-scatter ID is stable retry identity ('DYNAMIC_FILTER_REPORT.md:65-75'). Query results remain correct, but a supported preparation path silently disables the advertised global Bloom.

**Required property.** Exact-once identity must be the frozen logical/source batch ID and must not change when its physical representation is cloned, moved, or retried.

**Cheapest remedy.** Preserve the original ID as immutable task metadata and contribute that value. If the provenance cannot be retained cheaply, scope accumulated publication out of cloned inputs and fail open before contribution. Add a clone-path regression that starts with a frozen ID, prepares onto another GPU, and completes that same logical ID exactly once.

### [Medium] The move-only snapshot does not preserve or defensively recheck its structural invariant

**Defect.** 'complete_build_snapshot' advertises non-empty unique IDs and partition geometry as a type invariant ('src/include/op/dynamic_filter/dynamic_filter_publisher.hpp:40-45'), but both move operations are defaulted ('src/include/op/dynamic_filter/dynamic_filter_publisher.hpp:61-64'). A moved-from object remains callable but can have an empty moved-from ID vector while retaining its scalar row and partition values. Unlike the base accumulator, the new consuming constructor validates only that the plan is enabled ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:440-452'). Passing a moved-from snapshot to another session can therefore claim the session with no expected IDs, exclude the one-shot path, and remain incomplete until finalization. The public API states that its argument is a validated snapshot but neither specifies a moved-from precondition nor rejects the invalid state.

**Required property.** No consuming API may arm an accumulator unless the snapshot still satisfies every structural invariant on which exact completion relies.

**Cheapest remedy.** Recheck structural validity at the accumulator/session boundary (at least non-empty IDs and partition count, with uniqueness if the type is intended to defend against every valid-but-unspecified moved-from state), and reject without arming. Add a moved-from-reuse regression. A documented invalid moved-from sentinel is acceptable only if every consumer rejects it.

### [Medium] Throwing diagnostics can terminate the supposedly fail-open noexcept paths

**Defect.** 'dynamic_filter_publication_session::contribute(...) noexcept' calls 'SIRIUS_LOG_WARN' before aborting and committing terminal state in both exception handlers ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:879-911'). 'abort_if_incomplete() noexcept' and 'finalize_or_abort() noexcept' also log ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:773-781,958-986'), as does 'synchronize_after_failure(...) noexcept' ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:533-545'). Logging has no no-throw contract: 'format_and_log' calls 'get_sink', virtual 'should_log'/'log', and 'std::format' without a catch ('src/include/log/logging.hpp:28-37'), while the sink interface is not 'noexcept' ('src/include/log/sink.hpp:38-44'). A throwing configured sink or host allocation failure during formatting therefore invokes 'std::terminate'; in 'contribute', it can happen before the accumulator is aborted or the failed statistic is committed. 'try_arm' has the same diagnostic escape after catching construction failure ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:839-875'), so its advertised fail-open false return is not reliable either.

**Required property.** Diagnostics must not weaken the session's exception guarantees: contribution/finalization must never let logging escape, and terminal state/statistics must be committed even when diagnostics fail.

**Cheapest remedy.** Use a local best-effort no-throw logging wrapper (or omit these failure-path logs), and perform mandatory state cleanup before optional diagnostics. Add a throwing-sink regression around an injected contribution failure and accumulator finalization.

### [Medium, test coverage] Production freeze and source-residency retry guards are not pinned directly

**Defect.** The two '[partition_snapshot]' cases validate 'try_create' and the pure 'try_summarize_complete_build' helper only ('test/cpp/operator/test_dynamic_filter_publisher.cpp:338-404'). The forced integration query exercises the happy path, but no deterministic test invokes the real repository boundary at 'sirius_physical_partition::try_freeze_complete_build' ('src/op/sirius_physical_partition.cpp:334-367') to prove the FULL-source check, every-repository-ID accounting, GPU-representation rejection, and freeze-before-first-pop order at 'src/op/sirius_physical_partition.cpp:538-561'. Likewise, no test references 'record_source_not_resident' or 'publications_skipped_source_not_resident', even though the PR1277 contract requires that a nonresident delivery increment only that counter and leave the session open for a later resident delivery ('src/op/sirius_physical_hash_join.cpp:1997-2045'; 'src/op/dynamic_filter/dynamic_filter_publisher.cpp:989-1000'). The report currently describes the helper coverage as PARTITION coverage ('DYNAMIC_FILTER_REPORT.md:21-22,221-245').

**Required property.** A real FULL-barrier repository must freeze all original IDs/rows exactly once before any pop, reject incomplete/non-GPU snapshots without arming, and retry after a source-residency skip without incrementing build-not-whole or consuming the one publication attempt.

**Cheapest remedy.** Extend the existing PARTITION/operator fixture with one real-repository table-driven test for unfinished, exact, zero-row, missing/non-GPU, and first-pop cases. Add one publication-claim facade case that delivers nonresident then resident data and asserts the channel plus source-resident/build-not-whole/attempt/terminal counter deltas. No physical multi-GPU hardware is needed for these two orchestration tests.

### [Low, documentation] publication_attempts excludes a counter path in its field contract

**Defect.** The field comment defines 'publication_attempts' as 'OPEN' to 'PUBLISHING' or 'ACCUMULATING' transitions ('src/include/op/dynamic_filter/dynamic_filter_stats.hpp:101-104'). 'try_arm' increments it before accumulator construction, however, and construction failure transitions directly to 'FAILED' ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:843-875'). The broader type documentation correctly promises one finished/failed terminal counter for every attempt, so the field-level contract is internally inconsistent.

**Required property.** Public counter documentation must include every increment path and support one unambiguous accounting interpretation.

**Cheapest remedy.** Describe the counter as an 'OPEN' session claim/initialization attempt, including failed accumulator initialization; no behavior change is required.

### [Low, style] Publisher relies on a transitive include for cuda_set_device_raii

**Defect.** 'dynamic_filter_publisher.cpp' uses 'rmm::cuda_set_device_raii' ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:561-564') but does not include '<rmm/cuda_device.hpp>' in its include block ('src/op/dynamic_filter/dynamic_filter_publisher.cpp:17-43'). The build succeeds only because another dependency currently exposes that declaration transitively, contrary to C++ Core Guidelines SF.10.

**Required property.** Each translation unit must directly include the header that declares every external facility it uses.

**Cheapest remedy.** Add the single direct RMM include.

## Acceptance checks

- The production snapshot boundary distinguishes structural validation from the caller-owned completeness/pre-scatter proof, and the real implementation checks the FULL barrier and enumerates the repository before the base pop. The open findings concern moved-from consumption, clone identity, and missing direct negative-path coverage, not the normal ordering visible in the current source.
- Session one-shot/accumulator mutual exclusion, session-to-accumulator shared lifetime, and HJ/session lock ordering are coherent by inspection. Session and HJ mutexes are released before GPU work. Per-device partial locks do not invert the coordinator lock.
- The two finalize-versus-final-contribution winners and late-invalid-after-fan-out race are linearized correctly and now have bounded semaphore/future tests. Ordinary successful/aborted outcomes are folded exactly once.
- Zero-row accumulated snapshots complete without allocating or publishing a Bloom. Root reduction uses a durable memory-space stream, synchronizes before releasing non-root partials, verifies strict replicas before fan-out, and retains shared filters through target channels. No additional production leak or data race was found.
- The existing plan, channel, and free one-shot publisher entry points remain available. Physical two-GPU reduction/replication execution remains deferred; the single-GPU OR test is not a substitute for that release gate.
- Reviewer reruns passed: '[publication_session]' (93 assertions / 6 cases) and the broader one-shot/publisher/reduction selection (242 assertions / 19 cases). Developer-supplied evidence reports 2,373 assertions across 67 selected executions plus 'pixi run make', clang-format dry-run, and diff checks passing.
- 'git diff --check 13e0eb4078451f4b9a666e0bcf4a1d90dfb805f8' passed on the reviewed tree.

## Verdict

**Changes requested.** The exceptional one-shot exit does not preserve the pinned input lifetime and is a blocking memory-safety defect. The cross-GPU ID mismatch also contradicts the exact pre-scatter identity contract, while the moved-from and no-throw logging paths leave the new public/session guarantees unenforced. Keep multi-partition publication default-off and do not claim physical multi-GPU readiness until those defects and the stated regression gaps are resolved.

---

# Follow-up QA review: PR1451 remediation

Reviewed baseline 13e0eb4078451f4b9a666e0bcf4a1d90dfb805f8 plus the settled 16-file non-review change set (2,153 insertions / 580 deletions; diff SHA-256 892ec068be5a4d44f3b0c0134c590535f57ea65ce89a4b4bf71a5f6071d8354d). This section preserves the preceding review record and supersedes its Changes requested verdict for the PR1451 API-contract refactor only.

## Findings

No open correctness, ownership, race, API, documentation, style, or test finding remains in the reviewed single-GPU scope.

## Prior findings resolved

### [High] One-shot exceptional input lifetime

**Resolution.** The publisher installs a no-throw failure_stream_guard before filter work and dismisses it only after fan-out (src/op/dynamic_filter/dynamic_filter_publisher.cpp:111-147,224,484). An exceptional exit after a source read may have been queued therefore drains the stream before publish_one_shot commits failure and rethrows (lines 1085-1090), and before the caller's pinned build_ro unwinds (src/op/sirius_physical_hash_join.cpp:1997-2030,2070). The deterministic two-key regression blocks the stream, queues key 0, fails key 1, proves the call remains blocked until release, and verifies the original logic_error plus exactly one failed terminal result (test/cpp/operator/test_dynamic_filter_publisher.cpp:623-687).

### [Medium] Frozen identity across clone and retry

**Resolution.** A one-batch pipelineable_operator_data captures the original task-input ID (src/op/sirius_physical_operator.cpp:45-65). Preparation can replace the physical batch with a fresh-ID clone without changing that metadata (lines 111-158), PARTITION contributes the captured ID (src/op/sirius_physical_partition.cpp:170-198), and OOM requeue moves the same payload while preserving its device pin (src/pipeline/gpu_pipeline_executor.cpp:376-401). Physical clone/retry regressions are present at test/cpp/operator/test_dynamic_filter_publication_claim.cpp:508-569 and test/cpp/pipeline/test_batch_lock_utils.cpp:515-558; their device-1 branches compiled but WARN-skipped on this one-GPU host.

### [Medium] Snapshot invariant after move

**Resolution.** Custom moves clear the source geometry (src/op/dynamic_filter/dynamic_filter_publisher.cpp:68-83). Both accumulator construction and session arming reject invalid snapshots (lines 535-548 and 954-957); the session does so before locking, claiming, or incrementing publication_attempts. Moved-from construction, assignment, direct-accumulator, and pre-claim cases are covered at test/cpp/operator/test_dynamic_filter_publisher.cpp:1299-1331.

### [Medium] No-throw cleanup and terminal races

**Resolution.** Failure diagnostics use the swallowing wrapper at src/op/dynamic_filter/dynamic_filter_publisher.cpp:102-109. Abort state precedes logging (lines 622-629 and 859-896), and session exception/finalization accounting precedes diagnostics (lines 1014-1042 and 1094-1124). Throwing-sink regressions cover contribution abort and incomplete finalization (test/cpp/operator/test_dynamic_filter_publisher.cpp:1335-1380).

The contribution catch resolves the accumulator terminal state atomically before session commit (src/op/dynamic_filter/dynamic_filter_publisher.cpp:874-896,1014-1023). A delayed pending-result exception cannot overturn another contribution's completed publication, and an aborted result retains its counters. Bounded regressions cover that catch race, both finalize-versus-final-contribution winners, and a late invalid contribution after fan-out (test/cpp/operator/test_dynamic_filter_publisher.cpp:1666-1930).

### [Medium] Production freeze and retry coverage

**Resolution.** The real PARTITION boundary requires a finished FULL source, enumerates every repository ID and exact GPU-table row count, and creates the validated snapshot (src/op/sirius_physical_partition.cpp:337-370). Arming occurs under the PARTITION lock before the first base pop (lines 541-565). Production-repository tests cover an unfinished barrier, exact IDs, a zero-row batch, freeze-before-pop, non-GPU and missing representations, and a non-FULL barrier (test/cpp/operator/test_dynamic_filter_publication_claim.cpp:449-638).

The hash-join facade leaves the session open after a nonresident whole-build delivery and can claim a later resident delivery (src/op/sirius_physical_hash_join.cpp:1997-2070). Its regression verifies source-not-resident, build-not-whole, attempt, terminal, push, and no-second-claim counters (test/cpp/operator/test_dynamic_filter_publication_claim.cpp:400-447). Concurrent one-shot arbitration is bounded and exactly once (test/cpp/operator/test_dynamic_filter_publisher.cpp:690-739).

### [Low] Counter documentation and direct include

**Resolution.** Public statistics documentation defines one-shot and accumulated initialization attempts, includes accumulator-construction failure, and says post-close deliveries are counted nowhere (src/include/op/dynamic_filter/dynamic_filter_stats.hpp:68-83,101-113). dynamic_filter_publisher.cpp directly includes rmm/cuda_device.hpp at line 28.

## Additional acceptance checks

- Structural validation and caller-owned completeness are separated accurately: complete_build_snapshot::try_create validates representability, non-empty unique IDs, and partition geometry (src/op/dynamic_filter/dynamic_filter_publisher.cpp:50-65), while the FULL repository boundary supplies production exactness. Zero rows remain valid and complete without Bloom allocation.
- One-shot and accumulation claim the same state under the session mutex (src/op/dynamic_filter/dynamic_filter_publisher.cpp:954-1000,1054-1062). Session and hash-join mutexes are released before GPU work; finalization completes its session operation before taking op_state_mutex (src/op/sirius_physical_hash_join.cpp:2073-2089).
- Coordinator and per-device partial locks do not invert. Contributions reserve IDs under the coordinator, release it for device work, synchronize, then reacquire it to complete (src/op/dynamic_filter/dynamic_filter_publisher.cpp:728-833). Publication starts only after every unique expected ID leaves the in-flight set.
- Terminal transitions drop only the session's reference; in-flight contributions retain a local shared_ptr (src/op/dynamic_filter/dynamic_filter_publisher.cpp:1007-1012). Durable memory-space streams, failure drains, root synchronization, device guards, and channel shared ownership cover CUDA resource lifetime.
- Multi-GPU completion retains global geometry, deterministic root OR reduction, bounded scratch, and strict all-device replication before fan-out (src/op/dynamic_filter/dynamic_filter_publisher.cpp:632-725). Missing required replicas abort before publication. This is static/API review, not physical multi-GPU validation.
- Skip methods recheck OPEN under the session mutex (src/op/dynamic_filter/dynamic_filter_publisher.cpp:1127-1139), preventing stale post-close increments. The direct regression is test/cpp/operator/test_dynamic_filter_publisher.cpp:1930-1947.
- Public Doxygen accurately covers structural versus production completeness, moved-from invalidation, exceptional one-shot lifetime, shared accumulation ownership, exactly-once folding, claim semantics, and post-close statistics (src/include/op/dynamic_filter/dynamic_filter_publisher.hpp:40-71,132-168,195-256,273-365). The PR1277 plan, channel, hash-join plan accessor, and free one-shot publisher APIs remain available.
- All new semaphore and future waits are bounded, with timeout paths releasing gates before assertions or timing out internally.

## Validation

- Reviewer reruns passed: [publication_session] (189 assertions / 14 cases), [publication_claim] (144 / 9), [partition_snapshot] (87 / 5), and [batch_lock_utils] (36 / 9). Device-1 branches compiled and WARN-skipped on the one-GPU host.
- Developer evidence reports a clean SCCACHE_RECACHE=1 pixi run make (1,210 steps), final incremental pixi run make (300 steps), and 2,719 assertions across 94 selected executions, including forced multi-partition integration. Reported clang-format, diff, and artifact audits passed.
- Reviewer git diff --check 13e0eb4078451f4b9a666e0bcf4a1d90dfb805f8 passed.
- Physical two-GPU reduction, strict replication, clone, and retry remain intentionally deferred. Compiled WARN-skips are not runtime validation.

## Verdict

**Accepted.** All seven PR1451 findings are resolved with the required properties and focused regressions, and no new production ownership, race, API, documentation, or bounded-test defect was found. The multi-partition feature remains default-off; physical multi-GPU validation is still required before claiming multi-GPU production readiness or changing that default.
