# Concurrency PR stack: extraction and validation

## Scope and source

The complete implementation was regrouped from `concurrency4_stack_source_20261005`
(`8030a293c`, including the approved split plan) onto `main` at `c2f7f3a7539bf5f19717090c5c8ef720f8fb3aab`.
The original `concurrency4` branch remains at `7f1615c20d1b82c4cbde03470d362caedfa73018`.
The stack uses the repository's `stacked/` naming convention and gh-stack (stack #1999).
All PRs are drafts for review. Merge bottom-up through the repository's merge queue; do not use `gh stack merge` or "Enqueue stack".

## Published layers

| Layer | PR | Branch | Commit |
|---|---|---|---|
| 1 | [#1997](https://github.com/sirius-db/sirius/pull/1997) | `stacked/concurrency-01-query-errors-workers` | `4f57ac0d1` |
| 2 | [#1998](https://github.com/sirius-db/sirius/pull/1998) | `stacked/concurrency-02-lifecycle-contract` | `fe4529687` |
| 3 | [#2000](https://github.com/sirius-db/sirius/pull/2000) | `stacked/concurrency-03-lifecycle-integration` | `563b8189b` |
| 4 | [#2001](https://github.com/sirius-db/sirius/pull/2001) | `stacked/concurrency-04-memory-progress` | `1a3c75515` |
| 5 | [#2002](https://github.com/sirius-db/sirius/pull/2002) | `stacked/concurrency-05-session-options` | `94c22487b` |
| 6 | [#2003](https://github.com/sirius-db/sirius/pull/2003) | `stacked/concurrency-06-metadata-ownership` | `9f9719b30` |
| 7 | [#2004](https://github.com/sirius-db/sirius/pull/2004) | `stacked/concurrency-07-pin-publication` | `173ccd98a` |
| 8 | [#2005](https://github.com/sirius-db/sirius/pull/2005) | `stacked/concurrency-08-cuda-streams` | `739b140b1` |
| 9 | [#2006](https://github.com/sirius-db/sirius/pull/2006) | `stacked/concurrency-09-scan-capacity` | `78e2033cc` |
| 10 | [#2007](https://github.com/sirius-db/sirius/pull/2007) | `stacked/concurrency-10-runtime-health` | `08d2d6463` |
| 11 | [#2008](https://github.com/sirius-db/sirius/pull/2008) | `stacked/concurrency-11-concurrent-admission` | `a6d19fdee` |
| 12 | This documentation PR | `stacked/concurrency-12-review-record` | Documentation only |

Every implementation layer was built before its commit and before constructing the next layer.
PR descriptions provide purpose, invariants, reviewer reading order, validation and limits.
The largest coupled layer is PR 3: publication, continuous resource ownership and retirement must agree.
Its broad file list includes constructor/registration migration of existing fixtures.
The final mandatory-registry and submission-helper simplifications appear there immediately;
pin publication similarly introduces the final metadata API without obsolete attach methods.

## Validation by layer

Counts below are per invocation and overlap across layers; they are not a unique-test total.

### Layer 1

78 cases / 518 assertions; build and hooks passed.

### Layer 2

34 cases / 167 assertions; CPU registry ASan/UBSan and TSan probes passed; build and hooks passed.

### Layer 3

127 cases / 777 assertions plus 2 OOM cases / 8 assertions; build and hooks passed.

### Layer 4

74 cases / 296 assertions; build and hooks passed.

### Layer 5

Full build and commit hooks passed. Configuration, session isolation/RESET, runtime fallback and telemetry/NVTX tests passed: 117 cases / 1,076 assertions. Single-GPU execution; multi-GPU tests excluded from this gate.

### Layer 6

Full build and commit hooks passed. Query telemetry ownership, telemetry context, local Iceberg decoding/layout and cache eviction tests passed: 43 cases / 256 assertions. Dataset-dependent Iceberg integration was excluded from this focused gate.

### Layer 7

Full build and commit hooks passed. With SIRIUS_EXP_LATE_MAT=1, pin registry epochs, initial metadata lookup, uniqueness/merge provenance, retained handles and SQL MVCC tests passed: 50 cases / 735 assertions.

### Layer 8

Full build and commit hooks passed. Stream exclusivity/lifetime, memory-prefetch accounting and single-GPU dynamic-filter suites passed: 302 cases / 11,425 assertions. Hardware: one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB). Multi-GPU execution remains unrun.

### Layer 9

Full build and commit hooks passed. Shared scan budgets, readahead/gatekeeper cancellation and demand progress, cache eviction, query scan state and dispatcher tests passed: 56 cases / 354 assertions. Concurrent SQL scan oversubscription is qualified in the later admission layer.

### Layer 10

Full build and commit hooks passed. Lifecycle diagnostics/resource release, failed-runtime refusal, completion observers, fatal/recoverable classification, prefetch and SQL lifecycle tests passed: 51 cases / 413 assertions. No destructive CUDA fault injection was performed.

### Layer 11

Full build and commit hooks passed. Combined concurrency/lifecycle/settings/scan/fragment gate: 240 cases / 2,146 assertions, including all 20 real concurrent SQL variants. Final pin/MVCC gate with SIRIUS_EXP_LATE_MAT=1: 50 cases / 735 assertions. Initial extraction matched the preserved source exactly. The full runner then exposed a stale priority-wrap test; follow-up a6d19fdee corrects that test and three obsolete comments without changing runtime behavior. The follow-up build and hooks passed, and queue-ownership/query-ID/admission tests passed: 18 cases / 71 assertions. Standard runner results are recorded separately in the final review record.

## Stack-wide validation

- All eleven implementation layers compiled individually with `pixi run make` before commit. The admission test/comment follow-up also compiled and passed commit hooks.
- Final combined gate: **240 cases / 2,146 assertions passed**, including all 20 concurrent SQL watchdog variants.
- Final pin/MVCC gate with `SIRIUS_EXP_LATE_MAT=1`: **50 cases / 735 assertions passed**.
- After correcting the stale priority-wrap test, `[task_index_keys],[query_id],[query_admission]`: **18 cases / 71 assertions passed**.
- Separate ordinary-runner late-materialization step: **109 cases / 6,489 assertions passed**.
- **The complete ordinary runner did not pass.** Shard 1 finished with 1,876 of 1,877 cases passing (7,411,266 of 7,411,267 assertions); its sole failure was the stale priority-wrap test corrected above. Shard 0 stopped producing output at `sirius_knn_search - ANN (IVF-Flat) l2 matches exact top-k` and was manually interrupted after several minutes. No complete shard-0 result is claimed. The corrected focused gate was rerun; the complete shards were not rerun.
- The ANN test reproduced the stall in isolation and hit the **120-second timeout (exit 124)**. It had reached the first `k = 1` search. Its cause has not been established, and it remains a qualification blocker before calling the full suite green. This stack preserves the original runtime implementation; no ANN behavior was changed during extraction.
- The broad runner stopped before its multi-GPU and late-materialization steps. Late materialization was subsequently run separately as reported above; multi-GPU qualification remains unrun on this one-GPU host. Environment-gated dataset/service tests must be qualified with their required fixtures; broad suite counts do not imply those integrations were exercised.

Reproducible follow-up commands (from the repository root):

```bash
# Complete runner: did not finish successfully; see the failure details above.
pixi run python scripts/run_unit_tests.py

# Regression correction and separate late-materialization gate: passed.
pixi run timeout 120 build/release/extension/sirius/test/cpp/sirius_unittest \
  '[task_index_keys],[query_id],[query_admission]'
pixi run python scripts/run_unit_tests.py --steps late_mat

# Isolated unresolved ANN stall: timed out with exit 124.
pixi run timeout 120 build/release/extension/sirius/test/cpp/sirius_unittest \
  'sirius_knn_search - ANN (IVF-Flat) l2 matches exact top-k'
```

The combined and pin gates were run before the test/comment-only follow-up. Their implementation is unchanged. Logs: `stack-combined-tests.log`, `stack-pin-tests.log`, `pr11-followup-tests.log`, `stack-late-mat.log`, `stack-unit-runner.log` and `stack-ann-isolation.log` in the local log directory listed below.

## Source equivalence and extraction adjustments

Before adding the review record, the implementation, tests and test registration were compared directly against the preserved source.
Use tree comparison, not range-diff, because commits were deliberately regrouped:

```bash
git diff concurrency4_stack_source_20261005 HEAD -- src test cmake .gitignore
```

The initial extraction was byte-for-byte identical on those paths. The final validation pass then exposed a stale test: `test_task_index_keys.cpp` still expected query-priority overflow to wrap, although admission now rejects it.
The admission layer updates that test to use two valid, different query IDs for ownership and priority; it also corrects three stale comments in `task_creator.cpp`, `gpu_pipeline_task.hpp` and `sirius_pipeline.hpp`. Runtime behavior is unchanged. These four files are the complete implementation/test delta from the preserved source.

The remaining differences from the source are documentation:

- Historical banners on reports 101, 103 and 104, preserving their original development record.
- A completion pointer in plan 102 and this extraction/validation report.
- Corrections in `pipeline-execution.md`: query retirement leaves shared pools running, and retry backoff uses a scheduler deadline instead of sleeping in a worker.

Intermediate extraction adjustments do not remain in the final implementation:

- Layer 3 initially included the stream-exclusivity test too early; it was moved to layer 8 and layer 3 was clean-rebuilt.
- Layer 4 needed overflow-checked allocation while execution still used a serialized window. A local checked counter served that intermediate version; layer 11 replaced it with the final admission-ticket implementation. The failed initial build was followed by the required clean rebuild.
- Health diagnostics and fatal propagation were deferred until layer 10; actual concurrent SQL admission remained serialized until layer 11.
- Feature documentation and fixture migrations accompany their respective layers; the cross-cutting historical reports are retained here.

## Remaining qualification and review

- Local hardware: one NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB.
- The explicit `[concurrent_queries_mgpu]` target compiles but requires at least two real GPUs and was not executed here. Single-GPU passing results do not establish multi-GPU correctness.
- Destructive CUDA fault injection was not performed; classification, observer behavior and failed-runtime refusal have non-destructive regression coverage.
- Dataset/service-dependent gates and any ordinary-runner limitations are listed in the stack-wide validation results above.
- Local validation is separate from GitHub CI and maintainer review. Draft status has been preserved.
- Larger simplification opportunity 5 (a further health-ownership redesign) remains outside this extraction.

Raw local logs and the PR description files are retained under `/home/wmalpica/repos/logs/concurrency-stack-20261005/`.
