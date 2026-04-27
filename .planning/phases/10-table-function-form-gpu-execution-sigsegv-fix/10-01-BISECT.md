---
phase: 10-table-function-form-gpu-execution-sigsegv-fix
plan: 10-01
type: bisect
recorded: 2026-04-27T15:20:00Z
branch: feature/single-node-multi-gpu2
pre_bisect_head: 478c937b3eceef4a30bb1dfd97436fe82d700754
test_target: "gpu_execution - filter equality parquet"
test_target_location: "test/cpp/integration/test_gpu_execution_tpch.cpp:449"
sigsegv_signal: "exit=139"
regressing_commit: NONE
regressing_commit_n: N/A
---

# Phase 10 Plan 10-01 — TABLE_FUNCTION SIGSEGV bisect ledger

## Pre-bisect state
- Branch: feature/single-node-multi-gpu2
- HEAD: 478c937 docs(phase-10): plan 4 plans for TABLE_FUNCTION SIGSEGV closure
- Working tree: 6 modified src/ files (FU17 partial fix changes) + .planning/STATE.md + .ai-helper/commands.yaml stashed before bisect
- Stash refs: stash@{0} (planning artifacts: .planning/STATE.md, .ai-helper/commands.yaml), stash@{1} (FU17 partial fix: src/data/sirius_p2p_converter.cpp, src/include/pipeline/batch_lock_utils.hpp, src/op/sirius_physical_concat.cpp, src/op/sirius_physical_hash_join.cpp, src/op/sirius_physical_parquet_scan.cpp, src/op/sirius_physical_partition.cpp)
- Both stashes restored after bisect; HEAD confirmed restored to 478c937

## Test invocation note

Direct binary invocation (`./build/.../sirius_unittest 'gpu_execution - filter equality parquet'`) fails in the MCP sandbox with `cudaErrorContextIsDestroyed` / NVML-not-loaded because the sandbox does not expose the GPU driver to direct child processes. All test runs used `mcp__project-commands__run_command(name="unit-tests", filter="gpu_execution - filter equality parquet")` which routes through the MCP wrapper and correctly reaches the GPU. This is consistent with the MCP-only build/test policy (feedback_use_mcp_build.md, feedback_mcp_tests_scope.md).

## Bisect window (chronological, oldest to newest)

| # | Hash    | Short message                                                | build_exit | test_exit | Interpretation |
|---|---------|--------------------------------------------------------------|------------|-----------|----------------|
| 1 | 3b58258 | feat(09-01): add preferred_device_id accessors               | 0          | 0         | PASS |
| 2 | 863cc6c | fix(09-01): plumb target_gpu_id into local state             | 0          | 0         | PASS |
| 3 | 0c8068e | fix(09-01): two-tier preferred_device_id lookup              | 0          | 0         | PASS |
| 4 | a8a7985 | feat(09-02): add _batch_gpu_affinity map                     | 0          | 0         | PASS |
| 5 | c0e12f3 | fix(09-02): record affinity in select_target_gpu + reset     | 0          | 0         | PASS |

Interpretation values (emitted deterministically by Task 2, copied verbatim):
- `PASS`: build_exit==0 AND test_exit==0
- `SIGSEGV (139)`: build_exit==0 AND test_exit==139
- `OTHER_TEST_FAIL exit=<N>`: build_exit==0 AND test_exit not in {0, 139}
- `BUILD_FAIL exit=<N>`: build_exit != 0 (test skipped)

## Per-commit test output (MCP unit-tests, all identical)

### Commit 1 (3b58258)
```
Filters: gpu_execution - filter equality parquet
All tests passed (31 assertions in 1 test case)
```

### Commit 2 (863cc6c)
```
Filters: gpu_execution - filter equality parquet
All tests passed (31 assertions in 1 test case)
```

### Commit 3 (0c8068e)
```
Filters: gpu_execution - filter equality parquet
All tests passed (31 assertions in 1 test case)
```

### Commit 4 (a8a7985)
```
Filters: gpu_execution - filter equality parquet
All tests passed (31 assertions in 1 test case)
```

### Commit 5 (c0e12f3)
```
Filters: gpu_execution - filter equality parquet
All tests passed (31 assertions in 1 test case)
```

## Post-bisect HEAD validation (478c937 + FU17 partial fix changes)

After restoring all stashes and rebuilding at the pre-bisect HEAD (which includes FU17 partial fix changes), the same isolated test was run:

```
Filters: gpu_execution - filter equality parquet
FAILED: REQUIRE_FALSE( gpu_sorted->HasError() )
  gpu sorted error: Invalid Error: SiriusExecuteQuery error: Invalid Error:
  CUDA error encountered at: compute_column.cu:106: 709 cudaErrorContextIsDestroyed
  context is destroyed
test cases: 1 | 0 passed | 1 failed
assertions: 24 | 23 passed | 1 failed
Exit: 1
```

This is NOT a SIGSEGV (exit=139) but an assertion failure (exit=1) with `cudaErrorContextIsDestroyed`. The failure mode has changed from the 09-04 VALIDATION recording. The FU17 partial fix changes present at HEAD are likely causing the changed failure mode.

**Important implication:** The 09-04 VALIDATION SIGSEGV was observed during full-suite execution (`--abort ~[hive_partition]`), not isolated single-test execution. The isolated test at c0e12f3 (clean, no FU17 changes) passes cleanly. This means the SIGSEGV was either:
(a) Test-ordering dependent: some earlier test in the suite corrupted GPU state that the second `SELECT * FROM gpu_execution(...)` call then hit, or
(b) Only manifests under specific environmental conditions not reproduced in isolated runs.

## Conclusion

**Regressing commit:** NONE — no commit in the 5-commit window reproduces the SIGSEGV (or any failure) when the test is run in isolation.

**Interpretation:** The crash is NOT introduced by Plans 09-01 or 09-02 source edits when the test runs in isolation. The 09-04 SIGSEGV occurred during full-suite execution, suggesting test-ordering dependency. Two leading hypotheses for Phase 10-02:

1. **Test-ordering dependency:** An earlier test in the suite (`--abort ~[hive_partition]` run) leaves some GPU context state (e.g., CUDA context, stream, memory allocation) in a corrupted or destroyed state. When `gpu_execution - filter equality parquet` runs second (as the second `SELECT * FROM gpu_execution(...)` call), it encounters the already-destroyed context. This is consistent with the 09-04 VALIDATION evidence: the FIRST `CALL gpu_execution(...)` completes cleanly; only the SECOND call SIGSEGVs.

2. **FU17 partial fix interference:** The FU17 changes at HEAD (src/data/sirius_p2p_converter.cpp and related files) change the error mode from SIGSEGV to `cudaErrorContextIsDestroyed`. These changes are probe/logging additions but include a `cudaDeviceSynchronize` call that could alter memory ordering and expose the underlying bug differently.

**Revised recommended probe for Plan 10-02:** Run the full unit-test suite at the clean state (c0e12f3, without FU17 changes) with `--abort ~[hive_partition]` to reproduce the original SIGSEGV. Then attach gdb to the crashing binary. The hypothesis H2 (TABLE_FUNCTION vs CALL-form result materialization divergence) remains the leading structural hypothesis; the SIGSEGV is context-dependent, not commit-specific.

## Restore confirmation
- Pre-bisect HEAD: `478c937b3eceef4a30bb1dfd97436fe82d700754`
- Post-bisect HEAD: `478c937b3eceef4a30bb1dfd97436fe82d700754`
- Match: YES
- Working tree clean for src/ and test/cpp/ (bisect commits): YES — verified before each checkout; src/ had only FU17 stashed changes which were restored cleanly after bisect
- Stash restoration: Both stashes restored successfully (exit=0 for both pops)

## Raw transcript
Full per-commit transcript at `/tmp/claude-1002/sirius-ph10-bisect.log` (also: per-commit `/tmp/claude-1002/sirius-ph10-c{1..5}-build.log` and `.exit` files).

---
*Phase: 10-table-function-form-gpu-execution-sigsegv-fix*
*Plan: 10-01 (bisect)*
*Recorded autonomously: 2026-04-27T15:20:00Z*
