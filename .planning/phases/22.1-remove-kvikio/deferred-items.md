# Phase 22.1 Deferred Items

## Out-of-scope discoveries (not auto-fixed by 22.1-02)

### REQUIREMENTS.md hygiene — IO-MGPU-03 not yet registered
- **Found during:** 22.1-02 final state-update pass.
- **Issue:** Plan 22.1-02's frontmatter declares `requirements: [IO-MGPU-03]`, but `IO-MGPU-03` is not present in `.planning/REQUIREMENTS.md` (gsd-tools `requirements mark-complete IO-MGPU-03` returned `not_found`).
- **Why deferred:** Plan correctness is independent of the REQUIREMENTS.md registration — the plan delivers the policy flip per its own `<must_haves>` block. The missing requirement entry is a project-hygiene gap that the phase planner should backfill (likely during `/gsd:complete-phase` for 22.1).
- **Resolution path:** Whoever runs `/gsd:complete-phase 22.1` should add IO-MGPU-03 to REQUIREMENTS.md (description: "datasource_factory strict policy — registry resolves all schemes or throws kvikio-rejection text") and then mark it complete.

### Wave-2 in-flight collision RESOLVED: sirius_engine.cpp:385 caller wiring
- **Found during:** 22.1-04 Task 2 (mcp build gate, first run).
- **Issue (now resolved):** Sibling Plan 22.1-05's commit `5c3522b` added a `metadata_ioctx` parameter to `read_iceberg_delete_data` ahead of wiring the caller at `src/sirius_engine.cpp:385`. First mcp build during Plan 22.1-04 verification failed at this site.
- **Resolution:** Plan 22.1-05's commit `9ea53e9 feat(22.1-05): forward metadata_ioctx through public API to materialize step` (landed during my plan execution) wired the caller. Re-run of mcp build during Plan 22.1-04 Task 2 PASSED at exit 0 / 9.9s; `[pin_mgpu]` 2/2 PASS / 46 assertions / 7.1s.
- **Note:** This is documented for posterity as a Wave-2 parallel-execution coordination event — temporary header-vs-caller commit ordering inversion across sibling plans, self-resolved by Plan 22.1-05's own Task 4.

### Plan 22.1-06 commit `587a950` introduces liburing/concurrentqueue BLOCK_SIZE macro collision — RESOLVED
- **Found during:** 22.1-05 Task 4 mcp build verification (after `587a950` landed on the branch in parallel between my Task 3 and Task 4 commits).
- **Issue:** Two test translation units (`test/cpp/scan/test_parquet_scan_task.cpp` + `test/cpp/scan/test_parquet_split_provider.cpp`) FAIL to compile at branch HEAD = `587a950`. The error chain:
  ```
  test_helpers_ioctx.hpp:36 -> io/uring/uring_ioctx.hpp:20 -> io/uring/uring_reactor.hpp:24
    -> liburing.h:18 -> liburing/io_uring.h:11
    [BLOCK_SIZE macro] vs [duckdb/third_party/concurrentqueue/blockingconcurrentqueue.h:38: static const size_t BLOCK_SIZE = ConcurrentQueue::BLOCK_SIZE;]
  ```
  `liburing/io_uring.h` defines a `BLOCK_SIZE` macro; the concurrentqueue header declares a static const member of the same name. The macro replaces the identifier and the parser hits "expected unqualified-id before numeric constant".
- **Why out-of-scope for 22.1-05:** None of my plan's files (`iceberg_metadata_reader.cpp`, `iceberg_metadata_reader.hpp`, `sirius_engine.cpp`) appear in the error chain. The new file `test/cpp/scan/test_helpers_ioctx.hpp` and the `#include` modifications to `test_parquet_*.cpp` are owned by Plan 22.1-06 (commit `587a950 feat(22.1-06): delete unit-test fallback at parquet_split_provider:295 (D-08 site #7)`).
- **Empirical isolation experiment performed:** Checked out the pre-22.1-06 versions of the test files + removed `test_helpers_ioctx.hpp` from worktree, then ran mcp build → Exit 0, 9.8s. This proves my Plan 22.1-05 files build clean.
- **Resolution (Plan 22.1-03 Task 3, commit `a4d62fa`):** Plan 22.1-03 owner picked option (2) from the recommended paths — reordered the `#include <scan/test_helpers_ioctx.hpp>` to come AFTER `<utils/utils.hpp>` (which transitively pulls `blockingconcurrentqueue.h`) in BOTH `test_parquet_scan_task.cpp` and `test_parquet_split_provider.cpp`. Added an explanatory comment block in each test file mirroring the canonical Phase 19 IO-15 pattern at `src/op/scan/parquet_scan_task.cpp:30-35`. mcp build → Exit 0, 18.5s, [72/72] Linking sirius_unittest. `[multi_gpu_foundation]` 7/7 PASS, `[mgpu]` 16/16 PASS, `[TPC-H][parquet]` 22/22 PASS post-fix.
- **Note:** This was a Rule 3 auto-fix-blocking-issue deviation in Plan 22.1-03 — the test files were not in 22.1-03's `files_modified` list, but the include-order fix was the minimum-touch unblocker for 22.1-03's MCP build gate. Documented as a deviation in 22.1-03-SUMMARY.md.
