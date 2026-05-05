---
phase: 16-cucascade-submodule-rebase-pin-recovery
plan: 04
subsystem: infra
tags: [git, rebase, cherry-pick, cucascade, stream-lineage, writer-event, raii, conflict-resolution]

# Dependency graph
requires:
  - phase: "16-03"
    provides: "3 commits on top of 73d00c4 (Groups 1, 3, 2); P2P probe + stream wiring in converter"
provides:
  - "cucascade branch phase16-rebase-wip: 4 commits on top of 73d00c4 (Groups 1, 3, 2, 4)"
  - "gpu_table_representation: REQUIRED writer_stream on both ctors (simple-table AND cudf::table_view template)"
  - "record_writer_event(rmm::cuda_stream_view) + [[nodiscard]] cudaEvent_t get_writer_event() const accessors"
  - "cudaEventDestroy in ~gpu_table_representation (RAII event cleanup)"
  - "convert_gpu_to_gpu: cudaStreamWaitEvent(target_stream, src.get_writer_event(), 0) Phase 13 fix"
  - "read_only_data_batch::get_writer_event() proxy via dynamic_cast (D-B3)"
  - "All 12+ ctor call sites updated to 3-arg form: tests, benchmarks, bandwidth_profiler, wrap_column helper"
  - "Build COMPILE-CLEAN: cucascade library + tests + benchmarks all link"
affects: [16-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Full re-implementation per D-D2: when #117 reshaped the class, treat as rewrite not merge"
    - "writer_stream REQUIRED at construction time (not optional): compile-time enforced via no 2-arg ctor"
    - "cudaEvent_t _writer_event{nullptr}: appended after _table variant; destroyed in dtor if non-null"
    - "read_only_data_batch proxy: dynamic_cast through get_data() — safe nullptr path for non-GPU repr"
    - "3-arg ctor with rmm::cuda_stream_view{} as default stream in test/benchmark setup code"

key-files:
  created: []
  modified:
    - cucascade/include/cucascade/data/gpu_data_representation.hpp  # merged API: #117 RAII shape + Group 4 writer_stream/event
    - cucascade/src/data/gpu_data_representation.cpp                 # simple ctor + record_writer_event + dtor + variant dispatch
    - cucascade/src/data/representation_converter.cpp                # convert_gpu_to_gpu column-tree walk + cudaStreamWaitEvent
    - cucascade/include/cucascade/data/data_batch.hpp                # read_only_data_batch::get_writer_event() proxy (D-B3)
    - cucascade/test/data/test_data_batch.cpp                        # 5 ctor call sites -> 3-arg
    - cucascade/test/data/test_representation_converter.cpp          # ctor sites -> 3-arg
    - cucascade/test/data/test_disk_host_converters.cpp              # ctor site -> 3-arg
    - cucascade/test/data/test_gpu_disk_converters.cpp               # 4 ctor sites -> 3-arg
    - cucascade/test/data/test_data_representation.cpp               # wrap_column helper + all call sites -> 3-arg
    - cucascade/src/data/bandwidth_profiler.cpp                      # 1 ctor site -> 3-arg (missed in original cherry-pick)
    - cucascade/benchmark/benchmark_disk_converter.cpp               # 11 ctor sites -> 3-arg with stream.view()
    - cucascade/benchmark/benchmark_representation_converter.cpp     # 6 ctor sites -> 3-arg with appropriate stream

key-decisions:
  - "D-D2 full re-implementation applied: gpu_data_representation.hpp/cpp treated as clean rewrite against #117 RAII shape (variant, owning_table_view, get_table_view, release_table(stream)) with Group 4 writer_stream required param + writer_event accessors grafted in"
  - "read_only_data_batch::get_writer_event() proxy: dynamic_cast<gpu_table_representation*>(get_data()) — returns nullptr for non-GPU representations without throwing"
  - "3-arg ctor in benchmarks: setup-phase reprs use stream.view() of their local setup/warmup stream; thread-pool reprs use rmm::cuda_stream_view{} (created before streams are assigned)"
  - "wrap_column helper in test_data_representation.cpp: added default writer_stream parameter (rmm::cuda_stream_view{}) to avoid breaking existing 2-arg call sites"
  - "Build verified COMPILE-CLEAN: cucascade library + cucascade_tests + cucascade_benchmarks all link; ctest deferred to 16-05 (GPU environment issue in sandboxed shell, not a logic failure)"

patterns-established:
  - "Rule 1 auto-fix: bandwidth_profiler.cpp and 17 benchmark ctor sites were missed in original cherry-pick (9dddf77) but caught during build verification — amended into Group 4 commit (1c1e648)"
  - "wrap_column test helper: when a helper constructs gpu_table_representation, add optional writer_stream param rather than hardcoding rmm::cuda_stream_view{} internally"

requirements-completed: [CC-02, CC-03]

# Metrics
duration: 45min
completed: 2026-05-05
---

# Phase 16 Plan 04: Cherry-pick Group 4 (Phase 13 stream-lineage) Summary

**RAII-aware re-implementation of gpu_table_representation writer_stream/writer_event API on top of #117 variant model, with cudaStreamWaitEvent in convert_gpu_to_gpu and read_only_data_batch::get_writer_event() proxy — all 12+ ctor sites compile-clean**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-05-05T~20:00Z
- **Completed:** 2026-05-05T~20:45Z
- **Tasks:** 5 (Tasks 1-4 pre-completed in prior session as commit 9dddf77, Task 5 wrap-up executed now)
- **Files modified:** 12 (cucascade)

## Accomplishments

- Source work from prior session (9dddf77, now amended to 1c1e648) fully verified against plan acceptance criteria
- All D-B1/D-B2/D-B3/D-B4 traceability confirmed via grep gates
- Build verified COMPILE-CLEAN: cucascade library + tests + benchmarks (cmake --build exits 0)
- .bak cleanup: `test/data/test_data_representation.cpp.bak` deleted (no value, duplicate of pre-edit committed state)
- Round 4 entry appended to 16-rebase-log.md with full resolution notes

## D-B Traceability

| Req | Description | Implementation |
|-----|-------------|----------------|
| D-B1 | record_writer_event + get_writer_event on gpu_table_representation | `gpu_data_representation.hpp` lines (accessor decls) + `gpu_data_representation.cpp` (impls using cudaEventCreateWithFlags + cudaEventRecord) |
| D-B2 | writer_stream REQUIRED on BOTH ctors | `gpu_data_representation.hpp` line 67 (simple ctor) + line 86 (template ctor); 0-arg and 2-arg form removed |
| D-B3 | read_only_data_batch::get_writer_event() proxy | `data_batch.hpp` — dynamic_cast<gpu_table_representation*>(get_data()); returns nullptr for non-GPU repr |
| D-B4 | Recording stays caller-controlled (explicit record_writer_event calls) | Ctor body calls record_writer_event(writer_stream); no auto-record in set_data or accessors |

## Task Commits

Source work was pre-committed in prior session and amended here:

| Task | Name | Commit | Key files |
|------|------|--------|-----------|
| 1 | Re-implement gpu_data_representation.hpp + .cpp on #117 RAII | `1c1e648` (amended) | gpu_data_representation.hpp, gpu_data_representation.cpp |
| 2 | Replace convert_gpu_to_gpu with column-tree walk + cudaStreamWaitEvent | `1c1e648` (amended) | representation_converter.cpp |
| 3 | Add read_only_data_batch::get_writer_event() proxy | `1c1e648` (amended) | data_batch.hpp |
| 4 | Update test ctor sites + verify compile-clean | `1c1e648` (amended) | test_data_batch.cpp, test_representation_converter.cpp, test_disk_host_converters.cpp, test_gpu_disk_converters.cpp, test_data_representation.cpp |
| 5 | Update 16-rebase-log.md Round 4 section | (planning metadata commit) | 16-rebase-log.md |

**Plan metadata:** pending final docs commit

## Cucascade Commit Log (post-plan)

```
1c1e648 fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent
995bf4e fix(representation_converter): P2P override — target-bound stream, DMA probe at init
a1778f9 fix(pipeline_io_backend): reorder io_worker members so _thread is last
6236494 fix(memory): memory hygiene — Portable/Mapped pinning, ptds tracker, pool peer access
73d00c4 implement 3-class data_back model and get rid of state machine (#117)
```

`git -C cucascade rev-list --count 73d00c4..HEAD` = 4 (invariant preserved)

## Files Created/Modified

- `cucascade/include/cucascade/data/gpu_data_representation.hpp` — Merged #117 owning_table_view variant + Group 4 REQUIRED writer_stream on both ctors; record_writer_event/get_writer_event decls; _writer_event{nullptr} member; dtor decl
- `cucascade/src/data/gpu_data_representation.cpp` — Simple ctor calls record_writer_event(writer_stream); dtor destroys event; record_writer_event impl (cudaEventCreateWithFlags + cudaEventRecord); get_writer_event returns _writer_event; variant-dispatch methods (get_size_in_bytes, get_table_view, release_table, get_uncompressed_data_size_in_bytes, clone) copied verbatim from 73d00c4
- `cucascade/src/data/representation_converter.cpp` — convert_gpu_to_gpu: cudaStreamWaitEvent(target_stream.value(), source_repr.get_writer_event(), 0) before peer copy; column-tree walk implementation; Group 2's p2p_dma_supported routing preserved
- `cucascade/include/cucascade/data/data_batch.hpp` — read_only_data_batch::get_writer_event() const proxy (~7 LOC); dynamic_cast through get_data(); includes cuda_runtime.h
- `cucascade/test/data/test_data_batch.cpp` — 5 gpu_table_representation ctor sites updated to 3-arg (rmm::cuda_stream_view{})
- `cucascade/test/data/test_representation_converter.cpp` — ctor sites updated
- `cucascade/test/data/test_disk_host_converters.cpp` — ctor site updated
- `cucascade/test/data/test_gpu_disk_converters.cpp` — 4 ctor sites updated
- `cucascade/test/data/test_data_representation.cpp` — wrap_column helper signature updated (added optional writer_stream param); all call sites updated to pass rmm::cuda_stream_view{}
- `cucascade/src/data/bandwidth_profiler.cpp` — 1 ctor site updated (bootstrap_stream as writer_stream)
- `cucascade/benchmark/benchmark_disk_converter.cpp` — 11 ctor sites updated (stream.view() or stream for stream_view params)
- `cucascade/benchmark/benchmark_representation_converter.cpp` — 6 ctor sites updated (rmm::cuda_stream_view{} for setup-phase reprs; warmup_stream.view() and setup_stream.view() for warmup/setup reprs)

## Decisions Made

- **Full re-implementation (D-D2):** gpu_data_representation.hpp/cpp treated as a clean rewrite against #117's variant RAII shape, not a merge of conflict markers. The #117 variant dispatch (owning_table_view, std::any owner, cudf::table_view) was preserved verbatim from 73d00c4; Group 4's writer_stream/event additions grafted on top.
- **writer_stream in benchmark/bandwidth_profiler:** Missed in the original cherry-pick commit (9dddf77). Caught during build verification (Rule 1 auto-fix). Amended into Group 4 commit (1c1e648). Used `stream.view()` from each function's already-declared `rmm::cuda_stream stream` local for setup-phase construction sites; `rmm::cuda_stream_view{}` for thread-pool construction sites where no per-repr stream exists.
- **wrap_column helper:** Added `rmm::cuda_stream_view writer_stream = rmm::cuda_stream_view{}` as default parameter so call sites using `wrap_column(col, *space, rmm::cuda_stream_view{})` compile without requiring all existing 2-arg callers to change.
- **ctest outcome:** Runs in sandboxed shell return "no CUDA-capable device is detected" at runtime (rmm::cuda_stream constructor aborts). Build link step succeeded; this is an environment gap, not a logic failure. Full ctest gate deferred to 16-05 per plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing 3-arg ctor updates in bandwidth_profiler.cpp and all benchmark files**
- **Found during:** Task 4 (build verification)
- **Issue:** Original cherry-pick commit (9dddf77) updated test files but missed `src/data/bandwidth_profiler.cpp` (1 site), `benchmark/benchmark_disk_converter.cpp` (11 sites), `benchmark/benchmark_representation_converter.cpp` (6 sites), and `test/data/test_data_representation.cpp::wrap_column` helper signature (1 site + multiple call sites). Build failed with "no matching constructor" errors.
- **Fix:** Updated all missing sites with appropriate stream arguments (stream.view() for sites with a local rmm::cuda_stream; rmm::cuda_stream_view{} for thread-pool setup sites). Updated wrap_column to accept optional writer_stream parameter.
- **Files modified:** bandwidth_profiler.cpp, benchmark_disk_converter.cpp, benchmark_representation_converter.cpp, test_data_representation.cpp
- **Verification:** cmake --build exits 0; no "error:" lines in build output
- **Committed in:** `1c1e648` (amended Group 4 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — missed ctor update sites in benchmark/src files)
**Impact on plan:** Required for compile-clean gate. No scope creep — all sites are the same compile-time enforcement that was already applied to test files.

## Issues Encountered

- The original cherry-pick session committed 9dddf77 before running the build, so benchmark + bandwidth_profiler ctor sites were not caught. The wrap-up build verification (cmake --build) caught these immediately in the first build pass.

## Known Stubs

None — this plan is pure git rebase work with no new features or UI data flows.

## Build State

**COMPILE-CLEAN.** cucascade library (libcucascade.so + libcucascade.a), cucascade_tests, and cucascade_benchmarks all link. Build directory: `cucascade/build/` (CMakeCache.txt present).

ctest deferred to 16-05: the ctest invocation in the sandboxed shell returns "no CUDA-capable device is detected" at runtime (rmm::cuda_stream ctor abort). This is an environment issue, not a logic failure — GPUs are present on the host (nvidia-smi shows 2x RTX 6000 Ada). 16-05 will run ctest in the proper GPU-accessible environment.

## Next Phase Readiness

- 16-05 can run cucascade ctest and advance the submodule pin
- `git -C cucascade rev-list --count 73d00c4..HEAD` = 4 (Groups 1, 3, 2, 4 in apply-order)
- `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` exits 0
- CC-02 and CC-03 requirements closed by this plan
- CC-01 and CC-04 remain for 16-05 (pin advance + ctest gate)

## Self-Check: PASSED

- FOUND: `git -C cucascade rev-list --count 73d00c4..HEAD` = 4
- FOUND: `git -C cucascade merge-base --is-ancestor 73d00c4 HEAD` exits 0
- FOUND: cucascade HEAD = `1c1e648`
- FOUND: `grep -c "writer_stream" cucascade/include/cucascade/data/gpu_data_representation.hpp` = 12 (>= 4)
- FOUND: `grep -c "record_writer_event" cucascade/include/cucascade/data/gpu_data_representation.hpp` = 6 (>= 1)
- FOUND: `grep -c "get_writer_event" cucascade/include/cucascade/data/gpu_data_representation.hpp` = 2 (>= 1)
- FOUND: `grep -c "owning_table_view" cucascade/include/cucascade/data/gpu_data_representation.hpp` = 3 (>= 2)
- FOUND: `grep -c "cudaStreamWaitEvent" cucascade/src/data/representation_converter.cpp` = 1 (>= 1)
- FOUND: `grep -c "get_writer_event" cucascade/include/cucascade/data/data_batch.hpp` = 3 (>= 1)
- FOUND: `grep -c "dynamic_cast" cucascade/include/cucascade/data/data_batch.hpp` = 2 (>= 1)
- FOUND: Build exits 0 (cucascade_objects, cucascade_static, cucascade_shared, cucascade_benchmarks, cucascade_tests all Built target)
- FOUND: `cucascade/build/CMakeCache.txt` exists
- FOUND: No .bak files in cucascade tree (`git -C cucascade status --short` is empty)
- FOUND: 16-04-SUMMARY.md created at `.planning/phases/16-cucascade-submodule-rebase-pin-recovery/16-04-SUMMARY.md`

---
*Phase: 16-cucascade-submodule-rebase-pin-recovery*
*Completed: 2026-05-05*
