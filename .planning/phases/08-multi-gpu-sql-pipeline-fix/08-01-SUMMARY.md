---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 01
subsystem: gpu-pipeline

tags: [cuda, rmm, cucascade, stream-pool, multi-gpu, cudf, parquet-scan]

# Dependency graph
requires:
  - phase: 07-p2p-direct-transfer-adaptive-scan-partitioning
    provides: "Pattern 2 (sirius-side converter override) reference body in src/data/sirius_p2p_converter.cpp + `[mgpu-audit]` logging baseline"
provides:
  - "Per-GPU stream pool map (`_gpu_stream_pools`) in duckdb_scan_executor, keyed by device_id"
  - "Target-bound stream acquire + dispatch-guard idiom for the scan path (mirrors Pattern 2 shape)"
  - "Build-gate-verified FIX-01 implementation awaiting N=2 runtime confirmation at 08-06"
affects:
  - 08-02-fix-02-probe
  - 08-03-test-02
  - 08-04-audit
  - 08-05
  - 08-06-ship-gate

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-GPU `exclusive_stream_pool` map in executors that are shared across GPUs (vs. per-executor single-pool for single-GPU executors)"
    - "Two-phase device guard: (1) rmm::cuda_set_device_raii around acquire_stream() so lazy stream creation binds to correct device; (2) rmm::cuda_set_device_raii at dispatch-lambda entry so the GPU-agnostic worker thread is pinned before any cudf/RMM call"
    - "Hoisting select_target_gpu() to the top of manager_loop so dispatch-site captures reach it via lexical scope"

key-files:
  created:
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-REPRODUCTION.md"
  modified:
    - "src/include/op/scan/duckdb_scan_executor.hpp (replaced `_stream_pool` member with `_gpu_stream_pools` map)"
    - "src/op/scan/duckdb_scan_executor.cpp (per-GPU pool population in constructor + hoisted target_gpu_id + target-bound stream acquire + dispatch lambda device guard)"

key-decisions:
  - "Hoisted target_gpu_id selection above the parquet-only block rather than duplicating select_target_gpu() at the dispatch site. Rationale: single source of truth per batch; non-parquet scan tasks now also get a well-defined target device; matches the plan intent that dispatch-lambda captures target_gpu_id into its device guard."
  - "Used a lambda-initialized `cucascade::memory::borrowed_stream exc_stream = [&] {...}()` to express 'acquire under device guard, then hold after guard expires' — this keeps the acquire_guard scope minimal (just the acquire call) while the borrowed_stream owns the cleanup callback."
  - "Committed Task 1 (data model change) and Task 2 (dispatch rewrite) as two separate commits even though only Task 2's tip state builds cleanly. The plan's acceptance criteria for Task 1 explicitly require Task 2's matches to land, so the pair is a joint build-gate by design."

patterns-established:
  - "Pattern: when an executor is shared across multiple GPUs, its stream pool must be a per-GPU map not a single-pool-to-GPU-0. Constructor populates under `rmm::cuda_set_device_raii`; dispatch site looks up by target_gpu_id also under `rmm::cuda_set_device_raii`."
  - "Pattern: inside any dispatch lambda whose worker thread is GPU-agnostic, open the lambda body with `rmm::cuda_set_device_raii dispatch_guard{rmm::cuda_device_id{target_gpu_id}}` BEFORE any cudf/RMM call — cudaSetDevice is thread-local and can't be relied on from an earlier caller."

requirements-completed: [FIX-01]

# Metrics
duration: 6min
completed: 2026-04-22
---

# Phase 08 Plan 01: Per-GPU Stream Pool in `duckdb_scan_executor` Summary

**Replaced the single `_stream_pool` (bound to GPU 0) with an `unordered_map<int, unique_ptr<exclusive_stream_pool>>` keyed by device_id, and rewrote the scan dispatch site to acquire + dispatch under paired `rmm::cuda_set_device_raii` guards — closing FIX-01 at the build-gate level.**

## Performance

- **Duration:** ~6 min (wall clock, 319s)
- **Started:** 2026-04-22T01:13:54Z
- **Completed:** 2026-04-22T01:19:13Z
- **Tasks:** 3 (2 code, 1 docs)
- **Files modified:** 2 code files + 1 new reproduction note

## Accomplishments

- Root-cause fix for the v1.1 post-ship `cudaErrorInvalidValue` at `cuda_memcpy.cu`: scan tasks destined for GPU 1 now run on a stream bound to GPU 1 (not GPU 0), under a matching thread-local device context. The symptom (stream-device / allocation-device mismatch) is structurally eliminated on the scan path.
- Idiom replicated from `src/data/sirius_p2p_converter.cpp` (Pattern 2) into `src/op/scan/duckdb_scan_executor.cpp`: the "device-guard + target-bound stream" shape is now established as the standard for any cross-device dispatch site in Sirius.
- HYG-02 invariant preserved: zero net-new `rmm::cuda_stream_default` uses in modified files (baseline across `src/` remains 41, verified pre-and-post by grep).
- Build-gate passed via `mcp__project-commands__run_command build` (exit 0) — both the static `sirius.duckdb_extension` and the loadable variant compile cleanly; all downstream linked targets (unittest, duckdb CLI, libduckdb.so) link.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add `_gpu_stream_pools` map to duckdb_scan_executor header and populate in constructor** — `2ff8a3c` (feat)
2. **Task 2: Rewrite manager_loop dispatch site to acquire stream from target-GPU pool with device guard** — `2150777` (fix)
3. **Task 3: Manual 2-GPU reproduction smoke test to confirm FIX-01 closes the root cause** — `0d03ca1` (docs — single-GPU-host deferral recorded)

**Plan metadata:** pending final commit after SUMMARY.md + STATE.md + ROADMAP.md updates.

## Files Created/Modified

- `src/include/op/scan/duckdb_scan_executor.hpp` — replaced `std::unique_ptr<cucascade::memory::exclusive_stream_pool> _stream_pool;` with `std::unordered_map<int, std::unique_ptr<cucascade::memory::exclusive_stream_pool>> _gpu_stream_pools;` (plus comment explaining why).
- `src/op/scan/duckdb_scan_executor.cpp` — constructor now populates `_gpu_stream_pools` one pool per GPU under `rmm::cuda_set_device_raii`. `manager_loop` hoists `select_target_gpu()` above the parquet-only block, then acquires `exc_stream` from `_gpu_stream_pools[target_gpu_id]` inside an acquire-guard lambda, and opens the dispatch lambda body with a second `rmm::cuda_set_device_raii` pinned to `target_gpu_id` before any cudf/RMM call.
- `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-REPRODUCTION.md` (new) — records host availability (single-GPU, no driver), static invariant check, expected N=2 runtime behavior, and the explicit handoff to Plan 08-06 for N=2 confirmation.

## Static Invariants (all green)

| Check                                                             | Result                                                                                                                                                              |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_gpu_stream_pools` declared in hpp                               | 1 match (line 197 of `duckdb_scan_executor.hpp`)                                                                                                                    |
| `_gpu_stream_pools` used in cpp                                   | 3 matches (line 70 constructor emplace, lines 350-351 dispatch `find` + end check)                                                                                  |
| Old `_stream_pool` singular member                                | 0 matches (removed)                                                                                                                                                 |
| Old `_stream_pool->acquire_stream(...)` call                      | 0 matches (replaced)                                                                                                                                                |
| `rmm::cuda_set_device_raii` with `target_gpu_id` or `dev_id`      | 3 matches: constructor per-GPU guard (line 69), acquire guard in manager_loop (line 366), dispatch lambda guard (line 382)                                          |
| `rmm::cuda_stream_default` in modified files                      | 0 matches (HYG-02 preserved)                                                                                                                                        |
| `rmm::cuda_stream_default` baseline across `src/` (context check) | 41 — unchanged vs. pre-plan baseline                                                                                                                                |
| MCP `build`                                                       | exit 0                                                                                                                                                              |
| REPRODUCTION.md exists + contains deferral marker                 | yes ("Single-GPU host — no GPU available")                                                                                                                          |

## Decisions Made

- **Hoist `select_target_gpu()` to the top of the manager_loop body.** The plan prescribed replacing the dispatch-site `_stream_pool->acquire_stream(...)` with a target-GPU-scoped variant, but `target_gpu_id` was originally declared inside the parquet-only block (scope ends before the dispatch site). Hoisting gives the dispatch site lexical access and ensures non-parquet scan tasks (duckdb_scan_task, cpu_source_task) also route through a well-defined device. No behavior regression for non-parquet paths: the same GPU selection logic that ran for parquet now runs for all scan tasks (which is arguably already correct — those tasks will also benefit from the device guard).
- **IIFE pattern for acquire_guard.** Used `borrowed_stream exc_stream = [&] { rmm::cuda_set_device_raii g{...}; return pool_iter->second->acquire_stream(...); }();` so the acquire_guard only covers the acquire call (the borrowed_stream itself carries its release-callback across the dispatch lambda's lifetime, independent of the outer device context). This matches the `src/data/sirius_p2p_converter.cpp` Pattern 2 shape where `source_guard` covers only the pack, not the subsequent target-side work.
- **Defer 2-GPU runtime reproduction to 08-06.** Per the plan's HARDWARE AVAILABILITY NOTE and the explicit single-GPU host detection (`nvidia-smi -L` failed — no NVIDIA driver active), Task 3 records a deferral rather than attempting a runtime test that cannot produce meaningful signal. The full confirmation happens at Plan 08-06 (SF100 ship gate) on the verification host.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Hoisted `target_gpu_id` out of parquet-only block**

- **Found during:** Task 2 (rewriting dispatch site)
- **Issue:** The plan's Task 2 action said "find the region where `select_target_gpu()` is called followed by `_stream_pool->acquire_stream(...)`" as if they were adjacent, but in the actual source tree `target_gpu_id` was declared inside `if (scan_task->is<parquet_scan_task>())` at line 299 and went out of scope at line 336, while the dispatch site was at line 338. Without hoisting, the dispatch site cannot reference `target_gpu_id`, making the plan's prescribed change structurally impossible.
- **Fix:** Moved `int target_gpu_id = select_target_gpu();` from line 299 (inside the parquet `if`) to a new line just after `scan_task` is cast and before the parquet `if`. The parquet block now uses the hoisted variable; the dispatch site below also uses it. No redundant calls to `select_target_gpu()`.
- **Files modified:** `src/op/scan/duckdb_scan_executor.cpp` (in the Task 2 commit, not a separate follow-up)
- **Verification:** MCP build exits 0; grep for `select_target_gpu()` count in file is still 1 call-site (not duplicated).
- **Committed in:** `2150777` (Task 2 commit)

### Task-1 build-gate coupling (structural, not a deviation)

Plan Task 1's acceptance criterion "MCP build exits 0" cannot be satisfied by Task 1's changes alone: removing `_stream_pool` from the header while the dispatch site at `cpp:338` still references it leaves the build broken. This is acknowledged in the plan itself (Task 1 acceptance: "`grep -c '_gpu_stream_pools' cpp` >= 2 (... **Task 2 will add the second**)"). The MCP build gate is passed at Task 2's tip, not Task 1's. Documented as a structural reading of the plan; Task 1 and Task 2 are one logical change split across two commits for atomicity.

---

**Total deviations:** 1 auto-fixed (1 blocking) + 1 structural plan-coupling note.
**Impact on plan:** No scope creep. The hoist was the minimum change needed to make the plan's prescribed dispatch-site rewrite compile.

## Issues Encountered

- `nvidia-smi -L` on the current host failed (no driver active) — reported by the host_constraints block in the executor prompt. This made Task 3's runtime smoke test impossible; the deferral pattern was taken per the plan's HARDWARE AVAILABILITY NOTE.
- A spurious `.planning/STATE.md` modification was picked up at the start of execution (from the `init execute-phase` gsd-tools call, which normalizes frontmatter fields). Not related to plan work; will be consumed by the state updates at plan completion.

## Known Stubs

None. This plan landed complete, compiling, functional code. The "single-GPU host deferral" in REPRODUCTION.md is a **verification** deferral (the fix ships; the N=2 runtime confirmation moves to 08-06), not a code stub.

## Next Phase Readiness

- **FIX-01 build-gated complete.** Scan executor now has per-GPU stream pools + target-bound dispatch. The shape matches the proven Pattern 2 idiom from Plan 07-02.
- **08-02 (FIX-02 probe) is unblocked.** The probe examines whether any *other* cross-device memcpy site (beyond the scan path) still leaks. If FIX-01 closes the whole bug, 08-02 logs a clean probe. If a second site shows up, 08-02 applies Pattern 2 there.
- **08-06 (SF100 ship gate) is the canonical N=2 confirmation.** TPC-H Q1 (and the full suite) on `num_gpus: 2` with SF100 parquet will prove FIX-01 closes the v1.1 symptom end-to-end on real hardware.
- **No new blockers.** The single-GPU host is expected — verification hardware is a separate machine.

## Self-Check: PASSED

**Files verified to exist:**

- FOUND: `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-01-REPRODUCTION.md`
- FOUND: `src/include/op/scan/duckdb_scan_executor.hpp` (modified: `_gpu_stream_pools` member)
- FOUND: `src/op/scan/duckdb_scan_executor.cpp` (modified: per-GPU pool population + target-bound dispatch)

**Commits verified to exist:**

- FOUND: `2ff8a3c` (Task 1 — feat: add _gpu_stream_pools map)
- FOUND: `2150777` (Task 2 — fix: dispatch on target-GPU stream)
- FOUND: `0d03ca1` (Task 3 — docs: reproduction deferral)

**Grep invariants verified:**

- `_gpu_stream_pools` matches across hpp+cpp: 4 (>= 3 required)
- `rmm::cuda_set_device_raii` matches across cpp: 3 (>= 3 required per plan's verification block)
- `rmm::cuda_stream_default` matches in modified files: 0 (HYG-02 preserved)
- MCP `build` exits 0: confirmed at end of Task 2 and again after Task 3.

---
*Phase: 08-multi-gpu-sql-pipeline-fix*
*Completed: 2026-04-22*
