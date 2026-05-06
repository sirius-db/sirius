---
phase: 19-io-framework-adoption-pr-675
plan: 02
subsystem: io
tags: [hygiene, rmm, raii, io-uring, cuda-device, hyg-02, io-16]
one_liner: "Wrap io_uring reactor's raw cudaSetDevice at uring_reactor.cpp:276 with std::optional<rmm::cuda_set_device_raii>, preserving the device_id>=0 guard. Closes IO-16 (zero raw cudaSetDevice in src/io/); HYG-02 baseline 40 preserved."

# Dependency graph
requires:
  - phase: 19
    plan: 01
    provides: baseline grep counts (raw cudaSetDevice in src/io/ = 1; HYG-02 = 40); identified line 276 as the single fix target
provides:
  - IO-16 (src/io/ raw cudaSetDevice = 0)
  - Parallel-wave artifact for 19-03 to assert against (no shared file ownership; tests not touched)
affects: [19-04, 19-05, 19-06, 21-v1.4-ship-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "std::optional<rmm::cuda_set_device_raii> with .emplace() under guard — declarative way to preserve a runtime guard while gaining RAII restoration semantics. Same idiom as cucascade/src/data/bandwidth_profiler.cpp:321."

key-files:
  created:
    - .planning/phases/19-io-framework-adoption-pr-675/19-02-SUMMARY.md
  modified:
    - src/io/uring/uring_reactor.cpp

key-decisions:
  - "Used std::optional<rmm::cuda_set_device_raii> with .emplace() under the if-guard rather than unconditional RAII. RESEARCH.md anti-pattern explicitly forbids replacing the `>= 0` guard with unconditional RAII (would change single-GPU fast-path semantics — device_id < 0 is the sentinel for 'do not switch')."
  - "Scoped dev_guard inside the if (actual > 0 && !req.ctx->failed) branch (the H2D-copy block), NOT at function scope nor at the entire worker_loop scope. RESEARCH.md anti-pattern: do not widen scope; the guard only needs to span the cudaMemcpyAsync + cudaLaunchHostFunc that follow. Tightest correct scope."
  - "Added #include <rmm/cuda_device.hpp> and <optional> to .cpp directly (not the header). Header surface stays minimal — uring_reactor.hpp consumers don't need rmm transitively."

patterns-established:
  - "RAII-via-optional: when a runtime predicate determines whether to enter a guarded scope, use `std::optional<rmm::cuda_set_device_raii>` + `.emplace(rmm::cuda_device_id{...})` under the predicate. Default-constructed optional is the disengaged-guard fast-path."

requirements-completed: [IO-16]

# Metrics
duration: ~10min
completed: 2026-05-05
---

# Phase 19 Plan 02: io_uring Reactor RAII Migration Summary

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-06T00:12:27Z
- **Completed:** 2026-05-06T00:23Z
- **Tasks:** 1 (type=auto, single-edit)
- **Files modified:** 1 (`src/io/uring/uring_reactor.cpp`)

## Accomplishments

- Replaced raw `cudaSetDevice(req.device_id)` at `src/io/uring/uring_reactor.cpp:276` with a scoped `std::optional<rmm::cuda_set_device_raii>` guard. The guard is `.emplace()`-d only when `req.device_id >= 0` (preserving the single-GPU fast-path's negative-sentinel semantics) and goes out of scope at the end of the H2D-copy if-block, restoring the prior current device.
- Closed **IO-16**: `grep -rn "cudaSetDevice\b" src/io/` returns zero hits. This was the sole raw-device-set in `src/io/` per Plan 19-01 inventory.
- Preserved HYG-02 baseline of 40 `rmm::cuda_stream_default` hits in `src/` — no new introductions.

## Diff Hunk

```diff
@@ -20,8 +20,11 @@
 #include <spdlog/spdlog.h>
 #include <sys/stat.h>
 
+#include <rmm/cuda_device.hpp>
+
 #include <algorithm>
 #include <deque>
+#include <optional>
 #include <ranges>
 #include <stdexcept>
 
@@ -273,7 +276,13 @@ void uring_reactor::worker_loop()
           size_t actual = rd > req.data_off ? std::min(req.data_size, rd - req.data_off) : 0;
           if (actual > 0 && !req.ctx->failed.load(std::memory_order_relaxed)) {
             _bounce[si].cuda_done.store(false, std::memory_order_relaxed);
-            if (req.device_id >= 0) cudaSetDevice(req.device_id);
+            // Scoped RAII guard restores the prior current device on scope exit
+            // (HYG-02 / IO-16). The `>= 0` guard is preserved so the
+            // single-GPU fast-path (device_id < 0 sentinel) is unchanged.
+            std::optional<rmm::cuda_set_device_raii> dev_guard;
+            if (req.device_id >= 0) {
+              dev_guard.emplace(rmm::cuda_device_id{req.device_id});
+            }
             cudaMemcpyAsync(req.dst,
                             (uint8_t*)_bounce[si].buf.get() + req.data_off,
                             actual,
```

## Task Commits

1. **Task 1: Wrap raw cudaSetDevice with scoped rmm::cuda_set_device_raii at uring_reactor.cpp:~276** — `38f727b` (`refactor(19-02): wrap io_uring reactor cudaSetDevice with rmm RAII`).

Plan metadata commit (this SUMMARY + STATE/ROADMAP advance) follows separately and is captured by the orchestrator post-wave verification.

## Verification Gates

| Gate | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- |
| IO-16 grep gate (`src/io/`) | `grep -rn "cudaSetDevice\b" src/io/ \| grep -v rmm::cuda_set_device_raii` | 0 lines | 0 | PASS |
| HYG-02 (`rmm::cuda_stream_default` total) | `grep -rc "rmm::cuda_stream_default" src/ \| awk -F: '{s+=$2} END {print s}'` | ≤ 40 | 40 | PASS (unchanged) |
| Guard preserved | `grep -n "if (req.device_id >= 0)" src/io/uring/uring_reactor.cpp` | ≥ 1 hit | 1 (line 283) | PASS |
| RAII present | `grep -n "rmm::cuda_set_device_raii" src/io/uring/uring_reactor.cpp` | ≥ 1 hit | 1 (line 282) | PASS |
| Scope tight (local to if-block) | manual review of diff | dev_guard declared inside `if (actual > 0 && !req.ctx->failed)` block, not at function scope | PASS | PASS |
| MCP build | `mcp__project-commands__run_command build` | exit 0 | exit 0 (5.1s) | PASS |

Build linked both `sirius_extension` and `sirius_loadable_extension` cleanly; `sirius_unittest` re-linked successfully (the parallel 19-03 wave is not in this commit's scope).

## Decisions Made

- **`std::optional<rmm::cuda_set_device_raii>` over unconditional RAII** — the `req.device_id >= 0` guard is the runtime sentinel that says "this request didn't pin to a device, do not switch context." Replacing the guard with unconditional `rmm::cuda_set_device_raii{rmm::cuda_device_id{req.device_id}}` would invoke `cudaSetDevice` with a negative device id (likely UB or `cudaErrorInvalidDevice`). RESEARCH.md anti-pattern explicitly forbids this regression.
- **Tight scope (H2D-copy if-branch only)** — RESEARCH.md anti-pattern explicitly forbids widening the RAII scope to the entire `worker_loop()` body. The reactor processes both host-completion and device-completion paths in the same loop, and the device-completion path is only hit on `data & ~HOST_TAG`. Pinning the entire loop to a per-request device would break the host-path semantics. Scope kept exactly as wide as the cudaMemcpyAsync + cudaLaunchHostFunc that need it.
- **Includes added to .cpp not .hpp** — `uring_reactor.hpp` is consumed by callers (sirius_context.cpp via uring_ioctx.hpp transitively). Adding `<rmm/cuda_device.hpp>` to the header would forward an unnecessary template-heavy include onto every consumer. Keeping it in the .cpp is the minimal-surface choice and matches the existing pattern (uring_reactor.hpp already pulls in `<cuda_runtime.h>` only because the reactor types reference `cudaStream_t`).

## Deviations from Plan

None — plan executed exactly as written. Single task, single edit, single commit. RESEARCH.md Pattern 3 ("HYG-02 Mitigation in Reactor Thread") and the Anti-Patterns to Avoid section were both followed verbatim.

## Issues Encountered

- **Initial Edit attempt mis-targeted the wrong include line** — the plan's `<interfaces>` block hinted that `<cuda_runtime.h>` was directly in the .cpp, but it's actually in `src/include/io/uring/uring_reactor.hpp:21` and pulled in via the `#include "io/uring/uring_reactor.hpp"` at line 17. First Edit anchored on `#include <cuda_runtime.h>` failed; second Edit was retargeted to the actual include block boundary in the .cpp. No functional impact — the second Edit succeeded and produced the correct AST.
- **Initial Edit attempt mis-named the variable** — used `device_id_` (member) instead of `req.device_id` (per-request field). Plan's `<interfaces>` block had the correct identifier (`req.device_id`); my first Edit mis-paraphrased it. Re-read the canonical RESEARCH.md "Pattern 3" snippet, used `req.device_id`. Single retry with no commit; clean state preserved.

Both issues caught at edit-tool level (Edit returns "string not found" rather than corrupting the file) — no recovery needed.

## User Setup Required

None.

## Next Phase Readiness

**Plan 19-03 unblocked** (already running in parallel — Wave 1). 19-03 owns `test/cpp/` test-fixture rewrites. No file overlap with this plan.

**Downstream gates locked in for 19-04/19-05/19-06 + Phase 21 ship-gate:**

| Phase 19 sub-gate | Status after 19-02 |
| --- | --- |
| IO-16 (src/io/ raw cudaSetDevice = 0) | **CLOSED** — this plan's deliverable |
| HYG-02 baseline (rmm::cuda_stream_default ≤ 40) | PRESERVED (40, unchanged) |
| IO-13/14/15 (ioctx migration + datasource flip) | Not yet started — Plan 19-03..19-05 |
| IO-12 (vcpkg.json + liburing wiring) | PASS (closed in 19-01) |
| IO-17 (SF1 smoke regression) | Deferred to 19-06 |

## Self-Check: PASSED

- File `src/io/uring/uring_reactor.cpp` modified: FOUND (verified via `git diff` showing 10 insertions, 1 deletion)
- File `.planning/phases/19-io-framework-adoption-pr-675/19-02-SUMMARY.md` created: FOUND
- Commit `38f727b` exists: FOUND in `git log --oneline | grep 38f727b`
- IO-16 grep gate: 0 raw cudaSetDevice in src/io/ (PASS)
- HYG-02 grep gate: 40 rmm::cuda_stream_default in src/ (PASS, unchanged)
- MCP build: exit 0 (5.1s)
- Anti-patterns avoided: guard preserved (line 283), RAII scope is the H2D-copy if-block only (lines 282-285), branch unchanged (still inside `actual > 0 && !req.ctx->failed`)

All claims in this SUMMARY (file paths, commit hashes, grep counts, build exit code) are verified against working-tree state.

---
*Phase: 19-io-framework-adoption-pr-675*
*Plan: 02*
*Wave: 1 (parallel with 19-03)*
*Completed: 2026-05-05*
