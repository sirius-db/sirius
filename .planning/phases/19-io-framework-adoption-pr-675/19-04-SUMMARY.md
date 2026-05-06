---
phase: 19-io-framework-adoption-pr-675
plan: 04
subsystem: io-framework
tags: [io-framework, sirius-ioctx, uring-ioctx, sirius-context, per-gpu, raii, admission-control, wave-2, io-13, io-14]
one_liner: "SiriusContext::initialize() now constructs ONE sirius::io::uring_ioctx per GPU under rmm::cuda_set_device_raii alongside the legacy cucascade gpu_io_backends_ map. Each ioctx owns its own admission_control budget (P5). Teardown precedes memory_manager_->shutdown() (Pitfall 3). Old map preserved for 19-05 to retire."

# Dependency graph
requires:
  - phase: 19-io-framework-adoption-pr-675
    plan: 02
    provides: IO-16 closure (no raw cudaSetDevice in src/io/) — uring_reactor's H2D copy site is RAII-wrapped, so the new per-GPU ioctx reactors don't reintroduce HYG-02 violations
  - phase: 19-io-framework-adoption-pr-675
    plan: 03
    provides: test fixture helpers make_test_gpu_ioctxs / make_test_ioctx (call-site flip happens in 19-05; not consumed by 19-04 directly but unblocks the next wave)
  - phase: 17-sirius-origin-dev-merge-base-layer
    provides: in-tree IO Framework files (sirius_ioctx ABC, uring_ioctx, uring_reactor); CMakeLists.txt liburing wiring
provides:
  - SiriusContext::gpu_ioctxs_ field populated at initialize() time with one sirius::io::uring_ioctx per GPU memory space, each constructed under rmm::cuda_set_device_raii
  - Per-GPU admission_control budgets (P5 mitigation; via uring_ioctx default ctor)
  - get_ioctx_for(int) + get_gpu_ioctxs() accessors for plan 19-05 consumers to switch onto
  - Teardown ordering: gpu_ioctxs_.clear() runs BEFORE memory_manager_->shutdown() (Pitfall 3 mitigated)
affects: [19-05, 19-06, 21-v1.4-ship-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Coexistence migration pattern — populate the NEW per-GPU container alongside the OLD one in the same init scope (with separate gpu_spaces walk to keep the loops independently grep-locatable). Old machinery stays live until plan 19-05 retires consumers."
    - "Per-GPU resource init under rmm::cuda_set_device_raii — mirrors src/sirius_context.cpp:283 cucascade backend pattern; mandatory because uring_reactor ctor allocates 32 × 1 MiB pinned bounce slots via cudaHostAlloc(cudaHostAllocPortable) bound to current CUDA context."
    - "Teardown ordering for IO objects with worker threads — clear the per-GPU ioctx map BEFORE memory_manager_->shutdown() so ~uring_ioctx (which joins worker threads and runs cudaFreeHost) sees a live CUDA context."

key-files:
  created:
    - .planning/phases/19-io-framework-adoption-pr-675/19-04-SUMMARY.md
  modified:
    - src/include/sirius_context.hpp
    - src/sirius_context.cpp

key-decisions:
  - "Coexistence (alongside, not replace) — per CONTEXT.md scope, plan 19-04's blast radius is one file pair. The old gpu_io_backends_ + io_backend_registry_ + register_builtin_io_backends + get_io_backend_for / get_gpu_io_backends surface stays LIVE. Only plan 19-05 retires the cucascade map after consumer wiring is migrated. This keeps Wave 2 narrow and preserves a clean delta for 19-05 to assert against."
  - "uring_ioctx ctor defaults locked to host_ring_depth=16, ring_entries=64, n_reactors=4, bounce_slot_size=sirius::io::CHUNK_SIZE (1 MiB) — matches src/include/io/uring/uring_ioctx.hpp:85-88 verbatim. Explicit static_cast<size_t>(4) for n_reactors and unsigned literals (16u, 64u) for the ring sizes to avoid any narrowing ambiguity at the ctor call site."
  - "initialize_cache() NOT called per RESEARCH.md Open Q2 — sirius_datasource device_read falls through to device_read_io when _cache==nullptr (sirius_datasource.cpp:122-128). v1.1 baseline correctness already feasible without the cache; cache enablement (with the per-GPU buffer_pool ownership question that comes with it) deferred to Phase 20+."
  - "get_gpu_ioctxs accessor implemented INLINE in the header (mirrors the existing inline get_gpu_io_backends body); only get_ioctx_for is out-of-line in the .cpp. Plan's grep gate `grep -n \"get_ioctx_for\\|get_gpu_ioctxs\" src/sirius_context.cpp ≥ 2 hits` met by the 2-line out-of-line impl of get_ioctx_for (signature line + throw_msg line) — matches the existing pattern parity with get_io_backend_for."
  - "Two separate gpu_spaces walks in initialize() — could have hoisted into a single shared loop, but keeping them as 2 independent for-loops makes each migration milestone independently grep-locatable. Plan 19-05 will delete the cucascade-backend loop in one atomic edit; this plan adds the new loop in one atomic edit. Lower merge-conflict surface."

requirements-completed: [IO-13, IO-14]

# Metrics
duration: ~10min
completed: 2026-05-06
---

# Phase 19 Plan 04: Per-GPU sirius_ioctx Construction in SiriusContext Summary

**Wave 2 IO-13/IO-14 architectural piece — SiriusContext::initialize() now constructs ONE sirius::io::uring_ioctx per GPU memory space under rmm::cuda_set_device_raii. Each ioctx owns its own admission_control budget (P5). Teardown clears gpu_ioctxs_ BEFORE memory_manager_->shutdown() (Pitfall 3). Coexists with cucascade gpu_io_backends_ map; consumer wiring + cucascade retirement deferred to plan 19-05.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-06T00:19:04Z
- **Completed:** 2026-05-06T00:28:37Z
- **Tasks:** 2 (both type=auto)
- **Files modified:** 2 (`src/include/sirius_context.hpp` + `src/sirius_context.cpp`)
- **Build runs:** 2 successful builds via MCP (header-only build 56.6s; cpp-incremental 10.2s)
- **Smoke tests:** [multi_gpu_foundation] 7/7 PASS (4.3s, 38 assertions); [mgpu] 16/16 PASS (105.9s, 79091 assertions)

## Accomplishments

- **IO-13 closed at the SiriusContext layer** — `gpu_ioctxs_` field populated with one `sirius::io::uring_ioctx` per GPU memory space at init time. Each construction wrapped in `rmm::cuda_set_device_raii` per CONTEXT.md P4 lock + RESEARCH.md Pattern 1.
- **IO-14 closed (per-GPU CUDA-context binding)** — Each `uring_ioctx`'s reactors and pinned bounce slots are bound to the matching GPU's CUDA context at construction time. The reactor's worker thread already does `cudaSetDevice(req.device_id)` at request-handling time (uring_reactor.cpp:276, RAII-wrapped per plan 19-02). Per-GPU ownership ensures `device_read_req.device_id` always matches the owning ioctx's device.
- **P5 mitigation (per-GPU admission_control)** — Each `uring_ioctx` default-constructs its own `admission_control` instance via `templated_ioctx<uring_reactor>`. Zero shared budgets across GPUs; no I/O serialization at SF100+.
- **New accessors landed (`get_ioctx_for` + `get_gpu_ioctxs`)** — Plan 19-05 consumers can switch their accessor calls in one mechanical edit.
- **Teardown ordering preserved (Pitfall 3)** — `gpu_ioctxs_.clear()` runs immediately after `gpu_io_backends_.clear()` and BEFORE `memory_manager_->shutdown()`. Reactor worker thread joins + `cudaFreeHost` of pinned bounce slots run against a live CUDA context.
- **HYG-02 baseline preserved at 40** — Zero new `rmm::cuda_stream_default` introductions. The new init loop uses `rmm::cuda_set_device_raii` exclusively.
- **Old machinery still works** — `register_builtin_io_backends`, `gpu_io_backends_`, `io_backend_registry_`, `get_io_backend_for`, `get_gpu_io_backends` all preserved live for plan 19-05 to retire.

## Diff Hunks

### `src/include/sirius_context.hpp`

```diff
@@ -31,6 +31,7 @@
 #include <cucascade/data/disk_io_backend.hpp>
 #include <cucascade/data/io_backend_registry.hpp>
+#include <io/types.hpp>
 #include <duckdb/common/enums/optimizer_type.hpp>
@@ -195,6 +196,29 @@
   [[nodiscard]] std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> const&
   get_gpu_io_backends() const
   {
     return gpu_io_backends_;
   }

+  /// @brief Resolve the per-GPU sirius_ioctx for the given device (Phase 19 IO-13).
+  ///
+  /// Per-GPU sirius_ioctx instances are constructed once per GPU during
+  /// initialize() under rmm::cuda_set_device_raii so the reactor's pinned
+  /// bounce slots and admission_control are bound to the matching CUDA
+  /// context. Coexists with get_io_backend_for() until 19-05 retires the
+  /// cucascade backend map.
+  [[nodiscard]] std::shared_ptr<sirius::io::sirius_ioctx> get_ioctx_for(int device_id) const;
+
+  /// @brief Read-only view of the full per-GPU sirius_ioctx cache (Phase 19 IO-13).
+  [[nodiscard]] std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const&
+  get_gpu_ioctxs() const
+  {
+    return gpu_ioctxs_;
+  }
@@ -294,6 +320,17 @@
   cucascade::io_backend_registry io_backend_registry_;
   std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends_;
+  // Phase 19 IO-13: per-GPU sirius_ioctx; replaces gpu_io_backends_ once 19-05
+  // retires consumers. One ioctx per GPU memory space, constructed under
+  // rmm::cuda_set_device_raii in initialize() so the reactor's pinned bounce
+  // slots (cudaHostAlloc(cudaHostAllocPortable)) bind to the matching CUDA
+  // context. Each ioctx owns its own admission_control budget (P5 mitigation
+  // — default uring_ioctx ctor allocates one admission_control instance per
+  // ioctx). Cleared in terminate() BEFORE memory_manager_->shutdown() so
+  // ~uring_reactor's worker-thread join + cudaFreeHost run against a live
+  // CUDA context (mirrors gpu_io_backends_ teardown ordering).
+  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs_;
```

### `src/sirius_context.cpp`

**1. Includes:**

```diff
@@ -41,6 +41,8 @@
 #include <cucascade/memory/fixed_size_host_memory_resource.hpp>
 #include <cucascade/memory/small_pinned_host_memory_resource.hpp>
+#include <io/types.hpp>
+#include <io/uring/uring_ioctx.hpp>
```

**2. Init loop — added immediately after the existing cucascade init loop:**

```diff
@@ -293,6 +293,49 @@
       gpu_io_backends_[device_id] = std::move(backend);
     }
   }

+  // === Phase 19 IO-13: per-GPU sirius_ioctx ===
+  // Hosted alongside gpu_io_backends_ during the migration. Plan 19-05 will
+  // retire the cucascade map (and remove this comment marker). Each
+  // uring_ioctx ctor allocates pinned bounce slots via cudaHostAlloc with
+  // cudaHostAllocPortable bound to the current CUDA context, so the
+  // rmm::cuda_set_device_raii guard is mandatory (P4). Each ioctx also
+  // owns its own admission_control instance — the default ctor of
+  // uring_ioctx → templated_ioctx<uring_reactor> wires this up per-instance,
+  // which satisfies P5 mitigation (per-GPU admission budget; never shared).
+  {
+    auto gpu_spaces = memory_manager_->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
+    gpu_ioctxs_.reserve(gpu_spaces.size());
+    for (auto* gpu_space : gpu_spaces) {
+      auto const device_id = gpu_space->get_device_id();
+      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
+
+      // Defaults match src/include/io/uring/uring_ioctx.hpp:85-88:
+      //   host_ring_depth=16, ring_entries=64, n_reactors=4,
+      //   bounce_slot_size=sirius::io::CHUNK_SIZE (1 MiB).
+      // (RESEARCH.md Pattern 1 + Open Q2 — initialize_cache() not called.)
+      auto ioctx = std::make_shared<sirius::io::uring_ioctx>(
+        /*host_ring_depth=*/16u,
+        /*ring_entries=*/64u,
+        /*n_reactors=*/static_cast<size_t>(4),
+        /*bounce_slot_size=*/sirius::io::CHUNK_SIZE);
+
+      int readback = -1;
+      cudaGetDevice(&readback);
+      spdlog::info("SiriusContext: sirius_ioctx created for GPU {} (cudaGetDevice readback={})",
+                   device_id, readback);
+
+      gpu_ioctxs_[device_id] = std::move(ioctx);
+    }
+  }
```

**3. Teardown — added immediately after `gpu_io_backends_.clear()`:**

```diff
@@ -478,6 +522,12 @@
   gpu_io_backends_.clear();
+  // Phase 19 IO-13: tear down per-GPU sirius_ioctx instances BEFORE
+  // memory_manager_->shutdown() (Pitfall 3). Each ~uring_ioctx joins its
+  // reactor worker thread and frees pinned bounce slots via cudaFreeHost,
+  // which requires the CUDA context to still be live. The cudaDeviceSynchronize
+  // call further down already drains any pending async copies before pinned
+  // slabs are freed by memory_manager_.
+  gpu_ioctxs_.clear();
   // MGPU-06: clear peer-access cache. ...
```

**4. Accessor implementation — added after `get_io_backend_for`:**

```diff
@@ -533,3 +585,15 @@
   return it->second;
 }

+std::shared_ptr<sirius::io::sirius_ioctx> SiriusContext::get_ioctx_for(int device_id) const
+{
+  throw_if_not_initialized();
+  auto it = gpu_ioctxs_.find(device_id);
+  if (it == gpu_ioctxs_.end()) {
+    throw std::out_of_range(
+      "SiriusContext::get_ioctx_for: no sirius_ioctx registered for device_id=" +
+      std::to_string(device_id));
+  }
+  return it->second;
+}
```

## Task Commits

1. **Task 1: Add gpu_ioctxs_ field + accessors to SiriusContext header** — `98020d3` (feat) — `feat(19-04): add gpu_ioctxs_ field + accessors to SiriusContext header`
2. **Task 2: Implement per-GPU sirius_ioctx init loop + accessors + teardown in sirius_context.cpp** — `f1dd162` (feat) — `feat(19-04): construct per-GPU sirius_ioctx in SiriusContext::initialize (IO-13)`

Plan metadata commit (this SUMMARY + STATE/ROADMAP advance) follows separately.

## Verification Gates

| Gate | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- |
| New field exists in header | `grep -n "gpu_ioctxs_" src/include/sirius_context.hpp` | ≥1 hit | line 330 + 223 | PASS |
| New accessor in header | `grep -n "get_ioctx_for" src/include/sirius_context.hpp` | ≥1 hit | line 213 | PASS |
| Old field preserved in header | `grep -n "gpu_io_backends_" src/include/sirius_context.hpp` | ≥1 hit | line 320 + 199 | PASS |
| Old accessor preserved in header | `grep -n "get_io_backend_for" src/include/sirius_context.hpp` | ≥1 hit | line 185 + 207 | PASS |
| New ioctx ctor in cpp | `grep -n "uring_ioctx" src/sirius_context.cpp` | ≥1 hit | 7 hits (incl. ctor at 326) | PASS |
| New field populated in cpp | `grep -n "gpu_ioctxs_\[" src/sirius_context.cpp` | ≥1 hit | line 341 | PASS |
| New field cleared in teardown | `grep -n "gpu_ioctxs_.clear()" src/sirius_context.cpp` | ≥1 hit | line 534 | PASS |
| Both init loops RAII-guarded | `grep -cn "rmm::cuda_set_device_raii" src/sirius_context.cpp` | ≥2 hits | 5 (1 cucascade init + 1 NEW + 3 MGPU-06) | PASS |
| New accessor impl in cpp | `grep -n "get_ioctx_for\|get_gpu_ioctxs" src/sirius_context.cpp` | ≥2 hits | 2 (signature + throw msg of get_ioctx_for; get_gpu_ioctxs is inline in header) | PASS |
| register_builtin_io_backends preserved | `grep -n "register_builtin_io_backends" src/sirius_context.cpp` | 1 hit | 1 (line 279) | PASS |
| Old field preserved in cpp | `grep -cn "gpu_io_backends_" src/sirius_context.cpp` | ≥3 hits | 6 | PASS |
| HYG-02 baseline | `grep -rc "rmm::cuda_stream_default" src/ \| awk -F: '{s+=$2} END {print s}'` | 40 | 40 | PASS |
| IO-16 raw cudaSetDevice in src/io/ | `grep -rn "cudaSetDevice\b" src/io/ \| grep -v "//"` | 0 | 0 | PASS (preserved from 19-02) |
| MCP build (Task 1) | `mcp__project-commands__run_command build` | exit 0 | exit 0 (56.6s) | PASS |
| MCP build (Task 2) | `mcp__project-commands__run_command build` | exit 0 | exit 0 (10.2s) | PASS |
| Smoke: [multi_gpu_foundation] | `mcp unit-tests --filter "[multi_gpu_foundation]"` | 7/7 PASS | 7/7 PASS (38 assertions, 4.3s) | PASS |
| Regression: [mgpu] | `mcp unit-tests --filter "[mgpu]"` | 16/16 PASS | 16/16 PASS (79091 assertions, 105.9s) | PASS |

## Decisions Made

- **Coexistence (alongside, not replace)** — CONTEXT.md scoped this plan to one file pair (sirius_context.cpp + .hpp). Old cucascade machinery preserved live; plan 19-05 owns the cucascade retirement after consumer wiring is migrated. Lower blast radius + clean delta for 19-05 to assert against.
- **Two separate `gpu_spaces` walks** — Could have hoisted both init loops into a single shared `gpu_spaces` walk to avoid double iteration. Kept as two independent loops to make each migration milestone independently grep-locatable. Plan 19-05 will delete the cucascade loop in one atomic edit; this plan added the new loop in one atomic edit. Iteration cost is negligible (small N — at most num_gpus elements).
- **`initialize_cache()` not called** — RESEARCH.md Open Q2 recommendation. v1.1 baseline correctness is already feasible without the prefetching cache; sirius_datasource's device_read falls through to device_read_io on `_cache==nullptr`. Cache enablement requires per-GPU buffer_pool ownership (see CONTEXT.md anti-pattern: never share buffer_pool across ioctxs) — defer to Phase 20+.
- **Defaults locked to uring_ioctx.hpp:85-88** — `host_ring_depth=16, ring_entries=64, n_reactors=4, bounce_slot_size=CHUNK_SIZE (1 MiB)`. Explicit `static_cast<size_t>(4)` and `unsigned` literals (`16u`, `64u`) at the call site to avoid any narrowing ambiguity at the ctor.
- **`get_gpu_ioctxs` inline in header** — Mirrors the existing `get_gpu_io_backends` inline body pattern. Only `get_ioctx_for` needs out-of-line implementation in the .cpp (the throw branch). Pattern parity preserved.

## Deviations from Plan

None — plan executed exactly as written. The plan's `<acceptance_criteria>` for Task 2 specified `grep -n "get_ioctx_for\|get_gpu_ioctxs" src/sirius_context.cpp ≥ 2 hits`. Achieved by the 2-line out-of-line `get_ioctx_for` impl (signature line + throw-message line). `get_gpu_ioctxs` is inline in the header (matches the existing `get_gpu_io_backends` inline body — pattern parity), so doesn't appear in the .cpp grep. This is the mirror of the existing accessor structure and was the explicit intent per the plan's `<interfaces>` block (which described the new accessors in the same single-line declaration shape as the existing inline accessors).

## Issues Encountered

- **Sandboxed direct unittest invocation hit "0 GPUs" stub topology** — Initial bash invocation of `build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]"` from the sandbox failed with `cucascade::topology_discovery reported 0 GPUs — refusing to initialize on stub topology`. NVML driver isn't visible from sandbox. Per CLAUDE.md memory directive `feedback_use_mcp_build.md` + `feedback_mcp_tests_scope.md`, all GPU tests must run via MCP (which has driver visibility). Re-routed via `mcp__project-commands__run_command unit-tests --filter "[multi_gpu_foundation]"` — clean PASS 7/7. Same for `[mgpu]` — 16/16 PASS.

## User Setup Required

None — no env vars or external services touched.

## Next Phase Readiness

**Plan 19-05 (Wave 3 — consumer migration + cucascade_datasource retirement) is unblocked.**

19-05 can now flip the 4 `cucascade_datasource` construction sites (`parquet_scan_task.cpp:337`, `parquet_scan_task.cpp:910`; `iceberg_scan_task.cpp:113, 132`) to `sirius_datasource` constructed via `ioctx->make_datasource(io_object)` or directly with `std::make_shared<sirius::io::sirius_datasource>(ioctx, io_object)`, sourcing the ioctx from `SiriusContext::get_ioctx_for(device_id)` (or via the gpu_ioctxs_ map flowed through task_creator).

After 19-05 completes:
- `grep -rn "cucascade_datasource" src/ test/` → 0 (header + impl + test file all deleted)
- `grep -rn "cucascade::idisk_io_backend" src/ test/` → 0 (all type-flips done)
- `grep -rn "cucascade::io_backend_registry\|register_builtin_io_backends" src/ test/` → 0 (cucascade init machinery retired)

**Phase 19 wave-2 sub-gates locked in:**

| Sub-gate | Status |
| --- | --- |
| IO-13 (per-GPU sirius_ioctx in SiriusContext) | **CLOSED** (this plan's deliverable) |
| IO-14 (per-GPU CUDA-context binding; device_id matches owning ioctx) | **CLOSED** (this plan's deliverable; reinforced by per-GPU init under cuda_set_device_raii) |
| IO-12 (vcpkg.json + liburing wiring) | PASS (closed in 19-01) |
| IO-16 (src/io/ raw cudaSetDevice = 0) | PASS (closed in 19-02; preserved in this plan) |
| HYG-02 baseline (rmm::cuda_stream_default ≤ 40) | PRESERVED (40, unchanged) |
| IO-15 (cucascade_datasource retired) | Not yet — plan 19-05 |
| IO-17 (SF1 smoke regression) | Deferred to plan 19-06 |

**Smoke results [mgpu] 16/16 + [multi_gpu_foundation] 7/7 confirm:** the new per-GPU sirius_ioctx construction does NOT regress Phase 18 DB-01..05 nor any of Phase 6/8/9/10/12/14/15's multi-GPU correctness gates. The prior runtime semantics are fully preserved because the new ioctxs are unused (no consumer reads from `gpu_ioctxs_` yet — that flip lives in 19-05).

## Self-Check: PASSED

**Files verified to exist:**

```
$ test -f .planning/phases/19-io-framework-adoption-pr-675/19-04-SUMMARY.md && echo FOUND
FOUND
$ test -f src/include/sirius_context.hpp && echo FOUND
FOUND
$ test -f src/sirius_context.cpp && echo FOUND
FOUND
```

**Commits verified:**

```
$ git log --oneline | grep -q "98020d3" && echo "FOUND: 98020d3"
FOUND: 98020d3
$ git log --oneline | grep -q "f1dd162" && echo "FOUND: f1dd162"
FOUND: f1dd162
```

**Grep gates (final state):**

```
$ grep -n "gpu_ioctxs_\|get_ioctx_for\|get_gpu_ioctxs" src/include/sirius_context.hpp | wc -l
6  (field decl + 2 accessor decls + 3 comment refs)
$ grep -n "uring_ioctx" src/sirius_context.cpp | wc -l
7  (1 include + 1 ctor + 5 comment refs)
$ grep -n "gpu_ioctxs_" src/sirius_context.cpp src/include/sirius_context.hpp | wc -l
8  (decl + comment + new init loop reserve + populate + clear + accessor inline + impl + comment)
$ grep -rc "rmm::cuda_stream_default" src/ | awk -F: '{s+=$2} END {print s}'
40
$ grep -rn "cudaSetDevice\b" src/io/ | grep -v "//" | wc -l
0
```

**MCP verification gates:**

```
$ mcp build (Task 1, header-only) → exit 0, 56.6s
$ mcp build (Task 2, cpp incremental) → exit 0, 10.2s
$ mcp unit-tests --filter "[multi_gpu_foundation]" → 7/7 PASS, 38 assertions, 4.3s
$ mcp unit-tests --filter "[mgpu]" → 16/16 PASS, 79091 assertions, 105.9s
```

All claims in this SUMMARY (file paths, commit hashes, grep counts, build exit codes, test results) are verified against working-tree state.

---
*Phase: 19-io-framework-adoption-pr-675*
*Plan: 04*
*Wave: 2 (sequential)*
*Completed: 2026-05-06*
