---
phase: 19-io-framework-adoption-pr-675
plan: 05
subsystem: io-framework
tags: [io-framework, sirius-datasource, sirius-ioctx, cucascade-retirement, io-15, io-14, parquet-scan, iceberg-scan, wave-3]
one_liner: "Flip parquet/iceberg scan_task + task_creator + sirius_engine + SiriusContext to sirius_ioctx + sirius_datasource via ioctx->make_datasource(io_object) factory; retire cucascade_datasource entirely (header + impl + test deleted; IO-15 grep gate locked at 0); cache uring_io_objects on global_state per file (Open Q1)."

# Dependency graph
requires:
  - phase: 19-io-framework-adoption-pr-675
    plan: 03
    provides: test fixture helpers (make_test_gpu_ioctxs / make_test_ioctx) consumed by Task 2 call-site flips
  - phase: 19-io-framework-adoption-pr-675
    plan: 04
    provides: SiriusContext::gpu_ioctxs_ map populated under rmm::cuda_set_device_raii — the per-GPU ioctx registry that this plan's source-side migration consumes via get_gpu_ioctxs() / get_ioctx_for(int)
provides:
  - IO-14 closure (per-GPU sirius_ioctx ownership end-to-end; device_read_req.device_id matches the owning ioctx via per-task lookup)
  - IO-15 closure (cucascade_datasource fully retired; grep -rn "cucascade_datasource" src/ test/ returns 0)
  - parquet_scan_task_global_state migrated: type-flipped _gpu_ioctxs map; cached _file_io_objects per file at planning time (Open Q1)
  - iceberg_scan_task_global_state migrated: ctor + build_delete_pipeline body cleaned up; Q3 audit confirmed delete-file helpers don't construct sirius_datasource (DuckDB read_parquet + cudf::io::datasource::create paths)
  - task_creator + sirius_engine migrated: 3 call sites flipped (get_gpu_io_backends -> get_gpu_ioctxs)
  - SiriusContext: cucascade IO machinery FULLY DELETED (io_backend_registry_ + gpu_io_backends_ fields; register_builtin_io_backends call; get_io_backend_for/get_gpu_io_backends accessors; init+teardown loops)
  - Test fixtures: 4 + 3 call sites flipped to make_test_gpu_ioctxs / make_test_ioctx; old cucascade helpers deleted from both test files
affects: [19-06, 21-v1.4-ship-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-GPU ioctx + sirius_datasource via ioctx->make_datasource(io_object) factory — preserves Phase 9 two-tier preferred_device_id lookup; all device_read_req's device_id matches the owning ioctx's GPU"
    - "Cached uring_io_object on global_state per file (Open Q1 closure) — initialize_from_files() builds io_objects once at planning, get_file_io_object(file_idx) returns shared_ptr at hot-path. Avoids per-task fd reopens (uring_io_object ctor opens 2 fds: O_RDONLY + O_RDONLY|O_DIRECT)"
    - "Forward-declaration trick to avoid liburing.h BLOCK_SIZE macro pollution — parquet_scan_task.hpp forward-declares uring_io_object so consumers don't transitively pull <liburing.h>; .cpp includes uring_reactor.hpp LAST (after all blockingconcurrentqueue.h consumers) to escape macro collision"
    - "Atomic-rename grep-visibility — _gpu_io_backends -> _gpu_ioctxs at field level, get_gpu_io_backends -> get_gpu_ioctxs at accessor level; never preserved as deprecated alias (RESEARCH.md anti-pattern)"

key-files:
  created:
    - .planning/phases/19-io-framework-adoption-pr-675/19-05-SUMMARY.md
  modified:
    - src/include/op/scan/parquet_scan_task.hpp
    - src/op/scan/parquet_scan_task.cpp
    - src/include/op/scan/iceberg_scan_task.hpp
    - src/op/scan/iceberg_scan_task.cpp
    - src/creator/task_creator.cpp
    - src/sirius_engine.cpp
    - src/include/sirius_context.hpp
    - src/sirius_context.cpp
    - test/cpp/scan/test_parquet_scan_task.cpp
    - test/cpp/scan/test_metadata_gpu_scan_operators.cpp
    - CMakeLists.txt
  deleted:
    - src/include/io/cucascade_datasource.hpp
    - src/io/cucascade_datasource.cpp
    - test/cpp/io/test_cucascade_datasource.cpp

key-decisions:
  - "Forward-declare uring_io_object in parquet_scan_task.hpp (instead of including <io/uring/uring_reactor.hpp> there) — uring_reactor.hpp transitively pulls <liburing.h> which defines a BLOCK_SIZE macro that collides with blockingconcurrentqueue.h's static const member of the same name. Forward-decl + .cpp-only include of uring_reactor.hpp (LAST in the include block) sidesteps the collision. sirius_context.cpp uses the same ordering pattern."
  - "Cached _file_io_objects via std::vector<std::shared_ptr<sirius::io::uring_io_object>> on global_state per RESEARCH.md Open Q1 — populated in initialize_from_files() at planning time, reused at every per-task datasource construction. Cleanup is automatic via global_state destruction (we DID NOT enable initialize_cache() per Open Q2 — prefetching_cache::insert never extends io_object lifetime via shared_from_this; io_objects clean up naturally)."
  - "Iceberg build_delete_pipeline: kept the empty-ioctxs throw guard (renamed gpu_io_backends -> gpu_ioctxs) but removed the unused `iceberg_io_backend = gpu_io_backends.begin()->second` local variable. Q3 audit confirmed neither iceberg delete-file helper (read_positional_delete_file, read_equality_delete_file) constructs sirius_datasource — they use DuckDB read_parquet (CPU) + cudf::io::datasource::create directly. The ioctx map is still required by the base parquet_scan_task_global_state's planning-time footer reads, so the throw guard remains."
  - "_datasource member on parquet_scan_task stays as std::shared_ptr<cudf::io::datasource> (already the base type at line 763). The ioctx->make_datasource(io_object) factory returns std::unique_ptr<cudf::io::datasource>; explicit conversion to shared_ptr at the call site (`std::shared_ptr<cudf::io::datasource>(ioctx_it->second->make_datasource(...))`) preserves the existing storage type."
  - "test_metadata_gpu_scan_operators.cpp call sites flipped to make_test_ioctx() but file remains OUT of CMakeLists.txt TEST_SOURCES — sirius_parquet_metadata_scan_operator.hpp was deleted in Phase 17 merge (re-attached in Phase 20 SM-03). Edits keep the IO-15 grep gate clean and prepare the file for Phase 20+ re-add. Documented as Phase 20+ deferral (success-criterion option B)."
  - "RESEARCH.md anti-pattern compliance: did NOT preserve get_io_backend_for(int) / get_gpu_io_backends() as deprecated aliases (clean rename); did NOT keep both _gpu_io_backends and _gpu_ioctxs on global_state (atomic rename); did NOT call ioctx->initialize_cache() (Open Q2 deferred to Phase 20+)."

patterns-established:
  - "include-order workaround for header conflicts — when a third-party header defines a macro that collides with a member name in another header (here: liburing's BLOCK_SIZE vs blockingconcurrentqueue's static const BLOCK_SIZE member), forward-declare the type in your public header and include the colliding header LAST in the .cpp, after all consumers of the collision-victim header."
  - "Atomic source-side then test-side migration — Task 1 commits the source flip even if test fixtures don't yet build; Task 2 commits the test fixture flip + drops the build red. Plan-level atomicity is preserved via the per-task commit chain; final build gate at end-of-Task-2 (and post-Task-3 deletion) is the milestone-defining green."

requirements-completed: [IO-14, IO-15]

# Metrics
duration: ~33min
completed: 2026-05-06
---

# Phase 19 Plan 05: Consumer Migration to sirius_datasource + IO-15 Retirement Summary

**Wave 3 — the core migration plan and largest in Phase 19.** Flips all 4 cucascade_datasource construction sites (parquet_scan_task planning + hot path; iceberg_scan_task include + comment) to the sirius_datasource factory shape (`ioctx->make_datasource(io_object)`); type-flips the per-GPU container on global_states from cucascade backends to sirius_ioctxs; deletes the entire SiriusContext cucascade machinery (registry + map + accessors + init/teardown); deletes the three doomed files. IO-15 grep gate locked at 0; HYG-02 preserved at 40; build clean; [mgpu] 16/16 + [multi_gpu_foundation] 7/7 PASS.

## Performance

- **Duration:** ~33 min
- **Started:** 2026-05-06T00:39:41Z
- **Completed:** 2026-05-06T01:12:39Z
- **Tasks:** 3 (all type=auto)
- **Files modified:** 11 (8 src/, 2 test/, 1 CMakeLists.txt)
- **Files deleted:** 3 (cucascade_datasource.{hpp,cpp} + test_cucascade_datasource.cpp)
- **Build runs:** 4 (3 incremental + 1 final clean) — 11.2s+10.3s+16.5s+48.0s+3.0s ≈ 90s wall on build alone
- **Smoke + regression:** [multi_gpu_foundation] 7/7 (38 assertions, 4.3s); [mgpu] 16/16 (79091 assertions, 107.6s)

## Accomplishments

- **IO-15 closed (milestone-defining):** `grep -rn "cucascade_datasource" src/ test/` returns **0 hits** (down from 51 line hits / 6 files at Phase 19-01 baseline).
- **IO-14 closed (per-GPU CUDA-context binding end-to-end):** parquet_scan_task::compute_task now resolves a per-task sirius_ioctx via the Phase 9 two-tier preferred_device_id lookup (local-wins-over-global), constructs sirius_datasource via the ioctx->make_datasource factory; device_read_req.device_id always matches the owning ioctx's GPU.
- **`cucascade::idisk_io_backend` retired from src/:** Phase 19-01 baseline had 25 hits; post-19-05 has 0 hits in src/ (the 6 hits across 2 doomed files are deleted).
- **`cucascade::io_backend_registry` + `register_builtin_io_backends` retired:** 4 src/ hits gone (sirius_context.cpp:277 init call + sirius_context.hpp:294 field decl + 2 accessor decls/inline body).
- **Cached `_file_io_objects` per file (Open Q1 closure):** populated at planning time inside `initialize_from_files()`, reused on every hot-path datasource construction. Avoids per-task fd reopens at SF100+.
- **Test fixtures fully migrated:** 4 call sites in test_parquet_scan_task.cpp (lines 441/539/623/685) + 3 call sites in test_metadata_gpu_scan_operators.cpp (lines 249/354/400) flipped to make_test_gpu_ioctxs / make_test_ioctx. Old cucascade helpers deleted in both files.
- **HYG-02 baseline preserved at 40** — no `rmm::cuda_stream_default` introductions; new code uses explicit streams everywhere (and `rmm::cuda_set_device_raii` for device guards).
- **Build clean (mcp exit 0)** — both `sirius_extension` (static) and `sirius_loadable_extension` link cleanly; `sirius_unittest` re-builds and links cleanly.
- **No regression:** [multi_gpu_foundation] 7/7 PASS (38 assertions); [mgpu] 16/16 PASS (79091 assertions). All Phase 18 P1 deadlock + Phase 12+13+14+15 multi-GPU correctness gates preserved.

## Task Commits

1. **Task 1: Flip parquet/iceberg scan + task_creator to sirius_ioctx + sirius_datasource** — `7d22f9f` (refactor)
2. **Task 2: Retire SiriusContext cucascade IO machinery + flip test fixtures** — `4cd1530` (refactor)
3. **Task 3: Delete cucascade_datasource files + close IO-15 grep gate** — `36b4e76` (chore)

Plan metadata commit (this SUMMARY + STATE.md + ROADMAP.md updates) follows separately.

## Verification Gates

| Gate | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- |
| IO-15 grep gate | `grep -rn "cucascade_datasource" src/ test/` | 0 hits | 0 | **PASS (milestone)** |
| `cucascade::idisk_io_backend` in src/ | `grep -rn "cucascade::idisk_io_backend" src/` | 0 hits | 0 | PASS |
| `cucascade::io_backend_registry\|register_builtin_io_backends` in src/ | `grep -rn "cucascade::io_backend_registry\|register_builtin_io_backends" src/` | 0 hits | 0 | PASS |
| `gpu_io_backends_\|get_io_backend_for\|get_gpu_io_backends` in src/ | `grep -rn ...` | 0 hits | 0 | PASS |
| HYG-02 baseline | `grep -rc "rmm::cuda_stream_default" src/ \| awk -F: '{s+=$2} END {print s}'` | ≤ 40 | 40 | PASS (unchanged) |
| IO-16 raw cudaSetDevice in src/io/ | `grep -rn "cudaSetDevice\b" src/io/ \| grep -v "//"` | 0 | 0 | PASS (preserved from 19-02) |
| 3 files deleted via git rm | `git status` shows D for cucascade_datasource.{hpp,cpp} + test | yes | yes | PASS |
| MCP build (final) | `mcp__project-commands__run_command build` | exit 0 | exit 0 (3.0s incremental, 48.0s post-Task-2 full link) | PASS |
| Smoke: [multi_gpu_foundation] | `mcp unit-tests --filter "[multi_gpu_foundation]"` | 7/7 PASS | 7/7 PASS (38 assertions, 4.3s) | PASS |
| Regression: [mgpu] | `mcp unit-tests --filter "[mgpu]"` | 16/16 PASS | 16/16 PASS (79091 assertions, 107.6s) | PASS |
| `make_test_gpu_io_backends\|make_test_io_backend` in test/cpp/scan/ | `grep -rn "..." test/cpp/scan/` | 0 hits | 0 | PASS |
| `make_test_gpu_ioctxs\|make_test_ioctx` usage | `grep -rn "..." test/cpp/scan/` | ≥ 7 hits (4 parquet + 3 metadata) | ≥7 | PASS |

## Per-File Diff Highlights

### `src/include/op/scan/parquet_scan_task.hpp` (header type flip + new accessor)

```diff
- #include <cucascade/data/disk_io_backend.hpp>
+ #include <io/types.hpp>
+ // Forward-decl uring_io_object — avoid <liburing.h> BLOCK_SIZE macro
+ namespace sirius::io { class uring_io_object; }

- std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends = {});
+ std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs = {});

- get_gpu_io_backends() const { return _gpu_io_backends; }
+ get_gpu_ioctxs() const { return _gpu_ioctxs; }
+ // NEW accessor for cached uring_io_object per file
+ get_file_io_object(std::size_t file_idx) const { return _file_io_objects[file_idx]; }

- std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> _gpu_io_backends;
+ std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> _gpu_ioctxs;
+ std::vector<std::shared_ptr<sirius::io::uring_io_object>> _file_io_objects;
```

### `src/op/scan/parquet_scan_task.cpp` (planning + hot path flips)

**Planning path (~line 337):**
```diff
-  auto const planning_backend_it = _gpu_io_backends.begin();
-  for (auto const& file_path : _file_paths) {
-    auto const file_size = std::filesystem::file_size(file_path);
-    auto datasource      = std::make_unique<sirius::io::cucascade_datasource>(
-      planning_backend_it->second, std::filesystem::path{file_path}, file_size);
-    datasources.push_back(std::move(datasource));
+  auto const planning_ioctx_it = _gpu_ioctxs.begin();
+  for (auto const& file_path : _file_paths) {
+    auto io_object = std::make_shared<sirius::io::uring_io_object>(file_path);
+    auto const file_size = io_object->size();
+    auto datasource = planning_ioctx_it->second->make_datasource(io_object);
+    _file_io_objects.push_back(std::move(io_object));
+    datasources.push_back(std::move(datasource));
```

**Hot path (~line 910):**
```diff
-    auto const& backends = g_state.get_gpu_io_backends();
-    auto backend_it = preferred.has_value() ? backends.find(*preferred) : backends.begin();
-    if (backend_it == backends.end()) {
-      throw std::out_of_range("[parquet_scan_task::compute_task] no io_backend for device_id=" + ...
-    }
-    auto const& file_path = g_state.get_file_path(l_state.get_file_idx());
-    auto const file_size  = g_state.get_file_size(l_state.get_file_idx());
-    _datasource           = std::make_shared<sirius::io::cucascade_datasource>(
-      backend_it->second, std::filesystem::path{file_path}, file_size);
+    auto const& ioctxs = g_state.get_gpu_ioctxs();
+    auto ioctx_it = preferred.has_value() ? ioctxs.find(*preferred) : ioctxs.begin();
+    if (ioctx_it == ioctxs.end()) {
+      throw std::out_of_range("[parquet_scan_task::compute_task] no sirius_ioctx for device_id=" + ...
+    }
+    auto io_object = g_state.get_file_io_object(l_state.get_file_idx());
+    _datasource = std::shared_ptr<cudf::io::datasource>(
+      ioctx_it->second->make_datasource(std::move(io_object)));
```

**Include reorder for liburing.h's BLOCK_SIZE macro:**
```diff
 // sirius
- #include <io/cucascade_datasource.hpp>
- #include <io/uring/uring_reactor.hpp>
  #include <log/logging.hpp>
  ...
+ // Phase 19 IO-15: include uring_reactor LAST among sirius headers — liburing.h
+ // transitively pulled by uring_reactor.hpp defines a BLOCK_SIZE macro that
+ // collides with the BLOCK_SIZE static member in <blockingconcurrentqueue.h>.
+ #include <io/sirius_datasource.hpp>
+ #include <io/uring/uring_reactor.hpp>  // last
```

### `src/include/op/scan/iceberg_scan_task.hpp` + `src/op/scan/iceberg_scan_task.cpp`

```diff
- #include <cucascade/data/disk_io_backend.hpp>
+ #include <io/types.hpp>

  // ctor params (both .hpp + .cpp):
- std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends);
+ std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs);

  // build_delete_pipeline body:
- auto const& gpu_io_backends = this->get_gpu_io_backends();
- if (gpu_io_backends.empty()) { throw std::runtime_error("[iceberg] No GPU io_backends ..."); }
- auto iceberg_io_backend = gpu_io_backends.begin()->second;  // unused — DELETED
+ auto const& gpu_ioctxs = this->get_gpu_ioctxs();
+ if (gpu_ioctxs.empty()) { throw std::runtime_error("[iceberg] No GPU sirius_ioctxs ..."); }
```

### `src/creator/task_creator.cpp` + `src/sirius_engine.cpp`

```diff
- auto gpu_io_backends = sirius_ctx->get_gpu_io_backends();
+ auto gpu_ioctxs      = sirius_ctx->get_gpu_ioctxs();

  // sirius_engine.cpp:242 (last consumer):
- for (auto const& kv : ctx->get_gpu_io_backends()) {
+ for (auto const& kv : ctx->get_gpu_ioctxs()) {
```

### `src/include/sirius_context.hpp` + `src/sirius_context.cpp` (cucascade machinery DELETED)

Deletions:
- Header: `cucascade::io_backend_registry io_backend_registry_;` field
- Header: `std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends_;` field
- Header: `get_io_backend_for(int)` decl + `get_gpu_io_backends()` inline body
- Header: 2 cucascade includes (disk_io_backend.hpp, io_backend_registry.hpp)
- .cpp: `cucascade::register_builtin_io_backends(io_backend_registry_);` call
- .cpp: 14-line per-GPU init loop populating `gpu_io_backends_`
- .cpp: `gpu_io_backends_.clear();` + `io_backend_registry_.clear();` teardown
- .cpp: 11-line `get_io_backend_for(int device_id) const` body

### `test/cpp/scan/test_parquet_scan_task.cpp` (4 call sites + helper retirement)

```diff
- nullptr, physical_scan.get(), batch_size, make_test_gpu_io_backends());   // x4
+ nullptr, physical_scan.get(), batch_size, make_test_gpu_ioctxs());       // x4

- inline std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>>
- make_test_gpu_io_backends() { ... }                                       // DELETED
```

### `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` (3 call sites + helper retirement)

```diff
- make_test_io_backend());  // x3
+ make_test_ioctx());       // x3

- inline std::shared_ptr<cucascade::idisk_io_backend> make_test_io_backend() { ... }  // DELETED
```

NOTE: this file is NOT in CMakeLists.txt TEST_SOURCES (sirius_parquet_metadata_scan_operator.hpp was deleted in Phase 17 — re-attached in Phase 20 SM-03). Edits prepare the file for Phase 20+ re-add. The grep gates close cleanly because the (orphaned) helper + call sites no longer reference cucascade types.

### `CMakeLists.txt`

```diff
- # I/O adapters (Phase 5 — cucascade-backed parquet; not yet upstreamed)
- src/io/cucascade_datasource.cpp
+ # I/O adapters (Phase 19 — io_uring + sirius_datasource)
  src/io/admission_control.cpp

  # TEST_SOURCES:
- test/cpp/io/test_cucascade_datasource.cpp
```

## Decisions Made

- **Forward-declare uring_io_object in parquet_scan_task.hpp** instead of including `<io/uring/uring_reactor.hpp>` — uring_reactor.hpp transitively pulls liburing.h which defines a `BLOCK_SIZE` macro that collides with `blockingconcurrentqueue.h`'s `static const size_t BLOCK_SIZE` member. The .cpp includes uring_reactor.hpp LAST in the include block (after logging.hpp + all pipeline headers that pull blockingconcurrentqueue.h). Same ordering pattern as sirius_context.cpp:25 (logging) → 43 (uring_ioctx).
- **Cache `_file_io_objects` on global_state** per RESEARCH.md Open Q1 — uring_io_object ctor opens 2 fds; per-task reopen would exhaust fds at SF100+. Cleanup is automatic via global_state destruction (we DID NOT enable `initialize_cache()` per Open Q2 — prefetching_cache::insert never extends io_object lifetime via `shared_from_this`).
- **`_datasource` member type stays `std::shared_ptr<cudf::io::datasource>`** — already the base type at parquet_scan_task.hpp:763. The factory returns `std::unique_ptr<cudf::io::datasource>`; explicit conversion at the call site (`std::shared_ptr<cudf::io::datasource>(unique_ptr_factory_result)`) preserves the existing storage type without cascading changes.
- **Iceberg build_delete_pipeline still requires non-empty ioctxs map** — kept the throw guard (renamed to gpu_ioctxs). Q3 audit confirmed neither delete-file helper constructs sirius_datasource (DuckDB read_parquet + cudf::io::datasource::create paths bypass sirius_ioctx entirely), but the base parquet_scan_task_global_state's planning-time footer reads still need an ioctx, so the empty-map throw is still load-bearing.
- **Atomic field rename, never deprecated alias** — `_gpu_io_backends` -> `_gpu_ioctxs` at field level; `get_gpu_io_backends` -> `get_gpu_ioctxs` at accessor level. RESEARCH.md anti-pattern explicitly forbids preserving the old accessor as a deprecated wrapper — would silently mask incomplete migration.
- **test_metadata_gpu_scan_operators.cpp call sites flipped but file remains OUT of TEST_SOURCES** — sirius_parquet_metadata_scan_operator.hpp was deleted in Phase 17 (re-attached in Phase 20 SM-03). The file is on disk for documentation continuity but is orphaned from the build graph. Per the success criterion's option B, this is the explicit Phase 20+ deferral: the file's edits keep the IO-15 grep gate clean and prepare the call sites for Phase 20 SM-03's re-add.
- **Did NOT author test_sirius_datasource.cpp mirroring the 7 deleted TEST_CASEs** — RESEARCH.md ("could split to Wave 4") explicitly defers this to Phase 20+ polishing. Phase 19's IO-17 gate is `[TPC-H][parquet]` 22/22 + sanitizer cleanliness, not unit-test parity. Documented in Task 3 commit message.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] liburing.h BLOCK_SIZE macro collides with blockingconcurrentqueue.h**

- **Found during:** Task 1 first MCP build (parquet_scan_task.cpp build failure)
- **Issue:** Including `<io/uring/uring_reactor.hpp>` in parquet_scan_task.hpp transitively pulled `<liburing.h>` (via uring_reactor.hpp:24), which defined a `BLOCK_SIZE` macro. blockingconcurrentqueue.h:38 has a `static const size_t BLOCK_SIZE = ConcurrentQueue::BLOCK_SIZE;` declaration that the macro corrupted into `static const size_t (numeric_constant) = ...`. Errors: "expected unqualified-id before numeric constant" + "expected ')' before numeric constant".
- **Fix:** Forward-declare `uring_io_object` in parquet_scan_task.hpp (no liburing.h pull); include `<io/uring/uring_reactor.hpp>` LAST in parquet_scan_task.cpp's include block (after logging.hpp and all pipeline headers that pull blockingconcurrentqueue.h transitively). Mirrors the working pattern in sirius_context.cpp.
- **Files modified:** src/include/op/scan/parquet_scan_task.hpp, src/op/scan/parquet_scan_task.cpp
- **Commit:** 7d22f9f
- **Pattern:** documented in `patterns-established` for future header migrations.

**2. [Rule 3 - Blocking] sirius_engine.cpp:242 still called retired get_gpu_io_backends()**

- **Found during:** Task 2 first MCP build (sirius_engine.cpp build failure)
- **Issue:** RESEARCH.md inventory + plan listed task_creator.cpp + parquet/iceberg scan as the consumer sites for `get_gpu_io_backends`. sirius_engine.cpp:242 was a hidden consumer not catalogued in 19-01-INVENTORY.md — it iterates the per-GPU map to gather device IDs for the parquet-scan operator's filter-tree-per-GPU construction.
- **Fix:** Single-line flip `get_gpu_io_backends()` -> `get_gpu_ioctxs()`. Behavior unchanged — same map structure, same device IDs.
- **Files modified:** src/sirius_engine.cpp (1 line)
- **Commit:** 4cd1530
- **Note:** This is an inventory-miss in 19-01 (not a deviation from this plan's scope). The plan's `<acceptance_criteria>` `grep -rn "get_io_backend_for\|get_gpu_io_backends" src/` returns 0 hits is preserved post-fix.

No other deviations. The 3-task plan structure was followed exactly; per-task verify gates passed at end of each task; only the build-clean gate was deferred from end-of-Task-1 to end-of-Task-2 (since Task 2 owns the test fixture flips that close the build red introduced by Task 1's source-side migration).

## Issues Encountered

- **First MCP build (Task 1) failed with the BLOCK_SIZE macro collision** — diagnosed via include-graph analysis (uring_reactor.hpp → liburing.h → io_uring.h defines BLOCK_SIZE; blockingconcurrentqueue.h:38 references BLOCK_SIZE as a member). Fixed via forward-decl + include-order pattern; second build green for src/, expected red on test_parquet_scan_task.cpp (Task 2 scope). Recovery time: ~5 min.
- **Second MCP build (Task 2) failed at sirius_engine.cpp:242** — RESEARCH.md inventory listed only task_creator + scan consumers; sirius_engine was a hidden caller. Single-line fix; Task 2 build green after the patch. Recovery time: ~2 min.
- **No runtime issues** — both [multi_gpu_foundation] (4.3s) and [mgpu] (107.6s) PASS clean with the new sirius_ioctx + sirius_datasource path; per-GPU device_id binding is preserved by the Phase 9 two-tier preferred lookup carrying through to the new ioctx_it lookup.

## User Setup Required

None — no external service or env-var changes; no liburing rebuild needed (liburing 2.14 already in pixi env, CMakeLists wiring intact since Phase 17 merge).

## Next Phase Readiness

**Plan 19-06 (Wave 4 — verification gauntlet) is unblocked.**

Phase 19 sub-gates after this plan:

| Sub-gate | Status |
| --- | --- |
| IO-12 (vcpkg.json + liburing wiring) | PASS (closed in 19-01) |
| IO-13 (per-GPU sirius_ioctx in SiriusContext) | PASS (closed in 19-04) |
| IO-14 (per-GPU CUDA-context binding; device_read_req.device_id matches owning ioctx) | **CLOSED** (this plan's deliverable — flipped consumers complete the per-GPU end-to-end binding) |
| IO-15 (cucascade_datasource retired; grep gate at 0) | **CLOSED** (this plan's deliverable — header + impl + test deleted) |
| IO-16 (src/io/ raw cudaSetDevice = 0) | PASS (closed in 19-02; preserved in this plan) |
| HYG-02 baseline (rmm::cuda_stream_default ≤ 40) | PRESERVED (40, unchanged) |
| IO-17 (SF1 [TPC-H][parquet] 22/22 + [multi_gpu_foundation] sanitizer clean) | Deferred to plan 19-06 verification gauntlet |

**Phase 19 progress: 5/6 plans complete.** Plan 19-06 is the IO-17 verification leg ([TPC-H][parquet] 22/22 + compute-sanitizer memcheck on [multi_gpu_foundation] + [integration][gpu_execution][parquet][join]).

## Self-Check: PASSED

**Files verified to exist:**

```
$ test -f .planning/phases/19-io-framework-adoption-pr-675/19-05-SUMMARY.md && echo FOUND
FOUND
$ test -f src/include/op/scan/parquet_scan_task.hpp && echo FOUND
FOUND
$ test -f src/op/scan/parquet_scan_task.cpp && echo FOUND
FOUND
$ test -f src/include/op/scan/iceberg_scan_task.hpp && echo FOUND
FOUND
$ test -f src/op/scan/iceberg_scan_task.cpp && echo FOUND
FOUND
$ test -f src/creator/task_creator.cpp && echo FOUND
FOUND
$ test -f src/sirius_engine.cpp && echo FOUND
FOUND
$ test -f src/include/sirius_context.hpp && echo FOUND
FOUND
$ test -f src/sirius_context.cpp && echo FOUND
FOUND
$ test -f test/cpp/scan/test_parquet_scan_task.cpp && echo FOUND
FOUND
$ test -f test/cpp/scan/test_metadata_gpu_scan_operators.cpp && echo FOUND
FOUND
$ test -f CMakeLists.txt && echo FOUND
FOUND
```

**Files verified to be deleted:**

```
$ test -f src/include/io/cucascade_datasource.hpp && echo STILL_EXISTS || echo DELETED
DELETED
$ test -f src/io/cucascade_datasource.cpp && echo STILL_EXISTS || echo DELETED
DELETED
$ test -f test/cpp/io/test_cucascade_datasource.cpp && echo STILL_EXISTS || echo DELETED
DELETED
```

**Commits verified:**

```
$ git log --oneline | grep -q "7d22f9f" && echo FOUND: 7d22f9f
FOUND: 7d22f9f
$ git log --oneline | grep -q "4cd1530" && echo FOUND: 4cd1530
FOUND: 4cd1530
$ git log --oneline | grep -q "36b4e76" && echo FOUND: 36b4e76
FOUND: 36b4e76
```

**Grep gates (final state):**

```
$ grep -rn "cucascade_datasource" src/ test/ | wc -l
0
$ grep -rn "cucascade::idisk_io_backend" src/ | wc -l
0
$ grep -rn "cucascade::io_backend_registry\|register_builtin_io_backends" src/ | wc -l
0
$ grep -rn "get_io_backend_for\|gpu_io_backends_\|get_gpu_io_backends" src/ | wc -l
0
$ grep -rc "rmm::cuda_stream_default" src/ | awk -F: '{s+=$2} END {print s}'
40
$ grep -rn "cudaSetDevice\b" src/io/ | grep -v "//" | wc -l
0
$ grep -rn "make_test_gpu_io_backends\|make_test_io_backend\b" test/cpp/scan/ | wc -l
0
```

**MCP verification gates:**

```
$ mcp build (final, post-Task-3) -> exit 0 (3.0s incremental)
$ mcp unit-tests --filter "[multi_gpu_foundation]" -> 7/7 PASS, 38 assertions, 4.3s
$ mcp unit-tests --filter "[mgpu]" -> 16/16 PASS, 79091 assertions, 107.6s
```

All claims in this SUMMARY (file paths, commit hashes, grep counts, build exit codes, test results) are verified against working-tree state.

---
*Phase: 19-io-framework-adoption-pr-675*
*Plan: 05*
*Wave: 3 (sequential)*
*Completed: 2026-05-06*
