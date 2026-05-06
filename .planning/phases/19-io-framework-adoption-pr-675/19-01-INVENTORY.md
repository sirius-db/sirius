# Phase 19-01: Pre-flight Inventory

**Date:** 2026-05-05
**Plan:** 19-01 (Wave 0 — pre-flight inventory + IO-12 audit)
**Purpose:** Capture exact baseline state of Phase 19 retire targets so downstream plans (19-02..19-06) can assert deltas. Also resolve research's two open questions (Q3 iceberg helpers, Q4 vcpkg.json).

This inventory is the authoritative grep baseline for:
- IO-12 verdict (vcpkg.json + liburing configure-time discovery)
- IO-13 ioctx-migration target counts (idisk_io_backend / io_backend_registry retirement)
- IO-15 cucascade_datasource zero-hit gate
- IO-16 HYG-02 baseline (rmm::cuda_stream_default ≤ 40 in src/)

---

## Baseline grep counts

| Target | Command | Count | Expected (RESEARCH.md) | Status |
|--------|---------|-------|------------------------|--------|
| `cucascade_datasource` (line hits in src/+test/) | `grep -rn "cucascade_datasource" src/ test/ \| wc -l` | **51** | "8 hits" was per high-level site; line-count higher because ctor/dtor body, includes, helper signatures, test cases are counted | PASS — number reconciled (see Per-File Site List below) |
| `cucascade_datasource` (distinct files) | `grep -rln "cucascade_datasource" src/ test/ \| sort -u \| wc -l` | **6** | 5–6 files (header, impl, 2 scan consumers, 1 hpp, 1 test) | PASS |
| `cucascade::idisk_io_backend` (ref hits) | `grep -rn "cucascade::idisk_io_backend" src/ test/ \| wc -l` | **25** | 25 ref hits | PASS |
| `cucascade::io_backend_registry` + `register_builtin_io_backends` | `grep -rn "cucascade::io_backend_registry\\\|register_builtin_io_backends" src/ test/ \| wc -l` | **6** | ≥3 (sirius_context.cpp:277 + 2 test fixtures) | PASS — exceeds expectation (3 src/ + 4 test/ — actually 1 src/ + 1 src/include/ + 4 test/ ref pairs) |
| `rmm::cuda_stream_default` in src/ (HYG-02) | `grep -rc "rmm::cuda_stream_default" src/ \| awk -F: '{s+=\$2} END {print s}'` | **40** | 40 (HYG-02 ship-gate baseline from Phase 18) | PASS |
| Raw `cudaSetDevice` in src/io/ | `grep -rn "cudaSetDevice\\b" src/io/` | **1** | 1 hit at `uring_reactor.cpp:276` (IO-16 fix target) | PASS |
| `read_positional_delete_file` / `read_equality_delete_file` (definitions in src/) | `grep -rn "read_positional_delete_file\\\|read_equality_delete_file" src/ include/` | **4** (2 defs + 2 calls — all in `src/op/scan/iceberg_metadata_reader.cpp`) | unknown — Q3 audit | resolved (see Q3 below) |
| `read_positional_delete_file` / `read_equality_delete_file` (test/ refs) | `grep -rn "read_positional_delete_file\\\|read_equality_delete_file" test/` | **0** | unknown — Q3 audit | resolved — no test references |

**HYG-02 verdict:** PASS at 40 (ship-gate threshold ≤ 40). All 40 hits are in `src/legacy/` and `src/include/legacy/` (legacy `namespace duckdb` code path which is frozen per CLAUDE.md). Zero hits in active Super Sirius code paths. Phase 19 source changes must not regress this number.

### HYG-02 Per-File Distribution (informational)

```
src/include/legacy/expression_executor/gpu_dispatcher.hpp:4
src/include/legacy/expression_executor/gpu_expression_executor.hpp:2
src/include/legacy/operator/gpu_physical_substring.hpp:1
src/include/legacy/operator/gpu_physical_strings_matching.hpp:3
src/legacy/cuda/cudf/cudf_aggregate.cu:3
src/legacy/cuda/cudf/cudf_join.cu:6
src/legacy/cuda/cudf/cudf_groupby.cu:15
src/legacy/cuda/expression_executor/gpu_dispatch_materialize.cu:2
src/legacy/operator/gpu_physical_nested_loop_join.cpp:2
src/legacy/operator/gpu_physical_result_collector.cpp:1
src/legacy/operator/gpu_physical_ungrouped_aggregate.cpp:1
TOTAL: 40
```

All hits are confined to the legacy code path. New Phase 19 code in `src/io/`, `src/sirius_context.cpp`, `src/op/scan/parquet_scan_task.cpp`, `src/op/scan/iceberg_scan_task.cpp` MUST use explicit streams.

---

## Per-File Site List for `cucascade_datasource`

Distinct files containing `cucascade_datasource` references (6 files):

| File | Site Type | Action (per Phase 19 plan) |
|------|-----------|----------------------------|
| `src/include/io/cucascade_datasource.hpp` | Class declaration | DELETE (Plan 19-05 IO-15) |
| `src/io/cucascade_datasource.cpp` | Class implementation | DELETE (Plan 19-05 IO-15) |
| `src/op/scan/iceberg_scan_task.cpp` | `#include` line 31 + comment line 156 | REPLACE include → `<io/sirius_datasource.hpp>`; UPDATE comment (Plan 19-05) |
| `src/op/scan/parquet_scan_task.cpp` | `#include` line 25; `make_unique` ctor line 337 (planning path); `make_shared` ctor line 910 (hot path) | REPLACE include + 2 ctor sites → `sirius_datasource` (Plan 19-05) |
| `src/include/op/scan/parquet_scan_task.hpp` | Comment reference line 518 | UPDATE comment (Plan 19-05) |
| `test/cpp/io/test_cucascade_datasource.cpp` | Entire file (311 lines, 7 TEST_CASEs against `mock_io_backend`) | DELETE; replace with `test_sirius_datasource.cpp` mirroring 7 TEST_CASEs against a `mock_sirius_ioctx` (Plan 19-05) |

### Construction Sites (high-value for Plan 19-05 IO-15)

```
src/op/scan/parquet_scan_task.cpp:337    auto datasource      = std::make_unique<sirius::io::cucascade_datasource>(
src/op/scan/parquet_scan_task.cpp:910    _datasource           = std::make_shared<sirius::io::cucascade_datasource>(
```

These are the 2 production datasource construction sites. Test-only sites (test_cucascade_datasource.cpp) are deleted as part of file replacement.

### `idisk_io_backend` Migration Sites (Plan 19-03 + 19-05)

Critical sites that change type from `cucascade::idisk_io_backend` to `sirius::io::sirius_ioctx`:

| File | Line | Site | Plan |
|------|------|------|------|
| `src/include/sirius_context.hpp` | 184, 195, 295 | accessor signatures + member declarations | 19-03 (IO-13) |
| `src/include/sirius_context.hpp` | 294 | `cucascade::io_backend_registry io_backend_registry_;` (DELETE field) | 19-03 |
| `src/sirius_context.cpp` | 277 | `register_builtin_io_backends(io_backend_registry_)` (DELETE call) | 19-03 |
| `src/sirius_context.cpp` | 524 | `get_io_backend_for(int device_id)` (RENAME → `get_ioctx_for`) | 19-03 |
| `src/include/op/scan/parquet_scan_task.hpp` | 132, 260, 267, 460, 520 | ctor params + accessor + field | 19-05 |
| `src/op/scan/parquet_scan_task.cpp` | 217, 279, 334 | ctor bodies + comment | 19-05 |
| `src/op/scan/iceberg_scan_task.cpp` | 113, 132 | ctor bodies | 19-05 |
| `src/include/op/scan/iceberg_scan_task.hpp` | 74, 98 | ctor params | 19-05 |
| `test/cpp/scan/test_parquet_scan_task.cpp` | 108–117 | `make_test_gpu_io_backends` (RENAME → `make_test_gpu_ioctxs`) | 19-02 (Wave 1 fixture) |
| `test/cpp/scan/test_metadata_gpu_scan_operators.cpp` | 70–77 | `make_test_io_backend` (RENAME → `make_test_ioctx`) | 19-02 |
| `test/cpp/io/test_cucascade_datasource.cpp` | 48 | `mock_io_backend` (DELETED with whole file) | 19-05 |

### `io_backend_registry` Sites (6 hits)

```
src/sirius_context.cpp:277:  cucascade::register_builtin_io_backends(io_backend_registry_);
src/include/sirius_context.hpp:294:  cucascade::io_backend_registry io_backend_registry_;
test/cpp/scan/test_parquet_scan_task.cpp:111:  static cucascade::io_backend_registry registry;
test/cpp/scan/test_parquet_scan_task.cpp:113:  std::call_once(registry_init_flag, [&] { cucascade::register_builtin_io_backends(registry); });
test/cpp/scan/test_metadata_gpu_scan_operators.cpp:72:  static cucascade::io_backend_registry registry;
test/cpp/scan/test_metadata_gpu_scan_operators.cpp:74:  std::call_once(registry_init_flag, [&] { cucascade::register_builtin_io_backends(registry); });
```

All 6 sites become orphaned when `idisk_io_backend` is retired — DELETE field + DELETE call (sirius_context) and DELETE 4 lines × 2 test files.

---

## Q3 resolution: iceberg delete-file helpers

**Question:** Do `read_positional_delete_file` / `read_equality_delete_file` construct `cucascade_datasource` internally? If yes, they are hidden migration sites for Plan 19-05 (IO-15).

**Answer: NO — neither helper constructs `cucascade_datasource`.**

### Helper Definitions

| Helper | File | Line | Datasource Path |
|--------|------|------|----------------|
| `read_positional_delete_file` | `src/op/scan/iceberg_metadata_reader.cpp` | 164 | **DuckDB CPU `read_parquet`** — `conn.Query("SELECT file_path, pos FROM read_parquet(...)")`. Does NOT construct any cucascade or sirius datasource. |
| `read_equality_delete_file` | `src/op/scan/iceberg_metadata_reader.cpp` | 207 | **Direct cuDF `cudf::io::datasource::create(delete_file_path)`** at line 227. Does NOT construct `cucascade_datasource`. |

### Helper Call Sites (in-tree only)

```
src/op/scan/iceberg_metadata_reader.cpp:269:      read_positional_delete_file(db, del_path, out_map);
src/op/scan/iceberg_metadata_reader.cpp:365:    auto read_result = read_equality_delete_file(eq_entry.file_path);
```

Both helpers are called from `materialize_*_deletes` functions in the same translation unit. They do not flow through `parquet_scan_task` or `iceberg_scan_task` ioctx plumbing.

### Test References

```
$ grep -rn "read_positional_delete_file\|read_equality_delete_file" test/
(no output — 0 hits)
```

### Classification

- `read_positional_delete_file` — **CLEAN (no migration needed)**: uses `conn.Query("SELECT ... FROM read_parquet(...)")`. The only Phase 19 implication is that this path bypasses `sirius_datasource` entirely (DuckDB CPU read), which is acceptable per the helper's docstring: *"delete files are tiny metadata — no reason to allocate GPU memory."*
- `read_equality_delete_file` — **CLEAN (out-of-scope minor opportunity)**: uses `cudf::io::datasource::create(delete_file_path)` directly — bypassing the entire io_backend / ioctx layer. This is a cudf-default path. Phase 19 plan SHOULD NOT migrate this — equality-delete files are tiny metadata, single-threaded reader, and the project memory carries an existing decision pattern: *"datasource::create returns 0 hits in src/" was the v1.1 grep gate, except this single iceberg helper site that survived Phase 5* — confirmed by Phase 5 SUMMARY (RESEARCH §Sources). If a future phase wants per-GPU pinning for equality-delete reads, that's a v1.5+ optimization.

### Verdict for Plan 19-05

**No iceberg delete-file helper migration in Plan 19-05.** Iceberg surface for Plan 19-05 is exactly:
- `src/op/scan/iceberg_scan_task.cpp` line 31 (`#include` flip)
- `src/op/scan/iceberg_scan_task.cpp` lines 113, 132, 156 (ctor body / comment updates)
- `src/include/op/scan/iceberg_scan_task.hpp` lines 74, 98 (ctor param type flip)

No additional iceberg helper bodies need touching. Phase 19-05 scope is unchanged from RESEARCH.md inventory.

### Phase 19-05 Augmentation List

**None.** Q3 grep audit surfaced zero additional `cucascade_datasource` construction sites beyond the inventory in RESEARCH.md.

---

## Q4 Resolution: vcpkg.json liburing + IO-12 Audit

**Question:** Does `vcpkg.json` need a `liburing` dependency entry? Is the vcpkg path actually exercised in the v1.4 build pipeline?

### vcpkg.json Status

`vcpkg.json` (root) — verbatim `dependencies` block:

```json
"dependencies": [
  "cudf",
  "yaml-cpp",
  "abseil",
  "numactl",
  "liburing"
]
```

**`liburing` is already declared at line 17.** No edit required.

### vcpkg Path Exercise Probe

```
$ grep -rn "vcpkg" CMakeLists.txt extension-ci-tools/ .github/ 2>/dev/null | head -20
CMakeLists.txt:31:# In the vcpkg build, vcpkg headers must take priority over: 1) conda's CCCL
CMakeLists.txt:35:# below) to place vcpkg includes first.
CMakeLists.txt:43:  # The loadable extension must not depend on libcudart.so in the vcpkg build.
CMakeLists.txt:75:# as a bare library name (-lnuma). In the vcpkg build the static libnuma.a lives
CMakeLists.txt:76:# in the vcpkg installed lib dir, which isn't on the default linker search path.
CMakeLists.txt:294:  # In the vcpkg build, vcpkg include dir must be searched before DuckDB's
CMakeLists.txt:297:  # vcpkg includes precede DuckDB's in the search order.
extension-ci-tools/makefiles/duckdb_extension.Makefile:84:# Add the extension config step which ensures the vcpkg dependencies of all extensions get merged properly
extension-ci-tools/makefiles/duckdb_extension.Makefile:241:cp duckdb/build/extension_configuration/vcpkg.json build/extension_configuration/vcpkg.json
extension-ci-tools/makefiles/duckdb_extension.Makefile:248:cp duckdb/build/extension_configuration/vcpkg.json build/extension_configuration/vcpkg.json
extension-ci-tools/makefiles/duckdb_extension.Makefile:252:extension_configuration: build/extension_configuration/vcpkg.json
extension-ci-tools/makefiles/duckdb_extension.Makefile:254:build/extension_configuration/vcpkg.json: ${EXTENSION_CONFIG_TARGET}
extension-ci-tools/makefiles/vcpkg.Makefile:2:vcpkg/scripts/buildsystems/vcpkg.cmake:
extension-ci-tools/makefiles/vcpkg.Makefile:3:git -C vcpkg fetch || git clone --depth 1 --branch 2025.12.12 https://github.com/microsoft/vcpkg
extension-ci-tools/makefiles/vcpkg.Makefile:6:setup-vcpkg: vcpkg/scripts/buildsystems/vcpkg.cmake
extension-ci-tools/makefiles/vcpkg.Makefile:7:setup-vcpkg: vcpkg/scripts/buildsystems/vcpkg.cmake
```

Vcpkg is part of the DuckDB extension build pipeline (see `extension-ci-tools/makefiles/vcpkg.Makefile`). CMakeLists.txt has multiple `if(VCPKG_BUILD)` branches (lines 79–84, 294–297). The vcpkg path IS exercised in CI builds; pixi-only builds (the local development path) bypass vcpkg.

**vcpkg.json liburing status:** `declared` (already present in `dependencies`).

### liburing Configure-Time Discovery Probe (pixi env)

Probed via direct path to the pixi binary (sandbox PATH does not include pixi):

```
$ PATH=~/.pixi/bin:$PATH pixi run pkg-config --modversion liburing
2.14

$ PATH=~/.pixi/bin:$PATH pixi run pkg-config --cflags liburing
-I/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.pixi/envs/default/include

$ PATH=~/.pixi/bin:$PATH pixi run pkg-config --libs liburing
-L/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.pixi/envs/default/lib -luring
```

**liburing 2.14 is discoverable in the pixi env.** Configure-time `pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)` will succeed without the Phase 17 stub.

### CMakeLists.txt Verification

Lines 70–72 (verbatim):
```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(NUMA REQUIRED IMPORTED_TARGET numa)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
```

Lines 322–325 (verbatim):
```cmake
# Additional libraries only needed by the static extension
target_link_libraries(sirius_extension PkgConfig::NUMA PkgConfig::LIBURING
                      yaml-cpp::yaml-cpp absl::any_invocable)

target_link_libraries(sirius_loadable_extension PkgConfig::LIBURING)
```

Both the static extension (`sirius_extension`) and the loadable extension (`sirius_loadable_extension`) link against `PkgConfig::LIBURING`. Wiring is complete and idiomatic.

### IO-12 Verdict

**IO-12 verdict: PASS — no source changes required.**

- `liburing-dev` headers available via pixi env (2.14)
- `pkg-config --modversion liburing` returns 2.14 — configure-time discovery works
- CMakeLists.txt:71-72 wires `pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)` (idiomatic, REQUIRED)
- CMakeLists.txt:322-325 links `PkgConfig::LIBURING` to both extension targets
- `vcpkg.json` already declares `liburing` in `dependencies` (line 17) — vcpkg-leg requirement satisfied
- Phase 17 stub `.pc` file no longer needed; pixi env supplies real headers

No `vcpkg.json` modification was performed in this task. IO-12 ships as N/A-already-satisfied.

---

## Summary

| Section | Outcome |
|---------|---------|
| Baseline grep counts captured | All 7 baselines captured; HYG-02=40, cucascade_datasource=51 lines / 6 files, idisk_io_backend=25, registry/register=6, raw cudaSetDevice=1 |
| Q3 (iceberg helpers) | RESOLVED — neither helper constructs cucascade_datasource; no Plan 19-05 augmentation needed |
| Q4 (vcpkg.json + IO-12) | RESOLVED — `liburing` already declared in vcpkg.json; pkg-config probes 2.14; CMakeLists wiring complete; IO-12 verdict PASS with zero source changes |
| Phase 19-05 scope | Unchanged from RESEARCH.md — 6 file deletions/edits in src/, 1 file deletion in test/, 2 fixture-helper renames |

This baseline is the authoritative reference for downstream Plan 19-02..19-06 verification. After Plan 19-05 (IO-15 datasource flip) commits:
- `grep -rn "cucascade_datasource" src/ test/ | wc -l` MUST return **0**.
- `grep -rn "cucascade::idisk_io_backend" src/ test/ | wc -l` MUST return **0**.
- `grep -rn "cucascade::io_backend_registry\|register_builtin_io_backends" src/ test/ | wc -l` MUST return **0**.

After Plan 19-04 (IO-16 HYG-02 wrap):
- `grep -rn "cudaSetDevice\b" src/io/` MUST still show 1 hit, but with surrounding `rmm::cuda_set_device_raii` wrapping.
- `grep -rc "rmm::cuda_stream_default" src/ | awk -F: '{s+=$2} END {print s}'` MUST remain ≤ 40.

---

*Inventory captured 2026-05-05 at start of Phase 19 execution.*
