# Stack Research

**Domain:** GPU-native SQL engine rebase — v1.4 "Rebase After DataBatch Changes"
**Researched:** 2026-05-04
**Confidence:** HIGH (all findings from direct git inspection of the actual commits)

---

## Summary of New Dependencies Introduced by the Rebase

The v1.4 rebase absorbs 3 cucascade upstream PRs and 7 Sirius `origin/dev` PRs.
The only net-new build dependency that does not exist in the current
`feature/single-node-multi-gpu2` branch is **liburing** (`liburing-dev`), introduced by Sirius
PR #675. Everything else is either additive-but-dependency-free (cucascade PRs #112 / #116,
Sirius PRs #731 / #721 / #733–#735), or a pure source-code API migration with no new external
library (cucascade PR #117 / Sirius PR #739).

---

## 1. New External Dependencies

### 1.1 liburing — REQUIRED, introduced by Sirius PR #675

**What it is:** Linux `io_uring` userspace library. Provides `<liburing.h>` and `liburing.so`.

**Why PR #675 needs it:** The `uring_reactor` backend (`src/include/io/uring/uring_reactor.hpp`)
`#include <liburing.h>` directly. The CMakeLists.txt change in commit `4c0f1ac` adds:

```cmake
# CMakeLists.txt — lines added by PR #675 (commit 4c0f1ac)
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)   # line after PkgConfig::NUMA

target_link_libraries(sirius_extension   PkgConfig::NUMA PkgConfig::LIBURING …)
target_link_libraries(sirius_loadable_extension PkgConfig::LIBURING)
```

`vcpkg.json` gets `"liburing"` added to the `dependencies` array (commit `4c0f1ac`).

**Host status (2× RTX 6000 Ada, Ubuntu 24.04):**

- Runtime library present: `liburing2:amd64 2.5-1build1` — already installed (`dpkg -l`).
- Shared object at `/lib/x86_64-linux-gnu/liburing.so.2` — confirmed via `ldconfig -p`.
- Header package (`liburing-dev`) status: **NOT installed** as of 2026-05-04.
  - `pkg-config --modversion liburing` → "not found".
  - `find /usr/include -name liburing.h` → empty.
  - `apt-cache show liburing-dev` shows version `2.5-1build1`, available in the Ubuntu 24.04
    universe repo.

**Verdict:** The shared library is present but the dev headers are missing. The build will
fail at `pkg_check_modules(LIBURING REQUIRED …)` until `liburing-dev` is installed.

**Install action required (pixi env or system):**

```bash
# System-level install (requires sudo, one-time per host)
sudo apt-get install -y liburing-dev
```

`liburing` is a Linux-kernel-resident API (io_uring arrived in kernel 5.1); the required
version is 2.x. `liburing-dev 2.5` is correct for the kernel 5.1+ minimum. There is no conda
package for liburing-dev; it must be a system package. The vcpkg entry in `vcpkg.json` is for
the vcpkg build path, not the pixi/conda path.

**pixi.toml impact:** No change needed for the conda/pixi environment. `liburing` is a
system-level package unavailable on conda-forge; the build already handles it via
`pkg_check_modules` (the same mechanism used for `numa`). The pixi default and cuda13
environments do not need a `liburing` entry — mirrors how `libnuma-dev` is not in pixi.toml
either.

**vcpkg.json impact:** PR #675 adds `"liburing"` to the `dependencies` array. A custom vcpkg
port does not exist under `vcpkg_ports/liburing/` (no such directory in the repo). The vcpkg
port for liburing exists in the official vcpkg registry; the `builtin-baseline` in `vcpkg.json`
(`ffc071e0c0`) should be checked during Phase 16 to confirm it resolves. This only affects the
vcpkg build path (`pixi run -e vcpkg`), not the default pixi build.

---

## 2. cucascade Pin Version Target

### 2.1 Target commit: `73d00c4`

**Current pin on `feature/single-node-multi-gpu2`:** `62e0517`
(tip of our 11-commit local fix stack above `f47de0b`)

**Target tip on `origin/main`:** `73d00c4`
("implement 3-class data_back model and get rid of state machine", PR #117)

**Path from `62e0517` to `73d00c4`** (linear, 3 commits on `origin/main` not yet in our pin):

```
47e430e  Feature: Allow creation of gpu_data_represenaiton from cudf::table_view (#116)
0cd4a6a  feat(data): add memory-space bandwidth profiler (#112)
73d00c4  implement 3-class data_back model and get rid of state machine (#117)
```

Our local fix stack (11 commits above `f47de0b`) must be rebased onto `73d00c4`. The new
cucascade pin descended from `73d00c4` will be the new submodule commit in Sirius.

### 2.2 cucascade internal dependency changes across #116, #112, #117

**PR #116 (`47e430e`):** Additive only. Adds a `gpu_table_representation` constructor from
`cudf::table_view` + owner + alloc_size. Also changes `get_table()` → `get_table_view()` and
adds a stream parameter to `release_table()`. No new external deps. Impacts Sirius: every
call-site that did `.get_table()` must become `.get_table_view()`; every call to
`release_table()` must pass a stream. (PR #739 in Sirius already does this migration for the
non-mgpu code path.)

**PR #112 (`0cd4a6a`):** Additive. Introduces `bandwidth_profiler`, `chunked_resource_info`,
`fixed_size_host_memory_resource` changes, and a per-device cache in `pipeline_io_backend`
replacing the single shared `copy_stream/order_event` — fixing the cross-context failure
(GPU:N ↔ DISK for N>0). The per-device pipeline_io_backend fix is directly relevant to
multi-GPU: this unblocks any GPU≠0 disk I/O path that previously failed with
`cudaErrorInvalidResourceHandle`. No new external deps; all additions are in cucascade's
existing cudf+rmm dependency surface.

**PR #117 (`73d00c4`):** Breaking API rewrite. Replaces 4-state FSM
(`idle/task_created/processing/in_transit`) with 3-class RAII accessor model
(`read_only_data_batch` / `mutable_data_batch`). Key breaking changes for Sirius:

| Old API | New API | Notes |
|---------|---------|-------|
| `data_batch::to_read_only(ptr&&)` static | `batch->to_read_only()` non-static | caller keeps `shared_ptr<data_batch>` |
| `data_batch::to_mutable(ptr&&)` static | `batch->to_mutable()` non-static | same |
| `pop_data_batch(target_state)` with blocking | `pop_next_data_batch()` non-blocking FIFO | no condition variable |
| `idata_batch_probe`, `data_batch_processing_handle` | removed | no replacement |
| `set_state_change_cv` | removed | batches no longer back-ref repo cv |
| `batch_state` enum values: idle/task_created/processing/in_transit | idle/read_only/mutable_locked | |
| Locked-to-locked transitions | `readonly_to_mutable(ro&&)`, `mutable_to_readonly(mut&&)` | static consume via move |

`data_batch` now inherits `std::enable_shared_from_this<data_batch>` — callers must store it
as `shared_ptr<data_batch>` (raw/unique_ptr storage is now a compile error at the `to_read_only`
call site).

**cucascade NVTX dependency:** `CUCASCADE_NVTX` option exists on `origin/main` but defaults to
`OFF`. Sirius sets it OFF (neither `pixi.toml` nor `CMakeLists.txt` sets `CUCASCADE_NVTX`).
The `nvtx3` dep therefore does not need to be explicitly provided. No change required.

**cudf / RMM version pins inside cucascade:** cucascade `origin/main` CMakeLists.txt uses
`find_package(cudf REQUIRED CONFIG)` and `find_package(rmm REQUIRED CONFIG)` — no version
constraint embedded, it takes whatever Sirius's build environment provides. The Sirius pixi.toml
pins `libcudf = "26.04.*"` in both cuda12 and cuda13 features; cucascade does not override
this. No version bump required.

---

## 3. Sirius CMakeLists.txt and vcpkg.json Delta (All 7 dev PRs)

### 3.1 Cumulative build-graph changes from `origin/dev` to absorb

The 7 `origin/dev` PRs, applied in order (`986df0f` → `6f25eec` → `fd816f3` → `468f6e1` →
`aa0f29a` → `4c0f1ac` → `cdd6864`), produce the following net delta vs the current
`feature/single-node-multi-gpu2` CMakeLists.txt:

**Sources added to `EXTENSION_SOURCES`:**

```cmake
# Added by PR #675 (4c0f1ac)
src/io/admission_control.cpp
src/io/prefetching_cache.cpp
src/io/sirius_datasource.cpp
src/io/uring/uring_ioctx.cpp
src/io/uring/uring_reactor.cpp

# Added by PR #731 (aa0f29a)
src/scan_manager/parquet_split_provider.cpp
src/scan_manager/sirius_scan_manager.cpp
src/scan_manager/split_connector.cpp

# Added by PR #721 (cdd6864)
src/scan_manager/cached_split_provider.cpp
src/pin_table.cpp
```

**Sources removed from `EXTENSION_SOURCES`:**

```cmake
# Removed by PR #731 (aa0f29a)
src/op/scan/sirius_parquet_metadata_scan_operator.cpp
```

**New `find_package` / `pkg_check_modules` line (PR #675):**

```cmake
pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
```

**`target_link_libraries` changes (PR #675):**

```cmake
# Before (current branch)
target_link_libraries(sirius_extension PkgConfig::NUMA absl::any_invocable)

# After (origin/dev shape)
target_link_libraries(sirius_extension PkgConfig::NUMA PkgConfig::LIBURING
                      yaml-cpp::yaml-cpp absl::any_invocable)
target_link_libraries(sirius_loadable_extension PkgConfig::LIBURING)
```

Note: `yaml-cpp::yaml-cpp` is added to `sirius_extension` link in PR #675 despite already
being linked in the shared `foreach` loop above. This is harmless redundancy.

**Test sources changed (PRs #731 / #733):**

```cmake
# Removed by PR #731
test/cpp/pipeline/test_metadata_gpu_pipeline_task_counting.cpp
test/cpp/scan/test_metadata_gpu_scan_operators.cpp

# Added by PR #731
test/cpp/scan/test_split_connector.cpp

# Added by PR #733
test/cpp/pipeline/test_get_next_ports_after_sink.cpp
```

**vcpkg.json changes (PR #675 only):**

```json
// Added "liburing" to dependencies array
"dependencies": ["cudf", "yaml-cpp", "abseil", "numactl", "liburing"]
```

No other PRs (#731, #721, #739, #733, #734, #735) touch `CMakeLists.txt` for dependency
additions (PR #721 adds 2 source files; PR #733 adds 1 test source; PRs #734/#735 are CI-only).

### 3.2 pixi.toml changes

**None of the 7 `origin/dev` PRs modify `pixi.toml`.** The current pin
`libcudf = "26.04.*"` is unchanged. No new conda packages are needed.

---

## 4. CUDA / cuDF / RMM Compatibility

### 4.1 cucascade PRs #117, #112, #116 — no version bump required

All three PRs operate within the existing cudf 26.04 + RMM + CUDA 13 surface:

- PR #116: adds `cudf::table_view` ctor and `release_table(stream)` — these APIs exist in cudf
  26.04. No new cudf features required.
- PR #112: `bandwidth_profiler` uses `cuMemcpy` / converter-based copies — all within existing
  cudf+RMM. `pipeline_io_backend` per-device cache uses `cudaStream_t` + CUDA runtime APIs that
  exist in CUDA 12.0+.
- PR #117: RAII model uses `std::shared_mutex` (C++17), `std::enable_shared_from_this` (C++11),
  `std::atomic` with `wait/notify_all` (C++20). The atomic wait/notify requires C++20. Sirius is
  already compiled with C++20 (`CXX_STANDARD 20`). No new CUDA API usage.

### 4.2 Sirius PR #675 — no cudf/RMM version bump, but CUDA 13 kernel 5.1+ required

The `uring_reactor` uses `io_uring` which requires kernel >= 5.1. The build host runs kernel
6.17 (NVIDIA 595.58.03 driver), so this is satisfied.

The `sirius_datasource` sets `supports_device_read() = true` and performs H2D copies via
`cudaMemcpyAsync` from `cudaHostAllocPortable` bounce buffers. This is standard CUDA runtime
API. No cudf 26.04-specific API used in the io subsystem itself.

The `uring_ioctx` uses `templated_ioctx<uring_reactor>`, which uses C++20 concepts
(`io_object_c`, `io_reactor_c`) for the backend plug-in contract. Sirius already compiles with
C++20; CUDA standard 20 is also already set. No change needed.

### 4.3 CUDA arch sensitivity

No new CUDA kernels are introduced by any of the 7 Sirius PRs or 3 cucascade PRs. The io
subsystem is entirely CPU-side (host threads issuing `cudaMemcpyAsync`). The DataBatch RAII
model is also CPU-side locking. No new `.cu` files. Existing arch range `75-real` through `120`
is sufficient.

---

## 5. Toolchain / Compiler Considerations

### 5.1 C++20 concepts (PR #675)

`templated_ioctx.hpp` uses C++20 concepts (`io_object_c`, `io_reactor_c`) as load-bearing
constraints. GCC 12+ and Clang 14+ support C++20 concepts; the pixi.toml pins `clang = "21.*"`.
This is well within support range. No change needed.

### 5.2 `std::jthread` / `std::stop_token` (PR #675)

The `prefetching_cache` worker and evictor are `std::jthread` with `stop_token`. These are
C++20 standard library features. GCC 10+ and Clang 13+ support them. Clang 21 supports both.
No change needed.

### 5.3 `std::atomic::wait` / `notify_all` (PR #117, PR #675)

PR #117 uses `std::atomic<uint32_t>::wait` + `notify_all` (C++20 atomic wait). PR #675's
`prefetching_cache` also uses atomic wait. GCC 11+ and Clang 14+ support these on Linux.
Clang 21 is fine. No change needed.

### 5.4 `std::shared_mutex` (PR #117)

`data_batch` now holds a `std::shared_mutex`. This is C++17 standard; no concern.

### 5.5 Separable compilation

No new CUDA translation units are added by any PR. The existing
`CUDA_SEPARABLE_COMPILATION ON` setting is unchanged. The single `.cu` source
(`src/op/scan/equality_delete_mask.cu`) is unaffected.

---

## Recommended Stack (v1.4 Rebase Incremental View)

### Core Technologies (unchanged from v1.3, confirmed current)

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| cuDF | 26.04.* | GPU DataFrame ops, Parquet I/O | Unchanged |
| RMM | (bundled with cuDF 26.04) | GPU memory management | Unchanged |
| cuCascade | `73d00c4` (target pin) | Tiered memory, data_batch RAII | Bump from `62e0517` |
| CUDA | 13.x | GPU compute, cudaMemcpyAsync | Unchanged |
| DuckDB | 1.5.2 | SQL engine (pixi: `duckdb = "=1.5.2"`) | Unchanged |
| yaml-cpp | * (conda-forge) | Sirius config | Unchanged |
| spdlog | 1.8.* | Logging | Unchanged |
| abseil | * | `absl::any_invocable` | Unchanged |
| libnuma | * | NUMA-aware host allocation | Unchanged |

### New / Changed Dependencies for v1.4

| Dependency | Version | Source | Why Needed | PR |
|------------|---------|--------|------------|-----|
| liburing (runtime) | 2.5 (host) | System apt | io_uring userspace lib | #675 |
| liburing-dev (headers) | 2.5 | **`apt install liburing-dev`** | `<liburing.h>` for uring_reactor | #675 |
| liburing (vcpkg) | latest in baseline | vcpkg registry | vcpkg build path only | #675 |

### Supporting Libraries (build-graph additions)

| Library | How Linked | Why Added | PR |
|---------|-----------|------------|-----|
| `PkgConfig::LIBURING` | `sirius_extension` + `sirius_loadable_extension` | `uring_reactor.cpp` needs `-luring` | #675 |
| `moodycamel::ConcurrentQueue` | via DuckDB's global `include_directories(third_party/concurrentqueue)` | `uring_reactor.hpp` uses `duckdb_moodycamel::ConcurrentQueue` | #675 |

The concurrentqueue dependency does NOT require a new `find_package` or vcpkg entry.
`uring_reactor.hpp` uses the `duckdb_moodycamel::` namespace variant of moodycamel, whose
header is already on the include path because DuckDB's `CMakeLists.txt:709` calls
`include_directories(third_party/concurrentqueue)` as a global directive. Since Sirius is
compiled as a subdirectory within the DuckDB build, that path is inherited automatically.

---

## Installation

### System package (required before building)

```bash
# Required for PR #675 uring_reactor backend
sudo apt-get install -y liburing-dev          # installs liburing.h + pkg-config .pc file
```

### No pixi.toml changes needed

The default (`dev-libs + cuda13`) environment does not need modification. `liburing` has no
conda-forge package; the system package is the correct delivery mechanism (same as libnuma-dev
for `PkgConfig::NUMA`).

### vcpkg.json (vcpkg build path only)

The `"liburing"` entry added by PR #675 is already in the `origin/dev` vcpkg.json. When
merging, this entry must be preserved in the rebased `vcpkg.json`. Verify that the
`builtin-baseline` (`ffc071e0c08432c60c9b64f00334c0227667931b`) resolves `liburing` — if not,
a custom port under `vcpkg_ports/liburing/` may need to be added. No custom port exists today.

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| `pkg_check_modules(LIBURING …)` | Bundled io_uring shim | liburing 2.x ships with every Ubuntu 22.04+ host; system package is simpler and avoids vendoring a kernel-coupled library |
| cucascade RAII model (`shared_ptr<data_batch>`) | Keep old FSM + unique_ptr | PR #117 is already on `origin/main`; Sirius PR #739 already migrates the non-mgpu surface; RAII model eliminates TOCTOU races on batch state |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `libcufile` / kvikio backend in `sirius_datasource` | Single CUDA context binding; unsafe for multi-GPU dispatch | `uring_reactor` path (`io_uring` + pinned host bounce buffers) |
| `rmm::cuda_stream_default` anywhere in the io subsystem | User rule (HYG-01/02 gate); cross-device stream confusion | Explicit `rmm::cuda_stream_view` passed through `device_read_req.stream` |
| `data_batch::to_read_only(ptr&&)` static (old API) | Removed in cucascade PR #117 | `batch->to_read_only()` non-static (caller retains `shared_ptr<data_batch>`) |
| Storing `data_batch` as `unique_ptr` or raw pointer | `enable_shared_from_this` requires `shared_ptr` storage; `to_read_only()` calls `shared_from_this()` and will throw or UB if stored otherwise | `shared_ptr<data_batch>` |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| cucascade `73d00c4` | cuDF 26.04, RMM (bundled), CUDA 13 | No internal version pins; takes whatever Sirius provides |
| liburing 2.5 | Linux kernel 6.17 (host) | Requires kernel >= 5.1 for `io_uring_setup`; host at 6.17 |
| C++20 atomic wait | Clang 21, GCC 12+ | Used by cucascade PR #117 + Sirius PR #675 `prefetching_cache` |
| C++20 concepts | Clang 21 | Used by Sirius PR #675 `io_reactor_c` / `io_object_c` |
| C++20 `std::jthread` | Clang 21 | Used by Sirius PR #675 worker/evictor threads |

---

## Phase-by-Phase Dependency Sequencing

For the roadmap planner: dependencies must be introduced before the source migrations that use
them.

| Phase topic | Dependency action required | PR driving it |
|-------------|---------------------------|---------------|
| cucascade rebase (rebase 11 local fixes onto `73d00c4`) | Bump cucascade submodule to `73d00c4`-based pin FIRST | cucascade #117 |
| DataBatch API migration (12 operators + 16 tests) | cucascade pin must already be updated | cucascade #117, Sirius #739 |
| IO Framework adoption (retire cucascade_datasource, adopt sirius_datasource) | Install `liburing-dev`; add `pkg_check_modules(LIBURING)` to CMakeLists; add source files | Sirius #675 |
| Scan Manager integration (parquet_split_provider, split_connector) | IO Framework must be present (sirius_scan_manager depends on sirius_datasource) | Sirius #731 |
| Pin Tables DDL | Scan Manager must be present (cached_split_provider extends sirius_scan_manager) | Sirius #721 |
| Refactors / CI (#733 / #734 / #735) | No dependency; can land at any point | Sirius #733–#735 |

A CMake-only sub-phase is NOT required — the liburing pkg_check_modules line and source-file
additions can land in the same commit as the io subsystem source files. However, the host must
have `liburing-dev` installed before the first build attempt.

---

## Sources

All findings from direct `git show` inspection of actual commits. No WebSearch or Context7
required — the evidence is in the repository itself.

- `git show 4c0f1ac -- CMakeLists.txt vcpkg.json` — PR #675 build-graph delta (liburing)
- `git show 4c0f1ac -- src/include/io/uring/uring_reactor.hpp` — `#include <liburing.h>`
- `git -C cucascade show 73d00c4 -- include/cucascade/data/data_batch.hpp` — PR #117 RAII surface
- `git -C cucascade show 47e430e -- include/cucascade/data/gpu_data_representation.hpp` — PR #116 API changes
- `git -C cucascade show 0cd4a6a` (PR #112) — bandwidth profiler, no new deps
- `git show aa0f29a -- CMakeLists.txt` — PR #731 scan_manager source additions
- `git show cdd6864 -- CMakeLists.txt` — PR #721 cached_split_provider + pin_table additions
- `git show 468f6e1` — PR #739 API migration scope (`.get_table()` → `.get_table_view()`)
- `dpkg -l liburing*`, `apt-cache show liburing-dev` — host package state verification
- `ldconfig -p | grep liburing` — runtime library presence
- `duckdb/CMakeLists.txt:709` — `include_directories(third_party/concurrentqueue)` global

---
*Stack research for: Sirius v1.4 rebase — cucascade origin/main + Sirius origin/dev*
*Researched: 2026-05-04*
