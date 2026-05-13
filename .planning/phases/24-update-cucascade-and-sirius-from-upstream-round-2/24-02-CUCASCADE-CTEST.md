# 24-02: Cucascade Rebase + ctest Evidence

**Date:** 2026-05-13
**Executor:** Claude (Plan 24-02)
**Rebase target:** upstream `origin/main` HEAD `9ceebaa`
**Fork branch:** `fix/pinned-portable-flags`

---

## Section A: Post-rebase Commit List

`git log --oneline 9ceebaa..HEAD` (9 commits — 8 original + 1 test-fix):

```
5203de5 fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor
1522e0b fix(p23): run_p2p_probe_locked must restore device context on exit
4319726 fix(p23): cuda_set_device_raii guard for HtoD in alloc_and_peer_copy_async
b21bd97 fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)
e10bd4a style: pre-commit cleanup (clang-format + codespell)
c15cb01 fix(stream-lineage): writer_stream/writer_event on gpu_table_representation + cudaStreamWaitEvent
d5ac57b fix(representation_converter): P2P override — target-bound stream, DMA probe at init
3c44dae fix(pipeline_io_backend): reorder io_worker members so _thread is last
4b94571 fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene
```

**New cucascade fork HEAD (full):** `5203de5a028ccb57402a4105e35282c567c3ee5a`
**New cucascade fork HEAD (short):** `5203de5`

Pre-rebase commit `49134ff` (Stop enabling C for C++ and CUDA builds) was DROPPED as it was already upstream (already absorbed in `bcddb89` / `9ceebaa` base).

---

## Section B: Invariant Grep Gates

All gates verified on `fix/pinned-portable-flags` at `5203de5`:

| Gate | File | Expected | Result |
|------|------|----------|--------|
| HYG: no stream_default | `src/data/representation_converter.cpp` | 0 hits | **PASS: 0 hits** |
| alloc_and_peer_copy_async preserved | `src/data/representation_converter.cpp` | >= 1 hit | **PASS: 6 hits** |
| src_guard present (same-stream invariant) | `src/data/representation_converter.cpp` | line 622 | **PASS** |
| dst_guard present (Phase 23 HtoD guard) | `src/data/representation_converter.cpp` | line 649 | **PASS** |
| rmm::cuda_set_device_raii dst_guard | `src/data/representation_converter.cpp` | >= 1 hit | **PASS** |
| run_p2p_probe_locked preserved | `src/memory/common.cpp` | line 48 | **PASS** |
| saved_device save-restore pattern | `src/memory/common.cpp` | lines 56-57, 146 | **PASS** |
| writer_event / writer_stream (commit 4) | `include/cucascade/data/gpu_data_representation.hpp` | >= 1 hit | **PASS** |

**Key grep outputs:**

```
# representation_converter.cpp
622:    rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};
649:    rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}};

# common.cpp
48:void run_p2p_probe_locked(int device_count)
56:  int saved_device = 0;
57:  (void)cudaGetDevice(&saved_device);
146:  cudaSetDevice(saved_device);
```

---

## Section C: Cucascade Build Summary

Build environment:
- PIXI env: `/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.pixi/envs/default`
- GCC 14.3.0 (conda-forge), CUDA 13.1
- cmake 4.1.3
- CMAKE_CUDA_ARCHITECTURES=86 (RTX 6000 Ada, Ampere/Lovelace)
- CUCASCADE_BUILD_TESTS=ON

Build targets completed:
- `cucascade_objects` — static objects library
- `cucascade_static` — `libcucascade.a`
- `cucascade_shared` — `libcucascade.so`
- `cucascade_benchmarks` — benchmark executable
- `cucascade_tests` — Catch2 test executable

**One deviation from plan:** upstream `96bfea1` added a new test `test_data_representation:slice-roundtrip` that used the old 2-arg `gpu_table_representation` constructor. Our commit `c15cb01` (085d917 pre-rebase, stream-lineage) requires a `writer_stream` as 3rd argument. Added a separate commit `5203de5` to fix this incompatibility — see Section E.

---

## Section D: ctest PASS Evidence

```
Test project /home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade/build
    Start 1: cucascade_tests
1/1 Test #1: cucascade_tests ..................   Passed   14.49 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =  14.50 sec
```

**Exit code: 0 (PASS)**
**Test count:** 1/1 (single `cucascade_tests` target, itself a Catch2 multi-test runner)
**Runtime:** 14.49s

---

## Section E: OBSOLETED Dropped Commits + Re-derivation Actions

### OBSOLETED (dropped during rebase):

| Original SHA | Subject | Reason |
|-------------|---------|--------|
| `49134ff` | Stop enabling C for C++ and CUDA builds (#123) | Already present in upstream at `bcddb89` (new rebase base); git rebase auto-detected "patch contents already upstream" and dropped it |

### Re-derivation Actions:

**Commit 3 (8392c3d → d5ac57b): RE-DERIVE on new shape**

Conflict: `src/data/representation_converter.cpp` at `convert_host_fast_to_gpu()` function.

Resolution: The conflict was a single line in `convert_host_fast_to_gpu()`:
- Upstream (96bfea1): `reconstruct_column(col_meta, *fast_table->allocation, stream, mr, batch)` — `*` dereference because `allocation` changed from `unique_ptr` to `shared_ptr`
- Our commit: `reconstruct_column(col_meta, fast_table->allocation, target_stream, mr, batch)` — old unique_ptr style + our `target_stream` fix
- **Resolution:** `reconstruct_column(col_meta, *fast_table->allocation, target_stream, mr, batch)` — take upstream's `*` dereference AND keep our `target_stream` (multi-GPU fix)

This is the canonical D-02 re-derivation: keep our bug-fix intent on upstream's new code shape.

**Additional fix (5203de5): Upstream test + our API incompatibility**

Upstream `96bfea1` added test `host_data_representation::slice round-trip preserves selected columns` using the old 2-arg `gpu_table_representation(table, space)` constructor. Our commit `c15cb01` requires 3 args including `writer_stream`. Added commit `5203de5` to pass `stream.view()` as the third argument. This is a Rule 1 auto-fix (test compilation failure = bug in test's use of our modified API).

---

## Task 2 Outcome

**NEW_CC_HEAD (full):** `5203de5a028ccb57402a4105e35282c567c3ee5a`
**NEW_CC_HEAD (short):** `5203de5`

**D-04 Commit B (API adapter):**
- SHA: `ff06fac`
- Subject: `fix(p24): adapt sirius to cucascade 96bfea1 host_table_allocation API changes (D-04 Commit B)`
- Files: 6 (multiple_blocks_allocation_accessor.hpp, host_table_chunk_reader.hpp, host_table_chunk_reader.cpp, cpu_source_task.cpp, duckdb_scan_task.cpp, test_host_table_utils.cpp)
- Required because 96bfea1 made host_table_allocation constructor private and changed allocation from unique_ptr to shared_ptr. Our sirius source used make_unique<host_table_allocation> and unique_ptr const& references.
- Fix: templatize accessor methods, use ::create() factory, change _allocation member to shared_ptr

**D-04 Commit A (gitlink-bump):**
- SHA: `d228504`
- Subject: `submodule: bump cucascade to 5203de5 (p24 rebase onto 9ceebaa)`
- Atomic check: PASS — `git show --name-only d228504 | grep -v '^$'` = exactly `cucascade`
- Submodule status: no leading `+` (clean checkout matching gitlink)

**MCP build pre-commit:** PASS (`[120/120]` all targets, no errors)

**MCP build post-commit:** PASS (`[90/90]` all targets linked, no errors, warnings only in Rust component)

**git submodule status cucascade:**
```
 5203de5a028ccb57402a4105e35282c567c3ee5a cucascade (heads/fix/pinned-portable-flags)
```
(No leading `+` — clean checkout matching gitlink)
