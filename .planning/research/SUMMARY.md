# Project Research Summary

**Project:** Sirius v1.4 — Rebase After DataBatch Changes
**Domain:** GPU-native SQL engine — upstream API rebase (cucascade origin/main + Sirius origin/dev)
**Researched:** 2026-05-04
**Confidence:** HIGH

## Executive Summary

Sirius v1.4 integrates three cucascade upstream PRs (#116, #112, #117) and seven Sirius origin/dev PRs (#663, #706, #713, #733, #734, #735, #675, #731, #721, #739) onto the existing `feature/single-node-multi-gpu2` branch, which carries 11 local cucascade fixes and the entire v1.3 multi-GPU surface (Phases 9-15). The dominant challenge is cucascade PR #117's breaking RAII redesign of `data_batch`: it deletes 21 API symbols and requires every Sirius operator, pipeline utility, and test file that calls `batch->get_data()` to acquire a typed accessor (`read_only_data_batch` or `mutable_data_batch`) before touching batch contents. The Phase 13 stream-lineage fix (`writer_stream`/`writer_event`) must survive as a manual re-application on top of the new #117 shape; the only alternative is silent regression of the SF100 Q11 illegal-address fix.

The recommended approach is a strict 6-phase sequence governed by the compile-graph: cucascade submodule rebase first (Phase 16), then the Sirius dev-merge base layer before DataBatch API migration so that the auto-merge machinery starts from a compiling tree (Phase 17), then the DataBatch migration itself (Phase 18), then IO Framework (Phase 19), then Scan Manager plus Pin Tables (Phase 20), and finally the v1.4 ship gate (Phase 21). IO Framework (#675) must precede Scan Manager (#731) because `parquet_split_provider::run_batch` constructs a `sirius_datasource` for planning-time footer reads — landing #731 without #675 breaks at that call site. Splitting the dev-merge into its own phase (17) rather than bundling it with DataBatch migration (18) allows the CI+refactor auto-merges to be committed and reviewed separately from the RAII accessor rewrites, enabling clean bisection if either step regresses.

The key systemic risk is the cucascade rebase itself: six conflict files (`data_batch.hpp`, `gpu_data_representation.hpp`, `representation_converter.cpp`, `pipeline_io_backend.cpp`, `memory/common.cpp`, `memory/memory_space.cpp`) each carry Sirius-local fixes that must survive the merge. Portable/Mapped memory flags, the stream-lineage `writer_stream` constructor argument, and the `io_worker` destruction-order fix are all invisible to the compiler if lost — they appear only as SF100 or multi-GPU runtime failures. Each must be verified with targeted grep gates before integration tests begin.

---

## Key Findings

### Recommended Stack

The v1.4 rebase introduces one net-new system dependency: `liburing-dev` (the `<liburing.h>` headers for PR #675's `uring_reactor` backend). The runtime library (`liburing2:amd64 2.5`) is already installed on the build host, but `pkg_check_modules(LIBURING REQUIRED ...)` will fail until `sudo apt-get install -y liburing-dev` is run. No `pixi.toml` changes are needed — `liburing` has no conda-forge package and follows the same `PkgConfig::` pattern as `libnuma`. The cucascade submodule pin target changes from `62e0517` (current) to a new tip descending from `73d00c4` (PR #117) with our 11 local fixes rebased on top.

**Core technologies (unchanged):**
- cuDF 26.04, RMM (bundled), CUDA 13: unchanged from v1.3; no version bump required
- cucascade `73d00c4` target: breaking RAII DataBatch model; requires 11-fix rebase
- DuckDB 1.5.2: unchanged
- C++20 (atomic wait, concepts, jthread): already satisfied by Clang 21; no toolchain changes

**New dependency for v1.4:**
- `liburing-dev 2.5` (system apt, Phase 19 prereq): install before first build attempt after #675 lands

### Expected Features

**Table stakes (compile-blocking):**
- cucascade #117 DataBatch RAII model: 21 deleted API symbols; ~12 operators, ~16 tests, `batch_lock_utils.hpp` full rewrite
- cucascade #116 `get_table_view` / `release_table(stream)`: 14 call sites (migration reference in PR #739)
- Sirius #731 Scan Manager: deletes `sirius_parquet_metadata_scan_operator.hpp`; replaces metadata-scan pipeline pair with `parquet_split_provider` + `sirius_scan_manager`
- Sirius dev-merge auto-merges: 33 auto-merges across CI (#733, #734, #735) and config PRs (#663, #706, #713)

**Differentiators (additive, high value):**
- Sirius #675 IO Framework: `sirius_datasource` + `uring_ioctx` (per-GPU io_uring backend with pinned bounce buffers); retires `cucascade_datasource` stopgap
- cucascade #112 bandwidth profiler: per-device pipeline_io_backend cache fixing cross-context GPU-N disk I/O failure
- Sirius #721 Pin Tables DDL: `CALL pin_table(...)`/`CALL unpin_table(...)` builds on #731; single-GPU-resident pin is a known limitation (document and defer multi-GPU fix to v1.5)

**Defer to v1.5:**
- Per-GPU pinned copies in `cached_split_provider` (currently always pins to first GPU memory space)
- Multi-GPU-aware pinning via `cudaIpcMemHandle` or block-level split distribution
- bandwidth_profiler integration into SCHED-RR routing decisions

**Critical sequencing constraint (from FEATURES.md):**
PR #739 (`468f6e1`) targets cucascade pin `0cd4a6a` (pre-#117 API). It must NOT be cherry-picked before the cucascade rebase to `73d00c4`. Use it as a migration reference only — its file list identifies which operator files need touching; the actual per-site recipe changes under #117's accessor model.

### Architecture Approach

v1.4 retrofits four upstream changes onto nine v1.3 multi-GPU surfaces. The hardest conflict is Surface 7 (Phase 13 stream-lineage on `gpu_data_representation` / `representation_converter.cpp`) crossed with PR #117's restructuring of those same files. The recommended approach treats `representation_converter.cpp` as a re-implementation rather than a merge: start from `73d00c4`'s version, re-apply `writer_stream`/`writer_event` additions by hand, verify with a targeted grep gate before building. `SiriusContext` requires two field changes: `gpu_io_backends_` map retired and replaced with a per-GPU `sirius_ioctx` map; `scan_manager_` added.

**Phase 19/20 ordering resolution (FEATURES.md vs ARCHITECTURE.md disagreement):**
FEATURES.md proposed Scan Manager (Phase 19) before IO Framework (Phase 20). ARCHITECTURE.md proposes IO Framework (Phase 19) before Scan Manager (Phase 20). ARCHITECTURE.md's order is adopted here. The reason: `parquet_split_provider::run_batch` (introduced by #731) constructs a `sirius_datasource` for planning-time footer reads. If #731 lands before #675, that datasource type does not exist and the build fails at that call site. The compile-graph dependency `parquet_split_provider -> sirius_datasource` fixes the ordering unambiguously: IO Framework (#675) must precede Scan Manager (#731).

**Major components and integration seams:**
1. `cucascade/` submodule: RAII batch accessors, `gpu_data_representation` + `representation_converter` (stream-lineage lives here)
2. `src/include/io/`: `sirius_datasource` + `uring_ioctx` (per-GPU, constructed under `rmm::cuda_set_device_raii`)
3. `src/include/scan_manager/`: `parquet_split_provider`, `sirius_scan_manager`, `split_connector`
4. `src/include/sirius_context.hpp`: per-GPU resource caches; owns ioctx map and scan_manager field
5. `src/pipeline/task_scheduler.cpp`: SCHED-RR block at `management_eventloop` (must survive dev merge; verify `_no_pref_rr_counter` intact post-merge)
6. `src/include/pipeline/batch_lock_utils.hpp`: `lock_or_prepare_batch` requires RAII `to_mutable()` migration

### Critical Pitfalls

**P1 - RAII lock scope self-deadlock (Phase 16/18):**
Acquiring `read_only_data_batch` then calling any function that internally calls `to_mutable()` on the same batch blocks forever. Mitigation: scope every accessor to the narrowest block; use `readonly_to_mutable(std::move(ro))` for upgrades. Detect via TSan/Helgrind; `[mgpu_stress]` hang is the production signal.

**P2 - `writer_stream` lost in `representation_converter.cpp` conflict (Phase 16):**
Three-way merge of our `7ed84f2` + `62e0517` against `73d00c4` on this file can silently drop the `writer_stream` argument. Mitigation: re-implement the file from scratch from #117's shape; grep-verify `writer_stream`/`target_stream` at every construction site; run SF100 Q11 num_gpus=2 explicitly.

**P7 - PR #739 x #117 ordering mismatch (Phase 16):**
#739 targets pre-#117 cucascade API (`0cd4a6a`). Applying #739 before cucascade rebase to `73d00c4` produces 50+ build errors. Mitigation: complete cucascade rebase first; use #739 only as a file-list reference.

**P4 - `uring_reactor` inherits CUDA context from construction thread (Phase 19):**
Creating a single shared `sirius_ioctx` for all GPUs re-introduces the v1.1 kvikio anti-pattern. Mitigation: create one `uring_ioctx` per GPU in `SiriusContext::initialize()` under `rmm::cuda_set_device_raii`.

**P10 - Phase 13 stream-lineage work in deleted file (Phase 20):**
PR #731 deletes `sirius_parquet_metadata_scan_operator.hpp` which carried the Phase 13 `writer_stream` attachment. Accepting deletion without extracting stream-lineage logic regresses Q11. Mitigation: extract attachment points before accepting deletion; re-anchor in `sirius_gpu_parquet_scan_operator.cpp`.

**P6 - SCHED-RR counter stale after Scan Manager integration (Phase 20):**
v1.3 SCHED-RR counter increments in `management_eventloop`; under Scan Manager, split allocation moves to `parquet_split_provider`. If counter is not ported, all splits go to GPU 0. Mitigation: port `_no_pref_rr_counter` increment to `parquet_split_provider::start()`.

---

## Implications for Roadmap

### Phase 16: Cucascade Submodule Rebase + Pin Recovery
**Rationale:** All downstream work compiles against the cucascade API. The 11 local fixes must be rebased onto `73d00c4` before any Sirius migration can proceed. Highest conflict density and most critical correctness gates.
**Delivers:** cucascade pinned to `73d00c4`-descendant with all 11 local fixes intact; buildable cucascade against cuDF 26.04.
**Addresses:** cucascade #116, #112, #117
**Per-phase pitfalls:** P2 (writer_stream in representation_converter.cpp), P7 (#739 ordering), P8 (io_worker member-order), P9 (Portable/Mapped flags)
**Verification gates:**
- `grep -n "writer_stream" cucascade/src/data/representation_converter.cpp` non-zero at every construction site
- `grep -n "cudaHostAllocPortable" cucascade/src/memory/common.cpp cucascade/src/memory/memory_space.cpp` non-zero
- `_thread` is last member in `io_worker` class post-conflict-resolution
- cucascade builds cleanly against cuDF 26.04 in pixi env

### Phase 17: Sirius origin/dev Merge — Base Layer
**Rationale:** CI + refactor + CMake/config PRs (#663, #706, #713, #733, #734, #735) are largely auto-merges with no DataBatch API surface. Merging them as a dedicated phase keeps CI/config changes reviewable in isolation from the RAII rewrites and makes bisection viable. Compile errors from un-migrated DataBatch call sites are expected and acceptable at this phase boundary.
**Delivers:** `origin/dev` CI and CMake changes absorbed; 33 auto-merges committed; SCHED-RR block in `task_scheduler.cpp` verified intact.
**Addresses:** Sirius PRs #663, #706, #713, #733, #734, #735 (auto-merge surface)
**Sequencing constraint:** PR #739 must NOT be applied here as a cherry-pick — use only as a migration reference in Phase 18.
**Per-phase pitfalls:** P7 (verify old `batch_state` enum values not reintroduced), P6 (SCHED-RR counter must survive `task_scheduler.cpp` merge)
**Verification gates:**
- `grep -rn "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp` returns 1 match
- `grep "SCHED-RR" src/pipeline/task_scheduler.cpp` returns the round-robin block
- Zero occurrences of `task_created\|in_transit` from old `batch_state` enum in any merged file

### Phase 18: DataBatch RAII Migration
**Rationale:** Makes the Sirius codebase compile against the new cucascade pin. Until complete, no integration testing is possible. The #739 diff is used as a file-list reference; every call-site recipe is applied against the #117 API shape.
**Delivers:** All ~12 operators, ~16 tests, `batch_lock_utils.hpp`, `convertible_data_batch.hpp`, and `data_batch_utils.hpp` migrated to RAII accessors; `pop_data_batch(state)` calls replaced with `pop_next_data_batch()` plus proper wait patterns; Sirius compiles cleanly against `73d00c4`-pin cucascade.
**Addresses:** cucascade #117 full Sirius surface; Sirius #739 (as reference)
**Per-phase pitfalls:** P1 (RAII lock scope self-deadlock), P3 (pop_next_data_batch non-blocking semantics)
**Verification gates:**
- `grep -rn "task_created\|in_transit\|pop_data_batch(" src/` returns zero
- `grep -rn "batch->get_data()\|\.get_data()" src/` returns zero
- `[TPC-H][parquet]` 22/22 correctness on num_gpus=1
- `[mgpu]` 16/16 on num_gpus=2

### Phase 19: IO Framework Adoption (#675)
**Rationale:** IO Framework (#675) must precede Scan Manager (#731) because `parquet_split_provider::run_batch` calls `sirius_datasource` for planning-time footer reads — the Scan Manager won't compile without it. This also retires the `cucascade_datasource` stopgap and establishes per-GPU `sirius_ioctx` infrastructure.
**Prerequisite:** `sudo apt-get install -y liburing-dev` must be run before the first build attempt in this phase.
**Delivers:** `cucascade_datasource` retired; per-GPU `sirius_ioctx` map in `SiriusContext`; `parquet_scan_task.cpp` two call sites migrated to `sirius_datasource`; HYG-02 gate clean.
**Addresses:** Sirius #675; STACK.md liburing-dev requirement
**Per-phase pitfalls:** P4 (uring_reactor single CUDA context - per-GPU ioctx required), P5 (global admission_control budget), P11 (HYG-02 raw cudaSetDevice in uring_reactor.cpp)
**Verification gates:**
- `grep -rn "cudaSetDevice\b" src/io/` returns zero raw calls
- `grep -c "rmm::cuda_stream_default" src/` <= 40 (HYG-02 gate)
- `[TPC-H][parquet]` 22/22 on num_gpus=2
- `nvidia-smi dmon` shows non-zero PCIe activity on GPU 1 during SF10 parquet scan

### Phase 20: Scan Manager + Pin Tables Port (#731 + #721)
**Rationale:** Builds on Phase 19's `sirius_datasource` foundation. Requires re-planting `_batch_gpu_affinity`, porting Phase 13 stream-lineage to the new scan path, and attaching SCHED-RR counter to `parquet_split_provider`'s split-emission loop. PR #721 (Pin Tables) depends on #731 and lands in the same phase.
**Delivers:** `sirius_scan_manager` integrated into `SiriusContext`; `parquet_split_provider` + `split_connector` driving splits; `_batch_gpu_affinity` re-planted; Phase 13 stream-lineage re-anchored in `sirius_gpu_parquet_scan_operator.cpp`; `CALL pin_table(...)` DDL functional (single-GPU-resident, documented limitation).
**Addresses:** Sirius #731, #721; re-attachment of Phase 9 affinity and Phase 13 stream-lineage
**Per-phase pitfalls:** P6 (SCHED-RR counter stale under split_provider), P10 (Phase 13 work in deleted file)
**Verification gates:**
- `grep -rn "writer_stream\|record_writer_event" src/op/scan/` non-zero
- `[mgpu_stress]` 500-iter test exits 0
- AUDIT TEST_CASE disjointedness REQUIRE (`std::set_intersection == empty`) fires green on num_gpus=2
- SF100 Q11 num_gpus=2 passes (not just `[mgpu]` suite - scale matters for stream-lineage race)
- `CALL pin_table(...)` functional on single-GPU; multi-GPU limitation documented in code comment

### Phase 21: v1.4 Ship Gate
**Rationale:** Full v1.3 regression on the rebased shape; no new features. Confirm TPC-H 22/22, multi-GPU stress, and SF100 benchmarks all pass at the same quality bar as v1.3.
**Delivers:** Release tag for v1.4; all regression gates green; RETROSPECTIVE written.
**Addresses:** Full suite: `[TPC-H][parquet]` 22/22, `[mgpu]` 16/16, `[mgpu_stress]` 500-iter, SF100 Q1/Q11 num_gpus=2
**Per-phase pitfalls:** All P1-P11 should be fully resolved by this phase; this is final confirmation.

### Phase Ordering Rationale

- **Phase 16 before all others:** cucascade API shape governs every Sirius source file; Phase 13 conflict files must be re-implemented correctly before any Sirius migration can proceed.
- **Phase 17 (dev-merge base layer) before Phase 18 (DataBatch migration):** Splitting CI/CMake auto-merges from RAII rewrites allows each to be committed and reviewed independently; merge errors in unrelated files don't contaminate the DataBatch migration diff.
- **Phase 18 before Phase 19:** `sirius_datasource` creates batches that downstream code accesses via RAII accessors; landing IO framework on un-migrated batch accessor code produces a non-compiling tree.
- **Phase 19 before Phase 20:** `parquet_split_provider::run_batch` calls `sirius_datasource` — compile-graph dependency. This overrides FEATURES.md's original proposal (Scan Manager first); ARCHITECTURE.md's ordering is adopted.
- **Phase 20 combines #731 + #721:** `cached_split_provider` inherits from `split_provider` (defined in #731); the two PRs share no conflicting concerns, combining avoids a two-step scan manager integration cycle.

### Research Flags

Standard patterns (no additional per-phase research needed):
- **Phase 16:** Well-understood rebase procedure; all conflict files identified; verification greps known
- **Phase 17:** Auto-merge; SCHED-RR survival check is a 2-grep verification
- **Phase 18:** Mechanical migration; full recipe documented in FEATURES.md and ARCHITECTURE.md
- **Phase 21:** Standard regression run; no new patterns

May benefit from targeted pre-implementation review:
- **Phase 19:** Multi-GPU ioctx construction — per-GPU `rmm::cuda_set_device_raii` pattern in `SiriusContext::initialize()` should be designed before implementation; admission_control budget multiplier needs explicit configuration decision
- **Phase 20:** SCHED-RR counter attachment point in `parquet_split_provider` — exact split-emission loop location should be confirmed in PR #731's source before committing the porting approach

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All findings from direct `git show` on actual commits; `dpkg -l` host package state confirmed |
| Features | HIGH | All API surfaces inspected from actual PR diffs; migration scope from live grep on current branch |
| Architecture | HIGH | All v1.3 surface files inspected; build-order dependency graph derived from API call chains; PR landing order on origin/dev confirmed |
| Pitfalls | HIGH | Derived from v1.1-v1.3 post-mortems and direct source inspection of conflict files; no training-data assertions |

**Overall confidence:** HIGH

### Gaps to Address

- **SCHED-RR counter attachment point in parquet_split_provider:** Exact line in `parquet_split_provider::start()` for the round-robin increment not confirmed from source; inspect `git show aa0f29a -- src/scan_manager/parquet_split_provider.cpp` before Phase 20 implementation.
- **`_batch_gpu_affinity` in #731-shaped operator header:** Absence inferred from line-count drop (173 to shorter); confirm exact content of post-#731 `sirius_gpu_parquet_scan_operator.hpp` before designing the Phase 20 re-plant.
- **vcpkg `liburing` baseline resolution:** `builtin-baseline ffc071e0c0` may or may not resolve `liburing` from the official vcpkg registry; verify during Phase 19 before first vcpkg build path attempt. (Affects vcpkg path only, not default pixi build.)
- **`sirius_scan_manager::prepare_for_query` ordering in `sirius_engine.cpp`:** Confirm exact insertion point from PR #731 diff before Phase 20 implementation.

---

## Sources

All findings from direct git inspection of commits in the repository. No web search or external documentation required.

**cucascade commits (direct inspection):**
- `73d00c4` — PR #117 RAII DataBatch model (breaking API diff)
- `47e430e` — PR #116 `gpu_data_representation` from `cudf::table_view`
- `0cd4a6a` — PR #112 bandwidth profiler, per-device pipeline_io_backend fix

**Sirius origin/dev commits (direct inspection):**
- `4c0f1ac` — PR #675 IO Framework (`uring_reactor`, `sirius_datasource`, `liburing` CMake)
- `aa0f29a` — PR #731 Scan Manager (`parquet_split_provider`, `sirius_scan_manager`, filter translation at execute() time)
- `cdd6864` — PR #721 Pin Tables (`cached_split_provider`, `pin_table` DDL)
- `468f6e1` — PR #739 migration reference (pre-#117 cucascade compat)

**Current branch source files (direct inspection):**
- `src/include/sirius_context.hpp` — per-GPU backend cache fields
- `src/op/scan/parquet_scan_task.cpp` — `cucascade_datasource` call sites at lines 337, 904
- `cucascade/include/cucascade/data/gpu_data_representation.hpp` — `writer_stream` ctor shape
- `src/include/pipeline/task_scheduler.hpp` — SCHED-RR counter field

**Planning documents (direct inspection):**
- `.planning/phases/13-*/13-04-SUMMARY.md` — stream-lineage architecture
- `.planning/phases/14-*/14-CONTEXT.md` — SCHED-RR diff
- `.planning/phases/15-*/15-AUDIT-LOG.md` — colocation invariant classification
- `.planning/RETROSPECTIVE.md` — v1.2 Phase 10 destruction-order post-mortem
- `.planning/MILESTONES.md` — v1.3 Phase 13 conflict file list

**Host verification:**
- `dpkg -l liburing*`, `apt-cache show liburing-dev` — liburing-dev absent; runtime present
- `ldconfig -p | grep liburing` — `liburing.so.2` present at `/lib/x86_64-linux-gnu/`

---
*Research completed: 2026-05-04*
*Ready for roadmap: yes*
