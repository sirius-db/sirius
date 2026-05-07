---
phase: 22-multi-gpu-pinning-stream-lineage-hardening
plan: 03
subsystem: cucascade
tags: [cucascade, stream-lineage, fu17, cluster-b, mgpu]
status: PARTIAL
requirements:
  - fu17-cluster-b
dependency_graph:
  requires:
    - cucascade pin 1c1e648 (Phase 16 CC-03 baseline; carried through Phases 17-21)
    - 20-05-INVESTIGATION.md race taxonomy (Cluster A vs Cluster B; recommended fix shape 1 same-stream invariant)
    - cucascade gpu_data_representation writer-event API (cucascade/include/cucascade/data/gpu_data_representation.hpp:56-208)
  provides:
    - cucascade local-fork commit c666b21 advancing the local pin from 42a01c4 to c666b21 with same-stream invariant fix in alloc_and_peer_copy_async
    - Compile-clean cucascade objects (verified by parent build's cucascade compile + libcucascade.a link steps; both succeeded)
  affects:
    - Plan 22-04 (submodule pin bump; will advance Sirius parent's gitlink to c666b21)
    - Plan 22-06 (sanitizer gate that empirically asserts Cluster B = 0 post-fix; this plan defers the SF1 Q11 num_gpus=2 sanitizer micro-validation to Plan 22-04 due to parallel-wave parent-build incompleteness — see Deviations section)
tech-stack:
  added: []
  patterns:
    - "D-07 same-stream invariant: rmm::device_buffer::allocate_async and both cudaMemcpyAsync legs share target_stream as the single CUDA-stream argument; cuda_set_device_raii(src_device) provides only the device-context binding for the DtoH leg"
    - "Sync-then-cudaFreeHost ordering preserved (Pitfall 4); cudaStreamSynchronize(target_stream.value()) inside src_guard scope replaces the prior src_stream.synchronize()"
key-files:
  created: []
  modified:
    - cucascade/src/data/representation_converter.cpp (alloc_and_peer_copy_async, lines 593-633 post-fix)
decisions:
  - "Selected fix shape 1 (same-stream invariant) per CONTEXT.md D-07. Did not attempt fix shape 2 (event-bridge) because shape 1 is the smallest diff and mirrors the working peer-DMA path's discipline (lines 605-606 already pass target_stream as the only stream argument)."
  - "Did NOT advance Sirius parent's submodule pin (gitlink). Per CONTEXT.md D-08 and the plan's Task 1 step 7, the pin advance is Plan 22-04's explicit responsibility."
  - "Did NOT push the cucascade commit to the upstream cucascade origin (NVIDIA/cuCascade). Per CONTEXT.md D-08 / CC-UPSTREAM-01 carry pattern: local pin advance only; track diff in 22-CUCASCADE-DIFF.md (Plan 22-07 will write that)."
  - "DEFERRED Task 2 sanitizer micro-validation to Plan 22-04. Rationale: parallel-wave Plan 22-01 left Sirius parent in a transient build-broken state (pinned_entry::memory_space → chunk_memory_spaces vector rename in header without yet-completed .cpp updates), so a fresh sirius_unittest binary cannot be built against the bumped cucascade in this Plan 22-03 window. Plan 22-04 inherently runs after Wave 1 completion and is the natural execution site for the sanitizer micro-validation."
metrics:
  duration: 4min
  completed: 2026-05-07T23:00Z
  tasks_completed: 1
  tasks_deferred: 1
  files_modified: 1
  cucascade_commits: 1
  parent_commits: 0
---

# Phase 22 Plan 03: Cucascade Cluster B same-stream invariant fix Summary

Cucascade local-fork commit `c666b21` collapses `alloc_and_peer_copy_async`'s host-staging fallback path onto the single `target_stream` argument so that `rmm::device_buffer::allocate_async` (line 600) and both `cudaMemcpyAsync` legs (DtoH + HtoD) observe one stream timeline. The local rmm::cuda_stream `src_stream` and its `synchronize()` are removed; the DtoH leg is now issued under `cuda_set_device_raii(src_device)` on `target_stream` with a `cudaStreamSynchronize(target_stream.value())` inside the src_guard scope to preserve the sync-then-`cudaFreeHost` ordering required by Pitfall 4. The fix is the smallest diff (11 insertions / 3 deletions in one function) that mirrors the discipline already in place on the working peer-DMA path (lines 605-606).

## Status: PARTIAL

**Task 1 (same-stream invariant fix): COMPLETE** — committed at cucascade `c666b21` on branch `fix/pinned-portable-flags` (descendant of pin `1c1e648`); parent build's cucascade compile + libcucascade.a link both succeeded; all eight grep-gate acceptance criteria pass; clang-format clean.

**Task 2 (sanitizer micro-validation): DEFERRED to Plan 22-04** — the parallel-wave Plan 22-01 left Sirius parent in a transient build-broken state (header member rename `pinned_entry::memory_space` → `chunk_memory_spaces` vector without yet-completed `.cpp` updates), so no fresh `sirius_unittest` binary can be linked against the bumped cucascade pin in this Plan 22-03 window. Plan 22-04 explicitly runs after Wave 1 and is the natural execution site.

## Pre-fix vs Post-fix diff

### Pre-fix (lines 593-625, cucascade pin `42a01c4` baseline)

```cpp
static rmm::device_buffer alloc_and_peer_copy_async(const void* src_ptr,
                                                    int src_device,
                                                    std::size_t size,
                                                    int dst_device,
                                                    rmm::cuda_stream_view target_stream,
                                                    rmm::device_async_resource_ref target_mr)
{
  rmm::device_buffer buf(size, target_stream, target_mr);                     // L600
  if (size == 0 || src_ptr == nullptr) { return buf; }

  if (memory::probe_peer_dma_works(src_device, dst_device)) {                 // L603
    CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(
      buf.data(), dst_device, src_ptr, src_device, size, target_stream.value()));
    return buf;
  }

  // Peer DMA broken — explicitly stage through pinned host memory.
  void* host_buf = nullptr;
  CUCASCADE_CUDA_TRY(cudaMallocHost(&host_buf, size));
  {
    rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};    // L614
    rmm::cuda_stream src_stream;                                              // L615 BUG: split stream
    CUCASCADE_CUDA_TRY(
      cudaMemcpyAsync(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, src_stream.view().value())); // L617 BUG
    src_stream.synchronize();                                                 // L618 BUG: syncs only src_stream
  }
  CUCASCADE_CUDA_TRY(
    cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value())); // L621
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));           // L622
  cudaFreeHost(host_buf);                                                     // L623
  return buf;
}
```

### Post-fix (lines 593-633, cucascade `c666b21`)

```cpp
static rmm::device_buffer alloc_and_peer_copy_async(const void* src_ptr,
                                                    int src_device,
                                                    std::size_t size,
                                                    int dst_device,
                                                    rmm::cuda_stream_view target_stream,
                                                    rmm::device_async_resource_ref target_mr)
{
  rmm::device_buffer buf(size, target_stream, target_mr);                     // L600
  if (size == 0 || src_ptr == nullptr) { return buf; }

  if (memory::probe_peer_dma_works(src_device, dst_device)) {                 // L603
    CUCASCADE_CUDA_TRY(cudaMemcpyPeerAsync(
      buf.data(), dst_device, src_ptr, src_device, size, target_stream.value()));
    return buf;
  }

  // Peer DMA broken — explicitly stage through pinned host memory.
  void* host_buf = nullptr;
  CUCASCADE_CUDA_TRY(cudaMallocHost(&host_buf, size));
  {
    // Phase 22 D-07: same-stream invariant. Issue DtoH on target_stream     // L614
    // (matching rmm::device_buffer::allocate_async at the top of this        // L615
    // function) under cuda_set_device_raii(src_device) for src-side          // L616
    // context. Closes Cluster B sanitizer race shape A                       // L617
    // (16/21 of SF1 Q11 num_gpus=2 races per 20-05-INVESTIGATION.md).        // L618
    rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};    // L619 PRESERVED
    CUCASCADE_CUDA_TRY(
      cudaMemcpyAsync(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, target_stream.value())); // L621 FIXED: target_stream
    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));         // L622 FIXED: sync target_stream
    // Sync inside the src_guard scope: cudaFreeHost (after the closing       // L623
    // brace below) is host-synchronous and must not race with the DtoH       // L624
    // read; the sync also ensures host_buf is fully populated before         // L625
    // the HtoD enqueue executes on target_stream.                            // L626
  }
  CUCASCADE_CUDA_TRY(
    cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value())); // L629 PRESERVED
  CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));           // L630 PRESERVED
  cudaFreeHost(host_buf);                                                     // L631 PRESERVED
  return buf;
}
```

### Why this closes Cluster B

The 16/21 sanitizer race blocks all flagged "Use-before-alloc on allocation of size N bytes" with backtraces showing `rmm::device_buffer::device_buffer` (allocator on `target_stream` at line 600) as the producer and `cudaMemcpyAsync` (DtoH on the in-function `src_stream` at line 617) as the consumer. The producer-consumer pair was on two different streams with no event linkage between them — the sanitizer correctly classifies that as an unordered race even though the wall-clock `src_stream.synchronize()` typically masked it.

Post-fix, the producer (`rmm::device_buffer` ctor's `allocate_async` on `target_stream`) and consumer (DtoH on `target_stream`) are on the same stream — issued in source order — so the sanitizer sees a single ordered timeline. The HtoD leg at line 629 was already correctly on `target_stream`; this change just propagates that discipline backward to the DtoH leg.

## Cucascade commit

| Field           | Value                                                                                  |
| --------------- | -------------------------------------------------------------------------------------- |
| SHA (full)      | `c666b21926dec70b26a1febd509435635bea8deb`                                             |
| SHA (short)     | `c666b21`                                                                              |
| Branch          | `fix/pinned-portable-flags`                                                            |
| Parent          | `42a01c4` (style: pre-commit cleanup; clang-format + codespell)                        |
| Ancestry to pin | `git merge-base --is-ancestor 1c1e648 c666b21` ⇒ exit 0 (descendant of `1c1e648`) ✓    |
| Files changed   | `src/data/representation_converter.cpp` (+11 / −3)                                     |
| Commit message  | `fix(p22): same-stream invariant in alloc_and_peer_copy_async (Cluster B)`             |

This SHA is the value Plan 22-04 should use to advance the Sirius parent's submodule gitlink (currently still pointing at `42a01c4`).

## Acceptance criteria status

| Criterion                                                                                              | Status        |
| ------------------------------------------------------------------------------------------------------ | ------------- |
| Single-line grep: `cudaMemcpyAsync(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, target_stream.value())` returns 1 line | PASS (line 621) |
| `grep "rmm::cuda_stream src_stream"` returns 0 lines                                                   | PASS (0)      |
| `grep "src_stream\.synchronize()"` returns 0 lines                                                     | PASS (0)      |
| `grep "cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}}"` returns ≥1 line                | PASS (line 619) |
| HtoD `cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value())` ≥1   | PASS (line 629) |
| `cudaStreamSynchronize(target_stream.value())` count = 2 (one inside src_guard, one at function tail)  | PASS (2)      |
| Traceability: `Phase 22|D-07|Cluster B` ≥1 line                                                        | PASS (2 lines)|
| HYG-02: `rmm::cuda_stream_default` count = 0 in modified file                                          | PASS (0)      |
| clang-format dry-run --Werror                                                                          | PASS (exit 0) |
| Cucascade compile-only build via parent build chain (mcp build)                                        | PASS (steps [91/112] cucascade objects + [92/112] libcucascade.a link both succeeded) |
| Cucascade ctest from `cucascade/build/`                                                                | DEFERRED (standalone cucascade build dir is configured against an incompatible toolchain in this worktree; recommend running CC-04 ctest gate as part of Plan 22-04's post-bump validation) |
| `git -C cucascade log --oneline -1` shows the new fix commit on top of 1c1e648-descendant ancestry     | PASS (`c666b21 fix(p22): ...` parent `42a01c4`, ancestor `1c1e648`) |

## Sanitizer micro-validation (Task 2): DEFERRED to Plan 22-04

| Field                                  | Value                                                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `/tmp/p22_03_t2_sanitizer.log` exists  | NOT YET — deferred to Plan 22-04 post-bump                                                     |
| Cluster B count (`alloc_and_peer_copy_async` Host Frame) | not yet measured (gate target = 0)                                          |
| Cluster A count (`read_column_chunks_async`/`posix_device_io` Host Frame) | not yet measured (advisory only per D-09)                          |
| Total race blocks (`Use-before-alloc on allocation`) | not yet measured (informational only)                                              |
| Recommendation                         | Plan 22-04 should run the verbatim `timeout 600 compute-sanitizer ...` command from the plan's Task 2 against the post-bump `sirius_unittest` binary before landing the gitlink advance. |

### Rationale for deferral

Per the parallel-execution context, Plan 22-01 (Wave 1, parallel) edits the Sirius parent header `src/include/scan_manager/sirius_scan_manager.hpp` — specifically renaming `pinned_entry::memory_space` (single pointer) to `pinned_entry::chunk_memory_spaces` (per-chunk vector) per CONTEXT.md D-03. During Plan 22-03's execution window, the parallel agent had:

1. Updated the header (`src/include/scan_manager/sirius_scan_manager.hpp` shows `M` in `git status`)
2. Not yet completed updating all `.cpp` callers — concretely:
   - `src/scan_manager/sirius_scan_manager.cpp:107`: `entry.memory_space == nullptr` (now `chunk_memory_spaces`)
   - `src/scan_manager/sirius_scan_manager.cpp:176`: `*entry.memory_space` (now requires per-chunk indexing)
   - `src/sirius_extension.cpp:820`: `mem_space` passed to `insert_pinned_entry` (now expects `std::vector<memory_space*>`)

Three compile errors in the parent build prevent the link step that would produce a fresh `sirius_unittest` binary. Without that binary, the verbatim Task 2 sanitizer command cannot be executed:

```
timeout 600 compute-sanitizer --tool memcheck --track-stream-ordered-races=all \
  --show-backtrace=yes --launch-timeout=600 --log-file /tmp/p22_03_t2_sanitizer.log \
  --print-limit 100 \
  build/release/extension/sirius/test/cpp/sirius_unittest "gpu_execution - TPC-H Query 11 parquet"
```

The on-disk `sirius_unittest` binary at `build/release/extension/sirius/test/cpp/sirius_unittest` is dated 2026-05-07 14:32 — built before my cucascade edit at 17:58 — so running it would re-confirm the pre-fix Cluster B race count (16/21) rather than validate the fix.

**Plan 22-04 is the natural execution site for the sanitizer micro-validation:** it explicitly bumps the Sirius parent's submodule pin to the cucascade fix commit, and by then both parallel-wave agents (22-01 + 22-03) will have completed and the parent build will be clean. The Plan 22-04 SUMMARY should record:
- `CLUSTER_B` count from `/tmp/p22_03_t2_sanitizer.log` (or a renamed `/tmp/p22_04_*` log) with target = 0.
- `CLUSTER_A` count (advisory only per D-09).
- Total race blocks (informational only).
- If `CLUSTER_B > 0`: STOP and recommend escalation to event-bridge fix shape per CONTEXT.md D-07's allowed alternatives.

This deferral does NOT relax the sanitizer gate's strictness — it just relocates the gate to the first plan that has a fully-built post-bump binary. The cucascade commit `c666b21` is already in place; Plan 22-04 advances the gitlink and then runs the gate.

## Deviations from Plan

### Auto-classified

**1. [Rule 3 — Blocking] Cucascade ctest CC-04 gate run was not feasible from this worktree**

- **Found during:** Task 1 step 6 (cucascade ctest)
- **Issue:** The cucascade standalone build directory (`cucascade/build/`) in this worktree is configured against a CMake/CUDA toolchain combo (CMake 4.1 from a different env) that fails CMakeDetermineCompilerId for `.cu` files when re-run. The original cmake configure was performed in a different shell environment (probably a previous pixi shell that's no longer reproducible here).
- **What was done instead:** Verified compile-correctness via the parent build chain — MCP `build` invokes `cmake --build --preset release` which rebuilt `cucascade/CMakeFiles/cucascade_objects.dir/src/data/representation_converter.cpp.o` (step `[91/112]`) and re-linked `cucascade/libcucascade.a` (step `[92/112]`) successfully against my edit. This proves the cucascade source compiles clean under the production toolchain. Runtime behavior of the cucascade unit-test suite under my edit is therefore not directly verified in Plan 22-03; the empirical CC-04 verification falls to Plan 22-04 (where a fresh full Sirius build runs the sanitizer micro-validation, which exercises the same code path in a real query).
- **Files modified:** none
- **Commit:** none

**2. [Rule 4 — Architectural] Task 2 sanitizer micro-validation deferred to Plan 22-04 due to parallel-wave parent-build incompleteness**

- **Found during:** Task 2 step 1 (build prep)
- **Issue:** Per the parallel-execution context, Plan 22-01 is editing the Sirius parent's `pinned_entry` struct header concurrently. At the time Plan 22-03 reaches Task 2, the parent build is structurally broken (3 compile errors in sirius_scan_manager.cpp + sirius_extension.cpp from the in-progress rename). No fresh `sirius_unittest` binary can be linked against my bumped cucascade until 22-01 completes its `.cpp` updates.
- **What was done instead:** DEFERRED Task 2 to Plan 22-04 with a verbatim copy of the required command shape and a clear set of acceptance criteria for that plan. The plan itself anticipated this kind of sanitizer outcome with explicit handling for `CLUSTER_B > 0` (mark PARTIAL/escalate); deferring the run to Plan 22-04 is a structural variant of that — gate untriggered rather than failed. The cucascade fix is committed and verified compile-clean; the empirical race-count gate is the only piece that requires the post-Wave-1 integrated parent build.
- **Files modified:** none
- **Commit:** none
- **Risk:** If the same-stream invariant fix introduces a new race shape (e.g., due to the subtlety in Open Q1: does `cudaMemcpyAsync(..., DtoH, target_stream)` from a `cuda_set_device_raii(src_device)` scope produce a sanitizer-clean stream-ordered copy when target_stream lives on a different device?), Plan 22-04 will be the first plan to detect it. If detected, escalation to fix shape 2 (event-bridge) per D-07's allowed alternatives is required before the submodule pin advances.

**3. [Rule 3 — Blocking] clang-format reformatted the inserted DtoH cudaMemcpyAsync line back to the canonical 2-line form**

- **Found during:** Task 1 step 4 (`clang-format -i` on `src/data/representation_converter.cpp`)
- **Issue:** My initial Edit produced a multi-line wrapping that the cucascade clang-format style (WebKit + 100-char column) preferred to write differently. The single-line grep pattern `cudaMemcpyAsync\(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, target_stream\.value\(\)\)` defined in the plan's `<verify>` block matched 0 occurrences against the multi-line wrapped form.
- **What was done instead:** Ran clang-format with the cucascade `.clang-format` style; it reformatted the DtoH line back to the single-line form with the `CUCASCADE_CUDA_TRY(` / `host_buf, src_ptr, ...` continuation pattern. Post-format, the plan's verbatim grep pattern matches exactly 1 line (line 621). All 8 grep-gate acceptance criteria pass.
- **Files modified:** `cucascade/src/data/representation_converter.cpp` (in-place formatting; included in Task 1 commit)
- **Commit:** `c666b21` (Task 1 commit; format change merged with the logic change)
- **Note:** Project-mandated clang-format v20.1.4 was not directly available; v21.1.8 was used (minor version drift). The output exit-0 cleanly under the existing `.clang-format` style; no surprises.

## Implementation invariants verified

- **D-07 (same-stream invariant):** ✓ — `target_stream` is the single CUDA-stream argument shared by `rmm::device_buffer` ctor (line 600), DtoH `cudaMemcpyAsync` (line 621), and HtoD `cudaMemcpyAsync` (line 629). The local `rmm::cuda_stream src_stream` is gone.
- **D-08 (cucascade fork commit, no upstream PR):** ✓ — committed on `fix/pinned-portable-flags` branch (local fork); did NOT push to `origin` (NVIDIA/cuCascade); the `felipe` remote (felipeblazing/cuCascade_fork) was also not pushed to. Per CC-UPSTREAM-01, only the local pin advance is in scope; upstream PR is deferred.
- **D-15 / HYG-02 (no `rmm::cuda_stream_default` introduced):** ✓ — `grep -c "rmm::cuda_stream_default" cucascade/src/data/representation_converter.cpp` returns 0.
- **Pitfall 4 (preserve sync-then-cudaFreeHost ordering):** ✓ — `cudaStreamSynchronize(target_stream.value())` executes inside the src_guard scope before `cudaFreeHost(host_buf)`; the second `cudaStreamSynchronize(target_stream.value())` at the function tail (before `cudaFreeHost`) is preserved verbatim. Total `cudaStreamSynchronize(target_stream.value())` count = 2.
- **cucascade conventions (CUCASCADE_CUDA_TRY for CUDA calls):** ✓ — both legs and the sync are wrapped in `CUCASCADE_CUDA_TRY(...)`.
- **clang-format clean:** ✓ — `clang-format --dry-run --Werror -style=file src/data/representation_converter.cpp` exits 0.

## Pre-commit hooks

The cucascade `pre-commit run -a` invocation requires the cucascade pixi env (`cuda-12-nightly`) which failed to solve in this read-only-cache sandbox session (`Read-only file system (os error 30)` on the rattler cache directory). Equivalent guarantees were obtained piecewise:

- **clang-format:** ran from `/home/felipe/sirius/.pixi/envs/default/bin/clang-format` (v21.1.8); `--dry-run --Werror` exited 0 after applying the fix.
- **codespell:** the inserted comments use only English words present in standard project vocabulary; no new identifiers introduced. Manual review confirms no codespell triggers (`Phase`, `Cluster`, `D-07`, `target_stream`, `host_buf` — all already in the codebase).
- **cmake-format / cmake-lint:** no `CMakeLists.txt` changes.
- **black:** no Python changes.
- **trailing-whitespace / end-of-file-fixer:** verified by Edit tool's output (no trailing spaces; file ends with newline).

The cucascade pre-commit hooks are normally enforced via the cucascade local `.pre-commit-config.yaml`. In this parallel-execution context, the orchestrator validates hooks once after the wave completes (per the parallel-execution doctrine), and Plan 22-04's submodule-pin-bump validation is the natural site to re-run them.

## Output for downstream plans

**For Plan 22-04 (submodule pin bump):**
- Cucascade SHA to advance the gitlink to: `c666b21926dec70b26a1febd509435635bea8deb`
- Verify ancestry to `1c1e648` before bumping: `git -C cucascade merge-base --is-ancestor 1c1e648 c666b21` (must exit 0)
- After bumping, run the verbatim Task 2 sanitizer command (transcribed in the "Sanitizer micro-validation" section above) and record `CLUSTER_B == 0` as the gate.
- If `CLUSTER_B > 0` post-bump, STOP and escalate to fix shape 2 (event-bridge) per CONTEXT.md D-07.

**For Plan 22-07 (22-CUCASCADE-DIFF.md, advisory verdict aggregation):**
- The cucascade-side diff to track for upstreaming is the single commit `c666b21` (parent `42a01c4`).
- File scope: `cucascade/src/data/representation_converter.cpp` only; +11 / −3 lines; logic change concentrated in `alloc_and_peer_copy_async`'s host-staging fallback block (lines 610-627 post-fix).
- Upstream PR target: a future `feat(stream-lineage): same-stream invariant for host-staging fallback` upstream PR; per D-08 / CC-UPSTREAM-01 not in scope this milestone.

## Self-Check: PASSED

- File `/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/cucascade/src/data/representation_converter.cpp` exists (FOUND).
- Cucascade commit `c666b21` exists in cucascade submodule (FOUND via `git -C cucascade log --oneline -1`).
- All 8 grep-gate acceptance criteria for Task 1 verified in this SUMMARY's "Acceptance criteria status" table.
- File `/home/felipe/sirius/.worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/.planning/phases/22-multi-gpu-pinning-stream-lineage-hardening/22-03-SUMMARY.md` exists (this file; FOUND post-Write).
- Task 2 deferral to Plan 22-04 is documented with explicit rationale, sanitizer command shape, and acceptance criteria handoff.
