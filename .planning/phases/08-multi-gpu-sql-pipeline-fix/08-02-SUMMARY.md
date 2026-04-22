---
phase: 08-multi-gpu-sql-pipeline-fix
plan: 02
subsystem: data-converters

tags: [cuda, rmm, cucascade, multi-gpu, cudf, converter-registry, host-to-gpu]

# Dependency graph
requires:
  - phase: 08-multi-gpu-sql-pipeline-fix
    plan: 01
    provides: "FIX-01 per-GPU stream pool map in duckdb_scan_executor closing Site A; probe exit-criterion baseline"
  - phase: 07-p2p-direct-transfer-adaptive-scan-partitioning
    provides: "Pattern 2 (sirius-side converter override) reference body in src/data/sirius_p2p_converter.cpp + sirius_converter_registry::initialize() wiring shape"
provides:
  - "Sirius-side host_data_representation -> gpu_table_representation converter override (sirius_host_fast_to_gpu_factory) acquiring a target-bound stream and issuing H2D under rmm::cuda_set_device_raii target-device guard"
  - "Public-API-only column-tree reconstruction helper (reconstruct_column_target_stream) mirroring cucascade's reconstruct_column using only host_table.hpp + cudf column factories — consumable by any future Sirius override that must reconstruct a column tree without depending on cucascade-internal helpers"
  - "Probe verdict: FAIL (Branch B selected); post-override re-probe reveals distinct fix-site (host_parquet_representation path) handed off to 08-06"
affects:
  - 08-03-test-02
  - 08-04-audit
  - 08-05
  - 08-06-ship-gate

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern 2 extended from gpu->gpu (P2P) to host->gpu: acquire target-bound stream from target_memory_space->acquire_stream(), enter rmm::cuda_set_device_raii for target_device_id, issue every H2D copy on that stream, sync before returning"
    - "Public-API-only column-tree reconstruction (mirror cucascade's private reconstruct_column using only host_table.hpp + cpu_data_representation.hpp + cudf::make_{strings,lists,dictionary}_column + std::make_unique<cudf::column>)"
    - "Null-mask H2D-then-sync-before-factory-call pattern (cudf column factories read null masks at construction time, so the H2D must complete before the factory is invoked — matches cucascade's alloc_and_copy_h2d_sync approach)"
    - "STRING/LIST INT32-offset post-H2D cast to INT64 (cudf strings_column_view requires INT64 offsets for large-string support; cucascade does this via batch.flush + cudf::cast and we mirror the shape)"

key-files:
  created:
    - ".planning/phases/08-multi-gpu-sql-pipeline-fix/08-02-PROBE.md (updated — FAIL verdict + post-override re-probe)"
    - "src/include/data/sirius_host_to_gpu_converter.hpp (NEW — factory declaration)"
    - "src/data/sirius_host_to_gpu_converter.cpp (NEW — 237 lines; factory body + public-API-only reconstruct_column_target_stream)"
  modified:
    - "src/include/data/sirius_converter_registry.hpp (add cucascade/cpu_data_representation.hpp include + data/sirius_host_to_gpu_converter.hpp include; register override after MGPU-06 block inside converter_registry::initialize())"
    - "CMakeLists.txt (add src/data/sirius_host_to_gpu_converter.cpp to EXTENSION_SOURCES, alphabetically between host_parquet_representation_converters.cpp and sirius_p2p_converter.cpp)"

key-decisions:
  - "Used public-API-only reconstruction rather than depending on cucascade's private reconstruct_column/BatchCopyAccumulator symbols. Rationale: cucascade submodule pin (f47de0b) must stay unchanged per v1.2 scope; the private helpers are not exposed in cucascade/include/. The reimplemented helper is ~120 lines and mirrors cucascade's structure faithfully (same cudf factories, same null-mask-sync-first pattern, same INT32->INT64 offset cast for STRING/LIST)."
  - "Used individual cudaMemcpyAsync per block rather than cudaMemcpyBatchAsync. Rationale: cudaMemcpyBatchAsync (CUDA 12.8+) is the optimization cucascade uses internally, but it requires a dedicated BatchCopyAccumulator struct. Since we are reimplementing the helper anyway and the driver-call overhead is not on the critical hot path for correctness-first v1.2 work, individual async calls keep the code simpler. If profiling later shows driver overhead is measurable, a future plan can add BatchCopyAccumulator-equivalent batching."
  - "Registered host->gpu override AFTER the MGPU-06 P2P block inside converter_registry::initialize(). Rationale: unregister+register pattern matches Plan 07-02's MGPU-06 shape exactly; initialize() is the single entry point; no sirius_extension.cpp edit needed because LoadInternal at sirius_extension.cpp:1053 already calls initialize()."
  - "Did NOT extend the fix to convert_host_parquet_to_gpu_with_prefetched_data_source (Sirius-owned parquet converter) even though the post-override re-probe surfaced the SAME bug shape there. Rationale: the plan 08-02's files_modified and acceptance_criteria explicitly target host_data_representation -> gpu_table_representation only; the parquet-representation path is a distinct fix-site discovered during verification and explicitly deferred to 08-06 per <resume_instructions> step 3."

patterns-established:
  - "Pattern: any Sirius-owned host->gpu converter that consumes a CALLER-supplied stream MUST acquire its own target-bound stream from target_memory_space->acquire_stream() under rmm::cuda_set_device_raii{target_device_id} before issuing H2D copies. The caller's stream may be bound to a non-target device under num_gpus >= 2."
  - "Pattern: when cucascade exposes a CONVERTER but its internal helper symbols (reconstruct_column, BatchCopyAccumulator) are file-private, a Sirius-side override can reimplement the helper using ONLY public host_table.hpp + cudf factories — this is the minimum-intrusion approach that preserves the submodule pin."
  - "Pattern: null masks must be H2D-copied AND synced before passing them to cudf column factories, because factories read the null mask at construction (e.g. to compute null counts). Other buffers (data, offsets) can stream-submit because cudf column construction takes unique_ptr ownership and defers reads to actual kernel execution."

requirements-completed: [FIX-02]

# Metrics
duration: 15min
completed: 2026-04-21
---

# Phase 08 Plan 02: Sirius Host-to-GPU Converter Override (FIX-02 Branch B) Summary

**Authored a Sirius-side `host_data_representation -> gpu_table_representation` converter override (`sirius_host_fast_to_gpu_factory`) acquiring a target-bound stream under target-device RAII, closing the v1.1 cross-device stream-correctness bug shape on the host->gpu converter frame — and surfacing a distinct fix-site (`host_parquet_representation` path via Sirius's OWN parquet converter) for handoff to 08-06.**

## Performance

- **Duration:** ~15 min (wall clock)
- **Started:** 2026-04-21 (continuation agent)
- **Tasks:** 3 (PROBE update, Branch B authoring + build-gate, post-override re-probe + docs)
- **Files created/modified:** 2 new (hpp + cpp) + 3 modified (registry, CMakeLists, PROBE)

## Accomplishments

- **FIX-02 Branch B authored.** Sirius-side `host_data_representation -> gpu_table_representation` converter (`sirius_host_fast_to_gpu_factory`) mirroring Pattern 2's shape from `src/data/sirius_p2p_converter.cpp` but for host->gpu direction. Uses `cudaMemcpyHostToDevice` in place of `cudaMemcpyPeerAsync`; acquires target-bound stream from `target_memory_space->acquire_stream()` under `rmm::cuda_set_device_raii{target_device_id}`; syncs target_stream before returning.
- **Public-API-only column reconstruction.** Reimplemented cucascade's private `reconstruct_column` helper inside the Sirius .cpp using only `host_table.hpp` + `cpu_data_representation.hpp` + `gpu_data_representation.hpp` + cudf column factories. Handles the same cases as cucascade: STRING/LIST with INT32->INT64 offset cast, STRUCT via direct ctor, DICTIONARY32 via `cudf::make_dictionary_column`, DECIMAL with scale propagation, fixed-width leaf. Preserves cucascade pin (`f47de0b`) unchanged.
- **Registered cleanly at `converter_registry::initialize()`** immediately after the MGPU-06 P2P unregister+register block, using the exact same pattern. No `sirius_extension.cpp` edit needed — `LoadInternal` at `sirius_extension.cpp:1053` already calls `converter_registry::initialize()`.
- **Probe verdict recorded authoritatively.** `08-02-PROBE.md` updated from DEFERRED to FAIL with the full MCP-driven re-probe transcript (316 tests, 1 fail, `cudaErrorInvalidValue at cuda_memcpy.cu`), plus a post-override "Branch B did NOT close the hive-partition failure" section explaining that the failing test is actually routed through Sirius's OWN `host_parquet_representation` converter (distinct fix-site, handed off to 08-06).
- **Build + regression gate green.** `mcp__project-commands__run_command build` exits 0 (33s). `mcp__project-commands__run_command unit-tests` on default `num_gpus=1` config passes all 979 test cases (78,789,857 assertions; 224s). HYG-02 baseline preserved — `grep -rn 'rmm::cuda_stream_default' src/` still returns 41 matches across 12 files; new files contain 0.
- **Probe document is now self-consistent for 08-06 reviewer:** original DEFERRED note retained for history, new FAIL verdict and Branch B decision documented, post-override re-probe documented, distinct-fix-site handoff documented.

## Probe Results

### Original probe attempt (sandboxed bash, no GPU access)

Attempted earlier by a previous agent; returned `Probe DEFERRED — single-GPU host`. Retained in `08-02-PROBE.md` under "Hardware Availability" for history. Correctly reflected the sandbox's lack of GPU driver access.

### Orchestrator re-probe via MCP (real GPU access, 2026-04-21)

Via `mcp__project-commands__run_command unit-tests` with `integration.yaml` temporarily flipped to `num_gpus: 2`:

- **Exit code:** 1
- **Tests:** 316 run, 315 passed, 1 FAILED
- **Failing test:** `gpu_execution hive partition - filter on data column` at `test/cpp/integration/test_gpu_execution_multi_format.cpp:815`
- **Error:** `cudaErrorInvalidValue invalid argument` at `/tmp/conda-bld-output/.../cuda_memcpy.cu:42` — exact v1.1 bug signature.

### Post-override re-probe (after Branch B landed, commit `96481df`)

Same `num_gpus: 2` configuration, same command, after Branch B's override registered:

- **Exit code:** 1
- **Tests:** 316 run, 315 passed, 1 FAILED — **same test, same signature**
- **Interpretation:** Branch B closed the `host_data_representation` path but the failing test exercises `host_parquet_representation` — distinct converter, same bug shape, distinct fix-site. See "Deviations / Open Handoff" below.

## Task Commits

| Task                                                                       | Commit    | Type  |
| -------------------------------------------------------------------------- | --------- | ----- |
| Record FIX-02 probe FAIL + select Branch B (PROBE.md update)              | `46b933f` | docs  |
| Author Sirius host->gpu converter override (Branch B implementation)      | `96481df` | feat  |
| Record Branch B post-override re-probe + surface distinct fix-site        | `fae1915` | docs  |

**Plan metadata commit:** pending final commit after SUMMARY.md + STATE.md + ROADMAP.md updates.

## Files Created/Modified

### Created

- **`src/include/data/sirius_host_to_gpu_converter.hpp`** (NEW — 73 lines). Factory declaration `sirius_host_fast_to_gpu_factory` in `namespace sirius::data`. Docstring explains the v1.1 bug shape being closed and references PROBE.md for reproduction. Pattern-2 idiom reference for future readers.
- **`src/data/sirius_host_to_gpu_converter.cpp`** (NEW — 237 lines). Factory body + anonymous-namespace helper `reconstruct_column_target_stream` that mirrors cucascade's private `reconstruct_column` using only public `host_table.hpp` + cudf factories. Factory body: sync caller stream -> enter target RAII -> acquire target-bound stream -> reconstruct columns on target_stream -> assemble cudf::table -> sync target_stream -> return `cucascade::gpu_table_representation`. Hygiene: sticky `cudaGetLastError()` consume before returning.

### Modified

- **`src/include/data/sirius_converter_registry.hpp`** (+26 lines). Added `#include <cucascade/data/cpu_data_representation.hpp>` + `#include <data/sirius_host_to_gpu_converter.hpp>`; inside `initialize()` after the MGPU-06 P2P block, added the unregister+register pair for `host_data_representation -> gpu_table_representation`. Pattern matches MGPU-06 exactly.
- **`CMakeLists.txt`** (+1 line). Added `src/data/sirius_host_to_gpu_converter.cpp` to `EXTENSION_SOURCES` alphabetically between `host_parquet_representation_converters.cpp` and `sirius_p2p_converter.cpp`.
- **`.planning/phases/08-multi-gpu-sql-pipeline-fix/08-02-PROBE.md`** (+162 lines). Updated verdict from DEFERRED to FAIL; added "Probe re-run on 2026-04-21 (orchestrator)" section with MCP-driven transcript; added "Branch Decision (updated)" selecting Branch B; added "Post-Override Re-Probe" section documenting that the SAME failing test is actually on a distinct fix-site (`host_parquet_representation` via Sirius's own converter); added handoff for 08-06 with the canonical template reference.

## Static Invariants (all green)

| Check                                                                                                                            | Result                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `sirius_host_fast_to_gpu_factory` declared in hpp                                                                               | 1 match (factory declaration)                                                                       |
| `sirius_host_fast_to_gpu_factory` defined in cpp                                                                                | 1 match (factory body)                                                                              |
| `register_converter<cucascade::host_data_representation, cucascade::gpu_table_representation>` in registry                      | 1 match (inside `initialize()` after MGPU-06 block)                                                 |
| `sirius_host_to_gpu_converter.cpp` in CMakeLists.txt                                                                            | 1 match (in `EXTENSION_SOURCES`)                                                                    |
| `rmm::cuda_stream_default` in new files (hpp + cpp)                                                                             | 0 matches (HYG-02 preserved)                                                                        |
| `rmm::cuda_stream_default` baseline across `src/` (context check)                                                               | 41 matches across 12 files — unchanged from 08-01 SUMMARY                                          |
| `rmm::cuda_set_device_raii` used for target_device_id in converter body                                                         | 1 match (target_guard at factory entry)                                                             |
| `target_memory_space->acquire_stream()` consumed (not caller stream) for H2D                                                    | 1 match (target_stream captured, passed through reconstruct_column_target_stream and final sync)    |
| `cudaMemcpyHostToDevice` used for H2D (not `cudaMemcpyPeerAsync`)                                                               | 1 match (in `alloc_and_copy_h2d`)                                                                   |
| Sticky `cudaGetLastError()` consume after cuda* call                                                                            | 2 matches (in copy error path + before returning from factory)                                      |
| MCP `build`                                                                                                                     | exit 0 (33s)                                                                                        |
| MCP `unit-tests` (num_gpus=1, default)                                                                                          | exit 0, 979 test cases, 78,789,857 assertions, 0 failures                                           |
| MCP `unit-tests` (num_gpus=2, post-Branch-B)                                                                                    | exit 1, 316 run, 1 failed (hive partition — distinct fix-site; see "Deviations")                   |
| cucascade submodule clean                                                                                                        | `git status cucascade/` shows no changes                                                            |
| `test/cpp/integration/integration.yaml` reverted to num_gpus=1                                                                  | `git diff test/cpp/integration/integration.yaml` is empty                                           |

## Decisions Made

- **Public-API-only column reconstruction** over depending on cucascade's private helpers. `BatchCopyAccumulator` and `reconstruct_column` are defined in `cucascade/src/data/representation_converter.cpp` (not in `cucascade/include/`). Exposing them would require a cucascade submodule change, which v1.2 scope forbids. Reimplementing them in Sirius is ~120 lines and stays faithful to cucascade's structure.
- **Per-block `cudaMemcpyAsync` over `cudaMemcpyBatchAsync`.** Cucascade uses the batched API (CUDA 12.8+) for driver-call-overhead optimization. Our override uses individual async calls because (a) the code is 5 lines simpler without the batch accumulator struct, (b) correctness-first work dominates v1.2, (c) if profiling shows overhead matters, a future plan can add batch accumulation without touching the interface.
- **Register INSIDE `converter_registry::initialize()` (not at a new `LoadInternal` call site).** Pattern from MGPU-06 (Plan 07-02) — keeps all overrides co-located, no second extension-entry-point edit, same unregister+register idiom.
- **Do NOT extend the fix to `convert_host_parquet_to_gpu_with_prefetched_data_source`** even though the post-override re-probe surfaced the same bug shape there. The plan 08-02's scope explicitly targets only `host_data_representation -> gpu_table_representation`; extending it in-plan would violate the plan's files_modified contract and the `<resume_instructions>` step 3 ("if a different fix-site surfaces, document but do NOT chase it here"). Documented the distinct fix-site in `08-02-PROBE.md` with a full handoff to 08-06 including the canonical template reference.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Corrected header path convention during build**

- **Found during:** Task 3, first `mcp__project-commands__run_command build` invocation.
- **Issue:** Initially placed header at `src/data/sirius_host_to_gpu_converter.hpp`, mirroring the .cpp's directory. The Sirius include convention (verified via `sirius_p2p_converter.hpp` and all other `src/include/data/*.hpp`) puts public converter headers under `src/include/data/`, so the `-I src/include` include root resolves `<data/sirius_host_to_gpu_converter.hpp>`. The first build failed with `fatal error: data/sirius_host_to_gpu_converter.hpp: No such file or directory`.
- **Fix:** `mv src/data/sirius_host_to_gpu_converter.hpp src/include/data/sirius_host_to_gpu_converter.hpp`. No code changes.
- **Files modified:** N/A (just a filesystem move before commit).
- **Verification:** Second MCP build exits 0.
- **Committed in:** `96481df` (the header is committed to the correct path).

### Open Handoff — Distinct Fix-Site Discovered During Verification

**[NOT a deviation; plan-scoped handoff]** The post-override re-probe showed the SAME test still failing with the SAME error signature. Analysis: the failing test `gpu_execution hive partition - filter on data column` routes through Sirius's OWN converter `convert_host_parquet_to_gpu_with_prefetched_data_source` at `src/data/host_parquet_representation_converters.cpp:55-119`, which has the SAME bug shape as cucascade's `convert_host_fast_to_gpu`:

- Line 66: opens `rmm::cuda_set_device_raii target_device_raii(target_device_id)` (correct).
- Line 92: calls `cudf::io::read_parquet(opts, stream, mr_ref)` with CALLER's stream (same mismatch under num_gpus=2).

This is a DISTINCT fix-site (different representation: `host_parquet_representation`, not `host_data_representation`). Per plan 08-02's `<resume_instructions>` step 3, it is documented but NOT chased in this plan. Handoff recorded in `08-02-PROBE.md` under "Post-Override Re-Probe"; 08-06 (or a future FIX-02-extension plan) applies the same acquire-target-bound-stream + RAII-on-target pattern to that converter, using `src/data/sirius_host_to_gpu_converter.cpp` as the canonical template.

### Scope-Preserved Site Coverage

Branch B discharges FIX-02 on the `host_data_representation -> gpu_table_representation` path specifically (Site C from the static audit in `08-02-PROBE.md`). The plan 08-02's acceptance criteria are ALL satisfied:

- File `src/data/sirius_host_to_gpu_converter.cpp` exists with `sirius_host_fast_to_gpu_factory` body — YES.
- File contains `rmm::cuda_set_device_raii` usage and does NOT contain `rmm::cuda_stream_default` — YES.
- Registry contains `register_converter<cucascade::host_data_representation, cucascade::gpu_table_representation>` match — YES.
- CMakeLists.txt contains `sirius_host_to_gpu_converter.cpp` match — YES.
- MCP build exits 0 — YES.
- `08-02-PROBE.md` contains "Post-Override Re-Probe" section — YES.

---

**Total deviations:** 1 auto-fixed (blocking header-path correction) + 1 plan-scoped handoff (distinct fix-site documented for 08-06).
**Impact on plan:** Branch B code lands clean; plan 08-02 success criteria all met within scope. 08-06 ship gate now has a precise handoff to close the remaining host_parquet_representation path.

## Issues Encountered

- **Initial header-path mis-placement.** Put hpp next to cpp instead of under `src/include/data/`. Caught by the first MCP build's fatal include error. Fixed by moving the file; no code changes; zero additional compile cycles wasted (the first build's early-fail reproduced the problem immediately).
- **Branch B's target representation doesn't match the failing test's path.** The static audit in `08-02-PROBE.md` correctly identified Site C (cucascade's `convert_host_fast_to_gpu` on `host_data_representation` path) as the candidate, but the MCP-driven runtime re-probe exposed that the hive-partition test actually fails on `host_parquet_representation`. Both sites share the same bug shape. Branch B was authored per the plan's scope; the second site is handed off per the plan's explicit `<resume_instructions>` guidance.
- **cucascade's submodule has an independent GSD workflow (cucascade/CLAUDE.md).** The system-reminder surfaced cucascade's project rules during the read of its headers. Verified that our work stays fully on Sirius side — `git status cucascade/` is clean — so those rules are informational only for this plan.

## Known Stubs

None. The factory is fully implemented with all cudf column types handled (STRING, LIST, STRUCT, DICTIONARY32, DECIMAL32/64/128, fixed-width leaf). The only "unfinished" work is the distinct fix-site on `host_parquet_representation` which is explicitly plan-scoped as an 08-06 handoff, not a stub in this plan's code.

## Next Phase Readiness

- **FIX-02 Branch B build-gated + regression-tested complete on `host_data_representation` scope.** Single-GPU unit-tests suite (979 tests) passes without regression.
- **08-03 (TEST-02 parameterization)** is unblocked. It can now assume a working Sirius host->gpu override is registered for `host_data_representation`; when 08-03 writes the parameterized 2-GPU TPC-H test, any test that flows through this representation will use Branch B's code.
- **08-04 (AUDIT)** has a second worked example of Pattern 2 beyond `sirius_p2p_converter.cpp`. The `src/data/sirius_host_to_gpu_converter.cpp` implementation is the canonical template for any future converter that must acquire a target-bound stream over a caller-supplied one.
- **08-06 (SF100 ship gate)** has an explicit handoff written in `08-02-PROBE.md`: apply the same target-bound-stream pattern to `convert_host_parquet_to_gpu_with_prefetched_data_source` in `src/data/host_parquet_representation_converters.cpp`. Template file: `src/data/sirius_host_to_gpu_converter.cpp`.
- **No new blockers introduced.** The cucascade pin stays unchanged (`f47de0b`). HYG-02 baseline preserved. The distinct fix-site on the Sirius parquet converter is a pre-existing bug now explicitly documented for closure in 08-06.

## Self-Check: PASSED

**Files verified to exist:**

- FOUND: `.planning/phases/08-multi-gpu-sql-pipeline-fix/08-02-PROBE.md` (updated to FAIL verdict with post-override re-probe section)
- FOUND: `src/include/data/sirius_host_to_gpu_converter.hpp` (new — factory declaration)
- FOUND: `src/data/sirius_host_to_gpu_converter.cpp` (new — factory body + reconstruct_column_target_stream)
- FOUND: `src/include/data/sirius_converter_registry.hpp` (modified — host->gpu override registered after MGPU-06 block)
- FOUND: `CMakeLists.txt` (modified — new .cpp added to EXTENSION_SOURCES)

**Commits verified to exist:**

- FOUND: `46b933f` (docs — PROBE.md FAIL verdict + Branch B selection)
- FOUND: `96481df` (feat — Branch B implementation: hpp + cpp + registry + CMakeLists)
- FOUND: `fae1915` (docs — post-override re-probe + distinct-fix-site handoff)

**Grep invariants verified:**

- `sirius_host_fast_to_gpu_factory` in src/: 2 matches (declaration + definition) — required >= 2
- `register_converter<.*host_data_representation.*gpu_table_representation>` in registry: 1 match — required >= 1
- `sirius_host_to_gpu_converter.cpp` in CMakeLists.txt: 1 match — required >= 1
- `rmm::cuda_stream_default` in new hpp + cpp: 0 matches — HYG-02 preserved
- `rmm::cuda_stream_default` across src/: 41 matches across 12 files — unchanged from 08-01 baseline
- cucascade submodule: clean (git status cucascade/ empty)

---
*Phase: 08-multi-gpu-sql-pipeline-fix*
*Completed: 2026-04-21*
