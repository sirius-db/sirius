---
phase: 19-io-framework-adoption-pr-675
type: phase-verdict
status: PASS
verified: 2026-05-06
requirements_closed: [IO-12, IO-13, IO-14, IO-15, IO-16, IO-17]
---

# Phase 19 Verdict — IO Framework Adoption (PR #675)

**Status: PASS — all 6 IO-12..17 requirements closed.**

Phase 19 source migration (plans 19-01..19-05) replaced `sirius::io::cucascade_datasource` with `sirius::io::sirius_datasource` and stood up per-GPU `uring_ioctx` instances in `SiriusContext`. Plan 19-06 (this verdict) is the verification gauntlet covering the IO-17 ship-gate (`[TPC-H][parquet]` 22/22 + compute-sanitizer memcheck) and the IO-14 multi-GPU empirical PCIe probe.

## Section A — Functional Verification (Plan 19-06 Task 1)

### Grep Gauntlet (IO-12, IO-15, IO-16, HYG-02)

| Requirement | Criterion | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- | --- |
| IO-12 | liburing wired via pkg-config in pixi env | `pixi run pkg-config --modversion liburing` | non-empty | `2.14` | **PASS** |
| IO-15 | `cucascade_datasource` retired | `grep -rn "cucascade_datasource" src/ test/ \| wc -l` | 0 | 0 | **PASS** |
| IO-15 | `cucascade::idisk_io_backend` retired in src/ | `grep -rn "cucascade::idisk_io_backend" src/ \| wc -l` | 0 | 0 | **PASS** |
| IO-15 | `cucascade::io_backend_registry` / `register_builtin_io_backends` retired | `grep -rn "cucascade::io_backend_registry\|register_builtin_io_backends" src/ \| wc -l` | 0 | 0 | **PASS** |
| IO-15 | Old SiriusContext machinery retired | `grep -rn "gpu_io_backends_\|get_io_backend_for\|get_gpu_io_backends" src/ \| wc -l` | 0 | 0 | **PASS** |
| IO-16 | Raw `cudaSetDevice` in `src/io/` | `grep -rn "cudaSetDevice\b" src/io/ \| wc -l` | 0 | 0 | **PASS** |
| HYG-02 | `rmm::cuda_stream_default` baseline preserved | `grep -rc "rmm::cuda_stream_default" src/ \| awk -F: '{s+=$2} END {print s}'` | ≤ 40 | **40** | **PASS** |
| Migration witness | sirius IO surface live in src/ | `grep -rn "uring_ioctx\|sirius_ioctx\|sirius_datasource" src/ \| wc -l` | ≥ 50 | **107** | **PASS** |

### Build Gate (IO-17 prerequisite)

| Gate | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- |
| MCP build | `mcp__project-commands__run_command build` | exit 0 | exit 0 (0.2s incremental, no work to do) | **PASS** |

### Functional Test Gate (IO-17 functional leg)

| Gate | Command | Expected | Actual | Status |
| --- | --- | --- | --- | --- |
| `[TPC-H][parquet]` 22/22 (num_gpus=2) | `mcp__project-commands__run_command unit-tests --filter "[TPC-H][parquet]"` | 22/22 PASS | **22/22 PASS, 36256 assertions, 78.6s, exit 0** | **PASS** |
| `[multi_gpu_foundation]` smoke (canary) | `mcp__project-commands__run_command unit-tests --filter "[multi_gpu_foundation]"` | 7/7 PASS | **7/7 PASS, 38 assertions, 4.4s, exit 0** | **PASS** |

### Section A Per-Requirement Roll-up

| Req | Verdict | Evidence |
| --- | --- | --- |
| IO-12 | **PASS** | vcpkg.json declares liburing (line 17, confirmed in 19-01); pkg-config probes 2.14 in pixi env; CMakeLists.txt:71-72 + 322-325 wiring intact since Phase 17 merge. Zero source changes for IO-12. |
| IO-13 | **PASS** | `SiriusContext::initialize()` constructs ONE `sirius::io::uring_ioctx` per GPU memory space under `rmm::cuda_set_device_raii` (sirius_context.cpp post-19-04). `gpu_ioctxs_` map populated; `get_ioctx_for(int)` + `get_gpu_ioctxs()` accessors operational; consumers (parquet/iceberg scan, task_creator, sirius_engine) all flipped in 19-05. |
| IO-14 | **PASS (functional)** | Per-GPU CUDA-context binding end-to-end via Phase 9 two-tier preferred_device_id lookup carrying through to per-task `ioctx_it` lookup (parquet_scan_task.cpp post-19-05); `device_read_req.device_id` always matches the owning ioctx's GPU. Empirical multi-GPU PCIe probe in Section B. |
| IO-15 | **PASS** | `grep -rn "cucascade_datasource" src/ test/` returns 0 (down from 51 line hits / 6 files at 19-01 baseline). 3 files deleted: `src/include/io/cucascade_datasource.hpp`, `src/io/cucascade_datasource.cpp`, `test/cpp/io/test_cucascade_datasource.cpp`. All consumers flipped in 19-05. |
| IO-16 | **PASS** | `grep -rn "cudaSetDevice\b" src/io/` returns 0. 19-02 wrapped `uring_reactor.cpp:276` raw `cudaSetDevice` with `std::optional<rmm::cuda_set_device_raii>` under preserved `if (req.device_id >= 0)` guard. |
| IO-17 (functional) | **PASS** | `[TPC-H][parquet]` 22/22 PASS at num_gpus=2; 36256 assertions; 78.6s wall (well under 250s budget); exit 0. `[multi_gpu_foundation]` 7/7 PASS canary. Sanitizer leg in Section C. |
| HYG-02 baseline | **PASS** | `rmm::cuda_stream_default` count in src/ is 40 (matches baseline; entirely in src/legacy/ and src/include/legacy/). Zero new introductions from 19-02..19-05. |

**Section A Verdict: PASS** — all functional gates green. Empirical multi-GPU probe + sanitizer legs in Sections B and C.

## Section B — Empirical Multi-GPU PCIe Probe (Plan 19-06 Task 2 Step A; IO-14 empirical leg)

Detailed evidence: see `19-NVIDIA-SMI-DUAL-GPU.md`.

### Method

`nvidia-smi dmon -s pucvmet -i 0,1 -o T -d 1 -c 120` running in the background (120 samples × 1s = 120s) while a 102.5s `[mgpu]` integration workload (16/16 PASS, 79091 assertions) exercised the per-GPU `sirius_ioctx` + `sirius_datasource` path on 2 × NVIDIA RTX 6000 Ada Generation.

### Result Summary

| GPU | rxpci > 0 samples | Max rxpci | Cumulative rxpci over window | Verdict |
| --- | --- | --- | --- | --- |
| GPU 0 | **63 / 120** | **2892 MB/s** | 15141 MB | non-zero |
| GPU 1 | **54 / 120** | **453 MB/s** | 4273 MB | non-zero |

Both GPUs received non-zero PCIe read traffic across the workload window. Per-GPU `uring_ioctx` reactors are correctly driving distinct PCIe lanes — neither GPU was starved.

### IO-14 Empirical Verdict: **PASS**

- Pitfall 9 (lost Portable flags) NOT observed — GPU 1 rxpci is reliably non-zero (54 active samples; max 453 MB/s).
- Concurrent `sm` (compute %) on both GPUs in multiple samples confirms operators dispatched independently to each device.
- Distinct framebuffer growth (GPU 0 peak 639 MB; GPU 1 peak 681 MB) confirms separate device-side memory pools.

## Section C — compute-sanitizer memcheck (Plan 19-06 Task 2 Step B; IO-17 sanitizer leg)

Per project memory `feedback_sanitizer_via_bash_not_mcp.md`, run via Bash + timeout (NOT MCP).

### Leg 1: `[multi_gpu_foundation]`

```bash
timeout 600 /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  build/release/extension/sirius/test/cpp/sirius_unittest "[multi_gpu_foundation]" \
  > /tmp/p19_sanitizer_mgf.log 2>&1
```

| Metric | Result |
| --- | --- |
| Test result | **All tests passed (38 assertions in 7 test cases)** |
| Memcheck violations (Invalid __global__/shared__/local__ read/write, out-of-bounds, leaks, uninitialized) | **0** |
| CUDA API status returns reported by sanitizer's API tracer | 8 (all benign — see classification below) |
| Sanitizer runtime | within budget |

**Reported "errors" classification:**

| Count | Type | Source frame | Classification |
| --- | --- | --- | --- |
| 5 | `cudaErrorPeerAccessAlreadyEnabled (704)` on `cudaGetLastError` | `SiriusContext::initialize()` + TEST_CASE_16 | **Benign** — cucascade peer-access probe pattern from `tpch_q1_mgpu_string_bug` resolution; sticky-error-consume on `cudaGetLastError()` after the probe enables peer access (already enabled on the second iteration) |
| 2 | `cudaErrorInvalidDevice (101)` on `cudaSetDevice` | `sirius::parallel::downgrade_executor::start()` worker thread | **Pre-existing** — bounded_thread_pool worker init pattern. Pre-Phase-19 (no Phase 19 source in stack frame). Not in src/io/. |
| 1 | (Catch2 progress line concatenated to a 704 emission) | progress noise | **Benign** |

### Leg 2: `[integration][gpu_execution][parquet][join]`

```bash
timeout 1800 /usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 1 \
  build/release/extension/sirius/test/cpp/sirius_unittest "[integration][gpu_execution][parquet][join]" \
  > /tmp/p19_sanitizer_join.log 2>&1
```

| Metric | Result |
| --- | --- |
| Test result | **All tests passed (1922202 assertions in 42 test cases)** |
| Memcheck violations | **0** |
| CUDA API status returns | 9 (same benign pattern as Leg 1) |
| Sanitizer runtime | ~20s (1.6× baseline 12.6s) |

**Reported "errors" classification:**

| Count | Type | Source frame | Classification |
| --- | --- | --- | --- |
| 5 | `cudaErrorPeerAccessAlreadyEnabled (704)` on `cudaGetLastError` | `SiriusContext::initialize()` | Benign (same as Leg 1) |
| 3 | `cudaErrorInvalidDevice (101)` on `cudaSetDevice` | `downgrade_executor::start()` worker thread | Pre-existing (same as Leg 1) |
| 1 | Concatenated progress line | progress noise | Benign |

### Section C Verdict: **PASS**

Both legs produced **zero memcheck violations** (Phase 5/6 baseline preserved). All 8/9 reported items are CUDA API status reports from compute-sanitizer's API tracer — not memory access violations:

- The 5+5 `cudaErrorPeerAccessAlreadyEnabled` are the exact pattern documented in project memory: cucascade's startup peer-DMA probe enables peer access; the second pass observes "already enabled"; `cudaGetLastError()` consumes the sticky error. This is intentional + correct.
- The 2+3 `cudaErrorInvalidDevice` originate from the `bounded_thread_pool` / `downgrade_executor` worker thread (pre-Phase-19 code path; not in src/io/). Frame stack contains zero Phase 19 source frames.

**No new sanitizer-visible defects introduced by Phase 19 source migration.** Phase 5/6 baseline (0 errors / 1.92M assertions) preserved.

## Section D — Phase 19 Closing Verdict

**Phase 19 PASS — all 6 IO-12..17 closed.**

| Req | Verdict | Evidence | Reference |
| --- | --- | --- | --- |
| IO-12 | PASS | vcpkg.json line 17 declares liburing; pkg-config probes 2.14 in pixi env; CMakeLists.txt:71-72 + 322-325 wiring intact | 19-01-INVENTORY.md, Section A |
| IO-13 | PASS | Per-GPU `sirius::io::uring_ioctx` constructed in `SiriusContext::initialize()` under `rmm::cuda_set_device_raii`; per-GPU `admission_control` budgets (P5 mitigation); teardown precedes memory_manager shutdown (Pitfall 3) | 19-04-SUMMARY.md, Section A |
| IO-14 | PASS | (functional) Per-GPU CUDA-context binding end-to-end via Phase 9 two-tier preferred_device_id lookup; `device_read_req.device_id` always matches owning ioctx's GPU. (empirical) Both GPU 0 and GPU 1 received non-zero PCIe rxpci traffic during multi-GPU workload (63 + 54 active samples; max 2892 + 453 MB/s) | 19-04-SUMMARY.md, 19-05-SUMMARY.md, Section B, 19-NVIDIA-SMI-DUAL-GPU.md |
| IO-15 | PASS | `cucascade_datasource` retired; `grep -rn "cucascade_datasource" src/ test/` = 0 (down from 51); 3 files deleted; all consumers flipped to `sirius_datasource` via `ioctx->make_datasource(io_object)` factory | 19-05-SUMMARY.md, Section A |
| IO-16 | PASS | Raw `cudaSetDevice` in src/io/ = 0; `uring_reactor.cpp:276` wrapped in `std::optional<rmm::cuda_set_device_raii>` under preserved `device_id >= 0` guard | 19-02-SUMMARY.md, Section A |
| IO-17 | PASS | `[TPC-H][parquet]` 22/22 PASS at num_gpus=2 (36256 assertions, 78.6s); compute-sanitizer memcheck on `[multi_gpu_foundation]` (7/7) and `[integration][gpu_execution][parquet][join]` (42/42, 1.92M assertions) — both report **0 memcheck violations** | Section A, Section C |

**Supporting baselines preserved:**

- HYG-02: `rmm::cuda_stream_default` in src/ = **40** (entirely in src/legacy/, unchanged from 19-01 baseline)
- FSM regression: deleted-FSM symbols = 0 live (verified by Phase 17 D-G3 gate; Phase 19 added zero FSM hits)
- Build: `mcp__project-commands__run_command build` exit 0 (incremental — no work to do; clean state since 19-05)
- `[mgpu]` regression: 16/16 PASS, 79091 assertions, 102.5s (Section B run; matches 19-04/19-05 baseline)
- `[multi_gpu_foundation]` regression: 7/7 PASS, 38 assertions, 4.4s (Section A canary)

## Section E — Carryover to Phase 20+

Phase 19 closed all 6 IO-12..17 requirements but explicitly deferred two items per RESEARCH.md Open Questions:

| Item | Source | Defer to | Rationale |
| --- | --- | --- | --- |
| Author `test_sirius_datasource.cpp` mirroring the 7 deleted cucascade_datasource TEST_CASEs | 19-05-SUMMARY.md "Decisions Made" + RESEARCH.md "could split to Wave 4" | Phase 20 SM-XX or Phase 21 polishing | Phase 19 IO-17 ship-gate is `[TPC-H][parquet]` 22/22 + sanitizer cleanliness, not unit-test parity. Datasource is exercised end-to-end by the parquet TPC-H suite. |
| Enable `prefetching_cache` via `ioctx->initialize_cache()` | 19-04-SUMMARY.md Decision; RESEARCH.md Open Q2 | Phase 20+ | Cache enablement requires per-GPU `buffer_pool` ownership decision (CONTEXT.md anti-pattern: never share buffer_pool across ioctxs). v1.1 baseline correctness already met without prefetching cache; sirius_datasource device_read falls through to device_read_io when `_cache==nullptr`. |
| Re-attach `test_metadata_gpu_scan_operators.cpp` to `CMakeLists.txt TEST_SOURCES` | 19-05-SUMMARY.md (file's call sites already flipped to `make_test_ioctx()`) | Phase 20 SM-03 | `sirius_parquet_metadata_scan_operator.hpp` was deleted in Phase 17 merge; re-attached when SM-03 reintroduces metadata-scan operator. The 19-05 edits keep IO-15 grep gate clean and prepare the file for Phase 20 re-add. |
| Phase 20 dependency unblock | — | Phase 20 SM-01..SM-06 | Phase 19 closure makes `sirius_datasource` available to `parquet_split_provider::run_batch` (Scan Manager port). Phase 20 can now proceed. |

**No regressions identified.** Phase 18 P1 RAII lock-scope deadlock fix preserved; all multi-GPU correctness gates from Phases 6/8/9/10/12/13/14/15 remain green.

---

*Phase 19 verification gauntlet executed 2026-05-06 by plan 19-06.*
*All evidence reproducible from /tmp/p19_nvsmi_dmon.log + /tmp/p19_sanitizer_mgf.log + /tmp/p19_sanitizer_join.log.*

