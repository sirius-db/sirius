# Requirements: Sirius Multi-GPU v1.2 — Multi-GPU SQL Pipeline Fix

**Core Value:** Any SQL query — including TPC-H at any scale factor — returns correct results when `num_gpus: 2` is configured, with pipeline tasks distributed across both GPUs and `[mgpu-audit]` evidence showing work landing on each device.

---

## Milestone v1.2 Requirements (current)

**Defined:** 2026-04-21
**Goal:** Close the v1.1 gap exposed by post-ship e2e verification: cross-device stream-correctness bug in `pipeline::lock_or_prepare_batch` that throws `cudaErrorInvalidValue` on non-trivial SQL when `num_gpus >= 2`. Ship when TPC-H SF1 + SF10 pass on 2 GPUs with audit evidence of cross-GPU distribution.

### FIX — Cross-Device Stream-Correctness Bug

- [x] **FIX-01**: `pipelineable_operator_data::prepare_for_processing` → `pipeline::lock_or_prepare_batch` no longer throws `cudaErrorInvalidValue: invalid argument` when source and target devices differ. Pack on source-device RAII + source stream; copy on target stream (Pattern 2 — same shape as `src/data/sirius_p2p_converter.cpp` from Plan 07-02).
- [x] **FIX-02**: Audit every other cross-device CUDA memcpy call-site in `src/pipeline/` and `src/op/` for the same bug pattern; apply the same fix where present. Document surfaces covered.
- [x] **FIX-03**: Zero net-new `rmm::cuda_stream_default` uses (HYG discipline maintained).
- [x] **FIX-04**: Build clean on MCP (`mcp__project-commands__run_command build` exit 0) after fix.

### TEST — Integration Suite Multi-GPU Coverage

- [x] **TEST-01**: `test/cpp/integration/test_gpu_execution_tpch.cpp` parameterized on `num_gpus ∈ {1, 2}` (or equivalent — e.g., run both configs per TEST_CASE, or two TEST_CASE flavors per query).
- [x] **TEST-02**: `test/cpp/integration/integration.yaml` (or fixture config) supports `num_gpus: 2` at test runtime; no permanent flip of the default if that's too aggressive — but the 2-GPU variant MUST execute in the default `mcp__project-commands__run_command unit-tests` run.
- [x] **TEST-03**: All 22 TPC-H queries pass at SF1 on `num_gpus: 2` — results bitwise identical to the `num_gpus: 1` baseline.
- [x] **TEST-04**: TPC-H Q1, Q6, Q12 pass at SF10 on `num_gpus: 2` (smoke-test scale, matches v1.1 Phase-5 evidence set).

### AUDIT — `[mgpu-audit]` Acceptance Gate

- [x] **AUDIT-01**: A dedicated TEST_CASE (or check inside an existing TPC-H TEST_CASE) runs TPC-H SF1 on `num_gpus: 2` with `[mgpu-audit]` logging enabled, captures the log, and asserts `pipeline_task` count > 0 on **both** GPU 0 and GPU 1.
- [x] **AUDIT-02**: Same assertion for `scan_batch` count > 0 on both GPUs.
- [x] **AUDIT-03**: Audit gate is checked by the default `unit-tests` run — i.e., regressions to single-GPU-only distribution break the build.

---

## Out of Scope (v1.2)

| Feature | Reason |
|---------|--------|
| **Upstream cucascade `convert_gpu_to_gpu` PR** | Sirius override works; filing upstream is nice-to-have, not blocking multi-GPU SQL. Defer to v1.3 or a standalone upstream-contributions track. |
| Performance regression comparisons (Phase-5 vs Phase-4, Phase-6 vs Phase-5) | Per user directive 2026-04-21 ("let's just make sure everything is working, we can optimize later"). v1.2 is correctness-focused. |
| Cucascade `idisk_io_backend` file-handle cache | Upstream concern (research pitfall P1); not triggered by the v1.1 bug. |
| `cudaDeviceDisablePeerAccess` on explicit teardown | Current reliance on CUDA process-exit cleanup is adequate for correctness; revisit when teardown-coverage matters. |
| TPC-H Q4 parquet intermittent flake | Separate scoped investigation; not blocking v1.2. |
| Changes to Super Sirius dispatch architecture | v1.2 is a targeted fix, not a redesign. If the fix uncovers architectural gaps, surface as v1.3 scope. |
| Flipping default `num_gpus` to 2 for every integration fixture | Too aggressive for v1.2. Scope the flip to TPC-H parameterization first; other fixtures can follow if the pattern proves stable. |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-01 | Phase 8 | Complete |
| FIX-02 | Phase 8 | Complete |
| FIX-03 | Phase 8 | Complete |
| FIX-04 | Phase 8 | Complete |
| TEST-01 | Phase 8 | Complete |
| TEST-02 | Phase 8 | Complete |
| TEST-03 | Phase 8 | Complete |
| TEST-04 | Phase 8 | Complete |
| AUDIT-01 | Phase 8 | Complete |
| AUDIT-02 | Phase 8 | Complete |
| AUDIT-03 | Phase 8 | Complete |

**Coverage:**
- v1.2 requirements: 11 total (4 FIX + 4 TEST + 3 AUDIT)
- Mapped to phases: 11 / 11 (100%)
- Unmapped: 0

---

*v1.2 requirements defined: 2026-04-21*
*Traceability populated by roadmapper: 2026-04-21*
