# Phase 8: Multi-GPU SQL Pipeline Fix — Research

**Researched:** 2026-04-21
**Domain:** Multi-GPU cross-device stream-correctness in Sirius's scan → pipeline-task data path (cuDF / cucascade / RMM)
**Confidence:** HIGH (all fix sites, log sites, test sites, and the Pattern 2 reference are verified in the current working tree with exact line numbers)

## Summary

Phase 8 is a targeted, bounded fix. All upstream machinery from v1.1 (topology, per-GPU io_backends, bidirectional P2P, memory-proportional scan distribution, Sirius-side P2P converter override for GPU→GPU table conversion, `[mgpu-audit]` dispatch logs) is already shipped and operational. The v1.1 E2E verification (`.planning/milestones/v1.1-E2E-VERIFICATION.md`) confirmed that a trivial `SELECT count(*)` distributes across 2 GPUs correctly, but **any non-trivial SQL** — TPC-H Q1/Q6/Q12, filter+sort, aggregation — fails at `cuda_memcpy.cu:42` with `cudaErrorInvalidValue`. The failure is **not** in the P2P GPU→GPU converter (Plan 07-02 fixed that), it is in the `host_parquet_representation → gpu_table_representation` conversion path used by parquet scans and in the `host_data_representation → gpu_table_representation` path used by cached scans. The cross-device hazard is that `duckdb_scan_executor::_stream_pool` is hardcoded to GPU 0's stream pool (`src/op/scan/duckdb_scan_executor.cpp:57-58`), but when `select_target_gpu()` returns GPU 1, the scan task submits cuDF allocations + H2D copies on a GPU-0-bound stream while `rmm::cuda_set_device_raii` sets the current device to GPU 1 — RMM's per-stream tracking and cuDF's internal cuda_memcpy utility observe a stream/device mismatch and raise `cudaErrorInvalidValue`.

The fix pattern is exactly the Pattern 2 idiom already proven in `src/data/sirius_p2p_converter.cpp` (commit `18352b9`): **under `rmm::cuda_set_device_raii` for the target device, construct a fresh target-device-bound `rmm::cuda_stream` (or acquire one from the target memory_space's own stream pool), and issue all allocations + copies on that target stream.** Apply this in three seams: (1) `duckdb_scan_executor::manager_loop` around the parquet-scan dispatch so the scan task runs on a target-GPU-bound stream; (2) `pipeline::lock_or_prepare_batch` when it calls `batch->convert_to<gpu_table_representation>(registry, target_space, stream)` — the caller's stream (from `gpu_pipeline_executor._stream_pool`) IS target-bound, but the batch's source representation may reside on a different device, so the converter body (specifically `convert_host_parquet_to_gpu_with_prefetched_data_source` at `src/data/host_parquet_representation_converters.cpp:55-119`) already sets `target_device_raii` correctly — this path likely passes; the defect is elsewhere, probably in `convert_host_fast_to_gpu` in the cucascade submodule which does set device_guard at line 837 but consumes `stream` that may be bound to a different device for its cudf::table construction at line 851. Phase 8 must audit and, where the bug manifests, add a Sirius-side override for the host→gpu converter (same pattern as the P2P override). (3) Optionally extend `gpu_pipeline_executor._stream_pool` per-device binding verification so the executor's stream pool truly matches `_memory_space->get_device_id()` (already correct at `gpu_pipeline_executor.cpp:45`); this seam is likely clean.

**Primary recommendation:** Fix in three waves — (W1) make `duckdb_scan_executor` per-target-GPU stream-bound (the root cause), (W2) audit remaining `convert_to<gpu_table_representation>` paths in `lock_or_prepare_batch` and add a Sirius-side override for `host_data_representation → gpu_table_representation` if cucascade's built-in still races; and (W3) add `[mgpu-audit]` entries for `pipeline_task` / `scan_batch` with batch/task IDs so Catch2 tests can grep and count per-GPU, then parameterize `test_gpu_execution_tpch.cpp` on `num_gpus ∈ {1, 2}`.

<user_constraints>
## User Constraints (from CONTEXT.md)

*No CONTEXT.md exists for Phase 8 at research time — user skipped `/gsd:discuss-phase` and went directly to `/gsd:plan-phase 8`. The constraints below are therefore inherited from STATE.md / REQUIREMENTS.md / ROADMAP.md:*

### Locked Decisions (inherited from v1.2 roadmap + v1.1 STATE.md)

- **Single phase** — Phase 8 is the only phase in v1.2; all 11 requirements (FIX-01..04, TEST-01..04, AUDIT-01..03) ship together. Enforced by `granularity: coarse` in `.planning/config.json`.
- **Sirius-side converter override is the fix pattern.** Pattern 2 (`src/data/sirius_p2p_converter.cpp`) is the reference. Do NOT patch the cucascade submodule (`f47de0b` pin is preserved; cucascade/CLAUDE.md constraints forbid upstream changes in this phase).
- **No net-new `rmm::cuda_stream_default` uses** (HYG-02 preserved from v1.1). Every new stream must be an explicit `rmm::cuda_stream` or a pool-acquired `rmm::cuda_stream_view`.
- **Sticky `cudaGetLastError()` consume** after any `cuda*` call that can leave state in the thread-local slot (inherited from v1.1 decisions; Pattern 2 override already does this).
- **`[mgpu-audit]` log tag preserved** — new entries added by Phase 8 must use the exact same `[mgpu-audit]` prefix so the verification `grep` patterns keep working.
- **Integration fixture scope:** flip `num_gpus: 2` via **parameterization** of `test_gpu_execution_tpch.cpp`, NOT by editing `integration.yaml` to permanently change the default. Per STATE.md Blockers: the parameterization approach is what AUDIT-03 requires anyway — the 2-GPU variant MUST execute in the default `unit-tests` run, but the 1-GPU variant need not be removed.
- **Test framework = Catch2 v2.13.10** (already pinned; both Sirius and cucascade standardized on v2). Use `TEMPLATE_TEST_CASE_METHOD`, or duplicate TEST_CASEs with one `num_gpus=1` + one `num_gpus=2` flavor, or invoke `compare_gpu_vs_cpu` twice per query after swapping configs. Executor's choice at plan time.
- **MCP wrappers for all build/test** (user preference per memory file): use `mcp__project-commands__run_command build` and `mcp__project-commands__run_command unit-tests`, NOT `pixi run make` directly.
- **Datasets on disk**: `/datasets/tpch_parquet_sf1/` `/datasets/tpch_parquet_sf10/` `/datasets/tpch_parquet_sf100/` (22.8 GB lineitem) `/datasets/tpch_parquet_sf300/` all present on N=2 verification host. Phase 8 SF100 Q1 uses `/datasets/tpch_parquet_sf100/lineitem.parquet` for the ship gate per ROADMAP success criterion 1.

### Claude's Discretion

- **Whether to parameterize via `TEMPLATE_TEST_CASE_METHOD`, dual-TEST_CASE flavors, or a runtime config-swap inside a single TEST_CASE body** — pick the lowest-churn option that lets every TPC-H query run on both `num_gpus=1` and `num_gpus=2` without editing `integration.yaml`.
- **Whether FIX-02 audit yields exactly 1 new Sirius-side override or 0 or 2+ overrides** — depends on whether cucascade's `convert_host_fast_to_gpu` (`cucascade/src/data/representation_converter.cpp:825-856`) has the same cross-stream race as its gpu→gpu twin. Plan should include an explicit probe task that runs TPC-H Q1 with materialized_columns=true on num_gpus=2 and captures the failing stack; if that path hits `cuda_memcpy.cu`, add an override; if it doesn't, FIX-02 is covered by FIX-01's scan-executor stream fix.
- **`scan_batch` and `pipeline_task` count thresholds in AUDIT assertions** — ROADMAP criterion 4 locks "≥ 5 on BOTH GPUs", but the test implementation can pick any SQL query + SF factor that reliably produces ≥10 batches per GPU (Q1 on SF1 lineitem produces ~6 batches total at 100 MB `scan_task_batch_size`; SF10 produces ~60; executor picks the combo).
- **Whether to add a `SIRIUS_TEST_NUM_GPUS` env var hook or use a dedicated `integration-2gpu.yaml` fixture file** — functionally equivalent; pick the one that plays well with the existing `sirius_config_env_guard` + `g_integration_env` infrastructure (`test/cpp/utils/sirius_test_env.cpp`).
- **`[mgpu-audit]` payload shape for AUDIT-01/02** — emit `batch_id` or `task_id` so test can count unique IDs per GPU (robust to log-line repetition).

### Deferred Ideas (OUT OF SCOPE)

- **Upstream cucascade PRs** (convert_gpu_to_gpu fix, convert_host_fast_to_gpu fix if needed). Sirius override is the v1.2 fix; upstream is nice-to-have.
- **Performance regression comparisons** (Phase-5 vs Phase-4, SF10 vs SF1 throughput tables). Per user directive 2026-04-21 ("let's just make sure everything is working, we can optimize later").
- **Flipping default `num_gpus: 2` globally in `integration.yaml`** — too aggressive for v1.2; scope to TPC-H parameterization only.
- **Other fixture files beyond integration TPC-H** (test_gpu_execution_multi_format.cpp, test_gpu_execution_locality.cpp parameterization) — may follow in v1.3 if the pattern proves stable.
- **TPC-H Q4 parquet intermittent flake** — scoped separate investigation; v1.2 policy is "retry once per v1.1 precedent".
- **`cudaDeviceDisablePeerAccess` on explicit teardown** — current reliance on CUDA process-exit cleanup is adequate.
- **Cucascade `idisk_io_backend` file-handle cache** — upstream concern; not triggered by the v1.1 bug.
- **Super Sirius dispatch architecture changes** — v1.2 is a targeted fix, not a redesign.
- **SF300 as a hard ship gate** — "icing, not required" per ROADMAP criterion 6.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **FIX-01** | `pipelineable_operator_data::prepare_for_processing` → `pipeline::lock_or_prepare_batch` no longer throws `cudaErrorInvalidValue: invalid argument` when source and target devices differ. Pack on source-device RAII + source stream; copy on target stream (Pattern 2 — same shape as `src/data/sirius_p2p_converter.cpp`). | **Root cause located at `src/op/scan/duckdb_scan_executor.cpp:57-58` AND `:322`**: the scan executor has a SINGLE `_stream_pool` bound to `_gpu_memory_spaces[0]` (GPU 0). When `select_target_gpu()` returns GPU 1 (`:283`), the scan task still dispatches on a GPU-0-bound stream but allocates on GPU 1's memory_space via `set_materialized_columns(..., target_gpu_space)` (`:297-298`). `cudf::io::read_parquet(opts, stream, mr_ref)` at `src/data/host_parquet_representation_converters.cpp:92` then uses a GPU-0 stream with a GPU-1 allocator under a `rmm::cuda_set_device_raii{GPU 1}` (`:66`), and cuDF's cuda_memcpy utility catches the mismatch with `cudaErrorInvalidValue`. Fix shape: make `duckdb_scan_executor` maintain a per-GPU stream pool map (`unordered_map<int, exclusive_stream_pool>`) and acquire from the target GPU's pool after `select_target_gpu()`. |
| **FIX-02** | Audit every other cross-device CUDA memcpy call-site in `src/pipeline/` and `src/op/` for the same bug pattern; apply the same fix where present. Document surfaces covered. | Audit targets: (a) `pipeline::lock_or_prepare_batch` at `src/include/pipeline/batch_lock_utils.hpp:48-125` — caller passes `exec_stream` bound to target GPU (correct), but `convert_to<gpu_table_representation>` dispatches into cucascade's `convert_host_fast_to_gpu` at `cucascade/src/data/representation_converter.cpp:825-856` which builds `cudf::table` on the caller's stream after `device_guard` at L837 — needs probe (does cached-scan path hit `cuda_memcpy.cu:42`?); if yes, add a Sirius-side override mirroring `sirius_p2p_converter_factory`. (b) `src/op/scan/duckdb_scan_task.cpp:444-610` (`duckdb_scan_task::make_data_batch`) — builds host-resident `host_data_representation` from DuckDB DataChunks; no GPU allocation, safe. (c) `src/op/scan/cpu_source_task.cpp:86-304` — all host-side; safe. (d) Per the v1.1 E2E verify, trivial `COUNT(*)` PASSES — so `cpu_source_task` path is clean; only parquet + cached-scan paths fail. |
| **FIX-03** | Zero net-new `rmm::cuda_stream_default` uses (HYG discipline maintained). | Current state: `grep -rn 'rmm::cuda_stream_default' src/` returns 0 hits on `feature/single-node-multi-gpu2` HEAD (confirmed via Grep during research). Phase 8 must preserve 0. Every new stream construction must be `rmm::cuda_stream` under a `cuda_set_device_raii` guard, or an `acquire_stream()` from a memory_space/stream pool. |
| **FIX-04** | Build clean on MCP (`mcp__project-commands__run_command build` exit 0) after fix. | Build system details in CLAUDE.md §"Build System". MCP runner is pre-configured per `.planning/PROJECT.md`; treats `build` exit code != 0 as fail. No build-system changes required for Phase 8 (no new targets, no new CMakeLists edits expected unless FIX-02 adds a new `src/data/sirius_host_to_gpu_converter.cpp` override file). |
| **TEST-01** | `test/cpp/integration/test_gpu_execution_tpch.cpp` parameterized on `num_gpus ∈ {1, 2}`. | File is 4130 lines, 358 `TEST_CASE_METHOD` invocations. Fixture class `GPUExecutionFixtureBase` at `:62-180` gates on `sirius::test::g_integration_env` (from `test/cpp/unittest.cpp:104-108`). Shared env loads `test/cpp/integration/integration.yaml` (which sets `num_gpus: 1`). Parameterization options: **(A)** Extract a helper `fs::path integration_config_for(int num_gpus)` that returns `integration.yaml` or `integration-2gpu.yaml` based on argument + Env guard; use `GENERATE(1, 2)` Catch2 generator inside each TPC-H TEST_CASE body; call `compare_gpu_vs_cpu(query)` twice, once per config. **(B)** Convert each TPC-H TEST_CASE to `TEMPLATE_TEST_CASE_METHOD` with type tags `<num_gpus_1, num_gpus_2>`. **(C)** Duplicate each TPC-H TEST_CASE with a `_2gpu` suffix and a fixture that wires `integration-2gpu.yaml`. Option A has lowest churn (no TEST_CASE duplication, no template metaprogramming); option C is simplest to read but doubles TEST_CASE count. Option B fights the existing shared_env plumbing. **Recommend A.** |
| **TEST-02** | `test/cpp/integration/integration.yaml` (or fixture config) supports `num_gpus: 2` at test runtime; no permanent flip of the default. 2-GPU variant MUST execute in the default `unit-tests` run. | Strategy: add `test/cpp/integration/integration-2gpu.yaml` alongside the existing file, identical content except `num_gpus: 2`. Modify `unittest.cpp:104-108` to create a SECOND `shared_test_env` (`integration_env_2gpu_`) OR modify `sirius_config_env_guard` to accept `num_gpus` and pause/resume the integration env between configs. The shared-env pause/resume pattern already exists (`shared_test_env::pause()` + `resume()` at `sirius_test_env.cpp:69-80`) so adding a second path is consistent. Alternative: keep single `g_integration_env`, swap its config path via a test-local helper that does `g_integration_env->pause(); setenv(SIRIUS_CONFIG_FILE, new_path); g_integration_env->resume();` — more invasive because it mutates global state mid-test run. |
| **TEST-03** | All 22 TPC-H queries pass at SF1 on `num_gpus: 2` — results bitwise identical to the `num_gpus: 1` baseline. | TPC-H TEST_CASEs at lines 3229-4130 of `test_gpu_execution_tpch.cpp`. Each uses `compare_gpu_vs_cpu(query, [tolerance])` which runs GPU and CPU, sorts by all columns via DuckDB `ORDER BY`, and asserts `Value::ToString()` equality (with float tolerance). Running the same 22 queries with `num_gpus: 2` config and the FIX-01 fix in place is expected to pass — the GPU path produces the same results as the CPU path regardless of GPU topology. No dataset prep needed; `test/cpp/integration/data/duckdb/integration.duckdb` + `test/cpp/integration/data/parquet/*.parquet` are SF1 already. |
| **TEST-04** | TPC-H Q1, Q6, Q12 pass at SF10 on `num_gpus: 2` (smoke-test scale, matches v1.1 Phase-5 evidence). | SF10 parquet is at `/datasets/tpch_parquet_sf10/` on the verification host. Integration test binary doesn't currently load SF10; plan should either (a) add a new TEST_CASE with a parquet fixture pointed at SF10 path behind an env var (`SIRIUS_TEST_SF10_PATH`) that skips if unset, or (b) add a SQL-logic-test at `test/sql/tpch-sirius-sf10-2gpu.test` that's gated on the SF10 dataset. Option (a) integrates with the Catch2-unit-tests `mcp__project-commands__run_command unit-tests` flow (required by AUDIT-03); option (b) runs via `make test`. **Recommend (a)** for unit-tests integration. |
| **AUDIT-01** | A dedicated TEST_CASE runs TPC-H SF1 on `num_gpus: 2` with `[mgpu-audit]` logging, captures the log, asserts `pipeline_task` count > 0 on **both** GPU 0 and GPU 1. ROADMAP criterion 4 strengthens to ≥ 5 per GPU. | `[mgpu-audit]` emissions at `src/pipeline/pipeline_executor.cpp:249` (pipeline_task dispatch — emitted per task) and `src/op/scan/duckdb_scan_executor.cpp:182` (scan_batch assignment — emitted per scan batch). Test can route `SIRIUS_LOG_DIR` to a `tmp_path` set up by the test, run the query, then grep the log file for `[mgpu-audit].*pipeline_task dispatched to GPU 0` + `... GPU 1` and count occurrences. Choose a query that produces many batches: Q1 on lineitem @ SF1 (`scan_task_batch_size: 100000000` → ~6 batches on SF1 600k-row lineitem); more reliable at SF10. Log-file parse pattern: `grep -cE "\[mgpu-audit\] pipeline_task.*GPU 0"` + same for `GPU 1`. |
| **AUDIT-02** | Same assertion for `scan_batch` count > 0 on both GPUs (≥5 per ROADMAP criterion 4). | Identical shape to AUDIT-01 but grepping `scan_batch assigned to GPU`. Phase 8 can emit BOTH counts from the SAME TEST_CASE — one setUp phase runs the query, two separate assertions count pipeline_task and scan_batch per GPU. Consider emitting `batch_id` / `task_id` in the audit log so counts are uniquely attributable (robust to duplicate log lines from retries). |
| **AUDIT-03** | Audit gate is checked by the default `unit-tests` run — regressions to single-GPU-only distribution break the build. | Means the AUDIT-01/02 TEST_CASE must NOT carry a `[.]` hide tag, must NOT be filtered out by default, and must execute on the shared-env 2-GPU variant (TEST-02 infrastructure). On a single-GPU CI host the TEST_CASE should WARN+return (Catch2-v2 convention, per STATE.md Plan 01-03 decision, matching the existing `[multi_gpu_transfer]` WARN+return pattern at `test_downgrade_executor.cpp:489-492`). Single-GPU CI doesn't fail the TEST_CASE; a 2-GPU host with incomplete distribution DOES fail. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| CUDA Runtime | 13.0 (`/usr/local/cuda-13.0/bin/nsys`) | `cudaMemcpyAsync`, `cudaMemcpyPeerAsync`, `cudaSetDevice`, `cudaGetDevice`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost` | No alternative. Already used throughout Sirius + cucascade. |
| cuDF | 26.04 (pinned via `pixi.toml` default env) | `cudf::io::read_parquet`, `cudf::pack`/`cudf::unpack`, `cudf::table` — the GPU-tier data container | Pinned via pixi; `cudf::pack`/`cudf::unpack` come from `<cudf/contiguous_split.hpp>` (NOT `<cudf/copying.hpp>` — Plan 07-02 hit this confusion; executor should use `<cudf/contiguous_split.hpp>`). |
| RMM | cudf 26.04 pin | `rmm::cuda_set_device_raii`, `rmm::cuda_device_id`, `rmm::cuda_stream`, `rmm::cuda_stream_view`, `rmm::device_buffer`, `rmm::device_uvector`, `rmm::device_async_resource_ref` | Project-wide convention: ALWAYS explicit stream; NEVER `rmm::cuda_stream_default` (HYG-02). |
| cuCascade | `f47de0b` (Phase 4 BUMP-01 pin, **preserved unchanged in Phase 8**) | `representation_converter_registry`, `memory_space`, `exclusive_stream_pool`, `memory_reservation_manager`, `idata_representation`, `gpu_table_representation`, `host_data_representation`, `host_data_packed_representation` | Sirius-side overrides via `unregister_converter<S,T>() + register_converter<S,T>(factory)` are the approved v1.2 fix; cucascade submodule is NOT patched. |
| Catch2 | v2.13.10 | Test framework (both Sirius and cucascade standardize on v2); hidden-tag `[.]` convention, WARN+return idiom for missing hardware, `TEST_CASE_METHOD`, `GENERATE()` generator, `UNSCOPED_INFO`, `REQUIRE`, `REQUIRE_MESSAGE` | Already present; no version bump. Plan 07-02 SUMMARY warned that `REQUIRE_MESSAGE` may not be available — verify at plan time and fall back to `INFO() + REQUIRE()` if so. |
| spdlog | existing | `[mgpu-audit]` audit log emissions via `SIRIUS_LOG_INFO` macro | Already the project convention (MGPU-01 startup log, IO-11 audit log). Phase 8 extends existing tag, does not introduce new one. |
| DuckDB | bundled submodule | `duckdb::Connection`, `duckdb::DuckDB`, `con->Query()`, `compare_gpu_vs_cpu` harness relies on DuckDB CPU path as oracle | No new DuckDB usage; Phase 8 only modifies Sirius-side test fixtures + extension. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| nvtx3 | bundled | `nvtx3::scoped_range` | Already used in `gpu_pipeline_task::execute` at `src/pipeline/gpu_pipeline_task.cpp:298`; Phase 8 may add range markers for scan-task stream assignment if debugging surfaces a nsys profile need. |
| absl | bundled (`absl::any_invocable`, `absl::Cleanup`) | RAII stream-reservation release | Already used at `src/pipeline/gpu_pipeline_task.cpp:309`; Phase 8 does not introduce new absl usage. |
| std::format (C++20) | toolchain | Test assertion failure messages (`UNSCOPED_INFO("[...]")` payloads) | Already used throughout test files. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Per-GPU `exclusive_stream_pool` in `duckdb_scan_executor` | Single `cuda_stream_default` | Violates HYG-02 (FIX-03). Pool-per-GPU is the only correct shape; `std::unordered_map<int, unique_ptr<cucascade::memory::exclusive_stream_pool>>` keyed by device_id. |
| Sirius-side override for `host_data_representation → gpu_table_representation` | Patch cucascade's `convert_host_fast_to_gpu` | User constraints lock "no cucascade submodule changes" in v1.2; override is mandatory if the path is buggy. |
| `TEMPLATE_TEST_CASE_METHOD` parameterization (Option B) | Dual TEST_CASE flavors (Option C) or `GENERATE()` (Option A) | Option B requires type-tagging all 358 TEST_CASEs — high churn. Option C duplicates code. Option A (single TEST_CASE, two config swaps inside the body, two `compare_gpu_vs_cpu` calls) has lowest churn and is recommended. |
| Dedicated `integration-2gpu.yaml` fixture | Runtime `num_gpus` env var that overrides `integration.yaml` | Env-var route requires `SiriusContext::initialize` to read the override; adds a surface. Separate YAML file is cleaner and parallels the existing `test/cpp/scan/memory.yaml` vs `integration.yaml` pattern (each test env gets its own). |

**Installation:** No new libraries. Verify pixi env is active:
```bash
pixi shell  # one-time env activation
mcp__project-commands__run_command build  # verifies compile-gate
```

## Architecture Patterns

### Recommended File Structure

```
src/
├── op/scan/
│   ├── duckdb_scan_executor.cpp        # W1: per-GPU stream-pool map; FIX-01 primary fix
│   ├── duckdb_scan_executor.hpp        # W1: add _gpu_stream_pools map
│   └── parquet_scan_task.cpp           # W1: scan task receives target-GPU-bound stream from caller
├── data/
│   ├── sirius_host_to_gpu_converter.cpp    # W2 (CONDITIONAL): NEW — Sirius-side host→gpu override if FIX-02 probe shows cucascade's convert_host_fast_to_gpu races
│   └── sirius_host_to_gpu_converter.hpp    # W2 (CONDITIONAL): NEW
├── include/
│   ├── data/
│   │   └── sirius_converter_registry.hpp   # W2 (CONDITIONAL): add unregister+register for host→gpu if W2 override authored
│   └── op/scan/
│       └── duckdb_scan_executor.hpp        # W1: interface change
├── pipeline/
│   └── pipeline_executor.cpp           # W3: extend [mgpu-audit] emission at :249 with task_id
└── op/scan/
    └── duckdb_scan_executor.cpp        # W3: extend [mgpu-audit] emission at :182 with batch_id

test/cpp/integration/
├── integration.yaml                    # UNCHANGED (num_gpus: 1)
├── integration-2gpu.yaml               # W3: NEW — num_gpus: 2 variant
├── test_gpu_execution_tpch.cpp         # W3: parameterize TPC-H TEST_CASEs via GENERATE(num_gpus)
└── test_gpu_execution_tpch_mgpu_audit.cpp  # W3: NEW — AUDIT-01/02 dedicated TEST_CASE with log-grep assertions

test/cpp/utils/
├── sirius_test_env.hpp                 # W3: add optional g_integration_env_2gpu OR config-swap helper
└── sirius_test_env.cpp                 # W3: matching impl
test/cpp/
└── unittest.cpp                        # W3: wire second shared_test_env OR the config-swap seam
```

### Pattern 1 (**primary fix**): Per-GPU Stream Pool in `duckdb_scan_executor`

**What:** Replace the single `_stream_pool` bound to GPU 0 with a `std::unordered_map<int, std::unique_ptr<cucascade::memory::exclusive_stream_pool>>` keyed by device_id, so each target GPU has its own stream pool. When `select_target_gpu()` returns device_id `d`, acquire the exec_stream from `_gpu_stream_pools[d]`.

**When to use:** In `duckdb_scan_executor` constructor and in `manager_loop` immediately before dispatch. This is the FIX-01 primary fix site.

**Example** (source: ADAPTED from `src/op/scan/duckdb_scan_executor.cpp:43-59` + `:282-324`):

```cpp
// duckdb_scan_executor.hpp additions
std::unordered_map<int, std::unique_ptr<cucascade::memory::exclusive_stream_pool>>
    _gpu_stream_pools;

// duckdb_scan_executor.cpp constructor replaces lines 55-58:
for (auto* space : _gpu_memory_spaces) {
  auto const dev_id = space->get_device_id();
  _gpu_stream_pools.emplace(
    dev_id,
    std::make_unique<cucascade::memory::exclusive_stream_pool>(
      rmm::cuda_device_id{dev_id}, config.num_threads));
}
// Keep _gpu_memory_space backward-compat pointer; DO NOT keep _stream_pool singular member.

// duckdb_scan_executor.cpp manager_loop replaces line 322:
auto* pool_iter = _gpu_stream_pools.find(target_gpu_id);
if (pool_iter == _gpu_stream_pools.end()) {
  SIRIUS_LOG_ERROR("duckdb_scan_executor: no stream pool for GPU {}", target_gpu_id);
  _completion_handler->report_error(
    "duckdb_scan_executor: missing stream pool for target GPU " +
    std::to_string(target_gpu_id));
  break;
}
rmm::cuda_set_device_raii scan_device_guard{rmm::cuda_device_id{target_gpu_id}};
auto exc_stream = pool_iter->second->acquire_stream(
  cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
// The stream is now bound to target_gpu_id — cudf::io::read_parquet + RMM allocations
// + cuda_memcpy utilities all observe a matching (stream, current_device) pair.
```

**Why this shape:**
- Mirrors `gpu_pipeline_executor`'s already-correct pattern (`src/pipeline/gpu_pipeline_executor.cpp:45` — `_stream_pool` scoped to the executor's single GPU). Since `duckdb_scan_executor` is shared across all GPUs, it needs the map.
- `rmm::cuda_set_device_raii` around `acquire_stream()` ensures any lazy stream allocation happens on the correct device.
- The guard MUST stay live across the entire dispatch lambda; current code already captures `stream` into the lambda, so lambda entry is after `acquire_stream`. The lambda body then operates on a target-bound stream; the `rmm::cuda_set_device_raii` at the top of the lambda is still needed because `cudaSetDevice` is thread-local and the bounded_pool worker thread may have been pinned to a different device.

### Pattern 2: Sirius-Side Converter Override (Reference — reuse if FIX-02 needs it)

**What:** Register a Sirius-authored converter factory that replaces the cucascade built-in for a given `(Source, Target)` pair. The factory explicitly manages device context + stream binding to avoid cross-device hazards.

**When to use:** Exactly the pattern used by `src/data/sirius_p2p_converter.cpp` for `gpu_table_representation → gpu_table_representation`. Apply the SAME shape to `host_data_representation → gpu_table_representation` IF the FIX-02 probe surfaces a failure on the cached-scan path. Skip if FIX-01's scan-executor fix closes the bug entirely.

**Example** (source: EXACT pattern from `src/data/sirius_p2p_converter.cpp:33-126`, adapted for host→gpu direction):

```cpp
// src/data/sirius_host_to_gpu_converter.cpp (CONDITIONAL — only if FIX-02 probe demands it)
std::unique_ptr<cucascade::idata_representation> sirius_host_fast_to_gpu_factory(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)  // caller's stream; may be bound to a different device
{
  auto& fast_source = source.cast<cucascade::host_data_representation>();
  auto const target_device_id = target_memory_space->get_device_id();

  // Set target device context.
  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{target_device_id}};

  // Acquire a target-bound stream from the target memory_space's pool.
  // The caller's stream may belong to a different device; using it for H2D copies
  // + cudf::table construction causes cudaErrorInvalidValue inside cuDF's
  // cuda_memcpy utility (the same symptom as the v1.1 E2E verification bug).
  auto target_stream = target_memory_space->acquire_stream();
  auto mr = target_memory_space->get_default_allocator();

  // Reconstruct each column from metadata + host pinned blocks.
  // Copy shape mirrors cucascade::convert_host_fast_to_gpu at
  // cucascade/src/data/representation_converter.cpp:840-851 — same BatchCopyAccumulator
  // + reconstruct_column calls, but on target_stream instead of caller's stream.
  cucascade::BatchCopyAccumulator batch;
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(fast_source.get_host_table()->columns.size());
  for (const auto& col_meta : fast_source.get_host_table()->columns) {
    gpu_columns.push_back(cucascade::reconstruct_column(
      col_meta, fast_source.get_host_table()->allocation, target_stream, mr, batch));
  }
  batch.flush(target_stream, cudaMemcpySrcAccessOrderDuringApiCall);

  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));
  target_stream.synchronize();

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(new_table),
    *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}
```

Registration site (follow `src/include/data/sirius_converter_registry.hpp:69-82` exactly):
```cpp
auto const removed = instance_->unregister_converter<
  cucascade::host_data_representation,
  cucascade::gpu_table_representation>();
if (!removed) { spdlog::warn("sirius: host->gpu converter not previously registered"); }
instance_->register_converter<
  cucascade::host_data_representation,
  cucascade::gpu_table_representation>(&sirius::data::sirius_host_fast_to_gpu_factory);
spdlog::info("sirius: MGPU-08 host->gpu converter override registered");
```

**Why this shape:**
- Eliminates dependency on caller-supplied stream's device — factory acquires its own target-bound stream.
- `BatchCopyAccumulator` + `reconstruct_column` are **in the cucascade namespace**; Sirius code accesses them through `cucascade::` scope; if these symbols are not part of cucascade's public API, the factory must be implemented differently (manual column construction from `host_data_representation`'s underlying buffers). Plan task should verify symbol visibility via `grep -rn 'class BatchCopyAccumulator' cucascade/include/`.
- Alternative if cucascade symbols are private: pack/unpack shape (mirror `sirius_p2p_converter_factory` pack/peer-copy/unpack; use `cudaMemcpyHostToDevice` instead of `cudaMemcpyPeerAsync`) — same safety properties, different copy function.

### Pattern 3: `[mgpu-audit]` Log Extension for Assertion-Friendly Counting

**What:** Extend the existing `[mgpu-audit]` emissions to include unique identifiers (batch_id, task_id) so tests can count unique IDs per GPU via grep + sort + uniq. The tag is preserved verbatim; only the payload grows.

**When to use:** Every new emission in Phase 8 and the existing two emissions at `src/pipeline/pipeline_executor.cpp:249` + `src/op/scan/duckdb_scan_executor.cpp:182`.

**Example** (source: ADAPTED from existing emissions):

```cpp
// pipeline_executor.cpp line 249 (current):
SIRIUS_LOG_INFO("[mgpu-audit] pipeline_task dispatched to GPU {}", target_device_id);

// pipeline_executor.cpp line 249 (Phase 8):
auto const task_id = gpu_task ? gpu_task->get_task_id() : 0;
SIRIUS_LOG_INFO("[mgpu-audit] pipeline_task dispatched to GPU {} task_id={}",
                target_device_id, task_id);

// duckdb_scan_executor.cpp line 182 (current):
SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} (available: {} bytes)",
                space->get_device_id(), space->get_available_memory());

// duckdb_scan_executor.cpp line 182 (Phase 8) — add scan_round_robin counter as batch_id:
auto const batch_seq = _scan_round_robin.load();  // already fetched
SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} batch_id={} (available: {} bytes)",
                space->get_device_id(), batch_seq, space->get_available_memory());
```

**Why this shape:**
- Keeps `grep 'mgpu-audit'` + `grep -c 'GPU 0'` workflows from v1.1 E2E verification unchanged — older greps still count correctly.
- Adds a unique-ID suffix so tests can do: `grep '[mgpu-audit] pipeline_task.*GPU 0' log | awk '{print $NF}' | sort -u | wc -l` to get unique-task-to-GPU-0 count, which is the most robust assertion for AUDIT-01's "≥ 5 pipeline_task on BOTH GPUs".
- `SIRIUS_LOG_INFO` format string uses `{}` fmt-style — confirmed already the project convention (`src/pipeline/pipeline_executor.cpp:249`).

### Anti-Patterns to Avoid

- **DON'T use `rmm::cuda_stream_default`** anywhere — violates FIX-03/HYG-02. Every stream must be explicit.
- **DON'T edit `integration.yaml` to flip `num_gpus: 2`** — user constraint locks "no permanent flip". Use a separate `integration-2gpu.yaml` + parameterization.
- **DON'T patch `cucascade/src/data/representation_converter.cpp`** — cucascade submodule is pinned at `f47de0b`. Apply fix via Sirius-side override.
- **DON'T change `select_target_gpu()`'s distribution algorithm** — the algorithm is correct (MGPU-07 closed in Phase 7); the bug is in the *stream* choice after distribution, not the GPU choice itself.
- **DON'T use `CUCASCADE_CUDA_TRY` in the new converter body if W2 branch is taken** — match `sirius_p2p_converter.cpp`'s inline `cudaError_t` + `std::runtime_error` pattern (Plan 07-02 decision).
- **DON'T remove the existing `[mgpu-audit]` emissions** — they're the v1.1 verification baseline. Extending payload is fine; removing/renaming breaks the log-grep workflows.
- **DON'T introduce a new shared mutable global for 2-GPU env switching** — use Catch2 fixtures + pause/resume shape already present.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-GPU CUDA stream pool | Manual `std::vector<cudaStream_t>` + manual create/destroy | `cucascade::memory::exclusive_stream_pool` (already used at `duckdb_scan_executor.cpp:57` + `gpu_pipeline_executor.cpp:45`) | Handles acquire/release + grow/block policies; integrates with cucascade's memory accounting. |
| Device context switching | Raw `cudaSetDevice` + try/catch | `rmm::cuda_set_device_raii` | RAII restore on scope exit; no silent device-leak into caller's context. Already the project convention. |
| Converter factory registration | Custom dispatch table / `std::map<type_index, fn>` | `cucascade::representation_converter_registry::register_converter<S,T>(fn)` + `unregister_converter<S,T>()` | Already thread-safe, already used for the P2P override, already has the right type-pair dispatch shape. |
| Log-based test assertion | Custom in-process log capture + lambda callback | `SIRIUS_LOG_DIR` → `tmp_path` + `grep` post-facto | Mirrors v1.1 E2E verification's grep workflow; no new infrastructure. The audit TEST_CASE sets `SIRIUS_LOG_DIR` via `setenv` in a fixture, runs the query, reads the emitted log file, counts. |
| TPC-H data generation | Custom dbgen | Pre-generated parquet at `/datasets/tpch_parquet_sfN/` | Already on verification host per `.planning/STATE.md` + ROADMAP criterion 1. `test/cpp/integration/data/parquet/*.parquet` is bundled SF1 for unit tests. |
| Multi-config test fixture | Per-TEST_CASE `setenv` + DuckDB reconnect | Shared `g_integration_env` + pause/resume pattern (`sirius_test_env.cpp:69-80`) | Already exists; adding a second env mirrors the pattern without new invariants. |

**Key insight:** Phase 8 is 90% reusing Pattern 2 from Plan 07-02 and extending existing fixtures/logs. The only genuinely new code is the `_gpu_stream_pools` map in `duckdb_scan_executor` (~30 lines) and the audit TEST_CASE body (~80 lines). Everything else is small edits.

## Runtime State Inventory

Phase 8 is **not** a rename/refactor/migration phase — no strings to propagate, no secrets or stored data to migrate, no OS-level registrations. It's a stream-correctness code fix + test additions. This section is omitted.

## Common Pitfalls

### Pitfall 1: Silent `cudaSetDevice` on wrong thread

**What goes wrong:** `cudaSetDevice` is thread-local. The `gpu_pipeline_executor`'s per-thread init (`src/pipeline/gpu_pipeline_executor.cpp:54-72`) pins worker threads to the executor's device. But `duckdb_scan_executor`'s worker threads come from a different pool (`_bounded_pool`) and may NOT be pinned to `target_gpu_id`. A `cudaSetDevice` or `rmm::cuda_set_device_raii` inside the dispatch lambda is still required.

**Why it happens:** `_bounded_pool->dispatch` workers don't know which GPU the task targets; the main `manager_loop` selected the GPU via `select_target_gpu()` but the thread pool itself is GPU-agnostic.

**How to avoid:** Inside the dispatch lambda (`duckdb_scan_executor.cpp:324-347`), add `rmm::cuda_set_device_raii guard{rmm::cuda_device_id{target_gpu_id}}` BEFORE any cudf/RMM call. Or: pre-pin the bounded_pool threads per target-GPU by using a pool per GPU (more invasive; defer unless profiling shows repeated `cudaSetDevice` is a hot spot).

**Warning signs:** `cudaErrorInvalidValue` that only reproduces on 2+ GPUs; stream-allocated-on-X / copy-issued-on-Y log lines with mismatched device IDs; nsys timeline showing the scan kernel on GPU 0 while the allocation shows GPU 1 memory (or vice versa).

### Pitfall 2: cuDF's `cuda_memcpy.cu` is version-dependent

**What goes wrong:** The exact line number `cuda_memcpy.cu:42` is from libcudf 26.04 (pinned). If cucascade bumps cudf in a future phase, the line number changes. Error-message regex matches that hard-code `:42` may break.

**Why it happens:** Conda-built libcudf ships with debug info pointing at `/tmp/conda-bld-output/bld/rattler-build_libcudf/work/cpp/src/utilities/cuda_memcpy.cu:42` — path + line are baked into the error. Upstream cudf main currently has the function body around `:54` (verified: `/home/william/repos/cudf/cpp/src/utilities/cuda_memcpy.cu` exists locally at a different line).

**How to avoid:** Assertions / log-greps for the bug-absent state should match on `cudaErrorInvalidValue` (the error class) AND `cuda_memcpy.cu` (the file), NOT the exact line. Example: `grep -qE 'cuda_memcpy\.cu.*cudaErrorInvalidValue'`. ROADMAP criterion 1 already uses `cuda_memcpy.cu:42` as shorthand — planner should interpret "no `cudaErrorInvalidValue` from `cuda_memcpy.cu`" rather than line-specific.

**Warning signs:** Tests assert against the exact line number and become stale; cudf bump (future phase) silently masks the regression.

### Pitfall 3: `SIRIUS_LOG_DIR` env var conflicts between tests

**What goes wrong:** Audit TEST_CASE sets `SIRIUS_LOG_DIR` to a tmp path so it can grep the output, but other parallel tests (Catch2 doesn't run tests in parallel by default, but `mcp__project-commands__run_command unit-tests` might) may write to the same log file and pollute counts. Additionally, if the env var is set globally via `setenv`, later tests inherit it.

**Why it happens:** `SIRIUS_LOG_DIR` is read at `SiriusContext::initialize()`, which happens ONCE per DuckDB instance lifetime. If the shared `g_integration_env` is not paused before the audit test sets the env var, the audit's log dir is ignored (SiriusContext was already initialized with the stale dir).

**How to avoid:** Audit TEST_CASE must (a) `g_integration_env->pause()`, (b) `setenv("SIRIUS_LOG_DIR", tmp.c_str(), 1)`, (c) create a fresh DuckDB instance locally (not via shared env), (d) run the query, (e) grep the log, (f) `unsetenv`, (g) `g_integration_env->resume()` (or let the listener do it at teardown). Mirrors the existing `sirius_config_env_guard` RAII pattern.

**Warning signs:** Audit count is 0 on a 2-GPU host where manual reproduction shows non-zero; log file in tmp dir is empty or missing the query's run.

### Pitfall 4: Q4 parquet intermittent flake can mask real regressions

**What goes wrong:** Per STATE.md + ROADMAP criterion 2, TPC-H Q4 parquet has a pre-existing flake. If the 2-GPU run of Q4 fails once, a naive test harness reports TEST-03 as failing. But ROADMAP's policy is "retry once per v1.1 precedent, not treated as regression".

**Why it happens:** Q4 has a timing-sensitive ordering interaction unrelated to multi-GPU; pre-existing from v1.1.

**How to avoid:** Plan should include a "retry-once" wrapper in `compare_gpu_vs_cpu` for Q4 specifically, OR document in the plan SUMMARY that Q4 is treated as a soft-fail. Don't build generic retry infrastructure — isolate to Q4 to avoid masking real regressions.

**Warning signs:** TEST-03 assertion fails only on Q4 parquet; retry succeeds; stack trace doesn't reference `cuda_memcpy.cu`.

### Pitfall 5: Over-scope creep into dispatch architecture

**What goes wrong:** While fixing the scan-executor stream pool, a researcher/planner might notice that `gpu_pipeline_task::execute`'s `task_creator` integration has its own cross-device consideration, or that `downgrade_executor` has similar patterns. Temptation: "let's fix them all". Reality: the v1.1 E2E verification ONLY reproduces on the scan → pipeline-task path; other paths are not known-broken.

**Why it happens:** Pattern recognition — the Pattern 2 shape is applicable in many places.

**How to avoid:** FIX-02 scope is "audit every cross-device memcpy site in src/pipeline/ and src/op/; fix where present". "Fix where present" means "where there's evidence of a bug" — i.e., where the v1.1 reproduction path hits, or where the FIX-02 probe demonstrates a new failure. Don't add preemptive overrides. Per user constraint: "v1.2 is a targeted fix, not a redesign."

**Warning signs:** Plan task count growing past 6-8; plan references files outside `src/op/scan/`, `src/data/`, `src/pipeline/pipeline_executor.cpp`, `test/cpp/integration/`, `test/cpp/utils/`.

### Pitfall 6: Ada Lovelace + Sapphire Rapids silent data corruption (inherited from Plan 07 RESEARCH)

**What goes wrong:** On RTX 6000 Ada Generation behind Intel Xeon Sapphire Rapids CPUs, PCIe P2P transfers can experience **silent data corruption** because the platform doesn't enforce PCIe posted-write ordering. The v1.1 E2E verification host (6f7e4c9-lcedt, Xeon SPR generation is plausible) hits this risk.

**Why it happens:** Ada arch depends on host PCIe transaction ordering for posted writes; Sapphire Rapids doesn't guarantee it.

**How to avoid:** Phase 7's fix already adds FNV-1a checksums to the GPU→GPU P2P round-trip tests (see `src/data/sirius_p2p_converter.cpp` + Plan 07-02 SUMMARY). Phase 8's FIX-02 / W2 path (host→gpu override, if authored) should mirror this checksum guard on its round-trip test if the conversion uses peer-copy — though typically host→gpu goes via `cudaMemcpyHostToDevice` not peer-copy, so this pitfall is less relevant for W2. For the SF100 ship-criterion: if query results mismatch the 1-GPU baseline byte-for-byte, suspect Pitfall 6 before suspecting Sirius logic. Run `lscpu | grep "Model name"` on the ship-gate host.

**Warning signs:** Query result counts right but aggregate values subtly off; 1-GPU and 2-GPU results differ only on aggregates that span many scan batches.

## Code Examples

Verified patterns from the current working tree (HIGH confidence — direct source reads).

### Current `lock_or_prepare_batch` body (FIX-01 caller site)

```cpp
// Source: src/include/pipeline/batch_lock_utils.hpp:48-125
inline std::optional<cucascade::data_batch_processing_handle> lock_or_prepare_batch(
  const std::shared_ptr<cucascade::data_batch>& batch,
  const cucascade::memory::memory_space* requested_memory_space,
  rmm::cuda_stream_view stream)
{
  // ... lock-attempt loop ...
  while (!lock_result.success && lock_result.status == status::memory_space_mismatch) {
    try {
      auto& registry = sirius::converter_registry::get();
      switch (target_space->get_tier()) {
        case cucascade::memory::Tier::GPU: {
          auto prev_state = batch->get_state();
          if (!batch->try_to_lock_for_in_transit()) { /* ... */ }
          try {
            batch->convert_to<cucascade::gpu_table_representation>(registry, target_space, stream);
          } catch (...) {
            batch->try_to_release_in_transit();
            throw;
          }
          // ...
```

**Phase 8 implication:** The `stream` here is passed in by `pipelineable_operator_data::prepare_for_processing` at `src/op/sirius_physical_operator.cpp:51` which got it from `gpu_pipeline_task::execute` at `src/pipeline/gpu_pipeline_task.cpp:280` — bound to `gpu_pipeline_executor._stream_pool` which IS the target GPU's pool (`src/pipeline/gpu_pipeline_executor.cpp:45`). So this caller's stream is correct. The bug is NOT in `lock_or_prepare_batch` itself — it's that the BATCH being converted was created on a scan-task stream bound to GPU 0, and cucascade's converter (specifically `convert_host_fast_to_gpu` or `convert_host_parquet_to_gpu_with_prefetched_data_source`) does in-flight work that touches the source-side stream's completion before the target-bound stream takes over. **FIX-01 closes this from the scan-executor side.**

### Current `duckdb_scan_executor` constructor (FIX-01 primary edit)

```cpp
// Source: src/op/scan/duckdb_scan_executor.cpp:43-59
duckdb_scan_executor::duckdb_scan_executor(...)
  : ...
{
  auto gpu_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  for (auto* space : gpu_spaces) {
    _gpu_memory_spaces.push_back(const_cast<cucascade::memory::memory_space*>(space));
  }
  // Keep backward compat: _gpu_memory_space points to first GPU for stream pool init
  _gpu_memory_space = _gpu_memory_spaces.empty() ? nullptr : _gpu_memory_spaces[0];
  _stream_pool      = std::make_unique<cucascade::memory::exclusive_stream_pool>(
    rmm::cuda_device_id(_gpu_memory_space->get_device_id()), config.num_threads);   // <-- BUG: always GPU 0
}
```

### Current scan-task dispatch (FIX-01 secondary edit)

```cpp
// Source: src/op/scan/duckdb_scan_executor.cpp:282-347
int target_gpu_id = select_target_gpu();  // :283 — may return GPU 1
// ... set_materialized_columns binds task to target_gpu_space (GPU 1's space) ...
auto exc_stream = _stream_pool->acquire_stream(
  cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);   // <-- BUG: pool is GPU 0's
_bounded_pool->dispatch(
  std::move(slot),
  [this, stream = std::move(exc_stream), /* ... */]() mutable {
    try {
      auto output_data = get_scan_output(scan_task, stream);   // stream on GPU 0, but allocation on GPU 1
      // ...
```

### `[mgpu-audit]` current emissions (AUDIT baseline)

```cpp
// Source: src/pipeline/pipeline_executor.cpp:246-249
SIRIUS_LOG_DEBUG("management_eventloop: routing task to GPU {}", target_device_id);
// v1.1 e2e verification audit: info-level dispatch log so a real SQL query
// can be grepped for per-GPU task distribution without needing debug logs.
SIRIUS_LOG_INFO("[mgpu-audit] pipeline_task dispatched to GPU {}", target_device_id);

// Source: src/op/scan/duckdb_scan_executor.cpp:180-185
// v1.1 e2e verification audit: info-level scan-batch assignment log
// so a real SQL query can be grepped for per-GPU batch distribution.
SIRIUS_LOG_INFO("[mgpu-audit] scan_batch assigned to GPU {} (available: {} bytes)",
                space->get_device_id(), space->get_available_memory());
```

### Integration test fixture (TEST-01 parameterization target)

```cpp
// Source: test/cpp/integration/test_gpu_execution_tpch.cpp:62-180 (abbreviated)
class GPUExecutionFixtureBase {
 public:
  GPUExecutionFixtureBase() {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";  // <-- num_gpus: 1 hardcoded
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<sirius_config_env_guard>(cfg_path.string());
      db = std::make_unique<duckdb::DuckDB>(nullptr);
      con = std::make_unique<duckdb::Connection>(*db);
    }
  }
  // ... compare_gpu_vs_cpu uses this con ...
};

// Source: test/cpp/unittest.cpp:104-108
auto integration_config_path = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" /
                               "integration" / "integration.yaml";
sirius::test::shared_test_env integration_env(integration_config_path);
integration_env.pause();
sirius::test::g_integration_env = &integration_env;
```

### Pattern 2 reference body (EXACT — reuse for W2 if needed)

```cpp
// Source: src/data/sirius_p2p_converter.cpp:33-126 (VERBATIM)
// The source of truth for Phase 8's Pattern 2 applications.
// Key lines:
//   L62: rmm::cuda_set_device_raii source_guard{rmm::cuda_device_id{source_device_id}};
//   L63: rmm::cuda_stream source_stream;  // bound to source_device (current)
//   L64: auto packed = cudf::pack(gpu_source.get_table(), source_stream.view());
//   L66: source_stream.synchronize();
//   L71: auto target_stream = target_memory_space->acquire_stream();
//   L76: rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{target_device_id}};
//   L77: rmm::device_uvector<uint8_t> dst_uvector(bytes_to_copy, target_stream, mr);
//   L89-95: cudaMemcpyPeerAsync(dst_uvector.data(), target_device_id,
//                               packed.gpu_data->data(), source_device_id,
//                               bytes_to_copy, target_stream.value());
//   L109: target_stream.synchronize();
//   L120: auto new_table = std::make_unique<cudf::table>(new_table_view, target_stream, mr);
//   L121: target_stream.synchronize();
```

## State of the Art

| Old Approach (v1.0 / pre-v1.1) | Current Approach (v1.1) | When Changed | Impact |
|--------------------------------|-------------------------|--------------|--------|
| Host-staged GPU→GPU copy (D2H + H2D) | `cudaMemcpyPeerAsync` via cucascade built-in converter | cucascade bump `f47de0b` (Phase 4) | Eliminates 2x pinned-host-buffer round-trip; enables real multi-GPU data movement |
| Single converter registry (cucascade built-ins) | Sirius-side override for `gpu_table_representation → gpu_table_representation` via `unregister + register` pattern | Plan 07-02 commit `18352b9` | Works around cucascade's cross-stream race on the GPU1→GPU0 return leg; cucascade pin preserved |
| `DEBUG`-level dispatch routing log | `INFO`-level `[mgpu-audit]` tagged log, greppable in production | v1.1 commit `fd24174` | Per-GPU task/batch distribution observable without recompile |
| Round-robin scan distribution | Memory-proportional distribution via `select_target_gpu()` | Plan 07-03 (MGPU-07) | Scan batches go to the GPU with most free memory; prevents OOM skew |
| Single scan-executor stream pool bound to GPU 0 | **NOT YET FIXED — Phase 8 target** | TBD (Phase 8 FIX-01) | When fixed: scan-task allocations + copies run on the correct per-target-GPU stream |
| Integration test suite hardcoded to `num_gpus: 1` | **NOT YET FIXED — Phase 8 target** | TBD (Phase 8 TEST-01..03) | When fixed: default `unit-tests` run exercises 2-GPU SQL path, catches regressions |

**Deprecated / outdated (for Phase 8 researchers):**
- The "convert to host then H2D to target GPU" shape (old cucascade pre-`f47de0b`) — do NOT reintroduce as a fallback. Use Pattern 2 directly.
- The v1.1 Phase 7 `CONDITIONAL OVERRIDE_REQUIRED = false` gate — Plan 07-02 evolved to `OVERRIDE-REGISTERED` based on direct N=2 evidence. Phase 8's W2 should similarly run the probe first and gate the override on concrete failure evidence, not on theoretical necessity.

## Open Questions

1. **Does cucascade's `convert_host_fast_to_gpu` actually race on 2-GPU?**
   - What we know: it sets `rmm::cuda_set_device_raii device_guard{target}` at `cucascade/src/data/representation_converter.cpp:837`. It uses the caller's `stream` (parameter) for `reconstruct_column` + `batch.flush(stream, ...)` + `cudf::table` construction + `stream.synchronize()`.
   - What's unclear: whether the caller's stream (which at `lock_or_prepare_batch` callsite is `gpu_pipeline_executor`'s target-bound stream — correct) is enough, OR whether FIX-01 (scan-executor stream fix) leaves a latent race when the source batch is host-resident and came from a scan-task on GPU 0's stream but now must be converted to target GPU 1's table.
   - Recommendation: FIX-02 MUST include a probe task: after FIX-01 is landed, run TPC-H Q1 on 2-GPU with materialized_columns (cache) path exercised. If `cuda_memcpy.cu` still appears in the log, author a Sirius-side `host_data_representation → gpu_table_representation` override. If not, FIX-02 is satisfied with documentation only ("Audited X sites; FIX-01 closes all known failure modes; no additional override needed").

2. **Does `duckdb_scan_task::make_data_batch` path (DuckDB-native, non-parquet) have the same issue?**
   - What we know: the v1.1 E2E verification only reproduced on parquet queries. `duckdb_scan_task::make_data_batch` at `src/op/scan/duckdb_scan_task.cpp:444` builds a host representation from DuckDB DataChunks; no GPU allocation.
   - What's unclear: Q1 on DuckDB-native (non-parquet) 2-GPU path has not been explicitly verified to pass.
   - Recommendation: include a compile-gate test that runs TPC-H Q1 on BOTH parquet and DuckDB-native fixtures at 2-GPU to confirm the fix is uniform.

3. **Does the AUDIT-01/02 "≥ 5 per GPU" threshold hold for SF1?**
   - What we know: v1.1 Run A (TPC-H SF10) showed GPU 0: 5 scan_batches, GPU 1: 3 pipeline_tasks. v1.1 Run B (SF1 trivial): GPU 0: 2 scan, GPU 1: 4 pipeline. Both fail the ≥5-on-both criterion.
   - What's unclear: whether Q1 on SF1 lineitem (600k rows, `scan_task_batch_size: 100000000` → few batches) reliably produces ≥5 scan_batches on each GPU.
   - Recommendation: The AUDIT TEST_CASE should use SF10 (3M rows in lineitem → ~30 batches at 100 MB batch size → ~15 per GPU assuming balanced) to comfortably exceed the ≥5 threshold. OR lower the threshold to `≥ 1 per GPU` for SF1 and keep the stricter ≥5 threshold only for the SF10 AUDIT variant. Plan task must calibrate by measurement before hardcoding the threshold.

4. **Per-GPU `exclusive_stream_pool` API — does `exclusive_stream_pool::acquire_stream()` internally do `cudaSetDevice`?**
   - What we know: cucascade's `exclusive_stream_pool` is constructed with `rmm::cuda_device_id` (`cucascade/include/cucascade/memory/stream_pool.hpp`). Streams are RMM-owned; `rmm::cuda_stream` constructor calls `cudaStreamCreate` which implicitly uses the current device. So the pool must construct its streams under the right device context.
   - What's unclear: whether `acquire_stream()` is safe to call from any thread regardless of that thread's current device.
   - Recommendation: wrap `pool.acquire_stream()` under `rmm::cuda_set_device_raii{target_device_id}` to be safe (no observable downside — RAII is cheap). Plan W1 already shows this pattern.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| N=2 GPU hardware (for SF100 ship gate) | ROADMAP criterion 1 | ✓ on verification host (2 × RTX 6000 Ada Gen, 49140 MiB each, driver 595.58.03, CUDA 13.2 per `.planning/milestones/v1.1-E2E-VERIFICATION.md`) | — | Worktree host (current dev host) has no GPU available per `nvidia-smi` probe during research; execute on verification host OR via MCP remote to verification host. **Worktree host CAN still compile-gate + run unit-tests that WARN+return on missing GPU** per Catch2-v2 convention. |
| `/datasets/tpch_parquet_sf100/lineitem.parquet` (22.8 GB) | ROADMAP criterion 1 | ✓ on verification host (per ROADMAP text "using `/datasets/tpch_parquet_sf100/lineitem.parquet` (22.8 GB, ~600M rows)") | — | If missing, ship gate slips to SF10 (Q1 at `/datasets/tpch_parquet_sf10/`); flagged in plan as risk. |
| `/datasets/tpch_parquet_sf10/` (for TEST-04) | TEST-04 | ✓ on verification host; also present under `test_datasets/tpch_parquet_sf10/` in worktree per `ls` probe | — | Bundled SF1 under `test/cpp/integration/data/parquet/` is always available; SF1 substitution degrades TEST-04 assertions to SF1-only. |
| CUDA 13.0 (for `nsys`, `cudaMemcpyBatchAsync`) | any CUDA API | ✓ (cucascade's `prefetched_data_source.cpp:152` already uses `cudaMemcpyBatchAsync`) | CUDA 13+ required per `pixi.toml` | No fallback — Phase 8 does not introduce new CUDA-13-specific APIs beyond what v1.1 uses. |
| MCP `mcp__project-commands__run_command` | all build/test | ✓ per user preference + prior commit evidence | — | `pixi run make` as local fallback (forbidden by user memory directive). |
| libcudf `cudf::pack` / `<cudf/contiguous_split.hpp>` | Pattern 2 body | ✓ (pixi env provides) | 26.04 | No fallback — Pattern 2 requires it. |
| Catch2 v2 with `REQUIRE_MESSAGE` | TEST / AUDIT assertions | ⚠️ partial — Plan 07-02 SUMMARY notes "If Catch2 v2 `REQUIRE_MESSAGE` is not available, use `INFO(...) + REQUIRE(...)` or `CAPTURE(..., ...) + REQUIRE(...)` instead" | v2.13.10 | Fall back to `INFO()+REQUIRE()` — known project convention, structurally equivalent for log grep. |

**Missing dependencies with no fallback:** None for code/build. For runtime ship gate: absence of N=2 hardware blocks ROADMAP criterion 6 (VALIDATION.md requires N=2 log capture); worktree host alone cannot discharge it.

**Missing dependencies with fallback:** SF100 dataset absence → SF10 substitution; full Catch2 `REQUIRE_MESSAGE` absence → `INFO()+REQUIRE()` substitution; nvidia-smi-available host → skip hardware verification probes, rely on MCP remote run.

## Project Constraints (from CLAUDE.md)

- **Branch policy:** main/default = `dev`. Phase 8 work should land on a feature branch (per user memory preference `feedback_feature_branches.md`); never merge directly to `dev`.
- **Build command:** `CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make` via `pixi shell`, OR `mcp__project-commands__run_command build` (preferred per user memory). If build fails with memory exhaustion, reduce parallelism: `CMAKE_BUILD_PARALLEL_LEVEL=8 make`.
- **Test commands:** `mcp__project-commands__run_command unit-tests` for Catch2 suite (preferred); `make test` for SQLLogicTest. Test logs at `build/release/extension/sirius/test/cpp/log`.
- **Code formatting:** `pre-commit run -a` before commits. C++/CUDA via clang-format (`.clang-format`); Python via black; CMake via cmake-format; spell check via codespell. `CUCASCADE_WARNINGS_AS_ERRORS=ON` inherited by Sirius for its own sources.
- **Super Sirius is the active engine.** Phase 8 targets `src/op/scan/`, `src/pipeline/`, `src/data/` (Super Sirius). DO NOT modify legacy `gpu_processing` / `namespace duckdb` code in `src/operator/`, `src/plan/`, `src/gpu_executor.cpp`.
- **Before implementing:** run `/module-context <task description>` to load cudf/rmm/cucascade/libkvikio API docs. Especially relevant for Phase 8 since FIX-01 touches `exclusive_stream_pool`, `cudaSetDevice`, `rmm::cuda_stream`, and potentially `cudf::io::read_parquet`.
- **Fallback behavior:** Sirius falls back to DuckDB CPU on OOM / unsupported operators / cudf row-count overflow. The integration test fixture `compare_gpu_vs_cpu` DISABLES fallback (`SET enable_duckdb_fallback = false;` at `test_gpu_execution_tpch.cpp:99`) so GPU errors are not silently hidden. Phase 8 must preserve this — the 2-GPU variant must fail loudly on bugs, not silently fall back.
- **CUDA architectures:** Turing through Blackwell (75-120). Per `pixi.toml`. Phase 8 code is arch-agnostic; no arch-specific kernels added.
- **Logging:** `SIRIUS_LOG_DIR` env var (default: `${CMAKE_BINARY_DIR}/log`); `SIRIUS_LOG_LEVEL` levels `trace debug info warn error`. Audit tests require `SIRIUS_LOG_LEVEL=info` (or default — INFO is emitted by default).
- **cuCascade submodule constraints** (from `cucascade/CLAUDE.md`): `idisk_io_backend` interface must not change; buffer registration cannot assume pre-registration; RAII patterns for file handles. **Phase 8 does not modify cucascade — constraints inherited transitively; relevant only if W2 converter override interacts with cucascade private symbols (`BatchCopyAccumulator`, `reconstruct_column`). Plan must verify these are public in `cucascade/include/cucascade/data/cpu_data_representation.hpp` or equivalent before depending on them.**
- **GSD workflow enforcement** (cucascade/CLAUDE.md): all file-changing tools must go through a GSD command. `/gsd:plan-phase 8` is the parent; this research is part of its integrated flow. No ad-hoc edits.

## Wave Dependency Graph (suggested for planner)

```
W0 (Wave 0 — prep, no-op if infra exists):
  - Verify /datasets/tpch_parquet_sf10/ and SF100 lineitem.parquet accessible on exec host
  - Verify compute-sanitizer + nsys available in pixi env
  - Verify Catch2 REQUIRE_MESSAGE available (fallback plan if not)

W1 (Wave 1 — FIX-01 primary fix, unblocks everything else):
  - Plan 08-01: duckdb_scan_executor per-GPU stream pool map + dispatch-site guard
    - Files: src/op/scan/duckdb_scan_executor.{hpp,cpp}
    - Acceptance: TPC-H Q1 SF1 parquet on num_gpus=2 via manual reproduction passes
  (Sequential — W1 must complete before W2 since W2's probe runs on W1's binary.)

W2 (Wave 2 — FIX-02 audit + conditional override):
  - Plan 08-02: FIX-02 probe task (run TPC-H Q1/Q3/Q6/Q12 on 2-GPU, capture any remaining cuda_memcpy.cu:* errors)
    - If clean: Plan 08-02 ends with documentation-only; FIX-02 satisfied by FIX-01 scope
    - If failing: Plan 08-02 extends with Sirius-side host→gpu converter override
    - Files: (conditional) src/data/sirius_host_to_gpu_converter.{hpp,cpp}, src/include/data/sirius_converter_registry.hpp, CMakeLists.txt
    - Acceptance: all probe queries pass on num_gpus=2; FIX-03 grep for cuda_stream_default returns 0 net-new

W3 (Wave 3 — TEST infrastructure + AUDIT + integration; can parallelize sub-plans):
  - Plan 08-03: TEST-01/02 — integration-2gpu.yaml + fixture parameterization (Option A: GENERATE())
    - Files: test/cpp/integration/integration-2gpu.yaml (new), test/cpp/integration/test_gpu_execution_tpch.cpp, test/cpp/utils/sirius_test_env.{hpp,cpp}, test/cpp/unittest.cpp
    - Acceptance: mcp unit-tests runs TPC-H TEST_CASEs twice, once per num_gpus
  - Plan 08-04: TEST-03/04 — SF1 full pass + SF10 subset (Q1/Q6/Q12)
    - Files: test/cpp/integration/test_gpu_execution_tpch.cpp (SF10 fixture gated on env var)
    - Acceptance: all 22 SF1 TPC-H queries GREEN on 2-GPU; Q1/Q6/Q12 SF10 GREEN on 2-GPU
  - Plan 08-05: AUDIT-01/02/03 — mgpu-audit dedicated TEST_CASE + log-grep assertions; extend log payload with task_id/batch_id
    - Files: src/pipeline/pipeline_executor.cpp, src/op/scan/duckdb_scan_executor.cpp, test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp (new)
    - Acceptance: TEST_CASE runs by default; on 2-GPU host asserts >=5 of each counter per GPU; on 1-GPU host WARN+returns
  (W3 sub-plans run sequentially only because they all touch test/cpp/integration/ and unittest.cpp; a single plan combining all W3 work is acceptable if granularity `coarse` suggests fewer plans.)

W4 (Wave 4 — validation):
  - Plan 08-06: SF100 Q1 ship-gate on N=2 hardware
    - Captures [mgpu-audit] full log, wall-clock, verifies no cudaErrorInvalidValue
    - Output: .planning/phases/08-multi-gpu-sql-pipeline-fix/08-06-VALIDATION.md (ROADMAP criterion 6)
    - Acceptance: SF100 Q1 completes on num_gpus=2, result matches num_gpus=1 baseline, [mgpu-audit] log shows scan_batch distribution on both GPUs
```

**Order rationale:**
- W1 before W2: FIX-02 can't probe effectively without FIX-01 landed.
- W3 before W4: ship gate SF100 run depends on AUDIT log extensions being in place.
- W3 sub-plans sequential: file conflicts on test/cpp/integration/test_gpu_execution_tpch.cpp prevent parallel edits.
- Given `granularity: coarse` in config.json, planner may collapse W3 into a single plan 08-03. Recommendation: keep W1 and W2 separate (they're clearly different concerns); W3 can be 1 or 3 plans at the planner's discretion. W4 is always its own plan (validation artifact, not code).

## Sources

### Primary (HIGH confidence)

- `src/op/scan/duckdb_scan_executor.cpp:43-347` — scan executor source, constructor + `select_target_gpu` + `manager_loop` (FIX-01 root cause)
- `src/op/scan/duckdb_scan_executor.hpp` — header (implied; Plan should read)
- `src/op/scan/parquet_scan_task.cpp:810-890` — parquet scan task body (materialized_columns path)
- `src/include/op/scan/parquet_scan_task.hpp:660-695` — `set_materialized_columns` signature + members
- `src/data/host_parquet_representation_converters.cpp:55-119` — `convert_host_parquet_to_gpu_with_prefetched_data_source`
- `src/include/pipeline/batch_lock_utils.hpp:48-125` — `lock_or_prepare_batch` body (FIX-01 caller site)
- `src/op/sirius_physical_operator.cpp:37-72` — `pipelineable_operator_data::prepare_for_processing`
- `src/pipeline/gpu_pipeline_task.cpp:280-360` — `gpu_pipeline_task::execute` (where stream originates)
- `src/pipeline/gpu_pipeline_executor.cpp:39-72, 222-223` — executor constructor + stream_pool init + dispatch site
- `src/pipeline/pipeline_executor.cpp:225-265` — management_eventloop + `[mgpu-audit]` emission site
- `src/data/sirius_p2p_converter.cpp:33-126` — Pattern 2 reference (VERBATIM reuse target)
- `src/include/data/sirius_converter_registry.hpp:51-82` — converter registration site with `[MGPU-06 P2P converter override]` comment
- `cucascade/src/data/representation_converter.cpp:820-856` — `convert_host_fast_to_gpu` (FIX-02 probe target)
- `.planning/milestones/v1.1-E2E-VERIFICATION.md` — 389-line primary evidence document; reproduction steps, per-GPU distribution tables, root-cause signal
- `.planning/STATE.md` lines 36-67 — accumulated findings + decisions from v1.1 close
- `.planning/ROADMAP.md` lines 29-40 — Phase 8 goal + 6 success criteria
- `.planning/REQUIREMENTS.md` lines 1-72 — FIX-01..04 / TEST-01..04 / AUDIT-01..03 full text
- `.planning/phases/07-p2p-direct-transfer-adaptive-scan-partitioning/07-02-SUMMARY.md:50-228` — Pattern 2 lessons-learned and override registration precedent
- `.planning/phases/07-p2p-direct-transfer-adaptive-scan-partitioning/07-RESEARCH.md:170-370` — Pattern 2 + Pattern 3 no-P2P fallback + Pitfalls 1-5
- `test/cpp/integration/integration.yaml` — current `num_gpus: 1` fixture
- `test/cpp/integration/test_gpu_execution_tpch.cpp:1-250, 3229-4130` — fixture class + TPC-H TEST_CASE inventory
- `test/cpp/utils/sirius_test_env.{hpp,cpp}` — `shared_test_env` pause/resume implementation
- `test/cpp/unittest.cpp:87-121` — test binary entry point wiring the shared integration env
- `CLAUDE.md` (project-root) — build system, test commands, architecture map
- `cucascade/CLAUDE.md` (attached via system-reminder) — submodule conventions, API stability, GSD workflow enforcement
- Commit `fd24174` (current HEAD) — `[mgpu-audit]` info-level dispatch logs baseline
- Commit `a708b0b` — v1.1 end-to-end verification + gap documentation
- Commit `18352b9` — Sirius-side P2P converter override (W2 template reference)

### Secondary (MEDIUM confidence)

- `/datasets/tpch_parquet_sf100/lineitem.parquet` size (22.8 GB, ~600M rows) — as cited in ROADMAP criterion 1; not independently measured during research
- Integration host CPU (6f7e4c9-lcedt, Intel Xeon, exact model unknown) — per v1.1 verification header; Pitfall 6 (Sapphire Rapids silent corruption) applicability pending `lscpu` on ship-gate host

### Tertiary (LOW confidence)

- cucascade `BatchCopyAccumulator` + `reconstruct_column` public API visibility — Plan W2 must verify at implementation time
- Exact line number `cuda_memcpy.cu:42` after future cudf bumps — fragile; use error-class + file match only

## Metadata

**Confidence breakdown:**

- Root cause location (`duckdb_scan_executor._stream_pool` bound to GPU 0): HIGH — direct source read at `:57-58` + `:322`, combined with v1.1 E2E verification logs showing 1 scan → GPU 0, 2 pipeline_tasks → GPU 1 distribution that fails at cuda_memcpy in non-trivial SQL
- Pattern 2 applicability (fix shape): HIGH — already proven in Plan 07-02 `src/data/sirius_p2p_converter.cpp` + `sirius_converter_registry.hpp` override
- FIX-02 scope (need for host→gpu override): MEDIUM — depends on FIX-01 effectiveness; probe required
- TEST-01/02 infrastructure path: HIGH — existing `shared_test_env::pause/resume` pattern is the direct blueprint
- AUDIT log-grep assertion shape: HIGH — existing `[mgpu-audit]` tag is the greppable baseline
- SF100 Q1 ship-gate feasibility: MEDIUM — depends on N=2 host availability + FIX-01 holding under 22.8 GB scan volume
- TPC-H Q4 flake handling: HIGH — STATE.md + ROADMAP explicit policy "retry once, not a regression"
- Wave ordering: HIGH — W1 → W2 → W3 → W4 is the only correct order given the probe dependency and file-conflict constraints

**Research date:** 2026-04-21
**Valid until:** 30 days for primary findings (code references stable); 7 days for environment availability (dataset paths on verification host may change)

## Validation Architecture

No Validation Architecture section needed — `.planning/config.json` has `workflow.nyquist_validation: false`, AND Phase 8's success criteria in ROADMAP.md are self-validating (they directly specify the exact grep commands, assertion counts, and ship-gate query). The planner can write plans that assert against ROADMAP's criteria 1-6 directly. Where genuine unknowns exist (e.g., FIX-02 probe outcome, AUDIT threshold calibration), they are surfaced in "Open Questions" above so the planner builds explicit probe tasks rather than assuming.
