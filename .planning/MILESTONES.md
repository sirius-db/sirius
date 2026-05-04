# Milestones

## v1.3 Multi-GPU Distribution (Shipped: 2026-05-01)

**Phases completed:** 4 phases (12-15), 12 plans

**Key accomplishments:**

- **Phase 12 — Small-sort `vector::at(2)` correctness fix** — `prepare_join_keys` at `src/op/sirius_physical_hash_join.cpp:622-637`; consumer-side guard against SORT-as-HASH_JOIN partitioner emitting stale `key_col_indices ≥ num_columns()`; new regression TEST_CASE locks the bug class. HYG-02 baseline preserved at 40.
- **Phase 13 — Q11 multi-GPU illegal-address closed** via cucascade-side stream-event lineage: writer event recorded at `gpu_table_representation` construction; `cudaStreamWaitEvent` issued in `convert_gpu_to_gpu` before peer copy. Path-2 architectural fix (compiler-enforced ctor signature requiring `writer_stream`) succeeded after Path-1 (per-site grep migration) left ~22 producers un-migrated. Cucascade pin advanced `e4db3d8 → 7409c60 → 62e0517`.
- **Phase 14 — SCHED-RR distribution landed.** `_gpu_executors` switched from `std::unordered_map` to `std::map` for deterministic iteration; `std::atomic<size_t> _no_pref_rr_counter` distributes preference-less source-pipeline tasks via `fetch_add modulo size + std::advance` in `task_scheduler::management_eventloop`; counter resets per-query in `prepare_for_query` so cache=table_gpu warm path stays reproducible.
- **Phase 15 — Cross-GPU operator-colocation audit.** 11 INVARIANT (SCHED-RR contract) comments across 9 operator files; per-site classification SAFE=11 NEEDS-PATCH=0 UNCLEAR=0 in `15-AUDIT-LOG.md`. Every audited site is downstream of `gpu_pipeline_task::execute → prepare_for_processing(target_space, stream)`, so `batches[0]->get_memory_space() == target_space` is invariant. New `[mgpu_stress]` test: 100 iterations × 5 representative `[mgpu]` queries × varied SCHED-RR counter offsets = 500 inner runs, 77053 assertions, exit 0 (test-only setter on `task_scheduler::_no_pref_rr_counter`). `docs/super-sirius/pipeline-execution.md` gained "Per-task-device contract under SCHED-RR" section.
- **Test gauntlet (post-merge state on `feature/single-node-multi-gpu2`):** `[mgpu]` 16/16 in 120.3s (79091 assertions, exit 0); `[TPC-H][parquet]` 22/22 in 81.6s; `[integration][TPC-H]` 48/48 in 2:43 (71608 assertions). HYG-02 = 40 (no regression).
- **FU-A done 2026-05-01:** `fix/order-small-sort-rangecheck` (Phase 12) merged into the v1.3 release-branch tip — `[mgpu]` lifted from 12/13 to 16/16.
- **Pre-requisites already on the branch (carried into v1.4):** `86e821a` (parquet AST filter re-translation per-device), `e2cf105` (mgpu test pipeline_id assertions), `ce8b426` (cucascade peer-DMA probe + client-side sort).

**Carried open:** FU-B (extend MCP wrapper for env-passthrough OR add `num_gpus` arg to `tpch-benchmark` to lift C3 SF1 1-vs-2-GPU >1.2× speedup gate from DEFERRED) — non-blocking for v1.4.

---

## v1.2 Multi-GPU SQL Pipeline Fix (Shipped: 2026-04-28)

**Phases completed:** 3 phases, 18 plans, 39 tasks

**Key accomplishments:**

- Replaced the single `_stream_pool` (bound to GPU 0) with an `unordered_map<int, unique_ptr<exclusive_stream_pool>>` keyed by device_id, and rewrote the scan dispatch site to acquire + dispatch under paired `rmm::cuda_set_device_raii` guards — closing FIX-01 at the build-gate level.
- Authored a Sirius-side `host_data_representation -> gpu_table_representation` converter override (`sirius_host_fast_to_gpu_factory`) acquiring a target-bound stream under target-device RAII, closing the v1.1 cross-device stream-correctness bug shape on the host->gpu converter frame — and surfacing a distinct fix-site (`host_parquet_representation` path via Sirius's OWN parquet converter) for handoff to 08-06.
- Closed the log-payload side of AUDIT-01/02/03 — both `[mgpu-audit]` INFO emissions now carry unique IDs (`task_id` and `batch_id`) so Plan 08-05's Catch2 audit TEST_CASE can count UNIQUE events per GPU via `grep + awk + sort -u`, robust against log-line duplication from retries.
- New fixture YAML
- Authored the Catch2 acceptance gate for Multi-GPU SQL: a dedicated `[mgpu-audit]` TEST_CASE that parses `sirius.log` and asserts per-GPU unique-ID counts for pipeline_task and scan_batch on BOTH GPU 0 and GPU 1 — plus TPC-H Q1/Q6/Q12 SF10 2-GPU variants gated on `SIRIUS_TEST_SF10_PATH` — plus a Q4-scoped one-shot retry per the ROADMAP flake policy. Runtime-verification of the full matrix (22 × {DuckDB,parquet} × {1,2} + SF10 + AUDIT) deferred to 08-06 because the MCP's hardcoded `--abort` flag trips on the known-open 08-06 host_parquet converter bug at test 610/983.
- Closed FIX-03 (HYG-02 grep) + FIX-04 (clean build) on the static side with an explicit PASS verdict. Applied the orchestrator-directed carryover fix to `src/data/host_parquet_representation_converters.cpp` (Pattern 2 idiom — target-bound stream + target-device RAII — mirroring 08-02 Branch B's template). Produced `08-06-VALIDATION.md` recording criterion-by-criterion verdicts. Criteria 1/2/4/6 remain DEFERRED because the same `cudaErrorInvalidValue @ cuda_memcpy.cu:42` signature persists on num_gpus=2 parquet TPC-H Q1 (and hive-partition filter) AFTER the carryover fix landed — the fix addressed one known site but at least one additional site remains. Per plan directive ('Don't over-author'), the residual is handed off with four hypothesis candidates + suggested next actions rather than chased.
- Added three grep-stable `[mgpu-probe]` INFO breadcrumbs at two frame boundaries on the num_gpus=2 parquet failure path (host_parquet converter entry+exit, parquet_scan_task::compute_task entry) so plan 08-08's MCP reproduction produces a deterministic payload discriminating hypotheses A/B/C/D from 08-VERIFICATION.md.
- Shipped (authored and code-review-verifiable):
- One-liner:
- One-liner:
- One-liner:
- 1. [Rule 3 - Blocking] MCP unit-tests wrapper does not pass agent shell env to child process
- Bisect of 5-commit Phase-9 source span (3b58258..c0e12f3) finds NONE regressing: all commits pass in isolation; SIGSEGV is test-ordering dependent, not commit-specific
- H1 confirmed: stream-ordered race in `sirius_physical_parquet_scan.cpp` — `rmm::cuda_stream_default` for filter expression translation races with `planning_stream` in `parquet_scan_task.cpp:492`
- Stream use-after-destroy SIGSEGV in parquet filter translation — fixed by moving `translation_stream` into `translated_expression::owned_stream` with correct C++ destruction order
- SF100 Q1 2-GPU ship-gate PASS (5.70s, byte-identical vs 1-GPU baseline); Phase 10 fix verification PASS (filter equality parquet + tpch_q1_sf10_2gpu both GREEN); one pre-existing [mgpu-audit] SIGSEGV prevents full suite exit 0 — PARTIAL verdict

---

## v1.1 Multi-GPU Re-integration + Cucascade I/O Migration (Shipped: 2026-04-21)

**Phases completed:** 4 phases, 19 plans, 44 tasks

**Key accomplishments:**

- Re-authored v1.0's NUMA-aware downgrade (dd86dd0 + 3 test commits) onto dev's POD `downgrade_task` + `downgrade_request` queue architecture — not a cherry-pick; new diffs express v1.0 intent against dev's shape while preserving original authorship attribution via commit messages.
- PORT-03 verified as a no-op for code-change purposes (grep gate returned 0 libconfig symbols); all multi-GPU settings v1.0 consumes are reachable through dev's YAML config reader; pre-commit run flagged 10 pure-formatting fixups (4 in-scope Phase 4 code files + 6 pre-existing drifting planning artifacts) committed as style(04-04) f5afde1. MCP build verification blocked by sandbox-caused 'Permission denied' on dependency-file writes — identical to Plan 04-01's documented executor-sandbox pattern; requires orchestrator-side build run.
- Full v1.0 test suite passes end-to-end on the bumped + re-integrated branch; all 4 PORT-05 visible tags verified to actually run (per-tag explicit invocation); 3 of 5 hidden tags PASS on the N=2 GPU verification host — the 2 failing hidden tags ([.][multi_gpu_transfer] + [.][mem_04_p2p_transfer]) are deferred to Phase 6 (MGPU-03 device guards) and Phase 7 (MGPU-06 P2P direct transfer) per scope boundary; all structural grep gates green; Phase 4 is shippable.
- cucascade submodule bumped from `942c0bf` to `f47de0b` (origin/main — PRs #96/#100/#103/#104 absorbed); the 23-commit v1.0 multi-GPU branch re-landed on top of dev's 47-commit drift via 5 cherry-picks + 3 test + 1 feat re-authored commits against dev's PR #579 POD `downgrade_task` shape; PORT-01..05 and BUMP-01..03 (all 8 Phase 4 requirements) cleared; full unit-test suite green; 3 of 5 hidden multi-GPU tests pass on N=2 verification host with 2 failures at the pre-documented Phase 6 (MGPU-03) / Phase 7 (MGPU-06) scope boundary — Phase 4 ships.
- TPC-H SF1 correctness baseline captured + sirius::io::cucascade_datasource header declared with supports_device_read()==false locked; build graph registers stub src/io/ + test/cpp/io/ ready for plan 05-02 implementation.
- cucascade_datasource now implements host_read + host_read_async + size() + supports_device_read()==false — delegates to cucascade::idisk_io_backend, returns pinned host buffers via cudaMallocHost, rejects remote URI schemes at construction, and launches async reads on std::launch::async (not deferred). 7 Catch2 TEST_CASEs with mock idisk_io_backend cover every public method.
- 1. [Rule 3 - Blocking] Stale test_gpu_expression_executor.cpp.o causing mold linker failure
- IO-05 landed for `parquet_scan_task.cpp` (2 of 3 call sites) + HYG-01 closed for this file. Approach C plumbing (task_creator seeds parquet_scan_task_global_state with the SiriusContext-owned gpu_io_backends map) — parquet scan tasks are now kvikio-free and construct `sirius::io::cucascade_datasource` adapters at planning time (first-available GPU backend) and per-task hot path (preferred_device_id with first-available fallback). Pure-consumer invariant on `src/include/sirius_context.hpp` upheld.
- 1. Header modifications (metadata scan + iceberg_scan_task.hpp)
- All 7 parquet-I/O call sites migrated from `cudf::io::datasource::create(path)` (kvikio-backed) to `sirius::io::cucascade_datasource` (pinned-host-staged, multi-GPU-safe). Per-GPU `idisk_io_backend` cache operational on `SiriusContext` under `rmm::cuda_set_device_raii`. HYG-01 closed + HYG-02 swept clean across all 15 Phase-5-modified files. 13 / 13 Phase 5 requirements cleared with real N=2 hardware evidence (2 × RTX 6000 Ada, driver 595.58.03, CUDA 13.2): compute-sanitizer memcheck 0 errors across 57 test cases / 1.92M assertions, per-GPU `cudaGetDevice` readback matches target device_id (GPU 0→0, GPU 1→1), SF10 TPC-H scan-bound queries return correct results on 1-GPU (Q1=1.273s, Q6=0.233s, Q12=0.717s) and 2-GPU (1.047s / 0.302s / 0.724s) configs. Human sign-off checkpoint: `approved` on 2026-04-21. Phase 5 SHIPS.
- SiriusContext::initialize() now throws on zero-GPU topology, emits info-level startup log summarising the cached cucascade topology, and logs host-space count vs NUMA node count — closing MGPU-01 and MGPU-05 without touching cucascade or re-discovering NVML.
- Two-line cudaSetDevice wrap: the only two raw cudaSetDevice callsites in Super Sirius noexcept per-thread init callbacks now log spdlog::error on failure instead of silently dropping to GPU 0.
- Two Catch2 tests added to test_context.cpp: a non-hidden registration-gate asserting cucascade's peer-async GPU↔GPU converter is exposed after sirius::converter_registry::initialize(), and a hidden [.]-tagged GPU0→GPU1 forward-leg round-trip that exercises the converter on N≥2 hosts.
- All 5 structural v1.0 multi-GPU gaps (topology discovery, single-GPU no-regression, device-guard enforcement, GPU↔GPU converter registration, per-NUMA host memory spaces) closed via audit-and-enforcement across 3 code plans + 1 validation plan. Zero new cucascade surface registered in Sirius — Phase 6 is a consumer phase. SiriusContext::initialize() now fail-harder on zero-GPU topology and emits MGPU-01 topology summary + MGPU-05 host-space audit logs. Both Super Sirius noexcept per-thread init callbacks (gpu_pipeline_executor + downgrade_executor) now check cudaSetDevice return and log spdlog::error on failure, giving MGPU-03 device-guard teeth. Cucascade's built-in peer-async GPU↔GPU converter verified registered after sirius::converter_registry::initialize() + forward-leg round-trip PASS (GPU0→GPU1 bytes-equal) on N=2 hardware (2 × RTX 6000 Ada, driver 595.58.03, CUDA 13.2). compute-sanitizer memcheck reports 0 errors across 49 test cases / 1.92M assertions on `[multi_gpu_foundation]` + `[integration][gpu_execution][parquet][join]` tags. MGPU-02 absolute Phase-6 SF10 timings captured on 1-GPU config; formal Phase-5-baseline regression comparison deferred per user directive 2026-04-21. Human sign-off checkpoint (Task 2b): `approved` on 2026-04-21. Phase 6 SHIPS — Phase 7 (MGPU-06 P2P direct + MGPU-07 adaptive scan) is unblocked.
- Bug surfaced during unit-tests:
- MCP build
- Close MGPU-07 end-to-end via test-only work: un-hide the Phase-4-deferred scan distribution placeholder, replace its body with a real asymmetric-memory fixture, and add an integration-level TEST_CASE exercising the adaptive-scan + P2P path.
- MGPU-06 closed end-to-end on real N=2 hardware: driver-level P2P peer access is enabled at SiriusContext::initialize() for every (i, j) GPU pair where cudaDeviceCanAccessPeer returns true (Plan 07-01); three previously-hidden MGPU-06 round-trip tests are un-hidden with FNV-1a checksum integrity guards and PASS including the GPU1 → GPU0 return leg that was Phase-4-deferred (Plan 07-02); and a Sirius-side P2P converter override (Pattern 2 from RESEARCH.md) replaces cucascade's cross-stream-race built-in body with a stream-correct peer-async-only implementation registered inside sirius::converter_registry::initialize() so the override covers both extension and test code paths (Plan 07-02 Task 3). MGPU-07 closed via test-only work — duckdb_scan_executor::select_target_gpu was already memory-proportional since Phase 2 v1.0, so Phase 7's MGPU-07 scope was authoring the asymmetric-memory fixture tests that prove batch-count skew ≥ 2× matches free-memory ratio within 10% (Plan 07-03). 979/979 unit tests PASS on the N=2 verification host (2 × RTX 6000 Ada, driver 595.58.03, CUDA 13.2, Intel Core Ultra 9 285K) with 78,789,847 assertions. Human sign-off Task 2a response: `approved with deferrals`. Phase 7 SHIPS. Milestone v1.1 CLOSES — 28/28 requirements complete across Phases 4+5+6+7.

---

## v1.0 — Multi-GPU Execution (Foundation + Scheduling + NUMA) *(unmerged baseline)*

**Branch:** `refs/remotes/felipe-ssh/feature/multi-gpu-execution` (23 commits not on `dev`)
**Status:** Implemented and tested, never landed on `dev`. Carried forward as the behavioral baseline this milestone re-integrates.
**Completed plans:** 5 / 7

### What shipped on that branch

- **Phase 1 — Multi-GPU Foundation** (3/3 plans)
  - Plan 01-01: NUMA-aware downgrade, multi-device terminate sync, P2P access enablement
  - Plan 01-02: Device-guard audit + multi-GPU foundation validation tests
  - Plan 01-03: NUMA-aware downgrade tests + GPU-to-GPU transfer validation
  - **Requirements cleared:** FOUND-02, FOUND-03, FOUND-05, CUCS-03, CUCS-04, MEM-03

- **Phase 2 — Data-Locality Task Scheduling** (2/2 plans)
  - Plan 02-01: Data-locality computation in `task_creator` + locality-aware routing in `management_eventloop` (push-model dispatch, `preferred_device_id` plumbing)
  - Plan 02-02: Cross-GPU scan distribution + integration tests
  - **Requirements cleared:** SCHED-01, SCHED-02, SCHED-03, SCHED-04, SCHED-05

- **Phase 3 — NUMA-Aware Memory + Transfer Optimization** (1/2 plans)
  - Plan 03-01: NUMA downgrade ordering verification (MEM-01, MEM-02)
  - Plan 03-02: P2P transfer + adaptive scan distribution — **PENDING** (MEM-04, MEM-05)

### Gaps left open

- **Not cleared:** FOUND-01 (runtime topology discovery), FOUND-04 (single-GPU no-regression), FOUND-06 (device-guard enforcement across all threads), CUCS-01 (GPU↔GPU converter registration), CUCS-02 (per-NUMA host allocator), MEM-04 (P2P direct), MEM-05 (adaptive scan).
- **Never merged to `dev`** — 47 commits landed on `dev` after the branch diverged (sirius-native types, YAML config, AST expression executor, hive partitioning, row group pruning).

### Why v1.0 didn't ship

Dev drift. The Sirius type system was refactored (`logical_type` / `type_id`), libconfig++ was replaced with YAML, DuckDB vocabulary types were removed from the core engine — touching nearly every file the multi-GPU work modified. Merging produced conflicts across ~20 files; a clean replay on top of `dev` is cheaper than conflict resolution.

---

## v1.1 — Multi-GPU Re-integration + Cucascade I/O Migration *(current)*

**Branch:** `feature/single-node-multi-gpu2`
**Status:** Initialized 2026-04-20
**Goal:** Land the v1.0 multi-GPU behavior on top of current `dev`, replace kvikio-backed parquet I/O with cucascade's pluggable io_backend, and bump cucascade to `origin/main`.

See: `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md`.

---

*Maintained at milestone boundaries. Current milestones live in `PROJECT.md` under Active / Out of Scope.*
