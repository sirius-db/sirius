# Sirius — GPU-Native SQL Engine (Multi-GPU)

## What This Is

Sirius is a GPU-native SQL engine that runs as a DuckDB extension (`sirius.duckdb_extension`). It intercepts DuckDB's physical plan and routes supported operators to GPU execution via cuDF / RMM / cuCascade, falling back to DuckDB's CPU engine for unsupported cases. As of v1.1, Sirius executes transparently across multiple GPUs on a single node, with data-locality-aware task scheduling, NUMA-aware memory management, driver-level P2P peer access, and a multi-GPU-safe parquet I/O path built on cucascade's `idisk_io_backend` (no kvikio).

## Core Value

Any query can transparently execute across every GPU on the node — tasks are scheduled to the GPU where their input data already resides, memory pressure is absorbed by downgrading to the correct NUMA domain, and parquet I/O is routed through a multi-GPU-safe backend.

## Current State

**Phase 24 shipped 2026-05-13** — Update cucascade + sirius from upstream (round 2).
- 5 plans / 3 requirements validated (MERGE-CC-24, MERGE-DEV-24, GAUNTLET-24) — all PASS on first attempt; no gap-closure plans needed
- Cucascade fork rebased onto `origin/main` HEAD `9ceebaa` (PR #124 "Fix for: Invalid Error: reconstruct_column STRING #124" and PR #122 "feat: adding the ability to slice host table" as `96bfea1`); single RE-DERIVE conflict on `representation_converter.cpp` (shared_ptr dereference + our `target_stream` preserved); 1 test-fix commit added for `96bfea1` writer_stream API mismatch. Fork now 9 commits ahead at `5203de5`
- Sirius `origin/dev` merged into `feature/single-node-multi-gpu2` — `ba5ed27` (wire_data_repositories Phase 2 split) + `2e197c6` (pin_table tier='host'); 9 conflict files resolved INTEGRATE BOTH (PIN-MGPU-01 GPU-tier round-robin path coexists with new host-tier path); D-05 gitlink ours-wins at `5203de5`; D-04 Commit D post-merge fix-up (stream_view arg)
- 18/18 invariant gates PASS: REG-01..06 (all 17 Phase 23 baseline gates) + D-07 new `[pin_table_host]` gate (1/1, 51 assertions, upstream test from `2e197c6`); HYG-02=40; kvikio-free=0; Cluster A=0; Cluster B=0; sanitizer_gate_22.sh P22_SELFTEST PASS
- Two improvements over Phase 23 baseline: REG-06 Leg 1 memcheck 6/7 PARTIAL → 7/7 PASS (cudf library violations absent); D-07 new gate (pin_table tier='host' smoke) 1/1 PASS via upstream test — D-04 Commit E not needed (Branch A)
- D-01 upstream-as-source-of-truth META-RULE application: 1 cucascade RE-DERIVE + INTEGRATE BOTH for sirius merge; biased toward upstream tighter than Phase 23's symmetric triage; fork count held at 9 (no commits dropped, +1 test-fix)
- CC-UPSTREAM-01 carry pattern continues: 9 commits ahead of `9ceebaa`; no upstream PRs submitted (user handles separately); `24-CUCASCADE-DIFF.md` documents per-commit notes and recommended upstream PR groupings
- Branch: `feature/single-node-multi-gpu2` (local-only; no `git push`; cucascade fork stays on local branch per CC-UPSTREAM-01 carry pattern)
- See: [`24-VERDICT.md`](phases/24-update-cucascade-and-sirius-from-upstream-round-2/24-VERDICT.md), [`24-CUCASCADE-DIFF.md`](phases/24-update-cucascade-and-sirius-from-upstream-round-2/24-CUCASCADE-DIFF.md), [`24-CONFLICT-LOG.md`](phases/24-update-cucascade-and-sirius-from-upstream-round-2/24-CONFLICT-LOG.md)

**Phase 23 shipped 2026-05-13** — Update cucascade + sirius from upstream.
- 7 plans / 3 requirements validated (MERGE-CC-23, MERGE-DEV-23, GAUNTLET-23) — initial 5 plans (rebase + merge + gauntlet) PARTIAL; gap-closure 2 plans (23-06 dst_guard + 23-07 gitlink/script/verdict) PASS
- Cucascade fork rebased onto `origin/main` HEAD `bcddb89` (PR #121 "Make host memory portable" supersedes our portable-pinning hunks of `6236494`); surgical-split kept 3 ours-only files (ptds tracker, pool peer access, `pipeline_io_backend` hygiene); 5 remaining commits re-applied on top. Fork now 8 commits ahead at `9da4047`
- Sirius `origin/dev` merged into `feature/single-node-multi-gpu2` at `49b7b86` — 12 upstream commits absorbed (`7eeaab4` value AST Phase 2, `7cc7a79` task-creation race fix, `972cb32` converter rename, `e94ad4a` per-op memory estimate, `5d09a59` bytes-to-materialize fix); 6 conflicts resolved with behavioral-correctness triage logged in `23-04-CONFLICT-LOG.md`
- Gap closure: REG-05 `[mgpu_stress]` + REG-06 Leg 1 `[multi_gpu_foundation]` first-pass FAIL with `cudaErrorInvalidValue` at `representation_converter.cpp:628`. Root cause was rebased commit `8392c3d` introducing `convert_gpu_to_gpu` → `reconstruct_column_p2p` → `alloc_and_peer_copy_async` codepath where the outer `target_guard{dst_device_id}` does not propagate into the inner host-staging branch. Fixed via cucascade `37df815` (Plan 23-06 RAII `dst_guard` around HtoD memcpy) + `9da4047` (Plan 23-07 in-flight `run_p2p_probe_locked` device-context restore)
- Sanitizer-gate hardening: `test/scripts/sanitizer_gate_22.sh` updated with windowed awk counter that distinguishes race-section headers from API-error backtraces; closes the cluster_B false-positive flagged in `23-VERDICT.md` Section J; `P22_SELFTEST=1` mode added with synthetic positive+negative test
- 17/17 invariant gates PASS on re-verification: REG-01..06 + GATE-22.1-A/B/C + K.6/K.7 NO-REPRO + Cluster B same-stream + HYG-02 + datasource_factory + tpch_sf10 + mgpu-audit. REG-05 77053 assertions; REG-06 Leg 1 7/7 functional; REG-06 Leg 2 first-run memcheck 42/42, 1.92M assertions, 0 new violations
- HYG-02 = 40 preserved (all in `src/legacy/`); kvikio-free = 0 (with `data_source.get()` exclusion filter, per Phase 22.1 strict policy)
- Side-benefit confirmed: upstream `7cc7a79` task-creation race fix closed the Phase 22.3 `pin_table` suite-run flake — `[mgpu-audit]` 6/6 PASS in suite mode (was flaky)
- Branch: `feature/single-node-multi-gpu2` (local-only; no `git push`; cucascade fork stays on local branch per CC-UPSTREAM-01 carry pattern)
- See: [`23-VERDICT.md`](phases/23-update-cucascade-and-sirius-from-upstream/23-VERDICT.md), [`23-VERIFICATION.md`](phases/23-update-cucascade-and-sirius-from-upstream/23-VERIFICATION.md), [`23-CUCASCADE-DIFF.md`](phases/23-update-cucascade-and-sirius-from-upstream/23-CUCASCADE-DIFF.md), [`23-04-CONFLICT-LOG.md`](phases/23-update-cucascade-and-sirius-from-upstream/23-04-CONFLICT-LOG.md)

**Phase 22.3 shipped 2026-05-08** — CTE `_types` cleanup + K.7 NO-REPRO reclassification + datasource_factory test alignment. K.7 reclassified NO-REPRO (SQL fixture used constant `0.0001` instead of spec-compliant `0.0001/SF`; at SF10+ that puts threshold above max single-partkey value, so 0 rows is correct — DuckDB CPU agrees). Shipped CTE planner `_types` cleanup as cosmetic correctness improvement + new SF10 Q11 mgpu regression test. HEAD `275fd11`. See [`22.3-VERDICT.md`](phases/22.3-fix-cte-types/22.3-VERDICT.md).

**Phase 22.2 shipped 2026-05-08** — K.6 closure: `cudaSetDevice(-1)` fix in `downgrade_executor` (HEAD `7dc47a2`). See [`22.2-VERDICT.md`](phases/22.2-fix-k6-downgrade-executor/22.2-VERDICT.md).

**Phase 22.1 shipped 2026-05-08** — Remove kvikio (post-v1.4 follow-up to Phase 22).
- 7 plans / 1 requirement validated (IO-MGPU-03 — kvikio fully removed from `src/`) + 1 partial (IO-MGPU-02 — kvikio half closed; per-GPU iceberg residency renamed as IO-MGPU-04)
- All 7 D-01 bypass sites migrated to `sirius_ioctx::make_datasource(uring_io_object)`: `sirius_gpu_parquet_scan_operator.cpp:126`, `sirius_extension.cpp:813`, `iceberg_metadata_reader.cpp:211+227`, `datasource_factory.cpp:110+125`, `parquet_split_provider.cpp:295`
- v1.4 ship-gate gauntlet (REG-01..06) re-passed against post-22.1 HEAD with no regression vs Phase 22 baseline: `[mgpu]` 16/16 (111.4s); `[TPC-H][parquet]` 22/22 (85.3s, no Q11 retry); `[integration][TPC-H]` 48/48 (163.4s); SF100 Q1 num_gpus=2 wall-clock 2.842s (vs 1-GPU 3.421s; 1.20× speedup); `[mgpu_stress]` 500-iter PASS (80.8s); HYG-02 = 40 (preserved); compute-sanitizer 0 violations on both legs
- 3 new Phase 22.1 gates PASS: GATE-22.1-A bypass-grep returns 0 hits; GATE-22.1-B sanitizer Cluster A = 0 (was 6 at Phase 22 baseline; K.1 closed); GATE-22.1-C SF1 Q11 num_gpus=2 functional no-regression (50 result rows + header, sorted-diff vs 1-GPU = identical, 0 cudaSetDevice(-1) errors)
- `test/scripts/sanitizer_gate_22.sh` updated to gate on Cluster A = 0 in addition to Cluster B = 0 (K.1 closure trajectory now grep-enforceable both at source-level via GATE-22.1-A and at runtime via GATE-22.1-B)
- Advisory K.6 SF100 Q11 num_gpus=2 RECORDED as FAIL — empirically proves K.6 is independent of kvikio (same `cudaSetDevice(-1) failed: invalid device ordinal` symptom in `downgrade_executor` per-thread init even with all 7 kvikio bypass sites removed). Root cause now narrowed to HOST-tier executors calling `cudaSetDevice` with sentinel `device_id=-1` from `downgrade_executor` worker-init configuration; estimated ~15-LOC fix in `src/downgrade/downgrade_executor.cpp`; targeted as Phase 22.2 or v1.6+ scope
- Cucascade gitlink unchanged at `c666b21` throughout 22.1 (kvikio is in cudf, not cucascade); `22.1-CUCASCADE-DIFF.md` documents zero-change attestation
- Branch: `feature/single-node-multi-gpu2` (local-only; no `git push`; no merge to dev)
- Carry-forwards to v1.6+: K.6 (SF100 Q11 num_gpus=2 narrowed to downgrade_executor worker-init; targeted as Phase 22.2), IO-MGPU-04 (multi-GPU iceberg metadata residency; renamed from IO-MGPU-02 per-GPU half), CC-UPSTREAM-01 (now 12 local cucascade fixes; unchanged from 22.1), PIN-MGPU-02/03 (adaptive pin distribution + HOST-tier path), OOM-RETRY-01 (retry budget restoration)
- See: [`22.1-VERDICT.md`](phases/22.1-remove-kvikio/22.1-VERDICT.md), [`22.1-CUCASCADE-DIFF.md`](phases/22.1-remove-kvikio/22.1-CUCASCADE-DIFF.md), [`22.1-07-SUMMARY.md`](phases/22.1-remove-kvikio/22.1-07-SUMMARY.md)

**Phase 22 shipped 2026-05-08** — Multi-GPU Pinning + Stream Lineage Hardening (post-v1.4).
- 7 plans / 2 requirements cleared (PIN-MGPU-01, fu17-cluster-b)
- Cucascade pin advanced from `1c1e648` → `c666b21` (descended from `1c1e648` via intermediate `42a01c4` clang-format cleanup; Plan 22-03 lands `alloc_and_peer_copy_async` same-stream invariant fix)
- PIN-MGPU-01: `PinTableFunction` distributes parquet chunks round-robin across GPU memory spaces (`idx % gpu_spaces.size()`); `pinned_entry::chunk_memory_spaces` vector parallel to `data_batches_by_column`; `cached_split_provider` per-chunk lookup; `[pin_mgpu]` Catch2 distribution + routing gates PASS
- fu17 Cluster B closed: cucascade `c666b21` collapses producer + DtoH leg + HtoD leg onto a single `target_stream` in `alloc_and_peer_copy_async`; sanitizer reports `cluster_B=0` post-fix at both SF1 and SF100 scale (was 16 pre-fix per `20-05-INVESTIGATION.md`)
- v1.4 ship-gate gauntlet (REG-01..06) re-passed against the bumped pin with no regression: `[mgpu]` 16/16 (110.6s); `[TPC-H][parquet]` 22/22 (85.2s); `[integration][TPC-H]` 48/48 (162.4s); SF100 Q1 num_gpus=2 wall-clock 2.807s (faster than Phase 21 baseline 3.150s); `[mgpu_stress]` 500-iter PASS (80.5s); HYG-02 = 40 (preserved); compute-sanitizer 0 violations
- Phase 22 NEW gates: GATE-07 PIN-MGPU-01 distribution PASS; GATE-08 PIN-MGPU-01 routing PASS; GATE-09 fu17 Cluster B sanitizer PASS
- Advisory (NOT GATING): SF100 Q11 num_gpus=2 sanitizer recorded — `cluster_B=0` even at SF100 scale; query-level fallback (`downgrade_executor cudaSetDevice(-1)` error) tracked as separate carry-forward (follow-up #17)
- Branch: `feature/single-node-multi-gpu2` (local-only; no `git push`; no merge to dev)
- Carry-forwards to v1.6+: CC-UPSTREAM-01 (12 local cucascade fixes; `22-CUCASCADE-DIFF.md` captures the diff for future upstream PR), PIN-MGPU-02 (adaptive free-memory-proportional pin distribution), PIN-MGPU-03 (HOST-tier pin path with NUMA-local round-robin), OOM-RETRY-01 (retry budget restoration 100 → 10), follow-up #17 (SF100 Q11 num_gpus=2 query-level fallback root-cause)
- See: [`22-VERDICT.md`](phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md), [`22-CUCASCADE-DIFF.md`](phases/22-multi-gpu-pinning-stream-lineage-hardening/22-CUCASCADE-DIFF.md), [`22-07-SUMMARY.md`](phases/22-multi-gpu-pinning-stream-lineage-hardening/22-07-SUMMARY.md)

**v1.4 shipped 2026-05-06** — Rebase After DataBatch Changes.
- 6 phases / 29 plans / 32 requirements cleared (CC-01..04 + MERGE-01..05 + DB-01..05 + IO-12..17 + IO-15B + SM-01..06 + REG-01..06)
- Cucascade rebased to pin `1c1e648` (descended from `73d00c4` with PR #117 DataBatch RAII + PR #112 bandwidth profiler + PR #116 `gpu_data_representation` from `cudf::table_view`; 11 local Sirius-side fixes preserved per CC-01..04)
- DataBatch RAII migration complete (Phase 18; Path A architectural fix in 18-07 — drop R5 lock-and-hold from `gpu_pipeline_task::compute_task`; 23 test files migrated; `[mgpu]` 16/16 PASS)
- IO Framework adoption complete (Phase 19; per-GPU `sirius_ioctx` + `sirius_datasource` retiring `cucascade_datasource`; per-GPU `uring_ioctx` instances under `rmm::cuda_set_device_raii`)
- Scan Manager + Pin Tables port complete (Phase 20; SM-06 SF1 closed via 20-06 `parquet_split_provider` kvikio bypass fix — Sirius-side architectural gap re-classified, NOT cucascade-side issue; IO-15B strengthened grep gate added)
- v1.4 ship-gate (Phase 21 REG-01..06) PASSED on rebased branch:
  - `[mgpu]` 16/16 PASS (79091 assertions, 106.3s)
  - `[TPC-H][parquet]` 22/22 PASS (36256 assertions, 79.3s)
  - `[integration][TPC-H]` 48/48 PASS (71607 assertions, 152.4s — 1-line SM-02 fixture fix at commit `9f835cd` realigned v1.3-era multi-pipeline_task threshold with post-#731 single composite `gpu_pipeline_task` pattern; cross-GPU `scan_id` intersection invariant preserved verbatim)
  - SF100 Q1 num_gpus=2 wall-clock 3.150s (vs 5.7s gate; vs 1-GPU 4.422s baseline; byte-identical CSV; pipeline_task distribution GPU0=18 / GPU1=12 / intersect=0)
  - `[mgpu_stress]` 500-iter PASS (77053 assertions, 76.7s)
  - HYG-02 = 40 (preserved — entirely in `src/legacy/`)
  - compute-sanitizer memcheck clean: Leg 1 [multi_gpu_foundation] 7/7 + 38 assertions + 0 violations; Leg 2 [integration][gpu_execution][parquet][join] 42/42 + 1.92M assertions + 0 violations
- Branch: `feature/single-node-multi-gpu2`
- Open follow-ups carried to v1.5+: PIN-MGPU-01 (multi-GPU pin_table), IO-MGPU-02 (multi-GPU iceberg metadata), CC-UPSTREAM-01 (upstream cucascade PRs), FU-B (SF1 1-GPU vs 2-GPU speedup gate)
- Cucascade peer-DMA probe + host-staging fallback (Cluster B from 20-05 sanitizer trace) tracked under existing project memory `project_tpch_q1_mgpu_string_bug` (correctness-neutral on this hardware; uncommitted in cucascade)
- See: [`21-VERDICT.md`](phases/21-v1-4-ship-gate-full-v1-3-gauntlet-on-rebased-branch/21-VERDICT.md), [`20-06-VERDICT.md`](phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-06-VERDICT.md), [`19-VERDICT.md`](phases/19-io-framework-adoption-pr-675/19-VERDICT.md), [`18-VERDICT-V2.md`](phases/18-databatch-raii-migration-cucascade-117-surface/18-VERDICT-V2.md)

**v1.3 shipped 2026-05-01** — Multi-GPU Distribution.
- 4 phases / 12 plans / 5 phases of work delivered (Phases 12-15)
- SCHED-RR round-robin distribution for preference-less source-pipeline tasks landed
- Q11 multi-GPU illegal-address closed via cucascade-side stream-event lineage (writer_event in `gpu_table_representation` ctor; `cudaStreamWaitEvent` in `convert_gpu_to_gpu`)
- Cross-GPU operator-colocation audit: 11 INVARIANT (SCHED-RR contract) comments across 9 operator files, per-site classification SAFE=11 NEEDS-PATCH=0 UNCLEAR=0
- New `[mgpu_stress]` test: 100 iterations × 5 [mgpu] queries × varied SCHED-RR counter offsets = 500 inner runs, 77053 assertions, exit 0
- Test gauntlet: `[mgpu]` 16/16 in 120.3s (79091 assertions); `[TPC-H][parquet]` 22/22 in 81.6s; `[integration][TPC-H]` 48/48 in 2:43 (71608 assertions); HYG-02 baseline preserved at 40
- Branch: `feature/single-node-multi-gpu2` (release tip; FU-A merged)
- Open follow-up carry: FU-B (extend MCP wrapper for env-passthrough OR add `num_gpus` arg to `tpch-benchmark` to lift C3 SF1 1-vs-2-GPU speedup gate from DEFERRED)

**v1.2 shipped 2026-04-28** — Multi-GPU SQL Pipeline Fix.
- 3 phases / 18 plans / 39 tasks / 11 v1.2 requirements satisfied (8 fully + 3 partial via proxy)
- TPC-H SF100 Q1 num_gpus=2: 5.70s wall-clock, byte-identical to 1-GPU baseline (5.45s); 71 scan batches distributed GPU0=42 / GPU1=29 with cross-GPU intersection=0
- HYG-02 improved 41 → 40 (`rmm::cuda_stream_default` count) via Phase 10-03 stream-use-after-destroy fix
- Branch: `feature/single-node-multi-gpu2`
- Archive: `.planning/milestones/v1.2-*`
- Open Phase-11 candidate: pre-existing `[mgpu-audit]` SIGSEGV at `test_gpu_execution_tpch_mgpu_audit.cpp:200` in the `attach_integration_duckdb` path (orthogonal to v1.2 fixes; documented in `v1.2-MILESTONE-AUDIT.md`)

**v1.1 shipped 2026-04-21** — Multi-GPU Re-integration + Cucascade I/O Migration.
- 4 phases / 19 plans / 44 tasks / 28 requirements cleared
- Full test suite: 979/979 pass on N=2 hardware (2× RTX 6000 Ada, driver 595.58.03, CUDA 13.2)
- Archive: `.planning/milestones/v1.1-*`

## Current Milestone: v1.4 Rebase After DataBatch Changes

**Goal:** Land cucascade `origin/main` (PR #117 DataBatch RAII refactor + #112 + #116) and Sirius `origin/dev` (#675 IO Framework, #731 Scan Manager, #721 Pin Tables, #739 cucascade-compat, #733/#734/#735) onto `feature/single-node-multi-gpu2` while preserving every v1.1+v1.2+v1.3 multi-GPU behavior.

**Target features:**
- Cucascade rebase: 11 local Sirius-side fixes (writer_stream/writer_event, peer-DMA probe, io_worker member-order, pinned-host Portable/Mapped, ptds tracker, pool peer access, cudf::pack stream) re-applied on top of cucascade `origin/main` against PR #117's RAII accessor model (`read_only_data_batch` / `mutable_data_batch`). New pin descended from `73d00c4`.
- DataBatch API migration: ~12 Sirius operators + ~16 tests adapted from the pre-#117 `gpu_data_representation` shape to the post-#117 RAII-locked accessors. Mechanical-but-deep migration of every site that reads or mutates batch data.
- IO Framework adoption: retire stopgap `sirius::io::cucascade_datasource` and adopt `sirius::io::sirius_datasource` (#675 — uring reactor + prefetching cache + admission control), adapting it to multi-GPU (per-GPU reactor pools / per-GPU prefetching-cache scoping / `cudaSetDevice` RAII at device-read sites).
- Scan Manager + Pin Tables integration: port Phase 14 SCHED-RR distribution, Phase 9 `_batch_gpu_affinity`, MGPU-07 adaptive scan, Phase 13 stream-lineage (currently in deleted `sirius_parquet_metadata_scan_operator.hpp`) into `parquet_split_provider` / `sirius_scan_manager` / `split_connector`.
- Regression preservation: v1.3 ship-gate passes on the rebased shape — `[mgpu]` 16/16, `[TPC-H][parquet]` 22/22, `[integration][TPC-H]` 48/48, SF100 Q1 num_gpus=2 ≤ 5.7s, mgpu_stress 500-iter, HYG-02 ≤ 40.

**Key context:** Phase numbering continues from 16 (v1.3 ended at 15). All work in-place on `feature/single-node-multi-gpu2`. No upstream cucascade PRs this milestone — local pin carries the 11 fixes. v1.3 FU-B (SF1 1-vs-2-GPU speedup gate from DEFERRED) is carried forward; not blocking v1.4 ship.

## Requirements

### Validated

Shipped and validated in v1.1.

- ✓ **BUMP-01..03** — cucascade submodule bumped 942c0bf → f47de0b (PR #96 file-downgrade + io_backend_registry, PR #100 underflow fix, PR #103 stream sync, PR #104 NVML drop) — *v1.1*
- ✓ **PORT-01..05** — 23 v1.0 multi-GPU commits re-landed on current `dev`; push-model dispatch + `preferred_device_id` plumbing preserved; YAML config replaces libconfig++ — *v1.1*
- ✓ **IO-01..11** — `sirius::io::cucascade_datasource` replaces every `cudf::io::datasource::create(path)` call-site; per-GPU `idisk_io_backend` cache on `SiriusContext` under `rmm::cuda_set_device_raii`; kvikio removed; SF1 correctness preserved — *v1.1*
- ✓ **MGPU-01** — runtime topology discovery via cucascade (fail-hard on zero-GPU; 3-line startup log) — *v1.1*
- ✓ **MGPU-02** — single-GPU SF10 no-regression (absolute Phase-6 timings captured) — *v1.1*
- ✓ **MGPU-03** — device-guard `cudaSetDevice` enforcement with `spdlog::error` in both `noexcept` per-thread init callbacks; compute-sanitizer memcheck 0 errors — *v1.1*
- ✓ **MGPU-04** — GPU↔GPU converter registered in `sirius::converter_registry::initialize()`; forward-leg + return-leg round-trip PASS on N=2 — *v1.1*
- ✓ **MGPU-05** — per-NUMA host memory spaces via `numa_region_pinned_host_allocator` — *v1.1*
- ✓ **MGPU-06** — GPU-direct P2P via `cudaMemcpyPeerAsync`; `cudaDeviceEnablePeerAccess` loop at init; Sirius-side `sirius_p2p_converter_factory` override works around cucascade's cross-stream race — *v1.1*
- ✓ **MGPU-07** — adaptive scan partitioning proportional to free GPU memory (3.08× ratio → batch-count skew within 10% tolerance) — *v1.1*
- ✓ **HYG-01/02** — `rmm::cuda_stream_default` removed from `parquet_scan_task.cpp:468` and every Phase-5-modified file — *v1.1*

### Active

<!-- v1.4 requirements scoped during this /gsd:new-milestone run. See REQUIREMENTS.md once written. -->

(No active requirements yet — `/gsd:new-milestone` mid-flight; REQUIREMENTS.md is written next.)

**Phase 22.1 deliverables (now Validated):**
- ✓ **IO-MGPU-03** — Remove all kvikio usage from `src/`. All 7 D-01 bypass sites migrated to `sirius_ioctx::make_datasource(uring_io_object)`. GATE-22.1-A bypass-grep returns 0 hits across `src/` for both `cudf::io::datasource::create` and pointer-form `cudf::io::source_info{<path>}` (excluding pointer-form `datasource.get()` and code comments). GATE-22.1-B sanitizer Cluster A = 0 at SF1 Q11 num_gpus=2 (was 6 at Phase 22 baseline; K.1 closed). GATE-22.1-C SF1 Q11 num_gpus=2 functional no-regression PASS. v1.4 ship-gate REG-01..06 re-passes without regression. `test/scripts/sanitizer_gate_22.sh` now gates Cluster A = 0 in addition to Cluster B = 0 — *Phase 22.1 (v1.5+ scope)*

**Phase 22 deliverables (now Validated):**
- ✓ **PIN-MGPU-01** — Multi-GPU-aware `pin_table` (round-robin across GPU memory spaces). `PinTableFunction` distributes parquet chunks via `idx % gpu_spaces.size()` per-call counter + per-file `rmm::cuda_set_device_raii` guard around `chunked_parquet_reader`. `pinned_entry::chunk_memory_spaces` vector + `get_pinned_entries()` accessor. `cached_split_provider` per-chunk memory_space lookup. `[pin_mgpu]` Catch2 distribution + routing gates PASS — *Phase 22 (v1.5+ scope)*
- ✓ **fu17-cluster-b** — Cucascade `alloc_and_peer_copy_async` host-staging fallback same-stream invariant fix. Cucascade pin advanced `1c1e648` → `c666b21`. Sanitizer `cluster_B=0` post-fix at both SF1 and SF100 scale (was 16 pre-fix). Closes the post-Phase-13 fallback path race that 20-05-INVESTIGATION.md surfaced — *Phase 22 (v1.5+ scope)*

**v1.3 deliverables (now Validated):**
- ✓ **SORT-01** — Phase 12 small-sort `vector::at(2)` correctness fix in `prepare_join_keys` (`src/op/sirius_physical_hash_join.cpp:622-637`); consumer-side guard against SORT-as-HASH_JOIN partitioner emitting stale `key_col_indices` ≥ `num_columns()`; regression TEST_CASE locks the bug class — *v1.3*
- ✓ **Q11-01** — Phase 13 Q11 multi-GPU illegal-address closed via cucascade-side stream-event lineage (`writer_event` recorded at `gpu_table_representation` ctor; `cudaStreamWaitEvent` in `convert_gpu_to_gpu` before peer copy). Path-2 architectural fix (compiler-enforced ctor signature requiring `writer_stream`). Cucascade pin advanced to `62e0517` — *v1.3*
- ✓ **SCHED-RR-01** — Phase 14 round-robin distribution for preference-less source-pipeline tasks. `_gpu_executors` switched to `std::map`; `std::atomic<size_t> _no_pref_rr_counter` distributes via `fetch_add modulo size + std::advance` in `task_scheduler::management_eventloop`; counter resets per-query for cache=table_gpu warm-path reproducibility — *v1.3*
- ✓ **AUDIT-MGPU-01** — Phase 15 cross-GPU operator-colocation audit: 11 INVARIANT (SCHED-RR contract) comments across 9 operator files; per-site classification SAFE=11 NEEDS-PATCH=0 UNCLEAR=0; new `[mgpu_stress]` test (500 inner runs, 77053 assertions, exit 0); `docs/super-sirius/pipeline-execution.md` "Per-task-device contract under SCHED-RR" section — *v1.3*

**v1.2 deliverables (now Validated):**
- ✓ **FIX-01..04** — cross-device stream-correctness fixes (Pattern 2 idiom): per-GPU stream pool in `duckdb_scan_executor`, Sirius-side `host→gpu` converter override, per-GPU filter translation at plan time, `translated_expression::owned_stream` for scalar lifetime correctness — *v1.2*
- ✓ **TEST-01..04** — TPC-H integration parameterized on `num_gpus∈{1,2}` via Catch2 GENERATE; `integration-2gpu.yaml` fixture; SF1 22 queries × {1,2} GPUs all PASS; SF10 Q1/Q6/Q12 PASS — *v1.2*
- ✓ **AUDIT-01..03** — `[mgpu-audit]` payload extended with `task_id`/`batch_id`; AUDIT TEST_CASE wired in default unit-tests run; Phase 9 disjointedness REQUIRE (`std::set_intersection(scan_ids) == ∅`) fires in `tpch_q1_sf10_2gpu` — *v1.2 (canonical TEST_CASE blocked by pre-existing SIGSEGV; substantive evidence via SF100 + SF10 proxy runs)*

## Deferred to Future Milestones

- **K.6 — SF100 Q11 num_gpus=2 query-level fallback** *(narrowed 2026-05-08 in Phase 22.1; targeted as Phase 22.2 or v1.6+)*: Phase 22.1 advisory check confirmed K.6 is independent of kvikio (Phase 22 carry-forward closed for kvikio cause; symptom unchanged). Root cause: HOST-tier executors call `cudaSetDevice` with sentinel `device_id=-1` in `downgrade_executor` per-thread initialization, leading to `cudaSetDevice(-1) failed: invalid device ordinal` then drain-after-error returning 0 rows. Estimated ~15-LOC fix in `src/downgrade/downgrade_executor.cpp` worker-init device-ID configuration. Tracked under project memory `project_phase08_fu17` since v1.2 Phase 8.
- **IO-MGPU-04 — Multi-GPU iceberg metadata + equality-delete reads** *(renamed from IO-MGPU-02 per-GPU residency half after Phase 22.1; targeted as v1.6+)*: Phase 22.1 closed the kvikio-bypass half of IO-MGPU-02 by routing iceberg sites through GPU 0 sirius_ioctx (single-GPU correct). Multi-GPU residency would route iceberg metadata/delete reads to the consumer's preferred device. Trade-off identical to PIN-MGPU-01: kvikio's single-CUDA-context binding currently poses no correctness risk because these reads are not on the multi-GPU column-chunk hot path; perf gap is negligible.
- **`pin_table` single-GPU residency** *(closed 2026-05-08 in Phase 22 — PIN-MGPU-01 validated)*: `CALL pin_table(...)` now distributes parquet chunks round-robin across all available GPU memory spaces (`idx % gpu_spaces.size()` in `PinTableFunction`); `pinned_entry::chunk_memory_spaces` vector preserves per-chunk residency information end-to-end through `cached_split_provider`. Adaptive (free-memory-proportional) variant tracked as PIN-MGPU-02 in v1.6+ scope. HOST-tier `pin_table` (NUMA-local round-robin) tracked as PIN-MGPU-03.
- **`[mgpu-audit]` per-GPU distribution AUDIT TEST_CASE SIGSEGV** at `test_gpu_execution_tpch_mgpu_audit.cpp:200` (`attach_integration_duckdb` path; pre-existing on base before v1.2; orthogonal to parquet filter translation path; Phase 11 candidate, < 50 LOC expected)
- Upstream cucascade `convert_gpu_to_gpu` cross-stream fix (drop Sirius override once upstream lands)
- Phase-5 vs Phase-4 parquet I/O regression comparison
- Phase-6 vs Phase-5 single-GPU SF10 regression comparison
- Cucascade `idisk_io_backend` file-handle cache (research pitfall P1)
- `cudaDeviceDisablePeerAccess` on explicit teardown
- TPC-H Q4 parquet intermittent flake investigation

### Out of Scope

- **Distributed multi-node execution** — different problem domain (network serialization, fault tolerance).
- **GPU-Direct RDMA (network GDS)** — only relevant for multi-node; would re-introduce kvikio/GDS dependency we're removing.
- **KvikIO / cuFile backends for parquet** — explicitly replaced by cucascade io_backend; per-GPU CUDA-context scoping makes kvikio unsafe for multi-GPU scheduling.
- **Heterogeneous GPUs** — assume homogeneous GPUs (DGX/HGX configurations).
- **Query-optimizer-level GPU placement** — routing happens at task dispatch with actual data sizes, not plan time with estimates.
- **Data repartitioning / shuffle exchange** — single-node batch-level scheduling avoids global shuffle.
- **Changes to legacy `namespace duckdb` code path** — multi-GPU targets Super Sirius (`namespace sirius`) only.

## Context

- **Worktree branch:** `feature/single-node-multi-gpu2` (fresh worktree with no prior `.planning/`). Sibling to `feature/single-node-multi-gpu` which is at `dev` head.
- **Prior work:** `refs/remotes/felipe-ssh/feature/multi-gpu-execution` implemented Phases 1–3 (partial) of multi-GPU execution, landed 23 commits that never merged to `dev`.
- **Dev drift:** 47 commits on `dev` since the multi-gpu branch diverged, including: sirius-native type system (PR #643), YAML config replacing libconfig++ (#565), hive partition columns (#570), AST expression executor (#531), refactors removing DuckDB vocabulary types (#564/#626/#628), row group pruning (#363).
- **cucascade:** pinned to 942c0bf in the worktree; PR #96 (`Feature/file downgrade`) introduced `disk_io_backend`, `io_backend_registry`, `disk_data_representation`, `disk_file_format`. Additional commits on `origin/main` through f47de0b (NVML drop, stream sync, benchmark bump).
- **Parquet I/O surface in src/:** `hybrid_scan_reader` (used in `host_parquet_representation.{hpp,cpp}`, `host_parquet_representation_converters.cpp`) and direct `cudf::io::parquet_reader_options` in `op/scan/{parquet,iceberg}_scan_task.cpp` and `sirius_parquet_metadata_scan_operator.cpp`. cuDF internally uses kvikio for GPU-direct storage when available.

## Constraints

- **Tech stack:** CUDA 13+, C++20, CUDA std 20, separable compilation. GPU arches 75–120 (Turing → Blackwell).
- **Build:** pixi-driven, `pixi run make -jN`. Never use `pixi run` directly from Claude — route through `mcp__project-commands__run_command` per user preference.
- **Streams:** No `rmm::cuda_stream_default` — every allocation/copy/kernel uses an explicit stream (user rule).
- **cuCascade API:** All disk I/O and tier conversion must go through cucascade's converter + io_backend registries. No hand-rolled kvikio/cuFile/GDS calls anywhere in `src/`.
- **Super Sirius only:** Multi-GPU work targets `namespace sirius`. Legacy `gpu_processing` path (`namespace duckdb`) is frozen.
- **Fallback-first:** Any GPU path that can't run multi-GPU-safely must downgrade through the existing fallback mechanism, not crash.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace kvikio with cucascade io_backend | kvikio/cuFile bind to a single CUDA context → unsafe for multi-GPU task dispatch. | ✓ Good — v1.1 shipped; `grep -rnw 'datasource::create' src/` returns zero hits |
| Cucascade pinned to `origin/main` (f47de0b) | PR #96 introduces the `idisk_io_backend` + `io_backend_registry` Sirius depends on. | ✓ Good — all 28 v1.1 requirements validated on this pin |
| Re-integrate as a new milestone (v1.1), not merge | 47 dev commits include type-system and config refactors — fresh plan-by-plan replay cheaper than conflict resolution. | ✓ Good — v1.1 shipped in 4 phases; replay strategy validated |
| Push-model task dispatch for locality routing | Pull model couldn't use data-locality info; push (pop task first, route by preferred_device_id) enables SCHED-01..04. | ✓ Good — validated end-to-end in v1.1 |
| preferred_device_id on both local_state + global_state | Per-task override with pipeline-level default covers both scan distribution and inherited locality. | ✓ Good — v1.1 integration tests PASS |
| NUMA-aware downgrade via cucascade `any_memory_space_in_tier_with_preference` | Avoids bespoke NUMA logic in Sirius; cucascade owns tier selection. | ✓ Good — v1.1 re-authored onto dev's PR #579 shape; downgrade tests PASS on N=2 |
| KvikIO/GDS explicitly out of scope | Removing the dependency was the milestone goal; re-adding for any path defeats the purpose. | ✓ Good — v1.1 grep gate enforced |
| Sirius-side `sirius_p2p_converter` override for GPU↔GPU | cucascade's `convert_gpu_to_gpu` has a cross-stream race (`cudaMemcpyPeerAsync` on caller stream vs post-copy table on target_stream). Override issues peer copy on target_stream. | ⚠️ Revisit — works around upstream; upstream PR is tech debt in v1.2 |
| Consume `cudaGetLastError()` after `cudaDeviceEnablePeerAccess` | CUDA leaves the return code in thread-local error slot; subsequent unrelated calls fail spuriously with same code. | ✓ Good — pattern established for future CUDA-state-mutation code |
| `supports_device_read() == false` in cucascade_datasource | Host-stage via pinned memory + `cuda_memcpy_async` on caller's stream stays truly async and avoids GDS entirely. | ✓ Good — v1.1 IO-02/03 validated |
| Adaptive scan via existing `select_target_gpu` (no code change needed) | `duckdb_scan_executor::select_target_gpu` was already memory-proportional since v1.0 Phase 2; Phase 7 MGPU-07 scope was test-authoring only. | ✓ Good — 3.08× free-memory ratio test proves proportional skew within 10% tolerance |
| Per-GPU filter translation at plan time (Phase 8 residual closure 93fea6f) | `sirius_physical_parquet_scan` originally translated DuckDB filter expressions ONCE, binding scalars to the planner's current device. Tasks dispatched to other GPUs faulted. Build one tree per configured GPU at plan time, select per-task at converter time. | ✓ Good — closes the v1.2 ship-blocker on parquet TPC-H Q1 num_gpus=2 |
| `_batch_gpu_affinity` map records ownership but does NOT consult at dispatch time (Phase 9 minimum-viable) | Recording is sufficient for the disjointedness REQUIRE regression gate; consultation-at-dispatch was deferred to keep scope tight. Affinity is implicitly preserved because `_scan_round_robin` is monotonic. | ✓ Good — disjointedness REQUIRE fires green at SF10 + SF100; cross-GPU intersection=0 |
| `translated_expression::owned_stream` declared BEFORE `owned_literals` (Phase 10-03) | C++ reverse-destruction order: scalars `cudaFreeAsync` first (using stream handle), then stream destroys. Without this ordering, `cudaFreeAsync(ptr, stale_handle)` SIGSEGVs at next QueryBegin. | ✓ Good — closes the test-ordering-dependent SIGSEGV that 09-04 exposed; HYG-02 improved 41→40 |
| Run all integration/SF100 tests via MCP on this host (no human-delegated checkpoints) | 2026-04-24 host-capability discovery: `mcp__project-commands__run_command nvidia-smi` shows 2× RTX 6000 Ada visible; agent can run the full v1.2 ship-gate autonomously. | ✓ Good — Phase 9-04 + Phase 10-04 ship-gates ran fully autonomously via MCP |
| v1.4 cucascade strategy: rebase 11 local fixes onto `origin/main` (no upstream PRs this milestone) | Upstreaming would block the rebase on review cycles; PR #117 already touches the same surface as our writer_stream/event work, so the conflict resolution is the same regardless. Carry the local pin and revisit upstreaming separately. | Pending v1.4 |
| v1.4 IO Framework: retire `sirius::io::cucascade_datasource`, adopt `sirius::io::sirius_datasource` (#675) and adapt to multi-GPU | The cucascade_datasource was a v1.1 stopgap to dodge kvikio's single-CUDA-context binding; the new sirius_datasource (uring + prefetching cache + admission control) is the going-forward I/O surface. Adapting it for multi-GPU (per-GPU reactor pools, cudaSetDevice RAII) is cheaper than maintaining two parallel datasources. | Pending v1.4 |
| v1.4 in-place rebase on `feature/single-node-multi-gpu2` (no fresh branch) | Branch already carries v1.1+v1.2+v1.3 history and the cucascade pin with 11 local fixes; cutting a fresh branch would lose merge history. Phase numbering continues from 16. | Pending v1.4 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-13 — Phase 23 (Update cucascade + sirius from upstream) shipped on `feature/single-node-multi-gpu2`. MERGE-CC-23, MERGE-DEV-23, GAUNTLET-23 all promoted to Validated. Cucascade fork rebased onto `origin/main` HEAD `bcddb89` (PR #121) with surgical-split of `6236494`; 5 remaining fork commits re-applied. Sirius `origin/dev` merged at `49b7b86` — 12 upstream commits absorbed; 6 conflicts resolved with behavioral-correctness triage. First-pass PARTIAL (REG-05/REG-06 `cudaErrorInvalidValue` regression from rebased `8392c3d`); closed via cucascade `37df815` (dst_guard) + `9da4047` (probe-device-restore). Cucascade gitlink advanced `c666b21` → `9da4047` (8 commits ahead of upstream). `sanitizer_gate_22.sh` cluster_B false-positive closed via windowed awk counter + `P22_SELFTEST` self-test. 17/17 invariant gates PASS on re-verification; REG-05 77053 assertions; REG-06 Leg 1 7/7 functional; REG-06 Leg 2 memcheck 42/42 (1.92M assertions, 0 new violations). HYG-02 = 40 preserved; kvikio-free = 0. Side-benefit: upstream `7cc7a79` closed Phase 22.3 `pin_table` suite-run flake. Branch local-only; no `git push`; cucascade fork local-only per CC-UPSTREAM-01. Phase 23 verdict: PASS. See [`23-VERDICT.md`](phases/23-update-cucascade-and-sirius-from-upstream/23-VERDICT.md) and [`23-VERIFICATION.md`](phases/23-update-cucascade-and-sirius-from-upstream/23-VERIFICATION.md).*

*2026-05-08 — Phase 22.1 (Remove kvikio) shipped on `feature/single-node-multi-gpu2` (post-v1.4 follow-up to Phase 22). IO-MGPU-03 promoted to Validated; IO-MGPU-02 partial (kvikio half closed, per-GPU iceberg residency renamed as IO-MGPU-04 deferred). All 7 D-01 bypass sites migrated to `sirius_ioctx::make_datasource(uring_io_object)`. v1.4 ship-gate gauntlet (REG-01..06) re-passed against post-22.1 HEAD with no regression vs Phase 22 baseline; 3 new gates (GATE-22.1-A bypass-grep / GATE-22.1-B sanitizer Cluster A = 0 / GATE-22.1-C SF1 Q11 num_gpus=2 functional) PASS. K.1 (Cluster A) closed (6 → 0 race blocks); K.6 narrowed to `downgrade_executor` worker-init device-ID configuration (independent of kvikio per advisory empirical proof; targeted as Phase 22.2). Cucascade gitlink unchanged at `c666b21`. HYG-02 = 40 invariant preserved phase-wide. Branch local-only; no `git push`; no merge to dev. Phase 22.1 verdict: PASS. See [`22.1-VERDICT.md`](phases/22.1-remove-kvikio/22.1-VERDICT.md).*

*2026-05-08 (earlier) — Phase 22 (Multi-GPU pinning + stream lineage hardening) shipped on `feature/single-node-multi-gpu2`. PIN-MGPU-01 + fu17-cluster-b promoted to Validated. Cucascade pin advanced `1c1e648` → `c666b21` (Plan 22-03 same-stream invariant fix in `alloc_and_peer_copy_async`). v1.4 ship-gate gauntlet (REG-01..06) re-passed against bumped pin with no regression; 3 new gates (GATE-07/08/09) PASS. SF100 Q1 num_gpus=2 wall-clock 2.807s (faster than Phase 21 baseline 3.150s). HYG-02 = 40 invariant preserved phase-wide. Carry-forwards to v1.6+: CC-UPSTREAM-01 (now 12 local fixes; `22-CUCASCADE-DIFF.md` captures readable diff for future PR), PIN-MGPU-02 (adaptive distribution), PIN-MGPU-03 (HOST-tier pin path), OOM-RETRY-01, follow-up #17 (SF100 Q11 num_gpus=2 query-level fallback). Phase 22 verdict: PASS. See [`22-VERDICT.md`](phases/22-multi-gpu-pinning-stream-lineage-hardening/22-VERDICT.md).*
