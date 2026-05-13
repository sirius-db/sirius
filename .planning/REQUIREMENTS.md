# Requirements: Sirius Multi-GPU v1.4 — Rebase After DataBatch Changes

**Core Value:** All v1.1+v1.2+v1.3 multi-GPU behavior — correctness, performance, and architectural invariants — survives a rebase of `feature/single-node-multi-gpu2` onto cucascade `origin/main` (PR #117 DataBatch RAII refactor + #112 + #116) and Sirius `origin/dev` (#675 IO Framework, #731 Scan Manager, #721 Pin Tables, #739 cucascade-compat, #733/#734/#735).

---

## Milestone v1.4 Requirements (current)

**Defined:** 2026-05-04
**Goal:** Land cucascade `origin/main` and Sirius `origin/dev` onto `feature/single-node-multi-gpu2` while the v1.3 ship-gate (`[mgpu]` 16/16, `[TPC-H][parquet]` 22/22, `[integration][TPC-H]` 48/48, SF100 Q1 num_gpus=2 ≤ 5.7s, mgpu_stress 500-iter, HYG-02 ≤ 40) holds bitwise on the rebased branch.

**Verification policy:** Light gates per phase 16-20 (grep + cucascade unit tests + targeted unit tests + SF1 smoke). Full v1.3 gauntlet runs only at Phase 21.

---

### CC — Cucascade Submodule Rebase + Pin Recovery

- [x] **CC-01**: Cucascade submodule pin advanced to a commit descended from `73d00c4` (cucascade `origin/main` tip including PR #117 DataBatch RAII + PR #112 bandwidth profiler + PR #116 `gpu_data_representation` from `cudf::table_view`).
- [x] **CC-02**: All 11 local Sirius-side cucascade fixes preserved on the new pin: (a) `writer_stream` ctor requirement on `gpu_table_representation`, (b) `record_writer_event`/`get_writer_event` accessors, (c) peer-DMA probe at init, (d) `cudaStreamWaitEvent` in `convert_gpu_to_gpu`, (e) Sirius-side P2P converter override (target-bound stream in host→gpu / gpu→gpu), (f) `io_worker` member-init-order fix, (g) per-instance `ptds_allocation_tracker`, (h) `cudaHostAllocPortable`/`Mapped` flags on every pinned-host allocation, (i) cudf::pack stream argument + default-pool peer access, (j) cross-device pool peer access, (k) drop pool priming. Re-applied where original commits collide with PR #117's RAII model.
- [x] **CC-03**: Phase 13 stream-lineage semantics re-attached under PR #117's RAII accessor model. `record_writer_event` is callable with appropriate lock-scope; `convert_gpu_to_gpu` performs `cudaStreamWaitEvent` on `get_writer_event()` of the source batch before issuing peer copy.
- [x] **CC-04**: Cucascade unit-test suite passes on the rebased pin (`ctest` inside `cucascade/build`). Grep gates green: `grep -rn "record_writer_event\|get_writer_event" include/cucascade/data/` non-empty; `grep -rn "cudaHostAllocPortable" src/memory/` non-empty; `grep -rn "task_created\|in_transit" src/data/` returns zero (state-machine fully removed by #117).

### MERGE — Sirius origin/dev Merge — Base Layer

- [x] **MERGE-01**: `origin/dev` merged into `feature/single-node-multi-gpu2` (7 commits absorbed: #739 cucascade-compat, #675 IO Framework, #731 Scan Manager, #721 Pin Tables, #733 dedup, #734/#735 CI). Land as one or more atomic commits with clear conflict-resolution attribution.
- [x] **MERGE-02**: All 11 conflict files resolved mechanically (CMakeLists.txt, `cucascade` submodule pin, `src/expression_executor/gpu_expression_executor.cpp`, `src/include/creator/task_creator.hpp`, `src/include/exec/config.hpp`, `src/include/op/scan/parquet_scan_operator_data.hpp`, `src/op/scan/sirius_gpu_parquet_scan_operator.cpp`, `src/op/sirius_physical_table_scan.cpp`, `src/pipeline/sirius_pipeline_converter.cpp`, `src/scan_manager/parquet_split_provider.cpp`). 33 auto-merge files inspected for semantic conflict and either accepted or annotated with TODOs scoped to Phase 18-20.
- [x] **MERGE-03**: `sirius_parquet_metadata_scan_operator.hpp` modify/delete conflict: Phase 13 stream-lineage hooks (writer_stream wiring, writer_event acquisition) extracted to a holding location (e.g., a comment stub in `parquet_split_provider.cpp` or a temporary `phase13-extract.md` planning doc) BEFORE accepting the deletion. Re-attachment scheduled for SM-03.
- [x] **MERGE-04**: PR #739 cherry-pick is NOT performed in this phase. `git log --oneline --grep "Compat/update cucascade"` post-merge shows the dev commit absorbed, but its file changes are still TODO until DB-03 lands them on the post-#117 RAII shape.
- [x] **MERGE-05**: Build error count is bounded and documented (expected: 26+ `batch->get_data()` private-access errors + new RAII compile errors); zero unrelated build errors. Recorded in `17-MERGE-LOG.md`.

### DB — DataBatch RAII Migration (cucascade #117 surface)

- [x] **DB-01**: `src/include/data/batch_lock_utils.hpp` (129 LOC) rewritten for the post-#117 RAII model. Public helpers express lock acquisition via `to_read_only()` / `to_mutable()` / `try_to_*()` patterns; no remaining references to deleted FSM states (`task_created`, `processing`, `in_transit`, `idle`).
- [x] **DB-02**: All 26 `batch->get_data()` call sites (and any `pop_data_batch(state)` / `data_batch_processing_handle` usages) migrated to RAII accessors. `grep -rn "->get_data()\|pop_data_batch.*task_created\|data_batch_processing_handle" src/ test/` returns zero hits.
- [x] **DB-03**: ~12 operators + ~16 tests adapted using PR #739 as the file-list reference (NOT a cherry-pick; #739 targets cucascade #112 not #117). Lock scopes correct: read-only access uses `to_read_only()` for the duration of the read; mutation uses `to_mutable()` and releases before yielding to other consumers.
- [x] **DB-04**: Compile-clean state on the rebased branch — `mcp__project-commands__run_command build` exits 0 with no remaining migration TODOs. HYG-02 baseline ≤ 40.
- [x] **DB-05**: Targeted regression — `[mgpu]` filter passes 16/16 on the rebased shape (proxy for "DataBatch migration didn't break multi-GPU correctness"). Run `[mgpu_stress]` 1-iter (not 500) to smoke-test SCHED-RR survival; 500-iter is REG-05 in Phase 21. Compute-sanitizer racecheck on `[mgpu_foundation]` clean. **Closed by plan 18-07 Path A architectural fix**: [mgpu] 16/16 PASS (79091 assertions, 103.5s); [mgpu_stress] PASS (77053 assertions, 75.5s); racecheck on [downgrade_lifecycle] proxy 0 hazards ([mgpu_foundation] tag does not exist in suite — proxy retained from 18-06).

### IO — IO Framework Adoption (PR #675)

- [x] **IO-12**: `liburing-dev` apt package installed on the build host; CMakeLists.txt adds `pkg_check_modules(LIBURING REQUIRED liburing)`. vcpkg.json gains liburing entry IF the vcpkg path is exercised (`liburing` baseline resolution verified or custom port added).
- [x] **IO-13**: `SiriusContext::initialize()` constructs ONE `sirius_ioctx` per GPU (per-GPU instance), each under `rmm::cuda_set_device_raii` for its target device. Replaces `_gpu_io_backends` map from v1.1.
- [x] **IO-14**: `uring_ioctx` / `uring_reactor` instances bound to a single GPU's CUDA context — no shared ioctx across GPUs (avoids the v1.1 kvikio single-CUDA-context anti-pattern). `device_read_req.device_id` matches the ioctx's device for every request.
- [x] **IO-15**: `sirius::io::cucascade_datasource` retired — header file deleted, implementation file deleted, every include site replaced with `sirius::io::sirius_datasource`. `grep -rn "cucascade_datasource" src/ test/` returns zero hits. v1.1 IO-01..11 functionality preserved at the new datasource.
- [x] **IO-15B** (Phase 20.6 strengthened): No production code path constructs cudf-bundled file_source datasources outside the two known-deferred sites. Strengthened grep:
  ```bash
  grep -rn "cudf::io::datasource::create" src/ \
    | grep -v "src/op/scan/iceberg_metadata_reader.cpp" \
    | grep -v "src/op/scan/iceberg_scan_task.cpp"
  ```
  must return 0 hits. Known-deferred sites tracked under `IO-MGPU-02` for v1.5+:
  - `src/op/scan/iceberg_metadata_reader.cpp:227` — iceberg metadata reads (single-GPU at present)
  - `src/op/scan/iceberg_scan_task.cpp:159` — iceberg equality-delete reads (single-GPU at present)
  Closed by plan 20-06: `parquet_split_provider::run_batch` flipped from kvikio bypass to `sirius_ioctx::make_datasource(io_object)` (Sirius-side bypass eliminated; Phase 20.5 sanitizer Cluster A root cause).
- [x] **IO-16**: HYG-02 gate — raw `cudaSetDevice` calls in `uring_reactor.cpp` are wrapped in `rmm::cuda_set_device_raii` or guarded such that `rmm::cuda_stream_default` count does not regress beyond 40.
- [x] **IO-17**: SF1 smoke regression — `[TPC-H][parquet]` filter passes 22/22 on the new datasource. `[multi_gpu_foundation]` compute-sanitizer memcheck clean. **Closed by plan 19-06**: `[TPC-H][parquet]` 22/22 PASS at num_gpus=2 (36256 assertions, 78.6s, exit 0); compute-sanitizer memcheck on `[multi_gpu_foundation]` (7/7, 38 assertions) and `[integration][gpu_execution][parquet][join]` (42/42, 1.92M assertions) report 0 memcheck violations. See 19-VERDICT.md Section A + Section C.

### SM — Scan Manager + Pin Tables Port (PR #731 + #721)

- [x] **SM-01**: Phase 14 SCHED-RR distribution semantics preserved. Either `parquet_split_provider`'s split-emission is empirically round-robin across GPUs by construction (verified by `[mgpu_stress]` log inspection), OR `_no_pref_rr_counter` is ported to `parquet_split_provider`'s split-emission loop. Documented choice in `20-SCHED-RR-PORT.md`.
- [x] **SM-02**: Phase 9 `_batch_gpu_affinity` map (~20 LOC) re-planted into the #731-rewritten `sirius_gpu_parquet_scan_operator.hpp`. Phase 9 disjointedness REQUIRE (`std::set_intersection(scan_ids) == ∅`) still fires under the new architecture and gates regression.
- [x] **SM-03**: Phase 13 stream-lineage hooks (extracted in MERGE-03) re-attached. Either `parquet_split_provider::run_batch` records the writer_event when constructing the data_batch, or `sirius_gpu_parquet_scan_operator::execute` does so post-cudf-call. `cudaStreamWaitEvent` chain preserved on every cross-device peer copy. Documented in `20-STREAM-LINEAGE-REATTACH.md`.
- [x] **SM-04**: Per-task filter translation under SCHED-RR works on the new architecture — `gpu_expression_translator(stream, cudf::get_current_device_resource_ref())` is called inside `cudaSetDevice` RAII at task execution time. Verify by running TPC-H Q1 SF10 num_gpus=2 with `[mgpu-probe]` traces showing filter expressions instantiated on the dispatch device.
- [x] **SM-05**: `pin_table` single-GPU-resident behavior documented as a v1.4 limitation in `PROJECT.md` Deferred section. Follow-up requirement `PIN-MGPU-01` added to v1.5+ scope (multi-GPU-aware pinning).
- [x] **SM-06**: SF10 smoke regression — TPC-H Q1, Q6, Q12 PASS at SF10 on `num_gpus: 2` (3/3 PASS, 227 assertions, 12.01s — plan 20-04). SF1 [integration][TPC-H] num_gpus=2 — **CLOSED by plan 20-06**: 22/22 [TPC-H][parquet] PASS under compute-sanitizer (track-stream-ordered-races=all, 36256 assertions, 0 kvikio frames, exit 0); Q11 SF1 num_gpus=2 parquet PASS (9011 assertions, exit 0); [mgpu] 16/16 continuity PASS (79091 assertions, 109s). Root cause re-classified — 20-05 escalation was based on misclassified trace; the actual Sirius-side gap was `parquet_split_provider::run_batch:222` constructing cudf-bundled file_source datasources directly instead of routing through `sirius_ioctx::make_datasource(io_object)`. Fix: plumb `gpu_ioctxs` from `SiriusContext` through `sirius_scan_manager::prepare_for_query` into `parquet_split_provider`; replace the cudf factory call with the per-GPU ioctx pattern from `parquet_scan_task.cpp:343-350` (Phase 19 IO-15). [integration][TPC-H] is 47/48 PASS at SF1 num_gpus=2 — the single residual is the pre-existing SM-02 PARTIAL test-fixture mismatch ([mgpu-audit] per-GPU distribution Q1, `min_count REQUIRE` vs post-#731 single composite gpu_pipeline_task pattern, classified by 20-01) — NOT a 20-06 regression. See [`20-06-VERDICT.md`](phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-06-VERDICT.md), [`20-06-SUMMARY.md`](phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-06-SUMMARY.md).

### REG — v1.4 Ship Gate (Full v1.3 Gauntlet on Rebased Branch)

- [x] **REG-01**: `[mgpu]` filter passes 16/16, exit 0, ≥ 79091 assertions, runtime ≤ 130s. **Closed by Phase 21**: 16/16 PASS, 79091 assertions, 106.3s, exit 0. See `21-VERDICT.md` Section A.
- [x] **REG-02**: `[TPC-H][parquet]` filter passes 22/22 in ≤ 90s. **Closed by Phase 21**: 22/22 PASS, 36256 assertions, 79.3s, exit 0. See `21-VERDICT.md` Section B.
- [x] **REG-03**: `[integration][TPC-H]` filter passes 48/48 in ≤ 3 min, ≥ 71608 assertions. **Closed by Phase 21 (fixture-fix path)**: 48/48 PASS, 71607 assertions, 152.4s, exit 0. Net `-1` assertion accounted for by 1-line surgical fixture fix at `test/cpp/integration/test_gpu_execution_tpch_mgpu_audit.cpp:261-273` realigning the v1.3-era multi-pipeline_task threshold with the post-#731 single composite `gpu_pipeline_task` pattern (commit `9f835cd`). Cross-GPU `scan_id` intersection invariant (Phase 9 FIX-B regression gate at lines 286-299) preserved verbatim. See `21-VERDICT.md` Section C.
- [x] **REG-04**: SF100 TPC-H Q1 num_gpus=2 wall-clock ≤ 5.7s; result byte-identical to 1-GPU baseline; cross-GPU scan-id intersection = 0. **Closed by Phase 21**: 2-GPU 3.150s wall-clock (vs 1-GPU 4.422s baseline), byte-identical CSV, pipeline_task distribution GPU0=18 / GPU1=12 / intersect=0. See `21-VERDICT.md` Section D.
- [x] **REG-05**: `[mgpu_stress]` 500-iter PASS — 100 iterations × 5 representative `[mgpu]` queries × varied SCHED-RR counter offsets. ≥ 77053 assertions, exit 0. **Closed by Phase 21**: 1/1 PASS, 77053 assertions, 76.7s, exit 0. See `21-VERDICT.md` Section E.
- [x] **REG-06**: HYG-02 baseline preserved — `grep -rn "rmm::cuda_stream_default" src/` count ≤ 40. Compute-sanitizer memcheck clean on `[multi_gpu_foundation]` + `[integration][gpu_execution][parquet][join]`. **Closed by Phase 21**: HYG-02 = 40 (all in `src/legacy/`); Leg 1 [multi_gpu_foundation] 7/7 PASS 38 assertions, 0 memcheck violations; Leg 2 [integration][gpu_execution][parquet][join] 42/42 PASS 1.92M assertions, 0 memcheck violations. Reported "errors" (8 + 9) are exclusively benign CUDA API status returns (`cudaErrorPeerAccessAlreadyEnabled` 704 + `cudaErrorInvalidDevice` 101) per `19-VERDICT.md` Section C precedent. See `21-VERDICT.md` Section F.

---

## Validated post-v1.4 (Phase 22 + Phase 22.1)

- ✓ **PIN-MGPU-01** — Multi-GPU-aware `pin_table` (round-robin across GPU memory spaces). **Validated in Phase 22**: `PinTableFunction` distributes parquet chunks via per-call `std::size_t` round-robin counter (`idx % gpu_spaces.size()`; D-01 simple distribution policy chosen over D-02 free-memory-proportional which becomes PIN-MGPU-02 in v1.6+) + per-file `rmm::cuda_set_device_raii` guard around `chunked_parquet_reader` so cudf places columns on the intended GPU. `pinned_entry` carries per-chunk `std::vector<cucascade::memory::memory_space*> chunk_memory_spaces` parallel to `data_batches_by_column` inner vectors; `cached_split_provider` reads `_chunk_memory_spaces.at(batch_idx)` per batch. New `[pin_mgpu]` Catch2 gates: distribution gate (≥2 distinct GPU device_ids on 4-file × num_gpus=2 pin via `entry.chunk_memory_spaces` walk) + routing gate (≥1 `[mgpu-audit]` pipeline_task per GPU after CALL pin_table + SELECT through cached split provider). 2/2 PASS. See `22-VERDICT.md` Sections G+H.
- ✓ **fu17-cluster-b** — fu17 Cluster B (cucascade `alloc_and_peer_copy_async` host-staging fallback stream-ordered race, 16/21 races at SF1 Q11 num_gpus=2 per `20-05-INVESTIGATION.md`). **Closed in Phase 22**: cucascade local fork commit `c666b21` lands D-07 same-stream invariant fix in `alloc_and_peer_copy_async` (drop in-function `rmm::cuda_stream src_stream`; issue DtoH on `target_stream` under `cuda_set_device_raii(src_device)`; preserve sync-then-cudaFreeHost ordering with `cudaStreamSynchronize(target_stream.value())` inside `src_guard` scope). Sanitizer gate `test/scripts/sanitizer_gate_22.sh` reports `cluster_B=0` post-fix at both SF1 and SF100 scale (was 16 pre-fix). HYG-02 invariant preserved (0 `rmm::cuda_stream_default` in modified file). See `22-VERDICT.md` Section I + `22-CUCASCADE-DIFF.md`.
- ✓ **IO-MGPU-03** — Remove all kvikio usage from `src/`. **Validated in Phase 22.1**: all 7 D-01 bypass sites (`src/op/scan/sirius_gpu_parquet_scan_operator.cpp:126`, `src/sirius_extension.cpp:813`, `src/op/scan/iceberg_metadata_reader.cpp:211+227`, `src/io/datasource_factory.cpp:110+125`, `src/scan_manager/parquet_split_provider.cpp:295`) migrated to `sirius_ioctx::make_datasource(uring_io_object)`. GATE-22.1-A bypass-grep returns 0 hits (`grep -rn 'cudf::io::datasource::create\|cudf::io::source_info{' src/ | grep -v 'data_source\.get()\|datasource\.get()\|^[^:]*:.*//'`). GATE-22.1-B sanitizer Cluster A = 0 at SF1 Q11 num_gpus=2 (was 6 race blocks at Phase 22 baseline; K.1 closed). GATE-22.1-C SF1 Q11 num_gpus=2 functional no-regression: 50 result rows + header, identical-after-sort to 1-GPU baseline; 0 `cudaSetDevice(-1)` errors in stderr. v1.4 ship-gate gauntlet REG-01..06 re-passes against post-22.1 HEAD with no regression vs Phase 22 baseline. Cucascade gitlink unchanged at `c666b21` throughout 22.1. See `22.1-VERDICT.md` Sections G+H+I + `22.1-CUCASCADE-DIFF.md`.
- ✓ **IO-MGPU-02** (partial — kvikio half closed) — The kvikio-bypass half of IO-MGPU-02 is closed by Phase 22.1's iceberg site migration (`iceberg_metadata_reader.cpp:211+227` routed through GPU 0 sirius_ioctx via `read_iceberg_delete_data` `metadata_ioctx` parameter). Per-GPU iceberg metadata residency (the multi-GPU half) is renamed and tracked as **IO-MGPU-04** under Future Requirements. Phase 22.1 Plan 22.1-05 SUMMARY documents the GPU-0-only routing decision per CONTEXT.md D-06; IO-MGPU-04 captures the deferred multi-GPU work.

## Future Requirements (deferred to v1.6+)
- **IO-MGPU-04** (renamed from per-GPU residency half of IO-MGPU-02 after Phase 22.1) — Multi-GPU-aware iceberg metadata + equality-delete reads. Phase 22.1 closed the kvikio-bypass half of IO-MGPU-02 by routing `iceberg_metadata_reader.cpp:211+227` through GPU 0 sirius_ioctx (single-GPU correct). The remaining multi-GPU residency work routes iceberg metadata/delete reads to the consumer's preferred device:
  - `src/op/scan/iceberg_metadata_reader.cpp:211+227` — iceberg manifest/manifest-list reads (currently GPU 0 only post 22.1-05)
  - `src/op/scan/iceberg_scan_task.cpp` — iceberg equality-delete file reads (data-file reads already route through `sirius_ioctx::make_datasource` per Phase 19)
  Multi-GPU residency would require: (1) plumbing `gpu_ioctxs` into `iceberg_metadata_reader` and `iceberg_scan_task::read_equality_delete_file`, (2) constructing per-call `uring_io_object` instances, (3) routing through the appropriate ioctx by the consumer's preferred device. Trade-off: identical to PIN-MGPU-01 — currently single-GPU correct, no correctness risk because these reads are not on the multi-GPU column-chunk hot path; perf gap is negligible. Tracked here so multi-GPU iceberg residency is not lost when prioritized for v1.6+.
- **CC-UPSTREAM-01** — Open upstream cucascade PRs for the 11 local fixes so future rebases don't carry an N-commit local pin divergence. Carry the local pin in v1.4 (decision captured in PROJECT.md Key Decisions row 2026-05-04). **Phase 22 update**: 12th local fix landed (`c666b21` `alloc_and_peer_copy_async` same-stream invariant). Captured in `22-CUCASCADE-DIFF.md` for future upstream PR review per D-08/D-14.
- **PIN-MGPU-02** — Adaptive (free-memory-proportional) GPU pin distribution for `pin_table`. Phase 22 PIN-MGPU-01 ships simple `idx % N` round-robin per D-01; PIN-MGPU-02 becomes the adaptive variant if a real workload shows distribution skew (e.g., heterogeneous file sizes producing memory imbalance across GPUs). Target v1.6+ — opportunistic, contingent on observed skew at SF100 multi-table workloads.
- **PIN-MGPU-03** — HOST-tier `pin_table` path with NUMA-local round-robin. Phase 22 closes GPU-tier pinning only (`pin_table` rejects non-GPU tiers). HOST-tier sketch: read parquet to host pinned memory via existing `numa_region_pinned_host_allocator`, route each chunk to a NUMA-local GPU executor (reusing SCHED-02 `_numa_to_gpu_rr`). Target v1.6+ follow-up phase per D-06.
- **OOM-RETRY-01** — Restore OOM retry budget from 100 → 10 in `gpu_pipeline_executor.cpp:262`. Stretch goal not pursued in Phase 22; existing 100-iteration budget preserved (REG-05 `[mgpu_stress]` PASS at 77053 assertions / 80.5s with 100-budget). Target v1.6+ — opportunistic stress-suite revisit; would also be candidate workload for PIN-MGPU-02 skew check.
- **fu17-followup-17 / SF100-Q11-MGPU** — SF100 Q11 num_gpus=2 query-level fallback. Independent from Cluster B (which is now 0 even at SF100 scale per Phase 22). The trigger is `downgrade_executor per-thread init: cudaSetDevice(-1) failed: invalid device ordinal` → executor draining returns empty result. Tracked under project memory `project_phase08_fu17` since v1.2 Phase 8. Phase 22's Cluster B closure does NOT solve this. Target v1.6+ separate phase: root-cause analysis of the negative-device-ordinal binding in `downgrade_executor` per-thread init.
- **FU-B (carry from v1.3)** — Extend `mcp__project-commands__run_command` wrapper for env-passthrough OR add `num_gpus` arg to `tpch-benchmark` to lift v1.3 acceptance criterion C3 (SF1 1-GPU vs 2-GPU > 1.2× speedup) from DEFERRED.

---

## Out of Scope (v1.4)

| Feature | Reason |
|---------|--------|
| **New multi-GPU functionality** | v1.4 is a rebase milestone, not a feature milestone. v1.5+ for new capabilities. |
| **Performance uplifts beyond v1.3 baseline** | The bar is "preserve v1.3 perf"; faster is fine but not required. SF100 Q1 num_gpus=2 ≤ 5.7s is the gate. |
| **Upstream cucascade PRs for the 11 local fixes** | Decided 2026-05-04 — carry local pin this milestone; revisit upstreaming later. Captured as `CC-UPSTREAM-01` in Future. |
| **Multi-GPU-aware pin_table placement** | PR #721 is single-GPU-resident by design. Pinned splits go to GPU 0 in v1.4; multi-GPU pinning is `PIN-MGPU-01` in v1.5+. |
| **`cucascade_datasource` coexistence with `sirius_datasource`** | Decided 2026-05-04 — retire `cucascade_datasource` entirely (it was a v1.1 stopgap to dodge kvikio's CUDA-context binding). Replaced by multi-GPU-adapted `sirius_datasource`. |
| **Bisecting which dev commit causes any v1.3 regression** | Light gates per phase + full gauntlet at Phase 21. If a regression appears at Phase 21, bisect happens reactively, not preemptively per commit. |
| **Re-running v1.3 ship-gate at every phase boundary** | Per scoping decision 2026-05-04 — light gates per phase 16-20, full gauntlet at Phase 21 only. Trade-off: faster iteration; defers heavy validation until rebase complete. |
| **Cucascade `idisk_io_backend` file-handle cache** | Long-standing pitfall (v1.1 P1); not triggered by v1.4 work. Continue to defer. |
| **`cudaDeviceDisablePeerAccess` on explicit teardown** | Same as v1.2 — process-exit cleanup is adequate. |
| **TPC-H Q4 parquet intermittent flake** | Same as v1.2 — separate scoped investigation. |
| **Changes to legacy `namespace duckdb` code path** | All v1.4 work targets Super Sirius (`namespace sirius`) — same scope boundary as v1.1+v1.2+v1.3. |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CC-01 | 16 | Complete |
| CC-02 | 16 | Complete |
| CC-03 | 16 | Complete |
| CC-04 | 16 | Complete |
| MERGE-01 | 17 | Complete |
| MERGE-02 | 17 | Complete |
| MERGE-03 | 17 | Complete |
| MERGE-04 | 17 | Complete |
| MERGE-05 | 17 | Complete |
| DB-01 | 18 | Complete |
| DB-02 | 18 | Complete |
| DB-03 | 18 | Complete |
| DB-04 | 18 | Complete |
| DB-05 | 18 | Complete |
| IO-12 | 19 | Complete |
| IO-13 | 19 | Complete |
| IO-14 | 19 | Complete |
| IO-15 | 19 | Complete |
| IO-16 | 19 | Complete |
| IO-17 | 19 | Complete |
| SM-01 | 20 | Complete |
| SM-02 | 20 | Complete |
| SM-03 | 20 | Complete |
| SM-04 | 20 | Complete |
| SM-05 | 20 | Complete |
| SM-06 | 20 | Complete (SF10 PASS; SF1 PASS via plan 20-06 — Sirius-side parquet_split_provider kvikio bypass closed; Cluster A eliminated, 22/22 [TPC-H][parquet] PASS under sanitizer) |
| IO-15B | 20.6 | Complete (strengthened grep gate; closed by plan 20-06) |
| REG-01 | 21 | Complete |
| REG-02 | 21 | Complete |
| REG-03 | 21 | Complete (fixture-fix path; 71607 assertions = 71608 baseline -1 net delta from 1-line surgical SM-02 fixture fix) |
| REG-04 | 21 | Complete |
| REG-05 | 21 | Complete |
| REG-06 | 21 | Complete |
| PIN-MGPU-01 | 22 | Complete (Validated 2026-05-08) |
| fu17-cluster-b | 22 | Complete (Cluster B = 0 post-fix; cucascade pin c666b21) |
| IO-MGPU-03 | 22.1 | Complete (Validated 2026-05-08) — kvikio fully removed from src/; GATE-22.1-A bypass-grep zero; K.1 (Cluster A) closed; v1.4 ship-gate gauntlet PASS without regression |
| IO-MGPU-02 | 22.1 | Partial (kvikio half closed; per-GPU iceberg residency renamed as IO-MGPU-04 deferred to v1.6+) |
| MERGE-CC-23 | 23 | Complete — cucascade fork rebased onto bcddb89 (PR #121); 8 commits ahead of upstream after Plans 23-06 + 23-07 gap-closure (dst_guard + probe-device-restore fixes); surgical split of 6236494 correct; convert_gpu_to_gpu regression closed. See 23-VERDICT.md + 23-CUCASCADE-DIFF.md. |
| MERGE-DEV-23 | 23 | Complete — origin/dev merged into feature/single-node-multi-gpu2 at commit 49b7b86; 6 conflicts resolved (all behavioral-correctness-driven); 12 upstream commits absorbed; build green; all invariant grep gates preserved. |
| GAUNTLET-23 | 23 | Complete — 17/17 invariant gates PASS post-gap-closure (Plans 23-06 + 23-07 cucascade dst_guard + probe-device-restore + sanitizer-gate script triage). REG-05 [mgpu_stress] + REG-06 Leg 1 + Leg 2 + sanitizer_gate_22.sh all green. Side-benefit confirmed: 7cc7a79 closed pin_table suite-run flake. See 23-VERDICT.md. |
| MERGE-CC-24 | 24 | Complete — cucascade fork rebased onto 9ceebaa (upstream origin/main tip); 9 commits ahead of 9ceebaa at 5203de5; 1 RE-DERIVE conflict resolved (commit 3: representation_converter.cpp, shared_ptr dereference + target_stream preserved); 1 test-fix commit added for 96bfea1 slice-roundtrip API mismatch; ctest 1/1 PASS. See 24-CONFLICT-LOG.md Part 1 + 24-CUCASCADE-DIFF.md. |
| MERGE-DEV-24 | 24 | Complete — sirius origin/dev merged into feature/single-node-multi-gpu2 at commit ff04f31; 9 conflict files resolved upstream-favored per D-01 (INTEGRATE BOTH for PIN-MGPU-01 + host-tier parallel paths); D-05 gitlink ours-wins at 5203de5; post-merge fix-up 90fad83 (gpu_table_representation missing stream_view arg); MCP build PASS. See 24-CONFLICT-LOG.md Part 2. |
| GAUNTLET-24 | 24 | Complete — 18/18 gates PASS (17 Phase 23 invariants + D-07 new pin_table tier='host' smoke); zero regressions; two improvements (REG-06 Leg 1 memcheck 6/7 PARTIAL → 7/7 PASS; D-07 [pin_table_host] 1/1 PASS via upstream 2e197c6 test); D-04 Commit E not needed (Branch A — upstream tag used); sanitizer_gate_22.sh cluster_B=0 + P22_SELFTEST PASS. See 24-04-GAUNTLET-RESULTS.md + 24-VERDICT.md. |

**Coverage:** 39 / 39 requirements mapped to phases (32 v1.4 + 2 Phase 22 + 2 Phase 22.1). Validated by roadmapper 2026-05-04. Phase assignments confirmed against compile-graph dependency order: 16 → 17 → 18 → 19 → 20 → 21 → 22 → 22.1. Plan-level traceability filled in via per-phase SUMMARYs.

---

*Defined: 2026-05-04 — v1.4 milestone scoped via `/gsd:new-milestone`. 6 phases (16-21). Light gates per phase + full gauntlet at Phase 21. ROADMAP.md written 2026-05-04 — traceability validated. 2026-05-08 — Phase 22 added (PIN-MGPU-01 + fu17-cluster-b validated; cucascade pin advanced from `1c1e648` → `c666b21`; 22-VERDICT.md PASS). 2026-05-08 — Phase 22.1 added (IO-MGPU-03 validated; IO-MGPU-02 partial — kvikio half closed, per-GPU iceberg residency renamed as IO-MGPU-04; cucascade pin unchanged at `c666b21`; 22.1-VERDICT.md PASS). 2026-05-13 — Phase 24 complete (MERGE-CC-24 + MERGE-DEV-24 + GAUNTLET-24 all Complete; cucascade pin advanced from `9da4047` → `5203de5`; 18/18 gauntlet gates PASS; 24-VERDICT.md PASS).*
