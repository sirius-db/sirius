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
- [ ] **DB-02**: All 26 `batch->get_data()` call sites (and any `pop_data_batch(state)` / `data_batch_processing_handle` usages) migrated to RAII accessors. `grep -rn "->get_data()\|pop_data_batch.*task_created\|data_batch_processing_handle" src/ test/` returns zero hits.
- [ ] **DB-03**: ~12 operators + ~16 tests adapted using PR #739 as the file-list reference (NOT a cherry-pick; #739 targets cucascade #112 not #117). Lock scopes correct: read-only access uses `to_read_only()` for the duration of the read; mutation uses `to_mutable()` and releases before yielding to other consumers.
- [ ] **DB-04**: Compile-clean state on the rebased branch — `mcp__project-commands__run_command build` exits 0 with no remaining migration TODOs. HYG-02 baseline ≤ 40.
- [ ] **DB-05**: Targeted regression — `[mgpu]` filter passes 16/16 on the rebased shape (proxy for "DataBatch migration didn't break multi-GPU correctness"). Run `[mgpu_stress]` 1-iter (not 500) to smoke-test SCHED-RR survival; 500-iter is REG-05 in Phase 21. Compute-sanitizer racecheck on `[mgpu_foundation]` clean.

### IO — IO Framework Adoption (PR #675)

- [ ] **IO-12**: `liburing-dev` apt package installed on the build host; CMakeLists.txt adds `pkg_check_modules(LIBURING REQUIRED liburing)`. vcpkg.json gains liburing entry IF the vcpkg path is exercised (`liburing` baseline resolution verified or custom port added).
- [ ] **IO-13**: `SiriusContext::initialize()` constructs ONE `sirius_ioctx` per GPU (per-GPU instance), each under `rmm::cuda_set_device_raii` for its target device. Replaces `_gpu_io_backends` map from v1.1.
- [ ] **IO-14**: `uring_ioctx` / `uring_reactor` instances bound to a single GPU's CUDA context — no shared ioctx across GPUs (avoids the v1.1 kvikio single-CUDA-context anti-pattern). `device_read_req.device_id` matches the ioctx's device for every request.
- [ ] **IO-15**: `sirius::io::cucascade_datasource` retired — header file deleted, implementation file deleted, every include site replaced with `sirius::io::sirius_datasource`. `grep -rn "cucascade_datasource" src/ test/` returns zero hits. v1.1 IO-01..11 functionality preserved at the new datasource.
- [ ] **IO-16**: HYG-02 gate — raw `cudaSetDevice` calls in `uring_reactor.cpp` are wrapped in `rmm::cuda_set_device_raii` or guarded such that `rmm::cuda_stream_default` count does not regress beyond 40.
- [ ] **IO-17**: SF1 smoke regression — `[TPC-H][parquet]` filter passes 22/22 on the new datasource. `[multi_gpu_foundation]` compute-sanitizer memcheck clean.

### SM — Scan Manager + Pin Tables Port (PR #731 + #721)

- [ ] **SM-01**: Phase 14 SCHED-RR distribution semantics preserved. Either `parquet_split_provider`'s split-emission is empirically round-robin across GPUs by construction (verified by `[mgpu_stress]` log inspection), OR `_no_pref_rr_counter` is ported to `parquet_split_provider`'s split-emission loop. Documented choice in `20-SCHED-RR-PORT.md`.
- [ ] **SM-02**: Phase 9 `_batch_gpu_affinity` map (~20 LOC) re-planted into the #731-rewritten `sirius_gpu_parquet_scan_operator.hpp`. Phase 9 disjointedness REQUIRE (`std::set_intersection(scan_ids) == ∅`) still fires under the new architecture and gates regression.
- [ ] **SM-03**: Phase 13 stream-lineage hooks (extracted in MERGE-03) re-attached. Either `parquet_split_provider::run_batch` records the writer_event when constructing the data_batch, or `sirius_gpu_parquet_scan_operator::execute` does so post-cudf-call. `cudaStreamWaitEvent` chain preserved on every cross-device peer copy. Documented in `20-STREAM-LINEAGE-REATTACH.md`.
- [ ] **SM-04**: Per-task filter translation under SCHED-RR works on the new architecture — `gpu_expression_translator(stream, cudf::get_current_device_resource_ref())` is called inside `cudaSetDevice` RAII at task execution time. Verify by running TPC-H Q1 SF10 num_gpus=2 with `[mgpu-probe]` traces showing filter expressions instantiated on the dispatch device.
- [ ] **SM-05**: `pin_table` single-GPU-resident behavior documented as a v1.4 limitation in `PROJECT.md` Deferred section. Follow-up requirement `PIN-MGPU-01` added to v1.5+ scope (multi-GPU-aware pinning).
- [ ] **SM-06**: SF10 smoke regression — TPC-H Q1, Q6, Q12 PASS at SF10 on `num_gpus: 2`. `[integration][TPC-H]` 48/48 PASS at SF1.

### REG — v1.4 Ship Gate (Full v1.3 Gauntlet on Rebased Branch)

- [ ] **REG-01**: `[mgpu]` filter passes 16/16, exit 0, ≥ 79091 assertions, runtime ≤ 130s.
- [ ] **REG-02**: `[TPC-H][parquet]` filter passes 22/22 in ≤ 90s.
- [ ] **REG-03**: `[integration][TPC-H]` filter passes 48/48 in ≤ 3 min, ≥ 71608 assertions.
- [ ] **REG-04**: SF100 TPC-H Q1 num_gpus=2 wall-clock ≤ 5.7s; result byte-identical to 1-GPU baseline; cross-GPU scan-id intersection = 0.
- [ ] **REG-05**: `[mgpu_stress]` 500-iter PASS — 100 iterations × 5 representative `[mgpu]` queries × varied SCHED-RR counter offsets. ≥ 77053 assertions, exit 0.
- [ ] **REG-06**: HYG-02 baseline preserved — `grep -rn "rmm::cuda_stream_default" src/` count ≤ 40. Compute-sanitizer memcheck clean on `[multi_gpu_foundation]` + `[integration][gpu_execution][parquet][join]`.

---

## Future Requirements (deferred to v1.5+)

- **PIN-MGPU-01** — Multi-GPU-aware `pin_table`. Place pinned splits on the GPU with the lowest free memory ratio (or distribute across GPUs by table-row count) instead of always GPU 0. Trade-off: P2P copy overhead via `convert_gpu_to_gpu` is acceptable in v1.4 because Phase 13 `cudaStreamWaitEvent` chain ensures correctness; perf gap is small at SF1 but may show at SF100 multi-table workloads.
- **CC-UPSTREAM-01** — Open upstream cucascade PRs for the 11 local fixes so future rebases don't carry an N-commit local pin divergence. Carry the local pin in v1.4 (decision captured in PROJECT.md Key Decisions row 2026-05-04).
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
| **Bisecting which dev commit causes any v1.3 regression** | Light gates per phase + full gauntlet at Phase 21. If a regression appears at Phase 21, bisect happens reactively, not pre-emptively per commit. |
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
| DB-02 | 18 | Pending |
| DB-03 | 18 | Pending |
| DB-04 | 18 | Pending |
| DB-05 | 18 | Pending |
| IO-12 | 19 | Pending |
| IO-13 | 19 | Pending |
| IO-14 | 19 | Pending |
| IO-15 | 19 | Pending |
| IO-16 | 19 | Pending |
| IO-17 | 19 | Pending |
| SM-01 | 20 | Pending |
| SM-02 | 20 | Pending |
| SM-03 | 20 | Pending |
| SM-04 | 20 | Pending |
| SM-05 | 20 | Pending |
| SM-06 | 20 | Pending |
| REG-01 | 21 | Pending |
| REG-02 | 21 | Pending |
| REG-03 | 21 | Pending |
| REG-04 | 21 | Pending |
| REG-05 | 21 | Pending |
| REG-06 | 21 | Pending |

**Coverage:** 32 / 32 requirements mapped to phases. Validated by roadmapper 2026-05-04. Phase assignments confirmed against compile-graph dependency order: 16 → 17 → 18 → 19 → 20 → 21. Plan-level traceability will be filled in by /gsd:plan-phase as each phase is planned.

---

*Defined: 2026-05-04 — v1.4 milestone scoped via `/gsd:new-milestone`. 6 phases (16-21). Light gates per phase + full gauntlet at Phase 21. ROADMAP.md written 2026-05-04 — traceability validated.*
