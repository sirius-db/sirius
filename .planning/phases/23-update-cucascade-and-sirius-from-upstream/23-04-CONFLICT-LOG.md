# Phase 23 — Plan 23-04 — Sirius origin/dev Merge Conflict Log

**Branch:** `feature/single-node-multi-gpu2`
**Pre-merge HEAD:** `ac7c23a` (docs(23-03): complete gitlink bump + intermediate gauntlet — build green, 4 suites pass)
**Upstream:** `origin/dev` tip `8524c79` (12 commits ahead of merge base)
**Merge base:** `05c4dc6ff49a7e01ffbbd239c9c7e89d588dd596`
**Merge base divergence:** 12 left (origin/dev) / 399 right (our HEAD, = 393 Phase 17-22.3 + 6 Phase 23 plans)
**Resolution protocol:** Per CONTEXT.md `specifics` — `git log` both sides before resolving; document rationale per file.

---

## Predicted D-13..D-20 files that auto-merged cleanly (no conflict)

These files were predicted as conflict risks but git resolved them automatically.
Post-merge grep verification confirms our behavioral patterns survived.

### D-13: src/expression_executor/** (upstream 7eeaab4 sirius::value AST Phase 2)
- **Outcome:** Auto-merged cleanly. No conflict.
- **Verification:** Files in `src/expression_executor/` contain no conflict markers. Our expression-executor patterns survive in the merged tree; `7eeaab4`'s value AST changes did not overlap with our Phase 22 scan-task expression-executor instantiation.

### D-14 (partial): src/creator/task_creator.cpp (upstream 7cc7a79 task-creation race fix)
- **Outcome:** Auto-merged cleanly. No conflict.
- **Verification:** `grep -n "drain_after_error" src/pipeline/task_scheduler.cpp` returns non-zero (line 203).

### D-14 (partial): src/pipeline/task_scheduler.cpp (upstream 7cc7a79)
- **Outcome:** Auto-merged cleanly. No conflict.
- **Verification:** `grep -n "_no_pref_rr_counter\|SCHED-RR" src/pipeline/task_scheduler.cpp` returns non-zero (lines 156, 160, 253, 261) — SCHED-RR preserved.

### D-15 (partial): src/pipeline/gpu_pipeline_task.cpp (upstream 5d09a59 bytes-to-materialize)
- **Outcome:** Auto-merged cleanly. No conflict.
- **Verification:** `grep -n "column count mismatch" src/pipeline/gpu_pipeline_task.cpp` returns non-zero (line 57) — Phase 22.3 CTE _types validator preserved.

### D-17: src/pipeline/sirius_pipeline_converter.cpp (upstream 972cb32 rename + barrier annotations)
- **Outcome:** Auto-merged cleanly. No conflict.
- **Verification:** Pipeline converter auto-merged; upstream's `result.inserted_operators` rename (from `pipeline_breakers`) also auto-merged in `src/sirius_engine.cpp` (visible in the non-conflicted region at line ~401).

### D-20: src/pipeline/sirius_plan_printer.cpp (upstream 972cb32)
- **Outcome:** Auto-merged cleanly. No conflict.

---

## Conflicted Files (6 total)

---

## docs/super-sirius/README.md

### Upstream commits touching this file (origin/dev side):
- `16543e6` docs: refresh Super Sirius docs for scan manager and recent changes (#768)

### Our commits touching this file (feature/single-node-multi-gpu2 side):
- `abe5cdb` docs(15-03): document per-task-device contract under SCHED-RR
- `93d7bdc` merge origin/dev into feature/single-node-multi-gpu2

### Upstream diff summary:
Upstream changed the `last-updated-commit` watermark comment from the old base value to `d6baaedc0d2bb07b27a00c65135513f3c23f0b37`.

### Our diff summary:
We updated line 35 table entry for Pipeline Execution to add `per-task-device contract under SCHED-RR` text. We changed the watermark to `75392110240075a4152dbc6c1fcefa9dfab01f3d`.

### Conflict shape:
Single line conflict on the `<!-- last-updated-commit: ... -->` watermark. Both sides changed it to different values.

### Resolution chosen:
**Take upstream's watermark; keep our content update.** The table row update at line 35 was already auto-merged correctly (both sides changed different lines). The watermark is a doc-tracking marker; upstream's `16543e6` is the later commit so its watermark is authoritative.

### Rationale:
Both watermarks are doc-only metadata. Upstream's docs refresh (`16543e6`) post-dates our `abe5cdb` by recency. Content correctness is maintained: our SCHED-RR description in the table (line 35) auto-merged cleanly; only the watermark was disputed.

---

## src/include/op/sirius_physical_partition.hpp

### Upstream commits touching this file (origin/dev side):
- `e94ad4a` Feature: Per operator memory estimate for when we don't have historical info (#776)

### Our commits touching this file (feature/single-node-multi-gpu2 side):
- `93d7bdc` merge origin/dev into feature/single-node-multi-gpu2
- `1f80c2a` fix(08): multi-GPU task distribution + partition pinning

### Upstream diff summary:
Added `no_history_peak_memory_estimate(const op::input_stats& stats) const override` virtual method declaration (D-16 per-operator memory estimate).

### Our diff summary:
Added `set_min_num_partitions(int min_num_partitions, uint64_t small_table_bytes)` inline method + `_min_num_partitions` and `_small_table_bytes` private members (Phase 8 PIN-MGPU-01: PIN at least num_gpus partitions for multi-GPU workloads).

### Conflict shape:
Both additions were inserted at the same location in the public section (after `set_num_partitions`), causing a content conflict.

### Resolution chosen:
**Integrate both.** Keep `set_min_num_partitions` (ours, Phase 8 PIN-MGPU) then `no_history_peak_memory_estimate` (upstream, D-16). Both are additive public method declarations with no semantic overlap.

### Rationale:
Both methods serve independent purposes: `set_min_num_partitions` is the Phase 8 GPU-partition routing floor for multi-GPU dispatch; `no_history_peak_memory_estimate` is upstream's new per-op memory estimation for schedulability analysis. Neither modifies the other's behavior. Integration is the only correct answer.

---

## src/include/sirius_context.hpp

### Upstream commits touching this file (origin/dev side):
- `8524c79` fix python extension bug (#777)

### Our commits touching this file (feature/single-node-multi-gpu2 side):
- Multiple commits from Phases 5, 7, 17, 19, 22.1 — adding `gpu_ioctxs_`, `datasource_registry_`, `peer_access_enabled_pairs_`, `get_ioctx_for()`, `get_gpu_ioctxs()`, `get_datasource_registry()`, `is_peer_access_enabled()` members and accessors.

### Upstream diff summary:
Removed `<condition_variable>` and `<thread>` headers (formerly needed for `std::condition_variable query_lifecycle_cv_` and `std::thread::id active_query_owner_`). These were replaced by the simpler `std::atomic<bool> query_lifecycle_held_` + plain `std::mutex` approach from `7cc7a79` race fix. The `<thread>` removal and member changes in the `.cpp` auto-merged in the non-header portions.

### Our diff summary:
Added `<unordered_map>`, `<unordered_set>`, `<utility>` headers for our Phase 19/22 members (`gpu_ioctxs_`: `unordered_map<int, shared_ptr<sirius_ioctx>>`; `peer_access_enabled_pairs_`: `unordered_set<pair<int,int>, peer_pair_hash>`).

### Conflict shape:
Headers section: upstream side removed `<thread>` (and `<condition_variable>` auto-merged separately); we added `<thread>`, `<unordered_map>`, `<unordered_set>`, `<utility>`. Git produced a conflict on the block containing these additions.

### Resolution chosen:
**Drop `<thread>`; keep `<unordered_map>`, `<unordered_set>`, `<utility>`.** Our code never uses `std::thread` or `std::thread::id` in this header — those were from the merge-base's `active_query_owner_` member which upstream correctly removed. Our three new headers ARE required by `gpu_ioctxs_` (unordered_map), `peer_access_enabled_pairs_` (unordered_set + pair via utility).

### Rationale:
`<thread>` was in our HEAD only because the merge-base had it (for `std::thread::id active_query_owner_`). Upstream's `7cc7a79` race fix replaced the thread-owning pattern with `std::atomic<bool>`, removing the need for `<thread>`. Our Phase 19/22 additions are completely independent and require `<unordered_map>`, `<unordered_set>`, `<utility>`. The integrated resolution correctly keeps exactly the headers required by the surviving code.

---

## src/op/scan/duckdb_scan_executor.cpp

### Upstream commits touching this file (origin/dev side):
- `5d09a59` Fixed bug with bytes to materialize and other improvements (#769)
- `7cc7a79` Fix for race condition between task creation and finalizing pipelines (#766)

### Our commits touching this file (feature/single-node-multi-gpu2 side):
- Multiple commits from Phases 2, 4, 8, 9, 18 — per-GPU stream pools, target_gpu_id routing, NUMA-preference reservation, RAII accessor migration.

### Upstream diff summary:
Changed `scan_task->get_estimated_reservation_size()` (returns `size_t`) to `scan_task->get_estimated_reservation_size_info()` (returns a struct with `reservation_size` field plus extra bytes-to-materialize metadata). Changed `local_state->set_reservation(std::move(reservation))` to `local_state->set_reservation(std::move(reservation), reservation_info)`. Added new branches for `duckdb_scan_task` and `cpu_source_task` that also populate `reservation_info`. Used `any_memory_space_in_tier` (no NUMA preference) in the simple path.

### Our diff summary:
Changed the reservation request to use `any_memory_space_in_tier_with_preference{Tier::HOST, static_cast<size_t>(target_gpu_id)}` instead of plain `any_memory_space_in_tier{Tier::HOST}` — Phase 8/9 NUMA-aware host memory preference routing.

### Conflict shape:
Two lines of the reservation call — the `get_estimated_reservation_size*` call and the `_mem_mgr->request_reservation` call. Our side uses the old scalar API + NUMA preference; upstream uses the new struct API without NUMA preference.

### Resolution chosen:
**Integrate both.** Use upstream's `get_estimated_reservation_size_info()` + `reservation_info.reservation_size` (correct API, required by the non-conflicting `local_state->set_reservation(std::move(reservation), reservation_info)` at line 440); keep our `any_memory_space_in_tier_with_preference{Tier::HOST, target_gpu_id}` (Phase 8 NUMA routing).

### Rationale:
The non-conflicting region of the file ALREADY uses `reservation_info` (line 440 `local_state->set_reservation(..., reservation_info)`), so we MUST use `get_estimated_reservation_size_info()` or the code would not compile (undeclared `reservation_info` variable). At the same time, our NUMA preference routing (`any_memory_space_in_tier_with_preference`) is the Phase 8 PIN-MGPU-01 mechanism — without it, host memory is allocated without GPU-affinity preference, breaking the multi-GPU locality invariant. Both changes are independently correct and complementary. The integrated solution preserves the bytes-to-materialize bug fix AND the NUMA routing.

---

## src/sirius_context.cpp

### Upstream commits touching this file (origin/dev side):
- `8524c79` fix python extension bug (#777)
- `e94ad4a` Feature: Per operator memory estimate for when we don't have historical info (#776)
- `7cc7a79` Fix for race condition between task creation and finalizing pipelines (#766)

### Our commits touching this file (feature/single-node-multi-gpu2 side):
- Many commits from Phases 1, 2, 5, 6, 7, 19, 20, 22.1 — NUMA topology, peer access, per-GPU ioctx construction, datasource_registry registration, teardown ordering.

### Upstream diff summary:
(a) Added `<cstddef>` and `<iostream>` headers. (b) Added a disk-space warning block immediately after `memory_manager_` construction: checks if DISK tier is empty and emits `SIRIUS_LOG_WARN` if so. (c) Changed `is_query_lifecycle_active()` to use `query_lifecycle_held_.load()`. (d) Changed `acquire_query_lifecycle_slot()` and `release_query_lifecycle_slot()` to simpler mutex-lock pattern. These changes from `7cc7a79` auto-merged cleanly in the respective function bodies.

### Our diff summary:
(a) Added `<algorithm>` header. (b) Inserted MGPU-05 host-space assertion, Phase 19 IO-13 per-GPU ioctx construction loop, Phase 22.1 kFileScheme datasource_registry registration, and MGPU-06 P2P peer-access enable loop — all between `memory_manager_` creation and the cuDF pinned allocator block.

### Conflict shape (2 conflicts):
1. Headers: `<algorithm>` (ours) vs `<cstddef>` (upstream) — both need to coexist.
2. Large block: our ~160 lines of MGPU-05/IO-13/22.1/MGPU-06 initialization vs upstream's ~8-line disk warning block at the same insertion point.

### Resolution chosen:
**Integrate both.** (1) Headers: keep both `<algorithm>` and `<cstddef>`. (2) Large block: place upstream's disk warning block FIRST (it belongs right after memory_manager_ creation per upstream intent), then our entire MGPU-05/IO-13/22.1/MGPU-06 block (which also belongs right after memory_manager_ creation for the same initialization-sequencing reason).

### Rationale:
The disk warning and our initialization blocks serve completely different purposes and operate on different memory tiers (DISK vs HOST/GPU). Both must run after `memory_manager_` is constructed. Ordering: disk warning first (upstream added it at that point in initialization), then our GPU/HOST initialization (which must happen before the cuDF pinned allocator block below, per Phase 19 IO-13 ordering requirements). Neither block reads outputs from the other, so ordering between them is behaviorally equivalent — upstream-first is chosen for diff-review clarity.

---

## src/sirius_engine.cpp

### Upstream commits touching this file (origin/dev side):
- `7cc7a79` Fix for race condition between task creation and finalizing pipelines (#766)
- `972cb32` Improve sirius pipeline diagnostics: rename converter symbols and annotate plan printer with barrier info (#763)

### Our commits touching this file (feature/single-node-multi-gpu2 side):
- `615b76c` feat(22.1-05): pass GPU 0 sirius_ioctx to read_iceberg_delete_data
- `df7b666` fix(merge): drain task_creator on success path to prevent UAF + reactivate queue on restart
- Earlier commits from Phases 8, 19.

### Upstream diff summary:
Added a post-execute warning loop: after all tasks complete on the success path, iterates operators and warns (`SIRIUS_LOG_WARN`) if any operator was never finalized. Also renamed `result.pipeline_breakers` to `result.inserted_operators` (from `972cb32`) — this rename auto-merged in the non-conflicting region at line ~401.

### Our diff summary:
Added `sirius_ctx->get_task_scheduler().drain_after_error()` on the success path (Phase 22 `df7b666`). This is the critical UAF fix: drains the task_creator and all executors after the result_collector pipeline completes but while operators are still alive, preventing use-after-free when the engine is destroyed.

### Conflict shape:
Both additions inserted code on the success path immediately after the catch blocks close. Our 15-line `drain_after_error` call vs upstream's ~15-line unfinalized operator warning loop. Git created a conflict because both sides inserted at exactly the same line.

### Resolution chosen:
**Integrate both; drain_after_error first.** Keep our `drain_after_error()` call first (Phase 22 correctness invariant — must drain BEFORE operators are destroyed), then upstream's unfinalized operator warning loop (diagnostic only, safe to run while operators are still alive since we explicitly note they are alive at that point in the upstream comment).

### Rationale:
`drain_after_error()` is a correctness-critical Phase 22 fix — it prevents UAF when consecutive queries share a SiriusContext. It MUST remain on the success path. Upstream's warning loop is diagnostic-only and cannot harm correctness. The order matters: drain first (operators still alive, thread pool stopped cleanly), then log warnings about unfinalized operators. Both are safe to coexist; the upstream comment "All tasks completed — operators and pipelines are still alive here" is still accurate after the drain because drain stops task creation/execution but does not destroy operators.

---

## Auto-merged Predicted Risk Files — Grep Verification

### drain_after_error (Phase 22.2 invariant)
`grep -n "drain_after_error" src/pipeline/task_scheduler.cpp` → line 203: PRESENT

### SCHED-RR counter (Phase 14 invariant)
`grep -n "_no_pref_rr_counter\|SCHED-RR" src/pipeline/task_scheduler.cpp` → lines 156, 160, 253, 261: PRESENT

### CTE _types validator (Phase 22.3 invariant)
`grep -n "column count mismatch" src/pipeline/gpu_pipeline_task.cpp` → line 57: PRESENT

### SF10 Q11 test (Phase 22.3 invariant)
`grep -n "tpch_q11_sf10_2gpu" test/cpp/integration/test_gpu_execution_tpch.cpp` → line 4415: PRESENT

### downgrade_executor tier gate (Phase 22.2 invariant)
`grep -n "_space_id.tier == cucascade::memory::Tier::GPU\|tier == Tier::GPU" src/downgrade/downgrade_executor.cpp` — auto-merged cleanly (file not in conflict list).

---

## Summary

| File | Risk | Conflict Shape | Resolution |
|------|------|----------------|------------|
| docs/super-sirius/README.md | LOW (D-21) | Watermark comment clash | Take upstream watermark; our content update auto-merged |
| src/include/op/sirius_physical_partition.hpp | MEDIUM (D-16) | Two additive method declarations at same insertion point | Integrate both: set_min_num_partitions + no_history_peak_memory_estimate |
| src/include/sirius_context.hpp | LOW (D-21 / python fix) | Headers: <thread> removed upstream; we added <unordered_map/set/utility> | Drop <thread> (no longer needed); keep our 3 headers |
| src/op/scan/duckdb_scan_executor.cpp | MEDIUM (D-14/D-17) | reservation API: scalar→struct; NUMA preference routing | Integrate: use new struct API + keep NUMA preference |
| src/sirius_context.cpp | HIGH (D-14/D-16/D-21) | Large block: disk warning vs our MGPU-05/IO-13/22.1/MGPU-06 init | Integrate: disk warning first, then our block |
| src/sirius_engine.cpp | MEDIUM (D-14/D-20) | Success path: drain_after_error vs unfinalized-op warning | Integrate: drain first, then warning loop |

**6 conflicts total. 0 mechanical ours/theirs picks. All resolutions behavioral-correctness-driven.**
