# Phase 24 — Conflict Log (cucascade rebase + sirius origin/dev merge)

**Branch (sirius):** feature/single-node-multi-gpu2
**Branch (cucascade fork):** fix/pinned-portable-flags
**Pre-rebase cucascade HEAD:** 9da4047 (8 commits ahead of bcddb89; backup at fix/pinned-portable-flags-pre-phase24-backup)
**Pre-merge sirius HEAD:** fa321ee (tagged pre-phase24-merge)
**Cucascade upstream target:** origin/main HEAD 9ceebaa
**Sirius upstream target:** origin/dev HEAD ba5ed27
**Resolution policy:** D-01 upstream-as-source-of-truth — favor upstream by default; preserve our changes only where they add unique behavior or fix bugs upstream doesn't have. Document rationale per file per D-02.

---

## Part 1 — Cucascade Rebase (Plan 24-02 fills in details)

### Pre-rebase classification (from 24-01-UPSTREAM-DIFFS.md Section C)

| Commit | Subject | Classification |
|--------|---------|---------------|
| 9a23f4f | fix(memory): ptds tracker, pool peer access, pipeline_io_backend hygiene | CLEAN |
| 0c0a4af | fix(pipeline_io_backend): reorder io_worker members | CLEAN |
| 8392c3d | fix(representation_converter): P2P override + DMA probe at init | RE-DERIVE — HIGH CONFLICT on representation_converter.cpp |
| 085d917 | fix(stream-lineage): writer_stream/writer_event | CLEAN |
| 89d6a3f | style: pre-commit cleanup | CLEAN |
| 1e889d7 | fix(p22): same-stream invariant in alloc_and_peer_copy_async | CLEAN |
| 37df815 | fix(p23): dst_guard for HtoD | CLEAN |
| 9da4047 | fix(p23): run_p2p_probe_locked device-restore | CLEAN |

### Commit 1: 9a23f4f → 4b94571 — Memory hygiene (ptds tracker, pool peer access, io_worker)
Classification: CLEAN
Conflict files: None (upstream did not touch memory_space.cpp or pipeline_io_backend.cpp)
Resolution: Applied cleanly by git rebase as 4b94571

### Commit 2: 0c0a4af → 3c44dae — io_worker member ordering
Classification: CLEAN
Conflict files: None
Resolution: Applied cleanly by git rebase as 3c44dae

### Commit 3: 8392c3d → d5ac57b — P2P override + DMA probe at init
Classification: RE-DERIVE — CONFLICT ON representation_converter.cpp
Conflict files: src/data/representation_converter.cpp

**Actual conflict (line 1102–1107):**
```cpp
<<<<<<< HEAD
    gpu_columns.push_back(reconstruct_column(col_meta, *fast_table->allocation, stream, mr, batch));
=======
    gpu_columns.push_back(
      reconstruct_column(col_meta, fast_table->allocation, target_stream, mr, batch));
>>>>>>> 8392c3d
```

**Root cause:** `96bfea1` changed `host_table_allocation::allocation` from `unique_ptr<multiple_blocks_allocation>` to `shared_ptr<multiple_blocks_allocation>`. The `convert_host_fast_to_gpu()` function dereferences the allocation via `*fast_table->allocation` to pass a reference. Our commit used the old `fast_table->allocation` form (unique_ptr passed directly). Additionally our commit uses `target_stream` (target-device-bound stream from memory_space pool) while upstream uses caller's `stream`.

**Resolution applied (D-01 upstream-favored + D-02 re-derive on new shape):**
- Take upstream's `*fast_table->allocation` dereference (shared_ptr API)
- Keep our `target_stream` (multi-GPU correctness — avoids cudaErrorInvalidValue when caller stream belongs to different device context)
- Final: `reconstruct_column(col_meta, *fast_table->allocation, target_stream, mr, batch)`

All other hunks in this commit (P2P code insertion, convert_gpu_to_gpu stub, DMA probe, alloc_and_peer_copy_async, reconstruct_column_p2p) applied without conflict — upstream does not have any of these functions.

Rebased as: d5ac57b

### Commit 4: 085d917 → c15cb01 — Stream lineage writer_stream/writer_event
Classification: CLEAN
Conflict files: None (upstream does not touch gpu_data_representation.hpp or related files)
Resolution: Applied cleanly as c15cb01
Empirical verification: writer_stream/writer_event present in include/cucascade/data/gpu_data_representation.hpp

### Commit 5: 89d6a3f → e10bd4a — Pre-commit formatting cleanup
Classification: CLEAN
Conflict files: None
Resolution: Applied cleanly as e10bd4a

### Commit 6: 1e889d7 → b21bd97 — Same-stream invariant in alloc_and_peer_copy_async (Cluster B)
Classification: CLEAN
Conflict files: None (alloc_and_peer_copy_async is 100% our fork code)
Resolution: Applied cleanly as b21bd97
Empirical verification: src_guard at line 622, target_stream used throughout alloc_and_peer_copy_async; no stream_default introduced

### Commit 7: 37df815 → 4319726 — dst_guard for HtoD in alloc_and_peer_copy_async (Phase 23 gap-closure)
Classification: CLEAN
Conflict files: None (same rationale as commit 6)
Resolution: Applied cleanly as 4319726
Empirical verification: `rmm::cuda_set_device_raii dst_guard{rmm::cuda_device_id{dst_device}}` at line 649

### Commit 8: 9da4047 → 1522e0b — run_p2p_probe_locked device-restore (Phase 23 gap-closure)
Classification: CLEAN
Conflict files: None (common.cpp not touched by upstream 96bfea1/9ceebaa)
Resolution: Applied cleanly as 1522e0b
Empirical verification: saved_device at lines 56–57, restored at line 146

### Commit 9 (new): 5203de5 — fix(test): adapt 96bfea1 slice-roundtrip test to writer_stream constructor
Classification: DEVIATION (Rule 1 auto-fix)
Conflict files: test/data/test_data_representation.cpp (compilation error, not a conflict marker)
Root cause: Upstream 96bfea1 added a new slice-roundtrip test using old 2-arg gpu_table_representation constructor. Our commit c15cb01 (085d917 pre-rebase) requires writer_stream as 3rd argument. This caused a compilation error (not a rebase conflict, but a post-rebase build failure).
Resolution: Added stream.view() as 3rd argument to the gpu_table_representation constructor call in the new upstream test.
Committed as: 5203de5

---

### Rebase execution state (Plan 24-02 COMPLETE)

**Rebase command used:**
```
git rebase --onto origin/main bcddb89 fix/pinned-portable-flags
```

**Outcome at Plan 24-01 handoff:** PAUSED at commit 3 (`8392c3d`).

**Plan 24-02 resolution:**
```
Rebasing (1/9): 49134ff -- DROPPED (patch contents already upstream)
Rebasing (2/9): 9a23f4f -- APPLIED CLEAN → 4b94571
Rebasing (3/9): 0c0a4af -- APPLIED CLEAN → 3c44dae
Rebasing (4/9): 8392c3d -- CONFLICT RESOLVED → d5ac57b [Plan 24-02]
Rebasing (5/9): 085d917 -- APPLIED CLEAN → c15cb01
Rebasing (6/9): 89d6a3f -- APPLIED CLEAN → e10bd4a
Rebasing (7/9): 1e889d7 -- APPLIED CLEAN → b21bd97
Rebasing (8/9): 37df815 -- APPLIED CLEAN → 4319726
Rebasing (9/9): 9da4047 -- APPLIED CLEAN → 1522e0b
Extra: (new) fix(test) → 5203de5 [Rule 1 auto-fix for upstream test API mismatch]
```

**Final fork HEAD:** `5203de5` (9 commits ahead of 9ceebaa)

**ctest:** 1/1 PASS, 14.49s

**Backup branch:** `fix/pinned-portable-flags-pre-phase24-backup` at `9da4047` — INTACT

---

## Part 2 — Sirius origin/dev Merge (Plan 24-03 fills in details)

### Predicted D-08 collision surfaces (from CONTEXT.md):
- sirius_engine.cpp (drain_after_error vs ba5ed27 wire_data_repositories Phase 2)
- duckdb_scan_executor.cpp (reservation_info + NUMA-preference vs 2e197c6 host-tier + ba5ed27 descriptors split)
- cucascade gitlink (D-05: ours always wins)

### Conflicted files (Plan 24-03 enumerates):
(placeholder)

### Auto-merged but high-risk grep verifications (Plan 24-03 fills in):
- drain_after_error preserved in task_scheduler.cpp / sirius_engine.cpp
- _no_pref_rr_counter + SCHED-RR preserved in task_scheduler.cpp
- PIN-MGPU-01 chunk_memory_spaces grep >= baseline
- kvikio bypass grep = 0
- CTE producer_types fix preserved at src/planner/sirius_plan_cte.cpp:52
- downgrade_executor tier gate preserved at src/downgrade/downgrade_executor.cpp:79,89,182

---

## Summary table (Plans 24-02 + 24-03 fill in)

| Component | Conflicts | Resolution path | Verification |
|-----------|-----------|-----------------|--------------|
| cucascade rebase | 1 predicted (commit 3) | keep our P2P code + take upstream HOST-tier type changes | cucascade ctest + grep gates |
| sirius merge | <N> | upstream-favored per D-01 | build + invariant greps |
