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

### Commit 1: 9a23f4f — Memory hygiene (ptds tracker, pool peer access, io_worker)
Classification: CLEAN
Conflict files (if any): None predicted (upstream does not touch memory_space.cpp or pipeline_io_backend.cpp in 96bfea1/9ceebaa)
Resolution: Mechanical application

### Commit 2: 0c0a4af — io_worker member ordering
Classification: CLEAN
Conflict files (if any): None predicted
Resolution: Mechanical application

### Commit 3: 8392c3d — P2P override + DMA probe at init
Classification: RE-DERIVE — CONFLICT EXPECTED
Conflict files (if any): src/data/representation_converter.cpp
Resolution strategy (Plan 24-02 fills in actual resolution):
  - Keep our entire P2P block (alloc_and_peer_copy_async, alloc_and_peer_copy_sync, reconstruct_column_p2p, convert_gpu_to_gpu impl with forward-decl removal of upstream body)
  - Take upstream's parameter-type changes to HOST-tier functions (collect_d2h_ops, alloc_and_schedule_h2d, alloc_and_copy_h2d_sync, reconstruct_column HOST path, disk helpers)
  - Take upstream's host_table_allocation::create() factory calls replacing make_unique<>
  - Root cause: our 290-line P2P insertion lands at the same boundary where upstream's parameter-type refactor operates; git cannot automatically resolve line-range overlap
Resolution: <Plan 24-02 fills in — actual diff, accepted hunks, and any re-derivation>

### Commit 4: 085d917 — Stream lineage writer_stream/writer_event
Classification: CLEAN
Conflict files (if any): None predicted
Resolution: Mechanical application (applied after commit 3 conflict resolved)

### Commit 5: 89d6a3f — Pre-commit formatting cleanup
Classification: CLEAN
Conflict files (if any): Minor formatting conflicts possible if upstream changed formatting of shared context lines
Resolution: Accept both; re-run clang-format on any file with formatting conflicts

### Commit 6: 1e889d7 — Same-stream invariant in alloc_and_peer_copy_async (Cluster B)
Classification: CLEAN
Conflict files (if any): None predicted (alloc_and_peer_copy_async is 100% our fork code, not in upstream)
Resolution: Mechanical application
Empirical verification post-resolution: Phase 23 sanitizer_gate_22.sh cluster_B must read 0; REG-05 [mgpu_stress]

### Commit 7: 37df815 — dst_guard for HtoD in alloc_and_peer_copy_async (Phase 23 gap-closure)
Classification: CLEAN
Conflict files (if any): None predicted (same rationale as commit 6)
Resolution: Mechanical application
Empirical verification post-resolution: [multi_gpu_foundation] 7/7

### Commit 8: 9da4047 — run_p2p_probe_locked device-restore (Phase 23 gap-closure)
Classification: CLEAN
Conflict files (if any): None predicted (common.cpp not touched by upstream 96bfea1/9ceebaa)
Resolution: Mechanical application

---

### Rebase execution state (Plan 24-01 records; Plan 24-02 updates)

**Rebase command used:**
```
git rebase --onto origin/main bcddb89 fix/pinned-portable-flags
```

**Outcome at Plan 24-01 handoff:** PAUSED at commit 3 (`8392c3d`).

```
Rebasing (1/9): 49134ff -- DROPPED (patch contents already upstream)
Rebasing (2/9): 9a23f4f -- APPLIED CLEAN
Rebasing (3/9): 0c0a4af -- APPLIED CLEAN
Rebasing (4/9): 8392c3d -- CONFLICT on src/data/representation_converter.cpp
  [PAUSED — Plan 24-02 resolves]
Pending (5-9): 085d917, 89d6a3f, 1e889d7, 37df815, 9da4047
```

**Conflicted file:** `src/data/representation_converter.cpp`

**Rebase state location:** `/home/felipe/sirius/.git/worktrees/ws-9aa781df-6d8c-4395-9329-737a67e8e272/modules/cucascade/rebase-merge/`

**Helper files:** `/tmp/claude/p24_01_rebase_start.log`, `/tmp/claude/p24_01_rebase_status.txt`, `/tmp/claude/p24_01_rebase_log.txt`

**Plan 24-02 entry point:**
1. Read `24-01-UPSTREAM-DIFFS.md` Section A for the conflict resolution strategy.
2. Open `cucascade/src/data/representation_converter.cpp` — resolve conflict markers.
3. Keep our P2P block + take upstream's HOST-tier parameter-type changes.
4. `git -C cucascade add src/data/representation_converter.cpp && git -C cucascade rebase --continue`
5. Expect commits 085d917 through 9da4047 to apply cleanly.

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
