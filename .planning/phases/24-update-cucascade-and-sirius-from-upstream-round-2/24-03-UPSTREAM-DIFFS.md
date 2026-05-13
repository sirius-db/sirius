# Phase 24-03 — Upstream Sirius Diff Triage

**Date:** 2026-05-13
**Analyst:** Claude (Plan 24-03, Task 1a — D-02 read-the-diff-first gate)
**Branch:** `feature/single-node-multi-gpu2`
**Upstream sirius target:** `origin/dev` HEAD `ba5ed27`
**Commits past Phase 23 merge tip (`8524c79`):** 2
  - `ba5ed27` refactor: split wire_data_repositories into descriptors + runtime (Phase 2 of #601) (#770)
  - `2e197c6` feat(pin_table): support tier='host' for host-tier caching (#774)

---

## Section A: ba5ed27 — "refactor: split wire_data_repositories into descriptors + runtime (Phase 2 of #601) (#770)"

### Files touched

| File | Change | Risk to our branch |
|------|--------|--------------------|
| `CMakeLists.txt` | +2 lines: add `repository_wiring_materializer.cpp` to EXTENSION_SOURCES and `test_repository_wiring_materializer.cpp` to TEST_SOURCES | LOW — pure additions, no collision |
| `src/include/pipeline/repository_wiring.hpp` | NEW FILE: `repository_wiring` descriptor struct + `materialize_repository_wiring()` free function declaration | NONE — entirely new file |
| `src/include/pipeline/sirius_pipeline_converter.hpp` | Adds `repository_wiring.hpp` include, `<vector>` include, `repository_wirings` to `pipeline_conversion_result`, `repository_wirings_` to converter, removes `sirius_engine` forward decl, renames `wire_data_repositories()` to `compute_repository_wiring()` (no engine arg), removes `sirius_engine&` from `convert()` signature | **MEDIUM** — Phase 23-04 merged changes to `sirius_pipeline_converter.hpp`? Check if our branch has deviating state. |
| `src/include/sirius_engine.hpp` | Removes `insert_repository()` overloads (2 removed) | LOW — we never call `insert_repository()` directly in our patches; no collision expected |
| `src/pipeline/repository_wiring_materializer.cpp` | NEW FILE: implements `materialize_repository_wiring()` | NONE — entirely new file |
| `src/pipeline/sirius_pipeline_converter.cpp` | Large refactor: renames `wire_data_repositories(engine_)` to `compute_repository_wiring()`, removes engine references, replaces all `engine_.insert_repository()` calls with local `emit()` lambda that pushes `repository_wiring` descriptors, fixes `setup_pipeline_parents()` to drive off `repository_wirings_` descriptors, fixes `link_join_partition_siblings()` to mutate the wiring descriptor instead of a port | **HIGH** — our Phase 23 merge resolution touched `sirius_pipeline_converter.cpp` (Phase 23-04 file D-17: auto-merged cleanly there, but our branch may have accumulated further changes) |
| `src/sirius_engine.cpp` | Removes `insert_repository()` method implementations (~70 lines removed), changes `converter.convert(*root_pipeline, *this)` to `converter.convert(*root_pipeline)`, adds `materialize_repository_wiring()` call after `converter.convert()` returns | **HIGH** — our branch has `drain_after_error` on the success path in `sirius_engine.cpp`. The ba5ed27 diff touches `initialize_internal()` at exactly the point where our `drain_after_error` was placed (Phase 23-04 merge resolution). |
| `test/cpp/pipeline/test_repository_wiring_materializer.cpp` | NEW FILE: 6 test cases for `materialize_repository_wiring()` | NONE — entirely new file |

### Functional summary

`ba5ed27` extracts the pipeline wiring topology computation (`wire_data_repositories`) from runtime into pure-data `repository_wiring` descriptors emitted at plan time by `sirius_pipeline_converter::compute_repository_wiring()`. The runtime materialization (creating `shared_data_repository` instances, attaching ports, sink fanout) is lifted into a free function `materialize_repository_wiring()` called from `sirius_engine::initialize_internal()` after `converter.convert()` returns.

Key structural change: `converter.convert()` no longer takes a `sirius_engine&` argument. The engine is no longer passed to the converter at all. This means any code that previously passed `*this` to `convert()` now calls `convert(*root_pipeline)` only.

### Collision surfaces against our Phase 23 merge resolutions

**1. `src/sirius_engine.cpp` — MEDIUM-HIGH risk (D-08)**

Our Phase 23-04 resolution (`23-04-CONFLICT-LOG.md: src/sirius_engine.cpp`) integrated `drain_after_error` (ours) first, then upstream's unfinalized-op warning loop (upstream), both on the success path in `execute()`. The `ba5ed27` diff touches `initialize_internal()` (the converter call site), NOT `execute()`. So `drain_after_error` is in `execute()` and `ba5ed27` changes `initialize_internal()` — these are DIFFERENT functions. The collision risk is: if ba5ed27 also reorganizes `initialize_internal()` structure such that git context lines around `converter.convert(*root_pipeline, *this)` conflict with something we have.

Looking at the exact diff hunk:
```diff
-  auto result = converter.convert(*root_pipeline, *this);
+  auto result = converter.convert(*root_pipeline);
+
+  // Materialize plan-time wiring descriptors into runtime repositories and ports.
+  pipeline::materialize_repository_wiring(result.repository_wirings,
+                                          sirius_ctx_ptr->get_data_repository_manager());
```

Our Phase 23-04 resolution did NOT change the `converter.convert()` call site — it only added `drain_after_error()` in `execute()`. So the `initialize_internal()` change in `ba5ed27` should AUTO-MERGE against our branch (no collision at that exact site). BUT ba5ed27 also removes `insert_repository()` implementations (~70 lines) — those were in the upstream `origin/dev` tree but NOT in our branch (we never added them; they came from Phase 23 base but our merge incorporated them). Need to verify: does our branch have `insert_repository` implementations in `sirius_engine.cpp`?

**Prediction:** ba5ed27's changes to `sirius_engine.cpp` will likely AUTO-MERGE cleanly. The `drain_after_error` is in `execute()` which ba5ed27 does not touch. The `insert_repository` removal targets Phase 23 upstream code that is present in our merged tree. This is a favorable auto-merge.

**2. `src/pipeline/sirius_pipeline_converter.cpp` — MEDIUM risk**

Our Phase 23-04 resolution shows `sirius_pipeline_converter.cpp` auto-merged cleanly (D-17 in 23-04-CONFLICT-LOG.md). So our branch has the post-Phase 23 merge state (which includes upstream's Phase 23 changes). `ba5ed27` then makes a much larger refactor to this same file. The conflict is very likely because ba5ed27 rewrites the entire `wire_data_repositories` function — which is present in our Phase 23 merged tree — into `compute_repository_wiring`.

**Resolution plan (D-01 upstream-favored):** Take upstream's entire new structure (`compute_repository_wiring` + emit lambda). Do NOT re-insert engine references. Our Phase 23 merge did not add any custom code to `sirius_pipeline_converter.cpp` beyond what upstream Phase 23 already had — so there is nothing unique to preserve here. FULL upstream take.

### Re-derivation strategy for D-01

- `sirius_engine.cpp`: Take upstream's `initialize_internal()` change (materialize_repository_wiring call) verbatim. Verify `drain_after_error` is still in `execute()` (separate function — should survive auto-merge).
- `sirius_pipeline_converter.cpp`: Take upstream's entire refactored version. Our branch has no unique additions here beyond Phase 23's auto-merged state.

---

## Section B: 2e197c6 — "feat(pin_table): support tier='host' for host-tier caching (#774)"

### Files touched

| File | Change | Risk to our branch |
|------|--------|--------------------|
| `cucascade` gitlink | Bumps from `73d00c4` to `96bfea1` (pure-upstream cucascade) | **CRITICAL** — D-05: ours always wins. Our fork HEAD `5203de5` must win over `96bfea1`. |
| `src/include/memory/multiple_blocks_allocation_accessor.hpp` | Templatizes all methods on `Ptr` (accept both `unique_ptr` and `shared_ptr`) | **HIGH CONFLICT** — Plan 24-02 (ff06fac, D-04 Commit B) ALREADY applied this exact same change! Both we and upstream made identical changes from different commit paths. Git will see our version vs upstream's version as a textual conflict even though they are functionally identical. |
| `src/include/op/result/host_table_chunk_reader.hpp` | Changes `unique_ptr` to `shared_ptr` on `_allocation` field and method signatures | **HIGH CONFLICT** — Same situation: Plan 24-02 ff06fac already applied this. |
| `src/include/scan_manager/cached_split_provider.hpp` | Adds HOST-tier constructor to `cached_split_provider` | **MEDIUM** — Our branch does NOT have this constructor (it was added by upstream `2e197c6`). Should auto-merge as an addition. But merge-tree shows CONFLICT here. |
| `src/include/scan_manager/sirius_scan_manager.hpp` | Adds `host_chunks` and `tier` fields to `pinned_entry`, adds `insert_pinned_entry_host()` declaration | **MEDIUM** — Our branch does not have these additions. Should auto-merge but merge-tree shows CONFLICT. |
| `src/op/result/host_table_chunk_reader.cpp` | Changes `unique_ptr` to `shared_ptr` in `column_reader` method implementations | **HIGH CONFLICT** — Plan 24-02 ff06fac already applied this. |
| `src/op/scan/cpu_source_task.cpp` | `make_unique<host_table_allocation>` → `host_table_allocation::create()` | No conflict expected — Plan 24-02 already changed this, and this is the same change. |
| `src/op/scan/duckdb_scan_task.cpp` | Same `create()` factory change | No conflict expected — Plan 24-02 already changed this. |
| `src/include/op/scan/parquet_scan_operator_data.hpp` | Adds `prepare_for_processing()` method to `scan_cached_operator_data` + new includes | LOW-MEDIUM — Our branch does not have this method. Should auto-merge. |
| `src/scan_manager/cached_split_provider.cpp` | Adds HOST-tier constructor implementation + HOST-tier start() branch | LOW — Our branch does not have these. Should auto-merge. |
| `src/scan_manager/sirius_scan_manager.cpp` | Adds HOST-tier branch in `create_provider_for()` + `insert_pinned_entry_host()` implementation | **MEDIUM** — merge-tree shows CONFLICT here. Our branch may have changes to `sirius_scan_manager.cpp` adjacent to upstream's new HOST-tier block. |
| `src/sirius_extension.cpp` | Rewrites `PinTableBind()` and `PinTableFunction()` to support `tier='host'` | **HIGH** — merge-tree shows CONFLICT. Our branch has Phase 22.1+ changes to `sirius_extension.cpp` (kvikio-bypass related). Also note: 2e197c6 uses `cudf::io::source_info{file_paths}` (a vector of strings) — this is a valid use via make_datasource? Need to check. Actually it uses `cudf::io::parquet_reader_options::builder(cudf::io::source_info{file_paths})` directly which is the forbidden pattern per IO-15B. HOWEVER: `sirius_extension.cpp::PinTableFunction` passes `file_paths` (strings), NOT a datasource — this is an in-process scan where the extension reads directly for its own pin operation, NOT via the sirius ioctx. This is a distinct read path from the operator scan path where IO-15B applies. Need to verify: does our CLAUDE.md kvikio-free policy apply to the pin_table internal read path? |
| `test/cpp/integration/test_gpu_execution_tpch.cpp` | Adds `[pin_table_host]` integration test | LOW — addition only. |
| `test/cpp/memory/test_host_table_utils.cpp` | `make_unique<host_table_allocation>` → `::create()` factory | **HIGH CONFLICT** — Plan 24-02 ff06fac already applied this exact change. |

### Functional summary

`2e197c6` adds `tier='host'` support to `pin_table` table function. The key changes:
1. Reads GPU tables then immediately converts to `cucascade::host_data_representation` via GPU→HOST converter when `tier='host'`.
2. Stores `host_data_representation` chunks in `pinned_entry.host_chunks`.
3. `create_provider_for()` builds a `cached_split_provider` using the new HOST-tier constructor which slices chunks at scan time.
4. `scan_cached_operator_data::prepare_for_processing()` converts HOST-resident batches to GPU just before the scan executor runs.
5. **Cucascade fallout**: 2e197c6's commit message explicitly states it incorporated the cucascade 96bfea1 API changes (private ctor → `::create()`, `unique_ptr` → `shared_ptr`). Plan 24-02 (ff06fac) pre-applied these same API fixes to sirius BEFORE this merge. So when merging 2e197c6, we will see conflicts in those exact files because BOTH our ff06fac and 2e197c6 made the same changes from different commit paths.

### Collision surfaces against our Phase 23 merge resolutions + Plan 24-02

**1. multiple_blocks_allocation_accessor.hpp, host_table_chunk_reader.hpp/cpp, test_host_table_utils.cpp — CONFLICT (D-04 Commit B vs 2e197c6)**

Plan 24-02's `ff06fac` (D-04 Commit B) already applied the cucascade 96bfea1 API changes to sirius. Upstream's `2e197c6` applies the SAME changes from a different commit. When git merges, it will see our ff06fac version vs the 2e197c6 version — the changes are identical in spirit but may differ in whitespace or formatting. Resolution: take UPSTREAM (2e197c6) versions — they are functionally identical to ours, and upstream-favored per D-01. The result is the same code.

**2. sirius_extension.cpp — CONFLICT**

Our branch may have Phase 22.1+ additions adjacent to where 2e197c6 rewrites `PinTableFunction()`. Resolution per D-01: take upstream's new `PinTableFunction()` structure (tier='host' support) verbatim. Verify that any Phase 22.1 unique kvikio bypass code in `sirius_extension.cpp` is NOT in `PinTableFunction()` but in `PinTableBind()` or the datasource factory paths — check after seeing the actual conflict markers.

**Note on `cudf::io::source_info{file_paths}` in PinTableFunction:** This is a pin_table INTERNAL read (the extension function reads parquet files to pin them). The IO-15B kvikio-free policy targets OPERATOR scan paths (`src/op/scan/`, `src/op/scan/parquet_scan_operator_data.hpp`) to avoid cuDF's direct file I/O bypassing kvikio. The pin_table function at the extension level uses cudf's own chunked_parquet_reader for its initial pin read — this is a one-time administrative read, NOT a hot scan path. This usage is consistent with the pre-existing GPU-tier path that also uses `cudf::io::source_info{file_paths}`. The IO-15B check grep is: `grep -rn "cudf::io::datasource::create\|cudf::io::source_info{" src/ | grep -v "data_source\.get()\|datasource\.get()"`. `sirius_extension.cpp::PinTableFunction` uses `cudf::io::source_info{file_paths}` (vector of string). This WILL be flagged by the grep. Need to verify if the existing GPU-tier path already has this and was already in our baseline — if yes, it's a pre-existing baseline count, not a regression introduced by our merge.

**3. cucascade gitlink — D-05 ours-wins**

2e197c6 bumps the gitlink to `96bfea1` (upstream cucascade). Our fork HEAD is `5203de5` (rebased fork, 9 commits ahead of `9ceebaa` which contains `96bfea1`). Per D-05: `git checkout --ours cucascade` during resolution.

**4. duckdb_scan_executor.cpp — ABSENT from merge-tree conflict list**

The merge-tree shows only 9 conflicted files. `duckdb_scan_executor.cpp` is NOT in the list. D-08 predicted MEDIUM risk for this file from 2e197c6 changes to scan paths. Checking 2e197c6 diff: it does NOT touch `duckdb_scan_executor.cpp` directly. Our Phase 23-04 NUMA-preference changes auto-merged there previously. The new `duckdb_scan_task.cpp` change (one line, `::create()` factory) is the same as our ff06fac change — likely auto-merges because both sides changed the same line identically, or git uses one side. Need to verify post-merge that NUMA preference is still present.

### Re-derivation strategy for D-01

- **multiple_blocks_allocation_accessor.hpp, host_table_chunk_reader.hpp/cpp, test_host_table_utils.cpp:** Take UPSTREAM (2e197c6) versions. Our ff06fac made identical changes. The merged result is the same code as ours.
- **cached_split_provider.hpp/cpp, sirius_scan_manager.hpp/cpp, parquet_scan_operator_data.hpp:** Take UPSTREAM (2e197c6) versions — these are net-new additions we don't have. No unique behavior to preserve from our side.
- **sirius_extension.cpp:** Take UPSTREAM (2e197c6) HOST-tier support verbatim. Check for any Phase 22.1+ unique additions on our side that are NOT in 2e197c6.
- **cucascade gitlink:** `git checkout --ours cucascade` per D-05.

---

## Section C: Predicted Conflict File List

From `git merge-tree HEAD origin/dev` CONFLICT lines:

```
CONFLICT (content): Merge conflict in src/include/memory/multiple_blocks_allocation_accessor.hpp
CONFLICT (content): Merge conflict in src/include/op/result/host_table_chunk_reader.hpp
CONFLICT (content): Merge conflict in src/include/scan_manager/cached_split_provider.hpp
CONFLICT (content): Merge conflict in src/include/scan_manager/sirius_scan_manager.hpp
CONFLICT (content): Merge conflict in src/op/result/host_table_chunk_reader.cpp
CONFLICT (content): Merge conflict in src/pipeline/sirius_pipeline_converter.cpp
CONFLICT (content): Merge conflict in src/scan_manager/sirius_scan_manager.cpp
CONFLICT (content): Merge conflict in src/sirius_extension.cpp
CONFLICT (content): Merge conflict in test/cpp/memory/test_host_table_utils.cpp
```

**Total: 9 conflict files.**

**Notable auto-merges (high-risk, need post-merge grep verification):**

- `src/sirius_engine.cpp` — PREDICTED AUTO-MERGE (drain_after_error in execute(), ba5ed27 only touches initialize_internal()). **Post-merge grep required:** `grep -n "drain_after_error" src/sirius_engine.cpp`
- `src/pipeline/task_scheduler.cpp` — NOT touched by either upstream commit. SCHED-RR counter should survive.
- `src/op/scan/duckdb_scan_executor.cpp` — NOT in conflict list. NUMA-preference should survive.
- `src/op/scan/cpu_source_task.cpp` — Both we (ff06fac) and 2e197c6 changed `make_unique` → `::create()`. Likely auto-merges (same line, same change). But could conflict.
- `src/op/scan/duckdb_scan_task.cpp` — Same as cpu_source_task.cpp.
- `src/planner/sirius_plan_cte.cpp` — NOT touched by either upstream commit. CTE producer_types fix should survive.
- `src/downgrade/downgrade_executor.cpp` — NOT touched by either upstream commit. Tier gate should survive.

**Conflict root causes by file:**

| File | Root cause of conflict |
|------|----------------------|
| `multiple_blocks_allocation_accessor.hpp` | Our ff06fac AND upstream 2e197c6 both templatized methods — textual conflict on identical change |
| `host_table_chunk_reader.hpp` | Same — both changed unique_ptr → shared_ptr |
| `host_table_chunk_reader.cpp` | Same — method signature changes |
| `test_host_table_utils.cpp` | Same — make_unique → ::create() |
| `cached_split_provider.hpp` | 2e197c6 adds HOST-tier constructor at a location our ff06fac didn't touch — may be an ordering/context conflict |
| `sirius_scan_manager.hpp` | 2e197c6 adds tier/host_chunks fields + insert_pinned_entry_host — context conflict |
| `sirius_pipeline_converter.cpp` | ba5ed27 rewrites wire_data_repositories — which exists in our Phase 23 merged tree |
| `sirius_scan_manager.cpp` | 2e197c6 adds HOST-tier block + insert_pinned_entry_host impl — context conflict with our Phase 22.1/23 merged state |
| `sirius_extension.cpp` | 2e197c6 rewrites PinTableFunction — adjacent to our Phase 22.1+ additions |

---

## Section D: D-10 Drift Check Result

**Command:** `git log --oneline ^8524c79 origin/dev`

**Result:**
```
ba5ed27 refactor: split wire_data_repositories into descriptors + runtime (Phase 2 of #601) (#770)
2e197c6 feat(pin_table): support tier='host' for host-tier caching (#774)
```

**Count: 2 commits** — well within the ≤5 acceptable drift limit. No escalation needed. Both commits were pre-triaged in CONTEXT.md D-08. Proceed with merge as planned.

**origin/dev tip SHA:** `ba5ed27080726f30aaa828437b191c3db78b9621`
**Backup branch:** `phase24-pre-merge-backup` at `04b4c7e829dcedf29f33fe2622ff85eb2dec2556` (Plan 24-02 tip)

---

*Generated: 2026-05-13 by Plan 24-03 Task 1a analysis of /tmp/claude/p24_03_ba5ed27_full.diff and /tmp/claude/p24_03_2e197c6_full.diff*
