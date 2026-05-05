# Phase 17 Merge Audit Log (MERGE-01..05)

**Phase:** 17-sirius-origin-dev-merge-base-layer
**Started:** 2026-05-05
**Status:** in-progress (seeded by plan 17-01)
**Purpose:** Per-file conflict resolution outcomes, auto-merge audit results, and bounded build error inventory for the `git merge --no-ff origin/dev` operation. Documented per MERGE-01 ("clear conflict-resolution attribution") and MERGE-05 ("Build error count is bounded and documented").

***

## Pre-merge state (filled by plan 17-01)

- HEAD SHA at merge time: `5aee3143f60d66201bb82095166c69cba145d30f`
- Backup ref: `phase17-pre-merge-backup` -> SHA `98cdea20691a53a84c03eb2463ffc5d1027fe2df`
- Cucascade pin at HEAD: `1c1e648a282a06747328c78f62d2d676ce51a8ce` (per Phase 16 ship verdict)
- origin/dev tip SHA: `cdd6864cabbbd0bebca93167af4d5964104cad93`
- Pre-merge HYG-02 count (rmm::cuda_stream_default in src/): 40 (matches Phase 14 baseline)
- Conflict surface inventory: 10 conflicting source files (UU) + 1 modify/delete (UD) = 11 conflict files total; cucascade was auto-resolved by git to `1c1e648` (fast-forward to our pin) without manual intervention

***

## Section A — 11 Conflict Files: Per-File Resolution (filled by plan 17-02)

Order matches CONTEXT.md domain inventory.

### A.1 — `CMakeLists.txt`
- Resolution policy: D-D1 (keep both — multi-GPU runtime config + scan-manager config)
- Resolution outcome: Combined per D-D1. Kept our `src/io/cucascade_datasource.cpp` (Phase 5 adapter, not yet upstreamed) + accepted dev's new io source files: `src/io/admission_control.cpp`, `src/io/prefetching_cache.cpp`, `src/io/sirius_datasource.cpp`, `src/io/uring/uring_ioctx.cpp`, `src/io/uring/uring_reactor.cpp`. Single conflict block resolved. Markers removed.
- TODOs added: None (EXTENSION_SOURCES list merge was purely additive)

### A.2 — `cucascade` (submodule)
- Resolution policy: D-B1/B2 (keep ours unconditionally — pin must remain `1c1e648`)
- Resolution command: NOT NEEDED — git auto-resolved via "Fast-forwarding submodule cucascade to 1c1e648a282a06747328c78f62d2d676ce51a8ce" (git detected our commit was already ahead of dev's pin along the same lineage). Staged automatically at `160000 1c1e648a282a06747328c78f62d2d676ce51a8ce 0`.
- Verification: `git ls-files --stage cucascade` returns `160000 1c1e648a282a06747328c78f62d2d676ce51a8ce 0\tcucascade`. Phase 16 pin defended.

### A.3 — `src/expression_executor/gpu_expression_executor.cpp`
- Resolution policy: D-D5 (take dev's version; flag any post-#731 stream changes for Phase 18)
- Resolution outcome: Took dev's version via `git checkout --theirs`. Audited for `get_data()`, `pop_data_batch`, `cudaSetDevice` patterns — zero hits found. No Phase 18 TODOs needed in this file.

### A.4 — `src/include/creator/task_creator.hpp`
- Resolution policy: D-D4 (combine — preserve our `_no_pref_rr_counter`-related context; accept dev's other changes)
- Resolution outcome: Combined per D-D4. Single conflict block: our side added `#include <unordered_map>` (required for `_numa_to_gpu` field at line ~217) and `#include <variant>`. Dev removed both. Kept `<unordered_map>` (actively used by `std::unordered_map<int, std::vector<int>> _numa_to_gpu`), dropped `<variant>` (included but unused in header body per grep audit). `_no_pref_rr_counter` itself lives in `task_scheduler.hpp` (not this file — confirmed). SCHED-RR context in the header's comments at lines 213-219 (SCHED-02 round-robin doc) survived the merge via auto-merge. No TODO needed.

### A.5 — `src/include/exec/config.hpp`
- Resolution policy: D-D1 (keep both — multi-GPU runtime config + scan-manager config)
- Resolution outcome: Combined per D-D1. Single conflict block: our side added `#include <optional>` (for `std::optional<int> preferred_numa_node` field); dev's side added `#include <cstdint>` (for `uint64_t monitor_period_ms`). Kept both includes: `#include <cstdint>` (first, alphabetical) + `#include <optional>`. Both fields were preserved in the non-conflicted body via auto-merge. Markers removed.

### A.6 — `src/include/op/scan/parquet_scan_operator_data.hpp`
- Resolution policy: D-D4 (combine — preserve our `_batch_gpu_affinity`-related context if present; accept dev's other changes)
- Resolution outcome: Combined per D-D4. Two conflict blocks: (1) Our side had `parquet_metadata_input` and `partitioned_parquet_metadata` classes — accepted dev's removal since `sirius_parquet_metadata_scan_operator.hpp` was deleted (these were its data types). (2) Constructor signature for `parquet_scan_data` — took dev's simplified version (duckdb::Expression filter + scan_plan). Our multi-GPU fields `retranslation_filter` and `filter_name_resolver` were in the non-conflicted body (auto-merged in); replaced them with a `TODO(v1.4 Phase 20 — SM-02)` comment block. Added `TODO(v1.4 Phase 20 — SM-02)` above constructor. `_batch_gpu_affinity` confirmed to live in `duckdb_scan_executor.hpp:218` (not this file — not conflicted).

### A.7 — `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp` (modify/delete)
- Resolution policy: D-D6 — accept deletion AFTER 17-PHASE-13-EXTRACT.md is committed (plan 17-01 prereq).
- Resolution command: `git rm src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp`
- Re-attachment target documented in: `17-PHASE-13-EXTRACT.md` (Phase 20 SM-03)
- Resolution outcome: Accepted deletion via `git rm`. Phase 13 stream-lineage extraction confirmed in `17-PHASE-13-EXTRACT.md` (commit `2f3a786`). Re-attachment scheduled for Phase 20 SM-03 in `src/op/scan/sirius_gpu_parquet_scan_operator.cpp::execute()`. File is absent from working tree post-merge.

### A.8 — `src/op/scan/sirius_gpu_parquet_scan_operator.cpp`
- Resolution policy: D-D3 (take dev's; add TODOs for Phase 20 mgpu re-integration: `_batch_gpu_affinity` recording, writer_stream forwarding, per-task filter translation under SCHED-RR)
- Resolution outcome: Took dev's version via `git checkout --theirs`. Phase 20 SM-01/02/03/04 TODO block inserted above `execute()` (4 sub-items: SCHED-RR, _batch_gpu_affinity, writer_stream, per-task filter translation). Phase 18 DB-02 TODO inserted above `batch->get_data()` call in cached path at line ~191. Total: 1 Phase 20 block + 1 Phase 18 TODO.

### A.9 — `src/op/sirius_physical_table_scan.cpp`
- Resolution policy: D-D3 (take dev's; add TODOs for Phase 20)
- Resolution outcome: Took dev's version via `git checkout --theirs`. Phase 20 SM-01..06 TODO inserted above `execute()`. Phase 18 DB-02 TODOs inserted above 4 `get_data()` calls (lines ~89, ~127-128, ~144, ~160). Phase 18 DB-03 TODO inserted above `pop_data_batch` call (line ~86). Total: 1 Phase 20 TODO + 5 Phase 18 TODOs.

### A.10 — `src/pipeline/sirius_pipeline_converter.cpp`
- Resolution policy: D-D3 (take dev's; add TODOs for Phase 20)
- Resolution outcome: Took dev's version via `git checkout --theirs`. Audited for `get_data()`, `pop_data_batch`, `cudaSetDevice` — zero hits. Phase 18 DB-01 and Phase 20 SM-03 TODO block inserted above first function definition `split_parquet_scan_source()`. Total: 2 TODOs.

### A.11 — `src/scan_manager/parquet_split_provider.cpp` (net-new)
- Resolution policy: D-D2 (take dev's version as-is; net-new on dev, no local version exists)
- Verification: `ls src/scan_manager` should NOT exist before merge; should exist after
- Resolution outcome: Took dev's net-new file via `git checkout --theirs`. Note: git detected a rename conflict (HEAD path = `src/op/scan/sirius_parquet_metadata_scan_operator.cpp`, dev path = `src/scan_manager/parquet_split_provider.cpp`). Resolved by taking dev's theirs version. File LOC: 345 lines. No TODOs added — Phase 20 SM-01..03 will integrate v1.3 multi-GPU semantics (SCHED-RR, _batch_gpu_affinity, writer_stream) into this file.

***

## Section B — 33 Auto-Merge Audit (filled by plan 17-03)

### B.1 — Inventory

`git diff origin/dev...HEAD --stat` post-merge — auto-merged file list: `<filled by plan 17-03>`

### B.2 — FSM grep audit (D-E1 step 1; P7 / D-G3 gate)

For each auto-merged file, run:
```
grep -n "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" <file>
```
Expected: zero hits per file. Any hit means dev re-introduced FSM state names that #117 deleted. Annotate with TODO per D-E2.

Project-wide gate (D-G3):
```
grep -rn "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" src/
```
Expected: 0. Result: `<filled>`

Test-tree gate (per CONTEXT.md specifics — "FSM grep audit must extend to test/"):
```
grep -rn "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" test/
```
Expected: 0. Result: `<filled>`

### B.3 — HYG-02 grep audit (D-E1 step 2)

Project-wide HYG-02 baseline (Phase 14 baseline = 40 per ROADMAP REG-06):
```
grep -rc "rmm::cuda_stream_default" src/ | awk -F: '{s+=$2} END {print s}'
```
Pre-merge: `<filled by plan 17-02 step A>`
Post-merge: `<filled by plan 17-03>`
Net delta from dev auto-merges: `<filled>`

Per-file HYG-02 hits in auto-merged files: `<filled>`. Note: increases here are EXPECTED (#675 IO Framework code lands and Phase 19 IO-16 will clean it up). Documented as deferred per CONTEXT.md deferred ideas.

### B.4 — SCHED-RR survival (D-G2 / P6)

```
grep -c "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp
```
Expected: `>= 1` (currently 3 at HEAD). Result: `<filled>`

```
grep -n "SCHED-RR" src/pipeline/task_scheduler.cpp
```
Expected: non-empty (the round-robin distribution block at ~line 253 + the reset comment at ~line 156). Result: `<filled>`

### B.5 — TODO annotations added

For each auto-merged file with FSM hits or HYG-02 regressions, append per D-E2:
```
// TODO(v1.4 Phase 18 — DB-XX): wrap in to_read_only() accessor (origin/dev auto-merge re-introduced pre-#117 batch_state name)
// TODO(v1.4 Phase 19 — IO-16): wrap raw cudaSetDevice in rmm::cuda_set_device_raii
```
File list: `<filled>`

***

## Section C — Build Error Bounding (filled by plan 17-03; MERGE-05)

Per D-F1/F2/F3, the post-merge Sirius build is EXPECTED to fail. Bound the failure surface so plan 17-04 can verify the failure is "expected only".

### C.1 — Build invocation

Command (D-F2): `mcp__project-commands__run_command build` (per CLAUDE.md "Use MCP for build/test"). Fallback if MCP aborts on non-zero exit: `pixi run -- bash -c 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc) 2>&1 | tee build/17-build-output.log; exit 0'` — capture log even on failure.

Build log location: `<filled>`

### C.2 — Expected error count buckets

| Error pattern | Count | Notes |
|---|---|---|
| `'get_data' is a private member` (or equivalent) | `<filled>` | Expected: 26+ per ROADMAP / CONTEXT D-F1 — Phase 18 closes |
| `no member named 'pop_data_batch'` | `<filled>` | Expected: any non-zero count is OK; Phase 18 DB-02 closes |
| `no member named 'data_batch_processing_handle'` | `<filled>` | Expected: any non-zero count is OK; Phase 18 closes |
| Unknown identifier `task_created` / `in_transit` / `idata_batch_probe` | `<filled>` | Expected: 0 (we discarded #739's pre-#117 file changes); any non-zero is INVESTIGATE |
| Missing `liburing` header | `<filled>` | Expected: 0 if `liburing-dev` already installed; non-zero is Phase 19 IO-12 territory; document but do not block |
| RAII compile errors (`to_mutable` / `to_read_only` / `read_only_data_batch` / `mutable_data_batch`) | `<filled>` | Expected: any non-zero count; Phase 18 DB-02/DB-03 closes |
| **Unrelated errors** (any error NOT in above categories) | `<filled>` | Expected: 0 — D-F3 says "investigate before proceeding" |

### C.3 — Total error count

Total errors: `<filled>`
Expected categories sum: `<filled>`
Unrelated count: `<filled>`
Verdict (D-F3 gate): `<PASS / INVESTIGATE>`

***

## Section D — Verification Gates (filled by plan 17-04; D-G1..G6)

| Gate | Command | Expected | Actual |
|---|---|---|---|
| D-G1 (merge commit) | `git log --oneline --merges -1` | dev-merge commit | `<filled>` |
| D-G2 (SCHED-RR survival) | `grep -c "_no_pref_rr_counter" src/include/pipeline/task_scheduler.hpp` | >= 1 | `<filled>` |
| D-G2 (SCHED-RR block) | `grep "SCHED-RR" src/pipeline/task_scheduler.cpp` | non-empty | `<filled>` |
| D-G3 (no old FSM names src/) | `grep -rn "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" src/ \| wc -l` | 0 | `<filled>` |
| D-G3 (no old FSM names test/) | `grep -rn "task_created\|in_transit\|data_batch_processing_handle\|idata_batch_probe" test/ \| wc -l` | 0 | `<filled>` |
| D-G4 (extract file exists) | `test -f .planning/phases/17-sirius-origin-dev-merge-base-layer/17-PHASE-13-EXTRACT.md` | exit 0 | `<filled>` |
| D-G5 (this log exists + populated) | `test -f .planning/phases/17-sirius-origin-dev-merge-base-layer/17-MERGE-LOG.md` | exit 0 | `<filled>` |
| D-G6 (cucascade pin defended) | `git ls-tree HEAD cucascade \| awk '{print $3}'` | `1c1e648a282a06747328c78f62d2d676ce51a8ce` | `<filled>` |

***

## Section E — Note on PR #739 Bookkeeping (P7)

Per Pitfall P7 and ROADMAP P7 mapping: PR #739 (`Compat/update cucascade gpu table in sirius` — commit `468f6e1`) is one of the 7 origin/dev commits absorbed by this merge. Its FILE CHANGES are intentionally NOT applied here — they target the pre-#117 cucascade API (`0cd4a6a`). Phase 18 DB-03 will re-port #739's operator-file changes against the post-#117 RAII shape using #739 as a file-list reference only.

The merge commit therefore absorbs `468f6e1` as **bookkeeping-only** — `git log --oneline --grep "Compat/update cucascade"` after the merge will show the dev commit was absorbed, but its file edits to operator sites are deliberately TODO until Phase 18 lands them on the post-#117 RAII shape.

***

## Phase 17 Verdict (filled by plan 17-04)

- MERGE-01: `<PASS / FAIL>` — `<evidence>`
- MERGE-02: `<PASS / FAIL>` — `<evidence>`
- MERGE-03: `<PASS / FAIL>` — `<evidence>`
- MERGE-04: `<PASS / FAIL>` — `<evidence>`
- MERGE-05: `<PASS / FAIL>` — `<evidence>`

Final verdict: `<PASS / PARTIAL / FAIL>`
