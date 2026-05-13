# Phase 24-03 — Build Evidence

## Section A: Commits

| Commit | Role | SHA |
|--------|------|-----|
| D-02 triage gate | upstream diff analysis committed BEFORE merge | `8b2a774` |
| Merge commit (Commit C) | `git merge --no-ff origin/dev` | `ff04f31` |
| D-04 fix-up (Commit D) | post-merge build fix | `90fad83` |

## Section B: D-04 Fix-up Detail

**File:** `src/sirius_extension.cpp` line 896
**Error:**
```
error: no matching function for call to 'cucascade::gpu_table_representation::gpu_table_representation(
  std::remove_reference<std::unique_ptr<cudf::table>&>::type,
  cucascade::memory::memory_space&)'
```
**Root cause:** `gpu_table_representation` 3-arg ctor requires `(unique_ptr<cudf::table>, memory_space&, cuda_stream_view)`. Our merge resolution passed only 2 args — `stream_view` was already in scope from the HOST-tier setup block but omitted.
**Fix:** Added `stream_view` as third argument.
**Commit D:** `90fad83` — separate from merge commit per D-04 discipline.

## Section C: MCP Build Result

- **Pre-fix:** Exit code 2, 1 error (`src/sirius_extension.cpp:896`)
- **Post-fix:** Exit code 0, 79/79 targets linked, no errors (only warnings: SPDLOG_ACTIVE_LEVEL override, telemetry-bridge C++ keyword)
- **Build target:** Release extension (`sirius.duckdb_extension` + `sirius_unittest`)

## Section D: Invariant Gates (all measured POST Commit D)

| Gate | Pattern | Count | Limit | Status |
|------|---------|-------|-------|--------|
| drain_after_error | `drain_after_error` in src/ | 6 | ≥1 | PASS |
| SCHED-RR | `configure_partition_min_partitions\|SCHED_RR` | 4 | ≥1 | PASS |
| CTE producer_types | `producer_types` in src/ | 2 | ≥1 | PASS |
| downgrade tier gate | `downgrade.*tier\|tier.*downgrade` | 5 | ≥1 | PASS |
| HYG-02 cuda_stream_default | `cuda_stream_default` in src/ | 40 | ≤40 | PASS |
| kvikio-free | `source_info{path\|datasource::create` | 1 (comment) | 0 real | PASS |
| chunk_memory_spaces (PIN-MGPU-01) | `chunk_memory_spaces` in src/ | 42 | ≥baseline | PASS |
| D-05 gitlink | `git ls-tree HEAD cucascade \| awk '{print $3}'` | `5203de5` | must eq `5203de5` | PASS |

## Section E: Unit Test Results

| Tag | Tests | Assertions | Status |
|-----|-------|-----------|--------|
| `[pin_table]` | 1 | 51 | PASS |
| `[pin_table_host]` | 1 | 51 | PASS |
| `[mgpu]` | 16 | 79,091 | PASS |
| `[sirius]` | 3 | 15 | PASS |

## Section F: Commit Count Delta (D-10 Drift Check)

- **Commits merged from origin/dev:** 2 (`ba5ed27`, `2e197c6`)
- **Within D-10 ≤5 limit:** YES
- **Pre-merge branch lead:** 12 commits ahead of origin/dev (from D-02 triage doc)
- **Post-merge:** feature/single-node-multi-gpu2 leads origin/dev by 10 commits (2 consumed by merge + 2 new: merge commit + fix-up)
