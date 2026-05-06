---
phase: 20-scan-manager-pin-tables-port-pr-731-pr-721
plan: 03
subsystem: planning-docs
tags: [sm-05, pin_table, single-gpu, deferred, pin-mgpu-01]
requires:
  - .planning/PROJECT.md (Deferred section anchor)
  - .planning/REQUIREMENTS.md (Future Requirements section + existing PIN-MGPU-01 stub)
  - 20-RESEARCH.md (Code Example 4 — src/sirius_extension.cpp:726-733)
provides:
  - SM-05 documentation gate closure: PROJECT.md Deferred bullet for pin_table single-GPU residency
  - PIN-MGPU-01 cross-doc loop-close: src cite + PROJECT.md backref + Phase 20 SM-05 registration
affects:
  - SM-05 status: documentation-only, eligible for Pending → Complete pending Phase 20 verifier
  - Phase 20 ROADMAP success criterion 5 (SM-05 component) satisfied
tech-stack:
  added: []
  patterns:
    - cross-document loop-close (PROJECT.md ↔ REQUIREMENTS.md citing each other)
    - source-line citation with verbatim code excerpt
key-files:
  created: []
  modified:
    - .planning/PROJECT.md (Deferred section: +1 bullet for pin_table single-GPU residency)
    - .planning/REQUIREMENTS.md (PIN-MGPU-01 entry: augmented with src cite + Phase 13 re-attach site + PROJECT.md backref + SM-05 registration)
decisions:
  - "Branch B (augment) applied for Task 2: existing PIN-MGPU-01 stub lacked src cite (src/sirius_extension.cpp:733) and PROJECT.md backreference; augmentation closes the cross-doc loop required for SM-05."
  - "Cross-references locked bidirectionally: PROJECT.md Deferred bullet → PIN-MGPU-01; REQUIREMENTS.md PIN-MGPU-01 → PROJECT.md Deferred section. Future PIN-MGPU-01 implementation can grep either doc and trace to the other."
metrics:
  duration: 2min
  tasks: 2
  files: 2
  completed: "2026-05-06T10:04:11Z"
---

# Phase 20 Plan 03: Pin Tables Documentation (SM-05) Summary

Documented `pin_table` single-GPU residency (`src/sirius_extension.cpp:733` always pins to `gpu_spaces[0]`) as a v1.4 limitation in PROJECT.md Deferred section, and augmented `PIN-MGPU-01` in REQUIREMENTS.md Future Requirements with the src cite + Phase 13 re-attach site + bidirectional PROJECT.md backreference, closing the SM-05 documentation gate.

## Objective

SM-05 of Phase 20 requires:
1. PROJECT.md Deferred section documents `pin_table` single-GPU-resident behavior as a v1.4 limitation citing `src/sirius_extension.cpp:733`
2. REQUIREMENTS.md Future Requirements registers `PIN-MGPU-01` (multi-GPU-aware `pin_table`) for v1.5+ scope

Pure planning-doc edits; no source code changes; no test files touched.

## What Was Done

### Task 1 — PROJECT.md Deferred bullet (commit `03aadb3` — joint with parallel 20-01 due to staging race)

Added a new bullet under PROJECT.md `## Deferred to Future Milestones` (above the existing `[mgpu-audit]` SIGSEGV bullet):

```
- **`pin_table` single-GPU residency (v1.4)**: `CALL pin_table(...)` always pins to GPU 0
  in v1.4. The implementation at `src/sirius_extension.cpp:733` uses `gpu_spaces[0]`
  unconditionally — `auto& mem_space = const_cast<cucascade::memory::memory_space&>(*gpu_spaces[0]);`.
  Multi-GPU-aware placement is tracked as `PIN-MGPU-01` in `REQUIREMENTS.md` Future
  Requirements section, deferred to v1.5+. Trade-off: P2P copy overhead via
  `convert_gpu_to_gpu` is acceptable in v1.4 because Phase 13 `cudaStreamWaitEvent`
  chain (re-attached at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:263`)
  ensures correctness; perf gap is small at SF1 but may show at SF100 multi-table
  workloads. Documented for SM-05 closure during Phase 20.
```

The bullet satisfies all four PROJECT.md acceptance criteria:
- ✓ Cites `pin_table` + single-GPU-residency
- ✓ Cites `src/sirius_extension.cpp:733` verbatim with code excerpt
- ✓ Cross-references `PIN-MGPU-01` in REQUIREMENTS.md
- ✓ Cites Phase 13 / cudaStreamWaitEvent chain context (with `sirius_gpu_parquet_scan_operator.cpp:263` re-attach site)

**Commit attribution note:** my staged PROJECT.md edit was swept into parallel agent 20-01's commit `03aadb3` due to a git-staging race condition (see Deviations below). Content is identical to what Task 1 specified; the diff in `git show 03aadb3 -- .planning/PROJECT.md` confirms the bullet matches the plan's required content verbatim.

### Task 2 — REQUIREMENTS.md PIN-MGPU-01 augmentation (commit `25d89a5`)

Existing PIN-MGPU-01 entry on REQUIREMENTS.md line 70 (Future Requirements deferred to v1.5+) lacked:
1. Citation of `src/sirius_extension.cpp:733` (single-GPU residency proof)
2. Citation of Phase 13 cudaStreamWaitEvent re-attach site (`src/op/scan/sirius_gpu_parquet_scan_operator.cpp:263`)
3. Backreference to PROJECT.md Deferred section + Phase 20 SM-05 registration

Per plan Task 2 Branch B, augmented the entry:

**Before (line 70):**
```
- **PIN-MGPU-01** — Multi-GPU-aware `pin_table`. Place pinned splits on the GPU with
  the lowest free memory ratio (or distribute across GPUs by table-row count) instead
  of always GPU 0. Trade-off: P2P copy overhead via `convert_gpu_to_gpu` is acceptable
  in v1.4 because Phase 13 `cudaStreamWaitEvent` chain ensures correctness; perf gap
  is small at SF1 but may show at SF100 multi-table workloads.
```

**After:**
```
- **PIN-MGPU-01** — Multi-GPU-aware `pin_table`. Place pinned splits on the GPU with
  the lowest free memory ratio (or distribute across GPUs by table-row count) instead
  of always GPU 0 (`src/sirius_extension.cpp:733` — `auto& mem_space =
  const_cast<cucascade::memory::memory_space&>(*gpu_spaces[0]);`). Trade-off: P2P
  copy overhead via `convert_gpu_to_gpu` is acceptable in v1.4 because Phase 13
  `cudaStreamWaitEvent` chain (re-attached at
  `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:263`) ensures correctness; perf
  gap is small at SF1 but may show at SF100 multi-table workloads. Documented in
  PROJECT.md Deferred section; registered for v1.5+ scope during Phase 20 (SM-05).
```

All four Task 2 acceptance criteria PASS:
- ✓ PIN-MGPU-01 in Future Requirements section (verified via grep)
- ✓ References `src/sirius_extension.cpp:733`
- ✓ References Phase 13 cudaStreamWaitEvent context
- ✓ Branch B augmentation: PROJECT.md backref + Phase 20 SM-05 registration

## Verification

Plan-level automated verification:

| Gate | Command | Expected | Actual |
|------|---------|----------|--------|
| Task 1 grep | `grep -q "pin_table.*single-GPU" .planning/PROJECT.md && grep -q "sirius_extension.cpp:733" .planning/PROJECT.md && grep -q "PIN-MGPU-01" .planning/PROJECT.md` | exit 0 | exit 0 ✓ |
| Task 2 grep | `grep -q "PIN-MGPU-01" .planning/REQUIREMENTS.md` | exit 0 | exit 0 ✓ |
| Cross-doc cite (src) | `grep -c "sirius_extension.cpp:733" .planning/PROJECT.md .planning/REQUIREMENTS.md` | both 1+ | PROJECT.md=1, REQUIREMENTS.md=1 ✓ |
| Cross-doc loop (PROJECT.md → PIN-MGPU-01) | `grep -c "PIN-MGPU-01" .planning/PROJECT.md` | ≥ 1 | 1 ✓ |
| Cross-doc loop (REQUIREMENTS.md → PROJECT.md) | `grep -c "PROJECT.md" .planning/REQUIREMENTS.md` | ≥ 1 | 3 ✓ |

## Success Criteria

- ✓ SM-05 documentation gate: PROJECT.md Deferred section documents `pin_table` single-GPU residency as v1.4 limitation
- ✓ SM-05 follow-up gate: REQUIREMENTS.md Future Requirements registers PIN-MGPU-01 for v1.5+ scope
- ✓ ROADMAP Phase 20 success criterion 5 (SM-05 component) satisfied
- ✓ No source code changes; no test files touched; no other planning docs modified outside .planning/
- ✓ Both edits trace back to SM-05 + 20-RESEARCH.md verification of single-GPU-resident behavior at `src/sirius_extension.cpp:733`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking issue] Staging race with parallel 20-01 agent**

- **Found during:** Task 1 commit step
- **Issue:** When I ran `git commit --no-verify` for Task 1 (PROJECT.md edit only), parallel agent 20-01 had simultaneously staged additional files (ROADMAP.md, 20-01-EVIDENCE.md, etc.) and was in the process of committing. My commit attempt failed with "no changes added to commit" because by the time my commit ran, 20-01's commit had already swept up the staged PROJECT.md change as part of `03aadb3` ("docs(20-01): Task 1 — Static grep gates PASS"). I had earlier `git reset HEAD .planning/ROADMAP.md` to keep my commit scoped to PROJECT.md, but the next commit attempt was racing with 20-01.
- **Fix:** Verified that `03aadb3` contains the exact PROJECT.md content Task 1 specified (`git show 03aadb3 -- .planning/PROJECT.md` shows the bullet matches verbatim). No re-edit needed. Recorded `03aadb3` as the joint commit hash for Task 1; proceeded with Task 2 normally.
- **Files modified:** None (the edit landed in 20-01's commit; this is a commit-attribution issue, not a content issue).
- **Commit:** `03aadb3` (joint with 20-01)

**2. [Rule 3 — Blocking issue] `.planning/` directory is gitignored**

- **Found during:** Task 1 staging step
- **Issue:** `git add .planning/PROJECT.md` returned `paths are ignored by one of your .gitignore files`. The `.planning/` tree is gitignored at the project level but planning commits exist in history (verified via `git log -- .planning/`).
- **Fix:** Used `git add -f .planning/REQUIREMENTS.md` (force-add). Existing planning commit history confirms this is the project's standard pattern (e.g., `d318292`, `2f95f94` all touch `.planning/` files via `-f`).
- **Files modified:** None (workflow adjustment only)
- **Commit:** N/A

### Authentication Gates

None.

### Architectural Decisions

None.

## Branch Selection (Task 2)

**Branch B (augment) applied** rather than Branch A (verify-only). Rationale:

The pre-existing PIN-MGPU-01 stub at REQUIREMENTS.md:70 met the basic "registered for v1.5+ scope" requirement but did not satisfy the full SM-05 acceptance criteria, which require the entry to:
- Reference `src/sirius_extension.cpp:733` (single-GPU residency proof) — was MISSING
- Reference Phase 13 cudaStreamWaitEvent context — was PARTIAL (cited Phase 13 in general but not the re-attach site `sirius_gpu_parquet_scan_operator.cpp:263`)
- Cross-reference PROJECT.md Deferred + Phase 20 SM-05 registration — was MISSING (no loop-close)

Branch B augmentation adds all three for SM-05 completeness.

## Commits

| # | Task | Hash | Files |
|---|------|------|-------|
| 1 | PROJECT.md Deferred bullet | `03aadb3` (joint with 20-01) | `.planning/PROJECT.md` |
| 2 | REQUIREMENTS.md PIN-MGPU-01 augmentation | `25d89a5` | `.planning/REQUIREMENTS.md` |

## Self-Check: PASSED

- ✓ `.planning/PROJECT.md` Deferred section contains `pin_table.*single-GPU` bullet (line 92, verified via grep + Read)
- ✓ `.planning/PROJECT.md` Deferred bullet cites `src/sirius_extension.cpp:733` verbatim
- ✓ `.planning/PROJECT.md` Deferred bullet references `PIN-MGPU-01`
- ✓ `.planning/REQUIREMENTS.md` PIN-MGPU-01 entry contains `sirius_extension.cpp:733`
- ✓ `.planning/REQUIREMENTS.md` PIN-MGPU-01 entry references `cudaStreamWaitEvent` (Phase 13)
- ✓ `.planning/REQUIREMENTS.md` PIN-MGPU-01 entry references `PROJECT.md` (backref)
- ✓ `.planning/REQUIREMENTS.md` PIN-MGPU-01 entry references `Phase 20 (SM-05)` (registration)
- ✓ Commit `03aadb3` exists in `git log` and contains `.planning/PROJECT.md` change with the exact pin_table bullet
- ✓ Commit `25d89a5` exists in `git log` and contains `.planning/REQUIREMENTS.md` change with augmented PIN-MGPU-01
- ✓ No source code changes (no `src/` files touched); no test files (no `test/` files touched)
- ✓ No other `.planning/` files modified outside PROJECT.md and REQUIREMENTS.md by this plan
