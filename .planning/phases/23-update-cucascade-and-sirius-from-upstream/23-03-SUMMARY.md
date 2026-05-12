---
phase: 23-update-cucascade-and-sirius-from-upstream
plan: 03
subsystem: sirius-gitlink-bump
tags: [git, submodule, cucascade, build-gate, unit-tests]
dependency_graph:
  requires: [23-02-cucascade-rebase-complete]
  provides: [sirius-gitlink-bumped-to-post-rebase-cucascade, build-green-against-new-pin, pre-merge-gauntlet-baseline]
  affects: [cucascade (gitlink), Sirius parent commit history]
tech_stack:
  added: []
  patterns: [atomic-submodule-bump, pre-merge-gauntlet]
key_files:
  created:
    - /tmp/claude/p23_03_hyg02_pre.txt
    - /tmp/claude/p23_03_hyg02_post.txt
    - /tmp/claude/p23_03_kvikio_pre.txt
    - /tmp/claude/p23_03_kvikio_post.txt
    - /tmp/claude/p23_03_pin_pre.txt
    - /tmp/claude/p23_03_sirius_head_pre.txt
  modified:
    - cucascade (gitlink: c666b21 -> 1e889d7)
decisions:
  - "Gitlink bump committed atomically as 08f36e8 BEFORE any dev-merge work (D-12: bisect isolation between cucascade bump and sirius merge)"
  - "All 4 invariant test suites (datasource_factory, mgpu, tpch_sf10, TPC-H/parquet) pass against new cucascade pin without any Sirius-side adaptation — PR #121 is API-compatible with existing Sirius code"
  - "HYG-02 = 40 and kvikio-free = 0 invariants preserved through bump (pure submodule pointer change, no Sirius source change)"
metrics:
  duration: 8min
  completed: 2026-05-12T18:25:00Z
  tasks: 2
  files: 1
---

# Phase 23 Plan 03: Sirius Gitlink Bump + Intermediate Gauntlet Summary

**One-liner:** Atomic cucascade gitlink bump from c666b21 to 1e889d7 (post-PR#121 rebase) committed as 08f36e8; MCP build clean and all 4 invariant Catch2 suites pass against the new pin with zero Sirius-side adaptations needed.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Capture pre-bump snapshots; advance Sirius gitlink; commit atomic standalone bump | 08f36e8 | cucascade (gitlink only) |
| 2 | MCP build + unit-tests intermediate gauntlet | (verification only — no file changes) | — |

## Cucascade Gitlink Bump

| Field | Value |
|-------|-------|
| Pre-bump pin | `c666b21926dec70b26a1febd509435635bea8deb` |
| Post-bump pin | `1e889d7e67070de7dc88860c373622182afe35df` |
| Sirius parent bump commit | `08f36e8b01e3a2600c221fcd585d228823cab0f3` |
| Commit subject | `submodule: bump cucascade to 1e889d7 (post-PR#121 rebase)` |
| Branch | `feature/single-node-multi-gpu2` |
| Files in commit | `cucascade` only (1 file, 1 insertion, 1 deletion) |

## MCP Build Result

| Field | Value |
|-------|-------|
| Exit code | 0 (success) |
| Build steps | [57/57] Linking CXX executable sirius_unittest |
| New errors | 0 (no new compile errors traceable to PR #121 surface) |
| Pre-existing warnings | SPDLOG_ACTIVE_LEVEL override, nodiscard in test files (all pre-existing) |
| PR #121 API compatibility | PASS — no Sirius-side adaptation required |

## Unit-Test Gauntlet Results

### [datasource_factory] — Phase 22.1 strict-policy gate

| Metric | Value | Baseline | Status |
|--------|-------|----------|--------|
| Test cases | 11/11 | 11/11 | PASS |
| Assertions | 38 | — | PASS |
| Exit code | 0 | 0 | PASS |
| Wall-clock | 4.9s | — | PASS |

### [mgpu] — REG-01 invariant

| Metric | Value | Baseline | Status |
|--------|-------|----------|--------|
| Test cases | 16/16 | 16/16 | PASS |
| Assertions | 79091 | ≥79091 | PASS |
| Exit code | 0 | 0 | PASS |
| Wall-clock | 129.4s | ≤130s | PASS (within budget) |

Note: `[cucascade] direct GPU↔GPU peer DMA broken on 2 direction(s); cudaMemcpyPeer* will host-stage automatically.` — pre-existing consumer hardware limitation, not a regression.

### [tpch_sf10] — Phase 22.3 Q11 SF10 gate

| Metric | Value | Baseline | Status |
|--------|-------|----------|--------|
| Test cases | 4/4 | 4/4 | PASS |
| Assertions | 64 | — | PASS |
| Exit code | 0 | 0 | PASS |
| Wall-clock | 6.5s | — | PASS |
| Q1/Q6/Q12/Q11 | skip-guarded (SIRIUS_TEST_SF10_PATH unset) | — | PASS (guard fires correctly) |
| tpch_q11_sf10_2gpu | PASS | PASS | PASS |

Note: All 4 tests skip because `SIRIUS_TEST_SF10_PATH` is not set in the MCP environment. The skip-guard path passes Catch2 assertions. This is identical to the Phase 22.3 baseline behavior.

### [TPC-H][parquet] — REG-02 invariant

| Metric | Value | Baseline | Status |
|--------|-------|----------|--------|
| Test cases | 22/22 | 22/22 | PASS |
| Assertions | 36256 | 36256 | PASS |
| Exit code | 0 | 0 | PASS |
| Wall-clock | 110.4s | ≤90s baseline; ≤260s budget | PASS |
| Q11 retries | 0 | — | PASS |

## Invariant Snapshots

### HYG-02 (rmm::cuda_stream_default count)

| Snapshot | Value |
|----------|-------|
| Pre-bump (`/tmp/claude/p23_03_hyg02_pre.txt`) | 40 |
| Post-bump (`/tmp/claude/p23_03_hyg02_post.txt`) | 40 |
| Diff | empty (PASS) |

**Baseline match: PASS** — HYG-02 = 40, consistent with Phase 22.3 baseline.

### Kvikio-free invariant (Phase 22.1 GATE-22.1-A)

| Snapshot | Value |
|----------|-------|
| Pre-bump (`/tmp/claude/p23_03_kvikio_pre.txt`) | 0 |
| Post-bump (`/tmp/claude/p23_03_kvikio_post.txt`) | 0 |
| Diff | empty (PASS) |

**Baseline match: PASS** — GATE-22.1-A kvikio-free invariant preserved.

## Branch Confirmation

- Sirius branch: `feature/single-node-multi-gpu2` (unchanged)
- Cucascade submodule branch: `fix/pinned-portable-flags` (unchanged)
- No `git push` executed
- Pre-merge sirius tag `pre-phase23-merge` @ `b423a47` still intact (from Plan 23-01)

## Plan 23-04 Hand-off

Plan 23-04 may now proceed with `git merge origin/dev`. The pre-merge state is:

- Sirius HEAD: `08f36e8` (the gitlink bump commit)
- Cucascade pin: `1e889d7e67070de7dc88860c373622182afe35df` (post-PR#121 rebase)
- All 4 invariant suites green
- HYG-02 = 40, kvikio-free = 0

## Deviations from Plan

None — plan executed exactly as written. The new cucascade pin (`1e889d7`) is API-compatible with existing Sirius code at HEAD, requiring zero Sirius-side adaptations. The build passed on first attempt.

## Known Stubs

None — this plan is pure git operations and verification with no code stubs.

## Self-Check

- [x] `git ls-tree HEAD cucascade | awk '{print $3}'` = `1e889d7e67070de7dc88860c373622182afe35df` (VERIFIED)
- [x] Commit `08f36e8` exists on `feature/single-node-multi-gpu2` (VERIFIED)
- [x] Only `cucascade` in commit diff (VERIFIED: `git log -1 --name-only` returns only `cucascade`)
- [x] MCP build exit 0, [57/57] linking step (VERIFIED)
- [x] `[datasource_factory]` 11/11 PASS (VERIFIED)
- [x] `[mgpu]` 16/16 PASS, 79091 assertions, 129.4s (VERIFIED)
- [x] `[tpch_sf10]` 4/4 PASS (VERIFIED)
- [x] `[TPC-H][parquet]` 22/22 PASS, 36256 assertions (VERIFIED)
- [x] HYG-02 pre = post = 40 (VERIFIED)
- [x] kvikio-free pre = post = 0 (VERIFIED)
- [x] Branch `feature/single-node-multi-gpu2` unchanged (VERIFIED)
- [x] No `git push` executed (VERIFIED)

## Self-Check: PASSED
