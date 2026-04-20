---
phase: 03-numa-aware-memory-and-transfer-optimization
plan: 01
subsystem: testing
tags: [numa, downgrade, cucascade, memory-reservation, multi-gpu]

# Dependency graph
requires:
  - phase: 01-multi-gpu-foundation
    provides: NUMA-aware downgrade executor and cucascade any_memory_space_in_tier_with_preference strategy
provides:
  - NUMA downgrade ordering verification tests proving MEM-01 (local preference) and MEM-02 (cross-NUMA fallback)
affects: [03-numa-aware-memory-and-transfer-optimization]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-gpu memory manager test helper, cucascade strategy candidate ordering verification]

key-files:
  created: []
  modified:
    - test/cpp/downgrade/test_downgrade_executor.cpp

key-decisions:
  - "Used cucascade get_candidates() for candidate ordering verification instead of memory exhaustion approach for cross-NUMA fallback test"

patterns-established:
  - "make_multi_gpu_memory_manager(): reusable 2-GPU test helper for NUMA verification tests"
  - "Direct cucascade strategy.get_candidates() testing pattern for verifying reservation ordering"

requirements-completed: [MEM-01, MEM-02]

# Metrics
duration: 22min
completed: 2026-04-03
---

# Phase 3 Plan 1: NUMA Downgrade Ordering Verification Summary

**3 tests proving NUMA-local HOST preference (MEM-01) and cross-NUMA fallback ordering (MEM-02) via end-to-end downgrade and cucascade candidate ordering verification**

## Performance

- **Duration:** 22 min
- **Started:** 2026-04-03T17:53:40Z
- **Completed:** 2026-04-03T18:15:13Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Test `numa_downgrade_prefers_local_host_space` proves data on GPU 0 downgrades to HOST space with device_id=0 when NUMA preference is set to 0
- Test `numa_downgrade_falls_back_to_cross_numa_host` verifies candidate ordering: pref=0 puts device_id=0 first, pref=1 puts device_id=1 first
- Test `numa_downgrade_candidate_ordering_verified` covers pref=0, pref=1, and pref=nullopt (backward compat) candidate ordering
- All pre-existing downgrade_executor (9 tests) and numa_aware_downgrade (3 tests) still pass

## Task Commits

Each task was committed atomically:

1. **Task 1: NUMA downgrade ordering and cross-NUMA fallback tests** - `ec2399ef` (test)

**Plan metadata:** (pending final commit)

## Files Created/Modified
- `test/cpp/downgrade/test_downgrade_executor.cpp` - Added 3 new TEST_CASE blocks tagged [numa_downgrade_verification] and make_multi_gpu_memory_manager helper

## Decisions Made
- Used cucascade `get_candidates()` direct verification approach for the cross-NUMA fallback test (Test 2) instead of attempting to exhaust memory in one HOST space. This is more reliable and directly validates the stable_partition ordering that drives fallback behavior.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- NUMA downgrade ordering verified end-to-end
- Ready for Plan 03-02 (P2P transfer and scan distribution verification tests)

---
*Phase: 03-numa-aware-memory-and-transfer-optimization*
*Completed: 2026-04-03*
