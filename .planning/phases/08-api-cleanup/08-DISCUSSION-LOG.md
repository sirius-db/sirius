# Phase 8: API Cleanup + Processing Loop Refactor - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-16
**Phase:** 08-api-cleanup
**Areas discussed:** Candidate collection, request_free_memory scope, Logging replacements, downgrade_task elimination, Tiered fallback ordering, Predicate threading safety, collect_all_candidates removal

---

## Phase Fusion Decision

User decided to fuse Phase 8 (API Cleanup) and Phase 9 (Processing Loop Refactor) into a single phase. Rationale: removing `target_bytes` and replacing the collection mechanism with convertible_data providers are tightly coupled — a bridge strategy between the two phases would be throwaway work.

---

## Candidate Collection

| Option | Description | Selected |
|--------|-------------|----------|
| Collect all, predicate stops dispatch | Remove byte limit, predicate short-circuits | |
| Keep internal byte estimate | Bridge approach with internal estimate | |
| You decide | Claude picks | |

**User's choice:** User clarified that `collect_all_candidates` should not be used at all — replaced by `convertible_data_batch_provider` per data_repository, fetching lazily until predicate is satisfied. This triggered the phase fusion decision.
**Notes:** This was the key insight that made fusing Phase 8+9 the right call.

---

## request_free_memory Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Same public API, new internals | Keep signatures, build predicates internally | ✓ |
| Merge into request_downgrade | Eliminate request_free_memory, callers build predicates | |
| You decide | Claude picks | |

**User's choice:** Same public API, new internals
**Notes:** No external caller changes needed.

---

## Logging Replacements

| Option | Description | Selected |
|--------|-------------|----------|
| Per-request summary with tier breakdown | Single DEBUG log with repos/gpu_queue/pipeline_queue breakdown | ✓ |
| Separate per-tier TRACE + existing summary | TRACE lines per tier + DEBUG summary | |
| You decide | Claude picks | |

**User's choice:** Per-request summary with tier breakdown
**Notes:** Satisfies LOG-01 requirement.

---

## downgrade_task Elimination

| Option | Description | Selected |
|--------|-------------|----------|
| Eliminate entirely | Replace with convertible_data::convert() directly | ✓ |
| Keep as thin wrapper | Delegate to convert() but keep the type | |
| You decide after analysis | Claude compares during planning | |

**User's choice:** Eliminate entirely
**Notes:** convert() already handles state transitions, failure rollback, and converter registry access.

---

## Tiered Fallback Ordering (Queue Access)

| Option | Description | Selected |
|--------|-------------|----------|
| Constructor injection | Add optional queue refs to constructor | |
| Setter methods post-construction | Add setter methods for queues | |
| You decide | Claude analyzes construction order | ✓ |

**User's choice:** You decide
**Notes:** Claude will analyze executor construction order in sirius_context/sirius_engine during research.

---

## Predicate Threading Safety

| Option | Description | Selected |
|--------|-------------|----------|
| Dispatch loop only | Check predicate in main loop only | |
| Both dispatch loop and workers | Workers check after convert(), dispatch loop between dispatches | ✓ |
| You decide | Claude picks | |

**User's choice:** Both dispatch loop and workers
**Notes:** Thread-safe predicate contract preserved from current behavior.

---

## collect_all_candidates Removal

| Option | Description | Selected |
|--------|-------------|----------|
| Remove in this phase | Clean up all dead code as part of this phase | ✓ |
| Leave for separate cleanup | Focus on new code path, clean up later | |

**User's choice:** Remove in this phase
**Notes:** Same pattern as Phase 3 (dead code removal).

---

## Claude's Discretion

- Queue wiring mechanism (constructor injection vs setters)
- Thread pool dispatch strategy for new provider-based loop
- Exact processing loop structure

## Deferred Ideas

None — discussion stayed within phase scope.
