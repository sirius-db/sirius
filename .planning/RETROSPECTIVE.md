# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MVP

**Shipped:** 2026-04-14
**Phases:** 2 | **Plans:** 3 | **Sessions:** 3

### What Was Built
- Header-only `inspectable_mpsc<T>` template class with full MPSC queue API (push, pop, emplace, interrupt, reactivate, drain)
- Four predicate-based inspection methods (pop_if, get_if, mutable_pop_if, mutable_get_if) with bidirectional search
- 35 Catch2 tests (231 assertions): single-threaded, concurrency stress, and predicate inspection

### What Worked
- TDD workflow (RED then GREEN commits) caught no regressions and kept each plan focused
- Coarse granularity planning (2 phases, 3 plans) was right-sized for a focused library — no planning overhead
- Phase 1/Phase 2 split (core then differentiating features) let Phase 1 be testable independently
- Worktree-based execution kept main branch clean during development
- All 3 plans executed with zero deviations from plan

### What Was Inefficient
- Worktree submodule initialization failed in Phase 2 and required manual symlink workaround — recurring friction
- sccache sandbox restrictions caused first-build failure in Phase 1 — needed sandbox override
- REQUIREMENTS.md traceability table was never updated during execution (all remained "Pending") — archive had to fix retroactively

### Patterns Established
- `mutex+condition_variable` for MPSC queue (not atomic polling) — validated under 4-producer stress
- `std::next(rit).base()` for reverse iterator erase on deque — standard idiom for this codebase
- Timeout guard pattern using `steady_clock` for threaded test assertions

### Key Lessons
1. For small, focused libraries (< 1500 LOC), 2-3 phases with coarse granularity is the sweet spot — more phases would be overhead
2. TDD with atomic RED/GREEN commits makes code review trivial and provides clean git bisect targets
3. Worktree submodule issues need a one-time setup script rather than ad-hoc fixes each phase

### Cost Observations
- Model mix: quality profile (opus-heavy)
- Sessions: 3 (discuss+plan Phase 1, execute Phase 1, discuss+plan+execute Phase 2)
- Notable: ~69 min active execution for 1,153 LOC — high efficiency for thread-safe concurrent code

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 3 | 2 | Initial milestone — TDD, coarse granularity |

### Cumulative Quality

| Milestone | Tests | Assertions | LOC |
|-----------|-------|------------|-----|
| v1.0 | 35 | 231 | 1,153 |

### Top Lessons (Verified Across Milestones)

1. TDD with atomic commits keeps plans focused and review simple
2. Coarse granularity is sufficient for focused library work
