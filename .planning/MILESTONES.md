# Milestones

## v1.0 — Downgrade Executor Redesign

**Shipped:** 2026-04-06
**Phases:** 3 | **Plans:** 6 | **Requirements:** 20/20

Redesigned the downgrade_executor from an itask_executor-derived task scheduler into a standalone request-queue executor with predicate-driven dispatch, async/sync public APIs, and full pipeline integration.

**Key accomplishments:**
1. Standalone executor with own thread pool, decoupled from itask_executor hierarchy
2. Predicate-driven incremental dispatch engine with async/sync/custom-predicate APIs
3. Retry-with-downgrade loop in gpu_pipeline_executor (5 retries on reservation shortfall)
4. SiriusContext initialization reordered for correct dependency ordering
5. 6 lifecycle tests covering start/stop, drain, monitor loop, concurrency, CUDA streams
6. All 20 v1 requirements satisfied (RAPI, EXEC, LIFE, CAND, PIPE)

**Archives:** `milestones/v1.0-ROADMAP.md`, `milestones/v1.0-REQUIREMENTS.md`
