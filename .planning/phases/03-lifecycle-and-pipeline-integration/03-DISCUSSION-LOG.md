# Phase 3: Lifecycle and Pipeline Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 03-lifecycle-and-pipeline-integration
**Areas discussed:** Pipeline retry strategy, Downgrade executor access, Lifecycle verification, drain() shared_ptr guarantee

---

## Pipeline retry strategy

### Retry delay

| Option | Description | Selected |
|--------|-------------|----------|
| No delay (Recommended) | Retry immediately after request_free_memory_and_wait returns. The blocking call itself is the wait. | ✓ |
| Brief backoff | Small sleep (1-5ms) between retries to let in-flight GPU work settle. | |
| You decide | Claude picks based on what makes sense during implementation. | |

**User's choice:** No delay
**Notes:** None

### Partial reservation handling

| Option | Description | Selected |
|--------|-------------|----------|
| Proceed with partial (Recommended) | Accept whatever reservation was obtained and execute with reduced memory. Current behavior. | ✓ |
| Fail the task | Report an error and skip this pipeline task. | |
| You decide | Claude picks based on what makes sense during implementation. | |

**User's choice:** Proceed with partial
**Notes:** None

---

## Downgrade executor access

| Option | Description | Selected |
|--------|-------------|----------|
| Constructor parameter (Recommended) | Add downgrade_executor* to gpu_pipeline_executor's constructor. Direct, explicit dependency. | ✓ |
| Via SiriusContext lookup | gpu_pipeline_executor calls SiriusContext::get_downgrade_executor(space_id) at runtime. | |
| You decide | Claude picks based on what makes sense during implementation. | |

**User's choice:** Constructor parameter
**Notes:** None

---

## Lifecycle verification

| Option | Description | Selected |
|--------|-------------|----------|
| Verify with tests only (Recommended) | Write unit tests exercising start/stop/drain, thread safety, CUDA stream lifecycle. No code changes unless tests reveal gaps. | ✓ |
| Review and harden | Audit existing implementation for edge cases, add defensive guards, then write tests. | |
| Skip — already covered | Mark LIFE requirements as complete based on existing Phase 1-2 test coverage. | |

**User's choice:** Verify with tests only
**Notes:** None

---

## drain() shared_ptr guarantee

| Option | Description | Selected |
|--------|-------------|----------|
| Current impl is sufficient (Recommended) | pool->wait_all() ensures all dispatch lambdas returned, releasing shared_ptr captures. Queue drain drops pending requests. | ✓ |
| Add explicit verification | After wait_all(), walk all repos and assert no batches have in_transit or processing state. | |
| You decide | Claude evaluates during implementation and adds verification only if needed. | |

**User's choice:** Current impl is sufficient
**Notes:** None

---

## Claude's Discretion

- Retry loop structure (for vs while)
- Reservation release/re-acquire vs grow strategy
- Logging verbosity for retry attempts
- Test fixture design for lifecycle tests
- downgrade_executor* storage in gpu_pipeline_executor

## Deferred Ideas

None — discussion stayed within phase scope
