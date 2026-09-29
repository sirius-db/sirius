# concurrency4 implementation journal

This branch implements the recommendations in `100-concurrent-query-implementation-plan.md`.
Entries describe review boundaries, design decisions, and validation. A listed future step is
not an implementation claim. No branches have been published.

## Baseline and recovery

- Original `concurrency4`: `98280218b` (same as `concurrency3`).
- Backup: `concurrency4_before_rebase_20260929`.
- New base: `concurrency3_0_1` at `432f0630b`.
- Replayed the six concurrency3 commits. Resolved scheduler and creator conflicts by retaining
  the submission barriers alongside query-specific state and drains. Updated the old creator
  drop diagnostic to call the registry's advisory check directly.
- The rebased baseline required one creator diagnostic fix (`accepts_work` had been renamed).
  A clean full build completed; the first implementation commit includes that compatibility fix.

## Design decisions

- Supported concurrency means separate connections sharing one DatabaseInstance. One connection
  remains serialized by DuckDB. Independent DatabaseInstances sharing GPU allocator state are
  not covered by this contract.
- Default admission remains one query. Startup `sirius.max_concurrent_queries` will be the
  authority for admission and scan capacity; the old scan-manager key is a compatibility alias.
- Resource mutation (pin, unpin, cache reset, index operations) retains exclusive maintenance
  access. Waiting maintenance blocks new query admission.
- Query lifetime accounting belongs to asynchronous work, independently of worker capacity.
  Cleanup closes publication, drains query queues, waits for borrowers, and only then destroys
  plans, repositories and scan state.
- Ordinary errors remain query-local. Device loss/shared corruption can invalidate the runtime.
- Equal-share scheduling, kernel preemption, online cache replacement, and multi-GPU late
  materialization are outside the requested initial concurrency contract, as in the plan.
- Available hardware has one GPU. Multi-GPU paths will require qualification on a multi-GPU
  host; single-GPU tests cannot establish that qualification.

## Planned review boundaries

1. Continuous work leases, safe queue disposal, and retirement before plan destruction.
2. Query-local spilling, cleanup and exception containment.
3. Scheduling, memory progress and cancellation.
4. Shared metadata, configuration snapshots and maintenance/admission.
5. Scan/prefetch resource coordination and multiple GPUs.
6. Concurrent integration qualification and public documentation.

## Commit log

Implementation entries will be added here as each buildable change is validated and committed.

### 1. `fix(exec): retain query work across asynchronous handoffs`

- Creator requests and execution tasks carry the existing move-only work lease from publication
  until final disposal. Creator hint traversal is covered, including the interval before a
  worker is attributed to a query. GPU completion retains the lease after task destruction until
  downstream callbacks finish.
- Query drains detach work under the queue mutex and destroy it afterward. A reentrant destructor
  regression exercises both whole-queue and query-specific drain.
- Scheduler retirement waits for query work after settling publishers and draining queues.
  Production error cleanup no longer interrupts/restarts the shared GPU manager. Standalone
  executors without a lifecycle registry retain the old compatibility path.
- Rejected executor dispatch does not consume a parked GPU's readiness credit.
- Scan producers can be stopped without destroying their providers. Engine destruction stops
  producers and retires tasks before releasing the physical plan, including partial initialization.
- Validation: full `pixi run make`; queue regressions (27 cases, 140 assertions). Component tests: 45 cases,
  193 assertions; SQL lifecycle tests: 11 cases, 78 assertions. Formatting checks passed.
- Remaining dependencies: repository spill borrows and per-query downgrade cleanup are the next
  change; bounded admission is not enabled by this commit.

### 2. `fix(exec): isolate spill borrowing and failed query cleanup`

- Spill discovery acquires one victim's work lease at a time; dispatched candidates retain
  independent victim leases. Closing Q prevents new borrows. Retirement waits for Q's borrowers
  without canceling or draining shared downgrade requests.
- The lifecycle control now retains the physical plan. Engine destruction and failed cleanup
  cannot free the plan behind a task or spill borrower. Resource release checks quiescence and
  idle accounting and invokes destructors outside registry locks. Failed retirement latches
  runtime health and retains registered resources for shutdown.
- Ordinary registration/setup failures roll back their query, rather than marking the entire
  runtime unavailable. Execution polls DuckDB interruption while waiting for completion.
- Queue refusal reports an attributed failure before task destruction can signal success.
  Creator hints/lookahead, scheduler routing, GPU retry and completion exceptions reach the
  affected query's handler.
- Shutdown closes all lifecycle gates and drains stopped scheduler/executor queues while
  callback dependencies remain alive, then releases retained plans.
- Validation: full `pixi run make`; CPU registry tests (15 cases, 128 assertions);
  lifecycle, downgrade and runtime fallback tests (32 cases, 244 assertions).
- Previous commit: `671e8bd64`.

### 3. `fix(exec): yield GPU workers while awaiting memory`

- Failed reservations return tasks to the shared, spill-visible scheduler with a retry deadline.
  GPU workers do not wait for downgrade futures or retain partial reservations. Reclamation is
  rate limited per GPU; a task that cannot reserve for 30 seconds fails its own query.
- OOM retries also yield their worker. The scheduler chooses the oldest runnable compatible
  task across device-specific and unbound queues. Lookahead can advance a younger query when
  an older query cannot currently produce runnable work.
- HOST result transfer requires a successful reservation; it no longer bypasses accounting when
  memory is unavailable. Query priorities reject exhausted 31-bit IDs instead of wrapping.
- Validation: clean full `pixi run make`; focused scheduler, executor, OOM, query ID and lifetime
  tests passed (41 cases, 209 assertions). The existing two-GPU affinity test skipped on this
  one-GPU host; the new FIFO and retry-deadline tests executed.
- Previous commit: `2b3acbad0`.

### 4. `fix(metadata): retire shared bookkeeping by query owner`

- Batch telemetry placements and consumer ports carry query IDs. Retirement removes only that
  query's records, including when a pinned batch has placements in several queries.
- Iceberg delete-data memoization belongs to the connection and is cleared by its statement-end
  callback, including planning declines. Internal metadata queries preserve the outer memo.
- Cache summary baselines are synchronized. File-based pin compression plans are selected per
  invocation, so a process-global bare table name cannot reuse another catalog's policy.
- Validation: full `pixi run make`; Iceberg and telemetry regressions passed (70 cases, 1,344 assertions), after installing
  the required official DuckDB avro/Iceberg test extensions.
- Previous commit: `2b233ac6d`.
