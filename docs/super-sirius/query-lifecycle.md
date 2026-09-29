# Query lifecycle and submission tracking

Sirius has one shared lifecycle registry with a control block for each registered query.
Its gate decides whether new work may be submitted. Its counters track publishers and
explicit work leases. Closing one query's gate does not close another query's gate.

**This is an incremental concurrency foundation.** Query execution is still single-flight.
The existing creator counters, executor joins, and global downgrade drains remain necessary.
Production enqueue paths now track publication; they do not yet carry work leases through
every asynchronous operation. The registry does not own physical plans or repositories.

## Two kinds of responsibility

| Handle | Lifetime | Purpose |
|---|---|---|
| `submission_guard` | From admission of a publisher through its queue insertion or abandonment | Prevent cleanup from draining ahead of a publisher that already passed the gate |
| `work_lease` | Until the last use of the resources borrowed by an operation | Keep that operation counted across queue removal, worker handoffs, retries, and destruction |

Both handles are movable and non-copyable. Moving transfers responsibility without releasing
and reacquiring it. Destruction releases the corresponding count. A handle retains the control
block, so releasing it does not dereference a destroyed registry. This protects accounting
storage only; callers must still preserve the resources being used.

Acquisition briefly locks the query's control block to check the gate and increment the counts
atomically. No lifecycle mutex is held during queue insertion, execution, or callbacks. The
registry map lock is released before waiting on an individual query.

Lookup and admission use different mutexes: the registry mutex protects map membership, and
the query mutex protects that query's gate and counts. The lookup returns an owned reference,
allowing the registry mutex to be released before acquiring the query mutex. An unlocked
lookup would still require both protections. Lease release only acquires the query mutex.

A submission guard initially stores one control-block reference and owns both counts. If its
work claim has not been transferred, finishing releases both counts under one query lock.
Notifications are sent only when a quiescing query's wait predicate may have become true.

## Publication protocol

`accepts_work()` remains an advisory snapshot. A successful result does not authorize a later
push: cleanup may close the gate and drain the queue between the check and the push.

Production publishers instead retain a submission guard through insertion:

```cpp
auto submission = lifecycle.try_begin_submission(query_id);
if (!submission) {
  // status() distinguishes an unknown registration from expected quiescence.
  return;
}
auto request = make_request();
queue.push(std::move(request));
// submission's destructor settles publication, including on exception.
```

The guard also holds one work lease until it finishes. A future producer that carries lifetime
accounting in its queued work can transfer that claim with `submission.take_work_lease()`.
Finishing publication then releases only the publisher count; the work item's lease remains
counted until destruction. Independent resource borrowers can use `try_acquire_work()`.

Every queue publication needs a submission guard, including transfers of already-counted
tasks and resubmission after OOM or spilling. A work lease alone does not authorize another
insertion after the gate has closed. Guard acquisition failure preserves the reason observed
at acquisition, so callers do not misdiagnose a quiescing query as unknown after it is erased.

Current production coverage:

- Creator `schedule()` overloads and lookahead publication.
- Scheduler task insertion.
- Executor insertion, including OOM resubmission.
- Tier-2 spill task return to the scheduler queue.

These guards are scoped to publisher functions. They do not yet accompany the requests/tasks
through subsequent execution. Existing ownership must protect a caller's operator/task while
it obtains the query ID, and protect a refused task during disposal. Those requirements are
not established by a gate lookup.

Keep a guard short-lived: never retain it while waiting for worker capacity, memory, or task
completion. Declare it before locally owned work so exception unwinding destroys unsubmitted
work before settling the publisher. Existing standalone components with no bound registry
retain their previous behavior; production binds the shared registry.

## Quiescence and cleanup

`quiesce(query_id)` closes the gate without waiting. Previously admitted publishers can still
finish inserting work. Workers may use this nonblocking transition without waiting on themselves.

Before draining queues, an external cleanup thread calls:

```cpp
lifecycle.quiesce_and_wait_for_submissions(query_id);
// All previously admitted publishers have settled; no new publisher may enter.
// Run the existing queue drains and worker/downgrade joins.
lifecycle.close(query_id);
```

`wait_for_submissions()` is also available after a separate `quiesce()` call. The scheduler's
success and error cleanup, and the context's mandatory and best-effort cleanup, use the barrier
before their existing drains. The guard releases its publisher count and any untransferred work
count together under the query mutex, so a waiter cannot observe leftover untransferred accounting.

Do not wait while holding this query's submission guard, a queue lock, or any other lock a
publisher needs. Draining work may run callbacks; no lifecycle lock is held across drains.

`wait_for_work()` waits for explicit leases and publishers after quiescence. Its eventual use
is after publication has settled and queued work has been disposed of. It must never be called
by a holder of one of the leases it waits for. **Production cleanup does not yet use this as a
replacement for existing drains:** not every asynchronous borrower is represented by a lease.

For full lifetime coverage, the next stage must keep the plan, repositories, and scan providers
alive through the final task destructor, callback, spill borrow, and CUDA completion. Logical
query completion and resource retirement are separate milestones.

## Registration and removal

- `open_query()` creates a fresh control block. Duplicate registration throws, whether the
  existing entry is open or quiescing; it cannot silently reopen a gate being drained.
- Unknown IDs refuse work and submissions. Quiesce/wait/close on an unknown ID are harmless
  for partial initialization rollback and repeated cleanup.
- `close()` refuses to erase outstanding publisher or work counts. It closes the gate before
  erasing an idle entry, including for callers that already obtained its control block.
- `clear()` is for runtime shutdown after workers have stopped. It closes every gate and
  refuses to discard live accounting. Query IDs are not to be reused for delayed old work.
- Waiting on a known, still-open query is an error: another publisher could otherwise race a
  zero-count observation. Closing publication must precede retirement waits.

## Tests

`test/cpp/exec/test_query_lifecycle_registry.cpp` tests paused publishers, post-pop work leases,
move/destruction accounting, refusal reasons, exceptions, interrupted queues, duplicate
registration, and removal with live handles. The scheduler and executor tests check production
guard lifetimes and the barrier before scheduler error cleanup.

The post-pop test demonstrates the lease primitive's behavior; it does not claim that all
production handoffs have already been converted. End-to-end concurrent query admission and
query-local retirement remain follow-up work.
