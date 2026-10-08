# Query lifetime and retirement

`query_lifecycle_registry` provides per-query publication gates and explicit work
accounting. This layer introduces the contract and its tests. Runtime publishers,
spill borrowers and retirement coordinators will adopt it together in the next
layer; SQL admission and existing runtime cleanup are unchanged here.

## Ownership and handles

| Object | Responsibility |
|---|---|
| Registry | Owns the map of registered queries and their control blocks |
| Query control block | Holds the gate, counters and an optional retained resource owner |
| `submission_guard` | Counts a publisher through queue insertion or abandonment |
| `work_lease` | Counts resource use through queue removal, dispatch, callbacks and destruction |
| Runtime owner | Stops producers, drains work and waits for users before releasing resources |

Handles are movable and non-copyable. Moving a handle transfers its claim without
a decrement/reincrement gap. The registry map mutex protects membership; a separate
per-query mutex protects the gate and accounting. Lookup releases the map mutex
before taking the query mutex. Neither mutex is held during queue insertion or
resource destruction.

Unknown IDs refuse both publication and resource borrowing. Registration must
precede any producer; duplicate registration cannot reopen a quiescing query.
Query IDs must not be reused while delayed work could still refer to them.

## Publication

`accepts_work()` is an advisory snapshot. It cannot authorize a later queue push.
`try_begin_submission()` checks the gate and counts the publisher atomically.
The returned guard records whether admission succeeded, the query was quiescing,
or its ID was unknown; callers do not need a second lookup to diagnose refusal.

An accepted guard initially owns one work claim. `take_work_lease()` transfers it
to a request or task, leaving the guard responsible for publication. For example,
with a request type that owns the lease and a queue that retains caller ownership
on refusal:

```cpp
auto publication = registry.try_begin_submission(query_id);
if (publication) {
  auto request = std::make_unique<Request>(publication.take_work_lease());
  queue.try_push(request);
}
```

The guard is declared before the owned request: rejection or an exception destroys
the request and its callbacks before releasing the publisher. Keep the guard
through insertion or abandonment, rather than through task execution or a blocking
capacity wait. Existing leases do not authorize publication after quiescence;
every retry or queue handoff needs its own submission guard.

Independent resource users acquire a lease with `try_acquire_work()`. Integration
must carry leases through every interval of use, including callbacks and device
completion. A worker slot covers worker capacity; it does not cover a task held
between queue pop and worker attribution.

## Retirement contract

The eventual runtime coordinator must perform these phases in order:

1. Close the publication/borrow gate with `quiesce_and_wait_for_submissions()` and
   wait for already admitted publishers to finish inserting or abandoning work.
2. Stop producers that use separate lifetime tracking, retaining their buffers.
3. Drain the query's queued work on cancellation, or validate/settle successful
   execution without silently discarding its work.
4. Call `wait_for_work()` after queued work can no longer hold unretired leases.
5. Release retained resources and other query-owned state, then close the entry.

`quiesce()` closes the gate without waiting. `wait_for_submissions()` exposes the
separate publication barrier for coordinators that already closed it. Waiting on
an open gate is rejected. No caller may wait for its own submission guard or lease,
or hold a lock needed by a user it is waiting for.

`retain_resources()` requires registration and can keep a physical plan alive
independently of its engine. `release_resources()` requires a quiescing, idle
query. `close()` refuses live accounting and closes an idle query's gate before
removal. `clear()` closes all gates and checks all entries before removing any;
it is intended for shutdown after producers and workers stop. Repeated cleanup
of unknown IDs is harmless.

Resource destructors run outside registry locks and may reenter the registry.
Shared control blocks keep accounting storage alive if handles outlive the
registry; this does not replace the runtime owner's obligation to retire users.
Only explicitly counted users are covered by the work barrier.

## Validation

The primitive tests cover unknown IDs, duplicate registration, move accounting,
paused publication, work held after queue pop, both publisher/work release orders,
refused pushes, exception unwinding and reentrant retained-resource destruction.
Runtime integration, device health and admission qualification arrive in later
layers of the concurrency stack.
