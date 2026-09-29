# Query lifetime and retirement

The shared lifecycle registry has one control block per query. Its gate decides whether new
work can be published or resources borrowed. Unknown IDs refuse both operations.

## Ownership and handles

| Object | Responsibility |
|---|---|
| Engine | Builds the physical plan, starts execution and obtains the logical result |
| Lifecycle control | Retains the physical plan independently of engine destruction until retirement |
| `submission_guard` | Counts a publisher through queue insertion or abandonment |
| `work_lease` | Counts asynchronous resource use through queue removal, dispatch, callbacks and destruction |
| Runtime query registries | Own creator state, repositories and scan providers until their users retire |

Handles are movable and non-copyable. Queue removal transfers a handle without dropping its
count. The registry map mutex protects membership; a separate per-query mutex protects the
gate and accounting. Neither mutex is held during work, queue insertion or resource destruction.

`accepts_work()` is an advisory snapshot. It cannot authorize a later queue push. Publishers use
`try_begin_submission()` and retain the guard through insertion. `take_work_lease()` transfers
the guard's work claim into a request or task; the guard then accounts only for publication.
Standalone test components without a bound registry retain their previous behavior.

Creator requests keep their lease through hint traversal and the creation worker. Execution
tasks keep theirs through scheduler routing, executor dispatch, retries and destruction. The
GPU worker transfers the lease to its completion epilogue when it destroys the task before
scheduling consumers. A rejected handoff leaves the caller's existing claim intact through
disposal. Queue drains detach under their mutex and destroy outside it.

Tier-1 spilling acquires a borrow for the actual victim query before inspecting its repositories.
Each dispatched candidate receives its own lease. A manager snapshot alone does not pin every
query, and shared ownership of a repository manager alone does not protect repositories being
cleared. Tier-2 extraction carries the task's existing lease through conversion and return.

Scan producers use their scoped dispatcher join: `quiesce(query_id)` stops producers while
retaining providers and buffers for downstream consumers. This is separate from task accounting.

## Retirement

Logical completion may precede the final callback. Retirement must finish before resources
are released:

1. Close Q's publication/borrow gate and wait for admitted publishers.
2. Stop Q's scan/prefetch producers, retaining their buffers.
3. Drain Q's creator, scheduler and executor queues; leave other queries' workers running.
4. Wait for Q's work leases, including in-hand tasks, completion callbacks and spill victims.
5. Release Q's plan, repositories, scan state and telemetry in dependency order.
6. Close the registry entry; the surrounding execution scope releases admission.

An existing lease does not authorize publication after closure. Every queue handoff, including
retry and spill return, still needs a submission guard. No worker waits for its own lease.

The engine retires asynchronous users before member destruction. The registry retains the plan
as a backstop if cleanup cannot prove quiescence. Such a cleanup failure marks the runtime
unavailable and preserves the registered state for shutdown, rather than freeing borrowed
resources. Ordinary query errors and initialization failures do not by themselves poison the
runtime. Initialization rolls back only the new query's registrations.

`close()` and `release_resources()` reject live accounting. `clear()` is reserved for shutdown
after workers stop and queues drain. Resource destructors run outside registry locks. Query IDs
must not be reused for delayed work.

## Validation and scope

Registry tests cover publication races, in-hand leases, move accounting, exceptions, duplicate
registration and retained resources. Scheduler tests hold a task after removing it from the
production queue and prove retirement waits for its disposal. Queue tests exercise reentrant
destructors. SQL lifecycle tests cover cleanup and subsequent queries.

These ownership changes alone do not enable bounded concurrent admission. Admission, memory
progress, settings isolation and scan/prefetch qualification are tracked separately in the
concurrency implementation journal.
