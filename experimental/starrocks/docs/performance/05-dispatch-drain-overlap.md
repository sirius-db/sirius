**05 · Overlap local dispatch with remote drains**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: stop an otherwise ready same-CN receiver from waiting for all remote destinations of its producer to drain. Measurement prerequisite: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md). Full recovery rollout should include [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md); this path does not increase per-batch in-flight concurrency.

**Current behavior**

The sender completes execution, records local destinations, drains each remote destination synchronously, and returns ready local receivers afterward. The dispatch worker cannot run those returned receivers during the remote drain. This is independent of the per-batch transport serialization.

**Code map**

| Source | Planned change |
|---|---|
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L110) | Extend `FragmentOutcome` to carry remote completion ownership. |
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L242) | Dispatch local readiness before waiting for remote completions. |
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L1334) | Enqueue remote drains and return quickly. |
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L169) | Proposed enqueue/drain-ticket interface alongside blocking wrapper. |
| [parked_registry.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/parked_registry.rs) | Keep destination claims live until their individual drain completes. |
| [result_store.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/result_store.rs) | Surface asynchronous failures to the correct query/result. |

**Proposed design**

Split “drain accepted” from “drain complete.” A proposed `DrainTicket` owns query/producer/destination identity and a completion receiver. A bounded query completion supervisor observes tickets and propagates failures; the local dispatch worker must not become that blocking supervisor.

Keep FE RPC acknowledgement behavior explicit. A leaf RPC can still await all of its tickets on an appropriate worker after handing local readiness to dispatch. An intermediate receiver's dispatch loop must hand its tickets to the supervisor and remain available. Changing FE acknowledgement to “accepted” is a separate protocol decision, not an incidental side effect of this optimization.

Each parked output has one destination claim. Local relay releases only its claim; remote pack jobs and tickets retain theirs. Query cancellation stops new work, retires pending claims, and lets active transfer-owned buffers reach safe completion.

**Implementation slices**

1. **Ticket interface:** make transport enqueue return a completion object; retain a blocking adapter for current callers. Bound pending tickets and distinguish enqueue failure from transfer failure.
2. **Dispatch ordering:** return local ready fragments immediately after remote drains are accepted. Add a supervisor or continuation queue that cannot block dispatch or transport progress.
3. **Failure integration:** record the first meaningful asynchronous error in result/query state, cancel siblings as required, and resolve all tickets exactly once on shutdown.
4. **Scheduling observation:** add ticket enqueue/completion, local dispatch start, and engine queue waits. Preserve the single-thread agent and current sender sequence policy.

**Validation**

Use the existing compute-node service's fake executor/transport seams. Hold a remote drain unresolved, verify the local receiver is dispatched, then complete or fail the drain. Cover local/remote mixed destinations, all-local, all-remote, cancellation before completion, transport shutdown, and multiple ready descendants.

Acceptance: local receiver dispatch is observable before remote completion in the delayed-drain fixture; no destination loses its claim early; an asynchronous error reaches the FE-visible result and leaves unrelated queries intact. The dispatch worker can process another ready receiver while the first query still owns remote tickets.

**Performance experiment and limits**

Use a producer with one local receiver and one remote receiver, keeping producer output and FE plan constant. Vary remote delay and local receiver compute cost. Measure local start time, query critical path, parked bytes, and remote inter-batch idle gaps.

This does not make `Run` and export concurrent: they still share the engine queue. A long local run can delay the next remote export and erase some benefit. Record that explicitly and use [07 · Independent packing and CUDA completion](07-independent-gpu-packing.md) for independent packing. Do not treat a thread spawn or higher queue priority as a solution to an already-running engine call.

**Rollout and stop criteria**

Gate scheduling for new queries and retain the old blocking wrapper for A/B. Drain tickets before turning the mode off. Proceed if useful local overlap reduces the measured critical path without increased failures or memory growth. If export queue delay dominates, keep the ownership refactor but prioritize path 07 instead of adding more dispatch threads.
