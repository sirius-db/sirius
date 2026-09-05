**04 · Early ingress and bounded receive credits**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: make receive arena occupancy depend on active ingress work rather than the full materialized input of a receiver. Prerequisites: [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md) and [03 · Spillable exchange repositories and reload](03-exchange-spill-and-reload.md); measurement follows [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md).

**Current behavior and code map**

| Source | Current behavior / change |
|---|---|
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L658) | Arrival stores a descriptor; change to owned ingress admission. |
| [local_exchange.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/local_exchange.rs#L393) | Retains batches until all senders reach EOS; retain input handles and ingestion completion instead. |
| [engine.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/engine.rs#L547) | Builds, copies every remote input, releases leases, then runs; attach already ingested data. |
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L786) | Current `push_packed` requires a built fragment; factor a buffer-only ingress API. |
| [exchange_staging_arena.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/exchange_staging_arena.cpp) | Allocation and per-class credit accounting. |
| [fragment_executor.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/fragment_executor.rs) | Proposed neutral ingress handle and completion contract. |

**Proposed design**

Create an ingress service with its own bounded job queue and C++ resource owner. It accepts a validated lease token, format/schema metadata, row count, and destination identity. It must not borrow a thread-affine `Fragment` or call its current `push_packed` concurrently.

The service takes durable ownership at publication, orders copy after transport visibility, materializes a memory-managed batch or supported host-packed spill representation, and returns destination credit only after all reads from the lease complete. Publication acknowledgement and credit return are separate events. The rendezvous stores the resulting handle and exact rows; it can still wait for EOS in the first implementation.

Admission reserves both transport bytes and a guaranteed evacuation path. Partition total memory into compute/pin allocation, an ingress working reserve carved from the ordinary pool, arena capacity, and runtime overhead; do not double-count a reserve. Reserve host/spill capacity separately. Account for unpack expansion, temporary double residency, and pending jobs. A receiver-provided maximum batch/format limit must prevent one unserviceable grant; [10 · Small-batch batching and oversized-batch policy](10-small-batch-policy.md) can later split larger logical batches.

**Implementation slices**

1. **Buffer-only ingress seam:** introduce a proposed `IngressTicket`/owned input handle with deterministic completion/error cleanup. Validate metadata, device, schema compatibility, and required materialization capacity before consuming it.
2. **Bounded admission:** issue grants only when transport plus evacuation reservations can progress. Return asynchronous unavailable/queued status without blocking the worker needed to release credits. Use per-peer and global byte limits, with a small control/progress reserve.
3. **Early evacuation:** run copies independently of `EngineRequest::Run`, store batches in path 03's registered ownership, and release leases on event completion. Preserve arrival/sequence order even if copies finish out of order.
4. **Receiver attachment:** replace staged input vectors with managed handles. With the existing runtime, dispatch after EOS and completion of all admitted ingress jobs, then attach handles without a second deep copy. Preserve exact cardinality before build.
5. **Pressure and cancellation:** spill/reload as needed, release unpublished and queued ownership through path 01, and keep in-progress buffers until GPU operations are quiescent.

**Tests and acceptance**

Use fake ingress completions to test out-of-order completion, EOS before final copy completion, duplicates, empty frames, late cancellation, and receiver build failure. Add GPU cases where the engine is occupied by an unrelated long fragment while ingress progresses. Transfer total input several times larger than the arena while keeping individual batches below the supported limit.

Acceptance: live arena bytes plateau below the configured bound, host growth is bounded, and credits return before receiver EOS. Under insufficient evacuation capacity, admission backpressures or fails with a concrete capacity error; it must not deadlock or silently allocate outside budgets. Verify a subsequent query after cancellation, and exact values with strings/null masks.

**Benchmark and rollout**

Sweep total input, batch size, compute pressure, and consumer delay independently. Measure arrival-to-credit-return, pending ingestion bytes, D2D/D2H traffic, spills, and query latency. Include a memory-fitting control to expose extra copying costs.

Enable per query after both sides negotiate support; retain the current staged-until-EOS path for controlled A/B only. Never switch an active batch's ownership mode during rollback. The first release keeps the EOS execution barrier; incremental consumer execution belongs to [12 · Nonblocking query-scoped fragment execution](12-nonblocking-fragment-graph.md) and owning receive views to [13 · Consume owning packed receive views](13-owning-packed-ingress.md).

**Decisions**

Choose the supported spillable ingress representation and conservative materialization bound before grants are issued. If arbitrary packed data cannot be validated without metadata, extend the admission protocol to carry sufficient metadata or reserve a worst-case supported bound; do not trust an unchecked sender estimate.
