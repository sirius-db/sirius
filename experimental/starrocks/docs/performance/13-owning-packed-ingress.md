**13 · Consume owning packed receive views**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: avoid the ordinary GPU-table deep copy on ingress when consumers can safely read the registered receive allocation directly. Prerequisites: [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md), [03 · Spillable exchange repositories and reload](03-exchange-spill-and-reload.md), [04 · Early ingress and bounded receive credits](04-early-ingress-and-credits.md), and [06 · Fair transfer pipeline and asynchronous control](06-fair-transfer-pipeline.md). [12 · Nonblocking query-scoped fragment execution](12-nonblocking-fragment-graph.md) is strongly beneficial because an EOS-gated consumer otherwise retains receive leases.

**Current behavior and code map**

`unpack` creates a cuDF view over the receive lease; the subsequent table constructor copies it and stream synchronization makes the lease immediately reusable. Removing the copy requires an owner for the borrowed device pointers and a replacement reclamation policy.

| Source | Responsibility |
|---|---|
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L807) | Unpack validation and owning representation instead of unconditional copy. |
| [sirius_ffi.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/sirius_ffi.hpp) and [lib.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/rust/crates/sirius/src/lib.rs) | Lease-owning input API and completion semantics. |
| [batch_stream.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/batch_stream.cpp) | Input ownership through consumer handoff. |
| [owning_table_view.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/op/scan/owning_table_view.hpp) | Assess reuse of owner-backed views; no assumed interchangeability. |
| [downgrade_executor.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/downgrade/downgrade_executor.cpp) | Pressure conversion eligibility and reader safety. |
| [local_exchange.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/local_exchange.rs) | Preserve managed input ownership across rendezvous/cancellation. |

**Proposed representation**

A proposed `PackedInputOwner` holds lease token/epoch, pack metadata, schema, rows, device, readiness, and consumer references. It exposes immutable views only. Credit returns after all GPU readers finish or a completed migration makes the original allocation unnecessary.

Reference counting alone is insufficient: an operator may drop a host reference before its CUDA kernel completes. Reader registrations must retain ownership until associated completion events finish. An operator requiring mutable/native-owned columns materializes a private copy at that boundary.

Credit pressure triggers migration to path 04's ordinary/spillable representation. New readers switch to the migrated owner only after it is ready; old readers keep the original allocation alive until completion. A copy finishing does not free memory still read by an older kernel. Reserve enough capacity to complete this transition, and avoid duplicate pressure-triggered copies for the same owner.

**Implementation slices**

1. **Owner and readers:** add an immutable owning input type with explicit device-reader lifetime tests. Keep all consumers on a copy fallback until they support the type.
2. **Eligible operator path:** admit a small set of read-only consumers and validate sliced/string/null metadata. Prove end-to-end completion-based release.
3. **Pressure conversion:** add one migration job per owner, use path 03's accounting, and return credit only after old readers drain. Record reason, copied bytes, and time held.
4. **Policy:** choose direct view versus immediate copy from credit availability, expected consumer readiness, payload size, and retained age. Preserve immediate copy for EOS-gated or slow consumers unless a measured benefit justifies a bounded direct-view window.

**Tests**

Use delayed GPU reads to verify that host reference drops and query cancellation do not release a live lease. Test multiple consumers on different streams, pressure during reads, migration completion out of order, a mutable operator, spill/reload, peer restart, duplicate publication, and stale generation. Validate all values for strings/nulls and nonzero slice offsets.

Acceptance: no receiver copy for eligible fast-path batches; every returned credit corresponds to a quiescent original range; migration and active-reader bytes stay within budgets; a slow receiver cannot consume the entire arena indefinitely.

**Benchmark and decision gate**

Compare copy-on-ingress and owning views at equal compute/arena budgets for short pipelineable consumers, long-running consumers, and delayed dispatch. Report D2D bytes, copy time, lease hold-time distribution, blocked-credit time, peak retained bytes, and query latency.

The optimization is accepted only where reduced copying improves the critical path without increased transport stalls. If retaining leases makes throughput worse, use early copy-out. That is an intentional policy outcome, not a failure to achieve universal zero-copy.

**Rollout and open decisions**

Negotiate the representation per query/edge and restrict it initially to known compatible operators. Rollback affects new arrivals; active readers retain the correct owner to completion. Resolve how cuCascade batch representations expose immutable external allocations, how readers register completion, and whether current spill machinery can migrate that representation. Verify pinned APIs before committing to a particular wrapper.
