**06 · Fair transfer pipeline and asynchronous control**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: overlap service stages and keep healthy peers progressing when another peer is slow. Production prerequisites: [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md), [02 · Nonblocking peer establishment](02-peer-establishment.md), and [04 · Early ingress and bounded receive credits](04-early-ingress-and-credits.md). Independent export from [07 · Independent packing and CUDA completion](07-independent-gpu-packing.md) removes a further bottleneck but is not required for a transport-only prototype.

**Current behavior and code map**

The transport owner drains one destination completely. Each batch serially requests export, requests remote staging, posts/waits for WRITE, publishes metadata, and releases source staging. Each peer client uses one cached socket with one outstanding request.

| Source | Responsibility |
|---|---|
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L673) | Replace whole-drain loop with per-frame progress states. |
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L772) | Split post, poll, completion, and cleanup while retaining request lifetime. |
| [prpc_client.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/prpc_client.rs) and [prpc.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/prpc.rs) | Asynchronous control replies, correlation, reconnect behavior. |
| [fragment_executor.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/fragment_executor.rs) | Nonblocking export-ticket boundary. |
| [tunable.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/tunable.rs) | Validated byte windows, peer limits, and progress settings. |
| [nixl-exchange-proto.patch](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/patches/nixl-exchange-proto.patch) | Credit grants/returns or grouped publication when introduced. |

**Proposed scheduler**

Keep the NIXL agent on one owner thread. Track each frame through export-pending, source-ready, credit-pending, transfer-active, publication-pending, and complete. A pass advances ready work and polls active operations, then waits on completions/work with a bounded polling strategy. No pass waits synchronously for an engine or peer reply.

Use byte-weighted round-robin or deficit scheduling across peers, plus per-query admission limits so one large query cannot occupy every slot. Cap queued, packing, source-staged, remote-granted, and transfer-active bytes separately. An available transfer-handle slot is not permission to exceed any memory budget. Keep EOS behind completion/publication of all prior sequences, even if transfers finish out of order.

A bounded control-worker pool is a viable initial client change, provided one unresponsive peer cannot consume all workers. A later async multiplexed client must validate correlation IDs and method-specific retries. Measure first; do not require a new networking framework for its own sake.

**Implementation slices**

1. **State-machine extraction:** separate immutable frame identity and owned resources from blocking functions. Use fake engine/control/agent completions for deterministic scheduling tests.
2. **Asynchronous control/export:** return tickets for lease and publish requests; continue servicing other peers while they wait. Queue source packing only when its memory reservations are viable.
3. **Multiple WRITE handles:** post a bounded set, retain descriptors/registrations until guaranteed completion, and publish sequences in order. Integrate timeout/quarantine semantics from path 01.
4. **Credit/control amortization:** test reusable pregranted slots and grouped publications/credit returns. Keep logical batch metadata and idempotence distinct. Reuse transfer preparation only if the pinned Rust/NIXL APIs support safe descriptor reuse.

**Tests and acceptance**

Construct three peers: one stalls grants, one stalls publication, one progresses. The healthy peer must continue within the scheduler's declared scheduling bound. Test reversed transfer completions, lost/replayed publication, EOS with pending frames, exhausted global/per-peer byte budgets, cancellation, and peer epoch changes.

Acceptance: no event-loop blocking dependency, finite outstanding bytes, per-sender order, no use-after-release, and useful progress for a healthy peer. The acceptance bound is a deterministic scheduling property in fake tests; real latency also includes GPU/NIC contention and must be measured.

**Benchmark matrix**

Sweep windows 1/2/4/8 and payloads 1/4/16/64 MiB on fixed topology, then repeat with production size histograms and a slow peer. Record throughput, link idle gaps, control RTTs, pack queue waits, CPU polling cost, GPU copy contention, and memory peaks. Fit outstanding bytes to observed service latency and memory budget, not to a fixed handle count.

Proceed when the useful workload shows improved query/edge throughput without worse recovery or unbounded retention. If GPU packing saturates memory bandwidth, stop increasing the transfer window and evaluate copy paths instead.

**Rollout**

Enable for negotiated new sessions/queries, start with window one, then increase within measured byte budgets. Drain active operations before lowering a window below current occupancy; reject new admission until it converges. Rollback never destroys active handles to reclaim space early. This path preserves NIXL as the data transport; alternatives belong to path 17.
