**17 · Topology-aware transfer experiments**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: exploratory plan, not a commitment to replace NIXL. Baseline: `281b13bc`. Objective: investigate remaining fabric/data-movement cost after wrapper serialization and memory retention are controlled. Prerequisites: [00 · Trustworthy measurements and benchmark coverage](00-measurement-and-benchmarks.md), [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md), [02 · Nonblocking peer establishment](02-peer-establishment.md), and [06 · Fair transfer pipeline and asynchronous control](06-fair-transfer-pipeline.md). Broadcast experiments also use [08 · Pack broadcast output once](08-broadcast-pack-reuse.md).

**Scope and current evidence**

The branch has native same-process relay, one visible GPU per CN in the transport path, cached NIXL peer setup, and arena allocation modes for ordinary CUDA allocation and fabric handles. Preserve those mechanisms. Header comments contain historical bandwidth observations, but this plan does not adopt them as current measurements.

| Source | Responsibility |
|---|---|
| [gpu_affinity.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/gpu_affinity.rs) and [cluster8.sh](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/benchmarks/cluster8.sh) | Actual process/GPU placement and test topology. |
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs) | Peer capabilities, transport choice seam, timing and validation. |
| [cn-env.sh](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/scripts/cn-env.sh) | Installed NIXL/UCX configuration to record. |
| [exchange_staging_arena.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/exchange_staging_arena.cpp) and [exchange_staging_arena.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/exec/exchange_staging_arena.hpp) | Allocation/registration contract and fabric mode. |
| [engine.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/engine.rs#L518) | Preserve existing local native relay. |

**Candidate experiments**

| Topology | Candidate | Required proof before implementation |
|---|---|---|
| Same process, same GPU | Continue native handle relay; optimize scheduling/fusion. | Serialization is already avoided at this boundary; an extra copy is not an improvement. |
| Separate CNs, same GPU or same host | Tune and validate existing registered NIXL path first. | Actual selected transport, byte correctness, resource contention, and process/device identity. |
| Cross-host supported fabric | Validate current fabric allocation/registration and placement. | Installed driver/runtime/backend support, accessible fabric capability, and measured route; no inferred path from topology names alone. |
| Broadcast across hosts | Hierarchical fan-out after pack reuse. | Fewer expensive cross-host copies must exceed relay, replication, retention, and failure costs. |
| Future multi-GPU CN | Direct peer-copy for eligible intra-process edges. | Explicit device ownership and peer accessibility; current CN transport's single-visible-device contract must be redesigned first. |

**Experiment sequence**

1. **Topology inventory:** record GPU identities, links, process mapping, device visibility, NIC affinity, NUMA placement, NIXL/UCX versions, allocation kind, and selected route where tooling exposes it. Distinguish same-device IPC, cross-GPU traffic, and cross-host traffic.
2. **Validated link baseline:** use path 00's nonuniform payload test with registration/setup outside steady-state timing and also report setup costs separately. Measure both directions and simultaneous traffic.
3. **Production-edge baseline:** repeat with real pack, grants, publication, ingress, and representative concurrent queries. Attribute GPU memory-bandwidth contention before labeling a link slow.
4. **One candidate at a time:** prototype route selection or hierarchical fan-out behind explicit capabilities. Keep the same lease, epoch, cancellation, sequencing, and memory-budget contracts.
5. **Query confirmation:** benchmark a workload whose measured critical path is actually transfer-limited. Preserve FE distribution and row multiplicity when changing physical delivery routes.

**Correctness and failure handling**

Validate bytes, null/string table values, asymmetric peer availability, peer restart, relay failure, cancellation, and memory pressure. A hierarchical relay must hold data for downstream completion and cannot emit logical EOS before all required delivery is established. Prevent duplicate rows if a route is retried or changed.

Never switch transport/allocation type for an active registered range. A fallback after ambiguous completion can duplicate delivery or expose stale memory; route fallback must use the same logical frame identity and proven transfer quiescence. Reject unsupported topology rather than silently assuming a fast path.

**Metrics and acceptance**

Report effective payload bandwidth, registration/setup latency, control/pack/copy time, GPU/NIC utilization, CPU cost, retained bytes, cross-host bytes, and complete query latency. Repeat small and large payloads, 2/4/8 CNs where hardware permits, and one slow participant.

Adopt an alternative only when production-edge and query measurements improve beyond run-to-run variability with identical results and bounded memory. A raw-copy win alone is insufficient. If source packing, compute, or EOS retention dominates, stop this path and invest in the corresponding existing plans.

**Rollout and unresolved questions**

Start as an explicit experiment with no automatic route selection. Verify installed primary API documentation for the exact driver, NIXL, UCX, and hardware before designing calls. Introduce capability negotiation before enabling a topology-specific path by default. Keep NIXL as the baseline and retain allocation-path validation.

This path has the lowest initial priority: transport replacement cannot remove engine queue waits, fragment barriers, or redundant table copies by itself. Direct multi-GPU-CN work also requires a separate scope decision because this checkout currently enforces one visible GPU for its CN transport.
