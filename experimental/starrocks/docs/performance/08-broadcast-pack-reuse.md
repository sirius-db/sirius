**08 · Pack broadcast output once**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: avoid per-destination native clones and repeated packing for remote broadcast fan-out. Production prerequisites: [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md), [03 · Spillable exchange repositories and reload](03-exchange-spill-and-reload.md), [06 · Fair transfer pipeline and asynchronous control](06-fair-transfer-pipeline.md), and [07 · Independent packing and CUDA completion](07-independent-gpu-packing.md).

**Current behavior and evidence**

The streaming sink sends the original handle to destination zero and deep-clones it for other destinations. Remote destinations independently pack their stream. Clones protect mutable batch residency; simply sharing the existing handle would remove that protection.

| Source | Planned responsibility |
|---|---|
| [sirius_physical_streaming_sink.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/sirius_physical_streaming_sink.cpp#L137) | Remote broadcast representation instead of unconditional per-destination clones. |
| [sirius_physical_streaming_sink.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/op/sirius_physical_streaming_sink.hpp) | Proposed immutable fan-out descriptor. |
| [compute_node_service.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/compute_node_service.rs#L1300) | Preserve FE destination ordering and classify local/remote consumers. |
| [parked_registry.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/parked_registry.rs) | Per-destination claims over shared payload ownership. |
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L747) and [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L673) | One pack job and multiple source readers/transfers. |

**Proposed ownership**

A proposed `BroadcastPayload` contains immutable pack metadata, exact rows/schema, one packed allocation, readiness, and a set of destination references. Each transfer holds its own reference until it no longer reads that source. Publication retry state may retain metadata after payload reuse is safe. Cancellation removes only that destination's ownership unless the whole query is retired.

Local consumers should retain native handoff where safe. Do not force pack/unpack on all-local broadcast just to unify the interface. For mixed fan-out, choose an explicit native owner plus one packed remote owner; the existing sink/FFI contract may need destination classification before production.

One slow peer can hold the common packed allocation for a long time. Add age/byte limits and a pressure policy: copy or spill a slow peer's still-needed data into independent managed ownership once no transfer is reading the region being moved, or stop admitting more broadcasts. “Shared once” is not a license for unbounded retention.

**Implementation slices**

1. **Immutable payload seam:** introduce a destination-reference object and lifecycle tests without changing default sink behavior. Include logical identity independent of the parked output stream ID.
2. **Remote-only prototype:** pack one batch once and fan it out to multiple peers through path 06. Deduplicate pack work for simultaneous destination requests.
3. **Mixed local/remote routing:** preserve native local output, eliminate only remote clone/pack duplication, and keep FE slot/sequence mapping stable.
4. **Pressure and cancellation:** integrate spill/reload and slow-peer retention thresholds. Remove completed destination references incrementally; no global “all destinations done” wait is required to release unrelated batches.

**Validation**

Test 1/2/4/8 destinations, mixed local/remote placement, a delayed peer, one failed peer, query cancellation during several simultaneous transfers, duplicate publications, and an empty broadcast. Validate strings/nulls and metadata reuse. Explicitly check that a late consumer sees stable values after an earlier consumer spills or retires.

Acceptance: one packing operation per logical remote broadcast batch in the measured optimized path, correct values at every destination, bounded retained source bytes, and no additional serialization on all-local edges. Destination failure policy must match the query's existing failure semantics.

**Benchmark**

Hold logical broadcast bytes fixed while varying fan-out and peer skew. Measure clone D2D bytes, pack bytes/time, peak native plus packed residency, slowest-peer retention, transfer concurrency, and end-to-end join time. For remote fan-out D, the target mechanism is reducing repeated pack work toward one pack; no linear D-fold speedup is promised because transfer, build, and shared memory bandwidth remain.

Use a supported broadcast join with a range of build-table sizes. If broadcast data is tiny and cloning/packing is negligible, defer the ownership complexity.

**Rollout and alternatives**

Opt in only for negotiated remote broadcast edges and preserve the original path for other distributions. Revert for new batches after draining active references. A simpler intermediate experiment can reuse packed data while retaining current native clones; it isolates packing savings before changing sink ownership. Hierarchical broadcast is a separate topology experiment in path 17, not a prerequisite.
