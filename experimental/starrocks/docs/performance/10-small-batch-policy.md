**10 · Small-batch batching and oversized-batch policy**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: amortize per-frame control overhead for small hash slices and bound the memory needed by a single large batch. Prerequisites for production: [01 · Retry-safe leases and transport recovery](01-lease-lifecycle.md), [04 · Early ingress and bounded receive credits](04-early-ingress-and-credits.md), and [06 · Fair transfer pipeline and asynchronous control](06-fair-transfer-pipeline.md).

**Current behavior**

The sink emits every nonempty destination slice of each input batch. A frame incurs export, grant, WRITE, and publication work. The 8 MiB chunked-pack working span is not a network-frame cap: the whole packed table gets one lease plus slack. Small slices and oversized batches therefore require different policies.

**Code map**

| Source | Planned change |
|---|---|
| [sirius_physical_streaming_sink.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/op/sirius_physical_streaming_sink.cpp#L168) | Optional bounded per-destination aggregation with explicit flush. |
| [batch_stream.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/batch_stream.cpp#L41) | Preserve progress; current push has no byte backpressure. |
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L747) | Batch size estimation and valid subtable packing. |
| [nixl_transport.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/nixl_transport.rs#L673) | Group transfers/publications while retaining logical frame identity. |
| [nixl-exchange-proto.patch](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/patches/nixl-exchange-proto.patch) | Group envelope and per-subframe metadata if needed. |
| [local_exchange.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/local_exchange.rs) | Ordered logical subframe handling and EOS. |

**Policy options and initial choice**

Start with transport-level grouping of already-produced logical batches. If bindings support multiple source/destination descriptors, one group can amortize preparation/publication without concatenating GPU columns. It still owns each allocation, sequence, row count, and pack metadata separately. Grouped transport must not masquerade as one cuDF table.

Measure physical concatenation only as a second variant. It reduces logical pack calls but adds a GPU concatenate copy and a per-destination waiting buffer. Flush by bytes, age, EOS, or pressure; cap the aggregate of all destination buffers. A size/time threshold is an experiment parameter, not an unmeasured default.

For oversized data, split by valid row slices and repack each into a self-contained frame. Arbitrary byte chunks are not independently unpackable tables. Strings make row-count sizing approximate; use measured packed size and bounded subdivision. If one row exceeds the supported frame/evacuation bound, report a clear unsupported size or implement an explicit segmented format later.

**Implementation slices**

1. **Histograms and limits:** record payload/lease/slack distribution, frame count, control service time, and largest row/batch. Negotiate a supported maximum frame size and fail before granting an unserviceable allocation.
2. **Descriptor grouping:** group compatible ready frames for one peer within a byte limit and latency deadline. Keep ordered publication and idempotence per logical sequence. Verify actual descriptor-list support in the pinned bindings.
3. **Large-batch splitting:** add subtable slicing and packing, exact row-count conservation, sequence expansion, and cancellation cleanup. Keep one EOS per sender after all resulting frames.
4. **Optional sink coalescing:** prototype only if many tiny logical pack operations remain. With today's run-to-completion producer, a full coalescer must flush to its output rather than wait for a drain that starts after run; real bounded upstream backpressure requires path 12.

**Validation**

Test mixed frame sizes, timeout flush, EOS flush, one slow peer, pressure flush, empty senders, duplicate grouped publication, and failure partway through a group. Ensure retries cannot deliver half a group twice. Split fixed-width and string/null payloads at many row offsets and verify concatenated logical results match the original.

Acceptance: bounded queued/active bytes, conserved rows and destination mapping, no additional full-table copy in descriptor-grouping mode, and no producer/drain deadlock. Deadline batching must preserve the configured maximum added waiting time in deterministic tests.

**Benchmark and decision gate**

Use realistic payload histograms alongside controlled 1/4/16/64 MiB frames and genuinely small payloads below 1 MiB. Sweep fan-out, grouping byte limit, and flush deadline. Report frames/control calls per GiB, latency to first consumed data, copied bytes, throughput, and memory peaks.

Adopt grouping when control cost dominates and added wait does not damage latency. Adopt physical concatenation only when its reduced pack/control cost exceeds its extra copy and memory retention. Keep a direct-send path for large enough batches.

**Rollout**

Negotiate grouped-frame support and max size, enable per query, and drain pending groups before changing policy. Preserve the ungrouped wire path for old peers. Do not set a lower cap mid-transfer or reinterpret an old packed payload as a new segmented representation.
