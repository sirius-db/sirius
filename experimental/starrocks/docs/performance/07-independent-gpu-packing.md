**07 · Independent packing and CUDA completion**

[All paths](README.md) · [Source review](../../../../starrocks-plan-improvement.md)

Status: proposed implementation plan. Baseline: `281b13bc`. Objective: let already-produced exchange output be packed/transmitted while the engine runs another fragment, and replace unnecessary host waits with explicit GPU completion. Prerequisites: [03 · Spillable exchange repositories and reload](03-exchange-spill-and-reload.md) and the ticket lifecycle in [05 · Overlap local dispatch with remote drains](05-dispatch-drain-overlap.md). It integrates with [06 · Fair transfer pipeline and asynchronous control](06-fair-transfer-pipeline.md).

**Current behavior**

`ExportNext` uses the same engine queue as synchronous `Run`. The FFI export pulls a batch, holds a read-only residency guard, waits on its writer event, packs into staging, and synchronizes before returning. Moving that same call to another Rust thread would violate the current fragment/context ownership model.

**Code map**

| Source | Planned change |
|---|---|
| [engine.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/engine.rs#L234) | Separate batch acquisition from device pack progress. |
| [parked_registry.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/experimental/starrocks/src/parked_registry.rs) | Publish independently owned export providers when output is parked. |
| [sirius_ffi.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/sirius_ffi.cpp#L704) and [sirius_ffi.hpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/include/sirius_ffi.hpp) | Buffer-only packing job API, leaving fragment planning on its owner. |
| [lib.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/rust/crates/sirius/src/lib.rs) and [lib.rs](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/rust/crates/sirius-sys/src/lib.rs) | Explicit opaque ownership/completion wrappers. |
| [streaming_fragment.cpp](https://github.com/aocsa/sirius/blob/281b13bcb12321bac2927a8f4f996b710a463ec1/src/exec/streaming_fragment.cpp) | Export-provider lifetime after producer execution. |

**Proposed interface and ordering**

At producer completion, create a thread-safe export provider for already-produced batches. It owns repository/destination claims and can acquire a batch without borrowing a `Fragment` or consulting the blocked engine queue. Otherwise a “pack worker” would still starve waiting for the engine to hand it work.

A proposed `PackJob` owns batch/read reservation, metadata, allocator lifetime, source staging lease, and a producer-ready dependency. A dedicated C++ worker/device context uses a bounded number of CUDA streams. It waits on the writer event, optionally reloads via path 03, packs, and records a pack-ready event. Metadata construction and any cuDF-internal synchronization must be measured; an asynchronous wrapper does not guarantee every packer operation is nonblocking.

The transport may post WRITE only once the ready event is satisfied, using a supported interop mechanism or explicit event polling by a completion worker. Pack completion releases the original table's read reservation; transfer completion releases the packed source allocation when publication bookkeeping no longer needs it. Preserve these separate lifetimes on errors.

**Implementation slices**

1. **Buffer-only extraction:** refactor pack logic into an owned C++ job independent of connection/fragment state. Keep the synchronous wrapper and identical output for reference.
2. **Provider lifetime:** create export providers at parking time, maintain one destructive claim per destination stream, and make query retirement stop new claims while active jobs finish safely.
3. **Asynchronous GPU work:** add dedicated streams, completion events, bounded staging reservations, and error propagation. Replace host synchronization only where a readiness event takes over the exact ordering obligation.
4. **Pipeline integration:** connect job tickets to transport progress and early ingress. Permit completed producers' pack jobs to run during another fragment; future live-producer output uses path 12.

**Tests**

Extend FFI/stream-lineage tests with a producer writing a recognizable delayed pattern on a different stream. Verify packing never observes incomplete writes. Run pack versus spill/reload, cancellation versus pack completion, allocation failure, device error, and provider destruction with jobs active. Validate metadata and every payload value, including strings, nulls, empty data, and chunk boundaries.

Acceptance: a long unrelated engine `Run` does not stop packing from an already-published provider; no unsafe Rust `Send/Sync` assertion is used to bypass C++ ownership; source and staging memory remain valid through their actual readers. Use race tooling supported by the environment in addition to value checks.

**Benchmark and rollout**

Measure engine enqueue wait, batch acquisition, reload, pack CPU/GPU time, event wait, and NIXL readiness gap separately. Compare default-stream packing against one/two dedicated streams while compute runs; more overlap may reduce kernel throughput by competing for memory bandwidth. Accept only a query-level benefit or a demonstrated removal of a critical stall, not a prettier overlap trace.

Enable per query with a bounded worker/stream budget. Keep synchronous packing as an A/B path, but switching occurs only before jobs are created. Resolve worker-device affinity, memory-resource thread safety, and pinned cuDF chunked-pack guarantees using module-context before implementation.
