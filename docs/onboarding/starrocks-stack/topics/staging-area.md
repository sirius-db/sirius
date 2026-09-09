# Exchange staging area

[Back to the guide](../README.md)

**Classification:** topic: distributed exchange memory; feature: stable GPU transfer buffer; modules: C++ engine, C++/Rust foreign-function interface (FFI), Rust compute node (CN), NIXL/UCX transport; languages: C++ and Rust.

**Scope and status snapshot:** this page describes [`7610840c`](https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60) on `bench/sf500-2-mig-gpus`, reviewed statically on 2026-09-09. The allocator began as draft [PR #1693](https://github.com/sirius-db/sirius/pull/1693). The branch also contains proposed [P02 FFI](../pr-packages/p02.md), [P05 registration](../pr-packages/p05.md), [P06 transfer](../pr-packages/p06.md), and [P16 receive/copy-out](../pr-packages/p16.md). None is landed behavior on `dev`.

## What it is

An **exchange** is the handoff of rows from one distributed query fragment to another. A **staging arena** is one fixed region of GPU memory used only while those rows cross a process boundary. The arena is a raw `cudaMalloc` allocation outside the normal RMM (RAPIDS Memory Manager) and the engine's cuCascade memory pools. That separation is deliberate: the code records that UCX (Unified Communication X)'s CUDA inter-process-communication route, `cuda_ipc`, cannot export stream-ordered pool allocations and can silently fall back to slow host copies. A transport can instead register the arena's one stable device address once and use subranges of it for every transfer. See the [allocator contract](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/include/exec/exchange_staging_arena.hpp#L26-L63).

The arena is not a cache, a parquet-scan allocation, or a replacement for the engine's ordinary GPU allocator. It is an in-flight wire buffer. After the receiver has copied a frame out, the frame should live in ordinary pool-backed memory under a ticket, and its arena region can be used again.

## The allocator and its ownership rule

A **lease** is a contiguous part of the arena. The allocator returns an offset from the fixed base address, rather than a new device allocation. Each lease is rounded to 256-byte alignment, selected by an address-ordered first-fit free list, and returned explicitly by that offset. Returning a lease coalesces adjacent free blocks, so capacity limits concurrently live bytes rather than lifetime traffic. The [allocation and release code](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/exec/exchange_staging_arena.cpp#L213-L294) also reports total free space and the largest block separately.

The important identity caveat is that the current allocator identifies a lease by **offset**, not by an unrepeatable generation token. An offset can be safely reused after release. That makes ordering important: a replayed network frame must never release or read an offset already returned by its original delivery. It also explains why the required future remote-abort protocol must use a lease identity stronger than a bare offset.

The default arena uses `cudaMalloc` and is the intended same-host route. An opt-in `fabric` mode uses CUDA virtual-memory APIs and a FABRIC handle for a different-host peer. It requires IMEX/MNNVL setup and has no integration coverage in this review, so it should not be treated as a validated multi-host deployment path.

## One frame, step by step

1. A sender's `Fragment::export_packed()` reserves a local lease and packs its GPU table into it. **Pack metadata** is the small host-side description that lets cuDF, the GPU dataframe library, reconstruct columns; the payload remains in the device lease. Export reserves the packed length plus an 8 MiB margin for a complete chunk, then returns offset, length, exact row count, and metadata after synchronizing. A zero-row batch is metadata-only and needs no lease. The documented interface is [here](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/docs/super-sirius/streaming-fragments.md#L310-L346).

2. The sender asks the receiving CN for a remote arena lease. The sender uses NIXL to perform and wait for a GPU-to-GPU WRITE into that address. The CN then sends metadata, sequence, column names, rows and the remote offset through a control RPC; a final control message announces end of stream (**EOS**). The [send loop](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L672-L755) returns the local lease after the attempt.

3. The receiver claims the sender's sequence number, the ordered frame counter for one exchange, *before* examining payload. A duplicate RPC leaves its offset untouched because the first delivery may have released it for reuse; a gap is a lost frame. The [claim implementation](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/local_exchange.rs#L397-L430) protects against stale-offset replay.

4. For a fresh nonempty frame, `InboundStore::stage()` unpacks from the remote arena and deep-copies into ordinary GPU-pool memory on its own synchronized CUDA stream. It records the copy under a **ticket**, a local handle for the copied batch, and returns that ticket. The receiving RPC handler then releases the remote arena lease. A ticket is not a lease: it owns pool memory until `push_inbound()` consumes it or cleanup drops it. The [copy-out path](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/compute_node_service.rs#L841-L893) does this.

5. `LocalExchange` retains the received tickets until its required inputs are ready. Every remote sender must report EOS; a local input supplies parked output or an accepted plan to combine with the receiver. This **EOS barrier** delays dispatch while any required remote input remains incomplete. The [readiness check](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/local_exchange.rs#L500-L565) is why the arena bounds in-flight frames while pool memory can retain completed inputs.

## Current review findings and limits

**P1 (high priority) — remote lease leak after a failed WRITE.** After a receiver grants a remote lease, a failed `write_and_wait` exits before `transmit_packed`. The sender cleans up its local lease, but the receiver has not recorded an announced frame and cannot discover its outstanding remote lease. Repeated failures can exhaust the receiver arena. Add an idempotent remote abort/release RPC keyed by lease identity, call it on every post-grant failure, and fault-inject a write timeout.

**P1 — `InboundStore` teardown race.** `stage()` copies a raw GPU-memory-space pointer while holding the store lock, drops the lock for unpack/copy, then dereferences it. `Context` teardown can call `close()`, clear the store, and invalidate engine-owned memory in that interval. The second null check happens too late. The [two paths](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L154-L187) and [stage implementation](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/src/sirius_ffi.cpp#L418-L470) need an active-stage lifetime guard or close waiting for active stages.

**P2 (important capacity limit) — not a proven memory-safety failure.** The receiver grants arena space only by arena availability, then allocates pool memory for each copied frame. Because the EOS barrier delays consumption, queued tickets can fill the GPU pool before execution. The current branch has no verified receive-credit mechanism, inbound spill mechanism, or copying-thread reservation binding. The receive-credit/spill/bounds ideas from commits `91c0370f`, `222b646a`, and `7412ef4d` are non-ancestor performance-tree candidates, not fixes present or validated here. Likewise, the DISK helper documented in `plan-2mig/DISK-TIER.md` is from another branch and is not part of this source snapshot.

## Related topics

- [Exchange transport](exchange-transport.md) explains registration, control RPCs, and NIXL WRITE.
- [Memory and MIG](memory-and-mig.md) explains the separate pool and arena budget and the recorded MIG evidence.
- [Back to the guide](../README.md)
