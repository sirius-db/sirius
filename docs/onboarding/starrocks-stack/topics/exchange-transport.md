# Exchange transport

[Back to the guide](../README.md)

**Classification:** topic: distributed exchange transport; feature: GPU-to-GPU remote batch delivery; modules: Rust CN service, Rust NIXL transport, Rust PRPC client, C++ staging FFI; languages: Rust and C++.

**Scope and status snapshot:** this page covers [`7610840c`](https://github.com/aocsa/sirius/tree/7610840c03f9086edfa072be72a0eb4c96e03d60) on `bench/sf500-2-mig-gpus`, from a 2026-09-09 static review. It describes draft [PR #1693](https://github.com/sirius-db/sirius/pull/1693) as a prerequisite plus proposed [P04 PRPC](../pr-packages/p04.md), [P05 registration](../pr-packages/p05.md), [P06 transfer](../pr-packages/p06.md), [P08 cancellation](../pr-packages/p08.md), and [P16 receive lifecycle](../pr-packages/p16.md). None is a claim that the full path has landed on `dev`.

## Purpose and pieces

A **compute node (CN)** runs one part of a distributed query. An **exchange** sends rows from a sender CN to a receiver CN. This implementation uses two channels. The control channel is PRPC, a framed peer remote-procedure-call protocol used to request leases and deliver small metadata messages. The data channel is NIXL, the GPU-memory transport library, backed here by UCX (Unified Communication X). It copies a payload directly between the sender's and receiver's registered staging arenas.

At start-up, the NIXL agent checks for one visible CUDA device, registers the entire arena as VRAM (GPU memory), and retains that registration. It uses process-local device ordinal zero, so the launcher must pin a CN to one device first. The [registration code](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L449-L522) is the boundary between engine staging memory and transport.

## Delivery walkthrough

The sender drains output already parked by the local exchange. For each nonempty packed frame it:

1. asks the receiver's PRPC service for a staging lease;
2. executes and waits for a NIXL WRITE from the sender offset to the returned remote address;
3. sends `transmit_packed` over PRPC with metadata, column names, sequence number, payload length, remote offset, exact row count, and EOS state; and
4. releases the sender-side lease after the transfer attempt.

A zero-row frame skips the remote lease and WRITE: its metadata and row information are sent with a zero length. The concrete [send loop](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/nixl_transport.rs#L672-L755) shows this division. A **canary** is a small first-contact transport check used to establish and validate peer sessions; it is useful setup evidence, but it is not end-to-end query validation.

The receiving service reconstructs the exchange key, sender id, and sequence. It claims the sequence before it stages or releases the remote offset. This makes a reconnect replay harmless: a duplicate frame returns success but neither reads from nor returns its old offset, which may have been reused. A fresh frame is copied into an `InboundStore` ticket, then the receiver release returns the remote lease. The control service's [claim and copy-out sequence](https://github.com/aocsa/sirius/blob/7610840c03f9086edfa072be72a0eb4c96e03d60/experimental/starrocks/src/compute_node_service.rs#L797-L915) is the key correctness boundary.

The receiver-side **rendezvous** records frames by exchange and sender. A receiver starts only after every expected source is ready. Remote sources
become ready on EOS; local parked outputs and accepted deferred plans are ready
by construction. The wait for remote completion is the EOS barrier. Cancellation retires recorded state so late arrivals cannot re-create it, and rejected or retired frames release their ticket or lease.

## What needs review before landing

The most serious transport defect is after remote lease grant. If the NIXL WRITE fails before `transmit_packed`, the receiver knows it granted memory but has no announced frame to retire. The current sender cleanup returns only its local lease. The result is a P1 (high-priority) remote lease leak that can permanently consume peer capacity. The transport package should remain draft until it has an idempotent remote-abort RPC keyed by a lease identity, all post-grant failure paths call it, and a fault-injection test proves complete reuse after a timeout.

The snapshot has no verified receive credits, bounded-frame protocol, fair transfer window, or retry/replay redesign. Those are non-ancestor performance-tree candidates (`91c0370f`, `222b646a`, `7412ef4d`), so they must be rebased, reviewed, and tested as new work rather than described as current transport behavior. The default `cudaMalloc` arena is the same-host path. Fabric/VMM is opt-in for different hosts and lacks the two-host integration evidence needed to promise it.

## Related topics

- [Exchange staging area](staging-area.md) covers lease ownership and copy-out storage.
- [Memory and MIG](memory-and-mig.md) covers process/device placement and capacity.
- [Back to the guide](../README.md)
