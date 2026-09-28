# Unified Simpatico decoding

Every decode entry point (single column, table, predicate, mask, selected rows, scan filter) submits typed requests to one `decode_session` (`src/decode/decode_session.hpp`), which runs one plan interpreter per request and returns only completed results.

## Decomposition

`decode_session` owns submission and completion. `append` accepts a `column_decode_request` (a borrowed plan or standalone representation; a value or BOOL8 predicate result, optionally also writing a ballot mask; an optional `validated_selection`) or a `mask_decode_request` (a range predicate or membership probe writing a borrowed mask; its `accepted`/`declined` result describes semantic applicability, not completion). `finish` drains every supplied stream and transfers completed columns in request order; the `noexcept` destructor drains abandoned work.

`decode_frame` is one request's context: its stream (assigned in rotation), the memory resource, and the host state its queued work may still read (`host_array` uploads and compiled-kernel pins). It owns no device memory. The session keeps each frame and a copy of its request until the final drain, so predicate strings, descriptors and probe captures outlive pending work.

`DecodeWalk` interprets the plan, owns the structural memo, and targets predicate and selection semantics at the final value producer. Codec leaves take the frame and return an owning column; they do not know whether the caller decodes one column or a table.

## Completed public boundaries

Public decode calls return only after all output and mask writes complete. Callers establish input readiness and keep borrowed plans, representations, masks, row sets and indices alive through completion. The supplied resource must outlive returned allocations.

Supplied streams are borrowed `::cuda::stream_ref` handles, possibly repeated and possibly shared with other callers (for example a memory space's pool). Every wait on a supplied stream also covers whatever others queued there, so work on those streams must never be host-gated. Submission and destruction stay on the constructing CPU thread and current device, which preserves the engine's per-thread reservation attachment.

## Stream-ordered temporaries

Device temporaries (scratch, intermediate columns, memo values, representations rebuilt from decoded channels) are RAII owners released when the scope that enqueued their work ends, usually while that work is still queued. This is safe because deallocation through a `device_async_resource_ref` is stream-ordered. Two rules follow:

1. A temporary is read and written only on its allocation stream. Data that crosses streams stays phase-owned with explicit events and synchronization (the scan-filter phases).
2. No pointer to a temporary outlives its owner. Launchers bind decode-only scratch through a local overlay. A fused region binds only memo values keyed by codegen nodes, which stay in the memo until the walk ends; only a non-codegen node consumes memo values, moving them into the representation it rebuilds. `DecodeWalk::bind` enforces this and binds the exact value a slot consumes, because a bitjoin consumes several.

On failure, temporaries are released in stream order as the exception unwinds; the session then becomes terminal, drains every supplied stream, and rethrows the original exception. The drain protects what is not stream-ordered: host uploads, request copies, borrowed caller memory and outputs.

Host memory is different: an asynchronous copy from pageable memory may read its source after the call returns. Every upload decode issues reads from `host_array` storage (uninitialized; initialize every byte before use) or the retained request copy, both alive until the final drain. Synchronous readbacks (`read_bytes`, `read_scalar`) may target local storage; they stage through the calling thread's pinned slab (`read_device_bytes_completed`) and complete with an explicit stream wait.

cuDF scalars add a host source decode does not own: each uploads its validity flag from a pinned bounce buffer that returns to cuDF's pinned resource when the scalar is destroyed. Reuse is safe only because the pinned resource orders the next allocation after the release (cuCascade's small pinned resource in the engine, cuDF's stream-ordered pool standalone) and cuDF synchronizes the stream after allocating the buffer (`rmm_host_allocator::allocate`, marked for removal upstream). Decode therefore builds scalars only in the generic predicate comparison on a non-dictionary producer; the dictionary predicate, constant-width offsets and all-null dictionary columns are built on the device.

Compiled-kernel handles are pinned by their frame until the final drain, because unloading a module with a queued launch is unsafe. The resource must honor the stream-ordered deallocation contract; a synchronous resource stays correct but synchronizes the device on every release.

### Accounting

A call-time ledger such as cuCascade's `reservation_aware_resource_adaptor` credits a release when `deallocate` is called, as for every other engine operator. Decode's charge therefore peaks at earlier outputs plus the active request's working set. The asynchronous pool reuses a released block immediately on the same stream; a release pending on another lane becomes reusable when that lane reaches it, so the physical peak can exceed the ledger by roughly the other lanes' largest working sets.

## Errors and semantic decline

Semantic declines (an unsupported shape, an unselective batch, every membership probe declining) fall back to ordinary decoding. Submitted execution failures, including typed allocation (cuCascade OOM subtypes) and compilation failures, propagate unchanged after draining; they are never retried as ordinary decoding. Public compatibility functions keep their null/false plus `error_out` contract for host validation and explicit decline only.

`finish` checks every distinct supplied stream, including phase work queued after the last request; it reports its first failure. Append and finish failures make the session terminal and publish no results. Borrowed mask contents after a failed request are unspecified.

## Genuine host observations

Decode waits only where the host needs data to size or publish something:

| Path | Observation |
|---|---|
| nvCOMP codecs | Header and compressed-size readbacks for dimensions and scratch sizing. |
| ALP and ALP_RD | Shared ALP initialization before publication; reconstructed ALP_RD bit width. |
| Dictionary | Key width only for reconstructions without a published hint; variable-width output sizing. |
| Selected strings | Total character count before allocating the exact output. |
| Scan selection | Survivor count before exact-sized indices and compacted outputs (staged through pinned memory). |
| Predicate needles | cuDF synchronizes while allocating each scalar's pinned buffer; generic non-dictionary predicate only. |

### Dictionary width publication

The dictionary factory computes an immutable uniform key width (positive, or 0 for variable/empty keys) before publishing, with one CUB transform-reduction written directly into cuDF's pinned resource and read after the existing publication synchronization. A published `PlanTree` also carries it on the dictionary node as `PlanNode::dictionary_key_width_hint` (never serialized): the compress walk copies the prepared width and the reader derives it from the stored identity `keys_offsets` or self-stored representation. Decode publishes the hint on the representation it rebuilds and the gather specialization uses it instead of a readback; reconstructions without a hint measure it into local storage with one checked scalar observation.
