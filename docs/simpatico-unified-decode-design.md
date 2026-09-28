# Unified Simpatico decoding

Decode requests share one interpreter. Their enclosing session owns pending work and returns only completed results. Single-column, table, predicate, and selected-column entry points use this same mechanism; there is no synchronous/asynchronous decoder switch.

This document describes the current ownership and execution contracts. Benchmark histories and implementation experiments are not part of the API, and the design does not imply universal speedups or encoding parity with the earlier split implementation.

## Where to read the code

| Responsibility | Code |
|---|---|
| Completed public APIs and scan-filter phases | [`simpatico_codegen.cpp`](../src/compression/simpatico_codegen/src/simpatico_codegen.cpp) |
| Request types, the per-request decode frame, and the private session interface | [`decode_session.hpp`](../src/compression/simpatico_codegen/src/decode/decode_session.hpp) |
| Submission, host-state retention, and completion | [`decode_session.cpp`](../src/compression/simpatico_codegen/src/decode/decode_session.cpp) |
| Plan validation, `DecodeWalk`, fused-buffer binding, and selection routes | [`decompress.cpp`](../src/compression/simpatico_codegen/src/plan/decompress.cpp) |
| Representation reconstruction and standalone compatibility | [`representation_factory.cpp`](../src/compression/simpatico_codegen/src/plan/representation_factory.cpp) |
| JIT launch preparation and kernel retention | [`codegen_runtime.cpp`](../src/compression/simpatico_codegen/src/bridge/codegen_runtime.cpp) |
| Dictionary metadata preparation and decode | [`dictionary_compressor.cu`](../src/compression/simpatico_codegen/src/operators/dictionary_compressor.cu) |

## Decomposition

`decode_session` owns submission and completion. Its private `append` overloads accept typed requests; `finish` drains work and transfers completed columns in request order. Its destructor drains abandoned work without throwing. Callers never extract pending columns or mark frames complete.

`column_decode_request` describes a borrowed plan or standalone representation, a value or predicate result, and an optional `validated_selection`. Value requests can restore the stored logical type after decoding. Predicate requests own their predicate metadata and produce BOOL8, optionally also writing a ballot mask. A predicate result cannot simultaneously request value-type restoration. Standalone requests support unselected values.

`mask_decode_request` describes a range predicate or membership probe and a borrowed mask destination. It does not occupy a returned-column slot. Its `accepted` or `declined` result describes semantic applicability, not device completion. A declined membership probe submits an all-ones mask; the scan orchestrator treats an all-declined source set as a policy decline before counting it.

`decode_frame` is one request's context: its stream (assigned in rotation), the session's memory resource, and the host state its queued work may still read, namely `host_array` upload storage and pinned compiled kernels. It owns no device memory. The session keeps each frame, together with a copy of its request, in a stable list node until the final drain, so predicate strings, descriptors, and probe captures survive pending use. Compressed inputs and mask/selection device storage remain explicitly borrowed.

`DecodeWalk` interprets the plan and owns the request's structural memo. It targets predicate and selection semantics at the final value producer, not intermediate metadata. Codec leaves are producers: given a representation and the frame, they return an owning column. Leaves and JIT launchers receive the same mandatory frame; they do not know whether the caller is decoding a single column or a table.

## Completed public boundaries

Public decode calls return only after all their output and mask writes complete. The engine can then release compressed inputs or rebind output allocation streams according to its existing converter contract. Changing an RMM buffer's deallocation stream is not itself a CUDA ordering edge.

The supplied streams may be borrowed views, possibly repeated and possibly shared with other callers, such as streams from a memory space's pool. Every wait on a supplied stream also waits for whatever others queued there, so work on the supplied streams, including other callers', must never be host-gated.

Callers establish input readiness and retain borrowed plans, representations, masks, row sets, and indices through session completion or destruction. Stream views and memory-resource references do not own those resources. The supplied resource must outlive returned allocations, and their stored streams must remain valid until rebinding or destruction.

Submission and destruction stay on the constructing CPU thread and current device. This preserves the engine's per-thread reservation attachment: every device temporary is allocated and released inside the `append` call that needs it, and outputs are released by the caller or by the session destructor. Explicit decode allocations use the supplied memory resource; opaque library temporaries use cuDF's current device resource.

## Stream-ordered temporaries

Device temporaries (scratch buffers, intermediate columns, memo values, representations rebuilt from decoded channels, and predicate-needle scalars) are ordinary RAII owners scoped to the code that enqueues work on them. Each is allocated on the request's stream and released when its scope ends, usually while the work that reads it is still queued. This is safe because deallocation through a `device_async_resource_ref` is stream-ordered: the allocator reuses the memory only after earlier work on that stream. Two rules follow:

1. A temporary is read and written only by work on its allocation stream. Data that crosses streams stays phase-owned, with explicit events and synchronization (see the scan-filter flow below).
2. No pointer to a temporary is used after its owner is released. Launchers bind decode-only scratch through a local overlay, never through the caller's buffer map. A fused region binds only memo values keyed by codegen nodes, which stay in the memo until the walk ends. Only a non-codegen node consumes memo values (its decoded channels and stored terminal channels), moving them into the representation it rebuilds, which is released once the decode from it has been queued. `DecodeWalk::bind` refuses to bind a value keyed by any other node, and it binds the value the slot consumes rather than the consumer's first input, because a bitjoin consumes several.

For each append:

1. Reserve any returned-column slot and copy the request into a new frame on the next stream in rotation.
2. Run the request through the walker and its leaves. Transfer shared memo values only to their last consumer, copying for earlier consumers. A consumed memo entry remains identifiable so accidental reuse fails explicitly.
3. Move the terminal output into private session result storage. When `append` returns, only the output and the frame's host state remain.
4. On failure, leaf temporaries are released in stream order as the exception unwinds. The session then becomes terminal, drains all supplied physical streams, and rethrows the original exception. The drain protects what is not released in stream order: host upload storage, request copies, borrowed caller memory, and outputs.

Host memory is different: an asynchronous copy from pageable memory may read its source after the call returns. Every upload that decode code issues reads from storage that lives until the final drain: `host_array` storage (for example, the nvCOMP chunk tables) or the retained request copy (for example, predicate strings, which cuDF uploads into needle scalars). `host_array<T>` returns uninitialized, fundamentally aligned storage for trivially copyable values. Callers must initialize every consumed or uploaded byte before its first read or copy. Destinations of synchronous readbacks (`read_bytes`, `read_scalar`) may be local, because those calls wait for the stream before returning and drain it before rethrowing. Every such readback stages through the calling thread's pinned slab (`read_device_bytes_completed`, up to a documented cap) and completes with an explicit stream wait, so no copy into pageable memory waits inside the driver, where it would serialize other threads' CUDA calls behind it.

cuDF scalars add an upload source that decode does not own. Each scalar uploads its validity flag from a pinned bounce buffer inside the scalar, and a needle scalar is destroyed right after the comparison that reads it is queued, so that buffer can return to cuDF's pinned resource while the copy is still pending. This session does not protect that buffer; two current cuDF and cuCascade behaviors together keep its next user from overwriting it before the copy reads it. First, the pinned resource makes the stream of the next allocation wait for the buffer's release: in the engine, cuCascade's small pinned resource records an event when a slab is released and makes the next user's stream wait on it; standalone, cuDF's default pinned pool is stream-ordered, and its fallback allocations are freed synchronously. Second, cuDF synchronizes that stream after allocating a scalar's buffer and before writing to it (`rmm_host_allocator::allocate` in `cudf/detail/utilities/host_vector.hpp`, a synchronization cuDF marks for removal). The first orders reuse on the device; the second makes the host write wait for it. A pinned resource without reuse ordering, or a cuDF release that drops that synchronization, would break the guarantee. Decode therefore uses scalars only for predicate needles: constant-width offsets and all-null dictionary columns are built on the device, and no decode path constructs a numeric scalar, whose value cuDF uploads from a constructor argument on the stack.

Distinct compiled-kernel handles are pinned by their frame until the session's final stream drain. Releasing the last module handle can synchronize the CUDA context, and unloading a module with a queued launch is unsafe.

The supplied resource must honor the stream-ordered deallocation contract of `device_async_resource_ref`. A synchronous resource such as `cuda_memory_resource` remains correct but synchronizes the device on every release, which makes it unsuitable for production and for tests that gate a stream.

### Accounting

A call-time ledger such as cuCascade's `reservation_aware_resource_adaptor` credits a release when `deallocate` is called, before the GPU reaches it, as it does for every other engine operator. Decode's charge therefore peaks at the earlier outputs plus the active request's working set, so `pipeline_memory_history` and the downgrade trigger do not see dead temporaries. Physical reuse is per stream: the asynchronous pool reuses a released block immediately for later work on the same stream, while a release still pending on another lane becomes reusable when that lane reaches it. The physical peak can therefore exceed the ledger by roughly the other lanes' largest request working sets; under pressure, the pool waits on such pending releases before reporting out-of-memory.

## Errors and semantic decline

Private fused-buffer binders report validation failures through bool/optional results and an error string. Their required-region callers translate those failures to `std::runtime_error` before launching the region. Submitted execution failures, including typed allocation and compilation failures, propagate without conversion or whole-column retry.

Public compatibility functions retain their documented null/false plus `error_out` behavior for host validation or explicit decline. They do not turn submitted execution failures into successful ordinary decoding. A dictionary gather specialization may decline an unsupported shape, including after a completed metadata observation; OOM, malformed accepted data, compilation, or CUDA failures are not declines.

Predicate output type is validated once by `decode_request` before append succeeds. Session result storage retains only what final assembly needs: the column owner and an optional stored value type.

Once work has been submitted, `finish` checks every distinct supplied stream, including external phase work queued after the last frame submission. A host observation made during submission is not a reusable proof that the stream's current tail is complete. A query error does not prevent attempting synchronization and the remaining streams. An empty session returns without CUDA work; the caller completes external-only phase work separately.

Append failures preserve the original exception, including reservation OOM subtype and requested bytes; cleanup errors are secondary diagnostics. Finish reports its first completion failure. Both failures make the session terminal and prohibit result publication or further append. Successful finish also prohibits reuse.

The `noexcept` destructor attempts drainage in its body before destroying results and host state. A lost CUDA context cannot be repaired by a destructor; failed drainage is not a claim of safe pool reuse. Borrowed mask contents after a failed request are unspecified and must not be consumed without reinitialization.

## Genuine host observations

Removing completion-only leaf waits does not remove dependencies needed to discover output shape or publish shared state.

| Path | Observation retained |
|---|---|
| nvCOMP codecs | Header and compressed-size readbacks establish dimensions, pointers, and scratch sizing. Host upload arrays stay with the frame until the final drain; device metadata and scratch are released in stream order. |
| ALP and ALP_RD | Shared ALP initialization must complete before publication; reconstructed ALP_RD bit-width metadata needs a host observation. |
| Dictionary | An imported representation without prepared width metadata observes the reduction result during decode. Variable-width output sizing and selected-key inspection can also require observations. |
| Selected strings | Decoded lengths are scanned, then the total character count is observed before allocating the exact output buffer. |
| Scan selection | Survivor count is observed before exact-sized indices, policy decisions, and compacted output allocation; the count is staged through pinned memory like every other observation. |
| Predicate needles | Not an observation decode asks for: constructing each cuDF string scalar synchronizes its stream while cuDF allocates the scalar's pinned validity buffer. The generic comparison and the dictionary lookup table each wait once per needle. |

No path waits merely to release memory, and the dictionary gather specialization builds its offsets on the device, so it waits only for its key-offsets readback. The identity leaf copies its complete owned column, preserving the owning-copy path instead of converting it unnecessarily to a generic slice-aware view copy. Library-internal synchronization is not promised away.

### Dictionary width publication

The dictionary factory prepares immutable fixed-key-width metadata before publishing the representation. One CUB transform-reduction returns the common positive width, or zero if the keys are not uniformly positive-width. Empty-key shapes require no reduction allocation.

The factory allocates device scratch first, then an eight-byte result from cuDF's pinned, host-and-device-accessible resource. CUB writes directly into that result. The existing publication synchronization makes it readable on the CPU; no separate metadata D2H copy or new publication wait is needed. Scratch, pinned output, and key owners remain alive through failure drainage.

Imported or reconstructed dictionaries may lack this metadata. Their decode path shares the reduction algorithm but allocates scratch and the result in local device storage on the request's stream, then reads the result through the frame's checked scalar observation, which waits for the stream before that storage is released. The storage policies differ because their ownership boundaries differ, not because they select different decoders.

## Representative flows

Ordinary table decoding validates all requested indices, creates a session over the supplied stream views, appends one value request per requested column, and calls finish. Reordered or duplicate requests get distinct output owners in request order. Single-column and one-stream table facades use the same flow.

Scan-filter decoding keeps domain-specific phase orchestration outside the walker:

1. Preflight supported predicate and selection routes; own shared masks and an exception-safe phase cleanup scope.
2. Append numeric or membership mask requests and predicate BOOL8 requests to a first session. Preserve requested BOOL8 columns for dual delivery rather than decoding them again.
3. Join producer streams with dependency events, combine masks, and observe survivor count. Finish the first session. An all-declined source set takes the explicit no-filter policy before counting.
4. If filtering applies, create exact-sized indices and establish their readiness on consuming streams. Append selected requests or full-route candidates to a second session.
5. Finish the second session, perform any required grouped gather under phase ownership, validate output sizes/types, and publish the table.

The session does not expose lane assignments or pending column views to implement these phases. Final drainage includes phase work queued on its supplied streams. Post-join row sets enter the same validated-selection path with caller-established readiness and lifetime.

## Regression coverage

- [`test_async_decode.cpp`](../src/compression/simpatico_codegen/tests/test_async_decode.cpp): completed returns, concurrent submission, dictionary metadata, duplicate stream handles, release of temporaries during submission without waits (including the predicate, dictionary-gather, selected-string, full-width-then-gather, and mask routes), host-upload lifetime, pinned staging of host observations (`test_host_observation_staging`), abandonment, typed failures (including failure after enqueue), and error-priority drainage.
- [`test_scan_filter_session.cpp`](../src/compression/simpatico_codegen/tests/test_scan_filter_session.cpp): phase integration, predicate dual delivery, membership acceptance/decline, and selected output.
- Borrowed stream views: [`test_async_decode.cpp`](../src/compression/simpatico_codegen/tests/test_async_decode.cpp) checks that a table decode waits on each distinct view at most once and covers work others queued there; [`test_scan_filter_session.cpp`](../src/compression/simpatico_codegen/tests/test_scan_filter_session.cpp) decodes with repeated views that include the output stream and checks every result against expected values: a range, keep-mask, and full-width request and a table decode through both the `stream_pool` and view overloads, then membership, BOOL8-only (empty second wave), and predicate table requests through the views.
- [`test_masked_decode_variants.cpp`](../src/compression/simpatico_codegen/tests/test_masked_decode_variants.cpp): compacted kernel variants and selection boundaries.
- [`test_decode_reservation.cpp`](../test/cpp/compression/test_decode_reservation.cpp): actual engine reservation behavior, a peak charge of earlier outputs plus one request's temporaries, and preserved OOM handling.

Correctness checks must observe readiness before verification copies or extra synchronization can hide a missing wait. Performance checks compare matched unprofiled completed-call latency, while Nsight verifies launch/copy/synchronization behavior. Fewer lines and isolated kernel metrics do not by themselves establish unchanged end-to-end performance.
