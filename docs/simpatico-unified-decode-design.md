# Unified Simpatico decoding

Decode requests share one interpreter. Their enclosing session owns pending work and returns only completed results. Single-column, table, predicate, and selected-column entry points use this same mechanism; there is no synchronous/asynchronous decoder switch.

This document describes the current ownership and execution contracts. Benchmark histories and implementation experiments are not part of the API, and the design does not imply universal speedups or encoding parity with the earlier split implementation.

## Where to read the code

| Responsibility | Code |
|---|---|
| Completed public APIs and scan-filter phases | [`simpatico_codegen.cpp`](../src/compression/simpatico_codegen/src/simpatico_codegen.cpp) |
| Request types, ownership slots, and private session interface | [`decode_session.hpp`](../src/compression/simpatico_codegen/src/decode/decode_session.hpp) |
| Submission, retained-state accounting, and completion | [`decode_session.cpp`](../src/compression/simpatico_codegen/src/decode/decode_session.cpp) |
| Plan validation, `DecodeWalk`, fused-buffer binding, and selection routes | [`decompress.cpp`](../src/compression/simpatico_codegen/src/plan/decompress.cpp) |
| Representation reconstruction and standalone compatibility | [`representation_factory.cpp`](../src/compression/simpatico_codegen/src/plan/representation_factory.cpp) |
| JIT launch preparation and kernel retention | [`codegen_runtime.cpp`](../src/compression/simpatico_codegen/src/bridge/codegen_runtime.cpp) |
| Dictionary metadata preparation and decode | [`dictionary_compressor.cu`](../src/compression/simpatico_codegen/src/operators/dictionary_compressor.cu) |

## Decomposition

`decode_session` owns submission, memory-pressure retirement, and completion. Its private `append` overloads accept typed requests; `finish` drains work and transfers completed columns in request order. Its destructor drains abandoned work without throwing. Callers never extract pending columns or mark frames complete.

`column_decode_request` describes a borrowed plan or standalone representation, a value or predicate result, and an optional `validated_selection`. Value requests can restore the stored logical type after decoding. Predicate requests own their predicate metadata and produce BOOL8, optionally also writing a ballot mask. A predicate result cannot simultaneously request value-type restoration. Standalone requests support unselected values.

`mask_decode_request` describes a range predicate or membership probe and a borrowed mask destination. It does not occupy a returned-column slot. Its `accepted` or `declined` result describes semantic applicability, not device completion. A declined membership probe submits an all-ones mask; the scan orchestrator treats an all-declined source set as a policy decline before counting it.

`decode_frame` owns one request's temporary columns, structural memo, reconstructed representations, scratch, host uploads, and scalar backing storage. The session keeps a copy of the request so predicate strings, descriptors, and probe captures survive pending use. Compressed inputs and mask/selection device storage remain explicitly borrowed.

`DecodeWalk` interprets the plan with one structural memo. It targets predicate and selection semantics at the final value producer, not intermediate metadata. Codec leaves and JIT launchers receive the same mandatory frame; they do not know whether the caller is decoding a single column or a table.

The session owns each nonmovable frame directly in a stable list node. `decode_column_slot` is a non-owning handle to a preregistered unique owner, with single-assignment adoption. Column slots, buffer references, and host-upload spans stay valid as other owners are added. Upload records may move because their separately owned byte arrays do not.

## Completed public boundaries

Public decode calls return only after all their output and mask writes complete. The engine can then release compressed inputs or rebind output allocation streams according to its existing converter contract. Changing an RMM buffer's deallocation stream is not itself a CUDA ordering edge.

Callers establish input readiness and retain borrowed plans, representations, masks, row sets, and indices through session completion or destruction. Stream views and memory-resource references do not own those resources. The supplied resource must outlive returned allocations, and their stored streams must remain valid until rebinding or destruction.

Submission, retirement, and destruction stay on the constructing CPU thread and current device. This preserves the engine's per-thread reservation attachment. Explicit decode allocations use the supplied memory resource; opaque library allocations and working sets are separate from the session's retained-state estimate.

## Register before enqueue

For each append:

1. Apply pressure admission, reserve any returned-column slot, and register a stable frame before decode submission.
2. Retain every new GPU-read owner before exposing its pointer to subsequent work. Library constructors remain responsible for their own exception safety until ownership transfers.
3. Run the request through the walker and its leaves. Fill preregistered output slots; transfer shared memo values only to their last consumer, copying for earlier consumers. A consumed memo entry remains identifiable so accidental reuse fails explicitly.
4. Move the terminal output into private session result storage, cache retained-byte totals, and seal the frame. Sealing fixes ownership; it does not assert GPU completion. No later request may borrow sealed-frame scratch.
5. On failure, make the session terminal, drain all supplied physical streams while owners remain alive, and rethrow the original exception.

An adoption helper that temporarily receives an already-pending owner drains on bookkeeping failure before that local owner can unwind. The session catch alone would be too late to protect it.

Distinct compiled-kernel handles are pinned by the session until its final stream drain. Releasing the last module handle can synchronize the CUDA context, so kernel lifetime is not tied to earlier frame retirement or cache residency.

`host_array<T>` returns uninitialized, fundamentally aligned storage for trivially copyable values. Callers must initialize every consumed or uploaded byte before its first read or copy. The helper retains overflow checks and stable backing ownership; it is not a zero-initialized value container.

## Errors and semantic decline

Private fused-buffer binders report validation failures through bool/optional results and an error string. Their required-region callers translate those failures to `std::runtime_error` before launching the region. Submitted execution failures, including typed allocation and compilation failures, propagate without conversion or whole-column retry.

Public compatibility functions retain their documented null/false plus `error_out` behavior for host validation or explicit decline. They do not turn submitted execution failures into successful ordinary decoding. A dictionary gather specialization may decline an unsupported shape, including after a completed metadata observation; OOM, malformed accepted data, compilation, or CUDA failures are not declines.

Predicate output type is validated once by `decode_request` before append succeeds. Session result storage retains only what final assembly needs: the column owner and an optional stored value type.

Once work has been submitted, `finish` checks every distinct supplied stream, including external phase work queued after the last frame submission. A successful prior retirement, or an empty frame list, is not a reusable proof that the stream's current tail is complete. A query error does not prevent attempting synchronization and the remaining streams. An empty session returns without CUDA work; the caller completes external-only phase work separately.

Append failures preserve the original exception, including reservation OOM subtype and requested bytes; cleanup errors are secondary diagnostics. Finish reports its first completion failure. Both failures make the session terminal and prohibit result publication or further append. Successful finish also prohibits reuse.

The `noexcept` destructor attempts drainage in its body before destroying pending owners. A lost CUDA context cannot be repaired by a destructor; failed drainage is not a claim of safe pool reuse. Borrowed mask contents after a failed request are unspecified and must not be consumed without reinitialization.

## Genuine host observations

Removing completion-only leaf waits does not remove dependencies needed to discover output shape or publish shared state.

| Path | Observation retained |
|---|---|
| nvCOMP codecs | Header and compressed-size readbacks establish dimensions, pointers, and scratch sizing. Host upload arrays and device metadata remain frame-owned afterward. |
| ALP and ALP_RD | Shared ALP initialization must complete before publication; reconstructed ALP_RD bit-width metadata needs a host observation. |
| Dictionary | An imported representation without prepared width metadata observes the reduction result during decode. Variable-width output sizing and selected-key inspection can also require observations. |
| Selected strings | Decoded lengths are scanned, then the total character count is observed before allocating the exact output buffer. |
| Scan selection | Survivor count is observed before exact-sized indices, policy decisions, and compacted output allocation. |

Known-size identity and fused decoding retain dependencies rather than waiting merely to free them. The identity leaf copies its complete owned column, preserving the owning-copy path instead of converting it unnecessarily to a generic slice-aware view copy. Library-internal synchronization is not promised away.

### Dictionary width publication

The dictionary factory prepares immutable fixed-key-width metadata before publishing the representation. One CUB transform-reduction returns the common positive width, or zero if the keys are not uniformly positive-width. Empty-key shapes require no reduction allocation.

The factory allocates device scratch first, then an eight-byte result from cuDF's pinned, host-and-device-accessible resource. CUB writes directly into that result. The existing publication synchronization makes it readable on the CPU; no separate metadata D2H copy or new publication wait is needed. Scratch, pinned output, and key owners remain alive through failure drainage.

Imported or reconstructed dictionaries may lack this metadata. Their decode path shares the reduction algorithm but allocates scratch and the result in frame-owned device storage, then uses the frame's checked scalar observation. The storage policies differ because their ownership boundaries differ, not because they select different decoders.

## Pressure-only stream-tail retirement

The session has private soft limits of 64 MiB estimated retained temporary device bytes, 8 MiB host-upload bytes, and 64 registered frames. These are provisional queueing limits, not a batch-size limit, physical-memory cap, or established tuning optimum.

The device estimate combines direct buffer capacities, temporary-column `alloc_size()` values, representation-owned bytes, and supplied scalar estimates. It excludes borrowed input, terminal output, hidden allocation slack, allocator overhead, and opaque library scratch. A full decoded column used as predicate or gather input is an intermediate and is counted. Ownership transfer does not count an allocation twice; views and aliases add no owned capacity.

A 32 GiB returned table is therefore not charged as 32 GiB of retained scratch. Required phase masks and grouped-gather sources have their own ownership outside individual frames. Retained ownership, reservation accounting, and allocator/pool high-water usage are different measurements.

Retirement works as follows:

1. At admission and allocation checkpoints, combine cached sealed-frame totals with the active frame's current owners. If the prospective allocation/frame fits, return without a CUDA retirement call. Reusing a stream alone never waits.
2. Under pressure, query each distinct physical stream with sealed frames. A successful query permits erasing all sealed frames on that handle. A not-ready result preserves them; another CUDA error enters normal failure cleanup.
3. If pressure remains, wait for the oldest sealed frame's stream tail, retire its sealed peers, and reassess. Never erase the active frame, including one sharing a completed handle with sealed frames.
4. Allow the intrinsic working set of one active frame to exceed the byte window. After it seals, the next append must relieve pressure before admitting another frame. The frame-count limit is not waived.

Duplicate stream views retain their round-robin weighting but share one completion domain. Final drainage visits each physical handle once. Completion observations are not cached across later submissions or asynchronous frees.

There are no per-frame retirement events. Selection-phase dependency events remain because they establish actual producer/consumer ordering.

### Progress and physical reuse

Stream-tail observation is deliberately coarser than an event after an individual frame. If A finishes and later B remains pending on the same stream, A cannot be reclaimed by a tail query. Pressure can wait for B as well. Without pressure, both can remain retained until finish.

Every dependency already queued on a supplied stream must be satisfiable independently of a future call or return on the submitting CPU thread. A gate released only by that caller after a pressured append can deadlock. An independently progressing producer stream or thread satisfies the contract.

Retiring a frame destroys its owners on the caller thread. An asynchronous memory resource may only enqueue the corresponding frees at the current stream tail. The ledger decrement is not proof of immediate cross-stream physical reuse or newly available reservation capacity. Kernel pins remain until final drainage.

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

- [`test_async_decode.cpp`](../src/compression/simpatico_codegen/tests/test_async_decode.cpp): completed returns, concurrent submission, ownership stability, dictionary metadata, pressure and aliased streams, abandonment, typed failures, and error-priority drainage.
- [`test_scan_filter_session.cpp`](../src/compression/simpatico_codegen/tests/test_scan_filter_session.cpp): phase integration, predicate dual delivery, membership acceptance/decline, and selected output.
- [`test_masked_decode_variants.cpp`](../src/compression/simpatico_codegen/tests/test_masked_decode_variants.cpp): compacted kernel variants and selection boundaries.
- [`test_decode_reservation.cpp`](../test/cpp/compression/test_decode_reservation.cpp): actual engine reservation behavior and preserved OOM handling.

Correctness checks must observe readiness before verification copies or extra synchronization can hide a missing wait. Performance checks compare matched unprofiled completed-call latency, while Nsight verifies launch/copy/synchronization behavior. Fewer lines, lower retained-owner counts, and isolated kernel metrics do not by themselves establish unchanged end-to-end performance.
