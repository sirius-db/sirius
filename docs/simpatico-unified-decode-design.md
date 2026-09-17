# One decode path, one completion owner

Implementation design · revised 2026-09-16 · **v5 validated; scoped independent review approved**

The unified decoder is already implemented. This revision replaces its per-frame retirement events with private, pressure-only stream-tail retirement, removes eager host-container allocation and a redundant separately allocated frame owner, and restores an owning-column copy expression unintentionally replaced during unification. A narrow follow-up makes frame-owned host staging explicitly uninitialized, because all consumed bytes are subsequently written. It preserves the unified path; it neither restores the additive decoder nor introduces another execution mode. Selection-phase dependency events remain necessary and unchanged.

## Decision in one page

Use a **scoped `decode_session`** to own submission, temporary lifetimes, bounded retirement, and completion. Keep **one `DecodeWalk`** to interpret a column's plan. Give every decoding leaf the same, mandatory ownership context; whether the caller wants one column or a table does not reach the walker or kernel launcher as an execution-mode flag.

The session has two operations: append a typed decode request, and finish. Its destructor handles abandoned or failed work without throwing. Only `finish()` reports successful completion and releases results to the caller.

```cpp
// Private implementation API, not a new asynchronous engine API.
class decode_session final {
 public:
  decode_session(std::span<rmm::cuda_stream_view const> streams,
                 rmm::device_async_resource_ref mr);
  ~decode_session() noexcept;

  decode_session(decode_session const&) = delete;
  decode_session& operator=(decode_session const&) = delete;
  decode_session(decode_session&&) = delete;
  decode_session& operator=(decode_session&&) = delete;

  void append(column_decode_request const& request);
  mask_source_status append(mask_decode_request const& request);
  std::vector<std::unique_ptr<cudf::column>> finish();
};
```

The overloads describe different *results*, not different execution policies. A column request produces values or predicate BOOL8, optionally selected. A mask request produces a ballot into an explicitly borrowed mask destination, from a numeric range or an existing membership probe. Its `accepted`/`declined` return describes semantic applicability, **not device completion**. A predicate-column request can additionally ballot its BOOL8 result into a mask, preserving the existing dual-delivery optimization. Only column requests occupy returned-column slots, in request order.

Each request creates one private, stably addressed **frame**. A frame owns its resolver/memo, reconstructed representations, scratch, host upload storage, scalar backing storage, and pending output. Once the whole request has been submitted and its output has moved into private session result storage, the frame is **sealed**: its retained ownership no longer changes, but its GPU work need not be complete. Only under retained-memory/frame-count pressure does the session query stream tails and reclaim sealed frames on completed streams. If queries cannot relieve pressure, it waits on the oldest sealed frame's stream tail. No frame has a CUDA event or exposed completion handle. Several frames can share a stream; stream reuse alone never causes a wait.

The session pins each distinct loaded kernel before launch until its final stream drain: dropping the last module handle can itself synchronize the context. Final completion and failure recovery always check/drain all distinct supplied streams, including externally queued phase tails, before releasing still-pending ownership or publishing results. Compared with frame events, stream-tail retirement deliberately gives up reclamation of an earlier frame while later work on that stream remains pending; §7 states that progress contract explicitly.

```text
completed public decode APIs / scan-filter orchestration
                         │ typed requests
                         ▼
                 decode_session
          ownership · retirement · completion
                         │ owns frames
                         ▼
                    DecodeWalk
        one structural memo / root semantic target
                  ┌──────┴──────┐
                  ▼             ▼
          fused JIT launch   standalone leaves
                  └──────┬──────┘
                         ▼
             supplied streams and memory resource
```

The session is deliberately deep: callers do not coordinate `enqueue`, `mark_completed`, `take_output`, `seal`, or `drain`. The walker does not schedule tables. Leaves do not decide whether the caller is synchronous. Existing completed-on-return public signatures remain.

There is **no promise that every leaf is nonblocking**. Device-to-host observations needed to determine sizes, null counts, or selection counts remain local barriers. Completion-only waits disappear from the final design because their dependencies have explicit owners. The expected benefit is simpler correctness and broader removal of host submission gaps, not a new compression algorithm or a promised speedup.

## 1. Scope and evidence

In scope: direct column decoding, all/projected/reordered/duplicate table columns, fused metadata dependencies, standalone codecs, nullable identity and other existing null semantics, predicate substitution, mask/index/CSR row selection, compacted strings and dictionaries, and failure cleanup. Scope includes adapting decode-facing leaf APIs where lifetime ownership currently escapes them.

Out of scope: encoding redesign, disk tier, new compression formats, new predicate/nullable-selection algorithms, cross-query asynchronous engine results, persistent scratch caches, CPU worker pools, generic GPU tasks/futures/coroutines, allocator replacement, and a new configuration surface. Existing codec dispatch inheritance need not be redesigned merely to change its decode contract.

This document describes the unified working tree descended from `52d02a18`, including the eager dictionary-width publication fix. The selected end state has stream-tail retirement, directly embedded frames, three stable-address owner lists, a vector of separately owned upload arrays, and the original owning-column copy semantics in the identity leaf. Preserve the existing single walker, mandatory frame ownership, typed requests, supplied-MR propagation, and leaf observation boundaries.

The [bounded remediation history](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/EXPERIMENTS.md) records the measured event-free candidates and CPU attribution that led to these changes. The [final v5 validation report](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/REPORT.md) records completed correctness, sanitizer, engine, latency, profiler and memory checks. The earlier v4 **0.13% / 74 microsecond** plain-ANS regression was not waived: the staging amendment removed it, with the independent [additive/v5 confirmation](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/ans-confirm-v5.json) measuring **0.558% lower latency**, all five pairs favoring v5. The [independent review](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/REVIEW.md) gives scoped approval with no correctness findings and the measurement limitations below; it is not a universal no-overhead guarantee.

Numeric 32 GiB results are near parity, not a material speedup claim. Flagged small- and large-key aggregates did not reproduce in fresh [small-matrix](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/small-confirm-v5.json) and [large-key](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/large-key-confirm-v5.json) confirmations. The small identity/one-stream confirmation is **0.45 microseconds / 0.61% slower**, with five of seven pairs slower; its primary aggregate was 2.25% faster, but three of five individual pairs were slower. That is modest directional evidence for a possible sub-microsecond identity cost amid much larger process variation, not proof of zero overhead. Small delta/four-stream confirmation was 0.15% slower with mixed paired signs and an opposite primary aggregate. The report and scoped review preserve these limitations and both collections. Measured LZ4, entropy-tail, wide entropy and fixed-string improvements belong to the complete candidate, not solely to retirement removal or a new kernel algorithm.

The [fresh additive validation report](../../sirius-simpatico-eval-20260914/61-revalidate-20260915-DIJCYb/REPORT.md) remains the immediate performance baseline, not evidence for the proposed retirement policy. At **32 GiB total, eight distinct 4 GiB columns, four streams**, that additive patch reduced completed decode latency from 11.847796 to 11.307440 ms for bitpack, 13.836738 to 13.292158 ms for wider/key bitpack, 11.433585 to 10.926349 ms for delta-to-bitpack, and 10.915309 to 10.764460 ms for identity. Those are 4.56%, 3.94%, 4.44%, and 1.38% reductions. The baseline already includes the chunk-local index-width optimization.

Fresh Nsight evidence found 20 to 4 production stream waits, unchanged 32 kernels and zero numeric-decode copy bytes, and only 15.9–17.3 microseconds of value-kernel overlap in a 32 GiB decode. The principal demonstrated gain is reduced host gaps, **not extensive concurrent execution of large kernels**. The unified design must preserve that gain rather than recreate per-column waits under a cleaner name.

## 2. What changes now, and what must not regress

The first three rows describe the pre-change event candidate; links identify the private implementation areas being replaced, not a requirement to retain the old fields.

| Pre-change implementation / preserved boundary | Design problem | End state |
|---|---|---|
| [`submitted_frame` and session append/finish](../src/compression/simpatico_codegen/src/decode/decode_session.cpp#L248) create, record and destroy one event per request. | Even under all retained-state limits, short requests pay retirement API costs. The measured small-batch regressions were rejected; good large-batch results do not waive that gate. | No per-frame CUDA object or CUDA call to seal a frame. |
| [`retire_ready` / `pressure`](../src/compression/simpatico_codegen/src/decode/decode_session.cpp#L361) reclaim event-completed prefixes. | This stronger reclamation precision costs lifecycle work on every request. | Under pressure only, observe a physical stream's current tail and retire its sealed frames together. Explicitly accept the weaker progress property. |
| [`decode_session_stats`](../src/compression/simpatico_codegen/src/decode/decode_session.hpp#L168) counts event queries. | Event-specific diagnostics and fault injection would describe a mechanism that no longer exists. | Count pressure-time stream queries and actual pressure waits; remove retirement-event fault wrappers and assertions. |
| [`DecodeWalk`](../src/compression/simpatico_codegen/src/plan/decompress.cpp#L106) and [`decode_frame`](../src/compression/simpatico_codegen/src/decode/decode_session.hpp#L92) already share one mandatory ownership contract. | A literal rollback to additive decoding would restore whole-plan eligibility, optional lifetime modes and caller completion bookkeeping. | Keep this unification unchanged. No leaf, renderer or predicate-policy rewrite is needed for retirement. |

The original additive problems—optional workspace/lifetime modes, an async eligibility allowlist, and external `mark_completed()`—have already been removed. Retain that improvement. The session hides the retirement-policy change from the structural memo, fused renderer and codec leaves; that is the intended benefit of its module boundary.

## 3. The small set of concepts

### Session: one owner for a bounded decode scope

The session owns frames, result slots, stream-use/error state, and private retirement accounting. Construction copies the stream-handle list: it borrows the actual CUDA streams and MR, **not the backing array of the caller's span**. It does not create a CPU executor or acquire a reservation itself. Construction performs host setup only and enqueues nothing. All methods and destruction run on the constructing CPU thread and current device.

`append` validates and copies small request metadata, registers its frame and result slot, then invokes the walker. It can return with device work pending. It may block for a genuine leaf observation or explicit memory backpressure. This fact is internal: all existing public APIs still finish before returning.

`finish` checks/drains every distinct supplied stream once the session has submitted work, then validates and retags completed column outputs, and finally transfers the result vector. This includes supplied streams used by surrounding phase work, not just streams with remaining frames. No result or pending column view escapes earlier. A second finish or append after finish/failure is a deterministic logic error. Empty sessions return empty results without CUDA work; their caller must separately complete any external-only phase work.

### Frame: private storage, not a second execution object

Each `submitted_frame` list node directly contains its nonmoving `decode_frame`; `std::list<submitted_frame>` supplies stable, submission-ordered storage without an additional frame allocation. Appending or erasing other nodes never relocates a surviving frame. A frame contains the existing structural memo and one walker (or the walker is a short-lived visitor over that memo), plus typed retained storage. It has no public `enqueue`, `wait`, or completion token. It may refer back to its owning session for retirement pressure; that coupling stays inside one implementation module. A host-side sealed flag distinguishes a fully submitted request from the active frame. Sealing is not a completion observation.

Use a few domain-specific collections: device buffers, temporary columns, reconstructed representations, scalar owners, host-upload byte blocks, and `shared_ptr<CompiledKernel const>`. Merge `DecodeMemo::kept` into this ownership rather than keeping two representation-retention lists. Do not add `shared_ptr<void>`, `std::function` cleanup lists, arbitrary deleter registries, or a polymorphic GPU-task base class.

The frame's storage helpers reserve/register an owner **before its pointer is passed to GPU work**. If adopting an already-created object requires a potentially throwing container growth, the helper must either reserve beforehand or keep that object alive while draining its stream on the exceptional path. “Move it into a vector after launch” is not an exception-safety argument.

The owning output remains inside the session, even if its frame is retired early. Internal moves into pre-registered result slots are nonthrowing ownership bookkeeping, not publication. Stored-dtype restoration and public transfer wait for successful final completion of the whole session.

### Storage amendment: stable owners without eager empty allocations

The first stream-tail candidate still showed a repeatable short identity-decode cost against additive; removing events was not sufficient. Inspection of the installed GCC 14.3 libstdc++ establishes a concrete host cost: each default `deque` allocates a map and one node even when empty ([constructor](../.pixi/envs/default/lib/gcc/aarch64-conda-linux-gnu/14.3.0/include/c++/bits/stl_deque.h#L458), [initialization](../.pixi/envs/default/lib/gcc/aarch64-conda-linux-gnu/14.3.0/include/c++/bits/stl_deque.h#L639)). The earlier frame's four deques therefore required 64 such host allocations across an eight-column table before storing any element. This was a source-established cost, not proof that it explained the entire measured regression.

Use these concrete standard containers inside the existing frame:

| Member | Selected storage | Address that must remain stable |
|---|---|---|
| `columns_` | `std::list<std::unique_ptr<cudf::column>>` | The `unique_ptr` owner slot itself: `decode_column_slot` points to that slot, not merely its column pointee. |
| `buffers_`, `output_buffers_` | `std::list<rmm::device_buffer>` | The buffer object returned by reference, as well as its separately allocated device data. |
| `uploads_` | `std::vector<host_upload>` | The byte array owned by each record's `unique_ptr`, **not** the movable record or a vector iterator. |

The installed list uses an inline sentinel ([implementation](../.pixi/envs/default/lib/gcc/aarch64-conda-linux-gnu/14.3.0/include/c++/bits/stl_list.h#L450)); the installed vector starts with null storage ([implementation](../.pixi/envs/default/lib/gcc/aarch64-conda-linux-gnu/14.3.0/include/c++/bits/stl_vector.h#L101)). Thus these empty containers add no heap allocation on this toolchain. This is not a portable claim about every standard-library implementation. List insertion preserves existing element addresses; upload-vector growth moves `unique_ptr` records without moving their byte arrays. Keep `host_upload` nothrow-move-constructible, preferably guarded by a private `static_assert`, and never expose a reference/iterator to an upload record. Retain the existing fundamental-alignment constraint on typed upload spans. Include `<list>` and `<vector>` directly rather than relying on transitive headers.

No helper signature, frame state, codec branch, MR, retained-byte checkpoint, or completion rule changes. The helpers already use append/back/iteration, not random indexing. Existing `keep_column`/`keep_buffer` adoption catch-and-drain guards stay intact: a failed node allocation must not destroy an incoming owner before its queued use completes. New upload storage is not exposed to GPU work until insertion succeeds; allocation failure leaves earlier owned arrays intact and follows the existing session failure drain. Slot adoption remains nonthrowing and single-assignment. Final-output buffers remain excluded from temporary accounting; changing their host container does not relabel their device bytes.

This pays one host node allocation per **actual** list element and sacrifices deque locality/block amortization. Many-temporary codecs may therefore trade differently from identity; verify scratch-heavy cases as well as the small numeric matrix. Do not add a pool, custom small container, eager `reserve`, empty-frame mode, or another ownership wrapper. This is information hiding doing its job: a storage representation changes behind the same frame helpers, without a new caller protocol or another loss of retirement precision.

### Storage amendment: direct frame composition

The container-only candidate improved some workloads but did not clear the repeatable small-identity gate. Direct composition removes a redundant ownership indirection: the session already allocates a stable [`submitted_frame`](../src/compression/simpatico_codegen/src/decode/decode_session.cpp#L248) list node, so that node directly owns its `decode_frame` member instead of a separately allocated `unique_ptr<decode_frame>`. This removes one host allocation/deallocation pair and one pointer indirection per request, without reducing the number of requests, retained owners or completion checks. Its isolated trial also did not explain the full residual; it remains a simpler ownership representation, not a standalone speedup claim.

The node constructor accepts the existing stream, MR, enclosing session reference, request value and lane, then constructs the frame **in place**. Pass those constructor arguments to `frames.emplace_back`; do not construct or move a temporary frame/node. Keep `decode_frame` and the containing node noncopyable and nonmovable, with private compile-time assertions if helpful. Keep the frame at its existing first-member position so relative destruction order with the copied request is unchanged. No public frame constructor, factory, ownership wrapper or new API is needed.

The safety argument is unchanged but more direct: list insertion does not move existing nodes; list erasure destroys only selected nodes ([installed `erase_if`](../.pixi/envs/default/lib/gcc/aarch64-conda-linux-gnu/14.3.0/include/c++/list#L99)). Frame/request construction performs no GPU work, so failed node allocation or construction cannot leave newly queued work. The existing append catch still drains earlier requests. Only after successful node registration does submission begin. Sealing, active-frame exclusion, same-thread retirement, session-final module pins, and destructor-body drainage stay exactly as before. Embedded destruction releases the same owners at the same proven-complete point; it only avoids deleting a second host allocation.

Preserve the container-only candidate and compare each new binary against both it and additive. Larger list-node layout may affect host locality, so measure rather than promise a universal win. Reuse the retained-slot/reference/span growth test, under-limit multiappend gate tests, and active-frame-survives-other-retirement tests; do not mutate sealed frames or invent multiple simultaneous active walks just to test node stability. Keep frame-registration, adoption-failure, OOM and all-stream error-cleanup coverage. No other capacity, allocator or result-assembly change belongs in this candidate.

### Copy semantics: remove an unintended owning-to-view conversion

The final bounded correction restores the original copy expression inside the [mandatory-frame identity leaf](../src/compression/simpatico_codegen/src/plan/representation_factory.cpp#L953): `cudf::column(*channels_[0], frame.stream(), frame.mr())`, immediately adopted into its pre-registered slot. Unification had unnecessarily converted that complete owned column to `column_view` and selected cuDF's generic, offset-aware deep-copy constructor. Source comparison and CPU profiles established the changed call path; inclusive profile percentages alone were not a causal latency measurement.

The [owning-copy API](../.pixi/envs/default/include/cudf/column/column.hpp#L43) deep-copies the column using the explicit stream and MR; the [view-copy API](../.pixi/envs/default/include/cudf/column/column.hpp#L122) additionally handles view offsets. No slice is involved in this leaf. Dtype, row count, nulls, children, independent output ownership, source lifetime and frame adoption retain their original contracts. Selection/gather callers that really operate on views stay unchanged. This is removal of an unintended conversion, **not** an identity-specific execution mode or a new fast-path branch.

Verify empty/nullable/string/nested-owned inputs and physical allocation/copy counts: copying a whole owner can preserve padding that a logical view copy need not copy. The v4 attribution campaign and completed v5 short-batch, 32 GiB, mixed/scratch-heavy and engine campaigns preserve that distinction; the later staging amendment resolves v4's separate ANS residual. The plan-validation recursion was left unchanged: additive already performed equivalent cycle validation twice per eligible numeric column, versus once in a plain unified request. Do not generalize that comparison to non-codegen fallbacks such as ANS, whose former eligibility check declined before cycle validation. Preserve validation rather than removing a safety check to chase a small timing difference.

### Staging amendment: initialize once, before consumption

The private [`host_array<T>`](../src/compression/simpatico_codegen/src/decode/decode_session.hpp#L118) helper returns **uninitialized, frame-owned staging storage**. The caller must initialize every element or byte that will be consumed on the host or copied to the device before its first such use. Allocate its backing byte array with C++20 `std::make_unique_for_overwrite<std::byte[]>`, not value-initializing `make_unique`. There is no second zeroed/uninitialized mode: this helper is staging storage, not an initialized value container. Retain the existing trivial-copyability, fundamental-alignment and overflow constraints. Byte-array allocation still provides the same alignment and implicit-lifetime storage; only redundant value initialization is removed.

The complete current-call-site audit found no zero dependency: nvCOMP's header and compressed-size array, dictionary offsets, and `read_scalar` receive full checked device-to-host writes before consumption; nvCOMP's three upload arrays assign every scalar/pointer element before their full host-to-device copies. No production upload uses a partially initialized struct with padding. Existing tests leave some accounting-only storage unused, but never consume its unwritten bytes. Future callers must satisfy the same explicit write-before-read contract; memory that needs semantic zeros must write them deliberately.

This changes neither the byte checkpoint nor owner registration: allocate and insert the separately owned array before exposing its span. Allocation/insertion failures still preserve earlier arrays, retain their queued uses, and follow the existing session drain. Upload-vector growth still moves records, not backing arrays. Supplied MR, caller-thread cleanup, resource lifetimes, pressure limits, and host-observation barriers are unchanged.

The [historical ANS timeline evidence and read-only reproducer](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/ANS_TIMELINE.md) localize a v4 cost to the interval after the size-table readback completes and before the metadata device allocation begins: **93.945 to 99.618 microseconds per column**, or **45.38 microseconds per eight-column table**, across ten profiled decodes. That interval includes host-array allocation/initialization, metadata filling and ownership/accounting work; Nsight does not isolate their individual costs. Both additive and v4 zero-initialized the large arrays, so redundant zeroing is an opportunity in both, **not a demonstrated newly introduced cause**. Free-to-next-allocation observations do not support an allocator-blocking explanation for the residual.

The completed [controlled v4/v5 comparison](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/overwrite-direct-v5.json) changes this allocation expression and measures 57.021264 to 56.638318 ms, **0.6716% lower latency**, across five pairs with 31 hot samples per process; every pair favors v5. The separate additive/v5 confirmation measures 56.963696 to 56.645838 ms, **0.5580% lower latency**, again favoring v5 in every pair. Fresh [v5 timeline evidence](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/ans-timeline-v5.json) places the preparation interval at 72.905 microseconds per column. That diagnostic interval supports the mechanism but is not an additive causal breakdown of the unprofiled speedup. Targeted staging/nvCOMP, sanitizer, engine and broad performance checks were rerun on v5. No pressure-policy tuning, allocator reordering, new codec path, or acceptance waiver accompanies this amendment.

### Requests: semantics, not lifecycle options

Request values describe the existing behavior:

```cpp
struct value_result {
  std::optional<cudf::data_type> stored_type;
};
struct predicate_result {
  decode_predicate predicate;             // owns strings/other small metadata
  std::optional<mask_destination> ballot;
};
using decode_source = std::variant<
    std::reference_wrapper<PlanTree const>,
    std::reference_wrapper<standalone_compressed_representation const>>;
struct column_decode_request {
  decode_source source;                    // borrowed through finish/destruction
  std::variant<value_result, predicate_result> result;
  std::optional<validated_selection> selection;
};

// membership_source copies the existing pinned probe closure and its key dtype.
struct mask_decode_request {
  PlanTree const& plan;
  std::variant<range_predicate, membership_source> source;
  mask_destination destination;           // borrowed output, not public column result
};
```

These are illustrative C++20 value sketches, not additional public API commitments. `mask_destination` can reuse the existing typed mask-word view. `validated_selection` is a private validated copy of the existing descriptor, not another capability registry. It records one source (scan mask with optional indices, or a chunk row set), its established count, and the existing `probe_column` route. It rejects contradictory sources/sentinels/routes before enqueue. Public `decode_selection` compatibility can remain at the boundary while its nullable bag is translated once.

The small result variant prevents a predicate BOOL8 result from also requesting a source-dtype retag, and makes ballot delivery available only with a predicate result. It is a semantic sum type, not a request class hierarchy. The standalone source alternative lets the completed representation wrapper enter the same session without manufacturing a `PlanTree`; it directly dispatches the same mandatory leaf interface. Initially that compatibility source requests unselected values only; root predicate/selection semantics remain plan-based. This is a source-shape distinction, not asynchronous eligibility. A mask-only request has no column slot. Repeated or reordered column requests get distinct output owners and preserve request order; no implicit deduplication changes ownership or semantics.

### Walker and leaves: one materialization contract

`DecodeWalk(plan, frame, request)` has one resolver/memo. The frame supplies stream, MR, owned storage, checked host observations, and kernel retention. It does not supply an `async`, `deferred`, or `retain` flag.

All decode leaves fill a pre-registered, frame-owned column slot with established host-visible shape; contents may still be pending on the frame stream. `unique_ptr<cudf::column>` remains the actual storage/public result type, but a pending leaf result is not handed back through an unprotected local owning pointer. Every GPU-read dependency is either frame-owned, an explicitly borrowed input that outlives the session, or owned by a library whose documented stream/lifetime contract covers its internal temporaries. The **final architecture moves Sirius-owned completion-only dependencies into the frame**; “this codec is synchronous” is not a permanent alternative ownership mode.

Keep existing representation dispatch. Adapt its decode-facing method/helper to accept the mandatory decode context; standalone completed test/public wrappers create a one-request session. Do not add parallel `decompress_async` virtual methods beside old synchronous virtual methods. Representation inspection/export needed by encoding remains outside this change unless a decode call currently relies on mutable lazy state; that case is handled explicitly below.

## 4. Completion, ownership, and failure

### A completed API really is completed

The [host converter](../src/compression/compression_converters.cpp#L118) first completes H2D reconstruction, then calls table decode and immediately rebinds output buffers to the pipeline stream. The [GPU converter](../src/compression/compression_converters.cpp#L216) also immediately rebinds; compressed GPU pins are published only after producer completion in [`pin_table.cpp`](../src/pin_table.cpp#L873). Preserve those boundaries.

The public caller guarantees input readiness and keeps compressed plans/buffers alive through the call. Internal selection producers establish event dependencies into every consuming stream. On successful public return, all output and mask writes are complete and inputs can be released or output allocation streams rebound according to the existing engine contract. [`rmm::device_buffer::set_stream`](../.pixi/envs/default/include/rmm/device_buffer.hpp#L371) only changes the deallocation stream; it creates no ordering edge.

### Resource and thread contract

Use the explicitly supplied MR for every decode allocation, including JIT metadata/scratch, reconstructed channels, predicate scalars where their API accepts it, nullable copies, masks, and leaf upload/device metadata. Remove decode helpers' calls to `get_current_device_resource_ref()` when an MR is already supplied. Audit library operations that have separate output and scratch resource parameters; document unavoidable library-owned allocation behavior instead of implying that a parameter controls every hidden allocation.

`resource_ref` and `cuda_stream_view` are borrowed capabilities, not owners. The MR must outlive retained work and returned allocations; the streams associated with returned buffers must remain valid until the engine's completed rebind or buffer destruction. The owning wrapper type found in installed RMM does not imply ownership of a referenced upstream resource. Use installed headers, not legacy examples in generated docs.

This is not an async-MR-only design. Sirius-owned temporaries are retired after their last GPU use is proved complete, independent of whether freeing would otherwise be stream-ordered. The live engine happens to use [`cuda_async_memory_resource` plus a reservation adaptor](../cucascade/src/memory/memory_space.cpp#L114). Reservation accounting is separate: [`sirius_config.cpp`](../src/sirius_config.cpp#L455) selects per-thread tracking, and the adaptor's [TLS state](../cucascade/src/memory/reservation_aware_resource_adaptor.cpp#L95) is per thread **and adaptor instance**. Allocation, retirement, and destruction therefore remain on the calling thread while its reservation attachment is active.

### Register first; enqueue second

For every append:

1. Validate host-only request structure and semantic route. Reserve the result slot and any host bookkeeping needed to register a frame.
2. Install a stably addressed owning frame before work. Construction must not submit work; failed frame registration must not lose earlier pending frames. Frame ownership, not a completion token, protects the failure path.
3. Before a first potentially enqueuing allocation/library call, mark that lane as used. Adopt each new GPU-read owner before exposing its pointer; an allocation/library constructor must uphold its own exception guarantee until ownership can transfer.
4. Run the single walker, including predicate, probe, gather, and ballot work belonging to this request. Move its output into the pre-registered private result slot, compute cached retained-byte totals, then seal the frame using a host-only transition. Sealed ownership no longer changes, while the active frame still requires live accounting. An intermediate host observation does not seal the frame. No later request or external work may borrow sealed frame scratch; session result owners and phase-owned masks remain separate.
5. On any exception, make the session terminal, drain/check all distinct supplied streams while owners remain alive, then rethrow the original exception. Do not retry the whole column after partial submission.

An append that fails before registration/enqueue can still terminate the session and drain earlier appends. This keeps one simple failure contract. Allocation failure while growing a frame's host storage is as important as an RMM failure.

This is a basic exception guarantee for submitted work, not rollback: already-enqueued writes cannot be undone. No partial column result is published. A borrowed mask destination may contain partial/unspecified data after failure and must not be consumed or reused without reinitialization; successful drainage establishes lifetime safety, not valid mask contents.

### Preserve failures; do not turn them into policy

Use one throwing internal decode contract. Invalid plans, renderer failures, launch failures, and completion failures are errors; preserve useful diagnostic context without flattening dynamic exception types. In particular, propagate `rmm::out_of_memory` / `cucascade_out_of_memory` unchanged with `throw;` or the original `exception_ptr`. [`gpu_pipeline_task.cpp`](../src/pipeline/gpu_pipeline_task.cpp#L639) uses that exact type and `requested_bytes` to reschedule under a revised reservation.

A semantic optimization may decline only through an explicit result produced by preflight or a deliberately completed observation phase: unsupported predicate route, selectivity policy, or a dictionary specialization that does not apply. If a specialization needs metadata readback before deciding, retain its owners, complete that local observation, and then decide. It must not return “not applicable” for OOM, compilation, CUDA, or malformed-data failures. Remove the broad filtered-decode catch-and-fallback; ordinary decoding remains a valid response to an explicit policy decline, not an error-recovery mechanism.

Compatibility facades that currently expose `nullptr`/`false` plus `error_out` may translate the specifically documented validation/decline result at the outer boundary. They must not catch all exceptions or swallow allocator/device failure. Internal code has no second bool-error launch contract.

### Destruction order and error priority

`finish()` records the first completion error but attempts a checked stream drain on every distinct supplied stream before releasing any still-pending owner. Query each stream freshly; synchronize it if not ready, and still attempt synchronization/other streams after another query error while preserving the first failure. If append already failed, its original exception takes precedence over cleanup errors. Prior pressure retirement is not a reusable completion proof: later append, asynchronous frees, or external phase work may have extended that stream's tail. Abort also drains all supplied streams, covering an active, partially submitted frame. Only after all required completions succeed are results validated, retagged, and returned.

The session transitions `open -> finished` only after successful completion/assembly; any append or finish failure makes it `failed`. Both terminal states reject further work. Frames transition `active -> sealed -> retired`, with a failed active frame retained for session cleanup. Pressure never erases an active frame, even when its stream temporarily reports complete. Successful pressure checks do not set the session's final-drained state; frame-list emptiness is not evidence that external tails are complete. Host container growth and cached-accounting failure follow the same all-stream cleanup contract as a failed launch.

The destructor is `noexcept` and performs best-effort draining in its **body**, before members containing frames, kernels, uploads, or outputs are destroyed. It does not pretend to report successful completion. A fatal/device-context CUDA error does not establish safe recovery; surface it from the explicit path and do not promise an ordinary fallback or reusable pool. Cleanup still attempts all lanes and preserves the original exception. The design does not claim a CPU destructor can repair a lost CUDA context.

Any boundary temporarily holding an already-pending owner before adoption needs failure-safe local cleanup: a session catch alone is too late after leaf-local unwinding. Keep the existing adoption helpers' exceptional-path drains; they do not justify normal-path completion-only leaf waits or a second ownership mode.

## 5. Leaf barriers: what stays and what goes

| Path | Observation that remains | Ownership change / wait removed in the final design |
|---|---|---|
| Plain/delta/RLE/fused metadata | Any genuine host-discovered shape already required by a nonfused tail. Ordinary chunk metadata preparation is same-stream device work. | Retain synthesized offsets, scan scratch, and decoded metadata columns in the frame; pin distinct kernel handles in the session until the final stream drain. Remove `run_rendered_decode` and walk terminal waits. RLE placeholder bindings are not evidence of allocated RLE scratch; account actual owners. |
| Identity, including nullable fixed width | None merely to copy an already-known data/null-mask size. | Use the [owning-column copy](../src/compression/simpatico_codegen/src/plan/representation_factory.cpp#L953) on the frame stream with supplied MR; retain input borrow and output. Nullable identity is not a reason to take a legacy column path. |
| nvCOMP, including ANS/LZ4 | [Header and compressed-size-table D2H observations](../src/compression/simpatico_codegen/src/operators/nvcomp_batched_codec.cu#L201) establish host dimensions/pointers/scratch sizing. | Frame-owned device metadata/temp and **host** compressed sizes/pointers/uncompressed sizes/pointers cover the [H2D launch metadata](../src/compression/simpatico_codegen/src/operators/nvcomp_batched_codec.cu#L250); no completion-only decode tail wait remains. Six codecs share this helper. Opaque synchronization inside nvCOMP is not promised away. |
| ALP / ALP_RD / bitextract / bitjoin | ALP's one-time constants must be ready before shared initialization is published; ALP_RD's reconstructed `right_bw` D2H is a real observation. | Adopt reconstructed reps before invoking decode; retain bitjoin's packed column before field launches. [ALP](../src/compression/simpatico_codegen/src/operators/alp_compressor.cu#L682), [ALP_RD](../src/compression/simpatico_codegen/src/operators/alp_rd_compressor.cu#L541), and [bitextract](../src/compression/simpatico_codegen/src/operators/bitjoin_bitextract.cu#L249) use the mandatory frame without completion-only decode tail waits. Check initialization/copy statuses before publication. |
| Dictionary / predicate | Width/all-equal inspection and host null counts remain where the format requires them. | Retain padded keys, scalar needles, hit masks, LUTs, decoded generic predicate input, and reconstructed dictionary ownership. A generic predicate still decodes and compares; unification is not new predicate fusion. |
| `str_split`, nullable strings | Host null count, or [compacted `total_chars`](../src/compression/simpatico_codegen/src/plan/decompress.cpp#L1039), must be established before constructing exact-sized output. | Reuse the same frame resolver for offsets/entropy tails; retain lengths, offsets, source chars, scan scratch and char-copy kernel. No final char-copy/wrapper completion wait is needed solely for ownership. |
| Mask/count/index/CSR and full-route gather | Survivor count at [`selection_wave.cu:262`](../src/compression/simpatico_codegen/src/selection/selection_wave.cu#L262) fixes output sizes and policy. | Retain shared selection state at phase scope. Use events for index-producer/consumer streams. Retain full decoded input until gather completes; remove gather-only completion waits. Existing route/null/predicate restrictions remain. |

Leaf host readbacks use checked, narrowly named helpers, for example `read_scalar<T>(device_ptr)` constrained to trivially copyable `T`, or an explicit sized metadata read. Their destination storage remains valid during failure cleanup. This is an appropriate small C++20 template/concept use; a generic callback executor is not.

Dictionary representations have [mutable lazy channel exports](../src/compression/simpatico_codegen/include/codegen/plan/representation.hpp#L285) that allocate with the current MR and publish after waits. A decode session must not asynchronously mutate shared compressed input or leave session-resource allocations cached in that input. For decoding, synthesize required empty/key-char channel views into **frame-owned** storage using the supplied MR, or borrow already-published immutable channels. Keep encode/export behavior separate. This also makes duplicate requests over the same plan safe without relying on a mutex plus a hidden initialization wait. Broader representation-cache redesign is not required.

Dictionary key width is observed once before publishing an encoded or `from_outputs` representation, using its explicit stream and resource. The existing host field records a positive uniform byte width or zero for variable/empty keys; its layout and the serialized payload do not change. Frame-local reconstructed dictionaries may retain the unknown value (-1) and use the existing completed width observation without mutating borrowed input. This publication-time metadata avoids repeated width kernels/readbacks during warm decode; it does not add a lazy cache protocol or a decode execution mode.

## 6. One walker, including predicates and selections

The structural `(node, port)` memo remains the single source of reconstructed values. Preserve consumer counts and last-consumer moves. Frame ownership extends lifetime after a memo value has been consumed; it must not cause duplicate materialization or premature destruction of a source still read by queued work.

Root semantics are explicit. A predicate applies only to the producer of the final input value; a selected dictionary applies row selection to its indices region, not dictionary keys or offsets. Existing [`predicate_applies_to` and `selection_applies_to`](../src/compression/simpatico_codegen/src/plan/decompress.cpp#L767) encode the right distinction. Keep that targeting once in the walker. Never forward a root selection blindly to an entropy-tail or metadata node just because every function now takes a frame.

Merge the special dictionary/string helper's *tail resolution* into that resolver. A specialized root emitter can still exist, but it receives resolved bindings from the same frame/walker; it must not instantiate a second `DecodeWalk` with a subtly different selection/lifetime policy.

Preserve current result contracts:

- Predicate requests produce BOOL8, including generic decode-plus-compare and dictionary key predicates.
- Predicate plus selection composes only on the existing `dict_codes` and `full` routes; rejected write-skipping combinations remain rejected before submission.
- Compacted routes produce exactly `survivor_count` rows with the existing null restrictions. Null-masked selection does not become supported incidentally.
- A chunk row set remains restricted to its currently supported compacted route. Mask, flat index, and CSR enumeration remain semantic choices, not execution modes.
- `full` means decode plus gather. It can retain the existing batched full-route gather optimization at phase scope; it is not a route to the old completed column executor. Its sources remain owned through gather completion.
- A specialized string route that cannot implement the selected semantics reports an explicit semantic decline/error according to current policy. It never silently returns full-width values into survivor-sized allocations.

There are two explicit adapters, not two decode mechanisms. A **direct selected-column request** includes selection; its `full` route performs decode plus gather in that frame. A **table grouped-gather candidate** submits an ordinary *unselected* column request (retaining any predicate substitution); after finish the orchestrator gathers its full-sized result with the other candidates. Compact-in-kernel candidates submit selected requests and never enter that grouped gather. Do not submit a selected `full` request and then gather its already-compacted result again.

The scan orchestrator owns those completed full columns, enqueues its existing grouped gather under a one-phase ownership scope, and completes it before publishing. Full-width BOOL8 results already produced in wave 1 join that gather directly: they are **not decoded again in wave 2**. The phase restores results to requested positions, preserving BOOL8 and duplicates. This is a genuine cross-column operation, not a second decoder. Avoid replacing a single grouped cuDF gather with many per-column gathers solely to make the class diagram uniform; test this boundary explicitly.

### Existing membership probes are a typed mask source

Preserve [`membership_filter_directive`](../src/compression/simpatico_codegen/include/codegen/selection/selection.hpp#L151) and the [current key-decode → probe → ballot behavior](../src/compression/simpatico_codegen/src/plan/decompress.cpp#L1387). The membership alternative of `mask_decode_request` runs the ordinary full-key resolver in its frame, then invokes that existing pinned probe closure on the **same stream**. A same-physical-width logical-type *view* supplies the stored key type where appropriate; do not move/rebind owning output buffers early. This internal view preparation is separate from final public-output retagging. A mismatched carrier continues to reach the probe's existing applicability check rather than being silently cast.

Keep keys, returned flags, and the copied probe capture alive in the frame through ballot completion. The capture must pin an immutable filter snapshot for the session, as the existing contract requires. Preserve the callback's narrow contract: it enqueues all work on the supplied stream, takes a ready-in-stream key view, uses the supplied MR, and is responsible for its own internal temporary safety. Audit its existing implementations against that contract; do not generalize it into arbitrary session callbacks or a new task framework.

`nullptr` means the membership probe declines this key carrier: enqueue the all-ones AND-identity mask and return `mask_source_status::declined`. The frame still owns any queued dependencies. A non-null result of the wrong dtype, row count, or null policy is an error, not a decline; exceptions propagate unchanged. The orchestrator counts declined sources immediately from this semantic status. If **every real source** declined, finish/drain wave 1 and take the explicit no-filter policy result before CNT: an all-ones padded mask must not be counted as valid tail-cleared selection. If real sources remain, their ballots establish the proper tail and the authoritative downstream join still enforces the full membership condition. Preserve source caps, ordering, and keep-mask policy.

## 7. Bounded retirement without per-column synchronization

### Policy

Use a small private **retained-state window**, shared across the session's lanes. Do not reserve exactly one frame per lane: eight columns/four streams would add four lane-reuse waits before the four final joins, and one stream would become per-column blocking again.

Keep named internal soft limits of **64 MiB estimated retained temporary device bytes**, **8 MiB host upload bytes**, and **64 registered frames including the active frame**, with at most one active host walk. These remain provisional engineering constants, **not evidence-backed optima**, public tuning knobs, or a capacity guarantee. Their rationale is to bound queueing overhead independently of final table size and GPU capacity: modest device metadata, smaller CPU upload retention, and a finite host object count. Do not retune them as part of removing events merely to improve a benchmark. A large single column or a required shared selection structure can intrinsically exceed them. The implementation must make pressure waits and estimated retained high water observable in focused diagnostics; do not silently disable bounds or accept reintroduced baseline serialization.

### What is counted

The device ledger is a mixed retained-byte estimate: exposed `rmm::device_buffer::capacity()` for direct scratch, cuDF [`column::alloc_size()`](../.pixi/envs/default/include/cudf/column/column.hpp#L272) for temporary/memoized columns, `compressed_representation::owned_device_bytes_estimate()` for reconstructed owners, and the existing supplied scalar estimates. The installed cuDF implementation sums data/null buffer sizes recursively, including padding within those sizes but excluding retained capacity beyond them. The representation query observes every unique owner without enumerating borrowed channel views, allocating, launching CUDA work, querying/synchronizing streams or mutating lazy storage. This estimate excludes hidden column slack and allocator/pool overhead; it is neither wire size nor exact allocation capacity. Count host-upload block capacities separately, plus frame count for bounded host bookkeeping. JIT module memory remains governed by the existing kernel cache and distinct session pins; do not invent an estimate of driver module allocation or charge module handles as device scratch.

Do **not** count borrowed compressed input or terminal output storage as queued temporary overhead. If a decoded full column will feed a predicate or gather, it is an intermediate and *is* counted until that use completes. Shared phase masks/indices are accounted as required phase state separately, not falsely attributed to every consuming frame. Views and aliases do not add capacity; ownership transfers move an accounting entry rather than creating another. A representation that takes a memo column owns that same allocation, not a second copy. Frame-local lazy channels count; borrowed, already-published representation channels do not.

This is not another memory resource. Maintain accounting at the small set of ownership/adoption sites; do not intercept every allocation globally. Library-internal peak scratch exists in addition to retained-state accounting and must be measured with an instrumented MR. Document unavailable allocation information rather than treating this estimate as exact. Measuring hidden cuDF capacity would require an upstream observation API or broader owner-side tracking, neither of which belongs in this policy-preserving follow-up.

### Selected mechanism: pressure-only stream-tail reclamation

1. At admission and retained-allocation checkpoints, take one retained-state snapshot. Sealed frames contribute cached immutable byte totals; only the active frame needs live ownership inspection. If the snapshot plus the prospective allocation/frame remains within all limits, return immediately: **no CUDA retirement query, wait, create, record or destroy**. Stream reuse does not itself retire or synchronize anything.
2. Under pressure, visit each distinct physical stream handle with at least one sealed frame. Query its current tail. Success proves the queued uses of every sealed frame on that handle complete; erase all those sealed frames on the caller thread. `cudaErrorNotReady` is ordinary pending work. Any other error propagates into the session's all-stream abort path without erasing that handle's unproven owners.
3. If ready-stream reclamation leaves pressure, select the globally oldest sealed frame, synchronize its **physical stream's current tail**, then erase all sealed frames on that handle. Reevaluate the retained snapshot and other ready streams before waiting again. The single active frame is never erased, including when it shares the stream just completed. Stream assignment stays round-robin; this does not reorder submissions or schedule new work.
4. If only the active frame remains and its intrinsic working set exceeds a byte limit, allow that one necessary working set. Its later allocations still check pressure, but must not wait on or retire the active frame itself. Once it seals, the next append must relieve pressure before adding a new frame. Do not bypass the finite frame cap or retry an OOM column.
5. Append seals with no CUDA call after all request work is submitted and terminal output ownership has moved to the private result slot. No later work may borrow that frame's temporary owners. `sealed` means ownership is fixed, **not** that the GPU is complete; only a successful observation on its stream licenses retirement.
6. Finish, abort and destruction retain the distinct-all-supplied-stream drain contract in §4, regardless of whether pressure already removed every frame. Kernel pins are not frame-retired. A failed query/wait takes that same failure path; no event-failure or event-destruction branch remains.

Preserve the original copied stream list and its round-robin weighting. Duplicate views of one raw `cudaStream_t` are one completion domain: a query/synchronization success reclaims every sealed frame assigned to any alias, never an active frame. A small scan of earlier supplied handles skips duplicate query/drain calls without an extra map, canonical lane abstraction, allocation on the pressure path, or new persistent ready flags. Frame count is capped; the pressure-only search is intentionally simple. Do not cache stream-completion observations across later submissions or asynchronous deallocations.

The existing private `pressure`, `retire_ready`, and drain helpers are sufficient. Replace the event-specific diagnostic with `retirement_stream_queries`, counting only pressure-time queries, and retain `pressure_waits` for actual blocking pressure synchronizations. Final drain queries/waits are separately observable in Nsight. Keep the same `append`/`finish` API; leaves need no event, lane, budget or completion methods. There is no event pool, reclamation stream, callback, thread, new cuCascade dependency, or second mode.

```text
pressure(prospective bytes / frame):
  snapshot cached sealed totals + live active totals
  if within limits: return
  query distinct streams that own sealed frames; erase sealed frames only on success
  while still over limits and a sealed frame exists:
    wait for the oldest sealed frame's stream tail
    erase every sealed frame on that physical stream; never erase the active frame
    recompute pressure; query other eligible streams only if still over limits
  allow only the intrinsic active-frame byte overage, not another queued frame
```

The accounting model is **required input/output and phase state + estimated retained window + one intrinsic active-column working set**; hidden column capacity slack and opaque library working sets are additional. A finite frame count does not bound hidden slack, so this is not an exact ownership bound or a claim that decode uses at most 64 MiB. One oversized completed-host frame may remain pending; the next append retires it before admitting another. Allocation padding/prospective-size uncertainty also makes the byte limits soft. A 32 GiB final table is not charged against the scratch window, whereas a full multi-GiB decoded column used only as predicate/gather input is an intermediate and contributes its estimate. Required phase-owned grouped-gather sources are reported separately rather than hidden by changing ownership labels.

### Deliberately weaker reclamation progress

Let A's last GPU use precede B on the same stream. A frame event recorded between them could prove A complete while B remained pending. Stream-tail observation cannot: if B is pending, neither A nor another sealed frame on that handle is reclaimed by a query. Under pressure the caller waits for B too; without pressure both remain owned until a later pressure check or finish. This loss of prefix-before-later-work retirement is explicitly authorized, not an accidental weakening of a test. It trades reclamation precision for removing per-request CUDA retirement lifecycle costs.

**Caller progress contract:** every dependency already queued on a supplied stream must be satisfiable independently of a future append, finish, or return on the same submitting CPU thread. A caller must not queue a gate whose release requires that caller to append later work after a potentially pressured append. An independently progressing producer stream/thread is allowed. This is a documented behavioral constraint, not a generic claim that arbitrary external stream work is harmless.

The current call sites satisfy that contract: [ordinary table wrappers](../src/compression/simpatico_codegen/src/simpatico_codegen.cpp#L275) and direct/standalone wrappers append then finish; [scan wave 1](../src/compression/simpatico_codegen/src/simpatico_codegen.cpp#L569) submits producers before adding joins/count, while [wave 2](../src/compression/simpatico_codegen/src/simpatico_codegen.cpp#L769) queues its already-recorded index-producer dependency before appending consumers. Membership probes must queue their work on the supplied stream without depending on future session submissions. [Empty wave 2](../src/compression/simpatico_codegen/src/simpatico_codegen.cpp#L809) separately completes its external phase work. Keep selection dependency events; removing them would remove required ordering, not retirement overhead.

### Ownership retirement is not immediate physical reuse

After observing a stream tail complete, sealed frames' host uploads can be released safely and their device-buffer owners destroyed on the caller thread. Kernel pins remain session-owned until the final drain: prior cache-clear testing established that dropping a last module handle can implicitly wait for unrelated queued work. Removing retirement events does not remove that driver behavior. With the installed async MR, destroying a buffer queues `cudaFreeAsync` on its stored stream at the current tail; the free is not necessarily executed when the host destructor returns. The [installed async resource](../.pixi/envs/default/include/rmm/mr/cuda_async_view_memory_resource.hpp#L101) and reservation adaptor therefore separate three facts: a frame no longer owns bytes, host reservation accounting was updated, and those bytes are actually reusable by another stream.

Later allocations on the same stream can use the ordered free; cross-stream availability and physical high-water improvement depend on CUDA pool reuse and ordering. The retained-owner ledger is **not a physical-memory cap** and does not grant more reservation capacity. Report owner-retained bytes, adaptor accounting, and resource/pool high water separately, including delayed stream-ordered frees and library-internal scratch. Do not infer physical reclamation from a completed prior tail or ledger decrement. Introducing a separate reclamation stream/deallocation-stream rebinding would require new dependencies and allocator analysis and is explicitly out of scope.

For the eight-column numeric/identity workload, there should be no memory-pressure wait or retirement query if measured retained metadata stays under the window. Expected final completion checks cover **four distinct streams with a four-stream pool, one with one stream**, with at most that many blocking completion calls when work remains, plus the unchanged four/one extra harness waits. There are zero retirement event creates/records/destroys. Real host-readback barriers and phase-join events are separate. Verify these counts in profiles rather than claiming them from the class sketch.

Tradeoff: stream tails provide coarser reclamation and can increase pressure-wait duration or retained high water on skewed/wide workloads. Under-limit submission pays no CUDA retirement calls, but ordinary host bookkeeping still costs time; removing events alone is not a promised small-batch speedup. Finite-memory backpressure remains, and the executor is not universally nonblocking. Correctness and measured no-regression gates still apply.

## 8. Representative flows

These sketches omit validation/error text but show ownership and ordering. CUDA/library calls use checked status handling. They are not implementation patches.

### Ordinary or projected table

```cpp
validate_requested_columns(table, selected); // all indices first; duplicates allowed
decode_session session{pool.streams(), mr};
for (auto index : selected) {
  session.append(column_decode_request{
      .source = std::cref(table.columns[index].plan()),
      .result = value_result{table.columns[index].dtype}});
}
auto columns = session.finish();             // completion, validation, retag, transfer
return std::make_unique<cudf::table>(std::move(columns));
```

The stream-view overload supplies a one-element span and follows exactly this flow, as does a direct single-column wrapper. The pool/thread-count facade only supplies/leases streams; it does not select a decoder implementation.

### Direct single-column, synchronous to its caller

```cpp
decode_session session{one_stream(stream), mr};
session.append(column_request(tree, pred, selection));
auto columns = session.finish();
return std::move(columns.front());
```

There is no `sync=true`; completing the scope supplies the synchronous contract. Direct late-materialization entry points keep their completed semantics and share this adapter.

### Predicate/compaction with real host readback

```text
Preflight existing probe/policy, including supported predicate/selection combinations.
Own shared masks/count/indices and an exception-safe phase cleanup scope.
Create a wave-1 session on the supplied stream pool.
  append numeric ballot requests into owned mask destinations;
  append dictionary/generic BOOL8 requests with optional ballot + retained BOOL8 delivery.
  append membership mask requests: same-frame full keys -> external probe -> ballot;
    count accepted/declined semantic returns, retaining keys, flags and probe captures.
If every real source declined, finish wave 1 and take plain-decode policy before CNT.
Record a producer-done event on each supplied stream after the appends.
Make s0 wait on the other streams' events; no pending column view is exposed.
Queue combine + count on s0 after those dependencies; mask destinations are known.
Read survivor_count to host and complete that observation (required shape/policy barrier).
wave1.finish() performs the final checked stream drains; already joined lanes are complete,
  so these are completion checks without additional outstanding producer work.
  -> obtain completed BOOL8 columns for slots that need dual delivery.
If policy declines, release completed phase data and invoke ordinary requests.
Otherwise allocate exact-sized indices, enqueue mask->indices on s0, record ready event.
Queue a wait for that event on each other consuming stream, as today.
Create wave-2 session for slots not already supplied by wave-1 BOOL8 results.
  append selected requests for compact-in-kernel routes;
  append unselected requests for grouped-full-route candidates (preserving predicates).
wave2.finish(); gather the full-route candidates and retained wave-1 BOOL8 owners
  under the phase's grouped-gather ownership/cleanup scope; complete that gather.
Merge compacted results into requested positions; do not re-decode BOOL8 slots.
Validate survivor-sized outputs, preserve BOOL8 slots, then publish the table.
```

Selection orchestration retains masks, producer/join events, index storage, and dual-delivery inputs until their consumers finish. It remains a domain-specific phase algorithm outside the column walker, not a generic callback scheduled through the session. Recording events on all supplied streams avoids exposing the session's request-to-lane assignment. Enqueuing combine/count on s0 while wave 1 is open is intentional and ordered through those events; it does not require exposing a pending column view. The session's lane-completion check includes that stream tail. Preserve producer fanout: do not serialize wave 1 onto s0 or add an early host join merely to simplify ownership.

Post-join CSR row sets enter through the same validated selection request. Their caller establishes readiness and lifetime; no scan-mask production is invented for them.

### Failure after earlier columns were submitted

```text
A and B registered, work queued; C registered and partially materialized.
C allocation throws cucascade_out_of_memory.
No owner escapes; frame C already owns every published GPU dependency.
Session becomes failed, records original exception, attempts every distinct supplied stream.
Only after successful lane completion may its frames be retired.
Rethrow the original cucascade_out_of_memory, with requested_bytes intact.
Session destructor performs remaining best-effort cleanup on the same CPU thread.
Outer pipeline reservation attachment is still active during this cleanup.
Engine's existing OOM handler decides rescheduling; decoder does not retry/fallback.
```

If a drain itself reports a fatal CUDA error, retain the original failure priority and report diagnostic context; no successful quiescence or subsequent pool reuse is claimed. The normal recoverable OOM tests cover successful drainage, not recovery from a lost device context.

## 9. Implementation handoff and module boundaries

The following unified boundaries are already installed and must remain. The event-free patch changes the session's private retirement implementation, diagnostics, completion-contract comments and focused tests; the measured storage amendments change only frame member containers and direct composition within the existing list node. The final leaf correction restores one owning-column copy expression without changing that leaf's interface or dispatch. None repeats the original cross-codec migration.

| Area | Change | Remove / merge |
|---|---|---|
| `src/plan/decompress.cpp`, private `src/decode/decode_session.hpp/.cpp` | Keep session/frame, mandatory walker ownership, result validation and root targeting. | Do not restore `can_enqueue_column_decode`, `pending_column_decode`, external `mark_completed`, optional `deferred` state, terminal walk waits or second tail walkers. |
| Former `src/decode/deferred_launch.hpp`, `src/plan/pending_decode.hpp` | Keep the one private decode-context/session boundary. | These additive headers stay removed; no forwarding aliases with two contracts. |
| `src/bridge/codegen_runtime.cpp`, decode launch declarations / `masked_launch.hpp` | One internal throwing launch entry over existing render/variant descriptors, mandatory frame, explicit MR. Plain/masked/index/CSR/string requests prepare typed launch arguments and use that entry. | `enqueue_decode_fused_tree` versus completed-launch duplication; local-vs-retained scratch branch; decode-only `check_and_launch` wrapper ladders; swallowed decode exceptions; completion-only waits. Preserve distinct renderer algorithms and encode behavior. |
| `include/codegen/plan/representation.hpp`, `src/plan/representation_factory.cpp`, decode portions of operator `.cu` files | Propagate mandatory decode context; adopt reps before decode; retain codec host/device dependencies; isolate decode-local lazy channels; keep true observation barriers. | Parallel sync/async leaf interfaces, terminal ownership waits, decode allocation through hidden current MR. Existing standalone public/testing compatibility is one completed-session facade. |
| `src/simpatico_codegen.cpp` | All table/selected/predicate facades build requests and finish one session; preserve scan-filter phases and existing selection policy. Distinguish explicit decline from failure. | `decompress_columns_submitted` and separate legacy decode loops merge into one driver. Decode no longer uses `run_column_workers`; leave it for encoding if encoding still needs it. Remove broad exception-to-ordinary-decode fallback. |
| `include/codegen/plan/plan_interpreter.hpp`, `include/api/simpatico_codegen.hpp` | Document completion/input readiness once; validate legacy selection descriptors into the private semantic value. Keep completed APIs and current row-route meaning. | Stale worker-thread language, duplicate lifetime prose, any public pending-state protocol. Keep `probe_column`: semantic capabilities are not async eligibility. |
| JIT cache | Keep the existing shared owning kernel handle; sessions deduplicate and pin it before launch through the final stream drain. Cache clear removes cache ownership, not an in-flight module. | No new kernel-lifetime wrapper, no raw-pointer escape that defeats shared ownership, and no last-handle module unload during pressure retirement. Encoding call sites retain their current correct handle behavior. |
| Private session retirement state | Host-only active/sealed state, cached sealed byte totals, caller-thread pressure-only stream-tail reclamation. Final and failed paths retain all-supplied-stream drains. | Remove frame event handles, recorded state, create/record/destroy branches, event fault paths and `event_queries`. No cuCascade wrapper dependency or new execution mode. |
| CMake / tests | Replace retirement-event link wrappers/fault tests with stream query/synchronize tests; cover duplicate physical handles and the deliberately weaker tail-progress contract. | Do not remove selection-phase events or test-only GPU markers used to observe correctness. |

Within the new decode machinery, `shared_ptr` is needed at the kernel-cache boundary because cache entries and independent sessions genuinely share module ownership and cache clearing may happen while work is in flight. Existing membership closures also retain their already-required pinned filter ownership; that is not a new generic retention mechanism. Frames are direct node members; memo values, reconstructed reps and outputs retain their existing unique ownership. Borrowed compressed input does not become shared-owned just to avoid documenting its lifetime. Session stability is provided by nonmovability and stable list nodes, not pervasive shared ownership or redundant heap indirection.

### Stable interfaces: no new caller protocol

Keep `src/decode/decode_session.hpp` as the existing private session/frame/slot contract. Its request types and `append`/`finish` signatures do not change. The diagnostic field is renamed `event_queries -> retirement_stream_queries`; the class comment adds the independently-progressing external-tail contract. None of these types is an engine asynchronous API. Retirement and pressure remain private; leaves cannot query or complete frames.

```cpp
// A copyable, non-owning handle to stable frame/memo unique_ptr storage.
// Only decode_frame/core may construct it. An empty slot is single-assignment.
class decode_column_slot {
 public:
  void adopt(std::unique_ptr<cudf::column> column) const noexcept;
  cudf::column& get() const;                 // precondition: already adopted
  cudf::column_view view() const;            // same precondition
};

class decode_frame {
 public:
  rmm::cuda_stream_view stream() const noexcept;
  rmm::device_async_resource_ref mr() const noexcept;
  decode_column_slot make_column();         // registered/stable before any enqueue
  rmm::device_buffer& allocate_buffer(std::size_t bytes);
  compressed_representation& keep_representation(
      std::unique_ptr<compressed_representation> representation);
  cudf::scalar& keep_scalar(std::unique_ptr<cudf::scalar> scalar,
                           std::size_t device_bytes);
  void keep_kernel(std::shared_ptr<codegen::jit::CompiledKernel const> kernel);
  template<class T> requires std::is_trivially_copyable_v<T>
  std::span<T> host_array(std::size_t count);
  template<class T> requires std::is_trivially_copyable_v<T>
  T read_scalar(T const* device_source);     // checked, genuine host observation
  void read_bytes(void* destination, void const* device_source, std::size_t bytes);
};

// The one virtual leaf ABI; no parallel synchronous/asynchronous virtuals.
virtual void decompress(decode_frame& frame, decode_column_slot output) const = 0;

// A nonvirtual completed compatibility wrapper on the standalone base.
std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) const;
```

`host_array<T>` must provide stable, correctly aligned owned storage; do not assume a resizable byte vector is a stable/aligned typed array. Uploads retain that storage through observed completion of their frame's stream. `read_bytes` may borrow a synchronous destination only if its exceptional path completes any queued copy before the destination can unwind; otherwise use frame-owned readback storage. `allocate_buffer` and slot/retention helpers perform the private prospective-size pressure check without exposing pressure controls to leaves. Returned buffer references must remain stable when more scratch is added.

Create all memo/output slots before invoking a leaf or writing a result. Slot adoption moves into an empty owner without allocation, pressure handling, byte estimation, or replacement; never reset a slot holding a pending source. Refresh the retained-byte estimate after adoption at the next checked frame accounting point before another allocation/admission, with the owner already installed if that check throws. Session-created terminal result slots and temporary slots have their accounting role established before their first write; a multi-GiB output must not temporarily look like scratch and trigger a false pressure wait. Sirius kernels adopt their destination before their first write. A cuDF factory returning an already-pending `unique_ptr` is adopted immediately into a pre-registered slot; the library's internal exception/lifetime guarantee remains a separate obligation. Public wrappers ultimately extract owners only after `finish()`.

The structural memo can map `(node, port)` keys to stable slot handles, or expose stable frame-owned map entries through those handles; retain one memo, not a parallel cache. Last-consumer transfers into reconstructed reps must enter prepared ownership without a throwing gap. Reconstructed reps themselves enter frame ownership before their decoder is called. Audit constructors/factories that consume owning channels: a session catch cannot save a channel already destroyed inside a failed constructor. Stage potentially throwing host setup before moves, keep channels frame-owned while preparing a rep, or use a proven library ownership/stream contract; do not paper over this with another whole-column wait.

The completed standalone wrapper creates a one-stream session with a standalone-source value request and calls `finish()`. Derived classes may use `using standalone_compressed_representation::decompress` to expose that wrapper without duplicate implementations. Direct lower-level test adapters either use this wrapper or an internal completed-session adapter, never a nullable frame.

The runtime retains existing typed renderer/selection arguments and the `void`/throwing mandatory-`decode_frame&` decode contract: no stream/MR parameters competing with the frame, no `error_out` branch, no nullable workspace. Plain, mask, compacted, dictionary and string argument builders converge on **one** scratch/compile/launch implementation. The caller supplies a frame/phase-owned destination before calling them. Keep encoding entry signatures and policy unchanged, including the already-installed eager dictionary-width metadata publication.

### Exclusive file partitions and integration order

| Owner | Exclusive edit scope | Dependency / completion condition |
|---|---|---|
| Architect | This design document only | Freeze the retirement/progress/error contract; no production or test changes. |
| Lead developer | `src/decode/decode_session.cpp/.hpp`; stale frame-marker wording in `src/simpatico_codegen.cpp`; the single identity copy expression in `src/plan/representation_factory.cpp` | Remove event lifecycle, implement duplicate-safe sealed-frame stream-tail reclamation, preserve cached accounting and exception priority. For the measured storage amendments, change the four named containers, then embed the nonmoving frame in its list node with direct construction; preserve helper contracts and ordering. Restore the original owning-copy overload only; no other leaf/renderer/selection-algorithm changes. |
| Optional test worker | `tests/test_async_decode.cpp` and its fault-wrapper link options in CMake, by explicit assignment | Consume the frozen diagnostic rename and contracts; replace obsolete event fault/prefix assumptions, retain gates and independent completion observations. Add retained slot/reference/upload-span growth tests for the storage amendment. Coordinate CMake ownership before editing. |
| Root, then independent reviewer | Fresh artifact directory and benchmark/stat adapters; read-only implementation review after developer handoff | Root controls builds/GPU/provenance. Reviewer checks lifecycle/error invariants, one-path removals and measured gates independently. |

Stop and communicate a scope/interface change before crossing those boundaries. Root preserves both the additive comparison binary and the pre-change unified candidate with hashes; historical evidence is not rewritten. Preserve the user's unrelated dirty files and leave all changes uncommitted/unpushed.

Implementation order is this architecture handoff, bounded session/test patch, removal and caller-contract audit, correctness/sanitizer and paired benchmarks, then independent review. The container amendment follows the same cycle after preserving the measured stream-tail candidate. Any necessary scope expansion or failed performance gate returns to the root explicitly. Do not add empty-frame fast paths, alternate ownership modes, a pool, or new allocator behavior preemptively; first measure the selected storage substitution.

### Single-path end state

There is no staged second decoder for this patch. Leave the already-unified leaf ownership and necessary host observations intact; replace the private retirement mechanism directly. No codec becomes magically nonblocking because frame events disappear.

The final removal checklist is mechanical:

- No decode `deferred != nullptr` / null workspace test, async-mode boolean, or whole-plan `can_enqueue` registry.
- No `pending_column_decode`, `mark_completed`, or output extraction before session completion.
- One table request driver and one structural resolver, including predicate and selected-column requests.
- No Sirius-owned completion-only leaf wait; every remaining explicit wait is annotated as a named host observation, shared initialization publication, memory backpressure, or scope completion.
- No decode launch whose failure changes from exception to bool depending on caller mode.
- No broad OOM/device/compile-error-to-policy-decline catch.
- No GPU-read host/device/module owner destroyed before its proven completion; no hidden current-MR allocation on an audited supplied-MR path.
- No per-frame retirement event/recorded flag/create/record/destroy branch, event-specific diagnostic, or retirement-event fault wrapper. Keep selection dependency events and independent test markers.
- No active-frame reclamation, scratch borrowed after sealing, duplicate-handle false independence, stale completion-proof reuse, or skipped final/abort drain because the frame list is empty.
- No CUDA retirement API under all three limits; no unlimited frame admission or oversized-frame bypass for later requests.
- No CPU callback/worker, generic scheduler, event pool, reclamation stream, or new standalone cuCascade dependency introduced for retirement.
- No pending leaf output returned through an unregistered local owner; slot registration precedes enqueue and adoption/reparenting has no throwing ownership gap.
- No vector-backed owner slots or buffer objects whose addresses escape across growth; upload-vector records may move only because their byte-array pointees remain stable. No blanket frame-container reservation that recreates eager empty allocation.
- No separately allocated frame behind an already-stable submitted-node owner, and no move/copy of a registered frame. Node construction remains host-only and precedes the first enqueue.

## 10. Validation and acceptance

**Executed status:** v5 passed all 21 standalone CTest entries, both unsuppressed memcheck suites with zero errors/leaks, 29 engine compression cases / 1,224 assertions, the reservation case / 23 assertions, and three OOM/retry cases / 17 assertions. The primary performance campaign contains 182 processes / 4,130 hot samples; the controlled and baseline ANS comparisons add 20 / 620, the fresh full small-matrix confirmation 112 / 11,312, and the large-key confirmation 10 / 310. The [validation report and provenance](../../sirius-simpatico-eval-20260914/64-stream-retirement-wSVCu64J/REPORT.md) tie these to v5 binaries, and independent review gives scoped approval with the identity-cost limitation in §1. Single-device success is not multi-GPU validation, microbenchmarks do not establish TPC-H speedup, and the unchanged private window constants remain provisional. The contracts and acceptance criteria below remain regression requirements, not a list of unperformed v5 work.

### Correctness and ownership

Extend the existing suites rather than building a new broad harness. Run the current standalone targets plus the existing fused and operator-chain sweeps. Cover plain bitpack, delta, RLE, raw/identity, nullable fixed-width identity, nullable/string reconstruction, dictionary, ALP/ALP_RD, bitjoin/bitextract, ANS and LZ4; include empty/tail/extreme-width cases already covered by selection/layout tests.

Add focused session tests for:

- Completed direct, one-stream-table, pooled-table and selected-table return; reordered and duplicate columns, empty selection and invalid indices.
- Predicate BOOL8 and dual delivery; dictionary and generic predicate behavior; supported and refused predicate+selection combinations; mask/index/CSR targeting that leaves inner metadata unselected; compacted strings and dictionary gathers; full-route grouped gather compatibility, without double gathering or re-decoding BOOL8 slots. Verify wave-1 fanout and event-ordered combine/count, with no newly introduced pre-count host join. Membership tests cover accepted probes, mixed/all-declined sources, keep-mask/tail handling, malformed flags, thrown failures, pinned-filter lifetime, and supplied-stream/MR use.
- A stream held behind a deterministic gate while multiple known-size requests are submitted. Assert later requests are enqueued before releasing the gate, without asserting fragile elapsed-time speedups. Release the gate safely on every failure path.
- Replace the former event-prefix test with the approved stream-tail contract: A completes, then a gate holds later work on that stream. Forced pressure cannot reclaim A before that tail completes; an independently controlled release lets the pressure wait finish, after which all sealed peers may retire. Use a bounded watchdog and deterministic synchronization observations, not a latency threshold. Do not make gate release depend on return from the pressured append.
- Query-only reclamation of an already-complete stream must avoid a pressure wait. A second case keeps one lane blocked but another complete; retire the ready lane before choosing a blocking oldest-lane wait. Duplicate stream views, including a repeated handle in the round-robin list, must be queried/drained once per sweep and reclaim all sealed aliases together. Include an active frame sharing that handle and prove it survives successful observation and cohort erasure.
- Inject retirement stream-query/synchronize failures and final-drain errors. Verify every supplied physical stream is attempted, earlier submission/OOM exceptions retain priority, unproven owners remain until cleanup, and no partial results escape. Queue external phase work after request submission, including on a supplied stream with no remaining frame, and prove `finish()` still checks it. Remove obsolete event-create/record/destroy failure cases; retained test markers are observation tools, not production retirement objects.
- Kernel cache clear after submission, dropped/failed sessions, and reuse after a recoverable injected allocation failure. Inject failure at frame registration, memo growth, rep adoption, host upload growth, leaf allocations, and final assembly; verify no partial result publication and preserved OOM subtype/requested bytes.
- Retain the first column slot, scratch-buffer reference, output-buffer reference and typed upload span while adding many later owners/blocks to the same active frame. Force upload-vector growth, verify original addresses and contents, then enqueue reads/writes/copies through those original handles and verify completed results. Retain verification destinations outside the frame through `finish()`; do not dereference retired frame storage afterward. Exercise supplied MR/caller-thread cleanup and the existing adoption-failure safeguards; no broad host-allocation interception framework is required.
- Explicit MR propagation and caller-thread allocation/deallocation. Exercise both the installed async resource and a supported tracking/resource setup that does not mask early-free errors with deferred deallocation. Record reservation attachment and physical high water independently.
- Pressure during a wide table and midway through a large column, including one oversize intermediate. Verify the 64-frame cap, 64 MiB device and 8 MiB host soft limits, accounting transfer without alias double counting, and eventual retirement. A lone active intrinsic overage must not cause an active-stream retirement query/wait; its next append must relieve the sealed overage. Under-limit appends must report zero retirement queries/waits. Pressure is allowed to wait; lane reuse without pressure is not.
- Unsuppressed Compute Sanitizer memcheck, zero leaked allocations, and multi-device/context tests when hardware is available. Single-device success is not multi-GPU validation.

Rerun the relevant engine converter/late-materialization/filter integration cases and the earlier 29-case integration set. Public completed-return/input-ready/output-rebind assertions matter more than simply matching a standalone output after an extra external synchronization. Exercise reservation OOM rescheduling with the real adaptor, not only `cuda_async_memory_resource` in the microbenchmark.

### Performance and provenance

Use the existing fresh **48-process / 720-hot-sample** 32 GiB protocol as a template: eight distinct columns, four plans, one/four streams, three processes per arm, three verified warmups and 15 hot decodes. Rebuild/record the new unified candidate against the same benchmark objects and record source/diff, library, executable, environment, and payload hashes. The primary acceptance comparison is the archived **additive asynchronous** candidate, not the rejected event implementation. Keep the index32 synchronous baseline and previous unified event candidate as separately labeled diagnostic comparisons; their results cannot waive the additive no-regression gate. Do not overwrite old artifacts or label an old binary as the new design.

Thirty-two GiB denotes the **total logical table**, not each column or compressed input size. The checked SF=1000 YAML currently configures an 8 GB scan task and 32 GB hash-partition/build limits; this decoder matrix reflects the user's requested batch size, not an assertion that every configured scan decodes a 32 GiB batch.

Interleave/alternate arms; fully verify warmups and final timed outputs, state that intermediate hot outputs are not all checked, and keep every process/sample. Use unprofiled completed-decode timings for latency. Before calling a change a regression, rerun a flagged case in a fresh paired collection; three process medians are not a confidence interval.

Acceptance:

1. No unexplained correctness, OOM-contract, allocator, sanitizer, or engine-boundary regression.
2. In the existing eight-column numeric/identity case, no memory-pressure waits, retirement queries, frame-event APIs, or reintroduced per-column completion waits. Nsight should show the same decode kernels/copy bytes unless an explicitly reviewed semantic change explains a difference. Distinguish genuine selection-phase events, host-readback barriers and final stream drains; subtract unchanged extra harness waits. Check that registration/sealing does not allocate or synchronize through an unintended global/default-stream path.
3. No accepted repeatable small- or large-batch latency regression against the additive candidate. A >2% median change is a trigger to investigate/repeat, **not an automatic tolerance**; report smaller consistent changes too. The user rejected the previous small-batch tradeoff. Simpler ownership, removal of events or good 32 GiB results cannot justify retaining a reproducible short-batch regression without a new explicit user decision.
4. Add bounded small/short, mixed-codec, wide/skewed, nullable, predicate and selected-string cases to expose session overhead and backpressure. The 32 GiB homogeneous matrix does not validate these paths.
5. Measure whole-harness physical allocation high water, session-retained temporary high water, and reservation accounting separately. Record the effect of delayed async frees; early frame retirement is not evidence of immediate cross-stream pool reuse. Earlier +28.012 MiB bitpack/key and +14.009 MiB delta differences were **whole-harness** peaks, not isolated scratch measurements or a reservation bound.
6. Use Nsight Systems to distinguish host enqueue overlap, actual cross-stream kernel interval intersections, GPU busy fraction, mandatory host observations, and pressure waits. Use Nsight Compute only for isolated kernel behavior; unchanged kernels need not improve occupancy or DRAM utilization. Do not infer query speedup from decoder microbenchmarks.

### Bounded validation scope

The unified frame/leaf API and retained ownership map are already implemented; do not restart that design. The completed v5 campaign exercised these three bounded areas. Retain them as regression checks for subsequent changes:

1. **Retirement correctness:** forced device/host/frame pressure, duplicate physical handles, active-frame preservation, and external-tail progress. Separate a legitimate longer pressure wait from an accidental under-limit synchronization.
2. **Short-workload cost:** rerun all eight 64 MiB numeric/identity cases with paired provenance against additive, then the representative short/mixed/string cases. The confirmed residuals motivate the concrete storage amendments in §3; preserve and compare each candidate against both additive and its immediate predecessor. Report host-allocation reduction separately from completed latency. Rerun many-owner/scratch-heavy nvCOMP, dictionary/predicate/string and mixed cases: list-node costs, embedded-node layout and active-frame accounting traversal must not simply move a regression to those paths. Further residuals require attribution, not speculative empty-frame modes or a silent rollback of unification.
3. **Memory/throughput tradeoff:** wide/skewed and mixed/entropy workloads must retain bounded ownership and acceptable measured throughput. Report owner, reservation and physical high water separately. The unchanged window constants remain provisional; a separate tuning or allocator proposal requires evidence and review.

Pre-existing issues such as nvCOMP per-chunk status validation and its ignored `frame_size` are worth separate work but are not evidence for, nor prerequisites to, a new asynchronous architecture. Do not bury unrelated format-validation changes in this refactor.

## 11. Why this design, not the alternatives

An owning operation per column is useful when pending results escape independently. Here they do not: public callers require a completed column/table, and engine converters immediately rebind output. Exposing operation handles adds a completion protocol, makes table-wide error draining/retirement external, and duplicates policy in single-column and table adapters. A scoped session fits the actual ownership boundary and hides more complexity behind less interface.

A session that retains every frame to the table boundary is simpler but permits unbounded intermediate retention on wide/mixed tables. One frame per lane bounds retention cheaply but recreates avoidable waits. The chosen retained window keeps ordinary submission open and makes configured retained-state backpressure explicit. Its pressure waits follow the private thresholds; they do not establish that the GPU would otherwise run out of physical memory. It is a deliberate middle ground, not an assertion that the first constants are optimal.

A per-frame event queue is not inherently a generic scheduler and offers finer prefix reclamation. It was tried here, including pressure-only querying, cached sealed accounting and deferred event creation. The user rejected the remaining short-batch cost and authorized coarser stream-tail reclamation. A global event pool, lifecycle fast-path matrix or reclamation stream would add concepts before establishing that this simpler policy suffices; they are not part of this change.

A generic future/task scheduler would solve problems this engine has not asked the decoder to solve and would threaten per-thread reservation accounting. A new codec capability/async registry would restore the original split under different names. A second walker for compacted or predicated execution would duplicate structural correctness. These broader alternatives, including a literal rollback to the additive decoder, are rejected.

The design applies the deep-module/information-hiding guidance from *A Philosophy of Software Design* and the separation/composition guidance from *C++ Software Design*: lifecycle and retirement belong together; plan interpretation and selection policy do not belong in that owner. C++20 value types, spans, unique ownership, and small constrained helpers suffice; neither `std::expected` nor `std::scope_exit` is available without moving beyond the project's current C++20/CUDA20 target.

For this revision the architect read `.claude/CLAUDE.md` and the available role files `/localhome/local-kkristensen/AgentDocs/codex-agents/architect.toml` and `/localhome/local-kkristensen/AgentDocs/agents/architect.md`. The supplied books are present under `/localhome/local-kkristensen/AgentDocs/`: *A Philosophy of Software Design.pdf* (chapter 4, deep modules), *C++ Software Design.pdf* (guideline 2, separation of concerns/design for change), and *C++20 -- The Complete Guide.pdf* (§10.3, borrowed-span design). Their concrete application is to keep `append`/`finish` deep, change retirement inside one module, separate the three private admission limits from completion proof, and copy span-provided stream views without pretending those views own CUDA streams. Modern C++ is used to clarify ownership and typed semantics, not to add a new hierarchy or require a newer language standard.

Reference implementations were used for boundaries, not copied as frameworks: local [DuckDB `PipelineExecutor::PushFinalize`](../duckdb/src/parallel/pipeline_executor.cpp#L353) makes finalization an owner-controlled state transition; [Velox task coordination](https://facebookincubator.github.io/velox/develop/task.html) separates shared execution lifetime from individual operators; [DataFusion's execution-plan contract](https://docs.rs/datafusion/latest/datafusion/physical_plan/trait.ExecutionPlan.html#tymethod.execute) highlights error/cancellation lifetime obligations. Sirius's GPU streams and per-thread reservation attachment make their schedulers inappropriate to transplant. The installed Sirius/RMM/cuDF contracts above determine this proposal.

The result should be easier to explain in one sentence: **decode requests share one interpreter; their enclosing session owns everything queued and returns only completed results.**
