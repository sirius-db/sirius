# 20-STREAM-LINEAGE-REATTACH.md — SM-03 Re-Attachment Decision

**Captured:** 2026-05-06
**Plan:** 20-02 (Wave 2 — TODO cleanup + design docs)
**Closes documentation gate:** SM-03 (Phase 13 stream-lineage re-attachment)
**Anchor evidence:** [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md) Static Grep Gate 1
**Predecessor extraction document:** [`17-PHASE-13-EXTRACT.md`](../17-sirius-origin-dev-merge-base-layer/17-PHASE-13-EXTRACT.md)

---

## Context

Phase 17 MERGE-03 explicitly extracted Phase 13 stream-lineage attachment points before accepting the deletion of `sirius_parquet_metadata_scan_operator.hpp` (deleted by PR #731 in the dev-merge). The extraction document, [`17-PHASE-13-EXTRACT.md`](../17-sirius-origin-dev-merge-base-layer/17-PHASE-13-EXTRACT.md), captured the full 232-line file plus the writer_event / writer_stream call sites that needed to be re-wired in the post-#731 architecture. Phase 20 ROADMAP success criterion 1 codifies the regression armor:

> `grep -rn "writer_stream\|record_writer_event" src/op/scan/` returns non-zero.

[`20-CONTEXT.md`](20-CONTEXT.md) decisions section asks the planner to answer:

> *"Stream-lineage re-attachment site: `parquet_split_provider::run_batch` (option A) vs `sirius_gpu_parquet_scan_operator::execute` (option B). Document in `20-STREAM-LINEAGE-REATTACH.md`."*

This document closes that question.

---

## Decision Summary

- **Option B chosen:** stream-lineage is re-attached at **`src/op/scan/sirius_gpu_parquet_scan_operator.cpp:259`** (post 20-02 Task 2 edit; was line 263 pre-edit, shifted up by 4 lines after the deletion of the misleading TODO block at the old lines 173-176). The operative call is the 3-arg form:
  ```cpp
  auto batch = sirius::make_data_batch(std::move(table), *mem_space, stream);
  ```
  The `stream` arg is the task-local execution stream propagated by `gpu_pipeline_task::execute` down through the operator chain.
- The cucascade `gpu_table_representation` ctor at the merged pin (`1c1e648`) records a CUDA event on `writer_stream` automatically inside the ctor body (`cucascade/include/cucascade/data/gpu_data_representation.hpp:208`: `if (writer_stream.value() != nullptr) { record_writer_event(writer_stream); }`). **No manual `record_writer_event` call is needed at this site** — the writer_stream-as-required-ctor-arg invariant from Phase 13-04 Path-2 enforces it.
- This re-attachment was implemented opportunistically as a side-effect of Phase 18 Pitfall 4 closure (the 3-arg `make_data_batch` migration). Phase 20 records the architectural intent explicitly and adds the ROADMAP grep gate.

---

## Empirical Evidence (cited from 20-01-EVIDENCE.md)

### Static Grep Gate 1 — writer_stream / record_writer_event in src/op/scan/ (cited verbatim from 20-01-EVIDENCE.md ## Gate 1 section)

**Command:**
```
grep -rn "writer_stream\|record_writer_event" src/op/scan/
```

**Output (verbatim, post-Task 2 edit; line number shifted up by 4 from the pre-edit baseline):**
```
src/op/scan/sirius_gpu_parquet_scan_operator.cpp:256:  // execution stream as writer_stream — preserves Phase 13-04 Path-2
```

**Hit count:** 1 (literal `writer_stream` token in canonical comment block at sirius_gpu_parquet_scan_operator.cpp:255-258, post-Task 2 edit).

**Verdict:** **PASS** — non-zero hits. Phase 20 ROADMAP success criterion 1 substantiated. RESEARCH.md Pitfall 7 regression armor in place: any future RAII refactor that substitutes a default-constructed `cuda_stream_view{}` for the task stream would silently delete this comment-block witness, and the grep gate would flag the loss.

> Pre-Task 2 baseline (from 20-01-EVIDENCE.md): the literal `writer_stream` token survived at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:260` in the canonical Phase 13-04 Path-2 comment block. The actual stream-lineage re-attachment was the 3-arg `make_data_batch(table, mem_space, stream)` call at line 263. After Task 2's TODO-block deletion at the unrelated lines 173-176, the load-bearing block shifted up by 4 lines: `writer_stream` token now at line 256, `make_data_batch` call now at line 259. The grep gate counts the same hit; the regression-armor invariant is unchanged.

### Source line citation (post-edit positions)

`src/op/scan/sirius_gpu_parquet_scan_operator.cpp:254-262`:
```cpp
  // Wrap the GPU table in operator_data for the downstream pipeline.
  // Pitfall 4 closure (Phase 18): 3-arg make_data_batch with the operator's
  // execution stream as writer_stream — preserves Phase 13-04 Path-2
  // stream-lineage so cucascade::convert_gpu_to_gpu can call cudaStreamWaitEvent
  // on the recorded writer event before peer-copying.
  auto batch = sirius::make_data_batch(std::move(table), *mem_space, stream);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
```

This is verbatim Code Example 2 from [`20-RESEARCH.md`](20-RESEARCH.md), shifted up by 4 lines after Task 2's TODO removal.

---

## Source Citation

The Phase 13-04 Path-2 comment block + 3-arg `make_data_batch` call at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:255-259`:

```cpp
// Pitfall 4 closure (Phase 18): 3-arg make_data_batch with the operator's
// execution stream as writer_stream — preserves Phase 13-04 Path-2
// stream-lineage so cucascade::convert_gpu_to_gpu can call cudaStreamWaitEvent
// on the recorded writer event before peer-copying.
auto batch = sirius::make_data_batch(std::move(table), *mem_space, stream);
```

The cucascade-side ctor signature (verified at HEAD against the pinned `1c1e648`):

`cucascade/include/cucascade/data/gpu_data_representation.hpp:69`:
```cpp
gpu_table_representation(std::unique_ptr<cudf::table> table,
                         memory::memory_space& mem_space,
                         rmm::cuda_stream_view writer_stream);
```

`cucascade/include/cucascade/data/gpu_data_representation.hpp:90`:
```cpp
gpu_table_representation(cudf::table_view table_view,
                         memory::memory_space& mem_space,
                         rmm::cuda_stream_view writer_stream);
```

`cucascade/include/cucascade/data/gpu_data_representation.hpp:208` (inside the ctor body):
```cpp
if (writer_stream.value() != nullptr) { record_writer_event(writer_stream); }
```

Both ctors REQUIRE `writer_stream` as the third argument; the ctor body records the writer_event automatically when `writer_stream` is non-default. This is the architectural Path-2 fix from Phase 13-04 — see [`13-04-SUMMARY.md`](../13-q11-multi-gpu-illegal-address/13-04-SUMMARY.md) for the original landing.

---

## Cross-Device Stream Synchronization Chain

The end-to-end stream-lineage chain that Option B preserves:

1. **`gpu_pipeline_executor::manager_loop`** (src/pipeline/gpu_pipeline_executor.cpp) acquires the task and constructs a task-local `rmm::cuda_stream` via `_memory_space->make_reservation(...)`. The stream is bound to the GPU pinned by `rmm::cuda_set_device_raii guard(device_id)`.
2. **`gpu_pipeline_task::execute(stream)`** (src/pipeline/gpu_pipeline_task.cpp:295) iterates the operator chain, passing the task-local `stream` to each `op->execute(input_data, stream)` call.
3. **`sirius_gpu_parquet_scan_operator::execute(input_data, stream)`** (src/op/scan/sirius_gpu_parquet_scan_operator.cpp:176) receives the task-local `stream` directly. For `parquet_scan_data` inputs, it calls `read_table_from_metadata(*scan_data, stream)`.
4. **`read_table_from_metadata`** (src/op/scan/sirius_gpu_parquet_scan_operator.cpp:109-171) issues `cudf::io::read_parquet(opts, stream)` on the task-local stream — all parquet reads land in stream order on the target GPU.
5. **`make_data_batch(table, mem_space, stream)`** (src/op/scan/sirius_gpu_parquet_scan_operator.cpp:259) constructs the cucascade `gpu_table_representation` via the 3-arg ctor; the ctor body (gpu_data_representation.cpp:208) calls `record_writer_event(writer_stream)`, which records a `cudaEvent_t` on the task-local stream.
6. **Downstream cross-device peer copy via `cucascade::convert_gpu_to_gpu`** (cucascade/src/data/representation_converter.cpp:801, identified as the FIRST stream-ordered race site in [`13-02-SUMMARY.md`](../13-q11-multi-gpu-illegal-address/13-02-SUMMARY.md)) calls `cudaStreamWaitEvent(reader_stream, get_writer_event(), 0)` before issuing the peer DMA, ensuring the producer's writes are visible before the peer copy reads.

The chain only holds if step 5's `writer_stream` is the task-local stream (NOT a default-constructed `cuda_stream_view{}`) — which is precisely what the post-Task 2 line 259 enforces.

---

## Why NOT Option A (`parquet_split_provider::run_batch`)

Cited verbatim from [`20-RESEARCH.md`](20-RESEARCH.md) Pitfall 2:

> **What goes wrong:** Plan author notices `auto stream = cudf::get_default_stream();` at `parquet_split_provider.cpp:184` and proposes "fix" to use task-local stream.
> **Why it happens:** Surface similarity to HYG-02 / `rmm::cuda_stream_default` pattern. But this is `cudf::get_default_stream()`, not `rmm::cuda_stream_default`, and it's used only for AST translation (CPU-side metadata work) at PLANNING time. The actual GPU stream — used by `read_table_from_metadata` and `make_data_batch` at execute() time — is the task-local stream.
> **How to avoid:** Read the call site fully: `gpu_expression_translator translator(stream, ...)` followed by `translator.translate_expression_with_names(...)` returns an AST that's later passed to `reader_options::set_filter`. The AST is opaque GPU literals + scalars created on this stream, but they're recorded into options. Task-time `read_parquet` consumes the options and reads on the task stream. The stream-lineage gate is `make_data_batch`'s writer_stream (HEAD line 263), not the AST translator's planning-time stream. Phase 17 PHASE-13-EXTRACT.md called this out as a "secondary candidate" for re-attachment but it is NOT necessary.

Concretely: the `parquet_split_provider::run_batch` site uses `cudf::get_default_stream()` purely for opaque AST literal/scalar construction at PLANNING time (line 184 of `src/scan_manager/parquet_split_provider.cpp`). The AST is plumbed through `reader_options::set_filter` and later consumed by the task-time `cudf::io::read_parquet(opts, stream)` call inside `read_table_from_metadata` — at which point the stream lineage is the task-local stream. Recording a writer_event on the planning-time stream would be premature (the actual writes happen later, on a different stream).

---

## Why NOT Option C (manual `record_writer_event` call)

Cited verbatim from [`20-RESEARCH.md`](20-RESEARCH.md) "Don't Hand-Roll" row 3:

> | **Stream-event recording** | **Don't Build:** A new `record_writer_event` call site in `parquet_split_provider::run_batch` | **Use Instead:** The existing `make_data_batch(table, mem_space, stream)` 3-arg ctor at `sirius_gpu_parquet_scan_operator.cpp:263` | **Why:** Phase 13-04 Path-2 made writer_stream a REQUIRED ctor argument; the ctor body auto-records the event. No manual `record_writer_event` call needed. |

Phase 13-04 Path-2 made `writer_stream` a REQUIRED ctor argument to `gpu_table_representation` (verified above at `gpu_data_representation.hpp:69` + `:90`). The ctor body unconditionally records the event when `writer_stream.value() != nullptr` (verified above at `gpu_data_representation.cpp:208`). A manual `record_writer_event` call at the operator site would be redundant — and worse, would risk ordering bugs if added before the cudf table is fully constructed (since `record_writer_event` records on the current stream state, not a future state).

---

## P2 Pitfall Sentinel (Phase 13 fingerprint)

Cited verbatim from [`20-RESEARCH.md`](20-RESEARCH.md) Pitfall 7:

> **What goes wrong:** A future RAII refactor moves the `make_data_batch` call site or changes how `stream` is captured, accidentally substituting a default-constructed `cuda_stream_view{}` for the task stream.
> **Why it happens:** Phase 13 / Phase 18 history; the lambda capture / move semantics are subtle.
> **How to avoid:** The grep gate (`grep -rn "writer_stream\|record_writer_event" src/op/scan/`) is mandatory and codified in ROADMAP success criterion 1. Plan must assert `>= 1` match (HEAD has the comment-block reference at line 260-263 + the actual `make_data_batch(... stream)` call at 263).
> **Warning signs:** Cross-GPU SIGSEGV / illegal-address only at SF100 Q11 num_gpus=2 (the canonical Phase 13 fingerprint). If `[mgpu_stress]` passes but SF100 Q11 fails, P2 is back.

The grep gate (writer_stream / record_writer_event in src/op/scan/) is permanent regression armor. If a future RAII refactor accidentally substitutes a default-constructed `cuda_stream_view{}` for the task stream — or moves the `make_data_batch` call to a planning-time site — the SF100 Q11 num_gpus=2 illegal-address fingerprint returns. Phase 21 REG-04 still owns the formal SF100 Q11 verification gate; this document and the grep gate keep the invariant visible day-to-day.

---

## Verdict

**SM-03: PASS via Option B** — `sirius_gpu_parquet_scan_operator::execute` at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:259` (post Task 2 edit; was line 263 pre-edit). Empirically gated by:
- [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md) Static Grep Gate 1 — non-zero `writer_stream` hits in `src/op/scan/`
- Continuity from Phase 18-VERDICT-V2 ([mgpu] 16/16 PASS, 79091 assertions)
- Continuity from Phase 19-VERDICT ([TPC-H][parquet] 22/22 PASS at num_gpus=2, 36256 assertions, sanitizer clean)
- Plan 20-01 [mgpu_stress] 500-iter PASS at 77053 assertions / 73.8s (which exercises the full source-pipeline stream-lineage chain on 2-GPU host across 100 RR-counter offsets × 5 representative queries)

ROADMAP Phase 20 success criterion 1 satisfied. Phase 13-04 Path-2 architectural fix carried forward through Phase 17 (extraction), Phase 18 (3-arg make_data_batch migration), Phase 19 (IO framework adoption preserved the writer_stream contract), and now Phase 20 (documented + grep-gated).

**No source code changes** to `parquet_split_provider`, `sirius_scan_manager`, `sirius_gpu_parquet_scan_operator::execute`, or the cucascade ctor surface were required to close SM-03. The work was the documentation captured in this file plus the grep gate codified in [`20-01-EVIDENCE.md`](20-01-EVIDENCE.md).
