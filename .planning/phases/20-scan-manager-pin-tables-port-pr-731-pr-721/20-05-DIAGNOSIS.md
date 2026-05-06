# Phase 20 Plan 05 — DIAGNOSIS

**Captured:** 2026-05-06
**Branch:** feature/single-node-multi-gpu2
**Host:** 2 × NVIDIA RTX 6000 Ada Generation; cucascade peer-DMA broken on 2 direction(s) (consumer chipset); host-staging fallback active.
**Sanitizer log:** `/tmp/p20_sanitizer.out` (1217 lines, 213474 bytes)
**Sanitizer command:** Canonical Phase 13-02 shape (skipped 13 benign init API errors per protocol).

---

## Pre-Flight Invariant Snapshot

| Gate | Command | Count | Expected | Verdict |
|------|---------|-------|----------|---------|
| DB-grep | `grep -rEn "(->get_data\(\)\|pop_data_batch.*task_created\|data_batch_processing_handle)" src/ test/ \| wc -l` | 4 (2 in `src/legacy/`, 2 in test doc-comments) | 0 in active code | PASS (legacy + comments exempt; matches 20-04 baseline) |
| IO-15 | `grep -rn "cucascade_datasource" src/ test/ \| wc -l` | 0 | 0 | PASS |
| SM-03 | `grep -rn "writer_stream\|record_writer_event" src/op/scan/ \| wc -l` | 1 (sirius_gpu_parquet_scan_operator.cpp:256) | >= 1 | PASS |
| HYG-02 | `grep -rn "rmm::cuda_stream_default" src/ \| wc -l` | 40 | <= 40 | PASS |

**Verdict:** Invariants intact at HEAD. No source modifications since 20-04. Re-attached stream-lineage at sirius_gpu_parquet_scan_operator.cpp:256 (writer_stream comment block) preserved.

---

## Sanitizer Run Metadata

- **Command:** `/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --track-stream-ordered-races=all --show-backtrace=yes --launch-timeout=600 --log-file /tmp/p20_sanitizer.out --print-limit 100 build/release/extension/sirius/test/cpp/sirius_unittest "gpu_execution - TPC-H Query 11 parquet"`
- **Wall-clock:** ~30s (well under 900s budget; well under 132.9s historical Phase 13-02 baseline at 22-query filter — narrower filter here is 1-query)
- **Exit code:** 0 (test PASSED under sanitizer — Phase 13-02 anomaly: sanitizer's launch serialization masks the deadlock-shape failure mode while the underlying race signature remains in the log)
- **`========= ERROR SUMMARY:` line:** `34 errors`
- **Race-class blocks (`Use-before-alloc`):** 21 (the 13 non-race errors are benign init API errors per Phase 13-02 protocol: 12 × `cudaErrorPeerAccessAlreadyEnabled` + 1 × `cudaErrorInvalidDevice` from peer-access probing and downgrade_executor::start)
- **Test result under sanitizer:** `All tests passed (9011 assertions in 1 test case)` — masking effect, not absence of races

---

## FIRST Stream-Ordered Race (verbatim)

Extracted from `/tmp/p20_sanitizer.out` lines 135-192. Full block:

```
========= Use-before-alloc on allocation of size 5,592 bytes at 0x3362000200
=========     Address 0x3362000200 is potentially accessed before it is allocated
=========
=========     Saved host backtrace up to driver entry point at cudaMemcpy time
=========         Host Frame: cuMemcpyHtoDAsync_v2 [0x364679] in libcuda.so.1
=========         Host Frame: unsigned long kvikio::detail::posix_device_io<(kvikio::detail::IOOperationType)0, kvikio::BounceBufferPool<kvikio::CudaPinnedAllocator> >(...) [0x79080] in libkvikio.so
=========         Host Frame: kvikio::detail::posix_device_read(...) [0x7920a] in libkvikio.so
=========         Host Frame: kvikio::FileHandle::pread(...) [0x67654] in libkvikio.so
=========         Host Frame: cudf::io::(anonymous namespace)::file_source::device_read_async(unsigned long, unsigned long, unsigned char*, rmm::cuda_stream_view) [0x18e576d] in libcudf.so
=========         Host Frame: cudf::io::(anonymous namespace)::user_datasource_wrapper::device_read_async(...) [0x18e4dcf] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::read_column_chunks_async(...) [0x186d530] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::read_column_chunks() [0x184262e] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::read_compressed_data() [0x1843bcb] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::setup_next_pass(...) [0x17d7882] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::handle_chunking(...) [0x17e2cf4] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::read() [0x17cfd38] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader::read() [0x17c2f4f] in libcudf.so
=========         Host Frame: cudf::io::read_parquet(cudf::io::parquet_reader_options const&, rmm::cuda_stream_view, ...) [0x1479087] in libcudf.so
=========         Host Frame: sirius::op::scan::sirius_gpu_parquet_scan_operator::read_table_from_metadata(...) [0x14589ba] in sirius_unittest
=========         Host Frame: sirius::op::scan::sirius_gpu_parquet_scan_operator::execute(...) [0x145a6dd] in sirius_unittest
=========         Host Frame: sirius::pipeline::gpu_pipeline_task::compute_task(...) [0x14d35a8] in sirius_unittest
=========         Host Frame: sirius::pipeline::gpu_pipeline_task::execute(...) [0x14d52ec] in sirius_unittest
=========         Host Frame: sirius::pipeline::gpu_pipeline_executor::manager_loop()::{lambda()#2}::operator()() [0x14cc3c1] in sirius_unittest
=========         ... (bounded_thread_pool worker thread frames)
=========
=========     Saved host backtrace up to driver entry point at cudaMallocAsync time
=========         Host Frame: cuMemAllocFromPoolAsync [0x366979] in libcuda.so.1
=========         Host Frame: cudaMallocFromPoolAsync [0x8593e] in libcudart.so.13
=========         Host Frame: rmm::mr::cuda_async_view_memory_resource::do_allocate(unsigned long, rmm::cuda_stream_view) [0x1e08f45] in sirius_unittest
=========         Host Frame: cucascade::memory::detail::legacy_rmm_resource_adapter::__allocate_async(...)
=========         Host Frame: cucascade::memory::detail::reservation_aware_resource_adaptor_impl::do_allocate_unmanaged(...)
=========         Host Frame: cucascade::memory::detail::reservation_aware_resource_adaptor_impl::do_allocate_managed(...)
=========         Host Frame: cucascade::memory::detail::reservation_aware_resource_adaptor_impl::allocate(cuda::stream_ref, unsigned long, unsigned long) [0x1e1f61a] in sirius_unittest
=========         Host Frame: rmm::device_buffer::allocate_async(unsigned long) [0xc28f] in librmm.so
=========         Host Frame: rmm::device_buffer::device_buffer(unsigned long, rmm::cuda_stream_view, ...) [0xd33d] in librmm.so
=========         Host Frame: cudf::io::parquet::detail::read_column_chunks_async(...) [0x186cf41] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::read_column_chunks() [0x184262e] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::read_compressed_data() [0x1843bcb] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::setup_next_pass(...) [0x17d7882] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::handle_chunking(...) [0x17e2cf4] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader_impl::read() [0x17cfd38] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::reader::read() [0x17c2f4f] in libcudf.so
=========         Host Frame: cudf::io::read_parquet(...) [0x1479087] in libcudf.so
=========         Host Frame: sirius::op::scan::sirius_gpu_parquet_scan_operator::read_table_from_metadata(...) [0x14589ba] in sirius_unittest
=========         Host Frame: sirius::op::scan::sirius_gpu_parquet_scan_operator::execute(...) [0x145a6dd] in sirius_unittest
=========         Host Frame: sirius::pipeline::gpu_pipeline_task::compute_task(...) [0x14d35a8] in sirius_unittest
=========         (...same bounded_thread_pool path)
```

---

## File / Line / Subsystem

File: cudf::io::parquet::detail::read_column_chunks_async (libcudf.so)
Line: libcudf-internal; Sirius approximation src/op/scan/sirius_gpu_parquet_scan_operator.cpp:152 (cudf::io::read_parquet call)
Subsystem: cudf+kvikio (FIRST race at sanitizer line 135); secondary cluster subsystem: cucascade (pin 1c1e648)

Detail:
- **File (FIRST race):** `cudf::io::parquet::detail::read_column_chunks_async` (libcudf.so) — allocator and consumer are both INSIDE `read_column_chunks_async`. The Sirius caller is `sirius::op::scan::sirius_gpu_parquet_scan_operator::read_table_from_metadata` at `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:109-171`, which calls `cudf::io::read_parquet(opts, stream)` with the task-local stream — but the cross-stream race is INSIDE cudf+kvikio, beyond Sirius's direct stream control.
- **Line approximation:** `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:127` (gpu_expression_translator construction site that produced the AST) and `:152` (cudf::io::read_parquet call). The actual race site is libcudf-internal.
- **Subsystem:** **cudf+kvikio** for the FIRST race (sanitizer log lines 135-192). **cucascade** for the dominant cluster (races 6-21, 16/21 = 76% of races, all at `cucascade::convert_gpu_to_gpu → reconstruct_column_p2p → alloc_and_peer_copy_async`). Both subsystems are at pin levels Phase 20 cannot trivially modify.

---

## Race Shape Classification

**Shape: A (Use-before-alloc cross-stream) — but inside library boundaries.**

Distribution across 21 race blocks:

| Cluster | Count | Allocator (cudaMallocAsync) | Consumer (cudaMemcpy*) | Subsystem |
|---------|-------|------------------------------|-------------------------|-----------|
| **A: Parquet read** | 5 | `rmm::device_buffer` ctor invoked from `cudf::io::parquet::detail::read_column_chunks_async` on the task-local Sirius stream | `cuMemcpyHtoDAsync_v2` issued by `kvikio::detail::posix_device_io` (host-to-device read of compressed parquet column data) on a kvikio-internal stream and possibly kvikio thread-pool thread | **cudf + kvikio** — Sirius passes `stream` to `read_parquet`; cross-stream gap is inside cudf+kvikio, NOT a Sirius bug |
| **B: cucascade host-stage peer-copy** | 16 | `rmm::device_buffer` ctor invoked from `cucascade::alloc_and_peer_copy_async` on the target/reader stream | `cuMemcpyDtoHAsync_v2` (Device-to-Host) issued by `cucascade::alloc_and_peer_copy_async` for host-staging fallback (peer DMA broken on 2 direction(s); per cucascade init log) | **cucascade pin 1c1e648** — race INSIDE the host-staging fallback path of `convert_gpu_to_gpu` → `reconstruct_column_p2p` → `alloc_and_peer_copy_async` |

**Both clusters are Shape A (use-before-alloc cross-stream), but at different library layers — neither is a Sirius-side missing `cudaStreamWaitEvent` or accessor-scope leak. The Phase 13-04 fix at `cucascade::convert_gpu_to_gpu` (cudaStreamWaitEvent on the writer event) is preserved and IS firing — the FIRST race is one cudf+kvikio call BEFORE Phase 13-04's wait-event guard executes (i.e. during the parquet read itself, not during the cross-GPU transfer).**

The Phase 13-04 architectural fix is upstream of cluster A (the parquet read race fires inside cudf, before any cucascade code runs). Phase 13-04 fixed a different race shape (peer-copy reader vs writer-stream allocation visibility) — that fix preserves correctness for the non-host-stage peer-DMA path, but cluster B reveals a NEW race in the host-staging fallback path that was added post-Phase 13 to handle broken consumer-hardware peer DMA.

---

## Cascade Errors (excluded as downstream of FIRST)

`(21 race blocks total) - 1 (FIRST: cluster A race 1 at line 135) = 20 cascade.`

However, per Phase 13-02 protocol, cascade is "context-poisoning downstream of the FIRST race" — and clusters A vs B have DISJOINT call graphs (cluster A is purely cudf+kvikio; cluster B is purely cucascade peer-copy host-staging fallback). They are **independent races**, not cluster-A-cascade-into-cluster-B.

Refined cascade analysis:
- **Cluster A internal:** races 2-5 cascade from race 1 (same code path, different column chunks).
- **Cluster B internal:** races 7-21 cascade from race 6 (same code path, different columns/sizes within the same `convert_gpu_to_gpu` invocation).
- **Cluster A → Cluster B:** independent (cluster A is the parquet read; cluster B is downstream cross-GPU transfer).

So there are **2 independent root races** (one per cluster), not 1 root + 20 cascade.

---

## Comparison vs Phase 13-02 Race Site

| Dimension | Phase 13-02 (v1.3) | Plan 20-05 (v1.4) |
|-----------|---------------------|---------------------|
| FIRST race file | `cucascade/src/data/representation_converter.cpp:801` (convert_gpu_to_gpu peer-DMA path) | libcudf-internal (`cudf::io::parquet::detail::read_column_chunks_async` via kvikio) |
| FIRST race subsystem | cucascade (peer-DMA writer-event missing) | cudf + kvikio (parquet reader internal stream-ordering) |
| Total race count | 433 | 21 (94% reduction — Phase 13-04 + Phase 16 CC-03 closed the dominant cluster) |
| Cluster B cucascade race count | (subsumed in 433) | 16 (host-staging fallback path; NEW post-Phase 13) |
| Phase 13-04 fix preserved? | N/A (Phase 13-02 was pre-fix) | **Yes** — `cudaStreamWaitEvent(target_stream, src.get_writer_event(), 0)` at top of convert_gpu_to_gpu fires correctly; cluster B race is inside per-column `alloc_and_peer_copy_async` AFTER the entry-level wait-event |

**Conclusion:** Phase 13-04 closed the original race site (peer-DMA reader vs writer-stream alloc). Plan 20-05 reveals two NEW race sites that surfaced post-v1.3:
1. **Cluster A:** A pre-existing cudf+kvikio race that the original (smaller) sanitizer error count had buried in cascade noise. Now visible because Phase 13-04 + Phase 18 + Phase 19 + Phase 20 cleaned up everything else.
2. **Cluster B:** A new race introduced by the cucascade host-staging fallback path (added to handle broken consumer-hardware peer DMA, per cucascade init log "direct GPU↔GPU peer DMA broken on 2 direction(s); cudaMemcpyPeer* will host-stage automatically").

---

## Hypothesis Disposition (memory's "Remaining races to find")

| Memory hypothesis | Disposition | Evidence |
|-------------------|-------------|----------|
| **(a) HYG-02 regression** (`rmm::cuda_stream_default` introduced somewhere new) | **DEAD** | HYG-02 grep = 40 (unchanged from Phase 8-19 baseline; all in `src/legacy/`). Pre-flight invariant snapshot above. |
| **(b) `cudf::*_scalar` cross-stream pattern** (filter translator residual) | **DEAD** | No `cudf::*_scalar` frame in any sanitizer error block. Phase 10-04 fix (translation_stream + owned_stream RAII at translated_expression) holds. |
| **(c) `sirius_p2p_converter.cpp` cross-stream reads of source.get_table without sync** | **DEAD** | Sirius-side `sirius_p2p_converter.cpp` was retired in Phase 13-04 Path-2 → cucascade `convert_gpu_to_gpu`. No `sirius::data::sirius_p2p_converter` frame in any error. |
| **(d) NEW cudf+kvikio parquet-read race** | **MATCHES** | Cluster A (5/21 races) — root site is `cudf::io::parquet::detail::read_column_chunks_async` calling `kvikio` on a different stream than the `rmm::device_buffer` allocation. Library-internal; not a Sirius bug. |
| **(e) NEW cucascade host-stage peer-copy race** | **MATCHES** | Cluster B (16/21 races) — root site is `cucascade::alloc_and_peer_copy_async` host-staging fallback path; cudaMemcpyAsync DtoH races against immediately-preceding cudaMallocAsync. Inside cucascade pin 1c1e648; not a Sirius file edit. |

---

## Path Recommendation

**Path B (Escalate to user / Phase 21).**

**Rationale (per plan 20-05 decision gate Step 7):**

1. **Both root races are at library boundaries Phase 20 cannot trivially modify:**
   - Cluster A: cudf+kvikio internal parquet reader stream-ordering. Sirius passes a single `stream` to `cudf::io::read_parquet`; the cross-stream gap is inside cudf's parquet reader (uses kvikio's `BS::thread_pool` for filesystem reads on a separate cudaStream from the rmm::device_buffer alloc stream). A Sirius-side localized fix would have to go through cudf or use a different parquet reader entirely.
   - Cluster B: cucascade host-staging fallback path inside `alloc_and_peer_copy_async`. The fix is to add `cudaStreamWaitEvent` between the cudaMallocAsync (allocator stream) and cudaMemcpyAsync (DtoH on target stream) inside that helper. This requires editing cucascade pin 1c1e648 → submodule fork + bump.

2. **Race shape E (cucascade-internal lineage gap) per plan 20-05 Race-shape taxonomy → Path B (escalate) territory.** Plan explicitly states: "If Shape == E (cucascade-internal) → Path B".

3. **Both races are "novel race shapes" for Phase 20:** Phase 13-04 closed the peer-DMA path; this run reveals (a) a parquet-reader race inside cudf+kvikio that is upstream of the entire stream-lineage chain Phase 13/16/20 maintained, and (b) a new race in the host-staging fallback that doesn't exist on server hardware with working peer DMA. Per plan: "If Shape unrecognized → Path B (escalate with note 'novel race shape')."

4. **Phase 20 scoping correctness:** The 20-05 plan's <objective> states "If diagnosis reveals a structural issue (>1 day fix), document and escalate." Both fix shapes are >1 day — cucascade fork+bump is at minimum 1-2 days (fix + test + ctest re-run + Sirius bump), and the cudf+kvikio side is a multi-week investigation (or a workaround in Sirius like calling `cudaStreamSynchronize(stream)` after `cudf::io::read_parquet` returns — but that's a regression of the entire async-read framework Phase 19 just installed).

PATH: B

---

## Why the Test PASSES Under Sanitizer

This is the **Phase 13-02 anomaly recurrence** — under compute-sanitizer's `--track-stream-ordered-races=all`, the sanitizer's per-launch ordering checks serialize kernel dispatches enough that the cumulative race doesn't poison the GPU context fast enough to deadlock. The 21 reported races are still **definitive evidence** of the underlying race; the un-sanitized run at 20-04 still hit `cudaErrorIllegalAddress` at `cuda_stream_view.cpp:45` (which is the canonical "downstream surfacing of any earlier race" per `project_phase08_fu17`). The races fire; the failure mode is timing-dependent.

Plan 20-05 explicitly anticipated this anomaly: "Sanitizer can change observable failure mode (Phase 13-02 saw the same suite PASS under sanitizer due to launch serialization). The 433-style error count + first-error-block is the evidence; whether the test 'passes' or 'fails' under sanitizer is irrelevant."

---

## Cross-Reference: 20-04 Failure Fingerprint

20-04-RESULTS.md captured the un-sanitized failure shape:

```
gpu_execution - TPC-H Query 11 parquet
test/cpp/integration/test_gpu_execution_tpch.cpp:229: FAILED:
  REQUIRE_FALSE( gpu_result->HasError() )
with messages:
  num_gpus := 2
  transparent GPU execution error: INTERNAL Error: Sirius GPU execution failed:
  Invalid Error: CUDA error at: /tmp/conda-bld-output/bld/rattler-build_librmm/
  work/cpp/src/cuda_stream_view.cpp:45: cudaErrorIllegalAddress
```

Per memory's "Crash signatures seen and what they meant" table:
> | `cudaErrorIllegalAddress` at `cuda_stream_view.cpp:45` on sync | Downstream surfacing of any earlier race |

The 21 stream-ordered races identified in this diagnosis are exactly the "earlier race" that surfaces as `cudaErrorIllegalAddress` at sync time un-sanitized.

---

## Recommended Next-Step Shape (for INVESTIGATION.md)

**Cluster B (cucascade host-stage) — the more localized of the two:**

The cucascade `alloc_and_peer_copy_async` helper (in `cucascade/src/data/representation_converter.cpp`, anonymous namespace) needs:
- After `rmm::device_buffer` allocation completes on `dst_stream`, record an event on `dst_stream`.
- Before `cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, dst_stream)` (the host-staging copy), `cudaStreamWaitEvent(dst_stream, alloc_event, 0)`.
- OR, more conservatively: ensure the `rmm::device_buffer::allocate_async` and the subsequent `cudaMemcpyAsync` use the SAME stream (Phase 13 Path-2 invariant).

Inspecting the sanitizer trace, **the producer (alloc) and consumer (memcpy) frames both pass `rmm::cuda_stream_view` arguments** — but the sanitizer flags them as different. This suggests `alloc_and_peer_copy_async` is internally splitting work across two different streams (allocator stream + memcpy stream), which is the structural bug.

This is a cucascade fork+bump. Estimated effort: 1-2 days (1 day fix + 0.5 day cucascade ctest + 0.5 day Sirius re-validation + submodule bump + downstream test).

**Cluster A (cudf+kvikio) — harder:**

The cudf+kvikio race is in upstream library code Sirius doesn't own. Workarounds:
1. **Heavyweight:** Synchronize the task-local stream after `cudf::io::read_parquet` returns (`cudaStreamSynchronize(stream)`). Regresses async pipelining gains from Phase 19.
2. **Lightweight:** Switch from kvikio-backed datasource (post-PR-731) back to a simple cudf-internal datasource that doesn't fork to kvikio's BS::thread_pool. Defeats the IO-12..17 work in Phase 19.
3. **Upstream:** File cudf/kvikio bug report; wait for upstream fix.

Estimated effort: 1+ week minimum for any of (1)-(3), and (1)/(2) regress Phase 19 gains.

---

PATH: B
