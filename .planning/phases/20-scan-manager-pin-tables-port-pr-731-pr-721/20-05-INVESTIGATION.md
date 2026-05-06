# Phase 20 Plan 05 — INVESTIGATION (Path B Escalation)

**Captured:** 2026-05-06
**Branch:** feature/single-node-multi-gpu2
**Authoritative cite:** `20-05-DIAGNOSIS.md` (FIRST race fingerprint + race shape taxonomy + cluster distribution).
**Cucascade pin at HEAD:** `1c1e648` (Phase 16 CC-03 re-attach; Phase 20-02 SM-03 grep-gated).
**Sanitizer log:** `/tmp/p20_sanitizer.out` (1217 lines, 21 stream-ordered race blocks).

---

## Status: human_needed

STATUS: human_needed

The Q11 SF1 num_gpus=2 parquet failure is structurally beyond a localized Phase 20 fix. Phase 20 closes 5/6 SM-XX requirements; SM-06 SF1 component remains BLOCKED on this. **User decision required** on which fix shape to pursue and whether to proceed with v1.4 ship while carrying the gap, or absorb a 1-2 day cucascade fork+bump cycle into the schedule.

---

## Structural Finding

The compute-sanitizer FIRST stream-ordered race (sanitizer log line 135) and the dominant cluster (16/21 races) are at **library-internal boundaries** Phase 20 cannot trivially modify:

### Cluster A — cudf + kvikio (5 of 21 races)

```
========= Use-before-alloc on allocation of size 5,592 bytes at 0x3362000200
=========     Saved host backtrace up to driver entry point at cudaMemcpy time
=========         Host Frame: cuMemcpyHtoDAsync_v2 [...] in libcuda.so.1
=========         Host Frame: kvikio::detail::posix_device_io [...] in libkvikio.so
=========         Host Frame: cudf::io::file_source::device_read_async [...] in libcudf.so
=========         Host Frame: cudf::io::parquet::detail::read_column_chunks_async [...] in libcudf.so
=========         Host Frame: ... cudf::io::read_parquet [...]
=========         Host Frame: sirius::op::scan::sirius_gpu_parquet_scan_operator::read_table_from_metadata
=========     Saved host backtrace up to driver entry point at cudaMallocAsync time
=========         Host Frame: rmm::device_buffer::device_buffer [...] in librmm.so
=========         Host Frame: cudf::io::parquet::detail::read_column_chunks_async [...] in libcudf.so
=========         (same Sirius caller chain)
```

The producer (`rmm::device_buffer` ctor's `allocate_async(stream)`) and consumer (`cudaMemcpyHtoDAsync` issued via kvikio's `BS::thread_pool::worker`) are both nested inside `cudf::io::parquet::detail::read_column_chunks_async`. Sirius passes a SINGLE `stream` argument to `cudf::io::read_parquet`. The cross-stream gap is opened **inside cudf** when it dispatches the actual read I/O to kvikio's filesystem thread pool, which uses a different cudaStream than the device_buffer allocation stream.

This is **NOT a Sirius-side missing `cudaStreamWaitEvent`**. There is no Sirius-controllable stream where a Sirius-side fix could insert a wait-event. The cross-stream split happens entirely inside cudf's parquet reader implementation (between `read_column_chunks_async`'s allocator path and its kvikio-dispatched read path).

### Cluster B — cucascade host-staging fallback (16 of 21 races)

```
========= Use-before-alloc on allocation of size 128 bytes at 0x3362000000
=========     Saved host backtrace up to driver entry point at cudaMemcpy time
=========         Host Frame: cuMemcpyDtoHAsync_v2 [...]
=========         Host Frame: cudaMemcpyAsync [...] in libcudart.so.13
=========         Host Frame: cucascade::(anonymous namespace)::alloc_and_peer_copy_async [...]
=========         Host Frame: cucascade::(anonymous namespace)::reconstruct_column_p2p [...]
=========         Host Frame: cucascade::(anonymous namespace)::convert_gpu_to_gpu [...]
=========         Host Frame: cucascade::representation_converter_registry::convert_impl [...]
=========         Host Frame: sirius::op::pipelineable_operator_data::prepare_for_processing [...]
=========         Host Frame: sirius::pipeline::gpu_pipeline_task::execute
=========     Saved host backtrace up to driver entry point at cudaMallocAsync time
=========         Host Frame: rmm::device_buffer::device_buffer [...] in librmm.so
=========         Host Frame: cucascade::(anonymous namespace)::alloc_and_peer_copy_async [...]
=========         (same chain)
```

This fires inside cucascade's host-staging fallback for the broken peer-DMA case (per cucascade init log: `[cucascade] direct GPU↔GPU peer DMA broken on 2 direction(s); cudaMemcpyPeer* will host-stage automatically.`). The producer (`rmm::device_buffer` allocate_async) and consumer (`cudaMemcpyAsync` Device-to-Host for host-staging) are both inside `alloc_and_peer_copy_async` — the helper internally splits work across two streams (allocator stream + memcpy stream) without an event-wait between them.

This is the canonical Phase 13-04 race shape **resurfacing in a different code path** that did not exist when Phase 13-04 was authored:
- Phase 13-04 fixed `convert_gpu_to_gpu` for the peer-DMA path (cudaStreamWaitEvent on the writer event at the entry of `convert_gpu_to_gpu`).
- The host-staging fallback path was added LATER (post-Phase 13, pre-pin 1c1e648) to handle broken consumer-hardware peer DMA — and that path has its OWN per-column allocate-then-DtoH-copy structure that the Phase 13-04 entry-level wait-event does not cover.

This is **race shape E** per plan 20-05's taxonomy: "cucascade-internal lineage gap. Race fires inside cucascade code at pin 1c1e648 (e.g. convert_host_to_gpu missing wait). Submodule bump required — Path B (escalate) territory."

---

## Why Path A Was Not Pursued

Plan 20-05 explicitly directs Path B for race shape E (cucascade-internal) and for novel race shapes. Both apply here:

1. **Race shape E** for cluster B (cucascade host-staging fallback inside `alloc_and_peer_copy_async`). Per plan: "If Shape == E (cucascade-internal) → Path B (escalate; skip Task 2 fix and proceed to Task 3 INVESTIGATION + Task 4 verify-baseline-only)."

2. **Novel race shape** for cluster A (cudf+kvikio internal stream-ordering). The cudf+kvikio race is at a library boundary upstream of every Phase 13/16/20 stream-lineage attachment point. There is no Sirius-side single-file edit that closes the race without either:
   - **(a) Synchronously waiting** on the Sirius-side stream after `cudf::io::read_parquet` returns — which would regress the entire Phase 19 async-IO framework Sirius just adopted.
   - **(b) Switching back** from the kvikio-backed sirius_datasource (Phase 19 IO-12..17 work) to a non-kvikio cudf datasource — also regresses Phase 19 gains and would require IO-12..17 unwinding.
   - **(c) Upstream fix** in cudf or kvikio — out of Sirius's control, multi-week timeline.

3. **No fix attempted in Task 2.** The plan's path gate was honored: Task 2 returned "## Fix Skipped — Path B Selected" annotation in DIAGNOSIS.md without modifying any source files. Phase 18..20 invariants are intact (DB-grep matches 20-04 baseline at 4 hits in legacy + comments; IO-15=0; SM-03=1; HYG-02=40).

---

## Recommended Fix Shape

### Primary recommendation: Cucascade fork+bump (Cluster B)

**Site:** `cucascade/src/data/representation_converter.cpp`, anonymous-namespace helper `alloc_and_peer_copy_async`.

**Fix shape:**

Looking at the sanitizer trace, the producer (`rmm::device_buffer::allocate_async`) is invoked through `cucascade::reservation_aware_resource_adaptor_impl::allocate(stream_ref, size, alignment)` — which uses `stream_ref` (the allocator stream). The consumer (`cudaMemcpyAsync` Device-to-Host for host-staging) is on what looks like a separate stream argument to `alloc_and_peer_copy_async`.

Three viable fix shapes for cucascade-side closure:

1. **Same-stream invariant (preferred):** Ensure `alloc_and_peer_copy_async` issues `rmm::device_buffer::allocate_async` and the subsequent `cudaMemcpyAsync` on the SAME stream. Read the cucascade source to confirm the actual stream split (likely the allocator currently uses a per-resource-adaptor internal stream; force it to the function's `dst_stream` argument).

2. **Event-bridge:** Record an event on the allocator stream after `rmm::device_buffer` ctor returns; `cudaStreamWaitEvent(memcpy_stream, alloc_event, 0)` before `cudaMemcpyAsync`. More invasive but works regardless of the allocator's stream choice.

3. **Single-stream serialize:** Force `alloc_and_peer_copy_async` to call `rmm::device_buffer::allocate_async(size, dst_stream, mr)` explicitly with `dst_stream`, so allocator and memcpy stream are guaranteed identical.

**Estimated effort:** 1-2 days for cucascade-side fix:
- 0.25 day: reproduce + identify exact stream split inside `alloc_and_peer_copy_async`.
- 0.5 day: implement fix (likely shape 1 or 3).
- 0.25 day: cucascade ctest re-run + cucascade-side sanitizer verification.
- 0.5 day: Sirius-side submodule bump + re-run Phase 20 [integration][TPC-H] 48/48 + [mgpu] 16/16 gates + Phase 21 prelude.
- 0.5 day buffer for unforeseen pin-table / RAII compile fallout from the cucascade tree change.

**Why this is the smaller, more tractable fix:** Cluster B is 16/21 races (76% of the pipeline). Closing it should remove the dominant `cudaErrorIllegalAddress` failure mode at sync time. Cluster A may surface as a residual race but is upstream-library — a stop-gap workaround there (e.g., cudaStreamSynchronize after read_parquet) becomes acceptable once cluster B is closed.

### Secondary recommendation: Cluster A workaround OR upstream fix

**Site:** Sirius-side, `src/op/scan/sirius_gpu_parquet_scan_operator.cpp:152` (just after `cudf::io::read_parquet` returns).

**Fix shape (workaround):** After `cudf::io::read_parquet(opts, stream)` returns, insert `stream.synchronize()` (i.e., `cudaStreamSynchronize(stream)`). This forces the kvikio thread-pool reads to complete before any downstream operator consumes the resulting cudf::table. Cost: regresses async-IO pipelining gains from Phase 19 IO-12..17 by ~10-20% on parquet-bound queries (estimate; not measured).

**Fix shape (upstream):** File a cudf+kvikio bug report / PR. Wait. Multi-week timeline.

**Estimated effort (workaround):** 0.5 day (one-line edit + re-run [integration][TPC-H] + [TPC-H][parquet] gates). But if applied alone (without cluster B fix), cluster B still fails — workaround alone is insufficient.

### Alternative path: Disable parquet path for num_gpus>=2 at SF<10

**Site:** `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` or `src/sirius_extension.cpp` config gating.

**Fix shape:** Add a runtime gate that falls back to DuckDB-attach (CPU+1-GPU) for parquet scans when num_gpus>=2 AND data size below some threshold. SF1 still hits this path; SF10 may or may not (per 20-04 SM-06 SF10 PASS). Punts the bug.

**Why NOT recommended:** Defeats SM-06 acceptance criterion ([integration][TPC-H] 48/48 SF1 num_gpus=2 PASS). User-visible behavior change (parquet falls back to CPU on small datasets — slow). Acceptable only as a temporary v1.4 ship-with-known-limitation if cucascade fork+bump is too expensive.

---

## Estimated Effort

| Path | Effort | Risk | Coverage |
|------|--------|------|----------|
| Cucascade fork+bump (cluster B) + Sirius synchronize workaround (cluster A) | 1.5-2.5 days | Low (fork is mechanical; synchronize is one line) | Closes both clusters; SM-06 SF1 PASS |
| Cucascade fork+bump alone (cluster B only) | 1-2 days | Low | Closes 76% of races; cluster A residual may still fail at sync time |
| Sirius synchronize workaround alone (cluster A only) | 0.5 day | Medium (regresses Phase 19 gains by 10-20% on parquet-bound queries) | Closes 24% of races; cluster B residual fails |
| Upstream cudf+kvikio fix | Multi-week | High (out of Sirius control) | Closes cluster A only |
| Disable parquet at num_gpus>=2 small-SF (alternative path) | 0.5 day | Low for ship; behavior regression for users | Punts SM-06 SF1 acceptance criterion |

**Recommended escalation outcome:** User decides between (a) absorb 1.5-2.5 days for cucascade fork+bump + sync workaround → SM-06 SF1 PASS, full Phase 20 closure, full Phase 21 unblock; (b) ship v1.4 with known limitation + carry SM-06 SF1 + Q11 num_gpus=2 parquet to v1.5; (c) implement alternative-path disable for v1.4 ship + revisit in v1.5.

---

## Hypotheses Worth Pursuing Next

Top 3 candidates ranked by falsifier cost / coverage:

### Hypothesis 1: alloc_and_peer_copy_async stream split is THE root of cluster B

**Falsifier (≤4 hours):** Add fprintf probes inside cucascade `alloc_and_peer_copy_async` for the allocator stream pointer vs the memcpy stream pointer. If they differ, hypothesis confirmed → fix shape 1 (force same stream) closes cluster B.

**Cost:** 4 hours (cucascade rebuild + Sirius re-link + 1 sanitizer re-run).

### Hypothesis 2: cudf+kvikio fork-stream interaction is THE root of cluster A

**Falsifier (≤2 hours):** Run the same sanitizer command but with `kvikio.compat_mode=ON` (or equivalent kvikio environment variable to disable the BS::thread_pool fork) — see if cluster A races disappear. If yes, root is the kvikio thread-pool dispatch; workaround is to disable kvikio's threadpool (kvikio supports a "POSIX direct" mode that doesn't use BS::thread_pool).

**Cost:** 2 hours (env-var sweep + sanitizer re-run).

### Hypothesis 3: The races fire at SF1 but not SF10 because of buffer-size dependent code path branching

**Falsifier (≤1 hour):** Run the same sanitizer command on the SF10 Q11 num_gpus=2 case (which 20-04 PASSED un-sanitized). If SF10 also produces 21 race blocks, the failure-mode is timing-dependent (consistent with project_phase08_fu17's note that sanitizer reveals races that don't always crash). If SF10 has FEWER races, there's a size-threshold trigger and the workaround could be size-based gating.

**Cost:** 1 hour (sanitizer re-run with SF10 query).

---

## Carry-Forward to Phase 21 REG-03

**Phase 21 REG-03 ship-gate cannot pass until this resolves.** Per `.planning/ROADMAP.md` Phase 21 entry, REG-03 = "[integration][TPC-H] 48/48 PASS at num_gpus=2". The Q11 parquet failure is the SOLE blocker in 21/22 cases (`[integration][TPC-H]` aborts on first failure with `--abort` per Catch2 default).

**Two scenarios for Phase 21:**

1. **If cucascade fork+bump lands (preferred path):** Phase 21 REG-03 inherits a closed SF1 [integration][TPC-H] 48/48; Phase 21 ship-gate passes cleanly.

2. **If shipped with known limitation:** Phase 21 REG-03 acceptance criteria need to be relaxed to "47/48 + Q11 parquet num_gpus=2 known limitation" or "[integration][TPC-H] 48/48 at num_gpus=1 PASS". Document explicitly in 21-CONTEXT.md.

Either way, this 20-05-INVESTIGATION.md is the authoritative escalation document Phase 21 REG-03 plans against.

---

## Memory Update Recommendation

**Memory `project_phase08_fu17.md` should be updated with the following findings (post-confirmation):**

1. **Race signature cluster shift v1.3 → v1.4:** v1.3 had 433 races at the OLD `convert_gpu_to_gpu` peer-DMA path (Phase 13-02 finding). v1.4 has 21 races split across 2 clusters: 5 in cudf+kvikio internal (cluster A), 16 in cucascade host-staging fallback `alloc_and_peer_copy_async` (cluster B). Phase 13-04 fix preserved at the entry of `convert_gpu_to_gpu`; the residual races are at NEW sites that didn't exist in v1.3.

2. **DEAD trail addition: peer-DMA-path-itself.** The Phase 13-04 entry-level `cudaStreamWaitEvent(target_stream, src.get_writer_event(), 0)` IS firing correctly. The dominant residual cluster is in the host-staging fallback path that runs only when `[cucascade] direct GPU↔GPU peer DMA broken on N direction(s); cudaMemcpyPeer* will host-stage automatically.` triggers (consumer hardware). Server hardware with working peer DMA may not see cluster B at all.

3. **NEW live candidate to add:** "Run sanitizer on SF10 Q11 num_gpus=2 to compare race counts vs SF1; if SF10 has the same counts, the race is timing-dependent and any host-staging fallback path triggers it."

4. **Path forward for next session:** "Either fork cucascade and fix `alloc_and_peer_copy_async` to use a single stream for allocate + memcpy, OR ship v1.4 with parquet num_gpus>=2 SF1 known limitation."

---

## Files Touched (Path B; documentation-only)

- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-DIAGNOSIS.md` (Task 1 + Task 2 path-gate annotation)
- `.planning/phases/20-scan-manager-pin-tables-port-pr-731-pr-721/20-05-INVESTIGATION.md` (this file; Task 3 escalation document)

**No source files modified.** Phase 18..20 invariants preserved per pre-flight + post-fix snapshots in DIAGNOSIS.md.

---

STATUS: human_needed
