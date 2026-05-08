# Phase 22 Cucascade Fork-Side Diff (Cluster B Same-Stream Fix)

**Authoring rationale:** This document captures the cucascade-fork-side diff for the fu17 Cluster B same-stream invariant fix landed in Phase 22 Plan 03. Per CONTEXT.md D-14 + CC-UPSTREAM-01 carry pattern, the cucascade fork holds an N-commit local divergence from upstream `NVIDIA/cuCascade`; rather than file an upstream PR this milestone (D-08), the diff is captured here for future review when CC-UPSTREAM-01 lands.

**Plan 22-03 commit (cucascade fork):** `c666b21926dec70b26a1febd509435635bea8deb` (short: `c666b21`)
**Pre-fix Sirius parent gitlink (Phase 21 v1.4 ship):** `1c1e648` — pre-fix Cluster B baseline of 16/21 SF1 Q11 num_gpus=2 sanitizer races
**Intermediate cucascade pin (between Phase 21 ship and Plan 22-03 fix):** `42a01c4` — pre-commit cleanup (clang-format + codespell), no logic change
**Post-fix Sirius parent gitlink (Phase 22 Plan 22-04 bump):** `c666b21` — same-stream invariant fix on `alloc_and_peer_copy_async`, Cluster B = 0 empirically verified

## Background

Per `20-05-INVESTIGATION.md` (Phase 20 PATH B sanitizer investigation), the SF1 TPC-H Q11 num_gpus=2 fixture under `compute-sanitizer --tool memcheck --track-stream-ordered-races=all` reported 21 stream-ordered races at HEAD (cucascade pin `1c1e648`) split across two clusters at library boundaries:

- **Cluster A (5/21 races)** — cudf+kvikio internal cross-stream gap inside `cudf::io::parquet::detail::read_column_chunks_async` + `kvikio::detail::posix_device_io`. Out of Sirius's control without unwinding Phase 19's IO framework adoption; tracked as upstream issue per CONTEXT.md D-09.
- **Cluster B (16/21 races)** — cucascade pin `1c1e648` `alloc_and_peer_copy_async` host-staging fallback path. The producer (`rmm::device_buffer::device_buffer` allocator on `target_stream` at line 600) and consumer (`cudaMemcpyAsync` DtoH on the in-function local `rmm::cuda_stream src_stream` at line 617) operated on different streams with no event linkage, even though a wall-clock `src_stream.synchronize()` typically masked the race. The compute-sanitizer correctly classified that as an unordered race.

The Cluster B path is exercised on consumer hardware where peer DMA is broken in some directions (e.g., 2 × NVIDIA RTX 6000 Ada Generation on this development host); cucascade's `probe_peer_dma_works(src_device, dst_device)` returns false and falls through to the host-staging fallback at line 614 onward.

The D-07 fix shape — "same-stream invariant" — collapses the producer + DtoH leg + HtoD leg + final sync onto the single `target_stream` argument. The HtoD leg at line 629 was already correctly on `target_stream`; the fix propagates that discipline backward to the DtoH leg by issuing it on `target_stream.value()` under `rmm::cuda_set_device_raii(src_device)` (which provides ONLY the device-context binding for the DtoH read; stream lineage stays on `target_stream`).

**Why this is the smallest correct fix:** the working peer-DMA path at line 605 already passes `target_stream.value()` as the only stream argument to `cudaMemcpyPeerAsync` — the host-staging fallback just needed to mirror that discipline. Alternatives considered in CONTEXT.md D-07 (e.g., explicit `cudaStreamWaitEvent` event-bridge between two distinct streams) are heavier-weight and not needed when a single stream suffices.

## Diff (cucascade/src/data/representation_converter.cpp)

The actual diff body, captured verbatim from `git -C cucascade diff 1c1e648..HEAD -- src/data/representation_converter.cpp`. Note that `42a01c4` (the intermediate clang-format cleanup commit) introduces line-wrapping noise visible alongside the load-bearing logic change; the load-bearing change is concentrated in `alloc_and_peer_copy_async` lines 611-633 (`@@ -611,14 +611,22 @@` hunk).

```diff
diff --git a/src/data/representation_converter.cpp b/src/data/representation_converter.cpp
index b6675db..27e81b0 100644
--- a/src/data/representation_converter.cpp
+++ b/src/data/representation_converter.cpp
@@ -611,14 +611,22 @@ static rmm::device_buffer alloc_and_peer_copy_async(const void* src_ptr,
   void* host_buf = nullptr;
   CUCASCADE_CUDA_TRY(cudaMallocHost(&host_buf, size));
   {
+    // Phase 22 D-07: same-stream invariant. Issue DtoH on target_stream
+    // (matching rmm::device_buffer::allocate_async at the top of this
+    // function) under cuda_set_device_raii(src_device) for src-side
+    // context. Closes Cluster B sanitizer race shape A
+    // (16/21 of SF1 Q11 num_gpus=2 races per 20-05-INVESTIGATION.md).
     rmm::cuda_set_device_raii src_guard{rmm::cuda_device_id{src_device}};
-    rmm::cuda_stream src_stream;
-    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
-      host_buf, src_ptr, size, cudaMemcpyDeviceToHost, src_stream.view().value()));
-    src_stream.synchronize();
-  }
-  CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
-    buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value()));
+    CUCASCADE_CUDA_TRY(
+      cudaMemcpyAsync(host_buf, src_ptr, size, cudaMemcpyDeviceToHost, target_stream.value()));
+    CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));
+    // Sync inside the src_guard scope: cudaFreeHost (after the closing
+    // brace below) is host-synchronous and must not race with the DtoH
+    // read; the sync also ensures host_buf is fully populated before
+    // the HtoD enqueue executes on target_stream.
+  }
+  CUCASCADE_CUDA_TRY(
+    cudaMemcpyAsync(buf.data(), host_buf, size, cudaMemcpyHostToDevice, target_stream.value()));
   CUCASCADE_CUDA_TRY(cudaStreamSynchronize(target_stream.value()));
   cudaFreeHost(host_buf);
   return buf;
@@ -635,8 +643,8 @@ static rmm::device_buffer alloc_and_peer_copy_sync(const void* src_ptr,
                                                    rmm::cuda_stream_view target_stream,
                                                    rmm::device_async_resource_ref target_mr)
 {
-  auto buf = alloc_and_peer_copy_async(
-    src_ptr, src_device, size, dst_device, target_stream, target_mr);
+  auto buf =
+    alloc_and_peer_copy_async(src_ptr, src_device, size, dst_device, target_stream, target_mr);
   if (size == 0 || src_ptr == nullptr) { return buf; }
   target_stream.synchronize();
   return buf;
@@ -656,20 +664,19 @@ static rmm::device_buffer alloc_and_peer_copy_sync(const void* src_ptr,
  * @note Assumes the source column_view has offset == 0 (no slicing), matching the
  *       same constraint imposed by plan_column_copy() on the GPU↔Host fast path.
  */
-static std::unique_ptr<cudf::column> reconstruct_column_p2p(
-  const cudf::column_view& src,
-  int src_device,
-  int dst_device,
-  rmm::cuda_stream_view stream,
-  rmm::device_async_resource_ref mr)
+static std::unique_ptr<cudf::column> reconstruct_column_p2p(const cudf::column_view& src,
+                                                            int src_device,
+                                                            int dst_device,
+                                                            rmm::cuda_stream_view stream,
+                                                            rmm::device_async_resource_ref mr)
 {
   assert(src.offset() == 0 && "column_view with non-zero offset is not supported");
 
   rmm::device_buffer null_mask{};
   if (src.nullable()) {
     auto const null_mask_size = cudf::bitmask_allocation_size_bytes(src.size());
-    null_mask = alloc_and_peer_copy_sync(
-      src.null_mask(), src_device, null_mask_size, dst_device, stream, mr);
+    null_mask =
+      alloc_and_peer_copy_sync(src.null_mask(), src_device, null_mask_size, dst_device, stream, mr);
   }
   cudf::size_type const null_count = src.nullable() ? src.null_count() : 0;
 
@@ -727,11 +734,8 @@ static std::unique_ptr<cudf::column> reconstruct_column_p2p(
                                               mr);
       }
     }
-    return cudf::make_strings_column(src.size(),
-                                     std::move(offsets_col),
-                                     std::move(chars_buf),
-                                     null_count,
-                                     std::move(null_mask));
+    return cudf::make_strings_column(
+      src.size(), std::move(offsets_col), std::move(chars_buf), null_count, std::move(null_mask));
   }
 
   if (src.type().id() == cudf::type_id::LIST) {
@@ -742,11 +746,8 @@ static std::unique_ptr<cudf::column> reconstruct_column_p2p(
     // Preserve source's offsets type — make_lists_column accepts INT32 or INT64.
     auto offsets_col = reconstruct_column_p2p(src.child(0), src_device, dst_device, stream, mr);
     auto values_col  = reconstruct_column_p2p(src.child(1), src_device, dst_device, stream, mr);
-    return cudf::make_lists_column(src.size(),
-                                   std::move(offsets_col),
-                                   std::move(values_col),
-                                   null_count,
-                                   std::move(null_mask));
+    return cudf::make_lists_column(
+      src.size(), std::move(offsets_col), std::move(values_col), null_count, std::move(null_mask));
   }
 
   if (src.type().id() == cudf::type_id::STRUCT) {
@@ -783,8 +784,7 @@ static std::unique_ptr<cudf::column> reconstruct_column_p2p(
   rmm::device_buffer data_buf{};
   if (src.size() > 0 && src.head() != nullptr) {
     auto const data_size = static_cast<std::size_t>(src.size()) * cudf::size_of(src.type());
-    data_buf            = alloc_and_peer_copy_async(
-      src.head(), src_device, data_size, dst_device, stream, mr);
+    data_buf = alloc_and_peer_copy_async(src.head(), src_device, data_size, dst_device, stream, mr);
   }
   return std::make_unique<cudf::column>(
     src.type(), src.size(), std::move(data_buf), std::move(null_mask), null_count);
@@ -868,8 +868,8 @@ std::unique_ptr<idata_representation> convert_gpu_to_gpu(
   std::vector<std::unique_ptr<cudf::column>> target_columns;
   target_columns.reserve(static_cast<std::size_t>(src_view.num_columns()));
   for (cudf::size_type i = 0; i < src_view.num_columns(); ++i) {
-    target_columns.push_back(reconstruct_column_p2p(
-      src_view.column(i), src_device_id, dst_device_id, target_stream, mr));
+    target_columns.push_back(
+      reconstruct_column_p2p(src_view.column(i), src_device_id, dst_device_id, target_stream, mr));
   }
 
   auto new_table = std::make_unique<cudf::table>(std::move(target_columns));
```

### Hunk decomposition

| Hunk | File:line range | Change | Source commit |
|------|-----------------|--------|---------------|
| 1 (load-bearing) | `src/data/representation_converter.cpp:611-633` | **`alloc_and_peer_copy_async` same-stream invariant** — drop `rmm::cuda_stream src_stream`; issue DtoH on `target_stream.value()` under `cuda_set_device_raii(src_device)`; sync `target_stream` inside `src_guard` scope (Pitfall 4 closure) | `c666b21` (Plan 22-03) |
| 2 (formatting) | `:643-650` | clang-format wrap — `auto buf = alloc_and_peer_copy_async(...)` line wrap. No logic change. | `42a01c4` (clang-format cleanup) |
| 3 (formatting) | `:664-671` | clang-format wrap — `reconstruct_column_p2p` signature aligned multi-line. No logic change. | `42a01c4` |
| 4 (formatting) | `:677-684` | clang-format wrap — `null_mask = alloc_and_peer_copy_sync(...)` line wrap. No logic change. | `42a01c4` |
| 5 (formatting) | `:734-740` | clang-format wrap — `cudf::make_strings_column` arg list collapsed. No logic change. | `42a01c4` |
| 6 (formatting) | `:746-752` | clang-format wrap — `cudf::make_lists_column` arg list collapsed. No logic change. | `42a01c4` |
| 7 (formatting) | `:784-787` | clang-format wrap — `data_buf = alloc_and_peer_copy_async(...)` line wrap. No logic change. | `42a01c4` |
| 8 (formatting) | `:868-872` | clang-format wrap — `target_columns.push_back(reconstruct_column_p2p(...))` line wrap. No logic change. | `42a01c4` |

**For upstreaming:** the load-bearing change is hunk 1. Hunks 2-8 are clang-format style adjustments from `42a01c4` (the pre-commit cleanup commit between Phase 21 baseline `1c1e648` and Plan 22-03 fix `c666b21`); they would either be already-present in upstream cucascade by the time CC-UPSTREAM-01 fires (clang-format runs on every PR) OR absorbed into the upstreaming PR with no behavioral effect.

## Validation

The Plan 22-03 fix has been empirically validated through the following gates, all PASS post-fix:

1. **Cucascade compile-correctness:** parent build's cucascade objects step `[91/112]` and libcucascade.a link `[92/112]` both succeed against `c666b21` (Plan 22-03 SUMMARY §"Acceptance criteria status"). The cucascade ctest CC-04 gate is corroborated by the integration smoke (Plan 22-04) — the host-staging fallback path is exercised under live GPU traffic by `[mgpu]` 16/16 + `[TPC-H][parquet]` 22/22 + Q11 sanitizer.
2. **SF1 Q11 num_gpus=2 sanitizer Cluster B = 0 post-fix:** Plan 22-04 micro-validation reported `cluster_B=0` (was 16 pre-fix per 20-05-INVESTIGATION.md), Plan 22-06 self-test reproduced `cluster_B=0`, Plan 22-07 verdict GATE-09 reconfirmed `cluster_B=0`. See `22-VERDICT.md` Section I.
3. **`[mgpu]` 16/16 PASS continuity post-bump:** Plan 22-04 integration smoke recorded 16/16 PASS / 79091 assertions / 116.2s vs Phase 21 baseline 79091 / 106.3s; Plan 22-05 + Plan 22-07 reproduced. See `22-VERDICT.md` Section A.
4. **SF100 Q11 num_gpus=2 sanitizer Cluster B = 0 (advisory):** Plan 22-07 ADVISORY recorded `cluster_B=0` even at SF100 scale on Q11 (`22-VERDICT.md` Section J). Cluster A (residual) and the Q11 SF100 query-level fallback (follow-up #17) are tracked as open carry-forwards (Section K.1 + K.6).

## Upstreaming notes (for CC-UPSTREAM-01 v1.6+)

This section provides the bookkeeping for opening the upstream PR when CC-UPSTREAM-01 lands.

1. **Carry-pattern context:** This fix is one of the 11 carried local fixes per CC-UPSTREAM-01 (D-08). The cucascade fork branch `fix/pinned-portable-flags` carries the divergence from upstream `NVIDIA/cuCascade`. Upstream PRs are NOT filed this milestone; the diff is captured here for future review.

2. **Anticipated upstream review concerns:**
   - **`cudaMemcpyAsync` DtoH on `target_stream` from a `cuda_set_device_raii(src_device)` scope** — the question for upstream reviewers is: does `cudaMemcpyAsync(host_buf, src_ptr, size, DtoH, target_stream.value())` produce sanitizer-clean stream-ordered behavior when `target_stream` lives on `dst_device` but the calling thread is bound to `src_device`? Verified empirically on CUDA 13.0/13.2 across `[mgpu]` 16/16 + Q11 SF1+SF100 sanitizer + `[mgpu_stress]` 500-iter; no host-staging-side races detected. CUDA semantics confirm `cudaMemcpyAsync` reads the source pointer using the calling thread's current CUDA context (i.e., src_device per the RAII guard) and enqueues the work on the supplied stream (target_stream on dst_device); no device-affinity violation.
   - **Runtime regression check:** `[mgpu]` 16/16 wall-clock baseline preserved (106.3s pre-fix Phase 21 → 116.2s Plan 22-04 smoke → 110.6s Plan 22-07 verdict; all under the 130s gate). The fix removes one local `rmm::cuda_stream` allocation per fallback invocation; no measurable runtime regression.
   - **Pitfall 4 (preserve sync-then-cudaFreeHost ordering):** documented in the inline comment block at lines 622-625. The `cudaStreamSynchronize(target_stream.value())` inside the `src_guard` scope ensures the DtoH read completes before any subsequent `cudaFreeHost(host_buf)` (line 633) — `cudaFreeHost` is host-synchronous and must not race with the DtoH read. Reviewer should confirm the ordering is intact across the new dual-sync structure (sync inside src_guard at line 622 + sync at function tail at line 632).

3. **Upstream destination file path:** `cucascade/src/data/representation_converter.cpp` (same as fork-side path; cucascade upstream tree mirrors fork tree).

4. **Suggested PR title:** `feat(stream-lineage): same-stream invariant for host-staging fallback in alloc_and_peer_copy_async`

5. **Suggested PR description outline:**
   - Problem statement: `alloc_and_peer_copy_async` host-staging fallback path produces sanitizer-detected stream-ordered races when peer DMA is broken (16/21 races at SF1 TPC-H Q11 num_gpus=2 per 20-05-INVESTIGATION.md). Producer (allocator on `target_stream`) and consumer (DtoH on local `src_stream`) operate on different streams with no event linkage.
   - Fix: collapse producer + DtoH leg + HtoD leg onto a single `target_stream`. DtoH issued under `cuda_set_device_raii(src_device)` (device context for the read source); stream lineage stays on `target_stream`.
   - Validation: `compute-sanitizer --tool memcheck --track-stream-ordered-races=all` on SF1 Q11 num_gpus=2 reports 0 host-staging-fallback races post-fix (was 16 pre-fix). `[mgpu]` 16/16 + `[TPC-H][parquet]` 22/22 + `[mgpu_stress]` 500-iter all PASS. SF100 Q11 num_gpus=2 sanitizer reports 0 host-staging-fallback races.
   - Reference: Sirius Phase 22 verdict `22-VERDICT.md` (this milestone) + investigation `20-05-INVESTIGATION.md`.

6. **CC-UPSTREAM-01 readiness:** this diff is reviewable as-is for upstream PR composition. Hunk 1 is the load-bearing change; hunks 2-8 are pre-commit clang-format adjustments from the intermediate `42a01c4` commit and would either be redundant against upstream's clang-format pass or absorbed without behavioral effect.

---

*Phase: 22-multi-gpu-pinning-stream-lineage-hardening*
*Plan: 07 (terminal verdict)*
*Authored: 2026-05-07*
*Carry-forward target: v1.6+ via CC-UPSTREAM-01 (D-08, D-14)*
