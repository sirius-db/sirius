/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// RLE decode kernel.
//
// Pipeline per `decode_rle_data` call (one column's RLE run):
//   1. D2H every segment's 8-byte header in parallel; one stream sync.
//   2. D2H every segment's counts area in parallel; one stream sync.
//   3. On host, walk counts → inclusive prefix-sum array per segment. A
//      malformed walk (overflow, underflow, count==0) demotes the segment
//      to a zero-fill chunk; the kernel handles both shapes uniformly.
//   4. Concatenate per-segment cumsum arrays into one column-wide buffer,
//      H2D it. Build per-CTA chunk descriptors (one per RLE_ROWS_PER_CHUNK
//      slice of each segment), H2D them.
//   5. Launch one batched kernel: each CTA expands one chunk by binary-
//      searching its slice's row indices in the segment's cumsum, gathering
//      the matching value, and storing to global memory.
//
// Per-CTA shmem caches the cumsum array when entry_count <= RLE_SMEM_MAX_
// ENTRIES (16 KiB cap). Larger segments fall back to gmem-resident cumsum;
// L2 caches it well enough since CTAs operating on adjacent row chunks of
// the same segment touch the same array. The two paths each call
// `rle_upper_bound` with its address-space-specific pointer (shared vs.
// global) so the inlined loads specialise to `ld.shared` / `ld.global`
// respectively rather than the slower generic `ld.u32` the compiler would
// emit if both pointers fed a single dereference site.
//
// Output is written once and never reread within a single kernel, so global
// stores go through `__stwt`. Value gathers go through `__ldg` (the read-
// only / non-coherent global load) — RLE's broadcast-heavy access pattern
// (many threads in a warp hit the same entry within a long run) benefits
// from the per-SM read-only cache on Turing/Ampere.
//
// Phase-2 follow-ups, sorted by RLE-specificity:
//
// RLE-specific opportunities (high-confidence wins, not in
// ref_decode_kernel_patterns.md because they exploit run structure):
//
//  - Per-CTA narrow-window binary search: a 2048-row chunk spans far fewer
//    entries than the whole segment (cumsum length). Thread 0 binary-searches
//    cumsum for `local_row_start`; thread blockDim.x-1 searches for the
//    chunk's last row; broadcast (entry_lo, entry_hi) via __shfl_sync; all
//    threads then search inside cumsum[entry_lo..entry_hi]. Drops binary-
//    search levels from log2(ec) to log2(ec_per_chunk) — typically 12 → 6.
//
//  - Run-coalesced warp-vote fast path: when 32 consecutive output rows of a
//    warp all live in the same RLE entry (long-run shape — booleans, sorted
//    low-cardinality columns), `__ballot_sync(my_entry == lane_0_entry)`
//    detects it; one thread looks up the value and `__shfl_sync` broadcasts.
//    Workload-dependent: a no-op-or-worse on high-entry-count shapes where
//    runs are <32 rows. Bench-gate before merge — measure entry-sharing
//    frequency on real TPC-H RLE columns.
//
// Inherited from ref_decode_kernel_patterns.md (verified to apply to RLE,
// alternatives flagged):
//
//  - cp.async stage of cumsum on sm_80+ with uint4 vector loads. Caveat:
//    per-segment cumsum slices may not be 16-byte aligned within the
//    concatenated buffer (alignment depends on segment_cumsum_offset * 4),
//    so the uint4 path needs a scalar head/tail fallback or per-segment
//    pad-rounding.
//  - For Turing (sm_75, no cp.async): __ldg on the cumsum read inside
//    rle_upper_bound's gmem path. Binary-search early levels converge across
//    the warp → broadcast load → texture-cache route is the right cache.
//  - Three-way ec dispatch instead of the current two-way:
//      ec ≤ ~128       → __ldg from gmem direct (L1 absorbs the 512 B cumsum,
//                         skip shmem-load + __syncthreads overhead)
//      ec ≤ ~4096      → stage to shmem (current code path)
//      ec >  4096      → __ldg from gmem (current code path)
//    The tiny-ec case is currently subsumed into the shmem path; for short
//    cumsums the per-CTA __syncthreads + 1-iter shmem load probably costs
//    more than direct L1-resident reads would.
//===----------------------------------------------------------------------===//

#include "cuda/scan/gpu_decode_rle.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

/// Block dim used by the RLE expand kernel.
constexpr uint32_t BLOCK_DIM = 256;

/// Rows decoded per CTA. Picked so each CTA does VPT=8 rows per thread —
/// enough to amortise the binary-search prologue without overflowing
/// per-CTA register/shmem budgets.
constexpr uint32_t RLE_ROWS_PER_CHUNK = 2048;

/// VPT (values per thread) is the number of rows each thread handles per CTA.
/// Tied to BLOCK_DIM and RLE_ROWS_PER_CHUNK; static_assert keeps them aligned.
constexpr uint32_t VPT = RLE_ROWS_PER_CHUNK / BLOCK_DIM;
static_assert(BLOCK_DIM * VPT == RLE_ROWS_PER_CHUNK,
              "BLOCK_DIM and VPT must tile RLE_ROWS_PER_CHUNK exactly");

/// Maximum cumsum entries cached in per-CTA shared memory (16 KiB / 4 B).
/// Above this the kernel binary-searches gmem directly. Picked so per-CTA
/// shmem stays within the 48 KiB default cap with room for occupancy.
constexpr uint32_t RLE_SMEM_MAX_ENTRIES = 4096;
constexpr uint32_t RLE_SMEM_BYTES       = RLE_SMEM_MAX_ENTRIES * sizeof(uint32_t);

/// Per-CTA work unit. One descriptor describes one row slice within one
/// segment; chunks of the same segment share `d_values` / `d_cumsum`.
///
/// `entry_count == 0` is the zero-fill sentinel — the host marks malformed
/// segments this way and the kernel deterministically clears the chunk's
/// row range without touching `d_values` / `d_cumsum`.
struct rle_chunk_desc {
  uint8_t const* d_values;     ///< Device pointer to segment's values array.
  uint32_t const* d_cumsum;    ///< Device pointer to inclusive prefix sums.
  uint32_t entry_count;        ///< Cumsum length; 0 → zero-fill chunk.
  uint32_t base_global_row;    ///< Output offset (in rows) of segment row 0.
  uint32_t local_row_start;    ///< First row of this chunk within the segment.
  uint32_t chunk_rows;         ///< Rows this chunk handles.
};

//===----------------------------------------------------------------------===//
// Device: upper_bound on inclusive prefix sums.
//
// Returns the first index where `cumsum[idx] > key`. For an output row `r`,
// that index identifies the RLE entry that produced row `r` (since
// cumsum[i] = sum(counts[0..i]) and counts[i] is the run length for entry i).
//
// The kernel calls this from each thread at every iteration; each call is
// O(log entry_count). The branches are data-dependent so threads in a warp
// usually take different paths — but rows in a warp tend to land in the
// same or adjacent runs, so the early iterations stay convergent and only
// the final 1-2 levels diverge in practice.
//===----------------------------------------------------------------------===//

__device__ __forceinline__ uint32_t rle_upper_bound(uint32_t const* __restrict__ cumsum,
                                                    uint32_t n,
                                                    uint32_t key)
{
  uint32_t lo = 0;
  uint32_t hi = n;
  while (lo < hi) {
    uint32_t mid = lo + ((hi - lo) >> 1);
    if (cumsum[mid] <= key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

//===----------------------------------------------------------------------===//
// Batched RLE expand kernel.
//
// Two paths inside one kernel:
//   shmem path  (entry_count <= RLE_SMEM_MAX_ENTRIES) — cumsum is loaded
//               cooperatively into per-CTA shmem; binary search hits shmem.
//   gmem path   (entry_count >  RLE_SMEM_MAX_ENTRIES) — cumsum stays in
//               global memory and the binary search reads from there. L2
//               caches it across the segment's CTAs since they all hit the
//               same array.
//
// The two paths share the unpack loop layout (one thread per output row,
// striped across blockDim.x threads, VPT iterations to cover the chunk).
// A single `__syncthreads` at the end of the shmem load lifts the cumsum
// into visibility for every thread; gmem-path CTAs skip the sync.
//===----------------------------------------------------------------------===//

template <typename T>
__global__ void kernel_decode_rle(rle_chunk_desc const* __restrict__ descs,
                                  T* __restrict__ d_output,
                                  uint32_t num_chunks)
{
  uint32_t const cid = blockIdx.x;
  if (cid >= num_chunks) return;

  auto const desc          = descs[cid];
  uint32_t const rc        = desc.chunk_rows;
  T* out_chunk             = d_output + desc.base_global_row + desc.local_row_start;

  //===--------------------------------------------------------------------===//
  // Zero-fill path. The host marks malformed segments with entry_count == 0;
  // this branch covers them without dereferencing the (possibly null /
  // unsanitised) value or cumsum pointers. Use the trusted descriptor row
  // count — leaving the chunk uninitialised would expose prior device
  // contents (the same information-disclosure concern the BITPACKING kernel
  // addresses with its INVALID-mode zero-fill).
  //===--------------------------------------------------------------------===//
  if (desc.entry_count == 0) {
    for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
      __stwt(out_chunk + i, T(0));
    }
    return;
  }

  uint32_t const ec       = desc.entry_count;
  T const* values         = reinterpret_cast<T const*>(desc.d_values);
  uint32_t const lr_start = desc.local_row_start;

  //===--------------------------------------------------------------------===//
  // Striped expand. In iteration v, threads 0..blockDim.x-1 cover rows
  // [v*blockDim.x .. (v+1)*blockDim.x) of the chunk. Output stores are
  // contiguous per warp (one coalesced 128-byte transaction per warp per
  // iteration when sizeof(T) == 4).
  //
  // Each thread computes its absolute-within-segment row, binary-searches
  // cumsum for the owning entry, then gathers the value. We clamp the
  // upper_bound result to ec-1 as a defensive backstop — the host-built
  // cumsum's last entry equals the segment's row_count by construction, so
  // any in-range `local_row` MUST have a strict-greater entry; clamping
  // covers the edge case where rounding produces lo == ec on a malformed
  // tail without crashing the kernel.
  //
  // The two paths below are textually duplicated to keep the address space
  // visible at the binary-search call site; merging them via a unified
  // pointer would force `ld.u32 [generic]` (slower).
  //===--------------------------------------------------------------------===//
  if (ec <= RLE_SMEM_MAX_ENTRIES) {
    extern __shared__ uint32_t s_cumsum[];
    for (uint32_t i = threadIdx.x; i < ec; i += blockDim.x) {
      s_cumsum[i] = desc.d_cumsum[i];
    }
    __syncthreads();

#pragma unroll
    for (uint32_t v = 0; v < VPT; ++v) {
      uint32_t const i = v * blockDim.x + threadIdx.x;
      if (i >= rc) break;

      uint32_t const local_row = lr_start + i;
      uint32_t entry           = rle_upper_bound(s_cumsum, ec, local_row);
      if (entry >= ec) entry = ec - 1;

      __stwt(out_chunk + i, __ldg(values + entry));
    }
    return;
  }

  // gmem path — entry_count > RLE_SMEM_MAX_ENTRIES, cumsum stays in global.
#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    uint32_t const i = v * blockDim.x + threadIdx.x;
    if (i >= rc) break;

    uint32_t const local_row = lr_start + i;
    uint32_t entry           = rle_upper_bound(desc.d_cumsum, ec, local_row);
    if (entry >= ec) entry = ec - 1;

    __stwt(out_chunk + i, __ldg(values + entry));
  }
}

//===----------------------------------------------------------------------===//
// Per-segment host parsing.
//
// Two phases of D2H copies, one stream sync each. Phase 1 reads each
// segment's 8-byte header; phase 2 reads each segment's counts area.
// Walking the counts array on host gives us the inclusive prefix sums + a
// definitive entry_count without ever launching a device kernel just to
// scan the counts.
//===----------------------------------------------------------------------===//

/// Per-segment parse output. `entry_count == 0` flags a malformed segment
/// (any of: rle_count_offset out of range, walk doesn't reach row_count,
/// count == 0 inside the walk, walk overflows the counts area). The kernel
/// treats these as zero-fill chunks.
struct seg_parsed {
  std::vector<uint32_t> cumsum;  ///< Inclusive prefix sums; empty when malformed.
  uint8_t const* d_values;       ///< Cached for descriptor build.
  uint32_t base_global_row;      ///< Cached for descriptor build.
  uint32_t row_count;             ///< Trusted from the host descriptor.
};

/// Parse one segment's host-staged counts buffer into prefix sums. Returns
/// an empty `cumsum` vector to mark the segment malformed.
std::vector<uint32_t> walk_counts_to_cumsum(uint16_t const* counts,
                                            size_t counts_capacity_entries,
                                            uint32_t row_count)
{
  std::vector<uint32_t> cumsum;
  cumsum.reserve(std::min<size_t>(counts_capacity_entries, 256));

  uint32_t total = 0;
  for (size_t i = 0; i < counts_capacity_entries; ++i) {
    uint16_t c = counts[i];
    if (c == 0) {
      // DuckDB's encoder never writes zero counts. A zero here means the
      // counts area is malformed (or we walked past the live entries).
      // Either way, refuse rather than infinite-loop or under-fill.
      return {};
    }
    total += c;
    cumsum.push_back(total);
    if (total >= row_count) break;
  }

  // Walk must reach exactly `row_count`. Less means the counts area was
  // truncated; more means the on-disk run sizes overshoot the descriptor's
  // row count.
  if (total != row_count) return {};
  return cumsum;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public entry.
//===----------------------------------------------------------------------===//

void decode_rle_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type /*type*/,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  // Bit-level expand is independent of element signedness; route by size
  // alone (mirrors BITPACKING). {1,2,4,8} cover INT8/16/32/64, UINT8/16/32/64,
  // FLOAT, DOUBLE, DATE, TIME, TIMESTAMP. INT128 (HUGEINT / DECIMAL128) is
  // refused upstream by the viability walker; throw here as a defensive
  // backstop with the same wording the dispatcher uses elsewhere.
  if (type_size != 1 && type_size != 2 && type_size != 4 && type_size != 8) {
    throw std::runtime_error(
      "gpu_decode_table: viability invariant violated — RLE type_size " +
      std::to_string(type_size));
  }

  // Drop empty segments — they contribute no chunks and would otherwise
  // confuse the bounds checks below. Segments with bytes_size <
  // RLE_HEADER_SIZE stay in the live list; they'll skip both D2H phases and
  // surface as malformed (zero-fill) at descriptor time.
  std::vector<gpu_segment_desc const*> live;
  live.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    live.push_back(&seg);
  }
  if (live.empty()) return;

  size_t const n = live.size();

  //===--------------------------------------------------------------------===//
  // Phase 1: D2H every segment's 8-byte header. Issued back-to-back on the
  // same stream so the runtime can coalesce them; one sync drains all.
  //===--------------------------------------------------------------------===//
  std::vector<uint64_t> h_offsets(n, 0);
  for (size_t i = 0; i < n; ++i) {
    auto const& seg = *live[i];
    if (seg.bytes_size < RLE_HEADER_SIZE) continue;  // skip; will be malformed
    RMM_CUDA_TRY(cudaMemcpyAsync(&h_offsets[i],
                                 seg.d_bytes,
                                 sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  //===--------------------------------------------------------------------===//
  // Phase 2: D2H every segment's counts area. Bounds-check rle_count_offset
  // against bytes_size; segments with a bad header are queued as zero-fill
  // (no D2H issued for them).
  //===--------------------------------------------------------------------===//
  std::vector<std::vector<uint8_t>> h_count_buffers(n);
  for (size_t i = 0; i < n; ++i) {
    auto const& seg = *live[i];
    if (seg.bytes_size < RLE_HEADER_SIZE) continue;
    uint64_t off = h_offsets[i];
    // rle_count_offset must point past the header and within the segment.
    // Equality with bytes_size means no counts at all → malformed.
    if (off < RLE_HEADER_SIZE || off >= seg.bytes_size) continue;
    size_t counts_bytes = size_t{seg.bytes_size} - off;
    h_count_buffers[i].resize(counts_bytes);
    RMM_CUDA_TRY(cudaMemcpyAsync(h_count_buffers[i].data(),
                                 seg.d_bytes + off,
                                 counts_bytes,
                                 cudaMemcpyDeviceToHost,
                                 stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

  //===--------------------------------------------------------------------===//
  // Host: parse counts → inclusive prefix sums per segment. Concatenate
  // into one column-wide cumsum buffer for a single H2D upload.
  //===--------------------------------------------------------------------===//
  std::vector<seg_parsed> parsed(n);
  std::vector<uint32_t> h_concat_cumsum;
  std::vector<size_t> seg_cumsum_offset(n, 0);  // start index into h_concat_cumsum

  for (size_t i = 0; i < n; ++i) {
    auto const& seg                = *live[i];
    parsed[i].base_global_row      = seg.row_offset;
    parsed[i].row_count            = seg.row_count;
    parsed[i].d_values             = seg.d_bytes + RLE_HEADER_SIZE;

    if (h_count_buffers[i].empty()) {
      // Bad header / no counts area — leave cumsum empty (zero-fill marker).
      seg_cumsum_offset[i] = h_concat_cumsum.size();
      continue;
    }

    auto const* counts =
      reinterpret_cast<uint16_t const*>(h_count_buffers[i].data());
    size_t capacity_entries = h_count_buffers[i].size() / sizeof(uint16_t);

    parsed[i].cumsum =
      walk_counts_to_cumsum(counts, capacity_entries, seg.row_count);

    // Sanity: values area must hold at least entry_count values. If the
    // walk produced N entries but rle_count_offset only leaves room for
    // M < N values before counts begin, the segment is malformed.
    if (!parsed[i].cumsum.empty()) {
      uint64_t off          = h_offsets[i];
      size_t values_bytes   = static_cast<size_t>(off) - RLE_HEADER_SIZE;
      size_t needed_values  = parsed[i].cumsum.size() * size_t{type_size};
      if (values_bytes < needed_values) parsed[i].cumsum.clear();
    }

    seg_cumsum_offset[i] = h_concat_cumsum.size();
    h_concat_cumsum.insert(h_concat_cumsum.end(),
                           parsed[i].cumsum.begin(),
                           parsed[i].cumsum.end());
  }

  //===--------------------------------------------------------------------===//
  // H2D the concatenated cumsum first so we have a real device pointer to
  // base each per-segment slice off when we build descriptors below.
  //===--------------------------------------------------------------------===//
  rmm::device_uvector<uint32_t> d_concat_cumsum(h_concat_cumsum.size(), stream, mr);
  if (!h_concat_cumsum.empty()) {
    RMM_CUDA_TRY(cudaMemcpyAsync(d_concat_cumsum.data(),
                                 h_concat_cumsum.data(),
                                 h_concat_cumsum.size() * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice,
                                 stream.value()));
  }

  //===--------------------------------------------------------------------===//
  // Build per-CTA chunk descriptors. Every segment contributes
  // ceil(row_count / RLE_ROWS_PER_CHUNK) chunks; malformed segments produce
  // chunks with entry_count == 0 so the kernel zero-fills them uniformly.
  //===--------------------------------------------------------------------===//
  std::vector<rle_chunk_desc> h_descs;
  // Average ~1 chunk/segment for typical row-group-sized segments; ×2 keeps
  // re-allocs out of the way on workloads with smaller segments.
  h_descs.reserve(n * 2);

  for (size_t i = 0; i < n; ++i) {
    uint32_t const rc       = parsed[i].row_count;
    uint32_t const num_cnks = (rc + RLE_ROWS_PER_CHUNK - 1) / RLE_ROWS_PER_CHUNK;
    bool const malformed    = parsed[i].cumsum.empty();
    uint32_t const* d_cs    = malformed ? nullptr
                                        : d_concat_cumsum.data() + seg_cumsum_offset[i];

    for (uint32_t c = 0; c < num_cnks; ++c) {
      uint32_t local_start = c * RLE_ROWS_PER_CHUNK;
      uint32_t this_rows   = (c + 1u < num_cnks) ? RLE_ROWS_PER_CHUNK
                                                 : rc - local_start;

      rle_chunk_desc d;
      d.d_values        = malformed ? nullptr : parsed[i].d_values;
      d.d_cumsum        = d_cs;
      d.entry_count     = malformed ? 0u : static_cast<uint32_t>(parsed[i].cumsum.size());
      d.base_global_row = parsed[i].base_global_row;
      d.local_row_start = local_start;
      d.chunk_rows      = this_rows;
      h_descs.push_back(d);
    }
  }

  if (h_descs.empty()) return;

  rmm::device_uvector<rle_chunk_desc> d_descs(h_descs.size(), stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs.data(),
                               h_descs.size() * sizeof(rle_chunk_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  uint32_t const grid = static_cast<uint32_t>(h_descs.size());
  switch (type_size) {
    case 1:
      kernel_decode_rle<uint8_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), d_output, grid);
      break;
    case 2:
      kernel_decode_rle<uint16_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), reinterpret_cast<uint16_t*>(d_output), grid);
      break;
    case 4:
      kernel_decode_rle<uint32_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), reinterpret_cast<uint32_t*>(d_output), grid);
      break;
    case 8:
      kernel_decode_rle<uint64_t>
        <<<grid, BLOCK_DIM, RLE_SMEM_BYTES, stream.value()>>>(
          d_descs.data(), reinterpret_cast<uint64_t*>(d_output), grid);
      break;
    default:
      // Unreachable — guarded at the top of this function.
      break;
  }
}

}  // namespace sirius::cuda::scan
