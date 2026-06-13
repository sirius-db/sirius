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
// BITPACKING decode kernel.
//
// A bitpacked column segment is structured as
//   [metadata header,
//    data[group_0,..., group_N],
//    metadata[group_N,...,group_0]]
// - The metadata header (8B) points to the end of the array of metadata (`metadata_end`).
// - Each metadata entry (4B) contains a) the group's offset from the segment into its corresponding
//   data region (`data_off`) and b) the bitpacking mode of the groups, which is one of:
//     * CONSTANT
//     * CONSTANT_DELTA
//     * FOR
//     * DELTA_FOR
// - Each data group is structured as [T,T,(T,),packed data], where T is the output type, and the
//   number and meaning of the Ts depends on the mode.
//
// The work partitioning is one CTA per group (BP_META_GROUP_SIZE = 2048 rows). All groups
// across every segment of a column are dispatched in one batched kernel.
//
// Defensive metadata bounds: `metadata_end` and `data_off` come from disk
// and could be malformed. Each parse step that would produce an OOB read or
// write demotes `sm_mode` to INVALID; INVALID then deterministically
// zero-fills the group's output range (using the trusted descriptor row
// count, not the parsed metadata).
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/shared_staging.cuh"
#include "cuda/scan/gpu_decode_bitpacking.cuh"
#include "cuda/scan/unpack_value.cuh"

#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/warp/warp_scan.cuh>
#include <cuda/std/numeric>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::cuda::scan {

namespace {

constexpr uint32_t BLOCK_DIM = 256;

/// VPT (= Values Per Thread) for the FOR / DELTA_FOR loop. With BLOCK_DIM=256 this
/// covers BP_META_GROUP_SIZE (2048) in one pass without tail iterations.
constexpr uint32_t VPT = BP_META_GROUP_SIZE / BLOCK_DIM;
static_assert(BLOCK_DIM * VPT == BP_META_GROUP_SIZE,
              "BLOCK_DIM and VPT must tile the metadata group exactly");
static_assert(BLOCK_DIM % 32 == 0,
              "BLOCK_DIM must be a multiple of warpSize for the DELTA_FOR "
              "warp-aggregate scan");

/// One CTA's unit of work: one metadata group within one segment.
struct bp_group_desc {
  uint8_t const* d_segment;    ///< Device pointer to the segment's first byte.
  uint32_t segment_bytes;      ///< Size of the staged segment buffer.
  uint32_t group_idx;          ///< Metadata-group index within the segment.
  uint32_t group_row_count;    ///< Rows in this group (last group may be < 2048).
  uint32_t global_row_offset;  ///< Output offset, in rows, for this group.
};

//===----------------------------------------------------------------------===//
// Batched decode kernel.
//===----------------------------------------------------------------------===//
template <typename T, int SHMEM_BYTES>
__global__ void kernel_decode_bitpacking(bp_group_desc const* __restrict__ descs,
                                         T* __restrict__ d_output,
                                         uint32_t num_groups)
{
  using vec_t               = int4;
  auto constexpr VEC_BYTES  = sizeof(vec_t);
  auto constexpr WORD_BYTES = sizeof(uint32_t);
  auto constexpr WORD_BITS  = ::cuda::std::numeric_limits<uint32_t>::digits;

  static_assert(sizeof(T) <= 8 && sizeof(vec_t) % sizeof(T) == 0,
                "BITPACKING kernel only instantiates for type sizes in {1,2,4,8}; "
                "extending the dispatcher to a non-conforming type needs a scalar "
                "fallback path here");

  auto const gid = blockIdx.x;
  if (gid >= num_groups) return;

  auto const desc      = descs[gid];
  auto const* seg_base = desc.d_segment;
  auto const seg_bytes = desc.segment_bytes;

  // `d_output` is 256B-aligned by CUDA, but `out` is only 16B-aligned when
  // `global_row_offset * sizeof(T)` is a multiple of `sizeof(vec_t)`. That
  // holds for the first segment in a column and any stacked segment whose
  // prior-segment row count was a multiple of `sizeof(vec_t) / sizeof(T)` —
  // not in general. CONSTANT and CONSTANT_DELTA branch on this and use
  // scalar stores when unaligned (FOR / DELTA_FOR already store scalar).
  auto* out                  = d_output + desc.global_row_offset;
  bool const out_vec_aligned = (reinterpret_cast<::cuda::std::uintptr_t>(out) % sizeof(vec_t)) == 0;

  // Shared metadata — written by thread 0, read by all after the barrier.
  // `sm_aux` is overloaded by mode:
  //   CONSTANT       -> the constant value (broadcast to every output row).
  //   CONSTANT_DELTA -> the per-row delta (output[i] = frame + i*delta).
  //   FOR            -> unused (left zero).
  //   DELTA_FOR      -> the initial prefix-sum bias (delta_offset).
  __shared__ uint8_t sm_mode;
  __shared__ uint32_t sm_data_offset;
  __shared__ uint32_t sm_width;
  __shared__ uint32_t sm_row_count;
  __shared__ T sm_frame;
  __shared__ T sm_aux;

  if (threadIdx.x == 0) {
    sm_mode        = static_cast<uint8_t>(BitpackingMode::INVALID);
    sm_width       = 0;
    sm_frame       = T(0);
    sm_aux         = T(0);
    sm_data_offset = 0;
    sm_row_count   = 0;

    uint64_t metadata_end = 0;
    if (seg_bytes >= sizeof(uint64_t)) { memcpy(&metadata_end, seg_base, sizeof(uint64_t)); }

    // The metadata trailer holds one uint32 entry per group, ending at
    // metadata_end; entry K (0-based) lives at metadata_end - (K+1) * 4.
    // The trailer must fit *and* lie inside the segment buffer.
    bool metadata_ok =
      metadata_end >= sizeof(uint64_t) + (uint64_t{desc.group_idx} + 1) * sizeof(uint32_t) &&
      metadata_end <= seg_bytes;

    if (metadata_ok) {
      auto const* entry_addr = seg_base + metadata_end - (desc.group_idx + 1) * sizeof(uint32_t);
      uint32_t encoded       = 0;
      memcpy(&encoded, entry_addr, sizeof(uint32_t));

      auto const data_off    = encoded & 0x00FFFFFFu;
      auto const parsed_mode = (encoded >> 24) & 0xFFu;
      sm_row_count           = desc.group_row_count;

      // `data_off` is a within-segment offset; the kernel reads v0 and v1
      // unconditionally (2*sizeof(T)) for every mode below, then reads a
      // third T only for DELTA_FOR. Bound the unconditional read here and
      // re-bound for DELTA_FOR's third T inside its case to avoid falsely
      // rejecting tight CONSTANT / FOR segments.
      bool data_off_ok = uint64_t{data_off} + 2u * sizeof(T) <= metadata_end;

      if (data_off_ok) {
        auto const* dp = seg_base + data_off;
        T v0{}, v1{};
        memcpy(&v0, dp, sizeof(T));
        memcpy(&v1, dp + sizeof(T), sizeof(T));

        auto mode = static_cast<BitpackingMode>(parsed_mode);
        switch (mode) {
          case BitpackingMode::CONSTANT:
            sm_mode = parsed_mode;
            sm_aux  = v0;
            break;
          case BitpackingMode::CONSTANT_DELTA:
            sm_mode  = parsed_mode;
            sm_frame = v0;
            sm_aux   = v1;
            break;
          case BitpackingMode::FOR:
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_width       = static_cast<uint32_t>(v1);
            sm_data_offset = data_off + 2u * sizeof(T);
            break;
          case BitpackingMode::DELTA_FOR: {
            // DELTA_FOR adds a third T (delta_offset) before the packed
            // stream — re-bound to catch tight segments where the third
            // read would alias the metadata trailer.
            if (uint64_t{data_off} + 3u * sizeof(T) > metadata_end) break;
            T v2{};
            memcpy(&v2, dp + 2u * sizeof(T), sizeof(T));
            sm_mode        = parsed_mode;
            sm_frame       = v0;
            sm_width       = static_cast<uint32_t>(v1);
            sm_aux         = v2;
            sm_data_offset = data_off + 3u * sizeof(T);
            break;
          }
          default:
            // Unknown / INVALID mode — leave sm_mode at its default INVALID.
            break;
        }

        // Bitpacking widths larger than the target type width are invalid.
        if (sm_width > sizeof(T) * 8u) {
          sm_mode  = static_cast<uint8_t>(BitpackingMode::INVALID);
          sm_width = 0;
        }

        // Ensure the packed stream stays inside the segment buffer.
        if (sm_mode == static_cast<uint8_t>(BitpackingMode::FOR) ||
            sm_mode == static_cast<uint8_t>(BitpackingMode::DELTA_FOR)) {
          auto const packed_words = ::cuda::ceil_div(sm_row_count * sm_width, WORD_BITS);
          auto const packed_end   = uint64_t{sm_data_offset} + packed_words * WORD_BYTES;
          if (packed_end > metadata_end) {
            sm_mode  = static_cast<uint8_t>(BitpackingMode::INVALID);
            sm_width = 0;
          }
        }
      }
    }
  }
  __syncthreads();

  auto const rc   = sm_row_count;
  auto const mode = static_cast<BitpackingMode>(sm_mode);

  //===--------------------------------------------------------------------===//
  // CONSTANT — broadcast `sm_aux` to every row.
  //===--------------------------------------------------------------------===//
  if (mode == BitpackingMode::CONSTANT) {
    auto const val         = sm_aux;
    uint32_t constexpr TPV = sizeof(vec_t) / sizeof(T);
    if (out_vec_aligned) {
      auto const vec_count = rc / TPV;
      auto* out4           = reinterpret_cast<vec_t*>(out);
      vec_t packed;
      auto* lanes = reinterpret_cast<T*>(&packed);
#pragma unroll
      for (uint32_t i = 0; i < TPV; ++i)
        lanes[i] = val;
      for (uint32_t v = threadIdx.x; v < vec_count; v += blockDim.x) {
        __stcs(out4 + v, packed);
      }
      uint32_t tail_start = vec_count * TPV;
      for (uint32_t i = tail_start + threadIdx.x; i < rc; i += blockDim.x) {
        __stcs(out + i, val);
      }
    } else {
      for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
        __stcs(out + i, val);
      }
    }
    return;
  }

  //===--------------------------------------------------------------------===//
  // CONSTANT_DELTA — out[i] = frame + i*delta.
  //===--------------------------------------------------------------------===//
  if (mode == BitpackingMode::CONSTANT_DELTA) {
    auto const frame       = sm_frame;
    auto const delta       = sm_aux;
    uint32_t constexpr TPV = sizeof(vec_t) / sizeof(T);
    if (out_vec_aligned) {
      auto const vec_count = rc / TPV;
      auto* out4           = reinterpret_cast<vec_t*>(out);
      for (uint32_t v = threadIdx.x; v < vec_count; v += blockDim.x) {
        vec_t packed;
        auto* lanes       = reinterpret_cast<T*>(&packed);
        auto const base_v = v * TPV;
#pragma unroll
        for (uint32_t i = 0; i < TPV; ++i) {
          lanes[i] = static_cast<T>(frame + static_cast<T>(base_v + i) * delta);
        }
        __stcs(out4 + v, packed);
      }
      auto const tail_start = vec_count * TPV;
      for (uint32_t i = tail_start + threadIdx.x; i < rc; i += blockDim.x) {
        __stcs(out + i, static_cast<T>(frame + static_cast<T>(i) * delta));
      }
    } else {
      for (uint32_t i = threadIdx.x; i < rc; i += blockDim.x) {
        __stcs(out + i, static_cast<T>(frame + static_cast<T>(i) * delta));
      }
    }
    return;
  }

  //===--------------------------------------------------------------------===//
  // FOR / DELTA_FOR — load packed bytes into shmem, then unpack-and-store.
  //===--------------------------------------------------------------------===//

  // Anything other than FOR / DELTA_FOR is INVALID metadata or an unknown
  // mode: zero-fill the descriptor's row range.
  if (mode != BitpackingMode::FOR && mode != BitpackingMode::DELTA_FOR) {
    auto const fill_rows = desc.group_row_count;
    for (uint32_t i = threadIdx.x; i < fill_rows; i += blockDim.x) {
      __stcs(out + i, T(0));
    }
    return;
  }

  auto const width = sm_width;
  auto const frame = sm_frame;

  auto constexpr TARGET_ALIGNMENT_BYTES = sizeof(vec_t);
  alignas(TARGET_ALIGNMENT_BYTES) __shared__ uint32_t shmem[SHMEM_BYTES / WORD_BYTES];
  auto const* packed_bytes  = seg_base + sm_data_offset;
  auto const n_packed_words = ::cuda::ceil_div(rc * width, WORD_BITS);

  // Stage the packed bytes into shared memory; one guard word covers unpack_value's straddle read.
  detail::stage_packed_to_shmem<BLOCK_DIM>(shmem, packed_bytes, n_packed_words, /*guard_words=*/1);

  //===--------------------------------------------------------------------===//
  // FOR -- frame + value.
  //===--------------------------------------------------------------------===//

  if (mode == BitpackingMode::FOR) {
    // Striped fill
#pragma unroll
    for (uint32_t v = 0; v < VPT; ++v) {
      auto const idx = v * blockDim.x + threadIdx.x;
      if (idx >= rc) break;
      __stcs(out + idx, frame + unpack_value<T>(shmem, idx, width));
    }
    return;
  }

  //===--------------------------------------------------------------------===//
  // DELTA_FOR — frame + per-row delta, prefix-sum, then add delta_offset.
  //===--------------------------------------------------------------------===//
  // Two-stage scan in blocked layout The block-wide prefix-sum is built from per-warp
  // `cub::WarpScan` plus a single shmem exchange of the per-warp totals; one thread serially scans
  // those 8 totals and broadcasts back. Final values are exchanged through shmem
  // into striped layout for coalesced global stores.
  //
  // Used instead of `cub::BlockScan<T, 256>` to reduce register pressure and SMEM usage for
  // improved occupancy.

  T thread_data[VPT];
#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    auto const idx = threadIdx.x * VPT + v;
    thread_data[v] = (idx < rc) ? static_cast<T>(frame + unpack_value<T>(shmem, idx, width)) : T(0);
  }
#pragma unroll
  for (uint32_t v = 1; v < VPT; ++v) {
    thread_data[v] = static_cast<T>(thread_data[v] + thread_data[v - 1]);
  }

  // Stage 1 — per-warp inclusive scan of thread aggregates.
  using WarpScanT          = cub::WarpScan<T>;
  auto constexpr NUM_WARPS = BLOCK_DIM / 32;
  __shared__ typename WarpScanT::TempStorage warp_scan_temp[NUM_WARPS];
  __shared__ T warp_aggregates[NUM_WARPS];

  auto const warp_id = threadIdx.x / 32;
  auto const lane_id = threadIdx.x % 32;

  T thread_agg = thread_data[VPT - 1];
  T warp_inclusive;
  T warp_total;
  WarpScanT(warp_scan_temp[warp_id]).InclusiveSum(thread_agg, warp_inclusive, warp_total);
  if (lane_id == 31) warp_aggregates[warp_id] = warp_total;
  __syncthreads();

  // Stage 2 — serial scan of NUM_WARPS=8 warp totals by warp 0 lane 0.
  if (warp_id == 0 && lane_id == 0) {
    T running = T(0);
#pragma unroll
    for (uint32_t w = 0; w < NUM_WARPS; ++w) {
      auto const t       = warp_aggregates[w];
      warp_aggregates[w] = running;
      running            = static_cast<T>(running + t);
    }
  }
  __syncthreads();

  // Stage 3 — each thread's exclusive prefix in the block:
  //   warp_aggregates[warp_id] (sum of warps before this one)
  //   + (warp_inclusive - thread_agg) (this thread's exclusive prefix in warp)
  //   + delta_offset
  auto const delta_offset     = sm_aux;
  auto const warp_prefix      = warp_aggregates[warp_id];
  auto const thread_exclusive = static_cast<T>(warp_inclusive - thread_agg);
  auto const prefix           = static_cast<T>(warp_prefix + thread_exclusive + delta_offset);

  // Blocked->striped exchange via shmem (reusing the packed-words buffer,
  // no longer read after the unpack loop) for coalesced stores.
  auto* shmem_t = reinterpret_cast<T*>(shmem);
#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    auto const idx = threadIdx.x * VPT + v;
    if (idx < rc) shmem_t[idx] = static_cast<T>(thread_data[v] + prefix);
  }
  __syncthreads();
#pragma unroll
  for (uint32_t v = 0; v < VPT; ++v) {
    auto const idx = v * blockDim.x + threadIdx.x;
    if (idx < rc) __stcs(out + idx, shmem_t[idx]);
  }
}

//===----------------------------------------------------------------------===//
// Type-erased dispatch.
//===----------------------------------------------------------------------===//

/// Build per-group descriptors covering every metadata group across every
/// segment of `run`. Returns a host-side vector ready to be uploaded.
std::vector<bp_group_desc> build_group_descs(gpu_codec_run const& run)
{
  std::vector<bp_group_desc> descs;
  // Each segment contributes ceil(row_count / BP_META_GROUP_SIZE) groups;
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    auto const num_groups = ::cuda::ceil_div(seg.row_count, BP_META_GROUP_SIZE);
    for (uint32_t g = 0; g < num_groups; ++g) {
      // If it's the last group, ceiling the number of rows based on the segment row count
      auto const group_rows =
        (g + 1u < num_groups) ? BP_META_GROUP_SIZE : seg.row_count - g * BP_META_GROUP_SIZE;
      descs.push_back(
        {seg.d_bytes, seg.bytes_size, g, group_rows, seg.row_offset + g * BP_META_GROUP_SIZE});
    }
  }
  return descs;
}

template <typename T>
void launch_typed(bp_group_desc const* h_descs,
                  size_t num_groups,
                  T* d_output,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  // The +1 guard word satisfies `unpack_value`'s 3-word read contract for
  // 64-bit types when a value spans words [w, w+1, w+2].
  auto constexpr MAX_WIDTH        = ::cuda::std::numeric_limits<T>::digits;
  auto constexpr WORD_BITS        = ::cuda::std::numeric_limits<uint32_t>::digits;
  auto constexpr MAX_PACKED_WORDS = ::cuda::ceil_div(BP_META_GROUP_SIZE * MAX_WIDTH, WORD_BITS) + 1;
  auto constexpr SHMEM_BYTES      = static_cast<int>(MAX_PACKED_WORDS * sizeof(uint32_t));

  if (num_groups == 0) return;

  rmm::device_uvector<bp_group_desc> d_descs(num_groups, stream, mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                               h_descs,
                               num_groups * sizeof(bp_group_desc),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  kernel_decode_bitpacking<T, SHMEM_BYTES>
    <<<static_cast<uint32_t>(num_groups), BLOCK_DIM, 0, stream.value()>>>(
      d_descs.data(), d_output, static_cast<uint32_t>(num_groups));
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public entry.
//
// `type` is unused: bit-level decode is endianness- and signedness-independent,
// and the only mode-specific arithmetic (`frame + delta`, prefix sum) wraps
// correctly in two's complement, so the kernel routes by `type_size` alone.
// Kept in the signature for parity with the dispatcher's other codec entries.
//===----------------------------------------------------------------------===//

void decode_bitpacking_data(gpu_codec_run const& run,
                            uint8_t* d_output,
                            cudf::data_type /*type*/,
                            uint32_t type_size,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  auto descs = build_group_descs(run);
  if (descs.empty()) return;

  switch (type_size) {
    case 1: launch_typed<uint8_t>(descs.data(), descs.size(), d_output, stream, mr); break;
    case 2:
      launch_typed<uint16_t>(
        descs.data(), descs.size(), reinterpret_cast<uint16_t*>(d_output), stream, mr);
      break;
    case 4:
      launch_typed<uint32_t>(
        descs.data(), descs.size(), reinterpret_cast<uint32_t*>(d_output), stream, mr);
      break;
    case 8:
      launch_typed<uint64_t>(
        descs.data(), descs.size(), reinterpret_cast<uint64_t*>(d_output), stream, mr);
      break;
    default:
      // The kernel reads up to 3*sizeof(T) bytes of header per group; widths
      // outside {1,2,4,8} would also need a different `unpack_value`
      // instantiation. Upstream viability is expected to keep them out;
      // throw rather than silently return with the buffer untouched.
      throw std::runtime_error(
        "gpu_decode_table: viability invariant violated — BITPACKING type_size " +
        std::to_string(type_size));
  }
}

}  // namespace sirius::cuda::scan
