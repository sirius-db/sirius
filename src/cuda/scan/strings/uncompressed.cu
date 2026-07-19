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
// UNCOMPRESSED string codec. On-disk segment layout (DuckDB):
//
//   offset 0                                                       seg_end
//   +-----------+----------+----------------------+---------------------+
//   | dict_size | dict_end |       offsets        |        chars        |
//   |    4B     |    4B    |  int32 × end_row     |  grows BACKWARD     |
//   |  (unused) |          |  backward-cumul      |  from dict_end      |
//   +-----------+----------+----------------------+---------------------+
//   0           4          8                      ^                     ^
//                                                  8 + 4*end_row        dict_end
//
// offsets[i] is signed int32; sign bit is DuckDB's inline-vs-pointer flag
// for in-memory string_t slots and is irrelevant on disk — take abs() for
// length math. Row i lives in `[dict_end - |offsets[i]|, dict_end - |offsets[i-1]|)`.
// One CTA per chunk, grid-stride within the chunk.
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/load_unaligned.cuh"
#include "cuda/scan/strings/uncompressed.cuh"

#include <cstdint>
#include <cstring>

namespace sirius::cuda::scan {

namespace {

// Backward-cumulative offsets are int32 at base+8. The segment base is not guaranteed
// int32-aligned, so read each offset alignment-agnostically.
__device__ __forceinline__ int32_t duck_offset(uint8_t const* off_bytes, int seg_i)
{
  return detail::load_unaligned<int32_t>(off_bytes + static_cast<size_t>(seg_i) * sizeof(int32_t));
}

//! @brief Compute decoded string lengths for an UNCOMPRESSED varchar segment.
//!
//! Lengths come from successive differences of the backward-cumulative offsets
//! array. If the segment is too short to contain the offsets region the kernel
//! zero-fills.
__global__ void kernel_compute_lengths_uncomp(string_chunk_desc const* __restrict__ descs,
                                              uint32_t* __restrict__ d_lengths,
                                              int num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc    = descs[chunk_id];
  auto const* base   = desc.d_bytes;
  auto const limit   = desc.bytes_size;
  auto const end_row = desc.seg_row_start + desc.row_count;

  __shared__ bool sm_ok;
  __shared__ uint32_t sm_dict_end;
  if (threadIdx.x == 0) {
    sm_ok = (limit >= 8u + size_t{end_row} * 4u);
    if (sm_ok) { memcpy(&sm_dict_end, base + 4, sizeof(sm_dict_end)); }
  }
  __syncthreads();
  if (!sm_ok) {
    for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0u;
    }
    return;
  }

  auto const* off_bytes = base + 8;
  for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    auto const seg_i    = desc.seg_row_start + i;
    auto const cur      = duck_offset(off_bytes, seg_i);
    auto const prev     = (seg_i > 0) ? duck_offset(off_bytes, seg_i - 1) : 0;
    auto const abs_cur  = static_cast<uint32_t>(cur >= 0 ? cur : ~static_cast<uint32_t>(cur) + 1u);
    auto const abs_prev = static_cast<uint32_t>(prev >= 0 ? prev : ~static_cast<uint32_t>(prev) + 1u);
    // Bound abs_cur/abs_prev against sm_dict_end: the chars region grows
    // backward from dict_end, so a valid offset satisfies abs_* <= sm_dict_end.
    // A corrupt backward-cumulative offset with abs_cur > sm_dict_end would
    // otherwise yield an arbitrary huge length here that feeds the offset scan
    // and drives an OOB read/write in the gather kernel. Clamp to sm_dict_end
    // so the length is at most the chars-region size, and the gather's
    // dict_end - abs_cur stays within [base, dict_end].
    auto const bounded_cur  = abs_cur <= sm_dict_end ? abs_cur : sm_dict_end;
    auto const bounded_prev = abs_prev <= sm_dict_end ? abs_prev : sm_dict_end;
    d_lengths[desc.global_row_start + i] = bounded_cur > bounded_prev ? bounded_cur - bounded_prev : 0u;
  }
}

//! @brief Gather UNCOMPRESSED segment strings to the output chars buffer.
//!
//! Chars region grows backward from `dict_end`, so row i starts at
//! `dict_end - |offsets[i]|`. Work granularity is one thread per row.
__global__ void kernel_gather_uncomp(string_chunk_desc const* __restrict__ descs,
                                     int32_t const* __restrict__ d_offsets,
                                     uint8_t* __restrict__ d_chars,
                                     int num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc    = descs[chunk_id];
  auto const* base   = desc.d_bytes;
  auto const limit   = desc.bytes_size;
  auto const end_row = desc.seg_row_start + desc.row_count;

  __shared__ bool sm_ok;
  __shared__ uint32_t sm_dict_end;
  if (threadIdx.x == 0) {
    sm_ok = (limit >= 8u + size_t{end_row} * 4u);
    if (sm_ok) { memcpy(&sm_dict_end, base + 4, sizeof(sm_dict_end)); }
  }
  __syncthreads();
  if (!sm_ok) return;

  auto const* off_bytes = base + 8;
  auto const* dict_end  = base + sm_dict_end;

  for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    auto const seg_i    = desc.seg_row_start + i;
    auto const cur      = duck_offset(off_bytes, seg_i);
    auto const prev     = (seg_i > 0) ? duck_offset(off_bytes, seg_i - 1) : 0;
    auto const abs_cur  = static_cast<uint32_t>(cur >= 0 ? cur : ~static_cast<uint32_t>(cur) + 1u);
    auto const abs_prev = static_cast<uint32_t>(prev >= 0 ? prev : ~static_cast<uint32_t>(prev) + 1u);
    // Bound abs_cur/abs_prev against sm_dict_end: the chars region is
    // [base, dict_end), and a valid row's chars live at [dict_end - abs_cur,
    // dict_end - abs_prev). A corrupt backward-cumulative offset with
    // abs_cur > sm_dict_end would make `src = dict_end - abs_cur` underflow
    // below `base` (OOB read before the segment) and let str_len drive an OOB
    // write into d_chars. Clamp so src stays within [base, dict_end] and
    // str_len stays within the chars region.
    auto const bounded_cur  = abs_cur <= sm_dict_end ? abs_cur : sm_dict_end;
    auto const bounded_prev = abs_prev <= sm_dict_end ? abs_prev : sm_dict_end;
    auto const str_len  = bounded_cur > bounded_prev ? bounded_cur - bounded_prev : 0u;

    auto const out_pos = d_offsets[desc.global_row_start + i];
    auto const* src    = dict_end - bounded_cur;
    // Warp-coalesced copy: for short strings (<=16 bytes, the common case),
    // use a single 16-byte load+store instead of memcpy. This lets the
    // compiler issue a single 128-bit transaction per thread, which the
    // memory controller can coalesce across the warp.
    if (str_len <= 16) {
      // Manual inline copy for short strings — avoids memcpy call overhead
      // and enables the compiler to use wider loads.
      for (uint32_t b = 0; b < str_len; ++b) {
        d_chars[out_pos + b] = src[b];
      }
    } else {
      memcpy(d_chars + out_pos, src, str_len);
    }
  }
}

}  // namespace

prepared_uncomp prepare_uncomp(gpu_string_codec_run const& run)
{
  prepared_uncomp out;
  out.descs.reserve(run.segments.size());
  for (auto const& seg : run.segments) {
    if (seg.row_count == 0) continue;
    out.descs.push_back(
      {seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset, seg.seg_row_start});
  }
  return out;
}

void launch_uncomp_lengths(string_chunk_desc const* d_chunks,
                           uint32_t* d_lengths,
                           uint32_t n_chunks,
                           rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_compute_lengths_uncomp<<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_chunks, d_lengths, n_chunks);
}

void launch_uncomp_gather(string_chunk_desc const* d_chunks,
                          int32_t const* d_offsets,
                          uint8_t* d_chars,
                          uint32_t n_chunks,
                          rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_gather_uncomp<<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_chunks, d_offsets, d_chars, n_chunks);
}

}  // namespace sirius::cuda::scan
