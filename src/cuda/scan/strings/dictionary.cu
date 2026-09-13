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
// DICTIONARY string codec. On-disk segment layout (DuckDB):
//
//   offset 0                                                      seg_end
//   +--------+------------------+------------------+---------------------+
//   | header | selection_buffer |   index_buffer   |     dict bytes      |
//   |  20B   |  bitpacked per   |  forward-cumul   |  entries packed in  |
//   |        |  row -> dict idx |  uint32 × count  |  REVERSE order      |
//   +--------+------------------+------------------+---------------------+
//   0        20                 ^                                        ^
//                               hdr.index_buffer_offset                  hdr.dict_end
//
// Index 0 is reserved for NULL (decoded length 0).
// Length(i) = idx_buf[i] - idx_buf[i-1].
// Bytes for entry i live in `[dict_end - idx_buf[i], dict_end - idx_buf[i-1])`.
// One CTA per chunk, grid-stride within the chunk.
//===----------------------------------------------------------------------===//

#include "cuda/scan/detail/byte_copy.cuh"
#include "cuda/scan/detail/load_unaligned.cuh"
#include "cuda/scan/strings/dictionary.cuh"
#include "cuda/scan/unpack_value.cuh"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace sirius::cuda::scan {

namespace {

//! On-disk DICTIONARY header.
struct dict_header_t {
  uint32_t dict_size;
  uint32_t dict_end;
  uint32_t index_buffer_offset;  ///< to forward-cumulative dict-byte index
  uint32_t index_buffer_count;
  uint32_t bitpacking_width;  ///< selection buffer
};

constexpr char const* DICTIONARY_CODEC_NAME = "DICTIONARY";

//! Validate header bounds and row/bit index ranges before decoding.
//! Payload index values remain device-side checks.
void validate_dict_segment(size_t seg_idx,
                           gpu_string_segment_desc const& seg,
                           dict_header_t const& hdr)
{
  auto fail = [&](std::string const& what) {
    throw_malformed_segment(DICTIONARY_CODEC_NAME, seg_idx, seg, what);
  };
  uint64_t const bytes_size = seg.bytes_size;
  if (hdr.bitpacking_width > MAX_BITPACKING_WIDTH) {
    fail("bitpacking_width " + std::to_string(hdr.bitpacking_width) + " > 32");
  }
  uint64_t const rows_end = uint64_t{seg.seg_row_start} + seg.row_count;
  // The kernels form row and bit indices in 32-bit signed arithmetic.
  constexpr uint64_t kernel_index_limit = std::numeric_limits<int32_t>::max();
  if (rows_end > kernel_index_limit || rows_end * hdr.bitpacking_width > kernel_index_limit) {
    fail("row or bit index for rows " + std::to_string(rows_end) +
         " does not fit the kernels' 32-bit index arithmetic");
  }
  if (sizeof(dict_header_t) + packed_words_bytes(rows_end, hdr.bitpacking_width) >
      hdr.index_buffer_offset) {
    fail("selection buffer for rows " + std::to_string(rows_end) +
         " reaches into the index buffer");
  }
  if (uint64_t{hdr.index_buffer_offset} + uint64_t{hdr.index_buffer_count} * sizeof(uint32_t) >
      bytes_size) {
    fail("index buffer ends past bytes_size");
  }
  if (hdr.dict_end > bytes_size) { fail("dict_end past bytes_size"); }
}

//! @brief Parse DICTIONARY header @p hdr from @p base, bounded by the buffer size @p limit.
//! @return true if the header was successfully parsed (i.e. the header fits within the buffer), and
//! if the header metadata is valid; false otherwise.
__device__ __forceinline__ bool parse_dict_header(uint8_t const* base,
                                                  uint32_t limit,
                                                  dict_header_t* hdr)
{
  if (limit < sizeof(dict_header_t)) return false;
  memcpy(hdr, base, sizeof(*hdr));
  return hdr->index_buffer_offset + hdr->index_buffer_count * sizeof(uint32_t) <= limit &&
         hdr->dict_end <= limit && hdr->bitpacking_width <= MAX_BITPACKING_WIDTH;
}

// The index buffer holds uint32 forward-cumulative byte offsets at base+index_buffer_offset. The
// segment base is not guaranteed uint32-aligned, so read each entry alignment-agnostically.
__device__ __forceinline__ uint32_t dict_index_at(uint8_t const* idx_bytes, int i)
{
  return detail::load_unaligned<uint32_t>(idx_bytes + static_cast<size_t>(i) * sizeof(uint32_t));
}

//! @brief Compute decoded string lengths for a DICTIONARY segment.
//!
//! Walks the selection buffer to get dictionary indices, looks up lengths from the index buffer,
//! and writes per-row lengths to d_lengths.
__global__ void kernel_compute_lengths_dict(string_chunk_desc const* __restrict__ descs,
                                            uint32_t* __restrict__ d_lengths,
                                            int num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc          = descs[chunk_id];
  auto const* segment_base = desc.d_bytes;

  __shared__ bool sm_ok;
  __shared__ dict_header_t sm_hdr;
  if (threadIdx.x == 0) { sm_ok = parse_dict_header(segment_base, desc.bytes_size, &sm_hdr); }
  __syncthreads();

  // Malformed metadata → zero-fill using the trusted descriptor row count.
  if (!sm_ok) {
    for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
      d_lengths[desc.global_row_start + i] = 0;
    }
    return;
  }

  // Calculate lengths by unpacking the selection buffer to get dict indices, then looking up
  // lengths from the index buffer.
  auto const* d_sel     = reinterpret_cast<uint32_t const*>(segment_base + sizeof(dict_header_t));
  auto const* idx_bytes = segment_base + sm_hdr.index_buffer_offset;
  for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    auto const segment_idx = desc.seg_row_start + i;
    auto const sel         = unpack_value<uint32_t>(d_sel, segment_idx, sm_hdr.bitpacking_width);
    int len                = 0;
    if (sel != 0u && sel < sm_hdr.index_buffer_count) {
      len = dict_index_at(idx_bytes, sel) - dict_index_at(idx_bytes, sel - 1);
    }
    d_lengths[desc.global_row_start + i] = len;
  }
}

//! @brief Gather DICTIONARY segment strings to the output chars buffer.
//!
//! Walk the selection buffer to get dict indices, look up offsets from the index buffer, then copy
//! from the dict bytes to the output chars buffer at positions from d_offsets.
//! Work granularity is one thread per row.
__global__ void kernel_gather_dict(string_chunk_desc const* __restrict__ descs,
                                   int32_t const* __restrict__ d_offsets,
                                   uint8_t* __restrict__ d_chars,
                                   int num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc          = descs[chunk_id];
  auto const* segment_base = desc.d_bytes;

  __shared__ bool sm_ok;
  __shared__ dict_header_t sm_hdr;
  if (threadIdx.x == 0) { sm_ok = parse_dict_header(segment_base, desc.bytes_size, &sm_hdr); }
  __syncthreads();
  if (!sm_ok) return;

  auto const* d_sel     = reinterpret_cast<uint32_t const*>(segment_base + sizeof(dict_header_t));
  auto const* idx_bytes = segment_base + sm_hdr.index_buffer_offset;
  auto const* dict_end  = segment_base + sm_hdr.dict_end;

  for (int i = threadIdx.x; i < desc.row_count; i += blockDim.x) {
    auto const segment_idx = desc.seg_row_start + i;
    auto const sel         = unpack_value<uint32_t>(d_sel, segment_idx, sm_hdr.bitpacking_width);
    if (sel == 0) continue;
    auto const end_off = dict_index_at(idx_bytes, sel);
    auto const str_len = end_off - dict_index_at(idx_bytes, sel - 1);
    auto const out_pos = d_offsets[desc.global_row_start + i];
    auto const* src    = dict_end - end_off;
    memcpy(d_chars + out_pos, src, str_len);
  }
}

//! @brief Gather DICTIONARY segment strings to the output chars buffer.
//!
//! Similar to kernel_gather_dict but with warp-cooperative copying for long strings.
__global__ void kernel_gather_dict_warp(string_chunk_desc const* __restrict__ descs,
                                        int32_t const* __restrict__ d_offsets,
                                        uint8_t* __restrict__ d_chars,
                                        int num_chunks)
{
  auto const chunk_id = blockIdx.x;
  if (chunk_id >= num_chunks) return;
  auto const desc             = descs[chunk_id];
  uint8_t const* segment_base = desc.d_bytes;

  __shared__ bool sm_ok;
  __shared__ dict_header_t sm_hdr;
  if (threadIdx.x == 0) { sm_ok = parse_dict_header(segment_base, desc.bytes_size, &sm_hdr); }
  __syncthreads();
  if (!sm_ok) return;

  auto const* d_sel     = reinterpret_cast<uint32_t const*>(segment_base + sizeof(dict_header_t));
  auto const* idx_bytes = segment_base + sm_hdr.index_buffer_offset;
  auto const* dict_end  = segment_base + sm_hdr.dict_end;

  int const lane          = threadIdx.x % cub::detail::warp_threads;
  int const warp_id       = threadIdx.x / cub::detail::warp_threads;
  int const warps_per_cta = blockDim.x / cub::detail::warp_threads;

  for (int i = warp_id; i < desc.row_count; i += warps_per_cta) {
    auto const segment_idx = desc.seg_row_start + i;
    auto const sel         = unpack_value<uint32_t>(d_sel, segment_idx, sm_hdr.bitpacking_width);
    if (sel == 0) continue;
    auto const end_off    = dict_index_at(idx_bytes, sel);
    auto const str_len    = end_off - dict_index_at(idx_bytes, sel - 1);
    auto const offset_ptr = static_cast<uint32_t>(d_offsets[desc.global_row_start + i]);
    auto const* src       = dict_end - end_off;
    detail::warp_copy_bytes(d_chars + offset_ptr, src, str_len, lane);
  }
}

}  // namespace

prepared_dict prepare_dict(gpu_string_codec_run const& run, rmm::cuda_stream_view stream)
{
  prepared_dict out;
  if (run.segments.empty()) return out;
  static thread_local pinned_host_pool headers_pool;
  auto const* headers =
    fetch_segment_headers<dict_header_t>(run, headers_pool, DICTIONARY_CODEC_NAME, stream);
  for (size_t i = 0; i < run.segments.size(); ++i) {
    auto const& seg = run.segments[i];
    if (seg.row_count == 0) continue;
    validate_dict_segment(i, seg, headers[i]);
    string_chunk_desc d{
      seg.d_bytes, seg.bytes_size, seg.row_count, seg.row_offset, seg.seg_row_start};
    auto& bucket =
      (seg.max_string_length < DICT_WARP_COOP_MIN_LEN) ? out.descs_short : out.descs_long;
    bucket.push_back(d);
  }
  return out;
}

void launch_dict_lengths(string_chunk_desc const* d_chunks,
                         uint32_t* d_lengths,
                         uint32_t n_chunks,
                         rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_compute_lengths_dict<<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_chunks, d_lengths, n_chunks);
}

void launch_dict_gather_short(string_chunk_desc const* d_chunks,
                              int32_t const* d_offsets,
                              uint8_t* d_chars,
                              uint32_t n_chunks,
                              rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_gather_dict<<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_chunks, d_offsets, d_chars, n_chunks);
}

void launch_dict_gather_long(string_chunk_desc const* d_chunks,
                             int32_t const* d_offsets,
                             uint8_t* d_chars,
                             uint32_t n_chunks,
                             rmm::cuda_stream_view stream)
{
  if (n_chunks == 0) return;
  kernel_gather_dict_warp<<<n_chunks, STRINGS_BLOCK_DIM, 0, stream.value()>>>(
    d_chunks, d_offsets, d_chars, n_chunks);
}

}  // namespace sirius::cuda::scan
