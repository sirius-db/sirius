/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_decode.cuh"
#include "cuda/scan/pinned_bounce.cuh"
#include "log/logging.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>

namespace sirius::cuda::scan {

//===----------------------------------------------------------------------===//
// DuckDB dictionary segment layout (256KB block):
//   [0..19]  dictionary_compression_header_t (20 bytes)
//   [20..]   selection buffer (bitpacked indices into index_buffer)
//   [index_buffer_offset..] index_buffer: uint32_t[index_buffer_count]
//            each entry = cumulative byte offset from dict_end
//   [dict_end - dict_size .. dict_end]  dictionary data (strings, stored backwards)
//===----------------------------------------------------------------------===//

/// Fused step 1+2: Unpack bitpacked selection indices AND compute string
/// lengths in a single pass.  Saves one kernel launch per segment (~17μs each)
/// by combining what was previously two separate kernels.
/// Each thread unpacks its index, writes it (needed by gather), and immediately
/// looks up the string length from the index buffer.
__global__ void kernel_unpack_and_compute_lengths(
    const uint32_t* __restrict__ d_sel_buffer,    // Packed selection data
    const uint32_t* __restrict__ d_index_buffer,  // Dictionary index buffer
    uint32_t* __restrict__ d_indices,             // Output: unpacked indices (for gather)
    uint32_t* __restrict__ d_lengths,             // Output: string lengths
    uint32_t num_rows,
    uint32_t width)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  uint32_t sel = unpack_value<uint32_t>(d_sel_buffer, tid, width);
  d_indices[tid] = sel;
  d_lengths[tid] = (sel == 0) ? 0 : (d_index_buffer[sel] - d_index_buffer[sel - 1]);
}

/// Step 3: Gather string bytes from dictionary into contiguous output buffer.
/// Each thread copies one string from the dictionary to chars[offsets[i]].
__global__ void kernel_gather_strings(
    const uint32_t* __restrict__ d_indices,
    const uint32_t* __restrict__ d_index_buffer,
    const uint8_t* __restrict__ d_dict_end,       // Pointer to dict_end in segment
    const int32_t* __restrict__ d_offsets,         // Prefix-summed offsets
    uint8_t* __restrict__ d_chars,                 // Output char buffer
    uint32_t num_rows)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  uint32_t sel = d_indices[tid];
  if (sel == 0) return;  // null/empty, nothing to copy

  uint32_t dict_offset = d_index_buffer[sel];
  uint32_t str_len = d_index_buffer[sel] - d_index_buffer[sel - 1];
  int32_t out_offset = d_offsets[tid];

  // Dictionary strings are stored backwards from dict_end
  const uint8_t* str_ptr = d_dict_end - dict_offset;

  // Copy string bytes
  for (uint32_t b = 0; b < str_len; ++b) {
    d_chars[out_offset + b] = str_ptr[b];
  }
}

//===----------------------------------------------------------------------===//
// Host-side API
//===----------------------------------------------------------------------===//

/// Tiny kernel: write d_offsets[row_count] = d_offsets[row_count-1] + d_lengths[row_count-1].
/// This sets the sentinel value needed by cuDF string columns, entirely on GPU.
__global__ void kernel_write_offset_sentinel(
    const uint32_t* __restrict__ d_prefix_offsets,  // Prefix-summed offsets (uint32)
    const uint32_t* __restrict__ d_lengths,
    uint32_t* __restrict__ d_sentinel_slot,          // &d_offsets[row_count] as uint32
    uint32_t row_count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_sentinel_slot[0] = d_prefix_offsets[row_count - 1] + d_lengths[row_count - 1];
  }
}

void gpu_decode_dictionary(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t block_size,
    uint32_t row_count,
    int32_t* d_offsets,
    uint8_t** d_chars_out,
    size_t* total_chars_out,
    rmm::cuda_stream_view stream,
    void* d_scratch,
    uint32_t max_string_length,
    uint8_t* d_chars_preallocated,
    string_decode_temp* temp,
    bool skip_block_copy)
{
  const uint8_t* base = segment_data + block_offset;

  // Parse header on host
  struct {
    uint32_t dict_size;
    uint32_t dict_end;
    uint32_t index_buffer_offset;
    uint32_t index_buffer_count;
    uint32_t bitpacking_width;
  } header;
  std::memcpy(&header, base, sizeof(header));

  // Selection buffer starts right after the 20-byte header
  uint32_t sel_buf_offset = block_offset + 20;  // DICTIONARY_HEADER_SIZE

  // Index buffer
  uint32_t idx_buf_offset = block_offset + header.index_buffer_offset;
  // Dict end (absolute offset in segment_data)
  uint32_t dict_end_offset = block_offset + header.dict_end;

  SIRIUS_LOG_DEBUG(
      "[gpu_decode] dict: rows={}, dict_size={}, dict_end={}, idx_count={}, width={}, max_str_len={}",
      row_count, header.dict_size, header.dict_end,
      header.index_buffer_count, header.bitpacking_width, max_string_length);

  // 1. Copy the entire segment block to GPU — use scratch buffer if provided
  uint8_t* d_segment = nullptr;
  bool own_segment = false;
  if (d_scratch) {
    d_segment = static_cast<uint8_t*>(d_scratch);
  } else {
    cudaMallocAsync(&d_segment, segment_size, stream.value());
    own_segment = true;
  }
  if (!skip_block_copy) {
    bounce_h2d_async(d_segment, segment_data, segment_size, stream.value());
  }

  // 2. Resolve temp buffers: use caller-provided or allocate
  uint32_t* d_indices = nullptr;
  uint32_t* d_lengths = nullptr;
  void* d_cub_scratch = nullptr;
  size_t cub_scratch_bytes = 0;
  bool own_temp = (temp == nullptr);

  if (temp) {
    d_indices     = temp->d_buf_a;
    d_lengths     = temp->d_buf_b;
    d_cub_scratch = temp->d_cub_temp;
    cub_scratch_bytes = temp->cub_temp_bytes;
  } else {
    cudaMallocAsync(&d_indices, row_count * sizeof(uint32_t), stream.value());
    cudaMallocAsync(&d_lengths, row_count * sizeof(uint32_t), stream.value());
  }

  // 3+4. Fused: unpack selection indices AND compute string lengths in one pass
  constexpr uint32_t THREADS = 256;
  uint32_t blocks = (row_count + THREADS - 1) / THREADS;

  const uint32_t* d_sel_buf = reinterpret_cast<const uint32_t*>(d_segment + sel_buf_offset);
  const uint32_t* d_idx_buf = reinterpret_cast<const uint32_t*>(d_segment + idx_buf_offset);
  kernel_unpack_and_compute_lengths<<<blocks, THREADS, 0, stream.value()>>>(
      d_sel_buf, d_idx_buf, d_indices, d_lengths, row_count,
      header.bitpacking_width);

  // 5. Exclusive prefix sum of lengths -> offsets (using CUB)
  if (own_temp) {
    cub_scratch_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, cub_scratch_bytes,
        d_lengths, reinterpret_cast<uint32_t*>(d_offsets),
        row_count, stream.value());
    cudaMallocAsync(&d_cub_scratch, cub_scratch_bytes, stream.value());
  }

  cub::DeviceScan::ExclusiveSum(
      d_cub_scratch, cub_scratch_bytes,
      d_lengths, reinterpret_cast<uint32_t*>(d_offsets),
      row_count, stream.value());

  // 6. Allocate char buffer and gather strings.
  uint8_t* d_chars = nullptr;

  if (max_string_length > 0) {
    // FAST PATH: no sync needed — use upper-bound allocation or caller buffer
    size_t upper_bound_chars = static_cast<size_t>(row_count) * max_string_length;

    if (d_chars_preallocated) {
      d_chars = d_chars_preallocated;
    } else {
      cudaMallocAsync(&d_chars, upper_bound_chars, stream.value());
    }

    const uint8_t* d_dict_end_ptr = d_segment + dict_end_offset;
    kernel_gather_strings<<<blocks, THREADS, 0, stream.value()>>>(
        d_indices, d_idx_buf, d_dict_end_ptr,
        d_offsets, d_chars, row_count);

    // Write sentinel d_offsets[row_count] on GPU — no host readback
    kernel_write_offset_sentinel<<<1, 1, 0, stream.value()>>>(
        reinterpret_cast<uint32_t*>(d_offsets), d_lengths,
        reinterpret_cast<uint32_t*>(d_offsets) + row_count, row_count);

    *total_chars_out = upper_bound_chars;
  } else {
    // FALLBACK: sync to read exact total_chars for allocation
    cudaStreamSynchronize(stream.value());

    uint32_t last_offset = 0, last_length = 0;
    cudaMemcpy(&last_offset, reinterpret_cast<uint32_t*>(d_offsets) + row_count - 1,
               sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_length, d_lengths + row_count - 1,
               sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t total_chars = last_offset + last_length;

    cudaMemcpyAsync(reinterpret_cast<uint32_t*>(d_offsets) + row_count,
                    &total_chars, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream.value());

    *total_chars_out = total_chars;

    if (total_chars > 0) {
      cudaMallocAsync(&d_chars, total_chars, stream.value());

      const uint8_t* d_dict_end_ptr = d_segment + dict_end_offset;
      kernel_gather_strings<<<blocks, THREADS, 0, stream.value()>>>(
          d_indices, d_idx_buf, d_dict_end_ptr,
          d_offsets, d_chars, row_count);
    }
  }

  // When using preallocated buffer, don't return it (caller owns it).
  *d_chars_out = d_chars_preallocated ? nullptr : d_chars;

  // Cleanup (only if we allocated internally)
  if (own_temp) {
    cudaFreeAsync(d_indices, stream.value());
    cudaFreeAsync(d_lengths, stream.value());
    cudaFreeAsync(d_cub_scratch, stream.value());
  }
  if (own_segment) cudaFreeAsync(d_segment, stream.value());
}

//===----------------------------------------------------------------------===//
// Uncompressed string segment decode
//===----------------------------------------------------------------------===//
// DuckDB uncompressed VARCHAR layout:
//   [0..3]  dictionary_size (uint32)
//   [4..7]  dictionary_end  (uint32)
//   [8..]   duckdb_offsets[row_count] (int32 each, cumulative from dict_end)
//   string data stored backwards from dict_end
//
// duckdb_offsets[i] is the cumulative byte position.
// String i has length = abs(off[i]) - abs(off[i-1]), data at dict_end - abs(off[i]).

/// Convert DuckDB backward-cumulative offsets to cudf forward offsets and gather chars.
__global__ void kernel_uncompressed_string_decode(
    const int32_t* __restrict__ d_duckdb_offsets,  // DuckDB offset array on GPU
    const uint8_t* __restrict__ d_dict_end,        // Pointer to dict_end in GPU segment
    int32_t* __restrict__ d_cudf_offsets,           // Output: cudf-style offsets
    uint8_t* __restrict__ d_chars,                  // Output: char buffer
    uint32_t num_rows)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  int32_t cur_off = d_duckdb_offsets[tid];
  int32_t prev_off = (tid > 0) ? d_duckdb_offsets[tid - 1] : 0;

  // Absolute values (negative = overflow string, not handled on GPU)
  uint32_t abs_cur  = static_cast<uint32_t>(cur_off >= 0 ? cur_off : -cur_off);
  uint32_t abs_prev = static_cast<uint32_t>(prev_off >= 0 ? prev_off : -prev_off);
  uint32_t str_len = abs_cur - abs_prev;

  // cudf offset for this row = sum of all previous string lengths = abs(off[tid-1])
  // (This is the EXCLUSIVE prefix sum: offset[0]=0, offset[i]=abs(off[i-1]))
  d_cudf_offsets[tid] = static_cast<int32_t>(abs_prev);

  // Gather string bytes from dictionary (stored backwards from dict_end)
  const uint8_t* str_ptr = d_dict_end - abs_cur;
  int32_t out_pos = static_cast<int32_t>(abs_prev);
  for (uint32_t b = 0; b < str_len; ++b) {
    d_chars[out_pos + b] = str_ptr[b];
  }
}

/// Write sentinel offset[row_count] = abs(duckdb_offset[row_count-1]).
__global__ void kernel_uncompressed_write_sentinel(
    const int32_t* __restrict__ d_duckdb_offsets,
    int32_t* __restrict__ d_cudf_offsets,
    uint32_t row_count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int32_t last = d_duckdb_offsets[row_count - 1];
    d_cudf_offsets[row_count] = (last >= 0) ? last : -last;
  }
}

void gpu_decode_uncompressed_string(
    const uint8_t* segment_data,
    size_t segment_size,
    uint32_t block_offset,
    uint32_t row_count,
    int32_t* d_offsets,
    uint8_t* d_chars,
    rmm::cuda_stream_view stream,
    void* d_scratch,
    bool skip_block_copy)
{
  const uint8_t* base = segment_data + block_offset;

  // Parse header on host
  uint32_t dict_size, dict_end;
  std::memcpy(&dict_size, base, sizeof(uint32_t));
  std::memcpy(&dict_end, base + 4, sizeof(uint32_t));

  SIRIUS_LOG_DEBUG(
      "[gpu_decode] uncompressed string: rows={}, dict_size={}, dict_end={}",
      row_count, dict_size, dict_end);

  // Copy the segment block to GPU
  uint8_t* d_segment = nullptr;
  bool own_segment = false;
  if (d_scratch) {
    d_segment = static_cast<uint8_t*>(d_scratch);
  } else {
    cudaMallocAsync(&d_segment, segment_size, stream.value());
    own_segment = true;
  }
  if (!skip_block_copy) {
    bounce_h2d_async(d_segment, segment_data, segment_size, stream.value());
  }

  // DuckDB offsets start at byte 8 within the segment
  const int32_t* d_duckdb_offsets =
      reinterpret_cast<const int32_t*>(d_segment + block_offset + 8);
  const uint8_t* d_dict_end_ptr = d_segment + block_offset + dict_end;

  constexpr uint32_t THREADS = 256;
  uint32_t blocks = (row_count + THREADS - 1) / THREADS;

  // Single kernel: convert offsets + gather chars
  kernel_uncompressed_string_decode<<<blocks, THREADS, 0, stream.value()>>>(
      d_duckdb_offsets, d_dict_end_ptr,
      d_offsets, d_chars, row_count);

  // Write sentinel
  kernel_uncompressed_write_sentinel<<<1, 1, 0, stream.value()>>>(
      d_duckdb_offsets, d_offsets, row_count);

  if (own_segment) cudaFreeAsync(d_segment, stream.value());
}

}  // namespace sirius::cuda::scan
