/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_decode.cuh"
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

/// Step 1: Unpack bitpacked selection indices.
/// Each thread unpacks one index from the selection buffer.
__global__ void kernel_unpack_sel_indices(
    const uint32_t* __restrict__ d_sel_buffer,  // Packed selection data
    uint32_t* __restrict__ d_indices,            // Output: unpacked indices
    uint32_t num_rows,
    uint32_t width)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  // Standard horizontal bitpacking extraction (same as fastpforlib)
  d_indices[tid] = unpack_value<uint32_t>(d_sel_buffer, tid, width);
}

/// Step 2: Compute string lengths from index buffer lookups.
/// lengths[i] = index_buf[indices[i]] - index_buf[indices[i] - 1]
/// Index 0 is null/empty (length 0).
__global__ void kernel_compute_string_lengths(
    const uint32_t* __restrict__ d_indices,      // Unpacked selection indices
    const uint32_t* __restrict__ d_index_buffer,  // Dictionary index buffer
    uint32_t* __restrict__ d_lengths,             // Output: string lengths
    uint32_t num_rows)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;

  uint32_t sel = d_indices[tid];
  if (sel == 0) {
    d_lengths[tid] = 0;  // null/empty
  } else {
    d_lengths[tid] = d_index_buffer[sel] - d_index_buffer[sel - 1];
  }
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
    void* d_scratch)
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
      "[gpu_decode] dict: rows={}, dict_size={}, dict_end={}, idx_count={}, width={}",
      row_count, header.dict_size, header.dict_end,
      header.index_buffer_count, header.bitpacking_width);

  // 1. Copy the entire segment block to GPU — use scratch buffer if provided
  uint8_t* d_segment = nullptr;
  bool own_segment = false;
  if (d_scratch) {
    d_segment = static_cast<uint8_t*>(d_scratch);
  } else {
    cudaMallocAsync(&d_segment, segment_size, stream.value());
    own_segment = true;
  }
  cudaMemcpyAsync(d_segment, segment_data, segment_size,
                  cudaMemcpyHostToDevice, stream.value());

  // 2. Allocate temp buffers on GPU
  uint32_t* d_indices = nullptr;   // Unpacked selection indices
  uint32_t* d_lengths = nullptr;   // String lengths
  cudaMallocAsync(&d_indices, row_count * sizeof(uint32_t), stream.value());
  cudaMallocAsync(&d_lengths, row_count * sizeof(uint32_t), stream.value());

  // 3. Unpack selection indices
  constexpr uint32_t THREADS = 256;
  uint32_t blocks = (row_count + THREADS - 1) / THREADS;

  const uint32_t* d_sel_buf = reinterpret_cast<const uint32_t*>(d_segment + sel_buf_offset);
  kernel_unpack_sel_indices<<<blocks, THREADS, 0, stream.value()>>>(
      d_sel_buf, d_indices, row_count, header.bitpacking_width);

  // 4. Compute string lengths
  const uint32_t* d_idx_buf = reinterpret_cast<const uint32_t*>(d_segment + idx_buf_offset);
  kernel_compute_string_lengths<<<blocks, THREADS, 0, stream.value()>>>(
      d_indices, d_idx_buf, d_lengths, row_count);

  // 5. Exclusive prefix sum of lengths -> offsets (using CUB)
  size_t cub_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, cub_temp_bytes,
      d_lengths, reinterpret_cast<uint32_t*>(d_offsets),
      row_count, stream.value());

  uint8_t* d_cub_temp = nullptr;
  cudaMallocAsync(&d_cub_temp, cub_temp_bytes, stream.value());

  cub::DeviceScan::ExclusiveSum(
      d_cub_temp, cub_temp_bytes,
      d_lengths, reinterpret_cast<uint32_t*>(d_offsets),
      row_count, stream.value());

  // Read total_chars = last_offset + last_length. Requires sync since we need the
  // value to allocate the char buffer. This is the one unavoidable sync per segment.
  cudaStreamSynchronize(stream.value());

  uint32_t last_offset = 0, last_length = 0;
  cudaMemcpy(&last_offset, reinterpret_cast<uint32_t*>(d_offsets) + row_count - 1,
             sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&last_length, d_lengths + row_count - 1,
             sizeof(uint32_t), cudaMemcpyDeviceToHost);
  uint32_t total_chars = last_offset + last_length;

  // Set sentinel: d_offsets[row_count] = total_chars
  cudaMemcpyAsync(reinterpret_cast<uint32_t*>(d_offsets) + row_count,
                  &total_chars, sizeof(uint32_t),
                  cudaMemcpyHostToDevice, stream.value());

  *total_chars_out = total_chars;

  // 6. Allocate char buffer and gather strings
  uint8_t* d_chars = nullptr;
  if (total_chars > 0) {
    cudaMallocAsync(&d_chars, total_chars, stream.value());

    const uint8_t* d_dict_end_ptr = d_segment + dict_end_offset;
    kernel_gather_strings<<<blocks, THREADS, 0, stream.value()>>>(
        d_indices, d_idx_buf, d_dict_end_ptr,
        d_offsets, d_chars, row_count);
  }

  *d_chars_out = d_chars;

  // Cleanup temp buffers (async — caller syncs the stream)
  cudaFreeAsync(d_indices, stream.value());
  cudaFreeAsync(d_lengths, stream.value());
  cudaFreeAsync(d_cub_temp, stream.value());
  if (own_segment) cudaFreeAsync(d_segment, stream.value());
}

}  // namespace sirius::cuda::scan
