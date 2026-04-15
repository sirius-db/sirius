/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_native_decode.cuh"
#include "cuda/scan/gpu_decode.cuh"
#include "cuda/scan/gpu_decode_batched_string.cuh"
#include "log/logging.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <duckdb/common/types.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sirius::cuda::scan {

using sirius::op::scan::column_scan_result;
using sirius::op::scan::direct_block_scan_result;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Map DuckDB LogicalType to cudf data_type.
cudf::data_type to_cudf_type(const duckdb::LogicalType& type)
{
  switch (type.id()) {
    case duckdb::LogicalTypeId::TINYINT:   return cudf::data_type(cudf::type_id::INT8);
    case duckdb::LogicalTypeId::SMALLINT:  return cudf::data_type(cudf::type_id::INT16);
    case duckdb::LogicalTypeId::INTEGER:   return cudf::data_type(cudf::type_id::INT32);
    case duckdb::LogicalTypeId::BIGINT:    return cudf::data_type(cudf::type_id::INT64);
    case duckdb::LogicalTypeId::UTINYINT:  return cudf::data_type(cudf::type_id::UINT8);
    case duckdb::LogicalTypeId::USMALLINT: return cudf::data_type(cudf::type_id::UINT16);
    case duckdb::LogicalTypeId::UINTEGER:  return cudf::data_type(cudf::type_id::UINT32);
    case duckdb::LogicalTypeId::UBIGINT:   return cudf::data_type(cudf::type_id::UINT64);
    case duckdb::LogicalTypeId::FLOAT:     return cudf::data_type(cudf::type_id::FLOAT32);
    case duckdb::LogicalTypeId::DOUBLE:    return cudf::data_type(cudf::type_id::FLOAT64);
    case duckdb::LogicalTypeId::BOOLEAN:   return cudf::data_type(cudf::type_id::BOOL8);
    case duckdb::LogicalTypeId::DATE:      return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
    case duckdb::LogicalTypeId::TIMESTAMP: return cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
    case duckdb::LogicalTypeId::VARCHAR:   return cudf::data_type(cudf::type_id::STRING);
    case duckdb::LogicalTypeId::HUGEINT:   return cudf::data_type(cudf::type_id::INT64);
    case duckdb::LogicalTypeId::DECIMAL: {
      switch (type.InternalType()) {
        case duckdb::PhysicalType::INT32:
          return cudf::data_type(cudf::type_id::DECIMAL32,
                                 -duckdb::DecimalType::GetScale(type));
        case duckdb::PhysicalType::INT64:
          return cudf::data_type(cudf::type_id::DECIMAL64,
                                 -duckdb::DecimalType::GetScale(type));
        case duckdb::PhysicalType::INT128:
          return cudf::data_type(cudf::type_id::DECIMAL128,
                                 -duckdb::DecimalType::GetScale(type));
        default: break;
      }
    }
    default: break;
  }
  throw std::runtime_error("gpu_native_decode: unsupported DuckDB type " +
                           type.ToString());
}

/// Get byte size of a DuckDB physical type.
uint32_t get_type_size(duckdb::PhysicalType pt)
{
  switch (pt) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8:
    case duckdb::PhysicalType::UINT8:   return 1;
    case duckdb::PhysicalType::INT16:
    case duckdb::PhysicalType::UINT16:  return 2;
    case duckdb::PhysicalType::INT32:
    case duckdb::PhysicalType::UINT32:
    case duckdb::PhysicalType::FLOAT:   return 4;
    case duckdb::PhysicalType::INT64:
    case duckdb::PhysicalType::UINT64:
    case duckdb::PhysicalType::DOUBLE:  return 8;
    case duckdb::PhysicalType::INT128:  return 16;
    default: return 0;
  }
}

bool is_signed_type(duckdb::PhysicalType pt)
{
  switch (pt) {
    case duckdb::PhysicalType::INT8:
    case duckdb::PhysicalType::INT16:
    case duckdb::PhysicalType::INT32:
    case duckdb::PhysicalType::INT64:
    case duckdb::PhysicalType::INT128:  return true;
    default: return false;
  }
}

/// Check if a compression type is GPU-decodable for string columns.
bool is_gpu_decodable_string(duckdb::CompressionType ct)
{
  switch (ct) {
    case duckdb::CompressionType::COMPRESSION_DICTIONARY:
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_FSST:
      return true;
    default:
      return false;
  }
}

/// CUDA kernel to fill a buffer with a constant value (for CONSTANT segments).
template <typename T>
__global__ void kernel_fill_constant(T* output, T value, uint32_t count)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) { output[idx] = value; }
}

/// CUDA kernel to set all validity bits to 1 (all valid).
__global__ void kernel_fill_valid(uint64_t* mask, uint32_t num_words)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_words) { mask[idx] = ~0ULL; }
}

/// Count valid (set) bits in a validity mask entirely on GPU.
/// Single block of 256 threads — each thread popcounts its share of words,
/// then a shared-memory tree reduction produces the total.
/// Replaces: sync + full mask D2H + CPU popcountll loop.
__global__ void kernel_count_valid_bits(
    const uint64_t* __restrict__ mask,
    uint32_t num_words,
    uint32_t total_rows,
    uint32_t* __restrict__ d_valid_count)
{
  __shared__ uint32_t s_counts[256];

  uint32_t valid = 0;
  for (uint32_t i = threadIdx.x; i < num_words; i += blockDim.x) {
    uint64_t word = mask[i];
    // Mask off padding bits in the last word
    if (i == num_words - 1) {
      uint32_t tail = total_rows & 63;
      if (tail > 0) word &= (1ULL << tail) - 1;
    }
    valid += __popcll(word);
  }

  s_counts[threadIdx.x] = valid;
  __syncthreads();

  // Tree reduction
  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) s_counts[threadIdx.x] += s_counts[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) *d_valid_count = s_counts[0];
}

constexpr size_t DUCKDB_BLOCK_SIZE = 262144;  // 256KB

//===----------------------------------------------------------------------===//
// Fixed-width column decode
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> decode_fixed_width_column(
    column_scan_result& col_scan,
    const duckdb::LogicalType& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    void* d_scratch)
{
  auto cudf_type     = to_cudf_type(type);
  auto physical_type = type.InternalType();
  uint32_t type_size = get_type_size(physical_type);
  bool is_signed     = is_signed_type(physical_type);
  size_t total_rows  = col_scan.data.total_rows;

  if (type_size == 0 || total_rows == 0) {
    return cudf::make_empty_column(cudf_type);
  }

  // Allocate output data buffer on GPU
  rmm::device_buffer data_buf(total_rows * type_size, stream, mr);
  auto* d_output = static_cast<uint8_t*>(data_buf.data());

  // Pre-allocate RLE cumsum scratch buffer (reused across segments).
  // 4096 entries covers typical RLE segments; larger ones fall back to alloc.
  constexpr size_t RLE_CUMSUM_CAPACITY = 4096 * sizeof(uint32_t);
  uint32_t* d_rle_cumsum = nullptr;
  bool has_rle = false;
  for (auto const& seg : col_scan.data.segments) {
    if (seg.compression == duckdb::CompressionType::COMPRESSION_RLE) {
      has_rle = true;
      break;
    }
  }
  if (has_rle) {
    cudaMallocAsync(&d_rle_cumsum, RLE_CUMSUM_CAPACITY, stream.value());
  }

  size_t row_offset = 0;
  size_t gpu_decoded_segs = 0;
  const uint8_t* last_block_base = nullptr;

  for (auto& seg : col_scan.data.segments) {
    if (seg.row_count == 0) { row_offset += seg.row_count; continue; }
    // Skip non-decodable segments, but allow blockless CONSTANT segments
    // (persistent=true, data_ptr=null, value in constant_data).
    if (!seg.persistent || (!seg.data_ptr
        && seg.compression != duckdb::CompressionType::COMPRESSION_CONSTANT)) {
      row_offset += seg.row_count;
      continue;
    }

    auto* d_dest = d_output + row_offset * type_size;
    size_t seg_bytes = seg.row_count * type_size;

    switch (seg.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        cudaMemcpyAsync(d_dest, seg.data_ptr, seg_bytes,
                        cudaMemcpyHostToDevice, stream.value());
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_CONSTANT: {
        // Blockless CONSTANT segments have data_ptr=null; value is in constant_data.
        const uint8_t* val_src = seg.data_ptr ? seg.data_ptr : seg.constant_data;
        if (type_size == 4) {
          int32_t val; std::memcpy(&val, val_src, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int32_t*>(d_dest), val, seg.row_count);
        } else if (type_size == 8) {
          int64_t val; std::memcpy(&val, val_src, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int64_t*>(d_dest), val, seg.row_count);
        } else if (type_size == 2) {
          int16_t val; std::memcpy(&val, val_src, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int16_t*>(d_dest), val, seg.row_count);
        } else if (type_size == 1) {
          int8_t val; std::memcpy(&val, val_src, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int8_t*>(d_dest), val, seg.row_count);
        } else {
          std::vector<uint8_t> host_buf(seg_bytes);
          for (size_t r = 0; r < seg.row_count; ++r) {
            std::memcpy(host_buf.data() + r * type_size, val_src, type_size);
          }
          cudaMemcpyAsync(d_dest, host_buf.data(), seg_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        }
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_BITPACKING: {
        const uint8_t* block_base = seg.data_ptr - seg.block_offset;
        bool block_cached = (block_base == last_block_base);
        if (!block_cached) last_block_base = block_base;
        gpu_decode_bitpacking(
            block_base, DUCKDB_BLOCK_SIZE,
            static_cast<uint32_t>(seg.block_offset),
            static_cast<uint32_t>(seg.row_count),
            type_size, is_signed,
            d_dest, stream,
            d_scratch, nullptr, 0,
            block_cached);
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_RLE: {
        const uint8_t* block_base = seg.data_ptr - seg.block_offset;
        bool block_cached = (block_base == last_block_base);
        if (!block_cached) last_block_base = block_base;
        gpu_decode_rle(
            block_base, DUCKDB_BLOCK_SIZE,
            static_cast<uint32_t>(seg.block_offset),
            static_cast<uint32_t>(seg.row_count),
            type_size,
            d_dest, stream,
            d_scratch,
            block_cached,
            d_rle_cumsum, RLE_CUMSUM_CAPACITY);
        gpu_decoded_segs++;
        break;
      }

      default: {
        throw std::runtime_error(
            "gpu_native_decode: unsupported compression type " +
            std::to_string(static_cast<int>(seg.compression)) +
            " for fixed-width column — falling back to CPU scan");
      }
    }

    row_offset += seg.row_count;
  }

  // Free RLE cumsum scratch
  if (d_rle_cumsum) {
    cudaFreeAsync(d_rle_cumsum, stream.value());
  }

  // Decode validity
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;

  if (col_scan.has_nulls) {
    size_t mask_bytes = (total_rows + 63) / 64 * sizeof(uint64_t);
    null_mask = rmm::device_buffer(mask_bytes, stream, mr);
    auto* d_mask = static_cast<uint64_t*>(null_mask.data());

    // First set everything valid, then overlay actual validity segments
    uint32_t num_words = static_cast<uint32_t>((total_rows + 63) / 64);
    uint32_t fill_blocks = (num_words + 255) / 256;
    kernel_fill_valid<<<fill_blocks, 256, 0, stream.value()>>>(d_mask, num_words);

    size_t val_row_offset = 0;
    for (auto& vseg : col_scan.validity.segments) {
      if (vseg.row_count == 0) { val_row_offset += vseg.row_count; continue; }

      if (vseg.persistent && vseg.data_ptr &&
          vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        // DuckDB validity bitmap is LSB-first uint64_t, same as cuDF.
        // Copy at the correct bit offset.
        size_t seg_mask_bytes = (vseg.row_count + 7) / 8;

        if (val_row_offset % 64 == 0) {
          // Aligned — direct copy to the correct word offset
          size_t word_offset = val_row_offset / 64;
          cudaMemcpyAsync(d_mask + word_offset, vseg.data_ptr, seg_mask_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        } else {
          // Unaligned — copy to host, do bit-shift, then upload
          // For simplicity, read validity on host and copy the full mask region
          std::vector<uint8_t> host_mask(seg_mask_bytes);
          std::memcpy(host_mask.data(), vseg.data_ptr, seg_mask_bytes);

          // Write to the device at the byte-aligned position
          size_t byte_offset = val_row_offset / 8;
          cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_mask) + byte_offset,
                          host_mask.data(), seg_mask_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        }
      }
      // For non-persistent or EMPTY validity segments: bits remain set (all valid)
      // which is correct — EMPTY means no nulls in that range.

      val_row_offset += vseg.row_count;
    }

    // Count nulls on GPU — avoids copying the entire mask to host.
    uint32_t* d_valid_count;
    cudaMallocAsync(&d_valid_count, sizeof(uint32_t), stream.value());
    cudaMemsetAsync(d_valid_count, 0, sizeof(uint32_t), stream.value());
    kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_valid_count);
    stream.synchronize();
    uint32_t valid_count;
    cudaMemcpy(&valid_count, d_valid_count, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_valid_count, stream.value());
    null_count = static_cast<cudf::size_type>(total_rows - valid_count);
  }

  return std::make_unique<cudf::column>(
      cudf_type,
      static_cast<cudf::size_type>(total_rows),
      std::move(data_buf),
      std::move(null_mask),
      null_count);
}

//===----------------------------------------------------------------------===//
// String column decode
//===----------------------------------------------------------------------===//

/// Kernel to add a constant offset to a range of int32 values.
/// Used to adjust per-segment offsets to global positions in the concat buffer.
__global__ void kernel_adjust_offsets(
    int32_t* __restrict__ d_offsets,
    int32_t adjustment,
    uint32_t count)
{
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    d_offsets[tid] += adjustment;
  }
}

std::unique_ptr<cudf::column> decode_string_column(
    column_scan_result& col_scan,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    void* d_scratch)
{
  size_t total_rows = col_scan.data.total_rows;
  if (total_rows == 0) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }

  // Validate all segments and check if GPU concat path is available
  // (requires max_string_length on every dictionary segment)
  bool can_gpu_concat = true;
  for (auto const& seg : col_scan.data.segments) {
    if (!seg.persistent || !seg.data_ptr || seg.row_count == 0) continue;
    if (!is_gpu_decodable_string(seg.compression)) {
      throw std::runtime_error(
          "gpu_native_decode: unsupported string compression " +
          std::to_string(static_cast<int>(seg.compression)) +
          " — falling back to CPU scan");
    }
    if ((seg.compression == duckdb::CompressionType::COMPRESSION_DICTIONARY
         || seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED
         || seg.compression == duckdb::CompressionType::COMPRESSION_FSST)
        && seg.max_string_length == 0) {
      can_gpu_concat = false;
    }
  }

  if (!can_gpu_concat) {
    throw std::runtime_error(
        "gpu_native_decode: dictionary segment missing max_string_length stats "
        "— falling back to CPU scan");
  }

  //===----------------------------------------------------------------------===//
  // GPU CONCAT PATH: decode all segments directly into final buffers.
  //
  // Temp buffers (for bitunpack, prefix sums, CUB scratch) are allocated
  // ONCE per column and reused across segments, eliminating thousands of
  // per-segment cudaMallocAsync/cudaFreeAsync calls.
  //===----------------------------------------------------------------------===//

  // Phase 1: Compute per-segment layout and find max segment row count
  struct seg_layout {
    size_t row_start;       // first row index in the final column
    size_t char_start;      // byte offset in the final chars buffer
    size_t char_capacity;   // row_count * max_string_length (upper bound)
  };
  std::vector<seg_layout> layouts;
  layouts.reserve(col_scan.data.segments.size());

  size_t cum_rows = 0, cum_chars = 0;
  uint32_t max_seg_rows = 0;
  bool has_fsst = false;

  for (auto const& seg : col_scan.data.segments) {
    seg_layout l;
    l.row_start = cum_rows;
    l.char_start = cum_chars;
    if (seg.row_count > 0 && seg.persistent && seg.data_ptr
        && (seg.compression == duckdb::CompressionType::COMPRESSION_DICTIONARY
            || seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED
            || seg.compression == duckdb::CompressionType::COMPRESSION_FSST)) {
      l.char_capacity = static_cast<size_t>(seg.row_count) * seg.max_string_length;
      max_seg_rows = std::max(max_seg_rows, static_cast<uint32_t>(seg.row_count));
      if (seg.compression == duckdb::CompressionType::COMPRESSION_FSST)
        has_fsst = true;
    } else {
      l.char_capacity = 0;
    }
    layouts.push_back(l);
    cum_rows += seg.row_count;
    cum_chars += l.char_capacity;
  }
  size_t total_chars_upper = cum_chars;

  // Phase 2: Allocate final buffers on GPU (one alloc each)
  rmm::device_uvector<int32_t> d_offsets(total_rows + 1, stream, mr);
  rmm::device_buffer d_chars(total_chars_upper > 0 ? total_chars_upper : 1, stream, mr);
  auto* d_chars_base = static_cast<uint8_t*>(d_chars.data());

  // Phase 2b: Allocate shared temp buffers for all segment decodes
  string_decode_temp temp{};
  bool have_temp = (max_seg_rows > 0);

  if (have_temp) {
    cudaMallocAsync(&temp.d_buf_a, max_seg_rows * sizeof(uint32_t), stream.value());
    cudaMallocAsync(&temp.d_buf_b, max_seg_rows * sizeof(uint32_t), stream.value());
    if (has_fsst) {
      cudaMallocAsync(&temp.d_buf_c, max_seg_rows * sizeof(uint32_t), stream.value());
      cudaMallocAsync(&temp.d_fsst_len, 255, stream.value());
      cudaMallocAsync(&temp.d_fsst_sym,
                      255 * sizeof(unsigned long long), stream.value());
    }

    // Query CUB for max temp size (covers both InclusiveSum and ExclusiveSum)
    size_t cub_inc = 0, cub_exc = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, cub_inc, (uint32_t*)nullptr, (uint32_t*)nullptr,
        max_seg_rows, stream.value());
    cub::DeviceScan::ExclusiveSum(
        nullptr, cub_exc, (uint32_t*)nullptr, (uint32_t*)nullptr,
        max_seg_rows, stream.value());
    temp.cub_temp_bytes = std::max(cub_inc, cub_exc);
    cudaMallocAsync(&temp.d_cub_temp, temp.cub_temp_bytes, stream.value());
  }

  // Phase 3: Decode each segment into final buffers using ACTUAL char positions.
  // Per-segment sync to read sentinel for contiguous char positioning.
  // Block H2D dedup: skip copy when consecutive segments share the same 256KB block.
  constexpr uint32_t ADJ_THREADS = 256;
  size_t actual_cum_chars = 0;
  const uint8_t* last_block_base = nullptr;

  for (size_t si = 0; si < col_scan.data.segments.size(); ++si) {
    auto& seg = col_scan.data.segments[si];
    auto& layout = layouts[si];
    if (seg.row_count == 0) continue;

    int32_t* d_seg_offsets = d_offsets.data() + layout.row_start;
    size_t seg_char_start = actual_cum_chars;

    if (!seg.persistent || !seg.data_ptr
        || seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
      cudaMemsetAsync(d_seg_offsets, 0,
                      (seg.row_count + 1) * sizeof(int32_t), stream.value());
    } else {
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      bool block_cached = (block_base == last_block_base);
      if (!block_cached) last_block_base = block_base;
      uint8_t* d_seg_chars = d_chars_base + seg_char_start;

      if (seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        gpu_decode_uncompressed_string(
            block_base, DUCKDB_BLOCK_SIZE,
            static_cast<uint32_t>(seg.block_offset),
            static_cast<uint32_t>(seg.row_count),
            d_seg_offsets, d_seg_chars, stream, d_scratch,
            block_cached);
      } else if (seg.compression == duckdb::CompressionType::COMPRESSION_DICTIONARY) {
        uint8_t* d_chars_out = nullptr;
        size_t total_chars_out = 0;
        gpu_decode_dictionary(
            block_base, DUCKDB_BLOCK_SIZE,
            static_cast<uint32_t>(seg.block_offset),
            static_cast<uint32_t>(DUCKDB_BLOCK_SIZE),
            static_cast<uint32_t>(seg.row_count),
            d_seg_offsets, &d_chars_out, &total_chars_out,
            stream, d_scratch, seg.max_string_length, d_seg_chars,
            &temp, block_cached);
      } else if (seg.compression == duckdb::CompressionType::COMPRESSION_FSST) {
        gpu_decode_fsst(
            block_base, DUCKDB_BLOCK_SIZE,
            static_cast<uint32_t>(seg.block_offset),
            static_cast<uint32_t>(seg.row_count),
            d_seg_offsets, d_seg_chars, stream, d_scratch,
            &temp, block_cached);
      }
    }

    // Read back sentinel to learn actual char count for this segment.
    int32_t seg_actual_chars = 0;
    stream.synchronize();
    cudaMemcpy(&seg_actual_chars, d_seg_offsets + seg.row_count,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    actual_cum_chars += seg_actual_chars;

    // Adjust this segment's offsets by adding the actual char position.
    if (seg_char_start > 0) {
      bool is_last = (layout.row_start + seg.row_count >= total_rows);
      uint32_t adj_count = static_cast<uint32_t>(seg.row_count) + (is_last ? 1 : 0);
      uint32_t adj_blocks = (adj_count + ADJ_THREADS - 1) / ADJ_THREADS;
      kernel_adjust_offsets<<<adj_blocks, ADJ_THREADS, 0, stream.value()>>>(
          d_seg_offsets,
          static_cast<int32_t>(seg_char_start),
          adj_count);
    }
  }

  // Phase 4: Single sync — all GPU work is done
  stream.synchronize();

  // Read the sentinel to get the actual total chars (for logging)
  int32_t actual_total_chars = 0;
  cudaMemcpy(&actual_total_chars, d_offsets.data() + total_rows,
             sizeof(int32_t), cudaMemcpyDeviceToHost);

  // Free temp buffers
  if (have_temp) {
    cudaFreeAsync(temp.d_buf_a, stream.value());
    cudaFreeAsync(temp.d_buf_b, stream.value());
    if (has_fsst) {
      cudaFreeAsync(temp.d_buf_c, stream.value());
      cudaFreeAsync(temp.d_fsst_len, stream.value());
      cudaFreeAsync(temp.d_fsst_sym, stream.value());
    }
    cudaFreeAsync(temp.d_cub_temp, stream.value());
  }

  // Build offsets column (int32 — sufficient for per-column data)
  auto offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(total_rows + 1),
      d_offsets.release(),
      rmm::device_buffer{0, stream, mr},
      0);

  // Build validity for string column
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;

  if (col_scan.has_nulls) {
    size_t mask_bytes = (total_rows + 63) / 64 * sizeof(uint64_t);
    null_mask = rmm::device_buffer(mask_bytes, stream, mr);
    auto* d_mask = static_cast<uint64_t*>(null_mask.data());

    uint32_t num_words = static_cast<uint32_t>((total_rows + 63) / 64);
    uint32_t fill_blocks = (num_words + 255) / 256;
    kernel_fill_valid<<<fill_blocks, 256, 0, stream.value()>>>(d_mask, num_words);

    size_t val_row_offset = 0;
    for (auto& vseg : col_scan.validity.segments) {
      if (vseg.row_count == 0) { val_row_offset += vseg.row_count; continue; }
      if (vseg.persistent && vseg.data_ptr &&
          vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        size_t seg_mask_bytes = (vseg.row_count + 7) / 8;
        if (val_row_offset % 64 == 0) {
          size_t word_offset = val_row_offset / 64;
          cudaMemcpyAsync(d_mask + word_offset, vseg.data_ptr, seg_mask_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        } else {
          size_t byte_offset = val_row_offset / 8;
          cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_mask) + byte_offset,
                          vseg.data_ptr, seg_mask_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        }
      }
      val_row_offset += vseg.row_count;
    }

    // Count nulls on GPU
    uint32_t* d_valid_count;
    cudaMallocAsync(&d_valid_count, sizeof(uint32_t), stream.value());
    cudaMemsetAsync(d_valid_count, 0, sizeof(uint32_t), stream.value());
    kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_valid_count);
    stream.synchronize();
    uint32_t valid_count;
    cudaMemcpy(&valid_count, d_valid_count, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_valid_count, stream.value());
    null_count = static_cast<cudf::size_type>(total_rows - valid_count);
  }

  SIRIUS_LOG_INFO("[gpu_native_decode] string col: {} rows, {} actual chars "
                  "({} upper), {} segs, gpu_concat=true",
                  total_rows, actual_total_chars, total_chars_upper,
                  col_scan.data.segments.size());

  return cudf::make_strings_column(
      static_cast<cudf::size_type>(total_rows),
      std::move(offsets_col),
      std::move(d_chars),
      null_count,
      std::move(null_mask));
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API: decode full table
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> gpu_decode_table(
    std::vector<column_scan_result>& column_scans,
    const std::vector<duckdb::LogicalType>& column_types,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  using clock = std::chrono::steady_clock;
  auto t_start = clock::now();

  if (column_scans.size() != column_types.size()) {
    throw std::invalid_argument("gpu_decode_table: column_scans and column_types size mismatch");
  }

  size_t total_rows = column_scans.empty() ? 0 : column_scans[0].data.total_rows;

  // Pre-allocate scratch buffer — reused across all segments and columns.
  // Block-sized buffer for H2D segment data.  Bitpacking metadata is now
  // parsed on GPU directly from block data (no separate scratch needed).
  void* d_scratch = nullptr;
  cudaMallocAsync(&d_scratch, DUCKDB_BLOCK_SIZE, stream.value());

  auto t_alloc = clock::now();

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(column_scans.size());

  size_t n_fixed = 0, n_string = 0;
  double us_fixed = 0, us_string = 0;

  for (size_t ci = 0; ci < column_scans.size(); ++ci) {
    auto& col_scan = column_scans[ci];
    auto& col_type = column_types[ci];

    auto col_start = clock::now();

    if (col_type.id() == duckdb::LogicalTypeId::VARCHAR) {
      columns.push_back(decode_string_column_batched(col_scan, stream, mr));
      auto col_end = clock::now();
      us_string += std::chrono::duration_cast<std::chrono::microseconds>(col_end - col_start).count();
      n_string++;
    } else {
      columns.push_back(decode_fixed_width_column(
          col_scan, col_type, stream, mr, d_scratch));
      auto col_end = clock::now();
      us_fixed += std::chrono::duration_cast<std::chrono::microseconds>(col_end - col_start).count();
      n_fixed++;
    }
  }

  auto t_decode = clock::now();

  // Single sync point for all async decode work
  stream.synchronize();

  auto t_sync = clock::now();

  // Free scratch buffer
  cudaFreeAsync(d_scratch, stream.value());

  auto t_end = clock::now();

  auto us = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };

  fprintf(stderr,
      "[gpu_native_decode] table: %zu cols (%zu fixed + %zu str), %zu rows | "
      "alloc=%.1fms enqueue=%.1fms (fixed=%.1fms str=%.1fms) "
      "sync=%.1fms total=%.1fms\n",
      columns.size(), n_fixed, n_string, total_rows,
      us(t_start, t_alloc) / 1000.0,
      us(t_alloc, t_decode) / 1000.0,
      us_fixed / 1000.0,
      us_string / 1000.0,
      us(t_decode, t_sync) / 1000.0,
      us(t_start, t_end) / 1000.0);

  return std::make_unique<cudf::table>(std::move(columns));
}

//===----------------------------------------------------------------------===//
// Pipelined decode: fixed-width from pre-transferred device data
//===----------------------------------------------------------------------===//

namespace {

std::unique_ptr<cudf::column> decode_fixed_width_column_from_device(
    column_scan_result& col_scan, const duckdb::LogicalType& type,
    const device_block_map& blocks, uint8_t* device_staging,
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto cudf_type = to_cudf_type(type);
  auto physical_type = type.InternalType();
  uint32_t type_size = get_type_size(physical_type);
  bool is_signed = is_signed_type(physical_type);
  size_t total_rows = col_scan.data.total_rows;
  if (type_size == 0 || total_rows == 0) return cudf::make_empty_column(cudf_type);

  rmm::device_buffer data_buf(total_rows * type_size, stream, mr);
  auto* d_output = static_cast<uint8_t*>(data_buf.data());
  size_t row_offset = 0;

  // Pre-allocate RLE cumsum scratch (reused across segments)
  constexpr size_t RLE_CUMSUM_CAP = 4096 * sizeof(uint32_t);
  uint32_t* d_rle_cumsum = nullptr;
  bool has_rle = false;
  for (auto const& seg : col_scan.data.segments) {
    if (seg.compression == duckdb::CompressionType::COMPRESSION_RLE) {
      has_rle = true; break;
    }
  }
  if (has_rle) cudaMallocAsync(&d_rle_cumsum, RLE_CUMSUM_CAP, stream.value());

  for (auto& seg : col_scan.data.segments) {
    if (seg.row_count == 0) { row_offset += seg.row_count; continue; }
    if (!seg.persistent || (!seg.data_ptr
        && seg.compression != duckdb::CompressionType::COMPRESSION_CONSTANT)) {
      row_offset += seg.row_count; continue;
    }
    auto* d_dest = d_output + row_offset * type_size;
    size_t seg_bytes = seg.row_count * type_size;

    switch (seg.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        auto it = blocks.offsets.find(seg.block_id);
        if (it != blocks.offsets.end())
          cudaMemcpyAsync(d_dest, device_staging + it->second + seg.block_offset, seg_bytes, cudaMemcpyDeviceToDevice, stream.value());
        else
          cudaMemcpyAsync(d_dest, seg.data_ptr, seg_bytes, cudaMemcpyHostToDevice, stream.value());
        break;
      }
      case duckdb::CompressionType::COMPRESSION_CONSTANT: {
        const uint8_t* vs = seg.data_ptr ? seg.data_ptr : seg.constant_data;
        if (type_size == 4) { int32_t v; std::memcpy(&v, vs, 4); kernel_fill_constant<<<(seg.row_count+255)/256,256,0,stream.value()>>>(reinterpret_cast<int32_t*>(d_dest), v, seg.row_count); }
        else if (type_size == 8) { int64_t v; std::memcpy(&v, vs, 8); kernel_fill_constant<<<(seg.row_count+255)/256,256,0,stream.value()>>>(reinterpret_cast<int64_t*>(d_dest), v, seg.row_count); }
        else if (type_size == 2) { int16_t v; std::memcpy(&v, vs, 2); kernel_fill_constant<<<(seg.row_count+255)/256,256,0,stream.value()>>>(reinterpret_cast<int16_t*>(d_dest), v, seg.row_count); }
        else if (type_size == 1) { int8_t v; std::memcpy(&v, vs, 1); kernel_fill_constant<<<(seg.row_count+255)/256,256,0,stream.value()>>>(reinterpret_cast<int8_t*>(d_dest), v, seg.row_count); }
        break;
      }
      case duckdb::CompressionType::COMPRESSION_BITPACKING: {
        auto it = blocks.offsets.find(seg.block_id);
        void* d_block = (it != blocks.offsets.end()) ? static_cast<void*>(device_staging + it->second) : nullptr;
        gpu_decode_bitpacking(seg.data_ptr - seg.block_offset, DUCKDB_BLOCK_SIZE,
            seg.block_offset, seg.row_count, type_size, is_signed, d_dest, stream,
            d_block ? d_block : nullptr, nullptr, 0, d_block != nullptr);
        break;
      }
      case duckdb::CompressionType::COMPRESSION_RLE: {
        auto it = blocks.offsets.find(seg.block_id);
        void* d_block = (it != blocks.offsets.end()) ? static_cast<void*>(device_staging + it->second) : nullptr;
        gpu_decode_rle(seg.data_ptr - seg.block_offset, DUCKDB_BLOCK_SIZE,
            seg.block_offset, seg.row_count, type_size,
            d_dest, stream, d_block ? d_block : nullptr, d_block != nullptr,
            d_rle_cumsum, RLE_CUMSUM_CAP);
        break;
      }
      default: throw std::runtime_error("unsupported compression in pipelined decode");
    }
    row_offset += seg.row_count;
  }

  if (d_rle_cumsum) cudaFreeAsync(d_rle_cumsum, stream.value());

  // Validity
  rmm::device_buffer null_mask{}; cudf::size_type null_count = 0;
  if (col_scan.has_nulls) {
    size_t mask_bytes = (total_rows+63)/64*sizeof(uint64_t);
    null_mask = rmm::device_buffer(mask_bytes, stream, mr);
    auto* d_mask = static_cast<uint64_t*>(null_mask.data());
    uint32_t num_words = (total_rows+63)/64;
    kernel_fill_valid<<<(num_words+255)/256,256,0,stream.value()>>>(d_mask, num_words);
    size_t vo = 0;
    for (auto& vs : col_scan.validity.segments) {
      if (vs.row_count == 0) { vo += vs.row_count; continue; }
      if (vs.persistent && vs.data_ptr && vs.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        size_t mb = (vs.row_count+7)/8;
        if (vo%64==0) cudaMemcpyAsync(d_mask+vo/64, vs.data_ptr, mb, cudaMemcpyHostToDevice, stream.value());
        else cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_mask)+vo/8, vs.data_ptr, mb, cudaMemcpyHostToDevice, stream.value());
      }
      vo += vs.row_count;
    }
    uint32_t* d_vc; cudaMallocAsync(&d_vc, sizeof(uint32_t), stream.value());
    cudaMemsetAsync(d_vc, 0, sizeof(uint32_t), stream.value());
    kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_vc);
    stream.synchronize();
    uint32_t vc; cudaMemcpy(&vc, d_vc, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_vc, stream.value());
    null_count = static_cast<cudf::size_type>(total_rows - vc);
  }
  return std::make_unique<cudf::column>(cudf_type, static_cast<cudf::size_type>(total_rows),
      std::move(data_buf), std::move(null_mask), null_count);
}

//===----------------------------------------------------------------------===//
// Pipelined decode: string from pre-transferred device data (per-segment sync)
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> decode_string_column_from_device(
    column_scan_result& col_scan, const device_block_map& blocks,
    uint8_t* device_staging, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Same as serial decode_string_column but reads blocks from device staging.
  // Uses per-segment sync — same proven-correct approach as the serial path.
  size_t total_rows = col_scan.data.total_rows;
  if (total_rows == 0) return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));

  struct seg_layout { size_t row_start, char_start, char_capacity; };
  std::vector<seg_layout> layouts;
  size_t cum_rows = 0, cum_chars = 0; uint32_t max_seg_rows = 0; bool has_fsst = false;
  for (auto const& seg : col_scan.data.segments) {
    seg_layout l{cum_rows, cum_chars, 0};
    if (seg.row_count > 0 && seg.persistent && seg.data_ptr && is_gpu_decodable_string(seg.compression)) {
      l.char_capacity = static_cast<size_t>(seg.row_count) * seg.max_string_length;
      max_seg_rows = std::max(max_seg_rows, static_cast<uint32_t>(seg.row_count));
      if (seg.compression == duckdb::CompressionType::COMPRESSION_FSST) has_fsst = true;
    }
    layouts.push_back(l); cum_rows += seg.row_count; cum_chars += l.char_capacity;
  }

  rmm::device_uvector<int32_t> d_offsets(total_rows+1, stream, mr);
  rmm::device_buffer d_chars(cum_chars > 0 ? cum_chars : 1, stream, mr);
  auto* d_chars_base = static_cast<uint8_t*>(d_chars.data());

  string_decode_temp temp{};
  if (max_seg_rows > 0) {
    cudaMallocAsync(&temp.d_buf_a, max_seg_rows*4, stream.value());
    cudaMallocAsync(&temp.d_buf_b, max_seg_rows*4, stream.value());
    if (has_fsst) { cudaMallocAsync(&temp.d_buf_c, max_seg_rows*4, stream.value()); cudaMallocAsync(&temp.d_fsst_len, 255, stream.value()); cudaMallocAsync(&temp.d_fsst_sym, 255*sizeof(unsigned long long), stream.value()); }
    size_t ci=0, ce=0;
    cub::DeviceScan::InclusiveSum(nullptr,ci,(uint32_t*)nullptr,(uint32_t*)nullptr,max_seg_rows,stream.value());
    cub::DeviceScan::ExclusiveSum(nullptr,ce,(uint32_t*)nullptr,(uint32_t*)nullptr,max_seg_rows,stream.value());
    temp.cub_temp_bytes = std::max(ci,ce);
    cudaMallocAsync(&temp.d_cub_temp, temp.cub_temp_bytes, stream.value());
  }

  constexpr uint32_t ADJ_THREADS = 256;
  size_t actual_cum_chars = 0;
  for (size_t si = 0; si < col_scan.data.segments.size(); ++si) {
    auto& seg = col_scan.data.segments[si]; auto& layout = layouts[si];
    if (seg.row_count == 0) continue;
    int32_t* d_seg_offsets = d_offsets.data() + layout.row_start;
    size_t seg_char_start = actual_cum_chars;

    if (!seg.persistent || !seg.data_ptr || seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
      cudaMemsetAsync(d_seg_offsets, 0, (seg.row_count+1)*sizeof(int32_t), stream.value());
    } else {
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      uint8_t* d_seg_chars = d_chars_base + seg_char_start;
      auto it = blocks.offsets.find(seg.block_id);
      void* d_block = (it != blocks.offsets.end()) ? static_cast<void*>(device_staging + it->second) : nullptr;
      bool fd = (d_block != nullptr);

      if (seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED)
        gpu_decode_uncompressed_string(block_base, DUCKDB_BLOCK_SIZE, seg.block_offset, seg.row_count, d_seg_offsets, d_seg_chars, stream, fd?d_block:nullptr, fd);
      else if (seg.compression == duckdb::CompressionType::COMPRESSION_DICTIONARY) {
        uint8_t* co=nullptr; size_t tc=0;
        gpu_decode_dictionary(block_base, DUCKDB_BLOCK_SIZE, seg.block_offset, DUCKDB_BLOCK_SIZE, seg.row_count, d_seg_offsets, &co, &tc, stream, fd?d_block:nullptr, seg.max_string_length, d_seg_chars, &temp, fd);
      } else if (seg.compression == duckdb::CompressionType::COMPRESSION_FSST)
        gpu_decode_fsst(block_base, DUCKDB_BLOCK_SIZE, seg.block_offset, seg.row_count, d_seg_offsets, d_seg_chars, stream, fd?d_block:nullptr, &temp, fd);
    }

    int32_t sac = 0; stream.synchronize();
    cudaMemcpy(&sac, d_seg_offsets + seg.row_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    actual_cum_chars += sac;

    if (seg_char_start > 0) {
      bool is_last = (layout.row_start + seg.row_count >= total_rows);
      uint32_t ac = seg.row_count + (is_last ? 1 : 0);
      kernel_adjust_offsets<<<(ac+ADJ_THREADS-1)/ADJ_THREADS, ADJ_THREADS, 0, stream.value()>>>(d_seg_offsets, static_cast<int32_t>(seg_char_start), ac);
    }
  }
  stream.synchronize();

  if (max_seg_rows > 0) {
    cudaFreeAsync(temp.d_buf_a, stream.value()); cudaFreeAsync(temp.d_buf_b, stream.value());
    if (has_fsst) { cudaFreeAsync(temp.d_buf_c, stream.value()); cudaFreeAsync(temp.d_fsst_len, stream.value()); cudaFreeAsync(temp.d_fsst_sym, stream.value()); }
    cudaFreeAsync(temp.d_cub_temp, stream.value());
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(total_rows+1), d_offsets.release(), rmm::device_buffer{0,stream,mr}, 0);

  rmm::device_buffer null_mask{}; cudf::size_type null_count = 0;
  if (col_scan.has_nulls) {
    size_t mask_bytes = (total_rows+63)/64*sizeof(uint64_t);
    null_mask = rmm::device_buffer(mask_bytes, stream, mr);
    auto* dm = static_cast<uint64_t*>(null_mask.data());
    uint32_t nw = (total_rows+63)/64;
    kernel_fill_valid<<<(nw+255)/256,256,0,stream.value()>>>(dm, nw);
    size_t vo = 0;
    for (auto& vs : col_scan.validity.segments) {
      if (vs.row_count == 0) { vo += vs.row_count; continue; }
      if (vs.persistent && vs.data_ptr && vs.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        size_t mb = (vs.row_count+7)/8;
        if (vo%64==0) cudaMemcpyAsync(dm+vo/64, vs.data_ptr, mb, cudaMemcpyHostToDevice, stream.value());
        else cudaMemcpyAsync(reinterpret_cast<uint8_t*>(dm)+vo/8, vs.data_ptr, mb, cudaMemcpyHostToDevice, stream.value());
      }
      vo += vs.row_count;
    }
    uint32_t* d_vc; cudaMallocAsync(&d_vc, sizeof(uint32_t), stream.value());
    cudaMemsetAsync(d_vc, 0, sizeof(uint32_t), stream.value());
    kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        dm, nw, static_cast<uint32_t>(total_rows), d_vc);
    stream.synchronize();
    uint32_t vc; cudaMemcpy(&vc, d_vc, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_vc, stream.value());
    null_count = static_cast<cudf::size_type>(total_rows - vc);
  }
  return cudf::make_strings_column(static_cast<cudf::size_type>(total_rows), std::move(offsets_col), std::move(d_chars), null_count, std::move(null_mask));
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API: pipelined decode from pre-transferred device data
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> gpu_decode_table_pipelined(
    std::vector<column_scan_result>& col_scans,
    const std::vector<duckdb::LogicalType>& col_types,
    const device_block_map& blocks,
    void* device_staging,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  if (col_scans.size() != col_types.size())
    throw std::invalid_argument("gpu_decode_table_pipelined: size mismatch");

  auto* d_staging = static_cast<uint8_t*>(device_staging);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(col_scans.size());

  for (size_t ci = 0; ci < col_scans.size(); ++ci) {
    if (col_types[ci].id() == duckdb::LogicalTypeId::VARCHAR)
      columns.push_back(decode_string_column_batched(
          col_scans[ci], stream, mr,
          &blocks.offsets, static_cast<uint8_t*>(d_staging)));
    else
      columns.push_back(decode_fixed_width_column_from_device(col_scans[ci], col_types[ci], blocks, d_staging, stream, mr));
  }

  stream.synchronize();
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::cuda::scan
