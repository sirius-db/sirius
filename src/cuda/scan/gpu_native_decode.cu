/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_native_decode.cuh"
#include "cuda/scan/gpu_decode.cuh"
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

constexpr size_t DUCKDB_BLOCK_SIZE = 262144;  // 256KB

//===----------------------------------------------------------------------===//
// Fixed-width column decode
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> decode_fixed_width_column(
    column_scan_result& col_scan,
    const duckdb::LogicalType& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    void* d_scratch,
    void* d_meta_scratch,
    size_t meta_scratch_size)
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

  size_t row_offset = 0;
  size_t gpu_decoded_segs = 0;

  for (auto& seg : col_scan.data.segments) {
    if (!seg.persistent || !seg.data_ptr || seg.row_count == 0) {
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
        if (type_size == 4) {
          int32_t val; std::memcpy(&val, seg.data_ptr, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int32_t*>(d_dest), val, seg.row_count);
        } else if (type_size == 8) {
          int64_t val; std::memcpy(&val, seg.data_ptr, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int64_t*>(d_dest), val, seg.row_count);
        } else if (type_size == 2) {
          int16_t val; std::memcpy(&val, seg.data_ptr, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int16_t*>(d_dest), val, seg.row_count);
        } else if (type_size == 1) {
          int8_t val; std::memcpy(&val, seg.data_ptr, sizeof(val));
          uint32_t blocks = (seg.row_count + 255) / 256;
          kernel_fill_constant<<<blocks, 256, 0, stream.value()>>>(
              reinterpret_cast<int8_t*>(d_dest), val, seg.row_count);
        } else {
          std::vector<uint8_t> host_buf(seg_bytes);
          for (size_t r = 0; r < seg.row_count; ++r) {
            std::memcpy(host_buf.data() + r * type_size, seg.data_ptr, type_size);
          }
          cudaMemcpyAsync(d_dest, host_buf.data(), seg_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        }
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_BITPACKING: {
        const uint8_t* block_base = seg.data_ptr - seg.block_offset;
        gpu_decode_bitpacking(
            block_base, DUCKDB_BLOCK_SIZE,
            static_cast<uint32_t>(seg.block_offset),
            static_cast<uint32_t>(seg.row_count),
            type_size, is_signed,
            d_dest, stream,
            d_scratch, d_meta_scratch, meta_scratch_size);
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

    // Count nulls: sync and count on host (for column metadata).
    // Mask off tail bits beyond total_rows in the last word to avoid
    // counting padding bits as valid.
    stream.synchronize();
    std::vector<uint64_t> host_mask_copy(num_words);
    cudaMemcpy(host_mask_copy.data(), d_mask, num_words * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    size_t tail_bits = total_rows % 64;
    if (tail_bits > 0 && num_words > 0) {
      uint64_t tail_mask = (1ULL << tail_bits) - 1;
      host_mask_copy[num_words - 1] &= tail_mask;
    }

    size_t valid_count = 0;
    for (uint32_t w = 0; w < num_words; ++w) {
      valid_count += __builtin_popcountll(host_mask_copy[w]);
    }
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
         || seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED)
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
  // No D2H, no host merge, no H2D re-upload for string data.
  //===----------------------------------------------------------------------===//

  // Phase 1: Compute per-segment layout on host
  struct seg_layout {
    size_t row_start;       // first row index in the final column
    size_t char_start;      // byte offset in the final chars buffer
    size_t char_capacity;   // row_count * max_string_length (upper bound)
  };
  std::vector<seg_layout> layouts;
  layouts.reserve(col_scan.data.segments.size());

  size_t cum_rows = 0, cum_chars = 0;
  for (auto const& seg : col_scan.data.segments) {
    seg_layout l;
    l.row_start = cum_rows;
    l.char_start = cum_chars;
    if (seg.row_count > 0 && seg.persistent && seg.data_ptr
        && (seg.compression == duckdb::CompressionType::COMPRESSION_DICTIONARY
            || seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED)) {
      l.char_capacity = static_cast<size_t>(seg.row_count) * seg.max_string_length;
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

  // Phase 3: Decode each segment directly into its slice of the final buffers
  constexpr uint32_t ADJ_THREADS = 256;

  for (size_t si = 0; si < col_scan.data.segments.size(); ++si) {
    auto& seg = col_scan.data.segments[si];
    auto& layout = layouts[si];
    if (seg.row_count == 0) continue;

    int32_t* d_seg_offsets = d_offsets.data() + layout.row_start;

    if (!seg.persistent || !seg.data_ptr
        || seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
      // Empty / constant: zero offsets for this segment
      cudaMemsetAsync(d_seg_offsets, 0,
                      (seg.row_count + 1) * sizeof(int32_t), stream.value());
    } else if (seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      uint8_t* d_seg_chars = d_chars_base + layout.char_start;

      gpu_decode_uncompressed_string(
          block_base, DUCKDB_BLOCK_SIZE,
          static_cast<uint32_t>(seg.block_offset),
          static_cast<uint32_t>(seg.row_count),
          d_seg_offsets,
          d_seg_chars,
          stream,
          d_scratch);
    } else if (seg.compression == duckdb::CompressionType::COMPRESSION_DICTIONARY) {
      const uint8_t* block_base = seg.data_ptr - seg.block_offset;
      uint8_t* d_seg_chars = d_chars_base + layout.char_start;

      uint8_t* d_chars_out = nullptr;
      size_t total_chars_out = 0;

      gpu_decode_dictionary(
          block_base, DUCKDB_BLOCK_SIZE,
          static_cast<uint32_t>(seg.block_offset),
          static_cast<uint32_t>(DUCKDB_BLOCK_SIZE),
          static_cast<uint32_t>(seg.row_count),
          d_seg_offsets,
          &d_chars_out,
          &total_chars_out,
          stream,
          d_scratch,
          seg.max_string_length,
          d_seg_chars);  // pre-allocated: write chars here
    }

    // Adjust this segment's offsets by adding the char slice start.
    // For the last segment, also adjust the sentinel at d_offsets[total_rows].
    if (layout.char_start > 0) {
      bool is_last = (layout.row_start + seg.row_count >= total_rows);
      uint32_t adj_count = static_cast<uint32_t>(seg.row_count) + (is_last ? 1 : 0);
      uint32_t adj_blocks = (adj_count + ADJ_THREADS - 1) / ADJ_THREADS;
      kernel_adjust_offsets<<<adj_blocks, ADJ_THREADS, 0, stream.value()>>>(
          d_seg_offsets,
          static_cast<int32_t>(layout.char_start),
          adj_count);
    }
  }

  // Phase 4: Single sync — all GPU work is done
  stream.synchronize();

  // Read the sentinel to get the actual total chars (for logging)
  int32_t actual_total_chars = 0;
  cudaMemcpy(&actual_total_chars, d_offsets.data() + total_rows,
             sizeof(int32_t), cudaMemcpyDeviceToHost);

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

    stream.synchronize();
    std::vector<uint64_t> host_mask(num_words);
    cudaMemcpy(host_mask.data(), d_mask, num_words * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    size_t tail_bits = total_rows % 64;
    if (tail_bits > 0 && num_words > 0) {
      uint64_t tail_bitmask = (1ULL << tail_bits) - 1;
      host_mask[num_words - 1] &= tail_bitmask;
    }

    size_t valid_count = 0;
    for (uint32_t w = 0; w < num_words; ++w) {
      valid_count += __builtin_popcountll(host_mask[w]);
    }
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

  // Pre-allocate scratch buffers — reused across all segments and columns.
  // One block-sized buffer for H2D segment data, one for bitpacking metadata.
  constexpr size_t META_SCRATCH_SIZE = 64 * 1024;  // 64KB, enough for ~1300 groups
  void* d_scratch = nullptr;
  void* d_meta_scratch = nullptr;
  cudaMallocAsync(&d_scratch, DUCKDB_BLOCK_SIZE, stream.value());
  cudaMallocAsync(&d_meta_scratch, META_SCRATCH_SIZE, stream.value());

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
      columns.push_back(decode_string_column(col_scan, stream, mr, d_scratch));
      auto col_end = clock::now();
      us_string += std::chrono::duration_cast<std::chrono::microseconds>(col_end - col_start).count();
      n_string++;
    } else {
      columns.push_back(decode_fixed_width_column(
          col_scan, col_type, stream, mr,
          d_scratch, d_meta_scratch, META_SCRATCH_SIZE));
      auto col_end = clock::now();
      us_fixed += std::chrono::duration_cast<std::chrono::microseconds>(col_end - col_start).count();
      n_fixed++;
    }
  }

  auto t_decode = clock::now();

  // Single sync point for all async decode work
  stream.synchronize();

  auto t_sync = clock::now();

  // Free scratch buffers
  cudaFreeAsync(d_scratch, stream.value());
  cudaFreeAsync(d_meta_scratch, stream.value());

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

}  // namespace sirius::cuda::scan
