/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "cuda/scan/gpu_decode.cuh"
#include "cuda/scan/gpu_decode_batched_string.cuh"
#include "cuda/scan/gpu_native_decode.cuh"
#include "log/logging.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <duckdb/common/types.hpp>

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
    case duckdb::LogicalTypeId::TINYINT: return cudf::data_type(cudf::type_id::INT8);
    case duckdb::LogicalTypeId::SMALLINT: return cudf::data_type(cudf::type_id::INT16);
    case duckdb::LogicalTypeId::INTEGER: return cudf::data_type(cudf::type_id::INT32);
    case duckdb::LogicalTypeId::BIGINT: return cudf::data_type(cudf::type_id::INT64);
    case duckdb::LogicalTypeId::UTINYINT: return cudf::data_type(cudf::type_id::UINT8);
    case duckdb::LogicalTypeId::USMALLINT: return cudf::data_type(cudf::type_id::UINT16);
    case duckdb::LogicalTypeId::UINTEGER: return cudf::data_type(cudf::type_id::UINT32);
    case duckdb::LogicalTypeId::UBIGINT: return cudf::data_type(cudf::type_id::UINT64);
    case duckdb::LogicalTypeId::FLOAT: return cudf::data_type(cudf::type_id::FLOAT32);
    case duckdb::LogicalTypeId::DOUBLE: return cudf::data_type(cudf::type_id::FLOAT64);
    case duckdb::LogicalTypeId::BOOLEAN: return cudf::data_type(cudf::type_id::BOOL8);
    case duckdb::LogicalTypeId::DATE: return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
    case duckdb::LogicalTypeId::TIMESTAMP:
      return cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
    case duckdb::LogicalTypeId::VARCHAR: return cudf::data_type(cudf::type_id::STRING);
    case duckdb::LogicalTypeId::HUGEINT: return cudf::data_type(cudf::type_id::INT64);
    case duckdb::LogicalTypeId::DECIMAL: {
      switch (type.InternalType()) {
        case duckdb::PhysicalType::INT32:
          return cudf::data_type(cudf::type_id::DECIMAL32, -duckdb::DecimalType::GetScale(type));
        case duckdb::PhysicalType::INT64:
          return cudf::data_type(cudf::type_id::DECIMAL64, -duckdb::DecimalType::GetScale(type));
        case duckdb::PhysicalType::INT128:
          return cudf::data_type(cudf::type_id::DECIMAL128, -duckdb::DecimalType::GetScale(type));
        default: break;
      }
    }
    default: break;
  }
  throw std::runtime_error("gpu_native_decode: unsupported DuckDB type " + type.ToString());
}

/// Get byte size of a DuckDB physical type.
uint32_t get_type_size(duckdb::PhysicalType pt)
{
  switch (pt) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8:
    case duckdb::PhysicalType::UINT8: return 1;
    case duckdb::PhysicalType::INT16:
    case duckdb::PhysicalType::UINT16: return 2;
    case duckdb::PhysicalType::INT32:
    case duckdb::PhysicalType::UINT32:
    case duckdb::PhysicalType::FLOAT: return 4;
    case duckdb::PhysicalType::INT64:
    case duckdb::PhysicalType::UINT64:
    case duckdb::PhysicalType::DOUBLE: return 8;
    case duckdb::PhysicalType::INT128: return 16;
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
    case duckdb::PhysicalType::INT128: return true;
    default: return false;
  }
}

/// CUDA kernel to fill a buffer with a constant value (for CONSTANT segments).
template <typename T>
__global__ void kernel_fill_constant(T* output, T value, uint32_t count)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) { output[idx] = value; }
}

/// Launch kernel_fill_constant with the right type based on type_size.
/// Reads the constant value from val_src (host pointer).
void launch_fill_constant(uint8_t* d_dest,
                          const uint8_t* val_src,
                          uint32_t type_size,
                          uint32_t row_count,
                          cudaStream_t stream)
{
  uint32_t blocks = (row_count + 255) / 256;
  switch (type_size) {
    case 1: {
      int8_t v;
      std::memcpy(&v, val_src, 1);
      kernel_fill_constant<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<int8_t*>(d_dest), v, row_count);
      break;
    }
    case 2: {
      int16_t v;
      std::memcpy(&v, val_src, 2);
      kernel_fill_constant<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<int16_t*>(d_dest), v, row_count);
      break;
    }
    case 4: {
      int32_t v;
      std::memcpy(&v, val_src, 4);
      kernel_fill_constant<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<int32_t*>(d_dest), v, row_count);
      break;
    }
    case 8: {
      int64_t v;
      std::memcpy(&v, val_src, 8);
      kernel_fill_constant<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<int64_t*>(d_dest), v, row_count);
      break;
    }
    default: {
      // Fallback: expand on host and memcpy
      std::vector<uint8_t> host_buf(static_cast<size_t>(row_count) * type_size);
      for (uint32_t r = 0; r < row_count; ++r)
        std::memcpy(host_buf.data() + r * type_size, val_src, type_size);
      cudaMemcpyAsync(d_dest, host_buf.data(), host_buf.size(), cudaMemcpyHostToDevice, stream);
      break;
    }
  }
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
__global__ void kernel_count_valid_bits(const uint64_t* __restrict__ mask,
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

// DUCKDB_BLOCK_SIZE is defined in cuda/scan/gpu_decode.cuh

//===----------------------------------------------------------------------===//
// Fixed-width column decode
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> decode_fixed_width_column(column_scan_result& col_scan,
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

  if (type_size == 0 || total_rows == 0) { return cudf::make_empty_column(cudf_type); }

  // Allocate output data buffer on GPU
  rmm::device_buffer data_buf(total_rows * type_size, stream, mr);
  auto* d_output = static_cast<uint8_t*>(data_buf.data());

  // Pre-allocate RLE cumsum scratch buffer (reused across segments).
  // 4096 entries covers typical RLE segments; larger ones fall back to alloc.
  constexpr size_t RLE_CUMSUM_CAPACITY = 4096 * sizeof(uint32_t);
  uint32_t* d_rle_cumsum               = nullptr;
  bool has_rle                         = false;
  for (auto const& seg : col_scan.data.segments) {
    if (seg.compression == duckdb::CompressionType::COMPRESSION_RLE) {
      has_rle = true;
      break;
    }
  }
  if (has_rle) { cudaMallocAsync(&d_rle_cumsum, RLE_CUMSUM_CAPACITY, stream.value()); }

  size_t row_offset       = 0;
  size_t gpu_decoded_segs = 0;
  // Tracks last block base to skip redundant H2D when consecutive segments
  // share the same 256KB DuckDB block. This is safe because DuckDB's segment
  // tree is ordered by row offset within a column, so segments from the same
  // block always appear consecutively.
  const uint8_t* last_block_base = nullptr;

  for (auto& seg : col_scan.data.segments) {
    if (seg.row_count == 0) {
      row_offset += seg.row_count;
      continue;
    }
    // Skip non-decodable segments, but allow blockless CONSTANT segments
    // (persistent=true, data_ptr=null, value in constant_data).
    if (!seg.persistent ||
        (!seg.data_ptr && seg.compression != duckdb::CompressionType::COMPRESSION_CONSTANT)) {
      row_offset += seg.row_count;
      continue;
    }

    auto* d_dest     = d_output + row_offset * type_size;
    size_t seg_bytes = seg.row_count * type_size;

    switch (seg.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        cudaMemcpyAsync(d_dest, seg.data_ptr, seg_bytes, cudaMemcpyHostToDevice, stream.value());
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_CONSTANT: {
        const uint8_t* val_src = seg.data_ptr ? seg.data_ptr : seg.constant_data;
        launch_fill_constant(d_dest, val_src, type_size, seg.row_count, stream.value());
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_BITPACKING: {
        const uint8_t* block_base = seg.data_ptr - seg.block_offset;
        bool block_cached         = (block_base == last_block_base);
        if (!block_cached) last_block_base = block_base;
        gpu_decode_bitpacking(block_base,
                              DUCKDB_BLOCK_SIZE,
                              static_cast<uint32_t>(seg.block_offset),
                              static_cast<uint32_t>(seg.row_count),
                              type_size,
                              is_signed,
                              d_dest,
                              stream,
                              d_scratch,
                              block_cached);
        gpu_decoded_segs++;
        break;
      }

      case duckdb::CompressionType::COMPRESSION_RLE: {
        const uint8_t* block_base = seg.data_ptr - seg.block_offset;
        bool block_cached         = (block_base == last_block_base);
        if (!block_cached) last_block_base = block_base;
        gpu_decode_rle(block_base,
                       DUCKDB_BLOCK_SIZE,
                       static_cast<uint32_t>(seg.block_offset),
                       static_cast<uint32_t>(seg.row_count),
                       type_size,
                       d_dest,
                       stream,
                       d_scratch,
                       block_cached,
                       d_rle_cumsum,
                       RLE_CUMSUM_CAPACITY);
        gpu_decoded_segs++;
        break;
      }

      default: {
        throw std::runtime_error("gpu_native_decode: unsupported compression type " +
                                 std::to_string(static_cast<int>(seg.compression)) +
                                 " for fixed-width column — falling back to CPU scan");
      }
    }

    row_offset += seg.row_count;
  }

  // Free RLE cumsum scratch
  if (d_rle_cumsum) { cudaFreeAsync(d_rle_cumsum, stream.value()); }

  // Decode validity
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;

  if (col_scan.has_nulls) {
    size_t mask_bytes = (total_rows + 63) / 64 * sizeof(uint64_t);
    null_mask         = rmm::device_buffer(mask_bytes, stream, mr);
    auto* d_mask      = static_cast<uint64_t*>(null_mask.data());

    // First set everything valid, then overlay actual validity segments
    uint32_t num_words   = static_cast<uint32_t>((total_rows + 63) / 64);
    uint32_t fill_blocks = (num_words + 255) / 256;
    kernel_fill_valid<<<fill_blocks, 256, 0, stream.value()>>>(d_mask, num_words);

    size_t val_row_offset = 0;
    for (auto& vseg : col_scan.validity.segments) {
      if (vseg.row_count == 0) {
        val_row_offset += vseg.row_count;
        continue;
      }

      if (vseg.persistent && vseg.data_ptr &&
          vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        // DuckDB validity bitmap is LSB-first uint64_t, same as cuDF.
        // Copy at the correct bit offset.
        size_t seg_mask_bytes = (vseg.row_count + 7) / 8;

        if (val_row_offset % 64 == 0) {
          // Aligned — direct copy to the correct word offset
          size_t word_offset = val_row_offset / 64;
          cudaMemcpyAsync(d_mask + word_offset,
                          vseg.data_ptr,
                          seg_mask_bytes,
                          cudaMemcpyHostToDevice,
                          stream.value());
        } else {
          // Unaligned — copy to host, do bit-shift, then upload
          // For simplicity, read validity on host and copy the full mask region
          std::vector<uint8_t> host_mask(seg_mask_bytes);
          std::memcpy(host_mask.data(), vseg.data_ptr, seg_mask_bytes);

          // Write to the device at the byte-aligned position
          size_t byte_offset = val_row_offset / 8;
          cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_mask) + byte_offset,
                          host_mask.data(),
                          seg_mask_bytes,
                          cudaMemcpyHostToDevice,
                          stream.value());
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
    cudaMemcpy(&valid_count, d_valid_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_valid_count, stream.value());
    null_count = static_cast<cudf::size_type>(total_rows - valid_count);
  }

  return std::make_unique<cudf::column>(cudf_type,
                                        static_cast<cudf::size_type>(total_rows),
                                        std::move(data_buf),
                                        std::move(null_mask),
                                        null_count);
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API: decode full table
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> gpu_decode_table(std::vector<column_scan_result>& column_scans,
                                              const std::vector<duckdb::LogicalType>& column_types,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  using clock  = std::chrono::steady_clock;
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
      us_string +=
        std::chrono::duration_cast<std::chrono::microseconds>(col_end - col_start).count();
      n_string++;
    } else {
      columns.push_back(decode_fixed_width_column(col_scan, col_type, stream, mr, d_scratch));
      auto col_end = clock::now();
      us_fixed +=
        std::chrono::duration_cast<std::chrono::microseconds>(col_end - col_start).count();
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

  SIRIUS_LOG_INFO(
    "[gpu_native_decode] table: {} cols ({} fixed + {} str), {} rows | "
    "alloc={:.1f}ms enqueue={:.1f}ms (fixed={:.1f}ms str={:.1f}ms) "
    "sync={:.1f}ms total={:.1f}ms",
    columns.size(),
    n_fixed,
    n_string,
    total_rows,
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

/// @param d_valid_count_out  If non-null, write valid bit count here (async)
///        and skip stream.synchronize(). Caller must sync + compute null_count.
std::unique_ptr<cudf::column> decode_fixed_width_column_from_device(
  column_scan_result& col_scan,
  const duckdb::LogicalType& type,
  const device_block_map& blocks,
  uint8_t* device_staging,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  uint32_t* d_valid_count_out = nullptr)
{
  auto cudf_type     = to_cudf_type(type);
  auto physical_type = type.InternalType();
  uint32_t type_size = get_type_size(physical_type);
  bool is_signed     = is_signed_type(physical_type);
  size_t total_rows  = col_scan.data.total_rows;
  if (type_size == 0 || total_rows == 0) return cudf::make_empty_column(cudf_type);

  rmm::device_buffer data_buf(total_rows * type_size, stream, mr);
  auto* d_output    = static_cast<uint8_t*>(data_buf.data());
  size_t row_offset = 0;

  // Pre-allocate RLE cumsum scratch (reused across segments)
  constexpr size_t RLE_CUMSUM_CAP = 4096 * sizeof(uint32_t);
  uint32_t* d_rle_cumsum          = nullptr;
  bool has_rle                    = false;
  for (auto const& seg : col_scan.data.segments) {
    if (seg.compression == duckdb::CompressionType::COMPRESSION_RLE) {
      has_rle = true;
      break;
    }
  }
  if (has_rle) cudaMallocAsync(&d_rle_cumsum, RLE_CUMSUM_CAP, stream.value());

  for (auto& seg : col_scan.data.segments) {
    if (seg.row_count == 0) {
      row_offset += seg.row_count;
      continue;
    }
    if (!seg.persistent ||
        (!seg.data_ptr && seg.compression != duckdb::CompressionType::COMPRESSION_CONSTANT)) {
      row_offset += seg.row_count;
      continue;
    }
    auto* d_dest     = d_output + row_offset * type_size;
    size_t seg_bytes = seg.row_count * type_size;

    switch (seg.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        auto it = blocks.offsets.find(seg.block_id);
        if (it != blocks.offsets.end())
          cudaMemcpyAsync(d_dest,
                          device_staging + it->second + seg.block_offset,
                          seg_bytes,
                          cudaMemcpyDeviceToDevice,
                          stream.value());
        else
          cudaMemcpyAsync(d_dest, seg.data_ptr, seg_bytes, cudaMemcpyHostToDevice, stream.value());
        break;
      }
      case duckdb::CompressionType::COMPRESSION_CONSTANT: {
        const uint8_t* vs = seg.data_ptr ? seg.data_ptr : seg.constant_data;
        launch_fill_constant(d_dest, vs, type_size, seg.row_count, stream.value());
        break;
      }
      case duckdb::CompressionType::COMPRESSION_BITPACKING: {
        auto it = blocks.offsets.find(seg.block_id);
        void* d_block =
          (it != blocks.offsets.end()) ? static_cast<void*>(device_staging + it->second) : nullptr;
        gpu_decode_bitpacking(seg.data_ptr - seg.block_offset,
                              DUCKDB_BLOCK_SIZE,
                              seg.block_offset,
                              seg.row_count,
                              type_size,
                              is_signed,
                              d_dest,
                              stream,
                              d_block ? d_block : nullptr,
                              d_block != nullptr);
        break;
      }
      case duckdb::CompressionType::COMPRESSION_RLE: {
        auto it = blocks.offsets.find(seg.block_id);
        void* d_block =
          (it != blocks.offsets.end()) ? static_cast<void*>(device_staging + it->second) : nullptr;
        gpu_decode_rle(seg.data_ptr - seg.block_offset,
                       DUCKDB_BLOCK_SIZE,
                       seg.block_offset,
                       seg.row_count,
                       type_size,
                       d_dest,
                       stream,
                       d_block ? d_block : nullptr,
                       d_block != nullptr,
                       d_rle_cumsum,
                       RLE_CUMSUM_CAP);
        break;
      }
      default: throw std::runtime_error("unsupported compression in pipelined decode");
    }
    row_offset += seg.row_count;
  }

  if (d_rle_cumsum) cudaFreeAsync(d_rle_cumsum, stream.value());

  // Validity
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;
  if (col_scan.has_nulls) {
    size_t mask_bytes  = (total_rows + 63) / 64 * sizeof(uint64_t);
    null_mask          = rmm::device_buffer(mask_bytes, stream, mr);
    auto* d_mask       = static_cast<uint64_t*>(null_mask.data());
    uint32_t num_words = (total_rows + 63) / 64;
    kernel_fill_valid<<<(num_words + 255) / 256, 256, 0, stream.value()>>>(d_mask, num_words);
    size_t vo = 0;
    for (auto& vs : col_scan.validity.segments) {
      if (vs.row_count == 0) {
        vo += vs.row_count;
        continue;
      }
      if (vs.persistent && vs.data_ptr &&
          vs.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        size_t mb = (vs.row_count + 7) / 8;
        if (vo % 64 == 0)
          cudaMemcpyAsync(
            d_mask + vo / 64, vs.data_ptr, mb, cudaMemcpyHostToDevice, stream.value());
        else
          cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_mask) + vo / 8,
                          vs.data_ptr,
                          mb,
                          cudaMemcpyHostToDevice,
                          stream.value());
      }
      vo += vs.row_count;
    }
    if (d_valid_count_out) {
      // Deferred: write valid count to caller's slot, NO sync.
      cudaMemsetAsync(d_valid_count_out, 0, sizeof(uint32_t), stream.value());
      kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_valid_count_out);
      // Caller will sync once after all columns, then call set_null_count.
    } else {
      // Legacy path: sync per column.
      uint32_t* d_vc;
      cudaMallocAsync(&d_vc, sizeof(uint32_t), stream.value());
      cudaMemsetAsync(d_vc, 0, sizeof(uint32_t), stream.value());
      kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
        d_mask, num_words, static_cast<uint32_t>(total_rows), d_vc);
      stream.synchronize();
      uint32_t vc;
      cudaMemcpy(&vc, d_vc, sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaFreeAsync(d_vc, stream.value());
      null_count = static_cast<cudf::size_type>(total_rows - vc);
    }
  }
  return std::make_unique<cudf::column>(cudf_type,
                                        static_cast<cudf::size_type>(total_rows),
                                        std::move(data_buf),
                                        std::move(null_mask),
                                        null_count);
}

//===----------------------------------------------------------------------===//
// Pipelined decode: string from pre-transferred device data (per-segment sync)
//===----------------------------------------------------------------------===//

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
  size_t num_cols = col_scans.size();

  // Pre-allocate device slots for deferred null count readback.
  // One slot per column with nulls — avoids N per-column syncs.
  std::vector<size_t> null_col_indices;
  for (size_t ci = 0; ci < num_cols; ++ci) {
    if (col_scans[ci].has_nulls) null_col_indices.push_back(ci);
  }
  uint32_t* d_valid_counts = nullptr;
  if (!null_col_indices.empty()) {
    cudaMallocAsync(&d_valid_counts, null_col_indices.size() * sizeof(uint32_t), stream.value());
  }

  // Map column index → slot index in d_valid_counts
  std::vector<int> col_to_slot(num_cols, -1);
  for (size_t i = 0; i < null_col_indices.size(); ++i) {
    col_to_slot[null_col_indices[i]] = static_cast<int>(i);
  }

  // Decode all columns — null count kernels launched but NOT synced.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(num_cols);

  for (size_t ci = 0; ci < num_cols; ++ci) {
    uint32_t* nc_slot = (col_to_slot[ci] >= 0) ? d_valid_counts + col_to_slot[ci] : nullptr;

    if (col_types[ci].id() == duckdb::LogicalTypeId::VARCHAR)
      columns.push_back(decode_string_column_batched(
        col_scans[ci], stream, mr, &blocks.offsets, d_staging, nc_slot));
    else
      columns.push_back(decode_fixed_width_column_from_device(
        col_scans[ci], col_types[ci], blocks, d_staging, stream, mr, nc_slot));
  }

  // ONE sync for all columns — replaces N per-column syncs.
  stream.synchronize();

  // Read back all valid counts and fix up null_count on each column.
  if (!null_col_indices.empty()) {
    std::vector<uint32_t> h_valid_counts(null_col_indices.size());
    cudaMemcpy(h_valid_counts.data(),
               d_valid_counts,
               null_col_indices.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_valid_counts, stream.value());

    for (size_t i = 0; i < null_col_indices.size(); ++i) {
      auto ci         = null_col_indices[i];
      auto total_rows = col_scans[ci].data.total_rows;
      auto nc         = static_cast<cudf::size_type>(total_rows - h_valid_counts[i]);
      columns[ci]->set_null_count(nc);
    }
  }

  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::cuda::scan
