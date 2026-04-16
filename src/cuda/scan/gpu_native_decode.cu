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

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <duckdb/common/types.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::cuda::scan {

using sirius::op::scan::column_scan_result;

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

//===----------------------------------------------------------------------===//
// Bulk block transfer: stage every unique block referenced by any column
// into one contiguous device buffer, issuing one cudaMemcpyAsync per
// contiguous run of block_ids (host pointers mmap'd by DuckDB are consecutive
// for consecutive block_ids, so runs can coalesce into a single memcpy).
//===----------------------------------------------------------------------===//

/// Map of DuckDB block_id → byte offset in the device staging buffer.
struct device_block_map {
  std::unordered_map<int64_t, size_t> offsets;
  size_t total_bytes = 0;
};

std::pair<device_block_map, rmm::device_buffer> transfer_blocks_bulk_h2d(
  const std::vector<column_scan_result>& col_scans,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Collect unique (block_id → host block-base ptr) across all segments.
  std::unordered_map<int64_t, const uint8_t*> seen;
  for (auto const& cs : col_scans) {
    for (auto const& seg : cs.data.segments) {
      if (!seg.persistent || !seg.data_ptr || seg.row_count == 0 || seg.block_id < 0) continue;
      seen.emplace(seg.block_id, seg.data_ptr - seg.block_offset);
    }
  }

  device_block_map map;
  if (seen.empty()) return {std::move(map), rmm::device_buffer{}};

  // Sort by block_id so we can coalesce contiguous runs.
  struct entry {
    int64_t block_id;
    const uint8_t* host_base;
  };
  std::vector<entry> sorted;
  sorted.reserve(seen.size());
  for (auto const& [id, ptr] : seen) sorted.push_back({id, ptr});
  std::sort(sorted.begin(), sorted.end(),
            [](entry const& a, entry const& b) { return a.block_id < b.block_id; });

  rmm::device_buffer staging(sorted.size() * DUCKDB_BLOCK_SIZE, stream, mr);
  auto* d_staging = static_cast<uint8_t*>(staging.data());

  size_t offset = 0;
  for (size_t i = 0; i < sorted.size();) {
    size_t run_start = i;
    while (i + 1 < sorted.size() &&
           sorted[i + 1].block_id == sorted[i].block_id + 1 &&
           sorted[i + 1].host_base == sorted[i].host_base + DUCKDB_BLOCK_SIZE) {
      ++i;
    }
    ++i;
    size_t run_len = i - run_start;

    cudaMemcpyAsync(d_staging + offset,
                    sorted[run_start].host_base,
                    run_len * DUCKDB_BLOCK_SIZE,
                    cudaMemcpyHostToDevice,
                    stream.value());

    for (size_t j = run_start; j < i; ++j) {
      map.offsets[sorted[j].block_id] = offset;
      offset += DUCKDB_BLOCK_SIZE;
    }
  }
  map.total_bytes = offset;
  return {std::move(map), std::move(staging)};
}

//===----------------------------------------------------------------------===//
// Validity decode: build a cuDF-compatible null mask and enqueue a GPU
// null-count.  When d_valid_count_out is provided, the count readback is
// deferred (caller syncs once after all columns and fixes up null_count).
//===----------------------------------------------------------------------===//

std::pair<rmm::device_buffer, cudf::size_type> decode_validity_mask(
  column_scan_result const& col_scan,
  size_t total_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  uint32_t* d_valid_count_out)
{
  rmm::device_buffer null_mask{};
  cudf::size_type null_count = 0;
  if (!col_scan.has_nulls || total_rows == 0) return {std::move(null_mask), null_count};

  size_t mask_bytes  = (total_rows + 63) / 64 * sizeof(uint64_t);
  null_mask          = rmm::device_buffer(mask_bytes, stream, mr);
  auto* d_mask       = static_cast<uint64_t*>(null_mask.data());
  uint32_t num_words = static_cast<uint32_t>((total_rows + 63) / 64);

  // Start all-valid, then overlay any actual validity segments.
  kernel_fill_valid<<<(num_words + 255) / 256, 256, 0, stream.value()>>>(d_mask, num_words);

  size_t row_offset = 0;
  for (auto const& vseg : col_scan.validity.segments) {
    if (vseg.row_count == 0) continue;

    // DuckDB validity is LSB-first uint64, same layout as cuDF.
    // Non-persistent or non-UNCOMPRESSED validity segments are all-valid
    // (EMPTY means no nulls in that range) — leave the prefilled bits.
    if (vseg.persistent && vseg.data_ptr &&
        vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
      size_t seg_mask_bytes = (vseg.row_count + 7) / 8;
      // Aligned and unaligned cases both write byte-aligned; DuckDB validity
      // segments are byte-aligned at row_offset/8 even when not word-aligned.
      cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d_mask) + row_offset / 8,
                      vseg.data_ptr,
                      seg_mask_bytes,
                      cudaMemcpyHostToDevice,
                      stream.value());
    }
    row_offset += vseg.row_count;
  }

  // Enqueue null-count kernel. Deferred readback when caller owns the slot.
  if (d_valid_count_out) {
    cudaMemsetAsync(d_valid_count_out, 0, sizeof(uint32_t), stream.value());
    kernel_count_valid_bits<<<1, 256, 0, stream.value()>>>(
      d_mask, num_words, static_cast<uint32_t>(total_rows), d_valid_count_out);
    // null_count stays 0; caller fixes up after the single post-decode sync.
  } else {
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
  return {std::move(null_mask), null_count};
}

//===----------------------------------------------------------------------===//
// Fixed-width column decode from pre-staged device blocks.
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> decode_fixed_width_column(
  column_scan_result& col_scan,
  duckdb::LogicalType const& type,
  device_block_map const& blocks,
  uint8_t* device_staging,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  uint32_t* d_valid_count_out)
{
  auto cudf_type     = to_cudf_type(type);
  auto physical_type = type.InternalType();
  uint32_t type_size = get_type_size(physical_type);
  bool is_signed     = is_signed_type(physical_type);
  size_t total_rows  = col_scan.data.total_rows;
  if (type_size == 0 || total_rows == 0) return cudf::make_empty_column(cudf_type);

  rmm::device_buffer data_buf(total_rows * type_size, stream, mr);
  auto* d_output = static_cast<uint8_t*>(data_buf.data());

  // RLE cumsum scratch, lazily allocated if any segment needs it.
  constexpr size_t RLE_CUMSUM_CAP = 4096 * sizeof(uint32_t);
  uint32_t* d_rle_cumsum          = nullptr;
  for (auto const& seg : col_scan.data.segments) {
    if (seg.compression == duckdb::CompressionType::COMPRESSION_RLE) {
      cudaMallocAsync(&d_rle_cumsum, RLE_CUMSUM_CAP, stream.value());
      break;
    }
  }

  // Bitpacking segments are batched into one kernel launch across the column.
  // When the block is pre-staged, block_ptr is the device pointer and
  // on_device=true; otherwise block_ptr is the host base (for fallback H2D).
  struct bp_info {
    const uint8_t* block_ptr;
    bool on_device;
    uint32_t block_offset;
    uint32_t row_count;
    size_t output_row_offset;
  };
  std::vector<bp_info> bp_segments;

  size_t row_offset = 0;
  for (auto& seg : col_scan.data.segments) {
    if (seg.row_count == 0) continue;
    // Blockless CONSTANT segments are allowed (data_ptr=null, value in constant_data).
    if (!seg.persistent ||
        (!seg.data_ptr && seg.compression != duckdb::CompressionType::COMPRESSION_CONSTANT)) {
      row_offset += seg.row_count;
      continue;
    }

    auto* d_dest     = d_output + row_offset * type_size;
    size_t seg_bytes = seg.row_count * type_size;
    auto blk_it      = blocks.offsets.find(seg.block_id);
    const uint8_t* d_block =
      (blk_it != blocks.offsets.end()) ? device_staging + blk_it->second : nullptr;

    switch (seg.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: {
        if (d_block) {
          cudaMemcpyAsync(d_dest, d_block + seg.block_offset, seg_bytes,
                          cudaMemcpyDeviceToDevice, stream.value());
        } else {
          // Block wasn't pre-staged (block_id < 0 etc.) — per-segment H2D.
          cudaMemcpyAsync(d_dest, seg.data_ptr, seg_bytes,
                          cudaMemcpyHostToDevice, stream.value());
        }
        break;
      }
      case duckdb::CompressionType::COMPRESSION_CONSTANT: {
        const uint8_t* vs = seg.data_ptr ? seg.data_ptr : seg.constant_data;
        launch_fill_constant(d_dest, vs, type_size, seg.row_count, stream.value());
        break;
      }
      case duckdb::CompressionType::COMPRESSION_BITPACKING: {
        bp_segments.push_back({d_block ? d_block : seg.data_ptr - seg.block_offset,
                               d_block != nullptr,
                               static_cast<uint32_t>(seg.block_offset),
                               static_cast<uint32_t>(seg.row_count),
                               row_offset});
        break;
      }
      case duckdb::CompressionType::COMPRESSION_RLE: {
        gpu_decode_rle(seg.data_ptr - seg.block_offset,
                       DUCKDB_BLOCK_SIZE,
                       seg.block_offset,
                       seg.row_count,
                       type_size,
                       d_dest,
                       stream,
                       d_block ? const_cast<void*>(static_cast<const void*>(d_block)) : nullptr,
                       /*skip_block_copy=*/d_block != nullptr,
                       d_rle_cumsum,
                       RLE_CUMSUM_CAP);
        break;
      }
      default:
        throw std::runtime_error("gpu_native_decode: unsupported compression type " +
                                 std::to_string(static_cast<int>(seg.compression)) +
                                 " for fixed-width column — falling back to CPU scan");
    }
    row_offset += seg.row_count;
  }

  // Batch all bitpacking segments into one kernel launch.
  if (!bp_segments.empty()) {
    // Stage any blocks that weren't pre-transferred (rare: block_id < 0).
    std::unordered_map<const uint8_t*, size_t> fb_map;  // host_base → device offset
    for (auto const& bp : bp_segments) {
      if (!bp.on_device) fb_map.emplace(bp.block_ptr, fb_map.size() * DUCKDB_BLOCK_SIZE);
    }
    uint8_t* d_fb_staging = nullptr;
    if (!fb_map.empty()) {
      cudaMallocAsync(&d_fb_staging, fb_map.size() * DUCKDB_BLOCK_SIZE, stream.value());
      for (auto const& [host_base, off] : fb_map) {
        cudaMemcpyAsync(d_fb_staging + off, host_base, DUCKDB_BLOCK_SIZE,
                        cudaMemcpyHostToDevice, stream.value());
      }
    }

    std::vector<batched_bp_seg_desc> descs;
    for (auto const& bp : bp_segments) {
      const uint8_t* d_block =
        bp.on_device ? bp.block_ptr : d_fb_staging + fb_map[bp.block_ptr];

      uint32_t num_groups = (bp.row_count + BP_META_GROUP_SIZE - 1) / BP_META_GROUP_SIZE;
      for (uint32_t g = 0; g < num_groups; ++g) {
        uint32_t group_rows =
          (g < num_groups - 1) ? BP_META_GROUP_SIZE : bp.row_count - g * BP_META_GROUP_SIZE;
        descs.push_back({d_block,
                         bp.block_offset,
                         g,
                         group_rows,
                         static_cast<uint32_t>(bp.output_row_offset + g * BP_META_GROUP_SIZE)});
      }
    }

    gpu_decode_bitpacking_batched(descs.data(),
                                  static_cast<uint32_t>(descs.size()),
                                  d_output,
                                  type_size,
                                  is_signed,
                                  stream);

    if (d_fb_staging) cudaFreeAsync(d_fb_staging, stream.value());
  }

  if (d_rle_cumsum) cudaFreeAsync(d_rle_cumsum, stream.value());

  auto [null_mask, null_count] =
    decode_validity_mask(col_scan, total_rows, stream, mr, d_valid_count_out);

  return std::make_unique<cudf::column>(cudf_type,
                                        static_cast<cudf::size_type>(total_rows),
                                        std::move(data_buf),
                                        std::move(null_mask),
                                        null_count);
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> gpu_decode_table(std::vector<column_scan_result>& col_scans,
                                              const std::vector<duckdb::LogicalType>& col_types,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  using clock  = std::chrono::steady_clock;
  auto t_start = clock::now();

  if (col_scans.size() != col_types.size()) {
    throw std::invalid_argument("gpu_decode_table: col_scans and col_types size mismatch");
  }

  size_t total_rows = col_scans.empty() ? 0 : col_scans[0].data.total_rows;
  size_t num_cols   = col_scans.size();

  // 1. Bulk-stage every unique block to device in one coalesced H2D pass.
  auto [blocks, staging_buf] = transfer_blocks_bulk_h2d(col_scans, stream, mr);
  auto* d_staging            = static_cast<uint8_t*>(staging_buf.data());

  auto t_stage = clock::now();

  // 2. Pre-allocate one device slot per null-bearing column for the deferred
  //    null-count readback — avoids a per-column sync.
  std::vector<size_t> null_col_indices;
  for (size_t ci = 0; ci < num_cols; ++ci) {
    if (col_scans[ci].has_nulls) null_col_indices.push_back(ci);
  }
  uint32_t* d_valid_counts = nullptr;
  if (!null_col_indices.empty()) {
    cudaMallocAsync(&d_valid_counts, null_col_indices.size() * sizeof(uint32_t), stream.value());
  }
  std::vector<int> col_to_slot(num_cols, -1);
  for (size_t i = 0; i < null_col_indices.size(); ++i) {
    col_to_slot[null_col_indices[i]] = static_cast<int>(i);
  }

  // 3. Decode every column — all work enqueued on one stream, no syncs.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(num_cols);
  size_t n_fixed = 0, n_string = 0;
  double us_fixed = 0, us_string = 0;

  for (size_t ci = 0; ci < num_cols; ++ci) {
    uint32_t* nc_slot = (col_to_slot[ci] >= 0) ? d_valid_counts + col_to_slot[ci] : nullptr;
    auto col_start    = clock::now();
    if (col_types[ci].id() == duckdb::LogicalTypeId::VARCHAR) {
      columns.push_back(decode_string_column_batched(
        col_scans[ci], stream, mr, &blocks.offsets, d_staging, nc_slot));
      us_string += std::chrono::duration_cast<std::chrono::microseconds>(
                     clock::now() - col_start).count();
      ++n_string;
    } else {
      columns.push_back(decode_fixed_width_column(
        col_scans[ci], col_types[ci], blocks, d_staging, stream, mr, nc_slot));
      us_fixed += std::chrono::duration_cast<std::chrono::microseconds>(
                    clock::now() - col_start).count();
      ++n_fixed;
    }
  }

  auto t_decode = clock::now();

  // 4. Single sync for all enqueued work (H2D + decode + null-count kernels).
  stream.synchronize();

  // 5. Fix up null counts from the deferred slots.
  if (!null_col_indices.empty()) {
    std::vector<uint32_t> h_valid_counts(null_col_indices.size());
    cudaMemcpy(h_valid_counts.data(),
               d_valid_counts,
               null_col_indices.size() * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_valid_counts, stream.value());
    for (size_t i = 0; i < null_col_indices.size(); ++i) {
      auto ci = null_col_indices[i];
      auto nc =
        static_cast<cudf::size_type>(col_scans[ci].data.total_rows - h_valid_counts[i]);
      columns[ci]->set_null_count(nc);
    }
  }

  auto t_end                 = clock::now();
  [[maybe_unused]] auto us   = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
  };
  SIRIUS_LOG_INFO(
    "[gpu_native_decode] table: {} cols ({} fixed + {} str), {} rows, {} blocks | "
    "stage={:.1f}ms enqueue={:.1f}ms (fixed={:.1f}ms str={:.1f}ms) "
    "sync+nulls={:.1f}ms total={:.1f}ms",
    num_cols, n_fixed, n_string, total_rows, blocks.offsets.size(),
    us(t_start, t_stage) / 1000.0,
    us(t_stage, t_decode) / 1000.0,
    us_fixed / 1000.0,
    us_string / 1000.0,
    us(t_decode, t_end) / 1000.0,
    us(t_start, t_end) / 1000.0);

  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::cuda::scan
