/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/scan/duckdb_native_decoder.hpp"

#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/gpu_native_decode.cuh"
#include "cudf/cudf_utils.hpp"
#include "helper/type_conversions.hpp"
#include "io/io_context.hpp"
#include "io/sirius_datasource.hpp"
#include "io/types.hpp"
#include "op/scan/duckdb_block_layout.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "sirius_context.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/compression/roaring/roaring.hpp>
#include <duckdb/storage/single_file_block_manager.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/storage/table/column_segment.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

using ::sirius::cuda::scan::gpu_codec_run;
using ::sirius::cuda::scan::gpu_column_decode_input;
using ::sirius::cuda::scan::gpu_segment_desc;
using ::sirius::cuda::scan::gpu_string_codec_run;
using ::sirius::cuda::scan::gpu_string_column_decode_input;
using ::sirius::cuda::scan::gpu_string_segment_desc;

constexpr char const* kTag = "[sirius_gpu_duckdb_native_scan]";

void throw_unsupported(std::string what)
{
  throw std::runtime_error(std::string(kTag) + " unsupported: " + std::move(what));
}

bool is_constant_or_empty_validity(duckdb::CompressionType c)
{
  return c == duckdb::CompressionType::COMPRESSION_CONSTANT ||
         c == duckdb::CompressionType::COMPRESSION_EMPTY;
}

bool is_supported_fixed_width_codec(duckdb::CompressionType c)
{
  switch (c) {
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_RLE:
    case duckdb::CompressionType::COMPRESSION_BITPACKING:
    case duckdb::CompressionType::COMPRESSION_ALP:
    case duckdb::CompressionType::COMPRESSION_ALPRD: return true;
    default: return false;
  }
}

bool is_supported_varchar_codec(duckdb::CompressionType c)
{
  switch (c) {
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_DICTIONARY:
    case duckdb::CompressionType::COMPRESSION_FSST:
    case duckdb::CompressionType::COMPRESSION_DICT_FSST: return true;
    default: return false;
  }
}

bool column_has_real_nulls(duckdb_column_metadata const& col)
{
  for (auto const& v : col.validity_segments) {
    if (v.all_null) { return true; }
    auto c = v.compression;
    if (c == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED ||
        c == duckdb::CompressionType::COMPRESSION_ROARING) {
      return true;
    }
  }
  return false;
}

cudf::data_type sirius_to_cudf_type(sirius::logical_type const& t)
{
  return duckdb::GetCudfType(sirius::to_duckdb(t));
}

/// @brief Pinned host bytes for a segment (only for CONSTANT and ROARING validity segments for now)
struct pinned_segment_bytes {
  std::vector<uint8_t> owned_bytes;  // used for CONSTANT, ROARING, concat
  uint8_t const* host_ptr = nullptr;
  std::size_t bytes       = 0;
};

//===----------------------------------------------------------------------===//
// CONSTANT extraction.
//
// CONSTANT segments have block_id == -1; the constant value lives in the
// segment's own statistics, snapshotted onto the descriptor by the walker.
// Copy the value into an owned buffer the kernel can read.
//===----------------------------------------------------------------------===//

template <typename T>
void store_typed_min(duckdb::BaseStatistics const& stats, std::vector<uint8_t>& out)
{
  auto v = duckdb::NumericStats::GetMin<T>(stats);
  out.resize(sizeof(T));
  std::memcpy(out.data(), &v, sizeof(T));
}

pinned_segment_bytes extract_constant_bytes(duckdb::BaseStatistics const& stats,
                                            sirius::logical_type const& sirius_type)
{
  auto duckdb_type = sirius::to_duckdb(sirius_type);
  pinned_segment_bytes out;
  switch (duckdb_type.InternalType()) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8: store_typed_min<int8_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::INT16: store_typed_min<int16_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::INT32: store_typed_min<int32_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::INT64: store_typed_min<int64_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT8: store_typed_min<uint8_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT16: store_typed_min<uint16_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT32: store_typed_min<uint32_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::UINT64: store_typed_min<uint64_t>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::FLOAT: store_typed_min<float>(stats, out.owned_bytes); break;
    case duckdb::PhysicalType::DOUBLE: store_typed_min<double>(stats, out.owned_bytes); break;
    default:
      throw_unsupported("CONSTANT extraction for physical type " +
                        std::to_string(static_cast<int>(duckdb_type.InternalType())));
  }
  out.host_ptr = out.owned_bytes.data();
  out.bytes    = out.owned_bytes.size();
  return out;
}

//===----------------------------------------------------------------------===//
// ROARING validity host-decode. Reconstruct a transient ColumnSegment via the
// public factory, drive RoaringScanState chunk-by-chunk. owned_bytes is
// pre-filled 0xff so AllValid chunks need no memcpy. CHUNK is a multiple of 8
// so byte_offset never splits across chunks.
//===----------------------------------------------------------------------===//

pinned_segment_bytes decode_roaring_validity(duckdb::DatabaseInstance& db,
                                             duckdb::BlockManager& block_manager,
                                             duckdb_segment_descriptor const& desc)
{
  constexpr duckdb::idx_t CHUNK = duckdb::roaring::ROARING_CONTAINER_SIZE;

  auto validity_type = duckdb::LogicalType(duckdb::LogicalTypeId::VALIDITY);
  auto seg           = duckdb::ColumnSegment::CreatePersistentSegment(
    db,
    block_manager,
    desc.block_id,
    desc.block_offset,
    validity_type,
    desc.segment_count,
    duckdb::CompressionType::COMPRESSION_ROARING,
    duckdb::BaseStatistics::CreateEmpty(validity_type),
    /*segment_state=*/nullptr);

  auto const row_count = static_cast<duckdb::idx_t>(desc.segment_count);
  pinned_segment_bytes out;
  std::size_t const words = (row_count + 63) / 64;
  out.owned_bytes.assign(words * sizeof(uint64_t), 0xff);

  duckdb::roaring::RoaringScanState rs(*seg);
  duckdb::Vector tmp(duckdb::LogicalType::BOOLEAN, CHUNK);

  for (duckdb::idx_t scanned = 0; scanned < row_count; scanned += CHUNK) {
    auto const to_scan = std::min<duckdb::idx_t>(CHUNK, row_count - scanned);
    auto& vm           = duckdb::FlatVector::Validity(tmp);
    vm.SetAllValid(CHUNK);
    rs.ScanPartial(scanned, tmp, /*offset=*/0, to_scan);
    if (!vm.AllValid()) {
      std::size_t const byte_offset   = scanned / 8;
      std::size_t const bytes_to_copy = (to_scan + 7) / 8;
      std::memcpy(out.owned_bytes.data() + byte_offset,
                  reinterpret_cast<uint8_t const*>(vm.GetData()),
                  bytes_to_copy);
    }
  }

  out.host_ptr = out.owned_bytes.data();
  out.bytes    = out.owned_bytes.size();
  return out;
}

// All-NULL constant runs carry no bytes anywhere; ship zeroed validity bits.
pinned_segment_bytes make_all_null_validity_bytes(duckdb::idx_t row_count)
{
  pinned_segment_bytes out;
  out.owned_bytes.assign((static_cast<std::size_t>(row_count) + 7) / 8, 0x00);
  if (row_count % 8 != 0) {
    // The GPU overlays validity runs by whole-byte copy onto an all-valid
    // mask, so the padding bits of a trailing partial byte must stay 1 or
    // they would null the next segment's leading rows. Every other validity
    // source (DuckDB buffers, the ROARING decode above) keeps them set.
    out.owned_bytes.back() = static_cast<uint8_t>(0xFFu << (row_count % 8));
  }
  out.host_ptr = out.owned_bytes.data();
  out.bytes    = out.owned_bytes.size();
  return out;
}

//===----------------------------------------------------------------------===//
// Per-split staging
//===----------------------------------------------------------------------===//

struct staged_segment {
  std::size_t device_offset  = 0;
  std::size_t bytes          = 0;
  uint32_t row_offset        = 0;
  uint32_t row_count         = 0;
  uint32_t max_string_length = 0;
  duckdb::CompressionType compression{duckdb::CompressionType::COMPRESSION_AUTO};
};

struct staged_column {
  std::vector<staged_segment> data;
  std::vector<staged_segment> validity;
  /// ARRAY child data segments (empty for non-ARRAY columns)
  std::vector<staged_segment> array_child_data;
  /// ARRAY child validity segments (empty for non-ARRAY or when child has no nulls)
  std::vector<staged_segment> array_child_validity;
  bool has_nulls               = false;
  bool child_has_nulls         = false;  // ARRAY child validity flag
  std::size_t total_rows       = 0;
  std::size_t total_child_rows = 0;  // For ARRAY: total_rows * array_size
  bool is_varchar              = false;
  bool is_array                = false;
};

/// @brief A .db block-payload range to read into device buffer at device_offset.
struct device_read_job {
  std::size_t file_offset;    ///< absolute byte offset in the .db file to read from
  std::size_t size;           ///< number of bytes to read
  std::size_t device_offset;  ///< byte offset in the device buffer to read into (16B-aligned)
};

/// @brief A host memory range to copy into device buffer at device_offset (for CPU-produced bytes,
/// e.g., CONSTANT or ROARING).
struct host_copy_job {
  uint8_t const* src_ptr;     ///< host pointer to copy from
  std::size_t size;           ///< number of bytes to copy
  std::size_t device_offset;  ///< byte offset in the device buffer to copy into (16B-aligned)
};

/// @brief Per-split staging state: the set of device read jobs and host copy jobs to prepare for a
/// single scan task, plus the pinned host memory for all source segments (kept alive until H2D is
/// synced).
struct staging_state {
  std::vector<device_read_job> reads;
  std::vector<host_copy_job> host_copies;
  std::vector<pinned_segment_bytes> pinned_segments;  // keep host_copy_job.host_ptr alive
  std::size_t running_offset = 0;
};

/// @brief Return the next aligned device offset and advance the staging_state's running_offset by
/// the given segment byte size.
std::size_t reserve_segment(staging_state& s, std::size_t bytes)
{
  // 16B alignment: kernels cast d_bytes to typed pointers up to uint128.
  constexpr std::size_t SEGMENT_ALIGN = 16;

  // Align up the current running offset, return the aligned offset, and advance the running offset
  // by the segment's byte size.
  s.running_offset  = (s.running_offset + SEGMENT_ALIGN - 1) & ~(SEGMENT_ALIGN - 1);
  auto const offset = s.running_offset;
  s.running_offset += bytes;
  return offset;
}

/// @brief Add a host_copy_job to the staging_state for the given pinned_segment_bytes, reserving
/// device space for the segment's bytes and recording the pinned host memory for lifetime
/// management.
void stage_host_copy(staging_state& s, pinned_segment_bytes p, staged_segment& out_seg)
{
  out_seg.device_offset = reserve_segment(s, p.bytes);
  out_seg.bytes         = p.bytes;
  s.host_copies.push_back({p.host_ptr, p.bytes, out_seg.device_offset});
  s.pinned_segments.push_back(std::move(p));  // keep pinned memory alive until H2D is synced
}

/// @brief Append the on-disk byte ranges for one segment to @p out, in device-staging order (main
/// payload, then whole-block overflow). Single source of the block_id -> file offset math, shared
/// by the prefetch hint (@ref row_group_file_ranges) and the decode reads (@ref stage_device_read).
void append_segment_file_ranges(duckdb::SingleFileBlockManager const& bm,
                                duckdb_segment_descriptor const& seg,
                                std::vector<cudf::io::text::byte_range_info>& out)
{
  // Main payload. CONSTANT/blockless segments (bytes_size == 0) and
  // host-backed segments (payload but no block) read no file.
  if (seg.bytes_size > 0 && seg.block_id >= 0) {
    auto const off =
      duckdb_block_payload_offset(bm, seg.block_id) + static_cast<std::size_t>(seg.block_offset);
    out.emplace_back(static_cast<std::int64_t>(off), static_cast<std::int64_t>(seg.bytes_size));
  }
  for (auto add_id : seg.additional_blocks) {  // whole-block overflow payloads
    out.emplace_back(static_cast<std::int64_t>(duckdb_block_payload_offset(bm, add_id)),
                     static_cast<std::int64_t>(bm.GetBlockSize()));
  }
}

/// @brief Reserve device space for @p seg and queue its reads. File ranges come from
/// @ref append_segment_file_ranges so decode reads exactly the bytes prefetch warmed; each range
/// lands contiguously in the device buffer, in order.
void stage_device_read(staging_state& s,
                       duckdb::SingleFileBlockManager const& bm,
                       duckdb_segment_descriptor const& seg,
                       staged_segment& out_seg)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  append_segment_file_ranges(bm, seg, ranges);

  std::size_t total_size = 0;
  for (auto const& r : ranges) {
    total_size += static_cast<std::size_t>(r.size());
  }

  out_seg.device_offset = reserve_segment(s, total_size);
  out_seg.bytes         = total_size;

  // Each range lands immediately after the previous one. Overflow blocks need no separate 16B
  // alignment: they follow the 16B-aligned segment base, and alignment is relative to that base.
  std::size_t dst_offset = 0;
  for (auto const& r : ranges) {
    s.reads.push_back({static_cast<std::size_t>(r.offset()),
                       static_cast<std::size_t>(r.size()),
                       out_seg.device_offset + dst_offset});
    dst_offset += static_cast<std::size_t>(r.size());
  }
}

// The walker guarantees CONSTANT descriptors carry their segment's stats.
duckdb::BaseStatistics const& constant_segment_stats(duckdb_segment_descriptor const& seg,
                                                     duckdb::idx_t column_id)
{
  if (!seg.segment_stats) {
    throw std::runtime_error(std::string(kTag) +
                             " CONSTANT segment without carried segment stats (column " +
                             std::to_string(column_id) + ")");
  }
  return *seg.segment_stats;
}

/// @brief Stage the data and validity segments for a fixed-width column, returning the staged
/// segments and metadata for the scan kernel. Throws if an unsupported codec is encountered.
staged_column stage_one_fixed_width_column(staging_state& s,
                                           duckdb::DatabaseInstance& db,
                                           duckdb::BlockManager& block_manager,
                                           duckdb::SingleFileBlockManager const& sf_bm,
                                           std::vector<duckdb_row_group_metadata> const& row_groups,
                                           std::size_t projected_col_idx,
                                           sirius::logical_type const& projected_type)
{
  staged_column out;

  uint32_t row_cursor = 0;
  for (const auto& rg : row_groups) {
    auto const& col_md = rg.columns.at(projected_col_idx);

    //===----------Data Segments----------===//
    for (auto const& seg : col_md.data_segments) {
      if (!is_supported_fixed_width_codec(seg.compression)) {
        throw_unsupported("fixed-width data codec " +
                          std::to_string(static_cast<int>(seg.compression)) + " (column " +
                          std::to_string(col_md.column_id) + ")");
      }
      staged_segment ss;
      ss.row_offset  = row_cursor + static_cast<uint32_t>(seg.segment_start);
      ss.row_count   = static_cast<uint32_t>(seg.segment_count);
      ss.compression = seg.compression;

      if (seg.host_ptr != nullptr) {
        // Host-backed segment: the bytes sit in host memory owned by the
        // enclosing scan_info, so stage a host copy instead of a file read.
        stage_host_copy(s, {{}, seg.host_ptr, seg.bytes_size}, ss);
      } else if (seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        auto const& stats = constant_segment_stats(seg, col_md.column_id);
        stage_host_copy(s, extract_constant_bytes(stats, projected_type), ss);
      } else {
        stage_device_read(s, sf_bm, seg, ss);
      }
      out.data.push_back(ss);
    }

    //===----------Validity Segments----------===//
    if (column_has_real_nulls(col_md)) { out.has_nulls = true; }
    for (auto const& vseg : col_md.validity_segments) {
      if (is_constant_or_empty_validity(vseg.compression) && !vseg.all_null) { continue; }
      staged_segment vs;
      vs.row_offset = row_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count  = static_cast<uint32_t>(vseg.segment_count);
      // GPU validity dispatcher only knows UNCOMPRESSED, so report whatever
      // we ship as UNCOMPRESSED — even when source was ROARING.
      vs.compression = duckdb::CompressionType::COMPRESSION_UNCOMPRESSED;

      if (vseg.all_null) {
        stage_host_copy(s, make_all_null_validity_bytes(vseg.segment_count), vs);
      } else if (vseg.host_ptr != nullptr) {
        stage_host_copy(s, {{}, vseg.host_ptr, vseg.bytes_size}, vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        // ROARING stays on BufferManager: CreatePersistentSegment drives
        // reads internally and we don't have a host_read shape for it yet.
        stage_host_copy(s, decode_roaring_validity(db, block_manager, vseg), vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        stage_device_read(s, sf_bm, vseg, vs);
      } else {
        throw_unsupported("validity codec " + std::to_string(static_cast<int>(vseg.compression)) +
                          " (column " + std::to_string(col_md.column_id) + ")");
      }
      out.validity.push_back(vs);
    }
    row_cursor += static_cast<uint32_t>(rg.row_count);
  }

  out.total_rows = row_cursor;
  return out;
}

/// @brief Stage the data and validity segments for a varchar column, returning the staged
/// segments and metadata for the scan kernel. Throws if an unsupported codec is encountered.
staged_column stage_one_varchar_column(staging_state& s,
                                       duckdb::DatabaseInstance& db,
                                       duckdb::BlockManager& block_manager,
                                       duckdb::SingleFileBlockManager const& sf_bm,
                                       std::vector<duckdb_row_group_metadata> const& row_groups,
                                       std::size_t projected_col_idx)
{
  staged_column out;
  out.is_varchar = true;

  uint32_t row_cursor = 0;
  for (const auto& rg : row_groups) {
    auto const& col_md = rg.columns.at(projected_col_idx);

    //===----------Data Segments----------===//
    for (auto const& seg : col_md.data_segments) {
      if (!is_supported_varchar_codec(seg.compression)) {
        throw_unsupported("varchar data codec " +
                          std::to_string(static_cast<int>(seg.compression)) + " (column " +
                          std::to_string(col_md.column_id) + ")");
      }
      if (seg.host_ptr == nullptr && seg.block_id < 0) {
        throw_unsupported("varchar CONSTANT segment (column " + std::to_string(col_md.column_id) +
                          ")");
      }
      staged_segment ss;
      ss.row_offset        = row_cursor + static_cast<uint32_t>(seg.segment_start);
      ss.row_count         = static_cast<uint32_t>(seg.segment_count);
      ss.compression       = seg.compression;
      ss.max_string_length = *seg.max_string_length;  // walker/capture invariant

      if (seg.host_ptr != nullptr) {
        stage_host_copy(s, {{}, seg.host_ptr, seg.bytes_size}, ss);
      } else {
        stage_device_read(s, sf_bm, seg, ss);
      }
      out.data.push_back(ss);
    }

    //===----------Validity Segments----------===//
    if (column_has_real_nulls(col_md)) { out.has_nulls = true; }
    for (auto const& vseg : col_md.validity_segments) {
      if (is_constant_or_empty_validity(vseg.compression) && !vseg.all_null) { continue; }
      staged_segment vs;
      vs.row_offset  = row_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count   = static_cast<uint32_t>(vseg.segment_count);
      vs.compression = duckdb::CompressionType::COMPRESSION_UNCOMPRESSED;

      if (vseg.all_null) {
        stage_host_copy(s, make_all_null_validity_bytes(vseg.segment_count), vs);
      } else if (vseg.host_ptr != nullptr) {
        stage_host_copy(s, {{}, vseg.host_ptr, vseg.bytes_size}, vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        stage_host_copy(s, decode_roaring_validity(db, block_manager, vseg), vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        stage_device_read(s, sf_bm, vseg, vs);
      } else {
        throw_unsupported("validity codec " + std::to_string(static_cast<int>(vseg.compression)) +
                          " (varchar column " + std::to_string(col_md.column_id) + ")");
      }
      out.validity.push_back(vs);
    }
    row_cursor += static_cast<uint32_t>(rg.row_count);
  }

  out.total_rows = row_cursor;
  return out;
}

/// @brief Stage the segments for an ARRAY column with a fixed-width child element.
///
/// DuckDB stores ARRAY as: array-level validity (path [col,0]) + child data (path [col,1])
/// + optional child validity (path [col,1,0]). The child segments are in element-units and
/// DuckDB already emits child segment_start/segment_count in element units, so they drop
/// straight into staged_segment.
staged_column stage_one_array_column(staging_state& s,
                                     duckdb::DatabaseInstance& db,
                                     duckdb::BlockManager& block_manager,
                                     duckdb::SingleFileBlockManager const& sf_bm,
                                     std::vector<duckdb_row_group_metadata> const& row_groups,
                                     std::size_t projected_col_idx,
                                     sirius::logical_type const& projected_type)
{
  staged_column out;
  out.is_array = true;

  auto const& child_type = projected_type.array_child();
  auto const array_size  = static_cast<std::size_t>(projected_type.array_size());

  uint32_t row_cursor        = 0;
  uint32_t child_elem_cursor = 0;

  for (const auto& rg : row_groups) {
    auto const& col_md = rg.columns.at(projected_col_idx);

    //===----------Array-Level Validity (path [col, 0])----------===//
    // Walker routes array-level validity to col_md.data_segments for ARRAY
    // columns, so validity_segments (what column_has_real_nulls inspects) is
    // empty. Detect real nulls from the validity codec here.
    for (auto const& vseg : col_md.data_segments) {
      if (is_constant_or_empty_validity(vseg.compression) && !vseg.all_null) { continue; }
      out.has_nulls = true;
      staged_segment vs;
      vs.row_offset  = row_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count   = static_cast<uint32_t>(vseg.segment_count);
      vs.compression = duckdb::CompressionType::COMPRESSION_UNCOMPRESSED;

      if (vseg.all_null) {
        stage_host_copy(s, make_all_null_validity_bytes(vseg.segment_count), vs);
      } else if (vseg.host_ptr != nullptr) {
        // Host-backed (transient) array-level validity staged by the insert delta.
        stage_host_copy(s, {{}, vseg.host_ptr, vseg.bytes_size}, vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        stage_host_copy(s, decode_roaring_validity(db, block_manager, vseg), vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        stage_device_read(s, sf_bm, vseg, vs);
      } else {
        throw_unsupported("array validity codec " +
                          std::to_string(static_cast<int>(vseg.compression)) + " (column " +
                          std::to_string(col_md.column_id) + ")");
      }
      out.validity.push_back(vs);
    }

    //===----------Child Data (path [col, 1])----------===//
    for (auto const& seg : col_md.array_child_data_segments) {
      if (!is_supported_fixed_width_codec(seg.compression)) {
        throw_unsupported("array child data codec " +
                          std::to_string(static_cast<int>(seg.compression)) + " (column " +
                          std::to_string(col_md.column_id) + ")");
      }
      staged_segment ss;
      ss.row_offset  = child_elem_cursor + static_cast<uint32_t>(seg.segment_start);
      ss.row_count   = static_cast<uint32_t>(seg.segment_count);
      ss.compression = seg.compression;

      if (seg.host_ptr != nullptr) {
        // Host-backed (transient) child element bytes staged by the insert
        // delta; read from host memory rather than the .db file.
        stage_host_copy(s, {{}, seg.host_ptr, seg.bytes_size}, ss);
      } else if (seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        // The child segment's own stats are child-typed numeric stats, so
        // they extract directly — no ArrayStats unwrap.
        auto const& child_stats = constant_segment_stats(seg, col_md.column_id);
        stage_host_copy(s, extract_constant_bytes(child_stats, child_type), ss);
      } else {
        stage_device_read(s, sf_bm, seg, ss);
      }
      out.array_child_data.push_back(ss);
    }

    //===----------Child Validity (path [col, 1, 0])----------===//
    for (auto const& vseg : col_md.array_child_validity_segments) {
      if (is_constant_or_empty_validity(vseg.compression) && !vseg.all_null) { continue; }
      out.child_has_nulls = true;
      staged_segment vs;
      vs.row_offset  = child_elem_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count   = static_cast<uint32_t>(vseg.segment_count);
      vs.compression = duckdb::CompressionType::COMPRESSION_UNCOMPRESSED;

      if (vseg.all_null) {
        stage_host_copy(s, make_all_null_validity_bytes(vseg.segment_count), vs);
      } else if (vseg.host_ptr != nullptr) {
        // Host-backed (transient) child validity staged by the insert delta.
        stage_host_copy(s, {{}, vseg.host_ptr, vseg.bytes_size}, vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        stage_host_copy(s, decode_roaring_validity(db, block_manager, vseg), vs);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        stage_device_read(s, sf_bm, vseg, vs);
      } else {
        throw_unsupported("array child validity codec " +
                          std::to_string(static_cast<int>(vseg.compression)) + " (column " +
                          std::to_string(col_md.column_id) + ")");
      }
      out.array_child_validity.push_back(vs);
    }

    row_cursor += static_cast<uint32_t>(rg.row_count);
    auto const advanced = checked_array_child_advance(child_elem_cursor, rg.row_count, array_size);
    if (!advanced) {
      throw_unsupported("ARRAY column child-element count exceeds cudf size_type limit (column " +
                        std::to_string(col_md.column_id) + ")");
    }
    child_elem_cursor = *advanced;
  }

  out.total_rows       = row_cursor;
  out.total_child_rows = child_elem_cursor;
  return out;
}

//===----------------------------------------------------------------------===//
// Issue staged reads into a pinned host buffer, then copy them to the device.
//
// File-near segment reads are coalesced into large sequential reads (bridging the
// per-block header gaps) and dispatched as one batch via host_read_ranges_async_io,
// packed into a pinned multiple_blocks_allocation; the 16B device alignment the decode kernels need
// is imposed by the per-segment H2D scatter instead.
//===----------------------------------------------------------------------===//

using multiple_blocks_allocation =
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

/// @brief Batched pinned->device H2D (one cudaMemcpyBatchAsync launch); per-entry
/// fallback on toolkits without the batch API.
void batched_h2d(std::vector<void*> const& dst,
                 std::vector<void const*> const& src,
                 std::vector<std::size_t> const& size,
                 rmm::cuda_stream_view stream)
{
  if (dst.empty()) { return; }
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder  = cudaMemcpySrcAccessOrderStream;
  attrs.srcLocHint.type = cudaMemLocationTypeHost;
  attrs.dstLocHint.type = cudaMemLocationTypeDevice;
  attrs.flags           = 0;
  std::size_t attrs_idx = 0;  // single attrs entry applies to all copies
#if CUDART_VERSION < 13000
  std::size_t fail_idx = 0;
  // The CUDA 12.x batch API takes non-const pointers (it was made const-correct in 13.0).
  // These arrays are read-only inputs to the copy, so casting away const is safe here.
  RMM_CUDA_TRY(cudaMemcpyBatchAsync(const_cast<void**>(dst.data()),
                                    const_cast<void**>(src.data()),
                                    const_cast<std::size_t*>(size.data()),
                                    dst.size(),
                                    &attrs,
                                    &attrs_idx,
                                    1,
                                    &fail_idx,
                                    stream.value()));
#else
  RMM_CUDA_TRY(cudaMemcpyBatchAsync(
    dst.data(), src.data(), size.data(), dst.size(), &attrs, &attrs_idx, 1, stream.value()));
#endif
#else
  for (std::size_t i = 0; i < dst.size(); ++i) {
    RMM_CUDA_TRY(cudaMemcpyAsync(dst[i], src[i], size[i], cudaMemcpyHostToDevice, stream.value()));
  }
#endif
}

void submit_and_await(rmm::device_buffer& device_buf,
                      staging_state const& s,
                      const sirius::io::sirius_datasource& datasource,
                      cucascade::memory::memory_reservation_manager& host_mem_mgr,
                      int host_numa_node,
                      std::size_t coalesce_max_gap,
                      rmm::cuda_stream_view stream)
{
  namespace ccm = cucascade::memory;

  auto* device_base = static_cast<uint8_t*>(device_buf.data());

  // The FSMR block size is uniform across the per-NUMA host spaces (one host config), so
  // probe any host space for it up front: we need it to lay out the pieces below, and we
  // want to reserve exactly the host bytes we end up allocating (not the device-buffer size).
  auto const host_spaces = host_mem_mgr.get_memory_spaces_for_tier(ccm::Tier::HOST);
  if (host_spaces.empty()) {
    throw std::runtime_error(std::string(kTag) + " no HOST-tier memory space registered");
  }
  auto const* probe_fsmr =
    host_spaces.front()->get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
  if (probe_fsmr == nullptr) {
    throw std::runtime_error(std::string(kTag) +
                             " host memory space is not a fixed_size_host_memory_resource");
  }
  auto const bsz = probe_fsmr->get_block_size();

  // Coalesce file-near whole segments into contiguous host pieces
  auto reads = s.reads;
  std::sort(reads.begin(), reads.end(), [](auto const& a, auto const& b) {
    return a.file_offset < b.file_offset;
  });

  // Map a coalesced range of disk-resident data to its corresponding pinned host block + offset
  struct piece {
    std::size_t file_off, file_end, host_block, host_off;
  };
  // Map a pinned host block + offset to its corresponding device buffer destination
  struct seg_copy {
    std::size_t host_block, host_off, device_off, size;
  };

  std::vector<piece> pieces;
  std::vector<seg_copy> seg_copies;
  pieces.reserve(reads.size());
  seg_copies.reserve(reads.size());
  std::size_t cur_block = 0, cur_off = 0;
  for (auto const& r : reads) {
    if (r.size == 0) { continue; }
    bool new_piece = pieces.empty();
    if (!new_piece) {
      auto& p                       = pieces.back();
      std::size_t const gap         = r.file_offset - p.file_end;
      std::size_t const prospective = (r.file_offset + r.size) - p.file_off;
      if (gap > coalesce_max_gap || p.host_off + prospective > bsz) { new_piece = true; }
    }
    if (new_piece) {
      if (cur_off + r.size > bsz) {
        ++cur_block;
        cur_off = 0;
      }
      pieces.push_back({r.file_offset, r.file_offset + r.size, cur_block, cur_off});
    } else {
      pieces.back().file_end = r.file_offset + r.size;
    }
    auto& p = pieces.back();
    seg_copies.push_back(
      {p.host_block, p.host_off + (r.file_offset - p.file_off), r.device_offset, r.size});
    cur_off = p.host_off + (p.file_end - p.file_off);
  }

  // Reserve and allocate exactly the host bytes the pieces occupy (cur_block + 1 blocks),
  // not the device-buffer size, preferring the GPU's local NUMA node. The strategy may fall
  // back to another host space; guard that it shares the block size the pieces were laid out
  // against, otherwise the per-block offsets would be wrong.
  std::size_t const host_bytes = (cur_block + 1) * bsz;
  ccm::any_memory_space_in_tier_with_preference host_req(ccm::Tier::HOST,
                                                         static_cast<std::size_t>(host_numa_node));
  auto reservation = host_mem_mgr.request_reservation(host_req, host_bytes);
  if (!reservation) {
    throw std::runtime_error(std::string(kTag) + " failed to reserve " +
                             std::to_string(host_bytes) + " bytes of host staging memory");
  }
  auto* host_fsmr =
    reservation->get_memory_space().get_memory_resource_as<ccm::fixed_size_host_memory_resource>();
  if (host_fsmr == nullptr || host_fsmr->get_block_size() != bsz) {
    throw std::runtime_error(std::string(kTag) +
                             " host memory space is not a fixed_size_host_memory_resource with the "
                             "expected block size");
  }
  auto host_alloc = host_fsmr->allocate_multiple_blocks(host_bytes, reservation.get());

  // One coalesced range + contiguous dst span per piece.
  std::vector<io::io_object_segment> ranges;
  ranges.reserve(pieces.size());
  std::size_t total_read = 0;
  for (auto const& p : pieces) {
    std::size_t const sz = p.file_end - p.file_off;
    ranges.emplace_back(
      p.file_off,
      sz,
      reinterpret_cast<std::uint8_t*>(host_alloc->at(p.host_block).data()) + p.host_off);
    total_read += sz;
  }

  // Issue the coalesced reads as one batch and await completion.
  {
    nvtx3::scoped_range nvtx_reads{"native_reads"};
    auto io_ctx           = datasource.io_ctx();
    auto fut              = io_ctx->host_read_ranges_async_io(datasource.io_object(), ranges);
    std::size_t const got = std::move(fut).get();
    if (got != total_read) {
      throw std::runtime_error(std::string(kTag) + " short coalesced host read: got " +
                               std::to_string(got) + " expected " + std::to_string(total_read));
    }
  }

  // CPU-produced segments (CONSTANT/ROARING): copy straight to their device slots; no
  // overwrite hazard since each segment owns a disjoint device range.
  for (auto const& h : s.host_copies) {
    RMM_CUDA_TRY(cudaMemcpyAsync(
      device_base + h.device_offset, h.src_ptr, h.size, cudaMemcpyHostToDevice, stream.value()));
  }

  // Per-segment H2D: host (packed) -> device (16B-aligned), batched. Sync before
  // host_alloc / reservation drop so the copies finish reading pinned memory first.
  {
    nvtx3::scoped_range nvtx_h2d{"native_h2d"};
    std::vector<void*> h2d_dst;
    std::vector<void const*> h2d_src;
    std::vector<std::size_t> h2d_size;
    h2d_dst.reserve(seg_copies.size());
    h2d_src.reserve(seg_copies.size());
    h2d_size.reserve(seg_copies.size());
    for (auto const& c : seg_copies) {
      h2d_dst.push_back(device_base + c.device_off);
      h2d_src.push_back(reinterpret_cast<uint8_t*>(host_alloc->at(c.host_block).data()) +
                        c.host_off);
      h2d_size.push_back(c.size);
    }
    batched_h2d(h2d_dst, h2d_src, h2d_size, stream);
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  }
}

/// H2D for a split with no file reads: no io, no pinned staging blocks, no
/// SiriusContext. Synchronizes before returning so the caller may drop the
/// source buffers' owners, same as submit_and_await.
void submit_host_only_and_await(rmm::device_buffer& device_buf,
                                staging_state const& s,
                                rmm::cuda_stream_view stream)
{
  auto* device_base = static_cast<uint8_t*>(device_buf.data());
  for (auto const& h : s.host_copies) {
    RMM_CUDA_TRY(cudaMemcpyAsync(
      device_base + h.device_offset, h.src_ptr, h.size, cudaMemcpyHostToDevice, stream.value()));
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
}

//===----------------------------------------------------------------------===//
// Build codec runs from staged segments.
//===----------------------------------------------------------------------===//

void fill_fixed_width_runs(std::vector<staged_segment> const& staged,
                           rmm::device_buffer const& device_buf,
                           std::vector<gpu_codec_run>& out_runs)
{
  out_runs.clear();
  duckdb::CompressionType current = duckdb::CompressionType::COMPRESSION_AUTO;
  auto* device_base               = static_cast<uint8_t const*>(device_buf.data());
  for (auto const& s : staged) {
    if (out_runs.empty() || s.compression != current) {
      out_runs.push_back({s.compression, {}});
      current = s.compression;
    }
    gpu_segment_desc seg{};
    seg.d_bytes    = device_base + s.device_offset;
    seg.bytes_size = static_cast<uint32_t>(std::min<std::size_t>(s.bytes, UINT32_MAX));
    seg.row_offset = s.row_offset;
    seg.row_count  = s.row_count;
    out_runs.back().segments.push_back(seg);
  }
}

void fill_string_runs(std::vector<staged_segment> const& staged,
                      rmm::device_buffer const& device_buf,
                      std::vector<gpu_string_codec_run>& out_runs)
{
  out_runs.clear();
  duckdb::CompressionType current = duckdb::CompressionType::COMPRESSION_AUTO;
  auto* device_base               = static_cast<uint8_t const*>(device_buf.data());
  for (auto const& s : staged) {
    if (out_runs.empty() || s.compression != current) {
      out_runs.push_back({s.compression, {}});
      current = s.compression;
    }
    gpu_string_segment_desc seg{};
    seg.d_bytes           = device_base + s.device_offset;
    seg.bytes_size        = static_cast<uint32_t>(std::min<std::size_t>(s.bytes, UINT32_MAX));
    seg.row_offset        = s.row_offset;
    seg.row_count         = s.row_count;
    seg.seg_row_start     = 0;
    seg.max_string_length = s.max_string_length;
    out_runs.back().segments.push_back(seg);
  }
}

//===----------------------------------------------------------------------===//
// Rowid synthesis via cudf::sequence + cudf::concatenate.
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> build_rowid_column(
  std::vector<duckdb_row_group_metadata> const& row_groups,
  cudf::size_type total_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> per_rg;
  per_rg.reserve(row_groups.size());
  for (auto const& rg : row_groups) {
    if (rg.row_count == 0) continue;
    auto init = cudf::numeric_scalar<std::int64_t>(
      static_cast<std::int64_t>(rg.row_group_start), true, stream, mr);
    per_rg.push_back(cudf::sequence(static_cast<cudf::size_type>(rg.row_count), init, stream, mr));
  }
  if (per_rg.empty()) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT64}, total_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  }
  if (per_rg.size() == 1) { return std::move(per_rg[0]); }
  std::vector<cudf::column_view> views;
  views.reserve(per_rg.size());
  for (auto const& c : per_rg) {
    views.push_back(c->view());
  }
  return cudf::concatenate(views, stream, mr);
}

// Zero-row column carrying the projected schema of column @p type / @p pcol. Mirrors the
// non-empty decode outputs: rowid synthesizes INT64 (build_rowid_column), ARRAY becomes a cuDF
// LIST of its fixed-width element (make_empty_column rejects nested types), everything else maps
// through sirius_to_cudf_type. A rowid slot's declared type is arbitrary and never read.
std::unique_ptr<cudf::column> empty_column_for(projected_column const& pcol,
                                               sirius::logical_type const& type)
{
  if (pcol.is_rowid) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}); }
  return type.is_array() ? cudf::make_empty_lists_column(sirius_to_cudf_type(type.array_child()))
                         : cudf::make_empty_column(sirius_to_cudf_type(type));
}

}  // namespace

std::vector<cudf::io::text::byte_range_info> row_group_file_ranges(
  duckdb::SingleFileBlockManager const& block_manager, duckdb_row_group_metadata const& row_group)
{
  std::vector<cudf::io::text::byte_range_info> ranges;
  for (auto const& col : row_group.columns) {
    for (auto const& seg : col.data_segments) {
      append_segment_file_ranges(block_manager, seg, ranges);
    }
    for (auto const& seg : col.validity_segments) {
      append_segment_file_ranges(block_manager, seg, ranges);
    }
  }
  return ranges;
}

//===----------------------------------------------------------------------===//
// Public entry: decode_duckdb_native_split
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> decode_duckdb_native_split(
  std::vector<duckdb_row_group_metadata> const& row_groups,
  duckdb_native_ingestible_table_info const& table_info,
  sirius::io::sirius_datasource* datasource,
  cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  if (row_groups.empty()) {
    // Preserve the projected schema so an empty or fully pruned split follows
    // the normal filter/project/concat path.
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    empty_cols.reserve(table_info.projected_cols.size());
    for (std::size_t ci = 0; ci < table_info.projected_cols.size(); ++ci) {
      empty_cols.push_back(
        empty_column_for(table_info.projected_cols[ci], table_info.projected_types[ci]));
    }
    return std::make_unique<cudf::table>(std::move(empty_cols));
  }
  auto const& scan_info = table_info;
  auto& storage         = *scan_info.storage;
  auto& context         = *scan_info.context;

  auto& db            = duckdb::DatabaseInstance::GetDatabase(context);
  auto& sm            = storage.GetAttached().GetStorageManager();
  auto& block_manager = sm.GetBlockManager();

  // sirius_io routing
  auto const* sf_bm = dynamic_cast<duckdb::SingleFileBlockManager const*>(&block_manager);
  if (!sf_bm) {
    throw std::runtime_error(
      std::string(kTag) +
      " missing io_ctx, io_obj, or SingleFileBlockManager for duckdb_native_scan");
  }

  auto mr_ref = mem_space.get_default_allocator();

  auto const num_cols = scan_info.projected_cols.size();

  std::size_t total_rows = 0;
  for (auto const& rg : row_groups) {
    total_rows += rg.row_count;
  }
  if (total_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::runtime_error(std::string(kTag) + " split rows (" + std::to_string(total_rows) +
                             ") exceed cudf::size_type max");
  }

  staging_state staging;
  std::vector<staged_column> staged_cols;
  staged_cols.reserve(num_cols);
  std::vector<bool> is_rowid_col(num_cols, false);

  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    auto const& pcol = scan_info.projected_cols[ci];
    if (pcol.is_rowid) {
      is_rowid_col[ci] = true;
      staged_cols.emplace_back();
      staged_cols.back().total_rows = total_rows;
      continue;
    }
    if (scan_info.projected_types[ci].is_varchar()) {
      staged_cols.push_back(
        stage_one_varchar_column(staging, db, block_manager, *sf_bm, row_groups, ci));
    } else if (scan_info.projected_types[ci].is_array()) {
      staged_cols.push_back(stage_one_array_column(
        staging, db, block_manager, *sf_bm, row_groups, ci, scan_info.projected_types[ci]));
    } else {
      staged_cols.push_back(stage_one_fixed_width_column(
        staging, db, block_manager, *sf_bm, row_groups, ci, scan_info.projected_types[ci]));
    }
  }

  rmm::device_buffer device_buf(staging.running_offset, stream, mr_ref);
  if (!staging.reads.empty()) {
    if (datasource == nullptr) {
      throw std::runtime_error(std::string(kTag) +
                               " split staged file reads but carries no datasource");
    }
    auto sirius_st = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_st) {
      throw std::runtime_error(std::string(kTag) + " no sirius_state on the ClientContext");
    }
    // NUMA node of the GPU that owns device_buf (mem_space). Host staging is then
    // reserved on that node. topo.gpus is indexed by device id; normalize an
    // unknown/negative node to 0, matching SiriusContext's host-space convention.
    auto const gpu_dev  = mem_space.get_device_id();
    auto const& topo    = sirius_st->get_hw_topology();
    int const raw_numa  = (gpu_dev >= 0 && static_cast<std::size_t>(gpu_dev) < topo.gpus.size())
                            ? topo.gpus[gpu_dev].numa_node
                            : -1;
    int const host_numa = (raw_numa < 0) ? 0 : raw_numa;
    // Coalesce reads across at most the inter-block-payload gap (the block header):
    // consecutive full .db blocks are exactly GetBlockHeaderSize() apart, so this merges
    // them into one sequential read without pulling the larger unprojected-column waste
    // that sits between row groups.
    std::size_t const coalesce_max_gap = sf_bm->GetBlockHeaderSize();
    submit_and_await(device_buf,
                     staging,
                     *datasource,
                     sirius_st->get_memory_manager(),
                     host_numa,
                     coalesce_max_gap,
                     stream);
  } else if (!staging.host_copies.empty()) {
    // All-host split: no file io, no staging blocks, no SiriusContext needed.
    submit_host_only_and_await(device_buf, staging, stream);
  }

  // Group fixed-width columns for a single gpu_decode_table call; varchar
  // columns each go through gpu_decode_strings_column separately; array
  // columns decode child data as fixed-width, then wrap into cudf LIST
  // with offsets child.
  std::vector<gpu_column_decode_input> fw_inputs;
  std::vector<std::size_t> fw_to_final_idx;
  std::vector<gpu_string_column_decode_input> vc_inputs;
  std::vector<std::size_t> vc_to_final_idx;
  std::vector<gpu_column_decode_input> array_child_inputs;
  std::vector<std::size_t> array_to_final_idx;
  fw_inputs.reserve(num_cols);
  fw_to_final_idx.reserve(num_cols);
  array_child_inputs.reserve(num_cols);
  array_to_final_idx.reserve(num_cols);

  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    if (is_rowid_col[ci]) continue;
    auto const& staged = staged_cols[ci];
    if (staged.is_varchar) {
      gpu_string_column_decode_input input;
      input.total_rows = static_cast<uint32_t>(staged.total_rows);
      input.has_nulls  = staged.has_nulls;
      fill_string_runs(staged.data, device_buf, input.data);
      fill_fixed_width_runs(staged.validity, device_buf, input.validity);
      vc_inputs.push_back(std::move(input));
      vc_to_final_idx.push_back(ci);
    } else if (staged.is_array) {
      // Decode the child data as a fixed-width column
      gpu_column_decode_input child_input;
      child_input.out_type   = sirius_to_cudf_type(scan_info.projected_types[ci].array_child());
      child_input.total_rows = static_cast<uint32_t>(staged.total_child_rows);
      child_input.has_nulls  = staged.child_has_nulls;
      fill_fixed_width_runs(staged.array_child_data, device_buf, child_input.data);
      fill_fixed_width_runs(staged.array_child_validity, device_buf, child_input.validity);
      array_child_inputs.push_back(std::move(child_input));
      array_to_final_idx.push_back(ci);
    } else {
      gpu_column_decode_input input;
      input.out_type   = sirius_to_cudf_type(scan_info.projected_types[ci]);
      input.total_rows = static_cast<uint32_t>(staged.total_rows);
      input.has_nulls  = staged.has_nulls;
      fill_fixed_width_runs(staged.data, device_buf, input.data);
      fill_fixed_width_runs(staged.validity, device_buf, input.validity);
      fw_inputs.push_back(std::move(input));
      fw_to_final_idx.push_back(ci);
    }
  }

  std::vector<std::unique_ptr<cudf::column>> fw_cols;
  if (!fw_inputs.empty()) {
    auto fw_table = ::sirius::cuda::scan::gpu_decode_table(fw_inputs, stream, mr_ref);
    fw_cols       = fw_table->release();
  }

  std::vector<std::unique_ptr<cudf::column>> vc_cols;
  vc_cols.reserve(vc_inputs.size());
  for (auto const& vc : vc_inputs) {
    vc_cols.push_back(::sirius::cuda::scan::gpu_decode_strings_column(vc, stream, mr_ref));
  }

  // Decode ARRAY child data as fixed-width columns, then wrap into LIST with offsets
  std::vector<std::unique_ptr<cudf::column>> array_cols;
  array_cols.reserve(array_child_inputs.size());
  if (!array_child_inputs.empty()) {
    // Decode each child column on its own as different columns might have different array sizes
    std::vector<std::unique_ptr<cudf::column>> child_cols;
    child_cols.reserve(array_child_inputs.size());
    for (auto const& child_input : array_child_inputs) {
      auto child_table = ::sirius::cuda::scan::gpu_decode_table({child_input}, stream, mr_ref);
      auto cols        = child_table->release();
      child_cols.push_back(std::move(cols[0]));
    }

    for (std::size_t ai = 0; ai < array_to_final_idx.size(); ++ai) {
      auto const ci         = array_to_final_idx[ai];
      auto const& staged    = staged_cols[ci];
      auto const array_size = scan_info.projected_types[ci].array_size();
      auto const total_rows = static_cast<cudf::size_type>(staged.total_rows);

      // Filling stride offsets
      auto init_scalar = cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr_ref);
      auto step_scalar = cudf::numeric_scalar<cudf::size_type>(array_size, true, stream, mr_ref);
      auto offsets     = cudf::sequence(total_rows + 1, init_scalar, step_scalar, stream, mr_ref);

      // Decode array-level validity from staged.validity segments
      rmm::device_buffer parent_null_mask(0, stream, mr_ref);
      cudf::size_type parent_null_count = 0;
      if (staged.has_nulls && !staged.validity.empty()) {
        // Temporary decode input for the array validity mask
        gpu_column_decode_input validity_input;
        validity_input.out_type   = cudf::data_type{cudf::type_id::BOOL8};  // dummy type
        validity_input.total_rows = total_rows;
        validity_input.has_nulls  = true;
        fill_fixed_width_runs(staged.validity, device_buf, validity_input.validity);
        // Decode a dummy BOOL8 column to get the null mask
        // TODO: this wastes a throwaway BOOL8 column just to grab the null mask. If
        // decode_column_validity() were exposed in gpu_native_decode.cuh, we could
        // decode the array-level mask directly.
        auto validity_table =
          ::sirius::cuda::scan::gpu_decode_table({validity_input}, stream, mr_ref);
        auto validity_cols = validity_table->release();
        parent_null_count  = validity_cols[0]->null_count();
        auto released      = validity_cols[0]->release();
        parent_null_mask   = std::move(*released.null_mask);
      }

      // Assemble the LIST column: offsets + child values + array-level null mask
      auto list_col = cudf::make_lists_column(total_rows,
                                              std::move(offsets),
                                              std::move(child_cols[ai]),
                                              parent_null_count,
                                              std::move(parent_null_mask));
      array_cols.push_back(std::move(list_col));
    }
  }

  std::vector<std::unique_ptr<cudf::column>> final_cols(num_cols);
  for (std::size_t fi = 0; fi < fw_cols.size(); ++fi) {
    final_cols[fw_to_final_idx[fi]] = std::move(fw_cols[fi]);
  }
  for (std::size_t vi = 0; vi < vc_cols.size(); ++vi) {
    final_cols[vc_to_final_idx[vi]] = std::move(vc_cols[vi]);
  }
  for (std::size_t ai = 0; ai < array_cols.size(); ++ai) {
    final_cols[array_to_final_idx[ai]] = std::move(array_cols[ai]);
  }
  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    if (!is_rowid_col[ci]) continue;
    final_cols[ci] =
      build_rowid_column(row_groups, static_cast<cudf::size_type>(total_rows), stream, mr_ref);
  }

  return std::make_unique<cudf::table>(std::move(final_cols));
}

}  // namespace sirius::op::scan
