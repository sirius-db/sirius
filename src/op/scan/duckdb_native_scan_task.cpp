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

#include "op/scan/duckdb_native_scan_task.hpp"

#include "cuda/scan/gpu_decode_strings.cuh"
#include "cuda/scan/gpu_native_decode.cuh"
#include "cudf/cudf_utils.hpp"
#include "helper/type_conversions.hpp"
#include "io/io_context.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_block_layout.hpp"
#include "op/scan/duckdb_native_scan_info.hpp"
#include "sirius_context.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/function/partition_stats.hpp>
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
#include <cstring>
#include <numeric>
#include <optional>
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
    case duckdb::CompressionType::COMPRESSION_BITPACKING: return true;
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
  duckdb::vector<sirius::logical_type> wrap{t};
  auto duckdb_vec = sirius::to_duckdb_vec(wrap);
  return duckdb::GetCudfType(duckdb_vec[0]);
}

duckdb::LogicalType sirius_to_duckdb_type(sirius::logical_type const& t)
{
  duckdb::vector<sirius::logical_type> wrap{t};
  auto duckdb_vec = sirius::to_duckdb_vec(wrap);
  return duckdb_vec[0];
}

//===----------------------------------------------------------------------===//
// Host bytes for a segment.
//
// Three sources: a pinned BufferHandle (normal block), owned host bytes
// (CONSTANT extracted from stats, ROARING host-decoded, or main+additional
// blocks concatenated), or both (BufferHandle pinning the main block plus
// owned bytes from the additional-block concat that follows). We keep the
// handle alive until H2D is queued.
//===----------------------------------------------------------------------===//

struct pinned_segment_bytes {
  std::vector<duckdb::BufferHandle> handles;  // empty when source is owned_bytes only
  std::vector<uint8_t> owned_bytes;           // used for CONSTANT, ROARING, concat
  uint8_t const* host_ptr = nullptr;
  std::size_t bytes       = 0;
};

pinned_segment_bytes pin_block(duckdb::BlockManager& block_manager,
                               duckdb::BufferManager& buffer_manager,
                               duckdb_segment_descriptor const& seg)
{
  if (seg.block_id < 0) {
    throw std::runtime_error(std::string(kTag) +
                             " pin_block called with block_id<0 (CONSTANT segment?)");
  }
  auto handle = block_manager.RegisterBlock(seg.block_id);
  auto pinned = buffer_manager.Pin(handle);
  pinned_segment_bytes out;
  out.host_ptr = pinned.Ptr() + seg.block_offset;
  out.bytes    = seg.bytes_size;
  out.handles.push_back(std::move(pinned));
  return out;
}

// Pin main block + each additional block; concatenate into one owned buffer.
// The descriptor's block_offset applies only to the main block; additional
// blocks are taken whole-block. The resulting buffer has main-block bytes
// (from block_offset to end) followed by each additional block's full bytes.
//
// Whether the on-disk codec actually arranges its dictionary/heap to be
// readable as one contiguous slab is codec-dependent. For FSST/DICT_FSST
// inline-symbol-table segments the main block alone is enough; the concat
// is here for codecs that visit_block_ids.
pinned_segment_bytes pin_block_with_additional(duckdb::BlockManager& block_manager,
                                               duckdb::BufferManager& buffer_manager,
                                               duckdb_segment_descriptor const& seg)
{
  auto main_pinned      = block_manager.RegisterBlock(seg.block_id);
  auto main_handle      = buffer_manager.Pin(main_pinned);
  auto const block_size = block_manager.GetBlockSize();

  std::vector<duckdb::BufferHandle> handles;
  handles.push_back(std::move(main_handle));
  std::vector<uint8_t> concat;
  concat.resize(seg.bytes_size);
  std::memcpy(concat.data(), handles.front().Ptr() + seg.block_offset, seg.bytes_size);

  for (auto add_id : seg.additional_blocks) {
    auto add_handle = block_manager.RegisterBlock(add_id);
    auto h          = buffer_manager.Pin(add_handle);
    auto offset     = concat.size();
    concat.resize(offset + block_size);
    std::memcpy(concat.data() + offset, h.Ptr(), block_size);
    handles.push_back(std::move(h));
  }

  pinned_segment_bytes out;
  out.owned_bytes = std::move(concat);
  out.handles     = std::move(handles);
  out.host_ptr    = out.owned_bytes.data();
  out.bytes       = out.owned_bytes.size();
  return out;
}

// sirius_io variants of pin_block / pin_block_with_additional: read .db block
// payloads via sirius_ioctx::host_read, bypassing DuckDB's BufferManager.
// Output shape matches the BufferManager variants. See read_block_payload for
// the dispatch between the two.
//===----------------------------------------------------------------------===//

pinned_segment_bytes read_block_via_io(::sirius::io::sirius_ioctx& ctx,
                                       ::sirius::io::sirius_io_object& obj,
                                       duckdb::SingleFileBlockManager const& bm,
                                       duckdb_segment_descriptor const& seg)
{
  pinned_segment_bytes out;
  if (seg.bytes_size == 0) { return out; }
  out.owned_bytes.resize(seg.bytes_size);
  const std::size_t got = ctx.host_read(
    obj,
    duckdb_block_payload_offset(bm, seg.block_id) + static_cast<std::size_t>(seg.block_offset),
    seg.bytes_size,
    out.owned_bytes.data());
  if (got != seg.bytes_size) {
    throw std::runtime_error(std::string(kTag) + " short host_read for block_id " +
                             std::to_string(seg.block_id) + ": got " + std::to_string(got) +
                             " expected " + std::to_string(seg.bytes_size));
  }
  out.host_ptr = out.owned_bytes.data();
  out.bytes    = seg.bytes_size;
  return out;
}

pinned_segment_bytes read_blocks_with_additional_via_io(::sirius::io::sirius_ioctx& ctx,
                                                        ::sirius::io::sirius_io_object& obj,
                                                        duckdb::SingleFileBlockManager const& bm,
                                                        duckdb_segment_descriptor const& seg)
{
  const std::size_t block_size        = bm.GetBlockSize();
  const std::size_t main_payload_size = seg.bytes_size;

  std::vector<uint8_t> concat;
  concat.resize(main_payload_size + seg.additional_blocks.size() * block_size);

  // Main block: read just bytes_size from (payload + block_offset). No temp + memcpy.
  if (main_payload_size > 0) {
    const std::size_t got = ctx.host_read(
      obj,
      duckdb_block_payload_offset(bm, seg.block_id) + static_cast<std::size_t>(seg.block_offset),
      main_payload_size,
      concat.data());
    if (got != main_payload_size) {
      throw std::runtime_error(std::string(kTag) + " short host_read for main block_id " +
                               std::to_string(seg.block_id) + ": got " + std::to_string(got) +
                               " expected " + std::to_string(main_payload_size));
    }
  }

  // Additional blocks: read each full-payload directly into the concat buffer.
  std::size_t dst_off = main_payload_size;
  for (auto add_id : seg.additional_blocks) {
    const std::size_t got = ctx.host_read(
      obj, duckdb_block_payload_offset(bm, add_id), block_size, concat.data() + dst_off);
    if (got != block_size) {
      throw std::runtime_error(std::string(kTag) + " short host_read for additional block_id " +
                               std::to_string(add_id));
    }
    dst_off += block_size;
  }

  pinned_segment_bytes out;
  out.owned_bytes = std::move(concat);
  out.host_ptr    = out.owned_bytes.data();
  out.bytes       = out.owned_bytes.size();
  return out;
}

// Picks the via-sirius_io path when ioctx + io_object + SingleFileBlockManager
// are all available, otherwise falls back to BufferManager::Pin. Output shape
// matches pin_block: host pointer + segment-exact bytes.
pinned_segment_bytes read_block_payload(::sirius::io::sirius_ioctx* io_ctx,
                                        ::sirius::io::sirius_io_object* io_obj,
                                        duckdb::SingleFileBlockManager const* sf_bm,
                                        duckdb::BlockManager& block_manager,
                                        duckdb::BufferManager& buffer_manager,
                                        duckdb_segment_descriptor const& seg)
{
  const bool via_io = (io_ctx != nullptr && io_obj != nullptr && sf_bm != nullptr);
  if (via_io) {
    return seg.additional_blocks.empty()
             ? read_block_via_io(*io_ctx, *io_obj, *sf_bm, seg)
             : read_blocks_with_additional_via_io(*io_ctx, *io_obj, *sf_bm, seg);
  }
  return seg.additional_blocks.empty()
           ? pin_block(block_manager, buffer_manager, seg)
           : pin_block_with_additional(block_manager, buffer_manager, seg);
}

//===----------------------------------------------------------------------===//
// CONSTANT extraction.
//
// CONSTANT segments have block_id == -1; the constant value lives in
// per-(rg, col) statistics. We pull stats from PartitionRowGroup at scan
// time and copy the value into an owned buffer the kernel can read.
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
  auto duckdb_type = sirius_to_duckdb_type(sirius_type);
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
  constexpr duckdb::idx_t CHUNK =
    static_cast<duckdb::idx_t>(duckdb::roaring::ROARING_CONTAINER_SIZE);
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
  bool has_nulls         = false;
  std::size_t total_rows = 0;
  bool is_varchar        = false;
};

struct staging_state {
  std::vector<pinned_segment_bytes> pinned;
  std::vector<uint8_t const*> src_ptrs;
  std::vector<std::size_t> src_sizes;
  std::vector<std::size_t> dst_offsets;
  std::size_t running_offset = 0;
};

void record_staged(staging_state& s, pinned_segment_bytes p, staged_segment& out)
{
  // 16B alignment: kernels cast d_bytes to typed pointers up to uint128.
  constexpr std::size_t SEGMENT_ALIGN = 16;
  s.running_offset  = (s.running_offset + SEGMENT_ALIGN - 1) & ~(SEGMENT_ALIGN - 1);
  out.bytes         = p.bytes;
  out.device_offset = s.running_offset;
  s.src_ptrs.push_back(p.host_ptr);
  s.src_sizes.push_back(p.bytes);
  s.dst_offsets.push_back(s.running_offset);
  s.running_offset += p.bytes;
  s.pinned.push_back(std::move(p));
}

duckdb::BaseStatistics const& constant_stats_for(
  std::vector<duckdb::PartitionStatistics> const& partition_stats,
  duckdb::idx_t rg_idx,
  duckdb::idx_t storage_idx,
  std::vector<std::unique_ptr<duckdb::BaseStatistics>>& owned_stats_cache)
{
  if (rg_idx >= partition_stats.size() || !partition_stats[rg_idx].partition_row_group) {
    throw std::runtime_error(std::string(kTag) +
                             " no PartitionRowGroup for CONSTANT lookup on rg " +
                             std::to_string(rg_idx));
  }
  auto stats = partition_stats[rg_idx].partition_row_group->GetColumnStatistics(
    duckdb::StorageIndex(storage_idx));
  if (!stats) {
    throw std::runtime_error(std::string(kTag) +
                             " PartitionRowGroup returned null stats for CONSTANT lookup");
  }
  owned_stats_cache.push_back(std::move(stats));
  return *owned_stats_cache.back();
}

staged_column stage_one_fixed_width_column(
  staging_state& s,
  duckdb::DatabaseInstance& db,
  duckdb::BlockManager& block_manager,
  duckdb::SingleFileBlockManager const* sf_bm,
  duckdb::BufferManager& buffer_manager,
  ::sirius::io::sirius_ioctx* io_ctx,
  ::sirius::io::sirius_io_object* io_obj,
  std::vector<duckdb::PartitionStatistics> const& partition_stats,
  std::vector<std::unique_ptr<duckdb::BaseStatistics>>& owned_stats_cache,
  std::vector<duckdb_row_group_metadata> const& row_groups,
  std::size_t projected_col_idx,
  sirius::logical_type const& projected_type)
{
  staged_column out;

  uint32_t row_cursor = 0;
  for (std::size_t rg_i = 0; rg_i < row_groups.size(); ++rg_i) {
    auto const& rg     = row_groups[rg_i];
    auto const& col_md = rg.columns.at(projected_col_idx);
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

      pinned_segment_bytes p;
      if (seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        auto const& stats = constant_stats_for(
          partition_stats, rg.row_group_index, col_md.column_id, owned_stats_cache);
        p = extract_constant_bytes(stats, projected_type);
      } else {
        p = read_block_payload(io_ctx, io_obj, sf_bm, block_manager, buffer_manager, seg);
      }
      record_staged(s, std::move(p), ss);
      out.data.push_back(ss);
    }

    if (column_has_real_nulls(col_md)) { out.has_nulls = true; }
    for (auto const& vseg : col_md.validity_segments) {
      if (is_constant_or_empty_validity(vseg.compression)) { continue; }
      staged_segment vs;
      vs.row_offset = row_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count  = static_cast<uint32_t>(vseg.segment_count);
      // GPU validity dispatcher only knows UNCOMPRESSED, so report whatever
      // we ship as UNCOMPRESSED — even when source was ROARING.
      vs.compression = duckdb::CompressionType::COMPRESSION_UNCOMPRESSED;

      pinned_segment_bytes p;
      if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        // ROARING stays on BufferManager: CreatePersistentSegment drives
        // reads internally and we don't have a host_read shape for it yet.
        p = decode_roaring_validity(db, block_manager, vseg);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        p = read_block_payload(io_ctx, io_obj, sf_bm, block_manager, buffer_manager, vseg);
      } else {
        throw_unsupported("validity codec " + std::to_string(static_cast<int>(vseg.compression)) +
                          " (column " + std::to_string(col_md.column_id) + ")");
      }
      record_staged(s, std::move(p), vs);
      out.validity.push_back(vs);
    }
    row_cursor += static_cast<uint32_t>(rg.row_count);
  }

  out.total_rows = row_cursor;
  return out;
}

staged_column stage_one_varchar_column(staging_state& s,
                                       duckdb::DatabaseInstance& db,
                                       duckdb::BlockManager& block_manager,
                                       duckdb::SingleFileBlockManager const* sf_bm,
                                       duckdb::BufferManager& buffer_manager,
                                       ::sirius::io::sirius_ioctx* io_ctx,
                                       ::sirius::io::sirius_io_object* io_obj,
                                       std::vector<duckdb_row_group_metadata> const& row_groups,
                                       std::size_t projected_col_idx)
{
  staged_column out;
  out.is_varchar = true;

  uint32_t row_cursor = 0;
  for (std::size_t rg_i = 0; rg_i < row_groups.size(); ++rg_i) {
    auto const& rg     = row_groups[rg_i];
    auto const& col_md = rg.columns.at(projected_col_idx);
    for (auto const& seg : col_md.data_segments) {
      if (!is_supported_varchar_codec(seg.compression)) {
        throw_unsupported("varchar data codec " +
                          std::to_string(static_cast<int>(seg.compression)) + " (column " +
                          std::to_string(col_md.column_id) + ")");
      }
      if (seg.block_id < 0) {
        throw_unsupported("varchar CONSTANT segment (column " + std::to_string(col_md.column_id) +
                          ")");
      }
      staged_segment ss;
      ss.row_offset        = row_cursor + static_cast<uint32_t>(seg.segment_start);
      ss.row_count         = static_cast<uint32_t>(seg.segment_count);
      ss.compression       = seg.compression;
      ss.max_string_length = *seg.max_string_length;  // walker invariant

      auto p = read_block_payload(io_ctx, io_obj, sf_bm, block_manager, buffer_manager, seg);
      record_staged(s, std::move(p), ss);
      out.data.push_back(ss);
    }

    if (column_has_real_nulls(col_md)) { out.has_nulls = true; }
    for (auto const& vseg : col_md.validity_segments) {
      if (is_constant_or_empty_validity(vseg.compression)) { continue; }
      staged_segment vs;
      vs.row_offset  = row_cursor + static_cast<uint32_t>(vseg.segment_start);
      vs.row_count   = static_cast<uint32_t>(vseg.segment_count);
      vs.compression = duckdb::CompressionType::COMPRESSION_UNCOMPRESSED;

      pinned_segment_bytes p;
      if (vseg.compression == duckdb::CompressionType::COMPRESSION_ROARING) {
        p = decode_roaring_validity(db, block_manager, vseg);
      } else if (vseg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        p = read_block_payload(io_ctx, io_obj, sf_bm, block_manager, buffer_manager, vseg);
      } else {
        throw_unsupported("validity codec " + std::to_string(static_cast<int>(vseg.compression)) +
                          " (varchar column " + std::to_string(col_md.column_id) + ")");
      }
      record_staged(s, std::move(p), vs);
      out.validity.push_back(vs);
    }
    row_cursor += static_cast<uint32_t>(rg.row_count);
  }

  out.total_rows = row_cursor;
  return out;
}

//===----------------------------------------------------------------------===//
// Bulk H2D copy.
//===----------------------------------------------------------------------===//

void copy_staged_to_device(rmm::device_buffer& device_buf,
                           staging_state const& s,
                           rmm::cuda_stream_view stream)
{
  auto* device_base = static_cast<uint8_t*>(device_buf.data());
  for (std::size_t i = 0; i < s.src_ptrs.size(); ++i) {
    RMM_CUDA_TRY(cudaMemcpyAsync(device_base + s.dst_offsets[i],
                                 s.src_ptrs[i],
                                 s.src_sizes[i],
                                 cudaMemcpyHostToDevice,
                                 stream.value()));
  }
  // Pageable→cuda_async_memory_resource H2D has an empirical stream-ordering
  // hazard: same-stream kernels can read pool residue at the destination.
  // Sync once per upload batch. To drop: pinned source AND lifetime that
  // outlives the kernel (event-tagged pool return or operator-owned staging).
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

}  // namespace

//===----------------------------------------------------------------------===//
// Public entry: pick_gpu_memory_space_for_duckdb_native_scan
//===----------------------------------------------------------------------===//

cucascade::memory::memory_space* pick_gpu_memory_space_for_duckdb_native_scan(
  duckdb_native_scan_info const& scan_info)
{
  if (scan_info.context == nullptr) {
    throw std::runtime_error(std::string(kTag) + " scan_info.context is null");
  }
  auto& ctx      = *scan_info.context;
  auto sirius_st = ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_st) {
    throw std::runtime_error(std::string(kTag) + " no sirius_state on the ClientContext");
  }
  auto& mem_mgr   = sirius_st->get_memory_manager();
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw std::runtime_error(std::string(kTag) + " no GPU-tier memory space registered");
  }
  return const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
}

//===----------------------------------------------------------------------===//
// Public entry: decode_duckdb_native_split
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::table> decode_duckdb_native_split(
  scan_manager::duckdb_native_split_provider::split_payload const& split,
  cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  if (split.row_groups.empty()) {
    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  }
  auto const& scan_info = *split.scan_info;
  auto& storage         = *scan_info.storage;
  auto& context         = *scan_info.context;

  auto& db             = duckdb::DatabaseInstance::GetDatabase(context);
  auto& sm             = storage.GetAttached().GetStorageManager();
  auto& block_manager  = sm.GetBlockManager();
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);

  // sirius_io routing: when the split was minted with an io_ctx + io_object,
  // route .db block reads through host_read instead of BufferManager::Pin.
  // The block-offset math is SingleFileBlockManager-specific; downcast once
  // and pass a pointer (null disables via_io routing).
  auto* io_ctx      = split.io_ctx.get();
  auto* io_obj      = split.db_io_object.get();
  auto const* sf_bm = (io_ctx != nullptr && io_obj != nullptr)
                        ? dynamic_cast<duckdb::SingleFileBlockManager const*>(&block_manager)
                        : nullptr;

  // PartitionRowGroup lookup needed for CONSTANT segments + held alive for the
  // duration of the decode (its destructor releases an internal reference).
  auto partition_stats = storage.GetPartitionStats(context);
  std::vector<std::unique_ptr<duckdb::BaseStatistics>> owned_stats_cache;

  auto mr_ref = mem_space.get_default_allocator();

  std::size_t const num_cols = scan_info.projected_cols.size();

  std::size_t total_rows = 0;
  for (auto const& rg : split.row_groups) {
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
      staged_cols.push_back(stage_one_varchar_column(
        staging, db, block_manager, sf_bm, buffer_manager, io_ctx, io_obj, split.row_groups, ci));
    } else {
      staged_cols.push_back(stage_one_fixed_width_column(staging,
                                                         db,
                                                         block_manager,
                                                         sf_bm,
                                                         buffer_manager,
                                                         io_ctx,
                                                         io_obj,
                                                         partition_stats,
                                                         owned_stats_cache,
                                                         split.row_groups,
                                                         ci,
                                                         scan_info.projected_types[ci]));
    }
  }

  rmm::device_buffer device_buf(staging.running_offset, stream, mr_ref);
  if (staging.running_offset > 0) { copy_staged_to_device(device_buf, staging, stream); }

  // Group fixed-width columns for a single gpu_decode_table call; varchar
  // columns each go through gpu_decode_strings_column separately.
  std::vector<gpu_column_decode_input> fw_inputs;
  std::vector<std::size_t> fw_to_final_idx;
  std::vector<gpu_string_column_decode_input> vc_inputs;
  std::vector<std::size_t> vc_to_final_idx;
  fw_inputs.reserve(num_cols);
  fw_to_final_idx.reserve(num_cols);

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

  std::vector<std::unique_ptr<cudf::column>> final_cols(num_cols);
  for (std::size_t fi = 0; fi < fw_cols.size(); ++fi) {
    final_cols[fw_to_final_idx[fi]] = std::move(fw_cols[fi]);
  }
  for (std::size_t vi = 0; vi < vc_cols.size(); ++vi) {
    final_cols[vc_to_final_idx[vi]] = std::move(vc_cols[vi]);
  }
  for (std::size_t ci = 0; ci < num_cols; ++ci) {
    if (!is_rowid_col[ci]) continue;
    final_cols[ci] = build_rowid_column(
      split.row_groups, static_cast<cudf::size_type>(total_rows), stream, mr_ref);
  }

  return std::make_unique<cudf::table>(std::move(final_cols));
}

}  // namespace sirius::op::scan
