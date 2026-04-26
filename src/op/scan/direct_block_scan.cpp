/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

// File contract:
//   Walks DuckDB's per-column segment tree for a caller-supplied range of
//   row groups and returns a `column_scan_result` describing each segment's
//   on-disk byte layout (block id, offset, size, compression type) plus
//   pinned `BufferHandle`s that keep those blocks resident while the
//   downstream GPU decoder runs.
//
//   Read-only paths only. Walks the *data* segment tree; the validity tree
//   walk is deferred to PR E (gates on a `StandardColumnData::GetValidityData`
//   accessor that is not in upstream DuckDB 1.5.2). Until E lands,
//   `column_scan_result.validity` is left empty and `has_nulls` stays false —
//   E's viability check refuses any column with potential nulls so the gap
//   is plan-time-correct, not silently-wrong.

#include "op/scan/direct_block_scan.hpp"

#include <log/logging.hpp>

// duckdb
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/function/compression_function.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/column_segment_tree.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/segment_tree.hpp>

#include <chrono>
#include <cstring>
#include <vector>

namespace sirius::op::scan {

template <typename T>
static void extract_typed_constant(const duckdb::BaseStatistics& stats, uint8_t* dest)
{
  auto v = duckdb::NumericStats::GetMin<T>(stats);
  std::memcpy(dest, &v, sizeof(v));
}

static void extract_constant_from_stats(const duckdb::ColumnSegment& segment, uint8_t* dest)
{
  auto& stats = segment.stats.statistics;
  switch (segment.type.InternalType()) {
    case duckdb::PhysicalType::BOOL:
    case duckdb::PhysicalType::INT8: extract_typed_constant<int8_t>(stats, dest); break;
    case duckdb::PhysicalType::INT16: extract_typed_constant<int16_t>(stats, dest); break;
    case duckdb::PhysicalType::INT32: extract_typed_constant<int32_t>(stats, dest); break;
    case duckdb::PhysicalType::INT64: extract_typed_constant<int64_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT8: extract_typed_constant<uint8_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT16: extract_typed_constant<uint16_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT32: extract_typed_constant<uint32_t>(stats, dest); break;
    case duckdb::PhysicalType::UINT64: extract_typed_constant<uint64_t>(stats, dest); break;
    case duckdb::PhysicalType::FLOAT: extract_typed_constant<float>(stats, dest); break;
    case duckdb::PhysicalType::DOUBLE: extract_typed_constant<double>(stats, dest); break;
    case duckdb::PhysicalType::INT128:
      extract_typed_constant<duckdb::hugeint_t>(stats, dest);
      break;
    default: std::memset(dest, 0, 16); break;
  }
}

//===----------------------------------------------------------------------===//
// Segment tree scan (Pin-only path)
//
// Walks the segment tree of a single ColumnData, producing a
// `direct_block_scan_result` whose `segments` vector has one entry per
// segment. For each persistent segment with a backing block, pin the block
// via BufferManager and record the data pointer + offset + size so the
// downstream H2D step can copy raw bytes to GPU.
//
// COMPRESSION_CONSTANT segments are blockless — the value lives in segment
// stats. We extract it into segment_info::constant_data (16 bytes inline)
// and leave data_ptr null; the downstream decoder fills the column with the
// constant.
//===----------------------------------------------------------------------===//

static direct_block_scan_result scan_segment_tree(duckdb::ColumnData& col_data,
                                                  duckdb::BufferManager& buffer_manager,
                                                  bool* has_nulls_out)
{
  direct_block_scan_result result;
  auto& seg_tree = col_data.GetSegmentTree();
  auto seg_node  = seg_tree.GetRootSegment();

  while (seg_node) {
    auto& segment = seg_node->GetNode();
    direct_block_scan_result::segment_info seg_info;
    seg_info.row_count = segment.count.load();

    if (has_nulls_out && segment.stats.statistics.CanHaveNull()) { *has_nulls_out = true; }

    if (segment.block) {
      seg_info.block_offset = segment.GetBlockOffset();
      seg_info.segment_size = segment.SegmentSize();
      seg_info.block_id     = segment.GetBlockId();
      seg_info.persistent   = true;
      seg_info.compression  = segment.GetCompressionFunction().type;
      result.total_pinned_bytes += segment.SegmentSize();

      seg_info.handle   = buffer_manager.Pin(segment.block);
      seg_info.data_ptr = seg_info.handle.Ptr() + seg_info.block_offset;

      if (duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
        seg_info.max_string_length = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
      }
    } else {
      auto compression = segment.GetCompressionFunction().type;
      if (compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        seg_info.compression = compression;
        seg_info.persistent  = true;
        seg_info.data_ptr    = nullptr;
        extract_constant_from_stats(segment, seg_info.constant_data);
      } else {
        seg_info.data_ptr    = nullptr;
        seg_info.persistent  = false;
        seg_info.compression = duckdb::CompressionType::COMPRESSION_AUTO;
      }
    }

    result.segments.push_back(std::move(seg_info));
    seg_node = seg_tree.GetNextSegment(*seg_node);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Public API
//
// `column_scan_result.validity` is intentionally left unpopulated and
// `has_nulls` is the data-side has-null indicator only — it does not yet
// reflect a peek at the validity tree. PR E will populate validity once the
// `StandardColumnData::GetValidityData` accessor is available, and will
// gate gpu_execution at viability so any nullable column falls back to
// duckdb_scan_task until then.
//===----------------------------------------------------------------------===//

column_scan_result direct_block_scan_column_range(duckdb::DataTable& storage,
                                                  duckdb::StorageIndex col_idx,
                                                  duckdb::ClientContext& context,
                                                  const std::vector<duckdb::RowGroup*>& row_groups)
{
  (void)storage;  // reserved for the mmap fast path (per-BlockManager state cache)
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);

  column_scan_result result;
  result.data.total_rows     = 0;
  result.validity.total_rows = 0;

  auto start = std::chrono::steady_clock::now();

  for (auto* rg : row_groups) {
    auto& col_data = rg->GetColumnDirect(col_idx);

    auto data_scan = scan_segment_tree(col_data, buffer_manager, &result.has_nulls);
    result.data.total_pinned_bytes += data_scan.total_pinned_bytes;
    for (auto& s : data_scan.segments) {
      result.data.total_rows += s.row_count;
      result.data.segments.push_back(std::move(s));
    }
  }

  result.validity.total_rows = result.data.total_rows;

  auto end = std::chrono::steady_clock::now();
  result.data.pin_time_us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  result.validity.pin_time_us = result.data.pin_time_us;

  return result;
}

}  // namespace sirius::op::scan
