/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/direct_block_scan.hpp"

#include <log/logging.hpp>

// duckdb
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/column_segment_tree.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>

#include <chrono>
#include <cstring>

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
// DuckDB GetDirectBlockPointer() — zero-copy mmap bypass
//
// Uses SingleFileBlockManager::GetDirectBlockPointer(block_id) which does a
// lazy MAP_SHARED mmap (no MADV_POPULATE_READ) and returns a pointer directly
// to block data past the block header. Thread-safe: atomic fast path avoids
// lock contention once initialized (unlike the old g_mmap_mutex per-call lock).
//
// SIRIUS_DISABLE_MMAP=1 forces Pin() for benchmarking.
//===----------------------------------------------------------------------===//

static duckdb::BlockManager* try_get_block_manager(duckdb::DataTable& storage)
{
  if (const char* env = std::getenv("SIRIUS_DISABLE_MMAP")) {
    if (env[0] != '0') return nullptr;
  }
  try {
    auto& db   = storage.GetAttached();
    auto& smgr = db.GetStorageManager();
    return &smgr.GetBlockManager();
  } catch (...) {}
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Segment tree scan
//===----------------------------------------------------------------------===//

direct_block_scan_result scan_segment_tree(duckdb::ColumnData& col_data,
                                           duckdb::BufferManager& buffer_manager,
                                           bool* has_nulls_out,
                                           duckdb::BlockManager* block_manager)
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
      seg_info.compression  = segment.GetCompressionType();
      result.total_pinned_bytes += segment.SegmentSize();

      bool used_direct = false;
      if (block_manager && seg_info.block_id >= 0) {
        auto* block_base = block_manager->GetDirectBlockPointer(seg_info.block_id);
        if (block_base) {
          seg_info.data_ptr = block_base + seg_info.block_offset;
          used_direct = true;
        }
      }
      if (!used_direct) {
        seg_info.handle   = buffer_manager.Pin(segment.block);
        seg_info.data_ptr = seg_info.handle.Ptr() + seg_info.block_offset;
      }

      if (duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
        seg_info.max_string_length = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
      }
    } else {
      auto compression = segment.GetCompressionType();
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
//===----------------------------------------------------------------------===//

column_scan_result direct_block_scan_column_range(duckdb::DataTable& storage,
                                                  duckdb::StorageIndex col_idx,
                                                  duckdb::ClientContext& context,
                                                  const std::vector<duckdb::RowGroup*>& row_groups)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);
  auto* block_manager  = try_get_block_manager(storage);

  column_scan_result result;
  result.data.total_rows     = 0;
  result.validity.total_rows = 0;

  auto start = std::chrono::steady_clock::now();

  for (auto* rg : row_groups) {
    auto& col_data = rg->GetColumnDirect(col_idx);

    auto data_scan = scan_segment_tree(col_data, buffer_manager, &result.has_nulls, block_manager);
    result.data.total_pinned_bytes += data_scan.total_pinned_bytes;
    for (auto& s : data_scan.segments) {
      result.data.total_rows += s.row_count;
      result.data.segments.push_back(std::move(s));
    }

    try {
      auto& std_col  = col_data.Cast<duckdb::StandardColumnData>();
      auto& val_data = std_col.GetValidityData();

      if (result.has_nulls) {
        auto val_scan = scan_segment_tree(val_data, buffer_manager, nullptr, block_manager);
        result.validity.total_pinned_bytes += val_scan.total_pinned_bytes;
        for (auto& s : val_scan.segments) {
          result.validity.segments.push_back(std::move(s));
        }
      }
    } catch (...) {}
  }

  result.validity.total_rows = result.data.total_rows;

  auto end = std::chrono::steady_clock::now();
  result.data.pin_time_us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  result.validity.pin_time_us = result.data.pin_time_us;

  return result;
}

}  // namespace sirius::op::scan
