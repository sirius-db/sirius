/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "op/scan/direct_block_scan.hpp"

#include <log/logging.hpp>

// duckdb
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/column_segment_tree.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>

#include <chrono>

namespace sirius::op::scan {

direct_block_scan_result direct_block_scan_column(
  duckdb::DataTable& storage,
  duckdb::StorageIndex col_idx,
  duckdb::ClientContext& context)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);

  direct_block_scan_result result;
  result.total_rows = storage.GetTotalRows();

  auto start = std::chrono::steady_clock::now();

  // Walk row groups → column → segments directly, pinning each block
  auto& rg_collection = *storage.GetRowGroupCollectionRef();
  auto rg_tree_ptr    = rg_collection.GetRowGroupsDirect();
  auto rg_node        = rg_tree_ptr->GetRootSegment();

  while (rg_node) {
    auto& row_group = rg_node->GetNode();
    auto& col_data  = row_group.GetColumnDirect(col_idx);
    auto& seg_tree  = col_data.GetSegmentTree();

    auto seg_node = seg_tree.GetRootSegment();
    while (seg_node) {
      auto& segment = seg_node->GetNode();

      direct_block_scan_result::segment_info seg_info;
      seg_info.row_count = segment.count.load();

      if (segment.block) {
        // Pin the block — this gives us a raw pointer to the 256KB block data
        seg_info.handle       = buffer_manager.Pin(segment.block);
        seg_info.data_ptr     = seg_info.handle.Ptr() + segment.GetBlockOffset();
        seg_info.block_offset = segment.GetBlockOffset();
        seg_info.segment_size = segment.SegmentSize();
        seg_info.block_id     = segment.GetBlockId();
        seg_info.persistent   = true;

        seg_info.compression = segment.GetCompressionType();

        result.total_pinned_bytes += segment.SegmentSize();
      } else {
        seg_info.data_ptr   = nullptr;
        seg_info.persistent = false;
        seg_info.compression = duckdb::CompressionType::COMPRESSION_AUTO;
      }

      result.segments.push_back(std::move(seg_info));
      seg_node = seg_tree.GetNextSegment(*seg_node);
    }

    rg_node = rg_tree_ptr->GetNextSegment(*rg_node);
  }

  auto end = std::chrono::steady_clock::now();
  result.pin_time_us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Count compression types
  size_t n_uncompressed = 0, n_dictionary = 0, n_bitpacking = 0, n_constant = 0, n_other = 0;
  for (auto const& s : result.segments) {
    switch (s.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: n_uncompressed++; break;
      case duckdb::CompressionType::COMPRESSION_DICTIONARY: n_dictionary++; break;
      case duckdb::CompressionType::COMPRESSION_BITPACKING: n_bitpacking++; break;
      case duckdb::CompressionType::COMPRESSION_CONSTANT: n_constant++; break;
      default: n_other++; break;
    }
  }

  SIRIUS_LOG_INFO(
    "[direct_scan] col {}: {} segs pinned, {:.1f}MB, {}us ({:.1f}ms) "
    "[uncomp={} dict={} bitpack={} const={} other={}]",
    col_idx.GetPrimaryIndex(),
    result.segments.size(),
    result.total_pinned_bytes / (1024.0 * 1024.0),
    result.pin_time_us,
    result.pin_time_us / 1000.0,
    n_uncompressed, n_dictionary, n_bitpacking, n_constant, n_other);

  return result;
}

size_t direct_copy_fixed_column(
  direct_block_scan_result& scan_result,
  uint8_t* dest_buffer,
  size_t type_size)
{
  auto start = std::chrono::steady_clock::now();

  size_t dest_offset      = 0;
  size_t copied_bytes     = 0;
  size_t skipped_segments = 0;

  for (auto& seg : scan_result.segments) {
    if (!seg.data_ptr || !seg.persistent) { skipped_segments++; continue; }

    if (seg.compression == duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
      // Direct memcpy — data is contiguous fixed-width values at seg.data_ptr
      auto bytes = seg.row_count * type_size;
      std::memcpy(dest_buffer + dest_offset, seg.data_ptr, bytes);
      dest_offset += bytes;
      copied_bytes += bytes;
    } else if (seg.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
      // Constant segment — single value repeated for all rows
      // The value is at seg.data_ptr (or in stats). For now, skip.
      dest_offset += seg.row_count * type_size;
      skipped_segments++;
    } else {
      // Compressed (bitpacking, etc.) — need decompression, skip for now
      dest_offset += seg.row_count * type_size;
      skipped_segments++;
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto us  = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  SIRIUS_LOG_INFO("[direct_scan] fixed copy: {:.1f}MB in {}us ({:.1f}ms), {} skipped",
                  copied_bytes / (1024.0 * 1024.0), us, us / 1000.0, skipped_segments);

  return copied_bytes;
}

/// @brief Walk a ColumnData's segment tree, pin all blocks, return segment info vector.
direct_block_scan_result scan_segment_tree(
  duckdb::ColumnData& col_data,
  duckdb::BufferManager& buffer_manager)
{
  direct_block_scan_result result;
  auto& seg_tree = col_data.GetSegmentTree();
  auto seg_node  = seg_tree.GetRootSegment();

  while (seg_node) {
    auto& segment = seg_node->GetNode();
    direct_block_scan_result::segment_info seg_info;
    seg_info.row_count = segment.count.load();

    if (segment.block) {
      seg_info.handle       = buffer_manager.Pin(segment.block);
      seg_info.data_ptr     = seg_info.handle.Ptr() + segment.GetBlockOffset();
      seg_info.block_offset = segment.GetBlockOffset();
      seg_info.segment_size = segment.SegmentSize();
      seg_info.block_id     = segment.GetBlockId();
      seg_info.persistent   = true;
      seg_info.compression  = segment.GetCompressionType();
      result.total_pinned_bytes += segment.SegmentSize();

      // Extract max_string_length from segment stats (VARCHAR columns only)
      if (duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
        seg_info.max_string_length = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
      }
    } else {
      seg_info.data_ptr   = nullptr;
      seg_info.persistent = false;
      seg_info.compression = duckdb::CompressionType::COMPRESSION_AUTO;
    }

    result.segments.push_back(std::move(seg_info));
    seg_node = seg_tree.GetNextSegment(*seg_node);
  }
  return result;
}

column_scan_result direct_block_scan_column_full(
  duckdb::DataTable& storage,
  duckdb::StorageIndex col_idx,
  duckdb::ClientContext& context)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);

  column_scan_result result;
  result.data.total_rows = storage.GetTotalRows();
  result.validity.total_rows = result.data.total_rows;

  auto start = std::chrono::steady_clock::now();

  auto& rg_collection = *storage.GetRowGroupCollectionRef();
  auto rg_tree_ptr    = rg_collection.GetRowGroupsDirect();
  auto rg_node        = rg_tree_ptr->GetRootSegment();

  while (rg_node) {
    auto& row_group = rg_node->GetNode();
    auto& col_data  = row_group.GetColumnDirect(col_idx);

    // Scan data segments
    auto data_scan = scan_segment_tree(col_data, buffer_manager);
    result.data.total_pinned_bytes += data_scan.total_pinned_bytes;
    for (auto& s : data_scan.segments) {
      result.data.segments.push_back(std::move(s));
    }

    // Check nullability from data segment statistics
    {
      auto& seg_tree = col_data.GetSegmentTree();
      auto seg_node2 = seg_tree.GetRootSegment();
      while (seg_node2) {
        auto& segment = seg_node2->GetNode();
        if (segment.stats.statistics.CanHaveNull()) {
          result.has_nulls = true;
        }
        seg_node2 = seg_tree.GetNextSegment(*seg_node2);
      }
    }

    // Scan validity segments (StandardColumnData has a validity child)
    try {
      auto& std_col  = col_data.Cast<duckdb::StandardColumnData>();
      auto& val_data = std_col.GetValidityData();

      // Only scan validity if there are actually nulls — otherwise the validity
      // column may have an empty/uninitialized segment tree.
      if (result.has_nulls) {
        auto val_scan  = scan_segment_tree(val_data, buffer_manager);
        result.validity.total_pinned_bytes += val_scan.total_pinned_bytes;
        for (auto& s : val_scan.segments) {
          result.validity.segments.push_back(std::move(s));
        }
      }
    } catch (...) {
      // Not a StandardColumnData (e.g. struct/list) — no validity to scan
    }

    rg_node = rg_tree_ptr->GetNextSegment(*rg_node);
  }

  auto end = std::chrono::steady_clock::now();
  result.data.pin_time_us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  result.validity.pin_time_us = result.data.pin_time_us;

  // Log summary
  size_t n_uncomp = 0, n_dict = 0, n_bp = 0, n_const = 0, n_other = 0;
  for (auto const& s : result.data.segments) {
    switch (s.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: n_uncomp++; break;
      case duckdb::CompressionType::COMPRESSION_DICTIONARY:   n_dict++; break;
      case duckdb::CompressionType::COMPRESSION_BITPACKING:   n_bp++; break;
      case duckdb::CompressionType::COMPRESSION_CONSTANT:     n_const++; break;
      default: n_other++; break;
    }
  }

  size_t n_val_empty = 0, n_val_uncomp = 0, n_val_other = 0;
  for (auto const& s : result.validity.segments) {
    switch (s.compression) {
      case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED: n_val_uncomp++; break;
      // EMPTY validity uses a special compression type or has no block
      default:
        if (!s.persistent || !s.data_ptr) { n_val_empty++; }
        else { n_val_other++; }
        break;
    }
  }

  SIRIUS_LOG_INFO(
    "[direct_scan_full] col {}: {} data segs [uncomp={} dict={} bp={} const={} other={}], "
    "{} validity segs [empty={} uncomp={} other={}], has_nulls={}, "
    "{:.1f}MB pinned in {:.1f}ms",
    col_idx.GetPrimaryIndex(),
    result.data.segments.size(), n_uncomp, n_dict, n_bp, n_const, n_other,
    result.validity.segments.size(), n_val_empty, n_val_uncomp, n_val_other,
    result.has_nulls,
    (result.data.total_pinned_bytes + result.validity.total_pinned_bytes) / (1024.0 * 1024.0),
    result.data.pin_time_us / 1000.0);

  return result;
}

column_scan_result direct_block_scan_column_range(
  duckdb::DataTable& storage,
  duckdb::StorageIndex col_idx,
  duckdb::ClientContext& context,
  const std::vector<duckdb::RowGroup*>& row_groups)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(context);

  column_scan_result result;
  result.data.total_rows     = 0;
  result.validity.total_rows = 0;

  auto start = std::chrono::steady_clock::now();

  for (auto* rg : row_groups) {
    auto& col_data = rg->GetColumnDirect(col_idx);

    // Count rows in this row group
    auto& seg_tree = col_data.GetSegmentTree();
    {
      auto seg_node = seg_tree.GetRootSegment();
      while (seg_node) {
        result.data.total_rows += seg_node->GetNode().count.load();
        seg_node = seg_tree.GetNextSegment(*seg_node);
      }
    }

    // Scan data segments
    auto data_scan = scan_segment_tree(col_data, buffer_manager);
    result.data.total_pinned_bytes += data_scan.total_pinned_bytes;
    for (auto& s : data_scan.segments) {
      result.data.segments.push_back(std::move(s));
    }

    // Check nullability
    {
      auto seg_node = seg_tree.GetRootSegment();
      while (seg_node) {
        if (seg_node->GetNode().stats.statistics.CanHaveNull()) {
          result.has_nulls = true;
        }
        seg_node = seg_tree.GetNextSegment(*seg_node);
      }
    }

    // Scan validity segments
    try {
      auto& std_col  = col_data.Cast<duckdb::StandardColumnData>();
      auto& val_data = std_col.GetValidityData();

      if (result.has_nulls) {
        auto val_scan = scan_segment_tree(val_data, buffer_manager);
        result.validity.total_pinned_bytes += val_scan.total_pinned_bytes;
        for (auto& s : val_scan.segments) {
          result.validity.segments.push_back(std::move(s));
        }
      }
    } catch (...) {
      // Not a StandardColumnData — no validity to scan
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
