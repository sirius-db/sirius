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

}  // namespace sirius::op::scan
