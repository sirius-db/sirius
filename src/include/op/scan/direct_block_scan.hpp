/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_index.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius::op::scan {

struct direct_block_scan_result {
  struct segment_info {
    duckdb::BufferHandle handle;              // Keeps block pinned
    duckdb::data_ptr_t data_ptr = nullptr;    // Raw pointer to segment data
    duckdb::idx_t block_offset  = 0;
    duckdb::idx_t row_count     = 0;
    duckdb::idx_t segment_size  = 0;
    duckdb::block_id_t block_id = -1;
    bool persistent             = false;
    duckdb::CompressionType compression = duckdb::CompressionType::COMPRESSION_AUTO;
    uint32_t max_string_length  = 0;   // From segment stats (VARCHAR only, 0 = unknown)
  };

  std::vector<segment_info> segments;
  size_t total_rows         = 0;
  size_t total_pinned_bytes = 0;
  int64_t pin_time_us       = 0;
};

/// @brief Full column scan result including both data and validity segments.
struct column_scan_result {
  direct_block_scan_result data;
  direct_block_scan_result validity;
  bool has_nulls = false;  // True if any data segment reports HasNull()
};

/// @brief Walk segment tree, pin all blocks for a column.
/// Returns raw data pointers ready for cudaMemcpy or GPU decode.
direct_block_scan_result direct_block_scan_column(
  duckdb::DataTable& storage,
  duckdb::StorageIndex col_idx,
  duckdb::ClientContext& context);

/// @brief Copy uncompressed fixed-width segments directly via memcpy.
/// Returns total bytes copied. Skips compressed segments.
size_t direct_copy_fixed_column(
  direct_block_scan_result& scan_result,
  uint8_t* dest_buffer,
  size_t type_size);

/// @brief Walk segment tree for both data and validity, pin all blocks.
/// Returns data segments + validity segments ready for GPU decode.
column_scan_result direct_block_scan_column_full(
  duckdb::DataTable& storage,
  duckdb::StorageIndex col_idx,
  duckdb::ClientContext& context);

}  // namespace sirius::op::scan
