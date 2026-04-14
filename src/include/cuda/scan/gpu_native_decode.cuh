/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <op/scan/direct_block_scan.hpp>

#include <cudf/table/table.hpp>
#include <duckdb/common/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <unordered_map>
#include <vector>

namespace sirius::cuda::scan {

/// @brief Map of DuckDB block_id → byte offset in device staging buffer.
struct device_block_map {
  std::unordered_map<int64_t, size_t> offsets;
  size_t total_bytes = 0;
};

/// @brief Decode all columns from direct block scan results into a cudf::table on GPU.
///
/// Hybrid routing: each segment is decoded via GPU kernel (bitpacking, dictionary,
/// uncompressed, constant) or falls back to CPU decode + H2D memcpy for unsupported
/// compression types (RLE, FSST, ALP, etc.).
///
/// @param column_scans   Per-column scan results (data + validity segments)
/// @param column_types   DuckDB logical types for each column (same order as column_scans)
/// @param stream         CUDA stream for async operations
/// @param mr             Device memory resource for GPU allocations
/// @return Owning cudf::table with all columns decoded in GPU memory
std::unique_ptr<cudf::table> gpu_decode_table(
    std::vector<sirius::op::scan::column_scan_result>& column_scans,
    const std::vector<duckdb::LogicalType>& column_types,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// @brief Decode table from pre-transferred device data (bulk pre-transfer path).
/// Like gpu_decode_table but reads compressed blocks from device_staging instead of
/// doing per-segment H2D. Used after bulk H2D of all unique blocks.
std::unique_ptr<cudf::table> gpu_decode_table_pipelined(
    std::vector<sirius::op::scan::column_scan_result>& col_scans,
    const std::vector<duckdb::LogicalType>& col_types,
    const device_block_map& blocks,
    void* device_staging,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
