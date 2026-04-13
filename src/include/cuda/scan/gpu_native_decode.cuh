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
#include <vector>

namespace sirius::cuda::scan {

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

}  // namespace sirius::cuda::scan
