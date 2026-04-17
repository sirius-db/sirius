/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "op/scan/direct_block_scan.hpp"

#include <cudf/column/column.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <unordered_map>

namespace sirius::cuda::scan {

/// @brief Decode a string column using batched two-pass approach.
///
/// Pass 1: Compute string lengths for ALL segments (one batched kernel per
///         compression type), then a single global CUB ExclusiveSum produces
///         contiguous offsets with zero gaps.  ONE sync to read total chars.
///
/// Pass 2: Gather strings for ALL segments into exact-size chars buffer
///         (one batched kernel per compression type).
///
/// This replaces the per-segment decode loop that synced N times with a
/// pipeline that syncs exactly ONCE between passes.
///
/// @param col_scan        Column scan result with pinned segment data
/// @param stream          CUDA stream
/// @param mr              Device memory resource
/// @param device_blocks   Optional: pre-transferred block map (from pipelined path).
///                        When non-null, blocks are read from device_staging instead of H2D.
/// @param device_staging  Optional: device staging buffer (from pipelined path).
/// @param d_valid_count_out  If non-null, write valid count here (async, no sync).
/// @return cudf string column
std::unique_ptr<cudf::column> decode_string_column_batched(
    sirius::op::scan::column_scan_result& col_scan,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    const std::unordered_map<int64_t, size_t>* device_blocks = nullptr,
    uint8_t* device_staging = nullptr,
    uint32_t* d_valid_count_out = nullptr);

}  // namespace sirius::cuda::scan
