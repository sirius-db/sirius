/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// Public dispatcher for the GPU-native decode path.
//
// `gpu_decode_table` takes a vector of `column_scan_result` (per-column
// segment lists produced by direct_block_scan) and a parallel vector of
// DuckDB logical types, and produces a `cudf::table` whose columns hold
// the decoded data on GPU.  All work is enqueued on the supplied stream;
// a single `stream.synchronize()` runs internally before returning so
// caller-side state (deferred null counts) is consistent.
//
// Caller contracts:
//   * `col_scans` and `col_types` must be the same length and aligned by
//     index.
//   * Any column whose codec is not implemented in this layer triggers
//     a throw — callers must filter via `check_viability` upstream (lives
//     in the gpu_native_scan_task layer, future PR E).  The throw is a
//     defensive assert: it means viability fell out of sync with the
//     dispatcher, not a runtime fallback path.
//
// Codec coverage as A1 ships: BITPACKING, UNCOMPRESSED, CONSTANT for
// fixed-width columns.  RLE, dictionary / FSST / DICT_FSST strings, and
// ALP / ALPRD floats arrive in PR A2 and PR A3.
//===----------------------------------------------------------------------===//

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
/// Pre-stages every unique block referenced by any column to device memory in one
/// coalesced bulk H2D (runs of contiguous block_ids share a single cudaMemcpyAsync),
/// then decodes all segments from device staging.  Supported segments are decoded
/// via GPU kernels; any unsupported codec or type causes a viability-invariant
/// throw — upstream `check_viability` is responsible for keeping such segments
/// out of this path.
///
/// A single stream.synchronize() runs at the end, after all per-column null-count
/// kernels have been enqueued — replacing N per-column syncs with 1.
///
/// @param col_scans   Per-column scan results (data + validity segments)
/// @param col_types   DuckDB logical types for each column (same order as col_scans)
/// @param stream      CUDA stream for async operations
/// @param mr          Device memory resource for GPU allocations
/// @return Owning cudf::table with all columns decoded in GPU memory
std::unique_ptr<cudf::table> gpu_decode_table(
  std::vector<sirius::op::scan::column_scan_result>& col_scans,
  const std::vector<duckdb::LogicalType>& col_types,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
