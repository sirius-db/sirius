/*
 * Copyright 2025, Sirius Contributors.
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

#pragma once

//! @file
//! Public entry for the GPU-native DuckDB scan decode path. Takes per-column
//! descriptors that point at on-device segment bytes and decodes them into a
//! `cudf::table`. The dispatcher does no I/O — the caller stages segment bytes
//! on device first. Codec metadata (bitpacking mode, dictionary refs, FSST
//! symbol table, ...) lives inside the segment bytes; per-codec kernels parse
//! their own headers.

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <duckdb/common/enums/compression_type.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::cuda::scan {

/// One DuckDB segment, already staged on device.
///
/// `d_bytes`     device pointer to the segment's raw bytes.
/// `bytes_size`  size of the buffer behind `d_bytes`. Used as a sanity bound
///               by codec kernels (e.g. CONSTANT requires at least one full
///               value of bytes).
/// `row_offset`  starting row index of this segment within the column. The
///               column's segments tile a `[0, total_rows)` row range.
/// `row_count`   number of rows this segment produces.
struct gpu_segment_desc {
  const uint8_t* d_bytes;
  uint32_t bytes_size;
  uint32_t row_offset;
  uint32_t row_count;
};

/// A run of segments that share the same codec. Per-codec kernels consume the
/// full `segments` vector in one launch (this is the batching unit). A column
/// may have multiple runs if its segments use different codecs.
struct gpu_codec_run {
  duckdb::CompressionType codec;
  std::vector<gpu_segment_desc> segments;
};

/// Inputs for one column. `data` and `validity` are runs grouped by codec.
/// Fits codecs whose state is per-segment.
struct gpu_column_decode_input {
  cudf::data_type out_type;
  std::vector<gpu_codec_run> data;
  std::vector<gpu_codec_run> validity;
  uint32_t total_rows;
  bool has_nulls;
};

/// Decode every column in `cols` into a `cudf::table`. All device work runs
/// on `stream`; the dispatcher synchronises the stream once before returning,
/// so columns come back with their `null_count` already populated.
///
/// Allocations go through `mr`. Throws `std::runtime_error` if a column
/// violates a viability invariant (unsupported codec, non-fixed-width type,
/// malformed validity offset, row count > `cudf::size_type` max). Callers are
/// expected to pre-filter unsupported columns; these throws are a defensive
/// backstop, not the primary gate.
std::unique_ptr<cudf::table> gpu_decode_table(std::vector<gpu_column_decode_input> const& cols,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
