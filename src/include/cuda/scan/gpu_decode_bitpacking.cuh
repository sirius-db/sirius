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
//! BITPACKING decode entry for the GPU-native scan path.
//!
//! The dispatcher in `gpu_native_decode.cu` routes a `gpu_codec_run` whose
//! codec is COMPRESSION_BITPACKING here. Per-segment metadata (mode, width,
//! FOR base, delta offset) lives inside each segment's bytes and is parsed
//! on device by the kernel.
//!
//! Modes covered: CONSTANT, CONSTANT_DELTA, FOR, DELTA_FOR. AUTO is a meta-
//! mode that DuckDB resolves to one of the four concrete modes per segment;
//! it never appears at this layer. INVALID / unknown modes zero-fill the
//! affected group's output range.

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/scan/gpu_native_decode.cuh>

#include <duckdb/common/vector_size.hpp>

#include <cstdint>

namespace sirius::cuda::scan {

//! Number of values per metadata group within a bitpacked segment. Fixed by
//! DuckDB's on-disk format — every metadata-group entry encodes the mode and
//! data offset for exactly this many rows (the last group of a segment may
//! be short). See duckdb/src/storage/compression/bitpacking.cpp. The
//! static_assert catches redefinitions of STANDARD_VECTOR_SIZE that break this
//! mirror of BITPACKING_METADATA_GROUP_SIZE.
static constexpr uint32_t BP_META_GROUP_SIZE = 2048;
static_assert(STANDARD_VECTOR_SIZE <= 512 || STANDARD_VECTOR_SIZE == 2048);

//! Mirrors `duckdb::BitpackingMode`. Values must match the encoding DuckDB
//! writes into the high byte of each 32-bit metadata-group entry. See
//! duckdb/src/include/duckdb/storage/compression/bitpacking.hpp.
//! @note The DuckDB enum does not assign explicit values, so this mirror can
//! drift if upstream reorders the variants.
enum class BitpackingMode : uint8_t {
  INVALID        = 0,
  AUTO           = 1,
  CONSTANT       = 2,
  CONSTANT_DELTA = 3,
  DELTA_FOR      = 4,
  FOR            = 5
};

//! Decode a bitpacking codec run into `d_output`. Every segment in `run` is
//! split into its metadata groups; one CTA decodes one group in a single
//! batched kernel launch.
//!
//! `d_output` must be sized for the column's full row count; each segment
//! writes `seg.row_count` rows starting at `seg.row_offset * type_size`.
void decode_bitpacking_data(gpu_codec_run const& run,
                            uint8_t* d_output,
                            cudf::data_type type,
                            uint32_t type_size,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
