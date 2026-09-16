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

// ALP / ALPRD decode entries for the GPU-native scan path. On any bounds
// check failure the kernel zero-fills the trusted host descriptor row
// range. On-disk format reference and per-vector layouts: gpu_decode_alp.cu.

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/scan/gpu_native_decode.cuh>

#include <duckdb/storage/compression/alp/alp_constants.hpp>
#include <duckdb/storage/compression/alprd/alprd_constants.hpp>

#include <cstdint>

namespace sirius::cuda::scan {

// Aliases for DuckDB's per-codec format constants — keep this header in
// sync with the on-disk format by reading the value from DuckDB rather
// than redefining it.
inline constexpr uint32_t ALP_VECTOR_SIZE = duckdb::AlpConstants::ALP_VECTOR_SIZE;
inline constexpr uint8_t ALP_UNCOMPRESSED_SENTINEL =
  duckdb::AlpConstants::UNCOMPRESSED_MODE_SENTINEL;
inline constexpr uint16_t ALPRD_UNCOMPRESSED_SENTINEL =
  duckdb::AlpRDConstants::UNCOMPRESSED_MODE_SENTINEL;
inline constexpr uint32_t ALPRD_MAX_DICT_SIZE = duckdb::AlpRDConstants::MAX_DICTIONARY_SIZE;
inline constexpr uint32_t ALPRD_HEADER_SIZE   = duckdb::AlpRDConstants::HEADER_SIZE;

// `type_size` must be 4 (float) or 8 (double); other widths throw.
// Output is written at `seg.row_offset * type_size` for each segment.
void decode_alp_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type type,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr);

void decode_alprd_data(gpu_codec_run const& run,
                       uint8_t* d_output,
                       cudf::data_type type,
                       uint32_t type_size,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
