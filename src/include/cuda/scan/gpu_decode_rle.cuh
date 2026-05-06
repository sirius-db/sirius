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

// RLE decode entry for the GPU-native scan path. Routed from
// `dispatch_data_run` in gpu_native_decode.cu.

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/scan/gpu_native_decode.cuh>

#include <cstdint>

namespace sirius::cuda::scan {

/// 8-byte rle_count_offset header at the start of every RLE segment.
/// See `duckdb/src/storage/compression/rle.cpp::RLEConstants::RLE_HEADER_SIZE`.
inline constexpr uint32_t RLE_HEADER_SIZE = 8;

/// Decode an RLE codec run into `d_output`. Two device kernels per call:
/// `kernel_build_cumsum` (one CTA per segment) reads each segment's header
/// + counts and writes an inclusive prefix sum to a device-resident buffer;
/// `kernel_decode_rle` (one CTA per RLE_ROWS_PER_CHUNK-row slice) expands
/// rows via per-row upper_bound + value gather. Both kernels are launched on
/// `stream` with no D2H syncs.
void decode_rle_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type type,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
