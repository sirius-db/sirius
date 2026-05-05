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

//===----------------------------------------------------------------------===//
// RLE decode entry for the GPU-native scan path.
//
// The dispatcher in `gpu_native_decode.cu` routes a `gpu_codec_run` whose
// codec is COMPRESSION_RLE here. Per-segment header (8-byte rle_count_offset
// + values[entry_count] + counts[entry_count] of uint16) is parsed on the
// HOST: the dispatcher's contract says "kernels parse their own headers" by
// default, but RLE deviates because the host-walk-and-prefix-sum step is
// cheap (counts area ≤ ~256 KiB worst-case) and lets the kernel skip a
// device-side scan entirely. The cost is two D2H stream syncs per column
// regardless of segment count, batched across all segments.
//
// Malformed segments (out-of-range rle_count_offset, count walk that doesn't
// reach row_count, count == 0) are demoted to a zero-fill chunk on the
// device — never leave the output buffer carrying uninitialised contents.
// Viability is expected to keep malformed segments out of this dispatcher;
// zero-fill is a defensive backstop, not a runtime fallback.
//===----------------------------------------------------------------------===//

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/scan/gpu_native_decode.cuh>

#include <cstdint>

namespace sirius::cuda::scan {

/// Size of the per-segment header (one uint64_t holding rle_count_offset).
/// Fixed by DuckDB's on-disk format — see `duckdb/src/storage/compression/
/// rle.cpp::RLEConstants::RLE_HEADER_SIZE`.
inline constexpr uint32_t RLE_HEADER_SIZE = 8;

/// Decode an RLE codec run into `d_output`. Each segment is parsed on host
/// (header → counts → inclusive prefix sums); the kernel then expands one
/// chunk of `RLE_ROWS_PER_CHUNK` rows per CTA via per-row binary search of
/// the prefix-sum array. Multiple segments are issued as one batched kernel
/// launch.
///
/// `d_output` must be sized for the column's full row count; each segment
/// writes `seg.row_count` rows starting at `seg.row_offset * type_size`.
///
/// All work is enqueued on `stream`; the function performs two stream syncs
/// internally (header fetch, counts fetch). Descriptor + cumsum staging uses
/// `mr` so the allocation is tracked by the reservation system rather than
/// a bare cudaMallocAsync.
void decode_rle_data(gpu_codec_run const& run,
                     uint8_t* d_output,
                     cudf::data_type type,
                     uint32_t type_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr);

}  // namespace sirius::cuda::scan
