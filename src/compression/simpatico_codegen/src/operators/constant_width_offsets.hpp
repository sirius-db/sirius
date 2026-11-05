// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>

namespace simpatico {

/**
 * @brief Build the INT32 offsets `i * width`, for `i` in `[0, rows]`, of `rows` strings that all
 * have `width` bytes.
 *
 * Offsets are computed on the device, so no host value is uploaded and the stream is not waited
 * on. Callers that can fall back to another decode should check the size bound first and decline.
 *
 * @throw std::invalid_argument if `rows` or `width` is negative
 * @throw std::overflow_error if `rows + 1` or `rows * width` exceeds the largest `cudf::size_type`
 *
 * @param rows Number of strings
 * @param width Byte width shared by every string
 * @param stream CUDA stream used for the allocation and the fill
 * @param mr Device memory resource used to allocate the offsets
 * @return INT32 column of `rows + 1` offsets
 */
[[nodiscard]] std::unique_ptr<cudf::column> make_constant_width_offsets(
  cudf::size_type rows,
  std::int32_t width,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace simpatico
