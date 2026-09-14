/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace sirius::op::scan {

/**
 * @brief Filter @p view by a host-computed byte keep-mask.
 *
 * Uploads @p keep (one byte per row, 1 = keep) as a BOOL8 column and returns
 * @c cudf::apply_boolean_mask(view), so every host-computed keep-mask shares
 * one implementation.
 *
 * The upload is a @c cudaMemcpyAsync from pageable memory, which stages the
 * bytes out of @p keep before returning — the caller may free @p keep as soon
 * as this returns.
 *
 * @throws std::invalid_argument when keep.size() != view.num_rows().
 */
std::unique_ptr<cudf::table> apply_host_keep_mask(cudf::table_view const& view,
                                                  std::span<std::uint8_t const> keep,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

/**
 * @brief Filter @p view by a host-computed bit-packed keep-mask.
 *
 * @p keep_words is a cuDF-convention bitmask (bit `row % 32` of uint32 word
 * `row / 32`, 1 = keep; padding bits past @p row_count ignored). Uploads the
 * words, expands them with @c cudf::mask_to_bools, and returns
 * @c cudf::apply_boolean_mask(view). Used by the MVCC keep-mask path.
 *
 * ⚠ The upload from pinned host memory is a true-async DMA: @p keep_words
 * must stay alive until the work enqueued on @p stream completes.
 *
 * @throws std::invalid_argument on row_count != view.num_rows() or
 *         keep_words.size() != ceil(row_count / 32).
 */
std::unique_ptr<cudf::table> apply_host_keep_bitmask(cudf::table_view const& view,
                                                     std::span<std::uint32_t const> keep_words,
                                                     std::size_t row_count,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace sirius::op::scan
