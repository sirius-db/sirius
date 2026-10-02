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

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sirius::exec {

/// One column of a batch sent by direct exchange; its buffers travel separately.
struct direct_column {
  cudf::data_type type;
  cudf::size_type null_count;
  bool has_mask;
  cudf::type_id offsets{cudf::type_id::EMPTY};  ///< STRING only: INT32 or INT64.
  std::uint64_t chars{0};                       ///< STRING only: bytes of character data.

  bool operator==(direct_column const&) const = default;
};

/**
 * @brief The shape of a batch sent by direct exchange.
 *
 * Its buffers are walked per column in this order: the null mask if any, the data (the chars
 * for STRING), then the STRING offsets.
 */
struct direct_layout {
  cudf::size_type rows;
  std::vector<direct_column> columns;

  bool operator==(direct_layout const&) const = default;
};

/// One buffer of a direct_layout: `wire` bytes are sent into an allocation of `alloc` bytes.
struct direct_buffer {
  std::size_t column;
  std::size_t wire;
  std::size_t alloc;

  bool operator==(direct_buffer const&) const = default;
};

[[nodiscard]] std::vector<std::uint8_t> encode_layout(direct_layout const& layout);

/// @throws sirius::invalid_input_exception unless @p bytes is a well-formed layout of at least
///         one row and one fixed-width or STRING column.
[[nodiscard]] direct_layout decode_layout(std::span<std::uint8_t const> bytes);

/// @p layout's buffers in walk order, each allocation 256-byte aligned within @p limit bytes.
/// @throws sirius::invalid_input_exception when the aligned allocations exceed @p limit.
[[nodiscard]] std::vector<direct_buffer> plan_buffers(direct_layout const& layout,
                                                      std::size_t limit);

/// A table's layout and its buffer addresses, in plan_buffers order.
struct direct_export {
  direct_layout layout;
  std::vector<void const*> buffers;
};

/// Describes @p table, which must have rows and no sliced (non-zero offset) column. STRING chars
/// sizes are read on @p stream.
/// @throws sirius::invalid_input_exception on a column that is neither fixed-width nor STRING.
[[nodiscard]] direct_export describe_table(cudf::table_view const& table,
                                           rmm::cuda_stream_view stream);

}  // namespace sirius::exec
