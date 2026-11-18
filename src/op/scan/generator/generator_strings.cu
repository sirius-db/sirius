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

#include "op/scan/generator/generator_detail.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/bit.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::op::scan::generator::detail {

namespace {

/// Size-then-fill functor for cudf::strings::detail::make_strings_children: the first pass
/// (`d_chars == nullptr`) writes each row's byte length, the second writes its characters.
struct format_varchar_fn {
  const std::int64_t* keys;
  const cudf::bitmask_type* null_mask;  // nullptr when there are no nulls
  std::int64_t min_length;              // p1
  std::uint64_t length_range;           // p2 - p1 + 1
  std::uint64_t batch_key;              // detail::batch_key(seed, batch_index)
  std::uint32_t column_index;

  cudf::size_type* d_sizes{nullptr};
  char* d_chars{nullptr};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    // Every hash of this row shares the row link of the chain.
    const auto row_hash = row_key(batch_key, static_cast<std::uint64_t>(idx));
    auto key            = static_cast<std::uint64_t>(keys[idx] < 0 ? 0 : keys[idx]);

    std::int64_t digits = 1;
    for (auto rest = key / 10; rest != 0; rest /= 10) {
      ++digits;
    }
    const auto prefix_length = digits > min_length ? digits : min_length;
    const auto drawn_length =
      min_length +
      static_cast<std::int64_t>(counter_hash_from_row_key(row_hash, column_index, salt::STR_LEN) %
                                length_range);
    const auto length = drawn_length > prefix_length ? drawn_length : prefix_length;

    if (d_chars == nullptr) {
      d_sizes[idx] = static_cast<cudf::size_type>(length);
      return;
    }

    char* out = d_chars + d_offsets[idx];
    for (auto i = prefix_length - 1; i >= 0; --i) {
      out[i] = static_cast<char>('0' + key % 10);
      key /= 10;
    }
    const auto suffix_length = static_cast<std::uint32_t>(length - prefix_length);
    for (std::uint32_t i = 0; i < suffix_length; ++i) {
      out[prefix_length + i] = static_cast<char>(
        'a' + counter_hash_from_row_key(row_hash, column_index, salt::STR_CHAR_BASE + i) % 26);
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> format_varchar(cudf::device_span<const std::int64_t> keys,
                                             rmm::device_buffer null_mask,
                                             cudf::size_type null_count,
                                             const column_spec& column,
                                             std::uint64_t seed,
                                             std::uint64_t batch_index,
                                             std::uint32_t column_index,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  // generate_batch has validated the column already; this guards direct callers.
  check_column(column);
  if (column.type != value_type::varchar) {
    throw std::invalid_argument("generator: format_varchar needs a VARCHAR column, got '" +
                                column.name + "'");
  }
  if (keys.size() > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("generator: too many rows for one VARCHAR column");
  }
  const auto rows = static_cast<cudf::size_type>(keys.size());
  if (null_count < 0 || null_count > rows) {
    throw std::invalid_argument("generator: VARCHAR null count out of range");
  }
  if (null_count > 0 &&
      null_mask.size() < cudf::num_bitmask_words(rows) * sizeof(cudf::bitmask_type)) {
    throw std::invalid_argument("generator: VARCHAR null mask is too small for the row count");
  }
  if (rows == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  const auto min_length = static_cast<std::int64_t>(column.p1);
  const auto max_drawn  = static_cast<std::int64_t>(column.p2);
  format_varchar_fn fn{.keys         = keys.data(),
                       .null_mask    = null_count > 0
                                         ? static_cast<const cudf::bitmask_type*>(null_mask.data())
                                         : nullptr,
                       .min_length   = min_length,
                       .length_range = static_cast<std::uint64_t>(max_drawn - min_length + 1),
                       .batch_key    = batch_key(seed, batch_index),
                       .column_index = column_index};

  auto [offsets, chars] = cudf::strings::detail::make_strings_children(fn, rows, stream, mr);
  // A mask that marks no row null is dropped: the column is simply non-nullable.
  return cudf::make_strings_column(rows,
                                   std::move(offsets),
                                   chars.release(),
                                   null_count,
                                   null_count > 0 ? std::move(null_mask) : rmm::device_buffer{});
}

}  // namespace sirius::op::scan::generator::detail
