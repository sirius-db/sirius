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

#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::merge::detail {

namespace {

/// The (input, row) map as raw device pointers.
struct gather_map {
  std::int32_t const* input;
  cudf::size_type const* row;
};

/// Validity of output row j: the bit of its source row, or valid when the source window's column
/// has no null mask.
struct source_valid_fn {
  gather_map map;
  cudf::bitmask_type const* const* masks;  ///< Per window; nullptr when not nullable.
  cudf::size_type const* mask_offsets;     ///< Per window: the column view's offset().

  __device__ bool operator()(cudf::size_type j) const
  {
    auto const input = map.input[j];
    auto const* mask = masks[input];
    return mask == nullptr || cudf::bit_is_set(mask, mask_offsets[input] + map.row[j]);
  }
};

/// Copies one fixed-width value as the unsigned integer of its width.
template <typename Storage>
struct copy_fixed_fn {
  gather_map map;
  Storage const* const* sources;  ///< Per window: the first value of the (sliced) column.

  __device__ Storage operator()(cudf::size_type j) const
  {
    return sources[map.input[j]][map.row[j]];
  }
};

/// Where one window's STRING column keeps its rows.
struct string_source {
  char const* chars;    ///< Start of the chars buffer (offset values are relative to it).
  void const* offsets;  ///< Offset of the view's row 0: int32 or int64 values.
  bool large;           ///< True when the offsets are int64.

  __device__ std::int64_t offset_at(cudf::size_type row) const
  {
    return large ? static_cast<std::int64_t const*>(offsets)[row]
                 : static_cast<std::int64_t>(static_cast<std::int32_t const*>(offsets)[row]);
  }
};

/// (pointer, size) of the string of output row j; a null source row is an empty string.
struct string_pair_fn {
  gather_map map;
  string_source const* sources;
  bool nullable;
  source_valid_fn valid;

  __device__ cudf::strings::detail::string_index_pair operator()(cudf::size_type j) const
  {
    if (nullable && !valid(j)) { return {nullptr, 0}; }
    auto const& source = sources[map.input[j]];
    auto const row     = map.row[j];
    auto const begin   = source.offset_at(row);
    return {source.chars + begin, static_cast<cudf::size_type>(source.offset_at(row + 1) - begin)};
  }
};

/// The output null mask of one column and its null count; an empty buffer when no window's column
/// is nullable.
struct gathered_validity {
  rmm::device_buffer mask;
  cudf::size_type null_count{0};
  rmm::device_uvector<cudf::bitmask_type const*> masks;
  rmm::device_uvector<cudf::size_type> mask_offsets;
  bool nullable{false};

  [[nodiscard]] source_valid_fn device_fn(gather_map map) const
  {
    return source_valid_fn{map, masks.data(), mask_offsets.data()};
  }
};

gathered_validity gather_validity(std::span<const cudf::table_view> windows,
                                  cudf::size_type column,
                                  gather_map map,
                                  cudf::size_type rows,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  auto const temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::bitmask_type const*> masks(windows.size());
  std::vector<cudf::size_type> mask_offsets(windows.size());
  bool nullable = false;
  for (std::size_t i = 0; i < windows.size(); ++i) {
    auto const& col = windows[i].column(column);
    masks[i]        = col.null_mask();
    mask_offsets[i] = col.offset();
    nullable        = nullable || col.nullable();
  }
  gathered_validity result{rmm::device_buffer{},
                           0,
                           cudf::detail::make_device_uvector_async(masks, stream, temp_mr),
                           cudf::detail::make_device_uvector_async(mask_offsets, stream, temp_mr),
                           nullable};
  if (nullable) {
    auto const begin = cuda::counting_iterator<cudf::size_type>{0};
    std::tie(result.mask, result.null_count) =
      cudf::detail::valid_if(begin, begin + rows, result.device_fn(map), stream, mr);
  }
  return result;
}

template <typename Storage>
rmm::device_buffer gather_fixed_values(std::span<const cudf::table_view> windows,
                                       cudf::size_type column,
                                       gather_map map,
                                       cudf::size_type rows,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const temp_mr = cudf::get_current_device_resource_ref();
  std::vector<Storage const*> sources(windows.size());
  for (std::size_t i = 0; i < windows.size(); ++i) {
    auto const& col = windows[i].column(column);
    // data<T>() only accepts the column's own representation; the copy goes by width.
    sources[i] = static_cast<Storage const*>(col.head()) + col.offset();
  }
  auto const d_sources = cudf::detail::make_device_uvector_async(sources, stream, temp_mr);

  rmm::device_buffer values{static_cast<std::size_t>(rows) * sizeof(Storage), stream, mr};
  auto const begin = cuda::counting_iterator<cudf::size_type>{0};
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    begin,
                    begin + rows,
                    static_cast<Storage*>(values.data()),
                    copy_fixed_fn<Storage>{map, d_sources.data()});
  return values;
}

std::unique_ptr<cudf::column> gather_fixed_width(std::span<const cudf::table_view> windows,
                                                 cudf::size_type column,
                                                 gather_map map,
                                                 cudf::size_type rows,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const type = windows.front().column(column).type();
  rmm::device_buffer values;
  switch (cudf::size_of(type)) {
    case 1:
      values = gather_fixed_values<std::uint8_t>(windows, column, map, rows, stream, mr);
      break;
    case 2:
      values = gather_fixed_values<std::uint16_t>(windows, column, map, rows, stream, mr);
      break;
    case 4:
      values = gather_fixed_values<std::uint32_t>(windows, column, map, rows, stream, mr);
      break;
    case 8:
      values = gather_fixed_values<std::uint64_t>(windows, column, map, rows, stream, mr);
      break;
    case 16:
      values = gather_fixed_values<unsigned __int128>(windows, column, map, rows, stream, mr);
      break;
    default:
      throw std::invalid_argument("gather_kway: unsupported fixed-width size in column " +
                                  std::to_string(column));
  }
  auto validity = gather_validity(windows, column, map, rows, stream, mr);
  return std::make_unique<cudf::column>(
    type, rows, std::move(values), std::move(validity.mask), validity.null_count);
}

std::unique_ptr<cudf::column> gather_strings(std::span<const cudf::table_view> windows,
                                             cudf::size_type column,
                                             gather_map map,
                                             cudf::size_type rows,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const temp_mr = cudf::get_current_device_resource_ref();
  std::vector<string_source> sources(windows.size());
  for (std::size_t i = 0; i < windows.size(); ++i) {
    auto const& col = windows[i].column(column);
    // A 0-row strings column may have no children; the map cannot reference it.
    if (col.size() == 0 || col.num_children() == 0) {
      sources[i] = string_source{nullptr, nullptr, false};
      continue;
    }
    cudf::strings_column_view const strings{col};
    auto const offsets = strings.offsets();
    auto const large   = offsets.type().id() == cudf::type_id::INT64;
    auto const* first  = large
                           ? static_cast<void const*>(offsets.data<std::int64_t>() + col.offset())
                           : static_cast<void const*>(offsets.data<std::int32_t>() + col.offset());
    sources[i]         = string_source{strings.chars_begin(stream), first, large};
  }
  auto const d_sources = cudf::detail::make_device_uvector_async(sources, stream, temp_mr);
  auto validity        = gather_validity(windows, column, map, rows, stream, mr);

  auto const pairs = cuda::transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    string_pair_fn{map, d_sources.data(), validity.nullable, validity.device_fn(map)});
  auto const sizes = cuda::transform_iterator(
    pairs,
    cuda::proclaim_return_type<cudf::size_type>(
      [] __device__(cudf::strings::detail::string_index_pair const& pair) { return pair.second; }));

  // Switches to int64 offsets past the large-strings threshold.
  auto [offsets, bytes] =
    cudf::strings::detail::make_offsets_child_column(sizes, sizes + rows, stream, mr);
  auto chars =
    cudf::strings::detail::make_chars_buffer(offsets->view(), bytes, pairs, rows, stream, mr);
  return cudf::make_strings_column(
    rows, std::move(offsets), chars.release(), validity.null_count, std::move(validity.mask));
}

void validate_schema(std::span<const cudf::table_view> windows)
{
  if (windows.empty()) { throw std::invalid_argument("gather_kway: no windows"); }
  auto const& first = windows.front();
  for (cudf::size_type c = 0; c < first.num_columns(); ++c) {
    auto const type = first.column(c).type();
    if (!cudf::is_fixed_width(type) && type.id() != cudf::type_id::STRING) {
      throw std::invalid_argument("gather_kway: unsupported type in column " + std::to_string(c));
    }
  }
  for (std::size_t i = 1; i < windows.size(); ++i) {
    if (windows[i].num_columns() != first.num_columns()) {
      throw std::invalid_argument("gather_kway: window " + std::to_string(i) +
                                  " has a different column count");
    }
    for (cudf::size_type c = 0; c < first.num_columns(); ++c) {
      if (!cudf::have_same_types(first.column(c), windows[i].column(c))) {
        throw std::invalid_argument("gather_kway: window " + std::to_string(i) +
                                    " has a different type in column " + std::to_string(c));
      }
    }
  }
}

}  // namespace

std::unique_ptr<cudf::table> gather_kway(std::span<const cudf::table_view> windows,
                                         cudf::device_span<const std::int32_t> src_input,
                                         cudf::device_span<const cudf::size_type> src_row,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  validate_schema(windows);
  if (src_input.size() != src_row.size()) {
    throw std::invalid_argument("gather_kway: src_input and src_row differ in size");
  }
  if (src_input.size() > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("gather_kway: the map exceeds the column size limit");
  }
  auto const rows = static_cast<cudf::size_type>(src_input.size());
  if (rows == 0) { return cudf::empty_like(windows.front()); }

  gather_map const map{src_input.data(), src_row.data()};
  auto const num_columns = windows.front().num_columns();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(static_cast<std::size_t>(num_columns));
  for (cudf::size_type c = 0; c < num_columns; ++c) {
    columns.push_back(windows.front().column(c).type().id() == cudf::type_id::STRING
                        ? gather_strings(windows, c, map, rows, stream, mr)
                        : gather_fixed_width(windows, c, map, rows, stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op::merge::detail
