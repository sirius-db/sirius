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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda_runtime_api.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstring>
#include <limits>
#include <optional>
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

/// The byte size of a string_index_pair.
struct pair_size_fn {
  __device__ cudf::size_type operator()(cudf::strings::detail::string_index_pair const& pair) const
  {
    return pair.second;
  }
};

/// The output null mask of one column; an empty buffer when no window's column is nullable.
struct gathered_validity {
  rmm::device_buffer mask;
  rmm::device_uvector<cudf::bitmask_type const*> masks;
  rmm::device_uvector<cudf::size_type> mask_offsets;
  bool nullable{false};

  [[nodiscard]] source_valid_fn device_fn(gather_map map) const
  {
    return source_valid_fn{map, masks.data(), mask_offsets.data()};
  }
};

/// Builds the output null mask from the null mask bit of every source row (the null counts of the
/// windows are not read) and adds the number of valid rows to @p valid_count, a zeroed device
/// counter. Does not wait on the stream.
gathered_validity gather_validity(std::span<const cudf::table_view> windows,
                                  cudf::size_type column,
                                  gather_map map,
                                  cudf::size_type rows,
                                  cudf::size_type* valid_count,
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
                           cudf::detail::make_device_uvector_async(masks, stream, temp_mr),
                           cudf::detail::make_device_uvector_async(mask_offsets, stream, temp_mr),
                           nullable};
  if (nullable) {
    // cudf::detail::valid_if without its read-back of the count.
    result.mask = cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    constexpr cudf::size_type block_size{256};
    cudf::detail::grid_1d const grid{rows, block_size};
    cudf::detail::valid_if_kernel<block_size>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        static_cast<cudf::bitmask_type*>(result.mask.data()),
        cuda::counting_iterator<cudf::size_type>{0},
        rows,
        result.device_fn(map),
        valid_count);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
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

rmm::device_buffer gather_fixed_width(std::span<const cudf::table_view> windows,
                                      cudf::size_type column,
                                      gather_map map,
                                      cudf::size_type rows,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  switch (cudf::size_of(windows.front().column(column).type())) {
    case 1: return gather_fixed_values<std::uint8_t>(windows, column, map, rows, stream, mr);
    case 2: return gather_fixed_values<std::uint16_t>(windows, column, map, rows, stream, mr);
    case 4: return gather_fixed_values<std::uint32_t>(windows, column, map, rows, stream, mr);
    case 8: return gather_fixed_values<std::uint64_t>(windows, column, map, rows, stream, mr);
    case 16: return gather_fixed_values<unsigned __int128>(windows, column, map, rows, stream, mr);
    default:
      throw std::invalid_argument("gather_kway: unsupported fixed-width size in column " +
                                  std::to_string(column));
  }
}

/// One output column whose kernels are queued; its null count, and the chars size of a STRING
/// column, are still on the device.
struct pending_column {
  gathered_validity validity;
  rmm::device_buffer values;                                  ///< Fixed-width only.
  std::optional<rmm::device_uvector<string_source>> sources;  ///< STRING only.
  std::unique_ptr<cudf::column> offsets;                      ///< STRING only: int32 offsets.
};

/// The (pointer, size) pair of every output row of a STRING column.
auto string_pairs(pending_column const& pending, gather_map map)
{
  return cuda::transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    string_pair_fn{
      map, pending.sources->data(), pending.validity.nullable, pending.validity.device_fn(map)});
}

/// Queues the null mask and the int32 offsets of a STRING column; the last offset, the chars
/// size, also goes to the device slot @p chars_bytes with 64 bits. Does not wait on the stream.
pending_column start_strings(std::span<const cudf::table_view> windows,
                             cudf::size_type column,
                             gather_map map,
                             cudf::size_type rows,
                             cudf::size_type* valid_count,
                             std::int64_t* chars_bytes,
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
  pending_column pending{
    gather_validity(windows, column, map, rows, valid_count, stream, mr),
    rmm::device_buffer{},
    cudf::detail::make_device_uvector_async(sources, stream, temp_mr),
    cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, rows + 1, cudf::mask_state::UNALLOCATED, stream, mr)};

  // The first pass of cudf's make_offsets_child_column, without its read-back of the total: the
  // scan accumulates in 64 bits, stores 32-bit offsets and saves the exact total in the slot.
  // The input has rows + 1 values; the last one (0) is never added.
  auto const sizes = cuda::transform_iterator(string_pairs(pending, map), pair_size_fn{});
  auto const input = cudf::detail::make_counting_transform_iterator(
    0, cudf::strings::detail::string_offsets_fn{sizes, rows});
  auto* const d_offsets = pending.offsets->mutable_view().data<std::int32_t>();
  thrust::exclusive_scan(
    rmm::exec_policy_nosync(stream, temp_mr),
    input,
    input + rows + 1,
    cudf::detail::make_sizes_to_offsets_iterator(d_offsets, d_offsets + rows + 1, chars_bytes),
    std::int64_t{0});
  return pending;
}

/// Gathers the chars of a STRING column once its chars size is on the host.
std::unique_ptr<cudf::column> finish_strings(pending_column&& pending,
                                             gather_map map,
                                             cudf::size_type rows,
                                             cudf::size_type null_count,
                                             std::int64_t chars_bytes,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const pairs     = string_pairs(pending, map);
  auto const threshold = cudf::strings::get_offset64_threshold();
  CUDF_EXPECTS(cudf::strings::is_large_strings_enabled() || chars_bytes < threshold,
               "Size of output exceeds the column size limit",
               std::overflow_error);
  if (chars_bytes >= threshold) {
    // Past the large-strings threshold the offsets are int64: cudf builds them again (it waits
    // on the stream for the total).
    auto const sizes = cuda::transform_iterator(pairs, pair_size_fn{});
    std::tie(pending.offsets, chars_bytes) =
      cudf::strings::detail::make_offsets_child_column(sizes, sizes + rows, stream, mr);
  }
  auto chars = cudf::strings::detail::make_chars_buffer(
    pending.offsets->view(), chars_bytes, pairs, rows, stream, mr);
  return cudf::make_strings_column(rows,
                                   std::move(pending.offsets),
                                   chars.release(),
                                   null_count,
                                   std::move(pending.validity.mask));
}

/// Copies @p read_back.source to the host and waits for it.
void copy_to_host(gather_read_back const& read_back, rmm::cuda_stream_view stream)
{
  if (read_back.source.empty()) { return; }
  CUDF_CUDA_TRY(cudaMemcpyAsync(read_back.destination.data(),
                                read_back.source.data(),
                                read_back.source.size() * sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
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

// Every column's kernels are queued first; the null counts and the STRING chars sizes they
// leave on the device then come back in one copy, together with the caller's read-back, so the
// waits on the stream do not grow with the number of columns or windows.
std::unique_ptr<cudf::table> gather_kway(std::span<const cudf::table_view> windows,
                                         cudf::device_span<const std::int32_t> src_input,
                                         cudf::device_span<const cudf::size_type> src_row,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr,
                                         gather_read_back read_back)
{
  validate_schema(windows);
  if (src_input.size() != src_row.size()) {
    throw std::invalid_argument("gather_kway: src_input and src_row differ in size");
  }
  if (src_input.size() > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("gather_kway: the map exceeds the column size limit");
  }
  if (read_back.source.size() != read_back.destination.size()) {
    throw std::invalid_argument("gather_kway: the read-back source and destination differ in size");
  }
  auto const rows = static_cast<cudf::size_type>(src_input.size());
  if (rows == 0) {
    copy_to_host(read_back, stream);
    return cudf::empty_like(windows.front());
  }

  gather_map const map{src_input.data(), src_row.data()};
  auto const temp_mr     = cudf::get_current_device_resource_ref();
  auto const num_columns = static_cast<std::size_t>(windows.front().num_columns());
  auto const is_string   = [&](std::size_t c) {
    return windows.front().column(static_cast<cudf::size_type>(c)).type().id() ==
           cudf::type_id::STRING;
  };
  std::size_t num_strings = 0;
  for (std::size_t c = 0; c < num_columns; ++c) {
    num_strings += is_string(c) ? 1 : 0;
  }

  // What comes back from the device, as 64-bit words: the chars size of every STRING column,
  // then, as size_type, the valid rows of every column and the caller's read-back.
  auto const num_counts = num_columns + read_back.source.size();
  rmm::device_uvector<std::int64_t> totals(num_strings + (num_counts + 1) / 2, stream, temp_mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(totals.data(), 0, totals.size() * sizeof(std::int64_t), stream.value()));
  auto* const d_chars_bytes = totals.data();
  auto* const d_valid_count = reinterpret_cast<cudf::size_type*>(totals.data() + num_strings);
  if (!read_back.source.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_count + num_columns,
                                  read_back.source.data(),
                                  read_back.source.size() * sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToDevice,
                                  stream.value()));
  }

  std::vector<pending_column> pending;
  pending.reserve(num_columns);
  for (std::size_t c = 0, s = 0; c < num_columns; ++c) {
    auto const column = static_cast<cudf::size_type>(c);
    if (is_string(c)) {
      pending.push_back(start_strings(
        windows, column, map, rows, d_valid_count + c, d_chars_bytes + s++, stream, mr));
    } else {
      // The values before the validity, so a rejected width throws before anything else.
      auto values = gather_fixed_width(windows, column, map, rows, stream, mr);
      pending.push_back(
        pending_column{gather_validity(windows, column, map, rows, d_valid_count + c, stream, mr),
                       std::move(values),
                       std::nullopt,
                       nullptr});
    }
  }

  // The one wait every call makes.
  auto const host_totals = cudf::detail::make_pinned_vector_async(totals, stream);
  stream.synchronize();
  std::vector<cudf::size_type> counts(num_counts);
  std::memcpy(
    counts.data(), host_totals.data() + num_strings, num_counts * sizeof(cudf::size_type));
  if (!read_back.source.empty()) {
    std::memcpy(read_back.destination.data(),
                counts.data() + num_columns,
                read_back.source.size() * sizeof(cudf::size_type));
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(num_columns);
  for (std::size_t c = 0, s = 0; c < num_columns; ++c) {
    auto& column          = pending[c];
    auto const null_count = column.validity.nullable ? rows - counts[c] : 0;
    if (is_string(c)) {
      columns.push_back(
        finish_strings(std::move(column), map, rows, null_count, host_totals[s++], stream, mr));
    } else {
      columns.push_back(std::make_unique<cudf::column>(
        windows.front().column(static_cast<cudf::size_type>(c)).type(),
        rows,
        std::move(column.values),
        std::move(column.validity.mask),
        null_count));
    }
  }
  // The chars of the STRING columns are gathered after the read-back.
  if (num_strings > 0) { stream.synchronize(); }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op::merge::detail
