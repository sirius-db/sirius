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

#include "op/merge/detail/kway_comparator.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::op::merge::detail {

namespace {

static_assert(std::is_trivially_copyable_v<cudf::column_device_view>);
static_assert(std::is_trivially_copyable_v<kway_key_device_view>);

bool is_supported_key_type(cudf::data_type type)
{
  return cudf::is_fixed_width(type) || type.id() == cudf::type_id::STRING;
}

}  // namespace

struct kway_key_comparator::impl {
  /// Pinned host image of `views`. The copy to the device is asynchronous and may read it when
  /// the stream reaches the copy, so it lives as long as the device state.
  cudf::detail::host_vector<std::int8_t> staging;
  /// The column views of every window (window-major) followed by the views of their children.
  rmm::device_buffer views;
  const cudf::column_device_view* columns;  ///< Start of the column views inside `views`.
  rmm::device_uvector<cudf::order> column_order;
  rmm::device_uvector<cudf::null_order> null_precedence;
  int32_t num_inputs;
  cudf::size_type num_columns;
};

// Nothing here waits on the stream: the column views of all windows are laid out in one pinned
// host buffer and go to the device in one asynchronous copy, whatever the number of windows.
kway_key_comparator kway_key_comparator::create(std::span<const cudf::table_view> key_windows,
                                                std::span<const cudf::order> column_order,
                                                std::span<const cudf::null_order> null_precedence,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  if (key_windows.empty()) {
    throw std::invalid_argument("kway_key_comparator: at least one key window is required");
  }
  const auto num_columns = key_windows.front().num_columns();
  if (num_columns == 0) {
    throw std::invalid_argument("kway_key_comparator: at least one key column is required");
  }
  if (column_order.size() != static_cast<std::size_t>(num_columns) ||
      null_precedence.size() != static_cast<std::size_t>(num_columns)) {
    throw std::invalid_argument(
      "kway_key_comparator: column_order and null_precedence need one entry per key column");
  }
  for (cudf::size_type c = 0; c < num_columns; ++c) {
    const auto type = key_windows.front().column(c).type();
    if (!is_supported_key_type(type)) {
      throw std::invalid_argument("kway_key_comparator: unsupported type of key column " +
                                  std::to_string(c));
    }
  }
  for (const auto& window : key_windows) {
    if (window.num_columns() != num_columns) {
      throw std::invalid_argument("kway_key_comparator: key windows differ in column count");
    }
    for (cudf::size_type c = 0; c < num_columns; ++c) {
      if (window.column(c).type() != key_windows.front().column(c).type()) {
        throw std::invalid_argument(
          "kway_key_comparator: key windows differ in the type of column " + std::to_string(c));
      }
    }
  }

  // Column c of window i is entry i * num_columns + c.
  std::vector<cudf::column_view> columns;
  columns.reserve(key_windows.size() * static_cast<std::size_t>(num_columns));
  std::size_t view_bytes = 0;
  for (const auto& window : key_windows) {
    for (const auto& column : window) {
      view_bytes += cudf::column_device_view::extent(column);
      columns.push_back(column);
    }
  }

  // Both buffers get room to start the views on an aligned address; the host image and the
  // device copy have the same layout from there on.
  const std::size_t padded_bytes = view_bytes + alignof(cudf::column_device_view);
  auto staging = cudf::detail::make_pinned_vector_async<std::int8_t>(padded_bytes, stream);
  rmm::device_buffer views(padded_bytes, stream, mr);
  auto* const h_begin = cudf::detail::align_ptr_for_type<cudf::column_device_view>(staging.data());
  auto* const d_begin = cudf::detail::align_ptr_for_type<cudf::column_device_view>(views.data());
  // Places the column views at h_begin and their children's views behind them, with the child
  // pointers already relative to d_begin.
  const auto* const d_columns =
    cudf::detail::child_columns_to_device_array<cudf::column_device_view>(
      columns.begin(), columns.end(), h_begin, d_begin);
  cudf::detail::cuda_memcpy_async<std::int8_t>(
    cudf::device_span<std::int8_t>{reinterpret_cast<std::int8_t*>(d_begin), view_bytes},
    cudf::host_span<const std::int8_t>{reinterpret_cast<const std::int8_t*>(h_begin),
                                       view_bytes,
                                       staging.get_allocator().is_device_accessible()},
    stream);

  auto state = std::make_unique<impl>(impl{
    std::move(staging),
    std::move(views),
    d_columns,
    cudf::detail::make_device_uvector_async(
      cudf::host_span<const cudf::order>{column_order.data(), column_order.size()}, stream, mr),
    cudf::detail::make_device_uvector_async(
      cudf::host_span<const cudf::null_order>{null_precedence.data(), null_precedence.size()},
      stream,
      mr),
    static_cast<int32_t>(key_windows.size()),
    num_columns});
  return kway_key_comparator(std::move(state));
}

kway_key_comparator::kway_key_comparator(std::unique_ptr<impl> state) : _impl(std::move(state)) {}

kway_key_comparator::kway_key_comparator(kway_key_comparator&&) noexcept            = default;
kway_key_comparator& kway_key_comparator::operator=(kway_key_comparator&&) noexcept = default;
kway_key_comparator::~kway_key_comparator()                                         = default;

int32_t kway_key_comparator::num_inputs() const noexcept { return _impl->num_inputs; }

kway_key_device_view device_view_of(const kway_key_comparator& cmp)
{
  const auto& state = *cmp._impl;
  return kway_key_device_view{state.columns,
                              state.column_order.data(),
                              state.null_precedence.data(),
                              state.num_inputs,
                              state.num_columns};
}

void compare_pairs_for_test(const kway_key_comparator& cmp,
                            cudf::device_span<const int32_t> ia,
                            cudf::device_span<const cudf::size_type> ra,
                            cudf::device_span<const int32_t> ib,
                            cudf::device_span<const cudf::size_type> rb,
                            cudf::device_span<uint8_t> out,
                            rmm::cuda_stream_view stream)
{
  if (ia.size() != out.size() || ra.size() != out.size() || ib.size() != out.size() ||
      rb.size() != out.size()) {
    throw std::invalid_argument("compare_pairs_for_test: spans differ in size");
  }
  const auto view = device_view_of(cmp);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::counting_iterator<std::size_t>(0),
    thrust::counting_iterator<std::size_t>(out.size()),
    out.begin(),
    [view, ia = ia.data(), ra = ra.data(), ib = ib.data(), rb = rb.data()] __device__(
      std::size_t i) -> uint8_t { return view.less(ia[i], ra[i], ib[i], rb[i]) ? 1 : 0; });
}

}  // namespace sirius::op::merge::detail
