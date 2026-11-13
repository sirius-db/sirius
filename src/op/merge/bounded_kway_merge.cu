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

#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/detail/kway_merge_detail.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace sirius::op::merge {

namespace {

/// The rank of tile boundary t: t * tile_rows, capped at target (the last boundary).
struct tile_rank_fn {
  cudf::size_type tile_rows;
  cudf::size_type target;

  __device__ cudf::size_type operator()(cudf::size_type t) const noexcept
  {
    const auto rank = static_cast<std::int64_t>(t) * tile_rows;
    return rank < target ? static_cast<cudf::size_type>(rank) : target;
  }
};

void validate(std::span<const cudf::table_view> windows,
              const merge_key_spec& keys,
              std::span<const std::uint8_t> truncated,
              cudf::size_type max_rows)
{
  if (windows.empty()) {
    throw std::invalid_argument("bounded_kway_merge: at least one window is required");
  }
  if (truncated.size() != windows.size()) {
    throw std::invalid_argument("bounded_kway_merge: one truncated flag per window is required");
  }
  if (keys.key_cols.empty()) {
    throw std::invalid_argument("bounded_kway_merge: at least one key column is required");
  }
  if (keys.column_order.size() != keys.key_cols.size() ||
      keys.null_precedence.size() != keys.key_cols.size()) {
    throw std::invalid_argument(
      "bounded_kway_merge: column_order and null_precedence need one entry per key column");
  }
  if (max_rows < 0) { throw std::invalid_argument("bounded_kway_merge: max_rows is negative"); }

  const auto& first = windows.front();
  for (const auto key : keys.key_cols) {
    if (key < 0 || key >= first.num_columns()) {
      throw std::invalid_argument("bounded_kway_merge: key column " + std::to_string(key) +
                                  " is out of range");
    }
  }
  for (std::size_t i = 1; i < windows.size(); ++i) {
    if (windows[i].num_columns() != first.num_columns()) {
      throw std::invalid_argument("bounded_kway_merge: window " + std::to_string(i) +
                                  " differs in column count");
    }
    for (cudf::size_type c = 0; c < first.num_columns(); ++c) {
      if (windows[i].column(c).type() != first.column(c).type()) {
        throw std::invalid_argument("bounded_kway_merge: window " + std::to_string(i) +
                                    " differs in the type of column " + std::to_string(c));
      }
    }
  }
}

}  // namespace

bounded_merge_result bounded_kway_merge(std::span<const cudf::table_view> windows,
                                        const merge_key_spec& keys,
                                        std::span<const std::uint8_t> truncated,
                                        cudf::size_type max_rows,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  validate(windows, keys, truncated, max_rows);
  const auto k       = windows.size();
  const auto temp_mr = cudf::get_current_device_resource_ref();

  std::vector<cudf::table_view> key_windows;
  std::vector<cudf::size_type> host_lengths;
  key_windows.reserve(k);
  host_lengths.reserve(k);
  for (const auto& window : windows) {
    key_windows.push_back(window.select(keys.key_cols));
    host_lengths.push_back(window.num_rows());
  }
  // Nothing waits on the stream before safe_target: the comparator and the two arrays below are
  // filled by asynchronous copies.
  const auto cmp = detail::kway_key_comparator::create(
    key_windows, keys.column_order, keys.null_precedence, stream, temp_mr);
  const auto lengths     = cudf::detail::make_device_uvector_async(host_lengths, stream, temp_mr);
  const auto d_truncated = cudf::detail::make_device_uvector_async(
    cudf::host_span<const std::uint8_t>{truncated.data(), truncated.size()}, stream, temp_mr);

  bounded_merge_result result;
  // First sync: the target comes back to the host, it sizes everything below.
  const auto target = detail::safe_target(cmp, lengths, d_truncated, max_rows, stream);
  if (target == 0) {
    result.output = cudf::empty_like(windows.front());
    result.consumed.assign(k, 0);
    return result;
  }

  // Co-ranks at every tile boundary and, in the last row, at target.
  constexpr auto tile_rows = detail::kDefaultTileRows;
  const auto tiles         = static_cast<std::size_t>((target - 1) / tile_rows + 1);
  rmm::device_uvector<cudf::size_type> ranks(tiles + 1, stream, temp_mr);
  thrust::tabulate(
    rmm::exec_policy_nosync(stream), ranks.begin(), ranks.end(), tile_rank_fn{tile_rows, target});
  rmm::device_uvector<cudf::size_type> coranks((tiles + 1) * k, stream, temp_mr);
  detail::corank_split(cmp, lengths, ranks, coranks, stream);

  rmm::device_uvector<std::int32_t> out_input(static_cast<std::size_t>(target), stream, temp_mr);
  rmm::device_uvector<cudf::size_type> out_row(static_cast<std::size_t>(target), stream, temp_mr);
  detail::tile_merge(cmp, lengths, coranks, target, tile_rows, out_input, out_row, stream);

  // Second sync, inside the gather: consumed[i] = c_i(target) comes back with the gather's own
  // counts. The gather returns once the stream has run everything queued above, so the
  // comparator and the scratch buffers can go away.
  result.consumed.resize(k);
  result.output = detail::gather_kway(
    windows,
    out_input,
    out_row,
    stream,
    mr,
    detail::gather_read_back{
      cudf::device_span<const cudf::size_type>{coranks.data() + tiles * k, k}, result.consumed});
  return result;
}

}  // namespace sirius::op::merge
