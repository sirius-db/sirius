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

#include "op/scan/owning_table_view.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

void owning_table_view::ensure_view() const
{
  if (auto* table = std::get_if<std::unique_ptr<cudf::table>>(&_state)) {
    // Compute the view before moving the table into the type-erased owner.
    auto base = (*table)->view();
    _state    = std::make_unique<detail::my_view>(std::move(*table), base);
  }
}

cudf::table_view owning_table_view::view() const
{
  if (auto* table = std::get_if<std::unique_ptr<cudf::table>>(&_state)) { return (*table)->view(); }
  if (auto* view = std::get_if<std::unique_ptr<detail::my_view>>(&_state)) {
    return (*view)->view();
  }
  return cudf::table_view{};
}

std::size_t owning_table_view::n_columns() const
{
  if (auto* table = std::get_if<std::unique_ptr<cudf::table>>(&_state)) {
    return static_cast<std::size_t>((*table)->num_columns());
  }
  if (auto* view = std::get_if<std::unique_ptr<detail::my_view>>(&_state)) {
    return (*view)->n_columns();
  }
  return 0;
}

cudf::size_type owning_table_view::num_rows() const
{
  if (auto* table = std::get_if<std::unique_ptr<cudf::table>>(&_state)) {
    return (*table)->num_rows();
  }
  if (auto* view = std::get_if<std::unique_ptr<detail::my_view>>(&_state)) {
    return (*view)->num_rows();
  }
  return 0;
}

cudf::column_view owning_table_view::column(std::size_t index) const
{
  auto current = view();
  if (index >= static_cast<std::size_t>(current.num_columns())) {
    throw std::out_of_range("owning_table_view::column: index " + std::to_string(index) +
                            " is out of range (n_columns=" + std::to_string(current.num_columns()) +
                            ")");
  }
  return current.column(static_cast<cudf::size_type>(index));
}

std::vector<cudf::data_type> owning_table_view::column_types() const
{
  auto current = view();
  std::vector<cudf::data_type> types;
  types.reserve(static_cast<std::size_t>(current.num_columns()));
  for (cudf::size_type i = 0; i < current.num_columns(); ++i) {
    types.push_back(current.column(i).type());
  }
  return types;
}

void owning_table_view::reorder_columns(
  std::span<const std::pair<std::size_t, std::size_t>> swaps) const
{
  if (std::holds_alternative<std::monostate>(_state)) { return; }
  ensure_view();
  std::get<std::unique_ptr<detail::my_view>>(_state)->reorder_columns(swaps);
}

void owning_table_view::drop_columns(std::span<const std::size_t> positions) const
{
  if (std::holds_alternative<std::monostate>(_state)) { return; }
  ensure_view();
  std::get<std::unique_ptr<detail::my_view>>(_state)->drop_columns(positions);
}

void owning_table_view::select_columns(std::span<const std::size_t> positions) const
{
  if (std::holds_alternative<std::monostate>(_state)) { return; }
  ensure_view();
  std::get<std::unique_ptr<detail::my_view>>(_state)->select_columns(positions);
}

void owning_table_view::materialize(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (auto* view = std::get_if<std::unique_ptr<detail::my_view>>(&_state)) {
    _state = (*view)->materialize(stream, mr);
  }
}

std::unique_ptr<cudf::table> owning_table_view::release(rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  materialize(stream, mr);
  std::unique_ptr<cudf::table> out;
  if (auto* table = std::get_if<std::unique_ptr<cudf::table>>(&_state)) { out = std::move(*table); }
  _state = std::monostate{};
  return out;
}

void owning_table_view::drop() { _state = std::monostate{}; }

}  // namespace sirius::op::scan
