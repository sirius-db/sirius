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

/**
 * @file dynamic_filter_column_test_utils.hpp
 * @brief Builds small fixed-width cuDF columns from host values, with chosen null rows, and reads
 * them back.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::test::columns {

/// A column of @p type holding @p values bit-for-bit, with row `i` null iff `valid[i]` is false.
template <class T>
inline std::unique_ptr<cudf::column> make_column(cudf::data_type type,
                                                 std::vector<T> const& values,
                                                 ::cuda::stream_ref stream,
                                                 std::vector<bool> const& valid = {})
{
  auto column = cudf::make_fixed_width_column(
    type, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream);
  REQUIRE(cudaMemcpyAsync(column->mutable_view().head(),
                          values.data(),
                          values.size() * sizeof(T),
                          cudaMemcpyHostToDevice,
                          stream.get()) == cudaSuccess);
  if (!valid.empty()) {
    auto mask = cudf::create_null_mask(column->size(), cudf::mask_state::ALL_VALID, stream);
    cudf::size_type nulls = 0;
    for (std::size_t row = 0; row < valid.size(); ++row) {
      if (valid[row]) { continue; }
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()),
                          static_cast<cudf::size_type>(row),
                          static_cast<cudf::size_type>(row + 1),
                          false,
                          stream);
      ++nulls;
    }
    column->set_null_mask(std::move(mask), nulls);
  }
  stream.sync();
  return column;
}

/// The rows of @p column, which stores elements of @p T, copied to the host.
template <class T>
inline std::vector<T> to_host(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  std::vector<T> values(static_cast<std::size_t>(column.size()));
  REQUIRE(cudaMemcpyAsync(values.data(),
                          column.head<std::byte>() + column.offset() * sizeof(T),
                          values.size() * sizeof(T),
                          cudaMemcpyDeviceToHost,
                          stream.get()) == cudaSuccess);
  stream.sync();
  return values;
}

}  // namespace sirius::test::columns
