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
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/types.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
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

/// Rows of the key-family test builds.
inline constexpr std::size_t k_rows = 600;

/// Whether row @p row of a test build is valid: every fifth row is null.
inline bool valid_row(std::size_t row) { return row % 5 != 2; }

inline std::vector<bool> validity(std::size_t rows)
{
  std::vector<bool> valid(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    valid[row] = valid_row(row);
  }
  return valid;
}

inline std::size_t valid_rows(std::size_t rows)
{
  std::size_t count = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    count += valid_row(row) ? 1 : 0;
  }
  return count;
}

/// Distinct values that fit INT16, so every carrier below can hold them.
template <class T>
inline std::vector<T> values(std::size_t rows)
{
  std::vector<T> out(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    out[row] = static_cast<T>((static_cast<std::int64_t>(row) * 37) % 30'011 - 15'000);
  }
  return out;
}

/// A key family: the type the filter is built over, the build column at its carrier, and the same
/// keys as a column of the construction type for the column constructor. Each builder returns its
/// column with the work on its stream complete, as `make_column` does.
struct family {
  std::string name;
  cudf::data_type construction;
  std::function<std::unique_ptr<cudf::column>(::cuda::stream_ref)> carrier;
  std::function<std::unique_ptr<cudf::column>(::cuda::stream_ref)> oracle;
};

inline family numeric_family(std::string name,
                             cudf::data_type construction,
                             cudf::data_type carrier)
{
  auto const make = [](cudf::data_type type, ::cuda::stream_ref stream) {
    auto const valid = validity(k_rows);
    switch (cudf::size_of(type)) {
      case 2: return make_column(type, values<std::int16_t>(k_rows), stream, valid);
      case 4: return make_column(type, values<std::int32_t>(k_rows), stream, valid);
      case 8: return make_column(type, values<std::int64_t>(k_rows), stream, valid);
      default: return make_column(type, values<__int128_t>(k_rows), stream, valid);
    }
  };
  return {std::move(name),
          construction,
          [=](::cuda::stream_ref stream) { return make(carrier, stream); },
          [=](::cuda::stream_ref stream) { return make(construction, stream); }};
}

inline family string_family()
{
  auto const make = [](::cuda::stream_ref stream) {
    auto const integers = make_column(cudf::data_type{cudf::type_id::INT32},
                                      values<std::int32_t>(k_rows),
                                      stream,
                                      validity(k_rows));
    auto strings        = cudf::strings::from_integers(integers->view(), stream);
    stream.sync();
    return strings;
  };
  return {"STRING", cudf::data_type{cudf::type_id::STRING}, make, make};
}

inline std::vector<family> families()
{
  using cudf::type_id;
  return {
    numeric_family("INT32", cudf::data_type{type_id::INT32}, cudf::data_type{type_id::INT32}),
    numeric_family("INT64", cudf::data_type{type_id::INT64}, cudf::data_type{type_id::INT64}),
    numeric_family(
      "INT16 carrier of INT64", cudf::data_type{type_id::INT64}, cudf::data_type{type_id::INT16}),
    numeric_family("UINT16", cudf::data_type{type_id::UINT16}, cudf::data_type{type_id::UINT16}),
    numeric_family("INT16 carrier of DATE",
                   cudf::data_type{type_id::TIMESTAMP_DAYS},
                   cudf::data_type{type_id::INT16}),
    numeric_family("TIMESTAMP_MILLISECONDS",
                   cudf::data_type{type_id::TIMESTAMP_MILLISECONDS},
                   cudf::data_type{type_id::TIMESTAMP_MILLISECONDS}),
    numeric_family("DECIMAL32 carrier of DECIMAL64",
                   cudf::data_type{type_id::DECIMAL64, -2},
                   cudf::data_type{type_id::DECIMAL32, -2}),
    numeric_family("DECIMAL128",
                   cudf::data_type{type_id::DECIMAL128, -2},
                   cudf::data_type{type_id::DECIMAL128, -2}),
    string_family()};
}

}  // namespace sirius::test::columns
