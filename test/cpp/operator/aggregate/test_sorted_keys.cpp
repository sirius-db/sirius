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

#include "op/aggregate/aggregate_op_util.hpp"

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <catch.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace {

using sirius::op::keys_ascending;

template <typename T>
std::unique_ptr<cudf::column> make_column(cudf::data_type type,
                                          std::vector<T> const& values,
                                          std::optional<cudf::size_type> null_row = std::nullopt)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  rmm::device_buffer data(values.data(), values.size() * sizeof(T), stream, mr);
  auto const size = static_cast<cudf::size_type>(values.size());
  if (!null_row) {
    return std::make_unique<cudf::column>(type, size, std::move(data), rmm::device_buffer{}, 0);
  }
  auto mask = cudf::create_null_mask(size, cudf::mask_state::ALL_VALID, stream, mr);
  cudf::set_null_mask(
    static_cast<cudf::bitmask_type*>(mask.data()), *null_row, *null_row + 1, false);
  return std::make_unique<cudf::column>(type, size, std::move(data), std::move(mask), 1);
}

/// keys_ascending() on @p column, checked against cuDF's own answer.
bool check(cudf::column_view const& column)
{
  auto const stream  = cudf::get_default_stream();
  bool const typed   = keys_ascending(column, stream);
  bool const generic = cudf::is_sorted(
    cudf::table_view({column}), {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  REQUIRE(typed == generic);
  return typed;
}

/// An ascending and a descending column of one type: keys_ascending() must tell them apart.
template <typename T>
void check_pair(cudf::type_id id,
                std::vector<T> const& ascending,
                std::vector<T> const& descending,
                int scale = 0)
{
  CHECK(check(*make_column<T>(cudf::data_type{id, scale}, ascending)));
  CHECK_FALSE(check(*make_column<T>(cudf::data_type{id, scale}, descending)));
}

}  // namespace

TEST_CASE("keys_ascending - fixed-width keys", "[aggregate][sorted_keys]")
{
  using cudf::type_id;
  __int128_t const big = static_cast<__int128_t>(1) << 100;
  check_pair<int64_t>(type_id::INT64, {-5, -5, 0, 1, 1, 9}, {2, 1, 3});
  check_pair<int32_t>(type_id::INT32, {-2, -1, 0, 1}, {0, -1});
  check_pair<uint32_t>(type_id::UINT32, {1, 2, 0x80000000u, 0xffffffffu}, {0x80000000u, 1});
  check_pair<int64_t>(type_id::DECIMAL64, {-100, 0, 0, 250}, {250, -100}, -2);
  check_pair<__int128_t>(type_id::DECIMAL128, {-big, 0, big}, {big, -big}, -2);
  check_pair<int32_t>(type_id::TIMESTAMP_DAYS, {-1, 0, 10000}, {10000, 0});

  auto const int64 = cudf::data_type{type_id::INT64};
  CHECK(check(*make_column<int64_t>(int64, {})));
  CHECK(check(*make_column<int64_t>(int64, {7})));

  auto const sliced = make_column<int64_t>(int64, {9, 1, 2, 3, 0});
  CHECK(check(cudf::slice(sliced->view(), {1, 4})[0]));
  CHECK_FALSE(check(cudf::slice(sliced->view(), {0, 3})[0]));

  // A descent in the last pair, past one grid stride.
  std::vector<int64_t> values(3'000'000);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int64_t>(i / 4);
  }
  CHECK(check(*make_column<int64_t>(int64, values)));
  values.back() = -1;
  CHECK_FALSE(check(*make_column<int64_t>(int64, values)));
}

TEST_CASE("keys_ascending - keys checked by cuDF", "[aggregate][sorted_keys]")
{
  auto const int64 = cudf::data_type{cudf::type_id::INT64};
  CHECK(check(*make_column<int64_t>(int64, {1, 2, 0}, 2)));
  CHECK_FALSE(check(*make_column<int64_t>(int64, {1, 0, 2}, 1)));

  auto const float64 = cudf::data_type{cudf::type_id::FLOAT64};
  CHECK(check(*make_column<double>(float64, {-1.5, 0.0, 2.0, std::nan("")})));
  CHECK_FALSE(check(*make_column<double>(float64, {std::nan(""), 0.0})));
}
