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

#include "catch.hpp"
#include "exec/exchange_direct.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using namespace sirius::exec;

namespace {

cudf::data_type type(cudf::type_id id) { return cudf::data_type{id}; }

direct_column strings(cudf::type_id offsets, std::uint64_t chars)
{
  return {type(cudf::type_id::STRING), 0, false, offsets, chars};
}

template <typename T>
void poke(std::vector<std::uint8_t>& bytes, std::size_t at, T value)
{
  std::memcpy(bytes.data() + at, &value, sizeof(T));
}

}  // namespace

TEST_CASE("decode_layout inverts encode_layout", "[exchange_direct]")
{
  direct_layout const layout{5,
                             {{type(cudf::type_id::INT64), 0, false},
                              {cudf::data_type{cudf::type_id::DECIMAL64, -2}, 2, true},
                              {type(cudf::type_id::BOOL8), 0, true},
                              {type(cudf::type_id::TIMESTAMP_DAYS), 5, true},
                              {type(cudf::type_id::STRING), 1, true, cudf::type_id::INT32, 17},
                              strings(cudf::type_id::INT64, 0)}};
  CHECK(decode_layout(encode_layout(layout)) == layout);
}

TEST_CASE("decode_layout rejects each malformed field", "[exchange_direct]")
{
  // Encoded as: header at 0 (magic, rows at 4, ncols at 8); the INT32 column at 12 (type, scale
  // at 16, null count at 20, mask flag at 24); the STRING column at 25 (offsets type at 38, chars
  // at 42).
  direct_layout const base{
    3, {{type(cudf::type_id::INT32), 1, true}, strings(cudf::type_id::INT32, 5)}};
  auto const valid = encode_layout(base);
  REQUIRE(decode_layout(valid) == base);

  using mutation = std::function<void(std::vector<std::uint8_t>&)>;
  std::vector<std::pair<std::string_view, mutation>> const cases{
    {"bad magic", [](auto& b) { b[3] = '2'; }},
    {"truncated", [](auto& b) { b.pop_back(); }},
    {"trailing bytes", [](auto& b) { b.push_back(0); }},
    {"row count", [](auto& b) { poke<std::int32_t>(b, 4, 0); }},
    {"row count", [](auto& b) { poke(b, 4, std::numeric_limits<std::int32_t>::max()); }},
    {"column count", [](auto& b) { poke<std::uint32_t>(b, 8, 0); }},
    {"column count", [](auto& b) { poke<std::uint32_t>(b, 8, 3); }},
    {"type id", [](auto& b) { poke<std::int32_t>(b, 12, -1); }},
    {"type id", [](auto& b) { poke(b, 12, cudf::type_id::NUM_TYPE_IDS); }},
    {"neither fixed-width nor STRING", [](auto& b) { poke(b, 12, cudf::type_id::EMPTY); }},
    {"neither fixed-width nor STRING", [](auto& b) { poke(b, 12, cudf::type_id::LIST); }},
    {"scale", [](auto& b) { poke<std::int32_t>(b, 16, 2); }},
    {"null count", [](auto& b) { poke<std::int32_t>(b, 20, 4); }},
    {"nulls without a mask", [](auto& b) { poke<std::uint8_t>(b, 24, 0); }},
    {"mask flag", [](auto& b) { poke<std::uint8_t>(b, 24, 2); }},
    {"not INT32 or INT64", [](auto& b) { poke(b, 38, cudf::type_id::INT16); }},
    {"chars overflow INT32", [](auto& b) { poke<std::uint64_t>(b, 42, std::uint64_t{1} << 31); }},
  };
  for (auto const& [error, mutate] : cases) {
    auto bytes = valid;
    mutate(bytes);
    CHECK_THROWS_MATCHES(decode_layout(bytes),
                         sirius::invalid_input_exception,
                         MessageMatches(ContainsSubstring(std::string{error})));
  }
}

TEST_CASE("plan_buffers sizes buffers in walk order within the limit", "[exchange_direct]")
{
  direct_layout const layout{100,
                             {{cudf::data_type{cudf::type_id::DECIMAL128, 3}, 1, true},
                              strings(cudf::type_id::INT64, 1000)}};
  // The mask sends 4 words into a 64-byte allocation; aligned to 256 bytes the four take 4096.
  CHECK(plan_buffers(layout, 4096) ==
        std::vector<direct_buffer>{{0, 16, 64}, {0, 1600, 1600}, {1, 1000, 1000}, {1, 808, 808}});
  CHECK_THROWS_AS(plan_buffers(layout, 4095), sirius::invalid_input_exception);

  // Sizes whose aligned sum wraps around size_t must not pass as small ones.
  auto const max = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS_AS(plan_buffers({1, {strings(cudf::type_id::INT64, max)}}, max),
                  sirius::invalid_input_exception);
  CHECK_THROWS_AS(
    plan_buffers(
      {1, {strings(cudf::type_id::INT64, max / 2 + 1), strings(cudf::type_id::INT64, max / 2 + 1)}},
      max),
    sirius::invalid_input_exception);
}

TEST_CASE("describe_table lists a table's buffers in plan order", "[exchange_direct]")
{
  auto const stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
    cudf::make_numeric_column(type(cudf::type_id::INT32), 3, cudf::mask_state::ALL_NULL));
  columns.push_back(
    cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}, 3));
  columns.push_back(cudf::make_column_from_scalar(cudf::string_scalar{std::string_view{"abc"}}, 3));
  cudf::table const table(std::move(columns));
  auto const view = table.view();

  auto const [layout, buffers] = describe_table(view, stream);
  CHECK(layout == direct_layout{3,
                                {{type(cudf::type_id::INT32), 3, true},
                                 {cudf::data_type{cudf::type_id::DECIMAL64, -2}, 0, false},
                                 strings(cudf::type_id::INT32, 9)}});
  CHECK(buffers == std::vector<void const*>{view.column(0).null_mask(),
                                            view.column(0).head(),
                                            view.column(1).head(),
                                            view.column(2).head(),
                                            view.column(2).child(0).head()});
  CHECK(plan_buffers(layout, std::numeric_limits<std::size_t>::max()).size() == buffers.size());

  cudf::column_view const empty{type(cudf::type_id::EMPTY), 3, nullptr, nullptr, 3};
  CHECK_THROWS_AS(describe_table(cudf::table_view{std::vector<cudf::column_view>{empty}}, stream),
                  sirius::invalid_input_exception);
}
