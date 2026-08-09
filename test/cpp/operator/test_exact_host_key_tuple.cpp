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

#include "op/cudf_sort_order.hpp"
#include "op/dynamic_filter/exact_host_scalar.hpp"

#include <catch.hpp>

#include <cstdint>
#include <optional>
#include <vector>

using sirius::op::exact_host_key_tuple;
using sirius::op::exact_host_scalar;
using sirius::op::top_n_key_semantics;

namespace {

constexpr auto k_int32 = cudf::data_type{cudf::type_id::INT32};

exact_host_scalar int32_scalar(std::int32_t v) { return exact_host_scalar{v, k_int32}; }

top_n_key_semantics int32_key(cudf::order order, cudf::null_order null_order)
{
  return {.storage_type = k_int32, .order = order, .null_order = null_order};
}

exact_host_key_tuple int32_tuple(std::vector<std::optional<std::int32_t>> const& values)
{
  std::vector<std::optional<exact_host_scalar>> components;
  components.reserve(values.size());
  for (auto const& v : values) {
    components.push_back(v ? std::optional{int32_scalar(*v)} : std::nullopt);
  }
  return exact_host_key_tuple{std::move(components)};
}

}  // namespace

TEST_CASE("exact_host_scalar::compare honors direction per storage type",
          "[dynamic_filter][top_n]")
{
  auto const lo = exact_host_scalar{std::int64_t{-5}, cudf::data_type{cudf::type_id::INT64}};
  auto const hi = exact_host_scalar{std::int64_t{7}, cudf::data_type{cudf::type_id::INT64}};
  REQUIRE(lo.compare(hi, cudf::order::ASCENDING) < 0);
  REQUIRE(hi.compare(lo, cudf::order::ASCENDING) > 0);
  REQUIRE(lo.compare(lo, cudf::order::ASCENDING) == 0);
  // DESCENDING: the larger value orders first.
  REQUIRE(hi.compare(lo, cudf::order::DESCENDING) < 0);
  REQUIRE(lo.compare(hi, cudf::order::DESCENDING) > 0);

  auto const small = exact_host_scalar{std::int8_t{-128}, cudf::data_type{cudf::type_id::INT8}};
  auto const big   = exact_host_scalar{std::int8_t{127}, cudf::data_type{cudf::type_id::INT8}};
  REQUIRE(small.compare(big, cudf::order::ASCENDING) < 0);
  REQUIRE(big.compare(small, cudf::order::DESCENDING) < 0);
}

TEST_CASE("exact_host_key_tuple::lex_compare single component", "[dynamic_filter][top_n]")
{
  auto const one = int32_tuple({1});
  auto const two = int32_tuple({2});

  SECTION("ascending values")
  {
    std::vector<top_n_key_semantics> const keys{
      int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)};
    REQUIRE(one.lex_compare(two, keys) < 0);
    REQUIRE(two.lex_compare(one, keys) > 0);
    REQUIRE(one.lex_compare(one, keys) == 0);
  }
  SECTION("descending values")
  {
    std::vector<top_n_key_semantics> const keys{
      int32_key(cudf::order::DESCENDING, cudf::null_order::AFTER)};
    REQUIRE(two.lex_compare(one, keys) < 0);
    REQUIRE(one.lex_compare(two, keys) > 0);
  }
}

TEST_CASE("exact_host_key_tuple::lex_compare null placement mirrors cuDF",
          "[dynamic_filter][top_n]")
{
  auto const null_tuple  = int32_tuple({std::nullopt});
  auto const value_tuple = int32_tuple({5});

  SECTION("ascending: null_order applies directly")
  {
    std::vector<top_n_key_semantics> const before{
      int32_key(cudf::order::ASCENDING, cudf::null_order::BEFORE)};
    std::vector<top_n_key_semantics> const after{
      int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER)};
    REQUIRE(null_tuple.lex_compare(value_tuple, before) < 0);
    REQUIRE(value_tuple.lex_compare(null_tuple, before) > 0);
    REQUIRE(null_tuple.lex_compare(value_tuple, after) > 0);
    REQUIRE(null_tuple.lex_compare(null_tuple, after) == 0);
  }
  SECTION("descending: the per-key verdict reverses, nulls included")
  {
    // null_order::AFTER on a DESCENDING key places nulls first in the output order -- the same
    // contract to_cudf_null_order compensates for.
    std::vector<top_n_key_semantics> const after{
      int32_key(cudf::order::DESCENDING, cudf::null_order::AFTER)};
    std::vector<top_n_key_semantics> const before{
      int32_key(cudf::order::DESCENDING, cudf::null_order::BEFORE)};
    REQUIRE(null_tuple.lex_compare(value_tuple, after) < 0);
    REQUIRE(null_tuple.lex_compare(value_tuple, before) > 0);
  }
  SECTION("SQL placement derived through to_cudf_null_order")
  {
    // ORDER BY x DESC NULLS FIRST: nulls must order before every value.
    std::vector<top_n_key_semantics> const keys{int32_key(
      cudf::order::DESCENDING,
      sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                     duckdb::OrderByNullType::NULLS_FIRST))};
    REQUIRE(null_tuple.lex_compare(value_tuple, keys) < 0);
  }
}

TEST_CASE("exact_host_key_tuple::lex_compare multi-key", "[dynamic_filter][top_n]")
{
  std::vector<top_n_key_semantics> const keys{
    int32_key(cudf::order::ASCENDING, cudf::null_order::AFTER),
    int32_key(cudf::order::DESCENDING, cudf::null_order::BEFORE)};

  SECTION("the first differing key decides regardless of the tail")
  {
    REQUIRE(int32_tuple({1, 0}).lex_compare(int32_tuple({2, 100}), keys) < 0);
    REQUIRE(int32_tuple({3, 100}).lex_compare(int32_tuple({2, 0}), keys) > 0);
  }
  SECTION("equal heads fall through to the tail's direction")
  {
    // Tail is DESCENDING: the larger tail value orders first.
    REQUIRE(int32_tuple({1, 9}).lex_compare(int32_tuple({1, 3}), keys) < 0);
    REQUIRE(int32_tuple({1, 3}).lex_compare(int32_tuple({1, 9}), keys) > 0);
  }
  SECTION("null tail components order per the tail's null_order")
  {
    // Tail null_order BEFORE on DESCENDING places nulls last in the output order.
    REQUIRE(int32_tuple({1, std::nullopt}).lex_compare(int32_tuple({1, 3}), keys) > 0);
    REQUIRE(int32_tuple({1, 3}).lex_compare(int32_tuple({1, std::nullopt}), keys) < 0);
  }
  SECTION("fully equal tuples, null tails included")
  {
    REQUIRE(int32_tuple({1, std::nullopt}).lex_compare(int32_tuple({1, std::nullopt}), keys) == 0);
    REQUIRE(int32_tuple({4, 4}).lex_compare(int32_tuple({4, 4}), keys) == 0);
  }
}
