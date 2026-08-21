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

// Lowering a scan's constant comparisons into the DECODED integer domain — the
// values a decoder reconstructs, where a DATE is its stored day count and a
// DECIMAL is an unscaled integer at the COLUMN's scale.
//
// Every case here is a way to get the arithmetic subtly wrong while still
// producing a plausible range, and a wrong range silently drops rows during
// decode: an off-by-one on a strict inequality, rounding a decimal constant the
// wrong way when the column cannot represent it, or treating an empty range as
// unbounded.

#include "op/scan/scan_filter_analysis.hpp"
#include "op/scan/scan_utils.hpp"

#include <catch.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <helper/type_conversions.hpp>

#include <cstdint>
#include <limits>
#include <utility>

using namespace sirius::op;

namespace {

struct filter_fixture {
  duckdb::TableFilterSet filters;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<sirius::logical_type> returned_types;

  /// One filtered column at primary index 0, of @p type.
  explicit filter_fixture(duckdb::LogicalType const& type)
  {
    column_ids.emplace_back(0);
    returned_types.push_back(sirius::from_duckdb(type));
  }

  void push(duckdb::ExpressionType comparison, duckdb::Value constant)
  {
    filters.PushFilter(duckdb::ColumnIndex(0),
                       duckdb::make_uniq<duckdb::ConstantFilter>(comparison, std::move(constant)));
  }

  [[nodiscard]] scan_filter_analysis analyze() const
  {
    return analyze_scan_filters(filters, column_ids, returned_types);
  }

  /// The range extracted for the single column.
  [[nodiscard]] sirius::decode_range range() const
  {
    auto const result = analyze();
    REQUIRE(result.ranges.count(0) == 1);
    return result.ranges.at(0);
  }
};

}  // namespace

TEST_CASE("scan filter ranges: integer comparisons", "[scan]")
{
  SECTION("equality is a degenerate range")
  {
    filter_fixture f{duckdb::LogicalType::BIGINT};
    f.push(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value::BIGINT(42));
    auto const r = f.range();
    REQUIRE(r.lo == 42);
    REQUIRE(r.hi == 42);
  }

  SECTION("non-strict bounds keep the endpoint")
  {
    filter_fixture f{duckdb::LogicalType::INTEGER};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(10));
    f.push(duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO, duckdb::Value::INTEGER(20));
    auto const r = f.range();
    REQUIRE(r.lo == 10);
    REQUIRE(r.hi == 20);
  }

  SECTION("strict bounds tighten by one — the range is inclusive both ends")
  {
    filter_fixture f{duckdb::LogicalType::INTEGER};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::INTEGER(10));
    f.push(duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::INTEGER(20));
    auto const r = f.range();
    REQUIRE(r.lo == 11);
    REQUIRE(r.hi == 19);
  }

  SECTION("contradictory conjuncts are a provably empty range, not an unbounded one")
  {
    filter_fixture f{duckdb::LogicalType::INTEGER};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::INTEGER(20));
    f.push(duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::INTEGER(10));
    auto const r = f.range();
    REQUIRE(r.lo > r.hi);
  }
}

TEST_CASE("scan filter ranges: DATE lowers to its stored day count", "[scan]")
{
  filter_fixture f{duckdb::LogicalType::DATE};
  // 1994-01-01 is 8766 days after the 1970 epoch.
  f.push(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
         duckdb::Value::DATE(duckdb::Date::FromDate(1994, 1, 1)));
  auto const r = f.range();
  REQUIRE(r.lo == 8766);
  REQUIRE(r.hi == std::numeric_limits<std::int64_t>::max());
}

TEST_CASE("scan filter ranges: DECIMAL restates the constant at the column's scale", "[scan]")
{
  SECTION("column carries the constant's fractional digits exactly")
  {
    // DECIMAL(10,2) column, constant 12.34 -> unscaled 1234 at scale 2.
    filter_fixture f{duckdb::LogicalType::DECIMAL(10, 2)};
    f.push(duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO,
           duckdb::Value::DECIMAL(static_cast<int64_t>(1234), 10, 2));
    auto const r = f.range();
    REQUIRE(r.hi == 1234);
  }

  SECTION("an integral constant is scale 0 and scales up")
  {
    filter_fixture f{duckdb::LogicalType::DECIMAL(10, 2)};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(7));
    auto const r = f.range();
    REQUIRE(r.lo == 700);
  }

  SECTION("a constant finer than the column rounds AWAY from the surviving rows")
  {
    // DECIMAL(10,1) column cannot represent 1.25. For `col <= 1.25` the largest
    // representable value that satisfies it is 1.2, so the bound must FLOOR;
    // taking the ceil would admit 1.3, a row the filter rejects.
    filter_fixture f{duckdb::LogicalType::DECIMAL(10, 1)};
    f.push(duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO,
           duckdb::Value::DECIMAL(static_cast<int64_t>(125), 10, 2));
    REQUIRE(f.range().hi == 12);
  }

  SECTION("and the other direction rounds the other way")
  {
    // For `col >= 1.25` the smallest representable value that satisfies it is
    // 1.3, so the bound must CEIL.
    filter_fixture f{duckdb::LogicalType::DECIMAL(10, 1)};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
           duckdb::Value::DECIMAL(static_cast<int64_t>(125), 10, 2));
    REQUIRE(f.range().lo == 13);
  }

  SECTION("an equality the column's scale cannot represent selects nothing")
  {
    // No DECIMAL(10,1) value equals 1.25, so ceil(1.25) > floor(1.25) and the
    // range is provably empty rather than the spuriously-matching 1.2 or 1.3.
    filter_fixture f{duckdb::LogicalType::DECIMAL(10, 1)};
    f.push(duckdb::ExpressionType::COMPARE_EQUAL,
           duckdb::Value::DECIMAL(static_cast<int64_t>(125), 10, 2));
    auto const r = f.range();
    REQUIRE(r.lo > r.hi);
  }
}

TEST_CASE("scan filter ranges: coverage reflects what could not be converted", "[scan]")
{
  SECTION("a convertible filter set covers the whole filter")
  {
    filter_fixture f{duckdb::LogicalType::BIGINT};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::BIGINT(1));
    REQUIRE(f.analyze().ranges_cover_whole_filter);
  }

  SECTION("an unconvertible comparison keeps the range but clears coverage")
  {
    // != is not a range. The other conjunct still yields bounds — they remain a
    // sound over-approximation — but the scan must still evaluate the filter.
    filter_fixture f{duckdb::LogicalType::BIGINT};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::BIGINT(1));
    f.push(duckdb::ExpressionType::COMPARE_NOTEQUAL, duckdb::Value::BIGINT(5));
    auto const result = f.analyze();
    REQUIRE_FALSE(result.ranges_cover_whole_filter);
    REQUIRE(result.ranges.at(0).lo == 2);
  }

  SECTION("a float column yields no range at all")
  {
    filter_fixture f{duckdb::LogicalType::DOUBLE};
    f.push(duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::DOUBLE(1.5));
    auto const result = f.analyze();
    REQUIRE(result.ranges.empty());
    REQUIRE_FALSE(result.ranges_cover_whole_filter);
  }
}

// The column bookkeeping every walk over a filter set repeats. Two walks share
// it with deliberately different reactions to the third outcome, which is the
// part worth pinning: a conjunct that must be EVALUATED cannot reference an
// unmaterialized column, while one used only to PRUNE can be dropped.
TEST_CASE("resolve_filtered_column: the three outcomes", "[scan]")
{
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(7);  // filter column 0 -> primary index 7
  std::vector<std::optional<std::size_t>> batch_positions{std::optional<std::size_t>{3}};

  SECTION("usable: resolved to its batch position")
  {
    auto const r = resolve_filtered_column(0, column_ids, batch_positions, {});
    REQUIRE(r.status == filter_column_status::usable);
    REQUIRE(r.primary_index == 7);
    REQUIRE(r.batch_position == 3);
  }

  SECTION("skipped: a hive partition is enforced at the file-list level")
  {
    auto const r = resolve_filtered_column(0, column_ids, batch_positions, {7});
    REQUIRE(r.status == filter_column_status::skipped);
  }

  SECTION("skipped: a filter naming no column of the scan")
  {
    auto const r = resolve_filtered_column(9, column_ids, batch_positions, {});
    REQUIRE(r.status == filter_column_status::skipped);
  }

  SECTION("not_in_batch: the scan's column, but never materialized")
  {
    std::vector<std::optional<std::size_t>> absent{std::nullopt};
    auto const r = resolve_filtered_column(0, column_ids, absent, {});
    REQUIRE(r.status == filter_column_status::not_in_batch);
    // The primary index still resolves — a caller that fails on this reports
    // WHICH column was missing.
    REQUIRE(r.primary_index == 7);
  }

  SECTION("not_in_batch: a map shorter than the filter's column index")
  {
    std::vector<std::optional<std::size_t>> empty;
    auto const r = resolve_filtered_column(0, column_ids, empty, {});
    REQUIRE(r.status == filter_column_status::not_in_batch);
  }
}

TEST_CASE("decompose_table_filters fails loudly on an unmaterialized column", "[scan]")
{
  filter_fixture f{duckdb::LogicalType::BIGINT};
  f.push(duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::BIGINT(1));
  // The conjunct has to be evaluated, so a column that is not in the batch is a
  // wiring bug, not a shape to skip: silently dropping it would return rows the
  // filter rejects.
  std::vector<std::optional<std::size_t>> absent{std::nullopt};
  REQUIRE_THROWS(decompose_table_filters(f.filters, f.column_ids, f.returned_types, absent, {}));
}
