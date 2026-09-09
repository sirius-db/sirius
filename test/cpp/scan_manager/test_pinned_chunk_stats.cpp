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

// Host-only tests for the pinned zone-map filter classifier and prune check.
//
// The classifier is the release-mode safety line for cached-chunk pruning:
// ConstantFilter::CheckStatistics only D_ASSERTs the constant-vs-stats type
// match, so in release builds a mismatched probe silently misreads the stats
// value union instead of failing. These tests therefore matter MOST in
// release mode (run make test, not only test_debug): they pin down that no
// type-mismatched or unsupported filter shape ever reaches CheckStatistics.
// Conversely, passing under debug proves the classifier gates mismatches
// BEFORE the D_ASSERT would abort.
//
// compute_pinned_chunk_stats (the GPU half) is covered separately with the
// pin-capture tests; null/absent stats handling lives in the serve-plan tests.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/helper.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/dynamic_filter.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <duckdb/planner/filter/null_filter.hpp>
#include <duckdb/planner/filter/optional_filter.hpp>
#include <duckdb/planner/filter/struct_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/numeric_stats.hpp>
#include <scan_manager/pinned_chunk_stats.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

using duckdb::ExpressionType;
using duckdb::LogicalType;
using duckdb::Value;
using sirius::scan_manager::chunk_provably_empty;
using sirius::scan_manager::filter_safe_for_stats;
using sirius::scan_manager::pinned_zone_maps;

namespace {

// ============================================================================
// Builders. Filters are built exactly as DuckDB's binder would (valid
// invariants); tests for the classifier's defensive paths mutate the public
// fields afterwards, since the constructors enforce those invariants.
// ============================================================================

using filter_ptr = duckdb::unique_ptr<duckdb::TableFilter>;

template <class FILTER, class... ARGS>
filter_ptr make_filter(ARGS&&... args)
{
  filter_ptr filter = duckdb::make_uniq<FILTER>(std::forward<ARGS>(args)...);
  return filter;
}

filter_ptr cmp(ExpressionType comparison, Value constant)
{
  return make_filter<duckdb::ConstantFilter>(comparison, std::move(constant));
}

filter_ptr in_list(duckdb::vector<Value> values)
{
  return make_filter<duckdb::InFilter>(std::move(values));
}

filter_ptr and_of(filter_ptr a, filter_ptr b)
{
  auto conj = duckdb::make_uniq<duckdb::ConjunctionAndFilter>();
  conj->child_filters.push_back(std::move(a));
  conj->child_filters.push_back(std::move(b));
  return filter_ptr{std::move(conj)};
}

filter_ptr or_of(filter_ptr a, filter_ptr b)
{
  auto conj = duckdb::make_uniq<duckdb::ConjunctionOrFilter>();
  conj->child_filters.push_back(std::move(a));
  conj->child_filters.push_back(std::move(b));
  return filter_ptr{std::move(conj)};
}

filter_ptr optional_of(filter_ptr child)
{
  return make_filter<duckdb::OptionalFilter>(std::move(child));
}

filter_ptr dynamic_placeholder() { return make_filter<duckdb::DynamicFilter>(); }

/// Stats exactly as compute_pinned_chunk_stats builds them: CreateUnknown +
/// bounds + exact chunk-level null flags.
duckdb::BaseStatistics make_stats(LogicalType const& type,
                                  Value const& min,
                                  Value const& max,
                                  bool has_null = false)
{
  auto stats = duckdb::NumericStats::CreateUnknown(type);
  duckdb::NumericStats::SetMin(stats, min);
  duckdb::NumericStats::SetMax(stats, max);
  if (has_null) {
    stats.SetHasNull();
  } else {
    stats.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
  }
  return stats;
}

}  // namespace

// ============================================================================
// filter_safe_for_stats — allowed shapes
// ============================================================================

TEST_CASE("pinned_chunk_stats - classifier accepts allowed static shapes", "[pinned_chunk_stats]")
{
  auto const type = LogicalType::INTEGER;

  for (auto comparison : {ExpressionType::COMPARE_EQUAL,
                          ExpressionType::COMPARE_NOTEQUAL,
                          ExpressionType::COMPARE_LESSTHAN,
                          ExpressionType::COMPARE_LESSTHANOREQUALTO,
                          ExpressionType::COMPARE_GREATERTHAN,
                          ExpressionType::COMPARE_GREATERTHANOREQUALTO}) {
    REQUIRE(filter_safe_for_stats(*cmp(comparison, Value::INTEGER(42)), type));
  }

  REQUIRE(filter_safe_for_stats(*in_list({Value::INTEGER(1), Value::INTEGER(2)}), type));
  REQUIRE(filter_safe_for_stats(duckdb::IsNullFilter{}, type));
  REQUIRE(filter_safe_for_stats(duckdb::IsNotNullFilter{}, type));

  auto conjunctions = and_of(cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(1)),
                             cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(10)));
  REQUIRE(filter_safe_for_stats(*conjunctions, type));

  auto nested = optional_of(
    or_of(make_filter<duckdb::IsNullFilter>(),
          and_of(cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(5)),
                 cmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(0)))));
  REQUIRE(filter_safe_for_stats(*nested, type));
}

// ============================================================================
// filter_safe_for_stats — dynamic filters reject at any depth
// ============================================================================

TEST_CASE("pinned_chunk_stats - classifier rejects dynamic filters at any depth",
          "[pinned_chunk_stats]")
{
  auto const type = LogicalType::INTEGER;

  REQUIRE_FALSE(filter_safe_for_stats(*dynamic_placeholder(), type));
  REQUIRE_FALSE(filter_safe_for_stats(*optional_of(dynamic_placeholder()), type));
  REQUIRE_FALSE(filter_safe_for_stats(
    *or_of(make_filter<duckdb::IsNullFilter>(), dynamic_placeholder()), type));
  // The shape a runtime join filter actually arrives in.
  REQUIRE_FALSE(filter_safe_for_stats(
    *optional_of(or_of(make_filter<duckdb::IsNullFilter>(), dynamic_placeholder())), type));
  // Deep nesting: one dynamic leaf poisons the whole tree.
  REQUIRE_FALSE(filter_safe_for_stats(
    *and_of(or_of(cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(1)), dynamic_placeholder()),
            cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(0))),
    type));
}

// ============================================================================
// filter_safe_for_stats — unsupported shapes and malformed nodes
// ============================================================================

TEST_CASE("pinned_chunk_stats - classifier rejects unsupported shapes and malformed nodes",
          "[pinned_chunk_stats]")
{
  auto const type = LogicalType::INTEGER;

  // Distinct-from comparisons have NULL semantics CheckStatistics ignores.
  REQUIRE_FALSE(
    filter_safe_for_stats(*cmp(ExpressionType::COMPARE_DISTINCT_FROM, Value::INTEGER(1)), type));

  REQUIRE_FALSE(filter_safe_for_stats(
    duckdb::StructFilter{0, "child", cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(1))}, type));

  duckdb::ExpressionFilter expression_filter{
    duckdb::make_uniq<duckdb::BoundConstantExpression>(Value::BOOLEAN(true))};
  REQUIRE_FALSE(filter_safe_for_stats(expression_filter, type));

  // Childless conjunctions cannot occur from the binder; both reject (a
  // childless OR would fold to FILTER_ALWAYS_FALSE and prune unconditionally).
  REQUIRE_FALSE(filter_safe_for_stats(duckdb::ConjunctionAndFilter{}, type));
  REQUIRE_FALSE(filter_safe_for_stats(duckdb::ConjunctionOrFilter{}, type));

  // OptionalFilter's child defaults to nullptr; its CheckStatistics derefs the
  // child unconditionally, so the classifier must reject childless optionals.
  REQUIRE_FALSE(filter_safe_for_stats(duckdb::OptionalFilter{}, type));

  // InFilter's constructor bans null/empty values, so violate the invariants
  // by mutating the public field — the classifier must stay total anyway.
  auto in_with_null = in_list({Value::INTEGER(1)});
  in_with_null->Cast<duckdb::InFilter>().values.emplace_back(LogicalType::INTEGER);  // NULL value
  REQUIRE_FALSE(filter_safe_for_stats(*in_with_null, type));

  auto in_emptied = in_list({Value::INTEGER(1)});
  in_emptied->Cast<duckdb::InFilter>().values.clear();
  REQUIRE_FALSE(filter_safe_for_stats(*in_emptied, type));
}

// ============================================================================
// filter_safe_for_stats — exact logical type equality
// ============================================================================

TEST_CASE("pinned_chunk_stats - classifier requires exact logical type equality",
          "[pinned_chunk_stats]")
{
  // Same physical width, different logical type: the release-mode misread
  // cases the classifier exists for.
  REQUIRE_FALSE(filter_safe_for_stats(*cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(1)),
                                      LogicalType::DATE));  // both int32
  REQUIRE_FALSE(filter_safe_for_stats(*cmp(ExpressionType::COMPARE_EQUAL, Value::BIGINT(1)),
                                      LogicalType::TIMESTAMP));  // both int64
  REQUIRE_FALSE(filter_safe_for_stats(*in_list({Value::INTEGER(1)}), LogicalType::DATE));

  // Exactly-typed controls for the same physical widths.
  auto const jan_2020 = Value::DATE(duckdb::Date::FromDate(2020, 1, 1));
  REQUIRE(filter_safe_for_stats(*cmp(ExpressionType::COMPARE_EQUAL, jan_2020), LogicalType::DATE));
  REQUIRE(filter_safe_for_stats(
    *cmp(ExpressionType::COMPARE_EQUAL, Value::TIMESTAMP(duckdb::timestamp_t{1'000'000})),
    LogicalType::TIMESTAMP));

  // One mismatched leaf poisons a conjunction.
  REQUIRE_FALSE(
    filter_safe_for_stats(*and_of(cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(1)),
                                  cmp(ExpressionType::COMPARE_LESSTHAN, Value::BIGINT(10))),
                          LogicalType::INTEGER));
}

// ============================================================================
// chunk_provably_empty — prunes only on FILTER_ALWAYS_FALSE
// ============================================================================

TEST_CASE("pinned_chunk_stats - prune only on provable emptiness", "[pinned_chunk_stats]")
{
  auto const stats = make_stats(LogicalType::INTEGER, Value::INTEGER(10), Value::INTEGER(20));

  // Provably disjoint ranges prune.
  REQUIRE(
    chunk_provably_empty(*cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100)), stats));
  REQUIRE(chunk_provably_empty(*cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(5)), stats));
  REQUIRE(chunk_provably_empty(*cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(25)), stats));
  REQUIRE(
    chunk_provably_empty(*cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(20)), stats));

  // Overlapping or boundary-touching ranges keep.
  REQUIRE_FALSE(
    chunk_provably_empty(*cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(15)), stats));
  REQUIRE_FALSE(chunk_provably_empty(
    *cmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(20)), stats));

  // FILTER_ALWAYS_TRUE must not prune: only ALWAYS_FALSE does.
  REQUIRE_FALSE(
    chunk_provably_empty(*cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(0)), stats));

  // != prunes only a constant column matching the constant.
  REQUIRE_FALSE(
    chunk_provably_empty(*cmp(ExpressionType::COMPARE_NOTEQUAL, Value::INTEGER(15)), stats));
  auto const constant_column =
    make_stats(LogicalType::INTEGER, Value::INTEGER(15), Value::INTEGER(15));
  REQUIRE(chunk_provably_empty(*cmp(ExpressionType::COMPARE_NOTEQUAL, Value::INTEGER(15)),
                               constant_column));

  // IN prunes only when every member misses the range.
  REQUIRE(chunk_provably_empty(*in_list({Value::INTEGER(100), Value::INTEGER(200)}), stats));
  REQUIRE_FALSE(chunk_provably_empty(*in_list({Value::INTEGER(15), Value::INTEGER(100)}), stats));

  // Conjunctions fold; OPTIONAL delegates to its child.
  REQUIRE(
    chunk_provably_empty(*and_of(cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100)),
                                 cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(200))),
                         stats));
  REQUIRE(chunk_provably_empty(*or_of(cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100)),
                                      cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(5))),
                               stats));
  REQUIRE_FALSE(
    chunk_provably_empty(*or_of(cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(15)),
                                cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100))),
                         stats));
  REQUIRE(chunk_provably_empty(
    *optional_of(cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100))), stats));
}

// ============================================================================
// chunk_provably_empty — null-flag proofs
// ============================================================================

TEST_CASE("pinned_chunk_stats - prune handles null-based proofs", "[pinned_chunk_stats]")
{
  auto const no_nulls =
    make_stats(LogicalType::INTEGER, Value::INTEGER(10), Value::INTEGER(20), /*has_null=*/false);
  auto const with_nulls =
    make_stats(LogicalType::INTEGER, Value::INTEGER(10), Value::INTEGER(20), /*has_null=*/true);

  REQUIRE(chunk_provably_empty(duckdb::IsNullFilter{}, no_nulls));
  REQUIRE_FALSE(chunk_provably_empty(duckdb::IsNullFilter{}, with_nulls));
  REQUIRE_FALSE(chunk_provably_empty(duckdb::IsNotNullFilter{}, with_nulls));

  // All-null statistics: compute_pinned_chunk_stats never produces these
  // (all-null columns get a null stats cell), but the prune check's contract
  // covers any well-formed BaseStatistics a future producer hands it.
  auto all_null = duckdb::NumericStats::CreateUnknown(LogicalType::INTEGER);
  all_null.Set(duckdb::StatsInfo::CANNOT_HAVE_VALID_VALUES);
  all_null.SetHasNull();
  REQUIRE(chunk_provably_empty(duckdb::IsNotNullFilter{}, all_null));
  // Constant comparisons need a non-null row to match, so all-null prunes too.
  REQUIRE(chunk_provably_empty(*cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(15)), all_null));
}

// ============================================================================
// chunk_provably_empty — conservative on unsafe input
// ============================================================================

TEST_CASE("pinned_chunk_stats - prune is conservative on unsafe input", "[pinned_chunk_stats]")
{
  auto const date_stats = make_stats(LogicalType::DATE,
                                     Value::DATE(duckdb::Date::FromDate(2020, 1, 1)),
                                     Value::DATE(duckdb::Date::FromDate(2020, 6, 30)));

  // Type-mismatched probe: must return "keep" WITHOUT reaching CheckStatistics
  // — in a debug build reaching it would D_ASSERT-abort this very test, so
  // passing under test_debug is itself evidence the gate holds.
  REQUIRE_FALSE(
    chunk_provably_empty(*cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(1)), date_stats));
  REQUIRE_FALSE(chunk_provably_empty(*dynamic_placeholder(), date_stats));
  REQUIRE_FALSE(chunk_provably_empty(duckdb::OptionalFilter{}, date_stats));
}

// ============================================================================
// chunk_provably_empty — exactly-typed DATE / TIMESTAMP proofs
// ============================================================================

TEST_CASE("pinned_chunk_stats - DATE and TIMESTAMP stats prune with exactly-typed constants",
          "[pinned_chunk_stats]")
{
  auto const date_stats = make_stats(LogicalType::DATE,
                                     Value::DATE(duckdb::Date::FromDate(2020, 1, 1)),
                                     Value::DATE(duckdb::Date::FromDate(2020, 6, 30)));
  REQUIRE(chunk_provably_empty(
    *cmp(ExpressionType::COMPARE_GREATERTHAN, Value::DATE(duckdb::Date::FromDate(2021, 1, 1))),
    date_stats));
  REQUIRE_FALSE(chunk_provably_empty(
    *cmp(ExpressionType::COMPARE_GREATERTHAN, Value::DATE(duckdb::Date::FromDate(2020, 3, 1))),
    date_stats));

  auto const ts_stats = make_stats(LogicalType::TIMESTAMP,
                                   Value::TIMESTAMP(duckdb::timestamp_t{1'000'000}),
                                   Value::TIMESTAMP(duckdb::timestamp_t{2'000'000}));
  REQUIRE(chunk_provably_empty(
    *cmp(ExpressionType::COMPARE_GREATERTHAN, Value::TIMESTAMP(duckdb::timestamp_t{10'000'000})),
    ts_stats));
  REQUIRE_FALSE(chunk_provably_empty(
    *cmp(ExpressionType::COMPARE_EQUAL, Value::TIMESTAMP(duckdb::timestamp_t{1'500'000})),
    ts_stats));
}

// ============================================================================
// pinned_zone_maps — the entry sidecar (host-only: covers the insert/merge
// mirroring semantics of §5.4 without cudf tables; the scan-manager-level
// GPU merge rides the stage-6 integration tests)
// ============================================================================

namespace {

/// Chunk-major capture matrix (capture[c][i], as compute_pinned_chunk_stats
/// emits) of INTEGER cells whose bounds encode (chunk, column) so any
/// misrouted cell fails loudly: column i of chunk c gets [base, base + 9]
/// with base = 1000*c + 100*i.
std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> make_capture(
  std::size_t n_chunks, std::size_t n_columns)
{
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> capture(n_chunks);
  for (std::size_t c = 0; c < n_chunks; ++c) {
    for (std::size_t i = 0; i < n_columns; ++i) {
      auto const base = static_cast<int32_t>(1000 * c + 100 * i);
      capture[c].push_back(
        make_stats(LogicalType::INTEGER, Value::INTEGER(base), Value::INTEGER(base + 9))
          .ToUnique());
    }
  }
  return capture;
}

duckdb::vector<LogicalType> int_types(std::size_t n)
{
  return duckdb::vector<LogicalType>(n, LogicalType::INTEGER);
}

/// Assert @p cell carries the [expected_base, expected_base + 9] INTEGER range
/// make_capture stamped for its (chunk, column).
void require_cell_bounds(duckdb::BaseStatistics const* cell, int32_t expected_base)
{
  REQUIRE(cell != nullptr);
  REQUIRE(duckdb::NumericStats::Min(*cell) == Value::INTEGER(expected_base));
  REQUIRE(duckdb::NumericStats::Max(*cell) == Value::INTEGER(expected_base + 9));
}

}  // namespace

TEST_CASE("pinned_zone_maps - from_capture pivots a well-formed capture and preserves null cells",
          "[pinned_chunk_stats]")
{
  auto capture  = make_capture(/*n_chunks=*/3, /*n_columns=*/2);
  capture[1][0] = nullptr;  // an unsupported-type cell stays a null cell
  auto const zm = pinned_zone_maps::from_capture(int_types(2), std::move(capture), 2, 3);

  REQUIRE(zm.has_stats());
  REQUIRE(zm.column_count() == 2);
  REQUIRE(zm.column_type(0) == LogicalType::INTEGER);

  // cell(pos, chunk) == capture[chunk][pos]
  require_cell_bounds(zm.cell(0, 0), 0);
  require_cell_bounds(zm.cell(0, 2), 2000);
  require_cell_bounds(zm.cell(1, 0), 100);
  require_cell_bounds(zm.cell(1, 1), 1100);
  require_cell_bounds(zm.cell(1, 2), 2100);
  REQUIRE(zm.cell(0, 1) == nullptr);

  // Glue with the prune check: a served cell composes with chunk_provably_empty.
  REQUIRE(chunk_provably_empty(*cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(5000)),
                               *zm.cell(0, 2)));
  REQUIRE_FALSE(chunk_provably_empty(*cmp(ExpressionType::COMPARE_EQUAL, Value::INTEGER(2005)),
                                     *zm.cell(0, 2)));
}

TEST_CASE("pinned_zone_maps - from_capture normalizes malformed captures to absent",
          "[pinned_chunk_stats]")
{
  auto expect_absent = [](pinned_zone_maps const& zm) {
    REQUIRE_FALSE(zm.has_stats());
    REQUIRE(zm.column_count() == 0);
    REQUIRE(zm.cell(0, 0) == nullptr);  // total even when absent
  };

  // Statless capture (no stats supplied).
  expect_absent(pinned_zone_maps::from_capture(int_types(2), {}, 2, 3));
  // Stats without types.
  expect_absent(pinned_zone_maps::from_capture({}, make_capture(3, 2), 2, 3));
  // Types width disagrees with the column count.
  expect_absent(pinned_zone_maps::from_capture(int_types(3), make_capture(3, 2), 2, 3));
  // Chunk count disagrees.
  expect_absent(pinned_zone_maps::from_capture(int_types(2), make_capture(2, 2), 2, 3));
  // One ragged chunk row.
  auto ragged = make_capture(3, 2);
  ragged[1].pop_back();
  expect_absent(pinned_zone_maps::from_capture(int_types(2), std::move(ragged), 2, 3));
  // Zero-chunk pin.
  expect_absent(pinned_zone_maps::from_capture(int_types(2), {}, 2, 0));
}

TEST_CASE("pinned_zone_maps - accessors are total on well-formed sidecars", "[pinned_chunk_stats]")
{
  auto const zm = pinned_zone_maps::from_capture(int_types(1), make_capture(2, 1), 1, 2);
  REQUIRE(zm.has_stats());
  require_cell_bounds(zm.cell(0, 1), 1000);
  REQUIRE(zm.cell(1, 0) == nullptr);  // column position out of range
  REQUIRE(zm.cell(0, 2) == nullptr);  // chunk index out of range
}

TEST_CASE("pinned_zone_maps - append_column_from mirrors a stats-carrying merge",
          "[pinned_chunk_stats]")
{
  // Entry pinned with 2 columns; a re-pin arrives with 3 (same chunk layout)
  // and the data merge appends its column 2. A re-pin of only duplicate
  // columns appends nothing and trivially keeps the entry's stats.
  auto entry    = pinned_zone_maps::from_capture(int_types(2), make_capture(3, 2), 2, 3);
  auto incoming = pinned_zone_maps::from_capture(int_types(3), make_capture(3, 3), 3, 3);

  entry.append_column_from(incoming, 2);

  REQUIRE(entry.has_stats());
  REQUIRE(entry.column_count() == 3);
  REQUIRE(entry.column_type(2) == LogicalType::INTEGER);
  // The appended column carries incoming's per-chunk cells...
  require_cell_bounds(entry.cell(2, 0), 200);
  require_cell_bounds(entry.cell(2, 2), 2200);
  // ...and the pre-merge columns are untouched.
  require_cell_bounds(entry.cell(0, 0), 0);
  require_cell_bounds(entry.cell(1, 2), 2100);
}

TEST_CASE("pinned_zone_maps - append_column_from degrades one-way on incompatible merges",
          "[pinned_chunk_stats]")
{
  SECTION("incoming side is absent")
  {
    auto entry = pinned_zone_maps::from_capture(int_types(2), make_capture(3, 2), 2, 3);
    pinned_zone_maps statless_incoming;
    entry.append_column_from(statless_incoming, 0);
    REQUIRE_FALSE(entry.has_stats());

    // One-way: a later compatible append cannot resurrect the sidecar.
    auto good = pinned_zone_maps::from_capture(int_types(1), make_capture(3, 1), 1, 3);
    entry.append_column_from(good, 0);
    REQUIRE_FALSE(entry.has_stats());
  }

  SECTION("entry side is absent and stays absent")
  {
    pinned_zone_maps entry;
    auto incoming = pinned_zone_maps::from_capture(int_types(1), make_capture(3, 1), 1, 3);
    entry.append_column_from(incoming, 0);
    REQUIRE_FALSE(entry.has_stats());
    REQUIRE(incoming.has_stats());  // early-out leaves the incoming side intact
  }

  SECTION("incoming position out of range")
  {
    auto entry    = pinned_zone_maps::from_capture(int_types(2), make_capture(3, 2), 2, 3);
    auto incoming = pinned_zone_maps::from_capture(int_types(2), make_capture(3, 2), 2, 3);
    entry.append_column_from(incoming, 5);
    REQUIRE_FALSE(entry.has_stats());
  }

  SECTION("chunk counts disagree")
  {
    auto entry    = pinned_zone_maps::from_capture(int_types(2), make_capture(3, 2), 2, 3);
    auto incoming = pinned_zone_maps::from_capture(int_types(2), make_capture(2, 2), 2, 2);
    entry.append_column_from(incoming, 0);
    REQUIRE_FALSE(entry.has_stats());
  }
}

TEST_CASE("pinned_zone_maps - remap rebuilds a sidecar positionally", "[pinned_chunk_stats]")
{
  SECTION("reordering adoption")
  {
    auto incoming = pinned_zone_maps::from_capture(int_types(3), make_capture(2, 3), 3, 2);
    auto out      = pinned_zone_maps::remap(std::move(incoming), {2, 0});
    REQUIRE(out.has_stats());
    REQUIRE(out.column_count() == 2);
    require_cell_bounds(out.cell(0, 0), 200);
    require_cell_bounds(out.cell(0, 1), 1200);
    require_cell_bounds(out.cell(1, 0), 0);
    require_cell_bounds(out.cell(1, 1), 1000);
  }

  SECTION("absent incoming, empty mapping, out-of-range, and duplicates yield absent")
  {
    REQUIRE_FALSE(pinned_zone_maps::remap(pinned_zone_maps{}, {0}).has_stats());
    auto a = pinned_zone_maps::from_capture(int_types(2), make_capture(2, 2), 2, 2);
    REQUIRE_FALSE(pinned_zone_maps::remap(std::move(a), {}).has_stats());
    auto b = pinned_zone_maps::from_capture(int_types(2), make_capture(2, 2), 2, 2);
    REQUIRE_FALSE(pinned_zone_maps::remap(std::move(b), {0, 5}).has_stats());
    auto c = pinned_zone_maps::from_capture(int_types(2), make_capture(2, 2), 2, 2);
    REQUIRE_FALSE(pinned_zone_maps::remap(std::move(c), {1, 1}).has_stats());
  }
}

// ============================================================================
// compute_pinned_chunk_stats — GPU capture (exact bounds, null flags, and the
// null-entry degradations of the v1 type allowlist)
// ============================================================================

namespace {

using sirius::scan_manager::compute_pinned_chunk_stats;

struct gpu_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;

  gpu_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0))
  {
  }
};

gpu_env& genv()
{
  static gpu_env e;
  return e;
}

/// Fixed-width GPU column of @p id from host values (T = the physical rep, e.g.
/// int32_t for TIMESTAMP_DAYS); empty @p validity means no null mask, else
/// validity[i] == false makes row i NULL.
template <typename T>
std::unique_ptr<cudf::column> make_gpu_col(cudf::type_id id,
                                           std::vector<T> const& values,
                                           std::vector<bool> const& validity = {})
{
  auto& e     = genv();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto col    = cudf::make_fixed_width_column(cudf::data_type{id},
                                           static_cast<cudf::size_type>(values.size()),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  if (!values.empty()) {
    cudaMemcpy(col->mutable_view().head<T>(),
               values.data(),
               sizeof(T) * values.size(),
               cudaMemcpyHostToDevice);
  }
  if (!validity.empty()) {
    auto mask = cudf::create_null_mask(
      static_cast<cudf::size_type>(values.size()), cudf::mask_state::ALL_VALID, stream, mr);
    cudf::size_type null_count = 0;
    for (std::size_t i = 0; i < validity.size(); ++i) {
      if (validity[i]) { continue; }
      cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()),
                          static_cast<cudf::size_type>(i),
                          static_cast<cudf::size_type>(i + 1),
                          false,
                          stream);
      ++null_count;
    }
    col->set_null_mask(std::move(mask), null_count);
  }
  return col;
}

/// Run the capture over a single-column chunk with @p type declared for it.
duckdb::unique_ptr<duckdb::BaseStatistics> capture_one(cudf::column_view const& col,
                                                       LogicalType const& type)
{
  auto& e     = genv();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();
  auto stats  = compute_pinned_chunk_stats(
    cudf::table_view{{col}}, duckdb::vector<LogicalType>{type}, stream, mr);
  REQUIRE(stats.size() == 1);
  return std::move(stats[0]);
}

}  // namespace

TEST_CASE("compute_pinned_chunk_stats - exact bounds and null flags for allowed types",
          "[pinned_chunk_stats]")
{
  SECTION("INT32, no nulls")
  {
    auto col   = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 1, 9, 3});
    auto stats = capture_one(col->view(), LogicalType::INTEGER);
    REQUIRE(stats);
    REQUIRE(stats->GetType() == LogicalType::INTEGER);
    REQUIRE(duckdb::NumericStats::Min(*stats) == Value::INTEGER(1));
    REQUIRE(duckdb::NumericStats::Max(*stats) == Value::INTEGER(9));
    REQUIRE_FALSE(stats->CanHaveNull());
    REQUIRE(stats->CanHaveNoNull());
  }

  SECTION("INT32, nulls excluded from bounds and flagged")
  {
    auto col =
      make_gpu_col<int32_t>(cudf::type_id::INT32, {7, 2, 11, 4}, {true, false, true, true});
    auto stats = capture_one(col->view(), LogicalType::INTEGER);
    REQUIRE(stats);
    REQUIRE(duckdb::NumericStats::Min(*stats) == Value::INTEGER(4));
    REQUIRE(duckdb::NumericStats::Max(*stats) == Value::INTEGER(11));
    REQUIRE(stats->CanHaveNull());
    REQUIRE(stats->CanHaveNoNull());
  }

  SECTION("UBIGINT")
  {
    auto col   = make_gpu_col<std::uint64_t>(cudf::type_id::UINT64, {10, 3, 7});
    auto stats = capture_one(col->view(), LogicalType::UBIGINT);
    REQUIRE(stats);
    REQUIRE(duckdb::NumericStats::Min(*stats) == Value::UBIGINT(3));
    REQUIRE(duckdb::NumericStats::Max(*stats) == Value::UBIGINT(10));
  }

  SECTION("DATE from TIMESTAMP_DAYS")
  {
    auto col   = make_gpu_col<int32_t>(cudf::type_id::TIMESTAMP_DAYS, {18443, 18262, 18300});
    auto stats = capture_one(col->view(), LogicalType::DATE);
    REQUIRE(stats);
    REQUIRE(stats->GetType() == LogicalType::DATE);
    REQUIRE(duckdb::NumericStats::Min(*stats) == Value::DATE(duckdb::date_t{18262}));
    REQUIRE(duckdb::NumericStats::Max(*stats) == Value::DATE(duckdb::date_t{18443}));
  }

  SECTION("TIMESTAMP from TIMESTAMP_MICROSECONDS")
  {
    auto col = make_gpu_col<int64_t>(cudf::type_id::TIMESTAMP_MICROSECONDS, {5'000'000, 1'000'000});
    auto stats = capture_one(col->view(), LogicalType::TIMESTAMP);
    REQUIRE(stats);
    REQUIRE(stats->GetType() == LogicalType::TIMESTAMP);
    REQUIRE(duckdb::NumericStats::Min(*stats) == Value::TIMESTAMP(duckdb::timestamp_t{1'000'000}));
    REQUIRE(duckdb::NumericStats::Max(*stats) == Value::TIMESTAMP(duckdb::timestamp_t{5'000'000}));
  }
}

TEST_CASE("compute_pinned_chunk_stats - unsupported or degenerate columns yield null cells",
          "[pinned_chunk_stats]")
{
  // Outside the v1 allowlist entirely.
  auto float_col = make_gpu_col<double>(cudf::type_id::FLOAT64, {1.0, 2.0});
  REQUIRE_FALSE(capture_one(float_col->view(), LogicalType::DOUBLE));

  // Strings: allowlist excludes them regardless of the declared type.
  auto& e    = genv();
  auto mr    = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto strm  = sirius::test::operator_utils::default_stream();
  auto s_col = cudf::make_column_from_scalar(cudf::string_scalar("x", true, strm, mr), 3, strm, mr);
  REQUIRE_FALSE(capture_one(s_col->view(), LogicalType::VARCHAR));

  // Decode-type mismatch: DATE must decode as TIMESTAMP_DAYS, not raw INT32.
  auto i32_col = make_gpu_col<int32_t>(cudf::type_id::INT32, {18262, 18443});
  REQUIRE_FALSE(capture_one(i32_col->view(), LogicalType::DATE));

  // The timestamp unit trap: a millisecond decode bound to DuckDB TIMESTAMP (µs)
  // must NOT produce stats — the raw rep would be off by 10^3.
  auto ms_col = make_gpu_col<int64_t>(cudf::type_id::TIMESTAMP_MILLISECONDS, {1'000, 5'000});
  REQUIRE_FALSE(capture_one(ms_col->view(), LogicalType::TIMESTAMP));

  // All-null and empty columns have no bounds to capture.
  auto all_null = make_gpu_col<int32_t>(cudf::type_id::INT32, {1, 2}, {false, false});
  REQUIRE_FALSE(capture_one(all_null->view(), LogicalType::INTEGER));
  auto empty = make_gpu_col<int32_t>(cudf::type_id::INT32, {});
  REQUIRE_FALSE(capture_one(empty->view(), LogicalType::INTEGER));
}

TEST_CASE("compute_pinned_chunk_stats - per-column alignment and shape degradation",
          "[pinned_chunk_stats]")
{
  auto& e     = genv();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();

  auto supported_a   = make_gpu_col<int32_t>(cudf::type_id::INT32, {1, 2, 3});
  auto unsupported_b = make_gpu_col<double>(cudf::type_id::FLOAT64, {1.0, 2.0, 3.0});
  auto supported_c   = make_gpu_col<int64_t>(cudf::type_id::INT64, {30, 10, 20});
  cudf::table_view chunk{{supported_a->view(), unsupported_b->view(), supported_c->view()}};

  SECTION("unsupported columns get null cells; neighbors stay aligned")
  {
    duckdb::vector<LogicalType> types{
      LogicalType::INTEGER, LogicalType::DOUBLE, LogicalType::BIGINT};
    auto stats = compute_pinned_chunk_stats(chunk, types, stream, mr);
    REQUIRE(stats.size() == 3);
    REQUIRE(stats[0]);
    REQUIRE_FALSE(stats[1]);
    REQUIRE(stats[2]);
    REQUIRE(duckdb::NumericStats::Min(*stats[2]) == Value::BIGINT(10));
    REQUIRE(duckdb::NumericStats::Max(*stats[2]) == Value::BIGINT(30));
  }

  SECTION("column_types width mismatch degrades every cell to null")
  {
    duckdb::vector<LogicalType> too_few{LogicalType::INTEGER};
    auto stats = compute_pinned_chunk_stats(chunk, too_few, stream, mr);
    REQUIRE(stats.size() == 3);
    REQUIRE_FALSE(stats[0]);
    REQUIRE_FALSE(stats[1]);
    REQUIRE_FALSE(stats[2]);
  }
}

// ============================================================================
// build_cached_scan_plan — serve-time survivor plans (host-only: plan building
// never dereferences chunk data, so entries carry null chunk placeholders)
// ============================================================================

namespace {

using sirius::scan_manager::build_cached_scan_plan;
using sirius::scan_manager::pinned_entry;

using survivors_t = std::vector<std::size_t>;

/// HOST-tier entry with @p primaries.size() INTEGER columns over @p n_chunks
/// chunks, zone maps from make_capture — column i of chunk c covers
/// [1000*c + 100*i, 1000*c + 100*i + 9]. Host chunks stay null: the plan
/// builder only reads their count.
pinned_entry make_plan_entry(std::size_t n_chunks, std::vector<duckdb::idx_t> const& primaries)
{
  pinned_entry entry;
  for (std::size_t i = 0; i < primaries.size(); ++i) {
    entry.cache_info.column_ids.emplace_back(duckdb::ColumnIndex(primaries[i]));
    entry.cache_info.names.push_back("c" + std::to_string(i));
  }
  entry.tier = cucascade::memory::Tier::HOST;
  entry.host_chunks.resize(n_chunks);
  entry.zone_maps = pinned_zone_maps::from_capture(int_types(primaries.size()),
                                                   make_capture(n_chunks, primaries.size()),
                                                   primaries.size(),
                                                   n_chunks);
  return entry;
}

duckdb::TableFilterSet make_filter_set(duckdb::idx_t key, filter_ptr f)
{
  duckdb::TableFilterSet fs;
  fs.filters[key] = std::move(f);
  return fs;
}

}  // namespace

TEST_CASE("build_cached_scan_plan - prunes exactly the provably empty chunks",
          "[pinned_chunk_stats]")
{
  // Column 0 (primary 3) ranges: [0,9], [1000,1009], [2000,2009], [3000,3009].
  auto const entry = make_plan_entry(4, {3});
  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(3)};

  auto fs =
    make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(2000)));
  auto plan = build_cached_scan_plan(entry, &fs, &qcols);
  REQUIRE(plan.survivor_chunk_indices == survivors_t{2, 3});
  REQUIRE(plan.pruned == 2);

  auto in_fs   = make_filter_set(0, in_list({Value::INTEGER(5), Value::INTEGER(3005)}));
  auto in_plan = build_cached_scan_plan(entry, &in_fs, &qcols);
  REQUIRE(in_plan.survivor_chunk_indices == survivors_t{0, 3});
  REQUIRE(in_plan.pruned == 2);
}

TEST_CASE("build_cached_scan_plan - filter keys remap through query column_ids to entry positions",
          "[pinned_chunk_stats]")
{
  // Entry columns: pos 0 = primary 7 (ranges 1000c..), pos 1 = primary 3
  // (ranges 1000c+100..). The query selects only primary 3, so filter key 0
  // must land on entry position 1.
  auto const entry = make_plan_entry(3, {7, 3});
  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(3)};

  auto fs =
    make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(1100)));
  auto plan = build_cached_scan_plan(entry, &fs, &qcols);
  REQUIRE(plan.survivor_chunk_indices == survivors_t{1, 2});
  REQUIRE(plan.pruned == 1);
}

TEST_CASE("build_cached_scan_plan - all-pruned keeps the sentinel chunk", "[pinned_chunk_stats]")
{
  auto const entry = make_plan_entry(4, {3});
  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(3)};
  auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));

  auto plan = build_cached_scan_plan(entry, &fs, &qcols);
  REQUIRE(plan.survivor_chunk_indices == survivors_t{0});
  REQUIRE(plan.pruned == 3);
}

TEST_CASE("build_cached_scan_plan - identity plan whenever pruning is not provably safe",
          "[pinned_chunk_stats]")
{
  auto const entry = make_plan_entry(3, {3});
  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(3)};
  auto const identity = survivors_t{0, 1, 2};

  SECTION("no filters")
  {
    duckdb::TableFilterSet empty_fs;
    REQUIRE(build_cached_scan_plan(entry, nullptr, &qcols).survivor_chunk_indices == identity);
    REQUIRE(build_cached_scan_plan(entry, &empty_fs, &qcols).survivor_chunk_indices == identity);
  }

  SECTION("no query column_ids")
  {
    auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));
    REQUIRE(build_cached_scan_plan(entry, &fs, nullptr).survivor_chunk_indices == identity);
  }

  SECTION("statless entry")
  {
    auto statless      = make_plan_entry(3, {3});
    statless.zone_maps = pinned_zone_maps{};
    auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));
    auto plan = build_cached_scan_plan(statless, &fs, &qcols);
    REQUIRE(plan.survivor_chunk_indices == identity);
    REQUIRE(plan.pruned == 0);
  }

  SECTION("filter key out of range")
  {
    auto fs = make_filter_set(9, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));
    REQUIRE(build_cached_scan_plan(entry, &fs, &qcols).survivor_chunk_indices == identity);
  }

  SECTION("rowid sentinel column is gated before GetPrimaryIndex")
  {
    duckdb::vector<duckdb::ColumnIndex> rowid_cols{duckdb::ColumnIndex()};
    auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));
    REQUIRE(build_cached_scan_plan(entry, &fs, &rowid_cols).survivor_chunk_indices == identity);
  }

  SECTION("filtered column absent from the entry")
  {
    duckdb::vector<duckdb::ColumnIndex> other_cols{duckdb::ColumnIndex(8)};
    auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));
    REQUIRE(build_cached_scan_plan(entry, &fs, &other_cols).survivor_chunk_indices == identity);
  }

  SECTION("exact-type mismatch rejects the filter")
  {
    auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::BIGINT(100000)));
    REQUIRE(build_cached_scan_plan(entry, &fs, &qcols).survivor_chunk_indices == identity);
  }

  SECTION("dynamic filter rejects; a usable sibling filter still prunes")
  {
    duckdb::TableFilterSet fs;
    fs.filters[0] = dynamic_placeholder();
    duckdb::vector<duckdb::ColumnIndex> two_cols{duckdb::ColumnIndex(3), duckdb::ColumnIndex(3)};
    fs.filters[1] = cmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(1000));
    auto plan     = build_cached_scan_plan(entry, &fs, &two_cols);
    REQUIRE(plan.survivor_chunk_indices == survivors_t{1, 2});
    REQUIRE(plan.pruned == 1);
  }

  SECTION("zero-chunk entry yields an empty plan, no sentinel")
  {
    auto empty_entry = make_plan_entry(0, {3});
    auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(100000)));
    auto plan = build_cached_scan_plan(empty_entry, &fs, &qcols);
    REQUIRE(plan.survivor_chunk_indices.empty());
    REQUIRE(plan.pruned == 0);
  }
}

TEST_CASE("build_cached_scan_plan - absent cells keep their chunks", "[pinned_chunk_stats]")
{
  pinned_entry entry;
  entry.cache_info.column_ids.emplace_back(duckdb::ColumnIndex(3));
  entry.cache_info.names.push_back("c0");
  entry.tier = cucascade::memory::Tier::HOST;
  entry.host_chunks.resize(3);
  auto capture    = make_capture(3, 1);
  capture[1][0]   = nullptr;
  entry.zone_maps = pinned_zone_maps::from_capture(int_types(1), std::move(capture), 1, 3);

  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(3)};
  auto fs = make_filter_set(0, cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(0)));

  auto plan = build_cached_scan_plan(entry, &fs, &qcols);
  REQUIRE(plan.survivor_chunk_indices == survivors_t{1});
  REQUIRE(plan.pruned == 2);
}

TEST_CASE("build_cached_scan_plan - duplicate column names never alias (positional stats)",
          "[pinned_chunk_stats]")
{
  // Two columns share the name "x" but have distinct primaries and ranges;
  // pruning on primary 2 (entry pos 1, ranges 1000c+100..) must use ITS
  // ranges, not pos 0's.
  auto entry             = make_plan_entry(3, {1, 2});
  entry.cache_info.names = {"x", "x"};

  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(2)};

  // Pos 1 ranges: [100,109], [1100,1109], [2100,2109]; pos 0 ranges:
  // [0,9], [1000,1009], [2000,2009]. "< 1050" is the discriminating probe —
  // it keeps {0} against pos-1 bounds but would keep {0, 1} against pos-0
  // bounds, so an aliased lookup fails the assertion.
  auto fs   = make_filter_set(0, cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(1050)));
  auto plan = build_cached_scan_plan(entry, &fs, &qcols);
  REQUIRE(plan.survivor_chunk_indices == survivors_t{0});
  REQUIRE(plan.pruned == 2);
}

// ---------------------------------------------------------------------------
// compute_pinned_group_stats — the per-group (G * 1024 rows) capture.
// Same allowlist and same "null cell never prunes" contract as the whole-chunk
// capture; the difference is one segmented reduction per column instead of one
// reduction per group.
// ---------------------------------------------------------------------------

namespace {
/// Run the group capture over a single-column chunk with @p type declared for it.
sirius::scan_manager::chunk_group_stats capture_groups(cudf::column_view const& col,
                                                       LogicalType const& type,
                                                       std::size_t group_rows)
{
  auto& e     = genv();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();
  return sirius::scan_manager::compute_pinned_group_stats(
    cudf::table_view{{col}}, duckdb::vector<LogicalType>{type}, group_rows, stream, mr);
}

/// [min, max] of a group cell, as ints.
std::pair<int32_t, int32_t> int_bounds(duckdb::BaseStatistics const& s)
{
  return {duckdb::NumericStats::Min(s).GetValue<int32_t>(),
          duckdb::NumericStats::Max(s).GetValue<int32_t>()};
}
}  // namespace

TEST_CASE("compute_pinned_group_stats - exact per-group bounds", "[pinned_chunk_stats]")
{
  SECTION("groups divide the chunk evenly")
  {
    // 8 rows, 2 groups of 4: [5,1,9,3] -> [1,9]; [40,42,41,44] -> [40,44].
    auto col = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 1, 9, 3, 40, 42, 41, 44});
    auto gs  = capture_groups(col->view(), LogicalType::INTEGER, 4);
    REQUIRE(gs.group_rows == 4);
    REQUIRE(gs.group_count() == 2);
    REQUIRE(gs.groups[0][0] != nullptr);
    REQUIRE(int_bounds(*gs.groups[0][0]) == std::pair<int32_t, int32_t>{1, 9});
    REQUIRE(int_bounds(*gs.groups[1][0]) == std::pair<int32_t, int32_t>{40, 44});
  }

  SECTION("a short final group is captured, not dropped")
  {
    // 7 rows, group_rows 3 -> groups of 3, 3, 1. The parquet pin path does not
    // pack whole 122,880-row units, so a short tail group must be legal.
    auto col = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 1, 9, 30, 33, 31, 77});
    auto gs  = capture_groups(col->view(), LogicalType::INTEGER, 3);
    REQUIRE(gs.group_count() == 3);
    REQUIRE(int_bounds(*gs.groups[0][0]) == std::pair<int32_t, int32_t>{1, 9});
    REQUIRE(int_bounds(*gs.groups[1][0]) == std::pair<int32_t, int32_t>{30, 33});
    REQUIRE(int_bounds(*gs.groups[2][0]) == std::pair<int32_t, int32_t>{77, 77});
  }

  SECTION("one group per chunk reproduces the whole-chunk capture")
  {
    auto col    = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 1, 9, 3});
    auto coarse = capture_one(col->view(), LogicalType::INTEGER);
    auto gs     = capture_groups(col->view(), LogicalType::INTEGER, 4);
    REQUIRE(gs.group_count() == 1);
    REQUIRE(int_bounds(*gs.groups[0][0]) == int_bounds(*coarse));
  }

  SECTION("group_rows larger than the chunk yields a single group")
  {
    auto col = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 1, 9, 3});
    auto gs  = capture_groups(col->view(), LogicalType::INTEGER, 1024);
    REQUIRE(gs.group_count() == 1);
    REQUIRE(int_bounds(*gs.groups[0][0]) == std::pair<int32_t, int32_t>{1, 9});
  }

  SECTION("DATE reduces in the date domain")
  {
    auto col = make_gpu_col<int32_t>(cudf::type_id::TIMESTAMP_DAYS, {100, 90, 5000, 4000});
    auto gs  = capture_groups(col->view(), LogicalType::DATE, 2);
    REQUIRE(gs.group_count() == 2);
    REQUIRE(duckdb::NumericStats::Min(*gs.groups[0][0]) == Value::DATE(duckdb::date_t{90}));
    REQUIRE(duckdb::NumericStats::Max(*gs.groups[1][0]) == Value::DATE(duckdb::date_t{5000}));
  }
}

TEST_CASE("compute_pinned_group_stats - nulls and degenerate input", "[pinned_chunk_stats]")
{
  SECTION("a partly-null group reduces over its valid rows")
  {
    // Group 0 = [5, NULL, 9]; min/max must ignore the null, and the cell must
    // advertise that nulls are possible.
    auto col = make_gpu_col<int32_t>(
      cudf::type_id::INT32, {5, 7, 9, 30, 33, 31}, {true, false, true, true, true, true});
    auto gs = capture_groups(col->view(), LogicalType::INTEGER, 3);
    REQUIRE(gs.group_count() == 2);
    REQUIRE(int_bounds(*gs.groups[0][0]) == std::pair<int32_t, int32_t>{5, 9});
    REQUIRE(gs.groups[0][0]->CanHaveNull());
  }

  SECTION("an all-null group leaves its cell absent, which never prunes")
  {
    auto col = make_gpu_col<int32_t>(
      cudf::type_id::INT32, {5, 7, 9, 30, 33, 31}, {true, true, true, false, false, false});
    auto gs = capture_groups(col->view(), LogicalType::INTEGER, 3);
    REQUIRE(gs.group_count() == 2);
    REQUIRE(gs.groups[0][0] != nullptr);
    REQUIRE(gs.groups[1][0] == nullptr);
  }

  SECTION("a type outside the allowlist leaves every cell absent")
  {
    auto col = make_gpu_col<float>(cudf::type_id::FLOAT32, {1.0f, 2.0f, 3.0f, 4.0f});
    auto gs  = capture_groups(col->view(), LogicalType::FLOAT, 2);
    REQUIRE(gs.group_count() == 2);
    REQUIRE(gs.groups[0][0] == nullptr);
    REQUIRE(gs.groups[1][0] == nullptr);
  }

  SECTION("group_rows == 0 yields no statistics rather than throwing")
  {
    auto col = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 1, 9, 3});
    auto gs  = capture_groups(col->view(), LogicalType::INTEGER, 0);
    REQUIRE(gs.empty());
  }

  SECTION("a column count that disagrees with column_types captures nothing")
  {
    auto& e     = genv();
    auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
    auto stream = sirius::test::operator_utils::default_stream();
    auto a      = make_gpu_col<int32_t>(cudf::type_id::INT32, {1, 2, 3, 4});
    auto b      = make_gpu_col<int32_t>(cudf::type_id::INT32, {5, 6, 7, 8});
    auto gs     = sirius::scan_manager::compute_pinned_group_stats(
      cudf::table_view{{a->view(), b->view()}},
      duckdb::vector<LogicalType>{LogicalType::INTEGER},
      2,
      stream,
      mr);
    REQUIRE(gs.empty());
  }
}

// Not a correctness test: a timing probe for the representation question — does one
// duckdb::BaseStatistics object per (group, column) scale to the group counts a real pin
// produces? Tagged [.] so it only runs when named explicitly.
TEST_CASE("compute_pinned_group_stats - capture cost at realistic group counts",
          "[.][pinned_chunk_stats][bench]")
{
  auto& e     = genv();
  auto mr     = sirius::test::operator_utils::get_resource_ref(*e.gpu_space);
  auto stream = sirius::test::operator_utils::default_stream();

  constexpr std::size_t kRows = 8ull * 1024 * 1024;  // 8 M rows
  std::vector<int32_t> values(kRows);
  for (std::size_t i = 0; i < kRows; ++i) {
    values[i] = static_cast<int32_t>(i % 100000);
  }
  auto col = make_gpu_col<int32_t>(cudf::type_id::INT32, values);

  for (std::size_t group_rows : {kRows, std::size_t{65536}, std::size_t{8192}}) {
    auto const t0 = std::chrono::high_resolution_clock::now();
    auto gs       = sirius::scan_manager::compute_pinned_group_stats(
      cudf::table_view{{col->view()}},
      duckdb::vector<LogicalType>{LogicalType::INTEGER},
      group_rows,
      stream,
      mr);
    auto const t1 = std::chrono::high_resolution_clock::now();
    auto const ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    WARN("group_rows=" << group_rows << " groups=" << gs.group_count() << " -> " << ms
                       << " ms for 1 column of " << kRows << " rows");
    REQUIRE(gs.group_count() >= 1);
  }

  // Evaluation side: what a per-query plan pass over group cells costs. This is the number that
  // decides whether duckdb::BaseStatistics + CheckStatistics is a viable representation for the
  // group index, or whether 2b needs a packed columnar form.
  auto gs = sirius::scan_manager::compute_pinned_group_stats(
    cudf::table_view{{col->view()}},
    duckdb::vector<LogicalType>{LogicalType::INTEGER},
    8192,
    stream,
    mr);
  auto filter               = cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(50000));
  std::size_t empties       = 0;
  constexpr int kReps       = 200;
  std::size_t const n_cells = gs.group_count();
  auto const t0             = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < kReps; ++r) {
    for (auto const& group : gs.groups) {
      if (group[0] && sirius::scan_manager::chunk_provably_empty(*filter, *group[0])) { ++empties; }
    }
  }
  auto const t1    = std::chrono::high_resolution_clock::now();
  auto const cells = static_cast<double>(n_cells) * kReps;
  auto const ns    = std::chrono::duration<double, std::nano>(t1 - t0).count() / cells;
  WARN("chunk_provably_empty over " << n_cells << " group cells: " << ns
                                    << " ns/cell (empties=" << empties << ")");

  // The packed evaluator on the same cells: pack once into the arena, lower once, flat loop.
  std::vector<sirius::scan_manager::chunk_group_stats> capture;
  capture.push_back(std::move(gs));
  auto const arena = sirius::scan_manager::group_bounds_arena::from_capture(
    duckdb::vector<LogicalType>{LogicalType::INTEGER}, capture);
  REQUIRE_FALSE(arena.empty());
  auto const packed = arena.cell(0, 0);
  auto lowered = sirius::scan_manager::lowered_bound_filter::lower(*filter, LogicalType::INTEGER);
  REQUIRE(lowered.has_value());
  std::vector<std::uint32_t> survivors;
  auto const t2 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < kReps; ++r) {
    lowered->select_survivors(packed, survivors);
  }
  auto const t3  = std::chrono::high_resolution_clock::now();
  auto const ns2 = std::chrono::duration<double, std::nano>(t3 - t2).count() / cells;
  WARN("lowered_bound_filter over " << packed.size() << " group cells: " << ns2
                                    << " ns/cell (survivors=" << survivors.size() << ") -> "
                                    << (ns / ns2) << "x faster");
}

// ---------------------------------------------------------------------------
// lowered_bound_filter — the packed evaluator for the group index.
//
// It exists because chunk_provably_empty costs ~83 ns/cell and the group index has ~732k cells
// per pinned lineitem. It replaces DuckDB's CheckStatistics with bounds arithmetic, so the
// load-bearing test is not "does it prune" but "does it prune EXACTLY what CheckStatistics
// prunes" — a disagreement in the pruning direction drops rows and returns wrong answers.
// ---------------------------------------------------------------------------

namespace {
using sirius::scan_manager::group_bounds_arena;
using sirius::scan_manager::lowered_bound_filter;
using sirius::scan_manager::packed_column_bounds;

/// A one-column, one-chunk arena from explicit per-group [min, max] cells; nullopt = no
/// statistics for that group. Builds through from_capture so tests exercise the real packing.
group_bounds_arena make_arena(std::vector<std::optional<std::pair<int64_t, int64_t>>> const& cells,
                              std::size_t group_rows = 100)
{
  sirius::scan_manager::chunk_group_stats cs;
  cs.group_rows = group_rows;
  cs.groups.resize(cells.size());
  for (std::size_t g = 0; g < cells.size(); ++g) {
    cs.groups[g].resize(1);
    if (!cells[g]) { continue; }
    auto st = duckdb::NumericStats::CreateUnknown(LogicalType::INTEGER);
    duckdb::NumericStats::SetMin(st, Value::INTEGER(static_cast<int32_t>(cells[g]->first)));
    duckdb::NumericStats::SetMax(st, Value::INTEGER(static_cast<int32_t>(cells[g]->second)));
    st.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
    cs.groups[g][0] = st.ToUnique();
  }
  std::vector<sirius::scan_manager::chunk_group_stats> capture;
  capture.push_back(std::move(cs));
  return group_bounds_arena::from_capture(duckdb::vector<LogicalType>{LogicalType::INTEGER},
                                          capture);
}

/// Stats shaped exactly as the capture builds them, for cross-checking.
duckdb::unique_ptr<duckdb::BaseStatistics> int_stats(int32_t lo, int32_t hi, bool has_null)
{
  auto s = duckdb::NumericStats::CreateUnknown(LogicalType::INTEGER);
  duckdb::NumericStats::SetMin(s, Value::INTEGER(lo));
  duckdb::NumericStats::SetMax(s, Value::INTEGER(hi));
  if (has_null) {
    s.SetHasNull();
  } else {
    s.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
  }
  return s.ToUnique();
}
}  // namespace

TEST_CASE("lowered_bound_filter - agrees with chunk_provably_empty on every input",
          "[pinned_chunk_stats]")
{
  // Every filter shape the allowlist admits, over bounds chosen to straddle each constant:
  // disjoint below, touching, containing, touching above, disjoint above, and degenerate.
  std::vector<duckdb::unique_ptr<duckdb::TableFilter>> filters;
  for (auto cmp_type : {ExpressionType::COMPARE_EQUAL,
                        ExpressionType::COMPARE_NOTEQUAL,
                        ExpressionType::COMPARE_LESSTHAN,
                        ExpressionType::COMPARE_LESSTHANOREQUALTO,
                        ExpressionType::COMPARE_GREATERTHAN,
                        ExpressionType::COMPARE_GREATERTHANOREQUALTO}) {
    for (int32_t c : {-5, 0, 10, 50}) {
      filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(cmp_type, Value::INTEGER(c)));
    }
  }
  filters.push_back(duckdb::make_uniq<duckdb::IsNullFilter>());
  filters.push_back(duckdb::make_uniq<duckdb::IsNotNullFilter>());
  {
    duckdb::vector<Value> vals{Value::INTEGER(3), Value::INTEGER(11), Value::INTEGER(99)};
    filters.push_back(duckdb::make_uniq<duckdb::InFilter>(std::move(vals)));
  }
  {  // AND(x >= 10, x <= 20)
    auto conj = duckdb::make_uniq<duckdb::ConjunctionAndFilter>();
    conj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(10)));
    conj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_LESSTHANOREQUALTO, Value::INTEGER(20)));
    filters.push_back(std::move(conj));
  }
  {  // OR(x < 0, x > 40)
    auto disj = duckdb::make_uniq<duckdb::ConjunctionOrFilter>();
    disj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(0)));
    disj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(40)));
    filters.push_back(std::move(disj));
  }
  {  // OPTIONAL(x = 10) — an optional wrapping one child
    auto inner =
      duckdb::make_uniq<duckdb::ConstantFilter>(ExpressionType::COMPARE_EQUAL, Value::INTEGER(10));
    filters.push_back(duckdb::make_uniq<duckdb::OptionalFilter>(std::move(inner)));
  }
  {  // Nested: AND(OR(x < 0, x > 40), x != 50)
    auto disj = duckdb::make_uniq<duckdb::ConjunctionOrFilter>();
    disj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(0)));
    disj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(40)));
    auto conj = duckdb::make_uniq<duckdb::ConjunctionAndFilter>();
    conj->child_filters.push_back(std::move(disj));
    conj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_NOTEQUAL, Value::INTEGER(50)));
    filters.push_back(std::move(conj));
  }

  std::vector<std::pair<int32_t, int32_t>> ranges{{-100, -50},
                                                  {-10, -1},
                                                  {-5, -5},
                                                  {0, 0},
                                                  {0, 10},
                                                  {5, 5},
                                                  {10, 10},
                                                  {10, 20},
                                                  {11, 39},
                                                  {40, 100},
                                                  {50, 50},
                                                  {-100, 100}};

  std::size_t compared = 0, disagreements = 0;
  for (auto const& f : filters) {
    auto lowered = lowered_bound_filter::lower(*f, LogicalType::INTEGER);
    REQUIRE(lowered.has_value());  // every shape above is on the allowlist
    for (auto [lo, hi] : ranges) {
      for (bool has_null : {false, true}) {
        auto stats           = int_stats(lo, hi, has_null);
        bool const reference = sirius::scan_manager::chunk_provably_empty(*f, *stats);
        bool const fast      = lowered->provably_empty(lo, hi, has_null, false);
        ++compared;
        if (reference != fast) {
          ++disagreements;
          UNSCOPED_INFO("filter type " << static_cast<int>(f->filter_type) << " range [" << lo
                                       << "," << hi << "] has_null=" << has_null
                                       << " reference=" << reference << " fast=" << fast);
        }
      }
    }
  }
  INFO("compared " << compared << " (filter, range) pairs");
  REQUIRE(disagreements == 0);
  REQUIRE(compared > 300);
}

TEST_CASE("lowered_bound_filter - rejects what the allowlist rejects", "[pinned_chunk_stats]")
{
  SECTION("type mismatch between constant and stats type")
  {
    auto f =
      duckdb::make_uniq<duckdb::ConstantFilter>(ExpressionType::COMPARE_EQUAL, Value::BIGINT(10));
    REQUIRE_FALSE(lowered_bound_filter::lower(*f, LogicalType::INTEGER).has_value());
  }
  SECTION("a childless conjunction")
  {
    auto conj = duckdb::make_uniq<duckdb::ConjunctionAndFilter>();
    REQUIRE_FALSE(lowered_bound_filter::lower(*conj, LogicalType::INTEGER).has_value());
  }
  SECTION("a dynamic filter is never lowered")
  {
    auto dyn = duckdb::make_uniq<duckdb::DynamicFilter>();
    REQUIRE_FALSE(lowered_bound_filter::lower(*dyn, LogicalType::INTEGER).has_value());
  }
}

TEST_CASE("lowered_bound_filter - select_survivors over packed cells", "[pinned_chunk_stats]")
{
  //                  g0        g1          g2        g3 (no statistics)
  std::vector<std::optional<std::pair<int64_t, int64_t>>> const cells{
    std::pair<int64_t, int64_t>{0, 9},
    std::pair<int64_t, int64_t>{100, 109},
    std::pair<int64_t, int64_t>{200, 209},
    std::nullopt};
  auto const arena = make_arena(cells);
  auto const b     = arena.cell(0, 0);

  auto f       = duckdb::make_uniq<duckdb::ConstantFilter>(ExpressionType::COMPARE_LESSTHAN,
                                                     Value::INTEGER(150));
  auto lowered = lowered_bound_filter::lower(*f, LogicalType::INTEGER);
  REQUIRE(lowered.has_value());

  std::vector<std::uint32_t> survivors;
  lowered->select_survivors(b, survivors);
  // g0 and g1 can contain values < 150; g2 cannot; g3 has no statistics so it must survive.
  REQUIRE(survivors == std::vector<std::uint32_t>{0, 1, 3});
}

// ---------------------------------------------------------------------------
// build_cached_scan_plan — refining surviving chunks into surviving row ranges.
//
// The coarse pass keeps a chunk whose min/max overlaps the filter at all. Per-group bounds then
// say WHICH rows of that chunk can match. The properties that matter to every consumer:
// refinement only ever narrows an already-surviving chunk, adjacent surviving groups coalesce so
// an unprunable chunk costs nothing, and an entry with no group bounds keeps the old whole-chunk
// behaviour.
// ---------------------------------------------------------------------------

namespace {

/// A one-column, one-chunk entry whose chunk min/max spans all its groups, so the coarse pass
/// always keeps the chunk and only the refinement can narrow it.
pinned_entry make_group_entry(std::size_t n_groups, std::size_t group_rows)
{
  pinned_entry entry;
  entry.cache_info.column_ids.emplace_back(duckdb::ColumnIndex(3));
  entry.cache_info.names.push_back("c0");
  entry.tier = cucascade::memory::Tier::HOST;
  entry.host_chunks.resize(1);

  auto const lo = 0;
  auto const hi = static_cast<int32_t>(10 * n_groups - 1);
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> capture(1);
  capture[0].push_back(
    make_stats(LogicalType::INTEGER, Value::INTEGER(lo), Value::INTEGER(hi)).ToUnique());
  entry.zone_maps = pinned_zone_maps::from_capture(int_types(1), std::move(capture), 1, 1);

  // group g covers [10*g, 10*g + 9]: disjoint and strictly increasing, so a filter's cut point
  // is unambiguous.
  std::vector<std::optional<std::pair<int64_t, int64_t>>> cells;
  for (std::size_t g = 0; g < n_groups; ++g) {
    cells.emplace_back(
      std::pair<int64_t, int64_t>{static_cast<int64_t>(10 * g), static_cast<int64_t>(10 * g) + 9});
  }
  entry.group_bounds = make_arena(cells, group_rows);
  return entry;
}
}  // namespace

TEST_CASE("build_cached_scan_plan - refines surviving chunks into row ranges",
          "[pinned_chunk_stats]")
{
  duckdb::vector<duckdb::ColumnIndex> qcols{duckdb::ColumnIndex(3)};

  SECTION("a filter cutting mid-chunk yields the surviving prefix only")
  {
    // 8 groups of 100 rows spanning [0,79]; "< 25" can only match groups 0..2.
    auto const entry = make_group_entry(8, 100);
    auto fs   = make_filter_set(0, cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(25)));
    auto plan = build_cached_scan_plan(entry, &fs, &qcols);

    REQUIRE(plan.survivor_chunk_indices == survivors_t{0});  // chunk still survives coarsely
    REQUIRE(plan.survivor_row_ranges.size() == 1);
    REQUIRE(plan.survivor_row_ranges[0] == sirius::scan_manager::chunk_row_range{0, 0, 300});
    REQUIRE(plan.rows_pruned_within_chunks == 500);
  }

  SECTION("disjoint surviving groups produce separate ranges")
  {
    // OR(< 5, > 74) keeps group 0 and group 7 only.
    auto const entry = make_group_entry(8, 100);
    auto disj        = duckdb::make_uniq<duckdb::ConjunctionOrFilter>();
    disj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(5)));
    disj->child_filters.push_back(duckdb::make_uniq<duckdb::ConstantFilter>(
      ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(74)));
    auto fs   = make_filter_set(0, std::move(disj));
    auto plan = build_cached_scan_plan(entry, &fs, &qcols);

    REQUIRE(plan.survivor_row_ranges.size() == 2);
    REQUIRE(plan.survivor_row_ranges[0] == sirius::scan_manager::chunk_row_range{0, 0, 100});
    REQUIRE(plan.survivor_row_ranges[1] == sirius::scan_manager::chunk_row_range{0, 700, 800});
    REQUIRE(plan.rows_pruned_within_chunks == 600);
  }

  SECTION("an unprunable filter coalesces to exactly one range covering the chunk")
  {
    auto const entry = make_group_entry(8, 100);
    auto fs =
      make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHANOREQUALTO, Value::INTEGER(0)));
    auto plan = build_cached_scan_plan(entry, &fs, &qcols);

    REQUIRE(plan.survivor_row_ranges.size() == 1);
    REQUIRE(plan.survivor_row_ranges[0] == sirius::scan_manager::chunk_row_range{0, 0, 800});
    REQUIRE(plan.rows_pruned_within_chunks == 0);
  }

  SECTION("an entry without group bounds keeps the whole-chunk behaviour")
  {
    auto entry         = make_group_entry(8, 100);
    entry.group_bounds = sirius::scan_manager::group_bounds_arena{};
    auto fs   = make_filter_set(0, cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(25)));
    auto plan = build_cached_scan_plan(entry, &fs, &qcols);

    REQUIRE(plan.survivor_chunk_indices == survivors_t{0});
    REQUIRE(plan.survivor_row_ranges.empty());
    REQUIRE(plan.rows_pruned_within_chunks == 0);
  }

  SECTION("an absent group cell keeps its rows")
  {
    auto entry = make_group_entry(8, 100);
    // Rebuild with group 5 carrying no statistics.
    std::vector<std::optional<std::pair<int64_t, int64_t>>> cells;
    for (std::size_t g = 0; g < 8; ++g) {
      if (g == 5) {
        cells.emplace_back(std::nullopt);
      } else {
        cells.emplace_back(std::pair<int64_t, int64_t>{static_cast<int64_t>(10 * g),
                                                       static_cast<int64_t>(10 * g) + 9});
      }
    }
    entry.group_bounds = make_arena(cells, 100);
    auto fs   = make_filter_set(0, cmp(ExpressionType::COMPARE_LESSTHAN, Value::INTEGER(25)));
    auto plan = build_cached_scan_plan(entry, &fs, &qcols);

    // Groups 0..2 match the filter; group 5 has no evidence, so it must survive too.
    REQUIRE(plan.survivor_row_ranges.size() == 2);
    REQUIRE(plan.survivor_row_ranges[0] == sirius::scan_manager::chunk_row_range{0, 0, 300});
    REQUIRE(plan.survivor_row_ranges[1] == sirius::scan_manager::chunk_row_range{0, 500, 600});
  }

  SECTION("the all-pruned sentinel chunk is never refined")
  {
    // A filter no group can satisfy prunes the chunk coarsely; the sentinel that keeps the
    // pipeline alive must not then be narrowed to nothing.
    auto const entry = make_group_entry(8, 100);
    auto fs   = make_filter_set(0, cmp(ExpressionType::COMPARE_GREATERTHAN, Value::INTEGER(10000)));
    auto plan = build_cached_scan_plan(entry, &fs, &qcols);

    REQUIRE(plan.survivor_chunk_indices == survivors_t{0});
    REQUIRE(plan.survivor_row_ranges.empty());
    REQUIRE(plan.rows_pruned_within_chunks == 0);
  }
}

TEST_CASE("surviving_decode_chunks maps row ranges onto 1024-row decode chunks",
          "[pinned_chunk_stats]")
{
  using sirius::scan_manager::chunk_row_range;
  using sirius::scan_manager::surviving_decode_chunk_rows;
  using sirius::scan_manager::surviving_decode_chunks;
  using ids_t = std::vector<std::uint32_t>;

  constexpr std::size_t kDecode = 1024;

  SECTION("a group-aligned range expands to its decode chunks")
  {
    // Groups of 8 decode chunks: rows [8192, 16384) is decode chunks 8..15.
    auto const ids = surviving_decode_chunks(std::vector<chunk_row_range>{{0, 8192, 16384}},
                                             /*rows=*/32768);
    REQUIRE(ids == ids_t{8, 9, 10, 11, 12, 13, 14, 15});
    REQUIRE(surviving_decode_chunk_rows(ids, 32768) == 8192);
  }

  SECTION("disjoint ranges stay ascending and do not repeat a chunk")
  {
    auto const ids = surviving_decode_chunks(
      std::vector<chunk_row_range>{{0, 0, 2 * kDecode}, {0, 4 * kDecode, 5 * kDecode}},
      /*rows=*/10 * kDecode);
    REQUIRE(ids == ids_t{0, 1, 4});
    REQUIRE(surviving_decode_chunk_rows(ids, 10 * kDecode) == 3 * kDecode);
  }

  SECTION("a range past the end of a short final group is clamped")
  {
    // The last group of a chunk may be short, so the plan's end may exceed the chunk.
    auto const rows = 3 * kDecode + 100;
    auto const ids =
      surviving_decode_chunks(std::vector<chunk_row_range>{{0, 2 * kDecode, 8 * kDecode}}, rows);
    REQUIRE(ids == ids_t{2, 3});
    // Chunk 3 holds only the 100 rows that exist.
    REQUIRE(surviving_decode_chunk_rows(ids, rows) == kDecode + 100);
  }

  SECTION("covering every chunk asks for no subset at all")
  {
    // Empty means "serve the chunk whole" — synthesizing a header that reproduces the original
    // is pure cost.
    REQUIRE(surviving_decode_chunks(std::vector<chunk_row_range>{{0, 0, 4 * kDecode}}, 4 * kDecode)
              .empty());
    REQUIRE(surviving_decode_chunks({}, 4 * kDecode).empty());
    REQUIRE(surviving_decode_chunks(std::vector<chunk_row_range>{{0, 0, 100}}, 0).empty());
  }
}

// ---------------------------------------------------------------------------
// group_bounds_arena — the contiguous, column-major packing.
//
// The layout is the point: one allocation that can be read (or copied to device, or written to a
// file) as a unit, without touching the data it describes. These tests pin the properties that
// makes possible — contiguity, column-major ordering, and refusing to build anything inconsistent
// rather than building something subtly wrong.
// ---------------------------------------------------------------------------

namespace {
/// A capture of @p n_chunks chunks x @p n_columns columns, where column i of chunk c group g gets
/// [base, base+1] with base = 1000*c + 100*i + g. Every cell is therefore distinguishable, so a
/// misplaced one fails loudly rather than coincidentally matching.
std::vector<sirius::scan_manager::chunk_group_stats> make_multi_capture(
  std::size_t n_chunks, std::size_t n_columns, std::size_t n_groups, std::size_t group_rows = 100)
{
  std::vector<sirius::scan_manager::chunk_group_stats> capture(n_chunks);
  for (std::size_t c = 0; c < n_chunks; ++c) {
    capture[c].group_rows = group_rows;
    capture[c].groups.resize(n_groups);
    for (std::size_t g = 0; g < n_groups; ++g) {
      capture[c].groups[g].resize(n_columns);
      for (std::size_t i = 0; i < n_columns; ++i) {
        auto const base = static_cast<int32_t>(1000 * c + 100 * i + g);
        auto st         = duckdb::NumericStats::CreateUnknown(LogicalType::INTEGER);
        duckdb::NumericStats::SetMin(st, Value::INTEGER(base));
        duckdb::NumericStats::SetMax(st, Value::INTEGER(base + 1));
        st.Set(duckdb::StatsInfo::CANNOT_HAVE_NULL_VALUES);
        capture[c].groups[g][i] = st.ToUnique();
      }
    }
  }
  return capture;
}
}  // namespace

TEST_CASE("group_bounds_arena - packs column-major and addresses every cell",
          "[pinned_chunk_stats]")
{
  constexpr std::size_t kChunks = 3, kColumns = 2, kGroups = 4;
  auto const arena = group_bounds_arena::from_capture(
    int_types(kColumns), make_multi_capture(kChunks, kColumns, kGroups));
  REQUIRE_FALSE(arena.empty());
  REQUIRE(arena.column_count() == kColumns);
  REQUIRE(arena.chunk_count() == kChunks);
  REQUIRE(arena.group_rows() == 100);
  REQUIRE(arena.groups_in_chunk(0) == kGroups);

  SECTION("every (column, chunk, group) reads back the value it was given")
  {
    for (std::size_t c = 0; c < kChunks; ++c) {
      for (std::size_t i = 0; i < kColumns; ++i) {
        auto const cell = arena.cell(i, c);
        REQUIRE(cell.size() == kGroups);
        for (std::size_t g = 0; g < kGroups; ++g) {
          auto const base = static_cast<int64_t>(1000 * c + 100 * i + g);
          REQUIRE(cell.mins[g] == base);
          REQUIRE(cell.maxs[g] == base + 1);
          REQUIRE(cell.valid[g] == 1);
        }
      }
    }
  }

  SECTION("one column's chunks are contiguous, which is what column-major buys")
  {
    // Column 0's chunk 0 and chunk 1 must be adjacent in the arena: chunk 0 occupies
    // [mins | maxs], so chunk 1's mins start immediately after.
    auto const c0 = arena.cell(0, 0);
    auto const c1 = arena.cell(0, 1);
    REQUIRE(c1.mins.data() == c0.maxs.data() + c0.maxs.size());
  }

  SECTION("the whole arena is one contiguous allocation")
  {
    auto const raw = arena.raw();
    REQUIRE(raw.size() == kChunks * kColumns * kGroups * 2);  // mins + maxs
    auto const first = arena.cell(0, 0);
    auto const last  = arena.cell(kColumns - 1, kChunks - 1);
    REQUIRE(first.mins.data() == raw.data());
    REQUIRE(last.maxs.data() + last.maxs.size() == raw.data() + raw.size());
  }

  SECTION("out-of-range lookups return an empty cell rather than reading past the arena")
  {
    REQUIRE(arena.cell(kColumns, 0).empty());
    REQUIRE(arena.cell(0, kChunks).empty());
    REQUIRE(arena.groups_in_chunk(kChunks) == 0);
  }
}

TEST_CASE("group_bounds_arena - refuses inconsistent captures", "[pinned_chunk_stats]")
{
  // Each of these would otherwise produce an arena that silently describes the wrong rows, so
  // from_capture returns empty: no sub-chunk pruning, never a wrong one.
  SECTION("a chunk whose cells disagree with the column count")
  {
    auto capture = make_multi_capture(2, 2, 3);
    capture[1].groups[0].resize(1);  // one column short
    REQUIRE(group_bounds_arena::from_capture(int_types(2), capture).empty());
  }
  SECTION("chunks that disagree about group_rows")
  {
    auto capture          = make_multi_capture(2, 1, 3);
    capture[1].group_rows = 200;
    REQUIRE(group_bounds_arena::from_capture(int_types(1), capture).empty());
  }
  SECTION("a zero group_rows")
  {
    auto capture          = make_multi_capture(1, 1, 3);
    capture[0].group_rows = 0;
    REQUIRE(group_bounds_arena::from_capture(int_types(1), capture).empty());
  }
  SECTION("an empty capture or no columns")
  {
    REQUIRE(group_bounds_arena::from_capture(int_types(1), {}).empty());
    REQUIRE(group_bounds_arena::from_capture({}, make_multi_capture(1, 1, 3)).empty());
  }
}

TEST_CASE("group_bounds_arena - chunks may differ in group count", "[pinned_chunk_stats]")
{
  // The last chunk of a pin is usually short, so its group count legitimately differs.
  auto capture = make_multi_capture(2, 1, 4);
  capture[1].groups.resize(2);
  auto const arena = group_bounds_arena::from_capture(int_types(1), capture);
  REQUIRE_FALSE(arena.empty());
  REQUIRE(arena.groups_in_chunk(0) == 4);
  REQUIRE(arena.groups_in_chunk(1) == 2);
  REQUIRE(arena.cell(0, 1).size() == 2);
  REQUIRE(arena.cell(0, 1).mins[0] == 1000);  // chunk 1, column 0, group 0
}
