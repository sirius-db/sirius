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

// Boundary semantics of analyze_scan_filters on the EXPRESSION_FILTER shape
// `CAST(date_col AS TIMESTAMP) CMP <timestamp constant>`.
//
// Off-by-one-day correctness is the whole game: a fully-covered extraction lets
// the scan skip its residual filter, so [lo, hi] must keep EXACTLY the rows the
// SQL predicate keeps. Each case states the day-domain expectation that follows
// from midnight(d) = d * ticks_per_day being strictly monotonic in d.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/interval.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <op/scan/scan_filter_analysis.hpp>

#include <cstdint>
#include <limits>
#include <utility>

using duckdb::ExpressionType;

namespace {

constexpr std::int64_t kDayMicros = duckdb::Interval::MICROS_PER_DAY;
constexpr std::int64_t kInt64Min  = std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kInt64Max  = std::numeric_limits<std::int64_t>::max();

/// Stored day count of 1998-09-20 (q1's qgen cutoff for the measured regression).
std::int64_t cutoff_days() { return duckdb::Date::FromDate(1998, 9, 20).days; }

duckdb::unique_ptr<duckdb::Expression> date_col_ref()
{
  return duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::DATE, 0ULL);
}

duckdb::unique_ptr<duckdb::Expression> cast_expr(
  duckdb::unique_ptr<duckdb::Expression> child,
  duckdb::LogicalType target = duckdb::LogicalType::TIMESTAMP,
  bool try_cast              = false)
{
  return duckdb::BoundCastExpression::AddDefaultCastToType(std::move(child), target, try_cast);
}

duckdb::unique_ptr<duckdb::Expression> ts_const(std::int64_t micros)
{
  return duckdb::make_uniq<duckdb::BoundConstantExpression>(
    duckdb::Value::TIMESTAMP(duckdb::timestamp_t(micros)));
}

/// `CAST(col AS TIMESTAMP) CMP <micros>` as an EXPRESSION_FILTER; @p const_on_left builds
/// the mirrored `<micros> CMP CAST(col AS TIMESTAMP)` shape instead.
duckdb::unique_ptr<duckdb::TableFilter> cast_cmp_filter(ExpressionType cmp,
                                                        std::int64_t micros,
                                                        bool const_on_left = false)
{
  auto cast = cast_expr(date_col_ref());
  auto cst  = ts_const(micros);
  auto expr =
    const_on_left
      ? duckdb::make_uniq<duckdb::BoundComparisonExpression>(cmp, std::move(cst), std::move(cast))
      : duckdb::make_uniq<duckdb::BoundComparisonExpression>(cmp, std::move(cast), std::move(cst));
  return duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr));
}

sirius::op::scan_filter_analysis run_extraction(
  duckdb::unique_ptr<duckdb::TableFilter> filter,
  sirius::logical_type col_type = sirius::logical_type::make(sirius::type_id::DATE))
{
  duckdb::TableFilterSet filters;
  filters.PushFilter(duckdb::ColumnIndex(0), std::move(filter));
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(0ULL);
  duckdb::vector<sirius::logical_type> types;
  types.push_back(std::move(col_type));
  return sirius::op::analyze_scan_filters(filters, column_ids, types);
}

void expect_range(sirius::op::scan_filter_analysis const& r,
                  std::int64_t lo,
                  std::int64_t hi,
                  bool covered = true)
{
  REQUIRE(r.ranges.size() == 1);
  REQUIRE(r.ranges.count(0) == 1);
  CHECK(r.ranges.at(0).lo == lo);
  CHECK(r.ranges.at(0).hi == hi);
  CHECK(r.ranges_cover_whole_filter == covered);
}

void expect_refusal(sirius::op::scan_filter_analysis const& r)
{
  CHECK(r.ranges.empty());
  CHECK_FALSE(r.ranges_cover_whole_filter);
}

}  // namespace

// ── Midnight constants: the cast comparison is a plain day comparison ────────

TEST_CASE("cast-through range extraction: midnight timestamp constants are exact day bounds",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k        = cutoff_days();
  auto const midnight = k * kDayMicros;

  SECTION("< midnight(k) keeps days <= k-1")
  {
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, midnight)),
                 kInt64Min,
                 k - 1);
  }
  SECTION("<= midnight(k) keeps days <= k  (q1's folded qgen predicate)")
  {
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, midnight)),
      kInt64Min,
      k);
  }
  SECTION("> midnight(k) keeps days >= k+1")
  {
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHAN, midnight)),
                 k + 1,
                 kInt64Max);
  }
  SECTION(">= midnight(k) keeps days >= k")
  {
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO, midnight)),
      k,
      kInt64Max);
  }
  SECTION("= midnight(k) keeps exactly day k")
  {
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_EQUAL, midnight)), k, k);
  }
}

// ── Non-midnight constants: instants fall strictly between two days ─────────

TEST_CASE("cast-through range extraction: non-midnight constants stay exact (floor/ceil sides)",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k    = cutoff_days();
  auto const noon = k * kDayMicros + 12 * duckdb::Interval::MICROS_PER_HOUR;

  SECTION("< noon(k): midnight(k) < noon, so day k still passes — hi = k")
  {
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, noon)), kInt64Min, k);
  }
  SECTION("<= noon(k): hi = k")
  {
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, noon)),
                 kInt64Min,
                 k);
  }
  SECTION("> noon(k): day k's midnight is before noon — lo = k+1")
  {
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHAN, noon)), k + 1, kInt64Max);
  }
  SECTION(">= noon(k): lo = k+1")
  {
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO, noon)),
      k + 1,
      kInt64Max);
  }
  SECTION("= noon(k): no midnight equals a non-midnight instant — provably empty, still covered")
  {
    auto const r = run_extraction(cast_cmp_filter(ExpressionType::COMPARE_EQUAL, noon));
    REQUIRE(r.ranges.size() == 1);
    CHECK(r.ranges.at(0).lo > r.ranges.at(0).hi);  // canonical {0, -1}
    CHECK(r.ranges_cover_whole_filter);
  }

  SECTION("one microsecond after midnight(k)")
  {
    // < k*day+1us keeps day k; >= k*day+1us starts at k+1.
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, k * kDayMicros + 1)),
      kInt64Min,
      k);
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                                k * kDayMicros + 1)),
                 k + 1,
                 kInt64Max);
  }
  SECTION("one microsecond before midnight(k)")
  {
    // <= k*day-1us ends at k-1; > k*day-1us starts at k (midnight(k) > k*day-1us).
    expect_range(run_extraction(
                   cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, k * kDayMicros - 1)),
                 kInt64Min,
                 k - 1);
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHAN, k * kDayMicros - 1)),
      k,
      kInt64Max);
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, k * kDayMicros - 1)),
      kInt64Min,
      k - 1);
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                                k * kDayMicros - 1)),
                 k,
                 kInt64Max);
  }
}

// ── Pre-epoch (negative day) constants: floor must round toward -inf ────────

TEST_CASE("cast-through range extraction: pre-epoch instants floor toward negative infinity",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const d = duckdb::Date::FromDate(1969, 12, 31).days;  // -1
  REQUIRE(d == -1);

  SECTION("midnight of day -1")
  {
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, d * kDayMicros)),
      kInt64Min,
      -1);
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, d * kDayMicros)),
                 kInt64Min,
                 -2);
  }
  SECTION("one microsecond after midnight of day -1 (still inside day -1)")
  {
    // C++ integer division truncates toward zero; the bound must floor instead.
    expect_range(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, d * kDayMicros + 1)),
      kInt64Min,
      -1);
    expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                                d * kDayMicros + 1)),
                 0,
                 kInt64Max);
  }
}

// ── Operand order: constant on the left flips the comparison ────────────────

TEST_CASE("cast-through range extraction: constant-on-the-left shapes flip correctly",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k        = cutoff_days();
  auto const midnight = k * kDayMicros;
  auto const noon     = midnight + 12 * duckdb::Interval::MICROS_PER_HOUR;

  // T >= CAST(d)  ⇔  CAST(d) <= T
  expect_range(
    run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO, midnight, true)),
    kInt64Min,
    k);
  // T < CAST(d)  ⇔  CAST(d) > T
  expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, midnight, true)),
               k + 1,
               kInt64Max);
  // noon > CAST(d)  ⇔  CAST(d) < noon  ⇒  hi = k
  expect_range(
    run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHAN, noon, true)), kInt64Min, k);
  // T = CAST(d) is symmetric
  expect_range(
    run_extraction(cast_cmp_filter(ExpressionType::COMPARE_EQUAL, midnight, true)), k, k);
}

// ── Conjunctions: inside one ExpressionFilter and across table filters ──────

TEST_CASE("cast-through range extraction: AND shapes intersect into one range",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const lo_days = duckdb::Date::FromDate(1994, 1, 1).days;
  auto const hi_days = duckdb::Date::FromDate(1995, 1, 1).days;

  SECTION("BoundConjunctionExpression inside a single ExpressionFilter (q6 shape)")
  {
    auto ge = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHANOREQUALTO,
      cast_expr(date_col_ref()),
      ts_const(lo_days * kDayMicros));
    auto lt = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHAN, cast_expr(date_col_ref()), ts_const(hi_days * kDayMicros));
    auto conj =
      duckdb::make_uniq<duckdb::BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
    conj->children.push_back(std::move(ge));
    conj->children.push_back(std::move(lt));
    expect_range(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(conj))),
                 lo_days,
                 hi_days - 1);
  }
  SECTION("ConjunctionAndFilter of two ExpressionFilters")
  {
    auto conj = duckdb::make_uniq<duckdb::ConjunctionAndFilter>();
    conj->child_filters.push_back(
      cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO, lo_days * kDayMicros));
    conj->child_filters.push_back(
      cast_cmp_filter(ExpressionType::COMPARE_LESSTHAN, hi_days * kDayMicros));
    expect_range(run_extraction(std::move(conj)), lo_days, hi_days - 1);
  }
}

// ── The plain ConstantFilter path also accepts timestamp constants now ──────

TEST_CASE("to_decoded_bound: timestamp ConstantFilter against a DATE column",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k = cutoff_days();
  auto filter  = duckdb::make_uniq<duckdb::ConstantFilter>(
    ExpressionType::COMPARE_LESSTHANOREQUALTO,
    duckdb::Value::TIMESTAMP(duckdb::timestamp_t(k * kDayMicros)));
  expect_range(run_extraction(std::move(filter)), kInt64Min, k);
}

// ── Timestamp flavors: S / MS / NS lower with their own day length ──────────

TEST_CASE("cast-through range extraction: TIMESTAMP_S / TIMESTAMP_MS / TIMESTAMP_NS flavors",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k = cutoff_days();

  struct flavor {
    duckdb::LogicalType type;
    std::int64_t ticks_per_day;
    duckdb::Value (*make)(std::int64_t);
  };
  auto const flavors = {
    flavor{duckdb::LogicalType::TIMESTAMP_S,
           duckdb::Interval::SECS_PER_DAY,
           [](std::int64_t t) { return duckdb::Value::TIMESTAMPSEC(duckdb::timestamp_sec_t(t)); }},
    flavor{duckdb::LogicalType::TIMESTAMP_MS,
           duckdb::Interval::SECS_PER_DAY * duckdb::Interval::MSECS_PER_SEC,
           [](std::int64_t t) { return duckdb::Value::TIMESTAMPMS(duckdb::timestamp_ms_t(t)); }},
    flavor{duckdb::LogicalType::TIMESTAMP_NS,
           duckdb::Interval::NANOS_PER_DAY,
           [](std::int64_t t) { return duckdb::Value::TIMESTAMPNS(duckdb::timestamp_ns_t(t)); }},
  };
  for (auto const& f : flavors) {
    DYNAMIC_SECTION("flavor " << f.type.ToString())
    {
      // midnight: <= keeps day k; one tick later: <= still keeps day k, >= starts at k+1.
      auto mk = [&](ExpressionType cmp, std::int64_t ticks) {
        auto expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
          cmp,
          cast_expr(date_col_ref(), f.type),
          duckdb::make_uniq<duckdb::BoundConstantExpression>(f.make(ticks)));
        return duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr));
      };
      expect_range(
        run_extraction(mk(ExpressionType::COMPARE_LESSTHANOREQUALTO, k * f.ticks_per_day)),
        kInt64Min,
        k);
      expect_range(run_extraction(mk(ExpressionType::COMPARE_LESSTHAN, k * f.ticks_per_day + 1)),
                   kInt64Min,
                   k);
      expect_range(
        run_extraction(mk(ExpressionType::COMPARE_GREATERTHANOREQUALTO, k * f.ticks_per_day + 1)),
        k + 1,
        kInt64Max);
    }
  }
}

// ── Extreme finite constants stay well-defined ───────────────────────────────

TEST_CASE("cast-through range extraction: extreme finite timestamps",
          "[scan][range_pushdown][fused_scan_filter]")
{
  // Largest finite micros (int64max is +infinity, int64max-1 is a valid instant).
  auto const t_hi = kInt64Max - 1;
  expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, t_hi)),
               kInt64Min,
               t_hi / kDayMicros);
  // Most negative finite micros (-int64max is -infinity; one above it is finite).
  auto const t_lo = -kInt64Max + 1;
  // floor(t_lo / day) = trunc - 1 because t_lo is negative with a remainder.
  auto const floor_lo = t_lo / kDayMicros - 1;
  expect_range(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO, t_lo)),
               floor_lo + 1,  // ceil
               kInt64Max);
}

// ── Refusals: coverage cleared, no range extracted ───────────────────────────

TEST_CASE("cast-through range extraction: refused shapes keep the residual filter",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k        = cutoff_days();
  auto const midnight = k * kDayMicros;

  SECTION("TRY_CAST is not range-expressible (NULL-on-overflow vs day math)")
  {
    auto expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHANOREQUALTO,
      cast_expr(date_col_ref(), duckdb::LogicalType::TIMESTAMP, /*try_cast=*/true),
      ts_const(midnight));
    expect_refusal(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr))));
  }
  SECTION("TIMESTAMP_TZ: midnight depends on the session time zone")
  {
    auto expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHANOREQUALTO,
      cast_expr(date_col_ref(), duckdb::LogicalType::TIMESTAMP_TZ),
      duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value::TIMESTAMPTZ(duckdb::timestamp_tz_t(midnight))));
    expect_refusal(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr))));
  }
  SECTION("infinite constants: DATE ±infinity does not follow the linear day mapping")
  {
    expect_refusal(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO,
                                                  duckdb::timestamp_t::infinity().value)));
    expect_refusal(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                                  duckdb::timestamp_t::ninfinity().value)));
    expect_refusal(run_extraction(
      cast_cmp_filter(ExpressionType::COMPARE_EQUAL, duckdb::timestamp_t::infinity().value)));
  }
  SECTION("NULL constant")
  {
    auto expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHANOREQUALTO,
      cast_expr(date_col_ref()),
      duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(duckdb::LogicalType::TIMESTAMP)));
    expect_refusal(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr))));
  }
  SECTION("<> is not a range")
  {
    expect_refusal(run_extraction(cast_cmp_filter(ExpressionType::COMPARE_NOTEQUAL, midnight)));
  }
  SECTION("cast around something that is not the column placeholder")
  {
    auto expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHANOREQUALTO,
      cast_expr(duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value::DATE(duckdb::Date::FromDate(1998, 9, 20)))),
      ts_const(midnight));
    expect_refusal(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr))));
  }
  SECTION("no cast side at all (constant vs constant)")
  {
    auto expr = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHANOREQUALTO, ts_const(midnight), ts_const(midnight));
    expect_refusal(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(expr))));
  }
  SECTION("OR conjunction inside the expression")
  {
    auto a = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_LESSTHAN, cast_expr(date_col_ref()), ts_const(midnight));
    auto b = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
      ExpressionType::COMPARE_GREATERTHAN, cast_expr(date_col_ref()), ts_const(midnight));
    auto conj =
      duckdb::make_uniq<duckdb::BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR);
    conj->children.push_back(std::move(a));
    conj->children.push_back(std::move(b));
    expect_refusal(run_extraction(duckdb::make_uniq<duckdb::ExpressionFilter>(std::move(conj))));
  }
  SECTION("cast shape on a non-DATE column type is refused")
  {
    expect_refusal(
      run_extraction(cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, midnight),
                     sirius::logical_type::make(sirius::type_id::INTEGER)));
  }
}

// ── Mixed filter sets: partial coverage keeps convertible ranges ────────────

TEST_CASE("cast-through range extraction: unconvertible sibling clears coverage but keeps ranges",
          "[scan][range_pushdown][fused_scan_filter]")
{
  auto const k = cutoff_days();

  duckdb::TableFilterSet filters;
  filters.PushFilter(duckdb::ColumnIndex(0),
                     cast_cmp_filter(ExpressionType::COMPARE_LESSTHANOREQUALTO, k * kDayMicros));
  // An unconvertible shape on another column: `CAST(d2 AS TIMESTAMP) <> T`.
  filters.PushFilter(duckdb::ColumnIndex(1),
                     cast_cmp_filter(ExpressionType::COMPARE_NOTEQUAL, k * kDayMicros));

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(0ULL);
  column_ids.emplace_back(1ULL);
  duckdb::vector<sirius::logical_type> types;
  types.push_back(sirius::logical_type::make(sirius::type_id::DATE));
  types.push_back(sirius::logical_type::make(sirius::type_id::DATE));

  auto const r = sirius::op::analyze_scan_filters(filters, column_ids, types);
  REQUIRE(r.ranges.size() == 1);
  CHECK(r.ranges.at(0).hi == k);
  CHECK_FALSE(r.ranges_cover_whole_filter);
}
