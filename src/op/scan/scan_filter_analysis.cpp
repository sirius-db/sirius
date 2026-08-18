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

// sirius
#include <duckdb/common/types/value.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <expression/ast/constant_range.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/utils.hpp>
#include <expression/value.hpp>
#include <helper/type_conversions.hpp>
#include <log/logging.hpp>
#include <op/scan/scan_filter_analysis.hpp>

// standard library
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::op {

namespace {

//===----------------------------------------------------------------------===//
// Equality / IN sets
//===----------------------------------------------------------------------===//

/// Append @p value's string payload to @p out. False (leaving @p out untouched)
/// for a null or non-VARCHAR constant: a null never equals anything, and a
/// non-string constant means the filter is not the shape we can push down.
bool append_string_constant(duckdb::Value const& value, std::vector<std::string>& out)
{
  if (value.IsNull() || value.type().id() != duckdb::LogicalTypeId::VARCHAR) { return false; }
  out.push_back(duckdb::StringValue::Get(value));
  return true;
}

/// Collect the value set @p filter tests for equality against, or false if it is
/// any other shape. Recursive so `x = 'a' OR x = 'b'` and an ANDed IS NOT NULL
/// both resolve.
bool collect_equality_values(duckdb::TableFilter const& filter, std::vector<std::string>& out)
{
  switch (filter.filter_type) {
    case duckdb::TableFilterType::CONSTANT_COMPARISON: {
      auto const& cmp = filter.Cast<duckdb::ConstantFilter>();
      if (cmp.comparison_type != duckdb::ExpressionType::COMPARE_EQUAL) { return false; }
      return append_string_constant(cmp.constant, out);
    }
    case duckdb::TableFilterType::IN_FILTER: {
      auto const& in = filter.Cast<duckdb::InFilter>();
      if (in.values.empty()) { return false; }
      for (auto const& value : in.values) {
        if (!append_string_constant(value, out)) { return false; }
      }
      return true;
    }
    case duckdb::TableFilterType::CONJUNCTION_OR: {
      // Every branch must contribute, or the union would under-approximate.
      auto const& disjunction = filter.Cast<duckdb::ConjunctionOrFilter>();
      if (disjunction.child_filters.empty()) { return false; }
      for (auto const& child : disjunction.child_filters) {
        if (!collect_equality_values(*child, out)) { return false; }
      }
      return true;
    }
    case duckdb::TableFilterType::CONJUNCTION_AND: {
      // Only the redundant IS NOT NULL may accompany the equality: an equality
      // against a non-null constant is already false/null for a null row, so
      // absorbing it does not change which rows survive. Any other conjunct
      // would narrow the result further than the value set describes.
      auto const& conjunction = filter.Cast<duckdb::ConjunctionAndFilter>();
      bool found              = false;
      for (auto const& child : conjunction.child_filters) {
        if (child->filter_type == duckdb::TableFilterType::IS_NOT_NULL) { continue; }
        if (found) { return false; }  // two value-bearing conjuncts: not a plain equality
        if (!collect_equality_values(*child, out)) { return false; }
        found = true;
      }
      return found;
    }
    default: return false;
  }
}

//===----------------------------------------------------------------------===//
// Numeric ranges
//===----------------------------------------------------------------------===//

/// Wide intermediate for bound arithmetic: rescaling a decimal constant and the
/// ±1 of strict inequalities can step just past the int64 edge, so bounds are
/// intersected as __int128 and clamped once at the end.
using int128 = __int128;

/// 10^e as int128 (e ≤ 38 fits; callers never exceed decimal precision bounds).
int128 pow10_128(int e)
{
  int128 r = 1;
  while (e-- > 0) {
    r *= 10;
  }
  return r;
}

/// The constant a conjunct compares against, lowered into the DECODED integer
/// domain of a column of @p col_type: DATE → stored day count, DECIMAL →
/// unscaled integer at the COLUMN's scale, integers as-is.
///
/// A @c numeric_range rather than a single value because the lowering need not
/// be exact: a decimal constant with more fractional digits than the column can
/// store lands strictly BETWEEN two representable values, and which end a
/// comparison should take depends on its direction. @c minimum is then the
/// floor and @c maximum the ceil; they are equal whenever the domain represents
/// the constant exactly.
///
/// nullopt when the constant has no exact image in that domain — the caller
/// then keeps the conjunct for itself.
std::optional<sirius::numeric_range> to_decoded_bound(duckdb::Value const& value,
                                                      sirius::logical_type const& col_type)
{
  if (value.IsNull()) { return std::nullopt; }

  // DATE decodes to its stored day count, which is not a numeric literal domain
  // constant_numeric_range covers.
  if (col_type.id() == sirius::type_id::DATE) {
    if (value.type().id() != duckdb::LogicalTypeId::DATE) { return std::nullopt; }
    auto const days = static_cast<int128>(duckdb::DateValue::Get(value).days);
    return sirius::numeric_range{sirius::numeric_range_domain::SIGNED_INTEGER, days, days, 0};
  }

  if (!col_type.is_decimal() && !col_type.is_integer()) { return std::nullopt; }
  // Wider decimals are int128-backed and never bitpack-planned.
  if (col_type.is_decimal() &&
      col_type.decimal_precision() > sirius::logical_type::decimal_max_precision_int64) {
    return std::nullopt;
  }
  // UBIGINT/UHUGEINT are refused outright: the decoded domain is signed int64
  // end to end (the decode ballot widens a decoded lane with a plain
  // static_cast<int64_t>), so a value at or above 2^63 would decode to a
  // negative int64 and never satisfy a range built from its true unsigned
  // value here — that drops matching rows instead of merely under-filtering.
  // is_integer() alone would let both through (they carry no width/signedness
  // split), so they need their own check ahead of the general integer path.
  if (col_type.id() == sirius::type_id::UBIGINT || col_type.id() == sirius::type_id::UHUGEINT) {
    return std::nullopt;
  }
  // A constant wider than 64 bits is refused outright: the decoded domain is
  // int64, and rescaling an int128 payload by a power of ten could overflow the
  // accumulator before the clamp ever sees it.
  if (value.type().InternalType() == duckdb::PhysicalType::INT128) { return std::nullopt; }

  // The literal's exact value in its OWN domain, via the same lowering the
  // narrowing machinery uses — it owns which payload alternative a declared
  // type may carry, and rejects a constant whose payload disagrees.
  auto const constant_type = sirius::from_duckdb(value.type());
  auto const exact         = sirius::ast::constant_numeric_range(
    sirius::ast::constant{sirius::from_duckdb(value, constant_type), constant_type});
  if (!exact.has_value()) { return std::nullopt; }

  if (col_type.is_integer()) {
    // An integer column's decoded domain IS the constant's, so a decimal
    // literal has no exact image in it.
    if (exact->domain == sirius::numeric_range_domain::DECIMAL) { return std::nullopt; }
    return *exact;
  }

  // Decimal column: restate the constant at the COLUMN's scale. An integral
  // literal is scale 0.
  auto const scale_diff = static_cast<int>(col_type.decimal_scale()) - exact->decimal_scale;
  if (scale_diff >= 0) {
    // The column carries at least the constant's fractional digits: exact.
    auto const v = exact->minimum * pow10_128(scale_diff);
    return sirius::decimal_range(v, v, static_cast<uint8_t>(col_type.decimal_scale()));
  }
  // The constant is finer than the column's scale: floor and ceil of
  // m / 10^-diff, which straddle the constant unless it divides exactly.
  auto const divisor = pow10_128(-scale_diff);
  auto const m       = exact->minimum;
  int128 quotient    = m / divisor;
  int128 const rem   = m % divisor;
  if (rem != 0 && m < 0) { quotient -= 1; }  // truncation → floor
  return sirius::decimal_range(
    quotient, quotient + (rem != 0 ? 1 : 0), static_cast<uint8_t>(col_type.decimal_scale()));
}

/// The running intersection of one column's conjuncts, in the int128 the
/// numeric_range domain already uses so bound arithmetic cannot wrap. Starts at
/// the full int64 domain — the decoder only ever produces int64-representable
/// values.
sirius::numeric_range full_decoded_domain()
{
  return sirius::signed_integer_range(std::numeric_limits<std::int64_t>::min(),
                                      std::numeric_limits<std::int64_t>::max());
}

/// Fold @p filter into @p acc, returning true iff at least one bound was
/// contributed. @p fully_covered is cleared whenever some restricting part of
/// the filter could NOT be expressed in the range — the resulting range is then
/// a sound over-approximation (the conjuncts intersect, so skipping one only
/// under-filters), usable to drop rows during decode but never as grounds to
/// skip the scan's own filter.
///
/// IS_NOT_NULL child conjuncts are absorbed without affecting coverage:
/// today's post-decode filter (convert_table_filters_to_expression) drops them,
/// so ignoring them changes nothing. Unconvertible AND children are skipped
/// (coverage lost, bounds kept). OR/IN and other shapes contribute no bounds at
/// all — decomposing them soundly needs a hull, not an intersection.
bool fold_numeric_conjunct(duckdb::TableFilter const& filter,
                           sirius::logical_type const& col_type,
                           sirius::numeric_range& acc,
                           bool& fully_covered)
{
  switch (filter.filter_type) {
    case duckdb::TableFilterType::CONSTANT_COMPARISON: {
      auto const& cmp  = filter.Cast<duckdb::ConstantFilter>();
      auto const bound = to_decoded_bound(cmp.constant, col_type);
      if (!bound.has_value()) {
        fully_covered = false;
        return false;
      }
      switch (cmp.comparison_type) {
        case duckdb::ExpressionType::COMPARE_EQUAL:
          // Non-integral constant ⇒ ceil > floor ⇒ lo > hi: provably empty.
          acc.minimum = std::max(acc.minimum, bound->maximum);
          acc.maximum = std::min(acc.maximum, bound->minimum);
          return true;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
          acc.minimum = std::max(acc.minimum, bound->maximum);
          return true;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
          acc.minimum = std::max(acc.minimum, bound->minimum + 1);
          return true;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
          acc.maximum = std::min(acc.maximum, bound->minimum);
          return true;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
          acc.maximum = std::min(acc.maximum, bound->maximum - 1);
          return true;
        default:  // <>, IS DISTINCT FROM, ... — not a range
          fully_covered = false;
          return false;
      }
    }
    case duckdb::TableFilterType::CONJUNCTION_AND: {
      auto const& conjunction = filter.Cast<duckdb::ConjunctionAndFilter>();
      bool any_bound          = false;
      for (auto const& child : conjunction.child_filters) {
        if (child->filter_type == duckdb::TableFilterType::IS_NOT_NULL) { continue; }
        any_bound |= fold_numeric_conjunct(*child, col_type, acc, fully_covered);
      }
      return any_bound;
    }
    default: fully_covered = false; return false;
  }
}

/// Clamp the int128 intersection back into an inclusive int64 range. A range
/// lying entirely outside int64 (possible only through rescaled decimal bounds)
/// is empty for any decodable value, canonically {0, -1}.
sirius::decode_range clamp_to_decode_range(sirius::numeric_range const& acc)
{
  constexpr auto kMin = std::numeric_limits<std::int64_t>::min();
  constexpr auto kMax = std::numeric_limits<std::int64_t>::max();
  if (acc.minimum > acc.maximum || acc.minimum > static_cast<int128>(kMax) ||
      acc.maximum < static_cast<int128>(kMin)) {
    return {0, -1};
  }
  return {static_cast<std::int64_t>(std::max<int128>(acc.minimum, kMin)),
          static_cast<std::int64_t>(std::min<int128>(acc.maximum, kMax))};
}

}  // namespace

scan_filter_analysis analyze_scan_filters(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::unordered_set<std::size_t>& skip_primary_indices,
  const std::unordered_set<std::size_t>& filter_only_primary_indices)
{
  scan_filter_analysis result;
  result.ranges_cover_whole_filter = true;

  // An unsupported restricting conjunct does not discard the other columns'
  // ranges: what was extracted remains a sound conjunctive over-approximation,
  // so the decode can still drop rows with it. It does clear coverage — the
  // scan must keep its own filter, which re-checks the applied conjuncts
  // (idempotent) and evaluates the residual.
  auto const not_covered = [&result](duckdb::idx_t column_index, char const* why) {
    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN filter analysis: filter on column_index={} {} — the scan's own filter is still "
      "required",
      column_index,
      why);
    result.ranges_cover_whole_filter = false;
  };

  for (auto const& [column_index, filter] : filters.filters) {
    if (!filter) { continue; }
    // Non-restricting forms, exactly as convert_table_filters_to_expression
    // skips them: dynamic/optional filters run downstream, IS_NOT_NULL is
    // dropped from the post-decode conjunction today.
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }
    if (column_index >= column_ids.size()) {
      not_covered(column_index, "references no scan column");
      continue;
    }
    auto const& column_id = column_ids[column_index];
    if (!column_id.HasPrimaryIndex() || column_id.IsRowIdColumn() || column_id.IsEmptyColumn() ||
        column_id.IsVirtualColumn()) {
      not_covered(column_index, "targets a rowid/virtual column");
      continue;
    }
    auto const primary_idx = static_cast<std::size_t>(column_id.GetPrimaryIndex());
    // Hive-partition filters are enforced at the file-list level and dropped
    // from the post-decode conjunction, so they don't restrict batch rows.
    if (skip_primary_indices.count(primary_idx)) { continue; }
    if (primary_idx >= returned_types.size()) {
      not_covered(column_index, "has no returned type");
      continue;
    }
    auto const& col_type = returned_types[primary_idx];

    // A pure-filter string column whose whole filter is an equality / IN: the
    // decoder can answer it off a dictionary's key set. Guard the column type
    // as well as the constants — a non-VARCHAR column can never be answered by
    // a string key comparison.
    if (filter_only_primary_indices.count(primary_idx) &&
        sirius::to_duckdb(col_type).id() == duckdb::LogicalTypeId::VARCHAR) {
      std::vector<std::string> values;
      if (collect_equality_values(*filter, values) && !values.empty()) {
        result.equality_sets.emplace(primary_idx, std::move(values));
      }
    }

    auto acc             = full_decoded_domain();
    bool fully_covered   = true;
    bool const any_bound = fold_numeric_conjunct(*filter, col_type, acc, fully_covered);
    if (!fully_covered) {
      not_covered(column_index, "is not fully an AND-tree of numeric constant comparisons");
    }
    if (!any_bound) { continue; }
    auto const range    = clamp_to_decode_range(acc);
    auto [it, inserted] = result.ranges.emplace(primary_idx, range);
    if (!inserted) {  // same physical column filtered twice: intersect
      it->second.lo = std::max(it->second.lo, range.lo);
      it->second.hi = std::min(it->second.hi, range.hi);
    }
    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN filter analysis: primary_idx={} type={} → decoded-domain range [{}, {}]{}",
      primary_idx,
      col_type.to_string(),
      it->second.lo,
      it->second.hi,
      it->second.lo > it->second.hi ? " (provably empty)" : "");
  }

  SIRIUS_LOG_DEBUG(
    "TABLE_SCAN filter analysis: {} range(s), {} equality set(s), "
    "ranges_cover_whole_filter={}",
    result.ranges.size(),
    result.equality_sets.size(),
    result.ranges_cover_whole_filter);
  return result;
}

sirius::pushdown_request build_pushdown_request(scan_filter_analysis const& analysis,
                                                std::span<const std::size_t> primary_index_by_slot)
{
  sirius::pushdown_request request;
  if (analysis.equality_sets.empty() && analysis.ranges.empty()) { return request; }

  request.columns.resize(primary_index_by_slot.size());
  bool any = false;
  for (std::size_t slot = 0; slot < primary_index_by_slot.size(); ++slot) {
    auto const primary_idx = primary_index_by_slot[slot];
    if (auto const it = analysis.equality_sets.find(primary_idx);
        it != analysis.equality_sets.end()) {
      request.columns[slot].equals_any = it->second;
      any                              = true;
    }
    if (auto const it = analysis.ranges.find(primary_idx); it != analysis.ranges.end()) {
      request.columns[slot].range = it->second;
      any                         = true;
    }
  }
  if (!any) { return {}; }
  // Coverage only survives when every range reached a slot: a range that maps
  // to no decoded column is a conjunct the decode cannot apply.
  request.ranges_cover_whole_filter =
    analysis.ranges_cover_whole_filter &&
    std::all_of(analysis.ranges.begin(), analysis.ranges.end(), [&](auto const& entry) {
      return std::find(primary_index_by_slot.begin(), primary_index_by_slot.end(), entry.first) !=
             primary_index_by_slot.end();
    });
  return request;
}

}  // namespace sirius::op

//===----------------------------------------------------------------------===//
// residual_filter
//===----------------------------------------------------------------------===//

namespace sirius::op {

residual_filter::residual_filter(std::vector<table_filter_conjunct> conjuncts,
                                 std::unordered_set<std::size_t> const& answerable_batch_positions)
{
  _conjuncts.reserve(conjuncts.size());
  for (auto& source : conjuncts) {
    // Lower to Sirius AST once, here, rather than per batch: the comparison
    // never changes, only whether it is the form we use.
    auto lowered = sirius::ast::from_duckdb(*source.expr);
    if (!lowered) {
      // Fail here rather than per batch. A conjunct that cannot be lowered
      // cannot be evaluated at all, and an empty residual would read as "this
      // scan has no filter" — silently returning unfiltered rows. Lowering the
      // whole conjunction used to be attempted per batch, where the same
      // failure produced a null AST the evaluator then dereferenced.
      throw std::runtime_error(
        "[residual_filter] a pushed-down filter conjunct cannot be lowered to Sirius AST: " +
        source.expr->ToString());
    }
    std::optional<std::size_t> answered_at;
    if (answerable_batch_positions.count(source.batch_position)) {
      answered_at = source.batch_position;
    }
    _conjuncts.push_back({std::move(lowered), answered_at});
  }
}

std::unique_ptr<sirius::ast::node> residual_filter::against(
  std::vector<std::size_t> const& answered_positions, bool answers_enforced) const
{
  if (_conjuncts.empty()) { return nullptr; }

  auto const answered = [&](std::optional<std::size_t> position) {
    return position.has_value() &&
           std::find(answered_positions.begin(), answered_positions.end(), *position) !=
             answered_positions.end();
  };

  std::vector<std::unique_ptr<sirius::ast::node>> children;
  children.reserve(_conjuncts.size());
  for (auto const& conjunct : _conjuncts) {
    if (answered(conjunct.answered_at)) {
      // The decode already dropped the rows this conjunct rejects, so asking
      // again would be redundant work on a condition that cannot be false.
      if (answers_enforced) { continue; }
      // Otherwise the column IS the answer — a BOOL8 result, not values — so
      // reference it. Re-running the comparison would test a mask against the
      // original constant.
      children.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::reference{static_cast<std::uint32_t>(*conjunct.answered_at),
                               sirius::logical_type::make(sirius::type_id::BOOLEAN)}));
    } else {
      children.push_back(sirius::ast::clone(*conjunct.comparison));
    }
  }

  if (children.empty()) { return nullptr; }
  if (children.size() == 1) { return std::move(children[0]); }
  sirius::ast::conjunction all{sirius::ast::conjunction::kind::op_and, std::move(children)};
  return std::make_unique<sirius::ast::node>(std::move(all));
}

}  // namespace sirius::op
