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
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/utils.hpp>
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

/// A constant lowered into the decoded integer domain as a (possibly
/// non-integral) rational, kept as its integer floor and ceil. Equal for a
/// constant the domain represents exactly; one apart otherwise (a decimal with
/// more fractional digits than the column's scale).
struct decoded_bound {
  int128 floor;
  int128 ceil;
};

/// The constant's unscaled integer payload when its physical storage is an
/// integer this path handles.
std::optional<std::int64_t> physical_integer_payload(duckdb::Value const& value)
{
  switch (value.type().InternalType()) {
    case duckdb::PhysicalType::INT8: return duckdb::TinyIntValue::Get(value);
    case duckdb::PhysicalType::INT16: return duckdb::SmallIntValue::Get(value);
    case duckdb::PhysicalType::INT32: return duckdb::IntegerValue::Get(value);
    case duckdb::PhysicalType::INT64: return duckdb::BigIntValue::Get(value);
    case duckdb::PhysicalType::UINT8: return duckdb::UTinyIntValue::Get(value);
    case duckdb::PhysicalType::UINT16: return duckdb::USmallIntValue::Get(value);
    case duckdb::PhysicalType::UINT32: return duckdb::UIntegerValue::Get(value);
    default: return std::nullopt;  // UINT64/INT128/FLOAT/... — not this path
  }
}

/// 10^e as int128 (e ≤ 38 fits; callers never exceed decimal precision bounds).
int128 pow10_128(int e)
{
  int128 r = 1;
  while (e-- > 0) {
    r *= 10;
  }
  return r;
}

/// Lower @p value into the decoded integer domain of a column of @p col_type:
/// DATE → stored day count, DECIMAL → unscaled integer at the COLUMN's scale,
/// integers as-is. nullopt when the constant's type/width has no exact rational
/// image there (the caller then refuses the whole scan).
std::optional<decoded_bound> to_decoded_bound(duckdb::Value const& value,
                                              sirius::logical_type const& col_type)
{
  if (value.IsNull()) { return std::nullopt; }
  auto const value_type = value.type().id();

  if (col_type.id() == sirius::type_id::DATE) {
    if (value_type != duckdb::LogicalTypeId::DATE) { return std::nullopt; }
    auto const days = static_cast<int128>(duckdb::DateValue::Get(value).days);
    return decoded_bound{days, days};
  }

  if (col_type.is_decimal()) {
    // Wider decimals are int128-backed and never bitpack-planned.
    if (col_type.decimal_precision() > sirius::logical_type::decimal_max_precision_int64) {
      return std::nullopt;
    }
    int constant_scale = 0;
    if (value_type == duckdb::LogicalTypeId::DECIMAL) {
      constant_scale = duckdb::DecimalType::GetScale(value.type());
    } else if (!value.type().IsIntegral()) {
      return std::nullopt;  // integral constants are scale-0; anything else refuses
    }
    auto const unscaled = physical_integer_payload(value);
    if (!unscaled.has_value()) { return std::nullopt; }

    auto const scale_diff = static_cast<int>(col_type.decimal_scale()) - constant_scale;
    if (scale_diff >= 0) {
      // Column carries at least the constant's fractional digits: exact.
      auto const v = static_cast<int128>(*unscaled) * pow10_128(scale_diff);
      return decoded_bound{v, v};
    }
    // Constant is finer than the column's scale: floor/ceil of m / 10^-diff.
    auto const divisor = pow10_128(-scale_diff);
    auto const m       = static_cast<int128>(*unscaled);
    int128 quotient    = m / divisor;
    int128 const rem   = m % divisor;
    if (rem != 0 && m < 0) { quotient -= 1; }  // truncation → floor
    return decoded_bound{quotient, quotient + (rem != 0 ? 1 : 0)};
  }

  if (col_type.is_integer()) {
    if (!value.type().IsIntegral()) { return std::nullopt; }
    auto const v = physical_integer_payload(value);
    if (!v.has_value()) { return std::nullopt; }
    return decoded_bound{static_cast<int128>(*v), static_cast<int128>(*v)};
  }

  return std::nullopt;
}

/// Running intersection of all conjuncts on one column, in int128 so bound
/// arithmetic cannot wrap. Starts at the full int64 domain (the decoder only
/// ever produces int64-representable values).
struct range_accumulator {
  int128 lo = std::numeric_limits<std::int64_t>::min();
  int128 hi = std::numeric_limits<std::int64_t>::max();
};

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
                           range_accumulator& acc,
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
          acc.lo = std::max(acc.lo, bound->ceil);
          acc.hi = std::min(acc.hi, bound->floor);
          return true;
        case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
          acc.lo = std::max(acc.lo, bound->ceil);
          return true;
        case duckdb::ExpressionType::COMPARE_GREATERTHAN:
          acc.lo = std::max(acc.lo, bound->floor + 1);
          return true;
        case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
          acc.hi = std::min(acc.hi, bound->floor);
          return true;
        case duckdb::ExpressionType::COMPARE_LESSTHAN:
          acc.hi = std::min(acc.hi, bound->ceil - 1);
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
sirius::decode_range clamp_to_decode_range(range_accumulator const& acc)
{
  constexpr auto kMin = std::numeric_limits<std::int64_t>::min();
  constexpr auto kMax = std::numeric_limits<std::int64_t>::max();
  if (acc.lo > acc.hi || acc.lo > static_cast<int128>(kMax) || acc.hi < static_cast<int128>(kMin)) {
    return {0, -1};
  }
  return {static_cast<std::int64_t>(std::max<int128>(acc.lo, kMin)),
          static_cast<std::int64_t>(std::min<int128>(acc.hi, kMax))};
}

//===----------------------------------------------------------------------===//
// Column-vs-column comparisons
//===----------------------------------------------------------------------===//

void harvest_column_pairs(duckdb::Expression const& expr,
                          std::vector<sirius::column_pair_conjunct>& out)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONJUNCTION &&
      expr.type == duckdb::ExpressionType::CONJUNCTION_AND) {
    auto const& conjunction = expr.Cast<duckdb::BoundConjunctionExpression>();
    for (auto const& child : conjunction.children) {
      harvest_column_pairs(*child, out);
    }
    return;
  }
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COMPARISON) { return; }
  auto const& cmp = expr.Cast<duckdb::BoundComparisonExpression>();
  sirius::column_compare_op op;
  switch (cmp.type) {
    case duckdb::ExpressionType::COMPARE_LESSTHAN: op = sirius::column_compare_op::lt; break;
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
      op = sirius::column_compare_op::le;
      break;
    case duckdb::ExpressionType::COMPARE_GREATERTHAN: op = sirius::column_compare_op::gt; break;
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      op = sirius::column_compare_op::ge;
      break;
    default: return;  // =, <>, DISTINCT — no kernel evaluates those in-decode
  }
  if (cmp.left->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
      cmp.right->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return;  // casts/functions around a side ⇒ not a plain column pair
  }
  auto const& left  = cmp.left->Cast<duckdb::BoundReferenceExpression>();
  auto const& right = cmp.right->Cast<duckdb::BoundReferenceExpression>();
  out.push_back({left.index, right.index, op});
  SIRIUS_LOG_DEBUG("TABLE_SCAN pair harvest: binding {} {} binding {} (op={})",
                   left.index,
                   duckdb::ExpressionTypeToString(cmp.type),
                   right.index,
                   static_cast<int>(op));
}

}  // namespace

scan_filter_analysis analyze_scan_filters(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::unordered_set<std::size_t>& skip_primary_indices,
  const std::unordered_set<std::size_t>& filter_only_primary_indices,
  duckdb::Expression const* bound_filter)
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

    range_accumulator acc;
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

  if (bound_filter != nullptr) { harvest_column_pairs(*bound_filter, result.pairs); }

  SIRIUS_LOG_DEBUG(
    "TABLE_SCAN filter analysis: {} range(s), {} equality set(s), {} column pair(s), "
    "ranges_cover_whole_filter={}",
    result.ranges.size(),
    result.equality_sets.size(),
    result.pairs.size(),
    result.ranges_cover_whole_filter);
  return result;
}

sirius::scan_decode_request build_scan_decode_request(
  scan_filter_analysis const& analysis, std::span<const std::size_t> primary_index_by_slot)
{
  sirius::scan_decode_request request;
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
  // Column pairs are indexed in the bound filter expression's own binding
  // space, not by primary index, so mapping them belongs with whoever supplies
  // that expression; no caller wires them onto slots yet.
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
  std::vector<std::size_t> const& answered_positions) const
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
      // The column IS the answer to this conjunct — a BOOL8 result, not values.
      // Referencing it is both correct and cheaper than re-comparing; running
      // the comparison would test a mask against the original constant.
      children.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::reference{static_cast<std::uint32_t>(*conjunct.answered_at),
                               sirius::logical_type::make(sirius::type_id::BOOLEAN)}));
    } else {
      children.push_back(sirius::ast::clone(*conjunct.comparison));
    }
  }

  if (children.size() == 1) { return std::move(children[0]); }
  sirius::ast::conjunction all{sirius::ast::conjunction::kind::op_and, std::move(children)};
  return std::make_unique<sirius::ast::node>(std::move(all));
}

}  // namespace sirius::op
