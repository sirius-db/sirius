/*
 * Copyright 2025, Sirius Contributors.
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
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <helper/type_conversions.hpp>
#include <log/logging.hpp>
#include <op/scan/scan_utils.hpp>

// standard library
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op {

std::vector<std::optional<std::size_t>> build_batch_column_map(
  const duckdb::vector<duckdb::idx_t>& projection_ids, std::size_t column_ids_count)
{
  std::vector<std::optional<std::size_t>> map(column_ids_count);  // default-constructs to nullopt

  if (projection_ids.empty()) {
    for (std::size_t i = 0; i < column_ids_count; i++) {
      map[i] = i;
    }
    return map;
  }

  // Sort projected indices — this matches the iteration order in
  // make_selected_column_indices which walks column_ids[0..N) and
  // includes only indices present in the projected set.
  std::vector<duckdb::idx_t> sorted(projection_ids.begin(), projection_ids.end());
  std::sort(sorted.begin(), sorted.end());

  for (std::size_t batch_pos = 0; batch_pos < sorted.size(); batch_pos++) {
    if (sorted[batch_pos] < column_ids_count) { map[sorted[batch_pos]] = batch_pos; }
  }
  return map;
}

namespace {

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

// ── Numeric range extraction (fused scan-filter) ─────────────────────────────

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
/// integer no wider than 64 bits, nullopt otherwise. Reads the PHYSICAL value:
/// for a DECIMAL this is the unscaled integer, not the rounded SQL value.
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
    auto const divisor   = pow10_128(-scale_diff);
    auto const m         = static_cast<int128>(*unscaled);
    int128 quotient      = m / divisor;
    int128 const rem     = m % divisor;
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
/// the filter could NOT be expressed in the range — the resulting range is
/// then a sound over-approximation (mask conjuncts are conjunctive: skipping
/// one only under-filters), usable as a PARTIAL decode mask but never as
/// grounds to skip the post-decompress filter.
///
/// IS_NOT_NULL child conjuncts are absorbed without affecting coverage:
/// today's post-decompress filter (convert_table_filters_to_expression) drops
/// them, so ignoring them changes nothing. Unconvertible AND children are
/// skipped (coverage lost, bounds kept). OR/IN and other shapes contribute no
/// bounds at all — decomposing them soundly needs a hull, not an intersection.
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

/// Clamp the int128 intersection back into an inclusive int64 range_predicate.
/// A range lying entirely outside int64 (possible only through rescaled decimal
/// bounds) is empty for any decodable value, canonically {0, -1}.
sirius::codegen::range_predicate clamp_to_range_predicate(range_accumulator const& acc)
{
  constexpr auto kMin = std::numeric_limits<std::int64_t>::min();
  constexpr auto kMax = std::numeric_limits<std::int64_t>::max();
  if (acc.lo > acc.hi || acc.lo > static_cast<int128>(kMax) ||
      acc.hi < static_cast<int128>(kMin)) {
    return {0, -1};
  }
  return {static_cast<std::int64_t>(std::max<int128>(acc.lo, kMin)),
          static_cast<std::int64_t>(std::min<int128>(acc.hi, kMax))};
}

}  // namespace

std::unordered_map<std::size_t, std::vector<std::string>> extract_string_equality_pushdown(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types)
{
  std::unordered_map<std::size_t, std::vector<std::string>> result;
  for (auto const& [column_index, filter] : filters.filters) {
    if (!filter || column_index >= column_ids.size()) { continue; }
    auto const& column_id = column_ids[column_index];
    if (!column_id.HasPrimaryIndex() || column_id.IsRowIdColumn() || column_id.IsEmptyColumn() ||
        column_id.IsVirtualColumn()) {
      continue;
    }
    auto const primary_idx = static_cast<std::size_t>(column_id.GetPrimaryIndex());
    if (primary_idx >= returned_types.size()) { continue; }
    // Guard the column type as well as the constants: a non-VARCHAR column can
    // never be answered by a string key comparison.
    if (sirius::to_duckdb(returned_types[primary_idx]).id() != duckdb::LogicalTypeId::VARCHAR) {
      continue;
    }
    std::vector<std::string> values;
    if (!collect_equality_values(*filter, values) || values.empty()) { continue; }
    result.emplace(primary_idx, std::move(values));
  }
  return result;
}

numeric_range_extraction extract_numeric_range_pushdown(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::unordered_set<std::size_t>& skip_primary_indices)
{
  numeric_range_extraction result;
  result.all_conjuncts_convertible = true;

  // An unsupported restricting conjunct no longer discards the other columns'
  // ranges (iteration 3, mixed-mask): the extracted ranges remain a sound
  // conjunctive over-approximation usable as a PARTIAL decode mask. It does
  // clear the whole-filter flag — the scan must keep its post-decompress
  // filter, which re-checks masked conjuncts (idempotent) and evaluates the
  // residual.
  auto const not_covered = [&result](duckdb::idx_t column_index, char const* why) {
    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN range pushdown: filter on column_index={} {} — residual filter required "
      "(partial mask only)",
      column_index,
      why);
    result.all_conjuncts_convertible = false;
  };

  for (auto const& [column_index, filter] : filters.filters) {
    if (!filter) { continue; }
    // Non-restricting forms, exactly as convert_table_filters_to_expression
    // skips them: dynamic/optional filters run downstream, IS_NOT_NULL is
    // dropped from the post-decompress conjunction today.
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
    // from the post-decompress conjunction, so they don't restrict batch rows.
    if (skip_primary_indices.count(primary_idx)) { continue; }
    if (primary_idx >= returned_types.size()) {
      not_covered(column_index, "has no returned type");
      continue;
    }
    auto const& col_type = returned_types[primary_idx];

    range_accumulator acc;
    bool fully_covered = true;
    bool const any_bound = fold_numeric_conjunct(*filter, col_type, acc, fully_covered);
    if (!fully_covered) {
      not_covered(column_index, "is not fully an AND-tree of numeric constant comparisons");
    }
    if (!any_bound) { continue; }
    auto const range    = clamp_to_range_predicate(acc);
    auto [it, inserted] = result.ranges.emplace(primary_idx, range);
    if (!inserted) {  // same physical column filtered twice: intersect
      it->second.lo = std::max(it->second.lo, range.lo);
      it->second.hi = std::min(it->second.hi, range.hi);
    }
    SIRIUS_LOG_DEBUG(
      "TABLE_SCAN range pushdown: primary_idx={} type={} → decoded-domain range [{}, {}]{}",
      primary_idx,
      col_type.to_string(),
      it->second.lo,
      it->second.hi,
      it->second.lo > it->second.hi ? " (provably empty)" : "");
  }

  SIRIUS_LOG_DEBUG("TABLE_SCAN range pushdown: extracted {} range predicate(s), "
                   "all_conjuncts_convertible={}",
                   result.ranges.size(),
                   result.all_conjuncts_convertible);
  return result;
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices,
  const std::unordered_set<std::size_t>& boolean_substituted_primary_indices)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    auto primary_idx = column_ids.at(column_index).GetPrimaryIndex();
    if (skip_primary_indices.count(primary_idx)) {
      SIRIUS_LOG_DEBUG(
        "TABLE_SCAN filter: skipping filter on primary_idx={} (hive partition or equivalent)",
        primary_idx);
      continue;
    }
    auto const col_type = returned_types.at(primary_idx);

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: column_index={}, primary_idx={}, type={}, filter_type={}",
                     column_index,
                     primary_idx,
                     col_type.to_string(),
                     static_cast<int>(filter->filter_type));

    auto const& batch_pos = batch_position_by_column_id[column_index];
    if (!batch_pos.has_value()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) not in projected batch", column_index));
    }
    auto const batch_column_index = static_cast<duckdb::idx_t>(*batch_pos);

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: batch_column_index={}", batch_column_index);

    // The column already carries this filter's answer as a BOOL8 mask (the
    // predicate was resolved during decompression), so the batch column IS the
    // conjunct — re-expressing the comparison would compare against a mask.
    if (boolean_substituted_primary_indices.count(primary_idx)) {
      SIRIUS_LOG_DEBUG(
        "TABLE_SCAN filter: primary_idx={} substituted by a decode-time BOOL8 mask at batch "
        "position {}",
        primary_idx,
        batch_column_index);
      filter_expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(
        duckdb::LogicalType::BOOLEAN, batch_column_index));
      continue;
    }

    auto column_ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      sirius::to_duckdb(col_type), batch_column_index);
    auto expr = filter->ToExpression(*column_ref);
    filter_expressions.push_back(std::move(expr));
  }

  if (filter_expressions.empty()) { return nullptr; }
  if (filter_expressions.size() == 1) { return std::move(filter_expressions[0]); }

  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& expr : filter_expressions) {
    conjunction->children.push_back(std::move(expr));
  }
  return conjunction;
}

std::optional<gpu_expression_translator::translated_expression>
translate_duckdb_expression_with_names(gpu_expression_translator& translator,
                                       duckdb::Expression const& expr,
                                       gpu_expression_translator::column_name_resolver_fxn resolver)
{
  auto node = sirius::ast::from_duckdb(expr);
  if (!node) { return std::nullopt; }
  return translator.translate_expression_with_names(*node, std::move(resolver));
}

}  // namespace sirius::op
