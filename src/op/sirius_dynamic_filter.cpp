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
#include <op/sirius_dynamic_filter.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/search.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/timestamps.hpp>

// standard library
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sirius::op {

namespace {

cudf::ast::expression const& and_join(cudf::ast::tree& tree,
                                      cudf::ast::expression const& lhs,
                                      cudf::ast::expression const& rhs)
{
  return tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_AND, lhs, rhs);
}

cudf::ast::expression const& or_join(cudf::ast::tree& tree,
                                     cudf::ast::expression const& lhs,
                                     cudf::ast::expression const& rhs)
{
  return tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_OR, lhs, rhs);
}

// cudf::ast::literal only has typed constructors; dispatch on type_id to downcast a type-erased
// cudf::scalar to the correct concrete subtype before emplacing.
cudf::ast::expression const& emplace_literal_from_scalar(cudf::ast::tree& tree, cudf::scalar& s)
{
  switch (s.type().id()) {
    case cudf::type_id::INT8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int8_t>&>(s));
    case cudf::type_id::INT16:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int16_t>&>(s));
    case cudf::type_id::INT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int32_t>&>(s));
    case cudf::type_id::INT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int64_t>&>(s));
    case cudf::type_id::UINT8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint8_t>&>(s));
    case cudf::type_id::UINT16:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint16_t>&>(s));
    case cudf::type_id::UINT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint32_t>&>(s));
    case cudf::type_id::UINT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint64_t>&>(s));
    case cudf::type_id::FLOAT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<float>&>(s));
    case cudf::type_id::FLOAT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<double>&>(s));
    case cudf::type_id::BOOL8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<bool>&>(s));
    case cudf::type_id::TIMESTAMP_DAYS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_D>&>(s));
    case cudf::type_id::TIMESTAMP_SECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_s>&>(s));
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ms>&>(s));
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_us>&>(s));
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ns>&>(s));
    case cudf::type_id::DECIMAL32:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal32>&>(s));
    case cudf::type_id::DECIMAL64:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal64>&>(s));
    case cudf::type_id::DECIMAL128:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::fixed_point_scalar<numeric::decimal128>&>(s));
    case cudf::type_id::STRING:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::string_scalar&>(s));
    default:
      throw std::runtime_error(
        "[sirius_dynamic_zone_map_filter] Unsupported scalar type for AST literal");
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// AST mix-in
//===----------------------------------------------------------------------===//
cudf::ast::tree sirius_ast_lowerable::to_standalone_ast(
  std::function<cudf::ast::expression const&(cudf::ast::tree&)> const& column_ref_factory) const
{
  cudf::ast::tree tree;
  auto const& col_ref = column_ref_factory(tree);
  (void)to_ast(tree, col_ref);
  return tree;
}

//===----------------------------------------------------------------------===//
// sirius_dynamic_zone_map_filter
//===----------------------------------------------------------------------===//
sirius_dynamic_zone_map_filter::sirius_dynamic_zone_map_filter(std::vector<zone_map_entry> zones,
                                                               bool inclusive_min,
                                                               bool inclusive_max)
  : _zones(std::move(zones)), _inclusive_min(inclusive_min), _inclusive_max(inclusive_max)
{
  if (_zones.empty()) {
    throw std::invalid_argument("[sirius_dynamic_zone_map_filter] At least one zone is required");
  }
  for (auto const& z : _zones) {
    if (!z.min || !z.max) {
      throw std::invalid_argument(
        "[sirius_dynamic_zone_map_filter] Every zone must have non-null min and max");
    }
  }
}

cudf::ast::expression const& sirius_dynamic_zone_map_filter::to_ast(
  cudf::ast::tree& tree, cudf::ast::expression const& column_ref) const
{
  auto const lower_op =
    _inclusive_min ? cudf::ast::ast_operator::GREATER_EQUAL : cudf::ast::ast_operator::GREATER;
  auto const upper_op =
    _inclusive_max ? cudf::ast::ast_operator::LESS_EQUAL : cudf::ast::ast_operator::LESS;

  cudf::ast::expression const* result = nullptr;
  for (auto const& z : _zones) {
    auto const& min_lit   = emplace_literal_from_scalar(tree, *z.min);
    auto const& max_lit   = emplace_literal_from_scalar(tree, *z.max);
    auto const& lo        = tree.emplace<cudf::ast::operation>(lower_op, column_ref, min_lit);
    auto const& hi        = tree.emplace<cudf::ast::operation>(upper_op, column_ref, max_lit);
    auto const& zone_pred = and_join(tree, lo, hi);
    result                = (result == nullptr) ? &zone_pred : &or_join(tree, *result, zone_pred);
  }
  return *result;
}

//===----------------------------------------------------------------------===//
// sirius_dynamic_in_list_filter —
//   implemented in src/cuda/sirius_dynamic_in_list_filter.cu (the
//   persistent cuco::static_set is device code, PIMPL'd behind set_impl).
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// sirius_dynamic_filter_set
//===----------------------------------------------------------------------===//
bool sirius_dynamic_filter_set::push_filter(std::size_t col_idx,
                                            std::shared_ptr<sirius_dynamic_filter const> f)
{
  if (!f) { return false; }
  {
    std::scoped_lock lk(_mu);
    if (!_accepting_filters.load(std::memory_order_relaxed)) { return false; }
    if (_ignored_columns.count(col_idx) != 0) { return false; }
    _filters[col_idx].push_back(std::move(f));
  }
  // Release-store so a consumer that observes has_filters() (acquire) is guaranteed to then see the
  // map write above when it re-locks _mu in filters_for_column().
  _filter_count.fetch_add(1, std::memory_order_release);
  return true;
}

void sirius_dynamic_filter_set::ignore_columns(std::vector<std::size_t> const& cols)
{
  std::scoped_lock lk(_mu);
  _ignored_columns.insert(cols.begin(), cols.end());
}

void sirius_dynamic_filter_set::register_producer()
{
  _producer_count.fetch_add(1, std::memory_order_release);
}

void sirius_dynamic_filter_set::close_for_new_filters()
{
  std::scoped_lock lk(_mu);
  _accepting_filters.store(false, std::memory_order_release);
}

std::vector<std::shared_ptr<sirius_dynamic_filter const>>
sirius_dynamic_filter_set::filters_for_column(std::size_t col_idx) const
{
  std::scoped_lock lk(_mu);
  auto it = _filters.find(col_idx);
  if (it == _filters.end()) { return {}; }
  return it->second;
}

std::vector<std::size_t> sirius_dynamic_filter_set::filtered_columns() const
{
  std::scoped_lock lk(_mu);
  std::vector<std::size_t> out;
  out.reserve(_filters.size());
  for (auto const& [k, _] : _filters) {
    out.push_back(k);
  }
  return out;
}

bool sirius_dynamic_filter_set::empty() const
{
  std::scoped_lock lk(_mu);
  return _filters.empty();
}

cudf::ast::expression const& merge_ast_dynamic_filters_into_tree(
  cudf::ast::tree& tree,
  cudf::ast::expression const& existing_root,
  sirius_dynamic_filter_set const& set,
  column_ref_resolver_fn const& column_ref_resolver)
{
  if (set.empty()) { return existing_root; }

  auto cols = set.filtered_columns();
  std::sort(cols.begin(), cols.end());

  cudf::ast::expression const* dynamic_root = nullptr;
  for (auto const col_idx : cols) {
    auto filters = set.filters_for_column(col_idx);
    if (filters.empty()) { continue; }

    cudf::ast::expression const* column_ref = nullptr;
    cudf::ast::expression const* per_col    = nullptr;
    for (auto const& f : filters) {
      auto const* lowerable = dynamic_cast<sirius_ast_lowerable const*>(f.get());
      if (!lowerable) { continue; }
      if (!column_ref) { column_ref = &column_ref_resolver(col_idx); }
      auto const& fragment = lowerable->to_ast(tree, *column_ref);
      per_col              = per_col ? &and_join(tree, *per_col, fragment) : &fragment;
    }
    if (!per_col) { continue; }
    dynamic_root = dynamic_root ? &and_join(tree, *dynamic_root, *per_col) : per_col;
  }
  if (!dynamic_root) { return existing_root; }
  return and_join(tree, existing_root, *dynamic_root);
}

}  // namespace sirius::op
