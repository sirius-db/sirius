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

#include "planner/dynamic_filter_key_admission.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/type_conversions.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace sirius::planner {

namespace {

/**
 * @brief Whether this side's shape excludes the whole condition from the scan route
 *
 * Only cast keys are excluded -- the runtime publisher did the same before producer-key admission
 * moved to plan time, keeping build and probe base keys type-equivalent -- while computed
 * (materialized) keys publish their value-correct columns.
 *
 * @param[in] shape The carried classification of one condition side
 * @return True when the condition cannot take the scan route
 */
bool side_blocks_scan_route(op::dynamic_filter_key_shape shape) noexcept
{
  switch (shape) {
    case op::dynamic_filter_key_shape::direct: return false;
    case op::dynamic_filter_key_shape::cast: return true;
    case op::dynamic_filter_key_shape::computed: return false;
  }
  return true;  // unreachable; conservative for an unlisted enumerator
}

/**
 * @brief Convert an AST reference column ordinal to a build-table cuDF column ordinal
 *
 * This is the single checked conversion point between the two index spaces; they meet nowhere else.
 *
 * @throw std::invalid_argument if the index exceeds the cuDF column ordinal range
 *
 * @param[in] bound_reference_index The build side's AST reference column ordinal
 * @return The equivalent cudf column ordinal
 */
cudf::size_type to_build_key_ordinal(std::uint32_t bound_reference_index)
{
  constexpr auto k_max_ordinal =
    static_cast<std::uint32_t>(std::numeric_limits<cudf::size_type>::max());
  if (bound_reference_index > k_max_ordinal) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] A build-side reference ordinal exceeds the cuDF column "
      "ordinal range");
  }
  return static_cast<cudf::size_type>(bound_reference_index);
}

/**
 * @brief Build one admitted key from a scan-route-legal condition
 *
 * A condition is outside the scan route's legality when it is not an `equal` comparison (null-equal
 * is outside the supported scope), carries a cast on either side shape, has a build side that is
 * not a plain bound reference after materialization, or has a build type with no cudf
 * representation.
 *
 * @param[in] condition The join condition to admit
 * @param[in] shape The condition's carried pre-materialization side shapes
 * @param[in] condition_index The condition's index in original planner order
 * @param[in] domain_cardinality The build key's domain cardinality, or 0 when unknown
 * @param[in] build_side_unique_column The build child's sole proven-unique output ordinal, when
 * exactly one column is proven unique
 * @return The admitted key, or nullopt when the condition is not scan-route legal
 */
std::optional<op::dynamic_filter_publish_plan::admitted_key> admit_scan_route_key(
  sirius::join_condition const& condition,
  op::dynamic_filter_condition_shape shape,
  std::size_t condition_index,
  std::size_t domain_cardinality,
  std::optional<std::size_t> build_side_unique_column)
{
  if (condition.comparison != sirius::comparison_type::equal) { return std::nullopt; }
  if (side_blocks_scan_route(shape.probe) || side_blocks_scan_route(shape.build)) {
    return std::nullopt;
  }

  // Read the build side natively. The Sirius AST already records the column ordinal and its type,
  // so rebuilding a DuckDB expression to ask would allocate per condition and, for a node carrying
  // no type, would substitute INTEGER -- a fabricated type the runtime publisher would later find
  // disagrees with the build column.
  if (!condition.right->is_reference()) { return std::nullopt; }
  auto const& build_ref = condition.right->as_reference();

  // No cuDF representation: the runtime publisher also built no usable filter for such a key
  // before producer-key admission moved to plan time (membership kinds reject the column and zone
  // maps fail the probe-type equality gate).
  auto const storage_type = sirius::try_get_cudf_type(build_ref.return_type());
  if (!storage_type.has_value()) { return std::nullopt; }

  auto const build_key_ordinal = to_build_key_ordinal(build_ref.column_index);
  return op::dynamic_filter_publish_plan::admitted_key{
    .planner_condition_index      = condition_index,
    .build_key_ordinal            = build_key_ordinal,
    .storage_type                 = *storage_type,
    .key_shape                    = shape,
    .build_key_domain_cardinality = domain_cardinality,
    .build_key_proven_unique =
      build_side_unique_column == std::optional{static_cast<std::size_t>(build_key_ordinal)}};
}

}  // namespace

op::dynamic_filter_key_shape classify_key_side(duckdb::Expression const& key_side)
{
  auto const expression_class = key_side.GetExpressionClass();
  if (expression_class == duckdb::ExpressionClass::BOUND_REF) {
    return op::dynamic_filter_key_shape::direct;
  }
  if (expression_class == duckdb::ExpressionClass::BOUND_CAST) {
    return key_side.Cast<duckdb::BoundCastExpression>().child->GetExpressionClass() ==
               duckdb::ExpressionClass::BOUND_REF
             ? op::dynamic_filter_key_shape::cast
             : op::dynamic_filter_key_shape::computed;
  }
  return op::dynamic_filter_key_shape::computed;
}

std::vector<op::dynamic_filter_condition_shape> classify_join_key_shapes(
  duckdb::vector<duckdb::JoinCondition> const& conditions)
{
  std::vector<op::dynamic_filter_condition_shape> shapes;
  shapes.reserve(conditions.size());
  for (auto const& condition : conditions) {
    shapes.push_back(op::dynamic_filter_condition_shape{
      .probe = classify_key_side(*condition.left), .build = classify_key_side(*condition.right)});
  }
  return shapes;
}

// Which space each input is indexed in, in parameter order:
//   conditions                      condition index
//   condition_shapes                condition index
//   hinted_condition_indexes        filter ordinal -> condition index
//   scan_targets                    target index; each target's `columns` is by filter ordinal
//   condition_domain_cardinalities  condition index
//   build_side_unique_column        a build-child output ordinal, not a condition index
key_admission_result admit_dynamic_filter_keys(
  duckdb::vector<sirius::join_condition> const& conditions,
  std::vector<op::dynamic_filter_condition_shape> const& condition_shapes,
  std::optional<std::span<std::size_t const>> hinted_condition_indexes,
  std::vector<dynamic_filter_scan_target_input> const& scan_targets,
  std::vector<std::size_t> const& condition_domain_cardinalities,
  std::optional<std::size_t> build_side_unique_column)
{
  if (condition_shapes.size() != conditions.size()) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] Condition shapes must be aligned one-to-one with the join "
      "conditions");
  }
  if (!scan_targets.empty() && !hinted_condition_indexes.has_value()) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] Scan targets come from DuckDB's join-filter pushdown "
      "metadata; pass the condition indexes it recorded");
  }
  if (!condition_domain_cardinalities.empty() &&
      condition_domain_cardinalities.size() != conditions.size()) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] Domain cardinalities must be empty or aligned one-to-one "
      "with the join conditions");
  }
  if (hinted_condition_indexes.has_value()) {
    for (auto const condition_index : *hinted_condition_indexes) {
      if (condition_index >= conditions.size()) {
        throw std::invalid_argument(
          "[dynamic_filter_key_admission] A hinted condition index is out of range");
      }
    }
    for (auto const& target : scan_targets) {
      if (target.columns.size() != hinted_condition_indexes->size()) {
        throw std::invalid_argument(
          "[dynamic_filter_key_admission] A scan target's arity must match the hinted condition "
          "indexes");
      }
    }
  }

  // Hoisted once, immediately after validation: an absent hint and an empty hint drive the same
  // binding loop, so no later code has to reason about the optional's state.
  auto const hinted = hinted_condition_indexes.value_or(std::span<std::size_t const>{});

  key_admission_result result;

  // Admit in a single deterministic pass. When DuckDB named a condition set, admission is
  // restricted to it so publication constructs exactly the filters the runtime publisher
  // constructed before producer-key admission moved to plan time (legacy dynamic filters);
  // otherwise every legality-passing condition is admitted.
  std::unordered_map<std::size_t, std::size_t> admitted_index_by_condition;
  auto admit_condition = [&](std::size_t condition_index) {
    if (admitted_index_by_condition.contains(condition_index)) { return; }
    auto const domain_cardinality = condition_index < condition_domain_cardinalities.size()
                                      ? condition_domain_cardinalities[condition_index]
                                      : 0;
    auto admitted                 = admit_scan_route_key(conditions[condition_index],
                                         condition_shapes[condition_index],
                                         condition_index,
                                         domain_cardinality,
                                         build_side_unique_column);
    if (!admitted.has_value()) { return; }
    admitted_index_by_condition.emplace(condition_index, result.admitted_keys.size());
    result.admitted_keys.push_back(*std::move(admitted));
  };
  if (hinted_condition_indexes.has_value()) {
    for (auto const condition_index : hinted) {
      admit_condition(condition_index);
    }
  } else {
    for (std::size_t condition_index = 0; condition_index < conditions.size(); ++condition_index) {
      admit_condition(condition_index);
    }
  }

  // Bind admitted keys onto each target, sparsely: a filter ordinal whose condition was not
  // admitted simply has no binding on any target.
  result.per_target_key_bindings.resize(scan_targets.size());
  for (std::size_t target_index = 0; target_index < scan_targets.size(); ++target_index) {
    auto const& target = scan_targets[target_index];
    auto& bindings     = result.per_target_key_bindings[target_index];
    for (std::size_t filter_ordinal = 0; filter_ordinal < hinted.size(); ++filter_ordinal) {
      auto const admitted_it = admitted_index_by_condition.find(hinted[filter_ordinal]);
      if (admitted_it == admitted_index_by_condition.end()) { continue; }
      auto const& column = target.columns[filter_ordinal];
      bindings.push_back(op::dynamic_filter_publish_plan::key_binding{
        .admitted_key_index   = admitted_it->second,
        .channel_push_ordinal = column.channel_push_ordinal,
        .probe_storage_type   = column.probe_storage_type});
    }
  }

  return result;
}

bool direct_route_admissible(duckdb::JoinType join_type,
                             sirius::comparison_type comparison,
                             op::dynamic_filter_condition_shape shape,
                             cudf::data_type probe_storage_type,
                             cudf::data_type build_storage_type) noexcept
{
  bool const join_type_supported =
    join_type == duckdb::JoinType::INNER || join_type == duckdb::JoinType::SEMI;
  bool const shapes_supported = shape.probe == op::dynamic_filter_key_shape::direct &&
                                shape.build == op::dynamic_filter_key_shape::direct;
  bool const storage_type_supported =
    probe_storage_type == build_storage_type && (build_storage_type.id() == cudf::type_id::INT32 ||
                                                 build_storage_type.id() == cudf::type_id::INT64);
  return join_type_supported && comparison == sirius::comparison_type::equal && shapes_supported &&
         storage_type_supported;
}

}  // namespace sirius::planner
