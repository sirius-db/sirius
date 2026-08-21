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

#include "planner/dynamic_filter/dynamic_filter_key_admission.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/type_conversions.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sirius::planner {

namespace {

bool side_blocks_scan_route(op::dynamic_filter_key_shape shape) noexcept
{
  switch (shape) {
    case op::dynamic_filter_key_shape::direct: return false;
    case op::dynamic_filter_key_shape::cast: return true;
    case op::dynamic_filter_key_shape::computed: return false;
  }
  return true;  // Reject unrecognized enum values conservatively.
}

cudf::size_type to_key_column_ordinal(std::uint32_t bound_reference_index)
{
  constexpr auto k_max_ordinal =
    static_cast<std::uint32_t>(std::numeric_limits<cudf::size_type>::max());
  if (bound_reference_index > k_max_ordinal) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] A join-key reference ordinal exceeds the cuDF column "
      "ordinal range");
  }
  return static_cast<cudf::size_type>(bound_reference_index);
}

std::optional<op::dynamic_filter_publish_plan::admitted_key> admit_scan_route_key(
  sirius::join_condition const& condition,
  op::dynamic_filter_condition_shape shape,
  std::size_t condition_index,
  std::size_t domain_cardinality,
  std::optional<std::size_t> build_side_unique_column)
{
  // Null-equal keys could turn a pruned LEFT join match into an accepted NULL-padded row.
  if (condition.comparison != sirius::comparison_type::equal) { return std::nullopt; }
  if (side_blocks_scan_route(shape.probe) || side_blocks_scan_route(shape.build)) {
    return std::nullopt;
  }

  if (!condition.right->is_reference()) { return std::nullopt; }
  auto const& build_ref = condition.right->as_reference();

  auto const storage_type = sirius::try_get_cudf_type(build_ref.return_type());
  if (!storage_type.has_value()) { return std::nullopt; }

  auto const build_key_ordinal = to_key_column_ordinal(build_ref.column_index);

  // EMPTY preserves unsupported probe types for later route checks.
  auto const& probe_side = *condition.left;
  if (!probe_side.is_reference()) { return std::nullopt; }
  auto const probe_key_ordinal  = to_key_column_ordinal(probe_side.as_reference().column_index);
  auto const probe_storage_type = sirius::try_get_cudf_type(probe_side.as_reference().return_type())
                                    .value_or(cudf::data_type{cudf::type_id::EMPTY});

  return op::dynamic_filter_publish_plan::admitted_key{
    .planner_condition_index      = condition_index,
    .build_key_ordinal            = build_key_ordinal,
    .probe_key_ordinal            = probe_key_ordinal,
    .storage_type                 = *storage_type,
    .probe_storage_type           = probe_storage_type,
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

std::vector<op::dynamic_filter_publish_plan::admitted_key> admit_dynamic_filter_keys(
  duckdb::vector<sirius::join_condition> const& conditions,
  std::vector<op::dynamic_filter_condition_shape> const& condition_shapes,
  std::vector<std::size_t> const& condition_domain_cardinalities,
  std::optional<std::size_t> build_side_unique_column)
{
  if (condition_shapes.size() != conditions.size()) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] Condition shapes must be aligned one-to-one with the join "
      "conditions");
  }
  if (!condition_domain_cardinalities.empty() &&
      condition_domain_cardinalities.size() != conditions.size()) {
    throw std::invalid_argument(
      "[dynamic_filter_key_admission] Domain cardinalities must be empty or aligned one-to-one "
      "with the join conditions");
  }

  std::vector<op::dynamic_filter_publish_plan::admitted_key> admitted_keys;
  for (std::size_t condition_index = 0; condition_index < conditions.size(); ++condition_index) {
    auto const domain_cardinality = condition_index < condition_domain_cardinalities.size()
                                      ? condition_domain_cardinalities[condition_index]
                                      : 0;
    auto admitted                 = admit_scan_route_key(conditions[condition_index],
                                         condition_shapes[condition_index],
                                         condition_index,
                                         domain_cardinality,
                                         build_side_unique_column);
    if (!admitted.has_value()) { continue; }
    admitted_keys.push_back(*std::move(admitted));
  }
  return admitted_keys;
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
