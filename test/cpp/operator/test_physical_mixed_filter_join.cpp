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

#include "helper/type_conversions.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <algorithm>

using namespace sirius::test::operator_utils;

TEST_CASE("mixed SEMI/ANTI keeps NULL and valid build residuals distinct",
          "[physical_mixed_filter_join]")
{
  using namespace duckdb;
  using namespace sirius::op;
  const auto join_type =
    GENERATE(JoinType::SEMI, JoinType::ANTI, JoinType::RIGHT_SEMI, JoinType::RIGHT_ANTI);
  const bool null_safe    = GENERATE(false, true);
  const bool right_family = join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI;
  const bool anti         = join_type == JoinType::ANTI || join_type == JoinType::RIGHT_ANTI;

  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto keys = [&](std::vector<int32_t> const& values) {
    return make_numeric_batch<int32_t>(*space, values, cudf::type_id::INT32);
  };
  auto residual = [&](std::vector<int32_t> const& values, std::vector<bool> const& valid) {
    return make_numeric_batch_with_nulls<int32_t>(*space, values, valid, cudf::type_id::INT32);
  };

  // NULL and valid residuals deliberately have identical stored bytes. There is no unique
  // payload column that could accidentally keep the build rows distinct during deduplication.
  // Both residual columns need their own validity flag; duplicate valid rows may still collapse.
  auto build =
    concatenate_batches_horizontal({keys({1, 1, 1, 1, 2}),
                                    residual({0, 0, 0, 0, 0}, {false, true, true, true, false}),
                                    residual({0, 0, 0, 0, 0}, {true, false, true, true, false})},
                                   *space);
  const std::vector<int32_t> probe_values(4, null_safe ? 0 : 1);
  const std::vector<bool> probe_valid{true, true, !null_safe, true};
  auto probe = concatenate_batches_horizontal(
    {keys({1, 1, 2, 3}), residual(probe_values, probe_valid), residual(probe_values, probe_valid)},
    *space);

  LogicalComparisonJoin logical_join(join_type);
  logical_join.types = {LogicalType::INTEGER, LogicalType::INTEGER, LogicalType::INTEGER};
  auto make_child    = [&]() {
    return make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::PROJECTION, sirius::from_duckdb_vec(logical_join.types), 0);
  };
  vector<JoinCondition> conditions;
  for (idx_t index = 0; index < 3; ++index) {
    JoinCondition condition;
    condition.left       = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, index);
    condition.right      = make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, index);
    condition.comparison = index == 0  ? ExpressionType::COMPARE_EQUAL
                           : null_safe ? ExpressionType::COMPARE_NOT_DISTINCT_FROM
                                       : ExpressionType::COMPARE_NOTEQUAL;
    conditions.push_back(std::move(condition));
  }
  sirius_physical_hash_join join(logical_join,
                                 make_child(),
                                 make_child(),
                                 sirius::wrap_join_conditions(std::move(conditions)),
                                 join_type,
                                 {},
                                 {},
                                 {},
                                 5);
  join.operator_id              = 0;
  join.children[0]->operator_id = 1;
  join.children[1]->operator_id = 2;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches =
    right_family ? std::vector{build, probe} : std::vector{probe, build};
  auto output      = join.execute(pipelineable_operator_data(batches), cudf::get_default_stream());
  auto const& data = dynamic_cast<pipelineable_operator_data const&>(*output);
  REQUIRE(data.get_data_batches().size() == 1);
  auto const view = sirius::get_cudf_table_view(*data.get_data_batches()[0]);
  REQUIRE(view.num_columns() == 3);
  auto actual = copy_column_to_host<int32_t>(view.column(0));
  std::sort(actual.begin(), actual.end());
  const std::vector<int32_t> expected =
    anti ? (null_safe ? std::vector<int32_t>{3} : std::vector<int32_t>{2, 3})
         : (null_safe ? std::vector<int32_t>{1, 1, 2} : std::vector<int32_t>{1, 1});
  REQUIRE(actual == expected);
}
