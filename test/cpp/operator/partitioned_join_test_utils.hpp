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

#pragma once

#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"

#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <utility>

namespace sirius::test::operator_utils {

/**
 * @brief A pipeline whose completion the test sets, standing in for the producer that feeds an
 * operator's input port.
 */
class controllable_pipeline final : public sirius::pipeline::sirius_pipeline {
 public:
  explicit controllable_pipeline(bool finished_at_start = true)
    : sirius_pipeline{sirius::pipeline::pipeline_build_context{nullptr}},
      finished{finished_at_start}
  {
  }

  [[nodiscard]] bool is_pipeline_finished() const override { return finished.load(); }

  std::atomic<bool> finished;
};

/**
 * @brief A hash join whose two children are each a CONCAT over a PARTITION over a placeholder
 * PROJECTION, the shape the planner gives a partitioned hash join.
 */
struct partitioned_join {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius::op::sirius_physical_hash_join> join;
  sirius::op::sirius_physical_partition* probe_partition = nullptr;
  sirius::op::sirius_physical_partition* build_partition = nullptr;
};

/**
 * @brief Inputs to `make_partitioned_join`.
 */
struct partitioned_join_options {
  using partition_factory = std::function<duckdb::unique_ptr<sirius::op::sirius_physical_partition>(
    duckdb::vector<sirius::logical_type> types, sirius::op::sirius_physical_hash_join& join)>;

  duckdb::JoinType join_type = duckdb::JoinType::INNER;
  /** @brief Column types of each side; the join condition is equality on column 0. */
  duckdb::vector<duckdb::LogicalType> side_types      = {duckdb::LogicalType::INTEGER};
  sirius::op::dynamic_filter_publish_plan filter_plan = {};
  /** @brief Receives the join's dynamic-filter counters; may be null. */
  sirius::op::dynamic_filter_stats* stats = nullptr;
  /** @brief Builds the build-side PARTITION; empty builds a plain `sirius_physical_partition`. */
  partition_factory make_build_partition = {};
};

/**
 * @brief Builds a `partitioned_join` with its parent pointers set, leaving operator IDs, partition
 * counts, and ports to the test.
 */
[[nodiscard]] inline partitioned_join make_partitioned_join(partitioned_join_options options = {})
{
  namespace op = sirius::op;
  partitioned_join tree;
  tree.logical_join        = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(options.join_type);
  tree.logical_join->types = options.side_types;
  tree.logical_join->types.insert(
    tree.logical_join->types.end(), options.side_types.begin(), options.side_types.end());
  auto const types = sirius::from_duckdb_vec(options.side_types);
  auto const child = [&] {
    return duckdb::make_uniq<op::sirius_physical_operator>(
      op::SiriusPhysicalOperatorType::PROJECTION, types, 1);
  };

  duckdb::JoinCondition condition(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(options.side_types[0], 0),
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(options.side_types[0], 0),
    duckdb::ExpressionType::COMPARE_EQUAL);
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.push_back(std::move(condition));
  tree.join = duckdb::make_uniq<op::sirius_physical_hash_join>(
    *tree.logical_join,
    child(),
    child(),
    sirius::wrap_join_conditions(std::move(conditions)),
    options.join_type,
    duckdb::vector<std::size_t>{},
    duckdb::vector<std::size_t>{},
    duckdb::vector<sirius::logical_type>{},
    1,
    sirius::config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    std::move(options.filter_plan),
    sirius::config::DEFAULT_HASH_PARTITION_BYTES,
    sirius::config::DEFAULT_MAX_BROADCAST_JOIN_SIZE,
    options.stats);

  auto const wrap_side = [&](std::size_t child_index, bool is_build) {
    auto partition =
      is_build && options.make_build_partition
        ? options.make_build_partition(types, *tree.join)
        : duckdb::make_uniq<op::sirius_physical_partition>(types, 1, tree.join.get(), is_build);
    auto* const result = partition.get();
    partition->children.push_back(std::move(tree.join->children[child_index]));
    auto concat =
      duckdb::make_uniq<op::sirius_physical_concat>(types, 1, tree.join.get(), is_build);
    concat->children.push_back(std::move(partition));
    tree.join->children[child_index] = std::move(concat);
    return result;
  };
  tree.probe_partition = wrap_side(0, false);
  tree.build_partition = wrap_side(1, true);
  sirius::planner::sirius_physical_plan_generator::set_parent_ops(*tree.join, nullptr);
  return tree;
}

}  // namespace sirius::test::operator_utils
