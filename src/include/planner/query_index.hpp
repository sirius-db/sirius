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

#pragma once

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace sirius::planner {

class query;

/// Options for query_index::build_index. Reserved for future tuning; empty for now.
struct build_index_options {};

/**
 * @brief Precomputed structural index over a query's pipeline DAG.
 *
 * The index partitions the pipeline DAG into linear "branches". A branch is a maximal chain of
 * pipelines that runs from one branch point to the next, following consumer (downstream) edges.
 * A pipeline is a branch point when it is not a simple 1-producer / 1-consumer pass-through:
 * scans (0 producers), joins / merges (>1 producer, "fan-in"), forks (>1 consumer, "fan-out"),
 * and terminal result pipelines (0 consumers) are all branch points. Both endpoints of a branch
 * are branch points, so adjacent branches share their boundary pipeline.
 *
 * Example (numbers are pipelines; 3, 7, 10 are joins/forks):
 *   1 -> 2 -> 3        4 -> 5 -> 3       3 -> 6 -> 7      8 -> 9 -> 10 -> 7
 *   11 -> 10           3 -> 12
 * yields the branches [1,2,3], [4,5,3], [3,6,7], [3,12], [8,9,10,7], [11,10], ... with pipeline 3
 * appearing (as a shared endpoint) in several branches.
 *
 * Branches are stored in plan order (by the head pipeline's position in the query's execution
 * order), so earlier branches sit closer to the scans. Callers use this order to assign
 * scheduling priority: earlier branch => higher priority.
 */
class query_index {
 public:
  using pipeline_ptr = pipeline::sirius_pipeline*;
  /// A branch: a chain of pipelines from one branch point to the next (both endpoints included),
  /// head first (closest to the scans).
  using branch = std::span<pipeline_ptr const>;

  /**
   * @brief Build the structural index for a query.
   *
   * @param q The query whose pipeline DAG is indexed.
   * @param options Reserved for future tuning.
   * @return A shared, immutable index valid for as long as the query's pipelines live.
   */
  static std::shared_ptr<const query_index> build_index(const query& q,
                                                        build_index_options options = {});

  /**
   * @brief Build the structural index directly from a pipeline list (in execution order).
   *
   * Same as build_index(query) but bypasses the query wrapper; used by build_index(query) and by
   * unit tests that construct pipeline DAGs without a full query.
   */
  static std::shared_ptr<const query_index> build_index(
    const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
    build_index_options options = {});

  /**
   * @brief All branches, in plan order (earlier = closer to the scans).
   */
  [[nodiscard]] std::span<const branch> get_branches() const { return _branch_views; }

  /**
   * @brief The consumer-pipeline chain of the branch that starts at @p op's pipeline.
   *
   * Given a branch-point operator (typically a scan's source), returns the branch it heads:
   * the pipelines from @p op's pipeline down to the next branch point, in plan order. When
   * @p op heads several branches (a fan-out), the first one in plan order is returned. Returns an
   * empty span if @p op does not head a branch.
   *
   * @param op The operator whose pipeline heads the branch (its source operator).
   */
  [[nodiscard]] branch get_consumer_pipelines_till_next_branch(
    const op::sirius_physical_operator* op) const;

 private:
  query_index() = default;

  //! Owning storage for every branch's pipeline chain. Stable addresses back the spans below.
  std::vector<std::vector<pipeline_ptr>> _branches;
  //! Non-owning views over _branches, in the same (plan) order.
  std::vector<branch> _branch_views;
  //! Head operator id -> index of the first branch that operator heads.
  std::unordered_map<std::size_t, std::size_t> _head_op_to_branch;
};

}  // namespace sirius::planner
