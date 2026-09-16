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
#include <variant>
#include <vector>

namespace sirius::planner {

class query;

/// Branch-formation strategy: cut a branch at every multiport (fan-in) consumer operator. The
/// multiport pipeline starts the next branch; branches are disjoint. This is the default.
struct pipeline_order {};

/// Branch-formation strategy: like pipeline_order, but a branch that reaches a multiport consumer
/// through a PIPELINE- or PARTIAL-barrier port extends past that operator into the downstream
/// pipeline. Only FULL-barrier edges cut a branch, so pipelinable work stays in one branch.
struct barrier_order {};

/// Branch-formation strategy: exactly like barrier_order, except that an edge feeding the probe
/// side of a HASH_JOIN consumer is always treated as a PIPELINE barrier (regardless of its actual
/// barrier). This keeps the probe pipeline flowing through the join into the downstream branch,
/// while the build side still cuts on its own (typically FULL) barrier.
struct build_probe {};

/// Options for query_index::build_index.
struct build_index_options {
  /// How branches are formed from the pipeline DAG. Defaults to pipeline_order.
  std::variant<pipeline_order, barrier_order, build_probe> branch_order{pipeline_order{}};
};

/**
 * @brief Precomputed structural index over a query's pipeline DAG.
 *
 * The index partitions the pipeline DAG into linear "branches" following consumer (downstream)
 * edges. A branch is cut at a multiport consumer operator (fan-in, e.g. a join/merge whose source
 * operator has more than one input port); the multiport pipeline starts a *new* branch, so
 * pipeline_order branches are disjoint. barrier_order relaxes this: a branch reaching a multiport
 * consumer through a PIPELINE/PARTIAL-barrier port is extended past the operator into the
 * downstream pipeline (only FULL barriers cut). The chosen strategy is set via build_index_options.
 *
 * Example (numbers are pipelines; the operator merging into 4 is multiport):
 *   1 -> 2 -> 3 --[FULL]-->  4        5 -> 6 --[PIPELINE]--> 4        4 -> 7 -> 8
 * pipeline_order branches: [1,2,3], [5,6], [4,7,8].
 * barrier_order  branches: [1,2,3], [5,6,4,7,8]   (the PIPELINE edge extends 5,6 through 4,7,8).
 *
 * Branches are stored in plan order (by the head pipeline's position in the query's execution
 * order), so earlier branches sit closer to the scans. Callers use this order to assign
 * scheduling priority: earlier branch => higher priority.
 */
class query_index {
 public:
  using pipeline_ptr = pipeline::sirius_pipeline*;
  /// A branch: the chain of pipelines forming one branch, head first (closest to the scans).
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
