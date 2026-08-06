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
#include <cstdint>
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
 * @brief How hard the barrier is between a branch's output and whatever consumes it.
 *
 * A new enum, not an alias of @c op::MemoryBarrierType: it describes a *branch's* relationship
 * to the rest of the plan, and its values are chosen for the prefetch scheduler's vocabulary.
 * The mapping from the barrier of the branch's terminating edge is total and 1:1:
 *
 *   op::MemoryBarrierType::FULL     -> full_barrier
 *   op::MemoryBarrierType::PARTIAL  -> serial_barrier
 *   op::MemoryBarrierType::PIPELINE -> streaming
 *
 * @note @c serial_barrier has no pre-existing analogue in the codebase; @c PARTIAL is the only
 *       unclaimed @c MemoryBarrierType and the mapping is by elimination. PARTIAL is documented
 *       as "incremental but respects pipeline boundaries"
 *       (docs/super-sirius/pipeline-execution.md:65) — a *rate* property, whereas "serial" reads
 *       as an *ordering* property. If the two are not meant to coincide, this mapping is the
 *       thing to change.
 */
enum class order_type : std::uint8_t {
  full_barrier,    ///< The consumer waits for this branch to finish entirely.
  serial_barrier,  ///< The consumer makes incremental progress, bounded by pipeline boundaries.
  streaming,       ///< The consumer consumes batches as they arrive.
};

/// One entry of @ref query_index::prefetching_order: a scan and the barrier class of the branch
/// it heads.
struct prefetch_step {
  /// The branch head's source operator. Always a @c SiriusPhysicalOperatorType::GPU_SCAN.
  /// Typed as the base class so a synthetic test operator can stand in for a real
  /// @c sirius_gpu_scan_operator (which needs a @c gpu_ingestible to construct).
  op::sirius_physical_operator* scan{nullptr};
  order_type order{order_type::streaming};
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

  /**
   * @brief The order in which this query's scans should be prefetched, hardest barrier first.
   *
   * One entry per GPU scan operator that heads a branch, deduplicated (a fan-out head owns
   * several branches; the first in plan order wins, matching @c _head_op_to_branch).
   *
   * Ordering: branches are classified by the barrier of the edge leaving their tail
   * (@ref order_type), then **stable-sorted** by that class — @c full_barrier, then
   * @c serial_barrier, then @c streaming. Stability means that within one class the original
   * plan order (which is the branch order, which is the scheduling priority order produced by
   * @c creator::task_creator::compute_pipeline_priorities) is preserved.
   *
   * **This puts a hash join's build side before its probe side only when the probe branch's
   * terminating edge is weaker than the build branch's** — i.e. when the join's own output is not
   * itself cut by a `FULL` barrier. Under @ref build_probe the probe edge is rewritten to
   * PIPELINE and therefore never cuts (@c pipeline_dag::cuts requires FULL), so the probe branch
   * always absorbs the join and is classified by whatever is downstream of it. When that is a
   * FULL edge into a multiport consumer — a join feeding another join's build side, a common
   * TPC-H shape — both branches classify @c full_barrier and plan order decides, which puts the
   * probe first. The rewritten probe edge is never the operative mechanism.
   *
   * Making build-before-probe unconditional requires ranking branches by
   * distance-to-first-blocking-barrier rather than by their terminating edge:
   * @c blocking_distance(B) = the smallest index @c i in @c B such that some out-edge of @c B[i]
   * into a multiport consumer carries @c MemoryBarrierType::FULL (and @c |B| when none does),
   * sorted by @c (blocking_distance, order_rank, plan_index). That is a deferred follow-up: it
   * changes what @ref order_type means, and @ref order_type is also the public per-branch
   * accessor @ref get_branch_order_type.
   *
   * The classification is fixed at @ref build_index time, so it reflects whichever
   * @ref build_index_options the index was built with. **Production callers must pass
   * @ref build_probe** — the same option @c creator::task_creator::compute_pipeline_priorities
   * uses — so prefetch order and dispatch priority agree.
   *
   * Deterministic and terminating: no recursion, no graph search beyond the one
   * @c build_index already performs.
   *
   * Allocates (a vector, a hash set and a sort), so this is not @c noexcept and does not belong
   * in a hot loop or a destructor. Intended to be called once per query and the result cached.
   * The returned pointers are borrowed from the query's pipelines and are only valid while the
   * query lives.
   *
   * @return Steps in prefetch order. Empty when the query has no GPU scan heading a branch.
   */
  [[nodiscard]] std::vector<prefetch_step> prefetching_order() const;

  /**
   * @brief The barrier class of branch @p branch_index, parallel to @ref get_branches.
   * @throws std::out_of_range when @p branch_index is out of range.
   */
  [[nodiscard]] order_type get_branch_order_type(std::size_t branch_index) const;

 private:
  query_index() = default;

  //! Owning storage for every branch's pipeline chain. Stable addresses back the spans below.
  std::vector<std::vector<pipeline_ptr>> _branches;
  //! Non-owning views over _branches, in the same (plan) order.
  std::vector<branch> _branch_views;
  //! Head operator id -> index of the first branch that operator heads.
  std::unordered_map<std::size_t, std::size_t> _head_op_to_branch;
  //! Barrier class of each branch's terminating edge, parallel to _branch_views.
  std::vector<order_type> _branch_order_types;
  //! Source operator of each branch's head pipeline (null when the head has no source),
  //! parallel to _branch_views. Captured during build so prefetching_order() needs no re-walk.
  std::vector<op::sirius_physical_operator*> _branch_heads;
};

}  // namespace sirius::planner
