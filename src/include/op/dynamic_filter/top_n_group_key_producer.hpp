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

/**
 * @file top_n_group_key_producer.hpp
 * @brief The Top-N group-key producer seam installed in a grouped aggregate's sink
 *
 * `sirius::planner::sirius_physical_plan_generator` installs one of these on a
 * `sirius_physical_grouped_aggregate` when a `LogicalTopN` above that aggregate orders only by
 * grouping keys. Per input batch, `sirius_physical_grouped_aggregate::execute` calls
 * @ref top_n_group_key_producer::prefilter and then @ref top_n_group_key_producer::witness before
 * `gpu_aggregate_impl::local_grouped_aggregate`; the policy itself -- the bounded distinct-key
 * witness set, the boundary, and publication -- lives in the shared
 * @ref top_n_threshold_coordinator, which this class only feeds.
 *
 * Both operations are pure optimizations: whatever they do or decline to do, the aggregate
 * computes exactly the same result. Everything is per-operator state, created fresh per physical
 * plan.
 *
 * Threading: one instance serves every parallel task of its aggregate, and both methods are safe
 * to call concurrently. They own no mutable state themselves -- the keep-ratio decision lives
 * behind `scan::dynamic_filter_gate`'s own locks, accumulation behind the coordinator's mutex, and
 * the key columns are immutable after construction. Each call does its device work on the caller's
 * task stream.
 */

#pragma once

#include "op/dynamic_filter/top_n_boundary_filter.hpp"
#include "op/dynamic_filter/top_n_threshold_coordinator.hpp"
#include "op/scan/dynamic_filter_gate.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief Feeds one grouped aggregate's input batches to a `GROUP_KEY` threshold coordinator
 *
 * The witness object is the Kth-best *distinct grouping key*, never the Kth row: the top K rows of
 * the final Top-N may span fewer than K groups, so a row-level boundary would over-tighten. A
 * batch's contribution is therefore its best distinct key tuples, capped at K.
 *
 * Because the boundary tightens over time, a group can contribute rows from an early batch and
 * have its later rows dropped, so the aggregate's output can contain a **partially aggregated**
 * group. That is safe only because every ORDER BY key is a grouping key: a group's rank depends on
 * its key alone and not on its aggregate values, so the Top-N above discards such a group before
 * any partial value can be observed. Admitting an aggregate-output ORDER BY key would break
 * exactly this step, which is why the planner refuses one outright.
 */
class top_n_group_key_producer final {
 public:
  /**
   * @brief Maximum K for which a group-key producer is admitted
   *
   * Justified by collapsing pruning value, not by rising cost. Measured at 1M rows: per-batch cost
   * rises only ~16-21% from K=10 to K=1000 (`distinct` and `sort` dominate and are K-independent;
   * only slice, D2H, and merge scale, and the merge stays under 0.03 ms), while rows kept climb
   * from 0.2% at K=10 to 20% at K=1000 with 5000 distinct groups, and reach 83% when groups are
   * scarce -- the point where the keep-ratio gate disables the prefilter as unselective. Beyond
   * this bound the producer buys almost nothing, so it is refused rather than optimized. The row
   * producer needs no such cap: it extracts one row's key tuple per batch regardless of K.
   */
  static constexpr std::size_t k_max_admitted_k = 1024;

  /**
   * @brief Construct the seam for one aggregate operator
   *
   * @param[in] coordinator The `GROUP_KEY` coordinator this aggregate offers into; must not be null
   * @param[in] key_columns The Top-N's ORDER BY keys as column indices of the aggregate's *input*,
   * in ORDER BY key order -- the planner resolved them through the aggregate's `group_idx`
   * @param[in] gate_keep_threshold Keep ratio above which the prefilter stops paying for itself
   */
  top_n_group_key_producer(std::shared_ptr<top_n_threshold_coordinator> coordinator,
                           std::vector<cudf::size_type> key_columns,
                           double gate_keep_threshold);

  top_n_group_key_producer(top_n_group_key_producer const&)            = delete;
  top_n_group_key_producer& operator=(top_n_group_key_producer const&) = delete;

  /**
   * @brief Drop input rows the shared boundary already excludes, before the hash insert
   *
   * The predicate is **inclusive** at every key count: the boundary is the Kth-best grouping key,
   * so a row tying it belongs to a group that is in the answer, and dropping it would lower that
   * group's aggregate values. Rows strictly worse than the boundary belong to a group strictly
   * worse than K proven groups and can never reach the final K.
   *
   * Returns an empty result -- no filtering -- when no boundary exists yet or the keep-ratio gate
   * has stopped paying for the pass. A stale boundary prunes less, never more.
   */
  [[nodiscard]] detail::boundary_filter_result prefilter(cudf::table_view const& batch,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr);

  /**
   * @brief Offer this batch's best distinct grouping keys to the coordinator
   *
   * @p batch is the prefiltered view: rows the prefilter dropped are strictly worse than the
   * boundary and cannot be among the K best, so excluding them costs no evidence. The host copies
   * complete before the offer, and the caller's read-only accessor keeps @p batch alive across
   * both, which is the durability the offer requires.
   */
  void witness(cudf::table_view const& batch,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr);

  /// The shared coordinator; the aggregate's finalization drains it through `finish()`.
  [[nodiscard]] top_n_threshold_coordinator& coordinator() const noexcept { return *_coordinator; }

 private:
  std::shared_ptr<top_n_threshold_coordinator> _coordinator;
  std::vector<cudf::size_type> _key_columns;
  scan::dynamic_filter_gate _prefilter_gate;
};

}  // namespace sirius::op
