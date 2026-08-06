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

#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op {

class sirius_dynamic_bloom_filter;

//===----------------------------------------------------------------------===//
// publish_dynamic_filters
//===----------------------------------------------------------------------===//

/**
 * @brief What one publication attempt did
 *
 * `publish_dynamic_filters()` returns these counts without retaining a context or updating
 * `dynamic_filter_stats`. The producing `sirius_physical_hash_join` folds the outcome into its
 * optional stats sink.
 */
struct dynamic_filter_publication_outcome {
  std::size_t keys_considered            = 0;  ///< Bound admitted keys the attempt walked
  std::size_t keys_with_known_domain     = 0;  ///< Keys whose domain cardinality was nonzero
  std::size_t keys_build_exceeded_domain = 0;  ///< Build rows exceeded the domain bound
  std::size_t skipped_targets_drained    = 0;  ///< 1 when the attempt hit the all-drained return
  std::size_t keys_skipped_domain_gate   = 0;  ///< Skipped: build too dense a sample of the domain
  std::size_t keys_skipped_type_mismatch = 0;  ///< Skipped: plan type disagreed with the column
  std::size_t membership_filters_built   = 0;
  std::size_t zone_map_filters_built     = 0;
  std::size_t active_targets             = 0;  ///< Targets still accepting filters
  std::size_t filters_pushed             = 0;  ///< Accepted pushes across every target
};

/**
 * @brief Build, replicate, and fan out one immutable dynamic-filter snapshot from a complete
 * hash-join build table
 *
 * The immutable @ref dynamic_filter_publish_plan is the only key and target input. The function
 * constructs filters only for admitted keys with at least one binding, completes device
 * replication, and pushes each filter at the binding's channel push ordinal. It retains none of its
 * inputs.
 *
 * @ref sirius_physical_hash_join::publish_dynamic_filters owns source readiness and exactly-once
 * arbitration.
 *
 * A key whose recorded storage type disagrees with its runtime build column is skipped and counted.
 * Before accessing a build column, the function validates its ordinal against @p build_view; an
 * out-of-range ordinal fails the publication attempt with `std::logic_error`.
 *
 * @pre @p plan is enabled
 * @throw std::runtime_error if the source GPU cannot be identified
 * @throw std::logic_error if an admitted key's build ordinal lies outside `build_view`, if the
 * source GPU is absent from the plan's replica spaces, or if a constructed filter does not
 * implement `sirius_device_replicable`
 *
 * @param[in] plan The join's enabled publication plan (admitted keys, targets, policy, replica
 * placements)
 * @param[in] build_view The complete build table to reduce / build membership over; admitted build
 * key ordinals index its columns
 * @param[in] stream Stream used for filter construction
 * @return Counts describing what the attempt constructed, skipped, and pushed
 */
[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  rmm::cuda_stream_view stream);

/**
 * @brief Result of one exact-ID multi-partition contribution
 */
struct dynamic_filter_accumulation_result {
  enum class status : std::uint8_t { pending, duplicate, published, aborted };

  status state = status::pending;
  dynamic_filter_publication_outcome publication;
};

namespace detail {
/**
 * @brief Internal deterministic seams for accumulator concurrency and failure tests
 *
 * Empty callbacks preserve production behavior. This is not a runtime extension point.
 */
struct dynamic_filter_accumulator_test_hooks {
  std::function<void(std::uint64_t)> after_id_claim;     ///< After an ID becomes in flight
  std::function<void(std::uint64_t)> after_insert_sync;  ///< After insertion, before completion
  std::function<void(sirius_dynamic_bloom_filter&,
                     std::span<dynamic_filter_replica_space const>)>
    strict_replicate;  ///< Replaces strict replication at the pre-fan-out boundary
};
}  // namespace detail

/**
 * @brief Exact-ID accumulator for one globally complete multi-partition Bloom snapshot
 *
 * The expected original pre-scatter batch IDs and global row count are frozen at the build sizing
 * barrier. Contributions are idempotent by batch ID. No filter is replicated or exposed until
 * every expected ID has completed insertion.
 */
class dynamic_filter_accumulator final {
 public:
  dynamic_filter_accumulator(dynamic_filter_publish_plan const& plan,
                             std::size_t build_rows,
                             std::vector<std::uint64_t> expected_batch_ids);
  dynamic_filter_accumulator(dynamic_filter_publish_plan const& plan,
                             std::size_t build_rows,
                             std::vector<std::uint64_t> expected_batch_ids,
                             detail::dynamic_filter_accumulator_test_hooks test_hooks);
  ~dynamic_filter_accumulator();

  dynamic_filter_accumulator(dynamic_filter_accumulator const&)            = delete;
  dynamic_filter_accumulator& operator=(dynamic_filter_accumulator const&) = delete;

  [[nodiscard]] dynamic_filter_accumulation_result contribute(std::uint64_t batch_id,
                                                              cudf::table_view const& build_view,
                                                              rmm::cuda_stream_view stream);

  /// Atomically abort an incomplete accumulator and return its outcome only to the closing caller.
  [[nodiscard]] std::optional<dynamic_filter_publication_outcome> abort_if_incomplete() noexcept;

  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] bool aborted() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op
