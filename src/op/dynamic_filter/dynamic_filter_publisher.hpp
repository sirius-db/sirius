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
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/table/table_view.hpp>

#include <cuda/stream>

#include <cstddef>
#include <functional>
#include <memory>
#include <span>

namespace cucascade {
class data_batch;
}

namespace sirius::op {

class sirius_physical_hash_join;
struct dynamic_filter_stats;

/**
 * @brief A single-batch whole-build delivery
 */
class complete_build_delivery final {
 private:
  friend class sirius_physical_hash_join;
  friend class dynamic_filter_publication_session;
  explicit complete_build_delivery(std::shared_ptr<cucascade::data_batch> batch)
    : _batch(std::move(batch))
  {
  }
  std::shared_ptr<cucascade::data_batch> _batch;
};

/**
 * @brief Owns one hash join's optional dynamic filter publication and endpoint completion rights
 *
 * 3 responsibilities:
 *  - Own the producer handles for this join's target channels
 *  - Decide which delivery, if any, to publish to the channels
 *  - Finish those handles after publication, cancellation, or failure
 *
 * Ownership relationship: 1 join -> 1 publication session -> 1 producer handle per target channel.
 */
class dynamic_filter_publication_session final {
 public:
  explicit dynamic_filter_publication_session(dynamic_filter_publish_plan plan = {},
                                              dynamic_filter_stats* stats      = nullptr);
  ~dynamic_filter_publication_session();
  dynamic_filter_publication_session(dynamic_filter_publication_session const&)            = delete;
  dynamic_filter_publication_session& operator=(dynamic_filter_publication_session const&) = delete;

  [[nodiscard]] dynamic_filter_publish_plan const& plan() const noexcept;
  void restrict_replicas_to(std::vector<int> const& admitted_gpu_ids);

  /**
   * @brief Seal the publication plan, preventing further producer registration on the plan's
   *        channels.
   *
   * Call after all producers sharing these channels / consumer endpoints have registered and
   * replica placements are finalized. Idempotent; subsequent replica-placement changes are
   * rejected. Existing producers may still publish.
   */
  void seal_plan() noexcept;

  /**
   * @brief Observe a complete hash-join build table and publish filters to the session's channels.
   *
   * The winning delivery is claimed and pinned before deposit runs; readiness checks and optional
   * filter publication follow it. Without a claim, deposit still runs exactly once. The callback
   * runs synchronously outside session locks and is never stored. Pin or deposit failures terminate
   * a claimed attempt and propagate, including deposit allocation failures.
   *
   * @pre @p deposit is nonempty
   * @throw std::runtime_error if the CUDA call establishing source readiness fails
   * @param delivery Delivery certified to contain the join's entire build side, not a partial batch
   * @param deposit Callback that deposits this delivery in the join's existing repository
   */
  void observe_whole_build(complete_build_delivery const& delivery,
                           std::function<void()> const& deposit);

  /**
   * @brief Close input without cancelling an active publication.
   */
  void finish_input() noexcept;

  /**
   * @brief Close input, cancel future publication, and request cancellation of an active attempt.
   */
  void cancel() noexcept;

 private:
  struct state;
  std::shared_ptr<state> _state;
};

struct dynamic_filter_publication_outcome {
  std::size_t keys_considered            = 0;
  std::size_t keys_with_known_domain     = 0;
  std::size_t keys_build_exceeded_domain = 0;
  std::size_t skipped_targets_drained    = 0;
  std::size_t keys_skipped_domain_gate   = 0;
  std::size_t keys_skipped_type_mismatch = 0;
  std::size_t membership_filters_built   = 0;
  std::size_t zone_map_filters_built     = 0;
  std::size_t active_targets             = 0;
  std::size_t filters_pushed             = 0;
};

/**
 * @brief Builds and publishes filters from a complete hash-join build table
 *
 * Replicas are ready before filters reach bound channels. The SESSION supplies one publication
 * right (producer handle) per plan target (channel), pins the ready source (whole-build batch), and
 * retires submitted work on failure. Type-mismatched keys are skipped. before_fanout, when
 * supplied, arbitrates cancellation after construction and before the first push.
 *
 * @pre @p plan is enabled
 * @throw std::runtime_error if the source GPU cannot be identified
 * @throw std::logic_error for inconsistent plan or filter metadata
 */
[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  ::cuda::stream_ref stream,
  std::span<sirius_dynamic_filter_set::producer const> producers,
  std::function<bool()> const& before_fanout = {});

}  // namespace sirius::op
