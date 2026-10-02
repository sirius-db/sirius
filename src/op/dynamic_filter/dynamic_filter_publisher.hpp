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

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/table/table_view.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>

namespace cucascade {
class data_batch;
class read_only_data_batch;
}  // namespace cucascade

namespace sirius::op {

class sirius_physical_hash_join;
struct dynamic_filter_stats;

/**
 * @brief Why a collecting accumulation ends without a filter.
 */
enum class accumulation_decline : std::uint8_t {
  CONTRIBUTION_UNACCOUNTABLE,  ///< A task input cannot be matched to exactly one certified batch
  INPUT_UNREADABLE,            ///< Reading a task input's batch failed; counted as a masked error
};

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
 *
 * A session publishes either one whole-build delivery (`observe_whole_build`) or, for a
 * multi-partition build, a Bloom filter accumulated from every batch of a certified
 * `complete_build_inventory`. Accumulation moves through the phases open, collecting, publishing,
 * and terminal. Every failure on the accumulation path is optional: it ends the attempt with a
 * counter and a log line and is never rethrown. The session's shared state may outlive the session
 * while an `accumulation_job` holds it; the destructor only cancels and never waits.
 */
class dynamic_filter_publication_session final {
  struct state;

 public:
  /**
   * @brief Deferred remainder of one accumulation, run as after-task work.
   *
   * The publishing job, returned by the final contribution, reduces and replicates the partial
   * filters and publishes them unless the session was cancelled or no target still accepts filters.
   * A settle job releases a retired builder that a tracked task thread could not release. Every job
   * settles the session when it is invoked and when it is destroyed; destroying the publishing job
   * uninvoked ends the attempt as abandoned, or as cancelled if the session was cancelled. Jobs
   * never throw, and moving or wrapping one never allocates.
   */
  class accumulation_job final {
   public:
    accumulation_job() noexcept = default;
    ~accumulation_job();
    accumulation_job(accumulation_job const&)            = delete;
    accumulation_job& operator=(accumulation_job const&) = delete;
    accumulation_job(accumulation_job&& other) noexcept;
    accumulation_job& operator=(accumulation_job&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_state); }

    /**
     * @brief Runs the job on @p stream and leaves it empty.
     *
     * @pre The calling thread's current device is the publishing root, and the thread tracks no
     * allocation for it.
     */
    void operator()(::cuda::stream_ref stream) && noexcept;

   private:
    friend class dynamic_filter_publication_session;
    explicit accumulation_job(std::shared_ptr<state> operation) noexcept
      : _state(std::move(operation))
    {
    }
    std::shared_ptr<state> _state;
  };

  explicit dynamic_filter_publication_session(dynamic_filter_publish_plan plan = {},
                                              dynamic_filter_stats* stats      = nullptr);
  ~dynamic_filter_publication_session();
  dynamic_filter_publication_session(dynamic_filter_publication_session const&)            = delete;
  dynamic_filter_publication_session& operator=(dynamic_filter_publication_session const&) = delete;

  [[nodiscard]] dynamic_filter_publish_plan const& plan() const noexcept;
  void restrict_replicas_to(std::vector<int> const& admitted_gpu_ids);

  /**
   * @brief Starts accumulating against @p inventory. Call on an untracked thread (the task
   * creator).
   *
   * Selects the active keys, sizes their shared geometry, requires working peer DMA between every
   * pair of replica GPUs, and allocates zeroed per-GPU partial arrays under per-GPU leases. When it
   * returns false, the session behaves exactly as with accumulation disabled.
   *
   * @return true iff accumulation started; false when there is no inventory, no active key, a zero
   * or over-cap geometry, no peer DMA, a refused lease, a masked failure, or the session is not
   * open. The reason is counted.
   */
  [[nodiscard]] bool try_begin_accumulation(
    std::optional<complete_build_inventory> inventory) noexcept;

  /**
   * @brief Whether accumulation has started, whatever its outcome.
   */
  [[nodiscard]] bool accumulation_claimed() const noexcept;

  /**
   * @brief Contributes one certified original batch, on the contributing task's thread and stream.
   *
   * Validates the batch against the inventory, claims its entry, and enqueues its key inserts. A
   * batch whose entry is already claimed is counted as a duplicate and ignored.
   *
   * @param original_id The batch's original ID
   * @param source A read accessor of the batch; the caller retires @p stream before releasing it
   * @param stream The contributing task's stream
   * @return The publishing job for the final contribution; a settle job if a retired builder awaits
   * release; otherwise an empty job
   */
  [[nodiscard]] accumulation_job contribute(std::uint64_t original_id,
                                            cucascade::read_only_data_batch const& source,
                                            ::cuda::stream_ref stream) noexcept;

  /**
   * @brief Ends a collecting accumulation without a filter.
   *
   * @return A settle job if a retired builder awaits release; otherwise an empty job
   */
  [[nodiscard]] accumulation_job decline_accumulation(accumulation_decline reason) noexcept;

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
   *
   * A running publication may finish; an accumulation with an unclaimed batch ends without a
   * filter.
   */
  void finish_input() noexcept;

  /**
   * @brief Close input, cancel future publication, and request cancellation of an active attempt.
   */
  void cancel() noexcept;

 private:
  std::shared_ptr<state> _state;
};

static_assert(std::is_nothrow_invocable_v<dynamic_filter_publication_session::accumulation_job&&,
                                          ::cuda::stream_ref>);

struct dynamic_filter_publication_outcome {
  std::size_t accumulations_started                = 0;
  std::size_t accumulation_expected_contributions  = 0;
  std::size_t accumulation_completed_contributions = 0;
  std::size_t accumulation_duplicate_contributions = 0;
  std::size_t accumulations_skipped_inventory      = 0;
  std::size_t accumulations_skipped_admission      = 0;
  std::size_t accumulations_skipped_error          = 0;
  std::size_t accumulations_skipped_transient      = 0;
  std::size_t accumulations_incomplete             = 0;
  std::size_t accumulations_abandoned              = 0;
  std::size_t accumulation_storage_leaks           = 0;
  std::size_t accumulation_publications_finished   = 0;
  std::size_t accumulation_publication_latency_ns  = 0;
  std::size_t keys_skipped_bloom_size_gate         = 0;
  std::size_t keys_skipped_bloom_unsupported       = 0;
  std::size_t keys_considered                      = 0;
  std::size_t keys_with_known_domain               = 0;
  std::size_t keys_build_exceeded_domain           = 0;
  std::size_t skipped_targets_drained              = 0;
  std::size_t keys_skipped_domain_gate             = 0;
  std::size_t keys_skipped_type_mismatch           = 0;
  std::size_t membership_filters_built             = 0;
  std::size_t zone_map_filters_built               = 0;
  std::size_t active_targets                       = 0;
  std::size_t filters_pushed                       = 0;
  // Bindings whose recorded probe type no membership adapter can read (would decline every batch).
  std::size_t bindings_skipped_incompatible_probe = 0;
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
