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
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/table/table_view.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace cucascade {
class data_batch;
class read_only_data_batch;
}  // namespace cucascade

namespace sirius::op {

class sirius_physical_hash_join;

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
 * `complete_build_inventory`. Accumulation moves through the phases OPEN, COLLECTING, PUBLISHING,
 * and TERMINAL. Recoverable optional allocation and launch failures end accumulation locally;
 * invalid internal state and failed CUDA cleanup propagate. The session retains
 * initialization-leased arrays, including completed filters, until `release_retired_storage` can
 * safely free them outside task tracking (the filters cannot be owned be a single task). Engine
 * task drain precedes session destruction.
 */
class dynamic_filter_publication_session final {
  struct state;

 public:
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
   * returns false, the session behaves exactly as with accumulation disabled. Transitions the
   * session OPEN -> COLLECTING.
   *
   * @return true iff accumulation started; false when there is no inventory, no active key, a zero
   * or over-cap geometry, no peer DMA, a refused lease, a recoverable failure, or the session is
   * not open. The reason is counted.
   */
  [[nodiscard]] bool try_begin_accumulation(std::optional<complete_build_inventory> inventory);

  /**
   * @brief Whether accumulation has started, whatever its outcome.
   */
  [[nodiscard]] bool accumulation_claimed() const noexcept;

  /**
   * @brief Contributes one certified original batch, on the contributing task's thread and stream.
   *
   * Claims each original identity once and enqueues its active key inserts. Already claimed
   * identities are counted as duplicates before inspecting their representation; pending entries
   * must match their exact row count and active key types. Transitions the session COLLECTING ->
   * PUBLISHING when the last contribution completes the inventory.
   *
   * @param original_id The batch's original ID
   * @param source A read accessor of the batch; the caller retires @p stream before releasing it
   * @param stream The contributing task's stream
   */
  void contribute(std::uint64_t original_id,
                  cucascade::read_only_data_batch const& source,
                  ::cuda::stream_ref stream);

  /**
   * @brief Publishes synchronously if @p original_id completed the inventory and still owns its
   * pending publication.
   *
   * A mandatory scatter retry preserves the pending identity. The caller invokes this after
   * constructing scatter output and before depositing it. Scratch borrows @p task_space's
   * allocator; all replicas are ready before any channel push.
   *
   * @param original_id Original input identity retained across preparation and retries
   * @param task_space The task's GPU memory space
   * @param stream The task's stream, on the calling thread's current device
   */
  void publish_if_final(std::uint64_t original_id,
                        cucascade::memory::memory_space const& task_space,
                        ::cuda::stream_ref stream);

  /**
   * @brief Ends a collecting accumulation whose task input cannot identify one certified batch.
   *
   * Transitions the session COLLECTING -> TERMINAL.
   */
  void decline_accumulation() noexcept;

  /**
   * @brief Releases terminal accumulation storage when no session operation is active and its
   * allocators permit release.
   *
   * An active or tracked caller retains the builder until a later safe release or session
   * destruction.
   */
  void release_retired_storage() noexcept;

  /**
   * @brief Calls `finish_input`, then `release_retired_storage`.
   *
   * After build tasks drain, `sirius_physical_partition::on_finalize_operator` calls this for an
   * accumulating join, and `sirius_physical_hash_join::on_finalize_operator` calls it for every
   * join. Transitions the session to TERMINAL.
   */
  void finalize_input() noexcept;

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
   * a claimed attempt and propagate, including deposit allocation failures. Transitions the session
   * OPEN -> PUBLISHING (single-batch builds).
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
   * A running publication may finish; an incomplete accumulation ends without a filter, and an
   * unclaimed pending publication ends as abandoned. Storage release remains separate.
   */
  void finish_input() noexcept;

  /**
   * @brief Close input, cancel future publication, and request cancellation of an active attempt.
   *
   * An owed publication that never ran ends as cancelled. Transitions the session to TERMINAL.
   */
  void cancel() noexcept;

 private:
  std::unique_ptr<state> _state;
};

/**
 * @brief The counts one whole-build publication adds to `dynamic_filter_stats`, and the number of
 * targets that still accepted filters.
 */
struct dynamic_filter_publication_outcome : dynamic_filter_stats_snapshot {
  std::size_t active_targets = 0;
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
