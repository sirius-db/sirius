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
#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cucascade {
class read_only_data_batch;
}  // namespace cucascade

namespace sirius::op {

/**
 * @brief Owns one hash join's optional dynamic filter publication and endpoint completion rights
 *
 * 3 responsibilities:
 *  - Own the producer handles for this join's target channels
 *  - Run at most one construction attempt over the certified build and publish its filters
 *  - Finish those handles after publication, cancellation, or failure
 *
 * Ownership relationship: 1 join -> 1 publication session -> 1 producer handle per target channel.
 *
 * A session's construction site (the build PARTITION, or BUILD_FILTER where a join has one)
 * certifies the build input, then either declines (`decline`) or plans and leases construction
 * (`begin`); its tasks contribute every certified batch (`contribute`), and the task whose
 * contribution completes the inventory constructs, replicates, and publishes every planned filter
 * (`publish_if_final`) through `build_filter_construction`. The session moves through the phases
 * OPEN, COLLECTING, PUBLISHING, and TERMINAL, and is Terminal once its site finalizes
 * (`finalize_input`). Recoverable optional allocation and launch failures end an attempt locally;
 * invalid internal state and failed CUDA cleanup propagate. The session retains the leased
 * accumulator storage until `release_retired_storage` can safely free it outside task tracking.
 * Engine task drain precedes session destruction.
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
   * @brief Names the join in this session's `[dynamic_filter_key]` lines
   *
   * The session emits one such DEBUG line per admitted key with a binding when it becomes terminal,
   * after releasing its mutex (see `dynamic_filter_key_outcome.hpp`). `sirius_pipeline_converter`
   * sets the identity once operators are numbered; until then the lines carry an empty identity.
   */
  void set_join_identity(dynamic_filter_join_identity identity) noexcept;

  /**
   * @brief Plans and leases construction for @p inventory. Call on an untracked thread (the task
   * creator) after certification.
   *
   * OPEN only; otherwise a no-op that counts nothing. Seals the plan, plans every key
   * (`plan_construction`), and leases the accumulators (`build_filter_construction::try_create`),
   * recording one `publication_attempts`. An empty build or plan, no replica GPU, or a refused
   * lease ends the attempt SKIPPED at once, so the session is Terminal; otherwise the session moves
   * OPEN -> COLLECTING.
   *
   * @return true iff the session started collecting contributions
   */
  bool begin(complete_build_inventory inventory);

  /**
   * @brief Records why the construction site declined its build input. OPEN only: counts @p reason
   * and completes SKIPPED (Terminal); otherwise a no-op.
   */
  void decline(construction_decline reason) noexcept;

  /**
   * @brief Contributes one certified original batch, on the contributing task's thread and stream.
   *
   * Claims each original identity once, then, under the session mutex, checks the batch against
   * what the inventory recorded (`build_filter_construction::reserve`) and, outside it, adds its
   * keys to every accumulator. Already claimed identities are counted as duplicates before
   * inspecting their representation. Transitions the session COLLECTING -> PUBLISHING when the last
   * contribution completes the inventory.
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
   * A mandatory retry preserves the pending identity. The caller invokes this after constructing
   * its output and before depositing it. Root filters and scratch come from @p task_space's
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
   * @brief Ends a collecting attempt whose task input cannot identify one certified batch.
   *
   * COLLECTING only: the attempt ends with `accumulations_skipped_inventory` and settles; otherwise
   * a no-op.
   */
  void decline_task_input() noexcept;

  /**
   * @brief Bytes any construction-site task may take from its allocator for this session: a
   * contribution's temporaries and the publication's scratch, plus the root filters, since any task
   * may be the one that publishes. 0 while the session is not constructing: before `begin` starts
   * an attempt, and once the attempt has ended, although construction is released only when its
   * site finalizes.
   */
  [[nodiscard]] std::size_t task_scratch_bound() const noexcept;

  /**
   * @brief Whether the session has completed every producer right: published, skipped, failed, or
   * cancelled.
   */
  [[nodiscard]] bool terminal() const noexcept;

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
   * After its tasks drain, the construction site calls this unconditionally, and
   * `sirius_physical_hash_join::on_finalize_operator` calls it again as a backstop. Transitions the
   * session to TERMINAL.
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

}  // namespace sirius::op
