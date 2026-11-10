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
 * @file build_filter_construction.hpp
 * @brief Builds every planned dynamic filter of one hash join from the batches of its certified
 * build input.
 *
 * `build_filter_construction` is the mechanism behind a `construction_plan`: it owns the three
 * accumulators (`accumulated_bloom_builder` for Bloom keys, `exact_key_stager` for exact IN-lists,
 * `bounds_accumulator` for zone maps and DECIMAL128 fit checks), assigns each batch its staging
 * slices and bound slots by host arithmetic, and constructs the filters on the publication root. A
 * publication session drives it: `try_create` on the creator thread, `reserve` under the session
 * mutex, `contribute` outside it on each task, and `finish` once on the task that completes the
 * inventory.
 */

#pragma once

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/dynamic_filter/construction_plan.hpp"
#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
#include "op/dynamic_filter/detail/exact_key_stager.hpp"
#include "op/dynamic_filter/dynamic_filter_key_domain.hpp"
#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"
#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::op {

namespace detail {
class bounds_accumulator;
}  // namespace detail

/**
 * @brief What a batch's columns hold when a task claims it
 */
struct batch_key_facts {
  std::uint64_t rows;                 ///< The batch's rows
  std::vector<column_facts> columns;  ///< Its top-level columns, by ordinal

  /**
   * @brief The facts of @p table, read from its column views without waiting on the host
   */
  [[nodiscard]] static batch_key_facts of(cudf::table_view const& table);
};

/**
 * @brief Builds every planned filter of one hash join from its certified build batches
 */
class build_filter_construction final {
 public:
  /**
   * @brief Where one claimed batch writes: its batch index, its GPU, and its exact keys' slices
   */
  struct contribution_ticket {
    /** @brief The batch's index in the inventory's `batches()`: its bound slot */
    std::size_t batch_index;
    /** @brief The GPU that holds the batch */
    rmm::cuda_device_id device;
    /** @brief One slice per exact key, in the stager's key order */
    std::vector<detail::exact_key_stager::slice> slices;
  };

  /**
   * @brief The task that publishes: its GPU's memory space, stream, and allocator
   */
  struct publication_root {
    /** @brief The root GPU's memory space; the room left in it is checked before each root filter
     */
    cucascade::memory::memory_space& space;
    /** @brief The publishing task's stream, on the root GPU */
    ::cuda::stream_ref stream;
    /** @brief The publishing task's allocator: publication scratch and the root filters are charged
     * to it */
    rmm::device_async_resource_ref scratch;
  };

  /**
   * @brief The filters constructed for one planned key, ready for replication and fan-out
   */
  struct constructed_key {
    /** @brief The key's index in the plan's admitted keys */
    std::size_t admitted_key_index;
    /** @brief The key's zone map, if one was planned and constructed */
    std::shared_ptr<sirius_dynamic_filter> zone_map;
    /** @brief The key's membership filter, if one was planned and constructed */
    std::shared_ptr<sirius_dynamic_filter> membership;
    /** @brief The membership filter's domain, classified from the construction type */
    std::optional<membership_key_domain> domain;
    /** @brief Why a planned filter of the key was dropped (`decimal128_unfit`,
     * `construction_admission`, or `construction_invariant`); empty when every planned filter was
     * constructed */
    std::optional<dynamic_filter_key_reason> dropped;
  };

  /**
   * @brief Everything `finish` constructed
   */
  struct constructed_filters {
    /** @brief One entry per planned key, in plan order */
    std::vector<constructed_key> keys;
    /** @brief `filters_skipped_construction_admission` and `filters_skipped_construction_invariant`
     */
    dynamic_filter_stats_snapshot counts;
    /** @brief How the Bloom filters reached their replicas, when any key planned one */
    std::optional<detail::accumulated_bloom_builder::publication_transport> transport;
  };

  /**
   * @brief Leases every accumulator's storage for @p plan over @p inventory on every GPU of @p
   * targets. Call on an untracked thread (the task creator); waits on the host for nothing.
   *
   * @throw std::logic_error if a planned key's column is absent from @p inventory
   * @return The construction, or nullopt iff a lease was refused, after releasing everything built
   * so far
   */
  [[nodiscard]] static std::optional<build_filter_construction> try_create(
    construction_plan plan,
    complete_build_inventory const& inventory,
    std::span<dynamic_filter_replica_space const> targets);

  ~build_filter_construction();
  build_filter_construction(build_filter_construction const&)            = delete;
  build_filter_construction& operator=(build_filter_construction const&) = delete;
  build_filter_construction(build_filter_construction&&) noexcept;
  build_filter_construction& operator=(build_filter_construction&&) noexcept;

  /**
   * @brief Validates a claimed batch against the inventory and assigns its slices and slot.
   *
   * Host arithmetic only. For every planned key, the batch's carrier must equal the one the
   * inventory recorded for that batch, and its null count must equal the recorded count when that
   * count is known; the row count must match too.
   *
   * @return The ticket, or nullopt on any mismatch (the caller ends the attempt)
   */
  [[nodiscard]] std::optional<contribution_ticket> reserve(std::size_t batch_index,
                                                           rmm::cuda_device_id device,
                                                           batch_key_facts const& actual) const;

  /**
   * @brief Adds @p input's keys to every accumulator on the ticket's GPU, without a host wait
   * except for STRING zone-map bounds.
   *
   * If an accumulator fails, the work already accepted on @p stream is joined before the failure
   * propagates.
   *
   * @pre @p ticket came from `reserve` for this batch, and the caller retires @p stream before
   * releasing @p input
   * @param scratch The contributing task's allocator, for temporaries
   * @return false, before anything is enqueued, iff the ticket's GPU holds no storage
   */
  [[nodiscard]] bool contribute(contribution_ticket const& ticket,
                                cudf::table_view const& input,
                                ::cuda::stream_ref stream,
                                rmm::device_async_resource_ref scratch);

  /**
   * @brief Constructs every planned filter on @p root; waits on the host.
   *
   * Runs on the publishing task's thread, which holds the task's reservation on the root GPU. Bloom
   * filters come from their partials. Exact IN-lists and zone maps are built on the root task's
   * stream with storage from its allocator. Before each, the room left
   * (`memory_space::get_available_memory` on the task's stream: the unused part of the reservation
   * tracked for this thread or stream, plus unreserved capacity) must cover the filter's upper
   * bound (`in_list_storage_upper_bound`, or the bound scalars' bytes). A filter without that room,
   * or whose construction runs out of memory anyway, is dropped alone
   * (`filters_skipped_construction_admission`). A DECIMAL128 key whose merged bounds do not fit
   * int64 publishes no membership filter (`decimal128_unfit`); a staged key that overflowed its
   * slices publishes none either (`filters_skipped_construction_invariant`).
   *
   * Waits on the host for the Bloom merge, the bound slots, the staging gather, the restore of
   * narrower carriers' bounds, and once after every root filter is enqueued.
   *
   * @pre Every `contribute` returned; this is the only `finish` call; the root GPU is current
   * @return The filters, or nullopt when the Bloom publication workspace is refused or the root GPU
   * holds no storage
   */
  [[nodiscard]] std::optional<constructed_filters> finish(publication_root const& root);

  /**
   * @brief Bytes one task may take from its allocator: a contribution's temporaries plus the
   * publication's Bloom merge scratch, for the largest batch.
   */
  [[nodiscard]] std::size_t task_scratch_bytes() const noexcept;

  /**
   * @brief Bytes `finish` may take from the root task's allocator beyond `task_scratch_bytes`: the
   * larger of the root's IN-lists plus zone-map scalars, and the temporaries of restoring narrower
   * carriers' bounds.
   *
   * The characters of STRING bounds are known only at publish and are not included; each zone map
   * is still checked against the room left before it is built.
   */
  [[nodiscard]] std::size_t root_construction_bytes() const noexcept;

  /**
   * @brief True iff releasing every accumulator's storage now credits no allocation tracker.
   */
  [[nodiscard]] bool releasable_here() const noexcept;

  [[nodiscard]] construction_plan const& plan() const noexcept;

 private:
  struct impl;
  explicit build_filter_construction(std::unique_ptr<impl> state) noexcept;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op
