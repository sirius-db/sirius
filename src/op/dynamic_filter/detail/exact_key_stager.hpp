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

#pragma once

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op::detail {

/**
 * @brief Stages the valid keys of every build batch for the exact IN-lists, then builds each list
 * once from the complete set.
 *
 * Each exact key gets, on every replica GPU, one buffer that holds all of the key's valid rows at
 * its rep, leased on the untracked creator thread; a key with no valid rows gets no buffer. A batch
 * writes its valid rows into its own slice of the buffer on the GPU that holds it, at the slice's
 * offset (`prepare_build_keys`, bounded by the slice, with an overflow word). `finish` copies every
 * other GPU's written slices into the root GPU's buffer, so the root holds the valid rows of the
 * whole build in slice order, and builds the planned small or hash IN-list from it: the same list
 * the IN-list column constructors build from the whole build. Contributions never wait on the host;
 * `finish` does.
 */
class exact_key_stager final {
 public:
  /**
   * @brief One exact key.
   */
  struct key {
    /** @brief The key's column in every contributed table */
    cudf::size_type ordinal;
    /** @brief The construction type: the list's domain is classified from it, and each
     * contribution's column must be one of its carriers */
    cudf::data_type type;
    /** @brief `membership_filter_kind::small_in_list` or `membership_filter_kind::hash_in_list` */
    membership_filter_kind kind;
    /** @brief The key's valid rows across the whole build: the buffer's length */
    std::size_t valid_rows;
  };

  /**
   * @brief Where one batch's valid rows of one key go in that key's buffer.
   */
  struct slice {
    std::size_t offset;  ///< First row of the slice
    std::size_t rows;    ///< The batch's valid rows of the key
  };

  /**
   * @brief Leases and zeroes every key's buffer and overflow word on every GPU of @p targets
   * without waiting on the host.
   *
   * @throw accumulation_invariant_error if a key's kind is not an IN-list, its type has no key
   * domain, or a small IN-list's row count is out of range
   * @return The stager, or nullopt iff a lease was refused, after releasing everything built so far
   */
  [[nodiscard]] static std::optional<exact_key_stager> try_create(
    std::vector<key> keys, std::span<dynamic_filter_replica_space const> targets);

  ~exact_key_stager();
  exact_key_stager(exact_key_stager const&)            = delete;
  exact_key_stager& operator=(exact_key_stager const&) = delete;
  exact_key_stager(exact_key_stager&&) noexcept;
  exact_key_stager& operator=(exact_key_stager&&) noexcept;

  /**
   * @brief Writes @p input's valid rows of every key into that key's slice on @p device, without a
   * host wait.
   *
   * Thread-safe. Temporaries come from @p scratch and free stream-ordered on @p stream; the
   * partial's stream then waits for the writes, so every later read or free of the buffers follows
   * them. A valid row beyond its slice is not written; it sets the key's overflow word instead.
   *
   * @pre @p input holds every key column at its ordinal at a carrier of the key's type, and the
   * caller retires @p stream before releasing @p input
   * @throw accumulation_invariant_error if @p stream belongs to another GPU, a slice lies outside
   * its buffer, or a key column is not a carrier of its key, before anything is enqueued for it
   * @param slices One slice per key, in key order
   * @param input A build batch
   * @param device The GPU that holds @p input
   * @param stream The contributing task's stream
   * @param scratch The contributing task's allocator
   * @return false, before anything is enqueued, iff @p device holds no partial
   */
  [[nodiscard]] bool enqueue_add(std::span<slice const> slices,
                                 cudf::table_view const& input,
                                 rmm::cuda_device_id device,
                                 ::cuda::stream_ref stream,
                                 rmm::device_async_resource_ref scratch);

  /**
   * @brief The lists built from the staged keys.
   */
  struct publication {
    /** @brief One list per key, in key order; null for a key whose staging overflowed */
    std::vector<std::shared_ptr<sirius_dynamic_filter>> filters;
    /** @brief `filters_skipped_construction_invariant` counts the dropped keys */
    dynamic_filter_stats_snapshot counts;
  };

  /**
   * @brief Gathers every GPU's written slices into @p root's buffers, checks the overflow words,
   * and builds each key's list on @p root; waits on the host before returning.
   *
   * A key whose overflow word is set on any GPU publishes no list: some valid row was dropped, so
   * the list would have false negatives. That is logged as an error and counted.
   *
   * @pre Every `enqueue_add` returned; this is the only `finish` call; @p root is the current
   * device
   * @param root The GPU that builds the lists
   * @param stream A stream of @p root
   * @param mr Resource for the lists' storage
   * @return The lists, or nullopt when @p root holds no partial
   */
  [[nodiscard]] std::optional<publication> finish(rmm::cuda_device_id root,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr);

  /**
   * @brief True iff releasing the buffers now credits no allocation tracker.
   */
  [[nodiscard]] bool releasable_here() const noexcept;

  /**
   * @brief Copies key @p key_index's buffer on @p device to the host, after the work queued on it.
   *
   * For tests and diagnostics; waits on the host.
   *
   * @return The buffer's bytes, or nullopt for an absent key, GPU, or buffer
   */
  [[nodiscard]] std::optional<std::vector<std::byte>> inspect_buffer(
    std::size_t key_index, rmm::cuda_device_id device) const;

 private:
  struct impl;
  explicit exact_key_stager(std::unique_ptr<impl> state) noexcept;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::detail
