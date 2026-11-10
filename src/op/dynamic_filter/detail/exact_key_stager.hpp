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

struct accumulation_partial;

/**
 * @brief Stages the valid keys of every build batch for the exact IN-lists, then builds each list
 * once from the complete set.
 *
 * Each exact key gets, on every replica GPU, one buffer that holds all of the key's valid rows at
 * its rep, leased on the untracked creator thread; a key with no valid rows gets no buffer. A batch
 * writes its valid rows into its own slice of the buffer on the GPU that holds it, at the slice's
 * offset (`prepare_build_keys`, bounded by the slice, with an overflow word). `gather` copies every
 * other GPU's written slices into the root GPU's buffer, so the root holds the valid rows of the
 * whole build in slice order, and `gathered::build` builds the planned small or hash IN-list from
 * it: the same list the IN-list column constructors build from the whole build. Contributions never
 * wait on the host; `gather` does.
 */
class exact_key_stager final {
  struct impl;

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

  /**
   * @brief An upper bound on the task scratch one key's `enqueue_add` takes from a batch of up to
   * @p rows rows: the valid-row ranks and the scan's temporaries.
   */
  [[nodiscard]] static std::size_t contribution_scratch_bytes(std::size_t rows);

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
   * @brief The whole build's staged rows on the root GPU, after `gather`: builds each key's list
   * from them.
   *
   * Refers to its stager, which must outlive it and must not be moved from while it is in use.
   */
  class gathered {
   public:
    /**
     * @brief Whether key @p key_index staged more valid rows than its slices hold on some GPU, so
     * its rows are incomplete.
     */
    [[nodiscard]] bool overflowed(std::size_t key_index) const;

    /**
     * @brief Enqueues the construction of key @p key_index's planned list on @p stream, with its
     * storage from @p mr.
     *
     * The caller waits on @p stream before the list becomes visible or the stager is released.
     *
     * @pre @p stream is the stream `gather` ran on, or follows it
     * @throw accumulation_invariant_error if the key overflowed: its list would have false
     * negatives
     */
    [[nodiscard]] std::shared_ptr<sirius_dynamic_filter> build(
      std::size_t key_index, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr) const;

   private:
    friend class exact_key_stager;
    gathered(impl const& state, accumulation_partial const& root, std::vector<bool> overflowed);

    impl const* _state;
    accumulation_partial const* _root;
    std::vector<bool> _overflowed;
  };

  /**
   * @brief Copies every other GPU's written slices into @p root's buffers and reads every overflow
   * word.
   *
   * Waits on the host for the copies and the words together, after first waiting for any other GPU
   * that wrote slices (their copies may be staged through the host). Afterwards the root buffers
   * hold every staged row and are ordered before later work on @p stream.
   *
   * @pre Every `enqueue_add` returned; this is the only `gather` call; @p root is the current
   * device
   * @param stream A stream of @p root
   * @return The gathered rows, or nullopt when @p root holds no partial
   */
  [[nodiscard]] std::optional<gathered> gather(rmm::cuda_device_id root, ::cuda::stream_ref stream);

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
  explicit exact_key_stager(std::unique_ptr<impl> state) noexcept;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::detail
