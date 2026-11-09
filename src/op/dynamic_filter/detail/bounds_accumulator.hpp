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

#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op::detail {

/**
 * @brief Accumulates each key's minimum and maximum over the valid rows of every build batch, then
 * merges them into one pair at the key's recorded type.
 *
 * A batch's bounds are stored at the carrier the batch arrived with, in its own slot: for numeric,
 * temporal, decimal, and BOOL8 keys a device slot (`batches x 2 x 16` bytes per key on every
 * replica GPU, leased on the untracked creator thread) written by a CUB reduction that never waits
 * on the host; for STRING keys two host strings, read back from cuDF reductions, which is the one
 * host wait a contribution may make. `finish` takes the minimum and maximum per carrier on the
 * host, restores each carrier's pair to the recorded type (`restore_scalar`), and compares the
 * candidates: the bounds a single reduction over the whole build at the recorded type would give.
 */
class bounds_accumulator final {
 public:
  /**
   * @brief One key whose bounds are accumulated.
   */
  struct key {
    cudf::size_type ordinal;   ///< The key's column in every contributed table
    cudf::data_type recorded;  ///< The type the merged bounds are restored to
  };

  /**
   * @brief Leases every numeric key's slots on every GPU of @p targets, without waiting on the
   * host.
   *
   * @throw accumulation_invariant_error if a key's recorded type has no supported bounds
   * @param batches Batches of the build: the number of slots per key
   * @return The accumulator, or nullopt iff a lease was refused, after releasing everything built
   * so far
   */
  [[nodiscard]] static std::optional<bounds_accumulator> try_create(
    std::vector<key> keys,
    std::size_t batches,
    std::span<dynamic_filter_replica_space const> targets);

  ~bounds_accumulator();
  bounds_accumulator(bounds_accumulator const&)            = delete;
  bounds_accumulator& operator=(bounds_accumulator const&) = delete;
  bounds_accumulator(bounds_accumulator&&) noexcept;
  bounds_accumulator& operator=(bounds_accumulator&&) noexcept;

  /**
   * @brief Writes @p input's bounds of every key into slot @p batch_index.
   *
   * Thread-safe across distinct batches. A batch without valid rows of a key leaves that key's slot
   * empty; that is known on the host from the column's null count. Numeric keys never wait on the
   * host: temporaries come from @p scratch, and the partial's stream then waits for the writes.
   * STRING keys wait for their two reductions, whose results land in pinned host memory rather than
   * in @p scratch, then store the strings under an internal leaf mutex that is never held across
   * CUDA work.
   *
   * @pre Each batch index is contributed at most once, @p input holds every key column at its
   * ordinal, and the caller retires @p stream before releasing @p input
   * @throw accumulation_invariant_error if @p batch_index is out of range, @p stream belongs to
   * another GPU, or a column's type has no supported bounds
   * @return false, before anything is enqueued, iff @p device holds no partial
   */
  [[nodiscard]] bool enqueue_add(std::size_t batch_index,
                                 cudf::table_view const& input,
                                 rmm::cuda_device_id device,
                                 ::cuda::stream_ref stream,
                                 rmm::device_async_resource_ref scratch);

  /**
   * @brief One key's merged bounds at its recorded type, held on the host.
   *
   * The bounds become device scalars only when a filter needs them (`to_scalars`), so the caller
   * decides where, and whether, that memory is taken.
   */
  struct key_bounds {
    /** @brief A numeric, temporal, decimal, or BOOL8 key's minimum and maximum by integer storage
     */
    using storage_bounds = std::pair<__int128_t, __int128_t>;
    /** @brief A STRING key's minimum and maximum */
    using string_bounds = std::pair<std::string, std::string>;

    /** @brief The key's recorded type */
    cudf::data_type type;
    /** @brief The smallest and the largest valid value */
    std::variant<storage_bounds, string_bounds> values;
    /** @brief Whether a DECIMAL128 key's values fit its int64 membership rep (the
     * `column_values_fit` rule applied to the bounds); true for every other type */
    bool fits_membership_rep;

    /**
     * @brief Device bytes `to_scalars` takes from its resource: each bound's value (a STRING's
     * characters) and its validity flag, as allocation trackers charge them.
     */
    [[nodiscard]] std::size_t device_bytes() const noexcept;

    /**
     * @brief The minimum and the maximum as scalars of `type`, with storage from @p mr; never waits
     * on the host.
     */
    [[nodiscard]] std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>>
    to_scalars(::cuda::stream_ref stream, rmm::device_async_resource_ref mr) const;
  };

  /**
   * @brief Device bytes `key_bounds::to_scalars` takes for a key of fixed-width type @p recorded.
   */
  [[nodiscard]] static std::size_t fixed_width_scalar_bytes(cudf::data_type recorded) noexcept;

  /**
   * @brief Reads every slot and merges each key's bounds on the host.
   *
   * Waits on the host once for all slots of every GPU. A key that arrived at a carrier other than
   * its recorded type restores that carrier's pair with `restore_scalar` (which waits) and reads
   * the restored values back, once for all such keys. Temporaries of that restore come from @p mr
   * and are released before returning. Must not run concurrently with `enqueue_add`: the caller
   * orders every contribution before it.
   *
   * @pre The current device is the GPU of @p stream
   * @return One entry per key, in key order; none for a key without a valid row in any batch
   */
  [[nodiscard]] std::vector<std::optional<key_bounds>> finish(::cuda::stream_ref stream,
                                                              rmm::device_async_resource_ref mr);

  /**
   * @brief True iff releasing the slots now credits no allocation tracker.
   */
  [[nodiscard]] bool releasable_here() const noexcept;

 private:
  struct impl;
  explicit bounds_accumulator(std::unique_ptr<impl> state) noexcept;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::op::detail
