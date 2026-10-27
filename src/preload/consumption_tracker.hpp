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

#include "query_id.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace sirius::preload {

/// How a task found one of its inputs that was downgraded off the GPU during its query.
enum class input_readiness : std::uint8_t {
  ready,     ///< preload had already promoted it to the GPU
  waited,    ///< it reached the GPU while the task was locking it, typically behind a promotion
  upgraded,  ///< still on HOST or DISK: the task copied it to the GPU itself
};

/// Per-query totals of @ref input_readiness.
struct input_readiness_counts {
  std::uint64_t ready{0};
  std::uint64_t waited{0};
  std::uint64_t waited_ns{0};  ///< lock wait of the @c waited inputs
  std::uint64_t upgraded{0};
  std::uint64_t upgraded_ns{0};  ///< time the @c upgraded inputs took to prepare
};

/**
 * @brief Tells whether preload put a task's inputs on the GPU before the task asked for them.
 *
 * Preload reports each downgraded batch and the start and end of each GPU promotion; tasks report
 * how each such batch was prepared (see @c lock_and_prepare_trace). A batch counts at its first
 * preparation after a downgrade and is then forgotten until the next one.
 *
 * Process-wide, because tasks prepare inputs without a handle to the preload executor. Inactive
 * until a preload executor enables it, so tasks pay one relaxed load per input otherwise. Every
 * other call takes a leaf lock for O(1) work.
 */
class consumption_tracker {
 public:
  /// Where a tracked batch was when a task started preparing it.
  enum class batch_state : std::uint8_t {
    downgraded,  ///< on HOST or DISK
    promoting,   ///< a promotion to a GPU is running
    promoted,    ///< preload put it on a GPU
  };

  [[nodiscard]] static consumption_tracker& instance() noexcept;

  /// Start tracking; enables nest, so each call needs a matching @ref disable.
  void enable() noexcept;
  /// Undo one @ref enable. The last one forgets every batch and count.
  void disable() noexcept;
  [[nodiscard]] bool enabled() const noexcept
  {
    return _enabled.load(std::memory_order_relaxed) > 0;
  }

  // ---- preload side -------------------------------------------------------------------------

  /// Batch @p batch_id of query @p q left the GPU.
  void note_downgraded(query_id_t q, std::uint64_t batch_id) noexcept;
  /// A promotion of @p batch_id to a GPU started.
  void note_promotion_started(std::uint64_t batch_id) noexcept;
  /// The promotion of @p batch_id ended; @p promoted when the batch is now on the GPU.
  void note_promotion_finished(std::uint64_t batch_id, bool promoted) noexcept;

  // ---- task side ----------------------------------------------------------------------------

  /// The state of @p batch_id as a task starts preparing it for a GPU, or nullopt when it is not
  /// tracked (it never left the GPU, or tracking is off).
  [[nodiscard]] std::optional<batch_state> begin_input(std::uint64_t batch_id) noexcept;

  /**
   * @brief Record how a task prepared @p batch_id, given what @ref begin_input returned.
   *
   * @param before The state @ref begin_input returned.
   * @param upgraded_to_gpu The task converted the batch from HOST or DISK itself.
   * @param lock_wait_ns Time the task waited for the batch's locks.
   * @param prepare_ns Time the task spent preparing the batch.
   */
  void end_input(std::uint64_t batch_id,
                 batch_state before,
                 bool upgraded_to_gpu,
                 std::uint64_t lock_wait_ns,
                 std::uint64_t prepare_ns) noexcept;

  /// The counts of query @p q, removing it and its tracked batches.
  [[nodiscard]] input_readiness_counts take_query(query_id_t q) noexcept;

  /// How a task found an input that was in @p before: the classification behind @ref end_input.
  [[nodiscard]] static input_readiness classify(batch_state before, bool upgraded_to_gpu) noexcept;

 private:
  struct entry {
    query_id_t query;
    batch_state state;
  };

  /// Tracked batches beyond which new downgrades are ignored, bounding memory if queries are
  /// never forgotten.
  static constexpr std::size_t max_entries = std::size_t{1} << 20;

  std::atomic<int> _enabled{0};
  std::mutex _mu;
  std::unordered_map<std::uint64_t, entry> _batches;
  std::unordered_map<query_id_t, input_readiness_counts> _counts;
};

}  // namespace sirius::preload
