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

/**
 * @file promotion.hpp
 * @brief One promotion of one batch to a higher tier: HOST->GPU, DISK->GPU or DISK->HOST.
 */

#include "query_id.hpp"

#include <cucascade/memory/common.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>

namespace cucascade {
class data_batch;
class idata_representation;
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius::exec {
class query_lifecycle_registry;
}  // namespace sirius::exec

namespace sirius::preload {

class preload_streams;

[[nodiscard]] constexpr std::size_t sat_add(std::size_t a, std::size_t b) noexcept
{
  return a > std::numeric_limits<std::size_t>::max() - b ? std::numeric_limits<std::size_t>::max()
                                                         : a + b;
}

[[nodiscard]] constexpr std::size_t sat_mul(std::size_t a, std::size_t b) noexcept
{
  if (a == 0 || b == 0) { return 0; }
  return a > std::numeric_limits<std::size_t>::max() / b ? std::numeric_limits<std::size_t>::max()
                                                         : a * b;
}

/// @p bytes plus the larger of 1 MiB and 1/64 of it: the slack every size estimate carries.
[[nodiscard]] constexpr std::size_t with_slack(std::size_t bytes) noexcept
{
  constexpr std::size_t min_slack = std::size_t{1} << 20;
  return sat_add(bytes, std::max(min_slack, bytes / 64));
}

/**
 * @brief Consumed bytes of a space: `max_memory - min(max_memory, available)`.
 *
 * The quantity cuCascade's downgrade trigger compares with its thresholds, so every preload
 * ceiling is in these units.
 */
[[nodiscard]] constexpr std::size_t consumed_bytes(std::size_t max_memory,
                                                   std::size_t available) noexcept
{
  return max_memory - std::min(max_memory, available);
}

/// The representations a promotion distinguishes, by exact type.
enum class repr_kind : std::uint8_t {
  host_data,  ///< cucascade::host_data_representation
  disk_data,  ///< cucascade::disk_data_representation
  other       ///< anything else: never promoted
};

/// The kind of @p data, matched on its exact dynamic type.
[[nodiscard]] repr_kind classify(cucascade::idata_representation const& data) noexcept;

/**
 * @brief Peak GPU bytes needed to materialize @p data on a GPU.
 *
 * `peak_materialization_bytes(data)`, plus, for host and disk tables, the INT64 copy of every
 * STRING column's INT32 offsets (`8 x (rows + 1)`, at any nesting depth): the upload widens those
 * offsets while the INT32 column is still allocated. Saturates instead of overflowing.
 */
[[nodiscard]] std::size_t promotion_peak_bytes(cucascade::idata_representation const& data);

/// What one promotion attempt ended with.
enum class promotion_outcome : std::uint8_t {
  promoted,    ///< the batch is now in the target space
  not_needed,  ///< the query is gone, or the batch is gone, busy or already where it should be
  no_room,     ///< the target space refused the reservation or is above the ceiling
  failed       ///< unsupported representation, or an error during the copy
};

/// Result of one promotion attempt.
struct promotion_result {
  promotion_outcome outcome{promotion_outcome::failed};
  /// When promoted: bytes reserved for the copy (the exact need, with slack for a GPU).
  std::size_t bytes{0};
  /// When not needed because the batch is already off the source tier: the tier it is in.
  /// @c Tier::SIZE otherwise.
  cucascade::memory::Tier found{cucascade::memory::Tier::SIZE};
};

/// One promotion, built under the executor's lock. Holds no strong reference to the batch.
struct promotion_job {
  std::weak_ptr<cucascade::data_batch> batch;
  std::uint64_t batch_id{0};
  std::uint64_t task_id{0};
  query_id_t query{};
  /// GPU or HOST.
  cucascade::memory::Tier tier{cucascade::memory::Tier::SIZE};
  /// Index of the target in the executor's GPU or HOST list.
  std::size_t space{0};
  /// The target space itself.
  cucascade::memory::memory_space* target{nullptr};
  /// Bytes reserved before the batch is locked; also the budget the executor charged.
  std::size_t est_bytes{0};
  /// The target's preload ceiling, in consumed bytes.
  std::size_t ceiling_bytes{0};
};

/// Everything @ref promote_one reads besides the job. Borrowed; must outlive the call.
struct promotion_env {
  exec::query_lifecycle_registry& lifecycle;
  /// Preload's stream pools. Required for GPU targets, unused for HOST targets.
  preload_streams const* streams{nullptr};
  /// Tests only. Runs once the target reservation is made and admitted, before the batch is
  /// locked.
  std::function<void(promotion_job const&)> test_hook{};
};

/**
 * @brief Promote one batch to the job's target. Never throws and never waits for a batch lock.
 *
 * In order, each refusal returning at once:
 * 1. A work lease on the job's query, held for the whole call (@c not_needed without one).
 * 2. A non-blocking reservation of @c est_bytes in the target (@c no_room), then the ceiling:
 *    consumed bytes, counting the reservation, at or below @c ceiling_bytes and the space below
 *    its downgrade trigger (@c no_room).
 * 3. The batch: alive and lockable with @c try_to_mutable (@c not_needed), and not already
 *    there: on a GPU for a GPU target, off DISK for a HOST target (@c not_needed, with
 *    @ref promotion_result::found).
 * 4. The representation: a GPU takes a host or disk table, a HOST a disk table (@c failed).
 * 5. The exact need (@ref promotion_peak_bytes with slack for a GPU, the disk size for HOST):
 *    the reservation grows to it (@c no_room), and the ceiling is checked again (@c no_room).
 *
 * A GPU copy runs on a preload stream after the source's writer event, with the reservation
 * attached to the stream's allocation tracker under a fail-on-overrun policy, so a low estimate
 * throws @c rmm::out_of_memory (@c failed) instead of exceeding the reservation. A HOST copy
 * draws its blocks from the reservation (@c no_room when the pool runs out). Any other
 * exception gives @c failed; a fatal CUDA error is also reported to the query's lifecycle.
 */
[[nodiscard]] promotion_result promote_one(promotion_job const& job,
                                           promotion_env const& env) noexcept;

}  // namespace sirius::preload
