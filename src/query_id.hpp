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

#include <atomic>
#include <cstdint>
#include <format>
#include <stdexcept>

namespace sirius {

/**
 * @brief Identifies one execution window, i.e. one query, within a SiriusContext.
 *
 * Minted once when the window opens (`SiriusContext::StandaloneQueryScope`) and used as THE
 * query identity everywhere downstream: which data repository manager the query owns, which
 * repositories its cleanup drops, its task scheduling priority, and its log/telemetry
 * correlation key.
 *
 * 32-bit because `task_creator` packs it into the high bits of a 64-bit
 * `exec::queue_priority`; see `query_priority_bits()` for the packing contract.
 *
 * A distinct enum type rather than a `uint64_t` alias: operator ids, pipeline ids, connection
 * ids and per-connection query ordinals are all plain integers in this codebase, and letting a
 * query id be silently interchangeable with them is what allowed two independent query-id
 * counters to coexist unnoticed.
 */
enum class query_id_t : std::uint32_t {};

/// \brief The underlying integer, for formatting, hashing and bit packing.
[[nodiscard]] constexpr std::uint32_t value_of(query_id_t id) noexcept
{
  return static_cast<std::uint32_t>(id);
}

/// \brief Build a query id from a raw counter value.
[[nodiscard]] constexpr query_id_t make_query_id(std::uint32_t value) noexcept
{
  return static_cast<query_id_t>(value);
}

/**
 * @brief The task-scheduling priority contribution of a query: its id in the high 32 bits.
 *
 * The task priority queue pops the LOWEST value first, so packing the id above the
 * within-query pipeline rank makes every task of an earlier query dispatch before any task of
 * a later one, while the low 32 bits preserve pipeline order within a query.
 *
 * The 31-bit range is enforced explicitly. Exhaustion is an error rather than identity reuse
 * or FIFO inversion in a long-running runtime.
 */
inline constexpr std::uint32_t max_query_id = 0x7FFF'FFFFU;
[[nodiscard]] constexpr std::int64_t query_priority_bits(query_id_t id)
{
  if (value_of(id) > max_query_id) { throw std::overflow_error("Sirius query priority exhausted"); }
  return static_cast<std::int64_t>(value_of(id)) << 32;
}

inline query_id_t next_query_id(std::atomic<std::uint32_t>& counter)
{
  auto value = counter.load(std::memory_order_relaxed);
  for (;;) {
    if (value >= max_query_id) { throw std::overflow_error("Sirius query IDs exhausted"); }
    if (counter.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) {
      return make_query_id(value + 1);
    }
  }
}

}  // namespace sirius

/// Logging is std::format-based (see log/logging.hpp) and C++20 std::format has no built-in
/// support for enums, so query ids would otherwise have to be cast at every log site.
template <>
struct std::formatter<sirius::query_id_t> : std::formatter<std::uint32_t> {
  auto format(sirius::query_id_t id, std::format_context& ctx) const
  {
    return std::formatter<std::uint32_t>::format(sirius::value_of(id), ctx);
  }
};
