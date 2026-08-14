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

#include <cstdint>
#include <format>

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
 * Packing the id above the within-query pipeline rank gives every query a contiguous,
 * NON-OVERLAPPING band of priorities: the low 32 bits preserve pipeline order within a query,
 * and the queues' per-query indexes rely on the banding to group one query's levels together.
 * Cross-query dispatch order is NOT the raw value order — the queues' fair pops rotate
 * round-robin across query bands (see multi_index_priority_queue), because popping strictly
 * lowest-first would let every task of an earlier query outrank every task of a later one and
 * starve it (issue F1 in the concurrency register).
 *
 * Masked to 31 bits because the priority is a SIGNED 64-bit value: an id with bit 31 set would
 * shift into the sign bit and break the banding. Band VALUES therefore wrap every 2^31 queries
 * in a single process; since the fair pops rotate over live queries by id rather than trusting
 * the value order, a wrapped id only shifts the rotation's starting point.
 */
[[nodiscard]] constexpr std::int64_t query_priority_bits(query_id_t id) noexcept
{
  return static_cast<std::int64_t>(value_of(id) & 0x7FFF'FFFFU) << 32;
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
