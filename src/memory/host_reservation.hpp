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

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>

namespace sirius::memory {

/**
 * @brief Round a HOST staging size to the memory space's reservation granularity.
 *
 * @param space Candidate memory space
 * @param bytes Staging payload size in bytes
 * @return Block-rounded size, or std::nullopt if the space cannot support the request
 */
[[nodiscard]] inline std::optional<std::size_t> host_reservation_size(
  const cucascade::memory::memory_space& space, std::size_t bytes)
{
  const auto* resource = space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (!resource) { return std::nullopt; }
  const auto block  = resource->get_block_size();
  const auto blocks = bytes / block + static_cast<std::size_t>(bytes % block != 0);
  if (blocks > space.get_max_memory() / block) { return std::nullopt; }
  return blocks * block;
}

/**
 * @brief Estimate how much capacity is available below the HOST reservation limit.
 *
 * @param space HOST memory space backed by a fixed-size HOST resource
 * @return The reservation limit minus currently allocated bytes, clamped to zero
 */
[[nodiscard]] inline std::size_t host_reservation_headroom(
  const cucascade::memory::memory_space& space)
{
  const auto* resource = space.get_memory_resource_of<cucascade::memory::Tier::HOST>();
  const auto used      = resource->get_total_allocated_bytes();
  const auto limit     = space.get_max_memory();
  return used < limit ? limit - used : 0;
}

/**
 * @brief Try to reserve enough HOST blocks for a staging payload.
 *
 * @param space Candidate memory space whose reservation accounting may change
 * @param bytes Staging payload size in bytes
 * @return The reservation, or nullptr if the space cannot admit it
 */
[[nodiscard]] inline std::unique_ptr<cucascade::memory::reservation> try_reserve_host(
  cucascade::memory::memory_space& space, std::size_t bytes)
{
  if (auto rounded = host_reservation_size(space, bytes)) {
    return space.make_reservation_or_null(*rounded);
  }
  return nullptr;
}

/**
 * @brief Try to reserve HOST staging capacity from the first pool that can admit the request.
 *
 * @param spaces Candidate memory spaces in the order they should be tried
 * @param bytes Staging payload size in bytes
 * @return The reservation, or nullptr if no space can admit it
 */
[[nodiscard]] inline std::unique_ptr<cucascade::memory::reservation> try_reserve_host(
  std::span<cucascade::memory::memory_space* const> spaces, std::size_t bytes)
{
  for (auto* space : spaces) {
    if (auto reservation = try_reserve_host(*space, bytes)) { return reservation; }
  }
  return nullptr;
}

/**
 * @brief A pre-publication miss carrying the copy size needed to resume.
 */
class host_reservation_required : public std::runtime_error {
 public:
  /**
   * @brief Construct a HOST reservation miss.
   *
   * @param bytes Copy size needed to resume, before rounding for a candidate HOST pool
   */
  explicit host_reservation_required(std::size_t bytes)
    : std::runtime_error("Insufficient HOST memory to stage materialized query results"),
      bytes(bytes)
  {
  }

  const std::size_t bytes;
};

}  // namespace sirius::memory
