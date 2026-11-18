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

#include "sirius/exception.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace sirius::op::merge {

/**
 * @brief Pack a MERGE_SORT output position: `(partition << 32) | chunk`.
 *
 * A windowed merge emits several output chunks per partition; this orders them first by
 * partition, then by chunk within the partition.
 *
 * @throws sirius::internal_exception if @p partition or @p chunk is >= 2^32.
 */
[[nodiscard]] inline std::size_t make_merge_chunk_position(std::size_t partition, std::size_t chunk)
{
  constexpr std::uint64_t limit = std::uint64_t{1} << 32;
  if (partition >= limit || chunk >= limit) {
    throw sirius::internal_exception(
      "merge chunk position out of range: partition={}, chunk={} (both must be < 2^32)",
      partition,
      chunk);
  }
  return static_cast<std::size_t>((static_cast<std::uint64_t>(partition) << 32) |
                                  static_cast<std::uint64_t>(chunk));
}

/**
 * @brief Inverse of make_merge_chunk_position: returns `{partition, chunk}`.
 */
[[nodiscard]] inline std::pair<std::size_t, std::size_t> split_merge_chunk_position(
  std::size_t position) noexcept
{
  const auto value = static_cast<std::uint64_t>(position);
  return {static_cast<std::size_t>(value >> 32),
          static_cast<std::size_t>(value & std::uint64_t{0xFFFFFFFF})};
}

}  // namespace sirius::op::merge
