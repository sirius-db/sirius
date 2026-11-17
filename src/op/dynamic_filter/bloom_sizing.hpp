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
 * @file bloom_sizing.hpp
 * @brief Bloom filter sizing shared by `sirius_dynamic_bloom_filter`, built from a whole build, and
 * `detail::accumulated_bloom_geometry`, accumulated from a multi-partition build, so that retuning
 * the bits per key changes both.
 */

#pragma once

#include <cuda/cmath>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>

namespace sirius::op {

inline constexpr std::size_t bloom_bits_per_block      = 256;  ///< One cuco Bloom filter block
inline constexpr std::size_t bloom_bytes_per_block     = bloom_bits_per_block / 8;
inline constexpr std::size_t bloom_target_bits_per_key = 16;

static_assert(bloom_bits_per_block % bloom_target_bits_per_key == 0,
              "a block must hold a whole number of keys");

/**
 * @brief Number of Bloom blocks for @p num_keys keys: at least one, and never overflowing.
 */
[[nodiscard]] constexpr std::size_t bloom_blocks_for(std::size_t num_keys) noexcept
{
  return ::cuda::ceil_div(std::max<std::size_t>(num_keys, 1),
                          bloom_bits_per_block / bloom_target_bits_per_key);
}

/**
 * @brief The keys a Bloom is sized for when its build holds one part in @p scale of the keys:
 * @p rows times @p scale, at least @p min_keys, at most @p max_keys, rounded up to a power-of-two
 * number of blocks.
 *
 * Builds whose rows straddle a power of two get different sizes. Power-of-two block counts let a
 * merge fold the larger Blooms onto the smallest (sirius_dynamic_bloom_filter::merge_from).
 * @p max_keys is taken down to a power-of-two number of blocks, so the result never exceeds it.
 */
[[nodiscard]] constexpr std::size_t scaled_bloom_keys(std::size_t rows,
                                                      std::size_t scale,
                                                      std::size_t min_keys,
                                                      std::size_t max_keys) noexcept
{
  constexpr std::size_t keys_per_block = bloom_bits_per_block / bloom_target_bits_per_key;
  auto const max_blocks = std::bit_floor(std::max<std::size_t>(max_keys / keys_per_block, 1));
  auto const parts      = std::max<std::size_t>(scale, 1);
  auto const all_rows   = rows > std::numeric_limits<std::size_t>::max() / parts
                            ? std::numeric_limits<std::size_t>::max()
                            : rows * parts;
  auto const blocks     = bloom_blocks_for(std::max(all_rows, min_keys));
  return (blocks >= max_blocks ? max_blocks : std::bit_ceil(blocks)) * keys_per_block;
}

}  // namespace sirius::op
