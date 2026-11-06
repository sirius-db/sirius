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
#include <cstddef>

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

}  // namespace sirius::op
