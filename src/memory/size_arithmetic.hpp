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

#include <cstddef>
#include <limits>

namespace sirius::memory {

/** @brief Add byte counts, clamping overflow to the largest representable size. */
[[nodiscard]] constexpr std::size_t saturating_add(std::size_t lhs, std::size_t rhs) noexcept
{
  constexpr auto max_size = std::numeric_limits<std::size_t>::max();
  return rhs > max_size - lhs ? max_size : lhs + rhs;
}

/** @brief Multiply byte counts, clamping overflow to the largest representable size. */
[[nodiscard]] constexpr std::size_t saturating_mul(std::size_t lhs, std::size_t rhs) noexcept
{
  constexpr auto max_size = std::numeric_limits<std::size_t>::max();
  return lhs != 0 && rhs > max_size / lhs ? max_size : lhs * rhs;
}

}  // namespace sirius::memory
