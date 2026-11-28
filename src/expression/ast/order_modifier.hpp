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

// standard library
#include <cstdint>

namespace sirius::ast {

enum class sort_direction : std::uint8_t { ascending, descending };
enum class null_placement : std::uint8_t { first, last };

/**
 * @brief Direction and NULL placement of one ORDER BY key
 *
 * `nulls` is an absolute position: `first` puts NULL keys before every value in both directions.
 */
struct order_modifier {
  sort_direction direction{sort_direction::ascending};  ///< Direction of the key's values
  null_placement nulls{null_placement::last};           ///< Where NULL keys sort
  bool operator==(order_modifier const&) const = default;
};

}  // namespace sirius::ast
