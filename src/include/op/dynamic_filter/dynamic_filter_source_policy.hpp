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
#include <cstdint>

namespace sirius::op {

enum class membership_filter_kind : std::uint8_t { none, small_in_list, hash_in_list, bloom };

struct membership_choice_inputs {
  std::size_t build_rows               = 0;
  std::size_t l2_cache_bytes           = 0;  // 0 means unknown.
  std::size_t estimated_hash_set_bytes = 0;
  double inlist_max_l2_fraction        = 0.0;
  bool supports_small_in_list          = false;
  bool supports_hash_in_list           = false;
  bool supports_bloom                  = false;
};

/**
 * @brief Chooses a membership representation
 *
 * Small lists win. A hash IN-list requires known L2 and an estimate within both L2 and the
 * configured fraction. Without Bloom support, fitting L2 is enough.
 */
[[nodiscard]] constexpr membership_filter_kind choose_membership_filter(
  membership_choice_inputs const& inputs) noexcept
{
  if (inputs.supports_small_in_list) { return membership_filter_kind::small_in_list; }
  bool const hash_set_fits_l2 = inputs.supports_hash_in_list && inputs.l2_cache_bytes > 0 &&
                                inputs.estimated_hash_set_bytes <= inputs.l2_cache_bytes;
  if (hash_set_fits_l2 &&
      (!inputs.supports_bloom ||
       static_cast<double>(inputs.estimated_hash_set_bytes) <=
         inputs.inlist_max_l2_fraction * static_cast<double>(inputs.l2_cache_bytes))) {
    return membership_filter_kind::hash_in_list;
  }
  if (inputs.supports_bloom) { return membership_filter_kind::bloom; }
  return membership_filter_kind::none;
}

/**
 * @brief Tests whether a proven-unique build covers the configured fraction of a known domain
 *
 * Unknown domains, non-unique keys, and thresholds above 1 disable the gate.
 */
[[nodiscard]] constexpr bool domain_coverage_gate_fires(std::size_t build_rows,
                                                        std::size_t domain_cardinality,
                                                        bool build_key_proven_unique,
                                                        double threshold) noexcept
{
  if (threshold > 1.0) { return false; }
  if (!build_key_proven_unique) { return false; }
  if (domain_cardinality == 0) { return false; }
  return static_cast<double>(build_rows) / static_cast<double>(domain_cardinality) >= threshold;
}

/**
 * @brief Tests whether inclusive `[min_value, max_value]` covers the threshold of a known domain
 *
 * Not yet wired into production; exercised by tests only.
 */
[[nodiscard]] constexpr bool zone_map_range_gate_fires(double min_value,
                                                       double max_value,
                                                       std::size_t domain_cardinality,
                                                       double threshold) noexcept
{
  if (domain_cardinality == 0) { return false; }
  return ((max_value - min_value + 1.0) / static_cast<double>(domain_cardinality)) >= threshold;
}

}  // namespace sirius::op
