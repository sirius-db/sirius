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
 * @file dynamic_filter_source_policy.hpp
 * @brief Representation and gate policy for dynamic-filter publication
 *
 * The publisher (`dynamic_filter_publisher.hpp`) supplies runtime inputs and owns filter
 * construction, replication, and fan-out.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace sirius::op {

/**
 * @brief Membership representation chosen for one admitted build key
 */
enum class membership_filter_kind : std::uint8_t {
  none,           ///< No eligible membership representation
  small_in_list,  ///< Exact list scanned linearly; no hash build
  hash_in_list,   ///< Exact hash set, chosen when it fits the smallest probe GPU's L2 cache
  bloom           ///< Probabilistic fallback for sets too large to fit L2
};

/**
 * @brief Everything the representation choice depends on
 *
 * The small-list capability already incorporates its row-count limit.
 */
struct membership_choice_inputs {
  std::size_t build_rows               = 0;
  std::size_t l2_cache_bytes           = 0;  ///< 0 when unknown; hash IN-list is then ineligible
  std::size_t estimated_hash_set_bytes = 0;
  bool supports_small_in_list          = false;
  bool supports_hash_in_list           = false;
  bool supports_bloom                  = false;
};

/**
 * @brief Choose a key's membership representation
 *
 * @return The representation to construct, or `membership_filter_kind::none`
 */
[[nodiscard]] constexpr membership_filter_kind choose_membership_filter(
  membership_choice_inputs const& inputs) noexcept
{
  if (inputs.supports_small_in_list) { return membership_filter_kind::small_in_list; }
  if (inputs.supports_hash_in_list && inputs.l2_cache_bytes > 0 &&
      inputs.estimated_hash_set_bytes <= inputs.l2_cache_bytes) {
    return membership_filter_kind::hash_in_list;
  }
  if (inputs.supports_bloom) { return membership_filter_kind::bloom; }
  return membership_filter_kind::none;
}

/**
 * @brief Whether a build is too dense a sample of its key domain for a filter to repay itself
 *
 * The ratio is valid only for a key proven unique in its base relation; otherwise row count does
 * not equal distinct-key count.
 *
 * @param[in] build_rows Rows in the completed build
 * @param[in] domain_cardinality Unfiltered row upper bound of the base table the key traces to; 0
 * means untraceable and disables the gate
 * @param[in] build_key_proven_unique Whether the key is proven unique in its base relation; false
 * disables the gate
 * @param[in] threshold Coverage fraction at or above which the key is skipped; above 1.0 disables
 * the gate
 * @return True when the key should be skipped
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
 * @brief The zone-map analogue of @ref domain_coverage_gate_fires
 *
 * A `[min, max]` bound spanning most of the key domain prunes no row group. The gate requires a
 * value-range domain (a row-count domain cannot substitute), which the publisher does not yet
 * have, so this function is currently uncalled.
 *
 * @param[in] min_value Lowest build key value
 * @param[in] max_value Highest build key value
 * @param[in] domain_cardinality Unfiltered span of the key's domain; 0 disables the gate
 * @param[in] threshold Span fraction at or above which the zone map is skipped
 * @return True when the zone map should be skipped
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
