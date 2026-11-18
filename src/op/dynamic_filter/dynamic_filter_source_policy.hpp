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

/// Origin of the row-count evidence used only to decide dynamic-filter usefulness.
enum class dynamic_filter_domain_source : std::uint8_t {
  unknown,
  catalog_bound,
  pinned_parquet_exact
};

/// Spelling used by dynamic-filter diagnostics.
[[nodiscard]] constexpr char const* to_string(dynamic_filter_domain_source source) noexcept
{
  switch (source) {
    case dynamic_filter_domain_source::unknown: return "unknown";
    case dynamic_filter_domain_source::catalog_bound: return "catalog_bound";
    case dynamic_filter_domain_source::pinned_parquet_exact: return "pinned_parquet_exact";
  }
  return "invalid";
}

/// Row-domain evidence for one build key, copied so plans retain no registry entry. It decides
/// dynamic-filter usefulness only.
struct dynamic_filter_domain_evidence {
  /// Base-table row count, exact for pins and an upper bound from the catalog; 0 means unknown.
  std::size_t cardinality             = 0;
  dynamic_filter_domain_source source = dynamic_filter_domain_source::unknown;
  /// A pin's `unique_cols` declaration; never a correctness-bearing uniqueness proof.
  bool declared_unique = false;

  /// Whether a uniqueness declaration is backed by an exact pinned row count
  [[nodiscard]] constexpr bool has_backed_declaration() const noexcept
  {
    return declared_unique && source == dynamic_filter_domain_source::pinned_parquet_exact;
  }

  [[nodiscard]] bool operator==(dynamic_filter_domain_evidence const&) const = default;
};

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
 * @brief Tests whether an eligible build covers the configured fraction of a known domain
 *
 * Unknown domains, empty builds, build counts larger than the domain, keys without uniqueness
 * eligibility, and thresholds above 1 disable the gate.
 */
[[nodiscard]] constexpr bool domain_coverage_gate_fires(std::size_t build_rows,
                                                        std::size_t domain_cardinality,
                                                        bool build_key_eligible,
                                                        double threshold) noexcept
{
  if (threshold > 1.0) { return false; }
  if (!build_key_eligible) { return false; }
  if (build_rows == 0 || domain_cardinality == 0 || build_rows > domain_cardinality) {
    return false;
  }
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
