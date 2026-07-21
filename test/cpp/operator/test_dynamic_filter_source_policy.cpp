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
 * @file test_dynamic_filter_source_policy.cpp
 * @brief Tests for the decisions `publish_dynamic_filters` makes before it touches a device:
 * `choose_membership_filter`, `domain_coverage_gate_fires`, and `zone_map_range_gate_fires` from
 * `op/dynamic_filter/dynamic_filter_source_policy.hpp`.
 *
 * These functions read only counts, sizes, and capability answers, so every case here is stated as
 * a value and needs no GPU, no memory manager, and no fixture. Each test names the decision
 * boundary it pins; the publisher tests in `test_dynamic_filter_publisher.cpp` cover the
 * surrounding mechanism.
 */

#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"

#include <catch.hpp>

#include <cstddef>

namespace {

using sirius::op::choose_membership_filter;
using sirius::op::domain_coverage_gate_fires;
using sirius::op::membership_filter_kind;
using sirius::op::zone_map_range_gate_fires;

/// Cache budget every representation case is stated against, so that a fitting and an overflowing
/// hash set differ by one byte.
constexpr std::size_t kL2Bytes = 1024;

/// Threshold both gates are stated against, chosen exactly representable so the at-threshold cases
/// distinguish `>=` from `>` without rounding.
constexpr double kThreshold = 0.5;

}  // namespace

TEST_CASE("membership policy prefers the small IN-list whenever the key type supports it",
          "[dynamic_filter][source_policy]")
{
  // Every competing representation is available too, so a preference-order regression surfaces
  // here rather than only in a size-boundary case.
  REQUIRE(choose_membership_filter({.build_rows               = 3,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 64,
                                    .supports_small_in_list   = true,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::small_in_list);

  // The preference is unconditional: a hash set too large for L2 does not demote it.
  REQUIRE(choose_membership_filter({.build_rows               = 3,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .supports_small_in_list   = true,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::small_in_list);
}

TEST_CASE("membership policy chooses the hash IN-list exactly while its set fits L2",
          "[dynamic_filter][source_policy]")
{
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes - 1,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::hash_in_list);

  // A set of exactly the cache size still fits: the comparison is inclusive.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::hash_in_list);

  // One byte over, and the exact set gives way to the probabilistic fallback.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);
}

TEST_CASE("membership policy treats an unknown L2 size as no hash IN-list",
          "[dynamic_filter][source_policy]")
{
  // An unknown cache size is reported as 0, which a size comparison alone would read as "every set
  // of 0 estimated bytes fits". The eligibility test must reject it outright.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = 0,
                                    .estimated_hash_set_bytes = 0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);

  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = 0,
                                    .estimated_hash_set_bytes = 0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = false}) == membership_filter_kind::none);
}

TEST_CASE("membership policy chooses nothing when no representation is available",
          "[dynamic_filter][source_policy]")
{
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 64,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = false,
                                    .supports_bloom = false}) == membership_filter_kind::none);

  // The one supported representation is ruled out on size and nothing else can stand in.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = false}) == membership_filter_kind::none);
}

TEST_CASE("domain-coverage gate fires at and above its threshold",
          "[dynamic_filter][source_policy]")
{
  REQUIRE(domain_coverage_gate_fires(50, 100, /*build_key_proven_unique=*/true, kThreshold));
  REQUIRE(domain_coverage_gate_fires(51, 100, /*build_key_proven_unique=*/true, kThreshold));
  REQUIRE_FALSE(domain_coverage_gate_fires(49, 100, /*build_key_proven_unique=*/true, kThreshold));
}

TEST_CASE("domain-coverage gate is disabled for an untraceable key domain",
          "[dynamic_filter][source_policy]")
{
  // A key whose base table could not be traced reports cardinality 0. The gate must stand down
  // rather than divide by it -- even for a proven-unique key.
  REQUIRE_FALSE(
    domain_coverage_gate_fires(1'000'000, 0, /*build_key_proven_unique=*/true, kThreshold));
}

TEST_CASE("zone-map range gate fires at and above its threshold", "[dynamic_filter][source_policy]")
{
  // The span is inclusive of both bounds: [10, 59] covers 50 of the 100 domain values, so the
  // at-threshold case also pins that the endpoint is counted.
  REQUIRE(zone_map_range_gate_fires(10.0, 59.0, 100, kThreshold));
  REQUIRE(zone_map_range_gate_fires(10.0, 60.0, 100, kThreshold));
  REQUIRE_FALSE(zone_map_range_gate_fires(10.0, 58.0, 100, kThreshold));
}

TEST_CASE("zone-map range gate is disabled for an untraceable key domain",
          "[dynamic_filter][source_policy]")
{
  REQUIRE_FALSE(zone_map_range_gate_fires(0.0, 1'000'000.0, 0, kThreshold));
}
