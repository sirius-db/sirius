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
 * @brief Tests the device-independent dynamic-filter publication policy.
 */

#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"

#include <catch.hpp>

#include <cstddef>

namespace {

using sirius::op::choose_membership_filter;
using sirius::op::domain_coverage_gate_fires;
using sirius::op::membership_filter_kind;
using sirius::op::zone_map_range_gate_fires;

constexpr std::size_t kL2Bytes = 1024;

// Exactly representable, so the inclusive gate-boundary tests are deterministic.
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
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = true,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::small_in_list);

  // The preference is unconditional: a hash set too large for L2 does not demote it.
  REQUIRE(choose_membership_filter({.build_rows               = 3,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = true,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::small_in_list);

  // The residency fraction governs only the hash-vs-Bloom trade: even at 0 the small IN-list
  // still wins first.
  REQUIRE(choose_membership_filter({.build_rows               = 3,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 64,
                                    .inlist_max_l2_fraction   = 0.0,
                                    .supports_small_in_list   = true,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::small_in_list);
}

TEST_CASE("membership policy keeps the hash IN-list exactly while its set fits L2 at fraction 1.0",
          "[dynamic_filter][source_policy]")
{
  // A residency fraction of 1.0 reproduces the legacy inclusive L2-fit rule.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes - 1,
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::hash_in_list);

  // A set of exactly the cache size still fits: the comparison is inclusive.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes,
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::hash_in_list);

  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);
}

TEST_CASE("membership policy treats an unknown L2 size as no hash IN-list",
          "[dynamic_filter][source_policy]")
{
  // An unknown cache size is reported as 0 and must be rejected outright, not read as a fit.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = 0,
                                    .estimated_hash_set_bytes = 0,
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);

  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = 0,
                                    .estimated_hash_set_bytes = 0,
                                    .inlist_max_l2_fraction   = 1.0,
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
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = false,
                                    .supports_bloom = false}) == membership_filter_kind::none);

  // The one supported representation is ruled out on size and nothing else can stand in.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .inlist_max_l2_fraction   = 1.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = false}) == membership_filter_kind::none);
}

TEST_CASE("membership policy demotes the hash IN-list to the Bloom above the residency fraction",
          "[dynamic_filter][source_policy]")
{
  // A vanishing fraction makes the residency threshold smaller than any real set estimate, so
  // even a tiny L2-fitting set demotes to the Bloom.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 64,
                                    .inlist_max_l2_fraction   = 1e-12,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);

  // 0 x L2 = 0 and a non-empty build's estimate is positive, so no special case is needed.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 64,
                                    .inlist_max_l2_fraction   = 0.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);
}

TEST_CASE("membership policy's residency-fraction boundary is inclusive",
          "[dynamic_filter][source_policy]")
{
  // 0.5 x 1024 = 512.0 is exactly representable, so the inclusive comparison is deterministic.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 512,
                                    .inlist_max_l2_fraction   = 0.5,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = true}) ==
          membership_filter_kind::hash_in_list);

  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = 513,
                                    .inlist_max_l2_fraction   = 0.5,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom = true}) == membership_filter_kind::bloom);
}

TEST_CASE("membership policy keeps a fitting IN-list at any fraction without a Bloom fallback",
          "[dynamic_filter][source_policy]")
{
  // With no Bloom to demote to, exactness is the only membership option; the fraction only
  // arbitrates the hash-vs-Bloom trade, so the plain inclusive L2 fit decides alone.
  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes,
                                    .inlist_max_l2_fraction   = 0.0,
                                    .supports_small_in_list   = false,
                                    .supports_hash_in_list    = true,
                                    .supports_bloom           = false}) ==
          membership_filter_kind::hash_in_list);

  REQUIRE(choose_membership_filter({.build_rows               = 4096,
                                    .l2_cache_bytes           = kL2Bytes,
                                    .estimated_hash_set_bytes = kL2Bytes + 1,
                                    .inlist_max_l2_fraction   = 0.0,
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

TEST_CASE("domain-coverage gate fires only for proven-unique keys",
          "[dynamic_filter][source_policy]")
{
  // Row retention does not measure domain coverage for duplicate keys, so they never fire the
  // gate even at or above the table cardinality.
  REQUIRE_FALSE(domain_coverage_gate_fires(900'000, 1'000'000, false, 0.9));
  REQUIRE_FALSE(domain_coverage_gate_fires(1'000'000, 1'000'000, false, 0.9));
  REQUIRE_FALSE(domain_coverage_gate_fires(2'000'000, 1'000'000, false, 0.9));
  REQUIRE(domain_coverage_gate_fires(900'000, 1'000'000, true, 0.9));
  // Below threshold nothing fires either way.
  REQUIRE_FALSE(domain_coverage_gate_fires(100, 1'000'000, true, 0.9));
}

TEST_CASE("domain-coverage gate treats a threshold above 1.0 as disabled outright",
          "[dynamic_filter][source_policy]")
{
  REQUIRE_FALSE(domain_coverage_gate_fires(2'000'000, 1'000'000, true, 2.0));
  REQUIRE_FALSE(domain_coverage_gate_fires(2'000'000, 1'000'000, false, 2.0));
  REQUIRE_FALSE(domain_coverage_gate_fires(1'000'000, 1'000'000, true, 1.5));
  // Exactly 1.0 remains active and fires at full coverage of a proven-unique key.
  REQUIRE(domain_coverage_gate_fires(100, 100, true, 1.0));
  REQUIRE_FALSE(domain_coverage_gate_fires(99, 100, true, 1.0));
}

TEST_CASE("zone-map range gate fires at and above its threshold", "[dynamic_filter][source_policy]")
{
  // The span is inclusive of both bounds: [10, 59] covers 50 of the 100 domain values.
  REQUIRE(zone_map_range_gate_fires(10.0, 59.0, 100, kThreshold));
  REQUIRE(zone_map_range_gate_fires(10.0, 60.0, 100, kThreshold));
  REQUIRE_FALSE(zone_map_range_gate_fires(10.0, 58.0, 100, kThreshold));
}

TEST_CASE("zone-map range gate is disabled for an untraceable key domain",
          "[dynamic_filter][source_policy]")
{
  REQUIRE_FALSE(zone_map_range_gate_fires(0.0, 1'000'000.0, 0, kThreshold));
}
