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

#include "log/logging.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_source_policy.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"

#include <cstddef>
#include <cstdint>

namespace sirius::op::detail {

/// Publication path named in domain-gate diagnostics.
enum class publication_path : std::uint8_t { ordinary, accumulated, construction };

[[nodiscard]] constexpr char const* to_string(publication_path path) noexcept
{
  switch (path) {
    case publication_path::ordinary: return "ordinary";
    case publication_path::accumulated: return "accumulated";
    case publication_path::construction: return "construction";
  }
  return "invalid";
}

/**
 * @brief Counts a bound @p key of a build with @p build_rows rows in `keys_considered`,
 * `keys_with_known_domain` and `keys_build_exceeded_domain`, and logs the domain gate's decision.
 *
 * @param publishable False when the path has already rejected the key, so the gate never skips
 * it
 * @param threshold The plan's `dynamic_filter_domain_coverage_threshold`
 * @return Whether the domain coverage gate skips the key; the caller counts the skip where its own
 * checks order it
 */
inline bool consider_bound_key(dynamic_filter_stats_snapshot& counts,
                               dynamic_filter_publish_plan::admitted_key const& key,
                               std::size_t build_rows,
                               double threshold,
                               publication_path path,
                               bool publishable) noexcept
{
  ++counts.keys_considered;
  auto const& domain = key.build_key_domain;
  if (domain.cardinality != 0) {
    ++counts.keys_with_known_domain;
    if (build_rows > domain.cardinality) { ++counts.keys_build_exceeded_domain; }
  }
  bool const eligible = publishable && key.domain_gate_eligible();
  bool const gated =
    domain_coverage_gate_fires(build_rows, domain.cardinality, eligible, threshold);
  log::log_noexcept([&] {
    SIRIUS_LOG_DEBUG(
      "[dynamic_filter_domain] path={} condition={} rows={} domain={} provenance={} eligible={} "
      "threshold={} decision={}",
      to_string(path),
      key.planner_condition_index,
      build_rows,
      domain.cardinality,
      to_string(domain.source),
      static_cast<int>(eligible),
      threshold,
      gated ? "skip" : "keep");
  });
  return gated;
}

}  // namespace sirius::op::detail
