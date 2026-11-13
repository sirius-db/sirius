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
 * @file
 * @brief Hash-join partition sizing settings that `sirius::operator_params` and the sizing rules
 * share without depending on each other.
 */

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

namespace sirius::config {

/// Approximate cuco hash-table bytes per build row: 8-byte hash/index slots at 0.5 occupancy.
/// Capacity rounding and auxiliary allocations are excluded; this is an admission estimate.
inline constexpr std::uint64_t BUILD_PROBE_HASH_TABLE_BYTES_PER_ROW = 16;

}  // namespace sirius::config

namespace sirius::op {

/**
 * @brief Selects the rule `decide_build_fold` applies.
 */
enum class build_fold_policy : std::uint8_t {
  /**
   * @brief Fold when the join publishes dynamic filters and the build fits the resident limit and
   * the cuDF row limit: the rule that predates `decide_build_fold`.
   */
  legacy_filter_coupled,
  /** @brief Never fold; only builds that are `concat_all` from planning stay whole. */
  never,
  /**
   * @brief Fold when the folded build is known to fit cuDF and the resident limit, its estimated
   * hash table is no larger than the build, and folding changes the CONCAT's grouping.
   */
  bounded,
};

/**
 * @brief Every `build_fold_policy` with its spelling in the `build_fold_policy` setting.
 */
inline constexpr std::array build_fold_policy_names{
  std::pair{build_fold_policy::legacy_filter_coupled, std::string_view{"legacy_filter_coupled"}},
  std::pair{build_fold_policy::never, std::string_view{"never"}},
  std::pair{build_fold_policy::bounded, std::string_view{"bounded"}},
};

/**
 * @brief Spelling of `policy` in the `build_fold_policy` setting.
 */
[[nodiscard]] constexpr std::string_view to_string(build_fold_policy policy) noexcept
{
  for (auto const& [named, spelling] : build_fold_policy_names) {
    if (named == policy) { return spelling; }
  }
  return {};
}

/**
 * @brief Parses a `build_fold_policy` setting value.
 *
 * @param spelling Policy spelling, matched exactly
 * @return The named policy, or `std::nullopt` when `spelling` names none
 */
[[nodiscard]] constexpr std::optional<build_fold_policy> parse_build_fold_policy(
  std::string_view spelling) noexcept
{
  for (auto const& [policy, named] : build_fold_policy_names) {
    if (named == spelling) { return policy; }
  }
  return std::nullopt;
}

}  // namespace sirius::op
