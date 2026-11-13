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

// Host-only tests of decide_build_fold, the rule that decides whether a single-partition STANDARD
// or MIXED_JOIN build CONCAT folds into one batch.

#include "op/partition/build_fold.hpp"
#include "op/partition/config.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "sirius_config.hpp"

#include <catch.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>

using sirius::op::build_fold_inputs;
using sirius::op::build_fold_limits;
using sirius::op::build_fold_policy;
using sirius::op::decide_build_fold;
using sirius::op::HASH_JOIN_MODE;

namespace {

constexpr std::uint64_t bytes_per_row = sirius::config::BUILD_PROBE_HASH_TABLE_BYTES_PER_ROW;
constexpr std::uint64_t row_limit =
  static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
constexpr std::uint64_t mib = 1024ULL * 1024;

constexpr build_fold_limits roomy_limits{.max_resident_bytes = 1024 * mib};

/// A 1000-row build at exactly the hash-table width per row, above a 1-byte CONCAT target.
constexpr build_fold_inputs foldable{.rows               = 1000,
                                     .resident_bytes     = 1000 * bytes_per_row,
                                     .concat_batch_bytes = 1,
                                     .num_gpus           = 1,
                                     .mode               = HASH_JOIN_MODE::STANDARD};

bool bounded(build_fold_inputs const& inputs, build_fold_limits limits = roomy_limits)
{
  bool const folds = decide_build_fold(inputs, limits, build_fold_policy::bounded, false);
  // The bounded rule never reads dynamic-filter state.
  REQUIRE(decide_build_fold(inputs, limits, build_fold_policy::bounded, true) == folds);
  return folds;
}

/// The predicate `bind_partition_inputs` applied before `decide_build_fold`, restated verbatim.
bool predicate_before_rule(build_fold_inputs const& inputs,
                           build_fold_limits limits,
                           bool publishes_filters)
{
  return publishes_filters && inputs.resident_bytes < limits.max_resident_bytes && inputs.rows &&
         *inputs.rows <= row_limit;
}

}  // namespace

TEST_CASE("decide_build_fold never folds under the never policy", "[partition][build_fold]")
{
  auto const publishes_filters = GENERATE(false, true);
  CHECK_FALSE(
    decide_build_fold(foldable, roomy_limits, build_fold_policy::never, publishes_filters));
  build_fold_inputs wide = foldable;
  wide.resident_bytes    = 512 * mib;
  CHECK_FALSE(decide_build_fold(wide, roomy_limits, build_fold_policy::never, publishes_filters));
}

TEST_CASE("decide_build_fold legacy policy equals the filter-coupled predicate",
          "[partition][build_fold]")
{
  constexpr std::uint64_t cap = 64 * mib;
  constexpr std::array<std::optional<std::uint64_t>, 6> rows{
    std::nullopt, 0, 1, 1000, row_limit, row_limit + 1};
  constexpr std::array<std::uint64_t, 5> resident{0, 1, cap - 1, cap, cap + 1};
  constexpr std::array<std::uint64_t, 3> concat{
    0, cap - 1, std::numeric_limits<std::uint64_t>::max()};
  for (auto const publishes_filters : {false, true}) {
    for (auto const mode : {HASH_JOIN_MODE::STANDARD, HASH_JOIN_MODE::MIXED_JOIN}) {
      for (auto const r : rows) {
        for (auto const bytes : resident) {
          for (auto const target : concat) {
            build_fold_inputs const inputs{r, bytes, target, 2, mode};
            build_fold_limits const limits{cap};
            INFO("rows=" << r.value_or(0) << " known=" << r.has_value() << " resident=" << bytes
                         << " concat=" << target << " filters=" << publishes_filters);
            CHECK(decide_build_fold(
                    inputs, limits, build_fold_policy::legacy_filter_coupled, publishes_filters) ==
                  predicate_before_rule(inputs, limits, publishes_filters));
          }
        }
      }
    }
  }
}

TEST_CASE("decide_build_fold bounded folds a build that fits every limit",
          "[partition][build_fold]")
{
  CHECK(bounded(foldable));
  build_fold_inputs mixed = foldable;
  mixed.mode              = HASH_JOIN_MODE::MIXED_JOIN;
  mixed.num_gpus          = 8;
  CHECK(bounded(mixed));
}

TEST_CASE("decide_build_fold bounded requires a known row count within the cuDF limit",
          "[partition][build_fold]")
{
  // Wide enough that only the row count can decline: the table bound and the resident limit hold.
  constexpr build_fold_limits limits{.max_resident_bytes =
                                       std::numeric_limits<std::uint64_t>::max()};
  build_fold_inputs inputs = foldable;
  inputs.resident_bytes    = (row_limit + 1) * bytes_per_row;

  inputs.rows = row_limit;
  CHECK(bounded(inputs, limits));
  inputs.rows = row_limit + 1;
  CHECK_FALSE(bounded(inputs, limits));
  inputs.rows = std::nullopt;
  CHECK_FALSE(bounded(inputs, limits));
}

TEST_CASE("decide_build_fold bounded keeps the folded build under the resident limit",
          "[partition][build_fold]")
{
  build_fold_limits limits  = roomy_limits;
  limits.max_resident_bytes = foldable.resident_bytes + 1;
  CHECK(bounded(foldable, limits));
  limits.max_resident_bytes = foldable.resident_bytes;
  CHECK_FALSE(bounded(foldable, limits));
}

TEST_CASE("decide_build_fold bounded keeps the estimated table within the folded build",
          "[partition][build_fold]")
{
  SECTION("the table may equal the build but not exceed it")
  {
    build_fold_inputs inputs = foldable;
    inputs.resident_bytes    = *foldable.rows * bytes_per_row;
    CHECK(bounded(inputs));
    inputs.resident_bytes -= 1;
    CHECK_FALSE(bounded(inputs));
  }
  SECTION("a build narrower than the per-row table estimate never folds")
  {
    build_fold_inputs inputs = foldable;
    inputs.resident_bytes    = *foldable.rows * 8;  // One INT64 key column.
    CHECK_FALSE(bounded(inputs));
  }
  SECTION("a build far wider than the table estimate folds")
  {
    build_fold_inputs inputs = foldable;
    inputs.resident_bytes    = 512 * mib;
    CHECK(bounded(inputs));
  }
}

TEST_CASE("decide_build_fold bounded folds only builds that exceed the CONCAT target",
          "[partition][build_fold]")
{
  build_fold_inputs inputs  = foldable;
  inputs.concat_batch_bytes = foldable.resident_bytes - 1;
  CHECK(bounded(inputs));
  inputs.concat_batch_bytes = foldable.resident_bytes;
  CHECK_FALSE(bounded(inputs));
  inputs.concat_batch_bytes = 0;
  CHECK(bounded(inputs));
}

TEST_CASE("build_fold_policy names round-trip", "[partition][build_fold]")
{
  static_assert(sirius::op::build_fold_policy_names.size() == 3);
  using sirius::op::parse_build_fold_policy;
  for (auto const policy : {build_fold_policy::legacy_filter_coupled,
                            build_fold_policy::never,
                            build_fold_policy::bounded}) {
    CHECK(parse_build_fold_policy(sirius::op::to_string(policy)) == policy);
  }
  CHECK(sirius::op::to_string(build_fold_policy::legacy_filter_coupled) == "legacy_filter_coupled");
  CHECK(sirius::op::to_string(build_fold_policy::never) == "never");
  CHECK(sirius::op::to_string(build_fold_policy::bounded) == "bounded");
  CHECK_FALSE(parse_build_fold_policy("").has_value());
  CHECK_FALSE(parse_build_fold_policy("Bounded").has_value());
  CHECK_FALSE(parse_build_fold_policy("legacy").has_value());
}

TEST_CASE("operator defaults select the legacy build fold rule", "[partition][build_fold]")
{
  CHECK(sirius::operator_params{}.build_fold_policy == build_fold_policy::legacy_filter_coupled);
  CHECK(sirius::operator_params::with_batch_size(mib).build_fold_policy ==
        build_fold_policy::legacy_filter_coupled);
}
