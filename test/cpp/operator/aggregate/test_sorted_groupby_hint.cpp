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

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "utils/data_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
using I64Traits = gpu_type_traits<int64_t>;
using sirius::test::vector_to_cudf_column;

/// Build a (key, value) INT64 partial-aggregate-shaped batch.
std::shared_ptr<data_batch> make_partial_batch(const std::vector<int64_t>& keys,
                                               const std::vector<int64_t>& values,
                                               cucascade::memory::memory_space& space)
{
  auto stream = default_stream();
  auto mr     = get_resource_ref(space);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<I64Traits>(keys, stream, mr));
  cols.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(cols));
  return sirius::make_data_batch(
    std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
}

}  // namespace

TEST_CASE("sorted-groupby hint matches the hash path on sorted and unsorted keys",
          "[sorted_groupby_hint]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);

  constexpr std::size_t num_rows = 40000;
  std::vector<int64_t> sorted_keys(num_rows);
  std::vector<int64_t> values(num_rows);
  for (std::size_t i = 0; i < num_rows; ++i) {
    sorted_keys[i] = static_cast<int64_t>(i / 4);  // non-decreasing runs of 4
    values[i]      = static_cast<int64_t>(i % 97);
  }
  auto shuffled_keys = sorted_keys;
  std::mt19937 rng(42);
  std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), rng);

  const std::vector<int> group_idx{0};
  const std::vector<cudf::aggregation::Kind> aggregates{cudf::aggregation::Kind::SUM};
  const std::vector<int> aggregate_idx{1};

  auto run = [&](const std::vector<int64_t>& keys, const sorted_hint_options& hint) {
    auto batch = make_partial_batch(keys, values, *space);
    auto ro    = batch->to_read_only();
    return gpu_aggregate_impl::local_grouped_aggregate(ro,
                                                       group_idx,
                                                       aggregates,
                                                       aggregate_idx,
                                                       {},
                                                       default_stream(),
                                                       *space,
                                                       sirius::telemetry::batch_telemetry_info{},
                                                       hint);
  };

  const sorted_hint_options hint_on{/*enabled=*/true, /*min_rows=*/1};
  const sorted_hint_options hint_off{};

  SECTION("sorted keys: hinted path equals hash path")
  {
    auto hinted = run(sorted_keys, hint_on);
    auto hashed = run(sorted_keys, hint_off);
    REQUIRE(sirius::test::expect_data_batches_equivalent(hinted, hashed, /*sort=*/true));
  }

  SECTION("unsorted keys: is_sorted gate must fall back to the hash path, results identical")
  {
    auto hinted = run(shuffled_keys, hint_on);
    auto hashed = run(shuffled_keys, hint_off);
    REQUIRE(sirius::test::expect_data_batches_equivalent(hinted, hashed, /*sort=*/true));
  }
}
