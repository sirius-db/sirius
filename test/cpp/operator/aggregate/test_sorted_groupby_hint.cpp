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
#include "utils/log_test_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/null_mask.hpp>
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

std::shared_ptr<data_batch> make_partial_batch(const std::vector<int64_t>& keys,
                                               const std::vector<int64_t>& values,
                                               cucascade::memory::memory_space& space,
                                               rmm::cuda_stream_view stream)
{
  auto mr = get_resource_ref(space);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(vector_to_cudf_column<I64Traits>(keys, stream, mr));
  cols.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
  auto table = std::make_unique<cudf::table>(std::move(cols));
  return sirius::make_data_batch(
    std::move(table), space, stream, sirius::telemetry::batch_telemetry_info{});
}

std::unique_ptr<cudf::column> make_nullable_i64_column(const std::vector<int64_t>& values,
                                                       const std::vector<bool>& valid,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto column = vector_to_cudf_column<I64Traits>(values, stream, mr);
  auto mask   = cudf::create_null_mask(
    static_cast<cudf::size_type>(values.size()), cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_data            = static_cast<cudf::bitmask_type*>(mask.data());
  cudf::size_type null_count = 0;
  for (cudf::size_type index = 0; index < static_cast<cudf::size_type>(valid.size()); ++index) {
    if (!valid[static_cast<std::size_t>(index)]) {
      cudf::set_null_mask(mask_data, index, index + 1, false, stream);
      ++null_count;
    }
  }
  column->set_null_mask(std::move(mask), null_count);
  return column;
}

std::shared_ptr<data_batch> make_nullable_multikey_batch(const std::vector<int64_t>& first_keys,
                                                         const std::vector<bool>& first_valid,
                                                         const std::vector<int64_t>& second_keys,
                                                         const std::vector<bool>& second_valid,
                                                         const std::vector<int64_t>& values,
                                                         cucascade::memory::memory_space& space,
                                                         rmm::cuda_stream_view stream)
{
  auto mr = get_resource_ref(space);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_nullable_i64_column(first_keys, first_valid, stream, mr));
  columns.push_back(make_nullable_i64_column(second_keys, second_valid, stream, mr));
  columns.push_back(vector_to_cudf_column<I64Traits>(values, stream, mr));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                 space,
                                 stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

std::size_t sorted_hint_engagement_count(
  const std::vector<sirius::test::recording_log_sink::record>& records)
{
  std::size_t count = 0;
  for (auto const& record : records) {
    if (record.message.find("local_grouped_agg: sorted-groupby hint engaged") !=
        std::string::npos) {
      CHECK(record.level == sirius::log::level::debug);
      ++count;
    }
  }
  return count;
}

}  // namespace

TEST_CASE("sorted-groupby hint matches the hash path on sorted and unsorted keys",
          "[sorted_groupby_hint]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = space->acquire_stream();
  REQUIRE(stream.value() != default_stream().value());

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
    auto batch = make_partial_batch(keys, values, *space, stream);
    auto ro    = batch->to_read_only();
    return gpu_aggregate_impl::local_grouped_aggregate(ro,
                                                       group_idx,
                                                       aggregates,
                                                       aggregate_idx,
                                                       {},
                                                       stream,
                                                       *space,
                                                       sirius::telemetry::batch_telemetry_info{},
                                                       hint);
  };

  const sorted_hint_options hint_on{/*enabled=*/true, /*min_rows=*/1};
  const sorted_hint_options hint_off{};

  SECTION("sorted keys: hinted path equals hash path")
  {
    sirius::test::scoped_recording_log_sink logs{"debug"};
    auto hinted = run(sorted_keys, hint_on);
    auto hashed = run(sorted_keys, hint_off);
    stream.synchronize();
    REQUIRE(sirius::test::expect_data_batches_equivalent(hinted, hashed, /*sort=*/true));
    REQUIRE(sorted_hint_engagement_count(logs.records()) == 1);
  }

  SECTION("unsorted keys: is_sorted gate must fall back to the hash path, results identical")
  {
    sirius::test::scoped_recording_log_sink logs{"debug"};
    auto hinted = run(shuffled_keys, hint_on);
    auto hashed = run(shuffled_keys, hint_off);
    stream.synchronize();
    REQUIRE(sirius::test::expect_data_batches_equivalent(hinted, hashed, /*sort=*/true));
    REQUIRE(sorted_hint_engagement_count(logs.records()) == 0);
  }

  SECTION("disabled hint does not engage")
  {
    sirius::test::scoped_recording_log_sink logs{"debug"};
    auto hashed = run(sorted_keys, hint_off);
    stream.synchronize();
    REQUIRE(hashed != nullptr);
    REQUIRE(sorted_hint_engagement_count(logs.records()) == 0);
  }

  SECTION("minimum row threshold gates the hint")
  {
    const sorted_hint_options threshold_gated{true, num_rows + 1};
    sirius::test::scoped_recording_log_sink logs{"debug"};
    auto gated  = run(sorted_keys, threshold_gated);
    auto hashed = run(sorted_keys, hint_off);
    stream.synchronize();
    REQUIRE(sirius::test::expect_data_batches_equivalent(gated, hashed, /*sort=*/true));
    REQUIRE(sorted_hint_engagement_count(logs.records()) == 0);
  }

  SECTION("nullable multi-column keys use ASCENDING nulls-AFTER ordering")
  {
    const std::vector<int64_t> first_keys{0, 0, 0, 0, 1, 1, 1, 1, 0, 0};
    const std::vector<bool> first_valid{
      true, true, true, true, true, true, true, true, false, false};
    const std::vector<int64_t> second_keys{0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    const std::vector<bool> second_valid{
      true, true, false, false, true, true, false, false, true, true};
    const std::vector<int64_t> multikey_values(first_keys.size(), 1);

    auto run_multikey = [&](const std::vector<int64_t>& keys,
                            const std::vector<bool>& valid,
                            const sorted_hint_options& hint) {
      auto batch = make_nullable_multikey_batch(
        first_keys, first_valid, keys, valid, multikey_values, *space, stream);
      auto ro = batch->to_read_only();
      return gpu_aggregate_impl::local_grouped_aggregate(ro,
                                                         std::vector<int>{0, 1},
                                                         aggregates,
                                                         std::vector<int>{2},
                                                         {},
                                                         stream,
                                                         *space,
                                                         sirius::telemetry::batch_telemetry_info{},
                                                         hint);
    };

    sirius::test::scoped_recording_log_sink logs{"debug"};
    auto hinted = run_multikey(second_keys, second_valid, hint_on);
    auto hashed = run_multikey(second_keys, second_valid, hint_off);
    stream.synchronize();
    REQUIRE(sirius::test::expect_data_batches_equivalent(hinted, hashed, /*sort=*/true));
    REQUIRE(sorted_hint_engagement_count(logs.records()) == 1);

    auto unsorted_keys  = second_keys;
    auto unsorted_valid = second_valid;
    std::swap(unsorted_keys[1], unsorted_keys[2]);
    bool const second_was_valid = unsorted_valid[1];
    unsorted_valid[1]           = unsorted_valid[2];
    unsorted_valid[2]           = second_was_valid;
    hinted                      = run_multikey(unsorted_keys, unsorted_valid, hint_on);
    hashed                      = run_multikey(unsorted_keys, unsorted_valid, hint_off);
    stream.synchronize();
    REQUIRE(sirius::test::expect_data_batches_equivalent(hinted, hashed, /*sort=*/true));
    REQUIRE(sorted_hint_engagement_count(logs.records()) == 1);
  }
}
