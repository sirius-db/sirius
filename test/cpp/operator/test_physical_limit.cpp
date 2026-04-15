/*
 * Copyright 2025, Sirius Contributors.
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

#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <op/sirius_physical_limit.hpp>

#include <numeric>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;

namespace {

using namespace sirius::test::operator_utils;
}  // namespace

TEMPLATE_TEST_CASE("sirius_physical_streaming_limit limits rows in data_batch",
                   "[physical_limit]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   bool,
                   decimal64_tag,
                   string_tag,
                   timestamp_us_tag,
                   date32_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  std::vector<typename Traits::type> values(10);
  if constexpr (Traits::is_string) {
    values = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
  } else if constexpr (Traits::is_decimal) {
    for (int i = 0; i < 10; ++i) {
      values[i] = static_cast<typename Traits::type>(i * 100);
    }
  } else if constexpr (Traits::is_ts) {
    for (int i = 0; i < 10; ++i) {
      values[i] = static_cast<typename Traits::type>(i * 1'000'000);
    }
  } else if constexpr (std::is_same_v<typename Traits::type, int32_t> ||
                       std::is_same_v<typename Traits::type, int64_t> ||
                       std::is_same_v<typename Traits::type, int16_t>) {
    std::iota(values.begin(), values.end(), static_cast<typename Traits::type>(0));
  } else if constexpr (std::is_same_v<typename Traits::type, float> ||
                       std::is_same_v<typename Traits::type, double>) {
    for (int i = 0; i < 10; ++i) {
      values[i] = static_cast<typename Traits::type>(i);
    }
  } else if constexpr (std::is_same_v<typename Traits::type, bool>) {
    for (int i = 0; i < 10; ++i) {
      values[i] = (i % 2 == 0);
    }
  }

  std::shared_ptr<data_batch> input_batch;
  if constexpr (Traits::is_string) {
    input_batch = make_string_batch(*space, values);
  } else if constexpr (Traits::is_decimal) {
    input_batch = make_decimal64_batch(*space, values, Traits::scale);
  } else if constexpr (Traits::is_ts) {
    input_batch = make_timestamp_batch(*space, values, Traits::cudf_type);
  } else {
    input_batch = make_numeric_batch<typename Traits::type>(*space, values, Traits::cudf_type);
  }

  auto limit_node  = duckdb::BoundLimitNode::ConstantValue(3);
  auto offset_node = duckdb::BoundLimitNode::ConstantValue(2);

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(Traits::logical_type());

  sirius_physical_streaming_limit limiter(
    std::move(types), std::move(limit_node), std::move(offset_node), values.size(), false);

  std::vector<std::shared_ptr<cucascade::data_batch>> inputs{input_batch};
  auto outputs = limiter.execute(operator_data(inputs), cudf::get_default_stream());
  REQUIRE(outputs->get_data_batches().size() == 1);
  auto output_table =
    outputs->get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto host_vals = copy_column_to_host<typename Traits::type>(output_table.view().column(0));

  std::vector<typename Traits::type> expected = {values[2], values[3], values[4]};
  REQUIRE(host_vals == expected);
}

// Helper to build a single-column int64 batch with sequential values [start, start+count)
static std::shared_ptr<data_batch> make_range_batch(memory_space& space,
                                                    int64_t start,
                                                    int64_t count)
{
  std::vector<int64_t> values(count);
  std::iota(values.begin(), values.end(), start);
  return sirius::test::operator_utils::make_numeric_batch<int64_t>(
    space, values, cudf::type_id::INT64);
}

// Collect all int64 values from multiple output batches into a single host vector
static std::vector<int64_t> collect_all_rows(
  const std::vector<std::shared_ptr<data_batch>>& batches)
{
  std::vector<int64_t> all_rows;
  for (auto const& b : batches) {
    auto table = b->get_data()->cast<gpu_table_representation>().get_table();
    auto col   = sirius::test::operator_utils::copy_column_to_host<int64_t>(table.view().column(0));
    all_rows.insert(all_rows.end(), col.begin(), col.end());
  }
  return all_rows;
}

TEST_CASE("streaming_limit caps total rows across multiple batches",
          "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // 3 batches of 10 rows each: [0..9], [10..19], [20..29]
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 10));
  batches.push_back(make_range_batch(*space, 10, 10));
  batches.push_back(make_range_batch(*space, 20, 10));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_streaming_limit limiter(
    std::move(types), duckdb::BoundLimitNode::ConstantValue(5), duckdb::BoundLimitNode(), 30, true);

  auto outputs = limiter.execute(operator_data(batches), cudf::get_default_stream());
  auto rows    = collect_all_rows(outputs->get_data_batches());

  // Should return exactly 5 rows: [0, 1, 2, 3, 4]
  std::vector<int64_t> expected{0, 1, 2, 3, 4};
  REQUIRE(rows == expected);
}

TEST_CASE("streaming_limit spans across two batches returning correct data",
          "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // 2 batches of 150 rows each: [0..149], [150..299]
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 150));
  batches.push_back(make_range_batch(*space, 150, 150));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  sirius_physical_streaming_limit limiter(std::move(types),
                                          duckdb::BoundLimitNode::ConstantValue(200),
                                          duckdb::BoundLimitNode(),
                                          300,
                                          true);

  auto outputs = limiter.execute(operator_data(batches), cudf::get_default_stream());
  auto rows    = collect_all_rows(outputs->get_data_batches());

  // Should return exactly 200 rows: all 150 from batch 1, plus first 50 from batch 2
  REQUIRE(rows.size() == 200);

  std::vector<int64_t> expected(200);
  std::iota(expected.begin(), expected.end(), 0);  // [0, 1, ..., 199]
  REQUIRE(rows == expected);
}

TEST_CASE("streaming_limit offset spans across multiple batches", "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // 3 batches of 10 rows each: [0..9], [10..19], [20..29]
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 10));
  batches.push_back(make_range_batch(*space, 10, 10));
  batches.push_back(make_range_batch(*space, 20, 10));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  // offset=15 skips all of batch 1 (10 rows) + 5 rows of batch 2, limit=5
  sirius_physical_streaming_limit limiter(std::move(types),
                                          duckdb::BoundLimitNode::ConstantValue(5),
                                          duckdb::BoundLimitNode::ConstantValue(15),
                                          30,
                                          true);

  auto outputs = limiter.execute(operator_data(batches), cudf::get_default_stream());
  auto rows    = collect_all_rows(outputs->get_data_batches());

  // Should return [15, 16, 17, 18, 19]
  std::vector<int64_t> expected{15, 16, 17, 18, 19};
  REQUIRE(rows == expected);
}

TEST_CASE("streaming_limit with separate execute calls enforces global limit",
          "[physical_limit][multi_batch]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  // limit=5 shared across multiple execute() calls (simulating concurrent tasks)
  sirius_physical_streaming_limit limiter(
    std::move(types), duckdb::BoundLimitNode::ConstantValue(5), duckdb::BoundLimitNode(), 30, true);

  // First call with batch [0..9]
  std::vector<std::shared_ptr<data_batch>> batch1{make_range_batch(*space, 0, 10)};
  auto out1  = limiter.execute(operator_data(batch1), cudf::get_default_stream());
  auto rows1 = collect_all_rows(out1->get_data_batches());

  // Second call with batch [10..19] — limit should already be exhausted
  std::vector<std::shared_ptr<data_batch>> batch2{make_range_batch(*space, 10, 10)};
  auto out2  = limiter.execute(operator_data(batch2), cudf::get_default_stream());
  auto rows2 = collect_all_rows(out2->get_data_batches());

  // Total across both calls should be exactly 5
  REQUIRE(rows1.size() + rows2.size() == 5);

  // First call should have taken all 5
  std::vector<int64_t> expected{0, 1, 2, 3, 4};
  REQUIRE(rows1 == expected);
  REQUIRE(rows2.empty());
}
