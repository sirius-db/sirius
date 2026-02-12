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
  REQUIRE(outputs.get_data_batches().size() == 1);
  auto output_table =
    outputs.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto host_vals = copy_column_to_host<typename Traits::type>(output_table.view().column(0));

  std::vector<typename Traits::type> expected = {values[2], values[3], values[4]};
  REQUIRE(host_vals == expected);
}
