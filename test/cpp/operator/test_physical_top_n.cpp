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

#include <catch.hpp>
#include <duckdb/planner/bound_query_node.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_top_n.hpp>
#include <op/sirius_physical_top_n_merge.hpp>

using namespace sirius::op;
using sirius::op::operator_data;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

// Helper functions - defined outside anonymous namespace to avoid ODR issues with
// LogicalType::BIGINT
static duckdb::BoundOrderByNode make_order(duckdb::idx_t col_idx,
                                           duckdb::OrderType dir = duckdb::OrderType::DESCENDING)
{
  return duckdb::BoundOrderByNode(dir,
                                  duckdb::OrderByNullType::NULLS_LAST,
                                  duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                                    duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), col_idx));
}

static std::shared_ptr<data_batch> make_batch(memory_space& space,
                                              const std::vector<int64_t>& order_vals,
                                              const std::vector<int64_t>& payload_vals)
{
  return make_two_column_batch<int64_t, int64_t>(
    space, order_vals, payload_vals, cudf::type_id::INT64, std::nullopt);
}

static std::shared_ptr<data_batch> make_range_batch(memory_space& space,
                                                    int64_t start,
                                                    int64_t count,
                                                    int64_t payload_scale)
{
  std::vector<int64_t> order_vals;
  std::vector<int64_t> payload_vals;
  order_vals.reserve(count);
  payload_vals.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    auto value = start + i;
    order_vals.push_back(value);
    payload_vals.push_back(value * payload_scale);
  }
  return make_batch(space, order_vals, payload_vals);
}

TEST_CASE("sirius_physical_top_n single-key uses top_k per batch", "[physical_top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch(*space, {5, 1, 7, 3, 9, 2, 8}, {50, 10, 70, 30, 90, 20, 80}));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // order column
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // payload

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::OrderType::DESCENDING));

  sirius_physical_top_n topn(std::move(types),
                             std::move(orders),
                             /*limit=*/3,
                             /*offset=*/0,
                             nullptr,
                             0);

  auto out = topn.execute(operator_data({batches[0]}), cudf::get_default_stream());
  REQUIRE(out.get_data_batches().size() == 1);

  auto table = out.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto orders_out  = copy_column_to_host<int64_t>(view.column(0));
  auto payload_out = copy_column_to_host<int64_t>(view.column(1));

  std::vector<int64_t> expected_order{9, 8, 7};
  std::vector<int64_t> expected_payload{90, 80, 70};

  REQUIRE(orders_out == expected_order);
  REQUIRE(payload_out == expected_payload);
}

TEST_CASE("sirius_physical_top_n multi-key falls back to sort_by_key", "[physical_top_n]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // order by col0 desc, then col1 asc
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch(*space, {5, 5, 7, 7, 7, 6, 4, 8}, {2, 1, 3, 4, 1, 9, 5, 0}));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // order
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // payload

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::OrderType::DESCENDING));
  orders.push_back(make_order(1, duckdb::OrderType::ASCENDING));

  sirius_physical_top_n topn(std::move(types),
                             std::move(orders),
                             /*limit=*/4,
                             /*offset=*/0,
                             nullptr,
                             0);

  auto out = topn.execute(operator_data({batches[0]}), cudf::get_default_stream());
  REQUIRE(out.get_data_batches().size() == 1);

  auto table = out.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto orders_out  = copy_column_to_host<int64_t>(view.column(0));
  auto payload_out = copy_column_to_host<int64_t>(view.column(1));

  // Expected ordering: (8,0), (7,1), (7,3), (7,4)
  std::vector<int64_t> expected_order{8, 7, 7, 7};
  std::vector<int64_t> expected_payload{0, 1, 3, 4};

  REQUIRE(orders_out == expected_order);
  REQUIRE(payload_out == expected_payload);
}

TEST_CASE("sirius_physical_top_n_merge applies offset and limit", "[physical_top_n_merge]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 10, 10));
  batches.push_back(make_range_batch(*space, 10, 10, 10));
  batches.push_back(make_range_batch(*space, 20, 10, 10));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::OrderType::DESCENDING));

  sirius_physical_top_n_merge topn_merge(std::move(types),
                                         std::move(orders),
                                         /*limit=*/5,
                                         /*offset=*/3,
                                         nullptr,
                                         0);

  auto out = topn_merge.execute(operator_data(batches), cudf::get_default_stream());
  REQUIRE(out.get_data_batches().size() == 1);

  auto table = out.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view  = table.view();
  auto orders_out  = copy_column_to_host<int64_t>(view.column(0));
  auto payload_out = copy_column_to_host<int64_t>(view.column(1));

  std::vector<int64_t> expected_order{26, 25, 24, 23, 22};
  std::vector<int64_t> expected_payload{260, 250, 240, 230, 220};

  REQUIRE(orders_out == expected_order);
  REQUIRE(payload_out == expected_payload);
}

TEST_CASE("sirius_physical_top_n_merge returns empty for limit 0", "[physical_top_n_merge]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_range_batch(*space, 0, 5, 1));
  batches.push_back(make_range_batch(*space, 5, 5, 1));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::OrderType::DESCENDING));

  sirius_physical_top_n_merge topn_merge(std::move(types),
                                         std::move(orders),
                                         /*limit=*/0,
                                         /*offset=*/2,
                                         nullptr,
                                         0);

  auto out = topn_merge.execute(operator_data(batches), cudf::get_default_stream());
  REQUIRE(out.get_data_batches().empty());
}

TEST_CASE("sirius_physical_top_n_merge handles empty batches", "[physical_top_n_merge]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch(*space, {}, {}));
  batches.push_back(make_batch(*space, {}, {}));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, duckdb::OrderType::DESCENDING));

  sirius_physical_top_n_merge topn_merge(std::move(types),
                                         std::move(orders),
                                         /*limit=*/3,
                                         /*offset=*/0,
                                         nullptr,
                                         0);

  auto out = topn_merge.execute(operator_data(batches), cudf::get_default_stream());
  REQUIRE(out.get_data_batches().size() == 1);

  auto table = out.get_data_batches()[0]->get_data()->cast<gpu_table_representation>().get_table();
  REQUIRE(table.num_rows() == 0);
}
