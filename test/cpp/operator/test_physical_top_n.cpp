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
#include <duckdb/common/types.hpp>
#include <duckdb/planner/bound_query_node.hpp>
#include <duckdb/planner/bound_result_modifier.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_top_n.hpp>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {

BoundOrderByNode make_order(idx_t col_idx, OrderType dir = OrderType::DESCENDING)
{
  return BoundOrderByNode(dir,
                          OrderByNullType::NULLS_LAST,
                          make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, col_idx));
}

std::shared_ptr<data_batch> make_batch(data_repository_mgr& repo_mgr,
                                       memory_space& space,
                                       const std::vector<int64_t>& order_vals,
                                       const std::vector<int64_t>& payload_vals)
{
  return make_two_column_batch<int64_t, int64_t>(
    repo_mgr, space, order_vals, payload_vals, cudf::type_id::INT64, std::nullopt);
}

}  // namespace

TEST_CASE("sirius_physical_top_n single-key uses top_k across multiple batches", "[physical_top_n]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch(repo_mgr, *space, {5, 1}, {50, 10}));
  batches.push_back(make_batch(repo_mgr, *space, {7, 3}, {70, 30}));
  batches.push_back(make_batch(repo_mgr, *space, {9}, {90}));
  batches.push_back(make_batch(repo_mgr, *space, {2, 8}, {20, 80}));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType::BIGINT);  // order column
  types.push_back(duckdb::LogicalType::BIGINT);  // payload

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, OrderType::DESCENDING));

  sirius_physical_top_n topn(std::move(types),
                             std::move(orders),
                             /*limit=*/3,
                             /*offset=*/0,
                             nullptr,
                             0,
                             &repo_mgr);

  // Feed in two batches, then two more, ensuring accumulation via internal state.
  auto out1 = topn.execute({batches[0], batches[1]});
  REQUIRE(out1.size() == 1);
  auto out2 = topn.execute({batches[2], batches[3]});
  REQUIRE(out2.size() == 1);

  auto table       = out2[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view        = table.view();
  auto orders_out  = copy_column_to_host<int64_t>(view.column(0));
  auto payload_out = copy_column_to_host<int64_t>(view.column(1));

  std::vector<int64_t> expected_order{9, 8, 7};
  std::vector<int64_t> expected_payload{90, 80, 70};

  REQUIRE(orders_out == expected_order);
  REQUIRE(payload_out == expected_payload);
}

TEST_CASE("sirius_physical_top_n multi-key falls back to sort_by_key", "[physical_top_n]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  data_repository_mgr repo_mgr;

  // order by col0 desc, then col1 asc
  std::vector<std::shared_ptr<data_batch>> batches;
  batches.push_back(make_batch(repo_mgr, *space, {5, 5}, {2, 1}));
  batches.push_back(make_batch(repo_mgr, *space, {7, 7}, {3, 4}));
  batches.push_back(make_batch(repo_mgr, *space, {7, 6}, {1, 9}));
  batches.push_back(make_batch(repo_mgr, *space, {4, 8}, {5, 0}));

  duckdb::vector<duckdb::LogicalType> types;
  types.push_back(duckdb::LogicalType::BIGINT);  // order
  types.push_back(duckdb::LogicalType::BIGINT);  // payload

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.push_back(make_order(0, OrderType::DESCENDING));
  orders.push_back(make_order(1, OrderType::ASCENDING));

  sirius_physical_top_n topn(std::move(types),
                             std::move(orders),
                             /*limit=*/4,
                             /*offset=*/0,
                             nullptr,
                             0,
                             &repo_mgr);

  auto out1 = topn.execute({batches[0], batches[1]});
  REQUIRE(out1.size() == 1);
  auto out2 = topn.execute({batches[2], batches[3]});
  REQUIRE(out2.size() == 1);

  auto table       = out2[0]->get_data()->cast<gpu_table_representation>().get_table();
  auto view        = table.view();
  auto orders_out  = copy_column_to_host<int64_t>(view.column(0));
  auto payload_out = copy_column_to_host<int64_t>(view.column(1));

  // Expected ordering: (8,0), (7,1), (7,3), (7,4)
  std::vector<int64_t> expected_order{8, 7, 7, 7};
  std::vector<int64_t> expected_payload{0, 1, 3, 4};

  REQUIRE(orders_out == expected_order);
  REQUIRE(payload_out == expected_payload);
}
