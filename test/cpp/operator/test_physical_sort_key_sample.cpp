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

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_sort_key_sample.hpp>
#include <op/sort_key_sample.hpp>
#include <sirius/exception.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace sirius::op;
using namespace cucascade;
using namespace sirius::test::operator_utils;

namespace {

duckdb::vector<sirius::logical_type> bigint_types(std::size_t columns)
{
  duckdb::vector<duckdb::LogicalType> types;
  for (std::size_t i = 0; i < columns; ++i) {
    types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  }
  return sirius::from_duckdb_vec(std::move(types));
}

duckdb::vector<duckdb::BoundOrderByNode> order_on(duckdb::idx_t column)
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(duckdb::OrderType::ASCENDING,
                      duckdb::OrderByNullType::NULLS_LAST,
                      duckdb::make_uniq<duckdb::BoundReferenceExpression>(
                        duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT), column));
  return orders;
}

std::shared_ptr<data_batch> sorted_batch(cucascade::memory::memory_space& space,
                                         std::vector<std::int64_t> const& keys)
{
  return make_numeric_batch<std::int64_t>(space, keys, cudf::type_id::INT64);
}

}  // namespace

TEST_CASE("sirius_physical_sort_key_sample keeps the key rows of each batch once",
          "[physical_sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();

  auto first  = sorted_batch(*space, {1, 3, 5, 9});
  auto second = sorted_batch(*space, {2, 7, 8});

  auto keys = std::make_shared<sort_key_sample>();
  sirius_physical_sort_key_sample op(bigint_types(1), order_on(0), 0, keys);
  CHECK(op.is_sink());
  CHECK(op.is_source());

  pipelineable_operator_data input({first, second});
  auto const out     = op.execute(input, stream);
  auto const& passed = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
  REQUIRE(passed.size() == 2);
  CHECK(passed[0] == first);
  CHECK(passed[1] == second);
  CHECK(keys->seen().batches == 2);
  CHECK(keys->seen().rows == 7);

  // A retried task hands the same batches over again: they still count once.
  (void)op.execute(input, stream);
  CHECK(keys->seen().batches == 2);
  CHECK(keys->seen().rows == 7);

  // A retry on another GPU gets clones of its input, which have new batch IDs.
  auto const cloned = input.with_batches_preserving_rows(
    {sorted_batch(*space, {1, 3, 5, 9}), sorted_batch(*space, {2, 7, 8})});
  REQUIRE(cloned->original_batch_ids() ==
          std::vector<std::uint64_t>{first->get_batch_id(), second->get_batch_id()});
  (void)op.execute(*cloned, stream);
  CHECK(keys->seen().batches == 2);
  CHECK(keys->seen().rows == 7);
}

TEST_CASE("sirius_physical_sort_key_sample keeps the sort key columns only",
          "[physical_sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();

  std::vector<std::int64_t> const payload{7, 8, 9};
  std::vector<std::int64_t> const key{10, 20, 30};
  auto batch = make_two_column_batch<std::int64_t, std::int64_t>(
    *space, payload, key, cudf::type_id::INT64, std::nullopt);

  auto keys = std::make_shared<sort_key_sample>();
  sirius_physical_sort_key_sample op(bigint_types(2), order_on(1), 0, keys);
  (void)op.execute(pipelineable_operator_data({batch}), stream);

  auto const kept = keys->on_device(*space, stream);
  REQUIRE(kept.tables.size() == 1);
  REQUIRE(kept.tables[0].num_columns() == 1);
  CHECK(copy_column_to_host<std::int64_t>(kept.tables[0].column(0)) == key);
}

TEST_CASE("sirius_physical_sort_key_sample passes batches through once the store is closed",
          "[physical_sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);
  auto const stream = cudf::get_default_stream();

  auto keys = std::make_shared<sort_key_sample>();
  sirius_physical_sort_key_sample op(bigint_types(1), order_on(0), 0, keys);
  keys->close();

  auto batch         = sorted_batch(*space, {1, 2, 3});
  auto const out     = op.execute(pipelineable_operator_data({batch}), stream);
  auto const& passed = dynamic_cast<pipelineable_operator_data const&>(*out).get_data_batches();
  REQUIRE(passed.size() == 1);
  CHECK(passed[0] == batch);
  CHECK(keys->seen().batches == 0);
}

TEST_CASE("sirius_physical_sort_key_sample rejects a key that is not a column",
          "[physical_sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space);

  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(duckdb::OrderType::ASCENDING,
                      duckdb::OrderByNullType::NULLS_LAST,
                      duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::BIGINT(1)));
  sirius_physical_sort_key_sample op(
    bigint_types(1), std::move(orders), 0, std::make_shared<sort_key_sample>());

  auto batch = sorted_batch(*space, {1, 2, 3});
  CHECK_THROWS_AS(op.execute(pipelineable_operator_data({batch}), cudf::get_default_stream()),
                  sirius::not_implemented_exception);
}

TEST_CASE("sirius_physical_sort_key_sample needs a key-row store", "[physical_sort_key_sample]")
{
  CHECK_THROWS_AS(sirius_physical_sort_key_sample(bigint_types(1), order_on(0), 0, nullptr),
                  std::invalid_argument);
}
