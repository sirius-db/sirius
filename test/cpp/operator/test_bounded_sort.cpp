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

#include "catch.hpp"
#include "data/convertible_data_batch.hpp"
#include "op/order/bounded_sort.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_order.hpp"
#include "operator_test_utils.hpp"

#include <rmm/cuda_stream.hpp>

using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {
duckdb::vector<duckdb::BoundOrderByNode> ascending_orders()
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0));
  return orders;
}
sirius_physical_order make_local_sort()
{
  return sirius_physical_order(
    {sirius::logical_type::make(sirius::type_id::BIGINT)}, 1, ascending_orders(), {0}, 0);
}
}  // namespace

TEST_CASE("bounded initial sorting owns independent chunks", "[bounded_sort]")
{
  auto* space       = get_default_gpu_space();
  auto local        = make_local_sort();
  local.chunk_bytes = 24;
  auto batch =
    make_numeric_batch<int64_t>(*space, {9, 7, 8, 3, 1, 2, 6, 4, 5}, cudf::type_id::INT64);
  auto output        = local.execute(pipelineable_operator_data({batch}), default_stream());
  auto const& chunks = dynamic_cast<pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(chunks.size() == 3);
  std::vector<std::vector<int64_t>> expected{{7, 8, 9}, {1, 2, 3}, {4, 5, 6}};
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    auto ro    = chunks[i]->to_read_only();
    auto table = sirius::get_cudf_table_view(ro);
    CHECK(copy_column_to_host<int64_t>(table.column(0)) == expected[i]);
    CHECK(ro.get_data()->get_uncompressed_data_size_in_bytes() <= local.chunk_bytes);
  }
}

TEST_CASE("row slicing uses metadata and allows skew to exceed the byte target", "[bounded_sort]")
{
  auto* space = get_default_gpu_space();
  auto batch  = make_string_batch(*space, {"a", "b", std::string(200, 'c'), "d"});
  auto ro     = batch->to_read_only();
  auto table  = sirius::get_cudf_table_view(ro);
  auto slices = bounded_sort_slices(
    table, 128, ro.get_data()->get_uncompressed_data_size_in_bytes(), default_stream());
  REQUIRE(slices.size() >= 2);
  std::size_t rows     = 0;
  bool exceeded_target = false;
  for (auto slice : slices) {
    rows += slice.num_rows();
    auto copy = std::make_unique<cudf::table>(slice, default_stream(), get_resource_ref(*space));
    exceeded_target |= copy->alloc_size() > 128;
  }
  default_stream().sync();
  CHECK(rows == 4);
  CHECK(exceeded_target);
}

TEST_CASE("row sizing caps native arithmetic without allocating native-sized tables",
          "[bounded_sort][cpu]")
{
  auto native = static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max());
  CHECK(sort_chunk_rows(0, 0, 1) == 0);
  CHECK(sort_chunk_rows(1, 1000000, 1) == 1);
  CHECK(sort_chunk_rows(native * 4, 0, std::numeric_limits<std::size_t>::max()) <= native);
  CHECK(sort_chunk_rows(std::numeric_limits<uint64_t>::max(), 1, 1) == 1);
  CHECK_THROWS_AS(sort_chunk_rows(10, 10, 0), sirius::invalid_input_exception);
}

TEST_CASE("merge scratch registration preserves the input repository", "[bounded_sort]")
{
  auto local = make_local_sort();
  sirius_physical_merge_sort merge(&local);
  merge.chunk_bytes = 1024;
  merge.operator_id = 17;
  cucascade::shared_data_repository_manager manager;
  manager.add_new_repository(17, "default", std::make_unique<cucascade::shared_data_repository>());
  auto* original = manager.get_repository(17, "default").get();
  merge.register_scratch(manager);
  CHECK(manager.get_repositories().size() == 6);
  CHECK(manager.get_repository(17, "default").get() == original);
  CHECK(manager.get_repository(17, "sort_tile_inputs").get() != original);
  CHECK_THROWS_AS(merge.register_scratch(manager), sirius::internal_exception);
}

TEST_CASE("registered bounded chunks spill and restore independently", "[bounded_sort]")
{
  auto manager       = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu          = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* host         = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  auto local         = make_local_sort();
  local.chunk_bytes  = 16;
  auto batch         = make_numeric_batch<int64_t>(*gpu, {4, 3, 2, 1}, cudf::type_id::INT64);
  auto output        = local.execute(pipelineable_operator_data({batch}), default_stream());
  auto const& chunks = dynamic_cast<pipelineable_operator_data&>(*output).get_data_batches();
  REQUIRE(chunks.size() == 2);
  sirius_physical_merge_sort merge(&local);
  merge.operator_id = 1;
  merge.chunk_bytes = 16;
  cucascade::shared_data_repository_manager repositories;
  merge.register_scratch(repositories);
  auto* scratch = repositories.get_repository(1, "sort_tile_inputs").get();
  scratch->add_data_batch_at_position(chunks[0], 0);
  scratch->add_data_batch_at_position(chunks[1], 1);
  sirius::convertible_data_batch_provider candidates(scratch);
  auto candidate = candidates.get_next_convertible(gpu, true);
  REQUIRE(candidate);
  rmm::cuda_stream transfer;
  REQUIRE(candidate->convert({host}, transfer, *manager, true).has_value());
  {
    auto first  = chunks[0]->to_read_only();
    auto second = chunks[1]->to_read_only();
    CHECK(first.get_memory_space() == host);
    CHECK(second.get_memory_space() == gpu);
  }
  pipelineable_operator_data restored({chunks[0]});
  restored.prepare_for_processing(gpu, transfer);
  auto reads = restored.get_read_only_batches();
  CHECK(copy_column_to_host<int64_t>(sirius::get_cudf_table_view(reads.front()).column(0)) ==
        std::vector<int64_t>{3, 4});
  CHECK(reads.front().get_data()->get_uncompressed_data_size_in_bytes() <= 16);
}
