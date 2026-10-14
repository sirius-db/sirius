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

#include "op/partition/stable_hash_partition.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace {

using sirius::test::vector_to_cudf_column;
using sirius::test::operator_utils::gpu_type_traits;

template <typename T>
std::vector<T> to_host(cudf::column_view const& column)
{
  std::vector<T> values(column.size());
  cudaMemcpy(values.data(), column.data<T>(), values.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return values;
}

/// A table of [key INT64, row_id INT32], with keys drawn from [0, num_keys).
std::unique_ptr<cudf::table> make_table(int num_rows, int64_t num_keys, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int64_t> key(0, num_keys - 1);
  std::vector<int64_t> keys;
  std::vector<int32_t> row_ids;
  for (int row = 0; row < num_rows; ++row) {
    keys.push_back(key(rng));
    row_ids.push_back(row);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(vector_to_cudf_column<gpu_type_traits<int64_t>>(keys));
  columns.push_back(vector_to_cudf_column<gpu_type_traits<int32_t>>(row_ids));
  return std::make_unique<cudf::table>(std::move(columns));
}

/// Partitions @p table and returns each key's partition, checking order and offsets on the way.
std::map<int64_t, int> partition_and_check(cudf::table_view const& table, int num_partitions)
{
  auto [partitioned, offsets] =
    sirius::op::stable_hash_partition(table,
                                      {0},
                                      num_partitions,
                                      cudf::get_default_stream(),
                                      cudf::get_current_device_resource_ref());
  REQUIRE(partitioned->num_rows() == table.num_rows());
  REQUIRE(offsets.size() == static_cast<std::size_t>(num_partitions));
  REQUIRE(offsets[0] == 0);
  auto const keys    = to_host<int64_t>(partitioned->get_column(0));
  auto const row_ids = to_host<int32_t>(partitioned->get_column(1));
  std::map<int64_t, int> partition_of_key;
  for (int partition = 0; partition < num_partitions; ++partition) {
    auto const begin = offsets[partition];
    auto const end =
      partition + 1 < num_partitions ? offsets[partition + 1] : partitioned->num_rows();
    REQUIRE(begin <= end);
    for (auto row = begin; row < end; ++row) {
      // Rows keep their input order within a partition.
      if (row > begin) { REQUIRE(row_ids[row] > row_ids[row - 1]); }
      // Every occurrence of a key lands in one partition.
      auto const [it, inserted] = partition_of_key.emplace(keys[row], partition);
      REQUIRE(it->second == partition);
    }
  }
  return partition_of_key;
}

}  // namespace

TEST_CASE("stable_hash_partition keeps input order and groups equal keys",
          "[partition][stable_hash_partition]")
{
  auto const table     = make_table(100'000, 5'000, 1);
  auto const placement = partition_and_check(table->view(), 4);
  CHECK(placement.size() > 1);
}

TEST_CASE("stable_hash_partition places a key in the same partition for every input",
          "[partition][stable_hash_partition]")
{
  // Producers feeding one consumer must agree on each key's partition.
  auto const first        = partition_and_check(make_table(50'000, 2'000, 1)->view(), 7);
  auto const second       = partition_and_check(make_table(80'000, 2'000, 2)->view(), 7);
  std::size_t shared_keys = 0;
  for (auto const& [key, partition] : first) {
    auto const it = second.find(key);
    if (it == second.end()) { continue; }
    ++shared_keys;
    CHECK(it->second == partition);
  }
  CHECK(shared_keys > 1'000);
}

TEST_CASE("stable_hash_partition handles a table with no rows",
          "[partition][stable_hash_partition]")
{
  auto const table            = make_table(0, 1, 1);
  auto [partitioned, offsets] = sirius::op::stable_hash_partition(
    table->view(), {0}, 3, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  CHECK(partitioned->num_rows() == 0);
  CHECK(offsets == std::vector<cudf::size_type>{0, 0, 0});
}
