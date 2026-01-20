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

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/gpu_data_representation.hpp"
#include "memory/memory_space.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "scan/test_utils.hpp"
#include "utils/utils.hpp"

using namespace sirius;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::op;

namespace {

/**
 * @brief Create a batch with random data and acquire a processing handle.
 *
 * In production, tasks hold processing handles while operating on batches
 * to prevent them from being downgraded. This helper creates the batch and
 * returns both the batch and its processing handle.
 *
 * @return A pair of (batch, processing_handle) - keep the handle in scope while processing.
 */
std::pair<std::shared_ptr<data_batch>, data_batch_processing_handle> create_batch_with_random_data(
  const int num_rows,
  const std::vector<cudf::data_type>& column_types,
  std::vector<std::optional<std::pair<int, int>>>& ranges,
  memory_space& mem_space)
{
  // Base input batches, make value ranges small so that we have duplicated partition keys
  for (size_t i = 0; i < ranges.size(); ++i) {
    if (!ranges[i].has_value()) { ranges[i] = {0, 4}; }
  }
  auto table = create_cudf_table_with_random_data(
    num_rows, column_types, ranges, cudf::get_default_stream(), mem_space.get_default_allocator());
  auto batch = sirius::make_data_batch(std::move(table), mem_space);

  REQUIRE(batch->try_to_create_task());
  auto lock_result = batch->try_to_lock_for_processing(mem_space.get_id());
  REQUIRE(lock_result.success);
  data_batch_processing_handle handle = std::move(lock_result.handle);

  return {std::move(batch), std::move(handle)};
}

void copy_data_to_host_by_rows(cudf::table_view table, std::vector<std::vector<int64_t>>& h_rows)
{
  std::vector<std::vector<int64_t>> h_cols(table.num_columns());
  for (int c = 0; c < table.num_columns(); ++c) {
    const auto& col = table.column(c);
    switch (col.type().id()) {
      case cudf::type_id::INT32: {
        std::vector<int32_t> h_buf(table.num_rows());
        cudaMemcpy(h_buf.data(),
                   col.data<int32_t>(),
                   sizeof(int32_t) * table.num_rows(),
                   cudaMemcpyDeviceToHost);
        for (auto val : h_buf) {
          h_cols[c].push_back(val);
        }
        break;
      }
      case cudf::type_id::INT64: {
        std::vector<int64_t> h_buf(table.num_rows());
        cudaMemcpy(h_buf.data(),
                   col.data<int64_t>(),
                   sizeof(int64_t) * table.num_rows(),
                   cudaMemcpyDeviceToHost);
        for (auto val : h_buf) {
          h_cols[c].push_back(val);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported cudf::data_type in `pull_data_to_host()`: " +
                                 std::to_string(static_cast<int>(col.type().id())));
    }
  }
  for (int r = 0; r < table.num_rows(); ++r) {
    h_rows.emplace_back(table.num_columns());
    auto& row = h_rows.back();
    for (int c = 0; c < table.num_columns(); ++c) {
      row[c] = h_cols[c][r];
    }
  }
}

void validate_hash_partition(const data_batch& input_batch,
                             const std::vector<std::shared_ptr<data_batch>>& output_batches,
                             int num_partitions)
{
  cudf::table_view input_table_view = sirius::get_cudf_table_view(input_batch);
  std::vector<cudf::table_view> output_table_views;
  for (const auto& output_batch : output_batches) {
    output_table_views.push_back(sirius::get_cudf_table_view(*output_batch));
  }

  // Check metadata
  REQUIRE(output_batches.size() == static_cast<size_t>(num_partitions));
  int actual_num_rows = 0;
  for (const auto& output_table : output_table_views) {
    actual_num_rows += output_table.num_rows();
    REQUIRE(output_table.num_columns() == input_table_view.num_columns());
    for (int c = 0; c < output_table.num_columns(); ++c) {
      REQUIRE(output_table.column(c).type().id() == input_table_view.column(c).type().id());
    }
  }
  REQUIRE(actual_num_rows == input_table_view.num_rows());

  // Check data
  std::vector<std::vector<int64_t>> h_input_rows;
  copy_data_to_host_by_rows(input_table_view, h_input_rows);
  std::vector<std::vector<std::vector<int64_t>>> h_output_rows(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    copy_data_to_host_by_rows(output_table_views[i], h_output_rows[i]);
  }

  std::multiset<std::vector<int64_t>> output_set;
  for (const auto& partition : h_output_rows) {
    for (const auto& row : partition) {
      REQUIRE(!output_set.contains(row));
    }
    output_set.insert(partition.begin(), partition.end());
  }
  std::multiset<std::vector<int64_t>> input_set(h_input_rows.begin(), h_input_rows.end());
  REQUIRE(input_set == output_set);
}

}  // namespace

TEST_CASE("Hash partition basic", "[operator][hash_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 100;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  // Create batch and acquire processing handle (replaces old pin() call)
  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::hash_partition(
    input_batch, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_hash_partition(*input_batch, output_batches, num_partitions);
  // input_handle automatically releases when going out of scope (replaces old unpin() call)
}

TEST_CASE("Hash partition with invalid input", "[operator][hash_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 100;
  constexpr size_t num_partitions           = 1;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  REQUIRE_THROWS_AS(
    gpu_partition_impl::hash_partition(
      input_batch, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space),
    std::runtime_error);
}

TEST_CASE("Hash partition with empty input", "[operator][hash_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 0;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::hash_partition(
    input_batch, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_hash_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Hash partition with all the same partitioning keys", "[operator][hash_partition]")
{
  auto manager                                           = initialize_memory_manager();
  auto* mem_space                                        = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows                        = 100;
  constexpr size_t num_partitions                        = 4;
  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64},
                                                            cudf::data_type{cudf::type_id::INT32},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx                     = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges = {
    std::optional<std::pair<int, int>>({0, 0}),
    std::optional<std::pair<int, int>>({1, 1}),
    std::nullopt,
    std::nullopt};

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::hash_partition(
    input_batch, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_hash_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Hash partition with num partitions larger than input size", "[operator][hash_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 10;
  constexpr size_t num_partitions           = 20;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> partition_key_idx        = {0, 1};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::hash_partition(
    input_batch, partition_key_idx, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_hash_partition(*input_batch, output_batches, num_partitions);
}

namespace {

void validate_evenly_partition(const data_batch& input_batch,
                               const std::vector<std::shared_ptr<data_batch>>& output_batches,
                               int num_partitions)
{
  cudf::table_view input_table_view = sirius::get_cudf_table_view(input_batch);
  std::vector<cudf::table_view> output_table_views;
  for (const auto& output_batch : output_batches) {
    output_table_views.push_back(sirius::get_cudf_table_view(*output_batch));
  }

  // Check metadata
  REQUIRE(output_batches.size() == static_cast<size_t>(num_partitions));
  int actual_num_rows = 0;
  std::unordered_map<int, int> partition_num_rows_cnt;
  for (const auto& output_table : output_table_views) {
    ++partition_num_rows_cnt[output_table.num_rows()];
    actual_num_rows += output_table.num_rows();
    REQUIRE(output_table.num_columns() == input_table_view.num_columns());
    for (int c = 0; c < output_table.num_columns(); ++c) {
      REQUIRE(output_table.column(c).type().id() == input_table_view.column(c).type().id());
    }
  }
  REQUIRE(actual_num_rows == input_table_view.num_rows());
  REQUIRE(partition_num_rows_cnt[input_table_view.num_rows() / num_partitions + 1] ==
          input_table_view.num_rows() % num_partitions);
  REQUIRE(partition_num_rows_cnt[input_table_view.num_rows() / num_partitions] ==
          num_partitions - input_table_view.num_rows() % num_partitions);

  // Check data
  std::vector<std::vector<int64_t>> h_input_rows;
  copy_data_to_host_by_rows(input_table_view, h_input_rows);
  std::vector<std::vector<std::vector<int64_t>>> h_output_rows(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    copy_data_to_host_by_rows(output_table_views[i], h_output_rows[i]);
  }

  std::multiset<std::vector<int64_t>> output_set;
  for (const auto& partition : h_output_rows) {
    output_set.insert(partition.begin(), partition.end());
  }
  std::multiset<std::vector<int64_t>> input_set(h_input_rows.begin(), h_input_rows.end());
  REQUIRE(input_set == output_set);
}

}  // namespace

TEST_CASE("Evenly partition basic", "[operator][evenly_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 100;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::evenly_partition(
    input_batch, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_evenly_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Evenly partition basic with empty input", "[operator][evenly_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 0;
  constexpr size_t num_partitions           = 4;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::evenly_partition(
    input_batch, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_evenly_partition(*input_batch, output_batches, num_partitions);
}

TEST_CASE("Evenly partition basic with num partitions larger than input size",
          "[operator][evenly_partition]")
{
  auto manager                              = initialize_memory_manager();
  auto* mem_space                           = manager->get_memory_space(Tier::GPU, 0);
  constexpr size_t num_input_rows           = 10;
  constexpr size_t num_partitions           = 20;
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);

  auto [input_batch, input_handle] =
    create_batch_with_random_data(num_input_rows, column_types, ranges, *mem_space);
  auto output_batches = gpu_partition_impl::evenly_partition(
    input_batch, num_partitions, cudf::get_default_stream(), *mem_space);
  validate_evenly_partition(*input_batch, output_batches, num_partitions);
}
