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
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/order/gpu_order_impl.hpp"
#include "scan/test_utils.hpp"
#include "utils/utils.hpp"

#include <cudf/utilities/bit.hpp>

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

using namespace sirius;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::op;

namespace {

/**
 * @brief Get a shared memory space that persists across all test cases.
 *
 * Uses static initialization to create the memory manager only once,
 * avoiding repeated CUDA memory pool creation/destruction which can
 * crash the CUDA driver.
 */
memory_space* get_shared_mem_space()
{
  static auto manager = initialize_memory_manager();
  return manager->get_memory_space(Tier::GPU, 0);
}

/**
 * @brief Container for batches with their processing handles.
 *
 * In production, tasks hold processing handles while operating on batches
 * to prevent them from being downgraded. This struct keeps batches and handles
 * together so the handles remain in scope during processing.
 */
struct batches_with_handles {
  std::vector<std::shared_ptr<data_batch>> batches;
  std::vector<data_batch_processing_handle> handles;
};

batches_with_handles create_batches_with_random_data(
  const int num_batches,
  const std::vector<int>& num_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<std::optional<std::pair<int, int>>>& ranges,
  memory_space& mem_space)
{
  batches_with_handles result;
  for (int i = 0; i < num_batches; ++i) {
    auto table = create_cudf_table_with_random_data(num_rows[i],
                                                    column_types,
                                                    ranges,
                                                    cudf::get_default_stream(),
                                                    mem_space.get_default_allocator());
    auto batch = sirius::make_data_batch(std::move(table), mem_space);

    // Acquire processing handle (like the old pin() call)
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(mem_space.get_id());
    REQUIRE(lock_result.success);
    result.handles.emplace_back(std::move(lock_result.handle));
    result.batches.push_back(std::move(batch));
  }
  return result;
}

void validate_concat(const std::vector<std::shared_ptr<data_batch>>& input_batches,
                     const data_batch& output)
{
  std::vector<cudf::table_view> input_table_views;
  int expected_num_rows = 0;
  for (const auto& input_batch : input_batches) {
    input_table_views.push_back(sirius::get_cudf_table_view(*input_batch));
    expected_num_rows += input_table_views.back().num_rows();
  }
  cudf::table_view output_table_view = sirius::get_cudf_table_view(output);

  REQUIRE(expected_num_rows == output_table_view.num_rows());
  REQUIRE(input_table_views[0].num_columns() == output_table_view.num_columns());

  for (int c = 0; c < output_table_view.num_columns(); ++c) {
    REQUIRE(input_table_views[0].column(c).type().id() == output_table_view.column(c).type().id());
    if (expected_num_rows == 0) { continue; }

    switch (output_table_view.column(c).type().id()) {
      case cudf::type_id::INT32: {
        std::vector<int32_t> actual_data(expected_num_rows), expected_data(expected_num_rows);
        cudaMemcpy(actual_data.data(),
                   output_table_view.column(c).data<int32_t>(),
                   sizeof(int32_t) * expected_num_rows,
                   cudaMemcpyDeviceToHost);
        int num_input_copied = 0;
        for (const auto& input_table_view : input_table_views) {
          cudaMemcpy(expected_data.data() + num_input_copied,
                     input_table_view.column(c).data<int32_t>(),
                     sizeof(int32_t) * input_table_view.num_rows(),
                     cudaMemcpyDeviceToHost);
          num_input_copied += input_table_view.num_rows();
        }
        for (int r = 0; r < expected_num_rows; ++r) {
          REQUIRE(expected_data[r] == actual_data[r]);
        }
        break;
      }
      case cudf::type_id::STRING: {
        std::vector<cudf::size_type> actual_offsets(expected_num_rows + 1);
        cudf::strings_column_view str_col(output_table_view.column(c));
        cudaMemcpy(actual_offsets.data(),
                   str_col.offsets().data<cudf::size_type>(),
                   (expected_num_rows + 1) * sizeof(cudf::size_type),
                   cudaMemcpyDeviceToHost);
        std::vector<char> actual_data(actual_offsets.back());
        cudaMemcpy(actual_data.data(),
                   str_col.chars_begin(cudf::get_default_stream()),
                   actual_offsets.back(),
                   cudaMemcpyDeviceToHost);

        std::vector<cudf::size_type> expected_offsets{0};
        std::vector<char> expected_data(actual_data.size());
        for (size_t i = 0; i < input_batches.size(); ++i) {
          if (input_table_views[i].num_rows() == 0) { continue; }
          std::vector<cudf::size_type> input_offsets(input_table_views[i].num_rows() + 1);
          str_col = cudf::strings_column_view(input_table_views[i].column(c));
          cudaMemcpy(input_offsets.data(),
                     str_col.offsets().data<cudf::size_type>(),
                     (input_table_views[i].num_rows() + 1) * sizeof(cudf::size_type),
                     cudaMemcpyDeviceToHost);
          int curr_last_offset = expected_offsets.back();
          for (int r = 1; r <= input_table_views[i].num_rows(); ++r) {
            expected_offsets.push_back(curr_last_offset + input_offsets[r]);
          }
          cudaMemcpy(expected_data.data() + curr_last_offset,
                     str_col.chars_begin(cudf::get_default_stream()),
                     input_offsets.back(),
                     cudaMemcpyDeviceToHost);
        }

        for (int r = 0; r <= expected_num_rows; ++r) {
          REQUIRE(expected_offsets[r] == actual_offsets[r]);
        }
        for (size_t i = 0; i < expected_data.size(); ++i) {
          REQUIRE(expected_data[i] == actual_data[i]);
        }
        break;
      }
      default: throw std::runtime_error("Unsupported cudf::data_type in `validate_concat()`");
    }
  }
}

}  // namespace

TEST_CASE("Concatenate multiple data batches", "[operator][merge_concat]")
{
  auto* mem_space = get_shared_mem_space();
  REQUIRE(mem_space);

  constexpr int num_batches           = 10;
  constexpr size_t num_rows_per_batch = 100;
  std::vector<int> num_input_rows(num_batches, num_rows_per_batch);
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::STRING}};

  // Create batches with processing handles (replaces old pin() calls)
  auto input = create_batches_with_random_data(
    num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt}, *mem_space);
  auto output_batch = gpu_merge_impl::concat(input.batches, cudf::get_default_stream(), *mem_space);
  validate_concat(input.batches, *output_batch);
}

TEST_CASE("Concatenate multiple data batches with different size", "[operator][merge_concat]")
{
  auto* mem_space           = get_shared_mem_space();
  constexpr int num_batches = 10;
  std::vector<int> num_input_rows;
  for (int i = 0; i < num_batches; ++i) {
    num_input_rows.push_back((i + 1) * 10);
  }
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::STRING}};

  auto input = create_batches_with_random_data(
    num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt}, *mem_space);
  auto output_batch = gpu_merge_impl::concat(input.batches, cudf::get_default_stream(), *mem_space);
  validate_concat(input.batches, *output_batch);
}

TEST_CASE("Concatenate with invalid input", "[operator][merge_concat]")
{
  auto* mem_space                     = get_shared_mem_space();
  constexpr int num_batches           = 1;
  constexpr size_t num_rows_per_batch = 100;
  std::vector<int> num_input_rows(num_batches, num_rows_per_batch);
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::STRING}};

  // Invalid input: less than two input batches
  auto input = create_batches_with_random_data(
    num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt}, *mem_space);
  REQUIRE_THROWS_AS(gpu_merge_impl::concat(input.batches, cudf::get_default_stream(), *mem_space),
                    std::runtime_error);
}

TEST_CASE("Concatenate multiple data batches but no input rows", "[operator][merge_concat]")
{
  auto* mem_space                     = get_shared_mem_space();
  constexpr int num_batches           = 10;
  constexpr size_t num_rows_per_batch = 0;
  std::vector<int> num_input_rows(num_batches, num_rows_per_batch);
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::STRING}};

  auto input = create_batches_with_random_data(
    num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt}, *mem_space);
  auto output_batch = gpu_merge_impl::concat(input.batches, cudf::get_default_stream(), *mem_space);
  validate_concat(input.batches, *output_batch);
}

TEST_CASE("Concatenate mixed empty and non-empty data batches", "[operator][merge_concat]")
{
  auto* mem_space           = get_shared_mem_space();
  constexpr int num_batches = 10;
  std::vector<int> num_input_rows;
  for (int i = 0; i < num_batches; ++i) {
    num_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
  }
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::STRING}};

  auto input = create_batches_with_random_data(
    num_batches, num_input_rows, column_types, {column_types.size(), std::nullopt}, *mem_space);
  auto output_batch = gpu_merge_impl::concat(input.batches, cudf::get_default_stream(), *mem_space);
  validate_concat(input.batches, *output_batch);
}

namespace {

batches_with_handles create_batches_with_local_ungrouped_agg_result(
  const int num_batches,
  const std::vector<int>& num_base_input_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  memory_space& mem_space)
{
  // Base input batches (with processing handles)
  auto base_input = create_batches_with_random_data(
    num_batches, num_base_input_rows, column_types, {column_types.size(), std::nullopt}, mem_space);

  // Compute local ungrouped aggregates
  batches_with_handles result;
  std::vector<int> aggregate_idx(aggregates.size());
  for (size_t i = 0; i < aggregates.size(); ++i) {
    aggregate_idx[i] = i;
  }
  for (int i = 0; i < num_batches; ++i) {
    auto batch = gpu_aggregate_impl::local_ungrouped_aggregate(
      base_input.batches[i], aggregates, aggregate_idx, cudf::get_default_stream(), mem_space);
    // Acquire processing handle for the output batch
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(mem_space.get_id());
    REQUIRE(lock_result.success);
    result.handles.emplace_back(std::move(lock_result.handle));
    result.batches.push_back(std::move(batch));
  }
  // base_input.handles will release when going out of scope (base batches no longer needed)
  return result;
}

template <typename T>
void validate_ungrouped_aggregate_numeric(const std::vector<cudf::table_view>& input_table_views,
                                          cudf::table_view output_table_view,
                                          const std::vector<cudf::aggregation::Kind>& aggregates,
                                          int c)
{
  // Handle the case where there is no input
  int num_valid_input_rows = 0;
  for (const auto& input_table_view : input_table_views) {
    const auto& col = input_table_view.column(c);
    num_valid_input_rows += input_table_view.num_rows() - col.null_count();
  }
  if (num_valid_input_rows == 0) {
    REQUIRE(output_table_view.column(c).null_count() == 1);
    return;
  }

  // Compare result
  T actual_result;
  cudaMemcpy(
    &actual_result, output_table_view.column(c).data<T>(), sizeof(T), cudaMemcpyDeviceToHost);
  std::vector<T> input_data_without_nulls;
  for (const auto& input_table_view : input_table_views) {
    std::vector<T> input_data(input_table_view.num_rows());
    cudaMemcpy(input_data.data(),
               input_table_view.column(c).data<T>(),
               sizeof(T) * input_table_view.num_rows(),
               cudaMemcpyDeviceToHost);
    auto* d_null_mask = input_table_view.column(c).null_mask();
    if (d_null_mask == nullptr) {
      input_data_without_nulls.insert(
        input_data_without_nulls.end(), input_data.begin(), input_data.end());
    } else {
      std::vector<cudf::bitmask_type> h_null_mask(
        cudf::bitmask_allocation_size_bytes(input_table_view.num_rows()) /
        sizeof(cudf::bitmask_type));
      cudaMemcpy(h_null_mask.data(),
                 d_null_mask,
                 h_null_mask.size() * sizeof(cudf::bitmask_type),
                 cudaMemcpyDeviceToHost);
      for (int r = 0; r < input_table_view.num_rows(); ++r) {
        if (cudf::bit_is_set(h_null_mask.data(), r)) {
          input_data_without_nulls.push_back(input_data[r]);
        }
      }
    }
  }

  switch (aggregates[c]) {
    case cudf::aggregation::Kind::MIN: {
      T expected_result =
        *std::min_element(input_data_without_nulls.begin(), input_data_without_nulls.end());
      REQUIRE(expected_result == actual_result);
      break;
    }
    case cudf::aggregation::Kind::MAX: {
      T expected_result =
        *std::max_element(input_data_without_nulls.begin(), input_data_without_nulls.end());
      REQUIRE(expected_result == actual_result);
      break;
    }
    case cudf::aggregation::Kind::SUM:
    case cudf::aggregation::Kind::COUNT_ALL:
    case cudf::aggregation::Kind::COUNT_VALID: {
      int64_t expected_result = std::accumulate(
        input_data_without_nulls.begin(), input_data_without_nulls.end(), int64_t{0});
      REQUIRE(expected_result == actual_result);
      break;
    }
    default: break;
  }
}

void validate_ungrouped_aggregate(const std::vector<std::shared_ptr<data_batch>>& input_batches,
                                  const data_batch& output,
                                  const std::vector<cudf::aggregation::Kind>& aggregates)
{
  std::vector<cudf::table_view> input_table_views;
  for (const auto& input_batch : input_batches) {
    input_table_views.push_back(sirius::get_cudf_table_view(*input_batch));
  }
  cudf::table_view output_table_view = sirius::get_cudf_table_view(output);

  REQUIRE(output_table_view.num_rows() == 1);

  for (int c = 0; c < output_table_view.num_columns(); ++c) {
    REQUIRE(output_table_view.column(c).type().id() == input_table_views[0].column(c).type().id());

    switch (output_table_view.column(c).type().id()) {
      case cudf::type_id::INT32: {
        validate_ungrouped_aggregate_numeric<int32_t>(
          input_table_views, output_table_view, aggregates, c);
        break;
      }
      case cudf::type_id::INT64: {
        validate_ungrouped_aggregate_numeric<int64_t>(
          input_table_views, output_table_view, aggregates, c);
        break;
      }
      default:
        throw std::runtime_error(
          "Unsupported cudf::data_type in `validate_ungrouped_aggregate()`: " +
          std::to_string(static_cast<int>(output_table_view.column(c).type().id())));
    }
  }
}

}  // namespace

TEST_CASE("Ungrouped merge aggregate of min/max/count/sum", "[operator][merge_ungrouped_agg]")
{
  auto* mem_space                                = get_shared_mem_space();
  constexpr int num_batches                      = 10;
  constexpr size_t num_base_input_rows_per_batch = 100;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                     cudf::aggregation::Kind::MAX,
                                                     cudf::aggregation::Kind::COUNT_ALL,
                                                     cudf::aggregation::Kind::SUM};

  auto input = create_batches_with_local_ungrouped_agg_result(
    num_batches, num_base_input_rows, column_types, aggregates, *mem_space);
  std::vector<std::optional<cudf::size_type>> merge_nth_index(aggregates.size(), std::nullopt);
  auto output_batch = gpu_merge_impl::merge_ungrouped_aggregate(
    input.batches, aggregates, merge_nth_index, cudf::get_default_stream(), *mem_space);
  validate_ungrouped_aggregate(input.batches, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate with invalid input", "[operator][merge_ungrouped_agg]")
{
  auto* mem_space                                = get_shared_mem_space();
  int num_batches                                = 1;
  constexpr size_t num_base_input_rows_per_batch = 100;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::SUM};

  // Invalid input: less than two input batches
  auto input = create_batches_with_local_ungrouped_agg_result(
    num_batches, num_base_input_rows, column_types, aggregates, *mem_space);
  std::vector<std::optional<cudf::size_type>> merge_nth_index(aggregates.size(), std::nullopt);
  REQUIRE_THROWS_AS(
    gpu_merge_impl::merge_ungrouped_aggregate(
      input.batches, aggregates, merge_nth_index, cudf::get_default_stream(), *mem_space),
    std::runtime_error);

  // Invalid input: mismatch between num columns and num aggregations
  num_batches         = 10;
  num_base_input_rows = std::vector<int>(num_batches, num_base_input_rows_per_batch);
  auto input2         = create_batches_with_local_ungrouped_agg_result(
    num_batches, num_base_input_rows, column_types, aggregates, *mem_space);
  aggregates.push_back(cudf::aggregation::Kind::SUM);
  merge_nth_index.push_back(std::nullopt);
  REQUIRE_THROWS_AS(
    gpu_merge_impl::merge_ungrouped_aggregate(
      input2.batches, aggregates, merge_nth_index, cudf::get_default_stream(), *mem_space),
    std::runtime_error);
}

TEST_CASE("Ungrouped merge aggregate with empty local aggregate results",
          "[operator][merge_ungrouped_agg]")
{
  auto* mem_space                                = get_shared_mem_space();
  constexpr int num_batches                      = 10;
  constexpr size_t num_base_input_rows_per_batch = 0;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                     cudf::aggregation::Kind::MAX,
                                                     cudf::aggregation::Kind::COUNT_ALL,
                                                     cudf::aggregation::Kind::SUM};

  auto input = create_batches_with_local_ungrouped_agg_result(
    num_batches, num_base_input_rows, column_types, aggregates, *mem_space);
  std::vector<std::optional<cudf::size_type>> merge_nth_index(aggregates.size(), std::nullopt);
  auto output_batch = gpu_merge_impl::merge_ungrouped_aggregate(
    input.batches, aggregates, merge_nth_index, cudf::get_default_stream(), *mem_space);
  validate_ungrouped_aggregate(input.batches, *output_batch, aggregates);
}

TEST_CASE("Ungrouped merge aggregate with mixed empty and non-empty local aggregate results",
          "[operator][merge_ungrouped_agg]")
{
  auto* mem_space           = get_shared_mem_space();
  constexpr int num_batches = 10;
  std::vector<int> num_base_input_rows;
  for (int i = 0; i < num_batches; ++i) {
    num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
  }
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                     cudf::aggregation::Kind::MAX,
                                                     cudf::aggregation::Kind::COUNT_ALL,
                                                     cudf::aggregation::Kind::SUM};

  auto input = create_batches_with_local_ungrouped_agg_result(
    num_batches, num_base_input_rows, column_types, aggregates, *mem_space);
  std::vector<std::optional<cudf::size_type>> merge_nth_index(aggregates.size(), std::nullopt);
  auto output_batch = gpu_merge_impl::merge_ungrouped_aggregate(
    input.batches, aggregates, merge_nth_index, cudf::get_default_stream(), *mem_space);
  validate_ungrouped_aggregate(input.batches, *output_batch, aggregates);
}

namespace {

batches_with_handles create_batches_with_local_grouped_agg_result(
  const int num_batches,
  const std::vector<int>& num_base_input_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  memory_space& mem_space)
{
  // Base input batches, make group key value ranges small
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);
  for (int group_col_id : group_idx) {
    ranges[group_col_id] = {0, 3};
  }
  auto base_input = create_batches_with_random_data(
    num_batches, num_base_input_rows, column_types, ranges, mem_space);

  // Compute local grouped aggregates
  batches_with_handles result;
  for (int i = 0; i < num_batches; ++i) {
    auto batch = gpu_aggregate_impl::local_grouped_aggregate(base_input.batches[i],
                                                             group_idx,
                                                             aggregates,
                                                             aggregate_idx,
                                                             {},
                                                             cudf::get_default_stream(),
                                                             mem_space);
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(mem_space.get_id());
    REQUIRE(lock_result.success);
    result.handles.emplace_back(std::move(lock_result.handle));
    result.batches.push_back(std::move(batch));
  }

  return result;
}

void copy_data_to_host(cudf::table_view table, std::vector<std::vector<int64_t>>& h_data)
{
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
          h_data[c].push_back(val);
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
          h_data[c].push_back(val);
        }
        break;
      }
      default:
        throw std::runtime_error("Unsupported cudf::data_type in `copy_data_to_host()`: " +
                                 std::to_string(static_cast<int>(col.type().id())));
    }
  }
}

void validate_grouped_aggregate(const std::vector<std::shared_ptr<data_batch>>& input_batches,
                                const data_batch& output,
                                int num_group_cols,
                                const std::vector<cudf::aggregation::Kind>& aggregates)
{
  std::vector<cudf::table_view> input_table_views;
  for (const auto& input_batch : input_batches) {
    input_table_views.push_back(sirius::get_cudf_table_view(*input_batch));
  }
  cudf::table_view output_table_view = sirius::get_cudf_table_view(output);

  // Compute expected results
  std::vector<std::vector<int64_t>> h_input_data(input_table_views[0].num_columns());
  for (const auto& table : input_table_views) {
    copy_data_to_host(table, h_input_data);
  }
  std::vector<std::map<std::vector<int64_t>, int64_t>> expected(aggregates.size());
  for (size_t r = 0; r < h_input_data[0].size(); ++r) {
    std::vector<int64_t> group_key;
    for (int c = 0; c < num_group_cols; ++c) {
      group_key.push_back(h_input_data[c][r]);
    }
    for (size_t i = 0; i < aggregates.size(); ++i) {
      int64_t val = h_input_data[num_group_cols + i][r];
      switch (aggregates[i]) {
        case cudf::aggregation::Kind::MIN: {
          if (!expected[i].contains(group_key)) {
            expected[i][group_key] = val;
          } else {
            expected[i][group_key] = std::min(expected[i][group_key], val);
          }
          break;
        }
        case cudf::aggregation::Kind::MAX: {
          if (!expected[i].contains(group_key)) {
            expected[i][group_key] = val;
          } else {
            expected[i][group_key] = std::max(expected[i][group_key], val);
          }
          break;
        }
        case cudf::aggregation::Kind::SUM:
        case cudf::aggregation::Kind::COUNT_ALL:
        case cudf::aggregation::Kind::COUNT_VALID: {
          expected[i][group_key] += val;
          break;
        }
        default: break;
      }
    }
  }

  // Get actual results
  std::vector<std::vector<int64_t>> actual(output_table_view.num_columns());
  copy_data_to_host(output_table_view, actual);

  // Check results
  REQUIRE(static_cast<size_t>(output_table_view.num_rows()) == expected[0].size());
  REQUIRE(static_cast<size_t>(output_table_view.num_columns()) ==
          static_cast<size_t>(num_group_cols) + aggregates.size());
  for (int r = 0; r < output_table_view.num_rows(); ++r) {
    std::vector<int64_t> group_key;
    for (int c = 0; c < num_group_cols; ++c) {
      group_key.push_back(actual[c][r]);
    }
    for (size_t i = 0; i < aggregates.size(); ++i) {
      int64_t actual_val   = actual[num_group_cols + i][r];
      int64_t expected_val = expected[i][group_key];
      REQUIRE(actual_val == expected_val);
    }
  }
}

}  // namespace

TEST_CASE("Grouped merge aggregate of min/max/count/sum", "[operator][merge_grouped_agg]")
{
  auto* mem_space                                = get_shared_mem_space();
  constexpr int num_batches                      = 10;
  constexpr size_t num_base_input_rows_per_batch = 100;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> group_idx                      = {0, 1};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                     cudf::aggregation::Kind::MAX,
                                                     cudf::aggregation::Kind::COUNT_ALL,
                                                     cudf::aggregation::Kind::SUM};
  std::vector<int> aggregate_idx                  = {2, 3, 4, 5};

  auto input        = create_batches_with_local_grouped_agg_result(num_batches,
                                                            num_base_input_rows,
                                                            column_types,
                                                            group_idx,
                                                            aggregates,
                                                            aggregate_idx,
                                                            *mem_space);
  auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
    input.batches, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space);
  validate_grouped_aggregate(input.batches, *output_batch, group_idx.size(), aggregates);
}

TEST_CASE("Grouped merge aggregate with invalid input", "[operator][merge_grouped_agg]")
{
  auto* mem_space                                = get_shared_mem_space();
  int num_batches                                = 1;
  constexpr size_t num_base_input_rows_per_batch = 100;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> group_idx                      = {0};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN};
  std::vector<int> aggregate_idx                  = {1};

  // Invalid input: less than two input batches
  auto input = create_batches_with_local_grouped_agg_result(num_batches,
                                                            num_base_input_rows,
                                                            column_types,
                                                            group_idx,
                                                            aggregates,
                                                            aggregate_idx,
                                                            *mem_space);
  REQUIRE_THROWS_AS(
    gpu_merge_impl::merge_grouped_aggregate(
      input.batches, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space),
    std::runtime_error);

  // Invalid input: mismatch between num columns, num_groups, and num aggregations
  num_batches         = 10;
  num_base_input_rows = std::vector<int>(num_batches, num_base_input_rows_per_batch);
  auto input2         = create_batches_with_local_ungrouped_agg_result(
    num_batches, num_base_input_rows, column_types, aggregates, *mem_space);
  group_idx.push_back(1);
  REQUIRE_THROWS_AS(
    gpu_merge_impl::merge_grouped_aggregate(
      input2.batches, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space),
    std::runtime_error);
}

TEST_CASE("Grouped merge aggregate with empty local aggregate results",
          "[operator][merge_grouped_agg]")
{
  auto* mem_space                                = get_shared_mem_space();
  constexpr int num_batches                      = 10;
  constexpr size_t num_base_input_rows_per_batch = 0;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> group_idx                      = {0, 1};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                     cudf::aggregation::Kind::MAX,
                                                     cudf::aggregation::Kind::COUNT_ALL,
                                                     cudf::aggregation::Kind::SUM};
  std::vector<int> aggregate_idx                  = {2, 3, 4, 5};

  auto input        = create_batches_with_local_grouped_agg_result(num_batches,
                                                            num_base_input_rows,
                                                            column_types,
                                                            group_idx,
                                                            aggregates,
                                                            aggregate_idx,
                                                            *mem_space);
  auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
    input.batches, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space);
  validate_grouped_aggregate(input.batches, *output_batch, group_idx.size(), aggregates);
}

TEST_CASE("Grouped merge aggregate with mixed empty and non-empty local aggregate results",
          "[operator][merge_grouped_agg]")
{
  auto* mem_space           = get_shared_mem_space();
  constexpr int num_batches = 10;
  std::vector<int> num_base_input_rows;
  for (int i = 0; i < num_batches; ++i) {
    num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
  }
  std::vector<cudf::data_type> column_types       = {cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64},
                                                     cudf::data_type{cudf::type_id::INT32},
                                                     cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> group_idx                      = {0, 1};
  std::vector<cudf::aggregation::Kind> aggregates = {cudf::aggregation::Kind::MIN,
                                                     cudf::aggregation::Kind::MAX,
                                                     cudf::aggregation::Kind::COUNT_ALL,
                                                     cudf::aggregation::Kind::SUM};
  std::vector<int> aggregate_idx                  = {2, 3, 4, 5};

  auto input        = create_batches_with_local_grouped_agg_result(num_batches,
                                                            num_base_input_rows,
                                                            column_types,
                                                            group_idx,
                                                            aggregates,
                                                            aggregate_idx,
                                                            *mem_space);
  auto output_batch = gpu_merge_impl::merge_grouped_aggregate(
    input.batches, group_idx.size(), aggregates, cudf::get_default_stream(), *mem_space);
  validate_grouped_aggregate(input.batches, *output_batch, group_idx.size(), aggregates);
}

namespace {

batches_with_handles create_batches_with_local_orderby_result(
  const int num_batches,
  const std::vector<int>& num_base_input_rows,
  const std::vector<cudf::data_type>& column_types,
  const std::vector<int>& order_key_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence,
  memory_space& mem_space)
{
  // Base input batches, make order key value ranges small
  std::vector<std::optional<std::pair<int, int>>> ranges(column_types.size(), std::nullopt);
  for (int idx : order_key_idx) {
    ranges[idx] = {0, 4};
  }
  auto base_input = create_batches_with_random_data(
    num_batches, num_base_input_rows, column_types, ranges, mem_space);

  // Compute local order_by
  batches_with_handles result;
  std::vector<int> projections(column_types.size());
  for (size_t i = 0; i < column_types.size(); ++i) {
    projections[i] = i;
  }
  for (int i = 0; i < num_batches; ++i) {
    auto batch = gpu_order_impl::local_order_by(base_input.batches[i],
                                                order_key_idx,
                                                column_order,
                                                null_precedence,
                                                projections,
                                                cudf::get_default_stream(),
                                                mem_space);
    REQUIRE(batch->try_to_create_task());
    auto lock_result = batch->try_to_lock_for_processing(mem_space.get_id());
    REQUIRE(lock_result.success);
    result.handles.emplace_back(std::move(lock_result.handle));
    result.batches.push_back(std::move(batch));
  }
  return result;
}

void validate_order_by(const std::vector<std::shared_ptr<data_batch>>& input_batches,
                       const data_batch& output,
                       const std::vector<int>& order_key_idx,
                       const std::vector<cudf::order>& column_order)
{
  std::vector<cudf::table_view> input_table_views;
  int expected_num_rows = 0;
  for (const auto& input_batch : input_batches) {
    input_table_views.push_back(sirius::get_cudf_table_view(*input_batch));
    expected_num_rows += input_table_views.back().num_rows();
  }
  cudf::table_view output_table_view = sirius::get_cudf_table_view(output);

  REQUIRE(output_table_view.num_rows() == expected_num_rows);
  REQUIRE(output_table_view.num_columns() == input_table_views[0].num_columns());
  for (int c = 0; c < output_table_view.num_columns(); ++c) {
    REQUIRE(output_table_view.column(c).type().id() == input_table_views[0].column(c).type().id());
  }

  std::vector<std::vector<int64_t>> actual(output_table_view.num_columns());
  copy_data_to_host(output_table_view, actual);
  auto comp = [&](int r) {
    for (size_t i = 0; i < order_key_idx.size(); ++i) {
      int col = order_key_idx[i];
      if (actual[col][r] == actual[col][r - 1]) { continue; }
      return (column_order[i] == cudf::order::ASCENDING && actual[col][r] > actual[col][r - 1]) ||
             (column_order[i] == cudf::order::DESCENDING && actual[col][r] < actual[col][r - 1]);
    }
    return true;
  };
  for (int r = 1; r < output_table_view.num_rows(); ++r) {
    REQUIRE(comp(r));
  }
}

}  // namespace

TEST_CASE("Merge order-by basic", "[operator][merge_order_by]")
{
  auto* mem_space                                = get_shared_mem_space();
  constexpr int num_batches                      = 10;
  constexpr size_t num_base_input_rows_per_batch = 100;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> order_key_idx            = {0, 1, 2};
  std::vector<cudf::order> column_order     = {
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence = {
    cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

  auto input        = create_batches_with_local_orderby_result(num_batches,
                                                        num_base_input_rows,
                                                        column_types,
                                                        order_key_idx,
                                                        column_order,
                                                        null_precedence,
                                                        *mem_space);
  auto output_batch = gpu_merge_impl::merge_order_by(input.batches,
                                                     order_key_idx,
                                                     column_order,
                                                     null_precedence,
                                                     cudf::get_default_stream(),
                                                     *mem_space);
  validate_order_by(input.batches, *output_batch, order_key_idx, column_order);
}

TEST_CASE("Merge order-by with invalid input", "[operator][merge_order_by]")
{
  auto* mem_space                                = get_shared_mem_space();
  int num_batches                                = 1;
  constexpr size_t num_base_input_rows_per_batch = 100;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> order_key_idx            = {0, 1, 2};
  std::vector<cudf::order> column_order     = {
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence = {
    cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

  // Invalid input: less than two input batches
  auto input = create_batches_with_local_orderby_result(num_batches,
                                                        num_base_input_rows,
                                                        column_types,
                                                        order_key_idx,
                                                        column_order,
                                                        null_precedence,
                                                        *mem_space);
  REQUIRE_THROWS_AS(gpu_merge_impl::merge_order_by(input.batches,
                                                   order_key_idx,
                                                   column_order,
                                                   null_precedence,
                                                   cudf::get_default_stream(),
                                                   *mem_space),
                    std::runtime_error);

  // Invalid input: mismatch between sizes of `order_key_idx`, `column_order`, and `null_precedence`
  num_batches         = 10;
  num_base_input_rows = std::vector<int>(num_batches, num_base_input_rows_per_batch);
  auto input2         = create_batches_with_local_orderby_result(num_batches,
                                                         num_base_input_rows,
                                                         column_types,
                                                         order_key_idx,
                                                         column_order,
                                                         null_precedence,
                                                         *mem_space);
  order_key_idx.push_back(3);
  REQUIRE_THROWS_AS(gpu_merge_impl::merge_order_by(input2.batches,
                                                   order_key_idx,
                                                   column_order,
                                                   null_precedence,
                                                   cudf::get_default_stream(),
                                                   *mem_space),
                    std::runtime_error);
}

TEST_CASE("Merge order-by with empty local order-by results", "[operator][merge_order_by]")
{
  auto* mem_space                                = get_shared_mem_space();
  constexpr int num_batches                      = 10;
  constexpr size_t num_base_input_rows_per_batch = 0;
  std::vector<int> num_base_input_rows(num_batches, num_base_input_rows_per_batch);
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> order_key_idx            = {0, 1, 2};
  std::vector<cudf::order> column_order     = {
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence = {
    cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

  auto input        = create_batches_with_local_orderby_result(num_batches,
                                                        num_base_input_rows,
                                                        column_types,
                                                        order_key_idx,
                                                        column_order,
                                                        null_precedence,
                                                        *mem_space);
  auto output_batch = gpu_merge_impl::merge_order_by(input.batches,
                                                     order_key_idx,
                                                     column_order,
                                                     null_precedence,
                                                     cudf::get_default_stream(),
                                                     *mem_space);
  validate_order_by(input.batches, *output_batch, order_key_idx, column_order);
}

TEST_CASE("Merge order-by with mixed empty and non-empty local order-by results",
          "[operator][merge_order_by]")
{
  auto* mem_space           = get_shared_mem_space();
  constexpr int num_batches = 10;
  std::vector<int> num_base_input_rows;
  for (int i = 0; i < num_batches; ++i) {
    num_base_input_rows.push_back(i % 2 == 1 ? 0 : (i + 1) * 10);
  }
  std::vector<cudf::data_type> column_types = {cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64},
                                               cudf::data_type{cudf::type_id::INT32},
                                               cudf::data_type{cudf::type_id::INT64}};
  std::vector<int> order_key_idx            = {0, 1, 2};
  std::vector<cudf::order> column_order     = {
    cudf::order::ASCENDING, cudf::order::DESCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence = {
    cudf::null_order::AFTER, cudf::null_order::BEFORE, cudf::null_order::AFTER};

  auto input        = create_batches_with_local_orderby_result(num_batches,
                                                        num_base_input_rows,
                                                        column_types,
                                                        order_key_idx,
                                                        column_order,
                                                        null_precedence,
                                                        *mem_space);
  auto output_batch = gpu_merge_impl::merge_order_by(input.batches,
                                                     order_key_idx,
                                                     column_order,
                                                     null_precedence,
                                                     cudf::get_default_stream(),
                                                     *mem_space);
  validate_order_by(input.batches, *output_batch, order_key_idx, column_order);
}
