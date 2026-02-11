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

#pragma once

// test
#include "catch.hpp"

// sirius
#include "memory/sirius_memory_reservation_manager.hpp"

// data representations
#include <data/data_batch_utils.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/sirius_converter_registry.hpp>
#include <helper/helper.hpp>

// cudf
#include <cudf/strings/strings_column_view.hpp>

// duckdb
#include <duckdb.hpp>

// standard library
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace cucascade::memory;

/**
 * @brief Initialize the memory reservation manager for tests.
 *
 * Sets up GPU, HOST, and DISK memory tiers with test-appropriate sizes.
 * Uses static initialization to avoid reinitializing for every test (which is slow).
 * Only initializes once per test run.
 *
 */
inline std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> initialize_memory_manager(
  std::size_t n_gpus = 1)
{
  // Reset converter registry to avoid cross-test leakage
  sirius::converter_registry::reset_for_testing();

  reservation_manager_configurator builder;

  // Configure GPU (2GB limit, 75% reservation ratio)
  const size_t gpu_capacity  = 2ull << 30;  // 2GB
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 4ull << 30;  // 4GB

  builder.set_number_of_gpus(n_gpus)
    .set_gpu_usage_limit(gpu_capacity / n_gpus)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity / n_gpus)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  // Initialize converters used by data representations
  sirius::converter_registry::initialize();
  return manager;
}

namespace sirius::scan_test_utils {

inline std::filesystem::path get_test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path() / "memory.cfg";
}

inline cucascade::memory::memory_space* get_space(
  cucascade::memory::memory_reservation_manager& mem_mgr, cucascade::memory::Tier tier)
{
  auto* space = mem_mgr.get_memory_space(tier, 0);
  if (space) { return space; }
  auto spaces = mem_mgr.get_memory_spaces_for_tier(tier);
  if (!spaces.empty()) { return const_cast<cucascade::memory::memory_space*>(spaces.front()); }
  return nullptr;
}

/**
 * @brief Create a simple synthetic table with multiple columns and rows.
 */
inline void create_synthetic_table(duckdb::Connection& con,
                                   std::string const& table_name,
                                   size_t num_rows)
{
  // clang-format off
  std::string create_sql = "CREATE TABLE " + table_name + " ("
                           "id INTEGER, "
                           "value BIGINT, "
                           "price DOUBLE, "
                           "name VARCHAR"
                           ");";
  // clang-format on
  auto result = con.Query(create_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  constexpr size_t BATCH_SIZE = 1000;
  for (size_t start = 0; start < num_rows; start += BATCH_SIZE) {
    size_t end             = std::min(start + BATCH_SIZE, num_rows);
    std::string insert_sql = "INSERT INTO " + table_name + " VALUES ";

    for (size_t i = start; i < end; ++i) {
      if (i > start) { insert_sql += ", "; }
      auto id          = static_cast<int32_t>(i);
      auto value       = static_cast<int64_t>(i * 100);
      auto price       = static_cast<double>(i) * 1.5;
      std::string name = "item_" + std::to_string(i);
      insert_sql += "(" + std::to_string(id) + ", " + std::to_string(value) + ", " +
                    std::to_string(price) + ", " + "'" + name + "')";
    }

    result = con.Query(insert_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());
  }
}

/**
 * @brief Drain all batches from a shared data repository.
 */
inline std::vector<std::shared_ptr<cucascade::data_batch>> drain_data_repo(
  cucascade::shared_data_repository& data_repo)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  while (true) {
    auto batch = data_repo.pop_data_batch(cucascade::batch_state::task_created);
    if (!batch) { break; }
    batches.push_back(std::move(batch));
  }
  return batches;
}

inline std::vector<int64_t> copy_string_offsets(const cudf::column_view& offsets_col)
{
  auto num_offsets = offsets_col.size();
  std::vector<int64_t> offsets(num_offsets, 0);
  if (num_offsets == 0) { return offsets; }
  if (offsets_col.type().id() == cudf::type_id::INT64) {
    cudaMemcpy(offsets.data(),
               offsets_col.data<int64_t>(),
               sizeof(int64_t) * offsets.size(),
               cudaMemcpyDeviceToHost);
  } else if (offsets_col.type().id() == cudf::type_id::INT32) {
    std::vector<cudf::size_type> offsets32(num_offsets, 0);
    cudaMemcpy(offsets32.data(),
               offsets_col.data<cudf::size_type>(),
               sizeof(cudf::size_type) * offsets32.size(),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < offsets.size(); ++i) {
      offsets[i] = offsets32[i];
    }
  } else {
    FAIL("Unsupported offsets type in string column");
  }
  return offsets;
}

inline void validate_scanned_batches(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  size_t expected_rows,
  cucascade::memory::memory_reservation_manager& mem_mgr)
{
  auto* gpu_space = get_space(mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  auto& registry = sirius::converter_registry::get();

  if (expected_rows > 0) { REQUIRE_FALSE(batches.empty()); }

  std::vector<bool> seen(expected_rows, false);
  size_t total_rows = 0;

  for (auto const& batch : batches) {
    REQUIRE(batch != nullptr);
    batch->convert_to<cucascade::gpu_table_representation>(
      registry, gpu_space, cudf::get_default_stream());
    auto table_view = sirius::get_cudf_table_view(*batch);

    REQUIRE(table_view.num_columns() == 4);
    REQUIRE(table_view.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(table_view.column(1).type().id() == cudf::type_id::INT64);
    REQUIRE(table_view.column(2).type().id() == cudf::type_id::FLOAT64);
    REQUIRE(table_view.column(3).type().id() == cudf::type_id::STRING);

    auto const num_rows = table_view.num_rows();
    total_rows += num_rows;
    if (num_rows == 0) { continue; }

    std::vector<int32_t> ids(num_rows);
    std::vector<int64_t> values(num_rows);
    std::vector<double> prices(num_rows);
    cudaMemcpy(ids.data(),
               table_view.column(0).data<int32_t>(),
               sizeof(int32_t) * num_rows,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(values.data(),
               table_view.column(1).data<int64_t>(),
               sizeof(int64_t) * num_rows,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(prices.data(),
               table_view.column(2).data<double>(),
               sizeof(double) * num_rows,
               cudaMemcpyDeviceToHost);

    cudf::strings_column_view name_col(table_view.column(3));
    auto offsets = copy_string_offsets(name_col.offsets());
    REQUIRE(offsets.size() == static_cast<size_t>(num_rows + 1));
    std::vector<char> chars;
    if (!offsets.empty() && offsets.back() > 0) {
      chars.resize(static_cast<size_t>(offsets.back()));
      cudaMemcpy(chars.data(),
                 name_col.chars_begin(cudf::get_default_stream()),
                 chars.size(),
                 cudaMemcpyDeviceToHost);
    }

    for (cudf::size_type i = 0; i < num_rows; ++i) {
      auto id = ids[i];
      REQUIRE(id >= 0);
      REQUIRE(static_cast<size_t>(id) < expected_rows);
      REQUIRE_FALSE(seen[id]);
      seen[id] = true;

      auto const expected_value = static_cast<int64_t>(id) * 100;
      auto const expected_price = static_cast<double>(id) * 1.5;
      auto const expected_name  = "item_" + std::to_string(id);

      REQUIRE(values[i] == expected_value);
      REQUIRE(prices[i] == expected_price);

      auto const start = static_cast<size_t>(offsets[i]);
      auto const end   = static_cast<size_t>(offsets[i + 1]);
      std::string actual_name;
      if (end > start) { actual_name.assign(chars.data() + start, chars.data() + end); }
      REQUIRE(actual_name == expected_name);
    }
  }

  REQUIRE(total_rows == expected_rows);
  for (auto const& was_seen : seen) {
    REQUIRE(was_seen);
  }
}

inline void validate_projected_id_price_batches(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  size_t expected_rows,
  cucascade::memory::memory_reservation_manager& mem_mgr)
{
  auto* gpu_space = get_space(mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  auto& registry = sirius::converter_registry::get();

  if (expected_rows > 0) { REQUIRE_FALSE(batches.empty()); }

  std::vector<bool> seen(expected_rows, false);
  size_t total_rows = 0;

  for (auto const& batch : batches) {
    REQUIRE(batch != nullptr);
    batch->convert_to<cucascade::gpu_table_representation>(
      registry, gpu_space, cudf::get_default_stream());
    auto table_view = sirius::get_cudf_table_view(*batch);

    REQUIRE(table_view.num_columns() == 2);
    REQUIRE(table_view.column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(table_view.column(1).type().id() == cudf::type_id::FLOAT64);

    auto const num_rows = table_view.num_rows();
    total_rows += num_rows;
    if (num_rows == 0) { continue; }

    std::vector<int32_t> ids(num_rows);
    std::vector<double> prices(num_rows);
    cudaMemcpy(ids.data(),
               table_view.column(0).data<int32_t>(),
               sizeof(int32_t) * num_rows,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(prices.data(),
               table_view.column(1).data<double>(),
               sizeof(double) * num_rows,
               cudaMemcpyDeviceToHost);

    for (cudf::size_type i = 0; i < num_rows; ++i) {
      auto id = ids[i];
      REQUIRE(id >= 0);
      REQUIRE(static_cast<size_t>(id) < expected_rows);
      REQUIRE_FALSE(seen[id]);
      seen[id] = true;

      auto const expected_price = static_cast<double>(id) * 1.5;
      REQUIRE(prices[i] == expected_price);
    }
  }

  REQUIRE(total_rows == expected_rows);
  for (auto const& was_seen : seen) {
    REQUIRE(was_seen);
  }
}

}  // namespace sirius::scan_test_utils
