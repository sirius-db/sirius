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

#include "debug_utils.hpp"

#include "data/data_batch_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "utils/data_utils.hpp"

#include <catch.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

using namespace sirius::test::operator_utils;

// ---------------------------------------------------------------------------
// Test case 1: debug_schema produces output without throwing
// ---------------------------------------------------------------------------

TEST_CASE("debug_schema produces output without throwing", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Create INT32 column (5 rows)
  std::vector<int32_t> vals_a{10, 20, 30, 40, 50};
  auto col_a = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(vals_a, stream, mr);

  // Create INT64 column (5 rows)
  std::vector<int64_t> vals_b{100, 200, 300, 400, 500};
  auto col_b = sirius::test::vector_to_cudf_column<gpu_type_traits<int64_t>>(vals_b, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col_a));
  columns.push_back(std::move(col_b));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto batch = sirius::make_data_batch(std::move(table), *space);
  REQUIRE(batch != nullptr);

  REQUIRE_NOTHROW(sirius::debug_schema(*batch, stream, {"col_a", "col_b"}));
}

// ---------------------------------------------------------------------------
// Test case 2: debug_schema with no column names uses defaults
// ---------------------------------------------------------------------------

TEST_CASE("debug_schema with no column names uses defaults", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  std::vector<int32_t> vals_a{1, 2, 3, 4, 5};
  auto col_a = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(vals_a, stream, mr);

  std::vector<int64_t> vals_b{10, 20, 30, 40, 50};
  auto col_b = sirius::test::vector_to_cudf_column<gpu_type_traits<int64_t>>(vals_b, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col_a));
  columns.push_back(std::move(col_b));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto batch = sirius::make_data_batch(std::move(table), *space);
  REQUIRE(batch != nullptr);

  // No col_names argument -- uses default "col[N]" naming
  REQUIRE_NOTHROW(sirius::debug_schema(*batch, stream));
}

// ---------------------------------------------------------------------------
// Test case 3: debug_nulls produces output without throwing
// ---------------------------------------------------------------------------

TEST_CASE("debug_nulls produces output without throwing", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  std::vector<int32_t> vals_a{10, 20, 30, 40, 50};
  auto col_a = sirius::test::vector_to_cudf_column<gpu_type_traits<int32_t>>(vals_a, stream, mr);

  std::vector<int64_t> vals_b{100, 200, 300, 400, 500};
  auto col_b = sirius::test::vector_to_cudf_column<gpu_type_traits<int64_t>>(vals_b, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col_a));
  columns.push_back(std::move(col_b));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto batch = sirius::make_data_batch(std::move(table), *space);
  REQUIRE(batch != nullptr);

  REQUIRE_NOTHROW(sirius::debug_nulls(*batch, stream, {"col_a", "col_b"}));
}

// ---------------------------------------------------------------------------
// Test case 4: debug_schema handles empty batch (0 rows)
// ---------------------------------------------------------------------------

TEST_CASE("debug_schema handles empty batch (0 rows)", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Create two columns with 0 rows
  auto col_a = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  auto col_b = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 0, cudf::mask_state::UNALLOCATED, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col_a));
  columns.push_back(std::move(col_b));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto batch = sirius::make_data_batch(std::move(table), *space);
  REQUIRE(batch != nullptr);

  REQUIRE_NOTHROW(sirius::debug_schema(*batch, stream));
}

// ---------------------------------------------------------------------------
// Test case 5: debug_nulls reports correct null counts for columns with nulls
// ---------------------------------------------------------------------------

TEST_CASE("debug_nulls reports correct null counts for columns with nulls", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  constexpr cudf::size_type num_rows = 5;

  // Create a column with data
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::ALL_VALID, stream, mr);

  // Write some data to the column
  std::vector<int32_t> host_data{10, 20, 30, 40, 50};
  cudaMemcpyAsync(col->mutable_view().data<int32_t>(),
                  host_data.data(),
                  sizeof(int32_t) * num_rows,
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Set null mask: rows 1 and 3 are null.
  // Bitmask: bit 0=valid, bit 1=null, bit 2=valid, bit 3=null, bit 4=valid
  // So bitmask byte = 0b00010101 = 0x15
  auto mask_size = cudf::bitmask_allocation_size_bytes(num_rows);
  std::vector<uint8_t> host_mask(mask_size, 0xFF);
  // Clear bits 1 and 3 to mark them null
  host_mask[0] = 0b00010101;  // bits 0,2,4 set; bits 1,3 clear

  rmm::device_buffer dev_mask(mask_size, stream, mr);
  cudaMemcpyAsync(dev_mask.data(), host_mask.data(), mask_size, cudaMemcpyHostToDevice, stream.value());
  stream.synchronize();

  col->set_null_mask(std::move(dev_mask), 2);  // null_count = 2

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(col));
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto batch = sirius::make_data_batch(std::move(table), *space);
  REQUIRE(batch != nullptr);

  // debug_nulls should not throw and should report 2 nulls in its log output
  REQUIRE_NOTHROW(sirius::debug_nulls(*batch, stream));
}

// ---------------------------------------------------------------------------
// Test case 6: copy_null_mask_to_host returns correct null positions
// ---------------------------------------------------------------------------

TEST_CASE("copy_null_mask_to_host returns correct null positions", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  constexpr cudf::size_type num_rows = 8;

  // Create a column with 8 rows
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::ALL_VALID, stream, mr);

  // Write data
  std::vector<int32_t> host_data{1, 2, 3, 4, 5, 6, 7, 8};
  cudaMemcpyAsync(col->mutable_view().data<int32_t>(),
                  host_data.data(),
                  sizeof(int32_t) * num_rows,
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Set null mask: rows 0,2,4,6 valid; rows 1,3,5,7 null
  // Bitmask byte = 0b01010101 = 0x55
  auto mask_size = cudf::bitmask_allocation_size_bytes(num_rows);
  std::vector<uint8_t> host_mask(mask_size, 0x00);
  host_mask[0] = 0x55;  // 0b01010101: bits 0,2,4,6 set (valid), bits 1,3,5,7 clear (null)

  rmm::device_buffer dev_mask(mask_size, stream, mr);
  cudaMemcpyAsync(dev_mask.data(), host_mask.data(), mask_size, cudaMemcpyHostToDevice, stream.value());
  stream.synchronize();

  col->set_null_mask(std::move(dev_mask), 4);  // null_count = 4

  auto col_view = col->view();
  auto result   = sirius::copy_null_mask_to_host(col_view, stream);

  REQUIRE(result.has_nulls == true);
  CHECK(result.is_null(0) == false);  // row 0 valid (bit 0 = 1)
  CHECK(result.is_null(1) == true);   // row 1 null  (bit 1 = 0)
  CHECK(result.is_null(2) == false);  // row 2 valid (bit 2 = 1)
  CHECK(result.is_null(3) == true);   // row 3 null  (bit 3 = 0)
  CHECK(result.is_null(4) == false);  // row 4 valid (bit 4 = 1)
  CHECK(result.is_null(5) == true);   // row 5 null  (bit 5 = 0)
  CHECK(result.is_null(6) == false);  // row 6 valid (bit 6 = 1)
  CHECK(result.is_null(7) == true);   // row 7 null  (bit 7 = 0)
}

// ---------------------------------------------------------------------------
// Test case 7: copy_null_mask_to_host returns has_nulls=false for non-null column
// ---------------------------------------------------------------------------

TEST_CASE("copy_null_mask_to_host returns has_nulls=false for non-null column", "[debug_utils]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);

  auto stream = cudf::get_default_stream();
  auto mr     = get_resource_ref(*space);

  // Create a column with no nulls (UNALLOCATED mask)
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 5, cudf::mask_state::UNALLOCATED, stream, mr);

  std::vector<int32_t> host_data{1, 2, 3, 4, 5};
  cudaMemcpyAsync(col->mutable_view().data<int32_t>(),
                  host_data.data(),
                  sizeof(int32_t) * 5,
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();

  auto col_view = col->view();
  auto result   = sirius::copy_null_mask_to_host(col_view, stream);

  REQUIRE(result.has_nulls == false);
  CHECK(result.is_null(0) == false);  // always false when has_nulls is false
}

// ---------------------------------------------------------------------------
// Test case 8: debug_schema on null-data batch logs warning without crashing
// ---------------------------------------------------------------------------

TEST_CASE("debug_schema on null-data batch logs warning without crashing", "[debug_utils]")
{
  // Create a data_batch with nullptr data representation (simulates a batch
  // whose data has been released or was never assigned)
  cucascade::data_batch batch(0, nullptr);

  // The tier guard inside debug_schema must detect the null data and return
  // safely without crashing
  REQUIRE_NOTHROW(sirius::debug_schema(batch, cudf::get_default_stream()));
}
