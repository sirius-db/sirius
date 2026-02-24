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

// test
#include <catch.hpp>
#include <utils/utils.hpp>

// sirius
#include <helper/utils.hpp>
#include <op/scan/duckdb_scan_task.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>

// standard library
#include <climits>
#include <filesystem>
#include <numbers>

using namespace sirius::op::scan;
using namespace cucascade::memory;

//===----------------------------------------------------------------------===//
// Helper Functions: Sirius Context
//===----------------------------------------------------------------------===//

static std::filesystem::path get_test_config_path()
{
  return std::filesystem::path(__FILE__).parent_path() / "memory.cfg";
}

static memory_space* get_host_space(duckdb::SiriusContext& sirius_ctx)
{
  auto& mem_mgr = sirius_ctx.get_memory_manager();
  auto* space   = mem_mgr.get_memory_space(Tier::HOST, 0);
  if (space) { return space; }
  auto spaces = mem_mgr.get_memory_spaces_for_tier(Tier::HOST);
  if (!spaces.empty()) { return const_cast<memory_space*>(spaces.front()); }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helper Function: Create Allocation for Tests
//===----------------------------------------------------------------------===//

/**
 * @brief Helper to create a multi-block allocation for testing
 *
 * @param total_size Total size in bytes needed for the allocation
 * @return unique_ptr to the allocation
 */
static std::unique_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>
create_test_allocation(size_t total_size)
{
  auto sirius_ctx = sirius::get_sirius_context(get_test_config_path());
  auto* mem_space = get_host_space(*sirius_ctx);
  REQUIRE(mem_space != nullptr);
  auto* allocator = mem_space->template get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(allocator != nullptr);

  // Use the public allocate_multiple_blocks method instead of manually creating blocks
  return allocator->allocate_multiple_blocks(total_size);
}

//===----------------------------------------------------------------------===//
// Test: column_builder - Construction
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - construction", "[duckdb_scan_task][column_builder]")
{
  constexpr size_t DEFAULT_VARCHAR_SIZE = 256;

  SECTION("construct with INTEGER type")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, DEFAULT_VARCHAR_SIZE);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::INTEGER);
    REQUIRE(builder.type_size == sizeof(int32_t));
    REQUIRE(builder.total_data_bytes == 0);
    REQUIRE(builder.null_count == 0);
  }

  SECTION("construct with BIGINT type")
  {
    auto bigint_type = duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
    duckdb_scan_task_local_state::column_builder builder(bigint_type, DEFAULT_VARCHAR_SIZE);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::BIGINT);
    REQUIRE(builder.type_size == sizeof(int64_t));
    REQUIRE(builder.total_data_bytes == 0);
  }

  SECTION("construct with VARCHAR type")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, DEFAULT_VARCHAR_SIZE);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::VARCHAR);
    REQUIRE(builder.total_data_bytes == 0);
    REQUIRE(builder.type_size ==
            DEFAULT_VARCHAR_SIZE);  // For VARCHAR, type_size is the default size
  }

  SECTION("construct with DOUBLE type")
  {
    auto double_type = duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
    duckdb_scan_task_local_state::column_builder builder(double_type, DEFAULT_VARCHAR_SIZE);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::DOUBLE);
    REQUIRE(builder.type_size == sizeof(double));
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - Accessor Initialization with Shared Allocation
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - accessor initialization", "[duckdb_scan_task][column_builder]")
{
  constexpr size_t DEFAULT_VARCHAR_SIZE = 256;

  auto sirius_ctx = sirius::get_sirius_context(get_test_config_path());
  auto* mem_space = get_host_space(*sirius_ctx);
  REQUIRE(mem_space != nullptr);
  auto* allocator = mem_space->template get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(allocator != nullptr);

  SECTION("initialize accessors for fixed-width type")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, DEFAULT_VARCHAR_SIZE);

    size_t num_rows = 100;
    // Calculate total size needed: data + mask
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);

    auto allocation = create_test_allocation(total_size);

    // Initialize the accessors at byte offset 0
    builder.initialize_accessors(num_rows, 0, allocation);

    // Verify accessors were initialized correctly
    REQUIRE(builder.data_blocks_accessor.block_index == 0);
    REQUIRE(builder.data_blocks_accessor.offset_in_block == 0);
    REQUIRE(builder.mask_blocks_accessor.block_index == 0);
    // Mask starts after data
    size_t expected_mask_offset = sizeof(int32_t) * num_rows;
    REQUIRE(builder.mask_blocks_accessor.offset_in_block ==
            expected_mask_offset % allocator->get_block_size());
  }

  SECTION("initialize accessors for VARCHAR type")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, DEFAULT_VARCHAR_SIZE);

    size_t num_rows = 100;
    // Calculate total size needed: offsets + data + mask
    size_t total_size = sizeof(int64_t) * (num_rows + 1) +    // offsets
                        DEFAULT_VARCHAR_SIZE * num_rows +     // data
                        sirius::utils::ceil_div_8(num_rows);  // mask

    // Use the helper to create the allocation
    auto allocation = create_test_allocation(total_size);

    // Initialize the accessors at byte offset 0
    builder.initialize_accessors(num_rows, 0, allocation);

    // Verify accessors were initialized correctly
    REQUIRE(builder.offset_blocks_accessor.block_index == 0);
    REQUIRE(builder.offset_blocks_accessor.offset_in_block == 0);
    REQUIRE(builder.total_data_bytes_allocated == DEFAULT_VARCHAR_SIZE * num_rows);

    // Verify offset is initialized to zero
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - sufficient_space_for_column
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - sufficient_space_for_column", "[duckdb_scan_task][column_builder]")
{
  constexpr size_t DEFAULT_VARCHAR_SIZE = 256;

  auto sirius_ctx = sirius::get_sirius_context(get_test_config_path());
  auto* mem_space = get_host_space(*sirius_ctx);
  REQUIRE(mem_space != nullptr);
  auto* allocator = mem_space->template get_memory_resource_as<fixed_size_host_memory_resource>();
  REQUIRE(allocator != nullptr);

  SECTION("VARCHAR type space check - sufficient space")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, DEFAULT_VARCHAR_SIZE);

    // Allocate space for 100 rows with default VARCHAR size
    size_t num_rows   = 100;
    size_t total_size = sizeof(int64_t) * (num_rows + 1) +    // offsets
                        DEFAULT_VARCHAR_SIZE * num_rows +     // data
                        sirius::utils::ceil_div_8(num_rows);  // mask

    // Use the helper to create the allocation
    auto allocation = create_test_allocation(total_size);

    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with string data that fits
    duckdb::Vector vec(varchar_type, 10);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());

    // Fill with small strings (total < DEFAULT_VARCHAR_SIZE * 10)
    for (size_t i = 0; i < 10; ++i) {
      str_data[i] = duckdb::string_t("test");  // 4 bytes each = 40 bytes total
    }

    duckdb::ValidityMask validity(10);
    validity.Initialize(10);
    validity.SetAllValid(10);

    // Should have sufficient space
    REQUIRE(builder.sufficient_space_for_column(vec, validity, 10) == true);
  }

  SECTION("VARCHAR type space check - insufficient space")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, 10);  // Small default size

    // Allocate space for 10 rows with small VARCHAR size (10 bytes per row)
    size_t num_rows   = 10;
    size_t total_size = sizeof(int64_t) * (num_rows + 1) +    // offsets
                        10 * num_rows +                       // data (10 bytes per row)
                        sirius::utils::ceil_div_8(num_rows);  // mask

    // Use the helper to create the allocation
    auto allocation = create_test_allocation(total_size);

    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with strings that are too large
    duckdb::Vector vec(varchar_type, 5);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());

    // Fill with large strings (total > 10 * 5 = 50 bytes allocated)
    for (size_t i = 0; i < 5; ++i) {
      str_data[i] =
        duckdb::string_t("this_is_a_very_long_string");  // 26 bytes each = 130 bytes total
    }

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);

    // Should NOT have sufficient space
    REQUIRE(builder.sufficient_space_for_column(vec, validity, 5) == false);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - process_mask_for_column
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - process_mask_for_column", "[duckdb_scan_task][column_builder]")
{
  SECTION("byte-aligned mask processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a validity mask with some NULLs
    duckdb::ValidityMask validity(16);
    validity.Initialize(16);   // Must initialize to allocate backing storage
    validity.SetAllValid(16);  // Start with all valid
    validity.SetInvalid(3);    // Row 3 is NULL
    validity.SetInvalid(7);    // Row 7 is NULL
    validity.SetInvalid(15);   // Row 15 is NULL

    // Process the mask (byte-aligned: row_offset = 0)
    builder.process_mask_for_column(validity, 16, 0, allocation);

    // Verify the mask was copied correctly
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current(allocation);
    builder.mask_blocks_accessor.advance();
    auto mask_byte_1 = builder.mask_blocks_accessor.get_current(allocation);

    // Bit 3 and 7 should be 0 (invalid) in first byte
    REQUIRE((mask_byte_0 & (1 << 3)) == 0);
    REQUIRE((mask_byte_0 & (1 << 7)) == 0);

    // Bit 7 (15 - 8) should be 0 in second byte
    REQUIRE((mask_byte_1 & (1 << 7)) == 0);
  }

  SECTION("byte-unaligned mask processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // First, add 3 rows (to create a 3-bit offset)
    duckdb::ValidityMask validity1(3);
    validity1.Initialize(3);
    validity1.SetAllValid(3);
    builder.process_mask_for_column(validity1, 3, 0, allocation);

    // Now add 8 more rows starting at bit offset 3 (byte-unaligned)
    duckdb::ValidityMask validity2(8);
    validity2.Initialize(8);
    validity2.SetAllValid(8);
    validity2.SetInvalid(2);  // Should be at bit position 5 (3 + 2) overall
    builder.process_mask_for_column(validity2, 8, 3, allocation);

    // Verify the unaligned write worked
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current(allocation);

    // Bit 5 should be 0 (invalid)
    REQUIRE((mask_byte_0 & (1 << 5)) == 0);
  }

  SECTION("null_count reflects invalid rows in validity mask")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Initially, null_count should be 0
    REQUIRE(builder.null_count == 0);

    // Create a validity mask with actual NULLs
    duckdb::ValidityMask validity_with_nulls(10);
    validity_with_nulls.Initialize(10);
    validity_with_nulls.SetAllValid(10);
    validity_with_nulls.SetInvalid(5);  // Row 5 is NULL

    // Process the mask - null_count should reflect the number of invalid rows
    builder.process_mask_for_column(validity_with_nulls, 10, 0, allocation);
    REQUIRE(builder.null_count == 1);
  }

  SECTION("null_count remains zero when all rows are valid (mask pointer not null)")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Initially, null_count should be 0
    REQUIRE(builder.null_count == 0);

    // Create a validity mask with all valid rows but initialized (GetData() != nullptr)
    duckdb::ValidityMask validity_all_valid(10);
    validity_all_valid.Initialize(10);
    validity_all_valid.SetAllValid(10);

    // Process the mask - null_count should remain 0 even though mask pointer is not null
    builder.process_mask_for_column(validity_all_valid, 10, 0, allocation);
    REQUIRE(builder.null_count == 0);
  }

  SECTION("null_count remains zero when validity mask data pointer is null")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Initially, null_count should be 0
    REQUIRE(builder.null_count == 0);

    // Create a validity mask without initializing (GetData() will return nullptr)
    duckdb::ValidityMask validity_null_ptr;

    // Process the mask - null_count should remain 0 because mask pointer is null
    builder.process_mask_for_column(validity_null_ptr, 10, 0, allocation);
    REQUIRE(builder.null_count == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - process_column (fixed-width)
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - process_column for fixed-width types",
          "[duckdb_scan_task][column_builder]")
{
  SECTION("INTEGER column processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with integer data
    duckdb::Vector vec(int_type, 10);
    auto* data = reinterpret_cast<int32_t*>(vec.GetData());
    for (size_t i = 0; i < 10; ++i) {
      data[i] = static_cast<int32_t>(i * 100);
    }

    duckdb::ValidityMask validity(10);
    validity.Initialize(10);
    validity.SetAllValid(10);
    validity.SetInvalid(5);  // Row 5 is NULL

    // Process the column
    builder.process_column(vec, validity, 10, 0, allocation);

    // Verify data was copied
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 10);

    // Verify the data values
    builder.data_blocks_accessor.set_cursor(0);
    for (size_t i = 0; i < 10; ++i) {
      int32_t value;
      std::memcpy(&value,
                  reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]) + i * sizeof(int32_t),
                  sizeof(int32_t));
      REQUIRE(value == static_cast<int32_t>(i * 100));
    }
  }

  SECTION("BIGINT column processing")
  {
    auto bigint_type = duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
    duckdb_scan_task_local_state::column_builder builder(bigint_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int64_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with bigint data
    duckdb::Vector vec(bigint_type, 5);
    auto* data = reinterpret_cast<int64_t*>(vec.GetData());
    for (size_t i = 0; i < 5; ++i) {
      data[i] = static_cast<int64_t>(i * 1000000LL);
    }

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);

    // Process the column
    builder.process_column(vec, validity, 5, 0, allocation);

    // Verify data was copied
    REQUIRE(builder.total_data_bytes == sizeof(int64_t) * 5);

    // Verify the data values
    for (size_t i = 0; i < 5; ++i) {
      int64_t value;
      std::memcpy(&value,
                  reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]) + i * sizeof(int64_t),
                  sizeof(int64_t));
      REQUIRE(value == static_cast<int64_t>(i * 1000000LL));
    }
  }

  SECTION("DOUBLE column processing")
  {
    auto double_type = duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
    duckdb_scan_task_local_state::column_builder builder(double_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(double) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with double data
    duckdb::Vector vec(double_type, 5);
    auto* data = reinterpret_cast<double*>(vec.GetData());
    for (size_t i = 0; i < 5; ++i) {
      data[i] = i * std::numbers::pi;
    }

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);
    validity.SetInvalid(2);  // Row 2 is NULL

    // Process the column
    builder.process_column(vec, validity, 5, 0, allocation);

    // Verify data was copied
    REQUIRE(builder.total_data_bytes == sizeof(double) * 5);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - process_column (VARCHAR)
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - process_column for VARCHAR", "[duckdb_scan_task][column_builder]")
{
  constexpr size_t DEFAULT_VARCHAR_SIZE = 256;

  SECTION("VARCHAR column processing with all valid rows")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, DEFAULT_VARCHAR_SIZE);

    size_t num_rows   = 10;
    size_t total_size = sizeof(int64_t) * (num_rows + 1) +    // offsets
                        DEFAULT_VARCHAR_SIZE * num_rows +     // data
                        sirius::utils::ceil_div_8(num_rows);  // mask
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with string data
    duckdb::Vector vec(varchar_type, 5);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    str_data[0]    = duckdb::string_t("hello");
    str_data[1]    = duckdb::string_t("world");
    str_data[2]    = duckdb::string_t("foo");
    str_data[3]    = duckdb::string_t("bar");
    str_data[4]    = duckdb::string_t("baz");

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);

    // Process the column
    builder.process_column(vec, validity, 5, 0, allocation);

    // Verify total data bytes (5 + 5 + 3 + 3 + 3 = 19 bytes)
    REQUIRE(builder.total_data_bytes == 19);

    // Verify offsets were set correctly
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);  // Initial offset
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 5);  // After "hello"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 10);  // After "world"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 13);  // After "foo"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 16);  // After "bar"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 19);  // After "baz"
  }

  SECTION("VARCHAR column processing with NULL values")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, DEFAULT_VARCHAR_SIZE);

    size_t num_rows   = 10;
    size_t total_size = sizeof(int64_t) * (num_rows + 1) +    // offsets
                        DEFAULT_VARCHAR_SIZE * num_rows +     // data
                        sirius::utils::ceil_div_8(num_rows);  // mask
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create a DuckDB vector with string data
    duckdb::Vector vec(varchar_type, 4);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    str_data[0]    = duckdb::string_t("hello");
    str_data[1]    = duckdb::string_t("world");  // This will be NULL
    str_data[2]    = duckdb::string_t("foo");
    str_data[3]    = duckdb::string_t("bar");  // This will be NULL

    duckdb::ValidityMask validity(4);
    validity.Initialize(4);
    validity.SetAllValid(4);
    validity.SetInvalid(1);  // Row 1 is NULL
    validity.SetInvalid(3);  // Row 3 is NULL

    // Process the column
    builder.process_column(vec, validity, 4, 0, allocation);

    // Verify total data bytes (only valid strings: 5 + 3 = 8 bytes)
    REQUIRE(builder.total_data_bytes == 8);

    // Verify offsets - NULL rows should have same offset as previous row
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);  // Initial
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 5);  // After "hello"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) ==
            5);  // NULL - same as previous
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 8);  // After "foo"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) ==
            8);  // NULL - same as previous
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - Multiple batch processing
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - multiple batch processing", "[duckdb_scan_task][column_builder]")
{
  SECTION("process multiple batches of INTEGER data")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Process first batch
    duckdb::Vector vec1(int_type, 10);
    auto* data1 = reinterpret_cast<int32_t*>(vec1.GetData());
    for (size_t i = 0; i < 10; ++i) {
      data1[i] = static_cast<int32_t>(i);
    }
    duckdb::ValidityMask validity1(10);
    validity1.Initialize(10);
    validity1.SetAllValid(10);
    builder.process_column(vec1, validity1, 10, 0, allocation);

    // Process second batch
    duckdb::Vector vec2(int_type, 10);
    auto* data2 = reinterpret_cast<int32_t*>(vec2.GetData());
    for (size_t i = 0; i < 10; ++i) {
      data2[i] = static_cast<int32_t>(i + 100);
    }
    duckdb::ValidityMask validity2(10);
    validity2.Initialize(10);
    validity2.SetAllValid(10);
    builder.process_column(vec2, validity2, 10, 10, allocation);  // row_offset = 10

    // Verify total data bytes
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 20);
  }

  SECTION("process multiple batches of VARCHAR data")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, 256);

    size_t num_rows   = 20;
    size_t total_size = sizeof(int64_t) * (num_rows + 1) +    // offsets
                        256 * num_rows +                      // data
                        sirius::utils::ceil_div_8(num_rows);  // mask
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Process first batch
    duckdb::Vector vec1(varchar_type, 3);
    auto* str_data1 = reinterpret_cast<duckdb::string_t*>(vec1.GetData());
    str_data1[0]    = duckdb::string_t("first");
    str_data1[1]    = duckdb::string_t("second");
    str_data1[2]    = duckdb::string_t("third");

    duckdb::ValidityMask validity1(3);
    validity1.Initialize(3);
    validity1.SetAllValid(3);
    builder.process_column(vec1, validity1, 3, 0, allocation);

    // Process second batch
    duckdb::Vector vec2(varchar_type, 3);
    auto* str_data2 = reinterpret_cast<duckdb::string_t*>(vec2.GetData());
    str_data2[0]    = duckdb::string_t("fourth");
    str_data2[1]    = duckdb::string_t("fifth");
    str_data2[2]    = duckdb::string_t("sixth");

    duckdb::ValidityMask validity2(3);
    validity2.Initialize(3);
    validity2.SetAllValid(3);
    builder.process_column(vec2, validity2, 3, 3, allocation);  // row_offset = 3

    // Verify total data bytes (5+6+5+6+5+5 = 32)
    REQUIRE(builder.total_data_bytes == 32);

    // Verify offsets are cumulative
    // After processing 6 strings, the offset accessor cursor is at position 6 (after writing the
    // 6th offset) The cursor is pointing to the final cumulative offset value
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 32);
  }

  SECTION("process multiple batches with mixed NULL values")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // First batch: some NULLs
    duckdb::Vector vec1(int_type, 5);
    auto* data1 = reinterpret_cast<int32_t*>(vec1.GetData());
    for (size_t i = 0; i < 5; ++i) {
      data1[i] = static_cast<int32_t>(i);
    }
    duckdb::ValidityMask validity1(5);
    validity1.Initialize(5);
    validity1.SetAllValid(5);
    validity1.SetInvalid(2);
    builder.process_column(vec1, validity1, 5, 0, allocation);

    // Second batch: different NULLs
    duckdb::Vector vec2(int_type, 5);
    auto* data2 = reinterpret_cast<int32_t*>(vec2.GetData());
    for (size_t i = 0; i < 5; ++i) {
      data2[i] = static_cast<int32_t>(i + 100);
    }
    duckdb::ValidityMask validity2(5);
    validity2.Initialize(5);
    validity2.SetAllValid(5);
    validity2.SetInvalid(0);
    validity2.SetInvalid(4);
    builder.process_column(vec2, validity2, 5, 5, allocation);  // row_offset = 5

    // Verify total data bytes
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 10);
    REQUIRE(builder.null_count == 3);

    // Verify mask has correct NULLs (rows 2, 5, and 9 should be NULL)
    // In DuckDB validity masks: 1 = valid, 0 = invalid
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current(allocation);
    builder.mask_blocks_accessor.advance();
    auto mask_byte_1 = builder.mask_blocks_accessor.get_current(allocation);

    // Row 2 should be NULL (bit 2 in first byte should be 0)
    REQUIRE((mask_byte_0 & (1 << 2)) == 0);
    // Row 5 should be NULL (bit 5 in first byte should be 0)
    REQUIRE((mask_byte_0 & (1 << 5)) == 0);
    // Row 9 should be NULL (bit 1 in second byte should be 0, since 9 - 8 = 1)
    REQUIRE((mask_byte_1 & (1 << 1)) == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - Edge Cases
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - edge cases", "[duckdb_scan_task][column_builder]")
{
  SECTION("empty vector (0 rows)")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Process empty vector
    duckdb::Vector vec(int_type, static_cast<idx_t>(0));
    duckdb::ValidityMask validity(0);
    validity.Initialize(0);

    builder.process_column(vec, validity, 0, 0, allocation);

    // Should have no data bytes
    REQUIRE(builder.total_data_bytes == 0);
  }

  SECTION("all NULL vector")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 100;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create vector with all NULLs
    size_t processed_rows = 10;
    duckdb::Vector vec(int_type, static_cast<idx_t>(processed_rows));
    duckdb::ValidityMask validity(static_cast<idx_t>(processed_rows));
    validity.Initialize(static_cast<idx_t>(processed_rows));
    validity.SetAllInvalid(static_cast<idx_t>(processed_rows));

    builder.process_column(vec, validity, processed_rows, 0, allocation);

    // Should still have data bytes (for fixed-width types, data is always copied)
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 10);

    // Verify all bits in mask are 0
    // The mask accessor starts at byte offset (sizeof(int32_t) * 100) in the allocation
    // After process_column, it should be at the correct position to read the mask
    // Reset to the mask start position
    size_t mask_offset = sizeof(int32_t) * num_rows;
    builder.mask_blocks_accessor.set_cursor(mask_offset);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current(allocation);
    builder.mask_blocks_accessor.advance();
    auto mask_byte_1 = builder.mask_blocks_accessor.get_current(allocation);

    REQUIRE(mask_byte_0 == 0);
    auto const tail_bits = static_cast<uint8_t>(processed_rows % CHAR_BIT);
    if (tail_bits != 0) {
      auto const tail_mask = static_cast<uint8_t>((1u << tail_bits) - 1u);
      REQUIRE((mask_byte_1 & tail_mask) == 0);
    } else {
      REQUIRE(mask_byte_1 == 0);
    }
  }

  SECTION("empty strings in VARCHAR")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, 256);

    size_t num_rows      = 10;
    size_t max_data_size = 1024;
    size_t total_size =
      max_data_size + (num_rows + 1) * sizeof(int64_t) + sirius::utils::ceil_div_8(num_rows);
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create vector with empty strings
    duckdb::Vector vec(varchar_type, 3);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    str_data[0]    = duckdb::string_t("");
    str_data[1]    = duckdb::string_t("");
    str_data[2]    = duckdb::string_t("");

    duckdb::ValidityMask validity(3);
    validity.Initialize(3);
    validity.SetAllValid(3);

    builder.process_column(vec, validity, 3, 0, allocation);

    // Empty strings contribute 0 data bytes
    REQUIRE(builder.total_data_bytes == 0);

    // Offsets should all be 0 (empty strings have length 0)
    // After processing 3 empty strings, the cursor is at position 3
    // The value at position 3 should be 0 (cumulative offset after 3 empty strings)
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);

    // Verify all offsets by resetting and reading from the beginning
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);
  }

  SECTION("mixed empty and non-empty VARCHAR strings")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, 256);

    size_t num_rows      = 10;
    size_t max_data_size = 1024;
    size_t total_size =
      max_data_size + (num_rows + 1) * sizeof(int64_t) + sirius::utils::ceil_div_8(num_rows);
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Mix of empty and non-empty strings
    duckdb::Vector vec(varchar_type, 5);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    str_data[0]    = duckdb::string_t("");
    str_data[1]    = duckdb::string_t("hello");
    str_data[2]    = duckdb::string_t("");
    str_data[3]    = duckdb::string_t("world");
    str_data[4]    = duckdb::string_t("");

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);

    builder.process_column(vec, validity, 5, 0, allocation);

    // Total: 5 + 5 = 10 bytes
    REQUIRE(builder.total_data_bytes == 10);

    // Verify final cumulative offset
    // After processing 5 strings, cursor is at position 5, should read the final offset (10)
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 10);

    // Verify all offsets by resetting and reading from the beginning
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);  // Initial
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);  // After empty
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 5);  // After "hello"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 5);  // After empty
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 10);  // After "world"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 10);  // After empty
  }

  SECTION("single row processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    size_t num_rows   = 10;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Process single row
    duckdb::Vector vec(int_type, 1);
    auto* data = reinterpret_cast<int32_t*>(vec.GetData());
    data[0]    = 42;

    duckdb::ValidityMask validity(1);
    validity.Initialize(1);
    validity.SetAllValid(1);

    builder.process_column(vec, validity, 1, 0, allocation);

    REQUIRE(builder.total_data_bytes == sizeof(int32_t));

    // Verify the single value
    // Reset cursor to the start of the data
    builder.data_blocks_accessor.set_cursor(0);
    int32_t value;
    builder.data_blocks_accessor.memcpy_to(allocation, &value, sizeof(int32_t));
    REQUIRE(value == 42);
  }
}

//===----------------------------------------------------------------------===//
// Test: Packed allocation with multiple columns
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - packed allocation multiple columns",
          "[duckdb_scan_task][column_builder]")
{
  SECTION("two fixed-width columns in packed allocation")
  {
    // Simulate layout: [INT column data][INT column mask][BIGINT column data][BIGINT column mask]
    auto int_type    = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    auto bigint_type = duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);

    duckdb_scan_task_local_state::column_builder int_builder(int_type, 256);
    duckdb_scan_task_local_state::column_builder bigint_builder(bigint_type, 256);

    size_t num_rows         = 10;
    size_t int_data_size    = sizeof(int32_t) * num_rows;
    size_t int_mask_size    = sirius::utils::ceil_div_8(num_rows);
    size_t bigint_data_size = sizeof(int64_t) * num_rows;
    size_t bigint_mask_size = sirius::utils::ceil_div_8(num_rows);
    size_t total_size       = int_data_size + int_mask_size + bigint_data_size + bigint_mask_size;

    auto allocation = create_test_allocation(total_size);

    // Initialize INT column at offset 0
    size_t int_byte_offset = 0;
    int_builder.initialize_accessors(num_rows, int_byte_offset, allocation);

    // Initialize BIGINT column after INT column
    size_t bigint_byte_offset = int_data_size + int_mask_size;
    bigint_builder.initialize_accessors(num_rows, bigint_byte_offset, allocation);

    // Process INT data
    duckdb::Vector int_vec(int_type, 10);
    auto* int_data = reinterpret_cast<int32_t*>(int_vec.GetData());
    for (size_t i = 0; i < 10; ++i) {
      int_data[i] = static_cast<int32_t>(i * 10);
    }
    duckdb::ValidityMask int_validity(10);
    int_validity.Initialize(10);
    int_validity.SetAllValid(10);
    int_builder.process_column(int_vec, int_validity, 10, 0, allocation);

    // Process BIGINT data
    duckdb::Vector bigint_vec(bigint_type, 10);
    auto* bigint_data = reinterpret_cast<int64_t*>(bigint_vec.GetData());
    for (size_t i = 0; i < 10; ++i) {
      bigint_data[i] = static_cast<int64_t>(i * 100);
    }
    duckdb::ValidityMask bigint_validity(10);
    bigint_validity.Initialize(10);
    bigint_validity.SetAllValid(10);
    bigint_builder.process_column(bigint_vec, bigint_validity, 10, 0, allocation);

    // Verify INT data
    int_builder.data_blocks_accessor.set_cursor(int_byte_offset);
    for (size_t i = 0; i < 10; ++i) {
      auto value = int_builder.data_blocks_accessor.get_current_as<int32_t>(allocation);
      REQUIRE(value == static_cast<int32_t>(i * 10));
      int_builder.data_blocks_accessor.advance_as<int32_t>();
    }

    // Verify BIGINT data
    bigint_builder.data_blocks_accessor.set_cursor(bigint_byte_offset);
    for (size_t i = 0; i < 10; ++i) {
      auto value = bigint_builder.data_blocks_accessor.get_current_as<int64_t>(allocation);
      REQUIRE(value == static_cast<int64_t>(i * 100));
      bigint_builder.data_blocks_accessor.advance_as<int64_t>();
    }
  }

  SECTION("mixed fixed-width and VARCHAR in packed allocation")
  {
    // Simulate layout: [INT data][INT mask][VARCHAR offsets][VARCHAR data][VARCHAR mask]
    auto int_type     = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);

    duckdb_scan_task_local_state::column_builder int_builder(int_type, 256);
    duckdb_scan_task_local_state::column_builder varchar_builder(varchar_type, 256);

    size_t num_rows            = 5;
    size_t int_data_size       = sizeof(int32_t) * num_rows;
    size_t int_mask_size       = sirius::utils::ceil_div_8(num_rows);
    size_t varchar_offset_size = (num_rows + 1) * sizeof(int64_t);
    size_t varchar_data_size   = 256 * num_rows;  // Max data size
    size_t varchar_mask_size   = sirius::utils::ceil_div_8(num_rows);
    size_t total_size =
      int_data_size + int_mask_size + varchar_offset_size + varchar_data_size + varchar_mask_size;

    auto allocation = create_test_allocation(total_size);

    // Initialize INT column at offset 0
    size_t int_byte_offset = 0;
    int_builder.initialize_accessors(num_rows, int_byte_offset, allocation);

    // Initialize VARCHAR column after INT column
    size_t varchar_byte_offset = int_data_size + int_mask_size;
    varchar_builder.initialize_accessors(num_rows, varchar_byte_offset, allocation);

    // Process INT data
    duckdb::Vector int_vec(int_type, 5);
    auto* int_data = reinterpret_cast<int32_t*>(int_vec.GetData());
    for (size_t i = 0; i < 5; ++i) {
      int_data[i] = static_cast<int32_t>(i + 100);
    }
    duckdb::ValidityMask int_validity(5);
    int_validity.Initialize(5);
    int_validity.SetAllValid(5);
    int_builder.process_column(int_vec, int_validity, 5, 0, allocation);

    // Process VARCHAR data
    duckdb::Vector varchar_vec(varchar_type, 5);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(varchar_vec.GetData());
    str_data[0]    = duckdb::string_t("apple");
    str_data[1]    = duckdb::string_t("banana");
    str_data[2]    = duckdb::string_t("cherry");
    str_data[3]    = duckdb::string_t("date");
    str_data[4]    = duckdb::string_t("elderberry");

    duckdb::ValidityMask varchar_validity(5);
    varchar_validity.Initialize(5);
    varchar_validity.SetAllValid(5);
    varchar_builder.process_column(varchar_vec, varchar_validity, 5, 0, allocation);

    // Verify INT data
    int_builder.data_blocks_accessor.set_cursor(int_byte_offset);
    for (size_t i = 0; i < 5; ++i) {
      auto value = int_builder.data_blocks_accessor.get_current_as<int32_t>(allocation);
      REQUIRE(value == static_cast<int32_t>(i + 100));
      int_builder.data_blocks_accessor.advance_as<int32_t>();
    }

    // Verify VARCHAR offsets
    varchar_builder.offset_blocks_accessor.set_cursor(varchar_byte_offset);
    REQUIRE(varchar_builder.offset_blocks_accessor.get_current(allocation) == 0);
    varchar_builder.offset_blocks_accessor.advance();
    REQUIRE(varchar_builder.offset_blocks_accessor.get_current(allocation) == 5);  // "apple"
    varchar_builder.offset_blocks_accessor.advance();
    REQUIRE(varchar_builder.offset_blocks_accessor.get_current(allocation) == 11);  // + "banana"
    varchar_builder.offset_blocks_accessor.advance();
    REQUIRE(varchar_builder.offset_blocks_accessor.get_current(allocation) == 17);  // + "cherry"
    varchar_builder.offset_blocks_accessor.advance();
    REQUIRE(varchar_builder.offset_blocks_accessor.get_current(allocation) == 21);  // + "date"
    varchar_builder.offset_blocks_accessor.advance();
    REQUIRE(varchar_builder.offset_blocks_accessor.get_current(allocation) ==
            31);  // + "elderberry"

    REQUIRE(varchar_builder.total_data_bytes == 31);
  }

  SECTION("three columns with NULLs in packed allocation")
  {
    // Test that NULL handling works correctly with packed layout
    auto int_type     = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    auto double_type  = duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);

    duckdb_scan_task_local_state::column_builder int_builder(int_type, 256);
    duckdb_scan_task_local_state::column_builder double_builder(double_type, 256);
    duckdb_scan_task_local_state::column_builder varchar_builder(varchar_type, 256);

    size_t num_rows    = 8;
    size_t int_size    = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    size_t double_size = sizeof(double) * num_rows + sirius::utils::ceil_div_8(num_rows);
    size_t varchar_size =
      (num_rows + 1) * sizeof(int64_t) + 256 * num_rows + sirius::utils::ceil_div_8(num_rows);
    size_t total_size = int_size + double_size + varchar_size;

    auto allocation = create_test_allocation(total_size);

    int_builder.initialize_accessors(num_rows, 0, allocation);
    double_builder.initialize_accessors(num_rows, int_size, allocation);
    varchar_builder.initialize_accessors(num_rows, int_size + double_size, allocation);

    // INT column: NULLs at rows 2, 5
    duckdb::Vector int_vec(int_type, 8);
    auto* int_data = reinterpret_cast<int32_t*>(int_vec.GetData());
    for (size_t i = 0; i < 8; ++i) {
      int_data[i] = static_cast<int32_t>(i);
    }
    duckdb::ValidityMask int_validity(8);
    int_validity.Initialize(8);
    int_validity.SetAllValid(8);
    int_validity.SetInvalid(2);
    int_validity.SetInvalid(5);
    int_builder.process_column(int_vec, int_validity, 8, 0, allocation);

    // DOUBLE column: NULLs at rows 1, 6
    duckdb::Vector double_vec(double_type, 8);
    auto* double_data = reinterpret_cast<double*>(double_vec.GetData());
    for (size_t i = 0; i < 8; ++i) {
      double_data[i] = static_cast<double>(i) * 1.5;
    }
    duckdb::ValidityMask double_validity(8);
    double_validity.Initialize(8);
    double_validity.SetAllValid(8);
    double_validity.SetInvalid(1);
    double_validity.SetInvalid(6);
    double_builder.process_column(double_vec, double_validity, 8, 0, allocation);

    // VARCHAR column: NULLs at rows 0, 7
    duckdb::Vector varchar_vec(varchar_type, 8);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(varchar_vec.GetData());
    for (size_t i = 0; i < 8; ++i) {
      str_data[i] = duckdb::string_t("str" + std::to_string(i));
    }
    duckdb::ValidityMask varchar_validity(8);
    varchar_validity.Initialize(8);
    varchar_validity.SetAllValid(8);
    varchar_validity.SetInvalid(0);
    varchar_validity.SetInvalid(7);
    varchar_builder.process_column(varchar_vec, varchar_validity, 8, 0, allocation);

    // Verify all columns have expected NULL counts
    REQUIRE(int_builder.null_count == 2);
    REQUIRE(double_builder.null_count == 2);
    REQUIRE(varchar_builder.null_count == 2);

    // Verify INT NULLs
    int_builder.mask_blocks_accessor.set_cursor(sizeof(int32_t) * num_rows);
    uint8_t int_mask = int_builder.mask_blocks_accessor.get_current(allocation);
    REQUIRE((int_mask & (1 << 2)) == 0);  // Row 2 is NULL
    REQUIRE((int_mask & (1 << 5)) == 0);  // Row 5 is NULL
    REQUIRE((int_mask & (1 << 0)) != 0);  // Row 0 is valid
    REQUIRE((int_mask & (1 << 1)) != 0);  // Row 1 is valid

    // Verify DOUBLE NULLs
    double_builder.mask_blocks_accessor.set_cursor(int_size + sizeof(double) * num_rows);
    uint8_t double_mask = double_builder.mask_blocks_accessor.get_current(allocation);
    REQUIRE((double_mask & (1 << 1)) == 0);  // Row 1 is NULL
    REQUIRE((double_mask & (1 << 6)) == 0);  // Row 6 is NULL
    REQUIRE((double_mask & (1 << 0)) != 0);  // Row 0 is valid

    // Verify VARCHAR NULLs
    // For VARCHAR, the mask is initialized at: data_offset + total_data_bytes_allocated
    // Where total_data_bytes_allocated = num_rows * default_varchar_size = 8 * 256 = 2048
    size_t varchar_data_offset = int_size + double_size + (num_rows + 1) * sizeof(int64_t);
    size_t varchar_mask_offset =
      varchar_data_offset + (num_rows * 256);  // 256 is default_varchar_size
    varchar_builder.mask_blocks_accessor.set_cursor(varchar_mask_offset);
    uint8_t varchar_mask = varchar_builder.mask_blocks_accessor.get_current(allocation);
    REQUIRE((varchar_mask & (1 << 0)) == 0);  // Row 0 is NULL
    REQUIRE((varchar_mask & (1 << 7)) == 0);  // Row 7 is NULL
    REQUIRE((varchar_mask & (1 << 1)) != 0);  // Row 1 is valid
  }
}

//===----------------------------------------------------------------------===//
// Test: VARCHAR space checking and edge cases
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - VARCHAR space checking edge cases",
          "[duckdb_scan_task][column_builder]")
{
  SECTION("sufficient_space_for_column returns false when space exceeded")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(
      varchar_type, 4);  // Small default size: 5 rows * 4 bytes = 20 bytes allocated

    size_t num_rows      = 5;
    size_t max_data_size = 20;  // Only 20 bytes of data space
    size_t total_size =
      max_data_size + (num_rows + 1) * sizeof(int64_t) + sirius::utils::ceil_div_8(num_rows);
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // Create vector with strings that exceed allocated space
    duckdb::Vector vec(varchar_type, 5);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    str_data[0]    = duckdb::string_t("this");  // 4 bytes
    str_data[1]    = duckdb::string_t("is");    // 2 bytes
    str_data[2]    = duckdb::string_t("too");   // 3 bytes
    str_data[3]    = duckdb::string_t("much");  // 4 bytes
    str_data[4]    = duckdb::string_t("data");  // 4 bytes
    // Total: 17 bytes - should fit

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);

    REQUIRE(builder.sufficient_space_for_column(vec, validity, 5));

    // Now try with more data than allocated
    str_data[0] = duckdb::string_t("this_is_definitely");  // 18 bytes
    str_data[1] = duckdb::string_t("too_much");            // 8 bytes
    // Total: 26+ bytes - should NOT fit

    REQUIRE_FALSE(builder.sufficient_space_for_column(vec, validity, 5));
  }

  SECTION("VARCHAR with all NULLs uses no data space")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, 256);

    size_t num_rows      = 10;
    size_t max_data_size = 1024;
    size_t total_size =
      max_data_size + (num_rows + 1) * sizeof(int64_t) + sirius::utils::ceil_div_8(num_rows);
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    // All NULLs - strings don't matter
    duckdb::Vector vec(varchar_type, 10);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    for (size_t i = 0; i < 10; ++i) {
      str_data[i] = duckdb::string_t("this would be huge if it counted");
    }

    duckdb::ValidityMask validity(10);
    validity.Initialize(10);
    validity.SetAllInvalid(10);  // Set all rows as NULL

    builder.process_column(vec, validity, 10, 0, allocation);

    // No data bytes should be used since all rows are NULL
    REQUIRE(builder.total_data_bytes == 0);
    REQUIRE(builder.null_count == 10);

    // All offsets should be 0
    builder.offset_blocks_accessor.set_cursor(0);
    for (size_t i = 0; i <= 10; ++i) {
      REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);
      builder.offset_blocks_accessor.advance();
    }
  }

  SECTION("VARCHAR alternating NULL and valid pattern")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type, 256);

    size_t num_rows      = 10;
    size_t max_data_size = 1024;
    size_t total_size =
      max_data_size + (num_rows + 1) * sizeof(int64_t) + sirius::utils::ceil_div_8(num_rows);
    auto allocation = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    duckdb::Vector vec(varchar_type, 10);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    for (size_t i = 0; i < 10; ++i) {
      str_data[i] = duckdb::string_t("x");  // 1 byte each
    }

    // Alternating NULL/valid: NULL at even indices
    duckdb::ValidityMask validity(10);
    validity.Initialize(10);
    validity.SetAllValid(10);
    for (size_t i = 0; i < 10; i += 2) {
      validity.SetInvalid(i);
    }

    builder.process_column(vec, validity, 10, 0, allocation);

    // Only 5 valid strings, each 1 byte = 5 bytes total
    REQUIRE(builder.total_data_bytes == 5);
    REQUIRE(builder.null_count == 5);

    // Verify offsets: should increment by 0 for NULLs, by 1 for valid
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);  // Initial
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 0);  // After NULL at 0
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 1);  // After valid at 1
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 1);  // After NULL at 2
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current(allocation) == 2);  // After valid at 3
  }
}

//===----------------------------------------------------------------------===//
// Test: NULL handling at block boundaries
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - NULL handling at boundaries", "[duckdb_scan_task][column_builder]")
{
  SECTION("NULLs at byte boundaries in mask")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    // Test with exactly 16 rows (2 mask bytes)
    size_t num_rows   = 16;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    duckdb::Vector vec(int_type, 16);
    auto* data = reinterpret_cast<int32_t*>(vec.GetData());
    for (size_t i = 0; i < 16; ++i) {
      data[i] = static_cast<int32_t>(i);
    }

    // Set NULLs at rows 7 and 8 (boundary between two mask bytes)
    duckdb::ValidityMask validity(16);
    validity.Initialize(16);
    validity.SetAllValid(16);
    validity.SetInvalid(7);  // Last bit of first byte
    validity.SetInvalid(8);  // First bit of second byte

    builder.process_column(vec, validity, 16, 0, allocation);

    REQUIRE(builder.null_count == 2);

    // Check mask bytes
    builder.mask_blocks_accessor.set_cursor(sizeof(int32_t) * num_rows);
    uint8_t mask_byte_0 = builder.mask_blocks_accessor.get_current(allocation);
    builder.mask_blocks_accessor.advance();
    uint8_t mask_byte_1 = builder.mask_blocks_accessor.get_current(allocation);

    // Bit 7 of first byte should be 0
    REQUIRE((mask_byte_0 & (1 << 7)) == 0);
    // All other bits of first byte should be 1
    REQUIRE((mask_byte_0 & 0x7F) == 0x7F);

    // Bit 0 of second byte should be 0
    REQUIRE((mask_byte_1 & (1 << 0)) == 0);
    // All other bits of second byte should be 1
    REQUIRE((mask_byte_1 & 0xFE) == 0xFE);
  }

  SECTION("all rows NULL across multiple mask bytes")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    // Test with 24 rows (3 mask bytes)
    size_t num_rows   = 24;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    duckdb::Vector vec(int_type, 24);
    duckdb::ValidityMask validity(24);
    validity.Initialize(24);
    validity.SetAllInvalid(24);  // Set all rows as NULL

    builder.process_column(vec, validity, 24, 0, allocation);

    REQUIRE(builder.null_count == 24);

    // All mask bytes should be 0 (all rows NULL)
    builder.mask_blocks_accessor.set_cursor(sizeof(int32_t) * num_rows);
    for (size_t i = 0; i < 3; ++i) {
      uint8_t mask_byte = builder.mask_blocks_accessor.get_current(allocation);
      REQUIRE(mask_byte == 0);
      builder.mask_blocks_accessor.advance();
    }
  }

  SECTION("no NULLs across multiple mask bytes")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type, 256);

    // Test with 20 rows (3 mask bytes, last one partial)
    size_t num_rows   = 20;
    size_t total_size = sizeof(int32_t) * num_rows + sirius::utils::ceil_div_8(num_rows);
    auto allocation   = create_test_allocation(total_size);
    builder.initialize_accessors(num_rows, 0, allocation);

    duckdb::Vector vec(int_type, 20);
    auto* data = reinterpret_cast<int32_t*>(vec.GetData());
    for (size_t i = 0; i < 20; ++i) {
      data[i] = static_cast<int32_t>(i);
    }

    duckdb::ValidityMask validity(20);
    validity.Initialize(20);
    validity.SetAllValid(20);

    builder.process_column(vec, validity, 20, 0, allocation);

    REQUIRE(builder.null_count == 0);

    // First two mask bytes should be 0xFF (all valid)
    builder.mask_blocks_accessor.set_cursor(sizeof(int32_t) * num_rows);
    REQUIRE(builder.mask_blocks_accessor.get_current(allocation) == 0xFF);
    builder.mask_blocks_accessor.advance();
    REQUIRE(builder.mask_blocks_accessor.get_current(allocation) == 0xFF);
    builder.mask_blocks_accessor.advance();
    // Third mask byte should have first 4 bits set (rows 16-19)
    uint8_t mask_byte_2 = builder.mask_blocks_accessor.get_current(allocation);
    REQUIRE((mask_byte_2 & 0x0F) == 0x0F);
  }
}
