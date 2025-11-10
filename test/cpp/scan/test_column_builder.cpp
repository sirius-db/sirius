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
#include "test_utils.hpp"

// sirius
#include <helper/utils.hpp>
#include <scan/duckdb_scan_task.hpp>

// duckdb
#include <duckdb/common/types/validity_mask.hpp>
#include <duckdb/common/types/vector.hpp>

using namespace sirius::parallel;
using namespace sirius::memory;

//===----------------------------------------------------------------------===//
// Test: column_builder - Construction
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - construction", "[duckdb_scan_task][column_builder]")
{
  // Initialize memory reservation manager for these tests
  initialize_memory_manager();

  SECTION("construct with INTEGER type")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::INTEGER);
    REQUIRE(builder.type_size == sizeof(int32_t));
    REQUIRE(builder.total_data_bytes == 0);
    REQUIRE(builder.allocator != nullptr);
  }

  SECTION("construct with BIGINT type")
  {
    auto bigint_type = duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
    duckdb_scan_task_local_state::column_builder builder(bigint_type);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::BIGINT);
    REQUIRE(builder.type_size == sizeof(int64_t));
    REQUIRE(builder.total_data_bytes == 0);
  }

  SECTION("construct with VARCHAR type")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::VARCHAR);
    REQUIRE(builder.total_data_bytes == 0);
  }

  SECTION("construct with DOUBLE type")
  {
    auto double_type = duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
    duckdb_scan_task_local_state::column_builder builder(double_type);

    REQUIRE(builder.type.id() == duckdb::LogicalTypeId::DOUBLE);
    REQUIRE(builder.type_size == sizeof(double));
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - Memory Reservation
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - memory reservation and allocation",
          "[duckdb_scan_task][column_builder]")
{
  // Initialize memory reservation manager for these tests
  initialize_memory_manager();

  SECTION("reserve and allocate for fixed-width type")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    size_t num_rows = 100;
    builder.reserve_memory(num_rows);

    // Check that reservations were made
    REQUIRE(builder.data_reservation != nullptr);
    REQUIRE(builder.mask_reservation != nullptr);
    REQUIRE(builder.offset_reservation == nullptr);  // Not needed for non-VARCHAR

    // Check reservation sizes
    REQUIRE(builder.data_reservation->size >= sizeof(int32_t) * num_rows);
    REQUIRE(builder.mask_reservation->size >= sirius::utils::ceil_div_8(num_rows));

    // Allocate memory
    builder.allocate_memory();

    // Check that allocations were made
    REQUIRE(builder.data_blocks_accessor.allocation != nullptr);
    REQUIRE(builder.mask_blocks_accessor.allocation != nullptr);
  }

  SECTION("reserve and allocate for VARCHAR type")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    size_t num_rows = 100;
    builder.reserve_memory(num_rows);

    // Check that reservations were made
    REQUIRE(builder.data_reservation != nullptr);
    REQUIRE(builder.mask_reservation != nullptr);
    REQUIRE(builder.offset_reservation != nullptr);  // Needed for VARCHAR

    // Check offset reservation size
    REQUIRE(builder.offset_reservation->size >= sizeof(int64_t) * (num_rows + 1));

    // Allocate memory
    builder.allocate_memory();

    // Check that allocations were made
    REQUIRE(builder.data_blocks_accessor.allocation != nullptr);
    REQUIRE(builder.mask_blocks_accessor.allocation != nullptr);
    REQUIRE(builder.offset_blocks_accessor.allocation != nullptr);

    // Check that offset is initialized to zero
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - sufficient_space_for_column
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - sufficient_space_for_column", "[duckdb_scan_task][column_builder]")
{
  initialize_memory_manager();

  SECTION("fixed-width type space check")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    // Reserve memory for 100 rows
    size_t num_rows = 100;
    builder.reserve_memory(num_rows);
    builder.allocate_memory();

    // Create a DuckDB vector with some data
    duckdb::Vector vec(int_type, 50);
    auto* data = reinterpret_cast<int32_t*>(vec.GetData());
    for (size_t i = 0; i < 50; ++i) {
      data[i] = static_cast<int32_t>(i);
    }

    duckdb::ValidityMask validity(50);
    validity.Initialize(50);
    validity.SetAllValid(50);

    // Should have sufficient space for 50 rows
    REQUIRE(builder.sufficient_space_for_column(vec, validity, 50) == true);

    // Should have sufficient space for 100 rows total
    REQUIRE(builder.sufficient_space_for_column(vec, validity, 100) == true);

    // Should NOT have sufficient space for 101 rows
    REQUIRE(builder.sufficient_space_for_column(vec, validity, 101) == false);
  }

  SECTION("VARCHAR type space check")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    // Reserve memory for strings - estimate 10 bytes per string, 100 strings
    size_t num_rows = 100;
    builder.reserve_memory(num_rows);
    builder.allocate_memory();

    // Create a DuckDB vector with string data
    duckdb::Vector vec(varchar_type, 10);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());

    // Fill with small strings (5 bytes each = 50 bytes total)
    for (size_t i = 0; i < 10; ++i) {
      str_data[i] = duckdb::string_t("test" + std::to_string(i));
    }

    duckdb::ValidityMask validity(10);
    validity.Initialize(10);
    validity.SetAllValid(10);

    // Should have sufficient space
    REQUIRE(builder.sufficient_space_for_column(vec, validity, 10) == true);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - process_mask_for_column
//===----------------------------------------------------------------------===//
TEST_CASE("column_builder - process_mask_for_column", "[duckdb_scan_task][column_builder]")
{
  initialize_memory_manager();

  SECTION("byte-aligned mask processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

    // Create a validity mask with some NULLs
    duckdb::ValidityMask validity(16);
    validity.Initialize(16);   // Must initialize to allocate backing storage
    validity.SetAllValid(16);  // Start with all valid
    validity.SetInvalid(3);    // Row 3 is NULL
    validity.SetInvalid(7);    // Row 7 is NULL
    validity.SetInvalid(15);   // Row 15 is NULL

    // Process the mask (byte-aligned: row_offset = 0)
    builder.process_mask_for_column(validity, 16, 0);

    // Verify the mask was copied correctly
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current();
    builder.mask_blocks_accessor.advance();
    auto mask_byte_1 = builder.mask_blocks_accessor.get_current();

    // Bit 3 and 7 should be 0 (invalid) in first byte
    REQUIRE((mask_byte_0 & (1 << 3)) == 0);
    REQUIRE((mask_byte_0 & (1 << 7)) == 0);

    // Bit 7 (15 - 8) should be 0 in second byte
    REQUIRE((mask_byte_1 & (1 << 7)) == 0);
  }

  SECTION("byte-unaligned mask processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

    // First, add 3 rows (to create a 3-bit offset)
    duckdb::ValidityMask validity1(3);
    validity1.Initialize(3);
    validity1.SetAllValid(3);
    builder.process_mask_for_column(validity1, 3, 0);

    // Now add 8 more rows starting at bit offset 3 (byte-unaligned)
    duckdb::ValidityMask validity2(8);
    validity2.Initialize(8);
    validity2.SetAllValid(8);
    validity2.SetInvalid(2);  // Should be at bit position 5 (3 + 2) overall
    builder.process_mask_for_column(validity2, 8, 3);

    // Verify the unaligned write worked
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current();

    // Bit 5 should be 0 (invalid)
    REQUIRE((mask_byte_0 & (1 << 5)) == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - process_column (fixed-width)
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - process_column for fixed-width types",
          "[duckdb_scan_task][column_builder]")
{
  initialize_memory_manager();

  SECTION("INTEGER column processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

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
    builder.process_column(vec, validity, 10, 0);

    // Verify data was copied
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 10);

    // Verify the data values
    builder.data_blocks_accessor.set_cursor(0);
    for (size_t i = 0; i < 10; ++i) {
      int32_t value;
      std::memcpy(&value,
                  static_cast<uint8_t*>(builder.data_blocks_accessor.allocation->blocks[0]) +
                    i * sizeof(int32_t),
                  sizeof(int32_t));
      REQUIRE(value == static_cast<int32_t>(i * 100));
    }
  }

  SECTION("BIGINT column processing")
  {
    auto bigint_type = duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT);
    duckdb_scan_task_local_state::column_builder builder(bigint_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

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
    builder.process_column(vec, validity, 5, 0);

    // Verify data was copied
    REQUIRE(builder.total_data_bytes == sizeof(int64_t) * 5);

    // Verify the data values
    for (size_t i = 0; i < 5; ++i) {
      int64_t value;
      std::memcpy(&value,
                  static_cast<uint8_t*>(builder.data_blocks_accessor.allocation->blocks[0]) +
                    i * sizeof(int64_t),
                  sizeof(int64_t));
      REQUIRE(value == static_cast<int64_t>(i * 1000000LL));
    }
  }

  SECTION("DOUBLE column processing")
  {
    auto double_type = duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE);
    duckdb_scan_task_local_state::column_builder builder(double_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

    // Create a DuckDB vector with double data
    duckdb::Vector vec(double_type, 5);
    auto* data = reinterpret_cast<double*>(vec.GetData());
    for (size_t i = 0; i < 5; ++i) {
      data[i] = i * 3.14159;
    }

    duckdb::ValidityMask validity(5);
    validity.Initialize(5);
    validity.SetAllValid(5);
    validity.SetInvalid(2);  // Row 2 is NULL

    // Process the column
    builder.process_column(vec, validity, 5, 0);

    // Verify data was copied
    REQUIRE(builder.total_data_bytes == sizeof(double) * 5);
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - process_column (VARCHAR)
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - process_column for VARCHAR", "[duckdb_scan_task][column_builder]")
{
  initialize_memory_manager();

  SECTION("VARCHAR column processing with all valid rows")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    // Reserve enough memory for strings
    builder.reserve_memory(10);
    builder.allocate_memory();

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
    builder.process_column(vec, validity, 5, 0);

    // Verify total data bytes (5 + 5 + 3 + 3 + 3 = 19 bytes)
    REQUIRE(builder.total_data_bytes == 19);

    // Verify offsets were set correctly
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);  // Initial offset
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 5);  // After "hello"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 10);  // After "world"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 13);  // After "foo"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 16);  // After "bar"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 19);  // After "baz"
  }

  SECTION("VARCHAR column processing with NULL values")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    builder.reserve_memory(10);
    builder.allocate_memory();

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
    builder.process_column(vec, validity, 4, 0);

    // Verify total data bytes (only valid strings: 5 + 3 = 8 bytes)
    REQUIRE(builder.total_data_bytes == 8);

    // Verify offsets - NULL rows should have same offset as previous row
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);  // Initial
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 5);  // After "hello"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 5);  // NULL - same as previous
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 8);  // After "foo"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 8);  // NULL - same as previous
  }
}

//===----------------------------------------------------------------------===//
// Test: column_builder - Multiple batch processing
//===----------------------------------------------------------------------===//

TEST_CASE("column_builder - multiple batch processing", "[duckdb_scan_task][column_builder]")
{
  initialize_memory_manager();

  SECTION("process multiple batches of INTEGER data")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

    // Process first batch
    duckdb::Vector vec1(int_type, 10);
    auto* data1 = reinterpret_cast<int32_t*>(vec1.GetData());
    for (size_t i = 0; i < 10; ++i) {
      data1[i] = static_cast<int32_t>(i);
    }
    duckdb::ValidityMask validity1(10);
    validity1.Initialize(10);
    validity1.SetAllValid(10);
    builder.process_column(vec1, validity1, 10, 0);

    // Process second batch
    duckdb::Vector vec2(int_type, 10);
    auto* data2 = reinterpret_cast<int32_t*>(vec2.GetData());
    for (size_t i = 0; i < 10; ++i) {
      data2[i] = static_cast<int32_t>(i + 100);
    }
    duckdb::ValidityMask validity2(10);
    validity2.Initialize(10);
    validity2.SetAllValid(10);
    builder.process_column(vec2, validity2, 10, 10);  // row_offset = 10

    // Verify total data bytes
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 20);
  }

  SECTION("process multiple batches of VARCHAR data")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    builder.reserve_memory(20);
    builder.allocate_memory();

    // Process first batch
    duckdb::Vector vec1(varchar_type, 3);
    auto* str_data1 = reinterpret_cast<duckdb::string_t*>(vec1.GetData());
    str_data1[0]    = duckdb::string_t("first");
    str_data1[1]    = duckdb::string_t("second");
    str_data1[2]    = duckdb::string_t("third");

    duckdb::ValidityMask validity1(3);
    validity1.Initialize(3);
    validity1.SetAllValid(3);
    builder.process_column(vec1, validity1, 3, 0);

    // Process second batch
    duckdb::Vector vec2(varchar_type, 3);
    auto* str_data2 = reinterpret_cast<duckdb::string_t*>(vec2.GetData());
    str_data2[0]    = duckdb::string_t("fourth");
    str_data2[1]    = duckdb::string_t("fifth");
    str_data2[2]    = duckdb::string_t("sixth");

    duckdb::ValidityMask validity2(3);
    validity2.Initialize(3);
    validity2.SetAllValid(3);
    builder.process_column(vec2, validity2, 3, 3);  // row_offset = 3

    // Verify total data bytes (5+6+5+6+5+5 = 32)
    REQUIRE(builder.total_data_bytes == 32);

    // Verify offsets are cumulative
    builder.offset_blocks_accessor.set_cursor(6 * sizeof(int64_t));  // After all 6 strings
    REQUIRE(builder.offset_blocks_accessor.get_current() == 32);
  }

  SECTION("process multiple batches with mixed NULL values")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

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
    builder.process_column(vec1, validity1, 5, 0);

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
    builder.process_column(vec2, validity2, 5, 5);  // row_offset = 5

    // Verify total data bytes
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 10);

    // Verify mask has correct NULLs (rows 2, 5, and 9 should be NULL)
    // In DuckDB validity masks: 1 = valid, 0 = invalid
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current();
    builder.mask_blocks_accessor.advance();
    auto mask_byte_1 = builder.mask_blocks_accessor.get_current();

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
  initialize_memory_manager();

  SECTION("empty vector (0 rows)")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

    // Process empty vector
    duckdb::Vector vec(int_type, static_cast<idx_t>(0));
    duckdb::ValidityMask validity(0);
    validity.Initialize(0);

    builder.process_column(vec, validity, 0, 0);

    // Should have no data bytes
    REQUIRE(builder.total_data_bytes == 0);
  }

  SECTION("all NULL vector")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(100);
    builder.allocate_memory();

    // Create vector with all NULLs
    duckdb::Vector vec(int_type, 10);
    duckdb::ValidityMask validity(10);
    validity.Initialize(10);
    validity.SetAllInvalid(10);

    builder.process_column(vec, validity, 10, 0);

    // Should still have data bytes (for fixed-width types, data is always copied)
    REQUIRE(builder.total_data_bytes == sizeof(int32_t) * 10);

    // Verify all bits in mask are 0
    builder.mask_blocks_accessor.set_cursor(0);
    auto mask_byte_0 = builder.mask_blocks_accessor.get_current();
    builder.mask_blocks_accessor.advance();
    auto mask_byte_1 = builder.mask_blocks_accessor.get_current();

    REQUIRE(mask_byte_0 == 0);
    REQUIRE(mask_byte_1 == 0);
  }

  SECTION("empty strings in VARCHAR")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    builder.reserve_memory(10);
    builder.allocate_memory();

    // Create vector with empty strings
    duckdb::Vector vec(varchar_type, 3);
    auto* str_data = reinterpret_cast<duckdb::string_t*>(vec.GetData());
    str_data[0]    = duckdb::string_t("");
    str_data[1]    = duckdb::string_t("");
    str_data[2]    = duckdb::string_t("");

    duckdb::ValidityMask validity(3);
    validity.Initialize(3);
    validity.SetAllValid(3);

    builder.process_column(vec, validity, 3, 0);

    // Empty strings contribute 0 data bytes
    REQUIRE(builder.total_data_bytes == 0);

    // Offsets should all be 0
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);
  }

  SECTION("mixed empty and non-empty VARCHAR strings")
  {
    auto varchar_type = duckdb::LogicalType(duckdb::LogicalTypeId::VARCHAR);
    duckdb_scan_task_local_state::column_builder builder(varchar_type);

    builder.reserve_memory(10);
    builder.allocate_memory();

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

    builder.process_column(vec, validity, 5, 0);

    // Total: 5 + 5 = 10 bytes
    REQUIRE(builder.total_data_bytes == 10);

    // Verify offsets
    builder.offset_blocks_accessor.set_cursor(0);
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);  // Initial
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 0);  // After empty
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 5);  // After "hello"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 5);  // After empty
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 10);  // After "world"
    builder.offset_blocks_accessor.advance();
    REQUIRE(builder.offset_blocks_accessor.get_current() == 10);  // After empty
  }

  SECTION("single row processing")
  {
    auto int_type = duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER);
    duckdb_scan_task_local_state::column_builder builder(int_type);

    builder.reserve_memory(10);
    builder.allocate_memory();

    // Process single row
    duckdb::Vector vec(int_type, 1);
    auto* data = reinterpret_cast<int32_t*>(vec.GetData());
    data[0]    = 42;

    duckdb::ValidityMask validity(1);
    validity.Initialize(1);
    validity.SetAllValid(1);

    builder.process_column(vec, validity, 1, 0);

    REQUIRE(builder.total_data_bytes == sizeof(int32_t));

    // Verify the single value
    int32_t value;
    std::memcpy(&value, builder.data_blocks_accessor.allocation->blocks[0], sizeof(int32_t));
    REQUIRE(value == 42);
  }
}
