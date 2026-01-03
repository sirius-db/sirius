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

// sirius
#include <memory/fixed_size_host_memory_resource.hpp>
#include <memory/memory_reservation_manager.hpp>
#include <memory/numa_region_pinned_host_allocator.hpp>
#include <scan/duckdb_scan_task.hpp>

using namespace sirius::op::scan;
using namespace cucascade::memory;

// Standalone memory resource for testing with custom block size
static std::unique_ptr<numa_region_pinned_host_memory_resource> g_upstream_mr;
static std::unique_ptr<fixed_size_host_memory_resource> g_test_mr;

/**
 * @brief Initialize the test memory resource with custom block size
 */
static void initialize_test_memory_resource()
{
  static bool initialized = false;
  if (!initialized) {
    g_upstream_mr = std::make_unique<numa_region_pinned_host_memory_resource>(0);
    g_test_mr     = std::make_unique<fixed_size_host_memory_resource>(
      0,                     // device_id
      *g_upstream_mr,        // upstream allocator
      100ULL * 1024 * 1024,  // memory_limit (100MB)
      100ULL * 1024 * 1024,  // capacity (100MB)
      1024,                  // block_size = 1KB for testing
      16,                    // pool_size
      1                      // initial_pools
    );
    initialized = true;
  }
}

/**
 * @brief Helper to create a multi-block allocation for testing
 */
static std::unique_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>
create_test_allocation(size_t total_size)
{
  initialize_test_memory_resource();
  return g_test_mr->allocate_multiple_blocks(total_size);
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - Basic Operations
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - basic operations", "[duckdb_scan_task][accessor]")
{
  using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;

  SECTION("set_cursor and get_current")
  {
    accessor_type accessor;

    // Allocate 2 blocks (2KB total with 1KB blocks)
    auto allocation = create_test_allocation(2048);

    accessor.initialize(0, allocation);

    // Test cursor at the beginning
    accessor.set_cursor(0);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 0);

    // Test cursor in the middle of first block
    accessor.set_cursor(512);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 512);

    // Test cursor at the beginning of second block
    accessor.set_cursor(1024);
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }

  SECTION("set_current and get_current")
  {
    accessor_type accessor;

    auto allocation = create_test_allocation(1024);

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Set and get value
    accessor.set_current(42, allocation);
    REQUIRE(accessor.get_current(allocation) == 42);

    accessor.set_current(255, allocation);
    REQUIRE(accessor.get_current(allocation) == 255);
  }

  SECTION("advance operation")
  {
    accessor_type accessor;

    auto allocation = create_test_allocation(2048);

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Advance within first block
    accessor.advance();
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 1);

    // Advance to near end of first block
    accessor.set_cursor(1023);
    accessor.advance();
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }

  SECTION("memcpy_from within single block")
  {
    accessor_type accessor;

    auto allocation = create_test_allocation(1024);

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Copy data
    uint8_t data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    accessor.memcpy_from(data, 10, allocation);

    // Verify
    auto* block_data = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);
    for (size_t i = 0; i < 10; ++i) {
      REQUIRE(block_data[i] == data[i]);
    }
  }

  SECTION("memcpy_from across multiple blocks")
  {
    accessor_type accessor;

    // Allocate enough to span 2 blocks (with 1KB blocks, need > 1024 bytes)
    auto allocation = create_test_allocation(2048);

    // Start near end of first block
    accessor.initialize(1020, allocation);
    accessor.set_cursor(1020);

    // Copy 10 bytes (should span two blocks: 4 in first block, 6 in second)
    uint8_t data[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    accessor.memcpy_from(data, 10, allocation);

    // Verify first block (last 4 bytes)
    auto* block0_data = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);
    REQUIRE(block0_data[1020] == 10);
    REQUIRE(block0_data[1021] == 11);
    REQUIRE(block0_data[1022] == 12);
    REQUIRE(block0_data[1023] == 13);

    // Verify second block (first 6 bytes)
    auto* block1_data = reinterpret_cast<uint8_t*>(allocation->get_blocks()[1]);
    REQUIRE(block1_data[0] == 14);
    REQUIRE(block1_data[1] == 15);
    REQUIRE(block1_data[2] == 16);
    REQUIRE(block1_data[3] == 17);
    REQUIRE(block1_data[4] == 18);
    REQUIRE(block1_data[5] == 19);
  }
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - Typed Operations
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - int64_t type", "[duckdb_scan_task][accessor]")
{
  using accessor_type = multiple_blocks_allocation_accessor<int64_t>;

  SECTION("set and get int64_t values")
  {
    accessor_type accessor;

    auto allocation = create_test_allocation(1024);

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Set and get value
    accessor.set_current(123456789LL, allocation);
    REQUIRE(accessor.get_current(allocation) == 123456789LL);

    // Test advance
    accessor.advance();
    accessor.set_current(-987654321LL, allocation);
    REQUIRE(accessor.get_current(allocation) == -987654321LL);
  }

  SECTION("advance across block boundary")
  {
    accessor_type accessor;

    // Create an allocation with at least two blocks. We don't assume a specific
    // block size; compute number of int64_t values per block dynamically.
    auto allocation = create_test_allocation(2 * 1024);

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    const auto values_per_block = static_cast<int>(allocation->block_size() / sizeof(int64_t));

    // Fill first block with values_per_block entries
    for (int i = 0; i < values_per_block; ++i) {
      accessor.set_current(i * 100LL, allocation);
      accessor.advance();
    }

    // Should now be at the start of second block
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - Edge Cases
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - edge cases", "[duckdb_scan_task][accessor]")
{
  // Create an upstream pinned host resource on the stack
  numa_region_pinned_host_memory_resource upstream_mr(0);
  auto mr = std::make_unique<fixed_size_host_memory_resource>(
    0, upstream_mr, 1024 * 16 * 1, 1024 * 16 * 1, 1024, 16, 1);

  SECTION("initialize with misaligned type size - should throw")
  {
    // Using a type that doesn't divide evenly into block size
    // Block size is 1024, so a 96-byte type won't align (1024 % 96 != 0)
    struct MisalignedType {
      uint8_t data[96];
    };

    using accessor_type = multiple_blocks_allocation_accessor<MisalignedType>;
    accessor_type accessor;

    // Allocate 1 block using allocate_multiple_blocks
    auto allocation = mr->allocate_multiple_blocks(mr->get_block_size());

    // This should throw because 1024 % 96 != 0
    REQUIRE_THROWS_AS(accessor.initialize(0, allocation), std::runtime_error);
  }

  SECTION("cursor at last valid position in block")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    // Allocate 2 blocks
    auto allocation = mr->allocate_multiple_blocks(2 * mr->get_block_size());

    accessor.initialize(0, allocation);

    // Set cursor to last byte of first block
    accessor.set_cursor(1023);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 1023);

    accessor.set_current(99, allocation);
    REQUIRE(accessor.get_current(allocation) == 99);

    // Advance should move to second block
    accessor.advance();
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }

  SECTION("memcpy_from exactly fills single block")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    // Allocate 2 blocks
    auto allocation = mr->allocate_multiple_blocks(2 * mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Create data that exactly fills the block
    std::vector<uint8_t> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(i % 256);
    }

    accessor.memcpy_from(data.data(), 1024, allocation);

    // Should be at start of second block
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify data in first block
    auto* block_data = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);
    for (size_t i = 0; i < 1024; ++i) {
      REQUIRE(block_data[i] == static_cast<uint8_t>(i % 256));
    }
  }

  SECTION("memcpy_from exactly fills multiple blocks")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;

    // IMPORTANT: Declare small_mr BEFORE accessor so it destructs AFTER accessor
    // The accessor holds a pointer to small_mr, so small_mr must outlive accessor
    numa_region_pinned_host_memory_resource upstream_mr2(0);
    auto small_mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr2, 64 * 16 * 1, 64 * 16 * 1, 64, 16, 1);
    accessor_type accessor;

    // Allocate 3 blocks of 64 bytes each (192 bytes total)
    auto allocation = small_mr->allocate_multiple_blocks(3 * small_mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Copy exactly 2 blocks worth of data (128 bytes)
    std::vector<uint8_t> data(128);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(i);
    }

    accessor.memcpy_from(data.data(), 128, allocation);

    // Should be at start of third block (index 2)
    REQUIRE(accessor.block_index == 2);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify data in first two blocks using the allocation's blocks vector
    auto* block0_data = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);
    auto* block1_data = reinterpret_cast<uint8_t*>(allocation->get_blocks()[1]);

    for (size_t i = 0; i < 64; ++i) {
      REQUIRE(block0_data[i] == static_cast<uint8_t>(i));
      REQUIRE(block1_data[i] == static_cast<uint8_t>(i + 64));
    }
  }
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - Multi-Block Traversal
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - multi-block traversal",
          "[duckdb_scan_task][accessor]")
{
  // Use 32-byte blocks for easier testing
  numa_region_pinned_host_memory_resource upstream_mr(0);
  auto small_mr = std::make_unique<fixed_size_host_memory_resource>(
    0, upstream_mr, 32 * 16 * 1, 32 * 16 * 1, 32, 16, 1);

  SECTION("advance through 4 blocks with int32_t")
  {
    using accessor_type = multiple_blocks_allocation_accessor<int32_t>;
    accessor_type accessor;

    // Allocate 5 blocks of 32 bytes each (160 bytes total)
    auto allocation = small_mr->allocate_multiple_blocks(5 * small_mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Each 32-byte block holds 8 int32_t values
    // Write 32 values across 4 blocks
    for (int i = 0; i < 32; ++i) {
      accessor.set_current(i * 10, allocation);
      accessor.advance();
    }

    // Should be at start of block 4 (we allocated 5 blocks so this is valid)
    REQUIRE(accessor.block_index == 4);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify some values by resetting cursor to valid positions
    accessor.set_cursor(0);
    REQUIRE(accessor.get_current(allocation) == 0);

    accessor.set_cursor(4);  // Second value
    REQUIRE(accessor.get_current(allocation) == 10);

    accessor.set_cursor(32);  // First value in second block
    REQUIRE(accessor.get_current(allocation) == 80);

    accessor.set_cursor(96);  // First value in fourth block
    REQUIRE(accessor.get_current(allocation) == 240);
  }

  SECTION("memcpy_from spanning 5 blocks")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = small_mr->allocate_multiple_blocks(5 * small_mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(10);  // Start at offset 10 in first block

    // Copy 130 bytes - spans from middle of block 0 through end of block 4
    std::vector<uint8_t> data(130);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    accessor.memcpy_from(data.data(), 130, allocation);

    // Verify cursor position (10 + 130 = 140 = 4*32 + 12)
    REQUIRE(accessor.block_index == 4);
    REQUIRE(accessor.offset_in_block == 12);

    // Verify some data points
    auto* block0 = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);
    auto* block1 = reinterpret_cast<uint8_t*>(allocation->get_blocks()[1]);
    auto* block4 = reinterpret_cast<uint8_t*>(allocation->get_blocks()[4]);

    REQUIRE(block0[10] == 0);    // First byte of copy
    REQUIRE(block0[31] == 21);   // Last byte of block 0
    REQUIRE(block1[0] == 22);    // First byte of block 1
    REQUIRE(block4[11] == 129);  // Last byte of copy
  }

  SECTION("set_cursor to various positions across multiple blocks")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint16_t>;
    accessor_type accessor;

    auto allocation = small_mr->allocate_multiple_blocks(6 * small_mr->get_block_size());

    accessor.initialize(0, allocation);

    // Test various cursor positions
    accessor.set_cursor(0);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 0);

    accessor.set_cursor(64);  // Exactly at block 2
    REQUIRE(accessor.block_index == 2);
    REQUIRE(accessor.offset_in_block == 0);

    accessor.set_cursor(100);  // Middle of block 3
    REQUIRE(accessor.block_index == 3);
    REQUIRE(accessor.offset_in_block == 4);

    accessor.set_cursor(160);  // Exactly at block 5
    REQUIRE(accessor.block_index == 5);
    REQUIRE(accessor.offset_in_block == 0);

    accessor.set_cursor(175);  // Near end of block 5
    REQUIRE(accessor.block_index == 5);
    REQUIRE(accessor.offset_in_block == 15);
  }
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - Large Operations
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - large operations", "[duckdb_scan_task][accessor]")
{
  SECTION("large memcpy_from across many blocks")
  {
    // Use 256-byte blocks
    numa_region_pinned_host_memory_resource upstream_mr(0);
    auto mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr, 256 * 64 * 1, 256 * 64 * 1, 256, 64, 1);
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    // Allocate 20 blocks
    auto allocation = mr->allocate_multiple_blocks(20 * mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Copy 4000 bytes (spans ~15.6 blocks)
    std::vector<uint8_t> data(4000);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>((i * 7) & 0xFF);  // Some pattern
    }

    accessor.memcpy_from(data.data(), 4000, allocation);

    // Verify cursor (4000 = 15*256 + 160)
    REQUIRE(accessor.block_index == 15);
    REQUIRE(accessor.offset_in_block == 160);

    // Spot check some values across different blocks
    auto* block0  = reinterpret_cast<uint8_t*>(allocation->get_blocks()[0]);
    auto* block5  = reinterpret_cast<uint8_t*>(allocation->get_blocks()[5]);
    auto* block10 = reinterpret_cast<uint8_t*>(allocation->get_blocks()[10]);
    auto* block15 = reinterpret_cast<uint8_t*>(allocation->get_blocks()[15]);

    REQUIRE(block0[0] == 0);
    REQUIRE(block0[100] == static_cast<uint8_t>((100 * 7) & 0xFF));
    REQUIRE(block5[50] == static_cast<uint8_t>((1330 * 7) & 0xFF));    // 5*256 + 50 = 1330
    REQUIRE(block10[200] == static_cast<uint8_t>((2760 * 7) & 0xFF));  // 10*256 + 200 = 2760
    REQUIRE(block15[159] == static_cast<uint8_t>((3999 * 7) & 0xFF));  // Last byte
  }

  SECTION("many sequential advances")
  {
    numa_region_pinned_host_memory_resource upstream_mr(0);
    auto mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr, 128 * 32 * 1, 128 * 32 * 1, 128, 32, 1);
    using accessor_type = multiple_blocks_allocation_accessor<int64_t>;
    accessor_type accessor;

    // Allocate 11 blocks (each holds 16 int64_t values) - extra block for final position
    auto allocation = mr->allocate_multiple_blocks(11 * mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Write 160 values (spans 10 blocks exactly, ends at block 10 offset 0)
    for (int64_t i = 0; i < 160; ++i) {
      accessor.set_current(i * i, allocation);  // Store i^2
      accessor.advance();
    }

    // Should be at start of 11th block (index 10, which we allocated)
    REQUIRE(accessor.block_index == 10);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify some values by setting cursor to valid positions
    accessor.set_cursor(0);
    REQUIRE(accessor.get_current(allocation) == 0);

    accessor.set_cursor(8 * 50);  // 50th value
    REQUIRE(accessor.get_current(allocation) == 2500);

    accessor.set_cursor(8 * 100);  // 100th value
    REQUIRE(accessor.get_current(allocation) == 10000);

    accessor.set_cursor(8 * 159);  // Last value
    REQUIRE(accessor.get_current(allocation) == 25281);
  }
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - get_current_as and advance_as
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - get_current_as and advance_as",
          "[duckdb_scan_task][accessor]")
{
  numa_region_pinned_host_memory_resource upstream_mr(0);
  auto mr = std::make_unique<fixed_size_host_memory_resource>(
    0, upstream_mr, 1024 * 16 * 1, 1024 * 16 * 1, 1024, 16, 1);

  SECTION("get_current_as - reading uint8_t as different types")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Write some bytes that form a known int32_t value (little-endian: 0x04030201)
    accessor.set_current(0x01, allocation);
    accessor.advance();
    accessor.set_current(0x02, allocation);
    accessor.advance();
    accessor.set_current(0x03, allocation);
    accessor.advance();
    accessor.set_current(0x04, allocation);

    // Read as int32_t
    accessor.set_cursor(0);
    auto val = accessor.get_current_as<int32_t>(allocation);
    REQUIRE(val == 0x04030201);

    // Read as int16_t (first 2 bytes)
    accessor.set_cursor(0);
    auto val16 = accessor.get_current_as<int16_t>(allocation);
    REQUIRE(val16 == 0x0201);
  }

  SECTION("advance_as with different type sizes")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(2 * mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Advance as int32_t (4 bytes at a time)
    REQUIRE(accessor.offset_in_block == 0);
    accessor.advance_as<int32_t>();
    REQUIRE(accessor.offset_in_block == 4);
    accessor.advance_as<int32_t>();
    REQUIRE(accessor.offset_in_block == 8);

    // Advance as int64_t (8 bytes at a time)
    accessor.advance_as<int64_t>();
    REQUIRE(accessor.offset_in_block == 16);
    accessor.advance_as<int64_t>();
    REQUIRE(accessor.offset_in_block == 24);

    // Advance as int16_t (2 bytes at a time)
    accessor.advance_as<int16_t>();
    REQUIRE(accessor.offset_in_block == 26);
  }

  SECTION("advance_as across block boundary")
  {
    numa_region_pinned_host_memory_resource upstream_mr2(0);
    auto small_mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr2, 64 * 16 * 1, 64 * 16 * 1, 64, 16, 1);
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = small_mr->allocate_multiple_blocks(2 * small_mr->get_block_size());

    accessor.initialize(0, allocation);

    // Position at byte 60 (4 bytes from end of first block)
    accessor.set_cursor(60);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 60);

    // Advance by 4 bytes - should land exactly at start of second block
    accessor.advance_as<int32_t>();
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - memcpy_to
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - memcpy_to", "[duckdb_scan_task][accessor]")
{
  numa_region_pinned_host_memory_resource upstream_mr(0);
  auto mr = std::make_unique<fixed_size_host_memory_resource>(
    0, upstream_mr, 1024 * 16 * 1, 1024 * 16 * 1, 1024, 16, 1);

  SECTION("memcpy_to within single block")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(mr->get_block_size());

    accessor.initialize(0, allocation);
    accessor.set_cursor(0);

    // Write some data using memcpy_from
    uint8_t src_data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    accessor.memcpy_from(src_data, 10, allocation);

    // Read it back using memcpy_to
    accessor.set_cursor(0);
    uint8_t dest_data[10];
    accessor.memcpy_to(allocation, dest_data, 10);

    // Verify
    for (size_t i = 0; i < 10; ++i) {
      REQUIRE(dest_data[i] == src_data[i]);
    }
  }

  SECTION("memcpy_to across multiple blocks")
  {
    numa_region_pinned_host_memory_resource upstream_mr2(0);
    auto small_mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr2, 64 * 16 * 1, 64 * 16 * 1, 64, 16, 1);
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = small_mr->allocate_multiple_blocks(3 * small_mr->get_block_size());

    accessor.initialize(0, allocation);

    // Write data spanning multiple blocks
    std::vector<uint8_t> src_data(150);
    for (size_t i = 0; i < src_data.size(); ++i) {
      src_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    accessor.set_cursor(0);
    accessor.memcpy_from(src_data.data(), 150, allocation);

    // Read it back
    accessor.set_cursor(0);
    std::vector<uint8_t> dest_data(150);
    accessor.memcpy_to(allocation, dest_data.data(), 150);

    // Verify
    for (size_t i = 0; i < 150; ++i) {
      REQUIRE(dest_data[i] == src_data[i]);
    }

    // Verify cursor position (0 + 150 = 150 = 2*64 + 22)
    REQUIRE(accessor.block_index == 2);
    REQUIRE(accessor.offset_in_block == 22);
  }

  SECTION("memcpy_to from middle of block")
  {
    numa_region_pinned_host_memory_resource upstream_mr3(0);
    auto small_mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr3, 64 * 16 * 1, 64 * 16 * 1, 64, 16, 1);
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = small_mr->allocate_multiple_blocks(2 * small_mr->get_block_size());

    accessor.initialize(0, allocation);

    // Write data starting from position 50
    accessor.set_cursor(50);
    uint8_t src_data[30] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    accessor.memcpy_from(src_data, 30, allocation);

    // Read back from position 50
    accessor.set_cursor(50);
    uint8_t dest_data[30];
    accessor.memcpy_to(allocation, dest_data, 30);

    // Verify
    for (size_t i = 0; i < 30; ++i) {
      REQUIRE(dest_data[i] == src_data[i]);
    }
  }

  SECTION("memcpy_to exactly at block boundaries")
  {
    numa_region_pinned_host_memory_resource upstream_mr4(0);
    auto small_mr = std::make_unique<fixed_size_host_memory_resource>(
      0, upstream_mr4, 64 * 16 * 1, 64 * 16 * 1, 64, 16, 1);
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    auto allocation = small_mr->allocate_multiple_blocks(4 * small_mr->get_block_size());

    accessor.initialize(0, allocation);

    // Write exactly 2 blocks worth of data (128 bytes)
    std::vector<uint8_t> src_data(128);
    for (size_t i = 0; i < 128; ++i) {
      src_data[i] = static_cast<uint8_t>((i * 3) & 0xFF);
    }

    accessor.set_cursor(0);
    accessor.memcpy_from(src_data.data(), 128, allocation);

    // Read back exactly 2 blocks
    accessor.set_cursor(0);
    std::vector<uint8_t> dest_data(128);
    accessor.memcpy_to(allocation, dest_data.data(), 128);

    // Verify all data
    for (size_t i = 0; i < 128; ++i) {
      REQUIRE(dest_data[i] == src_data[i]);
    }

    // Should be at start of block 2
    REQUIRE(accessor.block_index == 2);
    REQUIRE(accessor.offset_in_block == 0);
  }
}

//===----------------------------------------------------------------------===//
// Test: reset_cursor functionality
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - reset_cursor", "[duckdb_scan_task][accessor]")
{
  numa_region_pinned_host_memory_resource upstream_mr(0);
  auto mr = std::make_unique<fixed_size_host_memory_resource>(
    0, upstream_mr, 1024 * 16 * 1, 1024 * 16 * 1, 1024, 16, 1);
  using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;

  SECTION("reset_cursor restores initial position")
  {
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(2 * mr->get_block_size());

    // Initialize at byte offset 0
    accessor.initialize(0, allocation);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 0);

    // Advance several times
    for (int i = 0; i < 100; ++i) {
      accessor.advance();
    }
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 100);

    // Reset cursor should restore to initial position (0)
    accessor.reset_cursor();
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 0);
  }

  SECTION("reset_cursor with non-zero initial byte offset")
  {
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(3 * mr->get_block_size());

    // Initialize at byte offset 500
    accessor.initialize(500, allocation);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 500);

    // Advance to cross block boundary
    for (int i = 0; i < 600; ++i) {
      accessor.advance();
    }
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 76);  // 500 + 600 - 1024 = 76

    // Reset should restore to initial position (500)
    accessor.reset_cursor();
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 500);
  }

  SECTION("reset_cursor after memcpy operations")
  {
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(4 * mr->get_block_size());

    // Initialize at byte offset 200
    accessor.initialize(200, allocation);

    // Write some data
    std::vector<uint8_t> data(500, 42);
    accessor.memcpy_from(data.data(), 500, allocation);

    // Cursor should have advanced
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 700);

    // Reset and verify
    accessor.reset_cursor();
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 200);

    // Read data should work correctly after reset
    std::vector<uint8_t> read_data(500);
    accessor.memcpy_to(allocation, read_data.data(), 500);
    for (size_t i = 0; i < 500; ++i) {
      REQUIRE(read_data[i] == 42);
    }
  }

  SECTION("reset_cursor with initial offset at block boundary")
  {
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(2 * mr->get_block_size());

    // Initialize at block boundary (1024)
    accessor.initialize(1024, allocation);
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);

    // Set cursor elsewhere
    accessor.set_cursor(500);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 500);

    // Reset should restore to block boundary
    accessor.reset_cursor();
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }

  SECTION("multiple reset_cursor calls are idempotent")
  {
    accessor_type accessor;

    auto allocation = mr->allocate_multiple_blocks(mr->get_block_size());

    accessor.initialize(100, allocation);

    // Advance then reset multiple times
    for (int iteration = 0; iteration < 3; ++iteration) {
      for (int i = 0; i < 50; ++i) {
        accessor.advance();
      }
      accessor.reset_cursor();
      REQUIRE(accessor.block_index == 0);
      REQUIRE(accessor.offset_in_block == 100);
    }
  }
}
