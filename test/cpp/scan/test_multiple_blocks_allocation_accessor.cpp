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
#include <scan/duckdb_scan_task.hpp>

using namespace sirius::parallel;
using namespace sirius::memory;

//===----------------------------------------------------------------------===//
// Test: multiple_blocks_allocation_accessor - Basic Operations
//===----------------------------------------------------------------------===//

TEST_CASE("multiple_blocks_allocation_accessor - basic operations", "[duckdb_scan_task][accessor]")
{
  // Create a real memory resource for testing
  auto mr = sirius::make_unique<fixed_size_host_memory_resource>(1024,  // block_size
                                                                 16,    // pool_size
                                                                 1      // initial_pools
  );

  using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;

  SECTION("set_cursor and get_current")
  {
    accessor_type accessor;

    // Allocate 2 blocks using the real memory resource
    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));

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

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Set and get value
    accessor.set_current(42);
    REQUIRE(accessor.get_current() == 42);

    accessor.set_current(255);
    REQUIRE(accessor.get_current() == 255);
  }

  SECTION("advance operation")
  {
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
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

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Copy data
    uint8_t data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    accessor.memcpy_from(data, 10);

    // Verify
    auto* block_data = static_cast<uint8_t*>(accessor.allocation->blocks[0]);
    for (size_t i = 0; i < 10; ++i) {
      REQUIRE(block_data[i] == data[i]);
    }
  }

  SECTION("memcpy_from across multiple blocks")
  {
    // Use smaller block size for this test
    auto small_mr = sirius::make_unique<fixed_size_host_memory_resource>(64, 16, 1);
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), small_mr.get(), small_mr->get_block_size());

    accessor.initialize(std::move(allocation));

    // Start near end of first block
    accessor.set_cursor(60);

    // Copy 10 bytes (should span two blocks)
    uint8_t data[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    accessor.memcpy_from(data, 10);

    // Verify first block
    auto* block0_data = static_cast<uint8_t*>(accessor.allocation->blocks[0]);
    REQUIRE(block0_data[60] == 10);
    REQUIRE(block0_data[61] == 11);
    REQUIRE(block0_data[62] == 12);
    REQUIRE(block0_data[63] == 13);

    // Verify second block
    auto* block1_data = static_cast<uint8_t*>(accessor.allocation->blocks[1]);
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
  auto mr             = sirius::make_unique<fixed_size_host_memory_resource>(1024, 16, 1);
  using accessor_type = multiple_blocks_allocation_accessor<int64_t>;

  SECTION("set and get int64_t values")
  {
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Set and get value
    accessor.set_current(123456789LL);
    REQUIRE(accessor.get_current() == 123456789LL);

    // Test advance
    accessor.advance();
    accessor.set_current(-987654321LL);
    REQUIRE(accessor.get_current() == -987654321LL);
  }

  SECTION("advance across block boundary")
  {
    // 64-byte blocks hold 8 int64_t values
    auto small_mr = sirius::make_unique<fixed_size_host_memory_resource>(64, 16, 1);
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), small_mr.get(), small_mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Fill first block (8 int64_t values)
    for (int i = 0; i < 8; ++i) {
      accessor.set_current(i * 100LL);
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
  auto mr = sirius::make_unique<fixed_size_host_memory_resource>(1024, 16, 1);

  SECTION("initialize with misaligned type size - should throw")
  {
    // Using a type that doesn't divide evenly into block size
    // Block size is 1024, so a 96-byte type won't align (1024 % 96 != 0)
    struct MisalignedType {
      uint8_t data[96];
    };

    using accessor_type = multiple_blocks_allocation_accessor<MisalignedType>;
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    // This should throw because 1024 % 96 != 0
    REQUIRE_THROWS_AS(accessor.initialize(std::move(allocation)), duckdb::InternalException);
  }

  SECTION("cursor at last valid position in block")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));

    // Set cursor to last byte of first block
    accessor.set_cursor(1023);
    REQUIRE(accessor.block_index == 0);
    REQUIRE(accessor.offset_in_block == 1023);

    accessor.set_current(99);
    REQUIRE(accessor.get_current() == 99);

    // Advance should move to second block
    accessor.advance();
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);
  }

  SECTION("memcpy_from exactly fills single block")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(mr->allocate(mr->get_block_size()));
    blocks.push_back(mr->allocate(mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Create data that exactly fills the block
    std::vector<uint8_t> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(i % 256);
    }

    accessor.memcpy_from(data.data(), 1024);

    // Should be at start of second block
    REQUIRE(accessor.block_index == 1);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify data in first block
    auto* block_data = static_cast<uint8_t*>(accessor.allocation->blocks[0]);
    for (size_t i = 0; i < 1024; ++i) {
      REQUIRE(block_data[i] == static_cast<uint8_t>(i % 256));
    }
  }

  SECTION("memcpy_from exactly fills multiple blocks")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    
    // IMPORTANT: Declare small_mr BEFORE accessor so it destructs AFTER accessor
    // The accessor holds a pointer to small_mr, so small_mr must outlive accessor
    auto small_mr = sirius::make_unique<fixed_size_host_memory_resource>(64, 16, 1);
    accessor_type accessor;

    std::vector<void*> blocks;
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    blocks.push_back(small_mr->allocate(small_mr->get_block_size()));

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), small_mr.get(), small_mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Copy exactly 2 blocks worth of data (128 bytes)
    std::vector<uint8_t> data(128);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(i);
    }

    accessor.memcpy_from(data.data(), 128);

    // Should be at start of third block (index 2)
    REQUIRE(accessor.block_index == 2);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify data in first two blocks using the allocation's blocks vector
    auto* block0_data = static_cast<uint8_t*>(accessor.allocation->blocks[0]);
    auto* block1_data = static_cast<uint8_t*>(accessor.allocation->blocks[1]);

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
  auto small_mr = sirius::make_unique<fixed_size_host_memory_resource>(32, 16, 1);

  SECTION("advance through 4 blocks with int32_t")
  {
    using accessor_type = multiple_blocks_allocation_accessor<int32_t>;
    accessor_type accessor;

    std::vector<void*> blocks;
    for (int i = 0; i < 5; ++i) {  // Allocate 5 blocks to handle the final position
      blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    }

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), small_mr.get(), small_mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Each 32-byte block holds 8 int32_t values
    // Write 32 values across 4 blocks
    for (int i = 0; i < 32; ++i) {
      accessor.set_current(i * 10);
      accessor.advance();
    }

    // Should be at start of block 4 (we allocated 5 blocks so this is valid)
    REQUIRE(accessor.block_index == 4);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify some values by resetting cursor to valid positions
    accessor.set_cursor(0);
    REQUIRE(accessor.get_current() == 0);

    accessor.set_cursor(4);  // Second value
    REQUIRE(accessor.get_current() == 10);

    accessor.set_cursor(32);  // First value in second block
    REQUIRE(accessor.get_current() == 80);

    accessor.set_cursor(96);  // First value in fourth block
    REQUIRE(accessor.get_current() == 240);
  }

  SECTION("memcpy_from spanning 5 blocks")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    std::vector<void*> blocks;
    for (int i = 0; i < 5; ++i) {
      blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    }

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), small_mr.get(), small_mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(10);  // Start at offset 10 in first block

    // Copy 130 bytes - spans from middle of block 0 through end of block 4
    std::vector<uint8_t> data(130);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    accessor.memcpy_from(data.data(), 130);

    // Verify cursor position (10 + 130 = 140 = 4*32 + 12)
    REQUIRE(accessor.block_index == 4);
    REQUIRE(accessor.offset_in_block == 12);

    // Verify some data points
    auto* block0 = static_cast<uint8_t*>(accessor.allocation->blocks[0]);
    auto* block1 = static_cast<uint8_t*>(accessor.allocation->blocks[1]);
    auto* block4 = static_cast<uint8_t*>(accessor.allocation->blocks[4]);

    REQUIRE(block0[10] == 0);    // First byte of copy
    REQUIRE(block0[31] == 21);   // Last byte of block 0
    REQUIRE(block1[0] == 22);    // First byte of block 1
    REQUIRE(block4[11] == 129);  // Last byte of copy
  }

  SECTION("set_cursor to various positions across multiple blocks")
  {
    using accessor_type = multiple_blocks_allocation_accessor<uint16_t>;
    accessor_type accessor;

    std::vector<void*> blocks;
    for (int i = 0; i < 6; ++i) {
      blocks.push_back(small_mr->allocate(small_mr->get_block_size()));
    }

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), small_mr.get(), small_mr->get_block_size());

    accessor.initialize(std::move(allocation));

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
    auto mr             = sirius::make_unique<fixed_size_host_memory_resource>(256, 64, 1);
    using accessor_type = multiple_blocks_allocation_accessor<uint8_t>;
    accessor_type accessor;

    // Allocate 20 blocks
    std::vector<void*> blocks;
    for (int i = 0; i < 20; ++i) {
      blocks.push_back(mr->allocate(mr->get_block_size()));
    }

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Copy 4000 bytes (spans ~15.6 blocks)
    std::vector<uint8_t> data(4000);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<uint8_t>((i * 7) & 0xFF);  // Some pattern
    }

    accessor.memcpy_from(data.data(), 4000);

    // Verify cursor (4000 = 15*256 + 160)
    REQUIRE(accessor.block_index == 15);
    REQUIRE(accessor.offset_in_block == 160);

    // Spot check some values across different blocks
    auto* block0  = static_cast<uint8_t*>(accessor.allocation->blocks[0]);
    auto* block5  = static_cast<uint8_t*>(accessor.allocation->blocks[5]);
    auto* block10 = static_cast<uint8_t*>(accessor.allocation->blocks[10]);
    auto* block15 = static_cast<uint8_t*>(accessor.allocation->blocks[15]);

    REQUIRE(block0[0] == 0);
    REQUIRE(block0[100] == static_cast<uint8_t>((100 * 7) & 0xFF));
    REQUIRE(block5[50] == static_cast<uint8_t>((1330 * 7) & 0xFF));    // 5*256 + 50 = 1330
    REQUIRE(block10[200] == static_cast<uint8_t>((2760 * 7) & 0xFF));  // 10*256 + 200 = 2760
    REQUIRE(block15[159] == static_cast<uint8_t>((3999 * 7) & 0xFF));  // Last byte
  }

  SECTION("many sequential advances")
  {
    auto mr             = sirius::make_unique<fixed_size_host_memory_resource>(128, 32, 1);
    using accessor_type = multiple_blocks_allocation_accessor<int64_t>;
    accessor_type accessor;

    // Allocate 11 blocks (each holds 16 int64_t values) - extra block for final position
    std::vector<void*> blocks;
    for (int i = 0; i < 11; ++i) {
      blocks.push_back(mr->allocate(mr->get_block_size()));
    }

    auto allocation =
      sirius::make_unique<fixed_size_host_memory_resource::multiple_blocks_allocation>(
        std::move(blocks), mr.get(), mr->get_block_size());

    accessor.initialize(std::move(allocation));
    accessor.set_cursor(0);

    // Write 160 values (spans 10 blocks exactly, ends at block 10 offset 0)
    for (int64_t i = 0; i < 160; ++i) {
      accessor.set_current(i * i);  // Store i^2
      accessor.advance();
    }

    // Should be at start of 11th block (index 10, which we allocated)
    REQUIRE(accessor.block_index == 10);
    REQUIRE(accessor.offset_in_block == 0);

    // Verify some values by setting cursor to valid positions
    accessor.set_cursor(0);
    REQUIRE(accessor.get_current() == 0);

    accessor.set_cursor(8 * 50);  // 50th value
    REQUIRE(accessor.get_current() == 2500);

    accessor.set_cursor(8 * 100);  // 100th value
    REQUIRE(accessor.get_current() == 10000);

    accessor.set_cursor(8 * 159);  // Last value
    REQUIRE(accessor.get_current() == 25281);
  }
}
