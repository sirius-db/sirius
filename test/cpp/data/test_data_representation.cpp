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
#include <memory>
#include <vector>
#include "data/cpu_data_representation.hpp"
#include "data/gpu_data_representation.hpp"
#include "data/common.hpp"
#include "memory/null_device_memory_resource.hpp"
#include "memory/host_table.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/contiguous_split.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cuda_runtime_api.h>
#include <rmm/cuda_stream.hpp>
#include <iostream>
#include <iomanip>

#include "memory/memory_reservation.hpp"
#include "memory_management/memory_test_common.hpp"
#include "utils/cudf_test_utils.hpp"

// Declarations provided by utils/cudf_test_utils.hpp

using namespace sirius;

// Mock memory_space for testing - provides a simple memory_space without real allocators
class mock_memory_space : public memory::memory_space {
 public:
  mock_memory_space(memory::Tier tier, size_t device_id = 0)
    : memory::memory_space(memory::memory_space_id{tier, static_cast<int>(device_id)},
                           1024 * 1024 * 1024,
                           (1024ULL * 1024ULL * 1024ULL) * 8 / 10,
                           (1024ULL * 1024ULL * 1024ULL) / 2,
                           create_null_allocators())
  {
  }

 private:
  static std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> create_null_allocators()
  {
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;
    allocators.push_back(std::make_unique<memory::null_device_memory_resource>());
    return allocators;
  }
};

// Helper function to create a mock host_table_allocation for testing
sirius::unique_ptr<memory::host_table_allocation> create_mock_host_table_allocation(
  std::size_t data_size)
{
  // Create empty allocation blocks (we're not testing actual allocation here)
  // Use an empty vector and nullptr since we're just mocking
  std::vector<void*> empty_blocks;
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation empty_allocation(
    std::move(empty_blocks),
    nullptr,  // No actual memory resource in mock
    0         // Block size doesn't matter for empty allocation
  );

  // Create mock metadata
  auto metadata = sirius::make_unique<sirius::vector<uint8_t>>();
  metadata->push_back(0x01);
  metadata->push_back(0x02);
  metadata->push_back(0x03);

  return sirius::make_unique<memory::host_table_allocation>(
    std::move(empty_allocation), std::move(metadata), data_size);
}

// Helper function to create a simple cuDF table for testing
cudf::table create_simple_cudf_table(int num_rows = 100)
{
  std::vector<std::unique_ptr<cudf::column>> columns;

  // Create and initialize a simple INT32 column
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED);
  {
    auto view  = col1->mutable_view();
    auto bytes = static_cast<size_t>(num_rows) * sizeof(int32_t);
    if (bytes > 0) cudaMemset(const_cast<void*>(view.head()), 0x11, bytes);
  }

  // Create and initialize another INT64 column
  auto col2 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED);
  {
    auto view  = col2->mutable_view();
    auto bytes = static_cast<size_t>(num_rows) * sizeof(int64_t);
    if (bytes > 0) cudaMemset(const_cast<void*>(view.head()), 0x22, bytes);
  }

  columns.push_back(std::move(col1));
  columns.push_back(std::move(col2));

  return cudf::table(std::move(columns));
}

// Initialize a minimal memory manager with one GPU(0) and one HOST(0)
static void initialize_memory_for_conversions()
{
  using namespace sirius::memory;
  memory_reservation_manager::reset_for_testing();
  std::vector<memory_reservation_manager::memory_space_config> configs;
  configs.emplace_back(Tier::GPU, 0, 2048ull * 1024 * 1024, create_test_allocators(Tier::GPU));
  configs.emplace_back(Tier::HOST, 0, 4096ull * 1024 * 1024, create_test_allocators(Tier::HOST));
  memory_reservation_manager::initialize(std::move(configs));
}

// =============================================================================
// host_table_representation Tests
// =============================================================================

TEST_CASE("host_table_representation Construction", "[cpu_data_representation]")
{
  mock_memory_space host_space(memory::Tier::HOST, 0);
  auto host_table = create_mock_host_table_allocation(2048);

  host_table_representation repr(std::move(host_table), &host_space);

  REQUIRE(repr.get_current_tier() == memory::Tier::HOST);
  REQUIRE(repr.get_device_id() == 0);
  REQUIRE(repr.get_size_in_bytes() == 2048);
}

TEST_CASE("host_table_representation get_size_in_bytes", "[cpu_data_representation]")
{
  mock_memory_space host_space(memory::Tier::HOST, 0);

  SECTION("Small data size")
  {
    auto host_table = create_mock_host_table_allocation(512);
    host_table_representation repr(std::move(host_table), &host_space);

    REQUIRE(repr.get_size_in_bytes() == 512);
  }

  SECTION("Large data size")
  {
    auto host_table = create_mock_host_table_allocation(1024 * 1024);
    host_table_representation repr(std::move(host_table), &host_space);

    REQUIRE(repr.get_size_in_bytes() == 1024 * 1024);
  }

  SECTION("Zero data size")
  {
    auto host_table = create_mock_host_table_allocation(0);
    host_table_representation repr(std::move(host_table), &host_space);

    REQUIRE(repr.get_size_in_bytes() == 0);
  }
}

TEST_CASE("host_table_representation memory tier", "[cpu_data_representation]")
{
  SECTION("HOST tier")
  {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    auto host_table = create_mock_host_table_allocation(1024);
    host_table_representation repr(std::move(host_table), &host_space);

    REQUIRE(repr.get_current_tier() == memory::Tier::HOST);
  }
}

TEST_CASE("host_table_representation device_id", "[cpu_data_representation]")
{
  SECTION("Device 0")
  {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    auto host_table = create_mock_host_table_allocation(1024);
    host_table_representation repr(std::move(host_table), &host_space);

    REQUIRE(repr.get_device_id() == 0);
  }

  SECTION("Device 1")
  {
    mock_memory_space host_space(memory::Tier::HOST, 1);
    auto host_table = create_mock_host_table_allocation(1024);
    host_table_representation repr(std::move(host_table), &host_space);

    REQUIRE(repr.get_device_id() == 1);
  }
}

TEST_CASE("host_table_representation converts to GPU and preserves contents",
          "[cpu_data_representation][gpu_data_representation]")
{
  initialize_memory_for_conversions();
  auto& mgr                              = memory::memory_reservation_manager::get_instance();
  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);
  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);

  // Start from a known cudf table; pack it and build a host_table_representation
  auto original = create_simple_cudf_table(128);
  rmm::cuda_stream pack_stream;
  auto view   = original.view();
  auto packed = cudf::pack(view, pack_stream.view());
  pack_stream.synchronize();
  auto host_mr = host_space->get_default_allocator_as<memory::fixed_size_host_memory_resource>();
  REQUIRE(host_mr != nullptr);

  // Copy device buffer to host allocation
  auto allocation         = host_mr->allocate_multiple_blocks(packed.gpu_data->size());
  size_t copied           = 0;
  size_t block_idx        = 0;
  size_t block_off        = 0;
  const size_t block_size = allocation.block_size;
  while (copied < packed.gpu_data->size()) {
    size_t remain = packed.gpu_data->size() - copied;
    size_t bytes  = std::min(remain, block_size - block_off);
    void* dst_ptr = static_cast<uint8_t*>(allocation[block_idx]) + block_off;
    cudaMemcpy(dst_ptr,
               static_cast<const uint8_t*>(packed.gpu_data->data()) + copied,
               bytes,
               cudaMemcpyDeviceToHost);
    copied += bytes;
    block_off += bytes;
    if (block_off == block_size) {
      block_off = 0;
      block_idx++;
    }
  }

  auto meta_copy  = sirius::make_unique<sirius::vector<uint8_t>>(*packed.metadata);
  auto host_alloc = sirius::make_unique<memory::host_table_allocation>(
    std::move(allocation), std::move(meta_copy), packed.gpu_data->size());
  host_table_representation host_repr(std::move(host_alloc),
                                      const_cast<memory::memory_space*>(host_space));

  // Convert to GPU and compare cudf tables
  auto gpu_stream = gpu_space->acquire_stream();
  auto gpu_any    = host_repr.convert_to_memory_space(gpu_space, pack_stream);
  pack_stream.synchronize();
  auto& gpu_repr = gpu_any->cast<gpu_table_representation>();
  // Compare using the same stream used for conversion to avoid cross-stream hazards
  sirius::test::expect_cudf_tables_equal_on_stream(
    original, gpu_repr.get_table(), pack_stream.view());
  const_cast<memory::memory_space*>(gpu_space)->release_stream(gpu_stream);
}

// =============================================================================
// gpu_table_representation Tests
// =============================================================================

TEST_CASE("gpu_table_representation Construction", "[gpu_data_representation]")
{
  mock_memory_space gpu_space(memory::Tier::GPU, 0);
  auto table = create_simple_cudf_table(100);

  gpu_table_representation repr(std::move(table), gpu_space);

  REQUIRE(repr.get_current_tier() == memory::Tier::GPU);
  REQUIRE(repr.get_device_id() == 0);
  REQUIRE(repr.get_size_in_bytes() > 0);
}

TEST_CASE("gpu_table_representation get_size_in_bytes", "[gpu_data_representation]")
{
  mock_memory_space gpu_space(memory::Tier::GPU, 0);

  SECTION("100 rows")
  {
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);

    // Size should be at least 100 rows * (4 bytes for INT32 + 8 bytes for INT64)
    std::size_t expected_min_size = 100 * (4 + 8);
    REQUIRE(repr.get_size_in_bytes() >= expected_min_size);
  }

  SECTION("1000 rows")
  {
    auto table = create_simple_cudf_table(1000);
    gpu_table_representation repr(std::move(table), gpu_space);

    // Size should be at least 1000 rows * (4 bytes for INT32 + 8 bytes for INT64)
    std::size_t expected_min_size = 1000 * (4 + 8);
    REQUIRE(repr.get_size_in_bytes() >= expected_min_size);
  }

  SECTION("Empty table")
  {
    auto table = create_simple_cudf_table(0);
    gpu_table_representation repr(std::move(table), gpu_space);

    REQUIRE(repr.get_size_in_bytes() == 0);
  }
}

TEST_CASE("gpu_table_representation get_table", "[gpu_data_representation]")
{
  mock_memory_space gpu_space(memory::Tier::GPU, 0);
  auto table = create_simple_cudf_table(100);

  // Store the number of columns before moving the table
  auto num_columns = table.num_columns();

  gpu_table_representation repr(std::move(table), gpu_space);

  const cudf::table& retrieved_table = repr.get_table();
  REQUIRE(retrieved_table.num_columns() == num_columns);
  REQUIRE(retrieved_table.num_rows() == 100);
}

TEST_CASE("gpu_table_representation memory tier", "[gpu_data_representation]")
{
  SECTION("GPU tier")
  {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);

    REQUIRE(repr.get_current_tier() == memory::Tier::GPU);
  }
}

TEST_CASE("gpu_table_representation device_id", "[gpu_data_representation]")
{
  SECTION("Device 0")
  {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);

    REQUIRE(repr.get_device_id() == 0);
  }

  SECTION("Device 1")
  {
    mock_memory_space gpu_space(memory::Tier::GPU, 1);
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);

    REQUIRE(repr.get_device_id() == 1);
  }
}

TEST_CASE("gpu->host->gpu roundtrip preserves cudf table contents", "[gpu_data_representation]")
{
  initialize_memory_for_conversions();
  auto& mgr                              = memory::memory_reservation_manager::get_instance();
  const memory::memory_space* gpu_space  = mgr.get_memory_space(memory::Tier::GPU, 0);
  const memory::memory_space* host_space = mgr.get_memory_space(memory::Tier::HOST, 0);

  auto table = create_simple_cudf_table(100);
  gpu_table_representation repr(std::move(table), *const_cast<memory::memory_space*>(gpu_space));

  // Use one stream for both conversions to enforce order
  auto chain_stream = gpu_space->acquire_stream();
  auto cpu_any      = repr.convert_to_memory_space(host_space, chain_stream);
  // Debug: dump host bytes before converting back to GPU
  {
    auto& host_repr_dbg   = cpu_any->cast<host_table_representation>();
    auto host_alloc_uptr  = host_repr_dbg.get_host_table();
    const auto data_size  = host_alloc_uptr->data_size;
    const auto block_size = host_alloc_uptr->allocation.block_size;
    std::vector<uint8_t> host_bytes;
    host_bytes.resize(data_size);
    size_t copied = 0, block_idx = 0, block_off = 0;
    while (copied < data_size) {
      size_t remaining = data_size - copied;
      size_t bytes     = std::min(remaining, block_size - block_off);
      void* src_ptr    = static_cast<uint8_t*>(host_alloc_uptr->allocation[block_idx]) + block_off;
      std::memcpy(host_bytes.data() + copied, src_ptr, bytes);
      copied += bytes;
      block_off += bytes;
      if (block_off == block_size) {
        block_off = 0;
        block_idx++;
      }
    }
    auto dump_hex = [](const uint8_t* p, size_t len, size_t max_len = 64) {
      std::ostringstream oss;
      oss << std::hex << std::setfill('0');
      size_t dump_len = std::min(len, max_len);
      for (size_t i = 0; i < dump_len; ++i) {
        if (i && (i % 16 == 0)) oss << " | ";
        oss << std::setw(2) << static_cast<unsigned int>(p[i]) << ' ';
      }
      if (len > dump_len) oss << "...";
      return oss.str();
    };
    std::cout << "[roundtrip] host bytes size=" << data_size << std::endl;
    std::cout << "[roundtrip] host first 64B: " << dump_hex(host_bytes.data(), data_size)
              << std::endl;
    size_t off = 400;  // expect second column start (INT32 first column = 100*4)
    if (data_size > off) {
      size_t ctx_len = std::min(static_cast<size_t>(128), data_size - off);
      std::cout << "[roundtrip] host bytes @400 (128B): "
                << dump_hex(host_bytes.data() + off, ctx_len) << std::endl;
    }
    // Re-wrap into a host_table_representation to continue conversion
    host_table_representation host_repr2(std::move(host_alloc_uptr),
                                         const_cast<memory::memory_space*>(host_space));
    cpu_any = sirius::make_unique<host_table_representation>(std::move(host_repr2));
  }
  auto gpu_any = cpu_any->convert_to_memory_space(gpu_space, chain_stream);

  auto& back = gpu_any->cast<gpu_table_representation>();
  chain_stream.synchronize();
  // Debug: dump column device pointers and a 64B context around offset 400 right after conversion
  {
    auto tv = back.get_table().view();
    std::cout << "[roundtrip] after_conversion: columns=" << tv.num_columns()
              << " rows=" << tv.num_rows() << std::endl
              << std::flush;
    for (int i = 0; i < tv.num_columns(); ++i) {
      auto col = tv.column(i);
      std::cout << "[roundtrip] after_conversion: col[" << i << "] head=" << col.head()
                << " size=" << col.size() << " type_id=" << static_cast<int>(col.type().id())
                << std::endl
                << std::flush;
    }
    auto packed_after = cudf::pack(back.get_table(), chain_stream);
    chain_stream.synchronize();
    std::vector<uint8_t> bytes_after(packed_after.gpu_data->size());
    if (!bytes_after.empty()) {
      cudaMemcpy(bytes_after.data(),
                 packed_after.gpu_data->data(),
                 bytes_after.size(),
                 cudaMemcpyDeviceToHost);
      auto dump_hex_ctx = [](const uint8_t* p, size_t len, size_t center, size_t ctx = 64) {
        size_t start = (center > ctx / 2) ? (center - ctx / 2) : 0;
        if (start + ctx > len) ctx = (len > start) ? (len - start) : 0;
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (size_t i = 0; i < ctx; ++i) {
          size_t idx = start + i;
          if (i && (i % 16 == 0)) oss << " | ";
          if (idx < len) { oss << std::setw(2) << static_cast<unsigned int>(p[idx]) << ' '; }
        }
        return oss.str();
      };
      size_t center = std::min<size_t>(400, bytes_after.size() ? bytes_after.size() - 1 : 0);
      std::cout << "[roundtrip] after_conversion packed @400 (64B): "
                << dump_hex_ctx(bytes_after.data(), bytes_after.size(), center) << std::endl
                << std::flush;
    }
  }
  // Debug: dump again right before equality to detect intervening modification
  {
    auto packed_before_eq = cudf::pack(back.get_table(), chain_stream);
    chain_stream.synchronize();
    std::vector<uint8_t> bytes_before(packed_before_eq.gpu_data->size());
    if (!bytes_before.empty()) {
      cudaMemcpy(bytes_before.data(),
                 packed_before_eq.gpu_data->data(),
                 bytes_before.size(),
                 cudaMemcpyDeviceToHost);
      auto dump_hex_ctx = [](const uint8_t* p, size_t len, size_t center, size_t ctx = 64) {
        size_t start = (center > ctx / 2) ? (center - ctx / 2) : 0;
        if (start + ctx > len) ctx = (len > start) ? (len - start) : 0;
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (size_t i = 0; i < ctx; ++i) {
          size_t idx = start + i;
          if (i && (i % 16 == 0)) oss << " | ";
          if (idx < len) { oss << std::setw(2) << static_cast<unsigned int>(p[idx]) << ' '; }
        }
        return oss.str();
      };
      size_t center = std::min<size_t>(400, bytes_before.size() ? bytes_before.size() - 1 : 0);
      std::cout << "[roundtrip] before_equality packed @400 (64B): "
                << dump_hex_ctx(bytes_before.data(), bytes_before.size(), center) << std::endl
                << std::flush;
    }
  }
  sirius::test::expect_cudf_tables_equal_on_stream(
    repr.get_table(), back.get_table(), chain_stream);
  const_cast<memory::memory_space*>(gpu_space)->release_stream(chain_stream);
}

// =============================================================================
// Multi-GPU Cross-Device Conversion Test
// =============================================================================
static void initialize_multi_gpu_for_conversions(int dev_a, int dev_b)
{
  using namespace sirius::memory;
  memory_reservation_manager::reset_for_testing();
  std::vector<memory_reservation_manager::memory_space_config> configs;
  configs.emplace_back(Tier::GPU, dev_a, 2048ull * 1024 * 1024, create_test_allocators(Tier::GPU));
  configs.emplace_back(Tier::GPU, dev_b, 2048ull * 1024 * 1024, create_test_allocators(Tier::GPU));
  memory_reservation_manager::initialize(std::move(configs));
}

TEST_CASE("gpu cross-device conversion when multiple GPUs are available",
          "[gpu_data_representation][.multi-device]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    SUCCEED("Single GPU or CUDA not available; skipping cross-device test");
    return;
  }

  // Pick first two GPUs
  int dev_src = 0;
  int dev_dst = 1;

  initialize_multi_gpu_for_conversions(dev_src, dev_dst);
  auto& mgr                             = memory::memory_reservation_manager::get_instance();
  const memory::memory_space* src_space = mgr.get_memory_space(memory::Tier::GPU, dev_src);
  const memory::memory_space* dst_space = mgr.get_memory_space(memory::Tier::GPU, dev_dst);
  REQUIRE(src_space != nullptr);
  REQUIRE(dst_space != nullptr);

  // Build a simple cudf table on source GPU and wrap it
  auto table = create_simple_cudf_table(256);
  gpu_table_representation src_repr(std::move(table),
                                    *const_cast<memory::memory_space*>(src_space));

  // Use a single stream for the peer copy
  auto xfer_stream = src_space->acquire_stream();
  auto dst_any     = src_repr.convert_to_memory_space(dst_space, xfer_stream);
  auto& dst_repr   = dst_any->cast<gpu_table_representation>();

  // Compare content equality using the same stream used for transfer
  sirius::test::expect_cudf_tables_equal_on_stream(
    src_repr.get_table(), dst_repr.get_table(), xfer_stream);

  const_cast<memory::memory_space*>(src_space)->release_stream(xfer_stream);
}
// =============================================================================
// idata_representation Interface Tests
// =============================================================================

TEST_CASE("idata_representation cast functionality",
          "[cpu_data_representation][gpu_data_representation]")
{
  SECTION("Cast host_table_representation")
  {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    auto host_table = create_mock_host_table_allocation(1024);
    host_table_representation repr(std::move(host_table), &host_space);

    idata_representation* base_ptr = &repr;

    // Cast to derived type
    host_table_representation& casted = base_ptr->cast<host_table_representation>();
    REQUIRE(&casted == &repr);
    REQUIRE(casted.get_size_in_bytes() == 1024);
  }

  SECTION("Cast gpu_table_representation")
  {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);

    idata_representation* base_ptr = &repr;

    // Cast to derived type
    gpu_table_representation& casted = base_ptr->cast<gpu_table_representation>();
    REQUIRE(&casted == &repr);
    REQUIRE(casted.get_table().num_rows() == 100);
  }
}

TEST_CASE("idata_representation const cast functionality",
          "[cpu_data_representation][gpu_data_representation]")
{
  SECTION("Const cast host_table_representation")
  {
    mock_memory_space host_space(memory::Tier::HOST, 0);
    auto host_table = create_mock_host_table_allocation(1024);
    host_table_representation repr(std::move(host_table), &host_space);

    const idata_representation* base_ptr = &repr;

    // Const cast to derived type
    const host_table_representation& casted = base_ptr->cast<host_table_representation>();
    REQUIRE(&casted == &repr);
    REQUIRE(casted.get_size_in_bytes() == 1024);
  }

  SECTION("Const cast gpu_table_representation")
  {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);
    auto table = create_simple_cudf_table(100);
    gpu_table_representation repr(std::move(table), gpu_space);

    const idata_representation* base_ptr = &repr;

    // Const cast to derived type
    const gpu_table_representation& casted = base_ptr->cast<gpu_table_representation>();
    REQUIRE(&casted == &repr);
    REQUIRE(casted.get_table().num_rows() == 100);
  }
}

// =============================================================================
// Cross-Tier Comparison Tests
// =============================================================================

TEST_CASE("Compare CPU and GPU representations",
          "[cpu_data_representation][gpu_data_representation]")
{
  mock_memory_space host_space(memory::Tier::HOST, 0);
  mock_memory_space gpu_space(memory::Tier::GPU, 0);

  auto host_table = create_mock_host_table_allocation(1200);
  host_table_representation host_repr(std::move(host_table), &host_space);

  auto gpu_table = create_simple_cudf_table(100);
  gpu_table_representation gpu_repr(std::move(gpu_table), gpu_space);

  // Verify they have different tiers
  REQUIRE(host_repr.get_current_tier() != gpu_repr.get_current_tier());
  REQUIRE(host_repr.get_current_tier() == memory::Tier::HOST);
  REQUIRE(gpu_repr.get_current_tier() == memory::Tier::GPU);

  // Both should have valid sizes
  REQUIRE(host_repr.get_size_in_bytes() > 0);
  REQUIRE(gpu_repr.get_size_in_bytes() > 0);
}

TEST_CASE("Multiple representations on same memory space",
          "[cpu_data_representation][gpu_data_representation]")
{
  SECTION("Multiple host representations")
  {
    mock_memory_space host_space(memory::Tier::HOST, 0);

    auto host_table1 = create_mock_host_table_allocation(1024);
    host_table_representation repr1(std::move(host_table1), &host_space);

    auto host_table2 = create_mock_host_table_allocation(2048);
    host_table_representation repr2(std::move(host_table2), &host_space);

    REQUIRE(repr1.get_current_tier() == repr2.get_current_tier());
    REQUIRE(repr1.get_device_id() == repr2.get_device_id());
    REQUIRE(repr1.get_size_in_bytes() != repr2.get_size_in_bytes());
  }

  SECTION("Multiple GPU representations")
  {
    mock_memory_space gpu_space(memory::Tier::GPU, 0);

    auto table1 = create_simple_cudf_table(100);
    gpu_table_representation repr1(std::move(table1), gpu_space);

    auto table2 = create_simple_cudf_table(200);
    gpu_table_representation repr2(std::move(table2), gpu_space);

    REQUIRE(repr1.get_current_tier() == repr2.get_current_tier());
    REQUIRE(repr1.get_device_id() == repr2.get_device_id());
    // Different row counts should result in different sizes
    REQUIRE(repr1.get_size_in_bytes() != repr2.get_size_in_bytes());
  }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_CASE("gpu_table_representation with single column", "[gpu_data_representation]")
{
  mock_memory_space gpu_space(memory::Tier::GPU, 0);

  std::vector<std::unique_ptr<cudf::column>> columns;
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED);
  columns.push_back(std::move(col));

  cudf::table table(std::move(columns));
  gpu_table_representation repr(std::move(table), gpu_space);

  REQUIRE(repr.get_table().num_columns() == 1);
  REQUIRE(repr.get_table().num_rows() == 100);
  REQUIRE(repr.get_size_in_bytes() >= 100 * 4);  // At least 100 rows * 4 bytes
}

TEST_CASE("gpu_table_representation with multiple column types", "[gpu_data_representation]")
{
  mock_memory_space gpu_space(memory::Tier::GPU, 0);

  std::vector<std::unique_ptr<cudf::column>> columns;

  // INT8 column
  auto col1 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT8}, 100, cudf::mask_state::UNALLOCATED);

  // INT16 column
  auto col2 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, 100, cudf::mask_state::UNALLOCATED);

  // INT32 column
  auto col3 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 100, cudf::mask_state::UNALLOCATED);

  // INT64 column
  auto col4 = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, 100, cudf::mask_state::UNALLOCATED);

  columns.push_back(std::move(col1));
  columns.push_back(std::move(col2));
  columns.push_back(std::move(col3));
  columns.push_back(std::move(col4));

  cudf::table table(std::move(columns));
  gpu_table_representation repr(std::move(table), gpu_space);

  REQUIRE(repr.get_table().num_columns() == 4);
  REQUIRE(repr.get_table().num_rows() == 100);
  // Size should be at least 100 * (1 + 2 + 4 + 8) = 1500 bytes
  REQUIRE(repr.get_size_in_bytes() >= 1500);
}

TEST_CASE("Representations polymorphism", "[cpu_data_representation][gpu_data_representation]")
{
  mock_memory_space host_space(memory::Tier::HOST, 0);
  mock_memory_space gpu_space(memory::Tier::GPU, 0);

  // Create vector of base class pointers
  std::vector<std::unique_ptr<idata_representation>> representations;

  auto host_table = create_mock_host_table_allocation(1024);
  representations.push_back(
    std::make_unique<host_table_representation>(std::move(host_table), &host_space));

  auto gpu_table = create_simple_cudf_table(100);
  representations.push_back(
    std::make_unique<gpu_table_representation>(std::move(gpu_table), gpu_space));

  // Access through base class interface
  REQUIRE(representations[0]->get_current_tier() == memory::Tier::HOST);
  REQUIRE(representations[1]->get_current_tier() == memory::Tier::GPU);

  REQUIRE(representations[0]->get_size_in_bytes() == 1024);
  REQUIRE(representations[1]->get_size_in_bytes() > 0);
}
