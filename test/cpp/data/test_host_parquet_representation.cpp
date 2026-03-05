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

// sirius
#include "cudf/cudf_utils.hpp"

#include <data/host_parquet_representation.hpp>
#include <data/host_parquet_representation_converters.hpp>
#include <data/sirius_converter_registry.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/small_pinned_host_memory_resource.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/span.hpp>

// rmm
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

// duckdb
#include <duckdb.hpp>

// standard library
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

using namespace sirius;
using namespace cucascade::memory;
using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

/**
 * @brief Create memory manager configs with one GPU(0) and one HOST(0).
 */
static std::vector<memory_space_config> create_test_configs()
{
  reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(2048ull * 1024 * 1024)
    .set_gpu_memory_resource_factory(
      [](int, size_t) { return std::make_unique<rmm::mr::cuda_memory_resource>(); })
    .use_host_per_gpu()
    .set_per_host_capacity(4096ull * 1024 * 1024);
  return builder.build();
}

/**
 * @brief Read the Parquet footer from a datasource.
 */
static std::unique_ptr<cudf::io::datasource::buffer> read_parquet_footer(cudf::io::datasource& src)
{
  auto constexpr file_tail_size    = sizeof(cudf::io::parquet::file_ender_s);
  auto constexpr footer_magic_size = sizeof(cudf::io::parquet::file_header_s);
  auto constexpr footer_size_size  = sizeof(uint32_t);

  auto const file_size = src.size();
  REQUIRE(file_size >= file_tail_size);

  auto tail = src.host_read(file_size - file_tail_size, file_tail_size);

  uint32_t footer_size;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));

  return src.host_read(file_size - file_tail_size - footer_size, footer_size);
}

/**
 * @brief RAII helper holding everything needed to build a host_parquet_representation in tests.
 *
 * Creates a DuckDB table, writes it to Parquet, reads the footer, creates a hybrid_scan_reader,
 * reads the column chunks into a fixed_multiple_blocks_allocation, and exposes them for building
 * a host_parquet_representation.
 */
struct parquet_test_fixture {
  std::filesystem::path parquet_path;
  std::unique_ptr<cudf::io::datasource::buffer> footer_buffer;
  cudf::io::parquet_reader_options reader_options;
  std::vector<cudf::size_type> row_group_indices;
  std::vector<cudf::io::text::byte_range_info> column_chunk_byte_ranges;
  std::size_t size_in_bytes              = 0;
  std::size_t uncompressed_size_in_bytes = 0;

  /**
   * @brief Set up the fixture by writing a parquet file and reading its metadata.
   *
   * @param num_rows Number of rows to write
   * @param table_name Name for the DuckDB table (also used in the file name)
   */
  void setup(size_t num_rows, std::string const& table_name = "test_table")
  {
    // Create a DuckDB table and write it to Parquet
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    std::string create_sql =
      "CREATE TABLE " + table_name + " (id INTEGER, value BIGINT, price DOUBLE)";
    auto result = con.Query(create_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());

    if (num_rows > 0) {
      std::string insert_sql = "INSERT INTO " + table_name +
                               " SELECT i, i * 100, i * 1.5 FROM range(" +
                               std::to_string(num_rows) + ") t(i)";
      result = con.Query(insert_sql);
      REQUIRE(result);
      REQUIRE(!result->HasError());
    }

    parquet_path = std::filesystem::temp_directory_path() / (table_name + "_repr_test.parquet");
    std::string copy_sql = "COPY " + table_name + " TO '" + parquet_path.string() +
                           "' (FORMAT PARQUET, COMPRESSION snappy)";
    result = con.Query(copy_sql);
    REQUIRE(result);
    REQUIRE(!result->HasError());

    // Read the footer
    auto datasource = cudf::io::datasource::create(parquet_path.string());
    footer_buffer   = read_parquet_footer(*datasource);

    // Build reader options
    reader_options = cudf::io::parquet_reader_options::builder().build();

    // Create a reader to get the metadata
    auto reader = std::make_unique<hybrid_scan_reader>(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()), reader_options);
    auto const& metadata = reader->parquet_metadata();

    // Populate the row group indices (all row groups)
    auto num_rgs = metadata.row_groups.size();
    row_group_indices.resize(num_rgs);
    std::iota(row_group_indices.begin(), row_group_indices.end(), 0);

    // Get byte ranges for all column chunks across all row groups
    auto rg_span =
      cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size());
    auto byte_ranges = reader->all_column_chunks_byte_ranges(rg_span, reader_options);

    // Compute total size and create rebased byte ranges
    column_chunk_byte_ranges.clear();
    column_chunk_byte_ranges.reserve(byte_ranges.size());
    size_in_bytes = 0;
    for (auto const& range : byte_ranges) {
      column_chunk_byte_ranges.emplace_back(static_cast<int64_t>(size_in_bytes), range.size());
      size_in_bytes += range.size();
    }

    uncompressed_size_in_bytes = size_in_bytes * 2;  // Doesn't matter

    // Store the original byte ranges for reading data
    _original_byte_ranges = std::move(byte_ranges);
    _datasource           = std::move(datasource);
  }

  /**
   * @brief Build a host_parquet_representation with the given memory space.
   */
  std::unique_ptr<host_parquet_representation> build_representation(memory_space* mem_space)
  {
    auto* host_mr = mem_space->get_memory_resource_as<fixed_size_host_memory_resource>();
    REQUIRE(host_mr != nullptr);

    auto allocation = host_mr->allocate_multiple_blocks(size_in_bytes);

    // Read the column chunk data from the file into the allocation
    size_t block_index  = 0;
    size_t block_offset = 0;
    auto block_size     = allocation->block_size();

    for (auto const& range : _original_byte_ranges) {
      auto file_data     = _datasource->host_read(range.offset(), range.size());
      size_t remaining   = range.size();
      size_t data_offset = 0;

      while (remaining > 0) {
        size_t bytes_to_copy = std::min(remaining, block_size - block_offset);
        auto* dst            = allocation->get_blocks()[block_index] + block_offset;
        std::memcpy(dst, file_data->data() + data_offset, bytes_to_copy);
        data_offset += bytes_to_copy;
        remaining -= bytes_to_copy;
        block_offset += bytes_to_copy;
        if (block_offset == block_size) {
          ++block_index;
          block_offset = 0;
        }
      }
    }

    // Create a new reader for this representation
    auto reader = std::make_unique<hybrid_scan_reader>(
      cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()), reader_options);

    return std::make_unique<host_parquet_representation>(mem_space,
                                                         std::move(allocation),
                                                         std::move(reader),
                                                         reader_options,
                                                         row_group_indices,
                                                         column_chunk_byte_ranges,
                                                         size_in_bytes,
                                                         uncompressed_size_in_bytes);
  }

  ~parquet_test_fixture()
  {
    if (!parquet_path.empty() && std::filesystem::exists(parquet_path)) {
      std::filesystem::remove(parquet_path);
    }
  }

 private:
  std::vector<cudf::io::text::byte_range_info> _original_byte_ranges;
  std::unique_ptr<cudf::io::datasource> _datasource;
};

//===----------------------------------------------------------------------===//
// host_parquet_representation Construction Tests
//===----------------------------------------------------------------------===//

TEST_CASE("host_parquet_representation construction", "[host_parquet_representation]")
{
  memory_reservation_manager mgr(create_test_configs());
  auto* host_space = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  REQUIRE(host_space != nullptr);

  parquet_test_fixture fixture;
  fixture.setup(500, "construct_test");

  auto repr = fixture.build_representation(host_space);
  REQUIRE(repr != nullptr);

  SECTION("has correct tier") { REQUIRE(repr->get_current_tier() == Tier::HOST); }

  SECTION("has correct size")
  {
    REQUIRE(repr->get_size_in_bytes() == fixture.size_in_bytes);
    REQUIRE(repr->get_size_in_bytes() > 0);
  }

  SECTION("has correct uncompressed size")
  {
    REQUIRE(repr->get_uncompressed_size_in_bytes() == fixture.uncompressed_size_in_bytes);
  }

  SECTION("has row group indices")
  {
    auto const& rg_indices = repr->get_row_group_indices();
    REQUIRE(rg_indices == fixture.row_group_indices);
  }

  SECTION("has column chunk byte ranges")
  {
    auto const& byte_ranges = repr->get_column_chunk_byte_ranges();
    REQUIRE(byte_ranges.size() == fixture.column_chunk_byte_ranges.size());
  }

  SECTION("has valid column chunks allocation")
  {
    auto const& chunks = repr->get_column_chunks();
    REQUIRE(chunks != nullptr);
  }
}

//===----------------------------------------------------------------------===//
// host_parquet_representation Clone Tests
//===----------------------------------------------------------------------===//

TEST_CASE("host_parquet_representation clone creates independent copy",
          "[host_parquet_representation][clone]")
{
  memory_reservation_manager mgr(create_test_configs());
  auto* host_space = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  REQUIRE(host_space != nullptr);

  parquet_test_fixture fixture;
  fixture.setup(1000, "clone_test");

  auto repr = fixture.build_representation(host_space);
  REQUIRE(repr != nullptr);

  auto cloned_base = repr->clone(rmm::cuda_stream_default);
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<host_parquet_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);

  SECTION("cloned representation has same tier")
  {
    REQUIRE(cloned->get_current_tier() == repr->get_current_tier());
  }

  SECTION("cloned representation has same device_id")
  {
    REQUIRE(cloned->get_device_id() == repr->get_device_id());
  }

  SECTION("cloned representation has same size")
  {
    REQUIRE(cloned->get_size_in_bytes() == repr->get_size_in_bytes());
  }

  SECTION("cloned representation has same uncompressed size")
  {
    REQUIRE(cloned->get_uncompressed_size_in_bytes() == repr->get_uncompressed_size_in_bytes());
  }

  SECTION("cloned representation has same row group indices")
  {
    REQUIRE(cloned->get_row_group_indices() == repr->get_row_group_indices());
  }

  SECTION("cloned representation has same column chunk byte ranges")
  {
    auto const& orig_ranges   = repr->get_column_chunk_byte_ranges();
    auto const& cloned_ranges = cloned->get_column_chunk_byte_ranges();
    REQUIRE(cloned_ranges.size() == orig_ranges.size());
    for (size_t i = 0; i < orig_ranges.size(); ++i) {
      REQUIRE(cloned_ranges[i].offset() == orig_ranges[i].offset());
      REQUIRE(cloned_ranges[i].size() == orig_ranges[i].size());
    }
  }

  SECTION("cloned data is independent (different allocation)")
  {
    auto const& orig_chunks   = repr->get_column_chunks();
    auto const& cloned_chunks = cloned->get_column_chunks();
    REQUIRE(cloned_chunks.get() != orig_chunks.get());

    // Underlying block pointers should be different
    auto orig_blocks   = orig_chunks->get_blocks();
    auto cloned_blocks = cloned_chunks->get_blocks();
    for (size_t i = 0; i < std::min(orig_blocks.size(), cloned_blocks.size()); ++i) {
      REQUIRE(orig_blocks[i] != cloned_blocks[i]);
    }
  }

  SECTION("cloned data content matches original")
  {
    auto const& orig_chunks   = repr->get_column_chunks();
    auto const& cloned_chunks = cloned->get_column_chunks();
    auto orig_blocks          = orig_chunks->get_blocks();
    auto cloned_blocks        = cloned_chunks->get_blocks();
    auto block_size           = orig_chunks->block_size();
    auto remaining            = repr->get_size_in_bytes();

    for (size_t i = 0; i < orig_blocks.size() && remaining > 0; ++i) {
      auto cmp_size = std::min(remaining, block_size);
      REQUIRE(std::memcmp(orig_blocks[i], cloned_blocks[i], cmp_size) == 0);
      remaining -= cmp_size;
    }
  }
}

TEST_CASE("host_parquet_representation clone with small data",
          "[host_parquet_representation][clone]")
{
  memory_reservation_manager mgr(create_test_configs());
  auto* host_space = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  REQUIRE(host_space != nullptr);

  parquet_test_fixture fixture;
  fixture.setup(10, "clone_small");

  auto repr = fixture.build_representation(host_space);
  REQUIRE(repr != nullptr);

  auto cloned_base = repr->clone(rmm::cuda_stream_default);
  REQUIRE(cloned_base != nullptr);

  auto* cloned = dynamic_cast<host_parquet_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);
  REQUIRE(cloned->get_size_in_bytes() == repr->get_size_in_bytes());

  // Verify data content matches
  auto const& orig_chunks   = repr->get_column_chunks();
  auto const& cloned_chunks = cloned->get_column_chunks();
  auto orig_blocks          = orig_chunks->get_blocks();
  auto cloned_blocks        = cloned_chunks->get_blocks();
  auto block_size           = orig_chunks->block_size();
  auto remaining            = repr->get_size_in_bytes();

  for (size_t i = 0; i < orig_blocks.size() && remaining > 0; ++i) {
    auto cmp_size = std::min(remaining, block_size);
    REQUIRE(std::memcmp(orig_blocks[i], cloned_blocks[i], cmp_size) == 0);
    remaining -= cmp_size;
  }
}

//===----------------------------------------------------------------------===//
// host_parquet_representation Converter Registration Tests
//===----------------------------------------------------------------------===//

TEST_CASE("register_parquet_converters registers expected converters",
          "[host_parquet_representation][converters]")
{
  cucascade::representation_converter_registry registry;

  register_parquet_converters(registry);

  SECTION("HOST Parquet -> GPU converter is registered")
  {
    REQUIRE(
      registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>());
  }

  SECTION("HOST Parquet -> HOST Parquet converter is registered")
  {
    REQUIRE(registry.has_converter<host_parquet_representation, host_parquet_representation>());
  }
}

TEST_CASE("register_parquet_converters is idempotent", "[host_parquet_representation][converters]")
{
  cucascade::representation_converter_registry registry;

  // First registration should work
  REQUIRE_NOTHROW(register_parquet_converters(registry));

  // Second registration should also not throw (guarded by has_converter)
  REQUIRE_NOTHROW(register_parquet_converters(registry));

  // Both converters should still be registered
  REQUIRE(
    registry.has_converter<host_parquet_representation, cucascade::gpu_table_representation>());
  REQUIRE(registry.has_converter<host_parquet_representation, host_parquet_representation>());
}

//===----------------------------------------------------------------------===//
// host_parquet_representation -> GPU Conversion Tests
//===----------------------------------------------------------------------===//

TEST_CASE("host_parquet_representation converts to gpu_table_representation",
          "[host_parquet_representation][converters][gpu]")
{
  memory_reservation_manager mgr(create_test_configs());
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  register_parquet_converters(registry);

  auto* host_space = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  auto* gpu_space  = const_cast<memory_space*>(mgr.get_memory_space(Tier::GPU, 0));
  REQUIRE(host_space != nullptr);
  REQUIRE(gpu_space != nullptr);

  // This test runs without a SiriusContext so cuDF's global pinned memory resource
  // may be unset or point to a stale allocator from a previously-paused context.
  // Explicitly install a slab allocator backed by the test's own fixed_size_host_memory_resource
  // so cuDF internal host allocations (e.g. hostdevice_vector) always succeed.
  auto* fsmr =
    host_space->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  REQUIRE(fsmr != nullptr);
  cucascade::memory::small_pinned_host_memory_resource slab_mr(*fsmr);
  // RAII guard: restores previous cuDF state before slab_mr is destroyed.
  // Declared AFTER slab_mr so it is destroyed FIRST (reverse construction order).
  struct cudf_pinned_guard {
    rmm::host_device_async_resource_ref prev_mr;
    std::size_t prev_threshold;
    ~cudf_pinned_guard() noexcept
    {
      cudf::set_pinned_memory_resource(prev_mr);
      cudf::set_allocate_host_as_pinned_threshold(prev_threshold);
    }
  } pinned_guard{cudf::set_pinned_memory_resource(slab_mr),
                 cudf::get_allocate_host_as_pinned_threshold()};
  cudf::set_allocate_host_as_pinned_threshold(
    cucascade::memory::small_pinned_host_memory_resource::MAX_SLAB_SIZE);

  parquet_test_fixture fixture;
  fixture.setup(500, "convert_to_gpu");

  auto repr = fixture.build_representation(host_space);
  REQUIRE(repr != nullptr);

  // Use a stream from the GPU memory space's pool so it outlives the reader
  // (the hybrid_scan_reader destructor frees internal device memory that needs a valid stream)
  auto stream     = gpu_space->acquire_stream();
  auto gpu_result = registry.convert<cucascade::gpu_table_representation>(*repr, gpu_space, stream);
  stream.synchronize();

  REQUIRE(gpu_result != nullptr);
  REQUIRE(gpu_result->get_current_tier() == Tier::GPU);
  REQUIRE(gpu_result->get_table().num_rows() == 500);
  // 3 columns: id (INT32), value (BIGINT), price (DOUBLE)
  REQUIRE(gpu_result->get_table().num_columns() == 3);
  REQUIRE(gpu_result->get_size_in_bytes() > 0);

  // Explicitly destroy GPU result and representation before the memory manager
  gpu_result.reset();
  repr.reset();
}

TEST_CASE("host_parquet_representation converts to GPU with projected columns",
          "[host_parquet_representation][converters][gpu][projection]")
{
  memory_reservation_manager mgr(create_test_configs());
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  register_parquet_converters(registry);

  auto* host_space = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  auto* gpu_space  = const_cast<memory_space*>(mgr.get_memory_space(Tier::GPU, 0));
  REQUIRE(host_space != nullptr);
  REQUIRE(gpu_space != nullptr);

  // Write a 3-column table and set projection to only read "id" and "price"
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  con.Query("CREATE TABLE projected_test (id INTEGER, value BIGINT, price DOUBLE)");
  con.Query("INSERT INTO projected_test SELECT i, i * 100, i * 1.5 FROM range(200) t(i)");

  auto parquet_path = std::filesystem::temp_directory_path() / "projected_repr_test.parquet";
  con.Query("COPY projected_test TO '" + parquet_path.string() +
            "' (FORMAT PARQUET, COMPRESSION snappy)");

  auto datasource    = cudf::io::datasource::create(parquet_path.string());
  auto footer_buffer = read_parquet_footer(*datasource);

  auto reader_options = cudf::io::parquet_reader_options::builder().build();
#if CUDF_VERSION_NUM >= 2604
  reader_options.set_column_names({"id", "price"});
#else
  reader_options.set_columns({"id", "price"});
#endif

  auto reader = std::make_unique<hybrid_scan_reader>(
    cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()), reader_options);
  auto const& metadata = reader->parquet_metadata();

  std::vector<cudf::size_type> rg_indices(metadata.row_groups.size());
  std::iota(rg_indices.begin(), rg_indices.end(), 0);

  auto rg_span     = cudf::host_span<cudf::size_type const>(rg_indices.data(), rg_indices.size());
  auto byte_ranges = reader->all_column_chunks_byte_ranges(rg_span, reader_options);

  // Read data into allocation
  std::vector<cudf::io::text::byte_range_info> rebased_ranges;
  size_t total_size = 0;
  for (auto const& range : byte_ranges) {
    rebased_ranges.emplace_back(static_cast<int64_t>(total_size), range.size());
    total_size += range.size();
  }

  auto* host_mr   = host_space->get_memory_resource_as<fixed_size_host_memory_resource>();
  auto allocation = host_mr->allocate_multiple_blocks(total_size);

  size_t block_index = 0, block_offset = 0;
  auto block_size = allocation->block_size();
  for (auto const& range : byte_ranges) {
    auto file_data   = datasource->host_read(range.offset(), range.size());
    size_t remaining = range.size(), data_offset = 0;
    while (remaining > 0) {
      size_t bytes_to_copy = std::min(remaining, block_size - block_offset);
      std::memcpy(allocation->get_blocks()[block_index] + block_offset,
                  file_data->data() + data_offset,
                  bytes_to_copy);
      data_offset += bytes_to_copy;
      remaining -= bytes_to_copy;
      block_offset += bytes_to_copy;
      if (block_offset == block_size) {
        ++block_index;
        block_offset = 0;
      }
    }
  }

  auto proj_reader = std::make_unique<hybrid_scan_reader>(
    cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()), reader_options);

  auto repr = std::make_unique<host_parquet_representation>(host_space,
                                                            std::move(allocation),
                                                            std::move(proj_reader),
                                                            reader_options,
                                                            rg_indices,
                                                            rebased_ranges,
                                                            total_size,
                                                            total_size * 2);

  auto stream     = gpu_space->acquire_stream();
  auto gpu_result = registry.convert<cucascade::gpu_table_representation>(*repr, gpu_space, stream);
  stream.synchronize();

  REQUIRE(gpu_result != nullptr);
  REQUIRE(gpu_result->get_table().num_rows() == 200);
  REQUIRE(gpu_result->get_table().num_columns() == 2);  // only "id" and "price"

  // Explicitly destroy GPU result and representation before the memory manager
  gpu_result.reset();
  repr.reset();
  reader.reset();

  std::filesystem::remove(parquet_path);
}

TEST_CASE("host_parquet_representation clone then convert to GPU",
          "[host_parquet_representation][clone][converters][gpu]")
{
  memory_reservation_manager mgr(create_test_configs());
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  register_parquet_converters(registry);

  auto* host_space = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  auto* gpu_space  = const_cast<memory_space*>(mgr.get_memory_space(Tier::GPU, 0));
  REQUIRE(host_space != nullptr);
  REQUIRE(gpu_space != nullptr);

  parquet_test_fixture fixture;
  fixture.setup(300, "clone_then_convert");

  auto repr = fixture.build_representation(host_space);
  REQUIRE(repr != nullptr);

  // Clone the representation
  auto cloned_base = repr->clone(rmm::cuda_stream_default);
  REQUIRE(cloned_base != nullptr);
  auto* cloned = dynamic_cast<host_parquet_representation*>(cloned_base.get());
  REQUIRE(cloned != nullptr);

  // Convert both original and clone to GPU
  auto stream   = gpu_space->acquire_stream();
  auto gpu_orig = registry.convert<cucascade::gpu_table_representation>(*repr, gpu_space, stream);
  auto gpu_cloned =
    registry.convert<cucascade::gpu_table_representation>(*cloned, gpu_space, stream);
  stream.synchronize();

  REQUIRE(gpu_orig != nullptr);
  REQUIRE(gpu_cloned != nullptr);

  // Both GPU representations should have the same shape and data
  REQUIRE(gpu_orig->get_table().num_rows() == gpu_cloned->get_table().num_rows());
  REQUIRE(gpu_orig->get_table().num_columns() == gpu_cloned->get_table().num_columns());
  REQUIRE(gpu_orig->get_table().num_rows() == 300);

  // Explicitly destroy GPU results and representations before the memory manager
  gpu_orig.reset();
  gpu_cloned.reset();
  repr.reset();
  cloned_base.reset();
}

//===----------------------------------------------------------------------===//
// host_parquet_representation Cross-Host Copy Converter Tests
//===----------------------------------------------------------------------===//

TEST_CASE("host_parquet_representation cross-host copy converter",
          "[host_parquet_representation][converters][cross_host]")
{
  // Create a config with 2 GPUs (each with its own host space) to test cross-host copy
  reservation_manager_configurator builder;
  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(2048ull * 1024 * 1024)
    .set_gpu_memory_resource_factory(
      [](int, size_t) { return std::make_unique<rmm::mr::cuda_memory_resource>(); })
    .use_host_per_gpu()
    .set_per_host_capacity(4096ull * 1024 * 1024);

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    SUCCEED("Fewer than 2 GPUs available; skipping cross-host test");
    return;
  }

  memory_reservation_manager mgr(builder.build());
  cucascade::representation_converter_registry registry;
  cucascade::register_builtin_converters(registry);
  register_parquet_converters(registry);

  auto* host_space_0 = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 0));
  auto* host_space_1 = const_cast<memory_space*>(mgr.get_memory_space(Tier::HOST, 1));
  REQUIRE(host_space_0 != nullptr);
  REQUIRE(host_space_1 != nullptr);
  REQUIRE(host_space_0->get_device_id() != host_space_1->get_device_id());

  parquet_test_fixture fixture;
  fixture.setup(200, "cross_host");

  auto repr = fixture.build_representation(host_space_0);
  REQUIRE(repr != nullptr);
  REQUIRE(repr->get_device_id() == host_space_0->get_device_id());

  // Convert from host_space_0 -> host_space_1
  rmm::cuda_stream stream;
  auto result = registry.convert<host_parquet_representation>(*repr, host_space_1, stream);

  REQUIRE(result != nullptr);
  REQUIRE(result->get_current_tier() == Tier::HOST);
  REQUIRE(result->get_device_id() == host_space_1->get_device_id());
  REQUIRE(result->get_size_in_bytes() == repr->get_size_in_bytes());
}
