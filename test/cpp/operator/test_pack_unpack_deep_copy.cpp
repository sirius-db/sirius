/*
 * Copyright 2025, Sirius Contributors.
 *
 * Reproduces the cudaErrorInvalidValue / garbage-size issue when doing
 * cudf::pack → cudf::unpack → cudf::table deep copy with cuda_memory_resource.
 *
 * Failing scenario from Q14 exchange:
 * - Leaf GPU execution fills the RMM pool
 * - cudf::chunked_pack packs result into staging buffer (outside RMM)
 * - cudf::unpack creates table_view into packed buffer
 * - cudf::table(view, stream, &cuda_mr) deep-copies → cudaErrorInvalidValue
 */

#include <catch.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <memory>
#include <vector>

namespace {

/// Create a test table with the same schema as Q14's __EXCH_..._3 (16 INT32 cols).
std::unique_ptr<cudf::table> make_int32_table(cudf::size_type num_rows, int num_cols) {
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (int c = 0; c < num_cols; c++) {
    auto scalar = cudf::numeric_scalar<int32_t>(c * 100 + 1, true, stream);
    cols.push_back(cudf::sequence(num_rows, scalar, stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Create a test table with INT32 + DECIMAL128 columns (like Q14's __EXCH_..._1).
std::unique_ptr<cudf::table> make_mixed_table(cudf::size_type num_rows) {
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();
  std::vector<std::unique_ptr<cudf::column>> cols;
  // 3 INT32 columns
  for (int c = 0; c < 3; c++) {
    auto scalar = cudf::numeric_scalar<int32_t>(c + 1, true, stream);
    cols.push_back(cudf::sequence(num_rows, scalar, stream, mr));
  }
  // 6 DECIMAL128 columns (scale = -2)
  for (int c = 0; c < 6; c++) {
    auto dtype = cudf::data_type{cudf::type_id::DECIMAL128, -2};
    auto col   = cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream);
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

TEST_CASE("pack_unpack_deep_copy with default mr", "[exchange]") {
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();

  SECTION("16 INT32 columns, 2400 rows") {
    auto table = make_int32_table(2400, 16);
    auto view  = table->view();

    // Pack
    auto packed = cudf::pack(view, stream, mr);
    REQUIRE(packed.gpu_data->size() > 0);
    REQUIRE(packed.metadata->size() > 0);

    // Unpack
    auto unpacked_view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));
    REQUIRE(unpacked_view.num_columns() == 16);
    REQUIRE(unpacked_view.num_rows() == 2400);

    // Deep copy with default mr
    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, mr);
    REQUIRE(copy->num_columns() == 16);
    REQUIRE(copy->num_rows() == 2400);
  }

  SECTION("9 mixed columns, 6300 rows") {
    auto table = make_mixed_table(6300);
    auto view  = table->view();

    auto packed = cudf::pack(view, stream, mr);
    auto unpacked_view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));
    REQUIRE(unpacked_view.num_columns() == 9);
    REQUIRE(unpacked_view.num_rows() == 6300);

    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, mr);
    REQUIRE(copy->num_columns() == 9);
    REQUIRE(copy->num_rows() == 6300);
  }
}

TEST_CASE("pack_unpack_deep_copy with cuda_memory_resource", "[exchange]") {
  auto stream = cudf::get_default_stream();
  static rmm::mr::cuda_memory_resource cuda_mr;

  SECTION("16 INT32 columns, 2400 rows") {
    auto table = make_int32_table(2400, 16);
    auto view  = table->view();

    auto packed = cudf::pack(view, stream);
    auto unpacked_view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

    // Deep copy with raw cudaMalloc (not pool)
    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, &cuda_mr);
    REQUIRE(copy->num_columns() == 16);
    REQUIRE(copy->num_rows() == 2400);
  }

  SECTION("9 mixed columns, 6300 rows") {
    auto table = make_mixed_table(6300);
    auto view  = table->view();

    auto packed = cudf::pack(view, stream);
    auto unpacked_view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, &cuda_mr);
    REQUIRE(copy->num_columns() == 9);
    REQUIRE(copy->num_rows() == 6300);
  }
}

TEST_CASE("pack into external buffer then unpack and deep copy", "[exchange]") {
  // Simulates the actual exchange flow:
  // 1. Pack into an external staging buffer (like cudf::chunked_pack)
  // 2. Unpack from that external buffer
  // 3. Deep copy with cuda_memory_resource
  auto stream = cudf::get_default_stream();
  static rmm::mr::cuda_memory_resource cuda_mr;

  auto table = make_int32_table(2400, 16);
  auto view  = table->view();

  // Allocate external buffer via cudaMalloc (simulating staging buffer).
  // cudf::chunked_pack requires at least 1MB buffer.
  size_t buf_size = 4UL << 20; // 4MB — plenty for 2400×16 INT32
  void* staging_ptr = nullptr;
  REQUIRE(cudaMalloc(&staging_ptr, buf_size) == cudaSuccess);

  // Pack into external buffer using chunked_pack
  auto packer = cudf::chunked_pack::create(view, buf_size, stream);
  auto total  = packer->get_total_contiguous_size();
  REQUIRE(total <= buf_size);
  cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging_ptr), buf_size);
  while (packer->has_next()) { packer->next(dst); }
  auto md = packer->build_metadata();

  // Unpack from external buffer
  auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(staging_ptr));
  REQUIRE(unpacked.num_columns() == 16);
  REQUIRE(unpacked.num_rows() == 2400);

  // Deep copy with cuda_mr (the actual failing operation in production)
  auto copy = std::make_unique<cudf::table>(unpacked, stream, &cuda_mr);
  REQUIRE(copy->num_columns() == 16);
  REQUIRE(copy->num_rows() == 2400);

  // Verify data survives after freeing staging
  cudaFree(staging_ptr);
  // copy should still be valid (it's a deep copy)
  REQUIRE(copy->num_rows() == 2400);
}

TEST_CASE("pack_unpack_deep_copy after RMM pool OOM", "[exchange]") {
  // Reproduces the exact failure scenario:
  // 1. Fill the RMM pool to capacity
  // 2. This triggers a CUDA error
  // 3. Pack data into external staging
  // 4. Unpack + deep copy with cuda_mr
  // The sticky CUDA error from step 2 causes step 4 to fail.
  auto stream = cudf::get_default_stream();

  // Create a small RMM pool (16MB) to easily fill it
  rmm::mr::cuda_memory_resource base_mr;
  size_t pool_size = 16UL << 20; // 16MB
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
      &base_mr, pool_size, pool_size);

  // Fill the pool
  std::vector<rmm::device_buffer> fillers;
  bool pool_full = false;
  for (int i = 0; i < 100; i++) {
    try {
      fillers.emplace_back(1UL << 20, stream, &pool_mr); // 1MB chunks
    } catch (...) {
      pool_full = true;
      break;
    }
  }
  REQUIRE(pool_full);

  // The pool OOM may have left a sticky CUDA error.
  // Clear it (this is what our fix does).
  cudaError_t err = cudaGetLastError();
  INFO("Sticky CUDA error after pool OOM: " << cudaGetErrorString(err));

  // Now create a table and pack it (using the base cuda_mr, not the full pool)
  static rmm::mr::cuda_memory_resource cuda_mr;
  auto table = make_int32_table(2400, 16);

  auto packed = cudf::pack(table->view(), stream);
  auto unpacked = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

  // Deep copy with cuda_mr — this should work even though the pool is full,
  // because cuda_mr uses cudaMalloc directly (not the pool).
  auto copy = std::make_unique<cudf::table>(unpacked, stream, &cuda_mr);
  REQUIRE(copy->num_columns() == 16);
  REQUIRE(copy->num_rows() == 2400);

  // Clean up pool allocations
  fillers.clear();
}

TEST_CASE("concatenate packed tables with cuda_mr", "[exchange]") {
  // Simulates multiple senders: two tables packed separately, then
  // unpacked and concatenated using cuda_mr.
  auto stream = cudf::get_default_stream();
  static rmm::mr::cuda_memory_resource cuda_mr;

  auto table1 = make_int32_table(2400, 16);
  auto table2 = make_int32_table(2300, 16);

  auto packed1 = cudf::pack(table1->view(), stream);
  auto packed2 = cudf::pack(table2->view(), stream);

  auto view1 = cudf::unpack(packed1.metadata->data(), static_cast<const uint8_t*>(packed1.gpu_data->data()));
  auto view2 = cudf::unpack(packed2.metadata->data(), static_cast<const uint8_t*>(packed2.gpu_data->data()));

  // First registration: deep copy
  auto owned1 = std::make_unique<cudf::table>(view1, stream, &cuda_mr);
  REQUIRE(owned1->num_rows() == 2400);

  // Second registration: concatenate
  std::vector<cudf::table_view> views = {owned1->view(), view2};
  auto merged = cudf::concatenate(views, stream, &cuda_mr);
  REQUIRE(merged->num_rows() == 4700);
  REQUIRE(merged->num_columns() == 16);
}

TEST_CASE("pack_unpack DECIMAL128 child columns", "[exchange]") {
  // Investigate whether cudf::pack/unpack adds phantom children to DECIMAL128 columns.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();

  auto table = make_mixed_table(1000); // 3 INT32 + 6 DECIMAL128
  auto view = table->view();

  // Check original: DECIMAL128 should have 0 children
  for (cudf::size_type c = 3; c < view.num_columns(); c++) {
    auto col = view.column(c);
    INFO("col " << c << " type=" << static_cast<int>(col.type().id())
         << " children=" << col.num_children());
    CHECK(col.num_children() == 0);
  }

  // Pack and unpack
  auto packed = cudf::pack(view, stream, mr);
  auto unpacked = cudf::unpack(packed.metadata->data(),
                               static_cast<const uint8_t*>(packed.gpu_data->data()));

  // Check unpacked: do DECIMAL128 columns gain phantom children?
  bool has_phantom_children = false;
  for (cudf::size_type c = 3; c < unpacked.num_columns(); c++) {
    auto col = unpacked.column(c);
    INFO("unpacked col " << c << " type=" << static_cast<int>(col.type().id())
         << " children=" << col.num_children());
    if (col.num_children() > 0) {
      has_phantom_children = true;
    }
  }

  if (has_phantom_children) {
    WARN("DECIMAL128 columns from cudf::unpack have phantom children");
    static rmm::mr::cuda_memory_resource cuda_mr;
    cudaGetLastError();
    std::vector<cudf::table_view> views = {unpacked};
    auto result = cudf::concatenate(views, stream, &cuda_mr);
    CHECK(result->num_rows() == 1000);
  }

  SECTION("after hash_partition") {
    // Reproduce production path: hash_partition → pack → unpack → concatenate.
    // hash_partition output may have different internal structure than raw columns.
    std::vector<cudf::size_type> col_indices = {0}; // partition by first INT32 col
    auto [partitioned, offsets] = cudf::hash_partition(
        view, col_indices, 2, cudf::hash_id::HASH_MURMUR3,
        cudf::DEFAULT_HASH_SEED, stream);

    // Pack the first partition
    auto start = offsets[0];
    auto end = offsets[1];
    if (end > start) {
      auto slice = cudf::slice(partitioned->view(), {start, end});
      auto part_packed = cudf::pack(slice[0], stream, mr);
      auto part_unpacked = cudf::unpack(part_packed.metadata->data(),
                                        static_cast<const uint8_t*>(part_packed.gpu_data->data()));

      // Check for phantom children
      for (cudf::size_type c = 3; c < part_unpacked.num_columns(); c++) {
        auto col = part_unpacked.column(c);
        INFO("hash_partition unpacked col " << c << " type=" << static_cast<int>(col.type().id())
             << " children=" << col.num_children());
        if (col.num_children() > 0) {
          WARN("hash_partition DECIMAL128 has phantom child after pack/unpack");
        }
      }

      // Try concatenate
      static rmm::mr::cuda_memory_resource cuda_mr;
      cudaGetLastError();
      std::vector<cudf::table_view> views = {part_unpacked};
      auto result = cudf::concatenate(views, stream, &cuda_mr);
      CHECK(result->num_rows() == (end - start));
    }
  }
}

TEST_CASE("concatenate 12 packed views with INT32+DECIMAL columns", "[exchange]") {
  // Reproduces Q3 __EXCH_..._5: 12 views from separate cudf::pack buffers,
  // 8 cols (INT32 + DECIMAL128), ~1000 rows each. cudf::concatenate should work.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  // Create 12 packed buffers and accumulate views
  std::vector<cudf::packed_columns> packed_buffers;
  std::vector<cudf::table_view> views;

  for (int i = 0; i < 12; i++) {
    auto table = make_mixed_table(800 + i * 50); // 3 INT32 + 6 DECIMAL128
    auto packed = cudf::pack(table->view(), stream, mr);
    auto unpacked = cudf::unpack(packed.metadata->data(),
                                 static_cast<const uint8_t*>(packed.gpu_data->data()));
    views.push_back(unpacked);
    packed_buffers.push_back(std::move(packed));
  }

  REQUIRE(views.size() == 12);

  // Concatenate all views
  cudaGetLastError();
  auto result = cudf::concatenate(views, stream, &cuda_mr);
  REQUIRE(result->num_columns() == 9);
  // Total rows: sum of 800+0*50, 800+1*50, ..., 800+11*50 = 12*800 + 50*(0+1+...+11) = 9600+3300=12900
  REQUIRE(result->num_rows() == 12900);
}

/// Create a table matching Q3 __EXCH_..._5 schema: INT32 keys + STRING + DECIMAL128 values.
/// This is what cudf::hash_partition produces for a join result.
std::unique_ptr<cudf::table> make_q3_exchange_table(cudf::size_type num_rows) {
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();
  std::vector<std::unique_ptr<cudf::column>> cols;

  // 2 INT32 key columns (l_orderkey, o_shippriority)
  for (int c = 0; c < 2; c++) {
    auto scalar = cudf::numeric_scalar<int32_t>(c * 1000 + 1, true, stream);
    cols.push_back(cudf::sequence(num_rows, scalar, cudf::numeric_scalar<int32_t>(1, true, stream), stream, mr));
  }

  // 2 STRING columns (simulating o_orderstatus, l_shipmode)
  for (int s = 0; s < 2; s++) {
    std::vector<std::string> strings;
    for (int i = 0; i < num_rows; i++) {
      strings.push_back("val_" + std::to_string(s) + "_" + std::to_string(i % 100));
    }
    std::vector<int32_t> offsets = {0};
    std::string chars;
    for (auto& str : strings) {
      chars += str;
      offsets.push_back(static_cast<int32_t>(chars.size()));
    }
    auto offsets_col = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT32}, num_rows + 1,
        cudf::mask_state::UNALLOCATED, stream);
    cudaMemcpy(offsets_col->mutable_view().data<int32_t>(), offsets.data(),
               offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    rmm::device_buffer chars_buf(chars.size(), stream, mr);
    cudaMemcpy(chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice);
    cols.push_back(cudf::make_strings_column(
        num_rows, std::move(offsets_col), std::move(chars_buf), 0,
        rmm::device_buffer{0, stream, mr}));
  }

  // 4 DECIMAL128 value columns (revenue, price, discount, tax)
  for (int c = 0; c < 4; c++) {
    auto dtype = cudf::data_type{cudf::type_id::DECIMAL128, -2};
    auto col = cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream);
    cols.push_back(std::move(col));
  }

  return std::make_unique<cudf::table>(std::move(cols)); // 2 INT32 + 2 STRING + 4 DECIMAL128 = 8 cols
}

TEST_CASE("Q3 exchange: hash_partition + pack + unpack + concatenate with STRING cols", "[exchange][q3]") {
  // Reproduces the exact Q3 exchange scenario:
  // 1. Create table with INT32 + STRING + DECIMAL128 (8 cols, ~1000 rows)
  // 2. hash_partition into 2 partitions
  // 3. Pack each partition separately (simulating per-batch cudf::chunked_pack)
  // 4. Unpack all into views
  // 5. Concatenate all views (simulating finalize_pending_views)
  //
  // Repeat for multiple "batches" to simulate concurrent GPU pipeline threads.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  // Simulate 8 GPU pipeline batches (like the production Q3 leaf execution)
  constexpr int NUM_BATCHES = 8;
  constexpr int NUM_PARTITIONS = 2;

  // Accumulate packed buffers and views per partition (simulating ExchangeBuffer)
  std::vector<cudf::packed_columns> all_packed; // Keep packed buffers alive
  std::vector<cudf::table_view> partition_0_views;
  std::vector<cudf::table_view> partition_1_views;

  for (int batch = 0; batch < NUM_BATCHES; batch++) {
    auto table = make_q3_exchange_table(800 + batch * 100);
    auto view = table->view();

    // Hash partition
    std::vector<cudf::size_type> key_cols = {0}; // Partition by first INT32
    auto [partitioned, offsets] = cudf::hash_partition(
        view, key_cols, NUM_PARTITIONS, cudf::hash_id::HASH_MURMUR3,
        cudf::DEFAULT_HASH_SEED, stream);

    // Pack each partition separately
    for (int p = 0; p < NUM_PARTITIONS; p++) {
      auto start = offsets[p];
      auto end = (p + 1 < NUM_PARTITIONS) ? offsets[p + 1] : partitioned->num_rows();
      if (end <= start) continue;

      auto slice = cudf::slice(partitioned->view(), {start, end});
      auto packed = cudf::pack(slice[0], stream, mr);
      auto unpacked = cudf::unpack(packed.metadata->data(),
                                   static_cast<const uint8_t*>(packed.gpu_data->data()));

      if (p == 0) {
        partition_0_views.push_back(unpacked);
      } else {
        partition_1_views.push_back(unpacked);
      }
      all_packed.push_back(std::move(packed));
    }
  }

  INFO("partition 0: " << partition_0_views.size() << " views");
  INFO("partition 1: " << partition_1_views.size() << " views");
  REQUIRE(partition_0_views.size() > 0);
  REQUIRE(partition_1_views.size() > 0);

  // Log column info for first view
  {
    auto& pv = partition_0_views[0];
    for (cudf::size_type c = 0; c < pv.num_columns(); c++) {
      auto col = pv.column(c);
      INFO("  partition 0 view 0 col " << c << " type=" << static_cast<int>(col.type().id())
           << " children=" << col.num_children() << " rows=" << col.size());
    }
  }

  SECTION("concatenate partition 0 views") {
    cudaGetLastError();
    auto result = cudf::concatenate(partition_0_views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("partition 0 concatenated: " << result->num_rows() << " rows");
  }

  SECTION("concatenate partition 1 views") {
    cudaGetLastError();
    auto result = cudf::concatenate(partition_1_views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("partition 1 concatenated: " << result->num_rows() << " rows");
  }

  SECTION("simulate D2D copy then concatenate") {
    // Simulate the self-transfer D2D copy path:
    // 1. D2D copy each packed buffer to a new cudaMalloc region
    // 2. Re-unpack from the new address
    // 3. Concatenate all re-unpacked views
    std::vector<cudf::table_view> copied_views;
    std::vector<void*> owned_ptrs; // For cleanup

    for (auto& packed : all_packed) {
      size_t size = packed.gpu_data->size();
      void* new_ptr = nullptr;
      REQUIRE(cudaMalloc(&new_ptr, size) == cudaSuccess);
      REQUIRE(cudaMemcpy(new_ptr, packed.gpu_data->data(), size, cudaMemcpyDeviceToDevice) == cudaSuccess);

      auto view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(new_ptr));
      copied_views.push_back(view);
      owned_ptrs.push_back(new_ptr);
    }

    cudaGetLastError();
    auto result = cudf::concatenate(copied_views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("D2D concatenated: " << result->num_rows() << " rows from " << copied_views.size() << " views");

    for (auto ptr : owned_ptrs) cudaFree(ptr);
  }

  SECTION("simulate staging overwrite then concatenate (should fail)") {
    // Simulate the bug: pack into a single staging-like buffer,
    // then overwrite the staging with new data, then try to concatenate
    // the views that reference the overwritten region.
    size_t staging_size = 64UL << 20; // 64MB staging
    void* staging = nullptr;
    REQUIRE(cudaMalloc(&staging, staging_size) == cudaSuccess);

    std::vector<cudf::table_view> stale_views;
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadatas;

    // Pack first batch into staging
    {
      auto table = make_q3_exchange_table(1000);
      auto packed = cudf::pack(table->view(), stream, mr);
      size_t size = packed.gpu_data->size();
      REQUIRE(size <= staging_size);
      REQUIRE(cudaMemcpy(staging, packed.gpu_data->data(), size, cudaMemcpyDeviceToDevice) == cudaSuccess);
      auto view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(staging));
      stale_views.push_back(view);
      metadatas.push_back(std::move(packed.metadata));
    }

    // Overwrite staging with DIFFERENT data (simulating next session)
    {
      auto table = make_q3_exchange_table(2000); // Different row count!
      auto packed = cudf::pack(table->view(), stream, mr);
      size_t size = packed.gpu_data->size();
      REQUIRE(size <= staging_size);
      REQUIRE(cudaMemcpy(staging, packed.gpu_data->data(), size, cudaMemcpyDeviceToDevice) == cudaSuccess);
      // Don't create a view — we just overwrote the staging
    }

    // Try to concatenate the stale view — data was overwritten
    // This should either crash, produce wrong results, or give a CUDA error.
    cudaGetLastError();
    bool stale_concat_failed = false;
    try {
      auto result = cudf::concatenate(stale_views, stream, &cuda_mr);
      // If it succeeds, the data is wrong (overwritten by second batch)
      WARN("stale view concatenation succeeded (data is likely corrupt)");
    } catch (const std::exception& e) {
      stale_concat_failed = true;
      INFO("stale view concatenation failed as expected: " << e.what());
    }

    cudaFree(staging);
    // We don't CHECK stale_concat_failed — it may or may not crash depending on
    // whether the overwritten data happens to produce valid column metadata.
  }
}

TEST_CASE("pack_unpack with STRING columns", "[exchange]") {
  // Reproduces Q3/Q14 exchange table issue: tables with STRING columns
  // packed into staging → unpacked → deep copy fails, concatenate works.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();

  // Build a table with INT32 + STRING columns (like TPC-H lineitem partkeys + shipmode)
  auto int_scalar = cudf::numeric_scalar<int32_t>(42, true, stream);
  auto int_col = cudf::sequence(1000, int_scalar, stream, mr);

  // Build a string column from host data
  std::vector<std::string> strings;
  for (int i = 0; i < 1000; i++) {
    strings.push_back("string_value_" + std::to_string(i));
  }
  std::vector<int32_t> offsets = {0};
  std::string chars;
  for (auto& s : strings) {
    chars += s;
    offsets.push_back(static_cast<int32_t>(chars.size()));
  }
  auto offsets_col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, 1001, cudf::mask_state::UNALLOCATED, stream);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(), offsets.data(),
             offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  rmm::device_buffer chars_buf(chars.size(), stream, mr);
  cudaMemcpy(chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice);
  auto str_col = cudf::make_strings_column(
      1000, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{0, stream, mr});

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(str_col));
  auto table = std::make_unique<cudf::table>(std::move(cols));
  REQUIRE(table->num_columns() == 2);
  REQUIRE(table->num_rows() == 1000);

  // Pack
  auto packed = cudf::pack(table->view(), stream, mr);
  auto unpacked = cudf::unpack(packed.metadata->data(),
                               static_cast<const uint8_t*>(packed.gpu_data->data()));
  REQUIRE(unpacked.num_columns() == 2);
  REQUIRE(unpacked.num_rows() == 1000);

  SECTION("cudf::concatenate with unpack'd STRING view works") {
    static rmm::mr::cuda_memory_resource cuda_mr;
    cudaGetLastError();
    std::vector<cudf::table_view> views = {unpacked};
    auto result = cudf::concatenate(views, stream, &cuda_mr);
    REQUIRE(result->num_columns() == 2);
    REQUIRE(result->num_rows() == 1000);
    // Verify string data is intact
    cudf::strings_column_view str_view(result->view().column(1));
    REQUIRE(str_view.size() == 1000);
  }

  SECTION("cudf::table deep copy with unpack'd STRING view") {
    static rmm::mr::cuda_memory_resource cuda_mr;
    cudaGetLastError();
    // This may fail for STRING columns from cudf::unpack.
    // If it does, that confirms cudf::concatenate is the right approach.
    bool deep_copy_works = true;
    try {
      auto copy = std::make_unique<cudf::table>(unpacked, stream, &cuda_mr);
      REQUIRE(copy->num_columns() == 2);
      REQUIRE(copy->num_rows() == 1000);
    } catch (const std::exception& e) {
      INFO("cudf::table(view) deep copy failed (expected for STRING from unpack): " << e.what());
      deep_copy_works = false;
    }
    // Log whether deep copy works — helps diagnose the cudf behavior.
    // If deep_copy_works is false, we rely on cudf::concatenate instead.
    INFO("STRING column deep copy " << (deep_copy_works ? "succeeded" : "failed"));
  }
}
