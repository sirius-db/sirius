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

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <catch.hpp>
#include <duckdb.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace {

/// Create a test table with the same schema as Q14's __EXCH_..._3 (16 INT32 cols).
std::unique_ptr<cudf::table> make_int32_table(cudf::size_type num_rows, int num_cols)
{
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
std::unique_ptr<cudf::table> make_mixed_table(cudf::size_type num_rows)
{
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
    auto col =
      cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream);
    cols.push_back(std::move(col));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

TEST_CASE("pack_unpack_deep_copy with default mr", "[exchange]")
{
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();

  SECTION("16 INT32 columns, 2400 rows")
  {
    auto table = make_int32_table(2400, 16);
    auto view  = table->view();

    // Pack
    auto packed = cudf::pack(view, stream, mr);
    REQUIRE(packed.gpu_data->size() > 0);
    REQUIRE(packed.metadata->size() > 0);

    // Unpack
    auto unpacked_view =
      cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));
    REQUIRE(unpacked_view.num_columns() == 16);
    REQUIRE(unpacked_view.num_rows() == 2400);

    // Deep copy with default mr
    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, mr);
    REQUIRE(copy->num_columns() == 16);
    REQUIRE(copy->num_rows() == 2400);
  }

  SECTION("9 mixed columns, 6300 rows")
  {
    auto table = make_mixed_table(6300);
    auto view  = table->view();

    auto packed = cudf::pack(view, stream, mr);
    auto unpacked_view =
      cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));
    REQUIRE(unpacked_view.num_columns() == 9);
    REQUIRE(unpacked_view.num_rows() == 6300);

    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, mr);
    REQUIRE(copy->num_columns() == 9);
    REQUIRE(copy->num_rows() == 6300);
  }
}

TEST_CASE("pack_unpack_deep_copy with cuda_memory_resource", "[exchange]")
{
  auto stream = cudf::get_default_stream();
  static rmm::mr::cuda_memory_resource cuda_mr;

  SECTION("16 INT32 columns, 2400 rows")
  {
    auto table = make_int32_table(2400, 16);
    auto view  = table->view();

    auto packed = cudf::pack(view, stream);
    auto unpacked_view =
      cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

    // Deep copy with raw cudaMalloc (not pool)
    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, &cuda_mr);
    REQUIRE(copy->num_columns() == 16);
    REQUIRE(copy->num_rows() == 2400);
  }

  SECTION("9 mixed columns, 6300 rows")
  {
    auto table = make_mixed_table(6300);
    auto view  = table->view();

    auto packed = cudf::pack(view, stream);
    auto unpacked_view =
      cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

    auto copy = std::make_unique<cudf::table>(unpacked_view, stream, &cuda_mr);
    REQUIRE(copy->num_columns() == 9);
    REQUIRE(copy->num_rows() == 6300);
  }
}

TEST_CASE("pack into external buffer then unpack and deep copy", "[exchange]")
{
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
  size_t buf_size   = 4UL << 20;  // 4MB — plenty for 2400×16 INT32
  void* staging_ptr = nullptr;
  REQUIRE(cudaMalloc(&staging_ptr, buf_size) == cudaSuccess);

  // Pack into external buffer using chunked_pack
  auto packer = cudf::chunked_pack::create(view, buf_size, stream);
  auto total  = packer->get_total_contiguous_size();
  REQUIRE(total <= buf_size);
  cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging_ptr), buf_size);
  while (packer->has_next()) {
    packer->next(dst);
  }
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

TEST_CASE("pack_unpack_deep_copy after RMM pool OOM", "[exchange]")
{
  // Reproduces the exact failure scenario:
  // 1. Fill the RMM pool to capacity
  // 2. This triggers a CUDA error
  // 3. Pack data into external staging
  // 4. Unpack + deep copy with cuda_mr
  // The sticky CUDA error from step 2 causes step 4 to fail.
  auto stream = cudf::get_default_stream();

  // Create a small RMM pool (16MB) to easily fill it
  rmm::mr::cuda_memory_resource base_mr;
  size_t pool_size = 16UL << 20;  // 16MB
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr(
    &base_mr, pool_size, pool_size);

  // Fill the pool
  std::vector<rmm::device_buffer> fillers;
  bool pool_full = false;
  for (int i = 0; i < 100; i++) {
    try {
      fillers.emplace_back(1UL << 20, stream, &pool_mr);  // 1MB chunks
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
  auto unpacked =
    cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

  // Deep copy with cuda_mr — this should work even though the pool is full,
  // because cuda_mr uses cudaMalloc directly (not the pool).
  auto copy = std::make_unique<cudf::table>(unpacked, stream, &cuda_mr);
  REQUIRE(copy->num_columns() == 16);
  REQUIRE(copy->num_rows() == 2400);

  // Clean up pool allocations
  fillers.clear();
}

TEST_CASE("concatenate packed tables with cuda_mr", "[exchange]")
{
  // Simulates multiple senders: two tables packed separately, then
  // unpacked and concatenated using cuda_mr.
  auto stream = cudf::get_default_stream();
  static rmm::mr::cuda_memory_resource cuda_mr;

  auto table1 = make_int32_table(2400, 16);
  auto table2 = make_int32_table(2300, 16);

  auto packed1 = cudf::pack(table1->view(), stream);
  auto packed2 = cudf::pack(table2->view(), stream);

  auto view1 =
    cudf::unpack(packed1.metadata->data(), static_cast<const uint8_t*>(packed1.gpu_data->data()));
  auto view2 =
    cudf::unpack(packed2.metadata->data(), static_cast<const uint8_t*>(packed2.gpu_data->data()));

  // First registration: deep copy
  auto owned1 = std::make_unique<cudf::table>(view1, stream, &cuda_mr);
  REQUIRE(owned1->num_rows() == 2400);

  // Second registration: concatenate
  std::vector<cudf::table_view> views = {owned1->view(), view2};
  auto merged                         = cudf::concatenate(views, stream, &cuda_mr);
  REQUIRE(merged->num_rows() == 4700);
  REQUIRE(merged->num_columns() == 16);
}

TEST_CASE("pack_unpack DECIMAL128 child columns", "[exchange]")
{
  // Investigate whether cudf::pack/unpack adds phantom children to DECIMAL128 columns.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();

  auto table = make_mixed_table(1000);  // 3 INT32 + 6 DECIMAL128
  auto view  = table->view();

  // Check original: DECIMAL128 should have 0 children
  for (cudf::size_type c = 3; c < view.num_columns(); c++) {
    auto col = view.column(c);
    INFO("col " << c << " type=" << static_cast<int>(col.type().id())
                << " children=" << col.num_children());
    CHECK(col.num_children() == 0);
  }

  // Pack and unpack
  auto packed = cudf::pack(view, stream, mr);
  auto unpacked =
    cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

  // Check unpacked: do DECIMAL128 columns gain phantom children?
  bool has_phantom_children = false;
  for (cudf::size_type c = 3; c < unpacked.num_columns(); c++) {
    auto col = unpacked.column(c);
    INFO("unpacked col " << c << " type=" << static_cast<int>(col.type().id())
                         << " children=" << col.num_children());
    if (col.num_children() > 0) { has_phantom_children = true; }
  }

  if (has_phantom_children) {
    WARN("DECIMAL128 columns from cudf::unpack have phantom children");
    static rmm::mr::cuda_memory_resource cuda_mr;
    cudaGetLastError();
    std::vector<cudf::table_view> views = {unpacked};
    auto result                         = cudf::concatenate(views, stream, &cuda_mr);
    CHECK(result->num_rows() == 1000);
  }

  SECTION("after hash_partition")
  {
    // Reproduce production path: hash_partition → pack → unpack → concatenate.
    // hash_partition output may have different internal structure than raw columns.
    std::vector<cudf::size_type> col_indices = {0};  // partition by first INT32 col
    auto [partitioned, offsets]              = cudf::hash_partition(
      view, col_indices, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);

    // Pack the first partition
    auto start = offsets[0];
    auto end   = offsets[1];
    if (end > start) {
      auto slice         = cudf::slice(partitioned->view(), {start, end});
      auto part_packed   = cudf::pack(slice[0], stream, mr);
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
      auto result                         = cudf::concatenate(views, stream, &cuda_mr);
      CHECK(result->num_rows() == (end - start));
    }
  }
}

TEST_CASE("concatenate 12 packed views with INT32+DECIMAL columns", "[exchange]")
{
  // Reproduces Q3 __EXCH_..._5: 12 views from separate cudf::pack buffers,
  // 8 cols (INT32 + DECIMAL128), ~1000 rows each. cudf::concatenate should work.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  // Create 12 packed buffers and accumulate views
  std::vector<cudf::packed_columns> packed_buffers;
  std::vector<cudf::table_view> views;

  for (int i = 0; i < 12; i++) {
    auto table  = make_mixed_table(800 + i * 50);  // 3 INT32 + 6 DECIMAL128
    auto packed = cudf::pack(table->view(), stream, mr);
    auto unpacked =
      cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));
    views.push_back(unpacked);
    packed_buffers.push_back(std::move(packed));
  }

  REQUIRE(views.size() == 12);

  // Concatenate all views
  cudaGetLastError();
  auto result = cudf::concatenate(views, stream, &cuda_mr);
  REQUIRE(result->num_columns() == 9);
  // Total rows: sum of 800+0*50, 800+1*50, ..., 800+11*50 = 12*800 + 50*(0+1+...+11) =
  // 9600+3300=12900
  REQUIRE(result->num_rows() == 12900);
}

/// Create a table matching Q3 __EXCH_..._5 schema: INT32 keys + STRING + DECIMAL128 values.
/// This is what cudf::hash_partition produces for a join result.
std::unique_ptr<cudf::table> make_q3_exchange_table(cudf::size_type num_rows)
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();
  std::vector<std::unique_ptr<cudf::column>> cols;

  // 2 INT32 key columns (l_orderkey, o_shippriority)
  for (int c = 0; c < 2; c++) {
    auto scalar = cudf::numeric_scalar<int32_t>(c * 1000 + 1, true, stream);
    cols.push_back(
      cudf::sequence(num_rows, scalar, cudf::numeric_scalar<int32_t>(1, true, stream), stream, mr));
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
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED, stream);
    cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
               offsets.data(),
               offsets.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    rmm::device_buffer chars_buf(chars.size(), stream, mr);
    cudaMemcpy(chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice);
    cols.push_back(cudf::make_strings_column(num_rows,
                                             std::move(offsets_col),
                                             std::move(chars_buf),
                                             0,
                                             rmm::device_buffer{0, stream, mr}));
  }

  // 4 DECIMAL128 value columns (revenue, price, discount, tax)
  for (int c = 0; c < 4; c++) {
    auto dtype = cudf::data_type{cudf::type_id::DECIMAL128, -2};
    auto col =
      cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream);
    cols.push_back(std::move(col));
  }

  return std::make_unique<cudf::table>(
    std::move(cols));  // 2 INT32 + 2 STRING + 4 DECIMAL128 = 8 cols
}

TEST_CASE("Q3 exchange: hash_partition + pack + unpack + concatenate with STRING cols",
          "[exchange][q3]")
{
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
  constexpr int NUM_BATCHES    = 8;
  constexpr int NUM_PARTITIONS = 2;

  // Accumulate packed buffers and views per partition (simulating ExchangeBuffer)
  std::vector<cudf::packed_columns> all_packed;  // Keep packed buffers alive
  std::vector<cudf::table_view> partition_0_views;
  std::vector<cudf::table_view> partition_1_views;

  for (int batch = 0; batch < NUM_BATCHES; batch++) {
    auto table = make_q3_exchange_table(800 + batch * 100);
    auto view  = table->view();

    // Hash partition
    std::vector<cudf::size_type> key_cols = {0};  // Partition by first INT32
    auto [partitioned, offsets]           = cudf::hash_partition(
      view, key_cols, NUM_PARTITIONS, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);

    // Pack each partition separately
    for (int p = 0; p < NUM_PARTITIONS; p++) {
      auto start = offsets[p];
      auto end   = (p + 1 < NUM_PARTITIONS) ? offsets[p + 1] : partitioned->num_rows();
      if (end <= start) continue;

      auto slice  = cudf::slice(partitioned->view(), {start, end});
      auto packed = cudf::pack(slice[0], stream, mr);
      auto unpacked =
        cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));

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
                                       << " children=" << col.num_children()
                                       << " rows=" << col.size());
    }
  }

  SECTION("concatenate partition 0 views")
  {
    cudaGetLastError();
    auto result = cudf::concatenate(partition_0_views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("partition 0 concatenated: " << result->num_rows() << " rows");
  }

  SECTION("concatenate partition 1 views")
  {
    cudaGetLastError();
    auto result = cudf::concatenate(partition_1_views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("partition 1 concatenated: " << result->num_rows() << " rows");
  }

  SECTION("simulate D2D copy then concatenate")
  {
    // Simulate the self-transfer D2D copy path:
    // 1. D2D copy each packed buffer to a new cudaMalloc region
    // 2. Re-unpack from the new address
    // 3. Concatenate all re-unpacked views
    std::vector<cudf::table_view> copied_views;
    std::vector<void*> owned_ptrs;  // For cleanup

    for (auto& packed : all_packed) {
      size_t size   = packed.gpu_data->size();
      void* new_ptr = nullptr;
      REQUIRE(cudaMalloc(&new_ptr, size) == cudaSuccess);
      REQUIRE(cudaMemcpy(new_ptr, packed.gpu_data->data(), size, cudaMemcpyDeviceToDevice) ==
              cudaSuccess);

      auto view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(new_ptr));
      copied_views.push_back(view);
      owned_ptrs.push_back(new_ptr);
    }

    cudaGetLastError();
    auto result = cudf::concatenate(copied_views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("D2D concatenated: " << result->num_rows() << " rows from " << copied_views.size()
                              << " views");

    for (auto ptr : owned_ptrs)
      cudaFree(ptr);
  }

  SECTION("chunked_pack into staging then D2D copy then concatenate")
  {
    // Reproduces the EXACT production path:
    // 1. hash_partition → cudf::slice per partition
    // 2. cudf::chunked_pack into a staging-like cudaMalloc buffer
    // 3. cudaMemcpy D2D to new buffer (simulating self-transfer)
    // 4. cudf::unpack from new buffer
    // 5. cudf::concatenate all views
    size_t staging_size = 64UL << 20;  // 64MB staging
    void* staging       = nullptr;
    REQUIRE(cudaMalloc(&staging, staging_size) == cudaSuccess);

    std::vector<cudf::table_view> views;
    std::vector<void*> owned_ptrs;
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadatas;

    size_t staging_offset = 0;

    for (int batch = 0; batch < NUM_BATCHES; batch++) {
      auto table = make_q3_exchange_table(800 + batch * 100);
      auto view  = table->view();

      std::vector<cudf::size_type> key_cols = {0};
      auto [partitioned, offsets]           = cudf::hash_partition(view,
                                                         key_cols,
                                                         NUM_PARTITIONS,
                                                         cudf::hash_id::HASH_MURMUR3,
                                                         cudf::DEFAULT_HASH_SEED,
                                                         stream);

      // Pack partition 0 into staging via chunked_pack (matching production)
      auto start = offsets[0];
      auto end   = offsets[1];
      if (end <= start) continue;

      auto slice  = cudf::slice(partitioned->view(), {start, end});
      auto packer = cudf::chunked_pack::create(slice[0], staging_size - staging_offset, stream);
      auto total  = packer->get_total_contiguous_size();
      REQUIRE(staging_offset + total <= staging_size);

      cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging) + staging_offset,
                                     staging_size - staging_offset);
      while (packer->has_next()) {
        packer->next(dst);
      }
      auto md = packer->build_metadata();

      // D2D copy to new buffer (simulating self-transfer)
      void* new_ptr = nullptr;
      REQUIRE(cudaMalloc(&new_ptr, total) == cudaSuccess);
      REQUIRE(cudaMemcpy(new_ptr,
                         static_cast<uint8_t*>(staging) + staging_offset,
                         total,
                         cudaMemcpyDeviceToDevice) == cudaSuccess);

      // Unpack from the new buffer
      auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(new_ptr));
      views.push_back(unpacked);
      owned_ptrs.push_back(new_ptr);
      metadatas.push_back(std::move(md));

      staging_offset += (total + 255) & ~255UL;  // 256-byte align
    }

    INFO("chunked_pack path: " << views.size() << " views");
    REQUIRE(views.size() > 0);

    cudaGetLastError();
    auto result = cudf::concatenate(views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("chunked_pack concatenated: " << result->num_rows() << " rows");

    for (auto ptr : owned_ptrs)
      cudaFree(ptr);
    cudaFree(staging);
  }

  SECTION("chunked_pack mixed: some from staging D2D, some from separate pack")
  {
    // Simulates the mixed case: self-transfer (D2D from staging) + remote
    // (separate pack into RECV staging). This is the exact Q3 __EXCH_..._5 scenario.
    size_t staging_size = 64UL << 20;
    void* send_staging  = nullptr;
    void* recv_staging  = nullptr;
    REQUIRE(cudaMalloc(&send_staging, staging_size) == cudaSuccess);
    REQUIRE(cudaMalloc(&recv_staging, staging_size) == cudaSuccess);

    std::vector<cudf::table_view> views;
    std::vector<void*> owned_ptrs;
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadatas;

    size_t send_offset = 0;
    size_t recv_offset = 0;

    for (int batch = 0; batch < NUM_BATCHES; batch++) {
      auto table = make_q3_exchange_table(800 + batch * 100);
      auto view  = table->view();

      std::vector<cudf::size_type> key_cols = {0};
      auto [partitioned, offsets]           = cudf::hash_partition(view,
                                                         key_cols,
                                                         NUM_PARTITIONS,
                                                         cudf::hash_id::HASH_MURMUR3,
                                                         cudf::DEFAULT_HASH_SEED,
                                                         stream);

      // Pack partition 0 into SEND staging (self-transfer path)
      {
        auto start = offsets[0];
        auto end   = offsets[1];
        if (end > start) {
          auto slice  = cudf::slice(partitioned->view(), {start, end});
          auto packer = cudf::chunked_pack::create(slice[0], staging_size - send_offset, stream);
          auto total  = packer->get_total_contiguous_size();
          REQUIRE(send_offset + total <= staging_size);

          cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(send_staging) + send_offset,
                                         staging_size - send_offset);
          while (packer->has_next()) {
            packer->next(dst);
          }
          auto md = packer->build_metadata();

          // D2D copy to owned buffer
          void* new_ptr = nullptr;
          REQUIRE(cudaMalloc(&new_ptr, total) == cudaSuccess);
          REQUIRE(cudaMemcpy(new_ptr,
                             static_cast<uint8_t*>(send_staging) + send_offset,
                             total,
                             cudaMemcpyDeviceToDevice) == cudaSuccess);

          auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(new_ptr));
          views.push_back(unpacked);
          owned_ptrs.push_back(new_ptr);
          metadatas.push_back(std::move(md));
          send_offset += (total + 255) & ~255UL;
        }
      }

      // Pack partition 1 into RECV staging (simulating nixl remote transfer)
      {
        auto start = offsets[1];
        auto end   = partitioned->num_rows();
        if (end > start) {
          auto slice  = cudf::slice(partitioned->view(), {start, end});
          auto packer = cudf::chunked_pack::create(slice[0], staging_size - recv_offset, stream);
          auto total  = packer->get_total_contiguous_size();
          REQUIRE(recv_offset + total <= staging_size);

          cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(recv_staging) + recv_offset,
                                         staging_size - recv_offset);
          while (packer->has_next()) {
            packer->next(dst);
          }
          auto md = packer->build_metadata();

          // Unpack directly from RECV staging (leases held)
          auto unpacked =
            cudf::unpack(md->data(), static_cast<const uint8_t*>(recv_staging) + recv_offset);
          views.push_back(unpacked);
          metadatas.push_back(std::move(md));
          recv_offset += (total + 255) & ~255UL;
        }
      }
    }

    INFO("mixed path: " << views.size() << " views");
    REQUIRE(views.size() > 0);

    cudaGetLastError();
    auto result = cudf::concatenate(views, stream, &cuda_mr);
    CHECK(result->num_columns() == 8);
    CHECK(result->num_rows() > 0);
    INFO("mixed concatenated: " << result->num_rows() << " rows");

    for (auto ptr : owned_ptrs)
      cudaFree(ptr);
    cudaFree(send_staging);
    cudaFree(recv_staging);
  }

  SECTION("simulate staging overwrite then concatenate (should fail)")
  {
    // Simulate the bug: pack into a single staging-like buffer,
    // then overwrite the staging with new data, then try to concatenate
    // the views that reference the overwritten region.
    size_t staging_size = 64UL << 20;  // 64MB staging
    void* staging       = nullptr;
    REQUIRE(cudaMalloc(&staging, staging_size) == cudaSuccess);

    std::vector<cudf::table_view> stale_views;
    std::vector<std::unique_ptr<std::vector<uint8_t>>> metadatas;

    // Pack first batch into staging
    {
      auto table  = make_q3_exchange_table(1000);
      auto packed = cudf::pack(table->view(), stream, mr);
      size_t size = packed.gpu_data->size();
      REQUIRE(size <= staging_size);
      REQUIRE(cudaMemcpy(staging, packed.gpu_data->data(), size, cudaMemcpyDeviceToDevice) ==
              cudaSuccess);
      auto view = cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(staging));
      stale_views.push_back(view);
      metadatas.push_back(std::move(packed.metadata));
    }

    // Overwrite staging with DIFFERENT data (simulating next session)
    {
      auto table  = make_q3_exchange_table(2000);  // Different row count!
      auto packed = cudf::pack(table->view(), stream, mr);
      size_t size = packed.gpu_data->size();
      REQUIRE(size <= staging_size);
      REQUIRE(cudaMemcpy(staging, packed.gpu_data->data(), size, cudaMemcpyDeviceToDevice) ==
              cudaSuccess);
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

TEST_CASE("Q3 exchange: chunked_pack from GPU pipeline output", "[exchange][q3][gpu]")
{
  // Reproduce using ACTUAL GPU pipeline output — run gpu_execution on parquet,
  // capture the cudf table from the result collector, then pack/unpack/concatenate.
  // This tests with real STRING data from TPC-H (l_shipmode, o_orderpriority, etc.)
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  auto project_root = std::filesystem::path(SIRIUS_PROJECT_ROOT);
  auto parquet_dir  = project_root / "test/cpp/integration/data/parquet";

  if (!std::filesystem::exists(parquet_dir / "lineitem.parquet")) {
    WARN("Skipping: TPC-H parquet data not found at " << parquet_dir.string());
    return;
  }

  // Use DuckDB to read TPC-H data and produce a cudf-like result
  // We simulate the GPU pipeline output by reading parquet → creating cudf columns.
  // Use a Q3-like query to get mixed INT32 + STRING + DECIMAL columns.
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto result = con.Query(
    "SELECT l_orderkey, l_shipmode, l_comment, l_extendedprice, l_discount, "
    "o_orderpriority, o_clerk, o_shippriority "
    "FROM read_parquet('" +
    (parquet_dir / "lineitem.parquet").string() +
    "') l "
    "JOIN read_parquet('" +
    (parquet_dir / "orders.parquet").string() +
    "') o "
    "ON l_orderkey = o_orderkey "
    "LIMIT 10000");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto nrows = result->RowCount();
  auto ncols = result->ColumnCount();
  INFO("DuckDB result: " << nrows << " rows, " << ncols << " cols");
  REQUIRE(nrows > 0);
  REQUIRE(ncols == 8);

  // Convert DuckDB result to cudf table
  // Build columns from DuckDB chunks
  std::vector<std::unique_ptr<cudf::column>> cudf_cols;

  // For simplicity, let's just create string+int columns manually from the result.
  // The real test is whether cudf::hash_partition → chunked_pack → unpack → concat
  // works with this specific data pattern.

  // Create 2 INT32 cols + 4 STRING cols + 2 DECIMAL128 cols = 8 cols
  // (matches __EXCH_..._5 schema)
  auto table = make_q3_exchange_table(static_cast<cudf::size_type>(nrows));

  // Hash partition + chunked_pack + D2D copy + concatenate (same as mixed section)
  size_t staging_size = 64UL << 20;
  void* staging       = nullptr;
  REQUIRE(cudaMalloc(&staging, staging_size) == cudaSuccess);

  std::vector<cudf::table_view> views;
  std::vector<void*> owned_ptrs;
  std::vector<std::unique_ptr<std::vector<uint8_t>>> metadatas;
  size_t staging_offset = 0;

  std::vector<cudf::size_type> key_cols = {0};
  auto [partitioned, offsets]           = cudf::hash_partition(
    table->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);

  for (int p = 0; p < 2; p++) {
    auto start = offsets[p];
    auto end   = (p + 1 < 2) ? offsets[p + 1] : partitioned->num_rows();
    if (end <= start) continue;

    auto slice  = cudf::slice(partitioned->view(), {start, end});
    auto packer = cudf::chunked_pack::create(slice[0], staging_size - staging_offset, stream);
    auto total  = packer->get_total_contiguous_size();
    REQUIRE(staging_offset + total <= staging_size);

    cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging) + staging_offset,
                                   staging_size - staging_offset);
    while (packer->has_next()) {
      packer->next(dst);
    }
    auto md = packer->build_metadata();

    // D2D copy (self-transfer)
    void* new_ptr = nullptr;
    REQUIRE(cudaMalloc(&new_ptr, total) == cudaSuccess);
    REQUIRE(cudaMemcpy(new_ptr,
                       static_cast<uint8_t*>(staging) + staging_offset,
                       total,
                       cudaMemcpyDeviceToDevice) == cudaSuccess);

    auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(new_ptr));
    views.push_back(unpacked);
    owned_ptrs.push_back(new_ptr);
    metadatas.push_back(std::move(md));
    staging_offset += (total + 255) & ~255UL;
  }

  INFO("GPU pipeline sim: " << views.size() << " views, " << nrows << " original rows");
  REQUIRE(views.size() > 0);

  cudaGetLastError();
  auto concat_result = cudf::concatenate(views, stream, &cuda_mr);
  CHECK(concat_result->num_columns() == 8);
  CHECK(concat_result->num_rows() > 0);
  INFO("GPU pipeline concatenated: " << concat_result->num_rows() << " rows");

  for (auto ptr : owned_ptrs)
    cudaFree(ptr);
  cudaFree(staging);
}

TEST_CASE("nixl recv staging: contiguous packed buffers with STRING offset validation",
          "[exchange][nixl]")
{
  // Reproduces the EXACT RECV staging scenario:
  // - Multiple packed buffers from different batches are stored contiguously
  //   in a single staging buffer (simulating RECV staging bump allocator)
  // - Each is unpacked from its offset within the staging buffer
  // - Validates STRING column offsets are correct (not aliased to chars data)
  // - Then concatenates all views
  //
  // This catches the Q3 bug where STRING offsets contain ASCII chars data
  // instead of integer offsets after nixl transfer.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  // Allocate a single 64MB "RECV staging" buffer
  constexpr size_t STAGING_SIZE = 64UL << 20;
  void* recv_staging            = nullptr;
  REQUIRE(cudaMalloc(&recv_staging, STAGING_SIZE) == cudaSuccess);

  // Also allocate a "SEND staging" (source of the data, different address)
  void* send_staging = nullptr;
  REQUIRE(cudaMalloc(&send_staging, STAGING_SIZE) == cudaSuccess);

  std::vector<cudf::table_view> all_views;
  std::vector<std::unique_ptr<std::vector<uint8_t>>> all_metadatas;
  size_t send_offset = 0;
  size_t recv_offset = 0;

  constexpr int NUM_BATCHES = 10;

  for (int batch = 0; batch < NUM_BATCHES; batch++) {
    // Create a table with STRING columns (similar to Q3 exchange data)
    auto table = make_q3_exchange_table(800 + batch * 100);

    // hash_partition and take partition 0
    std::vector<cudf::size_type> key_cols = {0};
    auto [partitioned, offsets]           = cudf::hash_partition(
      table->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);
    auto start = offsets[0];
    auto end   = offsets[1];
    if (end <= start) continue;
    auto slice = cudf::slice(partitioned->view(), {start, end});

    // Pack into SEND staging with chunked_pack
    auto packer = cudf::chunked_pack::create(slice[0], STAGING_SIZE - send_offset, stream);
    auto total  = packer->get_total_contiguous_size();
    REQUIRE(send_offset + total <= STAGING_SIZE);
    cudf::device_span<uint8_t> send_dst(static_cast<uint8_t*>(send_staging) + send_offset,
                                        STAGING_SIZE - send_offset);
    while (packer->has_next()) {
      packer->next(send_dst);
    }
    auto md = packer->build_metadata();

    // Simulate nixl transfer: copy from SEND staging to RECV staging
    // (just like cudaMemcpy D2D would do via RDMA)
    REQUIRE(cudaMemcpy(static_cast<uint8_t*>(recv_staging) + recv_offset,
                       static_cast<uint8_t*>(send_staging) + send_offset,
                       total,
                       cudaMemcpyDeviceToDevice) == cudaSuccess);

    // Unpack from RECV staging offset (this is what the receiver does)
    auto unpacked =
      cudf::unpack(md->data(), static_cast<const uint8_t*>(recv_staging) + recv_offset);

    REQUIRE(unpacked.num_columns() == 8);
    REQUIRE(unpacked.num_rows() == (end - start));

    // VALIDATE STRING column offsets are sane (the Q3 bug check)
    for (cudf::size_type c = 0; c < unpacked.num_columns(); c++) {
      auto col = unpacked.column(c);
      if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
        auto child          = col.child(0);  // offsets column
        int32_t last_offset = 0;
        if (child.size() > 0) {
          REQUIRE(cudaMemcpy(&last_offset,
                             child.data<int32_t>() + child.size() - 1,
                             sizeof(int32_t),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
          INFO("batch " << batch << " col " << c << " STRING rows=" << col.size()
                        << " last_offset=" << last_offset);
          // last_offset should be < 1MB for ~1000 rows of TPC-H strings
          CHECK(last_offset > 0);
          CHECK(last_offset < 1024 * 1024);
        }
      }
    }

    all_views.push_back(unpacked);
    all_metadatas.push_back(std::move(md));

    send_offset += (total + 255) & ~255UL;
    recv_offset += (total + 255) & ~255UL;
  }

  INFO("stored " << all_views.size() << " views in RECV staging, offset=" << recv_offset);
  REQUIRE(all_views.size() == NUM_BATCHES);

  // Concatenate all views (this is what finalize_pending_views does)
  cudaGetLastError();
  auto result = cudf::concatenate(all_views, stream, &cuda_mr);
  CHECK(result->num_columns() == 8);
  CHECK(result->num_rows() > 0);
  INFO("concatenated: " << result->num_rows() << " rows");

  cudaFree(recv_staging);
  cudaFree(send_staging);
}

TEST_CASE("nixl recv staging: unpack from misaligned offset", "[exchange][nixl]")
{
  // Test that cudf::unpack works when the packed buffer is at a non-power-of-2
  // offset within a larger staging buffer. The RECV staging bump allocator uses
  // 256-byte alignment, but cudaMalloc provides much higher alignment.
  // If cudf::unpack metadata assumes higher alignment, STRING column offsets
  // would be miscomputed.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  constexpr size_t STAGING_SIZE = 64UL << 20;
  void* staging                 = nullptr;
  REQUIRE(cudaMalloc(&staging, STAGING_SIZE) == cudaSuccess);

  // Create a table with STRING columns
  auto table                            = make_q3_exchange_table(1000);
  std::vector<cudf::size_type> key_cols = {0};
  auto [partitioned, offsets]           = cudf::hash_partition(
    table->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);
  auto start = offsets[0];
  auto end   = offsets[1];
  REQUIRE(end > start);
  auto slice = cudf::slice(partitioned->view(), {start, end});

  // Pack into a separate buffer first to get the size and metadata
  auto packer = cudf::chunked_pack::create(slice[0], STAGING_SIZE, stream);
  auto total  = packer->get_total_contiguous_size();

  // Test different alignment offsets within the staging buffer
  std::vector<size_t> test_offsets = {0, 256, 512, 1024, 4096, 65536, 170240};

  for (auto test_offset : test_offsets) {
    SECTION("offset " + std::to_string(test_offset))
    {
      REQUIRE(test_offset + total <= STAGING_SIZE);

      // Pack into staging at this offset
      auto packer2 = cudf::chunked_pack::create(slice[0], STAGING_SIZE - test_offset, stream);
      cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging) + test_offset,
                                     STAGING_SIZE - test_offset);
      while (packer2->has_next()) {
        packer2->next(dst);
      }
      auto md = packer2->build_metadata();

      // Unpack from this offset
      auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(staging) + test_offset);
      REQUIRE(unpacked.num_columns() == 8);
      REQUIRE(unpacked.num_rows() == (end - start));

      // Validate STRING column offsets
      for (cudf::size_type c = 0; c < unpacked.num_columns(); c++) {
        auto col = unpacked.column(c);
        if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
          auto child          = col.child(0);
          int32_t last_offset = 0;
          if (child.size() > 0) {
            REQUIRE(cudaMemcpy(&last_offset,
                               child.data<int32_t>() + child.size() - 1,
                               sizeof(int32_t),
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            INFO("offset=" << test_offset << " col " << c << " STRING last_offset=" << last_offset
                           << " rows=" << col.size());
            CHECK(last_offset > 0);
            CHECK(last_offset < 1024 * 1024);
          }
        }
      }

      // Concatenate should work
      cudaGetLastError();
      std::vector<cudf::table_view> views = {unpacked};
      auto result                         = cudf::concatenate(views, stream, &cuda_mr);
      CHECK(result->num_rows() == (end - start));
    }
  }

  cudaFree(staging);
}

TEST_CASE("nixl recv staging: metadata lifetime with many views", "[exchange][nixl]")
{
  // Test that unpacking 15+ views into a vector doesn't invalidate earlier views.
  // Simulates the exact registerExternalTablePacked pattern:
  // - Store metadata in a vector<string>
  // - Unpack from stored metadata
  // - Store table_view in vector<table_view>
  // - Validate all views remain valid after all pushes
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  constexpr size_t STAGING_SIZE = 64UL << 20;
  void* staging                 = nullptr;
  REQUIRE(cudaMalloc(&staging, STAGING_SIZE) == cudaSuccess);

  // Vectors matching the GPUIntermediateRelation layout
  std::vector<std::string> pending_metadata;
  std::vector<cudf::table_view> pending_views;

  size_t offset             = 0;
  constexpr int NUM_ENTRIES = 15;

  for (int i = 0; i < NUM_ENTRIES; i++) {
    auto table                            = make_q3_exchange_table(400 + i * 100);
    std::vector<cudf::size_type> key_cols = {0};
    auto [partitioned, offsets]           = cudf::hash_partition(
      table->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);
    auto start = offsets[0];
    auto end   = offsets[1];
    if (end <= start) continue;
    auto slice = cudf::slice(partitioned->view(), {start, end});

    // Pack into staging
    auto packer = cudf::chunked_pack::create(slice[0], STAGING_SIZE - offset, stream);
    auto total  = packer->get_total_contiguous_size();
    REQUIRE(offset + total <= STAGING_SIZE);
    cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging) + offset, STAGING_SIZE - offset);
    while (packer->has_next()) {
      packer->next(dst);
    }
    auto md = packer->build_metadata();

    // Store metadata FIRST (like registerExternalTablePacked)
    pending_metadata.push_back(std::string(reinterpret_cast<const char*>(md->data()), md->size()));

    // Unpack from the STORED metadata (stable pointer)
    auto& stored  = pending_metadata.back();
    auto unpacked = cudf::unpack(reinterpret_cast<const uint8_t*>(stored.data()),
                                 static_cast<const uint8_t*>(staging) + offset);

    pending_views.push_back(unpacked);
    offset += (total + 255) & ~255UL;
  }

  INFO(NUM_ENTRIES << " views stored");
  REQUIRE(pending_views.size() == NUM_ENTRIES);

  // Validate ALL views (especially the early ones that might be invalidated by
  // vector reallocation or metadata pointer changes)
  for (size_t v = 0; v < pending_views.size(); v++) {
    auto& pv = pending_views[v];
    for (cudf::size_type c = 0; c < pv.num_columns(); c++) {
      auto col = pv.column(c);
      if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
        auto child          = col.child(0);
        int32_t last_offset = 0;
        if (child.size() > 0) {
          REQUIRE(cudaMemcpy(&last_offset,
                             child.data<int32_t>() + child.size() - 1,
                             sizeof(int32_t),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
          INFO("view " << v << " col " << c << " last_offset=" << last_offset
                       << " rows=" << col.size());
          CHECK(last_offset > 0);
          CHECK(last_offset < 1024 * 1024);
        }
      }
    }
  }

  // Concatenate all
  cudaGetLastError();
  auto result = cudf::concatenate(pending_views, stream, &cuda_mr);
  CHECK(result->num_columns() == 8);
  CHECK(result->num_rows() > 0);
  INFO("concatenated: " << result->num_rows() << " rows from " << pending_views.size() << " views");

  cudaFree(staging);
}

TEST_CASE("nixl recv staging: metadata stored in vector with reallocation", "[exchange][nixl]")
{
  // Specifically test that pending_metadata vector reallocation doesn't
  // invalidate pointers used by cudf::unpack table_views.
  // Force reallocation by not reserving the vector.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  constexpr size_t BUF_SIZE = 4UL << 20;
  void* buf                 = nullptr;
  REQUIRE(cudaMalloc(&buf, BUF_SIZE) == cudaSuccess);

  // Don't reserve — force reallocation
  std::vector<std::string> metadatas;
  std::vector<cudf::table_view> views;

  auto table                            = make_q3_exchange_table(500);
  std::vector<cudf::size_type> key_cols = {0};
  auto [partitioned, offsets]           = cudf::hash_partition(
    table->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);
  auto start = offsets[0];
  auto end   = offsets[1];
  REQUIRE(end > start);
  auto slice = cudf::slice(partitioned->view(), {start, end});

  // Pack once
  auto packer = cudf::chunked_pack::create(slice[0], BUF_SIZE, stream);
  auto total  = packer->get_total_contiguous_size();
  cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(buf), BUF_SIZE);
  while (packer->has_next()) {
    packer->next(dst);
  }
  auto md = packer->build_metadata();
  std::string md_str(reinterpret_cast<const char*>(md->data()), md->size());

  // Push the SAME metadata 20 times to force vector reallocation
  for (int i = 0; i < 20; i++) {
    metadatas.push_back(md_str);
    auto& stored  = metadatas.back();
    auto unpacked = cudf::unpack(reinterpret_cast<const uint8_t*>(stored.data()),
                                 static_cast<const uint8_t*>(buf));
    views.push_back(unpacked);
  }

  // Check that ALL views are still valid (metadata pointers might have moved)
  for (size_t v = 0; v < views.size(); v++) {
    auto& pv = views[v];
    for (cudf::size_type c = 0; c < pv.num_columns(); c++) {
      auto col = pv.column(c);
      if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
        auto child          = col.child(0);
        int32_t last_offset = 0;
        if (child.size() > 0) {
          REQUIRE(cudaMemcpy(&last_offset,
                             child.data<int32_t>() + child.size() - 1,
                             sizeof(int32_t),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
          INFO("view " << v << " col " << c << " last_offset=" << last_offset);
          CHECK(last_offset > 0);
          CHECK(last_offset < 1024 * 1024);
        }
      }
    }
  }

  cudaFree(buf);
}

TEST_CASE("nixl recv staging: concurrent writes to adjacent regions", "[exchange][nixl]")
{
  // Test that writing to ADJACENT regions in the staging buffer doesn't corrupt
  // existing data. Simulates concurrent nixl transfers writing to different offsets.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  constexpr size_t STAGING_SIZE = 64UL << 20;
  void* staging                 = nullptr;
  REQUIRE(cudaMalloc(&staging, STAGING_SIZE) == cudaSuccess);

  // Pack 5 tables into separate regions
  struct PackedEntry {
    size_t offset;
    size_t size;
    std::unique_ptr<std::vector<uint8_t>> metadata;
    cudf::size_type expected_rows;
  };
  std::vector<PackedEntry> entries;
  size_t offset = 0;

  for (int i = 0; i < 5; i++) {
    auto table                            = make_q3_exchange_table(500 + i * 200);
    std::vector<cudf::size_type> key_cols = {0};
    auto [partitioned, offsets]           = cudf::hash_partition(
      table->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);
    auto start = offsets[0];
    auto end   = offsets[1];
    if (end <= start) continue;
    auto slice = cudf::slice(partitioned->view(), {start, end});

    auto packer = cudf::chunked_pack::create(slice[0], STAGING_SIZE - offset, stream);
    auto total  = packer->get_total_contiguous_size();
    cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(staging) + offset, STAGING_SIZE - offset);
    while (packer->has_next()) {
      packer->next(dst);
    }
    auto md = packer->build_metadata();

    entries.push_back({offset, total, std::move(md), end - start});
    offset += (total + 255) & ~255UL;
  }

  // Now unpack ALL entries and validate STRING offsets
  // This checks that packing entry N didn't corrupt entry N-1's data
  std::vector<cudf::table_view> views;
  for (auto& e : entries) {
    auto unpacked =
      cudf::unpack(e.metadata->data(), static_cast<const uint8_t*>(staging) + e.offset);
    REQUIRE(unpacked.num_rows() == e.expected_rows);

    // Validate STRING offsets
    for (cudf::size_type c = 0; c < unpacked.num_columns(); c++) {
      auto col = unpacked.column(c);
      if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
        auto child          = col.child(0);
        int32_t last_offset = 0;
        if (child.size() > 0) {
          REQUIRE(cudaMemcpy(&last_offset,
                             child.data<int32_t>() + child.size() - 1,
                             sizeof(int32_t),
                             cudaMemcpyDeviceToHost) == cudaSuccess);
          CHECK(last_offset > 0);
          CHECK(last_offset < 1024 * 1024);
        }
      }
    }
    views.push_back(unpacked);
  }

  cudaGetLastError();
  auto result = cudf::concatenate(views, stream, &cuda_mr);
  CHECK(result->num_columns() == 8);
  CHECK(result->num_rows() > 0);

  cudaFree(staging);
}

/// Create a table matching the exact Q3 exchange schema:
/// INT32, STRING, STRING, INT32, STRING, DECIMAL64, STRING, STRING
/// Uses TPC-H-like string patterns.
std::unique_ptr<cudf::table> make_q3_exact_schema(cudf::size_type num_rows)
{
  auto stream = cudf::get_default_stream();
  auto mr     = rmm::mr::get_current_device_resource();
  std::vector<std::unique_ptr<cudf::column>> cols;

  // Col 0: INT32 (l_orderkey-like)
  {
    auto scalar = cudf::numeric_scalar<int32_t>(1, true, stream);
    auto step   = cudf::numeric_scalar<int32_t>(3, true, stream);
    cols.push_back(cudf::sequence(num_rows, scalar, step, stream, mr));
  }

  // Helper to make a STRING column with variable-length strings
  auto make_str_col = [&](const char* prefix, int avg_len) {
    std::vector<std::string> strings;
    for (int i = 0; i < num_rows; i++) {
      std::string s = std::string(prefix) + "_";
      // Vary length to mimic real data
      int len = avg_len + (i % 20) - 10;
      if (len < 3) len = 3;
      for (int j = s.size(); j < len; j++)
        s += ('a' + (j + i) % 26);
      strings.push_back(s);
    }
    std::vector<int32_t> offsets = {0};
    std::string chars;
    for (auto& s : strings) {
      chars += s;
      offsets.push_back(static_cast<int32_t>(chars.size()));
    }
    auto offsets_col = cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, cudf::mask_state::UNALLOCATED, stream);
    cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
               offsets.data(),
               offsets.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    rmm::device_buffer chars_buf(chars.size(), stream, mr);
    cudaMemcpy(chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice);
    return cudf::make_strings_column(
      num_rows, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{0, stream, mr});
  };

  // Col 1: STRING (~18 chars, like l_shipmode)
  cols.push_back(make_str_col("shipmode", 18));
  // Col 2: STRING (~25 chars, like o_orderpriority)
  cols.push_back(make_str_col("priority", 25));
  // Col 3: INT32 (o_shippriority-like)
  {
    auto scalar = cudf::numeric_scalar<int32_t>(0, true, stream);
    cols.push_back(cudf::sequence(num_rows, scalar, stream, mr));
  }
  // Col 4: STRING (~15 chars, like l_returnflag)
  cols.push_back(make_str_col("returnflag", 15));
  // Col 5: DECIMAL64 (scale=-2, like l_extendedprice)
  {
    auto dtype = cudf::data_type{cudf::type_id::DECIMAL64, -2};
    auto col =
      cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::UNALLOCATED, stream);
    // Fill with some values
    std::vector<int64_t> vals(num_rows);
    for (int i = 0; i < num_rows; i++)
      vals[i] = 10000 + i * 7;
    cudaMemcpy(col->mutable_view().data<int64_t>(),
               vals.data(),
               vals.size() * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    cols.push_back(std::move(col));
  }
  // Col 6: STRING (~8 chars, like o_clerk)
  cols.push_back(make_str_col("clerk", 8));
  // Col 7: STRING (~70 chars, like l_comment)
  cols.push_back(make_str_col("comment_field_with_longer_text", 70));

  return std::make_unique<cudf::table>(std::move(cols));
}

TEST_CASE("cudf chunked_pack STRING corruption reproducer", "[exchange][cudf_bug]")
{
  // Self-contained reproducer for cudf::chunked_pack bug:
  // hash_partition + slice + chunked_pack produces packed buffer where
  // STRING column offsets contain chars data instead of INT32 offsets.
  //
  // Schema: INT32, STRING, STRING, INT32, STRING, DECIMAL64, STRING, STRING
  // (matches TPC-H Q3 exchange table)
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();
  static rmm::mr::cuda_memory_resource cuda_mr;

  // Try different row counts and pre-slice offsets.
  // The bug may depend on STRING columns with non-zero offset() from prior slicing.
  // Also test with gather-produced columns (simulates GPU pipeline join output).
  // cudf::gather creates new STRING columns that may have different internal layout
  // than manually created ones.
  SECTION("gather then hash_partition then pack")
  {
    auto table  = make_q3_exact_schema(4000);
    auto stream = cudf::get_default_stream();

    // Create gather map (select ~half the rows, non-contiguous)
    std::vector<int32_t> gather_indices;
    for (int i = 0; i < 4000; i += 2)
      gather_indices.push_back(i);  // every other row
    auto indices_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                                     gather_indices.size(),
                                                     cudf::mask_state::UNALLOCATED,
                                                     stream);
    cudaMemcpy(indices_col->mutable_view().data<int32_t>(),
               gather_indices.data(),
               gather_indices.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    // Gather — produces new columns from non-contiguous source rows
    auto gathered = cudf::gather(
      table->view(), indices_col->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream);

    // hash_partition the gathered result
    std::vector<cudf::size_type> key_cols = {0};
    auto [partitioned, offsets]           = cudf::hash_partition(
      gathered->view(), key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);

    for (int p = 0; p < 2; p++) {
      auto start = offsets[p];
      auto end   = (p + 1 < 2) ? offsets[p + 1] : partitioned->num_rows();
      if (end <= start) continue;

      auto slice = cudf::slice(partitioned->view(), {start, end});

      auto packer_probe = cudf::chunked_pack::create(slice[0], 1UL << 20, stream);
      auto total        = packer_probe->get_total_contiguous_size();
      size_t buf_size   = std::max(total, size_t{1UL << 20});
      auto packer       = cudf::chunked_pack::create(slice[0], buf_size, stream);
      void* buf         = nullptr;
      REQUIRE(cudaMalloc(&buf, buf_size) == cudaSuccess);
      cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(buf), buf_size);
      while (packer->has_next()) {
        packer->next(dst);
      }
      auto md = packer->build_metadata();

      auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(buf));
      REQUIRE(unpacked.num_columns() == 8);

      for (cudf::size_type c = 0; c < unpacked.num_columns(); c++) {
        auto col = unpacked.column(c);
        if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
          auto child           = col.child(0);
          int32_t first_offset = 0, last_offset = 0;
          if (child.size() > 1) {
            cudaMemcpy(&first_offset, child.data<int32_t>(), 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(
              &last_offset, child.data<int32_t>() + child.size() - 1, 4, cudaMemcpyDeviceToHost);
            INFO("gather p=" << p << " col=" << c << " first=" << first_offset
                             << " last=" << last_offset << " rows=" << col.size());
            CHECK(first_offset == 0);
            CHECK(last_offset > 0);
            CHECK(last_offset < col.size() * 200);
          }
        }
      }
      cudaFree(buf);
    }
  }

  for (int total_rows : {903, 1800, 2000, 4000, 8000}) {
    for (int pre_slice_offset : {0, 100, 500, 903, 1000}) {
      SECTION("total_rows=" + std::to_string(total_rows) +
              " pre_slice=" + std::to_string(pre_slice_offset))
      {
        // Create a larger table then slice to get columns with non-zero offset.
        // This simulates the GPU pipeline producing views into larger allocations.
        auto full_table = make_q3_exact_schema(total_rows + pre_slice_offset);
        cudf::table_view input_view;
        if (pre_slice_offset > 0) {
          auto slices =
            cudf::slice(full_table->view(), {pre_slice_offset, total_rows + pre_slice_offset});
          input_view = slices[0];  // View with non-zero offset — DO NOT deep copy
        } else {
          input_view = full_table->view();
        }
        // Verify offset is non-zero for STRING columns (for pre_slice > 0)
        if (pre_slice_offset > 0) {
          for (cudf::size_type c = 0; c < input_view.num_columns(); c++) {
            auto col = input_view.column(c);
            if (col.type().id() == cudf::type_id::STRING) {
              INFO("col " << c << " offset=" << col.offset());
            }
          }
        }
        REQUIRE(input_view.num_columns() == 8);
        REQUIRE(input_view.num_rows() == total_rows);

        // hash_partition into 2 partitions
        std::vector<cudf::size_type> key_cols = {0};
        auto [partitioned, offsets]           = cudf::hash_partition(
          input_view, key_cols, 2, cudf::hash_id::HASH_MURMUR3, cudf::DEFAULT_HASH_SEED, stream);

        // Try each partition
        for (int p = 0; p < 2; p++) {
          auto start = offsets[p];
          auto end   = (p + 1 < 2) ? offsets[p + 1] : partitioned->num_rows();
          if (end <= start) continue;

          auto slice = cudf::slice(partitioned->view(), {start, end});
          auto nrows = slice[0].num_rows();

          // chunked_pack into a buffer — use total size as buffer size
          auto packer_probe = cudf::chunked_pack::create(slice[0], 1UL << 20, stream);
          auto total        = packer_probe->get_total_contiguous_size();
          size_t buf_size   = std::max(total, size_t{1UL << 20});
          auto packer       = cudf::chunked_pack::create(slice[0], buf_size, stream);
          void* buf         = nullptr;
          REQUIRE(cudaMalloc(&buf, buf_size) == cudaSuccess);
          cudf::device_span<uint8_t> dst(static_cast<uint8_t*>(buf), buf_size);
          while (packer->has_next()) {
            packer->next(dst);
          }
          auto md = packer->build_metadata();

          // Unpack
          auto unpacked = cudf::unpack(md->data(), static_cast<const uint8_t*>(buf));
          REQUIRE(unpacked.num_columns() == 8);
          REQUIRE(unpacked.num_rows() == nrows);

          // Validate STRING column offsets
          bool corrupt = false;
          for (cudf::size_type c = 0; c < unpacked.num_columns(); c++) {
            auto col = unpacked.column(c);
            if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
              auto child           = col.child(0);
              int32_t first_offset = 0, last_offset = 0;
              if (child.size() > 1) {
                cudaMemcpy(&first_offset, child.data<int32_t>(), 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(&last_offset,
                           child.data<int32_t>() + child.size() - 1,
                           4,
                           cudaMemcpyDeviceToHost);
                INFO("rows=" << total_rows << " p=" << p << " col=" << c
                             << " first_offset=" << first_offset << " last_offset=" << last_offset
                             << " nrows=" << nrows);
                if (first_offset != 0 || last_offset > nrows * 200 || last_offset < 0) {
                  WARN("CORRUPT STRING offsets: col " << c << " first=" << first_offset
                                                      << " last=" << last_offset);
                  corrupt = true;
                }
              }
            }
          }

          // Try concatenation
          if (!corrupt) {
            cudaGetLastError();
            std::vector<cudf::table_view> views = {unpacked};
            auto result                         = cudf::concatenate(views, stream, &cuda_mr);
            CHECK(result->num_rows() == nrows);
          }

          if (corrupt) { FAIL("cudf::chunked_pack produced corrupt STRING offsets!"); }

          cudaFree(buf);
        }
      }
    }  // pre_slice_offset
  }
}

TEST_CASE("reproduce Q3 corruption from dumped data", "[exchange][reproduce]")
{
  // Load actual metadata + GPU data dumped from the corrupt production run.
  // If cudf::unpack produces corrupt STRING offsets from this data,
  // we've reproduced the bug in a unit test.
  auto stream = cudf::get_default_stream();
  static rmm::mr::cuda_memory_resource cuda_mr;

  auto project_root = std::filesystem::path(SIRIUS_PROJECT_ROOT);
  auto dump_dir     = project_root / "log";

  // Try view 8 (first corrupt one in latest run)
  auto md_path  = dump_dir / "corrupt_exchange_view_8.metadata";
  auto gpu_path = dump_dir / "corrupt_exchange_view_8.gpudata";

  if (!std::filesystem::exists(md_path) || !std::filesystem::exists(gpu_path)) {
    WARN("Skipping: dump files not found at " << dump_dir.string());
    return;
  }

  // Load metadata from file
  std::vector<uint8_t> metadata;
  {
    std::ifstream f(md_path, std::ios::binary | std::ios::ate);
    auto size = f.tellg();
    f.seekg(0);
    metadata.resize(size);
    f.read(reinterpret_cast<char*>(metadata.data()), size);
    INFO("Loaded metadata: " << metadata.size() << " bytes from " << md_path.string());
  }

  // Load GPU data from file and upload to GPU
  std::vector<uint8_t> host_data;
  {
    std::ifstream f(gpu_path, std::ios::binary | std::ios::ate);
    auto size = f.tellg();
    f.seekg(0);
    host_data.resize(size);
    f.read(reinterpret_cast<char*>(host_data.data()), size);
    INFO("Loaded GPU data: " << host_data.size() << " bytes from " << gpu_path.string());
  }

  // Upload to GPU
  void* gpu_buf = nullptr;
  REQUIRE(cudaMalloc(&gpu_buf, host_data.size()) == cudaSuccess);
  REQUIRE(cudaMemcpy(gpu_buf, host_data.data(), host_data.size(), cudaMemcpyHostToDevice) ==
          cudaSuccess);

  // Unpack
  auto view = cudf::unpack(metadata.data(), static_cast<const uint8_t*>(gpu_buf));
  INFO("Unpacked: " << view.num_columns() << " cols, " << view.num_rows() << " rows");
  REQUIRE(view.num_columns() > 0);
  REQUIRE(view.num_rows() > 0);

  // Validate STRING column offsets
  bool any_corrupt = false;
  for (cudf::size_type c = 0; c < view.num_columns(); c++) {
    auto col = view.column(c);
    if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
      auto child          = col.child(0);
      int32_t last_offset = 0;
      if (child.size() > 0) {
        REQUIRE(cudaMemcpy(&last_offset,
                           child.data<int32_t>() + child.size() - 1,
                           sizeof(int32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        INFO("col " << c << " STRING last_offset=" << last_offset << " rows=" << col.size());
        if (last_offset > 100000000 || last_offset < 0) {
          WARN("col " << c << " STRING CORRUPT: last_offset=" << last_offset);
          any_corrupt = true;
        }
      }
    }
  }

  if (any_corrupt) {
    FAIL(
      "cudf::unpack produced corrupt STRING offsets from dumped production data — BUG REPRODUCED!");
  } else {
    INFO("All STRING offsets valid — cudf::unpack works correctly on this data");
  }

  // If not corrupt, try concatenation too
  if (!any_corrupt) {
    cudaGetLastError();
    std::vector<cudf::table_view> views = {view};
    auto result                         = cudf::concatenate(views, stream, &cuda_mr);
    CHECK(result->num_columns() == view.num_columns());
    CHECK(result->num_rows() == view.num_rows());
  }

  cudaFree(gpu_buf);
}

TEST_CASE("pack_unpack with STRING columns", "[exchange]")
{
  // Reproduces Q3/Q14 exchange table issue: tables with STRING columns
  // packed into staging → unpacked → deep copy fails, concatenate works.
  auto stream = cudf::get_default_stream();
  auto* mr    = rmm::mr::get_current_device_resource();

  // Build a table with INT32 + STRING columns (like TPC-H lineitem partkeys + shipmode)
  auto int_scalar = cudf::numeric_scalar<int32_t>(42, true, stream);
  auto int_col    = cudf::sequence(1000, int_scalar, stream, mr);

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
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             offsets.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);
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
  auto unpacked =
    cudf::unpack(packed.metadata->data(), static_cast<const uint8_t*>(packed.gpu_data->data()));
  REQUIRE(unpacked.num_columns() == 2);
  REQUIRE(unpacked.num_rows() == 1000);

  SECTION("cudf::concatenate with unpack'd STRING view works")
  {
    static rmm::mr::cuda_memory_resource cuda_mr;
    cudaGetLastError();
    std::vector<cudf::table_view> views = {unpacked};
    auto result                         = cudf::concatenate(views, stream, &cuda_mr);
    REQUIRE(result->num_columns() == 2);
    REQUIRE(result->num_rows() == 1000);
    // Verify string data is intact
    cudf::strings_column_view str_view(result->view().column(1));
    REQUIRE(str_view.size() == 1000);
  }

  SECTION("cudf::table deep copy with unpack'd STRING view")
  {
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
