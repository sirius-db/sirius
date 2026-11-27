/*
 * Copyright 2026, Sirius Contributors.
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

// Component tests for equality deletes: the anti-join mask kernel, the group build and the filter
// that probes it. The SQL fixtures in test_gpu_execution_multi_format.cpp cover the route end to
// end; these use inputs no table produces cheaply -- an inverted mask, keys paired in the wrong
// order, a key type the join would compare as raw bytes.

#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/iceberg_delete_filter.hpp"
#include "op/scan/iceberg_equality_delete_mask.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <utils/local_ioctx.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace sirius::op::scan;

namespace {

::cuda::stream_ref test_stream() { return cudf::get_default_stream(); }

rmm::device_async_resource_ref test_mr() { return cudf::get_current_device_resource_ref(); }

/// An INT32 column from host values; `nulls` marks which entries are null.
std::unique_ptr<cudf::column> int32_column(std::vector<int32_t> const& values,
                                           std::vector<bool> const& nulls = {})
{
  auto stream = test_stream();
  auto col    = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(values.size()),
                                           cudf::mask_state::UNALLOCATED,
                                           stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().data<int32_t>(),
                                values.data(),
                                values.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  stream.sync();
  if (!nulls.empty()) {
    REQUIRE(nulls.size() == values.size());
    // Built on the host and copied whole: sized with bitmask_allocation_size_bytes because a
    // bit-tight buffer crashes cudf's deep copies.
    auto const bytes =
      cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(values.size()));
    std::vector<cudf::bitmask_type> host_mask(bytes / sizeof(cudf::bitmask_type), ~0u);
    cudf::size_type null_count = 0;
    for (std::size_t i = 0; i < nulls.size(); ++i) {
      if (!nulls[i]) { continue; }
      host_mask[i / 32] &= ~(cudf::bitmask_type{1} << (i % 32));
      ++null_count;
    }
    if (null_count > 0) {
      auto mask = cudf::create_null_mask(col->size(), cudf::mask_state::UNINITIALIZED, stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        mask.data(), host_mask.data(), bytes, cudaMemcpyHostToDevice, stream.get()));
      stream.sync();
      col->set_null_mask(std::move(mask), null_count);
    }
  }
  return col;
}

std::vector<int32_t> to_host(cudf::column_view const& col)
{
  std::vector<int32_t> out(static_cast<std::size_t>(col.size()));
  auto stream = test_stream();
  CUDF_CUDA_TRY(cudaMemcpyAsync(out.data(),
                                col.data<int32_t>(),
                                out.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();
  return out;
}

std::vector<uint8_t> mask_to_host(cudf::column_view const& col)
{
  std::vector<uint8_t> out(static_cast<std::size_t>(col.size()));
  auto stream = test_stream();
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    out.data(), col.data<uint8_t>(), out.size(), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  return out;
}

rmm::device_uvector<cudf::size_type> device_indices(std::vector<cudf::size_type> const& host)
{
  auto stream = test_stream();
  rmm::device_uvector<cudf::size_type> dev(host.size(), stream, test_mr());
  CUDF_CUDA_TRY(cudaMemcpyAsync(dev.data(),
                                host.data(),
                                host.size() * sizeof(cudf::size_type),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  stream.sync();
  return dev;
}

/// A group built with the test's stream and allocator.
EqualityDeleteGroup build_group(std::vector<int32_t> key_field_ids,
                                std::vector<std::size_t> key_primary_indices,
                                std::vector<cudf::table_view> const& views)
{
  return build_equality_group(
    std::move(key_field_ids), std::move(key_primary_indices), views, test_stream(), test_mr());
}

/// Adds @p group to the replica of the current device.
void add_group(IcebergDeleteData& data, EqualityDeleteGroup group)
{
  if (data.equality_replicas.empty())
    data.equality_replicas.push_back({rmm::get_current_cuda_device(), {}});
  data.equality_replicas.back().groups.push_back(std::move(group));
}

/// Routes cuDF's default device allocations through @p stats for a scope.
struct scoped_device_resource {
  explicit scoped_device_resource(rmm::mr::statistics_resource_adaptor& stats)
    : previous(cudf::set_current_device_resource(rmm::device_async_resource_ref{stats}))
  {
  }
  ~scoped_device_resource() { cudf::set_current_device_resource(std::move(previous)); }
  cuda::mr::any_resource<cuda::mr::device_accessible> previous;
};

}  // namespace

//===----------------------------------------------------------------------===//
// make_anti_join_mask
//===----------------------------------------------------------------------===//

TEST_CASE("iceberg equality delete - anti-join mask keeps unmatched rows", "[iceberg][scan]")
{
  // left_join() writes JoinNoMatch for a probe row that found no delete row. That row SURVIVES.
  // Any other value means the row matched a delete and must go. Inverting this deletes exactly
  // the rows the table keeps and keeps the ones it deletes, with no error anywhere.
  auto const indices = device_indices({cudf::JoinNoMatch, 0, cudf::JoinNoMatch, 7});

  auto mask = make_anti_join_mask(indices, 4, test_stream(), test_mr());
  REQUIRE(mask->size() == 4);
  REQUIRE(mask->type().id() == cudf::type_id::BOOL8);
  REQUIRE(mask_to_host(mask->view()) == std::vector<uint8_t>{1, 0, 1, 0});
}

TEST_CASE("iceberg equality delete - anti-join mask honours n_rows", "[iceberg][scan]")
{
  // The probe result can be longer than the batch; only the first n_rows entries are the batch's.
  auto const indices = device_indices({cudf::JoinNoMatch, 3, cudf::JoinNoMatch});

  auto mask = make_anti_join_mask(indices, 2, test_stream(), test_mr());
  REQUIRE(mask->size() == 2);
  REQUIRE(mask_to_host(mask->view()) == std::vector<uint8_t>{1, 0});
}

//===----------------------------------------------------------------------===//
// build_equality_group + equality_delete_filter
//===----------------------------------------------------------------------===//

TEST_CASE("iceberg equality delete - group deduplicates across delete files", "[iceberg][scan]")
{
  std::vector<std::unique_ptr<cudf::column>> a;
  a.push_back(int32_column({1, 2}));
  cudf::table file_a{std::move(a)};

  std::vector<std::unique_ptr<cudf::column>> b;
  b.push_back(int32_column({2, 3}));
  cudf::table file_b{std::move(b)};

  auto group = build_group({1}, {0}, {file_a.view(), file_b.view()});

  // {1,2} ∪ {2,3} = three distinct keys, not four rows.
  REQUIRE(group.delete_table->num_rows() == 3);
  REQUIRE(group.key_field_ids == std::vector<int32_t>{1});
  REQUIRE(group.key_primary_indices == std::vector<std::size_t>{0});
}

TEST_CASE("iceberg equality delete - filter removes exactly the matching rows", "[iceberg][scan]")
{
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({2, 4}));
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1}, {0}, {delete_file.view()}));

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 2, 3, 4, 5}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, rmm::get_current_cuda_device(), 0, {0}};
  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 5}};

  auto out = filter.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr());
  REQUIRE(out->num_rows() == 3);
  REQUIRE(to_host(out->get_column(0).view()) == std::vector<int32_t>{1, 3, 5});
}

TEST_CASE("iceberg equality delete - a NULL key deletes the NULL rows", "[iceberg][scan]")
{
  // Iceberg equality deletes match with null-equal semantics: a delete row whose key is NULL
  // removes data rows whose key is NULL. cudf's default is null_equality::UNEQUAL, so this is
  // the one place the group build's explicit choice shows up.
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({0}, {true}));  // single NULL key
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1}, {0}, {delete_file.view()}));

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 0, 3}, {false, true, false}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, rmm::get_current_cuda_device(), 0, {0}};
  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 3}};

  auto out = filter.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr());
  REQUIRE(out->num_rows() == 2);
  REQUIRE(to_host(out->get_column(0).view()) == std::vector<int32_t>{1, 3});
}

TEST_CASE("iceberg equality delete - missing key column throws", "[iceberg][scan]")
{
  // The key columns are force-projected into the scan. If that widening never happened, the
  // batch is short and returning it unchanged would hand back rows the table deleted.
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({1}));
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1}, {0}, {delete_file.view()}));

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 2}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{
    data, rmm::get_current_cuda_device(), 0, {3}};  // index past the end of the batch
  std::vector<batch_row_run> runs{{"data-0.parquet", 0, 0, 2}};

  REQUIRE_THROWS_AS(filter.apply(std::move(batch), batch_layout{runs}, test_stream(), test_mr()),
                    std::invalid_argument);
}

TEST_CASE("iceberg equality delete - a composite key matches on every key column",
          "[iceberg][scan]")
{
  // Delete (1, 20) only. A filter that matched on either column alone would also remove
  // (1, 10) or (2, 20).
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({1}));
  del.push_back(int32_column({20}));
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1, 2}, {0, 1}, {delete_file.view()}));

  // The batch holds the keys at positions 2 and 0, with an unrelated column between them: the
  // filter must pair them with the group's key ORDER, not with batch order.
  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({10, 20, 20}));  // second key
  batch_cols.push_back(int32_column({7, 8, 9}));     // payload
  batch_cols.push_back(int32_column({1, 1, 2}));     // first key
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, rmm::get_current_cuda_device(), 0, {2, 0}};
  auto out = filter.apply(std::move(batch), {}, test_stream(), test_mr());
  REQUIRE(out->num_rows() == 2);
  REQUIRE(to_host(out->get_column(1).view()) == std::vector<int32_t>{7, 9});
}

TEST_CASE("iceberg equality delete - a composite key with a NULL member matches NULL",
          "[iceberg][scan]")
{
  // `id = 4 AND category IS NULL` from the spec's example: (4, NULL) deletes (4, NULL) and
  // leaves (4, 5) and (NULL, NULL) alone.
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({4}));
  del.push_back(int32_column({0}, {true}));
  cudf::table delete_file{std::move(del)};

  auto data = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1, 2}, {0, 1}, {delete_file.view()}));

  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({4, 4, 0}, {false, false, true}));
  batch_cols.push_back(int32_column({0, 5, 0}, {true, false, true}));
  auto batch = std::make_unique<cudf::table>(std::move(batch_cols));

  equality_delete_filter filter{data, rmm::get_current_cuda_device(), 0, {0, 1}};
  auto out = filter.apply(std::move(batch), {}, test_stream(), test_mr());
  REQUIRE(out->num_rows() == 2);
  REQUIRE(out->get_column(0).view().null_count() == 1);
  REQUIRE(out->get_column(1).view().null_count() == 1);
}

TEST_CASE("iceberg equality delete - key count and key type mismatches throw", "[iceberg][scan]")
{
  std::vector<std::unique_ptr<cudf::column>> del;
  del.push_back(int32_column({1}));
  cudf::table delete_file{std::move(del)};
  auto data = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1}, {0}, {delete_file.view()}));

  auto make_batch = [] {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(int32_column({1, 2}));
    cols.push_back(cudf::make_fixed_width_column(
      cudf::data_type{cudf::type_id::INT64}, 2, cudf::mask_state::UNALLOCATED, test_stream()));
    return std::make_unique<cudf::table>(std::move(cols));
  };

  // Two batch columns offered for a one-key group.
  equality_delete_filter too_many{data, rmm::get_current_cuda_device(), 0, {0, 1}};
  REQUIRE_THROWS_AS(too_many.apply(make_batch(), {}, test_stream(), test_mr()),
                    std::invalid_argument);
  // An INT64 batch column against INT32 delete keys: the join would compare raw bytes.
  equality_delete_filter wrong_type{data, rmm::get_current_cuda_device(), 0, {1}};
  REQUIRE_THROWS_AS(wrong_type.apply(make_batch(), {}, test_stream(), test_mr()),
                    std::invalid_argument);
  // A batch on a device no tables were built on. Another device's tables need not be
  // addressable from it, and passing the batch through would return the deleted row.
  rmm::cuda_device_id const other{rmm::get_current_cuda_device().value() + 1};
  equality_delete_filter other_device{data, other, 0, {0}};
  REQUIRE_THROWS_WITH(
    other_device.apply(make_batch(), {}, test_stream(), test_mr()),
    Catch::Matchers::ContainsSubstring("no equality-delete tables were built on GPU"));
  // A group needs keys and matching tables.
  REQUIRE_THROWS_AS(build_group({1, 2}, {0, 1}, {delete_file.view()}), std::invalid_argument);
}

TEST_CASE("iceberg equality delete - a group retains a bounded number of bytes per key row",
          "[iceberg][scan]")
{
  // The statement ceilings in iceberg_metadata_reader.hpp assume about 24 retained device bytes
  // per INT64 key row (keys plus hash table). If the build starts retaining more, the ceilings
  // admit more device memory than they say.
  constexpr cudf::size_type rows = 1 << 20;
  auto init                      = cudf::make_fixed_width_scalar<int64_t>(1000000007);
  auto step                      = cudf::make_fixed_width_scalar<int64_t>(7919);
  auto keys                      = cudf::sequence(rows, *init, *step);

  // The group is charged to the allocator it is handed. The ambient one belongs to whichever
  // device the calling thread has current, which on a planning thread need not be the scan's.
  cuda::mr::any_resource<cuda::mr::device_accessible> const upstream{
    cudf::get_current_device_resource_ref()};
  rmm::mr::statistics_resource_adaptor handed{upstream};
  rmm::mr::statistics_resource_adaptor ambient{upstream};
  scoped_device_resource restore{ambient};
  auto group          = build_equality_group({1},
                                             {0},
                                             {cudf::table_view{{keys->view()}}},
                                    test_stream(),
                                    rmm::device_async_resource_ref{handed});
  auto const retained = handed.get_bytes_counter().value;
  REQUIRE(group.delete_table->num_rows() == rows);
  CHECK(retained >= int64_t{8} * rows);
  CHECK(retained <= int64_t{32} * rows);
  // cuDF keeps the join's column descriptors, tens of bytes, with the ambient resource.
  CHECK(ambient.get_bytes_counter().value < 1024);
}

TEST_CASE("iceberg equality delete - probing a batch stays within the row-group estimate",
          "[iceberg][scan]")
{
  // attach_delete_set budgets one copy of the batch plus 9 B/row for a probe. No row matches
  // here, so the compacted copy is the whole batch: the largest a probe allocates.
  constexpr cudf::size_type rows = 1 << 20;
  auto zero                      = cudf::make_fixed_width_scalar<int64_t>(0);
  auto one                       = cudf::make_fixed_width_scalar<int64_t>(1);
  auto absent                    = cudf::make_fixed_width_scalar<int64_t>(-1);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::sequence(rows, *zero, *one));  // key
  columns.push_back(cudf::sequence(rows, *zero, *one));  // payload
  auto batch             = std::make_unique<cudf::table>(std::move(columns));
  auto const batch_bytes = std::size_t{2} * rows * sizeof(int64_t);

  auto deleted = cudf::sequence(1, *absent, *one);
  auto data    = std::make_shared<IcebergDeleteData>();
  add_group(*data, build_group({1}, {0}, {cudf::table_view{{deleted->view()}}}));

  rmm::mr::statistics_resource_adaptor stats{
    cuda::mr::any_resource<cuda::mr::device_accessible>{cudf::get_current_device_resource_ref()}};
  scoped_device_resource restore{stats};
  equality_delete_filter filter{data, rmm::get_current_cuda_device(), 0, {0}};
  auto out =
    filter.apply(std::move(batch), {}, test_stream(), rmm::device_async_resource_ref{stats});
  REQUIRE(out->num_rows() == rows);
  auto const peak = static_cast<std::size_t>(stats.get_bytes_counter().peak);
  CAPTURE(peak, batch_bytes);
  CHECK(peak <= batch_bytes + std::size_t{9} * rows);
}

TEST_CASE("iceberg equality delete - each ceiling refuses the charge", "[iceberg][scan]")
{
  using files  = std::vector<std::pair<int64_t, int64_t>>;  // (record_count, file_size_in_bytes)
  auto plan_of = [](files const& entries) {
    iceberg_equality_plan plan;
    for (auto [rows, bytes] : entries)
      plan.files.push_back({"delete.parquet", {1}, rows, bytes});
    return plan;
  };
  auto fits = [&](files const& entries, int64_t row_limit = kMaxEqualityDeleteRowsPerStatement) {
    equality_delete_charge total;
    return charge_equality_deletes(plan_of(entries), total, row_limit);
  };
  CHECK(fits({{kMaxEqualityDeleteRowsPerFile, 1}}));
  CHECK_FALSE(fits({{kMaxEqualityDeleteRowsPerFile + 1, 1}}));
  CHECK(fits({{2, 1}, {1, 1}}, 3));
  CHECK_FALSE(fits({{2, 1}, {2, 1}}, 3));
  CHECK(fits({{1, kMaxEqualityDeleteBytesPerStatement}}));
  CHECK_FALSE(fits({{1, kMaxEqualityDeleteBytesPerStatement}, {1, 1}}));
  // A count the manifest did not supply cannot be charged.
  CHECK_FALSE(fits({{-1, 1}}));
  CHECK_FALSE(fits({{1, -1}}));

  // The total carries across the scans of one statement.
  equality_delete_charge total;
  CHECK(charge_equality_deletes(plan_of({{2, 1}}), total, 3));
  CHECK_FALSE(charge_equality_deletes(plan_of({{2, 1}}), total, 3));
}

TEST_CASE("iceberg equality delete - a delete file must match its manifest and key types",
          "[iceberg][scan]")
{
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir directory("equality_delete_keys");
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto written = con.Query("COPY (SELECT 1::INTEGER AS id, 'a' AS name) TO " +
                           directory.file_literal("delete.parquet") +
                           " (FORMAT PARQUET, FIELD_IDS {id: 1, name: 2})");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());

  cuda::mr::any_resource<cuda::mr::device_accessible> const upstream{
    cudf::get_current_device_resource_ref()};
  rmm::mr::statistics_resource_adaptor handed{upstream};
  auto const path  = directory.file("delete.parquet");
  auto const bytes = static_cast<int64_t>(std::filesystem::file_size(path));
  auto load_from   = [&](std::string const& file,
                       int32_t field_id,
                       duckdb::LogicalTypeId bound,
                       int64_t rows,
                       int64_t size,
                       std::size_t devices = 1) {
    iceberg_delete_discovery discovery;
    discovery.equality_delete_entries.emplace_back();
    auto& plan = discovery.equality_plan.emplace();
    plan.files.push_back({file, {field_id}, rows, size});
    plan.by_data_file.emplace("data.parquet", std::vector<uint32_t>{0});
    plan.keys.emplace(field_id, iceberg_equality_plan::key{0, bound});
    return load_delete_payload(
      *con.context,
      "table",
      sirius::test::make_test_local_ioctx(),
      discovery,
      std::vector<equality_build_resources>(
        devices,
        {rmm::get_current_cuda_device(), test_stream(), rmm::device_async_resource_ref{handed}}));
  };
  auto load = [&](int32_t field_id, duckdb::LogicalTypeId bound, int64_t rows, int64_t size) {
    return load_from(path, field_id, bound, rows, size);
  };
  CHECK_THROWS_WITH(load(1, duckdb::LogicalTypeId::INTEGER, 1, bytes - 1),
                    Catch::Matchers::ContainsSubstring("size differs from its manifest entry"));
  CHECK_THROWS_WITH(
    load(1, duckdb::LogicalTypeId::INTEGER, 0, bytes),
    Catch::Matchers::ContainsSubstring("row count differs from its manifest entry"));
  // The manifest lists a key the delete file does not store.
  CHECK_THROWS_WITH(load(3, duckdb::LogicalTypeId::INTEGER, 1, bytes),
                    Catch::Matchers::ContainsSubstring("has no top-level column with field id 3"));
  CHECK(handed.get_allocations_counter().total == 0);
  // A string stored for a BIGINT key: the join would compare unrelated bytes. Nothing is built
  // from it, so the allocations made so far are the file reader's own.
  CHECK_THROWS_WITH(
    load(2, duckdb::LogicalTypeId::BIGINT, 1, bytes),
    Catch::Matchers::ContainsSubstring("which cannot be compared with the bound BIGINT column"));
  CHECK(handed.get_allocations_counter().total > 0);
  // Equality deletes without a stream and an allocator to build them with are refused.
  iceberg_delete_discovery unbuilt;
  unbuilt.equality_delete_entries.emplace_back();
  unbuilt.equality_plan.emplace();
  CHECK_THROWS_WITH(
    load_delete_payload(*con.context, "table", sirius::test::make_test_local_ioctx(), unbuilt),
    Catch::Matchers::ContainsSubstring("no stream and allocator"));
  // So is a delete file under s3:// with no context to read it through. Going on as if it
  // deleted nothing would return the rows it removes.
  CHECK_THROWS_WITH(
    load_from("s3://bucket/delete.parquet", 1, duckdb::LogicalTypeId::INTEGER, 1, bytes),
    Catch::Matchers::ContainsSubstring("no S3 backend is available to read equality-delete file"));
  // Deletes that apply to no data file build nothing on any device: the payload is empty, and a
  // scan is not refused for groups that no data file maps to.
  CHECK(load_delete_payload(
          *con.context,
          "table",
          sirius::test::make_test_local_ioctx(),
          unbuilt,
          {equality_build_resources{
            rmm::get_current_cuda_device(), test_stream(), rmm::device_async_resource_ref{handed}}})
          ->empty());
  // INT32 stored for a BIGINT key is the one accepted difference.
  auto const data    = load(1, duckdb::LogicalTypeId::BIGINT, 1, bytes);
  auto const& groups = data->equality_groups_on(rmm::get_current_cuda_device());
  REQUIRE(groups.size() == 1);
  CHECK(groups[0].delete_table->view().column(0).type().id() == cudf::type_id::INT64);
  // Every device after the first indexes a copy of the first one's keys. The three entries name
  // this device, since the machine may have no other: what is covered is that each of them, not
  // only the first to be copied to, holds the key.
  auto const thrice = load_from(path, 1, duckdb::LogicalTypeId::BIGINT, 1, bytes, 3);
  REQUIRE(thrice->equality_replicas.size() == 3);
  for (auto const& replica : thrice->equality_replicas) {
    REQUIRE(replica.groups.size() == 1);
    auto const keys = replica.groups[0].delete_table->view().column(0);
    REQUIRE(keys.type().id() == cudf::type_id::INT64);
    REQUIRE(keys.size() == 1);
    int64_t key = 0;
    CUDF_CUDA_TRY(cudaMemcpy(&key, keys.data<int64_t>(), sizeof(key), cudaMemcpyDeviceToHost));
    CHECK(key == 1);
  }
}

TEST_CASE("iceberg equality delete - a second device probes keys held in its own memory",
          "[iceberg][scan][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) SKIP("Requires two visible GPUs");
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir directory("equality_delete_two_devices");
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto written =
    con.Query("COPY (SELECT 2::INTEGER AS id) TO " + directory.file_literal("delete.parquet") +
              " (FORMAT PARQUET, FIELD_IDS {id: 1})");
  REQUIRE(written);
  REQUIRE_FALSE(written->HasError());
  auto const path = directory.file("delete.parquet");

  // What a misplaced allocation looks like depends on the spaces. With pools, memory from the
  // other device's allocator is on the other GPU. Without, every allocation is a cudaMalloc, so
  // memory allocated with the other device current is, and no device can read another's.
  bool const pooled = GENERATE(true, false);
  sirius::test::operator_utils::memory_manager_options options;
  if (!pooled) {
    options.gpu_resource_factory = [](int device, std::size_t) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
      return ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
        rmm::mr::cuda_memory_resource{}};
    };
  }
  auto manager = sirius::test::operator_utils::initialize_memory_manager(2, std::move(options));
  // The first device's space is left out, so its keys reach the second through HOST staging
  // whether or not the second could read them where they are.
  std::vector<equality_build_resources> resources;
  for (auto const device : {0, 1}) {
    auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, device);
    REQUIRE(space);
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    resources.push_back({rmm::cuda_device_id{device},
                         space->acquire_stream(),
                         space->get_default_allocator(),
                         {},
                         device == 0 ? nullptr : space});
  }
  auto const* staging =
    sirius::memory::sirius_memory_reservation_manager::host_staging_space(*resources[1].space);
  REQUIRE(staging);
  auto const* staged =
    staging->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  REQUIRE(staged);
  REQUIRE(staged->get_peak_total_allocated_bytes() == 0);

  iceberg_delete_discovery discovery;
  discovery.equality_delete_entries.emplace_back();
  auto& plan = discovery.equality_plan.emplace();
  plan.files.push_back({path, {1}, 1, static_cast<int64_t>(std::filesystem::file_size(path))});
  plan.by_data_file.emplace("data.parquet", std::vector<uint32_t>{0});
  plan.keys.emplace(1, iceberg_equality_plan::key{0, duckdb::LogicalTypeId::INTEGER});
  auto const data = load_delete_payload(
    *con.context, "table", sirius::test::make_test_local_ioctx(), discovery, resources);
  REQUIRE(data->equality_replicas.size() == 2);
  CHECK(staged->get_peak_total_allocated_bytes() > 0);

  rmm::cuda_device_id const second{1};
  cudaPointerAttributes keys{};
  REQUIRE(cudaPointerGetAttributes(
            &keys, data->equality_groups_on(second).at(0).delete_table->view().column(0).head()) ==
          cudaSuccess);
  CHECK(keys.device == second.value());

  // With the first device's tables gone, the second still removes the matching row and no other.
  std::array<int, 1> const kept{second.value()};
  data->release_equality_replicas_outside(kept);
  REQUIRE(data->equality_replicas.size() == 1);
  rmm::cuda_set_device_raii guard{second};
  std::vector<std::unique_ptr<cudf::column>> batch_cols;
  batch_cols.push_back(int32_column({1, 2}));
  equality_delete_filter filter{data, second, 0, {0}};
  auto out = filter.apply(
    std::make_unique<cudf::table>(std::move(batch_cols)), {}, resources[1].stream, resources[1].mr);
  resources[1].stream.sync();
  REQUIRE(to_host(out->get_column(0).view()) == std::vector<int32_t>{1});
}

TEST_CASE("Iceberg per-file positional filters preserve multi-file row positions",
          "[scan_preparation][iceberg][positional_filter]")
{
  auto const legacy = GENERATE(false, true);
  CAPTURE(legacy);
  auto payload                             = std::make_shared<IcebergDeleteData>();
  payload->positional_deletes["a.parquet"] = {1, 1, 6};
  payload->positional_deletes["b.parquet"] = {2};
  std::unique_ptr<positional_delete_filter> filter;
  if (legacy) {
    filter = std::make_unique<positional_delete_filter>(payload);
  } else {
    iceberg_delete_sets sets;
    for (auto const& [path, positions] : payload->positional_deletes) {
      auto owner = std::shared_ptr<std::vector<int64_t> const>(payload, &positions);
      sets.emplace(path, std::make_shared<iceberg_delete_set const>(path, std::move(owner)));
    }
    sets.emplace("c.parquet", std::make_shared<iceberg_delete_set const>("c.parquet"));
    filter = std::make_unique<positional_delete_filter>(std::move(sets));
  }
  std::weak_ptr<IcebergDeleteData const> owner = payload;
  payload.reset();
  CHECK_FALSE(owner.expired());
  std::vector<batch_row_run> layout{
    {"a.parquet", 5, 0, 4}, {"b.parquet", 0, 4, 3}, {"c.parquet", 0, 7, 3}};
  CHECK(filter->affects(layout));
  CHECK_FALSE(filter->affects(std::vector<batch_row_run>{{"c.parquet", 0, 0, 3}}));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int32_column({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  auto result = filter->apply(
    std::make_unique<cudf::table>(std::move(columns)), layout, test_stream(), test_mr());
  CHECK(to_host(result->view().column(0)) == std::vector<int32_t>{0, 2, 3, 4, 5, 7, 8, 9});
  filter.reset();
  CHECK(owner.expired());
}
