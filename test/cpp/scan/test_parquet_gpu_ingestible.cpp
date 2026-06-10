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

// Unit tests for parquet_gpu_ingestible. These cover the pieces that were
// previously exercised by test_parquet_split_provider.cpp before step 10
// deleted the legacy provider:
//   - FLBA-decimal pushdown probe (DECIMAL(25,2) forces
//     FIXED_LEN_BYTE_ARRAY physical type; cudf's stats filter cannot
//     compare against a fixed_point_scalar AST literal, so the ingestible
//     must skip pushdown).
//   - INT64-decimal allow-pushdown (DECIMAL(10,2) fits in INT64; pushdown
//     should remain enabled).
//   - No-filter smoke (has_more_splits / next_split_provider drive a
//     batch end-to-end and emit one scan_operator_input per row-group
//     batch).

#include "catch.hpp"
#include "test_helpers_ioctx.hpp"
#include "test_utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <duckdb.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <helper/logical_type.hpp>
#include <io/io_context.hpp>
#include <io/sirius_datasource.hpp>
#include <io/types.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace sscan = sirius::op::scan;

// Drain an ingestible synchronously (no scheduler), collecting every
// emitted operator_data. The base split_provider::run loop is trivially
// reproducible by hand here — has_more_splits then claim then invoke —
// and avoids dragging in static_thread_pool just to test the claim/work
// behavior in isolation.
std::vector<std::unique_ptr<sirius::op::operator_data>> drain_ingestible(
  sscan::parquet_gpu_ingestible& ingestible)
{
  std::vector<std::unique_ptr<sirius::op::operator_data>> splits;
  while (ingestible.has_more_splits()) {
    auto work = ingestible.next_split_provider();
    if (!work) { continue; }
    auto batch = work();
    for (auto& s : batch) {
      splits.push_back(std::move(s));
    }
  }
  return splits;
}

// Write a parquet file with one INTEGER column and one DECIMAL(p,s) column.
// Returns the path. Caller owns the cleanup.
std::filesystem::path write_decimal_parquet(std::filesystem::path const& dir,
                                            std::string const& filename,
                                            int precision,
                                            int scale,
                                            std::size_t row_count,
                                            std::size_t row_group_size)
{
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  std::string const table = "tmp_" + filename;
  auto result             = con.Query("CREATE TABLE " + table +
                          " AS SELECT (range)::INTEGER AS id, "
                                      "CAST(range * 1.25 AS DECIMAL(" +
                          std::to_string(precision) + "," + std::to_string(scale) +
                          ")) AS amount FROM range(0, " + std::to_string(row_count) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto const path = dir / (filename + ".parquet");
  result          = con.Query("COPY " + table + " TO '" + path.string() +
                     "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " +
                     std::to_string(row_group_size) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  con.Query("DROP TABLE " + table);

  return path;
}

// Build a parquet_ingestible_table_info for the two-column decimal fixture.
std::unique_ptr<sscan::parquet_ingestible_table_info> make_table_info(
  std::filesystem::path const& path,
  int precision,
  int scale,
  duckdb::unique_ptr<duckdb::TableFilterSet> filters)
{
  auto info            = std::make_unique<sscan::parquet_ingestible_table_info>();
  info->returned_types = {
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make_decimal(precision, scale),
  };
  info->resolved_file_paths    = {path.string()};
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids         = {};
  info->names                  = {"id", "amount"};
  info->table_filters          = std::move(filters);
  info->partition_indices      = {};
  info->scan_output_arity      = info->returned_types.size();
  info->approximate_batch_size = std::size_t{1} << 30;
  info->max_file_processed     = 10;
  return info;
}

duckdb::unique_ptr<duckdb::TableFilterSet> make_amount_lt_filter(int precision, int scale)
{
  // amount < 6250.00 — applied to the second column (index 1).
  auto filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  filters->PushFilter(duckdb::ColumnIndex(1),
                      duckdb::make_uniq<duckdb::ConstantFilter>(
                        duckdb::ExpressionType::COMPARE_LESSTHAN,
                        duckdb::Value::DECIMAL(static_cast<int64_t>(625000), precision, scale)));
  return filters;
}

// =====================================================================
// Helpers for the kvikio-fallback datasource-routing guards (port of
// upstream sirius-db/sirius#889 to the gpu_ingestible architecture).
// =====================================================================

class routing_test_io_object final : public sirius::io::sirius_io_object {
 public:
  routing_test_io_object(std::string path, std::size_t size) : _path(std::move(path)), _size(size)
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _path; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _path;
  std::size_t _size;
};

// Spy ioctx — counts make_datasource() calls and returns a cudf datasource
// backed by a real local parquet file. Every other override reports an
// error so the test fails loudly if materialize_table takes an unexpected
// read route through the spy.
class routing_spy_ioctx final : public sirius::io::sirius_ioctx {
 public:
  explicit routing_spy_ioctx(std::filesystem::path local_parquet_path)
    : _local_parquet_path(std::move(local_parquet_path))
  {
  }

  void shutdown() override {}

  std::shared_ptr<sirius::io::sirius_io_object> create_io_object(std::string path) override
  {
    return std::make_shared<routing_test_io_object>(
      std::move(path), std::filesystem::file_size(_local_parquet_path));
  }

  std::unique_ptr<sirius::io::sirius_datasource> make_datasource(
    std::shared_ptr<sirius::io::sirius_io_object> io_object) override
  {
    ++make_datasource_calls;
    return std::make_unique<sirius::io::sirius_datasource>(
      std::static_pointer_cast<sirius::io::sirius_ioctx>(shared_from_this()), std::move(io_object));
  }

  [[nodiscard]] bool supports(std::string_view path) const override
  {
    return path.rfind("s3://", 0) == 0;
  }

  std::size_t host_read_io(sirius::io::sirius_io_object&,
                           std::size_t,
                           std::size_t,
                           std::uint8_t*) override
  {
    throw std::logic_error("routing_spy_ioctx: host_read_io should not be used");
  }

  void host_read_async_io(sirius::io::sirius_io_object&,
                          std::size_t,
                          std::size_t,
                          std::uint8_t*,
                          sirius::io::io_completion_handler handler) override
  {
    handler(0,
            std::make_exception_ptr(
              std::logic_error("routing_spy_ioctx: host_read_async_io should not be used")));
  }

  std::size_t device_read_io(sirius::io::sirius_io_object&,
                             std::size_t,
                             std::size_t,
                             std::uint8_t*,
                             rmm::cuda_stream_view) override
  {
    throw std::logic_error("routing_spy_ioctx: device_read_io should not be used");
  }

  void device_read_async_io(sirius::io::sirius_io_object&,
                            std::size_t,
                            std::size_t,
                            std::uint8_t*,
                            rmm::cuda_stream_view,
                            sirius::io::io_completion_handler handler) override
  {
    handler(0,
            std::make_exception_ptr(
              std::logic_error("routing_spy_ioctx: device_read_async_io should not be used")));
  }

  void host_read_ranges_async_io(sirius::io::sirius_io_object&,
                                 std::vector<cudf::io::text::byte_range_info> const&,
                                 std::span<cudf::host_span<std::byte>>,
                                 sirius::io::io_completion_handler handler) override
  {
    handler(0,
            std::make_exception_ptr(
              std::logic_error("routing_spy_ioctx: host_read_ranges_async_io should not be used")));
  }

  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         std::size_t) const override
  {
    return logical;
  }

  int make_datasource_calls{0};

 private:
  std::filesystem::path _local_parquet_path;
};

// Drain the ingestible once to grab a real parquet_split_info (so the test
// inherits real reader_options, scan_plan, file_metadata, and row_group
// indices), then mint a fresh single-slice parquet_split_info whose
// file_path and io_ctx are swapped for the routing-test fixtures.
std::unique_ptr<sscan::parquet_split_info> make_routing_split_info(
  std::filesystem::path const& local_parquet_path,
  sscan::parquet_gpu_ingestible& ingestible,
  std::string slice_path,
  std::shared_ptr<sirius::io::sirius_ioctx> io_ctx)
{
  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());

  auto* input = dynamic_cast<sscan::scan_operator_input*>(splits.front().get());
  REQUIRE(input != nullptr);
  REQUIRE(input->metadata != nullptr);
  auto const& real_pinfo = dynamic_cast<sscan::parquet_split_info const&>(input->metadata->scan());
  REQUIRE_FALSE(real_pinfo.rg_slices.empty());
  auto const& real_slice = real_pinfo.rg_slices.front();

  auto fake_pinfo                     = std::make_unique<sscan::parquet_split_info>();
  fake_pinfo->reader_options          = real_pinfo.reader_options;
  fake_pinfo->plan                    = real_pinfo.plan;
  fake_pinfo->disable_filter_pushdown = real_pinfo.disable_filter_pushdown;
  fake_pinfo->partition_values        = real_pinfo.partition_values;
  fake_pinfo->needs_assembly          = real_pinfo.needs_assembly;

  std::shared_ptr<cudf::io::datasource> ds;
  if (io_ctx) {
    auto io_object = std::make_shared<routing_test_io_object>(
      slice_path, std::filesystem::file_size(local_parquet_path));
    ds = std::shared_ptr<cudf::io::datasource>(io_ctx->make_datasource(io_object));
  }
  fake_pinfo->rg_slices.emplace_back(real_slice.file_metadata,
                                     std::move(slice_path),
                                     real_slice.row_group_indices,
                                     real_slice.reserved_uncompressed_bytes,
                                     real_slice.reserved_compressed_bytes,
                                     std::move(ds));
  return fake_pinfo;
}

}  // namespace

TEST_CASE("parquet_gpu_ingestible - FLBA decimal disables filter pushdown",
          "[scan][parquet_gpu_ingestible][filter][flba_decimal]")
{
  // DECIMAL(25,2) forces DuckDB's parquet writer to use FIXED_LEN_BYTE_ARRAY
  // physical type (p > 18 exceeds INT64 capacity). cudf's row-group stats
  // filter throws "Invalid type and stats combination" comparing a
  // fixed_point_scalar against FLBA stats. The ingestible's schema probe
  // must detect this and set disable_filter_pushdown on every emitted
  // parquet_split_info. The filter still applies post-decode.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_flba_test";
  auto const path = write_decimal_parquet(dir,
                                          "flba",
                                          /*precision=*/25,
                                          /*scale=*/2,
                                          /*row_count=*/10000,
                                          /*row_group_size=*/5000);

  auto table_info = make_table_info(
    path, /*precision=*/25, /*scale=*/2, make_amount_lt_filter(/*precision=*/25, /*scale=*/2));

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    auto* input = dynamic_cast<sscan::scan_operator_input*>(split.get());
    REQUIRE(input != nullptr);
    auto const* parquet_info =
      dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
    REQUIRE(parquet_info != nullptr);
    INFO("Every split from an FLBA-decimal file must have disable_filter_pushdown set");
    REQUIRE(parquet_info->disable_filter_pushdown);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - INT64 decimal allows filter pushdown",
          "[scan][parquet_gpu_ingestible][filter][flba_decimal]")
{
  // DECIMAL(10,2) fits in INT64 physical type — the probe must NOT disable
  // filter pushdown. Complement of the FLBA test above.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_int64dec_test";
  auto const path = write_decimal_parquet(dir,
                                          "int64dec",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/10000,
                                          /*row_group_size=*/5000);

  auto table_info = make_table_info(
    path, /*precision=*/10, /*scale=*/2, make_amount_lt_filter(/*precision=*/10, /*scale=*/2));

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    auto* input = dynamic_cast<sscan::scan_operator_input*>(split.get());
    REQUIRE(input != nullptr);
    auto const* parquet_info =
      dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
    REQUIRE(parquet_info != nullptr);
    INFO("INT64-decimal files should allow filter pushdown");
    REQUIRE_FALSE(parquet_info->disable_filter_pushdown);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - no-filter scan emits a single scan_operator_input",
          "[scan][parquet_gpu_ingestible]")
{
  // Sanity-check the bare construct → drive → drain path with no filters
  // attached. Confirms that has_more_splits / next_split_provider produce
  // a scan_operator_input whose metadata carries a parquet_split_info
  // with no disable_filter_pushdown flag (no filter expression means the
  // pushdown probe never runs).
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_nofilter_test";
  auto const path = write_decimal_parquet(dir,
                                          "nofilter",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/2000,
                                          /*row_group_size=*/1000);

  auto table_info = make_table_info(path,
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());

  auto* input = dynamic_cast<sscan::scan_operator_input*>(splits.front().get());
  REQUIRE(input != nullptr);
  REQUIRE(input->metadata != nullptr);
  REQUIRE_FALSE(input->metadata->has_filter());  // no filter → no post_filter_and_project info

  auto const* parquet_info =
    dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
  REQUIRE(parquet_info != nullptr);
  REQUIRE_FALSE(parquet_info->disable_filter_pushdown);
  REQUIRE(parquet_info->reader_options != nullptr);
  REQUIRE(parquet_info->plan != nullptr);
  REQUIRE_FALSE(parquet_info->rg_slices.empty());

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - s3 slice with non-null io_ctx routes through spy",
          "[.][scan][parquet_gpu_ingestible][s3][datasource-routing]")
{
  // Guard for sirius-db/sirius#889: when the scan_manager has no ioctx,
  // the materialize path must still honor slice.io_ctx and route
  // the read through it.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_s3_route_test";
  auto const path = write_decimal_parquet(dir,
                                          "s3_route",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/2000,
                                          /*row_group_size=*/1000);

  auto table_info = make_table_info(path,
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto spy = std::make_shared<routing_spy_ioctx>(path);
  auto fake_pinfo =
    make_routing_split_info(path, ingestible, "s3://fake-bucket/s3_route.parquet", spy);

  auto mem_mgr    = initialize_memory_manager(/*n_gpus=*/1);
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  auto filtered = ingestible.materialize_table(*fake_pinfo, *gpu_space, stream.view());

  CHECK(spy->make_datasource_calls == 1);
  REQUIRE(filtered.table != nullptr);

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - local slice with null io_ctx falls through to kvikio",
          "[.][scan][parquet_gpu_ingestible][datasource-routing]")
{
  // Complement of the s3 routing guard above: a slice with io_ctx == nullptr
  // must continue to use cudf's bundled datasource (kvikio) — the local
  // single-GPU fast path the bug fix preserves.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_local_kvikio_test";
  auto const path = write_decimal_parquet(dir,
                                          "local_kvikio",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/2000,
                                          /*row_group_size=*/1000);

  auto table_info = make_table_info(path,
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto fake_pinfo = make_routing_split_info(path, ingestible, path.string(), /*io_ctx=*/nullptr);

  auto mem_mgr    = initialize_memory_manager(/*n_gpus=*/1);
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  auto filtered = ingestible.materialize_table(*fake_pinfo, *gpu_space, stream.view());
  REQUIRE(filtered.table != nullptr);

  std::filesystem::remove_all(dir);
}
