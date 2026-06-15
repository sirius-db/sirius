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

// Unit tests for parquet_gpu_ingestible. They cover:
//   - FLBA-decimal pushdown probe: DECIMAL(25,2) is stored as FIXED_LEN_BYTE_ARRAY,
//     whose stats cudf cannot compare to a numeric literal, so pushdown is skipped.
//   - INT64-decimal allow-pushdown: DECIMAL(10,2) fits in INT64, so pushdown stays on.
//   - Coalescing: a file's row groups merge back into one slice, and a small byte cap
//     splits them across multiple batches stamped round-robin across GPUs.
//   - Datasource routing: local vs S3 slices, the cudf/kvikio fallback, and chunk prewarm.

#include "catch.hpp"
#include "io/prefetching_cache.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"
#include "test_helpers_ioctx.hpp"
#include "test_utils.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
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
#include <scan_manager/round_robin_strategy.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>
#include <scan_manager/split_provider.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace sscan = sirius::op::scan;
using namespace std::chrono_literals;

constexpr std::uint32_t cache_max_slabs = 3;
constexpr std::size_t cache_block_size  = 4096;

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

bool truthy_env(std::string_view name)
{
  auto value = env_or(name);
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

struct s3_test_env {
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
};

std::optional<s3_test_env> read_s3_test_env()
{
  auto endpoint   = env_or("SIRIUS_TEST_S3_ENDPOINT");
  auto access_key = env_or("SIRIUS_TEST_S3_ACCESS_KEY");
  auto secret_key = env_or("SIRIUS_TEST_S3_SECRET_KEY");
  auto bucket     = env_or("SIRIUS_TEST_S3_BUCKET");

  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    return std::nullopt;
  }

  return s3_test_env{std::move(endpoint),
                     env_or("SIRIUS_TEST_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket)};
}

bool skip_if_no_s3_env(std::optional<s3_test_env> const& env)
{
  if (env) { return false; }
  if (truthy_env("SIRIUS_TEST_S3_STRICT")) {
    FAIL("SIRIUS_TEST_S3_* environment is required in strict mode");
  }
  SUCCEED("SIRIUS_TEST_S3_* not set; skipping live S3 parquet_gpu_ingestible test");
  return true;
}

std::string s3_uri(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string{bucket} + "/" + std::string{key};
}

std::size_t cache_capacity_bytes(std::size_t block_size, std::uint32_t max_slabs)
{
  return block_size * static_cast<std::size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB) *
         static_cast<std::size_t>(max_slabs);
}

struct host_cache_memory {
  host_cache_memory()
    : upstream(0, true),
      host_mr(0,
              upstream,
              cache_capacity_bytes(cache_block_size, cache_max_slabs),
              cache_capacity_bytes(cache_block_size, cache_max_slabs),
              cache_block_size,
              static_cast<std::size_t>(sirius::io::buffer_pool::CHUNKS_PER_SLAB),
              1)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource host_mr;
};

sirius::io::s3::s3_ioctx_config make_s3_config(s3_test_env const& env)
{
  sirius::io::s3::static_credentials creds;
  creds.access_key_id     = env.access_key;
  creds.secret_access_key = env.secret_key;

  sirius::io::s3::s3_ioctx_config cfg{};
  cfg.creds = std::make_shared<sirius::io::s3::sirius_sigv4_presigned_authorizer>(
    std::move(creds), env.region, env.endpoint, 30min);
  cfg.max_connections    = 4;
  cfg.request_timeout_s  = 20;
  cfg.max_retry_attempts = 3;
  cfg.retry_backoff_base = 10ms;
  cfg.retry_jitter       = 0ms;
  cfg.honor_retry_after  = false;
  return cfg;
}

// Scheduler that runs each enqueued thunk synchronously, so split_provider::run()
// produces every carrier and closes the connector before returning.
struct inline_scheduler {
  template <typename F>
  void enqueue(F&& f)
  {
    f();
  }
};

// Drive the full producer -> connector -> coalescing-consumer path and collect
// every emitted scan_operator_input. The producer now emits format-blind
// coalescing_carriers; the sealed consume_next_input packs their units and calls
// make_batch, so the scan_operator_inputs only exist on the consumer side. Run
// the producer inline (no thread pool) just to exercise claim/work behavior.
std::vector<std::unique_ptr<sirius::op::operator_data>> drain_ingestible(
  sscan::parquet_gpu_ingestible& ingestible)
{
  sirius::scan_manager::split_connector connector;
  sirius::scan_manager::split_provider provider{ingestible};
  inline_scheduler sched;
  provider.run(sched, connector);  // pushes all carriers; closes the connector on return

  std::vector<std::unique_ptr<sirius::op::operator_data>> splits;
  for (;;) {
    auto batch = ingestible.consume_next_input(connector);
    if (!batch) { break; }
    splits.push_back(std::move(batch));
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
// Helpers for the datasource-routing guards: confirm materialize_table honors a
// slice's own datasource and falls back to cudf/kvikio for unclaimed local files.
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

// Spy ioctx — counts datasource/read calls and returns a cudf datasource whose
// public path looks like s3:// while its bytes come from a real local parquet
// file. This lets the routing guard materialize through the Sirius datasource
// without depending on MinIO.
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
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override
  {
    ++host_read_calls;
    return read_from_local(offset, size, dst);
  }

  void host_read_async_io(sirius::io::sirius_io_object& obj,
                          std::size_t offset,
                          std::size_t size,
                          std::uint8_t* dst,
                          sirius::io::io_completion_handler handler) override
  {
    try {
      handler(host_read_io(obj, offset, size, dst), nullptr);
    } catch (...) {
      handler(0, std::current_exception());
    }
  }

  std::size_t device_read_io(sirius::io::sirius_io_object&,
                             std::size_t offset,
                             std::size_t size,
                             std::uint8_t* dst,
                             rmm::cuda_stream_view stream) override
  {
    ++device_read_calls;
    std::vector<std::uint8_t> host(size);
    auto const got = read_from_local(offset, size, host.data());
    auto rc        = cudaMemcpyAsync(dst, host.data(), got, cudaMemcpyHostToDevice, stream.value());
    if (rc != cudaSuccess) {
      throw std::runtime_error(std::string{"routing_spy_ioctx: cudaMemcpyAsync failed: "} +
                               cudaGetErrorString(rc));
    }
    rc = cudaStreamSynchronize(stream.value());
    if (rc != cudaSuccess) {
      throw std::runtime_error(std::string{"routing_spy_ioctx: cudaStreamSynchronize failed: "} +
                               cudaGetErrorString(rc));
    }
    return got;
  }

  void device_read_async_io(sirius::io::sirius_io_object& obj,
                            std::size_t offset,
                            std::size_t size,
                            std::uint8_t* dst,
                            rmm::cuda_stream_view stream,
                            sirius::io::io_completion_handler handler) override
  {
    try {
      handler(device_read_io(obj, offset, size, dst, stream), nullptr);
    } catch (...) {
      handler(0, std::current_exception());
    }
  }

  void host_read_ranges_async_io(sirius::io::sirius_io_object&,
                                 std::vector<cudf::io::text::byte_range_info> const& ranges,
                                 std::span<cudf::host_span<std::byte>> dst,
                                 sirius::io::io_completion_handler handler) override
  {
    try {
      if (ranges.size() != dst.size()) {
        throw std::invalid_argument("routing_spy_ioctx: ranges/dst size mismatch");
      }
      ++range_read_calls;
      std::size_t total = 0;
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        auto const offset = static_cast<std::size_t>(ranges[i].offset());
        auto const size   = static_cast<std::size_t>(ranges[i].size());
        if (dst[i].size() < size) {
          throw std::runtime_error("routing_spy_ioctx: dst span too small");
        }
        total += read_from_local(offset, size, reinterpret_cast<std::uint8_t*>(dst[i].data()));
      }
      handler(total, nullptr);
    } catch (...) {
      handler(0, std::current_exception());
    }
  }

  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         std::size_t) const override
  {
    return logical;
  }

  int make_datasource_calls{0};
  int host_read_calls{0};
  int device_read_calls{0};
  int range_read_calls{0};

 private:
  std::size_t read_from_local(std::size_t offset, std::size_t size, std::uint8_t* dst) const
  {
    auto const file_size = std::filesystem::file_size(_local_parquet_path);
    if (offset >= file_size) { return 0; }
    auto const clipped = std::min(size, file_size - offset);
    std::ifstream in(_local_parquet_path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("routing_spy_ioctx: failed to open local backing parquet");
    }
    in.seekg(static_cast<std::streamoff>(offset));
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(clipped));
    if (in.gcount() != static_cast<std::streamsize>(clipped)) {
      throw std::runtime_error("routing_spy_ioctx: short read from local backing parquet");
    }
    return clipped;
  }

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

std::vector<sscan::parquet_split_info const*> split_infos(
  std::vector<std::unique_ptr<sirius::op::operator_data>> const& splits)
{
  std::vector<sscan::parquet_split_info const*> out;
  out.reserve(splits.size());
  for (auto const& split : splits) {
    auto* input = dynamic_cast<sscan::scan_operator_input*>(split.get());
    REQUIRE(input != nullptr);
    REQUIRE(input->metadata != nullptr);
    auto const* parquet_info =
      dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
    REQUIRE(parquet_info != nullptr);
    out.push_back(parquet_info);
  }
  return out;
}

std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  sscan::parquet_split_info const& info)
{
  REQUIRE_FALSE(info.rg_slices.empty());
  auto const& slice = info.rg_slices.front();
  REQUIRE(slice.file_metadata != nullptr);
  REQUIRE(info.reader_options != nullptr);
  sscan::hybrid_scan_reader reader(*slice.file_metadata, *info.reader_options);
  auto rg_span = cudf::host_span<cudf::size_type const>(slice.row_group_indices.data(),
                                                        slice.row_group_indices.size());
  auto ranges  = reader.all_column_chunks_byte_ranges(rg_span, *info.reader_options);
  REQUIRE_FALSE(ranges.empty());
  return ranges;
}

bool wait_for_cached_range(sirius::io::prefetching_cache& cache,
                           sirius::io::sirius_io_object const& obj,
                           cudf::io::text::byte_range_info const& range,
                           std::chrono::milliseconds timeout)
{
  // Give the worker a chance to claim queued prefetch entries before the first
  // read; reading a still-queued entry is a legitimate miss and can consume the
  // entry's request budget.
  std::this_thread::sleep_for(250ms);
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto view = cache.read(obj,
                               static_cast<std::size_t>(range.offset()),
                               static_cast<std::size_t>(range.size()),
                               nullptr);
        view) {
      return true;
    }
    std::this_thread::sleep_for(50ms);
  }
  return false;
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

TEST_CASE("parquet_gpu_ingestible leaves local slices unclaimed when Sirius datasource is disabled",
          "[scan][parquet_gpu_ingestible][datasource-routing]")
{
  // PR913 keeps the local uring backend constructed by scan_manager, but
  // use_sirius_datasource=false still means local parquet files are not claimed
  // by Sirius and materialize_table will fall back to cudf/kvikio.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_local_fallback_test";
  auto const path = write_decimal_parquet(dir,
                                          "local_fallback",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/2000,
                                          /*row_group_size=*/1000);

  auto exercise_local_fallback = [&](std::string scan_path, bool materialize) {
    auto table_info                 = make_table_info(path,
                                      /*precision=*/10,
                                      /*scale=*/2,
                                      /*filters=*/nullptr);
    table_info->resolved_file_paths = {std::move(scan_path)};

    sirius::scan_manager::scan_manager_config cfg{};
    cfg.use_sirius_datasource = false;
    sirius::scan_manager::sirius_scan_manager mgr(std::move(cfg));
    sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

    auto splits = drain_ingestible(ingestible);
    REQUIRE_FALSE(splits.empty());

    auto infos = split_infos(splits);
    for (auto const* parquet_info : infos) {
      REQUIRE_FALSE(parquet_info->rg_slices.empty());
      for (auto const& slice : parquet_info->rg_slices) {
        INFO("local slice " << slice.file_path);
        CHECK(slice.datasource == nullptr);
      }
    }

    if (materialize) {
      auto mem_mgr    = initialize_memory_manager(/*n_gpus=*/1);
      auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
      REQUIRE(gpu_space != nullptr);
      rmm::cuda_stream stream;
      for (auto const* parquet_info : infos) {
        auto filtered = ingestible.materialize_table(*parquet_info, *gpu_space, stream.view());
        REQUIRE(filtered.table != nullptr);
      }
    }
  };

  SECTION("bare local path") { exercise_local_fallback(path.string(), /*materialize=*/false); }

  SECTION("explicit file URI materializes through cudf fallback")
  {
    exercise_local_fallback("file://" + path.string(), /*materialize=*/true);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible keeps S3 slices claimed when local Sirius datasource is disabled",
          "[.][s3][integration][scan][parquet_gpu_ingestible][datasource-routing]")
{
  auto env = read_s3_test_env();
  if (skip_if_no_s3_env(env)) { return; }

  auto table_info                 = make_table_info(std::filesystem::path{},
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);
  table_info->resolved_file_paths = {s3_uri(env->bucket, "parquet/nation.parquet")};

  sirius::scan_manager::scan_manager_config cfg{};
  cfg.use_sirius_datasource      = false;
  cfg.s3_config                  = make_s3_config(*env);
  cfg.s3_use_async_backend       = true;
  cfg.uring_n_reactors           = 1;
  cfg.s3_thread_pool.num_threads = 2;
  sirius::scan_manager::sirius_scan_manager mgr(std::move(cfg));
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());
  for (auto const* parquet_info : split_infos(splits)) {
    REQUIRE_FALSE(parquet_info->rg_slices.empty());
    for (auto const& slice : parquet_info->rg_slices) {
      INFO("S3 slice " << slice.file_path);
      REQUIRE(slice.datasource != nullptr);
      auto sirius_ds = std::dynamic_pointer_cast<sirius::io::sirius_datasource>(slice.datasource);
      REQUIRE(sirius_ds != nullptr);
      auto* s3_backend = dynamic_cast<sirius::io::s3::s3_ioctx*>(sirius_ds->io_ctx().get());
      CHECK(s3_backend != nullptr);
    }
  }
}

TEST_CASE("parquet_gpu_ingestible scan-side prewarm inserts column chunk ranges only when enabled",
          "[scan][parquet_gpu_ingestible][prefetch]")
{
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_chunk_prewarm_test";
  auto const path = write_decimal_parquet(dir,
                                          "chunk_prewarm",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/2000,
                                          /*row_group_size=*/1000);

  auto run = [&](bool enable_chunk_prewarm) {
    host_cache_memory memory;
    sirius::scan_manager::scan_manager_config cfg{};
    cfg.use_sirius_datasource           = true;
    cfg.enable_prefetch_cache           = true;
    cfg.enable_chunk_prewarm            = enable_chunk_prewarm;
    cfg.prefetch_buffer_pool_bytes      = cache_capacity_bytes(cache_block_size, cache_max_slabs);
    cfg.prefetch_inflight_budget_chunks = 8;
    cfg.uring_n_reactors                = 1;
    sirius::scan_manager::sirius_scan_manager mgr(std::move(cfg), &memory.host_mr);

    auto table_info = make_table_info(path,
                                      /*precision=*/10,
                                      /*scale=*/2,
                                      /*filters=*/nullptr);
    sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

    auto splits = drain_ingestible(ingestible);
    REQUIRE_FALSE(splits.empty());
    auto infos = split_infos(splits);
    REQUIRE_FALSE(infos.empty());

    auto ranges = column_chunk_ranges(*infos.front());
    REQUIRE_FALSE(ranges.empty());
    auto const& first_range = ranges.front();

    auto ds = mgr.create_datasource(path.string());
    REQUIRE(ds != nullptr);
    auto ctx = ds->io_ctx();
    REQUIRE(ctx != nullptr);
    REQUIRE(ctx->cache() != nullptr);

    auto* cache               = ctx->cache();
    auto const hits_before    = cache->hit_count_total();
    auto const range_before   = cache->range_miss_count_total();
    auto const partial_before = cache->partial_miss_count_total();

    auto const hit = wait_for_cached_range(*cache, *ds->io_object(), first_range, 5s);

    return std::tuple{hit,
                      cache->hit_count_total() - hits_before,
                      cache->range_miss_count_total() - range_before,
                      cache->partial_miss_count_total() - partial_before};
  };

  SECTION("prewarm enabled")
  {
    auto const [hit, hit_delta, range_miss_delta, partial_miss_delta] =
      run(/*enable_chunk_prewarm=*/true);
    CHECK(hit);
    CHECK(hit_delta > 0);
    CHECK(range_miss_delta == 0);
    INFO("partial misses while waiting for prewarm: " << partial_miss_delta);
  }

  SECTION("prewarm disabled")
  {
    auto const [hit, hit_delta, range_miss_delta, partial_miss_delta] =
      run(/*enable_chunk_prewarm=*/false);
    CHECK_FALSE(hit);
    CHECK(hit_delta == 0);
    INFO("range misses with prewarm disabled: " << range_miss_delta);
    INFO("partial misses with prewarm disabled: " << partial_miss_delta);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - coalescer merges same-file row groups into one slice",
          "[scan][parquet_gpu_ingestible]")
{
  // One file with several row groups, and a batch cap large enough to hold them
  // all. The producer boxes one coalescing_unit per row group; make_batch must
  // merge those consecutive same-file units back into ONE row_group_slice whose
  // indices are contiguous and in walk order (the shape materialize_table reads).
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_merge_test";
  auto const path = write_decimal_parquet(dir,
                                          "merge",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/10000,
                                          /*row_group_size=*/1000);  // ~10 row groups

  auto table_info = make_table_info(path,
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);  // default cap = 1 GiB

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  auto splits = drain_ingestible(ingestible);
  REQUIRE(splits.size() == 1);  // the 1 GiB cap holds every row group in one batch

  auto* input = dynamic_cast<sscan::scan_operator_input*>(splits.front().get());
  REQUIRE(input != nullptr);
  auto const& parquet_info =
    dynamic_cast<sscan::parquet_split_info const&>(input->metadata->scan());

  REQUIRE(parquet_info.rg_slices.size() == 1);  // one file ⇒ all units merged into one slice
  auto const& idx = parquet_info.rg_slices.front().row_group_indices;
  REQUIRE(idx.size() >= 2);  // multiple row groups actually coalesced
  for (std::size_t i = 1; i < idx.size(); ++i) {
    REQUIRE(idx[i] == idx[i - 1] + 1);  // contiguous, in producer walk order
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - stamps coalesced batches round-robin across GPUs",
          "[scan][parquet_gpu_ingestible]")
{
  // Many row groups + a small byte cap → several coalesced batches. With a
  // balancing strategy installed on the (coalescing) ingestible, every emitted
  // batch must carry the next GPU in round-robin order. This is the parquet
  // analogue of the native test and is the consumer-side multi-GPU guarantee:
  // the producer's carriers are discarded, so the device id is stamped only on
  // the real decode batch in coalescing_gpu_ingestible::emit_batch. drain_ingestible
  // builds a split_provider with no strategy of its own, so a producer-side
  // double-stamp would break the strict 0,1,0,1 sequence and fail this test.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_multigpu_test";
  auto const path = write_decimal_parquet(dir,
                                          "multigpu",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/12000,
                                          /*row_group_size=*/1000);  // ~12 row groups

  auto table_info                    = make_table_info(path,
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);
  table_info->approximate_batch_size = 4096;  // override the 1 GiB default to force >1 batch

  sirius::scan_manager::sirius_scan_manager mgr({});
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr);

  // Two synthetic device ids: round_robin hands them out in order regardless of
  // how many physical GPUs exist, so the assertion is hardware-independent.
  std::vector<int> const devices{0, 1};
  ingestible.install_balancing(
    std::make_shared<sirius::scan_manager::round_robin_strategy>(devices), /*pipeline_id=*/0);

  auto splits = drain_ingestible(ingestible);
  REQUIRE(splits.size() >= 2);  // small cap → multiple coalesced batches to alternate over
  for (std::size_t i = 0; i < splits.size(); ++i) {
    auto const dev = splits[i]->get_preferred_device_id();
    REQUIRE(dev.has_value());
    REQUIRE(*dev == devices[i % devices.size()]);  // 0, 1, 0, 1, ...
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - s3 slice with non-null io_ctx routes through spy",
          "[scan][parquet_gpu_ingestible][s3][datasource-routing]")
{
  // When the scan_manager exposes no ioctx, materialize_table must still honor a
  // slice's own datasource and route the read through it.
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
  CHECK((spy->host_read_calls + spy->device_read_calls + spy->range_read_calls) > 0);

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - local slice with null io_ctx falls through to kvikio",
          "[scan][parquet_gpu_ingestible][datasource-routing]")
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
