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

#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/s3/s3_request_authorizer.hpp"
#include "io/s3/sirius_sigv4_authorizer.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/span.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <io/sirius_datasource.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using sirius::io::buffer_pool;
using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_blocking_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::s3_request_authorizer;
using sirius::io::s3::s3_request_method;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
using sirius::io::s3::static_credentials;

namespace {

namespace fs     = std::filesystem;
using clock_type = std::chrono::steady_clock;

constexpr std::uint64_t one_gib = 1ULL << 30;

std::string env_or(std::string_view name, std::string fallback = {})
{
  auto const* value = std::getenv(std::string{name}.c_str());
  return value ? std::string{value} : std::move(fallback);
}

struct bench_env {
  std::string backend;
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string bucket;
  std::string key;
};

std::optional<bench_env> read_bench_env()
{
  auto backend = env_or("SIRIUS_BENCH_BACKEND", "minio");
  if (backend == "aws-s3") {
    auto access_key = env_or("SIRIUS_BENCH_AWS_S3_ACCESS_KEY");
    auto secret_key = env_or("SIRIUS_BENCH_AWS_S3_SECRET_KEY");
    auto bucket     = env_or("SIRIUS_BENCH_AWS_S3_BUCKET");
    auto key        = env_or("SIRIUS_BENCH_AWS_S3_KEY", "tpch/lineitem_sf10.parquet");
    if (access_key.empty() || secret_key.empty() || bucket.empty()) { return std::nullopt; }
    return bench_env{std::move(backend),
                     env_or("SIRIUS_BENCH_AWS_S3_ENDPOINT", "https://s3.amazonaws.com"),
                     env_or("SIRIUS_BENCH_AWS_S3_REGION", "us-east-1"),
                     std::move(access_key),
                     std::move(secret_key),
                     std::move(bucket),
                     std::move(key)};
  }

  auto endpoint   = env_or("SIRIUS_BENCH_S3_ENDPOINT", env_or("SIRIUS_TEST_S3_ENDPOINT"));
  auto access_key = env_or("SIRIUS_BENCH_S3_ACCESS_KEY", env_or("SIRIUS_TEST_S3_ACCESS_KEY"));
  auto secret_key = env_or("SIRIUS_BENCH_S3_SECRET_KEY", env_or("SIRIUS_TEST_S3_SECRET_KEY"));
  auto bucket     = env_or("SIRIUS_BENCH_S3_BUCKET", env_or("SIRIUS_TEST_S3_BUCKET"));
  auto key        = env_or("SIRIUS_BENCH_S3_KEY", "tpch/lineitem_sf10.parquet");
  if (endpoint.empty() || access_key.empty() || secret_key.empty() || bucket.empty()) {
    return std::nullopt;
  }
  return bench_env{std::move(backend),
                   std::move(endpoint),
                   env_or("SIRIUS_BENCH_S3_REGION", env_or("SIRIUS_TEST_S3_REGION", "us-east-1")),
                   std::move(access_key),
                   std::move(secret_key),
                   std::move(bucket),
                   std::move(key)};
}

std::string s3_uri(bench_env const& env) { return "s3://" + env.bucket + "/" + env.key; }

class recording_request_authorizer final : public s3_request_authorizer {
 public:
  explicit recording_request_authorizer(bench_env const& env) : _endpoint(env.endpoint)
  {
    static_credentials creds;
    creds.access_key_id     = env.access_key;
    creds.secret_access_key = env.secret_key;
    _delegate               = std::make_shared<sirius_sigv4_presigned_authorizer>(
      std::move(creds), env.region, env.endpoint, std::chrono::minutes{30});
  }

  s3_authorized_request authorize(s3_object_ref const& obj,
                                  s3_request_method method,
                                  std::chrono::seconds timeout) override
  {
    if (method == s3_request_method::GET) { _get_count.fetch_add(1, std::memory_order_relaxed); }
    return _delegate->authorize(obj, method, timeout);
  }

  [[nodiscard]] std::string const& endpoint() const noexcept { return _endpoint; }
  [[nodiscard]] std::uint64_t get_count() const noexcept
  {
    return _get_count.load(std::memory_order_relaxed);
  }

 private:
  std::shared_ptr<s3_request_authorizer> _delegate;
  std::string _endpoint;
  std::atomic<std::uint64_t> _get_count{0};
};

std::shared_ptr<s3_blocking_ioctx> make_bench_ioctx(
  std::shared_ptr<recording_request_authorizer> provider)
{
  s3_ioctx_config cfg{};
  cfg.creds             = std::move(provider);
  cfg.max_connections   = 32;
  cfg.request_timeout_s = 600;
  return std::make_shared<s3_blocking_ioctx>(std::move(cfg));
}

std::size_t cache_capacity_bytes(std::size_t block_size, std::uint32_t max_slabs)
{
  return block_size * static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB) *
         static_cast<std::size_t>(max_slabs);
}

struct bench_cache_memory {
  static constexpr std::uint32_t max_slabs = 4;
  static constexpr std::size_t block_size  = 1ULL << 20;

  bench_cache_memory()
    : upstream(0, true),
      host_mr(0,
              upstream,
              cache_capacity_bytes(block_size, max_slabs),
              cache_capacity_bytes(block_size, max_slabs),
              block_size,
              static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB),
              1),
      pool(host_mr, max_slabs)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource host_mr;
  buffer_pool pool;
};

struct scan_measurement {
  double wall_clock_ms{0.0};
  std::uint64_t wire_bytes_read{0};
  std::int64_t rows{0};
};

std::unique_ptr<cudf::io::datasource::buffer> read_parquet_footer(cudf::io::datasource& source)
{
  auto constexpr footer_tail_size = sizeof(cudf::io::parquet::file_ender_s);

  auto const file_size = source.size();
  REQUIRE(file_size >= footer_tail_size);

  auto tail = source.host_read(file_size - footer_tail_size, footer_tail_size);

  std::uint32_t footer_size = 0;
  std::memcpy(&footer_size, tail->data(), sizeof(footer_size));
  REQUIRE(file_size >= footer_tail_size + footer_size);

  return source.host_read(file_size - footer_tail_size - footer_size, footer_size);
}

scan_measurement run_parquet_scan(s3_blocking_ioctx& ctx,
                                  std::string const& path,
                                  std::vector<std::string> const& columns)
{
  auto const before_bytes = ctx.bytes_read_total();

  auto probe         = ctx.open_datasource(path);
  auto footer_buffer = read_parquet_footer(*probe);
  auto opts          = cudf::io::parquet_reader_options::builder().column_names(columns).build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()), opts};
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  metadatas.push_back(reader.parquet_metadata());

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(ctx.open_datasource(path));

  auto const t0          = clock_type::now();
  auto [table, metadata] = cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts);
  (void)metadata;
  auto const ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
  auto const after_bytes = ctx.bytes_read_total();

  return scan_measurement{
    ms, static_cast<std::uint64_t>(after_bytes - before_bytes), table->num_rows()};
}

struct bench_record {
  std::string scenario;
  double wall_clock_ms{0.0};
  std::uint64_t wire_bytes_read{0};
  double effective_bytes_per_sec{0.0};
  double cold_warm_ratio{0.0};
  std::string warning;
};

std::string json_escape(std::string_view value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '"') {
      escaped.push_back('\\');
      escaped.push_back(c);
    } else if (c == '\n') {
      escaped += "\\n";
    } else {
      escaped.push_back(c);
    }
  }
  return escaped;
}

fs::path unittest_log_dir()
{
#ifdef SIRIUS_UNITTEST_LOG_DIR
  return fs::path{SIRIUS_UNITTEST_LOG_DIR};
#else
  return fs::path{"test/cpp/log"};
#endif
}

fs::path perf_json_path()
{
  auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  return unittest_log_dir() / ("perf_" + std::to_string(stamp) + ".json");
}

void write_perf_json(fs::path const& path,
                     bench_env const& env,
                     std::uint64_t dataset_bytes,
                     std::vector<bench_record> const& records)
{
  fs::create_directories(path.parent_path());

  std::ofstream out(path);
  REQUIRE(out.good());
  out << "{\n";
  out << "  \"git_sha\": \"" << json_escape(env_or("SIRIUS_BENCH_GIT_SHA", "unknown")) << "\",\n";
  out << "  \"host\": \"" << json_escape(env_or("HOSTNAME", "unknown")) << "\",\n";
  out << "  \"backend\": \"" << json_escape(env.backend) << "\",\n";
  out << "  \"dataset_bytes\": " << dataset_bytes << ",\n";
  out << "  \"config\": {\"bucket\": \"" << json_escape(env.bucket) << "\", \"key\": \""
      << json_escape(env.key) << "\"},\n";
  out << "  \"results\": [\n";
  for (std::size_t i = 0; i < records.size(); ++i) {
    auto const& r = records[i];
    out << "    {\"scenario\": \"" << json_escape(r.scenario) << "\", "
        << "\"wall_clock_ms\": " << std::fixed << std::setprecision(3) << r.wall_clock_ms << ", "
        << "\"wire_bytes_read\": " << r.wire_bytes_read << ", "
        << "\"effective_bytes_per_sec\": " << r.effective_bytes_per_sec << ", "
        << "\"cold_warm_ratio\": " << r.cold_warm_ratio << ", "
        << "\"warning\": \"" << json_escape(r.warning) << "\"}";
    out << (i + 1 == records.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";
}

std::string read_text_file(fs::path const& path)
{
  std::ifstream in(path);
  REQUIRE(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void require_json_schema(fs::path const& path)
{
  auto json = read_text_file(path);
  for (auto key : {"\"git_sha\"",
                   "\"host\"",
                   "\"backend\"",
                   "\"dataset_bytes\"",
                   "\"scenario\"",
                   "\"wall_clock_ms\"",
                   "\"effective_bytes_per_sec\"",
                   "\"cold_warm_ratio\"",
                   "\"config\""}) {
    CHECK(json.find(key) != std::string::npos);
  }
}

bench_record make_record(std::string scenario,
                         scan_measurement measurement,
                         std::uint64_t dataset_bytes,
                         double cold_warm_ratio = 0.0,
                         std::string warning    = {})
{
  auto seconds = measurement.wall_clock_ms / 1000.0;
  auto effective =
    seconds > 0.0 ? static_cast<double>(dataset_bytes) / seconds : static_cast<double>(0);
  return bench_record{std::move(scenario),
                      measurement.wall_clock_ms,
                      measurement.wire_bytes_read,
                      effective,
                      cold_warm_ratio,
                      std::move(warning)};
}

}  // namespace

TEST_CASE("S3 parquet perf benchmark emits portable JSON baseline", "[!benchmark][perf][bench]")
{
  auto env = read_bench_env();
  if (!env) {
    WARN("Skipping S3 perf benchmark because SIRIUS_BENCH_* environment is not configured");
    return;
  }

  auto provider = std::make_shared<recording_request_authorizer>(*env);
  auto ctx      = make_bench_ioctx(provider);
  auto path     = s3_uri(*env);

  std::uint64_t dataset_bytes = 0;
  try {
    dataset_bytes = ctx->head_object_size(env->bucket, env->key);
  } catch (std::exception const& e) {
    WARN("Skipping S3 perf benchmark because the fixture object is unavailable: " << e.what());
    return;
  }

  if (dataset_bytes <= one_gib) {
    WARN("S3 perf fixture is smaller than 1 GiB; generate SF=10 before using this as a baseline");
  }

  bench_cache_memory cache_memory;
  ctx->initialize_cache(cache_memory.pool, 2048);

  std::vector<std::string> full_columns{
    "l_orderkey",
    "l_partkey",
    "l_suppkey",
    "l_linenumber",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
  };
  std::vector<std::string> selective_columns{"l_orderkey", "l_extendedprice"};

  std::vector<bench_record> records;

  auto cold_full = run_parquet_scan(*ctx, path, full_columns);
  records.push_back(make_record("full_sequential_scan", cold_full, dataset_bytes));

  auto selective = run_parquet_scan(*ctx, path, selective_columns);
  records.push_back(make_record("selective_two_column_scan", selective, dataset_bytes));
  CHECK(selective.wire_bytes_read > 0);

  if (env->backend == "aws-s3") {
    CHECK((provider->endpoint().find("amazonaws") != std::string::npos ||
           selective.wire_bytes_read > 0));
    records.push_back(bench_record{"aws_s3_portability", 0.0, 0, 0.0, 0.0, {}});
  } else {
    records.push_back(
      bench_record{"aws_s3_portability_skipped", 0.0, 0, 0.0, 0.0, "aws env absent"});
  }

  auto warm_full = run_parquet_scan(*ctx, path, full_columns);
  auto ratio =
    cold_full.wall_clock_ms > 0.0 ? warm_full.wall_clock_ms / cold_full.wall_clock_ms : 0.0;
  std::string cache_warning;
  if (ctx->cache()->total_size_bytes() < cold_full.wire_bytes_read) {
    cache_warning = "cache_too_small_for_full_object";
  }
  records.push_back(make_record("warm_full_scan", warm_full, dataset_bytes, ratio, cache_warning));

  auto json_path = perf_json_path();
  write_perf_json(json_path, *env, dataset_bytes, records);
  CHECK(fs::exists(json_path));
  require_json_schema(json_path);
}
