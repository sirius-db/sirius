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
#include "io/s3/s3_blocking_ioctx.hpp"
#include "io/s3/s3_ioctx.hpp"
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

using sirius::io::s3::s3_authorized_request;
using sirius::io::s3::s3_ioctx;
using sirius::io::s3::s3_ioctx_config;
using sirius::io::s3::s3_object_ref;
using sirius::io::s3::s3_request_authorizer;
using sirius::io::s3::s3_request_method;
using sirius::io::s3::sirius_sigv4_presigned_authorizer;
using sirius::io::s3::static_credentials;

namespace {

namespace fs     = std::filesystem;
using clock_type = std::chrono::steady_clock;

constexpr std::uint64_t one_gib            = 1ULL << 30;
constexpr std::size_t kBenchMaxConnections = 32;

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

  bool const use_tls = !env_or("SIRIUS_BENCH_S3_USE_TLS").empty();
  auto endpoint =
    env_or("SIRIUS_BENCH_S3_ENDPOINT",
           use_tls ? env_or("SIRIUS_TEST_S3_HTTPS_ENDPOINT") : env_or("SIRIUS_TEST_S3_ENDPOINT"));
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

struct bench_cache_memory {
  static constexpr std::size_t block_size = sirius::io::CHUNK_SIZE;
  static constexpr std::size_t blocks     = kBenchMaxConnections;

  bench_cache_memory()
    : upstream(0, true),
      host_mr(0, upstream, block_size * blocks, block_size * blocks, block_size, blocks, 1)
  {
  }

  cucascade::memory::numa_region_pinned_host_memory_resource upstream;
  cucascade::memory::fixed_size_host_memory_resource host_mr;
};

std::shared_ptr<s3_ioctx> make_async_bench_ioctx(
  std::shared_ptr<recording_request_authorizer> provider,
  bench_cache_memory& memory,
  bool enable_perf_instrumentation)
{
  s3_ioctx_config cfg{};
  cfg.creds                = std::move(provider);
  cfg.max_connections      = kBenchMaxConnections;
  cfg.request_timeout_s    = 600;
  cfg.host_memory_resource = &memory.host_mr;
  cfg.ca_bundle_path = env_or("SIRIUS_BENCH_S3_CA_BUNDLE", env_or("SIRIUS_TEST_S3_CA_BUNDLE"));
  cfg.s3_perf_instrumentation = enable_perf_instrumentation;
  return std::make_shared<s3_ioctx>(std::move(cfg));
}

struct micro_snapshot {
  std::uint64_t chunk_get_ns_total{0};
  std::uint64_t chunk_get_ns_count{0};
  std::uint64_t chunk_get_ns_max{0};
  std::uint64_t queue_wait_ns_total{0};
  std::uint64_t queue_wait_ns_count{0};
  std::uint64_t h2d_observed_ns_total{0};
  std::uint64_t h2d_observed_ns_count{0};
  std::uint64_t h2d_observed_ns_max{0};
  std::uint64_t ttfb_ns{0};
  std::uint64_t retries_total{0};
  std::uint64_t terminal_failures_total{0};
};

micro_snapshot snapshot_micro(s3_ioctx const& ctx)
{
  return micro_snapshot{ctx.chunk_get_ns_total(),
                        ctx.chunk_get_ns_count(),
                        ctx.chunk_get_ns_max(),
                        ctx.queue_wait_ns_total(),
                        ctx.queue_wait_ns_count(),
                        ctx.h2d_observed_ns_total(),
                        ctx.h2d_observed_ns_count(),
                        ctx.h2d_observed_ns_max(),
                        ctx.ttfb_ns(),
                        ctx.retries_total(),
                        ctx.terminal_failures_total()};
}

micro_snapshot delta(micro_snapshot const& after, micro_snapshot const& before)
{
  return micro_snapshot{after.chunk_get_ns_total - before.chunk_get_ns_total,
                        after.chunk_get_ns_count - before.chunk_get_ns_count,
                        after.chunk_get_ns_max,
                        after.queue_wait_ns_total - before.queue_wait_ns_total,
                        after.queue_wait_ns_count - before.queue_wait_ns_count,
                        after.h2d_observed_ns_total - before.h2d_observed_ns_total,
                        after.h2d_observed_ns_count - before.h2d_observed_ns_count,
                        after.h2d_observed_ns_max,
                        after.ttfb_ns - before.ttfb_ns,
                        after.retries_total - before.retries_total,
                        after.terminal_failures_total - before.terminal_failures_total};
}

double elapsed_ms(clock_type::time_point start, clock_type::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double mean_ns_as_ms(std::uint64_t total_ns, std::uint64_t count)
{
  if (count == 0) return 0.0;
  return static_cast<double>(total_ns) / static_cast<double>(count) / 1'000'000.0;
}

struct scan_measurement {
  double wall_clock_ms{0.0};
  double open_ms{0.0};
  double footer_fetch_ms{0.0};
  double metadata_parse_ms{0.0};
  double scan_ms{0.0};
  std::uint64_t wire_bytes_read{0};
  std::int64_t rows{0};
  micro_snapshot micro;
  std::uint64_t device_peak_inflight{0};
  std::uint64_t device_stream_sync_total{0};
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

scan_measurement run_async_parquet_scan(s3_ioctx& ctx,
                                        std::string const& path,
                                        std::vector<std::string> const& columns)
{
  auto const before_bytes = ctx.bytes_read_total();
  auto const before_micro = snapshot_micro(ctx);

  auto const wall_start = clock_type::now();

  auto const open_start = clock_type::now();
  auto source           = ctx.open_datasource(path);
  auto const open_stop  = clock_type::now();

  auto const footer_start = clock_type::now();
  auto footer_buffer      = read_parquet_footer(*source);
  auto const footer_stop  = clock_type::now();
  auto opts = cudf::io::parquet_reader_options::builder().column_names(columns).build();

  auto const parse_start = clock_type::now();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer_buffer->data(), footer_buffer->size()), opts};
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  metadatas.push_back(reader.parquet_metadata());
  auto const parse_stop = clock_type::now();

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  sources.push_back(std::move(source));

  auto const scan_start  = clock_type::now();
  auto [table, metadata] = cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts);
  (void)metadata;
  auto const scan_stop   = clock_type::now();
  auto const wall_stop   = clock_type::now();
  auto const after_bytes = ctx.bytes_read_total();
  auto const after_micro = snapshot_micro(ctx);

  return scan_measurement{elapsed_ms(wall_start, wall_stop),
                          elapsed_ms(open_start, open_stop),
                          elapsed_ms(footer_start, footer_stop),
                          elapsed_ms(parse_start, parse_stop),
                          elapsed_ms(scan_start, scan_stop),
                          static_cast<std::uint64_t>(after_bytes - before_bytes),
                          table->num_rows(),
                          delta(after_micro, before_micro),
                          ctx.device_peak_inflight(),
                          ctx.device_stream_sync_total()};
}

struct bench_record {
  std::string scenario;
  double wall_clock_ms{0.0};
  double open_ms{0.0};
  double footer_fetch_ms{0.0};
  double metadata_parse_ms{0.0};
  double scan_ms{0.0};
  std::uint64_t wire_bytes_read{0};
  double effective_bytes_per_sec{0.0};
  std::uint64_t chunk_get_ns_total{0};
  std::uint64_t chunk_get_ns_count{0};
  std::uint64_t chunk_get_ns_max{0};
  double chunk_get_ms_mean{0.0};
  std::uint64_t queue_wait_ns_total{0};
  std::uint64_t queue_wait_ns_count{0};
  double queue_wait_ms_mean{0.0};
  std::uint64_t h2d_observed_ns_total{0};
  std::uint64_t h2d_observed_ns_count{0};
  std::uint64_t h2d_observed_ns_max{0};
  double h2d_observed_ms_mean{0.0};
  std::uint64_t ttfb_ns{0};
  std::uint64_t device_peak_inflight{0};
  std::uint64_t device_stream_sync_total{0};
  std::uint64_t retries_total{0};
  std::uint64_t terminal_failures_total{0};
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
        << "\"open_ms\": " << r.open_ms << ", "
        << "\"footer_fetch_ms\": " << r.footer_fetch_ms << ", "
        << "\"metadata_parse_ms\": " << r.metadata_parse_ms << ", "
        << "\"scan_ms\": " << r.scan_ms << ", "
        << "\"wire_bytes_read\": " << r.wire_bytes_read << ", "
        << "\"effective_bytes_per_sec\": " << r.effective_bytes_per_sec << ", "
        << "\"chunk_get_ns_total\": " << r.chunk_get_ns_total << ", "
        << "\"chunk_get_ns_count\": " << r.chunk_get_ns_count << ", "
        << "\"chunk_get_ns_max\": " << r.chunk_get_ns_max << ", "
        << "\"chunk_get_ms_mean\": " << r.chunk_get_ms_mean << ", "
        << "\"queue_wait_ns_total\": " << r.queue_wait_ns_total << ", "
        << "\"queue_wait_ns_count\": " << r.queue_wait_ns_count << ", "
        << "\"queue_wait_ms_mean\": " << r.queue_wait_ms_mean << ", "
        << "\"h2d_observed_ns_total\": " << r.h2d_observed_ns_total << ", "
        << "\"h2d_observed_ns_count\": " << r.h2d_observed_ns_count << ", "
        << "\"h2d_observed_ns_max\": " << r.h2d_observed_ns_max << ", "
        << "\"h2d_observed_ms_mean\": " << r.h2d_observed_ms_mean << ", "
        << "\"ttfb_ns\": " << r.ttfb_ns << ", "
        << "\"device_peak_inflight\": " << r.device_peak_inflight << ", "
        << "\"device_stream_sync_total\": " << r.device_stream_sync_total << ", "
        << "\"retries_total\": " << r.retries_total << ", "
        << "\"terminal_failures_total\": " << r.terminal_failures_total << ", "
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
                   "\"open_ms\"",
                   "\"footer_fetch_ms\"",
                   "\"metadata_parse_ms\"",
                   "\"scan_ms\"",
                   "\"effective_bytes_per_sec\"",
                   "\"chunk_get_ns_total\"",
                   "\"chunk_get_ns_count\"",
                   "\"chunk_get_ns_max\"",
                   "\"chunk_get_ms_mean\"",
                   "\"queue_wait_ns_total\"",
                   "\"queue_wait_ns_count\"",
                   "\"queue_wait_ms_mean\"",
                   "\"h2d_observed_ns_total\"",
                   "\"h2d_observed_ns_count\"",
                   "\"h2d_observed_ns_max\"",
                   "\"h2d_observed_ms_mean\"",
                   "\"ttfb_ns\"",
                   "\"device_peak_inflight\"",
                   "\"device_stream_sync_total\"",
                   "\"retries_total\"",
                   "\"terminal_failures_total\"",
                   "\"config\""}) {
    CHECK(json.find(key) != std::string::npos);
  }
}

bench_record make_record(std::string scenario,
                         scan_measurement measurement,
                         std::uint64_t dataset_bytes,
                         std::string warning = {})
{
  auto seconds = measurement.wall_clock_ms / 1000.0;
  auto effective =
    seconds > 0.0 ? static_cast<double>(dataset_bytes) / seconds : static_cast<double>(0);
  auto const& micro = measurement.micro;
  return bench_record{std::move(scenario),
                      measurement.wall_clock_ms,
                      measurement.open_ms,
                      measurement.footer_fetch_ms,
                      measurement.metadata_parse_ms,
                      measurement.scan_ms,
                      measurement.wire_bytes_read,
                      effective,
                      micro.chunk_get_ns_total,
                      micro.chunk_get_ns_count,
                      micro.chunk_get_ns_max,
                      mean_ns_as_ms(micro.chunk_get_ns_total, micro.chunk_get_ns_count),
                      micro.queue_wait_ns_total,
                      micro.queue_wait_ns_count,
                      mean_ns_as_ms(micro.queue_wait_ns_total, micro.queue_wait_ns_count),
                      micro.h2d_observed_ns_total,
                      micro.h2d_observed_ns_count,
                      micro.h2d_observed_ns_max,
                      mean_ns_as_ms(micro.h2d_observed_ns_total, micro.h2d_observed_ns_count),
                      micro.ttfb_ns,
                      measurement.device_peak_inflight,
                      measurement.device_stream_sync_total,
                      micro.retries_total,
                      micro.terminal_failures_total,
                      std::move(warning)};
}

}  // namespace

TEST_CASE("S3 async perf instrumentation accessors are noexcept", "[.][s3][bench]")
{
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().chunk_get_ns_total()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().chunk_get_ns_count()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().chunk_get_ns_max()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().queue_wait_ns_total()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().queue_wait_ns_count()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().h2d_observed_ns_total()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().h2d_observed_ns_count()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().h2d_observed_ns_max()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().ttfb_ns()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().retries_total()));
  STATIC_REQUIRE(noexcept(std::declval<s3_ioctx const&>().terminal_failures_total()));
}

TEST_CASE("S3 parquet perf benchmark emits async JSON baseline", "[.][s3][bench]")
{
  auto env = read_bench_env();
  if (!env) {
    WARN("Skipping S3 perf benchmark because SIRIUS_BENCH_* environment is not configured");
    return;
  }

  auto path = s3_uri(*env);

  std::uint64_t dataset_bytes = 0;
  try {
    bench_cache_memory probe_memory;
    auto provider = std::make_shared<recording_request_authorizer>(*env);
    auto ctx      = make_async_bench_ioctx(provider, probe_memory, false);
    dataset_bytes = ctx->head_object_size(env->bucket, env->key);
  } catch (std::exception const& e) {
    WARN("Skipping S3 perf benchmark because the fixture object is unavailable: " << e.what());
    return;
  }

  if (dataset_bytes <= one_gib) {
    WARN("S3 perf fixture is smaller than 1 GiB; generate SF=10 before using this as a baseline");
  }

  bench_cache_memory cache_memory;
  auto provider = std::make_shared<recording_request_authorizer>(*env);
  auto ctx      = make_async_bench_ioctx(provider, cache_memory, true);

  std::vector<std::string> full_columns{
    "l_orderkey",
    "l_partkey",
    "l_suppkey",
    "l_linenumber",
    "l_quantity",
    "l_extendedprice",
    "l_discount",
  };

  std::vector<bench_record> records;

  auto async_full = run_async_parquet_scan(*ctx, path, full_columns);
  CHECK(async_full.wire_bytes_read > 0);
  CHECK(async_full.open_ms > 0.0);
  CHECK(async_full.footer_fetch_ms > 0.0);
  CHECK(async_full.metadata_parse_ms > 0.0);
  CHECK(async_full.scan_ms > 0.0);
  CHECK(async_full.wall_clock_ms >=
        async_full.open_ms + async_full.footer_fetch_ms + async_full.scan_ms);
  CHECK(async_full.micro.chunk_get_ns_count > 0);
  CHECK(async_full.micro.chunk_get_ns_total > 0);
  CHECK(async_full.micro.chunk_get_ns_max > 0);
  CHECK(async_full.micro.queue_wait_ns_count > 0);
  CHECK(async_full.micro.h2d_observed_ns_count > 0);
  CHECK(async_full.micro.h2d_observed_ns_total > 0);
  CHECK(async_full.micro.ttfb_ns > 0);
  CHECK(async_full.micro.ttfb_ns <= async_full.micro.chunk_get_ns_max);
  CHECK(async_full.device_peak_inflight >= 1);
  CHECK(async_full.device_stream_sync_total == 0);
  CHECK(async_full.micro.retries_total == 0);
  CHECK(async_full.micro.terminal_failures_total == 0);
  records.push_back(make_record("async_full_scan", async_full, dataset_bytes));

  bench_cache_memory gate_off_memory;
  auto gate_off_provider = std::make_shared<recording_request_authorizer>(*env);
  auto gate_off_ctx      = make_async_bench_ioctx(gate_off_provider, gate_off_memory, false);
  auto gate_off          = run_async_parquet_scan(*gate_off_ctx, path, full_columns);
  CHECK(gate_off.wire_bytes_read > 0);
  CHECK(gate_off.device_peak_inflight >= 1);
  CHECK(gate_off.micro.chunk_get_ns_count == 0);
  CHECK(gate_off.micro.chunk_get_ns_total == 0);
  CHECK(gate_off.micro.queue_wait_ns_count == 0);
  CHECK(gate_off.micro.queue_wait_ns_total == 0);
  CHECK(gate_off.micro.h2d_observed_ns_count == 0);
  CHECK(gate_off.micro.h2d_observed_ns_total == 0);
  CHECK(gate_off.micro.ttfb_ns == 0);

  auto json_path = perf_json_path();
  write_perf_json(json_path, *env, dataset_bytes, records);
  CHECK(fs::exists(json_path));
  require_json_schema(json_path);
}
