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

#pragma once

// Shared harness for the two S3 benchmarks (s3_throughput_test, s3_parquet_test):
// CLI parsing, engine construction, bucket listing, parquet chunk discovery and
// reporting.  Everything here is backend-agnostic in principle but is only
// exercised against the REST reactor.
//
// AUTHENTICATION
//   Credentials come from a Sirius config YAML via --config, parsed by
//   sirius_config itself, so the benchmark authenticates exactly the way the
//   engine does (object_store: endpoint / region / access_key / secret_key /
//   session_token / transport / signing_mode / ca_bundle_path / tls_verify).
//   Individual fields can be overridden on the command line.
//
//   NOTE: there is no IMDS / instance-profile credential chain in the REST
//   backend -- only static credentials, with session_token for temporary ones.
//   Running on an EC2 instance role therefore still needs the keys supplied
//   explicitly (e.g. exported from `aws sts assume-role` or the IMDS endpoint).

#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "io/datasource_factory.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "scan_manager/config.hpp"
#include "sirius_config.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::bench {

using clock_type         = std::chrono::steady_clock;
using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

/// Host pinned pool geometry for the REST reactors' bounce staging.  Sized like
/// range_prefetch_benchmark's: 1 MiB blocks, 128 to a slab.
inline constexpr std::size_t host_block_size_v      = 1UL << 20;
inline constexpr std::size_t host_pool_size_v       = 128;
inline constexpr std::size_t host_initial_pools_v   = 64;  // 8 GiB of staging
inline constexpr std::size_t host_region_capacity_v = 32UL << 30;

inline double ms_since(clock_type::time_point t)
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------

struct bench_options {
  std::string bucket;   ///< s3://bucket/optional-prefix
  std::size_t n_files{12};
  std::size_t repeat{1};
  std::size_t n_threads{2};

  /// Cap on destination buffers fused into one scatter GET
  /// (io::rest::config::max_n_chunks).
  std::size_t max_segment{32};
  /// Max concurrent in-flight easy handles per reactor
  /// (io::rest::config::max_connections).
  std::size_t max_nconnection{128};
  /// Target maximum bytes per ranged GET, in MiB
  /// (io::rest::config::chunk_size).  Also the unit a byte range is split into
  /// when building the benchmark's own IO segments.
  std::size_t chunk_size_mib{1};

  /// Benchmark-specific selector: "grouped"/"chunked" for the throughput test,
  /// "pipelined"/"host_staged"/"gpu_staged" for the parquet test.
  std::string mode;

  /// Sirius config YAML supplying object_store credentials.
  std::string config_path;

  // Credential overrides; empty means "take it from --config".
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;

  std::size_t n_reactors{1};

  [[nodiscard]] std::size_t chunk_bytes() const noexcept { return chunk_size_mib << 20; }
};

/// Returns false when @p arg is not a recognised common flag, so the caller can
/// try its own.  Accepts both `--flag value` and `--flag=value`.
class arg_parser {
 public:
  arg_parser(int argc, char** argv) : _argc(argc), _argv(argv) {}

  /// Advances past the value when the flag matched.  @p i is the current index.
  bool match(int& i, std::string_view flag, std::string& out) const
  {
    const std::string_view arg = _argv[i];
    if (arg == flag) {
      if (i + 1 >= _argc) { throw std::runtime_error(std::string{flag} + " needs a value"); }
      out = _argv[++i];
      return true;
    }
    if (arg.starts_with(flag) && arg.size() > flag.size() && arg[flag.size()] == '=') {
      out = std::string{arg.substr(flag.size() + 1)};
      return true;
    }
    return false;
  }

  bool match(int& i, std::string_view flag, std::size_t& out) const
  {
    std::string s;
    if (!match(i, flag, s)) { return false; }
    out = std::stoul(s);
    return true;
  }

 private:
  int _argc;
  char** _argv;
};

/// Flags shared by both benchmarks.  Underscore and dash spellings are both
/// accepted, because the two specs use both.
inline bool parse_common_arg(arg_parser const& p, int& i, bench_options& o)
{
  return p.match(i, "--bucket", o.bucket) || p.match(i, "--n-files", o.n_files) ||
         p.match(i, "--n_files", o.n_files) || p.match(i, "--repeat", o.repeat) ||
         p.match(i, "--n-threads", o.n_threads) || p.match(i, "--n_threads", o.n_threads) ||
         p.match(i, "--max-segment", o.max_segment) || p.match(i, "--max_segment", o.max_segment) ||
         p.match(i, "--max-nconnection", o.max_nconnection) ||
         p.match(i, "--max_nconnection", o.max_nconnection) ||
         p.match(i, "--chunk-size", o.chunk_size_mib) ||
         p.match(i, "--chunk_size", o.chunk_size_mib) || p.match(i, "--mode", o.mode) ||
         p.match(i, "--strategy", o.mode) || p.match(i, "--config", o.config_path) ||
         p.match(i, "--endpoint", o.endpoint) || p.match(i, "--region", o.region) ||
         p.match(i, "--access-key", o.access_key) || p.match(i, "--secret-key", o.secret_key) ||
         p.match(i, "--session-token", o.session_token) ||
         p.match(i, "--n-reactors", o.n_reactors);
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

/// Reservation manager + REST ioctx + thread pool, torn down in the right
/// order.  Mirrors the setup in prefetch_hybrid_scan_benchmark: the
/// sirius_memory_reservation_manager installs the cucascade GPU allocator, and
/// the REST reactors take their pinned staging resource from its HOST tier.
class engine {
 public:
  explicit engine(bench_options const& opts)
  {
    // GPU + host tiers.  The host tier is what the REST reactors bounce
    // through, so it must exist even for the pure-host throughput arms.
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_reservation_fraction_per_gpu(0.9)
      .use_gpu_id_as_host_id()
      .set_per_numa_region_capacity(host_region_capacity_v)
      .set_reservation_fraction_per_numa_region(0.9)
      .set_host_pool_features(host_block_size_v, host_pool_size_v, host_initial_pools_v);
    _mgr = std::make_unique<memory::sirius_memory_reservation_manager>(builder.build());

    _cfg = build_scan_manager_config(opts);

    _registry = std::make_unique<io::io_context_registry>(_cfg, *_mgr);
    _registry->register_ioctx(
      io::io_context_type::restful,
      [](std::string_view path) { return path.starts_with("s3://"); },
      io::make_rest_ioctx_factory(*_mgr));

    _io_ctx = _registry->make_ioctx(io::io_context_type::restful);
    if (!_io_ctx) {
      throw std::runtime_error(
        "REST ioctx construction failed -- object_store endpoint/region/credentials are "
        "probably unset. Pass --config with an object_store block, or --endpoint/--region/"
        "--access-key/--secret-key.");
    }
    _io_ctx->start();

    _pool = std::make_unique<exec::static_thread_pool>(static_cast<int>(opts.n_threads),
                                                       "s3_bench");
  }

  ~engine()
  {
    if (_io_ctx) { _io_ctx->shutdown(); }
  }

  engine(engine const&)            = delete;
  engine& operator=(engine const&) = delete;

  [[nodiscard]] io::ioctx& io_ctx() const noexcept { return *_io_ctx; }
  [[nodiscard]] std::shared_ptr<io::ioctx> io_ctx_ptr() const noexcept { return _io_ctx; }
  [[nodiscard]] exec::static_thread_pool& pool() const noexcept { return *_pool; }
  [[nodiscard]] scan_manager::scan_manager_config const& config() const noexcept { return _cfg; }

  /// The REST ioctx as its concrete type, for LIST (not on the ioctx interface).
  [[nodiscard]] io::rest::rest_ioctx& rest() const
  {
    auto* r = dynamic_cast<io::rest::rest_ioctx*>(_io_ctx.get());
    if (r == nullptr) { throw std::runtime_error("ioctx is not a rest_ioctx"); }
    return *r;
  }

 private:
  static scan_manager::scan_manager_config build_scan_manager_config(bench_options const& opts)
  {
    scan_manager::scan_manager_config cfg;

    // Credentials: let sirius_config do the YAML parsing so the benchmark reads
    // the same object_store block the engine does.
    if (!opts.config_path.empty()) {
      sirius_config file_cfg;
      file_cfg.load_from_file(opts.config_path);
      cfg.object_store = file_cfg.get_scan_manager_config().object_store;
    }
    if (!opts.endpoint.empty()) { cfg.object_store.endpoint = opts.endpoint; }
    if (!opts.region.empty()) { cfg.object_store.region = opts.region; }
    if (!opts.access_key.empty()) { cfg.object_store.access_key = opts.access_key; }
    if (!opts.secret_key.empty()) { cfg.object_store.secret_key = opts.secret_key; }
    if (!opts.session_token.empty()) { cfg.object_store.session_token = opts.session_token; }

    // The knobs under test.
    cfg.rest.max_connections = opts.max_nconnection;
    cfg.rest.max_n_chunks    = opts.max_segment;
    cfg.rest.chunk_size      = opts.chunk_bytes();

    // No prefetching cache: these benchmarks measure the reactor, and a cache
    // in front of it would serve the repeat iterations from memory.
    cfg.cache = scan_manager::cache_mode::none;
    cfg.apply_cache_mode();
    return cfg;
  }

  std::unique_ptr<memory::sirius_memory_reservation_manager> _mgr;
  scan_manager::scan_manager_config _cfg;
  std::unique_ptr<io::io_context_registry> _registry;
  std::shared_ptr<io::ioctx> _io_ctx;
  std::unique_ptr<exec::static_thread_pool> _pool;
};

// ---------------------------------------------------------------------------
// bucket listing
// ---------------------------------------------------------------------------

/// Split `s3://bucket/prefix` into its bucket and prefix parts.
inline std::pair<std::string, std::string> split_s3_uri(std::string_view uri)
{
  constexpr std::string_view scheme = "s3://";
  if (!uri.starts_with(scheme)) {
    throw std::runtime_error("--bucket must be an s3:// URI, got: " + std::string{uri});
  }
  auto const rest  = uri.substr(scheme.size());
  auto const slash = rest.find('/');
  if (slash == std::string_view::npos) { return {std::string{rest}, std::string{}}; }
  return {std::string{rest.substr(0, slash)}, std::string{rest.substr(slash + 1)}};
}

/// The first @p n_files parquet objects under the bucket/prefix, as s3:// URIs.
inline std::vector<std::string> get_files_from_bucket(engine& eng,
                                                      std::string_view bucket_uri,
                                                      std::size_t n_files)
{
  auto const [bucket, prefix] = split_s3_uri(bucket_uri);

  std::vector<std::string> paths;
  // Paged rather than list_objects(): stop as soon as enough parquet keys have
  // been seen instead of accumulating a whole bucket listing first.
  eng.rest().list_objects_paged(
    bucket, prefix, 1000, [&](io::rest::s3::list_objects_v2_page const& page) {
      for (auto const& e : page.entries) {
        if (e.size == 0) { continue; }  // directory marker
        if (!std::string_view{e.key}.ends_with(".parquet")) { continue; }
        paths.push_back("s3://" + bucket + "/" + e.key);
        if (paths.size() >= n_files) { return false; }
      }
      return true;
    });

  if (paths.empty()) {
    throw std::runtime_error("no .parquet objects found under " + std::string{bucket_uri});
  }
  return paths;
}

// ---------------------------------------------------------------------------
// parquet chunk discovery
// ---------------------------------------------------------------------------

/// One file: its datasource, its row groups, its parsed footer, and the byte
/// ranges of every column chunk it will read.
struct file_ranges {
  std::string path;
  std::unique_ptr<io::sirius_datasource> ds;
  std::vector<cudf::size_type> row_groups;
  cudf::io::parquet::FileMetaData metadata;
  std::vector<cudf::io::text::byte_range_info> ranges;

  [[nodiscard]] std::size_t total_bytes() const
  {
    return std::accumulate(ranges.begin(), ranges.end(), std::size_t{0}, [](std::size_t a, auto& r) {
      return a + static_cast<std::size_t>(r.size());
    });
  }
};

/// Fetch each file's footer and ask a hybrid_scan_reader for every column
/// chunk's byte range.  All columns, since the benchmarks measure whole-file
/// throughput.
inline std::vector<file_ranges> use_hybrid_scan_to_get_column_chunks(
  engine& eng, std::vector<std::string> const& paths)
{
  auto const opts = cudf::io::parquet_reader_options::builder().build();

  std::vector<file_ranges> out;
  out.reserve(paths.size());
  for (auto const& path : paths) {
    file_ranges fr;
    fr.path = path;
    fr.ds   = eng.io_ctx().open_datasource(path);

    auto footer = cudf::io::parquet::fetch_footer_to_host(*fr.ds);
    hybrid_scan_reader reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                              opts);
    fr.row_groups = reader.all_row_groups(opts);
    fr.metadata   = reader.parquet_metadata();

    auto const rg_span =
      cudf::host_span<cudf::size_type const>(fr.row_groups.data(), fr.row_groups.size());
    fr.ranges = reader.all_column_chunks_byte_ranges(rg_span, opts);

    out.push_back(std::move(fr));
  }
  return out;
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

struct iteration_result {
  double duration_ms{0};
  std::size_t bytes{0};
};

inline void report(std::string_view label, std::vector<iteration_result> const& runs)
{
  if (runs.empty()) { return; }

  std::vector<double> gbps;
  gbps.reserve(runs.size());
  for (auto const& r : runs) {
    const double gb = static_cast<double>(r.bytes) / 1e9;
    gbps.push_back(r.duration_ms > 0 ? gb / (r.duration_ms / 1e3) : 0.0);
  }
  auto sorted = gbps;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted[sorted.size() / 2];

  std::cout << std::fixed;
  for (std::size_t i = 0; i < runs.size(); ++i) {
    std::cout << "  [" << label << "] iter " << i << ": throughput " << std::setprecision(3)
              << gbps[i] << " GB/s, data read " << std::setprecision(3)
              << static_cast<double>(runs[i].bytes) / 1e9 << " GB, duration "
              << std::setprecision(1) << runs[i].duration_ms << " msec\n";
  }
  std::cout << "  [" << label << "] median throughput: " << std::setprecision(3) << median
            << " GB/s\n";
}

}  // namespace sirius::bench
