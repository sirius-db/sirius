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

// Shared harness for S3 benchmarks (s3_throughput_test):
// CLI parsing, engine construction, bucket listing and reporting.
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

#include "io/datasource_factory.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "scan_manager/config.hpp"
#include "sirius_config.hpp"

#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::bench {

using clock_type = std::chrono::steady_clock;

/// Host pinned pool geometry for the REST reactors' bounce staging.
inline constexpr std::size_t host_pool_size_v       = 128;
inline constexpr std::size_t host_initial_pools_v   = 64;
inline constexpr std::size_t host_region_capacity_v = 32UL << 30;

inline double ms_since(clock_type::time_point t)
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - t).count();
}

// ---------------------------------------------------------------------------
// options
// ---------------------------------------------------------------------------

struct bench_options {
  std::vector<std::string> buckets;  ///< one or more s3:// prefixes (--bucket repeatable)
  std::size_t n_files{12};           ///< total files across all prefixes (divided evenly)
  std::size_t per_file_gib{1};       ///< GB of data to read per file (random segments)
  std::size_t repeat{3};

  /// Logical request size in MiB. The REST reactor may split a logical request
  /// further according to its current connection availability and backlog.
  std::size_t chunk_size_mib{1};

  /// Exact GET size in bytes, overriding @c chunk_size_mib when non-zero.  For
  /// a benchmark whose target GET size is not a whole number of MiB -- the
  /// autotune one derives it from bandwidth x latency.
  std::size_t chunk_size_bytes{0};

  /// Max concurrent in-flight easy handles per reactor (rest.max_connections).
  std::size_t max_nconnection{128};

  /// Number of REST reactor instances.
  std::size_t n_reactors{1};

  /// Share of each GPU's memory the pool may use.  Only matters for benchmarks
  /// that allocate device memory; raw-throughput ones can leave it alone.
  double gpu_usage_ratio{0.5};

  /// Host pinned pool slabs allocated up front.  host_pool_size_v blocks each,
  /// so initial bytes = host_initial_pools * host_pool_size_v * host_block_bytes().
  std::size_t host_initial_pools{host_initial_pools_v};

  /// Host memory pool block size in MiB.  Zero means "same as chunk_size_mib",
  /// which is what a throughput benchmark wants (staging block == GET size); a
  /// benchmark that stages differently from how it reads sets it explicitly.
  std::size_t host_chunk_mib{0};

  /// Sirius config YAML supplying object_store credentials.
  std::string config_path;

  // Credential overrides; empty means "take it from --config".
  std::string endpoint;
  std::string region;
  std::string access_key;
  std::string secret_key;
  std::string session_token;

  [[nodiscard]] std::size_t chunk_bytes() const noexcept
  {
    return chunk_size_bytes != 0 ? chunk_size_bytes : (chunk_size_mib << 20);
  }
  [[nodiscard]] std::size_t host_block_bytes() const noexcept
  {
    return (host_chunk_mib == 0 ? chunk_size_mib : host_chunk_mib) << 20;
  }
  [[nodiscard]] std::size_t per_file_bytes() const noexcept
  {
    return per_file_gib * 1'000'000'000ULL;
  }
};

class arg_parser {
 public:
  arg_parser(int argc, char** argv) : _argc(argc), _argv(argv) {}

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

  bool match(int& i, std::string_view flag, double& out) const
  {
    std::string s;
    if (!match(i, flag, s)) { return false; }
    out = std::stod(s);
    return true;
  }

  /// Valueless switch: `--name` present sets @p out.  Not an overload of
  /// @c match -- it consumes no value, so it must not take @c i by reference.
  bool toggle(int i, std::string_view name, bool& out) const
  {
    if (std::string_view{_argv[i]} != name) { return false; }
    out = true;
    return true;
  }

  /// Repeatable flag: appends each occurrence to a vector.
  bool match(int& i, std::string_view flag, std::vector<std::string>& out) const
  {
    std::string s;
    if (!match(i, flag, s)) { return false; }
    out.push_back(std::move(s));
    return true;
  }

 private:
  int _argc;
  char** _argv;
};

inline bool parse_common_arg(arg_parser const& p, int& i, bench_options& o)
{
  return p.match(i, "--bucket", o.buckets) || p.match(i, "--n-files", o.n_files) ||
         p.match(i, "--n_files", o.n_files) || p.match(i, "--per-file", o.per_file_gib) ||
         p.match(i, "--per_file", o.per_file_gib) || p.match(i, "--repeat", o.repeat) ||
         p.match(i, "--chunk-size", o.chunk_size_mib) ||
         p.match(i, "--chunk_size", o.chunk_size_mib) ||
         p.match(i, "--max-connection", o.max_nconnection) ||
         p.match(i, "--max_connection", o.max_nconnection) ||
         p.match(i, "--n-reactors", o.n_reactors) || p.match(i, "--n_reactors", o.n_reactors) ||
         p.match(i, "--config", o.config_path) || p.match(i, "--endpoint", o.endpoint) ||
         p.match(i, "--region", o.region) || p.match(i, "--access-key", o.access_key) ||
         p.match(i, "--secret-key", o.secret_key) || p.match(i, "--session-token", o.session_token);
}

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

class engine {
 public:
  explicit engine(bench_options const& opts)
  {
    // Defaults to chunk_bytes so the fixed-size host pool, the REST reactor's
    // chunk_size and the benchmark's pinned_staging are carved at the same
    // granularity; host_chunk_mib decouples them when that is not wanted.
    const std::size_t block_size = opts.host_block_bytes();
    cucascade::memory::reservation_manager_configurator builder;
    builder
      .set_number_of_gpus(1)
      // Without an explicit limit the GPU pool is not sized for a benchmark that
      // allocates device buffers (a raw-throughput one never notices; one that
      // decodes into device memory OOMs immediately).
      .set_usage_limit_ratio_per_gpu(opts.gpu_usage_ratio)
      .set_reservation_fraction_per_gpu(0.9)
      .use_gpu_id_as_host_id()
      .set_per_numa_region_capacity(host_region_capacity_v)
      .set_reservation_fraction_per_numa_region(0.9)
      .set_host_pool_features(block_size, host_pool_size_v, opts.host_initial_pools);
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
  }

  ~engine()
  {
    if (_io_ctx) { _io_ctx->shutdown(); }
  }

  engine(engine const&)            = delete;
  engine& operator=(engine const&) = delete;

  [[nodiscard]] io::ioctx& io_ctx() const noexcept { return *_io_ctx; }
  [[nodiscard]] std::shared_ptr<io::ioctx> io_ctx_ptr() const noexcept { return _io_ctx; }

  /// The reservation manager backing the ioctx — a benchmark that stages reads
  /// through the prefetching cache needs it to build one.
  [[nodiscard]] memory::sirius_memory_reservation_manager& mgr() const noexcept { return *_mgr; }
  [[nodiscard]] scan_manager::scan_manager_config const& config() const noexcept { return _cfg; }

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

    cfg.rest.max_connections = opts.max_nconnection;
    cfg.rest_n_reactors      = opts.n_reactors;

    cfg.cache.mode = io::cache::cache_mode::none;
    cfg.apply_cache_mode();
    return cfg;
  }

  std::unique_ptr<memory::sirius_memory_reservation_manager> _mgr;
  scan_manager::scan_manager_config _cfg;
  std::unique_ptr<io::io_context_registry> _registry;
  std::shared_ptr<io::ioctx> _io_ctx;
};

// ---------------------------------------------------------------------------
// bucket listing
// ---------------------------------------------------------------------------

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

struct s3_file {
  std::string path;
  std::size_t size_bytes{0};
};

/// List up to @p n_files parquet objects under a single s3:// prefix.
inline std::vector<s3_file> list_prefix(engine& eng,
                                        std::string_view bucket_uri,
                                        std::size_t n_files)
{
  auto const [bucket, prefix] = split_s3_uri(bucket_uri);

  std::vector<s3_file> files;
  eng.rest().list_objects_paged(
    bucket, prefix, 1000, [&](io::rest::s3::list_objects_v2_page const& page) {
      for (auto const& e : page.entries) {
        if (e.size == 0) { continue; }
        if (!std::string_view{e.key}.ends_with(".parquet")) { continue; }
        files.push_back({"s3://" + bucket + "/" + e.key, static_cast<std::size_t>(e.size)});
        if (files.size() >= n_files) { return false; }
      }
      return true;
    });

  if (files.empty()) {
    throw std::runtime_error("no .parquet objects found under " + std::string{bucket_uri});
  }
  return files;
}

/// Collect @p n_files total across all @p prefixes (divided as evenly as possible),
/// then interleave them in round-robin order so datasource opens and reads
/// alternate between prefixes.
inline std::vector<s3_file> get_files_from_prefixes(engine& eng,
                                                    std::vector<std::string> const& prefixes,
                                                    std::size_t n_files)
{
  if (prefixes.empty()) { throw std::runtime_error("no --bucket specified"); }

  const std::size_t n         = prefixes.size();
  const std::size_t base      = n_files / n;
  const std::size_t remainder = n_files % n;

  // Collect per-prefix file lists.
  std::vector<std::vector<s3_file>> per_prefix;
  per_prefix.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t want = base + (i < remainder ? 1 : 0);
    per_prefix.push_back(list_prefix(eng, prefixes[i], want));
  }

  // Round-robin interleave: round 0 takes index 0 from each prefix,
  // round 1 takes index 1, etc.
  std::vector<s3_file> result;
  result.reserve(n_files);
  for (std::size_t round = 0;; ++round) {
    bool any = false;
    for (auto const& files : per_prefix) {
      if (round < files.size()) {
        result.push_back(files[round]);
        any = true;
      }
    }
    if (!any) { break; }
  }
  return result;
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
