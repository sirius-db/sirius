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

// range_prefetch_benchmark — prefetch-vs-direct with NO parquet decoding.
//
// Every arm reads the SAME synthetic byte ranges out of the SAME files through
// the SAME function (read_ranges_to_device): allocate a device buffer per
// range, issue device_read_async for every range, block on all futures,
// synchronize the stream, then sleep `decode_ms` to stand in for decode work.
// The only difference between arms is whether the prefetching cache was
// populated before that function ran.
//
// Ranges are 1 MiB-aligned and sized in whole MiB, so the cache's outward
// align_and_coalesce is a no-op and read amplification is exactly 1.0.  The
// measured amplification is printed so this can be confirmed.
//
// Arms:
//   A  single file, no cache
//   B  single file, fadvise + prefetch, then the same read
//   C  n files, read each in turn, no cache
//   D  n files, sliding-window readahead by a dedicated prefetch thread
//
// Every arm builds its own uring ioctx (and, for the prefetch arms, its own
// cache) so no state leaks across arms.

#include "io/cache/config.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <glob.h>
#include <log/logging.hpp>
#include <log/spdlog_owning_sink.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <semaphore>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t MIB                  = 1ULL << 20;
constexpr std::size_t HOST_REGION_CAPACITY = 64ULL << 30;
constexpr double RESERVATION_FRACTION      = 0.9;
constexpr std::size_t HOST_BLOCK_SIZE      = MIB;
constexpr std::size_t HOST_POOL_SIZE       = 1024;
constexpr std::uint32_t BOUNCE_POOL_SLABS  = 20;
constexpr std::uint64_t RANGE_SEED         = 0x5EEDULL;

using clock_type = std::chrono::steady_clock;

double now_ms(clock_type::time_point t0)
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

/// Uring stack for one arm.  Member order matters: the ioctx is destroyed
/// first so its reactors stop touching the bounce resource below it.
struct io_stack {
  std::unique_ptr<cucascade::memory::numa_region_pinned_host_memory_resource> upstream;
  std::unique_ptr<cucascade::memory::fixed_size_host_memory_resource> bounce_mr;
  std::shared_ptr<sirius::io::uring::uring_ioctx> io_ctx;
};

struct read_result {
  std::size_t bytes{0};
  double read_ms{0};
  double sync_ms{0};
  double wait_ms{0};
  double total_ms{0};

  void accumulate(read_result const& o)
  {
    bytes += o.bytes;
    read_ms += o.read_ms;
    sync_ms += o.sync_ms;
    wait_ms += o.wait_ms;
    total_ms += o.total_ms;
  }
};

/**
 * @brief The one and only read path in this benchmark.
 *
 * Allocates one device buffer per range, issues @c device_read_async for all of
 * them, blocks on every future, synchronizes @p stream, then sleeps
 * @p decode_wait to stand in for decode work.  Buffers stay alive until after
 * the synchronize.  Each phase is timed separately.
 */
read_result read_ranges_to_device(sirius::io::sirius_datasource& ds,
                                  std::span<const cudf::io::text::byte_range_info> ranges,
                                  rmm::device_async_resource_ref mr,
                                  rmm::cuda_stream_view stream,
                                  std::chrono::milliseconds decode_wait)
{
  read_result r;
  auto const t_all = clock_type::now();

  std::vector<rmm::device_buffer> buffers;
  buffers.reserve(ranges.size());
  for (auto const& range : ranges) {
    buffers.emplace_back(static_cast<std::size_t>(range.size()), stream, mr);
  }

  auto const t_read = clock_type::now();
  std::vector<std::future<std::size_t>> futures;
  futures.reserve(ranges.size());
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    futures.push_back(ds.device_read_async(static_cast<std::size_t>(ranges[i].offset()),
                                           static_cast<std::size_t>(ranges[i].size()),
                                           static_cast<std::uint8_t*>(buffers[i].data()),
                                           stream));
  }
  for (auto& f : futures) {
    r.bytes += f.get();
  }
  r.read_ms = now_ms(t_read);

  auto const t_sync = clock_type::now();
  stream.synchronize();
  r.sync_ms = now_ms(t_sync);

  buffers.clear();

  auto const t_wait = clock_type::now();
  std::this_thread::sleep_for(decode_wait);
  r.wait_ms = now_ms(t_wait);

  r.total_ms = now_ms(t_all);
  return r;
}

/// Fixed-seed, 1 MiB-aligned, whole-MiB, non-overlapping ranges totalling
/// roughly @p target_bytes.  Candidates are drawn, sorted by offset, and any
/// that would overlap its predecessor is skipped.
std::vector<cudf::io::text::byte_range_info> make_ranges(std::size_t file_size,
                                                         std::size_t target_bytes,
                                                         std::uint64_t seed)
{
  constexpr std::size_t max_size = 8 * MIB;
  std::vector<cudf::io::text::byte_range_info> out;
  if (file_size <= max_size) { return out; }

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> off_dist(0, (file_size - max_size) / MIB);
  std::uniform_int_distribution<std::size_t> size_dist(2, 8);

  std::size_t const n_candidates = 4 * (target_bytes / (5 * MIB) + 1);
  std::vector<std::pair<std::size_t, std::size_t>> candidates;
  candidates.reserve(n_candidates);
  for (std::size_t i = 0; i < n_candidates; ++i) {
    candidates.emplace_back(off_dist(rng) * MIB, size_dist(rng) * MIB);
  }
  std::sort(candidates.begin(), candidates.end());

  std::size_t total    = 0;
  std::size_t prev_end = 0;
  bool first           = true;
  for (auto const& [offset, size] : candidates) {
    if (total >= target_bytes) { break; }
    if (!first && offset < prev_end) { continue; }
    out.emplace_back(static_cast<std::int64_t>(offset), static_cast<std::int64_t>(size));
    prev_end = offset + size;
    total += size;
    first = false;
  }
  return out;
}

/// One file's datasource plus the ranges every arm reads from it.
struct file_prep {
  std::string path;
  std::unique_ptr<sirius::io::sirius_datasource> ds;
  std::vector<cudf::io::text::byte_range_info> ranges;
  std::size_t range_bytes{0};
  std::size_t aligned_bytes{0};
};

std::vector<std::string> list_parts(std::string const& dir, std::size_t limit)
{
  std::string const pattern = dir + "/part.*.parquet";
  std::vector<std::string> paths;
  glob_t g{};
  if (::glob(pattern.c_str(), GLOB_TILDE, nullptr, &g) == 0) {
    paths.reserve(g.gl_pathc);
    for (std::size_t i = 0; i < g.gl_pathc; ++i) {
      paths.emplace_back(g.gl_pathv[i]);
    }
  }
  ::globfree(&g);
  std::sort(paths.begin(), paths.end());
  if (limit > 0 && paths.size() > limit) { paths.resize(limit); }
  return paths;
}

io_stack make_io_stack(std::size_t n_reactors)
{
  constexpr std::size_t chunks_per_slab =
    cucascade::memory::fixed_size_host_memory_resource::default_pool_size;
  constexpr std::size_t pool_capacity =
    static_cast<std::size_t>(BOUNCE_POOL_SLABS) * chunks_per_slab * HOST_BLOCK_SIZE;

  io_stack stack;
  stack.upstream =
    std::make_unique<cucascade::memory::numa_region_pinned_host_memory_resource>(0, true);
  stack.bounce_mr =
    std::make_unique<cucascade::memory::fixed_size_host_memory_resource>(0,
                                                                         *stack.upstream,
                                                                         pool_capacity,
                                                                         pool_capacity,
                                                                         HOST_BLOCK_SIZE,
                                                                         chunks_per_slab,
                                                                         BOUNCE_POOL_SLABS);

  auto ctx = std::make_shared<sirius::io::uring::uring_reactor::reactor_context>(
    sirius::io::uring::uring_reactor::reactor_config_type{}, stack.bounce_mr.get());
  stack.io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(n_reactors, std::move(ctx));
  stack.io_ctx->start();
  return stack;
}

file_prep prep_file(sirius::io::ioctx& io_ctx, std::string const& path, std::size_t target_bytes)
{
  file_prep prep;
  prep.path   = path;
  prep.ds     = io_ctx.open_datasource(path);
  prep.ranges = make_ranges(prep.ds->size(), target_bytes, RANGE_SEED);
  for (auto const& r : prep.ranges) {
    prep.range_bytes += static_cast<std::size_t>(r.size());
  }
  for (auto const& r : io_ctx.align_and_coalesce(prep.ranges, HOST_BLOCK_SIZE)) {
    prep.aligned_bytes += static_cast<std::size_t>(r.size());
  }
  return prep;
}

struct arm_result {
  std::string name;
  double wall_ms{0};
  std::size_t bytes{0};
  read_result reads;
  double prefetch_ms{0};
  bool has_prefetch{false};
  std::string cache_summary;
};

void print_header()
{
  std::cout << std::left << std::setw(26) << "arm" << std::right << std::setw(11) << "wall ms"
            << std::setw(12) << "MiB" << std::setw(9) << "GB/s" << std::setw(11) << "prefetch"
            << std::setw(10) << "read" << std::setw(9) << "sync" << std::setw(9) << "wait"
            << "\n"
            << std::string(97, '-') << "\n";
}

void print_row(arm_result const& r)
{
  double const gbps =
    r.wall_ms > 0 ? static_cast<double>(r.bytes) / (r.wall_ms / 1000.0) / 1e9 : 0.0;
  std::cout << std::left << std::setw(26) << r.name << std::right << std::fixed
            << std::setprecision(1) << std::setw(11) << r.wall_ms << std::setw(12)
            << static_cast<double>(r.bytes) / (1024.0 * 1024.0) << std::setprecision(2)
            << std::setw(9) << gbps << std::setprecision(1) << std::setw(11)
            << (r.has_prefetch ? r.prefetch_ms : 0.0) << std::setw(10) << r.reads.read_ms
            << std::setw(9) << r.reads.sync_ms << std::setw(9) << r.reads.wait_ms << "\n";
}

/// Pull an integer field out of a @c prefetching_cache::summary string, e.g.
/// @c hits from @c "global[reads=1 hits=512 ...]".  Returns -1 when absent.
std::int64_t summary_field(std::string const& summary, std::string const& key)
{
  auto const global = summary.find("global[");
  if (global == std::string::npos) { return -1; }
  auto const end = summary.find(']', global);
  auto const pos = summary.find(key + "=", global);
  if (pos == std::string::npos || pos > end) { return -1; }
  return static_cast<std::int64_t>(
    std::strtoll(summary.c_str() + pos + key.size() + 1, nullptr, 10));
}

void report_cache(arm_result const& r)
{
  std::cout << "\n" << r.name << " cache " << r.cache_summary << "\n";
  auto const hits  = summary_field(r.cache_summary, "hits");
  auto const h2d   = summary_field(r.cache_summary, "h2d");
  auto const miss  = summary_field(r.cache_summary, "miss");
  auto const total = hits + miss;
  if (hits < 0 || miss < 0 || h2d < 0) {
    std::cout << "  (could not parse cache counters)\n";
    return;
  }
  double const cached_bytes = static_cast<double>(hits) * static_cast<double>(HOST_BLOCK_SIZE);
  std::cout << "  chunks: hits=" << hits << " miss=" << miss << " h2d=" << h2d
            << "  hit_rate=" << std::fixed << std::setprecision(1)
            << (total > 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(total) : 0.0)
            << "%\n"
            << "  bytes served from cache: " << std::setprecision(1)
            << cached_bytes / (1024.0 * 1024.0) << " MiB of "
            << static_cast<double>(r.bytes) / (1024.0 * 1024.0) << " MiB ("
            << (r.bytes > 0 ? 100.0 * cached_bytes / static_cast<double>(r.bytes) : 0.0) << "%)\n";
  if (h2d != 0) {
    std::cout << "  *** WARNING: h2d=" << h2d
              << " is NON-ZERO — chunks were loaded DURING the read rather than served from "
                 "cache; the prefetch-vs-direct comparison is NOT apples-to-apples ***\n";
  }
}

/// Wall time covered by at least one IO, i.e. the union of the per-file
/// [start, end) intervals.
double union_span_ms(std::vector<double> const& starts, std::vector<double> const& ends)
{
  std::vector<std::pair<double, double>> iv;
  iv.reserve(starts.size());
  for (std::size_t i = 0; i < starts.size(); ++i) {
    iv.emplace_back(starts[i], ends[i]);
  }
  std::sort(iv.begin(), iv.end());
  double total = 0;
  double cur_s = 0;
  double cur_e = 0;
  bool open    = false;
  for (auto const& [s, e] : iv) {
    if (!open) {
      cur_s = s;
      cur_e = e;
      open  = true;
      continue;
    }
    if (s > cur_e) {
      total += cur_e - cur_s;
      cur_s = s;
      cur_e = e;
    } else {
      cur_e = std::max(cur_e, e);
    }
  }
  if (open) { total += cur_e - cur_s; }
  return total;
}

double median_of(std::vector<double> v)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  std::size_t const mid = v.size() / 2;
  return v.size() % 2 == 1 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 2 || argc > 7) {
    std::cerr << "usage: " << argv[0]
              << " <lineitem-dir> [n_files] [n_reactors] [window] [decode_ms] "
                 "[bytes_per_file_mib]\n";
    return 1;
  }

  std::string const dir  = argv[1];
  std::size_t n_files    = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 4;
  std::size_t n_reactors = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 1;
  std::size_t window     = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 3;
  std::size_t decode_ms  = argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 50;
  std::size_t bytes_per_file =
    (argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 512) * MIB;
  if (n_files == 0 || n_reactors == 0 || window == 0 || bytes_per_file == 0) {
    std::cerr << "n_files, n_reactors, window and bytes_per_file_mib must all be > 0\n";
    return 1;
  }
  std::chrono::milliseconds const decode_wait{decode_ms};

  auto paths = list_parts(dir, n_files);
  if (paths.empty()) {
    std::cerr << "no files matched " << dir << "/part.*.parquet\n";
    return 1;
  }
  n_files = paths.size();

  auto log_sink = sirius::log::make_spdlog_owning_sink({"log", std::nullopt});
  log_sink->set_level(sirius::log::level::info);
  sirius::log::set_sink(std::move(log_sink));

  cudaFree(nullptr);

  // Pre-allocate every pinned pool the deepest arm can ask for -- window files
  // staged plus one being consumed -- so no pool ever grows mid-measurement.
  std::size_t const pool_bytes = HOST_POOL_SIZE * HOST_BLOCK_SIZE;
  std::size_t const host_initial_pools =
    ((window + 1) * bytes_per_file + pool_bytes - 1) / pool_bytes;

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_reservation_fraction_per_gpu(RESERVATION_FRACTION)
    .use_gpu_id_as_host_id()
    .set_per_numa_region_capacity(HOST_REGION_CAPACITY)
    .set_reservation_fraction_per_numa_region(RESERVATION_FRACTION)
    .set_host_pool_features(HOST_BLOCK_SIZE, HOST_POOL_SIZE, host_initial_pools);
  auto mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::device_async_resource_ref device_mr{async_mr};
  rmm::mr::set_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible>{device_mr});

  rmm::cuda_stream stream;

  sirius::io::cache::config cache_cfg;
  cache_cfg.mode                            = sirius::io::cache::cache_mode::sirius;
  cache_cfg.eviction                        = sirius::io::cache::eviction_policy::lru;
  cache_cfg.min_prefetching_budget_fraction = 0.9;
  cache_cfg.eviction_threshold_fraction     = 0.9;
  cache_cfg.apply_mode();

  std::cout << "dir           : " << dir << "\n"
            << "files         : " << n_files << "\n"
            << "reactors      : " << n_reactors << "\n"
            << "window        : " << window << "\n"
            << "decode wait   : " << decode_ms << " ms\n"
            << "bytes/file    : " << (bytes_per_file >> 20) << " MiB (target)\n"
            << "host region   : " << (HOST_REGION_CAPACITY >> 30) << " GiB, block "
            << (HOST_BLOCK_SIZE >> 20) << " MiB\n"
            << "host pool     : " << host_initial_pools << " pools x " << (pool_bytes >> 20)
            << " MiB pre-allocated (no runtime growth)\n"
            << "bounce pool   : " << (BOUNCE_POOL_SLABS * 128 * HOST_BLOCK_SIZE >> 20)
            << " MiB pre-allocated\n\n";

  std::vector<arm_result> results;
  std::vector<double> per_file_c;
  std::vector<double> per_file_d;
  std::size_t d_ready         = 0;
  std::size_t d_waited        = 0;
  std::size_t d_peak_inflight = 0;
  std::vector<double> d_io_start(n_files, 0.0);
  std::vector<double> d_io_end(n_files, 0.0);
  std::vector<double> d_parse_start(n_files, 0.0);
  std::vector<double> d_parse_end(n_files, 0.0);
  bool b_issued = false;
  bool b_ok     = false;

  {
    auto stack                 = make_io_stack(n_reactors);
    auto prep                  = prep_file(*stack.io_ctx, paths.front(), bytes_per_file);
    double const amplification = prep.range_bytes > 0 ? static_cast<double>(prep.aligned_bytes) /
                                                          static_cast<double>(prep.range_bytes)
                                                      : 0.0;
    std::cout << "ranges/file   : " << prep.ranges.size() << " totalling " << std::fixed
              << std::setprecision(1) << static_cast<double>(prep.range_bytes) / (1024.0 * 1024.0)
              << " MiB\n"
              << "amplification : " << std::setprecision(3) << amplification << "\n";
    if (amplification < 0.999 || amplification > 1.001) {
      std::cout << "*** WARNING: read amplification is NOT 1.000 — alignment assumption is "
                   "broken, the arms are not comparable ***\n";
    }
    std::cout << "\n";

    auto t0 = clock_type::now();
    auto rr = read_ranges_to_device(*prep.ds, prep.ranges, device_mr, stream.view(), decode_wait);
    results.push_back({"A baseline 1 file", now_ms(t0), prep.range_bytes, rr, 0.0, false, {}});
  }

  {
    auto stack = make_io_stack(n_reactors);
    stack.io_ctx->initialize_cache(*mgr, cache_cfg, nullptr);
    if (!stack.io_ctx->uses_prefetching_cache()) {
      std::cerr << "FATAL: prefetching cache did not come up for arm B\n";
      return 2;
    }
    auto prep = prep_file(*stack.io_ctx, paths.front(), bytes_per_file);
    prep.ds->fadvise(prep.ranges, 0);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto t0 = clock_type::now();
    std::promise<bool> p;
    auto fut                 = p.get_future();
    b_issued                 = prep.ds->prefetch_async([&p](bool ok) noexcept { p.set_value(ok); });
    b_ok                     = fut.get();
    double const prefetch_ms = now_ms(t0);
    auto rr = read_ranges_to_device(*prep.ds, prep.ranges, device_mr, stream.view(), decode_wait);
    results.push_back({"B prefetch 1 file",
                       now_ms(t0),
                       prep.range_bytes,
                       rr,
                       prefetch_ms,
                       true,
                       stack.io_ctx->cache()->summary()});
  }

  {
    auto stack = make_io_stack(n_reactors);
    std::vector<file_prep> preps;
    preps.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, bytes_per_file));
    }
    read_result agg;
    std::size_t bytes = 0;
    auto t0           = clock_type::now();
    for (auto& prep : preps) {
      auto f0 = clock_type::now();
      agg.accumulate(
        read_ranges_to_device(*prep.ds, prep.ranges, device_mr, stream.view(), decode_wait));
      per_file_c.push_back(now_ms(f0));
    }
    double const ms = now_ms(t0);
    for (auto const& f : preps) {
      bytes += f.range_bytes;
    }
    results.push_back({"C baseline n files", ms, bytes, agg, 0.0, false, {}});
  }

  {
    auto stack = make_io_stack(n_reactors);
    stack.io_ctx->initialize_cache(*mgr, cache_cfg, nullptr);
    if (!stack.io_ctx->uses_prefetching_cache()) {
      std::cerr << "FATAL: prefetching cache did not come up for arm D\n";
      return 2;
    }
    std::vector<file_prep> preps;
    preps.reserve(n_files);
    for (auto const& p : paths) {
      preps.push_back(prep_file(*stack.io_ctx, p, bytes_per_file));
    }
    for (auto& f : preps) {
      f.ds->fadvise(f.ranges, 0);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::vector<std::promise<bool>> promises(n_files);
    std::vector<std::future<bool>> futures;
    futures.reserve(n_files);
    for (auto& pr : promises) {
      futures.push_back(pr.get_future());
    }

    std::atomic<std::size_t> parse_pos{0};
    std::counting_semaphore<> window_slots{static_cast<std::ptrdiff_t>(window)};
    std::size_t issued_ok = 0;

    read_result agg;
    std::size_t bytes = 0;
    auto t0           = clock_type::now();

    std::thread prefetcher([&] {
      for (std::size_t k = 0; k < n_files; ++k) {
        window_slots.acquire();
        std::size_t const inflight = k + 1 - parse_pos.load(std::memory_order_acquire);
        d_peak_inflight            = std::max(d_peak_inflight, inflight);
        d_io_start[k]              = now_ms(t0);
        if (preps[k].ds->prefetch_async([&promises, &d_io_end, t0, k](bool ok) noexcept {
              d_io_end[k] = now_ms(t0);
              promises[k].set_value(ok);
            })) {
          ++issued_ok;
        }
      }
    });

    for (std::size_t i = 0; i < n_files; ++i) {
      if (futures[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        ++d_ready;
      } else {
        ++d_waited;
      }
      futures[i].get();
      d_parse_start[i] = now_ms(t0);
      auto f0          = clock_type::now();
      agg.accumulate(read_ranges_to_device(
        *preps[i].ds, preps[i].ranges, device_mr, stream.view(), decode_wait));
      per_file_d.push_back(now_ms(f0));
      d_parse_end[i] = now_ms(t0);
      parse_pos.store(i + 1, std::memory_order_release);
      window_slots.release();
    }
    prefetcher.join();
    double const ms = now_ms(t0);
    for (auto const& f : preps) {
      bytes += f.range_bytes;
    }
    results.push_back({"D prefetch n files",
                       ms,
                       bytes,
                       agg,
                       union_span_ms(d_io_start, d_io_end),
                       true,
                       stack.io_ctx->cache()->summary()});
    std::cout << "arm D: prefetch_async issued IO for " << issued_ok << "/" << n_files
              << " files\n\n";
  }

  print_header();
  for (auto const& r : results) {
    print_row(r);
  }

  std::cout << "(prefetch column: arm B = blocking prefetch wait, arm D = union of the "
               "per-file prefetch intervals)\n";

  std::cout << "\narm B: prefetch_async issued=" << std::boolalpha << b_issued
            << " completed_ok=" << b_ok << " prefetch_ms=" << std::fixed << std::setprecision(1)
            << results[1].prefetch_ms << "\n";
  std::cout << "arm A read_ms=" << results[0].reads.read_ms
            << " vs arm B read_ms=" << results[1].reads.read_ms << "\n";
  std::cout << "arm C read_ms=" << results[2].reads.read_ms
            << " vs arm D read_ms=" << results[3].reads.read_ms << "\n";

  report_cache(results[1]);
  report_cache(results[3]);

  std::cout << "\narm C per-file ms:";
  for (auto ms : per_file_c) {
    std::cout << " " << std::fixed << std::setprecision(1) << ms;
  }
  std::cout << "\narm D per-file ms:";
  for (auto ms : per_file_d) {
    std::cout << " " << std::fixed << std::setprecision(1) << ms;
  }
  std::cout << "\n";

  std::cout << "\narm D per-file phases (ms relative to arm t0):\n"
            << std::right << std::setw(6) << "file" << std::setw(11) << "io_start" << std::setw(10)
            << "io_end" << std::setw(10) << "io_ms" << std::setw(14) << "parse_start"
            << std::setw(11) << "parse_end" << std::setw(11) << "parse_ms"
            << "\n"
            << std::string(73, '-') << "\n";
  double total_io_ms    = 0;
  double total_parse_ms = 0;
  for (std::size_t i = 0; i < n_files; ++i) {
    double const io    = d_io_end[i] - d_io_start[i];
    double const parse = d_parse_end[i] - d_parse_start[i];
    total_io_ms += io;
    total_parse_ms += parse;
    std::cout << std::right << std::fixed << std::setprecision(1) << std::setw(6) << i
              << std::setw(11) << d_io_start[i] << std::setw(10) << d_io_end[i] << std::setw(10)
              << io << std::setw(14) << d_parse_start[i] << std::setw(11) << d_parse_end[i]
              << std::setw(11) << parse << "\n";
  }

  double const wall_ms  = results.back().wall_ms;
  double const io_union = union_span_ms(d_io_start, d_io_end);

  double const first_io_start = *std::min_element(d_io_start.begin(), d_io_start.end());
  double const last_io_end    = *std::max_element(d_io_end.begin(), d_io_end.end());
  double const prefetch_span  = last_io_end - first_io_start;
  std::size_t all_bytes       = 0;
  for (auto const& r : results) {
    if (r.name.rfind("D ", 0) == 0) { all_bytes = r.bytes; }
  }
  std::cout << "\narm D prefetch span (first io_start -> last io_end, ALL " << n_files
            << " files staged to pinned host): " << std::fixed << std::setprecision(1)
            << prefetch_span << " ms for " << static_cast<double>(all_bytes) / (1024.0 * 1024.0)
            << " MiB = " << std::setprecision(2)
            << (prefetch_span > 0 ? static_cast<double>(all_bytes) / (prefetch_span / 1000.0) / 1e9
                                  : 0.0)
            << " GB/s   [gated by window=" << window << ", so this includes stall time]\n";
  std::cout << "  vs arm C wall (disk->device, all files, no cache): " << std::setprecision(1)
            << results[2].wall_ms << " ms of which read_ms=" << results[2].reads.read_ms << "\n";
  double const overlap_ms    = total_io_ms + total_parse_ms - wall_ms;
  double const hideable      = std::min(total_io_ms, total_parse_ms);
  double const hidden_pct    = hideable > 0 ? 100.0 * overlap_ms / hideable : 0.0;
  double const true_overlap  = io_union + total_parse_ms - wall_ms;
  double const true_hideable = std::min(io_union, total_parse_ms);
  double const true_pct      = true_hideable > 0 ? 100.0 * true_overlap / true_hideable : 0.0;
  std::cout << "\narm D aggregates: total_io_ms=" << std::fixed << std::setprecision(1)
            << total_io_ms << " total_parse_ms=" << total_parse_ms << " wall_ms=" << wall_ms
            << " overlap_ms=" << overlap_ms << " (" << hidden_pct << "% of hideable " << hideable
            << " ms)\n";
  std::cout << "arm D aggregates (IO interval union, valid when prefetches run concurrently): "
            << "io_union_ms=" << io_union << " overlap_ms=" << true_overlap << " (" << true_pct
            << "% of hideable " << true_hideable << " ms)\n";

  std::cout << "\narm D handoffs (lead_ms = parse_end[i] - io_end[i+1]):\n";
  std::vector<double> leads;
  std::size_t overlapped = 0;
  std::size_t waited     = 0;
  for (std::size_t i = 0; i + 1 < n_files; ++i) {
    double const lead = d_parse_end[i] - d_io_end[i + 1];
    leads.push_back(lead);
    if (lead > 0) {
      ++overlapped;
    } else {
      ++waited;
    }
    std::cout << "  " << i << " -> " << (i + 1) << " : lead_ms = " << std::showpos << std::fixed
              << std::setprecision(1) << lead << std::noshowpos
              << (lead > 0 ? "   (overlapped)" : "   (parser waited)") << "\n";
  }
  if (!leads.empty()) {
    std::cout << "  summary: overlapped=" << overlapped << " waited=" << waited
              << " min=" << std::fixed << std::setprecision(1)
              << *std::min_element(leads.begin(), leads.end()) << " median=" << median_of(leads)
              << " max=" << *std::max_element(leads.begin(), leads.end()) << "\n";
  }
  std::cout << "  cross-check: already_ready=" << d_ready << " (expected " << overlapped
            << ") had_to_wait=" << d_waited << " (expected " << (n_files - overlapped) << ")"
            << (d_ready == overlapped ? "  OK" : "  MISMATCH — measurement bug") << "\n";

  std::cout << "\narm D readahead: window=" << window << " peak_files_in_flight=" << d_peak_inflight
            << " already_ready=" << d_ready << " had_to_wait=" << d_waited << "\n";

  return 0;
}
