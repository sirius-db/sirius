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

// prefetch_hybrid_scan_benchmark — compare baseline read_parquet throughput
// against a direct-to-device hybrid scan.
//
// Usage:
//   ./prefetch_hybrid_scan_benchmark --dir DIR [--n_files N] --mode <b|p> [--nthreads N]
//
// Modes:
//   b  baseline    — enqueue all read_parquet calls directly to the thread pool
//   p  hybrid scan — fadvise all files so the cache stages them, then walk the
//                    files on the main thread: for each one allocate a device
//                    buffer per column-chunk byte range, device_read_async all
//                    of them, and hand the buffers plus their stream to the
//                    thread pool, which materializes the table through
//                    hybrid_scan_reader.  The main thread never waits for the
//                    decode, so IO of file N+1 overlaps decode of file N.

#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/config.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/stream_pool.hpp>
#include <glob.h>
#include <log/logging.hpp>
#include <log/spdlog_owning_sink.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <latch>
#include <string>
#include <utility>
#include <vector>

namespace {

using clock_type    = std::chrono::high_resolution_clock;
using stream_pool_t = cucascade::memory::exclusive_stream_pool;
using acquire_pol   = stream_pool_t::stream_acquire_policy;

double ms_since(clock_type::time_point t0) noexcept
{
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

std::vector<std::string> const COLUMNS = {
  "l_orderkey",
  "l_extendedprice",
  "l_discount",
  "l_shipdate",
};

// ---- file discovery --------------------------------------------------------

struct file_info {
  std::string path;
  std::unique_ptr<sirius::io::sirius_datasource> ds;
  std::vector<cudf::io::text::byte_range_info> ranges;
  std::size_t range_bytes{0};
  // Footer metadata and row groups cached during discovery so the decode task
  // can rebuild a hybrid_scan_reader without a second footer fetch + Thrift
  // parse (mirrors src/op/scan/parquet_gpu_ingestible.cpp).
  cudf::io::parquet::FileMetaData metadata;
  std::vector<cudf::size_type> row_groups;
};

std::vector<std::string> glob_parquet_files(std::string const& dir, std::size_t limit)
{
  static constexpr std::array<char const*, 2> kPatterns = {"part.*.parquet", "lineitem.*.parquet"};
  std::vector<std::string> paths;
  for (auto const* pat : kPatterns) {
    std::string const pattern = dir + "/" + pat;
    glob_t g{};
    if (::glob(pattern.c_str(), GLOB_TILDE, nullptr, &g) == 0) {
      for (std::size_t i = 0; i < g.gl_pathc; ++i) {
        paths.emplace_back(g.gl_pathv[i]);
      }
    }
    ::globfree(&g);
    if (!paths.empty()) break;
  }
  std::sort(paths.begin(), paths.end());
  if (limit > 0 && paths.size() > limit) { paths.resize(limit); }
  return paths;
}

// ---- parse_parquet ---------------------------------------------------------

// Parse a single parquet file using the sirius_datasource already backed by
// the uring io_ctx.  source_info takes a raw (non-owning) datasource* so no
// shim or ownership transfer is needed.  Stream is synchronised before
// returning so the caller may safely discard the table immediately.
std::unique_ptr<cudf::table> parse_parquet(sirius::io::sirius_datasource& ds,
                                           rmm::cuda_stream_view stream)
{
  auto opts = cudf::io::parquet_reader_options::builder(cudf::io::source_info{&ds})
                .column_names(COLUMNS)
                .build();
  auto result = cudf::io::read_parquet(opts, stream);
  cudaStreamSynchronize(stream.value());
  return std::move(result.tbl);
}

// ---- baseline --------------------------------------------------------------

void run_baseline(std::vector<file_info>& files,
                  sirius::exec::scoped_dispatcher& disp,
                  stream_pool_t& streams,
                  std::size_t total_bytes)
{
  std::atomic<std::size_t> total_rows{0};
  std::latch done(static_cast<std::ptrdiff_t>(files.size()));

  auto t0 = clock_type::now();
  for (std::size_t k = 0; k < files.size(); ++k) {
    disp.enqueue([k, &files, &streams, &total_rows, &done] {
      auto stream = streams.acquire_stream(acquire_pol::GROW);
      auto tbl    = parse_parquet(*files[k].ds, stream);
      total_rows.fetch_add(static_cast<std::size_t>(tbl->num_rows()), std::memory_order_relaxed);
      done.count_down();
    });
  }
  done.wait();
  double const elapsed = ms_since(t0);

  std::cout << "\n=== baseline results ===\n"
            << "  wall      : " << std::fixed << std::setprecision(1) << elapsed << " ms\n"
            << "  throughput: " << std::setprecision(2)
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / (elapsed / 1000.0)
            << " GiB/s\n"
            << "  rows      : " << total_rows.load() << "\n";
}

// ---- hybrid scan -----------------------------------------------------------

void run_hybrid_scan(std::vector<file_info>& files,
                     sirius::exec::scoped_dispatcher& disp,
                     stream_pool_t& streams,
                     sirius::io::uring::uring_ioctx& io_ctx,
                     sirius::memory::sirius_memory_reservation_manager& mgr,
                     std::size_t total_bytes)
{
  sirius::io::cache::config cache_cfg;
  cache_cfg.inflight_io_chunk_budget        = 16384;
  cache_cfg.min_prefetching_budget_fraction = 0.9;
  cache_cfg.eviction_threshold_fraction     = 0.9;
  cache_cfg.dispose_on_idle                 = false;
  io_ctx.initialize_cache(mgr, cache_cfg, nullptr);

  for (auto& fi : files) {
    fi.ds->fadvise(fi.ranges, 0);
  }

  // Block until prepare_loop has allocated staging buffers for every chunk of
  // every file (producer state >= prepared).  There is no prefetch_async call
  // in this mode — the device reads below are what pull the data through — but
  // the reads still land on the cache's staging chunks, so they must exist.
  for (auto& fi : files) {
    if (!fi.ds->wait_prefetch_prepared()) {
      SIRIUS_LOG_WARN("prefetch preparation abandoned for {}", fi.path);
    }
  }

  auto const mr_ref        = cudf::get_current_device_resource_ref();
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

  std::atomic<std::size_t> total_rows{0};
  std::latch done(static_cast<std::ptrdiff_t>(files.size()));

  auto t0 = clock_type::now();
  for (std::size_t k = 0; k < files.size(); ++k) {
    auto& fi = files[k];

    auto stream = streams.acquire_stream(acquire_pol::GROW);

    // One device buffer per column-chunk byte range, allocated on `stream`.
    std::vector<rmm::device_buffer> buffers;
    buffers.reserve(fi.ranges.size());
    for (auto const& range : fi.ranges) {
      buffers.emplace_back(static_cast<std::size_t>(range.size()), stream.get(), mr_ref);
    }

    std::vector<std::future<std::size_t>> futures;
    futures.reserve(fi.ranges.size());
    for (std::size_t i = 0; i < fi.ranges.size(); ++i) {
      futures.push_back(fi.ds->device_read_async(static_cast<std::size_t>(fi.ranges[i].offset()),
                                                 static_cast<std::size_t>(fi.ranges[i].size()),
                                                 static_cast<std::uint8_t*>(buffers[i].data()),
                                                 stream.get()));
    }
    for (auto& f : futures) {
      f.get();
    }

    // Deliberately NO stream synchronize here.  The H2D copies, the buffers'
    // stream-ordered allocation and deallocation, and the decode below all run
    // on this one stream, so the decode is already ordered after the copies.
    // Waiting would serialise IO and decode; instead the buffers and the
    // borrowed_stream move into the task (which keeps both alive until it
    // synchronises) and this thread goes straight on to the next file's IO.
    // `stream` is captured before `buffers` on purpose: captures are destroyed
    // in reverse declaration order, so the buffers release their (stream-
    // ordered) allocations before the borrowed_stream goes back to the pool.
    disp.enqueue([k,
                  &files,
                  &total_rows,
                  &done,
                  mr_ref,
                  stream  = std::move(stream),
                  buffers = std::move(buffers)] {
      auto const& f = files[k];

      std::vector<cudf::device_span<uint8_t const>> spans;
      spans.reserve(buffers.size());
      for (auto const& buf : buffers) {
        spans.emplace_back(static_cast<uint8_t const*>(buf.data()), buf.size());
      }

      auto opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();
      hybrid_scan_reader reader(f.metadata, opts);
      auto result = reader.materialize_all_columns(
        cudf::host_span<cudf::size_type const>(f.row_groups.data(), f.row_groups.size()),
        cudf::host_span<cudf::device_span<uint8_t const> const>(spans.data(), spans.size()),
        opts,
        stream.get(),
        mr_ref);

      // Buffers die when this lambda does, so drain the stream first.
      cudaStreamSynchronize(stream.get().value());
      total_rows.fetch_add(static_cast<std::size_t>(result.tbl->num_rows()),
                           std::memory_order_relaxed);
      done.count_down();
    });
  }

  done.wait();
  double const elapsed = ms_since(t0);

  std::cout << "\n=== hybrid scan results ===\n"
            << "  wall      : " << std::fixed << std::setprecision(1) << elapsed << " ms\n"
            << "  throughput: " << std::setprecision(2)
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / (elapsed / 1000.0)
            << " GiB/s\n"
            << "  rows      : " << total_rows.load() << "\n";
}

}  // namespace

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv)
{
  std::string dir;
  std::size_t n_files = 0;
  std::string mode;  // "b" or "p"
  std::size_t nthreads = 4;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--dir" && i + 1 < argc) {
      dir = argv[++i];
    } else if (arg == "--n_files" && i + 1 < argc) {
      n_files = std::stoull(argv[++i]);
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--nthreads" && i + 1 < argc) {
      nthreads = std::stoull(argv[++i]);
    } else {
      std::cerr << "usage: " << argv[0] << " --dir DIR [--n_files N] --mode <b|p> [--nthreads N]\n";
      return 1;
    }
  }
  if (dir.empty() || (mode != "b" && mode != "p")) {
    std::cerr << "usage: " << argv[0] << " --dir DIR [--n_files N] --mode <b|p> [--nthreads N]\n";
    return 1;
  }

  auto log_sink = sirius::log::make_spdlog_owning_sink({"log", std::nullopt});
  log_sink->set_level(sirius::log::level::warn);
  sirius::log::set_sink(std::move(log_sink));

  cudaFree(nullptr);

  // ---- memory setup --------------------------------------------------------
  // sirius_memory_reservation_manager installs the cucascade GPU allocator as
  // cudf's default device memory resource in its constructor.
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_usage_limit_ratio_per_gpu(0.5)  // GPU pool: 50% of total memory
    .set_reservation_fraction_per_gpu(0.9)
    .use_gpu_id_as_host_id()
    .set_per_numa_region_capacity(20ULL << 30)
    .set_reservation_fraction_per_numa_region(0.9)
    .set_host_pool_features(1ULL << 20,  // chunk_size  = 1 MiB
                            1024,        // pool_size   = blocks per slab (1 GiB each)
                            20);         // initial_pools (14 GiB, well above 4-col staging)

  auto mgr = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());

  // Bounce pool comes from the manager — no separate allocation.
  auto* host_space = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  auto* bounce_mr  = host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();

  // ---- io context ----------------------------------------------------------
  auto ctx = std::make_shared<sirius::io::uring::uring_reactor::reactor_context>(
    sirius::io::uring::uring_reactor::reactor_config_type{
      .bounce_size = bounce_mr->get_block_size(), .max_n_chunks = 1},
    bounce_mr);
  auto io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(1, std::move(ctx));
  io_ctx->start();

  // ---- thread pool + dispatcher --------------------------------------------
  sirius::exec::static_thread_pool pool(static_cast<int>(nthreads), "parse_pool");
  sirius::exec::scoped_dispatcher disp(pool);
  stream_pool_t streams(rmm::cuda_device_id{0}, nthreads);

  // ---- phase 1: discover files ---------------------------------------------
  auto paths = glob_parquet_files(dir, n_files);
  if (paths.empty()) {
    std::cerr << "no parquet files found in " << dir << "\n";
    return 1;
  }

  // opts used only in phase 1 for range discovery; the benchmark modes build
  // their own.
  auto disc_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

  std::cout << "found " << paths.size() << " file(s) in " << dir << "\n"
            << "mode=" << (mode == "b" ? "baseline" : "hybrid_scan") << "  nthreads=" << nthreads
            << "\n\n"
            << "=== phase 1: column-chunk discovery ===\n";

  std::vector<file_info> files;
  files.reserve(paths.size());
  std::size_t total_bytes = 0;

  for (auto const& path : paths) {
    file_info fi;
    fi.path = path;
    fi.ds   = io_ctx->open_datasource(path);

    auto footer = cudf::io::parquet::fetch_footer_to_host(*fi.ds);
    hybrid_scan_reader reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                              disc_opts);

    // Both the row groups and the parsed footer are kept on the file_info so
    // mode `p` can construct its reader straight from them.
    fi.row_groups = reader.all_row_groups(disc_opts);
    fi.metadata   = reader.parquet_metadata();
    fi.ranges     = reader.all_column_chunks_byte_ranges(
      cudf::host_span<cudf::size_type const>(fi.row_groups.data(), fi.row_groups.size()),
      disc_opts);
    for (auto const& r : fi.ranges) {
      fi.range_bytes += static_cast<std::size_t>(r.size());
    }
    total_bytes += fi.range_bytes;

    std::cout << std::filesystem::path(path).filename().string()
              << " { n_ranges=" << fi.ranges.size() << ", total_size=" << std::fixed
              << std::setprecision(2) << static_cast<double>(fi.range_bytes) / (1024.0 * 1024.0)
              << " MiB }\n";

    files.push_back(std::move(fi));
  }

  std::cout << "\ntotal: " << files.size() << " files, " << std::fixed << std::setprecision(2)
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB\n";

  // ---- phase 2: benchmark --------------------------------------------------
  if (mode == "b") {
    run_baseline(files, disp, streams, total_bytes);
  } else {
    run_hybrid_scan(files, disp, streams, *io_ctx, *mgr, total_bytes);
  }

  disp.wait_for_all();
  pool.stop();
  return 0;
}
