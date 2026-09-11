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
//   ./prefetch_hybrid_scan_benchmark --dir DIR --mode <b|p> [options]
//   ./prefetch_hybrid_scan_benchmark --s3 s3://BUCKET/PREFIX --config sirius.yml
//                                    --mode <b|p> [options]
//
// Options:
//   --n_files N         files to read
//   --nthreads N        reader/decode pool threads (and CUDA streams)
//   --host_chunk_mib N  host memory pool block size
//   --chunk_mib N       IO chunk size (REST: target bytes per GET before fusing)
//   --max_connections N in-flight requests per reactor (REST)
//   --n_reactors N      reactor count (REST)
//
// Sources:
//   --dir  local directory, read through io_uring
//   --s3   object-store prefix, read through the REST reactor.  Everything past
//          setup goes through ioctx, so the two differ only in which backend
//          serves the reads.
//
// There is no prefetching cache in either mode: reads go straight to the
// backend, so what is measured is the read path itself and not a cache-hit rate.
//
// Modes -- both enqueue one task per file onto the same pool, so they differ
// only in how a task gets its bytes:
//   b  baseline    — read_parquet(datasource, options)
//   p  hybrid scan — a device buffer per column-chunk range, one batched
//                    device_read_ranges_async for the file, then
//                    hybrid_scan_reader::materialize_all_columns over them.

#include "exec/scoped_dispatcher.hpp"
#include "exec/thread_pool.hpp"
#include "io/cache/config.hpp"
#include "io/sirius_datasource.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "s3_bench_common.hpp"

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
#include <limits>
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
                     std::size_t total_bytes)
{
  auto const mr_ref        = cudf::get_current_device_resource_ref();
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

  std::atomic<std::size_t> total_rows{0};
  std::latch done(static_cast<std::ptrdiff_t>(files.size()));

  auto t0 = clock_type::now();
  // Symmetric with the baseline: one task per file, and the task owns the whole
  // file -- its IO and its decode.  The main thread only hands out work, so
  // files are read concurrently and the backend sees every file's requests at
  // once rather than one file's at a time.
  for (std::size_t k = 0; k < files.size(); ++k) {
    disp.enqueue([k, &files, &streams, &total_rows, &done, mr_ref] {
      auto& f     = files[k];
      auto stream = streams.acquire_stream(acquire_pol::GROW);

      // One device buffer per column-chunk byte range, allocated on `stream`.
      std::vector<rmm::device_buffer> buffers;
      buffers.reserve(f.ranges.size());
      for (auto const& range : f.ranges) {
        buffers.emplace_back(static_cast<std::size_t>(range.size()), stream.get(), mr_ref);
      }

      // One batched request for every column chunk this file needs, not one
      // device_read_async per range: the backend gets the whole batch in a
      // single dispatch and can fuse and order it as it sees fit.
      std::vector<sirius::io::slice> reads;
      reads.reserve(f.ranges.size());
      for (std::size_t i = 0; i < f.ranges.size(); ++i) {
        reads.emplace_back(static_cast<std::size_t>(f.ranges[i].offset()),
                           static_cast<std::size_t>(f.ranges[i].size()),
                           static_cast<std::uint8_t*>(buffers[i].data()));
      }
      f.ds->device_read_ranges_async(reads, stream.get()).get();

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
  std::string s3_uri;
  std::string config_path;
  std::size_t n_files = 0;
  std::string mode;  // "b" or "p"
  std::size_t nthreads        = 4;
  std::size_t host_chunk_mib  = 1;    // host memory pool block size
  std::size_t chunk_mib       = 1;    // IO chunk size
  std::size_t max_connections = 128;  // REST only
  std::size_t n_reactors      = 8;    // REST only

  auto const usage = [&] {
    std::cerr << "usage: " << argv[0]
              << " (--dir DIR | --s3 s3://BUCKET/PREFIX --config YAML) --mode <b|p>\n"
                 "       [--n_files N] [--nthreads N] [--host_chunk_mib N] [--chunk_mib N]\n"
                 "       [--max_connections N] [--n_reactors N]\n";
  };

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--dir" && i + 1 < argc) {
      dir = argv[++i];
    } else if (arg == "--s3" && i + 1 < argc) {
      s3_uri = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--n_files" && i + 1 < argc) {
      n_files = std::stoull(argv[++i]);
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--nthreads" && i + 1 < argc) {
      nthreads = std::stoull(argv[++i]);
    } else if (arg == "--n_reactors" && i + 1 < argc) {
      n_reactors = std::stoull(argv[++i]);
    } else if (arg == "--max_connections" && i + 1 < argc) {
      max_connections = std::stoull(argv[++i]);
    } else if (arg == "--host_chunk_mib" && i + 1 < argc) {
      host_chunk_mib = std::max<std::size_t>(1, std::stoull(argv[++i]));
    } else if (arg == "--chunk_mib" && i + 1 < argc) {
      chunk_mib = std::max<std::size_t>(1, std::stoull(argv[++i]));
    } else {
      usage();
      return 1;
    }
  }
  if (dir.empty() == s3_uri.empty() || (mode != "b" && mode != "p")) {
    usage();
    return 1;
  }
  bool const use_s3 = !s3_uri.empty();

  auto log_sink = sirius::log::make_spdlog_owning_sink({"log", std::nullopt});
  log_sink->set_level(sirius::log::level::warn);
  sirius::log::set_sink(std::move(log_sink));

  cudaFree(nullptr);

  // ---- memory + io context -------------------------------------------------
  // Both sources end up as an ioctx plus the reservation manager backing it;
  // everything downstream is written against those two, not against a backend.
  //
  // S3 borrows both from s3_bench::engine, which builds the REST ioctx from the
  // --config YAML — so the object_store credentials and every REST tunable
  // (max_connections, n_max_concurrent_scans, ...) come from the
  // same file the engine itself reads.  The local path keeps its own hand-rolled
  // uring stack, whose knobs are not configurable from YAML.
  constexpr std::size_t host_region_bytes = sirius::bench::host_region_capacity_v;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> local_mgr;
  std::shared_ptr<sirius::io::ioctx> local_io_ctx;
  std::unique_ptr<sirius::bench::engine> s3_engine;

  if (use_s3) {
    sirius::bench::bench_options opts;
    opts.config_path     = config_path;
    opts.n_reactors      = n_reactors;
    opts.max_nconnection = max_connections;
    opts.chunk_size_mib  = chunk_mib;
    opts.host_chunk_mib  = host_chunk_mib;
    // Allocate the whole host region up front so a mid-run pool growth times the
    // allocator rather than the scan.
    opts.host_initial_pools = std::max<std::size_t>(
      1, host_region_bytes / (sirius::bench::host_pool_size_v * (host_chunk_mib << 20)));
    s3_engine = std::make_unique<sirius::bench::engine>(opts);
  } else {
    // sirius_memory_reservation_manager installs the cucascade GPU allocator as
    // cudf's default device memory resource in its constructor.
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_usage_limit_ratio_per_gpu(0.5)  // GPU pool: 50% of total memory
      .set_reservation_fraction_per_gpu(0.9)
      .use_gpu_id_as_host_id()
      .set_per_numa_region_capacity(host_region_bytes)
      .set_reservation_fraction_per_numa_region(0.9)
      .set_host_pool_features(
        host_chunk_mib << 20,
        1024,  // blocks per slab
        std::max<std::size_t>(1, host_region_bytes / (1024 * (host_chunk_mib << 20))));

    local_mgr =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());

    // Bounce pool comes from the manager — no separate allocation.
    auto* host_space = local_mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
    auto* bounce_mr  = host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();

    auto ctx = std::make_shared<sirius::io::uring::uring_reactor::reactor_context>(
      sirius::io::uring::uring_reactor::reactor_config_type{}, bounce_mr);
    local_io_ctx = std::make_shared<sirius::io::uring::uring_ioctx>(1, std::move(ctx));
    local_io_ctx->start();
  }

  auto& mgr_ref = use_s3 ? s3_engine->mgr() : *local_mgr;
  auto& io_ctx  = use_s3 ? s3_engine->io_ctx() : *local_io_ctx;

  // ---- thread pool + dispatcher --------------------------------------------
  sirius::exec::static_thread_pool pool(static_cast<int>(nthreads), "parse_pool");
  sirius::exec::scoped_dispatcher disp(pool);
  stream_pool_t streams(rmm::cuda_device_id{0}, nthreads);

  // ---- phase 1: discover files ---------------------------------------------
  std::vector<std::string> paths;
  if (use_s3) {
    for (auto const& f : sirius::bench::list_prefix(
           *s3_engine, s3_uri, n_files == 0 ? std::numeric_limits<std::size_t>::max() : n_files)) {
      paths.push_back(f.path);
    }
  } else {
    paths = glob_parquet_files(dir, n_files);
  }
  if (paths.empty()) {
    std::cerr << "no parquet files found in " << (use_s3 ? s3_uri : dir) << "\n";
    return 1;
  }

  // opts used only in phase 1 for range discovery; the benchmark modes build
  // their own.
  auto disc_opts = cudf::io::parquet_reader_options::builder().column_names(COLUMNS).build();
  using hybrid_scan_reader = cudf::io::parquet::experimental::hybrid_scan_reader;

  std::cout << "found " << paths.size() << " file(s) in " << (use_s3 ? s3_uri : dir) << "\n"
            << "source=" << (use_s3 ? "s3" : "local") << "  "
            << "mode=" << (mode == "b" ? "baseline" : "hybrid_scan") << "  nthreads=" << nthreads
            << "\n\n"
            << "=== phase 1: column-chunk discovery ===\n";

  std::vector<file_info> files;
  files.reserve(paths.size());
  std::size_t total_bytes = 0;

  for (auto const& path : paths) {
    file_info fi;
    fi.path = path;
    fi.ds   = io_ctx.open_datasource(path);

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
    run_hybrid_scan(files, disp, streams, total_bytes);
  }

  disp.wait_for_all();
  pool.stop();
  return 0;
}
