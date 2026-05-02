/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Stress test for prefetching_cache on top of the io_uring datasource.
 *
 * Three runs over identical task streams (fixed seed):
 *   - CACHE OFF     : uring_ioctx, no cache — every range hits O_DIRECT + memcpy.
 *   - CACHE ON      : uring_ioctx + prefetching_cache (insert fires before enqueue).
 *   - CUDF DEFAULT  : cudf::io::datasource::create(path) — whatever cudf picks
 *                     by default (kvikio / cuFile / pread fallback).
 *
 * Schedule:
 *   - 10 files under ~/Documents/tpch/sf200/parquet/lineitem/.
 *   - 30 random ranges pre-generated per file (size in [1, 8] MiB).
 *   - Thread pool pulls tasks: (file + 10 ranges sampled from its 30).
 *   - Each worker sleeps 10us, issues async device reads, sync stream, frees.
 */

#include "concurrentqueue.h"
#include "ctrack.hpp"
#include "io/prefetching_cache.hpp"
#include "io/uring/uring_ioctx.hpp"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Range  = cudf::io::text::byte_range_info;

// ---- configuration ---------------------------------------------------------

static constexpr size_t N_FILES                = 10;
static constexpr size_t RANGES_PER_FILE        = 30;
static constexpr size_t RANGES_PER_TASK        = 10;
static constexpr size_t RANGE_MIN_BYTES        = 1UL << 20;  // 1 MiB
static constexpr size_t RANGE_MAX_BYTES        = 8UL << 20;  // 8 MiB
static constexpr auto WORKER_SLEEP             = std::chrono::microseconds(10);
static constexpr uint64_t FIXED_SEED           = 0x5115'C0DE'DEAD'BEEFULL;
static constexpr size_t INFLIGHT_BUDGET_CHUNKS = 2048;

// ---- types -----------------------------------------------------------------

struct read_task {
  size_t file_idx;
  std::vector<Range> ranges;
};

using read_fn_t = std::function<std::future<size_t>(
  size_t file_idx, size_t offset, size_t size, uint8_t* dst, rmm::cuda_stream_view stream)>;

using insert_fn_t = std::function<void(size_t file_idx, std::vector<Range> const&)>;

struct run_result {
  double elapsed_s;
  size_t bytes;
};

// ---- setup -----------------------------------------------------------------

static std::vector<fs::path> enumerate_paths(fs::path const& dir)
{
  std::vector<fs::path> paths;
  for (auto const& de : fs::directory_iterator(dir))
    if (de.is_regular_file()) paths.push_back(de.path());
  std::sort(paths.begin(), paths.end());
  if (paths.size() < N_FILES)
    throw std::runtime_error("need >= " + std::to_string(N_FILES) + " files in " + dir.string());
  paths.resize(N_FILES);
  return paths;
}

static std::vector<std::vector<Range>> generate_ranges_per_file(
  std::vector<size_t> const& file_sizes, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<std::vector<Range>> out(file_sizes.size());
  std::uniform_int_distribution<size_t> size_dist(RANGE_MIN_BYTES, RANGE_MAX_BYTES);
  for (size_t fi = 0; fi < file_sizes.size(); ++fi) {
    auto file_size = file_sizes[fi];
    out[fi].reserve(RANGES_PER_FILE);
    for (size_t i = 0; i < RANGES_PER_FILE; ++i) {
      size_t sz = std::min(size_dist(rng), file_size);
      std::uniform_int_distribution<size_t> off_dist(0, file_size - sz);
      size_t off = off_dist(rng);
      out[fi].emplace_back(static_cast<int64_t>(off), static_cast<int64_t>(sz));
    }
  }
  return out;
}

// ---- generic runner --------------------------------------------------------

static run_result run_generic(std::string const& label,
                              std::vector<std::vector<Range>> const& per_file_ranges,
                              size_t total_tasks,
                              size_t n_workers,
                              read_fn_t read_fn,
                              insert_fn_t insert_fn,
                              std::function<void()> post_hook = nullptr)
{
  duckdb_moodycamel::ConcurrentQueue<read_task> queue;
  std::atomic<size_t> bytes_read{0};
  std::atomic<size_t> tasks_done{0};
  std::atomic<size_t> remaining{total_tasks};

  auto worker_fn = [&]() {
    cudaStream_t raw_stream = nullptr;
    cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking);
    rmm::cuda_stream_view stream{raw_stream};

    read_task t;
    while (true) {
      if (!queue.try_dequeue(t)) {
        if (remaining.load(std::memory_order_acquire) == 0) break;
        std::this_thread::yield();
        continue;
      }

      std::vector<rmm::device_buffer> bufs;
      bufs.reserve(t.ranges.size());
      std::vector<std::future<size_t>> futs;
      futs.reserve(t.ranges.size());

      {
        CTRACK_NAME("task::issue_reads");
        for (auto const& r : t.ranges) {
          auto sz = static_cast<size_t>(r.size());
          bufs.emplace_back(sz, stream);
          futs.push_back(read_fn(t.file_idx,
                                 static_cast<size_t>(r.offset()),
                                 sz,
                                 static_cast<uint8_t*>(bufs.back().data()),
                                 stream));
        }
      }

      size_t total = 0;
      {
        CTRACK_NAME("task::wait_futures");
        for (auto& f : futs)
          total += f.get();
      }

      // Per-task stream_sync deferred: rmm::device_buffer deallocation is
      // stream-ordered by the async MR, so the buffers can drop out of scope
      // while the copy is still on the stream — dealloc is queued behind the
      // memcpy.  One sync at worker exit below guarantees measurement
      // correctness.

      {
        CTRACK_NAME("task::sleep");
        std::this_thread::sleep_for(WORKER_SLEEP);
      }

      bytes_read.fetch_add(total, std::memory_order_relaxed);
      tasks_done.fetch_add(1, std::memory_order_relaxed);
      remaining.fetch_sub(1, std::memory_order_release);
    }

    {
      CTRACK_NAME("worker::final_sync");
      cudaStreamSynchronize(raw_stream);
    }
    cudaStreamDestroy(raw_stream);
  };

  // Phase 1 — generate all tasks up-front and register every prefetch
  // request before any worker starts.  This deliberately piles n_tasks *
  // ranges_per_task of prefetch pressure onto the cache before any drain
  // happens, exercising the eviction path.
  std::vector<read_task> all_tasks;
  all_tasks.reserve(total_tasks);
  {
    std::mt19937_64 rng(FIXED_SEED);
    std::uniform_int_distribution<size_t> file_dist(0, per_file_ranges.size() - 1);
    std::vector<size_t> indices(RANGES_PER_FILE);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t i = 0; i < total_tasks; ++i) {
      size_t fi       = file_dist(rng);
      auto const& all = per_file_ranges[fi];
      std::shuffle(indices.begin(), indices.end(), rng);
      std::vector<Range> picked;
      picked.reserve(RANGES_PER_TASK);
      for (size_t k = 0; k < RANGES_PER_TASK; ++k)
        picked.push_back(all[indices[k]]);
      std::sort(picked.begin(), picked.end(), [](Range const& a, Range const& b) {
        return a.offset() < b.offset();
      });
      all_tasks.push_back(read_task{fi, std::move(picked)});
    }
  }

  auto t0 = std::chrono::steady_clock::now();

  // Register all prefetch requests before any worker starts.
  if (insert_fn) {
    for (auto const& t : all_tasks)
      insert_fn(t.file_idx, t.ranges);
  }

  // Enqueue all tasks, then spawn workers.
  for (auto& t : all_tasks)
    queue.enqueue(std::move(t));
  all_tasks.clear();

  std::vector<std::thread> workers;
  workers.reserve(n_workers);
  for (size_t i = 0; i < n_workers; ++i)
    workers.emplace_back(worker_fn);

  // Periodic status dump — helps diagnose stalls.
  std::atomic<bool> done{false};
  std::thread watchdog([&] {
    size_t last_done = 0;
    int still_ticks  = 0;
    while (!done.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (done.load(std::memory_order_relaxed)) break;
      size_t cur_done = tasks_done.load(std::memory_order_relaxed);
      size_t rem      = remaining.load(std::memory_order_relaxed);
      std::cerr << "[watchdog] tasks_done=" << cur_done << " remaining=" << rem;
      if (cur_done == last_done) {
        ++still_ticks;
        std::cerr << "  (no progress x" << still_ticks << ")";
      } else {
        still_ticks = 0;
      }
      last_done = cur_done;
      std::cerr << "\n";
    }
  });

  for (auto& w : workers)
    w.join();
  done.store(true, std::memory_order_relaxed);
  watchdog.join();

  auto t1            = std::chrono::steady_clock::now();
  double elapsed     = std::chrono::duration<double>(t1 - t0).count();
  size_t total_bytes = bytes_read.load(std::memory_order_acquire);

  std::cout << "\n=== " << label << " ===\n";
  std::cout << "  tasks        : " << tasks_done.load() << "\n";
  std::cout << "  elapsed      : " << elapsed << " s\n";
  std::cout << "  bytes        : " << (total_bytes / double(1UL << 30)) << " GiB\n";
  std::cout << "  throughput   : " << (total_bytes / double(1UL << 30) / elapsed) << " GiB/s\n";

  if (post_hook) post_hook();

  std::cout << "\n--- ctrack breakdown for " << label << " ---\n";
  ctrack::result_print();

  return {elapsed, total_bytes};
}

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv)
{
  size_t total_tasks      = 1000;
  size_t n_workers        = std::max(1u, std::thread::hardware_concurrency());
  uint32_t pool_max_slabs = 8;
  if (argc > 1) total_tasks = std::stoul(argv[1]);
  if (argc > 2) n_workers = std::stoul(argv[2]);
  if (argc > 3) pool_max_slabs = static_cast<uint32_t>(std::stoul(argv[3]));

  fs::path dir = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/root") /
                 "Documents/tpch/sf200/parquet/lineitem";

  std::cout << "stress_prefetching\n"
            << "  dir          : " << dir << "\n"
            << "  files        : " << N_FILES << "\n"
            << "  ranges/file  : " << RANGES_PER_FILE << "\n"
            << "  ranges/task  : " << RANGES_PER_TASK << "\n"
            << "  range size   : [" << (RANGE_MIN_BYTES >> 20) << ", " << (RANGE_MAX_BYTES >> 20)
            << "] MiB\n"
            << "  total tasks  : " << total_tasks << "\n"
            << "  workers      : " << n_workers << "\n"
            << "  pool cap     : " << pool_max_slabs << " slabs (" << (pool_max_slabs * 500)
            << " MiB max)\n";

  rmm::mr::cuda_async_memory_resource async_mr;
  rmm::mr::set_current_device_resource(&async_mr);

  auto paths = enumerate_paths(dir);

  // Open uring io_objects; capture sizes to build the range map.
  std::vector<std::unique_ptr<sirius::io::uring_io_object>> u_objs;
  u_objs.reserve(paths.size());
  std::vector<size_t> sizes;
  sizes.reserve(paths.size());
  for (auto const& p : paths) {
    u_objs.push_back(std::make_unique<sirius::io::uring_io_object>(p.string()));
    sizes.push_back(u_objs.back()->size());
  }
  auto per_file_ranges = generate_ranges_per_file(sizes, FIXED_SEED);

  // ---- Run 1: CACHE OFF (uring_ioctx, no cache) ----------------------------
  {
    auto ioctx   = std::make_shared<sirius::io::uring_ioctx>();
    auto read_fn = [&](
                     size_t fi, size_t off, size_t sz, uint8_t* dst, rmm::cuda_stream_view stream) {
      auto p = std::make_shared<std::promise<size_t>>();
      auto f = p->get_future();
      ioctx->device_read_async(
        *u_objs[fi], off, sz, dst, stream, [p](size_t n, std::exception_ptr ep) {
          if (ep)
            p->set_exception(ep);
          else
            p->set_value(n);
        });
      return f;
    };
    run_generic("CACHE OFF  (uring, no cache)",
                per_file_ranges,
                total_tasks,
                n_workers,
                read_fn,
                /*insert_fn=*/nullptr);
    ioctx->shutdown();
  }

  // ---- Run 2: CACHE ON (uring_ioctx + prefetching_cache) -------------------
  {
    auto ioctx = std::make_shared<sirius::io::uring_ioctx>();
    sirius::io::buffer_pool pool(pool_max_slabs);
    ioctx->initialize_cache(pool, INFLIGHT_BUDGET_CHUNKS);

    auto read_fn = [&](
                     size_t fi, size_t off, size_t sz, uint8_t* dst, rmm::cuda_stream_view stream) {
      auto p = std::make_shared<std::promise<size_t>>();
      auto f = p->get_future();
      ioctx->device_read_async(
        *u_objs[fi], off, sz, dst, stream, [p](size_t n, std::exception_ptr ep) {
          if (ep)
            p->set_exception(ep);
          else
            p->set_value(n);
        });
      return f;
    };
    auto insert_fn = [&](size_t fi, std::vector<Range> const& ranges) {
      ioctx->cache()->insert(*u_objs[fi], nullptr, ranges);
    };
    run_generic("CACHE ON   (uring + prefetching_cache)",
                per_file_ranges,
                total_tasks,
                n_workers,
                read_fn,
                insert_fn,
                [&] { std::cout << "  cache        : " << ioctx->cache()->summary() << "\n"; });
    ioctx->shutdown();
  }

  // ---- Run 3: CUDF DEFAULT via host_read + cudaMemcpy ---------------------
  //
  // We explicitly go through host_read (whatever cudf's default datasource
  // does for buffered CPU reads) and then cudaMemcpy to the device buffer.
  // This matches the "no-cache, no GDS" path a typical caller would write.
  {
    std::vector<std::unique_ptr<cudf::io::datasource>> ds;
    ds.reserve(paths.size());
    for (auto const& p : paths)
      ds.push_back(cudf::io::datasource::create(p.string()));

    auto read_fn = [&](
                     size_t fi, size_t off, size_t sz, uint8_t* dst, rmm::cuda_stream_view stream) {
      return std::async(std::launch::async, [&, fi, off, sz, dst, stream]() -> size_t {
        std::vector<uint8_t> host_buf(sz);
        size_t n = ds[fi]->host_read(off, sz, host_buf.data());
        auto err = cudaMemcpyAsync(dst, host_buf.data(), n, cudaMemcpyHostToDevice, stream.value());
        if (err != cudaSuccess)
          throw std::runtime_error(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(err));
        // Wait on the copy so host_buf is safe to free on return.
        cudaStreamSynchronize(stream.value());
        return n;
      });
    };
    run_generic("CUDF DEFAULT (host_read + cudaMemcpy)",
                per_file_ranges,
                total_tasks,
                n_workers,
                read_fn,
                /*insert_fn=*/nullptr);
  }

  return 0;
}
