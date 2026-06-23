/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#include "catch.hpp"
#include "io/cache/prefetching_cache.hpp"
#include "io/io_context.hpp"

#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/numa_region_pinned_host_allocator.hpp>
#include <io/sirius_datasource.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

using sirius::io::buffer_pool;
using sirius::io::io_completion_handler;
using sirius::io::prefetching_cache;
using sirius::io::sirius_io_object;
using sirius::io::sirius_io_object_metadata;
using sirius::io::sirius_ioctx;

std::size_t cache_capacity_bytes(std::size_t block_size, std::uint32_t max_slabs)
{
  return block_size * static_cast<std::size_t>(buffer_pool::CHUNKS_PER_SLAB) *
         static_cast<std::size_t>(max_slabs);
}

struct cache_test_resources {
  static constexpr std::uint32_t max_slabs = 1;
  static constexpr std::size_t block_size  = 4096;

  cache_test_resources()
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

class test_io_object final : public sirius_io_object {
 public:
  test_io_object(std::string cache_id, std::size_t size)
    : _cache_id(std::move(cache_id)), _path(_cache_id), _size(size)
  {
  }

  [[nodiscard]] std::string const& raw_file_cache_id() const noexcept override { return _cache_id; }
  [[nodiscard]] std::string const& object_path() const noexcept override { return _path; }
  [[nodiscard]] std::size_t size() const noexcept override { return _size; }

 private:
  std::string _cache_id;
  std::string _path;
  std::size_t _size;
};

class memory_ioctx final : public sirius_ioctx {
 public:
  explicit memory_ioctx(std::vector<std::byte> data) : _data(std::move(data)) {}

  void shutdown() override {}

  std::shared_ptr<sirius_io_object> create_io_object(std::string path) override
  {
    return std::make_shared<test_io_object>(std::move(path), _data.size());
  }

  std::unique_ptr<sirius::io::sirius_datasource> make_datasource(
    std::shared_ptr<sirius_io_object>) override
  {
    throw std::runtime_error("memory_ioctx does not implement make_datasource");
  }

  [[nodiscard]] bool supports(std::string_view) const override { return true; }

  std::size_t host_read_io(sirius_io_object&,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override
  {
    if (offset >= _data.size()) { return 0; }
    auto const n = std::min(size, _data.size() - offset);
    std::memcpy(dst, _data.data() + offset, n);
    return n;
  }

  void host_read_async_io(sirius_io_object& obj,
                          std::size_t offset,
                          std::size_t size,
                          std::uint8_t* dst,
                          io_completion_handler handler) override
  {
    try {
      handler(host_read_io(obj, offset, size, dst), nullptr);
    } catch (...) {
      handler(0, std::current_exception());
    }
  }

  std::size_t device_read_io(
    sirius_io_object&, std::size_t, std::size_t, std::uint8_t*, rmm::cuda_stream_view) override
  {
    throw std::runtime_error("memory_ioctx does not implement device_read_io");
  }

  void device_read_async_io(sirius_io_object&,
                            std::size_t,
                            std::size_t,
                            std::uint8_t*,
                            rmm::cuda_stream_view,
                            io_completion_handler handler) override
  {
    handler(0,
            std::make_exception_ptr(
              std::runtime_error("memory_ioctx does not implement device_read_async_io")));
  }

  void host_read_ranges_async_io(sirius_io_object&,
                                 std::vector<cudf::io::text::byte_range_info> const& ranges,
                                 std::span<cudf::host_span<std::byte>> dst,
                                 io_completion_handler handler) override
  {
    try {
      if (ranges.size() != dst.size()) {
        throw std::runtime_error("range/destination size mismatch");
      }
      _range_dispatch_count.fetch_add(1, std::memory_order_relaxed);
      if (dispatch_started) {
        try {
          dispatch_started->set_value();
        } catch (std::future_error const&) {
        }
      }
      if (unblock_dispatch.valid()) { unblock_dispatch.wait(); }

      std::size_t total = 0;
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        auto const offset = static_cast<std::size_t>(ranges[i].offset());
        auto const size   = static_cast<std::size_t>(ranges[i].size());
        if (offset + size > _data.size()) { throw std::runtime_error("range exceeds data size"); }
        if (dst[i].size() != size) {
          throw std::runtime_error("destination span has unexpected size");
        }
        std::memcpy(dst[i].data(), _data.data() + offset, size);
        total += size;
      }
      handler(total, nullptr);
      if (dispatch_completed) {
        try {
          dispatch_completed->set_value();
        } catch (std::future_error const&) {
        }
      }
    } catch (...) {
      handler(0, std::current_exception());
    }
  }

  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         std::size_t file_size) const override
  {
    auto const offset = static_cast<std::size_t>(logical.offset());
    if (offset >= file_size) {
      return cudf::io::text::byte_range_info{static_cast<int64_t>(file_size), 0};
    }
    auto const size = std::min(static_cast<std::size_t>(logical.size()), file_size - offset);
    return cudf::io::text::byte_range_info{static_cast<int64_t>(offset),
                                           static_cast<int64_t>(size)};
  }

  [[nodiscard]] int range_dispatch_count() const noexcept
  {
    return _range_dispatch_count.load(std::memory_order_relaxed);
  }

  std::shared_ptr<std::promise<void>> dispatch_started;
  std::shared_ptr<std::promise<void>> dispatch_completed;
  std::shared_future<void> unblock_dispatch;

 private:
  std::vector<std::byte> _data;
  std::atomic<int> _range_dispatch_count{0};
};

std::vector<std::byte> make_data(std::size_t size)
{
  std::vector<std::byte> data(size);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<std::byte>(i % 251);
  }
  return data;
}

std::shared_ptr<test_io_object> make_object(std::string cache_id = "mem://object",
                                            std::size_t size     = 4096)
{
  return std::make_shared<test_io_object>(std::move(cache_id), size);
}

void require_counter_state(prefetching_cache const& cache,
                           std::uint64_t hits,
                           std::uint64_t hit_after_wait,
                           std::uint64_t partial_misses,
                           std::uint64_t full_misses,
                           std::uint64_t range_misses)
{
  CHECK(cache.hit_count_total() == hits);
  CHECK(cache.hit_after_wait_total() == hit_after_wait);
  CHECK(cache.partial_miss_count_total() == partial_misses);
  CHECK(cache.full_miss_count_total() == full_misses);
  CHECK(cache.range_miss_count_total() == range_misses);
}

std::future<void> future_from(std::shared_ptr<std::promise<void>> promise)
{
  return promise->get_future();
}

void wait_for(std::future<void>& future, std::chrono::milliseconds timeout)
{
  REQUIRE(future.wait_for(timeout) == std::future_status::ready);
}

}  // namespace

TEST_CASE("prefetching_cache telemetry counters start at zero", "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx = std::make_shared<memory_ioctx>(make_data(4096));
  prefetching_cache cache(resources.pool, ctx.get(), 16);

  require_counter_state(cache, 0, 0, 0, 0, 0);
}

TEST_CASE("prefetching_cache telemetry counts full misses for unknown files",
          "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx = std::make_shared<memory_ioctx>(make_data(4096));
  prefetching_cache cache(resources.pool, ctx.get(), 16);
  auto obj = make_object();

  auto view = cache.read(*obj, 0, 1024, nullptr);

  CHECK_FALSE(view);
  require_counter_state(cache, 0, 0, 0, 1, 0);
}

TEST_CASE("prefetching_cache telemetry counts metadata-only range misses",
          "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx = std::make_shared<memory_ioctx>(make_data(4096));
  prefetching_cache cache(resources.pool, ctx.get(), 16);
  auto obj = make_object();

  cache.insert(*obj, std::make_shared<sirius_io_object_metadata>(), {});
  auto view = cache.read(*obj, 0, 1024, nullptr);

  CHECK_FALSE(view);
  require_counter_state(cache, 0, 0, 0, 0, 1);
}

TEST_CASE("prefetching_cache telemetry counts completed prefetch hits",
          "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx                = std::make_shared<memory_ioctx>(make_data(4096));
  auto completed          = std::make_shared<std::promise<void>>();
  auto done               = future_from(completed);
  ctx->dispatch_completed = completed;
  prefetching_cache cache(resources.pool, ctx.get(), 16);
  auto obj = make_object();

  cache.insert(*obj, nullptr, {{0, 1024}});
  wait_for(done, 2s);

  auto view = cache.read(*obj, 0, 1024, nullptr);

  CHECK(view);
  require_counter_state(cache, 1, 0, 0, 0, 0);
}

TEST_CASE("prefetching_cache telemetry exposes hits after loading waits",
          "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx              = std::make_shared<memory_ioctx>(make_data(4096));
  auto started          = std::make_shared<std::promise<void>>();
  auto unblock          = std::make_shared<std::promise<void>>();
  auto dispatch_started = future_from(started);
  ctx->dispatch_started = started;
  ctx->unblock_dispatch = unblock->get_future().share();
  prefetching_cache cache(resources.pool, ctx.get(), 16);
  auto obj = make_object();

  cache.insert(*obj, nullptr, {{0, 1024}});
  wait_for(dispatch_started, 2s);

  auto read_future = std::async(std::launch::async, [&] {
    auto view = cache.read(*obj, 0, 1024, nullptr);
    return static_cast<bool>(view);
  });
  std::this_thread::sleep_for(50ms);
  unblock->set_value();

  REQUIRE(read_future.get());
  require_counter_state(cache, 1, 1, 0, 0, 0);
}

TEST_CASE("prefetching_cache telemetry documents sub-range read behavior",
          "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx                = std::make_shared<memory_ioctx>(make_data(4096));
  auto completed          = std::make_shared<std::promise<void>>();
  auto done               = future_from(completed);
  ctx->dispatch_completed = completed;
  prefetching_cache cache(resources.pool, ctx.get(), 16);
  auto obj = make_object();

  cache.insert(*obj, nullptr, {{0, 1024}});
  wait_for(done, 2s);

  auto view = cache.read(*obj, 100, 200, nullptr);

  INFO("Sub-range read inside cached [0,1024) returned "
       << (view ? "a cache hit" : "a cache miss"));
  if (view) {
    require_counter_state(cache, 1, 0, 0, 0, 0);
  } else {
    require_counter_state(cache, 0, 0, 0, 0, 1);
  }
}

TEST_CASE("prefetching_cache telemetry counters are thread-safe under contention",
          "[prefetching_cache][telemetry]")
{
  cache_test_resources resources;
  auto ctx                = std::make_shared<memory_ioctx>(make_data(8192));
  auto completed          = std::make_shared<std::promise<void>>();
  auto done               = future_from(completed);
  ctx->dispatch_completed = completed;
  prefetching_cache cache(resources.pool, ctx.get(), 64);
  auto cached_obj        = make_object("mem://cached", 8192);
  auto metadata_only_obj = make_object("mem://metadata-only", 8192);
  auto missing_obj       = make_object("mem://missing", 8192);

  cache.insert(*cached_obj, nullptr, {{0, 1024}});
  wait_for(done, 2s);
  cache.insert(*metadata_only_obj, std::make_shared<sirius_io_object_metadata>(), {});

  constexpr int n_threads          = 8;
  constexpr int iterations_per_thr = 1000;
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < iterations_per_thr; ++i) {
        if (!cache.read(*cached_obj, 0, 128, nullptr)) { failures.fetch_add(1); }
        if (cache.read(*metadata_only_obj, 0, 128, nullptr)) { failures.fetch_add(1); }
        if (cache.read(*missing_obj, 0, 128, nullptr)) { failures.fetch_add(1); }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  CHECK(failures.load() == 0);
  auto const n = static_cast<std::uint64_t>(n_threads * iterations_per_thr);
  require_counter_state(cache, n, 0, 0, n, n);
}
