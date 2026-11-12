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
#include "io/gds/gds_ioctx.hpp"
#include "io/gds/gds_stats.hpp"
#include "scan_manager/config.hpp"
#include "utils/log_test_utils.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using sirius::io::gds::gds_config;
using sirius::io::gds::gds_ioctx;

namespace {

constexpr std::size_t k_mib = 1ULL << 20;

/// A file of @p size bytes where byte i is a function of i, removed on scope exit.
struct pattern_file {
  explicit pattern_file(std::size_t size)
    : path(std::filesystem::temp_directory_path() /
           ("sirius_gds_ioctx_" + std::to_string(::getpid()) + "_" + std::to_string(size)))
  {
    bytes.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
      bytes[i] = static_cast<std::uint8_t>((i * 31 + (i >> 9)) & 0xff);
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char const*>(bytes.data()), static_cast<std::streamsize>(size));
  }
  ~pattern_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::filesystem::path path;
  std::vector<std::uint8_t> bytes;
};

/// POSIX reads through the same code path: runs where GDS is not set up.
gds_ioctx::options posix_options()
{
  gds_ioctx::options opts;
  opts.compat_mode = kvikio::CompatMode::ON;
  return opts;
}

::cuda::stream_ref default_stream() { return ::cuda::stream_ref{cudaStream_t{nullptr}}; }

std::vector<std::uint8_t> to_host(rmm::device_buffer const& buffer, std::size_t size)
{
  std::vector<std::uint8_t> host(size);
  REQUIRE(cudaMemcpy(host.data(), buffer.data(), size, cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

std::size_t read_device(gds_ioctx& ctx,
                        cucascade::io::io_object const& obj,
                        std::size_t offset,
                        std::size_t size,
                        rmm::device_buffer& dst)
{
  return ctx
    .device_read_async_io(
      obj, offset, size, static_cast<std::uint8_t*>(dst.data()), default_stream())
    .get();
}

/// Whether one message containing @p text and @p also is logged within five seconds.
bool logged(sirius::test::scoped_recording_log_sink const& logs,
            std::string_view text,
            std::string_view also)
{
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    for (auto const& record : logs.records()) {
      if (record.message.find(text) != std::string::npos &&
          record.message.find(also) != std::string::npos) {
        return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

}  // namespace

TEST_CASE("gds_ioctx reads an unaligned range spanning several pieces", "[gds_ioctx]")
{
  pattern_file file(5 * k_mib + 123);
  gds_config config;
  config.num_threads = 4;
  config.task_size   = k_mib;
  gds_ioctx ctx(config, posix_options());
  auto const obj = ctx.open_io_object(file.path.string());

  std::size_t const offset = 4097;
  std::size_t const size   = 3 * k_mib + 777;
  rmm::device_buffer dst(size, rmm::cuda_stream_default);
  REQUIRE(read_device(ctx, *obj, offset, size, dst) == size);

  auto const got = to_host(dst, size);
  CHECK(std::equal(got.begin(), got.end(), file.bytes.begin() + offset));
  auto const counters = ctx.counters();
  CHECK(counters.device_bytes == size);
  CHECK(counters.device_reads == 4);  // 3 full pieces and the remainder
}

TEST_CASE("gds_ioctx clamps reads at the end of the file", "[gds_ioctx]")
{
  pattern_file file(k_mib);
  gds_config config;
  config.num_threads = 2;
  gds_ioctx ctx(config, posix_options());
  auto const obj = ctx.open_io_object(file.path.string());

  rmm::device_buffer dst(4096, rmm::cuda_stream_default);
  CHECK(read_device(ctx, *obj, k_mib - 100, 4096, dst) == 100);
  CHECK(read_device(ctx, *obj, k_mib + 5, 4096, dst) == 0);

  std::vector<std::uint8_t> host(200);
  CHECK(ctx.host_read_io(*obj, k_mib - 150, host.size(), host.data()) == 150);
  CHECK(std::equal(host.begin(), host.begin() + 150, file.bytes.end() - 150));
}

TEST_CASE("gds_ioctx keeps the reads of separate requests in flight together", "[gds_ioctx]")
{
  pattern_file file(k_mib);
  std::atomic<int> in_flight{0};
  std::atomic<int> most{0};
  auto opts   = posix_options();
  opts.reader = [&](kvikio::FileHandle&, void*, std::size_t size, std::size_t) {
    auto const now = ++in_flight;
    for (auto seen = most.load(); now > seen && !most.compare_exchange_weak(seen, now);) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    --in_flight;
    return size;
  };
  gds_config config;
  config.num_threads = 4;
  gds_ioctx ctx(config, std::move(opts));
  auto const obj = ctx.open_io_object(file.path.string());

  // What cuDF does for a row group: issue every column-chunk read, then wait.
  rmm::device_buffer dst(4096, rmm::cuda_stream_default);
  std::vector<cucascade::exec::semi_future<std::size_t>> reads;
  for (int i = 0; i < 4; ++i) {
    reads.push_back(ctx.device_read_async_io(
      *obj, 0, 4096, static_cast<std::uint8_t*>(dst.data()), default_stream()));
  }
  for (auto& read : reads) {
    CHECK(std::move(read).get() == 4096);
  }
  CHECK(most.load() > 1);
}

TEST_CASE("gds_ioctx reports a failed piece and a short read", "[gds_ioctx]")
{
  pattern_file file(4 * k_mib);
  gds_config config;
  config.num_threads = 2;
  config.task_size   = k_mib;
  rmm::device_buffer dst(4 * k_mib, rmm::cuda_stream_default);

  SECTION("the reader throws")
  {
    auto opts   = posix_options();
    opts.reader = [](kvikio::FileHandle&, void*, std::size_t size, std::size_t offset) {
      if (offset >= 2 * k_mib) { throw std::runtime_error("injected read failure"); }
      return size;
    };
    gds_ioctx ctx(config, std::move(opts));
    auto const obj = ctx.open_io_object(file.path.string());
    CHECK_THROWS_WITH(read_device(ctx, *obj, 0, 4 * k_mib, dst), "injected read failure");
    // One bad request does not poison the context.
    CHECK(read_device(ctx, *obj, 0, k_mib, dst) == k_mib);
  }

  SECTION("the reader returns fewer bytes")
  {
    auto opts   = posix_options();
    opts.reader = [](kvikio::FileHandle&, void*, std::size_t size, std::size_t) {
      return size - 1;
    };
    gds_ioctx ctx(config, std::move(opts));
    auto const obj = ctx.open_io_object(file.path.string());
    CHECK_THROWS_WITH(read_device(ctx, *obj, 0, k_mib, dst), "gds_ioctx: short read");
  }
}

TEST_CASE("gds_ioctx fails reads once one is stuck past the timeout", "[gds_ioctx]")
{
  pattern_file file(k_mib);
  std::mutex mutex;
  std::condition_variable cv;
  bool release = false;
  std::atomic<int> fatal{0};

  auto opts   = posix_options();
  opts.reader = [&](kvikio::FileHandle&, void*, std::size_t size, std::size_t) {
    std::unique_lock lock{mutex};
    cv.wait(lock, [&] { return release; });
    return size;
  };
  opts.watchdog_period = std::chrono::milliseconds(20);
  opts.on_fatal_stall  = [&] { ++fatal; };
  gds_config config;
  config.num_threads  = 2;
  config.read_timeout = std::chrono::seconds(1);
  gds_ioctx ctx(config, std::move(opts));
  auto const obj = ctx.open_io_object(file.path.string());

  rmm::device_buffer dst(4096, rmm::cuda_stream_default);
  auto const started = std::chrono::steady_clock::now();
  CHECK_THROWS_AS(read_device(ctx, *obj, 0, 4096, dst), sirius::io::gds::read_stalled_error);
  CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(5));
  CHECK(ctx.stalled());

  // The process is told to end before any request is failed, and only once.
  CHECK(fatal.load() == 1);
  CHECK_THROWS_AS(read_device(ctx, *obj, 0, 4096, dst), sirius::io::gds::read_stalled_error);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(fatal.load() == 1);

  {
    std::lock_guard lock{mutex};
    release = true;
  }
  cv.notify_all();
}

TEST_CASE("gds_ioctx reports a held read while it is out and when it returns", "[gds_ioctx]")
{
  sirius::test::scoped_recording_log_sink logs{"warn"};
  // A size of its own: lines of an earlier test's context name another file.
  pattern_file file(k_mib + 4096);
  std::mutex mutex;
  std::condition_variable cv;
  bool release = false;
  // Releases the held read on every way out, so that a failed check cannot leave the
  // context's destructor waiting for it.
  struct releaser {
    std::mutex& mutex;
    std::condition_variable& cv;
    bool& release;
    ~releaser()
    {
      {
        std::lock_guard lock{mutex};
        release = true;
      }
      cv.notify_all();
    }
  };

  auto opts   = posix_options();
  opts.reader = [&](kvikio::FileHandle&, void*, std::size_t size, std::size_t offset) {
    if (offset == 0) {
      std::unique_lock lock{mutex};
      cv.wait(lock, [&] { return release; });
    }
    return size;
  };
  opts.watchdog_period = std::chrono::milliseconds(20);
  opts.held_after      = std::chrono::milliseconds(200);
  gds_config config;
  config.num_threads          = 2;
  config.slow_read_ms_per_mib = 0;
  gds_ioctx ctx(config, std::move(opts));
  releaser const on_exit{mutex, cv, release};
  auto const obj = ctx.open_io_object(file.path.string());

  rmm::device_buffer held_dst(4096, rmm::cuda_stream_default);
  auto held = ctx.device_read_async_io(
    *obj, 0, 4096, static_cast<std::uint8_t*>(held_dst.data()), default_stream());
  // The report names the read while it is still out, and other reads go on.
  auto const where = "4096 bytes at 0 of " + file.path.string();
  REQUIRE(logged(logs, "[gds] held: 1 reads have not returned, the oldest a cuFile read", where));
  CHECK(logged(logs, "[gds] held: 1 reads have not returned", "; GB/s received per RDMA port: "));
  rmm::device_buffer dst(4096, rmm::cuda_stream_default);
  CHECK(read_device(ctx, *obj, 8192, 4096, dst) == 4096);
  CHECK(ctx.counters().held_reads == 0);

  {
    std::lock_guard lock{mutex};
    release = true;
  }
  cv.notify_all();
  CHECK(std::move(held).get() == 4096);
  CHECK(ctx.counters().held_reads == 1);
  CHECK(logged(logs, "[gds] held: a cuFile read returned after", where));
  CHECK_FALSE(ctx.stalled());
}

TEST_CASE("gds_ioctx stops issuing reads when they stay slow", "[gds_ioctx]")
{
  pattern_file file(k_mib);
  auto opts   = posix_options();
  opts.reader = [](kvikio::FileHandle&, void*, std::size_t size, std::size_t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    return size;
  };
  opts.watchdog_period     = std::chrono::milliseconds(100);
  opts.slow_read_min_reads = 1;
  gds_config config;
  config.num_threads          = 2;
  config.slow_read_ms_per_mib = 5.0;  // a 4 KiB read is charged as 1 MiB: 30 ms per MiB
  gds_ioctx ctx(config, std::move(opts));
  auto const obj = ctx.open_io_object(file.path.string());

  // Slow reads still succeed until they have stayed slow for two seconds.
  rmm::device_buffer dst(4096, rmm::cuda_stream_default);
  auto const started = std::chrono::steady_clock::now();
  bool failed        = false;
  while (!failed && std::chrono::steady_clock::now() - started < std::chrono::seconds(10)) {
    try {
      // Outside CHECK, which would swallow the exception.
      std::size_t const n = read_device(ctx, *obj, 0, 4096, dst);
      REQUIRE(n == 4096);
    } catch (sirius::io::gds::read_stalled_error const&) {
      failed = true;
    }
  }
  CHECK(failed);
  CHECK(std::chrono::steady_clock::now() - started >= std::chrono::seconds(2));
  CHECK_FALSE(ctx.stalled());  // no read is stuck: the process is not told to exit
}

TEST_CASE("gds_ioctx serves local files only", "[gds_ioctx]")
{
  gds_config config;
  config.num_threads = 1;
  gds_ioctx ctx(config, posix_options());
  CHECK(ctx.supports("/data/lineitem/part.0.parquet"));
  CHECK_FALSE(ctx.supports("s3://bucket/key.parquet"));
  CHECK_THROWS(ctx.open_io_object("s3://bucket/key.parquet"));
}

TEST_CASE("nvidia-fs Mmap count is read from the driver statistics", "[gds_ioctx]")
{
  std::istringstream stats(
    "GDS Version: 1.17.0.46 \n"
    "Reads\t\t\t\t: err=0 io_state_err=311\n"
    "Mmap\t\t\t\t: n=49559 ok=49559 err=0 munmap=49559\n"
    "Bar1-map\t\t\t: n=7 ok=7 err=0 free=7 callbacks=0 active=0 delay-frees=0\n");
  CHECK(sirius::io::gds::parse_nvidia_fs_mmap_count(stats) == std::uint64_t{49559});

  std::istringstream none("Reads: err=0\n");
  CHECK_FALSE(sirius::io::gds::parse_nvidia_fs_mmap_count(none).has_value());
}

TEST_CASE("cuFile counters classify the read path", "[gds_ioctx]")
{
  using sirius::io::gds::classify;
  using sirius::io::gds::cufile_counters;
  using sirius::io::gds::read_path;
  CHECK(classify(cufile_counters{}) == read_path::none);
  CHECK(classify(cufile_counters{.read_bytes = 10, .total_reads = 4, .nvfs_reads = 4}) ==
        read_path::gds);
  CHECK(classify(cufile_counters{.read_bytes = 10, .total_reads = 4, .posix_reads = 4}) ==
        read_path::posix);
  CHECK(classify(cufile_counters{.read_bytes = 10, .total_reads = 4, .posix_reads = 3}) ==
        read_path::mixed);
  // A few tiny reads go through POSIX inside cuFile even under GDS.
  CHECK(classify(cufile_counters{.read_bytes = 10, .total_reads = 400, .posix_reads = 3}) ==
        read_path::gds);
}

TEST_CASE("backend gds routes local paths to the context type gds_ioctx reports", "[gds_ioctx]")
{
  sirius::scan_manager::io_backend backend{};
  REQUIRE(sirius::scan_manager::string_to_enum("gds", backend));
  CHECK(backend == sirius::scan_manager::io_backend::gds);

  // With any other mapping the registry resolves local files to uring and the
  // scan manager reads through io_uring without saying so.
  sirius::scan_manager::scan_manager_config config;
  config.backend = backend;
  CHECK(config.to_io_config().backend == cucascade::io::io_backend::kvikio);
}
