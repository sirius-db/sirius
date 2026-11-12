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

// gds_probe: read one file into GPU memory with cuFileRead, outside the engine.
//
// It is the first thing to run on a machine before `scan_manager.backend: gds`:
// it issues the same calls as gds_ioctx (kvikio::FileHandle::read in pieces on
// a few threads), starting as small as one piece, and says whether the bytes
// went through GDS. It exits by itself when a read stops returning.
//
//   gds_probe --file F [--bytes N] [--seconds S] [--threads T] [--task-size B]
//             [--offset O] [--stall-seconds S] [--verify] [--pool-gib N [--raw-pointer]]
//
// Reads N bytes starting at O (wrapping inside the file) or for S seconds,
// whichever is given. Exit status: 0 the reads went through GDS, 2 the reads
// worked but cuFile reported POSIX reads or the bytes differed, 70 a read
// stalled, 1 anything else.

#include "io/gds/gds_stats.hpp"

#include <kvikio/compat_mode.hpp>
#include <kvikio/cufile/driver.hpp>
#include <kvikio/file_handle.hpp>
#include <kvikio/shim/cufile.hpp>
#include <kvikio/shim/utils.hpp>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct arguments {
  std::string file;
  std::size_t bytes{0};
  double seconds{0};
  std::size_t threads{1};
  std::size_t task_size{16ULL << 20};
  std::size_t offset{0};
  double stall_seconds{10};
  bool verify{false};
  /// 0: one cudaMalloc per thread. N: one N GiB allocation shared by the threads, the way a
  /// memory pool hands out pieces of one large allocation.
  std::size_t pool_gib{0};
  /// With a pool: pass cuFile the piece's own address instead of the pool's base and an offset.
  bool raw_pointer{false};
};

arguments parse(int argc, char** argv)
{
  arguments args;
  for (int i = 1; i < argc; ++i) {
    std::string const key = argv[i];
    auto const value      = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "gds_probe: %s needs a value\n", key.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (key == "--file") {
      args.file = value();
    } else if (key == "--bytes") {
      args.bytes = std::stoull(value());
    } else if (key == "--seconds") {
      args.seconds = std::stod(value());
    } else if (key == "--threads") {
      args.threads = std::stoull(value());
    } else if (key == "--task-size") {
      args.task_size = std::stoull(value());
    } else if (key == "--offset") {
      args.offset = std::stoull(value());
    } else if (key == "--stall-seconds") {
      args.stall_seconds = std::stod(value());
    } else if (key == "--pool-gib") {
      args.pool_gib = std::stoull(value());
    } else if (key == "--raw-pointer") {
      args.raw_pointer = true;
    } else if (key == "--verify") {
      args.verify = true;
    } else {
      std::fprintf(stderr, "gds_probe: unknown argument %s\n", key.c_str());
      std::exit(1);
    }
  }
  if (args.file.empty() || args.threads == 0 || args.task_size == 0 ||
      (args.bytes == 0 && args.seconds <= 0)) {
    std::fprintf(stderr,
                 "usage: gds_probe --file F (--bytes N | --seconds S) [--threads T] "
                 "[--task-size B] [--offset O] [--stall-seconds S] [--verify]\n");
    std::exit(1);
  }
  return args;
}

void check(cudaError_t status, char const* what)
{
  if (status != cudaSuccess) {
    std::fprintf(stderr, "gds_probe: %s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
  }
}

std::int64_t now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch())
    .count();
}

}  // namespace

int main(int argc, char** argv)
{
  auto const args = parse(argc, argv);
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  // cuFile needs a CUDA context before its driver opens.
  check(cudaSetDevice(0), "cudaSetDevice");
  check(cudaFree(nullptr), "create CUDA context");

  // kvikIO reports GDS as unavailable until the cuFile driver is open.
  try {
    kvikio::cuFileAPI::instance().driver_open();
  } catch (std::exception const& e) {
    std::printf("driver_open failed: %s\n", e.what());
    return 1;
  }
  kvikio::DriverProperties props;
  bool const available = props.is_gds_available();
  std::printf("gds_available=%d nvidia_fs=%u.%u posix_fallback_allowed=%d\n",
              available,
              available ? props.get_nvfs_major_version() : 0,
              available ? props.get_nvfs_minor_version() : 0,
              available ? props.get_nvfs_allow_compat_mode() : 0);
  if (!available) {
    // Say which of kvikIO's conditions failed.
    std::printf("libcufile_loaded=%d cufile_usable=%d cufile_version=%d run_udev_readable=%d\n",
                kvikio::is_cufile_library_available(),
                kvikio::is_cufile_available(),
                kvikio::cufile_version(),
                kvikio::run_udev_readable());
    return 1;
  }

  bool const have_counters = sirius::io::gds::enable_cufile_counters();
  auto const cufile_before = sirius::io::gds::read_cufile_counters();
  auto const mmap_before   = sirius::io::gds::read_nvidia_fs_mmap_count();

  // --file takes a comma-separated list: the reads walk through the files in turn, so a run
  // over many files reads bytes nothing has cached.
  std::vector<std::string> files;
  for (std::size_t begin = 0; begin <= args.file.size();) {
    auto const end = std::min(args.file.find(',', begin), args.file.size());
    if (end > begin) { files.push_back(args.file.substr(begin, end - begin)); }
    begin = end + 1;
  }
  std::vector<kvikio::FileHandle> handles;
  std::size_t file_size = SIZE_MAX;
  for (auto const& file : files) {
    handles.emplace_back(file, "r", kvikio::FileHandle::m644, kvikio::CompatMode::OFF);
    file_size = std::min(file_size, handles.back().nbytes());
  }
  if (args.offset >= file_size) {
    std::fprintf(stderr, "gds_probe: offset is past the end of the file\n");
    return 1;
  }
  // Pieces are laid out from --offset to the end of a file (the shortest one's length), then
  // go on in the next file, and wrap around after the last.
  auto const span = file_size - args.offset;

  // Per thread: the address cuFile is given as the base, and the offset from it.
  std::vector<void*> buffers(args.threads);
  std::vector<std::size_t> base_offsets(args.threads, 0);
  void* pool = nullptr;
  if (args.pool_gib != 0) {
    check(cudaMalloc(&pool, args.pool_gib << 30), "cudaMalloc pool");
    if (args.threads * args.task_size > (args.pool_gib << 30)) {
      std::fprintf(stderr, "gds_probe: the pool is smaller than threads x task size\n");
      return 1;
    }
    for (std::size_t i = 0; i < args.threads; ++i) {
      if (args.raw_pointer) {
        buffers[i] = static_cast<char*>(pool) + i * args.task_size;
      } else {
        buffers[i]      = pool;
        base_offsets[i] = i * args.task_size;
      }
    }
  } else {
    for (auto& buffer : buffers) {
      check(cudaMalloc(&buffer, args.task_size), "cudaMalloc");
    }
  }

  std::atomic<std::size_t> claimed{0};
  std::atomic<std::size_t> done_bytes{0};
  std::atomic<std::size_t> mismatches{0};
  std::atomic<bool> failed{false};
  std::vector<std::atomic<std::int64_t>> started(args.threads);
  auto const begin    = clock_type::now();
  auto const deadline = begin + std::chrono::duration_cast<clock_type::duration>(
                                  std::chrono::duration<double>(args.seconds));

  auto const worker = [&](std::size_t index) {
    check(cudaSetDevice(0), "cudaSetDevice");
    std::vector<std::uint8_t> from_gpu;
    std::vector<std::uint8_t> from_posix;
    int const posix_fd = args.verify ? ::open(files.front().c_str(), O_RDONLY) : -1;
    try {
      for (;;) {
        if (failed.load()) { break; }
        auto const at = claimed.fetch_add(args.task_size);
        if (args.bytes != 0 ? at >= args.bytes : clock_type::now() >= deadline) { break; }
        auto const offset = args.offset + at % span;
        auto size         = std::min(args.task_size, file_size - offset);
        if (args.bytes != 0) { size = std::min(size, args.bytes - at); }

        started[index].store(now_ns());
        auto& handle   = handles[(at / span) % handles.size()];
        auto const got = handle.read(buffers[index], size, offset, base_offsets[index], false);
        started[index].store(0);
        if (got != size) { throw std::runtime_error("short read"); }
        done_bytes.fetch_add(got);

        if (args.verify && at / span == 0) {
          from_gpu.resize(size);
          from_posix.resize(size);
          check(cudaMemcpy(from_gpu.data(),
                           static_cast<char*>(buffers[index]) + base_offsets[index],
                           size,
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy");
          auto const n = ::pread(posix_fd, from_posix.data(), size, static_cast<off_t>(offset));
          if (n != static_cast<ssize_t>(size) ||
              std::memcmp(from_gpu.data(), from_posix.data(), size) != 0) {
            mismatches.fetch_add(1);
          }
        }
      }
    } catch (std::exception const& e) {
      started[index].store(0);
      failed.store(true);
      std::fprintf(stderr, "gds_probe: read failed: %s\n", e.what());
    }
    if (posix_fd >= 0) { ::close(posix_fd); }
  };

  std::vector<std::thread> threads;
  std::atomic<std::size_t> finished{0};
  for (std::size_t i = 0; i < args.threads; ++i) {
    threads.emplace_back([&, i] {
      worker(i);
      finished.fetch_add(1);
    });
  }

  // cuFileRead cannot be cancelled: when one stops returning, leave.
  auto const stall_ns = static_cast<std::int64_t>(args.stall_seconds * 1e9);
  while (finished.load() < args.threads) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto const now = now_ns();
    for (auto const& slot : started) {
      auto const since = slot.load();
      if (since != 0 && now - since > stall_ns) {
        std::printf("STALL a read has not returned for %.0f s; bytes_done=%zu\n",
                    args.stall_seconds,
                    done_bytes.load());
        std::_Exit(70);
      }
    }
  }
  for (auto& thread : threads) {
    thread.join();
  }
  auto const seconds = std::chrono::duration<double>(clock_type::now() - begin).count();

  auto const cufile_after = sirius::io::gds::read_cufile_counters();
  auto const mmap_after   = sirius::io::gds::read_nvidia_fs_mmap_count();
  std::printf("bytes=%zu seconds=%.3f GB/s=%.2f threads=%zu task_size=%zu\n",
              done_bytes.load(),
              seconds,
              static_cast<double>(done_bytes.load()) / 1e9 / seconds,
              args.threads,
              args.task_size);

  bool posix_seen = false;
  if (have_counters && cufile_before && cufile_after) {
    auto const delta = *cufile_after - *cufile_before;
    posix_seen       = delta.posix_reads != 0;
    std::printf(
      "cufile: reads=%llu nvfs_reads=%llu posix_reads=%llu read_bytes=%llu mmaps=%llu "
      "verdict=%s\n",
      static_cast<unsigned long long>(delta.total_reads),
      static_cast<unsigned long long>(delta.nvfs_reads),
      static_cast<unsigned long long>(delta.posix_reads),
      static_cast<unsigned long long>(delta.read_bytes),
      static_cast<unsigned long long>(delta.mmaps),
      std::string(sirius::io::gds::to_string(sirius::io::gds::classify(delta))).c_str());
  } else {
    std::printf("cufile: counters not available from this libcufile\n");
  }
  if (mmap_before && mmap_after) {
    std::printf("nvidia-fs: node Mmap count +%llu\n",
                static_cast<unsigned long long>(*mmap_after - *mmap_before));
  } else {
    std::printf("nvidia-fs: /proc/driver/nvidia-fs/stats not readable\n");
  }
  if (args.verify) { std::printf("verify: mismatched_pieces=%zu\n", mismatches.load()); }

  if (pool != nullptr) {
    cudaFree(pool);
  } else {
    for (auto* buffer : buffers) {
      cudaFree(buffer);
    }
  }
  if (failed.load()) { return 1; }
  return posix_seen || mismatches.load() != 0 ? 2 : 0;
}
