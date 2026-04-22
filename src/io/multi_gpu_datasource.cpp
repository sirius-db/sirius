/*
 * Copyright 2026, Sirius Contributors.
 *
 * Multi-GPU smoke test for sirius_datasource.
 *
 * One shared `uring_ioctx` drives parquet reads simultaneously on two GPUs.
 * Files are drawn from the same TPC-H lineitem directory used by
 * stress_prefetching.  The goal is to verify that:
 *   - Pinned bounce buffers allocated inside uring_reactor are reachable
 *     from cudaMemcpyAsync bound to any device's stream (UVA guarantees).
 *   - Device memory for cudf output tables is allocated on the correct
 *     per-thread device via the per-device RMM resource.
 *   - Two concurrent readers on different GPUs do not trample each other
 *     through the shared ioctx.
 */

#include "io/uring/uring_ioctx.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static void cuda_check(cudaError_t e, char const *what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

static constexpr size_t N_FILES_TOTAL = 10; // from stress_prefetching

int main(int argc, char **argv) {
  int n_devices = 0;
  cuda_check(cudaGetDeviceCount(&n_devices), "cudaGetDeviceCount");
  if (n_devices < 2) {
    std::cerr << "multi_gpu_datasource: need >= 2 GPUs, found " << n_devices
              << "\n";
    return 1;
  }

  size_t files_per_gpu = argc > 1 ? std::stoul(argv[1]) : N_FILES_TOTAL / 2;

  fs::path dir = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/root") /
                 "Documents/tpch/sf200/parquet/lineitem";
  std::vector<fs::path> paths;
  for (auto const &de : fs::directory_iterator(dir))
    if (de.is_regular_file())
      paths.push_back(de.path());
  std::sort(paths.begin(), paths.end());
  if (paths.size() < 2 * files_per_gpu)
    throw std::runtime_error("need >= " + std::to_string(2 * files_per_gpu) +
                             " files in " + dir.string());
  paths.resize(2 * files_per_gpu);

  // Split alternating across the two GPUs so both see the same distribution.
  std::array<std::vector<fs::path>, 2> per_gpu;
  for (size_t i = 0; i < paths.size(); ++i)
    per_gpu[i % 2].push_back(paths[i]);

  std::cout << "multi_gpu_datasource\n"
            << "  dir          : " << dir << "\n"
            << "  gpus         : 2\n"
            << "  files/gpu    : " << files_per_gpu << "\n\n";

  spdlog::set_level(spdlog::level::warn);

  // One shared ioctx for both GPUs.  Bounce slots are cudaHostAlloc'd with
  // cudaHostAllocPortable so every CUDA context can read them.
  auto io_ctx = std::make_shared<sirius::io::uring_ioctx>();

  // Per-device async memory resources.  Each worker thread sets its device
  // and installs this as the per-device resource so cudf allocates on the
  // right GPU.
  std::array<std::unique_ptr<rmm::mr::cuda_async_memory_resource>, 2> mrs;

  std::array<std::atomic<size_t>, 2> rows_read{};
  std::array<std::atomic<size_t>, 2> bytes_read{};
  std::array<std::atomic<double>, 2> elapsed_s{};
  std::array<std::exception_ptr, 2> errors{};

  auto worker = [&](int gpu) {
    cudaStream_t raw_stream = nullptr;
    try {
      cuda_check(cudaSetDevice(gpu), "cudaSetDevice");
      mrs[gpu] = std::make_unique<rmm::mr::cuda_async_memory_resource>();
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{gpu},
                                       mrs[gpu].get());

      // Explicit per-device stream — cudf's get_default_stream() is a single
      // singleton bound to whichever device initialised it, so a second
      // thread on a different device must supply its own stream to avoid
      // cross-device stream / context errors inside cudf/thrust.
      cuda_check(cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking),
                 "cudaStreamCreate");
      rmm::cuda_stream_view stream{raw_stream};

      auto t0 = std::chrono::steady_clock::now();
      size_t rows = 0;
      size_t bytes = 0;
      for (auto const &path : per_gpu[gpu]) {
        auto io_obj =
            std::make_unique<sirius::io::uring_io_object>(path.string());
        bytes += io_obj->size();
        auto ds = io_ctx->make_datasource(std::move(io_obj));

        std::vector<std::unique_ptr<cudf::io::datasource>> sources;
        sources.push_back(std::move(ds));

        auto opts = cudf::io::parquet_reader_options::builder().build();
        auto tbl = cudf::io::read_parquet(std::move(sources),
                                          /*metadatas=*/{}, opts, stream);
        rows += tbl.tbl->num_rows();
      }
      cuda_check(cudaStreamSynchronize(raw_stream), "cudaStreamSynchronize");
      auto t1 = std::chrono::steady_clock::now();
      elapsed_s[gpu].store(std::chrono::duration<double>(t1 - t0).count());
      rows_read[gpu].store(rows);
      bytes_read[gpu].store(bytes);
    } catch (...) {
      errors[gpu] = std::current_exception();
    }
    if (raw_stream)
      cudaStreamDestroy(raw_stream);
  };

  std::array<std::thread, 2> threads{std::thread(worker, 0),
                                     std::thread(worker, 1)};
  for (auto &t : threads)
    t.join();

  io_ctx->shutdown();

  int rc = 0;
  for (int g = 0; g < 2; ++g) {
    if (errors[g]) {
      std::cerr << "GPU " << g << ": exception — ";
      try {
        std::rethrow_exception(errors[g]);
      } catch (std::exception const &e) {
        std::cerr << e.what() << "\n";
      }
      rc = 1;
      continue;
    }
    double elapsed = elapsed_s[g].load();
    double gib = bytes_read[g].load() / double(1ULL << 30);
    std::cout << "GPU " << g << " : " << per_gpu[g].size() << " files, "
              << rows_read[g].load() << " rows, " << gib << " GiB, "
              << elapsed << " s (" << (gib / elapsed) << " GiB/s)\n";
  }
  return rc;
}
