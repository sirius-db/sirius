// A fresh process is essential: the production cache location is process-cached.
#include "codegen/jit/kernel_cache.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <nvrtc.h>

#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace jit = codegen::jit;

static const std::string embedded_source = R"(
#include "codegen/stdint_shim.hpp"
#include "codegen/decode/rle_block.cuh"
#include <cuda/std/type_traits>
#ifndef SIMPATICO_CACHE_TEST_VALUE
#define SIMPATICO_CACHE_TEST_VALUE 1
#endif
extern "C" __global__ void cache_test(int* out) { *out = SIMPATICO_CACHE_TEST_VALUE; }
)";
static const std::string external_source = R"(
#include <cache_fallback.cuh>
extern "C" __global__ void cache_test(int* out) { *out = CACHE_FALLBACK_VALUE; }
)";

static int launch(const jit::CompiledKernel* kernel)
{
  int* device = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device), sizeof(int)) != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed");
  void* arguments[] = {&device};
  const auto status = cuLaunchKernel(
    kernel->func_for_current_device(), 1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr);
  int result        = -1;
  const auto copied = cudaMemcpy(&result, device, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (status != CUDA_SUCCESS || copied != cudaSuccess)
    throw std::runtime_error("kernel launch failed");
  return result;
}

int main(int argc, char** argv)
{
  try {
    const std::string mode = argc > 1 ? argv[1] : "embedded";
    if (mode == "clear") {
      jit::clear_jit_disk_cache();
      std::puts("RESULT cleared");
      return 0;
    }
    if (mode == "compiler-path") {
      Dl_info library{};
      if (!dladdr(reinterpret_cast<const void*>(&nvrtcVersion), &library) || !library.dli_fname)
        throw std::runtime_error("cannot resolve compiler path");
      std::printf("NVRTC %s\n", library.dli_fname);
      return 0;
    }
    if (cudaSetDevice(0) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed");
    jit::CompileOptions options{jit::arch_cc_for_current_device()};
    auto& cache = jit::KernelCache::instance();
    if (mode == "threads") {
      constexpr int count = 6;
      std::barrier start(count);
      std::vector<const jit::CompiledKernel*> kernels(count);
      std::vector<std::exception_ptr> errors(count);
      std::vector<std::thread> workers;
      for (int i = 0; i < count; ++i) {
        workers.emplace_back([&, i] {
          // Every thread must reach the barrier, including error paths.
          const auto status = cudaSetDevice(0);
          start.arrive_and_wait();
          try {
            if (status != cudaSuccess) throw std::runtime_error("thread cudaSetDevice failed");
            kernels[i] = cache.get_or_compile_plain(embedded_source, "cache_test", options);
            if (launch(kernels[i]) != 1) throw std::runtime_error("thread kernel result");
          } catch (...) {
            errors[i] = std::current_exception();
          }
        });
      }
      for (auto& worker : workers)
        worker.join();
      for (int i = 0; i < count; ++i) {
        if (errors[i]) std::rethrow_exception(errors[i]);
        if (kernels[i] != kernels[0]) throw std::runtime_error("thread deduplication failed");
      }
      if (kernels[0] != cache.get_or_compile_plain(embedded_source, "cache_test", options))
        throw std::runtime_error("post-thread memory lookup failed");
      std::puts("RESULT value=1");
      return 0;
    }
    if (mode == "cycle") {
      const char* path = std::getenv("SIMPATICO_JIT_CCCL_INCLUDE");
      if (!path) throw std::runtime_error("cycle needs an isolated override directory");
      const auto header = std::filesystem::path(path) / "cache_fallback.cuh";
      auto* first       = cache.get_or_compile_plain(external_source, "cache_test", options);
      if (launch(first) != 41) throw std::runtime_error("first external value");
      std::ofstream(header) << "#define CACHE_FALLBACK_VALUE 42\n";
      auto* second = cache.get_or_compile_plain(external_source, "cache_test", options);
      if (first == second || launch(second) != 42 || launch(first) != 41)
        throw std::runtime_error("external change or retained handle lifetime");
      unsetenv("SIMPATICO_JIT_CCCL_INCLUDE");
      auto* third = cache.get_or_compile_plain(embedded_source, "cache_test", options);
      if (launch(third) != 1 ||
          third != cache.get_or_compile_plain(embedded_source, "cache_test", options))
        throw std::runtime_error("reuse after removing override");
      std::puts("RESULT cycle=41,42,41,1");
      return 0;
    }
    const auto& source = mode == "external" ? external_source : embedded_source;
    auto* first        = cache.get_or_compile_plain(source, "cache_test", options);
    const int value    = launch(first);
    auto* second       = cache.get_or_compile_plain(source, "cache_test", options);
    if (launch(second) != value) throw std::runtime_error("second launch changed result");
    const char* override = std::getenv("SIMPATICO_JIT_CCCL_INCLUDE");
    const bool external  = override && *override;
    if ((first == second) == external) throw std::runtime_error("unexpected memory reuse");
    std::printf("RESULT value=%d\n", value);
    return 0;
  } catch (const jit::CompileError& error) {
    std::fprintf(stderr, "%s\n%s\n", error.what(), error.log.c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
  }
  return 1;
}
