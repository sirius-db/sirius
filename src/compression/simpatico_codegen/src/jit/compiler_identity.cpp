#include "codegen/jit/compiler_identity.h"

#include "codegen/jit/cccl_embedded_headers.h"
#include "codegen/jit/embedded_headers.h"
#include "compilation_request.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <link.h>
#include <nvrtc.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace codegen::jit::detail {
namespace {

struct LoadedLibrary {
  std::string path;
  uintptr_t base;
};

bool loaded_identity(const LoadedLibrary& library, Digest& result)
{
  // Check that the pathname still denotes the mapped inode. An atomic package
  // replacement while this process is alive must not identify the replacement
  // file as the compiler already loaded into this process.
  struct stat metadata{};
  if (::stat(library.path.c_str(), &metadata) != 0) return false;
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    unsigned long begin = 0, end = 0, inode = 0;
    unsigned int major = 0, minor = 0;
    if (std::sscanf(
          line.c_str(), "%lx-%lx %*s %*s %x:%x %lu", &begin, &end, &major, &minor, &inode) != 5)
      continue;
    if (library.base >= begin && library.base < end) {
      if (metadata.st_ino != inode || metadata.st_dev != makedev(major, minor) ||
          !file_identity(library.path, result))
        return false;
      struct stat after{};
      return ::stat(library.path.c_str(), &after) == 0 && after.st_ino == metadata.st_ino &&
             after.st_dev == metadata.st_dev;
    }
  }
  return false;
}

bool shared_compiler_identity(std::string& result)
{
  Dl_info nvrtc{};
  if (!dladdr(reinterpret_cast<const void*>(&nvrtcVersion), &nvrtc) || !nvrtc.dli_fname)
    return false;
  if (!std::filesystem::path(nvrtc.dli_fname).filename().string().starts_with("libnvrtc.so"))
    return false;

  // Builtins are dlopened lazily. A preprocessor error loads them without
  // generating a kernel or using a synthetic compiler-version assumption.
  nvrtcProgram probe = nullptr;
  if (nvrtcCreateProgram(
        &probe, "#error simpatico_identity_probe\n", "identity_probe.cu", 0, nullptr, nullptr) !=
      NVRTC_SUCCESS)
    return false;
  const auto status = nvrtcCompileProgram(probe, 0, nullptr);
  nvrtcDestroyProgram(&probe);
  if (status != NVRTC_ERROR_COMPILATION) return false;

  std::vector<LoadedLibrary> builtins;
  const int scanned = dl_iterate_phdr(
    [](dl_phdr_info* info, std::size_t, void* data) {
      try {
        if (info->dlpi_name && std::filesystem::path(info->dlpi_name)
                                 .filename()
                                 .string()
                                 .starts_with("libnvrtc-builtins.so")) {
          static_cast<std::vector<LoadedLibrary>*>(data)->push_back(
            {info->dlpi_name, static_cast<uintptr_t>(info->dlpi_addr)});
        }
      } catch (...) {
        return 1;  // Never unwind through the loader's C callback.
      }
      return 0;
    },
    &builtins);
  // Multiple loaded builtins versions cannot be associated with this NVRTC
  // through a public API. Keep compilation working but decline persistence.
  if (scanned != 0 || builtins.size() != 1) return false;
  Digest compiler{}, builtin{};
  if (!loaded_identity({nvrtc.dli_fname, reinterpret_cast<uintptr_t>(nvrtc.dli_fbase)}, compiler) ||
      !loaded_identity(builtins.front(), builtin))
    return false;
  result = "shared-v1:nvrtc:" + hex_digest(compiler) + ":builtins:" + hex_digest(builtin);
  return true;
}

}  // namespace

const CacheEnvironment& cache_environment()
{
  static const CacheEnvironment environment = [] {
    const auto start = std::chrono::steady_clock::now();
    CacheEnvironment value;
    try {
      std::string compiler  = kStaticCompilerIdentity;
      int driver            = 0;
      const bool identified = !compiler.empty() || shared_compiler_identity(compiler);
      if (identified && cuDriverGetVersion(&driver) == CUDA_SUCCESS && driver > 0) {
        value.identity   = environment_identity({"embedded-libcudf",
                                                 kEmbeddedJitHeadersIdentity,
                                                 kCcclEmbeddedHeadersIdentity,
                                                 compiler,
                                                 CUDART_VERSION,
                                                 static_cast<uint32_t>(driver)});
        value.persistent = true;
      }
    } catch (const std::exception&) {
      // Discovery is an optimization prerequisite, not a compilation prerequisite.
    }
    if (std::getenv("SIMPATICO_JIT_STATS")) {
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
      std::fprintf(stderr,
                   "[jit-identity] persistent=%d discovery_us=%lld%s\n",
                   value.persistent,
                   static_cast<long long>(us),
                   value.persistent ? "" : " reason=compiler-or-driver-unidentified");
    }
    return value;
  }();
  return environment;
}

}  // namespace codegen::jit::detail
