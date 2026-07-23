#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The headers the rendered kernels #include are baked into the binary as named
// in-memory headers and fed to NVRTC at compile time, so the JIT needs no
// header tree on disk:
//   - project headers (codegen/decode/rle_block.cuh, codegen/stdint_shim.hpp),
//     embedded by cmake/embed_jit_headers.cmake
//   - the CCCL closure (<cuda/std/...>, <cub/...>), embedded by
//     cmake/embed_cccl_headers.cmake
// This lets a binary distribution JIT-compile with only the driver and the
// nvrtc runtime the extension already links -- no CUDA toolkit headers on disk.
#include "codegen/jit/cccl_embedded_headers.h"
#include "codegen/jit/embedded_headers.h"

namespace codegen::jit {

namespace {

// Optional escape hatch: if the embedded CCCL closure ever lacks a header a
// future renderer needs, SIMPATICO_JIT_CCCL_INCLUDE may point NVRTC at a real
// CCCL dir as an extra -I. Unset by default; not required for correctness.
const char* cccl_include_override()
{
  const char* e = std::getenv("SIMPATICO_JIT_CCCL_INCLUDE");
  return (e != nullptr && *e != '\0') ? e : nullptr;
}

}  // namespace

int arch_cc_for_current_device()
{
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess)
    throw std::runtime_error("arch_cc_for_current_device: cudaGetDevice failed");
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
    throw std::runtime_error("arch_cc_for_current_device: cudaGetDeviceProperties failed");
  return prop.major * 10 + prop.minor;
}

namespace {

[[noreturn]] void throw_nvrtc(const char* api, nvrtcResult r)
{
  std::string msg = "nvrtc ";
  msg += api;
  msg += " failed: ";
  msg += nvrtcGetErrorString(r);
  throw std::runtime_error(msg);
}

[[noreturn]] void throw_cu(const char* api, CUresult r)
{
  const char* name = nullptr;
  const char* desc = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &desc);
  std::string msg = "cu";
  msg += api;
  msg += " failed: ";
  msg += name ? name : "<unknown>";
  if (desc) {
    msg += " (";
    msg += desc;
    msg += ")";
  }
  throw std::runtime_error(msg);
}

#define NVRTC_OR_THROW(call)                         \
  do {                                               \
    nvrtcResult _r = (call);                         \
    if (_r != NVRTC_SUCCESS) throw_nvrtc(#call, _r); \
  } while (0)

#define CU_OR_THROW(call)                        \
  do {                                           \
    CUresult _r = (call);                        \
    if (_r != CUDA_SUCCESS) throw_cu(#call, _r); \
  } while (0)

}  // namespace

CUfunction CompiledKernel::func_for_current_device() const
{
  if (!kern) return nullptr;

  int device_id = 0;
  cudaGetDevice(&device_id);  // runtime API — no driver context required

  {
    std::lock_guard<std::mutex> lock(func_mu_);
    auto it = func_per_dev_.find(device_id);
    if (it != func_per_dev_.end()) return it->second;
  }

  // cuKernelGetFunction binds the kernel to the current device context,
  // which RMM/cuDF sets up correctly per-thread before calling encode/decode.
  CUfunction fn = nullptr;
  if (cuKernelGetFunction(&fn, kern) != CUDA_SUCCESS) return nullptr;

  std::lock_guard<std::mutex> lock(func_mu_);
  func_per_dev_[device_id] = fn;
  return fn;
}

CompiledKernel::~CompiledKernel()
{
  if (library) {
    cuLibraryUnload(library);
    library = nullptr;
    kern    = nullptr;
  }
}

CompiledKernel::CompiledKernel(CompiledKernel&& other) noexcept
  : library(other.library),
    kern(other.kern),
    cubin(std::move(other.cubin)),
    rendered_source(std::move(other.rendered_source))
{
  func_per_dev_ = std::move(other.func_per_dev_);
  other.library = nullptr;
  other.kern    = nullptr;
}

CompiledKernel& CompiledKernel::operator=(CompiledKernel&& other) noexcept
{
  if (this != &other) {
    if (library) cuLibraryUnload(library);
    library         = other.library;
    kern            = other.kern;
    cubin           = std::move(other.cubin);
    rendered_source = std::move(other.rendered_source);
    func_per_dev_   = std::move(other.func_per_dev_);
    other.library   = nullptr;
    other.kern      = nullptr;
  }
  return *this;
}

CompiledKernel compile_plain_kernel(const std::string& source,
                                    const std::string& entry_symbol,
                                    const CompileOptions& opts)
{
  if (source.empty()) { throw std::runtime_error("compile_plain_kernel: empty source"); }
  if (entry_symbol.empty()) {
    throw std::runtime_error("compile_plain_kernel: empty entry_symbol");
  }

  // All headers the rendered kernels #include are supplied to NVRTC as named
  // in-memory headers (embedded in the binary): the project headers plus the
  // full CCCL closure (<cuda/std/...>, <cub/...>). So no header tree is needed
  // on disk at runtime. The include NAMES must match the `#include` strings
  // seen by NVRTC (the renderers' `"codegen/..."` and CCCL's `<...>`).
  std::vector<const char*> hdr_sources;
  std::vector<const char*> hdr_names;
  hdr_sources.reserve(static_cast<std::size_t>(kEmbeddedJitHeaderCount + kCcclEmbeddedHeaderCount));
  hdr_names.reserve(static_cast<std::size_t>(kEmbeddedJitHeaderCount + kCcclEmbeddedHeaderCount));
  for (int i = 0; i < kEmbeddedJitHeaderCount; ++i) {
    hdr_sources.push_back(kEmbeddedJitHeaders[i].source);
    hdr_names.push_back(kEmbeddedJitHeaders[i].name);
  }
  for (int i = 0; i < kCcclEmbeddedHeaderCount; ++i) {
    hdr_sources.push_back(kCcclEmbeddedHeaders[i].source);
    hdr_names.push_back(kCcclEmbeddedHeaders[i].name);
  }

  nvrtcProgram prog = nullptr;
  NVRTC_OR_THROW(nvrtcCreateProgram(&prog,
                                    source.c_str(),
                                    "codegen_jit.cu",
                                    static_cast<int>(hdr_names.size()),
                                    hdr_sources.data(),
                                    hdr_names.data()));

  // Plain CUDA by default — no -default-device. Some rendered sources include
  // shared headers that use c++20 and unannotated constexpr accessors; those
  // opt into -default-device + c++20 via opts.default_device. The leaner path
  // keeps the c++17/no-default-device setup.
  const std::string arch_opt = "-arch=sm_" + std::to_string(opts.arch_cc);

  std::vector<const char*> nvrtc_opts = {
    opts.default_device ? "-std=c++20" : "-std=c++17",
    arch_opt.c_str(),
  };
  // No -I is required (headers are embedded); the env override, when set, adds
  // one as a fallback for a hypothetical embedded-closure gap.
  std::string cccl_inc;
  if (const char* ov = cccl_include_override()) {
    cccl_inc = std::string("-I") + ov;
    nvrtc_opts.push_back(cccl_inc.c_str());
  }
  if (opts.default_device) { nvrtc_opts.push_back("-default-device"); }

  nvrtcResult compile_result =
    nvrtcCompileProgram(prog, static_cast<int>(nvrtc_opts.size()), nvrtc_opts.data());

  // Capture the log unconditionally so warnings on success and errors
  // on failure surface to callers symmetrically.
  std::size_t log_size = 0;
  NVRTC_OR_THROW(nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  if (log_size > 0) {
    log.resize(log_size);
    NVRTC_OR_THROW(nvrtcGetProgramLog(prog, log.data()));
    if (!log.empty() && log.back() == '\0') log.pop_back();
  }

  if (compile_result != NVRTC_SUCCESS) {
    std::string what =
      std::string("nvrtcCompileProgram failed: ") + nvrtcGetErrorString(compile_result);
    nvrtcDestroyProgram(&prog);
    throw CompileError(std::move(what), std::move(log), source);
  }

  // nvrtc emits PTX/SASS directly into the cubin stream.
  std::size_t cubin_size = 0;
  NVRTC_OR_THROW(nvrtcGetCUBINSize(prog, &cubin_size));
  if (cubin_size == 0) {
    nvrtcDestroyProgram(&prog);
    throw CompileError("nvrtc produced empty cubin", std::move(log), source);
  }
  std::vector<char> cubin(cubin_size);
  NVRTC_OR_THROW(nvrtcGetCUBIN(prog, cubin.data()));
  NVRTC_OR_THROW(nvrtcDestroyProgram(&prog));

  if (const char* dump = std::getenv("CODEGEN_JIT_DUMP_CUBIN")) {
    FILE* fp = std::fopen(dump, "wb");
    if (fp) {
      std::fwrite(cubin.data(), 1, cubin.size(), fp);
      std::fclose(fp);
      std::fprintf(stderr, "[codegen_jit] dumped %zu byte cubin to %s\n", cubin.size(), dump);
    }
  }

  return load_kernel_from_cubin(std::move(cubin), entry_symbol, source);
}

CompiledKernel load_kernel_from_cubin(std::vector<char> cubin,
                                      const std::string& entry_symbol,
                                      std::string rendered_source)
{
  if (cubin.empty()) { throw std::runtime_error("load_kernel_from_cubin: empty cubin"); }

  // cuLibraryLoadData is multi-device and does not require a specific context.
  // cuLibraryGetKernel returns a device-independent CUkernel handle; the
  // per-device CUfunction is derived lazily via func_for_current_device().
  CUlibrary lib = nullptr;
  CU_OR_THROW(cuLibraryLoadData(&lib, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  CUkernel kern = nullptr;
  CUresult r    = cuLibraryGetKernel(&kern, lib, entry_symbol.c_str());
  if (r != CUDA_SUCCESS) {
    cuLibraryUnload(lib);
    throw_cu("LibraryGetKernel", r);
  }

  CompiledKernel out;
  out.library         = lib;
  out.kern            = kern;
  out.cubin           = std::move(cubin);
  out.rendered_source = std::move(rendered_source);
  return out;
}

}  // namespace codegen::jit
