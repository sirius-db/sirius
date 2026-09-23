#include "codegen/jit/nvrtc_compiler.hpp"

#include "compilation_request.hpp"

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

detail::CompilationRequest::CompilationRequest(const std::string& source,
                                               const std::string& entry,
                                               const CompileOptions& opts)
  : source(source), entry(entry)
{
  if (opts.arch_cc <= 0) throw std::invalid_argument("NVRTC architecture must be positive");
  std::snprintf(architecture.data(), architecture.size(), "-arch=sm_%d", opts.arch_cc);
  options[option_count++] = "-std=c++20";
  options[option_count++] = architecture.data();
  options[option_count++] = "-default-device";
  options[option_count++] = "--no-source-include";
  if (const char* path = std::getenv("SIMPATICO_JIT_CCCL_INCLUDE"); path && *path) {
    include_option          = std::string("-I") + path;
    options[option_count++] = include_option.c_str();
  }
  for (int i = 0; i < option_count; ++i)
    option_views[i] = options[i];
}

detail::RequestView detail::CompilationRequest::identity_view() const
{
  return {source, entry, program_name, std::span(option_views.data(), option_count)};
}

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
  return detail::compile_request(detail::CompilationRequest(source, entry_symbol, opts));
}

CompiledKernel detail::compile_request(const CompilationRequest& request)
{
  const auto& source       = request.source;
  const auto& entry_symbol = request.entry;
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
                                    CompilationRequest::program_name,
                                    static_cast<int>(hdr_names.size()),
                                    hdr_sources.data(),
                                    hdr_names.data()));

  nvrtcResult compile_result =
    nvrtcCompileProgram(prog, request.option_count, request.options.data());

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
