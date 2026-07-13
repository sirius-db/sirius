#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>

#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// NVRTC -I paths: set by CMake via target_include_directories on
// codegen_jit; these macros are optional overrides for tests.
#ifndef CODEGEN_JIT_CUDA_INCLUDE
#error "CODEGEN_JIT_CUDA_INCLUDE must be set by CMake (conda CUDA toolkit include)"
#endif
#ifndef CODEGEN_JIT_CCCL_INCLUDE
#error "CODEGEN_JIT_CCCL_INCLUDE must be set by CMake (conda CCCL include)"
#endif
#ifndef CODEGEN_JIT_PROJECT_INCLUDE
#error "CODEGEN_JIT_PROJECT_INCLUDE must be set by CMake (project include)"
#endif

namespace codegen::jit {

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

CompiledKernel compile_plain_kernel(const std::string& source,
                                    const std::string& entry_symbol,
                                    const CompileOptions& opts)
{
  if (source.empty()) { throw std::runtime_error("compile_plain_kernel: empty source"); }
  if (entry_symbol.empty()) {
    throw std::runtime_error("compile_plain_kernel: empty entry_symbol");
  }

  nvrtcProgram prog = nullptr;
  NVRTC_OR_THROW(
    nvrtcCreateProgram(&prog, source.c_str(), "codegen_jit_encode.cu", 0, nullptr, nullptr));

  // Plain CUDA — no -default-device, no _CODEGEN_CODEGEN_JIT.
  // The encode kernels include a small set of CUDA headers directly
  // (cstdint, climits, cuda_runtime.h) — we still pass the same -I
  // set so they can pull in cuda/std/* if a future helper wants to.
  const std::string arch_opt = "-arch=sm_" + std::to_string(opts.arch_cc);
  const std::string cuda_inc = std::string("-I") + CODEGEN_JIT_CUDA_INCLUDE;
  const std::string cccl_inc = std::string("-I") + CODEGEN_JIT_CCCL_INCLUDE;
  const std::string proj_inc = std::string("-I") + CODEGEN_JIT_PROJECT_INCLUDE;

  // The rendered decode source includes shared headers (tree.hpp via
  // rle_block.cuh) that use c++20 and unannotated constexpr accessors;
  // it opts into -default-device + c++20 (mirroring the JIT codegen path).
  // Plain encode keeps the leaner c++17/no-default-device setup.
  std::vector<const char*> nvrtc_opts = {
    opts.default_device ? "-std=c++20" : "-std=c++17",
    arch_opt.c_str(),
    cuda_inc.c_str(),
    cccl_inc.c_str(),
    proj_inc.c_str(),
  };
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

  // Plain cubin — nvrtc emits PTX/SASS directly into the cubin stream.
  std::size_t cubin_size = 0;
  NVRTC_OR_THROW(nvrtcGetCUBINSize(prog, &cubin_size));
  if (cubin_size == 0) {
    nvrtcDestroyProgram(&prog);
    throw CompileError("nvrtc produced empty cubin", std::move(log), source);
  }
  std::vector<char> cubin(cubin_size);
  NVRTC_OR_THROW(nvrtcGetCUBIN(prog, cubin.data()));
  NVRTC_OR_THROW(nvrtcDestroyProgram(&prog));

  if (const char* dump = std::getenv("CODEGEN_JIT_DUMP_ENCODE_CUBIN")) {
    FILE* fp = std::fopen(dump, "wb");
    if (fp) {
      std::fwrite(cubin.data(), 1, cubin.size(), fp);
      std::fclose(fp);
      std::fprintf(
        stderr, "[codegen_jit] dumped %zu byte encode cubin to %s\n", cubin.size(), dump);
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
