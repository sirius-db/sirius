#include "codegen/jit/nvrtc_compiler.hpp"

#include "codegen/util/cuda_check.hpp"
#include "compilation_request.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <nvrtc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The headers the rendered kernels #include are baked into the binary as named
// in-memory headers and fed to NVRTC at compile time, so the JIT needs no
// header tree on disk:
//   - project headers (codegen/decode/rle_block.cuh, codegen/stdint_shim.hpp),
//     embedded by cmake/embed_jit_headers.cmake
//   - the complete CCCL library bundle, embedded by
//     cmake/embed_cccl_headers.cmake
// This lets a binary distribution JIT-compile with only the driver and the
// nvrtc runtime the extension already links -- no CUDA toolkit headers on disk.
#include "codegen/jit/cccl_embedded_headers.h"
#include "codegen/jit/embedded_headers.h"

namespace codegen::jit {

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

#define NVRTC_OR_THROW(call)                         \
  do {                                               \
    nvrtcResult _r = (call);                         \
    if (_r != NVRTC_SUCCESS) throw_nvrtc(#call, _r); \
  } while (0)

struct StageCounters {
  std::atomic<std::uint64_t> compiles{0};
  std::atomic<std::uint64_t> source_bytes{0};
  std::atomic<std::uint64_t> create_program_us{0};
  std::atomic<std::uint64_t> compile_program_us{0};
  std::atomic<std::uint64_t> get_cubin_us{0};
  std::atomic<std::uint64_t> loads{0};
  std::atomic<std::uint64_t> load_us{0};
};

StageCounters g_stage_counters;

/** @brief Adds the wall time of its own lifetime, in microseconds, to a stage counter. */
class StageTimer {
 public:
  explicit StageTimer(std::atomic<std::uint64_t>& total_us) noexcept : total_us_(total_us) {}
  ~StageTimer()
  {
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    total_us_.fetch_add(static_cast<std::uint64_t>(
                          std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()),
                        std::memory_order_relaxed);
  }
  StageTimer(const StageTimer&)            = delete;
  StageTimer& operator=(const StageTimer&) = delete;

 private:
  std::atomic<std::uint64_t>& total_us_;
  std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
};

}  // namespace

CompileStats compile_stats() noexcept
{
  const auto read = [](const std::atomic<std::uint64_t>& counter) {
    return counter.load(std::memory_order_relaxed);
  };
  return {read(g_stage_counters.compiles),
          read(g_stage_counters.source_bytes),
          read(g_stage_counters.create_program_us),
          read(g_stage_counters.compile_program_us),
          read(g_stage_counters.get_cubin_us),
          read(g_stage_counters.loads),
          read(g_stage_counters.load_us)};
}

const detail::Digest& detail::cache_environment()
{
  static const Digest identity = [] {
    int major = 0, minor = 0;
    NVRTC_OR_THROW(nvrtcVersion(&major, &minor));
    // Both linkage modes use the reported version. Same-version compiler
    // patches intentionally retain reuse; header contents have their own digests.
    return environment_identity({kEmbeddedJitHeadersIdentity,
                                 kCcclEmbeddedHeadersIdentity,
                                 static_cast<uint32_t>(major),
                                 static_cast<uint32_t>(minor)});
  }();
  return identity;
}

CUfunction CompiledKernel::func_for_current_device() const
{
  if (!kern_) return nullptr;

  // cuKernelGetFunction binds the kernel to the calling thread's current context,
  // and cuLaunchKernel rejects the result from any other context
  // (CUDA_ERROR_INVALID_HANDLE). Threads on one device can hold different
  // contexts, so a device-keyed cache would hand every thread but the first a
  // function it cannot launch.
  CUcontext ctx = nullptr;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) return nullptr;
  if (ctx == nullptr) {
    // A thread that never touched the runtime has no context; make the current
    // device's primary context current instead of deriving against none.
    cudaFree(nullptr);
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) return nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(func_mu_);
    auto it = func_per_ctx_.find(ctx);
    if (it != func_per_ctx_.end()) return it->second;
  }

  CUfunction fn = nullptr;
  if (cuKernelGetFunction(&fn, kern_) != CUDA_SUCCESS) return nullptr;

  std::lock_guard<std::mutex> lock(func_mu_);
  func_per_ctx_[ctx] = fn;
  return fn;
}

CompiledModule::CompiledModule(ConstructionKey,
                               std::vector<char> cubin,
                               std::string rendered_source)
  : cubin_(std::move(cubin)), rendered_source_(std::move(rendered_source))
{
  if (cubin_.empty()) { throw std::runtime_error("CompiledModule::load: empty cubin"); }

  // cuLibraryLoadData is multi-device and does not require a specific context.
  {
    const StageTimer timer(g_stage_counters.load_us);
    simpatico::throw_if_cu_error(
      cuLibraryLoadData(&library_, cubin_.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
      "cuLibraryLoadData");
  }
  g_stage_counters.loads.fetch_add(1, std::memory_order_relaxed);
}

CompiledModule::~CompiledModule() { cuLibraryUnload(library_); }

std::shared_ptr<const CompiledModule> CompiledModule::load(std::vector<char> cubin,
                                                           std::string rendered_source)
{
  return std::make_shared<CompiledModule>(
    ConstructionKey{}, std::move(cubin), std::move(rendered_source));
}

std::shared_ptr<const CompiledKernel> CompiledModule::kernel(const std::string& entry_symbol) const
{
  const std::lock_guard<std::mutex> lock(kernels_mu_);
  auto it = kernels_.find(entry_symbol);
  if (it == kernels_.end()) {
    // cuLibraryGetKernel returns a device-independent CUkernel handle; the
    // per-device CUfunction is derived lazily via func_for_current_device().
    CUkernel kern = nullptr;
    simpatico::throw_if_cu_error(cuLibraryGetKernel(&kern, library_, entry_symbol.c_str()),
                                 ("cuLibraryGetKernel(" + entry_symbol + ")").c_str());
    it = kernels_.try_emplace(entry_symbol, CompiledKernel::ConstructionKey{}, *this, kern).first;
  }
  return {shared_from_this(), &it->second};
}

std::shared_ptr<const CompiledKernel> compile_plain_kernel(const std::string& source,
                                                           const std::string& entry_symbol,
                                                           const CompileOptions& opts)
{
  return detail::compile_request(detail::CompilationRequest(source, opts))->kernel(entry_symbol);
}

std::shared_ptr<const CompiledModule> detail::compile_request(const CompilationRequest& request)
{
  const auto& source = request.source;
  if (source.empty()) { throw std::runtime_error("compile_plain_kernel: empty source"); }

  // All headers the rendered kernels #include are supplied to NVRTC as named
  // in-memory headers (embedded in the binary): the project headers plus the
  // complete CCCL bundle. So no header tree is needed
  // on disk at runtime. The include NAMES must match the `#include` strings
  // seen by NVRTC (the renderers' `"codegen/..."` and CCCL's `<...>`).
  // A request that includes none of them is given none: NVRTC takes a few
  // milliseconds to register the bundle whether or not the source uses it.
  std::vector<const char*> hdr_sources;
  std::vector<const char*> hdr_names;
  if (request.embedded_headers) {
    hdr_sources.reserve(
      static_cast<std::size_t>(kEmbeddedJitHeaderCount + kCcclEmbeddedHeaderCount));
    hdr_names.reserve(static_cast<std::size_t>(kEmbeddedJitHeaderCount + kCcclEmbeddedHeaderCount));
    for (int i = 0; i < kEmbeddedJitHeaderCount; ++i) {
      hdr_sources.push_back(kEmbeddedJitHeaders[i].source);
      hdr_names.push_back(kEmbeddedJitHeaders[i].name);
    }
    for (int i = 0; i < kCcclEmbeddedHeaderCount; ++i) {
      hdr_sources.push_back(kCcclEmbeddedHeaders[i].source);
      hdr_names.push_back(kCcclEmbeddedHeaders[i].name);
    }
  }

  nvrtcProgram prog = nullptr;
  {
    const StageTimer timer(g_stage_counters.create_program_us);
    NVRTC_OR_THROW(nvrtcCreateProgram(&prog,
                                      source.c_str(),
                                      CompilationRequest::program_name,
                                      static_cast<int>(hdr_names.size()),
                                      hdr_sources.data(),
                                      hdr_names.data()));
  }

  nvrtcResult compile_result = NVRTC_SUCCESS;
  {
    const StageTimer timer(g_stage_counters.compile_program_us);
    compile_result =
      nvrtcCompileProgram(prog, static_cast<int>(request.options.size()), request.options.data());
  }

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
  std::vector<char> cubin;
  {
    const StageTimer timer(g_stage_counters.get_cubin_us);
    std::size_t cubin_size = 0;
    NVRTC_OR_THROW(nvrtcGetCUBINSize(prog, &cubin_size));
    if (cubin_size == 0) {
      nvrtcDestroyProgram(&prog);
      throw CompileError("nvrtc produced empty cubin", std::move(log), source);
    }
    cubin.resize(cubin_size);
    NVRTC_OR_THROW(nvrtcGetCUBIN(prog, cubin.data()));
    NVRTC_OR_THROW(nvrtcDestroyProgram(&prog));
  }
  g_stage_counters.compiles.fetch_add(1, std::memory_order_relaxed);
  g_stage_counters.source_bytes.fetch_add(source.size(), std::memory_order_relaxed);

  if (const char* dump = std::getenv("CODEGEN_JIT_DUMP_CUBIN")) {
    FILE* fp = std::fopen(dump, "wb");
    if (fp) {
      std::fwrite(cubin.data(), 1, cubin.size(), fp);
      std::fclose(fp);
      std::fprintf(stderr, "[codegen_jit] dumped %zu byte cubin to %s\n", cubin.size(), dump);
    }
  }

  return CompiledModule::load(std::move(cubin), source);
}

}  // namespace codegen::jit
