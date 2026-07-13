// Owned compiled-kernel handle for plain-CUDA nvrtc JIT (encode + decode renderers).
#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

typedef struct CUlib_st* CUlibrary;
typedef struct CUfunc_st* CUfunction;
typedef struct CUkern_st* CUkernel;

namespace codegen::jit {

struct CompileError : std::runtime_error {
  std::string log;
  std::string source;

  CompileError(std::string what_arg, std::string log_, std::string src_)
    : std::runtime_error(std::move(what_arg)), log(std::move(log_)), source(std::move(src_))
  {
  }
};

struct CompileOptions {
  int arch_cc         = 0;  // must be set by caller via arch_cc_for_current_device()
  bool default_device = false;
};

// Returns major*10+minor for the current CUDA device (e.g. 90 for H100).
// Throws std::runtime_error if the device cannot be queried.
int arch_cc_for_current_device();

struct CompiledKernel {
  CUlibrary library = nullptr;
  // Device-independent kernel handle (valid on all devices sharing this library).
  // Use func_for_current_device() to get a context-specific CUfunction for launch.
  CUkernel kern = nullptr;
  std::vector<char> cubin;
  std::string rendered_source;

  // Returns the CUfunction for the current CUDA device, deriving and caching it
  // on first call per device. Returns nullptr if kern is null or derivation fails.
  CUfunction func_for_current_device() const;

  CompiledKernel() = default;
  ~CompiledKernel();
  CompiledKernel(CompiledKernel&&) noexcept;
  CompiledKernel& operator=(CompiledKernel&&) noexcept;
  CompiledKernel(const CompiledKernel&)            = delete;
  CompiledKernel& operator=(const CompiledKernel&) = delete;

 private:
  mutable std::mutex func_mu_;
  mutable std::unordered_map<int, CUfunction> func_per_dev_;
};

// Compile a raw CUDA-C++ source string via nvrtc and load the resulting cubin.
// Throws CompileError on nvrtc rejection (with .log and .source) and
// std::runtime_error on driver-API failures.
CompiledKernel compile_plain_kernel(const std::string& source,
                                    const std::string& entry_symbol,
                                    const CompileOptions& opts = {});

// Load a CompiledKernel from an already-compiled cubin (skips nvrtc).
// rendered_source is stored for diagnostics only; it need not match the cubin.
// Throws std::runtime_error on driver-API failure so callers can fall back.
CompiledKernel load_kernel_from_cubin(std::vector<char> cubin,
                                      const std::string& entry_symbol,
                                      std::string rendered_source = {});

}  // namespace codegen::jit
