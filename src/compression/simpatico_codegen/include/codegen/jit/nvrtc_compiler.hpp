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
  int arch_cc             = 80;
  std::string kernel_name = "codegen_jit_kernel";
  bool default_device     = false;
};

struct CompiledKernel {
  CUlibrary library = nullptr;
  // Device-independent kernel handle (valid on all devices sharing this library).
  // Use func_for_current_device() to get a context-specific CUfunction for launch.
  CUkernel kern = nullptr;
  std::vector<char> cubin;
  std::string rendered_source;
  unsigned block_dim_x      = 1;
  unsigned shared_mem_bytes = 0;

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

// Ensure a CUDA driver context is current on this thread. Idempotent; safe to
// call from multiple threads. Throws std::runtime_error if the driver cannot
// be initialised (no GPU, missing libcuda, etc.).
void ensure_cuda_context();

}  // namespace codegen::jit
