#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

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

std::string cu_err_str(CUresult r)
{
  const char* desc = nullptr;
  cuGetErrorString(r, &desc);
  return desc ? std::string(desc) : ("CUresult=" + std::to_string(int{r}));
}

#define CU_OR_THROW(call)                                                   \
  do {                                                                      \
    CUresult _r = (call);                                                   \
    if (_r != CUDA_SUCCESS) {                                               \
      throw std::runtime_error(std::string(#call) + ": " + cu_err_str(_r)); \
    }                                                                       \
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

}  // namespace codegen::jit
