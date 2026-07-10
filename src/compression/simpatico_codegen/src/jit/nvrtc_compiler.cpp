#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace codegen::jit {

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

void ensure_cuda_context()
{
  CU_OR_THROW(cuInit(0));

  // Use the runtime API to identify the current device so we retain its
  // primary context rather than always forcing device 0.
  int device_id = 0;
  cudaGetDevice(&device_id);  // ignore error — falls back to ordinal 0

  static std::mutex mu;
  static std::unordered_map<int, CUcontext> ctx_by_device;

  std::lock_guard<std::mutex> lock(mu);
  auto it = ctx_by_device.find(device_id);
  if (it == ctx_by_device.end()) {
    CUdevice dev = 0;
    CU_OR_THROW(cuDeviceGet(&dev, device_id));
    CUcontext ctx = nullptr;
    CU_OR_THROW(cuDevicePrimaryCtxRetain(&ctx, dev));
    ctx_by_device[device_id] = ctx;
    it                       = ctx_by_device.find(device_id);
  }
  CU_OR_THROW(cuCtxSetCurrent(it->second));
}

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
    rendered_source(std::move(other.rendered_source)),
    block_dim_x(other.block_dim_x),
    shared_mem_bytes(other.shared_mem_bytes)
{
  func_per_dev_ = std::move(other.func_per_dev_);
  other.library = nullptr;
  other.kern    = nullptr;
}

CompiledKernel& CompiledKernel::operator=(CompiledKernel&& other) noexcept
{
  if (this != &other) {
    if (library) cuLibraryUnload(library);
    library          = other.library;
    kern             = other.kern;
    cubin            = std::move(other.cubin);
    rendered_source  = std::move(other.rendered_source);
    block_dim_x      = other.block_dim_x;
    shared_mem_bytes = other.shared_mem_bytes;
    func_per_dev_    = std::move(other.func_per_dev_);
    other.library    = nullptr;
    other.kern       = nullptr;
  }
  return *this;
}

}  // namespace codegen::jit
