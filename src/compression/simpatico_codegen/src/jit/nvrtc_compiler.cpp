#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>

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
  static CUcontext primary = nullptr;
  static bool initialised  = false;

  CU_OR_THROW(cuInit(0));
  if (!initialised) {
    CUdevice dev = 0;
    CU_OR_THROW(cuDeviceGet(&dev, 0));
    CU_OR_THROW(cuDevicePrimaryCtxRetain(&primary, dev));
    initialised = true;
  }
  CU_OR_THROW(cuCtxSetCurrent(primary));
}

CompiledKernel::~CompiledKernel()
{
  if (library) {
    cuLibraryUnload(library);
    library = nullptr;
    func    = nullptr;
  }
}

CompiledKernel::CompiledKernel(CompiledKernel&& other) noexcept
  : library(other.library),
    func(other.func),
    cubin(std::move(other.cubin)),
    rendered_source(std::move(other.rendered_source)),
    block_dim_x(other.block_dim_x),
    shared_mem_bytes(other.shared_mem_bytes)
{
  other.library = nullptr;
  other.func    = nullptr;
}

CompiledKernel& CompiledKernel::operator=(CompiledKernel&& other) noexcept
{
  if (this != &other) {
    if (library) cuLibraryUnload(library);
    library          = other.library;
    func             = other.func;
    cubin            = std::move(other.cubin);
    rendered_source  = std::move(other.rendered_source);
    block_dim_x      = other.block_dim_x;
    shared_mem_bytes = other.shared_mem_bytes;
    other.library    = nullptr;
    other.func       = nullptr;
  }
  return *this;
}

}  // namespace codegen::jit
