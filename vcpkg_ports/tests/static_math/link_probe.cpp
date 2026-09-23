#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <nvJitLink.h>

extern "C" void static_cuda_math_link_probe()
{
  cublasHandle_t blas{};
  cublasCreate(&blas);
  cublasDestroy(blas);

  cusolverDnHandle_t solver{};
  cusolverDnCreate(&solver);
  cusolverDnDestroy(solver);

  cusparseHandle_t sparse{};
  cusparseCreate(&sparse);
  cusparseDestroy(sparse);

  nvJitLinkHandle linker{};
  nvJitLinkCreate(&linker, 0, nullptr);
  nvJitLinkDestroy(&linker);
}
