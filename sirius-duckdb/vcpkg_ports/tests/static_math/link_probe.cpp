#include <cublas_v2.h>
#include <curand.h>
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

  curandGenerator_t generator{};
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandDestroyGenerator(generator);

  nvJitLinkHandle linker{};
  nvJitLinkCreate(&linker, 0, nullptr);
  nvJitLinkDestroy(&linker);
}
