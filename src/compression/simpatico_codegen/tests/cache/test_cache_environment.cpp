#include "codegen/jit/cccl_embedded_headers.h"
#include "codegen/jit/embedded_headers.h"
#include "jit/compilation_request.hpp"

#include <nvrtc.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace cache          = codegen::jit::detail;
static int queries       = 0;
static bool fail_version = true;

// Test-only linker substitution; no CUDA device or kernel compilation needed.
extern "C" nvrtcResult __wrap_nvrtcVersion(int* major, int* minor)
{
  ++queries;
  if (fail_version) return NVRTC_ERROR_INTERNAL_ERROR;
  *major = 13;
  *minor = 4;
  return NVRTC_SUCCESS;
}

static void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

int main()
{
  try {
    bool reported = false;
    try {
      (void)cache::cache_environment();
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      reported                  = message.find("nvrtcVersion") != std::string::npos &&
                 message.find("NVRTC_ERROR_INTERNAL_ERROR") != std::string::npos;
    }
    require(reported && queries == 1, "version-query failure was not propagated");
    fail_version            = false;
    const auto& environment = cache::cache_environment();
    require(environment == cache::environment_identity({codegen::jit::kEmbeddedJitHeadersIdentity,
                                                        codegen::jit::kCcclEmbeddedHeadersIdentity,
                                                        13,
                                                        4}),
            "environment does not identify embedded bundles and NVRTC major/minor");
    require(queries == 2, "failed initialization must retry the version query");
    fail_version = true;
    require(&cache::cache_environment() == &environment && queries == 2,
            "successful environment must be cached without another version query");
    std::puts("cache environment: version errors, retry and process reuse passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
