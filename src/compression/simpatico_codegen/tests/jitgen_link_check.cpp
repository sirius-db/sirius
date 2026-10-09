// Links against simpatico_jitgen's objects only (plus NVRTC and the CUDA
// runtime/driver), never the rest of simpatico. The JIT cache epoch hashes just
// those objects, so if the NVRTC driver starts calling code defined elsewhere in
// simpatico, that code could change the generated cubins without changing the
// epoch. This target then fails to link: move the callee into simpatico_jitgen
// (or make it header-only) instead.
//
// Running it computes a cache key on the host; no GPU is needed. Compiling is
// left to the GPU tests.

#include "codegen/jit/jit_epoch.h"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "jit/cache_identity.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace cj = codegen::jit;

int main(int argc, char** argv)
{
  const std::string source = "extern \"C\" __global__ void probe() {}\n";
  const std::string entry  = "probe";

  cj::CompileOptions opts;
  opts.arch_cc       = 80;
  const auto options = cj::nvrtc_options(opts);
  std::vector<std::string_view> option_views(options.begin(), options.end());
  const auto key =
    cj::detail::request_identity({source, entry, cj::kNvrtcProgramName, option_views});

  // Reference the compile path so it must link, without needing a GPU to run.
  if (argc > 1 && std::strcmp(argv[1], "--compile") == 0) {
    opts.arch_cc = cj::arch_cc_for_current_device();
    (void)cj::compile_plain_kernel(source, entry, opts);
  }
  std::printf(
    "jitgen_link_check: epoch %s, key %s\n", cj::kJitEpoch, cj::detail::hex_digest(key).c_str());
  return 0;
}
