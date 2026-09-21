// Shared type for headers embedded into the binary and fed to NVRTC as named
// in-memory headers (nvrtcCreateProgram). Used by the generated project-header
// and CCCL-closure tables so the runtime JIT needs no header tree on disk.
#pragma once

namespace codegen::jit {

struct EmbeddedJitHeader {
  const char* name;    // the exact `#include` name the rendered kernels use
  const char* source;  // full header text
};

}  // namespace codegen::jit
