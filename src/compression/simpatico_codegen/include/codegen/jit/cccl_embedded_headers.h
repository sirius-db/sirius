// Declares the embedded CCCL (<cuda/std/...>, <cub/...>) header closure the
// runtime NVRTC JIT compiles against. The table is defined in the generated
// cccl_embedded_headers.cpp (see cmake/embed_cccl_headers.cmake), which embeds
// each header as a raw-string literal so the JIT needs no CCCL tree on disk.
#pragma once

#include "codegen/jit/embedded_header.h"

namespace codegen::jit {

extern const EmbeddedJitHeader kCcclEmbeddedHeaders[];
extern const int kCcclEmbeddedHeaderCount;

}  // namespace codegen::jit
