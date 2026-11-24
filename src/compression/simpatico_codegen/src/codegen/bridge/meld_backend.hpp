// Meld's decoders beside the NVRTC JIT, chosen by SIMPATICO_CODEGEN_BACKEND.
#pragma once

#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simpatico::meld {

enum class Backend { nvrtc, meld, shadow };

// Throws on an unknown name, and on `meld` or `shadow` in a build without Meld.
Backend decode_backend();

// The `codec.scheme` decoding `tree`, numbered as the decode binder numbers its buffers. Nothing
// where Meld does not decode the tree.
std::optional<std::string> decode_scheme(codegen::jit::FusedTree const& tree,
                                         char const* cxx_dtype);

struct Parameter {
  enum class Kind { channel, output, row_count };
  Kind kind = Kind::channel;
  int node  = -1;  // a channel's buffer is buffer_key(node, field)
  std::string field;
};

struct DecodeKernel {
  std::shared_ptr<codegen::jit::CompiledKernel const> kernel;
  std::vector<Parameter> parameters;
  int block_threads    = 0;
  int shared_bytes     = 0;
  int chunks_per_block = 1;
};

// Meld's plain decoder of `tree`, through the kernel cache. Nothing where Meld does not decode it.
std::optional<DecodeKernel> plain_decode_kernel(codegen::jit::FusedTree const& tree,
                                                char const* cxx_dtype);

}  // namespace simpatico::meld
