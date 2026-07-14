// Runtime FusedTree IR.  Used by callers (beam-search explorer, DSL
// parser, Rust FFI) to describe a compression-tree shape at runtime;
// the encode/decode JIT renderers (`encode/jit/renderer.cpp`,
// `decode/jit/renderer.cpp`) walk it and emit plain CUDA source text,
// which is then compiled at runtime via NVRTC and cached by shape (see
// `KernelCache`/`ShapeKey` in `kernel_cache.hpp`, keyed on a hash of
// the rendered source).  The beam-search explorer enumerates hundreds
// of thousands of candidate shapes, so this has to be a runtime value
// rather than a compile-time template instantiation per shape.
//
// Determinism contract: `children` is `std::map` so iteration yields
// lex-sorted key order.  Two structurally equal trees therefore
// produce byte-identical rendered output, which is what the cubin
// cache hashes on.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace codegen {

enum class OpKind : int { Bitpack, For, Delta, Rle, Raw, Zigzag, None };

inline constexpr int kChunkSize = 1024;
inline constexpr int kTBSize    = 128;

inline constexpr int32_t num_chunks_for(int64_t n) noexcept
{
  int64_t nc = (n + kChunkSize - 1) / kChunkSize;
  return static_cast<int32_t>(nc < 1 ? 1 : nc);
}

}  // namespace codegen

namespace codegen::jit {

struct FusedTree {
  OpKind op{OpKind::None};

  // Named child branches keyed by output-port name ("values", "runs").
  // Shared-ownership so callers can reuse subtrees across multiple
  // FusedTrees without copying (the same subtree can appear in more
  // than one place).
  std::map<std::string, std::shared_ptr<FusedTree>> children;

  bool is_leaf() const noexcept { return children.empty(); }

  static std::shared_ptr<FusedTree> make(OpKind op)
  {
    auto t = std::make_shared<FusedTree>();
    t->op  = op;
    return t;
  }

  static std::shared_ptr<FusedTree> make(OpKind op,
                                         std::map<std::string, std::shared_ptr<FusedTree>> children)
  {
    auto t      = std::make_shared<FusedTree>();
    t->op       = op;
    t->children = std::move(children);
    return t;
  }
};

// Human-readable op tag for diagnostics (error messages).
inline std::string op_kind_name(OpKind op)
{
  switch (op) {
    case OpKind::Bitpack: return "BITPACK";
    case OpKind::For: return "FOR";
    case OpKind::Delta: return "DELTA";
    case OpKind::Rle: return "RLE";
    case OpKind::Raw: return "RAW";
    case OpKind::Zigzag: return "ZIGZAG";
    case OpKind::None: return "NONE";
  }
  return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Buffer contract between the JIT encoder and decoder: a tree-wide set of
// device buffers keyed by "<node_id>.<field_name>". The encoder produces them;
// the decoder binds them by the same key.
// ---------------------------------------------------------------------------

struct LabeledBuffer {
  const void* ptr{nullptr};  // device pointer
  std::size_t length{0};     // in elements
  std::size_t elem_size{0};  // bytes per element
};

using LabeledBuffers = std::unordered_map<std::string, LabeledBuffer>;

// Key construction shared by producer and consumer.
inline std::string buffer_key(int32_t node_id, std::string_view field_name)
{
  std::string out;
  out.reserve(field_name.size() + 8);
  out.append(std::to_string(node_id));
  out.push_back('.');
  out.append(field_name);
  return out;
}

}  // namespace codegen::jit
