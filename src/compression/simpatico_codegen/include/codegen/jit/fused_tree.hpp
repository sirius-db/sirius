// Runtime FusedTree IR — mirror of the Python `kernels/tree.py`
// dataclass.  Used by callers (beam-search explorer, DSL parser, Rust
// FFI) to describe a compression-tree shape at runtime, which the JIT
// pipeline then turns into a concrete codegen C++ kernel.
//
// Why a runtime IR?  The compile-time template tree in `tree.hpp` is
// great for known shapes (one cubin per shape, fully inlined), but the
// beam-search explorer enumerates hundreds of thousands of candidate
// shapes — we can't pre-instantiate them all.  The runtime IR feeds
// `render_type()` which emits a single C++ type-expression string that
// instantiates the compile-time templates inside an nvrtc-JITted
// translation unit.  The decode-time inlining is the same in both
// paths; what differs is *when* the compiler runs.
//
// Determinism contract (matches Python): `children` is `std::map` so
// iteration yields lex-sorted key order.  Two structurally equal trees
// therefore produce byte-identical rendered output, which is what the
// cubin cache hashes on.
//
// Op tags reuse `codegen::OpKind` from `tree.hpp` so the runtime
// IR and the compile-time IR share one source of truth.

#pragma once

#include "../tree.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace codegen::jit {

struct FusedTree {
  OpKind op{OpKind::None};

  // Named child branches keyed by output-port name ("values", "runs").
  // Shared-ownership so callers can reuse subtrees across multiple
  // FusedTrees without copying — matches Python's reference-graph
  // semantics where the same dataclass can appear in two places.
  std::map<std::string, std::shared_ptr<FusedTree>> children;

  // Per-node attribute: only meaningful for ``OpKind::Bitpack`` today.
  // When true, the JIT'd kernel computes a Bitpack chunk's base word
  // offset arithmetically (``chunk_id * STRIDE_WORDS``) instead of
  // loading from the ``bp_offsets`` metadata buffer.  The OverAllocate
  // encoder layout (Python's ``use_fixed_stride=True``) sizes the
  // ``packed`` channel to ``num_chunks * STRIDE_WORDS`` words so the
  // arithmetic offset is exact.  Setting it on a non-Bitpack node is
  // a no-op (the renderer just won't emit the extra template arg).
  //
  // The renderer hashes this into the kernel-cache key via the
  // emitted type expression (e.g. ``Bitpack<int32_t, true>`` vs
  // ``Bitpack<int32_t, false>``), so OverAllocate and Compact shapes
  // get distinct compiled kernels and can co-exist in one process.
  bool fixed_stride{false};

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

// Effective dtype for a child branch.  Single source of truth for the
// "RLE.runs is always int32" rule that the Python side enforces via
// `tree.py::child_dtype` and that the renderer threads through every
// recursive call.
std::string child_dtype(OpKind parent_op,
                        const std::string& child_key,
                        const std::string& parent_dtype);

// Human-readable op tag for diagnostics / cache keys.  Not the same as
// the rendered C++ type; this one is stable across releases.
std::string op_kind_name(OpKind op);

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
