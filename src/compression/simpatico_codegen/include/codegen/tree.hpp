// Compile-time IR for the codegen tree.
//
// `jit::FusedTree` (include/codegen/jit/fused_tree.hpp) is a runtime
// dataclass — `{ OpKind op; map<string, shared_ptr<FusedTree>> children }`.
// The encode/decode JIT renderers (src/encode/jit/renderer.cpp,
// src/decode/jit/renderer.cpp) walk this tree and switch on `op` at
// codegen time to emit plain CUDA source text, which is then compiled
// at runtime via NVRTC (src/jit/). There is no per-op node *type* and
// no compile-time-recursive template walker — `OpKind` here is just the
// runtime tag those renderers dispatch on.

#pragma once

#include "codegen/stdint_shim.hpp"

namespace codegen {

// ---------------------------------------------------------------------------
// Op tags stored on `jit::FusedTree` nodes.  The encode/decode JIT
// renderers switch on this to decide which text-emission routine to call.
// ---------------------------------------------------------------------------
enum class OpKind : int {
  Bitpack,
  For,
  Delta,
  Rle,
  Raw,
  // ZigZag — element-wise signed↔unsigned interleave leaf.  Like Raw it
  // is a sink that materialises one stored channel ("zigzag"), but it
  // applies the closed-form ZigZag map on store and its inverse on load.
  // Maps small-magnitude signed values to small unsigned codes so a
  // downstream byte/entropy coder (ans/snappy) compresses them far better
  // than two's-complement (e.g. -1 → 1 instead of 0xFFFFFFFF).
  Zigzag,
  // Sentinel for "no child in this role" — mirrors the absence of a key
  // in `FusedTree::children`.
  None,
};

// ---------------------------------------------------------------------------
// Constants shared by the encode/decode JIT renderers and the on-disk
// `.hpln` format.  Bumping either requires re-generating any persisted
// plans/data that assume the old chunking.
// ---------------------------------------------------------------------------
inline constexpr int kChunkSize = 1024;  // elements per chunk
inline constexpr int kTBSize    = 128;   // block-decode threads/block

// ---------------------------------------------------------------------------
// Chunk helpers
// ---------------------------------------------------------------------------

// Number of chunks needed to cover *n* elements (at least one, so an empty
// input still has one chunk's worth of metadata).
inline constexpr int32_t num_chunks_for(int64_t n) noexcept
{
  int64_t nc = (n + kChunkSize - 1) / kChunkSize;
  return static_cast<int32_t>(nc < 1 ? 1 : nc);
}

}  // namespace codegen
