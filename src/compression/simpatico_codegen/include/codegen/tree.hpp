// Compile-time IR for the codegen tree.
//
// In the Python port (kernels/tree.py) the IR is a runtime
// dataclass — `FusedTree { OpKind op; dict[str, FusedTree] children }`.
// The walker dispatches on `op` at codegen time and produces source.
//
// In the C++ port each op is a *type*.  The tree is built by nesting
// templates: `Delta<Bitpack<>>` is one type; `Rle<Bitpack<>, Bitpack<>>`
// is another.  The recursive walker is just `root.decode(pos, mask)` —
// the compiler performs the DFS preorder at instantiation time, no
// runtime dispatch and no string rendering.
//
// Each node type carries:
//   * a `kind()` constexpr returning the op tag for introspection
//   * data members holding pointers/scalars the kernel consumes
//   * a `decode(...)` member template that returns a tile of values at
//     the requested positions, recursing into child nodes.
//
// The decode signatures match the Python decompress_renderer's per-op
// helpers:
//
//   Python: `_render_<op>(nid, node, ..., pos_var, mask_var, ...)`
//   C++:    `template <PosTile, MaskTile> decode(PosTile, MaskTile, int cid)`
//
// where `cid` is the linear chunk id (`ct::bid().x` at kernel entry).

#pragma once

#include "codegen/stdint_shim.hpp"

namespace codegen {

// ---------------------------------------------------------------------------
// Op tags — used purely for compile-time introspection.  The actual
// dispatch is by overload of the node type, not by checking this tag at
// runtime.  Mirrors `tree.py::OpKind`.
// ---------------------------------------------------------------------------
enum class OpKind : int {
  Bitpack,
  For,
  Delta,
  Rle,
  Raw,
  // The "leaf" sentinel used by the templates below to terminate
  // recursion when a node has no children of a given role.  Mirrors
  // the absence of a key in `FusedTree.children` on the Python side.
  None,
};

// Sentinel node type that terminates recursion.  Has no decode() because
// it's never instantiated as a real child; it exists only as a default
// template argument so single-child ops like `Delta<>` can be written.
struct NoChild {
  static constexpr OpKind kind() noexcept { return OpKind::None; }
};

// Per-kernel-invocation context threaded through every recursive
// `decode()` call.  Kept tiny + trivially copyable so the compiler
// passes it in registers.  Mirrors the fixed variable names the Python
// renderer threads through every `_render_<op>` helper:
//
//   chunk_id        ←→ `cid`  (Python `cid = ct.bid(0)`)
//   chunk_elements  ←→ `n_cs` (Python `n_cs = ct.minimum(...)`)
//
// Adding fields here is free for nodes that don't consume them; the
// compiler optimises out unused reads.
struct DecodeContext {
  int32_t chunk_id;        // linear chunk index in [0, num_chunks)
  int32_t chunk_elements;  // logical length of this chunk in [1, kChunkSize]
};

// ---------------------------------------------------------------------------
// Sequential-decode state placeholder.
//
// Every node exposes a `SeqState` typedef and a `make_state()` factory
// so a *batched* caller (e.g., `Rle::decode_at` walking runs in
// RUN_TILE_SIZE groups) can thread cumulative state across calls to one
// child.  Stateless nodes (`Bitpack`, `Rle`, `RawRleLeaf`) use this
// empty struct; `Delta` uses a `tile<Element, shape<1>>` to carry the
// cumulative diff sum across batches — see `decode/delta.cuh` for the
// algorithm.
//
// Every node exposes exactly one `decode_at` signature:
//   `decode_at(idx, mask, ctx, state&)`
// The caller (`walker.cuh` at the root, parent nodes inside their own
// batch loops) is responsible for constructing the state via
// `Node::make_state()` and threading it through.  Stateless nodes
// ignore the `state&` parameter.
// ---------------------------------------------------------------------------
struct NoSeqState {};

// ---------------------------------------------------------------------------
// Constants that must stay in lockstep with the Python side.  See
// `kernels/primitives.py` and `decompress_renderer.py` for the
// originals.  Bumping any of these requires a coordinated change on
// both sides.
// ---------------------------------------------------------------------------
inline constexpr int kChunkSize       = 1024;  // primitives.CHUNK_SIZE
inline constexpr int kTBSize          = 128;   // block-decode threads/block (decode_block_to_smem)
inline constexpr int kBlockOut        = 256;   // _BLOCK_OUT (RLE 2D)
inline constexpr int kMaxRunsPerChunk = 16;    // primitives.MAX_RUNS_PER_CHUNK
inline constexpr int kSubBlocks       = kChunkSize / kBlockOut;  // = 4

// ---------------------------------------------------------------------------
// Forward declarations.  The actual decoders live in `decode/*.cuh` and
// are pulled in via the umbrella header `decode/walker.cuh`.
// ---------------------------------------------------------------------------
template <typename Element, bool FixedStride = false>
struct Bitpack;
template <typename Element, typename Values>
struct Delta;

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
