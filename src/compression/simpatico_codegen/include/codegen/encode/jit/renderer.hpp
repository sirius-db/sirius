// Encode-side renderer + layout calculator.
//
// Walks a `codegen::jit::FusedTree` and produces:
//   1. The CUDA-C++ source for a kernel that performs the GPU encode,
//      ready to hand to `compile_plain_kernel`.
//   2. The `BufferSpec` list the caller must allocate before launching
//      (per-op output buffers in the OverAllocate layout).
//   3. The launch geometry (grid/block/shared) the caller must use.
//
// Architectural intent — read this if you are about to add a new op.
// ====================================================================
//
// A naive approach requires every op to expose a closed-form,
// lane-addressable C++ expression (`prev_expr`) that its parent can
// splice inline.  That works for Bitpack and Delta but breaks for
// Rle, whose output is at data-dependent lane positions — it would
// reject shapes like `Rle{Delta{Rle{Bp,Bp}}, Bp}` (the
// nvcomp-cascaded default).
//
// We sidestep that by switching to STAGE DECOMPOSITION + SHARED-MEM
// MATERIALISATION:
//
//   * A "stage" is a sequence of ops that read at fixed input
//     positions.  Delta and Bitpack are stage-internal — their input
//     position equals their output position.
//   * A "stage boundary" is any op whose output is at data-dependent
//     positions.  Rle is the canonical stage boundary; deep Delta
//     chains may become one too if register pressure overflows.
//   * At a stage boundary the parent op writes its output into a
//     shared-memory slab and syncs the block; children read from
//     that slab as their input.
//   * All stages for a fused subtree run in one CUDA kernel per
//     chunk, separated by `__syncthreads()`.  No host orchestration,
//     no inter-kernel sync.
//
// This lifts the naive composability restriction completely: any
// tree shape the decode side accepts becomes encodable, including
// arbitrary RLE/Delta nesting.
//
// Supported ops
// -------------
// Bitpack (leaf); Raw (verbatim-passthrough leaf, synthesized for a
// delta/rle/for channel that isn't further fused); Delta, Rle, For, and
// Zigzag, composed arbitrarily — the staged-materialisation walker handles
// nested Rle/Delta/For/Zigzag and Raw children.  An op the renderer can't
// emit throws a `RenderError` with a clear diagnostic.
//
// Buffer layout contract
// ----------------------
// The encode kernel always emits the OverAllocate layout: `packed` is sized
// `num_chunks * kStrideWords` per Bitpack node, and each chunk's words land at
// the fixed arithmetic offset `chunk_id * kStrideWords` — so no `bp_offsets`
// and no host cumsum are needed to write. `live_words[c]` reports the actual
// bit-packed word count for chunk c.
//
// The OverAllocate→Compact story (compact_bitpack_packed before store, decode
// seeing only Compact) is documented canonically on that function in
// bridge/codegen_runtime.cpp.

#pragma once

#include "codegen/jit/fused_tree.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace codegen::encode::jit {

// Buffer specification — one entry per encoder-produced output channel.
// The caller must allocate `length * elem_size` bytes per spec before
// launching the kernel and pass the device pointers in `field` order
// (see `EncodeKernelSpec::buffer_field_order`).
//
// `field` matches the manifest key the decode-side reads from
// (`codegen::jit::buffer_manifest.hpp`), so a caller building
// LabeledBuffers for decode can use the spec list verbatim as the
// labels.
struct BufferSpec {
  int32_t node_id;          // FusedTree node id (DFS-preorder lex-sorted children)
  std::string field;        // manifest key ("chunk_min", "packed", "live_words", ...)
  std::size_t elem_size;    // sizeof one element of the buffer's logical dtype
  std::size_t length;       // number of elements
  bool no_pre_zero{false};  // skip cudaMemsetAsync pre-zero (smem path handles it)
};

// Rendering result.  Move-only; cheap to construct (no GPU resources).
//
// The compiled CUfunction is *not* part of this struct — pass `source`
// + `entry_symbol` to `compile_plain_kernel` (or the cache) to obtain
// a `CompiledKernel`.  Keeping render + compile separate lets the
// renderer be exercised by unit tests without a GPU.
struct EncodeKernelSpec {
  // The translation unit to compile.  Already type-substituted for
  // `element_dtype`; safe to hand straight to nvrtc.
  std::string source;

  // The `extern "C"` kernel symbol declared in `source`.  Stable per
  // (tree shape, element dtype) pair; used as the cache key by
  // higher layers.
  std::string entry_symbol;

  // Output buffer specs in the order the renderer expects them in the
  // kernel's parameter list (after the leading `flat`/`n` inputs).
  // Always sorted by `node_id` then by an op-defined field-order so
  // the binding is deterministic across runs.
  std::vector<BufferSpec> buffers;

  // Launch geometry.
  // grid_x = num_chunks_for(n).  The renderer hard-codes this rule;
  // the caller computes the grid size from input length.
  int block_x      = 1024;
  int block_y      = 1;
  int block_z      = 1;
  int shared_bytes = 0;

  // Diagnostic.  Empty for valid renders; populated with a short
  // human-readable note (e.g. "Bitpack root, OverAllocate") for
  // logging / `--verbose` builds.
  std::string note;
};

// Thrown when the renderer rejects a tree shape.  Distinct from
// `CompileError` (nvrtc) so call sites can distinguish "shape not
// supported yet" from "shape rendered but the source didn't compile".
struct RenderError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Render an EncodeKernelSpec for `tree` parametrised over `element_dtype`.
//
// `tree`           : the runtime FusedTree (caller-owned).
// `element_dtype`  : C++ scalar type string for the column ("int32_t",
//                    "int64_t").  Threaded into the kernel signature
//                    and the per-op intrinsics.
// `num_chunks`     : ceil(n / kChunkSize) — used to size the per-node
//                    output BufferSpecs.  Caller computes this from the
//                    input length so the spec list is self-contained
//                    (no second walk needed at launch time).
//
// Throws `RenderError` if `tree` contains an op the renderer can't emit.
// Supported ops: `Bitpack` (leaf, OverAllocate layout), `Raw` (leaf
// passthrough), `Delta`, `Rle`, `For`, `Zigzag` — composed arbitrarily.
//
// Throws `std::invalid_argument` for malformed inputs (empty dtype,
// num_chunks < 1).
EncodeKernelSpec render(const ::codegen::jit::FusedTree& tree,
                        const std::string& element_dtype,
                        int32_t num_chunks);

}  // namespace codegen::encode::jit
