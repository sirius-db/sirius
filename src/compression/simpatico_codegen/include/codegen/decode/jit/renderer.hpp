// Decode-side renderer — recursive composable walker (plain CUDA).
//
// This is the symmetric counterpart to
// `codegen::encode::jit::render` in
// `src/encode/jit/renderer.cpp`.  Where the encode walker
// emits ONE plain-CUDA `__global__` that reads the flat input column
// and writes the per-op OverAllocate output channels, this walker emits
// ONE plain-CUDA `__global__` that reads the per-op channels and writes
// the reconstructed flat output column.
//
// Why a string renderer
// =====================
// An earlier prototype materialised EVERY node's output into a
// shared-mem buffer with a `__syncthreads()` between each level (a block
// `__global__` `decode_block_to_smem`).
//
// This renderer brings the encode-side fusion model to decode: Delta is
// an inline transformer (its diff source is spliced into a single block
// scan — no intermediate shared-mem buffer), Bitpack is the leaf value
// reader, and RLE is the ONLY stage boundary (it materialises its
// `values`/`counts` children into shared memory, syncs, then expands).
// This mirrors `encode/jit/renderer.cpp` op-for-op so encode and decode
// share one symmetric plain-CUDA codegen codepath.
//
// Buffer contract
// ---------------
// The rendered kernel's parameter list is:
//
//     (<input channels in node_id/field order>, Element* out, int64_t n)
//
// where the input channels are exactly the decode-relevant manifest
// fields per node:
//   * Bitpack : chunk_min, chunk_bits, packed, bp_offsets (decode is
//               Compact-only, so bp_offsets — synthesized on-device — is
//               always present; chunk_count is unused at decode and
//               intentionally omitted from the signature)
//   * Delta   : delta_first
//   * Rle     : rle_runs_offsets
//   * For     : references (one per chunk)
//   * Zigzag  : zigzag (leaf store); no channel when inline-fused
//
// `buffers` lists those channels in the order the kernel expects them;
// the launcher binds device pointers by (node_id, field) and pushes
// them in this order, then `out` and `n`.

#pragma once

#include "codegen/jit/fused_tree.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace codegen::decode::jit {

// One decode-input channel the rendered kernel reads.  `field` is the
// manifest key (`buffer_key(node_id, field)` resolves the device
// pointer in the launcher's LabeledBuffers).
struct DecodeBufferSpec {
  std::int32_t node_id;
  std::string field;      // "chunk_min", "packed", "delta_first", ...
  std::size_t elem_size;  // bytes/elem of the channel's logical dtype
};

// Rendering result.  Move-only; cheap (no GPU resources).  Hand
// `source` + `entry_symbol` to `compile_plain_kernel` (or the kernel
// cache) to obtain a CompiledKernel.
struct DecodeKernelSpec {
  std::string source;        // full CUDA-C++ TU, already dtype-substituted
  std::string entry_symbol;  // extern "C" kernel symbol declared in `source`

  // Input channels in the kernel's parameter order (before out, n).
  std::vector<DecodeBufferSpec> buffers;

  // Launch geometry.  grid_x = num_chunks_for(n) (launcher computes).
  int block_x      = 128;  // plain-CUDA block; RLE/Delta primitives assume 128
  int shared_bytes = 0;    // dynamic shared workspace peak (RLE boundaries)

  std::string note;  // human-readable diagnostic
};

// Thrown when the renderer rejects a tree shape (unsupported op).
struct RenderError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Render a DecodeKernelSpec for `tree` parametrised over `element_dtype`
// ("int32_t" / "int64_t").  `num_chunks` is used only to size the
// informational BufferSpec lengths; the emitted source is invariant in
// it (chunk_id comes from blockIdx.x), so the kernel cache keys on the
// source alone.
//
// Supported ops: Bitpack (leaf), Delta (inline-fused), Rle (stage
// boundary), For (semi-inline transformer), Zigzag (inline transform or
// leaf store), and Raw (verbatim-passthrough leaf, valid as a
// delta/rle/for child), composed arbitrarily.
DecodeKernelSpec render(const ::codegen::jit::FusedTree& tree,
                        const std::string& element_dtype,
                        std::int32_t num_chunks);

}  // namespace codegen::decode::jit
