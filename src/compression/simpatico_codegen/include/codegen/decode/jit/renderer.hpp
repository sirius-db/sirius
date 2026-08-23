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
//               always present). Persisted chunk_count is consumed by the
//               bridge to synthesize bp_offsets, but is intentionally omitted
//               from the generated kernel signature.
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

// Kernels that evaluate a scan's filter while decoding (experimental; the
// engine gate SIRIUS_EXP_FUSED_SCAN_FILTER lives in the wave orchestrator, not
// here).
//
// A decode kernel is a point in a two-axis product, not a flat variant list.
// The axes are independent: HOW rows are enumerated within a chunk, and WHAT
// happens per enumerated row.  Everything a kernel needs — emitted body, `out`
// slot type, trailing parameters, entry-symbol suffix, launcher preconditions —
// derives from the pair, so adding a kernel shape is one descriptor rather than
// edits to a variant enum, two parameter switches, a launcher and a probe.
//
// Two rules hold across every point:
//
//   * Predicate constants and mask/index pointers are KERNEL PARAMETERS
//     (appended after `out, n`), never baked into the source, so one NVRTC
//     compile per (shape, dtype, tree) serves every literal.  The shape changes
//     the entry symbol, so each point gets its own cache entry automatically.
//   * The selection mask layout is selection.hpp's (32 words per 1024-row
//     chunk, tail bits and words zero); the ballot consumers produce it and the
//     compacting enumerators consume it.
//
// Roots each point accepts, which is what the RenderError refusals enforce:
// `ballot_range` and `write_column` take any value_source-supported root (a
// staged root reconstructs the chunk, then ballots or stores survivors from
// it), EXCEPT that `index_list` requires a Bitpack leaf — random access is the
// point, and a prefix sum cannot row-skip.  `dict_gather` requires a Bitpack
// code leaf; `offsets_meta` a Bitpack- or Delta-rooted offsets subtree, so the
// next chunk's first offset can be peeked from per-chunk scalars.
enum class Enumerator : std::uint8_t {
  all_rows = 0,  ///< every row of the chunk (full-width decode)
  mask_bits,     ///< survivors of a selection mask, compacted by rank
  index_list,    ///< an ascending survivor row-id list, compacted by slot
};

enum class Consumer : std::uint8_t {
  write_column = 0,  ///< store the reconstructed value
  ballot_range,      ///< 1 tree: compare against [lo,hi], ballot to mask words
  dict_gather,       ///< copy the code's fixed-width key bytes to compacted chars
  offsets_meta,      ///< emit survivor {source char offset, length}
};

struct DecodeShape {
  Enumerator enumerator                            = Enumerator::all_rows;
  Consumer consumer                                = Consumer::write_column;
  friend bool operator==(DecodeShape, DecodeShape) = default;
};

// The shipped points of the product; masked_launch.hpp declares one launcher
// per point.
inline constexpr DecodeShape kShapePlain{Enumerator::all_rows, Consumer::write_column};
inline constexpr DecodeShape kShapeMaskOut{Enumerator::all_rows, Consumer::ballot_range};
inline constexpr DecodeShape kShapeMaskConsume{Enumerator::mask_bits, Consumer::write_column};
inline constexpr DecodeShape kShapeIndexConsume{Enumerator::index_list, Consumer::write_column};
inline constexpr DecodeShape kShapeDictGather{Enumerator::mask_bits, Consumer::dict_gather};
inline constexpr DecodeShape kShapeStrSplitMeta{Enumerator::mask_bits, Consumer::offsets_meta};

/// False for product points with no meaning or no renderer support — e.g.
/// re-ballotting only the survivors of an existing mask.  Render rejects these
/// rather than emitting something plausible; the combinations that ARE
/// meaningful but unbuilt (index_list x dict_gather / offsets_meta) are listed
/// as false here until their emitters land.
[[nodiscard]] bool shape_is_supported(DecodeShape shape);

// One decode-input channel the rendered kernel reads.  `field` is the
// manifest key (`buffer_key(node_id, field)` resolves the device
// pointer in the launcher's LabeledBuffers).
struct DecodeBufferSpec {
  std::int32_t node_id;
  std::string field;      // "chunk_min", "packed", "delta_first", ...
  std::size_t elem_size;  // bytes/elem of the channel's logical dtype
};

// One trailing kernel parameter — those following the fixed `(out, n)` pair.
//
// The renderer emits the parameter DECLARATION TEXT and the matching tag list
// (DecodeKernelSpec::trailing) from a single table, and the launcher binds
// arguments by walking that list.  Emission order and binding order therefore
// cannot drift: previously they were two hand-maintained switches over
// the shape in two different files, with cuLaunchKernel's untyped void**
// between them, so a mismatch was silent argument misalignment rather than a
// compile error.
//
// Values are supplied by the launcher (predicate constants and mask pointers
// travel as kernel ARGUMENTS, never rendered into the source, so one compile
// per (shape, dtype, variant) serves every literal).
enum class TrailingParam : std::uint8_t {
  pred_lo = 0,    // mask_out: inclusive range low, decoded integer domain
  pred_hi,        // mask_out: inclusive range high
  sel_mask,       // mask_consume / dict_gather / str_split_meta: mask words in
  chunk_offsets,  // consuming variants: per-chunk exclusive survivor bases
  keys_chars,     // mask_dict_gather: constant-width key pool chars
  key_width,      // mask_dict_gather: bytes per key
  row_indices,    // index_consume: ascending global int32 survivor row ids
  len_out,        // str_split_meta: per-survivor byte lengths (output)
  kCount          // sentinel: size of a tag-indexed table
};

// Rendering result.  Move-only; cheap (no GPU resources).  Hand
// `source` + `entry_symbol` to `compile_plain_kernel` (or the kernel
// cache) to obtain a CompiledKernel.
struct DecodeKernelSpec {
  std::string source;        // full CUDA-C++ TU, already dtype-substituted
  std::string entry_symbol;  // extern "C" kernel symbol declared in `source`

  // Input channels in the kernel's parameter order (before out, n).
  std::vector<DecodeBufferSpec> buffers;

  // Trailing scalar/pointer parameters AFTER (out, n), in kernel parameter
  // order.  The launcher must push its arguments in exactly this order; it is
  // emitted alongside the declaration text, so the two cannot disagree.
  std::vector<TrailingParam> trailing;

  // Launch geometry.  grid_x = num_chunks_for(n) (launcher computes).
  int block_x      = 128;  // plain-CUDA block; RLE/Delta primitives assume 128
  int shared_bytes = 0;    // dynamic shared workspace peak (RLE boundaries)

  std::string note;  // human-readable diagnostic
};

// Renderer rejected a tree shape (an op it can't emit). Shared with the encode
// renderer — see codegen::jit::RenderError.
using ::codegen::jit::RenderError;

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
//
// `shape` selects the enumerator + consumer (see DecodeShape above); the
// default renders the plain full-column decode.  Unsupported points of the
// product are rejected with RenderError (see shape_is_supported).
// The one kernel with no tree behind it: str_split phase 2, copying each
// survivor's byte range out of the RAW chars buffer.  Rendered rather than kept
// as a string constant in the launcher, so ALL emitted CUDA comes from this
// file and is reachable by the dump-and-diff check the other shapes use.
// `buffers` is empty and `trailing` unused — the launcher binds its five
// arguments positionally, as the doc comment on it states.
DecodeKernelSpec render_masked_char_copy();

DecodeKernelSpec render(const ::codegen::jit::FusedTree& tree,
                        const std::string& element_dtype,
                        std::int32_t num_chunks,
                        DecodeShape shape = kShapePlain);

}  // namespace codegen::decode::jit
