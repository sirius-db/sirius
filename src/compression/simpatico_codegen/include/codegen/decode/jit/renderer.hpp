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

// Kernel variants that evaluate a scan's filter while decoding (experimental;
// the engine gate SIRIUS_EXP_FUSED_SCAN_FILTER lives in the wave orchestrator,
// not here).  The variant changes the rendered source AND the entry symbol, so
// each variant gets its own JIT-cache entry automatically; predicate
// constants and mask/offset pointers are KERNEL PARAMETERS (appended after
// `out, n`), never baked into the source — one compile covers all literals.
//
//   * plain        — full-column decode, today's behaviour (byte-identical
//                    source to the pre-variant renderer).
//   * mask_out     — decode fused with an inclusive [lo,hi] range
//                    predicate on the decoded integer domain, producing
//                    selection-mask words (row r -> bit r%32 of word r/32;
//                    32 words per 1024-row chunk, tail bits and tail words
//                    written as zero).  NO column write; the `out`
//                    parameter slot is `uint32_t* sel_mask`, which must
//                    hold ceil(n/1024)*32 words.  Extra params: `int64_t
//                    pred_lo, int64_t pred_hi` (values are widened to
//                    int64 for the compare).  Supported roots: Bitpack
//                    leaf (closed-form, skips nothing but stores nothing),
//                    and — for delta->bitpack orderkey shapes, the
//                    decode-evaluable form of min-max dynamic join
//                    filters — any value_source-supported root (the delta
//                    form:
//                    the chunk is reconstructed IN FULL via the existing
//                    emitters, then the predicate is balloted from the
//                    staged values, keeping the mask-word alignment).
//   * mask_consume — decode consuming a selection mask plus per-chunk
//                    exclusive survivor offsets (`chunk_offsets[c]` =
//                    compacted output base of chunk c, length
//                    num_chunks+1), writing compacted output in row order.
//                    Supported roots:
//                      - Bitpack leaf: skipped rows are never unpacked;
//                        zero-survivor chunks early-return.
//                      - Delta root (any value_source-supported
//                        `differences` child, e.g. o_orderkey's
//                        delta->bitpack): the per-chunk prefix-sum
//                        reconstruction still runs in full (deltas are
//                        sequential within a chunk), but only survivor
//                        rows are WRITTEN — saves the full-column store +
//                        the downstream compaction pass, not the unpack.
//                    Extra params: `const uint32_t* sel_mask, const
//                    uint32_t* chunk_offsets`.
//   * index_consume — Bitpack-LEAF-root decode consuming an int32 ROW-
//                    INDEX LIST (global row ids, ascending — the
//                    mask->indices wave output) plus the same per-chunk
//                    exclusive survivor offsets as mask_consume.  Block c reads its
//                    slice row_indices[chunk_offsets[c] ..
//                    chunk_offsets[c+1]) and random-access decodes ONLY
//                    those rows into compacted output — no mask-word
//                    staging, per-block work scales with the chunk's
//                    survivor count, so it beats the mask walk at low
//                    selectivity (the runtime pick is the caller's).  Delta
//                    roots are rejected (sequential reconstruction cannot
//                    row-skip — callers fall back to the delta mask walk).  Extra
//                    params: `const int32_t* row_indices, const uint32_t*
//                    chunk_offsets`.
//   * mask_dict_gather — Bitpack-LEAF-root decode of DICTIONARY CODES
//                    consuming the selection mask like mask_consume, but instead of
//                    storing the code it gathers the key's bytes from a
//                    CONSTANT-WIDTH, null-free key pool straight into the
//                    compacted chars output (survivor rank r of chunk c
//                    lands at (chunk_offsets[c]+r)*key_width).  Offsets
//                    are analytic (j*key_width) and built by the caller.
//                    Extra params: `const uint32_t* sel_mask, const
//                    uint32_t* chunk_offsets, const char* keys_chars,
//                    int32_t key_width`.
//
// Unsupported (shape, variant) combinations are rejected with RenderError
// (e.g. RLE cannot row-skip — run expansion; dict gather needs a bitpack
// code leaf).
//   * str_split_meta — for `input -> str_split -> {offsets
//                    subtree, raw chars}` string columns, the tree passed
//                    is the OFFSETS subtree (any value_source-decodable
//                    cascade — bitpack, delta->rle->bitpack, ...; its ROOT
//                    must be Bitpack or Delta so the next chunk's first
//                    offset can be peeked from per-chunk scalars).  The
//                    kernel is launched over the OFFSETS domain (n =
//                    n_strings + 1): each block reconstructs its offsets
//                    chunk IN FULL via the existing emitters, then for
//                    survivor rows only emits the source char offset
//                    (`out`, int64) and byte length (`len_out`, int32),
//                    compacted by survivor rank.  Non-survivor chars are
//                    never touched; the raw chars byte gather is phase 2
//                    (a fixed kernel, see masked_launch.hpp
//                    launch_masked_char_copy) after the caller scans the
//                    lengths into output offsets.  Entropy-coded chars
//                    are out of scope by construction (this never reads
//                    chars).  Extra params: `const uint32_t* sel_mask,
//                    const uint32_t* chunk_offsets` (both ROW-space),
//                    `int32_t* len_out`.
// A decode kernel is a point in a two-axis product, not a flat variant list.
// The axes are independent: HOW rows are enumerated within a chunk, and WHAT
// happens per enumerated row.  Everything a kernel needs — emitted body, `out`
// slot type, trailing parameters, entry-symbol suffix, launcher preconditions —
// derives from the pair, so adding a kernel shape is one descriptor rather than
// edits to a variant enum, two parameter switches, a launcher and a probe.
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
DecodeKernelSpec render(const ::codegen::jit::FusedTree& tree,
                        const std::string& element_dtype,
                        std::int32_t num_chunks,
                        DecodeShape shape = kShapePlain);

}  // namespace codegen::decode::jit
