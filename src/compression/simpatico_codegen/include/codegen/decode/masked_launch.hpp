// Masked JIT decode launchers — fused scan-filter pipeline (K1 / K3).
//
// Companions to ``simpatico::launch_decode_fused_tree`` (codegen_bridge.hpp):
// same LabeledBuffers contract (persisted channels keyed
// ``buffer_key(node_id, field)``; decode-only transients such as bp_offsets
// are synthesized inside), same stream-sync-on-return, same false-on-failure
// (logged to stderr).  Both require the fused tree to be a **Bitpack leaf
// root**; any other shape is rejected at render time and returns false, so
// callers can fall back to the plain decode path.
//
// These entry points are dead code unless the wave orchestrator calls them
// (engine gate SIRIUS_EXP_FUSED_SCAN_FILTER lives there): the plain decode
// path is byte-identical whether or not this header is used.
//
// Selection-mask contract (see codegen/selection/selection.hpp):
//   * mask words: 32 uint32 words per 1024-row chunk, i.e.
//     ``selection_mask::ChunksFor(num_rows) * 32`` words total.  K1 writes
//     every word of every chunk it covers (tail bits/words are zero).
//   * chunk_offsets: uint32[ChunksFor(num_rows) + 1], exclusive prefix sum
//     of per-chunk survivor counts (the CNT wave's output).  K3 uses
//     chunk_offsets[c] as the compacted output base of chunk c.
//
// Predicate constants and mask/offset pointers travel as KERNEL PARAMETERS,
// never as rendered source — one NVRTC compile per (tree shape, dtype,
// variant) serves every literal (see DecodeVariant in decode/jit/renderer.hpp).

#pragma once

#include "codegen/jit/fused_tree.hpp"
#include "codegen/selection/selection.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace simpatico {

/// K1: Bitpack-leaf decode fused with the inclusive range predicate
/// ``pred`` (decoded integer domain, values widened to int64 for the
/// compare), producing selection-mask words into ``mask.words``.  No column
/// output is written or allocated.  ``mask.num_rows`` must equal
/// ``num_rows`` and ``mask.words`` must hold ChunksFor(num_rows)*32 words.
/// ``mask.survivor_count`` / ``mask.chunk_offsets`` are untouched (the CNT
/// wave fills them).  Float32/float64 columns must not be routed here:
/// they decode as bit-reinterpreted integers, so an integer-domain range
/// compare would be meaningless.
bool launch_decode_fused_tree_mask_out(codegen::jit::FusedTree const& tree,
                                       codegen::jit::LabeledBuffers& labeled,
                                       char const* dtype,
                                       std::int64_t num_rows,
                                       ::sirius::codegen::range_predicate pred,
                                       ::sirius::codegen::selection_mask& mask,
                                       rmm::cuda_stream_view stream);

/// K3: decode consuming ``mask.words`` + ``mask.chunk_offsets`` (both
/// required non-null; chunk_offsets means the CNT wave already ran),
/// writing compacted output in row order into ``out``.  ``out`` must have
/// capacity for ``mask.survivor_count`` elements.  Supported tree roots:
/// Bitpack leaf (rows whose mask bit is 0 are never unpacked) and Delta
/// root with a value_source-supported ``differences`` child (o_orderkey's
/// delta->bitpack shape; the per-chunk prefix-sum reconstruction still
/// runs, only the stores are masked/compacted).  Zero-survivor chunks
/// early-return.
bool launch_decode_fused_tree_mask_consume(codegen::jit::FusedTree const& tree,
                                           codegen::jit::LabeledBuffers& labeled,
                                           char const* dtype,
                                           std::int64_t num_rows,
                                           ::sirius::codegen::selection_mask const& mask,
                                           void* out,
                                           rmm::cuda_stream_view stream);

/// K4: index-list-consuming compacting decode — the low-selectivity sibling
/// of K3 (runtime pick by survivor count is the caller's; microbench
/// crossover ~15% selectivity).  ``row_indices`` is the ascending GLOBAL
/// int32 row-index list of survivors (the mask->indices wave output,
/// ``mask.survivor_count`` entries, consistent with ``mask.chunk_offsets``:
/// chunk c's rows occupy row_indices[chunk_offsets[c] ..
/// chunk_offsets[c+1])).  Only listed rows are decoded (random access into
/// the packed bits); out slot j gets row row_indices[j]'s value.  ``out``
/// must have capacity for ``mask.survivor_count`` elements.  Bitpack leaf
/// roots only — Delta roots are rejected at render time (returns false);
/// fall back to K3-delta.
bool launch_decode_fused_tree_index_consume(codegen::jit::FusedTree const& tree,
                                            codegen::jit::LabeledBuffers& labeled,
                                            char const* dtype,
                                            std::int64_t num_rows,
                                            ::sirius::codegen::selection_mask const& mask,
                                            std::int32_t const* row_indices,
                                            void* out,
                                            rmm::cuda_stream_view stream);

/// K5: masked dictionary gather for dictionary->bitpack string columns with
/// CONSTANT-WIDTH, null-free keys (q1's l_returnflag / l_linestatus).  The
/// tree is the dictionary INDICES bitpack leaf (codes, int32 domain).  For
/// survivor rows only, decodes the code and copies the key's
/// ``key_width`` bytes from ``keys_chars`` (device pointer to the key
/// pool's chars, key k at ``keys_chars + k*key_width``) into
/// ``out_chars`` at ``(chunk_offsets[chunk] + rank) * key_width`` —
/// compacted, row order preserved.  ``out_chars`` must have capacity
/// ``mask.survivor_count * key_width`` bytes.  The offsets column is
/// analytic (``j * key_width``) and assembled by the caller together with
/// the strings column (same split as try_decode_constant_width's
/// tabulate).  Requires mask.chunk_offsets non-null (CNT ran).
bool launch_decode_fused_tree_mask_dict_gather(codegen::jit::FusedTree const& tree,
                                               codegen::jit::LabeledBuffers& labeled,
                                               char const* dtype,
                                               std::int64_t num_rows,
                                               ::sirius::codegen::selection_mask const& mask,
                                               void const* keys_chars,
                                               std::int32_t key_width,
                                               void* out_chars,
                                               rmm::cuda_stream_view stream);

/// K6 phase 1: str_split masked survivor metadata.  ``tree``/``labeled``
/// are the string column's OFFSETS subtree (Bitpack- or Delta-rooted; any
/// depth below — other roots return false).  ``num_string_rows`` is the
/// STRING row count n; the offsets column has n+1 elements and the kernel
/// is launched over that domain internally.  ``mask``/``chunk_offsets``
/// are row-space as usual.  Writes, compacted by survivor rank:
///   * ``src_offsets_out`` (int64[survivor_count]) — char-range starts in
///     the RAW chars buffer,
///   * ``lengths_out``     (int32[survivor_count]) — byte lengths (for the
///     caller's output-offsets scan / offsets-column rebuild).
/// Non-survivor chars are never read.  Entropy-coded chars are naturally
/// out of scope (this touches only the offsets subtree); route those
/// columns through TierB.
bool launch_decode_fused_tree_str_split_meta(codegen::jit::FusedTree const& tree,
                                             codegen::jit::LabeledBuffers& labeled,
                                             char const* dtype,
                                             std::int64_t num_string_rows,
                                             ::sirius::codegen::selection_mask const& mask,
                                             std::int64_t* src_offsets_out,
                                             std::int32_t* lengths_out,
                                             rmm::cuda_stream_view stream);

/// K6 phase 2: fixed (tree-independent, JIT-cached constant source) byte
/// gather from the RAW chars buffer.  For each survivor j in
/// [0, n_survivors): copies out_offsets[j+1]-out_offsets[j] bytes from
/// chars[src_offsets[j]] to out_chars[out_offsets[j]].  ``out_offsets`` is
/// the exclusive scan of K6-phase-1 lengths (n_survivors+1 entries, cudf
/// offsets layout — reuse it directly as the compacted offsets column).
bool launch_masked_char_copy(void const* chars,
                             std::int64_t const* src_offsets,
                             std::int32_t const* out_offsets,
                             std::int64_t n_survivors,
                             void* out_chars,
                             rmm::cuda_stream_view stream);

/// K1m2 comparison op codes (kernel parameter, applied as `a OP b`).
enum class pair_cmp : std::int32_t { lt = 0, le = 1, gt = 2, ge = 3 };

/// K1m2: two-column fused pair-predicate mask for two Bitpack-leaf columns
/// of the same table.  Decodes both columns in-chunk and ballots
/// `a OP b [&& a in range_a && b in range_b]` into ``mask.words`` (same
/// layout/contract as launch_decode_fused_tree_mask_out).  The columns
/// MUST share chunk geometry — verified here via the two chunk_count
/// channel lengths; mismatch returns false.  ``labeled_b`` is re-keyed
/// internally to node 1; both maps get their transients synthesized.  Pass
/// ``range_a``/``range_b`` = {INT64_MIN, INT64_MAX} when a column has no
/// constant range (defaulted).  q12-style multi-pair conjunctions = one
/// launch per pair term, AND-ed by the combine wave.
bool launch_decode_fused_tree_pair_mask_out(
  codegen::jit::FusedTree const& tree_a,
  codegen::jit::LabeledBuffers& labeled_a,
  char const* dtype_a,
  codegen::jit::FusedTree const& tree_b,
  codegen::jit::LabeledBuffers& labeled_b,
  char const* dtype_b,
  std::int64_t num_rows,
  pair_cmp op,
  ::sirius::codegen::selection_mask& mask,
  rmm::cuda_stream_view stream,
  ::sirius::codegen::range_predicate range_a = {INT64_MIN, INT64_MAX},
  ::sirius::codegen::range_predicate range_b = {INT64_MIN, INT64_MAX});

}  // namespace simpatico
