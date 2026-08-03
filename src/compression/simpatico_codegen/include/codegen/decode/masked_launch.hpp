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

}  // namespace simpatico
