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

/// K3: Bitpack-leaf decode consuming ``mask.words`` + ``mask.chunk_offsets``
/// (both required non-null; chunk_offsets means the CNT wave already ran),
/// writing compacted output in row order into ``out``.  ``out`` must have
/// capacity for ``mask.survivor_count`` elements.  Rows whose mask bit is 0
/// are never unpacked; zero-survivor chunks early-return.
bool launch_decode_fused_tree_mask_consume(codegen::jit::FusedTree const& tree,
                                           codegen::jit::LabeledBuffers& labeled,
                                           char const* dtype,
                                           std::int64_t num_rows,
                                           ::sirius::codegen::selection_mask const& mask,
                                           void* out,
                                           rmm::cuda_stream_view stream);

}  // namespace simpatico
