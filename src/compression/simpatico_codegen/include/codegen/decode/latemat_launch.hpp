// Late-materialization sparse JIT decode launchers (SIRIUS_EXP_LATE_MAT).
//
// Companions to the fused scan-filter launchers in masked_launch.hpp — same
// LabeledBuffers contract (persisted channels keyed ``buffer_key(node_id,
// field)``; decode-only transients synthesized inside), same false-on-failure
// (logged to stderr).  TWO deliberate differences:
//   1. The row selection is a chunk-bucketed CSR (codegen/selection/
//      row_set.hpp) — survivors grouped by 1024-row chunk, uint16 in-chunk
//      positions — and the launch grid is the TOUCHED-chunk list, so cost
//      scales with touched chunks, not batch chunks.
//   2. NO sync-on-return: all work is enqueued on ``stream`` and the call
//      returns (the late-mat scheduling contract; transients are
//      stream-ordered RMM allocations, safe to free without a host sync).
//
// These entry points are dead code unless the late materializer calls them;
// every shipped decode path is byte-identical whether or not this header is
// used.  New DecodeVariant values only — existing kernel templates and their
// JIT-cache entries are untouched.

#pragma once

#include "codegen/jit/fused_tree.hpp"
#include "codegen/selection/row_set.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace simpatico {

/// K8: sparse index-list decode -> compacted output, ascending row order.
/// ``rows`` is the batch's chunk-CSR (rows.num_rows must equal ``num_rows``).
/// ``out`` must have capacity for rows.num_survivors elements.  Supported
/// roots: any value_source-supported shape — Bitpack leaf (true random
/// access), Delta / RLE / FOR cascades (chunk staged in-SM, survivors-only
/// stores).  Returns false on render rejection or launch failure.
bool launch_decode_fused_tree_sparse_rows(codegen::jit::FusedTree const& tree,
                                          codegen::jit::LabeledBuffers& labeled,
                                          char const* dtype,
                                          std::int64_t num_rows,
                                          ::sirius::codegen::chunk_row_set const& rows,
                                          void* out,
                                          rmm::cuda_stream_view stream);

/// K5s: sparse constant-width dictionary gather.  ``tree`` is the dictionary
/// INDICES bitpack leaf (codes); for listed rows only, decodes the code and
/// copies ``key_width`` bytes from ``keys_chars`` into ``out_chars`` at
/// ``(out_offsets[b]+k)*key_width`` — compacted, ascending row order.
/// ``out_chars`` must hold rows.num_survivors * key_width bytes.
bool launch_decode_fused_tree_sparse_dict_gather(codegen::jit::FusedTree const& tree,
                                                 codegen::jit::LabeledBuffers& labeled,
                                                 char const* dtype,
                                                 std::int64_t num_rows,
                                                 ::sirius::codegen::chunk_row_set const& rows,
                                                 void const* keys_chars,
                                                 std::int32_t key_width,
                                                 void* out_chars,
                                                 rmm::cuda_stream_view stream);

/// K6s phase 1: sparse str_split survivor metadata.  ``tree``/``labeled`` are
/// the string column's OFFSETS subtree (Bitpack- or Delta-rooted, K6's
/// next-chunk-peek contract).  ``num_string_rows`` is the STRING row count;
/// ``rows`` is ROW-space (rows.num_rows == num_string_rows; row chunks and
/// offsets chunks are aligned).  Writes, compacted by list position:
/// ``src_offsets_out`` (int64[num_survivors]) and ``lengths_out``
/// (int32[num_survivors]).  Phase 2 is the shipped launch_masked_char_copy
/// (masked_launch.hpp), which is already list-driven and reused verbatim.
bool launch_decode_fused_tree_sparse_str_meta(codegen::jit::FusedTree const& tree,
                                              codegen::jit::LabeledBuffers& labeled,
                                              char const* dtype,
                                              std::int64_t num_string_rows,
                                              ::sirius::codegen::chunk_row_set const& rows,
                                              std::int64_t* src_offsets_out,
                                              std::int32_t* lengths_out,
                                              rmm::cuda_stream_view stream);

}  // namespace simpatico
