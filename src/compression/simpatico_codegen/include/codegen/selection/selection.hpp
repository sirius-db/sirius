// SPDX-License-Identifier: Apache-2.0
//
// Shared types for evaluating a scan's filter while decompressing
// (env gate: SIRIUS_EXP_FUSED_SCAN_FILTER).
//
// Owned by the wave orchestration. Included by:
//   - the JIT decode variants (ballot-to-mask, mask-consuming) as kernel-arg
//     types,
//   - the caller building a scan_filter_request,
//   - the batch contract (survivor-count-first allocation, full-width gathers),
//   - selection_wave.cu (AND-combine, count, mask->indices).
//
// This header is host-side plumbing: PODs holding device pointers, plus the
// request/result structs that cross the converter boundary. It must stay
// includable from both .cpp and .cu TUs, and must NOT be baked into JIT
// (NVRTC) source — predicate constants travel as kernel parameters (a new
// NVRTC compile per literal would defeat the JIT cache).

#pragma once

#include "codegen/jit/fused_tree.hpp"

#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace cudf {
class column;
}

namespace sirius::codegen {

// Number of rows covered by one CNT chunk (one chunk_offsets entry). This IS
// the simpatico bitpack chunk, not merely equal to it: a mask-consuming decode
// addresses its packed bits per chunk and uses chunk_offsets[chunk] as that
// chunk's base in the compacted output, so a selection chunk and a bitpack
// chunk are the same 1024 rows. Derived rather than restated, because the two
// silently diverging would mean every mask bit addressing the wrong row.
inline constexpr int64_t SELECTION_CHUNK_ROWS = ::codegen::kChunkSize;

// 1 bit per row, 1 = survivor. Words are uint32_t, row r -> word r/32, bit r%32.
// Tail bits beyond num_rows MUST be zero (CNT and gather rely on it).
//
// SIZE = ChunksFor(num_rows) * 32 = ceil(num_rows/1024) * 32 words: every
// kernel touching the mask (the ballot producer, AND-combine, CNT,
// mask->indices, the mask consumer) addresses a FULL 32-word strip per 1024-row
// chunk, so a partial tail chunk still owns 32 words. The producer keeps the
// tail zero by construction
// (out-of-range lanes ballot to 0). Sizing by ceil(num_rows/32) is an
// out-of-bounds write for any num_rows not a multiple of 1024.
struct selection_mask {
  uint32_t* words         = nullptr;  // device, WordsFor(num_rows) words, 128B-aligned
  int64_t num_rows        = 0;
  int64_t survivor_count  = -1;       // -1 until CNT wave ran
  uint32_t* chunk_offsets = nullptr;  // device, exclusive prefix sum of survivors per
                                      // 1024-row chunk (length = ChunksFor(num_rows)+1);
                                      // null until CNT ran.

  // Sizing helpers (host-side).
  static constexpr int64_t ChunksFor(int64_t num_rows)
  {
    return (num_rows + SELECTION_CHUNK_ROWS - 1) / SELECTION_CHUNK_ROWS;
  }
  static constexpr int64_t WordsFor(int64_t num_rows)
  {  // full 32-word chunk strips
    return ChunksFor(num_rows) * (SELECTION_CHUNK_ROWS / 32);
  }
  static constexpr int64_t AllocWordsFor(int64_t num_rows)
  {  // alias of WordsFor
    return WordsFor(num_rows);
  }
};

// Range predicate on the DECODED integer domain (dates = days-since-epoch as
// stored, decimals = scaled int, plain integers as-is). Inclusive both ends.
// Passed as KERNEL PARAMS, never baked into JIT source (NVRTC cache!).
struct range_predicate {
  int64_t lo;
  int64_t hi;
};

// ── Converter-boundary contract ─────────────────────────────────────────────

// How wave 2 produces one output column — the route its plan shape takes.
//
//   full        — decode full width, then compact with one `cudf::gather` over
//                 the shared survivor index map. Costs about the unfiltered
//                 path, so it is admitted only at low selectivity: once the
//                 survivor count is known the batch proceeds iff
//                 survivors/rows <= SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL
//                 (default 0.10), else the decode gives compaction up
//                 (measured losses at high selectivity: q1 +43.5%, q5 +6.2%).
//   bitpack_mask— a bitpack root decoded straight into a compacted
//                 survivor_count-row column, skipping the writes for rejected
//                 rows.
//   delta_mask  — the same, for a delta->bitpack root.
//   dict_codes  — dictionary strings: the codes decode under the mask and only
//                 the surviving keys are gathered, sizing the output from the
//                 survivor count. Economics differ from the write-skipping
//                 routes: this wins 2.1-2.6x at ALL selectivities (it skips the
//                 full string materialization round trip), so the selectivity
//                 ceiling does NOT apply to a batch with such an output.
//   str_split   — str_split strings: masked offsets reconstruction plus a
//                 survivor-only chars byte-gather. Shaped like the dictionary
//                 case (skipped char materialization scales with selectivity)
//                 but WEAKER at ~1-char average widths, so it stays under the
//                 ordinary selectivity ceiling rather than taking the
//                 dictionary exemption — move it only if measurements say so.
//
// `full` IS "not compactable", so there is no separate capability flag to keep
// consistent with the route: the two cannot disagree. Never serialized — the
// values live only inside one decompress call.
enum class decode_route : uint8_t {
  full = 0,
  bitpack_mask,
  delta_mask,
  dict_codes,
  str_split,
};

// True for routes wave 2 decodes compacted in one pass. `full` is NOT in this
// set — it is admitted separately through the full-decode + survivor-gather
// path at low selectivity.
constexpr bool route_decodes_compacted(decode_route r) { return r != decode_route::full; }

// One scan conjunct resolved to a decoded-domain range on a bitpack column.
struct filter_column_directive {
  std::size_t column;    // index into the decompress call's `selected` span
  range_predicate pred;  // inclusive [lo,hi] in the decoded integer domain
};

// One dynamic MEMBERSHIP conjunct (a join build's in_list / cuco set / Bloom
// over a scan key column). Wave 1 decodes the key column full
// width, invokes `probe` (device-side membership test -> BOOL8, nonzero =
// keep, NO null mask, exactly the batch's row count) and ANDs the packed
// result into the batch mask; wave 2 compacts everything else as usual.
// Probe contract (same as every wave-1 launcher): ALL device work enqueued on
// the given stream before returning — no internal stream hops, no host sync
// required; the closure must PIN the filter structure it captures (e.g. a
// shared_ptr to the published device set) for the duration of the
// decompress_scan_filter call. Directives snapshot the filter SET per batch —
// the converter must never hand a probe that reads mutable live state.
struct membership_filter_directive {
  std::size_t column;  // key column, indexes into `selected`
  std::function<std::unique_ptr<cudf::column>(
    cudf::column_view keys, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)>
    probe;
};

// One scan conjunct answered off a dictionary's key set: wave 1 decodes the
// column's BOOL8 predicate result (the decode_predicate path — indices unpack +
// key lookup, no chars gather), converts it to packed mask words
// (mask_from_bool8) and ANDs it into the batch mask like any other source.
// Partial coverage is the CALLER's contract: when the mask does not carry the
// whole conjunction, the batch must stay untagged so the residual filter still
// runs post-decode.
struct bool8_filter_directive {
  std::size_t column;                   // index into `selected`; dictionary-rooted plan
  std::vector<std::string> equals_any;  // decode_predicate payload (equality / IN)
};

// The whole-batch request. The caller may build one that carries only PART of
// the scan's conjunction, in which case it must leave the batch untagged so the
// residual filter still runs; the orchestrator re-checks every per-plan
// precondition and falls back to the unfiltered path when any fail.
// Source-count cap: ranges + equalities + probes <= 8, one source per kernel.
struct scan_filter_request {
  std::vector<filter_column_directive> filters;                 // range conjuncts
  std::vector<bool8_filter_directive> bool8_filters;            // dictionary-answered equalities
  std::vector<membership_filter_directive> membership_filters;  // dynamic probes
  std::vector<decode_route> routes;                             // parallel to `selected`
  // Dynamic-filter-set version at request build (0 = static-only). Echoed on
  // the result so a caller that stopped using the filters can reconsider when a
  // later, tighter set arrives.
  uint64_t source_generation = 0;
};

// Outcome of a decompress_scan_filter call. `declined_unselective` is worth
// remembering per scan: selectivity is uniform across a scan's batches (SF1000
// zone-map study: unclustered, <1% variance), so ONE such batch predicts the
// rest — the caller drops the row selection from its remaining batches and
// stops paying the wave-1 + count insurance cost. The orchestrator itself stays
// stateless.
enum class scan_filter_status : uint8_t {
  refused = 0,               // gate off / nothing requested / a precondition failed
                             // (no device work was done)
  applied              = 1,  // the filtered decode produced the batch
  declined_unselective = 2,  // too many rows survived to pay for compacting; wave-1 cost paid,
                             // ordinary full-width output
  failed = 3,                // mid-flight failure; full-width output (exceptional)
};

// Selection data surviving the converter call, owned by the batch (freed with
// it). Buffers are stream-ordered RMM allocations; call set_stream() when
// rebinding the batch to the pipeline stream (same discipline as
// rebind_column_stream in compression_converters.cpp).
struct scan_filter_result {
  bool applied               = false;  // false => output is the ordinary full-width decode
  scan_filter_status status  = scan_filter_status::refused;  // always applied ⇔ status==applied
  uint64_t source_generation = 0;                            // echo of the request
  int64_t num_rows           = 0;                            // pre-filter batch rows
  int64_t survivor_count     = -1;
  std::vector<decode_route> routes;  // EFFECTIVE per-output route (a requested
                                     // compacted route is demoted to `full` on
                                     // probe fail)
  rmm::device_buffer mask_words;     // uint32 x WordsFor(num_rows) (full chunk strips)
  rmm::device_buffer chunk_offsets;  // uint32 x (ChunksFor(num_rows)+1)
  rmm::device_buffer row_indices;    // int32 x survivor_count (empty when no
                                     // tier_b output or survivor_count == 0)

  selection_mask view()
  {
    return selection_mask{static_cast<uint32_t*>(mask_words.data()),
                          num_rows,
                          survivor_count,
                          static_cast<uint32_t*>(chunk_offsets.data())};
  }

  void set_stream(rmm::cuda_stream_view stream)
  {
    if (mask_words.size() != 0) mask_words.set_stream(stream);
    if (chunk_offsets.size() != 0) chunk_offsets.set_stream(stream);
    if (row_indices.size() != 0) row_indices.set_stream(stream);
  }
};

// ── Selection-wave device helpers (src/selection/selection_wave.cu) ─────────
// All are asynchronous on `stream` unless noted. Throws std::runtime_error on
// CUDA failures.

// dst[w] = src_words[0][w] & ... & src_words[num_srcs-1][w] over the FULL
// padded strip: pass num_words = WordsFor(num_rows) so tail words are combined
// too (0 & x = 0 keeps the tail-zero invariant). num_words must be a multiple
// of 4 (always true for 32-word chunk strips). dst may alias src_words[0].
// num_srcs in [1, 8].
void combine_masks_and(uint32_t* dst_words,
                       uint32_t const* const* src_words,
                       int num_srcs,
                       int64_t num_words,
                       rmm::cuda_stream_view stream);

// CNT wave: per-chunk popcount (word-per-thread, warp-reduced) + CUB exclusive
// scan into mask.chunk_offsets (pre-allocated, ChunksFor(num_rows)+1 entries),
// then a 1-thread tail kernel for the sentinel. Copies the total back to the
// host — this HOST-SYNCS `stream` once (the survivor count gates wave-2
// allocations). Fills mask.survivor_count and returns it.
int64_t run_selection_cnt(selection_mask& mask,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr);

// Expand the mask to ascending int32 survivor row ids (TierB gather map).
// Requires run_selection_cnt to have filled chunk_offsets/survivor_count.
// out_indices must hold >= mask.survivor_count entries.
void mask_to_row_indices(selection_mask const& mask,
                         int32_t* out_indices,
                         rmm::cuda_stream_view stream);

// BOOL8 -> packed mask adapter: flags is a
// BOOL8/uint8 device array of num_rows (nonzero = survivor, no null mask);
// writes the FULL WordsFor(num_rows) padded strip into mask_words (rows beyond
// num_rows ballot to 0, preserving the tail-zero invariant). The result is a
// normal AND-combine source.
void mask_from_bool8(uint8_t const* flags,
                     int64_t num_rows,
                     uint32_t* mask_words,
                     rmm::cuda_stream_view stream);

}  // namespace sirius::codegen
