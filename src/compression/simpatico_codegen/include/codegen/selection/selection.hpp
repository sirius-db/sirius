// SPDX-License-Identifier: Apache-2.0
//
// Shared selection types for the fused scan-filter pipeline
// (env gate: SIRIUS_EXP_FUSED_SCAN_FILTER).
//
// Owned by W4 (wave orchestration). Included by:
//   - W1 JIT decode variants (K1 mask-out, K3 mask-consume) as kernel-arg types,
//   - W2 predicate extraction (range_predicate / scan_filter_request production),
//   - W3 batch contract (survivor-count-first allocation, TierB gathers),
//   - W4 selection_wave.cu (AND-combine, CNT, mask->indices).
//
// This header is host-side plumbing: PODs holding device pointers, plus the
// request/result structs that cross the converter boundary. It must stay
// includable from both .cpp and .cu TUs, and must NOT be baked into JIT
// (NVRTC) source — predicate constants travel as kernel parameters (a new
// NVRTC compile per literal would defeat the JIT cache).

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius::codegen {

// Number of rows covered by one CNT chunk (one chunk_offsets entry). Matches
// the simpatico bitpack chunk (codegen::kChunkSize) and the microbench
// (scratchpad/fusebench/fusebench.cu): K3 uses chunk_offsets[chunk] as the
// compacted output base for its chunk.
inline constexpr int64_t SELECTION_CHUNK_ROWS = 1024;

// 1 bit per row, 1 = survivor. Words are uint32_t, row r -> word r/32, bit r%32.
// Tail bits beyond num_rows MUST be zero (CNT and gather rely on it).
//
// SIZE = ChunksFor(num_rows) * 32 = ceil(num_rows/1024) * 32 words: every
// kernel touching the mask (K1 producer, AND-combine, CNT, mask->indices, K3
// consumer) addresses a FULL 32-word strip per 1024-row chunk, so a partial
// tail chunk still owns 32 words. K1 keeps the tail zero by construction
// (out-of-range lanes ballot to 0). Sizing by ceil(num_rows/32) is an
// out-of-bounds write for any num_rows not a multiple of 1024.
struct selection_mask {
  uint32_t* words = nullptr;  // device, WordsFor(num_rows) words, 128B-aligned
  int64_t num_rows = 0;
  int64_t survivor_count = -1;      // -1 until CNT wave ran
  uint32_t* chunk_offsets = nullptr;  // device, exclusive prefix sum of survivors per
                                      // 1024-row chunk (length = ChunksFor(num_rows)+1);
                                      // null until CNT ran.

  // Sizing helpers (host-side).
  static constexpr int64_t ChunksFor(int64_t num_rows) {
    return (num_rows + SELECTION_CHUNK_ROWS - 1) / SELECTION_CHUNK_ROWS;
  }
  static constexpr int64_t WordsFor(int64_t num_rows) {  // full 32-word chunk strips
    return ChunksFor(num_rows) * (SELECTION_CHUNK_ROWS / 32);
  }
  static constexpr int64_t AllocWordsFor(int64_t num_rows) {  // alias of WordsFor
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

// ── Converter-boundary contract (W2 produces, W4 consumes, W3 unpacks) ───────

// Output-column tier for wave 2.
//   tier_a       — bitpack K3: decoded straight to a compacted
//                  survivor_count-row column (shipped, iteration 1).
//   tier_b       — full decode + survivor gather. Measured strictly worse than
//                  the classic path (q1 +43.5%, q5 +6.2%): RULE 1 refuses any
//                  batch with a tier_b output.
//   tier_a_delta — delta->bitpack root masked decode (live: W1 delta variant +
//                  W3 dispatch; same decode_selection{compact_capable} route
//                  as tier_a).
//   tier_dict_k5 — dictionary-string masked decode (live: W3 general route =
//                  mask->codes K3 + survivor-only key gather, count-first
//                  strings out; W1 fast path for constant-width identity-
//                  stored keys). Economics differ from the write-skip tiers:
//                  the microbench F5 case wins 2.1-2.6x at ALL selectivities
//                  (it skips the full string materialization round trip), so
//                  RULE 2's selectivity bail does NOT apply to batches with a
//                  tier_dict_k5 output.
// Values 0/1 are stable (shipped); new tiers append.
enum class output_tier : uint8_t {
  tier_a       = 0,
  tier_b       = 1,
  tier_a_delta = 2,
  tier_dict_k5 = 3,
};

// True for tiers wave 2 can decode compacted (RULE 1's admission set).
constexpr bool tier_is_fused_capable(output_tier t)
{
  return t == output_tier::tier_a || t == output_tier::tier_a_delta ||
         t == output_tier::tier_dict_k5;
}

// One scan conjunct resolved to a decoded-domain range on a bitpack column.
struct filter_column_directive {
  std::size_t column;   // index into the decompress call's `selected` span
  range_predicate pred; // inclusive [lo,hi] in the decoded integer domain
};

// One scan conjunct resolved by the shipped dict-code BOOL8 pushdown (q19-style
// mixed masks, iteration 3): wave 1 decodes the column's BOOL8 predicate result
// (decode_predicate path — indices unpack + key LUT, no chars gather), converts
// it to packed mask words (mask_from_bool8) and ANDs it into the batch mask
// like any K1 source. Partial-conjunction coverage is the CALLER's contract:
// when the mask does not cover the whole conjunction, the batch must stay
// untagged so the residual filter still runs post-decode (W2's design).
struct bool8_filter_directive {
  std::size_t column;                   // index into `selected`; dictionary-rooted plan
  std::vector<std::string> equals_any;  // decode_predicate payload (equality / IN)
};

// The whole-batch fused request. W2 only builds one when EVERY conjunct on the
// scanned table is decode-resolvable (iteration 1 rule; iteration 3 relaxes
// this to partial coverage via bool8_filters + an untagged batch); W4 re-checks
// the per-plan preconditions and falls back to the classic path when any fail.
struct scan_filter_request {
  std::vector<filter_column_directive> filters;      // range conjuncts (K1 sources)
  std::vector<bool8_filter_directive> bool8_filters;  // dict-code conjuncts (BOOL8 sources)
  std::vector<output_tier> tiers;                     // parallel to `selected`
};

// Per-selected-column directive, the shape W2's converter builder emits (see
// STATUS-W2). Adapter below folds a span of these into a scan_filter_request.
struct column_decode_directive {
  bool has_range = false;       // a decode-resolvable range exists on this column
  range_predicate range{0, 0};  // valid iff has_range
  bool in_scan_mask = false;    // participates in wave-1 mask production (K1)
  bool compact_output = false;  // TierA (K3 compacted) vs TierB (full + gather)
};

// Build the wave request from per-column directives (parallel to `selected`).
inline scan_filter_request make_scan_filter_request(
  std::vector<column_decode_directive> const& columns)
{
  scan_filter_request req;
  req.tiers.reserve(columns.size());
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const& c = columns[i];
    if (c.has_range && c.in_scan_mask) req.filters.push_back({i, c.range});
    req.tiers.push_back(c.compact_output ? output_tier::tier_a : output_tier::tier_b);
  }
  return req;
}

// Outcome of a decompress_scan_filter call. `bailed_high_selectivity` is the
// bail-memoization key: per-batch selectivity is uniform across a scan's
// batches (SF1000 zone-map study: unclustered, <1% variance), so ONE RULE-2
// bail predicts all remaining batches — the scan side sets a per-operator flag
// on seeing it and strips the range pushdown from subsequent batches
// (set_range_pushdown({}, false) before convert), dropping the wave-1+CNT
// insurance cost from every-batch to once-per-scan. Provider and converter
// stay stateless.
enum class scan_filter_status : uint8_t {
  refused = 0,                  // gate off / no directives / precondition / RULE-1
                                // (no device work was done)
  applied = 1,                  // fused pipeline produced the batch
  bailed_high_selectivity = 2,  // RULE-2 post-CNT bail (wave-1 cost paid, classic output)
  failed = 3,                   // mid-flight failure; classic output (exceptional)
};

// Selection data surviving the converter call, owned by the batch (freed with
// it). Buffers are stream-ordered RMM allocations; call set_stream() when
// rebinding the batch to the pipeline stream (same discipline as
// rebind_column_stream in compression_converters.cpp).
struct scan_filter_result {
  bool applied = false;      // false => output is the classic full-width decode
  scan_filter_status status = scan_filter_status::refused;  // always applied ⇔ status==applied
  int64_t num_rows = 0;      // pre-filter batch rows
  int64_t survivor_count = -1;
  std::vector<output_tier> tiers;    // EFFECTIVE per-output tier (W4 may demote
                                     // a requested tier_a to tier_b on probe fail)
  rmm::device_buffer mask_words;     // uint32 x WordsFor(num_rows) (full chunk strips)
  rmm::device_buffer chunk_offsets;  // uint32 x (ChunksFor(num_rows)+1)
  rmm::device_buffer row_indices;    // int32 x survivor_count (empty when no
                                     // tier_b output or survivor_count == 0)

  selection_mask view() {
    return selection_mask{static_cast<uint32_t*>(mask_words.data()),
                          num_rows,
                          survivor_count,
                          static_cast<uint32_t*>(chunk_offsets.data())};
  }

  void set_stream(rmm::cuda_stream_view stream) {
    if (mask_words.size() != 0) mask_words.set_stream(stream);
    if (chunk_offsets.size() != 0) chunk_offsets.set_stream(stream);
    if (row_indices.size() != 0) row_indices.set_stream(stream);
  }
};

// ── Selection-wave device helpers (src/selection/selection_wave.cu, W4) ──────
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

// BOOL8 -> packed mask adapter (mixed-mask combine, iteration 3): flags is a
// BOOL8/uint8 device array of num_rows (nonzero = survivor, no null mask);
// writes the FULL WordsFor(num_rows) padded strip into mask_words (rows beyond
// num_rows ballot to 0, preserving the tail-zero invariant). The result is a
// normal AND-combine source.
void mask_from_bool8(uint8_t const* flags,
                     int64_t num_rows,
                     uint32_t* mask_words,
                     rmm::cuda_stream_view stream);

}  // namespace sirius::codegen
