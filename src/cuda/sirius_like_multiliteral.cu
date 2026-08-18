/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// sirius
#include <expression_evaluator/like_multiliteral.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

// standard library
#include <cstdint>

namespace sirius {
namespace {

constexpr int max_literals      = like_multiliteral_max_literals;
constexpr int max_chunks        = like_multiliteral_max_literal_bytes / 8;
constexpr int threads_per_block = 256;

/// One literal, pre-digested for the kernel: SWAR broadcast bytes for the candidate digram
/// plus masked little-endian u64 chunks for full verification.
struct literal_desc {
  uint64_t chunk_val[max_chunks];   ///< literal bytes, 8 per chunk, little-endian, zero-padded
  uint64_t chunk_mask[max_chunks];  ///< 0xFF per literal byte within each chunk
  uint64_t b0_bcast;                ///< first literal byte broadcast to all 8 lanes
  uint64_t b1_bcast;                ///< second literal byte broadcast (unused when len == 1)
  int32_t len;                      ///< literal byte length (1..max_literal_bytes)
  int32_t nchunks;                  ///< ceil(len / 8)
};

/// The whole pattern, passed to the kernel by value.
struct pattern_desc {
  literal_desc literals[max_literals];
  int32_t n;          ///< number of literals (1..max_literals)
  int32_t total_len;  ///< sum of literal lengths — rows shorter than this cannot match
};

/// SWAR byte-equality candidate mask: a SUPERSET of the equal-byte positions of @p x vs the
/// broadcast byte of @p bcast, as bit 7 per byte. Every byte equal to bcast is flagged, but
/// borrow propagation from the subtraction can set spurious bits on unequal bytes — callers
/// MUST verify candidates against the actual bytes before acting on them.
__device__ __forceinline__ uint64_t byte_eq_mask(uint64_t x, uint64_t bcast)
{
  uint64_t const v = x ^ bcast;  // equal bytes become 0
  return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
}

/**
 * @brief Unaligned little-endian u64 window at byte position @p p of an aligned u64 stream.
 *
 * The caller guarantees the low word overlaps valid bytes (p < words_end_pos); the high word
 * is loaded only when needed and only when it also overlaps valid bytes, so no aligned load
 * ever starts at or past @p words_end_pos. Positions past @p words_end_pos read as 0 only in
 * a substituted high word; within a genuinely loaded word they are whatever the buffer holds
 * — callers must mask or bounds-reject such bytes either way.
 */
__device__ __forceinline__ uint64_t read_u64_window(uint64_t const* __restrict__ words,
                                                    int64_t words_end_pos,
                                                    int64_t p)
{
  int64_t const w   = p >> 3;
  int const off     = static_cast<int>(p & 7) * 8;
  uint64_t const lo = words[w];
  if (off == 0) { return lo; }
  uint64_t const hi = ((w + 1) << 3) < words_end_pos ? words[w + 1] : 0;
  return (lo >> off) | (hi << (64 - off));
}

/// Candidate positions for a literal within the aligned word @p cur: a SUPERSET of the byte
/// positions k where the literal's first byte is at k and (len >= 2) its second byte follows
/// (possibly in @p next), as bit 7 per byte — byte_eq_mask false positives carry through, so
/// every candidate must be verified. @p b0 / @p b1 are the literal's first two bytes
/// broadcast to all 8 lanes.
__device__ __forceinline__ uint64_t
candidate_mask(uint64_t cur, uint64_t next, uint64_t b0, uint64_t b1, int32_t len)
{
  uint64_t const c1 = byte_eq_mask(cur, b0);
  if (len == 1) { return c1; }
  uint64_t const c2  = byte_eq_mask(cur, b1);
  uint64_t const c2n = byte_eq_mask(next, b1);
  return c1 & ((c2 >> 8) | (c2n << 56));
}

/// Full literal verification at byte position @p p (caller guarantees p + lit.len fits the row).
__device__ __forceinline__ bool verify_literal(uint64_t const* __restrict__ words,
                                               int64_t words_end_pos,
                                               int64_t p,
                                               literal_desc const& lit)
{
#pragma unroll
  for (int c = 0; c < max_chunks; ++c) {
    if (c == lit.nchunks) { break; }
    uint64_t const win = read_u64_window(words, words_end_pos, p + 8 * c);
    if ((win & lit.chunk_mask[c]) != lit.chunk_val[c]) { return false; }
  }
  return true;
}

/**
 * @brief Thread-per-row multi-literal LIKE matcher.
 *
 * Each thread streams its row's aligned u64 words exactly once, SWAR-detects digram
 * candidates for the literal it currently needs, and verifies full literals only at
 * candidate hits. Greedy leftmost occurrence per literal with the next search starting at
 * the end of the previous match — exact for `%lit1%...%litN%` semantics.
 *
 * All positions are absolute byte positions into the chars buffer (offsets are stored that
 * way even for sliced views). Aligned loads are guarded by `buf_end = offsets[nrows]`: a
 * word starting at or past it may lie outside the mapped allocation and reads as 0 instead
 * — such bytes are past every row's end, so they never influence a result.
 *
 * Known-benign sanitizer note: the last aligned word may over-read up to 7 bytes past the
 * logical chars size (within rmm's 256B allocation-alignment slack; fault-free because the
 * base is 8B-aligned so no load crosses a page boundary) — compute-sanitizer memcheck may
 * flag this on exactly-sized allocations; do not "fix" it with a per-load guard.
 */
template <typename OffT>
__global__ void like_multiliteral_kernel(uint64_t const* __restrict__ words,
                                         OffT const* __restrict__ offsets,
                                         cudf::size_type nrows,
                                         cudf::bitmask_type const* __restrict__ null_mask,
                                         cudf::size_type mask_offset,
                                         pattern_desc pat,
                                         bool invert,
                                         bool* __restrict__ out)
{
  auto const row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= nrows) { return; }

  bool match = false;
  bool const is_valid =
    null_mask == nullptr ||
    cudf::bit_is_set(null_mask, mask_offset + static_cast<cudf::size_type>(row));
  if (is_valid) {
    int64_t const buf_end = static_cast<int64_t>(offsets[nrows]);  // broadcast load
    int64_t const s       = static_cast<int64_t>(offsets[row]);
    int64_t const e       = static_cast<int64_t>(offsets[row + 1]);
    if (e - s >= pat.total_len) {  // rows shorter than the literals cannot match
      int li          = 0;
      int64_t min_pos = s;  // next literal may start no earlier than this
      // Hot fields of the current literal, kept in registers.
      uint64_t b0 = pat.literals[0].b0_bcast;
      uint64_t b1 = pat.literals[0].b1_bcast;
      int32_t len = pat.literals[0].len;

      int64_t w        = s >> 3;
      int64_t const we = (e + 7) >> 3;  // exclusive
      // Safe unguarded: s < e (total_len >= 1), so word w overlaps byte s < buf_end.
      uint64_t cur = words[w];
      for (; w < we && !match; ++w) {
        uint64_t const next = ((w + 1) << 3) < buf_end ? words[w + 1] : 0;
        int64_t const base  = w << 3;
        uint64_t cand       = candidate_mask(cur, next, b0, b1, len);
        while (cand) {
          int const k = __ffsll(static_cast<long long>(cand)) - 1;
          cand &= cand - 1;
          int64_t const p = base + (k >> 3);
          // Bytes before the row start / previous match, or literals overrunning the row
          // end (including digram second-bytes that belong to the next row), are rejected
          // here — candidate detection may fire on them, verification never sees them.
          if (p < min_pos || p + len > e) { continue; }
          if (!verify_literal(words, buf_end, p, pat.literals[li])) { continue; }
          min_pos = p + len;
          ++li;
          if (li == pat.n) {
            match = true;
            break;
          }
          b0  = pat.literals[li].b0_bcast;
          b1  = pat.literals[li].b1_bcast;
          len = pat.literals[li].len;
          // Rescan the current word for the next literal (its match may start here too;
          // positions before min_pos are filtered above).
          cand = candidate_mask(cur, next, b0, b1, len);
        }
        cur = next;
      }
    }
  }
  out[row] = match != invert;
}

/// Build the kernel-side literal descriptor from a host literal.
literal_desc make_literal_desc(std::string const& lit)
{
  literal_desc d{};
  d.len     = static_cast<int32_t>(lit.size());
  d.nchunks = (d.len + 7) / 8;
  for (int i = 0; i < d.len; ++i) {
    auto const byte = static_cast<uint64_t>(static_cast<unsigned char>(lit[i]));
    d.chunk_val[i / 8] |= byte << (8 * (i % 8));
    d.chunk_mask[i / 8] |= 0xFFULL << (8 * (i % 8));
  }
  d.b0_bcast = 0x0101010101010101ULL * static_cast<unsigned char>(lit[0]);
  d.b1_bcast = d.len >= 2 ? 0x0101010101010101ULL * static_cast<unsigned char>(lit[1]) : 0;
  return d;
}

}  // namespace

std::optional<like_multiliteral_pattern> classify_like_multiliteral(std::string_view pattern,
                                                                    std::string_view escape)
{
  if (!escape.empty()) { return std::nullopt; }  // escape semantics: cudf path
  if (pattern.size() < 2 || pattern.front() != '%' || pattern.back() != '%') {
    return std::nullopt;  // anchored (prefix/suffix/exact) patterns: cudf path
  }
  like_multiliteral_pattern result;
  std::string current;
  for (char const ch : pattern) {
    if (ch == '%') {
      if (!current.empty()) {
        if (current.size() > static_cast<size_t>(like_multiliteral_max_literal_bytes) ||
            result.literals.size() == static_cast<size_t>(like_multiliteral_max_literals)) {
          return std::nullopt;
        }
        result.literals.push_back(std::move(current));
        current.clear();
      }
      continue;  // consecutive '%' collapse
    }
    auto const byte = static_cast<unsigned char>(ch);
    if (ch == '_' || byte == 0 || byte >= 0x80) {
      return std::nullopt;  // '_' wildcard / NUL / non-ASCII: cudf path
    }
    current.push_back(ch);
  }
  // The pattern ends with '%', so no literal is pending here.
  if (result.literals.empty()) { return std::nullopt; }  // pure '%'/'%%': cudf path
  return result;
}

std::unique_ptr<cudf::column> like_multiliteral(cudf::strings_column_view const& input,
                                                like_multiliteral_pattern const& pattern,
                                                bool invert,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<int64_t>(pattern.literals.size());
  if (n < 1 || n > max_literals) {
    throw internal_exception("[like_multiliteral] pattern with {} literals was not classified out",
                             n);
  }

  auto const nrows = input.size();
  if (nrows == 0) {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  // Column-layout eligibility: the kernel needs an 8-byte-aligned chars base for its aligned
  // u64 loads, and offsets of a known width. Anything else falls back to cudf.
  auto const offsets_id = input.offsets().type().id();
  if (offsets_id != cudf::type_id::INT32 && offsets_id != cudf::type_id::INT64) { return nullptr; }
  char const* chars_base = input.chars_begin(stream);
  if (reinterpret_cast<uintptr_t>(chars_base) & 7) { return nullptr; }

  pattern_desc pat{};
  pat.n = static_cast<int32_t>(n);
  for (int64_t i = 0; i < n; ++i) {
    auto const& lit = pattern.literals[i];
    if (lit.empty() || lit.size() > static_cast<size_t>(like_multiliteral_max_literal_bytes)) {
      throw internal_exception("[like_multiliteral] literal of {} bytes was not classified out",
                               lit.size());
    }
    pat.literals[i] = make_literal_desc(lit);
    pat.total_len += pat.literals[i].len;
  }

  auto result = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                          nrows,
                                          cudf::copy_bitmask(input.parent(), stream, mr),
                                          input.null_count(),
                                          stream,
                                          mr);
  auto* out   = result->mutable_view().data<bool>();

  auto const* null_mask = input.null_count() > 0 ? input.null_mask() : nullptr;
  auto const* words     = reinterpret_cast<uint64_t const*>(chars_base);
  auto const num_blocks = static_cast<unsigned>(
    (static_cast<int64_t>(nrows) + threads_per_block - 1) / threads_per_block);

  if (offsets_id == cudf::type_id::INT32) {
    auto const* offsets = input.offsets().data<int32_t>() + input.offset();
    like_multiliteral_kernel<int32_t><<<num_blocks, threads_per_block, 0, stream.value()>>>(
      words, offsets, nrows, null_mask, input.offset(), pat, invert, out);
  } else {
    auto const* offsets = input.offsets().data<int64_t>() + input.offset();
    like_multiliteral_kernel<int64_t><<<num_blocks, threads_per_block, 0, stream.value()>>>(
      words, offsets, nrows, null_mask, input.offset(), pat, invert, out);
  }
  CUDF_CHECK_CUDA(stream.value());
  return result;
}

}  // namespace sirius
