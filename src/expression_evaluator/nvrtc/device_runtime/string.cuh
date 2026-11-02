/*
 * Copyright 2025, Sirius Contributors.
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

#pragma once

/**
 * @file
 * @brief The @c str_view register and the string functions: comparison, lengths, prefix/suffix,
 *        contains, the generic backtracking LIKE matcher, the SWAR multi-literal LIKE matcher, and
 *        substring. The byte, UTF-8 and word-at-a-time primitives they share come first.
 *
 * Fragment 5 of the NVRTC expression evaluator device runtime.
 */

#include "codegen/stdint_shim.hpp"

//===----------STR_VIEW REGISTER----------===//
struct str_view {
  char const* ptr;
  int32_t len;
  char const*
    limit;  ///< End of the readable allocation the string lives in; word-at-a-time matchers may
            ///< read the aligned words that overlap [ptr, ptr + len) only up to this bound.
};

//===-------------------------------------------------------------------------===//
// SHARED PRIMITIVES:
//  - UTF-8 BYTES
//  - BYTE SEARCH
//  - SWAR WORDS
//===-------------------------------------------------------------------------===//

//===----------UTF-8 BYTES----------===//
/**
 * @brief Whether @p byte continues a multi-byte UTF-8 code point rather than starting one.
 */
__device__ __forceinline__ bool is_utf8_continuation(char byte)
{
  return (static_cast<uint8_t>(byte) & 0xC0u) == 0x80u;
}

__device__ __forceinline__ int32_t next_utf8_char(char const* s, int32_t slen, int32_t pos)
{
  int32_t next = pos + 1;
  while (next < slen && is_utf8_continuation(s[next])) {
    ++next;
  }
  return next;
}

//===----------BYTE SEARCH----------===//
__device__ __forceinline__ int32_t find_byte_plain(char const* s,
                                                   int32_t slen,
                                                   int32_t start,
                                                   char target)
{
  int32_t pos = start;
  while (pos < slen && s[pos] != target) {
    ++pos;
  }
  return pos;
}

//===----------SWAR WORDS----------===//
__device__ __forceinline__ uint64_t swar_byte_eq_mask(uint64_t x, uint64_t bcast)
{
  uint64_t const v = x ^ bcast;
  return (v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL;
}

/// @brief Aligned 8-byte word at @p addr, zero-padded past the allocation bound @p limit.
__device__ __forceinline__ uint64_t load_bounded_word(uintptr_t addr, uintptr_t limit)
{
  if (addr + 8U <= limit) { return *reinterpret_cast<uint64_t const*>(addr); }
  uint64_t value = 0ULL;
  for (uintptr_t b = addr; b < limit; ++b) {
    value |= static_cast<uint64_t>(*reinterpret_cast<uint8_t const*>(b)) << ((b - addr) << 3);
  }
  return value;
}

/**
 * @brief Unaligned 8-byte window at byte address @p p assembled from two bounded aligned words.
 */
__device__ __forceinline__ uint64_t read_window(uintptr_t p, uintptr_t limit)
{
  uintptr_t const a = p & ~static_cast<uintptr_t>(7u);
  int const bit_off = static_cast<int>(p & 7u) << 3;
  uint64_t const lo = load_bounded_word(a, limit);
  if (bit_off == 0) { return lo; }
  uint64_t const hi = load_bounded_word(a + 8u, limit);
  return (lo >> bit_off) | (hi << (64 - bit_off));
}

/**
 * @brief 8-byte window at @p byte_pos of a string, zero-padded past the string's own end.
 */
__device__ __forceinline__ uint64_t load_aligned_str_word(char const* base_ptr,
                                                          int32_t total_len,
                                                          int32_t byte_pos)
{
  if (byte_pos >= total_len) { return 0ULL; }
  uintptr_t const addr = reinterpret_cast<uintptr_t>(base_ptr + byte_pos);
  if ((addr & 7u) == 0u && (total_len - byte_pos) >= 8) {
    return *reinterpret_cast<uint64_t const*>(addr);
  }
  uintptr_t const aligned_addr = addr & ~static_cast<uintptr_t>(7u);
  int32_t const shift          = static_cast<int32_t>(addr & 7u) << 3;
  uintptr_t const end_addr     = reinterpret_cast<uintptr_t>(base_ptr + total_len);
  if (aligned_addr + 8u <= end_addr) {
    uint64_t const lo = *reinterpret_cast<uint64_t const*>(aligned_addr);
    if (shift == 0) { return lo; }
    if (aligned_addr + 16u <= end_addr) {
      uint64_t const hi = *reinterpret_cast<uint64_t const*>(aligned_addr + 8u);
      return (lo >> shift) | (hi << (64 - shift));
    }
  }
  uint64_t val      = 0ULL;
  int32_t const rem = (total_len - byte_pos) < 8 ? (total_len - byte_pos) : 8;
  for (int32_t i = 0; i < rem; ++i) {
    val |= static_cast<uint64_t>(static_cast<uint8_t>(base_ptr[byte_pos + i])) << (i << 3);
  }
  return val;
}

__device__ __forceinline__ int32_t find_byte_swar(char const* s,
                                                  int32_t slen,
                                                  int32_t start,
                                                  char target)
{
  if (start >= slen) { return slen; }
  uint64_t const bcast = 0x0101010101010101ULL * static_cast<uint8_t>(target);
  int32_t pos          = start;
  while (pos + 8 <= slen) {
    uint64_t const w = load_aligned_str_word(s, slen, pos);
    uint64_t cand    = swar_byte_eq_mask(w, bcast);
    while (cand != 0ULL) {
      int32_t const lane = (__ffsll(static_cast<long long>(cand)) - 1) >> 3;
      cand &= (cand - 1ULL);
      if (s[pos + lane] == target) { return pos + lane; }
    }
    pos += 8;
  }
  while (pos < slen) {
    if (s[pos] == target) { return pos; }
    ++pos;
  }
  return slen;
}

//===-------------------------------------------------------------------------===//
// STRING FUNCTIONS:
//  - COMPARISON
//  - LENGTH
//  - PREFIX / SUFFIX
//  - CONTAINS
//  - LIKE (GENERIC)
//  - LIKE (SWAR MULTI-LITERAL)
//  - SUBSTRING
//===-------------------------------------------------------------------------===//

//===----------STRING COMPARISON----------===//
__device__ __forceinline__ int str_view_cmp(str_view a, str_view b)
{
  int32_t const min_len = a.len < b.len ? a.len : b.len;
  for (int32_t i = 0; i < min_len; ++i) {
    uint8_t const ca = static_cast<uint8_t>(a.ptr[i]);
    uint8_t const cb = static_cast<uint8_t>(b.ptr[i]);
    if (ca != cb) { return (ca < cb) ? -1 : 1; }
  }
  return (a.len < b.len) ? -1 : ((a.len > b.len) ? 1 : 0);
}
__device__ __forceinline__ bool operator==(str_view a, str_view b)
{
  if (a.len != b.len) { return false; }
  for (int32_t i = 0; i < a.len; ++i) {
    if (a.ptr[i] != b.ptr[i]) { return false; }
  }
  return true;
}
__device__ __forceinline__ bool operator!=(str_view a, str_view b) { return !(a == b); }
__device__ __forceinline__ bool operator<(str_view a, str_view b) { return str_view_cmp(a, b) < 0; }
__device__ __forceinline__ bool operator<=(str_view a, str_view b)
{
  return str_view_cmp(a, b) <= 0;
}
__device__ __forceinline__ bool operator>(str_view a, str_view b) { return str_view_cmp(a, b) > 0; }
__device__ __forceinline__ bool operator>=(str_view a, str_view b)
{
  return str_view_cmp(a, b) >= 0;
}

//===----------STRING LENGTH----------===//
template <typename OutT>
__device__ __forceinline__ reg_val<OutT> eval_strlen(reg_val<str_view> in)
{
  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }
  return reg_val<OutT>{static_cast<OutT>(in.val.len), true, false};
}

template <typename OutT>
__device__ __forceinline__ reg_val<OutT> eval_utf8_length(reg_val<str_view> in)
{
  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }
  int32_t count = 0;
  for (int32_t i = 0; i < in.val.len; ++i) {
    if (!is_utf8_continuation(in.val.ptr[i])) { ++count; }
  }
  return reg_val<OutT>{static_cast<OutT>(count), true, false};
}

//===----------PREFIX / SUFFIX----------===//
__device__ __forceinline__ reg_val<bool> eval_prefix(reg_val<str_view> s, reg_val<str_view> pat)
{
  bool const ovf   = s.overflow || pat.overflow;
  bool const valid = s.valid && pat.valid;
  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }
  if (pat.val.len > s.val.len) { return reg_val<bool>{false, true, false}; }
  for (int32_t i = 0; i < pat.val.len; ++i) {
    if (s.val.ptr[i] != pat.val.ptr[i]) { return reg_val<bool>{false, true, false}; }
  }
  return reg_val<bool>{true, true, false};
}

__device__ __forceinline__ reg_val<bool> eval_suffix(reg_val<str_view> s, reg_val<str_view> pat)
{
  bool const ovf   = s.overflow || pat.overflow;
  bool const valid = s.valid && pat.valid;
  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }
  if (pat.val.len > s.val.len) { return reg_val<bool>{false, true, false}; }
  int32_t const start = s.val.len - pat.val.len;
  for (int32_t i = 0; i < pat.val.len; ++i) {
    if (s.val.ptr[start + i] != pat.val.ptr[i]) { return reg_val<bool>{false, true, false}; }
  }
  return reg_val<bool>{true, true, false};
}

//===----------CONTAINS----------===//
__device__ __forceinline__ reg_val<bool> eval_contains(reg_val<str_view> s, reg_val<str_view> pat)
{
  bool const ovf   = s.overflow || pat.overflow;
  bool const valid = s.valid && pat.valid;
  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }
  if (pat.val.len <= 0) { return reg_val<bool>{true, true, false}; }
  if (s.val.len < pat.val.len) { return reg_val<bool>{false, true, false}; }
  char const first    = pat.val.ptr[0];
  char const last     = pat.val.ptr[pat.val.len - 1];
  int32_t const limit = s.val.len - pat.val.len;
  int32_t i           = find_byte_swar(s.val.ptr, limit + 1, 0, first);
  while (i <= limit) {
    if (s.val.ptr[i + pat.val.len - 1] == last) {
      bool ok   = true;
      int32_t j = 1;
      for (; j + 8 <= pat.val.len - 1; j += 8) {
        if (load_aligned_str_word(s.val.ptr, s.val.len, i + j) !=
            load_aligned_str_word(pat.val.ptr, pat.val.len, j)) {
          ok = false;
          break;
        }
      }
      for (; ok && j < pat.val.len - 1; ++j) {
        if (s.val.ptr[i + j] != pat.val.ptr[j]) {
          ok = false;
          break;
        }
      }
      if (ok) { return reg_val<bool>{true, true, false}; }
    }
    i = find_byte_swar(s.val.ptr, limit + 1, i + 1, first);
  }
  return reg_val<bool>{false, true, false};
}

//===----------LIKE (GENERIC)----------===//
__device__ __forceinline__ bool raw_like(str_view s, str_view pat)
{
  int32_t si      = 0;
  int32_t pi      = 0;
  int32_t star_pi = -1;
  int32_t star_si = -1;
  while (si < s.len) {
    char const pc = (pi < pat.len) ? pat.ptr[pi] : '\0';
    if (pi < pat.len && pc == '%') {
      while (pi < pat.len && pat.ptr[pi] == '%') {
        ++pi;
      }
      if (pi == pat.len) { return true; }
      star_pi = pi;
      if (pat.ptr[star_pi] != '_') { si = find_byte_plain(s.ptr, s.len, si, pat.ptr[star_pi]); }
      star_si = si;
    } else if (pi < pat.len && pc == '_') {
      si = next_utf8_char(s.ptr, s.len, si);
      ++pi;
    } else if (pi < pat.len && s.ptr[si] == pc) {
      ++si;
      ++pi;
    } else if (star_pi != -1) {
      pi = star_pi;
      if (pat.ptr[star_pi] != '_') {
        star_si = find_byte_plain(s.ptr, s.len, star_si + 1, pat.ptr[star_pi]);
      } else {
        star_si = next_utf8_char(s.ptr, s.len, star_si);
      }
      si = star_si;
    } else {
      return false;
    }
  }
  while (pi < pat.len && pat.ptr[pi] == '%') {
    ++pi;
  }
  return pi == pat.len;
}

template <bool Invert>
__device__ __forceinline__ reg_val<bool> eval_like(reg_val<str_view> s, reg_val<str_view> pat)
{
  bool const ovf   = s.overflow || pat.overflow;
  bool const valid = s.valid && pat.valid;
  if (!valid || ovf) { return reg_val<bool>{false, valid, ovf}; }
  bool const m = raw_like(s.val, pat.val);
  return reg_val<bool>{Invert ? !m : m, true, false};
}

//===----------LIKE (SWAR MULTI-LITERAL)----------===//
// Mirrors @c sirius::detail::like_multiliteral_{literal,pattern}_desc.
struct like_literal_desc {
  uint64_t chunk_val[4];
  uint64_t chunk_mask[4];
  uint64_t b0_bcast;
  uint64_t b1_bcast;
  int32_t len;
  int32_t nchunks;
};

struct like_pattern_desc {
  like_literal_desc literals[4];
  int32_t n;
  int32_t total_len;
};

/**
 * @brief Superset of the byte lanes where the literal's first byte sits at lane k and (len >= 2)
 *        its second byte follows; bit 7 of each byte marks a candidate (SWAR false positives are
 *        filtered by verification).
 */
__device__ __forceinline__ uint64_t
like_candidates(uint64_t cur, uint64_t next, uint64_t b0, uint64_t b1, int32_t len)
{
  uint64_t const c1 = swar_byte_eq_mask(cur, b0);
  if (len == 1) { return c1; }
  uint64_t const c2  = swar_byte_eq_mask(cur, b1);
  uint64_t const c2n = swar_byte_eq_mask(next, b1);
  return c1 & ((c2 >> 8) | (c2n << 56));
}

__device__ __forceinline__ like_literal_desc const& like_literal_at(like_pattern_desc const& pat,
                                                                    int32_t li)
{
  switch (li) {
    case 0: return pat.literals[0];
    case 1: return pat.literals[1];
    case 2: return pat.literals[2];
    default: return pat.literals[3];
  }
}

__device__ __forceinline__ bool like_verify(uintptr_t p,
                                            uintptr_t limit,
                                            like_literal_desc const& lit)
{
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    if (c == lit.nchunks) { break; }
    uint64_t const win = read_window(p + static_cast<uintptr_t>(8 * c), limit);
    if ((win & lit.chunk_mask[c]) != lit.chunk_val[c]) { return false; }
  }
  return true;
}

/**
 *  @brief `%lit1%lit2%...%` (and contains = one literal): scan the string one aligned 8-byte word
 *         at a time, locating candidate starts of the current literal with SWAR digram masks and
 *         verifying them in order. Each literal must start at or after the previous match's end.
 */
template <bool Invert>
__device__ __forceinline__ reg_val<bool> eval_like_multiliteral(reg_val<str_view> s,
                                                                like_pattern_desc const& pat)
{
  if (!s.valid || s.overflow) { return reg_val<bool>{false, s.valid, s.overflow}; }
  bool match = false;
  if (s.val.len >= pat.total_len) {
    uintptr_t const start = reinterpret_cast<uintptr_t>(s.val.ptr);
    uintptr_t const end   = start + static_cast<uintptr_t>(s.val.len);
    uintptr_t const limit = reinterpret_cast<uintptr_t>(s.val.limit);
    int32_t li            = 0;
    uintptr_t min_pos     = start;
    uint64_t b0           = pat.literals[0].b0_bcast;
    uint64_t b1           = pat.literals[0].b1_bcast;
    int32_t len           = pat.literals[0].len;
    uintptr_t w           = start & ~static_cast<uintptr_t>(7u);
    uint64_t cur          = load_bounded_word(w, limit);
    for (; w < end && !match; w += 8u) {
      uint64_t const next = load_bounded_word(w + 8u, limit);
      uint64_t cand       = like_candidates(cur, next, b0, b1, len);
      while (cand != 0ULL) {
        int const k = __ffsll(static_cast<long long>(cand)) - 1;
        cand &= cand - 1ULL;
        uintptr_t const p = w + static_cast<uintptr_t>(k >> 3);
        if (p < min_pos || p + static_cast<uintptr_t>(len) > end) { continue; }
        if (!like_verify(p, limit, like_literal_at(pat, li))) { continue; }
        min_pos = p + static_cast<uintptr_t>(len);
        ++li;
        if (li == pat.n) {
          match = true;
          break;
        }
        like_literal_desc const& nl = like_literal_at(pat, li);
        b0                          = nl.b0_bcast;
        b1                          = nl.b1_bcast;
        len                         = nl.len;
        cand                        = like_candidates(cur, next, b0, b1, len);
      }
      cur = next;
    }
  }
  return reg_val<bool>{match != Invert, true, false};
}

//===----------SUBSTRING----------===//
__device__ __forceinline__ str_view slice_utf8_forward(str_view s,
                                                       int32_t start_char,
                                                       int32_t stop_char)
{
  if (stop_char <= start_char || s.len <= 0) { return str_view{s.ptr, 0, s.limit}; }
  int32_t char_idx   = 0;
  int32_t byte_start = s.len;
  int32_t byte_stop  = s.len;
  for (int32_t b = 0; b < s.len; ++b) {
    if (!is_utf8_continuation(s.ptr[b])) {
      if (char_idx == start_char) { byte_start = b; }
      if (char_idx == stop_char) {
        byte_stop = b;
        break;
      }
      ++char_idx;
    }
  }
  if (byte_start >= byte_stop) { return str_view{s.ptr, 0, s.limit}; }
  return str_view{s.ptr + byte_start, byte_stop - byte_start, s.limit};
}

__device__ __forceinline__ reg_val<str_view> eval_substring(reg_val<str_view> in,
                                                            int32_t start_char,
                                                            int32_t stop_char)
{
  if (!in.valid || in.overflow) {
    return reg_val<str_view>{str_view{nullptr, 0, nullptr}, in.valid, in.overflow};
  }
  if (start_char >= 0 && stop_char >= 0) {
    return reg_val<str_view>{slice_utf8_forward(in.val, start_char, stop_char), true, false};
  }
  int32_t char_len = 0;
  for (int32_t b = 0; b < in.val.len; ++b) {
    if (!is_utf8_continuation(in.val.ptr[b])) { ++char_len; }
  }
  int32_t r_start = (start_char < 0) ? (char_len + start_char) : start_char;
  if (r_start < 0) { r_start = 0; }
  if (r_start > char_len) { r_start = char_len; }
  int32_t r_stop = (stop_char < 0) ? (char_len + stop_char) : stop_char;
  if (r_stop < 0) { r_stop = 0; }
  if (r_stop > char_len) { r_stop = char_len; }
  return reg_val<str_view>{slice_utf8_forward(in.val, r_start, r_stop), true, false};
}
