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

// Synthetic ALP / ALPRD segment builders for unit tests and microbenches.
// Byte-by-byte format reference: src/cuda/scan/gpu_decode_alp.cu.

#include <cuda/scan/gpu_decode_alp.cuh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace sirius::test::decode::alp {

inline constexpr uint32_t VEC = 1024;

/// Pack `values.size()` width-bit values LSB-first into a fresh byte buffer
/// sized to DuckDB's `BitpackingPrimitives::GetRequiredSize` rule (count
/// padded up to a 32-value group, then `padded * width / 8` bytes).
template <typename T>
inline std::vector<uint8_t> pack_lsb(std::vector<T> const& values, uint32_t width)
{
  static_assert(std::is_unsigned_v<T>, "pack_lsb expects an unsigned mantissa type");
  if (width == 0u) { return std::vector<uint8_t>(0u, 0u); }
  uint32_t count  = static_cast<uint32_t>(values.size());
  uint32_t r      = count & 31u;
  uint32_t padded = r ? (count + 32u - r) : count;
  uint32_t bytes  = padded * width / 8u;
  std::vector<uint8_t> out(bytes, 0u);
  for (uint32_t i = 0; i < count; ++i) {
    uint64_t v    = static_cast<uint64_t>(values[i]);
    uint64_t mask = (width >= 64u) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1u);
    v &= mask;
    uint64_t bit_pos = uint64_t{i} * width;
    uint32_t byte0   = static_cast<uint32_t>(bit_pos >> 3);
    uint32_t bit_off = static_cast<uint32_t>(bit_pos & 7u);
    // Walk byte-by-byte writing each value's bit slice. The buffer is sized
    // to a 32-group multiple, which always provides tail headroom for the
    // last value's straddle bytes.
    int written = 0;
    while (written < static_cast<int>(width)) {
      int dst_byte   = static_cast<int>(byte0) + (static_cast<int>(bit_off) + written) / 8;
      int dst_bit    = (static_cast<int>(bit_off) + written) % 8;
      int take       = std::min(8 - dst_bit, static_cast<int>(width) - written);
      uint8_t byte_v = static_cast<uint8_t>((v >> written) & ((uint64_t{1} << take) - 1u));
      out[dst_byte] |= static_cast<uint8_t>(byte_v << dst_bit);
      written += take;
    }
  }
  return out;
}

//===----------------------------------------------------------------------===//
// ALP block builder. `vectors` is a list of per-vector inputs; each vector
// is either compressed (mantissas + exception list) or uncompressed (raw
// values). `single_for_value` defaults to 0 — callers can override to test
// non-trivial frame_of_reference paths.
//===----------------------------------------------------------------------===//

template <typename T>
struct alp_compressed_vector {
  uint8_t exponent;                           // ≤ AlpTypedConstants<T>::MAX_EXPONENT
  uint8_t factor;                             // ≤ 18
  uint64_t frame_of_ref;                      // unsigned wrap-around offset
  uint8_t bit_width;                          // ≤ sizeof(T)*8
  uint32_t row_count;                         // ≤ ALP_VECTOR_SIZE
  std::vector<uint64_t> mantissas;            // size == row_count, all fit in bit_width
  std::vector<T> exceptions;                  // T-typed real values
  std::vector<uint16_t> exception_positions;  // size == exceptions.size()
};

template <typename T>
struct alp_uncompressed_vector {
  uint32_t row_count;
  std::vector<T> values;  // size == row_count
};

template <typename T>
using alp_vector = std::variant<alp_compressed_vector<T>, alp_uncompressed_vector<T>>;

/// Append `bytes` to `seg` and return the offset of the first appended byte.
inline uint32_t append(std::vector<uint8_t>& seg, uint8_t const* bytes, size_t n)
{
  uint32_t off = static_cast<uint32_t>(seg.size());
  seg.insert(seg.end(), bytes, bytes + n);
  return off;
}

template <typename T>
inline std::vector<uint8_t> make_alp_block(std::vector<alp_vector<T>> const& vectors)
{
  static_assert(std::is_floating_point_v<T>, "ALP supports float / double only");
  std::vector<uint8_t> seg;
  seg.resize(4u, 0u);  // reserve metadata_end slot

  std::vector<uint32_t> per_vec_offsets;
  per_vec_offsets.reserve(vectors.size());

  for (auto const& vec : vectors) {
    uint32_t this_vec_off = static_cast<uint32_t>(seg.size());
    per_vec_offsets.push_back(this_vec_off);

    if (std::holds_alternative<alp_compressed_vector<T>>(vec)) {
      auto const& cv = std::get<alp_compressed_vector<T>>(vec);
      // 13-byte header.
      uint8_t hdr[13];
      hdr[0]             = cv.exponent;
      hdr[1]             = cv.factor;
      uint16_t exc_count = static_cast<uint16_t>(cv.exceptions.size());
      std::memcpy(hdr + 2, &exc_count, 2);
      std::memcpy(hdr + 4, &cv.frame_of_ref, 8);
      hdr[12] = cv.bit_width;
      seg.insert(seg.end(), hdr, hdr + 13);
      // Bitpacked mantissas.
      auto packed = pack_lsb<uint64_t>(cv.mantissas, cv.bit_width);
      seg.insert(seg.end(), packed.begin(), packed.end());
      // Exceptions (T-typed) followed by uint16 positions.
      uint8_t const* exc_bytes = reinterpret_cast<uint8_t const*>(cv.exceptions.data());
      seg.insert(seg.end(), exc_bytes, exc_bytes + cv.exceptions.size() * sizeof(T));
      uint8_t const* pos_bytes = reinterpret_cast<uint8_t const*>(cv.exception_positions.data());
      seg.insert(
        seg.end(), pos_bytes, pos_bytes + cv.exception_positions.size() * sizeof(uint16_t));
    } else {
      auto const& uv = std::get<alp_uncompressed_vector<T>>(vec);
      // Sentinel byte + raw values.
      uint8_t sentinel = ::sirius::cuda::scan::ALP_UNCOMPRESSED_SENTINEL;
      seg.push_back(sentinel);
      uint8_t const* vbytes = reinterpret_cast<uint8_t const*>(uv.values.data());
      seg.insert(seg.end(), vbytes, vbytes + uv.values.size() * sizeof(T));
    }
  }

  // Append the backwards pointer trailer (one uint32 per vector, written
  // in reverse so entry V lives at metadata_end - (V+1)*4).
  for (auto it = per_vec_offsets.rbegin(); it != per_vec_offsets.rend(); ++it) {
    uint32_t off = *it;
    uint8_t b[4];
    std::memcpy(b, &off, 4);
    seg.insert(seg.end(), b, b + 4);
  }

  uint32_t metadata_end = static_cast<uint32_t>(seg.size());
  std::memcpy(seg.data(), &metadata_end, 4);
  return seg;
}

//===----------------------------------------------------------------------===//
// ALPRD block builder.
//===----------------------------------------------------------------------===//

template <typename T>
struct alprd_compressed_vector {
  uint32_t row_count;
  std::vector<uint16_t> left_indices;           // size == row_count
  std::vector<uint64_t> right_parts;            // size == row_count
  std::vector<uint16_t> exception_left_values;  // size == exception_positions.size()
  std::vector<uint16_t> exception_positions;
};

template <typename T>
struct alprd_uncompressed_vector {
  uint32_t row_count;
  std::vector<T> values;
};

template <typename T>
using alprd_vector = std::variant<alprd_compressed_vector<T>, alprd_uncompressed_vector<T>>;

template <typename T>
inline std::vector<uint8_t> make_alprd_block(uint8_t right_bw,
                                             uint8_t left_bw,
                                             std::vector<uint16_t> const& dict,
                                             std::vector<alprd_vector<T>> const& vectors)
{
  static_assert(std::is_floating_point_v<T>, "ALPRD supports float / double only");
  if (dict.size() > ::sirius::cuda::scan::ALPRD_MAX_DICT_SIZE) {
    throw std::runtime_error("make_alprd_block: dict size > MAX_DICTIONARY_SIZE");
  }
  std::vector<uint8_t> seg;
  // 7-byte header + dict (uint16 each).
  seg.resize(::sirius::cuda::scan::ALPRD_HEADER_SIZE, 0u);
  seg[4]                    = right_bw;
  seg[5]                    = left_bw;
  seg[6]                    = static_cast<uint8_t>(dict.size());
  uint8_t const* dict_bytes = reinterpret_cast<uint8_t const*>(dict.data());
  seg.insert(seg.end(), dict_bytes, dict_bytes + dict.size() * sizeof(uint16_t));

  std::vector<uint32_t> per_vec_offsets;
  per_vec_offsets.reserve(vectors.size());

  for (auto const& vec : vectors) {
    uint32_t this_vec_off = static_cast<uint32_t>(seg.size());
    per_vec_offsets.push_back(this_vec_off);

    if (std::holds_alternative<alprd_compressed_vector<T>>(vec)) {
      auto const& cv     = std::get<alprd_compressed_vector<T>>(vec);
      uint16_t exc_count = static_cast<uint16_t>(cv.exception_positions.size());
      uint8_t exc_b[2];
      std::memcpy(exc_b, &exc_count, 2);
      seg.insert(seg.end(), exc_b, exc_b + 2);
      auto left_packed = pack_lsb<uint16_t>(cv.left_indices, left_bw);
      seg.insert(seg.end(), left_packed.begin(), left_packed.end());
      auto right_packed = pack_lsb<uint64_t>(cv.right_parts, right_bw);
      seg.insert(seg.end(), right_packed.begin(), right_packed.end());
      uint8_t const* exc_l = reinterpret_cast<uint8_t const*>(cv.exception_left_values.data());
      seg.insert(seg.end(), exc_l, exc_l + cv.exception_left_values.size() * 2u);
      uint8_t const* exc_p = reinterpret_cast<uint8_t const*>(cv.exception_positions.data());
      seg.insert(seg.end(), exc_p, exc_p + cv.exception_positions.size() * 2u);
    } else {
      auto const& uv    = std::get<alprd_uncompressed_vector<T>>(vec);
      uint16_t sentinel = ::sirius::cuda::scan::ALPRD_UNCOMPRESSED_SENTINEL;
      uint8_t b[2];
      std::memcpy(b, &sentinel, 2);
      seg.insert(seg.end(), b, b + 2);
      uint8_t const* vbytes = reinterpret_cast<uint8_t const*>(uv.values.data());
      seg.insert(seg.end(), vbytes, vbytes + uv.values.size() * sizeof(T));
    }
  }

  for (auto it = per_vec_offsets.rbegin(); it != per_vec_offsets.rend(); ++it) {
    uint32_t off = *it;
    uint8_t b[4];
    std::memcpy(b, &off, 4);
    seg.insert(seg.end(), b, b + 4);
  }

  uint32_t metadata_end = static_cast<uint32_t>(seg.size());
  std::memcpy(seg.data(), &metadata_end, 4);
  return seg;
}

/// Multi-vector ALP segment with `n_vecs` full vectors at `bit_width`. No
/// exceptions — phase-1 measures the bulk-decode hot path. Decoded value at
/// row `v*1024+i` is `(T)(int64_t)(((uint64_t)i ^ (uint64_t)v) & mask)`
/// since exponent=factor=frame_of_ref=0.
template <typename T>
inline std::vector<uint8_t> synth_alp_segment(uint32_t n_vecs, uint8_t bit_width)
{
  uint64_t mask = (bit_width >= 64u) ? ~uint64_t{0} : ((uint64_t{1} << bit_width) - 1u);
  std::vector<alp_vector<T>> vecs;
  vecs.reserve(n_vecs);
  for (uint32_t v = 0; v < n_vecs; ++v) {
    alp_compressed_vector<T> cv;
    cv.exponent     = 0;
    cv.factor       = 0;
    cv.frame_of_ref = 0;
    cv.bit_width    = bit_width;
    cv.row_count    = 1024u;
    cv.mantissas.resize(1024u);
    for (uint32_t i = 0; i < 1024u; ++i)
      cv.mantissas[i] = (uint64_t{i} ^ uint64_t{v}) & mask;
    vecs.emplace_back(std::move(cv));
  }
  return make_alp_block<T>(vecs);
}

/// Multi-vector ALPRD segment with `n_vecs` vectors at `right_bw`/`left_bw`,
/// using the supplied dictionary. No exceptions. Decoded bits at row
/// `v*1024+i` are `(uint64_t)dict[(i^v) & l_mask] << right_bw |
/// ((i * 0x9E3779B97F4A7C15ULL) & r_mask)` reinterpreted as `T`.
template <typename T>
inline std::vector<uint8_t> synth_alprd_segment(uint32_t n_vecs,
                                                uint8_t right_bw,
                                                uint8_t left_bw,
                                                std::vector<uint16_t> const& dict)
{
  uint64_t r_mask = (right_bw >= 64u) ? ~uint64_t{0} : ((uint64_t{1} << right_bw) - 1u);
  uint16_t l_mask =
    (left_bw >= 16u) ? uint16_t{0xFFFFu} : static_cast<uint16_t>((uint16_t{1} << left_bw) - 1u);
  std::vector<alprd_vector<T>> vecs;
  vecs.reserve(n_vecs);
  for (uint32_t v = 0; v < n_vecs; ++v) {
    alprd_compressed_vector<T> cv;
    cv.row_count = 1024u;
    cv.left_indices.resize(1024u);
    cv.right_parts.resize(1024u);
    for (uint32_t i = 0; i < 1024u; ++i) {
      cv.left_indices[i] = static_cast<uint16_t>((i ^ v) & l_mask);
      cv.right_parts[i]  = (uint64_t{i} * 0x9E3779B97F4A7C15ULL) & r_mask;
    }
    vecs.emplace_back(std::move(cv));
  }
  return make_alprd_block<T>(right_bw, left_bw, dict, vecs);
}

}  // namespace sirius::test::decode::alp
