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

// Synthetic on-disk segment builders for the strings-decode tests.
// FSST / DICT_FSST go through libduckdb's host FSST encoder so the segment
// bytes match DuckDB's on-disk format byte-for-byte.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// libduckdb FSST encoder API (signatures at duckdb/third_party/fsst/fsst.h).
extern "C" {
typedef void* duckdb_fsst_encoder_t;
duckdb_fsst_encoder_t* duckdb_fsst_create(size_t n,
                                          size_t len_in[],
                                          unsigned char* string_in[],
                                          int zeroTerminated);
void duckdb_fsst_destroy(duckdb_fsst_encoder_t*);
unsigned int duckdb_fsst_export(duckdb_fsst_encoder_t* encoder, unsigned char* buf);
size_t duckdb_fsst_compress(duckdb_fsst_encoder_t* encoder,
                            size_t nstrings,
                            size_t len_in[],
                            unsigned char* string_in[],
                            size_t outsize,
                            unsigned char* output,
                            size_t len_out[],
                            unsigned char* string_out[]);
}

namespace sirius::test::decode::strings {

/// Bits needed to represent [0, count).
inline uint32_t bitpack_width_for_count(uint32_t count)
{
  if (count <= 1) return 0;
  uint32_t w = 0;
  uint32_t v = count - 1;
  while (v > 0) {
    ++w;
    v >>= 1;
  }
  return w;
}

/// Inverse of unpack_value in gpu_decode_strings.cu (LSB-first into uint32 stream).
inline std::vector<uint32_t> pack_uint32(std::vector<uint32_t> const& values, uint32_t width)
{
  if (width == 0 || values.empty()) {
    return std::vector<uint32_t>(1, 0);  // one guard word for unpack_value
  }
  size_t total_bits = size_t{values.size()} * width;
  size_t words      = (total_bits + 31) / 32;
  std::vector<uint32_t> out(words + 1, 0);  // +1 guard word
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t pos      = uint64_t{i} * width;
    uint32_t word_idx = static_cast<uint32_t>(pos / 32);
    uint32_t bit_off  = static_cast<uint32_t>(pos & 31);
    uint64_t v        = static_cast<uint64_t>(values[i]) &
                 ((width == 64) ? ~uint64_t{0} : (uint64_t{1} << width) - 1);
    out[word_idx] |= static_cast<uint32_t>(v << bit_off);
    if (bit_off + width > 32) { out[word_idx + 1] |= static_cast<uint32_t>(v >> (32 - bit_off)); }
  }
  return out;
}

/// `dict_strings[0]` reserved for NULL; `selections[i] == 0` means NULL.
inline std::vector<uint8_t> make_dict_segment(std::vector<std::string> const& dict_strings,
                                              std::vector<uint32_t> const& selections)
{
  uint32_t const dict_count = static_cast<uint32_t>(dict_strings.size());

  std::vector<uint32_t> idx_buf(dict_count, 0);
  uint32_t cum = 0;
  for (uint32_t k = 0; k < dict_count; ++k) {
    cum += static_cast<uint32_t>(dict_strings[k].size());
    idx_buf[k] = cum;
  }
  uint32_t const dict_size = cum;
  uint32_t const width     = bitpack_width_for_count(dict_count);
  auto sel_packed          = pack_uint32(selections, width);

  uint32_t const header_size  = 20;
  uint32_t const sel_buf_off  = header_size;
  uint32_t const sel_buf_size = static_cast<uint32_t>(sel_packed.size() * sizeof(uint32_t));
  uint32_t const idx_buf_off  = sel_buf_off + sel_buf_size;
  uint32_t const idx_buf_size = dict_count * sizeof(uint32_t);
  uint32_t const dict_off     = idx_buf_off + idx_buf_size;
  uint32_t const dict_end     = dict_off + dict_size;
  uint32_t const total        = dict_end;

  std::vector<uint8_t> bytes(total, 0);
  uint32_t hdr[5] = {dict_size, dict_end, idx_buf_off, dict_count, width};
  std::memcpy(bytes.data(), hdr, sizeof(hdr));
  if (sel_buf_size > 0) {
    std::memcpy(bytes.data() + sel_buf_off, sel_packed.data(), sel_buf_size);
  }
  std::memcpy(bytes.data() + idx_buf_off, idx_buf.data(), idx_buf_size);
  // Dict bytes packed in REVERSE: entry K starts at dict_end - idx_buf[K].
  uint8_t* dict_region = bytes.data() + dict_off;
  for (uint32_t k = 0; k < dict_count; ++k) {
    uint32_t offset_from_dict_end = idx_buf[k];
    uint32_t entry_size           = static_cast<uint32_t>(dict_strings[k].size());
    uint8_t* dst                  = dict_region + dict_size - offset_from_dict_end;
    if (entry_size > 0) std::memcpy(dst, dict_strings[k].data(), entry_size);
  }
  return bytes;
}

// dbgen-style grammar — produces TPC-H-shaped comments (5-150B, high common-
// word repetition) so FSST microbenches see realistic symbol-table coverage.

namespace tpch_grammar {
inline char const* nouns[] = {
  "deposits",  "packages",   "accounts",    "ideas",     "instructions",   "requests",
  "platelets", "asymptotes", "theodolites", "courts",    "hockey players", "pinto beans",
  "excuses",   "frets",      "frays",       "warhorses", "dependencies",   "attainments",
  "warthogs",  "pearls",     "dolphins",    "epitaphs",  "patterns",       "foxes",
  "packages",  "deposits",   "accounts",    "requests",  "ideas",
};
inline char const* verbs[] = {
  "haggle",  "boost",  "are",  "sleep",  "wake",  "cajole", "thrash",
  "promise", "engage", "hang", "hinder", "x-ray", "use",    "integrate",
  "snooze",  "doze",   "nag",  "doubt",  "have",  "kindle", "play",
};
inline char const* adjectives[] = {
  "regular", "special", "ironic",    "even",     "bold",      "final",    "express",
  "thin",    "silent",  "pending",   "blithe",   "permanent", "unusual",  "furious",
  "quiet",   "fluffy",  "busy",      "stealthy", "careful",   "ruthless", "sly",
  "daring",  "brave",   "carefully", "slyly",    "boldly",    "blithely",
};
inline char const* adverbs[] = {
  "carefully",
  "slyly",
  "boldly",
  "quickly",
  "permanently",
  "fluffily",
  "ruthlessly",
  "regularly",
  "ironically",
  "blithely",
  "furiously",
  "even",
  "quietly",
  "always",
  "never",
  "stealthily",
  "finally",
  "silently",
};
inline char const* prepositions[] = {
  "above the",       "according to the", "across the", "after the",   "against the",
  "along the",       "alongside of the", "around the", "before the",  "between the",
  "beyond the",      "by the",           "during the", "for the",     "from the",
  "in place of the", "instead of the",   "on the",     "outside the", "to the",
  "toward the",      "without the",
};
inline char const* auxiliaries[] = {
  "do",
  "may",
  "might",
  "shall",
  "will",
  "would",
  "should",
  "can",
  "could",
  "must",
  "ought to",
  "have to",
};
}  // namespace tpch_grammar

/// Pick from a const char* array using a simple LCG seeded by `state`.
/// Updates `state` in-place.
template <size_t N>
inline char const* pick(char const* (&arr)[N], uint64_t& state)
{
  state = state * 6364136223846793005ull + 1442695040888963407ull;
  return arr[(state >> 33) % N];
}

/// Build one TPC-H-style sentence and append to `out`. Returns appended length.
inline size_t append_tpch_sentence(std::string& out, uint64_t& seed)
{
  using namespace tpch_grammar;
  size_t before = out.size();
  // Form: "<adj> <noun> <verb> <prep> <adj> <noun>"
  // e.g. "carefully ironic deposits boost regular accounts"
  out.append(pick(adjectives, seed));
  out.push_back(' ');
  out.append(pick(nouns, seed));
  out.push_back(' ');
  out.append(pick(verbs, seed));
  out.push_back(' ');
  out.append(pick(prepositions, seed));
  out.push_back(' ');
  out.append(pick(adjectives, seed));
  out.push_back(' ');
  out.append(pick(nouns, seed));
  return out.size() - before;
}

/// Build a TPC-H-like comment of length in [min_len, max_len], deterministic
/// from `seed`. Uses the simple grammar above; concatenates sentences until
/// the length budget is reached. Truncates if the next sentence would
/// overshoot. Lengths follow the TPC-H l_comment distribution shape: 5-150
/// bytes, mean ~30, higher density at the long end.
inline std::string make_tpch_like_comment(uint64_t seed,
                                          uint32_t min_len = 5,
                                          uint32_t max_len = 150)
{
  std::string out;
  out.reserve(max_len);
  uint32_t target =
    min_len + static_cast<uint32_t>((seed * 2654435769ull) % (max_len - min_len + 1));
  uint64_t s = seed ^ 0x9E3779B97F4A7C15ull;
  while (out.size() < target) {
    if (!out.empty()) {
      out.append((s % 3 == 0) ? ". " : " ");
      s = s * 1103515245ull + 12345ull;
    }
    append_tpch_sentence(out, s);
    if (out.size() >= target) {
      out.resize(target);
      break;
    }
  }
  return out;
}

/// Round n up to the nearest multiple of 8 (matches DuckDB AlignValue<idx_t>).
inline uint32_t synth_align_up8(uint32_t n) { return (n + 7u) & ~7u; }

/// Bit-width needed to represent `max_value` losslessly. Returns 0 for 0.
inline uint32_t bitpack_width_for_value(uint32_t max_value)
{
  if (max_value == 0) return 0;
  uint32_t w = 0;
  while (max_value > 0) {
    ++w;
    max_value >>= 1;
  }
  return w;
}

/// Layout matches src/storage/compression/fsst.cpp. Rows are stored
/// REVERSE-order in the compressed-bytes region.
inline std::vector<uint8_t> make_fsst_segment(std::vector<std::string> const& strings)
{
  uint32_t const row_count = static_cast<uint32_t>(strings.size());
  if (row_count == 0) return std::vector<uint8_t>(16, 0);

  std::vector<size_t> lens(row_count);
  std::vector<unsigned char*> ptrs(row_count);
  for (uint32_t i = 0; i < row_count; ++i) {
    lens[i] = strings[i].size();
    ptrs[i] = reinterpret_cast<unsigned char*>(const_cast<char*>(strings[i].data()));
  }
  duckdb_fsst_encoder_t* enc = duckdb_fsst_create(row_count, lens.data(), ptrs.data(), 0);
  if (!enc) throw std::runtime_error("make_fsst_segment: duckdb_fsst_create returned null");

  size_t total_in = 0;
  for (size_t l : lens)
    total_in += l;
  size_t comp_buf_size = 7 + 2 * total_in + 16 * row_count;  // conservative bound
  std::vector<unsigned char> comp_buf(comp_buf_size);
  std::vector<size_t> lenOut(row_count);
  std::vector<unsigned char*> strOut(row_count);
  size_t n_compressed = duckdb_fsst_compress(enc,
                                             row_count,
                                             lens.data(),
                                             ptrs.data(),
                                             comp_buf_size,
                                             comp_buf.data(),
                                             lenOut.data(),
                                             strOut.data());
  if (n_compressed != row_count) {
    duckdb_fsst_destroy(enc);
    throw std::runtime_error("make_fsst_segment: duckdb_fsst_compress fit fewer rows than asked");
  }

  std::vector<unsigned char> symtab_buf(8 + 1 + 8 + 2048 + 1);  // FSST_MAXHEADER
  unsigned int symtab_size = duckdb_fsst_export(enc, symtab_buf.data());
  duckdb_fsst_destroy(enc);

  uint32_t max_comp_len = 0;
  for (size_t l : lenOut)
    max_comp_len = std::max(max_comp_len, static_cast<uint32_t>(l));
  uint32_t bitpacking_width = std::max(1u, bitpack_width_for_value(max_comp_len));

  uint32_t const header_size = 16;
  uint32_t const lengths_off = header_size;
  size_t lengths_total_bits  = size_t{row_count} * bitpacking_width;
  uint32_t lengths_bytes_raw = static_cast<uint32_t>((lengths_total_bits + 31u) / 32u * 4u);
  uint32_t lengths_padded    = lengths_bytes_raw + 4u;  // +1 word: unpack_value reads 8B
  uint32_t symtab_off        = synth_align_up8(lengths_off + lengths_padded);
  uint32_t comp_bytes_off    = synth_align_up8(symtab_off + symtab_size);

  size_t total_comp_bytes = 0;
  for (size_t l : lenOut)
    total_comp_bytes += l;
  uint32_t dict_end = comp_bytes_off + static_cast<uint32_t>(total_comp_bytes);

  std::vector<uint8_t> bytes(dict_end, 0);
  uint32_t hdr[4] = {
    static_cast<uint32_t>(total_comp_bytes), dict_end, bitpacking_width, symtab_off};
  std::memcpy(bytes.data(), hdr, sizeof(hdr));
  std::vector<uint32_t> lengths_u32(row_count);
  for (uint32_t i = 0; i < row_count; ++i)
    lengths_u32[i] = static_cast<uint32_t>(lenOut[i]);
  auto packed_lengths = pack_uint32(lengths_u32, bitpacking_width);
  std::memcpy(bytes.data() + lengths_off,
              packed_lengths.data(),
              std::min(packed_lengths.size() * sizeof(uint32_t), size_t{lengths_padded}));
  std::memcpy(bytes.data() + symtab_off, symtab_buf.data(), symtab_size);
  uint32_t cum_from_end = 0;
  for (uint32_t i = 0; i < row_count; ++i) {  // rows stored in reverse
    cum_from_end += static_cast<uint32_t>(lenOut[i]);
    uint8_t* dst = bytes.data() + dict_end - cum_from_end;
    if (lenOut[i] > 0) std::memcpy(dst, strOut[i], lenOut[i]);
  }
  return bytes;
}

/// `mode`: 0=DICTIONARY, 1=DICT_FSST, 2=FSST_ONLY (row i -> dict idx i+1).
/// `unique_strings[0]` reserved for NULL; `selections` ignored for mode 2.
/// Layout matches dict_fsst/compression.cpp.
inline std::vector<uint8_t> make_dict_fsst_segment(std::vector<std::string> const& unique_strings,
                                                   std::vector<uint32_t> const& selections,
                                                   uint8_t mode)
{
  uint32_t const dict_count = static_cast<uint32_t>(unique_strings.size());
  uint32_t const row_count  = (mode == 2) ? static_cast<uint32_t>(unique_strings.size() - 1)
                                          : static_cast<uint32_t>(selections.size());

  // FSST modes only — encoder skips idx 0 (the reserved NULL slot).
  std::vector<unsigned char> symtab_buf(8 + 1 + 8 + 2048 + 1);
  unsigned int symtab_size = 0;
  std::vector<size_t> entry_lens(dict_count, 0);
  std::vector<std::vector<uint8_t>> entry_bytes(dict_count);

  if (mode != 0 && dict_count > 1) {
    std::vector<size_t> lens;
    std::vector<unsigned char*> ptrs;
    for (uint32_t k = 1; k < dict_count; ++k) {
      lens.push_back(unique_strings[k].size());
      ptrs.push_back(reinterpret_cast<unsigned char*>(const_cast<char*>(unique_strings[k].data())));
    }
    duckdb_fsst_encoder_t* enc = duckdb_fsst_create(lens.size(), lens.data(), ptrs.data(), 0);
    if (!enc) throw std::runtime_error("make_dict_fsst_segment: fsst_create returned null");

    size_t total_in = 0;
    for (size_t l : lens)
      total_in += l;
    size_t comp_buf_size = 7 + 2 * total_in + 16 * lens.size();
    std::vector<unsigned char> comp_buf(comp_buf_size);
    std::vector<size_t> lenOut(lens.size());
    std::vector<unsigned char*> strOut(lens.size());
    size_t n_compressed = duckdb_fsst_compress(enc,
                                               lens.size(),
                                               lens.data(),
                                               ptrs.data(),
                                               comp_buf_size,
                                               comp_buf.data(),
                                               lenOut.data(),
                                               strOut.data());
    if (n_compressed != lens.size()) {
      duckdb_fsst_destroy(enc);
      throw std::runtime_error(
        "make_dict_fsst_segment: fsst_compress fit fewer entries than asked");
    }
    symtab_size = duckdb_fsst_export(enc, symtab_buf.data());
    duckdb_fsst_destroy(enc);

    for (uint32_t k = 1; k < dict_count; ++k) {
      size_t i       = static_cast<size_t>(k - 1);
      entry_lens[k]  = lenOut[i];
      entry_bytes[k] = std::vector<uint8_t>(strOut[i], strOut[i] + lenOut[i]);
    }
  } else {
    for (uint32_t k = 0; k < dict_count; ++k) {  // mode 0: raw bytes
      entry_lens[k]  = unique_strings[k].size();
      entry_bytes[k] = std::vector<uint8_t>(unique_strings[k].begin(), unique_strings[k].end());
    }
  }

  uint32_t max_entry_len = 0;
  for (uint32_t k = 0; k < dict_count; ++k) {
    max_entry_len = std::max(max_entry_len, static_cast<uint32_t>(entry_lens[k]));
  }
  uint32_t string_lengths_width = std::max(1u, bitpack_width_for_value(max_entry_len));
  uint32_t dict_indices_width   = std::max(1u, bitpack_width_for_value(dict_count - 1));

  uint32_t dict_size = 0;
  for (size_t l : entry_lens)
    dict_size += static_cast<uint32_t>(l);

  // Region offsets mirror prepare_dict_fsst in gpu_decode_strings.cu.
  uint32_t const header_size = 16;
  uint32_t off_dict          = synth_align_up8(header_size);
  uint32_t off_symtab        = (mode == 0) ? 0 : synth_align_up8(off_dict + dict_size);
  uint32_t off_slens =
    (mode == 0) ? off_dict + synth_align_up8(dict_size) : synth_align_up8(off_symtab + symtab_size);
  uint32_t slens_bits = dict_count * string_lengths_width;
  uint32_t off_didx   = synth_align_up8(off_slens + (slens_bits + 7u) / 8u);
  uint32_t total =
    (mode == 2) ? off_didx : synth_align_up8(off_didx + (row_count * dict_indices_width + 7u) / 8u);

  std::vector<uint8_t> bytes(total, 0);
  // Header (matches dict_fsst_header_t in gpu_decode_strings.cu).
  std::memcpy(bytes.data(), &dict_size, 4);
  std::memcpy(bytes.data() + 4, &dict_count, 4);
  bytes[8]  = mode;
  bytes[9]  = static_cast<uint8_t>(string_lengths_width);
  bytes[10] = static_cast<uint8_t>(dict_indices_width);
  bytes[11] = 0;
  std::memcpy(bytes.data() + 12, &symtab_size, 4);

  // Dict region — entries laid forward (entry 0 first), tightly packed.
  uint32_t cursor = off_dict;
  for (uint32_t k = 0; k < dict_count; ++k) {
    if (entry_lens[k] > 0) {
      std::memcpy(bytes.data() + cursor, entry_bytes[k].data(), entry_lens[k]);
    }
    cursor += static_cast<uint32_t>(entry_lens[k]);
  }
  if (mode != 0) { std::memcpy(bytes.data() + off_symtab, symtab_buf.data(), symtab_size); }
  std::vector<uint32_t> lengths_u32(dict_count);
  for (uint32_t k = 0; k < dict_count; ++k)
    lengths_u32[k] = static_cast<uint32_t>(entry_lens[k]);
  auto packed_lengths = pack_uint32(lengths_u32, string_lengths_width);
  std::memcpy(bytes.data() + off_slens,
              packed_lengths.data(),
              std::min(packed_lengths.size() * sizeof(uint32_t), size_t{(slens_bits + 7u) / 8u}));
  if (mode != 2) {
    auto packed_didx = pack_uint32(selections, dict_indices_width);
    size_t didx_bytes_to_copy =
      std::min(packed_didx.size() * sizeof(uint32_t), size_t{total - off_didx});
    std::memcpy(bytes.data() + off_didx, packed_didx.data(), didx_bytes_to_copy);
  }
  return bytes;
}

}  // namespace sirius::test::decode::strings
