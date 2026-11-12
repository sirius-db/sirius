// SPDX-License-Identifier: Apache-2.0
// OnPair-GPU string codec ("onpair" in the plan DSL): the FastPair decoder (arXiv
// 2609.15034, Algorithm 1) in simpatico's leaf-codec shape. Data comes from encode()
// (onpair_encoder.cu: dictionary trained on the host from a sample, the characters parsed
// on the GPU).
//
// Under str_split, the codec sees the flat chars buffer, and decompress()
// returns it as a column of the original fixed-width type. Row offsets are str_split's.
//
// Layout:
//   * the dictionary, pruned to the tokens the codes use and renumbered densely, so
//     the codes need ceil(log2(dict_size)) bits;
//   * the dictionary in fixed-stride planes for W, the token bytes the decoder's first
//     read returns: for W=8 the first 8 bytes of every token followed by the remaining 8
//     (split reads), for W=16 the full 16-byte entries (full reads, chosen when most
//     codes are longer than 8 bytes);
//   * token lengths minus one, two per byte;
//   * the codes bitpacked LSB-first into u32 words, padded to whole batches;
//   * the sidecar: the u32 output byte offset of every batch of kOnPairBatchCodes codes.

#pragma once

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {

/// FastPair decodes K = 6 codes per lane, so a warp decodes a batch of 32 * K codes.
inline constexpr int kOnPairK                    = 6;
inline constexpr std::uint32_t kOnPairBatchCodes = 32u * kOnPairK;

/// Byte ranges to decode: [src[i], src[i] + dst[i + 1] - dst[i]) goes to out + dst[i]. The
/// ranges ascend and do not overlap; dst holds n + 1 offsets.
struct onpair_byte_ranges {
  std::int64_t const* src;
  std::int32_t const* dst;
  std::uint64_t n;
};

struct onpair_dictionary;

/// GPU OnPair encoder settings. The characters are cut into fixed segments that GPU
/// threads parse independently, so tokens may span rows but never a segment.
///
/// Training runs on the host and dominates the encode, so trained dictionaries are kept in
/// a process-wide cache and reused for later data they still fit: before training, each
/// cached dictionary parses a strided probe of the new data on the GPU, and the first one
/// reaching reuse_min_fit of the bytes per code it reached on the data it was trained for
/// is taken. Every dictionary holds all 256 single bytes, so reuse never loses data; a
/// reused dictionary only compresses less than a fresh one would.
struct onpair_encoder_config {
  int bits                  = 12;    ///< Dictionary capacity 2^bits, 9..16.
  std::size_t segment_bytes = 1024;  ///< Largest run of characters one GPU thread parses.
  /// Segments are halved, down to 64 bytes, until there are this many per SM to keep the
  /// GPU busy (0 keeps segment_bytes). Shorter segments cut more tokens at their ends:
  /// a 17 MB flag column parses ~3x faster for ~5% ratio.
  std::size_t min_segments_per_sm = 1024;
  /// Characters copied to the host to train on. 16 MB trains ~3x slower (~290 ms on
  /// l_comment) for ~2% ratio; only an encode no cached dictionary fits trains.
  std::size_t sample_bytes       = 4u << 20;
  std::size_t sample_slice_bytes = 4096;  ///< Contiguous run per training sample.
  std::uint64_t seed             = 0x0f5eedull;
  bool reuse_dictionaries        = true;
  double reuse_min_fit           = 0.9;       ///< Probe bytes per code vs. at training, 0..1.
  std::size_t probe_bytes        = 1u << 20;  ///< Characters each reuse probe parses.
  /// Parse each string on its own ("onpair_strings" in the plan DSL): segments start on
  /// string starts and no token spans two strings, so equal strings get equal codes and a
  /// string's codes start where it does. Needs the strings' offsets.
  bool string_aligned = false;
  /// Encode with this dictionary as is: no probe, no training, and the cache is left alone.
  /// A dictionary holds every single byte, so any data encodes, only less well than with a
  /// dictionary trained on it.
  std::shared_ptr<onpair_dictionary const> dictionary;
};

/// Characters to train a dictionary on, copied to the host.
struct onpair_training_sample {
  std::vector<std::uint8_t> bytes;
  std::vector<std::uint32_t> offsets;  ///< Run boundaries, 0 to bytes.size().
};

/// Copies cfg.sample_bytes of the @p n_chars device bytes at @p chars to the host, in runs
/// of cfg.sample_slice_bytes evenly strided over them. Synchronizes @p stream.
onpair_training_sample sample_onpair_training_data(std::uint8_t const* chars,
                                                   std::size_t n_chars,
                                                   onpair_encoder_config const& cfg,
                                                   ::cuda::stream_ref stream);

/// Trains a dictionary of up to 2^cfg.bits tokens on @p sample. Host only: needs no GPU,
/// so it can run off the thread that took the sample.
std::shared_ptr<onpair_dictionary const> train_onpair_dictionary(
  onpair_training_sample const& sample, onpair_encoder_config const& cfg);

/// Host bytes @p dict's parse tables take; an encode uploads them to the device.
std::size_t onpair_dictionary_bytes(onpair_dictionary const& dict);

struct onpair_compressed_representation : standalone_compressed_representation {
  std::size_t dict_size          = 0;  ///< Tokens after pruning.
  std::size_t n_tokens           = 0;
  std::size_t uncompressed_bytes = 0;
  std::uint32_t code_bits        = 0;
  int w                          = 8;      ///< 8 (split reads) or 16 (full reads).
  bool string_aligned            = false;  ///< Encoded with onpair_encoder_config::string_aligned.

  rmm::device_buffer dict_buf;      // dict_size * 16, laid out for w
  rmm::device_buffer lens_nib_buf;  // ceil(dict_size / 2)
  rmm::device_buffer codes_buf;     // packed u32 words
  rmm::device_buffer sidecar_buf;   // u32 per batch, plus the total

  onpair_compressed_representation(cudf::data_type type, cudf::size_type n_rows);

  /// Decodes the codes to a column of the original type and row count.
  [[nodiscard]] std::unique_ptr<cudf::column> decompress(decode_frame& frame) const override;

  /// The stored channels: codes, sidecar, dict and lens.
  std::vector<compressible_output> named_channels(::cuda::stream_ref) const override;
  leaf_meta_v describe_meta() const override;

  /// Rebuilds a representation from its stored channels (in named_channels() order).
  static std::unique_ptr<onpair_compressed_representation> from_channels(
    leaf_meta::onpair const& meta,
    std::unique_ptr<cudf::column> codes,
    std::unique_ptr<cudf::column> sidecar,
    std::unique_ptr<cudf::column> dict,
    std::unique_ptr<cudf::column> lens);

  /// Decodes only the byte ranges [src[i], src[i] + dst[i + 1] - dst[i]) of the
  /// uncompressed bytes, to out + dst[i]: the n ranges ascend and do not overlap, and only
  /// the batches they reach into are decoded.
  void copy_ranges(std::int64_t const* src,
                   std::int32_t const* dst,
                   std::size_t n,
                   std::uint8_t* out,
                   ::cuda::stream_ref stream) const;

  /// BOOL8 per row: whether row r, the bytes [offsets[r], offsets[r + 1]) of INT32 or INT64
  /// @p offsets, equals one of @p literals. Compares codes and decodes nothing, so needs a
  /// string_aligned encoding.
  std::unique_ptr<cudf::column> equal_any(cudf::column_view offsets,
                                          std::vector<std::string> const& literals,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr) const;

  /// The dictionary's tokens, host side, indexed by code.
  std::vector<std::string> tokens(::cuda::stream_ref stream) const;

  struct token_index;  ///< tokens() and their codes, built on first use.

  OpId kind() const override { return OpId::OnPair; }

  /// Encodes the @p n_chars bytes at @p chars in device memory, a column of @p n_rows
  /// values of fixed-width @p type. With cfg.string_aligned, @p string_offsets are the
  /// strings' INT32 or INT64 offsets into the characters, from 0 to n_chars.
  static std::unique_ptr<onpair_compressed_representation> encode(
    std::uint8_t const* chars,
    std::size_t n_chars,
    cudf::data_type type,
    cudf::size_type n_rows,
    onpair_encoder_config const& cfg,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr,
    cudf::column_view string_offsets = {});

 private:
  std::shared_ptr<token_index> token_index_;

  void launch(std::uint8_t* out, cudaStream_t stream, onpair_byte_ranges const* ranges) const;
};

/// Compresses the bytes of a fixed-width column, typically str_split's chars channel.
struct onpair_compressor : compressor {
  onpair_encoder_config config;
  /// The strings' offsets into the column, for config.string_aligned; the plan walk sets
  /// them from str_split's offsets channel.
  cudf::column_view string_offsets;

  std::unique_ptr<compressed_representation> compress(cudf::column_view col,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr) override
  {
    if (!cudf::is_fixed_width(col.type())) {
      throw std::invalid_argument("onpair: compress expects a fixed-width column");
    }
    auto const width = cudf::size_of(col.type());
    return onpair_compressed_representation::encode(
      col.head<std::uint8_t>() + static_cast<std::size_t>(col.offset()) * width,
      static_cast<std::size_t>(col.size()) * width,
      col.type(),
      col.size(),
      config,
      stream,
      mr,
      string_offsets);
  }
};

}  // namespace simpatico
