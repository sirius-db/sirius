// SPDX-License-Identifier: Apache-2.0
//
// Bitpack chunk decoding: the shared unpack_bitpack_chunks primitive and the decoder of a
// `bitpack` root plan.
//
// Layout of one bitpack node (see tests/test_bitpack_layout_contract.cpp):
//   chunk_min   one element (the leaf's type) per chunk
//   chunk_count int32 per chunk: values in the chunk
//   chunk_bits  uint8 per chunk: bits per packed residual
//   packed      dense uint32 words; chunk c starts at word sum_{i<c} n_words(i), where
//               n_words = count == 0 ? 1 : ceil(count * bits / 32) (NwordsFromChunk in
//               bridge/offsets_cumsum.cu), and holds residual i at bit i * bits, LSB first.
//               value = chunk_min + residual, in the element's unsigned arithmetic.
//
// The whole-table decode gathers 3 (or 5) words unconditionally and relies on guard words after
// the buffer. A span decode stages only the span's words, so the kernels here never load a word
// past the chunk's own.

#include "decode/chunk/chunk_decoders.hpp"

#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico::chunk {

namespace {

/// Per-chunk decode parameters of a whole node, built from chunk_count / chunk_bits and uploaded
/// once per state.
struct chunk_meta {
  std::int64_t word_begin;  ///< first packed word of the chunk, from the start of `packed`
  std::int32_t n_words;     ///< packed words of the chunk
  std::int32_t count;       ///< values in the chunk
  std::int32_t bits;        ///< bits per residual
  std::int32_t pad;
};

/// Little-endian element at `bytes`, which may sit at any address when !aligned (a leaf buffer
/// can start at any payload offset).
template <typename U>
__device__ __forceinline__ U load_element(const std::uint8_t* __restrict__ bytes, bool aligned)
{
  if (aligned) { return *reinterpret_cast<const U*>(bytes); }
  std::uint64_t v = 0;
  for (std::size_t k = 0; k < sizeof(U); ++k) {
    v |= static_cast<std::uint64_t>(bytes[k]) << (8 * k);
  }
  return static_cast<U>(v);
}

constexpr int kThreadsPerBlock = 256;

/// Residual `idx` of a chunk with 1 <= bits <= 64; loads only words [0, n_words).
__device__ __forceinline__ std::uint64_t unpack_one(const std::uint32_t* __restrict__ words,
                                                    std::int32_t n_words,
                                                    std::int32_t bits,
                                                    std::int32_t idx)
{
  const std::uint32_t bp     = static_cast<std::uint32_t>(idx) * static_cast<std::uint32_t>(bits);
  const std::int32_t word_in = static_cast<std::int32_t>(bp >> 5);
  const std::int32_t bit_in  = static_cast<std::int32_t>(bp & 31);
  const std::int32_t shift_l = 32 - bit_in;
  const std::uint64_t w0     = words[word_in];
  const std::uint64_t w1     = word_in + 1 < n_words ? words[word_in + 1] : 0;
  const std::uint64_t w2     = word_in + 2 < n_words ? words[word_in + 2] : 0;
  const std::uint64_t lo     = ((w0 >> bit_in) | (w1 << shift_l)) & 0xFFFFFFFFULL;
  const std::uint64_t hi     = ((w1 >> bit_in) | (w2 << shift_l)) & 0xFFFFFFFFULL;
  const std::uint64_t mask   = bits == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1;
  return ((hi << 32) | lo) & mask;
}

/// Residual `idx` of a chunk with 1 <= bits <= 128; loads only words [0, n_words).
__device__ __forceinline__ unsigned __int128 unpack_one_128(const std::uint32_t* __restrict__ words,
                                                            std::int32_t n_words,
                                                            std::int32_t bits,
                                                            std::int32_t idx)
{
  using u128                 = unsigned __int128;
  const std::uint32_t bp     = static_cast<std::uint32_t>(idx) * static_cast<std::uint32_t>(bits);
  const std::int32_t word_in = static_cast<std::int32_t>(bp >> 5);
  const std::int32_t bit_in  = static_cast<std::int32_t>(bp & 31);
  u128 v                     = 0;
  for (std::int32_t k = 0; k < 4; ++k) {
    if (word_in + k < n_words) { v |= static_cast<u128>(words[word_in + k]) << (32 * k); }
  }
  v >>= bit_in;
  if (bit_in != 0 && word_in + 4 < n_words) {
    v |= static_cast<u128>(words[word_in + 4]) << (128 - bit_in);
  }
  const u128 mask = bits == 128 ? ~static_cast<u128>(0) : (static_cast<u128>(1) << bits) - 1;
  return v & mask;
}

/// The staged device inputs of chunks [chunk_begin, chunk_end) of one node.
struct staged_chunks {
  const chunk_meta* meta{nullptr};      ///< entry of chunk_begin
  const std::uint8_t* mins{nullptr};    ///< chunk_min element of chunk_begin
  const std::uint32_t* words{nullptr};  ///< packed words from chunk_begin's first word
  std::int64_t word_base{0};            ///< chunk_begin's first word, from the start of `packed`
  bool mins_aligned{false};             ///< mins can be loaded as whole elements
};

/// out[i] = value at position first_pos + i of the staged chunks (chunk = position / 1024),
/// for i in [0, n). Src is the element's type (its width wraps the sum and its signedness
/// extends it); positions past a chunk's count are left unwritten.
template <typename Src, typename Out>
__global__ void unpack_kernel(staged_chunks in,
                              std::int64_t first_pos,
                              std::int64_t n,
                              Out* __restrict__ out)
{
  using U              = typename ::cuda::std::make_unsigned<Src>::type;
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const std::int64_t pos   = first_pos + i;
  const std::int64_t chunk = pos / kChunkRows;
  const chunk_meta m       = in.meta[chunk];
  const auto idx           = static_cast<std::int32_t>(pos % kChunkRows);
  if (idx >= m.count) { return; }
  std::uint64_t v = load_element<U>(in.mins + chunk * sizeof(U), in.mins_aligned);
  if (m.bits != 0) {
    v += unpack_one(in.words + (m.word_begin - in.word_base), m.n_words, m.bits, idx);
  }
  out[i] = static_cast<Out>(static_cast<Src>(v));
}

/// The 16-byte counterpart; out holds two uint64 (low, high) per value.
__global__ void unpack_kernel_128(staged_chunks in,
                                  std::int64_t first_pos,
                                  std::int64_t n,
                                  std::uint64_t* __restrict__ out)
{
  using u128           = unsigned __int128;
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const std::int64_t pos   = first_pos + i;
  const std::int64_t chunk = pos / kChunkRows;
  const chunk_meta m       = in.meta[chunk];
  const auto idx           = static_cast<std::int32_t>(pos % kChunkRows);
  if (idx >= m.count) { return; }
  const std::uint8_t* min = in.mins + chunk * 16;
  u128 v = (static_cast<u128>(load_element<std::uint64_t>(min + 8, in.mins_aligned)) << 64) |
           load_element<std::uint64_t>(min, in.mins_aligned);
  if (m.bits != 0) {
    v += unpack_one_128(in.words + (m.word_begin - in.word_base), m.n_words, m.bits, idx);
  }
  out[2 * i]     = static_cast<std::uint64_t>(v);
  out[2 * i + 1] = static_cast<std::uint64_t>(v >> 64);
}

void cuda_try(cudaError_t status, char const* what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"bitpack chunk decode: "} + what +
                             " failed: " + cudaGetErrorString(status));
  }
}

template <typename Src, typename Out>
void launch_unpack(staged_chunks const& in,
                   std::int64_t first_pos,
                   std::int64_t n,
                   Out* out,
                   rmm::cuda_stream_view stream)
{
  auto const blocks = static_cast<unsigned>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
  unpack_kernel<Src, Out><<<blocks, kThreadsPerBlock, 0, stream.value()>>>(in, first_pos, n, out);
  cuda_try(cudaGetLastError(), "unpack kernel launch");
}

hpln_buffer_range const& range_of(hpln_column_description const& column, buffer_ref ref)
{
  if (ref.leaf >= column.leaves.size() || ref.buffer >= column.leaves[ref.leaf].buffers.size()) {
    throw std::runtime_error("bitpack chunk decode: the locator names a missing buffer");
  }
  return column.leaves[ref.leaf].buffers[ref.buffer];
}

}  // namespace

struct bitpack_state {
  explicit bitpack_state(bitpack_locator const& locator) : loc(locator) {}

  bitpack_locator loc;
  // The node's chunk metadata, read from chunk_count / chunk_bits once (the first decode) and
  // kept for the state's lifetime: 5 bytes per 1024 rows on the wire. word_prefix[c] is chunk
  // c's first word in `packed` and has one more entry than counts. The host needs the prefix to
  // know which packed bytes a span covers; nothing else is read back, so a decode after the
  // first never waits for the stream.
  bool loaded{false};
  std::size_t elem_size{0};
  std::vector<std::int32_t> counts;
  std::vector<std::int64_t> word_prefix{0};
  std::vector<chunk_meta> host_meta;  ///< source of the one upload; alive until the state dies
  rmm::device_buffer device_meta;
  rmm::device_buffer aligned_words;  ///< grow-only scratch for an unaligned `packed`

  [[nodiscard]] std::size_t device_bytes() const noexcept
  {
    return device_meta.size() + aligned_words.size();
  }

  /// Chunks of the node and bytes per chunk_min element.
  struct shape {
    std::int64_t num_chunks{0};
    std::size_t elem_size{0};
  };

  [[nodiscard]] shape describe(hpln_column_description const& column) const
  {
    auto const& min_range   = range_of(column, loc.chunk_min);
    auto const& count_range = range_of(column, loc.chunk_count);
    auto const& bits_range  = range_of(column, loc.chunk_bits);
    shape s;
    s.num_chunks = static_cast<std::int64_t>(bits_range.size);
    if (count_range.size != bits_range.size * sizeof(std::int32_t)) {
      throw std::runtime_error("bitpack chunk decode: chunk_count and chunk_bits disagree");
    }
    if (s.num_chunks == 0) { return s; }
    s.elem_size = min_range.size / bits_range.size;
    if (min_range.size % bits_range.size != 0 ||
        (s.elem_size != 1 && s.elem_size != 2 && s.elem_size != 4 && s.elem_size != 8 &&
         s.elem_size != 16)) {
      throw std::runtime_error("bitpack chunk decode: chunk_min is not one element per chunk");
    }
    return s;
  }

  /// Reads the whole node's chunk_count / chunk_bits, builds the word prefix and uploads the
  /// per-chunk metadata. The only host readback (and stream wait) of the state.
  void load(leaf_access& leaves, shape const& s, rmm::cuda_stream_view stream)
  {
    auto const n = static_cast<std::size_t>(s.num_chunks);
    std::vector<std::int32_t> node_counts(n);
    std::vector<std::uint8_t> node_bits(n);
    leaves.host_bytes(loc.chunk_count.leaf,
                      loc.chunk_count.buffer,
                      0,
                      n * sizeof(std::int32_t),
                      node_counts.data(),
                      stream);
    leaves.host_bytes(loc.chunk_bits.leaf, loc.chunk_bits.buffer, 0, n, node_bits.data(), stream);

    std::vector<std::int64_t> prefix{0};
    std::vector<chunk_meta> meta(n);
    prefix.reserve(n + 1);
    for (std::size_t c = 0; c < n; ++c) {
      auto const count = node_counts[c];
      if (count < 0 || count > kChunkRows || node_bits[c] > s.elem_size * 8) {
        throw std::runtime_error("bitpack chunk decode: corrupt chunk metadata");
      }
      std::int64_t const n_words =
        count == 0 ? 1 : (static_cast<std::int64_t>(count) * node_bits[c] + 31) >> 5;
      meta[c] =
        chunk_meta{prefix.back(), static_cast<std::int32_t>(n_words), count, node_bits[c], 0};
      prefix.push_back(prefix.back() + n_words);
    }
    if (static_cast<std::uint64_t>(prefix.back()) * 4 >
        range_of(leaves.column(), loc.packed).size) {
      throw std::runtime_error("bitpack chunk decode: packed is shorter than its chunks");
    }

    rmm::device_buffer uploaded(n * sizeof(chunk_meta), stream, leaves.mr());
    cuda_try(
      cudaMemcpyAsync(
        uploaded.data(), meta.data(), uploaded.size(), cudaMemcpyHostToDevice, stream.value()),
      "chunk metadata upload");
    counts      = std::move(node_counts);
    word_prefix = std::move(prefix);
    host_meta   = std::move(meta);
    device_meta = std::move(uploaded);
    elem_size   = s.elem_size;
    loaded      = true;
  }

  /// Stages the metadata, minima and packed words of chunks [chunk_begin, chunk_end) (a
  /// non-empty range) on the device.
  staged_chunks stage(leaf_access& leaves,
                      std::int64_t chunk_begin,
                      std::int64_t chunk_end,
                      rmm::cuda_stream_view stream)
  {
    if (!loaded) {
      auto const s = describe(leaves.column());
      check_range(chunk_begin, chunk_end, s.num_chunks);
      load(leaves, s, stream);
    } else {
      check_range(chunk_begin, chunk_end, static_cast<std::int64_t>(counts.size()));
    }

    auto const c0 = static_cast<std::size_t>(chunk_begin);
    auto const n  = static_cast<std::size_t>(chunk_end - chunk_begin);
    staged_chunks out;
    out.meta      = static_cast<const chunk_meta*>(device_meta.data()) + c0;
    out.word_base = word_prefix[c0];
    out.mins      = reinterpret_cast<const std::uint8_t*>(leaves.device_bytes(
      loc.chunk_min.leaf, loc.chunk_min.buffer, c0 * elem_size, n * elem_size, stream));
    out.mins_aligned =
      reinterpret_cast<std::uintptr_t>(out.mins) % std::min<std::size_t>(elem_size, 8) == 0;

    auto const packed_bytes = static_cast<std::uint64_t>(word_prefix[c0 + n] - word_prefix[c0]) * 4;
    if (packed_bytes != 0) {
      auto const* bytes = leaves.device_bytes(loc.packed.leaf,
                                              loc.packed.buffer,
                                              static_cast<std::uint64_t>(word_prefix[c0]) * 4,
                                              packed_bytes,
                                              stream);
      if (reinterpret_cast<std::uintptr_t>(bytes) % alignof(std::uint32_t) != 0) {
        // A buffer may sit at any payload offset; the kernels load whole words.
        if (aligned_words.size() < packed_bytes) {
          aligned_words = rmm::device_buffer(packed_bytes, stream, leaves.mr());
        }
        cuda_try(
          cudaMemcpyAsync(
            aligned_words.data(), bytes, packed_bytes, cudaMemcpyDeviceToDevice, stream.value()),
          "packed word realignment");
        bytes = static_cast<const std::byte*>(aligned_words.data());
      }
      out.words = reinterpret_cast<const std::uint32_t*>(bytes);
    }
    return out;
  }

  static void check_range(std::int64_t chunk_begin, std::int64_t chunk_end, std::int64_t num_chunks)
  {
    if (chunk_begin < 0 || chunk_end <= chunk_begin || chunk_end > num_chunks) {
      throw std::invalid_argument("bitpack chunk decode: chunks [" + std::to_string(chunk_begin) +
                                  ", " + std::to_string(chunk_end) +
                                  ") are not within the node's " + std::to_string(num_chunks));
    }
  }
};

namespace {

void delete_bitpack_state(bitpack_state* state) { delete state; }

/// Decoder of a `bitpack` root: values are written in the column's storage type, which is the
/// leaf's element type (chrono, decimal and floating columns are packed as their same-width
/// integer storage).
class bitpack_decoder final : public chunk_decoder {
 public:
  explicit bitpack_decoder(bitpack_locator const& locator) : _state(locator) {}

  std::int64_t decode(leaf_access& leaves,
                      chunk_span const& span,
                      column_write_target const& dst,
                      rmm::cuda_stream_view stream) override
  {
    if (span.take <= 0) { return 0; }
    // Only the chunks the rows touch, which may be fewer than the span offers.
    auto const chunk_end =
      std::min(span.chunk_end,
               span.chunk_begin + (span.skip_head + span.take + kChunkRows - 1) / kChunkRows);
    auto const staged = _state.stage(leaves, span.chunk_begin, chunk_end, stream);
    auto const width  = static_cast<std::size_t>(cudf::size_of(leaves.column().dtype));
    if (_state.elem_size != width) {
      throw unsupported_chunk_decode("bitpack chunk decode: the packed element is " +
                                     std::to_string(_state.elem_size) + " bytes, the column's " +
                                     std::to_string(width));
    }
    auto* const out =
      static_cast<std::byte*>(dst.data) + static_cast<std::size_t>(dst.dst_row_offset) * width;
    auto launch = [&]<typename T>(T*) {
      launch_unpack<T, T>(staged, span.skip_head, span.take, reinterpret_cast<T*>(out), stream);
    };
    switch (width) {
      case 1: launch(static_cast<std::uint8_t*>(nullptr)); break;
      case 2: launch(static_cast<std::uint16_t*>(nullptr)); break;
      case 4: launch(static_cast<std::uint32_t*>(nullptr)); break;
      case 8: launch(static_cast<std::uint64_t*>(nullptr)); break;
      default: {
        auto const blocks =
          static_cast<unsigned>((span.take + kThreadsPerBlock - 1) / kThreadsPerBlock);
        unpack_kernel_128<<<blocks, kThreadsPerBlock, 0, stream.value()>>>(
          staged, span.skip_head, span.take, reinterpret_cast<std::uint64_t*>(out));
        cuda_try(cudaGetLastError(), "unpack kernel launch");
        break;
      }
    }
    return span.take;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept override { return _state.device_bytes(); }

 private:
  bitpack_state _state;
};

}  // namespace

std::unique_ptr<chunk_decoder> make_bitpack_decoder(hpln_column_description const& column)
{
  // nodes[0] is the column input; its one edge leads to the bitpack node.
  if (column.nodes.empty() || column.nodes[0].edges.size() != 1) {
    throw unsupported_chunk_decode("bitpack chunk decode: the plan has no single root operator");
  }
  auto const locator = find_bitpack(column, column.nodes[0].edges[0].second);
  if (!locator) {
    throw unsupported_chunk_decode("bitpack chunk decode: the plan root is not a bitpack leaf");
  }
  if (!cudf::is_fixed_width(column.dtype)) {
    throw unsupported_chunk_decode("bitpack chunk decode: the column is not fixed width");
  }
  auto decoder = std::make_unique<bitpack_decoder>(*locator);
  // Reject a width mismatch now, so the cursor reports the column unsupported up front.
  auto const shape = bitpack_state{*locator}.describe(column);
  if (shape.num_chunks != 0 &&
      shape.elem_size != static_cast<std::size_t>(cudf::size_of(column.dtype))) {
    throw unsupported_chunk_decode(
      "bitpack chunk decode: the packed element width differs from the column's");
  }
  return decoder;
}

std::unique_ptr<bitpack_state, void (*)(bitpack_state*)> make_bitpack_state(
  bitpack_locator const& locator)
{
  return {new bitpack_state{locator}, &delete_bitpack_state};
}

void unpack_bitpack_chunks(leaf_access& leaves,
                           bitpack_state& state,
                           std::int64_t chunk_begin,
                           std::int64_t chunk_end,
                           std::int64_t* out,
                           rmm::cuda_stream_view stream)
{
  if (chunk_end == chunk_begin) { return; }
  auto const staged = state.stage(leaves, chunk_begin, chunk_end, stream);
  if (state.elem_size > sizeof(std::int64_t)) {
    throw unsupported_chunk_decode("bitpack chunk decode: 16-byte elements do not fit int64");
  }
  // One 1024-value block per chunk; the last chunk's block ends at its count.
  auto const n = (chunk_end - chunk_begin - 1) * kChunkRows +
                 state.counts[static_cast<std::size_t>(chunk_end - 1)];
  if (n <= 0) { return; }
  auto const type_id     = state.loc.value_type.id();
  bool const is_unsigned = type_id == cudf::type_id::UINT8 || type_id == cudf::type_id::UINT16 ||
                           type_id == cudf::type_id::UINT32 || type_id == cudf::type_id::UINT64;
  auto launch = [&]<typename T>(T*) { launch_unpack<T, std::int64_t>(staged, 0, n, out, stream); };
  switch (state.elem_size) {
    case 1:
      is_unsigned ? launch(static_cast<std::uint8_t*>(nullptr))
                  : launch(static_cast<std::int8_t*>(nullptr));
      break;
    case 2:
      is_unsigned ? launch(static_cast<std::uint16_t*>(nullptr))
                  : launch(static_cast<std::int16_t*>(nullptr));
      break;
    case 4:
      is_unsigned ? launch(static_cast<std::uint32_t*>(nullptr))
                  : launch(static_cast<std::int32_t*>(nullptr));
      break;
    default: launch(static_cast<std::int64_t*>(nullptr)); break;
  }
}
}  // namespace simpatico::chunk
