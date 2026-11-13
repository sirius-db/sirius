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

#include "codegen/util/cuda_check.hpp"
#include "decode/chunk/chunk_decoders.hpp"

#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/type_traits>
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico::chunk {

namespace {

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

/// Packed words of a chunk of `count` values of `bits` bits (an empty chunk still takes one).
__host__ __device__ __forceinline__ std::int32_t chunk_words(std::int32_t count, std::int32_t bits)
{
  return count == 0
           ? 1
           : static_cast<std::int32_t>((static_cast<std::int64_t>(count) * bits + 31) >> 5);
}

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
  const std::uint8_t* counts{nullptr};      ///< chunk_count (int32) of chunk_begin
  const std::uint8_t* bits{nullptr};        ///< chunk_bits of chunk_begin
  const std::int64_t* word_begin{nullptr};  ///< each chunk's first word, from chunk_begin's
  const std::uint8_t* mins{nullptr};        ///< chunk_min element of chunk_begin
  const std::uint32_t* words{nullptr};      ///< packed words from chunk_begin's first word
  bool counts_aligned{false};               ///< counts can be loaded as whole elements
  bool mins_aligned{false};                 ///< mins can be loaded as whole elements

  __device__ __forceinline__ std::int32_t count(std::int64_t chunk) const
  {
    return load_element<std::int32_t>(counts + chunk * sizeof(std::int32_t), counts_aligned);
  }
};

/// Packed words of chunk i of the staged chunks: the input of the scan giving word_begin.
struct chunk_words_fn {
  staged_chunks in;

  __device__ std::int64_t operator()(std::int64_t i) const
  {
    return chunk_words(in.count(i), in.bits[i]);
  }
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
  const std::int32_t count = in.count(chunk);
  const auto idx           = static_cast<std::int32_t>(pos % kChunkRows);
  if (idx >= count) { return; }
  const std::int32_t bits = in.bits[chunk];
  std::uint64_t v         = load_element<U>(in.mins + chunk * sizeof(U), in.mins_aligned);
  if (bits != 0) {
    v += unpack_one(in.words + in.word_begin[chunk], chunk_words(count, bits), bits, idx);
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
  const std::int32_t count = in.count(chunk);
  const auto idx           = static_cast<std::int32_t>(pos % kChunkRows);
  if (idx >= count) { return; }
  const std::int32_t bits = in.bits[chunk];
  const std::uint8_t* min = in.mins + chunk * 16;
  u128 v = (static_cast<u128>(load_element<std::uint64_t>(min + 8, in.mins_aligned)) << 64) |
           load_element<std::uint64_t>(min, in.mins_aligned);
  if (bits != 0) {
    v += unpack_one_128(in.words + in.word_begin[chunk], chunk_words(count, bits), bits, idx);
  }
  out[2 * i]     = static_cast<std::uint64_t>(v);
  out[2 * i + 1] = static_cast<std::uint64_t>(v >> 64);
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
  throw_if_cuda_error(cudaGetLastError(), "bitpack chunk decode: unpack kernel launch");
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
  // kept on the host for the state's lifetime: 5 bytes per 1024 rows on the wire. word_prefix[c]
  // is chunk c's first word in `packed` and has one more entry than counts. The host needs the
  // prefix to know which packed bytes a span covers; nothing else is read back, so a decode
  // after the first never waits for the stream. The kernels read a span's counts and bits from
  // the staged leaf bytes, so nothing the state keeps on the device grows with the node.
  bool loaded{false};
  std::size_t elem_size{0};
  std::vector<std::int32_t> counts;
  std::vector<std::int64_t> word_prefix{0};
  rmm::device_buffer word_begin;     ///< grow-only scratch: first word of each chunk of a span
  rmm::device_buffer aligned_words;  ///< grow-only scratch for an unaligned `packed`

  [[nodiscard]] std::size_t device_bytes() const noexcept
  {
    return word_begin.size() + aligned_words.size();
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

  /// Reads the whole node's chunk_count / chunk_bits and builds the word prefix. The only host
  /// readback (and stream wait) of the state.
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
    prefix.reserve(n + 1);
    for (std::size_t c = 0; c < n; ++c) {
      auto const count = node_counts[c];
      if (count < 0 || count > kChunkRows || node_bits[c] > s.elem_size * 8) {
        throw std::runtime_error("bitpack chunk decode: corrupt chunk metadata");
      }
      prefix.push_back(prefix.back() + chunk_words(count, node_bits[c]));
    }
    if (static_cast<std::uint64_t>(prefix.back()) * 4 >
        range_of(leaves.column(), loc.packed).size) {
      throw std::runtime_error("bitpack chunk decode: packed is shorter than its chunks");
    }

    counts      = std::move(node_counts);
    word_prefix = std::move(prefix);
    elem_size   = s.elem_size;
    loaded      = true;
  }

  /// Stages the counts, bits, minima and packed words of chunks [chunk_begin, chunk_end) (a
  /// non-empty range) on the device, and scans the chunks' word counts into word_begin.
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
    out.counts =
      reinterpret_cast<const std::uint8_t*>(leaves.device_bytes(loc.chunk_count.leaf,
                                                                loc.chunk_count.buffer,
                                                                c0 * sizeof(std::int32_t),
                                                                n * sizeof(std::int32_t),
                                                                stream));
    out.counts_aligned = reinterpret_cast<std::uintptr_t>(out.counts) % alignof(std::int32_t) == 0;
    out.bits           = reinterpret_cast<const std::uint8_t*>(
      leaves.device_bytes(loc.chunk_bits.leaf, loc.chunk_bits.buffer, c0, n, stream));
    out.mins = reinterpret_cast<const std::uint8_t*>(leaves.device_bytes(
      loc.chunk_min.leaf, loc.chunk_min.buffer, c0 * elem_size, n * elem_size, stream));
    out.mins_aligned =
      reinterpret_cast<std::uintptr_t>(out.mins) % std::min<std::size_t>(elem_size, 8) == 0;

    if (word_begin.size() < n * sizeof(std::int64_t)) {
      word_begin = rmm::device_buffer(n * sizeof(std::int64_t), stream, leaves.mr());
    }
    auto* const span_word_begin = static_cast<std::int64_t*>(word_begin.data());
    thrust::transform_exclusive_scan(
      rmm::exec_policy_nosync(stream, leaves.mr()),
      thrust::counting_iterator<std::int64_t>{0},
      thrust::counting_iterator<std::int64_t>{static_cast<std::int64_t>(n)},
      span_word_begin,
      chunk_words_fn{out},
      std::int64_t{0},
      thrust::plus<std::int64_t>{});
    out.word_begin = span_word_begin;

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
        throw_if_cuda_error(
          cudaMemcpyAsync(
            aligned_words.data(), bytes, packed_bytes, cudaMemcpyDeviceToDevice, stream.value()),
          "bitpack chunk decode: packed word realignment");
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
        throw_if_cuda_error(cudaGetLastError(), "bitpack chunk decode: unpack kernel launch");
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
  auto const plan_shape = bitpack_state{*locator}.describe(column);
  if (plan_shape.num_chunks != 0 &&
      plan_shape.elem_size != static_cast<std::size_t>(cudf::size_of(column.dtype))) {
    throw unsupported_chunk_decode(
      "bitpack chunk decode: the packed element width differs from the column's");
  }
  return decoder;
}

std::size_t bitpack_state_bytes(bitpack_state const& state) noexcept
{
  return state.device_bytes();
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
