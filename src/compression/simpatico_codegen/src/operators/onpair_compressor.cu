// SPDX-License-Identifier: Apache-2.0
// OnPair-GPU decode, selected-row decode and equality over bitpacked codes, using the
// kernels in onpair_kernels.cuh: one warp decodes a batch of kOnPairBatchCodes codes,
// placed by the u32 output offset the sidecar stores for it.

#include "decode/decode_session.hpp"
#include "onpair_kernels.cuh"
#include "onpair_representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simpatico {
namespace {

namespace dd = onpair_kernels;

inline void check_cuda(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("onpair: ") + what + ": " + cudaGetErrorString(e));
  }
}

constexpr std::uint32_t kMaxTokenSize = dd::kMaxToken;
constexpr int kHoist                  = 1;

template <int W>
void launch_fastpair(onpair_compressed_representation const& rep,
                     std::uint8_t* out,
                     cudaStream_t stream,
                     onpair_byte_ranges const* ranges)
{
  std::uint64_t const num_batches = (rep.n_tokens + kOnPairBatchCodes - 1) / kOnPairBatchCodes;
  auto const* dict                = static_cast<std::uint8_t const*>(rep.dict_buf.data());
  dd::dictionary_planes<W> const planes{
    static_cast<std::uint32_t const*>(rep.codes_buf.data()),
    rep.code_bits,
    reinterpret_cast<typename dd::low_bytes<W>::type const*>(dict),
    reinterpret_cast<uint2 const*>(dict) + rep.dict_size,
    static_cast<std::uint8_t const*>(rep.lens_nib_buf.data())};
  auto const* sidecar = static_cast<std::uint32_t const*>(rep.sidecar_buf.data());
  if (ranges != nullptr) {
    // Enough ranges per warp to span about kBatchesPerWarp batches at the column's average
    // ranges per batch, so a batch shared by two warps is the exception; sparse ranges get a
    // warp each.
    constexpr std::uint64_t kBatchesPerWarp = 8;
    std::uint64_t const per_warp =
      std::clamp<std::uint64_t>((ranges->n * kBatchesPerWarp + num_batches - 1) / num_batches,
                                std::uint64_t{1},
                                std::uint64_t{1024});
    std::uint64_t const warps = (ranges->n + per_warp - 1) / per_warp;
    dd::decode_fixed_input_ranges<kOnPairK, W, kHoist>
      <<<static_cast<unsigned>((warps + dd::kWarpsPerBlock - 1) / dd::kWarpsPerBlock),
         dd::kThreads,
         0,
         stream>>>(planes,
                   sidecar,
                   rep.uncompressed_bytes,
                   rep.n_tokens,
                   num_batches,
                   ranges->src,
                   ranges->dst,
                   ranges->n,
                   per_warp,
                   out);
    return;
  }
  auto const blocks =
    static_cast<unsigned>((num_batches + dd::kWarpsPerBlock - 1) / dd::kWarpsPerBlock);
  dd::decode_fixed_input<kOnPairK, W, kHoist>
    <<<blocks, dd::kThreads, 0, stream>>>(planes, sidecar, out, rep.n_tokens, num_batches);
}

}  // namespace

void onpair_compressed_representation::launch(std::uint8_t* out,
                                              cudaStream_t stream,
                                              onpair_byte_ranges const* ranges) const
{
  if (w == 16) {
    launch_fastpair<16>(*this, out, stream, ranges);
  } else {
    launch_fastpair<8>(*this, out, stream, ranges);
  }
  check_cuda(cudaGetLastError(), "decode launch");
}

void onpair_compressed_representation::copy_ranges(std::int64_t const* src,
                                                   std::int32_t const* dst,
                                                   std::size_t n,
                                                   std::uint8_t* out,
                                                   ::cuda::stream_ref stream) const
{
  if (n == 0 || n_tokens == 0) return;
  onpair_byte_ranges const ranges{src, dst, n};
  launch(out, stream.get(), &ranges);
}

namespace {

template <typename Offset>
void launch_equal(onpair_compressed_representation const& rep,
                  Offset const* offsets,
                  std::uint64_t n_rows,
                  dd::code_literals list,
                  std::uint8_t* out,
                  cudaStream_t stream)
{
  std::uint64_t const num_batches = (rep.n_tokens + kOnPairBatchCodes - 1) / kOnPairBatchCodes;
  // A warp's batches share one search for their first row; 64K warps still fill the GPU.
  std::uint64_t const per_warp = std::clamp<std::uint64_t>(num_batches >> 16, 1, 16);
  std::uint64_t const warps    = (num_batches + per_warp - 1) / per_warp;
  dd::equal_fixed_input<kOnPairK, Offset>
    <<<static_cast<unsigned>((warps + dd::kWarpsPerBlock - 1) / dd::kWarpsPerBlock),
       dd::kThreads,
       0,
       stream>>>(static_cast<std::uint32_t const*>(rep.codes_buf.data()),
                 rep.code_bits,
                 static_cast<std::uint8_t const*>(rep.lens_nib_buf.data()),
                 static_cast<std::uint32_t const*>(rep.sidecar_buf.data()),
                 rep.uncompressed_bytes,
                 rep.n_tokens,
                 num_batches,
                 per_warp,
                 offsets,
                 n_rows,
                 list,
                 out);
}

}  // namespace

struct onpair_compressed_representation::token_index {
  std::once_flag built;
  std::vector<std::string> tokens;
  std::unordered_map<std::string_view, std::uint16_t> ids;
};

onpair_compressed_representation::onpair_compressed_representation(cudf::data_type type,
                                                                   cudf::size_type n_rows)
  : standalone_compressed_representation(type, n_rows),
    token_index_(std::make_shared<token_index>())
{
}

std::vector<std::string> onpair_compressed_representation::tokens(::cuda::stream_ref stream) const
{
  std::vector<std::uint8_t> bytes(dict_size * kMaxTokenSize);
  std::vector<std::uint8_t> nibbles((dict_size + 1) / 2);
  auto const s = stream.get();
  if (w == 16) {
    check_cuda(
      cudaMemcpyAsync(bytes.data(), dict_buf.data(), bytes.size(), cudaMemcpyDeviceToHost, s),
      "copy dictionary");
  } else {
    // Every token's first 8 bytes, then every token's last 8.
    auto const* split = static_cast<std::uint8_t const*>(dict_buf.data());
    check_cuda(
      cudaMemcpy2DAsync(bytes.data(), 16, split, 8, 8, dict_size, cudaMemcpyDeviceToHost, s),
      "copy dictionary");
    check_cuda(
      cudaMemcpy2DAsync(
        bytes.data() + 8, 16, split + dict_size * 8, 8, 8, dict_size, cudaMemcpyDeviceToHost, s),
      "copy dictionary");
  }
  check_cuda(
    cudaMemcpyAsync(nibbles.data(), lens_nib_buf.data(), nibbles.size(), cudaMemcpyDeviceToHost, s),
    "copy token lengths");
  stream.sync();
  std::vector<std::string> out(dict_size);
  for (std::size_t t = 0; t < dict_size; ++t) {
    std::size_t const len = ((nibbles[t / 2] >> ((t & 1) * 4)) & 15u) + 1;
    out[t].assign(reinterpret_cast<char const*>(bytes.data()) + t * kMaxTokenSize, len);
  }
  return out;
}

std::unique_ptr<cudf::column> onpair_compressed_representation::equal_any(
  cudf::column_view offsets,
  std::vector<std::string> const& literals,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  if (!string_aligned) {
    throw std::invalid_argument("onpair: equality on codes needs a string-aligned encoding");
  }
  auto const id = offsets.type().id();
  if (offsets.size() < 1 || offsets.has_nulls() ||
      (id != cudf::type_id::INT32 && id != cudf::type_id::INT64)) {
    throw std::invalid_argument("onpair: equality needs INT32 or INT64 row offsets");
  }
  auto const n_rows = static_cast<cudf::size_type>(offsets.size() - 1);
  auto result       = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                               n_rows,
                                               rmm::device_buffer(n_rows, stream, mr),
                                               rmm::device_buffer{},
                                               0);
  auto* out         = result->mutable_view().data<std::uint8_t>();
  if (n_rows == 0) return result;
  if (n_tokens == 0) {
    // Every row is empty.
    bool const empty = std::find(literals.begin(), literals.end(), std::string{}) != literals.end();
    check_cuda(cudaMemsetAsync(out, empty ? 1 : 0, n_rows, stream.get()), "fill");
    return result;
  }

  // Parse each literal as the encoder parses a string, longest token first. The dictionary
  // is pruned to the tokens the column uses, which hold every token a matching row's parse
  // takes, so a literal needing any other occurs in no row and is dropped.
  std::call_once(token_index_->built, [&] {
    token_index_->tokens = tokens(stream);
    for (std::size_t t = 0; t < token_index_->tokens.size(); ++t)
      token_index_->ids.emplace(token_index_->tokens[t], static_cast<std::uint16_t>(t));
  });
  auto const& ids = token_index_->ids;
  std::vector<std::uint32_t> lit_bytes, lit_first{0};
  std::vector<std::uint16_t> lit_codes;
  for (auto const& literal : literals) {
    std::size_t const mark = lit_codes.size();
    bool parsed            = true;
    for (std::size_t pos = 0; pos < literal.size() && parsed;) {
      parsed = false;
      for (std::size_t len = std::min<std::size_t>(kMaxTokenSize, literal.size() - pos); len > 0;
           --len) {
        if (auto const it = ids.find(std::string_view{literal}.substr(pos, len)); it != ids.end()) {
          lit_codes.push_back(it->second);
          pos += len;
          parsed = true;
          break;
        }
      }
    }
    if (!parsed) {
      lit_codes.resize(mark);
      continue;
    }
    lit_bytes.push_back(static_cast<std::uint32_t>(literal.size()));
    lit_first.push_back(static_cast<std::uint32_t>(lit_codes.size()));
  }
  if (lit_bytes.empty()) {
    check_cuda(cudaMemsetAsync(out, 0, n_rows, stream.get()), "fill");
    return result;
  }
  rmm::device_buffer d_bytes(
    lit_bytes.data(), lit_bytes.size() * sizeof(std::uint32_t), stream, mr);
  rmm::device_buffer d_first(
    lit_first.data(), lit_first.size() * sizeof(std::uint32_t), stream, mr);
  if (lit_codes.empty()) { lit_codes.push_back(0); }  // only empty literals
  rmm::device_buffer d_codes(
    lit_codes.data(), lit_codes.size() * sizeof(std::uint16_t), stream, mr);
  dd::code_literals const list{static_cast<std::uint32_t const*>(d_bytes.data()),
                               static_cast<std::uint32_t const*>(d_first.data()),
                               static_cast<std::uint16_t const*>(d_codes.data()),
                               static_cast<std::uint32_t>(lit_bytes.size())};
  if (id == cudf::type_id::INT32) {
    launch_equal(*this, offsets.data<std::int32_t>(), n_rows, list, out, stream.get());
  } else {
    launch_equal(*this, offsets.data<std::int64_t>(), n_rows, list, out, stream.get());
  }
  check_cuda(cudaGetLastError(), "equality kernel");
  // The literal buffers are freed stream-ordered, after the kernel.
  return result;
}

std::unique_ptr<cudf::column> onpair_compressed_representation::decompress(
  decode_frame& frame) const
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  rmm::device_buffer out(uncompressed_bytes, stream, mr);
  if (n_tokens > 0 && uncompressed_bytes > 0) {
    launch(static_cast<std::uint8_t*>(out.data()), stream.get(), nullptr);
  }
  return std::make_unique<cudf::column>(
    original_type, num_rows, std::move(out), rmm::device_buffer{}, 0);
}

namespace {

cudf::column_view buffer_view(cudf::type_id type, rmm::device_buffer const& b)
{
  auto const dt = cudf::data_type{type};
  return cudf::column_view(
    dt, static_cast<cudf::size_type>(b.size() / cudf::size_of(dt)), b.data(), nullptr, 0);
}

}  // namespace

std::vector<compressible_output> onpair_compressed_representation::named_channels(
  ::cuda::stream_ref) const
{
  auto const& names = op_info(OpId::OnPair).channels;
  return {{names[0], buffer_view(cudf::type_id::UINT32, codes_buf)},
          {names[1], buffer_view(cudf::type_id::UINT32, sidecar_buf)},
          {names[2], buffer_view(cudf::type_id::UINT8, dict_buf)},
          {names[3], buffer_view(cudf::type_id::UINT8, lens_nib_buf)}};
}

leaf_meta_v onpair_compressed_representation::describe_meta() const
{
  return leaf_meta::onpair{uncompressed_bytes,
                           static_cast<std::int32_t>(original_type.id()),
                           n_tokens,
                           static_cast<std::uint8_t>(code_bits),
                           static_cast<std::uint8_t>(w),
                           static_cast<std::uint8_t>(string_aligned)};
}

std::unique_ptr<onpair_compressed_representation> onpair_compressed_representation::from_channels(
  leaf_meta::onpair const& meta,
  std::unique_ptr<cudf::column> codes,
  std::unique_ptr<cudf::column> sidecar,
  std::unique_ptr<cudf::column> dict,
  std::unique_ptr<cudf::column> lens)
{
  if ((meta.w != 8 && meta.w != 16) || meta.code_bits == 0 || meta.code_bits > 16) {
    throw std::invalid_argument("onpair: unsupported stored configuration");
  }
  auto const type = cudf::data_type{static_cast<cudf::type_id>(meta.original_type_id)};
  if (!cudf::is_fixed_width(type) || meta.uncompressed_size % cudf::size_of(type) != 0) {
    throw std::invalid_argument("onpair: stored original type does not fit its size");
  }
  std::size_t const dict_bytes = static_cast<std::size_t>(dict->size());
  std::size_t const d          = dict_bytes / kMaxTokenSize;
  std::uint64_t const batches  = (meta.n_tokens + kOnPairBatchCodes - 1) / kOnPairBatchCodes;
  if (dict_bytes % kMaxTokenSize != 0 || static_cast<std::size_t>(lens->size()) != (d + 1) / 2 ||
      (std::uint64_t{1} << meta.code_bits) < d ||
      static_cast<std::uint64_t>(codes->size()) <
        batches * kOnPairBatchCodes * meta.code_bits / 32 + 1 ||
      static_cast<std::uint64_t>(sidecar->size()) != batches + 1) {
    throw std::invalid_argument("onpair: stored channels do not match the leaf metadata");
  }
  auto rep = std::make_unique<onpair_compressed_representation>(
    type, static_cast<cudf::size_type>(meta.uncompressed_size / cudf::size_of(type)));
  rep->dict_size          = d;
  rep->n_tokens           = meta.n_tokens;
  rep->uncompressed_bytes = meta.uncompressed_size;
  rep->code_bits          = meta.code_bits;
  rep->w                  = meta.w;
  rep->string_aligned     = meta.string_aligned != 0;
  rep->codes_buf          = std::move(*codes->release().data);
  rep->sidecar_buf        = std::move(*sidecar->release().data);
  rep->dict_buf           = std::move(*dict->release().data);
  rep->lens_nib_buf       = std::move(*lens->release().data);
  return rep;
}

}  // namespace simpatico
