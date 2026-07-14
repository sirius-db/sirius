/**
 * Dictionary compressor: STRING column -> dictionary column (encode); decompress via decode.
 * Stores the encoded dictionary column to avoid copying keys chars from cuDF (invalid pointers).
 */

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/tabulate.h>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>

namespace simpatico {

namespace {

constexpr size_t MAX_INDICES = 1 << 28;  // 256M rows (sanity bound)

// cudf::dictionary::encode faults with a context-corrupting illegal access on
// very large, very-high-cardinality strings — inputs a dictionary can't help
// anyway. Skip such columns before the full encode, gating on an estimate of
// the *full-column* distinct fraction: a prefix probe mis-reads long columns
// with moderate absolute cardinality (e.g. 1M distinct over 100M rows is ~0.23
// unique in a 256K-row prefix but 0.01 over the whole column — an ideal
// dictionary target). A fixed-memory HyperLogLog sketch gives the true
// fraction in one pass without materializing the keys column that faults.
constexpr size_t kDictCardCheckMinRows = 1 << 20;
constexpr double kDictMaxCardFraction  = 0.5;

// Constant key byte-width if every key has the same length, else 0 (STRING: not deducible from dtype).
int64_t measure_constant_key_width(cudf::strings_column_view const& keys,
                                   rmm::cuda_stream_view stream)
{
  auto const n = keys.size();
  if (n <= 0) return 0;
  auto const off = cudf::detail::offsetalator_factory::make_input_iterator(keys.offsets());
  int64_t w      = 0;
  {
    // offsets[1] - offsets[0], via a 1-element device round trip
    rmm::device_uvector<int64_t> first(1, stream);
    auto* d = first.data();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::counting_iterator<int>(0),
                       1,
                       [=] __device__(int) { d[0] = off[1] - off[0]; });
    cudaMemcpyAsync(&w, d, sizeof(w), cudaMemcpyDeviceToHost, stream.value());
    cudaStreamSynchronize(stream.value());
  }
  if (w <= 0) return 0;
  bool const all_equal = thrust::all_of(
    rmm::exec_policy(stream),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(n),
    [=] __device__(cudf::size_type i) { return off[i + 1] - off[i] == w; });
  return all_equal ? w : 0;
}

// Compile-time width lets the stitch loop fully unroll into registers — with a
// runtime width the byte shuffle spills to local memory.
template <int W>
void padded_gather_chunks(uint4 const* pool,
                          int32_t const* ix,
                          char* out,
                          int64_t nbytes,
                          rmm::cuda_stream_view stream)
{
  int64_t const nchunks = (nbytes + 15) / 16;
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::counting_iterator<int64_t>(0),
    nchunks,
    [=] __device__(int64_t t) {
      int64_t const base = t * 16;
      int const n        = nbytes - base < 16 ? static_cast<int>(nbytes - base) : 16;
      char b[16];
      int64_t row       = base / W;
      int64_t row_start = row * W;
      for (int i = 0; i < n;) {
        uint4 const v = pool[ix[row]];
        char a[16];
        memcpy(a, &v, 16);
        int const in_row = static_cast<int>(base + i - row_start);
        int const take   = W - in_row < n - i ? W - in_row : n - i;
#pragma unroll
        for (int j = 0; j < W; ++j)
          if (j < take) b[i + j] = a[in_row + j];
        i += take;
        row_start += W;
        ++row;
      }
      if (n == 16) {
        uint4 v;
        memcpy(&v, b, 16);
        *reinterpret_cast<uint4*>(out + base) = v;
      } else {
        for (int i = 0; i < n; ++i)
          out[base + i] = b[i];
      }
    });
}

// Constant-width null-free decode: analytic offsets + flat byte gather (skips cudf's
// batched-memcpy gather, offsets scan, and null-mask pass). nullptr = ineligible.
std::unique_ptr<cudf::column> try_decode_constant_width(cudf::strings_column_view const& keys,
                                                        cudf::column_view const& indices,
                                                        std::int64_t& cached_width,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  if (indices.null_count() > 0 || keys.parent().null_count() > 0) return nullptr;
  if (indices.type().id() != cudf::type_id::INT32) return nullptr;
  if (cached_width < 0) cached_width = measure_constant_key_width(keys, stream);
  int64_t const width = cached_width;
  if (width <= 0) return nullptr;
  auto const n_rows    = indices.size();
  int64_t const nbytes = static_cast<int64_t>(n_rows) * width;
  if (nbytes > std::numeric_limits<cudf::size_type>::max()) return nullptr;

  auto offsets = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::INT32), n_rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* d_off = offsets->mutable_view().data<int32_t>();
  thrust::tabulate(rmm::exec_policy(stream), d_off, d_off + n_rows + 1, [=] __device__(int64_t i) {
    return static_cast<int32_t>(i * width);
  });

  rmm::device_uvector<char> chars(nbytes, stream, mr);
  auto* out      = chars.data();
  auto const* kc = keys.chars_begin(stream);
  auto const* ix = indices.data<int32_t>();
  // One aligned 16B store per thread, assembled in registers from the rows
  // overlapping the chunk. Key-slice loads depend on pool size: small pools
  // are L1-resident so direct byte loads are fastest; a large pool (L2) is
  // first padded to a 16B stride so each overlapped row is one aligned uint4
  // load instead of per-byte L2 round trips.
  int64_t const nchunks = (nbytes + 15) / 16;
  auto const n_keys     = keys.size();
  bool const big_pool   = static_cast<int64_t>(n_keys) * width > (1 << 20);
  if (width <= 16 && big_pool) {
    rmm::device_uvector<char> padded(static_cast<int64_t>(n_keys) * 16, stream, mr);
    {
      auto* p = padded.data();
      thrust::for_each_n(rmm::exec_policy(stream),
                         thrust::counting_iterator<int64_t>(0),
                         static_cast<int64_t>(n_keys) * 16,
                         [=] __device__(int64_t i) {
                           int64_t const k = i / 16, o = i % 16;
                           p[i] = o < width ? kc[k * width + o] : 0;
                         });
    }
    auto const* pool = reinterpret_cast<uint4 const*>(padded.data());
    switch (width) {
      case 1: padded_gather_chunks<1>(pool, ix, out, nbytes, stream); break;
      case 2: padded_gather_chunks<2>(pool, ix, out, nbytes, stream); break;
      case 3: padded_gather_chunks<3>(pool, ix, out, nbytes, stream); break;
      case 4: padded_gather_chunks<4>(pool, ix, out, nbytes, stream); break;
      case 5: padded_gather_chunks<5>(pool, ix, out, nbytes, stream); break;
      case 6: padded_gather_chunks<6>(pool, ix, out, nbytes, stream); break;
      case 7: padded_gather_chunks<7>(pool, ix, out, nbytes, stream); break;
      case 8: padded_gather_chunks<8>(pool, ix, out, nbytes, stream); break;
      case 9: padded_gather_chunks<9>(pool, ix, out, nbytes, stream); break;
      case 10: padded_gather_chunks<10>(pool, ix, out, nbytes, stream); break;
      case 11: padded_gather_chunks<11>(pool, ix, out, nbytes, stream); break;
      case 12: padded_gather_chunks<12>(pool, ix, out, nbytes, stream); break;
      case 13: padded_gather_chunks<13>(pool, ix, out, nbytes, stream); break;
      case 14: padded_gather_chunks<14>(pool, ix, out, nbytes, stream); break;
      case 15: padded_gather_chunks<15>(pool, ix, out, nbytes, stream); break;
      case 16: padded_gather_chunks<16>(pool, ix, out, nbytes, stream); break;
    }
  } else {
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::counting_iterator<int64_t>(0),
      nchunks,
      [=] __device__(int64_t t) {
        int64_t const base = t * 16;
        int const n        = nbytes - base < 16 ? static_cast<int>(nbytes - base) : 16;
        char b[16];
        int64_t row       = base / width;
        int64_t row_start = row * width;
        for (int i = 0; i < n;) {
          auto const key_base = static_cast<int64_t>(ix[row]) * width;
          int const in_row    = static_cast<int>(base + i - row_start);
          int64_t const left  = width - in_row;
          int const take      = left < n - i ? static_cast<int>(left) : n - i;
          for (int j = 0; j < take; ++j)
            b[i + j] = kc[key_base + in_row + j];
          i += take;
          row_start += width;
          ++row;
        }
        if (n == 16) {
          uint4 v;
          memcpy(&v, b, 16);
          *reinterpret_cast<uint4*>(out + base) = v;
        } else {
          for (int i = 0; i < n; ++i)
            out[base + i] = b[i];
        }
      });
  }
  return cudf::make_strings_column(n_rows, std::move(offsets), chars.release(), 0, {});
}

std::unique_ptr<dictionary_compressed_representation> dictionary_compress_impl(
  cudf::column_view const& col,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  bool fast)
{
  if (col.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("dictionary_compressor: column must be STRING, got '" +
                             type_id_to_name(col.type()) + "'");
  }
  auto const n = col.size();
  if (n < 0) { throw std::runtime_error("dictionary_compressor: column size is negative"); }
  if (static_cast<size_t>(n) > MAX_INDICES) {
    throw std::runtime_error("dictionary_compressor: column size exceeds maximum");
  }
  if (n == 0) {
    if (fast) {
      // Empty fast rep exposes the {keys, indices} channels the fast plan expects.
      return std::make_unique<dictionary_compressed_representation>(
        cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING)),
        cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)));
    }
    auto empty_dict = cudf::make_empty_column(cudf::data_type(cudf::type_id::DICTIONARY32));
    return std::make_unique<dictionary_compressed_representation>(std::move(empty_dict));
  }

  if (static_cast<size_t>(n) > kDictCardCheckMinRows) {
    cudf::approx_distinct_count sketch(cudf::table_view{{col}},
                                       12,  // precision -> ~1.6% standard error
                                       cudf::null_policy::INCLUDE,
                                       cudf::nan_policy::NAN_IS_NULL,
                                       stream);
    auto const keys = sketch.estimate(stream);
    if (static_cast<double>(keys) > kDictMaxCardFraction * static_cast<double>(n)) {
      throw std::runtime_error("dictionary_compressor: cardinality too high (skipping)");
    }
  }

  auto dict_col = cudf::dictionary::encode(col, cudf::data_type(cudf::type_id::INT32), stream, mr);
  cudaStreamSynchronize(stream.value());
  if (fast) {
    // Fast mode: split the encoded dictionary into an owned keys (STRING) column and
    // an owned indices column carrying the parent null mask (get_indices_annotated), so
    // validity survives make_dictionary_column + decode. Leaner in-memory footprint than
    // the 3-buffer keys_offsets/keys_chars/indices form.
    cudf::dictionary_column_view dv(dict_col->view());
    auto keys    = std::make_unique<cudf::column>(dv.keys(), stream, mr);
    auto indices = std::make_unique<cudf::column>(dv.get_indices_annotated(), stream, mr);
    cudaStreamSynchronize(stream.value());
    return std::make_unique<dictionary_compressed_representation>(std::move(keys),
                                                                  std::move(indices));
  }
  return std::make_unique<dictionary_compressed_representation>(std::move(dict_col));
}

}  // namespace

std::unique_ptr<cudf::column> dictionary_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  nvtx3::scoped_range r{"dictionary_decompress"};
  // Fast reconstruction mode: create dictionary from keys + indices directly
  if (fast_mode) {
    if (!keys_column || !indices_only) { return nullptr; }
    if (indices_only->size() == 0) {
      return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
    }
    if (keys_column->size() == 0) {
      // Zero keys with rows present means every row is null (encode drops
      // null rows from the key set); decode would gather from the empty —
      // possibly childless — keys column, so build the all-null strings
      // column directly.
      return cudf::make_column_from_scalar(
        cudf::string_scalar("", false, stream, mr), indices_only->size(), stream, mr);
    }
    if (auto fast = try_decode_constant_width(cudf::strings_column_view(keys_column->view()),
                                              indices_only->view(),
                                              constant_key_width,
                                              stream,
                                              mr))
      return fast;
    // Create dictionary column from keys and indices, then decode
    // This avoids the expensive make_strings_column reconstruction from offsets+chars
    auto dict_col =
      cudf::make_dictionary_column(keys_column->view(), indices_only->view(), stream, mr);
    return cudf::dictionary::decode(dict_col->view(), stream, mr);
  }

  // Full dict_column mode: decode from the stored dictionary column.
  if (dict_column == nullptr) { return nullptr; }
  if (dict_column->size() == 0) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }
  if (cudf::dictionary_column_view(dict_column->view()).keys().size() == 0) {
    // Zero keys with rows present: all rows are null (see fast-mode note).
    return cudf::make_column_from_scalar(
      cudf::string_scalar("", false, stream, mr), dict_column->size(), stream, mr);
  }
  if (dict_column->null_count() == 0) {
    cudf::dictionary_column_view dv(dict_column->view());
    if (auto fast = try_decode_constant_width(
          cudf::strings_column_view(dv.keys()), dv.indices(), constant_key_width, stream, mr))
      return fast;
  }
  return cudf::dictionary::decode(dict_column->view(), stream, mr);
}

std::unique_ptr<compressed_representation> dictionary_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return dictionary_compress_impl(column_to_compress, stream, mr, fast_);
}

std::unique_ptr<cudf::column> dictionary_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* dict_repr =
    dynamic_cast<dictionary_compressed_representation const*>(&data_to_decompress);
  if (dict_repr == nullptr) { return nullptr; }
  return dict_repr->decompress(stream, mr);
}

}  // namespace simpatico
