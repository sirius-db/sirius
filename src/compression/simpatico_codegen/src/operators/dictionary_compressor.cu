/**
 * Dictionary compressor: STRING column -> dictionary column (encode); decompress via decode.
 * Stores the encoded dictionary column to avoid copying keys chars from cuDF (invalid pointers).
 */

#include "codegen/plan/representation.hpp"
#include "codegen/util/cuda_check.hpp"
#include "codegen/util/nvtx.hpp"
#include "constant_width_offsets.hpp"
#include "decode/decode_session.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/functional>
#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <concepts>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {

namespace {

// Structural row bound only: a cudf column cannot exceed size_type rows, and
// the INT32 dictionary codes index the KEY SET (K distinct values), not rows,
// so any representable column encodes. Audit notes (iteration 6): the decode
// gathers do their addressing in int64 (key_base/nbytes below) and the
// constant-width fast path self-guards on nbytes > size_type max; the historic
// `1 << 28` "sanity bound" predated the HyperLogLog cardinality gate below,
// which now handles the real hazard (dictionary::encode's illegal access on
// huge HIGH-CARDINALITY inputs) by distinct FRACTION rather than row count —
// the row cap only silently forced narrow pins (>= 2^28 rows/chunk, e.g. q12's
// 5-col lineitem pin at ~276M and 2-3-col orders pins at ~600-900M) to raw.
// Encode transients (cudf's hash set + INT32 indices) scale O(n); an
// allocation failure throws and the pin falls back to an uncompressed chunk,
// visibly (see the per-pin coverage line in pin_table.cpp).
constexpr size_t MAX_INDICES = static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());

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

struct matching_key_width {
  cudf::detail::input_offsetalator offsets;

  __device__ int64_t operator()(cudf::size_type i) const
  {
    auto const width = offsets[1] - offsets[0];
    return width > 0 && offsets[i + 1] - offsets[i] == width ? width : 0;
  }
};

struct key_width_storage {
  void* scratch;
  int64_t* result;
};

/**
 * Enqueue the positive uniform key width, or zero for variable/empty strings, for nonempty keys.
 * The caller owns the storage and must retain it and the keys until the supplied stream completes.
 */
template <typename Allocate>
  requires requires(Allocate allocate, std::size_t bytes) {
    { allocate(bytes) } -> std::same_as<key_width_storage>;
  }
int64_t* enqueue_constant_key_width(cudf::strings_column_view const& keys,
                                    ::cuda::stream_ref stream,
                                    Allocate allocate)
{
  matching_key_width const matching_width{
    cudf::detail::offsetalator_factory::make_input_iterator(keys.offsets(), keys.offset())};
  auto reduce = [&](void* scratch, std::size_t& scratch_bytes, int64_t* result) {
    throw_if_cuda_error(
      cub::DeviceReduce::TransformReduce(scratch,
                                         scratch_bytes,
                                         thrust::counting_iterator<cudf::size_type>(0),
                                         result,
                                         keys.size(),
                                         cuda::minimum<int64_t>{},
                                         matching_width,
                                         std::numeric_limits<int64_t>::max(),
                                         stream.get()),
      "dictionary key width reduction");
  };
  std::size_t scratch_bytes = 0;
  reduce(nullptr, scratch_bytes, nullptr);
  auto const storage = allocate(scratch_bytes);
  reduce(storage.scratch, scratch_bytes, storage.result);
  return storage.result;
}

int64_t measure_constant_key_width(cudf::strings_column_view const& keys, decode_frame& frame)
{
  if (keys.size() <= 0) return 0;
  std::optional<rmm::device_buffer> storage;
  auto const* width = enqueue_constant_key_width(
    keys, frame.stream(), [&](std::size_t scratch_bytes) -> key_width_storage {
      if (scratch_bytes > std::numeric_limits<std::size_t>::max() - sizeof(int64_t)) {
        throw std::overflow_error("dictionary key width scratch size overflow");
      }
      storage.emplace(sizeof(int64_t) + scratch_bytes, frame.stream(), frame.mr());
      // CUB's queried size includes padding to align the scratch following the scalar.
      return {static_cast<char*>(storage->data()) + sizeof(int64_t),
              static_cast<int64_t*>(storage->data())};
    });
  return frame.read_scalar(width);
}

void drain_dictionary_observation(::cuda::stream_ref stream) noexcept
{
  auto const status = cudaStreamSynchronize(stream.get());
  if (status != cudaSuccess) {
    std::fprintf(stderr, "simpatico dictionary cleanup failed: %s\n", cudaGetErrorString(status));
  }
}

// Compile-time width lets the stitch loop fully unroll into registers — with a
// runtime width the byte shuffle spills to local memory.
template <int W>
void padded_gather_chunks(
  uint4 const* pool, int32_t const* ix, char* out, int64_t nbytes, decode_frame& frame)
{
  int64_t const nchunks = (nbytes + 15) / 16;
  thrust::for_each_n(rmm::exec_policy_nosync(frame.stream(), frame.mr()),
                     thrust::counting_iterator<int64_t>(0),
                     nchunks,
                     [=] __device__(int64_t t) {
                       int64_t const base = t * 16;
                       int const n = nbytes - base < 16 ? static_cast<int>(nbytes - base) : 16;
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

// Device view of an equals_any predicate's needles packed for one upload: `offsets` holds
// count + 1 byte offsets into `chars`, the needles' bytes back to back.
struct needle_view {
  std::int32_t const* offsets;
  char const* chars;
  std::int32_t count;
};

// The upload's device storage together with its view; the storage is released in stream order once
// the owner goes out of scope, so it must outlive the enqueue of every kernel that reads the view.
struct uploaded_needles {
  rmm::device_buffer storage;
  needle_view view;
};

// Per-key membership test: `lut[k]` is true iff key `k` equals some needle byte for byte, the same
// comparison as cuDF's STRING EQUAL. Keys are read through an offsetalator so INT32 and INT64 key
// offsets share one instantiation; the offsets are absolute into `key_chars`.
struct key_matches_any_needle {
  cudf::detail::input_offsetalator key_offsets;
  char const* key_chars;
  needle_view needles;

  __device__ bool operator()(cudf::size_type k) const
  {
    auto const key_begin = key_offsets[k];
    auto const key_size  = key_offsets[k + 1] - key_begin;
    for (std::int32_t j = 0; j < needles.count; ++j) {
      auto const needle_begin = needles.offsets[j];
      if (needles.offsets[j + 1] - needle_begin != key_size) continue;
      bool equal = true;
      for (int64_t b = 0; b < key_size && equal; ++b) {
        equal = key_chars[key_begin + b] == needles.chars[needle_begin + b];
      }
      if (equal) return true;
    }
    return false;
  }
};

// Stage the needles in frame-owned host storage, which lives until the session drains, and upload
// them once. A pageable host-to-device copy is staged by the runtime without waiting for the
// stream, so this is the whole cost of the predicate's host side.
uploaded_needles upload_needles(std::vector<std::string> const& needles, decode_frame& frame)
{
  auto const count = needles.size();
  if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::length_error("dictionary predicate: too many values");
  }
  std::size_t total_chars = 0;
  for (auto const& needle : needles) {
    if (needle.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - total_chars) {
      throw std::length_error("dictionary predicate: values exceed the packed size limit");
    }
    total_chars += needle.size();
  }
  // Offsets first so they stay naturally aligned at the start of both blocks; the chars follow.
  auto const offsets_bytes = (count + 1) * sizeof(std::int32_t);
  auto const total_bytes   = offsets_bytes + total_chars;
  auto const host          = frame.host_array<char>(total_bytes);
  auto* const host_chars   = host.data() + offsets_bytes;
  std::vector<std::int32_t> offsets(count + 1);
  std::int32_t cursor = 0;
  for (std::size_t j = 0; j < count; ++j) {
    offsets[j] = cursor;
    if (!needles[j].empty()) std::memcpy(host_chars + cursor, needles[j].data(), needles[j].size());
    cursor += static_cast<std::int32_t>(needles[j].size());
  }
  offsets[count] = cursor;
  std::memcpy(host.data(), offsets.data(), offsets_bytes);
  rmm::device_buffer storage(total_bytes, frame.stream(), frame.mr());
  throw_if_cuda_error(
    cudaMemcpyAsync(
      storage.data(), host.data(), total_bytes, cudaMemcpyHostToDevice, frame.stream().get()),
    "dictionary predicate: upload values");
  needle_view const view{static_cast<std::int32_t const*>(storage.data()),
                         static_cast<char const*>(storage.data()) + offsets_bytes,
                         static_cast<std::int32_t>(count)};
  return {std::move(storage), view};
}

// Constant-width null-free decode: analytic offsets + flat byte gather (skips cudf's
// batched-memcpy gather, offsets scan, and null-mask pass). nullptr = ineligible.
std::unique_ptr<cudf::column> try_decode_constant_width(cudf::strings_column_view const& keys,
                                                        cudf::column_view const& indices,
                                                        std::int64_t cached_width,
                                                        decode_frame& frame)
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  if (indices.null_count() > 0 || keys.parent().null_count() > 0) return nullptr;
  if (indices.type().id() != cudf::type_id::INT32) return nullptr;
  int64_t const width = cached_width < 0 ? measure_constant_key_width(keys, frame) : cached_width;
  if (width <= 0) return nullptr;
  auto const n_rows    = indices.size();
  int64_t const nbytes = static_cast<int64_t>(n_rows) * width;
  if (nbytes > std::numeric_limits<cudf::size_type>::max()) return nullptr;

  auto offsets = make_constant_width_offsets(n_rows, static_cast<std::int32_t>(width), stream, mr);
  rmm::device_buffer chars(nbytes, stream, mr);
  auto output = cudf::make_strings_column(n_rows, std::move(offsets), std::move(chars), 0, {});

  auto* out      = output->mutable_view().head<char>();
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
    rmm::device_buffer padded(static_cast<std::size_t>(n_keys) * 16, stream, mr);
    {
      auto* p = static_cast<char*>(padded.data());
      thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                         thrust::counting_iterator<int64_t>(0),
                         static_cast<int64_t>(n_keys) * 16,
                         [=] __device__(int64_t i) {
                           int64_t const k = i / 16, o = i % 16;
                           p[i] = o < width ? kc[k * width + o] : 0;
                         });
    }
    auto const* pool = reinterpret_cast<uint4 const*>(padded.data());
    switch (width) {
      case 1: padded_gather_chunks<1>(pool, ix, out, nbytes, frame); break;
      case 2: padded_gather_chunks<2>(pool, ix, out, nbytes, frame); break;
      case 3: padded_gather_chunks<3>(pool, ix, out, nbytes, frame); break;
      case 4: padded_gather_chunks<4>(pool, ix, out, nbytes, frame); break;
      case 5: padded_gather_chunks<5>(pool, ix, out, nbytes, frame); break;
      case 6: padded_gather_chunks<6>(pool, ix, out, nbytes, frame); break;
      case 7: padded_gather_chunks<7>(pool, ix, out, nbytes, frame); break;
      case 8: padded_gather_chunks<8>(pool, ix, out, nbytes, frame); break;
      case 9: padded_gather_chunks<9>(pool, ix, out, nbytes, frame); break;
      case 10: padded_gather_chunks<10>(pool, ix, out, nbytes, frame); break;
      case 11: padded_gather_chunks<11>(pool, ix, out, nbytes, frame); break;
      case 12: padded_gather_chunks<12>(pool, ix, out, nbytes, frame); break;
      case 13: padded_gather_chunks<13>(pool, ix, out, nbytes, frame); break;
      case 14: padded_gather_chunks<14>(pool, ix, out, nbytes, frame); break;
      case 15: padded_gather_chunks<15>(pool, ix, out, nbytes, frame); break;
      case 16: padded_gather_chunks<16>(pool, ix, out, nbytes, frame); break;
    }
  } else {
    thrust::for_each_n(rmm::exec_policy_nosync(stream, mr),
                       thrust::counting_iterator<int64_t>(0),
                       nchunks,
                       [=] __device__(int64_t t) {
                         int64_t const base = t * 16;
                         int const n = nbytes - base < 16 ? static_cast<int>(nbytes - base) : 16;
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
  return output;
}

std::unique_ptr<dictionary_compressed_representation> dictionary_compress_impl(
  cudf::column_view const& col, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
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
    auto empty_dict = cudf::make_empty_column(cudf::data_type(cudf::type_id::DICTIONARY32));
    return dictionary_compressed_representation::from_encoded_column(
      std::move(empty_dict), stream, mr);
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
  return dictionary_compressed_representation::from_encoded_column(std::move(dict_col), stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> make_constant_width_offsets(cudf::size_type rows,
                                                          std::int32_t width,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  if (rows < 0 || width < 0) {
    throw std::invalid_argument("constant-width offsets: negative row count or width");
  }
  if (rows == std::numeric_limits<cudf::size_type>::max() ||
      static_cast<std::int64_t>(rows) * width > std::numeric_limits<cudf::size_type>::max()) {
    throw std::overflow_error(
      "constant-width offsets: offset count or total bytes exceed the INT32 range");
  }
  auto offsets = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, rows + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* const data = offsets->mutable_view().data<std::int32_t>();
  thrust::tabulate(
    rmm::exec_policy_nosync(stream, mr), data, data + rows + 1, [=] __device__(int64_t i) {
      return static_cast<std::int32_t>(i * width);
    });
  return offsets;
}

std::unique_ptr<dictionary_compressed_representation>
dictionary_compressed_representation::from_encoded_column(std::unique_ptr<cudf::column> dict_col,
                                                          ::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  std::unique_ptr<dictionary_compressed_representation> result;
  std::optional<rmm::device_buffer> width_storage;
  // The explicit pinned resource makes this buffer both host- and device-accessible.
  std::optional<rmm::device_buffer> width_result;
  try {
    if (!dict_col) throw std::invalid_argument("dictionary construction: missing column");
    result = std::make_unique<dictionary_compressed_representation>(std::move(dict_col));
    result->constant_key_width = 0;
    if (result->dict_column->size() > 0) {
      auto const keys = cudf::dictionary_column_view(result->dict_column->view()).keys();
      if (keys.size() > 0) {
        enqueue_constant_key_width(
          cudf::strings_column_view(keys), stream, [&](std::size_t bytes) -> key_width_storage {
            width_storage.emplace(bytes, stream, mr);
            width_result.emplace(sizeof(int64_t), stream, cudf::get_pinned_memory_resource());
            return {width_storage->data(), static_cast<int64_t*>(width_result->data())};
          });
      }
    }
    stream.sync();
    if (width_result) {
      result->constant_key_width = *static_cast<int64_t const*>(width_result->data());
    }
    return result;
  } catch (...) {
    // Keep the column, scratch, and private result destination alive until pending work drains.
    drain_dictionary_observation(stream);
    throw;
  }
}

std::unique_ptr<cudf::column> dictionary_compressed_representation::decompress(
  decode_frame& frame) const
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  nvtx_scoped_range r{"dictionary_decompress"};
  // Decode from the stored dictionary column.
  if (dict_column == nullptr) { throw std::invalid_argument("dictionary decode: missing column"); }
  if (dict_column->size() == 0) {
    return cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
  }
  if (cudf::dictionary_column_view(dict_column->view()).keys().size() == 0) {
    // Zero keys with rows present: every row is null (encode drops null rows
    // from the key set), so build the all-null strings column directly.
    auto const rows = dict_column->size();
    return cudf::make_strings_column(
      rows,
      make_constant_width_offsets(rows, 0, stream, mr),
      rmm::device_buffer{},
      rows,
      cudf::create_null_mask(rows, cudf::mask_state::ALL_NULL, stream, mr));
  }
  if (dict_column->null_count() == 0) {
    cudf::dictionary_column_view dv(dict_column->view());
    if (auto output = try_decode_constant_width(
          cudf::strings_column_view(dv.keys()), dv.indices(), constant_key_width, frame))
      return output;
  }
  return cudf::dictionary::decode(dict_column->view(), stream, mr);
}

std::unique_ptr<cudf::column> dictionary_compressed_representation::decompress_predicate(
  decode_predicate const& pred, decode_frame& frame) const
{
  auto const stream = frame.stream();
  auto const mr     = frame.mr();
  nvtx_scoped_range r{"dictionary_decompress_predicate"};
  if (dict_column == nullptr) {
    throw std::invalid_argument("dictionary predicate: missing column");
  }
  if (!pred.active()) { return nullptr; }

  auto const n_rows = dict_column->size();
  auto const bool_t = cudf::data_type{cudf::type_id::BOOL8};
  if (n_rows == 0) { return cudf::make_empty_column(bool_t); }

  cudf::dictionary_column_view const dv(dict_column->view());
  auto const keys    = dv.keys();
  auto const n_keys  = keys.size();
  auto const indices = dv.indices();

  // Zero keys with rows present: encode drops null rows from the key set, so
  // every row is null and the comparison is null throughout.
  if (n_keys == 0) {
    auto output =
      cudf::make_fixed_width_column(bool_t, n_rows, cudf::mask_state::ALL_NULL, stream, mr);
    output->set_null_count(n_rows);
    return output;
  }
  // The index lookup below reads indices[i] unconditionally, so anything other
  // than the INT32 index type encode produces is left to the generic path.
  if (indices.type().id() != cudf::type_id::INT32) { return nullptr; }
  if (keys.type().id() != cudf::type_id::STRING) { return nullptr; }
  // A null key would make the per-key comparison null rather than false, and the
  // OR-accumulate below would then propagate it. cudf::dictionary::encode never
  // produces one (nulls live on the parent's mask, not in the key set), so leave
  // the shape to the generic path instead of carrying a tri-state accumulate.
  if (keys.null_count() > 0) { return nullptr; }

  // One bool per distinct value: key in equals_any. The key set is the column's whole
  // distinct-value population (four entries for l_shipinstruct), so this pass is noise next to the
  // row-length pass below, which is the entire point: this is the work that replaces the decode
  // gather. No cuDF scalar is constructed, so nothing here waits for the stream or borrows from the
  // process-global pinned pool; the needles and the table live until the row pass has been queued.
  auto const needles = upload_needles(pred.equals_any, frame);
  auto lut =
    cudf::make_fixed_width_column(bool_t, n_keys, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::strings_column_view const key_strings(keys);
  thrust::tabulate(rmm::exec_policy_nosync(stream, mr),
                   lut->mutable_view().begin<bool>(),
                   lut->mutable_view().end<bool>(),
                   key_matches_any_needle{cudf::detail::offsetalator_factory::make_input_iterator(
                                            key_strings.offsets(), key_strings.offset()),
                                          key_strings.chars_begin(stream),
                                          needles.view});

  // Only the *row* validity needs carrying: the keys are non-null (checked
  // above), so a matching code is unambiguously true.
  auto const null_count = dict_column->null_count();
  auto output           = cudf::make_fixed_width_column(
    bool_t,
    n_rows,
    null_count > 0 ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  output->set_null_count(null_count);
  if (null_count > 0) {
    throw_if_cuda_error(cudaMemcpyAsync(output->mutable_view().null_mask(),
                                        dict_column->view().null_mask(),
                                        cudf::bitmask_allocation_size_bytes(n_rows),
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()),
                        "dictionary predicate: copy null mask");
  }

  auto* d_out       = output->mutable_view().data<bool>();
  auto const* d_lut = lut->view().data<bool>();
  auto const* d_idx = indices.data<int32_t>();
  // Null rows carry an unspecified index (encode does not promise 0), so clamp
  // rather than trust it — the value is masked off either way.
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    d_idx,
                    d_idx + n_rows,
                    d_out,
                    [d_lut, n_keys] __device__(int32_t code) {
                      return code >= 0 && code < n_keys ? d_lut[code] : false;
                    });
  return output;
}

std::unique_ptr<compressed_representation> dictionary_compressor::compress(
  cudf::column_view column_to_compress,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return dictionary_compress_impl(column_to_compress, stream, mr);
}

}  // namespace simpatico
