// ALP-RD (Right-Dictionary) compressor — FLOAT32 and FLOAT64.
// Reference: cwida/ALP `include/alp/rd.hpp` (MIT).
// Paper: "ALP: Adaptive Lossless Floating-Point Compression" (Afroozeh,
// Kuffo, Boncz, SIGMOD '24), §5 "ALP-RD".
//
// Deviates from the cwida CPU reference in two ways:
//   - Column-wide (left_bw, dict) instead of per-rowgroup, so right_parts
//     stays fixed-width across the column.
//   - Exceptions stored as (positions[], values[]) pair instead of per-vector
//     dense flag array — matches Simpatico's plain-ALP and FOR layout.
//
// Templated on T (float|double): the sampling pass, encode/decode kernels,
// and host orchestrator all parametrise on T via alp_rd_traits<T>. The
// per-type constants (input bit width, right_bw search range, intrinsic
// for bits→float) live in the trait struct.
//
// Output columns vary by type:
//   right_parts  UINT32 (f32) / UINT64 (f64)
//   dict_indices UINT8                                       (both)
//   dict         UINT16 — top-K column-wide left parts       (both)
//   metadata     UINT8  — one entry: right_bw                (both)
//   exceptions   UINT16 — rejected left parts                (both)
//   exception_positions INT32                                (both)
//
// For f64 we constrain the right_bw search so the "left" part still fits in
// 16 bits (right_bw ∈ [48..63]), which keeps the dict column UINT16 across
// types. This matches the upstream cwida convention.

#include "codegen/plan/representation.hpp"
#include "codegen/util/cuda_check.hpp"
#include "operators/alp_common.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace simpatico {

namespace {

// -----------------------------------------------------------------------------
// Common constants
// -----------------------------------------------------------------------------

constexpr int kAlpRdVectorSize    = 1024;  // same vector unit as ALP
constexpr int kAlpRdMaxDictSize   = 8;     // K, fixed at 8 entries
constexpr int kAlpRdDictBitWidth  = 3;     // log2(K)
constexpr int kAlpRdCuttingLimit  = 16;    // search left_bw in [1..16]
constexpr int kAlpRdSampleVectors = 32;    // vectors used to pick (left_bw, dict)

// Exception cost in bits per exception used by the size estimator. Matches
// the CPU reference's `RD_EXCEPTION_POSITION_SIZE + RD_EXCEPTION_SIZE`.
constexpr double kAlpRdExceptionCostBits = 16.0 + 16.0;

// -----------------------------------------------------------------------------
// alp_rd_traits<T>: per-type bit width, uint pattern type, right_bw search
// range, and the bits→float cast intrinsic.
// -----------------------------------------------------------------------------

template <typename T>
struct alp_rd_traits;

template <>
struct alp_rd_traits<float> {
  using value_t                               = float;
  using uint_t                                = uint32_t;
  static constexpr int total_bits             = 32;
  static constexpr cudf::type_id uint_type_id = cudf::type_id::UINT32;
  // right_bw search: cwida CPU reference scans left_bw ∈ [1, CUTTING_LIMIT]
  // which puts right_bw in [total_bits - CUTTING_LIMIT, total_bits - 1].
  // For f32 with CUTTING_LIMIT=16, that's right_bw ∈ [16..31].
  static constexpr int right_bw_min = total_bits - kAlpRdCuttingLimit;  // 16
  static constexpr int right_bw_max = total_bits - 1;                   // 31
  static constexpr uint_t all_ones  = 0xFFFFFFFFu;
  __device__ static value_t bits_to_value(uint_t bits)
  {
    return __int_as_float(static_cast<int32_t>(bits));
  }
};

template <>
struct alp_rd_traits<double> {
  using value_t                               = double;
  using uint_t                                = uint64_t;
  static constexpr int total_bits             = 64;
  static constexpr cudf::type_id uint_type_id = cudf::type_id::UINT64;
  // For f64 right_bw ∈ [48..63] keeps the "left" part ≤ 16 bits, matching
  // the uint16 dict column shared with the f32 path.
  static constexpr int right_bw_min = total_bits - kAlpRdCuttingLimit;  // 48
  static constexpr int right_bw_max = total_bits - 1;                   // 63
  static constexpr uint_t all_ones  = 0xFFFFFFFFFFFFFFFFull;
  __device__ static value_t bits_to_value(uint_t bits)
  {
    return __longlong_as_double(static_cast<long long>(bits));
  }
};

// -----------------------------------------------------------------------------
// Sampling + dictionary build (CPU side). The number of values sampled per
// column is small (≤ kAlpRdSampleVectors × kAlpRdVectorSize = 32 K), so doing
// this on the host is negligible vs. a per-vector GPU frequency search.
// -----------------------------------------------------------------------------

struct rd_choice {
  uint8_t right_bw;                  // = total_bits - left_bw
  uint8_t actual_dict_size;          // ≤ K = 8
  uint16_t dict[kAlpRdMaxDictSize];  // top-K left parts, freq desc
};

// Estimate compressed bits per element for a given (left_bw, dict) on the
// sample. Matches `rd_encoder::estimate_compression_size`. Lower is better.
double estimate_bits_per_value(int left_bw,
                               int right_bw,
                               uint64_t exceptions_count,
                               uint64_t sample_count)
{
  if (sample_count == 0) return 0.0;
  double per_value     = static_cast<double>(left_bw) + static_cast<double>(right_bw);
  double exc_amortised = (static_cast<double>(exceptions_count) * kAlpRdExceptionCostBits) /
                         static_cast<double>(sample_count);
  return per_value + exc_amortised;
}

// Templated on the host-side uint pattern type (uint32_t for f32, uint64_t
// for f64). Builds a frequency map of the top `right_bw`-bit-shifted prefixes
// (which always fit in 16 bits because we constrain right_bw appropriately)
// and returns the bits/value estimate. Populates out_choice when non-null.
template <typename UintT>
double try_right_bw(const UintT* sample, size_t sample_n, int right_bw, rd_choice* out_choice)
{
  std::unordered_map<uint16_t, uint64_t> freq;
  freq.reserve(sample_n / 4);
  for (size_t i = 0; i < sample_n; ++i) {
    uint16_t left = static_cast<uint16_t>(sample[i] >> right_bw);
    freq[left]++;
  }

  std::vector<std::pair<uint64_t, uint16_t>> sorted;
  sorted.reserve(freq.size());
  for (auto const& kv : freq)
    sorted.emplace_back(kv.second, kv.first);
  std::sort(
    sorted.begin(), sorted.end(), [](auto const& a, auto const& b) { return a.first > b.first; });

  uint64_t exc_count = 0;
  for (size_t i = kAlpRdMaxDictSize; i < sorted.size(); ++i) {
    exc_count += sorted[i].first;
  }

  uint8_t actual_dict_size =
    static_cast<uint8_t>(std::min<size_t>(kAlpRdMaxDictSize, sorted.size()));
  // Effective dict_indices width downstream bitpack will use. When the dict
  // covers every sample value (exc_count == 0) indices live in
  // [0, actual_dict_size) so log2(K)=3 bits suffice. With exceptions present
  // the encode kernel writes the marker K (= kAlpRdMaxDictSize) which needs
  // 4 bits. Keeping this honest in the cost estimate avoids picking a worse
  // right_bw when bitpack would charge an extra bit.
  int left_bw = (exc_count > 0) ? (kAlpRdDictBitWidth + 1) : kAlpRdDictBitWidth;

  if (out_choice != nullptr) {
    out_choice->right_bw         = static_cast<uint8_t>(right_bw);
    out_choice->actual_dict_size = actual_dict_size;
    for (int i = 0; i < kAlpRdMaxDictSize; ++i) {
      out_choice->dict[i] = (i < actual_dict_size) ? sorted[i].second : uint16_t{0};
    }
  }

  return estimate_bits_per_value(left_bw, right_bw, exc_count, sample_n);
}

template <typename T>
rd_choice find_best_choice(const typename alp_rd_traits<T>::uint_t* sample, size_t sample_n)
{
  using traits = alp_rd_traits<T>;
  rd_choice best{};
  rd_choice cand{};
  double best_cost = std::numeric_limits<double>::max();
  for (int right_bw = traits::right_bw_min; right_bw <= traits::right_bw_max; ++right_bw) {
    double cost = try_right_bw<typename traits::uint_t>(sample, sample_n, right_bw, &cand);
    if (cost < best_cost) {
      best_cost = cost;
      best      = cand;
    }
  }
  if (best.right_bw == 0) {
    // Degenerate case (e.g. all-zero sample). Default to a sensible right_bw
    // that won't divide-by-zero downstream — middle of the valid range.
    best.right_bw         = static_cast<uint8_t>((traits::right_bw_min + traits::right_bw_max) / 2);
    best.actual_dict_size = 0;
    for (int i = 0; i < kAlpRdMaxDictSize; ++i)
      best.dict[i] = 0;
  }
  return best;
}

// -----------------------------------------------------------------------------
// Encode kernel. One thread per row.
//
// Inputs:  raw bit patterns of the input column, the chosen (right_bw, dict)
// Outputs: right_parts[i], dict_indices[i], exception_left[i], flag[i]
//          (full exception_positions/exceptions are gathered by compact_exceptions
//           in the host orchestrator, same pattern as plain ALP)
//
// The dict (≤ K = 8 UINT16 entries) lives in shared memory so all 256 threads
// in the block can probe it without going to global memory each time.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void alp_rd_encode_kernel(
  const typename alp_rd_traits<T>::uint_t* __restrict__ in_bits,
  int32_t n_rows,
  uint8_t right_bw,
  uint8_t actual_dict_size,
  const uint16_t* __restrict__ g_dict,
  typename alp_rd_traits<T>::uint_t* __restrict__ out_right_parts,
  uint8_t* __restrict__ out_dict_indices,
  uint16_t* __restrict__ out_exception_left,
  uint8_t* __restrict__ out_exception_flag)
{
  using traits = alp_rd_traits<T>;
  using uint_t = typename traits::uint_t;

  __shared__ uint16_t s_dict[kAlpRdMaxDictSize];
  if (threadIdx.x < kAlpRdMaxDictSize) { s_dict[threadIdx.x] = g_dict[threadIdx.x]; }
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows) return;

  uint_t bits       = in_bits[i];
  uint_t right_mask = (right_bw >= traits::total_bits)
                        ? traits::all_ones
                        : ((static_cast<uint_t>(1) << right_bw) - static_cast<uint_t>(1));
  uint_t right      = bits & right_mask;
  uint16_t left     = static_cast<uint16_t>(bits >> right_bw);

  // Linear scan over the 8-entry dict. The exception marker is fixed at
  // kAlpRdMaxDictSize (= K = 8), not `actual_dict_size`: when the dict is
  // partially populated (< 8 entries) and the marker equalled actual_dict_size
  // decode couldn't distinguish "valid slot N" from "exception" because it
  // doesn't know actual_dict_size at run time (only the column-wide K).
  // Using K as the marker keeps decode's `idx < kAlpRdMaxDictSize` valid.
  uint8_t idx = kAlpRdMaxDictSize;
#pragma unroll
  for (int k = 0; k < kAlpRdMaxDictSize; ++k) {
    if (k < actual_dict_size && s_dict[k] == left) { idx = static_cast<uint8_t>(k); }
  }

  out_right_parts[i]    = right;
  out_dict_indices[i]   = idx;
  bool is_exc           = (idx == kAlpRdMaxDictSize);
  out_exception_left[i] = is_exc ? left : uint16_t{0};
  out_exception_flag[i] = is_exc ? 1u : 0u;
}

// -----------------------------------------------------------------------------
// Decode kernel. One thread per row.
//
// Branch-free for the common path: every row looks up its dict entry and
// composes `value = bits_to_value((dict[idx] << right_bw) | right)`. Exception
// rows are overwritten afterwards by alp_rd_scatter_exceptions_kernel — same
// data-parallel pattern G-ALP recommends for plain ALP exceptions.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void alp_rd_decode_kernel(
  const typename alp_rd_traits<T>::uint_t* __restrict__ right_parts,
  const uint8_t* __restrict__ dict_indices,
  const uint16_t* __restrict__ g_dict,
  uint8_t right_bw,
  int32_t n_rows,
  T* __restrict__ out)
{
  using traits = alp_rd_traits<T>;
  using uint_t = typename traits::uint_t;

  __shared__ uint16_t s_dict[kAlpRdMaxDictSize];
  if (threadIdx.x < kAlpRdMaxDictSize) { s_dict[threadIdx.x] = g_dict[threadIdx.x]; }
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows) return;

  // For exception rows dict_indices[i] is K (out of range). Read s_dict[0]
  // unconditionally to keep the path branch-free; the exception scatter pass
  // will overwrite the result. Reading s_dict[K] would be UB (only K entries).
  uint8_t idx   = dict_indices[i];
  uint16_t left = (idx < kAlpRdMaxDictSize) ? s_dict[idx] : s_dict[0];

  uint_t bits = (static_cast<uint_t>(left) << right_bw) | right_parts[i];
  out[i]      = traits::bits_to_value(bits);
}

// Patch exception rows with the recombined (exception_left << right_bw) | right.
template <typename T>
__global__ void alp_rd_scatter_exceptions_kernel(
  const uint16_t* __restrict__ exception_lefts,
  const int32_t* __restrict__ positions,
  const typename alp_rd_traits<T>::uint_t* __restrict__ right_parts,
  int32_t exc_count,
  uint8_t right_bw,
  T* __restrict__ out)
{
  using traits = alp_rd_traits<T>;
  using uint_t = typename traits::uint_t;
  int k        = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= exc_count) return;
  int32_t pos = positions[k];
  uint_t bits = (static_cast<uint_t>(exception_lefts[k]) << right_bw) | right_parts[pos];
  out[pos]    = traits::bits_to_value(bits);
}

// -----------------------------------------------------------------------------
// Host orchestration (compress)
// -----------------------------------------------------------------------------

template <typename T>
std::unique_ptr<alp_rd_compressed_representation> alp_rd_compress_impl(
  cudf::column_view const& col, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  using traits = alp_rd_traits<T>;
  using uint_t = typename traits::uint_t;

  auto const n    = col.size();
  auto make_empty = [&](cudf::type_id tid) {
    return cudf::make_fixed_width_column(
      cudf::data_type(tid), 0, cudf::mask_state::UNALLOCATED, stream, mr);
  };
  if (n == 0) {
    auto dict_col = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT16),
                                                  kAlpRdMaxDictSize,
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
    cudaMemsetAsync(dict_col->mutable_view().data<uint16_t>(),
                    0,
                    sizeof(uint16_t) * kAlpRdMaxDictSize,
                    stream.value());
    auto md_col = cudf::make_fixed_width_column(
      cudf::data_type(cudf::type_id::UINT8), 1, cudf::mask_state::UNALLOCATED, stream, mr);
    uint8_t default_right_bw =
      static_cast<uint8_t>((traits::right_bw_min + traits::right_bw_max) / 2);
    cudaMemcpyAsync(md_col->mutable_view().data<uint8_t>(),
                    &default_right_bw,
                    sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    cudaStreamSynchronize(stream.value());
    return std::make_unique<alp_rd_compressed_representation>(col.type(),
                                                              0,
                                                              default_right_bw,
                                                              make_empty(traits::uint_type_id),
                                                              make_empty(cudf::type_id::UINT8),
                                                              std::move(dict_col),
                                                              std::move(md_col),
                                                              make_empty(cudf::type_id::UINT16),
                                                              make_empty(cudf::type_id::INT32));
  }

  // Step 1: pull a sample to the host and build (right_bw, dict). Strided
  // sampling keeps the sample representative even if the column has
  // structural patterns. Sample size matches cwida/ALP (≤ 32 vectors).
  auto const n_sz = static_cast<size_t>(n);
  size_t sample_n =
    std::min<size_t>(n_sz, static_cast<size_t>(kAlpRdSampleVectors) * kAlpRdVectorSize);

  std::vector<uint_t> h_sample(sample_n);
  if (sample_n > 0) {
    // Pull every `stride`-th value via `cudaMemcpy2DAsync` — treats the device
    // column as a 2D array with row pitch = stride*sizeof(T), row width =
    // sizeof(T), and `sample_n` rows. The DMA engine touches at most
    // (sample_n-1)*stride + 1 source elements but only `sample_n * sizeof(T)`
    // bytes actually transfer to host.
    size_t stride        = std::max<size_t>(1, n_sz / sample_n);
    size_t safe_sample_n = sample_n;
    if (stride > 1 && safe_sample_n > 0) {
      size_t max_idx = (n_sz - 1) / stride;
      safe_sample_n  = std::min<size_t>(safe_sample_n, max_idx + 1);
    }
    throw_if_cuda_error(cudaMemcpy2DAsync(h_sample.data(),
                                          sizeof(T),
                                          col.data<T>(),
                                          stride * sizeof(T),
                                          sizeof(T),
                                          safe_sample_n,
                                          cudaMemcpyDeviceToHost,
                                          stream.value()),
                        "alp_rd sample copy (2D)");
    cudaStreamSynchronize(stream.value());
    if (safe_sample_n < sample_n) {
      uint_t pad = safe_sample_n > 0 ? h_sample[safe_sample_n - 1] : uint_t{0};
      for (size_t i = safe_sample_n; i < sample_n; ++i)
        h_sample[i] = pad;
    }
  }

  rd_choice choice = find_best_choice<T>(h_sample.data(), sample_n);

  // Step 2: allocate output buffers + upload dict / metadata.
  auto right_parts_col = cudf::make_fixed_width_column(
    cudf::data_type(traits::uint_type_id), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto dict_indices_col = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::UINT8), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto dict_col     = cudf::make_fixed_width_column(cudf::data_type(cudf::type_id::UINT16),
                                                kAlpRdMaxDictSize,
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
  auto metadata_col = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::UINT8), 1, cudf::mask_state::UNALLOCATED, stream, mr);

  cudaMemcpyAsync(dict_col->mutable_view().data<uint16_t>(),
                  choice.dict,
                  sizeof(uint16_t) * kAlpRdMaxDictSize,
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaMemcpyAsync(metadata_col->mutable_view().data<uint8_t>(),
                  &choice.right_bw,
                  sizeof(uint8_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Step 3: encode kernel → (right_parts, dict_indices, exception_left, flag).
  rmm::device_uvector<uint8_t> d_flags(n, stream, mr);
  rmm::device_uvector<uint16_t> d_exception_left(n, stream, mr);

  const int block = 256;
  int grid        = (n + block - 1) / block;
  alp_rd_encode_kernel<T>
    <<<grid, block, 0, stream.value()>>>(reinterpret_cast<const uint_t*>(col.data<T>()),
                                         n,
                                         choice.right_bw,
                                         choice.actual_dict_size,
                                         dict_col->view().data<uint16_t>(),
                                         right_parts_col->mutable_view().data<uint_t>(),
                                         dict_indices_col->mutable_view().data<uint8_t>(),
                                         d_exception_left.data(),
                                         d_flags.data());

  // Step 4: compact exceptions into (positions, rejected left parts).
  auto exc = compact_exceptions<uint16_t>(
    d_flags.data(), n, d_exception_left.data(), cudf::type_id::UINT16, stream, mr);

  throw_if_cuda_error(cudaStreamSynchronize(stream.value()), "alp_rd_compress_impl sync");

  return std::make_unique<alp_rd_compressed_representation>(col.type(),
                                                            n,
                                                            choice.right_bw,
                                                            std::move(right_parts_col),
                                                            std::move(dict_indices_col),
                                                            std::move(dict_col),
                                                            std::move(metadata_col),
                                                            std::move(exc.values),
                                                            std::move(exc.positions));
}

template <typename T>
std::unique_ptr<cudf::column> alp_rd_decompress_impl(alp_rd_compressed_representation const& repr,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  using traits = alp_rd_traits<T>;
  using uint_t = typename traits::uint_t;

  if (repr.num_rows == 0) {
    return cudf::make_fixed_width_column(
      repr.original_type, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto out = cudf::make_fixed_width_column(
    repr.original_type, repr.num_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  const int block = 256;
  int grid        = (repr.num_rows + block - 1) / block;
  alp_rd_decode_kernel<T>
    <<<grid, block, 0, stream.value()>>>(repr.right_parts()->view().data<uint_t>(),
                                         repr.dict_indices()->view().data<uint8_t>(),
                                         repr.dict()->view().data<uint16_t>(),
                                         repr.right_bw,
                                         repr.num_rows,
                                         out->mutable_view().data<T>());

  cudf::size_type exc_n = repr.exceptions() ? repr.exceptions()->size() : 0;
  if (exc_n > 0) {
    int egrid = (exc_n + block - 1) / block;
    alp_rd_scatter_exceptions_kernel<T>
      <<<egrid, block, 0, stream.value()>>>(repr.exceptions()->view().data<uint16_t>(),
                                            repr.exception_positions()->view().data<int32_t>(),
                                            repr.right_parts()->view().data<uint_t>(),
                                            exc_n,
                                            repr.right_bw,
                                            out->mutable_view().data<T>());
  }

  throw_if_cuda_error(cudaStreamSynchronize(stream.value()), "alp_rd_decompress sync");
  return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// alp_rd_compressed_representation
// -----------------------------------------------------------------------------

alp_rd_compressed_representation::alp_rd_compressed_representation(
  cudf::data_type type,
  cudf::size_type n_rows,
  uint8_t right_bw_in,
  std::unique_ptr<cudf::column> right_parts_in,
  std::unique_ptr<cudf::column> dict_indices_in,
  std::unique_ptr<cudf::column> dict_in,
  std::unique_ptr<cudf::column> metadata_in,
  std::unique_ptr<cudf::column> exceptions_in,
  std::unique_ptr<cudf::column> exception_positions_in)
  : compressed_representation(type, n_rows), right_bw(right_bw_in)
{
  channels_.push_back(std::move(right_parts_in));
  channels_.push_back(std::move(dict_indices_in));
  channels_.push_back(std::move(dict_in));
  channels_.push_back(std::move(metadata_in));
  channels_.push_back(std::move(exceptions_in));
  channels_.push_back(std::move(exception_positions_in));
}

std::unique_ptr<cudf::column> alp_rd_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  switch (original_type.id()) {
    case cudf::type_id::FLOAT32: return alp_rd_decompress_impl<float>(*this, stream, mr);
    case cudf::type_id::FLOAT64: return alp_rd_decompress_impl<double>(*this, stream, mr);
    default:
      throw std::runtime_error("alp_rd: only FLOAT32 / FLOAT64 are supported (got " +
                               type_id_to_name(original_type) + ")");
  }
}

// -----------------------------------------------------------------------------
// alp_rd_compressor
// -----------------------------------------------------------------------------

std::unique_ptr<compressed_representation> alp_rd_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const dt = column_to_compress.type();
  switch (dt.id()) {
    case cudf::type_id::FLOAT32: return alp_rd_compress_impl<float>(column_to_compress, stream, mr);
    case cudf::type_id::FLOAT64:
      return alp_rd_compress_impl<double>(column_to_compress, stream, mr);
    default:
      throw std::runtime_error("alp_rd: only FLOAT32 / FLOAT64 are supported (got " +
                               type_id_to_name(dt) + ")");
  }
}

std::unique_ptr<cudf::column> alp_rd_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* repr = dynamic_cast<alp_rd_compressed_representation const*>(&data_to_decompress);
  if (repr == nullptr) return nullptr;
  return repr->decompress(stream, mr);
}

}  // namespace simpatico
