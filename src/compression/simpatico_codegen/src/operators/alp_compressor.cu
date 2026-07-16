// ALP (Adaptive Lossless Floating-Point) compressor — FLOAT32 and FLOAT64.
// Refs: SIGMOD '24 (Afroozeh, Kuffo, Boncz) + G-ALP DaMoN '25 (data-parallel
// exception scatter, branch-free decode).
//
// Operator surface (per-type column outputs):
//   input -> alp -> integers, exceptions, exception_positions, metadata
//   integers             INT32  (f32 input) / INT64 (f64 input) — main payload
//   exceptions           FLOAT32 / FLOAT64 — raw values that failed lossless encode
//   exception_positions  INT32  — global row indices of exceptions
//   metadata             UINT16 — one per 1024-vector: (exp_idx << 8) | fac_idx
//
// Decode: v[i] = integers[i] * 10^f / 10^e; then scatter exceptions.
//
// Implementation note: the kernels and host orchestrator are templated on
// the float type T; per-type constants live in __constant__ symbols selected
// via the alp_traits<T> accessor. Both instantiations share the same code
// path, so kernel improvements apply to both precisions automatically.

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

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>

namespace simpatico {

namespace {

// Vector size is fixed at 1024 (matches ALP & FastLanes; aligned with G-ALP).
constexpr int kAlpVectorSize = 1024;

// -----------------------------------------------------------------------------
// Per-type constant tables. Two separate __constant__ symbol sets because CUDA
// does not allow __constant__ arrays inside templates with proper linkage.
// alp_traits<T> selects between them via the static accessor methods below.
// -----------------------------------------------------------------------------

namespace host_consts_f32 {
// FLOAT32: e ∈ [0..10], f ∈ [0..min(e, 9)]. FACT has 10 entries because
// 10^10 doesn't fit in int32.
constexpr float kExp[11]    = {1.0f,
                               10.0f,
                               100.0f,
                               1000.0f,
                               10000.0f,
                               100000.0f,
                               1000000.0f,
                               10000000.0f,
                               100000000.0f,
                               1000000000.0f,
                               10000000000.0f};
constexpr float kFrac[11]   = {1.0f,
                               0.1f,
                               0.01f,
                               0.001f,
                               0.0001f,
                               0.00001f,
                               0.000001f,
                               0.0000001f,
                               0.00000001f,
                               0.000000001f,
                               0.0000000001f};
constexpr int32_t kFact[10] = {
  1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
constexpr int kMaxExp  = 10;
constexpr int kMaxFrac = 9;
// 1+2+...+10 + 10 = 65 combos.
constexpr int kComboCount = 65;
}  // namespace host_consts_f32

namespace host_consts_f64 {
// FLOAT64: e ∈ [0..18], f ∈ [0..min(e, 18)]. FACT goes up to 10^18 (fits in
// int64; max int64 ≈ 9.22e18).
constexpr double kExp[19]   = {1e0,
                               1e1,
                               1e2,
                               1e3,
                               1e4,
                               1e5,
                               1e6,
                               1e7,
                               1e8,
                               1e9,
                               1e10,
                               1e11,
                               1e12,
                               1e13,
                               1e14,
                               1e15,
                               1e16,
                               1e17,
                               1e18};
constexpr double kFrac[19]  = {1e-0,
                               1e-1,
                               1e-2,
                               1e-3,
                               1e-4,
                               1e-5,
                               1e-6,
                               1e-7,
                               1e-8,
                               1e-9,
                               1e-10,
                               1e-11,
                               1e-12,
                               1e-13,
                               1e-14,
                               1e-15,
                               1e-16,
                               1e-17,
                               1e-18};
constexpr int64_t kFact[19] = {1LL,
                               10LL,
                               100LL,
                               1000LL,
                               10000LL,
                               100000LL,
                               1000000LL,
                               10000000LL,
                               100000000LL,
                               1000000000LL,
                               10000000000LL,
                               100000000000LL,
                               1000000000000LL,
                               10000000000000LL,
                               100000000000000LL,
                               1000000000000000LL,
                               10000000000000000LL,
                               100000000000000000LL,
                               1000000000000000000LL};
constexpr int kMaxExp       = 18;
constexpr int kMaxFrac      = 18;
// 1+2+...+19 = 190 combos.
constexpr int kComboCount = 190;
}  // namespace host_consts_f64

__constant__ float d_alp_exp_f32[11];
__constant__ float d_alp_frac_f32[11];
__constant__ int32_t d_alp_fact_f32[10];
__constant__ uint8_t d_alp_combos_f32[host_consts_f32::kComboCount];

__constant__ double d_alp_exp_f64[19];
__constant__ double d_alp_frac_f64[19];
__constant__ int64_t d_alp_fact_f64[19];
__constant__ uint8_t d_alp_combos_f64[host_consts_f64::kComboCount];

// One-shot initialisation of both __constant__ table sets. Idempotent.
// Uploads run async on the caller's stream and are bounded by a single
// stream sync, avoiding the legacy default stream entirely.
void alp_upload_constants(rmm::cuda_stream_view stream)
{
  // Build packed combo tables on the host then upload.
  uint8_t combos_f32[host_consts_f32::kComboCount];
  {
    int idx = 0;
    for (uint8_t e = 0; e <= host_consts_f32::kMaxExp; ++e) {
      uint8_t f_max = (e <= host_consts_f32::kMaxFrac) ? e : host_consts_f32::kMaxFrac;
      for (uint8_t f = 0; f <= f_max; ++f) {
        combos_f32[idx++] = static_cast<uint8_t>((e << 4) | f);
      }
    }
    if (idx != host_consts_f32::kComboCount) {
      throw std::runtime_error("alp: f32 combo table size mismatch");
    }
  }
  uint8_t combos_f64[host_consts_f64::kComboCount];
  {
    int idx = 0;
    for (uint8_t e = 0; e <= host_consts_f64::kMaxExp; ++e) {
      uint8_t f_max = (e <= host_consts_f64::kMaxFrac) ? e : host_consts_f64::kMaxFrac;
      for (uint8_t f = 0; f <= f_max; ++f) {
        // Pack (e << 5) | f. Both e and f are ≤ 18 < 32 so 5 bits each fit
        // in a uint8_t. f32 uses (e << 4) | f because both ≤ 10 there.
        combos_f64[idx++] = static_cast<uint8_t>((e << 5) | f);
      }
    }
    if (idx != host_consts_f64::kComboCount) {
      throw std::runtime_error("alp: f64 combo table size mismatch");
    }
  }

  auto const s = stream.value();
  cudaMemcpyToSymbolAsync(d_alp_exp_f32,
                          host_consts_f32::kExp,
                          sizeof(host_consts_f32::kExp),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  cudaMemcpyToSymbolAsync(d_alp_frac_f32,
                          host_consts_f32::kFrac,
                          sizeof(host_consts_f32::kFrac),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  cudaMemcpyToSymbolAsync(d_alp_fact_f32,
                          host_consts_f32::kFact,
                          sizeof(host_consts_f32::kFact),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  cudaMemcpyToSymbolAsync(
    d_alp_combos_f32, combos_f32, sizeof(combos_f32), 0, cudaMemcpyHostToDevice, s);

  cudaMemcpyToSymbolAsync(d_alp_exp_f64,
                          host_consts_f64::kExp,
                          sizeof(host_consts_f64::kExp),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  cudaMemcpyToSymbolAsync(d_alp_frac_f64,
                          host_consts_f64::kFrac,
                          sizeof(host_consts_f64::kFrac),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  cudaMemcpyToSymbolAsync(d_alp_fact_f64,
                          host_consts_f64::kFact,
                          sizeof(host_consts_f64::kFact),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  cudaMemcpyToSymbolAsync(
    d_alp_combos_f64, combos_f64, sizeof(combos_f64), 0, cudaMemcpyHostToDevice, s);

  // Constants must be visible before any kernel that reads them runs; the
  // upload is host-side one-time work so a bounded sync here is acceptable.
  cudaStreamSynchronize(s);
}

// Thread-safe one-shot init: safe under concurrent multi-stream compression.
void ensure_constants_initialized(rmm::cuda_stream_view stream)
{
  static std::once_flag flag;
  std::call_once(flag, alp_upload_constants, stream);
}

// -----------------------------------------------------------------------------
// alp_traits<T>: the bridge between templated kernels and per-type constants.
// -----------------------------------------------------------------------------

template <typename T>
struct alp_traits;

template <>
struct alp_traits<float> {
  using value_t                                = float;
  using int_t                                  = int32_t;
  using uint_t                                 = uint32_t;
  static constexpr cudf::type_id value_type_id = cudf::type_id::FLOAT32;
  static constexpr cudf::type_id int_type_id   = cudf::type_id::INT32;
  static constexpr int combo_count             = host_consts_f32::kComboCount;
  static constexpr value_t magic               = 12582912.0f;    // 2^23 + 2^22
  static constexpr value_t safe_max            = 2147483520.0f;  // ≈ INT32_MAX, f32-representable
  static constexpr value_t safe_min            = -2147483520.0f;
  // Hard min/max of the encoded integer type. We avoid std::numeric_limits in
  // device code (NVCC refuses without --expt-relaxed-constexpr).
  static constexpr int_t int_max = 0x7FFFFFFF;
  static constexpr int_t int_min = static_cast<int_t>(0x80000000);
  // Exception cost: payload (32) + position (16) bits per exception.
  static constexpr uint32_t exception_cost_bits = 32 + 16;
  __device__ static value_t exp_(int i) { return d_alp_exp_f32[i]; }
  __device__ static value_t frac_(int i) { return d_alp_frac_f32[i]; }
  __device__ static int_t fact_(int i) { return d_alp_fact_f32[i]; }
  __device__ static void unpack_combo(int c, uint8_t& e, uint8_t& f)
  {
    uint8_t packed = d_alp_combos_f32[c];
    e              = packed >> 4;
    f              = packed & 0xF;
  }
};

template <>
struct alp_traits<double> {
  using value_t                                = double;
  using int_t                                  = int64_t;
  using uint_t                                 = uint64_t;
  static constexpr cudf::type_id value_type_id = cudf::type_id::FLOAT64;
  static constexpr cudf::type_id int_type_id   = cudf::type_id::INT64;
  static constexpr int combo_count             = host_consts_f64::kComboCount;
  // 2^52 + 2^51. Adding/subtracting forces IEEE round-to-nearest-even into
  // the mantissa truncation, producing CPU-identical encoded integers.
  static constexpr value_t magic = 6755399441055744.0;
  // INT64_MAX = 9223372036854775807. We use a conservative bound below 2^63
  // that is exactly representable in f64 (= 2^63 - 1024).
  static constexpr value_t safe_max = 9223372036854774784.0;
  static constexpr value_t safe_min = -9223372036854774784.0;
  // Hard min/max of the encoded integer type. We avoid std::numeric_limits in
  // device code (NVCC refuses without --expt-relaxed-constexpr).
  static constexpr int_t int_max = 0x7FFFFFFFFFFFFFFFLL;
  static constexpr int_t int_min = static_cast<int_t>(0x8000000000000000ULL);
  // Exception cost: 64-bit payload + 16-bit position.
  static constexpr uint32_t exception_cost_bits = 64 + 16;
  __device__ static value_t exp_(int i) { return d_alp_exp_f64[i]; }
  __device__ static value_t frac_(int i) { return d_alp_frac_f64[i]; }
  __device__ static int_t fact_(int i) { return d_alp_fact_f64[i]; }
  __device__ static void unpack_combo(int c, uint8_t& e, uint8_t& f)
  {
    uint8_t packed = d_alp_combos_f64[c];
    e              = packed >> 5;
    f              = packed & 0x1F;
  }
};

// -----------------------------------------------------------------------------
// Atomic min/max overloads. CUDA's atomicMin/atomicMax for 64-bit ints want
// `long long*`, which is not the same type as int64_t on every platform.
// -----------------------------------------------------------------------------

__device__ inline int32_t atomic_min(int32_t* a, int32_t v) { return atomicMin(a, v); }
__device__ inline int32_t atomic_max(int32_t* a, int32_t v) { return atomicMax(a, v); }
__device__ inline int64_t atomic_min(int64_t* a, int64_t v)
{
  return static_cast<int64_t>(
    atomicMin(reinterpret_cast<long long*>(a), static_cast<long long>(v)));
}
__device__ inline int64_t atomic_max(int64_t* a, int64_t v)
{
  return static_cast<int64_t>(
    atomicMax(reinterpret_cast<long long*>(a), static_cast<long long>(v)));
}

// -----------------------------------------------------------------------------
// Device helpers
// -----------------------------------------------------------------------------

__device__ inline int bits_for_range_u64(uint64_t range)
{
  if (range == 0) return 0;
  return 64 - __clzll(static_cast<unsigned long long>(range));
}

// Encode one value; returns the (templated) integer encoded value and sets
// `is_exception` when the round-trip check fails (or the input is non-finite,
// ±Inf, NaN, -0.0, or overflows the safe-integer range when scaled).
template <typename T>
__device__ inline typename alp_traits<T>::int_t alp_encode_value(T v,
                                                                 uint8_t e,
                                                                 uint8_t f,
                                                                 bool& is_exception)
{
  using traits = alp_traits<T>;
  using int_t  = typename traits::int_t;
  if (!isfinite(v) || (v == T{0} && signbit(v))) {
    is_exception = true;
    return 0;
  }
  T tmp = v * traits::exp_(e) * traits::frac_(f);
  // Magic-number round-to-nearest-even.
  T rounded = (tmp + traits::magic) - traits::magic;
  if (!isfinite(rounded) || rounded > traits::safe_max || rounded < traits::safe_min) {
    is_exception = true;
    return 0;
  }
  int_t enc = static_cast<int_t>(rounded);
  // Round-trip check: decode and compare bit-exactly. The cast-back-to-T after
  // the integer multiply matches CPU ALP (it forces single/double rounding to
  // happen in the same order on both sides).
  T decoded    = static_cast<T>(enc * traits::fact_(f)) * traits::frac_(e);
  is_exception = (decoded != v);
  return enc;
}

// -----------------------------------------------------------------------------
// Encode kernel: 1 block == 1 vector of 1024 values. blockDim.x == 1024.
// Each thread handles one value. For each candidate (e, f) the threads
// atomically update per-combo (exc_count, min_enc, max_enc) in shared mem;
// thread 0 then picks the combo with the lowest cost
// (`vec_n * bitwidth + exc_count * exception_cost_bits`) and stores the
// chosen (e, f) into the metadata column. Finally every thread re-encodes
// its value with the winning (e, f) and writes (integer, exception_flag).
// -----------------------------------------------------------------------------
template <typename T>
__global__ void alp_encode_kernel(const T* __restrict__ in,
                                  int32_t n_rows,
                                  typename alp_traits<T>::int_t* __restrict__ out_integers,
                                  uint16_t* __restrict__ out_metadata,
                                  uint8_t* __restrict__ out_exception_flag)
{
  using traits              = alp_traits<T>;
  using int_t               = typename traits::int_t;
  constexpr int combo_count = traits::combo_count;

  int vec      = blockIdx.x;
  int tid      = threadIdx.x;
  int global_i = vec * kAlpVectorSize + tid;
  bool valid   = (global_i < n_rows);
  T v          = valid ? in[global_i] : T{0};

  int vec_n = min(n_rows - vec * kAlpVectorSize, kAlpVectorSize);

  __shared__ uint32_t s_exc_count[combo_count];
  __shared__ int_t s_min_enc[combo_count];
  __shared__ int_t s_max_enc[combo_count];
  __shared__ uint16_t s_best_md;  // (e << 8) | f

  // Init the per-combo accumulators. combo_count ≤ 1024 so the first
  // `combo_count` threads cover the init in one pass.
  if (tid < combo_count) {
    s_exc_count[tid] = 0u;
    s_min_enc[tid]   = traits::int_max;
    s_max_enc[tid]   = traits::int_min;
  }
  __syncthreads();

  // Sweep candidates. Atomic contention is per-combo across the block; for
  // typical inputs most threads hit the same min/max paths so this is
  // dominated by atomic latency, not contention.
  for (int c = 0; c < combo_count; ++c) {
    uint8_t e, f;
    traits::unpack_combo(c, e, f);

    if (valid) {
      bool is_exc;
      int_t enc = alp_encode_value<T>(v, e, f, is_exc);
      if (is_exc) {
        atomicAdd(&s_exc_count[c], 1u);
      } else {
        atomic_min(&s_min_enc[c], enc);
        atomic_max(&s_max_enc[c], enc);
      }
    }
  }
  __syncthreads();

  // Single-thread cost-based selection. combo_count is small; full pass is fine.
  if (tid == 0) {
    uint64_t best_cost = UINT64_MAX;
    int best           = 0;
    for (int c = 0; c < combo_count; ++c) {
      uint32_t exc     = s_exc_count[c];
      uint32_t non_exc = static_cast<uint32_t>(vec_n) - exc;
      uint32_t bits    = 0;
      if (non_exc > 0) {
        int_t lo = s_min_enc[c];
        int_t hi = s_max_enc[c];
        if (hi >= lo) {
          // Unsigned-subtract trick gives the correct range as a uint64
          // regardless of int_t width: two's complement bit patterns wrap
          // modulo 2^64, and (hi >= lo) guarantees the result is positive.
          uint64_t range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
          bits           = static_cast<uint32_t>(bits_for_range_u64(range));
        }
      }
      uint64_t cost = static_cast<uint64_t>(vec_n) * bits +
                      static_cast<uint64_t>(exc) * traits::exception_cost_bits;
      if (cost < best_cost) {
        best_cost = cost;
        best      = c;
      }
    }
    uint8_t best_e, best_f;
    traits::unpack_combo(best, best_e, best_f);
    s_best_md         = (static_cast<uint16_t>(best_e) << 8) | best_f;
    out_metadata[vec] = s_best_md;
  }
  __syncthreads();

  // Re-encode with the chosen (e, f). For exception positions we write 0 to
  // `integers` so downstream bit-packing on the chunk stays tight.
  if (valid) {
    uint8_t best_e = static_cast<uint8_t>(s_best_md >> 8);
    uint8_t best_f = static_cast<uint8_t>(s_best_md & 0xFF);
    bool is_exc;
    int_t enc                    = alp_encode_value<T>(v, best_e, best_f, is_exc);
    out_integers[global_i]       = is_exc ? int_t{0} : enc;
    out_exception_flag[global_i] = is_exc ? 1u : 0u;
  }
}

// -----------------------------------------------------------------------------
// Decode kernel: branch-free per-element multiply. Each thread handles one
// element. The exception scatter runs as a separate kernel afterwards.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void alp_decode_kernel(const typename alp_traits<T>::int_t* __restrict__ integers,
                                  const uint16_t* __restrict__ metadata,
                                  int32_t n_rows,
                                  T* __restrict__ out)
{
  using traits = alp_traits<T>;
  int i        = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows) return;
  int vec     = i / kAlpVectorSize;
  uint16_t md = metadata[vec];
  uint8_t e   = static_cast<uint8_t>(md >> 8);
  uint8_t f   = static_cast<uint8_t>(md & 0xFF);
  auto enc    = integers[i];
  // v = enc * 10^f / 10^e. The cast-to-T forces the integer multiply to land
  // in the target precision before the final scaling, matching CPU ALP.
  out[i] = static_cast<T>(enc * traits::fact_(f)) * traits::frac_(e);
}

// Exception scatter: fully data-parallel (G-ALP's key GPU optimisation).
template <typename T>
__global__ void alp_scatter_exceptions_kernel(const T* __restrict__ exceptions,
                                              const int32_t* __restrict__ positions,
                                              int32_t exc_count,
                                              T* __restrict__ out)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= exc_count) return;
  out[positions[k]] = exceptions[k];
}

// -----------------------------------------------------------------------------
// Host orchestration (compress + decompress) — templated on T = float|double.
// -----------------------------------------------------------------------------

template <typename T>
std::unique_ptr<alp_compressed_representation> alp_compress_impl(cudf::column_view const& col,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr)
{
  using traits = alp_traits<T>;
  using int_t  = typename traits::int_t;

  auto const n = col.size();
  if (n == 0) {
    auto empty_int = cudf::make_fixed_width_column(
      cudf::data_type(traits::int_type_id), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    auto empty_exc = cudf::make_fixed_width_column(
      cudf::data_type(traits::value_type_id), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    auto empty_pos = cudf::make_fixed_width_column(
      cudf::data_type(cudf::type_id::INT32), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    auto empty_md = cudf::make_fixed_width_column(
      cudf::data_type(cudf::type_id::UINT16), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    return std::make_unique<alp_compressed_representation>(col.type(),
                                                           0,
                                                           0,
                                                           std::move(empty_int),
                                                           std::move(empty_exc),
                                                           std::move(empty_pos),
                                                           std::move(empty_md));
  }

  ensure_constants_initialized(stream);

  cudf::size_type num_vectors =
    static_cast<cudf::size_type>((static_cast<int64_t>(n) + kAlpVectorSize - 1) / kAlpVectorSize);

  auto integers_col = cudf::make_fixed_width_column(
    cudf::data_type(traits::int_type_id), n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto metadata_col = cudf::make_fixed_width_column(
    cudf::data_type(cudf::type_id::UINT16), num_vectors, cudf::mask_state::UNALLOCATED, stream, mr);

  // Per-element exception flag buffer. Lives only inside compress().
  rmm::device_uvector<uint8_t> d_flags(n, stream, mr);

  alp_encode_kernel<T><<<num_vectors, kAlpVectorSize, 0, stream.value()>>>(
    col.data<T>(),
    n,
    integers_col->mutable_view().data<int_t>(),
    metadata_col->mutable_view().data<uint16_t>(),
    d_flags.data());

  // Compact the per-row exception flags into (positions, exception values).
  auto exc =
    compact_exceptions<T>(d_flags.data(), n, col.data<T>(), traits::value_type_id, stream, mr);

  throw_if_cuda_error(cudaStreamSynchronize(stream.value()), "alp_compress_impl sync");

  return std::make_unique<alp_compressed_representation>(col.type(),
                                                         n,
                                                         num_vectors,
                                                         std::move(integers_col),
                                                         std::move(exc.values),
                                                         std::move(exc.positions),
                                                         std::move(metadata_col));
}

template <typename T>
std::unique_ptr<cudf::column> alp_decompress_impl(alp_compressed_representation const& repr,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  using traits = alp_traits<T>;
  using int_t  = typename traits::int_t;

  if (repr.num_rows == 0) {
    return cudf::make_fixed_width_column(
      repr.original_type, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  ensure_constants_initialized(stream);

  auto out = cudf::make_fixed_width_column(
    repr.original_type, repr.num_rows, cudf::mask_state::UNALLOCATED, stream, mr);

  const int block = 256;
  int grid        = (repr.num_rows + block - 1) / block;
  alp_decode_kernel<T><<<grid, block, 0, stream.value()>>>(repr.integers()->view().data<int_t>(),
                                                           repr.metadata()->view().data<uint16_t>(),
                                                           repr.num_rows,
                                                           out->mutable_view().data<T>());

  cudf::size_type exc_n = repr.exceptions() ? repr.exceptions()->size() : 0;
  if (exc_n > 0) {
    int egrid = (exc_n + block - 1) / block;
    alp_scatter_exceptions_kernel<T>
      <<<egrid, block, 0, stream.value()>>>(repr.exceptions()->view().data<T>(),
                                            repr.exception_positions()->view().data<int32_t>(),
                                            exc_n,
                                            out->mutable_view().data<T>());
  }

  throw_if_cuda_error(cudaStreamSynchronize(stream.value()), "alp_decompress sync");
  return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// alp_compressed_representation
// -----------------------------------------------------------------------------

alp_compressed_representation::alp_compressed_representation(
  cudf::data_type type,
  cudf::size_type n_rows,
  cudf::size_type n_vectors,
  std::unique_ptr<cudf::column> integers_in,
  std::unique_ptr<cudf::column> exceptions_in,
  std::unique_ptr<cudf::column> exception_positions_in,
  std::unique_ptr<cudf::column> metadata_in)
  : compressed_representation(type, n_rows), num_vectors(n_vectors)
{
  channels_.push_back(std::move(integers_in));
  channels_.push_back(std::move(exceptions_in));
  channels_.push_back(std::move(exception_positions_in));
  channels_.push_back(std::move(metadata_in));
}

std::unique_ptr<cudf::column> alp_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  switch (original_type.id()) {
    case cudf::type_id::FLOAT32: return alp_decompress_impl<float>(*this, stream, mr);
    case cudf::type_id::FLOAT64: return alp_decompress_impl<double>(*this, stream, mr);
    default:
      throw std::runtime_error("alp: only FLOAT32 / FLOAT64 are supported (got " +
                               type_id_to_name(original_type) + ")");
  }
}

// -----------------------------------------------------------------------------
// alp_compressor
// -----------------------------------------------------------------------------

std::unique_ptr<compressed_representation> alp_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const dt = column_to_compress.type();
  switch (dt.id()) {
    case cudf::type_id::FLOAT32: return alp_compress_impl<float>(column_to_compress, stream, mr);
    case cudf::type_id::FLOAT64: return alp_compress_impl<double>(column_to_compress, stream, mr);
    default:
      throw std::runtime_error("alp: only FLOAT32 / FLOAT64 are supported (got " +
                               type_id_to_name(dt) + "). Use alp_rd for non-decimal floats.");
  }
}

std::unique_ptr<cudf::column> alp_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* repr = dynamic_cast<alp_compressed_representation const*>(&data_to_decompress);
  if (repr == nullptr) return nullptr;
  return repr->decompress(stream, mr);
}

}  // namespace simpatico
