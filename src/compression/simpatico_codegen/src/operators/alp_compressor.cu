// ALP (Adaptive Lossless Floating-Point) compressor — FLOAT32 and FLOAT64.
// Refs: SIGMOD '24 (Afroozeh, Kuffo, Boncz) + G-ALP DaMoN '25 (data-parallel
// exception scatter, branch-free decode).
//
// Operator surface (per-type column outputs):
//   input -> alp -> integers, exceptions, exception_positions, metadata
//   integers             INT32  (f32 input) / INT64 (f64 input) — main payload
//   exceptions           FLOAT32 / FLOAT64 — raw values that failed lossless encode
//   exception_positions  INT32  — global row indices of exceptions
//   metadata             UINT16 — one per 1024-vector: the scale exponent d
//
// Decode: v[i] = integers[i] * 10^-d; then scatter exceptions.
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

#include <cuda/std/type_traits>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace simpatico {

namespace {

// Vector size is fixed at 1024 (matches ALP & FastLanes; aligned with G-ALP).
constexpr int kAlpVectorSize = 1024;

// Scale-selection sample: contiguous runs spread evenly across the vector. The
// total must be a whole number of warps and no larger than the vector.
constexpr int kAlpSampleRuns   = 8;
constexpr int kAlpSampleRunLen = 8;
constexpr int kAlpSampleSize   = kAlpSampleRuns * kAlpSampleRunLen;  // 64 = 2 warps
static_assert(kAlpSampleSize % 32 == 0, "sample must be a whole number of warps");
static_assert(kAlpSampleSize <= kAlpVectorSize, "sample must fit in a vector");

// -----------------------------------------------------------------------------
// Per-type constant tables. Two separate __constant__ symbol sets because CUDA
// does not allow __constant__ arrays inside templates with proper linkage.
// alp_traits<T> selects between them via the static accessor methods below.
// -----------------------------------------------------------------------------

// Every 10^k below is exactly representable in its float type. Never scale by a
// rounded reciprocal instead: `enc * 0.01` and `enc / 100.0` differ by an ulp,
// and the encoder's bit-exact round-trip check turns that ulp into an exception.
namespace host_consts_f32 {
// FLOAT32: e ∈ [0..10], f ∈ [0..min(e, 9)].
constexpr float kExp[11] = {1.0f,
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
// Candidate scales are 10^d for d in [0..10]; see alp_encode_value for why the
// paper's (e, f) pair collapses to their difference.
constexpr int kCandCount = 11;
}  // namespace host_consts_f32

namespace host_consts_f64 {
// FLOAT64: e ∈ [0..18], f ∈ [0..min(e, 18)].
constexpr double kExp[19] = {1e0,
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
// Candidate scales are 10^d for d in [0..18].
constexpr int kCandCount = 19;
}  // namespace host_consts_f64

// Fixed-point path: a DECIMAL column's storage is already an integer mantissa,
// so its scale search is integer division with an exactness check.
constexpr int kP10I32Count = 10;  // 10^9 is the largest that fits int32
constexpr int kP10I64Count = 19;  // 10^18 is the largest that fits int64
__constant__ int32_t d_alp_p10_i32[kP10I32Count];
__constant__ int64_t d_alp_p10_i64[kP10I64Count];

__constant__ float d_alp_exp_f32[11];
__constant__ float d_alp_rhi_f32[11];
__constant__ float d_alp_rlo_f32[11];

__constant__ double d_alp_exp_f64[19];
__constant__ double d_alp_rhi_f64[19];
__constant__ double d_alp_rlo_f64[19];

// Unevaluated-sum split of 10^-k: rhi + rlo carries roughly twice the working
// precision, so fma(v, rhi, v * rlo) matches a divide by 10^k at two
// multiply-class ops (FP64 divide is punishing on parts with cut FP64). This
// only has to be accurate, not exact -- encode's round-trip check evaluates the
// identical expression, so what it cannot reproduce becomes an exception.
// Accuracy buys ratio, not correctness.
template <typename Wide, typename Narrow>
void fill_reciprocal_split(Narrow* rhi, Narrow* rlo, int count)
{
  Wide p = Wide{1};
  for (int k = 0; k < count; ++k) {
    Wide const r = Wide{1} / p;
    rhi[k]       = static_cast<Narrow>(r);
    rlo[k]       = static_cast<Narrow>(r - static_cast<Wide>(rhi[k]));
    p *= Wide{10};
  }
}

// One-shot initialisation of both __constant__ table sets. Idempotent.
// Uploads run async on the caller's stream and are bounded by a single
// stream sync, avoiding the legacy default stream entirely.
void alp_upload_constants(rmm::cuda_stream_view stream)
{
  int32_t p10_i32[kP10I32Count];
  int64_t p10_i64[kP10I64Count];
  {
    int32_t p32 = 1;
    for (int k = 0; k < kP10I32Count; ++k, p32 *= 10)
      p10_i32[k] = p32;
    int64_t p64 = 1;
    for (int k = 0; k < kP10I64Count; ++k, p64 *= 10)
      p10_i64[k] = p64;
  }

  auto const s = stream.value();
  cudaMemcpyToSymbolAsync(d_alp_p10_i32, p10_i32, sizeof(p10_i32), 0, cudaMemcpyHostToDevice, s);
  cudaMemcpyToSymbolAsync(d_alp_p10_i64, p10_i64, sizeof(p10_i64), 0, cudaMemcpyHostToDevice, s);
  cudaMemcpyToSymbolAsync(d_alp_exp_f32,
                          host_consts_f32::kExp,
                          sizeof(host_consts_f32::kExp),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  float rhi_f32[11], rlo_f32[11];
  fill_reciprocal_split<double, float>(rhi_f32, rlo_f32, 11);
  cudaMemcpyToSymbolAsync(d_alp_rhi_f32, rhi_f32, sizeof(rhi_f32), 0, cudaMemcpyHostToDevice, s);
  cudaMemcpyToSymbolAsync(d_alp_rlo_f32, rlo_f32, sizeof(rlo_f32), 0, cudaMemcpyHostToDevice, s);

  cudaMemcpyToSymbolAsync(d_alp_exp_f64,
                          host_consts_f64::kExp,
                          sizeof(host_consts_f64::kExp),
                          0,
                          cudaMemcpyHostToDevice,
                          s);
  double rhi_f64[19], rlo_f64[19];
  fill_reciprocal_split<long double, double>(rhi_f64, rlo_f64, 19);
  cudaMemcpyToSymbolAsync(d_alp_rhi_f64, rhi_f64, sizeof(rhi_f64), 0, cudaMemcpyHostToDevice, s);
  cudaMemcpyToSymbolAsync(d_alp_rlo_f64, rlo_f64, sizeof(rlo_f64), 0, cudaMemcpyHostToDevice, s);

  // Constants must be visible before any kernel that reads them runs; the
  // upload is host-side one-time work so a bounded sync here is acceptable.
  cudaStreamSynchronize(s);
}

// Thread-safe init: uploads the constant tables once PER DEVICE. The __constant__
// symbols are device-local (cudaMemcpyToSymbol targets the current device), so a
// process that touches more than one GPU -- e.g. a chunk encoded on GPU 0 and
// host-staged, then decoded on GPU 1 -- must upload to each device separately. A
// single process-wide once_flag would leave every device but the first with
// uninitialized constants, silently decoding garbage.
void ensure_constants_initialized(rmm::cuda_stream_view stream)
{
  int device = 0;
  throw_if_cuda_error(cudaGetDevice(&device), "alp: cudaGetDevice");

  // Keyed on the current device; the uploads below target its constant memory.
  static std::mutex mtx;
  static std::set<int> initialized_devices;

  std::lock_guard<std::mutex> const lock(mtx);
  if (initialized_devices.count(device) > 0) return;
  alp_upload_constants(stream);
  initialized_devices.insert(device);
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
  static constexpr int cand_count              = host_consts_f32::kCandCount;
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
  __device__ static value_t rhi_(int i) { return d_alp_rhi_f32[i]; }
  __device__ static value_t rlo_(int i) { return d_alp_rlo_f32[i]; }
};

template <>
struct alp_traits<double> {
  using value_t                                = double;
  using int_t                                  = int64_t;
  using uint_t                                 = uint64_t;
  static constexpr cudf::type_id value_type_id = cudf::type_id::FLOAT64;
  static constexpr cudf::type_id int_type_id   = cudf::type_id::INT64;
  static constexpr int cand_count              = host_consts_f64::kCandCount;
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
  __device__ static value_t rhi_(int i) { return d_alp_rhi_f64[i]; }
  __device__ static value_t rlo_(int i) { return d_alp_rlo_f64[i]; }
};

// Fixed-point specialisations; `value_t == int_t`. `value_type_id` is only the
// storage id -- compress passes the column's real data_type, carrying its scale.
template <>
struct alp_traits<int32_t> {
  using value_t                                 = int32_t;
  using int_t                                   = int32_t;
  using uint_t                                  = uint32_t;
  static constexpr cudf::type_id value_type_id  = cudf::type_id::DECIMAL32;
  static constexpr cudf::type_id int_type_id    = cudf::type_id::INT32;
  static constexpr int cand_count               = kP10I32Count;
  static constexpr int_t int_max                = 0x7FFFFFFF;
  static constexpr int_t int_min                = static_cast<int_t>(0x80000000);
  static constexpr uint32_t exception_cost_bits = 32 + 16;
  __device__ static int_t p10_(int i) { return d_alp_p10_i32[i]; }
};

template <>
struct alp_traits<int64_t> {
  using value_t                                 = int64_t;
  using int_t                                   = int64_t;
  using uint_t                                  = uint64_t;
  static constexpr cudf::type_id value_type_id  = cudf::type_id::DECIMAL64;
  static constexpr cudf::type_id int_type_id    = cudf::type_id::INT64;
  static constexpr int cand_count               = kP10I64Count;
  static constexpr int_t int_max                = 0x7FFFFFFFFFFFFFFFLL;
  static constexpr int_t int_min                = static_cast<int_t>(0x8000000000000000ULL);
  static constexpr uint32_t exception_cost_bits = 64 + 16;
  __device__ static int_t p10_(int i) { return d_alp_p10_i64[i]; }
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
// Full-warp shuffle reductions; callers must have all 32 lanes active.
// -----------------------------------------------------------------------------

template <typename V>
__device__ inline V warp_reduce_min(V x)
{
  for (int off = 16; off > 0; off >>= 1) {
    V const y = __shfl_down_sync(0xFFFFFFFFu, x, off);
    x         = (y < x) ? y : x;
  }
  return x;
}

template <typename V>
__device__ inline V warp_reduce_max(V x)
{
  for (int off = 16; off > 0; off >>= 1) {
    V const y = __shfl_down_sync(0xFFFFFFFFu, x, off);
    x         = (y > x) ? y : x;
  }
  return x;
}

__device__ inline uint32_t warp_reduce_add(uint32_t x)
{
  for (int off = 16; off > 0; off >>= 1)
    x += __shfl_down_sync(0xFFFFFFFFu, x, off);
  return x;
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
__device__ inline T alp_decode_value(typename alp_traits<T>::int_t enc, int d);

template <typename T>
__device__ inline typename alp_traits<T>::int_t alp_encode_value(T v, int d, bool& is_exception)
{
  using traits = alp_traits<T>;
  using int_t  = typename traits::int_t;

  if constexpr (cuda::std::is_integral_v<T>) {
    // |q * p| <= |v| by construction, so the check cannot overflow, and
    // truncation toward zero behaves the same on both signs.
    int_t const p = traits::p10_(d);
    int_t const q = v / p;
    is_exception  = (q * p != v);
    return q;
  } else {
    if (!isfinite(v) || (v == T{0} && signbit(v))) {
      is_exception = true;
      return 0;
    }
    // Published ALP encodes round(v * 10^e * 10^-f), but only d = e - f affects
    // the result and its combo table constrains f <= e, so sweeping d covers the
    // same ground in 19 scales rather than 190 pairs.
    T tmp = v * traits::exp_(d);
    // Magic-number round-to-nearest-even.
    T rounded = (tmp + traits::magic) - traits::magic;
    if (!isfinite(rounded) || rounded > traits::safe_max || rounded < traits::safe_min) {
      is_exception = true;
      return 0;
    }
    int_t enc = static_cast<int_t>(rounded);
    // Round-trip check: decode with the EXACT same expression as
    // alp_decode_kernel and compare bit-exactly.
    is_exception = (alp_decode_value<T>(enc, d) != v);
    return enc;
  }
}

// Inverse of alp_encode_value. One function so encode's round-trip check and
// the decode kernel cannot drift apart.
template <typename T>
__device__ inline T alp_decode_value(typename alp_traits<T>::int_t enc, int d)
{
  using traits = alp_traits<T>;
  if constexpr (cuda::std::is_integral_v<T>) {
    return enc * traits::p10_(d);
  } else {
    T const encd = static_cast<T>(enc);
    return fma(encd, traits::rhi_(d), encd * traits::rlo_(d));
  }
}

// -----------------------------------------------------------------------------
// Encode kernel: 1 block == 1 vector of 1024 values. blockDim.x == 1024.
// Selection: the first kAlpSampleSize threads score every candidate scale d on
// one sampled value each, warp-reducing per-candidate (exc_count, min_enc,
// max_enc) before a single shared atomic; thread 0 picks the lowest-cost d
// (`sample_n * bitwidth + exc_count * exception_cost_bits`) into the metadata
// column. Emit: every thread re-encodes its own value with the winning d, so
// sampling costs ratio (a worse d), never correctness.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void alp_encode_kernel(const T* __restrict__ in,
                                  int32_t n_rows,
                                  typename alp_traits<T>::int_t* __restrict__ out_integers,
                                  uint16_t* __restrict__ out_metadata,
                                  uint8_t* __restrict__ out_exception_flag)
{
  using traits             = alp_traits<T>;
  using int_t              = typename traits::int_t;
  constexpr int cand_count = traits::cand_count;

  int vec      = blockIdx.x;
  int tid      = threadIdx.x;
  int vec_base = vec * kAlpVectorSize;
  int global_i = vec_base + tid;
  bool valid   = (global_i < n_rows);
  T v          = valid ? in[global_i] : T{0};

  int vec_n = min(n_rows - vec_base, kAlpVectorSize);

  __shared__ uint32_t s_exc_count[cand_count];
  __shared__ int_t s_min_enc[cand_count];
  __shared__ int_t s_max_enc[cand_count];
  __shared__ uint16_t s_best_d;  // the winning scale exponent, stored verbatim
  __shared__ int_t s_fill;       // value written at exception slots

  // cand_count <= 1024, so the first cand_count threads init in one pass.
  if (tid < cand_count) {
    s_exc_count[tid] = 0u;
    s_min_enc[tid]   = traits::int_max;
    s_max_enc[tid]   = traits::int_min;
  }
  __syncthreads();

  // Contiguous runs, not a fixed stride: a stride can land on a period of the
  // data and observe only one phase of it. A short trailing vector collapses
  // runs onto each other, which merely resamples -- harmless for a ranking.
  bool const samples = (tid < kAlpSampleSize);
  T sv               = T{0};
  bool sv_valid      = false;
  if (samples) {
    int const run = tid / kAlpSampleRunLen;
    int const off = tid % kAlpSampleRunLen;
    int const idx = (run * vec_n) / kAlpSampleRuns + off;
    sv_valid      = (idx < vec_n);
    if (sv_valid) sv = in[vec_base + idx];
  }

  // kAlpSampleSize is a whole number of warps, so every lane in a participating
  // warp is active and the full-mask shuffles below are well formed.
  if (samples) {
    int const lane = tid & 31;
    for (int c = 0; c < cand_count; ++c) {
      bool is_exc = true;
      int_t enc   = int_t{0};
      if (sv_valid) enc = alp_encode_value<T>(sv, c, is_exc);

      // Non-participating lanes fold in as reduction identities.
      uint32_t const exc_bit = (sv_valid && is_exc) ? 1u : 0u;
      bool const counts      = (sv_valid && !is_exc);
      int_t const lo_in      = counts ? enc : traits::int_max;
      int_t const hi_in      = counts ? enc : traits::int_min;

      uint32_t const exc_sum = warp_reduce_add(exc_bit);
      int_t const lo         = warp_reduce_min<int_t>(lo_in);
      int_t const hi         = warp_reduce_max<int_t>(hi_in);

      if (lane == 0) {
        if (exc_sum != 0u) atomicAdd(&s_exc_count[c], exc_sum);
        if (lo <= hi) {
          atomic_min(&s_min_enc[c], lo);
          atomic_max(&s_max_enc[c], hi);
        }
      }
    }
  }
  __syncthreads();

  // Single-thread cost-based selection. cand_count is small; full pass is fine.
  if (tid == 0) {
    // sample_n, not vec_n: both cost terms must share the denominator the
    // exception counts were drawn from.
    uint32_t const sample_n = static_cast<uint32_t>(min(vec_n, kAlpSampleSize));
    uint64_t best_cost      = UINT64_MAX;
    int best                = 0;
    for (int c = 0; c < cand_count; ++c) {
      uint32_t exc     = s_exc_count[c];
      uint32_t non_exc = sample_n - exc;
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
      uint64_t cost = static_cast<uint64_t>(sample_n) * bits +
                      static_cast<uint64_t>(exc) * traits::exception_cost_bits;
      if (cost < best_cost) {
        best_cost = cost;
        best      = c;
      }
    }
    s_best_d          = static_cast<uint16_t>(best);
    out_metadata[vec] = s_best_d;
    // Exception slots take the winning candidate's minimum encoded value -- the
    // frame of reference a downstream bitpack subtracts. Writing 0 instead would
    // stretch the chunk's range whenever the real values cluster away from it.
    // A sample minimum can sit above the true one but never below, so it always
    // lands inside bitpack's range. int_max sentinel survives when every sampled
    // value is an exception; 0 is as good as anything there.
    s_fill = (s_exc_count[best] < sample_n) ? s_min_enc[best] : int_t{0};
  }
  __syncthreads();

  // Exception slots carry the fill value; their real payload lives in the
  // `exceptions` channel and is scattered back over them on decode.
  if (valid) {
    bool is_exc;
    int_t enc                    = alp_encode_value<T>(v, static_cast<int>(s_best_d), is_exc);
    out_integers[global_i]       = is_exc ? s_fill : enc;
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows) return;
  int const vec = i / kAlpVectorSize;
  int const d   = static_cast<int>(metadata[vec]);
  out[i]        = alp_decode_value<T>(integers[i], d);
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
    auto empty_exc =
      cudf::make_fixed_width_column(col.type(), 0, cudf::mask_state::UNALLOCATED, stream, mr);
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
  auto exc = compact_exceptions<T>(d_flags.data(), n, col.data<T>(), col.type(), stream, mr);

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
  : standalone_compressed_representation(type, n_rows), num_vectors(n_vectors)
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
    case cudf::type_id::DECIMAL32: return alp_decompress_impl<int32_t>(*this, stream, mr);
    case cudf::type_id::DECIMAL64: return alp_decompress_impl<int64_t>(*this, stream, mr);
    default:
      throw std::runtime_error(
        "alp: only FLOAT32 / FLOAT64 / DECIMAL32 / DECIMAL64 are supported (got " +
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
    // DECIMAL128 stays out: its mantissa is __int128, with no table or atomics.
    case cudf::type_id::DECIMAL32:
      return alp_compress_impl<int32_t>(column_to_compress, stream, mr);
    case cudf::type_id::DECIMAL64:
      return alp_compress_impl<int64_t>(column_to_compress, stream, mr);
    default:
      throw std::runtime_error(
        "alp: only FLOAT32 / FLOAT64 / DECIMAL32 / DECIMAL64 are supported (got " +
        type_id_to_name(dt) + "). Use alp_rd for non-decimal floats.");
  }
}

}  // namespace simpatico
