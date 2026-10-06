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

#include "cuda/vss/brute_force_search.hpp"
#include "cuda/vss/brute_force_threshold.hpp"
#include "vss/size_limits.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_scan.cuh>

#include <cublas_v2.h>
#include <cuvs/selection/select_k.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>

namespace sirius::vss {

namespace {

constexpr int kBlock = 256;
constexpr int kWarp  = 32;

// How a metric becomes "smaller GEMM score is closer". L2 ranks by |x|^2 - 2 q.x, which is one
// GEMM over [q, 1] and [-2x, |x|^2]; cosine ranks by -q^.x^, a GEMM over unit vectors. Either
// way the selection or the threshold reads the GEMM output directly, and only survivors are
// turned into the metric's own distance.
// A cosine THRESHOLD instead keeps the raw dot product and applies 1 - q.x / (|q||x|) per pair:
// on the boundary that is what decides membership, and it is the formula DuckDB and cuVS round
// the same way, where normalizing first moves a few boundary pairs.
enum class score_kind : int { l2_squared, l2, cosine, cosine_raw };

std::optional<score_kind> score_kind_of(cuvs::distance::DistanceType metric)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded: return score_kind::l2_squared;
    case cuvs::distance::DistanceType::L2SqrtExpanded: return score_kind::l2;
    case cuvs::distance::DistanceType::CosineExpanded: return score_kind::cosine;
    default: return std::nullopt;
  }
}

// Corpus rows, one warp per row: [-2x, |x|^2, 0...] for L2, [x/|x|, 0...] for cosine. A zero
// row stays zero under cosine, i.e. similarity 0 to everything.
__global__ void prepare_dataset_kernel(
  float const* x, int64_t n, int64_t d, int64_t dp, int kind, float* out, float* root_norms)
{
  bool const cosine = kind == static_cast<int>(score_kind::cosine) ||
                      kind == static_cast<int>(score_kind::cosine_raw);
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarp);
  auto const lane  = static_cast<int64_t>(threadIdx.x % kWarp);
  for (int64_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp; r < n; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += kWarp) {
      auto const v = x[r * d + j];
      acc += v * v;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    auto const scale = kind == static_cast<int>(score_kind::cosine_raw)
                         ? 1.f
                         : (cosine ? (acc > 0.f ? rsqrtf(acc) : 0.f) : -2.f);
    for (int64_t j = lane; j < d; j += kWarp) {
      out[r * dp + j] = scale * x[r * d + j];
    }
    auto const extra = cosine ? d : d + 1;
    for (int64_t j = extra + lane; j < dp; j += kWarp) {
      out[r * dp + j] = 0.f;
    }
    if (lane == 0) {
      if (!cosine) { out[r * dp + d] = acc; }
      root_norms[r] = sqrtf(acc);
    }
  }
}

// Query rows: [q, 1, 0...] and |q|^2 for L2, [-q/|q|, 0...] for cosine (negated so that the
// smallest score is the most similar, as for L2).
__global__ void prepare_queries_kernel(
  float const* q, int64_t m, int64_t d, int64_t dp, int kind, float* out, float* norms)
{
  bool const cosine = kind == static_cast<int>(score_kind::cosine) ||
                      kind == static_cast<int>(score_kind::cosine_raw);
  auto const warps = static_cast<int64_t>(gridDim.x) * (blockDim.x / kWarp);
  auto const lane  = static_cast<int64_t>(threadIdx.x % kWarp);
  for (int64_t r = blockIdx.x * (blockDim.x / kWarp) + threadIdx.x / kWarp; r < m; r += warps) {
    float acc = 0.f;
    for (int64_t j = lane; j < d; j += kWarp) {
      auto const v = q[r * d + j];
      acc += v * v;
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    auto const scale = kind == static_cast<int>(score_kind::cosine_raw)
                         ? 1.f
                         : (cosine ? (acc > 0.f ? -rsqrtf(acc) : 0.f) : 1.f);
    for (int64_t j = lane; j < d; j += kWarp) {
      out[r * dp + j] = scale * q[r * d + j];
    }
    auto const extra = cosine ? d : d + 1;
    for (int64_t j = extra + lane; j < dp; j += kWarp) {
      out[r * dp + j] = 0.f;
    }
    if (lane == 0) {
      if (!cosine) { out[r * dp + d] = 1.f; }
      norms[r] = acc;
    }
  }
}

// The metric's distance from a GEMM score: + |q|^2 (clamped at the zero rounding can undershoot)
// and square-rooted for L2; 1 + score = 1 - cos for cosine.
__device__ __forceinline__ float score_to_distance(float s, float qn, int kind)
{
  if (kind == static_cast<int>(score_kind::cosine)) { return 1.f + s; }
  auto const v = fmaxf(s + qn, 0.f);
  return kind == static_cast<int>(score_kind::l2) ? sqrtf(v) : v;
}

__global__ void finish_topk_kernel(
  float* vals, int64_t rows, int64_t k, float const* norms, int kind)
{
  auto const total = rows * k;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    vals[p] = score_to_distance(vals[p], norms[p / k], kind);
  }
}

// One pass over a [t x n] score tile: every entry at or under its row's bound is appended as an
// edge. blockIdx.y is the row, so no per-element division; each thread takes kPerThread
// consecutive columns and a block scan places the block's matches with ONE global atomic, which
// is what keeps a dense threshold (hundreds of millions of pairs) from serializing on the counter.
constexpr int kPerThread = 8;

__global__ void threshold_emit_kernel(float const* scores,
                                      int64_t n,
                                      float const* bounds,
                                      float const* norms,
                                      int kind,
                                      float const* col_root_norms,
                                      float eps,
                                      int64_t row0,
                                      int64_t* out_rows,
                                      int64_t* out_cols,
                                      float* out_dist,
                                      unsigned long long* count,
                                      unsigned long long capacity)
{
  using block_scan = cub::BlockScan<int, kBlock>;
  __shared__ typename block_scan::TempStorage scan_storage;
  __shared__ unsigned long long block_base;

  auto const r     = static_cast<int64_t>(blockIdx.y);
  auto const bound = bounds[r];
  auto const* row  = scores + r * n;
  auto const c0 =
    (static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x) * static_cast<int64_t>(kPerThread);

  bool const raw_cosine = kind == static_cast<int>(score_kind::cosine_raw);
  auto const q_root     = raw_cosine ? sqrtf(norms[r]) : 0.f;
  // For the raw-dot cosine, v holds the distance itself and the bound is eps; a zero-norm row
  // on either side has no direction and joins nothing.
  float v[kPerThread];
  int local = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    auto const c = c0 + i;
    if (c >= n) {
      v[i] = INFINITY;
    } else if (raw_cosine) {
      auto const denom = q_root * col_root_norms[c];
      v[i]             = denom > 0.f ? 1.f - row[c] / denom : INFINITY;
    } else {
      v[i] = row[c];
    }
    local += v[i] <= (raw_cosine ? eps : bound);
  }
  int offset = 0;
  int total  = 0;
  block_scan(scan_storage).ExclusiveSum(local, offset, total);
  if (threadIdx.x == 0) {
    block_base = total == 0 ? 0 : atomicAdd(count, static_cast<unsigned long long>(total));
  }
  __syncthreads();
  if (local == 0) { return; }
  auto pos      = block_base + static_cast<unsigned long long>(offset);
  auto const qn = norms[r];
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    if (v[i] <= (raw_cosine ? eps : bound)) {
      if (pos < capacity) {
        out_rows[pos] = row0 + r;
        out_cols[pos] = c0 + i;
        out_dist[pos] = raw_cosine ? v[i] : score_to_distance(v[i], qn, kind);
      }
      ++pos;
    }
  }
}

// Per query row, the score bound equivalent to distance <= eps.
__global__ void threshold_bounds_kernel(
  float const* norms, int64_t m, float eps, int kind, float* bounds)
{
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < m;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (kind == static_cast<int>(score_kind::cosine)) {
      bounds[r] = eps - 1.f;  // 1 + s <= eps
    } else {
      auto const eps2 = kind == static_cast<int>(score_kind::l2) ? eps * eps : eps;
      bounds[r]       = eps2 - norms[r];  // s + |q|^2 <= eps^2
    }
  }
}

int grid_for_warps(int64_t rows)
{
  constexpr int64_t per_block = kBlock / kWarp;
  return static_cast<int>(std::clamp<int64_t>((rows + per_block - 1) / per_block, 1, 65535));
}

int grid_for(int64_t n)
{
  return static_cast<int>(std::clamp<int64_t>((n + kBlock - 1) / kBlock, 1, 65535));
}

/// The prepared operands of a GEMM-ranked search and the tiled GEMM over them.
struct gemm_search {
  int64_t n, m, d, dp;
  score_kind kind;
  rmm::device_uvector<float> xa, qa, qn;
  rmm::device_uvector<float> x_root_norms;  ///< |x| per corpus row
  rmm::device_uvector<float> scores;
  int64_t tile_rows;

  gemm_search(raft::device_resources const& res,
              dataset_matrix_view dataset,
              dataset_matrix_view queries,
              score_kind kind_,
              rmm::device_async_resource_ref mr)
    : n(dataset.extent(0)),
      m(queries.extent(0)),
      d(dataset.extent(1)),
      // L2 carries |x|^2 through one extra column; the pad keeps rows 16-byte aligned.
      dp((d + (kind_ == score_kind::cosine || kind_ == score_kind::cosine_raw ? 0 : 1) + 3) / 4 *
         4),
      kind(kind_),
      xa(static_cast<std::size_t>(n * dp), raft::resource::get_cuda_stream(res), mr),
      qa(static_cast<std::size_t>(m * dp), raft::resource::get_cuda_stream(res), mr),
      qn(static_cast<std::size_t>(m), raft::resource::get_cuda_stream(res), mr),
      x_root_norms(static_cast<std::size_t>(n), raft::resource::get_cuda_stream(res), mr),
      // The score tile is the only large buffer, bounded whatever the corpus chunk, which
      // bounds peak memory the way cuVS's own tiling does.
      scores(0, raft::resource::get_cuda_stream(res), mr),
      tile_rows(
        std::clamp<int64_t>(static_cast<int64_t>(gemm_search_tile_bytes() /
                                                 (static_cast<std::size_t>(n) * sizeof(float))),
                            1,
                            // The threshold emission puts a tile's rows on the grid's y dimension.
                            std::min<int64_t>(std::max<int64_t>(m, 1), 65535)))
  {
    CUDF_EXPECTS(queries.extent(1) == d, "VSS dataset and query dimensionality must match");
    CUDF_EXPECTS(n <= std::numeric_limits<int>::max(), "GEMM search: corpus chunk too large");
    auto const stream = raft::resource::get_cuda_stream(res);
    prepare_dataset_kernel<<<grid_for_warps(n), kBlock, 0, stream.value()>>>(
      dataset.data_handle(), n, d, dp, static_cast<int>(kind), xa.data(), x_root_norms.data());
    prepare_queries_kernel<<<grid_for_warps(m), kBlock, 0, stream.value()>>>(
      queries.data_handle(), m, d, dp, static_cast<int>(kind), qa.data(), qn.data());
    CUDF_CHECK_CUDA(stream.value());
    scores.resize(static_cast<std::size_t>(tile_rows * n), stream);
  }

  /// Run the GEMM tile by tile; @p consume(q0, t) reads scores [t x n] for queries [q0, q0 + t).
  void for_each_tile(raft::device_resources const& res,
                     std::function<void(int64_t, int64_t)> const& consume)
  {
    auto const stream = raft::resource::get_cuda_stream(res);
    auto handle       = raft::resource::get_cublas_handle(res);
    CUDF_EXPECTS(cublasSetStream(handle, stream.value()) == CUBLAS_STATUS_SUCCESS,
                 "GEMM search: cublasSetStream failed");
    float const alpha = 1.f;
    float const beta  = 0.f;
    for (int64_t q0 = 0; q0 < m; q0 += tile_rows) {
      auto const t = std::min(tile_rows, m - q0);
      // Row-major scores[t x n] = qa_tile[t x dp] * xa[n x dp]^T, which column-major cuBLAS sees
      // as scores^T[n x t] = xa^T(op T of a dp x n matrix) * qa_tile^T. FP32 compute, no TF32:
      // the answer is exact to FP32 rounding, like the brute-force search it replaces.
      auto const status = cublasGemmEx(handle,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       static_cast<int>(n),
                                       static_cast<int>(t),
                                       static_cast<int>(dp),
                                       &alpha,
                                       xa.data(),
                                       CUDA_R_32F,
                                       static_cast<int>(dp),
                                       qa.data() + q0 * dp,
                                       CUDA_R_32F,
                                       static_cast<int>(dp),
                                       &beta,
                                       scores.data(),
                                       CUDA_R_32F,
                                       static_cast<int>(n),
                                       CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT);
      CUDF_EXPECTS(status == CUBLAS_STATUS_SUCCESS, "GEMM search: cublasGemmEx failed");
      consume(q0, t);
    }
  }
};

template <typename T>
std::unique_ptr<cudf::column> uvector_to_column(rmm::device_uvector<T>&& v, cudf::type_id id)
{
  auto const size = column_size(static_cast<std::int64_t>(v.size()), "vector join threshold");
  return std::make_unique<cudf::column>(
    cudf::data_type{id}, size, v.release(), rmm::device_buffer{}, 0);
}

// int32 dot products -> float scores |x|^2 - 2 q.x, in place (same width). Exact: both terms are
// integers under 2^24 for byte-valued data of dimension up to 512.
// Rows are ldc wide (n rounded up to 4, which the int8 GEMM requires of its output); the padding
// becomes +inf so the selection never picks it. blockIdx.y is the row.
__global__ void int8_scores_kernel(int32_t* tile, int64_t n, int64_t ldc, int32_t const* x_sq)
{
  auto* row   = tile + static_cast<int64_t>(blockIdx.y) * ldc;
  auto* row_f = reinterpret_cast<float*>(row);
  for (int64_t c = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; c < ldc;
       c += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    row_f[c] = c < n ? static_cast<float>(x_sq[c] - 2 * row[c]) : INFINITY;
  }
}

__global__ void finish_int8_topk_kernel(
  float* vals, int64_t rows, int64_t k, int32_t const* q_sq, bool take_sqrt)
{
  auto const total = rows * k;
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; p < total;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    auto const v = fmaxf(vals[p] + static_cast<float>(q_sq[p / k]), 0.f);
    vals[p]      = take_sqrt ? sqrtf(v) : v;
  }
}

}  // namespace

std::size_t gemm_search_tile_bytes()
{
  static std::size_t const bytes = [] {
    auto const* v = std::getenv("SIRIUS_VSS_GEMM_TILE_MB");
    return (v != nullptr ? std::strtoull(v, nullptr, 10) : 512ull) << 20;
  }();
  return bytes;
}

bool gemm_search_supports(cuvs::distance::DistanceType metric)
{
  return score_kind_of(metric).has_value();
}

knn_result gemm_topk(raft::device_resources const& res,
                     dataset_matrix_view dataset,
                     dataset_matrix_view queries,
                     int64_t k,
                     cuvs::distance::DistanceType metric,
                     rmm::device_async_resource_ref mr)
{
  auto const kind = score_kind_of(metric);
  CUDF_EXPECTS(kind.has_value(), "gemm_topk: unsupported metric");
  CUDF_EXPECTS(k >= 1 && k <= dataset.extent(0), "VSS k must satisfy 1 <= k <= n_rows");
  auto const stream = raft::resource::get_cuda_stream(res);
  gemm_search search(res, dataset, queries, *kind, mr);
  auto const m = search.m;
  auto const n = search.n;

  auto const out_size = column_size(m * k, "vector join top-k");
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();

  search.for_each_tile(res, [&](int64_t q0, int64_t t) {
    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        search.scores.data(), t, n),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_d + q0 * k, t, k),
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(out_n + q0 * k, t, k),
      /*select_min=*/true,
      /*sorted=*/true);
  });
  finish_topk_kernel<<<grid_for(m * k), kBlock, 0, stream.value()>>>(
    out_d, m, k, search.qn.data(), static_cast<int>(*kind));
  CUDF_CHECK_CUDA(stream.value());
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

knn_result gemm_l2_topk(raft::device_resources const& res,
                        dataset_matrix_view dataset,
                        dataset_matrix_view queries,
                        int64_t k,
                        bool take_sqrt,
                        rmm::device_async_resource_ref mr)
{
  return gemm_topk(res,
                   dataset,
                   queries,
                   k,
                   take_sqrt ? cuvs::distance::DistanceType::L2SqrtExpanded
                             : cuvs::distance::DistanceType::L2Expanded,
                   mr);
}

threshold_join_result gemm_threshold(raft::device_resources const& res,
                                     dataset_matrix_view dataset,
                                     dataset_matrix_view queries,
                                     float eps,
                                     cuvs::distance::DistanceType metric,
                                     rmm::device_async_resource_ref mr)
{
  auto kind = score_kind_of(metric);
  CUDF_EXPECTS(kind.has_value(), "gemm_threshold: unsupported metric");
  if (*kind == score_kind::cosine) { kind = score_kind::cosine_raw; }
  auto const stream = raft::resource::get_cuda_stream(res);
  gemm_search search(res, dataset, queries, *kind, mr);
  auto const m = search.m;
  auto const n = search.n;

  rmm::device_uvector<float> bounds(static_cast<std::size_t>(m), stream, mr);
  threshold_bounds_kernel<<<grid_for(m), kBlock, 0, stream.value()>>>(
    search.qn.data(), m, eps, static_cast<int>(*kind), bounds.data());

  // Edges accumulate in one buffer that doubles when a tile overflows it. The overflowing tile's
  // scores are still in place, so only its emission is re-run, never its GEMM.
  std::size_t capacity = std::size_t{1} << 20;
  std::size_t size     = 0;
  rmm::device_uvector<int64_t> rows(capacity, stream, mr);
  rmm::device_uvector<int64_t> cols(capacity, stream, mr);
  rmm::device_uvector<float> dist(capacity, stream, mr);
  rmm::device_uvector<unsigned long long> counter(1, stream, mr);

  auto const per_block = static_cast<int64_t>(kBlock) * kPerThread;
  search.for_each_tile(res, [&](int64_t q0, int64_t t) {
    for (;;) {
      CUDF_CUDA_TRY(cudaMemsetAsync(counter.data(), 0, sizeof(unsigned long long), stream.value()));
      dim3 const grid(static_cast<unsigned>((n + per_block - 1) / per_block),
                      static_cast<unsigned>(t));
      threshold_emit_kernel<<<grid, kBlock, 0, stream.value()>>>(search.scores.data(),
                                                                 n,
                                                                 bounds.data() + q0,
                                                                 search.qn.data() + q0,
                                                                 static_cast<int>(*kind),
                                                                 search.x_root_norms.data(),
                                                                 eps,
                                                                 q0,
                                                                 rows.data() + size,
                                                                 cols.data() + size,
                                                                 dist.data() + size,
                                                                 counter.data(),
                                                                 capacity - size);
      CUDF_CHECK_CUDA(stream.value());
      unsigned long long emitted = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        &emitted, counter.data(), sizeof(emitted), cudaMemcpyDeviceToHost, stream.value()));
      stream.synchronize();
      if (size + emitted <= capacity) {
        size += emitted;
        return;
      }
      auto const grown = static_cast<std::size_t>(grown_pair_capacity(
        capacity, size + emitted, 2 * sizeof(int64_t) + sizeof(float), "gemm_threshold"));
      rows.resize(grown, stream);
      cols.resize(grown, stream);
      dist.resize(grown, stream);
      capacity = grown;
    }
  });
  rows.resize(size, stream);
  cols.resize(size, stream);
  dist.resize(size, stream);
  rows.shrink_to_fit(stream);
  cols.shrink_to_fit(stream);
  dist.shrink_to_fit(stream);
  return threshold_join_result{uvector_to_column(std::move(rows), cudf::type_id::INT64),
                               uvector_to_column(std::move(cols), cudf::type_id::INT64),
                               uvector_to_column(std::move(dist), cudf::type_id::FLOAT32),
                               static_cast<int64_t>(size)};
}

// scores[j * ldc + i] = x_i . q_j for i < rows, j < cols (int8 in, exact int32 out), on the tensor
// cores. cuBLAS finds no int8 kernel for some shapes -- e.g. 110,904 rows x 17..40 columns at
// d = 96, while 110,904 x 16 and 200,000 x 40 run -- so a NOT_SUPPORTED product is re-issued in
// column slices of at most 16, and a slice cuBLAS still refuses in two row halves (multiples of 4).
static cublasStatus_t int8_gemm_tn(cublasHandle_t handle,
                                   std::int8_t const* x,
                                   std::int8_t const* q,
                                   int32_t* scores,
                                   int64_t rows,
                                   int64_t cols,
                                   int64_t d,
                                   int ldc)
{
  int32_t const alpha = 1;
  int32_t const beta  = 0;
  auto const status   = cublasGemmEx(handle,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   static_cast<int>(rows),
                                   static_cast<int>(cols),
                                   static_cast<int>(d),
                                   &alpha,
                                   x,
                                   CUDA_R_8I,
                                   static_cast<int>(d),
                                   q,
                                   CUDA_R_8I,
                                   static_cast<int>(d),
                                   &beta,
                                   scores,
                                   CUDA_R_32I,
                                   ldc,
                                   CUBLAS_COMPUTE_32I,
                                   CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_NOT_SUPPORTED) { return status; }
  if (cols > 16) {
    for (int64_t c0 = 0; c0 < cols; c0 += 16) {
      auto const st = int8_gemm_tn(
        handle, x, q + c0 * d, scores + c0 * ldc, rows, std::min<int64_t>(16, cols - c0), d, ldc);
      if (st != CUBLAS_STATUS_SUCCESS) { return st; }
    }
    return CUBLAS_STATUS_SUCCESS;
  }
  if (rows >= 8) {
    auto const half = rows / 8 * 4;
    auto const st   = int8_gemm_tn(handle, x, q, scores, half, cols, d, ldc);
    if (st != CUBLAS_STATUS_SUCCESS) { return st; }
    return int8_gemm_tn(handle, x + half * d, q, scores + half, rows - half, cols, d, ldc);
  }
  return status;
}

knn_result gemm_int8_topk(raft::device_resources const& res,
                          std::int8_t const* x,
                          std::int32_t const* x_sq,
                          int64_t n,
                          std::int8_t const* q,
                          std::int32_t const* q_sq,
                          int64_t m,
                          int64_t d,
                          int64_t k,
                          bool take_sqrt,
                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 1 && k <= n, "VSS k must satisfy 1 <= k <= n_rows");
  CUDF_EXPECTS(d % 4 == 0, "gemm_int8_topk: dimension must be a multiple of 4");
  CUDF_EXPECTS(n <= std::numeric_limits<int>::max(), "gemm_int8_topk: corpus slice too large");
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const ldc    = (n + 3) / 4 * 4;
  auto const tile_rows =
    std::clamp<int64_t>(static_cast<int64_t>(gemm_search_tile_bytes() /
                                             (static_cast<std::size_t>(ldc) * sizeof(float))),
                        1,
                        std::min<int64_t>(std::max<int64_t>(m, 1), 65535));
  rmm::device_uvector<int32_t> scores(static_cast<std::size_t>(tile_rows * ldc), stream, mr);

  auto const out_size = column_size(m * k, "vector join top-k");
  auto neighbors      = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* out_n = neighbors->mutable_view().data<int64_t>();
  auto* out_d = distances->mutable_view().data<float>();

  auto handle = raft::resource::get_cublas_handle(res);
  CUDF_EXPECTS(cublasSetStream(handle, stream.value()) == CUBLAS_STATUS_SUCCESS,
               "gemm_int8_topk: cublasSetStream failed");
  for (int64_t q0 = 0; q0 < m; q0 += tile_rows) {
    auto const t = std::min(tile_rows, m - q0);
    // Same TN layout as the FP32 search: scores^T[n x t] = x^T * q_tile^T, on the int8 tensor
    // cores with int32 accumulation -- every dot product exact.
    // The int8 GEMM wants its row count a multiple of 4 (NOT_SUPPORTED otherwise), so it runs over
    // ldc corpus rows: the up-to-3 past the slice are the next slice's, or the slack the lists keep
    // at their end, and their scores are masked to +inf below.
    auto const status =
      int8_gemm_tn(handle, x, q + q0 * d, scores.data(), ldc, t, d, static_cast<int>(ldc));
    CUDF_EXPECTS(status == CUBLAS_STATUS_SUCCESS,
                 "gemm_int8_topk: cublasGemmEx failed with status " + std::to_string(status));
    dim3 const grid(
      static_cast<unsigned>(std::clamp<int64_t>((ldc + kBlock - 1) / kBlock, 1, 65535)),
      static_cast<unsigned>(t));
    int8_scores_kernel<<<grid, kBlock, 0, stream.value()>>>(scores.data(), n, ldc, x_sq);
    CUDF_CHECK_CUDA(stream.value());
    cuvs::selection::select_k(
      res,
      raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        reinterpret_cast<float const*>(scores.data()), t, ldc),
      std::nullopt,
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_d + q0 * k, t, k),
      raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(out_n + q0 * k, t, k),
      /*select_min=*/true,
      /*sorted=*/true);
  }
  finish_int8_topk_kernel<<<grid_for(m * k), kBlock, 0, stream.value()>>>(
    out_d, m, k, q_sq, take_sqrt);
  CUDF_CHECK_CUDA(stream.value());
  return knn_result{std::move(neighbors), std::move(distances), m, k};
}

}  // namespace sirius::vss
