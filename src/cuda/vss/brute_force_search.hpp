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

#pragma once

#include "cuda/vss/cudf_raft_interop.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/resource_ref.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstdint>
#include <memory>

namespace sirius::vss {

/**
 * @brief Result of a brute-force k-NN search.
 *
 * Both columns are flattened row-major with length `n_queries * k`: query `q`'s
 * results occupy the half-open range `[q * k, (q + 1) * k)`, ordered nearest
 * first.
 */
struct knn_result {
  std::unique_ptr<cudf::column> neighbors;  ///< INT64 row indices into the dataset.
  std::unique_ptr<cudf::column> distances;  ///< FLOAT32 distances to those rows.
  int64_t n_queries;
  int64_t k;
};

/**
 * @brief A brute-force index over a borrowed dataset, reusable across searches.
 *
 * cuVS's index build precomputes the dataset norms. Building one inside every search
 * recomputes them per query batch, which the clustered fold pays for repeatedly: it
 * searches one corpus slice with every probe run that wants it. Hoisting the index to
 * the slice pays the build once.
 *
 * Non-owning: the dataset passed to @ref brute_force_build must outlive the index.
 */
class brute_force_index {
 public:
  brute_force_index(brute_force_index&&) noexcept;
  brute_force_index& operator=(brute_force_index&&) noexcept;
  brute_force_index(brute_force_index const&)            = delete;
  brute_force_index& operator=(brute_force_index const&) = delete;
  ~brute_force_index();

  struct impl;

 private:
  explicit brute_force_index(std::unique_ptr<impl> pimpl);

  friend brute_force_index brute_force_build(raft::device_resources const&,
                                             dataset_matrix_view,
                                             cuvs::distance::DistanceType);
  friend knn_result brute_force_knn_untrimmed(raft::device_resources const&,
                                              brute_force_index const&,
                                              dataset_matrix_view,
                                              int64_t,
                                              rmm::device_async_resource_ref);

  std::unique_ptr<impl> _pimpl;
};

/**
 * @brief Build a reusable brute-force index over @p dataset.
 *
 * Enqueued on @p res's stream, like the search that follows it.
 */
brute_force_index brute_force_build(
  raft::device_resources const& res,
  dataset_matrix_view dataset,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded);

/**
 * @brief Search a prebuilt index -- otherwise identical to the one-shot @ref brute_force_knn.
 *
 * @p queries must match the index's dataset dimensionality, and @p k must satisfy
 * `1 <= k <= n_rows` of that dataset.
 */
knn_result brute_force_knn(
  raft::device_resources const& res,
  brute_force_index const& index,
  dataset_matrix_view queries,
  int64_t k,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief @ref brute_force_knn on a prebuilt index, leaving any over-search untrimmed.
 *
 * The result may hold more than @p k columns per row -- its own `k` says how many -- and the
 * first @p k of each row are the answer, nearest first. For a caller that reads the rows with
 * its own kernel anyway, the trim would only be a second pass over them.
 */
knn_result brute_force_knn_untrimmed(
  raft::device_resources const& res,
  brute_force_index const& index,
  dataset_matrix_view queries,
  int64_t k,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Exact (brute-force) k-nearest-neighbor search via cuVS.
 *
 * For every query vector, finds the @p k nearest dataset vectors under @p
 * metric. @p dataset and @p queries must share the same dimensionality and be
 * row-major `[n, dim]` FLOAT32 matrices (see @ref list_column_as_dataset_view).
 *
 * Output columns are allocated through @p mr (defaulting to cudf's current
 * device resource). In Sirius, pass the owning memory space's allocator so the
 * results are reserved against that exact space rather than the ambient default.
 * cuVS's internal scratch is separate: it draws from rmm's current device
 * resource (not @p mr), which Sirius installs as the cucascade allocator, so it
 * is still reserved, just against the ambient current space rather than @p mr.
 *
 * The search is enqueued on @p res's stream and is not synchronized before returning.
 * Results are only valid to read on the host after the caller syncs that stream. The
 * returned columns, the borrowed inputs, and the caller's downstream work therefore all
 * order on @p res's stream. Pass one @p res reused across chunks so the handle's
 * workspace setup is paid once rather than per call.
 *
 * @param res     Caller-owned RAFT resources; the search runs on its stream.
 * @param dataset Row-major dataset to search.
 * @param queries Row-major query vectors.
 * @param k       Number of neighbors per query.
 * @param metric  Distance metric.
 * @param mr      Device resource for the output columns (default: cudf's current).
 * @return Flattened neighbor-index and distance columns, plus `n_queries`/`k`.
 */
knn_result brute_force_knn(
  raft::device_resources const& res,
  dataset_matrix_view dataset,
  dataset_matrix_view queries,
  int64_t k,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded,
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

/**
 * @brief Expanded-L2 top-k as one GEMM and one selection, with no distance pass in between.
 *
 * cuVS writes the [queries x corpus] dot products, rewrites every one of them into a distance,
 * then selects -- three passes over a matrix of 1e10 entries at SIFT1M x 10k, where the GEMM was
 * 43% of the kernel time. Ranking a query's corpus rows needs only |x|^2 - 2 q.x, since |q|^2 is
 * the same for all of them, and that is itself a single GEMM over [q, 1] and [-2x, |x|^2]. So
 * the selection reads the GEMM output directly and only the k survivors are turned into
 * distances. Same FP32 arithmetic as the expanded metric (no TF32), same output contract as
 * @ref brute_force_knn; @p take_sqrt selects L2SqrtExpanded over L2Expanded.
 */
knn_result gemm_l2_topk(
  raft::device_resources const& res,
  dataset_matrix_view dataset,
  dataset_matrix_view queries,
  int64_t k,
  bool take_sqrt,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/// Whether @ref gemm_topk (and gemm_threshold) handle @p metric: the expanded L2 variants and
/// CosineExpanded. Cosine ranks by -q^.x^ over unit vectors, one GEMM with no extra column.
bool gemm_search_supports(cuvs::distance::DistanceType metric);

/// Largest [queries x corpus] score tile a GEMM-ranked search holds at once (default 512 MiB,
/// SIRIUS_VSS_GEMM_TILE_MB overrides). Part of what a task has to reserve for one.
std::size_t gemm_search_tile_bytes();

/// @ref gemm_l2_topk for any metric @ref gemm_search_supports accepts.
knn_result gemm_topk(raft::device_resources const& res,
                     dataset_matrix_view dataset,
                     dataset_matrix_view queries,
                     int64_t k,
                     cuvs::distance::DistanceType metric,
                     rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Expanded-L2 top-k over byte-valued vectors held as int8 (value - 128), on the int8 tensor
 *        cores.
 *
 * L2 does not change when both sides shift by 128, and the shifted values fit int8, so the dot
 * products come out of an int8 GEMM exact in int32. |x|^2 and |q|^2 of the shifted rows are
 * supplied (@p x_sq, @p q_sq; exact int32). The distances are exact up to the final sqrt.
 * @p x is [n x d] and @p q [m x d], row-major; @p d a multiple of 4. @p x must stay readable for
 * up to 3 rows past @p n (the GEMM runs over n rounded up to 4; those rows are ignored).
 */
knn_result gemm_int8_topk(
  raft::device_resources const& res,
  std::int8_t const* x,
  std::int32_t const* x_sq,
  int64_t n,
  std::int8_t const* q,
  std::int32_t const* q_sq,
  int64_t m,
  int64_t d,
  int64_t k,
  bool take_sqrt,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

#ifdef SIRIUS_ENABLE_FAISS_KERNEL
/**
 * @brief The same search, run by FAISS-GPU instead of cuVS.
 *
 * Same contract as @ref brute_force_knn -- same layout, same unsquared L2 distances -- so the
 * two are interchangeable under one query and the difference measured is the kernel's.
 * Selected by SIRIUS_VSS_KERNEL=faiss. L2 only; FAISS has no cosine metric and this refuses
 * rather than quietly normalizing on the caller's behalf.
 */
knn_result brute_force_knn_faiss(
  raft::device_resources const& res,
  dataset_matrix_view dataset,
  dataset_matrix_view queries,
  int64_t k,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded,
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());
#endif

}  // namespace sirius::vss
