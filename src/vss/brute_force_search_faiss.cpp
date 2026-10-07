/*
 * Copyright 2026, Sirius Contributors.
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

/**
 * @file brute_force_search_faiss.cpp
 * @brief FAISS-GPU behind the same seam as the cuVS brute-force kernel.
 *
 * Why this exists: measured on identical shapes (SIFT1M, M=10000), FAISS is 1.62x faster
 * than our cuVS path at k=1 and 3.56x at k=1000. The gap at small k cannot be k-selection --
 * there is almost nothing to select at k=1 -- so it is the distance path, and it is a kernel
 * gap rather than operator overhead: our operator already runs at raw cuVS's standalone rate.
 * This lets the two be swapped under one query so the claim is measured, not argued.
 *
 * Not built by default. Configure with -DSIRIUS_ENABLE_FAISS_KERNEL=ON plus FAISS_INCLUDE_DIR
 * and FAISS_LIBRARY, then select at runtime with SIRIUS_VSS_KERNEL=faiss.
 *
 * Two semantic differences from the cuVS path, both handled here rather than by the caller:
 *
 *  - FAISS METRIC_L2 returns *squared* L2, our metric mapping asks cuVS for L2Sqrt*, and
 *    downstream (threshold eps, the emitted distance column, every exactness check against
 *    DuckDB array_distance) reads unsquared distances. The square root is taken here.
 *  - FAISS has no cosine metric. Rather than silently normalize on the caller's behalf, cosine
 *    is refused so it keeps running on cuVS.
 *
 * Caveat worth keeping in view: StandardGpuResources allocates its own scratch through CUDA
 * directly, so that memory escapes cucascade's reservation accounting. Acceptable for a
 * measurement, not for a shipped path, which is why temp memory is pinned to a small fixed
 * budget below instead of FAISS's default (a fraction of total VRAM).
 */

#include "cuda/vss/brute_force_search.hpp"
#include "vss/size_limits.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/unary.hpp>

#include <raft/core/resource/cuda_stream.hpp>

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <cstddef>
#include <stdexcept>

namespace sirius::vss {

namespace {

// FAISS's resources object is not documented as thread-safe and owns a scratch allocator, so
// each worker thread gets its own. Temp memory is capped explicitly: the default is a fraction
// of total VRAM, which on a 40 GB card would reserve more than the join's whole budget, times
// the number of pipeline threads.
constexpr std::size_t kFaissTempMemoryBytes = 512UL * 1024 * 1024;

faiss::gpu::StandardGpuResources& faiss_resources()
{
  thread_local faiss::gpu::StandardGpuResources resources = [] {
    faiss::gpu::StandardGpuResources r;
    r.setTempMemory(kFaissTempMemoryBytes);
    return r;
  }();
  return resources;
}

}  // namespace

knn_result brute_force_knn_faiss(raft::device_resources const& res,
                                 dataset_matrix_view dataset,
                                 dataset_matrix_view queries,
                                 int64_t k,
                                 cuvs::distance::DistanceType metric,
                                 rmm::device_async_resource_ref mr)
{
  auto const n_rows    = dataset.extent(0);
  auto const n_queries = queries.extent(0);
  auto const dims      = dataset.extent(1);

  CUDF_EXPECTS(queries.extent(1) == dims, "VSS dataset and query dimensionality must match");
  CUDF_EXPECTS(k >= 1 && k <= n_rows, "VSS k must satisfy 1 <= k <= n_rows");

  switch (metric) {
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2Unexpanded: break;
    default:
      throw std::invalid_argument(
        "brute_force_knn_faiss: only L2 is supported (FAISS has no cosine metric); "
        "unset SIRIUS_VSS_KERNEL to run this query on cuVS");
  }
  // Whether the caller asked for a square root back. Every metric accepted above that is not
  // an L2Sqrt* variant wants squared distances, which is what FAISS produces natively.
  bool const want_sqrt = metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtUnexpanded;

  auto const stream  = raft::resource::get_cuda_stream(res);
  auto const out_len = column_size(n_queries * k, "vector search top-k");

  auto neighbors_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_len, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_len, cudf::mask_state::UNALLOCATED, stream, mr);

  auto& resources = faiss_resources();
  int device      = 0;
  cudaGetDevice(&device);
  // Order FAISS's work on the caller's stream so the search, our output allocations and the
  // downstream fold all sequence on one stream, exactly as the cuVS path does.
  resources.setDefaultStream(device, stream);

  faiss::gpu::GpuDistanceParams args;
  args.metric          = faiss::METRIC_L2;
  args.k               = static_cast<int>(k);
  args.dims            = static_cast<int>(dims);
  args.vectors         = dataset.data_handle();
  args.vectorType      = faiss::gpu::DistanceDataType::F32;
  args.vectorsRowMajor = true;
  args.numVectors      = static_cast<faiss::idx_t>(n_rows);
  args.queries         = queries.data_handle();
  args.queryType       = faiss::gpu::DistanceDataType::F32;
  args.queriesRowMajor = true;
  args.numQueries      = static_cast<faiss::idx_t>(n_queries);
  args.outDistances    = distances_col->mutable_view().data<float>();
  args.outIndicesType  = faiss::gpu::IndicesDataType::I64;
  args.outIndices      = neighbors_col->mutable_view().data<int64_t>();
  args.device          = device;

  // bfKnn, not bfKnn_tiling: the corpus chunk handed to us is already device-resident because
  // the operator's corpus_source staged it. Tiling is FAISS's answer to a host-resident
  // corpus, which is the job our streaming fold is already doing one level up.
  faiss::gpu::bfKnn(&resources, args);

  if (want_sqrt) {
    distances_col =
      cudf::unary_operation(distances_col->view(), cudf::unary_operator::SQRT, stream, mr);
  }

  return knn_result{std::move(neighbors_col), std::move(distances_col), n_queries, k};
}

}  // namespace sirius::vss
