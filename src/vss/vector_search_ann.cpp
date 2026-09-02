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

#include "duckdb/common/exception.hpp"
#include "sirius_context.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/distance_metric.hpp"
#include "vss/ivf_flat_index.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace sirius::vss {

namespace {

// Gather @p output_columns from the pinned table by global row indices, without
// coalescing the chunks. Chunks are in build order, so global index g lives in the
// single chunk whose base <= g < base + rows. We gather each chunk with NULLIFY
// (indices outside it become null) and null-coalesce across chunks: every output
// position is filled by exactly the one chunk that owns its index. Result columns
// are row-aligned with @p global_indices.
std::vector<std::unique_ptr<cudf::column>> gather_pinned_by_global_index(
  const vector_search_context& c,
  const std::vector<std::vector<cudf::column_view>>& out_chunks,
  const std::vector<std::int64_t>& chunk_bases,
  cudf::column_view const& global_indices)
{
  auto const n_chunks   = chunk_bases.size();
  auto const n_out_cols = out_chunks.size();
  if (n_out_cols == 0) { return {}; }

  std::vector<std::unique_ptr<cudf::column>> out_cols(n_out_cols);
  for (std::size_t ci = 0; ci < n_chunks; ++ci) {
    std::vector<cudf::column_view> chunk_cols;
    chunk_cols.reserve(n_out_cols);
    for (auto const& oc : out_chunks) {
      chunk_cols.push_back(oc[ci]);
    }

    // Local indices into this chunk; out-of-range (including negative) become null.
    auto local =
      cudf::binary_operation(global_indices,
                             cudf::numeric_scalar<std::int64_t>(
                               static_cast<std::int64_t>(chunk_bases[ci]), true, c.stream),
                             cudf::binary_operator::SUB,
                             cudf::data_type{cudf::type_id::INT64},
                             c.stream,
                             c.mr);
    auto part      = cudf::gather(cudf::table_view(chunk_cols),
                             local->view(),
                             cudf::out_of_bounds_policy::NULLIFY,
                             cudf::negative_index_policy::NOT_ALLOWED,
                             c.stream,
                             c.mr);
    auto part_cols = part->release();
    restore_native_carriers(part_cols, c.req.output_column_types, c.stream, c.mr);

    for (std::size_t j = 0; j < n_out_cols; ++j) {
      if (ci == 0) {
        out_cols[j] = std::move(part_cols[j]);
      } else {
        // Keep the already-filled value; otherwise take this chunk's.
        auto keep   = cudf::is_valid(out_cols[j]->view(), c.stream, c.mr);
        out_cols[j] = cudf::copy_if_else(
          out_cols[j]->view(), part_cols[j]->view(), keep->view(), c.stream, c.mr);
      }
    }
  }
  return out_cols;
}

}  // namespace

std::unique_ptr<cucascade::host_data_representation> run_vector_search_ann(
  const vector_search_context& c)
{
  auto const& req   = c.req;
  auto const metric = ann_distance_type_from_metric(req.metric);

  auto index_entry = c.ctx.get_cuvs_index_cache().find_by_column(
    req.catalog, req.schema, req.table_name, req.column_name, metric);
  if (index_entry == nullptr || !index_entry->index) {
    throw duckdb::InvalidInputException(
      "sirius_knn_search: no ANN index for '" + req.table_name + "." + req.column_name +
      "' under the requested metric; create one with sirius_create_ann_index or pass "
      "use_index => false");
  }

  auto const n_lists =
    index_entry->meta.n_lists > 0 ? static_cast<std::uint32_t>(index_entry->meta.n_lists) : 1u;
  std::uint32_t const n_probes =
    req.n_probes > 0 ? static_cast<std::uint32_t>(std::min<std::int64_t>(req.n_probes, n_lists))
                     : std::min<std::uint32_t>(n_lists, 32u);

  auto search = search_ivf_flat_index(
    *index_entry->index, c.query_device, req.dim, c.k, n_probes, c.stream, c.mr);

  // When the probed lists hold fewer than k vectors, IVF-Flat pads the result with
  // dummy slots whose distance is the sort-key sentinel. So here we drop that padding.
  auto const finite_max =
    cudf::numeric_scalar<float>(std::numeric_limits<float>::max(), true, c.stream);
  auto valid = cudf::binary_operation(search.distances->view(),
                                      finite_max,
                                      cudf::binary_operator::LESS,
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      c.stream,
                                      c.mr);
  auto kept =
    cudf::apply_boolean_mask(cudf::table_view{{search.neighbors->view(), search.distances->view()}},
                             valid->view(),
                             c.stream,
                             c.mr);
  auto kept_cols = kept->release();
  auto neighbors = std::move(kept_cols[0]);  // global row ids, aligned with distances
  auto distances = std::move(kept_cols[1]);

  // Per-chunk views of each output column, in build order, plus the chunk row boundaries
  auto const vec_chunks = pinned_column_chunk_views(c.pin, req.column_name, c.space);
  auto const n_chunks   = vec_chunks.size();
  std::vector<std::vector<cudf::column_view>> out_chunks;
  out_chunks.reserve(req.output_columns.size());
  std::vector<std::int64_t> chunk_bases(n_chunks);
  {
    std::int64_t base = 0;
    for (std::size_t ci = 0; ci < n_chunks; ++ci) {
      chunk_bases[ci] = base;
      base += static_cast<std::int64_t>(vec_chunks[ci].size());
    }
  }
  for (auto const& name : req.output_columns) {
    auto views = pinned_column_chunk_views(c.pin, name, c.space);
    if (views.size() != n_chunks) {
      throw duckdb::InvalidInputException(
        "sirius_knn_search: pinned table columns have inconsistent chunk counts");
    }
    out_chunks.push_back(std::move(views));
  }

  auto out_cols = gather_pinned_by_global_index(c, out_chunks, chunk_bases, neighbors->view());
  out_cols.push_back(std::move(distances));  // trailing distance column
  auto output_table = std::make_unique<cudf::table>(std::move(out_cols));

  // ANN output order is not guaranteed, so sort nearest-first by distance.
  auto const dist_idx = static_cast<cudf::size_type>(req.output_columns.size());
  output_table        = cudf::sort_by_key(output_table->view(),
                                   cudf::table_view({output_table->view().column(dist_idx)}),
                                          {cudf::order::ASCENDING},
                                          {cudf::null_order::AFTER},
                                   c.stream,
                                   c.mr);

  return vss_result_to_host(c, std::move(output_table));
}

}  // namespace sirius::vss
