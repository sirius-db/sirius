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
#include "vss/enn_top_k.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <raft/core/device_resources.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::vss {

std::unique_ptr<cucascade::host_data_representation> run_vector_search_enn(
  const vector_search_context& c)
{
  auto const& req = c.req;

  // Per-chunk views of the vector column and each output column, in pin order.
  auto const vec_chunks = pinned_column_chunk_views(c.pin, req.column_name, c.space);
  auto const n_chunks   = vec_chunks.size();
  std::vector<std::vector<cudf::column_view>> out_chunks;
  out_chunks.reserve(req.output_columns.size());
  for (auto const& name : req.output_columns) {
    auto views = pinned_column_chunk_views(c.pin, name, c.space);
    if (views.size() != n_chunks) {
      throw duckdb::InvalidInputException(
        "sirius_knn_search: pinned table columns have inconsistent chunk counts");
    }
    out_chunks.push_back(std::move(views));
  }

  // One RAFT handle for every chunk: bound to c.stream and reused across the
  // loop so its workspace setup is paid once, not per chunk. Each per-chunk
  // search runs async on c.stream; res must outlive (until vss_result_to_host syncs).
  raft::device_resources res{c.stream};

  // Brute-force top-k per chunk. The per-chunk input table is laid out as
  // [vector, out0, out1, ...]; compute_enn_top_k returns [out0, ..., distance],
  // matching the table function's [output_columns..., distance] schema.
  std::vector<std::unique_ptr<cudf::table>> candidates;
  candidates.reserve(n_chunks);
  for (std::size_t ci = 0; ci < n_chunks; ++ci) {
    std::vector<cudf::column_view> cols;
    cols.reserve(out_chunks.size() + 1);
    cols.push_back(vec_chunks[ci]);
    for (auto const& oc : out_chunks) {
      cols.push_back(oc[ci]);
    }
    auto cand = compute_enn_top_k(c, cudf::table_view(cols), res);
    if (cand->num_rows() > 0) { candidates.push_back(std::move(cand)); }
  }

  if (candidates.empty()) {
    return vss_result_to_host(c, make_empty_vss_output(c.pin, req.output_columns));
  }

  // K-way merge the per-chunk top-k lists into the global top-k, sorted by distance
  std::vector<cudf::table_view> views;
  views.reserve(candidates.size());
  for (auto const& t : candidates) {
    views.push_back(t->view());
  }

  auto merged = merge_enn_top_k(c, views);
  return vss_result_to_host(c, std::move(merged));
}

}  // namespace sirius::vss
