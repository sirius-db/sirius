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
#include "op/vss_top_k.hpp"
#include "vss/ivf_flat_index.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search_internal.hpp"
#include "vss/vss_pattern.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <optional>
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
        "sirius_vector_search: pinned table columns have inconsistent chunk counts");
    }
    out_chunks.push_back(std::move(views));
  }

  // One pattern reused across chunks. The per-chunk input table is laid out as
  // [vector, out0, out1, ...]; the outputs are the passthroughs (input_index
  // 1..N) followed by the distance column, matching the table function's
  // [output_columns..., distance] schema.
  vss_top_k_pattern pattern;
  pattern.vector_column_index = 0;
  pattern.query               = req.query;
  pattern.dim                 = req.dim;
  pattern.metric              = enn_distance_type_from_metric(req.metric);
  for (std::size_t i = 0; i < req.output_columns.size(); ++i) {
    pattern.output_columns.push_back(
      {vss_output_column::kind::gather_input, static_cast<cudf::size_type>(i + 1)});
  }
  pattern.output_columns.push_back({vss_output_column::kind::distance, 0});
  pattern.distance_output_index = static_cast<cudf::size_type>(req.output_columns.size());

  auto const query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    c.query_device, int64_t{1}, req.dim);

  // Brute-force top-k per chunk.
  std::vector<std::unique_ptr<cudf::table>> candidates;
  candidates.reserve(n_chunks);
  for (std::size_t ci = 0; ci < n_chunks; ++ci) {
    std::vector<cudf::column_view> cols;
    cols.reserve(out_chunks.size() + 1);
    cols.push_back(vec_chunks[ci]);
    for (auto const& oc : out_chunks) {
      cols.push_back(oc[ci]);
    }
    auto cand = sirius::op::compute_vss_top_k(cudf::table_view(cols),
                                              pattern,
                                              static_cast<std::size_t>(c.k),
                                              0,
                                              c.stream,
                                              c.mr,
                                              query_view);
    if (cand->num_rows() > 0) { candidates.push_back(std::move(cand)); }
  }

  if (candidates.empty()) {
    return vss_result_to_host(c, make_empty_vss_output(c.pin, req.output_columns));
  }

  // Merge per-chunk candidates into the global top-k (sort by distance).
  std::unique_ptr<cudf::table> combined;
  if (candidates.size() == 1) {
    combined = std::move(candidates.front());
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(candidates.size());
    for (auto const& t : candidates) {
      views.push_back(t->view());
    }
    combined = cudf::concatenate(views, c.stream, c.mr);
  }

  auto merged = sirius::op::merge_vss_top_k(combined->view(),
                                            pattern.distance_output_index,
                                            static_cast<std::size_t>(c.k),
                                            0,
                                            c.stream,
                                            c.mr);
  return vss_result_to_host(c, std::move(merged));
}

}  // namespace sirius::vss
