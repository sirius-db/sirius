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

#include "vss/enn_top_k.hpp"

#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/merge.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <raft/core/device_mdspan.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::vss {

namespace {

// Empty output for the no-work cases (empty input or k == 0): the passthrough
// columns 1..N of `input` followed by a FLOAT32 distance column. These get
// dropped by the caller's `num_rows() > 0` filter, but keep the schema aligned.
std::unique_ptr<cudf::table> make_empty_enn_output(cudf::table_view const& input)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(input.num_columns());
  for (int i = 1; i < input.num_columns(); ++i) {
    cols.push_back(cudf::empty_like(input.column(i)));
  }
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

std::unique_ptr<cudf::table> compute_enn_top_k(const vector_search_context& c,
                                               cudf::table_view input,
                                               raft::device_resources const& res)
{
  auto const stream = c.stream;
  auto const mr     = c.mr;
  auto const limit  = static_cast<std::size_t>(c.k);

  if (limit == 0 || input.num_rows() == 0) { return make_empty_enn_output(input); }

  // Column 0 is the search vector. A nullable or sliced vector column can't be
  // zero-copy reinterpreted as a RAFT matrix (raw-blob read, no null/offset
  // awareness). Compact the whole table to drop null rows and reset the offset,
  // keeping passthrough columns row-aligned so neighbor indices map straight
  // into the compacted rows. Element-level nulls survive this and are rejected
  // in list_column_as_dataset_view.
  std::unique_ptr<cudf::table> compacted;
  if (auto const& vec = input.column(0); vec.offset() != 0 || vec.null_count() != 0) {
    auto valid_mask = cudf::is_valid(vec, stream, mr);
    compacted       = cudf::apply_boolean_mask(input, valid_mask->view(), stream, mr);
    input           = compacted->view();
    if (input.num_rows() == 0) { return make_empty_enn_output(input); }
  }

  auto const keep = std::min<int64_t>(input.num_rows(), static_cast<int64_t>(limit));
  if (keep == 0) { return make_empty_enn_output(input); }

  // Zero-copy reinterpretation from cudf LIST column into a matrix view.
  auto dataset_view = list_column_as_dataset_view(input.column(0), c.req.dim);
  auto query_view   = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    c.query_device, int64_t{1}, c.req.dim);

  auto knn = brute_force_knn(
    res, dataset_view, query_view, keep, enn_distance_type_from_metric(c.req.metric), mr);

  auto gathered =
    cudf::gather(input, knn.neighbors->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  // Output is the gathered passthroughs (columns 1..N) plus the cuVS distance.
  // Each output position maps to its own gathered column, so move them straight
  // out instead of deep-copying.
  auto gathered_cols = gathered->release();
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(gathered_cols.size());
  for (std::size_t i = 1; i < gathered_cols.size(); ++i) {
    out_cols.push_back(std::move(gathered_cols[i]));
  }
  out_cols.push_back(std::move(knn.distances));
  return std::make_unique<cudf::table>(std::move(out_cols));
}

std::unique_ptr<cudf::table> merge_enn_top_k(const vector_search_context& c,
                                             std::vector<cudf::table_view> const& candidates)
{
  auto const stream = c.stream;
  auto const mr     = c.mr;
  auto const limit  = static_cast<std::size_t>(c.k);

  // Distance is the trailing column of every [out0, ..., distance] candidate.
  auto const distance_index = candidates.front().num_columns() - 1;

  if (limit == 0) { return duckdb::make_empty_like(candidates.front()); }

  // Each candidate is already sorted ascending by distance (cuVS select_k), so a
  // k-way merge on the distance column yields one globally sorted table without
  // concatenating first.
  auto merged = cudf::merge(
    candidates, {distance_index}, {cudf::order::ASCENDING}, {cudf::null_order::AFTER}, stream, mr);

  auto const keep =
    std::min<cudf::size_type>(merged->num_rows(), static_cast<cudf::size_type>(limit));
  if (keep == merged->num_rows()) { return merged; }

  // Zero-copy view of the nearest `keep`; deep-copy into an owning table.
  auto head = cudf::slice(merged->view(), {0, keep}).front();
  return std::make_unique<cudf::table>(head, stream, mr);
}

}  // namespace sirius::vss
