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

#include "op/scan/positional_visibility.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <log/logging.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::op::scan {

std::unique_ptr<cudf::table> apply_positional_visibility(std::unique_ptr<cudf::table> tbl,
                                                         std::span<batch_row_run const> layout,
                                                         position_lookup const& positions_for,
                                                         ::cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  int64_t offset = 0;
  for (auto const& run : layout) {
    if (run.batch_row_offset != offset || run.num_rows < 0 || run.file_row_offset < 0 ||
        run.file_row_offset > std::numeric_limits<int64_t>::max() - run.num_rows ||
        run.num_rows > tbl->num_rows() - offset)
      throw std::invalid_argument("invalid positional visibility row layout");
    offset += run.num_rows;
  }
  if (offset != tbl->num_rows()) throw std::invalid_argument("incomplete visibility row layout");
  auto const num_rows = static_cast<int64_t>(tbl->num_rows());
  if (num_rows == 0 || !std::any_of(layout.begin(), layout.end(), [&](auto const& run) {
        return !positions_for(run.data_file_path).empty();
      })) {
    return tbl;
  }

  // Host keep-mask over the whole batch. Delete positions are sorted per file, so each run's
  // in-range window is two binary searches: O(deletes in range), not O(deletes in table).
  std::vector<uint8_t> keep(static_cast<size_t>(num_rows), 1u);
  bool any_deleted = false;

  for (auto const& run : layout) {
    auto const delete_positions = positions_for(run.data_file_path);
    if (delete_positions.empty()) { continue; }
    auto const run_end = run.file_row_offset + run.num_rows;
    auto lo =
      std::lower_bound(delete_positions.begin(), delete_positions.end(), run.file_row_offset);
    auto hi = std::lower_bound(lo, delete_positions.end(), run_end);

    for (auto pos = lo; pos != hi; ++pos) {
      auto const batch_row = run.batch_row_offset + (*pos - run.file_row_offset);
      // Out of range would silently corrupt a neighbouring row, so this is a guard, not an
      // assert.
      if (batch_row < 0 || batch_row >= num_rows) {
        throw std::out_of_range("positional visibility: delete position " + std::to_string(*pos) +
                                " in '" + run.data_file_path +
                                "' maps outside the decoded batch — the batch row layout does not "
                                "describe the decoded table");
      }
      keep[static_cast<size_t>(batch_row)] = 0u;
      any_deleted                          = true;
    }
  }

  if (!any_deleted) { return tbl; }

  auto bool_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                static_cast<cudf::size_type>(num_rows),
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(bool_col->mutable_view().data<uint8_t>(),
                               keep.data(),
                               static_cast<size_t>(num_rows) * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  // `keep` is host memory feeding an async copy and must outlive it. A pinned staging buffer
  // would avoid the sync; a one-byte-per-row mask does not justify it yet.
  stream.sync();

  return sirius::apply_retention_mask(tbl->view(), bool_col->view(), stream, mr);
}

void apply_positional_visibility(materialized_batch& batch,
                                 position_lookup const& positions,
                                 ::cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  batch.lineage.validate(batch.rows);
  if (batch.rows && !batch.lineage.available())
    throw std::invalid_argument("visibility requires original positions");
  auto runs = batch.lineage.runs;
  if (batch.lineage.positions) {
    runs.reserve(batch.lineage.positions->size());
    int64_t offset = 0;
    for (auto const& p : *batch.lineage.positions)
      runs.push_back({batch.lineage.files.at(p.file).identity, p.row, offset++, 1, p.file});
  } else {
    for (auto& run : runs)
      run.data_file_path = batch.lineage.files.at(run.file_index).identity;
  }
  batch.table = owning_table_view(
    apply_positional_visibility(batch.table.release(stream, mr), runs, positions, stream, mr));
  batch.rows = batch.table.num_rows();
  // Visibility is the final consumer of source positions in this program.
  batch.lineage             = {};
  batch.visibility_complete = true;
}
}  // namespace sirius::op::scan
