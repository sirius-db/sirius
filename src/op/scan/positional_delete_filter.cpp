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

#include <cudf/column/column_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <log/logging.hpp>
#include <op/scan/iceberg_delete_filter.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::op::scan {

positional_delete_filter::positional_delete_filter(
  std::shared_ptr<const IcebergDeleteData> delete_data)
  : _delete_data(std::move(delete_data))
{
}

bool positional_delete_filter::affects(batch_layout layout) const
{
  return std::any_of(layout.begin(), layout.end(), [this](batch_row_run const& run) {
    auto it = _delete_data->positional_deletes.find(run.data_file_path);
    return it != _delete_data->positional_deletes.end() && !it->second.empty();
  });
}

std::unique_ptr<cudf::table> positional_delete_filter::apply(std::unique_ptr<cudf::table> tbl,
                                                             batch_layout layout,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  auto const num_rows = static_cast<int64_t>(tbl->num_rows());
  if (num_rows == 0 || !affects(layout)) { return tbl; }

  // Build a keep-mask on the host: true = keep row, false = deleted. The mask covers the whole
  // decoded batch; each run contributes the slice of its file's delete positions that lands
  // inside it. Delete positions are sorted per file, so the in-range window is two binary
  // searches — the cost is O(deletes in range), not O(deletes in table).
  std::vector<uint8_t> keep(static_cast<size_t>(num_rows), 1u);
  bool any_deleted = false;

  for (auto const& run : layout) {
    auto it = _delete_data->positional_deletes.find(run.data_file_path);
    if (it == _delete_data->positional_deletes.end() || it->second.empty()) { continue; }

    auto const& delete_positions = it->second;
    auto const run_end           = run.file_row_offset + run.num_rows;
    auto lo =
      std::lower_bound(delete_positions.begin(), delete_positions.end(), run.file_row_offset);
    auto hi = std::lower_bound(lo, delete_positions.end(), run_end);

    for (auto pos = lo; pos != hi; ++pos) {
      auto const batch_row = run.batch_row_offset + (*pos - run.file_row_offset);
      // A position outside the batch would corrupt neighbouring rows rather than fail, so
      // this stays a guard rather than an assert: a bad offset drops the row silently.
      if (batch_row < 0 || batch_row >= num_rows) {
        throw std::out_of_range("[iceberg] positional_delete_filter: delete position " +
                                std::to_string(*pos) + " in '" + run.data_file_path +
                                "' maps outside the decoded batch — the batch row layout does not "
                                "describe the decoded table");
      }
      keep[static_cast<size_t>(batch_row)] = 0u;
      any_deleted                          = true;
    }
  }

  if (!any_deleted) { return tbl; }

  // Copy to GPU as a BOOL8 column.
  auto bool_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                static_cast<cudf::size_type>(num_rows),
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
  RMM_CUDA_TRY(cudaMemcpyAsync(bool_col->mutable_view().data<uint8_t>(),
                               keep.data(),
                               static_cast<size_t>(num_rows) * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               stream.value()));
  // `keep` is host memory feeding an async copy — it must outlive the copy. Synchronizing here
  // is the simple correct choice; the alternative (a pinned staging buffer tied to the stream)
  // is a later optimization, and delete masks are one byte per row of a single batch.
  stream.synchronize();

  return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream, mr);
}

}  // namespace sirius::op::scan
