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

namespace sirius::op::scan {

positional_delete_filter::positional_delete_filter(
  std::shared_ptr<const IcebergDeleteData> delete_data)
  : _delete_data(std::move(delete_data))
{
}

std::unique_ptr<cudf::table> positional_delete_filter::apply(std::unique_ptr<cudf::table> tbl,
                                                             std::string const& data_file_path,
                                                             int64_t first_row,
                                                             rmm::cuda_stream_view stream)
{
  auto it = _delete_data->positional_deletes.find(data_file_path);
  if (it == _delete_data->positional_deletes.end() || it->second.empty()) {
    return tbl;  // No deletes for this file
  }

  auto const& delete_positions = it->second;
  auto const num_rows          = static_cast<int64_t>(tbl->num_rows());
  auto const last_row          = first_row + num_rows;

  // Find the sub-range of delete positions that fall within this batch.
  auto lo = std::lower_bound(delete_positions.begin(), delete_positions.end(), first_row);
  auto hi = std::lower_bound(lo, delete_positions.end(), last_row);

  if (lo == hi) { return tbl; }  // Nothing to delete in this batch

  // Build a keep-mask on the host: true = keep row, false = deleted.
  std::vector<uint8_t> keep(static_cast<size_t>(num_rows), 1u);
  for (auto it2 = lo; it2 != hi; ++it2) {
    keep[static_cast<size_t>(*it2 - first_row)] = 0u;
  }

  // Copy to GPU as a BOOL8 column.
  auto bool_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                                static_cast<cudf::size_type>(num_rows),
                                                cudf::mask_state::UNALLOCATED,
                                                stream);
  RMM_CUDA_TRY(cudaMemcpyAsync(bool_col->mutable_view().data<uint8_t>(),
                               keep.data(),
                               static_cast<size_t>(num_rows) * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               stream.value()));

  return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream);
}

}  // namespace sirius::op::scan
