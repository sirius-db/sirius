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

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>

#include <log/logging.hpp>
#include <op/scan/iceberg_delete_filter.hpp>
#include <op/scan/iceberg_equality_delete_mask.hpp>

namespace sirius::op::scan {

equality_delete_filter::equality_delete_filter(std::unique_ptr<cudf::table> delete_key_table,
                                               std::unique_ptr<cudf::distinct_hash_join> hash_join,
                                               std::vector<cudf::size_type> data_key_indices)
  : _delete_key_table(std::move(delete_key_table)),
    _hash_join(std::move(hash_join)),
    _data_key_indices(std::move(data_key_indices))
{
}

std::unique_ptr<cudf::table> equality_delete_filter::apply(std::unique_ptr<cudf::table> tbl,
                                                           std::string const& /*data_file_path*/,
                                                           int64_t /*first_row*/,
                                                           rmm::cuda_stream_view stream)
{
  auto const n_rows = tbl->num_rows();
  if (n_rows == 0) { return tbl; }

  // Verify all key columns are present in this chunk.
  for (auto idx : _data_key_indices) {
    if (idx >= static_cast<cudf::size_type>(tbl->num_columns())) {
      SIRIUS_LOG_WARN("[equality_delete_filter] Key column index {} >= num_columns {}; skipping.",
                      idx,
                      tbl->num_columns());
      return tbl;
    }
  }

  // Project data chunk to the equality key columns.
  auto data_key_view = tbl->select(_data_key_indices);

  auto build_indices = _hash_join->left_join(data_key_view, stream);

  // Anti-join mask entirely on GPU — no host roundtrip.
  auto bool_col = make_anti_join_mask(*build_indices, n_rows, stream);

  return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream);
}

}  // namespace sirius::op::scan
