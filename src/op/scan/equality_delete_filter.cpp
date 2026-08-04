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
#include <op/scan/iceberg_metadata_reader.hpp>

namespace sirius::op::scan {

equality_delete_filter::equality_delete_filter(std::shared_ptr<const IcebergDeleteData> delete_data,
                                               size_t group_index,
                                               std::vector<cudf::size_type> data_key_indices)
  : _delete_data(std::move(delete_data)),
    _group_index(group_index),
    _data_key_indices(std::move(data_key_indices))
{
}

std::unique_ptr<cudf::table> equality_delete_filter::apply(std::unique_ptr<cudf::table> tbl,
                                                           std::string const& data_file_path,
                                                           int64_t /*first_row*/,
                                                           rmm::cuda_stream_view stream)
{
  auto const n_rows = tbl->num_rows();
  if (n_rows == 0) { return tbl; }

  // Sequence number filtering: per Iceberg spec, equality deletes only apply
  // to data files whose sequence number is strictly LOWER than the delete's.
  // Each group has exactly one sequence number (grouped by schema + seq).
  auto const& group = _delete_data->equality_delete_groups[_group_index];
  auto seq_it       = _delete_data->data_file_sequence_numbers.find(data_file_path);
  if (group.sequence_number > 0 && seq_it != _delete_data->data_file_sequence_numbers.end() &&
      seq_it->second > 0 && seq_it->second >= group.sequence_number) {
    // Data file's sequence number >= delete's — deletes don't apply.
    SIRIUS_LOG_DEBUG(
      "[equality_delete_filter] Skipping group (data_seq={} >= delete_seq={}) "
      "for file '{}'",
      seq_it->second,
      group.sequence_number,
      data_file_path);
    return tbl;
  }

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

  auto build_indices = group.hash_join->left_join(data_key_view, stream);

  // Anti-join mask entirely on GPU — no host roundtrip.
  auto bool_col = make_anti_join_mask(*build_indices, n_rows, stream);

  return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream);
}

}  // namespace sirius::op::scan
