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

#include <stdexcept>
#include <string>
#include <vector>

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
                                                           batch_layout layout,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  auto const n_rows = tbl->num_rows();
  if (n_rows == 0) { return tbl; }

  // Per spec, equality deletes apply only to data files with a strictly lower sequence number.
  // That is a per-file answer, but the mask below is per batch, so a batch mixing files that
  // disagree is refused: keeping them apart is the caller's job.
  auto const& group = _delete_data->equality_delete_groups[_group_index];

  // An unknown file is refused rather than assumed deletable: answering "applies" would delete
  // rows from a data file that may post-date the delete.
  auto applies_to = [&](std::string const& path) {
    if (group.sequence_number <= 0) { return true; }
    auto seq_it = _delete_data->data_file_sequence_numbers.find(path);
    if (seq_it == _delete_data->data_file_sequence_numbers.end()) {
      throw std::invalid_argument(
        "[equality_delete_filter] no sequence number recorded for data file '" + path +
        "'; equality deletes apply only to data files strictly older than the delete, so this "
        "cannot be decided (the manifest-to-scan path translation did not cover this file)");
    }
    if (seq_it->second <= 0) { return true; }
    return seq_it->second < group.sequence_number;
  };

  bool const first_applies = layout.empty() || applies_to(layout.front().data_file_path);
  for (auto const& run : layout) {
    if (applies_to(run.data_file_path) != first_applies) {
      throw std::invalid_argument(
        "[equality_delete_filter] batch mixes data files that disagree on whether delete group "
        "sequence " +
        std::to_string(group.sequence_number) +
        " applies; equality deletes need one sequence number per batch");
    }
  }

  if (!first_applies) {
    SIRIUS_LOG_DEBUG(
      "[equality_delete_filter] Skipping group (delete_seq={}) — batch's data "
      "file(s) are at or above it",
      group.sequence_number);
    return tbl;
  }

  // Missing key column means projection widening never reached the scan. Returning the batch
  // unchanged would drop this group's deletes and hand back deleted rows.
  for (auto idx : _data_key_indices) {
    if (idx >= static_cast<cudf::size_type>(tbl->num_columns())) {
      throw std::invalid_argument("[equality_delete_filter] equality-delete key column index " +
                                  std::to_string(idx) + " is absent from the decoded batch (" +
                                  std::to_string(tbl->num_columns()) +
                                  " columns); the key columns must be appended to the projection");
    }
  }

  auto data_key_view = tbl->select(_data_key_indices);

  auto build_indices = group.hash_join->left_join(data_key_view, stream);

  auto bool_col = make_anti_join_mask(*build_indices, n_rows, stream);

  return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream, mr);
}

}  // namespace sirius::op::scan
