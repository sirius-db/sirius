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

  // Sequence number filtering: per Iceberg spec, equality deletes only apply
  // to data files whose sequence number is strictly LOWER than the delete's.
  // Each group has exactly one sequence number (grouped by schema + seq).
  //
  // Applicability is per data file, but the mask below is per batch, so a batch mixing files
  // that disagree cannot be served by one mask. Rather than silently applying one file's
  // answer to another's rows, that case is refused — the caller's job is to keep such files
  // out of the same batch. It is unreachable while the planner routes equality deletes to CPU.
  auto const& group = _delete_data->equality_delete_groups[_group_index];

  // A data file whose sequence number we cannot find is REFUSED, not assumed deletable. The
  // lookup is keyed on the path the manifest recorded, and the caller translates each run to
  // that form; a miss therefore means the translation did not cover this file, and answering
  // "the deletes apply" would remove rows whose data file may well post-date the delete. That
  // is the silent-wrong direction, so it throws and takes the runtime fallback instead.
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

  // Verify all key columns are present in this chunk.
  //
  // Returning the batch unchanged here would drop the group's deletes and hand back rows the
  // table deleted — the silent-wrong failure this path exists to prevent. A key column absent
  // from the decoded batch means the planner's projection widening did not reach the scan, so
  // it is a defect in this code, not a table we can serve; throwing turns it into a runtime
  // fallback with correct rows.
  for (auto idx : _data_key_indices) {
    if (idx >= static_cast<cudf::size_type>(tbl->num_columns())) {
      throw std::invalid_argument("[equality_delete_filter] equality-delete key column index " +
                                  std::to_string(idx) + " is absent from the decoded batch (" +
                                  std::to_string(tbl->num_columns()) +
                                  " columns); the key columns must be appended to the projection");
    }
  }

  // Project data chunk to the equality key columns.
  auto data_key_view = tbl->select(_data_key_indices);

  auto build_indices = group.hash_join->left_join(data_key_view, stream);

  // Anti-join mask entirely on GPU — no host roundtrip.
  auto bool_col = make_anti_join_mask(*build_indices, n_rows, stream);

  return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream, mr);
}

}  // namespace sirius::op::scan
