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

#include <cudf/cudf_utils.hpp>
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
                                               rmm::cuda_device_id device,
                                               std::vector<probe> probes)
  : _delete_data(std::move(delete_data)), _device(device), _probes(std::move(probes))
{
}

equality_delete_filter::equality_delete_filter(std::shared_ptr<const IcebergDeleteData> delete_data,
                                               rmm::cuda_device_id device,
                                               size_t group_index,
                                               std::vector<cudf::size_type> data_key_indices)
  : _delete_data(std::move(delete_data)), _device(device)
{
  _probes.push_back({group_index, std::move(data_key_indices)});
}

std::unique_ptr<cudf::table> equality_delete_filter::apply(std::unique_ptr<cudf::table> tbl,
                                                           batch_layout,
                                                           ::cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr)
{
  auto const n_rows = tbl->num_rows();
  if (n_rows == 0 || _probes.empty()) { return tbl; }

  // Every group probes the whole batch and marks its matches in one mask, so the batch is
  // copied once however many groups apply.
  std::unique_ptr<cudf::column> keep;
  for (auto const& [group_index, data_key_indices] : _probes) {
    auto const& group = _delete_data->equality_groups_on(_device).at(group_index);
    auto const keys   = group.delete_table->view();
    if (data_key_indices.size() != static_cast<std::size_t>(keys.num_columns())) {
      throw std::invalid_argument("[equality_delete_filter] " +
                                  std::to_string(data_key_indices.size()) +
                                  " batch key column(s) supplied for a delete group keyed on " +
                                  std::to_string(keys.num_columns()));
    }

    // Missing key column means projection widening never reached the scan. Returning the batch
    // unchanged would drop this group's deletes and hand back deleted rows.
    for (std::size_t k = 0; k < data_key_indices.size(); ++k) {
      auto const idx = data_key_indices[k];
      if (idx < 0 || idx >= static_cast<cudf::size_type>(tbl->num_columns())) {
        throw std::invalid_argument(
          "[equality_delete_filter] equality-delete key column index " + std::to_string(idx) +
          " is absent from the decoded batch (" + std::to_string(tbl->num_columns()) +
          " columns); the key columns must be appended to the projection");
      }
      // The hash join compares raw values; a type mismatch would be a wrong answer or a crash.
      auto const batch_type  = tbl->get_column(idx).type();
      auto const delete_type = keys.column(static_cast<cudf::size_type>(k)).type();
      if (batch_type != delete_type) {
        throw std::invalid_argument(
          "[equality_delete_filter] key column " + std::to_string(k) + " is cuDF type " +
          std::to_string(static_cast<int>(batch_type.id())) + " in the batch but " +
          std::to_string(static_cast<int>(delete_type.id())) + " in the delete group");
      }
    }

    // These allocations are part of this batch's footprint, so they come from the scan's
    // resource -- the default one is invisible to the accounting that drives downgrade decisions.
    auto build_indices = group.hash_join->left_join(tbl->select(data_key_indices), stream, mr);
    if (!keep) {
      keep = make_anti_join_mask(*build_indices, n_rows, stream, mr);
    } else {
      clear_matched_rows(*build_indices, keep->mutable_view(), stream);
    }
  }

  if (count_kept_rows(keep->view(), stream) == n_rows) { return tbl; }
  return sirius::apply_retention_mask(tbl->view(), keep->view(), stream, mr);
}

}  // namespace sirius::op::scan
