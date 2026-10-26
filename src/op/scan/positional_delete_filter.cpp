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
{
  for (auto const& [path, positions] : delete_data->positional_deletes) {
    auto owner = std::shared_ptr<std::vector<int64_t> const>(delete_data, &positions);
    _delete_sets.emplace(path, std::make_shared<iceberg_delete_set const>(path, std::move(owner)));
  }
}

positional_delete_filter::positional_delete_filter(iceberg_delete_sets delete_sets)
  : _delete_sets(std::move(delete_sets))
{
  for (auto const& [path, set] : _delete_sets) {
    if (!set) throw std::invalid_argument("positional delete filter requires complete delete sets");
  }
}

std::span<int64_t const> positional_delete_filter::positions_for(std::string const& path) const
{
  auto it = _delete_sets.find(path);
  return it == _delete_sets.end() ? std::span<int64_t const>{} : it->second->positions;
}

bool positional_delete_filter::affects(batch_layout layout) const
{
  return std::any_of(layout.begin(), layout.end(), [this](batch_row_run const& run) {
    return !positions_for(run.data_file_path).empty();
  });
}

std::unique_ptr<cudf::table> positional_delete_filter::apply(std::unique_ptr<cudf::table> tbl,
                                                             batch_layout layout,
                                                             ::cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  return apply_positional_visibility(
    std::move(tbl),
    layout,
    [this](std::string const& path) { return positions_for(path); },
    stream,
    mr);
}

}  // namespace sirius::op::scan
