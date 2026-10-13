/*
 * Copyright 2026, Sirius Contributors.
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

#include "data/data_batch_utils.hpp"

#include "compression/compressed_representation.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/column_metadata.hpp>

#include <vector>

namespace sirius {

namespace {

/// Row count of the first column in @p columns, or `std::nullopt` when there are no columns.
std::optional<std::size_t> first_column_rows(
  std::vector<cucascade::memory::column_metadata> const& columns)
{
  if (columns.empty()) { return std::nullopt; }
  return static_cast<std::size_t>(columns.front().num_rows);
}

}  // namespace

std::optional<std::size_t> representation_num_rows(cucascade::idata_representation const& data)
{
  if (auto const* gpu = dynamic_cast<cucascade::gpu_table_representation const*>(&data)) {
    auto const view = gpu->get_table_view();
    if (view.num_columns() == 0) { return std::nullopt; }
    return static_cast<std::size_t>(view.num_rows());
  }
  if (auto const* host = dynamic_cast<cucascade::host_data_representation const*>(&data)) {
    auto const& table = host->get_host_table();
    if (!table) { return std::nullopt; }
    return first_column_rows(table->columns);
  }
  if (auto const* disk = dynamic_cast<cucascade::disk_data_representation const*>(&data)) {
    return first_column_rows(disk->get_disk_table().columns);
  }
  if (auto const* compressed = dynamic_cast<compressed_host_representation const*>(&data)) {
    return static_cast<std::size_t>(compressed->num_rows());
  }
  if (auto const* compressed = dynamic_cast<compressed_device_representation const*>(&data)) {
    return static_cast<std::size_t>(compressed->num_rows());
  }
  return std::nullopt;
}

batch_rows_and_bytes get_batch_rows_and_bytes(cucascade::data_batch const& batch)
{
  auto const read_only = batch.to_read_only();
  auto const* data     = read_only.get_data();
  if (data == nullptr) { return {}; }
  return {.rows  = representation_num_rows(*data),
          .bytes = data->get_uncompressed_data_size_in_bytes()};
}

}  // namespace sirius
