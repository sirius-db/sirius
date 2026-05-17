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

#include "scan_manager/duckdb_native_split_provider.hpp"

#include "log/logging.hpp"

#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

namespace {

std::vector<duckdb_native_split_provider::row_group_batch> partition_row_groups_into_batches(
  const std::vector<op::scan::duckdb_row_group_metadata>& row_groups,
  std::size_t approximate_batch_size)
{
  std::vector<duckdb_native_split_provider::row_group_batch> batches;
  if (row_groups.empty()) return batches;

  if (approximate_batch_size == 0) {
    batches.push_back({0, row_groups.size()});
    return batches;
  }

  std::size_t batch_first = 0;
  std::size_t batch_bytes = 0;
  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    batch_bytes += row_groups[i].decoded_bytes_budget;
    if (batch_bytes >= approximate_batch_size) {
      batches.push_back({batch_first, i + 1 - batch_first});
      batch_first = i + 1;
      batch_bytes = 0;
    }
  }
  if (batch_first < row_groups.size()) {
    batches.push_back({batch_first, row_groups.size() - batch_first});
  }
  return batches;
}

}  // namespace

duckdb_native_split_provider::duckdb_native_split_provider(op::scan::duckdb_native_scan_info info)
  : _scan_info(std::make_shared<op::scan::duckdb_native_scan_info const>(std::move(info)))
{
  if (_scan_info->storage == nullptr) {
    throw std::invalid_argument(
      "duckdb_native_split_provider: scan_info.storage must be non-null");
  }
  if (_scan_info->context == nullptr) {
    throw std::invalid_argument(
      "duckdb_native_split_provider: scan_info.context must be non-null");
  }
  if (_scan_info->projected_cols.size() != _scan_info->projected_types.size()) {
    throw std::invalid_argument(
      "duckdb_native_split_provider: projected_cols and projected_types must be parallel");
  }

  _metadata = op::scan::walk_duckdb_native_metadata(*_scan_info->storage,
                                                    *_scan_info->context,
                                                    _scan_info->projected_cols,
                                                    _scan_info->projected_types);
  if (!_metadata.viable) {
    SPDLOG_DEBUG("[duckdb_native_split_provider] non-viable: {}",
                 _metadata.viability_failure_reason);
    _batches.clear();
    return;
  }

  _batches =
    partition_row_groups_into_batches(_metadata.row_groups, _scan_info->approximate_batch_size);
}

duckdb_native_split_provider::~duckdb_native_split_provider() = default;

bool duckdb_native_split_provider::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
duckdb_native_split_provider::next_split_provider()
{
  std::size_t idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _batches.size()) { return {}; }

  row_group_batch batch = _batches[idx];
  return [this, batch]() -> std::vector<std::unique_ptr<op::operator_data>> {
    auto payload       = std::make_unique<split_payload>();
    payload->scan_info = _scan_info;
    payload->row_groups.reserve(batch.count);
    for (std::size_t i = batch.first_idx; i < batch.first_idx + batch.count; ++i) {
      payload->row_groups.push_back(_metadata.row_groups[i]);
    }
    std::vector<std::unique_ptr<op::operator_data>> out;
    out.push_back(std::move(payload));
    return out;
  };
}

}  // namespace sirius::scan_manager
