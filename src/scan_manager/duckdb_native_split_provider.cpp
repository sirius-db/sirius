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

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sirius::scan_manager {

namespace {

// cudf strings columns use int32 offsets, so the per-column decoded byte budget
// for a single scan output must stay under INT32_MAX. 64 MB headroom covers
// max_string_length being an upper bound + offsets/validity allocations.
constexpr std::size_t VARCHAR_BYTE_CAP =
  (static_cast<std::size_t>(1) << 31) - (static_cast<std::size_t>(64) << 20);

std::size_t rg_varchar_bytes_for_col(const op::scan::duckdb_row_group_metadata& rg,
                                     std::size_t col_idx)
{
  std::size_t total = 0;
  for (const auto& seg : rg.columns[col_idx].data_segments) {
    std::uint32_t per_row = seg.max_string_length == 0
                              ? op::scan::VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES
                              : seg.max_string_length;
    total += static_cast<std::size_t>(seg.segment_count) * static_cast<std::size_t>(per_row);
  }
  return total;
}

std::vector<duckdb_native_split_provider::row_group_batch> partition_row_groups_into_batches(
  const std::vector<op::scan::duckdb_row_group_metadata>& row_groups,
  std::size_t approximate_batch_size,
  const std::vector<sirius::logical_type>& projected_types)
{
  std::vector<duckdb_native_split_provider::row_group_batch> batches;
  if (row_groups.empty()) return batches;

  const std::size_t num_cols = projected_types.size();
  std::vector<bool> is_varchar(num_cols, false);
  for (std::size_t c = 0; c < num_cols; ++c) {
    is_varchar[c] = projected_types[c].is_varchar();
  }
  const bool any_varchar = std::any_of(is_varchar.begin(), is_varchar.end(), [](bool b) { return b; });

  // Fast path: no batch-bytes cap AND no varchar columns → single batch.
  if (approximate_batch_size == 0 && !any_varchar) {
    batches.push_back({0, row_groups.size()});
    return batches;
  }

  std::size_t batch_first = 0;
  std::size_t batch_bytes = 0;
  std::vector<std::size_t> col_bytes(num_cols, 0);

  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    const std::size_t this_rg_bytes = row_groups[i].decoded_bytes_budget;
    std::vector<std::size_t> this_rg_col_bytes(num_cols, 0);
    if (any_varchar) {
      for (std::size_t c = 0; c < num_cols; ++c) {
        if (is_varchar[c]) this_rg_col_bytes[c] = rg_varchar_bytes_for_col(row_groups[i], c);
      }
    }

    // Decide whether to close the in-progress batch before adding this RG.
    // Cap (c): only fire when the batch is non-empty, so a single oversized RG
    // still makes forward progress as its own singleton batch.
    if (i > batch_first) {
      const bool would_exceed_total =
        (approximate_batch_size > 0) && (batch_bytes + this_rg_bytes > approximate_batch_size);

      bool would_exceed_varchar = false;
      if (any_varchar) {
        for (std::size_t c = 0; c < num_cols; ++c) {
          if (is_varchar[c] && col_bytes[c] + this_rg_col_bytes[c] > VARCHAR_BYTE_CAP) {
            would_exceed_varchar = true;
            break;
          }
        }
      }

      if (would_exceed_total || would_exceed_varchar) {
        batches.push_back({batch_first, i - batch_first});
        batch_first = i;
        batch_bytes = 0;
        std::fill(col_bytes.begin(), col_bytes.end(), 0);
      }
    }

    batch_bytes += this_rg_bytes;
    if (any_varchar) {
      for (std::size_t c = 0; c < num_cols; ++c) col_bytes[c] += this_rg_col_bytes[c];
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

  _batches = partition_row_groups_into_batches(
    _metadata.row_groups, _scan_info->approximate_batch_size, _scan_info->projected_types);
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
