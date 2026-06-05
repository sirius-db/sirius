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

#include "io/io_context.hpp"
#include "io/types.hpp"
#include "log/logging.hpp"

#include <stdexcept>
#include <utility>

// Work placement note: diverges from `parquet_split_provider`. For parquet a
// scan-manager worker thread parses metadata and emits splits, because each
// parquet file means real IO. For duckdb-native the metadata is already in
// memory, so the walker + batch partitioning runs here in the ctor on the
// query-executor thread; worker threads only claim pre-built batches.

namespace sirius::scan_manager {

namespace {

std::vector<duckdb_native_split_provider::row_group_batch> partition_row_groups_into_batches(
  const std::vector<op::scan::duckdb_row_group_metadata>& row_groups,
  std::size_t approximate_batch_size,
  const std::vector<sirius::logical_type>& projected_types)
{
  std::vector<duckdb_native_split_provider::row_group_batch> batches;
  if (row_groups.empty()) return batches;

  const std::size_t num_cols = projected_types.size();
  std::vector<bool> is_varchar(num_cols, false);
  bool any_varchar = false;
  for (std::size_t c = 0; c < num_cols; ++c) {
    is_varchar[c] = projected_types[c].is_varchar();
    any_varchar   = any_varchar || is_varchar[c];
  }

  // No caps in play → single batch.
  if (approximate_batch_size == 0 && !any_varchar) {
    batches.push_back({0, row_groups.size()});
    return batches;
  }

  std::size_t batch_first = 0;
  std::size_t batch_bytes = 0;
  std::vector<std::size_t> col_bytes(num_cols, 0);

  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    const auto& rg                  = row_groups[i];
    const std::size_t this_rg_bytes = rg.decoded_bytes_budget;

    // Only close a non-empty batch; an over-cap row group always lands in
    // the current batch as its first member (singleton if it's alone).
    if (i > batch_first) {
      const bool would_exceed_total =
        (approximate_batch_size > 0) && (batch_bytes + this_rg_bytes > approximate_batch_size);

      bool would_exceed_varchar = false;
      if (any_varchar) {
        for (std::size_t c = 0; c < num_cols; ++c) {
          if (is_varchar[c] &&
              col_bytes[c] + rg.varchar_bytes_per_col[c] >= op::scan::kCudfInt32StringsThreshold) {
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
      for (std::size_t c = 0; c < num_cols; ++c) {
        col_bytes[c] += rg.varchar_bytes_per_col[c];
      }
    }
  }
  if (batch_first < row_groups.size()) {
    batches.push_back({batch_first, row_groups.size() - batch_first});
  }
  return batches;
}

}  // namespace

duckdb_native_split_provider::duckdb_native_split_provider(
  op::scan::duckdb_native_scan_info info, std::shared_ptr<sirius::io::sirius_ioctx> io_ctx)
  : _scan_info(std::make_shared<op::scan::duckdb_native_scan_info const>(std::move(info))),
    _io_ctx(std::move(io_ctx))
{
  if (_scan_info->storage == nullptr) {
    throw std::invalid_argument("duckdb_native_split_provider: scan_info.storage must be non-null");
  }
  if (_scan_info->context == nullptr) {
    throw std::invalid_argument("duckdb_native_split_provider: scan_info.context must be non-null");
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
    // Empty splits would hang the pipeline on the FULL barrier. The throw
    // surfaces as a query error and the extension's transparent fallback
    // routes to DuckDB CPU.
    SPDLOG_DEBUG("[duckdb_native_split_provider] non-viable: {}",
                 _metadata.viability_failure_reason);
    throw std::runtime_error("duckdb-native scan rejected query: " +
                             _metadata.viability_failure_reason);
  }

  // Mint the .db io_object once per query when the manager exposes sirius_ioctx AND the
  // database is file-backed. The scan task threads this onto every split so reads of .db
  // blocks go through sirius_ioctx::host_read instead of DuckDB's BufferManager.
  //
  // In-memory databases (":memory:") have no file to open, and some environments lack
  // working io_uring / O_DIRECT. In those cases leave _db_io_object null so the decoder
  // reads blocks via DuckDB's BufferManager (read_block_payload's fallback). This keeps
  // the GPU-native scan — now the default path — robust everywhere instead of hard-failing.
  if (_io_ctx && !_scan_info->db_path.empty() && _scan_info->db_path != ":memory:") {
    try {
      _db_io_object = _io_ctx->create_io_object(_scan_info->db_path);
    } catch (const std::exception& e) {
      SPDLOG_DEBUG("[duckdb_native_split_provider] sirius_io unavailable for '{}' ({}); "
                   "falling back to DuckDB BufferManager for block reads",
                   _scan_info->db_path,
                   e.what());
      _db_io_object = nullptr;
    }
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
  // `_metadata.row_groups[i]` is moved out — batches are disjoint and each
  // batch is claimed exactly once, so the source entry is never read again.
  return [this, batch]() -> std::vector<std::unique_ptr<op::operator_data>> {
    auto payload          = std::make_unique<split_payload>();
    payload->scan_info    = _scan_info;
    payload->io_ctx       = _io_ctx;
    payload->db_io_object = _db_io_object;
    payload->row_groups.reserve(batch.count);
    for (std::size_t i = batch.first_idx; i < batch.first_idx + batch.count; ++i) {
      payload->row_groups.push_back(std::move(_metadata.row_groups[i]));
    }
    std::vector<std::unique_ptr<op::operator_data>> out;
    out.push_back(std::move(payload));
    return out;
  };
}

}  // namespace sirius::scan_manager
