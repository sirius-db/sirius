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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::scan_manager {

namespace {

// Row groups per parse range. Smaller ranges balance load and keep the read
// queue full across the worker pool; larger ranges cut per-range dispatch
// overhead. Overridable via SIRIUS_METADATA_PARSE_CHUNK for tuning.
// A `parse range` (row group range) is the unit of work for a scan manager thread
// providing splits.
constexpr std::size_t kDefaultParseChunkRowGroups = 8;

std::size_t parse_chunk_row_groups()
{
  if (char const* env = std::getenv("SIRIUS_METADATA_PARSE_CHUNK")) {
    try {
      auto const v = std::stoull(env);
      if (v > 0) { return static_cast<std::size_t>(v); }
    } catch (...) {
      // Unparsable value → fall through to the default.
    }
  }
  return kDefaultParseChunkRowGroups;
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
  if (_io_ctx == nullptr) {
    throw std::invalid_argument("duckdb_native_split_provider: io_ctx must be non-null");
  }
  if (_scan_info->db_path.empty()) {
    throw std::invalid_argument(
      "duckdb_native_split_provider: scan_info.db_path must be non-empty");
  }

  _plan = op::scan::prepare_duckdb_native_walk(*_scan_info->storage,
                                               *_scan_info->context,
                                               _scan_info->projected_cols,
                                               _scan_info->projected_types);
  if (!_plan.viable) {
    SPDLOG_DEBUG("[duckdb_native_split_provider] non-viable: {}", _plan.viability_failure_reason);
    throw std::runtime_error("duckdb-native scan rejected query: " +
                             _plan.viability_failure_reason);
  }
  if (_plan.n_row_groups == 0) {
    // Zero ranges would close the connector empty; keep the historical refuse so
    // an empty/unreadable table routes to DuckDB CPU.
    throw std::runtime_error("duckdb-native scan rejected query: no row groups in table");
  }

  // Mint the .db io_object once per query; the scan task threads it onto every
  // split so reads of .db blocks go through sirius_ioctx::host_read.
  _db_io_object = _io_ctx->create_io_object(_scan_info->db_path);

  _chunk_row_groups = parse_chunk_row_groups();
  _num_ranges       = (_plan.n_row_groups + _chunk_row_groups - 1) / _chunk_row_groups;
}

duckdb_native_split_provider::~duckdb_native_split_provider() = default;

bool duckdb_native_split_provider::has_more_splits() const
{
  return _next_range_idx.load(std::memory_order_relaxed) < _num_ranges;
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
duckdb_native_split_provider::next_split_provider()
{
  auto const idx = _next_range_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _num_ranges) { return {}; }

  auto const rg_begin = idx * _chunk_row_groups;
  auto const rg_end   = std::min(rg_begin + _chunk_row_groups, _plan.n_row_groups);

  // The walk reads `_plan` (const after the ctor) and builds its own
  // QueryContext, so disjoint ranges run concurrently on the worker pool. The
  // provider outlives every thunk it hands out (split_provider contract).
  return [this, rg_begin, rg_end]() -> std::vector<std::unique_ptr<op::operator_data>> {
    auto range = op::scan::walk_duckdb_native_row_group_range(_plan, rg_begin, rg_end);
    if (!range.viable) {
      // Rides the worker_state -> connector.close(eptr) channel; the scan
      // operator rethrows it. Compression-unsupported is a hard error for now;
      // type-based fallback was already handled by the ctor's prepare step.
      throw std::runtime_error("duckdb-native scan rejected query: " +
                               range.viability_failure_reason);
    }

    auto payload          = std::make_unique<split_payload>();
    payload->scan_info    = _scan_info;
    payload->io_ctx       = _io_ctx;
    payload->db_io_object = _db_io_object;
    payload->row_groups   = std::move(range.row_groups);

    std::vector<std::unique_ptr<op::operator_data>> out;
    out.push_back(std::move(payload));
    return out;
  };
}

}  // namespace sirius::scan_manager
