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

#pragma once

#include "sirius_config.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace duckdb {

/// Lightweight exchange table representation.
/// Replaces GPUIntermediateRelation for the exchange path — holds only
/// pending cudf::table_views (from cudf::unpack) and a finalized cudf::table
/// (from cudf::concatenate). No legacy GPUColumn vector.
struct ExchangeTable {
  /// Finalized cudf::table (after finalize_pending_views).
  /// Shared ownership allows multiple scan operators in the same fragment
  /// to reuse the same read-only GPU table.
  std::shared_ptr<cudf::table> finalized;

  /// Pending table_views accumulated from multiple senders (zero-copy).
  /// Each view points into a packed/staging buffer that stays alive until
  /// finalize_pending_views() concatenates them.
  std::vector<cudf::table_view> pending_views;

  /// Keeps metadata buffers alive — cudf::unpack creates table_views that
  /// reference the host metadata buffer internally (for STRING child column pointers).
  std::vector<std::string> pending_metadata;

  /// GPU data pointers for re-unpack during finalization.
  std::vector<uint8_t*> pending_gpu_ptrs;

  /// Per-view projection indices for re-unpack during finalization.
  std::vector<std::vector<int32_t>> pending_projection_indices;

  /// Total rows across all pending views (before finalization).
  size_t pending_total_rows = 0;

  [[nodiscard]] bool has_data() const { return finalized || pending_total_rows > 0; }

  [[nodiscard]] size_t num_rows() const
  {
    return finalized ? static_cast<size_t>(finalized->num_rows()) : pending_total_rows;
  }

  /// Concatenate all pending views into a single owned cudf::table.
  /// Always uses cudf::concatenate (even for single view) because
  /// cudf::table(column_view) deep copy fails for string columns from cudf::unpack.
  void finalize_pending_views();
};

/// Manages exchange staging buffers and the exchange table registry.
/// Owned by SiriusContext. Allocates staging from the cudf device resource
/// (which is already set to cuCascade by sirius_memory_reservation_manager).
///
/// For the C API edge case (sirius_exchange_c_api.cpp), provides a
/// process-level accessor via GetActive() (same pattern as LastGPUBuffers).
class ExchangeMemoryManager {
 public:
  explicit ExchangeMemoryManager(const sirius::exchange_params& params);
  ~ExchangeMemoryManager();

  ExchangeMemoryManager(const ExchangeMemoryManager&)            = delete;
  ExchangeMemoryManager& operator=(const ExchangeMemoryManager&) = delete;

  /// Process-level accessor for C API (not a singleton — SiriusContext owns lifetime).
  static ExchangeMemoryManager* GetActive();

  /// Staging buffer access. Returns (address, size).
  [[nodiscard]] std::pair<uintptr_t, size_t> GetSendStaging() const;
  [[nodiscard]] std::pair<uintptr_t, size_t> GetRecvStaging() const;

  /// Register a packed cudf buffer from NIXL. Stores metadata, calls cudf::unpack,
  /// pushes the resulting table_view to pending_views.
  void registerExternalTablePacked(const std::string& table_name,
                                   uint8_t* gpu_data,
                                   size_t gpu_size,
                                   std::string metadata,
                                   const std::vector<int32_t>& projection_indices,
                                   int& out_num_cols,
                                   int& out_num_rows);

  /// Finalize a single exchange table (cudf::concatenate pending views).
  void finalizeExchangeTable(const std::string& table_name);

  /// Finalize all exchange tables with __EXCH_ prefix.
  void finalizeExchangeTables();

  /// Look up an exchange table by name. Returns nullptr if not found.
  [[nodiscard]] std::shared_ptr<ExchangeTable> findTable(const std::string& name) const;

 private:
  void* send_staging_ptr_  = nullptr;
  void* recv_staging_ptr_  = nullptr;
  size_t send_staging_size_ = 0;
  size_t recv_staging_size_ = 0;

  std::map<std::string, std::shared_ptr<ExchangeTable>> tables_;
  mutable std::mutex mutex_;

  static std::atomic<ExchangeMemoryManager*> active_;
};

}  // namespace duckdb
