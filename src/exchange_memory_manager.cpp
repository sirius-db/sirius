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

#include "exchange_memory_manager.hpp"
#include "last_gpu_buffers.hpp"
#include "log/logging.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <stdexcept>

namespace duckdb {

// ---------------------------------------------------------------------------
// ExchangeTable
// ---------------------------------------------------------------------------

void ExchangeTable::finalize_pending_views()
{
  if (pending_views.empty()) { return; }

  SIRIUS_LOG_INFO("[finalize_pending_views] finalizing {} pending views, {} total rows",
                  pending_views.size(), pending_total_rows);

  // Re-unpack from stored metadata + gpu_ptrs to get fresh table_views.
  // The stored pending_views may have corrupted column_view pointers if
  // the vector reallocated.
  if (pending_gpu_ptrs.size() == pending_views.size() &&
      pending_metadata.size() == pending_views.size()) {
    pending_views.clear();
    for (size_t i = 0; i < pending_metadata.size(); i++) {
      auto* md_ptr = reinterpret_cast<const uint8_t*>(pending_metadata[i].data());
      auto view    = cudf::unpack(md_ptr, pending_gpu_ptrs[i]);
      if (i < pending_projection_indices.size() && !pending_projection_indices[i].empty()) {
        std::vector<cudf::column_view> projected;
        projected.reserve(pending_projection_indices[i].size());
        for (auto idx : pending_projection_indices[i]) {
          projected.push_back(view.column(idx));
        }
        view = cudf::table_view(projected);
      }
      pending_views.push_back(view);
    }
  }

  auto merged = cudf::concatenate(pending_views);
  finalized   = std::shared_ptr<cudf::table>(merged.release());

  pending_views.clear();
  pending_metadata.clear();
  pending_gpu_ptrs.clear();
  pending_projection_indices.clear();
  if (finalized) {
    pending_total_rows = static_cast<size_t>(finalized->num_rows());
  }
}

// ---------------------------------------------------------------------------
// ExchangeMemoryManager
// ---------------------------------------------------------------------------

std::atomic<ExchangeMemoryManager*> ExchangeMemoryManager::active_{nullptr};

namespace {

cudf::table_view apply_projection_indices(cudf::table_view view,
                                          const std::vector<int32_t>& projection_indices)
{
  if (projection_indices.empty()) { return view; }

  std::vector<cudf::column_view> projected;
  projected.reserve(projection_indices.size());
  for (auto idx : projection_indices) {
    if (idx < 0 || idx >= view.num_columns()) {
      throw std::runtime_error("projection index " + std::to_string(idx) +
                               " out of range for packed table with " +
                               std::to_string(view.num_columns()) + " columns");
    }
    projected.push_back(view.column(idx));
  }
  return cudf::table_view(projected);
}

}  // namespace

ExchangeMemoryManager::ExchangeMemoryManager(const sirius::exchange_params& params)
{
  // cudf device resource is already set to cuCascade by sirius_memory_reservation_manager.
  auto* mr        = rmm::mr::get_current_device_resource();
  send_staging_ptr_  = mr->allocate(rmm::cuda_stream_view{}, params.send_staging_size);
  recv_staging_ptr_  = mr->allocate(rmm::cuda_stream_view{}, params.recv_staging_size);
  send_staging_size_ = params.send_staging_size;
  recv_staging_size_ = params.recv_staging_size;

  SIRIUS_LOG_INFO("[ExchangeMemoryManager] allocated send staging: 0x{:x} ({} MB), "
                  "recv staging: 0x{:x} ({} MB)",
                  reinterpret_cast<uintptr_t>(send_staging_ptr_),
                  send_staging_size_ / (1024 * 1024),
                  reinterpret_cast<uintptr_t>(recv_staging_ptr_),
                  recv_staging_size_ / (1024 * 1024));

  // Tell LastGPUBuffers about the send staging address so the result collector
  // knows where to pack GPU data via cudf::chunked_pack.
  LastGPUBuffers::GetInstance().SetStagingBuffer(
    reinterpret_cast<uintptr_t>(send_staging_ptr_), send_staging_size_);

  active_.store(this, std::memory_order_release);
}

ExchangeMemoryManager::~ExchangeMemoryManager()
{
  active_.store(nullptr, std::memory_order_release);

  auto* mr = rmm::mr::get_current_device_resource();
  if (send_staging_ptr_) {
    mr->deallocate(rmm::cuda_stream_view{}, send_staging_ptr_, send_staging_size_);
  }
  if (recv_staging_ptr_) {
    mr->deallocate(rmm::cuda_stream_view{}, recv_staging_ptr_, recv_staging_size_);
  }
  SIRIUS_LOG_INFO("[ExchangeMemoryManager] destroyed, staging freed");
}

ExchangeMemoryManager* ExchangeMemoryManager::GetActive()
{
  return active_.load(std::memory_order_acquire);
}

std::pair<uintptr_t, size_t> ExchangeMemoryManager::GetSendStaging() const
{
  return {reinterpret_cast<uintptr_t>(send_staging_ptr_), send_staging_size_};
}

std::pair<uintptr_t, size_t> ExchangeMemoryManager::GetRecvStaging() const
{
  return {reinterpret_cast<uintptr_t>(recv_staging_ptr_), recv_staging_size_};
}

void ExchangeMemoryManager::registerExternalTablePacked(
  const std::string& table_name,
  uint8_t* gpu_data,
  size_t gpu_size,
  std::string metadata,
  const std::vector<int32_t>& projection_indices,
  int& out_num_cols,
  int& out_num_rows)
{
  std::string up = table_name;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = tables_.find(up);
  if (it != tables_.end()) {
    // Append to existing table.
    auto& tbl = it->second;
    tbl->pending_metadata.push_back(std::move(metadata));

    auto& stored_md = tbl->pending_metadata.back();
    auto* md_ptr    = reinterpret_cast<const uint8_t*>(stored_md.data());
    auto raw_view   = cudf::unpack(md_ptr, gpu_data);
    auto view       = apply_projection_indices(raw_view, projection_indices);

    tbl->pending_views.push_back(view);
    tbl->pending_gpu_ptrs.push_back(gpu_data);
    tbl->pending_projection_indices.push_back(projection_indices);
    tbl->pending_total_rows += static_cast<size_t>(view.num_rows());

    out_num_cols = view.num_columns();
    out_num_rows = static_cast<int>(tbl->pending_total_rows);

    SIRIUS_LOG_INFO("[registerExternalTablePacked] appended '{}': {} rows this view, "
                    "{} total ({} views)",
                    up, view.num_rows(), tbl->pending_total_rows,
                    tbl->pending_views.size());
    return;
  }

  // First registration: create ExchangeTable, store metadata, unpack.
  auto tbl = std::make_shared<ExchangeTable>();
  tbl->pending_metadata.push_back(std::move(metadata));

  auto& stored_md = tbl->pending_metadata.back();
  auto* md_ptr    = reinterpret_cast<const uint8_t*>(stored_md.data());
  auto raw_view   = cudf::unpack(md_ptr, gpu_data);
  auto view       = apply_projection_indices(raw_view, projection_indices);

  tbl->pending_views.push_back(view);
  tbl->pending_gpu_ptrs.push_back(gpu_data);
  tbl->pending_projection_indices.push_back(projection_indices);
  tbl->pending_total_rows = static_cast<size_t>(view.num_rows());

  tables_[up] = tbl;

  out_num_cols = view.num_columns();
  out_num_rows = static_cast<int>(view.num_rows());

  SIRIUS_LOG_INFO("[registerExternalTablePacked] registered '{}': {} cols, {} rows",
                  up, view.num_columns(), view.num_rows());
}

void ExchangeMemoryManager::finalizeExchangeTable(const std::string& table_name)
{
  std::string up = table_name;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = tables_.find(up);
  if (it == tables_.end() || !it->second || it->second->pending_views.empty()) { return; }

  SIRIUS_LOG_INFO("[finalizeExchangeTable] finalizing '{}': {} pending views, {} total rows",
                  up, it->second->pending_views.size(), it->second->pending_total_rows);
  try {
    it->second->finalize_pending_views();
    SIRIUS_LOG_INFO("[finalizeExchangeTable] '{}' finalized: {} cols, {} rows",
                    up, it->second->finalized->num_columns(),
                    it->second->finalized->num_rows());
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("[finalizeExchangeTable] failed for '{}': {}", up, e.what());
  }
}

void ExchangeMemoryManager::finalizeExchangeTables()
{
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& [name, tbl] : tables_) {
    if (name.find("__EXCH_") != std::string::npos && tbl && !tbl->pending_views.empty()) {
      SIRIUS_LOG_INFO("[finalizeExchangeTables] finalizing '{}'", name);
      try {
        tbl->finalize_pending_views();
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("[finalizeExchangeTables] failed for '{}': {}", name, e.what());
      }
    }
  }
}

std::shared_ptr<ExchangeTable> ExchangeMemoryManager::findTable(const std::string& name) const
{
  std::string up = name;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = tables_.find(up);
  if (it != tables_.end()) { return it->second; }
  return nullptr;
}

}  // namespace duckdb
