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

#include <gpu_buffer_manager.hpp>
#include <last_gpu_buffers.hpp>
#include <sirius_exchange_c_api.hpp>
#include <sirius_extension.hpp>

#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace {

thread_local std::string g_sirius_exchange_last_error;

void set_last_error(std::string message) {
  g_sirius_exchange_last_error = std::move(message);
}

template <typename Fn>
auto with_error_boundary(Fn&& fn) -> decltype(fn()) {
  try {
    g_sirius_exchange_last_error.clear();
    return fn();
  } catch (const std::exception& e) {
    set_last_error(e.what());
  } catch (...) {
    set_last_error("unknown sirius exchange error");
  }
  using return_type = decltype(fn());
  if constexpr (std::is_pointer_v<return_type>) {
    return nullptr;
  } else {
    return return_type{};
  }
}

}  // namespace

extern "C" {

struct sirius_exchange_artifact_handle {
  uint64_t session_id = 0;
  std::unique_ptr<duckdb::ExchangeSession> session;
};

const char* sirius_exchange_last_error() {
  return g_sirius_exchange_last_error.c_str();
}

int sirius_finalize_exchange_tables_direct() {
  return with_error_boundary([]() -> int {
    duckdb::SiriusExtension::EnsureExchangeBufferManager();
    duckdb::GPUBufferManager::GetInstance().finalizeExchangeTables();
    return 1;
  });
}

uint64_t sirius_begin_exchange_capture(
  uint64_t num_partitions,
  const int32_t* column_indices,
  size_t num_columns) {
  return with_error_boundary([&]() -> uint64_t {
    auto& lgb = duckdb::LastGPUBuffers::GetInstance();
    auto session_id = lgb.BeginSession();
    if (num_partitions > 0 && column_indices && num_columns > 0) {
      std::vector<int> cols;
      cols.reserve(num_columns);
      for (size_t i = 0; i < num_columns; i++) {
        cols.push_back(static_cast<int>(column_indices[i]));
      }
      lgb.SetPartitionConfig(static_cast<int>(num_partitions), std::move(cols));
      SIRIUS_LOG_INFO("[sirius_begin_exchange_capture] session={} num_partitions={} cols={}",
                      session_id, num_partitions, num_columns);
    } else {
      SIRIUS_LOG_INFO("[sirius_begin_exchange_capture] session={} broadcast capture", session_id);
    }
    return session_id;
  });
}

sirius_exchange_artifact_handle* sirius_take_exchange_artifact() {
  return with_error_boundary([]() -> sirius_exchange_artifact_handle* {
    auto session = duckdb::LastGPUBuffers::GetInstance().TakeSession();
    if (!session) {
      return nullptr;
    }
    auto handle = std::make_unique<sirius_exchange_artifact_handle>();
    handle->session_id = session->session_id;
    duckdb::LastGPUBuffers::GetInstance().RegisterInflightArtifact(
      session->session_id, session->staging_offset);
    SIRIUS_LOG_INFO("[sirius_take_exchange_artifact] took session {} staging_base=0x{:x} staging_offset={}",
                    session->session_id, session->staging_lease_base, session->staging_offset);
    handle->session = std::move(session);
    return handle.release();
  });
}

void sirius_exchange_artifact_destroy(sirius_exchange_artifact_handle* handle) {
  if (!handle) {
    return;
  }
  duckdb::LastGPUBuffers::GetInstance().ReleaseInflightArtifact(handle->session_id);
  SIRIUS_LOG_INFO("[sirius_exchange_artifact_destroy] session {}", handle->session_id);
  delete handle;
}

uint64_t sirius_exchange_artifact_staging_base(const sirius_exchange_artifact_handle* handle) {
  return with_error_boundary([&]() -> uint64_t {
    if (!handle || !handle->session) {
      return 0;
    }
    return static_cast<uint64_t>(handle->session->staging_lease_base);
  });
}

uint64_t sirius_exchange_artifact_partition_count(const sirius_exchange_artifact_handle* handle) {
  return with_error_boundary([&]() -> uint64_t {
    if (!handle || !handle->session) {
      return 0;
    }
    return static_cast<uint64_t>(handle->session->packed_partitions.size());
  });
}

int sirius_exchange_artifact_get_partition(
  const sirius_exchange_artifact_handle* handle,
  size_t index,
  sirius_packed_partition_view* out) {
  return with_error_boundary([&]() -> int {
    if (!handle || !handle->session || !out) {
      return 0;
    }
    if (index >= handle->session->packed_partitions.size()) {
      return 0;
    }
    const auto& part = handle->session->packed_partitions[index];
    out->partition_id = static_cast<uint64_t>(index);
    out->staging_offset = static_cast<uint64_t>(part.staging_offset);
    out->packed_size = static_cast<uint64_t>(part.packed_size);
    out->metadata_ptr = part.metadata && !part.metadata->empty() ? part.metadata->data() : nullptr;
    out->metadata_len = part.metadata ? static_cast<uint64_t>(part.metadata->size()) : 0;
    out->num_rows = part.num_rows;
    out->overflow_gpu_addr = static_cast<uint64_t>(part.overflow_gpu_addr);
    return 1;
  });
}

uint64_t sirius_exchange_artifact_broadcast_count(const sirius_exchange_artifact_handle* handle) {
  return with_error_boundary([&]() -> uint64_t {
    if (!handle || !handle->session) {
      return 0;
    }
    return static_cast<uint64_t>(handle->session->packed_broadcast_entries.size());
  });
}

int sirius_exchange_artifact_get_broadcast_entry(
  const sirius_exchange_artifact_handle* handle,
  size_t index,
  sirius_packed_broadcast_entry_view* out) {
  return with_error_boundary([&]() -> int {
    if (!handle || !handle->session || !out) {
      return 0;
    }
    if (index >= handle->session->packed_broadcast_entries.size()) {
      return 0;
    }
    const auto& entry = handle->session->packed_broadcast_entries[index];
    out->entry_id = static_cast<uint64_t>(index);
    out->staging_offset = static_cast<uint64_t>(entry.staging_offset);
    out->packed_size = static_cast<uint64_t>(entry.packed_size);
    out->metadata_ptr = entry.metadata && !entry.metadata->empty() ? entry.metadata->data() : nullptr;
    out->metadata_len = entry.metadata ? static_cast<uint64_t>(entry.metadata->size()) : 0;
    out->num_rows = entry.num_rows;
    out->overflow_gpu_addr = static_cast<uint64_t>(entry.overflow_gpu_addr);
    return 1;
  });
}

}  // extern "C"
