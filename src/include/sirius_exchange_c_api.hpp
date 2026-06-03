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

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define SIRIUS_EXCHANGE_C_API_EXPORT __declspec(dllexport)
#else
#define SIRIUS_EXCHANGE_C_API_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

struct sirius_exchange_artifact_handle;

struct sirius_packed_partition_view {
  uint64_t partition_id;
  uint64_t staging_offset;
  uint64_t packed_size;
  const uint8_t* metadata_ptr;
  uint64_t metadata_len;
  int32_t num_rows;
  uint64_t overflow_gpu_addr;
};

struct sirius_packed_broadcast_entry_view {
  uint64_t entry_id;
  uint64_t staging_offset;
  uint64_t packed_size;
  const uint8_t* metadata_ptr;
  uint64_t metadata_len;
  int32_t num_rows;
  uint64_t overflow_gpu_addr;
};

SIRIUS_EXCHANGE_C_API_EXPORT const char* sirius_exchange_last_error();
SIRIUS_EXCHANGE_C_API_EXPORT int sirius_finalize_exchange_tables_direct();
SIRIUS_EXCHANGE_C_API_EXPORT uint64_t sirius_begin_exchange_capture(
  uint64_t num_partitions,
  const int32_t* column_indices,
  size_t num_columns,
  const int32_t* projection_indices,
  size_t num_projection_indices,
  uint32_t capture_mode);
SIRIUS_EXCHANGE_C_API_EXPORT sirius_exchange_artifact_handle* sirius_take_exchange_artifact();
SIRIUS_EXCHANGE_C_API_EXPORT void sirius_exchange_artifact_destroy(
  sirius_exchange_artifact_handle* handle);
SIRIUS_EXCHANGE_C_API_EXPORT uint64_t sirius_exchange_artifact_staging_base(
  const sirius_exchange_artifact_handle* handle);
SIRIUS_EXCHANGE_C_API_EXPORT uint64_t sirius_exchange_artifact_partition_count(
  const sirius_exchange_artifact_handle* handle);
SIRIUS_EXCHANGE_C_API_EXPORT int sirius_exchange_artifact_get_partition(
  const sirius_exchange_artifact_handle* handle,
  size_t index,
  sirius_packed_partition_view* out);
SIRIUS_EXCHANGE_C_API_EXPORT uint64_t sirius_exchange_artifact_broadcast_count(
  const sirius_exchange_artifact_handle* handle);
SIRIUS_EXCHANGE_C_API_EXPORT int sirius_exchange_artifact_get_broadcast_entry(
  const sirius_exchange_artifact_handle* handle,
  size_t index,
  sirius_packed_broadcast_entry_view* out);
SIRIUS_EXCHANGE_C_API_EXPORT int sirius_register_packed_table_direct(
  const char* table_name,
  uint64_t gpu_addr,
  uint64_t gpu_size,
  const uint8_t* metadata_ptr,
  size_t metadata_len,
  const int32_t* projection_indices,
  size_t num_projection_indices,
  int32_t* out_num_cols,
  int32_t* out_num_rows);

}  // extern "C"
