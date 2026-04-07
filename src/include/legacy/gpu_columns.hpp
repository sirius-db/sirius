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

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/types.hpp"
#include "helper/common.h"
#include "log/logging.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

namespace duckdb {

class GPUBufferManager;

int32_t* convertUInt64ToInt32(uint64_t* data, size_t N);
uint64_t* convertInt32ToUInt64(int32_t* data, size_t N);
template <typename T>
void callCudaMemcpyHostToDevice(T* dest, T* src, size_t size, int gpu);
template <typename T>
void callCudaMemcpyDeviceToHost(T* dest, T* src, size_t size, int gpu);
template <typename T>
void callCudaMemcpyDeviceToDevice(T* dest, T* src, size_t size, int gpu);
void callCudaMemset(void* ptr, int value, size_t size, int gpu);
template <typename T>
void materializeExpression(T* a,
                           T*& result,
                           uint64_t* row_ids,
                           uint64_t result_len,
                           cudf::bitmask_type* mask,
                           cudf::bitmask_type*& out_mask);
template <typename T>
void materializeWithoutNull(T* a, T*& result, uint64_t* row_ids, uint64_t result_len);
void materializeString(uint8_t* data,
                       uint64_t* offset,
                       uint8_t*& result,
                       uint64_t*& result_offset,
                       uint64_t* row_ids,
                       uint64_t*& result_bytes,
                       uint64_t result_len,
                       cudf::bitmask_type* mask,
                       cudf::bitmask_type*& out_mask);
size_t getMaskBytesSize(uint64_t column_length);
cudf::bitmask_type* createNullMask(size_t size,
                                   cudf::mask_state state = cudf::mask_state::ALL_VALID);

enum class GPUColumnTypeId {
  INVALID = 0,
  INT16,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
  BOOLEAN,
  DATE,
  TIMESTAMP_SEC,
  TIMESTAMP_MS,
  TIMESTAMP_US,
  TIMESTAMP_NS,
  VARCHAR,
  INT128,
  DECIMAL
};

struct GPUDecimalTypeInfo {
  GPUDecimalTypeInfo(uint8_t width, uint8_t scale) : width_(width), scale_(scale)
  {
    D_ASSERT(width >= scale);
  }

  size_t GetDecimalTypeSize() const;

  uint8_t width_;
  uint8_t scale_;
};

struct GPUColumnType {
 public:
  GPUColumnType() : id_(GPUColumnTypeId::INVALID) {}
  explicit GPUColumnType(GPUColumnTypeId id) : id_(id) {}
  ~GPUColumnType() = default;

  inline GPUColumnTypeId id() const { return id_; }

  inline GPUDecimalTypeInfo* GetDecimalTypeInfo() const { return decimal_type_info_.get(); }

  void SetDecimalTypeInfo(uint8_t width, uint8_t scale)
  {
    decimal_type_info_ = make_shared_ptr<GPUDecimalTypeInfo>(width, scale);
  }

 private:
  GPUColumnTypeId id_;
  shared_ptr<GPUDecimalTypeInfo> decimal_type_info_;
};

inline GPUColumnType convertLogicalTypeToColumnType(LogicalType type)
{
  switch (type.id()) {
    case LogicalTypeId::SMALLINT:  // [Reserved] SMALLINT -> INT16
      return GPUColumnType(GPUColumnTypeId::INT16);
    case LogicalTypeId::INTEGER: return GPUColumnType(GPUColumnTypeId::INT32);
    case LogicalTypeId::BIGINT: return GPUColumnType(GPUColumnTypeId::INT64);
    case LogicalTypeId::HUGEINT: return GPUColumnType(GPUColumnTypeId::INT128);
    case LogicalTypeId::FLOAT: return GPUColumnType(GPUColumnTypeId::FLOAT32);
    case LogicalTypeId::DOUBLE: return GPUColumnType(GPUColumnTypeId::FLOAT64);
    case LogicalTypeId::BOOLEAN: return GPUColumnType(GPUColumnTypeId::BOOLEAN);
    case LogicalTypeId::DATE: return GPUColumnType(GPUColumnTypeId::DATE);
    case LogicalTypeId::TIMESTAMP_SEC: return GPUColumnType(GPUColumnTypeId::TIMESTAMP_SEC);
    case LogicalTypeId::TIMESTAMP_MS: return GPUColumnType(GPUColumnTypeId::TIMESTAMP_MS);
    case LogicalTypeId::TIMESTAMP: return GPUColumnType(GPUColumnTypeId::TIMESTAMP_US);
    case LogicalTypeId::TIMESTAMP_NS: return GPUColumnType(GPUColumnTypeId::TIMESTAMP_NS);
    case LogicalTypeId::VARCHAR: return GPUColumnType(GPUColumnTypeId::VARCHAR);
    case LogicalTypeId::DECIMAL: {
      GPUColumnType column_type(GPUColumnTypeId::DECIMAL);
      column_type.SetDecimalTypeInfo(DecimalType::GetWidth(type), DecimalType::GetScale(type));
      return column_type;
    }
    default:
      throw InvalidInputException(
        "Unsupported duckdb column type in `convertLogicalTypeToColumnType`: %d",
        static_cast<int>(type.id()));
  }
}

inline LogicalType convertColumnTypeToLogicalType(const GPUColumnType& type)
{
  switch (type.id()) {
    case GPUColumnTypeId::INT16:  // [Reserved] INT16 -> SMALLINT
      return LogicalType::SMALLINT;
    case GPUColumnTypeId::INT32: return LogicalType::INTEGER;
    case GPUColumnTypeId::INT64: return LogicalType::BIGINT;
    case GPUColumnTypeId::FLOAT32: return LogicalType::FLOAT;
    case GPUColumnTypeId::FLOAT64: return LogicalType::DOUBLE;
    case GPUColumnTypeId::BOOLEAN: return LogicalType::BOOLEAN;
    case GPUColumnTypeId::DATE: return LogicalType::DATE;
    case GPUColumnTypeId::TIMESTAMP_SEC: return LogicalType::TIMESTAMP_S;
    case GPUColumnTypeId::TIMESTAMP_MS: return LogicalType::TIMESTAMP_MS;
    case GPUColumnTypeId::TIMESTAMP_US: return LogicalType::TIMESTAMP;
    case GPUColumnTypeId::TIMESTAMP_NS: return LogicalType::TIMESTAMP_NS;
    case GPUColumnTypeId::VARCHAR: return LogicalType::VARCHAR;
    case GPUColumnTypeId::INT128: return LogicalType::HUGEINT;
    case GPUColumnTypeId::DECIMAL: {
      GPUDecimalTypeInfo* decimal_type_info = type.GetDecimalTypeInfo();
      if (decimal_type_info == nullptr) {
        throw InternalException(
          "`decimal_type_info` not set for DECIMAL type in `ColumnTypeToLogicalType`");
      }
      return LogicalType::DECIMAL(decimal_type_info->width_, decimal_type_info->scale_);
    }
    default:
      throw NotImplementedException(
        "Unsupported sirius column type in `ColumnTypeToLogicalType`: %d",
        static_cast<int>(type.id()));
  }
}

class DataWrapper {
 public:
  DataWrapper() = default;
  DataWrapper(GPUColumnType type, uint8_t* data, size_t size, cudf::bitmask_type* validity_mask);
  DataWrapper(GPUColumnType type,
              uint8_t* data,
              uint64_t* offset,
              size_t size,
              size_t num_bytes,
              bool is_string_data,
              cudf::bitmask_type* validity_mask);
  GPUColumnType type;
  uint8_t* data;
  size_t size;  // number of rows in the column (currently equals to column_length)
  uint64_t* offset{nullptr};
  size_t num_bytes;  // number of bytes in the column
  size_t getColumnTypeSize() const;
  bool is_string_data{false};
  cudf::bitmask_type* validity_mask{
    nullptr};  // validity mask for the column, used to represent NULL values
  size_t mask_bytes{0};
};

class GPUColumn {
 public:
  GPUColumn(size_t column_length,
            GPUColumnType type,
            uint8_t* data,
            cudf::bitmask_type* validity_mask);
  GPUColumn(size_t _column_length,
            GPUColumnType type,
            uint8_t* data,
            uint64_t* offset,
            size_t num_bytes,
            bool is_string_data,
            cudf::bitmask_type* validity_mask);
  GPUColumn(shared_ptr<GPUColumn> other);
  ~GPUColumn() {};
  int* GetDataInt32();
  uint64_t* GetDataUInt64();
  float* GetDataFloat32();
  double* GetDataFloat64();
  char* GetDataVarChar();
  uint8_t* GetDataBoolean();

  // [Critical Fix] Must declare here for .cpp implementation
  int16_t* GetDataInt16();

  uint64_t* GetRowIds();
  uint8_t* GetData();

  DataWrapper data_wrapper;
  uint64_t* row_ids;
  size_t row_id_count;   // number of rows in the row_ids array
  size_t column_length;  // number of rows in the column (currently equals to column_length)
  bool is_unique;        // indicator whether the column has unique values

  uint8_t* segment_start_ptr{nullptr};
  int segment_id{-1};

  cudf::column_view convertToCudfColumn();
  int32_t* convertSiriusOffsetToCudfOffset();
  int32_t* convertSiriusRowIdsToCudfRowIds();
  void convertCudfRowIdsToSiriusRowIds(int32_t* cudf_row_ids);
  void convertCudfOffsetToSiriusOffset(int32_t* cudf_offset);
  void setFromCudfColumn(cudf::column& cudf_column,
                         bool _is_unique,
                         int32_t* _row_ids,
                         uint64_t _row_id_count,
                         GPUBufferManager* gpuBufferManager);
  void setFromCudfScalar(cudf::scalar& cudf_scalar, GPUBufferManager* gpuBufferManager);

  size_t getTotalColumnSize();
};

class GPUIntermediateRelation {
 public:
  GPUIntermediateRelation(size_t column_count);
  ~GPUIntermediateRelation() {};
  bool checkLateMaterialization(size_t idx);

  string names;
  vector<string> column_names;
  vector<shared_ptr<GPUColumn>> columns;
  size_t column_count;

  /// Pre-owned cudf::table from registerExternalTable (cudf::unpack'd data).
  /// Raw pointer — ownership managed manually by the exchange lifecycle.
  /// nullptr when not populated from packed nixl transfer.
  cudf::table* packed_cudf_table = nullptr;

  /// Pending table_views accumulated from multiple senders (zero-copy).
  /// Each view points into a packed/staging buffer that stays alive until
  /// release_gpu_buffers(). Call finalize_pending_views() to concatenate
  /// them into a single cudf::table for duckdb_scan_task.
  ///
  /// IMPORTANT: cudf::unpack creates table_views that may reference the host
  /// metadata buffer internally (for STRING child column pointers). We must
  /// keep the metadata alive alongside the views.
  std::vector<cudf::table_view> pending_views;
  std::vector<std::string> pending_metadata; // Keeps metadata buffers alive
  std::vector<uint8_t*> pending_gpu_ptrs;    // GPU data pointers for re-unpack
  size_t pending_total_rows = 0;

  /// Concatenate all pending views into a single owned cudf::table.
  /// Called just before GPU execution of the exchange fragment.
  ///
  /// Always uses cudf::concatenate (even for single view) because
  /// cudf::table(column_view) deep copy fails for string columns from
  /// cudf::unpack. cudf::concatenate handles all column types correctly.
  void finalize_pending_views() {
    if (pending_views.empty()) return;
    // Log details about each pending view for debugging.
    for (size_t v = 0; v < pending_views.size(); v++) {
      auto& pv = pending_views[v];
      SIRIUS_LOG_INFO("[finalize_pending_views] view {}/{}: {} cols, {} rows",
                      v, pending_views.size(), pv.num_columns(), pv.num_rows());
      for (cudf::size_type c = 0; c < pv.num_columns(); c++) {
        auto col = pv.column(c);
        auto data_addr = reinterpret_cast<uintptr_t>(col.head<uint8_t>());
        // For STRING columns, also log child (offsets) info.
        if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
          auto child = col.child(0); // offsets column
          auto child_addr = reinterpret_cast<uintptr_t>(child.head<uint8_t>());
          // Read the last offset value (D2H) to check if it's absolute or relative.
          int32_t last_offset = 0;
          if (child.size() > 0) {
            auto* offset_ptr = child.data<int32_t>() + child.size() - 1;
            cudaMemcpy(&last_offset, offset_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
          }
          SIRIUS_LOG_INFO("[finalize_pending_views]   col {} STRING data=0x{:x} rows={} child_addr=0x{:x} child_rows={} last_offset={}",
                          c, data_addr, col.size(), child_addr, child.size(), last_offset);
        } else {
          SIRIUS_LOG_INFO("[finalize_pending_views]   col {} type={} data=0x{:x} rows={} null_cnt={} children={} offset={}",
                          c, static_cast<int>(col.type().id()), data_addr,
                          col.size(), col.null_count(), col.num_children(), col.offset());
        }
      }
    }
    static rmm::mr::cuda_memory_resource cuda_mr;
    cudaDeviceSynchronize();
    cudaGetLastError();

    // Re-unpack from stored metadata + gpu_ptrs to get fresh table_views.
    // The stored pending_views may have corrupted column_view pointers if
    // the vector reallocated or if cudf::unpack has state issues.
    if (pending_gpu_ptrs.size() == pending_views.size() &&
        pending_metadata.size() == pending_views.size()) {
      SIRIUS_LOG_INFO("[finalize_pending_views] re-unpacking {} views from stored metadata",
                      pending_views.size());
      pending_views.clear();
      for (size_t i = 0; i < pending_metadata.size(); i++) {
        auto* md_ptr = reinterpret_cast<const uint8_t*>(pending_metadata[i].data());
        auto view = cudf::unpack(md_ptr, pending_gpu_ptrs[i]);
        pending_views.push_back(view);
      }
    }

    // Pre-finalize integrity check: verify the first/last 4 bytes of each view's buffer.
    for (size_t v = 0; v < pending_views.size(); v++) {
      auto& pv = pending_views[v];
      auto head_addr = reinterpret_cast<uintptr_t>(pv.column(0).head<uint8_t>());
      // Read first 4 bytes via cudaMemcpy
      uint32_t first4 = 0;
      cudaMemcpy(&first4, reinterpret_cast<const void*>(head_addr), 4, cudaMemcpyDeviceToHost);
      // For STRING columns, check the offsets child's last value
      for (cudf::size_type c = 0; c < pv.num_columns(); c++) {
        auto col = pv.column(c);
        if (col.type().id() == cudf::type_id::STRING && col.num_children() > 0) {
          auto child = col.child(0);
          int32_t last_off = 0;
          if (child.size() > 0) {
            cudaMemcpy(&last_off, child.data<int32_t>() + child.size() - 1, 4, cudaMemcpyDeviceToHost);
            if (last_off > 100000000 || last_off < 0) {
              SIRIUS_LOG_ERROR("[PRE-FINALIZE] view {} col {} STRING last_offset={} CORRUPT (> 1MB for {} rows)",
                              v, c, last_off, col.size());
            }
          }
        }
      }
    }

    // Try concatenating incrementally to identify which view causes the failure.
    if (pending_views.size() > 1) {
      for (size_t i = 0; i < pending_views.size(); i++) {
        try {
          std::vector<cudf::table_view> single = {pending_views[i]};
          auto test = cudf::concatenate(single, cudf::get_default_stream(), &cuda_mr);
          SIRIUS_LOG_INFO("[finalize_pending_views] single view {} OK: {} rows", i, test->num_rows());
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("[finalize_pending_views] single view {} FAILED: {}", i, e.what());
        }
        cudaGetLastError(); // Clear error for next attempt
      }
    }

    auto merged = cudf::concatenate(pending_views, cudf::get_default_stream(), &cuda_mr);
    packed_cudf_table = merged.release();
    pending_views.clear();
    pending_metadata.clear(); // Metadata no longer needed after concatenation.
    if (packed_cudf_table) {
      pending_total_rows = static_cast<size_t>(packed_cudf_table->num_rows());
    }
  }
};

}  // namespace duckdb
