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

#include "operator/gpu_physical_limit.hpp"

#include "log/logging.hpp"
#include "operator/gpu_materialize.hpp"

#include <algorithm>

namespace duckdb {

GPUPhysicalStreamingLimit::GPUPhysicalStreamingLimit(vector<LogicalType> types,
                                                     BoundLimitNode limit_val_p,
                                                     BoundLimitNode offset_val_p,
                                                     idx_t estimated_cardinality,
                                                     bool parallel)
  : GPUPhysicalOperator(
      PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
    limit_val(std::move(limit_val_p)),
    offset_val(std::move(offset_val_p)),
    parallel(parallel)
{
}

// Returns element size in bytes for fixed-width GPU column types.
static size_t GetElementSizeBytes(const GPUColumnType& col_type)
{
  switch (col_type.id()) {
    case GPUColumnTypeId::BOOLEAN: return sizeof(uint8_t);
    case GPUColumnTypeId::INT16: return sizeof(int16_t);
    case GPUColumnTypeId::INT32:
    case GPUColumnTypeId::DATE:
    case GPUColumnTypeId::FLOAT32: return sizeof(int32_t);
    case GPUColumnTypeId::INT64:
    case GPUColumnTypeId::FLOAT64:
    case GPUColumnTypeId::TIMESTAMP_SEC:
    case GPUColumnTypeId::TIMESTAMP_MS:
    case GPUColumnTypeId::TIMESTAMP_US:
    case GPUColumnTypeId::TIMESTAMP_NS: return sizeof(int64_t);
    case GPUColumnTypeId::INT128: return sizeof(__int128_t);
    case GPUColumnTypeId::DECIMAL: {
      auto* info = col_type.GetDecimalTypeInfo();
      return info ? info->GetDecimalTypeSize() : sizeof(int64_t);
    }
    default: return 0;
  }
}

OperatorResultType GPUPhysicalStreamingLimit::Execute(
  GPUIntermediateRelation& input_relation, GPUIntermediateRelation& output_relation) const
{
  SIRIUS_LOG_DEBUG("Executing streaming limit");
  SIRIUS_LOG_DEBUG("Limit value {}", limit_val.GetConstantValue());
  if (limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
    throw NotImplementedException("Streaming limit other than constant value not implemented");
  }
  auto limit_const = limit_val.GetConstantValue();

  idx_t offset_const = 0;
  if (offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
    offset_const = offset_val.GetConstantValue();
  }

  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
    shared_ptr<GPUColumn> materialize_column =
      HandleMaterializeExpression(input_relation.columns[col_idx], gpuBufferManager);

    idx_t col_len = materialize_column->column_length;
    idx_t skip    = std::min(offset_const, col_len);
    idx_t take    = std::min(limit_const, col_len - skip);

    uint8_t* new_data    = materialize_column->data_wrapper.data;
    uint64_t* new_offset = materialize_column->data_wrapper.offset;

    if (skip > 0) {
      auto type_id = materialize_column->data_wrapper.type.id();
      if (type_id == GPUColumnTypeId::VARCHAR) {
        // Offsets are absolute char positions; shift pointer so new_offset[0] = offset[skip].
        // The char data pointer stays at base — downstream reads data[new_offset[i]] correctly.
        new_offset = materialize_column->data_wrapper.offset + skip;
      } else {
        size_t elem_size = GetElementSizeBytes(materialize_column->data_wrapper.type);
        new_data         = materialize_column->data_wrapper.data + skip * elem_size;
      }
    }

    output_relation.columns[col_idx] =
      make_shared_ptr<GPUColumn>(take,
                                 materialize_column->data_wrapper.type,
                                 new_data,
                                 new_offset,
                                 materialize_column->data_wrapper.num_bytes,
                                 materialize_column->data_wrapper.is_string_data,
                                 materialize_column->data_wrapper.validity_mask);
    output_relation.columns[col_idx]->is_unique = materialize_column->is_unique;

    // For VARCHAR: read new_offset[take] = offset[skip+take] as the absolute end char position.
    if (take > 0 &&
        output_relation.columns[col_idx]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
      Allocator& allocator = Allocator::DefaultAllocator();
      uint64_t* end_pos    = reinterpret_cast<uint64_t*>(allocator.AllocateData(sizeof(uint64_t)));
      callCudaMemcpyDeviceToHost<uint64_t>(end_pos, new_offset + take, 1, 0);
      output_relation.columns[col_idx]->data_wrapper.num_bytes = end_pos[0];
    }
    SIRIUS_LOG_DEBUG("Column {} has {} rows (offset={}, take={})",
                     col_idx,
                     output_relation.columns[col_idx]->column_length,
                     skip,
                     take);
  }

  return OperatorResultType::FINISHED;
}
}  // namespace duckdb
