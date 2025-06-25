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

#include "cudf/cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

void cudf_duplicate_elimination(vector<shared_ptr<GPUColumn>>& keys, uint64_t num_keys) 
{
  if (keys[0]->column_length == 0) {
    SIRIUS_LOG_DEBUG("Input size is 0");
    for (idx_t group = 0; group < num_keys; group++) {
      bool old_unique = keys[group]->is_unique;
      if (keys[group]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        keys[group] = make_shared_ptr<GPUColumn>(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data, keys[group]->data_wrapper.offset, 0, true);
      } else {
        keys[group] = make_shared_ptr<GPUColumn>(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data);
      }
      keys[group]->is_unique = old_unique;
    }
    return;
  }

  SIRIUS_LOG_DEBUG("CUDF Group By");
  SIRIUS_LOG_DEBUG("Input size: {}", keys[0]->column_length);
  SETUP_TIMING();
  START_TIMER();

  GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudf::set_current_device_resource(gpuBufferManager->mr);

  std::vector<cudf::column_view> keys_cudf;

  //TODO: This is a hack to get the size of the keys
  size_t size = 0;

  for (int key = 0; key < num_keys; key++) {
    if (keys[key]->data_wrapper.data != nullptr) {
      auto cudf_column = keys[key]->convertToCudfColumn();
      keys_cudf.push_back(cudf_column);
      size = keys[key]->column_length;
    } else {
      throw NotImplementedException("Group by on non-nullable column not supported");
    }
  }

  auto keys_table = cudf::table_view(keys_cudf);
  cudf::groupby::groupby grpby_obj(keys_table);
  auto result = grpby_obj.get_groups();

  auto result_key = std::move(result.keys);
  for (int key = 0; key < num_keys; key++) {
      cudf::column group_key = result_key->get_column(key);
      keys[key]->setFromCudfColumn(group_key, keys[key]->is_unique, nullptr, 0, gpuBufferManager);
  }

  STOP_TIMER();
  SIRIUS_LOG_DEBUG("CUDF Groupby result count: {}", keys[0]->column_length);
}

} //namespace duckdb