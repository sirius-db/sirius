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

#include "operator/gpu_combine.hpp"
#include "operator/gpu_materialize.hpp"

#include <cudf/concatenate.hpp>

namespace duckdb {

shared_ptr<GPUIntermediateRelation> CombineChunks(const vector<shared_ptr<GPUIntermediateRelation>> &input){
  // Special cases
  if (input.empty()) {
    throw InternalException("`input` is empty in `CombineChunks`");
  }
  size_t num_columns = input[0]->column_count;
  if (input.size() == 1 || num_columns == 0) {
    return input[0];
  }
  vector<shared_ptr<GPUIntermediateRelation>> non_empty_input;
  for (const auto &chunk: input) {
    int num_rows = chunk->columns[0]->row_ids == nullptr
      ? chunk->columns[0]->column_length : chunk->columns[0]->row_id_count;
    if (num_rows > 0) {
      non_empty_input.push_back(chunk);
    }
  }
  if (non_empty_input.empty()) {
    return input[0];
  }

  // Create output table
  auto output = make_shared_ptr<GPUIntermediateRelation>(num_columns);
  output->column_names = input[0]->column_names;
  for (int i = 0; i < num_columns; ++i) {
    output->columns[i] = make_shared_ptr<GPUColumn>(
      0, input[0]->columns[i]->data_wrapper.type, nullptr, nullptr,
      0, input[0]->columns[i]->data_wrapper.is_string_data, nullptr);
  }

  // Materialize each input chunk and convert to cudf column
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  vector<vector<cudf::column_view>> input_cudf_columns;
  input_cudf_columns.resize(num_columns);
  for (auto& chunk: input) {
    for (int i = 0; i < num_columns; ++i) {
      chunk->columns[i] = HandleMaterializeExpression(chunk->columns[i], gpuBufferManager);
      input_cudf_columns[i].push_back(chunk->columns[i]->convertToCudfColumn());
    }
  }

  // Call cudf concatenate
  for (int i = 0; i < num_columns; ++i) {
    const auto &cols = input_cudf_columns[i];
    auto output_cudf_column = cudf::concatenate(cudf::host_span<cudf::column_view const>(cols.data(), cols.size()));
    output->columns[i]->setFromCudfColumn(*output_cudf_column, input[0]->columns[i]->is_unique,
                                          nullptr, 0, gpuBufferManager);;
  }

  return output;
}

}
