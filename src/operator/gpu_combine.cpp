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

void CombineChunks(vector<shared_ptr<GPUIntermediateRelation>> &input, GPUIntermediateRelation& output) {
  // Special cases
  if (input.empty() || input[0]->column_count == 0) {
    return;
  }
  output.column_names = input[0]->column_names;
  output.column_count = input[0]->column_count;
  output.columns = input[0]->columns;
  if (input.size() <= 1) {
    return;
  }

  // Materialize each input chunk and convert to cudf column
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  vector<vector<cudf::column_view>> input_cudf_columns;
  input_cudf_columns.resize(output.column_count);
  for (auto& chunk: input) {
    for (int i = 0; i < chunk->column_count; ++i) {
      chunk->columns[i] = HandleMaterializeExpression(chunk->columns[i], gpuBufferManager);
      input_cudf_columns[i].push_back(chunk->columns[i]->convertToCudfColumn());
    }
  }

  // Call cudf concatenate
  for (int i = 0; i < output.column_count; ++i) {
    const auto &cols = input_cudf_columns[i];
    auto output_cudf_column = cudf::concatenate(cudf::host_span<cudf::column_view const>(cols.data(), cols.size()));
    output.columns[i]->setFromCudfColumn(*output_cudf_column, input[0]->columns[i]->is_unique,
                                         nullptr, 0, gpuBufferManager);;
  }
}

}
