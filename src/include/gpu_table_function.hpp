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
#include "duckdb/function/table_function.hpp"

namespace duckdb {

typedef unique_ptr<FunctionData> (*gpu_table_function_bind_t)(ClientContext &context, TableFunctionBindInput &input,
                                                          vector<LogicalType> &return_types, vector<string> &names);

typedef void (*gpu_table_function_t)(ClientContext &context, TableFunctionInput &data, DataChunk &output);

class GPUTableFunction : public TableFunction {
public:
	GPUTableFunction(string name, vector<LogicalType> arguments, gpu_table_function_t function,
	              gpu_table_function_bind_t bind = nullptr, table_function_init_global_t init_global = nullptr,
	              table_function_init_local_t init_local = nullptr);

    gpu_table_function_bind_t gpu_bind;
    gpu_table_function_t gpu_function;
};

} // namespace duckdb