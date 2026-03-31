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

#include <duckdb/function/table_function.hpp>

namespace duckdb {

/// Bind function for gpu_explain table function.
unique_ptr<FunctionData> GPUExplainBind(ClientContext& context,
                                        TableFunctionBindInput& input,
                                        vector<LogicalType>& return_types,
                                        vector<string>& names);

/// Execute function for gpu_explain table function.
void GPUExplainFunction(ClientContext& context, TableFunctionInput& data_p, DataChunk& output);

}  // namespace duckdb
