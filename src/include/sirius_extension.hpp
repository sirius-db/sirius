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

#include "duckdb.hpp"

namespace duckdb {
class GPUBufferManager;
class SiriusExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
	void InitializeGPUExtension(Connection &con);
	static void GPUProcessingSubstraitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static void GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	// static void GPUCachingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static unique_ptr<FunctionData> GPUProcessingSubstraitBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	static unique_ptr<FunctionData> GPUProcessingBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	// static unique_ptr<FunctionData> GPUCachingBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	static void GPUBufferInitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static unique_ptr<FunctionData> GPUBufferInitBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);

	static bool buffer_is_initialized;
};

} // namespace duckdb