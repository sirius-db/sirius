#pragma once

#include "duckdb.hpp"
#include "communication.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
extern void myKernel();

class GPUBufferManager;
class SiriusExtension : public Extension {
public:
	void Load(DuckDB &db) override;
	std::string Name() override;
	void InitializeGPUExtension(Connection &con);
	static void GPUProcessingSubstraitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static void GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static void GPUCachingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static unique_ptr<FunctionData> GPUProcessingSubstraitBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	static unique_ptr<FunctionData> GPUProcessingBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	static unique_ptr<FunctionData> GPUCachingBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);

	// If false, then use duckdb to process substrait.
	static constexpr bool USE_SIRIUS_FOR_SUBSTRAIT = false;
};

} // namespace duckdb