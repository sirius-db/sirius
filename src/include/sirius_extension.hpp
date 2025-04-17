#pragma once

#include "duckdb.hpp"
#include "communication.hpp"
#include "gpu_context.hpp"

namespace duckdb {

// Declaration of the CUDA kernel
extern void myKernel();

class GPUBufferManager;
class SiriusExtension : public Extension {
public:
	struct GPUTableFunctionData : public TableFunctionData {
		GPUTableFunctionData() = default;
		shared_ptr<Relation> plan;
		shared_ptr<GPUPreparedStatementData> gpu_prepared;
		unique_ptr<QueryResult> res;
		unique_ptr<Connection> conn;
		unique_ptr<GPUContext> gpu_context;
		string query;
		bool enable_optimizer;
		bool finished = false;
		bool plan_error = false;

		// Set when used by sirius server
		bool is_sirius_server = false;		// should not consume result if called by sirius server
		string* result_intermediate_table_name = nullptr;
	};

	struct GPUCachingFunctionData : public TableFunctionData {
		GPUCachingFunctionData() = default;
		unique_ptr<Connection> conn;
		GPUBufferManager *gpuBufferManager;
		ColumnType type;
		uint8_t *data;
		string column;
		string table;
		bool finished = false;
	};

	void Load(DuckDB &db) override;
	std::string Name() override;
	void InitializeGPUExtension(Connection &con);
	static void GPUProcessingSubstraitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static void GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static void GPUCachingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);
	static unique_ptr<FunctionData> GPUProcessingSubstraitBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	static unique_ptr<FunctionData> GPUProcessingBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);
	static unique_ptr<FunctionData> GPUCachingBind(ClientContext &context, TableFunctionBindInput &input, vector<LogicalType> &return_types, vector<string> &names);

	// Make and extract additional inputs used by sirius server
	static const string KEY_PLAN_IN_JSON;
	static const string KEY_SOURCE_EXCHANGE_TABLES;
	static Value MakeSiriusServerAdditionalInputsStruct(bool plan_in_json, const std::vector<string>& source_exchange_tables);
	static void ExtractSiriusServerAdditionalInputs(const Value& additional_input_struct, bool* plan_in_json, std::vector<string>& source_exchange_tables);

	// If false, then use duckdb to process substrait.
	static constexpr bool USE_SIRIUS_FOR_SUBSTRAIT = true;
};

} // namespace duckdb