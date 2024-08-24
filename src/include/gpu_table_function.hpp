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