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

#include "gpu_physical_operator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/common/extra_operator_info.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

namespace duckdb {

enum ScanDataType {
	INT16,
	INT32,
	INT64,
	FLOAT32,
	FLOAT64,
	BOOLEAN,
	DATE,
	VARCHAR,
	DECIMAL32,
	DECIMAL64,
	SQLNULL
};

enum CompareType {
	EQUAL,
	NOTEQUAL,
	GREATERTHAN,
	GREATERTHANOREQUALTO,
	LESSTHAN,
	LESSTHANOREQUALTO,
	IS_NULL,
	IS_NOT_NULL
};

template <typename T> void comparisonConstantExpression(T *a, T b, T c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template <typename T> void comparisonExpression(T *a, T* b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
void comparisonStringBetweenExpression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, std::string lower_string, std::string upper_string, 
    bool is_lower_inclusive, bool is_upper_inclusive, uint64_t* &row_id, uint64_t* &count);
void comparisonStringExpression(char* char_data, uint64_t num_chars, uint64_t* str_indices, uint64_t num_strings, std::string comparison_string, int op_mode, uint64_t* &row_id, uint64_t* &count);
void tableScanExpression(uint8_t **col, uint64_t** offset, cudf::bitmask_type** bitmask, uint8_t *constant_compare, uint64_t *constant_offset, 
	ScanDataType* data_type, uint64_t *&row_ids, uint64_t* &count, uint64_t N, CompareType* compare_mode, int num_expr);

class GPUPhysicalTableScan : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::TABLE_SCAN;

public:
	//! Table scan that immediately projects out filter columns that are unused in the remainder of the query plan
	GPUPhysicalTableScan(vector<LogicalType> types, TableFunction function, unique_ptr<FunctionData> bind_data,
		vector<LogicalType> returned_types, vector<ColumnIndex> column_ids, vector<idx_t> projection_ids,
		vector<string> names, unique_ptr<TableFilterSet> table_filters, idx_t estimated_cardinality,
		ExtraOperatorInfo extra_info, vector<Value> parameters);

	//! The table function
	TableFunction function;
	//! Bind data of the function
	unique_ptr<FunctionData> bind_data;
	//! The types of ALL columns that can be returned by the table function
	vector<LogicalType> returned_types;
	//! The column ids used within the table function
	vector<ColumnIndex> column_ids;
	//! The projected-out column ids
	vector<idx_t> projection_ids;
	//! The names of the columns
	vector<string> names;
	//! The table filters
	unique_ptr<TableFilterSet> table_filters;
	//! Currently stores info related to filters pushed down into MultiFileLists and sample rate pushed down into the
	//! table scan
	ExtraOperatorInfo extra_info;
	//! Parameters
	vector<Value> parameters;
	//! Contains a reference to dynamically generated table filters (through e.g. a join up in the tree)
	shared_ptr<DynamicTableFilterSet> dynamic_filters;
	//! Virtual columns
	// virtual_column_map_t virtual_columns;

	PhysicalTableScan* physical_table_scan;

	unique_ptr<ColumnDataCollection> collection;

	uint64_t* column_size;

	uint64_t* mask_size;

	bool* already_cached;

	vector<LogicalType> scanned_types;

	vector<idx_t> scanned_ids;

	unique_ptr<TableFilterSet> fake_table_filters;

	//! Whether it's required to generate a seperate row id column (e.g., in some select *)
	bool gen_row_id_column;

	//! Only used in optimized table scan
	uint64_t num_rows;
	vector<idx_t> uncached_scan_column_ids;
	vector<cudaStream_t> cuda_streams;
	vector<ColumnIndex> orig_column_ids;
	vector<idx_t> orig_scanned_ids;
	vector<LogicalType> orig_scanned_types;

public:
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	SourceResultType GetDataDuckDBOpt(ExecutionContext &exec_context);
	void ScanDataDuckDBOpt(ExecutionContext &exec_context,
												 GPUBufferManager* gpuBufferManager,
												 string up_table_name);

	SourceResultType GetDataDuckDB(ExecutionContext &exec_context);
	void ScanDataDuckDB(GPUBufferManager* gpuBufferManager, string up_table_name) const;

	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}

	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
		GlobalSourceState &gstate) const override;
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
};

} // namespace duckdb