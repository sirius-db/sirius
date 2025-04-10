#pragma once

#include "gpu_physical_operator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/common/extra_operator_info.hpp"
#include "gpu_expression_executor.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"

namespace duckdb {

enum ScanDataType {
	INT32,
	INT64,
	FLOAT32,
	FLOAT64,
	BOOLEAN,
	VARCHAR
};

enum CompareType {
	EQUAL,
	NOTEQUAL,
	GREATERTHAN,
	GREATERTHANOREQUALTO,
	LESSTHAN,
	LESSTHANOREQUALTO
};

void tableScanExpression(uint8_t **col, uint64_t** offset, uint8_t *constant_compare, uint64_t *constant_offset, 
	ScanDataType* data_type, uint64_t *&row_ids, uint64_t* &count, uint64_t N, CompareType* compare_mode, int num_expr);

class GPUPhysicalTableScan : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::TABLE_SCAN;

public:
	//! Table scan that immediately projects out filter columns that are unused in the remainder of the query plan
	GPUPhysicalTableScan(vector<LogicalType> types, TableFunction function, unique_ptr<FunctionData> bind_data,
		vector<LogicalType> returned_types, vector<ColumnIndex> column_ids, vector<idx_t> projection_ids,
		vector<string> names, unique_ptr<TableFilterSet> table_filters, idx_t estimated_cardinality,
		ExtraOperatorInfo extra_info, vector<Value> parameters, virtual_column_map_t virtual_columns);

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
	virtual_column_map_t virtual_columns;

	PhysicalTableScan* physical_table_scan;

	unique_ptr<ColumnDataCollection> collection;

	uint64_t* column_size;

	bool* already_cached;

	vector<LogicalType> scanned_types;

	vector<idx_t> scanned_ids;

	unique_ptr<TableFilterSet> fake_table_filters;
public:
	// string GetName() const override;
	// string ParamsToString() const override;

	// bool Equals(const GPUPhysicalOperator &other) const override;

public:
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	void ScanDataDuckDB(GPUBufferManager* gpuBufferManager, string up_table_name) const;

	SourceResultType GetDataDuckDB(ExecutionContext &exec_context);

	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}

	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
		GlobalSourceState &gstate) const override;
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	// SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override;
	// OperatorPartitionData GetPartitionData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
	// LocalSourceState &lstate,
	// const OperatorPartitionInfo &partition_info) const override;

	// bool SupportsPartitioning(const OperatorPartitionInfo &partition_info) const override;

	// ProgressData GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;
};

} // namespace duckdb