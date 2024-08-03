#include "gpu_operator_converter.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "operator/gpu_physical_order.hpp"

namespace duckdb {

// unique_ptr<GPUPhysicalOperator> 
GPUPhysicalOperator& 
GPUOperatorConverter::ConvertOperator(PhysicalOperator &op) {
	//converting from PhysicalOperator to GPUPhysicalOperator recursively
	// unique_ptr<GPUPhysicalOperator> gpu_operator = make_unique<GPUPhysicalOperator>(operator);

	// case statement mapping PhysicalOperator to GPUPhysicalOperator
	switch (op.type) {
	case PhysicalOperatorType::TABLE_SCAN:
		// return GPUPhysicalTableScan(operator);
		throw NotImplementedException("TABLE_SCAN not supported for GPU");
	case PhysicalOperatorType::DUMMY_SCAN:
		throw NotImplementedException("DUMMY_SCAN not supported for GPU");
	case PhysicalOperatorType::CHUNK_SCAN:
		throw NotImplementedException("CHUNK_SCAN not supported for GPU");
	case PhysicalOperatorType::COLUMN_DATA_SCAN:
		throw NotImplementedException("COLUMN_DATA_SCAN not supported for GPU");
	case PhysicalOperatorType::DELIM_SCAN:
		throw NotImplementedException("DELIM_SCAN not supported for GPU");
	case PhysicalOperatorType::ORDER_BY:
		// return GPUPhysicalOrder(operator);
		throw NotImplementedException("ORDER_BY not supported for GPU");
	case PhysicalOperatorType::LIMIT:
		throw NotImplementedException("LIMIT not supported for GPU");
	case PhysicalOperatorType::LIMIT_PERCENT:
		throw NotImplementedException("LIMIT_PERCENT not supported for GPU");
	case PhysicalOperatorType::STREAMING_LIMIT:
		throw NotImplementedException("STREAMING_LIMIT not supported for GPU");
	case PhysicalOperatorType::RESERVOIR_SAMPLE:
		throw NotImplementedException("RESERVOIR_SAMPLE not supported for GPU");
	case PhysicalOperatorType::STREAMING_SAMPLE:
		throw NotImplementedException("STREAMING_SAMPLE not supported for GPU");
	case PhysicalOperatorType::TOP_N:
		throw NotImplementedException("TOP_N not supported for GPU");
	case PhysicalOperatorType::WINDOW:
		throw NotImplementedException("WINDOW not supported for GPU");
	case PhysicalOperatorType::STREAMING_WINDOW:
		throw NotImplementedException("STREAMING_WINDOW not supported for GPU");
	case PhysicalOperatorType::UNNEST:
		throw NotImplementedException("UNNEST not supported for GPU");
	case PhysicalOperatorType::UNGROUPED_AGGREGATE:
		throw NotImplementedException("UNGROUPED_AGGREGATE not supported for GPU");
	case PhysicalOperatorType::HASH_GROUP_BY:
		throw NotImplementedException("HASH_GROUP_BY not supported for GPU");
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY:
		throw NotImplementedException("PERFECT_HASH_GROUP_BY not supported for GPU");
	case PhysicalOperatorType::FILTER:
		throw NotImplementedException("FILTER not supported for GPU");
	case PhysicalOperatorType::PROJECTION:
		throw NotImplementedException("PROJECTION not supported for GPU");
	case PhysicalOperatorType::COPY_TO_FILE:
		throw NotImplementedException("COPY_TO_FILE not supported for GPU");
	case PhysicalOperatorType::BATCH_COPY_TO_FILE:
		throw NotImplementedException("BATCH_COPY_TO_FILE not supported for GPU");
	case PhysicalOperatorType::LEFT_DELIM_JOIN:
		throw NotImplementedException("LEFT_DELIM_JOIN not supported for GPU");
	case PhysicalOperatorType::RIGHT_DELIM_JOIN:
		throw NotImplementedException("RIGHT_DELIM_JOIN not supported for GPU");
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
		throw NotImplementedException("BLOCKWISE_NL_JOIN not supported for GPU");
	case PhysicalOperatorType::NESTED_LOOP_JOIN:
		throw NotImplementedException("NESTED_LOOP_JOIN not supported for GPU");
	case PhysicalOperatorType::HASH_JOIN:
		throw NotImplementedException("HASH_JOIN not supported for GPU");
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
		throw NotImplementedException("PIECEWISE_MERGE_JOIN not supported for GPU");
	case PhysicalOperatorType::IE_JOIN:
		throw NotImplementedException("IE_JOIN not supported for GPU");
	case PhysicalOperatorType::ASOF_JOIN:
		throw NotImplementedException("ASOF_JOIN not supported for GPU");
	case PhysicalOperatorType::CROSS_PRODUCT:
		throw NotImplementedException("CROSS_PRODUCT not supported for GPU");
	case PhysicalOperatorType::POSITIONAL_JOIN:
		throw NotImplementedException("POSITIONAL_JOIN not supported for GPU");
	case PhysicalOperatorType::POSITIONAL_SCAN:
		throw NotImplementedException("POSITIONAL_SCAN not supported for GPU");
	case PhysicalOperatorType::UNION:
		throw NotImplementedException("UNION not supported for GPU");
	case PhysicalOperatorType::INSERT:
		throw NotImplementedException("INSERT not supported for GPU");
	case PhysicalOperatorType::BATCH_INSERT:
		throw NotImplementedException("BATCH_INSERT not supported for GPU");
	case PhysicalOperatorType::DELETE_OPERATOR:
		throw NotImplementedException("DELETE_OPERATOR not supported for GPU");
	case PhysicalOperatorType::UPDATE:
		throw NotImplementedException("UPDATE not supported for GPU");
	case PhysicalOperatorType::EMPTY_RESULT:
		throw NotImplementedException("EMPTY_RESULT not supported for GPU");
	case PhysicalOperatorType::CREATE_TABLE:
		throw NotImplementedException("CREATE_TABLE not supported for GPU");
	case PhysicalOperatorType::CREATE_TABLE_AS:
		throw NotImplementedException("CREATE_TABLE_AS not supported for GPU");
	case PhysicalOperatorType::BATCH_CREATE_TABLE_AS:
		throw NotImplementedException("BATCH_CREATE_TABLE_AS not supported for GPU");
	case PhysicalOperatorType::CREATE_INDEX:
		throw NotImplementedException("CREATE_INDEX not supported for GPU");
	case PhysicalOperatorType::EXPLAIN:
		throw NotImplementedException("EXPLAIN not supported for GPU");
	case PhysicalOperatorType::EXPLAIN_ANALYZE:
		throw NotImplementedException("EXPLAIN_ANALYZE not supported for GPU");
	case PhysicalOperatorType::EXECUTE:
		throw NotImplementedException("EXECUTE not supported for GPU");
	case PhysicalOperatorType::VACUUM:
		throw NotImplementedException("VACUUM not supported for GPU");
	case PhysicalOperatorType::RECURSIVE_CTE:
		throw NotImplementedException("RECURSIVE_CTE not supported for GPU");
	case PhysicalOperatorType::CTE:
		throw NotImplementedException("CTE not supported for GPU");
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
		throw NotImplementedException("RECURSIVE_CTE_SCAN not supported for GPU");
	case PhysicalOperatorType::CTE_SCAN:
		throw NotImplementedException("CTE_SCAN not supported for GPU");
	case PhysicalOperatorType::EXPRESSION_SCAN:
		throw NotImplementedException("EXPRESSION_SCAN not supported for GPU");
	case PhysicalOperatorType::ALTER:
		throw NotImplementedException("ALTER not supported for GPU");
	case PhysicalOperatorType::CREATE_SEQUENCE:
		throw NotImplementedException("CREATE_SEQUENCE not supported for GPU");
	case PhysicalOperatorType::CREATE_VIEW:
		throw NotImplementedException("CREATE_VIEW not supported for GPU");
	case PhysicalOperatorType::CREATE_SCHEMA:
		throw NotImplementedException("CREATE_SCHEMA not supported for GPU");
	case PhysicalOperatorType::CREATE_MACRO:
		throw NotImplementedException("CREATE_MACRO not supported for GPU");
	case PhysicalOperatorType::CREATE_SECRET:
		throw NotImplementedException("CREATE_SECRET not supported for GPU");
	case PhysicalOperatorType::DROP:
		throw NotImplementedException("DROP not supported for GPU");
	case PhysicalOperatorType::PRAGMA:
		throw NotImplementedException("PRAGMA not supported for GPU");
	case PhysicalOperatorType::TRANSACTION:
		throw NotImplementedException("TRANSACTION not supported for GPU");
	case PhysicalOperatorType::PREPARE:
		throw NotImplementedException("PREPARE not supported for GPU");
	case PhysicalOperatorType::EXPORT:
		throw NotImplementedException("EXPORT not supported for GPU");
	case PhysicalOperatorType::SET:
		throw NotImplementedException("SET not supported for GPU");
	case PhysicalOperatorType::RESET:
		throw NotImplementedException("RESET not supported for GPU");
	case PhysicalOperatorType::LOAD:
		throw NotImplementedException("LOAD not supported for GPU");
	case PhysicalOperatorType::INOUT_FUNCTION:
		throw NotImplementedException("INOUT_FUNCTION not supported for GPU");
	case PhysicalOperatorType::CREATE_TYPE:
		throw NotImplementedException("CREATE_TYPE not supported for GPU");
	case PhysicalOperatorType::ATTACH:
		throw NotImplementedException("ATTACH not supported for GPU");
	case PhysicalOperatorType::DETACH:
		throw NotImplementedException("DETACH not supported for GPU");
	case PhysicalOperatorType::RESULT_COLLECTOR:
		throw NotImplementedException("RESULT_COLLECTOR not supported for GPU");
	case PhysicalOperatorType::EXTENSION:
		throw NotImplementedException("EXTENSION not supported for GPU");
	case PhysicalOperatorType::PIVOT:
		throw NotImplementedException("PIVOT not supported for GPU");
	case PhysicalOperatorType::COPY_DATABASE:
		throw NotImplementedException("COPY_DATABASE not supported for GPU");
	case PhysicalOperatorType::VERIFY_VECTOR:
		throw NotImplementedException("VERIFY_VECTOR not supported for GPU");
	case PhysicalOperatorType::UPDATE_EXTENSIONS:
		throw NotImplementedException("UPDATE_EXTENSIONS not supported for GPU");
	case PhysicalOperatorType::INVALID:
		throw NotImplementedException("INVALID not supported for GPU");
	}


	// return gpu_operator;
}

} // namespace duckdb