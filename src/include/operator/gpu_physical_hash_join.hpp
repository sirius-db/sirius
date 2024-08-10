#pragma once

#include "gpu_physical_operator.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_comparison_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/operator/logical_join.hpp"

namespace duckdb {

class GPUPhysicalHashJoin : public GPUPhysicalOperator {
public:
	GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
	                 vector<JoinCondition> cond, JoinType join_type, const vector<idx_t> &left_projection_map,
	                 const vector<idx_t> &right_projection_map, vector<LogicalType> delim_types,
	                 idx_t estimated_cardinality);
	GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left, unique_ptr<GPUPhysicalOperator> right,
	                 vector<JoinCondition> cond, JoinType join_type, idx_t estimated_cardinalitye);

};
} // namespace duckdb