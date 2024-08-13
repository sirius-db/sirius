#pragma once

#include "gpu_physical_operator.hpp"
#include "duckdb/planner/bound_query_node.hpp"

namespace duckdb {

class GPUPhysicalOrder : public GPUPhysicalOperator {
public:
    // GPUPhysicalOrder(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality);

    GPUPhysicalOrder(vector<LogicalType> types, vector<BoundOrderByNode> orders, vector<idx_t> projections,
	              idx_t estimated_cardinality);

	//! Input data
	vector<BoundOrderByNode> orders;
	vector<idx_t> projections;

};
} // namespace duckdb