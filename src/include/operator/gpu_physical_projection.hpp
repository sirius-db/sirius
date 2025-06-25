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
// #include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

class GPUPhysicalProjection : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::PROJECTION;

public:
	GPUPhysicalProjection(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
	                   idx_t estimated_cardinality);

	vector<unique_ptr<Expression>> select_list;

	// OperatorResultType Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation,
	// 									GlobalOperatorState &gstate, OperatorState &state) const override;

	OperatorResultType Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const override;

	// GPUExpressionExecutor* gpu_expression_executor;

public:
	// unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	// OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	//                            GlobalOperatorState &gstate, OperatorState &state) const override;

	// bool ParallelOperator() const override {
	// 	return true;
	// }

	// string ParamsToString() const override;

	// static unique_ptr<PhysicalOperator>
	// CreateJoinProjection(vector<LogicalType> proj_types, const vector<LogicalType> &lhs_types,
	//                      const vector<LogicalType> &rhs_types, const vector<idx_t> &left_projection_map,
	//                      const vector<idx_t> &right_projection_map, const idx_t estimated_cardinality);


};
} // namespace duckdb