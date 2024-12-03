#include "operator/gpu_physical_ungrouped_aggregate.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>

namespace duckdb {

GPUPhysicalUngroupedAggregate::GPUPhysicalUngroupedAggregate(vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> expressions,
                                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
      aggregates(std::move(expressions)) {

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(aggregates);
	aggregation_result = new GPUIntermediateRelation(aggregates.size());
	if (!distinct_collection_info) {
		return;
	}
	distinct_data = make_uniq<DistinctAggregateData>(*distinct_collection_info);

}

// SinkResultType
// GPUPhysicalUngroupedAggregate::Sink(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const {

SinkResultType 
GPUPhysicalUngroupedAggregate::Sink(GPUIntermediateRelation &input_relation) const {
	printf("Performing ungrouped aggregation\n");

	if (distinct_data) {
		std::cout << "Running SinkDistinct" << std::endl;
		SinkDistinct(input_relation);
	}

	idx_t payload_idx = 0;
	idx_t next_payload_idx = 0;

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		D_ASSERT(aggregates[aggr_idx]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

		payload_idx = next_payload_idx;
		next_payload_idx = payload_idx + aggregate.children.size();
		std::cout << "Got payload indicies (" << payload_idx << "," << next_payload_idx << ")" << std::endl;

		if (aggregate.IsDistinct()) {
			std::cout << "Skipping due to aggregate being a filter" << std::endl;
			continue;
		}

		if (aggregate.filter) {
			auto &bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
			printf("Reading filter column from index %ld\n", bound_ref_expr.index);
		}

		int num_children = aggregate.children.size();
		std::cout << "Got aggregate children of size " << num_children << std::endl;
		if(aggregate.function.ToString().find("count_star") != std::string::npos) {
			// Update the aggregate function to store the number of records
			aggregation_result->columns[aggr_idx] = input_relation.columns[aggr_idx];
			aggregation_result->columns[aggr_idx]->data_wrapper.type = ColumnType::INT64;
			aggregation_result->columns[aggr_idx]->data_wrapper.is_string_data = false;
		
			int64_t num_records = (int64_t) aggregation_result->columns[aggr_idx]->row_id_count;
			aggregation_result->columns[aggr_idx]->data_wrapper.size = sizeof(int64_t);
			int64_t* gpu_data_col = reinterpret_cast<int64_t*>(aggregation_result->columns[aggr_idx]->data_wrapper.data);
			cudaMemcpy(gpu_data_col, &num_records, sizeof(int64_t), cudaMemcpyHostToDevice);

			next_payload_idx = payload_idx + 1;
			std::cout << "Set the column " << aggr_idx << " to value " << num_records << std::endl;

			// Set it to be materalized
			aggregation_result->columns[aggr_idx]->row_id_count = 0;
			aggregation_result->columns[aggr_idx]->row_ids = nullptr;
			aggregation_result->columns[aggr_idx]->column_length = 1;
		} else {
			idx_t payload_cnt = 0;
			for (idx_t i = 0; i < aggregate.children.size(); ++i) {
				for (auto &child_expr : aggregate.children) {
					D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
					printf("Reading aggregation column from index %ld and passing it to index %ld in aggregation result\n", payload_idx + payload_cnt, aggr_idx);
					input_relation.checkLateMaterialization(payload_idx + payload_cnt);
					// printf("aggregation_result.columns.size() %ld\n", aggregation_result->columns.size());
					// printf("input_relation.columns.size() %ld\n", input_relation.columns.size());
					aggregation_result->columns[aggr_idx] = input_relation.columns[payload_idx + payload_cnt];
					payload_cnt++;
				}
			}
		}
	}

  return SinkResultType::FINISHED;
}

// SourceResultType
// GPUPhysicalUngroupedAggregate::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {
  
SourceResultType
GPUPhysicalUngroupedAggregate::GetData(GPUIntermediateRelation &output_relation) const {
  for (int col = 0; col < aggregation_result->columns.size(); col++) {
    printf("Writing aggregation result to column %ld\n", col);
    output_relation.columns[col] = aggregation_result->columns[col];
  }

  return SourceResultType::FINISHED;
}

// void
// GPUPhysicalUngroupedAggregate::SinkDistinct(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const {
void
GPUPhysicalUngroupedAggregate::SinkDistinct(GPUIntermediateRelation &input_relation) const {
	auto &distinct_info = *distinct_collection_info;
	auto &distinct_indices = distinct_info.Indices();
	auto &distinct_filter = distinct_info.Indices();

	for (auto &idx : distinct_indices) {
		auto &aggregate = aggregates[idx]->Cast<BoundAggregateExpression>();

		D_ASSERT(distinct_info.table_map.count(idx));

		if (aggregate.filter) {
			auto &bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
			printf("Reading filter column from index %ld\n", bound_ref_expr.index);

			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				printf("Reading aggregation column from index %ld and passing it to index %ld in groupby result\n", bound_ref.index, bound_ref.index);
				input_relation.checkLateMaterialization(bound_ref.index);
				aggregation_result->columns[bound_ref.index] = input_relation.columns[bound_ref.index];
			}
		} else {
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				printf("Reading aggregation column from index %ld and passing it to index %ld in groupby result\n", bound_ref.index, bound_ref.index);
				input_relation.checkLateMaterialization(bound_ref.index);
				aggregation_result->columns[bound_ref.index] = input_relation.columns[bound_ref.index];
			}
		}
	}

}

} // namespace duckdb