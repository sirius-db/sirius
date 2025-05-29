#include "operator/gpu_physical_ungrouped_aggregate.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_materialize.hpp"

namespace duckdb {

template <typename T>
void
ResolveTypeAggregateExpression(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
	uint8_t** aggregate_data = new uint8_t*[aggregates.size()];
	uint8_t** result = new uint8_t*[aggregates.size()];
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		result[agg_idx] = nullptr;
	}

	size_t size = aggregate_keys[0]->column_length;

	int* agg_mode = new int[aggregates.size()];

	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
			agg_mode[agg_idx] = 5;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
			agg_mode[agg_idx] = 5;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("sum") == 0) {
			agg_mode[agg_idx] = 0;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("avg") == 0) {
			if (aggregate_keys[agg_idx]->data_wrapper.type != ColumnType::FLOAT64) throw NotImplementedException("Column type is supposed to be double");
			agg_mode[agg_idx] = 1;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("max") == 0) {
			agg_mode[agg_idx] = 2;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("min") == 0) {
			agg_mode[agg_idx] = 3;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);
		} else if (expr.function.name.compare("count_star") == 0) {
			agg_mode[agg_idx] = 4;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = 4;
			aggregate_data[agg_idx] = nullptr;
		} else if (expr.function.name.compare("first") == 0) {
			agg_mode[agg_idx] = 6;
			aggregate_data[agg_idx] = (aggregate_keys[agg_idx]->data_wrapper.data);;
		} else {
			throw NotImplementedException("Aggregate function not supported");
		}
	}

	ungroupedAggregate<T>(aggregate_data, result, size, agg_mode, aggregates.size());

	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count_star") == 0 || expr.function.name.compare("count") == 0) {
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(1, ColumnType::INT64, reinterpret_cast<uint8_t*>(result[agg_idx]));
		} else if (size == 0){
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(0, ColumnType::INT64, reinterpret_cast<uint8_t*>(result[agg_idx]));
		} else { 
			aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(1, aggregate_keys[agg_idx]->data_wrapper.type, reinterpret_cast<uint8_t*>(result[agg_idx]));
		}
	}
}

void
HandleAggregateExpression(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
	//check if all the aggregate functions are of the same type
	bool same_type = true;
	ColumnType prev_type;
	for (int i = 0; i < aggregates.size(); i++) {
		if (aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count") != 0 && 
					aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count_star") != 0) {
			prev_type = aggregate_keys[i]->data_wrapper.type;
			break;
		}
	}
	for (int i = 0; i < aggregates.size(); i++) {
		if (aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count") != 0 && 
					aggregates[i]->Cast<BoundAggregateExpression>().function.name.compare("count_star") != 0) {
			ColumnType aggregate_type = aggregate_keys[i]->data_wrapper.type;
			if (aggregate_type != prev_type) {
				throw NotImplementedException("All aggregate functions must be of the same type");
			}
			prev_type = aggregate_type;
		}
	}

    switch(aggregate_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
		ResolveTypeAggregateExpression<uint64_t>(aggregate_keys, gpuBufferManager, aggregates);
		break;
      case ColumnType::FLOAT64:
	  	ResolveTypeAggregateExpression<double>(aggregate_keys, gpuBufferManager, aggregates);
		break;
      default:
        throw NotImplementedException("Unsupported column type");
    }
}

void
HandleAggregateExpressionCuDF(vector<shared_ptr<GPUColumn>> &aggregate_keys, GPUBufferManager* gpuBufferManager, const vector<unique_ptr<Expression>> &aggregates) {
	AggregationType* agg_mode = new AggregationType[aggregates.size()];
	printf("Handling ungrouped aggregate expression\n");
	for (int agg_idx = 0; agg_idx < aggregates.size(); agg_idx++) {
		auto& expr = aggregates[agg_idx]->Cast<BoundAggregateExpression>();
		if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr && aggregate_keys[agg_idx]->column_length == 0) {
			agg_mode[agg_idx] = AggregationType::COUNT;
		} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr && aggregate_keys[agg_idx]->column_length == 0) {
			agg_mode[agg_idx] = AggregationType::SUM;
		} else if (expr.function.name.compare("sum") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = AggregationType::SUM;
		} else if (expr.function.name.compare("avg") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = AggregationType::AVERAGE;
		} else if (expr.function.name.compare("max") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = AggregationType::MAX;
		} else if (expr.function.name.compare("min") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = AggregationType::MIN;
		} else if (expr.function.name.compare("count_star") == 0 && aggregate_keys[agg_idx]->data_wrapper.data == nullptr) {
			agg_mode[agg_idx] = AggregationType::COUNT_STAR;
		} else if (expr.function.name.compare("count") == 0 && aggregate_keys[agg_idx]->data_wrapper.data != nullptr) {
			agg_mode[agg_idx] = AggregationType::COUNT;
		} else if (expr.function.name.compare("first") == 0) {
			agg_mode[agg_idx] = AggregationType::FIRST;
		} else {
			printf("Aggregate function not supported: %s\n", expr.function.name.c_str());
			throw NotImplementedException("Aggregate function not supported");
		}
	}

	cudf_aggregate(aggregate_keys, aggregates.size(), agg_mode);
}

GPUPhysicalUngroupedAggregate::GPUPhysicalUngroupedAggregate(vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> expressions,
                                                       idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
      aggregates(std::move(expressions)) {

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(aggregates);
	aggregation_result = make_shared_ptr<GPUIntermediateRelation>(aggregates.size());
	if (!distinct_collection_info) {
		return;
	}
	distinct_data = make_uniq<DistinctAggregateData>(*distinct_collection_info);

}

SinkResultType 
GPUPhysicalUngroupedAggregate::Sink(GPUIntermediateRelation &input_relation) const {
	printf("Performing ungrouped aggregation\n");

	if (distinct_data) {
		SinkDistinct(input_relation);
	}

	idx_t payload_idx = 0;
	idx_t next_payload_idx = 0;
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	vector<shared_ptr<GPUColumn>> aggregate_column(aggregates.size());
	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		aggregate_column[aggr_idx] = nullptr;
	}

	uint64_t column_size = 0;
	for (int i = 0; i < input_relation.columns.size(); i++) {
		if (input_relation.columns[i] != nullptr) {
			column_size = input_relation.columns[i]->column_length;
			break;
		}
	}

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		D_ASSERT(aggregates[aggr_idx]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();

		payload_idx = next_payload_idx;
		next_payload_idx = payload_idx + aggregate.children.size();

		if (aggregate.IsDistinct()) {
			continue;
		}

		if (aggregate.filter) {
			auto &bound_ref_expr = aggregate.filter->Cast<BoundReferenceExpression>();
			printf("Reading filter column from index %ld\n", bound_ref_expr.index);
		}

		idx_t payload_cnt = 0;

		printf("Aggregate type: %s\n", aggregate.function.name.c_str());
		if (aggregate.children.size() > 1) throw NotImplementedException("Aggregates with multiple children not supported yet");
		for (idx_t i = 0; i < aggregate.children.size(); ++i) {
			for (auto &child_expr : aggregate.children) {
				D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
				printf("Reading aggregation column from index %ld and passing it to index %ld in aggregation result\n", payload_idx + payload_cnt, aggr_idx);
				auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
				aggregate_column[aggr_idx] = HandleMaterializeExpression(input_relation.columns[payload_idx + payload_cnt], bound_ref_expr, gpuBufferManager);
				payload_cnt++;
			}
		}
	}

	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
		//here we probably have count(*) or sum(*) or something like that
		if (aggregate.children.size() == 0) {
			printf("Passing * aggregate to index %d in aggregation result\n", aggr_idx);
			aggregate_column[aggr_idx] = make_shared_ptr<GPUColumn>(column_size, ColumnType::INT64, nullptr);
		}
	}

	bool string_cudf_supported = true;
	for (int col = 0; col < aggregates.size(); col++) {
		// if types is VARCHAR, check the number of bytes
		if (aggregate_column[col]->data_wrapper.type == ColumnType::VARCHAR) {
			throw NotImplementedException("String column not supported");
		}
	}
	if (aggregate_column[0]->column_length > INT32_MAX) {
		HandleAggregateExpression(aggregate_column, gpuBufferManager, aggregates);
	} else {
		HandleAggregateExpressionCuDF(aggregate_column, gpuBufferManager, aggregates);
	}

	for (int aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		//TODO: has to fix this for columns with partially NULL values
		if (aggregation_result->columns[aggr_idx] == nullptr && aggregate_column[aggr_idx]->column_length > 0 && aggregate_column[aggr_idx]->data_wrapper.data != nullptr) {
			aggregation_result->columns[aggr_idx] = aggregate_column[aggr_idx];
			aggregation_result->columns[aggr_idx]->row_ids = nullptr;
			aggregation_result->columns[aggr_idx]->row_id_count = 0;
		}
	}

  	return SinkResultType::FINISHED;
}
  
SourceResultType
GPUPhysicalUngroupedAggregate::GetData(GPUIntermediateRelation &output_relation) const {
  for (int col = 0; col < aggregation_result->columns.size(); col++) {
    printf("Writing aggregation result to column %ld\n", col);
	output_relation.columns[col] = make_shared_ptr<GPUColumn>(aggregation_result->columns[col]->column_length, aggregation_result->columns[col]->data_wrapper.type, aggregation_result->columns[col]->data_wrapper.data);
  }

  return SourceResultType::FINISHED;
}

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