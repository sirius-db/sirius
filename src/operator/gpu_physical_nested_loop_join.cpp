#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/execution/operator/join/outer_join_marker.hpp"
#include "gpu_physical_nested_loop_join.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"

namespace duckdb {

template <typename T>
void ResolveTypeNestedLoopJoin(GPUColumn** &left_keys, GPUColumn** &right_keys, uint64_t* &count, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
		const vector<JoinCondition> &conditions, JoinType join_type, GPUBufferManager* gpuBufferManager) {
	int num_keys = conditions.size();
	T** left_data = new T*[num_keys];
	T** right_data = new T*[num_keys];

	for (int key = 0; key < num_keys; key++) {
		left_data[key] = reinterpret_cast<T*>(left_keys[key]->data_wrapper.data);
		right_data[key] = reinterpret_cast<T*>(right_keys[key]->data_wrapper.data);
	}
	size_t left_size = left_keys[0]->column_length;
	size_t right_size = right_keys[0]->column_length;


	int* condition_mode = new int[num_keys];
	for (int key = 0; key < num_keys; key++) {
		if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL || conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			condition_mode[key] = 0;
		} else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL || conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
			condition_mode[key] = 1;
		} else if (conditions[key].comparison == ExpressionType::COMPARE_LESSTHAN) { 
			condition_mode[key] = 2;
		} else if (conditions[key].comparison == ExpressionType::COMPARE_GREATERTHAN) { 
			condition_mode[key] = 3;
		} else {
			throw NotImplementedException("Unsupported comparison type");
		}
	}

	//TODO: Need to handle special case for unique keys for better performance
	if (join_type == JoinType::INNER) {
		// printGPUColumn<T>(left_data, 100, 0);
		nestedLoopJoin<T>(left_data, right_data, row_ids_left, row_ids_right, count, left_size, right_size, condition_mode, num_keys);
	} else {
		throw NotImplementedException("Unsupported join type");
	}
}

void
HandleNestedLoopJoin(GPUColumn** &left_keys, GPUColumn** &right_keys, uint64_t* &count, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
		const vector<JoinCondition> &conditions, JoinType join_type, GPUBufferManager* gpuBufferManager) {
    switch(left_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
		ResolveTypeNestedLoopJoin<uint64_t>(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);
		break;
      case ColumnType::FLOAT64:
	  	ResolveTypeNestedLoopJoin<double>(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);
		break;
      default:
        throw NotImplementedException("Unsupported column type");
    }
}

GPUPhysicalNestedLoopJoin::GPUPhysicalNestedLoopJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left,
                                               unique_ptr<GPUPhysicalOperator> right, vector<JoinCondition> cond,
                                               JoinType join_type, idx_t estimated_cardinality)
    // : PhysicalComparisonJoin(op, PhysicalOperatorType::NESTED_LOOP_JOIN, std::move(cond), join_type,
    //                          estimated_cardinality) {
    : GPUPhysicalOperator(PhysicalOperatorType::NESTED_LOOP_JOIN, op.types, estimated_cardinality), join_type(join_type) {
	conditions.resize(cond.size());
	// we reorder conditions so the ones with COMPARE_EQUAL occur first
	idx_t equal_position = 0;
	idx_t other_position = cond.size() - 1;
	for (idx_t i = 0; i < cond.size(); i++) {
		if (cond[i].comparison == ExpressionType::COMPARE_EQUAL ||
		    cond[i].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// COMPARE_EQUAL and COMPARE_NOT_DISTINCT_FROM, move to the start
			conditions[equal_position++] = std::move(cond[i]);
		} else {
			// other expression, move to the end
			conditions[other_position--] = std::move(cond[i]);
		}
	}

	children.push_back(std::move(left));
	children.push_back(std::move(right));

    // for (auto &child : children) {
    //     for (auto &type : child->GetTypes()) {
    //         condition_types.push_back(type);
    //     }
    // }

    right_temp_data = new GPUIntermediateRelation(children[1]->GetTypes().size());

}

bool GPUPhysicalNestedLoopJoin::IsSupported(const vector<JoinCondition> &conditions, JoinType join_type) {
	if (join_type == JoinType::MARK) {
		return true;
	}
	for (auto &cond : conditions) {
		if (cond.left->return_type.InternalType() == PhysicalType::STRUCT ||
		    cond.left->return_type.InternalType() == PhysicalType::LIST ||
		    cond.left->return_type.InternalType() == PhysicalType::ARRAY) {
			return false;
		}
	}
	// To avoid situations like https://github.com/duckdb/duckdb/issues/10046
	// If there is an equality in the conditions, a hash join is planned
	// with one condition, we can use mark join logic, otherwise we should use physical blockwise nl join
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		return conditions.size() == 1;
	}
	return true;
}

vector<LogicalType> GPUPhysicalNestedLoopJoin::GetJoinTypes() const {
	vector<LogicalType> result;
	for (auto &op : conditions) {
		result.push_back(op.right->return_type);
	}
	return result;
}

SinkResultType 
GPUPhysicalNestedLoopJoin::Sink(GPUIntermediateRelation &input_relation) const {

	auto start = std::chrono::high_resolution_clock::now();

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
        printf("Reading join key from right side from idx %ld\n", join_key_index);
	}

    for (int i = 0; i < input_relation.columns.size(); i++) {
        printf("Passing column idx %d from right side to idx %d in right temp relation\n", i, i);
		right_temp_data->columns[i] = new GPUColumn(input_relation.columns[i]->column_length, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data);
		right_temp_data->columns[i]->row_ids = input_relation.columns[i]->row_ids;
		right_temp_data->columns[i]->row_id_count = input_relation.columns[i]->row_id_count;
    }

	//measure time
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Result collector time: %.2f ms\n", duration.count()/1000.0);
	
	return SinkResultType::FINISHED;

}

OperatorResultType 
GPUPhysicalNestedLoopJoin::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {

	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
		// simple joins can have max STANDARD_VECTOR_SIZE matches per chunk
		throw NotImplementedException("Unimplemented type " + JoinTypeToString(join_type) + " for nested loop join!");
		ResolveSimpleJoin(input_relation, output_relation);
		return OperatorResultType::FINISHED;
	case JoinType::LEFT:
	case JoinType::OUTER:
	case JoinType::RIGHT:
		throw NotImplementedException("Unimplemented type " + JoinTypeToString(join_type) + " for nested loop join!");
		return OperatorResultType::FINISHED;
	case JoinType::INNER:
		return ResolveComplexJoin(input_relation, output_relation);
	default:
		throw NotImplementedException("Unimplemented type " + JoinTypeToString(join_type) + " for nested loop join!");
	}
}

void GPUPhysicalNestedLoopJoin::ResolveSimpleJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.left->Cast<BoundReferenceExpression>().index;
        printf("Reading join key from left side from index %ld\n", join_key_index);
        input_relation.checkLateMaterialization(join_key_index);
	}

	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::MARK) {
		printf("Writing row IDs from left side to output relation\n");
		uint64_t* left_row_ids = new uint64_t[1];
		for (idx_t i = 0; i < input_relation.column_count; i++) {
			printf("Passing column idx %ld from LHS (late materialized) to idx %ld in output relation\n", i, i);
			output_relation.columns[i] = input_relation.columns[i];
			output_relation.columns[i]->row_ids = left_row_ids;
		}
	} else {
        throw NotImplementedException("Unimplemented type for simple nested loop join!");
    }
}

OperatorResultType 
GPUPhysicalNestedLoopJoin::ResolveComplexJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	GPUColumn** left_keys = new GPUColumn*[conditions.size()];
	GPUColumn** right_keys = new GPUColumn*[conditions.size()];
	for (int i = 0; i < conditions.size(); i++) {
		left_keys[i] = nullptr;
		right_keys[i] = nullptr;
	}
	uint64_t* count;
	uint64_t* row_ids_left = nullptr;
	uint64_t* row_ids_right = nullptr;

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.left->Cast<BoundReferenceExpression>().index;
        printf("Reading join key from left side from index %ld\n", join_key_index);
        // input_relation.checkLateMaterialization(join_key_index);
		left_keys[cond_idx] = HandleMaterializeExpression(input_relation.columns[join_key_index], condition.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
	}

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
        printf("Reading join key from right side from index %ld\n", join_key_index);
        // right_temp_data->checkLateMaterialization(join_key_index);
		right_keys[cond_idx] = HandleMaterializeExpression(right_temp_data->columns[join_key_index], condition.right->Cast<BoundReferenceExpression>(), gpuBufferManager);
	}

	//check if all probe keys are int64 or all the probe keys are float64
	bool all_int64 = true;
	bool all_float64 = true;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		if (left_keys[cond_idx]->data_wrapper.type != ColumnType::INT64) {
			all_int64 = false;
		}
		if (left_keys[cond_idx]->data_wrapper.type != ColumnType::FLOAT64) {
			all_float64 = false;
		}
	}
	if (!all_int64 && !all_float64) {
		throw NotImplementedException("Hash join only supports integer or float64 keys");
	}

	if (join_type == JoinType::INNER) {
		printf("Nested loop join\n");
		count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
		HandleNestedLoopJoin(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);

		vector<column_t> rhs_output_columns;
		for (idx_t i = 0; i < right_temp_data->columns.size(); i++) rhs_output_columns.push_back(i);

		if (count[0] == 0) throw NotImplementedException("No match found in nested loop join");
		printf("Writing row IDs from LHS to output relation\n");
		HandleMaterializeRowIDs(input_relation, output_relation, count[0], row_ids_left, gpuBufferManager, false);
		printf("Writing row IDs from RHS to output relation\n");
		HandleMaterializeRowIDsRHS(*right_temp_data, output_relation, rhs_output_columns, input_relation.column_count, count[0], row_ids_right, gpuBufferManager, false);

	} else {
        throw NotImplementedException("Unimplemented type for complex nested loop join!");
    }

    return OperatorResultType::FINISHED;
}

SourceResultType 
GPUPhysicalNestedLoopJoin::GetData(GPUIntermediateRelation& output_relation) const {

    // check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	idx_t left_column_count = output_relation.columns.size() - right_temp_data->columns.size();
	if (join_type == JoinType::RIGHT || join_type == JoinType::OUTER) {
		for (idx_t col = 0; col < left_column_count; col++) {
			//pretend this to be NUll column from the left table (it should be NULL for the RIGHT join)
			output_relation.columns[col] = new GPUColumn(0, ColumnType::INT64, nullptr);
		}
	} else {
		throw InvalidInputException("Get data not supported for this join type");
	}

	for (idx_t i = 0; i < right_temp_data->columns.size(); i++) {
		printf("Writing right temp data column idx %ld to idx %ld in output relation\n", i, i);
		output_relation.columns[left_column_count + i] = right_temp_data->columns[i];
	}

    return SourceResultType::FINISHED;

}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalNestedLoopJoin::BuildJoinPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline, GPUPhysicalOperator &op,
                                      bool build_rhs) {
	op.op_state.reset();
	op.sink_state.reset();

	// 'current' is the probe pipeline: add this operator
	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, op);

	// save the last added pipeline to set up dependencies later (in case we need to add a child pipeline)
	vector<shared_ptr<GPUPipeline>> pipelines_so_far;
	meta_pipeline.GetPipelines(pipelines_so_far, false);
	auto &last_pipeline = *pipelines_so_far.back();

	if (build_rhs) {
		// on the RHS (build side), we construct a child MetaPipeline with this operator as its sink
		auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, op);
		child_meta_pipeline.Build(*op.children[1]);
	}

	// continue building the current pipeline on the LHS (probe side)
	op.children[0]->BuildPipelines(current, meta_pipeline);

	switch (op.type) {
	case PhysicalOperatorType::POSITIONAL_JOIN:
        throw NotImplementedException("POSITIONAL_JOIN is not implemented yet");
		// Positional joins are always outer
		meta_pipeline.CreateChildPipeline(current, op, last_pipeline);
		return;
	case PhysicalOperatorType::CROSS_PRODUCT:
        throw NotImplementedException("CROSS_PRODUCT is not implemented yet");
		return;
	default:
		break;
	}

	// Join can become a source operator if it's RIGHT/OUTER, or if the hash join goes out-of-core
	bool add_child_pipeline = false;
	auto &join_op = op.Cast<GPUPhysicalNestedLoopJoin>();
	if (join_op.IsSource()) {
		add_child_pipeline = true;
	}

	if (add_child_pipeline) {
		meta_pipeline.CreateChildPipeline(current, op, last_pipeline);
	}
}

void GPUPhysicalNestedLoopJoin::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	GPUPhysicalNestedLoopJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

} // namespace duckdb
