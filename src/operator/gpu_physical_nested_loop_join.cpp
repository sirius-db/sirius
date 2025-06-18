#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/execution/operator/join/outer_join_marker.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "gpu_physical_hash_join.hpp"
#include "gpu_physical_nested_loop_join.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"
#include "log/logging.hpp"

namespace duckdb {

template <typename T>
void ResolveTypeNestedLoopJoin(vector<shared_ptr<GPUColumn>> &left_keys, vector<shared_ptr<GPUColumn>> &right_keys, uint64_t* &count, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
		const vector<JoinCondition> &conditions, JoinType join_type, GPUBufferManager* gpuBufferManager) {
	int num_keys = conditions.size();
	T** left_data = gpuBufferManager->customCudaHostAlloc<T*>(num_keys);
	T** right_data = gpuBufferManager->customCudaHostAlloc<T*>(num_keys);

	for (int key = 0; key < num_keys; key++) {
		left_data[key] = reinterpret_cast<T*>(left_keys[key]->data_wrapper.data);
		right_data[key] = reinterpret_cast<T*>(right_keys[key]->data_wrapper.data);
	}
	size_t left_size = left_keys[0]->column_length;
	size_t right_size = right_keys[0]->column_length;


	int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
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
HandleNestedLoopJoin(vector<shared_ptr<GPUColumn>> &left_keys, vector<shared_ptr<GPUColumn>> &right_keys, uint64_t* &count, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
		const vector<JoinCondition> &conditions, JoinType join_type, GPUBufferManager* gpuBufferManager) {
    switch(left_keys[0]->data_wrapper.type.id()) {
      case GPUColumnTypeId::INT64:
		ResolveTypeNestedLoopJoin<uint64_t>(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);
		break;
      case GPUColumnTypeId::FLOAT64:
	  	ResolveTypeNestedLoopJoin<double>(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);
		break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleNestedLoopJoin`: %d",
																			static_cast<int>(left_keys[0]->data_wrapper.type.id()));
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

    right_temp_data = make_shared_ptr<GPUIntermediateRelation>(children[1]->GetTypes().size());

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
        SIRIUS_LOG_DEBUG("Reading join key from right side from idx {}", join_key_index);
	}

    for (int i = 0; i < input_relation.columns.size(); i++) {
        SIRIUS_LOG_DEBUG("Passing column idx {} from right side to idx {} in right temp relation", i, i);
		right_temp_data->columns[i] = make_shared_ptr<GPUColumn>(input_relation.columns[i]->column_length, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data);
		right_temp_data->columns[i]->row_ids = input_relation.columns[i]->row_ids;
		right_temp_data->columns[i]->row_id_count = input_relation.columns[i]->row_id_count;
    }

	//measure time
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Nested loop join Sink time: {:.2f} ms", duration.count()/1000.0);
	
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
        SIRIUS_LOG_DEBUG("Reading join key from left side from index {}", join_key_index);
        input_relation.checkLateMaterialization(join_key_index);
	}

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::MARK) {
		SIRIUS_LOG_DEBUG("Writing row IDs from left side to output relation");
		uint64_t* left_row_ids = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
		for (idx_t i = 0; i < input_relation.column_count; i++) {
			SIRIUS_LOG_DEBUG("Passing column idx {} from LHS (late materialized) to idx {} in output relation", i, i);
			output_relation.columns[i] = input_relation.columns[i];
			output_relation.columns[i]->row_ids = left_row_ids;
		}
	} else {
        throw NotImplementedException("Unimplemented type for simple nested loop join!");
    }
}

OperatorResultType 
GPUPhysicalNestedLoopJoin::ResolveComplexJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {

	auto start = std::chrono::high_resolution_clock::now();

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	vector<shared_ptr<GPUColumn>> left_keys(conditions.size());
	vector<shared_ptr<GPUColumn>> right_keys(conditions.size());
	for (int i = 0; i < conditions.size(); i++) {
		left_keys[i] = nullptr;
		right_keys[i] = nullptr;
	}
	uint64_t* count;
	uint64_t* row_ids_left = nullptr;
	uint64_t* row_ids_right = nullptr;

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		if (condition.left->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			auto join_key_index = condition.left->Cast<BoundReferenceExpression>().index;
			SIRIUS_LOG_DEBUG("Reading join key from left relation from index {}", join_key_index);
			left_keys[cond_idx] = HandleMaterializeExpression(input_relation.columns[join_key_index], condition.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
		} else if (condition.left->GetExpressionClass() == ExpressionClass::BOUND_CAST) {
			auto& child = condition.left->Cast<BoundCastExpression>().child;
			if (child->GetExpressionClass() != ExpressionClass::BOUND_REF) {
				throw NotImplementedException("Unsupported expression type of left join condition in nested loop join: %d",
																			static_cast<int>(child->GetExpressionClass()));
			}
			auto join_key_index = child->Cast<BoundReferenceExpression>().index;
			SIRIUS_LOG_DEBUG("Reading join key from left relation from index {}", join_key_index);
			left_keys[cond_idx] = HandleMaterializeExpression(input_relation.columns[join_key_index], child->Cast<BoundReferenceExpression>(), gpuBufferManager);
			// Perform cast
			auto from_cudf_column_view = left_keys[cond_idx]->convertToCudfColumn();
			auto to_cudf_type = sirius::GpuExpressionState::GetCudfType(condition.left->return_type);
			auto to_cudf_column = cudf::cast(from_cudf_column_view,
																			 to_cudf_type,
																			 rmm::cuda_stream_default,
																			 GPUBufferManager::GetInstance().mr);
			left_keys[cond_idx]->setFromCudfColumn(*to_cudf_column, false, nullptr, 0, gpuBufferManager);
		} else {
			throw NotImplementedException("Unsupported expression type of left join condition in nested loop join: %d",
																		static_cast<int>(condition.left->GetExpressionClass()));
		}
	}

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
			SIRIUS_LOG_DEBUG("Reading join key from right relation from index {}", join_key_index);
			right_keys[cond_idx] = HandleMaterializeExpression(right_temp_data->columns[join_key_index], condition.right->Cast<BoundReferenceExpression>(), gpuBufferManager);
		} else if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_CAST) {
			auto& child = condition.right->Cast<BoundCastExpression>().child;
			if (child->GetExpressionClass() != ExpressionClass::BOUND_REF) {
				throw NotImplementedException("Unsupported expression type of right join condition in nested loop join: %d",
																			static_cast<int>(child->GetExpressionClass()));
			}
			auto join_key_index = child->Cast<BoundReferenceExpression>().index;
			SIRIUS_LOG_DEBUG("Reading join key from right relation from index {}", join_key_index);
			right_keys[cond_idx] = HandleMaterializeExpression(right_temp_data->columns[join_key_index], child->Cast<BoundReferenceExpression>(), gpuBufferManager);
			// Perform cast
			auto from_cudf_column_view = right_keys[cond_idx]->convertToCudfColumn();
			auto to_cudf_type = sirius::GpuExpressionState::GetCudfType(condition.right->return_type);
			auto to_cudf_column = cudf::cast(from_cudf_column_view,
																			 to_cudf_type,
																			 rmm::cuda_stream_default,
																			 GPUBufferManager::GetInstance().mr);
			right_keys[cond_idx]->setFromCudfColumn(*to_cudf_column, false, nullptr, 0, gpuBufferManager);
		} else {
			throw NotImplementedException("Unsupported expression type of right join condition in nested loop join: %d",
																		static_cast<int>(condition.right->GetExpressionClass()));
		}
	}

	//check if all probe keys are int64 or all the probe keys are float64
	bool all_int64 = true;
	bool all_float64 = true;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		if (left_keys[cond_idx]->data_wrapper.type.id() != GPUColumnTypeId::INT64) {
			all_int64 = false;
		}
		if (left_keys[cond_idx]->data_wrapper.type.id() != GPUColumnTypeId::FLOAT64) {
			all_float64 = false;
		}
	}
	SIRIUS_LOG_DEBUG("Nested loop join");
	if (!all_int64 && !all_float64) {
		// Not supported by Sirius implementation, use cudf instead
		if (join_type == JoinType::INNER) {
			cudf_mixed_or_conditional_inner_join(left_keys, right_keys, conditions, join_type, row_ids_left, row_ids_right, count);
		} else {
			throw NotImplementedException("Unimplemented type for complex nested loop join using cudf!");
    }
	} else {
		// Supported by Sirius implementation
		if (join_type == JoinType::INNER) {
			HandleNestedLoopJoin(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);
		} else {
			throw NotImplementedException("Unimplemented type for complex nested loop join not using cudf!");
    }
	}
	vector<column_t> rhs_output_columns;
	for (idx_t i = 0; i < right_temp_data->columns.size(); i++) rhs_output_columns.push_back(i);

	// if (count[0] == 0) throw NotImplementedException("No match found in nested loop join");
	SIRIUS_LOG_DEBUG("Writing row IDs from LHS to output relation");
	HandleMaterializeRowIDs(input_relation, output_relation, count[0], row_ids_left, gpuBufferManager, false);
	SIRIUS_LOG_DEBUG("Writing row IDs from RHS to output relation");
	HandleMaterializeRowIDsRHS(*right_temp_data, output_relation, rhs_output_columns, input_relation.column_count, count[0], row_ids_right, gpuBufferManager, false);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Nested loop join Execute time: {:.2f} ms", duration.count()/1000.0);

	return OperatorResultType::FINISHED;
}

SourceResultType 
GPUPhysicalNestedLoopJoin::GetData(GPUIntermediateRelation& output_relation) const {

	auto start = std::chrono::high_resolution_clock::now();

    // check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	idx_t left_column_count = output_relation.columns.size() - right_temp_data->columns.size();
	if (join_type == JoinType::RIGHT || join_type == JoinType::OUTER) {
		for (idx_t col = 0; col < left_column_count; col++) {
			//pretend this to be NUll column from the left table (it should be NULL for the RIGHT join)
			output_relation.columns[col] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), nullptr);
		}
	} else {
		throw InvalidInputException("Get data not supported for this join type");
	}

	for (idx_t i = 0; i < right_temp_data->columns.size(); i++) {
		SIRIUS_LOG_DEBUG("Writing right temp data column idx {} to idx {} in output relation", i, left_column_count + i);
		output_relation.columns[left_column_count + i] = right_temp_data->columns[i];
	}

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Nested loop join GetData time: {:.2f} ms", duration.count()/1000.0);
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
