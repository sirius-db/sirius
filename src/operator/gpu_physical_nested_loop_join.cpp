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

	printf("im here\n");

	//TODO: Need to handle special case for unique keys for better performance
	if (join_type == JoinType::INNER) {
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

// bool PhysicalJoin::HasNullValues(DataChunk &chunk) {
// 	for (idx_t col_idx = 0; col_idx < chunk.ColumnCount(); col_idx++) {
// 		UnifiedVectorFormat vdata;
// 		chunk.data[col_idx].ToUnifiedFormat(chunk.size(), vdata);

// 		if (vdata.validity.AllValid()) {
// 			continue;
// 		}
// 		for (idx_t i = 0; i < chunk.size(); i++) {
// 			auto idx = vdata.sel->get_index(i);
// 			if (!vdata.validity.RowIsValid(idx)) {
// 				return true;
// 			}
// 		}
// 	}
// 	return false;
// }

// template <bool MATCH>
// static void ConstructSemiOrAntiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
// 	D_ASSERT(left.ColumnCount() == result.ColumnCount());
// 	// create the selection vector from the matches that were found
// 	idx_t result_count = 0;
// 	SelectionVector sel(STANDARD_VECTOR_SIZE);
// 	for (idx_t i = 0; i < left.size(); i++) {
// 		if (found_match[i] == MATCH) {
// 			sel.set_index(result_count++, i);
// 		}
// 	}
// 	// construct the final result
// 	if (result_count > 0) {
// 		// we only return the columns on the left side
// 		// project them using the result selection vector
// 		// reference the columns of the left side from the result
// 		result.Slice(left, sel, result_count);
// 	} else {
// 		result.SetCardinality(0);
// 	}
// }

// void PhysicalJoin::ConstructSemiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
// 	ConstructSemiOrAntiJoinResult<true>(left, result, found_match);
// }

// void PhysicalJoin::ConstructAntiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
// 	ConstructSemiOrAntiJoinResult<false>(left, result, found_match);
// }

// void PhysicalJoin::ConstructMarkJoinResult(DataChunk &join_keys, DataChunk &left, DataChunk &result, bool found_match[],
//                                            bool has_null) {
// 	// for the initial set of columns we just reference the left side
// 	result.SetCardinality(left);
// 	for (idx_t i = 0; i < left.ColumnCount(); i++) {
// 		result.data[i].Reference(left.data[i]);
// 	}
// 	auto &mark_vector = result.data.back();
// 	mark_vector.SetVectorType(VectorType::FLAT_VECTOR);
// 	// first we set the NULL values from the join keys
// 	// if there is any NULL in the keys, the result is NULL
// 	auto bool_result = FlatVector::GetData<bool>(mark_vector);
// 	auto &mask = FlatVector::Validity(mark_vector);
// 	for (idx_t col_idx = 0; col_idx < join_keys.ColumnCount(); col_idx++) {
// 		UnifiedVectorFormat jdata;
// 		join_keys.data[col_idx].ToUnifiedFormat(join_keys.size(), jdata);
// 		if (!jdata.validity.AllValid()) {
// 			for (idx_t i = 0; i < join_keys.size(); i++) {
// 				auto jidx = jdata.sel->get_index(i);
// 				mask.Set(i, jdata.validity.RowIsValid(jidx));
// 			}
// 		}
// 	}
// 	// now set the remaining entries to either true or false based on whether a match was found
// 	if (found_match) {
// 		for (idx_t i = 0; i < left.size(); i++) {
// 			bool_result[i] = found_match[i];
// 		}
// 	} else {
// 		memset(bool_result, 0, sizeof(bool) * left.size());
// 	}
// 	// if the right side contains NULL values, the result of any FALSE becomes NULL
// 	if (has_null) {
// 		for (idx_t i = 0; i < left.size(); i++) {
// 			if (!bool_result[i]) {
// 				mask.SetInvalid(i);
// 			}
// 		}
// 	}
// }

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

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
// class NestedLoopJoinLocalState : public LocalSinkState {
// public:
// 	explicit NestedLoopJoinLocalState(ClientContext &context, const vector<JoinCondition> &conditions)
// 	    : rhs_executor(context) {
// 		vector<LogicalType> condition_types;
// 		for (auto &cond : conditions) {
// 			rhs_executor.AddExpression(*cond.right);
// 			condition_types.push_back(cond.right->return_type);
// 		}
// 		right_condition.Initialize(Allocator::Get(context), condition_types);
// 	}

// 	//! The chunk holding the right condition
// 	DataChunk right_condition;
// 	//! The executor of the RHS condition
// 	ExpressionExecutor rhs_executor;
// };

// class NestedLoopJoinGlobalState : public GlobalSinkState {
// public:
// 	explicit NestedLoopJoinGlobalState(ClientContext &context, const GPUPhysicalNestedLoopJoin &op)
// 	    : right_payload_data(context, op.children[1]->types), right_condition_data(context, op.GetJoinTypes()),
// 	      has_null(false), right_outer(PropagatesBuildSide(op.join_type)) {
// 	}

// 	mutex nj_lock;
// 	//! Materialized data of the RHS
// 	ColumnDataCollection right_payload_data;
// 	//! Materialized join condition of the RHS
// 	ColumnDataCollection right_condition_data;
// 	//! Whether or not the RHS of the nested loop join has NULL values
// 	atomic<bool> has_null;
// 	//! A bool indicating for each tuple in the RHS if they found a match (only used in FULL OUTER JOIN)
// 	OuterJoinMarker right_outer;
// };

vector<LogicalType> GPUPhysicalNestedLoopJoin::GetJoinTypes() const {
	vector<LogicalType> result;
	for (auto &op : conditions) {
		result.push_back(op.right->return_type);
	}
	return result;
}

// SinkResultType GPUPhysicalNestedLoopJoin::Sink(ExecutionContext &context, DataChunk &chunk,
//                                             OperatorSinkInput &input) const {
SinkResultType 
GPUPhysicalNestedLoopJoin::Sink(GPUIntermediateRelation &input_relation) const {
	// auto &gstate = input.global_state.Cast<NestedLoopJoinGlobalState>();
	// auto &nlj_state = input.local_state.Cast<NestedLoopJoinLocalState>();

	// resolve the join expression of the right side
	// nlj_state.right_condition.Reset();
	// nlj_state.rhs_executor.Execute(chunk, nlj_state.right_condition);

	// if we have not seen any NULL values yet, and we are performing a MARK join, check if there are NULL values in
	// this chunk
	// if (join_type == JoinType::MARK && !gstate.has_null) {
	// 	if (HasNullValues(nlj_state.right_condition)) {
	// 		gstate.has_null = true;
	// 	}
	// }

	// // append the payload data and the conditions
	// lock_guard<mutex> nj_guard(gstate.nj_lock);
	// gstate.right_payload_data.Append(chunk);
	// gstate.right_condition_data.Append(nlj_state.right_condition);

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
        printf("Reading join key from right side from idx %ld\n", join_key_index);
        // input_relation.checkLateMaterialization(join_key_index);
		// right_key[cond_idx] = HandleMaterializeExpression(input_relation.columns[join_key_index], condition.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
	}

    for (int i = 0; i < input_relation.columns.size(); i++) {
        printf("Passing column idx %d from right side to idx %d in right temp relation\n", i, i);
        // right_temp_data->columns[i] = input_relation.columns[i];
        // right_temp_data->columns[i]->row_ids = new uint64_t[1];
		right_temp_data->columns[i] = new GPUColumn(input_relation.columns[i]->column_length, input_relation.columns[i]->data_wrapper.type, input_relation.columns[i]->data_wrapper.data);
		right_temp_data->columns[i]->row_ids = input_relation.columns[i]->row_ids;
		right_temp_data->columns[i]->row_id_count = input_relation.columns[i]->row_id_count;
    }

	return SinkResultType::FINISHED;

}

// SinkCombineResultType GPUPhysicalNestedLoopJoin::Combine(ExecutionContext &context,
//                                                       OperatorSinkCombineInput &input) const {
// 	auto &state = input.local_state.Cast<NestedLoopJoinLocalState>();
// 	auto &client_profiler = QueryProfiler::Get(context.client);

// 	context.thread.profiler.Flush(*this, state.rhs_executor, "rhs_executor", 1);
// 	client_profiler.Flush(context.thread.profiler);

// 	return SinkCombineResultType::FINISHED;
// }

// SinkFinalizeType GPUPhysicalNestedLoopJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
//                                                   OperatorSinkFinalizeInput &input) const {
// 	auto &gstate = input.global_state.Cast<NestedLoopJoinGlobalState>();
// 	gstate.right_outer.Initialize(gstate.right_payload_data.Count());
// 	if (gstate.right_payload_data.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
// 		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
// 	}
// 	return SinkFinalizeType::READY;
// }

// unique_ptr<GlobalSinkState> GPUPhysicalNestedLoopJoin::GetGlobalSinkState(ClientContext &context) const {
// 	return make_uniq<NestedLoopJoinGlobalState>(context, *this);
// }

// unique_ptr<LocalSinkState> GPUPhysicalNestedLoopJoin::GetLocalSinkState(ExecutionContext &context) const {
// 	return make_uniq<NestedLoopJoinLocalState>(context.client, conditions);
// }

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
// class PhysicalNestedLoopJoinState : public CachingOperatorState {
// public:
// 	PhysicalNestedLoopJoinState(ClientContext &context, const GPUPhysicalNestedLoopJoin &op,
// 	                            const vector<JoinCondition> &conditions)
// 	    : fetch_next_left(true), fetch_next_right(false), lhs_executor(context), left_tuple(0), right_tuple(0),
// 	      left_outer(IsLeftOuterJoin(op.join_type)) {
// 		vector<LogicalType> condition_types;
// 		for (auto &cond : conditions) {
// 			lhs_executor.AddExpression(*cond.left);
// 			condition_types.push_back(cond.left->return_type);
// 		}
// 		auto &allocator = Allocator::Get(context);
// 		left_condition.Initialize(allocator, condition_types);
// 		right_condition.Initialize(allocator, condition_types);
// 		right_payload.Initialize(allocator, op.children[1]->GetTypes());
// 		left_outer.Initialize(STANDARD_VECTOR_SIZE);
// 	}

// 	bool fetch_next_left;
// 	bool fetch_next_right;
// 	DataChunk left_condition;
// 	//! The executor of the LHS condition
// 	ExpressionExecutor lhs_executor;

// 	ColumnDataScanState condition_scan_state;
// 	ColumnDataScanState payload_scan_state;
// 	DataChunk right_condition;
// 	DataChunk right_payload;

// 	idx_t left_tuple;
// 	idx_t right_tuple;

// 	OuterJoinMarker left_outer;

// public:
// 	void Finalize(const GPUPhysicalOperator &op, ExecutionContext &context) override {
// 		context.thread.profiler.Flush(op, lhs_executor, "lhs_executor", 0);
// 	}
// };

// unique_ptr<OperatorState> GPUPhysicalNestedLoopJoin::GetOperatorState(ExecutionContext &context) const {
// 	return make_uniq<PhysicalNestedLoopJoinState>(context.client, *this, conditions);
// }

// OperatorResultType GPUPhysicalNestedLoopJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
//                                                            DataChunk &chunk, GlobalOperatorState &gstate_p,
//                                                            OperatorState &state_p) const {

OperatorResultType 
GPUPhysicalNestedLoopJoin::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
	// auto &gstate = sink_state->Cast<NestedLoopJoinGlobalState>();

	// if (gstate.right_payload_data.Count() == 0) {
	// 	// empty RHS
	// 	if (!EmptyResultIfRHSIsEmpty()) {
	// 		ConstructEmptyJoinResult(join_type, gstate.has_null, input, chunk);
	// 		return OperatorResultType::NEED_MORE_INPUT;
	// 	} else {
	// 		return OperatorResultType::FINISHED;
	// 	}
	// }

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

// void GPUPhysicalNestedLoopJoin::ResolveSimpleJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
//                                                OperatorState &state_p) const {
void GPUPhysicalNestedLoopJoin::ResolveSimpleJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
// 	auto &state = state_p.Cast<PhysicalNestedLoopJoinState>();
// 	auto &gstate = sink_state->Cast<NestedLoopJoinGlobalState>();

// 	// resolve the left join condition for the current chunk
// 	state.left_condition.Reset();
// 	state.lhs_executor.Execute(input, state.left_condition);

// 	bool found_match[STANDARD_VECTOR_SIZE] = {false};
// 	NestedLoopJoinMark::Perform(state.left_condition, gstate.right_condition_data, found_match, conditions);
// 	switch (join_type) {
// 	case JoinType::MARK:
// 		// now construct the mark join result from the found matches
// 		PhysicalJoin::ConstructMarkJoinResult(state.left_condition, input, chunk, found_match, gstate.has_null);
// 		break;
// 	case JoinType::SEMI:
// 		// construct the semi join result from the found matches
// 		PhysicalJoin::ConstructSemiJoinResult(input, chunk, found_match);
// 		break;
// 	case JoinType::ANTI:
// 		// construct the anti join result from the found matches
// 		PhysicalJoin::ConstructAntiJoinResult(input, chunk, found_match);
// 		break;
// 	default:
// 		throw NotImplementedException("Unimplemented type for simple nested loop join!");
// 	}

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

// OperatorResultType GPUPhysicalNestedLoopJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &input,
//                                                               DataChunk &chunk, OperatorState &state_p) const {
OperatorResultType 
GPUPhysicalNestedLoopJoin::ResolveComplexJoin(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
// 	auto &state = state_p.Cast<PhysicalNestedLoopJoinState>();
// 	auto &gstate = sink_state->Cast<NestedLoopJoinGlobalState>();

// 	idx_t match_count;
// 	do {
// 		if (state.fetch_next_right) {
// 			// we exhausted the chunk on the right: move to the next chunk on the right
// 			state.left_tuple = 0;
// 			state.right_tuple = 0;
// 			state.fetch_next_right = false;
// 			// check if we exhausted all chunks on the RHS
// 			if (gstate.right_condition_data.Scan(state.condition_scan_state, state.right_condition)) {
// 				if (!gstate.right_payload_data.Scan(state.payload_scan_state, state.right_payload)) {
// 					throw InternalException("Nested loop join: payload and conditions are unaligned!?");
// 				}
// 				if (state.right_condition.size() != state.right_payload.size()) {
// 					throw InternalException("Nested loop join: payload and conditions are unaligned!?");
// 				}
// 			} else {
// 				// we exhausted all chunks on the right: move to the next chunk on the left
// 				state.fetch_next_left = true;
// 				if (state.left_outer.Enabled()) {
// 					// left join: before we move to the next chunk, see if we need to output any vectors that didn't
// 					// have a match found
// 					state.left_outer.ConstructLeftJoinResult(input, chunk);
// 					state.left_outer.Reset();
// 				}
// 				return OperatorResultType::NEED_MORE_INPUT;
// 			}
// 		}
// 		if (state.fetch_next_left) {
// 			// resolve the left join condition for the current chunk
// 			state.left_condition.Reset();
// 			state.lhs_executor.Execute(input, state.left_condition);

// 			state.left_tuple = 0;
// 			state.right_tuple = 0;
// 			gstate.right_condition_data.InitializeScan(state.condition_scan_state);
// 			gstate.right_condition_data.Scan(state.condition_scan_state, state.right_condition);

// 			gstate.right_payload_data.InitializeScan(state.payload_scan_state);
// 			gstate.right_payload_data.Scan(state.payload_scan_state, state.right_payload);
// 			state.fetch_next_left = false;
// 		}
// 		// now we have a left and a right chunk that we can join together
// 		// note that we only get here in the case of a LEFT, INNER or FULL join
// 		auto &left_chunk = input;
// 		auto &right_condition = state.right_condition;
// 		auto &right_payload = state.right_payload;

// 		// sanity check
// 		left_chunk.Verify();
// 		right_condition.Verify();
// 		right_payload.Verify();

// 		// now perform the join
// 		SelectionVector lvector(STANDARD_VECTOR_SIZE), rvector(STANDARD_VECTOR_SIZE);
// 		match_count = NestedLoopJoinInner::Perform(state.left_tuple, state.right_tuple, state.left_condition,
// 		                                           right_condition, lvector, rvector, conditions);
// 		// we have finished resolving the join conditions
// 		if (match_count > 0) {
// 			// we have matching tuples!
// 			// construct the result
// 			state.left_outer.SetMatches(lvector, match_count);
// 			gstate.right_outer.SetMatches(rvector, match_count, state.condition_scan_state.current_row_index);

// 			chunk.Slice(input, lvector, match_count);
// 			chunk.Slice(right_payload, rvector, match_count, input.ColumnCount());
// 		}

// 		// check if we exhausted the RHS, if we did we need to move to the next right chunk in the next iteration
// 		if (state.right_tuple >= right_condition.size()) {
// 			state.fetch_next_right = true;
// 		}
// 	} while (match_count == 0);
// 	return OperatorResultType::HAVE_MORE_OUTPUT;

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

	if (join_type == JoinType::INNER) {
	// if (join_type == JoinType::LEFT || join_type == JoinType::INNER || join_type == JoinType::OUTER || join_type == JoinType::RIGHT) {
		// printf("Writing row IDs from left side to output relation\n");
		// uint64_t* left_row_ids = new uint64_t[1];
		// for (idx_t i = 0; i < input_relation.column_count; i++) {
		// 	printf("Passing column idx %ld from LHS (late materialized) to idx %ld in output relation\n", i, i);
		// 	output_relation.columns[i] = input_relation.columns[i];
		// 	output_relation.columns[i]->row_ids = left_row_ids;
		// }
        // printf("Writing row IDs from right side to output relation\n");
		// uint64_t* right_row_ids = new uint64_t[1];
		// for (idx_t i = 0; i < right_temp_data->columns.size(); i++) {
		// 	printf("Passing column idx %ld from right_temp_data to idx %ld in output relation\n", i, input_relation.column_count + i);
		// 	output_relation.columns[input_relation.column_count + i] = right_temp_data->columns[i];
		// 	output_relation.columns[input_relation.column_count + i]->row_ids = right_row_ids;
		// }

		printf("Nested loop join\n");
		count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
		HandleNestedLoopJoin(left_keys, right_keys, count, row_ids_left, row_ids_right, conditions, join_type, gpuBufferManager);

		vector<idx_t> rhs_output_columns;
		for (idx_t i = 0; i < right_temp_data->columns.size(); i++) rhs_output_columns.push_back(i);

		if (count[0] == 0) throw NotImplementedException("No match found in nested loop join");
		printf("Writing row IDs from LHS to output relation\n");
		HandleMaterializeRowIDs(input_relation, output_relation, count[0], row_ids_left, gpuBufferManager);
		printf("Writing row IDs from RHS to output relation\n");
		HandleMaterializeRowIDsRHS(*right_temp_data, output_relation, rhs_output_columns, input_relation.column_count, count[0], row_ids_right, gpuBufferManager);

	} else {
        throw NotImplementedException("Unimplemented type for complex nested loop join!");
    }

    return OperatorResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
// class NestedLoopJoinGlobalScanState : public GlobalSourceState {
// public:
// 	explicit NestedLoopJoinGlobalScanState(const GPUPhysicalNestedLoopJoin &op) : op(op) {
// 		D_ASSERT(op.sink_state);
// 		auto &sink = op.sink_state->Cast<NestedLoopJoinGlobalState>();
// 		sink.right_outer.InitializeScan(sink.right_payload_data, scan_state);
// 	}

// 	const GPUPhysicalNestedLoopJoin &op;
// 	OuterJoinGlobalScanState scan_state;

// public:
// 	idx_t MaxThreads() override {
// 		auto &sink = op.sink_state->Cast<NestedLoopJoinGlobalState>();
// 		return sink.right_outer.MaxThreads();
// 	}
// };

// class NestedLoopJoinLocalScanState : public LocalSourceState {
// public:
// 	explicit NestedLoopJoinLocalScanState(const GPUPhysicalNestedLoopJoin &op, NestedLoopJoinGlobalScanState &gstate) {
// 		D_ASSERT(op.sink_state);
// 		auto &sink = op.sink_state->Cast<NestedLoopJoinGlobalState>();
// 		sink.right_outer.InitializeScan(gstate.scan_state, scan_state);
// 	}

// 	OuterJoinLocalScanState scan_state;
// };

// unique_ptr<GlobalSourceState> GPUPhysicalNestedLoopJoin::GetGlobalSourceState(ClientContext &context) const {
// 	return make_uniq<NestedLoopJoinGlobalScanState>(*this);
// }

// unique_ptr<LocalSourceState> GPUPhysicalNestedLoopJoin::GetLocalSourceState(ExecutionContext &context,
//                                                                          GlobalSourceState &gstate) const {
// 	return make_uniq<NestedLoopJoinLocalScanState>(*this, gstate.Cast<NestedLoopJoinGlobalScanState>());
// }

// SourceResultType GPUPhysicalNestedLoopJoin::GetData(ExecutionContext &context, DataChunk &chunk,
//                                                  OperatorSourceInput &input) const {
SourceResultType 
GPUPhysicalNestedLoopJoin::GetData(GPUIntermediateRelation& output_relation) const {
	// D_ASSERT(PropagatesBuildSide(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	// auto &sink = sink_state->Cast<NestedLoopJoinGlobalState>();
	// auto &gstate = input.global_state.Cast<NestedLoopJoinGlobalScanState>();
	// auto &lstate = input.local_state.Cast<NestedLoopJoinLocalScanState>();

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan chunks we still need to output
	// sink.right_outer.Scan(gstate.scan_state, lstate.scan_state, chunk);

	// return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;

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
	printf("I'm casting here\n");
	printf("op type: %s\n", PhysicalOperatorToString(op.type).c_str());
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
