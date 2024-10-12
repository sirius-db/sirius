#include "operator/gpu_physical_hash_join.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

GPUPhysicalHashJoin::GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left,
                                   unique_ptr<GPUPhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   const vector<idx_t> &left_projection_map, const vector<idx_t> &right_projection_map,
                                   vector<LogicalType> delim_types, idx_t estimated_cardinality)
    // : PhysicalComparisonJoin(op, PhysicalOperatorType::HASH_JOIN, std::move(cond), join_type, estimated_cardinality),
    //   delim_types(std::move(delim_types)), perfect_join_statistics(std::move(perfect_join_stats))
	// : PhysicalJoin(op, type, join_type, estimated_cardinality)
	// : CachingPhysicalOperator(type, op.types, estimated_cardinality), join_type(join_type)
    : GPUPhysicalOperator(PhysicalOperatorType::HASH_JOIN, op.types, estimated_cardinality), join_type(join_type) {

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

	D_ASSERT(left_projection_map.empty());

	children.push_back(std::move(left));
	children.push_back(std::move(right));

	// Collect condition types, and which conditions are just references (so we won't duplicate them in the payload)
	unordered_map<idx_t, idx_t> build_columns_in_conditions;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		condition_types.push_back(condition.left->return_type);
		if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			build_columns_in_conditions.emplace(condition.right->Cast<BoundReferenceExpression>().index, cond_idx);
		}
	}

	// For ANTI, SEMI and MARK join, we only need to store the keys, so for these the payload/RHS types are empty
	if (join_type == JoinType::ANTI || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
		hash_table_result = new GPUIntermediateRelation(0, build_columns_in_conditions.size());
		return;
	}

	auto &rhs_input_types = children[1]->GetTypes();

	// Create a projection map for the RHS (if it was empty), for convenience
	auto right_projection_map_copy = right_projection_map;
	if (right_projection_map_copy.empty()) {
		right_projection_map_copy.reserve(rhs_input_types.size());
		for (idx_t i = 0; i < rhs_input_types.size(); i++) {
			right_projection_map_copy.emplace_back(i);
		}
	}

	// Now fill payload expressions/types and RHS columns/types
	for (auto &rhs_col : right_projection_map_copy) {
		auto &rhs_col_type = rhs_input_types[rhs_col];

		auto it = build_columns_in_conditions.find(rhs_col);
		if (it == build_columns_in_conditions.end()) {
			// This rhs column is not a join key
			payload_column_idxs.push_back(rhs_col);
			payload_types.push_back(rhs_col_type);
			rhs_output_columns.push_back(condition_types.size() + payload_types.size() - 1);
		} else {
			// This rhs column is a join key
			rhs_output_columns.push_back(it->second);
		}
		rhs_output_types.push_back(rhs_col_type);
	}

	hash_table_result = new GPUIntermediateRelation(0, rhs_output_columns.size());

};

// SourceResultType
// GPUPhysicalHashJoin::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {
SourceResultType
GPUPhysicalHashJoin::GetData(GPUIntermediateRelation &output_relation) const {
	idx_t left_column_count = output_relation.columns.size() - hash_table_result->columns.size();
	if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		left_column_count = 0;
	} else if (join_type == JoinType::RIGHT) {
		for (idx_t col = 0; col < left_column_count; col++) {
			//pretend this to be NUll column from the left table (it should be NULL for the RIGHT join)
			output_relation.columns[col] = new GPUColumn(0, ColumnType::INT64, nullptr);
		}
	} else {
		throw InvalidInputException("Get data not supported for this join type");
	}

	throw NotImplementedException("Not implemented yet");
	for (idx_t i = 0; i < rhs_output_columns.size(); i++) {
		const auto rhs_col = rhs_output_columns[i];
		printf("Writing hash_table column %ld to column %ld\n", i, rhs_col);
		output_relation.columns[left_column_count + rhs_col] = hash_table_result->columns[i];
	}

	return SourceResultType::FINISHED;
}

//probing hash table
// OperatorResultType
// GPUPhysicalHashJoin::Execute(ExecutionContext &context, GPUIntermediateRelation &input_relation, 
// 	GPUIntermediateRelation &output_relation, GlobalOperatorState &gstate, OperatorState &state) const {
OperatorResultType
GPUPhysicalHashJoin::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {
    //Read the key from input relation
    //Check if late materialization is needed
    //Probe the hash table
	//Create output relation
    //Write the result to output relation
	if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		// for RIGHT SEMI and RIGHT ANTI joins, the output is the RHS
		// we only need to output the RHS columns
		// the LHS columns are NULL
		if (output_relation.columns.size() != rhs_output_columns.size()) {
			throw InvalidInputException("Wrong input size");
		}
	} else if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		// for SEMI and ANTI join, the output is the LHS
		// we only need to output the LHS columns
		// the RHS columns are NULL
		if (output_relation.columns.size() != input_relation.columns.size()) {
			throw InvalidInputException("Wrong input size");
		}
	} else if (join_type == JoinType::RIGHT || join_type == JoinType::LEFT || join_type == JoinType::INNER || join_type == JoinType::OUTER) {
		// for INNER and OUTER join, we output all columns
		if (output_relation.columns.size() != input_relation.columns.size() + rhs_output_columns.size()) {
			throw InvalidInputException("Wrong input size");
		}
	} else if (join_type == JoinType::MARK) {
		// for MARK join, we output all columns from the LHS and one extra boolean column
		if (output_relation.columns.size() != input_relation.columns.size() + 1) {
			throw InvalidInputException("Wrong input size");
		}
	} else {
		throw InvalidInputException("Unsupported join type");
	}

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	uint64_t* probe_key = nullptr;
	uint64_t size;
	uint64_t* count;
	uint64_t* row_ids_left = nullptr;
	uint64_t* row_ids_right = nullptr;
	if (conditions.size() > 1) throw NotImplementedException("Multiple conditions not supported yet");

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.left->Cast<BoundReferenceExpression>().index;
        printf("Reading join key for probing hash table from index %ld\n", join_key_index);
        input_relation.checkLateMaterialization(join_key_index);
		if (input_relation.checkLateMaterialization(join_key_index)) {
			uint64_t* temp = reinterpret_cast<uint64_t*> (input_relation.columns[join_key_index]->data_wrapper.data);
			uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[join_key_index]->row_ids);
			probe_key = gpuBufferManager->customCudaMalloc<uint64_t>(input_relation.columns[join_key_index]->row_id_count, 0, 0);
			materializeExpression<uint64_t>(temp, probe_key, row_ids_input, input_relation.columns[join_key_index]->row_id_count);
			size = input_relation.columns[join_key_index]->row_id_count;
		} else {
			probe_key = reinterpret_cast<uint64_t*> (input_relation.columns[join_key_index]->data_wrapper.data);
			size = input_relation.columns[join_key_index]->column_length;
		}
		count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
		probeHashTable<uint64_t>(probe_key, gpu_hash_table, ht_len, row_ids_left, row_ids_right, count, size, 0);
	}
    printf("Probing hash table\n");
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::MARK || join_type == JoinType::INNER || join_type == JoinType::OUTER || join_type == JoinType::RIGHT || join_type == JoinType::LEFT) {
		printf("Writing row IDs from LHS to output relation\n");
		// uint64_t* left_row_ids = new uint64_t[1];
		for (idx_t i = 0; i < input_relation.column_count; i++) {
			printf("Passing column idx %ld from LHS (late materialized) to idx %ld in output relation\n", i, i);
			output_relation.columns[i] = input_relation.columns[i];
			// output_relation.columns[i]->row_ids = left_row_ids;
            if (row_ids_left) {
                if (input_relation.columns[i]->row_ids == nullptr) {
                    output_relation.columns[i]->row_ids = row_ids_left;
                } else {
                    uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[i]->row_ids);
                    uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
                    materializeExpression<uint64_t>(row_ids_input, new_row_ids, row_ids_left, count[0]);
                    output_relation.columns[i]->row_ids = new_row_ids;
                }
            }
		}
	}

	if (join_type == JoinType::MARK) {
		printf("Writing boolean column to output relation\n");
		uint8_t* mark_row_ids = new uint8_t[1];
		output_relation.columns[input_relation.column_count] = new GPUColumn(0, ColumnType::INT32, mark_row_ids);
	}

	if (join_type == JoinType::INNER || join_type == JoinType::OUTER || join_type == JoinType::RIGHT || join_type == JoinType::LEFT) {
		printf("Writing row IDs from RHS to output relation\n");
		// on the RHS, we need to fetch the data from the hash table
		uint64_t* right_row_ids = new uint64_t[1];
		for (idx_t i = 0; i < rhs_output_columns.size(); i++) {
			const auto output_col_idx = rhs_output_columns[i];
			printf("Passing column idx %ld from RHS (late materialized) to idx %ld in output relation\n", output_col_idx, input_relation.column_count + output_col_idx);
			output_relation.columns[input_relation.column_count + output_col_idx] = hash_table_result->columns[output_col_idx];
			// output_relation.columns[input_relation.column_count + output_col_idx]->row_ids = right_row_ids;
            if (row_ids_right) {
                if (hash_table_result->columns[output_col_idx]->row_ids == nullptr) {
                    output_relation.columns[input_relation.column_count + output_col_idx]->row_ids = row_ids_right;
                } else {
                    uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (hash_table_result->columns[output_col_idx]->row_ids);
                    uint64_t* new_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
                    materializeExpression<uint64_t>(row_ids_input, new_row_ids, row_ids_right, count[0]);
                    output_relation.columns[input_relation.column_count + output_col_idx]->row_ids = new_row_ids;
                }
            }
		}
	} else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		printf("Writing row IDs from RHS to output relation\n");
		// on the RHS, we need to fetch the data from the hash table
		uint64_t* right_row_ids = new uint64_t[1];
		for (idx_t i = 0; i < rhs_output_columns.size(); i++) {
			const auto output_col_idx = rhs_output_columns[i];
			printf("Passing column idx %ld from RHS (late materialized) to idx %ld in output relation\n", output_col_idx, output_col_idx);
			output_relation.columns[output_col_idx] = hash_table_result->columns[output_col_idx];
			output_relation.columns[output_col_idx]->row_ids = right_row_ids;
		}
	}

    return OperatorResultType::FINISHED;
};

//building hash table
// SinkResultType 
// GPUPhysicalHashJoin::Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input) const {
SinkResultType 
GPUPhysicalHashJoin::Sink(GPUIntermediateRelation &input_relation) const {
    //Read the key and payload from input relation
    //Check if late materialization is needed
    //Build the hash table

	// printf("input relation size %d\n", input_relation.columns.size());
	// for (auto col : input_relation.columns) {
	// 	printf("input relation column size %d\n", col->column_length);
	// }

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	uint64_t* build_key = nullptr;
	uint64_t size;
	if (conditions.size() > 1) throw NotImplementedException("Multiple conditions not supported yet");

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
        // printf("Reading join key for building hash table from index %ld\n", join_key_index);
        // input_relation.checkLateMaterialization(join_key_index);

		if (input_relation.checkLateMaterialization(join_key_index)) {
			uint64_t* temp = reinterpret_cast<uint64_t*> (input_relation.columns[join_key_index]->data_wrapper.data);
			uint64_t* row_ids_input = reinterpret_cast<uint64_t*> (input_relation.columns[join_key_index]->row_ids);
			build_key = gpuBufferManager->customCudaMalloc<uint64_t>(input_relation.columns[join_key_index]->row_id_count, 0, 0);
			materializeExpression<uint64_t>(temp, build_key, row_ids_input, input_relation.columns[join_key_index]->row_id_count);
			size = input_relation.columns[join_key_index]->row_id_count;
		} else {
			build_key = reinterpret_cast<uint64_t*> (input_relation.columns[join_key_index]->data_wrapper.data);
			size = input_relation.columns[join_key_index]->column_length;
		}

		gpu_hash_table = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(size * 2 * 2, 0, 0);
		ht_len = size * 2;
		buildHashTable<uint64_t>(build_key, gpu_hash_table, ht_len, size, 0);
	}

	int right_idx = 0;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		if (condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
			throw InvalidInputException("Unsupported join condition");
		}
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
		hash_table_result->columns[cond_idx] = input_relation.columns[join_key_index];
		right_idx++;
	}

	printf("Building hash table\n");
    for (idx_t i = 0; i < payload_column_idxs.size(); i++) {
        auto payload_idx = payload_column_idxs[i];
        // D_ASSERT(vector.GetType() == ht.layout.GetTypes()[output_col_idx]);
		printf("Passing column idx %d from input relation to index %ld in RHS hash table\n", payload_idx, right_idx + i);
        // hash_table_result->columns[rhs_col] = input_relation.columns[i];
		hash_table_result->columns[right_idx + i] = input_relation.columns[payload_idx];
    }

    return SinkResultType::FINISHED;
};


//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalHashJoin::BuildJoinPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline, GPUPhysicalOperator &op,
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
	auto &join_op = op.Cast<GPUPhysicalHashJoin>();
	if (join_op.IsSource()) {
		add_child_pipeline = true;
	}

	if (add_child_pipeline) {
		meta_pipeline.CreateChildPipeline(current, op, last_pipeline);
	}
}

void GPUPhysicalHashJoin::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	GPUPhysicalHashJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

} // namespace duckdb