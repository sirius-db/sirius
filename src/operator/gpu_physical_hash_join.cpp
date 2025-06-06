#include "operator/gpu_physical_hash_join.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/enums/physical_operator_type.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"
#include "log/logging.hpp"

namespace duckdb {

void 
ResolveTypeProbeExpression(vector<shared_ptr<GPUColumn>> &probe_keys, uint64_t* &count, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
		unsigned long long* ht, uint64_t ht_len, const vector<JoinCondition> &conditions, JoinType join_type,
		bool unique_build_keys, GPUBufferManager* gpuBufferManager) {
	int num_keys = conditions.size();
	uint8_t** probe_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);

	for (int key = 0; key < num_keys; key++) {
		probe_data[key] = probe_keys[key]->data_wrapper.data;
	}
	size_t size = probe_keys[0]->column_length;

	int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
	for (int key = 0; key < num_keys; key++) {
		if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL || conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			condition_mode[key] = 0;
		} else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL || conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
			condition_mode[key] = 1;
		} else {
			throw NotImplementedException("Unsupported comparison type");
		}
	}

	//TODO: Need to handle special case for unique keys for better performance
	if (join_type == JoinType::INNER) {
		if (unique_build_keys) {
			probeHashTableSingleMatch(probe_data, ht, ht_len, row_ids_left, row_ids_right, count, size, condition_mode, num_keys, 0);
		} else {
			probeHashTable(probe_data, ht, ht_len, row_ids_left, row_ids_right, count, size, condition_mode, num_keys, false);
		}
	} else if (join_type == JoinType::SEMI) {
		probeHashTableSingleMatch(probe_data, ht, ht_len, row_ids_left, row_ids_right, count, size, condition_mode, num_keys, 1);
	} else if (join_type == JoinType::ANTI) {
		probeHashTableSingleMatch(probe_data, ht, ht_len, row_ids_left, row_ids_right, count, size, condition_mode, num_keys, 2);
	} else if (join_type == JoinType::RIGHT) {
		if (unique_build_keys) {
			probeHashTableSingleMatch(probe_data, ht, ht_len, row_ids_left, row_ids_right, count, size, condition_mode, num_keys, 3);
		} else {
			probeHashTable(probe_data, ht, ht_len, row_ids_left, row_ids_right, count, size, condition_mode, num_keys, true);
		}
	} else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		if (unique_build_keys) {
			probeHashTableRightSemiAntiSingleMatch(probe_data, ht, ht_len, size, condition_mode, num_keys);
		} else {
			probeHashTableRightSemiAnti(probe_data, ht, ht_len, size, condition_mode, num_keys);
		}
	} else {
		throw NotImplementedException("Unsupported join type");
	}
}

void
HandleProbeExpression(vector<shared_ptr<GPUColumn>> &probe_keys, uint64_t* &count, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
		unsigned long long* ht, uint64_t ht_len, const vector<JoinCondition> &conditions, JoinType join_type, 
		bool unique_build_keys, GPUBufferManager* gpuBufferManager) {
    switch(probe_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
		ResolveTypeProbeExpression(probe_keys, count, row_ids_left, row_ids_right, ht, ht_len, conditions, join_type, unique_build_keys, gpuBufferManager);
		break;
      case ColumnType::FLOAT64:
	  	ResolveTypeProbeExpression(probe_keys, count, row_ids_left, row_ids_right, ht, ht_len, conditions, join_type, unique_build_keys, gpuBufferManager);
		break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleProbeExpression`: %d",
																			static_cast<int>(probe_keys[0]->data_wrapper.type));
    }
}

void
ResolveTypeMarkExpression(vector<shared_ptr<GPUColumn>> &probe_keys, uint8_t* &output, 
		unsigned long long* ht, uint64_t ht_len, const vector<JoinCondition> &conditions, GPUBufferManager* gpuBufferManager) {

	int num_keys = conditions.size();
	uint8_t** probe_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);

	for (int key = 0; key < num_keys; key++) {
		probe_data[key] = probe_keys[key]->data_wrapper.data;
	}
	size_t size = probe_keys[0]->column_length;

	int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
	for (int key = 0; key < num_keys; key++) {
		if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL || conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			condition_mode[key] = 0;
		} else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL || conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
			// TODO: Currently only support TPC-H Q21: l2.l_orderkey = l1.l_orderkey and l2.l_suppkey != l1.l_suppkey
			if (key != 1 || num_keys != 2) throw NotImplementedException("Unsupported comparison type");
			condition_mode[key] = 1;
		} else {
			throw NotImplementedException("Unsupported comparison type");
		}
	}

	probeHashTableMark(probe_data, ht, ht_len, output, size, condition_mode, num_keys);
}

void
HandleMarkExpression(vector<shared_ptr<GPUColumn>> &probe_keys, uint8_t* &output, 
		unsigned long long* ht, uint64_t ht_len, const vector<JoinCondition> &conditions, GPUBufferManager* gpuBufferManager) {
    switch(probe_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
		ResolveTypeMarkExpression(probe_keys, output, ht, ht_len, conditions, gpuBufferManager);
		break;
      case ColumnType::FLOAT64:
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleMarkExpression`: %d",
																			static_cast<int>(probe_keys[0]->data_wrapper.type));
    }
}

void 
ResolveTypeBuildExpression(vector<shared_ptr<GPUColumn>> &build_keys, unsigned long long* ht, uint64_t ht_len, 
	const vector<JoinCondition> &conditions, JoinType join_type, GPUBufferManager* gpuBufferManager) {
	int num_keys = conditions.size();
	uint8_t** build_data = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);

	for (int key = 0; key < num_keys; key++) {
		build_data[key] = build_keys[key]->data_wrapper.data;
	}
	size_t size = build_keys[0]->column_length;

	int* condition_mode = gpuBufferManager->customCudaHostAlloc<int>(num_keys);
	for (int key = 0; key < num_keys; key++) {
		if (conditions[key].comparison == ExpressionType::COMPARE_EQUAL || conditions[key].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			condition_mode[key] = 0;
		} else if (conditions[key].comparison == ExpressionType::COMPARE_NOTEQUAL || conditions[key].comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
			// TODO: Currently only support TPC-H Q21: l2.l_orderkey = l1.l_orderkey and l2.l_suppkey != l1.l_suppkey
			if (key != 1 || num_keys != 2) throw NotImplementedException("Unsupported comparison type");
			condition_mode[key] = 1;
		} else {
			throw NotImplementedException("Unsupported comparison type");
		}
	}

	if (join_type == JoinType::INNER || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
		buildHashTable(build_data, ht, ht_len, size, condition_mode, num_keys, 0);
	} else if (join_type == JoinType::RIGHT || join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		buildHashTable(build_data, ht, ht_len, size, condition_mode, num_keys, 1);
	} else {
		throw NotImplementedException("Unsupported join type");
	}
}

void
HandleBuildExpression(vector<shared_ptr<GPUColumn>> &build_keys, unsigned long long* ht, uint64_t ht_len, 
	const vector<JoinCondition> &conditions, JoinType join_type, GPUBufferManager* gpuBufferManager) {
    switch(build_keys[0]->data_wrapper.type) {
      case ColumnType::INT64:
		ResolveTypeBuildExpression(build_keys, ht, ht_len, conditions, join_type, gpuBufferManager);
		break;
      case ColumnType::FLOAT64:
	  	ResolveTypeBuildExpression(build_keys, ht, ht_len, conditions, join_type, gpuBufferManager);
		break;
      default:
        throw NotImplementedException("Unsupported sirius column type in `HandleBuildExpression`: %d",
																			static_cast<int>(build_keys[0]->data_wrapper.type));
    }
}

void
HandleScanHTExpression(unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids, uint64_t* &count, JoinType join_type, const vector<JoinCondition> &conditions) {
	int num_keys = conditions.size();
	if (join_type == JoinType::RIGHT_SEMI) {
		scanHashTableRight(ht, ht_len, row_ids, count, 0, num_keys);
	} else if (join_type == JoinType::RIGHT || join_type == JoinType::RIGHT_ANTI) {
		scanHashTableRight(ht, ht_len, row_ids, count, 1, num_keys);
	} else {
		throw NotImplementedException("Unsupported join type");
	}
}

void ReorderConditions(vector<JoinCondition> &conditions) {
	// we reorder conditions so the ones with COMPARE_EQUAL occur first
	// check if this is already the case
	bool is_ordered = true;
	bool seen_non_equal = false;
	for (auto &cond : conditions) {
		if (cond.comparison == ExpressionType::COMPARE_EQUAL ||
		    cond.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			if (seen_non_equal) {
				is_ordered = false;
				break;
			}
		} else {
			seen_non_equal = true;
		}
	}
	if (is_ordered) {
		// no need to re-order
		return;
	}
	// gather lists of equal/other conditions
	vector<JoinCondition> equal_conditions;
	vector<JoinCondition> other_conditions;
	for (auto &cond : conditions) {
		if (cond.comparison == ExpressionType::COMPARE_EQUAL ||
		    cond.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			equal_conditions.push_back(std::move(cond));
		} else {
			other_conditions.push_back(std::move(cond));
		}
	}
	conditions.clear();
	// reconstruct the sorted conditions
	for (auto &cond : equal_conditions) {
		conditions.push_back(std::move(cond));
	}
	for (auto &cond : other_conditions) {
		conditions.push_back(std::move(cond));
	}
}

GPUPhysicalHashJoin::GPUPhysicalHashJoin(LogicalOperator &op, unique_ptr<GPUPhysicalOperator> left,
                                   unique_ptr<GPUPhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   const vector<idx_t> &left_projection_map, const vector<idx_t> &right_projection_map,
                                   vector<LogicalType> delim_types, idx_t estimated_cardinality,
                                   unique_ptr<JoinFilterPushdownInfo> pushdown_info_p)
    : GPUPhysicalOperator(PhysicalOperatorType::HASH_JOIN, op.types, estimated_cardinality), join_type(join_type), conditions(std::move(cond)) {

	ReorderConditions(conditions);

	filter_pushdown = std::move(pushdown_info_p);

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

	auto &lhs_input_types = children[0]->GetTypes();

	// Create a projection map for the LHS (if it was empty), for convenience
	lhs_output_columns.col_idxs = left_projection_map;
	if (lhs_output_columns.col_idxs.empty()) {
		lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
		for (idx_t i = 0; i < lhs_input_types.size(); i++) {
			lhs_output_columns.col_idxs.emplace_back(i);
		}
	}

	for (auto &lhs_col : lhs_output_columns.col_idxs) {
		auto &lhs_col_type = lhs_input_types[lhs_col];
		lhs_output_columns.col_types.push_back(lhs_col_type);
	}

	// For ANTI, SEMI and MARK join, we only need to store the keys, so for these the payload/RHS types are empty
	if (join_type == JoinType::ANTI || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
		materialized_build_key = make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
		hash_table_result = make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
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
			payload_columns.col_idxs.push_back(rhs_col);
			payload_columns.col_types.push_back(rhs_col_type);
			rhs_output_columns.col_idxs.push_back(condition_types.size() + payload_columns.col_types.size() - 1);
		} else {
			// This rhs column is a join key
			rhs_output_columns.col_idxs.push_back(it->second);
		}
		rhs_output_columns.col_types.push_back(rhs_col_type);
	}

	hash_table_result = make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size() + payload_columns.col_idxs.size());
	materialized_build_key = make_shared_ptr<GPUIntermediateRelation>(build_columns_in_conditions.size());
};

SourceResultType
GPUPhysicalHashJoin::GetData(GPUIntermediateRelation &output_relation) const {
	auto start = std::chrono::high_resolution_clock::now();

	idx_t left_column_count = output_relation.columns.size() - rhs_output_columns.col_idxs.size();
	if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		SIRIUS_LOG_DEBUG("Right semi or right anti join so there will be no columns from LHS");
		left_column_count = 0;
	} else if (join_type == JoinType::RIGHT) {
		for (idx_t col = 0; col < left_column_count; col++) {
			//pretend this to be NUll column from the left table (it should be NULL for the RIGHT join)
			SIRIUS_LOG_DEBUG("Right join so columns from LHS will be null");
			output_relation.columns[col] = make_shared_ptr<GPUColumn>(0, ColumnType::INT64, nullptr);
		}
	} else {
		throw InvalidInputException("Get data not supported for this join type");
	}

	uint64_t* row_ids = nullptr;
	uint64_t* count = nullptr;
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	HandleScanHTExpression(gpu_hash_table, ht_len, row_ids, count, join_type, conditions);

	for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
		const auto rhs_col = rhs_output_columns.col_idxs[i];
		SIRIUS_LOG_DEBUG("Writing hash_table column {} to column {}", rhs_col, i);
	}
	//TODO: Check if we need to maintain unique for the RHS columns
	if (unique_probe_keys) {
		HandleMaterializeRowIDsRHS(*hash_table_result, output_relation, rhs_output_columns.col_idxs, left_column_count, count[0], row_ids, gpuBufferManager, true);
	} else {
		HandleMaterializeRowIDsRHS(*hash_table_result, output_relation, rhs_output_columns.col_idxs, left_column_count, count[0], row_ids, gpuBufferManager, false);
	}
	// for (idx_t i = 0; i < hash_table_result->columns.size(); i++) {
	// 	if (find(rhs_output_columns.col_idxs.begin(), rhs_output_columns.col_idxs.end(), i) == rhs_output_columns.col_idxs.end()) {
	// 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.data), 0);
	// 		if (hash_table_result->columns[i]->data_wrapper.type == ColumnType::VARCHAR) {
	// 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.offset), 0);
	// 		}
	// 	}
	// }

	//check if all output columns has the same size
	// for (idx_t i = 1; i < output_relation.columns.size(); i++) {
	// 	if (output_relation.columns[i]->column_length != output_relation.columns[0]->column_length) {
	// 		printf("Column %d has length %zu, while column 0 has length %zu\n", i, output_relation.columns[i]->column_length, output_relation.columns[0]->column_length);
	// 		throw InvalidInputException("Output columns have different sizes");
	// 	}
	// }

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Hash Join GetData time: {:.2f} ms", duration.count()/1000.0);

	return SourceResultType::FINISHED;
}

OperatorResultType
GPUPhysicalHashJoin::Execute(GPUIntermediateRelation &input_relation, GPUIntermediateRelation &output_relation) const {

	auto start = std::chrono::high_resolution_clock::now();

	if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		// for RIGHT SEMI and RIGHT ANTI joins, the output is the RHS
		// we only need to output the RHS columns
		// the LHS columns are NULL
		if (output_relation.columns.size() != rhs_output_columns.col_idxs.size()) {
			throw InvalidInputException("Wrong input size");
		}
	} else if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		// for SEMI and ANTI join, the output is the LHS
		// we only need to output the LHS columns
		// the RHS columns are NULL
		if (output_relation.columns.size() != lhs_output_columns.col_idxs.size()) {
			throw InvalidInputException("Wrong input size");
		}
	} else if (join_type == JoinType::RIGHT || join_type == JoinType::LEFT || join_type == JoinType::INNER || join_type == JoinType::OUTER) {
		// for INNER and OUTER join, we output all columns
		if (output_relation.columns.size() != lhs_output_columns.col_idxs.size() + rhs_output_columns.col_idxs.size()) {
			throw InvalidInputException("Wrong input size");
		}
	} else if (join_type == JoinType::MARK) {
		// for MARK join, we output all columns from the LHS and one extra boolean column
		if (output_relation.columns.size() != lhs_output_columns.col_idxs.size() + 1) {
			throw InvalidInputException("Wrong input size");
		}
	} else {
		throw InvalidInputException("Unsupported join type");
	}

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	vector<shared_ptr<GPUColumn>> probe_key(conditions.size());
	for (int i = 0; i < conditions.size(); i++) {
		probe_key[i] = nullptr;
	}
	uint64_t* count;
	uint64_t* row_ids_left = nullptr;
	uint64_t* row_ids_right = nullptr;
	uint8_t* output = nullptr; // for MARK JOIN
	// if (conditions.size() > 1) throw NotImplementedException("Multiple conditions not supported yet");

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.left->Cast<BoundReferenceExpression>().index;
		if (input_relation.columns[join_key_index]->is_unique) {
			unique_probe_keys = true;
		}
		SIRIUS_LOG_DEBUG("Materializing join key for probing hash table from index {}", join_key_index);
		probe_key[cond_idx] = HandleMaterializeExpression(input_relation.columns[join_key_index], condition.left->Cast<BoundReferenceExpression>(), gpuBufferManager);
	}

	//probing hash table
	SIRIUS_LOG_DEBUG("Probing hash table");
	if (join_type == JoinType::INNER) {
		// check if there is a non-equality condition
		vector<shared_ptr<GPUColumn>> build_key(conditions.size());
		for (int cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
			build_key[cond_idx] = materialized_build_key->columns[cond_idx];
		}
		for (int col = 0; col < conditions.size(); col++) {
			// if types is VARCHAR, check the number of bytes
			if (build_key[col]->data_wrapper.type == ColumnType::VARCHAR) {
				if (build_key[col]->data_wrapper.num_bytes > INT32_MAX) {
					throw NotImplementedException("String column size greater than INT32_MAX is not supported");
				}
			}
			if (probe_key[col]->data_wrapper.type == ColumnType::VARCHAR) {
				if (probe_key[col]->data_wrapper.num_bytes > INT32_MAX) {
					throw NotImplementedException("String column size greater than INT32_MAX is not supported");
				}
			}
		}
		if (build_key[0]->column_length > INT32_MAX || probe_key[0]->column_length > INT32_MAX) {
			HandleBuildExpression(build_key, gpu_hash_table, ht_len, conditions, join_type, gpuBufferManager);
			HandleProbeExpression(probe_key, count, row_ids_left, row_ids_right, gpu_hash_table, ht_len, conditions, join_type, unique_build_keys, gpuBufferManager);
		} else {
			bool has_non_equality_condition = false;
			for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
				if (conditions[cond_idx].comparison != ExpressionType::COMPARE_EQUAL && conditions[cond_idx].comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
					has_non_equality_condition = true;
					break;
				}
			}
			if (!has_non_equality_condition) {
				cudf_inner_join(probe_key, build_key, conditions.size(), row_ids_left, row_ids_right, count);
			} else {
				cudf_mixed_join(probe_key, build_key, conditions, join_type, row_ids_left, row_ids_right, count);
			}
		}
	} else if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::OUTER || join_type == JoinType::RIGHT) {
		HandleProbeExpression(probe_key, count, row_ids_left, row_ids_right, gpu_hash_table, ht_len, conditions, join_type, unique_build_keys, gpuBufferManager);
		// if (count[0] == 0) throw NotImplementedException("No match found");
	} else if (join_type == JoinType::MARK) {
		SIRIUS_LOG_DEBUG("Writing boolean column to output relation");
		HandleMarkExpression(probe_key, output, gpu_hash_table, ht_len, conditions, gpuBufferManager);
	} else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		HandleProbeExpression(probe_key, count, row_ids_left, row_ids_right, gpu_hash_table, ht_len, conditions, join_type, unique_build_keys, gpuBufferManager);
	} else {
		throw NotImplementedException("Unsupported join type");
	}

	//materialize columns from the left table
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || join_type == JoinType::INNER || join_type == JoinType::OUTER || join_type == JoinType::RIGHT) {
		SIRIUS_LOG_DEBUG("Writing LHS columns to output relation");

		if (join_type == JoinType::SEMI || join_type == JoinType::ANTI || unique_build_keys) {
			HandleMaterializeRowIDsLHS(input_relation, output_relation, lhs_output_columns.col_idxs, count[0], row_ids_left, gpuBufferManager, true);
		} else {
			HandleMaterializeRowIDsLHS(input_relation, output_relation, lhs_output_columns.col_idxs, count[0], row_ids_left, gpuBufferManager, false);
		}
		// free all the columns in the input relation that are not in the lhs_output_columns
		// for (idx_t i = 0; i < input_relation.columns.size(); i++) {
		// 	if (find(lhs_output_columns.col_idxs.begin(), lhs_output_columns.col_idxs.end(), i) == lhs_output_columns.col_idxs.end()) {
		// 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.data), 0);
		// 		if (input_relation.columns[i]->data_wrapper.type == ColumnType::VARCHAR) {
		// 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.offset), 0);
		// 		}
		// 	}
		// }
	} else if (join_type == JoinType::MARK) {
		SIRIUS_LOG_DEBUG("Writing LHS columns to output relation");
		for (idx_t i = 0; i < lhs_output_columns.col_idxs.size(); i++) {
			auto lhs_col = lhs_output_columns.col_idxs[i];
			SIRIUS_LOG_DEBUG("Passing column idx {} from LHS to idx {} in output relation", lhs_col, i);
			output_relation.columns[i] = make_shared_ptr<GPUColumn>(input_relation.columns[lhs_col]->column_length, input_relation.columns[lhs_col]->data_wrapper.type, input_relation.columns[lhs_col]->data_wrapper.data);
			output_relation.columns[i]->row_ids = input_relation.columns[lhs_col]->row_ids;
			output_relation.columns[i]->row_id_count = input_relation.columns[lhs_col]->row_id_count;
			if (unique_build_keys) {
				output_relation.columns[i]->is_unique = input_relation.columns[lhs_col]->is_unique;
			} else {
				output_relation.columns[i]->is_unique = false;
			}
		}
		output_relation.columns[lhs_output_columns.col_idxs.size()] = make_shared_ptr<GPUColumn>(probe_key[0]->column_length, ColumnType::BOOLEAN, output);
		output_relation.columns[lhs_output_columns.col_idxs.size()]->row_ids = probe_key[0]->row_ids;
		output_relation.columns[lhs_output_columns.col_idxs.size()]->row_id_count = probe_key[0]->row_id_count;
		// free all the columns in the input relation that are not in the lhs_output_columns
		// for (idx_t i = 0; i < input_relation.columns.size(); i++) {
		// 	if (find(lhs_output_columns.col_idxs.begin(), lhs_output_columns.col_idxs.end(), i) == lhs_output_columns.col_idxs.end()) {
		// 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.data), 0);
		// 		if (input_relation.columns[i]->data_wrapper.type == ColumnType::VARCHAR) {
		// 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(input_relation.columns[i]->data_wrapper.offset), 0);
		// 		}
		// 	}
		// }
	} else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		// WE SHOULD NOT NEED TO DO ANYTHING HERE
	} else {
		throw NotImplementedException("Unsupported join type");
	}

	//materialize columns from the right tables
	if (join_type == JoinType::INNER || join_type == JoinType::OUTER || join_type == JoinType::LEFT || join_type == JoinType::RIGHT) {
		SIRIUS_LOG_DEBUG("Writing row IDs from RHS to output relation");
		for (int col = 0; col < hash_table_result->columns.size(); col++) {
			gpuBufferManager->lockAllocation(reinterpret_cast<uint8_t*>(hash_table_result->columns[col]->data_wrapper.data), 0);
			gpuBufferManager->lockAllocation(reinterpret_cast<uint8_t*>(hash_table_result->columns[col]->row_ids), 0);
			// If the column type is VARCHAR, also lock the offset allocation
			if (hash_table_result->columns[col]->data_wrapper.type == ColumnType::VARCHAR) {
				gpuBufferManager->lockAllocation(reinterpret_cast<uint8_t*>(hash_table_result->columns[col]->data_wrapper.offset), 0);
			}
		}
		if (unique_probe_keys) {
			HandleMaterializeRowIDsRHS(*hash_table_result, output_relation, rhs_output_columns.col_idxs, lhs_output_columns.col_idxs.size(), count[0], row_ids_right, gpuBufferManager, true);
		} else {
			HandleMaterializeRowIDsRHS(*hash_table_result, output_relation, rhs_output_columns.col_idxs, lhs_output_columns.col_idxs.size(), count[0], row_ids_right, gpuBufferManager, false);
		}
		if (join_type == JoinType::INNER || join_type == JoinType::LEFT) {
			// free all the columns in the hash table result that are not in the rhs_output_columns
			// for (idx_t i = 0; i < hash_table_result->columns.size(); i++) {
			// 	if (find(rhs_output_columns.col_idxs.begin(), rhs_output_columns.col_idxs.end(), i) == rhs_output_columns.col_idxs.end()) {
			// 		gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.data), 0);
			// 		if (hash_table_result->columns[i]->data_wrapper.type == ColumnType::VARCHAR) {
            // 			gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(hash_table_result->columns[i]->data_wrapper.offset), 0);
        	// 		}
			// 	}
			// }
		}
	} else if (join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		SIRIUS_LOG_DEBUG("Writing row IDs from RHS to output relation");
		// on the RHS, we need to fetch the data from the hash table
		for (idx_t i = 0; i < rhs_output_columns.col_idxs.size(); i++) {
			const auto rhs_col = rhs_output_columns.col_idxs[i];
			SIRIUS_LOG_DEBUG("Passing column idx {} from RHS (late materialized) to idx {} in output relation", rhs_col, i);
			output_relation.columns[i] = make_shared_ptr<GPUColumn>(0, hash_table_result->columns[rhs_col]->data_wrapper.type, nullptr);
		}
	}

	if (join_type == JoinType::INNER || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
		if (gpu_hash_table != nullptr) gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(gpu_hash_table), 0);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Hash Join Execute time: {:.2f} ms", duration.count()/1000.0);

    return OperatorResultType::FINISHED;
};

SinkResultType 
GPUPhysicalHashJoin::Sink(GPUIntermediateRelation &input_relation) const {
	
	auto start = std::chrono::high_resolution_clock::now();

	if (!delim_types.empty() && join_type == JoinType::MARK) {
		// correlated MARK join
		throw NotImplementedException("Correlated MARK join not supported yet");
	}

	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	vector<shared_ptr<GPUColumn>> build_keys(conditions.size());
	for (idx_t i = 0; i < conditions.size(); i++) {
		build_keys[i] = nullptr;
	}

	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
		if (input_relation.columns[join_key_index]->is_unique) {
			unique_build_keys = true;
		}
		SIRIUS_LOG_DEBUG("Materializing join key for building hash table from index {}", join_key_index);
		build_keys[cond_idx] = HandleMaterializeExpression(input_relation.columns[join_key_index], condition.right->Cast<BoundReferenceExpression>(), gpuBufferManager);
	}

	SIRIUS_LOG_DEBUG("Building hash table");
	ht_len = build_keys[0]->column_length * 2;
	if (join_type == JoinType::INNER || join_type == JoinType::SEMI || join_type == JoinType::MARK) {
		if (ht_len == 0) gpu_hash_table = nullptr;
		else gpu_hash_table = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(ht_len * (conditions.size() + 1), 0, 0);
	} else if (join_type == JoinType::RIGHT || join_type == JoinType::RIGHT_SEMI || join_type == JoinType::RIGHT_ANTI) {
		if (ht_len == 0) gpu_hash_table = nullptr;
		else gpu_hash_table = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(ht_len * (conditions.size() + 2), 0, 0);
	}
	

	if (join_type == JoinType::INNER) {
		// check if there is a non-equality condition
		// bool has_non_equality_condition = false;
		// for (idx_t i = 0; i < conditions.size(); i++) {
		// 	if (conditions[i].comparison != ExpressionType::COMPARE_EQUAL && conditions[i].comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
		// 		has_non_equality_condition = true;
		// 		break;
		// 	}
		// }
		// if (!has_non_equality_condition) {
		// 	cudf_build(build_keys, cudf_hash_table, conditions.size());
		// }
	} else {
		HandleBuildExpression(build_keys, gpu_hash_table, ht_len, conditions, join_type, gpuBufferManager);
	}

	int right_idx = 0;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		if (condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
			throw InvalidInputException("Unsupported join condition");
		}
        auto join_key_index = condition.right->Cast<BoundReferenceExpression>().index;
		SIRIUS_LOG_DEBUG("Passing column idx {} from input relation to index {} in RHS hash table", join_key_index, cond_idx);
		hash_table_result->columns[cond_idx] = input_relation.columns[join_key_index];
		materialized_build_key->columns[cond_idx] = build_keys[cond_idx];
		right_idx++;
	}

	
    for (idx_t i = 0; i < payload_columns.col_idxs.size(); i++) {
        auto payload_idx = payload_columns.col_idxs[i];
		SIRIUS_LOG_DEBUG("Passing column idx {} from input relation to index {} in RHS hash table", payload_idx, right_idx + i);
		hash_table_result->columns[right_idx + i] = input_relation.columns[payload_idx];
    }

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	SIRIUS_LOG_DEBUG("Hash Join Sink time: {:.2f} ms", duration.count()/1000.0);

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