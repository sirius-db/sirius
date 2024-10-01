#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace duckdb {

static vector<LogicalType> CreateGroupChunkTypes(vector<unique_ptr<Expression>> &groups) {
	set<idx_t> group_indices;

	if (groups.empty()) {
		return {};
	}

	for (auto &group : groups) {
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		group_indices.insert(bound_ref.index);
	}
	idx_t highest_index = *group_indices.rbegin();
	vector<LogicalType> types(highest_index + 1, LogicalType::SQLNULL);
	for (auto &group : groups) {
		auto &bound_ref = group->Cast<BoundReferenceExpression>();
		types[bound_ref.index] = bound_ref.return_type;
	}
	return types;
}

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality)
    : GPUPhysicalGroupedAggregate(context, std::move(types), std::move(expressions), {}, estimated_cardinality) {


}

GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p, idx_t estimated_cardinality)
    : GPUPhysicalGroupedAggregate(context, std::move(types), std::move(expressions), std::move(groups_p), {}, {},
                            estimated_cardinality) {
}

//expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression to a column
//groups_p is the list of group by columns. Each group by column is a bound_ref expression to a column
//grouping_sets_p is the list of grouping set. Each grouping set is a set of indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets and for every grouping set there is one radix_table
//grouping_functions_p is a list of indexes to the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the grouping set and the second level is the indexes to the groupby expression for that set.
GPUPhysicalGroupedAggregate::GPUPhysicalGroupedAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p,
                                             vector<GroupingSet> grouping_sets_p,
                                             vector<unsafe_vector<idx_t>> grouping_functions_p,
                                             idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
      grouping_sets(std::move(grouping_sets_p)) {

	// get a list of all aggregates to be computed
	const idx_t group_count = groups_p.size();
	if (grouping_sets.empty()) {
		GroupingSet set;
		for (idx_t i = 0; i < group_count; i++) {
			set.insert(i);
		}
		grouping_sets.push_back(std::move(set));
	}
	input_group_types = CreateGroupChunkTypes(groups_p);

	grouped_aggregate_data.InitializeGroupby(std::move(groups_p), std::move(expressions),
	                                         std::move(grouping_functions_p));

	auto &aggregates = grouped_aggregate_data.aggregates;
	// filter_indexes must be pre-built, not lazily instantiated in parallel...
	// Because everything that lives in this class should be read-only at execution time
	idx_t aggregate_input_idx = 0;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		aggregate_input_idx += aggr.children.size();
		if (aggr.aggr_type == AggregateType::DISTINCT) {
			distinct_filter.push_back(i);
		} else if (aggr.aggr_type == AggregateType::NON_DISTINCT) {
			non_distinct_filter.push_back(i);
		} else { // LCOV_EXCL_START
			throw NotImplementedException("AggregateType not implemented in PhysicalHashAggregate");
		} // LCOV_EXCL_STOP
	}

	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			auto &bound_ref_expr = aggr.filter->Cast<BoundReferenceExpression>();
			if (!filter_indexes.count(aggr.filter.get())) {
				// Replace the bound reference expression's index with the corresponding index of the payload chunk
       			//TODO: Still not quite sure why duckdb replace the index
				filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
				bound_ref_expr.index = aggregate_input_idx;
			}
			aggregate_input_idx++;
		}
	}

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

	for (idx_t i = 0; i < grouping_sets.size(); i++) {
		groupings.emplace_back(grouping_sets[i], grouped_aggregate_data, distinct_collection_info);
	}

  //The output of groupby is ordered as the grouping columns first followed by the aggregate columns
  //See RadixHTLocalSourceState::Scan for more details
  idx_t total_output_columns = 0;
  for (auto &aggregate : aggregates) {
	auto &aggr = aggregate->Cast<BoundAggregateExpression>();
    total_output_columns++;
  }
  total_output_columns += grouped_aggregate_data.GroupCount();
  group_by_result = new GPUIntermediateRelation(0, total_output_columns);
}

// SinkResultType
// GPUPhysicalGroupedAggregate::Sink(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const {

SinkResultType
GPUPhysicalGroupedAggregate::Sink(GPUIntermediateRelation& input_relation) const {
  	printf("Perform groupby and aggregation\n");
	
	if (distinct_collection_info) {
		SinkDistinct(input_relation);
	}

	// DataChunk &aggregate_input_chunk = local_state.aggregate_input_chunk;
	auto &aggregates = grouped_aggregate_data.aggregates;
	idx_t aggregate_input_idx = 0;

	if (groupings.size() > 1) throw NotImplementedException("Multiple groupings not supported yet");

	// Reading groupby columns based on the grouping set
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		int idx = 0;
		for (auto &group_idx : grouping_sets[i]) {
			// Retrieve the expression containing the index in the input chunk
			auto &group = grouped_aggregate_data.groups[group_idx];
			D_ASSERT(group->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = group->Cast<BoundReferenceExpression>();
			printf("Reading groupby columns from index %d and passing it to index %d in groupby result\n", bound_ref_expr.index, bound_ref_expr.index);
			input_relation.checkLateMaterialization(bound_ref_expr.index);
			// printf("group by result size %d %d\n", group_by_result->columns.size(), grouped_aggregate_data.GroupCount());
			group_by_result->columns[idx] = input_relation.columns[bound_ref_expr.index];
			idx++;
		}
	}
	int aggr_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = child_expr->Cast<BoundReferenceExpression>();
			printf("Reading aggregation column from index %d and passing it to index %d in groupby result\n", bound_ref_expr.index, grouped_aggregate_data.groups.size() + aggr_idx);
			input_relation.checkLateMaterialization(bound_ref_expr.index);
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = input_relation.columns[bound_ref_expr.index];
		}
		if (aggr.children.size() == 0) {
			printf("Passing * aggregate to index %d in groupby result\n", grouped_aggregate_data.groups.size() + aggr_idx);
			group_by_result->columns[grouped_aggregate_data.groups.size() + aggr_idx] = new GPUColumn(0, ColumnType::INT64, nullptr);
		}
		aggr_idx++;
	}
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.filter) {
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			printf("Reading aggregation filter from index %d\n", it->second);
			input_relation.checkLateMaterialization(it->second);
		}
	}

  return SinkResultType::FINISHED;
}

// SourceResultType
// GPUPhysicalGroupedAggregate::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation, OperatorSourceInput &input) const {
SourceResultType
GPUPhysicalGroupedAggregate::GetData(GPUIntermediateRelation &output_relation) const {
//   printf("group by result size %d\n", group_by_result->columns.size());
  for (int col = 0; col < group_by_result->columns.size(); col++) {
    printf("Writing group by result to column %d\n", col);
    output_relation.columns[col] = group_by_result->columns[col];
  }

  return SourceResultType::FINISHED;
}

// void 
// GPUPhysicalGroupedAggregate::SinkDistinct(ExecutionContext &context, GPUIntermediateRelation& input_relation, OperatorSinkInput &input) const {
void
GPUPhysicalGroupedAggregate::SinkDistinct(GPUIntermediateRelation& input_relation) const {
	for (idx_t i = 0; i < groupings.size(); i++) {
		SinkDistinctGrouping(input_relation, i);
	}
}

// void
// GPUPhysicalGroupedAggregate:: SinkDistinctGrouping(ExecutionContext &context, GPUIntermediateRelation &input_relation, OperatorSinkInput &input,
	                        //   idx_t grouping_idx) const {
void
GPUPhysicalGroupedAggregate::SinkDistinctGrouping(GPUIntermediateRelation& input_relation, idx_t grouping_idx) const {
	auto &distinct_info = *distinct_collection_info;

	for (idx_t &idx : distinct_info.indices) {
		auto &aggregate = grouped_aggregate_data.aggregates[idx]->Cast<BoundAggregateExpression>();

		D_ASSERT(distinct_info.table_map.count(idx));

		if (aggregate.filter) {
			auto it = filter_indexes.find(aggregate.filter.get());
      		printf("Reading filter columns from index %d\n", it->second);

			for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
				auto &group = grouped_aggregate_data.groups[group_idx];
				auto &bound_ref = group->Cast<BoundReferenceExpression>();
				printf("Reading groupby columns from index %d and passing it to index %d in groupby result\n", bound_ref.index, group_idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[group_idx] = input_relation.columns[bound_ref.index];
			}
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				printf("Reading aggregation column from index %d and passing it to index %d in groupby result\n", bound_ref.index, grouped_aggregate_data.groups.size() + idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[grouped_aggregate_data.groups.size() + idx] = input_relation.columns[bound_ref.index];
			}
		} else {
			for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
				auto &group = grouped_aggregate_data.groups[group_idx];
				auto &bound_ref = group->Cast<BoundReferenceExpression>();
				printf("Reading groupby columns from index %d and passing it to index %d in groupby result\n", bound_ref.index, group_idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[group_idx] = input_relation.columns[bound_ref.index];
			}
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = child->Cast<BoundReferenceExpression>();
				printf("Reading aggregation column from index %d and passing it to index %d in groupby result\n", bound_ref.index, grouped_aggregate_data.groups.size() + idx);
				input_relation.checkLateMaterialization(bound_ref.index);
				group_by_result->columns[grouped_aggregate_data.groups.size() + idx] = input_relation.columns[bound_ref.index];
			}
		}
	}

}

} // namespace duckdb