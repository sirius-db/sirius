// #include "duckdb/execution/operator/projection/physical_projection.hpp"
// #include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
// #include "duckdb/execution/operator/scan/physical_table_scan.hpp"
// #include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include "gpu_physical_table_scan.hpp"
#include "gpu_physical_projection.hpp"
#include "gpu_physical_plan_generator.hpp"
// #include "duckdb/common/types.hpp"

namespace duckdb {

unique_ptr<TableFilterSet> GPUCreateTableFilterSet(TableFilterSet &table_filters, vector<column_t> &column_ids) {
	// create the table filter map
	auto table_filter_set = make_uniq<TableFilterSet>();
	for (auto &table_filter : table_filters.filters) {
		// find the relative column index from the absolute column index into the table
		idx_t column_index = DConstants::INVALID_INDEX;
		for (idx_t i = 0; i < column_ids.size(); i++) {
			if (table_filter.first == column_ids[i]) {
				column_index = i;
				break;
			}
		}
		if (column_index == DConstants::INVALID_INDEX) {
			throw InternalException("Could not find column index for table filter");
		}
		table_filter_set->filters[column_index] = std::move(table_filter.second);
	}
	return table_filter_set;
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalGet &op) {
	if (!op.children.empty()) {
		// this is for table producing functions that consume subquery results
		// D_ASSERT(op.children.size() == 1);
		// auto node = make_uniq<PhysicalTableInOutFunction>(op.types, op.function, std::move(op.bind_data), op.column_ids,
		//                                                   op.estimated_cardinality, std::move(op.projected_input));
		// node->children.push_back(CreatePlan(std::move(op.children[0])));
		// return std::move(node);
		throw NotImplementedException("Table producing functions that consume subquery results are not supported yet");
	}
	if (!op.projected_input.empty()) {
		throw InternalException("LogicalGet::project_input can only be set for table-in-out functions");
	}

	// FIXME: Upgrading to DuckDB v1.2.0
	unique_ptr<TableFilterSet> table_filters;
	const auto& column_ids = op.GetColumnIds();
	vector<column_t> column_id_vals;
	for (const auto& column_id: column_ids) {
		column_id_vals.push_back(column_id.GetPrimaryIndex());
	}
	if (!op.table_filters.filters.empty()) {
		table_filters = GPUCreateTableFilterSet(op.table_filters, column_id_vals);
	}

	if (op.function.dependency) {
		op.function.dependency(dependencies, op.bind_data.get());
	}
	// create the table scan node
	if (!op.function.projection_pushdown) {
		// function does not support projection pushdown
		auto node = make_uniq<GPUPhysicalTableScan>(op.returned_types, op.function, std::move(op.bind_data),
		                                         op.returned_types, column_id_vals, vector<column_t>(), op.names,
		                                         std::move(table_filters), op.estimated_cardinality, op.extra_info);
		// first check if an additional projection is necessary
		if (column_id_vals.size() == op.returned_types.size()) {
			bool projection_necessary = false;
			for (idx_t i = 0; i < column_id_vals.size(); i++) {
				if (column_id_vals[i] != i) {
					projection_necessary = true;
					break;
				}
			}
			if (!projection_necessary) {
				// a projection is not necessary if all columns have been requested in-order
				// in that case we just return the node

				return std::move(node);
			}
		}
		// push a projection on top that does the projection
		vector<LogicalType> types;
		vector<unique_ptr<Expression>> expressions;
		for (auto &column_id : column_id_vals) {
			if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType(LogicalTypeId::BIGINT));
				expressions.push_back(make_uniq<BoundConstantExpression>(Value::BIGINT(0)));
			} else {
				auto type = op.returned_types[column_id];
				types.push_back(type);
				expressions.push_back(make_uniq<BoundReferenceExpression>(type, column_id));
			}
		}

		auto projection =
		    make_uniq<GPUPhysicalProjection>(std::move(types), std::move(expressions), op.estimated_cardinality);
		projection->children.push_back(std::move(node));
		return std::move(projection);
	} else {
		return make_uniq<GPUPhysicalTableScan>(op.types, op.function, std::move(op.bind_data), op.returned_types,
		                                    column_id_vals, op.projection_ids, op.names, std::move(table_filters),
		                                    op.estimated_cardinality, op.extra_info);
	}
}

} // namespace duckdb
