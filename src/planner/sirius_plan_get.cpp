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

#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
// #include "duckdb/common/types.hpp"

namespace sirius::planner {

duckdb::unique_ptr<duckdb::TableFilterSet> create_table_filter_set(
  duckdb::TableFilterSet& table_filters, const duckdb::vector<duckdb::ColumnIndex>& column_ids)
{
  // create the table filter map
  auto table_filter_set = duckdb::make_uniq<duckdb::TableFilterSet>();
  for (auto& table_filter : table_filters.filters) {
    // find the relative column index from the absolute column index into the table
    duckdb::optional_idx column_index;
    for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
      if (table_filter.first == column_ids[i].GetPrimaryIndex()) {
        column_index = i;
        break;
      }
    }
    if (!column_index.IsValid()) {
      throw duckdb::InternalException("Could not find column index for table filter");
    }
    table_filter_set->filters[column_index.GetIndex()] = std::move(table_filter.second);
  }
  return table_filter_set;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalGet& op)
{
  auto column_ids = op.GetColumnIds();
  if (!op.children.empty()) {
    throw duckdb::NotImplementedException("Table Input Output functions are not supported yet");
    // duckdb::reference<sirius::op::sirius_physical_operator> child =
    // ResolveAndPlan(std::move(op.children[0])); auto &child_types = child.get().types;

    // // this is for table producing functions that consume subquery results
    // // push a projection node with casts if required
    // if (child_types.size() < op.input_table_types.size()) {
    // 	throw duckdb::InternalException(
    // 	    "Mismatch between input table types and child node types - expected %llu but got %llu",
    // 	    op.input_table_types.size(), child_types.size());
    // }

    // duckdb::vector<duckdb::LogicalType> return_types;
    // duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    // bool any_cast_required = false;
    // for (duckdb::idx_t proj_idx = 0; proj_idx < child_types.size(); proj_idx++) {
    // 	auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(child_types[proj_idx],
    // proj_idx); 	auto &target_type = 	    proj_idx < op.input_table_types.size() ?
    // op.input_table_types[proj_idx] : child_types[proj_idx]; 	if (child_types[proj_idx] !=
    // target_type) {
    // 		// cast is required - push a cast
    // 		any_cast_required = true;
    // 		auto cast = duckdb::BoundCastExpression::AddCastToType(context, std::move(ref),
    // target_type); 		expressions.push_back(std::move(cast)); 	} else {
    // 		expressions.push_back(std::move(ref));
    // 	}
    // 	return_types.push_back(target_type);
    // }

    // if (any_cast_required) {
    // 	auto &proj = Make<sirius::op::sirius_physical_operator>(std::move(return_types),
    // std::move(expressions), child.get().estimated_cardinality); 	proj.children.push_back(child);
    // 	child = proj;
    // }

    // auto &table_in_out =
    //     Make<PhysicalTableInOutFunction>(op.types, op.function, std::move(op.bind_data),
    //     column_ids,
    //                                      op.estimated_cardinality,
    //                                      std::move(op.projected_input));
    // table_in_out.children.push_back(child);
    // auto &cast_table_in_out = table_in_out.Cast<PhysicalTableInOutFunction>();
    // cast_table_in_out.ordinality_idx = op.ordinality_idx;
    // return table_in_out;
  }

  if (!op.projected_input.empty()) {
    throw duckdb::InternalException(
      "LogicalGet::project_input can only be set for table-in-out functions");
  }

  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  if (!op.table_filters.filters.empty()) {
    table_filters = create_table_filter_set(op.table_filters, column_ids);
  }

  if (op.function.dependency) { op.function.dependency(dependencies, op.bind_data.get()); }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> filter;
  auto& projection_ids = op.projection_ids;

  if (table_filters && op.function.supports_pushdown_type) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    duckdb::unique_ptr<duckdb::Expression> unsupported_filter;
    duckdb::unordered_set<duckdb::idx_t> to_remove;
    for (auto& entry : table_filters->filters) {
      auto column_id = column_ids[entry.first].GetPrimaryIndex();
      auto& type     = op.returned_types[column_id];
      if (!op.function.supports_pushdown_type(*op.bind_data, column_id)) {
        duckdb::idx_t column_id_filter = entry.first;
        bool found_projection          = false;
        for (duckdb::idx_t i = 0; i < projection_ids.size(); i++) {
          if (column_ids[projection_ids[i]] == column_ids[entry.first]) {
            column_id_filter = i;
            found_projection = true;
            break;
          }
        }
        if (!found_projection) {
          projection_ids.push_back(entry.first);
          column_id_filter = projection_ids.size() - 1;
        }
        auto column = duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, column_id_filter);
        select_list.push_back(entry.second->ToExpression(*column));
        to_remove.insert(entry.first);
      }
    }
    for (auto& col : to_remove) {
      table_filters->filters.erase(col);
    }

    if (!select_list.empty()) {
      duckdb::vector<duckdb::LogicalType> filter_types;
      for (auto& c : projection_ids) {
        auto column_id = column_ids[c].GetPrimaryIndex();
        filter_types.push_back(op.returned_types[column_id]);
      }
      filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
        filter_types, std::move(select_list), op.estimated_cardinality);
    }
  }
  op.ResolveOperatorTypes();
  // create the table scan node
  if (!op.function.projection_pushdown) {
    // function does not support projection pushdown
    // auto &table_scan = Make<sirius::op::sirius_physical_table_scan>(
    //     op.returned_types, op.function, std::move(op.bind_data), op.returned_types, column_ids,
    //     duckdb::vector<duckdb::column_t>(), op.names, std::move(table_filters),
    //     op.estimated_cardinality, std::move(op.extra_info), std::move(op.parameters),
    //     std::move(op.virtual_columns));

    auto node =
      duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(op.returned_types,
                                                                op.function,
                                                                std::move(op.bind_data),
                                                                op.returned_types,
                                                                column_ids,
                                                                duckdb::vector<duckdb::column_t>(),
                                                                op.names,
                                                                std::move(table_filters),
                                                                op.estimated_cardinality,
                                                                std::move(op.extra_info),
                                                                std::move(op.parameters),
                                                                std::move(op.virtual_columns));
    // first check if an additional projection is necessary
    if (column_ids.size() == op.returned_types.size()) {
      bool projection_necessary = false;
      for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
        if (column_ids[i].GetPrimaryIndex() != i) {
          projection_necessary = true;
          break;
        }
      }
      if (!projection_necessary) {
        // a projection is not necessary if all columns have been requested in-order
        // in that case we just return the node
        if (filter) {
          filter->children.push_back(std::move(node));
          return std::move(filter);
        }
        return std::move(node);
      }
    }
    // push a projection on top that does the projection
    duckdb::vector<duckdb::LogicalType> types;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
    for (auto& column_id : column_ids) {
      if (column_id.IsVirtualColumn()) {
        throw duckdb::NotImplementedException("Virtual columns require projection pushdown");
      } else {
        auto col_id = column_id.GetPrimaryIndex();
        auto type   = op.returned_types[col_id];
        types.push_back(type);
        expressions.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, col_id));
      }
    }
    duckdb::unique_ptr<sirius::op::sirius_physical_projection> projection =
      duckdb::make_uniq<sirius::op::sirius_physical_projection>(
        std::move(types), std::move(expressions), op.estimated_cardinality);
    if (filter) {
      filter->children.push_back(std::move(node));
      projection->children.push_back(std::move(filter));
    } else {
      projection->children.push_back(std::move(node));
    }
    return std::move(projection);
  }

  auto node =
    duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(op.types,
                                                              op.function,
                                                              std::move(op.bind_data),
                                                              op.returned_types,
                                                              column_ids,
                                                              op.projection_ids,
                                                              op.names,
                                                              std::move(table_filters),
                                                              op.estimated_cardinality,
                                                              std::move(op.extra_info),
                                                              std::move(op.parameters),
                                                              std::move(op.virtual_columns));
  node->dynamic_filters = op.dynamic_filters;
  if (filter) {
    filter->children.push_back(std::move(node));
    return std::move(filter);
  }
  return std::move(node);
}

}  // namespace sirius::planner
