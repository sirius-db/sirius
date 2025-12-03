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

#include "gpu_physical_plan_generator.hpp"

#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/execution/operator/helper/physical_verify_vector.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "gpu_csr_construction_operator.hpp"
#include "gpu_graph_traversal_operator.hpp"
#include "gpu_physical_table_scan.hpp"
#include "gpu_table_function.hpp"
#include "logical_graph_operator.hpp"

#include "log/logging.hpp"

namespace duckdb {

// class DependencyExtractor : public LogicalOperatorVisitor {
// public:
// 	explicit DependencyExtractor(LogicalDependencyList &dependencies) : dependencies(dependencies) {
// 	}

// protected:
// 	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expr, unique_ptr<Expression> *expr_ptr) override {
// 		// extract dependencies from the bound function expression
// 		if (expr.function.dependency) {
// 			expr.function.dependency(expr, dependencies);
// 		}
// 		return nullptr;
// 	}

// private:
// 	LogicalDependencyList &dependencies;
// };

GPUPhysicalPlanGenerator::GPUPhysicalPlanGenerator(ClientContext &context, GPUContext& gpu_context) : 
	context(context), gpu_context(gpu_context) {
}

GPUPhysicalPlanGenerator::~GPUPhysicalPlanGenerator() {
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(unique_ptr<LogicalOperator> op) {
	auto &profiler = QueryProfiler::Get(context);

	// first resolve column references
  if (op->type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
    profiler.StartPhase(MetricsType::PHYSICAL_PLANNER_COLUMN_BINDING);
    ColumnBindingResolver resolver;
    resolver.VisitOperator(*op);
    profiler.EndPhase();
  }

	// now resolve types of all the operators
	profiler.StartPhase(MetricsType::PHYSICAL_PLANNER_RESOLVE_TYPES);
	op->ResolveOperatorTypes();
	profiler.EndPhase();

	// extract dependencies from the logical plan
	// DependencyExtractor extractor(dependencies);
	// extractor.VisitOperator(*op);

	// then create the main physical plan
	profiler.StartPhase(MetricsType::PHYSICAL_PLANNER_CREATE_PLAN);
	auto plan = CreatePlan(*op);
	profiler.EndPhase();

	plan->Verify();
	return plan;
}

unique_ptr<GPUPhysicalOperator>
GPUPhysicalPlanGenerator::CreateEdgeTableScan(const string& table_name, const string& weight_column_name) {

  SIRIUS_LOG_INFO("CreateEdgeTableScan: Reading edge table '{}'", table_name);

  auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);
  auto &schema = catalog.GetSchema(context, DEFAULT_SCHEMA);
  auto transaction = schema.GetCatalogTransaction(context);
  auto table_or_view = schema.GetEntry(transaction, CatalogType::TABLE_ENTRY, table_name);
  if (!table_or_view) {
    throw CatalogException("Table '%s' not found", table_name);
  }

  auto &table_entry = table_or_view->Cast<TableCatalogEntry>();
  const auto &columns = table_entry.GetColumns();

  // Find the source and destination column indices
  vector<LogicalType> column_types;
  vector<string> column_names;
  vector<ColumnIndex> column_ids;  // Track which columns we're reading

  bool found_src = false;
  bool found_dst = false;
  bool found_weight = false;
  bool needs_weight = !weight_column_name.empty();

  for (idx_t i = 0; i < columns.PhysicalColumnCount(); i++) {
    auto &col = columns.GetColumn(PhysicalIndex(i));
    string col_name = col.GetName();

    // Add source column
    if (col_name == "src" || col_name == "source") { // Common names for source
      column_types.push_back(col.GetType());
      column_names.push_back(col_name);
      column_ids.push_back(ColumnIndex(i));
      found_src = true;
      SIRIUS_LOG_DEBUG("Found source column: {} at index {}", col_name, i);
    }

    // Add dest column
    else if (col_name == "dst" || col_name == "dest" || col_name == "target") { // Common names for dest
      column_types.push_back(col.GetType());
      column_names.push_back(col_name);
      column_ids.push_back(ColumnIndex(i));
      found_dst = true;
      SIRIUS_LOG_DEBUG("Found dest column: {} at index {}", col_name, i);
    }

    // Add weight column
    else if (needs_weight && (col_name == weight_column_name || col_name == "weight" || col_name == "cost" || col_name == "distance")) { // Common names for weight
      column_types.push_back(col.GetType());
      column_names.push_back(col_name);
      column_ids.push_back(ColumnIndex(i));
      found_weight = true;
      SIRIUS_LOG_DEBUG("Found weight column: {} at index {}", col_name, i);
    }
  }

  if (!found_src || !found_dst) {
    throw BinderException("Edge table '%s' must have source and destination columns", table_name);
  }

  if (needs_weight && !found_weight) {
    SIRIUS_LOG_WARN("Weight column requested but not found in table '{}', will use unweighted", table_name);
  }

  SIRIUS_LOG_DEBUG("Table '{}' has {} columns", table_name, column_names.size());
  for (size_t i = 0; i < column_names.size(); i++) {
    SIRIUS_LOG_DEBUG("  Column {}: {} ({})", i, column_names[i], column_types[i].ToString());
  }

  // Get the table function and bind data for scanning
  unique_ptr<FunctionData> bind_data;
  auto table_function = table_entry.GetScanFunction(context, bind_data);

  // Create a LogicalGet for the table with proper column information
  auto logical_get = make_uniq<LogicalGet>(
      0,  // binding index, will be reassigned during binding
      table_function,
      std::move(bind_data),
      column_types,
      column_names
  );

  // Set table index
  logical_get->table_index = table_entry.oid;
  logical_get->SetColumnIds(std::move(column_ids));

  SIRIUS_LOG_DEBUG("Created LogicalGet for table '{}' with oid={}", table_name, table_entry.oid);

  // Convert to physical plan
  return CreatePlan(*logical_get);
}


unique_ptr<GPUPhysicalOperator>
GPUPhysicalPlanGenerator::CreateGraphPhysicalPlan(LogicalGraphOperator* graph_op) {
  SIRIUS_LOG_DEBUG("Creating graph physical plan");

  // Create a table scan to read the edge table
  auto edge_scan = CreateEdgeTableScan(graph_op->edge_table, graph_op->weight_column);

  // Create CSR construction operator
  auto csr_builder = make_uniq<GPUCSRConstructionOperator>(
      std::move(edge_scan),
      graph_op->source_column,
      graph_op->dest_column,
      graph_op->weight_column,
      context,
      gpu_context
  );

  // Create graph traversal operator
  auto traversal = make_uniq<GPUGraphTraversalOperator>(
      std::move(csr_builder),
      graph_op->source_vertex,
      graph_op->source_vertices,
      graph_op->dest_vertex,
      graph_op->dest_vertices,
      graph_op->weight_column,
      graph_op->algorithm_type,
      graph_op->path_pattern,
      graph_op->max_hops,
      graph_op->output_columns,
      context,
      gpu_context
  );

  return traversal;
}

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalOperator &op) {
	op.estimated_cardinality = op.EstimateCardinality(context);
	unique_ptr<GPUPhysicalOperator> plan = nullptr;

	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET:
		plan = CreatePlan(op.Cast<LogicalGet>());
		break;
	case LogicalOperatorType::LOGICAL_PROJECTION:
		plan = CreatePlan(op.Cast<LogicalProjection>());
		break;
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		plan = CreatePlan(op.Cast<LogicalEmptyResult>());
		break;
	case LogicalOperatorType::LOGICAL_FILTER:
		plan = CreatePlan(op.Cast<LogicalFilter>());
		break;
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		plan = CreatePlan(op.Cast<LogicalAggregate>());
		break;
	case LogicalOperatorType::LOGICAL_WINDOW:
		throw NotImplementedException("Window not supported");
		// plan = CreatePlan(op.Cast<LogicalWindow>());
		break;
	case LogicalOperatorType::LOGICAL_UNNEST:
		throw NotImplementedException("Unnest not supported");
		// plan = CreatePlan(op.Cast<LogicalUnnest>());
		break;
	case LogicalOperatorType::LOGICAL_LIMIT:
		plan = CreatePlan(op.Cast<LogicalLimit>());
		break;
	case LogicalOperatorType::LOGICAL_SAMPLE:
		throw NotImplementedException("Sample not supported");
		// plan = CreatePlan(op.Cast<LogicalSample>());
		break;
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		plan = CreatePlan(op.Cast<LogicalOrder>());
		break;
	case LogicalOperatorType::LOGICAL_TOP_N:
		plan = CreatePlan(op.Cast<LogicalTopN>());
		break;
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		throw NotImplementedException("Copy to file not supported");
		// plan = CreatePlan(op.Cast<LogicalCopyToFile>());
		break;
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		plan = CreatePlan(op.Cast<LogicalDummyScan>());
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		throw NotImplementedException("Any join not supported");
		// plan = CreatePlan(op.Cast<LogicalAnyJoin>());
		break;
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		throw NotImplementedException("Asof join not supported");
		break;
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		plan = CreatePlan(op.Cast<LogicalComparisonJoin>());
		break;
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		throw NotImplementedException("Cross product not supported");
		// plan = CreatePlan(op.Cast<LogicalCrossProduct>());
		break;
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		throw NotImplementedException("Positional join not supported");
		// plan = CreatePlan(op.Cast<LogicalPositionalJoin>());
		break;
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		throw NotImplementedException("Set operation not supported");
		// plan = CreatePlan(op.Cast<LogicalSetOperation>());
		break;
	case LogicalOperatorType::LOGICAL_INSERT:
		throw NotImplementedException("Insert not supported");
		// plan = CreatePlan(op.Cast<LogicalInsert>());
		break;
	case LogicalOperatorType::LOGICAL_DELETE:
		throw NotImplementedException("Delete not supported");
		// plan = CreatePlan(op.Cast<LogicalDelete>());
		break;
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		plan = CreatePlan(op.Cast<LogicalColumnDataGet>());
		break;
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		plan = CreatePlan(op.Cast<LogicalDelimGet>());
		break;
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		plan = CreatePlan(op.Cast<LogicalExpressionGet>());
		break;
	case LogicalOperatorType::LOGICAL_UPDATE:
		throw NotImplementedException("Update not supported");
		// plan = CreatePlan(op.Cast<LogicalUpdate>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		throw NotImplementedException("Create table not supported");
		// plan = CreatePlan(op.Cast<LogicalCreateTable>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		throw NotImplementedException("Create index not supported");
		// plan = CreatePlan(op.Cast<LogicalCreateIndex>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_SECRET:
		throw NotImplementedException("Create secret not supported");
		// plan = CreatePlan(op.Cast<LogicalCreateSecret>());
		break;
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		throw NotImplementedException("Explain not supported");
		// plan = CreatePlan(op.Cast<LogicalExplain>());
		break;
	case LogicalOperatorType::LOGICAL_DISTINCT:
		throw NotImplementedException("Distinct not supported");
		// plan = CreatePlan(op.Cast<LogicalDistinct>());
		break;
	case LogicalOperatorType::LOGICAL_PREPARE:
		throw NotImplementedException("Prepare not supported");
		// plan = CreatePlan(op.Cast<LogicalPrepare>());
		break;
	case LogicalOperatorType::LOGICAL_EXECUTE:
		throw NotImplementedException("Execute not supported");
		// plan = CreatePlan(op.Cast<LogicalExecute>());
		break;
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		throw NotImplementedException("Create not supported");
		// plan = CreatePlan(op.Cast<LogicalCreate>());
		break;
	case LogicalOperatorType::LOGICAL_PRAGMA:
		throw NotImplementedException("Pragma not supported");
		// plan = CreatePlan(op.Cast<LogicalPragma>());
		break;
	case LogicalOperatorType::LOGICAL_VACUUM:
		throw NotImplementedException("Vacuum not supported");
		// plan = CreatePlan(op.Cast<LogicalVacuum>());
		break;
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:
		throw NotImplementedException("Simple not supported");
		// plan = CreatePlan(op.Cast<LogicalSimple>());
		break;
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		throw NotImplementedException("Recursive CTE not supported");
		// plan = CreatePlan(op.Cast<LogicalRecursiveCTE>());
		break;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		plan = CreatePlan(op.Cast<LogicalMaterializedCTE>());
		break;
	case LogicalOperatorType::LOGICAL_CTE_REF:
		plan = CreatePlan(op.Cast<LogicalCTERef>());
		break;
	case LogicalOperatorType::LOGICAL_EXPORT:
		throw NotImplementedException("Export not supported");
		// plan = CreatePlan(op.Cast<LogicalExport>());
		break;
	case LogicalOperatorType::LOGICAL_SET:
		throw NotImplementedException("Set not supported");
		// plan = CreatePlan(op.Cast<LogicalSet>());
		break;
	case LogicalOperatorType::LOGICAL_RESET:
		throw NotImplementedException("Reset not supported");
		// plan = CreatePlan(op.Cast<LogicalReset>());
		break;
	case LogicalOperatorType::LOGICAL_PIVOT:
		throw NotImplementedException("Pivot not supported");
		// plan = CreatePlan(op.Cast<LogicalPivot>());
		break;
	case LogicalOperatorType::LOGICAL_COPY_DATABASE:
		throw NotImplementedException("Copy database not supported");
		// plan = CreatePlan(op.Cast<LogicalCopyDatabase>());
		break;
	case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
		throw NotImplementedException("Update extensions not supported");
		// plan = CreatePlan(op.Cast<LogicalSimple>());
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR: {
	  // Check if it's a LogicalGraphOperator
	  auto graph_op = dynamic_cast<LogicalGraphOperator*>(&op);
	  if (graph_op) {
	    return CreateGraphPhysicalPlan(graph_op);
	  }
	  throw NotImplementedException("Unknown extension operator");
	}
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
	case LogicalOperatorType::LOGICAL_INVALID: {
		throw NotImplementedException("Unimplemented logical operator type!");
	}
	default:
		throw NotImplementedException("Unimplemented logical operator type");
	}
	if (!plan) {
		throw InternalException("Physical plan generator - no plan generated");
	}

	plan->estimated_cardinality = op.estimated_cardinality;
#ifdef DUCKDB_VERIFY_VECTOR_OPERATOR
	auto verify = make_uniq<PhysicalVerifyVector>(std::move(plan));
	plan = std::move(verify);
#endif

	return plan;
}

} // namespace duckdb