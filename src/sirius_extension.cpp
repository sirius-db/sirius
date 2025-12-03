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

#define DUCKDB_EXTENSION_MAIN

#include "sirius_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/statement/relation_statement.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"

#include "substrait_extension.hpp"
#include "to_substrait.hpp"
#include "from_substrait.hpp"

#include "log/logging.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "gpu_buffer_manager.hpp"
#include "config.hpp"
#include "logical_graph_operator.hpp"
#include "graph_query_parser.hpp"

#include <cstdlib>

#include <signal.h>
#include <execinfo.h>

namespace duckdb {

const std::string PINNED_MEMORY_PARAM_KEY = "pinned_memory_size";
bool SiriusExtension::buffer_is_initialized = false;
std::unordered_map<string, GraphMetadata> SiriusExtension::graph_registry; // Static registry (persists per session)

struct GPUTableFunctionData : public TableFunctionData {
	GPUTableFunctionData() = default;
	shared_ptr<Relation> plan;
	shared_ptr<GPUPreparedStatementData> gpu_prepared;
	unique_ptr<QueryResult> res;
	unique_ptr<Connection> conn;
	unique_ptr<GPUContext> gpu_context;
	string query;
	bool enable_optimizer;
	bool finished = false;
	bool plan_error = false;
};

struct GraphProcessingFunctionData : public TableFunctionData {
  GraphProcessingFunctionData() = default;

  // Execution resources
  shared_ptr<GPUPreparedStatementData> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<GPUContext> gpu_context;

  string graph_query;
  string edge_table;
  int64_t source_vertex;
  string algorithm_type;

  // Graph query parameters
  bool is_path_query = false;
  string path_pattern = "";
  string weight_column = "";
  bool is_left_directed = false;
  bool is_right_directed = false;
  bool is_any_directed = false;
  bool is_left_right_directed = false;

  // Execution state flags
  bool finished = false;
  bool parse_error = false;
  bool plan_error = false;
};

// struct GPUCachingFunctionData : public TableFunctionData {
// 	GPUCachingFunctionData() = default;
// 	unique_ptr<Connection> conn;
// 	GPUBufferManager *gpuBufferManager;
// 	GPUColumnType type;
// 	uint8_t *data;
// 	string column;
// 	string table;
// 	bool finished = false;
// };

void do_nothing_context(ClientContext *) {
}

//This function is used to extract the query plan from the SQL query
unique_ptr<LogicalOperator> SiriusInitPlanExtractor(ClientContext& context, GPUTableFunctionData &data, Connection &new_conn) {
	// The user might want to disable the optimizer of the new connection
	new_conn.context->config.enable_optimizer = data.enable_optimizer;
	new_conn.context->config.use_replacement_scans = false;

	// We want for sure to disable the internal compression optimizations.
	// These are DuckDB specific, no other system implements these. Also,
	// respect the user's settings if they chose to disable any specific optimizers.
	//
	// The InClauseRewriter optimization converts large `IN` clauses to a
	// "mark join" against a `ColumnDataCollection`, which may not make
	// sense in other systems and would complicate the conversion to Substrait.
	set<OptimizerType> disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;
	disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
	disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
	DBConfig::GetConfig(*new_conn.context).options.disabled_optimizers = disabled_optimizers;
	
	return new_conn.context->ExtractPlan(data.query);
}

unique_ptr<GPUPhysicalOperator> GPUGeneratePhysicalPlan(ClientContext& context, GPUContext& gpu_context, unique_ptr<LogicalOperator> &logical_plan, Connection &new_conn) {
	GPUPhysicalPlanGenerator physical_planner = GPUPhysicalPlanGenerator(context, gpu_context);
	auto physical_plan = physical_planner.CreatePlan(std::move(logical_plan));
	return physical_plan;
}

//The result of the GPUProcessingBind function is a unique pointer to a FunctionData object.
//This result of this function is used as an argument to the GPUProcessingFunction function (data_p argument), which is called to execute the table function.
// unique_ptr<FunctionData> 
// SiriusExtension::GPUCachingBind(ClientContext &context, TableFunctionBindInput &input,
//                                                 vector<LogicalType> &return_types, vector<string> &names) {
// 	auto result = make_uniq<GPUCachingFunctionData>();
// 	result->conn = make_uniq<Connection>(*context.db);
// 	if (input.inputs[0].IsNull()) {
// 		throw BinderException("gpu_caching cannot be called with a NULL parameter");
// 	}

// 	result->gpuBufferManager = &(GPUBufferManager::GetInstance());

// 	string input_string = input.inputs[0].ToString();
//     size_t pos = input_string.find('.');  // Find the position of the period

//     if (pos != string::npos) {
//         string table_name = input_string.substr(0, pos);  // Extract the first word
//         string column_name = input_string.substr(pos + 1); // Extract the second word
// 		result->table = table_name;
// 		result->column = column_name;
//     } else {
//         throw InvalidInputException("Incorrect input format, use table.column");
//     }

// 	return_types.emplace_back(LogicalType(LogicalTypeId::VARCHAR));
// 	names.emplace_back("GPU Caching");

// 	return std::move(result);
// }

// void SiriusExtension::GPUCachingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
// 	auto &data = (GPUCachingFunctionData &)*data_p.bind_data;
// 	if (data.finished) {
// 		return;
// 	}

// 	if (!buffer_is_initialized) {
// 		printf("\033[1;31m"); printf("GPUBufferManager not initialized, please call gpu_buffer_init first\n"); printf("\033[0m");
// 		return;
// 	}

// 	//get data in CPU buffer
// 	string query = "SELECT " + data.column + " FROM " + data.table + ";";
// 	SIRIUS_LOG_DEBUG("Query: {}", query);
// 	auto cpu_res = data.conn->Query(query);
	
// 	auto &catalog_table = Catalog::GetCatalog(context, INVALID_CATALOG);
// 	data.gpuBufferManager->createTableAndColumnInGPU(catalog_table, context, data.table, data.column);

// 	DataWrapper buffered_data = data.gpuBufferManager->allocateColumnBufferInCPU(move(cpu_res));
// 	// update the catalog in GPU buffer manager (adding tables/columns)

// 	data.gpuBufferManager->cacheDataInGPU(buffered_data, data.table, data.column, 0);  // Send data to GPU

// 	output.SetCardinality(1);
// 	output.SetValue(0, 0, "Successful");
// 	data.finished = true;

// 	return;
// }

//The result of the GPUProcessingBind function is a unique pointer to a FunctionData object.
//This result of this function is used as an argument to the GPUProcessingFunction function (data_p argument), which is called to execute the table function.
unique_ptr<FunctionData> 
SiriusExtension::GPUProcessingBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<GPUTableFunctionData>();
	result->conn = make_uniq<Connection>(*context.db);
	result->query = input.inputs[0].ToString();
	result->enable_optimizer = true;
	result->gpu_context = make_uniq<GPUContext>(context);
	if (input.inputs[0].IsNull()) {
		throw BinderException("gpu_processing cannot be called with a NULL parameter");
	}

	//Parse the query just to get the result type information and to create preparedstatmement data
	auto statements = result->conn->context->ParseStatements(result->query);
	Planner planner(context);
	auto statement_type = statements[0]->type;
	planner.CreatePlan(std::move(statements[0]));
	D_ASSERT(planner.plan);

	auto prepared = make_shared_ptr<PreparedStatementData>(statement_type);
	prepared->names = planner.names;
	prepared->types = planner.types;
	prepared->value_map = std::move(planner.value_map);
	prepared->plan = make_uniq<PhysicalOperator>(PhysicalOperatorType::DUMMY_SCAN, vector<LogicalType>{LogicalType::BOOLEAN}, 0);

	//generate physical plan from the logical plan
	unique_ptr<LogicalOperator> query_plan = SiriusInitPlanExtractor(context, *result, *result->conn);
	SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());
	if (buffer_is_initialized) {
		try {
			auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
			auto gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared), std::move(gpu_physical_plan));
			result->gpu_prepared = gpu_prepared;
		} catch (std::exception &e) {
			ErrorData error(e);
			SIRIUS_LOG_ERROR("Error in GPUGeneratePhysicalPlan: {}", error.RawMessage());
			result->plan_error = true;
		}
	} else {
		result->gpu_prepared = nullptr;
	}

	for (auto &column : planner.names) {
		names.emplace_back(column);
	}
	for (auto &type : planner.types) {
		return_types.emplace_back(type);
	}

	return std::move(result);
}

void SiriusExtension::GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (GPUTableFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}

	if (!data.res) {
		auto start = std::chrono::high_resolution_clock::now();
		if (!buffer_is_initialized) {
			printf("\033[1;31m"); printf("GPUBufferManager not initialized, please call gpu_buffer_init first\n"); printf("\033[0m");
			printf("=============================================\nError in GPUExecuteQuery, fallback to DuckDB\n=============================================\n");
			data.res = data.conn->Query(data.query);
		} else if (data.plan_error) {
			printf("=============================================\nError in GPUExecuteQuery, fallback to DuckDB\n=============================================\n");
			data.res = data.conn->Query(data.query);
		} else {
			data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
			if (data.res->HasError()) {
				printf("=============================================\nError in GPUExecuteQuery, fallback to DuckDB\n=============================================\n");
				data.res = data.conn->Query(data.query);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count()/1000.0);
	}

	auto result_chunk = data.res->Fetch();
	if (result_chunk == nullptr) {
		output.SetCardinality(0);
		return;
	}

	output.Reference(*result_chunk);
	return;
}

unique_ptr<LogicalOperator> OptimizePlan(ClientContext &context, Planner &planner, Connection &new_conn) {
	unique_ptr<LogicalOperator> plan;
	plan = std::move(planner.plan);

	Optimizer optimizer(*planner.binder, context);
	plan = optimizer.Optimize(std::move(plan));
	SIRIUS_LOG_DEBUG("Query plan:\n{}", plan->ToString());

	ColumnBindingResolver resolver;
	resolver.Verify(*plan);
	resolver.VisitOperator(*plan);

	plan->ResolveOperatorTypes();

	return plan;
}

unique_ptr<FunctionData> 
SiriusExtension::GPUProcessingSubstraitBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<GPUTableFunctionData>();
	result->conn = make_uniq<Connection>(*context.db);
	result->query = input.inputs[0].ToString();
	result->enable_optimizer = true;
	result->gpu_context = make_uniq<GPUContext>(context);
	if (input.inputs[0].IsNull()) {
		throw BinderException("gpu_processing cannot be called with a NULL parameter");
	}
	string serialized = input.inputs[0].GetValueUnsafe<string>();
	// result->plan = GPUSubstraitPlanToDuckDBRel(*result->conn, serialized, false);
	bool is_json = false;
	shared_ptr<ClientContext> c_ptr(&context, do_nothing_context);
	SubstraitToDuckDB transformer_s2d(c_ptr, serialized, is_json, false);
	result->plan = transformer_s2d.TransformPlan();

	auto relation_stmt = make_uniq<RelationStatement>(result->plan);
	unique_ptr<SQLStatement> statements = std::move(relation_stmt);
	auto statement_type = statements->type;
	SIRIUS_LOG_DEBUG("{}", statements->query);

	set<OptimizerType> disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;
	disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
	disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
	DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;

	Planner planner(context);
	planner.CreatePlan(std::move(statements));
	D_ASSERT(planner.plan);

	auto prepared = make_shared_ptr<PreparedStatementData>(statement_type);
	prepared->names = planner.names;
	prepared->types = planner.types;
	prepared->value_map = std::move(planner.value_map);
	prepared->plan = make_uniq<PhysicalOperator>(PhysicalOperatorType::DUMMY_SCAN, vector<LogicalType>{LogicalType::BOOLEAN}, 0);
	
	auto query_plan = OptimizePlan(context, planner, *result->conn);
	try {
		auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
		auto gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared), std::move(gpu_physical_plan));
		result->gpu_prepared = gpu_prepared;
	} catch (std::exception &e) {
		ErrorData error(e);
		SIRIUS_LOG_ERROR("Error in GPUGeneratePhysicalPlan: {}", error.RawMessage());
		result->plan_error = true;
	}


	for (auto &column : planner.names) {
		names.emplace_back(column);
	}
	for (auto &type : planner.types) {
		return_types.emplace_back(type);
	}

	return std::move(result);
}

void SiriusExtension::GPUProcessingSubstraitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (GPUTableFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}
	if (!data.res) {
		auto start = std::chrono::high_resolution_clock::now();
		if (!buffer_is_initialized) {
			printf("\033[1;31m"); printf("GPUBufferManager not initialized, please call gpu_buffer_init first\n"); printf("\033[0m");
			printf("=============================================\nError in GPUExecuteQuery, fallback to DuckDB\n=============================================\n");
			auto con = Connection(*context.db);
			data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
			data.res = data.plan->Execute();
		} else if (data.plan_error) {
			printf("=============================================\nError in GPUExecuteQuery, fallback to DuckDB\n=============================================\n");
			auto con = Connection(*context.db);
			data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
			data.res = data.plan->Execute();
		} else {
			data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
			if (data.res->HasError()) {
				printf("=============================================\nError in GPUExecuteQuery, fallback to DuckDB\n=============================================\n");
				auto con = Connection(*context.db);
				data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
				data.res = data.plan->Execute();
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		SIRIUS_LOG_INFO("GPU Execute query time: {:.2f} ms", duration.count()/1000.0);
	}

	auto result_chunk = data.res->Fetch();
	if (!result_chunk) {
		return;
	}
	output.Move(*result_chunk);
	return;
}

struct GPUBufferInitFunctionData : public TableFunctionData {
	GPUBufferInitFunctionData() {
	}
	bool finished = false;
	size_t cache_size;
	size_t processing_size;
	size_t pinned_memory_size;
};

unique_ptr<FunctionData> 
SiriusExtension::GPUBufferInitBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<GPUBufferInitFunctionData>();

	string gpu_cache_size = input.inputs[0].ToString();
	string gpu_processing_size = input.inputs[1].ToString();
	string pinned_memory_size("0 GB"); // Default size of pinned memory
	if(input.named_parameters.find(PINNED_MEMORY_PARAM_KEY) != input.named_parameters.end()) { 
		// If the pinned memory size is specified in the arguments then use that
		pinned_memory_size = input.named_parameters[PINNED_MEMORY_PARAM_KEY].ToString();
	}

	//parsing 2GB or 2GiB to size_t
	// Function to parse size strings like "2GB" or "2GiB" to size_t
	auto parse_size = [](const string &size_str) -> size_t {
		size_t result = 0;
		size_t multiplier = 1;
		string num_part;
		string unit_part;

		size_t i = 0;
		// Skip any whitespace between number and unit
		while (i < size_str.length() && isspace(size_str[i])) {
			i++;
		}

		// Find where the number ends and unit begins
		while (i < size_str.length() && (isdigit(size_str[i]) || size_str[i] == '.')) {
			num_part += size_str[i];
			i++;
		}
		
		// Skip any whitespace between number and unit
		while (i < size_str.length() && isspace(size_str[i])) {
			i++;
		}
		
		// Extract unit part
		unit_part = size_str.substr(i);

		// Convert number part to double
		double num_value = stod(num_part);
		
		// Determine multiplier based on unit
		if (unit_part == "B") {
			multiplier = 1;
		} else if (unit_part == "KB" || unit_part == "KiB") {
			multiplier = 1024;
		} else if (unit_part == "MB" || unit_part == "MiB") {
			multiplier = 1024 * 1024;
		} else if (unit_part == "GB" || unit_part == "GiB") {
			multiplier = 1024 * 1024 * 1024;
		} else if (unit_part == "TB" || unit_part == "TiB") {
			multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
		} else {
			throw InvalidInputException("Invalid format");
		}
		
		result = (size_t)(num_value * multiplier);
		return result;
	};

	// Parse the input sizes
	result->cache_size = parse_size(gpu_cache_size);
	result->processing_size = parse_size(gpu_processing_size);
	result->pinned_memory_size = parse_size(pinned_memory_size);

	auto type = LogicalType(LogicalTypeId::BOOLEAN);
	return_types.emplace_back(type);
	names.emplace_back("Success");
	return std::move(result);
}

void 
SiriusExtension::GPUBufferInitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.bind_data->CastNoConst<GPUBufferInitFunctionData>();
	if (data.finished) {
		return;
	}

	size_t cache_size = data.cache_size;
	size_t processing_size = data.processing_size;
	size_t pinned_memory_size = data.pinned_memory_size;
	if(pinned_memory_size == 0) {
		pinned_memory_size = std::max(cache_size, processing_size);
	}

	if (!buffer_is_initialized) {
		SIRIUS_LOG_DEBUG("GPU Buffer Manager initialized with args: Cache Size - {}, Processing Size - {}, Pinned Mem Size - {}\n", 
			cache_size, processing_size, pinned_memory_size);
		GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance(
			cache_size, processing_size, pinned_memory_size));
		buffer_is_initialized = true;
	} else {
		SIRIUS_LOG_WARN("GPUBufferManager already initialized");
	}
	data.finished = true;
}

ParsedGraphQuery SiriusExtension::ParseGraphQuery(const string& query) {
  ParsedGraphQuery result;

  SIRIUS_LOG_DEBUG("Parsing graph query: {}", query);

  result.edge_table = GraphQueryParser::ExtractEdgeTable(query);
  result.source_column = "src";
  result.dest_column = "dst";
  SIRIUS_LOG_DEBUG("Edge table: {}", result.edge_table);

  result.is_path_query = GraphQueryParser::IsPathQuery(query);
  auto algo_info = GraphQueryParser::DetectAlgorithm(query);
  result.algorithm_type = algo_info.algorithm_type;
  result.path_pattern = algo_info.path_pattern;
  SIRIUS_LOG_DEBUG("Algorithm: {}, Path pattern: {}",
                   result.algorithm_type, result.path_pattern);

  auto direction = GraphQueryParser::DetectDirection(query);
  result.is_left_directed = direction.is_left_directed;
  result.is_right_directed = direction.is_right_directed;
  result.is_any_directed = direction.is_any_directed;
  result.is_left_right_directed = direction.is_left_right_directed;
  auto vertex_patterns = GraphQueryParser::ExtractVertexPatterns(query);

  std::vector<int64_t> all_source_vertices;
  std::vector<int64_t> all_dest_vertices;
  for (size_t i = 0; i < vertex_patterns.size(); i++) {
    const auto& pattern = vertex_patterns[i];
    auto where_result = GraphQueryParser::ParseWhereClause(pattern);

    if (where_result.has_where && !where_result.vertex_ids.empty()) {
      // First vertex with WHERE = source, second = destination
      if (all_source_vertices.empty()) {
        all_source_vertices = where_result.vertex_ids;
        SIRIUS_LOG_DEBUG("Found {} source vertex(ies)", where_result.vertex_ids.size());
        for (auto id : where_result.vertex_ids) {
          SIRIUS_LOG_DEBUG("  Source: {}", id);
        }
      } else {
        all_dest_vertices = where_result.vertex_ids;
        SIRIUS_LOG_DEBUG("Found {} destination vertex(ies)", where_result.vertex_ids.size());
        for (auto id : where_result.vertex_ids) {
          SIRIUS_LOG_DEBUG("  Destination: {}", id);
        }
      }
    }
  }

  // Store vertices
  if (!all_source_vertices.empty()) {
    result.source_vertex = all_source_vertices[0];
    result.source_vertices = all_source_vertices;
  }
  if (!all_dest_vertices.empty()) {
    result.dest_vertex = all_dest_vertices[0];
    result.dest_vertices = all_dest_vertices;
  }

  result.output_columns = GraphQueryParser::ExtractColumns(query);
  SIRIUS_LOG_DEBUG("Parsed {} output columns", result.output_columns.size());

  if (result.algorithm_type == "WEIGHTED_SHORTEST_PATH") {
    string weight_var = GraphQueryParser::ExtractWeightColumn(query);
    if (!weight_var.empty() && weight_var != result.edge_table) {
      result.weight_column = weight_var;
    }
  }

  result.parse_success = !result.edge_table.empty() && result.source_vertex >= 0;
  SIRIUS_LOG_INFO("Parse complete - edge_table: {}, algo: {}, sources: {}, dest: {}, columns: {}",
                  result.edge_table, result.algorithm_type, result.source_vertices.size(),
                  result.dest_vertex, result.output_columns.size());

  return result;
}

unique_ptr<LogicalOperator>
SiriusExtension::CreateGraphLogicalPlan(const ParsedGraphQuery& parsed, ClientContext& context, Connection& conn) {

  if (!parsed.parse_success) {
    throw InvalidInputException("Failed to parse graph query");
  }

  SIRIUS_LOG_INFO("Creating graph logical plan:");

  // Create the logical graph operator
  auto graph_op = make_uniq<LogicalGraphOperator>(parsed);

  return graph_op;
}

unique_ptr<FunctionData>
SiriusExtension::GPUProcessingGraphBind(ClientContext &context, TableFunctionBindInput &input,
                                vector<LogicalType> &return_types, vector<string> &names) {
  auto result = make_uniq<GraphProcessingFunctionData>();
  result->conn = make_uniq<Connection>(*context.db);
  result->gpu_context = make_uniq<GPUContext>(context);

  if (input.inputs[0].IsNull()) {
      throw BinderException("graph_table cannot be called with a NULL parameter");
  }

  result->graph_query = input.inputs[0].ToString();

  SIRIUS_LOG_INFO("GPUProcessingGraphBind called with: {}", result->graph_query);

  // Parse the graph query
  auto parsed = ParseGraphQuery(result->graph_query);

  if (!parsed.parse_success) {
    result->parse_error = true;
    SIRIUS_LOG_ERROR("Failed to parse graph query: {}", result->graph_query);
    // Still return the function data, but mark it as errored
    return_types.emplace_back(LogicalType::BIGINT);
    names.emplace_back("error");
    return std::move(result);
  }

  result->edge_table = parsed.edge_table;
  result->source_vertex = parsed.source_vertex;
  result->algorithm_type = parsed.algorithm_type;

  SIRIUS_LOG_INFO("Successfully parsed graph query:");
  SIRIUS_LOG_INFO("  Edge table: {}", result->edge_table);
  SIRIUS_LOG_INFO("  Source vertex: {}", result->source_vertex);
  SIRIUS_LOG_INFO("  Algorithm: {}", result->algorithm_type);

  // Create logical plan for graph
  try {
    unique_ptr<LogicalOperator> query_plan = CreateGraphLogicalPlan(parsed, context, *result->conn);
    SIRIUS_LOG_DEBUG("Graph query plan:\n{}", query_plan->ToString());

    if (!parsed.output_columns.empty()) {
      for (const auto& col_spec : parsed.output_columns) {
        string col_name = col_spec;

        // Extract field name from dotted notation
        size_t dot_pos = col_name.find(".");
        if (dot_pos != string::npos) {
          col_name = col_name.substr(dot_pos + 1);
        }

        // Determine type and add directly - NO temporary variable
        if (col_name == "id" || col_name == "vertex_id" || col_name == "predecessor") {
          return_types.emplace_back(LogicalType(LogicalTypeId::BIGINT));
          names.emplace_back(col_name);
        } else if (col_name == "distance" || col_name == "weight" || col_name == "cost" || col_name == "hop") {
          bool is_weighted = (parsed.algorithm_type == "WEIGHTED_SHORTEST_PATH");

          if (is_weighted) {
            return_types.emplace_back(LogicalType(LogicalTypeId::DOUBLE));  // ← DOUBLE for weighted
          } else {
            return_types.emplace_back(LogicalType(LogicalTypeId::BIGINT));  // ← BIGINT for unweighted
          }
          names.emplace_back(col_name);
        } else if (col_name == "path") {
          return_types.emplace_back(LogicalType(LogicalTypeId::VARCHAR));
          names.emplace_back(col_name);
        } else {
          // Default fallback
          return_types.emplace_back(LogicalType(LogicalTypeId::BIGINT));
          names.emplace_back(col_name);
        }

        SIRIUS_LOG_DEBUG("Output column: {} ({})", col_name, return_types.back().ToString());
      }
    } else {
      // Default output
      return_types.emplace_back(LogicalType(LogicalTypeId::BIGINT));
      return_types.emplace_back(LogicalType(LogicalTypeId::BIGINT));
      names.emplace_back("vertex_id");
      names.emplace_back("distance");

      if (parsed.is_path_query) {
        return_types.emplace_back(LogicalType(LogicalTypeId::BIGINT));
        names.emplace_back("predecessor");
      }
    }

    // Create prepared statement
    auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
    prepared->names = names;
    prepared->types = return_types;
    prepared->plan = make_uniq<PhysicalOperator>(
      PhysicalOperatorType::DUMMY_SCAN,
      vector<LogicalType>{LogicalType::BOOLEAN},
      0
    );

    if (buffer_is_initialized) {
      try {
        // Generate GPU physical plan
        auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
        auto gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared), std::move(gpu_physical_plan));
        result->gpu_prepared = gpu_prepared;
      } catch (std::exception &e) {
        ErrorData error(e);
        SIRIUS_LOG_ERROR("Error in GPUGeneratePhysicalPlan: {}", error.RawMessage());
        result->plan_error = true;
      }
    } else {
      result->gpu_prepared = nullptr;
    }
  } catch (std::exception &e) {
    ErrorData error(e);
    SIRIUS_LOG_ERROR("Error creating graph logical plan: {}", error.RawMessage());
    result->plan_error = true;
  }

  return std::move(result);
}

void
SiriusExtension::GPUProcessingGraphFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
  auto &data = (GraphProcessingFunctionData &)*data_p.bind_data;
  if (data.finished) {
    return;
  }

  if (data.parse_error) {
    printf("\033[1;31m"); // Red color
    printf("Failed to parse graph query\n");
    printf("\033[0m"); // Reset color
    printf("=============================================\n");
    printf("Error in graph query parsing\n");
    printf("=============================================\n");
    SIRIUS_LOG_ERROR("Cannot execute graph query due to parse error");
    output.SetCardinality(0);
    data.finished = true;
    return;
  }

    if (!data.res) {
      auto start = std::chrono::high_resolution_clock::now();

      if (!buffer_is_initialized) {
        printf("\033[1;31m"); // Red color
        printf("GPUBufferManager not initialized, please call gpu_buffer_init first\n");
        printf("\033[0m"); // Reset color
        printf("=============================================\n");
        printf("Error in GPUExecuteQuery, fallback to DuckDB\n");
        printf("=============================================\n");
        SIRIUS_LOG_ERROR("GPUBufferManager not initialized, please call gpu_buffer_init first");
        output.SetCardinality(0);
        data.finished = true;
        return;
      } else if (data.plan_error) {
        printf("\033[1;31m"); // Red color
        printf("Error in query planning\n");
        printf("\033[0m"); // Reset color
        printf("=============================================\n");
        printf("Error in graph query planning\n");
        printf("=============================================\n");
        SIRIUS_LOG_ERROR("Error in query planning, cannot execute");
        output.SetCardinality(0);
        data.finished = true;
        return;
      } else {
        data.res = data.gpu_context->GPUExecuteQuery(context, data.graph_query, data.gpu_prepared, {});
        if (data.res->HasError()) {
          printf("\033[1;31m"); // Red color
          printf("Error in GPUExecuteQuery: %s\n", data.res->GetError().c_str());
          printf("\033[0m"); // Reset color
          printf("=============================================\n");
          printf("Error in graph query execution\n");
          printf("=============================================\n");
          SIRIUS_LOG_ERROR("Error in GPUExecuteQuery: {}", data.res->GetError());
          output.SetCardinality(0);
          data.finished = true;
          return;
        }
      }

      auto end = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      SIRIUS_LOG_INFO("Graph query execution time: {:.2f} ms", duration.count()/1000.0);
    }

    auto result_chunk = data.res->Fetch();
    if (result_chunk == nullptr) {
      output.SetCardinality(0);
      data.finished = true;
      return;
    }

    output.Reference(*result_chunk);
}

void SiriusExtension::RegisterGraph(const string& graph_name, const GraphMetadata& metadata) {
  graph_registry[graph_name] = metadata;
  SIRIUS_LOG_INFO("Registered graph '{}': vertex={}, edge={}, src={}, dst={}, weight={}",
                  graph_name, metadata.vertex_table, metadata.edge_table,
                  metadata.src_column, metadata.dst_column, metadata.weight_column);
}

GraphMetadata* SiriusExtension::LookupGraph(const string& graph_name) {
  auto it = graph_registry.find(graph_name);
  if (it != graph_registry.end()) {
    return &(it->second);
  }
  return nullptr;
}

// Scalar function for registration
static void RegisterGraphScalarFunction(DataChunk &args, ExpressionState &state, Vector &result) {
  auto &graph_name_vec = args.data[0];
  auto &vertex_table_vec = args.data[1];
  auto &edge_table_vec = args.data[2];
  auto &src_col_vec = args.data[3];
  auto &dst_col_vec = args.data[4];
  auto &weight_col_vec = args.data[5];

  auto graph_name = graph_name_vec.GetValue(0).ToString();
  auto vertex_table = vertex_table_vec.GetValue(0).ToString();
  auto edge_table = edge_table_vec.GetValue(0).ToString();
  auto src_col = src_col_vec.GetValue(0).ToString();
  auto dst_col = dst_col_vec.GetValue(0).ToString();

  // Weight column is optional (can be NULL)
  string weight_col = "";
  if (!weight_col_vec.GetValue(0).IsNull()) {
    weight_col = weight_col_vec.GetValue(0).ToString();
  }

  GraphMetadata metadata{vertex_table, edge_table, src_col, dst_col, weight_col};
  SiriusExtension::RegisterGraph(graph_name, metadata);

  // Return success
  result.SetValue(0, Value::BOOLEAN(true));
}


void SiriusExtension::InitializeGPUExtension(Connection &con) {
	auto &catalog = Catalog::GetSystemCatalog(*con.context);

	TableFunction gpu_buffer_init("gpu_buffer_init", {LogicalType::VARCHAR, LogicalType::VARCHAR}, GPUBufferInitFunction, GPUBufferInitBind);
	gpu_buffer_init.named_parameters[PINNED_MEMORY_PARAM_KEY] = LogicalType::VARCHAR;
	CreateTableFunctionInfo gpu_buffer_init_info(gpu_buffer_init);
	catalog.CreateTableFunction(*con.context, gpu_buffer_init_info);

	// TableFunction gpu_caching("gpu_caching", {LogicalType::VARCHAR}, GPUCachingFunction, GPUCachingBind);
	// CreateTableFunctionInfo gpu_caching_info(gpu_caching);
	// catalog.CreateTableFunction(*con.context, gpu_caching_info);

	TableFunction gpu_processing("gpu_processing", {LogicalType::VARCHAR}, GPUProcessingFunction, GPUProcessingBind);
	gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_info(gpu_processing);
	catalog.CreateTableFunction(*con.context, gpu_processing_info);

	TableFunction gpu_processing_substrait("gpu_processing_substrait", {LogicalType::BLOB}, GPUProcessingSubstraitFunction, GPUProcessingSubstraitBind);
	// gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_substrait_info(gpu_processing_substrait);
	catalog.CreateTableFunction(*con.context, gpu_processing_substrait_info);

  // graph
  TableFunction gpu_processing_graph("gpu_processing_graph", {LogicalType::VARCHAR}, GPUProcessingGraphFunction, GPUProcessingGraphBind);
  CreateTableFunctionInfo gpu_processing_graph_info(gpu_processing_graph);
  catalog.CreateTableFunction(*con.context, gpu_processing_graph_info);

	// size_t cache_size_per_gpu = 100UL * 1024 * 1024 * 1024; // 10GB
	// size_t processing_size_per_gpu = 80UL * 1024 * 1024 * 1024; //11GB
	// size_t processing_size_per_cpu = 100UL * 1024 * 1024 * 1024; //16GB
	// size_t cache_size_per_gpu = 10UL * 1024 * 1024 * 1024; // 10GB
	// size_t processing_size_per_gpu = 11UL * 1024 * 1024 * 1024; //11GB
	// size_t processing_size_per_cpu = 16UL * 1024 * 1024 * 1024; //16GB
	// GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance(cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu));

}

static void SetUsePinMemory(ClientContext &context, SetScope scope, Value &parameter) {
	Config::USE_PIN_MEM_FOR_CPU_PROCESSING = BooleanValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CPU_PROCESSING to {}", Config::USE_PIN_MEM_FOR_CPU_PROCESSING);
}

static void SetUseCudfExpr(ClientContext &context, SetScope scope, Value &parameter) {
	Config::USE_CUDF_EXPR = BooleanValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config USE_CUDF_EXPR to {}", Config::USE_CUDF_EXPR);
}

static void SetUseCustomTopN(ClientContext &context, SetScope scope, Value &parameter) {
	Config::USE_CUSTOM_TOP_N = BooleanValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config USE_CUSTOM_TOP_N to {}", Config::USE_CUSTOM_TOP_N);
}

static void SetUseOptTableScan(ClientContext &context, SetScope scope, Value &parameter) {
	Config::USE_OPT_TABLE_SCAN = BooleanValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config USE_OPT_TABLE_SCAN to {}", Config::USE_OPT_TABLE_SCAN);
}

static void SetOptTableScanNumStreams(ClientContext &context, SetScope scope, Value &parameter) {
	Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS = IntegerValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_NUM_CUDA_STREAMS to {}", Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS);
}

static void SetOptTableScanMemcpySize(ClientContext &context, SetScope scope, Value &parameter) {
	Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = UBigIntValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE to {}", Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE);
}

static void SetPrintGPUTableMaxRows(ClientContext &context, SetScope scope, Value &parameter) {
	Config::PRINT_GPU_TABLE_MAX_ROWS = UBigIntValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config PRINT_GPU_TABLE_MAX_ROWS to {}", Config::PRINT_GPU_TABLE_MAX_ROWS);
}

static void SetEnableFallbackCheck(ClientContext &context, SetScope scope, Value &parameter) {
	Config::ENABLE_FALLBACK_CHECK = BooleanValue::Get(parameter);
	SIRIUS_LOG_DEBUG("Updated config ENABLE_FALLBACK_CHECK to {}", Config::ENABLE_FALLBACK_CHECK);
}

void SiriusExtension::InitialGPUConfigs(DuckDB &db) {
	auto &config = DBConfig::GetConfig(*db.instance);

	// Add in config option for gpu buffer manager
	config.AddExtensionOption("use_pin_memory", "Whether or not the buffer manager is initialized with pinned memory", LogicalType::BOOLEAN, 
		Value::BOOLEAN(Config::USE_PIN_MEM_FOR_CPU_PROCESSING), SetUsePinMemory);

	// Add in config option for expression executor
	config.AddExtensionOption("use_cudf_expr", "Whether or not cudf is used to evaluate expressions", LogicalType::BOOLEAN, 
		Value::BOOLEAN(Config::USE_CUDF_EXPR), SetUseCudfExpr);

	// Add in config option for top-N
	config.AddExtensionOption("use_custom_top_n", "Whether or not custom kernel is used to evalaute top n", LogicalType::BOOLEAN, 
		Value::BOOLEAN(Config::USE_CUSTOM_TOP_N), SetUseCustomTopN);

	// Add in config options for custom table scan
	config.AddExtensionOption("use_opt_table_scan", "Whether or not the optional table scan is used", LogicalType::BOOLEAN, 
		Value::BOOLEAN(Config::USE_OPT_TABLE_SCAN), SetUseOptTableScan);
	config.AddExtensionOption("opt_table_scan_num_streams", "The number of cuda streams to use in the optional table scan", LogicalType::INTEGER, 
		Value::INTEGER(Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS), SetOptTableScanNumStreams);
	config.AddExtensionOption("opt_table_scan_memcpy_size", "The memcpy size (in bytes) used by the optional table scan", LogicalType::UBIGINT, 
		Value::UBIGINT(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE), SetOptTableScanMemcpySize);

	// Add in config options for printing gpu table
	config.AddExtensionOption("print_gpu_table_max_rows", "Maximal amount of rows to render when printing gpu table", LogicalType::UBIGINT, 
		Value::UBIGINT(Config::PRINT_GPU_TABLE_MAX_ROWS), SetPrintGPUTableMaxRows);
	
	// Add in config options for duckdb fallback checking
	config.AddExtensionOption("enable_fallback_check", "Whether to enable checking of fallback to duckdb execution", LogicalType::BOOLEAN, 
		Value::BOOLEAN(Config::ENABLE_FALLBACK_CHECK), SetEnableFallbackCheck);
}


void sigsegv_handler(int sig) {
  void *array[10];
  size_t size = backtrace(array, 10);
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

void SiriusExtension::Load(DuckDB &db) {
  // debugging helper
  signal(SIGSEGV, sigsegv_handler);

	// First initialize the config before acquring a connection the database
	InitialGPUConfigs(db);
	
	Connection con(db);
	con.BeginTransaction();

	InitGlobalLogger();
	InitializeGPUExtension(con);

	con.Commit();

  // Register the graph registration function
  auto register_graph_func = ScalarFunction(
    "sirius_register_graph",
    {
      LogicalType::VARCHAR,   // graph_name
      LogicalType::VARCHAR,   // vertex_table
      LogicalType::VARCHAR,   // edge_table
      LogicalType::VARCHAR,   // src_column
      LogicalType::VARCHAR,   // dst_column
      LogicalType::VARCHAR  // weight_column (optional, can pass NULL)
    },
    LogicalType::BOOLEAN,    // returns true on success
    RegisterGraphScalarFunction
  );

  ExtensionUtil::RegisterFunction(*db.instance, register_graph_func);

  SIRIUS_LOG_INFO("Registered sirius_register_graph function");
}

std::string SiriusExtension::Name() {
	return "GPU	Extension";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void sirius_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::SiriusExtension>();
}

DUCKDB_EXTENSION_API const char *sirius_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
