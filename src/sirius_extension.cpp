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
#include "duckdb/common/assert.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"

#include "substrait_extension.hpp"
#include "to_substrait.hpp"
#include "from_substrait.hpp"

#include "gpu_physical_plan_generator.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

void do_nothing_context(ClientContext *) {
}

// shared_ptr<Relation> GPUSubstraitPlanToDuckDBRel(Connection &conn, const string &serialized, bool json = false) {
// 	SubstraitToDuckDB transformer_s2d(conn, serialized, json);
// 	return transformer_s2d.TransformPlan();
// };

//This function is used to extract the query plan from the SQL query
unique_ptr<LogicalOperator> SiriusInitPlanExtractor(
	ClientContext& context, SiriusExtension::GPUTableFunctionData &data, Connection &new_conn) {
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
unique_ptr<FunctionData> 
SiriusExtension::GPUCachingBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<GPUCachingFunctionData>();
	result->conn = make_uniq<Connection>(*context.db);
	if (input.inputs[0].IsNull()) {
		throw BinderException("gpu_caching cannot be called with a NULL parameter");
	}

	// size_t cache_size_per_gpu = 2UL * 1024 * 1024 * 1024; // 10GB
	// size_t processing_size_per_gpu = 2UL * 1024 * 1024 * 1024; //11GB
	// size_t processing_size_per_cpu = 4UL * 1024 * 1024 * 1024; //16GB
	// size_t cache_size_per_gpu = 120UL * 1024 * 1024 * 1024;
	// size_t processing_size_per_gpu = 80UL * 1024 * 1024 * 1024;
	// size_t processing_size_per_cpu = 120UL * 1024 * 1024 * 1024;
	// result->gpuBufferManager = &(GPUBufferManager::GetInstance(cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu));
	result->gpuBufferManager = &(GPUBufferManager::GetInstance());

	string input_string = input.inputs[0].ToString();
    size_t pos = input_string.find('.');  // Find the position of the period

    if (pos != string::npos) {
        string table_name = input_string.substr(0, pos);  // Extract the first word
        string column_name = input_string.substr(pos + 1); // Extract the second word
		result->table = table_name;
		result->column = column_name;
    } else {
        throw InvalidInputException("Incorrect input format, use table.column");
    }

	return_types.emplace_back(LogicalType(LogicalTypeId::VARCHAR));
	names.emplace_back("GPU Caching");

	return std::move(result);
}

void SiriusExtension::GPUCachingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (GPUCachingFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}

	//get data in CPU buffer
	string query = "SELECT " + data.column + " FROM " + data.table + ";";
	cout << "Query: " << query << endl;
	// string query = "SELECT l_orderkey FROM lineitem;";
	auto cpu_res = data.conn->Query(query);
	if (cpu_res->HasError()) {
		throw InternalException(cpu_res->GetError());
	}
	
	//check if table exist in the catalog
	//check if table already exist in the gpu buffer
	//check if column exist in the catalog
	//check if column already exist in the gpu buffer
	auto &catalog_table = Catalog::GetCatalog(context, INVALID_CATALOG);
	data.gpuBufferManager->createTableAndColumnInGPU(catalog_table, context, data.table, data.column);

	DataWrapper buffered_data = data.gpuBufferManager->allocateColumnBufferInCPU(move(cpu_res));
	// update the catalog in GPU buffer manager (adding tables/columns)

	data.gpuBufferManager->cacheDataInGPU(buffered_data, data.table, data.column, 0);  // Send data to GPU

	output.SetCardinality(1);
	output.SetValue(0, 0, "Successful");
	data.finished = true;

	return;
}

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
	query_plan->Print();
	try {
		auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
		auto gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared), std::move(gpu_physical_plan));
		result->gpu_prepared = gpu_prepared;
	} catch (std::exception &e) {
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

void SiriusExtension::GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (GPUTableFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}

	if (!data.res) {
		auto start = std::chrono::high_resolution_clock::now();
		if (data.plan_error) {
			data.res = data.conn->Query(data.query);
		} else {
			data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
			if (data.res->HasError()) {
				printf("Error in GPUExecuteQuery, fallback to DuckDB\n");
				data.res = data.conn->Query(data.query);
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		printf("GPU Execute query time: %.2f ms\n", duration.count()/1000.0);
	}

	// data.finished = true;
	// printf("Fetching chunk first\n");
	// auto start = std::chrono::high_resolution_clock::now();
	auto result_chunk = data.res->Fetch();
	if (!result_chunk) {
		// printf("Not doing anything\n");
		return;
	}
	// output.Move(*result_chunk);
	// while (result_chunk) {
		// printf("Fetching chunk %d\n", result_chunk->size());
		output.Move(*result_chunk);
		// result_chunk = data.res->Fetch();
	// }
	// printf("Finished Fetching chunk\n");
	// auto end = std::chrono::high_resolution_clock::now();
	// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	// printf("Fetching time: %.2f ms\n", duration.count()/1000.0);
	return;
}

unique_ptr<LogicalOperator> OptimizePlan(ClientContext &context, Planner &planner, Connection &new_conn) {
	unique_ptr<LogicalOperator> plan;
	plan = std::move(planner.plan);

	Optimizer optimizer(*planner.binder, context);
	plan = optimizer.Optimize(std::move(plan));
	plan->Print();

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
	bool is_json;

	// Additional inputs if used by sirius server
	if (input.inputs.size() > 1) {
		ExtractSiriusServerAdditionalInputs(input.inputs[1], &is_json);
	} else {
		is_json = true;
	}

	shared_ptr<ClientContext> c_ptr(&context, do_nothing_context);
	SubstraitToDuckDB transformer_s2d(c_ptr, serialized, is_json, false);
	result->plan = transformer_s2d.TransformPlan();

	auto relation_stmt = make_uniq<RelationStatement>(result->plan);
	unique_ptr<SQLStatement> statements = std::move(relation_stmt);
	auto statement_type = statements->type;

	Planner planner(context);
	planner.CreatePlan(std::move(statements));
	D_ASSERT(planner.plan);

	if (USE_SIRIUS_FOR_SUBSTRAIT) {
		// Get output column names of this plan, only used by sirius server in distributed execution
		auto substrait_plan = transformer_s2d.get_substrait_plan();
		for (const auto& column_name: substrait_plan->relations(0).root().names()) {
			result->result_exchange_table_info.column_names.push_back(column_name);
		}

		set<OptimizerType> disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;
		disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
		disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
		DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;

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
			result->plan_error = true;
		}
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
		bool use_duckdb = !USE_SIRIUS_FOR_SUBSTRAIT;

		// Check plan error
		if (!use_duckdb) {
			if (data.plan_error) {
				if (data.is_sirius_server) {
					// Fallback should not be used by sirius server
					throw InternalException("Plan Error in GPUProcessingSubstraitBind");
				} else {
					// Fallback is used in standalone sirius
					printf("Plan Error in GPUProcessingSubstraitBind, fallback to DuckDB\n");
					use_duckdb = true;
				}
			}
		}

		// Try sirius execution
		if (!use_duckdb) {
			if (data.is_sirius_server && data.has_result_exchange_table_info) {
				data.gpu_context->result_exchange_table_info = &data.result_exchange_table_info;
			}
			data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
			if (data.res != nullptr && data.res->HasError()) {
				if (data.is_sirius_server) {
					// Fallback should not be used by sirius server
					throw InternalException("Error in GPUExecuteQuery: " + data.res->GetError());
				} else {
					// Fallback is used in standalone sirius
					printf("Error in GPUExecuteQuery, fallback to DuckDB. Error: %s\n", data.res->GetError().c_str());
					use_duckdb = true;
				}
			}
		}

		// Fallback to duckdb execution if needed
		if (use_duckdb) {
			auto con = Connection(*context.db);
			data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
			data.res = data.plan->Execute();
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		printf("GPU Execute query time: %.2f ms\n", duration.count()/1000.0);
	}

	// Should not consume result if the call is from Sirius server
	if (!data.is_sirius_server) {
		if (auto result_chunk = data.res->Fetch()) {
			output.Move(*result_chunk);
		}
	}

	return;
}

void SiriusExtension::InitializeGPUExtension(Connection &con) {
	auto &catalog = Catalog::GetSystemCatalog(*con.context);
	// auto &catalog_table = Catalog::GetCatalog(*con.context, INVALID_CATALOG);
	// string name = catalog_table.GetName();
	// printf("catalog name %s\n", catalog_table.GetName().c_str());

	// auto &schema = catalog_table.GetSchema(*con.context, DEFAULT_SCHEMA);
	// string schema_name = schema.name;
	// printf("schema name %s\n", schema.name.c_str());

	// auto duck_schema = schema.Cast<DuckSchemaEntry>();

	// TableCatalogEntry &table = catalog_table.GetEntry(*con.context, CatalogType::TABLE_ENTRY, schema_name, "lineitem").Cast<TableCatalogEntry>();
	// printf("%s\n", table.name.c_str());

	// for (auto &column_name : table.GetColumns().GetColumnNames()) {
	// 	printf("column name %s\n", column_name.c_str());
	// }

	TableFunction gpu_caching("gpu_caching", {LogicalType::VARCHAR}, GPUCachingFunction, GPUCachingBind);
	CreateTableFunctionInfo gpu_caching_info(gpu_caching);
	catalog.CreateTableFunction(*con.context, gpu_caching_info);

	TableFunction gpu_processing("gpu_processing", {LogicalType::VARCHAR}, GPUProcessingFunction, GPUProcessingBind);
	gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_info(gpu_processing);
	catalog.CreateTableFunction(*con.context, gpu_processing_info);

	TableFunction gpu_processing_substrait("gpu_processing_substrait", {LogicalType::VARCHAR}, GPUProcessingSubstraitFunction, GPUProcessingSubstraitBind);
	// gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_substrait_info(gpu_processing_substrait);
	catalog.CreateTableFunction(*con.context, gpu_processing_substrait_info);

	size_t cache_size_per_gpu = 1UL * 1024 * 1024 * 1024; // 1GB
	size_t processing_size_per_gpu = 1UL * 1024 * 1024 * 1024; //1GB
	size_t processing_size_per_cpu = 64UL * 1024 * 1024 * 1024; //16GB
	GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance(cache_size_per_gpu, processing_size_per_gpu, processing_size_per_cpu));	
}

void SiriusExtension::Load(DuckDB &db) {
	Connection con(db);
	con.BeginTransaction();

	InitializeGPUExtension(con);

	con.Commit();
}

std::string SiriusExtension::Name() {
	return "GPU	Extension";
}

const string SiriusExtension::KEY_PLAN_IN_JSON = "plan_in_json";

Value SiriusExtension::MakeSiriusServerAdditionalInputsStruct(bool plan_in_json) {
	return Value::STRUCT({
		{KEY_PLAN_IN_JSON, Value::BOOLEAN(plan_in_json)}
	});
}

void SiriusExtension::ExtractSiriusServerAdditionalInputs(const Value& additional_input_struct, bool* plan_in_json) {
	const auto& struct_type = additional_input_struct.type();
	const auto& struct_children = StructValue::GetChildren(additional_input_struct);
	for (idx_t i = 0; i < struct_children.size(); ++i) {
		const auto& key = StructType::GetChildName(struct_type, i);
		if (key == KEY_PLAN_IN_JSON) {
			*plan_in_json = BooleanValue::Get(struct_children[i]);
		} else {
			throw InternalException("Invalid key in sirius server additional inputs: " + key);
		}
	}
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void Sirius_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::SiriusExtension>();
}

DUCKDB_EXTENSION_API const char *Sirius_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
