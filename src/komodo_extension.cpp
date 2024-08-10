#define DUCKDB_EXTENSION_MAIN

#include "komodo_extension.hpp"
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

#include "substrait_extension.hpp"
#include "to_substrait.hpp"
#include "from_substrait.hpp"

#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
// OpenSSL linked through vcpkg
// #include <openssl/opensslv.h>

namespace duckdb {

struct GPUTableFunctionData : public TableFunctionData {
	GPUTableFunctionData() = default;
	// unique_ptr<LogicalOperator> logical_plan;
	shared_ptr<Relation> plan;
	shared_ptr<GPUPreparedStatementData> gpu_prepared;
	unique_ptr<QueryResult> res;
	unique_ptr<Connection> conn;
	unique_ptr<GPUContext> gpu_context;
	string query;
	bool enable_optimizer;
	bool finished = false;
};

shared_ptr<Relation> GPUSubstraitPlanToDuckDBRel(Connection &conn, const string &serialized, bool json = false) {
	SubstraitToDuckDB transformer_s2d(conn, serialized, json);
	return transformer_s2d.TransformPlan();
};

//This function is used to extract the query plan from the SQL query
unique_ptr<LogicalOperator> KomodoInitPlanExtractor(ClientContext& context, GPUTableFunctionData &data, Connection &new_conn) {
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
static unique_ptr<FunctionData> GPUProcessingBind(ClientContext &context, TableFunctionBindInput &input,
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
	// prepared->catalog_version = MetaTransaction::Get(context).catalog_version;

	//generate physical plan from the logical plan
	unique_ptr<LogicalOperator> query_plan = KomodoInitPlanExtractor(context, *result, *result->conn);
	query_plan->Print();
	auto gpu_physical_plan = GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
	auto gpu_prepared = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared), std::move(gpu_physical_plan));
	
	throw BinderException("GPUProcessingBind not implemented yet");
	
	result->gpu_prepared = gpu_prepared;

	for (auto &column : planner.names) {
		names.emplace_back(column);
	}
	for (auto &type : planner.types) {
		return_types.emplace_back(type);
	}

	return std::move(result);
}

static unique_ptr<FunctionData> GPUProcessingSubstraitBind(ClientContext &context, TableFunctionBindInput &input,
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
	result->plan = GPUSubstraitPlanToDuckDBRel(*result->conn, serialized, false);

	for (auto &column : result->plan->Columns()) {
		return_types.emplace_back(column.Type());
		names.emplace_back(column.Name());
	}

	return std::move(result);
}

static void GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (GPUTableFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}

	if (!data.res) {
		// data.res = data.plan->Execute();
		std::cout << "Calling CUDA kernel from C++..." << std::endl;
		myKernel();  // Call the CUDA kernel defined in komodo_extension_cuda.cu
		int size = 10;
		int* temp = new int[size];
		int* ptr = sendDataToGPU(temp, size);  // Send data to GPU
		std::cout << "CUDA kernel call finished." << std::endl;
		data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
		data.res = data.conn->Query(data.query);
	}

	// data.finished = true;
	printf("Fetching chunk first\n");
	auto result_chunk = data.res->Fetch();
	if (!result_chunk) {
		printf("Not doing anything\n");
		return;
	}
	// output.Move(*result_chunk);
	while (result_chunk) {
		// printf("Fetching chunk %d\n", result_chunk->size());
		output.Move(*result_chunk);
		result_chunk = data.res->Fetch();
	}
	printf("Finished Fetching chunk\n");
	return;
}

static void GPUProcessingSubstraitFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (GPUTableFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}

	if (!data.res) {
		// data.res = data.plan->Execute();
		std::cout << "Calling CUDA kernel from C++..." << std::endl;
		myKernel();  // Call the CUDA kernel defined in komodo_extension_cuda.cu
		int size = 10;
		int* temp = new int[size];
		int* ptr = sendDataToGPU(temp, size);  // Send data to GPU
		std::cout << "CUDA kernel call finished." << std::endl;
		auto temp2 = data.gpu_context->GPUExecuteRelation(context, data.plan);
		data.res = data.plan->Execute();
	}

	auto result_chunk = data.res->Fetch();
	if (!result_chunk) {
		return;
	}
	while (result_chunk) {
		output.Move(*result_chunk);
		result_chunk = data.res->Fetch();
	}
	return;
}

void InitializeGPUExtension(Connection &con) {
	auto &catalog = Catalog::GetSystemCatalog(*con.context);
	// string name = catalog.GetName();
	// printf("catalog name %s\n", catalog.GetName().c_str());

	// auto &schema = catalog.GetSchema(*con.context, DEFAULT_SCHEMA);
	// string s_name = schema.name;
	// printf("schema name %s\n", schema.name.c_str());

	// auto &table = catalog.GetEntry(*con.context, CatalogType::TABLE_ENTRY, s_name, "lineitem");
	// printf("%s\n", table.name.c_str());

	auto &catalog2 = Catalog::GetCatalog(*con.context, INVALID_CATALOG);
	string name = catalog2.GetName();
	printf("catalog name %s\n", catalog2.GetName().c_str());

	auto &schema = catalog2.GetSchema(*con.context, DEFAULT_SCHEMA);
	string s_name = schema.name;
	printf("schema name %s\n", schema.name.c_str());

	TableCatalogEntry &table = catalog2.GetEntry(*con.context, CatalogType::TABLE_ENTRY, s_name, "lineitem").Cast<TableCatalogEntry>();
	printf("%s\n", table.name.c_str());

	for (auto &column_name : table.GetColumns().GetColumnNames()) {
		printf("column name %s\n", column_name.c_str());
	}

	// create the get_substrait table function that allows us to get a substrait
	// JSON from a valid SQL Query
	TableFunction gpu_processing("gpu_processing", {LogicalType::VARCHAR}, GPUProcessingFunction, GPUProcessingBind);
	gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_info(gpu_processing);
	catalog.CreateTableFunction(*con.context, gpu_processing_info);

	// create the get_substrait table function that allows us to get a substrait
	// JSON from a valid SQL Query
	TableFunction gpu_processing_substrait("gpu_processing_substrait", {LogicalType::BLOB}, GPUProcessingSubstraitFunction, GPUProcessingSubstraitBind);
	// gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_substrait_info(gpu_processing_substrait);
	catalog.CreateTableFunction(*con.context, gpu_processing_substrait_info);
}

void KomodoExtension::Load(DuckDB &db) {
	Connection con(db);
	con.BeginTransaction();

	InitializeGPUExtension(con);

	con.Commit();
}

std::string KomodoExtension::Name() {
	return "GPU	Extension";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void Komodo_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::KomodoExtension>();
}

DUCKDB_EXTENSION_API const char *Komodo_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
