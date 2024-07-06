#define DUCKDB_EXTENSION_MAIN

#include "komodo_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"

#include "substrait_extension.hpp"
#include "to_substrait.hpp"
#include "from_substrait.hpp"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

struct KomodoTableFunctionData : public TableFunctionData {
	KomodoTableFunctionData() = default;
	shared_ptr<Relation> plan;
	unique_ptr<QueryResult> res;
	unique_ptr<Connection> conn;
	string query;
	bool enable_optimizer;
	bool finished = false;
};

shared_ptr<Relation> KomodoSubstraitPlanToDuckDBRel(Connection &conn, const string &serialized, bool json = false) {
	SubstraitToDuckDB transformer_s2d(conn, serialized, json);
	return transformer_s2d.TransformPlan();
};

//This function is used to extract the query plan from the SQL query
static DuckDBToSubstrait KomodoInitPlanExtractor(ClientContext &context, KomodoTableFunctionData &data, Connection &new_conn,
                                           unique_ptr<LogicalOperator> &query_plan) {
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

	query_plan = new_conn.context->ExtractPlan(data.query);
	return DuckDBToSubstrait(context, *query_plan, false);
}

//The result of the GPUProcessingBind function is a unique pointer to a FunctionData object.
//This object is used as an argument to the GPUProcessingFunction function (data_p argument), which is called to execute the table function.
static unique_ptr<FunctionData> GPUProcessingBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_uniq<KomodoTableFunctionData>();
	result->conn = make_uniq<Connection>(*context.db);
	result->query = input.inputs[0].ToString();
	result->enable_optimizer = true;
	if (input.inputs[0].IsNull()) {
		throw BinderException("gpu_processing cannot be called with a NULL parameter");
	}
	string serialized = input.inputs[0].GetValueUnsafe<string>();

	//If serialized is a SQL statement, convert to substrait
	//This is not an ideal solution. Ideally, the SQL statement does not have to be converted to substrait, and should be converted directly to logical operators to be executed.
	if (serialized.find("SELECT") != string::npos) {
		unique_ptr<LogicalOperator> query_plan;
		auto transformer_d2s = KomodoInitPlanExtractor(context, *result, *result->conn, query_plan);
		serialized = transformer_d2s.SerializeToString();
	}

	result->plan = KomodoSubstraitPlanToDuckDBRel(*result->conn, serialized, false);
	result->plan->Print();

	for (auto &column : result->plan->Columns()) {
		return_types.emplace_back(column.Type());
		names.emplace_back(column.Name());
	}

	return std::move(result);
}

unique_ptr<PendingQueryResult> GPUPendingStatementInternal(
    ClientContextLock &lock, const string &query, unique_ptr<SQLStatement> statement,
    shared_ptr<PreparedStatementData> &prepared, const PendingQueryParameters &parameters) {

	// check if we are on AutoCommit. In this case we should start a transaction.
	if (statement) {
		// query verification is enabled
		// create a copy of the statement, and use the copy
		// this way we verify that the copy correctly copies all properties
		auto copied_statement = statement->Copy();
		switch (statement->type) {
		case StatementType::SELECT_STATEMENT: {
			// in case this is a select query, we verify the original statement
			ErrorData error;
			try {
				error = VerifyQuery(lock, query, std::move(statement));
			} catch (std::exception &ex) {
				error = ErrorData(ex);
			}
			if (error.HasError()) {
				// error in verifying query
				return ErrorResult<PendingQueryResult>(std::move(error), query);
			}
			statement = std::move(copied_statement);
			break;
		}
		default: {
			bool reparse_statement = false;
			statement = std::move(copied_statement);
			if (statement->type == StatementType::RELATION_STATEMENT) {
				reparse_statement = false;
			}
			if (reparse_statement) {
				try {
					Parser parser(GetParserOptions());
					ErrorData error;
					parser.ParseQuery(statement->ToString());
					statement = std::move(parser.statements[0]);
				} catch (const NotImplementedException &) {
					// ToString was not implemented, just use the copied statement
				}
			}
			break;
		}
		}
	}
	return PendingStatementOrPreparedStatement(lock, query, std::move(statement), prepared, parameters);
}

unique_ptr<PendingQueryResult> GPUPendingQueryInternal(ClientContextLock &lock,
                                                                   const shared_ptr<Relation> &relation,
                                                                   bool allow_stream_result) {
	// InitialCleanup(lock);

	auto relation_stmt = make_uniq<RelationStatement>(relation);
	PendingQueryParameters parameters;
	parameters.allow_stream_result = allow_stream_result;

	unique_ptr<SQLStatement> statement = std::move(relation_stmt);
	auto query = statement->query;
	shared_ptr<PreparedStatementData> prepared;

	PendingStatementOrPreparedStatementInternal(lock, query, std::move(statement), prepared, parameters);


	return PendingQueryInternal(lock, std::move(relation_stmt), parameters);


}

unique_ptr<QueryResult> GPUExecute(shared_ptr<Relation> relation, ClientContext &context) {
	//The plan is executed on the GPU

	//Currently, context_lock is not used.
	mutex context_lock;
	unique_ptr<ClientContextLock> lock = make_uniq<ClientContextLock>(context_lock);

	// auto lock = context.LockContext();
	auto &expected_columns = relation->Columns();
	auto pending = GPUPendingQueryInternal(*lock, relation, false);
	// if (!pending->success) {
	// 	return ErrorResult<MaterializedQueryResult>(pending->GetErrorObject());
	// }

	// unique_ptr<QueryResult> result;
	// result = context.ExecutePendingQueryInternal(*lock, *pending);
	// if (result->HasError()) {
	// 	return result;
	// }
	// // verify that the result types and result names of the query match the expected result types/names
	// if (result->types.size() == expected_columns.size()) {
	// 	bool mismatch = false;
	// 	for (idx_t i = 0; i < result->types.size(); i++) {
	// 		if (result->types[i] != expected_columns[i].Type() || result->names[i] != expected_columns[i].Name()) {
	// 			mismatch = true;
	// 			break;
	// 		}
	// 	}
	// 	if (!mismatch) {
	// 		// all is as expected: return the result
	// 		return result;
	// 	}
	// }

	// // result mismatch
	// string err_str = "Result mismatch in query!\nExpected the following columns: [";
	// for (idx_t i = 0; i < expected_columns.size(); i++) {
	// 	if (i > 0) {
	// 		err_str += ", ";
	// 	}
	// 	err_str += expected_columns[i].Name() + " " + expected_columns[i].Type().ToString();
	// }
	// err_str += "]\nBut result contained the following: ";
	// for (idx_t i = 0; i < result->types.size(); i++) {
	// 	err_str += i == 0 ? "[" : ", ";
	// 	err_str += result->names[i] + " " + result->types[i].ToString();
	// }
	// err_str += "]";
	// return ErrorResult<MaterializedQueryResult>(ErrorData(err_str));

}

static void GPUProcessingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (KomodoTableFunctionData &)*data_p.bind_data;
	if (data.finished) {
		return;
	}
	auto new_conn = Connection(*context.db);

	if (!data.res) {
		// data.res = data.plan->Execute();
		data.res = GPUExecute(data.plan, context);
	}

	data.finished = true;

	auto result_chunk = data.res->Fetch();
	if (!result_chunk) {
		return;
	}
	output.Move(*result_chunk);

	// auto result = new_conn.Query(data.query);
}

void InitializeKomodo(Connection &con) {
	auto &catalog = Catalog::GetSystemCatalog(*con.context);

	// create the get_substrait table function that allows us to get a substrait
	// JSON from a valid SQL Query
	TableFunction gpu_processing("gpu_processing", {LogicalType::VARCHAR}, GPUProcessingFunction, GPUProcessingBind);
	gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_info(gpu_processing);
	catalog.CreateTableFunction(*con.context, gpu_processing_info);

	// create the get_substrait table function that allows us to get a substrait
	// JSON from a valid SQL Query
	TableFunction gpu_processing_substrait("gpu_processing_substrait", {LogicalType::BLOB}, GPUProcessingFunction, GPUProcessingBind);
	// gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
	CreateTableFunctionInfo gpu_processing_substrait_info(gpu_processing_substrait);
	catalog.CreateTableFunction(*con.context, gpu_processing_substrait_info);
}

void KomodoExtension::Load(DuckDB &db) {
	Connection con(db);
	con.BeginTransaction();

	InitializeKomodo(con);

	con.Commit();
}
std::string KomodoExtension::Name() {
	return "Komodo";
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
