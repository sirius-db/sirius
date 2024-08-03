#include "gpu_context.hpp"
#include "komodo_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/statement/relation_statement.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"

namespace duckdb {

void GPUProcessError(ClientContext &context, ErrorData &error, const string &query) {
    if (context.config.errors_as_json) {
        error.ConvertErrorToJSON();
    } else if (!query.empty()) {
        error.AddErrorLocation(query);
    }
}

template <class T>
unique_ptr<T> GPUErrorResult(ClientContext &context, ErrorData error, const string &query = string()) {
	GPUProcessError(context, error, query);
	return make_uniq<T>(std::move(error));
}


unique_ptr<LogicalOperator> ExtractPlanFromRelation(ClientContext &context, shared_ptr<Relation> relation) {
	auto relation_stmt = make_uniq<RelationStatement>(relation);
	unique_ptr<SQLStatement> statements = std::move(relation_stmt);

	unique_ptr<LogicalOperator> plan;
	Planner planner(context);
	planner.CreatePlan(std::move(statements));
	D_ASSERT(planner.plan);

	plan = std::move(planner.plan);

	Optimizer optimizer(*planner.binder, context);
	plan = optimizer.Optimize(std::move(plan));

	ColumnBindingResolver resolver;
	resolver.Verify(*plan);
	resolver.VisitOperator(*plan);

	plan->ResolveOperatorTypes();

	return plan;
}

void GPUBindPreparedStatementParameters(PreparedStatementData &statement, const PendingQueryParameters &parameters) {
	case_insensitive_map_t<Value> owned_values;
	// if (parameters.parameters) {
	// 	auto &params = *parameters.parameters;
	// 	for (auto &val : params) {
	// 		owned_values.emplace(val);
	// 	}
	// }
	statement.Bind(std::move(owned_values));
}

GPUContext::GPUContext(ClientContext& client_context) : client_context(client_context) {
};

unique_ptr<PendingQueryResult> 
GPUContext::GPUPendingQuery(ClientContext &context, shared_ptr<PreparedStatementData> &statement_p,
												  const PendingQueryParameters &parameters) {
	D_ASSERT(gpu_active_query);
	auto &statement = *statement_p;

	GPUBindPreparedStatementParameters(statement, parameters);

	unique_ptr<GPUExecutor> temp = make_uniq<GPUExecutor>(context, *this);
	gpu_active_query->gpu_executor = std::move(temp);
	auto &gpu_executor = GetGPUExecutor();
	// auto stream_result = parameters.allow_stream_result && statement.properties.allow_stream_result;
	bool stream_result = false;

	get_result_collector_t get_method = PhysicalResultCollector::GetResultCollector;
	auto &client_config = ClientConfig::GetConfig(context);
	if (!stream_result && client_config.result_collector) {
		get_method = client_config.result_collector;
	}
	statement.is_streaming = stream_result;
	unique_ptr<PhysicalResultCollector> collector = get_method(context, statement);
	D_ASSERT(collector->type == PhysicalOperatorType::RESULT_COLLECTOR);
	auto types = collector->GetTypes();
	D_ASSERT(types == statement.types);
	gpu_executor.Initialize(std::move(collector));

	D_ASSERT(!gpu_active_query->HasOpenResult());

	auto pending_result =
	    make_uniq<PendingQueryResult>(context.shared_from_this(), *statement_p, std::move(types), stream_result);
	gpu_active_query->prepared = std::move(statement_p);
	gpu_active_query->SetOpenResult(*pending_result);
	return pending_result;
};

GPUExecutor& GPUContext::GetGPUExecutor() {
	D_ASSERT(gpu_active_query);
	D_ASSERT(gpu_active_query->gpu_executor);
	return *gpu_active_query->gpu_executor;
}

bool 
GPUContext::HasError() const {
	D_ASSERT(error.HasError() == !success);
	return !success;
}

void 
GPUContext::CheckExecutableInternal(PendingQueryResult &result) {
	// bool invalidated = HasError() || !(client_context);
	bool invalidated = HasError();
	if (!invalidated) {
		if (!gpu_active_query) {
			invalidated = false;
		}
		invalidated = gpu_active_query->IsOpenResult(result);
	}
	if (invalidated) {
		if (HasError()) {
			throw InvalidInputException(
			    "Attempting to execute an unsuccessful or closed pending query result\n");
		}
		throw InvalidInputException("Attempting to execute an unsuccessful or closed pending query result");
	}
}

unique_ptr<QueryResult> 
GPUContext::GPUExecuteQuery(ClientContext &context, const string &query, shared_ptr<PreparedStatementData> &statement_p,
												  const PendingQueryParameters &parameters) {

	BeginQueryInternal(query);
	auto pending = GPUPendingQuery(context, statement_p, parameters);

	unique_ptr<QueryResult> result;
// 	auto result = pending->Execute();
// 	if (result->HasError()) {
		// return result;
// 	}
// 	// verify that the result types and result names of the query match the expected result types/names
// 	if (result->types.size() == expected_columns.size()) {
// 		bool mismatch = false;
// 		for (idx_t i = 0; i < result->types.size(); i++) {
// 			if (result->types[i] != expected_columns[i].Type() || result->names[i] != expected_columns[i].Name()) {
// 				mismatch = true;
// 				break;
// 			}
// 		}
// 		if (!mismatch) {
// 			// all is as expected: return the result
			return std::move(result);

	CheckExecutableInternal(*pending);
	// Busy wait while execution is not finished
	// Execution happen here
	// if (allow_stream_result) {
	// 	while (!IsFinishedOrBlocked(ExecuteTaskInternal(lock))) {
	// 	}
	// } else {
	// 	while (!IsFinished(ExecuteTaskInternal(lock))) {
	// 	}
	// }
	if (HasError()) {
		return make_uniq<MaterializedQueryResult>(error);
	}
	result = FetchResultInternal(*pending);
	// context.reset();
	return result;

};

void GPUContext::BeginQueryInternal(const string &query) {
	// check if we are on AutoCommit. In this case we should start a transaction
	D_ASSERT(!gpu_active_query);
	// auto &db_inst = DatabaseInstance::GetDatabase(*this);
	// if (ValidChecker::IsInvalidated(db_inst)) {
	// 	throw ErrorManager::InvalidatedDatabase(*this, ValidChecker::InvalidatedMessage(db_inst));
	// }
	gpu_active_query = make_uniq<GPUActiveQueryContext>();
	// if (transaction.IsAutoCommit()) {
	// 	transaction.BeginTransaction();
	// }
	// transaction.SetActiveQuery(db->GetDatabaseManager().GetNewQueryNumber());
	// LogQueryInternal(lock, query);
	gpu_active_query->query = query;

	// query_progress.Initialize();
	// // Notify any registered state of query begin
	// for (auto const &s : registered_state) {
	// 	s.second->QueryBegin(*this);
	// }
}

unique_ptr<QueryResult> 
GPUContext::FetchResultInternal(PendingQueryResult &pending) {
	D_ASSERT(gpu_active_query);
	D_ASSERT(gpu_active_query->IsOpenResult(pending));
	D_ASSERT(gpu_active_query->prepared);
	auto &gpu_executor = GetGPUExecutor();
	auto &prepared = *gpu_active_query->prepared;
	// bool create_stream_result = prepared.properties.allow_stream_result && pending->allow_stream_result;
	unique_ptr<QueryResult> result;
	D_ASSERT(gpu_executor.HasResultCollector());
	// we have a result collector - fetch the result directly from the result collector
	result = gpu_executor.GetResult();
	// if (!create_stream_result) {
		CleanupInternal(result.get(), false);
	// } else {
	// 	active_query->SetOpenResult(*result);
	// }
	return result;
}

void 
GPUContext::CleanupInternal(BaseQueryResult *result, bool invalidate_transaction) {
	if (!gpu_active_query) {
		// no query currently active
		return;
	}
	if (gpu_active_query->gpu_executor) {
		gpu_active_query->gpu_executor->CancelTasks();
	}
	gpu_active_query->progress_bar.reset();

	// Relaunch the threads if a SET THREADS command was issued
	// auto &scheduler = TaskScheduler::GetScheduler(*this);
	// scheduler.RelaunchThreads();

	auto error = EndQueryInternal(result ? !result->HasError() : false, invalidate_transaction);
	if (result && !result->HasError()) {
		// if an error occurred while committing report it in the result
		result->SetError(error);
	}
	D_ASSERT(!gpu_active_query);
}

ErrorData 
GPUContext::EndQueryInternal(bool success, bool invalidate_transaction) {
	// client_data->profiler->EndQuery();

	if (gpu_active_query->gpu_executor) {
		gpu_active_query->gpu_executor->CancelTasks();
	}
	// Notify any registered state of query end
	// for (auto const &s : registered_state) {
	// 	s.second->QueryEnd(*this);
	// }
	// active_query->progress_bar.reset();

	D_ASSERT(gpu_active_query.get());
	gpu_active_query.reset();
	// query_progress.Initialize();
	ErrorData error;
	// try {
	// 	if (transaction.HasActiveTransaction()) {
	// 		transaction.ResetActiveQuery();
	// 		if (transaction.IsAutoCommit()) {
	// 			if (success) {
	// 				transaction.Commit();
	// 			} else {
	// 				transaction.Rollback();
	// 			}
	// 		} else if (invalidate_transaction) {
	// 			D_ASSERT(!success);
	// 			ValidChecker::Invalidate(ActiveTransaction(), "Failed to commit");
	// 		}
	// 	}
	// } catch (std::exception &ex) {
	// 	error = ErrorData(ex);
	// 	if (Exception::InvalidatesDatabase(error.Type())) {
	// 		auto &db_inst = DatabaseInstance::GetDatabase(*this);
	// 		ValidChecker::Invalidate(db_inst, error.RawMessage());
	// 	}
	// } catch (...) { // LCOV_EXCL_START
	// 	error = ErrorData("Unhandled exception!");
	// } // LCOV_EXCL_STOP
	return error;
}

unique_ptr<QueryResult> 
GPUContext::GPUExecuteRelation(ClientContext &context, shared_ptr<Relation> relation) {

	printf("Executing relation\n");
	auto logical_plan = ExtractPlanFromRelation(context, relation);
	// for (auto &column: logical_plan->ColumnBindingsToString()) {
	// 	printf("Logical Column: %s\n", column.c_str());
	// }
	// printf("Printing logical plan\n");
	// logical_plan->Print();
	// printf("Done printing logical plan\n");
	// now convert logical query plan into a physical query plan

	PhysicalPlanGenerator physical_planner(context);
	auto physical_plan = physical_planner.CreatePlan(std::move(logical_plan));
	// printf("Printing physical plan\n");
	// physical_plan->Print();
	// printf("Done printing physical plan\n");

	// printf("Mapping DuckDB plan to GPU plan\n");
	// GPUProcessingExecute(context, *physical_plan, {}, {});

// 	// auto lock = LockContext();
// 	auto &expected_columns = relation->Columns();
// 	auto pending = new_conn.context->PendingQuery(relation, false);
// 	if (!pending->HasError()) {
// 		return GPUErrorResult<MaterializedQueryResult>(context, pending->GetErrorObject());
// 	}

	unique_ptr<QueryResult> result;
// 	auto result = pending->Execute();
// 	if (result->HasError()) {
		// return result;
// 	}
// 	// verify that the result types and result names of the query match the expected result types/names
// 	if (result->types.size() == expected_columns.size()) {
// 		bool mismatch = false;
// 		for (idx_t i = 0; i < result->types.size(); i++) {
// 			if (result->types[i] != expected_columns[i].Type() || result->names[i] != expected_columns[i].Name()) {
// 				mismatch = true;
// 				break;
// 			}
// 		}
// 		if (!mismatch) {
// 			// all is as expected: return the result
			return result;
// 		}
// 	}
// 	// result mismatch
// 	string err_str = "Result mismatch in query!\nExpected the following columns: [";
// 	for (idx_t i = 0; i < expected_columns.size(); i++) {
// 		if (i > 0) {
// 			err_str += ", ";
// 		}
// 		err_str += expected_columns[i].Name() + " " + expected_columns[i].Type().ToString();
// 	}
// 	err_str += "]\nBut result contained the following: ";
// 	for (idx_t i = 0; i < result->types.size(); i++) {
// 		err_str += i == 0 ? "[" : ", ";
// 		err_str += result->names[i] + " " + result->types[i].ToString();
// 	}
// 	err_str += "]";
// 	return GPUErrorResult<MaterializedQueryResult>(context, ErrorData(err_str));

}

}; // namespace duckdb