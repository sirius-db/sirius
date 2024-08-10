#pragma once

#include "duckdb/main/client_context.hpp"
#include "gpu_executor.hpp"

namespace duckdb {

class GPUPreparedStatementData {
public:
	GPUPreparedStatementData(shared_ptr<PreparedStatementData> _prepared, unique_ptr<GPUPhysicalOperator> _gpu_physical_plan) 
	: prepared(_prepared), gpu_physical_plan(move(_gpu_physical_plan)) {
	}
	unique_ptr<GPUPhysicalOperator> gpu_physical_plan;
	shared_ptr<PreparedStatementData> prepared;
};

struct GPUActiveQueryContext {
public:
	//! The query that is currently being executed
	string query;
	//! Prepared statement data
	shared_ptr<GPUPreparedStatementData> gpu_prepared;
	//! The query executor
	unique_ptr<GPUExecutor> gpu_executor;
	//! The progress bar
	unique_ptr<ProgressBar> progress_bar;

public:
	void SetOpenResult(BaseQueryResult &result) {
		open_result = &result;
	}
	bool IsOpenResult(BaseQueryResult &result) {
		return open_result == &result;
	}
	bool HasOpenResult() const {
		return open_result != nullptr;
	}

private:
	//! The currently open result
	BaseQueryResult *open_result = nullptr;
};

class GPUContext {

public:
	GPUContext(ClientContext& client_context);
	// ~GPUContext();

	ClientContext &client_context;

	//! The currently active query context
	unique_ptr<GPUActiveQueryContext> gpu_active_query;
	//! The current query progress
	QueryProgress query_progress;

    GPUExecutor &GetGPUExecutor();

	unique_ptr<PendingQueryResult> GPUPendingQuery(ClientContext &context, shared_ptr<GPUPreparedStatementData> &statement_p,
												  const PendingQueryParameters &parameters);

    unique_ptr<QueryResult> GPUExecuteQuery(ClientContext &context, const string &query, shared_ptr<GPUPreparedStatementData> &statement_p,
												  const PendingQueryParameters &parameters);

    unique_ptr<QueryResult> GPUExecuteRelation(ClientContext &context, shared_ptr<Relation> relation);

	void CheckExecutableInternal(PendingQueryResult &pending);

	unique_ptr<QueryResult> FetchResultInternal(PendingQueryResult &pending);

	void CleanupInternal(BaseQueryResult *result, bool invalidate_transaction);

	void BeginQueryInternal(const string &query);
	ErrorData EndQueryInternal(bool success, bool invalidate_transaction);

protected:
	//! Whether or not execution was successful
	bool success;
	//! The error (in case execution was not successful)
	ErrorData error;

	bool HasError() const;
};

} // namespace duckdb