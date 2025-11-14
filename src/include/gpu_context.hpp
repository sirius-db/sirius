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

#pragma once

#include "duckdb/main/client_context.hpp"
#include "gpu_executor.hpp"
#include "gpu_buffer_manager.hpp"

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

	unique_ptr<PendingQueryResult> GPUPendingStatementInternal(ClientContext &context, shared_ptr<GPUPreparedStatementData> &statement_p,
												  const PendingQueryParameters &parameters);

	unique_ptr<PendingQueryResult> GPUPendingStatementOrPreparedStatement(ClientContext &context, const string &query, shared_ptr<GPUPreparedStatementData> &statement_p,
																		  const PendingQueryParameters &parameters);

    unique_ptr<QueryResult> GPUExecuteQuery(ClientContext &context, const string &query, shared_ptr<GPUPreparedStatementData> &statement_p,
												  const PendingQueryParameters &parameters);

	unique_ptr<QueryResult> GPUExecutePendingQueryResult(PendingQueryResult &pending);

    unique_ptr<QueryResult> GPUExecuteRelation(ClientContext &context, shared_ptr<Relation> relation);

	void CheckExecutableInternal(PendingQueryResult &pending);

	unique_ptr<QueryResult> FetchResultInternal(PendingQueryResult &pending);

	void CleanupInternal(BaseQueryResult *result, bool invalidate_transaction);

	void BeginQueryInternal(const string &query);
	ErrorData EndQueryInternal(bool success, bool invalidate_transaction);

	void GPUProcessError(ErrorData &error, const string &query) const;

	template <class T>
	unique_ptr<T> GPUErrorResult(ErrorData error, const string &query = string());
};

} // namespace duckdb