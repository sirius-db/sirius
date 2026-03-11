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

#include "gpu_context.hpp"

#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/planner/planner.hpp"
#include "log/logging.hpp"
#include "sirius_extension.hpp"

namespace duckdb {

void GPUBindPreparedStatementParameters(PreparedStatementData& statement,
                                        const PendingQueryParameters& parameters)
{
  case_insensitive_map_t<BoundParameterData> owned_values;
  // if (parameters.parameters) {
  // 	auto &params = *parameters.parameters;
  // 	for (auto &val : params) {
  // 		owned_values.emplace(val);
  // 	}
  // }
  statement.Bind(std::move(owned_values));
}

GPUContext::GPUContext(ClientContext& client_context) : client_context(client_context) {}

// This function is based on ClientContext::PendingStatementOrPreparedStatement
unique_ptr<PendingQueryResult> GPUContext::GPUPendingStatementOrPreparedStatement(
  ClientContext& context,
  const string& query,
  shared_ptr<GPUPreparedStatementData>& statement_p,
  const PendingQueryParameters& parameters)
{
  BeginQueryInternal(query);

  bool invalidate_query = true;
  unique_ptr<PendingQueryResult> pending =
    GPUPendingStatementInternal(context, statement_p, parameters);

  if (pending->HasError()) {
    // query failed: abort now
    // throw InvalidInputException("Error in GPUPendingStatementOrPreparedStatement");
    // EndQueryInternal(false, invalidate_query);
    return pending;
  }
  D_ASSERT(gpu_active_query->IsOpenResult(*pending));
  return pending;
}

void GPUContext::GPUProcessError(ErrorData& error, const string& query) const
{
  error.FinalizeError();
  if (client_context.config.errors_as_json) {
    error.ConvertErrorToJSON();
  } else {
    error.AddErrorLocation(query);
  }
}

template <class T>
unique_ptr<T> GPUContext::GPUErrorResult(ErrorData error, const string& query)
{
  GPUProcessError(error, query);
  return make_uniq<T>(std::move(error));
}

// This function is based on ClientContext::PendingPreparedStatementInternal
unique_ptr<PendingQueryResult> GPUContext::GPUPendingStatementInternal(
  ClientContext& context,
  shared_ptr<GPUPreparedStatementData>& statement_p,
  const PendingQueryParameters& parameters)
{
  D_ASSERT(gpu_active_query);
  auto& statement = *(statement_p->prepared);

  GPUBindPreparedStatementParameters(statement, parameters);

  unique_ptr<GPUExecutor> temp = make_uniq<GPUExecutor>(context, *this);
  auto prop                    = temp->context.GetClientProperties();
  // SIRIUS_LOG_DEBUG("Properties: {}", prop.time_zone);
  gpu_active_query->gpu_executor = std::move(temp);
  auto& gpu_executor             = GetGPUExecutor();
  // auto stream_result = parameters.allow_stream_result &&
  // statement.properties.allow_stream_result;
  bool stream_result = false;

  unique_ptr<GPUPhysicalResultCollector> gpu_collector =
    make_uniq_base<GPUPhysicalResultCollector, GPUPhysicalMaterializedCollector>(*statement_p);
  if (gpu_collector->type != PhysicalOperatorType::RESULT_COLLECTOR) {
    // throw InvalidInputException("Error in GPUPendingStatementInternal");
    return GPUErrorResult<PendingQueryResult>(ErrorData("Error in GPUPendingStatementInternal"));
  }
  D_ASSERT(gpu_collector->type == PhysicalOperatorType::RESULT_COLLECTOR);
  auto types = gpu_collector->GetTypes();
  D_ASSERT(types == statement.types);
  gpu_executor.Initialize(std::move(gpu_collector));
  // SIRIUS_LOG_DEBUG("type {}", gpu_executor.gpu_physical_plan.get()->type);

  D_ASSERT(!gpu_active_query->HasOpenResult());

  auto pending_result = make_uniq<PendingQueryResult>(
    context.shared_from_this(), *(statement_p->prepared), std::move(types), stream_result);
  gpu_active_query->gpu_prepared = std::move(statement_p);
  gpu_active_query->SetOpenResult(*pending_result);
  return pending_result;
}

GPUExecutor& GPUContext::GetGPUExecutor()
{
  D_ASSERT(gpu_active_query);
  D_ASSERT(gpu_active_query->gpu_executor);
  return *gpu_active_query->gpu_executor;
}

void GPUContext::CheckExecutableInternal(PendingQueryResult& pending)
{
  // bool invalidated = HasError() || !(client_context);
  D_ASSERT(gpu_active_query->IsOpenResult(pending));
  bool invalidated = pending.HasError();
  if (!invalidated) {
    D_ASSERT(gpu_active_query);
    invalidated = !gpu_active_query->IsOpenResult(pending);
  }
  if (invalidated) {
    if (pending.HasError()) {
      throw InvalidInputException("Attempting to execute an unsuccessful pending query result\n");
    }
    throw InvalidInputException("Attempting to execute a closed pending query result");
  }
}

// This function is based on PendingQueryResult::ExecuteInternal
unique_ptr<QueryResult> GPUContext::GPUExecutePendingQueryResult(PendingQueryResult& pending)
{
  // auto lock = pending.LockContext();
  D_ASSERT(gpu_active_query->IsOpenResult(pending));
  CheckExecutableInternal(pending);
  auto& gpu_executor = GetGPUExecutor();
  try {
    gpu_executor.Execute();
  } catch (std::exception& e) {
    ErrorData error(e);
    SIRIUS_LOG_ERROR("Error in GPUExecutePendingQueryResult: {}", error.RawMessage());
    gpu_executor.gpuBufferManager->ResetBuffer();
    return GPUErrorResult<MaterializedQueryResult>(error);
  }
  if (pending.HasError()) {
    // throw InvalidInputException("Error in GPUExecutePendingQueryResult");
    ErrorData error = pending.GetErrorObject();
    return make_uniq<MaterializedQueryResult>(error);
  }
  SIRIUS_LOG_DEBUG("Done ExecutePendingQueryResult");
  auto result = FetchResultInternal(pending);
  // context.reset();
  return result;
}

// This function is based on ClientContext::Query
unique_ptr<QueryResult> GPUContext::GPUExecuteQuery(
  ClientContext& context,
  const string& query,
  shared_ptr<GPUPreparedStatementData>& statement_p,
  const PendingQueryParameters& parameters)
{
  auto pending_query =
    GPUPendingStatementOrPreparedStatement(context, query, statement_p, parameters);
  D_ASSERT(gpu_active_query->IsOpenResult(*pending_query));
  unique_ptr<QueryResult> current_result;
  if (pending_query->HasError()) {
    // throw InvalidInputException("Error in GPUExecuteQuery");
    current_result = GPUErrorResult<MaterializedQueryResult>(pending_query->GetErrorObject());
  } else {
    current_result = GPUExecutePendingQueryResult(*pending_query);
  }
  SIRIUS_LOG_DEBUG("Done GPUExecuteQuery");
  return current_result;
}

void GPUContext::BeginQueryInternal(const string& query)
{
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

unique_ptr<QueryResult> GPUContext::FetchResultInternal(PendingQueryResult& pending)
{
  D_ASSERT(gpu_active_query);
  D_ASSERT(gpu_active_query->IsOpenResult(pending));
  D_ASSERT(gpu_active_query->gpu_prepared->prepared);
  auto& gpu_executor = GetGPUExecutor();
  auto& prepared     = *gpu_active_query->gpu_prepared->prepared;
  // bool create_stream_result = prepared.properties.allow_stream_result &&
  // pending->allow_stream_result;
  unique_ptr<QueryResult> result;
  D_ASSERT(gpu_executor.HasResultCollector());
  // we have a result collector - fetch the result directly from the result collector
  // SIRIUS_LOG_DEBUG("Getting result");
  result = gpu_executor.GetResult();
  // SIRIUS_LOG_DEBUG("Fetching result");
  // if (!create_stream_result) {
  CleanupInternal(result.get(), false);
  // } else {
  // 	active_query->SetOpenResult(*result);
  // }
  return result;
}

void GPUContext::CleanupInternal(BaseQueryResult* result, bool invalidate_transaction)
{
  if (!gpu_active_query) {
    // no query currently active
    return;
  }
  // SIRIUS_LOG_DEBUG("Cleaning up");
  if (gpu_active_query->gpu_executor) { gpu_active_query->gpu_executor->CancelTasks(); }
  gpu_active_query->progress_bar.reset();

  auto error = EndQueryInternal(result ? !result->HasError() : false, invalidate_transaction);
  if (result && !result->HasError()) { result->SetError(error); }
  D_ASSERT(!gpu_active_query);
}

ErrorData GPUContext::EndQueryInternal(bool success, bool invalidate_transaction)
{
  // client_data->profiler->EndQuery();

  if (gpu_active_query->gpu_executor) {
    gpu_active_query->gpu_executor->CancelTasks();
    gpu_active_query->gpu_executor->gpuBufferManager->ResetBuffer();
  }
  // Notify any registered state of query end
  // for (auto const &s : registered_state) {
  // 	s.second->QueryEnd(*this);
  // }
  gpu_active_query->progress_bar.reset();

  D_ASSERT(gpu_active_query.get());
  gpu_active_query.reset();
  // query_progress.Initialize();
  ErrorData error;
  // SIRIUS_LOG_DEBUG("Ending query");
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

// This function is based on ClientContext::PendingStatementOrPreparedStatement
// unique_ptr<PendingQueryResult> GPUContext::SiriusPendingStatementOrPreparedStatement(
//   ClientContext& context,
//   const string& query,
//   shared_ptr<SiriusPreparedStatementData>& statement_p,
//   const PendingQueryParameters& parameters)
// {
//   BeginQueryInternal(query);

//   bool invalidate_query = true;
//   unique_ptr<PendingQueryResult> pending =
//     SiriusPendingStatementInternal(context, statement_p, parameters);

//   if (pending->HasError()) {
//     // query failed: abort now
//     // throw InvalidInputException("Error in SiriusPendingStatementOrPreparedStatement");
//     // EndQueryInternal(false, invalidate_query);
//     return pending;
//   }
//   D_ASSERT(gpu_active_query->IsOpenResult(*pending));
//   return pending;
// };

// This function is based on ClientContext::PendingPreparedStatementInternal
// unique_ptr<PendingQueryResult> GPUContext::SiriusPendingStatementInternal(
//   ClientContext& context,
//   shared_ptr<SiriusPreparedStatementData>& statement_p,
//   const PendingQueryParameters& parameters)
// {
//   D_ASSERT(gpu_active_query);
//   auto& statement = *(statement_p->prepared);

//   GPUBindPreparedStatementParameters(statement, parameters);

//   unique_ptr<::sirius::sirius_engine> temp = make_uniq<::sirius::sirius_engine>(context, *this);
//   auto prop                                = temp->context.GetClientProperties();
//   gpu_active_query->engine                 = std::move(temp);
//   auto& engine                             = GetSiriusEngine();
//   bool stream_result                       = false;

//   unique_ptr<::sirius::op::sirius_physical_result_collector> sirius_collector =
//     make_uniq_base<::sirius::op::sirius_physical_result_collector,
//                    ::sirius::op::sirius_physical_materialized_collector>(*statement_p,
//                                                                          client_context);
//   if (sirius_collector->type != ::sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
//     return GPUErrorResult<PendingQueryResult>(ErrorData("Error in
//     SiriusPendingStatementInternal"));
//   }
//   D_ASSERT(sirius_collector->type == ::sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR);
//   auto types = sirius_collector->get_types();
//   D_ASSERT(types == statement.types);
//   engine.initialize(std::move(sirius_collector));

//   D_ASSERT(!gpu_active_query->HasOpenResult());

//   auto pending_result = make_uniq<PendingQueryResult>(
//     context.shared_from_this(), *(statement_p->prepared), std::move(types), stream_result);
//   gpu_active_query->sirius_prepared = std::move(statement_p);
//   gpu_active_query->SetOpenResult(*pending_result);
//   return pending_result;
// };

// // This function is based on PendingQueryResult::ExecuteInternal
// unique_ptr<QueryResult> GPUContext::SiriusExecutePendingQueryResult(PendingQueryResult& pending)
// {
//   D_ASSERT(gpu_active_query->IsOpenResult(pending));
//   CheckExecutableInternal(pending);
//   auto& engine = GetSiriusEngine();
//   try {
//     engine.execute();
//   } catch (std::exception& e) {
//     ErrorData error(e);
//     SIRIUS_LOG_ERROR("Error in SiriusExecutePendingQueryResult: {}", error.RawMessage());
//     return GPUErrorResult<MaterializedQueryResult>(error);
//   }
//   if (pending.HasError()) {
//     ErrorData error = pending.GetErrorObject();
//     return make_uniq<MaterializedQueryResult>(error);
//   }
//   SIRIUS_LOG_DEBUG("Done ExecutePendingQueryResult");
//   auto result = FetchResultInternal(pending);
//   return result;
// }

// // This function is based on ClientContext::Query
// unique_ptr<QueryResult> GPUContext::SiriusExecuteQuery(
//   ClientContext& context,
//   const string& query,
//   shared_ptr<SiriusPreparedStatementData>& statement_p,
//   const PendingQueryParameters& parameters)
// {
//   auto pending_query =
//     SiriusPendingStatementOrPreparedStatement(context, query, statement_p, parameters);
//   D_ASSERT(gpu_active_query->IsOpenResult(*pending_query));
//   unique_ptr<QueryResult> current_result;
//   if (pending_query->HasError()) {
//     current_result = GPUErrorResult<MaterializedQueryResult>(pending_query->GetErrorObject());
//   } else {
//     current_result = SiriusExecutePendingQueryResult(*pending_query);
//   }
//   SIRIUS_LOG_DEBUG("Done SiriusExecuteQuery");
//   return current_result;
// };

// sirius::sirius_engine& GPUContext::GetSiriusEngine()
// {
//   D_ASSERT(gpu_active_query);
//   D_ASSERT(gpu_active_query->engine);
//   return *gpu_active_query->engine;
// }

}  // namespace duckdb
