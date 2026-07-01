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

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry-bridge/gen/query.rs.h"
#include "telemetry-bridge/gen/uuid.rs.h"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/data/data_repository_manager.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {
class ClientContext;
}  // namespace duckdb

namespace sirius::op {
class sirius_physical_table_scan;
}  // namespace sirius::op

namespace sirius {

struct operator_params;
class sirius_interface;

class sirius_engine {
  friend class pipeline::sirius_pipeline_build_state;
  friend class pipeline::sirius_pipeline;
  friend class pipeline::sirius_meta_pipeline;

 public:
  explicit sirius_engine(duckdb::ClientContext& context, sirius_interface& sirius_iface);
  ~sirius_engine();

  duckdb::ClientContext& context;
  sirius_interface& sirius_iface;
  duckdb::unique_ptr<op::sirius_physical_operator> sirius_owned_plan;
  duckdb::optional_ptr<op::sirius_physical_operator> sirius_physical_plan;

  //! All pipelines of the query plan
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> sirius_pipelines;
  //! The root pipelines of the query
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> sirius_root_pipelines;
  //! Storage for pipeline breaker created during pipeline splitting
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> new_pipeline_breakers;
  //! The current root pipeline index
  std::size_t root_pipeline_idx;
  //! The total amount of pipelines in the query
  std::size_t total_pipelines;
  //! Whether or not the root of the pipeline is a result collector object
  bool has_result_collector();
  //! Returns the query result - can only be used if `HasResultCollector` returns true
  duckdb::unique_ptr<duckdb::QueryResult> get_result();
  //! Initialize the sirius engine
  void initialize(duckdb::unique_ptr<op::sirius_physical_operator> physical_plan);
  //! Initialize the sirius engine internally
  void initialize_internal(op::sirius_physical_operator& physical_result_collector);
  //! Execute the sirius engine
  void execute();
  //! Reset the sirius engine
  void reset();
  //! Cancel the tasks
  void cancel_tasks();
  //! Construct the sirius specific operator
  duckdb::unique_ptr<op::sirius_physical_operator> construct_sirius_specific_operator(
    op::sirius_physical_operator* op);
  //! Construct a sirius iceberg scan operator, populating delete file lists from cache.
  duckdb::unique_ptr<op::sirius_physical_operator> construct_iceberg_scan_operator(
    op::sirius_physical_table_scan& scan_op);
  //! Pre-fetch iceberg table metadata (delete files) for all iceberg scans in the plan.
  //! Must be called from initialize() BEFORE initialize_internal() assigns operator IDs
  //! to pipeline-breaker operators (PARTITION, CONCAT, etc.).
  void prefetch_iceberg_delete_data(op::sirius_physical_operator& plan);
  //! Create a child pipeline
  duckdb::shared_ptr<pipeline::sirius_pipeline> create_child_pipeline(
    pipeline::sirius_pipeline& current, op::sirius_physical_operator& op);
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> new_scheduled;
  //! Wait for the query to finish
  void wait_for_query_finish();
  //! Mutex for thread-safe access to query finish
  std::mutex query_finish_mutex;
  //! Condition variable for thread-safe access to query finish
  std::condition_variable query_finish_cv;
  //! Whether the query has finished
  bool query_finished;

  // ---------------------------------------------------------------------------
  // Iceberg metadata cache
  //
  // Populated by prefetch_iceberg_delete_data() in initialize(), BEFORE
  // initialize_internal() runs.  Keyed by iceberg table path string.
  // ---------------------------------------------------------------------------
  std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>
    iceberg_delete_data_cache_;

 private:
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context_;
  rust::Box<quent::query::QueryHandle> query_handle_;
};

}  // namespace sirius
