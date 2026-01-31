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
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_repository_manager.hpp>
namespace duckdb {
class ClientContext;
class GPUContext;
}  // namespace duckdb

namespace sirius {

class sirius_engine {
  friend class pipeline::sirius_pipeline_build_state;
  friend class pipeline::sirius_pipeline;
  friend class pipeline::sirius_meta_pipeline;

 public:
  explicit sirius_engine(duckdb::ClientContext& context, duckdb::GPUContext& gpu_context)
    : context(context), gpu_context(gpu_context) {};
  ~sirius_engine() {}

  duckdb::ClientContext& context;
  duckdb::GPUContext& gpu_context;
  duckdb::unique_ptr<op::sirius_physical_operator> sirius_owned_plan;
  duckdb::optional_ptr<op::sirius_physical_operator> sirius_physical_plan;

  //! All pipelines of the query plan
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> sirius_pipelines;
  //! The root pipelines of the query
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> sirius_root_pipelines;
  //! The scheduled pipelines
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> sirius_scheduled;
  //! Storage for pipeline breaker created during pipeline splitting
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> new_pipeline_breakers;
  //! Storage for concatenated operators during pipeline splitting
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> concat_ops;
  //! The current root pipeline index
  duckdb::idx_t root_pipeline_idx;
  //! The total amount of pipelines in the query
  duckdb::idx_t total_pipelines;
  //! Inserting repository
  void insert_repository(std::string_view port_id,
                         duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
                         duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline);
  void insert_repository(std::string_view port_id,
                         op::sirius_physical_operator* cur_op,
                         duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
                         duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline);

  //! Whether or not the root of the pipeline is a result collector object
  bool has_result_collector();
  //! Returns the query result - can only be used if `HasResultCollector` returns true
  duckdb::unique_ptr<duckdb::QueryResult> get_result();

  void initialize(duckdb::unique_ptr<op::sirius_physical_operator> physical_plan);
  void initialize_internal(op::sirius_physical_operator& physical_result_collector);
  void execute();
  void reset();
  void cancel_tasks();
  duckdb::unique_ptr<op::sirius_physical_operator> construct_sirius_specific_operator(
    op::sirius_physical_operator* op);
  duckdb::shared_ptr<pipeline::sirius_pipeline> create_child_pipeline(
    pipeline::sirius_pipeline& current, op::sirius_physical_operator& op);
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> new_scheduled;

  //! Convert the DuckDB physical plan to a GPU physical plan
};

}  // namespace sirius
