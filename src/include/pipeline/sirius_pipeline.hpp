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

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "op/sirius_physical_operator.hpp"
// #include "duckdb/parallel/executor_task.hpp"
#include "duckdb/parallel/pipeline.hpp"

namespace sirius {

class sirius_engine;

namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {

class sirius_pipeline;
class sirius_meta_pipeline;

class sirius_pipeline_build_state {
 public:
  //! How much to increment batch indexes when multiple pipelines share the same source
  constexpr static duckdb::idx_t BATCH_INCREMENT = 10000000000000;

 public:
  //! Duplicate eliminated join scan dependencies
  duckdb::reference_map_t<const op::sirius_physical_operator, duckdb::reference<sirius_pipeline>>
    delim_join_dependencies;
  //! Materialized CTE scan dependencies
  duckdb::reference_map_t<const op::sirius_physical_operator, duckdb::reference<sirius_pipeline>>
    cte_dependencies;

 public:
  void set_pipeline_source(sirius_pipeline& pipeline, op::sirius_physical_operator& op);
  void set_pipeline_sink(sirius_pipeline& pipeline,
                         duckdb::optional_ptr<op::sirius_physical_operator> op,
                         duckdb::idx_t sink_pipeline_count);
  void set_pipeline_operators(
    sirius_pipeline& pipeline,
    duckdb::vector<duckdb::reference<op::sirius_physical_operator>> operators);
  void add_pipeline_operator(sirius_pipeline& pipeline, op::sirius_physical_operator& op);
  duckdb::shared_ptr<sirius_pipeline> create_child_pipeline(sirius_engine& engine,
                                                            sirius_pipeline& pipeline,
                                                            op::sirius_physical_operator& op);

  duckdb::optional_ptr<op::sirius_physical_operator> get_pipeline_source(sirius_pipeline& pipeline);
  duckdb::optional_ptr<op::sirius_physical_operator> get_pipeline_sink(sirius_pipeline& pipeline);
  duckdb::vector<duckdb::reference<op::sirius_physical_operator>> get_pipeline_operators(
    sirius_pipeline& pipeline);
};

//! The sirius_pipeline class represents an execution pipeline starting at a
class sirius_pipeline : public duckdb::enable_shared_from_this<sirius_pipeline> {
  friend class ::sirius::sirius_engine;
  friend class sirius_pipeline_build_state;
  friend class sirius_meta_pipeline;

 public:
  explicit sirius_pipeline(sirius_engine& engine);
  virtual ~sirius_pipeline() = default;

 public:
  duckdb::ClientContext& get_client_context();

  void add_dependency(duckdb::shared_ptr<sirius_pipeline>& pipeline);

  void is_ready();
  void reset();
  void reset_sink();
  void reset_source(bool force);
  void clear_source();
  void schedule(duckdb::shared_ptr<duckdb::Event>& event);

  // std::string to_string() const;
  // void print() const;
  // void print_dependencies() const;

  //! Returns query progress
  // bool get_progress(double &current_percentage, duckdb::idx_t &estimated_cardinality);

  //! Returns a list of all operators (including source and sink) involved in this pipeline
  // duckdb::vector<duckdb::reference<op::sirius_physical_operator>> get_all_operators();

  // duckdb::vector<duckdb::const_reference<op::sirius_physical_operator>> get_all_operators()
  // const;

  //! Returns a list of all inner operators (excluding source and sink) involved in this pipeline
  duckdb::vector<duckdb::reference<op::sirius_physical_operator>> get_operators();

  duckdb::optional_ptr<op::sirius_physical_operator> get_sink() { return sink; }

  duckdb::optional_ptr<op::sirius_physical_operator> get_sink() const noexcept { return sink; }

  duckdb::optional_ptr<op::sirius_physical_operator> get_source() { return source; }

  duckdb::optional_ptr<op::sirius_physical_operator> get_source() const noexcept { return source; }

  //! Set the pipeline ID
  void set_pipeline_id(size_t id) { pipeline_id = id; }
  //! Get the pipeline ID
  size_t get_pipeline_id() const { return pipeline_id; }
  //! Returns the parent pipelines (pipelines that depend on this pipeline)
  std::vector<sirius_pipeline*> get_parents();

  //! Returns whether any of the operators in the pipeline care about preserving order
  bool is_order_dependent() const;

  //! Registers a new batch index for a pipeline executor - returns the current minimum batch index
  duckdb::idx_t register_new_batch_index();

  //! Updates the batch index of a pipeline (and returns the new minimum batch index)
  duckdb::idx_t update_batch_index(duckdb::idx_t old_index, duckdb::idx_t new_index);

  //! The dependencies of this pipeline
  // duckdb::vector<std::weak_ptr<sirius_pipeline>> dependencies;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> dependencies;

  //! Updates the pipeline status
  void update_pipeline_status();
  //! Checks if the pipeline has been finished
  virtual bool is_pipeline_finished() const;

  void mark_task_created();
  void mark_task_completed();

 private:
  //! Whether or not the pipeline has been readied
  bool ready;
  //! Whether or not the pipeline has been initialized
  std::atomic<bool> initialized;
  //! The source of this pipeline
  duckdb::optional_ptr<op::sirius_physical_operator> source;
  //! The chain of intermediate operators
  duckdb::vector<duckdb::reference<op::sirius_physical_operator>> operators;
  //! The sink (i.e. destination) for data; this is e.g. a hash table to-be-built
  duckdb::optional_ptr<op::sirius_physical_operator> sink;

  //! The global source state
  duckdb::unique_ptr<duckdb::GlobalSourceState> source_state;
  //! The parent pipelines (i.e. pipelines that are dependent on this pipeline to finish)
  duckdb::vector<duckdb::weak_ptr<sirius_pipeline>> parents;

  //! The base batch index of this pipeline
  duckdb::idx_t base_batch_index = 0;
  //! Lock for accessing the set of batch indexes
  std::mutex batch_lock;
  //! The set of batch indexes that are currently being processed
  //! Despite batch indexes being unique - this is a multiset
  //! The reason is that when we start a new pipeline we insert the current minimum batch index as a
  //! placeholder Which leads to duplicate entries in the set of active batch indexes
  std::multiset<duckdb::idx_t> batch_indexes;

  void schedule_sequential_task(duckdb::shared_ptr<duckdb::Event>& event);
  bool launch_scan_tasks(duckdb::shared_ptr<duckdb::Event>& event, duckdb::idx_t max_threads);

  bool schedule_parallel(duckdb::shared_ptr<duckdb::Event>& event);

  //! The unique ID of this pipeline (assigned based on new_scheduled order)
  size_t pipeline_id = 0;
  sirius_engine& engine;

  //! Whether the pipeline has been finished
  std::atomic<bool> pipeline_finished = false;

  std::atomic<std::size_t> tasks_created   = 0;
  std::atomic<std::size_t> tasks_completed = 0;
};

}  // namespace pipeline
}  // namespace sirius
