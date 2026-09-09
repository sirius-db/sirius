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

#include "common/optional_ptr.hpp"
#include "common/reference_map.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/pipeline_memory_history.hpp"
#include "telemetry-bridge/gen/uuid.rs.h"

#include <nvtx3/nvtx3.hpp>

#include <mutex>
#include <utility>
#include <vector>

namespace sirius {

class sirius_engine;

namespace creator {
class task_creator;
}  // namespace creator

namespace telemetry {
class telemetry_context;
}  // namespace telemetry

namespace pipeline {

class sirius_pipeline;
class sirius_meta_pipeline;

class sirius_pipeline_build_state {
 public:
  //! How much to increment batch indexes when multiple pipelines share the same source
  constexpr static std::size_t BATCH_INCREMENT = 10000000000000;

 public:
  //! Duplicate eliminated join scan dependencies
  sirius::reference_map_t<const op::sirius_physical_operator,
                          std::reference_wrapper<sirius_pipeline>>
    delim_join_dependencies;
  //! Materialized CTE scan dependencies
  sirius::reference_map_t<const op::sirius_physical_operator,
                          std::reference_wrapper<sirius_pipeline>>
    cte_dependencies;
  //! CTE_SCAN → consumer pipeline, populated by
  //! `sirius_physical_column_data_scan::build_pipelines`. CTE_SCAN never lands in any pipeline's
  //! `operators[]`, so tree-based wiring resolves consumers through this map, not `dest_for_op`.
  sirius::reference_map_t<const op::sirius_physical_operator,
                          std::reference_wrapper<sirius_pipeline>>
    cte_scan_consumers;

 public:
  void set_pipeline_source(sirius_pipeline& pipeline, op::sirius_physical_operator& op);
  void set_pipeline_sink(sirius_pipeline& pipeline,
                         sirius::optional_ptr<op::sirius_physical_operator> op,
                         std::size_t sink_pipeline_count);
  void set_pipeline_operators(
    sirius_pipeline& pipeline,
    duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>> operators);
  void add_pipeline_operator(sirius_pipeline& pipeline, op::sirius_physical_operator& op);
  duckdb::shared_ptr<sirius_pipeline> create_child_pipeline(const pipeline_build_context& ctx,
                                                            sirius_pipeline& pipeline,
                                                            op::sirius_physical_operator& op);

  sirius::optional_ptr<op::sirius_physical_operator> get_pipeline_source(sirius_pipeline& pipeline);
  sirius::optional_ptr<op::sirius_physical_operator> get_pipeline_sink(sirius_pipeline& pipeline);
  duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>> get_pipeline_operators(
    sirius_pipeline& pipeline);
};

//! The sirius_pipeline class represents an execution pipeline starting at a
class sirius_pipeline : public duckdb::enable_shared_from_this<sirius_pipeline> {
  friend class ::sirius::sirius_engine;
  friend class sirius_pipeline_build_state;
  friend class sirius_meta_pipeline;
  friend class sirius_pipeline_converter;

 public:
  explicit sirius_pipeline(const pipeline_build_context& ctx);
  virtual ~sirius_pipeline() = default;

 public:
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
  // bool get_progress(double &current_percentage, std::size_t &estimated_cardinality);

  //! Returns a list of all operators (including source and sink) involved in this pipeline
  // duckdb::vector<duckdb::reference<op::sirius_physical_operator>> get_all_operators();

  // duckdb::vector<duckdb::const_reference<op::sirius_physical_operator>> get_all_operators()
  // const;

  //! Returns a list of all operators (including source and sink) involved in this pipeline
  duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>> get_operators();
  duckdb::vector<std::reference_wrapper<const op::sirius_physical_operator>> get_operators() const;

  sirius::optional_ptr<op::sirius_physical_operator> get_sink() { return sink; }

  sirius::optional_ptr<op::sirius_physical_operator> get_sink() const noexcept { return sink; }

  sirius::optional_ptr<op::sirius_physical_operator> get_source() { return source; }

  sirius::optional_ptr<op::sirius_physical_operator> get_source() const noexcept { return source; }

  /// The source operator's identity as the query stage manager wants it: its id
  /// and its type.
  ///
  /// Every @c notify_* that names "the operator this pipeline is stuck on" means
  /// this one, so the lookup lives here rather than being re-derived at each
  /// call site.  Reporting must not throw -- a plan that never ran
  /// @c assign_operator_ids still has to dispatch -- so a missing source or an
  /// unnumbered one yields the sentinels rather than aborting.
  [[nodiscard]] std::pair<std::size_t, op::SiriusPhysicalOperatorType> get_source_operator()
    const noexcept
  {
    auto src = get_source();
    if (!src) {
      return {op::sirius_physical_operator::invalid_operator_id,
              op::SiriusPhysicalOperatorType::INVALID};
    }
    return {src->has_operator_id() ? src->get_operator_id()
                                   : op::sirius_physical_operator::invalid_operator_id,
            src->type};
  }

  // Returns the next ports of the pipeline's sink operator, handling special-cased composite
  // operators like left and right delim joins. Returns an empty vector if the sink is not set.
  [[nodiscard]] std::vector<op::sirius_physical_operator::next_port_info>
  get_next_ports_after_sink() const;

  //! A cross-pipeline port described as its peer operator plus the port's barrier type.
  using port_barrier_info = std::pair<op::sirius_physical_operator*, op::MemoryBarrierType>;

  //! Returns the (producer operator, barrier) pair for every port feeding data into this
  //! pipeline. The producer operator is the sink of the upstream (source) pipeline.
  [[nodiscard]] std::vector<port_barrier_info> get_ingress_ports_info() const;

  //! Returns the (consumer operator, barrier) pair for every port this pipeline feeds data into.
  //! The consumer operator is the downstream operator that receives the sink's output.
  [[nodiscard]] std::vector<port_barrier_info> get_egress_ports_info() const;

  //! Set the pipeline ID
  void set_pipeline_id(size_t id) { pipeline_id = id; }
  //! Get the pipeline ID
  size_t get_pipeline_id() const { return pipeline_id; }
  //! Returns the parent pipelines (pipelines that depend on this pipeline)
  std::vector<sirius_pipeline*> get_parents() const;

  std::vector<op::sirius_physical_operator*> get_output_consumers() const;

  //! Notifies downstream pipelines to re-evaluate their status after this pipeline finishes
  //! @param original_pipeline Whether this is the original pipeline whose task finished and called
  //! update_pipeline_status. If it is, we dont want to schedule tasks for its consumers, that will
  //! be done later.
  void notify_downstream_pipelines(bool original_pipeline);

  //! Returns whether any of the operators in the pipeline care about preserving order
  bool is_order_dependent() const;

  //! Registers a new batch index for a pipeline executor - returns the current minimum batch index
  std::size_t register_new_batch_index();

  //! Updates the batch index of a pipeline (and returns the new minimum batch index)
  std::size_t update_batch_index(std::size_t old_index, std::size_t new_index);

  //! The dependencies of this pipeline
  // duckdb::vector<std::weak_ptr<sirius_pipeline>> dependencies;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> dependencies;

  //! Updates the pipeline status
  //! @param original_pipeline Whether this is the original pipeline whose task finished and called
  //! update_pipeline_status. If it is, we dont want to schedule tasks for its consumers, that will
  //! be done later.
  void update_pipeline_status(bool original_pipeline = true);
  //! Checks if the pipeline has been finished
  virtual bool is_pipeline_finished() const;

  //! Query-terminal sink (RESULT_COLLECTOR or STREAMING_SINK): no downstream schedule/wiring;
  //! completion signals execute().
  [[nodiscard]] bool is_query_terminal() const;

  void mark_task_created();
  void mark_task_completed();

  //! Observers for the per-pipeline task counters (testing / diagnostics).
  [[nodiscard]] std::size_t get_tasks_created() const { return tasks_created.load(); }
  [[nodiscard]] std::size_t get_tasks_completed() const { return tasks_completed.load(); }

  //! Set the task_creator pointer so this pipeline can schedule downstream consumers on finish.
  void set_task_creator(sirius::creator::task_creator* tc);

  //! task_creator for schedule(), or nullptr when unwired. Streaming sources use this to
  //! re-arm a starved head; schedule() only enqueues, so off-thread calls are safe.
  [[nodiscard]] sirius::creator::task_creator* get_task_creator() const noexcept
  {
    return _task_creator;
  }

  //! Returns a scoped lock on the pipeline status mutex.
  //! Callers must hold this lock across the operation that consumes pipeline state
  //! (port data pop, partition claim, etc.) and the task constructor that calls
  //! mark_task_created(), so that update_pipeline_status() cannot observe an
  //! empty-port / balanced-counter state while a task is mid-creation.
  [[nodiscard]] std::unique_lock<std::mutex> get_task_creation_lock();

  [[nodiscard]] uuid::UUID pipeline_uuid() const { return _pipeline_uuid; }

  //! Completed-task memory and size history, owned here so upstream estimators can access it
  //! through `port::src_pipeline`.
  [[nodiscard]] pipeline_memory_history& get_memory_history() noexcept { return _memory_history; }
  [[nodiscard]] const pipeline_memory_history& get_memory_history() const noexcept
  {
    return _memory_history;
  }

  //! The SiriusContext-wide telemetry context carried in this pipeline's build
  //! context (set at convert time in sirius_engine). Operators read it via
  //! sirius_physical_operator::get_telemetry_context() to build data_batch probes.
  [[nodiscard]] const telemetry::telemetry_context* get_telemetry_context() const
  {
    return build_ctx_.telemetry_context().get();
  }

  [[nodiscard]] const sirius::operator_params& get_operator_params() const noexcept
  {
    return build_ctx_.get_operator_params();
  }

  [[nodiscard]] const std::shared_ptr<const sirius::like_multiliteral_cache>&
  get_like_multiliteral_cache() const noexcept
  {
    return build_ctx_.get_like_multiliteral_cache();
  }

 private:
  //! Whether or not the pipeline has been readied
  bool ready;
  //! Whether or not the pipeline has been initialized
  std::atomic<bool> initialized;
  //! The source of this pipeline
  sirius::optional_ptr<op::sirius_physical_operator> source;
  //! The chain of intermediate operators
  duckdb::vector<std::reference_wrapper<op::sirius_physical_operator>> operators;
  //! The sink (i.e. destination) for data; this is e.g. a hash table to-be-built
  sirius::optional_ptr<op::sirius_physical_operator> sink;

  //! The global source state
  duckdb::unique_ptr<duckdb::GlobalSourceState> source_state;
  //! The parent pipelines (i.e. pipelines that are dependent on this pipeline to finish)
  duckdb::vector<duckdb::weak_ptr<sirius_pipeline>> parents;

  //! The base batch index of this pipeline
  std::size_t base_batch_index = 0;
  //! Lock for accessing the set of batch indexes
  std::mutex batch_lock;
  //! The set of batch indexes that are currently being processed
  //! Despite batch indexes being unique - this is a multiset
  //! The reason is that when we start a new pipeline we insert the current minimum batch index as a
  //! placeholder Which leads to duplicate entries in the set of active batch indexes
  std::multiset<std::size_t> batch_indexes;

  void schedule_sequential_task(duckdb::shared_ptr<duckdb::Event>& event);
  bool launch_scan_tasks(duckdb::shared_ptr<duckdb::Event>& event, std::size_t max_threads);

  bool schedule_parallel(duckdb::shared_ptr<duckdb::Event>& event);

  //! Task creator pointer for scheduling downstream consumers when this pipeline finishes
  sirius::creator::task_creator* _task_creator{nullptr};

  //! The unique ID of this pipeline (assigned based on new_scheduled order)
  size_t pipeline_id = 0;
  //! Plan-time context (replaces sirius_engine& for plan-time needs)
  pipeline_build_context build_ctx_;

  //! Serialises update_pipeline_status() checks against task_creator's port-pop +
  //! mark_task_created() sequences so neither can observe a transiently-balanced
  //! counter while the other is mid-operation.
  mutable std::mutex _status_mutex;

  //! Whether the pipeline has been finished
  std::atomic<bool> pipeline_finished = false;

  std::atomic<std::size_t> tasks_created   = 0;
  std::atomic<std::size_t> tasks_completed = 0;

  //! Completed-task history; see get_memory_history().
  pipeline_memory_history _memory_history;

  //! NVTX process-wide range tracking the pipeline's active lifetime
  std::atomic<bool> _nvtx_range_started{false};
  nvtxRangeId_t _nvtx_pipeline_range_id{0};

  uuid::UUID _pipeline_uuid{uuid::now_v7()};
};

}  // namespace pipeline
}  // namespace sirius
