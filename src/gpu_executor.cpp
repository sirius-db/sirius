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

#include "config.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "fallback.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_operator.hpp"
#include "gpu_pipeline_hashmap.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "operator/gpu_physical_concat.hpp"
#include "operator/gpu_physical_cte.hpp"
#include "operator/gpu_physical_delim_join.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_partition.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "operator/gpu_physical_table_scan.hpp"

#include <cucascade/data/data_repository_manager.hpp>
#include <stdio.h>

#include <iostream>

namespace duckdb {

void GPUExecutor::Reset()
{
  // lock_guard<mutex> elock(executor_lock);
  gpu_physical_plan    = nullptr;
  sirius_physical_plan = nullptr;
  // cancelled = false;
  gpu_owned_plan.reset();
  sirius_owned_plan.reset();
  // root_executor.reset();
  root_pipelines.clear();
  sirius_root_pipelines.clear();
  root_pipeline_idx   = 0;
  completed_pipelines = 0;
  total_pipelines     = 0;
  // error_manager.Reset();
  pipelines.clear();
  sirius_pipelines.clear();
  new_pipeline_breakers.clear();
  concat_ops.clear();
  operator_to_id.clear();
  next_operator_id.store(0);
  // events.clear();
  // to_be_rescheduled_tasks.clear();
  // execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

size_t GPUExecutor::get_operator_id(const ::sirius::op::sirius_physical_operator* op)
{
  std::lock_guard<std::mutex> lock(operator_id_mutex);
  auto it = operator_to_id.find(op);
  if (it != operator_to_id.end()) { return it->second; }
  size_t id          = next_operator_id++;
  operator_to_id[op] = id;
  return id;
}

void GPUExecutor::Initialize(unique_ptr<GPUPhysicalOperator> plan)
{
  SIRIUS_LOG_DEBUG("Initializing GPUExecutor");
  Reset();
  gpu_owned_plan = std::move(plan);
  InitializeInternal(*gpu_owned_plan);
}

void GPUExecutor::Execute()
{
  // Check if we should fall back to duckdb execution.
  if (Config::ENABLE_FALLBACK_CHECK) {
    FallbackChecker fallback_checker(scheduled);
    fallback_checker.Check();
  }

  // Execution starts here.
  int initial_idx = 0;

  SIRIUS_LOG_DEBUG("Total meta pipelines {}", scheduled.size());

  for (int pipeline_idx = 0; pipeline_idx < scheduled.size(); pipeline_idx++) {
    auto pipeline = scheduled[pipeline_idx];
    SIRIUS_LOG_DEBUG("Executing pipeline {}", pipeline_idx);

    // TODO: This is temporary solution
    // if (pipeline->source->type == PhysicalOperatorType::HASH_JOIN || pipeline->source->type ==
    // PhysicalOperatorType::RESULT_COLLECTOR) { 	continue;
    // }

    vector<shared_ptr<GPUIntermediateRelation>> intermediate_relations;
    shared_ptr<GPUIntermediateRelation> final_relation;
    // vector<unique_ptr<OperatorState>> intermediate_states;
    intermediate_relations.reserve(pipeline->operators.size());
    // intermediate_states.reserve(pipeline->operators.size());

    // SIRIUS_LOG_DEBUG("Executing pipeline op size {}", pipeline->operators.size());
    for (idx_t i = 0; i < pipeline->operators.size(); i++) {
      auto& prev_operator    = i == 0 ? *(pipeline->source) : pipeline->operators[i - 1].get();
      auto& current_operator = pipeline->operators[i].get();

      // auto chunk = make_uniq<DataChunk>();
      // chunk->Initialize(Allocator::Get(context.client), prev_operator.GetTypes());
      shared_ptr<GPUIntermediateRelation> inter_rel =
        make_shared_ptr<GPUIntermediateRelation>(prev_operator.GetTypes().size());
      intermediate_relations.push_back(std::move(inter_rel));

      // auto op_state = current_operator.GetOperatorState(context);
      // intermediate_states.push_back(std::move(op_state));

      // if (current_operator.IsSink() && current_operator.sink_state->state ==
      // SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
      // 	// one of the operators has already figured out no output is possible
      // 	// we can skip executing the pipeline
      // 	FinishProcessing();
      // }
    }
    // InitializeChunk(final_chunk);
    auto& last_op =
      pipeline->operators.empty() ? *pipeline->source : pipeline->operators.back().get();
    final_relation = make_shared_ptr<GPUIntermediateRelation>(last_op.GetTypes().size());

    // auto thread_context = ThreadContext(context);
    // auto exec_context = GPUExecutionContext(context, thread_context, pipeline.get());

    // pipeline->Reset();
    // auto prop = pipeline->executor.context.GetClientProperties();
    // SIRIUS_LOG_DEBUG("Properties: {}", prop.time_zone);
    auto is_empty         = pipeline->operators.empty();
    auto& source_relation = is_empty ? final_relation : intermediate_relations[0];
    // auto source_result = FetchFromSource(source_chunk);

    // StartOperator(*pipeline.source);
    // auto interrupt_state = InterruptState();
    // auto local_source_state = pipeline.source->GetLocalSourceState(exec_context,
    // *pipeline.source_state); OperatorSourceInput source_input = {*pipeline.source_state,
    // *local_source_state, interrupt_state}; pipeline->source->GetData(exec_context,
    // source_relation, source_input);
    auto source_type = pipeline->source.get()->type;
    SIRIUS_LOG_DEBUG("pipeline source type {}", PhysicalOperatorToString(source_type));
    if (source_type == PhysicalOperatorType::TABLE_SCAN) {
      // initialize pipeline
      Pipeline duckdb_pipeline(*executor);
      ThreadContext thread_context(context);
      ExecutionContext exec_context(context, thread_context, &duckdb_pipeline);
      auto& table_scan = pipeline->source->Cast<GPUPhysicalTableScan>();
      table_scan.GetDataDuckDB(exec_context);
    }
    pipeline->source->GetData(*source_relation);
    // SIRIUS_LOG_DEBUG("source relation size {}", source_relation->columns.size());
    // for (auto col : source_relation->columns) {
    // 	SIRIUS_LOG_DEBUG("source relation column size {} column name {}", col->column_length,
    // col->name);
    // }
    // EndOperator(*pipeline.source, &result);

    // call source
    //  SIRIUS_LOG_DEBUG("{}", pipeline->source.get()->GetName());
    for (int current_idx = 1; current_idx <= pipeline->operators.size(); current_idx++) {
      auto op      = pipeline->operators[current_idx - 1];
      auto op_type = op.get().type;
      SIRIUS_LOG_DEBUG("pipeline operator type {}", PhysicalOperatorToString(op_type));
      // SIRIUS_LOG_DEBUG("{}", op.get().GetName());
      // call operator

      auto current_intermediate = current_idx;
      auto& current_relation    = current_intermediate >= intermediate_relations.size()
                                    ? final_relation
                                    : intermediate_relations[current_intermediate];
      // current_chunk.Reset();

      auto& prev_relation    = current_intermediate == initial_idx + 1
                                 ? source_relation
                                 : intermediate_relations[current_intermediate - 1];
      auto operator_idx      = current_idx - 1;
      auto& current_operator = pipeline->operators[operator_idx];

      // auto op_state = current_operator.GetOperatorState(context);
      // intermediate_states.push_back(std::move(op_state));

      // StartOperator(current_operator);
      // auto result = current_operator.get().Execute(exec_context, prev_relation, current_relation,
      // *current_operator.op_state,
      //                                        *intermediate_states[current_intermediate - 1]);

      auto result = current_operator.get().Execute(*prev_relation, *current_relation);
      // EndOperator(current_operator, &current_chunk);
    }
    if (pipeline->sink) {
      auto sink_type = pipeline->sink.get()->type;
      SIRIUS_LOG_DEBUG("pipeline sink type {}", PhysicalOperatorToString(sink_type));
      // SIRIUS_LOG_DEBUG("{}", pipeline->sink.get()->GetName());
      // call sink
      auto& sink_relation = final_relation;
      // SIRIUS_LOG_DEBUG("sink relation size {}", final_relation->columns.size());
      // int i = 0;
      // for (auto col : final_relation->columns) {
      // 	if (col == nullptr) SIRIUS_LOG_DEBUG("{}", i);
      // 	i++;
      // 	// SIRIUS_LOG_DEBUG("sink relation column size {}", col->column_length);
      // }
      // auto interrupt_state = InterruptState();
      // auto local_sink_state = pipeline->sink->GetLocalSinkState(exec_context);
      // OperatorSinkInput sink_input {*pipeline->sink->sink_state, *local_sink_state,
      // interrupt_state}; pipeline->sink->Sink(exec_context, *sink_relation, sink_input);
      pipeline->sink->Sink(*sink_relation);
    }
  }
}

void GPUExecutor::insert_repository(
  std::string_view port_id,
  shared_ptr<sirius::pipeline::sirius_pipeline> input_pipeline,
  shared_ptr<sirius::pipeline::sirius_pipeline> dependent_pipeline)
{
  auto next_op = dependent_pipeline->operators.size() == 0
                   ? dependent_pipeline->get_sink().get()
                   : &dependent_pipeline->operators[0].get();
  size_t op_id = get_operator_id(next_op);
  data_repo_manager->add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<sirius::op::sirius_physical_operator::port>(
                      sirius::op::MemoryBarrierType::FULL,
                      data_repo_manager->get_repository(op_id, port_id).get(),
                      input_pipeline,
                      dependent_pipeline));
  input_pipeline->get_sink()->add_next_port_after_sink({next_op, port_id});
}

void GPUExecutor::insert_repository(
  std::string_view port_id,
  sirius::op::sirius_physical_operator* cur_op,
  shared_ptr<sirius::pipeline::sirius_pipeline> input_pipeline,
  shared_ptr<sirius::pipeline::sirius_pipeline> dependent_pipeline)
{
  auto next_op = dependent_pipeline->operators.size() == 0
                   ? dependent_pipeline->get_sink().get()
                   : &dependent_pipeline->operators[0].get();
  size_t op_id = get_operator_id(next_op);
  data_repo_manager->add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<sirius::op::sirius_physical_operator::port>(
                      sirius::op::MemoryBarrierType::FULL,
                      data_repo_manager->get_repository(op_id, port_id).get(),
                      input_pipeline,
                      dependent_pipeline));
  cur_op->add_next_port_after_sink({next_op, port_id});
}

void GPUExecutor::InitializeInternal(GPUPhysicalOperator& plan)
{
  // auto &scheduler = TaskScheduler::GetScheduler(context);
  {
    // lock_guard<mutex> elock(executor_lock);
    gpu_physical_plan = &plan;

    // this->profiler = ClientData::Get(context).profiler;
    // profiler->Initialize(plan);
    // this->producer = scheduler.CreateProducer();

    // build and ready the pipelines
    GPUPipelineBuildState state;
    auto root_pipeline = make_shared_ptr<GPUMetaPipeline>(*this, state, nullptr);
    root_pipeline->Build(*gpu_physical_plan);
    root_pipeline->Ready();

    // ready recursive cte pipelines too
    // TODO: SUPPORT RECURSIVE CTE FOR GPU
    // for (auto &rec_cte_ref : recursive_ctes) {
    // 	auto &rec_cte = rec_cte_ref.get().Cast<PhysicalRecursiveCTE>();
    // 	// rec_cte.recursive_meta_pipeline->Ready();
    // }

    // set root pipelines, i.e., all pipelines that end in the final sink
    root_pipeline->GetPipelines(root_pipelines, false);
    root_pipeline_idx = 0;

    // collect all meta-pipelines from the root pipeline
    vector<shared_ptr<GPUMetaPipeline>> to_schedule;
    scheduled.clear();
    new_scheduled.clear();
    root_pipeline->GetMetaPipelines(to_schedule, true, true);

    // number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to
    // set it here
    total_pipelines = to_schedule.size();

    SIRIUS_LOG_DEBUG("Total meta pipelines {}", to_schedule.size());
    int schedule_count = 0;
    int meta           = 0;
    while (schedule_count < to_schedule.size()) {
      vector<shared_ptr<GPUMetaPipeline>> children;
      to_schedule[to_schedule.size() - 1 - meta]->GetMetaPipelines(children, false, true);
      auto base_pipeline   = to_schedule[to_schedule.size() - 1 - meta]->GetBasePipeline();
      bool should_schedule = true;

      // already scheduled
      if (find(scheduled.begin(), scheduled.end(), base_pipeline) != scheduled.end()) {
        should_schedule = false;
      } else {
        // check if all children are scheduled
        for (auto& child : children) {
          if (find(scheduled.begin(), scheduled.end(), child->GetBasePipeline()) ==
              scheduled.end()) {
            should_schedule = false;
            break;
          }
        }
        // check if all dependencies are scheduled
        for (int dep = 0; dep < base_pipeline->dependencies.size(); dep++) {
          if (find(scheduled.begin(), scheduled.end(), base_pipeline->dependencies[dep]) ==
              scheduled.end()) {
            should_schedule = false;
            break;
          }
        }
      }
      if (should_schedule) {
        vector<shared_ptr<GPUPipeline>> pipeline_inside;
        to_schedule[to_schedule.size() - 1 - meta]->GetPipelines(pipeline_inside, false);
        for (int pipeline_idx = 0; pipeline_idx < pipeline_inside.size(); pipeline_idx++) {
          auto& pipeline = pipeline_inside[pipeline_idx];
          if (pipeline_inside[pipeline_idx]->source->type == PhysicalOperatorType::HASH_JOIN) {
            auto& temp = pipeline_inside[pipeline_idx]->source.get()->Cast<GPUPhysicalHashJoin>();
            if (temp.join_type == JoinType::RIGHT || temp.join_type == JoinType::RIGHT_SEMI ||
                temp.join_type == JoinType::RIGHT_ANTI) {
              if (!Config::MODIFIED_PIPELINE) scheduled.push_back(pipeline);
            }
            continue;
          } else {
            scheduled.push_back(pipeline);
          }
        }
        schedule_count++;
      }
      meta = (meta + 1) % to_schedule.size();
    }

    // collect all pipelines from the root pipelines (recursively) for the progress bar and verify
    // them
    root_pipeline->GetPipelines(pipelines, true);
    SIRIUS_LOG_DEBUG("total_pipelines = {}", pipelines.size());
  }
}

void GPUExecutor::CancelTasks()
{
  pipelines.clear();
  root_pipelines.clear();
}

shared_ptr<GPUPipeline> GPUExecutor::CreateChildPipeline(GPUPipeline& current,
                                                         GPUPhysicalOperator& op)
{
  D_ASSERT(!current.operators.empty());
  D_ASSERT(op.IsSource());
  // found another operator that is a source, schedule a child pipeline
  // 'op' is the source, and the sink is the same
  auto child_pipeline    = make_shared_ptr<GPUPipeline>(*this);
  child_pipeline->sink   = current.sink;
  child_pipeline->source = &op;

  // the child pipeline has the same operators up until 'op'
  for (auto current_op : current.operators) {
    if (&current_op.get() == &op) { break; }
    child_pipeline->operators.push_back(current_op);
  }

  return child_pipeline;
}

shared_ptr<::sirius::pipeline::sirius_pipeline> GPUExecutor::create_child_pipeline(
  ::sirius::pipeline::sirius_pipeline& current, ::sirius::op::sirius_physical_operator& op)
{
  D_ASSERT(!current.operators.empty());
  D_ASSERT(op.is_source());
  // found another operator that is a source, schedule a child pipeline
  // 'op' is the source, and the sink is the same
  auto child_pipeline    = make_shared_ptr<sirius::pipeline::sirius_pipeline>(*this);
  child_pipeline->sink   = current.sink;
  child_pipeline->source = &op;

  // the child pipeline has the same operators up until 'op'
  for (auto current_op : current.operators) {
    if (&current_op.get() == &op) { break; }
    child_pipeline->operators.push_back(current_op);
  }

  return child_pipeline;
}

bool GPUExecutor::HasResultCollector()
{
  return gpu_physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> GPUExecutor::GetResult()
{
  D_ASSERT(HasResultCollector());
  if (!gpu_physical_plan) throw InvalidInputException("gpu_physical_plan is NULL");
  if (gpu_physical_plan.get() == NULL) throw InvalidInputException("gpu_physical_plan is NULL");
  auto& result_collector = gpu_physical_plan.get()->Cast<GPUPhysicalMaterializedCollector>();
  D_ASSERT(result_collector.sink_state);
  result_collector.sink_state = result_collector.GetGlobalSinkState(context);
  unique_ptr<QueryResult> res = result_collector.GetResult(*(result_collector.sink_state));
  return res;
}

void GPUExecutor::initialize(unique_ptr<::sirius::op::sirius_physical_operator> plan)
{
  SIRIUS_LOG_DEBUG("Initializing GPUExecutor");
  Reset();
  sirius_owned_plan = std::move(plan);
  initialize_internal(*sirius_owned_plan);
}

void GPUExecutor::initialize_internal(::sirius::op::sirius_physical_operator& plan)
{
  // auto &scheduler = TaskScheduler::GetScheduler(context);
  {
    // lock_guard<mutex> elock(executor_lock);
    sirius_physical_plan = &plan;

    // this->profiler = ClientData::Get(context).profiler;
    // profiler->Initialize(plan);
    // this->producer = scheduler.CreateProducer();

    // build and ready the pipelines
    sirius::pipeline::sirius_pipeline_build_state state;
    auto root_pipeline =
      make_shared_ptr<sirius::pipeline::sirius_meta_pipeline>(*this, state, nullptr);
    root_pipeline->build(*sirius_physical_plan);
    root_pipeline->ready();

    // ready recursive cte pipelines too
    // TODO: SUPPORT RECURSIVE CTE FOR GPU
    // for (auto &rec_cte_ref : recursive_ctes) {
    // 	auto &rec_cte = rec_cte_ref.get().Cast<PhysicalRecursiveCTE>();
    // 	// rec_cte.recursive_meta_pipeline->Ready();
    // }

    // set root pipelines, i.e., all pipelines that end in the final sink
    root_pipeline->get_pipelines(sirius_root_pipelines, false);
    root_pipeline_idx = 0;

    // collect all meta-pipelines from the root pipeline
    vector<shared_ptr<sirius::pipeline::sirius_meta_pipeline>> to_schedule;
    sirius_scheduled.clear();
    new_scheduled.clear();
    root_pipeline->get_meta_pipelines(to_schedule, true, true);

    // number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to
    // set it here
    total_pipelines = to_schedule.size();

    SIRIUS_LOG_DEBUG("Total meta pipelines {}", to_schedule.size());
    int schedule_count = 0;
    int meta           = 0;
    while (schedule_count < to_schedule.size()) {
      vector<shared_ptr<sirius::pipeline::sirius_meta_pipeline>> children;
      to_schedule[to_schedule.size() - 1 - meta]->get_meta_pipelines(children, false, true);
      auto base_pipeline   = to_schedule[to_schedule.size() - 1 - meta]->get_base_pipeline();
      bool should_schedule = true;

      // already scheduled
      if (find(sirius_scheduled.begin(), sirius_scheduled.end(), base_pipeline) !=
          sirius_scheduled.end()) {
        should_schedule = false;
      } else {
        // check if all children are scheduled
        for (auto& child : children) {
          if (find(sirius_scheduled.begin(), sirius_scheduled.end(), child->get_base_pipeline()) ==
              sirius_scheduled.end()) {
            should_schedule = false;
            break;
          }
        }
        // check if all dependencies are scheduled
        for (int dep = 0; dep < base_pipeline->dependencies.size(); dep++) {
          if (find(sirius_scheduled.begin(),
                   sirius_scheduled.end(),
                   base_pipeline->dependencies[dep]) == sirius_scheduled.end()) {
            should_schedule = false;
            break;
          }
        }
      }
      if (should_schedule) {
        vector<shared_ptr<sirius::pipeline::sirius_pipeline>> pipeline_inside;
        to_schedule[to_schedule.size() - 1 - meta]->get_pipelines(pipeline_inside, false);
        for (int pipeline_idx = 0; pipeline_idx < pipeline_inside.size(); pipeline_idx++) {
          auto& pipeline = pipeline_inside[pipeline_idx];
          if (pipeline_inside[pipeline_idx]->source->type == PhysicalOperatorType::HASH_JOIN) {
            auto& temp = pipeline_inside[pipeline_idx]
                           ->source.get()
                           ->Cast<sirius::op::sirius_physical_hash_join>();
            if (temp.join_type == JoinType::RIGHT || temp.join_type == JoinType::RIGHT_SEMI ||
                temp.join_type == JoinType::RIGHT_ANTI) {
              if (!Config::MODIFIED_PIPELINE) sirius_scheduled.push_back(pipeline);
            }
            continue;
          } else {
            sirius_scheduled.push_back(pipeline);
          }
        }
        schedule_count++;
      }
      meta = (meta + 1) % to_schedule.size();
    }

    if (Config::MODIFIED_PIPELINE) {
      // perform deep copy on scheduled pipelines
      vector<shared_ptr<sirius::pipeline::sirius_pipeline>> copied_scheduled;
      for (size_t i = 0; i < sirius_scheduled.size(); i++) {
        auto copied_pipeline = make_shared_ptr<sirius::pipeline::sirius_pipeline>(*this);
        // copy source
        copied_pipeline->source = sirius_scheduled[i]->source;
        // copy operators
        for (size_t j = 0; j < sirius_scheduled[i]->operators.size(); j++) {
          copied_pipeline->operators.push_back(sirius_scheduled[i]->operators[j]);
        }
        // copy sink
        copied_pipeline->sink = sirius_scheduled[i]->sink;
        copied_scheduled.push_back(copied_pipeline);
      }

      // SIRIUS_LOG_DEBUG("Initial Scheduled pipelines: {}", scheduled.size());
      // for (size_t i = 0; i < scheduled.size(); i++) {
      //   auto pipeline = scheduled[i];
      //   SIRIUS_LOG_DEBUG("Source {}", pipeline->source->GetName());
      //   for (size_t j = 0; j < pipeline->operators.size(); j++) {
      //     SIRIUS_LOG_DEBUG(" Op {}", pipeline->operators[j].get().GetName());
      //   }
      //   SIRIUS_LOG_DEBUG("Sink {}", pipeline->sink->GetName());
      //   SIRIUS_LOG_DEBUG("");  // Blank line for separation
      // }

      data_repo_manager = ::std::make_unique<::cucascade::shared_data_repository_manager>();
      unordered_map<const sirius::op::sirius_physical_operator*,
                    vector<shared_ptr<sirius::pipeline::sirius_pipeline>>>
        source_to_pipelines;

      for (size_t i = 0; i < copied_scheduled.size(); i++) {
        auto current_pipeline = copied_scheduled[i];  // Copy shared_ptr to avoid invalidation

        // Store original dependencies to preserve them
        auto original_dependencies = std::move(current_pipeline->dependencies);

        vector<idx_t> join_positions;

        for (idx_t op_idx = 0; op_idx < current_pipeline->operators.size(); op_idx++) {
          if (current_pipeline->operators[op_idx].get().type == PhysicalOperatorType::HASH_JOIN ||
              current_pipeline->operators[op_idx].get().type ==
                PhysicalOperatorType::NESTED_LOOP_JOIN) {
            join_positions.push_back(op_idx);
          }
        }

        bool group_agg_sort_topn_sink = false;
        if (current_pipeline->sink->type == PhysicalOperatorType::HASH_GROUP_BY ||
            current_pipeline->sink->type == PhysicalOperatorType::ORDER_BY ||
            current_pipeline->sink->type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
          group_agg_sort_topn_sink = true;
        }

        bool join_sink = false;
        if (current_pipeline->sink->type == PhysicalOperatorType::HASH_JOIN ||
            current_pipeline->sink->type == PhysicalOperatorType::NESTED_LOOP_JOIN) {
          join_sink = true;
        }

        bool right_left_delim_join_sink = false;
        if (current_pipeline->sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN ||
            current_pipeline->sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
          right_left_delim_join_sink = true;
        }

        shared_ptr<sirius::pipeline::sirius_pipeline> previous_pipeline = nullptr;
        sirius::op::sirius_physical_partition* prev_partition_ptr       = nullptr;

        if (join_sink) {
          // replace hash join sink with partition
          unique_ptr<sirius::op::sirius_physical_partition> partition_op;
          if (current_pipeline->operators.size() == 0) {
            // source -> partition -> hash join
            partition_op = make_uniq<sirius::op::sirius_physical_partition>(
              current_pipeline->get_source()->types,
              current_pipeline->get_source()->estimated_cardinality,
              current_pipeline->get_sink().get(),
              true);
          } else {
            partition_op = make_uniq<sirius::op::sirius_physical_partition>(
              current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
              current_pipeline->operators[current_pipeline->operators.size() - 1]
                .get()
                .estimated_cardinality,
              current_pipeline->get_sink().get(),
              true);
          }

          // replace sink with partition_op
          sirius::op::sirius_physical_partition* partition_ptr =
            static_cast<sirius::op::sirius_physical_partition*>(partition_op.get());

          auto hash_join_op      = current_pipeline->get_sink();
          current_pipeline->sink = partition_ptr;
          // current_pipeline->sink->add_next_port_after_sink({hash_join_op.get(), "left"});
          new_pipeline_breakers.push_back(std::move(partition_op));
        }

        if (!join_positions.empty()) {
          for (size_t hj_idx = 0; hj_idx < join_positions.size(); hj_idx++) {
            idx_t join_pos = join_positions[hj_idx];

            // Create a PARTITION operator
            if (join_pos == 0) {
              auto partition_op = make_uniq<sirius::op::sirius_physical_partition>(
                current_pipeline->get_source()->types,
                current_pipeline->get_source()->estimated_cardinality,
                &current_pipeline->operators[join_pos].get(),
                false);
              new_pipeline_breakers.push_back(std::move(partition_op));
            } else {
              auto partition_op = make_uniq<sirius::op::sirius_physical_partition>(
                current_pipeline->operators[join_pos - 1].get().types,
                current_pipeline->operators[join_pos - 1].get().estimated_cardinality,
                &current_pipeline->operators[join_pos].get(),
                false);
              new_pipeline_breakers.push_back(std::move(partition_op));
            }

            sirius::op::sirius_physical_partition* partition_ptr =
              static_cast<sirius::op::sirius_physical_partition*>(
                new_pipeline_breakers.back().get());
            // Create new pipeline: PARTITION -> HASH_JOIN -> ... -> SINK
            auto new_pipeline = make_shared_ptr<sirius::pipeline::sirius_pipeline>(*this);

            new_pipeline->sink = partition_ptr;
            // new_pipeline->sink->add_next_port_after_sink(
            //   {&current_pipeline->operators[join_pos].get(), "right"});

            if (hj_idx == 0) {
              // Move operators from current pipeline to new pipeline
              for (idx_t j = 0; j < join_pos; j++) {
                new_pipeline->operators.push_back(current_pipeline->operators[j]);
              }
              new_pipeline->source       = current_pipeline->source;
              new_pipeline->dependencies = std::move(original_dependencies);
            } else {
              // Move operators from current pipeline to new pipeline
              for (idx_t j = join_positions[hj_idx - 1]; j < join_pos; j++) {
                new_pipeline->operators.push_back(current_pipeline->operators[j]);
              }
              new_pipeline->source = prev_partition_ptr;
              new_pipeline->dependencies.push_back(previous_pipeline);
            }

            new_scheduled.push_back(new_pipeline);
            if (hj_idx == join_positions.size() - 1) {
              // remove operators from current pipeline
              current_pipeline->operators.erase(current_pipeline->operators.begin(),
                                                current_pipeline->operators.begin() + join_pos);

              // add new pipeline to dependencies
              current_pipeline->source = partition_ptr;
              current_pipeline->dependencies.clear();
              current_pipeline->dependencies.push_back(new_pipeline);
            }

            // create a shared ptr from new pipeline
            previous_pipeline  = new_pipeline;
            prev_partition_ptr = partition_ptr;
          }
        }

        if (group_agg_sort_topn_sink) {
          // Create a PARTITION operator
          auto partition_op = make_uniq<sirius::op::sirius_physical_partition>(
            current_pipeline->get_sink()->types,
            current_pipeline->get_sink()->estimated_cardinality,
            current_pipeline->get_sink().get(),
            false);
          auto concat_op = make_uniq<sirius::op::sirius_physical_concat>(
            partition_op->types, partition_op->estimated_cardinality);
          new_pipeline_breakers.push_back(std::move(partition_op));

          sirius::op::sirius_physical_partition* partition_ptr =
            static_cast<sirius::op::sirius_physical_partition*>(new_pipeline_breakers.back().get());

          auto group_sort_topn = current_pipeline->sink;
          current_pipeline->operators.push_back(*group_sort_topn);
          current_pipeline->sink = partition_ptr;
          // current_pipeline->sink->add_next_port_after_sink({concat_op.get(), "default"});
          concat_ops.push_back(std::move(concat_op));

          new_scheduled.push_back(current_pipeline);

          // Create new pipeline: PARTITION -> SINK
          auto new_pipeline = make_shared_ptr<sirius::pipeline::sirius_pipeline>(*this);

          new_pipeline->sink = group_sort_topn;
          new_pipeline->operators.push_back(*concat_ops.back());
          new_pipeline->source = partition_ptr;
          new_pipeline->dependencies.push_back(current_pipeline);

          new_scheduled.push_back(new_pipeline);
        }

        if (right_left_delim_join_sink) {
          auto delim_join   = current_pipeline->get_sink();
          auto& join_op     = delim_join->Cast<sirius::op::sirius_physical_delim_join>().join;
          auto& distinct_op = delim_join->Cast<sirius::op::sirius_physical_delim_join>().distinct;

          unique_ptr<sirius::op::sirius_physical_partition> partition_join;
          if (delim_join->type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
            if (current_pipeline->operators.size() == 0) {
              // source -> partition -> hash join
              partition_join = make_uniq<sirius::op::sirius_physical_partition>(
                current_pipeline->get_source()->types,
                current_pipeline->get_source()->estimated_cardinality,
                join_op.get(),
                delim_join->type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
            } else {
              partition_join = make_uniq<sirius::op::sirius_physical_partition>(
                current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
                current_pipeline->operators[current_pipeline->operators.size() - 1]
                  .get()
                  .estimated_cardinality,
                join_op.get(),
                delim_join->type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
            }
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_join =
              static_cast<sirius::op::sirius_physical_partition*>(partition_join.get());

            new_pipeline_breakers.push_back(std::move(partition_join));
          }

          auto partition_distinct = make_uniq<sirius::op::sirius_physical_partition>(
            distinct_op->types, distinct_op->estimated_cardinality, distinct_op.get(), false);

          delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_distinct =
            static_cast<sirius::op::sirius_physical_partition*>(partition_distinct.get());

          new_pipeline_breakers.push_back(std::move(partition_distinct));

          new_scheduled.push_back(current_pipeline);

          sirius::op::sirius_physical_partition* partition_distinct_ptr =
            static_cast<sirius::op::sirius_physical_partition*>(new_pipeline_breakers.back().get());

          auto concat_op = make_uniq<sirius::op::sirius_physical_concat>(
            distinct_op->types, distinct_op->estimated_cardinality);

          concat_ops.push_back(std::move(concat_op));

          // Create new pipeline: PARTITION -> SINK
          auto new_pipeline = make_shared_ptr<sirius::pipeline::sirius_pipeline>(*this);

          new_pipeline->sink = distinct_op.get();
          new_pipeline->operators.push_back(*concat_ops.back());
          new_pipeline->source = partition_distinct_ptr;
          new_pipeline->dependencies.push_back(current_pipeline);

          new_scheduled.push_back(new_pipeline);
        }

        if (!group_agg_sort_topn_sink && !right_left_delim_join_sink) {
          new_scheduled.push_back(current_pipeline);
        }
      }

      // build source to pipelines map
      for (size_t i = 0; i < new_scheduled.size(); i++) {
        source_to_pipelines[new_scheduled[i]->source.get()].push_back(new_scheduled[i]);
      }

      // add data repositories and ports
      for (size_t i = 0; i < new_scheduled.size(); i++) {
        const bool is_top_n_merge_sink = dynamic_cast<sirius::op::sirius_physical_top_n_merge*>(
                                           new_scheduled[i]->sink.get()) != nullptr;
        const bool is_ungrouped_agg_merge_sink =
          dynamic_cast<sirius::op::sirius_physical_ungrouped_aggregate_merge*>(
            new_scheduled[i]->sink.get()) != nullptr;
        if (new_scheduled[i]->sink->type == PhysicalOperatorType::HASH_GROUP_BY ||
            new_scheduled[i]->sink->type == PhysicalOperatorType::ORDER_BY ||
            new_scheduled[i]->sink->type == PhysicalOperatorType::TOP_N || is_top_n_merge_sink ||
            is_ungrouped_agg_merge_sink ||
            new_scheduled[i]->sink->type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
          auto sink_op             = new_scheduled[i]->get_sink().get();
          std::string_view port_id = "default";
          for (auto dependent_pipeline : source_to_pipelines[sink_op]) {
            insert_repository(port_id, new_scheduled[i], dependent_pipeline);
          }
        } else if (new_scheduled[i]->sink->type == PhysicalOperatorType::CTE) {
          auto& cte_op = new_scheduled[i]->get_sink()->Cast<sirius::op::sirius_physical_cte>();
          std::string_view port_id = "default";
          for (auto cte_scan : cte_op.cte_scans) {
            for (auto dependent_pipeline : source_to_pipelines[&cte_scan.get()]) {
              insert_repository(port_id, new_scheduled[i], dependent_pipeline);
            }
          }
        } else if (new_scheduled[i]->sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto delim_join = new_scheduled[i]->get_sink();
          auto partition_join =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_join;
          auto partition_distinct =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_distinct;
          // Find the pipeline containing the join as the first operator
          sirius::op::sirius_physical_operator* join_op = partition_join->get_parent_op();
          bool found                                    = false;
          for (size_t j = 0; j < new_scheduled.size(); j++) {
            if (new_scheduled[j]->operators.size() > 0 &&
                &new_scheduled[j]->operators[0].get() == join_op) {
              insert_repository("build", partition_join, new_scheduled[i], new_scheduled[j]);
              found = true;
              break;
            }
          }
          if (!found) {
            throw std::runtime_error(
              "DELIM_JOIN partition_join: could not find pipeline with join as first operator");
          }
          for (auto dependent_pipeline : source_to_pipelines[partition_distinct]) {
            insert_repository("default", partition_distinct, new_scheduled[i], dependent_pipeline);
          }
        } else if (new_scheduled[i]->sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto delim_join = new_scheduled[i]->get_sink();
          auto partition_distinct =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_distinct;
          for (auto dependent_pipeline : source_to_pipelines[partition_distinct]) {
            insert_repository("default", partition_distinct, new_scheduled[i], dependent_pipeline);
          }
          auto column_data_scan =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().join->children[0].get();
          for (auto dependent_pipeline : source_to_pipelines[column_data_scan]) {
            insert_repository("default", column_data_scan, new_scheduled[i], dependent_pipeline);
          }
        } else if (auto* partition = dynamic_cast<sirius::op::sirius_physical_partition*>(
                     new_scheduled[i]->sink.get())) {
          std::string_view port_id = partition->is_build_partition() ? "build" : "default";

          if (partition->is_build_partition()) {
            // For build partitions, no pipeline uses it as source.
            // Instead, connect directly to the HASH_JOIN operator stored in parent_op.
            // Find the pipeline containing this HASH_JOIN as the first operator.
            sirius::op::sirius_physical_operator* hash_join_op = partition->get_parent_op();
            bool found                                         = false;
            for (size_t j = 0; j < new_scheduled.size(); j++) {
              // The join is guaranteed to be the first operator in the pipeline
              if (new_scheduled[j]->operators.size() > 0 &&
                  &new_scheduled[j]->operators[0].get() == hash_join_op) {
                insert_repository(port_id, new_scheduled[i], new_scheduled[j]);
                found = true;
                break;
              }
            }
            if (!found) {
              throw std::runtime_error(
                "Build partition: could not find pipeline with HASH_JOIN as first operator");
            }
          } else {
            // Probe partitions have dependent pipelines in source_to_pipelines
            for (auto dependent_pipeline :
                 source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
              insert_repository(port_id, new_scheduled[i], dependent_pipeline);
            }
          }
        } else if (new_scheduled[i]->sink->type == PhysicalOperatorType::RESULT_COLLECTOR) {
          std::string_view port_id = "final";
          size_t sink_op_id        = get_operator_id(new_scheduled[i]->get_sink().get());
          data_repo_manager->add_new_repository(
            sink_op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
          new_scheduled[i]->sink->add_port(
            port_id,
            std::make_unique<sirius::op::sirius_physical_operator::port>(
              sirius::op::MemoryBarrierType::FULL,
              data_repo_manager->get_repository(sink_op_id, port_id).get(),
              new_scheduled[i],
              nullptr));
        } else {
          throw std::runtime_error("Unsupported sink type for modified pipeline");
        }

        if (new_scheduled[i]->source->type == PhysicalOperatorType::TABLE_SCAN) {
          ::std::unique_ptr<::cucascade::shared_data_repository> repo =
            ::std::make_unique<::cucascade::shared_data_repository>();
          std::string port_id = "scan";
          auto next_op        = new_scheduled[i]->operators.size() == 0
                                  ? new_scheduled[i]->get_sink().get()
                                  : &new_scheduled[i]->operators[0].get();
          size_t op_id        = get_operator_id(next_op);
          data_repo_manager->add_new_repository(op_id, port_id, std::move(repo));
          next_op->add_port(port_id,
                            std::make_unique<sirius::op::sirius_physical_operator::port>(
                              sirius::op::MemoryBarrierType::PIPELINE,
                              data_repo_manager->get_repository(op_id, port_id).get(),
                              nullptr,
                              new_scheduled[i]));
        }
      }

      SIRIUS_LOG_DEBUG("Final Scheduled pipelines: {}", new_scheduled.size());
      for (size_t i = 0; i < new_scheduled.size(); i++) {
        auto pipeline = new_scheduled[i];
        SIRIUS_LOG_DEBUG("Source {}", pipeline->source->get_name());
        for (size_t j = 0; j < pipeline->operators.size(); j++) {
          SIRIUS_LOG_DEBUG(" Op {}", pipeline->operators[j].get().get_name());
        }
        if (pipeline->sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto delim_join = pipeline->get_sink();
          auto partition_join =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_join;
          auto partition_distinct =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_distinct;
          {
            std::string msg =
              "Sink " + pipeline->sink->get_name() + " partition join next op after sink: ";
            for (auto next_port : partition_join->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
          {
            std::string msg =
              "Sink " + pipeline->sink->get_name() + " partition distinct next op after sink: ";
            for (auto next_port : partition_distinct->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
        } else if (pipeline->sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto delim_join = pipeline->get_sink();
          auto column_data_scan =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().join->children[0].get();
          auto partition_distinct =
            delim_join->Cast<sirius::op::sirius_physical_delim_join>().partition_distinct;
          {
            std::string msg =
              "Sink " + pipeline->sink->get_name() + " column data scan next op after sink: ";
            for (auto next_port : column_data_scan->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
          {
            std::string msg =
              "Sink " + pipeline->sink->get_name() + " partition distinct next op after sink: ";
            for (auto next_port : partition_distinct->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
        } else if (pipeline->sink->type == PhysicalOperatorType::HASH_GROUP_BY ||
                   pipeline->sink->type == PhysicalOperatorType::ORDER_BY ||
                   pipeline->sink->type == PhysicalOperatorType::TOP_N ||
                   dynamic_cast<sirius::op::sirius_physical_top_n_merge*>(pipeline->sink.get()) !=
                     nullptr ||
                   pipeline->sink->type == PhysicalOperatorType::UNGROUPED_AGGREGATE ||
                   pipeline->sink->type == PhysicalOperatorType::EXTENSION ||
                   pipeline->sink->type == PhysicalOperatorType::CTE) {
          std::string msg = "Sink " + pipeline->sink->get_name() + " next op after sink: ";
          for (auto next_port : pipeline->sink->get_next_port_after_sink()) {
            msg += next_port.first->get_name() + " ";
          }
          SIRIUS_LOG_DEBUG("{}", msg);
        } else {
          SIRIUS_LOG_DEBUG("Sink {}", pipeline->sink->get_name());
        }
        SIRIUS_LOG_DEBUG("");
      }
    }

    // collect all pipelines from the root pipelines (recursively) for the progress bar and verify
    // them
    root_pipeline->get_pipelines(sirius_pipelines, true);
    SIRIUS_LOG_DEBUG("total_pipelines = {}", sirius_pipelines.size());
  }
}

};  // namespace duckdb
