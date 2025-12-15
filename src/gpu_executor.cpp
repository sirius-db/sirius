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
#include "data/data_repository_manager.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "fallback.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_operator.hpp"
#include "log/logging.hpp"
#include "operator/gpu_physical_concat.hpp"
#include "operator/gpu_physical_cte.hpp"
#include "operator/gpu_physical_delim_join.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_partition.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "operator/gpu_physical_table_scan.hpp"

#include <stdio.h>

#include <iostream>

namespace duckdb {

void GPUExecutor::Reset()
{
  // lock_guard<mutex> elock(executor_lock);
  gpu_physical_plan = nullptr;
  // cancelled = false;
  gpu_owned_plan.reset();
  // root_executor.reset();
  root_pipelines.clear();
  root_pipeline_idx   = 0;
  completed_pipelines = 0;
  total_pipelines     = 0;
  // error_manager.Reset();
  pipelines.clear();
  new_pipeline_breakers.clear();
  concat_ops.clear();
  // events.clear();
  // to_be_rescheduled_tasks.clear();
  // execution_result = PendingExecutionResult::RESULT_NOT_READY;
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

    if (Config::MODIFIED_PIPELINE) {
      SIRIUS_LOG_DEBUG("Initial Scheduled pipelines: {}", scheduled.size());
      for (size_t i = 0; i < scheduled.size(); i++) {
        auto pipeline = scheduled[i];
        SIRIUS_LOG_DEBUG("Source {}", pipeline->source->GetName());
        for (size_t j = 0; j < pipeline->operators.size(); j++) {
          SIRIUS_LOG_DEBUG(" Op {}", pipeline->operators[j].get().GetName());
        }
        SIRIUS_LOG_DEBUG("Sink {}", pipeline->sink->GetName());
        SIRIUS_LOG_DEBUG("");  // Blank line for separation
      }

      auto data_repo_manager = ::sirius::make_unique<::sirius::data_repository_manager>();
      unordered_map<const GPUPhysicalOperator*, vector<shared_ptr<GPUPipeline>>>
        source_to_pipelines;

      vector<shared_ptr<GPUPipeline>> new_scheduled;
      for (size_t i = 0; i < scheduled.size(); i++) {
        auto current_pipeline = scheduled[i];  // Copy shared_ptr to avoid invalidation

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
            current_pipeline->sink->type == PhysicalOperatorType::TOP_N ||
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

        shared_ptr<GPUPipeline> previous_pipeline = nullptr;
        GPUPhysicalPartition* prev_partition_ptr  = nullptr;

        if (join_sink) {
          // replace hash join sink with partition
          unique_ptr<GPUPhysicalPartition> partition_op;
          if (current_pipeline->operators.size() == 0) {
            // source -> partition -> hash join
            partition_op =
              make_uniq<GPUPhysicalPartition>(current_pipeline->GetSource()->types,
                                              current_pipeline->GetSource()->estimated_cardinality,
                                              current_pipeline->GetSink().get(),
                                              true);
          } else {
            partition_op = make_uniq<GPUPhysicalPartition>(
              current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
              current_pipeline->operators[current_pipeline->operators.size() - 1]
                .get()
                .estimated_cardinality,
              current_pipeline->GetSink().get(),
              true);
          }

          // replace sink with partition_op
          GPUPhysicalPartition* partition_ptr =
            static_cast<GPUPhysicalPartition*>(partition_op.get());

          auto hash_join_op      = current_pipeline->GetSink();
          current_pipeline->sink = partition_ptr;
          current_pipeline->sink->add_next_port_after_sink({hash_join_op.get(), "right"});
          new_pipeline_breakers.push_back(std::move(partition_op));
        }

        if (!join_positions.empty()) {
          for (size_t hj_idx = 0; hj_idx < join_positions.size(); hj_idx++) {
            idx_t join_pos = join_positions[hj_idx];

            // Create a PARTITION operator
            if (join_pos == 0) {
              auto partition_op = make_uniq<GPUPhysicalPartition>(
                current_pipeline->GetSource()->types,
                current_pipeline->GetSource()->estimated_cardinality,
                &current_pipeline->operators[join_pos].get(),
                false);
              new_pipeline_breakers.push_back(std::move(partition_op));
            } else {
              auto partition_op = make_uniq<GPUPhysicalPartition>(
                current_pipeline->operators[join_pos - 1].get().types,
                current_pipeline->operators[join_pos - 1].get().estimated_cardinality,
                &current_pipeline->operators[join_pos].get(),
                false);
              new_pipeline_breakers.push_back(std::move(partition_op));
            }

            GPUPhysicalPartition* partition_ptr =
              static_cast<GPUPhysicalPartition*>(new_pipeline_breakers.back().get());

            // Create new pipeline: PARTITION -> HASH_JOIN -> ... -> SINK
            auto new_pipeline = make_shared_ptr<GPUPipeline>(*this);

            new_pipeline->sink = partition_ptr;
            new_pipeline->sink->add_next_port_after_sink(
              {&current_pipeline->operators[join_pos].get(), "right"});

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
          auto partition_op =
            make_uniq<GPUPhysicalPartition>(current_pipeline->GetSink()->types,
                                            current_pipeline->GetSink()->estimated_cardinality,
                                            current_pipeline->GetSink().get(),
                                            false);
          auto concat_op =
            make_uniq<GPUPhysicalConcat>(partition_op->types, partition_op->estimated_cardinality);
          new_pipeline_breakers.push_back(std::move(partition_op));

          GPUPhysicalPartition* partition_ptr =
            static_cast<GPUPhysicalPartition*>(new_pipeline_breakers.back().get());

          auto group_sort_topn = current_pipeline->sink;
          current_pipeline->operators.push_back(*group_sort_topn);
          current_pipeline->sink = partition_ptr;
          current_pipeline->sink->add_next_port_after_sink({concat_op.get(), "default"});
          concat_ops.push_back(std::move(concat_op));

          new_scheduled.push_back(current_pipeline);

          // Create new pipeline: PARTITION -> SINK
          auto new_pipeline = make_shared_ptr<GPUPipeline>(*this);

          new_pipeline->sink = group_sort_topn;
          new_pipeline->operators.push_back(*concat_ops.back());
          new_pipeline->source = partition_ptr;
          new_pipeline->dependencies.push_back(current_pipeline);

          new_scheduled.push_back(new_pipeline);
        }

        if (right_left_delim_join_sink) {
          auto delim_join   = current_pipeline->GetSink();
          auto& join_op     = delim_join->Cast<GPUPhysicalDelimJoin>().join;
          auto& distinct_op = delim_join->Cast<GPUPhysicalDelimJoin>().distinct;

          unique_ptr<GPUPhysicalPartition> partition_join;
          if (current_pipeline->operators.size() == 0) {
            // source -> partition -> hash join
            partition_join = make_uniq<GPUPhysicalPartition>(
              current_pipeline->GetSource()->types,
              current_pipeline->GetSource()->estimated_cardinality,
              join_op.get(),
              delim_join->type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
          } else {
            partition_join = make_uniq<GPUPhysicalPartition>(
              current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
              current_pipeline->operators[current_pipeline->operators.size() - 1]
                .get()
                .estimated_cardinality,
              join_op.get(),
              delim_join->type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
          }

          auto partition_distinct = make_uniq<GPUPhysicalPartition>(
            distinct_op->types, distinct_op->estimated_cardinality, distinct_op.get(), false);

          delim_join->Cast<GPUPhysicalDelimJoin>().partition_join =
            static_cast<GPUPhysicalPartition*>(partition_join.get());
          delim_join->Cast<GPUPhysicalDelimJoin>().partition_distinct =
            static_cast<GPUPhysicalPartition*>(partition_distinct.get());

          new_pipeline_breakers.push_back(std::move(partition_join));
          new_pipeline_breakers.push_back(std::move(partition_distinct));

          delim_join->Cast<GPUPhysicalDelimJoin>().partition_join->add_next_port_after_sink(
            {join_op.get(), "left"});

          new_scheduled.push_back(current_pipeline);

          GPUPhysicalPartition* partition_distinct_ptr =
            static_cast<GPUPhysicalPartition*>(new_pipeline_breakers.back().get());

          auto concat_op =
            make_uniq<GPUPhysicalConcat>(distinct_op->types, distinct_op->estimated_cardinality);
          delim_join->Cast<GPUPhysicalDelimJoin>().partition_distinct->add_next_port_after_sink(
            {concat_op.get(), "default"});
          concat_ops.push_back(std::move(concat_op));

          // Create new pipeline: PARTITION -> SINK
          auto new_pipeline = make_shared_ptr<GPUPipeline>(*this);

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
        if (new_scheduled[i]->sink->type == PhysicalOperatorType::HASH_GROUP_BY ||
            new_scheduled[i]->sink->type == PhysicalOperatorType::ORDER_BY ||
            new_scheduled[i]->sink->type == PhysicalOperatorType::TOP_N ||
            new_scheduled[i]->sink->type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
          for (auto dependent_pipeline : source_to_pipelines[new_scheduled[i]->sink.get()]) {
            if (dependent_pipeline->operators.size() == 0) {
              new_scheduled[i]->sink->add_next_port_after_sink(
                {dependent_pipeline->sink.get(), "default"});
            } else {
              new_scheduled[i]->sink->add_next_port_after_sink(
                {&dependent_pipeline->operators[0].get(), "default"});
            }
          }
        } else if (new_scheduled[i]->sink->type == PhysicalOperatorType::CTE) {
          auto& cte_op = new_scheduled[i]->sink->Cast<GPUPhysicalCTE>();
          for (auto cte_scan : cte_op.cte_scans) {
            for (auto dependent_pipeline : source_to_pipelines[&cte_scan.get()]) {
              if (dependent_pipeline->operators.size() == 0) {
                new_scheduled[i]->sink->add_next_port_after_sink(
                  {dependent_pipeline->sink.get(), "default"});
              } else {
                new_scheduled[i]->sink->add_next_port_after_sink(
                  {&dependent_pipeline->operators[0].get(), "default"});
              }
            }
          }
        }

        for (auto next_port : new_scheduled[i]->sink->get_next_port_after_sink()) {
          ::sirius::unique_ptr<::sirius::idata_repository> repo =
            ::sirius::make_unique<::sirius::idata_repository>();
          std::string_view port_id = next_port.second;
          auto next_op             = next_port.first;
          data_repo_manager->add_new_repository(next_op, port_id, std::move(repo));
          next_op->add_port(port_id,
                            std::make_unique<GPUPhysicalOperator::port>(
                              MemoryBarrierType::FULL,
                              data_repo_manager->get_repository(next_op, port_id).get(),
                              new_scheduled[i]));
        }

        if (new_scheduled[i]->source->type == PhysicalOperatorType::TABLE_SCAN) {
          ::sirius::unique_ptr<::sirius::idata_repository> repo =
            ::sirius::make_unique<::sirius::idata_repository>();
          std::string port_id = "scan";
          data_repo_manager->add_new_repository(
            new_scheduled[i]->source.get(), port_id, std::move(repo));
          new_scheduled[i]->source->add_port(
            port_id,
            std::make_unique<GPUPhysicalOperator::port>(
              MemoryBarrierType::PIPELINE,
              data_repo_manager->get_repository(new_scheduled[i]->source.get(), port_id).get(),
              new_scheduled[i]));
        }

        if (new_scheduled[i]->sink->type == PhysicalOperatorType::RESULT_COLLECTOR) {
          ::sirius::unique_ptr<::sirius::idata_repository> repo =
            ::sirius::make_unique<::sirius::idata_repository>();
          std::string port_id = "final";
          data_repo_manager->add_new_repository(
            new_scheduled[i]->sink.get(), port_id, std::move(repo));
          new_scheduled[i]->sink->add_port(
            port_id,
            std::make_unique<GPUPhysicalOperator::port>(
              MemoryBarrierType::FULL,
              data_repo_manager->get_repository(new_scheduled[i]->sink.get(), port_id).get(),
              new_scheduled[i]));
        }
      }

      SIRIUS_LOG_DEBUG("Final Scheduled pipelines: {}", new_scheduled.size());
      for (size_t i = 0; i < new_scheduled.size(); i++) {
        auto pipeline = new_scheduled[i];
        SIRIUS_LOG_DEBUG("Source {}", pipeline->source->GetName());
        for (size_t j = 0; j < pipeline->operators.size(); j++) {
          SIRIUS_LOG_DEBUG(" Op {}", pipeline->operators[j].get().GetName());
        }
        if (pipeline->sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN ||
            pipeline->sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto delim_join         = pipeline->GetSink();
          auto partition_join     = delim_join->Cast<GPUPhysicalDelimJoin>().partition_join;
          auto partition_distinct = delim_join->Cast<GPUPhysicalDelimJoin>().partition_distinct;
          {
            std::string msg =
              "Sink " + pipeline->sink->GetName() + " partition join next op after sink: ";
            for (auto next_port : partition_join->get_next_port_after_sink()) {
              msg += next_port.first->GetName() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
          {
            std::string msg =
              "Sink " + pipeline->sink->GetName() + " partition distinct next op after sink: ";
            for (auto next_port : partition_distinct->get_next_port_after_sink()) {
              msg += next_port.first->GetName() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
        } else if (pipeline->sink->type == PhysicalOperatorType::HASH_GROUP_BY ||
                   pipeline->sink->type == PhysicalOperatorType::ORDER_BY ||
                   pipeline->sink->type == PhysicalOperatorType::TOP_N ||
                   pipeline->sink->type == PhysicalOperatorType::UNGROUPED_AGGREGATE ||
                   pipeline->sink->type == PhysicalOperatorType::INVALID ||
                   pipeline->sink->type == PhysicalOperatorType::CTE) {
          std::string msg = "Sink " + pipeline->sink->GetName() + " next op after sink: ";
          for (auto next_port : pipeline->sink->get_next_port_after_sink()) {
            msg += next_port.first->GetName() + " ";
          }
          SIRIUS_LOG_DEBUG("{}", msg);
        } else {
          SIRIUS_LOG_DEBUG("Sink {}", pipeline->sink->GetName());
        }
        SIRIUS_LOG_DEBUG("");
      }
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

};  // namespace duckdb
