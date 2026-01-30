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
#include "log/logging.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_merge_grouped_aggregate.hpp"
#include "op/sirius_physical_merge_top_n.hpp"
#include "op/sirius_physical_merge_ungrouped_aggregate.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "sirius_engine.hpp"

#include <cucascade/data/data_repository_manager.hpp>
#include <stdio.h>

#include <iostream>

namespace sirius {

void sirius_engine::reset()
{
  sirius_physical_plan = nullptr;
  sirius_owned_plan.reset();
  sirius_root_pipelines.clear();
  root_pipeline_idx   = 0;
  total_pipelines     = 0;
  sirius_pipelines.clear();
  new_pipeline_breakers.clear();
  concat_ops.clear();
  operator_to_id.clear();
  next_operator_id.store(0);
}

size_t sirius_engine::get_operator_id(const op::sirius_physical_operator* op)
{
  std::lock_guard<std::mutex> lock(operator_id_mutex);
  auto it = operator_to_id.find(op);
  if (it != operator_to_id.end()) { return it->second; }
  size_t id          = next_operator_id++;
  operator_to_id[op] = id;
  return id;
}

void sirius_engine::insert_repository(
  std::string_view port_id,
  duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
  duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline)
{
  auto next_op = dependent_pipeline->get_inner_operators().size() == 0
                   ? dependent_pipeline->get_sink().get()
                   : &dependent_pipeline->get_inner_operators()[0].get();
  size_t op_id = get_operator_id(next_op);
  data_repo_manager->add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<op::sirius_physical_operator::port>(
                      op::MemoryBarrierType::FULL,
                      data_repo_manager->get_repository(op_id, port_id).get(),
                      input_pipeline,
                      dependent_pipeline));
  input_pipeline->get_sink()->add_next_port_after_sink({next_op, port_id});
}

void sirius_engine::insert_repository(
  std::string_view port_id,
  op::sirius_physical_operator* cur_op,
  duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
  duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline)
{
  auto next_op = dependent_pipeline->get_inner_operators().size() == 0
                   ? dependent_pipeline->get_sink().get()
                   : &dependent_pipeline->get_inner_operators()[0].get();
  size_t op_id = get_operator_id(next_op);
  data_repo_manager->add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<op::sirius_physical_operator::port>(
                      op::MemoryBarrierType::FULL,
                      data_repo_manager->get_repository(op_id, port_id).get(),
                      input_pipeline,
                      dependent_pipeline));
  cur_op->add_next_port_after_sink({next_op, port_id});
}

void sirius_engine::cancel_tasks()
{
  sirius_pipelines.clear();
  sirius_root_pipelines.clear();
}

duckdb::shared_ptr<pipeline::sirius_pipeline> sirius_engine::create_child_pipeline(
  pipeline::sirius_pipeline& current, op::sirius_physical_operator& op)
{
  D_ASSERT(!current.operators.empty());
  D_ASSERT(op.is_source());
  // found another operator that is a source, schedule a child pipeline
  // 'op' is the source, and the sink is the same
  auto child_pipeline    = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
  child_pipeline->sink   = current.get_sink();
  child_pipeline->source = &op;

  // the child pipeline has the same operators up until 'op'
  for (auto current_op : current.get_inner_operators()) {
    if (&current_op.get() == &op) { break; }
    child_pipeline->operators.push_back(current_op);
  }

  return child_pipeline;
}

bool sirius_engine::has_result_collector()
{
  return sirius_physical_plan->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR;
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_engine::get_result()
{
  D_ASSERT(has_result_collector());
  if (!sirius_physical_plan) throw duckdb::InvalidInputException("sirius_physical_plan is NULL");
  if (sirius_physical_plan.get() == NULL) throw duckdb::InvalidInputException("sirius_physical_plan is NULL");
  auto& result_collector = sirius_physical_plan.get()->Cast<op::sirius_physical_materialized_collector>();
  D_ASSERT(result_collector.sink_state);
  result_collector.sink_state = result_collector.get_global_sink_state(context);
  duckdb::unique_ptr<duckdb::QueryResult> res = result_collector.get_result(*(result_collector.sink_state));
  return res;
}

void sirius_engine::initialize(duckdb::unique_ptr<op::sirius_physical_operator> plan)
{
  SIRIUS_LOG_DEBUG("Initializing sirius_engine");
  reset();
  sirius_owned_plan = std::move(plan);
  initialize_internal(*sirius_owned_plan);
}

void sirius_engine::execute()
{
  // get the task creator from sirius context
  // register sirius pipeline to the task creator (by calling task_creator::set_pipeline_hashmap)
  // wait until the query finish
  // take the query result from sirius_physical_result_collector
  // return the result to duckdb
  printf("Client Context Pointer on execute: %p\n", (void*)&context);
  // get sirius context
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    throw duckdb::InvalidInputException("Sirius context is not initialized.");
  }
  printf("Sirius Context Pointer on execute: %p\n", (void*)sirius_ctx.get());
  auto& task_creator = sirius_ctx->get_task_creator();
  printf("Task Creator get next task id: %lu\n", task_creator.get_next_task_id());
}

duckdb::unique_ptr<op::sirius_physical_operator> 
sirius_engine::construct_sirius_specific_operator(op::sirius_physical_operator* op) {

  if (op->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& scan_physical_op = op->Cast<op::sirius_physical_table_scan>();
    return duckdb::make_uniq<op::sirius_physical_duckdb_scan>(&scan_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    auto& group_by_physical_op = op->Cast<op::sirius_physical_grouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_merge_grouped_aggregate>(&group_by_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::ORDER_BY) {
    auto& order_by_physical_op = op->Cast<op::sirius_physical_order>();
    return duckdb::make_uniq<op::sirius_physical_merge_sort>(&order_by_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::TOP_N) {
    auto& topn_physical_op = op->Cast<op::sirius_physical_top_n>();
    return duckdb::make_uniq<op::sirius_physical_merge_top_n>(&topn_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
    auto& ungrouped_agg_physical_op = op->Cast<op::sirius_physical_ungrouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_merge_ungrouped_aggregate>(&ungrouped_agg_physical_op);
  } else {
    throw duckdb::InternalException("Unsupported operator type" + SiriusPhysicalOperatorToString(op->type) +
                                    " for constructing sirius specific operator.");
  }
}

void sirius_engine::initialize_internal(op::sirius_physical_operator& plan)
{
  // auto &scheduler = TaskScheduler::GetScheduler(context);
  {
    // lock_guard<mutex> elock(executor_lock);
    sirius_physical_plan = &plan;

    // this->profiler = ClientData::Get(context).profiler;
    // profiler->Initialize(plan);
    // this->producer = scheduler.CreateProducer();

    // build and ready the pipelines
    pipeline::sirius_pipeline_build_state state;
    auto root_pipeline =
      duckdb::make_shared_ptr<pipeline::sirius_meta_pipeline>(*this, state, nullptr);
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
    duckdb::vector<duckdb::shared_ptr<pipeline::sirius_meta_pipeline>> to_schedule;
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
      duckdb::vector<duckdb::shared_ptr<pipeline::sirius_meta_pipeline>> children;
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
        duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> pipeline_inside;
        to_schedule[to_schedule.size() - 1 - meta]->get_pipelines(pipeline_inside, false);
        for (int pipeline_idx = 0; pipeline_idx < pipeline_inside.size(); pipeline_idx++) {
          auto& pipeline = pipeline_inside[pipeline_idx];
          if (pipeline_inside[pipeline_idx]->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN) {
            auto& temp = pipeline_inside[pipeline_idx]
                           ->source.get()
                           ->Cast<op::sirius_physical_hash_join>();
            if (temp.join_type == duckdb::JoinType::RIGHT || temp.join_type == duckdb::JoinType::RIGHT_SEMI ||
                temp.join_type == duckdb::JoinType::RIGHT_ANTI) {
              if (!duckdb::Config::MODIFIED_PIPELINE) sirius_scheduled.push_back(pipeline);
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

    // perform deep copy on scheduled pipelines
    duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> copied_scheduled;
    for (size_t i = 0; i < sirius_scheduled.size(); i++) {
      auto copied_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
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
    unordered_map<const op::sirius_physical_operator*,
                  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>>
      source_to_pipelines;

    for (size_t i = 0; i < copied_scheduled.size(); i++) {
      auto current_pipeline = copied_scheduled[i];  // Copy duckdb::shared_ptr to avoid invalidation

      // Store original dependencies to preserve them
      auto original_dependencies = std::move(current_pipeline->dependencies);

      duckdb::vector<duckdb::idx_t> join_positions;

      for (duckdb::idx_t op_idx = 0; op_idx < current_pipeline->operators.size(); op_idx++) {
        if (current_pipeline->operators[op_idx].get().type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
            current_pipeline->operators[op_idx].get().type ==
              op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
          join_positions.push_back(op_idx);
        }
      }

      bool group_agg_sort_topn_sink = false;
      if (current_pipeline->sink->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY ||
          current_pipeline->sink->type == op::SiriusPhysicalOperatorType::ORDER_BY ||
          current_pipeline->sink->type == op::SiriusPhysicalOperatorType::TOP_N ||
          current_pipeline->sink->type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
        group_agg_sort_topn_sink = true;
      }

      bool join_sink = false;
      if (current_pipeline->sink->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
          current_pipeline->sink->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
        join_sink = true;
      }

      bool right_left_delim_join_sink = false;
      if (current_pipeline->sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
          current_pipeline->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
        right_left_delim_join_sink = true;
      }

      bool scan_source = false;
      if (current_pipeline->source->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
        scan_source = true;
      }

      duckdb::shared_ptr<pipeline::sirius_pipeline> previous_pipeline = nullptr;
      op::sirius_physical_concat* prev_concat_ptr       = nullptr;

      if (scan_source) {
        auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        auto scan_op      = current_pipeline->get_source();
        auto new_scan_op = construct_sirius_specific_operator(scan_op.get());

        // todo(bobbi) currently this can be set to any operator since it's never used, and now we set it to scan_op
        new_pipeline->source = scan_op.get();
        new_pipeline->sink   = new_scan_op.get();

        current_pipeline->source = new_scan_op.get();
        // move scan_op to current_pipeline.operator[0], current_pipeline.operator[0] to current_pipeline.operator[1], ...
        current_pipeline->operators.insert(
          current_pipeline->operators.begin(), *scan_op);
        current_pipeline->dependencies.push_back(new_pipeline);

        new_scheduled.push_back(new_pipeline);
        new_pipeline_breakers.push_back(std::move(new_scan_op));
      }

      if (join_sink) {
        // replace hash join sink with partition
        duckdb::unique_ptr<op::sirius_physical_partition> partition_op;
        auto hash_join_op      = current_pipeline->get_sink();
        if (current_pipeline->operators.size() == 0) {
          // source -> partition -> hash join
          partition_op = make_uniq<op::sirius_physical_partition>(
            current_pipeline->get_source()->types,
            current_pipeline->get_source()->estimated_cardinality,
            hash_join_op.get(),
            true);
        } else {
          partition_op = make_uniq<op::sirius_physical_partition>(
            current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
            current_pipeline->operators[current_pipeline->operators.size() - 1]
              .get()
              .estimated_cardinality,
            hash_join_op.get(),
            true);
        }

        // replace sink with partition_op
        op::sirius_physical_partition* partition_ptr =
          static_cast<op::sirius_physical_partition*>(partition_op.get());
        current_pipeline->sink = partition_ptr;

        new_scheduled.push_back(current_pipeline);
        
        // create new pipeline for concat_op
        auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        duckdb::unique_ptr<op::sirius_physical_concat> concat_op = make_uniq<op::sirius_physical_concat>(
          partition_ptr->types, partition_ptr->estimated_cardinality, hash_join_op.get(), true);
        new_pipeline->source = partition_ptr;
        new_pipeline->sink = concat_op.get();
        new_pipeline->dependencies.push_back(current_pipeline);

        new_scheduled.push_back(new_pipeline);
        
        new_pipeline_breakers.push_back(std::move(partition_op));
        new_pipeline_breakers.push_back(std::move(concat_op));
      }

      if (!join_positions.empty()) {
        for (size_t hj_idx = 0; hj_idx < join_positions.size(); hj_idx++) {
          duckdb::idx_t join_pos = join_positions[hj_idx];

          // Create a PARTITION operator
          if (join_pos == 0) {
            auto partition_op = make_uniq<op::sirius_physical_partition>(
              current_pipeline->get_source()->types,
              current_pipeline->get_source()->estimated_cardinality,
              &current_pipeline->operators[join_pos].get(),
              false);
            new_pipeline_breakers.push_back(std::move(partition_op));
          } else {
            auto partition_op = make_uniq<op::sirius_physical_partition>(
              current_pipeline->operators[join_pos - 1].get().types,
              current_pipeline->operators[join_pos - 1].get().estimated_cardinality,
              &current_pipeline->operators[join_pos].get(),
              false);
            new_pipeline_breakers.push_back(std::move(partition_op));
          }

          op::sirius_physical_partition* partition_ptr =
            static_cast<op::sirius_physical_partition*>(
              new_pipeline_breakers.back().get());
          // Create new pipeline: PARTITION -> HASH_JOIN -> ... -> SINK
          auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
          new_pipeline->sink = partition_ptr;

          if (hj_idx == 0) {
            // Move operators from current pipeline to new pipeline
            for (duckdb::idx_t j = 0; j < join_pos; j++) {
              new_pipeline->operators.push_back(current_pipeline->operators[j]);
            }
            new_pipeline->source       = current_pipeline->source;
            new_pipeline->dependencies = std::move(original_dependencies);
          } else {
            // Move operators from current pipeline to new pipeline
            for (duckdb::idx_t j = join_positions[hj_idx - 1]; j < join_pos; j++) {
              new_pipeline->operators.push_back(current_pipeline->operators[j]);
            }
            new_pipeline->source = prev_concat_ptr;
            new_pipeline->dependencies.push_back(previous_pipeline);
          }

          new_scheduled.push_back(new_pipeline);

          // new pipeline for concat_op
          auto more_new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
          duckdb::unique_ptr<op::sirius_physical_concat> concat_op = make_uniq<op::sirius_physical_concat>(
            partition_ptr->types, partition_ptr->estimated_cardinality, &current_pipeline->operators[join_pos].get(), false);
          more_new_pipeline->source = partition_ptr;
          more_new_pipeline->sink = concat_op.get();
          more_new_pipeline->dependencies.push_back(new_pipeline);

          new_pipeline_breakers.push_back(std::move(concat_op));
          op::sirius_physical_concat* concat_ptr =
            static_cast<op::sirius_physical_concat*>(new_pipeline_breakers.back().get());

          new_scheduled.push_back(more_new_pipeline);
          
          if (hj_idx == join_positions.size() - 1) {
            // remove operators from current pipeline
            current_pipeline->operators.erase(current_pipeline->operators.begin(),
                                              current_pipeline->operators.begin() + join_pos);

            // add new pipeline to dependencies
            current_pipeline->source = concat_ptr;
            current_pipeline->dependencies.clear();
            current_pipeline->dependencies.push_back(more_new_pipeline);
          }

          // create a shared ptr from new pipeline
          previous_pipeline  = more_new_pipeline;
          prev_concat_ptr = concat_ptr;
        }
      }

      if (group_agg_sort_topn_sink) {
        auto group_sort_topn = current_pipeline->sink;
        if (group_sort_topn->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY || 
            group_sort_topn->type == op::SiriusPhysicalOperatorType::ORDER_BY) {

          // Create a PARTITION operator
          auto partition_op = make_uniq<op::sirius_physical_partition>(
            current_pipeline->get_sink()->types,
            current_pipeline->get_sink()->estimated_cardinality,
            current_pipeline->get_sink().get(),
            false);
          new_pipeline_breakers.push_back(std::move(partition_op));

          op::sirius_physical_partition* partition_ptr =
            static_cast<op::sirius_physical_partition*>(new_pipeline_breakers.back().get());

          current_pipeline->operators.push_back(*group_sort_topn);
          current_pipeline->sink = partition_ptr;
        }

        new_scheduled.push_back(current_pipeline);
        auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);

        if (group_sort_topn->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY ||
            group_sort_topn->type == op::SiriusPhysicalOperatorType::ORDER_BY) {
          op::sirius_physical_partition* partition_ptr =
            static_cast<op::sirius_physical_partition*>(new_pipeline_breakers.back().get());
          new_pipeline->source = partition_ptr;
        } else {
          new_pipeline->source = group_sort_topn;
        }
        auto merge_op = construct_sirius_specific_operator(group_sort_topn.get());
        new_pipeline->sink = merge_op.get();
        new_pipeline->dependencies.push_back(current_pipeline);
        new_scheduled.push_back(new_pipeline);
        new_pipeline_breakers.push_back(std::move(merge_op));
      }

      if (right_left_delim_join_sink) {
        auto delim_join   = current_pipeline->get_sink();
        auto& join_op     = delim_join->Cast<op::sirius_physical_delim_join>().join;
        auto& distinct_op = delim_join->Cast<op::sirius_physical_delim_join>().distinct;

        duckdb::unique_ptr<op::sirius_physical_partition> partition_join;
        if (delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
          if (current_pipeline->operators.size() == 0) {
            // source -> partition -> hash join
            partition_join = make_uniq<op::sirius_physical_partition>(
              current_pipeline->get_source()->types,
              current_pipeline->get_source()->estimated_cardinality,
              join_op.get(),
              delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
          } else {
            partition_join = make_uniq<op::sirius_physical_partition>(
              current_pipeline->operators[current_pipeline->operators.size() - 1].get().types,
              current_pipeline->operators[current_pipeline->operators.size() - 1]
                .get()
                .estimated_cardinality,
              join_op.get(),
              delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
          }
          delim_join->Cast<op::sirius_physical_delim_join>().partition_join =
            static_cast<op::sirius_physical_partition*>(partition_join.get());
        }

        auto partition_distinct = make_uniq<op::sirius_physical_partition>(
          distinct_op->types, distinct_op->estimated_cardinality, distinct_op.get(), false);

        delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct =
          static_cast<op::sirius_physical_partition*>(partition_distinct.get());

        new_scheduled.push_back(current_pipeline);

        if (delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
          // new pipeline for concat_op
          auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
          duckdb::unique_ptr<op::sirius_physical_concat> concat_op = make_uniq<op::sirius_physical_concat>(
            partition_join.get()->types, partition_join.get()->estimated_cardinality, join_op.get(), delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
          new_pipeline->source = partition_join.get();
          new_pipeline->sink = concat_op.get();
          new_pipeline->dependencies.push_back(current_pipeline);
          
          new_pipeline_breakers.push_back(std::move(partition_join));
          new_pipeline_breakers.push_back(std::move(concat_op));
          new_scheduled.push_back(new_pipeline);
        }

        auto merge_distinct_op = construct_sirius_specific_operator(distinct_op.get());

        // Create new pipeline: PARTITION -> SINK
        auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);

        new_pipeline->source = partition_distinct.get();
        new_pipeline->sink = merge_distinct_op.get();
        new_pipeline->dependencies.push_back(current_pipeline);

        new_pipeline_breakers.push_back(std::move(partition_distinct));
        new_pipeline_breakers.push_back(std::move(merge_distinct_op));
        new_scheduled.push_back(new_pipeline);
      }

      if (!group_agg_sort_topn_sink && !right_left_delim_join_sink && !join_sink) {
        new_scheduled.push_back(current_pipeline);
      }
    }

    for (size_t i = 0; i < new_scheduled.size(); i++) {
      if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_GROUP_BY ||
          new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_SORT ||
          new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_TOP_N ||
          new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_AGGREGATE) {
        op::sirius_physical_operator* child_op;
        auto sink_op = new_scheduled[i]->get_sink();
        if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_GROUP_BY) {
          child_op = sink_op->Cast<op::sirius_physical_merge_grouped_aggregate>().get_child_op();
        } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_SORT) {
          child_op = sink_op->Cast<op::sirius_physical_merge_sort>().get_child_op();
        } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_TOP_N) {
          child_op = sink_op->Cast<op::sirius_physical_merge_top_n>().get_child_op();
        } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_AGGREGATE) {
          child_op = sink_op->Cast<op::sirius_physical_merge_ungrouped_aggregate>().get_child_op();
        }
        for (size_t j = 0; j < new_scheduled.size(); j++) {
          if (new_scheduled[j]->source.get() == child_op) {
            new_scheduled[j]->source = new_scheduled[i]->get_sink().get();
          }
        }
      }
    }

    // build source to pipelines map
    for (size_t i = 0; i < new_scheduled.size(); i++) {
      source_to_pipelines[new_scheduled[i]->source.get()].push_back(new_scheduled[i]);
    }

    // add data repositories and ports
    for (size_t i = 0; i < new_scheduled.size(); i++) {
      if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_GROUP_BY ||
          new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_SORT ||
          new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_TOP_N ||
          new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_AGGREGATE) {
        auto sink_op             = new_scheduled[i]->get_sink().get();
        std::string_view port_id = "default";
        for (auto dependent_pipeline : source_to_pipelines[sink_op]) {
          insert_repository(port_id, new_scheduled[i], dependent_pipeline);
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::CTE) {
        auto& cte_op = new_scheduled[i]->get_sink()->Cast<op::sirius_physical_cte>();
        std::string_view port_id = "default";
        for (auto cte_scan : cte_op.cte_scans) {
          for (auto dependent_pipeline : source_to_pipelines[&cte_scan.get()]) {
            insert_repository(port_id, new_scheduled[i], dependent_pipeline);
          }
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
        auto delim_join = new_scheduled[i]->get_sink();
        auto partition_join =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_join;
        auto partition_distinct =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
        // Find the pipeline containing the join as the first operator
        op::sirius_physical_operator* join_op = partition_join->get_parent_op();
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
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
        auto delim_join = new_scheduled[i]->get_sink();
        auto partition_distinct =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
        for (auto dependent_pipeline : source_to_pipelines[partition_distinct]) {
          insert_repository("default", partition_distinct, new_scheduled[i], dependent_pipeline);
        }
        auto column_data_scan =
          delim_join->Cast<op::sirius_physical_delim_join>().join->children[0].get();
        for (auto dependent_pipeline : source_to_pipelines[column_data_scan]) {
          insert_repository("default", column_data_scan, new_scheduled[i], dependent_pipeline);
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::CONCAT) {
        auto& concat =
          new_scheduled[i]->get_sink()->Cast<op::sirius_physical_concat>();
        std::string_view port_id = concat.is_build_concat() ? "build" : "default";

        if (concat.is_build_concat()) {
          // For build concats, no pipeline uses it as source.
          // Instead, connect directly to the HASH_JOIN operator stored in parent_op.
          // Find the pipeline containing this HASH_JOIN as the first operator.
          op::sirius_physical_operator* hash_join_op = concat.get_parent_op();
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
              "Build concat: could not find pipeline with HASH_JOIN as first operator");
          }
        } else {
          // Probe concats have dependent pipelines in source_to_pipelines
          for (auto dependent_pipeline :
                source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
            insert_repository(port_id, new_scheduled[i], dependent_pipeline);
          }
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::PARTITION) {
          // Partition operators have dependent pipelines in source_to_pipelines
          for (auto dependent_pipeline :
                source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
            insert_repository("default", new_scheduled[i], dependent_pipeline);
          }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
          for (auto dependent_pipeline :
                source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
            auto next_op = dependent_pipeline->get_inner_operators().size() == 0
                            ? dependent_pipeline->get_sink().get()
                            : &dependent_pipeline->get_inner_operators()[0].get();
            size_t op_id = get_operator_id(next_op);
            std::string_view port_id = "scan";
            data_repo_manager->add_new_repository(
              op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
            next_op->add_port(port_id,
                              std::make_unique<op::sirius_physical_operator::port>(
                                op::MemoryBarrierType::PIPELINE,
                                data_repo_manager->get_repository(op_id, port_id).get(),
                                new_scheduled[i],
                                dependent_pipeline));
            new_scheduled[i]->get_sink()->add_next_port_after_sink({next_op, port_id});
          }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
        // No action needed for RESULT_COLLECTOR sinks
      } else {
        throw std::runtime_error("Unsupported sink type for modified pipeline");
      }
    }

    SIRIUS_LOG_DEBUG("Final Scheduled pipelines: {}", new_scheduled.size());
    for (size_t i = 0; i < new_scheduled.size(); i++) {
      auto pipeline = new_scheduled[i];
      SIRIUS_LOG_DEBUG("Source {}", pipeline->source->get_name());
      for (size_t j = 0; j < pipeline->operators.size(); j++) {
        SIRIUS_LOG_DEBUG(" Op {}", pipeline->operators[j].get().get_name());
      }
      if (pipeline->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
        auto delim_join = pipeline->get_sink();
        auto partition_join =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_join;
        auto partition_distinct =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
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
      } else if (pipeline->sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
        auto delim_join = pipeline->get_sink();
        auto column_data_scan =
          delim_join->Cast<op::sirius_physical_delim_join>().join->children[0].get();
        auto partition_distinct =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
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
      } else {
        std::string msg = "Sink " + pipeline->sink->get_name() + " next op after sink: ";
        for (auto next_port : pipeline->sink->get_next_port_after_sink()) {
          msg += next_port.first->get_name() + " ";
        }
        SIRIUS_LOG_DEBUG("{}", msg);
      }
      SIRIUS_LOG_DEBUG("");
    }

    // collect all pipelines from the root pipelines (recursively) for the progress bar and verify
    // them
    root_pipeline->get_pipelines(sirius_pipelines, true);
    SIRIUS_LOG_DEBUG("total_pipelines = {}", sirius_pipelines.size());
  }
}

};  // namespace duckdb
