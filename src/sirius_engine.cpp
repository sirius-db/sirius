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
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_table_scan.hpp"
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
  return sirius_physical_plan->type == duckdb::PhysicalOperatorType::RESULT_COLLECTOR;
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
          if (pipeline_inside[pipeline_idx]->get_source()->type == duckdb::PhysicalOperatorType::HASH_JOIN) {
            auto& temp = pipeline_inside[pipeline_idx]
                           ->get_source()
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

    if (duckdb::Config::MODIFIED_PIPELINE) {
      // perform deep copy on scheduled pipelines
      duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> copied_scheduled;
      for (size_t i = 0; i < sirius_scheduled.size(); i++) {
        auto copied_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        // copy source
        copied_pipeline->source = sirius_scheduled[i]->get_source();
        // copy operators
        for (size_t j = 0; j < sirius_scheduled[i]->get_inner_operators().size(); j++) {
          copied_pipeline->operators.push_back(sirius_scheduled[i]->get_inner_operators()[j]);
        }
        // copy sink
        copied_pipeline->sink = sirius_scheduled[i]->get_sink();
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
      unordered_map<const op::sirius_physical_operator*,
                    duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>>
        source_to_pipelines;

      for (size_t i = 0; i < copied_scheduled.size(); i++) {
        auto current_pipeline = copied_scheduled[i];  // Copy duckdb::shared_ptr to avoid invalidation

        // Store original dependencies to preserve them
        auto original_dependencies = std::move(current_pipeline->dependencies);

        duckdb::vector<idx_t> join_positions;

        for (idx_t op_idx = 0; op_idx < current_pipeline->get_inner_operators().size(); op_idx++) {
          if (current_pipeline->get_inner_operators()[op_idx].get().type == duckdb::PhysicalOperatorType::HASH_JOIN ||
              current_pipeline->get_inner_operators()[op_idx].get().type ==
                duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN) {
            join_positions.push_back(op_idx);
          }
        }

        bool group_agg_sort_topn_sink = false;
        if (current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::HASH_GROUP_BY ||
            current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::ORDER_BY ||
            current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::TOP_N ||
            current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE) {
          group_agg_sort_topn_sink = true;
        }

        bool join_sink = false;
        if (current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::HASH_JOIN ||
            current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN) {
          join_sink = true;
        }

        bool right_left_delim_join_sink = false;
        if (current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::LEFT_DELIM_JOIN ||
            current_pipeline->get_sink()->type == duckdb::PhysicalOperatorType::RIGHT_DELIM_JOIN) {
          right_left_delim_join_sink = true;
        }

        duckdb::shared_ptr<pipeline::sirius_pipeline> previous_pipeline = nullptr;
        op::sirius_physical_partition* prev_partition_ptr       = nullptr;

        if (join_sink) {
          // replace hash join sink with partition
          duckdb::unique_ptr<op::sirius_physical_partition> partition_op;
          if (current_pipeline->get_inner_operators().size() == 0) {
            // source -> partition -> hash join
            partition_op = make_uniq<op::sirius_physical_partition>(
              current_pipeline->get_source()->types,
              current_pipeline->get_source()->estimated_cardinality,
              current_pipeline->get_sink().get(),
              true);
          } else {
            partition_op = make_uniq<op::sirius_physical_partition>(
              current_pipeline->get_inner_operators()[current_pipeline->get_inner_operators().size() - 1].get().types,
              current_pipeline->get_inner_operators()[current_pipeline->get_inner_operators().size() - 1]
                .get()
                .estimated_cardinality,
              current_pipeline->get_sink().get(),
              true);
          }

          // replace sink with partition_op
          op::sirius_physical_partition* partition_ptr =
            static_cast<op::sirius_physical_partition*>(partition_op.get());

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
              auto partition_op = make_uniq<op::sirius_physical_partition>(
                current_pipeline->get_source()->types,
                current_pipeline->get_source()->estimated_cardinality,
                &current_pipeline->get_inner_operators()[join_pos].get(),
                false);
              new_pipeline_breakers.push_back(std::move(partition_op));
            } else {
              auto partition_op = make_uniq<op::sirius_physical_partition>(
                current_pipeline->get_inner_operators()[join_pos - 1].get().types,
                current_pipeline->get_inner_operators()[join_pos - 1].get().estimated_cardinality,
                &current_pipeline->get_inner_operators()[join_pos].get(),
                false);
              new_pipeline_breakers.push_back(std::move(partition_op));
            }

            op::sirius_physical_partition* partition_ptr =
              static_cast<op::sirius_physical_partition*>(
                new_pipeline_breakers.back().get());
            // Create new pipeline: PARTITION -> HASH_JOIN -> ... -> SINK
            auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);

            new_pipeline->sink = partition_ptr;
            // new_pipeline->sink->add_next_port_after_sink(
            //   {&current_pipeline->operators[join_pos].get(), "right"});

            if (hj_idx == 0) {
              // Move operators from current pipeline to new pipeline
              for (idx_t j = 0; j < join_pos; j++) {
                new_pipeline->operators.push_back(current_pipeline->get_inner_operators()[j]);
              }
              new_pipeline->source       = current_pipeline->get_source();
              new_pipeline->dependencies = std::move(original_dependencies);
            } else {
              // Move operators from current pipeline to new pipeline
              for (idx_t j = join_positions[hj_idx - 1]; j < join_pos; j++) {
                new_pipeline->operators.push_back(current_pipeline->get_inner_operators()[j]);
              }
              new_pipeline->source = prev_partition_ptr;
              new_pipeline->dependencies.push_back(previous_pipeline);
            }

            new_scheduled.push_back(new_pipeline);
            if (hj_idx == join_positions.size() - 1) {
              // remove operators from current pipeline
              current_pipeline->operators.erase(current_pipeline->get_inner_operators().begin(),
                                                current_pipeline->get_inner_operators().begin() + join_pos);

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
          auto partition_op = make_uniq<op::sirius_physical_partition>(
            current_pipeline->get_sink()->types,
            current_pipeline->get_sink()->estimated_cardinality,
            current_pipeline->get_sink().get(),
            false);
          auto concat_op = make_uniq<op::sirius_physical_concat>(
            partition_op->types, partition_op->estimated_cardinality);
          new_pipeline_breakers.push_back(std::move(partition_op));

          op::sirius_physical_partition* partition_ptr =
            static_cast<op::sirius_physical_partition*>(new_pipeline_breakers.back().get());

          auto group_sort_topn = current_pipeline->get_sink();
          current_pipeline->get_inner_operators().push_back(*group_sort_topn);
          current_pipeline->sink = partition_ptr;
          // current_pipeline->sink->add_next_port_after_sink({concat_op.get(), "default"});
          concat_ops.push_back(std::move(concat_op));

          new_scheduled.push_back(current_pipeline);

          // Create new pipeline: PARTITION -> SINK
          auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);

          new_pipeline->sink = group_sort_topn;
          new_pipeline->operators.push_back(*concat_ops.back());
          new_pipeline->source = partition_ptr;
          new_pipeline->dependencies.push_back(current_pipeline);

          new_scheduled.push_back(new_pipeline);
        }

        if (right_left_delim_join_sink) {
          auto delim_join   = current_pipeline->get_sink();
          auto& join_op     = delim_join->Cast<op::sirius_physical_delim_join>().join;
          auto& distinct_op = delim_join->Cast<op::sirius_physical_delim_join>().distinct;

          duckdb::unique_ptr<op::sirius_physical_partition> partition_join;
          if (delim_join->type == duckdb::PhysicalOperatorType::RIGHT_DELIM_JOIN) {
            if (current_pipeline->get_inner_operators().size() == 0) {
              // source -> partition -> hash join
              partition_join = make_uniq<op::sirius_physical_partition>(
                current_pipeline->get_source()->types,
                current_pipeline->get_source()->estimated_cardinality,
                join_op.get(),
                delim_join->type == duckdb::PhysicalOperatorType::RIGHT_DELIM_JOIN);
            } else {
              partition_join = make_uniq<op::sirius_physical_partition>(
                current_pipeline->get_inner_operators()[current_pipeline->get_inner_operators().size() - 1].get().types,
                current_pipeline->get_inner_operators()[current_pipeline->get_inner_operators().size() - 1]
                  .get()
                  .estimated_cardinality,
                join_op.get(),
                delim_join->type == duckdb::PhysicalOperatorType::RIGHT_DELIM_JOIN);
            }
            delim_join->Cast<op::sirius_physical_delim_join>().partition_join =
              static_cast<op::sirius_physical_partition*>(partition_join.get());

            new_pipeline_breakers.push_back(std::move(partition_join));
          }

          auto partition_distinct = make_uniq<op::sirius_physical_partition>(
            distinct_op->types, distinct_op->estimated_cardinality, distinct_op.get(), false);

          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct =
            static_cast<op::sirius_physical_partition*>(partition_distinct.get());

          new_pipeline_breakers.push_back(std::move(partition_distinct));

          new_scheduled.push_back(current_pipeline);

          op::sirius_physical_partition* partition_distinct_ptr =
            static_cast<op::sirius_physical_partition*>(new_pipeline_breakers.back().get());

          auto concat_op = make_uniq<op::sirius_physical_concat>(
            distinct_op->types, distinct_op->estimated_cardinality);

          concat_ops.push_back(std::move(concat_op));

          // Create new pipeline: PARTITION -> SINK
          auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);

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
        source_to_pipelines[new_scheduled[i]->get_source().get()].push_back(new_scheduled[i]);
      }

      // add data repositories and ports
      for (size_t i = 0; i < new_scheduled.size(); i++) {
        if (new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::HASH_GROUP_BY ||
            new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::ORDER_BY ||
            new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::TOP_N ||
            new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE) {
          auto sink_op             = new_scheduled[i]->get_sink().get();
          std::string_view port_id = "default";
          for (auto dependent_pipeline : source_to_pipelines[sink_op]) {
            insert_repository(port_id, new_scheduled[i], dependent_pipeline);
          }
        } else if (new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::CTE) {
          auto& cte_op = new_scheduled[i]->get_sink()->Cast<op::sirius_physical_cte>();
          std::string_view port_id = "default";
          for (auto cte_scan : cte_op.cte_scans) {
            for (auto dependent_pipeline : source_to_pipelines[&cte_scan.get()]) {
              insert_repository(port_id, new_scheduled[i], dependent_pipeline);
            }
          }
        } else if (new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto delim_join = new_scheduled[i]->get_sink();
          auto partition_join =
            delim_join->Cast<op::sirius_physical_delim_join>().partition_join;
          auto partition_distinct =
            delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
          // Find the pipeline containing the join as the first operator
          op::sirius_physical_operator* join_op = partition_join->get_parent_op();
          bool found                                    = false;
          for (size_t j = 0; j < new_scheduled.size(); j++) {
            if (new_scheduled[j]->get_inner_operators().size() > 0 &&
                &new_scheduled[j]->get_inner_operators()[0].get() == join_op) {
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
        } else if (new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::LEFT_DELIM_JOIN) {
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
        } else if (new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::INVALID) {
          auto& partition =
            new_scheduled[i]->get_sink()->Cast<op::sirius_physical_partition>();
          std::string_view port_id = partition.is_build_partition() ? "build" : "default";

          if (partition.is_build_partition()) {
            // For build partitions, no pipeline uses it as source.
            // Instead, connect directly to the HASH_JOIN operator stored in parent_op.
            // Find the pipeline containing this HASH_JOIN as the first operator.
            op::sirius_physical_operator* hash_join_op = partition.get_parent_op();
            bool found                                         = false;
            for (size_t j = 0; j < new_scheduled.size(); j++) {
              // The join is guaranteed to be the first operator in the pipeline
              if (new_scheduled[j]->get_inner_operators().size() > 0 &&
                  &new_scheduled[j]->get_inner_operators()[0].get() == hash_join_op) {
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
        } else if (new_scheduled[i]->get_sink()->type == duckdb::PhysicalOperatorType::RESULT_COLLECTOR) {
          std::string_view port_id = "final";
          size_t sink_op_id        = get_operator_id(new_scheduled[i]->get_sink().get());
          data_repo_manager->add_new_repository(
            sink_op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
          new_scheduled[i]->sink->add_port(
            port_id,
            std::make_unique<op::sirius_physical_operator::port>(
              op::MemoryBarrierType::FULL,
              data_repo_manager->get_repository(sink_op_id, port_id).get(),
              new_scheduled[i],
              nullptr));
        } else {
          throw std::runtime_error("Unsupported sink type for modified pipeline");
        }

        if (new_scheduled[i]->get_source()->type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
          std::unique_ptr<::cucascade::shared_data_repository> repo =
            std::make_unique<::cucascade::shared_data_repository>();
          std::string port_id = "scan";
          auto next_op        = new_scheduled[i]->get_inner_operators().size() == 0
                                  ? new_scheduled[i]->get_sink().get()
                                  : &new_scheduled[i]->get_inner_operators()[0].get();
          size_t op_id        = get_operator_id(next_op);
          data_repo_manager->add_new_repository(op_id, port_id, std::move(repo));
          next_op->add_port(port_id,
                            std::make_unique<op::sirius_physical_operator::port>(
                              op::MemoryBarrierType::PIPELINE,
                              data_repo_manager->get_repository(op_id, port_id).get(),
                              nullptr,
                              new_scheduled[i]));
        }
      }

      SIRIUS_LOG_DEBUG("Final Scheduled pipelines: {}", new_scheduled.size());
      for (size_t i = 0; i < new_scheduled.size(); i++) {
        auto pipeline = new_scheduled[i];
        SIRIUS_LOG_DEBUG("Source {}", pipeline->get_source()->get_name());
        for (size_t j = 0; j < pipeline->get_inner_operators().size(); j++) {
          SIRIUS_LOG_DEBUG(" Op {}", pipeline->get_inner_operators()[j].get().get_name());
        }
        if (pipeline->get_sink()->type == duckdb::PhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto delim_join = pipeline->get_sink();
          auto partition_join =
            delim_join->Cast<op::sirius_physical_delim_join>().partition_join;
          auto partition_distinct =
            delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
          {
            std::string msg =
              "Sink " + pipeline->get_sink()->get_name() + " partition join next op after sink: ";
            for (auto next_port : partition_join->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
          {
            std::string msg =
              "Sink " + pipeline->get_sink()->get_name() + " partition distinct next op after sink: ";
            for (auto next_port : partition_distinct->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
        } else if (pipeline->get_sink()->type == duckdb::PhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto delim_join = pipeline->get_sink();
          auto column_data_scan =
            delim_join->Cast<op::sirius_physical_delim_join>().join->children[0].get();
          auto partition_distinct =
            delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
          {
            std::string msg =
              "Sink " + pipeline->get_sink()->get_name() + " column data scan next op after sink: ";
            for (auto next_port : column_data_scan->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
          {
            std::string msg =
              "Sink " + pipeline->get_sink()->get_name() + " partition distinct next op after sink: ";
            for (auto next_port : partition_distinct->get_next_port_after_sink()) {
              msg += next_port.first->get_name() + " ";
            }
            SIRIUS_LOG_DEBUG("{}", msg);
          }
        } else if (pipeline->get_sink()->type == duckdb::PhysicalOperatorType::HASH_GROUP_BY ||
                   pipeline->get_sink()->type == duckdb::PhysicalOperatorType::ORDER_BY ||
                   pipeline->get_sink()->type == duckdb::PhysicalOperatorType::TOP_N ||
                   pipeline->get_sink()->type == duckdb::PhysicalOperatorType::UNGROUPED_AGGREGATE ||
                   pipeline->get_sink()->type == duckdb::PhysicalOperatorType::INVALID ||
                   pipeline->get_sink()->type == duckdb::PhysicalOperatorType::CTE) {
          std::string msg = "Sink " + pipeline->get_sink()->get_name() + " next op after sink: ";
          for (auto next_port : pipeline->get_sink()->get_next_port_after_sink()) {
            msg += next_port.first->get_name() + " ";
          }
          SIRIUS_LOG_DEBUG("{}", msg);
        } else {
          SIRIUS_LOG_DEBUG("Sink {}", pipeline->get_sink()->get_name());
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
