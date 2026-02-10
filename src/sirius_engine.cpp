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

#include "sirius_engine.hpp"

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
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"

#include <cucascade/data/data_repository_manager.hpp>
#include <stdio.h>

namespace sirius {

void sirius_engine::reset()
{
  sirius_physical_plan = nullptr;
  sirius_owned_plan.reset();
  sirius_root_pipelines.clear();
  root_pipeline_idx = 0;
  total_pipelines   = 0;
  sirius_pipelines.clear();
  new_pipeline_breakers.clear();
  sirius_scheduled.clear();
  new_scheduled.clear();
}

void sirius_engine::insert_repository(
  std::string_view port_id,
  duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
  duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline)
{
  auto next_op            = dependent_pipeline->get_operators().size() == 0
                              ? dependent_pipeline->get_sink().get()
                              : &dependent_pipeline->get_operators()[0].get();
  size_t op_id            = next_op->operator_id;
  auto& data_repo_manager = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                              ->get_data_repository_manager();
  data_repo_manager.add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<op::sirius_physical_operator::port>(
                      op::MemoryBarrierType::FULL,
                      data_repo_manager.get_repository(op_id, port_id).get(),
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
  auto& data_repo_manager = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                              ->get_data_repository_manager();
  auto next_op = dependent_pipeline->get_operators().size() == 0
                   ? dependent_pipeline->get_sink().get()
                   : &dependent_pipeline->get_operators()[0].get();
  size_t op_id = next_op->operator_id;
  data_repo_manager.add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<op::sirius_physical_operator::port>(
                      op::MemoryBarrierType::FULL,
                      data_repo_manager.get_repository(op_id, port_id).get(),
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
  for (auto current_op : current.get_operators()) {
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
  if (sirius_physical_plan.get() == NULL)
    throw duckdb::InvalidInputException("sirius_physical_plan is NULL");
  auto& result_collector =
    sirius_physical_plan.get()->Cast<op::sirius_physical_materialized_collector>();
  result_collector.sink_state = result_collector.get_global_sink_state(context);
  duckdb::unique_ptr<duckdb::QueryResult> res =
    result_collector.get_result(*(result_collector.sink_state));
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
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    throw duckdb::InvalidInputException("Sirius context is not initialized.");
  }

  // Create the query with the pipeline hashmap
  sirius_pipeline_hashmap pipeline_map(new_scheduled);
  sirius_ctx->create_query(std::move(pipeline_map));
  auto future = sirius_ctx->get_pipeline_executor().start_query();
  try {
    future.get();
  } catch (const std::exception& e) {
    /// todo(bobbi) we should handle the error properly, clean the query context and then return the
    /// error to duckdb
    SIRIUS_LOG_ERROR("Error executing query: {}", e.what());
    throw;
  } catch (...) {
    SIRIUS_LOG_ERROR("Unknown error executing query");
    throw;
  }
}

duckdb::unique_ptr<op::sirius_physical_operator> sirius_engine::construct_sirius_specific_operator(
  op::sirius_physical_operator* op)
{
  if (op->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& scan_physical_op = op->Cast<op::sirius_physical_table_scan>();
    return duckdb::make_uniq<op::sirius_physical_duckdb_scan>(&scan_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    auto& group_by_physical_op = op->Cast<op::sirius_physical_grouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_grouped_aggregate_merge>(&group_by_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::ORDER_BY) {
    auto& order_by_physical_op = op->Cast<op::sirius_physical_order>();
    return duckdb::make_uniq<op::sirius_physical_merge_sort>(&order_by_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::TOP_N) {
    auto& topn_physical_op = op->Cast<op::sirius_physical_top_n>();
    return duckdb::make_uniq<op::sirius_physical_top_n_merge>(&topn_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
    auto& ungrouped_agg_physical_op = op->Cast<op::sirius_physical_ungrouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_ungrouped_aggregate_merge>(
      &ungrouped_agg_physical_op);
  } else {
    throw duckdb::InternalException("Unsupported operator type" +
                                    SiriusPhysicalOperatorToString(op->type) +
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
          if (pipeline_inside[pipeline_idx]->source->type ==
              op::SiriusPhysicalOperatorType::HASH_JOIN) {
            auto& temp =
              pipeline_inside[pipeline_idx]->source.get()->Cast<op::sirius_physical_hash_join>();
            if (temp.join_type == duckdb::JoinType::RIGHT ||
                temp.join_type == duckdb::JoinType::RIGHT_SEMI ||
                temp.join_type == duckdb::JoinType::RIGHT_ANTI) {
              // if (!duckdb::Config::MODIFIED_PIPELINE) sirius_scheduled.push_back(pipeline);
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

    // get data_repo_manager from sirius context
    auto& data_repo_manager = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                                ->get_data_repository_manager();
    std::unordered_map<const op::sirius_physical_operator*,
                       duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>>
      source_to_pipelines;

    for (size_t i = 0; i < copied_scheduled.size(); i++) {
      auto current_pipeline = copied_scheduled[i];  // Copy duckdb::shared_ptr to avoid invalidation

      // Store original dependencies to preserve them
      auto original_dependencies = current_pipeline->dependencies;

      if (current_pipeline->source->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
        auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        auto scan_op      = current_pipeline->get_source();
        auto new_scan_op  = construct_sirius_specific_operator(scan_op.get());

        // todo(bobbi) currently this can be set to any operator since it's never used, and now we
        // set it to scan_op
        new_pipeline->source = scan_op.get();
        new_pipeline->sink   = new_scan_op.get();

        current_pipeline->source = new_scan_op.get();
        // move scan_op to current_pipeline.operator[0], current_pipeline.operator[0] to
        // current_pipeline.operator[1], ...
        current_pipeline->operators.insert(current_pipeline->operators.begin(), *scan_op);
        current_pipeline->dependencies.push_back(new_pipeline);

        new_scheduled.push_back(new_pipeline);
        new_pipeline_breakers.push_back(std::move(new_scan_op));
      }

      duckdb::vector<duckdb::idx_t> join_positions;
      duckdb::shared_ptr<pipeline::sirius_pipeline> previous_pipeline = nullptr;
      op::sirius_physical_concat* prev_concat_ptr                     = nullptr;

      for (duckdb::idx_t op_idx = 0; op_idx < current_pipeline->operators.size(); op_idx++) {
        if (current_pipeline->operators[op_idx].get().type ==
              op::SiriusPhysicalOperatorType::HASH_JOIN ||
            current_pipeline->operators[op_idx].get().type ==
              op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
          join_positions.push_back(op_idx);
        }
      }

      bool group_agg_sort_topn_sink = false;
      if (current_pipeline->sink->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY ||
          current_pipeline->sink->type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
        group_agg_sort_topn_sink = true;
      }

      bool order_by_sink =
        (current_pipeline->sink->type == op::SiriusPhysicalOperatorType::ORDER_BY);

      bool top_n_sink = (current_pipeline->sink->type == op::SiriusPhysicalOperatorType::TOP_N);

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
            static_cast<op::sirius_physical_partition*>(new_pipeline_breakers.back().get());
          // Create new pipeline: PARTITION -> HASH_JOIN -> ... -> SINK
          auto new_pipeline  = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
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
          duckdb::unique_ptr<op::sirius_physical_concat> concat_op =
            make_uniq<op::sirius_physical_concat>(partition_ptr->types,
                                                  partition_ptr->estimated_cardinality,
                                                  &current_pipeline->operators[join_pos].get(),
                                                  false);
          more_new_pipeline->source = partition_ptr;
          more_new_pipeline->sink   = concat_op.get();
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
          previous_pipeline = more_new_pipeline;
          prev_concat_ptr   = concat_ptr;
        }
      }

      if (join_sink) {
        // replace hash join sink with partition
        duckdb::unique_ptr<op::sirius_physical_partition> partition_op;
        auto hash_join_op = current_pipeline->get_sink();
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
        duckdb::unique_ptr<op::sirius_physical_concat> concat_op =
          make_uniq<op::sirius_physical_concat>(
            partition_ptr->types, partition_ptr->estimated_cardinality, hash_join_op.get(), true);
        new_pipeline->source = partition_ptr;
        new_pipeline->sink   = concat_op.get();
        new_pipeline->dependencies.push_back(current_pipeline);

        new_scheduled.push_back(new_pipeline);

        new_pipeline_breakers.push_back(std::move(partition_op));
        new_pipeline_breakers.push_back(std::move(concat_op));
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
        auto merge_op      = construct_sirius_specific_operator(group_sort_topn.get());
        new_pipeline->sink = merge_op.get();

        for (int j = i + 1; j < copied_scheduled.size(); j++) {
          if (copied_scheduled[j]->source.get() == group_sort_topn.get()) {
            copied_scheduled[j]->source = merge_op.get();
          }
        }
        new_pipeline->dependencies.push_back(current_pipeline);
        new_scheduled.push_back(new_pipeline);
        new_pipeline_breakers.push_back(std::move(merge_op));
      }

      if (order_by_sink) {
        auto order_op   = current_pipeline->sink;
        auto* order_ptr = static_cast<op::sirius_physical_order*>(order_op.get());

        // Save the original projection and replace with identity so ORDER outputs all columns.
        // Sort keys must remain in the output for SORT_SAMPLE and SORT_PARTITION to reference.
        // MERGE_SORT will apply the final projection.
        auto original_projections = order_ptr->projections;
        {
          auto& child_types = current_pipeline->operators.size() > 0
                                ? current_pipeline->operators.back().get().types
                                : current_pipeline->source->types;
          duckdb::vector<duckdb::idx_t> identity_proj;
          for (duckdb::idx_t i = 0; i < child_types.size(); i++) {
            identity_proj.push_back(i);
          }
          order_ptr->projections = std::move(identity_proj);
          order_ptr->types       = child_types;
        }

        // Pipeline A: current pipeline keeps ORDER as sink (local sort per batch)
        new_scheduled.push_back(current_pipeline);

        // Create SORT_SAMPLE operator
        auto sample_op = duckdb::unique_ptr<op::sirius_physical_sort_sample>(
          new op::sirius_physical_sort_sample(order_ptr));
        auto* sample_ptr = sample_op.get();
        if (duckdb::Config::MAX_SORT_PARTITION_BYTES > 0) {
          sample_ptr->set_max_partition_bytes(duckdb::Config::MAX_SORT_PARTITION_BYTES);
        }

        // Pipeline B: ORDER (source) → SORT_SAMPLE (sink)
        auto sample_pipeline    = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        sample_pipeline->source = order_op.get();
        sample_pipeline->sink   = sample_ptr;
        sample_pipeline->dependencies.push_back(current_pipeline);
        new_scheduled.push_back(sample_pipeline);

        // Create SORT_PARTITION operator
        auto partition_op = duckdb::unique_ptr<op::sirius_physical_sort_partition>(
          new op::sirius_physical_sort_partition(order_ptr));
        auto* partition_ptr = partition_op.get();

        // Wire sort_partition to read boundaries from sort_sample
        partition_ptr->set_sample_op(sample_ptr);

        // Pipeline C: SORT_SAMPLE (source) → SORT_PARTITION (sink)
        auto partition_pipeline    = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        partition_pipeline->source = sample_ptr;
        partition_pipeline->sink   = partition_ptr;
        partition_pipeline->dependencies.push_back(sample_pipeline);
        new_scheduled.push_back(partition_pipeline);

        // Create MERGE_SORT operator
        auto merge_op = duckdb::unique_ptr<op::sirius_physical_merge_sort>(
          new op::sirius_physical_merge_sort(order_ptr));
        auto* merge_ptr = merge_op.get();

        // If ORDER had a non-identity projection, set it as MERGE_SORT's final projection
        {
          bool is_identity = (original_projections.size() == order_ptr->types.size());
          if (is_identity) {
            for (duckdb::idx_t i = 0; i < original_projections.size(); i++) {
              if (original_projections[i] != i) {
                is_identity = false;
                break;
              }
            }
          }
          if (!is_identity) {
            duckdb::vector<duckdb::LogicalType> output_types;
            for (auto idx : original_projections) {
              output_types.push_back(order_ptr->types[idx]);
            }
            merge_ptr->set_final_projections(std::move(original_projections),
                                             std::move(output_types));
          }
        }

        // Pipeline D: SORT_PARTITION (source) → MERGE_SORT (sink)
        auto merge_pipeline    = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        merge_pipeline->source = partition_ptr;
        merge_pipeline->sink   = merge_ptr;
        merge_pipeline->dependencies.push_back(partition_pipeline);
        new_scheduled.push_back(merge_pipeline);

        // Update downstream pipelines to use MERGE_SORT as source
        for (size_t j = i + 1; j < copied_scheduled.size(); j++) {
          if (copied_scheduled[j]->source.get() == order_op.get()) {
            copied_scheduled[j]->source = merge_ptr;
          }
        }

        // Store ownership
        new_pipeline_breakers.push_back(std::move(sample_op));
        new_pipeline_breakers.push_back(std::move(partition_op));
        new_pipeline_breakers.push_back(std::move(merge_op));
      }

      if (top_n_sink) {
        auto top_n_op  = current_pipeline->sink;
        auto* topn_ptr = static_cast<op::sirius_physical_top_n*>(top_n_op.get());

        // Pipeline A: current pipeline keeps TOP_N as sink
        new_scheduled.push_back(current_pipeline);

        // Create MERGE_TOP_N operator
        auto merge_op = duckdb::unique_ptr<op::sirius_physical_top_n_merge>(
          new op::sirius_physical_top_n_merge(topn_ptr));
        auto* merge_ptr = merge_op.get();

        // Pipeline B: TOP_N (source) → MERGE_TOP_N (sink)
        auto merge_pipeline    = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
        merge_pipeline->source = top_n_op.get();
        merge_pipeline->sink   = merge_ptr;
        merge_pipeline->dependencies.push_back(current_pipeline);
        new_scheduled.push_back(merge_pipeline);

        // Update downstream pipelines to use MERGE_TOP_N as source
        for (size_t j = i + 1; j < copied_scheduled.size(); j++) {
          if (copied_scheduled[j]->source.get() == top_n_op.get()) {
            copied_scheduled[j]->source = merge_ptr;
          }
        }

        // Store ownership
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
          duckdb::unique_ptr<op::sirius_physical_concat> concat_op =
            make_uniq<op::sirius_physical_concat>(
              partition_join.get()->types,
              partition_join.get()->estimated_cardinality,
              join_op.get(),
              delim_join->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
          new_pipeline->source = partition_join.get();
          new_pipeline->sink   = concat_op.get();
          new_pipeline->dependencies.push_back(current_pipeline);

          new_pipeline_breakers.push_back(std::move(partition_join));
          new_pipeline_breakers.push_back(std::move(concat_op));
          new_scheduled.push_back(new_pipeline);
        }

        auto merge_distinct_op = construct_sirius_specific_operator(distinct_op.get());

        // Create new pipeline: PARTITION -> SINK
        auto new_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);

        new_pipeline->source = partition_distinct.get();
        new_pipeline->sink   = merge_distinct_op.get();

        for (int j = i + 1; j < copied_scheduled.size(); j++) {
          if (copied_scheduled[j]->source.get() == distinct_op.get()) {
            copied_scheduled[j]->source = merge_distinct_op.get();
          }
        }
        new_pipeline->dependencies.push_back(current_pipeline);

        new_pipeline_breakers.push_back(std::move(partition_distinct));
        new_pipeline_breakers.push_back(std::move(merge_distinct_op));
        new_scheduled.push_back(new_pipeline);
      }

      if (!group_agg_sort_topn_sink && !right_left_delim_join_sink && !join_sink &&
          !order_by_sink && !top_n_sink) {
        new_scheduled.push_back(current_pipeline);
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
        auto& cte_op             = new_scheduled[i]->get_sink()->Cast<op::sirius_physical_cte>();
        std::string_view port_id = "default";
        for (auto cte_scan : cte_op.cte_scans) {
          for (auto dependent_pipeline : source_to_pipelines[&cte_scan.get()]) {
            insert_repository(port_id, new_scheduled[i], dependent_pipeline);
          }
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
        auto delim_join     = new_scheduled[i]->get_sink();
        auto partition_join = delim_join->Cast<op::sirius_physical_delim_join>().partition_join;
        auto partition_distinct =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
        // Find the pipeline containing the join as the first operator
        bool found = false;
        for (auto dependent_pipeline : source_to_pipelines[partition_join]) {
          insert_repository("default", partition_join, new_scheduled[i], dependent_pipeline);
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
        auto& concat             = new_scheduled[i]->get_sink()->Cast<op::sirius_physical_concat>();
        std::string_view port_id = concat.is_build_concat() ? "build" : "default";

        if (concat.is_build_concat()) {
          // For build concats, no pipeline uses it as source.
          // Instead, connect directly to the HASH_JOIN operator stored in parent_op.
          // Find the pipeline containing this HASH_JOIN as the first operator.
          op::sirius_physical_operator* hash_join_op = concat.get_parent_op();
          bool found                                 = false;
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
          for (auto dependent_pipeline : source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
            insert_repository(port_id, new_scheduled[i], dependent_pipeline);
          }
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::PARTITION ||
                 new_scheduled[i]->sink->type ==
                   op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE ||
                 new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::TOP_N ||
                 new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::MERGE_SORT) {
        // Full barrier operators — wait for upstream to finish before processing
        for (auto dependent_pipeline : source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
          insert_repository("default", new_scheduled[i], dependent_pipeline);
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::ORDER_BY ||
                 new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::SORT_SAMPLE ||
                 new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::SORT_PARTITION) {
        // Pipeline barrier — sort operators process batches as they arrive
        // (sort_sample overrides get_next_task_hint to wait for N batches)
        for (auto dependent_pipeline : source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
          auto next_op             = dependent_pipeline->get_operators().size() == 0
                                       ? dependent_pipeline->get_sink().get()
                                       : &dependent_pipeline->get_operators()[0].get();
          size_t op_id             = next_op->operator_id;
          std::string_view port_id = "default";
          data_repo_manager.add_new_repository(
            op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
          next_op->add_port(port_id,
                            std::make_unique<op::sirius_physical_operator::port>(
                              op::MemoryBarrierType::PIPELINE,
                              data_repo_manager.get_repository(op_id, port_id).get(),
                              new_scheduled[i],
                              dependent_pipeline));
          new_scheduled[i]->get_sink()->add_next_port_after_sink(std::make_pair(next_op, port_id));
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
        for (auto dependent_pipeline : source_to_pipelines[new_scheduled[i]->get_sink().get()]) {
          auto next_op             = dependent_pipeline->get_operators().size() == 0
                                       ? dependent_pipeline->get_sink().get()
                                       : &dependent_pipeline->get_operators()[0].get();
          size_t op_id             = next_op->operator_id;
          std::string_view port_id = "scan";
          data_repo_manager.add_new_repository(
            op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
          next_op->add_port(port_id,
                            std::make_unique<op::sirius_physical_operator::port>(
                              op::MemoryBarrierType::PIPELINE,
                              data_repo_manager.get_repository(op_id, port_id).get(),
                              new_scheduled[i],
                              dependent_pipeline));
          new_scheduled[i]->get_sink()->add_next_port_after_sink(std::make_pair(next_op, port_id));
        }
      } else if (new_scheduled[i]->sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
        // No action needed for RESULT_COLLECTOR sinks
      } else {
        throw std::runtime_error("Unsupported sink type for modified pipeline");
      }
    }

    // Assign pipeline IDs based on new_scheduled order
    for (size_t i = 0; i < new_scheduled.size(); i++) {
      new_scheduled[i]->set_pipeline_id(i);
      new_scheduled[i]->operators.push_back(*new_scheduled[i]->sink);
      new_scheduled[i]->source = &new_scheduled[i]->operators[0].get();
      new_scheduled[i]->parents.clear();
      auto& first_op = new_scheduled[i]->operators[0].get();
      // iterate through ports at first_op
      if (new_scheduled[i]->sink.get()) {
        for (auto& [next_op, port_id] : new_scheduled[i]->sink.get()->get_next_port_after_sink()) {
          if (next_op->get_port(port_id)->dest_pipeline) {
            new_scheduled[i]->parents.push_back(duckdb::weak_ptr<sirius::pipeline::sirius_pipeline>(
              next_op->get_port(port_id)->dest_pipeline));
          }
        }
      }
    }

    // Detailed pipeline debugging information
    SIRIUS_LOG_INFO("\n=== DETAILED PIPELINE DEBUG INFO ===");
    for (size_t i = 0; i < new_scheduled.size(); i++) {
      auto pipeline = new_scheduled[i];
      SIRIUS_LOG_INFO("Pipeline #{}", i);
      SIRIUS_LOG_INFO("  Source: {}", pipeline->source->get_name());

      // Print operators
      for (size_t j = 0; j < pipeline->operators.size(); j++) {
        SIRIUS_LOG_INFO("    Operator[{}]: {}", j, pipeline->operators[j].get().get_name());
      }

      SIRIUS_LOG_INFO("  Sink: {}", pipeline->sink->get_name());

      // Print ports at operator[0] (beginning of pipeline)
      if (pipeline->operators.size() > 0) {
        auto& first_op = pipeline->operators[0].get();
        SIRIUS_LOG_INFO("  Ports at Operator[0] ({}):", first_op.get_name());

        // Check for different port types based on operator type
        if (first_op.type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
            first_op.type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
          // Joins have "default" and "build" ports
          auto* default_port = first_op.get_port("default");
          if (default_port) {
            SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                            static_cast<int>(default_port->type),
                            static_cast<void*>(default_port->repo));
          }
          auto* build_port = first_op.get_port("build");
          if (build_port) {
            SIRIUS_LOG_INFO("    Port 'build': barrier_type={}, repo={}",
                            static_cast<int>(build_port->type),
                            static_cast<void*>(build_port->repo));
          }
        } else if (first_op.type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
          // Scans have "scan" port
          auto* scan_port = first_op.get_port("scan");
          if (scan_port) {
            SIRIUS_LOG_INFO("    Port 'scan': barrier_type={}, repo={}",
                            static_cast<int>(scan_port->type),
                            static_cast<void*>(scan_port->repo));
          }
        } else if (first_op.type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
          // ignore DUCKDB_SCAN since it doesn't have port
        } else {
          // Most operators have "default" port
          auto* default_port = first_op.get_port("default");
          if (default_port) {
            SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                            static_cast<int>(default_port->type),
                            static_cast<void*>(default_port->repo));
          }
        }
      } else {
        SIRIUS_LOG_INFO("  No operators in pipeline - checking sink ports");
        auto* sink = pipeline->sink.get();

        if (sink->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
            sink->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
          auto* default_port = sink->get_port("default");
          if (default_port) {
            SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                            static_cast<int>(default_port->type),
                            static_cast<void*>(default_port->repo));
          }
          auto* build_port = sink->get_port("build");
          if (build_port) {
            SIRIUS_LOG_INFO("    Port 'build': barrier_type={}, repo={}",
                            static_cast<int>(build_port->type),
                            static_cast<void*>(build_port->repo));
          }
        } else if (sink->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
          auto* scan_port = sink->get_port("scan");
          if (scan_port) {
            SIRIUS_LOG_INFO("    Port 'scan': barrier_type={}, repo={}",
                            static_cast<int>(scan_port->type),
                            static_cast<void*>(scan_port->repo));
          }
        } else if (sink->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
          // ignore DUCKDB_SCAN since it doesn't have port
        } else {
          auto* default_port = sink->get_port("default");
          if (default_port) {
            SIRIUS_LOG_INFO("    Port 'default': barrier_type={}, repo={}",
                            static_cast<int>(default_port->type),
                            static_cast<void*>(default_port->repo));
          }
        }
      }

      // Print ports and next operators after sink
      SIRIUS_LOG_INFO("  Sink next operators and ports:");
      for (auto& next_port : pipeline->sink->get_next_port_after_sink()) {
        auto next_op = next_port.first;
        auto port_id = next_port.second;
        SIRIUS_LOG_INFO("    Next Op: {}, Port: '{}'", next_op->get_name(), port_id.data());

        // Print the port details if it exists
        auto* port = next_op->get_port(port_id);
        if (port) {
          SIRIUS_LOG_INFO("      Port barrier_type={}, repo={}",
                          static_cast<int>(port->type),
                          static_cast<void*>(port->repo));
        }
      }

      // Special handling for delim joins
      if (pipeline->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN ||
          pipeline->sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
        auto delim_join = pipeline->get_sink();

        if (pipeline->sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto partition_join = delim_join->Cast<op::sirius_physical_delim_join>().partition_join;
          SIRIUS_LOG_INFO("  Partition Join next operators:");
          for (auto& next_port : partition_join->get_next_port_after_sink()) {
            SIRIUS_LOG_INFO("    Next Op: {}, Port: '{}' Repo:'{}'",
                            next_port.first->get_name(),
                            next_port.second.data(),
                            static_cast<void*>(next_port.first->get_port(next_port.second)->repo));
          }
        }

        auto partition_distinct =
          delim_join->Cast<op::sirius_physical_delim_join>().partition_distinct;
        SIRIUS_LOG_INFO("  Partition Distinct next operators:");
        for (auto& next_port : partition_distinct->get_next_port_after_sink()) {
          SIRIUS_LOG_INFO("    Next Op: {}, Port: '{}' Repo:'{}'",
                          next_port.first->get_name(),
                          next_port.second.data(),
                          static_cast<void*>(next_port.first->get_port(next_port.second)->repo));
        }

        if (pipeline->sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto column_data_scan =
            delim_join->Cast<op::sirius_physical_delim_join>().join->children[0].get();
          SIRIUS_LOG_INFO("  Column Data Scan next operators:");
          for (auto& next_port : column_data_scan->get_next_port_after_sink()) {
            SIRIUS_LOG_INFO("    Next Op: {}, Port: '{}' Repo:'{}'",
                            next_port.first->get_name(),
                            next_port.second.data(),
                            static_cast<void*>(next_port.first->get_port(next_port.second)->repo));
          }
        }
      }

      SIRIUS_LOG_INFO("");  // Blank line between pipelines
    }
    SIRIUS_LOG_INFO("=== END DETAILED PIPELINE DEBUG INFO ===\n");

    // Flush immediately to ensure all debug output is written synchronously
    // spdlog::default_logger()->flush();

    // create invalid operators
    auto invalid_op = make_uniq<op::sirius_physical_operator>(
      op::SiriusPhysicalOperatorType::INVALID, duckdb::vector<duckdb::LogicalType>{}, 0);

    // collect all pipelines from the root pipelines (recursively) for the progress bar and verify
    // them
    root_pipeline->get_pipelines(sirius_pipelines, true);
    SIRIUS_LOG_DEBUG("total_pipelines = {}", sirius_pipelines.size());
  }
}

}  // namespace sirius
