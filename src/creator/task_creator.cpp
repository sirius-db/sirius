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

#include "creator/task_creator.hpp"

#include "log/logging.hpp"
#include "op/scan/cpu_source_task.hpp"
#include "op/scan/duckdb_native_metadata.hpp"
#include "op/scan/duckdb_native_scan_task.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/iceberg_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_iceberg_scan.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "planner/query.hpp"
#include "sirius_context.hpp"

#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace sirius::creator {

namespace {

//------------------------------------------------------------------------------
// Default-deny escape gates for the DuckDB-native scan path.
//
// duckdb_native_scan bypasses DuckDB's `init_global` entirely, so it inherits
// none of the implicit handling (sampling pushdown, virtual-column synthesis,
// dynamic-filter consumption) that the standard table function gives the
// CPU and parquet paths. The CPU path's `duckdb_scan_task` either forwards
// these to `init_global` or throws on them; we mirror that behavior here by
// refusing the GPU-native fast path when any non-default surface is present.
//
// Any future LogicalGet field that represents a pushdown should add a check
// here so silent wrong-results don't sneak in across DuckDB versions.
//
// Returned string is empty when accepted; non-empty otherwise (logged).
//------------------------------------------------------------------------------
std::string check_native_scan_operator_gates(op::sirius_physical_duckdb_scan const& scan_op)
{
  // (1) Dynamic join filters. CPU path throws NotImplementedException; native
  // path historically silently ignored — that's a wrong-results bug because
  // Sirius does not run DuckDB's JoinFilterPushdown operator above the scan.
  if (scan_op.dynamic_filters) {
    return "dynamic_filters set (Sirius does not consume DynamicTableFilterSet)";
  }

  // (2) TABLESAMPLE pushed into scan via sampling_pushdown=true. CPU path
  // forwards `extra_info.sample_options` to init_global so DuckDB samples
  // correctly. Native bypasses init_global, so this would scan all rows and
  // return too many.
  if (scan_op.extra_info.sample_options) {
    return "extra_info.sample_options present (sampling pushed into scan)";
  }

  // (3) Virtual columns other than rowid. Rowid is synthesized on GPU via
  // per-RG (start, count) ranges; non-rowid virtual columns (e.g.
  // COLUMN_IDENTIFIER_EMPTY) we don't synthesize, so refuse those.
  for (auto const& col_id : scan_op.column_ids) {
    if (col_id.IsRowIdColumn()) { continue; }
    if (col_id.GetPrimaryIndex() == duckdb::COLUMN_IDENTIFIER_EMPTY) {
      return "non-rowid virtual column projection";
    }
  }

  return {};
}

}  // namespace

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(exec::thread_pool_config config,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr)
  : _running(false), _config(config), _mem_res_mgr(mem_res_mgr)
{
}

task_creator::~task_creator() { stop(); }

void task_creator::set_client_context(::duckdb::ClientContext& client_context)
{
  _client_context = std::addressof(client_context);
  _thread_context = std::make_unique<duckdb::ThreadContext>(client_context);
  _execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_context, *_thread_context, nullptr);
}

void task_creator::set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler)
{
  _task_scheduler = &task_scheduler;
}

void task_creator::prepare_for_query(const sirius::planner::query& query)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);

  _scan_operator_global_state_map.clear();
  _duckdb_native_scan_global_state_map.clear();
  _gpu_operator_global_state_map.clear();

  const auto& pipelines = query.get_pipelines();
  for (const auto& pipeline : pipelines) {
    // Give each pipeline a pointer to this task_creator so that when a pipeline
    // finishes (including via downstream notification), it can schedule output consumers.
    pipeline->set_task_creator(this);

    auto source_operator = pipeline->get_source();
    if (source_operator == nullptr) { continue; }

    size_t operator_id = source_operator->get_operator_id();

    if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      auto* scan_op = &source_operator->Cast<op::sirius_physical_duckdb_scan>();

      // Try the GPU-native fast path. Falls back to standard duckdb_scan_task
      // when any escape gate fires, when the metadata walk reports unsupported
      // compression, or when SIRIUS_DISABLE_GPU_NATIVE_SCAN is set.
      bool native_dispatched     = false;
      char const* disable_native = std::getenv("SIRIUS_DISABLE_GPU_NATIVE_SCAN");
      if (disable_native == nullptr || disable_native[0] == '0') {
        auto gate_failure = check_native_scan_operator_gates(*scan_op);
        if (!gate_failure.empty()) {
          SIRIUS_LOG_INFO(
            "[task_creator] duckdb_native_scan refused (operator gate): {} — falling back",
            gate_failure);
        } else {
          // Extract projected columns + types. Rowid columns get an
          // is_rowid marker; the walker synthesizes per-RG ranges for them
          // instead of pinning segments. Non-rowid virtual columns are
          // refused above so every other entry is a real storage column.
          std::vector<op::scan::projected_column> projected_cols;
          std::vector<sirius::logical_type> projected_types;
          auto add_col = [&](std::size_t column_ids_idx, std::size_t scanned_types_idx) {
            auto const& col_id = scan_op->column_ids[column_ids_idx];
            op::scan::projected_column pc;
            pc.is_rowid    = col_id.IsRowIdColumn();
            pc.storage_idx = pc.is_rowid ? duckdb::StorageIndex(0)
                                         : duckdb::StorageIndex(col_id.GetPrimaryIndex());
            projected_cols.push_back(pc);
            projected_types.push_back(scan_op->scanned_types[scanned_types_idx]);
          };
          if (!scan_op->projection_ids.empty()) {
            projected_cols.reserve(scan_op->projection_ids.size());
            projected_types.reserve(scan_op->projection_ids.size());
            for (size_t i = 0; i < scan_op->projection_ids.size(); ++i) {
              add_col(scan_op->projection_ids[i], i);
            }
          } else {
            projected_cols.reserve(scan_op->column_ids.size());
            projected_types.reserve(scan_op->column_ids.size());
            for (size_t ci = 0; ci < scan_op->column_ids.size(); ++ci) {
              add_col(ci, ci);
            }
          }

          // Walk row groups and metadata via the Level-1 free function.
          auto& bind_data = scan_op->bind_data->Cast<duckdb::TableScanBindData>();
          auto& table     = bind_data.table.Cast<duckdb::DuckTableEntry>();
          auto& storage   = table.GetStorage();

          std::vector<duckdb::RowGroup*> all_row_groups;
          std::vector<duckdb::idx_t> all_row_group_starts;
          {
            auto& rg_collection = *storage.GetRowGroupCollectionRef();
            auto rg_tree        = rg_collection.GetRowGroupsDirect();
            auto rg_node        = rg_tree->GetRootSegment();
            while (rg_node) {
              all_row_groups.push_back(&rg_node->GetNode());
              // GetRowStart() is on the segment-tree node wrapper, not on
              // RowGroup itself — capture here while we have the wrapper.
              all_row_group_starts.push_back(rg_node->GetRowStart());
              rg_node = rg_tree->GetNextSegment(*rg_node);
            }
          }

          auto metadata = op::scan::walk_duckdb_native_metadata(storage,
                                                                projected_cols,
                                                                projected_types,
                                                                all_row_groups,
                                                                all_row_group_starts,
                                                                *_client_context);

          if (metadata.viable) {
            auto state = std::make_shared<op::scan::duckdb_native_scan_global_state>(
              pipeline, *_task_scheduler, *_client_context, scan_op, std::move(metadata));
            if (state->viable()) {
              _duckdb_native_scan_global_state_map.emplace(operator_id, std::move(state));
              native_dispatched = true;
            } else {
              SIRIUS_LOG_INFO(
                "[task_creator] duckdb_native_scan global-state init non-viable: {} — falling back",
                state->viability_failure_reason());
            }
          } else {
            SIRIUS_LOG_INFO(
              "[task_creator] duckdb_native_scan metadata non-viable: {} — falling back",
              metadata.viability_failure_reason);
          }
        }
      }

      if (!native_dispatched) {
        _scan_operator_global_state_map.emplace(
          operator_id,
          std::make_shared<op::scan::duckdb_scan_task_global_state>(
            pipeline, *_task_scheduler, *_client_context, scan_op));
      }
    } else if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::CPU_SOURCE) {
      _cpu_source_operator_global_state_map.emplace(
        operator_id,
        std::make_shared<op::scan::cpu_source_task_global_state>(
          pipeline, &source_operator->Cast<op::sirius_physical_cpu_source>()));
    } else if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::ICEBERG_SCAN) {
      SIRIUS_LOG_INFO("[task_creator::prepare_for_query] ICEBERG_SCAN operator_id={}", operator_id);
      auto it = _parquet_scan_operator_global_state_map.find(operator_id);
      if (it != _parquet_scan_operator_global_state_map.end()) {
        SIRIUS_LOG_INFO("[task_creator::prepare_for_query] rebind existing state for id={}",
                        operator_id);
        it->second->rebind(pipeline, &source_operator->Cast<op::sirius_physical_iceberg_scan>());
      } else {
        SIRIUS_LOG_INFO("[task_creator::prepare_for_query] creating NEW state for id={}",
                        operator_id);
        const auto& op_params =
          _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
            ->get_config()
            .get_operator_params();
        _parquet_scan_operator_global_state_map.emplace(
          operator_id,
          std::make_shared<op::scan::iceberg_scan_task_global_state>(
            pipeline,
            &source_operator->Cast<op::sirius_physical_iceberg_scan>(),
            op_params.scan_task_batch_size));
      }
    } else {
      auto gs = std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline);
      _gpu_operator_global_state_map.emplace(operator_id, std::move(gs));
    }
  }
  _num_scans_in_plan = _scan_operator_global_state_map.size() +
                       _duckdb_native_scan_global_state_map.size() +
                       _parquet_scan_operator_global_state_map.size();
}

void task_creator::drain_pending_tasks()
{
  // Drain any queued task creation requests that haven't been picked up yet
  _task_creation_queue.drain();
  // Wait for any in-flight task creation lambdas to finish
  _bounded_pool->wait_all();
}

void task_creator::reset(bool keep_parquet_metadata)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _scan_operator_global_state_map.clear();
  _duckdb_native_scan_global_state_map.clear();
  if (!keep_parquet_metadata) { _parquet_scan_operator_global_state_map.clear(); }
  _gpu_operator_global_state_map.clear();
  _cpu_source_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node)
{
  if (node == nullptr) { return nullptr; }

  if (node->type == ::sirius::op::SiriusPhysicalOperatorType::ICEBERG_SCAN) {
    size_t operator_id             = node->get_operator_id();
    auto parquet_task_global_state = _parquet_scan_operator_global_state_map.at(operator_id);
    if (parquet_task_global_state->has_more_partitions()) {
      return node;
    } else {
      return nullptr;
    }
  }
  auto hint = node->get_next_task_hint();

  if (hint.has_value() && hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    // WSM TODO: how do we handle other ports that are not default?
    return hint.value().producer;
  } else if (hint.has_value() &&
             hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    auto* producer = hint.value().producer;
    // DuckDB scan tasks create their own continuations internally, so the
    // task creator should never schedule additional scans from downstream.
    // (Parquet scans are fine — they use partition indices that self-limit.)
    if (producer != nullptr && producer->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      // GPU-native scan self-manages via continuations once started. Until
      // it has started (i.e., before the initial fan-out is dispatched),
      // let the request through so manager_loop creates the first task.
      auto native_it = _duckdb_native_scan_global_state_map.find(producer->get_operator_id());
      if (native_it != _duckdb_native_scan_global_state_map.end() &&
          native_it->second->has_started()) {
        return nullptr;
      }
      auto cpu_it = _scan_operator_global_state_map.find(producer->get_operator_id());
      if (cpu_it != _scan_operator_global_state_map.end()) {
        if (cpu_it->second->is_source_drained() || !cpu_it->second->can_create_more_tasks()) {
          return nullptr;
        }
      }
    }
    return get_operator_for_next_task(producer);
  }
  return nullptr;
}

void task_creator::stop()
{
  _task_creation_queue.interrupt();
  stop_thread_pool();
}

void task_creator::start_thread_pool()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _bounded_pool = std::make_unique<exec::bounded_thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
  _manager_thread = std::thread(&task_creator::manager_loop, this);
}

void task_creator::stop_thread_pool()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _bounded_pool->interrupt();
  _task_creation_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
  _bounded_pool->stop();
  _bounded_pool.reset();
}

void task_creator::schedule(op::sirius_physical_operator* node)
{
  auto request  = std::make_unique<task_creation_request>();
  request->node = node;
  _task_creation_queue.push(std::move(request));
}

void task_creator::manager_loop()
{
  while (_running.load()) {
    auto slot = _bounded_pool->reserve();  // block till a thread is available
    if (!slot) {
      SIRIUS_LOG_INFO("Task Creator: pool interrupted, stopping manager loop");
      break;
    }

    auto request = _task_creation_queue.pop();
    if (!request) {
      SIRIUS_LOG_INFO("Task Creator: task queue interrupted, stopping manager loop");
      break;
    }

    auto node = request->node;
    if (node == nullptr) { continue; }

    node = get_operator_for_next_task(node);

    if (node == nullptr) { continue; }

    // Dispatch the task creation work to the pool
    _bounded_pool->dispatch(std::move(slot), [this, node]() mutable {
      try {
        // Get what we need to create the task
        auto pipeline = node->get_pipeline();
        std::vector<cucascade::shared_data_repository*> destination_data_repositories;
        // special handling for delim joins
        if (pipeline->get_sink()->type ==
            ::sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto& delim_join    = pipeline->get_sink()->Cast<op::sirius_physical_right_delim_join>();
          auto partition_join = delim_join.partition_join;
          auto distinct_op    = delim_join.distinct.get();
          for (auto& next_port : partition_join->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
          for (auto& next_port : distinct_op->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
        } else if (pipeline->get_sink()->type ==
                   ::sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto& delim_join      = pipeline->get_sink()->Cast<op::sirius_physical_left_delim_join>();
          auto distinct_op      = delim_join.distinct.get();
          auto column_data_scan = delim_join.column_data_scan;
          for (auto& next_port : column_data_scan->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
          for (auto& next_port : distinct_op->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
        } else {
          for (auto& next_port : pipeline->get_sink()->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
        }
        // scheduling scan task
        if (node->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
          size_t operator_id = node->get_operator_id();
          if (destination_data_repositories.empty()) {
            throw std::runtime_error(
              "No destination data repositories provided for scan task creation.");
          }

          // GPU-native fast path: launch N initial tasks (N = scan executor
          // thread count); each task self-continues until row groups exhaust.
          auto native_it = _duckdb_native_scan_global_state_map.find(operator_id);
          if (native_it != _duckdb_native_scan_global_state_map.end()) {
            auto& native_state = native_it->second;
            int num_initial    = _task_scheduler->get_scan_executor().get_num_threads();
            num_initial = std::min(num_initial, static_cast<int>(native_state->total_row_groups()));
            num_initial = std::max(num_initial, 1);
            for (int i = 0; i < num_initial; ++i) {
              auto scan_task = std::make_unique<op::scan::duckdb_native_scan_task>(
                get_next_task_id(), destination_data_repositories[0], native_state);
              pipeline->mark_task_created();
              _task_scheduler->schedule(std::move(scan_task));
            }
          } else {
            // Fallback: standard duckdb_scan_task (CPU path, materializes
            // through DuckDB's table function).
            auto scan_task_global_state = _scan_operator_global_state_map.at(operator_id);

            const auto& op_params =
              _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
                ->get_config()
                .get_operator_params();
            auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
              *scan_task_global_state,
              *_execution_context,
              op_params.scan_task_batch_size,
              op_params.default_scan_task_varchar_size);
            auto scan_task = std::make_unique<op::scan::duckdb_scan_task>(
              get_next_task_id(),
              destination_data_repositories[0],  // WSM amin TODO: multiple destinations?
              std::move(scan_task_local_state),
              scan_task_global_state);
            pipeline->mark_task_created();  // WSM TODO: atomic with task creation
            _task_scheduler->schedule(std::move(scan_task));
          }
        } else if (node->type == ::sirius::op::SiriusPhysicalOperatorType::ICEBERG_SCAN) {
          size_t operator_id             = node->get_operator_id();
          auto parquet_task_global_state = _parquet_scan_operator_global_state_map.at(operator_id);
          // ICEBERG_SCAN inherits from PARQUET_SCAN; Cast<> is type-checked by enum so use
          // static_cast for iceberg nodes.
          auto* parquet_scan = (node->type == op::SiriusPhysicalOperatorType::ICEBERG_SCAN)
                                 ? static_cast<op::sirius_physical_parquet_scan*>(node)
                                 : &node->Cast<op::sirius_physical_parquet_scan>();
          while (true) {
            pipeline->mark_task_created();
            auto partition = parquet_task_global_state->claim_next_rg_partition();
            if (!partition.has_value()) {
              pipeline->mark_task_completed();
              if (pipeline->is_pipeline_finished()) {
                auto output_consumers = pipeline->get_output_consumers();
                for (auto& output_consumer : output_consumers) {
                  schedule(output_consumer);
                }
              }
              return;
            }
            if (!parquet_task_global_state->has_more_partitions()) {
              parquet_scan->has_more_partitions = false;
            }

            auto parquet_task_local_state =
              std::make_unique<op::scan::parquet_scan_task_local_state>(*parquet_task_global_state,
                                                                        *partition);

            if (destination_data_repositories.empty()) {
              throw std::runtime_error(
                "No destination data repositories provided for parquet scan task creation.");
            }
            auto parquet_task =
              std::make_unique<op::scan::parquet_scan_task>(get_next_task_id(),
                                                            destination_data_repositories[0],
                                                            std::move(parquet_task_local_state),
                                                            parquet_task_global_state);
            _task_scheduler->schedule(std::move(parquet_task));

            // If there is only a single scan in the plan, continue creating scan tasks to create
            // I/O parallelism. Otherwise, let the plan drive the creation of more tasks.
            if (_num_scans_in_plan >= 2) { break; }
          }
        } else if (node->type == ::sirius::op::SiriusPhysicalOperatorType::CPU_SOURCE) {
          SIRIUS_LOG_DEBUG("Task Creator: creating cpu_source_task for operator {}",
                           node->get_name());
          size_t operator_id     = node->get_operator_id();
          auto cpu_source_global = _cpu_source_operator_global_state_map.at(operator_id);

          pipeline->mark_task_created();

          if (destination_data_repositories.empty()) {
            throw std::runtime_error(
              "No destination data repositories provided for cpu source task creation.");
          }

          auto local_state = std::make_unique<op::scan::cpu_source_task_local_state>();
          auto task        = std::make_unique<op::scan::cpu_source_task>(get_next_task_id(),
                                                                  destination_data_repositories[0],
                                                                  std::move(local_state),
                                                                  cpu_source_global,
                                                                  *_client_context);
          SIRIUS_LOG_DEBUG("Task Creator: scheduling cpu_source_task, dest_repos={}",
                           destination_data_repositories.size());
          _task_scheduler->schedule(std::move(task));
        } else {
          // need to exhaust input batches until all ports are empty
          while (!node->all_ports_empty()) {
            // Mark task created BEFORE popping data from ports to prevent a race
            // condition where update_pipeline_status() sees empty ports and matching
            // task counters, prematurely marking the pipeline as finished.
            pipeline->mark_task_created();

            auto input_data = node->get_next_task_input_data();
            auto* pipelineable_input =
              dynamic_cast<op::pipelineable_operator_data*>(input_data.get());
            if (!input_data ||
                (pipelineable_input && pipelineable_input->get_data_batches().empty())) {
              // No data was available (e.g., another thread already consumed it).
              // Balance the counter. mark_task_completed() calls update_pipeline_status()
              // which is correct: if all ports are truly empty and all real tasks have
              // completed, the pipeline should finish.
              pipeline->mark_task_completed();
              if (pipeline->is_pipeline_finished()) {
                auto output_consumers = pipeline->get_output_consumers();
                for (auto& output_consumer : output_consumers) {
                  this->schedule(output_consumer);
                }
              }
              break;
            }

            // Check to see if you need to create a new global state for this operator
            size_t operator_id                  = node->get_operator_id();
            auto gpu_pipeline_task_global_state = _gpu_operator_global_state_map.at(operator_id);

            auto local_state =
              std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(input_data));
            auto task =
              std::make_unique<pipeline::gpu_pipeline_task>(get_next_task_id(),
                                                            destination_data_repositories,
                                                            std::move(local_state),
                                                            gpu_pipeline_task_global_state);
            _task_scheduler->schedule(std::move(task));
          }
        }
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("Task Creator: Exception during task creation: {}", e.what());
        _task_scheduler->terminate_query(std::current_exception());
        stop();
      }
    });
  }
}

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
