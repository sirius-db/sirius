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

#include "duckdb/main/client_context.hpp"
#include "helper/helper.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task_executor.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_pipeline_hashmap.hpp"

#include <blockingconcurrentqueue.h>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

namespace sirius::creator {

/**
 * @brief Contains information needed to create a task.
 *
 * This class holds a reference to a sirius physical operator node and its associated
 * pipeline, which together provide the context needed for task creation.
 */
class task_creation_info {
 public:
  task_creation_info(sirius::op::sirius_physical_operator* node,
                     duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline)
    : _node(node), _pipeline(std::move(pipeline))
  {
    if (!_pipeline) {
      return;  // Skip port setup if no pipeline provided
    }
    // get next port after sink and then get the data repository from the port
    auto next_port_after_sink = _pipeline->get_sink()->get_next_port_after_sink();
    for (auto& [next_op, port_id] : next_port_after_sink) {
      destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
    }
    if (_node->type == ::duckdb::PhysicalOperatorType::TABLE_SCAN) {
      auto& first_operator = _pipeline->get_inner_operators()[0].get();
      destination_data_repositories.push_back(first_operator.get_port("scan")->repo);
    }
    if (_pipeline->get_sink()->type == ::duckdb::PhysicalOperatorType::RESULT_COLLECTOR) {
      destination_data_repositories.push_back(_node->get_port("final")->repo);
    }
  };
  ~task_creation_info() = default;
  sirius::op::sirius_physical_operator* _node;
  std::vector<cucascade::shared_data_repository*> destination_data_repositories;
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> _pipeline;
};

/**
 * @brief A thread-safe queue for managing task creation requests.
 *
 * This queue allows multiple producers to push task creation info and multiple
 * consumers to pull tasks for processing. It supports open/close semantics to
 * control when the queue accepts and returns tasks.
 */
class task_creation_queue {
 public:
  /**
   * @brief Construct a new task_creation_queue object.
   *
   * @param num_threads The number of worker threads that will consume from this queue.
   *                    Used to send sentinel values when closing the queue.
   */
  task_creation_queue(size_t num_threads);

  /**
   * @brief Opens the task queue to start accepting and returning tasks.
   */
  void open();

  /**
   * @brief Closes the task queue from accepting new tasks or returning tasks.
   *
   * This method wakes up all threads blocked in pull() by pushing nullptr sentinels.
   */
  void close();

  /**
   * @brief Push a new task creation info to be scheduled.
   *
   * @param info The task creation info to be scheduled.
   * @throws sirius::runtime_error If the scheduler is not currently accepting requests.
   */
  void push(std::unique_ptr<task_creation_info> info);

  /**
   * @brief Pull a task to execute.
   *
   * This is a blocking call that waits for a task to become available. If the queue
   * is closed and empty, it returns nullptr to signal that no more tasks will arrive.
   *
   * @return A unique pointer to the task creation info if available, nullptr if the
   *         queue is closed and empty.
   */
  std::unique_ptr<task_creation_info> pull();

  /**
   * @brief Check if the queue is currently open.
   *
   * @return true if the queue is open, false otherwise.
   */
  bool is_open() const { return _is_open.load(std::memory_order_acquire); }

 private:
  size_t _num_threads;
  duckdb_moodycamel::BlockingConcurrentQueue<std::unique_ptr<task_creation_info>> _queue;
  std::atomic<bool> _is_open{false};  ///< Whether the queue is open for pushing/pulling tasks
};

/**
 * @brief Manages the creation and scheduling of GPU pipeline tasks.
 *
 * The task_creator is responsible for creating tasks from GPU pipelines and scheduling
 * them for execution. It maintains a thread pool that processes task creation requests
 * from the task_creation_queue. The creator prioritizes table scan pipelines and uses
 * hints from operators to determine the next tasks to create.
 *
 * Usage:
 *   1. Construct with a task_creation_queue, thread count, and pipeline map.
 *   2. Call start_thread_pool() to begin processing tasks.
 *   3. Call start() to schedule initial scan pipelines.
 *   4. Call stop_thread_pool() when done.
 */
class task_creator {
 public:
  /**
   * @brief Construct a new task_creator.
   *
   * @param num_threads The number of worker threads to use.
   * @param gpu_pipeline_map A mapping of operators to their pipelines.
   * @param pipeline_executor Reference to the pipeline executor.
   * @param duckdb_scan_executor Reference to the duckdb scan executor.
   */
  task_creator(size_t num_threads,
               sirius::pipeline::pipeline_executor& pipeline_executor,
               sirius::op::scan::duckdb_scan_executor& duckdb_scan_executor,
               sirius::memory::sirius_memory_reservation_manager& mem_res_mgr);

  /**
   * @brief Destructor that ensures the thread pool is stopped.
   */
  virtual ~task_creator();

  // Non-copyable and movable
  task_creator(const task_creator&)            = delete;
  task_creator& operator=(const task_creator&) = delete;
  task_creator(task_creator&&)                 = delete;
  task_creator& operator=(task_creator&&)      = delete;

  /// \brief sets client context needed for task creation
  void set_client_context(::duckdb::ClientContext& client_context);

  /// \brief sets gpu pipeline hash map needed for task creation
  void set_pipeline_hashmap(sirius_pipeline_hashmap& sirius_pipeline_map);

  /// \brief clean-up query bound resources and prepare the task creator for next query
  void reset();

  /**
   * @brief Process and schedule the next task based on operator hints.
   *
   * This method queries the given node for a hint about what task to create next.
   * The hint can be another operator, a pipeline, or empty (in which case a
   * priority scan is scheduled if available).
   *
   * @param node The operator node to get the next task hint from.
   */
  void process_next_task(op::sirius_physical_operator* node);

  /**
   * @brief Start scheduling initial scan pipelines.
   *
   * This method schedules all priority scan pipelines that were identified
   * during construction.
   */
  void start();

  /**
   * @brief Stop the task creator and its thread pool.
   */
  void stop();

  /**
   * @brief Start the worker thread pool.
   *
   * Creates and starts the worker threads that process task creation requests.
   * This method is idempotent - calling it multiple times has no additional effect.
   */
  void start_thread_pool();

  /**
   * @brief Stop the worker thread pool.
   *
   * Stops all worker threads and waits for them to finish. This method is
   * idempotent - calling it multiple times has no additional effect.
   */
  void stop_thread_pool();

  /**
   * @brief Schedule a task creation info for processing.
   *
   * @param info The task creation info to schedule.
   */
  virtual void schedule(std::unique_ptr<task_creation_info> info);

  /**
   * @brief Get the next task id.
   *
   * @return uint64_t The next task id.
   */
  uint64_t get_next_task_id();

 protected:
  /**
   * @brief Worker function executed by each thread in the pool.
   *
   * Continuously pulls task creation requests from the queue and processes them
   * until the thread pool is stopped or the queue is closed.
   *
   * @param worker_id The identifier of this worker thread.
   */
  void worker_function(int worker_id);

  /**
   * @brief Called when the thread pool starts.
   *
   * Opens the task creation queue to accept requests.
   */
  void on_start();

  /**
   * @brief Called when the thread pool stops.
   *
   * Closes the task creation queue to signal workers to exit.
   */
  void on_stop();

  size_t _num_threads;
  std::atomic<bool> _running;
  std::vector<std::thread> _threads;
  std::queue<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> _priority_scans;
  std::unique_ptr<task_creation_queue> _task_creation_queue;
  sirius_pipeline_hashmap* _sirius_pipeline_map;
  ::duckdb::ClientContext* _client_context;
  sirius::pipeline::pipeline_executor& _pipeline_executor;
  sirius::op::scan::duckdb_scan_executor& _duckdb_scan_executor;
  sirius::memory::sirius_memory_reservation_manager& _mem_res_mgr;
  std::atomic<uint64_t> _task_id{0};
};

}  // namespace sirius::creator
