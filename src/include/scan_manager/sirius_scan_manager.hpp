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

#include "exec/config.hpp"
#include "exec/thread_pool.hpp"

#include <memory>
#include <vector>

namespace sirius::creator {
class task_creator;
}  // namespace sirius::creator

namespace sirius::op::scan {
class sirius_gpu_parquet_scan_operator;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

struct scan_op_state;

/**
 * @brief Manages scan-side preparation for a query.
 *
 * The scan manager owns a configurable-size thread pool and is given a chance
 * to set up per-scan state before a query runs (via prepare_for_query).
 */
class sirius_scan_manager {
 public:
  /**
   * @brief Construct a new source manager.
   *
   * @param config Configuration for the thread pool (thread count, name prefix, CPU affinity).
   */
  explicit sirius_scan_manager(exec::thread_pool_config config);

  ~sirius_scan_manager();

  // Non-copyable and non-movable
  sirius_scan_manager(const sirius_scan_manager&)            = delete;
  sirius_scan_manager& operator=(const sirius_scan_manager&) = delete;
  sirius_scan_manager(sirius_scan_manager&&)                 = delete;
  sirius_scan_manager& operator=(sirius_scan_manager&&)      = delete;

  /// \brief Prepare per-scan state for the given query.
  ///
  /// Walks @p query 's pipelines, locates GPU parquet scan sources, and registers
  /// each one (binding a split_connector and recording per-operator state).
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief Register a GPU parquet scan operator with this scan manager.
  ///
  /// Creates a fresh split_connector, installs it on @p op, takes ownership of
  /// the metadata scan operator parked on @p op, and dispatches metadata-scan
  /// tasks to the scan-manager thread pool. Each task runs the metadata scan
  /// against cudf::get_default_stream() and pushes the resulting parquet_scan_data
  /// splits into the connector. When the last task completes, the connector is
  /// closed.
  void register_scan_operator(op::scan::sirius_gpu_parquet_scan_operator* op);

  /// \brief Clear all registrations from the previous query.
  void reset();

  /// \brief Start the worker thread pool. Idempotent.
  void start();

  /// \brief Stop the worker thread pool. Idempotent.
  void stop();

 private:
  exec::thread_pool_config _config;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  creator::task_creator* _task_creator{nullptr};
  std::vector<op::scan::sirius_gpu_parquet_scan_operator*> _registered_scan_operators;
  std::vector<std::shared_ptr<scan_op_state>> _scan_op_states;
};

}  // namespace sirius::scan_manager
