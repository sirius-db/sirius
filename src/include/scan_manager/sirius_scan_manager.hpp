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
#include "scan_manager/split_provider.hpp"

#include <memory>
#include <thread>
#include <vector>

namespace sirius::op::scan {
class sirius_gpu_parquet_scan_operator;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

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
  /// Walks @p query 's pipelines in scan-operator order, registers each GPU
  /// parquet scan source (binding it a fresh split_connector and taking the
  /// parked split_provider), and launches a driver thread that runs the
  /// providers SEQUENTIALLY: provider[0] starts, when its future completes
  /// provider[1] starts, and so on. Consumers (the gpu scan operators) block
  /// in split_connector::get_next_split until splits arrive or the connector
  /// is closed, so no separate wake-up channel is needed.
  void prepare_for_query(const sirius::planner::query& query);

  /// \brief Register a GPU parquet scan operator with this scan manager.
  ///
  /// Installs a fresh split_connector on @p op, takes ownership of the parked
  /// split_provider, and queues the (op, provider) pair for sequential
  /// execution by the driver thread. Idempotent per-query.
  void register_scan_operator(op::scan::sirius_gpu_parquet_scan_operator* op);

  /// \brief Clear all registrations from the previous query and join the
  ///        driver thread if it is still running.
  void reset();

  /// \brief Start the worker thread pool. Idempotent.
  void start();

  /// \brief Stop the worker thread pool and the driver. Idempotent.
  void stop();

 private:
  struct registration {
    op::scan::sirius_gpu_parquet_scan_operator* op;
    std::unique_ptr<split_provider> provider;
  };

  /// \brief Run providers sequentially: start each, wait on its future, advance.
  void run_driver_loop();

  exec::thread_pool_config _config;
  std::unique_ptr<exec::thread_pool> _thread_pool;
  std::vector<registration> _registrations;
  std::thread _driver_thread;
};

}  // namespace sirius::scan_manager
