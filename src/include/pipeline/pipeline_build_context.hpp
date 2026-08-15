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

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius {

namespace telemetry {
class telemetry_context;
}  // namespace telemetry

namespace pipeline {

//! Context for plan-time pipeline construction.
//! Replaces the sirius_engine& dependency so that pipelines can be built
//! without an engine instance (e.g. at optimizer/bind time).
class pipeline_build_context {
 public:
  //! @param telemetry_context SiriusContext-wide telemetry context; may be null
  //!        when pipelines are built without an engine (tests, optimizer/bind).
  //!        Carried into each sirius_pipeline so operators can build data_batch
  //!        probes.
  //! @param preserve_insertion_order Whether query results must preserve
  //!        insertion order (from DuckDB's PreserveInsertionOrderSetting).
  //! @param num_gpus Number of GPUs available for partition-floor heuristics when constructing a
  //!        context without an engine (primarily unit tests).
  //!        Enables sirius_pipeline_converter::configure_partition_min_partitions
  //!        to ensure big partition-consuming operators (hash_join,
  //!        merge_group_by) get at least num_gpus partitions to spread work.
  explicit pipeline_build_context(
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
    bool preserve_insertion_order = true,
    int num_gpus                  = 1)
    : _telemetry_context(std::move(telemetry_context)),
      _preserve_insertion_order(preserve_insertion_order),
      _num_gpus(num_gpus)
  {
  }

  //! Construct an engine-backed context from the configured GPU set. The GPU count is derived
  //! from the same ids used for execution routing, so planning cannot accidentally use the raw
  //! hardware count when the Sirius config selected a subset of visible GPUs.
  pipeline_build_context(std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
                         bool preserve_insertion_order,
                         std::vector<int> active_gpu_ids)
    : _telemetry_context(std::move(telemetry_context)),
      _preserve_insertion_order(preserve_insertion_order),
      _active_gpu_ids(std::move(active_gpu_ids))
  {
    if (_active_gpu_ids.empty()) {
      throw std::invalid_argument(
        "pipeline_build_context requires at least one configured GPU device id");
    }
    _num_gpus = static_cast<int>(_active_gpu_ids.size());
  }

  [[nodiscard]] bool preserve_insertion_order() const { return _preserve_insertion_order; }

  [[nodiscard]] int num_gpus() const { return _num_gpus; }

  //! Sorted, deduped device ids of the GPUs the query runs on — the same list task_creator routes
  //! partitions across. Used by broadcast join partitioning to map a probe batch's residence GPU
  //! back to its partition slot. Empty only when built through the engine-free test constructor.
  [[nodiscard]] const std::vector<int>& active_gpu_ids() const { return _active_gpu_ids; }

  [[nodiscard]] const std::shared_ptr<const telemetry::telemetry_context>& telemetry_context() const
  {
    return _telemetry_context;
  }

  //! Per-query OOM-retry cap for this query's GPU pipeline tasks. Stamped by the engine
  //! from the query's admission-time operator_params snapshot (register E1) before pipeline
  //! construction; engine-free construction paths (tests, optimizer/bind) keep the process
  //! default. Copied into every sirius_pipeline via this context, so the executor's retry
  //! decision reads a per-query constant, never a live struct mid-execution.
  void set_reservation_max_retries(uint32_t max_retries) noexcept
  {
    _reservation_max_retries = max_retries;
  }

  [[nodiscard]] uint32_t reservation_max_retries() const noexcept
  {
    return _reservation_max_retries;
  }

 private:
  std::shared_ptr<const telemetry::telemetry_context> _telemetry_context;
  bool _preserve_insertion_order = true;
  int _num_gpus                  = 1;
  std::vector<int> _active_gpu_ids;
  uint32_t _reservation_max_retries = exec::default_gpu_reservation_max_retries;
};

}  // namespace pipeline
}  // namespace sirius
