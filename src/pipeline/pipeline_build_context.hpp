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

#include "expression_evaluator/like_multiliteral.hpp"
#include "sirius_config.hpp"

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
  //! @param operator_params Immutable per-query operator-policy snapshot. A null value installs a
  //!        fail-closed default snapshot.
  //! @param like_cache Query-owned cache shared by all copied contexts, pipelines, and task-local
  //!        evaluators. A null value creates a new cache.
  explicit pipeline_build_context(
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
    bool preserve_insertion_order                                     = true,
    int num_gpus                                                      = 1,
    std::shared_ptr<const sirius::operator_params> operator_params    = nullptr,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache = nullptr)
    : _telemetry_context(std::move(telemetry_context)),
      _preserve_insertion_order(preserve_insertion_order),
      _num_gpus(num_gpus),
      _operator_params(operator_params ? std::move(operator_params)
                                       : std::make_shared<const sirius::operator_params>()),
      _like_cache(like_cache ? std::move(like_cache)
                             : std::make_shared<sirius::like_multiliteral_cache>())
  {
  }

  //! Construct an engine-backed context from the configured GPU set. The GPU count is derived
  //! from the same ids used for execution routing, so planning cannot accidentally use the raw
  //! hardware count when the Sirius config selected a subset of visible GPUs.
  //!
  //! @param telemetry_context SiriusContext-wide telemetry context
  //! @param preserve_insertion_order Whether query results preserve insertion order
  //! @param active_gpu_ids GPU device ids used for execution routing
  //! @param operator_params Immutable per-query operator-policy snapshot. A null value installs a
  //!        fail-closed default snapshot.
  //! @param like_cache Query-owned cache shared by all copied contexts, pipelines, and task-local
  //!        evaluators. A null value creates a new cache.
  pipeline_build_context(
    std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
    bool preserve_insertion_order,
    std::vector<int> active_gpu_ids,
    std::shared_ptr<const sirius::operator_params> operator_params    = nullptr,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache = nullptr)
    : _telemetry_context(std::move(telemetry_context)),
      _preserve_insertion_order(preserve_insertion_order),
      _active_gpu_ids(std::move(active_gpu_ids)),
      _operator_params(operator_params ? std::move(operator_params)
                                       : std::make_shared<const sirius::operator_params>()),
      _like_cache(like_cache ? std::move(like_cache)
                             : std::make_shared<sirius::like_multiliteral_cache>())
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

  [[nodiscard]] const sirius::operator_params& get_operator_params() const noexcept
  {
    return *_operator_params;
  }

  [[nodiscard]] const std::shared_ptr<const sirius::like_multiliteral_cache>&
  get_like_multiliteral_cache() const noexcept
  {
    return _like_cache;
  }

 private:
  std::shared_ptr<const telemetry::telemetry_context> _telemetry_context;
  bool _preserve_insertion_order = true;
  int _num_gpus                  = 1;
  std::vector<int> _active_gpu_ids;
  std::shared_ptr<const sirius::operator_params> _operator_params;
  std::shared_ptr<const sirius::like_multiliteral_cache> _like_cache;
};

}  // namespace pipeline
}  // namespace sirius
