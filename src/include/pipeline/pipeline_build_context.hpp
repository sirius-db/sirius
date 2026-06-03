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

namespace sirius::telemetry {
class telemetry_context;
}  // namespace sirius::telemetry

namespace sirius::pipeline {

//! Lightweight context for plan-time pipeline construction.
//! Replaces the sirius_engine& dependency so that pipelines can be built
//! without an engine instance (e.g. at optimizer/bind time).
struct pipeline_build_context {
  //! Whether query results must preserve insertion order
  //! (from DuckDB's PreserveInsertionOrderSetting)
  bool preserve_insertion_order = true;

  //! Number of GPUs available for partition-floor heuristics. Defaults to 1
  //! (single-GPU). Set from sirius_engine's hardware topology at convert time;
  //! enables sirius_pipeline_converter::configure_partition_min_partitions to
  //! ensure big partition-consuming operators (hash_join, merge_group_by) get
  //! at least num_gpus partitions to spread work across devices.
  int num_gpus = 1;

  //! Query telemetry context carried by every pipeline built from this context.
  const telemetry::telemetry_context* telemetry_context = nullptr;
};

}  // namespace sirius::pipeline
