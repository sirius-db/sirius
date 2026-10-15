/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/aggregate/probe_accumulate_impl.hpp"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::op {

/**
 * @brief State shared between a probe-accumulate hash join and the emit operator above it.
 *
 * The join folds every probe batch into `state` and hands back empty batches. Once its pipeline
 * has finished, `sirius_physical_probe_accumulate_emit` reads the finished accumulators and the
 * build key column from here. The planner creates one context per fused aggregate and gives it to
 * both operators; it is destroyed with the plan.
 *
 * Single partition only: one build table, one set of accumulators, one device.
 */
struct probe_accumulate_context {
  explicit probe_accumulate_context(std::vector<accumulate_spec> specs) : state(std::move(specs)) {}

  probe_accumulate_state state;

  /// Written once by the join when its build table is built; read by the emit operator after the
  /// join's pipeline has finished. `mutex` orders the two.
  std::mutex mutex;
  /// Key column of the build batch, which is also the group key of the fused aggregate.
  cudf::size_type build_key_column = 0;
  /// The build batch, kept alive so emit can read its key column.
  std::optional<::cucascade::read_only_data_batch> build_batch;
  /// Memory space the accumulators live in; the output batch is allocated there.
  ::cucascade::memory::memory_space* space = nullptr;
  /// The emitted result. `probe_accumulate_state::emit` consumes the accumulators, so the emit
  /// task caches its table here: a rescheduled task (e.g. after an out-of-memory elsewhere in the
  /// pipeline) copies it instead of calling emit again on buffers that were already moved out.
  std::unique_ptr<cudf::table> emitted;
};

}  // namespace sirius::op
