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

#include "op/sirius_physical_operator.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::op {

/**
 * @brief Rows a producer still holds in operator state when its pipeline ends.
 *
 * The end of a pipeline's input is not a task, so the producer cannot publish these rows itself.
 * Its consumer drains them instead, the way MERGE_AGGREGATE runs one task from operator state
 * (`input_free_operator_data`): once the producer's pipeline has finished, the consumer claims one
 * input-free task per unit of state, runs `emit` inside that task, and takes the returned batches
 * as ordinary input. Each claimed task is pinned to the GPU that holds its unit.
 *
 * The table aggregation (`table_aggregate_state`) is the one implementation: a unit is the table
 * of one GPU.
 */
class pending_state_source {
 public:
  struct totals {
    std::uint64_t bytes = 0;  ///< Estimated uncompressed bytes of the rows still held
    std::uint64_t rows  = 0;  ///< Rows still held
  };

  virtual ~pending_state_source() = default;

  /// Whether a unit is still unclaimed. Callable from any thread at any time; it may be true
  /// while the producer's pipeline is still running.
  [[nodiscard]] virtual bool has_unclaimed() const noexcept = 0;

  /// Size of the unclaimed units. Call only after the producer's pipeline has finished.
  [[nodiscard]] virtual totals unclaimed_totals() const = 0;

  /// Claims one unit as the input of one task, stamped with the GPU the task must run on; nullptr
  /// when none is left. Call only after the producer's pipeline has finished.
  [[nodiscard]] virtual std::unique_ptr<operator_data> claim() = 0;

  /**
   * @brief Turns the unit claimed as @p unit into batches. Runs inside the claimed task.
   *
   * A replay of the task after an out-of-memory error calls this again with the same @p unit and
   * returns every batch exactly once across the attempts that fail and the one that succeeds.
   *
   * @throws rmm::out_of_memory if device memory runs out; the task is then replayed
   */
  [[nodiscard]] virtual std::vector<std::shared_ptr<cucascade::data_batch>> emit(
    const operator_data& unit,
    ::cuda::stream_ref stream,
    const telemetry::batch_telemetry_info& telemetry_info) = 0;
};

}  // namespace sirius::op
