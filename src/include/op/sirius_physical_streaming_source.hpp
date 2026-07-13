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

#include "exec/exchange_channel.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/data/data_repository.hpp>

#include <memory>
#include <optional>

namespace sirius::op {

/// Source operator that pulls data_batch handles from an exchange_channel, resolves each
/// handle via the input repository, and publishes the batch into the pipeline.
class sirius_physical_streaming_source : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SOURCE;

  /// Throws invalid_input_exception when input_channel or input_repository is null.
  sirius_physical_streaming_source(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<exec::exchange_channel> input_channel,
    std::shared_ptr<cucascade::shared_data_repository> input_repository);

  /// Wires the channel's on-close callback to the pipeline so an empty or late-closed
  /// stream still finishes even when no task is in flight to re-evaluate completion.
  /// The callback captures a weak reference to the pipeline (never `this`), so a close()
  /// racing with operator destruction cannot dereference a destroyed operator.
  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline) override;

  // -----------------------------------------------------------------------
  // Source interface
  // -----------------------------------------------------------------------

  bool is_source() const override { return true; }

  /// READY{this} when channel non-empty (open or closed); WAITING{nullptr} when open+empty;
  /// nullopt when closed && drained (EOS).
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// True only when the channel is closed AND empty (EOS). Drives both the task-creation
  /// guard and the port-less source pipeline-finish predicate.
  [[nodiscard]] bool all_ports_empty() override;

  /// Non-blocking: try_pop a handle, resolve via repo, wrap in pipelineable_operator_data.
  /// Returns nullptr when the channel is empty. Throws on a handle missing from the repo.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------------------------------------------------
  // Execution
  // -----------------------------------------------------------------------

  /// Pass-through: returns the input batches unchanged.
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override;

  /// Pass-through allocates nothing new; return input bytes so the reservation system
  /// does not over-reserve with the default 2× heuristic.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 private:
  std::shared_ptr<exec::exchange_channel> _input_channel;
  std::shared_ptr<cucascade::shared_data_repository> _input_repository;
};

}  // namespace sirius::op
