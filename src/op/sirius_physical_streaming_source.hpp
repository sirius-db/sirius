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

#include "exec/batch_stream.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/data/data_repository.hpp>

#include <exception>
#include <memory>
#include <optional>
#include <set>

namespace sirius::op {

/// Plan leaf over a batch_stream. Repository IS the queue; sender-set EOS; task-protocol glue.
class sirius_physical_streaming_source : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::STREAMING_SOURCE;

  /// @param input_repository The queue this source drains. Must not be null.
  /// @param expected_senders Senders that must `close_input` before EOS.
  /// @throws sirius::invalid_input_exception when `input_repository` is null.
  sirius_physical_streaming_source(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<cucascade::shared_data_repository> input_repository,
    std::set<exec::sender_id_t> expected_senders);

  /// Wire EOS → update_pipeline_status(false); on_data → schedule(head) (self-nomination).
  /// Without on_data, a WAITING source stays dropped until a task completes — which never happens.
  void set_pipeline(duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline) override;

  // -----------------------------------------------------------------------
  // Producer side — session / wrapper, any thread
  // -----------------------------------------------------------------------

  /// @return false if terminal (S1). Ignoring the return silently drops the batch.
  [[nodiscard]] bool push(std::shared_ptr<cucascade::data_batch> batch);

  /// Sender-set EOS: idempotent per sender.
  /// @throws sirius::invalid_input_exception when `sender` is not expected.
  void close_input(exec::sender_id_t sender);

  /// Poison the input stream (S2 / P1–P4); rethrown from get_next_task_input_data(), never a
  /// clean EOS.
  /// @throws sirius::invalid_input_exception when `error` is null.
  void fail_input(std::exception_ptr error);

  [[nodiscard]] exec::batch_stream& stream() { return *_input; }

  // -----------------------------------------------------------------------
  // Source interface
  // -----------------------------------------------------------------------

  bool is_source() const override { return true; }

  /// READY on HAS_DATA (incl. pending error / P4); WAITING_FOR_INPUT_DATA{nullptr} while
  /// open+empty; nullopt only at clean EOS.
  [[nodiscard]] std::optional<task_creation_hint> get_next_task_hint() override;

  /// Clean EOS only (S3). Drives task-creation and port-less pipeline-finish.
  [[nodiscard]] bool all_ports_empty() override;

  /// Pop one batch. @throws pending producer error before anything queued (S4).
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------------------------------------------------
  // Execution
  // -----------------------------------------------------------------------

  /// Pass-through: batches are already materialized.
  std::unique_ptr<operator_data> execute(const operator_data& input,
                                         rmm::cuda_stream_view stream) override;

  /// Pass-through: return input bytes so the default 2× heuristic does not over-reserve.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

 private:
  /// Shared so producer threads co-own the stream past this operator.
  std::shared_ptr<exec::batch_stream> _input;
};

}  // namespace sirius::op
