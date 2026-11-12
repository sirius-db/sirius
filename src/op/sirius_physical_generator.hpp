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

#include "op/scan/generator/generator_spec.hpp"
#include "op/sirius_physical_operator.hpp"

#include <atomic>
#include <cstdint>

namespace sirius::op {

/**
 * @brief GPU source operator behind the `sirius_generate` table function.
 *
 * Produces `spec.num_batches` batches of `spec.rows_per_batch` synthetic rows directly on the
 * GPU. Like GPU_VALUES it is an input-free source driven by the normal task creator loop, but it
 * hands out one task per batch: each task input carries its batch index as the
 * input_free_operator_data sequence number, so an OOM retry regenerates the same batch.
 */
class sirius_physical_generator : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::GENERATOR;

  sirius_physical_generator(duckdb::vector<sirius::logical_type> types,
                            scan::generator::generator_spec spec,
                            duckdb::idx_t estimated_cardinality);

  // -----------------------------
  // Source interface
  // -----------------------------
  bool is_source() const override { return true; }

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  /// input_free_operator_data(estimated_batch_bytes(spec), batch_index), or nullptr once every
  /// batch has been handed out.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /// Generates the batch named by the input's sequence number in the task's GPU memory space.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  /// Exact whole-query input basis, known at plan time.
  [[nodiscard]] std::optional<std::size_t> total_source_input_bytes() const override;

  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override;

  [[nodiscard]] const scan::generator::generator_spec& spec() const noexcept { return _spec; }

 private:
  scan::generator::generator_spec _spec;
  //! Index of the next batch to hand out; batches are claimed with fetch_add.
  std::atomic<std::uint64_t> _next_batch{0};
};

}  // namespace sirius::op
